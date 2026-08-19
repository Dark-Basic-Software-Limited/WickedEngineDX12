#include "wiProfiler.h"
#include "wiGraphicsDevice.h"
#include "wiFont.h"
#include "wiImage.h"
#include "wiTimer.h"
#include "wiTextureHelper.h"
#include "wiHelper.h"
#include "wiUnorderedMap.h"
#include "wiBacklog.h"
#include "wiRenderer.h"
#include "wiEventHandler.h"

#if __has_include("Superluminal/PerformanceAPI_capi.h")
#include "Superluminal/PerformanceAPI_capi.h"
#include "Superluminal/PerformanceAPI_loader.h"
#endif // superluminal

#include <string>
#include <stack>
#include <mutex>
#include <atomic>
#include <sstream>
#include <vector>
#include <algorithm>

using namespace wi::graphics;

// GGMAX wall-gap tracer: creation counters defined in wiGraphicsDevice_DX12.cpp (global namespace)
extern std::atomic<unsigned long long> gg_dbg_pso_compiles, gg_dbg_pso_compile_us, gg_dbg_tex_creates;
extern std::atomic<unsigned long long> gg_dbg_copywait_us, gg_dbg_copywait_events; // GGMAX 2.61: blocking copy-queue waits
// GGMAX wall-gap tracer: pump aggregates (incremented by the game's main.cpp message loop)
std::atomic<unsigned long long> gg_dbg_pump_dispatches{ 0 };
std::atomic<unsigned long long> gg_dbg_pump_us{ 0 };

namespace wi::profiler
{
	bool ENABLED = false;
	bool ENABLED_REQUEST = false;
	bool initialized = false;
	std::mutex lock;
	range_id cpu_frame;
	range_id gpu_frame;
	GPUQueryHeap queryHeap;
	GPUBuffer queryResultBuffer[GraphicsDevice::GetBufferCount()];
	std::atomic<uint32_t> nextQuery{ 0 };
	uint32_t queryheap_idx = 0;
	bool drawn_this_frame = false;
	wi::Color background_color = wi::Color(20, 20, 20, 230);
	wi::Color text_color = wi::Color::White();

#if PERFORMANCEAPI_ENABLED
	PerformanceAPI_ModuleHandle superluminal_handle = {};
	PerformanceAPI_Functions superluminal_functions = {};
#endif // PERFORMANCEAPI_ENABLED

	struct Range
	{
		bool in_use = false;
		std::string name;
		float times[20] = {};
		int avg_counter = 0;
		float time = 0;
		CommandList cmd;

		wi::Timer cpuTimer;

		int gpuBegin[arraysize(queryResultBuffer)];
		int gpuEnd[arraysize(queryResultBuffer)];

		bool IsCPURange() const { return !cmd.IsValid(); }
	};
	wi::unordered_map<size_t, Range> ranges;

	// GGMAX 2.91: GPU Busy / GPU Idle accounting.
	// "GPU Frame" is a wall-clock SPAN — its begin query rides the frame's FIRST command list
	// and its end query the LAST — so it structurally contains three things no child range can
	// account for: passes with no range, driver work at RenderPassBegin/End (CLEAR / STORE /
	// MSAA resolve / barriers), and any interval where the GPU simply had nothing to run.
	// Reading "GPU Frame" as "my GPU workload" is therefore wrong; on a paced frame it tends to
	// the frame period.
	// ⚠ Busy is deliberately the UNION of the child intervals, NOT their sum. Ranges NEST —
	// "Occlusion Culling" wraps "Occlusion Culling Render" — so a sum double-counts, which is
	// exactly the mistake this counter exists to stop anyone making by hand. A union is also
	// correct if async work ever overlaps again (see gg_single_queue).
	float gg_gpu_busy_time = 0;
	float gg_gpu_idle_time = 0;
	float gg_gpu_busy_times[20] = {};
	float gg_gpu_idle_times[20] = {};
	int   gg_gpu_acc_counter = 0;
	// scratch, reused each frame so the resolve loop allocates nothing
	wi::vector<std::pair<uint64_t, uint64_t>> gg_gpu_intervals;

	void gg_ClearTextDataCaches(); // GGMAX 1.67: defined next to GetTextData below

	// GGMAX wall-gap tracer (see wiProfiler.h). Main thread only; independent of ENABLED.
	struct GGTraceMark { const char* name; unsigned long long us; char buf[24]; };
	static GGTraceMark gg_trace_marks[128];
	static int gg_trace_mark_count = 0;
	static unsigned long long gg_trace_prev_begin_us = 0;
	static unsigned long long gg_trace_pso_snap = 0, gg_trace_pso_us_snap = 0, gg_trace_tex_snap = 0;
	static unsigned long long gg_trace_pump_snap = 0, gg_trace_pump_us_snap = 0;
	static unsigned long long gg_trace_copywait_snap = 0, gg_trace_copywait_us_snap = 0; // GGMAX 2.61
	static int gg_trace_gaps_written = 0;
	std::atomic<unsigned long long> gg_trace_gap_count{ 0 };   // read by harness GAPS: line
	std::atomic<unsigned long long> gg_trace_gap_last_ms{ 0 };
	unsigned long long gg_trace_threshold_us = 100000;         // 100ms

	// GGMAX 1.82: HITCH HISTOGRAM.
	//
	// The gap tracer above only fires above 100 ms — loading-scale stalls. A first-use PSO compile
	// is 1-30 ms, which is invisible to it AND invisible to an FPS average (57 compiles smeared
	// over 600 frames move the mean by nothing) yet is exactly what a player feels as a stutter.
	// So: count EVERY frame into buckets, keep the worst one, and provide an explicit reset so a
	// window can be scoped to "the 20 s after a level load" instead of "since launch".
	//
	// Deliberately lives on the gap tracer's path because that runs at every BeginFrame regardless
	// of whether the profiler is ENABLED — turning the profiler on to measure a hitch would change
	// the thing being measured.
	static const unsigned long long gg_hitch_thresholds_us[GG_HITCH_BUCKETS] = { 16700, 25000, 33000, 50000, 100000 };
	static std::atomic<unsigned long long> gg_hitch_frames{ 0 };
	static std::atomic<unsigned long long> gg_hitch_over[GG_HITCH_BUCKETS];
	static std::atomic<unsigned long long> gg_hitch_max_us{ 0 };
	static std::atomic<unsigned long long> gg_hitch_total_us{ 0 };
	static unsigned long long gg_hitch_pso_snap = 0, gg_hitch_pso_us_snap = 0;

	void gg_hitch_reset(void)
	{
		gg_hitch_frames.store(0, std::memory_order_relaxed);
		for (int i = 0; i < GG_HITCH_BUCKETS; ++i) gg_hitch_over[i].store(0, std::memory_order_relaxed);
		gg_hitch_max_us.store(0, std::memory_order_relaxed);
		gg_hitch_total_us.store(0, std::memory_order_relaxed);
		gg_hitch_pso_snap = gg_dbg_pso_compiles.load();
		gg_hitch_pso_us_snap = gg_dbg_pso_compile_us.load();
	}
	void gg_hitch_get(unsigned long long* frames, unsigned long long* over, unsigned long long* max_us,
		unsigned long long* total_us, unsigned long long* pso_compiles, unsigned long long* pso_compile_us)
	{
		if (frames) *frames = gg_hitch_frames.load(std::memory_order_relaxed);
		if (over) for (int i = 0; i < GG_HITCH_BUCKETS; ++i) over[i] = gg_hitch_over[i].load(std::memory_order_relaxed);
		if (max_us) *max_us = gg_hitch_max_us.load(std::memory_order_relaxed);
		if (total_us) *total_us = gg_hitch_total_us.load(std::memory_order_relaxed);
		// Reported as deltas since the reset, so the window means what it says.
		if (pso_compiles) *pso_compiles = gg_dbg_pso_compiles.load() - gg_hitch_pso_snap;
		if (pso_compile_us) *pso_compile_us = gg_dbg_pso_compile_us.load() - gg_hitch_pso_us_snap;
	}
	static void gg_hitch_accumulate(unsigned long long dt_us)
	{
		gg_hitch_frames.fetch_add(1, std::memory_order_relaxed);
		gg_hitch_total_us.fetch_add(dt_us, std::memory_order_relaxed);
		for (int i = 0; i < GG_HITCH_BUCKETS; ++i)
		{
			if (dt_us > gg_hitch_thresholds_us[i]) gg_hitch_over[i].fetch_add(1, std::memory_order_relaxed);
		}
		unsigned long long prev = gg_hitch_max_us.load(std::memory_order_relaxed);
		while (dt_us > prev && !gg_hitch_max_us.compare_exchange_weak(prev, dt_us, std::memory_order_relaxed)) {}
	}
	static unsigned long long gg_trace_qpc_us(void)
	{
		LARGE_INTEGER f, c;
		QueryPerformanceFrequency(&f);
		QueryPerformanceCounter(&c);
		return (unsigned long long)((c.QuadPart * 1000000.0) / (double)f.QuadPart);
	}
	void gg_trace_mark(const char* name)
	{
		if (gg_trace_mark_count < (int)arraysize(gg_trace_marks))
		{
			gg_trace_marks[gg_trace_mark_count].name = name;
			gg_trace_marks[gg_trace_mark_count].us = gg_trace_qpc_us();
			gg_trace_mark_count++;
		}
	}
	void gg_trace_mark_id(const char* prefix, unsigned int id)
	{
		if (gg_trace_mark_count < (int)arraysize(gg_trace_marks))
		{
			GGTraceMark& m = gg_trace_marks[gg_trace_mark_count];
			sprintf_s(m.buf, sizeof(m.buf), "%s_%04X", prefix, id);
			m.name = m.buf;
			m.us = gg_trace_qpc_us();
			gg_trace_mark_count++;
		}
	}
	unsigned long long gg_trace_now_us(void)
	{
		return gg_trace_qpc_us();
	}
	// GGMAX 2.71: false = keep every counter and the hitch histogram live (they feed
	// GET_PERF_DATA), but write no gap_trace.txt — standalones run producelogfiles=0
	// and must not drop trace files where players see them.
	static bool gg_trace_file_enabled = true;
	void gg_trace_file_enable(bool enable)
	{
		gg_trace_file_enabled = enable;
	}
	static void gg_trace_frame_boundary(void)
	{
		const unsigned long long now = gg_trace_qpc_us();
		if (gg_trace_prev_begin_us != 0)
		{
			const unsigned long long dt = now - gg_trace_prev_begin_us;
			gg_hitch_accumulate(dt); // GGMAX 1.82 — every frame, not just the >100ms ones
			if (dt > gg_trace_threshold_us)
			{
				gg_trace_gap_count.fetch_add(1, std::memory_order_relaxed);
				gg_trace_gap_last_ms = dt / 1000;
				if (gg_trace_file_enabled && gg_trace_gaps_written < 300) // file gate (2.71) + size backstop
				{
					gg_trace_gaps_written++;
					FILE* f = nullptr;
					fopen_s(&f, "gap_trace.txt", "a");
					if (f)
					{
						fprintf(f, "GAP #%llu  %.1f ms  (boundary t=%llu us)\n",
							gg_trace_gap_count.load(), dt / 1000.0, now);
						unsigned long long prev = gg_trace_prev_begin_us;
						for (int i = 0; i < gg_trace_mark_count; ++i)
						{
							fprintf(f, "  %-26s +%9.2f ms\n", gg_trace_marks[i].name, (gg_trace_marks[i].us - prev) / 1000.0);
							prev = gg_trace_marks[i].us;
						}
						fprintf(f, "  %-26s +%9.2f ms\n", "[outside-Run: pump/etc]", (now - prev) / 1000.0);
						fprintf(f, "  psoCompiles=+%llu psoCompileMs=+%.1f texCreates=+%llu copyWaits=+%llu copyWaitMs=+%.1f pumpDispatches=+%llu pumpMs=+%.1f\n\n",
							gg_dbg_pso_compiles.load() - gg_trace_pso_snap,
							(gg_dbg_pso_compile_us.load() - gg_trace_pso_us_snap) / 1000.0,
							gg_dbg_tex_creates.load() - gg_trace_tex_snap,
							gg_dbg_copywait_events.load() - gg_trace_copywait_snap,
							(gg_dbg_copywait_us.load() - gg_trace_copywait_us_snap) / 1000.0,
							gg_dbg_pump_dispatches.load() - gg_trace_pump_snap,
							(gg_dbg_pump_us.load() - gg_trace_pump_us_snap) / 1000.0);
						fclose(f);
					}
				}
			}
		}
		gg_trace_prev_begin_us = now;
		gg_trace_mark_count = 0;
		gg_trace_pso_snap = gg_dbg_pso_compiles.load();
		gg_trace_pso_us_snap = gg_dbg_pso_compile_us.load();
		gg_trace_tex_snap = gg_dbg_tex_creates.load();
		gg_trace_copywait_snap = gg_dbg_copywait_events.load(); // GGMAX 2.61
		gg_trace_copywait_us_snap = gg_dbg_copywait_us.load();
		gg_trace_pump_snap = gg_dbg_pump_dispatches.load();
		gg_trace_pump_us_snap = gg_dbg_pump_us.load();
	}

	void BeginFrame()
	{
		gg_trace_frame_boundary(); // GGMAX wall-gap tracer — must run even when profiler disabled

		if (ENABLED_REQUEST != ENABLED)
		{
			ranges.clear();
			gg_ClearTextDataCaches(); // GGMAX 1.67: forget names from the previous profiling session
			ENABLED = ENABLED_REQUEST;
		}

		if (!ENABLED)
			return;

		if (!initialized)
		{
			initialized = true;

			ranges.reserve(100);

			GraphicsDevice* device = wi::graphics::GetDevice();

			GPUQueryHeapDesc desc;
			desc.type = GpuQueryType::TIMESTAMP;
			desc.query_count = 1024;
			bool success = device->CreateQueryHeap(&desc, &queryHeap);
			assert(success);

			GPUBufferDesc bd;
			bd.usage = Usage::READBACK;
			bd.size = desc.query_count * sizeof(uint64_t);

			for (int i = 0; i < arraysize(queryResultBuffer); ++i)
			{
				success = device->CreateBuffer(&bd, nullptr, &queryResultBuffer[i]);
				assert(success);
			}

#if PERFORMANCEAPI_ENABLED
			superluminal_handle = PerformanceAPI_LoadFrom(L"PerformanceAPI.dll", &superluminal_functions);
			if (superluminal_handle)
			{
				wi::backlog::post("[wi::profiler] Superluminal Performance API loaded");
			}
#endif // PERFORMANCEAPI_ENABLED
		}

		cpu_frame = BeginRangeCPU("CPU Frame");

		GraphicsDevice* device = wi::graphics::GetDevice();
		CommandList cmd = device->BeginCommandList();
		queryheap_idx = device->GetBufferIndex();

		// Read results of previous timings:
		// This should be done before we begin reallocating new queries for current buffer index
		const uint64_t* queryResults = (const uint64_t*)queryResultBuffer[queryheap_idx].mapped_data;
		double gpu_frequency = (double)device->GetTimestampFrequency() / 1000.0;
		gg_gpu_intervals.clear();       // GGMAX 2.91
		uint64_t gg_frame_span_ticks = 0;
		for (auto& x : ranges)
		{
			auto& range = x.second;
			if (!range.in_use)
				continue;

			if (!range.IsCPURange())
			{
				const int begin_idx = range.gpuBegin[queryheap_idx];
				const int end_idx = range.gpuEnd[queryheap_idx];
				if (queryResults != nullptr && begin_idx >= 0 && end_idx >= 0)
				{
					const uint64_t begin_result = queryResults[begin_idx];
					const uint64_t end_result = queryResults[end_idx];
					range.time = (float)abs((double)(end_result - begin_result) / gpu_frequency);

					// GGMAX 2.91: collect raw tick intervals for the Busy union below.
					const uint64_t lo = std::min(begin_result, end_result);
					const uint64_t hi = std::max(begin_result, end_result);
					if (x.first == gpu_frame)
						gg_frame_span_ticks = hi - lo;   // the span itself, not a child
					else
						gg_gpu_intervals.push_back(std::make_pair(lo, hi));
				}
				range.gpuBegin[queryheap_idx] = -1;
				range.gpuEnd[queryheap_idx] = -1;
			}
			range.times[range.avg_counter++ % arraysize(range.times)] = range.time;

			if (range.avg_counter > arraysize(range.times))
			{
				float avg_time = 0;
				for (int i = 0; i < arraysize(range.times); ++i)
				{
					avg_time += range.times[i];
				}
				range.time = avg_time / arraysize(range.times);
			}

			range.in_use = false;
		}

		// GGMAX 2.91: fold the collected child intervals into Busy (union) and Idle.
		if (gg_frame_span_ticks > 0)
		{
			std::sort(gg_gpu_intervals.begin(), gg_gpu_intervals.end());
			uint64_t busy_ticks = 0;
			uint64_t cur_lo = 0, cur_hi = 0;
			bool have = false;
			for (auto& iv : gg_gpu_intervals)
			{
				if (!have) { cur_lo = iv.first; cur_hi = iv.second; have = true; continue; }
				if (iv.first <= cur_hi)                     // overlaps or nests -> extend
					cur_hi = std::max(cur_hi, iv.second);
				else { busy_ticks += cur_hi - cur_lo; cur_lo = iv.first; cur_hi = iv.second; }
			}
			if (have) busy_ticks += cur_hi - cur_lo;
			if (busy_ticks > gg_frame_span_ticks) busy_ticks = gg_frame_span_ticks; // clamp: a
				// child on another queue can in principle sit outside the span
			const float busy_ms = (float)((double)busy_ticks / gpu_frequency);
			const float span_ms = (float)((double)gg_frame_span_ticks / gpu_frequency);
			// average over the same 20-frame window the ranges use, so these are comparable
			gg_gpu_busy_times[gg_gpu_acc_counter % arraysize(gg_gpu_busy_times)] = busy_ms;
			gg_gpu_idle_times[gg_gpu_acc_counter % arraysize(gg_gpu_idle_times)] = std::max(0.0f, span_ms - busy_ms);
			gg_gpu_acc_counter++;
			const int n = (int)std::min((size_t)gg_gpu_acc_counter, arraysize(gg_gpu_busy_times));
			float b = 0, i2 = 0;
			for (int k = 0; k < n; ++k) { b += gg_gpu_busy_times[k]; i2 += gg_gpu_idle_times[k]; }
			gg_gpu_busy_time = b / (float)n;
			gg_gpu_idle_time = i2 / (float)n;
		}

		device->QueryReset(
			&queryHeap,
			0,
			queryHeap.desc.query_count,
			cmd
		);

		gpu_frame = BeginRangeGPU("GPU Frame", cmd);
		drawn_this_frame = false;
	}
	void EndFrame(CommandList cmd)
	{
		if (!ENABLED || !initialized)
			return;

		GraphicsDevice* device = wi::graphics::GetDevice();

		// note: read the GPU Frame end range manually because it will be on a separate command list than start point:
		auto& gpu_range = ranges[gpu_frame];
		gpu_range.gpuEnd[queryheap_idx] = nextQuery.fetch_add(1);
		device->QueryEnd(&queryHeap, gpu_range.gpuEnd[queryheap_idx], cmd);

		EndRange(cpu_frame);

		device->QueryResolve(
			&queryHeap,
			0,
			nextQuery.load(),
			&queryResultBuffer[queryheap_idx],
			0ull,
			cmd
		);

		nextQuery.store(0);
	}

	range_id BeginRangeCPU(const char* name)
	{
		if (!ENABLED || !initialized)
			return 0;

#if PERFORMANCEAPI_ENABLED
		if (superluminal_handle)
		{
			superluminal_functions.BeginEvent(name, nullptr, 0xFF0000FF);
		}
#endif // PERFORMANCEAPI_ENABLED

		range_id id = wi::helper::string_hash(name);

		lock.lock();

		// If one range name is hit multiple times, differentiate between them!
		size_t differentiator = 0;
		while (ranges[id].in_use)
		{
			wi::helper::hash_combine(id, differentiator++);
		}
		ranges[id].in_use = true;
		ranges[id].name = name;
		ranges[id].cpuTimer.record();

		lock.unlock();

		return id;
	}
	range_id BeginRangeGPU(const char* name, CommandList cmd)
	{
		if (!ENABLED || !initialized)
			return 0;

		range_id id = wi::helper::string_hash(name);

		lock.lock();

		// If one range name is hit multiple times, differentiate between them!
		size_t differentiator = 0;
		while (ranges[id].in_use)
		{
			wi::helper::hash_combine(id, differentiator++);
		}
		ranges[id].in_use = true;
		ranges[id].name = name;
		ranges[id].cmd = cmd;

		GraphicsDevice* device = wi::graphics::GetDevice();
		ranges[id].gpuBegin[queryheap_idx] = nextQuery.fetch_add(1);
		device->QueryEnd(&queryHeap, ranges[id].gpuBegin[queryheap_idx], cmd);

		lock.unlock();

		return id;
	}
	void EndRange(range_id id)
	{
		if (!ENABLED || !initialized)
			return;

		lock.lock();

		auto it = ranges.find(id);
		if (it != ranges.end())
		{
			if (it->second.IsCPURange())
			{
				it->second.time = (float)it->second.cpuTimer.elapsed();

#if PERFORMANCEAPI_ENABLED
				if (superluminal_handle)
				{
					superluminal_functions.EndEvent();
				}
#endif // PERFORMANCEAPI_ENABLED
			}
			else
			{
				GraphicsDevice* device = wi::graphics::GetDevice();
				ranges[id].gpuEnd[queryheap_idx] = nextQuery.fetch_add(1);
				device->QueryEnd(&queryHeap, it->second.gpuEnd[queryheap_idx], it->second.cmd);
			}
		}
		else
		{
			assert(0);
		}

		lock.unlock();
	}


	PipelineState pso_linestrip;
	PipelineState pso_linelist;
	const uint32_t graph_vertex_count = 120;
	float cpu_graph[graph_vertex_count] = {};
	float gpu_graph[graph_vertex_count] = {};
	float cpu_memory_graph[graph_vertex_count] = {};
	float gpu_memory_graph[graph_vertex_count] = {};
	void LoadShaders()
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		PipelineStateDesc desc;
		desc.vs = wi::renderer::GetShader(wi::enums::VSTYPE_VERTEXCOLOR);
		desc.ps = wi::renderer::GetShader(wi::enums::PSTYPE_VERTEXCOLOR);
		desc.il = wi::renderer::GetInputLayout(wi::enums::ILTYPE_VERTEXCOLOR);
		desc.dss = wi::renderer::GetDepthStencilState(wi::enums::DSSTYPE_DEPTHDISABLED);
		desc.rs = wi::renderer::GetRasterizerState(wi::enums::RSTYPE_WIRE_SMOOTH);
		desc.bs = wi::renderer::GetBlendState(wi::enums::BSTYPE_TRANSPARENT);
		desc.pt = PrimitiveTopology::LINESTRIP;
		bool success = device->CreatePipelineState(&desc, &pso_linestrip);

		desc.pt = PrimitiveTopology::LINELIST;
		success = device->CreatePipelineState(&desc, &pso_linelist);
		assert(success);
	}

	struct Hits
	{
		uint32_t num_hits = 0;
		float total_time = 0;
	};
	wi::unordered_map<std::string, Hits> time_cache_cpu;
	wi::unordered_map<std::string, Hits> time_cache_gpu;
	void DrawData(
		const wi::Canvas& canvas,
		float x,
		float y,
		CommandList cmd,
		ColorSpace colorspace
	)
	{
		if (!ENABLED || !ENABLED_REQUEST || !initialized || drawn_this_frame)
			return;
		drawn_this_frame = true;

		const XMFLOAT2 graph_size = XMFLOAT2(190, 100);
		const float graph_left_offset = 45;
		const float graph_padding_y = 40;

		wi::image::SetCanvas(canvas);
		wi::font::SetCanvas(canvas);

		std::stringstream ss("");
		ss.precision(2);

		for (auto& x : ranges)
		{
			if (!x.second.in_use)
				continue;
			if (x.second.IsCPURange())
			{
				if (x.first == cpu_frame)
					continue;
				time_cache_cpu[x.second.name].num_hits++;
				time_cache_cpu[x.second.name].total_time += x.second.time;
			}
			else
			{
				if (x.first == gpu_frame)
					continue;
				time_cache_gpu[x.second.name].num_hits++;
				time_cache_gpu[x.second.name].total_time += x.second.time;
			}
		}

		// Print CPU ranges:
		ss << ranges[cpu_frame].name << ": " << std::fixed << ranges[cpu_frame].time << " ms" << std::endl;
		for (auto& x : time_cache_cpu)
		{
			if (x.second.num_hits > 1)
			{
				ss << "\t" << x.first << " (" << x.second.num_hits << "x)" << ": " << std::fixed << x.second.total_time << " ms" << std::endl;
			}
			else if(x.second.num_hits == 1)
			{
				ss << "\t" << x.first << ": " << std::fixed << x.second.total_time << " ms" << std::endl;
			}
			x.second.num_hits = 0;
			x.second.total_time = 0;
		}
		ss << std::endl;

		// Print GPU ranges:
		ss << ranges[gpu_frame].name << ": " << std::fixed << ranges[gpu_frame].time << " ms" << std::endl;
		for (auto& x : time_cache_gpu)
		{
			if (x.second.num_hits > 1)
			{
				ss << "\t" << x.first << " (" << x.second.num_hits << "x)" << ": " << std::fixed << x.second.total_time << " ms" << std::endl;
			}
			else if (x.second.num_hits == 1)
			{
				ss << "\t" << x.first << ": " << std::fixed << x.second.total_time << " ms" << std::endl;
			}
			x.second.num_hits = 0;
			x.second.total_time = 0;
		}

		wi::font::Params params = wi::font::Params(x, y + (graph_size.y + graph_padding_y) * 2, wi::font::WIFONTSIZE_DEFAULT - 6, wi::font::WIFALIGN_LEFT, wi::font::WIFALIGN_TOP, text_color);

		// Background:
		wi::image::Params fx;
		fx.pos.x = x - 10;
		fx.pos.y = y - 10;
		fx.siz.x = std::max(graph_size.x, (float)wi::font::TextWidth(ss.str(), params)) + 200;
		fx.siz.y = (float)wi::font::TextHeight(ss.str(), params) + (graph_size.y + graph_padding_y) * 2 + 20;
		fx.color = background_color;

		if (colorspace != ColorSpace::SRGB)
		{
			params.enableLinearOutputMapping(9);
			fx.enableLinearOutputMapping(9);
		}

		wi::image::Draw(nullptr, fx, cmd);
		wi::font::Draw(ss.str(), params, cmd);


		// Graph:
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			wi::graphics::GraphicsDevice::MemoryUsage gpu_memory_usage = device->GetMemoryUsage();
			wi::helper::MemoryUsage cpu_memory_usage = wi::helper::GetMemoryUsage();

			static bool shaders_loaded = false;
			if (!shaders_loaded)
			{
				shaders_loaded = true;
				static wi::eventhandler::Handle handle = wi::eventhandler::Subscribe(wi::eventhandler::EVENT_RELOAD_SHADERS, [](uint64_t userdata) { LoadShaders(); });
				LoadShaders();
			}

			float graph_max = 33;
			float graph_max_gpu_memory = 0;
			float graph_max_cpu_memory = 0;
			for (uint32_t i = graph_vertex_count - 1; i > 0; --i)
			{
				cpu_graph[i] = cpu_graph[i - 1];
				gpu_graph[i] = gpu_graph[i - 1];
				cpu_memory_graph[i] = cpu_memory_graph[i - 1];
				gpu_memory_graph[i] = gpu_memory_graph[i - 1];
				graph_max = std::max(graph_max, cpu_graph[i]);
				graph_max = std::max(graph_max, gpu_graph[i]);
				graph_max_gpu_memory = std::max(graph_max_gpu_memory, gpu_memory_graph[i]);
				graph_max_cpu_memory = std::max(graph_max_cpu_memory, cpu_memory_graph[i]);
			}
			cpu_graph[0] = ranges[cpu_frame].time;
			gpu_graph[0] = ranges[gpu_frame].time;
			cpu_memory_graph[0] = float(double(cpu_memory_usage.process_physical) / (1024.0 * 1024.0 * 1024.0)); // Gigabytes
			gpu_memory_graph[0] = float(double(gpu_memory_usage.usage) / (1024.0 * 1024.0 * 1024.0)); // Gigabytes
			graph_max = std::max(graph_max, cpu_graph[0]);
			graph_max = std::max(graph_max, gpu_graph[0]);
			graph_max_gpu_memory = std::max(graph_max_gpu_memory, gpu_memory_graph[0]);
			graph_max_cpu_memory = std::max(graph_max_cpu_memory, cpu_memory_graph[0]);
			const float graph_max_memory = std::max(graph_max_cpu_memory, graph_max_gpu_memory) * 1.1f;

			struct Vertex
			{
				XMFLOAT4 position;
				XMFLOAT4 color;
			};
			const Vertex graph_info[] = {
				// axes:
				{XMFLOAT4(0, 0, 0, 1), text_color},
				{XMFLOAT4(0, 1, 0, 1), text_color},
				{XMFLOAT4(0, 1, 0, 1), text_color},
				{XMFLOAT4(1, 1, 0, 1), text_color},

				// markers:
				{XMFLOAT4(0, 0, 0, 1), text_color}, // graph_max
				{XMFLOAT4(-10.0f / graph_size.x, 0, 0, 1), text_color}, // graph_max
				{XMFLOAT4(0, 1 - 16.6f / graph_max, 0, 1), text_color}, // 16.6f
				{XMFLOAT4(-10.0f / graph_size.x, 1 - 16.6f / graph_max, 0, 1), text_color}, // 16.6f
				{XMFLOAT4(60.0f / float(graph_vertex_count),1,0,1), text_color}, // 60 frames
				{XMFLOAT4(60.0f / float(graph_vertex_count),1 + 10.0f / graph_size.x,0,1), text_color}, // 60 frames
				{XMFLOAT4(1,1,0,1), text_color}, // 0 frames
				{XMFLOAT4(1,1 + 10.0f / graph_size.x,0,1), text_color}, // 0 frames
			};
			const Vertex graph_memory_info[] = {
				// axes:
				{XMFLOAT4(0, 0, 0, 1), text_color},
				{XMFLOAT4(0, 1, 0, 1), text_color},
				{XMFLOAT4(0, 1, 0, 1), text_color},
				{XMFLOAT4(1, 1, 0, 1), text_color},

				// markers:
				{XMFLOAT4(0, 0, 0, 1), text_color}, // graph_max_memory
				{XMFLOAT4(-10.0f / graph_size.x, 0, 0, 1), text_color}, // graph_max_memory
				{XMFLOAT4(0, 0.5f, 0, 1), text_color}, // half_memory_budget
				{XMFLOAT4(-10.0f / graph_size.x, 0.5f, 0, 1), text_color}, // half_memory_budget
				{XMFLOAT4(60.0f / float(graph_vertex_count),1,0,1), text_color}, // 60 frames
				{XMFLOAT4(60.0f / float(graph_vertex_count),1 + 10.0f / graph_size.x,0,1), text_color}, // 60 frames
				{XMFLOAT4(1,1,0,1), text_color}, // 0 frames
				{XMFLOAT4(1,1 + 10.0f / graph_size.x,0,1), text_color}, // 0 frames
			};

			GraphicsDevice::GPUAllocation allocation = device->AllocateGPU(sizeof(Vertex) * graph_vertex_count * 4 + sizeof(graph_info) + sizeof(graph_memory_info), cmd);
			if (!allocation.IsValid())
				return;

			for (uint32_t i = 0; i < graph_vertex_count; ++i)
			{
				Vertex vert;
				vert.color = XMFLOAT4(1, 0.2f, 0.2f, 1);
				vert.position = XMFLOAT4(float(graph_vertex_count - 1 - i) / float(graph_vertex_count), 1 - cpu_graph[i] / graph_max, 0, 1);
				std::memcpy((Vertex*)allocation.data + i, &vert, sizeof(vert));

				vert.color = XMFLOAT4(0.2f, 1, 0.2f, 1);
				vert.position = XMFLOAT4(float(graph_vertex_count - 1 - i) / float(graph_vertex_count), 1 - gpu_graph[i] / graph_max, 0, 1);
				std::memcpy((Vertex*)allocation.data + graph_vertex_count + i, &vert, sizeof(vert));

				vert.color = XMFLOAT4(1, 1, 0.2f, 1);
				vert.position = XMFLOAT4(float(graph_vertex_count - 1 - i) / float(graph_vertex_count), 1 - gpu_memory_graph[i] / graph_max_memory, 0, 1);
				std::memcpy((Vertex*)allocation.data + graph_vertex_count * 2 + i, &vert, sizeof(vert));

				vert.color = XMFLOAT4(1, 0.2f, 1, 1);
				vert.position = XMFLOAT4(float(graph_vertex_count - 1 - i) / float(graph_vertex_count), 1 - cpu_memory_graph[i] / graph_max_memory, 0, 1);
				std::memcpy((Vertex*)allocation.data + graph_vertex_count * 3 + i, &vert, sizeof(vert));
			}
			std::memcpy((Vertex*)allocation.data + graph_vertex_count * 4, graph_info, sizeof(graph_info));
			std::memcpy((Vertex*)allocation.data + graph_vertex_count * 4 + sizeof(graph_info) / sizeof(Vertex), graph_memory_info, sizeof(graph_memory_info));

			device->BindPipelineState(&pso_linestrip, cmd);

			MiscCB cb;
			cb.g_xColor = XMFLOAT4(1, 1, 1, 1);
			XMStoreFloat4x4(&cb.g_xTransform,
				XMMatrixScaling(graph_size.x - graph_left_offset, graph_size.y, 1) *
				XMMatrixTranslation(x + graph_left_offset, y, 0) *
				canvas.GetProjection()
			);
			device->BindDynamicConstantBuffer(cb, CB_GETBINDSLOT(MiscCB), cmd);

			const GPUBuffer* buffers[] = {
				&allocation.buffer
			};
			const uint32_t strides[] = {
				sizeof(Vertex)
			};
			const uint64_t offsets[] = {
				allocation.offset
			};
			device->BindVertexBuffers(buffers, 0, arraysize(buffers), strides, offsets, cmd);

			device->Draw(graph_vertex_count, 0, cmd);
			device->Draw(graph_vertex_count, graph_vertex_count, cmd);

			device->BindPipelineState(&pso_linelist, cmd);
			device->Draw(sizeof(graph_info) / sizeof(Vertex), graph_vertex_count * 4, cmd);

			// Memory graph:
			const float memory_graph_y = y + graph_size.y + graph_padding_y;
			XMStoreFloat4x4(&cb.g_xTransform,
				XMMatrixScaling(graph_size.x - graph_left_offset, graph_size.y, 1) *
				XMMatrixTranslation(x + graph_left_offset, memory_graph_y, 0) *
				canvas.GetProjection()
			);
			device->BindDynamicConstantBuffer(cb, CB_GETBINDSLOT(MiscCB), cmd);
			device->BindPipelineState(&pso_linestrip, cmd);
			device->Draw(graph_vertex_count, graph_vertex_count * 2, cmd);
			device->Draw(graph_vertex_count, graph_vertex_count * 3, cmd);
			device->BindPipelineState(&pso_linelist, cmd);
			device->Draw(sizeof(graph_memory_info) / sizeof(Vertex), graph_vertex_count * 4 + sizeof(graph_info) / sizeof(Vertex), cmd);

			wi::font::Params params;
			params.size = 10;
			params.v_align = wi::font::WIFALIGN_CENTER;
			params.color = text_color;
			params.position.x = x;
			params.position.y = y;
			std::stringstream ss;
			ss.precision(1);
			ss << std::fixed << graph_max << " ms";
			wi::font::Draw(ss.str(), params, cmd);
			params.position.y = y + graph_size.y - (16.6f / graph_max) * graph_size.y;
			wi::font::Draw("16.6 ms", params, cmd);

			params.position.x = x + graph_left_offset - 40;
			params.position.y = memory_graph_y;
			ss.str("");
			ss << graph_max_memory << " GB";
			wi::font::Draw(ss.str(), params, cmd);
			params.position.y = memory_graph_y + graph_size.y - 0.5f * graph_size.y;
			ss.str("");
			ss << graph_max_memory * 0.5f << " GB";
			wi::font::Draw(ss.str(), params, cmd);

			params.position.x = x + graph_size.x + 5;
			params.position.y = y + graph_size.y - cpu_graph[0] / graph_max * graph_size.y;
			params.color = wi::Color::fromFloat4(XMFLOAT4(1, 0.2f, 0.2f, 1));
			ss.str("");
			ss.clear();
			ss << "cpu: " << cpu_graph[0] << " ms";
			wi::font::Draw(ss.str(), params, cmd);

			float cpu_position = params.position.y;
			params.position.x = x + graph_size.x + 5;
			params.position.y = y + graph_size.y - gpu_graph[0] / graph_max * graph_size.y;
			if (std::abs(params.position.y - cpu_position) < params.size)
			{
				if (params.position.y < cpu_position)
				{
					params.position.y = cpu_position - params.size;
				}
				else
				{
					params.position.y = cpu_position + params.size;
				}
			}
			params.color = wi::Color::fromFloat4(XMFLOAT4(0.2f, 1, 0.2f, 1));
			ss.str("");
			ss.clear();
			ss << "gpu: " << gpu_graph[0] << " ms";
			wi::font::Draw(ss.str(), params, cmd);

			params.position.x = x + graph_left_offset + graph_size.x - graph_left_offset + 5;
			params.position.y = memory_graph_y + graph_size.y - cpu_memory_graph[0] / graph_max_memory * graph_size.y;
			params.color = wi::Color::fromFloat4(XMFLOAT4(1, 0.2f, 1, 1));
			ss.str("");
			ss.clear();
			ss << "RAM: " << cpu_memory_graph[0] << " GB";
			wi::font::Draw(ss.str(), params, cmd);

			cpu_position = params.position.y;
			params.position.y = memory_graph_y + graph_size.y - gpu_memory_graph[0] / graph_max_memory * graph_size.y;
			if (std::abs(params.position.y - cpu_position) < params.size)
			{
				if (params.position.y < cpu_position)
				{
					params.position.y = cpu_position - params.size;
				}
				else
				{
					params.position.y = cpu_position + params.size;
				}
			}
			params.color = wi::Color::fromFloat4(XMFLOAT4(1, 1, 0.2f, 1));
			ss.str("");
			ss.clear();
			ss << "VRAM: " << gpu_memory_graph[0] << " GB";
			wi::font::Draw(ss.str(), params, cmd);

			params.h_align = wi::font::WIFALIGN_CENTER;
			params.color = text_color;
			params.position.x = x + graph_left_offset + (graph_size.x - graph_left_offset) * 0.5f;
			params.position.y = y + graph_size.y + 10;
			wi::font::Draw("60 frames", params, cmd);
			params.position.x = x + graph_size.x;
			wi::font::Draw("current frame", params, cmd);

			// Memory graph:
			params.position.x = x + graph_left_offset + (graph_size.x - graph_left_offset) * 0.5f;
			params.position.y = memory_graph_y + graph_size.y + 10;
			wi::font::Draw("60 frames", params, cmd);
			params.position.x = x + graph_left_offset + graph_size.x - graph_left_offset;
			wi::font::Draw("current frame", params, cmd);
		}
	}
	void DisableDrawForThisFrame()
	{
		drawn_this_frame = true;
	}

	void SetEnabled(bool value)
	{
		// Don't enable/disable the profiler immediately, only on the next frame
		//	to avoid enabling inside a Begin/End by mistake
		ENABLED_REQUEST = value;
	}

	bool IsEnabled()
	{
		return ENABLED;
	}

	void SetBackgroundColor(wi::Color color)
	{
		background_color = color;
	}
	void SetTextColor(wi::Color color)
	{
		text_color = color;
	}

	// GGMAX 1.67: persistent print caches for GetTextData — once a range name has been
	// seen it stays in the printout (0.00 ms on frames it didn't run) and the output is
	// sorted by name. Consumers that render this text as a list (the in-game Performance
	// panel) get a fixed row count/order instead of rows appearing/disappearing as
	// conditionally-executed ranges (TerrainW - *, RP3D-rec UpdateTex, Planar
	// Reflections, ...) come and go between frames.
	wi::unordered_map<std::string, Hits> text_cache_cpu_persist;
	wi::unordered_map<std::string, Hits> text_cache_gpu_persist;
	void gg_ClearTextDataCaches()
	{
		text_cache_cpu_persist.clear();
		text_cache_gpu_persist.clear();
	}

	std::string GetTextData()
	{
		if (!ENABLED || !initialized)
			return "";

		// Same data collection logic as DrawData, but no GPU calls
		for (auto& x : ranges)
		{
			if (!x.second.in_use)
				continue;
			if (x.second.IsCPURange())
			{
				if (x.first == cpu_frame)
					continue;
				text_cache_cpu_persist[x.second.name].num_hits++;
				text_cache_cpu_persist[x.second.name].total_time += x.second.time;
			}
			else
			{
				if (x.first == gpu_frame)
					continue;
				text_cache_gpu_persist[x.second.name].num_hits++;
				text_cache_gpu_persist[x.second.name].total_time += x.second.time;
			}
		}

		std::stringstream ss("");
		ss.precision(2);

		std::vector<std::pair<const std::string*, Hits*>> sorted;
		auto print_sorted = [&](wi::unordered_map<std::string, Hits>& cache)
		{
			sorted.clear();
			sorted.reserve(cache.size());
			for (auto& x : cache)
				sorted.push_back(std::make_pair(&x.first, &x.second));
			std::sort(sorted.begin(), sorted.end(),
				[](const std::pair<const std::string*, Hits*>& a, const std::pair<const std::string*, Hits*>& b)
				{ return *a.first < *b.first; });
			for (auto& x : sorted)
			{
				if (x.second->num_hits > 1)
					ss << "\t" << *x.first << " (" << x.second->num_hits << "x): " << std::fixed << x.second->total_time << " ms" << std::endl;
				else
					ss << "\t" << *x.first << ": " << std::fixed << x.second->total_time << " ms" << std::endl;
				x.second->num_hits = 0;
				x.second->total_time = 0;
			}
		};

		ss << ranges[cpu_frame].name << ": " << std::fixed << ranges[cpu_frame].time << " ms" << std::endl;
		print_sorted(text_cache_cpu_persist);
		ss << std::endl;

		// GGMAX 2.91: GPU Frame is a wall-clock SPAN, so it never equals the sum of the rows
		// below it. Busy = union of the child intervals (union, not sum — ranges nest);
		// Idle = Frame - Busy = unranged passes + barrier/clear/resolve + genuine GPU idle.
		ss << ranges[gpu_frame].name << ": " << std::fixed << ranges[gpu_frame].time << " ms" << std::endl;
		ss << "  GPU Busy (union of rows): " << std::fixed << gg_gpu_busy_time << " ms" << std::endl;
		ss << "  GPU Idle + unranged:      " << std::fixed << gg_gpu_idle_time << " ms" << std::endl;
		print_sorted(text_cache_gpu_persist);

		return ss.str();
	}

	float GetCPUFrameTime()
	{
		if (!ENABLED || !initialized)
			return 0.0f;
		return ranges[cpu_frame].time;
	}

	float GetGPUFrameTime()
	{
		if (!ENABLED || !initialized)
			return 0.0f;
		return ranges[gpu_frame].time;
	}
}
