#pragma once
#include "wiGraphicsDevice.h"
#include "wiCanvas.h"
#include "wiColor.h"
#include <string>


// QoL macros, allows writing just ScopedXxxProfiling without needing to declare a variable manually
#define ScopedCPUProfiling(name) wi::profiler::ScopedRangeCPU WI_PROFILER_CONCAT(_wi_profiler_cpu_range,__LINE__)(name)
#define ScopedGPUProfiling(name, cmd) wi::profiler::ScopedRangeGPU WI_PROFILER_CONCAT(_wi_profiler_gpu_range,__LINE__)(name, cmd)

// same as ScopedXxxProfiling, just will automatically use the function name as name, should only be used at the beginning of a function
#define ScopedCPUProfilingF ScopedCPUProfiling(__FUNCTION__)
#define ScopedGPUProfilingF(cmd) ScopedGPUProfiling(__FUNCTION__, cmd)

// internal helper macros to make somewhat unique variable names based on line numbers to prevent some compilers
// warning about shadowed variables
#define WI_PROFILER_CONCAT(x,y) WI_PROFILER_CONCAT_INDIRECT(x,y)
#define WI_PROFILER_CONCAT_INDIRECT(x,y) x##y

namespace wi::profiler
{
	typedef size_t range_id;

	// Begin collecting profiling data for the current frame
	void BeginFrame();

	// Finalize collecting profiling data for the current frame
	void EndFrame(wi::graphics::CommandList cmd);

	// Start a CPU profiling range
	range_id BeginRangeCPU(const char* name);

	// Start a GPU profiling range
	range_id BeginRangeGPU(const char* name, wi::graphics::CommandList cmd);

	// End a profiling range
	void EndRange(range_id id);

	// helper using RAII to avoid having to manually call BeginRangeCPU/EndRange at beginning/end
	struct ScopedRangeCPU
	{
		range_id id;
		inline ScopedRangeCPU(const char* name) { id = BeginRangeCPU(name); }
		inline ~ScopedRangeCPU() { EndRange(id); }
	};

	// same for BeginRangeGPU
	struct ScopedRangeGPU
	{
		range_id id;
		inline ScopedRangeGPU(const char* name, wi::graphics::CommandList cmd) { id = BeginRangeGPU(name, cmd); }
		inline ~ScopedRangeGPU() { EndRange(id); }
	};

	// Renders a basic text of the Profiling results to the (x,y) screen coordinate
	void DrawData(
		const wi::Canvas& canvas,
		float x,
		float y,
		wi::graphics::CommandList cmd,
		wi::graphics::ColorSpace colorspace = wi::graphics::ColorSpace::SRGB
	);
	void DisableDrawForThisFrame();

	// Enable/disable profiling
	void SetEnabled(bool value);

	bool IsEnabled();

	void SetBackgroundColor(wi::Color color);
	void SetTextColor(wi::Color color);

	// GGMAX wall-gap tracer (Horseshoe Bend warm-up stall hunt): runs INDEPENDENTLY of the
	// profiler enable state so it never distorts the timing it measures. Main-loop phases
	// drop cheap timestamped marks; when the frame-to-frame wall gap at BeginFrame exceeds
	// the threshold, the previous frame's full segment ledger + PSO/texture creation deltas
	// are appended to gap_trace.txt next to the exe. Averages smear one-frame hitches —
	// this catches individual offenders.
	void gg_trace_mark(const char* name); // main thread only
	void gg_trace_mark_id(const char* prefix, unsigned int id); // formatted "<prefix>_<hex id>" mark
	unsigned long long gg_trace_now_us(void); // tracer clock, for caller-side gating
	void gg_trace_file_enable(bool enable); // GGMAX 2.71: false = counters stay live, no gap_trace.txt file

	// GGMAX 1.82: HITCH HISTOGRAM — every frame bucketed, not just the >100ms ones the tracer
	// above catches. Built to measure the cost of lazy object PSOs (first-use compile stalls),
	// where the sum of compile time and the mean frame rate both hide the thing that matters.
	// Rides the same always-on path as the tracer, so enabling the profiler is not required.
	// Buckets are frames slower than 16.7 / 25 / 33 / 50 / 100 ms.
#define GG_HITCH_BUCKETS 5
	void gg_hitch_reset(void);
	void gg_hitch_get(unsigned long long* frames, unsigned long long* over /*[GG_HITCH_BUCKETS]*/,
		unsigned long long* max_us, unsigned long long* total_us,
		unsigned long long* pso_compiles, unsigned long long* pso_compile_us);

	// Safe data access (no GPU calls, no rendering - just reads internal timing data)
	// Returns formatted text of all CPU and GPU profiling ranges with timing in ms.
	// Profiler must be enabled via SetEnabled(true) for data to be collected.
	std::string GetTextData();

	// GGMAX 2.94c: the biggest holes between measured GPU ranges in the latest frame, each
	// labelled with the range that closed before it and the range that opens after it.
	// Answers "where is the GPU Idle + unranged time", which no per-row total can.
	std::string GetGPUGapReport();

	// Returns CPU frame time in milliseconds (0 if profiler not enabled)
	float GetCPUFrameTime();

	// Returns GPU frame time in milliseconds (0 if profiler not enabled)
	float GetGPUFrameTime();
};

