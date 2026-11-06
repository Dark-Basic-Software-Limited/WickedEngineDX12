#include "wiTerrain.h"
#include "wiProfiler.h"
#include "wiTimer.h"
#include "wiRenderer.h"
#include "wiHelper.h"
#include "wiScene.h"
#include "wiPhysics.h"
#include "wiImage.h"
#include "wiFont.h"
#include "wiTextureHelper.h"
#include "wiBacklog.h"

#include <mutex>
#include <thread>
#include <condition_variable>
#include <shared_mutex>
#include "wiTerrain_ggdetail.h"
#include <string>
#include <atomic>
#include <deque>

using namespace wi::ecs;
using namespace wi::scene;
using namespace wi::graphics;

namespace wi::scene
{
	// GGMAX PROTOTYPE (BR-0009 round 7): object entity -> detail mesh the renderer draws instead (wiScene.cpp)
	extern wi::unordered_map<wi::ecs::Entity, wi::ecs::Entity> gg_detail_render_mesh;
}

namespace wi::terrain
{
	// GGMAX 2.53: generation-center override (see wiTerrain.h) — written by the game's
	// Terrain Generator each frame it is open; auto-cleared by the GGTerrainWicked bridge
	// whenever the generator is not the active mode.
	float gg_generation_center_override_x = 0.0f;
	float gg_generation_center_override_z = 0.0f;
	bool  gg_generation_center_override_enabled = false;

	// GGMAX 2.58: skip the per-chunk CPU triangle BVH build during generation. Measured at
	// 8.2 of the 10.9 ms per chunk (75%). The Terrain Generator sets this: its only terrain
	// picks are the marker-drag rays, which fall back to brute-force triangle tests on the
	// few AABB-passing chunks. Chunks born without a BVH are wiped on generator exit, so
	// the editor/test game always regenerate with BVHs.
	bool gg_generation_skip_bvh = false;

	// GGMAX 3.159 (BR-0009 round 2, m2): give terrain chunk meshes a CPU triangle pick BVH at all?
	// true = stock Wicked. GG sets it false in GGTerrainWicked_Init: wiScene's ray Intersects /
	// IntersectsAll / IntersectsFirst walk a chunk's regular LOD0 grid directly (exact, ~the same
	// cost as the BVH), so the BVH - 4.46 MB and 36 ms per chunk at 128+3 - buys nothing. Launch-time
	// only: turning it on at run time would make the heal build every chunk on the main thread.
	bool gg_terrain_pick_bvh = true;

	// GGMAX 3.159 (BR-0009 FPS): the game's per-chunk LOD bias rule (world AABB -> lod_bias), applied to chunks the
	// moment they merge. nullptr = stock (terrain.lod_bias). GG also re-applies it every frame for camera movement.
	float (*gg_chunk_lod_bias_fn)(const XMFLOAT3& aabb_min, const XMFLOAT3& aabb_max) = nullptr;

	// GGMAX 3.159 (BR-0009 round 2, m1): chunk meshes share ONE GPU copy of their byte-identical
	// topology (all-LOD indices, provoking indices, reorder table, UVs) - see
	// MeshComponent::SharedTopology. false = stock (every chunk uploads its own). GG sets it at Init.
	bool gg_share_chunk_topology = false;
	std::atomic<uint32_t> gg_topology_shared_chunks{ 0 };   // chunk meshes created with the shared topology
	std::atomic<uint32_t> gg_topology_unshared_chunks{ 0 }; // ... without it (flag off, or Matches() failed)
	std::atomic<uint32_t> gg_topology_builds{ 0 };          // shared topologies built

	// GGMAX 2.60: skip grass seeding/creation at chunk generation. The Terrain Generator
	// sets it (user request): grass is invisible at generator view distances, it carries a
	// known bug there, and the seeding costs ~0.3 ms/chunk. Generator chunks are wiped on
	// exit, so the editor/test game always regenerate WITH grass.
	bool gg_generation_skip_grass = false;

	// GGMAX 2.94d: how often the VT tile-request round trip runs, in frames. 1 = every frame
	// (stock upstream behaviour), 4 = the shipped default.
	//
	// Measured 2026-08-22 on TESTPRO2 (625-chunk ring): WritebackTileRequestsGPU 3.96 ms/frame
	// and AllocateTileRequests 1.09 ms/frame, against 0.62 ms to draw the entire scene. Writeback
	// issues FOUR CopyResource calls per virtual texture per frame - one allocation readback plus
	// three clear-copies - so ~2500 copies and their barriers every frame, purely for residency
	// bookkeeping. Residency feedback does not need per-frame latency; it decides which terrain
	// mip tiles to stream, and a 4-frame lag on that is imperceptible.
	//
	// ALLOCATE AND WRITEBACK MUST SHARE THE CADENCE. Writeback is what CLEARS feedbackMap,
	// requestBuffer and allocationBuffer. Skipping only writeback would leave TILEREQUESTS
	// re-appending into an uncleared requestBuffer every frame -> duplicate requests and a
	// plausible free-tile exhaustion. Skipping the pair is coherent instead: feedback simply
	// ACCUMULATES across the interval (the shader InterlockedOrs into it and nothing else
	// touches it), then one allocate pass converts the accumulated set and one writeback
	// exports and clears. More samples per decision, not fewer.
	//
	// CopyVirtualTexturePageStatusGPU is deliberately NOT gated - it is the opposite direction
	// (CPU page table -> GPU), it is already incremental via gg_vt_incremental (1.33), and it
	// measures 0.02 ms.
	int gg_vt_writeback_interval = 4;
	uint32_t gg_bake_resolution = 0;   // GGMAX 3.02: 0 = SVT_TILE_SIZE

	// GGMAX 2.94f: the GG bridge's terrain idle gate applies to the BRIDGE's Generation_Update
	// call only. Scene::Update (wiScene.cpp) calls Generation_Update a SECOND time on the same
	// frame, and that caller was never gated - so on a parked, settled scene the full chunk
	// walk (per chunk: frustum test + 4 component hash lookups, then UpdateVirtualTexturesCPU
	// over all of them again) still ran EVERY frame, 8 frames in 8.
	//
	// The bridge already computes exactly the right predicate and runs BEFORE __super::Update
	// in the same frame, so it just publishes it here. Default false = stock behaviour, so if
	// the bridge never sets it (standalone paths, other hosts) nothing changes.
	bool gg_skip_generation_update = false;

	// GGMAX 2.58 diagnostic: per-phase chunk-generation cost accumulators (microseconds,
	// cumulative since launch). Answers "what takes the most time when generating a chunk";
	// dumped by the game's TERRAIN_GENPROF harness command. renderdata is measured inside
	// its async job so phase sums rank consumers but overlap — wall time is gg_genprof_total.
	std::atomic<uint64_t> gg_genprof_heights_us{ 0 };
	std::atomic<uint64_t> gg_genprof_vertex_us{ 0 };
	std::atomic<uint64_t> gg_genprof_renderdata_us{ 0 };
	std::atomic<uint64_t> gg_genprof_bvh_us{ 0 };
	std::atomic<uint64_t> gg_genprof_bvh_events{ 0 };
	std::atomic<uint64_t> gg_genprof_grass_us{ 0 };
	std::atomic<uint64_t> gg_genprof_blendcb_us{ 0 };
	std::atomic<uint64_t> gg_genprof_regiontex_us{ 0 };
	std::atomic<uint64_t> gg_genprof_physics_us{ 0 };
	std::atomic<uint64_t> gg_genprof_total_us{ 0 };
	std::atomic<uint64_t> gg_genprof_chunks{ 0 };

	// GGMAX 2.61: main-thread VT/texture allocation attribution (see wiTerrain.h)
	std::atomic<uint64_t> gg_vtprof_updatecpu_us{ 0 };
	std::atomic<uint64_t> gg_vtprof_updatecpu_calls{ 0 };
	std::atomic<uint64_t> gg_vtprof_updatecpu_max_us{ 0 };
	std::atomic<uint64_t> gg_vtprof_wait0_us{ 0 };
	std::atomic<uint64_t> gg_vtprof_vtinit_us{ 0 };
	std::atomic<uint64_t> gg_vtprof_vtinit_events{ 0 };
	std::atomic<uint64_t> gg_vtprof_resinit_us{ 0 };
	std::atomic<uint64_t> gg_vtprof_resinit_events{ 0 };
	std::atomic<uint64_t> gg_vtprof_regionmain_us{ 0 };
	std::atomic<uint64_t> gg_vtprof_regionmain_events{ 0 };

	struct ChunkIndices
	{
		wi::vector<uint32_t> indices;
		struct LOD
		{
			uint32_t indexOffset = 0;
			uint32_t indexCount = 0;
		};
		wi::vector<LOD> lods;

		ChunkIndices()
		{
			const int max_lod = (int)std::log2(chunk_width - 3) + 1;
			lods.resize(max_lod);
			for (int lod = 0; lod < max_lod; ++lod)
			{
				lods[lod].indexOffset = (uint32_t)indices.size();

				if (lod == 0)
				{
					for (int x = 0; x < chunk_width - 1; x++)
					{
						for (int z = 0; z < chunk_width - 1; z++)
						{
							int lowerLeft = x + z * chunk_width;
							int lowerRight = (x + 1) + z * chunk_width;
							int topLeft = x + (z + 1) * chunk_width;
							int topRight = (x + 1) + (z + 1) * chunk_width;

							indices.push_back(topLeft);
							indices.push_back(lowerLeft);
							indices.push_back(lowerRight);

							indices.push_back(topLeft);
							indices.push_back(lowerRight);
							indices.push_back(topRight);
						}
					}
				}
				else
				{
					const int step = 1 << lod;
					// inner grid:
					for (int x = 1; x < chunk_width - 2; x += step)
					{
						for (int z = 1; z < chunk_width - 2; z += step)
						{
							int lowerLeft = x + z * chunk_width;
							int lowerRight = (x + step) + z * chunk_width;
							int topLeft = x + (z + step) * chunk_width;
							int topRight = (x + step) + (z + step) * chunk_width;

							indices.push_back(topLeft);
							indices.push_back(lowerLeft);
							indices.push_back(lowerRight);

							indices.push_back(topLeft);
							indices.push_back(lowerRight);
							indices.push_back(topRight);
						}
					}
					// bottom border:
					for (int x = 0; x < chunk_width - 1; ++x)
					{
						const int z = 0;
						int current = x + z * chunk_width;
						int neighbor = x + 1 + z * chunk_width;
						int connection = 1 + ((x + (step + 1) / 2 - 1) / step) * step + (z + 1) * chunk_width;

						indices.push_back(current);
						indices.push_back(neighbor);
						indices.push_back(connection);

						if (((x - 1) % (step)) == step / 2) // halfway fill triangle
						{
							int connection1 = 1 + (((x - 1) + (step + 1) / 2 - 1) / step) * step + (z + 1) * chunk_width;

							indices.push_back(current);
							indices.push_back(connection);
							indices.push_back(connection1);
						}
					}
					// top border:
					for (int x = 0; x < chunk_width - 1; ++x)
					{
						const int z = chunk_width - 1;
						int current = x + z * chunk_width;
						int neighbor = x + 1 + z * chunk_width;
						int connection = 1 + ((x + (step + 1) / 2 - 1) / step) * step + (z - 1) * chunk_width;

						indices.push_back(current);
						indices.push_back(connection);
						indices.push_back(neighbor);

						if (((x - 1) % (step)) == step / 2) // halfway fill triangle
						{
							int connection1 = 1 + (((x - 1) + (step + 1) / 2 - 1) / step) * step + (z - 1) * chunk_width;

							indices.push_back(current);
							indices.push_back(connection1);
							indices.push_back(connection);
						}
					}
					// left border:
					for (int z = 0; z < chunk_width - 1; ++z)
					{
						const int x = 0;
						int current = x + z * chunk_width;
						int neighbor = x + (z + 1) * chunk_width;
						int connection = x + 1 + (((z + (step + 1) / 2 - 1) / step) * step + 1) * chunk_width;

						indices.push_back(current);
						indices.push_back(connection);
						indices.push_back(neighbor);

						if (((z - 1) % (step)) == step / 2) // halfway fill triangle
						{
							int connection1 = x + 1 + ((((z - 1) + (step + 1) / 2 - 1) / step) * step + 1) * chunk_width;

							indices.push_back(current);
							indices.push_back(connection1);
							indices.push_back(connection);
						}
					}
					// right border:
					for (int z = 0; z < chunk_width - 1; ++z)
					{
						const int x = chunk_width - 1;
						int current = x + z * chunk_width;
						int neighbor = x + (z + 1) * chunk_width;
						int connection = x - 1 + (((z + (step + 1) / 2 - 1) / step) * step + 1) * chunk_width;

						indices.push_back(current);
						indices.push_back(neighbor);
						indices.push_back(connection);

						if (((z - 1) % (step)) == step / 2) // halfway fill triangle
						{
							int connection1 = x - 1 + ((((z - 1) + (step + 1) / 2 - 1) / step) * step + 1) * chunk_width;

							indices.push_back(current);
							indices.push_back(connection);
							indices.push_back(connection1);
						}
					}
				}

				lods[lod].indexCount = (uint32_t)indices.size() - lods[lod].indexOffset;
			}
		}
	};
	inline ChunkIndices& chunk_indices()
	{
		static ChunkIndices state;
		return state;
	}

	struct Generator
	{
		wi::scene::Scene scene; // The background generation thread can safely add things to this, it will be merged into the main scene when it is safe to do so
		wi::jobsystem::context workload;
		std::atomic_bool cancelled{ false };
		wi::vector<SplineComponent> splines;
		wi::vector<Chunk> removable_chunks; // chunks that were invalidated are regenerated on the generator thread. Before merging them with the scene, the previous version of them will need to be removed from the destination scene
		std::deque<Chunk> priority_invalidation; // to not let invalidation stuck at same chunks every frame while editing splines, for more appealing visual feedback
	};

	wi::jobsystem::context virtual_texture_ctx;

	// GGMAX 2.94: the async VT job is a FILE-SCOPE context, so game code tearing a Terrain down
	// (the "Terrain Off" brutal off-switch) has no way to join it. Generation_Restart already
	// does this join for exactly the reason spelled out in the GGMAX 1.45 comment below - the
	// job holds raw VirtualTexture pointers via virtual_textures_in_use and allocates/steals
	// from the atlas, so freeing VTs while it runs is a use-after-free. Generation_Cancel does
	// NOT join it (it only waits on generator->workload), so any external teardown that calls
	// only Generation_Cancel inherits that race. Expose the join.
	void gg_WaitVirtualTextureJob()
	{
		wi::jobsystem::Wait(virtual_texture_ctx);
	}

	static std::mutex locker;

	constexpr void weight_norm(XMFLOAT4& weights)
	{
		const float norm = 1.0f / (weights.x + weights.y + weights.z + weights.w);
		weights.x *= norm;
		weights.y *= norm;
		weights.z *= norm;
		weights.w *= norm;
	};


	void VirtualTextureAtlas::Residency::init(uint32_t resolution)
	{
		wi::Timer gg_t_resinit; // GGMAX 2.61: pool-miss cost (~10 device creates, 4 blocking uploads)
		this->resolution = resolution;
		GraphicsDevice* device = GetDevice();

		uint32_t lod_count = 0;
		uint32_t tile_count = 0;
		uint32_t side = resolution;
		while (side >= SVT_TILE_SIZE)
		{
			tile_count += (side / SVT_TILE_SIZE) * (side / SVT_TILE_SIZE);
			side /= 2;
			lod_count++;
		}
		tile_count += 1; // packed mips

		if (resolution > SVT_TILE_SIZE)
		{
			uint32_t granularity = resolution / SVT_TILE_SIZE;

			TextureDesc td;
			td.width = granularity;
			td.height = granularity;

			td.format = Format::R8G8B8A8_UINT;
			td.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			td.usage = Usage::DEFAULT;
			td.layout = ResourceState::UNORDERED_ACCESS;
			td.mip_levels = lod_count;
			wi::vector<uint32_t> default_data(td.width * td.height);
			wi::vector<SubresourceData> initdata(lod_count);
			for (uint32_t lod = 0; lod < lod_count; ++lod)
			{
				initdata[lod].data_ptr = default_data.data();
				initdata[lod].row_pitch = sizeof(uint32_t) * std::max(1u, td.width >> lod);
			}
			bool success = device->CreateTexture(&td, initdata.data(), &residencyMap);
			assert(success);
			device->SetName(&residencyMap, "VirtualTexture::residencyMap");

			for (uint32_t i = 0; i < lod_count; ++i)
			{
				int subresource_index = device->CreateSubresource(&residencyMap, SubresourceType::UAV, 0, 1, i, 1);
				assert(subresource_index == i);
			}

			td.format = Format::R32_UINT; // shader atomic support needed
			td.bind_flags = BindFlag::UNORDERED_ACCESS | BindFlag::SHADER_RESOURCE;
			td.usage = Usage::DEFAULT;
			td.layout = ResourceState::UNORDERED_ACCESS;
			td.mip_levels = 1;
			std::fill(default_data.begin(), default_data.end(), 0xFF);
			success = device->CreateTexture(&td, initdata.data(), &feedbackMap);
			assert(success);
			device->SetName(&feedbackMap, "VirtualTexture::feedbackMap");

			{
				GPUBufferDesc bd;
				bd.misc_flags = ResourceMiscFlag::BUFFER_RAW;
				bd.usage = Usage::DEFAULT;
				bd.bind_flags = BindFlag::UNORDERED_ACCESS;
				bd.size = sizeof(uint32_t) * tile_count;
				wi::vector<uint32_t> data(tile_count);
				std::fill(data.begin(), data.end(), 0xFF);
				success = device->CreateBuffer(&bd, data.data(), &requestBuffer);
				assert(success);
				device->SetName(&requestBuffer, "VirtualTexture::requestBuffer");
			}
			{
				GPUBufferDesc bd;
				bd.misc_flags = ResourceMiscFlag::BUFFER_RAW;
				bd.usage = Usage::DEFAULT;
				bd.bind_flags = BindFlag::UNORDERED_ACCESS;
				bd.size = sizeof(uint32_t) * (tile_count + 1); // +1: atomic global counter
				wi::vector<uint32_t> data(tile_count + 1);
				success = device->CreateBuffer(&bd, data.data(), &allocationBuffer);
				assert(success);
				device->SetName(&allocationBuffer, "VirtualTexture::allocationBuffer");

				bd.misc_flags = {};
				bd.bind_flags = BindFlag::NONE;
				bd.usage = Usage::READBACK;
				for (int i = 0; i < arraysize(allocationBuffer_CPU_readback); ++i)
				{
					success = device->CreateBuffer(&bd, nullptr, &allocationBuffer_CPU_readback[i]);
					assert(success);
					device->SetName(&allocationBuffer_CPU_readback[i], "VirtualTexture::allocationBuffer_CPU_readback[i]");
				}
			}
			{
				GPUBufferDesc bd;
				bd.format = Format::R16_UINT;
				bd.usage = Usage::DEFAULT;
				bd.bind_flags = BindFlag::SHADER_RESOURCE;
				bd.size = GetFormatStride(bd.format) * tile_count;
				success = device->CreateBuffer(&bd, nullptr, &pageBuffer);
				assert(success);
				device->SetName(&pageBuffer, "VirtualTexture::pageBuffer");

				bd.misc_flags = {};
				bd.usage = Usage::UPLOAD;
				bd.bind_flags = {};
				for (int i = 0; i < arraysize(pageBuffer_CPU_upload); ++i)
				{
					success = device->CreateBuffer(&bd, nullptr, &pageBuffer_CPU_upload[i]);
					assert(success);
					device->SetName(&pageBuffer_CPU_upload[i], "VirtualTexture::pageBuffer_CPU_upload[i]");
				}
			}
		}
		gg_vtprof_resinit_us.fetch_add(uint64_t(gg_t_resinit.elapsed_milliseconds() * 1000.0)); // GGMAX 2.61
		gg_vtprof_resinit_events.fetch_add(1);
	}
	void VirtualTextureAtlas::Residency::reset()
	{
		for (int i = 0; i < arraysize(data_available_CPU); ++i)
		{
			data_available_CPU[i] = false;
		}
		cpu_resource_id = 0;
	}

	// GGMAX 1.33: master switch for incremental VT bookkeeping (see wiTerrain.h)
	bool gg_vt_incremental = true;

	// GGMAX diag (Horseshoe Bend 42ms FreeSort hunt): free-list rebuild forensics,
	// running totals read via the game harness GET_PERF_DATA "VT:" line.
	// GGMAX 1.71 / 1.76: SVT physical atlas height (see the atlas creation site). Upstream stock
	// is 16384 = 3844 tiles = a 768 MB pool paid by EVERY level, indoor cellars included.
	//
	// DEFAULT IS NOW 12288 (2852 tiles, ~576 MB) — soaked, not guessed. tools/svt_soak.sh flies
	// Z Island over 8 waypoints for 3 laps with short dwells so the atlas must evict and
	// re-stream. Peak tile demand under that travel is 1971:
	//     16384  3844 tiles, min free 1873  (49% headroom)
	//     12288  2852 tiles, min free  881  (31% headroom) — peak used 1971, IDENTICAL to stock,
	//                                        i.e. every request served, no mip fallback
	//      8192  1922 tiles, min free    0  STARVED — 1922 < 1971, terrain visibly blurs
	// So 8192 is NOT viable; do not "just halve it". 12288 saves ~192-256 MB on every level.
	//
	// setup.ini `svtatlasheight` still overrides (2048..16384) — set 16384 to A/B against stock.
	// Applied when the atlas is next created, so it needs a level (re)load.
	uint32_t gg_svt_atlas_height = 12288u;

	// GGMAX 1.98 (A4): MAX never assigns EMISSIVEMAP on any terrain material (only basecolor/
	// normal/surface — GGTerrainWicked.cpp), yet the atlas allocated a fourth sparse map and
	// its pool share unconditionally. BC1 emissive = 1/6 of the pool = 96 MB at the shipping
	// 12288 atlas. Dropping it requires binding an EXPLICIT 1x1 black into the material's
	// emissive slot — the 2026-08-02 attempt left the slot unbound and the whole scene washed
	// to white (undefined sparse sample feeding the tonemapper). setup.ini `svtemissive=1`
	// restores the stock fourth map (needs level reload).
	bool gg_svt_keep_emissive = false;

	std::atomic<unsigned long long> gg_dbg_vt_rebuilds{ 0 };   // rebuild executions
	std::atomic<unsigned long long> gg_dbg_vt_scan_us{ 0 };    // clear+scan phase time
	std::atomic<unsigned long long> gg_dbg_vt_sort_us{ 0 };    // std::sort phase time
	std::atomic<unsigned long long> gg_dbg_vt_free{ 0 };       // last free-list size after rebuild
	std::atomic<unsigned long long> gg_dbg_vt_requests{ 0 };   // cumulative allocation requests
	std::atomic<unsigned long long> gg_dbg_vt_reason{ 0 };     // cumulative trigger bits: 1=knob 2=requests 4=dirty 8=lowwater 16=center (per-frame OR summed)
	std::atomic<unsigned long long> gg_dbg_vt_tiles{ 0 };      // physical tile count (last)
	static unsigned long long gg_dbg_vt_qpc_us(void)
	{
		LARGE_INTEGER f, c;
		QueryPerformanceFrequency(&f);
		QueryPerformanceCounter(&c);
		return (unsigned long long)((c.QuadPart * 1000000.0) / (double)f.QuadPart);
	}
	// GGMAX 1.53b: terrain VT tiling repeat CAP — every bake rung finer than this many repeats
	// per chunk region bakes the SAME cap-scale layout (pure downsample chain = invisible
	// trilinear transitions from the camera through the mid field); rungs at/below the cap keep
	// the stock repeat-halving anti-tiling for the far field. World-anchored, so resolution
	// rings/promotions cannot cause seams or scale pops. 0 = stock (scale cross-fades start at
	// the camera). Chunk ≈ 5120 in: cap 8 ≈ 16m texture repeat, 16 ≈ 8m, 32 ≈ 4m. Live-tunable:
	// harness SET_TERRAINTILE <cap> [hold] (queues a fast repaint of resident chunk VTs on change).
	// GGMAX 3.113: SPLAT blend for the terrain compositor (terrainVirtualTextureUpdateCS / terrainBakeChunkCS,
	// push.gg_tile_share bit 0): blendmap weights are splat weights summing to 1 and are AVERAGED, instead of
	// the stock front-to-back opacity stack. GameGuru sets it (its auto-blend writes DX11 splat weights).
	bool gg_splat_blend = false;
	bool gg_blendmap_half_res = false; // GGMAX PROTOTYPE (BR-0009 round 6), see wiTerrain.h
	// texels a side of the GPU blendmap (every reader samples it at normalised chunk UV, never by texel index)
	static inline uint32_t gg_BlendmapWidth()
	{
		return gg_blendmap_half_res ? uint32_t((chunk_width - 1) / 2 + 1) : uint32_t(chunk_width);
	}
	// =============================================================================================================
	// GGMAX PROTOTYPE (BR-0009 round 7 + round 8 LOD): NEAR-CAMERA DETAIL MESHES.
	// Lee 2026-11-06: "match the DX11 sculpt result as close as practicable" (DX11 GGTerrain draws 8-unit vertices
	// near the camera; the chunk grid is 40/80). Chunks within gg_detail_ring chunks (Chebyshev) of the camera chunk
	// get a second mesh on a lattice gg_detail_factor x denser per side, sampled from the SAME height modifiers. The
	// renderer draws it instead of the chunk mesh (wi::scene::gg_detail_render_mesh, consulted ONLY where Scene picks
	// object.mesh_index); the chunk keeps its ObjectComponent (material/VT, occlusion, shadow flags, wetmap) and its
	// coarse MeshComponent, so every CPU consumer (picks, GG blend/grass/bake passes) is untouched.
	// Round 8 (Lee 2026-11-06 evening: "the mid-distance wireframe is practically filled in ... WAY too much density
	// at this range"): DISTANCE LOD inside the detail meshes. A detail mesh is 13 x 13 TILES (10 chunk cells, 400
	// units at scale 40); a tile's vertex spacing is gg_detail_lod_step0 lattice steps (1 = 8 units at factor 5)
	// within gg_detail_lod_dist0 of the LOD centre, step1 within dist1, and the chunk grid's own spacing (factor
	// steps = 40 units) beyond - the same doubling-then-coarse ladder DX11's clipmap draws (8 / 16 / 32...).
	// Seams, by construction: a tile EDGE uses the COARSER spacing of the two tiles that share it (a neighbour chunk
	// outside the ring, or not generated yet, counts as the chunk grid), every vertex is sampled exactly from the
	// height function at its lattice position (with the chunk generator's own lerp, so a detail border vertex is
	// bit-identical to the chunk mesh's), so both sides of every edge - inside a mesh, between two detail meshes and
	// against a coarse chunk mesh - hold IDENTICAL vertices: no stitching, no T-junctions. Each tile's border band
	// (its edge vertices -> its first interior row) is a zipper triangulation, so any two spacings meet.
	// The LOD centre and the ring centre follow the camera with hysteresis (gg_detail_lod_hyst), and only once the
	// previous generation has swapped in (a moving camera cannot keep superseding it). A new centre (or ring /
	// setting) rebuilds only the ring chunks whose tile spacings changed (a per-chunk signature), and the rebuilt
	// meshes swap in TOGETHER, in one frame, once all of them are ready - the old set (self-consistent) draws
	// meanwhile, so a moving camera never shows a crack between a new and an old mesh. Builds run on a PRIVATE thread
	// pool (gg_DetailPool), never in wi::jobsystem: a jobsystem Wait on a Low context runs every queued Low job inline,
	// so GG's Generation_Cancel (every sculpt frame) could otherwise run a whole detail build on the main thread.
	// -1 = off (stock). GameGuru sets these (GGTerrainWicked_Init / harness SET_TERRAINDETAIL, SET_TERRAINDETAILLOD).
	int gg_detail_ring = -1;
	int gg_detail_factor = 5;           // 5 x 40-unit chunk grid = 8-unit lattice, DX11's finest GGTerrain spacing
	int gg_detail_budget = 3;           // detail builds launched per frame at most (camera chunk first)
	int gg_detail_lod_step0 = 1;        // lattice steps between vertices within dist0 (1 = 8 units at factor 5)
	int gg_detail_lod_step1 = 2;        // within dist1 (2 = 16 units); beyond: factor steps (the chunk grid, 40)
	float gg_detail_lod_dist0 = 1500;   // world units, LOD centre -> nearest point of a tile (Lee 2026-11-06 19:15: 1500 / 3750;
	float gg_detail_lod_dist1 = 3750;   // GG drives both from the per-level Terrain Detail Distance, dist1 = 2.5 x dist0)
	float gg_detail_lod_hyst = 400;     // the LOD / ring centre follows the camera once it strays this far
	// GGMAX round 10 (Lee 2026-11-06: "blend the normals/lighting ... a gaussian effect"): GG's sculpt / heightmap data is
	// BILINEAR on its own grid (the 4096 sculpt map over the editable area = ~24 units a texel at the default size), so
	// the surface's slope jumps at every texel edge; normals from +-8-unit differences reproduce those jumps and the
	// lighting shows hard, triangle-like facets although the shape is right. > 0 = each detail vertex's normal is a Sobel
	// (smoothed) slope at +-this many world units instead: positions untouched, the shading rolls over the texel edges.
	// 0 = the round-8 central differences at the finest spacing. World units; GG sets it (harness SET_TERRAINNORMALSMOOTH).
	float gg_detail_normal_radius = 0.0f;
	// GGMAX round 10t (Lee 2026-11-06, the "fingers" at the foot of a steep sculpt): the detail meshes' TANGENT was the
	// chunk convention corner[x+1] - corner[z+1] = (s, h(x+s,z) - h(x,z+s), -s) - a one-sided difference between two
	// samples s apart in different directions. On a steep wall's foot one sample lands on the ground and the other up
	// the wall, so the tangent flipped between up and down from vertex to vertex; interpolated over the long wall
	// triangles it turned the material normal map's x axis around in saw-teeth (TANGENTVIS 1) = the fingers in the
	// lighting. 1 = the same convention taken from the vertex NORMAL, T = (n.y, n.z - n.x, -n.y) normalised: the
	// diagonal direction (1, dh/dx - dh/dz, -1) of the surface the normal describes - identical to first order on gentle
	// ground, orthogonal to N by construction, and as smooth as the normal. 0 = the old difference (A/B).
	int gg_detail_tangent_from_normal = 1;
	// GGMAX round 10c (Lee 2026-11-06, the "fingers" at the foot of a steep sculpt - the cause found by elimination:
	// not the normals, not the tangent, not GG's bilinear creases; AO OFF removes them and a 4-unit lattice halves
	// them): the detail lattice POINT-samples the height, and a sharp crease (a wall meeting the ground) that runs
	// obliquely across the lattice is aliased - every vertex lands either on the ground or up the wall, the foot
	// becomes a staircase, and the long wall triangles rising from each step are real ridges that AO then darkens.
	// 1 = anti-alias the creases: a vertex whose +-step0 second difference exceeds gg_detail_crease_k x the vertex
	// spacing (the slope bends by more than k within a cell) takes a 5 x 5 tent-weighted AREA sample of the height
	// over +-2/3 of the spacing instead of its point sample - the crease's sub-cell position survives as intermediate
	// heights, so the foot runs smoothly instead of in steps. Only crease vertices pay (24 extra samples each); gentle
	// ground is untouched (point samples, bit-identical). Chunk-edge vertices keep their point samples (both chunks
	// must compute them identically; a neighbour's pad samples can differ from its own by an ulp). 0 = off (A/B).
	int gg_detail_crease_aa = 1;
	float gg_detail_crease_k = 0.5f;
	std::atomic<uint32_t> gg_detail_aa_verts{ 0 };   // crease vertices area-sampled by the last build to finish
	// GGMAX round 10d (Lee 2026-11-06, the "fingers" - the cause, from GG's own height function dumped at 1 unit and
	// re-meshed offline): every quad was split along the SAME diagonal (TL-LR, the chunk's). A steep sculpt's foot that
	// runs against that diagonal is cut into a row of saw-teeth one cell apart - long thin triangles rising from each
	// tooth = the fingers, which AO darkens. 1 = each interior quad takes its SHORTER 3D diagonal (the one joining the
	// two corners of closest height, the classic heightfield rule): the foot follows the outline instead. Vertices and
	// normals are unchanged (only the index order), ties keep the chunk's diagonal (a plane is the same either way), the
	// zipper bands at tile edges keep theirs. 0 = the fixed diagonal (A/B).
	int gg_detail_adaptive_diagonal = 1;
	std::atomic<uint32_t> gg_detail_flipped_quads{ 0 }; // interior quads the last build to finish flipped
	uint32_t gg_detail_count = 0;       // chunks drawing a detail mesh this frame
	uint32_t gg_detail_builds = 0;      // detail meshes swapped in since start
	uint32_t gg_detail_last_built = 0;  // swapped in by the last frame that swapped any
	float gg_detail_last_ms = 0;        // main-thread ms of that frame's swaps
	uint64_t gg_detail_tris = 0;        // triangles of the last mesh swapped in
	float gg_detail_sample_ms = 0;      // last mesh swapped in: topology+sampling+attributes / CreateRenderData ms
	float gg_detail_upload_ms = 0;
	float gg_detail_build_ms = 0;       // last mesh swapped in: its whole build (one pool thread, serial)
	float gg_detail_build_ms_max = 0;   // the slowest build of the last frame that swapped any
	uint64_t gg_detail_drawn_tris = 0;  // this frame: triangles / vertices of every detail mesh drawn
	uint64_t gg_detail_drawn_verts = 0;
	uint32_t gg_detail_swaps = 0;       // generation swaps (a new centre / ring / setting fully swapped in)
	uint32_t gg_detail_pending = 0;     // ring chunks still waiting for their rebuild (the generation swap waits)
	uint32_t gg_detail_not_current = 0; // drawn ring meshes whose signature is not the current target's (old generation)
	uint32_t gg_detail_threads = 0;     // private build threads
	float gg_detail_centre_x = 0, gg_detail_centre_z = 0;
	int gg_detail_eff_step0 = 0, gg_detail_eff_step1 = 0; // what the builder actually uses (steps validated, dists ordered)
	float gg_detail_eff_dist0 = 0, gg_detail_eff_dist1 = 0;

	static constexpr int gg_detail_tiles = 13; // tiles a side per chunk: 130 chunk cells / 10
	static_assert((chunk_width - 1) % gg_detail_tiles == 0, "detail tiles must split the chunk cells evenly");
	struct GGDetailLod
	{
		float cx = 0, cz = 0;          // LOD centre (world)
		float stride = 0;              // chunk size (world)
		float dist0 = 0, dist1 = 0;
		int F = 5;                     // lattice steps per chunk cell (gg_detail_factor)
		int step0 = 1, step1 = 2;      // validated: each divides the tile size and is <= F
		int nsteps = 0;                // round 10: Sobel normal stencil in lattice steps (0 = central differences at step0)
		int tmode = 1;                 // round 10t: gg_detail_tangent_from_normal (captured per target)
		float aak = 0;                 // round 10c: crease anti-aliasing threshold (0 = off; captured per target)
		int adiag = 1;                 // round 10d: gg_detail_adaptive_diagonal (captured per target)
	};
	// a step a tile can use: <= F (never coarser than the chunk grid, which every ring-border edge must match) and a
	// divisor of the tile size 10*F (so tile corners and edges land on its vertices)
	static inline int gg_DetailValidStep(int s, int F)
	{
		const int TS = ((chunk_width - 1) / gg_detail_tiles) * F;
		s = std::max(1, std::min(s, F));
		while (TS % s != 0)
			s--;
		return s;
	}
	// round 10: gg_detail_normal_radius in lattice steps (>= step0, so never finer than the mesh); 0 = off
	static inline int gg_DetailNormalSteps(int F, int step0, float chunk_scale)
	{
		if (!(gg_detail_normal_radius > 0.0f) || F < 1 || !(chunk_scale > 0.0f))
			return 0;
		const float unit = chunk_scale / float(F);
		return std::max(step0, std::min(64, (int)std::lround(gg_detail_normal_radius / unit)));
	}
	// lattice step of tile (tx, tz) of chunk (x, z). tx / tz may be -1 or 13: the neighbour chunk's border tile, which
	// is first normalised to ITS chunk so the arithmetic is bit-identical to that chunk's own (a step that flipped at a
	// threshold on one side only would open a crack). A pure function: the main thread and every build agree.
	static inline int gg_DetailTileStep(const GGDetailLod& L, int x, int z, int tx, int tz)
	{
		const int T = gg_detail_tiles;
		if (tx < 0) { tx += T; x -= 1; } else if (tx >= T) { tx -= T; x += 1; }
		if (tz < 0) { tz += T; z -= 1; } else if (tz >= T) { tz -= T; z += 1; }
		const float tw = L.stride / float(T);
		const float x0 = float(x) * L.stride - 0.5f * L.stride + float(tx) * tw;
		const float z0 = float(z) * L.stride - 0.5f * L.stride + float(tz) * tw;
		const float dx = std::max(0.0f, std::max(x0 - L.cx, L.cx - (x0 + tw)));
		const float dz = std::max(0.0f, std::max(z0 - L.cz, L.cz - (z0 + tw)));
		const float d2 = dx * dx + dz * dz;
		if (d2 < L.dist0 * L.dist0) return L.step0;
		if (d2 < L.dist1 * L.dist1) return L.step1;
		return L.F;
	}
	struct GGDetailInputs
	{
		XMFLOAT3 position = XMFLOAT3(0, 0, 0);  // the chunk's world position (its centre)
		float scale = 1, bottomLevel = 0, topLevel = 1;
		wi::vector<std::shared_ptr<Modifier>> modifiers; // copies of the shared_ptrs: the job never touches Terrain
		GGDetailLod L;
		int cx = 0, cz = 0;                    // chunk coordinates
		uint32_t nmask = 0;                    // edge bits 1:-x 2:+x 4:-z 8:+z set = that neighbour is a detail chunk
		bool doubleSidedShadow = false;
		Entity material = INVALID_ENTITY;      // the chunk entity (its material / virtual texture)
		bool external = false;                 // round 9: an external ring's build - stop before CreateRenderData (the host packs)
	};
	// One background build. The record keeps it in flight; done is set last, after mesh and stats are complete.
	struct GGDetailJob
	{
		std::atomic<bool> done{ false };
		std::atomic<bool> cancelled{ false };  // superseded / torn down: the build stops at its next phase
		GGDetailInputs in;
		Entity chunk_entity = INVALID_ENTITY;
		uint64_t sig = 0;
		uint32_t tris = 0, verts = 0;
		uint32_t unexpected = 0;               // triangles whose lattice winding disagreed with their emitter's (a fold)
		float build_ms = 0, sample_ms = 0, upload_ms = 0;
		MeshComponent mesh;
	};
	struct GGDetailRec
	{
		Entity mesh_entity = INVALID_ENTITY;   // the detail MeshComponent (an entity of its own, no object/transform)
		Entity chunk_entity = INVALID_ENTITY;  // the chunk entity it was built for (a Generation_Restart reuses coords)
		uint64_t sig = 0;                      // signature of the mesh in mesh_entity (0 = none)
		uint32_t tris = 0, verts = 0, unexpected = 0;
		bool dirty = true;                     // the chunk was regenerated (sculpt) since the last launch
		std::shared_ptr<GGDetailJob> job;      // a build in flight (or finished, not yet swapped in)
	};
	static wi::unordered_map<Chunk, GGDetailRec> gg_detail_recs; // ska flat map: NEVER hold a reference across an insert
	static GGDetailLod gg_detail_L;               // the current target (its centre carries the hysteresis)
	static bool gg_detail_have_centre = false;
	static Chunk gg_detail_cc;                    // the ring centre (hysteresis like the LOD centre)
	static bool gg_detail_have_cc = false;
	static uint32_t gg_detail_pending_frames = 0; // frames in a row with a generation pending

	// The private build pool: plain threads at the jobsystem's Low OS priority, a FIFO of jobs, no work stealing by
	// anyone else. Each build runs serially on one thread; up to 3 build at once. Never destroyed (its threads are
	// detached and sleep for the life of the process); every teardown path drains it first (gg_DetailPoolDrain).
	class GGDetailPool
	{
	public:
		void Push(std::function<void()> f)
		{
			std::unique_lock<std::mutex> lock(m);
			if (threads == 0)
				Start();
			q.push_back(std::move(f));
			cv.notify_one();
		}
		// queued jobs are dropped (their records are being cleared), running ones finish
		void Drain()
		{
			std::unique_lock<std::mutex> lock(m);
			q.clear();
			idle.wait(lock, [this] { return busy == 0; });
		}
		uint32_t Load()
		{
			std::unique_lock<std::mutex> lock(m);
			return uint32_t(q.size()) + uint32_t(busy);
		}
		uint32_t Threads() const { return threads; }
	private:
		void Start() // m held
		{
			const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
			const unsigned n = std::max(1u, std::min(3u, hc / 4));
			for (unsigned i = 0; i < n; ++i)
			{
				std::thread th([this] {
#ifdef _WIN32
					SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST); // as wi::jobsystem's Low pool
					SetThreadDescription(GetCurrentThread(), L"gg_terrain_detail");
#endif
					for (;;)
					{
						std::function<void()> f;
						{
							std::unique_lock<std::mutex> lock(m);
							cv.wait(lock, [this] { return !q.empty(); });
							f = std::move(q.front());
							q.pop_front();
							busy++;
						}
						f();
						{
							std::unique_lock<std::mutex> lock(m);
							busy--;
							if (busy == 0)
								idle.notify_all();
						}
					}
				});
				th.detach();
				threads++;
			}
		}
		std::mutex m;
		std::condition_variable cv, idle;
		std::deque<std::function<void()>> q;
		uint32_t threads = 0;
		int busy = 0;
	};
	static GGDetailPool& gg_DetailPool() { static GGDetailPool* p = new GGDetailPool(); return *p; }
	// round 9: the EXTERNAL ring's own pool (the live pool's Drain - Generation_Restart, teardown - clears ITS queue only)
	static GGDetailPool& gg_DetailPoolExt() { static GGDetailPool* p = new GGDetailPool(); return *p; }
	// round 9: the height-data fence (wiTerrain_ggdetail.h). Never destroyed: detached build threads may hold it at exit.
	std::shared_mutex& gg_DetailHeightLock() { static std::shared_mutex* m = new std::shared_mutex(); return *m; }

	static inline Chunk gg_MakeChunk(int x, int z) { Chunk c; c.x = x; c.z = z; return c; }

	// what a chunk's detail mesh depends on besides the height data: its tile steps, its detail neighbours' border tile
	// steps, the normal step, the lattice and the chunk entity. Equal signature = the drawn mesh is still right.
	static uint64_t gg_DetailSig(const GGDetailLod& L, const Chunk& c, uint32_t nmask, Entity e)
	{
		uint64_t h = 1469598103934665603ull;
		auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ull; };
		mix(uint64_t(e)); mix(uint64_t(L.F)); mix(uint64_t(L.step0)); mix(uint64_t(nmask)); mix(uint64_t(L.nsteps)); mix(uint64_t(L.tmode)); { uint32_t kb; std::memcpy(&kb, &L.aak, 4); mix(uint64_t(kb)); } mix(uint64_t(L.adiag));
		const int T = gg_detail_tiles;
		for (int tz = 0; tz < T; ++tz)
			for (int tx = 0; tx < T; ++tx)
				mix(uint64_t(gg_DetailTileStep(L, c.x, c.z, tx, tz)));
		for (int k = 0; k < T; ++k)
		{
			mix((nmask & 1) ? uint64_t(gg_DetailTileStep(L, c.x, c.z, -1, k)) : 0);
			mix((nmask & 2) ? uint64_t(gg_DetailTileStep(L, c.x, c.z, T, k)) : 0);
			mix((nmask & 4) ? uint64_t(gg_DetailTileStep(L, c.x, c.z, k, -1)) : 0);
			mix((nmask & 8) ? uint64_t(gg_DetailTileStep(L, c.x, c.z, k, T)) : 0);
		}
		return h == 0 ? 1 : h;
	}

	// the wetmap is one value per DRAWN vertex, in the drawn mesh's vertex order: whenever the drawn topology changes
	// (wiScene only compares the SIZE - an all-chunk-grid detail mesh has the chunk mesh's 131 x 131 vertices in another
	// order), drop it; wiScene recreates it zeroed and the ocean refresh re-wets the underwater vertices the same frame.
	static void gg_DetailResetWetmap(Scene& scene, Entity chunk_entity)
	{
		ObjectComponent* o = scene.objects.GetComponent(chunk_entity);
		if (o != nullptr)
			o->wetmap = {};
	}

	static void gg_DetailClearAll(Scene* scene)
	{
		for (auto& it : gg_detail_recs)
		{
			if (it.second.job)
				it.second.job->cancelled.store(true);
			if (scene != nullptr && it.second.mesh_entity != INVALID_ENTITY)
			{
				scene->Entity_Remove(it.second.mesh_entity);
				gg_DetailResetWetmap(*scene, it.second.chunk_entity);
			}
		}
		gg_detail_recs.clear(); // jobs still in flight keep their own reference; their result is dropped
		wi::scene::gg_detail_render_mesh.clear();
		gg_detail_count = 0;
		gg_detail_drawn_tris = 0;
		gg_detail_drawn_verts = 0;
		gg_detail_pending = 0;
		gg_detail_not_current = 0;
		gg_detail_have_centre = false;
		gg_detail_have_cc = false;
		gg_detail_pending_frames = 0;
	}

	// GGMAX PROTOTYPE (BR-0009 round 8): every path that tears the terrain down (Generation_Restart, the terrain-removed
	// branch, GameGuru's Terrain Off / Terrain Bake via GGTerrainWicked_Shutdown): drain the build pool (a running build
	// reads the height modifiers - GG's heightmap - until it finishes), then remove every detail mesh entity.
	void gg_DetailShutdown(Scene* scene)
	{
		for (auto& it : gg_detail_recs)
			if (it.second.job)
				it.second.job->cancelled.store(true);
		gg_DetailPool().Drain();
		gg_DetailClearAll(scene);
	}

	// Builds one detail mesh into job.mesh (a fresh MeshComponent), on a pool thread, serially (no nested parallelism:
	// see gg_DetailPool). Heights through the same modifiers and lerp as the chunk generator; CreateRenderData here.
	static void gg_BuildDetailMesh(GGDetailJob& job)
	{
		wi::Timer gg_t_build;
		const GGDetailInputs& in = job.in;
		MeshComponent& mesh = job.mesh;
		const GGDetailLod& L = in.L;
		const int F = L.F;
		const int C = chunk_width - 1;        // chunk cells a side (130)
		const int T = gg_detail_tiles;        // tiles a side (13)
		const int TS = (C / T) * F;           // lattice steps a tile side (50 at factor 5)
		const int D = C * F;                  // lattice steps a chunk side (650)
		const int N = D + 1;                  // lattice vertices a side
		const int ns = L.step0;               // normal step: central differences at the finest spacing in use
		const int nd = std::max(L.nsteps, 0); // round 10: > 0 = Sobel normals at +-nd lattice steps (smoothed)
		const int pad = std::max(ns, nd);     // the ring outside the chunk those differences reach into
		const int NP = N + 2 * pad;
		const float scale = in.scale;
		const float invF = 1.0f / float(F);
		// local coordinate of lattice column i: the chunk vertex formula at fractional coord i/F - so i = c*F lands on
		// exactly the float the chunk (and its neighbour) computes for chunk vertex c
		auto lc = [&](int i) { return (float(i) * invF - chunk_half_width) * scale; };

		// 1. tile steps, with a border of the neighbours' tiles (a neighbour that is not a detail chunk = the grid, F)
		wi::vector<int> tstep(size_t(T + 2) * size_t(T + 2), F);
		auto TSt = [&](int tx, int tz) -> int& { return tstep[size_t(tz + 1) * size_t(T + 2) + size_t(tx + 1)]; };
		for (int tz = -1; tz <= T; ++tz)
		{
			for (int tx = -1; tx <= T; ++tx)
			{
				const bool outx = tx < 0 || tx >= T, outz = tz < 0 || tz >= T;
				if (outx && outz)
					continue; // corner tiles never share an edge with this chunk
				const uint32_t bit = tx < 0 ? 1u : (tx >= T ? 2u : (tz < 0 ? 4u : (tz >= T ? 8u : 0u)));
				TSt(tx, tz) = (bit == 0 || (in.nmask & bit) != 0) ? gg_DetailTileStep(L, in.cx, in.cz, tx, tz) : F;
			}
		}

		// 2. topology: one vertex per lattice point used, triangles wound like the chunk's (TL LL LR: +cross in x,z).
		//    Each emitter knows its triangles' natural winding (interior / -z / +x bands +, +z / -x bands -: mirrored
		//    mappings); a triangle that disagrees would be a fold, and is counted (job.unexpected, harness).
		wi::vector<uint32_t> vid(size_t(N) * size_t(N), ~0u);
		wi::vector<XMINT2> vlat; // lattice coordinate of each vertex
		wi::vector<uint32_t> idx;
		wi::vector<uint32_t> quadAt; // round 10d: idx offset of each interior quad's 6 indices (TL LL LR TL LR TR)
		vlat.reserve(65536);
		idx.reserve(65536 * 6);
		auto V = [&](int i, int j) -> uint32_t {
			uint32_t& v = vid[size_t(j) * size_t(N) + size_t(i)];
			if (v == ~0u)
			{
				v = (uint32_t)vlat.size();
				vlat.push_back(XMINT2(i, j));
			}
			return v;
		};
		auto tri = [&](uint32_t a, uint32_t b, uint32_t c, int expect) {
			const XMINT2 A = vlat[a], B = vlat[b], Cc = vlat[c];
			const int64_t cr = int64_t(B.x - A.x) * int64_t(Cc.y - A.y) - int64_t(B.y - A.y) * int64_t(Cc.x - A.x);
			if (cr == 0 || (cr > 0) != (expect > 0))
				job.unexpected++;
			idx.push_back(a);
			if (cr >= 0) { idx.push_back(b); idx.push_back(c); }
			else { idx.push_back(c); idx.push_back(b); }
		};
		wi::vector<uint32_t> O, I;
		wi::vector<int> Oa, Ia;
		for (int tz = 0; tz < T; ++tz)
		{
			for (int tx = 0; tx < T; ++tx)
			{
				const int s = TSt(tx, tz);
				const int n = TS / s;               // cells a side at this tile's spacing (>= 10)
				const int bx = tx * TS, bz = tz * TS; // tile origin (lattice)
				// interior quads (the chunk's quad shape and diagonal)
				for (int b = 1; b < n - 1; ++b)
				{
					for (int a = 1; a < n - 1; ++a)
					{
						const int x0 = bx + a * s, z0 = bz + b * s;
						const uint32_t LL = V(x0, z0), LR = V(x0 + s, z0), TL = V(x0, z0 + s), TR = V(x0 + s, z0 + s);
						if (L.adiag != 0)
							quadAt.push_back((uint32_t)idx.size());
						tri(TL, LL, LR, 1);
						tri(TL, LR, TR, 1);
					}
				}
				// a border band: outer = the tile edge at spacing e (the coarser of the two tiles sharing it), inner = the
				// first interior row at s. at(u, d) maps (along the edge 0..TS, depth 0 or s) to the lattice. The first
				// and last rungs are the tile's corner diagonals, shared with the neighbouring bands; the zipper walks
				// both rows in order, so its rungs never cross and every triangle has positive area.
				auto band = [&](int e, int expect, auto&& at) {
					O.clear(); Oa.clear(); I.clear(); Ia.clear();
					for (int u = 0; u <= TS; u += e) { const XMINT2 p = at(u, 0); O.push_back(V(p.x, p.y)); Oa.push_back(u); }
					for (int u = s; u <= TS - s; u += s) { const XMINT2 p = at(u, s); I.push_back(V(p.x, p.y)); Ia.push_back(u); }
					size_t io = 0, ii = 0;
					while (io + 1 < O.size() || ii + 1 < I.size())
					{
						const bool outer = ii + 1 >= I.size() || (io + 1 < O.size() && Oa[io + 1] <= Ia[ii + 1]);
						if (outer) { tri(O[io], O[io + 1], I[ii], expect); ++io; }
						else { tri(O[io], I[ii + 1], I[ii], expect); ++ii; }
					}
				};
				band(std::max(s, TSt(tx, tz - 1)), 1, [&](int u, int d) { return XMINT2(bx + u, bz + d); });       // -z edge
				band(std::max(s, TSt(tx, tz + 1)), -1, [&](int u, int d) { return XMINT2(bx + u, bz + TS - d); }); // +z edge
				band(std::max(s, TSt(tx - 1, tz)), -1, [&](int u, int d) { return XMINT2(bx + d, bz + u); });      // -x edge
				band(std::max(s, TSt(tx + 1, tz)), 1, [&](int u, int d) { return XMINT2(bx + TS - d, bz + u); });  // +x edge
			}
		}
		const size_t vc = vlat.size();
		if (job.cancelled.load())
			return;

		// 3. heights: every vertex and its four normal neighbours, sampled exactly with the chunk generator's own
		//    expression (lerp(bottomLevel, topLevel, height)), so shared border vertices are bit-identical
		wi::vector<uint8_t> need(size_t(NP) * size_t(NP), 0);
		wi::vector<float> h(size_t(NP) * size_t(NP), 0.0f);
		auto P = [&](int i, int j) { return size_t(j + pad) * size_t(NP) + size_t(i + pad); };
		for (const XMINT2& p : vlat)
		{
			need[P(p.x, p.y)] = 1;
			if (nd > 0)
			{
				for (int dz = -nd; dz <= nd; dz += nd)
					for (int dx = -nd; dx <= nd; dx += nd)
						need[P(p.x + dx, p.y + dz)] = 1;   // the 3 x 3 Sobel stencil
				if (L.aak > 0.0f)
				{
					need[P(p.x - ns, p.y)] = 1;            // round 10c: the crease test's +-ns cross
					need[P(p.x + ns, p.y)] = 1;
					need[P(p.x, p.y - ns)] = 1;
					need[P(p.x, p.y + ns)] = 1;
				}
			}
			else
			{
				need[P(p.x - ns, p.y)] = 1;
				need[P(p.x + ns, p.y)] = 1;
				need[P(p.x, p.y - ns)] = 1;
				need[P(p.x, p.y + ns)] = 1;
			}
		}
		// round 9: hold the height-data fence SHARED per 64-row block - GameGuru frees arrays its height function reads
		// (flat areas, the main heightmap) under it EXCLUSIVE, so a level load / terrain reset can never pull them away
		// mid-sample. Released between blocks so a waiting free gets through within one block.
		std::shared_lock<std::shared_mutex> gg_hl(gg_DetailHeightLock(), std::defer_lock);
		for (int pj = 0; pj < NP; ++pj)
		{
			if ((pj & 63) == 0)
			{
				if (gg_hl.owns_lock())
					gg_hl.unlock();
				if (job.cancelled.load())
					return;
				gg_hl.lock();
			}
			const float lz = lc(pj - pad);
			for (int pi = 0; pi < NP; ++pi)
			{
				const size_t k = size_t(pj) * size_t(NP) + size_t(pi);
				if (need[k] == 0)
					continue;
				float height = 0;
				const XMFLOAT2 world_pos = XMFLOAT2(in.position.x + lc(pi - pad), in.position.z + lz);
				for (auto& modifier : in.modifiers)
				{
					modifier->Apply(world_pos, height);
				}
				h[k] = lerp(in.bottomLevel, in.topLevel, height); // EXACTLY the chunk generator's expression
			}
		}
		if (gg_hl.owns_lock())
			gg_hl.unlock();

		// 3b. round 10c: anti-alias sharp creases (gg_detail_crease_aa). Decisions read the point samples only, so the
		//     order never matters; area samples go through the same modifiers + lerp, under the height-data fence,
		//     re-taken every 256 crease vertices so a waiting free gets through. The result moves the vertex POSITION
		//     only (aaY): h - and so every normal - stays the point-sampled height function, which both chunks of a
		//     shared edge sample identically (an area-sampled interior vertex of one chunk is a pad sample of the next).
		uint32_t aa_count = 0;
		wi::vector<float> aaY;       // per vertex: the area-sampled height (valid where aaHas)
		wi::vector<uint8_t> aaHas;
		if (L.aak > 0.0f && ns >= 1)
		{
			const float sw = scale * invF * float(ns);       // vertex spacing at step0 (world)
			const float thr = L.aak * sw;                    // second difference threshold (world)
			static const float tap[5] = { -2.0f / 3.0f, -1.0f / 3.0f, 0.0f, 1.0f / 3.0f, 2.0f / 3.0f };
			static const float tw[5] = { 1.0f / 9.0f, 2.0f / 9.0f, 3.0f / 9.0f, 2.0f / 9.0f, 1.0f / 9.0f };
			aaY.resize(vc, 0.0f);
			aaHas.resize(vc, 0);
			for (size_t v = 0; v < vc; ++v)
			{
				const XMINT2 p = vlat[v];
				if (p.x <= 0 || p.y <= 0 || p.x >= D || p.y >= D)
					continue;                                // chunk edges keep the point sample (see the global)
				const float c = h[P(p.x, p.y)];
				const float cxx = std::abs(h[P(p.x + ns, p.y)] + h[P(p.x - ns, p.y)] - 2.0f * c);
				const float czz = std::abs(h[P(p.x, p.y + ns)] + h[P(p.x, p.y - ns)] - 2.0f * c);
				if (cxx + czz <= thr)
					continue;
				if ((aa_count & 255) == 0)
				{
					if (gg_hl.owns_lock())
						gg_hl.unlock();
					if (job.cancelled.load())
						return;
					gg_hl.lock();
				}
				const float wx0 = in.position.x + lc(p.x), wz0 = in.position.z + lc(p.y);
				float sum = 0;
				for (int b = 0; b < 5; ++b)
				{
					for (int a = 0; a < 5; ++a)
					{
						float s;
						if (a == 2 && b == 2)
							s = c;                               // the point sample itself
						else
						{
							float height = 0;
							const XMFLOAT2 wp = XMFLOAT2(wx0 + tap[a] * sw, wz0 + tap[b] * sw);
							for (auto& modifier : in.modifiers)
								modifier->Apply(wp, height);
							s = lerp(in.bottomLevel, in.topLevel, height);
						}
						sum += tw[a] * tw[b] * s;
					}
				}
				aaY[v] = sum;
				aaHas[v] = 1;
				aa_count++;
			}
			if (gg_hl.owns_lock())
				gg_hl.unlock();
		}
		auto posY = [&](size_t v) { const XMINT2 q = vlat[v]; return (!aaHas.empty() && aaHas[v]) ? aaY[v] : h[P(q.x, q.y)]; };
		gg_detail_aa_verts.store(aa_count);

		// 3c. round 10d: each interior quad on its shorter 3D diagonal (gg_detail_adaptive_diagonal). The replacement
		//     triangles keep the emitters' winding (lattice cross > 0): (LL LR TR) and (LL TR TL).
		uint32_t flipped = 0;
		for (const uint32_t o : quadAt)
		{
			const uint32_t TLv = idx[o], LLv = idx[o + 1], LRv = idx[o + 2], TRv = idx[o + 5];
			auto hv = [&](uint32_t v) { return posY(v); };   // the drawn heights
			const float dMain = std::abs(hv(TLv) - hv(LRv));   // the chunk's diagonal
			const float dAlt = std::abs(hv(LLv) - hv(TRv));
			if (dAlt + 1e-3f * (1.0f + dMain) < dMain)
			{
				idx[o] = LLv; idx[o + 1] = LRv; idx[o + 2] = TRv;
				idx[o + 3] = LLv; idx[o + 4] = TRv; idx[o + 5] = TLv;
				flipped++;
			}
		}
		gg_detail_flipped_quads.store(flipped);

		// 4. vertex attributes
		mesh.SetQuantizedPositionsDisabled(true); // as the chunk mesh: neighbouring AABBs differ
		mesh.SetDoubleSidedShadow(in.doubleSidedShadow);
		mesh.vertex_positions.resize(vc);
		mesh.vertex_normals.resize(vc);
		mesh.vertex_tangents.resize(vc);
		mesh.vertex_uvset_0.resize(vc);
		const int d = nd > 0 ? nd : ns;       // the normal stencil's reach
		const float nstep = scale * invF * float(d);
		const float scale_rcp = 1.0f / scale; // = Terrain::chunk_scale_rcp (private), the same expression
		for (size_t v = 0; v < vc; ++v)
		{
			const XMINT2 p = vlat[v];
			const float x = lc(p.x), z = lc(p.y);
			const float y = posY(v);   // round 10c: the area sample at a crease vertex, else the point sample
			XMFLOAT3 n;
			if (nd > 0)
			{
				// round 10: Sobel - the +-d central difference, averaged 1:2:1 across the other axis = the slope of the
				// surface smoothed over ~2d, so a crease in GG's bilinear height data no longer lands on one vertex row
				const float sx = (h[P(p.x + d, p.y - d)] + 2.0f * h[P(p.x + d, p.y)] + h[P(p.x + d, p.y + d)])
					- (h[P(p.x - d, p.y - d)] + 2.0f * h[P(p.x - d, p.y)] + h[P(p.x - d, p.y + d)]);
				const float sz = (h[P(p.x - d, p.y + d)] + 2.0f * h[P(p.x, p.y + d)] + h[P(p.x + d, p.y + d)])
					- (h[P(p.x - d, p.y - d)] + 2.0f * h[P(p.x, p.y - d)] + h[P(p.x + d, p.y - d)]);
				n = XMFLOAT3(-0.25f * sx, 2.0f * nstep, -0.25f * sz);
			}
			else
			{
				n = XMFLOAT3(h[P(p.x - ns, p.y)] - h[P(p.x + ns, p.y)], 2.0f * nstep, h[P(p.x, p.y - ns)] - h[P(p.x, p.y + ns)]);
			}
			XMStoreFloat3(&n, XMVector3Normalize(XMLoadFloat3(&n)));
			mesh.vertex_positions[v] = XMFLOAT3(x, y, z);
			mesh.vertex_normals[v] = n;
			if (L.tmode != 0)
			{
				// round 10t: the chunk's tangent convention, derived from the normal (see gg_detail_tangent_from_normal).
				// Degenerate only on a vertical wall facing the lattice diagonal (n.y = 0, n.x = n.z), where (1, 0, -1)
				// is itself orthogonal to n.
				float tx = n.y, ty = n.z - n.x, tz = -n.y;
				float len2 = tx * tx + ty * ty + tz * tz;
				if (!(len2 > 1e-8f)) { tx = 1; ty = 0; tz = -1; len2 = 2; }
				const float inv = 1.0f / std::sqrt(len2);
				mesh.vertex_tangents[v] = XMFLOAT4(tx * inv, ty * inv, tz * inv, 1);
			}
			else
			{
				// the chunk's tangent convention (corner[x+1] - corner[z+1], w = 1), at the normal stencil's reach
				mesh.vertex_tangents[v] = XMFLOAT4(nstep, h[P(p.x + d, p.y)] - h[P(p.x, p.y + d)], -nstep, 1);
			}
			mesh.vertex_uvset_0[v] = XMFLOAT2(x * scale_rcp * chunk_width_rcp + 0.5f, z * scale_rcp * chunk_width_rcp + 0.5f);
		}
		mesh.indices = std::move(idx);
		mesh.subsets.emplace_back();
		mesh.subsets.back().materialID = in.material; // the chunk's material (its virtual texture, sampled by chunk UV)
		mesh.subsets.back().indexOffset = 0;
		mesh.subsets.back().indexCount = (uint32_t)mesh.indices.size();
		mesh.subsets_per_lod = 1;
		job.sample_ms = (float)gg_t_build.elapsed_milliseconds();
		if (job.cancelled.load())
			return;
		if (in.external)
		{
			// round 9: an external ring's mesh - the host packs it into its own buffers (wiTerrain_ggdetail.h)
			job.tris = (uint32_t)(mesh.indices.size() / 3);
			job.verts = (uint32_t)vc;
			job.build_ms = (float)gg_t_build.elapsed_milliseconds();
			return;
		}
		wi::Timer gg_t_upload;
		mesh.CreateRenderData(); // own index buffer: every chunk's LOD topology differs
		job.upload_ms = (float)gg_t_upload.elapsed_milliseconds();
		job.tris = (uint32_t)(mesh.indices.size() / 3);
		job.verts = (uint32_t)vc;
		job.build_ms = (float)gg_t_build.elapsed_milliseconds();
	}

	// Per frame, main thread, generator idle (Generation_Update after the chunk loop): retire, swap finished builds
	// in, launch new ones. Never builds on the main thread - a sculpt stroke (a chunk regen per frame) used to cost
	// ~75-100 ms per frame here (round 7 first cut: 11 FPS while sculpting).
	static void gg_UpdateDetailMeshes(Terrain& t, const CameraComponent& camera)
	{
		Scene& scene = *t.scene;
		const bool on = gg_detail_ring >= 0 && gg_detail_factor >= 2;
		const int R = gg_detail_ring;
		const float stride = float(chunk_width - 1) * t.chunk_scale;
		const float hyst = std::max(0.0f, gg_detail_lod_hyst);
		// a new target only once the previous generation has swapped in (or has been stuck ~1.5 s)
		const bool retarget = gg_detail_pending == 0 || gg_detail_pending_frames > 90;

		// the ring centre: the chunk UNDER the camera (chunk c spans c*stride +- stride/2), with hysteresis. (Not
		// t.center_chunk: the stock formula adds chunk_half_width in VERTEX units before the world-unit divide, so at
		// chunk_scale 40 it lags up to half a chunk.)
		const Chunk camc = gg_MakeChunk((int)std::floor(camera.Eye.x / stride + 0.5f), (int)std::floor(camera.Eye.z / stride + 0.5f));
		if (!on || !gg_detail_have_cc)
		{
			gg_detail_cc = camc;
			gg_detail_have_cc = on;
		}
		else if (retarget && !(camc == gg_detail_cc))
		{
			const float reach = 0.5f * stride + hyst;
			if (std::abs(camera.Eye.x - float(gg_detail_cc.x) * stride) > reach || std::abs(camera.Eye.z - float(gg_detail_cc.z) * stride) > reach)
				gg_detail_cc = camc;
		}
		const Chunk cc = gg_detail_cc;
		auto inRing = [&](const Chunk& c) { return on && std::max(std::abs(c.x - cc.x), std::abs(c.z - cc.z)) <= R; };
		// a chunk that exists and draws a chunk-grid mesh (runs after the merge loop: nothing is merge_pending here
		// unless the generator is mid-regen; a sculpt regen keeps the entity and its mesh)
		auto present = [&](const Chunk& c) {
			auto cit = t.chunks.find(c);
			if (cit == t.chunks.end())
				return false;
			const MeshComponent* m = scene.meshes.GetComponent(cit->second.entity);
			return m != nullptr && m->vertex_positions.size() == vertexCount && scene.objects.Contains(cit->second.entity);
		};
		auto isDetail = [&](const Chunk& c) { return inRing(c) && present(c); };

		// the target LOD: the settings, and a centre that follows the camera with hysteresis
		GGDetailLod L;
		L.F = std::min(std::max(gg_detail_factor, 1), 16);
		L.stride = stride;
		L.step0 = gg_DetailValidStep(gg_detail_lod_step0, L.F);
		L.step1 = std::max(L.step0, gg_DetailValidStep(gg_detail_lod_step1, L.F));
		L.nsteps = gg_DetailNormalSteps(L.F, L.step0, L.stride / float(chunk_width - 1));   // round 10
		L.tmode = gg_detail_tangent_from_normal != 0 ? 1 : 0;                              // round 10t
		L.aak = (gg_detail_crease_aa != 0 && gg_detail_crease_k > 0.0f) ? gg_detail_crease_k : 0.0f; // round 10c
		L.adiag = gg_detail_adaptive_diagonal != 0 ? 1 : 0;                                    // round 10d
		L.dist0 = std::max(0.0f, gg_detail_lod_dist0);
		L.dist1 = std::max(L.dist0, gg_detail_lod_dist1);
		if (!on || !gg_detail_have_centre || (retarget && (std::abs(camera.Eye.x - gg_detail_L.cx) > hyst || std::abs(camera.Eye.z - gg_detail_L.cz) > hyst)))
		{
			gg_detail_L.cx = std::round(camera.Eye.x);
			gg_detail_L.cz = std::round(camera.Eye.z);
			gg_detail_have_centre = on;
		}
		L.cx = gg_detail_L.cx;
		L.cz = gg_detail_L.cz;
		gg_detail_L = L;
		gg_detail_centre_x = L.cx;
		gg_detail_centre_z = L.cz;
		gg_detail_eff_step0 = L.step0;
		gg_detail_eff_step1 = L.step1;
		gg_detail_eff_dist0 = L.dist0;
		gg_detail_eff_dist1 = L.dist1;
		gg_detail_threads = gg_DetailPool().Threads();

		// 1. retire at once: records whose chunk is gone or regenerated under a new entity (or detail switched off).
		//    Records that only LEFT the ring keep drawing (the old set is self-consistent) until the next generation swap.
		for (auto it = gg_detail_recs.begin(); it != gg_detail_recs.end();)
		{
			auto cit = t.chunks.find(it->first);
			const bool chunk_ok = cit != t.chunks.end() && cit->second.entity == it->second.chunk_entity;
			const bool fresh = it->second.mesh_entity == INVALID_ENTITY && it->second.job != nullptr; // first build in flight
			const bool leaving = !inRing(it->first);
			if (!on || (!chunk_ok && !fresh) || (leaving && it->second.mesh_entity == INVALID_ENTITY))
			{
				if (it->second.job)
					it->second.job->cancelled.store(true);
				wi::scene::gg_detail_render_mesh.erase(it->second.chunk_entity);
				if (cit != t.chunks.end())
				{
					wi::scene::gg_detail_render_mesh.erase(cit->second.entity); // (a regenerated entity at the same coords)
					if (it->second.mesh_entity != INVALID_ENTITY)
						gg_DetailResetWetmap(scene, cit->second.entity);
				}
				if (it->second.mesh_entity != INVALID_ENTITY)
					scene.Entity_Remove(it->second.mesh_entity);
				it = gg_detail_recs.erase(it); // an in-flight job keeps its own reference; its result is dropped
				continue;
			}
			if (leaving && it->second.job)
			{
				it->second.job->cancelled.store(true); // nothing more to build for it
				it->second.job.reset();
			}
			++it;
		}
		gg_detail_count = 0;
		gg_detail_drawn_tris = 0;
		gg_detail_drawn_verts = 0;
		if (!on)
		{
			wi::scene::gg_detail_render_mesh.clear();
			gg_detail_pending = 0;
			gg_detail_not_current = 0;
			gg_detail_pending_frames = 0;
			return;
		}

		// 2. ring chunks, nearest first (the launch budget serves the camera chunk before its neighbours)
		wi::vector<Chunk> ring;
		for (int d = 0; d <= R; ++d)
			for (int dz = -d; dz <= d; ++dz)
				for (int dx = -d; dx <= d; ++dx)
					if (std::max(std::abs(dx), std::abs(dz)) == d)
						ring.push_back(gg_MakeChunk(cc.x + dx, cc.z + dz));

		wi::Timer timer;
		uint32_t launched = 0, swapped = 0, pending = 0, not_current = 0;
		uint32_t load = gg_DetailPool().Load();
		bool blocked = false;           // a ring chunk needs a new mesh that is not ready yet
		bool same_in_flight = false;    // a same-signature (height) rebuild is still running
		wi::vector<Chunk> ready_same;   // finished rebuilds at an UNCHANGED signature (a sculpt): swap as one batch
		wi::vector<Chunk> ready_new;    // finished rebuilds for the new generation: swap together
		for (const Chunk& c : ring)
		{
			if (!present(c))
				continue; // not generated (yet): nothing drawn there to replace; its neighbours treat it as the grid
			const ChunkData& cd = t.chunks.find(c)->second;
			const MeshComponent* cm = scene.meshes.GetComponent(cd.entity);
			const bool mergeable = !cd.merge_pending;
			uint32_t nmask = 0;
			if (isDetail(gg_MakeChunk(c.x - 1, c.z))) nmask |= 1;
			if (isDetail(gg_MakeChunk(c.x + 1, c.z))) nmask |= 2;
			if (isDetail(gg_MakeChunk(c.x, c.z - 1))) nmask |= 4;
			if (isDetail(gg_MakeChunk(c.x, c.z + 1))) nmask |= 8;
			const uint64_t sig = gg_DetailSig(L, c, nmask, cd.entity);
			GGDetailRec& r = gg_detail_recs[c]; // (may insert: no reference to another record is held across this)
			if (r.job && (r.job->sig != sig || r.job->chunk_entity != cd.entity))
			{
				r.job->cancelled.store(true); // superseded (setting / ring / neighbour moved on): its result is dropped
				r.job.reset();
			}
			const bool have = r.mesh_entity != INVALID_ENTITY && scene.meshes.Contains(r.mesh_entity) && r.chunk_entity == cd.entity;
			const bool current = have && r.sig == sig;
			const bool ready = r.job != nullptr && r.job->done.load();
			if (have && !current)
				not_current++;
			if (!current)
			{
				pending++;
				if (!ready)
					blocked = true;
			}
			else if (r.job && !ready)
			{
				same_in_flight = true;
			}
			if (ready)
				(current ? ready_same : ready_new).push_back(c);
			if ((!current || r.dirty) && r.job == nullptr && mergeable && launched < (uint32_t)std::max(1, gg_detail_budget) && load < 9)
			{
				auto job = std::make_shared<GGDetailJob>();
				job->chunk_entity = cd.entity;
				job->sig = sig;
				job->in.position = cd.position;
				job->in.scale = t.chunk_scale;
				job->in.bottomLevel = t.bottomLevel;
				job->in.topLevel = t.topLevel;
				job->in.modifiers = t.modifiers;
				job->in.L = L;
				job->in.cx = c.x;
				job->in.cz = c.z;
				job->in.nmask = nmask;
				job->in.doubleSidedShadow = cm->IsDoubleSidedShadow();
				job->in.material = cd.entity;
				r.job = job;
				r.dirty = false; // a regen after this point marks it again (merge loop) and relaunches after the swap
				if (!have)
					r.chunk_entity = cd.entity; // the retire loop must not take a first build in flight for a dead chunk
				gg_DetailPool().Push([job] {
					if (!job->cancelled.load())
						gg_BuildDetailMesh(*job);
					job->done.store(true);
				});
				launched++;
				load++;
				if (current)
					same_in_flight = true;
			}
		}
		gg_detail_pending = pending;
		gg_detail_not_current = not_current;
		gg_detail_pending_frames = pending > 0 ? gg_detail_pending_frames + 1 : 0;

		// 3. swaps. A generation in ONE frame once complete; same-signature rebuilds (sculpt) as one batch once none of
		//    them is still running (neighbours dirtied by one stroke swap together)
		float build_max = 0;
		auto swapIn = [&](const Chunk& c) {
			auto rit = gg_detail_recs.find(c);
			if (rit == gg_detail_recs.end() || rit->second.job == nullptr || rit->second.job->cancelled.load())
				return;
			GGDetailRec& r = rit->second;
			GGDetailJob& job = *r.job;
			const bool topology_changed = r.mesh_entity == INVALID_ENTITY || r.sig != job.sig || r.chunk_entity != job.chunk_entity;
			if (r.mesh_entity == INVALID_ENTITY || !scene.meshes.Contains(r.mesh_entity))
			{
				r.mesh_entity = CreateEntity();
				scene.meshes.Create(r.mesh_entity); // (a component pool insert: no ComponentManager reference is held)
			}
			*scene.meshes.GetComponent(r.mesh_entity) = std::move(job.mesh); // the old GPU buffers release deferred
			if (topology_changed)
				gg_DetailResetWetmap(scene, job.chunk_entity);
			r.chunk_entity = job.chunk_entity;
			r.sig = job.sig;
			r.tris = job.tris;
			r.verts = job.verts;
			r.unexpected = job.unexpected;
			gg_detail_tris = job.tris;
			gg_detail_sample_ms = job.sample_ms;
			gg_detail_upload_ms = job.upload_ms;
			gg_detail_build_ms = job.build_ms;
			build_max = std::max(build_max, job.build_ms);
			r.job.reset();
			swapped++;
			gg_detail_builds++;
		};
		if (!same_in_flight)
			for (const Chunk& c : ready_same)
				swapIn(c);
		if (!blocked)
		{
			for (const Chunk& c : ready_new)
				swapIn(c);
			// the new set is complete: records that left the ring stop drawing now
			bool retired = false;
			for (auto it = gg_detail_recs.begin(); it != gg_detail_recs.end();)
			{
				if (!inRing(it->first))
				{
					auto cit = t.chunks.find(it->first);
					if (cit != t.chunks.end())
					{
						wi::scene::gg_detail_render_mesh.erase(cit->second.entity);
						gg_DetailResetWetmap(scene, cit->second.entity);
					}
					wi::scene::gg_detail_render_mesh.erase(it->second.chunk_entity);
					if (it->second.mesh_entity != INVALID_ENTITY)
						scene.Entity_Remove(it->second.mesh_entity);
					it = gg_detail_recs.erase(it);
					retired = true;
					continue;
				}
				++it;
			}
			if (!ready_new.empty() || retired)
				gg_detail_swaps++;
		}

		// 4. what draws: every record with a mesh built for its chunk's current entity
		for (auto& [c, r] : gg_detail_recs)
		{
			auto cit = t.chunks.find(c);
			if (cit == t.chunks.end())
				continue;
			const ChunkData& cd = cit->second;
			const bool have = r.mesh_entity != INVALID_ENTITY && scene.meshes.Contains(r.mesh_entity) && r.chunk_entity == cd.entity;
			auto mit = wi::scene::gg_detail_render_mesh.find(cd.entity);
			if (have)
			{
				if (mit == wi::scene::gg_detail_render_mesh.end() || mit->second != r.mesh_entity)
				{
					gg_DetailResetWetmap(scene, cd.entity); // chunk mesh -> detail mesh
					wi::scene::gg_detail_render_mesh[cd.entity] = r.mesh_entity;
				}
				gg_detail_count++;
				gg_detail_drawn_tris += r.tris;
				gg_detail_drawn_verts += r.verts;
			}
			else if (mit != wi::scene::gg_detail_render_mesh.end())
			{
				gg_DetailResetWetmap(scene, cd.entity); // detail mesh -> chunk mesh
				wi::scene::gg_detail_render_mesh.erase(mit); // not built yet: the chunk mesh draws meanwhile
			}
		}
		if (swapped > 0)
		{
			gg_detail_last_built = swapped;
			gg_detail_last_ms = (float)timer.elapsed_milliseconds(); // main-thread cost of the frame that swapped
			gg_detail_build_ms_max = build_max;
		}
	}

	// border polyline of a DRAWN mesh along its chunk edge e (0:-x 1:+x 2:-z 3:+z): (world along-edge coord, world y),
	// sorted, duplicates removed. Works for the chunk lattice and the detail LOD meshes alike (picked by position).
	static void gg_DetailBorder(const MeshComponent& m, const XMFLOAT3& p, float half, int e, wi::vector<XMFLOAT2>& out)
	{
		out.clear();
		const float target = (e & 1) ? half : -half;
		for (const XMFLOAT3& v : m.vertex_positions)
		{
			const float a = e < 2 ? v.x : v.z;
			if (std::abs(a - target) <= 0.01f)
				out.push_back(XMFLOAT2(e < 2 ? v.z + p.z : v.x + p.x, v.y + p.y));
		}
		std::sort(out.begin(), out.end(), [](const XMFLOAT2& a, const XMFLOAT2& b) { return a.x < b.x; });
		out.erase(std::unique(out.begin(), out.end(), [](const XMFLOAT2& a, const XMFLOAT2& b) { return std::abs(a.x - b.x) < 0.001f; }), out.end());
	}
	// height of a sorted polyline at along-coordinate u (linear between its vertices); false outside it. `exact` is set
	// when a vertex of the polyline sits at u (|du| < 0.01) - a vertex with no such partner is a T-junction.
	static bool gg_DetailPolyAt(const wi::vector<XMFLOAT2>& pl, float u, float& y, bool& exact)
	{
		exact = false;
		if (pl.size() < 2 || u < pl.front().x - 0.01f || u > pl.back().x + 0.01f)
			return false;
		auto it = std::lower_bound(pl.begin(), pl.end(), u, [](const XMFLOAT2& a, float v) { return a.x < v; });
		if (it != pl.end() && std::abs(it->x - u) < 0.01f) { y = it->y; exact = true; return true; }
		if (it != pl.begin() && std::abs((it - 1)->x - u) < 0.01f) { y = (it - 1)->y; exact = true; return true; }
		if (it == pl.begin()) { y = it->y; return true; }
		if (it == pl.end()) { y = pl.back().y; return true; }
		const XMFLOAT2& b = *it;
		const XMFLOAT2& a = *(it - 1);
		const float f = (b.x - a.x) > 0 ? (u - a.x) / (b.x - a.x) : 0.0f;
		y = a.y + (b.y - a.y) * f;
		return true;
	}

	// GGMAX PROTOTYPE (BR-0009 round 7, generalised round 8): the seam instrument (harness TERRAIN_DETAIL_SEAMS). For
	// every chunk drawing a detail mesh and each of its 4 edges, compares its DRAWN border polyline with the
	// neighbour's DRAWN border polyline (detail or chunk mesh), BOTH ways: every vertex of each side must lie on the
	// other side's polyline (max height gap per kind; 0 = watertight) AND have a vertex partner there (a vertex
	// without one is a T-junction - invisible on planar ground, a crack where the surface curves). maxXZ = the
	// largest along-edge extent mismatch (the two polylines must span the same edge). The caller must have stopped
	// the generator (it inserts into t->chunks).
	void gg_DetailSeamTest(Terrain* t, float* maxGapDetail, float* maxGapCoarse, float* maxXZ, int* edgesDetail, int* edgesCoarse, int* tjunctions)
	{
		*maxGapDetail = 0; *maxGapCoarse = 0; *maxXZ = 0; *edgesDetail = 0; *edgesCoarse = 0; *tjunctions = 0;
		if (t == nullptr || t->scene == nullptr)
			return;
		Scene& scene = *t->scene;
		const float half = chunk_half_width * t->chunk_scale;
		auto drawn = [&](const ChunkData& cd) -> const MeshComponent* {
			auto it = wi::scene::gg_detail_render_mesh.find(cd.entity);
			return scene.meshes.GetComponent(it != wi::scene::gg_detail_render_mesh.end() ? it->second : cd.entity);
		};
		wi::vector<XMFLOAT2> A, B;
		for (auto& [c, cd] : t->chunks)
		{
			if (wi::scene::gg_detail_render_mesh.find(cd.entity) == wi::scene::gg_detail_render_mesh.end())
				continue;
			const MeshComponent* m = drawn(cd);
			if (m == nullptr) continue;
			for (int e = 0; e < 4; ++e)
			{
				const int dx = e == 0 ? -1 : (e == 1 ? 1 : 0), dz = e == 2 ? -1 : (e == 3 ? 1 : 0);
				auto nit = t->chunks.find(gg_MakeChunk(c.x + dx, c.z + dz));
				if (nit == t->chunks.end()) continue;
				const MeshComponent* nm = drawn(nit->second);
				if (nm == nullptr) continue;
				const bool nDetail = wi::scene::gg_detail_render_mesh.find(nit->second.entity) != wi::scene::gg_detail_render_mesh.end();
				gg_DetailBorder(*m, cd.position, half, e, A);
				gg_DetailBorder(*nm, nit->second.position, half, e ^ 1, B);
				if (A.size() < 2 || B.size() < 2) continue;
				(nDetail ? *edgesDetail : *edgesCoarse) += 1;
				*maxXZ = std::max(*maxXZ, std::max(std::abs(A.front().x - B.front().x), std::abs(A.back().x - B.back().x)));
				float& g = nDetail ? *maxGapDetail : *maxGapCoarse;
				float y = 0;
				bool exact = false;
				for (const XMFLOAT2& a : A)
					if (gg_DetailPolyAt(B, a.x, y, exact)) { g = std::max(g, std::abs(a.y - y)); if (!exact) (*tjunctions)++; }
				for (const XMFLOAT2& b : B)
					if (gg_DetailPolyAt(A, b.x, y, exact)) { g = std::max(g, std::abs(b.y - y)); if (!exact) (*tjunctions)++; }
			}
		}
	}

	// one mesh's topology (see gg_DetailMeshCheck): open edges not on the chunk border, duplicate directed edges,
	// non-positive triangles in x,z, relative area error against the chunk square
	static void gg_DetailCheckOne(const MeshComponent& m, float half, int& openInside, int& dupEdges, int& badTris, float& areaErr)
	{
		openInside = 0; dupEdges = 0; badTris = 0; areaErr = 0;
		const double full = double(2.0f * half) * double(2.0f * half);
		const auto& P = m.vertex_positions;
		const size_t nt = m.indices.size() / 3;
		wi::unordered_map<uint64_t, int> edges;
		edges.reserve(nt * 3);
		double area = 0;
		for (size_t k = 0; k < nt; ++k)
		{
			const uint32_t i0 = m.indices[k * 3], i1 = m.indices[k * 3 + 1], i2 = m.indices[k * 3 + 2];
			const XMFLOAT3 &a = P[i0], &b = P[i1], &cc = P[i2];
			const double cr = double(b.x - a.x) * double(cc.z - a.z) - double(b.z - a.z) * double(cc.x - a.x);
			if (!(cr > 0)) badTris++;
			area += 0.5 * cr;
			edges[(uint64_t(i0) << 32) | i1]++;
			edges[(uint64_t(i1) << 32) | i2]++;
			edges[(uint64_t(i2) << 32) | i0]++;
		}
		auto onBorder = [&](const XMFLOAT3& p, const XMFLOAT3& q) {
			return (std::abs(p.x - q.x) < 0.01f && std::abs(std::abs(p.x) - half) < 0.01f) ||
				(std::abs(p.z - q.z) < 0.01f && std::abs(std::abs(p.z) - half) < 0.01f);
		};
		for (auto& [key, count] : edges)
		{
			if (count > 1) dupEdges++;
			const uint32_t i0 = uint32_t(key >> 32), i1 = uint32_t(key & 0xffffffffu);
			if (edges.find((uint64_t(i1) << 32) | i0) == edges.end() && !onBorder(P[i0], P[i1]))
				openInside++;
		}
		areaErr = float(std::abs(area - full) / full);
	}

	// GGMAX PROTOTYPE (BR-0009 round 8): watertightness of every DRAWN detail mesh, topologically. Each directed edge
	// must appear once; an edge with no reverse twin is open and must lie on the chunk border (both ends on the same
	// side) - an open edge INSIDE the chunk is a crack (a T-junction or hole between LOD tiles). Every triangle must
	// have positive area in x,z (the chunk's winding), the areas must sum to the chunk's square, and no triangle may
	// have been re-wound against its emitter's natural winding at build time (foldTris: a crossed zipper rung).
	// notCurrent = drawn ring meshes still from the previous generation (a check taken mid-swap says so).
	void gg_DetailMeshCheck(Terrain* t, int* meshes, int* openInside, int* dupEdges, int* badTris, float* areaErr, uint64_t* tris, int* foldTris, int* notCurrent)
	{
		*meshes = 0; *openInside = 0; *dupEdges = 0; *badTris = 0; *areaErr = 0; *tris = 0; *foldTris = 0;
		*notCurrent = (int)gg_detail_not_current;
		if (t == nullptr || t->scene == nullptr)
			return;
		Scene& scene = *t->scene;
		const float half = chunk_half_width * t->chunk_scale;
		const double full = double(2.0f * half) * double(2.0f * half);
		for (auto& [c, cd] : t->chunks)
		{
			auto it = wi::scene::gg_detail_render_mesh.find(cd.entity);
			if (it == wi::scene::gg_detail_render_mesh.end()) continue;
			const MeshComponent* m = scene.meshes.GetComponent(it->second);
			if (m == nullptr) continue;
			(*meshes)++;
			auto rit = gg_detail_recs.find(c);
			if (rit != gg_detail_recs.end())
				*foldTris += (int)rit->second.unexpected;
			*tris += m->indices.size() / 3;
			int oi = 0, de = 0, bt = 0;
			float ae = 0;
			gg_DetailCheckOne(*m, half, oi, de, bt, ae);
			*openInside += oi; *dupEdges += de; *badTris += bt;
			*areaErr = std::max(*areaErr, ae);
		}
		(void)full;
	}

	// =============================================================================================================
	// GGMAX PROTOTYPE (BR-0009 round 9): EXTERNAL DETAIL RING (wiTerrain_ggdetail.h). GameGuru's Terrain Bake tears the
	// live terrain - and the live detail meshes - down and draws its own chunk copies on the 40/80-unit grid (Lee
	// 2026-11-06: a sculpted spike became a wide pyramid). This ring gives those copies the same near-camera detail:
	// the live ring's builder (gg_BuildDetailMesh, in.external), tile/LOD rule (gg_DetailTileStep), signature
	// (gg_DetailSig) and generation-swap rule (a new set publishes only when every ring chunk has its mesh), driven by the
	// same settings globals, with no Terrain and no entities. Ring chunks facing a chunk the host draws coarse use the
	// 40-unit edge (factor steps), exactly the host's full perimeter, so the seams hold by construction. No dirty path:
	// the bake never regenerates in place - an edit marks it stale (Freeze) and it is rebuilt from scratch.
	// =============================================================================================================
	bool gg_detail_ext_check = false;
	struct GGDetailExtJob
	{
		GGDetailJob b;
		std::shared_ptr<GGDetailExtMesh> out;  // set by the build thread before b.done
	};
	struct GGDetailExtRing
	{
		GGDetailExtDesc desc;
		wi::unordered_map<Chunk, XMFLOAT3> origin;   // the host's chunks (present = drawn with its full perimeter)
		struct Rec
		{
			std::shared_ptr<const GGDetailExtMesh> drawn;   // in the published set
			std::shared_ptr<GGDetailExtJob> job;            // a build for the target signature
		};
		wi::unordered_map<Chunk, Rec> recs;          // ska flat map: no reference across an insert
		wi::vector<std::shared_ptr<const GGDetailExtMesh>> published;
		GGDetailLod L;
		bool have_centre = false, have_cc = false;
		Chunk cc;
		uint32_t pending = 0, pending_frames = 0, swaps = 0, builds = 0;
		float last_build_ms = 0, max_build_ms = 0;
		bool frozen = false, failed = false, on = false;
	};

	GGDetailExtRing* gg_DetailExt_Create(GGDetailExtDesc desc)
	{
		GGDetailExtRing* r = new GGDetailExtRing();
		r->desc = std::move(desc);
		for (const GGDetailExtChunk& c : r->desc.chunks)
			r->origin[gg_MakeChunk(c.cx, c.cz)] = c.origin;
		return r;
	}

	bool gg_DetailExt_Update(GGDetailExtRing* r, const XMFLOAT3& eye)
	{
		if (r == nullptr || r->frozen)
			return false;
		bool changed = false;
		r->on = gg_detail_ring >= 0 && gg_detail_factor >= 2 && !r->failed && (bool)r->desc.pack;
		const int R = gg_detail_ring;
		const float stride = float(chunk_width - 1) * r->desc.chunk_scale;
		const float hyst = std::max(0.0f, gg_detail_lod_hyst);
		const bool retarget = r->pending == 0 || r->pending_frames > 90;   // as the live ring

		// ring centre + LOD centre: the live ring's rules (gg_UpdateDetailMeshes)
		const Chunk camc = gg_MakeChunk((int)std::floor(eye.x / stride + 0.5f), (int)std::floor(eye.z / stride + 0.5f));
		if (!r->on || !r->have_cc)
		{
			r->cc = camc;
			r->have_cc = r->on;
		}
		else if (retarget && !(camc == r->cc))
		{
			const float reach = 0.5f * stride + hyst;
			if (std::abs(eye.x - float(r->cc.x) * stride) > reach || std::abs(eye.z - float(r->cc.z) * stride) > reach)
				r->cc = camc;
		}
		const Chunk cc = r->cc;
		auto inRing = [&](const Chunk& c) { return r->on && std::max(std::abs(c.x - cc.x), std::abs(c.z - cc.z)) <= R; };
		auto present = [&](const Chunk& c) { return r->origin.find(c) != r->origin.end(); };
		auto isDetail = [&](const Chunk& c) { return inRing(c) && present(c); };

		GGDetailLod L;
		L.F = std::min(std::max(gg_detail_factor, 1), 16);
		L.stride = stride;
		L.step0 = gg_DetailValidStep(gg_detail_lod_step0, L.F);
		L.step1 = std::max(L.step0, gg_DetailValidStep(gg_detail_lod_step1, L.F));
		L.nsteps = gg_DetailNormalSteps(L.F, L.step0, L.stride / float(chunk_width - 1));   // round 10
		L.tmode = gg_detail_tangent_from_normal != 0 ? 1 : 0;                              // round 10t
		L.aak = (gg_detail_crease_aa != 0 && gg_detail_crease_k > 0.0f) ? gg_detail_crease_k : 0.0f; // round 10c
		L.adiag = gg_detail_adaptive_diagonal != 0 ? 1 : 0;                                    // round 10d
		L.dist0 = std::max(0.0f, gg_detail_lod_dist0);
		L.dist1 = std::max(L.dist0, gg_detail_lod_dist1);
		if (!r->on || !r->have_centre || (retarget && (std::abs(eye.x - r->L.cx) > hyst || std::abs(eye.z - r->L.cz) > hyst)))
		{
			r->L.cx = std::round(eye.x);
			r->L.cz = std::round(eye.z);
			r->have_centre = r->on;
		}
		L.cx = r->L.cx;
		L.cz = r->L.cz;
		r->L = L;

		if (!r->on)
		{
			for (auto& it : r->recs)
				if (it.second.job)
					it.second.job->b.cancelled.store(true);
			r->recs.clear();
			if (!r->published.empty())
			{
				r->published.clear();
				changed = true;
			}
			r->pending = 0;
			r->pending_frames = 0;
			return changed;
		}

		wi::vector<Chunk> ring;
		for (int d = 0; d <= R; ++d)
			for (int dz = -d; dz <= d; ++dz)
				for (int dx = -d; dx <= d; ++dx)
					if (std::max(std::abs(dx), std::abs(dz)) == d)
						ring.push_back(gg_MakeChunk(cc.x + dx, cc.z + dz));

		uint32_t pending = 0, launched = 0;
		uint32_t load = gg_DetailPoolExt().Load();
		bool blocked = false;
		wi::vector<Chunk> ready;
		const float half = chunk_half_width * r->desc.chunk_scale;
		for (const Chunk& c : ring)
		{
			auto oit = r->origin.find(c);
			if (oit == r->origin.end())
				continue;   // the host draws nothing there
			const XMFLOAT3 org = oit->second;
			uint32_t nmask = 0;
			if (isDetail(gg_MakeChunk(c.x - 1, c.z))) nmask |= 1;
			if (isDetail(gg_MakeChunk(c.x + 1, c.z))) nmask |= 2;
			if (isDetail(gg_MakeChunk(c.x, c.z - 1))) nmask |= 4;
			if (isDetail(gg_MakeChunk(c.x, c.z + 1))) nmask |= 8;
			const uint64_t sig = gg_DetailSig(L, c, nmask, 1);   // 1: no chunk entity here
			GGDetailExtRing::Rec& rec = r->recs[c];             // (may insert: no reference to another record is held)
			if (rec.job && rec.job->b.sig != sig)
			{
				rec.job->b.cancelled.store(true);   // superseded: its result is dropped when it finishes
				rec.job.reset();
			}
			const bool current = rec.drawn && rec.drawn->sig == sig;
			const bool isReady = rec.job != nullptr && rec.job->b.done.load();
			if (!current)
			{
				pending++;
				if (!isReady)
					blocked = true;
			}
			if (isReady)
				ready.push_back(c);
			if (!current && rec.job == nullptr && launched < (uint32_t)std::max(1, gg_detail_budget) && load < 9)
			{
				auto job = std::make_shared<GGDetailExtJob>();
				GGDetailJob& b = job->b;
				b.sig = sig;
				b.in.position = org;
				b.in.scale = r->desc.chunk_scale;
				b.in.bottomLevel = r->desc.bottomLevel;
				b.in.topLevel = r->desc.topLevel;
				b.in.modifiers = r->desc.modifiers;
				b.in.L = L;
				b.in.cx = c.x;
				b.in.cz = c.z;
				b.in.nmask = nmask;
				b.in.external = true;
				rec.job = job;
				auto pack = r->desc.pack;   // each job owns a copy: it may outlive the ring
				const bool check = gg_detail_ext_check;
				const int cx = c.x, cz = c.z;
				gg_DetailPoolExt().Push([job, org, half, pack, check, cx, cz] {
					GGDetailJob& jb = job->b;
					if (!jb.cancelled.load())
						gg_BuildDetailMesh(jb);
					if (!jb.cancelled.load() && jb.verts > 0)
					{
						auto m = std::make_shared<GGDetailExtMesh>();
						m->cx = cx;
						m->cz = cz;
						m->sig = jb.sig;
						m->verts = jb.verts;
						m->tris = jb.tris;
						m->unexpected = jb.unexpected;
						m->build_ms = jb.build_ms;
						for (int e = 0; e < 4; ++e)
							gg_DetailBorder(jb.mesh, org, half, e, m->border[e]);
						if (check)
							gg_DetailCheckOne(jb.mesh, half, m->openInside, m->dupEdges, m->badTris, m->areaErr);
						m->gpu = pack(org, jb.mesh);
						job->out = m;
					}
					jb.mesh = MeshComponent();   // the CPU arrays go now, not with the job
					jb.done.store(true);
				});
				launched++;
				load++;
			}
		}
		r->pending = pending;
		r->pending_frames = pending > 0 ? r->pending_frames + 1 : 0;

		// publish a COMPLETE generation only (the live rule): the old set draws until every ring chunk has its new mesh
		if (!blocked)
		{
			bool any = false;
			for (const Chunk& c : ready)
			{
				auto rit = r->recs.find(c);
				if (rit == r->recs.end() || rit->second.job == nullptr)
					continue;
				GGDetailExtRing::Rec& rec = rit->second;
				if (rec.job->out == nullptr || rec.job->out->gpu == nullptr)
				{
					r->failed = true;   // pack() could not create the buffers: the host keeps drawing its own grid
				}
				else
				{
					rec.drawn = rec.job->out;
					r->builds++;
					r->last_build_ms = rec.drawn->build_ms;
					r->max_build_ms = std::max(r->max_build_ms, rec.drawn->build_ms);
				}
				rec.job.reset();
				any = true;
			}
			for (auto it = r->recs.begin(); it != r->recs.end();)
			{
				if (!inRing(it->first))
				{
					if (it->second.job)
						it->second.job->b.cancelled.store(true);
					it = r->recs.erase(it);
					any = true;
					continue;
				}
				++it;
			}
			if (r->failed)
			{
				for (auto& it : r->recs)
					if (it.second.job)
						it.second.job->b.cancelled.store(true);
				r->recs.clear();
				r->published.clear();
				return true;
			}
			if (any)
			{
				r->published.clear();
				for (auto& it : r->recs)
					if (it.second.drawn)
						r->published.push_back(it.second.drawn);
				r->swaps++;
				changed = true;
			}
		}
		return changed;
	}

	const wi::vector<std::shared_ptr<const GGDetailExtMesh>>& gg_DetailExt_Published(const GGDetailExtRing* r)
	{
		static const wi::vector<std::shared_ptr<const GGDetailExtMesh>> none;
		return r != nullptr ? r->published : none;
	}

	void gg_DetailExt_Freeze(GGDetailExtRing* r)
	{
		if (r == nullptr)
			return;
		r->frozen = true;
		for (auto& it : r->recs)
		{
			if (it.second.job)
			{
				it.second.job->b.cancelled.store(true);
				it.second.job.reset();
			}
		}
	}

	void gg_DetailExt_Destroy(GGDetailExtRing* r)
	{
		if (r == nullptr)
			return;
		gg_DetailExt_Freeze(r);
		gg_DetailPoolExt().Drain();   // queued builds dropped, running ones finish (each owns its inputs and pack copy)
		delete r;
	}

	void gg_DetailExt_GetStats(const GGDetailExtRing* r, GGDetailExtStats& s)
	{
		s = GGDetailExtStats();
		s.threads = gg_DetailPoolExt().Threads();
		s.poolLoad = gg_DetailPoolExt().Load();
		if (r == nullptr)
			return;
		s.on = r->on;
		s.frozen = r->frozen;
		s.failed = r->failed;
		s.published = (uint32_t)r->published.size();
		s.pending = r->pending;
		s.swaps = r->swaps;
		s.builds = r->builds;
		for (const auto& m : r->published)
		{
			s.tris += m->tris;
			s.verts += m->verts;
		}
		s.last_build_ms = r->last_build_ms;
		s.max_build_ms = r->max_build_ms;
		s.centre_x = r->L.cx;
		s.centre_z = r->L.cz;
	}

	void gg_DetailPolyCompare(const wi::vector<XMFLOAT2>& A, const wi::vector<XMFLOAT2>& B, float* maxGap, int* tjunctions)
	{
		float y = 0;
		bool exact = false;
		for (const XMFLOAT2& a : A)
			if (gg_DetailPolyAt(B, a.x, y, exact)) { *maxGap = std::max(*maxGap, std::abs(a.y - y)); if (!exact) (*tjunctions)++; }
		for (const XMFLOAT2& b : B)
			if (gg_DetailPolyAt(A, b.x, y, exact)) { *maxGap = std::max(*maxGap, std::abs(b.y - y)); if (!exact) (*tjunctions)++; }
	}

	uint32_t gg_terrain_tile_share_mips = 32; // USER-TUNED 2026-07-28: "human scaled" underfoot detail (~4m repeat)
	// GGMAX 1.53c: hold — delays the halving ladder by this many rungs before it descends from
	// the cap, pushing the FIRST visible scale cross-fade ~1.4x further out per +1 while the
	// feature size stays fixed. 0 = the cap's natural handoff distance.
	// USER-TUNED 2026-07-28 (live walk on the snowy/moss test scene, bracketed 8->6->5->4):
	// 4 = handoff invisible while walking, horizon breaks up before it reads as wallpaper.
	uint32_t gg_terrain_tile_hold_mips = 4;

	void VirtualTexture::init(VirtualTextureAtlas& atlas, uint resolution)
	{
		wi::Timer gg_t_vtinit; // GGMAX 2.61 (includes any nested Residency::init pool miss)
		this->resolution = resolution;
		gg_page_dirty = true; // GGMAX 1.33: fresh mapping — must upload on first job run

		lod_count = 0;
		uint32_t tile_count = 0;
		uint32_t side = resolution;
		while (side >= SVT_TILE_SIZE)
		{
			tile_count += (side / SVT_TILE_SIZE) * (side / SVT_TILE_SIZE);
			side /= 2;
			lod_count++;
		}

		std::scoped_lock lck(locker);
		free(atlas);
		if (tile_count > 1)
		{
			// close by chunks have residency and packed mips
			residency = atlas.allocate_residency(resolution);
			tile_count += 1; // packed mip chain
			lod_count += 1; // packed mip chain
			tiles.resize(tile_count);
			VirtualTexture* gg_stolen = nullptr; // GGMAX 1.33
			if (atlas.allocate_tile(tiles[tile_count - 1], this, &gg_stolen))
			{
				if (gg_stolen) gg_stolen->gg_page_dirty = true; // GGMAX 1.33
				// packed mip tail update:
				UpdateRequest& request = update_requests.emplace_back();
				request.lod = lod_count - 1;
				request.tile_x = tiles[tile_count - 1].x;
				request.tile_y = tiles[tile_count - 1].y;
				request.x = 0;
				request.y = 0;
			}
			if (atlas.allocate_tile(tiles[tile_count - 2], this, &gg_stolen))
			{
				if (gg_stolen) gg_stolen->gg_page_dirty = true; // GGMAX 1.33
				// last nonpacked mip update:
				UpdateRequest& request = update_requests.emplace_back();
				request.lod = lod_count - 2;
				request.tile_x = tiles[tile_count - 2].x;
				request.tile_y = tiles[tile_count - 2].y;
				request.x = 0;
				request.y = 0;
			}
		}
		else
		{
			// far away chunks only have 1 tile
			tiles.resize(tile_count);
			VirtualTexture* gg_stolen = nullptr; // GGMAX 1.33
			if (atlas.allocate_tile(tiles.back(), this, &gg_stolen))
			{
				if (gg_stolen) gg_stolen->gg_page_dirty = true; // GGMAX 1.33
				// update the only tile there is:
				UpdateRequest& request = update_requests.emplace_back();
				request.lod = lod_count - 1;
				request.tile_x = tiles.back().x;
				request.tile_y = tiles.back().y;
				request.x = 0;
				request.y = 0;
			}
		}
		gg_vtprof_vtinit_us.fetch_add(uint64_t(gg_t_vtinit.elapsed_milliseconds() * 1000.0)); // GGMAX 2.61
		gg_vtprof_vtinit_events.fetch_add(1);
	}

	Terrain::Terrain()
	{
		weather.ambient = XMFLOAT3(0.4f, 0.4f, 0.4f);
		weather.SetRealisticSky(true);
		weather.SetRealisticSkyAerialPerspective(true);
		weather.SetVolumetricClouds(true);
		weather.volumetricCloudParameters.layerFirst.extinctionCoefficient = XMFLOAT3(0.71f * 0.05f, 0.86f * 0.05f, 1.0f * 0.05f);
		weather.volumetricCloudParameters.layerFirst.totalNoiseScale = 0.0009f;
		weather.volumetricCloudParameters.layerFirst.curlScale = 0.15f;
		weather.volumetricCloudParameters.layerFirst.detailNoiseModifier = 0.5;
		weather.volumetricCloudParameters.layerFirst.weatherScale = 0.000035f;
		weather.volumetricCloudParameters.layerFirst.coverageMinimum = 0.25f;
		weather.volumetricCloudParameters.layerFirst.typeAmount = 0.0f;
		weather.volumetricCloudParameters.layerFirst.typeMinimum = 0.0f;
		weather.volumetricCloudParameters.layerFirst.rainAmount = 0.9f;
		weather.volumetricCloudParameters.layerFirst.rainMinimum = 0.0f;
		weather.volumetricCloudParameters.layerFirst.gradientSmall = XMFLOAT4(0.01f, 0.07f, 0.08f, 0.14f);
		weather.volumetricCloudParameters.layerFirst.windSpeed = 20.0f;
		weather.volumetricCloudParameters.layerFirst.coverageWindSpeed = 35.0f;
		weather.volumetricCloudParameters.layerSecond.coverageAmount = 0.0f;
		weather.oceanParameters.waterHeight = -10;
		weather.oceanParameters.wave_amplitude = 1000;
		weather.oceanParameters.patch_length = 20;
		weather.fogStart = 0;
		weather.fogDensity = 0.001f;
		weather.SetHeightFog(true);
		weather.fogHeightStart = 0;
		weather.fogHeightEnd = 500;
		weather.windDirection = XMFLOAT3(0.05f, 0.05f, 0.05f);
		weather.windSpeed = 4;
		weather.stars = 0.6f;

		grass_properties.viewDistance = chunk_width;

		generator = std::make_shared<Generator>();

		materialEntities.resize(MATERIAL_COUNT);
	}
	Terrain::~Terrain()
	{
		Generation_Cancel();
		wi::jobsystem::Wait(virtual_texture_ctx);
	}

	void Terrain::Generation_Restart()
	{
		SetGenerationStarted(true);
		Generation_Cancel();
		// GGMAX 1.45: the async VT job may still be running (it holds raw VT pointers via
		// virtual_textures_in_use and allocates/steals from the atlas). Freeing every VT
		// below while it runs is a use-after-free — the pre-existing travel-churn /
		// level-switch corruption (missing chunks, far-field garbage, AMD DEVICE_HUNG,
		// user-repro 2026-07-26). Join it before any teardown.
		wi::jobsystem::Wait(virtual_texture_ctx);
		generator->scene.Clear();

		// save material parameters:
		materials.resize(materialEntities.size());
		for (int i = 0; i < materialEntities.size(); ++i)
		{
			MaterialComponent* material = scene->materials.GetComponent(materialEntities[i]);
			if (material == nullptr)
				continue;
			materials[i] = *material;
			materials[i].SetDirty(false);
		}

		// save grass parameters:
		if (scene->hairs.Contains(grassEntity))
		{
			grass_properties = *scene->hairs.GetComponent(grassEntity);
			grass_properties.SetDirty(false);
		}
		if (scene->materials.Contains(grassEntity))
		{
			grass_material = *scene->materials.GetComponent(grassEntity);
			grass_material.SetDirty(false);
		}

		for (auto it = chunks.begin(); it != chunks.end(); it++)
		{
			ChunkData& chunk_data = it->second;
			if (chunk_data.vt != nullptr)
			{
				chunk_data.vt->free(atlas);
			}
		}
		chunks.clear();
		virtual_textures_in_use.clear();
		// GGMAX PROTOTYPE (BR-0009 round 7): detail builds read the height modifiers (GG's CPU heightmap) on workers. Joined
		// HERE (a restart: level load / terrain reset), NOT in Generation_Cancel - GG cancels the generator many times per
		// sculpt stroke (blend passes) and a join there held every stroke frame for a whole ~100 ms build (11 FPS).
		gg_DetailShutdown(scene); // GGMAX PROTOTYPE (BR-0009 round 7/8): drain the detail build pool, remove every detail mesh

		if (chunkGroupEntity != INVALID_ENTITY)
		{
			scene->Entity_Remove(chunkGroupEntity);
		}

		perlin_noise.init(seed);
		for (auto& modifier : modifiers)
		{
			modifier->Seed(seed);
		}

		// Add some nice weather and lighting if there are no weathers in the scene yet:
		bool created_terrain_weather = false;
		if (scene->weathers.GetCount() == 0)
		{
			if (!scene->weathers.Contains(terrainEntity))
			{
				scene->weathers.Create(terrainEntity);
				created_terrain_weather = true;
			}
		}
		if (created_terrain_weather)
		{
			*scene->weathers.GetComponent(terrainEntity) = weather;
		}
		if (!weather.IsOceanEnabled())
		{
			scene->ocean = {};
		}

		// Restore surface source materials:
		{
			// Clean up invalid material entities that no longer exist in the scene
			for (size_t i = 0; i < materialEntities.size(); )
			{
				if (materialEntities[i] != INVALID_ENTITY && !scene->materials.Contains(materialEntities[i]))
				{
					materialEntities.erase(materialEntities.begin() + i);
					if (i < materials.size())
					{
						materials.erase(materials.begin() + i);
					}
					continue;
				}
				++i;
			}

			// Restore valid materials
			for (size_t i = 0; i < materialEntities.size(); ++i)
			{
				if (materialEntities[i] == INVALID_ENTITY)
				{
					materialEntities[i] = CreateEntity();
				}
				scene->Component_Attach(materialEntities[i], terrainEntity);
				if (!scene->materials.Contains(materialEntities[i]))
				{
					scene->materials.Create(materialEntities[i]);
				}
				*scene->materials.GetComponent(materialEntities[i]) = materials[i];
			}
		}

		// Restore grass parameters:
		if (IsGrassEnabled()) {
			if (grassEntity == INVALID_ENTITY)
			{
				grassEntity = CreateEntity();
			}
			if (!scene->hairs.Contains(grassEntity))
			{
				scene->hairs.Create(grassEntity);
			}
			if (!scene->materials.Contains(grassEntity))
			{
				scene->materials.Create(grassEntity);
			}
			if (!scene->names.Contains(grassEntity))
			{
				scene->names.Create(grassEntity) = "grass";
			}
			*scene->hairs.GetComponent(grassEntity) = grass_properties;
			*scene->materials.GetComponent(grassEntity) = grass_material;
		}

		if (chunkGroupEntity == INVALID_ENTITY)
		{
			chunkGroupEntity = CreateEntity();
		}
		scene->Component_Attach(chunkGroupEntity, terrainEntity);
		if (!scene->names.Contains(chunkGroupEntity))
		{
			scene->names.Create(chunkGroupEntity) = "chunks";
		}
	}

	void Terrain::Generation_Update(const CameraComponent& camera)
	{
		if (wi::jobsystem::IsBusy(generator->workload))
			return; // updating can't run while generation is running. Note: we could cancel here, but it could take long until cancel request is fulfilled (happened with physics mesh creations)
		wi::profiler::gg_trace_mark("tg-begin"); // GGMAX 2.61 gap decomposition

		bool restart_generation = false;
		if (!IsGenerationStarted())
		{
			restart_generation = true;
		}

		// Check whether any modifiers need to be removed, and we will really remove them here if so:
		if (!modifiers_to_remove.empty())
		{
			for (auto& modifier : modifiers_to_remove)
			{
				for (auto it = modifiers.begin(); it != modifiers.end(); ++it)
				{
					if (it->get() == modifier)
					{
						modifiers.erase(it);
						break;
					}
				}
			}
			restart_generation = true;
			modifiers_to_remove.clear();
		}
		if (generation_restart_on_dirty_materials) // GGMAX: GG registers materials at runtime and owns the blendmaps — no rebuild on dirt
		{
			for (wi::ecs::Entity entity : materialEntities)
			{
				MaterialComponent* material = scene->materials.GetComponent(entity);
				if (material == nullptr)
					continue;
				if (material->IsDirty())
				{
					restart_generation = true;
					break;
				}
			}
		}
		splineMaterialEntities.clear();
		for (size_t i = 0; i < scene->splines.GetCount(); ++i)
		{
			const SplineComponent& spline = scene->splines[i];
			if (spline.terrain_modifier_amount > 0 || spline.prev_terrain_modifier_amount > 0)
			{
				bool spline_terrain_invalidation = false;
				if (spline.dirty_terrain)
				{
					spline.dirty_terrain = false;
					spline_terrain_invalidation = true;
					spline.prev_terrain_modifier_amount = spline.terrain_modifier_amount;
					spline.prev_terrain_pushdown = spline.terrain_pushdown;
					spline.prev_terrain_texture_falloff = spline.terrain_texture_falloff;
					spline.prev_terrain_generation_nodes = (int)spline.spline_node_entities.size();
				}
				spline_terrain_invalidation |= spline.materialEntity != spline.materialEntity_terrainPrev;
				const MaterialComponent* splineMaterial = scene->materials.GetComponent(spline.materialEntity);
				if (splineMaterial != nullptr)
				{
					spline_terrain_invalidation |= splineMaterial->IsDirty();
					splineMaterialEntities.push_back(spline.materialEntity);
				}
				spline.materialEntity_terrainPrev = spline.materialEntity;

				if (spline_terrain_invalidation)
				{
					InvalidateChunksAtSpline(spline);
				}
			}
		}

		if (restart_generation)
		{
			Generation_Restart();
		}

		if (terrainEntity == INVALID_ENTITY)
		{
			wi::jobsystem::Wait(virtual_texture_ctx); // GGMAX 1.45: same use-after-free race as Generation_Restart
			for (auto it = chunks.begin(); it != chunks.end(); it++)
			{
				ChunkData& chunk_data = it->second;
				if (chunk_data.vt != nullptr)
				{
					chunk_data.vt->free(atlas);
				}
			}
			chunks.clear();
			virtual_textures_in_use.clear();
			gg_DetailShutdown(scene); // GGMAX PROTOTYPE (BR-0009 round 7/8): see Generation_Restart
			return;
		}

		if (chunkGroupEntity == INVALID_ENTITY)
		{
			chunkGroupEntity = terrainEntity;
		}

		WeatherComponent* weather_component = scene->weathers.GetComponent(terrainEntity);
		if (weather_component != nullptr)
		{
			weather = *weather_component; // feedback default weather
		}

		// Invalidated chunks replacements, originals are removed before merging updated ones:
		for (Chunk chunk : generator->removable_chunks)
		{
			auto it = chunks.find(chunk);
			if (it != chunks.end())
			{
				ChunkData& chunk_data = it->second;
				scene->Entity_Remove(chunk_data.entity);
				chunk_data.props_entity = INVALID_ENTITY;
				if (chunk_data.vt != nullptr)
				{
					// GGMAX: when preserving blendmaps across in-place regen, keep the VT residency
					// too — only the mesh changed. The freshly merged MaterialComponent is re-bound
					// to the existing vt in UpdateVirtualTexturesCPU (no reset, nothing re-streams).
					if (!gg_preserve_blendmap_on_regen)
						chunk_data.vt->invalidate();
				}
			}
		}
		generator->removable_chunks.clear();

		// What was generated, will be merged in to the main scene
		scene->MergeFastInternal(generator->scene);
		wi::profiler::gg_trace_mark("tg-merge"); // GGMAX 2.61 gap decomposition (fires per caller: S1 + bridge)

		// GGMAX: everything the (now idle) generator produced is live in the main scene —
		// lift the merge_pending flags so mesh-baking consumers may process these chunks
		for (auto it = chunks.begin(); it != chunks.end(); it++)
		{
			// GGMAX 3.159 (BR-0009 FPS, review A1): a chunk merged THIS frame gets the game's LOD bias before the LOD
			// is computed (RunObjectUpdateSystem runs after this), not one frame late at terrain.lod_bias - no pop.
			if (it->second.merge_pending && gg_chunk_lod_bias_fn != nullptr)
			{
				ObjectComponent* gg_o = scene->objects.GetComponent(it->second.entity);
				const MeshComponent* gg_m = scene->meshes.GetComponent(it->second.entity);
				if (gg_o != nullptr && gg_m != nullptr)
				{
					const XMFLOAT3& p = it->second.position;
					const XMFLOAT3 mn(gg_m->aabb._min.x + p.x, gg_m->aabb._min.y + p.y, gg_m->aabb._min.z + p.z);
					const XMFLOAT3 mx(gg_m->aabb._max.x + p.x, gg_m->aabb._max.y + p.y, gg_m->aabb._max.z + p.z);
					gg_o->lod_bias = gg_chunk_lod_bias_fn(mn, mx);
				}
			}
			// GGMAX PROTOTYPE (BR-0009 round 7): a regenerated chunk (sculpt) needs its detail mesh rebuilt
			if (it->second.merge_pending)
			{
				auto gg_dr = gg_detail_recs.find(it->first);
				if (gg_dr != gg_detail_recs.end())
					gg_dr->second.dirty = true;
			}
			it->second.merge_pending = false;
		}

		chunk_scale_rcp = 1.0f / chunk_scale;

		if (IsCenterToCamEnabled())
		{
			// GGMAX 2.53: the Terrain Generator pins the ring to the editable-area marker;
			// everything downstream (creation ring, removal threshold, priority sorts) keys
			// off center_chunk, so overriding here covers the whole generation system.
			float gg_cx = camera.Eye.x;
			float gg_cz = camera.Eye.z;
			if (gg_generation_center_override_enabled)
			{
				gg_cx = gg_generation_center_override_x;
				gg_cz = gg_generation_center_override_z;
			}
			center_chunk.x = (int)std::floor((gg_cx + chunk_half_width) * chunk_width_rcp * chunk_scale_rcp);
			center_chunk.z = (int)std::floor((gg_cz + chunk_half_width) * chunk_width_rcp * chunk_scale_rcp);
		}

		// GGMAX: track how long the camera has stayed within one chunk — VT residency
		// upgrades are deferred while it is crossing boundaries (gg_vt_upgrade_hysteresis)
		if (center_chunk == gg_prev_center_chunk)
		{
			gg_center_stable_frames++;
		}
		else
		{
			gg_center_stable_frames = 0;
			gg_prev_center_chunk = center_chunk;
		}

		const int removal_threshold = generation + 2 + gg_removal_margin; // GGMAX: bounded islands keep chunks alive across zoom travel
		GraphicsDevice* device = GetDevice();

		// Check whether there are any materials that would write to virtual textures:
		bool virtual_texture_any = false;

		if (scene->materials.GetCount() > 0)
		{
			for (wi::ecs::Entity entity : materialEntities)
			{
				MaterialComponent* material = scene->materials.GetComponent(entity);
				if (material == nullptr)
					continue;

				for (int i = 0; i < TEXTURESLOT_COUNT && !virtual_texture_any; ++i)
				{
					switch (i)
					{
					case MaterialComponent::BASECOLORMAP:
					case MaterialComponent::NORMALMAP:
					case MaterialComponent::SURFACEMAP:
					case MaterialComponent::EMISSIVEMAP:
						if (material->textures[i].resource.IsValid())
						{
							virtual_texture_any = true;
						}
						break;
					default:
						break;
					}
				}

				if (virtual_texture_any)
					break;
			}

			if (grassEntity != INVALID_ENTITY)
			{
				MaterialComponent* material_grassparticle_in_scene = scene->materials.GetComponent(grassEntity);
				if (material_grassparticle_in_scene != nullptr)
				{
					grass_material = *material_grassparticle_in_scene;
				}
				HairParticleSystem* hair = scene->hairs.GetComponent(grassEntity);
				if (hair != nullptr)
				{
					grass_properties = *hair;
				}
			}
		}

		// Ensure that enough grass chunks are generated so that grass view distance will not cause popping:
		grass_chunk_dist = int(grass_properties.viewDistance / chunk_width + 0.5f);

		// GGMAX 1.45: one-shot flag — the removal branch below joins the async VT job at
		// most once per Generation_Update call, and only on frames that actually tear a
		// chunk down (frames without removals keep the full VT-job overlap).
		bool gg_vt_teardown_joined = false;

		for (auto it = chunks.begin(); it != chunks.end();)
		{
			const Chunk& chunk = it->first;
			ChunkData& chunk_data = it->second;
			chunk_data.visible = camera.frustum.CheckSphere(chunk_data.sphere.center, chunk_data.sphere.radius);
			const int dist = std::max(std::abs(center_chunk.x - chunk.x), std::abs(center_chunk.z - chunk.z));

			if (wi::renderer::GetOcclusionCullingEnabled())
			{
				size_t object_index = scene->objects.GetIndex(chunk_data.entity);
				if (object_index < scene->occlusion_results_objects.size())
				{
					if (scene->occlusion_results_objects[object_index].IsOccluded())
					{
						chunk_data.visible = false;
					}
				}
			}

			// pointer refresh:
			MeshComponent* chunk_mesh = scene->meshes.GetComponent(chunk_data.entity);
			if (chunk_mesh != nullptr)
			{
				chunk_data.mesh_vertex_positions = chunk_mesh->vertex_positions.data();

				if (IsTessellationEnabled())
				{
					chunk_mesh->tessellationFactor = 32;
				}
				else
				{
					chunk_mesh->tessellationFactor = 0;
				}
				// GGMAX 2.61: this is a HEAL — any chunk born without a BVH gets a SYNCHRONOUS
				// main-thread build here. It silently defeated the 2.58 generator skip: the 8.2
				// ms/chunk build didn't disappear, it moved from the generation thread (parallel)
				// to THIS loop (serialized into frames — the 400-550ms tg-chunkloop gap frames,
				// ~6s of a generator fill). While the generator owns the screen, chunks are MEANT
				// to be BVH-less (drag picks brute-force; the exit wipe regenerates WITH BVHs), so
				// the heal honors the same flag. Every other mode: unchanged.
				// GGMAX 3.159 (BR-0009 m2): with gg_terrain_pick_bvh off (GG) chunks never get a BVH - the
				// heal MUST honour that too, or it moves 36 ms/chunk (at 128+3) onto the main thread.
				if (gg_terrain_pick_bvh && !gg_generation_skip_bvh && !chunk_mesh->bvh.IsValid())
				{
					chunk_mesh->SetBVHEnabled(true);
				}
			}
			else
			{
				chunk_data.mesh_vertex_positions = nullptr;
			}

			ObjectComponent* chunk_object = scene->objects.GetComponent(chunk_data.entity);
			if (chunk_object != nullptr)
			{
				chunk_object->SetWetmapEnabled(scene->IsWetmapProcessingRequired());
				chunk_object->emissiveColor.w = virtual_texture_any ? 1.0f : 0.0f; // without virtual textures, emissive will be disabled on the object-level (most likely missing content)
			}

			// chunk removal:
			if (IsRemovalEnabled())
			{
				if (dist > removal_threshold)
				{
					// GGMAX 1.45: the async VT job holds raw pointers to this VT (last frame's
					// virtual_textures_in_use) and mutates the shared atlas. free()+erase under
					// a live job = use-after-free — the dominant teardown site during sustained
					// camera travel (a chunk falls behind removal_threshold every few frames).
					// Join once per Generation_Update call, only on frames that actually remove.
					if (!gg_vt_teardown_joined)
					{
						wi::jobsystem::Wait(virtual_texture_ctx);
						gg_vt_teardown_joined = true;
					}
					if (chunk_data.vt != nullptr)
					{
						chunk_data.vt->free(atlas);
					}
					scene->Entity_Remove(it->second.entity);
					it = chunks.erase(it);
					continue; // don't increment iterator
				}
				else
				{
					// Grass patch removal:
					if (chunk_data.grass_entity != INVALID_ENTITY && (dist > grass_chunk_dist || !IsGrassEnabled()))
					{
						scene->Entity_Remove(chunk_data.grass_entity);
						chunk_data.grass_entity = INVALID_ENTITY; // grass can be generated here by generation thread...
					}

					// Prop removal:
					if (chunk_data.props_entity != INVALID_ENTITY && (dist > prop_generation || std::abs(chunk_data.prop_density_current - prop_density) > std::numeric_limits<float>::epsilon()))
					{
						scene->Entity_Remove(chunk_data.props_entity);
						chunk_data.props_entity = INVALID_ENTITY; // prop can be generated here by generation thread...
					}
				}
			}

			// Grass property update:
			if (chunk_data.visible && chunk_data.grass_entity != INVALID_ENTITY)
			{
				wi::HairParticleSystem* grass = scene->hairs.GetComponent(chunk_data.grass_entity);
				if (grass != nullptr)
				{
					chunk_data.grass_density_current = grass_density;
					grass->strandCount = uint32_t(chunk_data.grass.strandCount * chunk_data.grass_density_current);
					grass->length = grass_properties.length;
					grass->randomness = grass_properties.randomness;
					grass->randomSeed = grass_properties.randomSeed;
					grass->stiffness = grass_properties.stiffness;
					grass->viewDistance = grass_properties.viewDistance;
					grass->width = grass_properties.width;
					grass->uniformity = grass_properties.uniformity;
					grass->atlas_rects = grass_properties.atlas_rects;
					grass->segmentCount = grass_properties.segmentCount;
					grass->billboardCount = grass_properties.billboardCount;
					grass->drag = grass_properties.drag;
					grass->gravityPower = grass_properties.gravityPower;
				}

				MaterialComponent* chunkGrassMaterial = scene->materials.GetComponent(chunk_data.grass_entity);
				if (chunkGrassMaterial != nullptr)
				{
					*chunkGrassMaterial = grass_material;
				}
			}

			RigidBodyPhysicsComponent* rigidbody = scene->rigidbodies.GetComponent(chunk_data.entity);
			if (IsPhysicsEnabled())
			{
				const ObjectComponent* object = scene->objects.GetComponent(chunk_data.entity);

				if (dist < physics_generation)
				{
					if (rigidbody == nullptr)
					{
						RigidBodyPhysicsComponent& newrigidbody = scene->rigidbodies.Create(chunk_data.entity);
						newrigidbody.shape = RigidBodyPhysicsComponent::HEIGHTFIELD;
						newrigidbody.mass = 0; // terrain chunks are static
						newrigidbody.friction = 0.8f;
						//newrigidbody.mesh_lod = 2;
					}
					else
					{
						rigidbody->shape = RigidBodyPhysicsComponent::HEIGHTFIELD;
					}
				}
				else if(rigidbody != nullptr)
				{
					scene->rigidbodies.Remove(chunk_data.entity);
				}
			}
			else
			{
				if (rigidbody != nullptr)
				{
					scene->rigidbodies.Remove(chunk_data.entity);
				}
			}

			it++;
		}
		wi::profiler::gg_trace_mark("tg-chunkloop"); // GGMAX 2.61

		gg_UpdateDetailMeshes(*this, camera); // GGMAX PROTOTYPE (BR-0009 round 7): near-camera detail meshes

		if (virtual_texture_any)
		{
			UpdateVirtualTexturesCPU();
		}
		else
		{
			virtual_textures_in_use.clear();
		}
		wi::profiler::gg_trace_mark("tg-vtcpu"); // GGMAX 2.61

		const uint64_t required_chunk_buffer_size = sizeof(ShaderTerrainChunk) * (chunk_buffer_range * 2 + 1) * (chunk_buffer_range * 2 + 1);
		if (chunk_buffer.desc.size < required_chunk_buffer_size)
		{
			GPUBufferDesc desc;
			desc.usage = Usage::DEFAULT;
			desc.size = required_chunk_buffer_size;
			desc.bind_flags = BindFlag::SHADER_RESOURCE;
			desc.misc_flags = ResourceMiscFlag::BUFFER_STRUCTURED;
			desc.stride = sizeof(ShaderTerrainChunk);
			bool success = device->CreateBuffer(&desc, nullptr, &chunk_buffer);
			assert(success);
			device->SetName(&chunk_buffer, "wi::terrain::Terrain::chunk_buffer");
		}

		// Start the generation on a background thread and keep it running until the next frame
		generator->cancelled.store(false);
		// GGMAX: optionally promote generation to the High pool — the Low pool runs at
		// THREAD_PRIORITY_LOWEST and starves while the CPU is busy (level load).
		generator->workload.priority = generation_high_priority ? wi::jobsystem::Priority::High : wi::jobsystem::Priority::Low;
		size_t terrain_spline_count_scene = 0; // Count the current terrain modifier splines in scene
		for (size_t i = 0; i < scene->splines.GetCount(); ++i)
		{
			const SplineComponent& spline = scene->splines[i];
			if (spline.terrain_modifier_amount > 0)
			{
				terrain_spline_count_scene++;
			}
		}
		if (terrain_spline_count_scene != generator->splines.size())
		{
			// Need to invalidate terrain if spline was removed:
			for (const SplineComponent& spline : generator->splines)
			{
				InvalidateChunksAtSpline(spline);
			}
		}
		generator->splines.clear();
		for (size_t i = 0; i < scene->splines.GetCount(); ++i)
		{
			const SplineComponent& spline = scene->splines[i];
			if (spline.terrain_modifier_amount > 0)
			{
				generator->splines.push_back(spline);
			}
		}
		// GGMAX: capture the camera's horizontal look direction for view-cone-first
		// generation. Zero vector = disabled (also when looking straight down).
		XMFLOAT2 view_cone_forward = XMFLOAT2(0, 0);
		if (generation_view_cone_priority)
		{
			const float fx = camera.At.x;
			const float fz = camera.At.z;
			const float len = std::sqrt(fx * fx + fz * fz);
			if (len > 0.001f)
			{
				view_cone_forward = XMFLOAT2(fx / len, fz / len);
			}
		}

		wi::jobsystem::Execute(generator->workload, [=](wi::jobsystem::JobArgs a) {

			wi::Timer timer;
			bool generated_something = false;

			auto request_chunk = [&](int offset_x, int offset_z)
			{
				Chunk chunk = center_chunk;
				chunk.x += offset_x;
				chunk.z += offset_z;
				auto it = chunks.find(chunk);
				// GGMAX: remember whether this visit actually (re)generates — the spiral
				// visits every existing chunk every job run, and only generated chunks
				// may be flagged merge_pending below (flagging every visited chunk made
				// GG's blendmap passes skip the whole map forever)
				const bool gg_chunk_generated = (it == chunks.end() || it->second.entity == INVALID_ENTITY || it->second.invalidated);
				// GGMAX: in-place regen (sculpt/edit invalidation) of an existing chunk may keep its
				// blendmap — see gg_preserve_blendmap_on_regen. Computed from `it` BEFORE chunks[chunk]
				// below can insert/rehash. Only kicks in when the old chunk has usable layers.
				const bool gg_preserve_blend = gg_preserve_blendmap_on_regen &&
					it != chunks.end() && it->second.entity != INVALID_ENTITY && it->second.invalidated &&
					it->second.blendmap_layers.size() >= 4 &&
					it->second.blendmap_layers[0].pixels.size() == vertexCount;
				if (it == chunks.end() || it->second.entity == INVALID_ENTITY || it->second.invalidated)
				{
					wi::Timer gg_t_total; // GGMAX 2.58: per-chunk generation cost breakdown
					// Generate a new chunk:
					ChunkData& chunk_data = chunks[chunk];

					std::string chunk_name = "chunk_" + std::to_string(chunk.x) + "_" + std::to_string(chunk.z);
					if (chunk_data.entity == INVALID_ENTITY)
					{
						chunk_data.entity = generator->scene.Entity_CreateObject(chunk_name);
					}
					else
					{
						// replacement will be made instead of simple merge, entity ID can be reused:
						generator->scene.names.Create(chunk_data.entity) = std::move(chunk_name);
						generator->scene.layers.Create(chunk_data.entity);
						generator->scene.transforms.Create(chunk_data.entity);
						generator->scene.objects.Create(chunk_data.entity);
						generator->removable_chunks.push_back(chunk);
					}
					ObjectComponent& object = *generator->scene.objects.GetComponent(chunk_data.entity);
					object.lod_bias = lod_bias;
					object.filterMask |= wi::enums::FILTER_NAVIGATION_MESH;
					object.filterMask |= wi::enums::FILTER_TERRAIN;
					generator->scene.Component_Attach(chunk_data.entity, chunkGroupEntity);

					TransformComponent& transform = *generator->scene.transforms.GetComponent(chunk_data.entity);
					transform.ClearTransform();
					chunk_data.position = XMFLOAT3(float(chunk.x * (chunk_width - 1)) * chunk_scale, 0, float(chunk.z * (chunk_width - 1)) * chunk_scale);
					transform.Translate(chunk_data.position);
					transform.UpdateTransform();

					MaterialComponent& material = generator->scene.materials.Create(chunk_data.entity);
					// material params will be 1 because they will be created from only texture maps
					//	because region materials are blended together into one texture
					material.SetInternal();
					material.SetRoughness(1);
					material.SetMetalness(1);
					material.SetReflectance(1);
					material.SetEmissiveStrength(100);

					MeshComponent& mesh = generator->scene.meshes.Create(chunk_data.entity);
					mesh.SetQuantizedPositionsDisabled(true); // connecting meshes quantization is not correct because mismatching AABBs
					object.meshID = chunk_data.entity;
					mesh.indices = chunk_indices().indices;
					for (auto& lod : chunk_indices().lods)
					{
						mesh.subsets.emplace_back();
						mesh.subsets.back().materialID = chunk_data.entity;
						mesh.subsets.back().indexCount = lod.indexCount;
						mesh.subsets.back().indexOffset = lod.indexOffset;
					}
					mesh.subsets_per_lod = 1;
					mesh.vertex_positions.resize(vertexCount);
					mesh.vertex_normals.resize(vertexCount);
					mesh.vertex_tangents.resize(vertexCount);
					mesh.vertex_uvset_0.resize(vertexCount);

					if (!gg_preserve_blend) // GGMAX: keep GG's multi-layer blendmap across in-place regen
					{
						chunk_data.blendmap_layers.resize(4);
						for (auto& x : chunk_data.blendmap_layers)
						{
							x.pixels.resize(vertexCount);
						}
					}

					chunk_data.spline_blendmap_layers.resize(splineMaterialEntities.size());
					for (auto& x : chunk_data.spline_blendmap_layers)
					{
						x.pixels.resize(vertexCount);
					}

					chunk_data.mesh_vertex_positions = mesh.vertex_positions.data();

					chunk_data.heightmap_data.resize(vertexCount);

					// GGMAX 3.159 (BR-0009 round 2, m3): seed the engine's per-chunk grass ONLY when engine grass is
					// enabled - its one consumer is the IsGrassEnabled()-gated placement below. GG disables engine grass
					// (GGTerrainWicked.cpp SetGrassEnabled(false)) and builds its own hair systems from the chunk mesh, so
					// the seed (perlin per vertex, vertex_lengths, a CPU copy of the LOD0 list via CreateFromMesh - up to
					// ~0.6 MB/chunk at 128+3) was pure waste in every GG mode except the generator (2.60 skip). One
					// snapshot gates the resize, the per-vertex store and the CreateFromMesh, so they can never disagree.
					const bool gg_seed_grass = IsGrassEnabled() && !gg_generation_skip_grass;
					wi::HairParticleSystem grass;
					if (gg_seed_grass)
					{
						grass = grass_properties;
						grass.vertex_lengths.resize(vertexCount);
					}
					std::atomic<uint32_t> grass_valid_vertex_count{ 0 };

					// Shadow casting will only be enabled for sloped terrain chunks:
					std::atomic_bool slope_cast_shadow;
					slope_cast_shadow.store(false);

					// Do a parallel for loop over all the chunk's vertices and compute their properties:
					wi::jobsystem::context ctx;
					ctx.priority = generation_high_priority ? wi::jobsystem::Priority::High : wi::jobsystem::Priority::Low; // GGMAX


					// Preload height grid with padding, because neighbors will need to be accessed to determine slopes:
					constexpr int chunk_width_padded = chunk_width + 1;
					constexpr uint32_t vertexCount_padded = chunk_width_padded * chunk_width_padded;
					float heights_padded[chunk_width_padded][chunk_width_padded];
					const XMVECTOR UP = XMVectorSet(0, 1, 0, 0);
					wi::Timer gg_t_heights; // GGMAX 2.58
					wi::jobsystem::Dispatch(ctx, vertexCount_padded, chunk_width_padded * 4, [&](wi::jobsystem::JobArgs args) {
						const uint32_t index = args.jobIndex;
						const XMUINT2 coord = XMUINT2(index % chunk_width_padded, index / chunk_width_padded);
						const float x = (float(coord.x) - chunk_half_width) * chunk_scale;
						const float z = (float(coord.y) - chunk_half_width) * chunk_scale;

						float height = 0;
						const XMFLOAT2 world_pos = XMFLOAT2(chunk_data.position.x + x, chunk_data.position.z + z);
						for (auto& modifier : modifiers)
						{
							modifier->Apply(world_pos, height);
						}
						height = lerp(bottomLevel, topLevel, height);

						const bool is_real_vertex = coord.x < chunk_width && coord.y < chunk_width;
						const uint32_t real_index = coord.x + coord.y * chunk_width;

						// Apply splines to height only:
						const XMVECTOR P = XMVectorSet(world_pos.x, -100000, world_pos.y, 0);
						const wi::primitive::Ray ray(P, UP);
						int splinematerialcnt = -1;
						for (size_t j = 0; j < generator->splines.size(); ++j)
						{
							const SplineComponent& spline = generator->splines[j];
							if (spline.materialEntity != INVALID_ENTITY)
								splinematerialcnt++;
							if (!spline.bvh.IntersectsFirst(ray, [&](uint32_t index) { return spline.precomputed_aabbs[index].intersects(ray); }))
								continue;
							XMVECTOR S = spline.TraceSplinePlane(P, UP, 4);
							S = spline.ClosestPointOnSpline(S, 4);
							const float splineheight = XMVectorGetY(S);
							const float splinedist = wi::math::Distance(XMVectorSetY(P, splineheight), S);
							const float splinefactor = 1.0f - smoothstep(0.0f, 1.0f, saturate(splinedist * sqr(spline.terrain_modifier_amount)));
							if (is_real_vertex && spline.materialEntity != INVALID_ENTITY)
							{
								chunk_data.spline_blendmap_layers[splinematerialcnt].pixels[real_index] = uint8_t(smoothstep(clamp(spline.terrain_texture_falloff, 0.0f, 0.999f), 1.0f, splinefactor) * 255);
							}
							height = lerp(height, splineheight - spline.terrain_pushdown, splinefactor);
						}

						heights_padded[coord.x][coord.y] = height;
					});
					wi::jobsystem::Wait(ctx);
					gg_genprof_heights_us.fetch_add(uint64_t(gg_t_heights.elapsed_milliseconds() * 1000.0)); // GGMAX 2.58

					wi::Timer gg_t_vertex; // GGMAX 2.58
					wi::jobsystem::Dispatch(ctx, vertexCount, chunk_width * 4, [&](wi::jobsystem::JobArgs args) {
						ChunkData& chunk_data = chunks[chunk];
						const uint32_t index = args.jobIndex;
						const XMUINT2 coord = XMUINT2(index % chunk_width, index / chunk_width);
						const float x = (float(coord.x) - chunk_half_width) * chunk_scale;
						const float z = (float(coord.y) - chunk_half_width) * chunk_scale;
						const float height = heights_padded[coord.x][coord.y];
						// Fix: horizontal steps must match the real world spacing
						// between neighbouring vertices (chunk_scale), otherwise
						// the cross product sees the height delta amplified by
						// chunk_scale and the resulting normal lies near
						// horizontal. With chunk_scale = 1 (Wicked default) this
						// reduces to the original "+ 1" behaviour.
						const XMVECTOR corners[3] = {
							XMVectorSet(chunk_data.position.x + x, height, chunk_data.position.z + z, 0),
							XMVectorSet(chunk_data.position.x + x + chunk_scale, heights_padded[coord.x + 1][coord.y], chunk_data.position.z + z, 0),
							XMVectorSet(chunk_data.position.x + x, heights_padded[coord.x][coord.y + 1], chunk_data.position.z + z + chunk_scale, 0),
						};
						const XMVECTOR T = XMVectorSubtract(corners[1], corners[2]);
						const XMVECTOR B = XMVectorSubtract(corners[0], corners[1]);
						const XMVECTOR N = XMVector3Normalize(XMVector3Cross(T, B));
						XMFLOAT3 normal;
						XMStoreFloat3(&normal, N);

						const float slope_amount = 1.0f - saturate(normal.y);
						if (slope_amount > 0.1f)
							slope_cast_shadow.store(true);

						float region_base = 1;
						float region_slope = region1 == 0 ? 1 : smoothstep(0.0f, region1, slope_amount);
						float region_low_altitude = region2 == 0 ? 1 : smoothstep(0.0f, region2, wi::math::InverseLerp(0, bottomLevel, height));
						float region_high_altitude = region3 == 0 ? 1 : smoothstep(0.0f, region3, wi::math::InverseLerp(0, topLevel, height));

						region_low_altitude = saturate(region_low_altitude - region_slope);
						region_high_altitude = saturate(region_high_altitude - region_slope);

						XMFLOAT4 materialBlendWeights(region_base, region_slope, region_low_altitude, region_high_altitude);

						if (!gg_preserve_blend) // GGMAX: engine-default region weights would overwrite GG's blendmap
						{
							chunk_data.blendmap_layers[0].pixels[index] = uint8_t(materialBlendWeights.x * 255);
							chunk_data.blendmap_layers[1].pixels[index] = uint8_t(materialBlendWeights.y * 255);
							chunk_data.blendmap_layers[2].pixels[index] = uint8_t(materialBlendWeights.z * 255);
							chunk_data.blendmap_layers[3].pixels[index] = uint8_t(materialBlendWeights.w * 255);
						}

						// Normalize after store, blending shader wants unnormalized!
						weight_norm(materialBlendWeights);

						mesh.vertex_positions[index] = XMFLOAT3(x, height, z);
						mesh.vertex_normals[index] = normal;
						XMStoreFloat4(&mesh.vertex_tangents[index], T);
						mesh.vertex_tangents[index].w = 1;
						const XMFLOAT2 uv = XMFLOAT2(x * chunk_scale_rcp * chunk_width_rcp + 0.5f, z * chunk_scale_rcp * chunk_width_rcp + 0.5f);
						mesh.vertex_uvset_0[index] = uv;

						XMFLOAT3 vertex_pos(chunk_data.position.x + x, height, chunk_data.position.z + z);

						if (gg_seed_grass) // GGMAX 3.159 (BR-0009 m3): same snapshot as the vertex_lengths resize above
						{
							float spline_factor = 0;
							if (!chunk_data.spline_blendmap_layers.empty())
							{
								for (auto& y : chunk_data.spline_blendmap_layers)
								{
									spline_factor += float(y.pixels[index]) / 255.0f;
								}
								spline_factor /= float(chunk_data.spline_blendmap_layers.size());
							}

							const float grass_noise_frequency = 0.1f;
							const float grass_noise = perlin_noise.compute(vertex_pos.x * grass_noise_frequency, vertex_pos.y * grass_noise_frequency, vertex_pos.z * grass_noise_frequency) * 0.5f + 0.5f;
							const float region_grass = std::pow(materialBlendWeights.x * (1 - materialBlendWeights.w), 8.0f) * grass_noise * (1 - saturate(spline_factor));
							if (region_grass > 0.1f)
							{
								grass_valid_vertex_count.fetch_add(1);
								grass.vertex_lengths[index] = region_grass;
							}
							else
							{
								grass.vertex_lengths[index] = 0;
							}
						}
						chunk_data.heightmap_data[index] = uint16_t(inverse_lerp(bottomLevel, topLevel, height) * 65535);
					});
					wi::jobsystem::Wait(ctx); // wait until chunk's vertex buffer is fully generated
					gg_genprof_vertex_us.fetch_add(uint64_t(gg_t_vertex.elapsed_milliseconds() * 1000.0)); // GGMAX 2.58

					object.SetCastShadow(slope_cast_shadow.load());
					mesh.SetDoubleSidedShadow(slope_cast_shadow.load());

					// GGMAX 3.159 (BR-0009 round 2, m1): every chunk's indices (one static list) and UVs (chunk-local) are
					// byte-identical, so they are uploaded ONCE and shared; each chunk's own GPU buffer keeps only its
					// positions/normals/tangents (~1.46 -> ~0.34 MB per chunk at 128+3). Built here on the generator job,
					// like every chunk's CreateRenderData. Matches() guards it: anything different gets the stock path.
					if (gg_share_chunk_topology)
					{
						if (!gg_chunk_topology || !gg_chunk_topology->Matches(mesh))
						{
							gg_chunk_topology = wi::scene::MeshComponent::CreateSharedTopology(mesh);
							gg_topology_builds.fetch_add(1);
						}
						mesh.gg_shared_topology = gg_chunk_topology;
					}

					wi::jobsystem::Execute(ctx, [&](wi::jobsystem::JobArgs args) {
						wi::Timer gg_t_rd; // GGMAX 2.58: timed inside the async job (overlaps physics below)
						mesh.CreateRenderData();
						if (mesh.gg_shared_topology) gg_topology_shared_chunks.fetch_add(1); else gg_topology_unshared_chunks.fetch_add(1); // GGMAX 3.159
						chunk_data.sphere.center = mesh.aabb.getCenter();
						chunk_data.sphere.center.x += chunk_data.position.x;
						chunk_data.sphere.center.y += chunk_data.position.y;
						chunk_data.sphere.center.z += chunk_data.position.z;
						chunk_data.sphere.radius = mesh.aabb.getRadius();
						gg_genprof_renderdata_us.fetch_add(uint64_t(gg_t_rd.elapsed_milliseconds() * 1000.0));
						if (gg_terrain_pick_bvh && !gg_generation_skip_bvh) // GGMAX 2.58: 8.2 ms/chunk — the generator skips it; 3.159 (BR-0009 m2): GG builds none (grid walk)
						{
							wi::Timer gg_t_bvh; // GGMAX 2.58a: SetBVHEnabled BUILDS the CPU triangle BVH synchronously
							mesh.SetBVHEnabled(true);
							gg_genprof_bvh_us.fetch_add(uint64_t(gg_t_bvh.elapsed_milliseconds() * 1000.0));
							gg_genprof_bvh_events.fetch_add(1);
						}
					});

					// If there were any vertices in this chunk that could be valid for grass, store the grass particle system:
					if (gg_seed_grass && grass_valid_vertex_count.load() > 0) // GGMAX 2.60 / 3.159 (BR-0009 m3)
					{
						wi::Timer gg_t_grass; // GGMAX 2.58
						chunk_data.grass = std::move(grass); // the grass will be added to the scene later, only when the chunk is close to the camera (center chunk's neighbors)
						chunk_data.grass.meshID = chunk_data.entity;
						chunk_data.grass.strandCount = uint32_t(grass_valid_vertex_count.load() * 3 * chunk_scale * chunk_scale); // chunk_scale * chunk_scale : grass density increases with squared amount with chunk scale (x*z)
						chunk_data.grass.CreateFromMesh(mesh);
						gg_genprof_grass_us.fetch_add(uint64_t(gg_t_grass.elapsed_milliseconds() * 1000.0));
					}

					// GGMAX: let the game fill the blendmap with its own weights (generator
					// thread) so the chunk is born correct — the engine-default region weights
					// written above only remain if there is no callback or it declines.
					// Preserved (in-place regen) chunks keep their existing layers untouched.
					if (!gg_preserve_blend)
					{
						wi::Timer gg_t_blendcb; // GGMAX 2.58
						chunk_data.gg_blendmap_generated = false;
						if (gg_generate_blendmap)
						{
							chunk_data.gg_blendmap_generated = gg_generate_blendmap(chunk_data, mesh);
						}
						gg_genprof_blendcb_us.fetch_add(uint64_t(gg_t_blendcb.elapsed_milliseconds() * 1000.0));
					}

					// Create the textures for virtual texture update:
					wi::Timer gg_t_regiontex; // GGMAX 2.58
					chunk_data.heightmap = {};
					if (!gg_preserve_blend) // GGMAX: keep the existing GPU blendmap texture (layers untouched above)
					{
						chunk_data.blendmap = {};
					}
					CreateChunkRegionTexture(chunk_data);
					gg_genprof_regiontex_us.fetch_add(uint64_t(gg_t_regiontex.elapsed_milliseconds() * 1000.0));

					if (IsPhysicsEnabled())
					{
						wi::Timer gg_t_physics; // GGMAX 2.58
						// Precompute the physics shape here on separate thread, because computing shape for triangle mesh would be slow on main thread:
						//	Note that this is mesh.precomputed_rigidbody_physics_shape and not a component in scene.rigidbodies, so this only contains the shape, not the simulated rigid bodies
						RigidBodyPhysicsComponent& newrigidbody = mesh.precomputed_rigidbody_physics_shape;
						newrigidbody.shape = RigidBodyPhysicsComponent::HEIGHTFIELD;
						newrigidbody.mass = 0; // terrain chunks are static
						newrigidbody.friction = 0.8f;
						//newrigidbody.mesh_lod = 2;
						wi::physics::CreateRigidBodyShape(newrigidbody, transform.scale_local, &mesh);
						gg_genprof_physics_us.fetch_add(uint64_t(gg_t_physics.elapsed_milliseconds() * 1000.0));
					}

					wi::jobsystem::Wait(ctx); // wait until mesh.CreateRenderData() async task finishes

					// GGMAX 2.58: close the per-chunk breakdown
					gg_genprof_total_us.fetch_add(uint64_t(gg_t_total.elapsed_milliseconds() * 1000.0));
					gg_genprof_chunks.fetch_add(1);

					generated_something = true;
				}

				const int dist = std::max(std::abs(center_chunk.x - chunk.x), std::abs(center_chunk.z - chunk.z));

				// Grass patch placement:
				if (dist <= grass_chunk_dist && IsGrassEnabled())
				{
					it = chunks.find(chunk);
					if (it != chunks.end() && it->second.entity != INVALID_ENTITY)
					{
						ChunkData& chunk_data = it->second;
						if ((chunk_data.grass_entity == INVALID_ENTITY || chunk_data.invalidated) && chunk_data.grass.meshID != INVALID_ENTITY)
						{
							// add patch for this chunk
							if (chunk_data.grass_entity == INVALID_ENTITY)
							{
								chunk_data.grass_entity = CreateEntity();
							}
							wi::HairParticleSystem& grass = generator->scene.hairs.Create(chunk_data.grass_entity);
							grass = chunk_data.grass;
							chunk_data.grass_density_current = grass_density;
							grass.strandCount = uint32_t(grass.strandCount * chunk_data.grass_density_current);
							grass.CreateRenderData();
							generator->scene.materials.Create(chunk_data.grass_entity) = grass_material;
							generator->scene.transforms.Create(chunk_data.grass_entity);
							generator->scene.names.Create(chunk_data.grass_entity) = "grass";
							generator->scene.Component_Attach(chunk_data.grass_entity, chunk_data.entity, true);
							generated_something = true;
						}
					}
				}

				// Prop placement:
				if (dist <= prop_generation)
				{
					it = chunks.find(chunk);
					if (it != chunks.end() && it->second.entity != INVALID_ENTITY)
					{
						ChunkData& chunk_data = it->second;

						if (prop_density > 0 && chunk_data.props_entity == INVALID_ENTITY && chunk_data.mesh_vertex_positions != nullptr)
						{
							chunk_data.props_entity = CreateEntity();
							generator->scene.transforms.Create(chunk_data.props_entity);
							generator->scene.names.Create(chunk_data.props_entity) = "props";
							generator->scene.Component_Attach(chunk_data.props_entity, chunk_data.entity, true);
							chunk_data.prop_density_current = prop_density;

							wi::random::RNG rng(chunk.compute_hash());

							for (const auto& prop : props)
							{
								if (prop.data.empty())
									continue;
								const int gen_count = rng.next_int(
									int(std::floor(float(prop.min_count_per_chunk) * chunk_data.prop_density_current)),
									int(std::ceil(float(prop.max_count_per_chunk) * chunk_data.prop_density_current))
								);
								for (int i = 0; i < gen_count; ++i)
								{
									const uint32_t tri = rng.next_uint(0, chunk_indices().lods[0].indexCount / 3); // random triangle on the chunk mesh
									const uint32_t ind0 = chunk_indices().indices[tri * 3 + 0];
									const uint32_t ind1 = chunk_indices().indices[tri * 3 + 1];
									const uint32_t ind2 = chunk_indices().indices[tri * 3 + 2];
									const XMFLOAT3& pos0 = chunk_data.mesh_vertex_positions[ind0];
									const XMFLOAT3& pos1 = chunk_data.mesh_vertex_positions[ind1];
									const XMFLOAT3& pos2 = chunk_data.mesh_vertex_positions[ind2];
									XMFLOAT4 region0 = wi::Color(chunk_data.blendmap_layers[0].pixels[ind0], chunk_data.blendmap_layers[1].pixels[ind0], chunk_data.blendmap_layers[2].pixels[ind0], chunk_data.blendmap_layers[3].pixels[ind0]);
									XMFLOAT4 region1 = wi::Color(chunk_data.blendmap_layers[0].pixels[ind1], chunk_data.blendmap_layers[1].pixels[ind1], chunk_data.blendmap_layers[2].pixels[ind1], chunk_data.blendmap_layers[3].pixels[ind1]);
									XMFLOAT4 region2 = wi::Color(chunk_data.blendmap_layers[0].pixels[ind2], chunk_data.blendmap_layers[1].pixels[ind2], chunk_data.blendmap_layers[2].pixels[ind2], chunk_data.blendmap_layers[3].pixels[ind2]);
									weight_norm(region0);
									weight_norm(region1);
									weight_norm(region2);
									float spline_factor0 = 0;
									float spline_factor1 = 0;
									float spline_factor2 = 0;
									if (!chunk_data.spline_blendmap_layers.empty())
									{
										for (auto& y : chunk_data.spline_blendmap_layers)
										{
											spline_factor0 += float(y.pixels[ind0]) / 255.0f;
											spline_factor1 += float(y.pixels[ind1]) / 255.0f;
											spline_factor2 += float(y.pixels[ind2]) / 255.0f;
										}
										const float rcp = 1.0f / float(chunk_data.spline_blendmap_layers.size());
										spline_factor0 *= rcp;
										spline_factor1 *= rcp;
										spline_factor2 *= rcp;
									}
									// random barycentric coords on the triangle:
									float f = rng.next_float();
									float g = rng.next_float();
									if (f + g > 1)
									{
										f = 1 - f;
										g = 1 - g;
									}
									const XMFLOAT3 vertex_pos = XMFLOAT3(
										pos0.x + f * (pos1.x - pos0.x) + g * (pos2.x - pos0.x),
										pos0.y + f * (pos1.y - pos0.y) + g * (pos2.y - pos0.y),
										pos0.z + f * (pos1.z - pos0.z) + g * (pos2.z - pos0.z)
									);
									const XMFLOAT4 region = XMFLOAT4(
										region0.x + f * (region1.x - region0.x) + g * (region2.x - region0.x),
										region0.y + f * (region1.y - region0.y) + g * (region2.y - region0.y),
										region0.z + f * (region1.z - region0.z) + g * (region2.z - region0.z),
										region0.w + f * (region1.w - region0.w) + g * (region2.w - region0.w)
									);
									const float spline_factor = spline_factor0 + f * (spline_factor1 - spline_factor0) + g * (spline_factor2 - spline_factor0);

									const float noise = std::pow(perlin_noise.compute((vertex_pos.x + chunk_data.position.x) * prop.noise_frequency, vertex_pos.y * prop.noise_frequency, (vertex_pos.z + chunk_data.position.z) * prop.noise_frequency) * 0.5f + 0.5f, prop.noise_power);
									const float chance = std::pow(((float*)&region)[prop.region], prop.region_power) * noise * (1 - saturate(spline_factor));
									if (chance > prop.threshold)
									{
										wi::Archive archive = wi::Archive(prop.data.data(), prop.data.size());
										EntitySerializer seri;
										Entity entity = generator->scene.Entity_Serialize(
											archive,
											seri,
											INVALID_ENTITY,
											wi::scene::Scene::EntitySerializeFlags::RECURSIVE |
											wi::scene::Scene::EntitySerializeFlags::KEEP_INTERNAL_ENTITY_REFERENCES
										);
										NameComponent* name = generator->scene.names.GetComponent(entity);
										if (name != nullptr)
										{
											name->name += std::to_string(i);
										}
										TransformComponent* transform = generator->scene.transforms.GetComponent(entity);
										if (transform == nullptr)
										{
											transform = &generator->scene.transforms.Create(entity);
										}
										transform->translation_local = vertex_pos;
										transform->translation_local.y += wi::math::Lerp(prop.min_y_offset, prop.max_y_offset, rng.next_float());
										const float scaling = wi::math::Lerp(prop.min_size, prop.max_size, rng.next_float());
										transform->Scale(XMFLOAT3(scaling, scaling, scaling));
										transform->RotateRollPitchYaw(XMFLOAT3(0, XM_2PI * rng.next_float(), 0));
										transform->SetDirty();
										transform->UpdateTransform();
										generator->scene.Component_Attach(entity, chunk_data.props_entity, true);
										generated_something = true;
									}
								}
							}
						}
					}
				}

				it = chunks.find(chunk); // re-query!
				if (it != chunks.end() && it->second.entity != INVALID_ENTITY)
				{
					ChunkData& chunk_data = it->second;
					chunk_data.invalidated = false;
					if (gg_chunk_generated)
						chunk_data.merge_pending = true; // GGMAX: freshly (re)generated — main-scene mesh is stale until the next merge
				}

				if (generated_something && timer.elapsed_milliseconds() > generation_time_budget_milliseconds)
				{
					generator->cancelled.store(true);
				}

			};

			// priority invalidation queue:
			//	This doesn't necessarily finish every frame, that's why it's a queue, next frame will pick up earlier requests before newer ones
			while (!generator->priority_invalidation.empty())
			{
				Chunk chunk = generator->priority_invalidation.front();
				generator->priority_invalidation.pop_front();
				auto it = chunks.find(chunk);
				if (it != chunks.end() && it->second.invalidated) // Check here too in this special case, because multiple of the same entries can easily exist on the queue. Already refreshes chunks will not be refreshed again
				{
					request_chunk(chunk.x, chunk.z);
					if (generator->cancelled.load()) return;
				}
			}

			// GGMAX view-cone priority pre-pass: sweep the same outward spiral once,
			// restricted to chunks in front of the camera (plus the two rings
			// immediately around it), so the terrain the user is looking at builds
			// first. The full spiral below then fills in the rest — request_chunk
			// fast-skips chunks that already exist, and the generation budget makes
			// each job launch resume wherever the previous one ran out.
			if (view_cone_forward.x != 0 || view_cone_forward.y != 0)
			{
				request_chunk(0, 0);
				if (generator->cancelled.load()) return;
				for (int growth = 0; growth < generation; ++growth)
				{
					const int side = 2 * (growth + 1);
					int x = -growth - 1;
					int z = -growth - 1;
					auto cone_request = [&](int cx, int cz)
					{
						if (growth >= 2) // rings 1-2 around the camera are unconditional
						{
							const float lenr = std::sqrt(float(cx * cx + cz * cz));
							const float d = (float(cx) * view_cone_forward.x + float(cz) * view_cone_forward.y) / std::max(0.001f, lenr);
							if (d < 0.35f) return; // outside ~70 degree half-angle
						}
						request_chunk(cx, cz);
					};
					for (int i = 0; i < side; ++i) { cone_request(x, z); if (generator->cancelled.load()) return; x++; }
					for (int i = 0; i < side; ++i) { cone_request(x, z); if (generator->cancelled.load()) return; z++; }
					for (int i = 0; i < side; ++i) { cone_request(x, z); if (generator->cancelled.load()) return; x--; }
					for (int i = 0; i < side; ++i) { cone_request(x, z); if (generator->cancelled.load()) return; z--; }
				}
			}

			// generate center chunk first:
			request_chunk(0, 0);
			if (generator->cancelled.load()) return;

			// then generate neighbor chunks in outward spiral:
			for (int growth = 0; growth < generation; ++growth)
			{
				const int side = 2 * (growth + 1);
				int x = -growth - 1;
				int z = -growth - 1;
				for (int i = 0; i < side; ++i)
				{
					request_chunk(x, z);
					if (generator->cancelled.load()) return;
					x++;
				}
				for (int i = 0; i < side; ++i)
				{
					request_chunk(x, z);
					if (generator->cancelled.load()) return;
					z++;
				}
				for (int i = 0; i < side; ++i)
				{
					request_chunk(x, z);
					if (generator->cancelled.load()) return;
					x--;
				}
				for (int i = 0; i < side; ++i)
				{
					request_chunk(x, z);
					if (generator->cancelled.load()) return;
					z--;
				}
			}

			});

		wi::profiler::gg_trace_mark("tg-kick"); // GGMAX 2.61
	}

	void Terrain::Generation_Cancel()
	{
		if (generator == nullptr)
			return;
		generator->cancelled.store(true); // tell the generation thread that work must be stopped
		wi::jobsystem::Wait(generator->workload); // waits until generation thread exits
		generator->cancelled.store(false); // the next generation can run
	}

	void Terrain::CreateChunkRegionTexture(ChunkData& chunk_data)
	{
		GraphicsDevice* device = GetDevice();

		if (!chunk_data.heightmap.IsValid() && !chunk_data.heightmap_data.empty())
		{
			TextureDesc desc;
			desc.width = (uint32_t)chunk_width;
			desc.height = (uint32_t)chunk_width;
			desc.format = Format::R16_UNORM;
			desc.bind_flags = BindFlag::SHADER_RESOURCE;

			SubresourceData data;
			data.data_ptr = chunk_data.heightmap_data.data();
			data.row_pitch = chunk_width * sizeof(uint16_t);

			bool success = device->CreateTexture(&desc, &data, &chunk_data.heightmap);
			assert(success);
			device->SetName(&chunk_data.heightmap, "wi::terrain::ChunkData::heightmap");
		}

		const uint32_t required_layers = uint32_t(chunk_data.blendmap_layers.size() + chunk_data.spline_blendmap_layers.size());
		const uint32_t gg_bw = gg_BlendmapWidth(); // GGMAX PROTOTYPE (BR-0009 round 6): 66 at 131 when half-res is on
		if (!chunk_data.blendmap.IsValid() || chunk_data.blendmap.desc.array_size != required_layers || chunk_data.blendmap.desc.width != gg_bw)
		{
			TextureDesc desc;
			desc.width = gg_bw;
			desc.height = gg_bw;
			desc.array_size = required_layers;
			desc.format = Format::R8_UNORM;
			desc.bind_flags = BindFlag::SHADER_RESOURCE;

			wi::vector<SubresourceData> data(required_layers);
			for (size_t i = 0; i < chunk_data.blendmap_layers.size(); ++i)
			{
				data[i].data_ptr = chunk_data.blendmap_layers[i].pixels.data();
				data[i].row_pitch = chunk_width * sizeof(uint8_t);
			}
			const size_t offset = chunk_data.blendmap_layers.size();
			for (size_t i = 0; i < chunk_data.spline_blendmap_layers.size(); ++i)
			{
				data[offset + i].data_ptr = chunk_data.spline_blendmap_layers[i].pixels.data();
				data[offset + i].row_pitch = chunk_width * sizeof(uint8_t);
			}

			// GGMAX PROTOTYPE (BR-0009 round 6): half-res upload. Texel (tx,ty) is centred on vertex (2tx,2ty); interior
			// texels take a [1 2 1]x[1 2 1] tent (point decimation would drop odd-vertex paint strokes), border rows and
			// columns filter ALONG the edge only and corners are point samples - so a border texel is built from border
			// vertices alone, which neighbouring chunks share: seams stay closed. ("Centred on vertex 2tx" is the CPU side: the
			// readers sample at uv = i/130 with no texel-centre remap - the stock convention - so on screen a texel sits up to
			// ~1 vertex (~39 units at scale 40) from vertex 2tx near the chunk edges: the same world error as main's 67-texel
			// map at scale 80. A remap (uv*(bw-1)+0.5)/bw in all three readers would remove it; review 11-06.) HEAP scratch (the generator thread's
			// frame is already large); CreateTexture copies initial data before it returns, so it may die with the scope.
			wi::vector<uint8_t> gg_half;
			const uint32_t W = (uint32_t)chunk_width;
			if (gg_bw != W && required_layers > 0)
			{
				gg_half.resize(size_t(required_layers) * gg_bw * gg_bw);
				for (uint32_t l = 0; l < required_layers; ++l)
				{
					const wi::vector<uint8_t>& px = l < offset ? chunk_data.blendmap_layers[l].pixels : chunk_data.spline_blendmap_layers[l - offset].pixels;
					uint8_t* dst = gg_half.data() + size_t(l) * gg_bw * gg_bw;
					if (px.size() != size_t(W) * W) { std::memset(dst, 0, size_t(gg_bw) * gg_bw); } // wrong-sized layer: empty, never an overread
					else
					{
						for (uint32_t ty = 0; ty < gg_bw; ++ty)
						{
							const uint32_t vy = 2 * ty;
							const bool yEdge = ty == 0 || ty == gg_bw - 1;
							for (uint32_t tx = 0; tx < gg_bw; ++tx)
							{
								const uint32_t vx = 2 * tx;
								const bool xEdge = tx == 0 || tx == gg_bw - 1;
								uint32_t sum = 0, wsum = 0;
								for (int dy = yEdge ? 0 : -1; dy <= (yEdge ? 0 : 1); ++dy)
								{
									const uint32_t wy = dy == 0 ? 2u : 1u;
									const uint8_t* row = px.data() + size_t(vy + dy) * W;
									for (int dx = xEdge ? 0 : -1; dx <= (xEdge ? 0 : 1); ++dx)
									{
										const uint32_t w = wy * (dx == 0 ? 2u : 1u);
										sum += w * row[vx + dx];
										wsum += w;
									}
								}
								dst[size_t(ty) * gg_bw + tx] = uint8_t((sum + wsum / 2) / wsum);
							}
						}
					}
					data[l].data_ptr = dst;
					data[l].row_pitch = gg_bw * sizeof(uint8_t);
				}
			}

			bool success = device->CreateTexture(&desc, data.data(), &chunk_data.blendmap);
			assert(success);
			device->SetName(&chunk_data.blendmap, "wi::terrain::ChunkData::blendmap");
		}
	}

	void Terrain::UpdateVirtualTexturesCPU()
	{
		wi::Timer gg_t_updcpu; // GGMAX 2.61: whole main-thread body (schedules the async job at the end)
		wi::jobsystem::Wait(virtual_texture_ctx);
		gg_vtprof_wait0_us.fetch_add(uint64_t(gg_t_updcpu.elapsed_milliseconds() * 1000.0)); // GGMAX 2.61: entry wait on last frame's VT job
		virtual_texture_ctx.priority = wi::jobsystem::Priority::Low;

		GraphicsDevice* device = GetDevice();
		virtual_textures_in_use.clear();

		// GGMAX: per-frame cap on VT residency upgrades once the camera settles, so the
		// near ring sharpens over a few frames instead of hitching (gg_vt_upgrade_hysteresis)
		uint32_t gg_upgrade_budget = 4;

		// GGMAX: while the camera is crossing chunk boundaries, FREEZE the tile allocator
		// completely — no tile aging, no free-list rebuild, no allocations, no stealing.
		// The GPU feedback that keeps visible tiles alive arrives on a 2-3 frame delayed
		// readback; during fast camera moves the keep-alive gaps let STILL-VISIBLE tiles
		// age into the free list, and the flood of misses from newly-visible terrain then
		// steals them — another chunk renders its pixels into a square that was correct
		// and on screen (the "random squares" during violent zooms). Frozen terrain keeps
		// every resident tile; newly approached terrain simply stays coarse until the
		// camera settles, then streams in as designed.
		const bool gg_vt_frozen = gg_vt_upgrade_hysteresis && gg_center_stable_frames < 10;
		if (gg_vt_frozen && !atlas.free_tiles.empty())
		{
			// The free list entering this frame's chunk loop (used by chunk-init tail
			// allocations) was built by the previous job, possibly before the freeze —
			// filter it to the provably off-screen set so init can't steal visible tiles.
			atlas.free_tiles.erase(std::remove_if(atlas.free_tiles.begin(), atlas.free_tiles.end(),
				[&](const VirtualTextureAtlas::Tile& t) { return atlas.get_tile_frames(t) < 60; }),
				atlas.free_tiles.end());
		}

		if (!sampler.IsValid())
		{
			SamplerDesc samplerDesc;
			samplerDesc.filter = Filter::ANISOTROPIC;
			// NOTE: this sampler is used by the SVT terrain via SampleLevel() (explicit LOD), which
			// IGNORES the sampler's anisotropy and mip_lod_bias — so changing max_anisotropy here has
			// NO visible effect on the terrain. (An earlier x4->x8 attempt was a confirmed no-op and
			// was reverted; left at stock 4.) Terrain filtering/LOD is driven by SVT_MIP_BIAS +
			// get_lod(GetAnisotropy()) in ShaderInterop_Renderer.h, not by this sampler.
			samplerDesc.max_anisotropy = 4;
			// Note: using wrap mode by intention!
			//	Terrain itself doesn't need wrap mode, but decals will reuse the base material's sampler
			//	and decals can use wrapped textures (texmuladd)
			samplerDesc.address_u = TextureAddressMode::WRAP;
			samplerDesc.address_v = TextureAddressMode::WRAP;
			samplerDesc.address_w = TextureAddressMode::WRAP;
			bool success = device->CreateSampler(&samplerDesc, &sampler);
			assert(success);
		}

		for (auto& it : chunks)
		{
			const Chunk& chunk = it.first;
			ChunkData& chunk_data = it.second;
			const int dist = std::max(std::abs(center_chunk.x - chunk.x), std::abs(center_chunk.z - chunk.z));

			MaterialComponent* material = scene->materials.GetComponent(chunk_data.entity);
			if (material == nullptr)
				continue;

			material->sampler_descriptor = device->GetDescriptorIndex(&sampler);

			// This should have been created on generation thread, but if not (serialized), create it last minute:
			{
				// GGMAX 2.61: detect + time when this "last minute" path ACTUALLY creates textures on
				// the main thread (mirror of the function's own guards; a no-op call costs nothing).
				const uint32_t gg_req_layers = uint32_t(chunk_data.blendmap_layers.size() + chunk_data.spline_blendmap_layers.size());
				const bool gg_will_create =
					(!chunk_data.heightmap.IsValid() && !chunk_data.heightmap_data.empty()) ||
					(!chunk_data.blendmap.IsValid() || chunk_data.blendmap.desc.array_size != gg_req_layers ||
					 chunk_data.blendmap.desc.width != gg_BlendmapWidth()); // GGMAX PROTOTYPE (BR-0009 round 6): mirror
				if (gg_will_create)
				{
					wi::Timer gg_t_regionmain;
					CreateChunkRegionTexture(chunk_data);
					gg_vtprof_regionmain_us.fetch_add(uint64_t(gg_t_regionmain.elapsed_milliseconds() * 1000.0));
					gg_vtprof_regionmain_events.fetch_add(1);
				}
				else
				{
					CreateChunkRegionTexture(chunk_data);
				}
			}

			if (!atlas.IsValid())
			{
				// GGMAX 1.71 (VRAM census): the SVT physical atlas is a FIXED cost — 4 sparse
				// maps at 16384x16384 back a 768 MB tile pool no matter how much of it a level
				// uses. Measured residency across Island Showdown / Horseshoe Bend / Zombie
				// Cellar / Switch Escape: only ~890-1090 of 3844 tiles ever resident (~26%).
				// gg_svt_atlas_height trims the atlas (and the pool) proportionally; 8192
				// halves it to 384 MB and still leaves ~1.8x headroom over the measured peak.
				// Takes effect on the next atlas creation, i.e. reload the level.
				const uint32_t physical_width = 16384u;
				const uint32_t physical_height = gg_svt_atlas_height;
				GPUBufferDesc tile_pool_desc;

				for (uint32_t map_type = 0; map_type < arraysize(atlas.maps); ++map_type)
				{
					if (map_type == MaterialComponent::EMISSIVEMAP && !gg_svt_keep_emissive)
						continue; // GGMAX 1.98 (A4): no emissive map, no pool share
					TextureDesc desc;
					desc.width = physical_width;
					desc.height = physical_height;
					desc.misc_flags = ResourceMiscFlag::SPARSE;
					desc.bind_flags = BindFlag::SHADER_RESOURCE;
					desc.mip_levels = 1;
					desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;

					TextureDesc desc_raw_block = desc;
					desc_raw_block.width /= 4;
					desc_raw_block.height /= 4;
					desc_raw_block.bind_flags = BindFlag::UNORDERED_ACCESS;
					desc_raw_block.layout = ResourceState::UNORDERED_ACCESS;

					switch (map_type)
					{
					default:
					case MaterialComponent::BASECOLORMAP:
					case MaterialComponent::EMISSIVEMAP:
						desc.format = Format::BC1_UNORM_SRGB;
						desc_raw_block.format = Format::R32G32_UINT;
						break;
					case MaterialComponent::NORMALMAP:
						desc.format = Format::BC5_UNORM;
						desc_raw_block.format = Format::R32G32B32A32_UINT;
						desc.swizzle.r = ComponentSwizzle::R;
						desc.swizzle.g = ComponentSwizzle::G;
						desc.swizzle.b = ComponentSwizzle::ONE;
						desc.swizzle.a = ComponentSwizzle::ONE;
						break;
					case MaterialComponent::SURFACEMAP:
						desc.format = Format::BC3_UNORM;
						desc_raw_block.format = Format::R32G32B32A32_UINT;
						break;
					}

					bool success = device->CreateTexture(&desc, nullptr, &atlas.maps[map_type].texture);
					assert(success);
					device->SetName(&atlas.maps[map_type].texture, "VirtualTextureAtlas::texture");

					success = device->CreateTexture(&desc_raw_block, nullptr, &atlas.maps[map_type].texture_raw_block);
					assert(success);
					device->SetName(&atlas.maps[map_type].texture_raw_block, "VirtualTextureAtlas::texture_raw_block");

					assert(atlas.maps[map_type].texture.sparse_properties->total_tile_count == atlas.maps[map_type].texture_raw_block.sparse_properties->total_tile_count);
					assert(atlas.maps[map_type].texture.sparse_page_size == atlas.maps[map_type].texture_raw_block.sparse_page_size);

					tile_pool_desc.size += atlas.maps[map_type].texture.sparse_properties->total_tile_count * atlas.maps[map_type].texture.sparse_page_size;
					tile_pool_desc.alignment = std::max(tile_pool_desc.alignment, atlas.maps[map_type].texture.sparse_page_size);

					for (uint32_t i = 0; i < atlas.maps[map_type].texture_raw_block.desc.mip_levels; ++i)
					{
						int subresource_index = device->CreateSubresource(&atlas.maps[map_type].texture_raw_block, SubresourceType::UAV, 0, 1, i, 1);
						assert(subresource_index == i);
					}
				}

				tile_pool_desc.misc_flags = ResourceMiscFlag::SPARSE_TILE_POOL_TEXTURE_NON_RT_DS;
				bool success = device->CreateBuffer(&tile_pool_desc, nullptr, &atlas.tile_pool);
				assert(success);

				atlas.physical_tile_count_x = uint8_t(physical_width / SVT_TILE_SIZE_PADDED);
				atlas.physical_tile_count_y = uint8_t(physical_height / SVT_TILE_SIZE_PADDED);
				atlas.physical_tiles.resize(size_t(atlas.physical_tile_count_x) * size_t(atlas.physical_tile_count_y));

				uint64_t init_frames = 0;
				atlas.free_tiles.clear();
				atlas.free_tiles.reserve(atlas.physical_tile_count_x * atlas.physical_tile_count_y);
				for (uint8_t y = 0; y < atlas.physical_tile_count_y; ++y)
				{
					for (uint8_t x = 0; x < atlas.physical_tile_count_x; ++x)
					{
						VirtualTextureAtlas::Tile& tile = atlas.free_tiles.emplace_back();
						tile.x = x;
						tile.y = y;
						atlas.physical_tiles[x + y * atlas.physical_tile_count_x].free_frames = ++init_frames;
					}
				}

				uint32_t offset = 0;
				for (uint32_t map_type = 0; map_type < arraysize(atlas.maps); ++map_type)
				{
					if (!atlas.maps[map_type].texture.IsValid())
						continue; // GGMAX 1.98 (A4): emissive map not created
					// Sparse mapping for block compression aliasing:
					SparseUpdateCommand commands[2];
					commands[0].sparse_resource = &atlas.maps[map_type].texture;
					commands[0].tile_pool = &atlas.tile_pool;
					commands[0].num_resource_regions = 1;
					SparseResourceCoordinate coordinate;
					coordinate.x = 0;
					coordinate.y = 0;
					commands[0].coordinates = &coordinate;
					SparseRegionSize region;
					region.width = atlas.maps[map_type].texture.desc.width / atlas.maps[map_type].texture.sparse_properties->tile_width;
					region.height = atlas.maps[map_type].texture.desc.height / atlas.maps[map_type].texture.sparse_properties->tile_height;
					commands[0].sizes = &region;
					TileRangeFlags flags = {};
					commands[0].range_flags = &flags;
					commands[0].range_start_offsets = &offset;
					uint32_t count = atlas.maps[map_type].texture.sparse_properties->total_tile_count;
					commands[0].range_tile_counts = &count;
					commands[1] = commands[0];
					commands[1].sparse_resource = &atlas.maps[map_type].texture_raw_block;
					device->SparseUpdate(QUEUE_COMPUTE, commands, arraysize(commands));
					offset += count;
				}
			}

			if (chunk_data.vt == nullptr)
			{
				chunk_data.vt = wi::allocator::make_shared<VirtualTexture>();
			}
			VirtualTexture& vt = *chunk_data.vt;

			// GGMAX 3.02: bake-mode resolution override. 0 = SVT_TILE_SIZE (256, a SINGLE tile,
			// which is precisely why the bake path needs no residency object and skips all four
			// VT GPU passes). Anything larger is >1 tile and pulls residency back in - see
			// VirtualTexture::init, "if (tile_count > 1) residency = atlas.allocate_residency".
			// Exposed so that trade can be MEASURED rather than argued.
			const uint32_t min_resolution = ( gg_bake_resolution > 0 ) ? gg_bake_resolution : SVT_TILE_SIZE;
			const uint32_t max_resolution = 65536u;

			//GGMAX
			//const uint32_t required_resolution = dist < 2 ? max_resolution : min_resolution;
			//const uint32_t required_resolution = std::max(min_resolution, max_resolution >> std::min(7, std::max(0, dist - 1)));
			//const int d = std::max(0, dist - 4); // hold max_resolution for dist 0..4
			//const uint32_t required_resolution = std::max(min_resolution, max_resolution >> std::min(7, d));
			const uint32_t required_resolution = dist < gg_near_ring_dist ? max_resolution : min_resolution; // GGMAX: widened full-res zone (stock: dist < 2)

			// GGMAX: an in-place chunk regen merge replaces the chunk's MaterialComponent with a
			// fresh one (no atlas bindings). Detect that and re-run the bind block against the
			// EXISTING vt — without vt.init(), so residency/tiles survive and nothing re-streams.
			const bool gg_material_rebind = vt.resolution == required_resolution && vt.resolution != 0 &&
				!material->textures[0].resource.IsValid();
			// GGMAX: a chunk whose tail-tile allocation failed (tile pool exhausted at init
			// time) renders flat untextured squares and stock has NO retry path — force a
			// re-init once the pool has tiles again. Never deferred by the hysteresis.
			const bool gg_tail_invalid = vt.resolution != 0 &&
				(vt.tiles.empty() || !vt.tiles.back().IsValid()) && !atlas.free_tiles.empty();
			// GGMAX: VT hysteresis — while the camera is crossing chunk boundaries, FREEZE
			// resolution changes entirely: chunks keep rendering their existing correct
			// tiles at their current resolution instead of re-initializing mid-motion
			// (residency churn showed as square tiles of mixed sharpness/stale content
			// flickering during fast zooms). Once the camera holds still for 10 frames,
			// downgrades run immediately (frees tile pool) and upgrades run a few per
			// frame. Fresh (resolution 0) and unbound-material chunks are never deferred —
			// they must bind now.
			bool gg_defer_change = false;
			if (gg_vt_upgrade_hysteresis && vt.resolution != 0 && vt.resolution != required_resolution &&
				material->textures[0].resource.IsValid())
			{
				if (gg_center_stable_frames < 10)
				{
					gg_defer_change = true; // camera in motion: keep the old VT references
				}
				else if (required_resolution > vt.resolution)
				{
					if (gg_upgrade_budget == 0)
						gg_defer_change = true;
					else
						gg_upgrade_budget--;
				}
			}
			if ((vt.resolution != required_resolution && !gg_defer_change) || gg_material_rebind || gg_tail_invalid)
			{
				if (vt.resolution != required_resolution || gg_tail_invalid)
					vt.init(atlas, required_resolution);

				for (uint32_t map_type = 0; map_type < arraysize(atlas.maps); ++map_type)
				{
					if (!atlas.maps[map_type].texture.IsValid())
					{
						// GGMAX 1.98 (A4): the map was dropped. Bind an EXPLICIT 1x1 black —
						// leaving the slot unbound feeds an undefined sparse sample into the
						// shader and the whole scene washes to white (2026-08-02 post-mortem).
						material->textures[map_type].resource.SetTexture(*wi::texturehelper::getBlack());
						material->textures[map_type].sparse_residencymap_descriptor = -1;
						material->textures[map_type].sparse_feedbackmap_descriptor = -1;
						continue;
					}
					material->textures[map_type].resource.SetTexture(atlas.maps[map_type].texture);
					if (vt.residency != nullptr)
					{
						material->texMulAdd = XMFLOAT4(1, 1, 0, 0);
						material->textures[map_type].virtual_anisotropy = sampler.desc.max_anisotropy;
						material->textures[map_type].sparse_residencymap_descriptor = device->GetDescriptorIndex(&vt.residency->residencyMap, SubresourceType::SRV);
						if (map_type == 0)
						{
							// Only first texture slot will write feedback
							material->textures[map_type].sparse_feedbackmap_descriptor = device->GetDescriptorIndex(&vt.residency->feedbackMap, SubresourceType::UAV);
						}
						else
						{
							material->textures[map_type].sparse_feedbackmap_descriptor = -1;
						}
						material->textures[map_type].resource.SetTextureVirtual(atlas.tile_pool, vt.residency->residencyMap, vt.residency->feedbackMap);
					}
					else
					{
						// Simple chunks without residency only have 1 mapped tile, and no mips so they simply use a texMulAdd (todo)
						auto tile = vt.tiles.back();
						const float2 resolution_rcp = float2(
							1.0f / (float)atlas.maps[map_type].texture.desc.width,
							1.0f / (float)atlas.maps[map_type].texture.desc.height
						);
						if (map_type == 0)
						{
							material->texMulAdd.x = (float)SVT_TILE_SIZE * resolution_rcp.x;
							material->texMulAdd.y = (float)SVT_TILE_SIZE * resolution_rcp.y;
							material->texMulAdd.z = ((float)tile.x * (float)SVT_TILE_SIZE_PADDED + SVT_TILE_BORDER) * resolution_rcp.x;
							material->texMulAdd.w = ((float)tile.y * (float)SVT_TILE_SIZE_PADDED + SVT_TILE_BORDER) * resolution_rcp.y;
						}
						material->textures[map_type].sparse_residencymap_descriptor = -1;
						material->textures[map_type].sparse_feedbackmap_descriptor = -1;
						material->textures[map_type].resource.SetTextureVirtual(atlas.tile_pool, Texture(), Texture());
					}
					material->textures[map_type].lod_clamp = (float)vt.lod_count - 2;
				}
				vt.blendmap = chunk_data.blendmap;
				// GGMAX 1.42: the writes above changed the material's composed state (descriptor
				// indices, texMulAdd, lod_clamp) WITHOUT SetDirty — harmless pre-1.41 because
				// RunMaterialUpdateSystem recomposed every material every frame, but the 1.41
				// recompose cache would keep serving the STALE composition for up to 63 frames
				// (heartbeat) after any VT rebind: the chunk samples the previous residency/
				// feedback maps (pooled, possibly re-issued to ANOTHER chunk) or stale single-
				// tile coordinates — grey/foreign chunk textures during load, reload and
				// resolution-ring crossings. Mark dirty so the cache recomposes this frame.
				material->SetDirty();
			}

			// GGMAX: blendmap-edit repaint — pick up the recreated blendmap texture here
			// on the main thread and LATCH the request for the update job below. The job
			// consumes the latch, never the live flag: game code sets the live flag at any
			// point in the frame, possibly while a prior job is still in flight, and a job
			// that consumed the live flag before this re-bind ran would re-render tiles
			// against the OLD blendmap and drop the request — the edit would never land.
			// (The previous frame's job was joined at the top of this function, so no job
			// is in flight while this block transfers the flag.)
			if (vt.pending_repaint_blendmap)
			{
				vt.pending_repaint_blendmap = false;
				vt.gg_repaint_blendmap_latched = true;
				vt.blendmap = chunk_data.blendmap;
			}

			virtual_textures_in_use.push_back(&vt);

			if (vt.residency == nullptr)
				continue;
		}

		wi::jobsystem::Execute(virtual_texture_ctx, [this, gg_vt_frozen](wi::jobsystem::JobArgs args) {
			auto gg_range_total = wi::profiler::BeginRangeCPU("VT-job Total"); // GGMAX 1.32 instrumentation

			// Update state of physical tiles:
			//	Potentially each physical tile is getting marked as unused here (free_frames > 0), unless GPU requested them to be resident
			// GGMAX: aging must ALWAYS run — freezing it starved the recycle pool during
			// sustained camera motion (nothing ever crossed the recycle threshold), chunk
			// inits then failed to allocate tail tiles and the whole terrain degraded to
			// flat untextured squares. The protection against mid-motion steals is the
			// recycle threshold below, not an aging freeze.
			for (auto& x : atlas.physical_tiles)
			{
				x.free_frames++;
			}

			// Process GPU allocation requests from last frame:
			//	GPU writes allocation requests by virtualTextureTileAllocateCS.hlsl compute shader
			for (VirtualTexture* vt : virtual_textures_in_use)
			{
				// Last two tiles (last non-packed and packed mips) always kept resident:
				if (vt->tiles.size() > 0)
				{
					atlas.request_residency(vt->tiles[vt->tiles.size() - 1]);
					if (vt->tiles.size() > 1)
					{
						atlas.request_residency(vt->tiles[vt->tiles.size() - 2]);
					}
				}
				if (vt->residency == nullptr)
					continue;

				const uint32_t width = vt->residency->feedbackMap.desc.width;
				const uint32_t height = vt->residency->feedbackMap.desc.height;

				// We must only access persistently mapped resources by CPU that the GPU is not using currently:
				vt->residency->cpu_resource_id = (vt->residency->cpu_resource_id + 1) % arraysize(vt->residency->allocationBuffer_CPU_readback);
				const bool data_available_CPU = vt->residency->data_available_CPU[vt->residency->cpu_resource_id]; // indicates whether any GPU data is readable at this point or not
				vt->residency->data_available_CPU[vt->residency->cpu_resource_id] = true;

				if (data_available_CPU)
				{
					uint32_t page_count = 0;
					uint32_t lod_offsets[10] = {};
					for (uint32_t i = 0; i < vt->lod_count; ++i)
					{
						const uint32_t l_width = std::max(1u, width >> i);
						const uint32_t l_height = std::max(1u, height >> i);
						lod_offsets[i] = page_count;
						page_count += l_width * l_height;
					}

					uint32_t allocation_count = *(const uint32_t*)vt->residency->allocationBuffer_CPU_readback[vt->residency->cpu_resource_id].mapped_data;
					allocation_count = std::min(uint32_t(vt->tiles.size() - 1), allocation_count);
					const uint32_t* allocation_requests = ((const uint32_t*)vt->residency->allocationBuffer_CPU_readback[vt->residency->cpu_resource_id].mapped_data) + 1; // +1 offset of the allocation counter
					for (uint32_t i = 0; i < allocation_count; ++i)
					{
						const uint32_t allocation_request = allocation_requests[i];
						const uint8_t x = (allocation_request >> 24u) & 0xFF;
						const uint8_t y = (allocation_request >> 16u) & 0xFF;
						const uint8_t lod = (allocation_request >> 8u) & 0xFF;
						if (lod >= vt->lod_count)
							continue;
						const uint32_t l_offset = lod_offsets[lod];
						const uint32_t l_width = std::max(1u, width >> lod);
						const uint32_t l_height = std::max(1u, height >> lod);
						const uint32_t l_index = l_offset + x + y * l_width;
						if (x >= l_width || y >= l_height)
							continue;

						VirtualTextureAtlas::Tile& tile = vt->tiles[l_index];
						if (!atlas.request_residency(tile))
						{
							// The request_residency returned false, meaning it is not resident, so it will need to be allocated
							//	Allocation can be made only after all GPU requests have been processed, because some of them will
							//	remain resident, and those occupied tiles shoudln't be given out right now
							VirtualTexture::AllocationRequest& request = vt->allocation_requests.emplace_back();
							request.x = x;
							request.y = y;
							request.lod = lod;
							request.tile_index = l_index;
						}
					}
				}
			}

			// Determine reusable tiles:
			//	After all GPU requests were processed, we can make the unused tiles available for allocations
			// GGMAX: while the camera is in motion, only tiles unused for 60+ frames may be
			// recycled — provably off-screen, since the delayed feedback keep-alive gaps
			// during fast moves are only 2-3 frames (visible tiles oscillate at 0-3 and can
			// never be stolen). Tiles below the threshold belong to terrain that was
			// recently visible and correct: stealing them mid-motion is what painted random
			// foreign squares over the terrain during violent zooms. Aging keeps running,
			// so genuinely off-screen tiles keep flowing into the pool even mid-flight.
			const uint64_t gg_min_free_frames = gg_vt_frozen ? 60 : 1;
			// GGMAX 1.33: the rebuild+sort of the free list costs ~2ms — only pay it on frames
			// that can actually consume or invalidate it: pending allocation requests (rebuilt
			// fresh right before fulfilment, so keep-alive resets this frame are respected),
			// the list was consumed since the last build (main-thread chunk-init or fulfil),
			// or it is running low (headroom for main-thread chunk-init between rebuilds).
			// While parked with no requests, nothing consumes the list, so a stale-but-
			// conservative list is safe — aging still runs every frame above.
			size_t gg_total_requests = 0;
			for (VirtualTexture* vt : virtual_textures_in_use)
			{
				gg_total_requests += vt->allocation_requests.size();
			}
			// Low-water refill is rate-limited: with a near-full atlas the free list can sit
			// under the threshold permanently, which must not re-introduce an every-frame sort.
			// (Review F3: Terrain members, not lambda statics — no cross-instance sharing.)
			if (gg_freesort_cooldown > 0) gg_freesort_cooldown--;
			const bool gg_low_water = atlas.free_tiles.size() < 256 && gg_freesort_cooldown == 0;
			// Review F2: a center-chunk change (teleport/zoom) invalidates the list's keep-alive
			// assumptions — force a rebuild so chunk-init can't steal recently-live tiles.
			const bool gg_center_moved = !(gg_freesort_last_center == center_chunk);
			const bool gg_rebuild_free = !gg_vt_incremental || gg_total_requests > 0 || atlas.gg_free_dirty || gg_low_water || gg_center_moved;
			// GGMAX diag: record what pulled the trigger
			gg_dbg_vt_requests += (unsigned long long)gg_total_requests;
			if (gg_rebuild_free)
			{
				gg_dbg_vt_rebuilds++;
				gg_dbg_vt_reason |= (!gg_vt_incremental ? 1ull : 0) | (gg_total_requests > 0 ? 2ull : 0) |
					(atlas.gg_free_dirty ? 4ull : 0) | (gg_low_water ? 8ull : 0) | (gg_center_moved ? 16ull : 0);
				gg_dbg_vt_tiles = (unsigned long long)atlas.physical_tiles.size();
			}
			if (gg_rebuild_free)
			{
			if (gg_low_water) gg_freesort_cooldown = 8;
			gg_freesort_last_center = center_chunk; // review F2
			auto gg_range_free = wi::profiler::BeginRangeCPU("VT-job FreeSort"); // GGMAX 1.32 instrumentation
			unsigned long long gg_t0 = gg_dbg_vt_qpc_us();
			atlas.free_tiles.clear();
			for (uint8_t y = 0; y < atlas.physical_tile_count_y; ++y)
			{
				for (uint8_t x = 0; x < atlas.physical_tile_count_x; ++x)
				{
					if (atlas.physical_tiles[x + y * atlas.physical_tile_count_x].free_frames >= gg_min_free_frames)
					{
						VirtualTextureAtlas::Tile& tile = atlas.free_tiles.emplace_back();
						tile.x = x;
						tile.y = y;
					}
				}
			}
			unsigned long long gg_t1 = gg_dbg_vt_qpc_us();
			// Sort them by unused frame counts, this will make them be given out in least recently used (LRU) order:
			std::sort(atlas.free_tiles.begin(), atlas.free_tiles.end(), [&](const VirtualTextureAtlas::Tile& a, const VirtualTextureAtlas::Tile& b) {
				return atlas.get_tile_frames(a) < atlas.get_tile_frames(b);
			});
			unsigned long long gg_t2 = gg_dbg_vt_qpc_us();
			gg_dbg_vt_scan_us += gg_t1 - gg_t0;
			gg_dbg_vt_sort_us += gg_t2 - gg_t1;
			gg_dbg_vt_free = (unsigned long long)atlas.free_tiles.size();
			wi::profiler::EndRange(gg_range_free); // GGMAX 1.32
			atlas.gg_free_dirty = false; // GGMAX 1.33
			}

			// Fulfill new allocation requests:
			for (VirtualTexture* vt : virtual_textures_in_use)
			{
				for (auto& alloc_request : vt->allocation_requests)
				{
					VirtualTextureAtlas::Tile& tile = vt->tiles[alloc_request.tile_index];
					VirtualTexture* gg_stolen = nullptr; // GGMAX 1.33
					if (atlas.allocate_tile(tile, vt, &gg_stolen))
					{
						if (gg_stolen) gg_stolen->gg_page_dirty = true; // GGMAX 1.33: victim's mapping broke
						vt->gg_page_dirty = true; // GGMAX 1.33
						VirtualTexture::UpdateRequest& request = vt->update_requests.emplace_back();
						request.x = (uint16_t)alloc_request.x;
						request.y = (uint16_t)alloc_request.y;
						request.lod = (uint16_t)alloc_request.lod;
						request.tile_x = tile.x;
						request.tile_y = tile.y;
					}
				}
				vt->allocation_requests.clear();
			}

			// GGMAX: blendmap-edit repaint — re-render every currently resident page with
			// the updated blendmap. Residency is untouched, so this refreshes the visible
			// terrain next frame; the old full invalidate() path re-streamed the chunk
			// through several seconds of feedback round-trips after every paint stroke.
			for (VirtualTexture* vt : virtual_textures_in_use)
			{
				if (!vt->gg_repaint_blendmap_latched)
					continue;
				vt->gg_repaint_blendmap_latched = false;
				if (vt->residency == nullptr)
					continue; // single-tile far chunks take the cheap invalidate() path instead

				const uint32_t width = vt->residency->feedbackMap.desc.width;
				const uint32_t height = vt->residency->feedbackMap.desc.height;
				uint32_t page_count = 0;
				uint32_t lod_offsets[10] = {};
				for (uint32_t i = 0; i < vt->lod_count; ++i)
				{
					const uint32_t l_width = std::max(1u, width >> i);
					const uint32_t l_height = std::max(1u, height >> i);
					lod_offsets[i] = page_count;
					page_count += l_width * l_height;
				}
				for (uint32_t lod = 0; lod < vt->lod_count; ++lod)
				{
					const uint32_t l_width = std::max(1u, width >> lod);
					const uint32_t l_height = std::max(1u, height >> lod);
					for (uint32_t y = 0; y < l_height; ++y)
					{
						for (uint32_t x = 0; x < l_width; ++x)
						{
							const uint32_t l_index = lod_offsets[lod] + x + y * l_width;
							if (l_index >= vt->tiles.size())
								continue;
							VirtualTextureAtlas::Tile& tile = vt->tiles[l_index];
							if (!atlas.check_tile_resident(tile))
								continue;
							VirtualTexture::UpdateRequest& request = vt->update_requests.emplace_back();
							request.x = (uint16_t)x;
							request.y = (uint16_t)y;
							request.lod = (uint16_t)lod;
							request.tile_x = tile.x;
							request.tile_y = tile.y;
						}
					}
				}
			}

			auto gg_range_pagebuf = wi::profiler::BeginRangeCPU("VT-job PageBuf"); // GGMAX 1.32 instrumentation
			// GGMAX 1.33: a VT's page table only changes when a tile is allocated INTO it or
			// stolen FROM it — both mark gg_page_dirty. Clean VTs skip the (huge, uncached)
			// upload write AND their GPU copy + residency recompute stay skipped downstream,
			// because the GPU-side pageBuffer still holds the identical data. A rotating
			// heartbeat force-refreshes one resident VT per frame as insurance against any
			// missed dirty path (full sweep every N frames at ~1/N of the cost).
			gg_vt_frame_counter++; // review F3: Terrain member, not a lambda static
			const uint32_t gg_vt_frame = gg_vt_frame_counter;
			uint32_t gg_resident_count = 0;
			for (VirtualTexture* vt : virtual_textures_in_use)
			{
				if (vt->residency != nullptr)
					gg_resident_count++;
			}
			const uint32_t gg_hb_index = (gg_resident_count > 0) ? (gg_vt_frame % gg_resident_count) : 0;
			uint32_t gg_vt_index = 0;
			for (VirtualTexture* vt : virtual_textures_in_use)
			{
				if (vt->residency == nullptr)
					continue;

				const bool gg_heartbeat = (gg_vt_index++ == gg_hb_index);
				// Review F1: also rewrite when a previous upload was never consumed — cpu_resource_id
				// advances every job run, so if Render didn't run since the last write (load screens,
				// path switches) the still-pending flag would otherwise copy a STALE slot later.
				if (gg_vt_incremental && !vt->gg_page_dirty && !gg_heartbeat && !vt->gg_page_upload_pending)
					continue;

				// Update page buffer for GPU:
				//	This must be after VirtualTextureAtlas::physical_tiles is no longer being updated
				uint16_t* page_buffer = (uint16_t*)vt->residency->pageBuffer_CPU_upload[vt->residency->cpu_resource_id].mapped_data;
				for (size_t i = 0; i < vt->tiles.size(); ++i)
				{
					if (!atlas.check_tile_resident(vt->tiles[i]))
					{
						vt->tiles[i] = {};
					}
					uint16_t page = vt->tiles[i];
					// force memcpy into uncached memory to avoid read stall by mistake:
					std::memcpy(page_buffer + i, &page, sizeof(page));
				}
				vt->gg_page_dirty = false; // GGMAX 1.33
				vt->gg_page_upload_pending = true; // GGMAX 1.33 -> CopyVirtualTexturePageStatusGPU
				vt->gg_residency_update_pending = true; // GGMAX 1.33 -> UpdateVirtualTexturesGPU
			}
			wi::profiler::EndRange(gg_range_pagebuf); // GGMAX 1.32
			wi::profiler::EndRange(gg_range_total); // GGMAX 1.32
		});

		// GGMAX 2.61: attribution totals for the whole main-thread body
		{
			const uint64_t gg_us = uint64_t(gg_t_updcpu.elapsed_milliseconds() * 1000.0);
			gg_vtprof_updatecpu_us.fetch_add(gg_us);
			gg_vtprof_updatecpu_calls.fetch_add(1);
			uint64_t gg_prev = gg_vtprof_updatecpu_max_us.load();
			while (gg_us > gg_prev && !gg_vtprof_updatecpu_max_us.compare_exchange_weak(gg_prev, gg_us)) {}
		}
	}

	// =========================================================================================
	// GGMAX 3.25: TERRAIN BAKE - resolve one chunk's painted surface into a plain Texture2D.
	//
	// The "Terrain Bake" low-spec switch turns the whole rendered terrain into a set of ordinary
	// meshes and ordinary textures, then removes the Wicked terrain entirely (the 2.94 Terrain
	// Off teardown). This is the texture half: it runs the SAME blend the virtual-texture tile
	// bake runs - blendmap layer weights against the layer materials' basecolour maps, front to
	// back with early-out - but writes an uncompressed RGBA8 pixel instead of a BC1 block into a
	// sparse atlas tile. See terrainBakeChunkCS.hlsl for why the compression cannot come along.
	//
	// Deliberately reads chunk_data.blendmap and NOT vt->blendmap. They are the same texture
	// (UpdateVirtualTexturesCPU does `vt.blendmap = chunk_data.blendmap`), but the chunk owns it
	// and a chunk can be perfectly bakeable while its virtual texture is still null or evicted.
	// Depending on the VT here would make the bake silently camera-history-dependent: chunks the
	// player had walked near would bake and the rest would come out empty.
	//
	// `output` is created on first use and reused after that, so a re-bake of the same chunk at
	// the same resolution costs one dispatch and no allocation.
	// =========================================================================================
	// GGMAX 3.25k: the bake's shared uncompressed scratch. File scope rather than function-static
	// so it can be RELEASED once a bake finishes - at 8192 with a mip chain it is about 340 MB,
	// which is far too much to leave resident for a one-shot conversion.
	static Texture gg_bake_scratch;

	// Mip chain length that stops at 4x4. BC1 compresses 4x4 blocks, so a 2x2 or 1x1 BC mip is
	// not representable; the engine's own DDS import aligns to 4 for the same reason.
	uint32_t gg_BakeMipCount(uint32_t resolution)
	{
		uint32_t mips = 1, r = resolution;
		while (r > 4u) { r >>= 1; mips++; }
		return mips;
	}

	void gg_ReleaseBakeScratch()
	{
		gg_bake_scratch = {};
	}

	bool Terrain::gg_BakeChunkBasecolor(const ChunkData& chunk_data, wi::graphics::Texture& output, uint32_t resolution, CommandList cmd) const
	{
		if (!chunk_data.blendmap.IsValid())
			return false;
		// 8192 ceiling, not 4096: the GGMAX two-tier bake promotes chunks inside the play area to a
		// much higher resolution than the distant ones, and 4096 over a ~5120-unit chunk is still
		// only about 1.25 units per texel. BC1 keeps even 8192 at 32 MB a chunk, and the caller
		// budgets how many chunks may be promoted.
		if (resolution < 8 || resolution > 8192)
			return false;
		if (scene == nullptr)
			return false;

		GraphicsDevice* device = GetDevice();

		// ★ ONE SHARED SCRATCH for the whole bake, not one per chunk. The compute shader needs a
		// UAV and DX12 forbids a UAV on a BC format, so the uncompressed surface is written here
		// and immediately compressed into the caller's BC1 texture. Only one chunk is in flight
		// at a time, so a single scratch serves all 625 - at 1024 that is 4 MB total instead of
		// 2.5 GB of per-chunk uncompressed textures.
		const uint32_t mip_count = gg_BakeMipCount(resolution);
		if (!gg_bake_scratch.IsValid() || gg_bake_scratch.desc.width != resolution)
		{
			gg_bake_scratch = {};
			TextureDesc sd;
			sd.width = resolution;
			sd.height = resolution;
			sd.mip_levels = mip_count;
			// UNORM, not _SRGB: DX12 forbids a UAV on an sRGB format. The shader applies the sRGB
			// curve itself and the BC1_UNORM_SRGB destination decodes it in hardware on sample.
			sd.format = Format::R8G8B8A8_UNORM;
			sd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			sd.layout = ResourceState::SHADER_RESOURCE;
			if (!device->CreateTexture(&sd, nullptr, &gg_bake_scratch))
				return false;
			device->SetName(&gg_bake_scratch, "GGMAX::TerrainBake::scratch");
			// per-mip SRV and UAV, which is what GenerateMipChain and BlockCompress index by.
			for (uint32_t i = 0; i < gg_bake_scratch.desc.mip_levels; ++i)
			{
				device->CreateSubresource(&gg_bake_scratch, SubresourceType::SRV, 0, 1, i, 1);
				device->CreateSubresource(&gg_bake_scratch, SubresourceType::UAV, 0, 1, i, 1);
			}
		}

		// The caller owns `output` and must have created it as a committed BC1 texture with
		// SHADER_RESOURCE only - BlockCompress transitions it to COPY_DST and back itself.
		if (!output.IsValid())
			return false;

		// subresource 0 = mip 0. With a chain present the default UAV is ambiguous, so name it.
		const int uav = device->GetDescriptorIndex(&gg_bake_scratch, SubresourceType::UAV, 0);
		const int blendmap_srv = device->GetDescriptorIndex(&chunk_data.blendmap, SubresourceType::SRV);
		if (uav < 0 || blendmap_srv < 0)
			return false;

		TerrainVirtualTexturePush push = {};
		push.output_texture = uav;
		push.blendmap_texture = blendmap_srv;
		push.blendmap_layers = chunk_data.blendmap.desc.array_size;
		if (push.blendmap_layers == 0)
			return false;

		// Material index table, one uint per blendmap layer. Same clamping as the tile bake: an
		// entity without a live MaterialComponent gives GetIndex() == SIZE_MAX, which the shader
		// would turn into an out-of-bounds ShaderMaterial read, a garbage texture descriptor and
		// a GPU page fault. That exact path was the 2.04 Aztec DEVICE_HUNG, so it is clamped here
		// too rather than assumed impossible on the bake path.
		auto mem = device->AllocateGPU(sizeof(uint32_t) * push.blendmap_layers, cmd);
		if (mem.data == nullptr)
			return false;
		const uint32_t splineMaterialCount = (uint32_t)splineMaterialEntities.size();
		const uint32_t baseMaterialCount = (push.blendmap_layers > splineMaterialCount)
			? (push.blendmap_layers - splineMaterialCount) : 0u;
		const size_t material_count = scene->materials.GetCount();
		auto safe_index = [&](Entity entity) -> uint32_t
		{
			const size_t idx = scene->materials.GetIndex(entity);
			return (idx >= material_count) ? 0u : (uint32_t)idx;
		};
		for (uint32_t i = 0; i < baseMaterialCount; ++i)
		{
			const uint32_t mi = (i < (uint32_t)materialEntities.size()) ? safe_index(materialEntities[i]) : 0u;
			std::memcpy((uint32_t*)mem.data + i, &mi, sizeof(uint32_t)); // memcpy: no uncached read back from a GPU pointer
		}
		for (uint32_t i = 0; i < splineMaterialCount && (baseMaterialCount + i) < push.blendmap_layers; ++i)
		{
			const uint32_t mi = safe_index(splineMaterialEntities[i]);
			std::memcpy((uint32_t*)mem.data + baseMaterialCount + i, &mi, sizeof(uint32_t));
		}
		push.blendmap_buffer = device->GetDescriptorIndex(&mem.buffer, SubresourceType::SRV);
		if (push.blendmap_buffer < 0)
			return false;
		push.blendmap_buffer_offset = (uint32_t)mem.offset;

		// Whole chunk in one go: no tile borders, no packed mip tail, no page offsets.
		push.offset = int2(0, 0);
		push.write_offset = uint2(0, 0);
		push.write_size = resolution;              // PIXELS here, not 4x4 blocks
		push.resolution_rcp = 1.0f / (float)resolution;
		// Same distance-tiling policy as the live terrain, or the baked chunks would disagree
		// with any still-live chunk on texture scale.
		push.gg_tile_share = (std::min(gg_terrain_tile_share_mips, 255u) << 24u)
			| (std::min(gg_terrain_tile_hold_mips, 255u) << 16u)
			| (gg_splat_blend ? 1u : 0u); // GGMAX 3.113

		device->EventBegin("GGMAX Terrain Bake Chunk", cmd);
		wi::renderer::BindCommonResources(cmd);
		device->BindComputeShader(wi::renderer::GetShader(wi::enums::CSTYPE_GG_TERRAIN_BAKECHUNK), cmd);

		{
			GPUBarrier barrier = GPUBarrier::Image(&gg_bake_scratch, ResourceState::SHADER_RESOURCE, ResourceState::UNORDERED_ACCESS);
			device->Barrier(&barrier, 1, cmd);
		}

		device->PushConstants(&push, sizeof(push), cmd);
		device->Dispatch((resolution + 7u) / 8u, (resolution + 7u) / 8u, 1u, cmd);

		{
			// back to SHADER_RESOURCE, which is exactly the state BlockCompress requires its
			// source to be in - it never transitions the source itself.
			GPUBarrier barrier = GPUBarrier::Image(&gg_bake_scratch, ResourceState::UNORDERED_ACCESS, ResourceState::SHADER_RESOURCE);
			device->Barrier(&barrier, 1, cmd);
		}

		// Fill the rest of the chain from mip 0, then compress every level. BlockCompress binds
		// source mip N for destination mip N, so the two counts must agree - which is why the
		// caller creates its BC1 with gg_BakeMipCount(resolution) levels as well.
		wi::renderer::GenerateMipChain(gg_bake_scratch, wi::renderer::MIPGENFILTER_LINEAR, cmd);
		wi::renderer::BlockCompress(gg_bake_scratch, output, cmd);

		device->EventEnd(cmd);
		return true;
	}

	void Terrain::UpdateVirtualTexturesGPU(CommandList cmd) const
	{
		{
			auto gg_range_wait = wi::profiler::BeginRangeCPU("VT-WaitInRecord"); // GGMAX 1.32 instrumentation
			wi::jobsystem::Wait(virtual_texture_ctx);
			wi::profiler::EndRange(gg_range_wait);
		}
		auto gg_range_rec = wi::profiler::BeginRangeCPU("VT-record"); // GGMAX 1.32 instrumentation
		GraphicsDevice* device = GetDevice();
		device->EventBegin("Terrain - UpdateVirtualTexturesGPU", cmd);
		auto range = wi::profiler::BeginRangeGPU("Terrain - UpdateVirtualTexturesGPU", cmd);

		device->EventBegin("Update Residency Maps", cmd);
		device->BindComputeShader(wi::renderer::GetShader(wi::enums::CSTYPE_VIRTUALTEXTURE_RESIDENCYUPDATE), cmd);
		for (const VirtualTexture* vt : virtual_textures_in_use)
		{
			if (vt->residency == nullptr)
				continue;
			// GGMAX 1.33: residencyMap is a pure function of pageBuffer — skip the recompute
			// when the page table didn't change (pageBuffer holds identical data).
			if (gg_vt_incremental && !vt->gg_residency_update_pending)
				continue;
			vt->gg_residency_update_pending = false;
			VirtualTextureResidencyUpdateCB cb;
			cb.lodCount = vt->lod_count;
			cb.width = vt->residency->residencyMap.desc.width;
			cb.height = vt->residency->residencyMap.desc.height;
			cb.pageBufferRO = device->GetDescriptorIndex(&vt->residency->pageBuffer, SubresourceType::SRV);
			for (uint mip = 0; mip < arraysize(cb.residencyTextureRW_mips); ++mip)
			{
				if (mip < vt->lod_count)
				{
					cb.residencyTextureRW_mips[mip].x = device->GetDescriptorIndex(&vt->residency->residencyMap, SubresourceType::UAV, mip);
				}
				else
				{
					cb.residencyTextureRW_mips[mip].x = -1;
				}
			}
			device->BindDynamicConstantBuffer(cb, 0, cmd);

			device->Dispatch(
				(vt->residency->residencyMap.desc.width + 7u) / 8u,
				(vt->residency->residencyMap.desc.height + 7u) / 8u,
				1u,
				cmd
			);
		}
		device->EventEnd(cmd);

		device->EventBegin("Render Tile Regions", cmd);
		wi::renderer::BindCommonResources(cmd);
		for (const VirtualTexture* vt : virtual_textures_in_use)
		{
			if (vt->update_requests.empty())
				continue;
			for (uint32_t map_type = 0; map_type < arraysize(atlas.maps); map_type++)
			{
				if (!atlas.maps[map_type].texture_raw_block.IsValid())
					continue; // GGMAX 1.98 (A4): emissive map not created (belt+braces; push.output_texture would be -1 anyway)
				TerrainVirtualTexturePush push = {};

				switch (map_type)
				{
				case MaterialComponent::BASECOLORMAP:
					device->BindComputeShader(wi::renderer::GetShader(wi::enums::CSTYPE_TERRAIN_VIRTUALTEXTURE_UPDATE_BASECOLORMAP), cmd);
					push.output_texture = device->GetDescriptorIndex(&atlas.maps[MaterialComponent::BASECOLORMAP].texture_raw_block, SubresourceType::UAV);
					break;
				case MaterialComponent::NORMALMAP:
					device->BindComputeShader(wi::renderer::GetShader(wi::enums::CSTYPE_TERRAIN_VIRTUALTEXTURE_UPDATE_NORMALMAP), cmd);
					push.output_texture = device->GetDescriptorIndex(&atlas.maps[MaterialComponent::NORMALMAP].texture_raw_block, SubresourceType::UAV);
					break;
				case MaterialComponent::SURFACEMAP:
					device->BindComputeShader(wi::renderer::GetShader(wi::enums::CSTYPE_TERRAIN_VIRTUALTEXTURE_UPDATE_SURFACEMAP), cmd);
					push.output_texture = device->GetDescriptorIndex(&atlas.maps[MaterialComponent::SURFACEMAP].texture_raw_block, SubresourceType::UAV);
					break;
				case MaterialComponent::EMISSIVEMAP:
					device->BindComputeShader(wi::renderer::GetShader(wi::enums::CSTYPE_TERRAIN_VIRTUALTEXTURE_UPDATE_EMISSIVEMAP), cmd);
					push.output_texture = device->GetDescriptorIndex(&atlas.maps[MaterialComponent::EMISSIVEMAP].texture_raw_block, SubresourceType::UAV);
					break;
				default:
					assert(0);
					break;
				}
				if (push.output_texture < 0)
					continue;

				push.blendmap_texture = device->GetDescriptorIndex(&vt->blendmap, SubresourceType::SRV);
				push.blendmap_layers = vt->blendmap.desc.array_size;
				if (push.blendmap_texture < 0)
					continue;

				auto mem = device->AllocateGPU(sizeof(uint) * push.blendmap_layers, cmd);
				const uint splineMaterialCount = (uint)splineMaterialEntities.size();
				const uint baseMaterialCount = push.blendmap_layers - splineMaterialCount;
				// GGMAX 2.04: an entity in these lists without a live MaterialComponent makes
				// GetIndex return SIZE_MAX -> material_index 0xFFFFFFFF -> the tile-render CS
				// reads a ShaderMaterial miles out of bounds -> garbage texture descriptor ->
				// GPU page fault -> DXGI_ERROR_DEVICE_HUNG (Aztec PLAY GAME, 6/6 repro, always
				// mid "Render Tile Regions"). Clamp to material 0 and shout once per session.
				const size_t gg_mat_count = scene->materials.GetCount();
				static bool gg_badmat_reported = false;
				auto gg_safe_material_index = [&](Entity entity, const char* which, uint slot) -> uint
				{
					size_t idx = scene->materials.GetIndex(entity);
					if (idx >= gg_mat_count)
					{
						if (!gg_badmat_reported)
						{
							gg_badmat_reported = true;
							wi::backlog::post("GGMAX 2.04: terrain tile render hit a material entity WITHOUT a live MaterialComponent (" + std::string(which) + " slot " + std::to_string(slot) + ", entity " + std::to_string((unsigned long long)entity) + ") - clamped to material 0. This was the DEVICE_HUNG page-fault path.", wi::backlog::LogLevel::Error);
						}
						return 0u;
					}
					return (uint)idx;
				};
				for (uint i = 0; i < splineMaterialCount; ++i)
				{
					const uint material_index = gg_safe_material_index(splineMaterialEntities[i], "spline", i);
					std::memcpy((uint*)mem.data + baseMaterialCount + i, &material_index, sizeof(uint)); // force memcpy to avoid uncached read from GPU pointer!
				}
				// GGMAX 2.04: baseMaterialCount comes from the CHUNK's blendmap array size — if a
				// chunk carries more painted layers than the registered material entities (set
				// swap, paint data ahead of slot registration), materialEntities[i] was an OOB
				// vector read (UB) before the entity check even ran. Clamp the loop too.
				const uint gg_live_base = std::min(baseMaterialCount, (uint)materialEntities.size());
				for (uint i = 0; i < baseMaterialCount; ++i)
				{
					uint material_index = 0u;
					if (i < gg_live_base)
					{
						material_index = gg_safe_material_index(materialEntities[i], "base/painted", i);
					}
					else if (!gg_badmat_reported)
					{
						gg_badmat_reported = true;
						wi::backlog::post("GGMAX 2.04: chunk blendmap has " + std::to_string(baseMaterialCount) + " base layers but only " + std::to_string((uint)materialEntities.size()) + " material entities are registered - layer " + std::to_string(i) + " clamped to material 0. This was the DEVICE_HUNG page-fault path.", wi::backlog::LogLevel::Error);
					}
					std::memcpy((uint*)mem.data + i, &material_index, sizeof(uint)); // force memcpy to avoid uncached read from GPU pointer!
				}

				push.blendmap_buffer = device->GetDescriptorIndex(&mem.buffer, SubresourceType::SRV);
				if (push.blendmap_buffer < 0)
					continue;
				push.blendmap_buffer_offset = (uint)mem.offset;

				// GGMAX 1.53b/c: tiling repeat cap (bits 24..31; 0 = stock) + ladder hold (bits 16..23)
				push.gg_tile_share = (std::min(gg_terrain_tile_share_mips, 255u) << 24u)
					| (std::min(gg_terrain_tile_hold_mips, 255u) << 16u)
					| (gg_splat_blend ? 1u : 0u); // GGMAX 3.113

				for (auto& request : vt->update_requests)
				{
					uint request_lod_resolution = std::max(1u, vt->resolution >> request.lod);
					const uint2 write_offset_original = uint2(
						request.tile_x * SVT_TILE_SIZE_PADDED / 4,
						request.tile_y * SVT_TILE_SIZE_PADDED / 4
					);

					push.offset = int2(
						int(request.x * SVT_TILE_SIZE) - int(SVT_TILE_BORDER),
						int(request.y * SVT_TILE_SIZE) - int(SVT_TILE_BORDER)
					);

					if (request_lod_resolution < SVT_TILE_SIZE)
					{
						// packed mips
						uint32_t tail_mip_idx = 0;
						while (request_lod_resolution >= 4u)
						{
							push.resolution_rcp = 1.0f / request_lod_resolution;
							push.write_offset = write_offset_original;
							push.write_offset.x += SVT_PACKED_MIP_OFFSETS[tail_mip_idx].x / 4;
							push.write_offset.y += SVT_PACKED_MIP_OFFSETS[tail_mip_idx].y / 4;
							push.write_size = (SVT_TILE_BORDER + request_lod_resolution + SVT_TILE_BORDER) / 4;
							device->PushConstants(&push, sizeof(push), cmd);

							device->Dispatch(
								(push.write_size + 7u) / 8u,
								(push.write_size + 7u) / 8u,
								1,
								cmd
							);

							request_lod_resolution /= 2u;
							tail_mip_idx++;
						}
					}
					else
					{
						push.resolution_rcp = 1.0f / request_lod_resolution;
						push.write_offset = write_offset_original;
						push.write_size = SVT_TILE_SIZE_PADDED / 4u;
						device->PushConstants(&push, sizeof(push), cmd);

						device->Dispatch(
							(push.write_size + 7u) / 8u,
							(push.write_size + 7u) / 8u,
							1,
							cmd
						);
					}
				}
			}
			vt->update_requests.clear();
		}
		device->Barrier(GPUBarrier::Memory(), cmd);
		device->EventEnd(cmd);

		wi::profiler::EndRange(range);
		device->EventEnd(cmd);
		wi::profiler::EndRange(gg_range_rec); // GGMAX 1.32
	}

	void Terrain::CopyVirtualTexturePageStatusGPU(CommandList cmd) const
	{
		wi::jobsystem::Wait(virtual_texture_ctx);
		GraphicsDevice* device = GetDevice();
		device->EventBegin("Terrain - CopyVirtualTexturePageStatusGPU", cmd);
		auto gg_vtrange = wi::profiler::BeginRangeGPU("TerrainVT - CopyPageStatus", cmd); // GGMAX 2.94c
		for (const VirtualTexture* vt : virtual_textures_in_use)
		{
			if (vt->residency == nullptr)
				continue;
			// GGMAX 1.33: only copy upload slots the VT job actually wrote this frame —
			// unwritten slots hold stale data from N frames ago and the GPU pageBuffer
			// already has the current (unchanged) table.
			if (gg_vt_incremental && !vt->gg_page_upload_pending)
				continue;
			vt->gg_page_upload_pending = false;
			device->CopyResource(&vt->residency->pageBuffer, &vt->residency->pageBuffer_CPU_upload[vt->residency->cpu_resource_id], cmd);
		}
		wi::profiler::EndRange(gg_vtrange); // GGMAX 2.94c
		device->EventEnd(cmd);

		if (chunk_buffer.IsValid())
		{
			device->EventBegin("Update chunk buffer", cmd);
			auto mem = device->AllocateGPU(chunk_buffer.desc.size, cmd);
			ShaderTerrainChunk* shader_chunks = (ShaderTerrainChunk*)mem.data;
			for (int y = -chunk_buffer_range; y <= chunk_buffer_range; ++y)
			{
				for (int x = -chunk_buffer_range; x <= chunk_buffer_range; ++x)
				{
					ShaderTerrainChunk shader_chunk;
					shader_chunk.init();

					auto chunk = center_chunk;
					chunk.x += x;
					chunk.z += y;
					auto it = chunks.find(chunk);
					if (it != chunks.end())
					{
						const auto& chunk_data = it->second;
						shader_chunk.heightmap = device->GetDescriptorIndex(&chunk_data.heightmap, SubresourceType::SRV);
						shader_chunk.materialID = (uint)scene->materials.GetIndex(chunk_data.entity);
					}

					const uint idx = (x + chunk_buffer_range) + (y + chunk_buffer_range) * (chunk_buffer_range * 2 + 1);
					std::memcpy(shader_chunks + idx, &shader_chunk, sizeof(shader_chunk));
				}
			}
			device->CopyBuffer(&chunk_buffer, 0, &mem.buffer, mem.offset, chunk_buffer.desc.size, cmd);
			device->EventEnd(cmd);
		}
	}

	void Terrain::AllocateVirtualTextureTileRequestsGPU(CommandList cmd) const
	{
		wi::jobsystem::Wait(virtual_texture_ctx);
		GraphicsDevice* device = GetDevice();

		device->EventBegin("Terrain - AllocateVirtualTextureTileRequestsGPU", cmd);
		auto gg_vtrange = wi::profiler::BeginRangeGPU("TerrainVT - AllocateTileRequests", cmd); // GGMAX 2.94c

		{
			device->EventBegin("Tile Requests", cmd);
			device->BindComputeShader(wi::renderer::GetShader(wi::enums::CSTYPE_VIRTUALTEXTURE_TILEREQUESTS), cmd);
			for (const VirtualTexture* vt : virtual_textures_in_use)
			{
				if (vt->residency == nullptr)
					continue;
				VirtualTextureTileRequestsPush push;
				push.lodCount = vt->lod_count;
				push.width = vt->residency->feedbackMap.desc.width;
				push.height = vt->residency->feedbackMap.desc.height;
				push.feedbackTextureRW = device->GetDescriptorIndex(&vt->residency->feedbackMap, SubresourceType::UAV);
				push.requestBufferRW = device->GetDescriptorIndex(&vt->residency->requestBuffer, SubresourceType::UAV);
				device->PushConstants(&push, sizeof(push), cmd);

				device->Dispatch(
					(std::max(1u, vt->residency->feedbackMap.desc.width / 2) + 7u) / 8u,
					(std::max(1u, vt->residency->feedbackMap.desc.height / 2) + 7u) / 8u,
					1u,
					cmd
				);
			}
			device->EventEnd(cmd);
		}

		device->Barrier(GPUBarrier::Memory(), cmd);

		{
			device->EventBegin("Tile Allocation Requests", cmd);
			device->BindComputeShader(wi::renderer::GetShader(wi::enums::CSTYPE_VIRTUALTEXTURE_TILEALLOCATE), cmd);
			for (const VirtualTexture* vt : virtual_textures_in_use)
			{
				if (vt->residency == nullptr)
					continue;
				VirtualTextureTileAllocatePush push;
				push.threadCount = (uint)vt->tiles.size();
				push.lodCount = vt->lod_count;
				push.width = vt->residency->feedbackMap.desc.width;
				push.height = vt->residency->feedbackMap.desc.height;
				push.pageBufferRO = device->GetDescriptorIndex(&vt->residency->pageBuffer, SubresourceType::SRV);
				push.requestBufferRW = device->GetDescriptorIndex(&vt->residency->requestBuffer, SubresourceType::UAV);
				push.allocationBufferRW = device->GetDescriptorIndex(&vt->residency->allocationBuffer, SubresourceType::UAV);
				device->PushConstants(&push, sizeof(push), cmd);

				device->Dispatch(
					(push.threadCount + 63u) / 64u,
					1u,
					1u,
					cmd
				);
			}
			device->EventEnd(cmd);
		}

		wi::profiler::EndRange(gg_vtrange); // GGMAX 2.94c
		device->EventEnd(cmd);
	}

	void Terrain::WritebackTileRequestsGPU(CommandList cmd) const
	{
		wi::jobsystem::Wait(virtual_texture_ctx);
		GraphicsDevice* device = GetDevice();

		device->EventBegin("Terrain - WritebackTileRequestsGPU", cmd);
		auto gg_vtrange = wi::profiler::BeginRangeGPU("TerrainVT - WritebackTileRequests", cmd); // GGMAX 2.94c

		for (const VirtualTexture* vt : virtual_textures_in_use)
		{
			if (vt->residency == nullptr)
				continue;
			device->CopyResource(&vt->residency->allocationBuffer_CPU_readback[vt->residency->cpu_resource_id], &vt->residency->allocationBuffer, cmd);
			wi::renderer::PushBarrier(GPUBarrier::Buffer(&vt->residency->allocationBuffer, ResourceState::COPY_SRC, ResourceState::COPY_DST));
		}

		wi::renderer::FlushBarriers(cmd);

		device->EventEnd(cmd);

		device->EventBegin("Clear Metadata", cmd);
		for (const VirtualTexture* vt : virtual_textures_in_use)
		{
			if (vt->residency == nullptr)
				continue;

			// Note: instead of ClearUAV, we use copies from default initialized resources
			//	With this it can be put on the async copy queue
			//	Also, using the ClearUAV was having a very bad performance especially with DX12
			static wi::unordered_map<uint32_t, VirtualTextureAtlas::Residency> clear_residencies;
			VirtualTextureAtlas::Residency& clear = clear_residencies[vt->resolution];
			if (clear.resolution != vt->resolution)
			{
				clear.init(vt->resolution);
			}
			device->CopyResource(&vt->residency->feedbackMap, &clear.feedbackMap, cmd);
			device->CopyResource(&vt->residency->requestBuffer, &clear.requestBuffer, cmd);
			device->CopyResource(&vt->residency->allocationBuffer, &clear.allocationBuffer, cmd);

			wi::renderer::PushBarrier(GPUBarrier::Image(&vt->residency->residencyMap, ResourceState::COPY_DST, ResourceState::COPY_SRC));
		}
		wi::renderer::FlushBarriers(cmd);
		wi::profiler::EndRange(gg_vtrange); // GGMAX 2.94c
		device->EventEnd(cmd);
	}

	ShaderTerrain Terrain::GetShaderTerrain() const
	{
		GraphicsDevice* device = GetDevice();

		ShaderTerrain shader_terrain;
		shader_terrain.init();

		auto it = chunks.find(center_chunk);
		if (it != chunks.end())
		{
			const auto& chunk_data = it->second;
			shader_terrain.chunk_size = chunk_half_width * 2 * chunk_scale;
			shader_terrain.center_chunk_pos.x = center_chunk.x * shader_terrain.chunk_size - shader_terrain.chunk_size * 0.5f;
			shader_terrain.center_chunk_pos.y = 0;
			shader_terrain.center_chunk_pos.z = center_chunk.z * shader_terrain.chunk_size - shader_terrain.chunk_size * 0.5f;
			shader_terrain.chunk_buffer = device->GetDescriptorIndex(&chunk_buffer, SubresourceType::SRV);
			shader_terrain.chunk_buffer_range = chunk_buffer_range;
			shader_terrain.min_height = bottomLevel;
			shader_terrain.max_height = topLevel;
		}

		return shader_terrain;
	}

	void Terrain::InvalidateProps()
	{
		for (auto it = chunks.begin(); it != chunks.end(); it++)
		{
			ChunkData& chunk_data = it->second;
			chunk_data.prop_density_current = 0;
		}
	}

	void Terrain::InvalidateChunksAtSpline(const wi::scene::SplineComponent& spline)
	{
		for (auto it = chunks.begin(); it != chunks.end(); it++)
		{
			ChunkData& chunk_data = it->second;
			if (chunk_data.invalidated)
				continue;
			BoundingBox bb(chunk_data.sphere.center, XMFLOAT3(chunk_data.sphere.radius, 1000000, chunk_data.sphere.radius));
			if (spline.bvh.IntersectsFirst(bb, [&](uint32_t index) { return spline.precomputed_obbs[index].Intersects(bb); }))
			{
				chunk_data.invalidated = true;
				generator->priority_invalidation.push_back(it->first);
			}
		}
	}

	void Terrain::Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri)
	{
		Generation_Cancel();

		// Note: separate component types serialized within terrain must NOT use the version of the terrain, but their own!
		ComponentLibrary& library = *seri.componentlibrary;
		const uint64_t terrain_version = seri.GetVersion();
		const uint64_t grass_version = seri.GetVersion("wi::scene::Scene::hairs");
		const uint64_t material_version = seri.GetVersion("wi::scene::Scene::materials");
		const uint64_t weather_version = seri.GetVersion("wi::scene::Scene::weathers");

		if (archive.IsReadMode())
		{
			archive >> _flags;
			archive >> lod_bias;
			if (terrain_version < 3)
			{
				float texlod;
				archive >> texlod;
			}
			archive >> generation;
			archive >> prop_generation;
			archive >> prop_density;
			archive >> grass_density;
			archive >> chunk_scale;
			archive >> seed;
			archive >> bottomLevel;
			archive >> topLevel;
			archive >> region1;
			archive >> region2;
			archive >> region3;

			archive >> center_chunk.x;
			archive >> center_chunk.z;

			if (terrain_version >= 1)
			{
				archive >> physics_generation;
			}
			if (terrain_version >= 2 && terrain_version < 3)
			{
				uint32_t target_texture_resolution;
				archive >> target_texture_resolution;
			}

			size_t count = 0;
			archive >> count;
			props.resize(count);
			for (size_t i = 0; i < props.size(); ++i)
			{
				Prop& prop = props[i];
				if (terrain_version >= 1)
				{
					archive >> prop.data;

					if (!prop.data.empty())
					{
						// Serialize the prop data in read mode and remap internal entity references:
						Scene tmp_scene;
						wi::Archive tmp_archive = wi::Archive(prop.data.data(), prop.data.size());
						Entity entity = tmp_scene.Entity_Serialize(
							tmp_archive,
							seri,
							INVALID_ENTITY,
							wi::scene::Scene::EntitySerializeFlags::RECURSIVE
						);

						// Serialize again with the remapped references:
						wi::Archive remapped_archive;
						remapped_archive.SetReadModeAndResetPos(false);
						tmp_scene.Entity_Serialize(
							remapped_archive,
							seri,
							entity,
							wi::scene::Scene::EntitySerializeFlags::RECURSIVE
						);

						// Replace original data with remapped references for current session:
						remapped_archive.WriteData(prop.data);
					}
				}
				else
				{
					// Back compat reading of terrain version 0:
					std::string name;
					Entity mesh_entity;
					ObjectComponent object;
					archive >> name;
					SerializeEntity(archive, mesh_entity, seri);
					object.Serialize(archive, seri);
					Scene tmp_scene;
					Entity object_entity = CreateEntity();
					tmp_scene.names.Create(object_entity) = name;
					tmp_scene.objects.Create(object_entity) = object;
					tmp_scene.transforms.Create(object_entity);
					wi::Archive archive;
					EntitySerializer seri;
					tmp_scene.Entity_Serialize(
						archive,
						seri,
						object_entity,
						wi::scene::Scene::EntitySerializeFlags::RECURSIVE | wi::scene::Scene::EntitySerializeFlags::KEEP_INTERNAL_ENTITY_REFERENCES
					);
					archive.WriteData(prop.data);
				}
				archive >> prop.min_count_per_chunk;
				archive >> prop.max_count_per_chunk;
				archive >> prop.region;
				archive >> prop.region_power;
				archive >> prop.noise_frequency;
				archive >> prop.noise_power;
				archive >> prop.threshold;
				archive >> prop.min_size;
				archive >> prop.max_size;
				archive >> prop.min_y_offset;
				archive >> prop.max_y_offset;
				if (terrain_version >= 6)
				{
					SerializeEntity(archive, prop.source_entity, seri);
				}
			}

			archive >> count;
			chunks.reserve(count);
			for (size_t i = 0; i < count; ++i)
			{
				Chunk chunk;
				archive >> chunk.x;
				archive >> chunk.z;
				ChunkData& chunk_data = chunks[chunk];
				SerializeEntity(archive, chunk_data.entity, seri);
				SerializeEntity(archive, chunk_data.grass_entity, seri);
				SerializeEntity(archive, chunk_data.props_entity, seri);
				archive >> chunk_data.prop_density_current;

				seri.version = grass_version;
				chunk_data.grass.Serialize(archive, seri);
				seri.version = terrain_version;

				archive >> chunk_data.grass_density_current;
				if (terrain_version >= 5)
				{
					size_t blendmapCount = 0;
					archive >> blendmapCount;
					chunk_data.blendmap_layers.resize(blendmapCount);
					for (size_t i = 0; i < chunk_data.blendmap_layers.size(); ++i)
					{
						archive >> chunk_data.blendmap_layers[i].pixels;
					}
					archive >> chunk_data.heightmap_data;
				}
				else
				{
					wi::vector<wi::Color> blendmap4;
					archive >> blendmap4;
					chunk_data.blendmap_layers.resize(4);
					for (size_t i = 0; i < 4; ++i)
					{
						chunk_data.blendmap_layers[i].pixels.resize(blendmap4.size());
					}
					for (size_t i = 0; i < blendmap4.size(); ++i)
					{
						chunk_data.blendmap_layers[0].pixels[i] = blendmap4[i].getR();
						chunk_data.blendmap_layers[1].pixels[i] = blendmap4[i].getG();
						chunk_data.blendmap_layers[2].pixels[i] = blendmap4[i].getB();
						chunk_data.blendmap_layers[3].pixels[i] = blendmap4[i].getA();
					}
				}
				archive >> chunk_data.sphere.center;
				archive >> chunk_data.sphere.radius;
				archive >> chunk_data.position;
			}

			archive >> count;
			modifiers.resize(count);
			for (size_t i = 0; i < modifiers.size(); ++i)
			{
				uint32_t value;
				archive >> value;
				Modifier::Type type = (Modifier::Type)value;

				switch (type)
				{
				default:
				case Modifier::Type::Perlin:
					{
						std::shared_ptr<PerlinModifier> modifier = std::make_shared<PerlinModifier>();
						modifiers[i] = modifier;
						archive >> modifier->octaves;
						archive >> modifier->seed;
						modifier->perlin_noise.Serialize(archive);
					}
					break;
				case Modifier::Type::Voronoi:
					{
						std::shared_ptr<VoronoiModifier> modifier = std::make_shared<VoronoiModifier>();
						modifiers[i] = modifier;
						archive >> modifier->fade;
						archive >> modifier->shape;
						archive >> modifier->falloff;
						archive >> modifier->perturbation;
						archive >> modifier->seed;
						modifier->perlin_noise.Serialize(archive);
					}
					break;
				case Modifier::Type::Heightmap:
					{
						std::shared_ptr<HeightmapModifier> modifier = std::make_shared<HeightmapModifier>();
						modifiers[i] = modifier;
						archive >> modifier->amount;
						archive >> modifier->data;
						archive >> modifier->width;
						archive >> modifier->height;
					}
					break;
				}

				Modifier* modifier = modifiers[i].get();
				archive >> value;
				modifier->blend = (Modifier::BlendMode)value;
				archive >> modifier->weight;
				archive >> modifier->frequency;
			}

		}
		else
		{
			archive << _flags;
			archive << lod_bias;
			if (terrain_version < 3)
			{
				float texlod = 1;
				archive << texlod;
			}
			archive << generation;
			archive << prop_generation;
			archive << prop_density;
			archive << grass_density;
			archive << chunk_scale;
			archive << seed;
			archive << bottomLevel;
			archive << topLevel;
			archive << region1;
			archive << region2;
			archive << region3;

			archive << center_chunk.x;
			archive << center_chunk.z;

			if (terrain_version >= 1)
			{
				archive << physics_generation;
			}
			if (terrain_version >= 2 && terrain_version < 3)
			{
				uint32_t target_texture_resolution = 1024;
				archive << target_texture_resolution;
			}

			archive << props.size();
			for (size_t i = 0; i < props.size(); ++i)
			{
				Prop& prop = props[i];
				archive << prop.data;
				archive << prop.min_count_per_chunk;
				archive << prop.max_count_per_chunk;
				archive << prop.region;
				archive << prop.region_power;
				archive << prop.noise_frequency;
				archive << prop.noise_power;
				archive << prop.threshold;
				archive << prop.min_size;
				archive << prop.max_size;
				archive << prop.min_y_offset;
				archive << prop.max_y_offset;
				SerializeEntity(archive, prop.source_entity, seri);
			}

			archive << chunks.size();
			for (auto& it : chunks)
			{
				const Chunk& chunk = it.first;
				archive << chunk.x;
				archive << chunk.z;
				ChunkData& chunk_data = it.second;
				SerializeEntity(archive, chunk_data.entity, seri);
				SerializeEntity(archive, chunk_data.grass_entity, seri);
				SerializeEntity(archive, chunk_data.props_entity, seri);
				archive << chunk_data.prop_density_current;

				seri.version = grass_version;
				chunk_data.grass.Serialize(archive, seri);
				seri.version = terrain_version;

				archive << chunk_data.grass_density_current;
				archive << chunk_data.blendmap_layers.size();
				for (size_t i = 0; i < chunk_data.blendmap_layers.size(); ++i)
				{
					archive << chunk_data.blendmap_layers[i].pixels;
				}
				archive << chunk_data.heightmap_data;
				archive << chunk_data.sphere.center;
				archive << chunk_data.sphere.radius;
				archive << chunk_data.position;
			}

			archive << modifiers.size();
			for (auto& modifier : modifiers)
			{
				archive << (uint32_t)modifier->type;
				switch (modifier->type)
				{
				default:
				case Modifier::Type::Perlin:
					archive << ((PerlinModifier*)modifier.get())->octaves;
					archive << ((PerlinModifier*)modifier.get())->seed;
					((PerlinModifier*)modifier.get())->perlin_noise.Serialize(archive);
					break;
				case Modifier::Type::Voronoi:
					archive << ((VoronoiModifier*)modifier.get())->fade;
					archive << ((VoronoiModifier*)modifier.get())->shape;
					archive << ((VoronoiModifier*)modifier.get())->falloff;
					archive << ((VoronoiModifier*)modifier.get())->perturbation;
					archive << ((VoronoiModifier*)modifier.get())->seed;
					((VoronoiModifier*)modifier.get())->perlin_noise.Serialize(archive);
					break;
				case Modifier::Type::Heightmap:
					archive << ((HeightmapModifier*)modifier.get())->amount;
					archive << ((HeightmapModifier*)modifier.get())->data;
					archive << ((HeightmapModifier*)modifier.get())->width;
					archive << ((HeightmapModifier*)modifier.get())->height;
					break;
				}

				archive << (uint32_t)modifier->blend;
				archive << modifier->weight;
				archive << modifier->frequency;
			}
		}

		// Caution: seri.version changes must be handled carefully!

		if (terrain_version >= 5)
		{
			if (archive.IsReadMode())
			{
				SerializeEntity(archive, chunkGroupEntity, seri);
				size_t materialCount = 0;
				archive >> materialCount;
				materialEntities.resize(materialCount);
				for (size_t i = 0; i < materialEntities.size(); ++i)
				{
					SerializeEntity(archive, materialEntities[i], seri);
				}
				SerializeEntity(archive, grassEntity, seri);
			}
			else
			{
				SerializeEntity(archive, chunkGroupEntity, seri);
				archive << materialEntities.size();
				for (size_t i = 0; i < materialEntities.size(); ++i)
				{
					SerializeEntity(archive, materialEntities[i], seri);
				}
				SerializeEntity(archive, grassEntity, seri);
			}
		}
		else if (terrain_version >= 4)
		{
			SerializeEntity(archive, chunkGroupEntity, seri);
			for (size_t i = 0; i < MATERIAL_COUNT; ++i)
			{
				SerializeEntity(archive, materialEntities[i], seri);
			}
			SerializeEntity(archive, grassEntity, seri);
		}
		else
		{
			// Convert terrain version below 4 to newer version:
			seri.version = material_version;
			wi::scene::MaterialComponent materials[MATERIAL_COUNT];
			materials[MATERIAL_BASE].Serialize(archive, seri);
			materials[MATERIAL_SLOPE].Serialize(archive, seri);
			materials[MATERIAL_LOW_ALTITUDE].Serialize(archive, seri);
			materials[MATERIAL_HIGH_ALTITUDE].Serialize(archive, seri);
			grass_material.Serialize(archive, seri);
			wi::jobsystem::Wait(seri.ctx); // wait for material CreateRenderData() that was asynchronously launched in Serialize()!

			materialEntities[MATERIAL_BASE] = CreateEntity();
			materialEntities[MATERIAL_SLOPE] = CreateEntity();
			materialEntities[MATERIAL_LOW_ALTITUDE] = CreateEntity();
			materialEntities[MATERIAL_HIGH_ALTITUDE] = CreateEntity();
			grassEntity = CreateEntity();

			ComponentManager<MaterialComponent>* scene_materials = library.Get<MaterialComponent>("wi::scene::Scene::materials");
			scene_materials->Create(materialEntities[MATERIAL_BASE]) = materials[MATERIAL_BASE];
			scene_materials->Create(materialEntities[MATERIAL_SLOPE]) = materials[MATERIAL_SLOPE];
			scene_materials->Create(materialEntities[MATERIAL_LOW_ALTITUDE]) = materials[MATERIAL_LOW_ALTITUDE];
			scene_materials->Create(materialEntities[MATERIAL_HIGH_ALTITUDE]) = materials[MATERIAL_HIGH_ALTITUDE];
			scene_materials->Create(grassEntity) = grass_material;

			ComponentManager<NameComponent>* scene_names = library.Get<NameComponent>("wi::scene::Scene::names");
			scene_names->Create(materialEntities[MATERIAL_BASE]).name = "material_base";
			scene_names->Create(materialEntities[MATERIAL_SLOPE]).name = "material_slope";
			scene_names->Create(materialEntities[MATERIAL_LOW_ALTITUDE]).name = "material_low_altitude";
			scene_names->Create(materialEntities[MATERIAL_HIGH_ALTITUDE]).name = "material_high_altitude";
			scene_names->Create(grassEntity).name = "grass";
		}

		seri.version = weather_version;
		weather.Serialize(archive, seri);

		if (terrain_version < 4)
		{
			seri.version = grass_version;
			grass_properties.Serialize(archive, seri);
			seri.version = terrain_version;

			ComponentManager<wi::HairParticleSystem>* scene_hairs = library.Get<wi::HairParticleSystem>("wi::scene::Scene::hairs");
			scene_hairs->Create(grassEntity) = grass_properties;
		}

		perlin_noise.Serialize(archive);
	}

}
