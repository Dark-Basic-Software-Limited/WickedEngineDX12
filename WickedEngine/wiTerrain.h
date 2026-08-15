#pragma once
#include "CommonInclude.h"
#include "wiScene_Decl.h"
#include "wiScene_Components.h"
#include "wiNoise.h"
#include "wiECS.h"
#include "wiColor.h"
#include "wiHairParticle.h"
#include "wiVector.h"

#include <memory>
#include <functional>
#include <atomic> // GGMAX 2.58: genprof accumulators

namespace wi::terrain
{
	struct Chunk
	{
		union
		{
			struct
			{
				int32_t x, z;
			};
			uint64_t raw = 0;
		};
		constexpr bool operator==(const Chunk& other) const
		{
			return raw == other.raw;
		}
		constexpr uint64_t compute_hash() const
		{
			return raw;
		}
	};

	// GGMAX 2.53: generation-center override — while enabled, Generation_Update rings the
	// chunk set around this world-space XZ instead of the camera eye. The Terrain Generator
	// pins it to the editable-area marker so flying the preview camera cannot drag generated
	// chunks away from the playable area; every other mode leaves it disabled (camera-centred).
	// Deliberately namespace globals, NOT Terrain members: transient UI-mode state must not
	// ride Serialize / Entity_Duplicate. Plain floats to avoid header dependencies.
	extern float gg_generation_center_override_x;
	extern float gg_generation_center_override_z;
	extern bool  gg_generation_center_override_enabled;
	extern bool gg_generation_skip_bvh; // GGMAX 2.58: generator-only, see wiTerrain.cpp

	// GGMAX 2.58 diagnostic: per-phase chunk-generation cost accumulators (us, cumulative).
	extern std::atomic<uint64_t> gg_genprof_heights_us;
	extern std::atomic<uint64_t> gg_genprof_vertex_us;
	extern std::atomic<uint64_t> gg_genprof_renderdata_us;
	extern std::atomic<uint64_t> gg_genprof_bvh_us;
	extern std::atomic<uint64_t> gg_genprof_grass_us;
	extern std::atomic<uint64_t> gg_genprof_blendcb_us;
	extern std::atomic<uint64_t> gg_genprof_regiontex_us;
	extern std::atomic<uint64_t> gg_genprof_physics_us;
	extern std::atomic<uint64_t> gg_genprof_total_us;
	extern std::atomic<uint64_t> gg_genprof_chunks;
}

namespace std
{
	template <>
	struct hash<wi::terrain::Chunk>
	{
		constexpr uint64_t operator()(const wi::terrain::Chunk& chunk) const
		{
			return chunk.compute_hash();
		}
	};
}

namespace wi::terrain
{
	// GGMAX 1.71: SVT physical atlas height — the fixed backing cost of the terrain virtual
	// texture (16384 = stock, 768 MB tile pool; halving halves it). Applied at atlas creation.
	extern uint32_t gg_svt_atlas_height;
	extern bool gg_svt_keep_emissive; // GGMAX 1.98 (A4): setup.ini svtemissive=1 restores the unused 4th SVT map

	static constexpr int chunk_width = 64 + 3; // + 3: filler vertices for lod apron and grid perimeter
	static constexpr float chunk_half_width = (chunk_width - 1) * 0.5f;
	static constexpr float chunk_width_rcp = 1.0f / (chunk_width - 1);
	static constexpr uint32_t vertexCount = chunk_width * chunk_width;
	enum
	{
		MATERIAL_BASE,
		MATERIAL_SLOPE,
		MATERIAL_LOW_ALTITUDE,
		MATERIAL_HIGH_ALTITUDE,
		MATERIAL_COUNT
	};

	struct VirtualTexture; // GGMAX 1.33: fwd-decl for PhysicalTile::gg_owner

	// GGMAX 1.33: master switch for the incremental virtual-texture bookkeeping
	//	(dirty-tracked page-table uploads + lazy free-list rebuild). false = stock
	//	every-frame behaviour (A/B and emergency fallback). Defined in wiTerrain.cpp.
	extern bool gg_vt_incremental;

	struct VirtualTextureAtlas
	{
		struct Map
		{
			wi::graphics::Texture texture;
			wi::graphics::Texture texture_raw_block;
		};
		Map maps[4];
		wi::graphics::GPUBuffer tile_pool;

		uint8_t physical_tile_count_x = 0;
		uint8_t physical_tile_count_y = 0;

		struct Tile
		{
			uint8_t x = 0xFF;
			uint8_t y = 0xFF;
			constexpr bool IsValid() const
			{
				return x != 0xFF && y != 0xFF;
			}
			constexpr operator uint16_t() const { return uint16_t(uint16_t(x) | (uint16_t(y) << 8u)); }
		};
		wi::vector<Tile> free_tiles;

		struct PhysicalTile
		{
			const Tile* last_used = nullptr;
			uint64_t free_frames = 0;
			// GGMAX 1.33: which VirtualTexture currently owns this physical tile. Used ONLY
			// to mark the VICTIM's page table dirty when a tile is stolen — a stale pointer
			// is impossible because VirtualTexture::free() clears it before the VT dies.
			VirtualTexture* gg_owner = nullptr;
		};
		wi::vector<PhysicalTile> physical_tiles;

		// GGMAX 1.33: set whenever allocate_tile consumes from free_tiles (list shrank /
		// residency changed) so the async job knows the cached free list must be rebuilt.
		bool gg_free_dirty = true;

		struct Residency
		{
			wi::graphics::Texture feedbackMap;
			wi::graphics::Texture residencyMap;
			wi::graphics::GPUBuffer requestBuffer;
			wi::graphics::GPUBuffer allocationBuffer;
			wi::graphics::GPUBuffer allocationBuffer_CPU_readback[wi::graphics::GraphicsDevice::GetBufferCount()];
			wi::graphics::GPUBuffer pageBuffer;
			wi::graphics::GPUBuffer pageBuffer_CPU_upload[wi::graphics::GraphicsDevice::GetBufferCount()];
			bool data_available_CPU[wi::graphics::GraphicsDevice::GetBufferCount()] = {};
			int16_t cpu_resource_id = 0;
			uint32_t resolution = 0;

			void init(uint32_t resolution);
			void reset();
		};
		wi::unordered_map<uint32_t, wi::vector<wi::allocator::shared_ptr<Residency>>> free_residencies; // per resolution residencies

		// GGMAX 1.33: owner-aware allocation. gg_new_owner = the VT this tile is being mapped
		// into; if the physical tile was owned by a DIFFERENT live VT (a steal), that victim is
		// reported through gg_stolen_from so the caller can mark its page table dirty (its
		// mapping just silently broke — check_tile_resident will fail for it from now on).
		bool allocate_tile(Tile& tile, VirtualTexture* gg_new_owner = nullptr, VirtualTexture** gg_stolen_from = nullptr)
		{
			if (free_tiles.empty())
				return false;
			tile = free_tiles.back();
			free_tiles.pop_back();
			gg_free_dirty = true; // GGMAX 1.33
			PhysicalTile& physical_tile = physical_tiles[tile.x + tile.y * physical_tile_count_x];
			if (gg_stolen_from != nullptr)
			{
				*gg_stolen_from = (physical_tile.gg_owner != nullptr && physical_tile.gg_owner != gg_new_owner) ? physical_tile.gg_owner : nullptr;
			}
			physical_tile.gg_owner = gg_new_owner; // GGMAX 1.33
			physical_tile.last_used = &tile;
			physical_tile.free_frames = 0;
			return true;
		}
		bool request_residency(const Tile& tile)
		{
			if (check_tile_resident(tile))
			{
				physical_tiles[tile.x + tile.y * physical_tile_count_x].free_frames = 0;
				return true;
			}
			return false;
		}
		bool check_tile_resident(const Tile& tile) const
		{
			if (!tile.IsValid())
				return false;
			return physical_tiles[tile.x + tile.y * physical_tile_count_x].last_used == &tile;
		}
		uint64_t get_tile_frames(const Tile& tile) const
		{
			if (!tile.IsValid())
				return ~0ull;
			return physical_tiles[tile.x + tile.y * physical_tile_count_x].free_frames;
		}
		wi::allocator::shared_ptr<Residency> allocate_residency(uint32_t resolution)
		{
			if (free_residencies[resolution].empty())
			{
				wi::allocator::shared_ptr<Residency> residency = wi::allocator::make_shared<Residency>();
				residency->init(resolution);
				free_residencies[resolution].push_back(residency);
			}
			wi::allocator::shared_ptr<Residency> residency = free_residencies[resolution].back();
			free_residencies[resolution].pop_back();
			residency->reset();
			return residency;
		}
		void free_residency(wi::allocator::shared_ptr<Residency>& residency)
		{
			if (residency == nullptr)
				return;
			free_residencies[residency->resolution].push_back(residency);
			residency = {};
		}
		inline bool IsValid() const
		{
			return tile_pool.IsValid();
		}
	};

	struct VirtualTexture
	{
		wi::allocator::shared_ptr<VirtualTextureAtlas::Residency> residency;
		wi::vector<VirtualTextureAtlas::Tile> tiles;
		uint32_t lod_count = 0;
		uint32_t resolution = 0;

		void init(VirtualTextureAtlas& atlas, uint resolution);

		void free(VirtualTextureAtlas& atlas)
		{
			// GGMAX: release physical-tile ownership BEFORE the tiles vector is destroyed.
			// physical_tiles[].last_used stores raw pointers into this vector; a later
			// tiles allocation can reuse the same heap block, making the stale last_used
			// spuriously match a brand-new tile by address. check_tile_resident then
			// reports tiles this VT never allocated as resident: the page table maps
			// recycled physical tiles holding another chunk's pixels AND the re-render is
			// skipped — random foreign squares during fast zoom in/out cycles.
			for (auto& tile : tiles)
			{
				if (!tile.IsValid())
					continue;
				VirtualTextureAtlas::PhysicalTile& physical_tile = atlas.physical_tiles[tile.x + tile.y * atlas.physical_tile_count_x];
				if (physical_tile.last_used == &tile)
				{
					physical_tile.last_used = nullptr;
				}
				// GGMAX 1.33: this VT is going away — physical tiles must not keep a
				// dangling owner pointer (it is dereferenced on steal to mark dirty).
				if (physical_tile.gg_owner == this)
				{
					physical_tile.gg_owner = nullptr;
				}
			}
			tiles.clear();
			atlas.free_residency(residency);
		}

		void invalidate()
		{
			resolution = 0;
		}

		// GGMAX: set when the chunk's blendmap texture was rebuilt after a paint/blend
		// edit — the next UpdateVirtualTexturesCPU rebinds the blendmap and re-renders
		// every currently RESIDENT tile in place. Unlike invalidate(), this keeps the
		// residency intact, so the refresh lands next frame instead of re-streaming
		// the whole chunk through multi-frame GPU feedback round-trips.
		// Ownership: game code only SETS this; UpdateVirtualTexturesCPU's main-thread
		// loop (which runs after the previous frame's job was joined) consumes it into
		// gg_repaint_blendmap_latched together with the vt.blendmap re-bind; only the
		// async job reads/clears the latch. The job must never touch this live flag —
		// a job in flight while game code sets it would consume the request against
		// the OLD blendmap binding and the edit would silently never land.
		bool pending_repaint_blendmap = false;
		bool gg_repaint_blendmap_latched = false;

		// GGMAX 1.33: incremental page-table upload state. gg_page_dirty = the CPU-side page
		// table (tiles[] residency mapping) changed since the last upload-buffer write; set at
		// creation, on every allocation into this VT, and on every steal FROM this VT. The
		// async VT job consumes it (writes the upload buffer, sets the two pending flags).
		// Each pending flag has exactly one consumer on its own command list:
		//	gg_page_upload_pending      -> CopyVirtualTexturePageStatusGPU (copy to GPU pageBuffer)
		//	gg_residency_update_pending -> UpdateVirtualTexturesGPU (residency map recompute)
		// All are mutable because the recording functions are const.
		mutable bool gg_page_dirty = true;
		mutable bool gg_page_upload_pending = false;
		mutable bool gg_residency_update_pending = false;

		struct AllocationRequest
		{
			uint32_t x = 0;
			uint32_t y = 0;
			uint32_t lod = 0;
			uint32_t tile_index = 0;
		};
		wi::vector<AllocationRequest> allocation_requests;

		// Attach this data to Virtual Texture because we will record these by separate CPU thread:
		struct UpdateRequest
		{
			uint16_t x = 0;
			uint16_t y = 0;
			uint16_t lod = 0;
			uint8_t tile_x = 0;
			uint8_t tile_y = 0;
		};
		mutable wi::vector<UpdateRequest> update_requests;
		wi::graphics::Texture blendmap;
	};

	struct BlendmapLayer
	{
		wi::vector<uint8_t> pixels;
	};

	struct ChunkData
	{
		wi::ecs::Entity entity = wi::ecs::INVALID_ENTITY;
		wi::ecs::Entity grass_entity = wi::ecs::INVALID_ENTITY;
		wi::ecs::Entity props_entity = wi::ecs::INVALID_ENTITY;
		float prop_density_current = 1;
		float grass_density_current = 1;
		const XMFLOAT3* mesh_vertex_positions = nullptr;
		wi::HairParticleSystem grass;
		wi::vector<BlendmapLayer> blendmap_layers;
		wi::vector<BlendmapLayer> spline_blendmap_layers;
		wi::graphics::Texture blendmap;
		wi::primitive::Sphere sphere;
		XMFLOAT3 position = XMFLOAT3(0, 0, 0);
		bool visible = true;
		bool invalidated = false;
		// GGMAX: set by the generator when it (re)generates this chunk, cleared when the
		// result is merged into the main scene. Between those two points the main-scene
		// mesh is the STALE pre-regeneration version — consumers that bake data from the
		// chunk mesh (GG blendmap passes) must skip the chunk while this is up.
		bool merge_pending = false;
		// GGMAX: set when this chunk's blendmap was filled by Terrain::gg_generate_blendmap
		// on the generator thread (born game-correct, GPU texture already built from it).
		// Game blend passes latch their processed keys and skip the rewrite while this is
		// up; the game's edit bridge clears it so real edits reprocess normally.
		bool gg_blendmap_generated = false;
		wi::allocator::shared_ptr<VirtualTexture> vt;
		wi::vector<uint16_t> heightmap_data;
		wi::graphics::Texture heightmap;

		void enable_blendmap_layer(size_t materialIndex)
		{
			while (blendmap_layers.size() < materialIndex + 1)
			{
				blendmap_layers.emplace_back().pixels.resize(vertexCount);
			}
		}
	};

	struct Prop
	{
		wi::vector<uint8_t> data; // serialized component data storage
		wi::ecs::Entity source_entity = wi::ecs::INVALID_ENTITY; // the original entity that this prop was serialized from
		int min_count_per_chunk = 0; // a chunk will try to generate min this many props of this type
		int max_count_per_chunk = 10; // a chunk will try to generate max this many props of this type
		int region = 0; // region selection in range [0,3] (0: base/grass, 1: slopes, 2: low altitude (bottom level-0), 3: high altitude (0-top level))
		float region_power = 1; // region weight affection power factor
		float noise_frequency = 1; // perlin noise's frequency for placement factor
		float noise_power = 1; // perlin noise's power
		float threshold = 0.5f; // the chance of placement (higher is less chance)
		float min_size = 1; // scaling randomization range min
		float max_size = 1; // scaling randomization range max
		float min_y_offset = 0; // min randomized offset on Y axis
		float max_y_offset = 0; // max randomized offset on Y axis
	};

	struct Modifier;
	struct Generator;

	struct Terrain
	{
		enum FLAGS
		{
			EMPTY = 0,
			CENTER_TO_CAM = 1 << 0,
			REMOVAL = 1 << 1,
			GRASS = 1 << 2,
			GENERATION_STARTED = 1 << 4,
			PHYSICS = 1 << 5,
			TESSELLATION = 1 << 6,
		};
		uint32_t _flags = CENTER_TO_CAM | REMOVAL | GRASS;

		wi::ecs::Entity terrainEntity = wi::ecs::INVALID_ENTITY;
		wi::ecs::Entity chunkGroupEntity = wi::ecs::INVALID_ENTITY;
		wi::scene::Scene* scene = nullptr;
		wi::vector<wi::ecs::Entity> materialEntities;
		wi::vector<wi::ecs::Entity> splineMaterialEntities;
		wi::ecs::Entity grassEntity = wi::ecs::INVALID_ENTITY;
		wi::scene::WeatherComponent weather;
		wi::HairParticleSystem grass_properties;
		wi::scene::MaterialComponent grass_material;
		wi::unordered_map<Chunk, ChunkData> chunks;
		Chunk center_chunk;
		wi::noise::Perlin perlin_noise;
		wi::vector<Prop> props;
		int grass_chunk_dist = 1;

		// For generating scene on a background thread:
		float generation_time_budget_milliseconds = 8; // after this much time, the generation thread will start to exit. This can help avoid a very long running, resource consuming and slow cancellation generation
		bool generation_view_cone_priority = false; // GGMAX: when enabled, chunks within the camera's horizontal view cone are generated before the rest of the spiral (the visible terrain builds first)
		bool generation_restart_on_dirty_materials = true; // GGMAX: stock Wicked restarts generation (full chunk teardown) whenever a terrain material is dirty — an editor convenience. GG creates/updates terrain materials at runtime (incremental painted-slot registration) and owns the blendmaps itself, so a dirty material must NOT rebuild the island (it caused a 4-5s full-terrain flicker on the first paint stroke with each new texture)
		bool gg_preserve_blendmap_on_regen = false; // GGMAX: in-place chunk regeneration (sculpt/edit invalidation) keeps the chunk's existing blendmap layers, GPU blendmap texture and virtual-texture residency instead of regenerating engine-default region weights and resetting the VT. GG owns the blendmaps (its DX11-style passes rewrite them right after regen) — without this, every sculpt-drag frame flashed the engine's 4-region default blend and re-streamed all tiles (chunk-shaped blur/checker until mouse release)
		// GGMAX: when set, freshly generated (non-preserved) chunks get their blendmap filled
		// by this callback on the generator thread, right after the chunk's vertex data is
		// complete and before the region texture is created — streamed-in chunks are born
		// with game-correct blending instead of flashing the engine-default region weights
		// until the game's main-thread blend passes catch up. Return false to fall back to
		// the engine-default weights (e.g. game data not ready during initial level load).
		std::function<bool(ChunkData& chunk_data, const wi::scene::MeshComponent& mesh)> gg_generate_blendmap;
		// GGMAX: FREEZE virtual-texture resolution changes (both upgrades and downgrades)
		// while the camera is crossing chunk boundaries. Without this, a fast camera zoom
		// sweeps the dist<2 ring across the terrain and every crossing chunk re-inits its
		// VT mid-motion — residency churn visible as square tiles of mixed sharpness/stale
		// content flickering until the camera settles. Frozen chunks keep rendering their
		// existing (correct) tiles at their current resolution. Once the camera holds one
		// chunk for 10 frames: downgrades run immediately (frees tile pool), upgrades run
		// budgeted per frame. Fresh (resolution 0) and unbound-material chunks are never
		// deferred. Default false = stock.
		bool gg_vt_upgrade_hysteresis = false;
		wi::terrain::Chunk gg_prev_center_chunk = {};
		uint32_t gg_center_stable_frames = 0;
		// GGMAX 1.33 (review F3): per-Terrain state for the incremental-VT job — members, not
		// lambda statics, so multiple Terrain instances never share/race them.
		uint32_t gg_freesort_cooldown = 0;   // rate-limits the low-water free-list rebuild
		uint32_t gg_vt_frame_counter = 0;    // drives the rotating page-buffer heartbeat
		// GGMAX 1.33 (review F2): center chunk at the last free-list rebuild — a center change
		// (teleport/zoom) forces a rebuild so main-thread chunk-init never pops from a list
		// whose keep-alive info predates the move.
		wi::terrain::Chunk gg_freesort_last_center = {};
		// GGMAX: radius (in chunks) of the full-resolution virtual-texture zone. Stock = 2:
		// a tiny ring the camera crosses in milliseconds, forcing VT re-inits/re-streams on
		// every fast move. A bounded game island can sit entirely INSIDE a wider zone —
		// then camera travel never crosses a resolution boundary at all and the VT cache
		// for near, correct terrain is never re-referenced (tile residency stays feedback-
		// driven, so the physical pool cost barely changes).
		int gg_near_ring_dist = 2;
		// GGMAX: extra chunks beyond the stock removal threshold (generation + 2) before a
		// chunk is destroyed. Bounded islands should keep their chunks alive across zoom
		// travel — destruction/recreation churn is a VT cache flush. Default 0 = stock.
		int gg_removal_margin = 0;
		bool generation_high_priority = false; // GGMAX: run the generation job + its per-chunk dispatches on the HIGH priority job pool (Low pool threads are THREAD_PRIORITY_LOWEST and starve while the CPU is busy, e.g. during level load). Set only for burst scenarios like an initial build.
		std::shared_ptr<Generator> generator;

		wi::vector<VirtualTexture*> virtual_textures_in_use;
		wi::graphics::Sampler sampler;
		VirtualTextureAtlas atlas;

		wi::graphics::GPUBuffer chunk_buffer;
		int chunk_buffer_range = 3; // how many chunks to upload to GPU in X and Z directions

		constexpr bool IsCenterToCamEnabled() const { return _flags & CENTER_TO_CAM; }
		constexpr bool IsRemovalEnabled() const { return _flags & REMOVAL; }
		constexpr bool IsGrassEnabled() const { return _flags & GRASS; }
		constexpr bool IsGenerationStarted() const { return _flags & GENERATION_STARTED; }
		constexpr bool IsPhysicsEnabled() const { return _flags & PHYSICS; }
		constexpr bool IsTessellationEnabled() const { return _flags & TESSELLATION; }

		constexpr void SetCenterToCamEnabled(bool value) { if (value) { _flags |= CENTER_TO_CAM; } else { _flags &= ~CENTER_TO_CAM; } }
		constexpr void SetRemovalEnabled(bool value) { if (value) { _flags |= REMOVAL; } else { _flags &= ~REMOVAL; } }
		constexpr void SetGrassEnabled(bool value) { if (value) { _flags |= GRASS; } else { _flags &= ~GRASS; } }
		constexpr void SetGenerationStarted(bool value) { if (value) { _flags |= GENERATION_STARTED; } else { _flags &= ~GENERATION_STARTED; } }
		constexpr void SetPhysicsEnabled(bool value) { if (value) { _flags |= PHYSICS; } else { _flags &= ~PHYSICS; } }
		constexpr void SetTessellationEnabled(bool value) { if (value) { _flags |= TESSELLATION; } else { _flags &= ~TESSELLATION; } }

		float lod_bias = 0;
		int generation = 12;
		int prop_generation = 10;
		int physics_generation = 3;
		float prop_density = 1;
		float grass_density = 1;
		float chunk_scale = 1;
		uint32_t seed = 3926;
		float bottomLevel = -60;
		float topLevel = 380;
		float region1 = 1;
		float region2 = 2;
		float region3 = 8;

		wi::vector<std::shared_ptr<Modifier>> modifiers;
		wi::vector<Modifier*> modifiers_to_remove;

		Terrain();
		~Terrain();

		// Restarts the terrain generation from scratch
		//	This will remove previously existing terrain
		void Generation_Restart();
		// This will run the actual generation tasks, call it once per frame
		void Generation_Update(const wi::scene::CameraComponent& camera);
		// Tells the generation thread that it should be cancelled and blocks until that is confirmed
		void Generation_Cancel();
		// Creates the textures for a chunk data
		void CreateChunkRegionTexture(ChunkData& chunk_data);

		void UpdateVirtualTexturesCPU();
		void UpdateVirtualTexturesGPU(wi::graphics::CommandList cmd) const;
		void CopyVirtualTexturePageStatusGPU(wi::graphics::CommandList cmd) const;
		void AllocateVirtualTextureTileRequestsGPU(wi::graphics::CommandList cmd) const;
		void WritebackTileRequestsGPU(wi::graphics::CommandList cmd) const;

		ShaderTerrain GetShaderTerrain() const;

		void InvalidateProps();

		void InvalidateChunksAtSpline(const wi::scene::SplineComponent& spline);

		void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);

	private:
		wi::vector<wi::scene::MaterialComponent> materials; // temp storage allocation
		float chunk_scale_rcp = 1.0f / chunk_scale;
	};

	struct Modifier
	{
		virtual ~Modifier() = default;

		enum class Type
		{
			Perlin,
			Voronoi,
			Heightmap,
		} type = Type::Perlin;

		enum class BlendMode
		{
			Normal,
			Additive,
			Multiply,
		} blend = BlendMode::Normal;

		float weight = 0.5f;
		float frequency = 0.0008f;

		// helpers for more user friendly setup with scaling in world space:
		constexpr void SetScale(float scale) { frequency = 1.0f / scale; }
		constexpr float GetScale() const { return 1.0f / frequency; }

		virtual void Seed(uint32_t seed) {}
		virtual void Apply(const XMFLOAT2& world_pos, float& height) = 0;
		constexpr void Blend(float& height, float value)
		{
			switch (blend)
			{
			default:
			case BlendMode::Normal:
				height = wi::math::Lerp(height, value, weight);
				break;
			case BlendMode::Multiply:
				height *= value * weight;
				break;
			case BlendMode::Additive:
				height += value * weight;
				break;
			}
		}
	};
	struct PerlinModifier : public Modifier
	{
		int octaves = 6;
		uint32_t seed = 0;
		wi::noise::Perlin perlin_noise;

		PerlinModifier() { type = Type::Perlin; }
		void Seed(uint32_t seed) override
		{
			this->seed = seed;
			perlin_noise.init(seed);
		}
		void Apply(const XMFLOAT2& world_pos, float& height) override
		{
			XMFLOAT2 p = world_pos;
			p.x *= frequency;
			p.y *= frequency;
			Blend(height, perlin_noise.compute(p.x, p.y, 0, octaves) * 0.5f + 0.5f);
		}
	};
	struct VoronoiModifier : public Modifier
	{
		float fade = 2.59f;
		float shape = 0.7f;
		float falloff = 6;
		float perturbation = 0.1f;
		uint32_t seed = 0;
		wi::noise::Perlin perlin_noise;

		VoronoiModifier() { type = Type::Voronoi; }
		void Seed(uint32_t seed) override
		{
			this->seed = seed;
			perlin_noise.init(seed);
		}
		void Apply(const XMFLOAT2& world_pos, float& height) override
		{
			XMFLOAT2 p = world_pos;
			p.x *= frequency;
			p.y *= frequency;
			if (perturbation > 0)
			{
				const float angle = perlin_noise.compute(p.x, p.y, 0, 6) * XM_2PI;
				p.x += std::sin(angle) * perturbation;
				p.y += std::cos(angle) * perturbation;
			}
			wi::noise::voronoi::Result res = wi::noise::voronoi::compute(p.x, p.y, (float)seed);
			float weight = std::pow(1 - saturate((res.distance - shape) * fade), std::max(0.0001f, falloff));
			Blend(height, weight);
		}
	};
	struct HeightmapModifier : public Modifier
	{
		float amount = 0.1f; // multiplier for height values

		wi::vector<uint8_t> data;
		int width = 0;
		int height = 0;

		HeightmapModifier() { type = Type::Heightmap; SetScale(1.0f); }
		void Apply(const XMFLOAT2& world_pos, float& height) override
		{
			XMFLOAT2 p = world_pos;
			p.x *= frequency;
			p.y *= frequency;
			XMFLOAT2 pixel = XMFLOAT2(p.x + this->width * 0.5f, p.y + this->height * 0.5f);
			if (pixel.x >= 0 && pixel.x < this->width && pixel.y >= 0 && pixel.y < this->height)
			{
				const int idx = int(pixel.x) + int(pixel.y) * this->width;
				float value = 0;
				if (data.size() == this->width * this->height * sizeof(uint8_t))
				{
					value = ((float)data[idx] / 255.0f);
				}
				else if (data.size() == this->width * this->height * sizeof(uint16_t))
				{
					value = ((float)((uint16_t*)data.data())[idx] / 65535.0f);
				}
				Blend(height, value * amount);
			}
		}
	};

}
