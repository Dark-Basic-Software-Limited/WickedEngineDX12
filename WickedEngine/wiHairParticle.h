#pragma once
#include "CommonInclude.h"
#include "wiGraphicsDevice.h"
#include "wiEnums.h"
#include "wiMath.h"
#include "wiECS.h"
#include "wiPrimitive.h"
#include "wiVector.h"
#include "wiScene_Decl.h"
#include "wiScene_Components.h"

namespace wi
{
	class Archive;
}

namespace wi
{
	class HairParticleSystem
	{
	public:
		wi::graphics::GPUBuffer constantBuffer;
		wi::graphics::GPUBuffer generalBuffer;
		wi::allocator::PageAllocator::Allocation generalBufferOffsetAllocation;
		wi::graphics::GPUBuffer generalBufferOffsetAllocationAlias;
		wi::scene::MeshComponent::BufferView simulation_view;
		wi::scene::MeshComponent::BufferView vb_pos[2];
		wi::scene::MeshComponent::BufferView vb_nor;
		wi::scene::MeshComponent::BufferView vb_pos_raytracing;
		wi::scene::MeshComponent::BufferView vb_uvs;
		wi::scene::MeshComponent::BufferView wetmap;
		wi::scene::MeshComponent::BufferView ib_culled;
		wi::scene::MeshComponent::BufferView indirect_view;
		wi::scene::MeshComponent::BufferView prim_view;

		wi::graphics::GPUBuffer indexBuffer;
		wi::graphics::GPUBuffer vertexBuffer_length;

		wi::graphics::RaytracingAccelerationStructure BLAS;

		void CreateFromMesh(const wi::scene::MeshComponent& mesh);
		void DeleteRenderData();
		void CreateRenderData();
		void CreateRaytracingRenderData();

		// GG-MAX: rebuild ONLY the vertexBuffer_length GPU buffer from the current vertex_lengths
		// data. Leaves generalBuffer (and so simulation_view, vb_pos, vb_nor, …) intact, so the
		// per-strand simulation state stays alive across the update — no settling pop on the sway
		// animation. Intended for in-place paint updates where strandCount / index list / mesh have
		// NOT changed; only which vertices are "painted" has. CreateRenderData() must have been
		// called previously to set up the rest of the GPU state.
		void UpdateVertexLengthsBuffer();

		void UpdateCPU(
			const wi::scene::TransformComponent& transform,
			const wi::scene::MeshComponent& mesh,
			float dt
		);

		static void InitializeGPUBuffersIfNeeded(
			const HairParticleSystem* hairs,
			size_t hairCount,
			wi::graphics::CommandList cmd
		);

		struct UpdateGPUItem
		{
			const HairParticleSystem* hair = nullptr;
			uint32_t instanceIndex = 0;
			const wi::scene::MeshComponent* mesh = nullptr;
			const wi::scene::MaterialComponent* material = nullptr;
		};
		// Update a batch of hair particles by GPU
		static void UpdateGPU(
			const UpdateGPUItem* items,
			uint32_t itemCount,
			wi::graphics::CommandList cmd
		);

		void Draw(
			const wi::scene::MaterialComponent& material,
			wi::enums::RENDERPASS renderPass,
			wi::graphics::CommandList cmd
		) const;

		wi::ecs::Entity meshID = wi::ecs::INVALID_ENTITY;

		enum FLAGS
		{
			EMPTY = 0,
			_DEPRECATED_REGENERATE_FRAME = 1 << 0,
			REBUILD_BUFFERS = 1 << 1,
			DIRTY = 1 << 2,
			CAMERA_BEND_ENABLED = 1 << 3,
		};
		uint32_t _flags = CAMERA_BEND_ENABLED;

		uint32_t strandCount = 0;
		uint32_t segmentCount = 1;
		uint32_t billboardCount = 1;
		uint32_t randomSeed = 1;
		float length = 1.0f;
		float stiffness = 0.5f;
		float drag = 0.1f;
		float gravityPower = 0;
		float randomness = 0.2f;
		float viewDistance = 200;
		wi::vector<float> vertex_lengths;
		float width = 1;
		float uniformity = 1;

		struct AtlasRect
		{
			XMFLOAT4 texMulAdd = XMFLOAT4(1, 1, 0, 0);
			float size = 1;
		};
		wi::vector<AtlasRect> atlas_rects;

		// GG-MAX Stage 3 Option B (non-serialized): per-strand visibility from a world-space
		// paint mask. Non-zero grass_type opts in — the simulate CS samples the bound mask at
		// the strand's world XZ and zeros strand_length when the cell doesn't match. Default
		// grass_type = 0 disables the feature entirely, leaving upstream behavior unchanged.
		// grass_visibility_texture is per-entity so multiple hair systems can each point at
		// their own mask; nullptr falls through to a 1x1 placeholder bound by Wicked.
		uint32_t grass_type = 0;
		float grass_map_inv_world_size = 0.0f;
		float grass_map_origin_x = 0.0f;
		float grass_map_origin_z = 0.0f;
		const wi::graphics::Texture* grass_visibility_texture = nullptr;

		// GG-MAX Stage B.10 (non-serialized): altitude filter — only active when grass_type != 0.
		// Above water strands must sit in [grass_min_height, grass_max_height] (inclusive).
		// Below water strands must sit in [grass_min_height_underwater, grass_max_height_underwater].
		// Defaults span the full range so callers who don't set them get "no filter".
		float grass_water_height = 0.0f;
		float grass_min_height = -1e30f;
		float grass_max_height = 1e30f;
		float grass_min_height_underwater = -1e30f;
		float grass_max_height_underwater = 1e30f;

		// Non-serialized attributes:
		XMFLOAT4X4 world;
		wi::primitive::AABB aabb;
		wi::vector<uint32_t> indices; // it is dependent on vertex_lengths and contains triangles with non-zero lengths
		uint32_t layerMask = ~0u;
		mutable bool regenerate_frame = true;
		wi::graphics::Format position_format = wi::graphics::Format::R16G16B16A16_UNORM;
		mutable bool must_rebuild_blas = true;
		mutable bool gpu_initialized = false;

		void Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri);

		static void Initialize();

		constexpr uint32_t GetParticleCount() const { return strandCount * segmentCount; }
		constexpr uint32_t GetVertexCount() const { return strandCount * (segmentCount * 2 + 2) * billboardCount; }
		constexpr uint32_t GetIndexCount() const { return 6 * GetParticleCount() * billboardCount; }
		uint64_t GetMemorySizeInBytes() const;

		constexpr bool IsDirty() const { return _flags & DIRTY; }
		constexpr bool IsCameraBendEnabled() const { return _flags & CAMERA_BEND_ENABLED; }

		constexpr void SetDirty(bool value = true) { if (value) { _flags |= DIRTY; } else { _flags &= ~DIRTY; } }
		constexpr void SetCameraBendEnabled(bool value = true) { if (value) { _flags |= CAMERA_BEND_ENABLED; } else { _flags &= ~CAMERA_BEND_ENABLED; } }

		void ConvertFromOLDSpriteSheet(uint32_t framesX, uint32_t framesY, uint32_t frameCount, uint32_t frameStart);
	};
}
