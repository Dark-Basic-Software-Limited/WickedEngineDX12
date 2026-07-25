#include "wiHairParticle.h"
#include "wiRenderer.h"
#include "wiResourceManager.h"
#include "wiPrimitive.h"
#include "wiRandom.h"
#include "wiArchive.h"
#include "shaders/ShaderInterop.h"
#include "wiTextureHelper.h"
#include "wiScene.h"
#include "shaders/ShaderInterop_HairParticle.h"
#include "wiBacklog.h"
#include "wiEventHandler.h"
#include "wiTimer.h"

using namespace wi::primitive;
using namespace wi::graphics;
using namespace wi::scene;
using namespace wi::enums;

namespace wi
{
	static Shader vs;
	static Shader ps_prepass;
	static Shader ps_prepass_depthonly;
	static Shader ps;
	static Shader ps_shadow;
	static Shader ps_simple;
	static Shader cs_simulate;
	// GG-MAX Stage 3 Option B: 1x1 all-zero R8_UNORM bound to the simulate CS's t4 slot
	// for hair entities that don't supply a paint mask. With xHairGrassType == 0 the
	// shader skips the sample entirely, so the placeholder is read by nothing in upstream
	// usage — but DX12 will validate that the SRV slot has a bound resource regardless,
	// so we need this placeholder to keep non-GG hair systems happy.
	static wi::graphics::Texture grass_visibility_placeholder;
	static DepthStencilState dss_default, dss_equal, dss_shadow;
	static RasterizerState rs, ncrs, wirers, rs_shadow;
	static BlendState bs;
	static BlendState bs_shadow;
	static PipelineState PSO[RENDERPASS_COUNT];
	static PipelineState PSO_wire;

	uint64_t HairParticleSystem::GetMemorySizeInBytes() const
	{
		if (!generalBuffer.IsValid())
			return 0;

		uint64_t retVal = 0;

		retVal += constantBuffer.GetDesc().size;
		retVal += generalBuffer.GetDesc().size;
		retVal += indexBuffer.GetDesc().size;
		retVal += vertexBuffer_length.GetDesc().size;

		return retVal;
	}

	void HairParticleSystem::CreateFromMesh(const wi::scene::MeshComponent& mesh)
	{
		if (mesh.so_pos.IsValid())
		{
			position_format = MeshComponent::Vertex_POS32W::FORMAT;
		}
		else
		{
			position_format = MeshComponent::Vertex_POS16::FORMAT;
		}

		if (vertex_lengths.size() != mesh.vertex_positions.size())
		{
			vertex_lengths.resize(mesh.vertex_positions.size());
			std::fill(vertex_lengths.begin(), vertex_lengths.end(), 1.0f);
		}

		indices.clear();
		uint32_t first_subset = 0;
		uint32_t last_subset = 0;
		mesh.GetLODSubsetRange(0, first_subset, last_subset);
		for (uint32_t subsetIndex = first_subset; subsetIndex < last_subset; ++subsetIndex)
		{
			const MeshComponent::MeshSubset& subset = mesh.subsets[subsetIndex];
			for (size_t i = 0; i < subset.indexCount; i += 3)
			{
				const uint32_t i0 = mesh.indices[subset.indexOffset + i + 0];
				const uint32_t i1 = mesh.indices[subset.indexOffset + i + 1];
				const uint32_t i2 = mesh.indices[subset.indexOffset + i + 2];
				if (vertex_lengths[i0] > 0 || vertex_lengths[i1] > 0 || vertex_lengths[i2] > 0)
				{
					indices.push_back(i0);
					indices.push_back(i1);
					indices.push_back(i2);
				}
			}
		}

		_flags |= REBUILD_BUFFERS;
	}

	void HairParticleSystem::DeleteRenderData()
	{
		constantBuffer = {};
		generalBuffer = {};
		generalBufferOffsetAllocation = {};
		generalBufferOffsetAllocationAlias = {};
		simulation_view = {};
		vb_pos[0] = {};
		vb_pos[1] = {};
		vb_nor = {};
		vb_pos_raytracing = {};
		vb_uvs = {};
		wetmap = {};
		ib_culled = {};
		indirect_view = {};
		prim_view = {};
		indexBuffer = {};
		vertexBuffer_length = {};
		BLAS = {};
	}

	// GG-MAX (Grass Stage 3): rebuild ONLY the vertexBuffer_length GPU buffer from the current
	// vertex_lengths data. CreateRenderData() destroys generalBuffer and resets every per-strand
	// simulation tail to rest, producing a visible "settling pop" on the wind animation each time
	// it's called. For incremental paint updates where strandCount / indices / mesh have NOT
	// changed and only the per-vertex paint mask flipped, this path leaves generalBuffer (and so
	// simulation_view, vb_pos, etc.) completely intact — the sway continues uninterrupted while
	// the GPU picks up the new length mask on its next dispatch.
	void HairParticleSystem::UpdateVertexLengthsBuffer()
	{
		if (vertex_lengths.empty()) return;
		// Safety: if the rest of the GPU resources were never built (generalBuffer invalid),
		// we'd be uploading lengths against nothing — fall through to the full path instead.
		if (!generalBuffer.IsValid()) { CreateRenderData(); return; }

		GraphicsDevice* device = wi::graphics::GetDevice();

		auto pack_lengths = [&](void* dest) {
			uint8_t* vertex_lengths_packed = (uint8_t*)dest;
			for (size_t i = 0; i < vertex_lengths.size(); ++i)
			{
				uint8_t packed = uint8_t(wi::math::Clamp(vertex_lengths[i], 0, 1) * 255.0f);
				std::memcpy(vertex_lengths_packed + i, &packed, sizeof(uint8_t));
			}
		};
		GPUBufferDesc bd;
		bd.misc_flags = ResourceMiscFlag::NONE;
		bd.bind_flags = BindFlag::SHADER_RESOURCE;
		bd.format = Format::R8_UNORM;
		bd.stride = sizeof(uint8_t);
		bd.size = bd.stride * vertex_lengths.size();
		device->CreateBuffer2(&bd, pack_lengths, &vertexBuffer_length);
		device->SetName(&vertexBuffer_length, "HairParticleSystem::vertexBuffer_length");
		// Deliberately NOT touching: generalBuffer, simulation_view, vb_pos[0/1], vb_nor,
		// vb_uvs, wetmap, ib_culled, prim_view, indirect_view, vb_pos_raytracing, indexBuffer,
		// BLAS, must_rebuild_blas, regenerate_frame, _flags. The simulate CS rebinds
		// vertexBuffer_length per dispatch (wiHairParticle.cpp ~line 560) so the new buffer is
		// picked up on the next frame automatically.
	}

	void HairParticleSystem::CreateRenderData()
	{
		DeleteRenderData();

		GraphicsDevice* device = wi::graphics::GetDevice();

		BLAS = {};
		_flags &= ~REBUILD_BUFFERS;
		regenerate_frame = true;
		gg_sim_runs = 0; // GGMAX 1.37: fresh buffers must simulate before the static-skip may engage

		const uint32_t particleCount = GetParticleCount();
		const uint32_t gfx_vertexcount = GetVertexCount();
		const uint32_t gfx_indexcount = GetIndexCount();

		if (particleCount > 0)
		{
			GPUBufferDesc bd;
			bd.usage = Usage::DEFAULT;
			bd.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS | BindFlag::INDEX_BUFFER;
			bd.misc_flags = ResourceMiscFlag::BUFFER_RAW | ResourceMiscFlag::TYPED_FORMAT_CASTING | ResourceMiscFlag::INDIRECT_ARGS | ResourceMiscFlag::NO_DEFAULT_DESCRIPTORS;
			if (device->CheckCapability(GraphicsDeviceCapability::RAYTRACING))
			{
				bd.misc_flags |= ResourceMiscFlag::RAY_TRACING;
			}

			const size_t position_stride = GetFormatStride(position_format);
			const Format ib_format = GetIndexBufferFormatRaw(gfx_vertexcount);
			const uint32_t ib_stride = GetFormatStride(ib_format);
			const uint64_t alignment =
				device->GetMinOffsetAlignment(&bd) *
				sizeof(IndirectDrawArgsIndexedInstanced) * // additional alignment
				sizeof(PatchSimulationData) // additional alignment
				;

			simulation_view.size = sizeof(PatchSimulationData) * particleCount;
			vb_pos[0].size = position_stride * gfx_vertexcount;
			vb_pos[1].size = position_stride * gfx_vertexcount;
			vb_nor.size = sizeof(MeshComponent::Vertex_NOR) * gfx_vertexcount;
			vb_uvs.size = sizeof(MeshComponent::Vertex_UVS) * gfx_vertexcount;
			wetmap.size = sizeof(uint16_t) * gfx_vertexcount;
			ib_culled.size = ib_stride * gfx_indexcount;
			prim_view.size = ib_stride * gfx_indexcount;
			indirect_view.size = sizeof(IndirectDrawArgsIndexedInstanced);
			vb_pos_raytracing.size = position_stride * gfx_vertexcount;

			bd.size =
				AlignTo(indirect_view.size, alignment) +
				AlignTo(simulation_view.size, alignment) +
				AlignTo(vb_pos[0].size, alignment) +
				AlignTo(vb_pos[1].size, alignment) +
				AlignTo(vb_nor.size, alignment) +
				AlignTo(vb_uvs.size, alignment) +
				AlignTo(wetmap.size, alignment) +
				AlignTo(ib_culled.size, alignment) +
				AlignTo(prim_view.size, alignment) +
				AlignTo(vb_pos_raytracing.size, alignment)
				;
#if 0 // Suballocation is disabled for hair particle system for now, there is some memory overwrite-like issue somewhere
			wi::renderer::BufferSuballocation suballoc = wi::renderer::SuballocateGPUBuffer(bd.size);
			if (suballoc.IsValid())
			{
				bool success = device->CreateBuffer(&bd, nullptr, &generalBuffer, &suballoc.alias, suballoc.allocation.byte_offset);
				assert(success);
				device->SetName(&generalBuffer, "HairParticleSystem::generalBuffer (suballocated)");
				generalBufferOffsetAllocation = std::move(suballoc.allocation);
				generalBufferOffsetAllocationAlias = std::move(suballoc.alias);
			}
			else
#endif
			{
				// If suballocation was not successful, a standalone buffer can be created instead:
				bool success = device->CreateBuffer(&bd, nullptr, &generalBuffer);
				assert(success);
				device->SetName(&generalBuffer, "HairParticleSystem::generalBuffer");
			}
			gpu_initialized = false;

			uint64_t buffer_offset = 0ull;

			const uint32_t indirect_stride = sizeof(IndirectDrawArgsIndexedInstanced);
			buffer_offset = AlignTo(buffer_offset, alignment);
			indirect_view.offset = buffer_offset;
			indirect_view.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, indirect_view.offset, indirect_view.size, nullptr, &indirect_stride);
			indirect_view.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, indirect_view.offset, indirect_view.size, nullptr, &indirect_stride);
			indirect_view.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, indirect_view.subresource_srv);
			indirect_view.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, indirect_view.subresource_uav);
			buffer_offset += indirect_view.size;

			const uint32_t simulation_stride = sizeof(PatchSimulationData);
			buffer_offset = AlignTo(buffer_offset, alignment);
			simulation_view.offset = buffer_offset;
			simulation_view.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, simulation_view.offset, simulation_view.size, nullptr, &simulation_stride);
			simulation_view.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, simulation_view.offset, simulation_view.size, nullptr, &simulation_stride);
			simulation_view.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, simulation_view.subresource_srv);
			simulation_view.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, simulation_view.subresource_uav);
			buffer_offset += simulation_view.size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_pos[0].offset = buffer_offset;
			vb_pos[0].subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, vb_pos[0].offset, vb_pos[0].size, &position_format);
			vb_pos[0].subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_pos[0].offset, vb_pos[0].size, &position_format);
			vb_pos[0].descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, vb_pos[0].subresource_srv);
			vb_pos[0].descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_pos[0].subresource_uav);
			buffer_offset += vb_pos[0].size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_pos[1].offset = buffer_offset;
			vb_pos[1].subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, vb_pos[1].offset, vb_pos[1].size, &position_format);
			vb_pos[1].subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_pos[1].offset, vb_pos[1].size, &position_format);
			vb_pos[1].descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, vb_pos[1].subresource_srv);
			vb_pos[1].descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_pos[1].subresource_uav);
			buffer_offset += vb_pos[1].size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_nor.offset = buffer_offset;
			vb_nor.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, vb_nor.offset, vb_nor.size, &MeshComponent::Vertex_NOR::FORMAT);
			vb_nor.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_nor.offset, vb_nor.size, &MeshComponent::Vertex_NOR::FORMAT);
			vb_nor.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, vb_nor.subresource_srv);
			vb_nor.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_nor.subresource_uav);
			buffer_offset += vb_nor.size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_uvs.offset = buffer_offset;
			vb_uvs.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, vb_uvs.offset, vb_uvs.size, &MeshComponent::Vertex_UVS::FORMAT);
			vb_uvs.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_uvs.offset, vb_uvs.size, &MeshComponent::Vertex_UVS::FORMAT);
			vb_uvs.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, vb_uvs.subresource_srv);
			vb_uvs.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_uvs.subresource_uav);
			buffer_offset += vb_uvs.size;

			constexpr Format wetmap_fmt = Format::R16_UNORM;
			buffer_offset = AlignTo(buffer_offset, alignment);
			wetmap.offset = buffer_offset;
			wetmap.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, wetmap.offset, wetmap.size, &wetmap_fmt);
			wetmap.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, wetmap.offset, wetmap.size, &wetmap_fmt);
			wetmap.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, wetmap.subresource_srv);
			wetmap.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, wetmap.subresource_uav);
			buffer_offset += wetmap.size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			ib_culled.offset = buffer_offset;
			ib_culled.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, ib_culled.offset, ib_culled.size, &ib_format);
			ib_culled.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, ib_culled.offset, ib_culled.size, &ib_format);
			ib_culled.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, ib_culled.subresource_srv);
			ib_culled.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, ib_culled.subresource_uav);
			buffer_offset += ib_culled.size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			prim_view.offset = buffer_offset;
			prim_view.subresource_srv = device->CreateSubresource(&generalBuffer, SubresourceType::SRV, prim_view.offset, prim_view.size, &ib_format);
			prim_view.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, prim_view.offset, prim_view.size, &ib_format);
			prim_view.descriptor_srv = device->GetDescriptorIndex(&generalBuffer, SubresourceType::SRV, prim_view.subresource_srv);
			prim_view.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, prim_view.subresource_uav);
			buffer_offset += prim_view.size;

			buffer_offset = AlignTo(buffer_offset, alignment);
			vb_pos_raytracing.offset = buffer_offset;
			vb_pos_raytracing.subresource_uav = device->CreateSubresource(&generalBuffer, SubresourceType::UAV, vb_pos_raytracing.offset, vb_pos_raytracing.size, &position_format);
			vb_pos_raytracing.descriptor_uav = device->GetDescriptorIndex(&generalBuffer, SubresourceType::UAV, vb_pos_raytracing.subresource_uav);
			buffer_offset += vb_pos_raytracing.size;

		}

		GPUBufferDesc bd;
		bd.usage = Usage::DEFAULT;
		bd.size = sizeof(HairParticleCB);
		bd.bind_flags = BindFlag::CONSTANT_BUFFER;
		bd.misc_flags = ResourceMiscFlag::NONE;
		device->CreateBuffer(&bd, nullptr, &constantBuffer);
		device->SetName(&constantBuffer, "HairParticleSystem::constantBuffer");

		if (!vertex_lengths.empty())
		{
			auto pack_lengths = [&](void* dest) {
				uint8_t* vertex_lengths_packed = (uint8_t*)dest;
				for (size_t i = 0; i < vertex_lengths.size(); ++i)
				{
					uint8_t packed = uint8_t(wi::math::Clamp(vertex_lengths[i], 0, 1) * 255.0f);
					std::memcpy(vertex_lengths_packed + i, &packed, sizeof(uint8_t));
				}
			};
			bd.misc_flags = ResourceMiscFlag::NONE;
			bd.bind_flags = BindFlag::SHADER_RESOURCE;
			bd.format = Format::R8_UNORM;
			bd.stride = sizeof(uint8_t);
			bd.size = bd.stride * vertex_lengths.size();
			device->CreateBuffer2(&bd, pack_lengths, &vertexBuffer_length);
			device->SetName(&vertexBuffer_length, "HairParticleSystem::vertexBuffer_length");
		}
		if (!indices.empty())
		{
			bd.misc_flags = ResourceMiscFlag::NONE;
			bd.bind_flags = BindFlag::SHADER_RESOURCE;
			bd.format = Format::R32_UINT;
			bd.stride = sizeof(uint32_t);
			bd.size = bd.stride * indices.size();
			device->CreateBuffer(&bd, indices.data(), &indexBuffer);
			device->SetName(&indexBuffer, "HairParticleSystem::indexBuffer");
		}
	}
	void HairParticleSystem::CreateRaytracingRenderData()
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		if (device->CheckCapability(GraphicsDeviceCapability::RAYTRACING) && prim_view.IsValid())
		{
			RaytracingAccelerationStructureDesc desc;
			desc.type = RaytracingAccelerationStructureDesc::Type::BOTTOMLEVEL;
			desc.flags |= RaytracingAccelerationStructureDesc::FLAG_PREFER_FAST_BUILD;
			desc.flags |= RaytracingAccelerationStructureDesc::FLAG_ALLOW_UPDATE;

			desc.bottom_level.geometries.emplace_back();
			auto& geometry = desc.bottom_level.geometries.back();
			geometry.type = RaytracingAccelerationStructureDesc::BottomLevel::Geometry::Type::TRIANGLES;
			geometry.triangles.vertex_buffer = generalBuffer;
			geometry.triangles.index_buffer = generalBuffer;
			geometry.triangles.index_format = GetIndexBufferFormat(GetVertexCount());
			geometry.triangles.index_count = GetIndexCount();
			geometry.triangles.index_offset = prim_view.offset / GetFormatStride(GetIndexBufferFormatRaw(GetVertexCount()));
			geometry.triangles.vertex_count = (uint32_t)(vb_pos_raytracing.size / GetFormatStride(position_format));
			geometry.triangles.vertex_format = position_format == Format::R32G32B32A32_FLOAT ? Format::R32G32B32_FLOAT : position_format;
			geometry.triangles.vertex_stride = GetFormatStride(position_format);
			geometry.triangles.vertex_byte_offset = vb_pos_raytracing.offset;

			bool success = device->CreateRaytracingAccelerationStructure(&desc, &BLAS);
			assert(success);
			device->SetName(&BLAS, "HairParticleSystem::BLAS");

			must_rebuild_blas = true;
		}
	}

	void HairParticleSystem::UpdateCPU(const TransformComponent& transform, const MeshComponent& mesh, float dt)
	{
		// GGMAX 1.37: a moved system must re-simulate (world feeds the sim CS).
		if (std::memcmp(&gg_prev_world, &transform.world, sizeof(XMFLOAT4X4)) != 0)
		{
			gg_prev_world = transform.world;
			gg_sim_runs = 0;
		}
		world = transform.world;

		XMFLOAT3 _min = mesh.aabb.getMin();
		XMFLOAT3 _max = mesh.aabb.getMax();

		float maxlen = length;

		for (auto& x : atlas_rects)
		{
			maxlen = std::max(maxlen, x.size);
		}

		_max.x += maxlen;
		_max.y += maxlen;
		_max.z += maxlen;

		_min.x -= maxlen;
		_min.y -= maxlen;
		_min.z -= maxlen;

		aabb = AABB(_min, _max);
		aabb = aabb.transform(world);

		if ((_flags & REBUILD_BUFFERS) || indices.empty())
		{
			CreateFromMesh(mesh);
		}

		const uint32_t particleCount = GetParticleCount();
		const uint32_t gfx_indexcount = GetIndexCount();

		if (
			(_flags & REBUILD_BUFFERS) ||
			!constantBuffer.IsValid() ||
			particleCount != simulation_view.size / sizeof(PatchSimulationData) ||
			(gfx_indexcount > 0 && prim_view.size != gfx_indexcount * GetFormatStride(GetIndexBufferFormatRaw(GetVertexCount())))
			)
		{
			CreateRenderData();
		}

		std::swap(vb_pos[0], vb_pos[1]);
	}
	void HairParticleSystem::InitializeGPUBuffersIfNeeded(
		const HairParticleSystem* hairs,
		size_t hairCount,
		CommandList cmd
	)
	{
		GraphicsDevice* device = wi::graphics::GetDevice();
		device->EventBegin("HairParticleSystem - InitializeGPUBuffersIfNeeded", cmd);

		for (uint32_t i = 0; i < hairCount; ++i)
		{
			const HairParticleSystem& hair = hairs[i];
			if (hair.strandCount == 0 || !hair.generalBuffer.IsValid())
			{
				continue;
			}
			if (!hair.gpu_initialized)
			{
				hair.gpu_initialized = true;
				device->ClearUAV(&hair.generalBuffer, 0, cmd);
				wi::renderer::PushBarrier(GPUBarrier::Buffer(&hair.generalBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::COPY_DST));
			}
		}
		wi::renderer::FlushBarriers(cmd);
		device->EventEnd(cmd);
	}
	void HairParticleSystem::UpdateGPU(
		const UpdateGPUItem* items,
		uint32_t itemCount,
		CommandList cmd
	)
	{
		GraphicsDevice* device = wi::graphics::GetDevice();
		device->EventBegin("HairParticleSystem - UpdateGPU", cmd);

		for (uint32_t i = 0; i < itemCount; ++i)
		{
			const UpdateGPUItem& item = items[i];
			const HairParticleSystem& hair = *item.hair;
			if (hair.strandCount == 0 || !hair.generalBuffer.IsValid())
			{
				continue;
			}
			const MeshComponent& mesh = *item.mesh;
			const MaterialComponent& material = *item.material;

			TextureDesc desc;
			if (material.textures[MaterialComponent::BASECOLORMAP].resource.IsValid())
			{
				desc = material.textures[MaterialComponent::BASECOLORMAP].resource.GetTexture().GetDesc();
			}
			HairParticleCB hcb = {};
			hcb.xHairTransform.Create(hair.world);
			if (!IsFormatUnorm(mesh.position_format) || mesh.so_pos.IsValid())
			{
				hcb.xHairBaseMeshUnormRemap.init();
			}
			else
			{
				XMFLOAT4X4 unormRemap;
				XMStoreFloat4x4(&unormRemap, mesh.aabb.getUnormRemapMatrix());
				hcb.xHairBaseMeshUnormRemap.Create(unormRemap);
			}
			if (IsFormatUnorm(hair.position_format))
			{
				hcb.xHairFlags |= HAIR_FLAG_UNORM_POS;
			}
			if (hair.regenerate_frame)
			{
				hcb.xHairFlags |= HAIR_FLAG_REGENERATE_FRAME;
			}
			if (hair.IsCameraBendEnabled())
			{
				hcb.xHairFlags |= HAIR_FLAG_CAMERA_BEND;
			}
			hcb.xHairAspect = hair.width * (float)std::max(1u, desc.width) / (float)std::max(1u, desc.height);
			hcb.xHairLength = hair.length;
			hcb.xHairStiffness = hair.stiffness;
			hcb.xHairDrag = hair.drag;
			hcb.xHairGravityPower = hair.gravityPower;
			hcb.xHairRandomness = hair.randomness;
			hcb.xHairStrandCount = hair.strandCount;
			hcb.xHairSegmentCount = std::max(hair.segmentCount, 1u);
			hcb.xHairBillboardCount = std::max(hair.billboardCount, 1u);
			hcb.xHairParticleCount = hcb.xHairStrandCount * hcb.xHairSegmentCount;
			hcb.xHairRandomSeed = hair.randomSeed;
			hcb.xHairViewDistance = hair.viewDistance;
			hcb.xHairBaseMeshIndexCount = (uint)hair.indices.size();
			hcb.xHairLayerMask = hair.layerMask;
			hcb.xHairInstanceIndex = item.instanceIndex;
			if (hair.atlas_rects.empty())
			{
				// unspecified rects will default to view the whole texture:
				hcb.xHairAtlasRects[0].texMulAdd = float4(1, 1, 0, 0);
				hcb.xHairAtlasRects[0].size = 1;
				hcb.xHairAtlasRects[0].aspect = 1;
				hcb.xHairAtlasRectCount = 1;
			}
			else
			{
				// custom atlas rects:
				hcb.xHairAtlasRectCount = uint(std::min(hair.atlas_rects.size(), arraysize(hcb.xHairAtlasRects)));
				for (uint a = 0; a < hcb.xHairAtlasRectCount; ++a)
				{
					hcb.xHairAtlasRects[a].texMulAdd = hair.atlas_rects[a].texMulAdd;
					hcb.xHairAtlasRects[a].size = hair.atlas_rects[a].size;
					hcb.xHairAtlasRects[a].aspect = hcb.xHairAtlasRects[a].texMulAdd.x / hcb.xHairAtlasRects[a].texMulAdd.y;
				}
			}
			hcb.xHairUniformity = hair.uniformity;

			// GG-MAX Stage 3 Option B: paint-mask CB constants. grass_type == 0 -> shader skips
			// the sample entirely (upstream behavior); GG sets grass_type = 1..N on each per-type
			// chunk entity. Transform is uv = (worldXZ - origin) * invWorldSize + 0.5 — matches
			// the GG_GetGrassMap CPU formula with origin = (0,0) and invWorldSize = 0.5/halfExtent.
			hcb.xHairGrassType = hair.grass_type;
			hcb.xHairGrassMapInvWorldSize = hair.grass_map_inv_world_size;
			hcb.xHairGrassMapOriginX = hair.grass_map_origin_x;
			hcb.xHairGrassMapOriginZ = hair.grass_map_origin_z;

			// GG-MAX Stage B.10: altitude filter (only used when xHairGrassType != 0).
			hcb.xHairGrassWaterHeight = hair.grass_water_height;
			hcb.xHairGrassMinHeight = hair.grass_min_height;
			hcb.xHairGrassMaxHeight = hair.grass_max_height;
			hcb.xHairGrassMinHeightUnderwater = hair.grass_min_height_underwater;
			hcb.xHairGrassMaxHeightUnderwater = hair.grass_max_height_underwater;

			device->UpdateBuffer(&hair.constantBuffer, &hcb, cmd);
			wi::renderer::PushBarrier(GPUBarrier::Buffer(&hair.constantBuffer, ResourceState::COPY_DST, ResourceState::CONSTANT_BUFFER));

			IndirectDrawArgsIndexedInstanced args = {};
			args.BaseVertexLocation = 0;
			args.IndexCountPerInstance = 0; // this will use shader atomic
			args.InstanceCount = 1;
			args.StartIndexLocation = 0;
			args.StartInstanceLocation = 0;
			device->UpdateBuffer(&hair.generalBuffer, &args, cmd, sizeof(args), hair.indirect_view.offset);
			wi::renderer::PushBarrier(GPUBarrier::Buffer(&hair.generalBuffer, ResourceState::COPY_DST, ResourceState::UNORDERED_ACCESS));

			if (hair.regenerate_frame)
			{
				hair.regenerate_frame = false;
			}
		}

		wi::renderer::FlushBarriers(cmd);

		// Simulate:
		device->BindComputeShader(&cs_simulate, cmd);
		for (uint32_t i = 0; i < itemCount; ++i)
		{
			const UpdateGPUItem& item = items[i];
			const HairParticleSystem& hair = *item.hair;
			if (hair.strandCount == 0 || !hair.generalBuffer.IsValid())
			{
				continue;
			}
			const MeshComponent& mesh = *item.mesh;

			device->BindConstantBuffer(&hair.constantBuffer, CB_GETBINDSLOT(HairParticleCB), cmd);

			device->BindUAV(&hair.generalBuffer, 0, cmd, hair.simulation_view.subresource_uav);
			device->BindUAV(&hair.generalBuffer, 1, cmd, hair.vb_pos[0].subresource_uav);
			device->BindUAV(&hair.generalBuffer, 2, cmd, hair.vb_uvs.subresource_uav);
			device->BindUAV(&hair.generalBuffer, 3, cmd, hair.ib_culled.subresource_uav);
			device->BindUAV(&hair.generalBuffer, 4, cmd, hair.indirect_view.subresource_uav);
			device->BindUAV(&hair.generalBuffer, 5, cmd, hair.vb_pos_raytracing.subresource_uav);
			device->BindUAV(&hair.generalBuffer, 6, cmd, hair.vb_nor.subresource_uav);
			device->BindUAV(&hair.generalBuffer, 7, cmd, hair.prim_view.subresource_uav);

			if (hair.indexBuffer.IsValid())
			{
				device->BindResource(&hair.indexBuffer, 0, cmd);
			}
			else
			{
				device->BindResource(&mesh.generalBuffer, 0, cmd, mesh.ib.subresource_srv);
			}
			if (mesh.streamoutBuffer.IsValid())
			{
				device->BindResource(&mesh.streamoutBuffer, 1, cmd, mesh.so_pos.subresource_srv);
				device->BindResource(&mesh.streamoutBuffer, 2, cmd, mesh.so_nor.subresource_srv);
			}
			else
			{
				device->BindResource(&mesh.generalBuffer, 1, cmd, mesh.vb_pos_wind.subresource_srv);
				device->BindResource(&mesh.generalBuffer, 2, cmd, mesh.vb_nor.subresource_srv);
			}
			device->BindResource(&hair.vertexBuffer_length, 3, cmd);

			// GG-MAX Stage 3 Option B: per-entity paint mask SRV. The shader only reads from it
			// when xHairGrassType != 0; non-GG hair entities get the placeholder and pay nothing.
			{
				const Texture* grassTex = (hair.grass_visibility_texture && hair.grass_visibility_texture->IsValid())
					? hair.grass_visibility_texture
					: &grass_visibility_placeholder;
				device->BindResource(grassTex, 4, cmd);
			}

			device->Dispatch((hair.strandCount + THREADCOUNT_SIMULATEHAIR - 1) / THREADCOUNT_SIMULATEHAIR, 1, 1, cmd);

			wi::renderer::PushBarrier(GPUBarrier::Buffer(&hair.generalBuffer, ResourceState::UNORDERED_ACCESS, ResourceState::INDIRECT_ARGUMENT | ResourceState::INDEX_BUFFER | ResourceState::SHADER_RESOURCE));
		}

		wi::renderer::FlushBarriers(cmd);

		device->EventEnd(cmd);
	}

	void HairParticleSystem::Draw(const MaterialComponent& material, wi::enums::RENDERPASS renderPass, CommandList cmd) const
	{
		if (strandCount == 0 || !constantBuffer.IsValid())
		{
			return;
		}

		if (renderPass == RENDERPASS_SHADOW && !material.IsCastingShadow())
			return;

		GraphicsDevice* device = wi::graphics::GetDevice();
		device->EventBegin("HairParticle - Draw", cmd);

		device->BindStencilRef(STENCILREF_DEFAULT, cmd);

		if (wi::renderer::IsWireRender())
		{
			if (renderPass == RENDERPASS_PREPASS || renderPass == RENDERPASS_PREPASS_DEPTHONLY)
			{
				return;
			}
			device->BindPipelineState(&PSO_wire, cmd);
		}
		else
		{
			device->BindPipelineState(&PSO[renderPass], cmd);

			if (renderPass != RENDERPASS_PREPASS || renderPass == RENDERPASS_PREPASS_DEPTHONLY) // depth only alpha test will be full res
			{
				device->BindShadingRate(material.shadingRate, cmd);
			}
		}

		device->BindConstantBuffer(&constantBuffer, CB_GETBINDSLOT(HairParticleCB), cmd);
		device->BindResource(&generalBuffer, 0, cmd, prim_view.subresource_srv);

		device->BindIndexBuffer(&generalBuffer, GetIndexBufferFormat(GetVertexCount()), ib_culled.offset, cmd);

		device->DrawIndexedInstancedIndirect(&generalBuffer, indirect_view.offset, cmd);

		device->EventEnd(cmd);
	}


	void HairParticleSystem::Serialize(wi::Archive& archive, wi::ecs::EntitySerializer& seri)
	{
		if (archive.IsReadMode())
		{
			archive >> _flags;
			wi::ecs::SerializeEntity(archive, meshID, seri);
			archive >> strandCount;
			archive >> segmentCount;
			archive >> randomSeed;
			archive >> length;
			archive >> stiffness;
			archive >> randomness;
			archive >> viewDistance;
			if (archive.GetVersion() >= 39)
			{
				archive >> vertex_lengths;
			}
			if (archive.GetVersion() >= 42)
			{
				if (seri.GetVersion() < 1)
				{
					// Conversion from old frame format to new:
					uint32_t framesX = 1;
					uint32_t framesY = 1;
					uint32_t frameCount = 1;
					uint32_t frameStart = 0;

					archive >> framesX;
					archive >> framesY;
					archive >> frameCount;
					archive >> frameStart;

					ConvertFromOLDSpriteSheet(framesX, framesY, frameCount, frameStart);

				}
			}

			if (archive.GetVersion() == 48)
			{
				uint8_t shadingRate;
				archive >> shadingRate; // no longer needed
			}

			if (seri.GetVersion() >= 1)
			{
				archive >> width;
				archive >> uniformity;

				size_t rect_count = 0;
				archive >> rect_count;
				atlas_rects.resize(rect_count);
				for (size_t i = 0; i < rect_count; ++i)
				{
					archive >> atlas_rects[i].texMulAdd;
					archive >> atlas_rects[i].size;
				}
			}

			if (seri.GetVersion() >= 2)
			{
				archive >> drag;
			}
			else
			{
				// Old stiffness remap to new:
				stiffness *= 0.05f;
			}

			if (seri.GetVersion() >= 3)
			{
				archive >> gravityPower;
				archive >> billboardCount;
			}
			else
			{
				// This param was always true before:
				SetCameraBendEnabled(true);
			}
		}
		else
		{
			archive << _flags;
			wi::ecs::SerializeEntity(archive, meshID, seri);
			archive << strandCount;
			archive << segmentCount;
			archive << randomSeed;
			archive << length;
			archive << stiffness;
			archive << randomness;
			archive << viewDistance;
			if (archive.GetVersion() >= 39)
			{
				archive << vertex_lengths;
			}

			if (seri.GetVersion() >= 1)
			{
				archive << width;
				archive << uniformity;
				archive << atlas_rects.size();
				for (size_t i = 0; i < atlas_rects.size(); ++i)
				{
					archive << atlas_rects[i].texMulAdd;
					archive << atlas_rects[i].size;
				}
			}

			if (seri.GetVersion() >= 2)
			{
				archive << drag;
			}
			if (seri.GetVersion() >= 3)
			{
				archive << gravityPower;
				archive << billboardCount;
			}
		}
	}

	namespace HairParticleSystem_Internal
	{
		void LoadShaders()
		{
			wi::renderer::LoadShader(ShaderStage::VS, vs, "hairparticleVS.cso");

			wi::renderer::LoadShader(ShaderStage::PS, ps_simple, "hairparticlePS_simple.cso");
			wi::renderer::LoadShader(ShaderStage::PS, ps_prepass, "hairparticlePS_prepass.cso");
			wi::renderer::LoadShader(ShaderStage::PS, ps_prepass_depthonly, "hairparticlePS_prepass_depthonly.cso");
			wi::renderer::LoadShader(ShaderStage::PS, ps_shadow, "hairparticlePS_shadow.cso");
			wi::renderer::LoadShader(ShaderStage::PS, ps, "hairparticlePS.cso");

			wi::renderer::LoadShader(ShaderStage::CS, cs_simulate, "hairparticle_simulateCS.cso");

			GraphicsDevice* device = wi::graphics::GetDevice();

			for (int i = 0; i < RENDERPASS_COUNT; ++i)
			{
				if (i == RENDERPASS_PREPASS || i == RENDERPASS_PREPASS_DEPTHONLY || i == RENDERPASS_MAIN || i == RENDERPASS_SHADOW)
				{
					PipelineStateDesc desc;
					desc.vs = &vs;
					desc.bs = &bs;
					desc.rs = &ncrs;
					desc.dss = &dss_default;
					desc.pt = PrimitiveTopology::TRIANGLELIST;

					switch (i)
					{
					case RENDERPASS_PREPASS:
						desc.ps = &ps_prepass;
						break;
					case RENDERPASS_PREPASS_DEPTHONLY:
						desc.ps = &ps_prepass_depthonly;
						break;
					case RENDERPASS_MAIN:
						desc.ps = &ps;
						desc.dss = &dss_equal;
						break;
					case RENDERPASS_SHADOW:
						desc.ps = &ps_shadow;
						desc.rs = &rs_shadow;
						desc.bs = &bs_shadow;
						desc.dss = &dss_shadow;
						break;
					}

					device->CreatePipelineState(&desc, &PSO[i]);
				}
			}

			{
				PipelineStateDesc desc;
				desc.vs = &vs;
				desc.ps = &ps_simple;
				desc.bs = &bs;
				desc.rs = &wirers;
				desc.dss = &dss_default;
				desc.pt = PrimitiveTopology::TRIANGLELIST;
				device->CreatePipelineState(&desc, &PSO_wire);
			}


		}
	}

	void HairParticleSystem::Initialize()
	{
		wi::Timer timer;

		RasterizerState rsd;
		rsd.fill_mode = FillMode::SOLID;
		rsd.cull_mode = CullMode::BACK;
		rsd.front_counter_clockwise = true;
		rsd.depth_bias = 0;
		rsd.depth_bias_clamp = 0;
		rsd.slope_scaled_depth_bias = 0;
		rsd.depth_clip_enable = true;
		rsd.multisample_enable = false;
		rsd.antialiased_line_enable = false;
		rs = rsd;

		rsd.fill_mode = FillMode::SOLID;
		rsd.cull_mode = CullMode::NONE;
		rsd.front_counter_clockwise = true;
		rsd.depth_bias = 0;
		rsd.depth_bias_clamp = 0;
		rsd.slope_scaled_depth_bias = 0;
		rsd.depth_clip_enable = true;
		rsd.multisample_enable = false;
		rsd.antialiased_line_enable = false;
		ncrs = rsd;

		rsd.fill_mode = FillMode::WIREFRAME;
		rsd.cull_mode = CullMode::NONE;
		rsd.front_counter_clockwise = true;
		rsd.depth_bias = 0;
		rsd.depth_bias_clamp = 0;
		rsd.slope_scaled_depth_bias = 0;
		rsd.depth_clip_enable = true;
		rsd.multisample_enable = false;
		rsd.antialiased_line_enable = false;
		wirers = rsd;

		rs_shadow = ncrs;

		if (IsFormatUnorm(wi::renderer::format_depthbuffer_shadowmap))
		{
			rs_shadow.depth_bias = -1;
		}
		else
		{
			rs.depth_bias = -10;
		}


		DepthStencilState dsd;
		dsd.depth_enable = true;
		dsd.depth_write_mask = DepthWriteMask::ALL;
		dsd.depth_func = ComparisonFunc::GREATER;

		dsd.stencil_enable = false;
		dss_shadow = dsd;

		dsd.stencil_enable = true;
		dsd.stencil_read_mask = 0xFF;
		dsd.stencil_write_mask = 0xFF;
		dsd.front_face.stencil_func = ComparisonFunc::ALWAYS;
		dsd.front_face.stencil_pass_op = StencilOp::REPLACE;
		dsd.front_face.stencil_fail_op = StencilOp::KEEP;
		dsd.front_face.stencil_depth_fail_op = StencilOp::KEEP;
		dsd.back_face.stencil_func = ComparisonFunc::ALWAYS;
		dsd.back_face.stencil_pass_op = StencilOp::REPLACE;
		dsd.back_face.stencil_fail_op = StencilOp::KEEP;
		dsd.back_face.stencil_depth_fail_op = StencilOp::KEEP;
		dss_default = dsd;

		dsd.depth_write_mask = DepthWriteMask::ZERO;
		dsd.depth_func = ComparisonFunc::EQUAL;
		dsd.stencil_enable = false;
		dss_equal = dsd;


		BlendState bld;
		bld.render_target[0].blend_enable = false;
		bld.alpha_to_coverage_enable = false;
		bs = bld;

		bld.render_target[0].render_target_write_mask = ColorWrite::DISABLE;
		bs_shadow = bld;

		static wi::eventhandler::Handle handle = wi::eventhandler::Subscribe(wi::eventhandler::EVENT_RELOAD_SHADERS, [](uint64_t userdata) { HairParticleSystem_Internal::LoadShaders(); });
		HairParticleSystem_Internal::LoadShaders();

		// GG-MAX Stage 3 Option B: 1x1 zero-init placeholder bound to simulate CS slot t4 for
		// hair entities that don't supply a paint mask. With xHairGrassType == 0 the shader
		// won't read it; this just keeps the SRV slot validly bound.
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			TextureDesc td = {};
			td.bind_flags = BindFlag::SHADER_RESOURCE;
			td.sample_count = 1;
			td.mip_levels = 1;
			td.array_size = 1;
			td.format = Format::R8_UNORM;
			td.usage = Usage::DEFAULT;
			td.width = 1;
			td.height = 1;
			const uint8_t zero = 0;
			SubresourceData initdata = {};
			initdata.data_ptr = &zero;
			initdata.row_pitch = 1;
			device->CreateTexture(&td, &initdata, &grass_visibility_placeholder);
			device->SetName(&grass_visibility_placeholder, "grass_visibility_placeholder");
		}

		wilog("wi::HairParticleSystem Initialized (%d ms)", (int)std::round(timer.elapsed()));
	}

	void HairParticleSystem::ConvertFromOLDSpriteSheet(uint32_t framesX, uint32_t framesY, uint32_t frameCount, uint32_t frameStart)
	{
		framesX = std::max(1u, framesX);
		framesY = std::max(1u, framesY);
		width = (float)framesX / (float)framesY;
		float2 texmul = float2(1.0f / (float)framesX, 1.0f / (float)framesY);
		atlas_rects.resize(frameCount);
		for (uint32_t i = 0; i < frameCount; ++i)
		{
			uint currentFrame = frameStart + i;
			uint2 offset = uint2(currentFrame % framesX, currentFrame / framesY);
			atlas_rects[i].texMulAdd = XMFLOAT4(texmul.x, texmul.y, offset.x * texmul.x, offset.y * texmul.y);
		}
	}
}
