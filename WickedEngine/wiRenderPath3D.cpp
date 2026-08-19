#include "wiRenderPath3D.h"
#include "wiRenderer.h"
#include "wiImage.h"
#include "wiHelper.h"
#include "wiTextureHelper.h"
#include "wiProfiler.h"

using namespace wi::graphics;
using namespace wi::enums;
using namespace wi::scene;
using namespace wi::ecs;

namespace wi
{
	namespace graphics { extern bool gg_dred_armed; } // GGMAX 2.05: defined in wiGraphicsDevice_DX12.cpp

	static constexpr float foreground_depth_range = 0.01f;

	// GGMAX 1.39: skip the underwater postprocess while the camera is safely above the
	// waterline (see the ocean block in RenderPostprocessChain). false = stock always-run.
	bool gg_skip_underwater_above_water = true;

	// GGMAX 1.40: merge small command lists to cut submit fragmentation. Each BeginCommandList
	// is a real allocator+list and every cross-list wait splits the queue submission — ~17 lists
	// with ~10 waits leave measurable GPU idle bubbles between submits. When true:
	//	(a) the occlusion-culling pass records at the tail of the main prepass list
	//	(b) transparents + postprocess chain record into ONE list (camera components hoisted
	//	    before it — their output is consumed next frame either way)
	// false = stock list structure.
	bool gg_render_merge_lists = true;

	// GGMAX 1.48c: lean-async (knob defined in wiGraphicsDevice_DX12.cpp; see comment there).
	// When true, the four tiny helper lists (VT copy-pages, ocean sim + readback, VT
	// tile-request + writeback) run on QUEUE_GRAPHICS instead of async COPY/COMPUTE —
	// the two big compute lists (prepare-async, main compute effects) stay async.
	namespace graphics { extern bool gg_lean_async; }
	static inline wi::graphics::QUEUE_TYPE gg_lean(wi::graphics::QUEUE_TYPE q)
	{
		return wi::graphics::gg_lean_async ? wi::graphics::QUEUE_GRAPHICS : q;
	}

	void RenderPath3D::DeleteGPUResources()
	{
		RenderPath2D::DeleteGPUResources();

		// GGMAX 2.05 DIAGNOSTIC (standalone-play DEVICE_HUNG hunt, DRED-armed only): keep
		// every generation of the depth-history chain alive for the process lifetime. The
		// page-fault VA matched freed depthBuffer_Copy1/rtLinearDepth in 11/11 dumps — if
		// the hang STOPS with this leak, the faulted resource is one of these (a reader
		// holds a stale texture_depth_index_prev-style bindless index across the resize);
		// if it PERSISTS, the depth chain is exonerated. Remove once the hunt closes.
		if (wi::graphics::gg_dred_armed)
		{
			static wi::vector<wi::graphics::Texture> gg_depth_keepalive;
			if (depthBuffer_Main.IsValid()) gg_depth_keepalive.push_back(depthBuffer_Main);
			if (depthBuffer_Copy.IsValid()) gg_depth_keepalive.push_back(depthBuffer_Copy);
			if (depthBuffer_Copy1.IsValid()) gg_depth_keepalive.push_back(depthBuffer_Copy1);
			if (rtLinearDepth.IsValid()) gg_depth_keepalive.push_back(rtLinearDepth);
		}

		rtMain = {};
		rtMain_render = {};
		rtPrimitiveID = {};
		rtPrimitiveID_render = {};
		rtVelocity = {};
		rtReflection = {};
		rtRaytracedDiffuse = {};
		rtSSR = {};
		rtSceneCopy = {};
		rtSceneCopy_tmp = {};
		rtWaterRipple = {};
		rtParticleDistortion_render = {};
		rtParticleDistortion = {};
		rtVolumetricLights = {};
		rtBloom = {};
		rtBloom_tmp = {};
		rtAO = {};
		rtShadow = {};
		rtSun[0] = {};
		rtSun[1] = {};
		rtSun[2] = {};
		rtSun_resolved = {};
		rtGUIBlurredBackground[0] = {};
		rtGUIBlurredBackground[1] = {};
		rtGUIBlurredBackground[2] = {};
		rtShadingRate = {};
		rtFSR[0] = {};
		rtFSR[1] = {};
		rtOutlineSource = {};

		rtPostprocess = {};

		depthBuffer_Main = {};
		depthBuffer_Copy = {};
		depthBuffer_Copy1 = {};
		depthBuffer_Reflection = {};
		rtLinearDepth = {};
		reprojectedDepth = {};

		debugUAV = {};
		tiledLightResources = {};
		tiledLightResources_planarReflection = {};
		luminanceResources = {};
		ssaoResources = {};
		msaoResources = {};
		rtaoResources = {};
		rtdiffuseResources = {};
		rtreflectionResources = {};
		ssrResources = {};
		rtshadowResources = {};
		screenspaceshadowResources = {};
		depthoffieldResources = {};
		motionblurResources = {};
		aerialperspectiveResources = {};
		aerialperspectiveResources_reflection = {};
		volumetriccloudResources = {};
		volumetriccloudResources_reflection = {};
		bloomResources = {};
		surfelGIResources = {};
		temporalAAResources = {};
		visibilityResources = {};
		fsr2Resources = {};
		vxgiResources = {};
		meshblendResources = {};
	}

	void RenderPath3D::ResizeBuffers()
	{
		first_frame = true;
		DeleteGPUResources();

		GraphicsDevice* device = wi::graphics::GetDevice();

		XMUINT2 internalResolution = GetInternalResolution();
		camera->width = (float)internalResolution.x;
		camera->height = (float)internalResolution.y;

		// Render targets:

		{
			TextureDesc desc;
			desc.format = wi::renderer::format_rendertarget_main;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			desc.sample_count = 1;
			device->CreateTexture(&desc, nullptr, &rtMain);
			device->SetName(&rtMain, "rtMain");

			if (getMSAASampleCount() > 1)
			{
				desc.sample_count = getMSAASampleCount();
				desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;

				device->CreateTexture(&desc, nullptr, &rtMain_render);
				device->SetName(&rtMain_render, "rtMain_render");
			}
			else
			{
				rtMain_render = rtMain;
			}
		}
		{
			TextureDesc desc;
			desc.format = wi::renderer::format_idbuffer;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;
			if (getMSAASampleCount() > 1)
			{
				desc.bind_flags |= BindFlag::UNORDERED_ACCESS;
			}
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			desc.sample_count = 1;
			desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
			desc.misc_flags = ResourceMiscFlag::ALIASING_TEXTURE_RT_DS;
			device->CreateTexture(&desc, nullptr, &rtPrimitiveID);
			device->SetName(&rtPrimitiveID, "rtPrimitiveID");

			if (getMSAASampleCount() > 1)
			{
				desc.sample_count = getMSAASampleCount();
				desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;
				desc.misc_flags = ResourceMiscFlag::NONE;
				device->CreateTexture(&desc, nullptr, &rtPrimitiveID_render);
				device->SetName(&rtPrimitiveID_render, "rtPrimitiveID_render");
			}
			else
			{
				rtPrimitiveID_render = rtPrimitiveID;
			}
		}
		{
			TextureDesc desc;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;
			desc.format = Format::R16G16_FLOAT;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			desc.sample_count = 1;
			desc.misc_flags = ResourceMiscFlag::ALIASING_TEXTURE_RT_DS;
			device->CreateTexture(&desc, nullptr, &rtParticleDistortion);
			device->SetName(&rtParticleDistortion, "rtParticleDistortion");
			if (getMSAASampleCount() > 1)
			{
				desc.sample_count = getMSAASampleCount();
				desc.misc_flags = ResourceMiscFlag::NONE;
				device->CreateTexture(&desc, nullptr, &rtParticleDistortion_render);
				device->SetName(&rtParticleDistortion_render, "rtParticleDistortion_render");
			}
			else
			{
				rtParticleDistortion_render = rtParticleDistortion;
			}
		}
		{
			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = wi::renderer::format_rendertarget_main;
			desc.width = internalResolution.x / 4;
			desc.height = internalResolution.y / 4;
			desc.mip_levels = std::min(8u, (uint32_t)std::log2(std::max(desc.width, desc.height)));
			device->CreateTexture(&desc, nullptr, &rtSceneCopy);
			device->SetName(&rtSceneCopy, "rtSceneCopy");
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS | BindFlag::RENDER_TARGET; // render target for aliasing
			device->CreateTexture(&desc, nullptr, &rtSceneCopy_tmp, &rtPrimitiveID);
			device->SetName(&rtSceneCopy_tmp, "rtSceneCopy_tmp");

			device->CreateMipgenSubresources(rtSceneCopy);
			device->CreateMipgenSubresources(rtSceneCopy_tmp);

			// because this is used by SSR and SSGI before it gets a chance to be normally rendered, it MUST be cleared!
			CommandList cmd = device->BeginCommandList();
			device->Barrier(GPUBarrier::Image(&rtSceneCopy, rtSceneCopy.desc.layout, ResourceState::UNORDERED_ACCESS), cmd);
			device->ClearUAV(&rtSceneCopy, 0, cmd);
			device->Barrier(GPUBarrier::Image(&rtSceneCopy, ResourceState::UNORDERED_ACCESS, rtSceneCopy.desc.layout), cmd);
		}
		{
			TextureDesc desc;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = wi::renderer::format_rendertarget_main;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			assert(ComputeTextureMemorySizeInBytes(desc) <= ComputeTextureMemorySizeInBytes(rtPrimitiveID.desc)); // Aliased check
			device->CreateTexture(&desc, nullptr, &rtPostprocess, &rtPrimitiveID); // Aliased!
			device->SetName(&rtPostprocess, "rtPostprocess");
		}
		{
			TextureDesc desc;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = Format::R10G10B10A2_UNORM;
			desc.width = internalResolution.x / 4;
			desc.height = internalResolution.y / 4;
			desc.bind_flags = BindFlag::UNORDERED_ACCESS | BindFlag::SHADER_RESOURCE;
			device->CreateTexture(&desc, nullptr, &rtGUIBlurredBackground[0]);
			device->SetName(&rtGUIBlurredBackground[0], "rtGUIBlurredBackground[0]");

			desc.width /= 4;
			desc.height /= 4;
			device->CreateTexture(&desc, nullptr, &rtGUIBlurredBackground[1]);
			device->SetName(&rtGUIBlurredBackground[1], "rtGUIBlurredBackground[1]");
			device->CreateTexture(&desc, nullptr, &rtGUIBlurredBackground[2]);
			device->SetName(&rtGUIBlurredBackground[2], "rtGUIBlurredBackground[2]");
		}
		if (device->CheckCapability(GraphicsDeviceCapability::VARIABLE_RATE_SHADING_TIER2) &&
			wi::renderer::GetVariableRateShadingClassification())
		{
			uint32_t tileSize = device->GetVariableRateShadingTileSize();

			TextureDesc desc;
			desc.layout = ResourceState::UNORDERED_ACCESS;
			desc.bind_flags = BindFlag::UNORDERED_ACCESS | BindFlag::SHADING_RATE;
			desc.format = Format::R8_UINT;
			desc.width = (internalResolution.x + tileSize - 1) / tileSize;
			desc.height = (internalResolution.y + tileSize - 1) / tileSize;

			device->CreateTexture(&desc, nullptr, &rtShadingRate);
			device->SetName(&rtShadingRate, "rtShadingRate");
		}

		// Depth buffers:
		{
			TextureDesc desc;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;

			desc.sample_count = getMSAASampleCount();
			desc.layout = ResourceState::DEPTHSTENCIL;
			desc.format = wi::renderer::format_depthbuffer_main;
			desc.bind_flags = BindFlag::DEPTH_STENCIL;
			device->CreateTexture(&desc, nullptr, &depthBuffer_Main);
			device->SetName(&depthBuffer_Main, "depthBuffer_Main");

			desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
			desc.format = Format::R32_FLOAT;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.sample_count = 1;
			desc.mip_levels = 5;
			device->CreateTexture(&desc, nullptr, &depthBuffer_Copy);
			device->SetName(&depthBuffer_Copy, "depthBuffer_Copy");
			device->CreateTexture(&desc, nullptr, &depthBuffer_Copy1);
			device->SetName(&depthBuffer_Copy1, "depthBuffer_Copy1");

			for (uint32_t i = 0; i < depthBuffer_Copy.desc.mip_levels; ++i)
			{
				int subresource = 0;
				subresource = device->CreateSubresource(&depthBuffer_Copy, SubresourceType::SRV, 0, 1, i, 1);
				assert(subresource == i);
				subresource = device->CreateSubresource(&depthBuffer_Copy, SubresourceType::UAV, 0, 1, i, 1);
				assert(subresource == i);
				subresource = device->CreateSubresource(&depthBuffer_Copy1, SubresourceType::SRV, 0, 1, i, 1);
				assert(subresource == i);
				subresource = device->CreateSubresource(&depthBuffer_Copy1, SubresourceType::UAV, 0, 1, i, 1);
				assert(subresource == i);
			}
		}
		{
			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = Format::R32_FLOAT;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			desc.mip_levels = 5;
			desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
			device->CreateTexture(&desc, nullptr, &rtLinearDepth);
			device->SetName(&rtLinearDepth, "rtLinearDepth");

			for (uint32_t i = 0; i < desc.mip_levels; ++i)
			{
				int subresource_index;
				subresource_index = device->CreateSubresource(&rtLinearDepth, SubresourceType::SRV, 0, 1, i, 1);
				assert(subresource_index == i);
				subresource_index = device->CreateSubresource(&rtLinearDepth, SubresourceType::UAV, 0, 1, i, 1);
				assert(subresource_index == i);
			}
		}

		// GGMAX: debugUAV (viewport-res RGBA8, ~5 MB) is only consumed by debug visualizations
		// (light-culling debug, VRS classification debug, SurfelGI). Created on demand in
		// PreRender when one of those turns on; dropped here so a resize re-evaluates.
		debugUAV = {};
		wi::renderer::CreateTiledLightResources(tiledLightResources, internalResolution);
		wi::renderer::CreateScreenSpaceShadowResources(screenspaceshadowResources, internalResolution);

		// These can trigger resource creations if needed:
		setAO(ao);
		setSSREnabled(ssrEnabled);
		setSSGIEnabled(ssgiEnabled);
		setRaytracedReflectionsEnabled(raytracedReflectionsEnabled);
		setRaytracedDiffuseEnabled(raytracedDiffuseEnabled);
		setFSREnabled(fsrEnabled);
		setFSR2Enabled(fsr2Enabled);
		setEyeAdaptionEnabled(eyeAdaptionEnabled);
		setReflectionsEnabled(reflectionsEnabled);
		setBloomEnabled(bloomEnabled);
		setVolumeLightsEnabled(volumeLightsEnabled);
		setLightShaftsEnabled(lightShaftsEnabled);
		setOutlineEnabled(outlineEnabled);

		RenderPath2D::ResizeBuffers();
	}

	void RenderPath3D::PreUpdate()
	{
		camera_previous = *camera;
		camera_reflection_previous = camera_reflection;
	}

	void RenderPath3D::Update(float dt)
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		RenderPath2D::Update(dt);

		wi::renderer::SetShadowsEnabled(getShadowsEnabled());

		const bool hw_raytrace = device->CheckCapability(GraphicsDeviceCapability::RAYTRACING);
		if (getSceneUpdateEnabled())
		{
			if (wi::renderer::GetSurfelGIEnabled() ||
				wi::renderer::GetDDGIEnabled() ||
				(hw_raytrace && wi::renderer::GetRaytracedShadowsEnabled()) ||
				(hw_raytrace && getAO() == AO_RTAO) ||
				(hw_raytrace && getRaytracedReflectionEnabled()) ||
				(hw_raytrace && getRaytracedDiffuseEnabled())
				)
			{
				scene->SetAccelerationStructureUpdateRequested(true);
			}
			scene->camera = *camera;
			scene->Update(dt * wi::renderer::GetGameSpeed());
		}
	}

	void RenderPath3D::PreRender()
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		if (rtMain_render.desc.sample_count != msaaSampleCount)
		{
			ResizeBuffers();
		}

		// Frustum culling for main camera:
		visibility_main.layerMask = getLayerMask();
		visibility_main.scene = scene;
		visibility_main.camera = camera;
		visibility_main.flags = wi::renderer::Visibility::ALLOW_EVERYTHING;
		if (!getOcclusionCullingEnabled())
		{
			visibility_main.flags &= ~wi::renderer::Visibility::ALLOW_OCCLUSION_CULLING;
		}
		{
			auto gg_range = wi::profiler::BeginRangeCPU("RP3D-VisMain"); // GGMAX 1.32 instrumentation
			wi::renderer::UpdateVisibility(visibility_main);
			wi::profiler::EndRange(gg_range);
		}

		if (visibility_main.planar_reflection_visible)
		{
			// Frustum culling for planar reflections:
			camera_reflection = *camera;
			camera_reflection.jitter = XMFLOAT2(0, 0);
			camera_reflection.Reflect(visibility_main.reflectionPlane);
			visibility_reflection.layerMask = getLayerMask();
			visibility_reflection.scene = scene;
			visibility_reflection.camera = &camera_reflection;
			visibility_reflection.flags =
				wi::renderer::Visibility::ALLOW_OBJECTS |
				wi::renderer::Visibility::ALLOW_EMITTERS |
				wi::renderer::Visibility::ALLOW_HAIRS |
				wi::renderer::Visibility::ALLOW_LIGHTS
				;
			auto gg_range = wi::profiler::BeginRangeCPU("RP3D-VisRefl"); // GGMAX 1.32 instrumentation
			wi::renderer::UpdateVisibility(visibility_reflection);
			wi::profiler::EndRange(gg_range);
		}

		XMUINT2 internalResolution = GetInternalResolution();

		{
			auto gg_range = wi::profiler::BeginRangeCPU("RP3D-PerFrameData"); // GGMAX 1.32 instrumentation
			wi::renderer::UpdatePerFrameData(
				*scene,
				visibility_main,
				frameCB,
				getSceneUpdateEnabled() ? scene->dt : 0
			);
			wi::profiler::EndRange(gg_range);
		}

		if (getFSR2Enabled())
		{
			camera->jitter = fsr2Resources.GetJitter();
		}
		else if (wi::renderer::GetTemporalAAEnabled() && wi::renderer::GetWireframeMode() == wi::renderer::WIREFRAME_DISABLED)
		{
			const XMFLOAT4& halton = wi::math::GetHaltonSequence(wi::graphics::GetDevice()->GetFrameCount() % 256);
			camera->jitter.x = (halton.x * 2 - 1) / (float)internalResolution.x;
			camera->jitter.y = (halton.y * 2 - 1) / (float)internalResolution.y;
			if (!temporalAAResources.IsValid())
			{
				wi::renderer::CreateTemporalAAResources(temporalAAResources, internalResolution);
			}
		}
		else
		{
			camera->jitter = XMFLOAT2(0, 0);
			temporalAAResources = {};
		}

		camera_reflection.jitter = XMFLOAT2(0, 0);

		camera->UpdateCamera();
		if (visibility_main.planar_reflection_visible)
		{
			camera_reflection.UpdateCamera();
		}

		if (getAO() != AO_RTAO)
		{
			rtaoResources.frame = 0;
		}
		if (!wi::renderer::GetRaytracedShadowsEnabled())
		{
			rtshadowResources.frame = 0;
		}
		if (!getSSREnabled() && !getRaytracedReflectionEnabled())
		{
			rtSSR = {};
		}
		if (!getSSGIEnabled())
		{
			rtSSGI = {};
		}
		if (!getRaytracedDiffuseEnabled())
		{
			rtRaytracedDiffuse = {};
		}
		if (getAO() == AO_DISABLED)
		{
			rtAO = {};
		}

		if (wi::renderer::GetRaytracedShadowsEnabled() && device->CheckCapability(GraphicsDeviceCapability::RAYTRACING))
		{
			if (!rtshadowResources.denoised.IsValid())
			{
				wi::renderer::CreateRTShadowResources(rtshadowResources, internalResolution);
			}
		}
		else
		{
			rtshadowResources = {};
		}

		if (scene->weather.IsRealisticSky() && scene->weather.IsRealisticSkyAerialPerspective())
		{
			if (!aerialperspectiveResources.texture_output.IsValid())
			{
				wi::renderer::CreateAerialPerspectiveResources(aerialperspectiveResources, internalResolution);
			}
			if (getReflectionsEnabled() && depthBuffer_Reflection.IsValid())
			{
				if (!aerialperspectiveResources_reflection.texture_output.IsValid())
				{
					wi::renderer::CreateAerialPerspectiveResources(aerialperspectiveResources_reflection, XMUINT2(depthBuffer_Reflection.desc.width, depthBuffer_Reflection.desc.height));
				}
			}
			else
			{
				aerialperspectiveResources_reflection = {};
			}
		}
		else
		{
			aerialperspectiveResources = {};
		}

		if (scene->weather.IsVolumetricClouds())
		{
			if (!volumetriccloudResources.texture_cloudRender.IsValid())
			{
				wi::renderer::CreateVolumetricCloudResources(volumetriccloudResources, internalResolution);
			}
			if (getReflectionsEnabled() && depthBuffer_Reflection.IsValid())
			{
				if (!volumetriccloudResources_reflection.texture_cloudRender.IsValid())
				{
					wi::renderer::CreateVolumetricCloudResources(volumetriccloudResources_reflection, XMUINT2(depthBuffer_Reflection.desc.width, depthBuffer_Reflection.desc.height));
				}
			}
			else
			{
				volumetriccloudResources_reflection = {};
			}
			volumetriccloudResources.AdvanceFrame();
			volumetriccloudResources_reflection.AdvanceFrame();
		}
		else
		{
			volumetriccloudResources = {};
		}

		if (!scene->waterRipples.empty() && rtParticleDistortion.IsValid())
		{
			if (!rtWaterRipple.IsValid())
			{
				TextureDesc desc;
				desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;
				desc.format = Format::R16G16_FLOAT;
				desc.width = internalResolution.x / 8;
				desc.height = internalResolution.y / 8;
				assert(ComputeTextureMemorySizeInBytes(desc) <= ComputeTextureMemorySizeInBytes(rtParticleDistortion.desc)); // aliasing check
				device->CreateTexture(&desc, nullptr, &rtWaterRipple, &rtParticleDistortion); // aliased!
				device->SetName(&rtWaterRipple, "rtWaterRipple");
			}
		}
		else
		{
			rtWaterRipple = {};
		}

		if (wi::renderer::GetSurfelGIEnabled())
		{
			if (!surfelGIResources.result.IsValid())
			{
				wi::renderer::CreateSurfelGIResources(surfelGIResources, internalResolution);
			}
		}

		if (wi::renderer::GetVXGIEnabled())
		{
			if (!vxgiResources.IsValid())
			{
				wi::renderer::CreateVXGIResources(vxgiResources, internalResolution);
			}
		}
		else
		{
			vxgiResources = {};
		}

		// Check whether reprojected depth is required:
		if (!first_frame && wi::renderer::IsMeshShaderAllowed() && wi::renderer::IsMeshletOcclusionCullingEnabled())
		{
			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = Format::R16_UNORM;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			desc.mip_levels = GetMipCount(desc.width, desc.height, 1, 4);
			desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
			device->CreateTexture(&desc, nullptr, &reprojectedDepth);
			device->SetName(&reprojectedDepth, "reprojectedDepth");

			for (uint32_t i = 0; i < reprojectedDepth.desc.mip_levels; ++i)
			{
				int subresource_index;
				subresource_index = device->CreateSubresource(&reprojectedDepth, SubresourceType::SRV, 0, 1, i, 1);
				assert(subresource_index == i);
				subresource_index = device->CreateSubresource(&reprojectedDepth, SubresourceType::UAV, 0, 1, i, 1);
				assert(subresource_index == i);
			}
		}
		else
		{
			reprojectedDepth = {};
		}

		// Check whether velocity buffer is required:
		if (
			getMotionBlurEnabled() ||
			wi::renderer::GetTemporalAAEnabled() ||
			getSSREnabled() ||
			getSSGIEnabled() ||
			getRaytracedReflectionEnabled() ||
			getRaytracedDiffuseEnabled() ||
			wi::renderer::GetRaytracedShadowsEnabled() ||
			getAO() == AO::AO_RTAO ||
			wi::renderer::GetVariableRateShadingClassification() ||
			getFSR2Enabled() ||
			reprojectedDepth.IsValid()
			)
		{
			if (!rtVelocity.IsValid())
			{
				TextureDesc desc;
				desc.format = Format::R16G16_FLOAT;
				desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS | BindFlag::RENDER_TARGET;
				desc.width = internalResolution.x;
				desc.height = internalResolution.y;
				desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
				device->CreateTexture(&desc, nullptr, &rtVelocity);
				device->SetName(&rtVelocity, "rtVelocity");
			}
		}
		else
		{
			rtVelocity = {};
		}

		// Check whether shadow mask is required:
		if (wi::renderer::GetScreenSpaceShadowsEnabled() || wi::renderer::GetRaytracedShadowsEnabled())
		{
			if (!rtShadow.IsValid())
			{
				TextureDesc desc;
				desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
				desc.format = Format::R8_UNORM;
				desc.array_size = 16;
				desc.width = internalResolution.x;
				desc.height = internalResolution.y;
				desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
				device->CreateTexture(&desc, nullptr, &rtShadow);
				device->SetName(&rtShadow, "rtShadow");
			}
		}
		else
		{
			rtShadow = {};
		}

		if (getFSR2Enabled())
		{
			// FSR2 also acts as a temporal AA, so we inform the shaders about it here
			//	This will allow improved stochastic alpha test transparency
			frameCB.options |= OPTION_BIT_TEMPORALAA_ENABLED;
			uint x = frameCB.frame_count % 4;
			uint y = frameCB.frame_count / 4;
			frameCB.temporalaa_samplerotation = (x & 0x000000FF) | ((y & 0x000000FF) << 8);
		}

		// Check whether visibility resources are required:
		if (
			visibility_shading_in_compute ||
			getSSREnabled() ||
			getSSGIEnabled() ||
			getRaytracedReflectionEnabled() ||
			getRaytracedDiffuseEnabled() ||
			wi::renderer::GetScreenSpaceShadowsEnabled() ||
			wi::renderer::GetRaytracedShadowsEnabled() ||
			wi::renderer::GetVXGIEnabled()
			)
		{
			// GGMAX: payloads only exist for compute shading; recreate if that mode turns on later
			if (!visibilityResources.IsValid() ||
				(visibility_shading_in_compute && !visibilityResources.texture_payload_0.IsValid()))
			{
				wi::renderer::CreateVisibilityResources(visibilityResources, internalResolution, visibility_shading_in_compute);
			}
		}
		else
		{
			visibilityResources = {};
		}

		// GGMAX: lazy debugUAV — see ResizeBuffers. SurfelGI is included because its coverage
		// CS binds the UAV unconditionally while SurfelGI is enabled, not only in debug mode.
		{
			const bool need_debugUAV =
				wi::renderer::GetDebugLightCulling() ||
				wi::renderer::GetVariableRateShadingClassificationDebug() ||
				wi::renderer::GetSurfelGIEnabled() ||
				wi::renderer::GetSurfelGIDebugEnabled();
			if (need_debugUAV && (!debugUAV.IsValid() || debugUAV.desc.width != internalResolution.x || debugUAV.desc.height != internalResolution.y))
			{
				TextureDesc desc;
				desc.width = internalResolution.x;
				desc.height = internalResolution.y;
				desc.mip_levels = 1;
				desc.array_size = 1;
				desc.format = Format::R8G8B8A8_UNORM;
				desc.sample_count = 1;
				desc.usage = Usage::DEFAULT;
				desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
				desc.layout = ResourceState::SHADER_RESOURCE;
				device->CreateTexture(&desc, nullptr, &debugUAV);
				device->SetName(&debugUAV, "debugUAV");
			}
			else if (!need_debugUAV && debugUAV.IsValid())
			{
				debugUAV = {};
			}
		}

		// Check for depth of field allocation:
		if (getDepthOfFieldEnabled() &&
			getDepthOfFieldStrength() > 0 &&
			camera->aperture_size > 0
			)
		{
			if (!depthoffieldResources.IsValid())
			{
				XMUINT2 resolution = GetInternalResolution();
				if (getFSR2Enabled())
				{
					resolution = XMUINT2(GetPhysicalWidth(), GetPhysicalHeight());
				}
				wi::renderer::CreateDepthOfFieldResources(depthoffieldResources, resolution);
			}
		}
		else
		{
			depthoffieldResources = {};
		}

		// Check for motion blur allocation:
		if (getMotionBlurEnabled() && getMotionBlurStrength() > 0)
		{
			if (!motionblurResources.IsValid())
			{
				XMUINT2 resolution = GetInternalResolution();
				if (getFSR2Enabled())
				{
					resolution = XMUINT2(GetPhysicalWidth(), GetPhysicalHeight());
				}
				wi::renderer::CreateMotionBlurResources(motionblurResources, resolution);
			}
		}
		else
		{
			motionblurResources = {};
		}

		// Keep a copy of last frame's depth buffer for temporal disocclusion checks, so swap with current one every frame:
		std::swap(depthBuffer_Copy, depthBuffer_Copy1);

		visibilityResources.depthbuffer = &depthBuffer_Copy;
		visibilityResources.lineardepth = &rtLinearDepth;
		if (getMSAASampleCount() > 1)
		{
			visibilityResources.primitiveID_resolved = &rtPrimitiveID;
		}
		else
		{
			visibilityResources.primitiveID_resolved = nullptr;
		}

		camera->canvas.init(*this);
		camera->width = (float)internalResolution.x;
		camera->height = (float)internalResolution.y;
		camera->scissor = GetScissorInternalResolution();
		camera->sample_count = depthBuffer_Main.desc.sample_count;
		camera->shadercamera_options = SHADERCAMERA_OPTION_NONE;
		camera->texture_primitiveID_index = device->GetDescriptorIndex(&rtPrimitiveID, SubresourceType::SRV);
		camera->texture_depth_index = device->GetDescriptorIndex(&depthBuffer_Copy, SubresourceType::SRV);
		camera->texture_lineardepth_index = device->GetDescriptorIndex(&rtLinearDepth, SubresourceType::SRV);
		camera->texture_velocity_index = device->GetDescriptorIndex(&rtVelocity, SubresourceType::SRV);
		camera->texture_normal_index = device->GetDescriptorIndex(&visibilityResources.texture_normals, SubresourceType::SRV);
		camera->texture_roughness_index = device->GetDescriptorIndex(&visibilityResources.texture_roughness, SubresourceType::SRV);
		camera->buffer_entitytiles_index = device->GetDescriptorIndex(&tiledLightResources.entityTiles, SubresourceType::SRV);
		// GG delta: guard the planar-reflection descriptor index with the SAME condition that gates the
		// planar reflection render below (getReflectionsEnabled() && IsRequestedPlanarReflections()). Without
		// this, turning reflections OFF left texture_reflection_index pointing at rtReflection_resolved, which
		// setReflectionsEnabled(false) does NOT free and which is no longer rendered -> the ocean PS took its
		// planar branch and sampled stale/uninitialised GPU memory (bright blocky garbage on the water).
		// Forcing -1 when off makes oceanSurfacePS fall into its EnvironmentReflection_Global fallback (reflect
		// the sky/global probe) — stable, ~free, and matches how DX11 (getTransparent() bind) behaved.
		if (getReflectionsEnabled() && visibility_main.IsRequestedPlanarReflections())
		{
			camera->texture_reflection_index = device->GetDescriptorIndex(&rtReflection_resolved, SubresourceType::SRV);
			camera->texture_reflection_depth_index = device->GetDescriptorIndex(&depthBuffer_Reflection_resolved, SubresourceType::SRV);
		}
		else
		{
			camera->texture_reflection_index = -1;
			camera->texture_reflection_depth_index = -1;
		}
		camera->texture_refraction_index = device->GetDescriptorIndex(&rtSceneCopy, SubresourceType::SRV);
		camera->texture_waterriples_index = device->GetDescriptorIndex(&rtWaterRipple, SubresourceType::SRV);
		camera->texture_ao_index = device->GetDescriptorIndex(&rtAO, SubresourceType::SRV);
		camera->texture_ssr_index = device->GetDescriptorIndex(&rtSSR, SubresourceType::SRV);
		camera->texture_ssgi_index = device->GetDescriptorIndex(&rtSSGI, SubresourceType::SRV);
		if (rtShadow.IsValid())
		{
			camera->shadercamera_options |= SHADERCAMERA_OPTION_USE_SHADOW_MASK;
			camera->texture_rtshadow_index = device->GetDescriptorIndex(&rtShadow, SubresourceType::SRV);
		}
		else
		{
			camera->texture_rtshadow_index = device->GetDescriptorIndex(wi::texturehelper::getWhite(), SubresourceType::SRV); // AMD descriptor branching fix
		}
		camera->texture_rtdiffuse_index = device->GetDescriptorIndex(&rtRaytracedDiffuse, SubresourceType::SRV);
		camera->texture_surfelgi_index = device->GetDescriptorIndex(&surfelGIResources.result, SubresourceType::SRV);
		camera->texture_vxgi_diffuse_index = device->GetDescriptorIndex(&vxgiResources.diffuse, SubresourceType::SRV);
		if (wi::renderer::GetVXGIReflectionsEnabled())
		{
			camera->texture_vxgi_specular_index = device->GetDescriptorIndex(&vxgiResources.specular, SubresourceType::SRV);
		}
		else
		{
			camera->texture_vxgi_specular_index = -1;
		}
		camera->texture_reprojected_depth_index = device->GetDescriptorIndex(&reprojectedDepth, SubresourceType::SRV);

		camera_reflection.canvas.init(*this);
		camera_reflection.width = (float)depthBuffer_Reflection.desc.width;
		camera_reflection.height = (float)depthBuffer_Reflection.desc.height;
		camera_reflection.scissor.left = 0;
		camera_reflection.scissor.top = 0;
		camera_reflection.scissor.right = (int)depthBuffer_Reflection.desc.width;
		camera_reflection.scissor.bottom = (int)depthBuffer_Reflection.desc.height;
		camera_reflection.sample_count = depthBuffer_Reflection.desc.sample_count;
		camera_reflection.shadercamera_options = SHADERCAMERA_OPTION_NONE;
		camera_reflection.texture_primitiveID_index = -1;
		camera_reflection.texture_depth_index = device->GetDescriptorIndex(&depthBuffer_Reflection_resolved, SubresourceType::SRV);
		camera_reflection.texture_lineardepth_index = -1;
		camera_reflection.texture_velocity_index = -1;
		camera_reflection.texture_normal_index = -1;
		camera_reflection.texture_roughness_index = -1;
		camera_reflection.buffer_entitytiles_index = device->GetDescriptorIndex(&tiledLightResources_planarReflection.entityTiles, SubresourceType::SRV);
		camera_reflection.texture_reflection_index = -1;
		camera_reflection.texture_reflection_depth_index = -1;
		camera_reflection.texture_refraction_index = -1;
		camera_reflection.texture_waterriples_index = -1;
		camera_reflection.texture_ao_index = -1;
		camera_reflection.texture_ssr_index = -1;
		camera_reflection.texture_ssgi_index = -1;
		camera_reflection.texture_rtshadow_index = device->GetDescriptorIndex(wi::texturehelper::getWhite(), SubresourceType::SRV); // AMD descriptor branching fix
		camera_reflection.texture_rtdiffuse_index = -1;
		camera_reflection.texture_surfelgi_index = -1;
		camera_reflection.texture_vxgi_diffuse_index = -1;
		camera_reflection.texture_vxgi_specular_index = -1;
		camera_reflection.texture_reprojected_depth_index = -1;

		video_cmd = {};
		if (getSceneUpdateEnabled() && scene->videos.GetCount() > 0)
		{
			for (size_t i = 0; i < scene->videos.GetCount(); ++i)
			{
				const wi::scene::VideoComponent& video = scene->videos[i];
				if (wi::video::IsDecodingRequired(&video.videoinstance))
				{
					video_cmd = device->BeginCommandList(QUEUE_VIDEO_DECODE);
					break;
				}
			}
			for (size_t i = 0; i < scene->videos.GetCount(); ++i)
			{
				wi::scene::VideoComponent& video = scene->videos[i];
				wi::video::DecodeVideo(&video.videoinstance, video_cmd);
			}
		}

		if (getMeshBlendEnabled() && visibility_main.IsMeshBlendVisible())
		{
			if (!meshblendResources.IsValid())
			{
				wi::renderer::CreateMeshBlendResources(meshblendResources, internalResolution);
			}
		}
		else
		{
			meshblendResources = {};
		}

		prerender_happened = true;

		RenderPath2D::PreRender();
	}

	void RenderPath3D::Render() const
	{
		if (!prerender_happened)
		{
			// Since 0.71.694: PreRender must be called before Render() because it sets up rendering resources!
			//	The proper fix is to call PreRender() yourself for a manually managed RenderPath3D
			//	But if you don't do that, as a last resort it will be called here using const_cast
			assert(0);
			const_cast<RenderPath3D*>(this)->PreRender();
		}
		prerender_happened = false;

		GraphicsDevice* device = wi::graphics::GetDevice();
		wi::jobsystem::context ctx;

		// GGMAX 1.32: serial main-thread span of Render() (BeginCommandList + job dispatch + ocean record + RenderPath2D)
		auto gg_range_serial = wi::profiler::BeginRangeCPU("RP3D-RenderSerial");

		CommandList cmd_copypages;
		if (scene->terrains.GetCount() > 0)
		{
			cmd_copypages = device->BeginCommandList(gg_lean(QUEUE_COPY)); // GGMAX 1.48c
			wi::jobsystem::Execute(ctx, [this, cmd_copypages](wi::jobsystem::JobArgs args) {
				for (size_t i = 0; i < scene->terrains.GetCount(); ++i)
				{
					scene->terrains[i].CopyVirtualTexturePageStatusGPU(cmd_copypages);
				}
			});
		}

		// Preparing the frame:
		CommandList cmd = device->BeginCommandList();
		wi::renderer::ProcessDeferredTextureRequests(cmd); // Execute it first thing in the frame here, on main thread, to not allow other thread steal it and execute on different command list!
		CommandList cmd_prepareframe = cmd;
		wi::jobsystem::Execute(ctx, [this, cmd](wi::jobsystem::JobArgs args) {
			GraphicsDevice* device = wi::graphics::GetDevice();
			auto gg_range = wi::profiler::BeginRangeCPU("RP3D-rec PrepareFrame"); // GGMAX 1.32

			wi::renderer::BindCameraCB(
				*camera,
				camera_previous,
				camera_reflection,
				cmd
			);
			wi::renderer::UpdateRenderData(visibility_main, frameCB, cmd);

			// GGMAX: debugUAV barrier only when the lazy texture exists
			GPUBarrier barriers[3];
			uint32_t num_barriers = 0;
			if (debugUAV.IsValid())
			{
				barriers[num_barriers++] = GPUBarrier::Image(&debugUAV, debugUAV.desc.layout, ResourceState::UNORDERED_ACCESS);
			}
			barriers[num_barriers++] = GPUBarrier::Aliasing(&rtPostprocess, &rtPrimitiveID);
			if (visibility_shading_in_compute)
			{
				barriers[num_barriers++] = GPUBarrier::Image(&rtMain, rtMain.desc.layout, ResourceState::SHADER_RESOURCE_COMPUTE); // prepares transition for discard in dx12
			}
			device->Barrier(barriers, num_barriers, cmd);

			wi::profiler::EndRange(gg_range); // GGMAX 1.32
		});

		// async compute parallel with depth prepass
		cmd = device->BeginCommandList(QUEUE_COMPUTE);
		CommandList cmd_prepareframe_async = cmd;
		device->WaitCommandList(cmd, cmd_prepareframe);
		if (cmd_copypages.IsValid())
		{
			device->WaitCommandList(cmd, cmd_copypages);
		}
		wi::jobsystem::Execute(ctx, [this, cmd](wi::jobsystem::JobArgs args) {
			auto gg_range = wi::profiler::BeginRangeCPU("RP3D-rec PrepareAsync"); // GGMAX 1.32

			wi::renderer::BindCameraCB(
				*camera,
				camera_previous,
				camera_reflection,
				cmd
			);
			wi::renderer::UpdateRenderDataAsync(visibility_main, frameCB, cmd);

			if (scene->IsWetmapProcessingRequired())
			{
				wi::renderer::RefreshWetmaps(visibility_main, cmd);
			}

			if (scene->IsAccelerationStructureUpdateRequested())
			{
				wi::renderer::UpdateRaytracingAccelerationStructures(*scene, cmd);
			}

			if (wi::renderer::GetSurfelGIEnabled())
			{
				wi::renderer::SurfelGI(
					surfelGIResources,
					*scene,
					cmd
				);
			}

			if (wi::renderer::GetDDGIEnabled() && getSceneUpdateEnabled())
			{
				wi::renderer::DDGI(
					*scene,
					cmd
				);
			}

			wi::profiler::EndRange(gg_range); // GGMAX 1.32
		});

		static const uint32_t drawscene_flags =
			wi::renderer::DRAWSCENE_OPAQUE |
			wi::renderer::DRAWSCENE_IMPOSTOR |
			wi::renderer::DRAWSCENE_HAIRPARTICLE |
			wi::renderer::DRAWSCENE_TESSELLATION |
			wi::renderer::DRAWSCENE_OCCLUSIONCULLING |
			wi::renderer::DRAWSCENE_MAINCAMERA
			;

		// GGMAX 1.40: occlusion-culling recording body, shared between the stock own-list path
		// and the merged prepass-tail path. Safe on the prepass list: it renders against
		// depthBuffer_Main which the prepass just finished writing (same-list sequencing).
		auto gg_record_occlusion = [this](CommandList cmd) {
			GraphicsDevice* device = wi::graphics::GetDevice();
			auto gg_range = wi::profiler::BeginRangeCPU("RP3D-rec Occlusion"); // GGMAX 1.32

			device->EventBegin("Occlusion Culling", cmd);
			ScopedGPUProfiling("Occlusion Culling", cmd);

			wi::renderer::BindCameraCB(
				*camera,
				camera_previous,
				camera_reflection,
				cmd
			);

			wi::renderer::OcclusionCulling_Reset(visibility_main, cmd); // must be outside renderpass!

			RenderPassImage rp[] = {
				RenderPassImage::DepthStencil(&depthBuffer_Main),
			};
			device->RenderPassBegin(rp, arraysize(rp), cmd);

			Rect scissor = GetScissorInternalResolution();
			device->BindScissorRects(1, &scissor, cmd);

			Viewport vp;
			vp.width = (float)depthBuffer_Main.GetDesc().width;
			vp.height = (float)depthBuffer_Main.GetDesc().height;
			device->BindViewports(1, &vp, cmd);

			wi::renderer::OcclusionCulling_Render(*camera, visibility_main, cmd);

			device->RenderPassEnd(cmd);

			wi::renderer::OcclusionCulling_Resolve(visibility_main, cmd); // must be outside renderpass!

			device->EventEnd(cmd);
			wi::profiler::EndRange(gg_range); // GGMAX 1.32
		};

		// Main camera depth prepass:
		cmd = device->BeginCommandList();
		CommandList cmd_maincamera_prepass = cmd;
		const bool gg_merge_occlusion = gg_render_merge_lists && getOcclusionCullingEnabled(); // GGMAX 1.40
		wi::jobsystem::Execute(ctx, [this, cmd, gg_merge_occlusion, &gg_record_occlusion](wi::jobsystem::JobArgs args) {

			GraphicsDevice* device = wi::graphics::GetDevice();
			auto gg_range_cpu = wi::profiler::BeginRangeCPU("RP3D-rec Prepass"); // GGMAX 1.32

			wi::renderer::BindCameraCB(
				*camera,
				camera_previous,
				camera_reflection,
				cmd
			);

			wi::renderer::RefreshImpostors(*scene, cmd);

			if (reprojectedDepth.IsValid())
			{
				wi::renderer::ComputeReprojectedDepthPyramid(
					depthBuffer_Copy,
					rtVelocity,
					reprojectedDepth,
					cmd
				);
			}

			RenderPassImage rp[] = {
				RenderPassImage::DepthStencil(
					&depthBuffer_Main,
					RenderPassImage::LoadOp::CLEAR,
					RenderPassImage::StoreOp::STORE,
					ResourceState::DEPTHSTENCIL,
					ResourceState::DEPTHSTENCIL,
					ResourceState::DEPTHSTENCIL
				),
				RenderPassImage::RenderTarget(
					&rtPrimitiveID_render,
					RenderPassImage::LoadOp::CLEAR,
					RenderPassImage::StoreOp::STORE,
					ResourceState::SHADER_RESOURCE_COMPUTE,
					ResourceState::SHADER_RESOURCE_COMPUTE
				),
			};
			device->RenderPassBegin(rp, arraysize(rp), cmd);

			device->EventBegin("Opaque Z-prepass", cmd);
			auto range = wi::profiler::BeginRangeGPU("Z-Prepass", cmd);

			Rect scissor = GetScissorInternalResolution();
			device->BindScissorRects(1, &scissor, cmd);

			Viewport vp;
			vp.width = (float)depthBuffer_Main.GetDesc().width;
			vp.height = (float)depthBuffer_Main.GetDesc().height;

			// Foreground:
			vp.min_depth = 1 - foreground_depth_range;
			vp.max_depth = 1;
			device->BindViewports(1, &vp, cmd);
			wi::renderer::DrawScene(
				visibility_main,
				RENDERPASS_PREPASS,
				cmd,
				wi::renderer::DRAWSCENE_OPAQUE |
				wi::renderer::DRAWSCENE_FOREGROUND_ONLY |
				wi::renderer::DRAWSCENE_MAINCAMERA
			);

			// Regular:
			vp.min_depth = 0;
			vp.max_depth = 1;
			device->BindViewports(1, &vp, cmd);
			wi::renderer::DrawScene(
				visibility_main,
				RENDERPASS_PREPASS,
				cmd,
				drawscene_flags
			);

			// GGMAX 2.09: a second Z-prepass pass for the FEW transparent-pass materials that are
			// actually solid — the DX11 fork did exactly this, and its comment names the case:
			//   "write depth for transparent objects that are solid (opaque=100%) like guns and
			//    solid doors with windows in"  (WickedRepo/WickedEngine/RenderPath3D.cpp:1056)
			// The first-person weapon has to live in the transparent pass so its depth carve lands
			// after the world's opaque depth, but it IS a solid object, and everything downstream that
			// reads the prepass — GPU occlusion queries, the light-shaft sun cutout, velocity
			// reconstruction from the primitive-ID buffer, SSAO — would otherwise behave as if the
			// weapon were not there. Worst case without it: the occlusion query sees the weapon as
			// fully hidden behind the very crate it is supposed to carve through, and culls it.
			// RenderMeshes admits ONLY materials carrying GG_FORCEDEPTH to this pass, so the cost is
			// one DrawScene over a handful of subsets, not over the whole transparent set.
			if (wi::renderer::gg_weapon_forcedepth)
			{
				wi::renderer::DrawScene(
					visibility_main,
					RENDERPASS_PREPASS,
					cmd,
					wi::renderer::DRAWSCENE_TRANSPARENT |
					wi::renderer::DRAWSCENE_MAINCAMERA
				);
			}

			// Custom scene draw (terrain/trees/grass depth prepass):
			// GGMAX 2.13: state-safe hook boundary (see the transparent hook / game task #120)
			if (customDraw_Prepass)
			{
				customDraw_Prepass(&camera->frustum, cmd);
				device->GG_InvalidateCommandListState(cmd);
				wi::renderer::BindCameraCB(*camera, camera_previous, camera_reflection, cmd);
				wi::renderer::BindCommonResources(cmd);
			}

			wi::profiler::EndRange(range);
			device->EventEnd(cmd);

			device->RenderPassEnd(cmd);

			// After prepass render pass: virtual texture readback (compute + copy, must be outside render pass)
			// GGMAX 2.13: state-safe hook boundary (see the transparent hook / game task #120)
			if (customDraw_AfterPrepass)
			{
				customDraw_AfterPrepass(rtPrimitiveID_render, getMSAASampleCount(), cmd);
				wi::graphics::GetDevice()->GG_InvalidateCommandListState(cmd);
			}

			// GGMAX 1.40 (a): record the occlusion pass at the tail of this list instead of its own
			if (gg_merge_occlusion)
			{
				gg_record_occlusion(cmd);
			}

			wi::profiler::EndRange(gg_range_cpu); // GGMAX 1.32
		});

		// Main camera compute effects:
		//	(async compute, parallel to "shadow maps" and "update textures",
		//	must finish before "main scene opaque color pass")
		cmd = device->BeginCommandList(QUEUE_COMPUTE);
		device->WaitCommandList(cmd, cmd_maincamera_prepass);
		if (video_cmd.IsValid())
		{
			device->WaitCommandList(cmd, video_cmd);
		}
		CommandList cmd_maincamera_compute_effects = cmd;
		wi::jobsystem::Execute(ctx, [this, cmd](wi::jobsystem::JobArgs args) {

			GraphicsDevice* device = wi::graphics::GetDevice();
			auto gg_range = wi::profiler::BeginRangeCPU("RP3D-rec ComputeFX"); // GGMAX 1.32

			for (size_t i = 0; i < scene->videos.GetCount(); ++i)
			{
				wi::scene::VideoComponent& video = scene->videos[i];
				wi::video::ResolveVideoToRGB(&video.videoinstance, cmd);
			}

			wi::renderer::BindCameraCB(
				*camera,
				camera_previous,
				camera_reflection,
				cmd
			);

			wi::renderer::Visibility_Prepare(
				visibilityResources,
				rtPrimitiveID_render,
				cmd
			);

			wi::renderer::ComputeTiledLightCulling(
				tiledLightResources,
				visibility_main,
				debugUAV,
				cmd
			);

			if (visibility_shading_in_compute)
			{
				wi::renderer::Visibility_Surface(
					visibilityResources,
					rtMain,
					cmd
				);
			}
			else if (
				getSSREnabled() ||
				getSSGIEnabled() ||
				getRaytracedReflectionEnabled() ||
				getRaytracedDiffuseEnabled() ||
				wi::renderer::GetScreenSpaceShadowsEnabled() ||
				wi::renderer::GetRaytracedShadowsEnabled() ||
				wi::renderer::GetVXGIEnabled()
				)
			{
				// These post effects require surface normals and/or roughness
				wi::renderer::Visibility_Surface_Reduced(
					visibilityResources,
					cmd
				);
			}

			if (rtVelocity.IsValid())
			{
				wi::renderer::Visibility_Velocity(
					rtVelocity,
					cmd
				);
			}

			if (wi::renderer::GetSurfelGIEnabled())
			{
				wi::renderer::SurfelGI_Coverage(
					surfelGIResources,
					*scene,
					rtLinearDepth,
					debugUAV,
					cmd
				);
			}

			RenderAO(cmd);

			if (wi::renderer::GetVariableRateShadingClassification() && device->CheckCapability(GraphicsDeviceCapability::VARIABLE_RATE_SHADING_TIER2))
			{
				wi::renderer::ComputeShadingRateClassification(
					rtShadingRate,
					debugUAV,
					cmd
				);
			}

			RenderSSR(cmd);

			RenderSSGI(cmd);

			if (wi::renderer::GetScreenSpaceShadowsEnabled())
			{
				wi::renderer::Postprocess_ScreenSpaceShadow(
					screenspaceshadowResources,
					tiledLightResources.entityTiles,
					rtLinearDepth,
					rtShadow,
					cmd,
					getScreenSpaceShadowRange(),
					getScreenSpaceShadowSampleCount()
				);
			}

			if (wi::renderer::GetRaytracedShadowsEnabled() && wi::renderer::GetWireframeMode() == wi::renderer::WIREFRAME_DISABLED)
			{
				wi::renderer::Postprocess_RTShadow(
					rtshadowResources,
					*scene,
					tiledLightResources.entityTiles,
					rtLinearDepth,
					rtShadow,
					cmd
				);
			}
			if (scene->weather.IsVolumetricClouds() && !scene->weather.IsVolumetricCloudsReceiveShadow())
			{
				// When volumetric cloud DOESN'T receive shadow it can be done async to shadow maps!
				wi::renderer::Postprocess_VolumetricClouds(
					volumetriccloudResources,
					cmd,
					*camera,
					camera_previous,
					camera_reflection,
					(wi::renderer::GetTemporalAAEnabled() && wi::renderer::GetWireframeMode() == wi::renderer::WIREFRAME_DISABLED) || getFSR2Enabled(),
					scene->weather.volumetricCloudsWeatherMapFirst.IsValid() ? &scene->weather.volumetricCloudsWeatherMapFirst.GetTexture() : nullptr,
					scene->weather.volumetricCloudsWeatherMapSecond.IsValid() ? &scene->weather.volumetricCloudsWeatherMapSecond.GetTexture() : nullptr
				);
			}
			if (getMeshBlendEnabled() && visibility_main.IsMeshBlendVisible())
			{
				wi::renderer::PostProcess_MeshBlend_EdgeProcess(meshblendResources, cmd);
			}

			wi::profiler::EndRange(gg_range); // GGMAX 1.32
		});

		// Occlusion culling (stock own-list path — GGMAX 1.40 records it on the prepass list instead):
		CommandList cmd_occlusionculling;
		if (getOcclusionCullingEnabled() && !gg_merge_occlusion)
		{
			cmd = device->BeginCommandList();
			cmd_occlusionculling = cmd;
			wi::jobsystem::Execute(ctx, [this, cmd, &gg_record_occlusion](wi::jobsystem::JobArgs args) {
				gg_record_occlusion(cmd);
			});
		}

		CommandList cmd_ocean;
		if (scene->weather.IsOceanEnabled() && scene->ocean.IsValid())
		{
			// Ocean simulation can be updated async to opaque passes:
			cmd_ocean = device->BeginCommandList(gg_lean(QUEUE_COMPUTE)); // GGMAX 1.48c (sim is ~0.2ms — fences cost more than the overlap wins)
			if (cmd_occlusionculling.IsValid())
			{
				// Ocean occlusion culling must be waited
				device->WaitCommandList(cmd_ocean, cmd_occlusionculling);
			}
			else if (gg_merge_occlusion)
			{
				// GGMAX 1.40 (a): occlusion now records at the tail of the prepass list
				device->WaitCommandList(cmd_ocean, cmd_maincamera_prepass);
			}
			wi::renderer::UpdateOcean(visibility_main, cmd_ocean);

			// Copying to readback is done on copy queue to use DMA instead of compute warps:
			CommandList cmd_oceancopy = device->BeginCommandList(gg_lean(QUEUE_COPY)); // GGMAX 1.48c
			device->WaitCommandList(cmd_oceancopy, cmd_ocean);
			wi::renderer::ReadbackOcean(visibility_main, cmd_oceancopy);
		}

		// Shadow maps:
		if (getShadowsEnabled())
		{
			cmd = device->BeginCommandList();
			wi::jobsystem::Execute(ctx, [this, cmd](wi::jobsystem::JobArgs args) {
				wi::renderer::DrawShadowmaps(visibility_main, cmd);
			});
		}

		if (wi::renderer::GetVXGIEnabled() && getSceneUpdateEnabled())
		{
			cmd = device->BeginCommandList();
			wi::jobsystem::Execute(ctx, [cmd, this](wi::jobsystem::JobArgs args) {
				wi::renderer::VXGI_Voxelize(visibility_main, cmd);
			});
		}

		// Updating textures:
		if (getSceneUpdateEnabled())
		{
			cmd = device->BeginCommandList();
			device->WaitCommandList(cmd, cmd_prepareframe_async);
			wi::jobsystem::Execute(ctx, [cmd, this](wi::jobsystem::JobArgs args) {
				auto gg_range = wi::profiler::BeginRangeCPU("RP3D-rec UpdateTex"); // GGMAX 1.32
				wi::renderer::BindCommonResources(cmd);
				wi::renderer::BindCameraCB(
					*camera,
					camera_previous,
					camera_reflection,
					cmd
				);
				wi::renderer::RefreshLightmaps(*scene, cmd);
				wi::renderer::RefreshEnvProbes(visibility_main, cmd);
				wi::renderer::PaintDecals(*scene, cmd);
				wi::profiler::EndRange(gg_range); // GGMAX 1.32
			});
		}

		if (getReflectionsEnabled() && visibility_main.IsRequestedPlanarReflections())
		{
			// Planar reflections depth prepass:
			cmd = device->BeginCommandList();
			wi::jobsystem::Execute(ctx, [cmd, this](wi::jobsystem::JobArgs args) {

				GraphicsDevice* device = wi::graphics::GetDevice();

				wi::renderer::BindCameraCB(
					camera_reflection,
					camera_reflection_previous,
					camera_reflection,
					cmd
				);

				device->EventBegin("Planar reflections Z-Prepass", cmd);
				auto range = wi::profiler::BeginRangeGPU("Planar Reflections Z-Prepass", cmd);

				RenderPassImage rp[] = {
					RenderPassImage::DepthStencil(
						&depthBuffer_Reflection,
						RenderPassImage::LoadOp::CLEAR,
						RenderPassImage::StoreOp::STORE,
						ResourceState::SHADER_RESOURCE,
						ResourceState::DEPTHSTENCIL,
						ResourceState::SHADER_RESOURCE
					)
				};
				device->RenderPassBegin(rp, arraysize(rp), cmd);

				Viewport vp;
				vp.width = (float)depthBuffer_Reflection.GetDesc().width;
				vp.height = (float)depthBuffer_Reflection.GetDesc().height;
				vp.min_depth = 0;
				vp.max_depth = 1;
				device->BindViewports(1, &vp, cmd);

				wi::renderer::DrawScene(
					visibility_reflection,
					RENDERPASS_PREPASS_DEPTHONLY,
					cmd,
					wi::renderer::DRAWSCENE_OPAQUE |
					wi::renderer::DRAWSCENE_IMPOSTOR |
					wi::renderer::DRAWSCENE_HAIRPARTICLE |
					wi::renderer::DRAWSCENE_SKIP_PLANAR_REFLECTION_OBJECTS
				);

				// Custom scene draw (terrain/trees reflection depth prepass):
				// GGMAX 2.13: state-safe hook boundary (see the transparent hook / game task #120)
				if (customDraw_Prepass_Reflections)
				{
					customDraw_Prepass_Reflections(&camera_reflection.frustum, cmd);
					wi::graphics::GetDevice()->GG_InvalidateCommandListState(cmd);
					wi::renderer::BindCameraCB(camera_reflection, camera_reflection_previous, camera_reflection, cmd);
					wi::renderer::BindCommonResources(cmd);
				}

				device->RenderPassEnd(cmd);

				wi::renderer::ResolveMSAADepthBuffer(depthBuffer_Reflection_resolved, depthBuffer_Reflection, cmd);

				if (scene->weather.IsRealisticSky() && scene->weather.IsRealisticSkyAerialPerspective())
				{
					wi::renderer::Postprocess_AerialPerspective(
						aerialperspectiveResources_reflection,
						cmd
					);
				}

				wi::profiler::EndRange(range); // Planar Reflections
				device->EventEnd(cmd);

			});

			// Planar reflections color pass:
			cmd = device->BeginCommandList();
			wi::jobsystem::Execute(ctx, [cmd, this](wi::jobsystem::JobArgs args) {

				GraphicsDevice* device = wi::graphics::GetDevice();

				wi::renderer::BindCameraCB(
					camera_reflection,
					camera_reflection_previous,
					camera_reflection,
					cmd
				);

				wi::renderer::ComputeTiledLightCulling(
					tiledLightResources_planarReflection,
					visibility_reflection,
					Texture(),
					cmd
				);

				if (scene->weather.IsVolumetricClouds())
				{
					wi::renderer::Postprocess_VolumetricClouds(
						volumetriccloudResources_reflection,
						cmd,
						camera_reflection,
						camera_reflection_previous,
						camera_reflection,
						(wi::renderer::GetTemporalAAEnabled() && wi::renderer::GetWireframeMode() == wi::renderer::WIREFRAME_DISABLED) || getFSR2Enabled(),
						scene->weather.volumetricCloudsWeatherMapFirst.IsValid() ? &scene->weather.volumetricCloudsWeatherMapFirst.GetTexture() : nullptr,
						scene->weather.volumetricCloudsWeatherMapSecond.IsValid() ? &scene->weather.volumetricCloudsWeatherMapSecond.GetTexture() : nullptr
					);
				}
				device->EventBegin("Planar reflections", cmd);
				auto range = wi::profiler::BeginRangeGPU("Planar Reflections", cmd);

				RenderPassImage rp[] = {
					RenderPassImage::RenderTarget(
						&rtReflection,
						RenderPassImage::LoadOp::CLEAR,
						RenderPassImage::StoreOp::DONTCARE,
						ResourceState::RENDERTARGET,
						ResourceState::RENDERTARGET
					),
					RenderPassImage::Resolve(&rtReflection_resolved),
					RenderPassImage::DepthStencil(
						&depthBuffer_Reflection,
						RenderPassImage::LoadOp::LOAD,
						RenderPassImage::StoreOp::STORE,
						ResourceState::SHADER_RESOURCE,
						ResourceState::DEPTHSTENCIL,
						ResourceState::SHADER_RESOURCE
					),
				};
				device->RenderPassBegin(rp, arraysize(rp), cmd);

				Viewport vp;
				vp.width = (float)depthBuffer_Reflection.GetDesc().width;
				vp.height = (float)depthBuffer_Reflection.GetDesc().height;
				vp.min_depth = 0;
				vp.max_depth = 1;
				device->BindViewports(1, &vp, cmd);

				wi::renderer::DrawScene(
					visibility_reflection,
					RENDERPASS_MAIN,
					cmd,
					wi::renderer::DRAWSCENE_OPAQUE |
					wi::renderer::DRAWSCENE_IMPOSTOR |
					wi::renderer::DRAWSCENE_HAIRPARTICLE |
					wi::renderer::DRAWSCENE_SKIP_PLANAR_REFLECTION_OBJECTS
				);
				// Custom scene draw (terrain/trees reflection opaque):
				// GGMAX 2.13: state-safe hook boundary (see the transparent hook / game task #120)
				if (customDraw_Opaque)
				{
					customDraw_Opaque(&camera_reflection.frustum, 1, cmd);
					wi::graphics::GetDevice()->GG_InvalidateCommandListState(cmd);
					wi::renderer::BindCameraCB(camera_reflection, camera_reflection_previous, camera_reflection, cmd);
					wi::renderer::BindCommonResources(cmd);
				}
				wi::renderer::DrawSky(*scene, cmd);
				wi::renderer::DrawScene(
					visibility_reflection,
					RENDERPASS_MAIN,
					cmd,
					wi::renderer::DRAWSCENE_TRANSPARENT |
					wi::renderer::DRAWSCENE_SKIP_PLANAR_REFLECTION_OBJECTS
				); // separate renderscene, to be drawn after opaque and transparent sort order

				if (scene->weather.IsRealisticSky() && scene->weather.IsRealisticSkyAerialPerspective())
				{
					// Blend Aerial Perspective on top:
					device->EventBegin("Aerial Perspective Reflection Blend", cmd);
					wi::image::Params fx;
					fx.enableFullScreen();
					fx.blendFlag = BLENDMODE_PREMULTIPLIED;
					wi::image::Draw(&aerialperspectiveResources_reflection.texture_output, fx, cmd);
					device->EventEnd(cmd);
				}

				// Blend the volumetric clouds on top:
				//	For planar reflections, we don't use upsample, because there is no linear depth here
				if (scene->weather.IsVolumetricClouds())
				{
					device->EventBegin("Volumetric Clouds Reflection Blend", cmd);
					wi::image::Params fx;
					fx.enableFullScreen();
					fx.blendFlag = BLENDMODE_PREMULTIPLIED;
					wi::image::Draw(&volumetriccloudResources_reflection.texture_reproject[volumetriccloudResources_reflection.GetTemporalOutputIndex()], fx, cmd);
					device->EventEnd(cmd);
				}

				wi::renderer::DrawSoftParticles(visibility_reflection, false, cmd);
				wi::renderer::DrawSpritesAndFonts(*scene, camera_reflection, false, cmd);

				device->RenderPassEnd(cmd);

				wi::profiler::EndRange(range); // Planar Reflections
				device->EventEnd(cmd);
			});
		}

		// Main camera opaque color pass:
		cmd = device->BeginCommandList();
		device->WaitCommandList(cmd, cmd_maincamera_compute_effects);
		wi::jobsystem::Execute(ctx, [this, cmd](wi::jobsystem::JobArgs args) {

			GraphicsDevice* device = wi::graphics::GetDevice();
			auto gg_range_cpu = wi::profiler::BeginRangeCPU("RP3D-rec Opaque"); // GGMAX 1.32
			device->EventBegin("Opaque Scene", cmd);

			wi::renderer::BindCameraCB(
				*camera,
				camera_previous,
				camera_reflection,
				cmd
			);

			if (scene->weather.IsRealisticSky() && scene->weather.IsRealisticSkyAerialPerspective())
			{
				wi::renderer::Postprocess_AerialPerspective(
					aerialperspectiveResources,
					cmd
				);
			}
			if (scene->weather.IsVolumetricClouds() && scene->weather.IsVolumetricCloudsReceiveShadow())
			{
				// When volumetric cloud receives shadow it must be done AFTER shadow maps!
				wi::renderer::Postprocess_VolumetricClouds(
					volumetriccloudResources,
					cmd,
					*camera,
					camera_previous,
					camera_reflection,
					(wi::renderer::GetTemporalAAEnabled() && wi::renderer::GetWireframeMode() == wi::renderer::WIREFRAME_DISABLED) || getFSR2Enabled(),
					scene->weather.volumetricCloudsWeatherMapFirst.IsValid() ? &scene->weather.volumetricCloudsWeatherMapFirst.GetTexture() : nullptr,
					scene->weather.volumetricCloudsWeatherMapSecond.IsValid() ? &scene->weather.volumetricCloudsWeatherMapSecond.GetTexture() : nullptr
				);
			}
			if (getRaytracedReflectionEnabled())
			{
				wi::renderer::Postprocess_RTReflection(
					rtreflectionResources,
					*scene,
					rtSSR,
					cmd,
					getRaytracedReflectionsRange(),
					getReflectionRoughnessCutoff()
				);
			}
			if (getRaytracedDiffuseEnabled())
			{
				wi::renderer::Postprocess_RTDiffuse(
					rtdiffuseResources,
					*scene,
					rtRaytracedDiffuse,
					cmd,
					getRaytracedDiffuseRange()
				);
			}
			if (wi::renderer::GetVXGIEnabled())
			{
				wi::renderer::VXGI_Resolve(
					vxgiResources,
					*scene,
					rtLinearDepth,
					cmd
				);
			}

			// Depth buffers were created on COMPUTE queue, so make them available for pixel shaders here:
			{
				GPUBarrier barriers[] = {
					GPUBarrier::Image(&rtLinearDepth, rtLinearDepth.desc.layout, ResourceState::SHADER_RESOURCE),
					GPUBarrier::Image(&depthBuffer_Copy, depthBuffer_Copy.desc.layout, ResourceState::SHADER_RESOURCE),
				};
				device->Barrier(barriers, arraysize(barriers), cmd);
			}

			if (wi::renderer::GetRaytracedShadowsEnabled() || wi::renderer::GetScreenSpaceShadowsEnabled())
			{
				GPUBarrier barrier = GPUBarrier::Image(&rtShadow, rtShadow.desc.layout, ResourceState::SHADER_RESOURCE);
				device->Barrier(&barrier, 1, cmd);
			}

			if (visibility_shading_in_compute)
			{
				wi::renderer::Visibility_Shade(
					visibilityResources,
					rtMain,
					cmd
				);
			}

			Viewport vp;
			vp.width = (float)depthBuffer_Main.GetDesc().width;
			vp.height = (float)depthBuffer_Main.GetDesc().height;
			device->BindViewports(1, &vp, cmd);

			Rect scissor = GetScissorInternalResolution();
			device->BindScissorRects(1, &scissor, cmd);

			if (getOutlineEnabled())
			{
				// Cut off outline source from linear depth:
				device->EventBegin("Outline Source", cmd);

				RenderPassImage rp[] = {
					RenderPassImage::RenderTarget(&rtOutlineSource, RenderPassImage::LoadOp::CLEAR),
					RenderPassImage::DepthStencil(&depthBuffer_Main, RenderPassImage::LoadOp::LOAD)
				};
				device->RenderPassBegin(rp, arraysize(rp), cmd);
				wi::image::Params params;
				params.enableFullScreen();
				params.stencilRefMode = wi::image::STENCILREFMODE_ENGINE;
				params.stencilComp = wi::image::STENCILMODE_EQUAL;
				params.stencilRef = wi::enums::STENCILREF_OUTLINE;
				wi::image::Draw(&rtLinearDepth, params, cmd);
				params.stencilRef = wi::enums::STENCILREF_CUSTOMSHADER_OUTLINE;
				wi::image::Draw(&rtLinearDepth, params, cmd);
				device->RenderPassEnd(cmd);
				device->EventEnd(cmd);
			}

			RenderPassImage rp[4] = {};
			uint32_t rp_count = 0;
			rp[rp_count++] = RenderPassImage::RenderTarget(
				&rtMain_render,
				visibility_shading_in_compute ? RenderPassImage::LoadOp::LOAD : RenderPassImage::LoadOp::CLEAR
			);
			if (getMSAASampleCount() > 1)
			{
				rp[rp_count++] = RenderPassImage::Resolve(&rtMain);
			}
			if (device->CheckCapability(GraphicsDeviceCapability::VARIABLE_RATE_SHADING_TIER2) && rtShadingRate.IsValid())
			{
				rp[rp_count++] = RenderPassImage::ShadingRateSource(&rtShadingRate, ResourceState::UNORDERED_ACCESS, ResourceState::UNORDERED_ACCESS);
			}
			rp[rp_count++] = RenderPassImage::DepthStencil(
				&depthBuffer_Main,
				RenderPassImage::LoadOp::LOAD,
				RenderPassImage::StoreOp::STORE,
				ResourceState::DEPTHSTENCIL,
				ResourceState::DEPTHSTENCIL,
				ResourceState::DEPTHSTENCIL
			);
			device->RenderPassBegin(rp, rp_count, cmd, RenderPassFlags::ALLOW_UAV_WRITES);

			if (visibility_shading_in_compute)
			{
				// In visibility compute shading, the impostors must still be drawn using rasterization:
				wi::renderer::DrawScene(
					visibility_main,
					RENDERPASS_MAIN,
					cmd,
					wi::renderer::DRAWSCENE_IMPOSTOR
				);
			}
			else
			{
				auto range = wi::profiler::BeginRangeGPU("Opaque Scene", cmd);

				// Foreground:
				vp.min_depth = 1 - foreground_depth_range;
				vp.max_depth = 1;
				device->BindViewports(1, &vp, cmd);
				wi::renderer::DrawScene(
					visibility_main,
					RENDERPASS_MAIN,
					cmd,
					wi::renderer::DRAWSCENE_OPAQUE |
					wi::renderer::DRAWSCENE_FOREGROUND_ONLY |
					wi::renderer::DRAWSCENE_MAINCAMERA
				);

				// Regular:
				vp.min_depth = 0;
				vp.max_depth = 1;
				device->BindViewports(1, &vp, cmd);
				wi::renderer::DrawScene(
					visibility_main,
					RENDERPASS_MAIN,
					cmd,
					drawscene_flags
				);
				// Custom scene draw (terrain/trees/grass main opaque):
				// GGMAX 2.13: state-safe hook boundary (see the transparent hook / game task #120)
				if (customDraw_Opaque)
				{
					customDraw_Opaque(&camera->frustum, 0, cmd);
					wi::graphics::GetDevice()->GG_InvalidateCommandListState(cmd);
					wi::renderer::BindCameraCB(*camera, camera_previous, camera_reflection, cmd);
					wi::renderer::BindCommonResources(cmd);
				}
				wi::renderer::DrawSky(*scene, cmd);
				wi::profiler::EndRange(range); // Opaque Scene
			}

			// Blend Aerial Perspective on top:
			if (scene->weather.IsRealisticSky() && scene->weather.IsRealisticSkyAerialPerspective())
			{
				device->EventBegin("Aerial Perspective Blend", cmd);
				wi::image::Params fx;
				fx.enableFullScreen();
				fx.blendFlag = BLENDMODE_PREMULTIPLIED;
				wi::image::Draw(&aerialperspectiveResources.texture_output, fx, cmd);
				device->EventEnd(cmd);
			}

			// Blend the volumetric clouds on top:
			if (scene->weather.IsVolumetricClouds())
			{
				wi::renderer::Postprocess_VolumetricClouds_Upsample(volumetriccloudResources, cmd);
			}

			RenderOutline(cmd);

			device->RenderPassEnd(cmd);

			if (getMeshBlendEnabled() && visibility_main.IsMeshBlendVisible())
			{
				rp[0].loadop = RenderPassImage::LoadOp::LOAD;
				wi::renderer::PostProcess_MeshBlend_Resolve(meshblendResources, rtMain, rp, rp_count, cmd);
			}

			if (wi::renderer::GetRaytracedShadowsEnabled() || wi::renderer::GetScreenSpaceShadowsEnabled())
			{
				GPUBarrier barrier = GPUBarrier::Image(&rtShadow, ResourceState::SHADER_RESOURCE, rtShadow.desc.layout);
				device->Barrier(&barrier, 1, cmd);
			}

			if (rtAO.IsValid())
			{
				device->Barrier(GPUBarrier::Aliasing(&rtAO, &rtParticleDistortion), cmd);
			}

			device->EventEnd(cmd);
			wi::profiler::EndRange(gg_range_cpu); // GGMAX 1.32
		});

		if (scene->terrains.GetCount() > 0)
		{
			CommandList cmd_allocation_tilerequest = device->BeginCommandList(gg_lean(QUEUE_COMPUTE)); // GGMAX 1.48c
			device->WaitCommandList(cmd_allocation_tilerequest, cmd); // wait for opaque scene
			wi::jobsystem::Execute(ctx, [this, cmd_allocation_tilerequest](wi::jobsystem::JobArgs args) {
				for (size_t i = 0; i < scene->terrains.GetCount(); ++i)
				{
					scene->terrains[i].AllocateVirtualTextureTileRequestsGPU(cmd_allocation_tilerequest);
				}
			});

			CommandList cmd_writeback_tilerequest = device->BeginCommandList(gg_lean(QUEUE_COPY)); // GGMAX 1.48c
			device->WaitCommandList(cmd_writeback_tilerequest, cmd_allocation_tilerequest);
			wi::jobsystem::Execute(ctx, [this, cmd_writeback_tilerequest](wi::jobsystem::JobArgs args) {
				for (size_t i = 0; i < scene->terrains.GetCount(); ++i)
				{
					scene->terrains[i].WritebackTileRequestsGPU(cmd_writeback_tilerequest);
				}
			});
		}

		// GGMAX 1.40 (b): camera components hoisted BEFORE the merged transparents+postFX list —
		// their render-to-texture output is consumed by scene materials next frame either way,
		// and hoisting keeps queue submission order intact when the two lists below become one.
		if (gg_render_merge_lists)
		{
			RenderCameraComponents(ctx);
		}

		// Transparents, post processes, etc:
		cmd = device->BeginCommandList();
		if (cmd_ocean.IsValid())
		{
			device->WaitCommandList(cmd, cmd_ocean);
		}
		const bool gg_merge_postfx = gg_render_merge_lists; // GGMAX 1.40
		wi::jobsystem::Execute(ctx, [this, cmd, gg_merge_postfx](wi::jobsystem::JobArgs args) {

			GraphicsDevice* device = wi::graphics::GetDevice();
			auto gg_range = wi::profiler::BeginRangeCPU("RP3D-rec Transparent"); // GGMAX 1.32

			wi::renderer::BindCameraCB(
				*camera,
				camera_previous,
				camera_reflection,
				cmd
			);
			wi::renderer::BindCommonResources(cmd);

			RenderLightShafts(cmd);

			RenderVolumetrics(cmd);

			RenderTransparents(cmd);

			// Depth buffers expect a non-pixel shader resource state as they are generated on compute queue:
			{
				// GGMAX: debugUAV barrier only when the lazy texture exists
				GPUBarrier barriers[3];
				uint32_t num_barriers = 0;
				barriers[num_barriers++] = GPUBarrier::Image(&rtLinearDepth, ResourceState::SHADER_RESOURCE, rtLinearDepth.desc.layout);
				barriers[num_barriers++] = GPUBarrier::Image(&depthBuffer_Copy, ResourceState::SHADER_RESOURCE, depthBuffer_Copy.desc.layout);
				if (debugUAV.IsValid())
				{
					barriers[num_barriers++] = GPUBarrier::Image(&debugUAV, ResourceState::UNORDERED_ACCESS, debugUAV.desc.layout);
				}
				device->Barrier(barriers, num_barriers, cmd);
			}
			wi::profiler::EndRange(gg_range); // GGMAX 1.32

			// GGMAX 1.40 (b): postprocess chain rides the same list (one less list + submit)
			if (gg_merge_postfx)
			{
				auto gg_range2 = wi::profiler::BeginRangeCPU("RP3D-rec PostFX"); // GGMAX 1.32
				RenderPostprocessChain(cmd);
				wi::renderer::TextureStreamingReadbackCopy(*scene, cmd);
				wi::profiler::EndRange(gg_range2); // GGMAX 1.32
			}
		});

		if (!gg_render_merge_lists)
		{
			RenderCameraComponents(ctx);

			cmd = device->BeginCommandList();
			wi::jobsystem::Execute(ctx, [this, cmd](wi::jobsystem::JobArgs args) {
				auto gg_range = wi::profiler::BeginRangeCPU("RP3D-rec PostFX"); // GGMAX 1.32
				RenderPostprocessChain(cmd);
				wi::renderer::TextureStreamingReadbackCopy(*scene, cmd);
				wi::profiler::EndRange(gg_range); // GGMAX 1.32
			});
		}

		RenderPath2D::Render();

		wi::profiler::EndRange(gg_range_serial); // GGMAX 1.32
		auto gg_range_wait = wi::profiler::BeginRangeCPU("RP3D-RenderWait");
		wi::jobsystem::Wait(ctx);
		wi::profiler::EndRange(gg_range_wait); // GGMAX 1.32

		first_frame = false;
	}

	void RenderPath3D::Compose(CommandList cmd) const
	{
		GraphicsDevice* device = wi::graphics::GetDevice();
		device->EventBegin("RenderPath3D::Compose", cmd);

		wi::image::Params fx;
		fx.blendFlag = BLENDMODE_OPAQUE;
		fx.quality = wi::image::QUALITY_LINEAR;
		fx.enableFullScreen();

		wi::image::Draw(GetLastPostprocessRT(), fx, cmd);

		if (
			(wi::renderer::GetDebugLightCulling() ||
			wi::renderer::GetVariableRateShadingClassificationDebug() ||
			wi::renderer::GetSurfelGIDebugEnabled())
			&& debugUAV.IsValid() // GGMAX: lazy — first frame after a debug toggle may not have it yet
			)
		{
			fx.enableFullScreen();
			fx.blendFlag = BLENDMODE_PREMULTIPLIED;
			wi::image::Draw(&debugUAV, fx, cmd);
		}

		// GGMAX 2.13: state-safe hook boundary (see the transparent hook / game task #120)
		if (customDraw_Compose)
		{
			customDraw_Compose(cmd);
			wi::graphics::GetDevice()->GG_InvalidateCommandListState(cmd);
		}

		device->EventEnd(cmd);

		RenderPath2D::Compose(cmd);
	}

	void RenderPath3D::Stop()
	{
		DeleteGPUResources();
	}

	void RenderPath3D::Start()
	{
		ResizeBuffers();
	}

	void RenderPath3D::RenderAO(CommandList cmd) const
	{
		if (rtAO.IsValid())
		{
			GetDevice()->Barrier(GPUBarrier::Aliasing(&rtParticleDistortion, &rtAO), cmd);
		}

		if (getAOEnabled())
		{
			switch (getAO())
			{
			case AO_SSAO:
				wi::renderer::Postprocess_SSAO(
					ssaoResources,
					rtLinearDepth,
					rtAO,
					cmd,
					getAORange(),
					getAOSampleCount(),
					getAOPower()
				);
				break;
			case AO_HBAO:
				wi::renderer::Postprocess_HBAO(
					ssaoResources,
					*camera,
					rtLinearDepth,
					rtAO,
					cmd,
					getAOPower()
				);
				break;
			case AO_MSAO:
				wi::renderer::Postprocess_MSAO(
					msaoResources,
					*camera,
					rtLinearDepth,
					rtAO,
					cmd,
					getAOPower()
				);
				break;
			case AO_RTAO:
				wi::renderer::Postprocess_RTAO(
					rtaoResources,
					*scene,
					rtLinearDepth,
					rtAO,
					cmd,
					getAORange(),
					getAOPower()
				);
				break;
			case AO_DISABLED:
				break;
			}
		}
	}
	void RenderPath3D::RenderSSR(CommandList cmd) const
	{
		if (getSSREnabled() && !getRaytracedReflectionEnabled())
		{
			wi::renderer::Postprocess_SSR(
				ssrResources,
				rtSceneCopy,
				rtSSR,
				cmd,
				getReflectionRoughnessCutoff()
			);
		}
	}
	void RenderPath3D::RenderSSGI(CommandList cmd) const
	{
		if (getSSGIEnabled())
		{
			wi::renderer::Postprocess_SSGI(
				ssgiResources,
				rtSceneCopy,
				depthBuffer_Copy,
				visibilityResources.texture_normals,
				rtSSGI,
				cmd,
				getSSGIDepthRejection()
			);
		}
	}
	void RenderPath3D::RenderOutline(CommandList cmd) const
	{
		if (getOutlineEnabled())
		{
			wi::renderer::Postprocess_Outline(
				rtOutlineSource,
				cmd,
				getOutlineThreshold(),
				getOutlineThickness(),
				getOutlineColor()
			);
		}
	}
	void RenderPath3D::RenderLightShafts(CommandList cmd) const
	{
		const XMVECTOR sunDirection = XMLoadFloat3(&scene->weather.sunDirection);
		const float sunDotCamera = XMVectorGetX(XMVector3Dot(sunDirection, camera->GetAt()));

		if (getLightShaftsEnabled() && sunDotCamera > 0)
		{
			constexpr float fadeThreshold = 0.25f;

			// Calculate target fade factor based on sun-camera angle
			float targetFadeFactor = 0.0f;
			if (sunDotCamera > 0.25)
			{
				targetFadeFactor = 1.0f;
			}

			float fadeSpeed = getLightShaftsFadeSpeed();
			if (targetFadeFactor < lightShaftsFadeFactor)
			{
				// Adaptive fade-out: accelerate as we approach the cutoff threshold
				const float normalizedDistance = wi::math::saturate(sunDotCamera / fadeThreshold);
				constexpr float fadeOutMultiplier = 13.0f; // Multiplier for fast fade-out

				// When normalizedDistance is 1.0 (at threshold): slow fade (fadeSpeed)
				// When normalizedDistance is 0.0 (cutoff): very fast fade (fadeOutSpeedMax)
				fadeSpeed = wi::math::Lerp(fadeSpeed * fadeOutMultiplier, fadeSpeed, normalizedDistance);
			}

			lightShaftsFadeFactor = wi::math::Lerp(lightShaftsFadeFactor, targetFadeFactor, 1.0f - exp(-fadeSpeed * scene->dt));

			GraphicsDevice* device = wi::graphics::GetDevice();

			device->EventBegin("Light Shafts", cmd);

			const Texture* texture_fullres = nullptr;

			// Render sun stencil cutout:
			{
				if (getMSAASampleCount() > 1)
				{
					RenderPassImage rp[] = {
						RenderPassImage::RenderTarget(&rtSun[0], RenderPassImage::LoadOp::CLEAR, RenderPassImage::StoreOp::DONTCARE),
						RenderPassImage::Resolve(&rtSun_resolved),
						RenderPassImage::DepthStencil(
							&depthBuffer_Main,
							RenderPassImage::LoadOp::LOAD,
							RenderPassImage::StoreOp::STORE,
							ResourceState::DEPTHSTENCIL,
							ResourceState::DEPTHSTENCIL,
							ResourceState::DEPTHSTENCIL
						),
					};
					device->RenderPassBegin(rp, arraysize(rp), cmd);
					texture_fullres = &rtSun_resolved;
				}
				else
				{
					RenderPassImage rp[] = {
						RenderPassImage::DepthStencil(
							&depthBuffer_Main,
							RenderPassImage::LoadOp::LOAD,
							RenderPassImage::StoreOp::STORE,
							ResourceState::DEPTHSTENCIL,
							ResourceState::DEPTHSTENCIL,
							ResourceState::DEPTHSTENCIL
						),
						RenderPassImage::RenderTarget(&rtSun[0], RenderPassImage::LoadOp::CLEAR),
					};
					device->RenderPassBegin(rp, arraysize(rp), cmd);
					texture_fullres = &rtSun[0];
				}

				Viewport vp;
				vp.width = (float)depthBuffer_Main.GetDesc().width;
				vp.height = (float)depthBuffer_Main.GetDesc().height;
				device->BindViewports(1, &vp, cmd);

				Rect scissor = GetScissorInternalResolution();
				device->BindScissorRects(1, &scissor, cmd);

				wi::renderer::DrawSun(cmd);

				if (scene->weather.IsVolumetricClouds())
				{
					device->EventBegin("Volumetric cloud occlusion mask", cmd);
					wi::image::Params fx;
					fx.enableFullScreen();
					fx.blendFlag = BLENDMODE_MULTIPLY;
					wi::image::Draw(&volumetriccloudResources.texture_cloudMask, fx, cmd);
					device->EventEnd(cmd);
				}

				device->RenderPassEnd(cmd);
			}

			// Radial blur on the sun:
			{
				XMVECTOR sunPos = XMVector3Project(camera->GetEye() + sunDirection * camera->zFarP, 0, 0,
					1.0f, 1.0f, 0.1f, 1.0f,
					camera->GetProjection(), camera->GetView(), XMMatrixIdentity());
				{
					// Downsample to low res first:
					wi::renderer::Postprocess_Downsample4x(*texture_fullres, rtSun[2], cmd);

					XMFLOAT2 sun;
					XMStoreFloat2(&sun, sunPos);
					wi::renderer::Postprocess_LightShafts(
						rtSun[2],
						rtSun[1],
						cmd,
						sun,
						getLightShaftsStrength()
					);
				}
			}
			device->EventEnd(cmd);
		}
	}
	void RenderPath3D::RenderVolumetrics(CommandList cmd) const
	{
		if (getVolumeLightsEnabled() && visibility_main.IsRequestedVolumetricLights())
		{
			auto range = wi::profiler::BeginRangeGPU("Volumetric Lights", cmd);

			GraphicsDevice* device = wi::graphics::GetDevice();

			RenderPassImage rp[] = {
				RenderPassImage::RenderTarget(&rtVolumetricLights, RenderPassImage::LoadOp::CLEAR),
			};
			device->RenderPassBegin(rp, arraysize(rp), cmd);

			Viewport vp;
			vp.width = (float)rtVolumetricLights.GetDesc().width;
			vp.height = (float)rtVolumetricLights.GetDesc().height;
			device->BindViewports(1, &vp, cmd);

			wi::renderer::DrawVolumeLights(visibility_main, cmd);

			device->RenderPassEnd(cmd);

			wi::profiler::EndRange(range);
		}
	}
	void RenderPath3D::RenderSceneMIPChain(CommandList cmd) const
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		auto range = wi::profiler::BeginRangeGPU("Scene MIP Chain", cmd);
		device->EventBegin("RenderSceneMIPChain", cmd);

		{
			GPUBarrier barriers[] = {
				GPUBarrier::Aliasing(&rtPrimitiveID, &rtSceneCopy_tmp),
				GPUBarrier::Image(&rtSceneCopy_tmp, rtSceneCopy_tmp.desc.layout, ResourceState::UNORDERED_ACCESS),
			};
			device->Barrier(barriers, arraysize(barriers), cmd);
			device->ClearUAV(&rtSceneCopy_tmp, 0, cmd);
		}

		wi::renderer::Postprocess_Downsample4x(rtMain, rtSceneCopy, cmd);

		device->Barrier(GPUBarrier::Image(&rtSceneCopy_tmp, ResourceState::UNORDERED_ACCESS, rtSceneCopy_tmp.desc.layout), cmd);

		wi::renderer::MIPGEN_OPTIONS mipopt;
		mipopt.gaussian_temp = &rtSceneCopy_tmp;
		wi::renderer::GenerateMipChain(rtSceneCopy, wi::renderer::MIPGENFILTER_GAUSSIAN, cmd, mipopt);

		device->Barrier(GPUBarrier::Aliasing(&rtSceneCopy_tmp, &rtPrimitiveID), cmd);

		device->EventEnd(cmd);
		wi::profiler::EndRange(range);
	}
	void RenderPath3D::RenderTransparents(CommandList cmd) const
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		// Water ripple rendering:
		if (!scene->waterRipples.empty())
		{
			device->Barrier(GPUBarrier::Aliasing(&rtParticleDistortion, &rtWaterRipple), cmd);
			RenderPassImage rp[] = {
				RenderPassImage::RenderTarget(&rtWaterRipple, RenderPassImage::LoadOp::CLEAR),
			};
			device->RenderPassBegin(rp, arraysize(rp), cmd);

			Viewport vp;
			vp.width = (float)rtWaterRipple.GetDesc().width;
			vp.height = (float)rtWaterRipple.GetDesc().height;
			device->BindViewports(1, &vp, cmd);

			wi::renderer::DrawWaterRipples(visibility_main, cmd);
			device->RenderPassEnd(cmd);
		}

		if (getFSR2Enabled())
		{
			// Save the pre-alpha for FSR2 reactive mask:
			//	Note that rtFSR temp resource is always larger or equal to rtMain, so CopyTexture is used instead of CopyResource!
			GPUBarrier barriers[] = {
				GPUBarrier::Image(&rtMain, rtMain.desc.layout, ResourceState::COPY_SRC),
				GPUBarrier::Image(&rtFSR[1], rtFSR->desc.layout, ResourceState::COPY_DST),
			};
			device->Barrier(barriers, arraysize(barriers), cmd);
			device->CopyTexture(
				&rtFSR[1], 0, 0, 0, 0, 0,
				&rtMain, 0, 0,
				cmd
			);
			for (int i = 0; i < arraysize(barriers); ++i)
			{
				std::swap(barriers[i].image.layout_before, barriers[i].image.layout_after);
			}
			device->Barrier(barriers, arraysize(barriers), cmd);
		}

		Rect scissor = GetScissorInternalResolution();
		device->BindScissorRects(1, &scissor, cmd);

		Viewport vp;
		vp.width = (float)depthBuffer_Main.GetDesc().width;
		vp.height = (float)depthBuffer_Main.GetDesc().height;
		vp.min_depth = 0;
		vp.max_depth = 1;
		device->BindViewports(1, &vp, cmd);

		RenderPassImage rp[3];
		uint32_t rp_count = 0;
		if (getMSAASampleCount() > 1)
		{
			rp[rp_count++] = RenderPassImage::RenderTarget(&rtMain_render, RenderPassImage::LoadOp::LOAD);
			rp[rp_count++] = RenderPassImage::Resolve(&rtMain);
			rp[rp_count++] = RenderPassImage::DepthStencil(
				&depthBuffer_Main,
				RenderPassImage::LoadOp::LOAD,
				RenderPassImage::StoreOp::STORE,
				ResourceState::DEPTHSTENCIL,
				ResourceState::DEPTHSTENCIL,
				ResourceState::DEPTHSTENCIL
			);
		}
		else
		{

			rp[rp_count++] = RenderPassImage::RenderTarget(&rtMain_render, RenderPassImage::LoadOp::LOAD);
			rp[rp_count++] = RenderPassImage::DepthStencil(
				&depthBuffer_Main,
				RenderPassImage::LoadOp::LOAD,
				RenderPassImage::StoreOp::STORE,
				ResourceState::DEPTHSTENCIL,
				ResourceState::DEPTHSTENCIL,
				ResourceState::DEPTHSTENCIL
			);
		}

		// Draw only the ocean first, fog and lightshafts will be blended on top:
		if (scene->weather.IsOceanEnabled() && scene->ocean.IsValid() && (!scene->ocean.IsOccluded() || !wi::renderer::GetOcclusionCullingEnabled()))
		{
			device->EventBegin("Copy scene tex only mip0 for ocean", cmd);
			wi::renderer::Postprocess_Downsample4x(rtMain, rtSceneCopy, cmd);
			device->EventEnd(cmd);

			device->RenderPassBegin(rp, rp_count, cmd);

			wi::renderer::DrawScene(
				visibility_main,
				RENDERPASS_MAIN,
				cmd,
				wi::renderer::DRAWSCENE_OCEAN
			);

			device->RenderPassEnd(cmd);
		}

		if (visibility_main.IsTransparentsVisible())
		{
			RenderSceneMIPChain(cmd);
		}

		device->RenderPassBegin(rp, rp_count, cmd);

		// Note: volumetrics and light shafts are blended before transparent scene, because they used depth of the opaques
		//	But the ocean is special, because it does have depth for them implicitly computed from ocean plane

		if (getVolumeLightsEnabled() && visibility_main.IsRequestedVolumetricLights())
		{
			device->EventBegin("Contribute Volumetric Lights", cmd);
			wi::renderer::Postprocess_Upsample_Bilateral(
				rtVolumetricLights,
				rtLinearDepth,
				rtMain,
				cmd,
				true,
				1.5f
			);
			device->EventEnd(cmd);
		}

		XMVECTOR sunDirection = XMLoadFloat3(&scene->weather.sunDirection);
		if (getLightShaftsEnabled() && XMVectorGetX(XMVector3Dot(sunDirection, camera->GetAt())) > 0)
		{
			device->EventBegin("Contribute LightShafts", cmd);
			wi::image::Params fx;
			fx.enableFullScreen();
			fx.blendFlag = BLENDMODE_ADDITIVE;
			fx.opacity = lightShaftsFadeFactor;
			wi::image::Draw(&rtSun[1], fx, cmd);
			device->EventEnd(cmd);
		}

		// Transparent scene
		if (visibility_main.IsTransparentsVisible())
		{
			auto range = wi::profiler::BeginRangeGPU("Transparent Scene", cmd);
			device->EventBegin("Transparent Scene", cmd);

			// Regular:
			vp.min_depth = 0;
			vp.max_depth = 1;
			device->BindViewports(1, &vp, cmd);
			wi::renderer::DrawScene(
				visibility_main,
				RENDERPASS_MAIN,
				cmd,
				wi::renderer::DRAWSCENE_TRANSPARENT |
				wi::renderer::DRAWSCENE_TESSELLATION |
				wi::renderer::DRAWSCENE_OCCLUSIONCULLING |
				wi::renderer::DRAWSCENE_MAINCAMERA
			);

			// Foreground:
			vp.min_depth = 1 - foreground_depth_range;
			vp.max_depth = 1;
			device->BindViewports(1, &vp, cmd);
			wi::renderer::DrawScene(
				visibility_main,
				RENDERPASS_MAIN,
				cmd,
				wi::renderer::DRAWSCENE_TRANSPARENT |
				wi::renderer::DRAWSCENE_FOREGROUND_ONLY |
				wi::renderer::DRAWSCENE_MAINCAMERA
			);

			// Reset normal viewport:
			vp.min_depth = 0;
			vp.max_depth = 1;
			device->BindViewports(1, &vp, cmd);

			device->EventEnd(cmd);
			wi::profiler::EndRange(range); // Transparent Scene
		}

		// GGMAX 2.91: everything from here to RenderPassEnd used to sit in NO profiler range —
		// the "Transparent Scene" range closes above, but the pass keeps going. That orphaned
		// window holds customDraw_Transparent (the GG gpup/.arx particle draw), DrawDebugWorld,
		// DrawWireframeOverlay, DrawLightVisualizers, DrawSpritesAndFonts and DrawLensFlares —
		// only DrawSoftParticles inside it was ever measured. Wrap the remainder so the panel
		// accounts for it instead of dumping it into the GPU Frame gap.
		// ⚠ Explicit Begin/End, NOT ScopedGPUProfiling: the enclosing scope here runs well past
		// RenderPassEnd, so a scoped object would also swallow the distortion pass and the
		// postprocess chain and report a meaninglessly large row.
		auto range_ttail = wi::profiler::BeginRangeGPU("Transparent Tail (custom/debug/sprites/flares)", cmd);

		// Custom scene draw (terrain transparent overlays):
		// GGMAX 2.13 (game task #120 — the steam-scene fullscreen white-out): the hook
		// records GG-shader draws (legacy gpup steam, GG terrain) whose constant buffers
		// bind on slots b0/b1 of the shared binder — b1 is CBSLOT_RENDERER_CAMERA. The
		// draws below that read GetCamera() without rebinding it (LENS FLARES: canvas_size_rcp
		// scales the quad) then execute against gpup's particle constants misread as the
		// camera — the sun flare rasterized FULLSCREEN sampling the bright flare-texture
		// center: a uniform ~88%-opacity white veil ("whole screen washed out"). Restore the
		// pass's CB contract + invalidate the state trackers after the hook returns.
		if (customDraw_Transparent)
		{
			customDraw_Transparent(&camera->frustum, cmd);
			device->GG_InvalidateCommandListState(cmd);
			wi::renderer::BindCameraCB(*camera, camera_previous, camera_reflection, cmd);
			wi::renderer::BindCommonResources(cmd);
		}

		wi::renderer::DrawDebugWorld(*scene, *camera, *this, cmd);

		wi::renderer::DrawWireframeOverlay(visibility_main, wi::enums::RENDERPASS_MAIN, cmd);

		wi::renderer::DrawLightVisualizers(visibility_main, cmd);

		wi::renderer::DrawSoftParticles(visibility_main, false, cmd);
		wi::renderer::DrawSpritesAndFonts(*scene, *camera, false, cmd);

		if (getLensFlareEnabled())
		{
			wi::renderer::DrawLensFlares(
				visibility_main,
				cmd,
				scene->weather.IsVolumetricClouds() ? &volumetriccloudResources.texture_cloudMask : nullptr
			);
		}

		device->RenderPassEnd(cmd);
		wi::profiler::EndRange(range_ttail); // GGMAX 2.91: close at the pass boundary

		// Distortion particles:
		{
			if (rtWaterRipple.IsValid())
			{
				device->Barrier(GPUBarrier::Aliasing(&rtWaterRipple, &rtParticleDistortion), cmd);
			}

			if (getMSAASampleCount() > 1)
			{
				RenderPassImage rp[] = {
					RenderPassImage::RenderTarget(&rtParticleDistortion_render, RenderPassImage::LoadOp::CLEAR),
					RenderPassImage::Resolve(&rtParticleDistortion),
					RenderPassImage::DepthStencil(
						&depthBuffer_Main,
						RenderPassImage::LoadOp::LOAD,
						RenderPassImage::StoreOp::STORE,
						ResourceState::DEPTHSTENCIL,
						ResourceState::DEPTHSTENCIL,
						ResourceState::DEPTHSTENCIL
					),
				};
				device->RenderPassBegin(rp, arraysize(rp), cmd);
			}
			else
			{
				RenderPassImage rp[] = {
					RenderPassImage::RenderTarget(&rtParticleDistortion, RenderPassImage::LoadOp::CLEAR),
					RenderPassImage::DepthStencil(
						&depthBuffer_Main,
						RenderPassImage::LoadOp::LOAD,
						RenderPassImage::StoreOp::STORE,
						ResourceState::DEPTHSTENCIL,
						ResourceState::DEPTHSTENCIL,
						ResourceState::DEPTHSTENCIL
					),
				};
				device->RenderPassBegin(rp, arraysize(rp), cmd);
			}

			Viewport vp;
			vp.width = (float)rtParticleDistortion.GetDesc().width;
			vp.height = (float)rtParticleDistortion.GetDesc().height;
			device->BindViewports(1, &vp, cmd);

			wi::renderer::DrawSoftParticles(visibility_main, true, cmd);
			wi::renderer::DrawSpritesAndFonts(*scene, *camera, true, cmd);

			device->RenderPassEnd(cmd);
		}

		wi::renderer::Postprocess_Downsample4x(rtMain, rtSceneCopy, cmd);
	}
	void RenderPath3D::RenderPostprocessChain(CommandList cmd) const
	{
		GraphicsDevice* device = wi::graphics::GetDevice();

		wi::renderer::BindCommonResources(cmd);
		wi::renderer::BindCameraCB(*camera, camera_previous, camera_reflection, cmd);

		const Texture* rt_first = nullptr; // not ping-ponged with read / write
		const Texture* rt_read = &rtMain;
		const Texture* rt_write = &rtPostprocess;

		// rtPostprocess aliasing transition:
		{
			GPUBarrier barriers[] = {
				GPUBarrier::Aliasing(&rtPrimitiveID, &rtPostprocess),
				GPUBarrier::Image(&rtPostprocess, rtPostprocess.desc.layout, ResourceState::UNORDERED_ACCESS),
			};
			device->Barrier(barriers, arraysize(barriers), cmd);
			device->ClearUAV(&rtPostprocess, 0, cmd);
			device->Barrier(GPUBarrier::Image(&rtPostprocess, ResourceState::UNORDERED_ACCESS, rtPostprocess.desc.layout), cmd);
		}

		// 1.) HDR post process chain
		{
			if (getFSR2Enabled() && fsr2Resources.IsValid())
			{
				wi::renderer::Postprocess_FSR2(
					fsr2Resources,
					*camera,
					rtFSR[1],
					*rt_read,
					depthBuffer_Copy,
					rtVelocity,
					rtFSR[0],
					cmd,
					scene->dt,
					getFSR2Sharpness()
				);

				// rebind these, because FSR2 binds other things to those constant buffers:
				wi::renderer::BindCameraCB(
					*camera,
					camera_previous,
					camera_reflection,
					cmd
				);
				wi::renderer::BindCommonResources(cmd);

				rt_read = &rtFSR[0];
				rt_write = &rtFSR[1];
			}
			else if (wi::renderer::GetTemporalAAEnabled() && !wi::renderer::GetTemporalAADebugEnabled() && temporalAAResources.IsValid() && wi::renderer::GetWireframeMode() == wi::renderer::WIREFRAME_DISABLED)
			{
				wi::renderer::Postprocess_TemporalAA(
					temporalAAResources,
					*rt_read,
					cmd
				);
				rt_first = temporalAAResources.GetCurrent();
			}
			if (scene->weather.IsOceanEnabled())
			{
				// GGMAX 1.39: the underwater postprocess only affects a SUBMERGED camera — skip
				// the full-screen dispatch (+ClearUAV +barriers) while the eye is safely above the
				// waterline (margin covers wave displacement). Chain-safe: skipping a stage leaves
				// rt_first/rt_read for the next consumer exactly like the stage never existed.
				const float gg_waterline = scene->weather.oceanParameters.waterHeight + 200.0f;
				if (!gg_skip_underwater_above_water || camera->Eye.y <= gg_waterline)
				{
					wi::renderer::Postprocess_Underwater(
						rt_first == nullptr ? *rt_read : *rt_first,
						*rt_write,
						cmd
					);

					rt_first = nullptr;
					std::swap(rt_read, rt_write);
				}
			}

			for (auto& x : custom_post_processes)
			{
				if (x.stage == CustomPostprocess::Stage::BeforeTonemap)
				{
					wi::renderer::Postprocess_Custom(
						x.computeshader,
						rt_first == nullptr ? *rt_read : *rt_first,
						*rt_write,
						cmd,
						x.params0,
						x.params1,
						x.name.c_str()
					);

					rt_first = nullptr;
					std::swap(rt_read, rt_write);
				}
			}

			if (getDepthOfFieldEnabled() && camera->aperture_size > 0.001f && getDepthOfFieldStrength() > 0.001f && depthoffieldResources.IsValid())
			{
				wi::renderer::Postprocess_DepthOfField(
					depthoffieldResources,
					rt_first == nullptr ? *rt_read : *rt_first,
					*rt_write,
					cmd,
					getDepthOfFieldStrength()
				);

				rt_first = nullptr;
				std::swap(rt_read, rt_write);
			}

			if (getMotionBlurEnabled() && getMotionBlurStrength() > 0 && motionblurResources.IsValid())
			{
				wi::renderer::Postprocess_MotionBlur(
					scene->dt,
					motionblurResources,
					rt_first == nullptr ? *rt_read : *rt_first,
					*rt_write,
					cmd,
					getMotionBlurStrength()
				);

				rt_first = nullptr;
				std::swap(rt_read, rt_write);
			}
		}

		// 2.) Tone mapping HDR -> LDR
		{
			// Bloom and eye adaption is not part of post process "chain",
			//	because they will be applied to the screen in tonemap
			if (getEyeAdaptionEnabled())
			{
				wi::renderer::ComputeLuminance(
					luminanceResources,
					rt_first == nullptr ? *rt_read : *rt_first,
					cmd,
					getEyeAdaptionRate(),
					getEyeAdaptionKey()
				);
			}
			if (getBloomEnabled())
			{
				wi::renderer::ComputeBloom(
					bloomResources,
					rt_first == nullptr ? *rt_read : *rt_first,
					cmd,
					getBloomThreshold(),
					getExposure(),
					getEyeAdaptionEnabled() ? &luminanceResources.luminance : nullptr
				);
			}

			wi::renderer::Postprocess_Tonemap(
				rt_first == nullptr ? *rt_read : *rt_first,
				*rt_write,
				cmd,
				getExposure(),
				getBrightness(),
				getContrast(),
				getSaturation(),
				getDitherEnabled(),
				getColorGradingEnabled() ? (scene->weather.colorGradingMap.IsValid() ? &scene->weather.colorGradingMap.GetTexture() : nullptr) : nullptr,
				&rtParticleDistortion,
				getEyeAdaptionEnabled() ? &luminanceResources.luminance : nullptr,
				getBloomEnabled() ? &bloomResources.texture_bloom : nullptr,
				colorspace,
				getTonemap(),
				&distortion_overlay,
				getHDRCalibration()
			);

			rt_first = nullptr;
			std::swap(rt_read, rt_write);
		}

		// 3.) LDR post process chain
		{
			for (auto& x : custom_post_processes)
			{
				if (x.stage == CustomPostprocess::Stage::AfterTonemap)
				{
					wi::renderer::Postprocess_Custom(
						x.computeshader,
						*rt_read,
						*rt_write,
						cmd,
						x.params0,
						x.params1,
						x.name.c_str()
					);

					std::swap(rt_read, rt_write);
				}
			}

			if (getSharpenFilterEnabled())
			{
				wi::renderer::Postprocess_Sharpen(*rt_read, *rt_write, cmd, getSharpenFilterAmount());

				std::swap(rt_read, rt_write);
			}

			if (getFXAAEnabled())
			{
				wi::renderer::Postprocess_FXAA(*rt_read, *rt_write, cmd);

				std::swap(rt_read, rt_write);
			}

			if (getChromaticAberrationEnabled())
			{
				wi::renderer::Postprocess_Chromatic_Aberration(*rt_read, *rt_write, cmd, getChromaticAberrationAmount());

				std::swap(rt_read, rt_write);
			}

			if (getCRTFilterEnabled())
			{
				wi::renderer::Postprocess_CRT(*rt_read, *rt_write, cmd, 0, 0, true);

				std::swap(rt_read, rt_write);
			}

			lastPostprocessRT = rt_read;

			// GUI Background blurring:
			{
				auto range = wi::profiler::BeginRangeGPU("GUI Background Blur", cmd);
				device->EventBegin("GUI Background Blur", cmd);
				bool hdrToSRGB = colorspace != ColorSpace::SRGB;
				wi::renderer::Postprocess_Downsample4x(*rt_read, rtGUIBlurredBackground[0], cmd, hdrToSRGB);
				wi::renderer::Postprocess_Downsample4x(rtGUIBlurredBackground[0], rtGUIBlurredBackground[2], cmd);
				wi::renderer::Postprocess_Blur_Gaussian(rtGUIBlurredBackground[2], rtGUIBlurredBackground[1], rtGUIBlurredBackground[2], cmd, -1, -1, true);
				device->EventEnd(cmd);
				wi::profiler::EndRange(range);
			}

			if (rtFSR[0].IsValid() && getFSREnabled())
			{
				wi::renderer::Postprocess_FSR(*rt_read, rtFSR[1], rtFSR[0], cmd, getFSRSharpness());
				lastPostprocessRT = &rtFSR[0];
			}
		}
	}

	void RenderPath3D::RenderCameraComponents(wi::jobsystem::context& ctx) const
	{
		// Render-to-texture camera components:
		for (uint32_t i = 0; i < scene->cameras.GetCount() && getSceneUpdateEnabled(); ++i)
		{
			wi::scene::CameraComponent& camera = scene->cameras[i];
			if (camera.render_to_texture.resolution.x == 0 || camera.render_to_texture.resolution.y == 0)
			{
				camera.render_to_texture = {};
				continue;
			}

			GraphicsDevice* device = GetDevice();
			CommandList cmd = device->BeginCommandList();

			if (!camera.render_to_texture.rendertarget_render.IsValid() ||
				camera.render_to_texture.rendertarget_render.desc.width != camera.render_to_texture.resolution.x ||
				camera.render_to_texture.rendertarget_render.desc.height != camera.render_to_texture.resolution.y ||
				camera.render_to_texture.rendertarget_MSAA.desc.sample_count != camera.render_to_texture.sample_count
				)
			{
				TextureDesc desc;
				desc.width = camera.render_to_texture.resolution.x;
				desc.height = camera.render_to_texture.resolution.y;
				desc.format = wi::renderer::format_rendertarget_main;
				desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
				desc.mip_levels = 0;
				bool success = device->CreateTexture(&desc, nullptr, &camera.render_to_texture.rendertarget_render);
				assert(success);
				device->SetName(&camera.render_to_texture.rendertarget_render, "CameraComponent::RenderToTexture::rendertarget_render");
				success = device->CreateTexture(&desc, nullptr, &camera.render_to_texture.rendertarget_display);
				assert(success);
				device->SetName(&camera.render_to_texture.rendertarget_display, "CameraComponent::RenderToTexture::rendertarget_display");

				for (uint32_t i = 0; i < camera.render_to_texture.rendertarget_render.desc.mip_levels; ++i)
				{
					int subresource_index;
					subresource_index = device->CreateSubresource(&camera.render_to_texture.rendertarget_render, SubresourceType::SRV, 0, 1, i, 1);
					assert(subresource_index == i);
					subresource_index = device->CreateSubresource(&camera.render_to_texture.rendertarget_display, SubresourceType::SRV, 0, 1, i, 1);
					assert(subresource_index == i);
					subresource_index = device->CreateSubresource(&camera.render_to_texture.rendertarget_render, SubresourceType::UAV, 0, 1, i, 1);
					assert(subresource_index == i);
					subresource_index = device->CreateSubresource(&camera.render_to_texture.rendertarget_display, SubresourceType::UAV, 0, 1, i, 1);
					assert(subresource_index == i);
				}

				desc.mip_levels = 1;
				if (camera.render_to_texture.sample_count > 1)
				{
					desc.sample_count = camera.render_to_texture.sample_count;
					desc.layout = ResourceState::RENDERTARGET;
					desc.bind_flags = BindFlag::RENDER_TARGET;
					success = device->CreateTexture(&desc, nullptr, &camera.render_to_texture.rendertarget_MSAA);
					assert(success);
					device->SetName(&camera.render_to_texture.rendertarget_MSAA, "CameraComponent::RenderToTexture::rendertarget_MSAA");
				}
				else
				{
					camera.render_to_texture.rendertarget_MSAA = {};
				}

				desc.format = wi::renderer::format_depthbuffer_main;
				desc.bind_flags = BindFlag::DEPTH_STENCIL | BindFlag::SHADER_RESOURCE;
				desc.layout = ResourceState::SHADER_RESOURCE;
				success = device->CreateTexture(&desc, nullptr, &camera.render_to_texture.depthstencil);
				assert(success);
				device->SetName(&camera.render_to_texture.depthstencil, "CameraComponent::RenderToTexture::depthstencil");

				if (camera.render_to_texture.sample_count > 1)
				{
					desc.sample_count = 1;
					desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
					desc.layout = ResourceState::SHADER_RESOURCE;
					desc.format = Format::R32_FLOAT;
					success = device->CreateTexture(&desc, nullptr, &camera.render_to_texture.depthstencil_resolved);
					assert(success);
					device->SetName(&camera.render_to_texture.depthstencil_resolved, "CameraComponent::RenderToTexture::depthstencil_resolved");
				}
				else
				{
					camera.render_to_texture.depthstencil_resolved = {};
				}

				wi::renderer::TiledLightResources tiledres;
				wi::renderer::CreateTiledLightResources(tiledres, camera.render_to_texture.resolution);
				camera.render_to_texture.tileCount = tiledres.tileCount;
				camera.render_to_texture.entityTiles = tiledres.entityTiles;

				camera.render_to_texture.visibility = std::make_shared<wi::renderer::Visibility>();
			}
			if (getSceneUpdateEnabled())
			{
				std::swap(camera.render_to_texture.rendertarget_render, camera.render_to_texture.rendertarget_display);
			}
			camera.width = (float)camera.render_to_texture.resolution.x;
			camera.height = (float)camera.render_to_texture.resolution.y;
			if (camera.render_to_texture.depthstencil_resolved.IsValid())
			{
				camera.texture_depth_index = device->GetDescriptorIndex(&camera.render_to_texture.depthstencil_resolved, SubresourceType::SRV);
			}
			else
			{
				camera.texture_depth_index = device->GetDescriptorIndex(&camera.render_to_texture.depthstencil, SubresourceType::SRV);
			}
			camera.buffer_entitytiles_index = device->GetDescriptorIndex(&camera.render_to_texture.entityTiles, SubresourceType::SRV);


			wi::jobsystem::Execute(ctx, [this, cmd, i](wi::jobsystem::JobArgs args) {
				GraphicsDevice* device = GetDevice();
				wi::scene::CameraComponent& camera = scene->cameras[i]; // reload, not captured in lambda (alloc)
				wi::renderer::Visibility& visibility = *(wi::renderer::Visibility*)camera.render_to_texture.visibility.get();
				visibility.layerMask = getLayerMask();
				visibility.scene = scene;
				visibility.camera = &camera;
				visibility.flags = wi::renderer::Visibility::ALLOW_OBJECTS;
				visibility.flags |= wi::renderer::Visibility::ALLOW_HAIRS;
				visibility.flags |= wi::renderer::Visibility::ALLOW_LIGHTS;
				visibility.flags |= wi::renderer::Visibility::ALLOW_DECALS;
				visibility.flags |= wi::renderer::Visibility::ALLOW_ENVPROBES;
				wi::renderer::UpdateVisibility(visibility);

				ScopedGPUProfiling("Camera Entity", cmd);
				device->EventBegin("Camera Entity", cmd);
				wi::renderer::BindCommonResources(cmd);
				wi::renderer::BindCameraCB(
					camera,
					camera,
					camera,
					cmd
				);
				Rect scissor;
				scissor.right = (int32_t)camera.render_to_texture.depthstencil.desc.width;
				scissor.bottom = (int32_t)camera.render_to_texture.depthstencil.desc.height;
				device->BindScissorRects(1, &scissor, cmd);
				Viewport vp;
				vp.width = (float)camera.render_to_texture.depthstencil.desc.width;
				vp.height = (float)camera.render_to_texture.depthstencil.desc.height;
				device->BindViewports(1, &vp, cmd);
				// prepass:
				{
					RenderPassImage rp[] = {
						RenderPassImage::DepthStencil(&camera.render_to_texture.depthstencil, RenderPassImage::LoadOp::CLEAR, RenderPassImage::StoreOp::STORE, camera.render_to_texture.depthstencil.desc.layout, ResourceState::DEPTHSTENCIL, ResourceState::SHADER_RESOURCE),
					};
					device->RenderPassBegin(rp, arraysize(rp), cmd);
					wi::renderer::DrawScene(
						visibility,
						RENDERPASS_PREPASS_DEPTHONLY,
						cmd,
						wi::renderer::DRAWSCENE_OPAQUE |
						wi::renderer::DRAWSCENE_IMPOSTOR |
						wi::renderer::DRAWSCENE_HAIRPARTICLE
					);
					device->RenderPassEnd(cmd);
				}
				if (camera.render_to_texture.depthstencil_resolved.IsValid())
				{
					wi::renderer::ResolveMSAADepthBuffer(camera.render_to_texture.depthstencil_resolved, camera.render_to_texture.depthstencil, cmd);
				}
				wi::renderer::TiledLightResources tiledres;
				tiledres.tileCount = camera.render_to_texture.tileCount;
				tiledres.entityTiles = camera.render_to_texture.entityTiles;
				wi::renderer::ComputeTiledLightCulling(tiledres, visibility, {}, cmd);
				// color pass:
				{
					RenderPassImage rp[3];
					uint32_t rp_count = 0;
					if (camera.render_to_texture.rendertarget_MSAA.IsValid())
					{
						rp[rp_count++] = RenderPassImage::RenderTarget(&camera.render_to_texture.rendertarget_MSAA, RenderPassImage::LoadOp::CLEAR, RenderPassImage::StoreOp::DONTCARE, ResourceState::RENDERTARGET, ResourceState::RENDERTARGET);
						rp[rp_count++] = RenderPassImage::Resolve(&camera.render_to_texture.rendertarget_render, ResourceState::SHADER_RESOURCE, ResourceState::SHADER_RESOURCE, 0);
					}
					else
					{
						rp[rp_count++] = RenderPassImage::RenderTarget(&camera.render_to_texture.rendertarget_render, RenderPassImage::LoadOp::CLEAR);
					}
					rp[rp_count++] = RenderPassImage::DepthStencil(&camera.render_to_texture.depthstencil, RenderPassImage::LoadOp::LOAD, RenderPassImage::StoreOp::DONTCARE, ResourceState::SHADER_RESOURCE, ResourceState::DEPTHSTENCIL, camera.render_to_texture.depthstencil.desc.layout);
					device->RenderPassBegin(rp, rp_count, cmd);
					wi::renderer::DrawScene(
						visibility,
						RENDERPASS_MAIN,
						cmd,
						wi::renderer::DRAWSCENE_OPAQUE |
						wi::renderer::DRAWSCENE_IMPOSTOR |
						wi::renderer::DRAWSCENE_HAIRPARTICLE
					);
					wi::renderer::DrawScene(
						visibility,
						RENDERPASS_MAIN,
						cmd,
						wi::renderer::DRAWSCENE_TRANSPARENT
					);
					wi::renderer::DrawSky(*scene, cmd);
					wi::renderer::DrawLightVisualizers(visibility, cmd);
					device->RenderPassEnd(cmd);

					if (camera.IsCRT() && getSceneUpdateEnabled())
					{
						wi::renderer::Postprocess_CRT(camera.render_to_texture.rendertarget_render, camera.render_to_texture.rendertarget_display, cmd, 0.2f, frameCB.time * 100, false);
						std::swap(camera.render_to_texture.rendertarget_render, camera.render_to_texture.rendertarget_display);
					}

					wi::renderer::GenerateMipChain(camera.render_to_texture.rendertarget_render, wi::renderer::MIPGENFILTER_LINEAR, cmd);
				}
				device->EventEnd(cmd);
			});
		}
	}

	void RenderPath3D::setAO(AO value)
	{
		ao = value;

		if (!rtParticleDistortion.IsValid())
			return; // ResizeBuffers hasn't been called yet

		rtAO = {};
		ssaoResources = {};
		msaoResources = {};
		rtaoResources = {};

		if (ao == AO_DISABLED)
		{
			return;
		}

		XMUINT2 internalResolution = GetInternalResolution();
		if (internalResolution.x == 0 || internalResolution.y == 0)
			return;

		TextureDesc desc;
		desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS | BindFlag::RENDER_TARGET; // render target binding for aliasing (in case resource heap tier < 2)
		desc.format = Format::R8_UNORM;
		desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;

		switch (ao)
		{
		case RenderPath3D::AO_SSAO:
		case RenderPath3D::AO_HBAO:
			desc.width = internalResolution.x / 2;
			desc.height = internalResolution.y / 2;
			break;
		case RenderPath3D::AO_MSAO:
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			break;
		case RenderPath3D::AO_RTAO:
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			break;
		default:
			break;
		}

		if (ComputeTextureMemorySizeInBytes(desc) > ComputeTextureMemorySizeInBytes(rtParticleDistortion.desc))
		{
			// There would be resource aliasing error if we proceed like this!
			//	looks like ResizeBuffers() hasn't been called yet for the current internal resolution
			//	if this happens, then ResizeBuffers() will be called next frame probably and then AO resources
			//	will be created successdully
			return;
		}

		switch (ao)
		{
		case RenderPath3D::AO_SSAO:
		case RenderPath3D::AO_HBAO:
			wi::renderer::CreateSSAOResources(ssaoResources, internalResolution);
			break;
		case RenderPath3D::AO_MSAO:
			wi::renderer::CreateMSAOResources(msaoResources, internalResolution);
			break;
		case RenderPath3D::AO_RTAO:
			wi::renderer::CreateRTAOResources(rtaoResources, internalResolution);
			break;
		default:
			break;
		}

		GraphicsDevice* device = wi::graphics::GetDevice();
		assert(ComputeTextureMemorySizeInBytes(desc) <= ComputeTextureMemorySizeInBytes(rtParticleDistortion.desc)); // aliasing check
		device->CreateTexture(&desc, nullptr, &rtAO, &rtParticleDistortion); // aliasing!
		device->SetName(&rtAO, "rtAO");
	}
	void RenderPath3D::setSSREnabled(bool value)
	{
		ssrEnabled = value;

		if (value)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			XMUINT2 internalResolution = GetInternalResolution();
			if (internalResolution.x == 0 || internalResolution.y == 0)
				return;

			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = Format::R16G16B16A16_FLOAT;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
			device->CreateTexture(&desc, nullptr, &rtSSR);
			device->SetName(&rtSSR, "rtSSR");

			wi::renderer::CreateSSRResources(ssrResources, internalResolution);
		}
		else
		{
			ssrResources = {};
		}
	}
	void RenderPath3D::setSSGIEnabled(bool value)
	{
		ssgiEnabled = value;

		if (value)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			XMUINT2 internalResolution = GetInternalResolution();
			if (internalResolution.x == 0 || internalResolution.y == 0)
				return;

			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = Format::R16G16B16A16_FLOAT;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			desc.layout = ResourceState::SHADER_RESOURCE_COMPUTE;
			device->CreateTexture(&desc, nullptr, &rtSSGI);
			device->SetName(&rtSSGI, "rtSSGI");

			wi::renderer::CreateSSGIResources(ssgiResources, internalResolution);
		}
		else
		{
			ssgiResources = {};
		}
	}
	void RenderPath3D::setRaytracedReflectionsEnabled(bool value)
	{
		raytracedReflectionsEnabled = value;

		if (value)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			XMUINT2 internalResolution = GetInternalResolution();
			if (internalResolution.x == 0 || internalResolution.y == 0)
				return;

			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = Format::R16G16B16A16_FLOAT;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			device->CreateTexture(&desc, nullptr, &rtSSR);
			device->SetName(&rtSSR, "rtSSR");

			wi::renderer::CreateRTReflectionResources(rtreflectionResources, internalResolution);
		}
		else
		{
			rtreflectionResources = {};
		}
	}
	void RenderPath3D::setRaytracedDiffuseEnabled(bool value)
	{
		raytracedDiffuseEnabled = value;

		if (value)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			XMUINT2 internalResolution = GetInternalResolution();
			if (internalResolution.x == 0 || internalResolution.y == 0)
				return;

			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = Format::R16G16B16A16_FLOAT;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			device->CreateTexture(&desc, nullptr, &rtRaytracedDiffuse);
			device->SetName(&rtRaytracedDiffuse, "rtRaytracedDiffuse");

			wi::renderer::CreateRTDiffuseResources(rtdiffuseResources, internalResolution);
		}
		else
		{
			rtRaytracedDiffuse = {};
			rtdiffuseResources = {};
		}
	}
	void RenderPath3D::setFSREnabled(bool value)
	{
		fsrEnabled = value;

		if (resolutionScale < 1.0f && fsrEnabled)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			if (GetPhysicalWidth() == 0 || GetPhysicalHeight() == 0)
				return;

			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = wi::renderer::format_rendertarget_main;
			desc.width = GetPhysicalWidth();
			desc.height = GetPhysicalHeight();
			device->CreateTexture(&desc, nullptr, &rtFSR[0]);
			device->SetName(&rtFSR[0], "rtFSR[0]");
			device->CreateTexture(&desc, nullptr, &rtFSR[1]);
			device->SetName(&rtFSR[1], "rtFSR[1]");
		}
		else
		{
			if (!getFSR2Enabled())
			{
				// These are used both for FSR and FSR2
				rtFSR[0] = {};
				rtFSR[1] = {};
			}
		}
	}
	void RenderPath3D::setFSR2Enabled(bool value)
	{
		fsr2Enabled = value;

		if (fsr2Enabled)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			if (GetPhysicalWidth() == 0 || GetPhysicalHeight() == 0)
				return;

			XMUINT2 displayResolution = XMUINT2(
				std::max(GetPhysicalWidth(), GetInternalResolution().x),
				std::max(GetPhysicalHeight(), GetInternalResolution().y)
			);

			wi::renderer::CreateFSR2Resources(fsr2Resources, GetInternalResolution(), displayResolution);

			TextureDesc desc;
			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.format = wi::renderer::format_rendertarget_main;
			desc.width = displayResolution.x;
			desc.height = displayResolution.y;
			device->CreateTexture(&desc, nullptr, &rtFSR[0]);
			device->SetName(&rtFSR[0], "rtFSR[0]");
			device->CreateTexture(&desc, nullptr, &rtFSR[1]);
			device->SetName(&rtFSR[1], "rtFSR[1]");
		}
		else
		{
			fsr2Resources = {};
			if (!getFSREnabled())
			{
				// These are used both for FSR and FSR2
				rtFSR[0] = {};
				rtFSR[1] = {};
			}
		}

		// Depending on FSR2 is on/off, these either need to run at display or internal resolution:
		motionblurResources = {};
		depthoffieldResources = {};
	}
	void RenderPath3D::setFSR2Preset(FSR2_Preset preset)
	{
		wi::graphics::SamplerDesc desc = wi::renderer::GetSampler(wi::enums::SAMPLER_OBJECTSHADER)->GetDesc();
		switch (preset)
		{
		default:
		case FSR2_Preset::Quality:
			resolutionScale = 1.0f / 1.5f;
			desc.mip_lod_bias = -1.58f;
			break;
		case FSR2_Preset::Balanced:
			resolutionScale = 1.0f / 1.7f;
			desc.mip_lod_bias = -1.76f;
			break;
		case FSR2_Preset::Performance:
			resolutionScale = 1.0f / 2.0f;
			desc.mip_lod_bias = -2.0f;
			break;
		case FSR2_Preset::Ultra_Performance:
			resolutionScale = 1.0f / 3.0f;
			desc.mip_lod_bias = -2.58f;
			break;
		}
		wi::renderer::ModifyObjectSampler(desc);
	}
	void RenderPath3D::setMotionBlurEnabled(bool value)
	{
		motionBlurEnabled = value;
	}
	void RenderPath3D::setDepthOfFieldEnabled(bool value)
	{
		depthOfFieldEnabled = value;
	}
	void RenderPath3D::setEyeAdaptionEnabled(bool value)
	{
		eyeAdaptionEnabled = value;

		if (value)
		{
			wi::renderer::CreateLuminanceResources(luminanceResources, GetInternalResolution());
		}
		else
		{
			luminanceResources = {};
		}
	}
	void RenderPath3D::setReflectionsEnabled(bool value)
	{
		reflectionsEnabled = value;

		if (value)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			XMUINT2 internalResolution = GetInternalResolution();
			if (internalResolution.x == 0 || internalResolution.y == 0)
				return;

			TextureDesc desc;
			desc.sample_count = 4;
			desc.bind_flags = BindFlag::RENDER_TARGET;
			desc.format = wi::renderer::format_rendertarget_main;
			desc.width = internalResolution.x / 4;
			desc.height = internalResolution.y / 4;
			desc.misc_flags = ResourceMiscFlag::TRANSIENT_ATTACHMENT;
			desc.layout = ResourceState::RENDERTARGET;
			device->CreateTexture(&desc, nullptr, &rtReflection);
			device->SetName(&rtReflection, "rtReflection");

			desc.misc_flags = ResourceMiscFlag::NONE;
			desc.bind_flags = BindFlag::DEPTH_STENCIL | BindFlag::SHADER_RESOURCE;
			desc.format = wi::renderer::format_depthbuffer_main;
			desc.layout = ResourceState::SHADER_RESOURCE;
			device->CreateTexture(&desc, nullptr, &depthBuffer_Reflection);
			device->SetName(&depthBuffer_Reflection, "depthBuffer_Reflection");


			desc.sample_count = 1;
			desc.format = wi::renderer::format_rendertarget_main;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;
			device->CreateTexture(&desc, nullptr, &rtReflection_resolved);
			device->SetName(&rtReflection_resolved, "rtReflection_resolved");

			desc.format = Format::R16_UNORM;
			desc.bind_flags = BindFlag::UNORDERED_ACCESS | BindFlag::SHADER_RESOURCE;
			device->CreateTexture(&desc, nullptr, &depthBuffer_Reflection_resolved);
			device->SetName(&depthBuffer_Reflection_resolved, "depthBuffer_Reflection_resolved");

			wi::renderer::CreateTiledLightResources(tiledLightResources_planarReflection, XMUINT2(depthBuffer_Reflection.desc.width, depthBuffer_Reflection.desc.height));
		}
		else
		{
			rtReflection = {};
			depthBuffer_Reflection = {};
			tiledLightResources_planarReflection = {};
		}
	}
	void RenderPath3D::setBloomEnabled(bool value)
	{
		bloomEnabled = value;

		if (value)
		{
			wi::renderer::CreateBloomResources(bloomResources, GetInternalResolution());
		}
		else
		{
			bloomResources = {};
		}
	}
	void RenderPath3D::setVolumeLightsEnabled(bool value)
	{
		volumeLightsEnabled = value;

		if (value)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			XMUINT2 internalResolution = GetInternalResolution();
			if (internalResolution.x == 0 || internalResolution.y == 0)
				return;

			TextureDesc desc;
			desc.format = Format::R16G16B16A16_FLOAT;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.width = internalResolution.x / 2;
			desc.height = internalResolution.y / 2;
			device->CreateTexture(&desc, nullptr, &rtVolumetricLights);
			device->SetName(&rtVolumetricLights, "rtVolumetricLights");
		}
		else
		{
			rtVolumetricLights = {};
		}
	}
	void RenderPath3D::setLightShaftsEnabled(bool value)
	{
		lightShaftsEnabled = value;

		if (value)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			XMUINT2 internalResolution = GetInternalResolution();
			if (internalResolution.x == 0 || internalResolution.y == 0)
				return;

			TextureDesc desc;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;
			desc.format = wi::renderer::format_rendertarget_main;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			desc.sample_count = getMSAASampleCount();
			device->CreateTexture(&desc, nullptr, &rtSun[0]);
			device->SetName(&rtSun[0], "rtSun[0]");

			desc.bind_flags = BindFlag::SHADER_RESOURCE | BindFlag::UNORDERED_ACCESS;
			desc.sample_count = 1;
			desc.width = internalResolution.x / 4;
			desc.height = internalResolution.y / 4;
			device->CreateTexture(&desc, nullptr, &rtSun[1]);
			device->SetName(&rtSun[1], "rtSun[1]");
			device->CreateTexture(&desc, nullptr, &rtSun[2]);
			device->SetName(&rtSun[2], "rtSun[2]");

			if (getMSAASampleCount() > 1)
			{
				desc.width = internalResolution.x;
				desc.height = internalResolution.y;
				desc.sample_count = 1;
				device->CreateTexture(&desc, nullptr, &rtSun_resolved);
				device->SetName(&rtSun_resolved, "rtSun_resolved");
			}
		}
		else
		{
			rtSun[0] = {};
			rtSun[1] = {};
			rtSun[2] = {};
			rtSun_resolved = {};
		}
	}
	// GGMAX 2026-08-08 (game task #120): sun-chain forensics. The steam white-out proved to
	// be the light-shafts fullscreen additive contribution washing the frame; every gpup
	// input was exonerated byte-for-byte, so the poison enters between DrawSun and the
	// radial blur. This dumps per-stage readback stats to name the stage that goes white.
	int RenderPath3D::GG_DumpSunChain(char* out, int outSize) const
	{
		if (out == nullptr || outSize <= 0) return -1;
		int off = 0;
		const XMVECTOR sunDirection = XMLoadFloat3(&scene->weather.sunDirection);
		const float sunDotCamera = XMVectorGetX(XMVector3Dot(sunDirection, camera->GetAt()));
		off += snprintf(out + off, outSize - off, "shafts=%d fade=%.3f strength=%.3f sunDot=%.3f | ",
			getLightShaftsEnabled() ? 1 : 0, lightShaftsFadeFactor, getLightShaftsStrength(), sunDotCamera);
		// The sun flare's push constants, recomputed exactly as DrawLensFlares does
		// (wiRenderer.cpp:7707-7742). xLensFlarePos.z <= 0 makes the VS's occlusion loop run
		// zero iterations -> visibility = 0/0 = NaN opacity; wild xy = off-screen projection.
		{
			for (uint32_t lightIndex : visibility_main.visibleLights)
			{
				const wi::scene::LightComponent& light = scene->lights[lightIndex];
				if (light.lensFlareRimTextures.empty()) continue;
				XMVECTOR POS;
				if (light.GetType() == wi::scene::LightComponent::DIRECTIONAL)
				{
					XMVECTOR D = XMVector3Normalize(-XMVector3Transform(XMVectorSet(0, 1, 0, 1), XMMatrixRotationQuaternion(XMLoadFloat4(&light.rotation))));
					POS = camera->GetEye() + D * -camera->zFarP;
				}
				else
				{
					POS = XMLoadFloat3(&light.position);
				}
				const float facing = XMVectorGetX(XMVector3Dot(XMVectorSubtract(POS, camera->GetEye()), camera->GetAt()));
				XMFLOAT3 fp;
				XMStoreFloat3(&fp, XMVector3Project(POS, 0, 0, 1, 1, 1, 0, camera->GetProjection(), camera->GetView(), XMMatrixIdentity()));
				off += snprintf(out + off, outSize - off, "flare[L%u t%d n%d] facing=%.2f pos=(%.3f,%.3f,%.4f) | ",
					lightIndex, (int)light.GetType(), (int)light.lensFlareRimTextures.size(), facing, fp.x, fp.y, fp.z);
				if (off >= outSize - 8) break;
			}
		}
		const Texture* stages[4] = { &rtSun[0], &rtSun[1], &rtSun[2], &rtSun_resolved };
		const char* names[4] = { "sun0", "sun1", "sun2", "sunR" };
		for (int s = 0; s < 4 && off < outSize - 8; ++s)
		{
			if (!stages[s]->IsValid()) { off += snprintf(out + off, outSize - off, "%s=n/a ", names[s]); continue; }
			wi::vector<uint8_t> data;
			if (!wi::helper::saveTextureToMemory(*stages[s], data) || data.empty())
			{
				off += snprintf(out + off, outSize - off, "%s=FAIL ", names[s]);
				continue;
			}
			uint64_t sum = 0, nz = 0, hi = 0;
			for (size_t i = 0; i < data.size(); i++)
			{
				sum += data[i];
				if (data[i] != 0) nz++;
				if (data[i] >= 200) hi++;
			}
			off += snprintf(out + off, outSize - off, "%s[%ux%u]=mean %.1f nz %.1f%% hi %.1f%% ",
				names[s], stages[s]->desc.width, stages[s]->desc.height,
				(double)sum / (double)data.size(),
				100.0 * nz / (double)data.size(), 100.0 * hi / (double)data.size());
		}
		out[outSize - 1] = 0;
		return 0;
	}

	void RenderPath3D::setOutlineEnabled(bool value)
	{
		outlineEnabled = value;

		if (value)
		{
			GraphicsDevice* device = wi::graphics::GetDevice();
			XMUINT2 internalResolution = GetInternalResolution();
			if (internalResolution.x == 0 || internalResolution.y == 0)
				return;

			TextureDesc desc;
			desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;
			desc.format = Format::R32_FLOAT;
			desc.width = internalResolution.x;
			desc.height = internalResolution.y;
			device->CreateTexture(&desc, nullptr, &rtOutlineSource);
			device->SetName(&rtOutlineSource, "rtOutlineSource");
		}
		else
		{
			rtOutlineSource = {};
		}
	}

	Texture RenderPath3D::CreateScreenshotWithAlphaBackground()
	{
		TextureDesc desc = rtMain_render.GetDesc();
		desc.format = Format::R8G8B8A8_UNORM;
		desc.bind_flags = BindFlag::RENDER_TARGET | BindFlag::SHADER_RESOURCE;
		Texture tex;
		GraphicsDevice* device = GetDevice();
		bool success = device->CreateTexture(&desc, nullptr, &tex);
		assert(success);

		Texture tex_resolved;
		if (desc.sample_count > 1)
		{
			desc.sample_count = 1;
			success = device->CreateTexture(&desc, nullptr, &tex_resolved);
			assert(success);
		}

		CommandList cmd = device->BeginCommandList();
		RenderPassImage rp[] = {
			RenderPassImage::RenderTarget(&tex, RenderPassImage::LoadOp::CLEAR),
			RenderPassImage::DepthStencil(GetDepthStencil()),
			RenderPassImage::Resolve(&tex_resolved),
		};
		device->RenderPassBegin(rp, tex_resolved.IsValid() ? 3 : 2, cmd);
		Viewport vp;
		vp.width = (float)desc.width;
		vp.height = (float)desc.height;
		device->BindViewports(1, &vp, cmd);
		wi::image::Params fx;
		fx.stencilComp = wi::image::STENCILMODE_NOT;
		fx.stencilRef = 0;
		fx.stencilRefMode = wi::image::STENCILREFMODE_ALL;
		fx.enableFullScreen();
		wi::image::Draw(GetLastPostprocessRT(), fx, cmd);
		device->RenderPassEnd(cmd);

		if (tex_resolved.IsValid())
			return tex_resolved;
		return tex;
	}

}
