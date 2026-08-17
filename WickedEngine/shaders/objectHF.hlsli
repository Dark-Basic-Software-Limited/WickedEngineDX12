#ifndef WI_OBJECTSHADER_HF
#define WI_OBJECTSHADER_HF

#ifdef TRANSPARENT
#define TRANSPARENT_SHADOWMAP_SECONDARY_DEPTH_CHECK
#else
#define SHADOW_MASK_ENABLED
#endif // TRANSPARENT

#if !defined(TRANSPARENT) && !defined(PREPASS) && !defined(ENVMAPRENDERING)
#define DISABLE_ALPHATEST
#endif // !defined(TRANSPARENT) && !defined(PREPASS) && !defined(ENVMAPRENDERING)

#ifdef PLANARREFLECTION
#define DISABLE_ENVMAPS
#define DISABLE_VOXELGI
#endif // PLANARREFLECTION

#ifdef WATER
#define DISABLE_ENVMAPS
#define DISABLE_VOXELGI
#endif // WATER

//#define LIGHTMAP_QUALITY_BICUBIC

#ifdef DISABLE_ALPHATEST
#define EARLY_DEPTH_STENCIL
#endif // DISABLE_ALPHATEST

#ifdef EARLY_DEPTH_STENCIL
#define SVT_FEEDBACK
#endif // EARLY_DEPTH_STENCIL

#ifdef TERRAINBLENDED
#define TEXTURE_SLOT_NONUNIFORM
#endif // TERRAINBLENDED

#ifndef OBJECTSHADER_LAYOUT_COMMON
#define DISABLE_SVT
#endif // OBJECTSHADER_LAYOUT_COMMON

#include "globals.hlsli"
#include "brdf.hlsli"
#include "lightingHF.hlsli"
#include "skyAtmosphere.hlsli"
#include "fogHF.hlsli"
#include "ShaderInterop_SurfelGI.h"
#include "ShaderInterop_DDGI.h"
#include "shadingHF.hlsli"

// DEFINITIONS
//////////////////

PUSHCONSTANT(push, ObjectPushConstants);

#define GetMesh() (load_geometry(push.geometryIndex))
#define GetMaterial() (load_material(push.materialIndex))

//#define sampler_objectshader bindless_samplers[descriptor_index(GetMaterial().sampler_descriptor)]
#define sampler_objectshader bindless_samplers[descriptor_index(push.wrapSamplerIndex)] // This one loads it faster from push constant than the above that loads from material struct but it must be fed per draw call
#define sampler_objectshader_clamp bindless_samplers[descriptor_index(push.clampSamplerIndex)]

// Use these to compile this file as shader prototype:
//#define OBJECTSHADER_COMPILE_VS				- compile vertex shader prototype
//#define OBJECTSHADER_COMPILE_PS				- compile pixel shader prototype

// Use these to define the expected layout for the shader:
//#define OBJECTSHADER_LAYOUT_SHADOW			- layout for shadow pass
//#define OBJECTSHADER_LAYOUT_SHADOW_TEX		- layout for shadow pass and alpha test or transparency
//#define OBJECTSHADER_LAYOUT_PREPASS			- layout for prepass
//#define OBJECTSHADER_LAYOUT_PREPASS_TEX		- layout for prepass and alpha test or dithering
//#define OBJECTSHADER_LAYOUT_COMMON			- layout for common passes

// Use these to enable features for the shader:
//	(Some of these are enabled automatically with OBJECTSHADER_LAYOUT defines)
//#define OBJECTSHADER_USE_CLIPPLANE				- shader will be clipped according to camera clip planes
//#define OBJECTSHADER_USE_COLOR					- shader will use colors (material color, vertex color...)
//#define OBJECTSHADER_USE_DITHERING				- shader will use dithered transparency
//#define OBJECTSHADER_USE_UVSETS					- shader will sample textures with uv sets
//#define OBJECTSHADER_USE_NORMAL					- shader will use normals
//#define OBJECTSHADER_USE_TANGENT					- shader will use tangents, normal mapping
//#define OBJECTSHADER_USE_EMISSIVE					- shader will use emissive
//#define OBJECTSHADER_USE_RENDERTARGETARRAYINDEX	- shader will use dynamic render target slice selection
//#define OBJECTSHADER_USE_VIEWPORTARRAYINDEX		- shader will use dynamic viewport selection
//#define OBJECTSHADER_USE_NOCAMERA					- shader will not use camera space transform
//#define OBJECTSHADER_USE_INSTANCEINDEX			- shader will use instance ID
//#define OBJECTSHADER_USE_CAMERAINDEX				- shader will use camera ID
//#define OBJECTSHADER_USE_COMMON					- shader will use atlas, ambient occlusion, wetmap
//#define OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER	- shader will load vertex ID through provoking index buffer reordering


#ifdef OBJECTSHADER_LAYOUT_SHADOW
#define OBJECTSHADER_USE_CAMERAINDEX
#endif // OBJECTSHADER_LAYOUT_SHADOW

#ifdef OBJECTSHADER_LAYOUT_SHADOW_TEX
#define OBJECTSHADER_USE_INSTANCEINDEX
#define OBJECTSHADER_USE_UVSETS
#define OBJECTSHADER_USE_CAMERAINDEX
#endif // OBJECTSHADER_LAYOUT_SHADOW_TEX

#ifdef OBJECTSHADER_LAYOUT_PREPASS
#define PREPASS
#define OBJECTSHADER_USE_CLIPPLANE
#define OBJECTSHADER_USE_INSTANCEINDEX
#ifndef OBJECTSHADER_COMPILE_MS
#define OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER
#endif // OBJECTSHADER_COMPILE_MS
#endif // OBJECTSHADER_LAYOUT_SHADOW

#ifdef OBJECTSHADER_LAYOUT_PREPASS_TEX
#define PREPASS
#define OBJECTSHADER_USE_CLIPPLANE
#define OBJECTSHADER_USE_UVSETS
#define OBJECTSHADER_USE_DITHERING
#define OBJECTSHADER_USE_INSTANCEINDEX
#ifndef OBJECTSHADER_COMPILE_MS
#define OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER
#endif // OBJECTSHADER_COMPILE_MS
#endif // OBJECTSHADER_LAYOUT_SHADOW_TEX

#ifdef OBJECTSHADER_LAYOUT_COMMON
#define OBJECTSHADER_USE_CLIPPLANE
#define OBJECTSHADER_USE_UVSETS
#define OBJECTSHADER_USE_COLOR
#define OBJECTSHADER_USE_NORMAL
#define OBJECTSHADER_USE_TANGENT
#define OBJECTSHADER_USE_EMISSIVE
#define OBJECTSHADER_USE_INSTANCEINDEX
#define OBJECTSHADER_USE_COMMON
#ifndef OBJECTSHADER_COMPILE_MS
#define OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER // even though primitiveID is not exported, color pass required to have same primitive order on Intel GPU, otherwise depth mismatch occurs
#endif // OBJECTSHADER_COMPILE_MS
#endif // OBJECTSHADER_LAYOUT_COMMON

struct VertexInput
{
	uint vertexID : SV_VertexID;
	uint instanceID : SV_InstanceID;
	
#ifdef OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER
	uint GetPrimitiveID()
	{
		// For prepass the meshopt_generateProvokingIndexBuffer is used to emulate SV_PrimitiveID via provoking vertex
		return vertexID;
	}
#endif // OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER

	uint GetVertexID()
	{
#ifdef OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER
		// For prepass the meshopt_generateProvokingIndexBuffer is used to emulate SV_PrimitiveID via provoking vertex
		return bindless_buffers_uint[descriptor_index(GetMesh().ib_reorder)][vertexID];
#else
		return vertexID;
#endif // OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER
	}

	float4 GetPositionWind()
	{
		return bindless_buffers_float4[descriptor_index(GetMesh().vb_pos_wind)][GetVertexID()];
	}

	float4 GetUVSets()
	{
		[branch]
		if (GetMesh().vb_uvs < 0)
			return 0;
		return lerp(GetMesh().uv_range_min.xyxy, GetMesh().uv_range_max.xyxy, bindless_buffers_float4[descriptor_index(GetMesh().vb_uvs)][GetVertexID()]);
	}

	ShaderMeshInstancePointer GetInstancePointer()
	{
		if (push.instances >= 0)
			return bindless_buffers[descriptor_index(push.instances)].Load<ShaderMeshInstancePointer>(push.instance_offset + instanceID * sizeof(ShaderMeshInstancePointer));

		ShaderMeshInstancePointer poi;
		poi.init();
		return poi;
	}

	float2 GetAtlasUV()
	{
		[branch]
		if (GetMesh().vb_atl < 0)
			return 0;
		return bindless_buffers_float2[descriptor_index(GetMesh().vb_atl)][GetVertexID()];
	}

	half4 GetVertexColor()
	{
		[branch]
		if (GetMesh().vb_col < 0)
			return 1;
		return bindless_buffers_half4[descriptor_index(GetMesh().vb_col)][GetVertexID()];
	}
	
	float3 GetNormal()
	{
		[branch]
		if (GetMesh().vb_nor < 0)
			return 0;
		return bindless_buffers_float4[descriptor_index(GetMesh().vb_nor)][GetVertexID()].xyz;
	}

	float4 GetTangent()
	{
		[branch]
		if (GetMesh().vb_tan < 0)
			return 0;
		return bindless_buffers_float4[descriptor_index(GetMesh().vb_tan)][GetVertexID()];
	}

	ShaderMeshInstance GetInstance()
	{
		if (push.instances >= 0)
			return load_instance(GetInstancePointer().GetInstanceIndex());

		ShaderMeshInstance inst;
		inst.init();
		return inst;
	}

	half GetVertexAO()
	{
		[branch]
		if (GetInstance().vb_ao < 0)
			return 1;
		return bindless_buffers_half[NonUniformResourceIndex(descriptor_index(GetInstance().vb_ao))][GetVertexID()];
	}

	half GetWetmap()
	{
		//[branch]
		//if (GetInstance().vb_wetmap < 0)
		//	return 0;
		//return bindless_buffers_half[NonUniformResourceIndex(descriptor_index(GetInstance().vb_wetmap))][GetVertexID()];

		// There is something seriously bad with AMD driver's shader compiler as the above commented version works incorrectly and this works correctly but only for wetmap
		[branch]
		if (GetInstance().vb_wetmap >= 0)
			return bindless_buffers_half[NonUniformResourceIndex(descriptor_index(GetInstance().vb_wetmap))][GetVertexID()];
		return 0;
	}
};


struct VertexSurface
{
	float4 position;
	float4 uvsets;
	float2 atlas;
	half4 color;
	float3 normal;
	float4 tangent;
	half ao;
	half wet;

	inline void create(in ShaderMaterial material, in VertexInput input)
	{
		float4 pos_wind = input.GetPositionWind();
		position = float4(pos_wind.xyz, 1);
		normal = input.GetNormal();
		color = half4(material.GetBaseColor() * input.GetInstance().GetColor());
		color.a *= half(1 - input.GetInstancePointer().GetDither());

		[branch]
		if (material.IsUsingVertexColors())
		{
			color *= input.GetVertexColor();
		}

		[branch]
		if (material.IsUsingVertexAO())
		{
			ao = input.GetVertexAO();
		}
		else
		{
			ao = 1;
		}

		normal = mul(input.GetInstance().transformRaw.GetMatrixAdjoint(), normal);
		normal = any(normal) ? normalize(normal) : 0;

		tangent = input.GetTangent();
		tangent.xyz = mul(input.GetInstance().transformRaw.GetMatrixAdjoint(), tangent.xyz);
		tangent.xyz = any(tangent.xyz) ? normalize(tangent.xyz) : 0;
		
		uvsets = input.GetUVSets();
		uvsets.xy = mad(uvsets.xy, material.texMulAdd.xy, material.texMulAdd.zw);

		atlas = input.GetAtlasUV();

		position = mul(input.GetInstance().transform.GetMatrix(), position);

		wet = input.GetWetmap();

#ifndef DISABLE_WIND
		[branch]
		if (material.IsUsingWind())
		{
			position.xyz += sample_wind(position.xyz, pos_wind.w);
		}
#endif // DISABLE_WIND
	}
};

struct PixelInput
{
	precise float4 pos : SV_Position;

#ifdef OBJECTSHADER_USE_CLIPPLANE
	float clip : SV_ClipDistance0;
#endif // OBJECTSHADER_USE_CLIPPLANE

#if defined(OBJECTSHADER_USE_INSTANCEINDEX) || defined(OBJECTSHADER_USE_DITHERING) || defined(OBJECTSHADER_USE_CAMERAINDEX)
	uint poi : INSTANCEPOINTER;
#endif // OBJECTSHADER_USE_INSTANCEINDEX || OBJECTSHADER_USE_DITHERING || OBJECTSHADER_USE_CAMERAINDEX

#if defined(PREPASS) && !defined(OBJECTSHADER_COMPILE_MS)
	uint primitiveID : PRIMITIVEID;
#endif // defined(PREPASS) && !defined(OBJECTSHADER_COMPILE_MS)

#ifdef OBJECTSHADER_USE_UVSETS
	float4 uvsets : UVSETS;
#endif // OBJECTSHADER_USE_UVSETS

#ifdef OBJECTSHADER_USE_TANGENT
	float4 tan : TANGENT;
#endif // OBJECTSHADER_USE_TANGENT

#ifdef OBJECTSHADER_USE_NORMAL
	float3 nor : NORMAL;
#endif // OBJECTSHADER_USE_NORMAL

#ifdef OBJECTSHADER_USE_COMMON
	half2 ao_wet : COMMON;
	float2 atl : ATLAS;
#endif // OBJECTSHADER_USE_COMMON

#ifdef OBJECTSHADER_USE_COLOR
	half4 color : COLOR;
#endif // OBJECTSHADER_USE_COLOR

#if !defined(OBJECTSHADER_COMPILE_PS) && !defined(OBJECTSHADER_COMPILE_MS)
#ifdef OBJECTSHADER_USE_RENDERTARGETARRAYINDEX
	uint RTIndex : SV_RenderTargetArrayIndex;
#endif // OBJECTSHADER_USE_RENDERTARGETARRAYINDEX
#ifdef OBJECTSHADER_USE_VIEWPORTARRAYINDEX
	uint VPIndex : SV_ViewportArrayIndex;
#endif // OBJECTSHADER_USE_VIEWPORTARRAYINDEX
#endif // !defined(OBJECTSHADER_COMPILE_PS) && !defined(OBJECTSHADER_COMPILE_MS)

#ifdef OBJECTSHADER_USE_INSTANCEINDEX
	inline uint GetInstanceIndex()
	{
		ShaderMeshInstancePointer pointer;
		pointer.data = poi;
		return pointer.GetInstanceIndex();
	}
#endif // OBJECTSHADER_USE_INSTANCEINDEX

#ifdef OBJECTSHADER_USE_DITHERING
	inline half GetDither()
	{
		ShaderMeshInstancePointer pointer;
		pointer.data = poi;
		return pointer.GetDither();
	}
#endif // OBJECTSHADER_USE_DITHERING

#ifdef OBJECTSHADER_USE_CAMERAINDEX
	inline uint GetCameraIndex()
	{
		ShaderMeshInstancePointer pointer;
		pointer.data = poi;
		return pointer.GetCameraIndex();
	}
#endif // OBJECTSHADER_USE_CAMERAINDEX
	
#ifdef OBJECTSHADER_USE_UVSETS
	inline float4 GetUVSets()
	{
		return uvsets;
	}
#endif // OBJECTSHADER_USE_UVSETS

	inline float3 GetPos3D()
	{
#ifdef OBJECTSHADER_USE_CAMERAINDEX
		ShaderCamera camera = GetCameraIndexed(GetCameraIndex());
#else
		ShaderCamera camera = GetCamera();
#endif // OBJECTSHADER_USE_CAMERAINDEX

		return camera.screen_to_world(pos);
	}

	inline float3 GetViewVector()
	{
#ifdef OBJECTSHADER_USE_CAMERAINDEX
		ShaderCamera camera = GetCameraIndexed(GetCameraIndex());
#else
		ShaderCamera camera = GetCamera();
#endif // OBJECTSHADER_USE_CAMERAINDEX

		return camera.screen_to_nearplane(pos) - GetPos3D(); // ortho support, cannot use cameraPos!
	}
};

PixelInput vertex_to_pixel_export(VertexInput input)
{
	VertexSurface surface;
	surface.create(GetMaterial(), input);

	PixelInput Out;
	
	Out.pos = surface.position;

#ifdef OBJECTSHADER_USE_CAMERAINDEX
	ShaderCamera camera = GetCameraIndexed(input.GetInstancePointer().GetCameraIndex());
#else
	ShaderCamera camera = GetCamera();
#endif // OBJECTSHADER_USE_CAMERAINDEX
	
#if defined(PREPASS) && defined(OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER)
	Out.primitiveID = input.GetPrimitiveID();
#endif // defined(PREPASS) && defined(OBJECTSHADER_USE_PROVOKING_INDEX_BUFFER)

#ifndef OBJECTSHADER_USE_NOCAMERA
	Out.pos = mul(camera.view_projection, Out.pos);
#endif // OBJECTSHADER_USE_NOCAMERA

#ifdef OBJECTSHADER_USE_CLIPPLANE
	Out.clip = dot(surface.position, camera.clip_plane);
#endif // OBJECTSHADER_USE_CLIPPLANE

#if defined(OBJECTSHADER_USE_INSTANCEINDEX) || defined(OBJECTSHADER_USE_DITHERING) || defined(OBJECTSHADER_USE_CAMERAINDEX)
	Out.poi = input.GetInstancePointer().data;
#endif // OBJECTSHADER_USE_INSTANCEINDEX || OBJECTSHADER_USE_DITHERING || OBJECTSHADER_USE_CAMERAINDEX

#ifdef OBJECTSHADER_USE_COLOR
	Out.color = surface.color;
#endif // OBJECTSHADER_USE_COLOR

#ifdef OBJECTSHADER_USE_UVSETS
	Out.uvsets = surface.uvsets;
#endif // OBJECTSHADER_USE_UVSETS

#ifdef OBJECTSHADER_USE_NORMAL
	Out.nor = surface.normal;
#endif // OBJECTSHADER_USE_NORMAL

#ifdef OBJECTSHADER_USE_COMMON
	Out.atl = surface.atlas;
	Out.ao_wet = half2(surface.ao, surface.wet);
#endif // OBJECTSHADER_USE_COMMON

#ifdef OBJECTSHADER_USE_TANGENT
	Out.tan = surface.tangent;
#endif // OBJECTSHADER_USE_TANGENT

#if !defined(OBJECTSHADER_COMPILE_PS) && !defined(OBJECTSHADER_COMPILE_MS)
#ifdef OBJECTSHADER_USE_RENDERTARGETARRAYINDEX
	Out.RTIndex = camera.output_index;
#endif // OBJECTSHADER_USE_RENDERTARGETARRAYINDEX
#ifdef OBJECTSHADER_USE_VIEWPORTARRAYINDEX
	Out.VPIndex = camera.output_index;
#endif // OBJECTSHADER_USE_VIEWPORTARRAYINDEX
#endif // !defined(OBJECTSHADER_COMPILE_PS) && !defined(OBJECTSHADER_COMPILE_MS)

	return Out;
}


// OBJECT SHADER PROTOTYPE
///////////////////////////

#ifdef OBJECTSHADER_COMPILE_VS

// Vertex shader base:
PixelInput main(VertexInput input)
{
	return vertex_to_pixel_export(input);
}

#endif // OBJECTSHADER_COMPILE_VS


#ifdef OBJECTSHADER_COMPILE_PS

// Possible switches:
//	PREPASS				-	assemble object shader for depth prepass rendering
//	DEPTHONLY			-	assemble object shader for depth prepass rendering with no return
//	TRANSPARENT			-	assemble object shader for tiled forward transparent rendering
//	ENVMAPRENDERING		-	modify object shader for envmap rendering
//	PLANARREFLECTION	-	include planar reflection sampling
//	PARALLAXOCCLUSIONMAPPING					-	include parallax occlusion mapping computation
//	WATER				-	include specialized water shader code
//  ... and other material type specific defines

#if defined(__PSSL__) && defined(PREPASS) && !defined(DEPTHONLY)
#pragma PSSL_target_output_format (target 0 FMT_32_R)
#endif // __PSSL__ && PREPASS

#ifdef DISABLE_ALPHATEST
#define APPEND_COVERAGE_OUTPUT
#else
#define APPEND_COVERAGE_OUTPUT , out uint coverage : SV_Coverage
#endif // DISABLE_ALPHATEST

#ifdef EARLY_DEPTH_STENCIL
[earlydepthstencil]
#endif // EARLY_DEPTH_STENCIL

// entry point:
#ifdef PREPASS
#ifdef DEPTHONLY
void main(PixelInput input APPEND_COVERAGE_OUTPUT)
#else
uint main(PixelInput input APPEND_COVERAGE_OUTPUT) : SV_Target
#endif // DEPTHONLY
#else
float4 main(PixelInput input, in bool is_frontface : SV_IsFrontFace APPEND_COVERAGE_OUTPUT) : SV_Target
#endif // PREPASS


// Pixel shader base:
{
#ifdef OBJECTSHADER_USE_CAMERAINDEX
	ShaderCamera camera = GetCameraIndexed(input.GetCameraIndex());
#else
	ShaderCamera camera = GetCamera();
#endif // OBJECTSHADER_USE_CAMERAINDEX

	const min16uint2 pixel = input.pos.xy; // no longer pixel center!
	const float2 ScreenCoord = input.pos.xy * camera.internal_resolution_rcp; // use pixel center!
	
	Surface surface;
	surface.init();
	surface.P = input.GetPos3D();
	surface.V = input.GetViewVector();
	float dist = length(surface.V);
	surface.V /= dist;
	
#ifdef OBJECTSHADER_USE_UVSETS
	float4 uvsets = input.GetUVSets();
#endif // OBJECTSHADER_USE_UVSETS

#ifdef TILEDFORWARD
	write_mipmap_feedback(push.materialIndex, ddx_coarse(uvsets), ddy_coarse(uvsets));
#endif // TILEDFORWARD

#ifdef OBJECTSHADER_USE_INSTANCEINDEX
	ShaderMeshInstance meshinstance = load_instance(input.GetInstanceIndex());
#endif // OBJECTSHADER_USE_INSTANCEINDEX

	ShaderMaterial material = GetMaterial();


#ifdef OBJECTSHADER_USE_NORMAL
	if (is_frontface == false)
	{
		input.nor = -input.nor;
	}
	surface.N = normalize(input.nor);
	surface.facenormal = surface.N;
#ifndef PREPASS
	if (GetFrame().options & OPTION_BIT_DEBUG_NORMAL_VIS)
	{
		return float4(surface.facenormal * 0.5 + 0.5, 1.0);
	}
#endif
#endif // OBJECTSHADER_USE_NORMAL

#ifdef OBJECTSHADER_USE_COMMON
	surface.occlusion = input.ao_wet.x;
#endif // OBJECTSHADER_USE_COMMON

#ifdef OBJECTSHADER_USE_TANGENT
	surface.T = input.tan;
	surface.T.w = surface.T.w < 0 ? -1 : 1;
	half3 bitangent = cross(surface.T.xyz, input.nor) * surface.T.w;
	float3x3 TBN = float3x3(surface.T.xyz, bitangent, input.nor); // unnormalized TBN! http://www.mikktspace.com/
	
	surface.T.xyz = normalize(surface.T.xyz);

#ifndef PREPASS
	// GGMAX 1.62: tangent-frame visualization (harness SET_TANGENTVIS) — per-frame stability
	// forensics for the skinned normal-map flicker. Early-out with raw data as color.
	[branch]
	switch (GetFrame().gg_debugvis)
	{
	case 1: return float4(surface.T.xyz * 0.5 + 0.5, 1);                       // world tangent
	case 2: return float4(normalize(input.nor) * 0.5 + 0.5, 1);                // vertex normal
	case 4: return surface.T.w < 0 ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);  // handedness
	case 16: return float4(frac(surface.P * 0.05), 1);                         // world-position grid (skinned vertex stability)
	default: break;
	}
#endif // PREPASS

#ifdef PARALLAXOCCLUSIONMAPPING
	[branch]
	if (material.textures[DISPLACEMENTMAP].IsValid())
	{
		Texture2D<half4> tex = bindless_textures_half4[descriptor_index(material.textures[DISPLACEMENTMAP].texture_descriptor)];
		float2 uv = material.textures[DISPLACEMENTMAP].GetUVSet() == 0 ? uvsets.xy : uvsets.zw;
		float2 uv_dx = ddx_coarse(uv);
		float2 uv_dy = ddy_coarse(uv);

		ParallaxOcclusionMapping_Impl(
			uvsets,
			surface.V,
			TBN,
			material.GetParallaxOcclusionMapping(),
			tex,
			uv,
			uv_dx,
			uv_dy
		);
	}
#endif // PARALLAXOCCLUSIONMAPPING

#endif // OBJECTSHADER_USE_TANGENT


#ifdef OBJECTSHADER_USE_UVSETS

#ifndef PREPASS
	// GGMAX 1.62b tangent-vis: FINAL UV of the basecolor lookup (post-parallax), animation live.
	// 6 = raw frac(uv) as RG. 7 = frac(uv*64) amplified grid — one 8-bit gray step = uv shift of
	// ~6e-5 (~1/8 texel at 2048), so even sub-texel per-frame UV drift shows as pattern crawl.
	[branch]
	switch (GetFrame().gg_debugvis)
	{
	case 6:
	{
		float2 gg_uv = material.textures[BASECOLORMAP].GetUVSet() == 0 ? uvsets.xy : uvsets.zw;
		return float4(frac(gg_uv), 0, 1);
	}
	case 7:
	{
		float2 gg_uv = material.textures[BASECOLORMAP].GetUVSet() == 0 ? uvsets.xy : uvsets.zw;
		return float4(frac(gg_uv * 64.0), 0, 1);
	}
	default: break;
	}
#endif // PREPASS

#ifndef INTERIORMAPPING
	[branch]
#ifdef PREPASS
	if (material.textures[BASECOLORMAP].IsValid())
#else
	if (material.textures[BASECOLORMAP].IsValid() && (GetFrame().options & OPTION_BIT_DISABLE_ALBEDO_MAPS) == 0)
#endif // PREPASS
	{
		surface.baseColor *= material.textures[BASECOLORMAP].Sample(sampler_objectshader, uvsets);
	}
#endif // INTERIORMAPPING
	
#if defined(PREPASS) || defined(TRANSPARENT)
	[branch]
	if (material.textures[TRANSPARENCYMAP].IsValid())
	{
		surface.baseColor.a *= material.textures[TRANSPARENCYMAP].Sample(sampler_objectshader, uvsets).r;
	}
#endif // PREPASS || TRANSPARENT

#endif // OBJECTSHADER_USE_UVSETS


#ifdef OBJECTSHADER_USE_COLOR
	surface.baseColor *= input.color;
#endif // OBJECTSHADER_USE_COLOR

#ifndef PREPASS
	// GGMAX 1.62b tangent-vis: albedo-chain contributors
	[branch]
	switch (GetFrame().gg_debugvis)
	{
#ifdef OBJECTSHADER_USE_UVSETS
	case 8:  // raw basecolor texture sample
		if (material.textures[BASECOLORMAP].IsValid())
			return float4(material.textures[BASECOLORMAP].Sample(sampler_objectshader, uvsets).rgb, 1);
		return float4(1, 0, 1, 1); // magenta = no basecolor map
#endif
	case 9: return float4(surface.baseColor.rgb, 1);                            // final albedo input (x material & vertex color)
#ifdef OBJECTSHADER_USE_COLOR
	case 14: return float4(input.color.rgb, 1);                                 // vertex color
#endif
	default: break;
	}
#endif // PREPASS

#ifndef WATER
#ifdef OBJECTSHADER_USE_TANGENT
	[branch]
	if (material.textures[NORMALMAP].IsValid())
	{
		surface.bumpColor = half3(material.textures[NORMALMAP].Sample(sampler_objectshader, uvsets).rg, 1);
		surface.bumpColor = surface.bumpColor * 2 - 1;
		// GGMAX 1.63: do NOT pre-scale rg by normal strength here — see the apply site below.
		// Pre-scaling pushes the normal near-parallel to the surface at strength>1, where
		// sub-texel sample drift under animation swings it wildly (the "boiling texture"
		// character shimmer, churn-proven 3x calmer at DX11 semantics).
	}
#endif // OBJECTSHADER_USE_TANGENT
#endif // WATER

	surface.layerMask = material.layerMask & meshinstance.layerMask;


	half4 surfaceMap = 1;
#ifdef OBJECTSHADER_USE_UVSETS
	[branch]
	if (material.textures[SURFACEMAP].IsValid())
	{
		surfaceMap = material.textures[SURFACEMAP].Sample(sampler_objectshader, uvsets);
	}
#endif // OBJECTSHADER_USE_UVSETS

#ifdef OBJECTSHADER_USE_EMISSIVE
	// Emissive map:
	surface.emissiveColor = material.GetEmissive();

#ifdef OBJECTSHADER_USE_UVSETS
	[branch]
	if (any(surface.emissiveColor) && material.textures[EMISSIVEMAP].IsValid())
	{
		half4 emissiveMap = material.textures[EMISSIVEMAP].Sample(sampler_objectshader, uvsets);
		surface.emissiveColor *= emissiveMap.rgb * emissiveMap.a;
	}
#endif // OBJECTSHADER_USE_UVSETS

	surface.emissiveColor *= meshinstance.GetEmissive();
#endif // OBJECTSHADER_USE_EMISSIVE

#ifndef PREPASS
	// GGMAX 1.62b tangent-vis: surface-map / emissive contributors
	[branch]
	switch (GetFrame().gg_debugvis)
	{
	case 10: return float4(surfaceMap.rgb, 1);                    // ORM texture sample (occ/rough/metal)
	case 15: return float4(surface.emissiveColor, 1);             // emissive
	default: break;
	}
#endif // PREPASS

#ifdef OBJECTSHADER_USE_UVSETS
#ifdef TERRAINBLENDED
	[branch]
	if (material.GetTerrainBlendRcp() > 0)
	{
		// Blend object into terrain material:
		ShaderTerrain terrain = GetScene().terrain;
		[branch]
		if(terrain.chunk_buffer >= 0)
		{
			int2 chunk_coord = floor((surface.P.xz - terrain.center_chunk_pos.xz) / terrain.chunk_size);
			if(chunk_coord.x >= -terrain.chunk_buffer_range && chunk_coord.x <= terrain.chunk_buffer_range && chunk_coord.y >= -terrain.chunk_buffer_range && chunk_coord.y <= terrain.chunk_buffer_range)
			{
				uint chunk_idx = flatten2D(chunk_coord + terrain.chunk_buffer_range, terrain.chunk_buffer_range * 2 + 1);
				ShaderTerrainChunk chunk = bindless_structured_terrain_chunks[descriptor_index(terrain.chunk_buffer)][chunk_idx];
				
				[branch]
				if(chunk.heightmap >= 0)
				{
					Texture2D terrain_heightmap = bindless_textures[NonUniformResourceIndex(descriptor_index(chunk.heightmap))];
					float2 chunk_min = terrain.center_chunk_pos.xz + chunk_coord * terrain.chunk_size;
					float2 chunk_max = terrain.center_chunk_pos.xz + terrain.chunk_size + chunk_coord * terrain.chunk_size;
					float2 terrain_uv = saturate(inverse_lerp(chunk_min, chunk_max, surface.P.xz));
					float terrain_height0 = terrain_heightmap.SampleLevel(sampler_linear_clamp, terrain_uv, 0).r;
					float terrain_height1 = terrain_heightmap.SampleLevel(sampler_linear_clamp, terrain_uv, 0, int2(1, 0)).r;
					float terrain_height2 = terrain_heightmap.SampleLevel(sampler_linear_clamp, terrain_uv, 0, int2(0, 1)).r;
					float3 P0 = float3(0, terrain_height0, 0); 
					float3 P1 = float3(1, terrain_height1, 0); 
					float3 P2 = float3(0, terrain_height2, 1);
					float3 terrain_normal = normalize(cross(P2 - P0, P1 - P0));
					float terrain_height = lerp(terrain.min_height, terrain.max_height, terrain_height0);
					float object_height = surface.P.y;
					float diff = (object_height - terrain_height) * material.GetTerrainBlendRcp();
					float blend = 1 - sqr(saturate(diff));
					//blend *= lerp(1, saturate((noise_gradient_3D(surface.P * 2) * 0.5 + 0.5) * 2), saturate(diff));
					//terrain_uv = lerp(saturate(inverse_lerp(chunk_min, chunk_max, surface.P.xz - surface.N.xz * diff)), terrain_uv, saturate(surface.N.y)); // uv stretching improvement: stretch in normal direction if normal gets horizontal
					ShaderMaterial terrain_material = load_material(chunk.materialID);
					terrain_uv = mad(terrain_uv, terrain_material.texMulAdd.xy, terrain_material.texMulAdd.zw);
					float4 terrain_baseColor = terrain_material.textures[BASECOLORMAP].Sample(sampler_objectshader, terrain_uv.xyxy);
					float4 terrain_bumpColor = terrain_material.textures[NORMALMAP].Sample(sampler_objectshader, terrain_uv.xyxy);
					float4 terrain_surfaceMap = terrain_material.textures[SURFACEMAP].Sample(sampler_objectshader, terrain_uv.xyxy);
					float3 terrain_emissiveMap = terrain_material.textures[EMISSIVEMAP].Sample(sampler_objectshader, terrain_uv.xyxy).rgb;
					surface.baseColor = lerp(surface.baseColor, terrain_baseColor, blend);
					surface.bumpColor = lerp(surface.bumpColor, terrain_bumpColor.rgb * 2 - 1, blend);
					surfaceMap = lerp(surfaceMap, terrain_surfaceMap, blend);
					surface.emissiveColor += terrain_emissiveMap * terrain_material.GetEmissive() * blend;
					input.nor = lerp(input.nor, terrain_normal, blend);
					TBN[2] = input.nor;
					surface.N = normalize(input.nor);
				}
			}
		}
	}
#endif // TERRAINBLENDED
#endif // OBJECTSHADER_USE_UVSETS

	[branch]
	if (!material.IsUsingSpecularGlossinessWorkflow())
	{
		// Premultiply these before evaluating decals:
		surfaceMap.g *= material.GetRoughness();
		surfaceMap.b *= material.GetMetalness();
		surfaceMap.a *= material.GetReflectance();
	}

#ifdef TILEDFORWARD
	const uint flat_tile_index = GetFlatTileIndex(pixel);
#endif // TILEDFORWARD

#ifndef PREPASS
#ifndef WATER
#ifdef FORWARD
	ForwardDecals(surface, surfaceMap, sampler_objectshader_clamp);
#endif // FORWARD

#ifdef TILEDFORWARD
	TiledDecals(surface, flat_tile_index, surfaceMap, sampler_objectshader_clamp);
#endif // TILEDFORWARD
#endif // WATER
#endif // PREPASS


#ifndef WATER
#ifdef OBJECTSHADER_USE_TANGENT
	[branch]
	if (any(surface.bumpColor))
	{
		// GGMAX 1.63: DX11-parity normal strength. Old engine: N = normalize(lerp(N, bumped,
		// strength)) on the UNSCALED sample, then bumpColor *= strength for downstream users
		// (planar reflection UV shift, refraction) — the response saturates gracefully above 1.
		// New-engine rg pre-scale made strength 4 boil under animation. Water path (below)
		// already uses this exact lerp pattern.
		surface.N = normalize(lerp(surface.N, mul(surface.bumpColor, TBN), material.GetNormalMapStrength()));
		surface.bumpColor.rg *= material.GetNormalMapStrength();
	}
#ifndef PREPASS
	// GGMAX 1.62 tangent-vis (continued): post-bump modes
	[branch]
	switch (GetFrame().gg_debugvis)
	{
	case 3: return float4(surface.N * 0.5 + 0.5, 1);                                    // final bumped normal
	case 5: return float4(surface.bumpColor.rg * 0.5 + 0.5, 0, 1);                      // strength-scaled map sample
#ifdef OBJECTSHADER_USE_UVSETS
	case 21: // RAW normal-map texel data, no scale/expand (content-churn detector)
		if (material.textures[NORMALMAP].IsValid())
			return float4(material.textures[NORMALMAP].Sample(sampler_objectshader, uvsets).rg, 0, 1);
		return float4(1, 0, 1, 1);
	case 22: // RAW normal-map, forced mip 0 point-of-truth (kills aniso/mip footprint variation)
		if (material.textures[NORMALMAP].IsValid())
		{
			float2 gg_nuv = material.textures[NORMALMAP].GetUVSet() == 0 ? uvsets.xy : uvsets.zw;
			return float4(bindless_textures_half4[descriptor_index(material.textures[NORMALMAP].texture_descriptor)].SampleLevel(sampler_linear_wrap, gg_nuv, 0).rg, 0, 1);
		}
		return float4(1, 0, 1, 1);
#endif
	default: break;
	}
#endif // PREPASS
#endif // OBJECTSHADER_USE_TANGENT
#endif // WATER


	half4 specularMap = 1;

#ifdef OBJECTSHADER_USE_UVSETS
	[branch]
	if (material.textures[SPECULARMAP].IsValid())
	{
		specularMap = material.textures[SPECULARMAP].Sample(sampler_objectshader, uvsets);
	}
#endif // OBJECTSHADER_USE_UVSETS


	surface.create(material, surface.baseColor, surfaceMap, specularMap);
	
	

#ifdef OBJECTSHADER_USE_COMMON
	half wet = input.ao_wet.y;
	if(wet > 0)
	{
		surface.albedo = lerp(surface.albedo, 0, wet);
		surface.roughness = clamp(surface.roughness * sqr(1 - wet), 0.01, 1);
		surface.N = normalize(lerp(surface.N, input.nor, wet));
	}
#endif // OBJECTSHADER_USE_COMMON


#ifdef OBJECTSHADER_USE_UVSETS
	// Secondary occlusion map:
	[branch]
	if (material.IsOcclusionEnabled_Secondary() && material.textures[OCCLUSIONMAP].IsValid())
	{
		surface.occlusion *= material.textures[OCCLUSIONMAP].Sample(sampler_objectshader, uvsets).r;
	}
#endif // OBJECTSHADER_USE_UVSETS


#ifndef PREPASS
#ifndef ENVMAPRENDERING
#ifndef TRANSPARENT
#ifndef CARTOON
	[branch]
	if (camera.texture_ao_index >= 0)
	{
		surface.occlusion *= bindless_textures_half4[descriptor_index(camera.texture_ao_index)].SampleLevel(sampler_linear_clamp, ScreenCoord, 0).r;
	}
#endif // CARTOON
#endif // TRANSPARENT
#endif // ENVMAPRENDERING
#endif // PREPASS

#ifndef PREPASS
	// GGMAX 1.62b tangent-vis: derived surface parameters (post surface.create)
	[branch]
	switch (GetFrame().gg_debugvis)
	{
	case 11: return float4(surface.roughness.xxx, 1);             // final roughness
	case 12: return float4(surface.f0, 1);                        // specular F0
	case 13: return float4(surface.occlusion.xxx, 1);             // occlusion (vertex AO x map x SSAO)
	default: break;
	}
#endif // PREPASS


#ifdef ANISOTROPIC
	surface.aniso.strength = material.GetAnisotropy();
	surface.aniso.direction = half2(material.GetAnisotropyCos(), material.GetAnisotropySin());

#ifdef OBJECTSHADER_USE_UVSETS
	[branch]
	if (material.textures[ANISOTROPYMAP].IsValid())
	{
		half2 anisotropyTexture = material.textures[ANISOTROPYMAP].Sample(sampler_objectshader, uvsets).rg * 2 - 1;
		surface.aniso.strength *= length(anisotropyTexture);
		surface.aniso.direction = mul(half2x2(surface.aniso.direction.x, surface.aniso.direction.y, -surface.aniso.direction.y, surface.aniso.direction.x), normalize(anisotropyTexture));
	}
#endif // OBJECTSHADER_USE_UVSETS

	surface.aniso.T = normalize(mul(TBN, half3(surface.aniso.direction, 0)));

#endif // ANISOTROPIC


#ifdef SHEEN
	surface.sheen.color = material.GetSheenColor();
	surface.sheen.roughness = material.GetSheenRoughness();

#ifdef OBJECTSHADER_USE_UVSETS
	[branch]
	if (material.textures[SHEENCOLORMAP].IsValid())
	{
		surface.sheen.color = material.textures[SHEENCOLORMAP].Sample(sampler_objectshader, uvsets).rgb;
	}
	[branch]
	if (material.textures[SHEENROUGHNESSMAP].IsValid())
	{
		surface.sheen.roughness = material.textures[SHEENROUGHNESSMAP].Sample(sampler_objectshader, uvsets).a;
	}
#endif // OBJECTSHADER_USE_UVSETS
#endif // SHEEN


#ifdef CLEARCOAT
	surface.clearcoat.factor = material.GetClearcoat();
	surface.clearcoat.roughness = material.GetClearcoatRoughness();
	surface.clearcoat.N = input.nor;

#ifdef OBJECTSHADER_USE_UVSETS
	[branch]
	if (material.textures[CLEARCOATMAP].IsValid())
	{
		surface.clearcoat.factor = material.textures[CLEARCOATMAP].Sample(sampler_objectshader, uvsets).r;
	}
	[branch]
	if (material.textures[CLEARCOATROUGHNESSMAP].IsValid())
	{
		surface.clearcoat.roughness = material.textures[CLEARCOATROUGHNESSMAP].Sample(sampler_objectshader, uvsets).g;
	}
#ifdef OBJECTSHADER_USE_TANGENT
	[branch]
	if (material.textures[CLEARCOATNORMALMAP].IsValid())
	{
		half3 clearcoatNormalMap = half3(material.textures[CLEARCOATNORMALMAP].Sample(sampler_objectshader, uvsets).rg, 1);
		clearcoatNormalMap = clearcoatNormalMap * 2 - 1;
		surface.clearcoat.N = mul(clearcoatNormalMap, TBN);
	}
#endif // OBJECTSHADER_USE_TANGENT

	surface.clearcoat.N = normalize(surface.clearcoat.N);

#endif // OBJECTSHADER_USE_UVSETS
#endif // CLEARCOAT

	surface.sss = material.GetSSS();
	surface.sss_inv = material.GetSSSInverse();

#ifdef WATER
	surface.extinction = material.GetSheenColor().rgb; // Note: sheen color is repurposed as extinction color for water
#endif // WATER

	surface.pixel = pixel;
	surface.screenUV = ScreenCoord;

	surface.update();

	half3 ambient = GetAmbient(surface.N);
	ambient = lerp(ambient, ambient * surface.sss.rgb, saturate(surface.sss.a));

	Lighting lighting;
	lighting.create(0, 0, ambient, 0);

	
	half4 color = surface.baseColor;

#ifdef WATER
	//NORMALMAP
	half2 bumpColor0 = 0;
	half2 bumpColor1 = 0;
	half2 bumpColor2 = 0;
	[branch]
	if (material.textures[NORMALMAP].IsValid())
	{
		Texture2D<half4> texture_normalmap = bindless_textures_half4[descriptor_index(material.textures[NORMALMAP].texture_descriptor)];
		const float2 UV_normalMap = material.textures[NORMALMAP].GetUVSet() == 0 ? uvsets.xy : uvsets.zw;
		bumpColor0 = 2 * texture_normalmap.Sample(sampler_objectshader, UV_normalMap - material.texMulAdd.ww).rg - 1;
		bumpColor1 = 2 * texture_normalmap.Sample(sampler_objectshader, UV_normalMap + material.texMulAdd.zw).rg - 1;
	}
	[branch]
	if (camera.texture_waterriples_index >= 0)
	{
		bumpColor2 = bindless_textures_half4[descriptor_index(camera.texture_waterriples_index)].SampleLevel(sampler_linear_clamp, ScreenCoord, 0).rg;
	}
	surface.bumpColor = half3(bumpColor0 + bumpColor1 + bumpColor2, 1)  * material.GetRefraction();
	surface.N = normalize(lerp(surface.N, mul(normalize(surface.bumpColor), TBN), material.GetNormalMapStrength()));
	surface.bumpColor.rg *= material.GetNormalMapStrength();

	[branch]
	if (camera.texture_reflection_index >= 0)
	{
		//REFLECTION
		float4 reflectionUV = mul(camera.reflection_view_projection, float4(surface.P, 1));
		reflectionUV.xy /= reflectionUV.w;
		reflectionUV.xy = clipspace_to_uv(reflectionUV.xy) + surface.bumpColor.rg;
		half3 reflectiveColor = bindless_textures_half4[descriptor_index(camera.texture_reflection_index)].SampleLevel(sampler_linear_mirror, reflectionUV.xy, 0).rgb;
		[branch]
		if(camera.texture_reflection_depth_index >= 0)
		{
			float reflectiveDepth = bindless_textures[descriptor_index(camera.texture_reflection_depth_index)].SampleLevel(sampler_point_clamp, reflectionUV.xy, 0).r;
			float3 reflectivePosition = reconstruct_position(reflectionUV.xy, reflectiveDepth, camera.reflection_inverse_view_projection);
			float4 water_plane = camera.reflection_plane;
			float water_depth = -dot(float4(reflectivePosition, 1), water_plane);
			reflectiveColor.rgb = lerp(color.rgb, reflectiveColor.rgb, saturate(exp(-water_depth * color.a)));
		}
		lighting.indirect.specular += reflectiveColor * surface.F;
	}
#endif // WATER



#ifdef TRANSPARENT
	surface.transmission = lerp(material.GetTransmission(), 1, material.GetCloak());
	
	[branch]
	if (surface.transmission > 0)
	{
#ifdef OBJECTSHADER_USE_UVSETS
		[branch]
		if (material.textures[TRANSMISSIONMAP].IsValid())
		{
			half transmissionMap = (half)material.textures[TRANSMISSIONMAP].Sample(sampler_objectshader, uvsets).r;
			surface.transmission *= transmissionMap;
		}
#endif // OBJECTSHADER_USE_UVSETS

		[branch]
		if (camera.texture_refraction_index >= 0)
		{
			Texture2D<half4> texture_refraction = bindless_textures_half4[descriptor_index(camera.texture_refraction_index)];
			float2 size;
			float mipLevels;
			texture_refraction.GetDimensions(0, size.x, size.y, mipLevels);
			const float2 normal2D = mul((float3x3)camera.view, surface.N.xyz).xy;
			float2 perturbatedRefrTexCoords = ScreenCoord.xy + normal2D * lerp(material.GetRefraction(), 0.1, material.GetCloak());
			float mip = lerp(surface.roughness, 0.1, material.GetCloak()) * mipLevels;
			float chromatic = material.GetChromaticAberration() / size;
			half refractiveColorR = texture_refraction.SampleLevel(sampler_linear_clamp, perturbatedRefrTexCoords + float2(1, 1) * chromatic, mip).r;
			half refractiveColorG = texture_refraction.SampleLevel(sampler_linear_clamp, perturbatedRefrTexCoords + float2(0, 0) * chromatic, mip).g;
			half refractiveColorB = texture_refraction.SampleLevel(sampler_linear_clamp, perturbatedRefrTexCoords - float2(1, 1) * chromatic, mip).b;
			half3 refractiveColor = half3(refractiveColorR, refractiveColorG, refractiveColorB);
			surface.refraction.rgb = lerp(surface.albedo, 1, material.GetCloak()) * refractiveColor.rgb;
			surface.refraction.a = surface.transmission;
		}
	}
#endif // TRANSPARENT


#ifdef OBJECTSHADER_USE_COMMON
	LightMapping(meshinstance.lightmap, input.atl, lighting, surface);
#endif // OBJECTSHADER_USE_COMMON


#ifdef PLANARREFLECTION
	lighting.indirect.specular += PlanarReflection(surface, surface.bumpColor.rg) * surface.F;
#endif


#ifdef FORWARD
	ForwardLighting(surface, lighting);
#endif // FORWARD


#ifdef TILEDFORWARD
	TiledLighting(surface, lighting, flat_tile_index);
#endif // TILEDFORWARD


#ifndef WATER
#ifndef ENVMAPRENDERING
#ifndef TRANSPARENT
#ifndef CARTOON
	[branch]
	if (camera.texture_ssr_index >= 0)
	{
		half4 ssr = bindless_textures_half4[descriptor_index(camera.texture_ssr_index)].SampleLevel(sampler_linear_clamp, ScreenCoord, 0);
		lighting.indirect.specular = lerp(lighting.indirect.specular, ssr.rgb * surface.F, ssr.a);
	}
	[branch]
	if (camera.texture_ssgi_index >= 0)
	{
		surface.ssgi = bindless_textures_half4[descriptor_index(camera.texture_ssgi_index)].SampleLevel(sampler_linear_clamp, ScreenCoord, 0).rgb;
	}
#endif // CARTOON
#endif // TRANSPARENT
#endif // ENVMAPRENDERING
#endif // WATER

#ifdef WATER
	[branch]
	if (camera.texture_refraction_index >= 0)
	{
		// Water refraction:
		float4 water_plane = camera.reflection_plane;
		const float camera_above_water = dot(float4(camera.position, 1), water_plane) < 0; 
		Texture2D<half4> texture_refraction = bindless_textures_half4[descriptor_index(camera.texture_refraction_index)];
		// First sample using full perturbation:
		float2 refraction_uv = ScreenCoord.xy + surface.bumpColor.rg;
		float refraction_depth = find_max_depth(refraction_uv, 2, 2);
		float3 refraction_position = reconstruct_position(refraction_uv, refraction_depth);
		float water_depth = -dot(float4(refraction_position, 1), water_plane);
		if(camera_above_water)
			water_depth = -water_depth;
		if(water_depth <= 0)
		{
			// Above water, fill holes by taking unperturbed sample:
			refraction_uv = ScreenCoord.xy;
		}
		else
		{
			// Below water, compute perturbation according to first sample water depth:
			refraction_uv = ScreenCoord.xy + surface.bumpColor.rg * saturate(1 - exp(-water_depth));
		}
		surface.refraction.rgb = (half3)texture_refraction.SampleLevel(sampler_linear_mirror, refraction_uv, 0).rgb;
		// Recompute depth params again with actual perturbation:
		refraction_depth = texture_depth.SampleLevel(sampler_point_clamp, refraction_uv, 0);
		refraction_position = reconstruct_position(refraction_uv, refraction_depth);
		water_depth = max(water_depth, -dot(float4(refraction_position, 1), water_plane));
		if(camera_above_water)
			water_depth = -water_depth;
		// Water fog computation:
		float waterfog = saturate(exp(-water_depth * color.a));
		float3 transmittance = saturate(exp(-water_depth * surface.extinction * color.a));
		surface.refraction.a = waterfog;
		surface.refraction.rgb *= transmittance;
		color.a = 1;
	}
#endif // WATER

#ifndef PREPASS
	// GGMAX 1.62b tangent-vis: lighting channels (Reinhard x/(1+x) so HDR stays readable)
	[branch]
	switch (GetFrame().gg_debugvis)
	{
	case 17: { float3 c = lighting.direct.diffuse;    return float4(c / (1 + c), 1); }  // sun/local diffuse incl shadows
	case 18: { float3 c = lighting.direct.specular;   return float4(c / (1 + c), 1); }  // direct specular
	case 19: { float3 c = lighting.indirect.diffuse;  return float4(c / (1 + c), 1); }  // ambient/GI diffuse
	case 20: { float3 c = lighting.indirect.specular; return float4(c / (1 + c), 1); }  // reflections/probes
	default: break;
	}
#endif // PREPASS

	ApplyLighting(surface, lighting, color);


#ifdef OBJECTSHADER_USE_INSTANCEINDEX
	half4 rimHighlight = meshinstance.GetRimHighlight();
	color.rgb += rimHighlight.rgb * pow(1 - surface.NdotV, rimHighlight.w);
#endif // OBJECTSHADER_USE_INSTANCEINDEX


#ifdef UNLIT
	color = surface.baseColor;
#endif // UNLIT

#ifdef INTERIORMAPPING
	surface.baseColor.rgb += surface.emissiveColor;
	color = surface.baseColor * InteriorMapping(surface.P, surface.N, surface.V, material, meshinstance);
#endif // INTERIORMAPPING


// Transparent objects has been rendered separately from opaque, so let's apply it now.
// Must also be applied before fog since fog is layered over.
#ifdef TRANSPARENT
	ApplyAerialPerspective(ScreenCoord, surface.P, color);
#endif // TRANSPARENT


	ApplyFog(dist, surface.V, color);

	color.rgb = mul(saturationMatrix(material.GetSaturation()), color.rgb);

#ifndef ENVMAPRENDERING
	// GGMAX 2.79 (#157 debug rig): ENV-ONLY. Throw away EVERYTHING this shader just computed
	// and output the raw GLOBAL env-probe cube texel along the mirror reflection vector, so
	// what is on screen IS the cube and nothing else: no basecolor/normal/surface/emissive/
	// occlusion maps, no lightmap, no lights or shadows, no SSAO/SSR/GI/decals, no fog, no
	// aerial perspective, no saturation, and NOT multiplied by fresnel (surface.F) or by
	// gg_envprobe_brightness the way EnvironmentReflection_Global does it (lightingHF:705).
	// Guarded out of ENVMAPRENDERING so a probe capture never photographs the debug output
	// back into the cube it is displaying.
	// GGMAX 2.79a: modes, so the cube can be separated from the VECTOR used to sample it.
	//   1 = cube along surface.R      (mirror reflection using the SHADING normal = normal-mapped)
	//   2 = cube along the FACE normal (geometry only — normal map excluded)
	//   3 = surface.N    as colour     (shading normal, x*0.5+0.5)
	//   4 = surface.facenormal as colour (geometric normal)
	//   5 = surface.R    as colour     (the reflection vector itself)
	// If a pattern survives in 1 but vanishes in 2, the normals are bending R into a different
	// part of the cube — the cube is innocent and the mesh/normal map is the story.
	[branch]
	if (GetScene().gg_envonly > 0 && GetScene().globalprobe >= 0)
	{
		const int gg_envonly_mode = (int)GetScene().gg_envonly;
		const half3 gg_envonly_Rgeo = -reflect(surface.V, surface.facenormal);
		if (gg_envonly_mode == 3)
		{
			color = half4(surface.N * 0.5 + 0.5, 1);
		}
		else if (gg_envonly_mode == 4)
		{
			color = half4(surface.facenormal * 0.5 + 0.5, 1);
		}
		else if (gg_envonly_mode == 5)
		{
			color = half4(surface.R * 0.5 + 0.5, 1);
		}
		else
		{
			TextureCube<half4> gg_envonly_cube = bindless_cubemaps_half4[descriptor_index(GetScene().globalprobe)];
			uint2 gg_envonly_dim;
			uint gg_envonly_mipcount;
			gg_envonly_cube.GetDimensions(0, gg_envonly_dim.x, gg_envonly_dim.y, gg_envonly_mipcount);
			const half gg_envonly_lod = (half)clamp(GetScene().gg_envonly_mip, 0.0, (float)gg_envonly_mipcount - 1.0);
			const half3 gg_envonly_dir = (gg_envonly_mode == 2) ? gg_envonly_Rgeo : surface.R;
			// 2.80: if the cube has been replaced by a solid colour, this rig shows that too —
			// the two debug paths must never disagree about what the cube contains.
			half3 gg_envonly_rgb = (GetScene().gg_envsolid.w > 0)
				? (half3)GetScene().gg_envsolid.rgb
				: gg_envonly_cube.SampleLevel(sampler_linear_clamp, gg_envonly_dir, gg_envonly_lod).rgb;

			// GGMAX 2.79b: the LADDER — cube plus exactly ONE more source at a time, so each
			// contributor can be added and judged on its own. 9 and 10 show a source ALONE.
			if (gg_envonly_mode == 6)
			{
				gg_envonly_rgb += GetAmbient(surface.N);				// + ambient (SAME cube at its coarsest mip, plus the flat weather ambient colour)
			}
			else if (gg_envonly_mode == 7)
			{
				gg_envonly_rgb *= surface.F;						// x fresnel, exactly as EnvironmentReflection_Global weights it
			}
			else if (gg_envonly_mode == 8)
			{
				gg_envonly_rgb *= surface.occlusion;				// x occlusion (ORM red channel, secondary AO map and SSAO)
			}
			else if (gg_envonly_mode == 9)
			{
				gg_envonly_rgb = GetAmbient(surface.N);				// ambient ALONE, no cube reflection
			}
			else if (gg_envonly_mode == 10)
			{
				gg_envonly_rgb = (half3)GetDynamicSkyColor(surface.R);	// the SKY alone, sampled along the reflection vector
			}
			color = half4(gg_envonly_rgb, 1);
		}
	}
#endif // ENVMAPRENDERING

	color = saturateMediump(color);

	half alphatest = material.GetAlphaTest() + meshinstance.GetAlphaTest();

	half dithering = 0;
#ifndef DISABLE_ALPHATEST
#ifndef TRANSPARENT
#ifndef ENVMAPRENDERING
#ifdef OBJECTSHADER_USE_DITHERING
	dithering = input.GetDither();
#endif // OBJECTSHADER_USE_DITHERING
#endif // DISABLE_ALPHATEST
#endif // TRANSPARENT
#endif // ENVMAPRENDERING

#ifndef DISABLE_ALPHATEST
	coverage = AlphaToCoverage(color.a, alphatest, dithering, input.pos); // opaque soft alpha test (MSAA, temporal AA support)
#endif // DISABLE_ALPHATEST
	
	// end point:
#ifdef PREPASS
#ifndef DEPTHONLY
	PrimitiveID prim;
	prim.init();
	prim.primitiveIndex = input.primitiveID - GetMesh().indexOffset / 3;
	prim.instanceIndex = input.GetInstanceIndex();
	prim.subsetIndex = push.geometryIndex - meshinstance.geometryOffset;
	return prim.pack();
#endif // DEPTHONLY
#else
	return color;
#endif // PREPASS
}


#endif // OBJECTSHADER_COMPILE_PS



#endif // WI_OBJECTSHADER_HF

