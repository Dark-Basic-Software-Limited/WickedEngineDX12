#define DISABLE_DECALS
#define DISABLE_ENVMAPS
#define SHADOW_MASK_ENABLED
#include "globals.hlsli"
#include "objectHF.hlsli"
#include "hairparticleHF.hlsli"

[earlydepthstencil]
float4 main(VertexToPixel input) : SV_Target
{
	// GGMAX 1.90 flicker instrument (SET_GRASSTYPEFREEZE 5): paint the raw per-strand grasstype
	// as a flat colour and return immediately. This value NEVER touches a descriptor, so the
	// consecutive-frame diff splits the remaining search space in half:
	//   STABLE image  -> the type value is steady, so the corruption happens at texture FETCH
	//   CHURNING image-> the type value itself varies per frame, so the bug is UPSTREAM of the fetch
	// Deliberately before every other read so nothing else can contaminate the reading.
	if (xHairFlags & HAIR_FLAG_GG_DEBUG_TYPEVIS)
	{
		const uint t = input.grasstype;
		// Distinct, high-contrast colour per type so a single flipped strand is unmissable, and
		// so the pixel-diff metric responds strongly to any change.
		return float4(
			((t * 37u) & 255u) / 255.0,
			((t * 91u) & 255u) / 255.0,
			((t * 173u) & 255u) / 255.0,
			1);
	}

	ShaderMaterial material = HairGetMaterial();
	ShaderMeshInstance meshinstance = HairGetInstance();
	
	write_mipmap_feedback(HairGetGeometry().materialIndex, ddx_coarse(input.tex.xyxy), ddy_coarse(input.tex.xyxy));

	half4 color = 1;

	// GGMAX 1.74 merged grass: a merged chunk holds every painted type in one system, so the
	// blade texture is chosen per strand rather than per material.
	// GGMAX 1.96 flicker fix: the subscript+Sample must be ONE expression with the NonUniform
	// annotation at the subscript — dxc drops the annotation if the handle passes through a
	// local/out-param (see GGGetGrassTextureIndex). Verify with dxc -dumpbin: nonUniformIndex=true.
	uint ggtexidx;
	[branch]
	if (input.GGGetGrassTextureIndex(ggtexidx) && (GetFrame().options & OPTION_BIT_DISABLE_ALBEDO_MAPS) == 0)
	{
		color = bindless_textures_half4[NonUniformResourceIndex(descriptor_index(ggtexidx))].Sample(sampler_linear_wrap, input.tex.xy);
	}
	else
	[branch]
	if (material.textures[BASECOLORMAP].IsValid() && (GetFrame().options & OPTION_BIT_DISABLE_ALBEDO_MAPS) == 0)
	{
		color = material.textures[BASECOLORMAP].Sample(sampler_linear_wrap, input.tex.xyxy);
	}
	color *= material.GetBaseColor();

	float3 V = input.GetViewVector();
	float dist = length(V);
	V /= dist;
	half emissive = 0;

	const min16uint2 pixel = input.pos.xy; // no longer pixel center!
	const float2 ScreenCoord = input.pos.xy * GetCamera().internal_resolution_rcp; // use pixel center!

	Surface surface;
	surface.init();
	surface.create(material, color, surfacemap_simple);
	surface.P = input.GetPos3D();
	surface.N = input.nor_wet.xyz;
	surface.V = V;
	surface.pixel = input.pos.xy;

#ifndef PREPASS
#ifndef ENVMAPRENDERING
#ifndef TRANSPARENT
#ifndef CARTOON
	[branch]
	if (GetCamera().texture_ao_index >= 0)
	{
		surface.occlusion *= bindless_textures_half4[descriptor_index(GetCamera().texture_ao_index)].SampleLevel(sampler_linear_clamp, ScreenCoord, 0).r;
	}
	[branch]
	if (GetCamera().texture_ssgi_index >= 0)
	{
		surface.ssgi = bindless_textures[descriptor_index(GetCamera().texture_ssgi_index)].SampleLevel(sampler_linear_clamp, ScreenCoord, 0).rgb;
	}
#endif // CARTOON
#endif // TRANSPARENT
#endif // ENVMAPRENDERING
#endif // PREPASS

	half wet = input.nor_wet.w;
	if(wet > 0)
	{
		surface.albedo = lerp(surface.albedo, 0, wet);
	}

	surface.update();

	Lighting lighting;
	lighting.create(0, 0, GetAmbient(surface.N), 0);

	TiledLighting(surface, lighting, GetFlatTileIndex(pixel));
	
	ApplyLighting(surface, lighting, color);

#ifdef TRANSPARENT
	ApplyAerialPerspective(ScreenCoord, surface.P, color);
#endif // TRANSPARENT
	
	ApplyFog(dist, V, color);

	color.rgb = mul(saturationMatrix(material.GetSaturation()), color.rgb);
	
	return color;
}
