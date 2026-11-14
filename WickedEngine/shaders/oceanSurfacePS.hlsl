#define DISABLE_DECALS
#define DISABLE_ENVMAPS
#define DISABLE_TRANSPARENT_SHADOWMAP
#define DISABLE_SOFT_SHADOWMAP
#define TRANSPARENT
#define WATER
#include "globals.hlsli"
#include "oceanSurfaceHF.hlsli"
#include "objectHF.hlsli"

Texture2D<float4> texture_ocean_displacementmap : register(t0);
Texture2D<float4> texture_gradientmap : register(t1);

[earlydepthstencil]
float4 main(PSIn input) : SV_TARGET
{
	float lineardepth = GetCamera().IsOrtho() ? ((1 - input.pos.z) * GetCamera().z_far) : input.pos.w;
	half4 color = xOceanWaterColor;
	// GG BR-0039 round 2 (Lee 11-14, "Same as Water Bake"): with the DX11 water fog on (xOceanFogMax > 0, camera above
	// the water) the Water Base Color alpha is a MINIMUM OPACITY, as GG's Water Bake treats it (GGWaterBakePS: fade capped
	// at 1 - max(Water Fog Minimum, A)) - not stock Wicked's exp(-depth x A) extinction, which made any A > 0 water (the
	// Plains biome sets 255) opaque within inches and the Water Fog sliders inert. DX11 ignored A. A 0 = DX11 exactly.
	const float gg_wc_alpha = color.a;
	const bool gg_fog_on = xOceanFogMax > 0 && (GetCamera().position.y - xOceanWaterHeight) >= 0;
	const float gg_extinction_a = gg_fog_on ? 0 : gg_wc_alpha;
	float2 ScreenCoord = input.pos.xy * GetCamera().internal_resolution_rcp;
	
	float3 V = input.GetViewVector();
	float dist = length(V);
	V /= dist;
	
	uint2 pixel = input.pos.xy;

	const half gradient_fade = saturate(dist * 0.001);
	const half4 gradientNear = texture_gradientmap.Sample(sampler_aniso_wrap, input.uv);
	const half4 gradientFar = texture_gradientmap.Sample(sampler_aniso_wrap, input.uv * 0.125);
	half4 gradient = lerp(gradientNear, gradientFar, gradient_fade);
	
	[branch]
	if (GetCamera().texture_waterriples_index >= 0)
	{
		gradient.rg += bindless_textures_half4[descriptor_index(GetCamera().texture_waterriples_index)].SampleLevel(sampler_linear_clamp, ScreenCoord, 0).rg * 0.025;
	}

	Surface surface;
	surface.init();
	surface.SetReceiveShadow(true);
	surface.pixel = pixel;
	float depth = input.pos.z;
	surface.albedo = color.rgb;
	surface.f0 = 0.02;
	surface.roughness = 0.1;
	surface.P = input.GetPos3D();
	surface.N = normalize(float3(gradient.x, xOceanTexelLength * 2, gradient.y));
	surface.V = V;
	surface.sss = 1;
	surface.sss_inv = 1.0f / ((1 + surface.sss) * (1 + surface.sss));
	surface.extinction = xOceanExtinctionColor.rgb;
	surface.update();

	Lighting lighting;
	lighting.create(0, 0, GetAmbient(surface.N), 0);

	TiledLighting(surface, lighting, GetFlatTileIndex(pixel));

	const float bump_strength = 0.1;
	
	float4 water_plane = GetCamera().reflection_plane;
	
	[branch]
	if (GetCamera().texture_reflection_index >= 0)
	{
		//REFLECTION
		float4 reflectionPos = mul(GetCamera().reflection_view_projection, float4(surface.P, 1));
		float2 reflectionUV = clipspace_to_uv(reflectionPos.xy / reflectionPos.w) + surface.N.xz * bump_strength;
		half4 reflectiveColor = bindless_textures[descriptor_index(GetCamera().texture_reflection_index)].SampleLevel(sampler_linear_mirror, reflectionUV, 0);
		[branch]
		if(GetCamera().texture_reflection_depth_index >=0)
		{
			// GGMAX 3.90: sampler_point_clamp -> sampler_linear_clamp. This depth drives a hard
			// exp() blend, so a point tap snaps between adjacent quarter-res texels. Inert on a
			// level whose Water Base Color alpha is 0 (GG's default - the lerp becomes exp(0)=1),
			// which is why this is a rider and not the fix. Twin: objectHF.hlsli.
			float reflectiveDepth = bindless_textures[descriptor_index(GetCamera().texture_reflection_depth_index)].SampleLevel(sampler_linear_clamp, reflectionUV, 0).r;
			float3 reflectivePosition = reconstruct_position(reflectionUV, reflectiveDepth, GetCamera().reflection_inverse_view_projection);
			float water_depth = -dot(float4(reflectivePosition, 1), water_plane);
			water_depth += texture_ocean_displacementmap.SampleLevel(sampler_linear_wrap, reflectivePosition.xz * xOceanPatchSizeRecip, 0).z; // texture contains xzy!
			reflectiveColor.rgb = lerp(color.rgb, reflectiveColor.rgb, saturate(exp(-water_depth * gg_extinction_a)));
		}
		lighting.indirect.specular = reflectiveColor.rgb * surface.F * saturate(dist * 0.1); // fade out very close to camera, doesn't look good
	}
	else
	{
		lighting.indirect.specular = EnvironmentReflection_Global(surface);
	}

	float water_depth = FLT_MAX;
	bool gg_dx11_water = false; // GG BR-0039: the DX11 water fog ran - composite as DX11 below

	[branch]
	if (GetCamera().texture_refraction_index >= 0)
	{
		// Water refraction:
		const float camera_above_water = dot(float4(GetCamera().position, 1), water_plane) < 0; 
		Texture2D texture_refraction = bindless_textures[descriptor_index(GetCamera().texture_refraction_index)];
		// First sample using full perturbation:
		float2 refraction_uv = ScreenCoord.xy + surface.N.xz * bump_strength;
		float refraction_depth = find_max_depth(refraction_uv, 2, 2);
		float3 refraction_position = reconstruct_position(refraction_uv, refraction_depth);
		water_depth = -dot(float4(refraction_position, 1), water_plane);
		water_depth += texture_ocean_displacementmap.SampleLevel(sampler_linear_wrap, refraction_position.xz * xOceanPatchSizeRecip, 0).z; // texture contains xzy!
		if (camera_above_water && V.y < 0)
			water_depth = -water_depth;
		if (water_depth <= 0)
		{
			// Above water, fill holes by taking unperturbed sample:
			refraction_uv = ScreenCoord.xy;
		}
		else
		{
			// Below water, compute perturbation according to first sample water depth:
			refraction_uv = ScreenCoord.xy + surface.N.xz * bump_strength * saturate(1 - exp(-water_depth));
		}
		surface.refraction.rgb = texture_refraction.SampleLevel(sampler_linear_mirror, refraction_uv, 0).rgb;
		// Recompute depth params again with actual perturbation:
		refraction_depth = texture_depth.SampleLevel(sampler_point_clamp, refraction_uv, 0);
		refraction_position = reconstruct_position(refraction_uv, refraction_depth);
		water_depth = max(water_depth, -dot(float4(refraction_position, 1), water_plane));
		water_depth += texture_ocean_displacementmap.SampleLevel(sampler_linear_wrap, refraction_position.xz * xOceanPatchSizeRecip, 0).z; // texture contains xzy!
		if (camera_above_water && V.y < 0)
			water_depth = -water_depth;
		// Water fog computation:
		float waterfog = saturate(exp(-water_depth * gg_extinction_a));
		float3 transmittance = saturate(exp(-water_depth * surface.extinction * gg_extinction_a));
		surface.refraction.a = waterfog;
		surface.refraction.rgb *= transmittance;
		// GG delta 1.24: tint the see-through (refraction) toward the Water Base Color with depth,
		// so a chosen colour is visible even on fully-transparent water (color.a == 0) — clear
		// shallows, coloured depths. xOceanWaterColorDepth 0 = off (stock).
		surface.refraction.rgb = lerp(surface.refraction.rgb, color.rgb, saturate(abs(water_depth) * xOceanWaterColorDepth));
		// GG BR-0039: DX11's water fog (DX11 oceanSurfacePS "apply water fog"). The see-through fades
		// toward the LIT water colour - ApplyLighting's lerp(albedo * diffuse, refraction, refraction.a)
		// - by the thickness of water along the view ray: the linear-depth gap from the surface to the
		// bed (sky = z_far: open water goes opaque), boosted by the camera height as DX11 did. Same
		// maths as GG's Water Bake (GGWaterBakePS). Camera below the water: unchanged (underwaterCS
		// fogs that view). xOceanFogMax 0 = off (stock).
		// ⚠ Not the stock "camera_above_water" above: reflection_plane is (0,1,0,-height), so that one is
		// TRUE with the camera BELOW the water (its sign is only used to flip water_depth). DX11 tested
		// the camera Y against the water height directly, and so does this.
		const float camera_height = GetCamera().position.y - xOceanWaterHeight;
		[branch]
		if (gg_fog_on)
		{
			const float bedZ = texture_lineardepth.SampleLevel(sampler_point_clamp, refraction_uv, 0) * GetCamera().z_far;
			float t = max(0, bedZ - lineardepth);
			t += t * camera_height / max(dist, 0.0001);
			float fade = exp(-4.0 * max(0, t - xOceanFogMin) / max(xOceanFogMax - xOceanFogMin, 0.1));
			fade = min(saturate(1 - max(xOceanFogMinAmount, gg_wc_alpha)), fade);   // + the authored A as a floor (Water Bake)
			surface.refraction.a *= fade;
			gg_dx11_water = true;
		}
		color.a = 1;
	}
	
#if 1
	// FOAM:
	// All the constants below were tuned for a world measured in METERS. xOceanFoamUnitScale
	// converts world distances/positions into those tuned units (1 = stock; GGMAX passes
	// ~0.0254 because its world is inch-scaled — without this the shore band is ~40x too thin
	// and the foam noise repeats every inch). xOceanFoamAmount is an artistic intensity knob.
	const float foam_unit = xOceanFoamUnitScale;
	float water_depth_diff = abs(texture_lineardepth[pixel] * GetCamera().z_far - lineardepth) * foam_unit; // Note: for the shore foam, this is more accurate than water plane distance
	float water_depth_foam = water_depth * foam_unit;
	float2 foamP = surface.P.xz * foam_unit;
	float foam_shore = saturate(exp(-water_depth_diff * 2));
	float foam_wave = pow(saturate(gradient.a), 4) * saturate(exp(-water_depth_foam * 0.1));
	float foam_combined = saturate(foam_shore + foam_wave);
	float foam_simplex = 0;
	foam_simplex += smoothstep(0, 0.8, noise_simplex_2D(foamP * 1 + GetTime()));
	foam_simplex += smoothstep(0, 0.8, noise_simplex_2D(foamP * 2 + GetTime()));
	foam_simplex += smoothstep(0, 0.8, noise_simplex_2D(foamP.yx * 4 - GetTime() * 2));
	float foam_voronoi = 0;
	foam_voronoi += smoothstep(0.5, 0.8, noise_voronoi(foamP * 1, GetTime()).x);
	foam_voronoi += smoothstep(0.5, 0.8, noise_voronoi(foamP * 2, GetTime()).x);
	foam_voronoi += smoothstep(0.5, 0.8, noise_voronoi(foamP * 4, GetTime()).x);
	float foam = 0;
	foam += foam_voronoi * foam_simplex * foam_combined;
	foam += smoothstep(0.5, 0.6, saturate(foam_combined + 0.1));
	foam *= 2 * xOceanFoamAmount;
	foam = saturate(foam);
	surface.albedo = lerp(surface.albedo, 0.6, foam);
	surface.refraction.a *= 1 - foam;
	surface.refraction.a = saturate(surface.refraction.a + saturate(exp(-water_depth_diff * 4)));
#endif

	// GG BR-0039: DX11's composite with its water fog - color = lerp(fogged see-through, 0.5 x reflection, F * 0.5)
	// + sun specular (DX11 oceanSurfacePS "reflection *= 0.5" then "fresnel = surface.F * 0.5"; Water Bake's
	// "keep" term). indirect.specular is already reflection x F, so x 0.25 = DX11's 0.5 F x 0.5 R. Stock adds
	// F x reflection ON TOP of the full see-through, so at grazing angles the sky reflection drowns the water
	// colour - why the far water barely changed with the Water Base Color. The minimum fog is re-applied after
	// the stock shoreline term above (which lifts refraction.a to ~1 at the waterline), so Water Fog Minimum
	// tints right up to the shore as in DX11 and Water Bake. Not DX11: the lit colour keeps Wicked's own
	// lighting (full ambient x GI boost; DX11/Water Bake: sun/PI + 0.5 ambient), and with planar reflections
	// off the global probe reflection is kept at 0.25 (DX11 blended toward black there).
	[branch]
	if (gg_dx11_water)
	{
		surface.refraction.a = min(surface.refraction.a, saturate(1 - max(xOceanFogMinAmount, gg_wc_alpha)));
		const half3 keep = 1 - 0.5 * surface.F;
		surface.albedo *= keep;
		surface.refraction.rgb *= keep;
		lighting.indirect.specular *= 0.25;
	}

	ApplyLighting(surface, lighting, color);
	
	// Blend out at distance:
	color.a = saturate(1 - saturate(dist / GetCamera().z_far - 0.8) * 5.0); // fade will be on edge and inwards 20%
	
	ApplyAerialPerspective(ScreenCoord, surface.P, color);
	
	ApplyFog(dist, V, color);

	return saturateMediump(color);
}

