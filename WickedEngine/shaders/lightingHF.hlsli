#ifndef WI_LIGHTING_HF
#define WI_LIGHTING_HF
#include "globals.hlsli"
#include "shadowHF.hlsli"
#include "brdf.hlsli"
#include "voxelConeTracingHF.hlsli"
#include "skyHF.hlsli"

#ifdef CARTOON
#define DISABLE_SOFT_SHADOWMAP
#endif // CARTOON

#ifdef WATER
#define LIGHTING_SCATTER
#endif // WATER

struct LightingPart
{
	half3 diffuse;
	half3 specular;
};
struct Lighting
{
	LightingPart direct;
	LightingPart indirect;

	inline void create(
		in half3 diffuse_direct,
		in half3 specular_direct,
		in half3 diffuse_indirect,
		in half3 specular_indirect
	)
	{
		direct.diffuse = diffuse_direct;
		direct.specular = specular_direct;
		indirect.diffuse = diffuse_indirect;
		indirect.specular = specular_indirect;
	}
};

inline void ApplyLighting(in Surface surface, in Lighting lighting, inout half4 color)
{
	half3 diffuse = lighting.direct.diffuse / PI + lighting.indirect.diffuse * GetGIBoost() * (1 - surface.F) * surface.occlusion + surface.ssgi;
	half3 specular = lighting.direct.specular + lighting.indirect.specular * surface.occlusion; // reminder: cannot apply surface.F for whole indirect specular, because multiple layers have separate fresnels (sheen, clearcoat)
	color.rgb = lerp(surface.albedo * diffuse, surface.refraction.rgb, surface.refraction.a);
	color.rgb += specular;
	color.rgb += surface.emissiveColor;
}

//#define CASCADE_DITHERING
// GGMAX 2.14: DX11 SHADERTYPE_WEAPON / WEAPON_SHADOW parity.
//
// DX11 (WickedRepo lightingHF.hlsli:451/569/635, under #ifdef WEAPON_SHADOW) pulled the shading
// position to a THIRD of the camera distance at the top of DirectionalLight/PointLight/SpotLight:
//     diffxytz = surface.P - camPos;  diffxytz /= 3;  surface.P = camPos + diffxytz;
// Its own comment: "ensure geometry not STUCK INSIDE a wall that would cast a shadow on it". The
// first-person weapon sits within ~1 unit of the eye, so the pull barely moves it in world space,
// but it lands the shadow lookup back on the PLAYER'S side of any wall the gun is clipping into —
// otherwise the gun goes dark whenever the player presses against geometry.
//
// DX11 achieved this with a whole shader permutation (objectPS_weapon.hlsl + SHADERTYPE_WEAPON).
// Here it is a wave-uniform bool on Surface instead: no new SHADERTYPE (that enum is upstream-owned
// AND serialized into materials), no SHADERTYPE_BIN_COUNT bump, no extra permutation.
//
// `surface` is an `in` parameter — the mutation is on the callee's local copy, exactly as in DX11,
// so it affects only this light's shadow lookup and attenuation, never the caller's surface.
// NOT applied to light_rect: DX11 had no rect equivalent, so leaving it out is the parity choice.
// surface.IsWeaponShadow() already has OPTION_BIT_GG_WEAPON_SHADOW folded into it by
// Surface::create(), so this is one wave-uniform SGPR bool per light — no FrameCB re-read.
inline float3 gg_weapon_shadow_P(in Surface surface)
{
	if (surface.IsWeaponShadow())
	{
		const float3 camPos = GetCamera().position;
		return camPos + (surface.P - camPos) / 3.0f;
	}
	return surface.P;
}

inline void light_directional(in ShaderEntity light, in Surface surface, inout Lighting lighting, in half shadow_mask = 1)
{
	if (shadow_mask <= 0.001)
		return; // shadow mask zero
	if ((light.layerMask & surface.layerMask) == 0)
		return; // layer mismatch

	surface.P = gg_weapon_shadow_P(surface); // GGMAX 2.14 — before gg_camera_distance and the cascade lookups

	half3 L = light.GetDirection();
	SurfaceToLight surface_to_light;
	surface_to_light.create(surface, L);

	if (!any(surface_to_light.NdotL_sss))
		return; // facing away from light
		
	half3 light_color = light.GetColor().rgb * shadow_mask;

	[branch]
	if (light.IsCastingShadow() && surface.IsReceiveShadow() && (GetFrame().options & OPTION_BIT_DISABLE_SHADOWMAPS) == 0)
	{
		if (GetFrame().options & OPTION_BIT_VOLUMETRICCLOUDS_CAST_SHADOW)
		{
			light_color *= shadow_2D_volumetricclouds(surface.P);
		}

#if defined(SHADOW_MASK_ENABLED) && !defined(TRANSPARENT)
		[branch]
		if ((GetFrame().options & OPTION_BIT_RAYTRACED_SHADOWS) == 0 || GetCamera().texture_rtshadow_index < 0 || (GetCamera().options & SHADERCAMERA_OPTION_USE_SHADOW_MASK) == 0)
#endif // SHADOW_MASK_ENABLED
		{
			// GGMAX 1.58: camera distance drives the feathered sampler's 8..1 tap count (DX11 parity)
			const float gg_camera_distance = length(GetCamera().position - surface.P);
			// Loop through cascades from closest (smallest) to furthest (largest)
			[loop]
			for (min16uint cascade = 0; cascade < light.GetShadowCascadeCount(); ++cascade)
			{
				// Project into shadow map space (no need to divide by .w because ortho projection!):
				const float4x4 cascade_projection = load_entitymatrix(light.GetMatrixIndex() + cascade);
				float3 shadow_pos = mul(cascade_projection, float4(surface.P, 1)).xyz;
				shadow_pos.z += GetFrame().gg_shadow_receiver_bias; // GGMAX 1.57: receiver tolerance (see ShaderInterop_Renderer.h)
				float3 shadow_uv = clipspace_to_uv(shadow_pos);

				// Determine if pixel is inside current cascade bounds and compute shadow if it is:
				[branch]
				if (is_saturated(shadow_uv))
				{
					const half3 shadow_box = half3(shadow_pos.xy, shadow_pos.z * 2 - 1);
					const half3 cascade_edgefactor = saturate(saturate(abs(shadow_box)) - 0.8) * 5.0; // fade will be on edge and inwards 10%
					const half cascade_fade = max3(cascade_edgefactor);
						
#ifdef CASCADE_DITHERING
					// If we are on cascade edge threshold and not the last cascade, then fallback to a larger cascade:
					[branch]
					if (cascade_fade > 0 && dither(surface.pixel + GetTemporalAASampleRotation()) < cascade_fade)
						continue;
						
					light_color *= shadow_2D(light, shadow_pos.z, shadow_uv.xy, cascade);
					break;
#else
					const half3 shadow_main = shadow_2D_feathered(light, shadow_pos.z, shadow_uv.xy, cascade, gg_camera_distance); // GGMAX 1.58
					
					// If we are on cascade edge threshold and not the last cascade, then fallback to a larger cascade:
					[branch]
					if (cascade_fade > 0 && cascade < light.GetShadowCascadeCount() - 1)
					{
						// Project into next shadow cascade (no need to divide by .w because ortho projection!):
						cascade += 1;
						shadow_pos = mul(load_entitymatrix(light.GetMatrixIndex() + cascade), float4(surface.P, 1)).xyz;
						shadow_pos.z += GetFrame().gg_shadow_receiver_bias; // GGMAX 1.57
						shadow_uv = clipspace_to_uv(shadow_pos);
						// GGMAX 1.59c: only blend when the pixel actually lies INSIDE the next cascade.
						// Concentric sun cascades always satisfy this (edge fade unchanged). Character
						// DEDICATED slots do not: slot i's Z-edge band would otherwise blend with slot
						// i+1 (a DIFFERENT character's map) at out-of-range UVs, and the border clamp
						// smears that foreign slot's edge texels across the band — the rectangular
						// ground artifacts near characters.
						[branch]
						if (is_saturated(shadow_uv))
						{
							const half3 shadow_fallback = shadow_2D_feathered(light, shadow_pos.z, shadow_uv.xy, cascade, gg_camera_distance); // GGMAX 1.58
							light_color *= lerp(shadow_main, shadow_fallback, cascade_fade);
						}
						else
						{
							light_color *= shadow_main;
						}
					}
					else
					{
						light_color *= shadow_main;
					}
					break;
#endif // CASCADE_DITHERING
				}
			}
		}
		
		if (!any(light_color))
			return; // light color lost after shadow
	}

	[branch]
	if (GetFrame().options & OPTION_BIT_REALISTIC_SKY)
	{
		light_color *= GetAtmosphericLightTransmittance(GetWeather().atmosphere, surface.P, L, texture_transmittancelut);
	}

	lighting.direct.diffuse = mad(light_color, BRDF_GetDiffuse(surface, surface_to_light), lighting.direct.diffuse);
	lighting.direct.specular = mad(light_color, BRDF_GetSpecular(surface, surface_to_light), lighting.direct.specular);

#ifdef LIGHTING_SCATTER
	const half scattering = ComputeScattering(saturate(dot(L, -surface.V)));
	lighting.indirect.specular += scattering * light_color * (1 - surface.extinction) * (1 - sqr(1 - saturate(1 - surface.N.y)));
#endif // LIGHTING_SCATTER
			
#ifndef WATER
	// On non-water surfaces there can be procedural caustic if it's under ocean:
	const ShaderOcean ocean = GetWeather().ocean;
	if (ocean.texture_displacementmap >= 0)
	{
		Texture2D displacementmap = bindless_textures[descriptor_index(ocean.texture_displacementmap)];
		float2 ocean_uv = surface.P.xz * ocean.patch_size_rcp;
		float3 displacement = displacementmap.SampleLevel(sampler_linear_wrap, ocean_uv, 0).xzy;
		float water_height = ocean.water_height + displacement.y;
		if (surface.P.y < water_height)
		{
			// Caustic UV is scaled independently of patch_size, so the seabed light-ripple size can
			// be tuned WITHOUT changing wave size (patch_size otherwise drives both). The
			// displacement/water_height lookup above deliberately keeps the unscaled ocean_uv.
			// ocean.caustic_scale == 1 reproduces stock behaviour exactly.
			half3 caustic = texture_caustics.SampleLevel(sampler_linear_mirror, ocean_uv * ocean.caustic_scale, 0).rgb;
			caustic *= sqr(saturate((water_height - surface.P.y) * 0.5)); // fade out at shoreline
			caustic *= light_color;
			lighting.indirect.diffuse += caustic;

			// fade out specular at depth, it looks weird when specular appears under ocean from wetmap
			half water_depth = water_height - surface.P.y;
			lighting.direct.specular *= saturate(exp(-water_depth * 10));
		}
	}
#endif // WATER
}

// GGMAX 2.07g: attenuation math promoted HALF -> FLOAT. fp16 overflows at 65504, so any
// light with range > 255.9 had range2 = +INF: dist2/range2 collapsed to 0, the falloff
// WINDOW vanished, and the light decayed as raw 1/d^2 that NEVER reaches zero. Its tiled-
// culling boundary (16px screen tiles) then cut a still-visible intensity/d^2 contribution
// = the user-reported screen-tile-aligned staircase divides on the floor (spot range 503
// showed it; point lights with range < 256 stayed windowed and clean - the misleading
// spot/point asymmetry). Beyond ~256u dist2 itself also overflowed (rsqrt(INF)=0 -> L=0),
// hard-killing big lights at a fixed 256u circle regardless of authored range.
inline half attenuation_pointlight(in float dist2, in float range, in float range2)
{
	// GGMAX 2.10: DX11 product falloff behind OPTION_BIT_GG_DX11_LIGHT_FALLOFF.
	// The DX11 fork shaded punctual lights as energy * (1 - d2/r2)^2 with NO inverse-square
	// term (WickedRepo lightingHF.hlsli:621). NO 1/PI correction belongs here: both engines
	// carry exactly one Lambert 1/PI on diffuse (DX11 inside BRDF_GetDiffuse, brdf.hlsli:449;
	// DX12 at ApplyLighting's direct.diffuse/PI) and both specular D terms carry their own,
	// so with this curve LightComponent::intensity is in DX11 "energy" units 1:1 and the
	// game passes the DX11 product's constant (30) straight through. The curve reaches
	// exactly zero at range, so the Forward+ tile cull boundary never truncates a visible
	// contribution (the 2.07g artifact class). Every consumer follows automatically:
	// surface lighting, volumetricLight_*, raytraceCS, ddgi, surfels, renderlightmap.
	[branch]
	if (GetFrame().options & OPTION_BIT_GG_DX11_LIGHT_FALLOFF)
	{
		float att = saturate(1 - dist2 / max(0.0001, range2));
		return (half)(att * att);
	}

	// GLTF recommendation: https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_lights_punctual#range-property
	//return saturate(1 - pow(dist / range, 4)) / dist2;

	// Removed pow(x, 4):
	float dist_per_range = dist2 / range2; // pow2 (note: range cannot be 0, in that case light is not uploaded to GPU, so here will not be zero-division)
	dist_per_range *= dist_per_range; // pow4
	return (half)(saturate(1 - dist_per_range) / max(0.0001, dist2));
}
inline void light_point(in ShaderEntity light, in Surface surface, inout Lighting lighting, in half shadow_mask = 1)
{
	if (shadow_mask <= 0.001)
		return; // shadow mask zero
	if ((light.layerMask & surface.layerMask) == 0)
		return; // layer mismatch

	surface.P = gg_weapon_shadow_P(surface); // GGMAX 2.14 — before Lunnormalized / shadow_cube

	float3 Lunnormalized = light.position - surface.P;
	const float3 LunnormalizedShadow = Lunnormalized;

#ifndef DISABLE_AREA_LIGHTS
	if (light.GetLength() > 0)
	{
		// Diffuse representative point on line:
		const float3 line_point = closest_point_on_segment(
			light.position - light.GetDirection() * light.GetLength() * 0.5,
			light.position + light.GetDirection() * light.GetLength() * 0.5,
			surface.P
		);
		Lunnormalized = line_point - surface.P;
	}
#endif // DISABLE_AREA_LIGHTS

	const float dist2 = dot(Lunnormalized, Lunnormalized); // GGMAX 2.07g: float — fp16 overflows past 255.9u
	const float range = light.GetRange();
	const float range2 = range * range;

	if (dist2 > range2)
		return; // outside range
		
	const half dist_rcp = rsqrt(dist2);
	half3 L = Lunnormalized * dist_rcp;

	SurfaceToLight surface_to_light;
	surface_to_light.create(surface, L);
		
	if (!any(surface_to_light.NdotL_sss))
		return; // facing away from light
		
	half3 light_color = light.GetColor().rgb * shadow_mask;

	[branch]
	if (light.IsCastingShadow() && surface.IsReceiveShadow())
	{
#if defined(SHADOW_MASK_ENABLED) && !defined(TRANSPARENT)
		[branch]
		if ((GetFrame().options & OPTION_BIT_RAYTRACED_SHADOWS) == 0 || GetCamera().texture_rtshadow_index < 0 || (GetCamera().options & SHADERCAMERA_OPTION_USE_SHADOW_MASK) == 0)
#endif // SHADOW_MASK_ENABLED
		{
			light_color *= shadow_cube(light, LunnormalizedShadow, surface.pixel);
		}
		
		if (!any(light_color))
			return; // light color lost after shadow
	}

	const uint maskTex = light.GetTextureIndex();
	[branch]
	if (maskTex > 0)
	{
		half4 mask = bindless_cubemaps_half4[descriptor_index(maskTex)].SampleLevel(sampler_linear_clamp, -LunnormalizedShadow, 0);
		light_color *= mask.rgb * mask.a;
	}
		
	light_color *= attenuation_pointlight(dist2, range, range2);

	lighting.direct.diffuse = mad(light_color, BRDF_GetDiffuse(surface, surface_to_light), lighting.direct.diffuse);

#ifndef DISABLE_AREA_LIGHTS
	if (light.GetLength() > 0)
	{
		// Specular representative point on line:
		float3 P0 = light.position - light.GetDirection() * light.GetLength() * 0.5;
		float3 P1 = light.position + light.GetDirection() * light.GetLength() * 0.5;
		float3 L0 = P0 - surface.P;
		float3 L1 = P1 - surface.P;
		float3 Ld = L1 - L0;
		float RdotLd = dot(surface.R, Ld);
		float t = dot(surface.R, L0) * RdotLd - dot(L0, Ld);
		t /= dot(Ld, Ld) - RdotLd * RdotLd;
		Lunnormalized = (L0 + saturate(t) * Ld);
	}
	else
	{
		Lunnormalized = light.position - surface.P;
	}
	if(light.GetRadius() > 0)
	{
		// Specular representative point on sphere:
		float3 centerToRay = mad(dot(Lunnormalized, surface.R), surface.R, -Lunnormalized);
		Lunnormalized = mad(centerToRay, saturate(light.GetRadius() / length(centerToRay)), Lunnormalized);
		// Energy conservation for radius:
		light_color /= max(1, sphere_volume(light.GetRadius()));
	}
	if (light.GetLength() > 0 || light.GetRadius() > 0)
	{
		L = normalize(Lunnormalized);
		surface_to_light.create(surface, L); // recompute all surface-light vectors
	}
#endif // DISABLE_AREA_LIGHTS

	lighting.direct.specular = mad(light_color, BRDF_GetSpecular(surface, surface_to_light), lighting.direct.specular);
				
#ifdef LIGHTING_SCATTER
	const half scattering = ComputeScattering(saturate(dot(L, -surface.V)));
	lighting.indirect.specular += scattering * light_color * (1 - surface.extinction) * (1 - sqr(1 - saturate(1 - surface.N.y)));
#endif // LIGHTING_SCATTER
}

inline half attenuation_spotlight(in float dist2, in float range, in float range2, in half spot_factor, in half angle_scale, in half angle_offset)
{
	half attenuation = attenuation_pointlight(dist2, range, range2);
	half angularAttenuation = saturate(mad(spot_factor, angle_scale, angle_offset));

	// GGMAX 2.10: DX11 cone falloff = saturate(1 - (1-SpotFactor)/(1-coneCos)) — LINEAR from
	// the cone edge to dead center, never squared (WickedRepo lightingHF.hlsli:700). GG never
	// sets innerConeAngle (default 0 -> cos=1), so the packed angle_scale/angle_offset already
	// reduce mad(spot_factor, scale, offset) to exactly that DX11 term; only the modern
	// squaring must be skipped. The distance curve comes from attenuation_pointlight above.
	[branch]
	if (GetFrame().options & OPTION_BIT_GG_DX11_LIGHT_FALLOFF)
	{
		return attenuation * angularAttenuation;
	}

	angularAttenuation *= angularAttenuation;
	attenuation *= angularAttenuation;
	return attenuation;
}
inline void light_spot(in ShaderEntity light, in Surface surface, inout Lighting lighting, in half shadow_mask = 1)
{
	if (shadow_mask <= 0.001)
		return; // shadow mask zero
	if ((light.layerMask & surface.layerMask) == 0)
		return; // layer mismatch

	surface.P = gg_weapon_shadow_P(surface); // GGMAX 2.14 — before Lunnormalized / the spot shadow lookup

	float3 Lunnormalized = light.position - surface.P;
	const float dist2 = dot(Lunnormalized, Lunnormalized); // GGMAX 2.07g: float — fp16 overflows past 255.9u
	const float range = light.GetRange();
	const float range2 = range * range;
	
	if (dist2 > range2)
		return; // outside range
		
	const half dist_rcp = rsqrt(dist2);
	half3 L = Lunnormalized * dist_rcp;

	SurfaceToLight surface_to_light;
	surface_to_light.create(surface, L);
		
	if (!any(surface_to_light.NdotL_sss))
		return; // facing away from light
			
	const half spot_factor = dot(L, light.GetDirection());
	const half spot_cutoff = light.GetConeAngleCos();
			
	if (spot_factor < spot_cutoff)
		return; // outside spotlight cone

	half3 light_color = light.GetColor().rgb * shadow_mask;
	
	[branch]
	if (light.IsCastingShadow() && surface.IsReceiveShadow())
	{
#if defined(SHADOW_MASK_ENABLED) && !defined(TRANSPARENT)
		[branch]
		if ((GetFrame().options & OPTION_BIT_RAYTRACED_SHADOWS) == 0 || GetCamera().texture_rtshadow_index < 0 || (GetCamera().options & SHADERCAMERA_OPTION_USE_SHADOW_MASK) == 0)
#endif // SHADOW_MASK_ENABLED
		{
			float4 shadow_pos = mul(load_entitymatrix(light.GetMatrixIndex() + 0), float4(surface.P, 1));
			shadow_pos.xyz /= shadow_pos.w;
			float2 shadow_uv = clipspace_to_uv(shadow_pos.xy);
			[branch]
			if (is_saturated(shadow_uv))
			{
				// GGMAX 2.07e: SPOT uses the DX11 feathered compare like the sun does (1.58 ported
				// it directional-only). The stock hard SampleCmp has ZERO receiver-side tolerance on
				// D32 (raster bias units are float-exponent-scaled = negligible), so directly-lit
				// surfaces sat on a knife-edge depth compare and camera pose jitter flipped pixels
				// IN/OUT of self-shadow — the mouselook "shadow flicker". DX11's shadowCascadeSpot
				// ran the same graded-tolerance gather (scaleFactor 65536, spot = cascade 0) and
				// never acne'd. POINT never had this problem: shadow_cube carries its own 0.989
				// distance cushion (the user-observed spot/point asymmetry).
				// GGMAX 2.07f: camera_distance -1 = FIXED 8 taps (DX11 spot parity — the sun's
				// distance-stepped tap count put camera-anchored brightness divides in spot shadows).
				light_color *= shadow_2D_feathered(light, shadow_pos.z, shadow_uv.xy, 0, -1.0);
			}
		}

		if (!any(light_color))
			return; // light color lost after shadow
	}

	const uint maskTex = light.GetTextureIndex();
	[branch]
	if (maskTex > 0)
	{
		float4 shadow_pos = mul(load_entitymatrix(light.GetMatrixIndex() + 0), float4(surface.P, 1));
		shadow_pos.xyz /= shadow_pos.w;
		float2 shadow_uv = clipspace_to_uv(shadow_pos.xy);
		half4 mask = bindless_textures_half4[descriptor_index(maskTex)].SampleLevel(sampler_linear_clamp, shadow_uv, 0);
		light_color *= mask.rgb * mask.a;
	}
	
	light_color *= attenuation_spotlight(dist2, range, range2, spot_factor, light.GetAngleScale(), light.GetAngleOffset());
		
	lighting.direct.diffuse = mad(light_color, BRDF_GetDiffuse(surface, surface_to_light), lighting.direct.diffuse);

#ifndef DISABLE_AREA_LIGHTS
	if (light.GetRadius() > 0)
	{
		// Specular representative point on sphere:
		Lunnormalized = light.position - surface.P;
		float3 centerToRay = mad(dot(Lunnormalized, surface.R), surface.R, -Lunnormalized);
		Lunnormalized = mad(centerToRay, saturate(light.GetRadius() / length(centerToRay)), Lunnormalized);
		L = normalize(Lunnormalized);
		surface_to_light.create(surface, L); // recompute all surface-light vectors
		// Energy conservation for radius:
		light_color /= max(1, sphere_volume(light.GetRadius()));
	}
#endif // DISABLE_AREA_LIGHTS

	lighting.direct.specular = mad(light_color, BRDF_GetSpecular(surface, surface_to_light), lighting.direct.specular);
					
#ifdef LIGHTING_SCATTER
	const half scattering = ComputeScattering(saturate(dot(L, -surface.V)));
	lighting.indirect.specular += scattering * light_color * (1 - surface.extinction) * (1 - sqr(1 - saturate(1 - surface.N.y)));
#endif // LIGHTING_SCATTER
}

inline void light_rect(in ShaderEntity light, in Surface surface, inout Lighting lighting, in half shadow_mask = 1)
{
#ifndef DISABLE_AREA_LIGHTS
	if (shadow_mask <= 0.001)
		return; // shadow mask zero
	if ((light.layerMask & surface.layerMask) == 0)
		return; // layer mismatch
	
	const half4 quaternion = light.GetQuaternion();
	const half3 right = rotate_vector(half3(1, 0, 0), quaternion);
	const half3 up = rotate_vector(half3(0, 1, 0), quaternion);
	const half3 forward = cross(up, right);
	const half light_length = max(0.01, light.GetLength());
	const half light_height = max(0.01, light.GetHeight());
	const half light_area = light_length * light_height;
	const float3 p0 = light.position - right * light_length * 0.5 + up * light_height * 0.5;
	const float3 p1 = light.position + right * light_length * 0.5 + up * light_height * 0.5;
	const float3 p2 = light.position + right * light_length * 0.5 - up * light_height * 0.5;
	const float3 p3 = light.position - right * light_length * 0.5 - up * light_height * 0.5;
	
	if (dot(surface.P - light.position, forward) <= 0)
		return; // behind light

	// Determine closest point on rectangle to surface position:
	const float3 closest_point_on_plane_to_surface = point_on_plane(surface.P, light.position, forward);
	const float3 closest_vector_on_plane = closest_point_on_plane_to_surface - light.position;
	const float2 plane_point = float2(dot(closest_vector_on_plane, right), dot(closest_vector_on_plane, up));
	const float2 nearest_point = float2(clamp(plane_point.x, -light_length * 0.5, light_length * 0.5), clamp(plane_point.y, -light_height * 0.5, light_height * 0.5));
	const float3 rectangle_point = light.position + nearest_point.x * right + nearest_point.y * up;
		
	float3 Lunnormalized = rectangle_point - surface.P;

	const float dist2 = dot(Lunnormalized, Lunnormalized); // GGMAX 2.07g: float — fp16 overflows past 255.9u
	const float range = light.GetRange();
	const float range2 = range * range;

	if (dist2 > range2)
		return; // outside range
		
	const half dist_rcp = rsqrt(dist2);
	half3 L = Lunnormalized * dist_rcp;

	SurfaceToLight surface_to_light;
	surface_to_light.create(surface, L);
	
	// Solid angle based on the Frostbite presentation: Moving Frostbite to Physically Based Rendering by Sebastien Lagarde, Charles de Rousiers, Siggraph 2014
	//	https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/course-notes-moving-frostbite-to-pbr-v2.pdf
	const float3 v0 = normalize(p0 - surface.P);
	const float3 v1 = normalize(p1 - surface.P);
	const float3 v2 = normalize(p2 - surface.P);
	const float3 v3 = normalize(p3 - surface.P);
	const float3 n0 = normalize(cross(v0, v1));
	const float3 n1 = normalize(cross(v1, v2));
	const float3 n2 = normalize(cross(v2, v3));
	const float3 n3 = normalize(cross(v3, v0));
	const float g0 = acosFast(dot(-n0, n1));
	const float g1 = acosFast(dot(-n1, n2));
	const float g2 = acosFast(dot(-n2, n3));
	const float g3 = acosFast(dot(-n3, n0));
	const float solid_angle = saturate(g0 + g1 + g2 + g3 - 2 * PI);
	
	surface_to_light.NdotL = solid_angle * 0.2 * (
		saturate(dot(v0, surface.N)) +
		saturate(dot(v1, surface.N)) +
		saturate(dot(v2, surface.N)) +
		saturate(dot(v3, surface.N)) +
		surface_to_light.NdotL
	);
	surface_to_light.NdotL_sss = surface_to_light.NdotL;
		
	if (!any(surface_to_light.NdotL_sss))
		return; // facing away from light
		
	half3 light_color = light.GetColor().rgb * shadow_mask;
	
	[branch]
	if (light.IsCastingShadow() && surface.IsReceiveShadow())
	{
#if defined(SHADOW_MASK_ENABLED) && !defined(TRANSPARENT)
		[branch]
		if ((GetFrame().options & OPTION_BIT_RAYTRACED_SHADOWS) == 0 || GetCamera().texture_rtshadow_index < 0 || (GetCamera().options & SHADERCAMERA_OPTION_USE_SHADOW_MASK) == 0)
#endif // SHADOW_MASK_ENABLED
		{
			float4 shadow_pos = mul(load_entitymatrix(light.GetMatrixIndex() + 0), float4(surface.P, 1));
			shadow_pos.xyz /= shadow_pos.w;
			float2 shadow_uv = clipspace_to_uv(shadow_pos.xy);
			[branch]
			if (is_saturated(shadow_uv))
			{
				// GGMAX 2.07e: same feathered receiver compare as SPOT (see light_spot) — rect
				// lights share the perspective 2D shadow path and the same acne exposure.
				// GGMAX 2.07f: fixed taps (camera_distance -1), same reason as light_spot.
				light_color *= shadow_2D_feathered(light, shadow_pos.z, shadow_uv.xy, 0, -1.0);
			}
		}

		if (!any(light_color))
			return; // light color lost after shadow
	}

	light_color *= attenuation_pointlight(dist2, range, range2); // dist2 is the closest point on rectangle, so it will not be a falloff from light center, but as if a point light is placed on the closest rectangle point
	
	half3 light_color_diffuse = light_color * light_area * PI; // I increase the light color by the surface area, because I want larger lights to illuminate more.
	
	half3 light_color_specular = light_color;

	// Intersects the plane of the rectangle with reflection ray, then computes closest point on rectangle, source: https://alextardif.com/arealights.html
	const float3 intersectPoint = surface.P + surface.R * trace_plane(surface.P, surface.R, light.position, forward);
	const float3 intersectionVector = intersectPoint - light.position;
	const float2 intersectPlanePoint = float2(dot(intersectionVector,right), dot(intersectionVector,up));
	const float2 nearest2DPoint = float2(clamp(intersectPlanePoint.x, -light_length * 0.5, light_length * 0.5), clamp(intersectPlanePoint.y, -light_height * 0.5, light_height * 0.5));
	const float3 specular_rect = light.position + nearest2DPoint.x * right + nearest2DPoint.y * up;

	const uint maskTex = light.GetTextureIndex();
	[branch]
	if (maskTex > 0)
	{
		Texture2D<half4> tex = bindless_textures_half4[descriptor_index(maskTex)];
		uint2 dim;
		uint mipcount;
		tex.GetDimensions(0, dim.x, dim.y, mipcount);
		
		float4 shadow_pos = mul(load_entitymatrix(light.GetMatrixIndex() + 0), float4(surface.P, 1));
		shadow_pos.xyz /= shadow_pos.w;
		float2 diffuse_uv = clipspace_to_uv(shadow_pos.xy);
		half4 diffuse_mask = tex.SampleLevel(sampler_linear_clamp, diffuse_uv, mipcount - 2);
		light_color_diffuse *= diffuse_mask.rgb * diffuse_mask.a;

		float2 specular_uv = clipspace_to_uv(nearest2DPoint / float2(light_length * 0.5, light_height * 0.5));
		half4 specular_mask = tex.SampleLevel(sampler_linear_clamp, specular_uv, (1 - sqr(1 - saturate(surface.roughness))) * mipcount);
		light_color_specular *= specular_mask.rgb * specular_mask.a;
	}
	
	lighting.direct.diffuse = mad(light_color_diffuse, BRDF_GetDiffuse(surface, surface_to_light), lighting.direct.diffuse);
	
	Lunnormalized = specular_rect - surface.P;
	L = normalize(Lunnormalized);
	surface_to_light.create(surface, L); // recompute all surface-light vectors
	lighting.direct.specular = mad(light_color_specular, BRDF_GetSpecular(surface, surface_to_light), lighting.direct.specular);
				
#ifdef LIGHTING_SCATTER
	const half scattering = ComputeScattering(saturate(dot(L, -surface.V)));
	lighting.indirect.specular += scattering * light_color * (1 - surface.extinction) * (1 - sqr(1 - saturate(1 - surface.N.y)));
#endif // LIGHTING_SCATTER

#endif // DISABLE_AREA_LIGHTS
}

// ENVIRONMENT MAPS


inline half3 GetAmbient(in float3 N)
{
	half3 ambient;

#ifdef ENVMAPRENDERING

	// Set realistic_sky_stationary to true so we capture ambient at float3(0.0, 0.0, 0.0), similar to the standard sky to avoid flickering and weird behavior
	ambient = lerp(
		GetDynamicSkyColor(float3(0, -1, 0), false, false, true),
		GetDynamicSkyColor(float3(0, 1, 0), false, false, true),
		saturate(N.y * 0.5 + 0.5));

#else

	[branch]
	if (GetScene().globalprobe >= 0)
	{
		TextureCube<half4> cubemap = bindless_cubemaps_half4[descriptor_index(GetScene().globalprobe)];
		uint2 dim;
		uint mipcount;
		cubemap.GetDimensions(0, dim.x, dim.y, mipcount);
		ambient = cubemap.SampleLevel(sampler_linear_clamp, N, mipcount).rgb;
	}
	
#endif // ENVMAPRENDERING

#ifndef NO_FLAT_AMBIENT
	// This is not entirely correct if we have probes, because it shouldn't be added twice.
	//	However, it is not correct if we leave it out from probes, because if we render a scene
	//	with dark sky but ambient, we still want some visible result.
	ambient += GetAmbientColor();
#endif // NO_FLAT_AMBIENT

	return ambient;
}

// surface:				surface descriptor
// MIP:					mip level to sample
// return:				color of the environment color (rgb)
inline half3 EnvironmentReflection_Global(in Surface surface)
{
	half3 envColor;

#ifdef ENVMAPRENDERING

	// There is no access to envmaps, so approximate sky color:
	// Set realistic_sky_stationary to true so we capture environment at float3(0.0, 0.0, 0.0), similar to the standard sky to avoid flickering and weird behavior
	float3 skycolor_real = GetDynamicSkyColor(surface.R, false, false, true); // false: disable sun disk and clouds
	float3 skycolor_rough = lerp(
		GetDynamicSkyColor(float3(0, -1, 0), false, false, true),
		GetDynamicSkyColor(float3(0, 1, 0), false, false, true),
		saturate(surface.R.y * 0.5 + 0.5));

	envColor = lerp(skycolor_real, skycolor_rough, surface.roughness) * surface.F;

#else
	
	[branch]
	if (GetScene().globalprobe < 0)
		return 0;
	
	TextureCube<half4> cubemap = bindless_cubemaps_half4[descriptor_index(GetScene().globalprobe)];
	uint2 dim;
	uint mipcount;
	cubemap.GetDimensions(0, dim.x, dim.y, mipcount);
	half mipcount16f = half(mipcount);

	half MIP = surface.roughness * mipcount16f;
	envColor = cubemap.SampleLevel(sampler_linear_clamp, surface.R, MIP).rgb * surface.F;

#ifdef SHEEN
	envColor *= surface.sheen.albedoScaling;
	MIP = surface.sheen.roughness * mipcount16f;
	envColor += cubemap.SampleLevel(sampler_linear_clamp, surface.R, MIP).rgb * surface.sheen.color * surface.sheen.DFG;
#endif // SHEEN

#ifdef CLEARCOAT
	envColor *= 1 - surface.clearcoat.F;
	MIP = surface.clearcoat.roughness * mipcount16f;
	envColor += cubemap.SampleLevel(sampler_linear_clamp, surface.clearcoat.R, MIP).rgb * surface.clearcoat.F;
#endif // CLEARCOAT

	// GGMAX 1.55: global env-probe brightness (1 = stock) — restores the DX11 slider
	envColor *= (half)GetScene().gg_envprobe_brightness;

#endif // ENVMAPRENDERING

	return envColor;
}

// surface:				surface descriptor
// probe :				the shader entity holding properties
// probeProjection:		the inverse OBB transform matrix
// clipSpacePos:		world space pixel position transformed into OBB space by probeProjection matrix
// MIP:					mip level to sample
// return:				color of the environment map (rgb), blend factor of the environment map (a)
inline half4 EnvironmentReflection_Local(in TextureCube<half4> cubemap, in Surface surface, in ShaderEntity probe, in float4x4 probeProjection, in half3 clipSpacePos)
{
	if ((probe.layerMask & surface.layerMask) == 0)
		return 0; // early exit: layer mismatch

	// GGMAX 2.89 (#157): ISOLATION — drop local probes so everything falls through to the
	// global cube. "Local" here means an OBB small enough that its parallax stays inside fp16
	// (half-extent below 37820); the global probe's 50000 box is deliberately not dropped.
	// Returning 0 leaves the accumulation alpha untouched, so shadingHF's
	// "if (envmapAccumulation.a < 0.99)" fallback supplies EnvironmentReflection_Global.
	if (GetScene().gg_probeonlyglobal != 0 && length(probeProjection[0].xyz) * 37820.0 >= 1.0)
		return 0;

	// Perform parallax correction of reflection ray (R) into OBB.
	//
	// GGMAX 2.89 (#157) — THE CIRCLES. Stock ran this in half (min16float = real fp16 on this
	// hardware, max 65504). "Distance" is the ray's exit distance in WORLD units, so a probe
	// whose OBB half-extent exceeds 65504/sqrt(3) ~= 37,820 units overflows to +INF over most
	// of the direction sphere and the sampled direction becomes garbage.
	// NOTE the shipping default is mode 3, which skips this block entirely for level-sized
	// boxes — but the precision fix below still matters for any probe between a room and the
	// 37,820 ceiling, and mode 1 keeps it available.
	// GG's globalEnvProbe box is 50,000 units (GGTerrain_part0: globalrange), which leaves
	// exactly six ~40 degree caps around +-X/+-Y/+-Z finite and everything between them INF —
	// six discs of correct reflection separated by bands of rubbish. That is the "circles on
	// every reflective surface" defect, and it is why LOCAL probes always looked clean: their
	// boxes are a couple of units to a few hundred, nowhere near the fp16 ceiling.
	// Same bug class as 2.07g (half range2 overflowed past range 255.9), same file, same fix.
	float3 R_parallaxCorrected;
	const int gg_pp = GetScene().gg_probeparallax;
	[branch]
	if (gg_pp == 0)
	{
		// Stock half math, kept so the defect can be reproduced for a same-session A/B.
		half3 RayLS = mul((half3x3)probeProjection, surface.R);
		half3 FirstPlaneIntersect = (1 - clipSpacePos) / RayLS;
		half3 SecondPlaneIntersect = (-1 - clipSpacePos) / RayLS;
		half3 FurthestPlane = max(FirstPlaneIntersect, SecondPlaneIntersect);
		half Distance = min(FurthestPlane.x, min(FurthestPlane.y, FurthestPlane.z));
		R_parallaxCorrected = surface.P - probe.position + surface.R * Distance;
	}
	else if (gg_pp == 3 && length(probeProjection[0].xyz) * 37820.0 < 1.0)
	{
		// Mode 3 — THE DEFAULT (Lee's call, 08-18 after seeing the A/B). A box this size is not
		// a room, it is "the whole level" — GG's globalEnvProbe. Parallax-correcting against it
		// is not modelling anything physical: it drags the reflected horizon by however far the
		// surface sits from the probe centre (the map origin), which makes the look POSITION
		// DEPENDENT across a level. DX11 never did it. Read the raw reflection vector instead,
		// which is what EnvironmentReflection_Global does for the same cube.
		// Mode 1 keeps the parallax and is the precision-only change, retained for comparison.
		// The half-extent is derived from the inverse matrix (its row length is 1/half-extent)
		// rather than from probe.GetRange(): GetRange() is ALSO fp16 (SetRange packs through
		// XMConvertFloatToHalf), so the global probe's authored range of 100000 comes back as
		// +INF. Same overflow class as the parallax bug — worth knowing before trusting a
		// probe range in any shader comparison.
		R_parallaxCorrected = (float3)surface.R;
	}
	else
	{
		float3 RayLS_f = mul((float3x3)probeProjection, (float3)surface.R);
		float3 csp_f = (float3)clipSpacePos;
		float3 FirstPlaneIntersect_f = (1 - csp_f) / RayLS_f;
		float3 SecondPlaneIntersect_f = (-1 - csp_f) / RayLS_f;
		float3 FurthestPlane_f = max(FirstPlaneIntersect_f, SecondPlaneIntersect_f);
		float Distance_f = min(FurthestPlane_f.x, min(FurthestPlane_f.y, FurthestPlane_f.z));
		R_parallaxCorrected = surface.P - probe.position + (float3)surface.R * Distance_f;

		[branch]
		if (gg_pp == 2)
		{
			// DIAGNOSTIC: would the stock fp16 path have overflowed at this pixel? Paint it.
			// The magenta this produces IS the predicted overflow map — if it lands exactly on
			// the bands between the circles, the root cause is proven visually, not argued.
			if (Distance_f >= 65504.0 || !isfinite(Distance_f))
				return half4(6, 0, 6, 1); // magenta, alpha 1 so it fully replaces the probe result
		}
	}

	uint2 dim;
	uint mipcount;
	cubemap.GetDimensions(0, dim.x, dim.y, mipcount);
	half mipcount16f = half(mipcount);

	// Sample cubemap texture:
	half MIP = surface.roughness * mipcount16f;
	half3 envColor = cubemap.SampleLevel(sampler_linear_clamp, R_parallaxCorrected, MIP).rgb * surface.F;

#ifdef SHEEN
	envColor *= surface.sheen.albedoScaling;
	MIP = surface.sheen.roughness * mipcount16f;
	envColor += cubemap.SampleLevel(sampler_linear_clamp, R_parallaxCorrected, MIP).rgb * surface.sheen.color * surface.sheen.DFG;
#endif // SHEEN

#ifdef CLEARCOAT
	// GGMAX 2.89 (#157): float precision here too — same overflow, same reason (see above).
	{
		float3 RayLS_cc = mul((float3x3)probeProjection, (float3)surface.clearcoat.R);
		float3 csp_cc = (float3)clipSpacePos;
		float3 FirstPlaneIntersect_cc = (1 - csp_cc) / RayLS_cc;
		float3 SecondPlaneIntersect_cc = (-1 - csp_cc) / RayLS_cc;
		float3 FurthestPlane_cc = max(FirstPlaneIntersect_cc, SecondPlaneIntersect_cc);
		float Distance_cc = min(FurthestPlane_cc.x, min(FurthestPlane_cc.y, FurthestPlane_cc.z));
		R_parallaxCorrected = surface.P - probe.position + (float3)surface.clearcoat.R * Distance_cc;
	}

	envColor *= 1 - surface.clearcoat.F;
	MIP = surface.clearcoat.roughness * mipcount16f;
	envColor += cubemap.SampleLevel(sampler_linear_clamp, R_parallaxCorrected, MIP).rgb * surface.clearcoat.F;
#endif // CLEARCOAT

	// blend out if close to any cube edge:
	half edgeBlend = 1 - pow8(saturate(max(abs(clipSpacePos.x), max(abs(clipSpacePos.y), abs(clipSpacePos.z)))));

	return half4(envColor, edgeBlend);
}



// VOXEL RADIANCE

inline void VoxelGI(inout Surface surface, inout Lighting lighting)
{
	[branch]
	if (GetFrame().vxgi.resolution != 0 && GetFrame().vxgi.texture_radiance >= 0)
	{
		Texture3D<half4> voxels = bindless_textures3D_half4[descriptor_index(GetFrame().vxgi.texture_radiance)];

		// diffuse:
		half4 trace = ConeTraceDiffuse(voxels, surface.P, surface.N);
		lighting.indirect.diffuse = mad(lighting.indirect.diffuse, 1 - trace.a, trace.rgb);

		// specular:
		[branch]
		if (GetFrame().options & OPTION_BIT_VXGI_REFLECTIONS_ENABLED)
		{
			half roughnessBRDF = sqr(clamp(surface.roughness, min_roughness, 1));
			half4 trace = ConeTraceSpecular(voxels, surface.P, surface.N, surface.V, roughnessBRDF, surface.pixel);
			lighting.indirect.specular = mad(lighting.indirect.specular, 1 - trace.a, trace.rgb * surface.F);
		}
	}
}

#endif // WI_LIGHTING_HF
