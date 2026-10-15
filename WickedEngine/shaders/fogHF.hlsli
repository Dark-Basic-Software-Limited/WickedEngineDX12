#ifndef WI_FOG_HF
#define WI_FOG_HF
#include "globals.hlsli"
#include "skyAtmosphere.hlsli"

// [-0.999; 0.999] Describes how the lighting is destributed across sky
#define FOG_INSCATTERING_PHASE_G 0.6

// Exponential height fog based on: https://www.iquilezles.org/www/articles/fog/fog.htm
// Non constant density function
//	distance	: sample to point distance
//	O			: sample position
//	V			: sample to point vector
inline half GetFogAmount(float distance, float3 O, float3 V)
{
	ShaderFog fog = GetWeather().fog;
	
	float startDistanceFalloff = saturate((distance - fog.start) / fog.start);
	
	if (GetFrame().options & OPTION_BIT_HEIGHT_FOG)
	{
		float fogFalloffScale = rcp(max(0.01, fog.height_end - fog.height_start));

		// solve for x, e^(-h * x) = 0.001
		// x = 6.907755 * h^-1
		float fogFalloff = 6.907755 * fogFalloffScale;
		
		float originHeight = O.y;
		float Z = V.y;
		float effectiveZ = max(abs(Z), 0.001);

		float endLineHeight = mad(distance, Z, originHeight); // Isolated vector equation for y
		float minLineHeight = min(originHeight, endLineHeight);
		float heightLineFalloff = max(minLineHeight - fog.height_start, 0);
		
		float baseHeightFogDistance = clamp((fog.height_start - minLineHeight) / effectiveZ, 0, distance);
		float exponentialFogDistance = distance - baseHeightFogDistance; // Exclude distance below base height
		float exponentialHeightLineIntegral = exp(-heightLineFalloff * fogFalloff) * (1.0 - exp(-exponentialFogDistance * effectiveZ * fogFalloff)) / (effectiveZ * fogFalloff);
		
		float opticalDepthHeightFog = fog.density * startDistanceFalloff * (baseHeightFogDistance + exponentialHeightLineIntegral);
		float transmittanceHeightFog = exp(-opticalDepthHeightFog);
		
		float fogAmount = transmittanceHeightFog;
		return 1.0 - fogAmount;
	}
	else
	{
		// Height fog algorithm (above) reduced with infinity start and end heights:

		// GGMAX 3.114: DX11 GG's fog curve - optical depth over the distance PAST fog.start, which is
		// what DX11's objectHF ApplyFog and GG's ApplyFogCustom both did (exp((d - near) * 4 / (near
		// - far)), and GG stores density = 4 / (far - near)). The stock curve integrates from the CAMERA
		// once past 2 x start, so with a non-zero start every distance was over-fogged: at Lee's
		// DX12TESTLEVEL (start 40000) the restored horizon mountain sat at 42% fog against DX11's 26%
		// and read as a pale silhouette. ★ 40000 is the DEFAULT fog start of a GameGuru level
		// (visuals_resetvalues), so this changes most levels' distant haze - toward DX11. Identical only
		// at start 0. OPTION_BIT_GG_DX11_FOG_CURVE (setup.ini fogdx11curve=0) restores the stock line.
		[branch] if (GetFrame().options & OPTION_BIT_GG_DX11_FOG_CURVE)
		{
			return 1.0 - exp(-fog.density * max(0, distance - fog.start));
		}
		float opticalDepthHeightFog = fog.density * startDistanceFalloff * distance;
		float transmittanceHeightFog = exp(-opticalDepthHeightFog);
		
		float fogAmount = transmittanceHeightFog;
		return 1.0 - fogAmount;
	}
}

// GGMAX 1.65: DX11-parity fog color model (replaces the stock flat-average + inscatter fog).
//
// DX11 reference (GGCommonFunctions.hlsli ApplyFogCustom / objectHF ApplyFog):
//   realistic sky : fogColor = GetDynamicSkyColor(horizon-flattened view dir, sun off,
//                   stationary) — the sky's ACTUAL color at the horizon behind this pixel,
//                   then lerp(fogColor, userFogRGB, FogOpacity). Distant terrain hazes into
//                   the sky it stands against; silhouettes stay readable. The stock code's
//                   flat skyluminance-LUT average (one color for the whole screen) made
//                   distant mountains melt into the sky uniformly.
//   other skies   : fogColor = the surface's OWN color lerped toward userFogRGB by
//                   FogOpacity — i.e. fog is a pure opt-in recolor, a NO-OP at FogOpacity 0.
//                   lerp(C, lerp(C, F, o), a) == lerp(C, F, a*o), so this is encoded as
//                   alpha *= FogOpacity and every ApplyFog call site inherits it. The stock
//                   code fogged toward weather.horizon at FULL strength, which painted the
//                   None-mode Horizon/Fog color over terrain in Sky Box mode.
//   The stock sun HgPhase inscatter add is dropped in both branches: DX11 has no such term
//   (the skyview LUT sample already carries the azimuthal mie glow toward the sun).
inline half4 GetFog(float distance, float3 O, float3 V)
{
	half3 fogColor = 0;
	half fogOpacityScale = 1;

	if ((GetFrame().options & OPTION_BIT_REALISTIC_SKY) && (GetFrame().options & OPTION_BIT_OVERRIDE_FOG_COLOR) == 0)
	{
		// Sample the realistic sky at the horizon in this pixel's view direction (matches
		// the LUT path of AccurateAtmosphericScattering with stationary origin, sun off):
		AtmosphereParameters atmosphere = GetWeather().atmosphere;
		float2 xz = normalize(V.xz + 1e-5f) * 0.995f;
		float3 dir = normalize(float3(xz.x, 0.1f, xz.y));
		float3 worldPosition = GetCameraPlanetPos(atmosphere, float3(0.00001, 0.00001, 0.00001));
		float viewHeight = length(worldPosition);
		float3 upVector = normalize(worldPosition);
		float viewZenithCosAngle = dot(dir, upVector);
		float3 sunDirection = GetSunDirection();
		float3 sideVector = normalize(cross(upVector, dir));
		float3 forwardVector = normalize(cross(sideVector, upVector));
		float2 lightOnPlane = normalize(float2(dot(sunDirection, forwardVector), dot(sunDirection, sideVector)));
		bool intersectGround = RaySphereIntersectNearest(worldPosition, dir, float3(0, 0, 0), atmosphere.bottomRadius) >= 0.0;
		float2 uv;
		SkyViewLutParamsToUv(atmosphere, intersectGround, viewZenithCosAngle, lightOnPlane.x, viewHeight, uv);
		fogColor = texture_skyviewlut.SampleLevel(sampler_linear_clamp, uv, 0).rgb;
		fogColor *= GetWeather().sky_exposure;

		// DX11: user fog color mixes over the sky sample proportionally to Fog Opacity
		fogColor = lerp(fogColor, GetHorizonColor(), saturate(GetWeather().gg_fog_opacity));
	}
	else
	{
		fogColor = GetHorizonColor();
		fogOpacityScale = saturate(GetWeather().gg_fog_opacity);
	}

	return half4(fogColor, GetFogAmount(distance, O, V) * fogOpacityScale);
}


#endif // WI_FOG_HF
