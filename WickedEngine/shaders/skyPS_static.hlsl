#include "objectHF.hlsli"
#include "skyHF.hlsli"

float4 main(float4 pos : SV_POSITION, float2 clipspace : TEXCOORD) : SV_TARGET
{
	float4 unprojected = mul(GetCamera().inverse_view_projection, float4(clipspace, 0.0f, 1.0f));
	unprojected.xyz /= unprojected.w;

	const float3 V = normalize(unprojected.xyz - GetCamera().position);

	float4 color = float4(GetStaticSkyColor(V), 1);

	// GGMAX 1.65: DX11 parity — the skybox is tinted toward the user fog color by Fog
	// Opacity (skyPS_static DX11 line 26), keeping the horizon seam coherent with fogged
	// geometry. No-op at Fog Opacity 0.
	color.rgb = lerp(color.rgb, GetHorizonColor(), saturate(GetWeather().gg_fog_opacity));

	float4 pos2DPrev = mul(GetCamera().previous_view_projection, float4(unprojected.xyz, 1));
	float2 velocity = ((pos2DPrev.xy / pos2DPrev.w - GetCamera().temporalaa_jitter_prev) - (clipspace - GetCamera().temporalaa_jitter)) * float2(0.5f, -0.5f);

	color = saturateMediump(color);
	return color;
}

