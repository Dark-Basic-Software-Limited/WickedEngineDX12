#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"
// This will cut out bright parts (>1) and also downsample 4x

PUSHCONSTANT(bloom, Bloom);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const float2 uv = DTid.xy + 0.5;

	half3 color = 0;

	float exposure = bloom.exposure;

	[branch]
	if (bloom.buffer_input_luminance >= 0)
	{
		exposure *= bindless_buffers[descriptor_index(bloom.buffer_input_luminance)].Load<float>(LUMINANCE_BUFFER_OFFSET_EXPOSURE);
	}

	Texture2D<half4> texture_input = bindless_textures_half4[descriptor_index(bloom.texture_input)];

	[branch]
	if (bloom.before_exposure)
	{
		// as GameGuru MAX's DX11 bloom: each sample capped at 30 and the threshold taken on the scene's own values, the
		// exposure applied to what is left
		color += min(texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(-0.5, -0.5)) * bloom.resolution_rcp, 0).rgb, 30);
		color += min(texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(0.5, -0.5)) * bloom.resolution_rcp, 0).rgb, 30);
		color += min(texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(-0.5, 0.5)) * bloom.resolution_rcp, 0).rgb, 30);
		color += min(texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(0.5, 0.5)) * bloom.resolution_rcp, 0).rgb, 30);

		color /= 4.0;

		color = max(color - bloom.threshold, 0);
		color *= exposure;
	}
	else
	{
		color += texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(-0.5, -0.5)) * bloom.resolution_rcp, 0).rgb;
		color += texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(0.5, -0.5)) * bloom.resolution_rcp, 0).rgb;
		color += texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(-0.5, 0.5)) * bloom.resolution_rcp, 0).rgb;
		color += texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(0.5, 0.5)) * bloom.resolution_rcp, 0).rgb;

		color /= 4.0;

		color *= exposure;

		color = min(color, 10); // clamp upper limit: avoid incredibly large values to overly dominate bloom (high speculars were causing problems)
		color = max(color - bloom.threshold, 0);
	}

	// the strength: the blur and the tonemap's taps are linear, so this scales what the tonemap adds
	color *= bloom.strength;

	bindless_rwtextures[descriptor_index(bloom.texture_output)][DTid.xy] = float4(color, 1);
}
