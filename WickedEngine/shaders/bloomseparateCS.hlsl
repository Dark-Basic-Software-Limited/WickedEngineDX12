#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"
// This will cut out bright parts (>1) and also downsample 4x

PUSHCONSTANT(bloom, Bloom);

[numthreads(POSTPROCESS_BLOCKSIZE, POSTPROCESS_BLOCKSIZE, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	const float2 uv = DTid.xy + 0.5;

	half3 color = 0;

	Texture2D<half4> texture_input = bindless_textures_half4[descriptor_index(bloom.texture_input)];
	// GG (BR-0024, Lee: "cap at 30 for better DX11 parity"): each sample capped at 30 BEFORE averaging, as the DX11 fork
	// did - a lone very bright specular texel then adds at most 30/4 instead of saturating the whole cap
	color += min(texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(-0.5, -0.5)) * bloom.resolution_rcp, 0).rgb, 30);
	color += min(texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(0.5, -0.5)) * bloom.resolution_rcp, 0).rgb, 30);
	color += min(texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(-0.5, 0.5)) * bloom.resolution_rcp, 0).rgb, 30);
	color += min(texture_input.SampleLevel(sampler_linear_clamp, (uv + float2(0.5, 0.5)) * bloom.resolution_rcp, 0).rgb, 30);

	color /= 4.0;

	float exposure = bloom.exposure;

	[branch]
	if (bloom.buffer_input_luminance >= 0)
	{
		exposure *= bindless_buffers[descriptor_index(bloom.buffer_input_luminance)].Load<float>(LUMINANCE_BUFFER_OFFSET_EXPOSURE);
	}
	
	color *= exposure;

	color = min(color, 30); // clamp upper limit: avoid incredibly large values to overly dominate bloom (high speculars were causing problems) - GG: was 10; 30 (BR-0024) so the per-sample DX11 cap above is not undone; still applied after exposure (DX11 applied exposure later, in the tonemap)
	color = max(color - bloom.threshold, 0);

	bindless_rwtextures[descriptor_index(bloom.texture_output)][DTid.xy] = float4(color, 1);
}
