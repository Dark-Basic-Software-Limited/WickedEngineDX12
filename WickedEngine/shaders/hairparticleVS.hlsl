#include "globals.hlsli"
#include "hairparticleHF.hlsli"
#include "ShaderInterop_HairParticle.h"

Buffer<uint> primitiveBuffer : register(t0);

VertexToPixel main(uint vid : SV_VertexID)
{
	ShaderMeshInstance inst = HairGetInstance();
	ShaderGeometry geometry = HairGetGeometry();

	VertexToPixel Out;
	Out.primitiveID = vid / 3;

	uint vertexID = primitiveBuffer[vid];
	float4 pos_wind = bindless_buffers_float4[descriptor_index(geometry.vb_pos_wind)][vertexID];
	float3 position = mul(inst.transform.GetMatrix(), float4(pos_wind.xyz, 1)).xyz;
	float4 nor_raw = bindless_buffers_float4[descriptor_index(geometry.vb_nor)][vertexID];
	float3 normal = normalize(nor_raw.xyz);
	float4 uvsets = bindless_buffers_float4[descriptor_index(geometry.vb_uvs)][vertexID];
	// GGMAX 1.92 — THE FLICKER FIX. The grass type now arrives in vb_uvs.w, NOT vb_nor.w.
	// The old claim on this line ("SNORM 8-bit stores integer steps exactly") was simply false:
	// vb_nor is R8G8B8A8_SNORM with a 2/255 = 0.0078431 step, while the CS wrote types spaced
	// 1/127 = 0.0078740 apart — adjacent types sat UNDER one quantisation step apart and the
	// 0.4% mismatch accumulated with index, so strands decoded a neighbouring type and sampled
	// its blade texture. That was the flickering coloured squares.
	// vb_uvs is R16G16B16A16_UNORM: 65535 steps, /255 gives 257 steps of margin per type.
	Out.grasstype = (uint)round(saturate(uvsets.w) * 255.0);

	Out.fade = saturate(distance(position.xyz, GetCamera().position.xyz) / xHairViewDistance);
	Out.fade = saturate(Out.fade - 0.8f) * 5.0f; // fade will be on edge and inwards 20%

	Out.pos = float4(position, 1);
	Out.clip = dot(Out.pos, GetCamera().clip_plane);
	Out.pos = mul(GetCamera().view_projection, Out.pos);

	Out.nor_wet = half4(normal, (half)bindless_buffers_float[descriptor_index(inst.vb_wetmap)][vertexID]);
	Out.tex = uvsets.xy;
	
	return Out;
}
