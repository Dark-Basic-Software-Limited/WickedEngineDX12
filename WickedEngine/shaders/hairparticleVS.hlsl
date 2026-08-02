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
	// GGMAX 1.74: .w carries (grasstype + 1) / 127 written by the simulate CS in merged mode.
	// SNORM 8-bit stores integer steps exactly, so rounding recovers the index losslessly.
	Out.grasstype = (uint)round(saturate(nor_raw.w) * 127.0);
	float4 uvsets = bindless_buffers_float4[descriptor_index(geometry.vb_uvs)][vertexID];

	Out.fade = saturate(distance(position.xyz, GetCamera().position.xyz) / xHairViewDistance);
	Out.fade = saturate(Out.fade - 0.8f) * 5.0f; // fade will be on edge and inwards 20%

	Out.pos = float4(position, 1);
	Out.clip = dot(Out.pos, GetCamera().clip_plane);
	Out.pos = mul(GetCamera().view_projection, Out.pos);

	Out.nor_wet = half4(normal, (half)bindless_buffers_float[descriptor_index(inst.vb_wetmap)][vertexID]);
	Out.tex = uvsets.xy;
	
	return Out;
}
