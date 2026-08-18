#include "globals.hlsli"

TextureCube<float4> cubeMap : register(t0);

struct VSOut_Sphere
{
	float4 pos : SV_POSITION;
	float3 nor : TEXCOORD0;
	float3 pos3D : TEXCOORD1;
};

float4 main(VSOut_Sphere input) : SV_TARGET
{
	float3 P = input.pos3D;
	float3 N = normalize(input.nor);
	float3 V = normalize(GetCamera().position - P);
	// GGMAX 2.89 (#157): the mip this inspection sphere samples. Stock hard-coded 0, which is
	// exactly why this sphere always looked clean — mip 0 is the unfiltered capture. The debug
	// draw now supplies the level in g_xColor.x (0 = stock behaviour) so a mip ladder can be
	// stepped on a live cube and the level where corruption enters can be NAMED.
	return float4(cubeMap.SampleLevel(sampler_linear_clamp, -reflect(V, N), g_xColor.x).rgb, 1);
}
