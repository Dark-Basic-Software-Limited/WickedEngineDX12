// This is a modified version of the SSAO from Microsoft MiniEngine at https://github.com/Microsoft/DirectX-Graphics-Samples

#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

Texture2D<float> DS4x : register(t0);

RWTexture2D<float> DS8x : register(u0);
RWTexture2DArray<float> DS8xAtlas : register(u1);
RWTexture2D<float> DS16x : register(u2);
RWTexture2DArray<float> DS16xAtlas : register(u3);

groupshared float g_Pick[64]; // GGMAX 3.126b: this thread's DS8x texel, for the DS16x 2x2 pick

PUSHCONSTANT(msao_prepare, MSAO_PREPARE);

// GGMAX 3.126b: stock texel 'a' (top-left of the 2x2) unless the block straddles a depth edge
float gg_pick_depth(float a, float b, float c, float d)
{
	if (msao_prepare.gg_mode == 0)
		return a;
	float lo = min(min(a, b), min(c, d));
	float hi = max(max(a, b), max(c, d));
	if (hi - lo <= msao_prepare.gg_edge * lo)
		return a;
	return msao_prepare.gg_mode == 1 ? hi : lo;
}

[numthreads(8, 8, 1)]
void main(uint3 Gid : SV_GroupID, uint GI : SV_GroupIndex, uint3 GTid : SV_GroupThreadID, uint3 DTid : SV_DispatchThreadID)
{
	uint2 dim;
	DS4x.GetDimensions(dim.x, dim.y);

    uint2 s0 = DTid.xy << 1;
    float m1 = gg_pick_depth(DS4x[min(s0, dim - 1)], DS4x[min(s0 + uint2(1, 0), dim - 1)],
        DS4x[min(s0 + uint2(0, 1), dim - 1)], DS4x[min(s0 + uint2(1, 1), dim - 1)]);
    g_Pick[GI] = m1;
    GroupMemoryBarrierWithGroupSync();

    uint2 st = DTid.xy;
    uint2 stAtlas = st >> 2;
    uint stSlice = (st.x & 3) | ((st.y & 3) << 2);
    DS8x[st] = m1;
    DS8xAtlas[uint3(stAtlas, stSlice)] = m1;

    if ((GI & 011) == 0)
    {
        float m4 = gg_pick_depth(m1, g_Pick[GI + 1], g_Pick[GI + 8], g_Pick[GI + 9]);
        uint2 st = DTid.xy >> 1;
        uint2 stAtlas = st >> 2;
        uint stSlice = (st.x & 3) | ((st.y & 3) << 2);
        DS16x[st] = m4;
        DS16xAtlas[uint3(stAtlas, stSlice)] = m4;
    }
}
