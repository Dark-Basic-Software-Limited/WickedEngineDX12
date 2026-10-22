// This is a modified version of the SSAO from Microsoft MiniEngine at https://github.com/Microsoft/DirectX-Graphics-Samples

#include "globals.hlsli"
#include "ShaderInterop_Postprocess.h"

RWTexture2D<float> DS2x : register(u0);
RWTexture2DArray<float> DS2xAtlas : register(u1);
RWTexture2D<float> DS4x : register(u2);
RWTexture2DArray<float> DS4xAtlas : register(u3);

groupshared float g_CacheW[256];
groupshared float g_Pick[64]; // GGMAX 3.126b: this thread's DS2x texel, for the DS4x 2x2 pick

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
	texture_lineardepth.GetDimensions(dim.x, dim.y);

    uint2 startST = Gid.xy << 4 | GTid.xy;
    uint destIdx = GTid.y << 4 | GTid.x;
    g_CacheW[destIdx + 0] = texture_lineardepth[min(startST | uint2(0, 0), dim - 1)];
    g_CacheW[destIdx + 8] = texture_lineardepth[min(startST | uint2(8, 0), dim - 1)];
    g_CacheW[destIdx + 128] = texture_lineardepth[min(startST | uint2(0, 8), dim - 1)];
    g_CacheW[destIdx + 136] = texture_lineardepth[min(startST | uint2(8, 8), dim - 1)];

    GroupMemoryBarrierWithGroupSync();

    uint ldsIndex = (GTid.x << 1) | (GTid.y << 5);

    float w1 = gg_pick_depth(g_CacheW[ldsIndex], g_CacheW[ldsIndex + 1], g_CacheW[ldsIndex + 16], g_CacheW[ldsIndex + 17]);
    g_Pick[GI] = w1;
    GroupMemoryBarrierWithGroupSync();

    uint2 st = DTid.xy;
    uint slice = (st.x & 3) | ((st.y & 3) << 2);
    DS2x[st] = w1;
    DS2xAtlas[uint3(st >> 2, slice)] = w1;

    if ((GI & 011) == 0)
    {
        // stock: w1 (the 4x4 block's top-left texel); GG modes pick over this thread's 2x2 of DS2x texels
        float w4 = gg_pick_depth(w1, g_Pick[GI + 1], g_Pick[GI + 8], g_Pick[GI + 9]);
        st = DTid.xy >> 1;
        slice = (st.x & 3) | ((st.y & 3) << 2);
        DS4x[st] = w4;
        DS4xAtlas[uint3(st >> 2, slice)] = w4;
    }

}
