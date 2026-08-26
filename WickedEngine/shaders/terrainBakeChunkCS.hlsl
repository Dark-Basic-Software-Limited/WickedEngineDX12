#include "globals.hlsli"
#include "ColorSpaceUtility.hlsli"

// =============================================================================================
// GGMAX 3.25 - TERRAIN BAKE, per-chunk basecolour.
//
// This is terrainVirtualTextureUpdateCS.hlsl's BASECOLORMAP path with exactly one thing changed:
// it writes a plain RGBA8 pixel instead of a BC1 block, so the result is an ordinary Texture2D
// that an ordinary pixel shader can sample - no sparse atlas, no residency map, no page table,
// no feedback pass.
//
// WHY A SEPARATE SHADER AND NOT THE STOCK ONE
//   The stock CS emits BC1 blocks into a uint2 UAV that ALIASES a BC1 texture through DX12
//   sparse tile mapping (VirtualTextureAtlas::texture / texture_raw_block share a tile pool).
//   That aliasing is the whole reason it can compress at all, and it is not available for a
//   plain committed texture: UAVs on BC formats are prohibited, and the alias needs the sparse
//   pool. Rather than reproduce the sparse plumbing for a low-spec fallback path, this variant
//   drops the compression and writes the colour directly. The trade is video memory, which the
//   caller controls with the bake resolution knob, against removing the entire ~576 MB SVT
//   atlas - so the mode is a large net saving even uncompressed.
//
// SRGB
//   Matches the stock shader's storage convention: the sRGB CURVE is applied here and the value
//   is stored in a UNORM (not _SRGB) texture, because DX12 forbids a UAV on an sRGB format. The
//   pixel shader that samples this therefore has to undo the curve itself. Storing linear in 8
//   bits instead would band visibly in the dark half of the range, which is where terrain lives.
//
// The blend loop below is copied from the stock shader deliberately and must stay in step with
// it: front-to-back accumulation with early-out, same weight semantics, same GGMAX 1.53b
// distance-tiling policy. If that policy changes there, it changes here, or a baked chunk and
// a live chunk will not agree on texture scale at the seam between them.
// =============================================================================================

PUSHCONSTANT(push, TerrainVirtualTexturePush);

// GGMAX 1.53b distance-tiling policy - IDENTICAL to terrainVirtualTextureUpdateCS.hlsl.
// See the long comment there for why the ladder is capped rather than anchored.
float3 gg_tile_uv_scale_lod(float2 dim)
{
	const float res = 1.0 / push.resolution_rcp;
	const float cap = (float)(push.gg_tile_share >> 24u);
	const float hold = (float)((push.gg_tile_share >> 16u) & 0xFFu);
	const float2 diff = dim * push.resolution_rcp;
	const float lod_stock = log2(max(diff.x, diff.y));
	const float2 overscale = lod_stock < 0 ? diff : float2(1, 1);
	float2 repeats;
	if (cap > 0)
	{
		repeats = res * exp2(hold) / dim;
		const float rescale = min(1.0, cap / max(repeats.x, repeats.y));
		repeats = max(float2(1, 1), repeats * rescale);
	}
	else
	{
		repeats = 1.0 / overscale;
	}
	const float lod = log2(max(dim.x * repeats.x, dim.y * repeats.y) * push.resolution_rcp);
	return float3(repeats, max(0.0, lod));
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	// push.write_size is in PIXELS here, not 4x4 blocks as in the stock shader.
	if (DTid.x >= push.write_size || DTid.y >= push.write_size)
		return;

	Texture2DArray blendmap = bindless_textures2DArray[descriptor_index(push.blendmap_texture)];
	ByteAddressBuffer blendmap_buffer = bindless_buffers[descriptor_index(push.blendmap_buffer)];
	RWTexture2D<float4> output = bindless_rwtextures[descriptor_index(push.output_texture)];

	const int2 pixel = push.offset + int2(DTid.xy);
	const float2 uv = (pixel.xy + 0.5f) * push.resolution_rcp;
	const float2 uv2 = float2(uv.x, 1 - uv.y);

	half4 total_color = 0;
	half accumulation = 0;

	// front-to-back with early exit, like decals - same as the stock tile bake
	for (int blendmap_index = push.blendmap_layers - 1; blendmap_index >= 0; blendmap_index--)
	{
		float weight = blendmap.SampleLevel(sampler_linear_clamp, float3(uv, blendmap_index), 0).r;
		if (weight == 0)
			continue;

		uint materialIndex = blendmap_buffer.Load(push.blendmap_buffer_offset + blendmap_index * sizeof(uint));
		ShaderMaterial material = load_material(materialIndex);

		float4 baseColor = material.GetBaseColor();
		[branch]
		if (material.textures[BASECOLORMAP].IsValid())
		{
			Texture2D tex = bindless_textures[descriptor_index(material.textures[BASECOLORMAP].texture_descriptor)];
			float2 dim = 0;
			tex.GetDimensions(dim.x, dim.y);
			float3 ts = gg_tile_uv_scale_lod(dim);
			half4 baseColorMap = tex.SampleLevel(sampler_linear_wrap, uv2 * ts.xy, ts.z);
			baseColor *= baseColorMap;
		}
		total_color = mad(1 - accumulation, weight * baseColor, total_color);
		accumulation = mad(1 - weight, accumulation, weight);
		if (accumulation >= 1)
			break;
	}

	// Alpha carries the accumulated coverage so the caller can tell a genuinely black chunk from
	// one whose blendmap had no layers at all - a distinction that cost real time to diagnose by
	// screenshot on the far-tree work, and is one line to keep here.
	output[push.write_offset + DTid.xy] = float4(ApplySRGBCurve_Fast(total_color.rgb), accumulation);
}
