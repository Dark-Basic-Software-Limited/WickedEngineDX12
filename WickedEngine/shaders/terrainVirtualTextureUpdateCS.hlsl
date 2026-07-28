#include "globals.hlsli"
#include "BlockCompress.hlsli"
#include "ColorSpaceUtility.hlsli"

PUSHCONSTANT(push, TerrainVirtualTexturePush);

#if !defined(UPDATE_NORMALMAP) && !defined(UPDATE_SURFACEMAP) && !defined(UPDATE_EMISSIVEMAP)
#define UPDATE_BASECOLORMAP
#endif

static const uint2 block_offsets[16] = {
	uint2(0, 0), uint2(1, 0), uint2(2, 0), uint2(3, 0),
	uint2(0, 1), uint2(1, 1), uint2(2, 1), uint2(3, 1),
	uint2(0, 2), uint2(1, 2), uint2(2, 2), uint2(3, 2),
	uint2(0, 3), uint2(1, 3), uint2(2, 3), uint2(3, 3),
};

// GGMAX 1.53: distance-tiling policy for terrain material sampling.
// Stock bakes each VT mip with the texture repeat count halved per mip (repeats = region
// resolution / texture size, floored at 1) — an anti-tiling feature (fewer, larger repeats at
// distance), but adjacent VT mips then hold DIFFERENT tilings, and the renderer's trilinear
// blend between them (SampleVirtual) cross-fades two UV scales of the same texture — a visible
// swimming band that sits close to the camera in inch-scale worlds.
// Policy: the K mips below mip0 keep mip0's repeat count (their bakes become true downsamples
// of the same layout = invisible transitions); beyond K the halving resumes, pushing the first
// scale cross-fade K mips further out while keeping the far-field anti-tiling.
// push.gg_tile_share: bits 0..23 = region mip0 resolution, bits 24..31 = K. K=0 = stock output.
// Returns: xy = UV scale (repeat count per axis), z = texture LOD to sample.
float3 gg_tile_uv_scale_lod(float2 dim)
{
	const float res = 1.0 / push.resolution_rcp;
	const float res0 = max((float)(push.gg_tile_share & 0xFFFFFFu), res);
	const float K = (float)(push.gg_tile_share >> 24u);
	const float m = log2(res0 / res); // which VT mip of the region this bake fills
	const float res_eff = res0 * exp2(-max(0.0, m - K)); // K=0: == res (stock); within the share window: == res0
	const float2 diff = dim / res_eff;
	const float lod = log2(max(diff.x, diff.y));
	const float2 overscale = lod < 0 ? diff : float2(1, 1);
	return float3(1.0 / overscale, max(0.0, lod) + log2(res_eff / res));
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= push.write_size || DTid.y >= push.write_size)
		return;

	Texture2DArray blendmap = bindless_textures2DArray[descriptor_index(push.blendmap_texture)];
	ByteAddressBuffer blendmap_buffer = bindless_buffers[descriptor_index(push.blendmap_buffer)];
	
#if defined(UPDATE_BASECOLORMAP) || defined(UPDATE_EMISSIVEMAP)
	RWTexture2D<uint2> output = bindless_rwtextures_uint2[descriptor_index(push.output_texture)];
#else
	RWTexture2D<uint4> output = bindless_rwtextures_uint4[descriptor_index(push.output_texture)];
#endif
		
#ifdef UPDATE_BASECOLORMAP
	float3 block_rgb[16];
#endif // UPDATE_BASECOLORMAP
		
#ifdef UPDATE_EMISSIVEMAP
	float3 block_rgb[16];
#endif // UPDATE_EMISSIVEMAP

#ifdef UPDATE_NORMALMAP
	float block_x[16];
	float block_y[16];
#endif // UPDATE_NORMALMAP

#ifdef UPDATE_SURFACEMAP
	float3 block_rgb[16];
	float block_a[16];
#endif // UPDATE_SURFACEMAP

	for(uint idx = 0; idx < 16; ++idx)
	{
		const uint2 block_offset = block_offsets[idx];
		const int2 pixel = push.offset + DTid.xy * 4 + block_offset;
		const float2 uv = (pixel.xy + 0.5f) * push.resolution_rcp;
		const float2 uv2 = float2(uv.x, 1 - uv.y);
		
		half4 total_color = 0;
		half accumulation = 0;

		// Note: blending is front-to back with early exit like decals
		for(int blendmap_index = push.blendmap_layers - 1; blendmap_index >= 0; blendmap_index--)
		{
			float weight = blendmap.SampleLevel(sampler_linear_clamp, float3(uv, blendmap_index), 0).r;
			if(weight == 0)
				continue;

			uint materialIndex = blendmap_buffer.Load(push.blendmap_buffer_offset + blendmap_index * sizeof(uint));
			ShaderMaterial material = load_material(materialIndex);

#ifdef UPDATE_BASECOLORMAP
			float4 baseColor = material.GetBaseColor();
			[branch]
			if (material.textures[BASECOLORMAP].IsValid())
			{
				Texture2D tex = bindless_textures[descriptor_index(material.textures[BASECOLORMAP].texture_descriptor)];
				float2 dim = 0;
				tex.GetDimensions(dim.x, dim.y);
				float3 ts = gg_tile_uv_scale_lod(dim); // GGMAX 1.53
				half4 baseColorMap = tex.SampleLevel(sampler_linear_wrap, uv2 * ts.xy, ts.z);
				baseColor *= baseColorMap;
			}
			total_color = mad(1 - accumulation, weight * baseColor, total_color);
			accumulation = mad(1 - weight, accumulation, weight);
			if(accumulation >= 1)
				break;
#endif // UPDATE_BASECOLORMAP

#ifdef UPDATE_NORMALMAP
			[branch]
			if (material.textures[NORMALMAP].IsValid())
			{
				Texture2D tex = bindless_textures[descriptor_index(material.textures[NORMALMAP].texture_descriptor)];
				float2 dim = 0;
				tex.GetDimensions(dim.x, dim.y);
				float3 ts = gg_tile_uv_scale_lod(dim); // GGMAX 1.53
				half2 normalMap = tex.SampleLevel(sampler_linear_wrap, uv2 * ts.xy, ts.z).rg;
				total_color.rg = mad(1 - accumulation, weight * normalMap, total_color.rg);
				accumulation = mad(1 - weight, accumulation, weight);
				if(accumulation >= 1)
					break;
			}
#endif // UPDATE_NORMALMAP

#ifdef UPDATE_SURFACEMAP
			float4 surface = float4(1, material.GetRoughness(), material.GetMetalness(), material.GetReflectance());
			[branch]
			if (material.textures[SURFACEMAP].IsValid())
			{
				Texture2D tex = bindless_textures[descriptor_index(material.textures[SURFACEMAP].texture_descriptor)];
				float2 dim = 0;
				tex.GetDimensions(dim.x, dim.y);
				float3 ts = gg_tile_uv_scale_lod(dim); // GGMAX 1.53
				half4 surfaceMap = tex.SampleLevel(sampler_linear_wrap, uv2 * ts.xy, ts.z);
				surface *= surfaceMap;
			}
			total_color = mad(1 - accumulation, weight * surface, total_color);
			accumulation = mad(1 - weight, accumulation, weight);
			if(accumulation >= 1)
				break;
#endif // UPDATE_SURFACEMAP

#ifdef UPDATE_EMISSIVEMAP
			float4 emissiveColor = 0;
			[branch]
			if (material.textures[EMISSIVEMAP].IsValid())
			{
				Texture2D tex = bindless_textures[descriptor_index(material.textures[EMISSIVEMAP].texture_descriptor)];
				float2 dim = 0;
				tex.GetDimensions(dim.x, dim.y);
				float3 ts = gg_tile_uv_scale_lod(dim); // GGMAX 1.53
				half4 emissiveMap = tex.SampleLevel(sampler_linear_wrap, uv2 * ts.xy, ts.z);
				emissiveColor.rgb = emissiveMap.rgb * emissiveMap.a;
				emissiveColor.a = emissiveMap.a;
			}
			total_color = mad(1 - accumulation, weight * emissiveColor, total_color);
			accumulation = mad(1 - weight, accumulation, weight);
			if(accumulation >= 1)
				break;
#endif // UPDATE_EMISSIVEMAP
		}

#ifdef UPDATE_BASECOLORMAP
		block_rgb[idx] = ApplySRGBCurve_Fast(total_color.rgb);
#endif // UPDATE_BASECOLORMAP

#ifdef UPDATE_NORMALMAP
		block_x[idx] = total_color.r;
		block_y[idx] = total_color.g;
#endif // UPDATE_NORMALMAP

#ifdef UPDATE_SURFACEMAP
		block_rgb[idx] = total_color.rgb;
		block_a[idx] = total_color.a;
#endif // UPDATE_SURFACEMAP

#ifdef UPDATE_EMISSIVEMAP
		block_rgb[idx] = ApplySRGBCurve_Fast(total_color.rgb);
#endif // UPDATE_EMISSIVEMAP
	}

	const uint2 write_coord = push.write_offset + DTid.xy;

#ifdef UPDATE_BASECOLORMAP
	output[write_coord] = CompressBC1Block(block_rgb);
#endif // UPDATE_BASECOLORMAP

#ifdef UPDATE_NORMALMAP
	output[write_coord] = CompressBC5Block(block_x, block_y);
#endif // UPDATE_NORMALMAP

#ifdef UPDATE_SURFACEMAP
	output[write_coord] = CompressBC3Block(block_rgb, block_a);
#endif // UPDATE_SURFACEMAP

#ifdef UPDATE_EMISSIVEMAP
	output[write_coord] = CompressBC1Block(block_rgb);
#endif // UPDATE_EMISSIVEMAP
}

