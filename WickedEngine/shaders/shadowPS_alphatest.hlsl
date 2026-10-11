#define OBJECTSHADER_LAYOUT_SHADOW_TEX
#include "objectHF.hlsli"

void main(PixelInput input)
{
	ShaderMaterial material = GetMaterial();
	
	half alpha = 1;

	[branch]
	if (material.textures[BASECOLORMAP].IsValid())
	{
		alpha = material.textures[BASECOLORMAP].Sample(sampler_point_wrap, input.GetUVSets()).a;
	}
	
	[branch]
	if (material.textures[TRANSPARENCYMAP].IsValid())
	{
		alpha *= material.textures[TRANSPARENCYMAP].Sample(sampler_point_wrap, input.GetUVSets()).r;
	}
	
	ShaderMeshInstance meshinstance = load_instance(input.GetInstanceIndex());
	
	clip(alpha - material.GetAlphaTest() - meshinstance.GetAlphaTest());

	// GGMAX 3.108: distance dither. Stock shadow batches always carry dither 0; an object flagged
	// SHADOW_DISTANCE_FADE (the tree pool) now carries the same dither as its camera pass, and this
	// applies the same 8x8 Bayer cutout the camera pass uses without TAA (shadingHF.hlsli
	// AlphaToCoverage), so its shadow dissolves together with the mesh instead of vanishing in one
	// frame. The stipple lands in shadow-map texels, where PCF turns it into partial occlusion.
	ShaderMeshInstancePointer gg_poi;
	gg_poi.data = input.poi;
	const half gg_dither = gg_poi.GetDither();
	[branch]
	if (gg_dither > 0)
	{
		clip(dither((min16uint2)input.pos.xy) - gg_dither);
	}
}
