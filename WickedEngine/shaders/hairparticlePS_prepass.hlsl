#include "globals.hlsli"
#include "objectHF.hlsli"
#include "hairparticleHF.hlsli"
#include "ShaderInterop_HairParticle.h"

#ifdef __PSSL__
#pragma PSSL_target_output_format (target 0 FMT_32_R)
#endif // __PSSL__

uint main(VertexToPixel input, out uint coverage : SV_Coverage) : SV_Target
{
	ShaderMaterial material = HairGetMaterial();

	half alpha = 1;

	// GGMAX 1.74 merged grass: alpha must come from the SAME texture the lit pass will use, or
	// the depth silhouette is cut from the wrong sprite.
	// GGMAX 1.96 flicker fix: NonUniform annotation AT the subscript, one expression (see
	// GGGetGrassTextureIndex — the out-param Texture2D version dropped it in the DXIL).
	uint ggtexidx;
	[branch]
	if (input.GGGetGrassTextureIndex(ggtexidx))
	{
		alpha = bindless_textures_half4[NonUniformResourceIndex(descriptor_index(ggtexidx))].Sample(sampler_linear_clamp, input.tex.xy).a;
	}
	else
	[branch]
	if (material.textures[BASECOLORMAP].IsValid())
	{
		alpha = material.textures[BASECOLORMAP].Sample(sampler_linear_clamp, input.tex.xyxy).a;
	}
	
	coverage = AlphaToCoverage(alpha, material.GetAlphaTest(), input.GetDither(), input.pos);

	PrimitiveID prim;
	prim.init();
	prim.primitiveIndex = input.primitiveID;
	prim.instanceIndex = xHairInstanceIndex;
	prim.subsetIndex = 0;
	return prim.pack();
}
