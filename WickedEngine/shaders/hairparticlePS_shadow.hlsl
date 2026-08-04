#include "globals.hlsli"
#include "objectHF.hlsli"
#include "hairparticleHF.hlsli"
#include "ShaderInterop_HairParticle.h"

void main(VertexToPixel input)
{
	// Distance dithered fade:
	clip(dither(input.pos.xy) - input.fade);
	
	ShaderMaterial material = HairGetMaterial();

	float bias = 0;
	if (GetCamera().options & SHADERCAMERA_OPTION_DEDICATED_SHADOW_LODBIAS)
	{
		// Note: this hack is to improve the look of dedicated character shadow cascade which otherwise has too sharp grass shadows and cascade transition becomes too obvious
		bias = 3.2;
	}

	// GGMAX 1.74 merged grass: the shadow cutout must use the strand's own sprite too, or blades
	// cast the silhouette of whichever type happened to own the material.
	// GGMAX 1.96 flicker fix: NonUniform annotation AT the subscript, one expression (see
	// GGGetGrassTextureIndex — the out-param Texture2D version dropped it in the DXIL).
	uint ggtexidx;
	[branch]
	if (input.GGGetGrassTextureIndex(ggtexidx))
	{
		clip(bindless_textures_half4[NonUniformResourceIndex(descriptor_index(ggtexidx))].SampleBias(sampler_linear_clamp, input.tex.xy, bias).a - material.GetAlphaTest());
	}
	else
	[branch]
	if (material.textures[BASECOLORMAP].IsValid())
	{
		clip(material.textures[BASECOLORMAP].SampleBias(sampler_linear_clamp, input.tex.xyxy, bias).a - material.GetAlphaTest());
	}
}
