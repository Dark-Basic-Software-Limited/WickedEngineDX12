#include "globals.hlsli"
#include "objectHF.hlsli"
#include "hairparticleHF.hlsli"
#include "ShaderInterop_HairParticle.h"

void main(VertexToPixel input, out uint coverage : SV_Coverage)
{
	ShaderMaterial material = HairGetMaterial();

	float alpha = 1;

	// GGMAX 1.74 merged grass: per-strand texture, same reason as the prepass.
	Texture2D<half4> ggtex;
	[branch]
	if (input.GGGetGrassTexture(ggtex))
	{
		alpha = ggtex.Sample(sampler_linear_clamp, input.tex.xy).a;
	}
	else
	[branch]
	if (material.textures[BASECOLORMAP].IsValid())
	{
		alpha = material.textures[BASECOLORMAP].Sample(sampler_linear_clamp, input.tex.xyxy).a;
	}

	coverage = AlphaToCoverage(alpha, material.GetAlphaTest(), input.GetDither(), input.pos);
}
