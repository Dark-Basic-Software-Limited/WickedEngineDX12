#ifndef WI_SHADERINTEROP_OCEAN_H
#define WI_SHADERINTEROP_OCEAN_H
#include "ShaderInterop.h"

static const uint OCEAN_COMPUTE_TILESIZE = 8;

CBUFFER(OceanCB, CBSLOT_OTHER_OCEAN)
{
	float4 xOceanWaterColor;
	float4 xOceanExtinctionColor;
	float4 xOceanScreenSpaceParams;

	float xOceanTexelLength;
	float xOceanPatchSizeRecip;
	float xOceanMapHalfTexel;
	float xOceanWaterHeight;

	float xOceanSurfaceDisplacementTolerance;
	uint xOceanActualDim;
	uint xOceanInWidth;
	uint xOceanOutWidth;

	uint xOceanOutHeight;
	uint xOceanDtxAddressOffset;
	uint xOceanDtyAddressOffset;
	float xOceanTimeScale;

	float xOceanChoppyScale;
	float xOceanGridLen;
	float xOceanFoamUnitScale;	// converts world units to the meters the foam math was tuned in (1 = stock meters; GGMAX passes ~0.0254 for inch-scaled worlds)
	float xOceanFoamAmount;		// artistic multiplier on final foam intensity (1 = stock)

	float xOceanWaterColorDepth;	// GG delta 1.24: tints the refraction (see-through) toward xOceanWaterColor with depth so the base colour shows on transparent water. 0 = off (stock)
	float xOceanWCPad0;
	float xOceanWCPad1;
	float xOceanWCPad2;
};

#endif // WI_SHADERINTEROP_OCEAN_H
