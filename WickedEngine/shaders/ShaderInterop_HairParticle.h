#ifndef WI_SHADERINTEROP_HAIRPARTICLE_H
#define WI_SHADERINTEROP_HAIRPARTICLE_H

#include "ShaderInterop.h"
#include "ShaderInterop_Renderer.h"

#define THREADCOUNT_SIMULATEHAIR 64

struct alignas(16) PatchSimulationData
{
	float3 prevTail;
	float padding0;
	float3 currentTail;
	float padding1;
};

enum HAIR_FLAGS
{
	HAIR_FLAG_REGENERATE_FRAME = 1 << 0,
	HAIR_FLAG_UNORM_POS = 1 << 1,
	HAIR_FLAG_CAMERA_BEND = 1 << 2,
};

struct HairParticleAtlasRect
{
	float4 texMulAdd;
	float size;
	float aspect;
	float padding1;
	float padding2;
};

CBUFFER(HairParticleCB, CBSLOT_OTHER_HAIRPARTICLE)
{
	ShaderTransform xHairTransform;
	ShaderTransform xHairBaseMeshUnormRemap;

	uint xHairFlags;
	float xHairLength;
	float xHairStiffness;
	float xHairDrag;

	uint xHairParticleCount;
	uint xHairStrandCount;
	uint xHairSegmentCount;
	uint xHairRandomSeed;

	float xHairViewDistance;
	uint xHairBaseMeshIndexCount;
	uint xHairInstanceIndex;
	uint xHairLayerMask;

	float xHairRandomness;
	float xHairAspect;
	float xHairUniformity;
	uint xHairAtlasRectCount;

	float xHairGravityPower;
	uint xHairBillboardCount;
	// GGMAX 1.49 grass strand LOD (0 = disabled -> stock behavior). Beyond Step2Dist only
	// every 2nd strand draws; beyond Step4Dist only every 4th. Survivors widen by
	// xHairGGLodWidthBoost (once at step 2, squared at step 4) to preserve screen coverage.
	// Deterministic per strand (hashed strand id + camera distance only), so the pattern is
	// stable at a parked camera — no temporal pops (the 1.37 lesson).
	float xHairGGLodStep2Dist;
	float xHairGGLodStep4Dist;

	// GG-MAX Stage 3 Option B: per-strand visibility from a world-space paint mask.
	// xHairGrassType > 0 turns it on; the simulate CS samples a Texture2D<float> bound
	// at t4 in the simulate dispatch (R8_UNORM byte / 255) at uv = (worldXZ - origin) *
	// invWorldSize + 0.5 and zeros strand_length when the cell type doesn't match.
	// 0 = disabled — pure Wicked behavior, no shader sampling, no GG plumbing required.
	uint xHairGrassType;
	float xHairGrassMapInvWorldSize;
	float xHairGrassMapOriginX;
	float xHairGrassMapOriginZ;

	// GG-MAX Stage B.10: per-strand altitude filter — applied only when xHairGrassType != 0.
	// Strand world Y (base.y) is compared to xHairGrassWaterHeight; strands above use the
	// above-water pair [min, max], strands below use the underwater pair. Both cutoffs are
	// inclusive. Set min < world floor and max > world ceiling to disable that half — the
	// GG-side defaults (min = -1000, max = 30000; min_uw = -7000, max_uw = 1000) act as
	// "no filter" so out-of-the-box levels see zero behavioral change.
	float xHairGrassWaterHeight;
	float xHairGrassMinHeight;
	float xHairGrassMaxHeight;
	float xHairGrassMinHeightUnderwater;
	float xHairGrassMaxHeightUnderwater;
	float xHairGGLodWidthBoost; // GGMAX 1.49 (see above); 1.0 = no widening
	float xHair_padding_alt1;
	float xHair_padding_alt2;

	HairParticleAtlasRect xHairAtlasRects[64];
};

#endif // WI_SHADERINTEROP_HAIRPARTICLE_H
