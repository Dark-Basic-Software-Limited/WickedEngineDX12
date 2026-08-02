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

// GGMAX 1.74 merged grass: one hair system per terrain CHUNK instead of one per
// (chunk x painted type). Measured on the TESTPRO1 benchmark the chunks carry 7-8 painted
// types each, so the per-type split was allocating ~10x the strand buffers actually needed
// and running ~10x the strand physics, with all but one type per cell emitting degenerate
// zero-area quads. In merged mode the simulate CS resolves each strand's type from the paint
// map and indexes this table instead of killing the strand, so one system draws what N did.
//
// Placement is bit-identical, not merely equivalent: every per-type system in a chunk already
// shared the same randomSeed, emitter mesh and index list, and strandCount is per CHUNK. So
// strand i sits on the same triangle with the same barycentrics in every type's system, a
// paint cell holds exactly one type, and the surviving sets are disjoint with union exactly
// "strands on painted cells".
#define GG_HAIR_MAX_GRASS_TYPES 88
#define GG_HAIR_GRASS_MERGED 0xFFFFFFFFu   // xHairGrassType sentinel meaning "this system owns every type"

struct GGHairGrassType
{
	float length;
	float width;
	float stiffness;
	float drag;

	float viewDistance;
	uint textureIndex;      // bindless SRV descriptor index of this type's blade DDS
	uint billboardCount;    // 1 for weed/kelp/seaweed, 2 for the rest
	// 1 = this type is painted in this chunk, 0 = absent. Absent types must NOT render: the
	// per-type build only ever created a system for types the chunk scan reported, so a cell
	// naming an unscanned type drew nothing. Merged mode would happily draw it and read denser
	// than the reference — measured +1.45 pp coverage on the flower benchmark before this gate.
	float present;
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
	// GGMAX 1.49b grass strand LOD (Step2Dist 0 = disabled -> stock behavior). Around
	// Step2Dist half the strands drop, around Step4Dist half the remainder — each strand at
	// its own hash-jittered radius (+-15%) so camera motion causes sparse single-blade events,
	// never a synchronized ring (the 1.49 two-shade shimmer). Survivors widen by an exact
	// hyperbolic 1/survivorFraction ramp aligned to the drop window; at WidthBoost=2.0 the
	// expected screen coverage is conserved through both bands. Deterministic per strand
	// (hashed strand id + camera distance only): bit-stable at a parked camera (1.37 lesson).
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
	float xHairGGLodWidthBoost; // GGMAX 1.49b widening endpoint per halving; 2.0 = coverage-neutral, <2 = far thinning
	float xHair_padding_alt1;
	float xHair_padding_alt2;

	HairParticleAtlasRect xHairAtlasRects[64];

	// GGMAX 1.74: per-type parameters, indexed by the strand's resolved grass type. Only read
	// when xHairGrassType == GG_HAIR_GRASS_MERGED; costs 2816 B of a 64 KB budget.
	GGHairGrassType xHairGrassTypes[GG_HAIR_MAX_GRASS_TYPES];
};

#endif // WI_SHADERINTEROP_HAIRPARTICLE_H
