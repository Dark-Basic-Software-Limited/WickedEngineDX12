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
	// GGMAX 1.87 (flicker hunt): freeze merged grass's per-strand type resolution. Strands keep
	// the entity's own type instead of adopting the paint cell's, so every per-type parameter
	// becomes uniform across the system. If the scene-wide per-frame churn dies with this set,
	// the flicker IS the type resolution; if it survives, type resolution is exonerated.
	// Diagnostic only — it changes what renders, so judge it on CONSECUTIVE-FRAME diff within
	// one config, never against a different config's image.
	HAIR_FLAG_GG_FREEZE_TYPE = 1 << 3,
	// GGMAX 1.88: SELECTIVE freezes — same idea, one parameter at a time, so the surviving
	// suspects can be separated. Whole-freeze proved the flicker is in the type-dependent path
	// (11.9 -> 0.2 meanAbsDiff); these say WHICH parameter.
	HAIR_FLAG_GG_FREEZE_PRESENT = 1 << 4,   // ignore per-type `present` (never kill a strand by it)
	HAIR_FLAG_GG_FREEZE_TEXTURE = 1 << 5,   // every strand uses type 0's blade texture
	HAIR_FLAG_GG_FREEZE_LENGTH  = 1 << 6,   // every strand uses type 0's length
	// GGMAX 1.90: debug visualisation - hair PS returns the raw per-strand grasstype as a flat
	// colour. Never touches a descriptor, so it separates "type value is wrong" from "texture
	// fetch is wrong". SET_GRASSTYPEFREEZE 5.
	HAIR_FLAG_GG_DEBUG_TYPEVIS = 1 << 7,
	// GGMAX 1.93: write a STABLE per-strand hash into the type channel instead of the resolved
	// type. Combined with TYPEVIS this is the discriminator: if the visualisation goes stable the
	// vertex write/read plumbing is sound and gg_resolved_type is the churning input; if it still
	// churns the plumbing itself is corrupt. Mask bit 32.
	HAIR_FLAG_GG_DEBUG_STABLETYPE = 1 << 8,
	// GGMAX 1.94: suppress the cull early-return so EVERY strand writes its vertex slots every
	// frame. If the churn collapses with this set, the unwritten-slot path at the early return
	// is the corruption. Mask bit 64.
	HAIR_FLAG_GG_DEBUG_ALWAYSWRITE = 1 << 9,
	// GGMAX 1.95: resolve xHairGrassTypes[].textureIndex via a UNIFORM-index compare loop in the
	// hair PS instead of a divergent dynamic CB-array read. The 2026-08-04 ladder proved the
	// flicker is the texture fetch alone (mode 4 collapses 12.0 -> 0.3 while width/length/etc
	// still vary), the fetched INDEX VALUE is stable (mode 48), and streaming is exonerated
	// (paused-from-second-zero churn unchanged). The one unhardened step left is the CB read at
	// hairparticleHF.hlsli GGGetGrassTexture: 1.89 wrapped the DESCRIPTOR index in
	// NonUniformResourceIndex but the CB-array read one line earlier can be scalarized
	// (readfirstlane) by the compiler, handing every pixel-wave the first active lane's texture
	// — square wave-tile patches, re-rolled per frame by rasterizer wave packing. It also
	// explains why 1.89's predicted waterfall never appeared: idx was already wave-uniform.
	// Mask bit 128. If churn collapses with this set, the CB read is the flicker.
	// RESULT 2026-08-04: churn UNCHANGED (12.26/12.29/12.36 vs control 12.5) — the CB read is
	// NOT scalarized. Kept for the ledger. Together with FREEZE_TEXTURE collapsing to 0.30 this
	// narrows the defect to the per-type textures themselves being sampled: either one type's
	// descriptor points at per-frame-changing content, or divergent-resource sampling per se.
	HAIR_FLAG_GG_DEBUG_UNIFORMCB = 1 << 10,
	// GGMAX 1.95b: bits 16-23 of xHairFlags carry (forcedType+1); non-zero forces EVERY pixel's
	// texture lookup to that type — a "FREEZE_TEXTURE for each k". The bisector for the two
	// worlds above: if some forced type k churns while the others are clean, type k's texture
	// content is the flicker (name the resource via DUMP_GRASSTYPES); if every forced type is
	// clean individually but the real mixed scene churns, divergent sampling itself is the bug.
	// Harness: SET_GRASSTYPEFREEZE (k+1)*256.
};
#define HAIR_FLAG_GG_FORCETYPE_SHIFT 16u
#define HAIR_FLAG_GG_FORCETYPE_MASK 0xFFu

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
	// GGMAX 1.95: number of live entries in xHairGrassTypes[] (0 outside merged mode), so the
	// uniform-index resolve loop is bounded by the ~7-8 painted types instead of all 88 slots.
	uint xHairGrassTypeCount;
	float xHair_padding_alt2;

	HairParticleAtlasRect xHairAtlasRects[64];

	// GGMAX 1.74: per-type parameters, indexed by the strand's resolved grass type. Only read
	// when xHairGrassType == GG_HAIR_GRASS_MERGED; costs 2816 B of a 64 KB budget.
	GGHairGrassType xHairGrassTypes[GG_HAIR_MAX_GRASS_TYPES];
};

#endif // WI_SHADERINTEROP_HAIRPARTICLE_H
