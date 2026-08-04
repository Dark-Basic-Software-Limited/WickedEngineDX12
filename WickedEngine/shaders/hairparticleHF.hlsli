#ifndef WI_HAIRPARTICLE_HF
#define WI_HAIRPARTICLE_HF
#include "globals.hlsli"
#include "ShaderInterop_HairParticle.h"

#define HairGetInstance() (load_instance(xHairInstanceIndex))
#define HairGetGeometry() (load_geometry(HairGetInstance().geometryOffset))
#define HairGetMaterial() (load_material(HairGetGeometry().materialIndex))

struct VertexToPixel
{
	precise float4 pos : SV_POSITION;
	float clip : SV_ClipDistance0;
	float2 tex : TEXCOORD;
	nointerpolation float fade : DITHERFADE;
	uint primitiveID : PRIMITIVEID;
	half4 nor_wet : NORMAL_WET;
	// GGMAX 1.74 merged grass: this strand's grass type, decoded from vb_nor.w in the VS.
	// nointerpolation because it is an index, not a quantity. 0 = stock hair (no per-strand type).
	nointerpolation uint grasstype : GRASSTYPE;

	// Blade-texture DESCRIPTOR INDEX for this strand. In merged mode every blade in a chunk may
	// use a different DDS, so the material's BASECOLORMAP is bypassed for a bindless lookup keyed
	// on the strand's type. Returns false when the strand has no per-type texture and the caller
	// should fall back to the material as before.
	//
	// GGMAX 1.96 — THE MERGED-GRASS FLICKER FIX (dxc-verified this time). This used to return a
	// Texture2D through an out parameter with NonUniformResourceIndex applied inside (1.89). The
	// compiled DXIL (dxc -dumpbin, 2026-08-04) showed EVERY createHandle in the PS carrying
	// nonUniformIndex=false: dxc DROPS the annotation when the resource handle crosses a function
	// boundary through a local/out-param. So the divergent Sample ran as undefined behaviour —
	// the driver readfirstlaned the descriptor and whole pixel-waves took one strand's texture,
	// re-rolled every frame by rasterizer wave packing = the scene-wide square-patch flicker.
	// (That also explains why 1.89 showed no FPS cost: the waterfall it predicted was never
	// generated.) Callers must therefore build the sample AS ONE EXPRESSION at the subscript:
	//   bindless_textures_half4[NonUniformResourceIndex(descriptor_index(gidx))].Sample(...)
	// — the one pattern dxc guarantees to preserve. NEVER reintroduce a resource-typed local or
	// helper return here, and verify any change with dxc -dumpbin: the grass Sample's
	// createHandle must say nonUniformIndex=true (i1 true).
	//
	// Elimination record for this hunt (2026-08-04, all measured): texture streaming (paused
	// from second zero — unchanged), CB-read scalarization (uniform-index loop probe, bit 128 —
	// unchanged), per-texture content (forced-type ladder, all 8 types individually clean),
	// captured-vs-live descriptor staleness (DUMP_GRASSTYPES — all equal). Only divergent
	// selection flickered, and only this annotation was missing from the DXIL.
	inline bool GGGetGrassTextureIndex(out uint gidx)
	{
		gidx = 0;
		if (grasstype == 0)
			return false;
		// GGMAX 1.88 selective probe (mode 3): every strand takes type 0's blade texture, so the
		// per-strand texture choice can no longer vary. If the per-frame churn collapses with
		// only this frozen, the flicker is textureIndex selection.
		uint gg_typeidx = (xHairFlags & HAIR_FLAG_GG_FREEZE_TEXTURE)
			? 0u : min(grasstype - 1, GG_HAIR_MAX_GRASS_TYPES - 1);
		// GGMAX 1.95b bisect: non-zero forced-type field pins EVERY lookup to one type — the
		// per-type version of FREEZE_TEXTURE. Uniform by construction (comes from the CB).
		const uint gg_forced = (xHairFlags >> HAIR_FLAG_GG_FORCETYPE_SHIFT) & HAIR_FLAG_GG_FORCETYPE_MASK;
		if (gg_forced != 0u)
			gg_typeidx = min(gg_forced - 1u, GG_HAIR_MAX_GRASS_TYPES - 1u);
		// GGMAX 1.95 probe (mask bit 128): the 2026-08-04 ladder pinned the merged flicker to this
		// fetch alone (FREEZE_TEXTURE collapses churn 12.0 -> 0.3; the index VALUE is stable per
		// the stable-hash test; streaming exonerated by pause-from-second-zero). The direct read
		// below dynamically indexes a CB array with a PER-PIXEL divergent index — a load the
		// compiler may scalarize to the wave's first active lane (1.89 hardened the descriptor
		// index one line later, but by then idx would already be wave-uniform, which is also why
		// 1.89's predicted waterfall never appeared). The probe resolves the same value through
		// compares against a wave-UNIFORM loop counter, so every CB load is scalarization-safe by
		// construction and per-lane selection happens in the compare, not the load.
		uint idx;
		[branch]
		if (xHairFlags & HAIR_FLAG_GG_DEBUG_UNIFORMCB)
		{
			idx = 0;
			for (uint t = 0; t < xHairGrassTypeCount; ++t)
			{
				// load OUTSIDE the compare: uniform index AND uniform control flow, so no
				// scalarization assumption can misfire; selection is pure lane-local ALU.
				const uint v = xHairGrassTypes[t].textureIndex;
				if (t == gg_typeidx)
					idx = v;
			}
		}
		else
		{
			idx = xHairGrassTypes[gg_typeidx].textureIndex;
		}
		gidx = idx;
		return idx != 0;
	}

	inline float3 GetPos3D()
	{
		return GetCamera().screen_to_world(pos);
	}

	inline float3 GetViewVector()
	{
		return GetCamera().screen_to_nearplane(pos) - GetPos3D(); // ortho support, cannot use cameraPos!
	}

	inline half GetDither()
	{
		return fade;
	}
};

#endif // WI_HAIRPARTICLE_HF
