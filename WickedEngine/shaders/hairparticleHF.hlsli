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

	// Base colour texture for this strand. In merged mode every blade in a chunk may use a
	// different DDS, so the material's BASECOLORMAP is bypassed for a bindless lookup keyed on
	// the strand's type. Returns false when the strand has no per-type texture and the caller
	// should fall back to the material as before.
	inline bool GGGetGrassTexture(out Texture2D<half4> tex)
	{
		tex = bindless_textures_half4[descriptor_index(0)];
		if (grasstype == 0)
			return false;
		// GGMAX 1.88 selective probe (mode 3): every strand takes type 0's blade texture, so the
		// per-strand texture choice can no longer vary. If the per-frame churn collapses with
		// only this frozen, the flicker is textureIndex selection.
		const uint gg_typeidx = (xHairFlags & HAIR_FLAG_GG_FREEZE_TEXTURE)
			? 0u : min(grasstype - 1, GG_HAIR_MAX_GRASS_TYPES - 1);
		const uint idx = xHairGrassTypes[gg_typeidx].textureIndex;
		if (idx == 0)
			return false;
		// GGMAX 1.89 — THE MERGED-GRASS FLICKER FIX. `idx` varies PER STRAND, so within one pixel
		// wave the lanes can want different descriptors. Indexing an unbounded descriptor array
		// divergently without NonUniformResourceIndex is undefined behaviour under the D3D12 spec
		// (globals.hlsli declares `Texture2D<half4> bindless_textures_half4[] : register(space24)`),
		// and on AMD the compiler scalarizes it — v_readfirstlane broadcasts the FIRST ACTIVE
		// LANE's descriptor to the whole wave, so every pixel in that wave samples one strand's
		// blade texture. A wave is a quad-granular screen tile, which is why the artefact is
		// square patches of another grass type; wave packing is re-decided every frame as the
		// blades sway, which is why it churns with no cross-frame memory.
		//
		// Per-type grass never hit this: it writes grasstype 0 and early-outs above to the single
		// material texture, uniform by construction. Every other divergent bindless access in this
		// engine is already wrapped (ShaderInterop_Renderer.h UniformTextureSlot, objectHF/
		// surfaceHF); the 1.74 GG line was the only one that bypassed it.
		//
		// Cost: a waterfall loop, one iteration per unique descriptor in the wave (~7-8 live types
		// per merged chunk). If that ever proves expensive the answer is a blade-texture ATLAS or
		// explicit scalarization — never a revert to the undefined behaviour.
		tex = bindless_textures_half4[NonUniformResourceIndex(descriptor_index(idx))];
		return true;
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
