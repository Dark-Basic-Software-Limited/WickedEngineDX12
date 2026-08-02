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
		const uint idx = xHairGrassTypes[min(grasstype - 1, GG_HAIR_MAX_GRASS_TYPES - 1)].textureIndex;
		if (idx == 0)
			return false;
		tex = bindless_textures_half4[descriptor_index(idx)];
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
