#ifdef EMITTEDPARTICLE_LIGHTING
#define TRANSPARENT // the transparent pass's light lists, as the lit pixel shader
#define DISABLE_DECALS
#define DISABLE_VOXELGI
#endif // EMITTEDPARTICLE_LIGHTING
#include "globals.hlsli"
#include "ShaderInterop_EmittedParticle.h"
#include "emittedparticleHF.hlsli"
#ifdef EMITTEDPARTICLE_LIGHTING
#include "shadingHF.hlsli"

// A legacy .PE layer (xEmitterFadeinTime >= 0) is lit here, at each corner and at the sprite's centre, as the DX11 build
// lit it, rather than by every pixel: the light across a sprite is smooth, so the puffs of a cloud blend into each other
// instead of each reading as a lit ball with a dark rim. The light at a point of the sprite facing N: the lights of the
// screen tile the point falls in, their shadows, the ambient; the direct light divided by PI as objects have it, and no
// Fresnel (no reflectance), as DX11 had none on particles
float3 ParticleLight(float3 P, float3 N)
{
	ShaderMaterial material = EmitterGetMaterial();
	Lighting lighting;
	lighting.create(0, 0, GetAmbient(N), 0);

	Surface surface;
	surface.init();
	surface.create(material, half4(1, 1, 1, 1), surfacemap_simple);
	surface.P = P;
	surface.N = N;
	surface.V = normalize(GetCamera().position - P);
	const float4 clip = mul(GetCamera().view_projection, float4(P, 1));
	const float2 ndc = clip.xy / max(clip.w, 0.0001);
	const float2 res = float2(GetCamera().internal_resolution);
	surface.pixel = (min16uint2)clamp((ndc * float2(0.5, -0.5) + 0.5) * res, 0, res - 1);
	surface.sss = material.GetSSS();
	surface.sss_inv = material.GetSSSInverse();
	surface.update();
	surface.f0 = 0;

	TiledLighting(surface, lighting, GetFlatTileIndex(surface.pixel));
	return lighting.direct.diffuse / PI + lighting.indirect.diffuse;
}

// the sprite's rounded normal at a place on it (0-1 across, unrotated), as the pixel shader makes it
float3 ParticleNormal(float2 unrotated_uv)
{
	float3 N;
	N.x = -cos(PI * unrotated_uv.x);
	N.y = cos(PI * unrotated_uv.y);
	N.z = -sin(PI * length(unrotated_uv));
	return normalize(mul((float3x3)GetCamera().inverse_view, N));
}
#endif // EMITTEDPARTICLE_LIGHTING

static const float3 BILLBOARD[] = {
	float3(-1, -1, 0),	// 0
	float3(1, -1, 0),	// 1
	float3(-1, 1, 0),	// 2
	float3(1, 1, 0),	// 4
};

StructuredBuffer<Particle> particleBuffer : register(t1);
StructuredBuffer<uint> culledIndirectionBuffer : register(t2);
StructuredBuffer<uint> culledIndirectionBuffer2 : register(t3);

VertextoPixel main(uint vid : SV_VertexID, uint instanceID : SV_InstanceID)
{
	ShaderGeometry geometry = EmitterGetGeometry();

	uint particleIndex = culledIndirectionBuffer2[culledIndirectionBuffer[instanceID]];
	uint vertexID = particleIndex * 4 + vid;

	float3 position = bindless_buffers_float4[descriptor_index(geometry.vb_pos_wind)][vertexID].xyz;
	float3 normal = normalize(bindless_buffers_float4[descriptor_index(geometry.vb_nor)][vertexID].xyz);
	float4 uvsets = bindless_buffers_float4[descriptor_index(geometry.vb_uvs)][vertexID];
	float4 color = bindless_buffers_float4[descriptor_index(geometry.vb_col)][vertexID];

	// load particle data:
	Particle particle = particleBuffer[particleIndex];

	// calculate render properties from life:
	float lifeLerp = 1 - particle.life / particle.maxLife;
	float size = lerp(particle.sizeBeginEnd.x, particle.sizeBeginEnd.y, lifeLerp);

	// Sprite sheet UV transform:
	const float spriteframe = xEmitterFrameRate == 0 ? 
		lerp(xEmitterFrameStart, xEmitterFrameCount, lifeLerp) : 
		((xEmitterFrameStart + (particle.maxLife - particle.life) * xEmitterFrameRate) % xEmitterFrameCount);
	const float frameBlend = frac(spriteframe);

	VertextoPixel Out;
	Out.pos = float4(position, 1);
	Out.clip = dot(Out.pos, GetCamera().clip_plane);
	Out.pos = mul(GetCamera().view_projection, Out.pos);
	Out.tex = uvsets;
	Out.size = half(size);
	Out.color = pack_rgba(color);
	Out.unrotated_uv = half2(BILLBOARD[vertexID % 4].xy * float2(1, -1) * 0.5f + 0.5f);
	Out.frameBlend = half(frameBlend);
#ifdef EMITTEDPARTICLE_LIGHTING
	Out.litCorner = 0;
	Out.litCentre = 0;
	[branch]
	if (xEmitterFadeinTime >= 0)
	{
		Out.litCorner = ParticleLight(position, ParticleNormal(float2(Out.unrotated_uv)));
		Out.litCentre = ParticleLight(particle.position, ParticleNormal(float2(0.5, 0.5)));
	}
#endif // EMITTEDPARTICLE_LIGHTING
	return Out;
}
