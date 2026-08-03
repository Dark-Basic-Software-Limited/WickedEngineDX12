#include "globals.hlsli"
#include "hairparticleHF.hlsli"
#include "ShaderInterop_HairParticle.h"

static const half3 HAIRPATCH[] = {
	// root (for every strand):
	half3(-1, -1, 0),
	half3(1, -1, 0),

	// cap (for every segment of every strand):
	half3(-1, 1, 0),
	half3(1, 1, 0),
};

Buffer<uint> meshIndexBuffer : register(t0);
Buffer<float4> meshVertexBuffer_POS : register(t1);
Buffer<half4> meshVertexBuffer_NOR : register(t2);
Buffer<half> meshVertexBuffer_length : register(t3);

// GG-MAX Stage 3 Option B: paint mask. R8_UNORM byte/255 — convert to byte with
// uint(sample * 255 + 0.5). xHairGrassType == 0 leaves this unused; otherwise the
// strand-position-driven sample happens inside main() (added in B.3).
Texture2D<float> texHairGrassMap : register(t4);

RWStructuredBuffer<PatchSimulationData> simulationBuffer : register(u0);
RWBuffer<float4> vertexBuffer_POS : register(u1);
RWBuffer<float4> vertexBuffer_UVS : register(u2);
RWBuffer<uint> culledIndexBuffer : register(u3);
RWStructuredBuffer<IndirectDrawArgsIndexedInstanced> indirectBuffer : register(u4);
RWBuffer<float4> vertexBuffer_POS_RT : register(u5);
RWBuffer<float4> vertexBuffer_NOR : register(u6);
RWBuffer<uint> primitiveBuffer : register(u7);

[numthreads(THREADCOUNT_SIMULATEHAIR, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint3 Gid : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
	if (DTid.x >= xHairStrandCount)
		return;

	const bool regenerate_frame = xHairFlags & HAIR_FLAG_REGENERATE_FRAME;
	const uint gfx_vertexcount_per_strand = (xHairSegmentCount * 2 + 2) * xHairBillboardCount;
	const uint gfx_indexcount_per_strand = 6 * xHairBillboardCount * xHairSegmentCount;
	const uint index0 = DTid.x * gfx_indexcount_per_strand;
	const uint vertexID0 = DTid.x * gfx_vertexcount_per_strand;
	uint v0 = vertexID0;
		
	ShaderGeometry geometry = HairGetGeometry();
	
	RNG rng;
	rng.init(uint2(xHairRandomSeed, DTid.x), 0);
	
	// random triangle on emitter surface:
	const uint triangleCount = xHairBaseMeshIndexCount / 3;
	const uint tri = rng.next_uint(triangleCount);

	// load indices of triangle from index buffer
	uint i0 = meshIndexBuffer[tri * 3 + 0];
	uint i1 = meshIndexBuffer[tri * 3 + 1];
	uint i2 = meshIndexBuffer[tri * 3 + 2];

	// load vertices of triangle from vertex buffer:
	float3 pos0 = meshVertexBuffer_POS[i0].xyz;
	float3 pos1 = meshVertexBuffer_POS[i1].xyz;
	float3 pos2 = meshVertexBuffer_POS[i2].xyz;
	half3 nor0 = meshVertexBuffer_NOR[i0].xyz;
	half3 nor1 = meshVertexBuffer_NOR[i1].xyz;
	half3 nor2 = meshVertexBuffer_NOR[i2].xyz;
	half length0 = meshVertexBuffer_length[i0];
	half length1 = meshVertexBuffer_length[i1];
	half length2 = meshVertexBuffer_length[i2];

	// random barycentric coords:
	float f = rng.next_float();
	float g = rng.next_float();
	[flatten]
	if (f + g > 1)
	{
		f = 1 - f;
		g = 1 - g;
	}
	float2 bary = float2(f, g);

	// compute final surface position on triangle from barycentric coords:
	float3 position = attribute_at_bary(pos0, pos1, pos2, bary);
	position = mul(xHairBaseMeshUnormRemap.GetMatrix(), float4(position, 1)).xyz; // position UNORM -> FLOAT
	half3 target = normalize(attribute_at_bary(nor0, nor1, nor2, bary));
	target = normalize(mul(xHairTransform.GetMatrixAdjoint(), target));
	// GG fix: the stored per-vertex normals from wiTerrain.cpp are computed
	// from a fixed (V, V+x, V+z) reference triangle, which does NOT match
	// the actual mesh triangulation at most vertices. At chunk_scale > 1
	// (we use 80) the resulting per-vertex normals can disagree wildly with
	// the real face normals, producing chaotic blade orientation. We
	// override with the face normal of the triangle the blade actually sits
	// on, computed in-shader from the three vertex positions (with the same
	// UNORM->FLOAT remap that the simulation applies to `position`). Note
	// the operand order — terrain mesh winding makes cross(P1-P0, P2-P0)
	// point downward, so swap to cross(P2-P0, P1-P0) for +Y.
	{
		float3 P0 = mul(xHairBaseMeshUnormRemap.GetMatrix(), float4(pos0, 1)).xyz;
		float3 P1 = mul(xHairBaseMeshUnormRemap.GetMatrix(), float4(pos1, 1)).xyz;
		float3 P2 = mul(xHairBaseMeshUnormRemap.GetMatrix(), float4(pos2, 1)).xyz;
		target = (half3)normalize(cross(P2 - P0, P1 - P0));
	}
	half3 tangent = normalize(mul(half3(hemispherepoint_cos(rng.next_float(), rng.next_float()).xy, 0), get_tangentspace(target)));
	half3 binormal = cross(target, tangent);
	half strand_length = attribute_at_bary(length0, length1, length2, bary);
	
	const uint currentFrame = uint(noise_gradient_3D(position * xHairUniformity) * 1000) % xHairAtlasRectCount;
	const HairParticleAtlasRect atlas_rect = xHairAtlasRects[currentFrame];
	
	// Transform particle by the emitter object matrix:
	float3 base = mul(xHairTransform.GetMatrix(), float4(position.xyz, 1)).xyz;

	// GG-MAX Stage 3 Option B: per-strand visibility check against the GG paint mask.
	// pGrassMap stores `flattened (0x80) | encodedID`, where encodedID is 0 (none),
	// 1 (legacy default = type 0), or 2..N+1 (type 0..N-1, with +2 offset). We sample
	// the R8_UNORM byte at this strand's world XZ and zero strand_length when the cell
	// is flattened, unpainted, or paints a different grass type than this entity owns.
	// xHairGrassType == 0 disables the check entirely — preserves upstream Wicked hair.
	// GGMAX 1.74: resolved grass type for this strand. In merged mode it is read from the paint
	// map and drives every per-type parameter below; in per-type mode it stays at the entity's
	// own type so the existing code path is untouched.
	uint gg_resolved_type = (xHairGrassType == GG_HAIR_GRASS_MERGED) ? 0u : (xHairGrassType - 1u);

	if (xHairGrassType != 0u)
	{
		float2 uv =
			(float2(base.x, base.z) - float2(xHairGrassMapOriginX, xHairGrassMapOriginZ))
			* xHairGrassMapInvWorldSize + 0.5;
		if (any(uv < 0.0) || any(uv >= 1.0))
		{
			strand_length = 0;
		}
		else
		{
			uint2 dim;
			texHairGrassMap.GetDimensions(dim.x, dim.y);
			uint2 px = uint2(uv * float2(dim));
			float sample = texHairGrassMap.Load(int3(px, 0));
			uint byteVal = uint(sample * 255.0 + 0.5);
			if ((byteVal & 0x80u) != 0u)
			{
				strand_length = 0; // flattened
			}
			else
			{
				uint encoded = byteVal & 0x7Fu;
				uint cellType = (encoded >= 2u) ? (encoded - 2u) : 0u;
				if (encoded == 0u)
				{
					strand_length = 0; // unpainted cell — no grass of any type here
				}
				else if (xHairGrassType == GG_HAIR_GRASS_MERGED && (xHairFlags & HAIR_FLAG_GG_FREEZE_TYPE))
				{
					// GGMAX 1.87 flicker probe: keep the strand but do NOT adopt the cell's type,
					// so gg_resolved_type stays 0 for every strand and all per-type parameters go
					// uniform. Density is wrong under this flag on purpose — it exists only to
					// answer whether per-frame churn comes from the type resolution.
				}
				else if (xHairGrassType == GG_HAIR_GRASS_MERGED)
				{
					// Merged: this system owns every type, so adopt the cell's type instead of
					// killing the strand. This is the whole saving — the strands that used to be
					// discarded here (one system per type meant ~7/8 of them on this content)
					// were still allocated, still simulated, and still emitted degenerate quads.
					gg_resolved_type = min(cellType, (uint)(GG_HAIR_MAX_GRASS_TYPES - 1));
					// ...but only for types this chunk actually built. The per-type build made a
					// system solely for scanned types, so a cell naming any other type rendered
					// nothing; matching that keeps density identical instead of adding grass.
					// GGMAX 1.88 selective probe: skip the present gate entirely so no strand is
					// killed by it. If the per-frame churn collapses with only this frozen,
					// `present` is the flicker. Density is wrong while on — consecutive-frame
					// diff within this config is the only valid reading.
					if (!(xHairFlags & HAIR_FLAG_GG_FREEZE_PRESENT)
						&& xHairGrassTypes[gg_resolved_type].present < 0.5)
					{
						strand_length = 0;
					}
				}
				else if ((cellType + 1u) != xHairGrassType)
				{
					strand_length = 0;
				}
			}
		}

		// GG-MAX Stage B.10: per-strand slope filter (grass on cliffs). DX11's
		// GGGrass_UpdateInstances rejected instances with `ny < 0.7` sampled from a
		// coarse per-chunk normal map; jittered per-instance sampling made that filter
		// look clean in DX11. In Wicked we place many more strands per chunk against
		// a shared paint mask, and CPU cell filtering is doomed by a mismatch of
		// resolutions (grass-map cell ~4.8 units vs. DX11 normal-map texel hundreds
		// of units) plus the fact that a strand belonging to a flat paint cell can
		// still sit on the adjacent cliff triangle. Filter here instead: `target` is
		// the face normal of the exact triangle this strand sits on (recomputed above
		// as the fix 1.2 grass-orientation workaround), so target.y < 0.7 kills the
		// strand only when its OWN triangle is steep — cliffs go grass-free, gentle
		// slopes keep grass, and paint cells needn't be mutated. Only fires for GG
		// grass entities (the xHairGrassType != 0u gate keeps upstream hair clean).
		if (target.y < (half)0.7)
		{
			strand_length = 0;
		}

		// GG-MAX Stage B.10 altitude filter: strands whose world Y falls outside
		// the configured band are hidden. Above-water and underwater use disjoint
		// [min, max] pairs so a single hair entity can represent, say, grass on
		// hills AND seaweed on the seabed without either half fighting the other.
		// base.y is the strand's world Y (mesh vertex position through the entity
		// transform), which for terrain-mounted grass IS the terrain height at
		// that strand's XZ — matches DX11 UpdateInstances line 447/452 semantics.
		if (base.y > xHairGrassWaterHeight)
		{
			if (base.y < xHairGrassMinHeight || base.y > xHairGrassMaxHeight)
			{
				strand_length = 0;
			}
		}
		else
		{
			if (base.y < xHairGrassMinHeightUnderwater || base.y > xHairGrassMaxHeightUnderwater)
			{
				strand_length = 0;
			}
		}
	}

	// GGMAX 1.74: per-type parameters. In per-type mode these collapse to the entity's own
	// uniforms, so the generated code is identical to before for every non-merged hair system.
	const bool gg_merged = (xHairGrassType == GG_HAIR_GRASS_MERGED);
	const float gg_length      = gg_merged ? xHairGrassTypes[gg_resolved_type].length       : xHairLength;
	const float gg_width       = gg_merged ? xHairGrassTypes[gg_resolved_type].width        : (atlas_rect.aspect * xHairAspect);
	const float gg_stiffness   = gg_merged ? xHairGrassTypes[gg_resolved_type].stiffness    : xHairStiffness;
	const float gg_drag        = gg_merged ? xHairGrassTypes[gg_resolved_type].drag         : xHairDrag;
	const float gg_viewdist    = gg_merged ? xHairGrassTypes[gg_resolved_type].viewDistance : xHairViewDistance;
	// Billboard count is STRUCTURAL: the vertex/index stride per strand is fixed at
	// xHairBillboardCount (the max across the chunk's types), so a strand whose type wants
	// fewer must still occupy its slots — the extras collapse to zero area below rather than
	// shortening the stride, which would corrupt every later strand's vertex range.
	const uint gg_billboards   = gg_merged ? min(xHairGrassTypes[gg_resolved_type].billboardCount, xHairBillboardCount) : xHairBillboardCount;

	float3 diff = GetCamera().position - base;
	const float distsq = dot(diff, diff);
	const bool distance_culled = distsq > sqr(gg_viewdist);

	// GGMAX 1.49b grass strand LOD (motion-clean rework of 1.49). Two mechanisms, both pure
	// functions of (strand id hash, camera distance) so a parked camera is bit-stable:
	//  - DROPS: half the strands (hash bit 0) drop around Step2Dist, half the remainder
	//    (hash bit 1) around Step4Dist — but each strand drops at its OWN radius, jittered
	//    ±15% by its hash. 1.49's single hard ring made a whole radius-band flip in lockstep
	//    while the camera moved = visible two-shade shimmer sweeping the mid field (user-
	//    reported); jitter atomizes the transition into sparse single-blade events.
	//  - WIDTH: survivors widen by a CONTINUOUS distance ramp (1 -> boost across the Step2
	//    zone, boost -> boost^2 across the Step4 zone) instead of 1.49's step change, so
	//    camera motion produces slow smooth growth, never a per-frame width pop (and no
	//    velocity spike for TAA).
	bool gg_lod_dropped = false;
	half gg_lod_boost = 1;
	// GGMAX 1.74: the host computes the LOD step distances from the SYSTEM's viewDistance, which
	// in merged mode is the max across the chunk's types. Rescale per strand to its own type's
	// view distance, or a type with a deliberately shorter range (GG halves it for FLOWER) would
	// keep its strands far past where the per-type build dropped them — measured as +1.8 pp
	// coverage on the pink-flower benchmark, i.e. visibly denser rather than identical.
	const float gg_lod_scale = gg_merged ? (gg_viewdist / max(xHairViewDistance, 1.0f)) : 1.0f;
	const float gg_lod_step2 = xHairGGLodStep2Dist * gg_lod_scale;
	const float gg_lod_step4 = xHairGGLodStep4Dist * gg_lod_scale;
	if (gg_lod_step2 > 0)
	{
		uint gg_lod_hash = DTid.x * 2654435761u;
		gg_lod_hash ^= gg_lod_hash >> 16;
		const float gg_dist = sqrt(distsq);
		// per-strand jitter in [0.85, 1.15] — each strand drops at its own radius inside the band
		const float gg_j2 = 0.85 + 0.3 * (float)((gg_lod_hash >> 2) & 1023u) / 1023.0;
		const float gg_j4 = 0.85 + 0.3 * (float)((gg_lod_hash >> 12) & 1023u) / 1023.0;
		const bool gg_has4 = gg_lod_step4 > 0; // hardening: step4<=0 must mean "no 4x stage", not "drop everywhere"
		if (gg_lod_hash & 1u)
			gg_lod_dropped = gg_dist > gg_lod_step2 * gg_j2;
		else if ((gg_lod_hash & 2u) && gg_has4)
			gg_lod_dropped = gg_dist > gg_lod_step4 * gg_j4;
		// EXACT coverage compensation (adversarial-review fix): survivors widen by
		// 1/survivorFraction, ramped over the SAME [0.85,1.15]*R window the drops occupy.
		// t = fraction of the band's droppers already gone at this distance (uniform jitter
		// makes that linear in d); each band loses half its remaining population, so the
		// exact per-band factor is hyperbolic 1/(1-0.5t), hitting exactly 2.0 per halving
		// (the earlier linear ramp over a wider window left a +16%/-26% coverage wiggle).
		const float gg_t2 = saturate((gg_dist / gg_lod_step2 - 0.85) / 0.3);
		const float gg_t4 = gg_has4 ? saturate((gg_dist / gg_lod_step4 - 0.85) / 0.3) : 0.0;
		const float gg_exact = (1.0 / (1.0 - 0.5 * gg_t2)) * (1.0 / (1.0 - 0.5 * gg_t4)); // 1 -> 4
		// the knob scales the widening: 2.0 = exactly coverage-neutral (the default),
		// below 2.0 = deliberate far-field thinning for extra perf
		gg_lod_boost = (half)(1.0 + (gg_exact - 1.0) * (max(xHairGGLodWidthBoost, 1.0) * 0.5));
	}

	// Frustum culling the whole strand at once:
	//	intentionally overestimated, to not disappear as soon in different views (shadow map, etc)
	ShaderSphere sphere;
	sphere.center = base;
	sphere.radius = gg_length; // GGMAX 1.74: cull radius must follow the strand's own type length
	//draw_sphere(sphere.center, sphere.radius);
	const bool visible = !distance_culled && !gg_lod_dropped && GetCamera().frustum.intersects(sphere);
		
	// Optimization: reduce to 1 atomic operation per wave
	const uint waveAppendCount = WaveActiveCountBits(visible);
	uint waveOffset;
	if (WaveIsFirstLane() && waveAppendCount > 0)
	{
		InterlockedAdd(indirectBuffer[0].IndexCountPerInstance, waveAppendCount * gfx_indexcount_per_strand, waveOffset);
	}
	waveOffset = WaveReadLaneFirst(waveOffset);

	// Append visible indices:
	if (visible)
	{
		uint prevCount = waveOffset + WavePrefixSum(gfx_indexcount_per_strand);
		uint i0 = index0;
		uint ii0 = prevCount;
		for (uint segmentID = 0; segmentID < xHairSegmentCount; ++segmentID)
		{
			for (uint billboardID = 0; billboardID < xHairBillboardCount; ++billboardID)
			{
				culledIndexBuffer[ii0++] = i0++;
				culledIndexBuffer[ii0++] = i0++;
				culledIndexBuffer[ii0++] = i0++;
				culledIndexBuffer[ii0++] = i0++;
				culledIndexBuffer[ii0++] = i0++;
				culledIndexBuffer[ii0++] = i0++;
			}
		}
	}
	
	// GG perf: a strand that isn't drawn this frame (distance/frustum culled, so NOT appended to the
	// culled index buffer above) still ran the full billboard-write + physics below — pure waste, its
	// output is never referenced by the draw. Skip that work for non-visible strands. The wave-atomic
	// index append above already ran for every lane, so bailing here is wave-safe. Exceptions: on
	// regenerate_frame every strand must build the static primitive buffer + seed its sim state. Drawn
	// (visible) strands are untouched, so the rendered grass is byte-identical; a culled strand's sim
	// state simply freezes until it re-enters view (imperceptible settle under wind, none when static).
	if (!visible && !regenerate_frame)
		return;

	half len = lerp(1, rng.next_float(), saturate(xHairRandomness)) * strand_length;
	len *= gg_length;
	len *= atlas_rect.size;
	len /= (half)xHairSegmentCount;
	float2 frame = float2(gg_width * xHairSegmentCount, 1) * len * 0.5;
	frame.x *= gg_lod_boost; // GGMAX 1.49: surviving far strands widen to preserve coverage
	const float segment_radius = max(frame.x, frame.y);

	//draw_line(base, base + tangent, float4(1, 0, 0, 1));
	//draw_line(base, base + target, float4(0, 1, 0, 1));
	//draw_line(base, base + binormal, float4(0, 0, 1, 1));

	half3 bend = 0;
	if (xHairFlags & HAIR_FLAG_CAMERA_BEND)
	{
		// Bend down to camera up vector to avoid seeing flat planes from above
		bend = GetCamera().up * (1 - saturate(dot(target, GetCamera().up))) * 0.8;
	}

	// Bottom vertices:
	half3x3 TBN = half3x3(tangent, normalize(target + bend), binormal);
	rng.init(uint2(xHairRandomSeed, DTid.x), 1); // reinit random for consistent billboard variation!
	for (uint billboardID = 0; billboardID < xHairBillboardCount; ++billboardID)
	{
		half siz = billboardID == 0 ? 1 : lerp(0.2, 1, rng.next_float());
		half rot = billboardID == 0 ? 0 : (rng.next_float() * PI);
		// GGMAX 1.74: a merged chunk's stride is the MAX billboard count across its types, so a
		// type that wants fewer collapses its surplus billboards to zero area. The rng calls above
		// still run unconditionally, which keeps the random sequence — and therefore every other
		// billboard's variation — identical to the per-type build.
		if (billboardID >= gg_billboards)
		{
			siz = 0;
		}
		half2 rot_sincos;
		sincos(rot, rot_sincos.x, rot_sincos.y);
		half3x3 variationMatrix = half3x3(
			rot_sincos.y * siz, 0, -rot_sincos.x,
			0, siz, 0,
			rot_sincos.x, 0, rot_sincos.y * siz
		);
		variationMatrix = mul(variationMatrix, TBN);
		
		for (uint vertexID = 0; vertexID < 2; ++vertexID)
		{
			half3 patchPos = HAIRPATCH[vertexID];
			float2 uv = patchPos.xy;
			uv = uv * float2(0.5, 0.5) + 0.5;
			uv.y = 1 - uv.y;
			patchPos.y += 1;

			// Sprite sheet UV transform:
			uv.xy = mad(uv.xy, atlas_rect.texMulAdd.xy, atlas_rect.texMulAdd.zw);

			// scale the billboard by the texture aspect:
			patchPos.xyz *= frame.xyx;

			// variation based on billboardID:
			patchPos = mul(patchPos, variationMatrix);

			float3 position = base + patchPos;

			if (xHairFlags & HAIR_FLAG_UNORM_POS)
			{
				position = inverse_lerp(geometry.aabb_min, geometry.aabb_max, position); // remap to UNORM
			}
			
			vertexBuffer_POS[v0] = float4(position, 0);
			// GGMAX 1.74: .w carries the strand's grass type to the pixel shaders so each blade
			// samples its own texture. The buffer is R8G8B8A8_SNORM and the VS reads only .xyz,
			// so this channel was genuinely unused. Encode as (type+1)/127 — an exact SNORM
			// integer step — and 0 keeps "no per-strand type" for stock hair.
			vertexBuffer_NOR[v0] = half4(target, gg_merged ? ((gg_resolved_type + 1u) / 127.0) : 0);
			vertexBuffer_UVS[v0] = uv.xyxy; // a second uv set could be used here
			
			if (distance_culled)
			{
				position = 0; // We can only zero out for raytracing geometry to keep correct prevpos swapping motion vectors!
			}
			vertexBuffer_POS_RT[v0] = float4(position, 0);

			v0++;
		}
	}
	
	const half dt = clamp(GetFrame().delta_time, 0, 1.0 / 30.0); // clamp delta time to avoid simulation blowing up

	const half gravityPower = xHairGravityPower;
	const half stiffnessForce = gg_stiffness;
	const half dragForce = gg_drag;
	const half3 boneAxis = target;
	const half boneLength = len;
	
	for (uint segmentID = 0; segmentID < xHairSegmentCount; ++segmentID)
	{
		// Identifies the hair strand segment particle:
		const uint particleID = DTid.x * xHairSegmentCount + segmentID;

		if (regenerate_frame)
		{
			float3 tail = base + boneAxis * boneLength;
			simulationBuffer[particleID].prevTail = tail;
			simulationBuffer[particleID].currentTail = tail;
		}
		
		float3 tail_current = simulationBuffer[particleID].currentTail;
		float3 tail_prev = simulationBuffer[particleID].prevTail;
		half3 inertia = (tail_current - tail_prev) * (1 - dragForce);
		half3 stiffness = boneAxis * stiffnessForce;
		half3 external = gravityPower * float3(0, -1, 0);
		half3 wind = sample_wind(tail_current, ((float)segmentID + 1) / (float)xHairSegmentCount);
		external += wind;
		
		float3 tail_next = tail_current + inertia + dt * (stiffness + external);
		half3 to_tail = normalize(tail_next - base);
		tail_next = base + to_tail * boneLength;

		//draw_sphere(tail_next, len);

		// Apply every force and collider:
		for (uint i = forces().first_item(); !distance_culled && (i < forces().end_item()); ++i)
		{
			ShaderEntity entity = load_entity(i);

			[branch]
			if (entity.layerMask & xHairLayerMask)
			{
				const float range = entity.GetRange();
				const uint type = entity.GetType();

				if (type == ENTITY_TYPE_COLLIDER_CAPSULE)
				{
					float3 A = entity.position;
					float3 B = entity.GetColliderTip();
					half3 N = normalize(A - B);
					A -= N * range;
					B += N * range;
					//if (DTid.x == 0)
					//{
					//	draw_sphere(A, range);
					//	draw_sphere(B, range);
					//}
					float3 C = closest_point_on_segment(A, B, tail_next);
					float3 dir = C - tail_next;
					float dist = length(dir);
					dir /= dist;
					dist = dist - range - segment_radius;
					if (dist < 0)
					{
						tail_next += dir * dist;
						to_tail = normalize(tail_next - base);
						tail_next = base + to_tail * boneLength;
					}
				}
				else
				{
					float3 closest_point = closest_point_on_segment(base, tail_next, entity.position);
					float3 dir = entity.position - closest_point;
					float dist = length(dir);
					dir /= dist;

					switch (type)
					{
						case ENTITY_TYPE_FORCEFIELD_POINT:
							tail_next += dt * dir * entity.GetGravity() * (1 - saturate(dist / range));
							to_tail = normalize(tail_next - base);
							tail_next = base + to_tail * boneLength;
							break;
						case ENTITY_TYPE_FORCEFIELD_PLANE:
							tail_next += dt * entity.GetDirection() * entity.GetGravity() * (1 - saturate(dist / range));
							to_tail = normalize(tail_next - base);
							tail_next = base + to_tail * boneLength;
							break;
						case ENTITY_TYPE_COLLIDER_SPHERE:
							dist = dist - range - segment_radius;
							if (dist < 0)
							{
								tail_next += dir * dist;
								to_tail = normalize(tail_next - base);
								tail_next = base + to_tail * boneLength;
							}
							break;
						case ENTITY_TYPE_COLLIDER_PLANE:
							dir = normalize(entity.GetDirection());
							dist = plane_point_distance(entity.position, dir, closest_point);
							if (dist < 0)
							{
								dir *= -1;
								dist = abs(dist);
							}
							dist = dist - segment_radius;
							if (dist < 0)
							{
								float4x4 planeProjection = load_entitymatrix(entity.GetMatrixIndex());
								const float3 clipSpacePos = mul(planeProjection, float4(closest_point, 1)).xyz;
								const float3 uvw = clipspace_to_uv(clipSpacePos.xyz);
								[branch]
								if (is_saturated(uvw))
								{
									tail_next -= dir * dist;
									to_tail = normalize(tail_next - base);
									tail_next = base + to_tail * boneLength;
								}
							}
							break;
						default:
							break;
					}
				}
			}
		}

		// Don't allow tail to go below the axis plane:
		float below_plane = plane_point_distance(base, boneAxis, tail_next);
		if (below_plane < 0)
		{
			tail_next -= boneAxis * below_plane;
		}

		// Store simulation:
		simulationBuffer[particleID].prevTail = tail_current;
		simulationBuffer[particleID].currentTail = tail_next;

		//draw_point(tail_next, 0.1, float4(0,1,0,1));
		//draw_line(base, tail_next, float4(1,0,0,1));
		
		half3 normal = to_tail;
		
		// Write out render buffers:
		//	These must be persistent, not culled (raytracing, surfels...)
		half3 normal_bend = normalize(normal + bend);
		binormal = cross(normal_bend, tangent);
		tangent = cross(binormal, normal_bend);
		TBN = half3x3(tangent, normal_bend, binormal);
		
		//draw_line(base, base + tangent, float4(1, 0, 0, 1));
		//draw_line(base, base + normal, float4(0, 1, 0, 1));
		//draw_line(base, base + binormal, float4(0, 0, 1, 1));
	
		rng.init(uint2(xHairRandomSeed, DTid.x), 1); // reinit random for consistent billboard variation!
		for(uint billboardID = 0; billboardID < xHairBillboardCount; ++billboardID)
		{
			half siz = billboardID == 0 ? 1 : lerp(0.2, 1, rng.next_float());
			half rot = billboardID == 0 ? 0 : (rng.next_float() * PI);
			// GGMAX 1.86: THE MERGED-GRASS OVER-DENSITY BUG. This is the CAP half of the segment
			// (vertexID 2..3); the ROOT half above (the loop at the top of this file) already
			// collapses a surplus billboard to zero size, and this one did not. The result was a
			// quad with a zero-width root and a FULL-WIDTH cap — a wedge with real screen area
			// instead of the intended nothing — drawn for every billboard a merged strand's own
			// type does not want. The two loops must agree or the collapse is only half applied.
			if (billboardID >= gg_billboards)
			{
				siz = 0;
			}
			half2 rot_sincos;
			sincos(rot, rot_sincos.x, rot_sincos.y);
			half3x3 variationMatrix = half3x3(
				rot_sincos.y * siz, 0, -rot_sincos.x,
				0, siz, 0,
				rot_sincos.x, 0, rot_sincos.y * siz
			);
			variationMatrix = mul(variationMatrix, TBN);
			
			for (uint vertexID = 2; vertexID < 4; ++vertexID)
			{
				half3 patchPos = HAIRPATCH[vertexID];
				float2 uv = patchPos.xy;
				uv = uv * float2(0.5, 0.5) + 0.5;
				uv.y = lerp((float)segmentID / (float)xHairSegmentCount, ((float)segmentID + 1) / (float)xHairSegmentCount, uv.y);
				uv.y = 1 - uv.y;
				patchPos.y += 1;
		
				// Sprite sheet UV transform:
				uv.xy = mad(uv.xy, atlas_rect.texMulAdd.xy, atlas_rect.texMulAdd.zw);
		
				// scale the billboard by the texture aspect:
				patchPos.xyz *= frame.xyx;

				// variation based on billboardID:
				patchPos = mul(patchPos, variationMatrix);
		
				float3 position = base + patchPos;
		
				if (xHairFlags & HAIR_FLAG_UNORM_POS)
				{
					position = inverse_lerp(geometry.aabb_min, geometry.aabb_max, position); // remap to UNORM
				}
			
				vertexBuffer_POS[v0] = float4(position, 0);
				vertexBuffer_NOR[v0] = half4(normal, gg_merged ? ((gg_resolved_type + 1u) / 127.0) : 0); // GGMAX 1.74: see above
				vertexBuffer_UVS[v0] = uv.xyxy; // a second uv set could be used here
			
				if (distance_culled)
				{
					position = 0; // We can only zero out for raytracing geometry to keep correct prevpos swapping motion vectors!
				}
				vertexBuffer_POS_RT[v0] = float4(position, 0);
		
				v0++;
			}
		}

		// Offset next segment root to current tip:
		base = tail_next;
		target = normal;
	}

	// Primitive buffer creation is done here instead of CPU to reduce CPU time spent in buffer creations:
	if (regenerate_frame)
	{
		uint i = index0;
		v0 = vertexID0;
		uint rootOffset = v0;
		uint capOffset = rootOffset + 2 * xHairBillboardCount;
		for (uint billboardID = 0; billboardID < xHairBillboardCount; ++billboardID)
		{
			for (uint segmentID = 0; segmentID < xHairSegmentCount; ++segmentID)
			{
				primitiveBuffer[i++] = rootOffset + 0;
				primitiveBuffer[i++] = rootOffset + 1;
				primitiveBuffer[i++] = capOffset + 0;
				primitiveBuffer[i++] = capOffset + 0;
				primitiveBuffer[i++] = rootOffset + 1;
				primitiveBuffer[i++] = capOffset + 1;
				rootOffset += 2;
				capOffset += 2;
				v0 += 2;
			}
			v0 += 2;
		}
	}
}
