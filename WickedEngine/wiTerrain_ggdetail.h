#pragma once
// =================================================================================================================
// GGMAX PROTOTYPE (BR-0009 round 9): the round-8 near-camera detail ring for an EXTERNAL drawer (GameGuru's Terrain
// Bake), which tears the live terrain - and with it the live detail meshes - down and draws its own chunk copies.
// Same builder, tile/LOD rule, signature and generation-swap rule as the live ring (wiTerrain.cpp gg_UpdateDetailMeshes),
// driven by the same settings globals (gg_detail_ring / gg_detail_lod_*), but no Terrain and no scene entities: the host
// hands over the chunk set it draws (each with its full 40-unit perimeter) and a pack() that turns a built mesh into its
// own GPU buffers, on the build thread. See wiTerrain.cpp "EXTERNAL DETAIL RING".
// =================================================================================================================
#include "wiTerrain.h"

#include <functional>
#include <memory>
#include <shared_mutex>

namespace wi::terrain
{
	// The height-data fence. Every detail build (live ring and external ring) holds it SHARED while it samples the height
	// modifiers (per block of rows); GameGuru holds it EXCLUSIVE around the frees of arrays its height function reads
	// (GGTerrain_part0.cpp: the main heightmap swap in CheckParams, the flat-area array grow / reset).
	std::shared_mutex& gg_DetailHeightLock();

	struct GGDetailExtChunk
	{
		int cx = 0, cz = 0;
		XMFLOAT3 origin = XMFLOAT3(0, 0, 0); // the chunk's position (centre) - the vertex fold the host uses
	};
	struct GGDetailExtDesc
	{
		float chunk_scale = 1, bottomLevel = 0, topLevel = 1;
		wi::vector<std::shared_ptr<Modifier>> modifiers;  // copies taken while the live terrain existed
		wi::vector<GGDetailExtChunk> chunks;             // every chunk the host draws (full 40-unit perimeter each)
		// BUILD thread: chunk-local mesh (positions, normals, uvset_0, indices) -> the host's GPU object; nullptr = failed
		std::function<std::shared_ptr<void>(const XMFLOAT3& origin, const wi::scene::MeshComponent& mesh)> pack;
	};
	// one published detail mesh (immutable once published)
	struct GGDetailExtMesh
	{
		int cx = 0, cz = 0;
		uint64_t sig = 0;
		uint32_t verts = 0, tris = 0, unexpected = 0;
		float build_ms = 0;
		wi::vector<XMFLOAT2> border[4];   // world (along-edge, y) per edge 0:-x 1:+x 2:-z 3:+z - the seam instrument
		int openInside = -1, dupEdges = -1, badTris = -1;   // gg_detail_ext_check only (-1 = not checked)
		float areaErr = -1;
		std::shared_ptr<void> gpu;        // what pack() returned
	};
	struct GGDetailExtStats
	{
		bool on = false, frozen = false, failed = false;
		uint32_t published = 0, pending = 0, swaps = 0, builds = 0, threads = 0, poolLoad = 0;
		uint64_t tris = 0, verts = 0;
		float last_build_ms = 0, max_build_ms = 0, centre_x = 0, centre_z = 0;
	};
	struct GGDetailExtRing;

	GGDetailExtRing* gg_DetailExt_Create(GGDetailExtDesc desc);
	// main thread, once a frame, before rendering: true = the published set changed
	bool gg_DetailExt_Update(GGDetailExtRing* ring, const XMFLOAT3& eye);
	const wi::vector<std::shared_ptr<const GGDetailExtMesh>>& gg_DetailExt_Published(const GGDetailExtRing* ring);
	// cancel every build, build nothing more, keep the published set (the host's data went stale); no wait
	void gg_DetailExt_Freeze(GGDetailExtRing* ring);
	// Freeze, drain the external pool (running builds finish), delete
	void gg_DetailExt_Destroy(GGDetailExtRing* ring);
	void gg_DetailExt_GetStats(const GGDetailExtRing* ring, GGDetailExtStats& stats);
	// both-ways polyline comparison (the live seam instrument's): max height gap of each side's vertices on the other's
	// polyline, and the vertices with no partner on the other side (T-junctions)
	void gg_DetailPolyCompare(const wi::vector<XMFLOAT2>& A, const wi::vector<XMFLOAT2>& B, float* maxGap, int* tjunctions);
	extern bool gg_detail_ext_check;  // also check each built mesh topologically (instrument)
}
