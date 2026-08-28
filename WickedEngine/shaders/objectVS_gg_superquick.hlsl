// ★★★ GGMAX 3.35: the Super Quick vertex shader.
//
// Identical to objectVS_common.hlsl except for the layout. The position chain -
// GetPositionWind() -> mul(instance.transform) -> sample_wind - is the same source through the
// same VertexSurface::create, and SV_Position is `precise`, which is what already lets the
// stock prepass VS and the stock main VS (two different binaries) agree exactly enough for the
// colour pass's ComparisonFunc::EQUAL depth test.
//
// ⚠ Do not be tempted to reuse objectVS_simple here. It drops the provoking index buffer,
// which changes primitive order, which is precisely the depth mismatch the reorder exists to
// prevent - and the symptom would be black geometry, not a warning.
#define OBJECTSHADER_COMPILE_VS
#define OBJECTSHADER_LAYOUT_GG_SUPERQUICK
#include "objectHF.hlsli"
