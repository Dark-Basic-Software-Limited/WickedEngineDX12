#define OBJECTSHADER_COMPILE_PS
// GGMAX 3.35: the Super Quick rungs use a reduced layout so their vertex shader can stop
// exporting the tangent / atlas / ao_wet they never read. Same source file, so the rung
// shaders can never drift out of step with the stock one.
#if defined(GG_SQ_FLAT) || defined(GG_SQ_AMBIENT) || defined(GG_SQ_LIT)
#define OBJECTSHADER_LAYOUT_GG_SUPERQUICK
#else
#define OBJECTSHADER_LAYOUT_COMMON
#endif
#define TILEDFORWARD
#include "objectHF.hlsli"

