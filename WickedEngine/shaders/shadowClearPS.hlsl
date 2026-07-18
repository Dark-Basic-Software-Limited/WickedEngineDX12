// GGMAX delayed shadow cascades: paired with screenVS (fullscreen triangle at
// z=0 = reversed-Z far) and a depth-ALWAYS write state, a scissored draw with
// this shader "clears" one shadow-atlas rect from inside the renderpass —
// needed because the delayed-cascade mode loads the atlas (LoadOp::LOAD) to
// preserve the cascades that are skipped this frame. The color output is the
// transparent shadow atlas clear value.
float4 main() : SV_TARGET
{
	return float4(1, 1, 1, 0);
}
