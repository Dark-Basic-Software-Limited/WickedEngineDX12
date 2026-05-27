# CLAUDE.md

Read [STATE.md](STATE.md) first for current project state, recent work, and MVP roadmap. That file is the source of truth for what has been done and what is left.

This file holds **durable conventions** for working in this codebase — facts that should not drift as features land.

## Codebase conventions

- This is a fork of Wicked Engine. Engine source lives in `WickedEngine/`. Editor app lives in `Editor/`. Shaders live in `WickedEngine/shaders/`.
- Engine code uses the `wi::` namespace, with files named `wiX.h` / `wiX.cpp` (e.g. `wiRenderer`, `wiHairParticle`, `wiTerrain`).
- Shaders are HLSL, compiled at runtime by `dxcompiler.dll`. Shared CPU/GPU layouts live in `shaders/ShaderInterop_*.h`.
- Prefer editing existing engine code over forking subsystems. Where this fork diverges from upstream, the divergence is intentional and should be preserved unless the user says otherwise.

## Integration surface for GGMAX

The custom draw callbacks are the **only** sanctioned integration point for GameGuru MAX's renderer to inject draws into the engine's frame. Do not bypass them with ad-hoc render-loop edits.

- Per-pass callbacks (declared in [wiRenderPath3D.h](WickedEngine/wiRenderPath3D.h)):
  `customDraw_Prepass`, `customDraw_Prepass_Reflections`, `customDraw_AfterPrepass`, `customDraw_Opaque`, `customDraw_Transparent`, `customDraw_Compose`.
- Global renderer callbacks (declared in [wiRenderer.h](WickedEngine/wiRenderer.h)):
  `customDraw_ShadowMap` (per cascade), `customDraw_EnvProbe` (per cubemap face set).

Each callback supplies the frustum (or sphere + frustum array for env probes) and the active `CommandList`. Use these for culling and for emitting GGMAX-side draws.

## DX12-specific care

- **Strip-cut / primitive restart**: triangle/line strip PSOs in the DX12 backend explicitly enable primitive restart with index value `0xFFFF`. Do not regress this; terrain rendering relies on it.
- **Descriptor binder limits** (Phase 2, in [wiGraphicsDevice.h:27](WickedEngine/wiGraphicsDevice.h)):
  `DESCRIPTORBINDER_SRV_COUNT = 64`, `DESCRIPTORBINDER_SAMPLER_COUNT = 16`. GGMAX custom shaders are sized against these. If you need to raise them further, check the Vulkan pool sizes in `wiGraphicsDevice_Vulkan.cpp` and the root signature in `globals.hlsli` together.

## Performance work

Use the built-in profiler — do not introduce a parallel timing path.

- CPU: `wi::profiler::ScopedCPUProfiling(name)`
- GPU: `wi::profiler::ScopedGPUProfiling(name, cmd)`
- On-screen HUD: `wi::profiler::DrawData(canvas, x, y, cmd)`
- Toggle: `wi::profiler::SetEnabled(true)`

GPU event markers go through `GraphicsDevice::EventBegin/End()` and are visible in PIX and RenderDoc.

Before any optimization commit, **capture a baseline** (frame time, GPU/CPU breakdown on a representative scene) so the impact of the change is measurable.

## Things to leave alone unless asked

- The `bc5dbfea` "GGMAX TEMP Tweaks" HEAD commit — see STATE.md for why.
- The `DEBUG_NORMAL_VIS` shader option — retained for ongoing migration validation.
- Pre-existing TODOs in [wiGraphicsDevice_DX12.cpp](WickedEngine/wiGraphicsDevice_DX12.cpp) (Xbox mesh shader, H.265 video decode, sparse update queue) — these are upstream platform stubs, not this fork's work.

## Workflow

- The user's primary worktree is at `D:\max\WickedEngineDX12\.claude\worktrees\nice-mclaren` on `claude/nice-mclaren`. The main repo (`D:\max\WickedEngineDX12`) tracks `master`.
- After landing a themed batch of work, update STATE.md (bump "Last updated", refresh "Recent themed work", move completed MVP items into the readiness table).
- Do not commit unless the user asks.
