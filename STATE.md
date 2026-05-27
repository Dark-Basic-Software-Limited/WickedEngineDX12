# Project State

> Snapshot of where the project stands. Re-read this first when picking up work after a break.
> Last updated: 2026-05-27.

## What this is

A fork of [Wicked Engine](https://github.com/turanszkij/WickedEngine) maintained as a DirectX 12 port with integration hooks for GameGuru MAX. The fork's purpose is to provide a modern rendering backend (DX12) and a defined integration surface (custom draw callbacks + expanded descriptor binder) so GGMAX can render its own content alongside the engine.

The MVP target is a build suitable for **limited public testing**, with the three remaining feature gaps being grass scattering, tree rendering, and a performance pass.

## Tech stack

| Layer | Choice |
|---|---|
| Language | C++17 |
| Build (Windows) | MSBuild via `WickedEngine.sln` (Visual Studio 2022+) |
| Build (Linux) | CMake |
| Graphics API | DirectX 12 (primary), Vulkan (secondary, cross-platform) |
| Shader language | HLSL, compiled at runtime via `dxcompiler.dll` |
| GPU memory | D3D12MemAlloc · Vulkan Memory Allocator |
| Physics | Jolt (in `WickedEngine/Jolt/`) |
| Audio | XAudio2 + Miniaudio |
| Scripting | Lua (via `wi::lua`) |
| CI | GitHub Actions (`.github/workflows/`) — Windows + Linux nightly + PR |

## Current branch & HEAD

- Active branch: `claude/nice-mclaren`
- HEAD: `bc5dbfea` — "GGMAX TEMP Tweaks" (2026-04-03)
- The branch carries the DX12-port + GGMAX integration work described below.

## Recent themed work

Most recent first. Each line is a logical batch, not always a single commit.

- **`bc5dbfea` GGMAX TEMP Tweaks** — animation system rough integration, `DEBUG_NORMAL_VIS` shader option for migration validation, perf tuning (45 fps → 70 fps, targeting 90). Flagged as TEMP; see "Known TEMP items" below.
- **`acc11cdb` Compose-stage hooks** — added `customDraw_AfterPrepass` (virtual-texture readback point) and `customDraw_Compose` (overlay/UI integration point) callbacks.
- **`6330ab46` DX12 strip-cut fix** — enabled primitive restart (`0xFFFF` cut value) on triangle/line strip PSOs in the DX12 backend. Resolved terrain spike artifacts.
- **`0a8ea4fb` Shadow + env probe hooks** — `customDraw_ShadowMap` (per cascade) and `customDraw_EnvProbe` (per cubemap face set) callbacks added to the global renderer.
- **`9118befe` Phase 2 + Phase 3 integration surface** — descriptor binder expanded (SRV 16→64, sampler 8→16) and four per-pass custom draw hooks added: `customDraw_Prepass`, `customDraw_Prepass_Reflections`, `customDraw_Opaque`, `customDraw_Transparent`.

Earlier commits cover the DX12 port itself, profiler safety, shader cache, and build script setup.

## MVP readiness

| Area | State | Key files |
|---|---|---|
| Terrain | Working in DX12 (strip-cut fix landed) | [wiTerrain.cpp](WickedEngine/wiTerrain.cpp) · [wiTerrain.h](WickedEngine/wiTerrain.h) |
| Grass (hair particle system) | Present, DX12-capable, integrated per terrain chunk | [wiHairParticle.cpp](WickedEngine/wiHairParticle.cpp) · [wiHairParticle.h](WickedEngine/wiHairParticle.h) |
| Trees (impostor + instancing) | PSOs + `RefreshImpostors()` in place; no scatter/placement layer yet | [wiRenderer.cpp](WickedEngine/wiRenderer.cpp) · `RefreshImpostors()` called from [wiRenderPath3D.cpp](WickedEngine/wiRenderPath3D.cpp) |
| Custom draw hooks | All 8 wired: `Prepass`, `Prepass_Reflections`, `AfterPrepass`, `Opaque`, `Transparent`, `Compose`, `ShadowMap`, `EnvProbe` | [wiRenderPath3D.h](WickedEngine/wiRenderPath3D.h) · [wiRenderer.h](WickedEngine/wiRenderer.h) |
| Descriptor binder | SRV 64, sampler 16 (Phase 2 limits) | [wiGraphicsDevice.h:27](WickedEngine/wiGraphicsDevice.h) |
| Profiler | CPU/GPU timing, on-screen HUD, PIX/RenderDoc event markers | [wiProfiler.h](WickedEngine/wiProfiler.h) · [wiProfiler.cpp](WickedEngine/wiProfiler.cpp) |
| Build | `build_wicked.bat`, `build_release.cmd`, CMake; CI green on Windows + Linux | — |

## Known TEMP items (kept deliberately — do not clean up yet)

- **`bc5dbfea` "GGMAX TEMP Tweaks" HEAD commit.** Contains animation-system tweaks bundled with perf tweaks and the debug option below. Left as a single commit until the animation migration is fully validated; do not amend or split until the user explicitly asks for it.
- **`DEBUG_NORMAL_VIS` shader option.** Lives in [wiRenderer.cpp](WickedEngine/wiRenderer.cpp) (~L4323), [objectHF.hlsli](WickedEngine/shaders/objectHF.hlsli) (~L570), and `OPTION_BIT_DEBUG_NORMAL_VIS` in [ShaderInterop_Renderer.h](WickedEngine/shaders/ShaderInterop_Renderer.h) (~L1243). Defaults to `false` — zero runtime cost when off — and is intentionally retained to validate ongoing animation/mesh migration work. Do not remove until the migration is signed off.

## MVP remaining work

### 1. Grass scattering

Backend is ready: [wiHairParticle](WickedEngine/wiHairParticle.cpp) implements strand-based particles with wind/stiffness/drag, and [wiTerrain](WickedEngine/wiTerrain.cpp) already owns a `grass_properties` `HairParticleSystem` plus per-chunk `ChunkData::grass`. The missing piece is the **scatter/placement layer**: density mask, slope/region/noise rules, view-distance culling, and editor controls for authoring it. Some of these knobs exist in upstream — verify which are wired here before adding new ones.

### 2. Trees

Pipeline groundwork is present: impostor capture/render PSOs and `RefreshImpostors()` exist in the renderer. The missing pieces are:

- A **placement system** (instance scatter on terrain, similar in spirit to grass but mesh-instance-driven).
- **LOD cross-fade** between full mesh, low-poly mesh, and impostor billboard.
- **Wind animation** on tree meshes (the hair particle wind source is reusable, but tree vertex animation is separate).
- Authoring/import path so GGMAX can supply tree assets via the standard mesh component.

Integration into the frame should go through the existing custom draw hooks rather than bypassing them.

### 3. Performance tuning

Profiler is ready ([wi::profiler](WickedEngine/wiProfiler.h) — `ScopedCPUProfiling` / `ScopedGPUProfiling`, on-screen HUD, PIX/RenderDoc events). Recent work has lifted the working scene from ~45 fps to ~70 fps; the target for limited public testing is around 90 fps. Approach: establish a fresh baseline on a representative scene, capture a PIX trace, identify GPU and CPU hotspots, then iterate. No "blind" optimization until baselines are recorded.

## How to build and run

- **Windows / debug**: `build_wicked.bat` (defaults to Debug — pass `Release` for release).
- **Windows / release**: `build_release.cmd`.
- **Linux**: standard CMake — `cmake -S . -B build && cmake --build build -j`.
- **Run**: the flagship target is `Editor_Windows` from `WickedEngine.sln`. Sample apps include `Template_Windows`, `Tests`, and the two ImGui examples.

## When you finish a chunk of work

Update this file. Specifically: bump "Last updated", refresh "Recent themed work" with the new themed batch, and move any MVP items that have landed out of "remaining work" into the readiness table.
