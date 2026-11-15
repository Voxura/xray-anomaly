# X-Ray Monolith Pure-Rust Engine — Architecture Guide

This document describes the architecture, subsystem boundaries, data flow, and verification limits of the **Pure-Rust Standalone X-Ray Monolith Engine** located under [`crates/`](../crates/Cargo.toml).

---

## 1. Design Principles & Clean-Room Verification

The Rust workspace targets a standalone runtime for X-Ray assets. Build separation and clean-room provenance are distinct claims.

- **Source/build inspection:** no `build.rs`, `.c`, or `.cpp` files were found under `crates/`. The engine dependency graph excludes the optional `xray-core-ffi` wrapper and legacy C++ targets.
- **Dependencies:** platform bindings use Microsoft's `windows` crate family and Rust `std`. Windows system libraries remain required. Rendered execution invokes external `dxc.exe` to compile HLSL; a standalone executable is not a dependency-free distribution.
- **Provenance limit:** source and binary checks cannot certify clean-room authorship or licensing. Canonical C++ references occur in implementation history. Independent specification and source-access records would be needed to substantiate that stronger claim.
- **Networking integration limit:** direct IPv4 hosting is wired in the CLI. IPv6 hosting, STUN coordination and XRRL relay integration are library capabilities, not an automatic CLI fallback chain. See the README for operational constraints.

Current evidence and distribution instructions are recorded in [Phase E verification](PHASE_E_VERIFICATION.md).

---

## 2. Workspace Crate Graph

```text
                        +--------------------+
                        |    xray-engine     |  Standalone executable & runtime loop
                        +---------+----------+
                                  |
       +-------------+------------+------------+-------------+
       |             |                         |             |
       v             v                         v             v
+-------------+ +----------+           +---------------+ +-------------+
|  xray-net   | |xray-audio|           |xray-render-dx12| |xray-formats |
+------+------+ +----+-----+           +-------+-------+ +------+------+
       |             |                         |                |
       |             +------------+------------+----------------+
       |                          |
       v                          v
+-------------+           +---------------+      +---------------+
|  xray-core  |<----------+   xray-vfs    |      | xray-core-ffi | (optional C ABI)
+-------------+           +---------------+      +---------------+
```

### Crate Responsibilities

| Crate | Purpose & Key Modules |
| --- | --- |
| `xray-core` | Zero-copy LTX parser, multi-parent section inheritance, DLTX differential override/tombstone resolution (`depth` + insertion ordering), recursive `#include` / wildcard LTX tree loader (`ltx_loader`), concurrent `shared_str` interning (`string_table`). |
| `xray-core-ffi` | Optional `cdylib` / `rlib` C ABI wrapper over `xray-core` (`LtxDocument` handle table with generational stale-handle guards). Unused by `xray-engine` itself. |
| `xray-vfs` | Win32 memory-mapped archive backing (`mmap`), X-Ray `.db*` / `.xdb*` Chunk 666 archive reader (`archive`), bounds-checked pure-Rust LZO1X (`lzo`) & LZHuf (`lzhuf`) decompressors, priority `gamedata/` -> `.xdb*` -> `.db*` virtual filesystem (`vfs`) with `$game_*$` alias normalization and traversal guards. |
| `xray-formats` | Clean-room parsers & CPU runtimes for X-Ray asset formats: `level` / `level.geom` / `level.geomx`, `cform` BVH + SAT box queries, `hom` + `OccRasterizer`, `ogf`, `omf`, `skinning`, `details` + SSFX grass benders, `particles` / `particle_sim`, `shaders_xr`, `sun` CSM hulls, `lighting`, `postprocess`, `weather`, `lanims`, `sound_env`, `gamemtl`, `ai_graph`, `spawns` + `spawn_overlays`, `scop` / `savegames` / `scoc` Lua marshal codec, `thm`, `fonts`, `xml_ui`, `world_maps`, and `vfs_bridge`. |
| `xray-render-dx12` | DirectX 12 SM 6.6 bindless renderer: `device`, `swapchain`, `graph` DAG barrier compiler, `heaps` transient resource aliasing, `allocators` fenced command ring, `geometry` / `geometry_abi`, `bindless`, `texture` (DXT1/DXT5/ATI1/ATI2/BGRA8), `deferred` G-Buffer MRT + combine, `shadow` 3-split CSM atlas, `sky`, `clouds`, `details`, `wallmarks`, `rain`, `particle`, `lighting`, `postprocess` (bloom, Uncharted2/ACES/Reinhard), and `ui_canvas` (rects, textured atlases, font glyphs). |
| `xray-net` | Pure-Rust co-op networking (`protocol`, `reliability` seq/ack-32bit bitmask, `transport` UDP IPv4/IPv6, `stun` RFC 5389 binding & UDP hole punching, `relay` XRRL fallback relay, `session` `HostSession` / `ClientSession` including `LevelChangeNotice` handshake). |
| `xray-audio` | Pure-Rust OggS + Vorbis I stream decoder & `xr_sound` metadata reader (`ogg_vorbis`), 3D attenuation/panning/`level.som` occlusion (`spatial`), 48 kHz resampling mixer + `tanh` limiter (`mixer`), Windows WASAPI MTA output + `HeadlessNull` backend (`backend`), and gameplay `AudioManager` (`manager`: `gamemtl.xr` footsteps, weather, static sounds, weapon/ALife 3D emitters, UI SFX). |
| `xray-engine` | Standalone executable (`main.rs`) and runtime coordinator (`game_loop.rs`): `window`, `input`, `camera`, `timer` (60 Hz fixed accumulator), `physics` (`StalkerController` capsule + stair-step + BVH), `world_loader`, `alife_manager`, `hud_weapon`, `ui_inventory`, `ui_pda`, `level_transition`, and `render_bridge`. |

---

## 3. Frame Execution & 7-Stage DirectX 12 Render Graph

Inside `EngineRuntime::step_frame` (`crates/xray-engine/src/game_loop.rs` + `render_bridge.rs`):

1. **Input & Modal Routing**: Pump Win32 `WindowEvent`s. When Inventory (`Tab`/`I`), PDA (`P`), NPC Dialogue (`E`), or Level Travel Prompt is active, the cursor unlocks, combat inputs are masked via `EngineInput::without_combat_inputs()`, and pointer/key clicks route to UI modals.
2. **Fixed-Timestep Simulation (60 Hz accumulator)**:
   - `StalkerController` capsule gravity, wall sliding, stair step-up, and ground material query against `CformBvh` + `gamemtl.xr`.
   - `HudWeaponController` state machine (`Idle`, `Aiming`, `Firing`, `Reloading`, `Holstering`, `Unholstering`), ADS FOV zoom, RPM cadence, multi-layer `CformBvh` ballistic penetration + live `AlifeManager` hitboxes, wallmark decal projection, sparks, and muzzle flash `GpuLight`.
   - `WorldWeatherState` cycle evaluation (`MixedEnvironment`, thunderbolts, dynamic rain drops, animated lights).
   - `AlifeManager` entity updates, `.scop`/`.scoc` state tracking, and `xray-net` entity deltas / level-change handshake packets.
   - `LevelTransitionManager` trigger-volume containment checks (`AlifeLevelChangerState` `Sphere` and oriented 4x3 `Box` shapes) and multi-stage level loading state machine.
3. **7-Stage DirectX 12 Render Graph (`RenderBridge::render_frame_with_transition`)**:
   - **(a) Early-Z Depth Prepass**: populates hardware depth buffer (`D32_FLOAT`) for Hi-Z / forward occlusion.
   - **(b) G-Buffer MRT Pass**: writes static world geometry, `LoadedDynamicModel` entities, and remote co-op peer avatars into multiple render targets.
   - **(c) 3-Split Cascaded Shadow Map (CSM) Pass**: builds stabilized frustum cascades (`build_3split_csm_atlas`) into `CascadedShadowAtlas` (`1024` px default per cascade).
   - **(d) Deferred Lighting Combine**: runs full-screen combine (`execute_deferred_combine_csm`) with CSM PCF filtering, 16-sample **SSFX volumetric sun shafts** + Henyey-Greenstein phase, hemispheric sky lighting, and up to 64 active point/spot `GpuLight`s (campfire/muzzle/ALife).
   - **(e) Filmic Postprocessing Resolve**: applies luminance/bloom adaptation + filmic tonemapping (`Uncharted2`, `ACES`, `Reinhard`) onto the swapchain RTV before transparent forward passes.
   - **(f) Forward Alpha, Environment & First-Person HUD Weapon Pass**: executes depth-tested `draw_sky`, `draw_clouds`, instanced `draw_details` grass, `draw_wallmarks` bullet holes/blood, `draw_rain` streaks, `draw_particles` sparks, followed by depth-range HUD clear + `hud_projection_matrix` first-person weapon & hands viewmodel pass (preventing near-wall clipping).
   - **(g) `UiCanvas` Overlay Pass**: draws crosshair (auto-hidden in ADS or modal UI), health/stamina/bleeding/radiation/ammo HUD meters, Inventory 10x12 grid + equipment/trader dialogue, PDA minimap (`WorldMapsConfig` + co-op peer/quest blips), Level Travel Prompt, or full-screen Loading Splash (`intro_<level>.dds`, level lore, 4-stage progress pips & bar).
4. **3D Spatial Audio Update (`AudioManager`)**: syncs `AudioListener` to `FirstPersonCamera`, triggers material footsteps, weather wind/rain/3D thunder, ambient static emitters, weapon shots/impacts, and UI sounds via WASAPI.

---

## 4. Level Transition & Co-Op Sync State Machine

`crates/xray-engine/src/level_transition.rs` + `crates/xray-net`:

1. When all actors step inside an `AlifeLevelChangerState` trigger volume, a cursor-unlocked `TravelPrompt` offers `E` (Confirm) or `Esc` (Decline).
2. Confirming initiates `TransitionState` progression:
   - `SavingPlayerState`: exports actor inventory/consumables/equipment via `InventoryManager::export_to_scoc()` and preserves `AlifeManager` `.scop` story quests & peer slots. Broadcasts `LevelChangeNotice { phase: PrepareTransition }`.
   - `UnloadLevel`: tears down old `LoadedWorld`, level changers, and world audio emitters; waits/exchanges `ReadyAtDestination` across connected co-op peers (`HostSession` / `ClientSession`).
   - `LoadAssets`: mounts destination `WorldLoader::load_from_vfs`, destination `level.cform` BVH, weather cycles, `AlifeManager` spawns, and re-discovers destination changers; places capsule cleanly on destination spawn height resolved against `CformBvh`.
   - `Ready` -> `Idle`: emits `CommitSpawn`, settles physics on ground, syncs GPU meshes/textures (`sync_world_to_gpu`), and resumes gameplay seamlessly across peers.
