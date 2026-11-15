# X-Ray Monolith — Pure-Rust Standalone Engine

A clean-room, pure-Rust reimplementation of the **X-Ray 1.6 / S.T.A.L.K.E.R. Anomaly (Monolith)** engine runtime (`crates/`), featuring a bindless **DirectX 12 (Shader Model 6.6)** 7-stage render graph, **pure-Rust Ogg Vorbis I / WASAPI 3D spatial audio**, native **3-tier P2P co-op multiplayer** (no third-party VPN drivers required), native VFS streaming of `.db*`/`.xdb*`/gamedata assets, first-person ballistic combat, interactive Inventory/PDA UI, and synchronized co-op level transitions.

> For a deep dive into crate boundaries, the render graph pass order, and state machines, see the **[Architecture Guide](docs/ARCHITECTURE.md)**.

---

## Table of Contents

- [Workspace Architecture](#workspace-architecture)
- [Prerequisites](#prerequisites)
- [Single-Command Build & Verification](#single-command-build--verification)
- [CLI Usage Guide](#cli-usage-guide)
- [Controls](#controls)
- [Native P2P Co-Op Multiplayer](#native-p2p-co-op-multiplayer)
- [Visual & Audio Subsystems](#visual--audio-subsystems)
- [Clean-Room Verification & Legacy Provenance](#clean-room-verification--legacy-provenance)

---

## Workspace Architecture

All engine code resides in the Cargo workspace under [`crates/Cargo.toml`](crates/Cargo.toml):

| Crate | Role |
| --- | --- |
| [`xray-core`](crates/xray-core/Cargo.toml) | Zero-copy LTX parser, multi-parent inheritance, **DLTX** mod diff/override/tombstone resolution, `#include`/wildcard loader, and concurrent `shared_str` string table. |
| [`xray-core-ffi`](crates/xray-core-ffi/Cargo.toml) | Optional C ABI `cdylib` wrapper around `xray-core` with generational handle safety. |
| [`xray-vfs`](crates/xray-vfs/Cargo.toml) | Win32 memory-mapped `.db*` / `.xdb*` Chunk 666 archive reader, pure-Rust **LZO1X** & **LZHuf** decompressors, `$game_*$` alias resolver, and `gamedata/` priority layering. |
| [`xray-formats`](crates/xray-formats/Cargo.toml) | Pure-Rust binary/config codecs for `level` / `level.geom` / `level.cform` (BVH + SAT), `level.hom` (`OccRasterizer`), `ogf` / `omf` / skinning, `details`, `particles.xr`, `shaders.xr`, `sun` CSM, `weather`, `lanims`, `gamemtl.xr`, `level.som` / `snd_static`, `ai_graph` / `game.graph`, `all.spawn` / `spawn_overlays`, `.scop` saves, `.scoc` Lua marshal state, UI XML DOM & bitmap fonts, and `world_maps`. |
| [`xray-render-dx12`](crates/xray-render-dx12/Cargo.toml) | DirectX 12 SM 6.6 bindless renderer, DAG Render Graph barrier compiler, transient heap aliasing, 3-split CSM shadows, deferred MRT + SSFX volumetric lighting, filmic postprocessing, weather/particles/wallmarks, and `UiCanvas`. |
| [`xray-net`](crates/xray-net/Cargo.toml) | Reliable UDP co-op protocol (`HostSession` / `ClientSession`), 32-bit ACK bitmask reliability, RFC 5389 STUN hole punching, XRRL fallback relay, and synchronous `LevelChangeNotice` handshakes. |
| [`xray-audio`](crates/xray-audio/Cargo.toml) | Pure-Rust OggS + Vorbis I stream decoder, `xr_sound` metadata parser, `level.som` occlusion + 3D spatial attenuation/panning, 48 kHz software mixer, and Windows WASAPI backend with headless fallback. |
| [`xray-engine`](crates/xray-engine/Cargo.toml) | Standalone runtime & binary (`xray-engine.exe`) combining Win32 windowing, input, 60 Hz capsule/BVH physics, ALife & co-op sync, HUD weapons & ballistics, Inventory/PDA/Dialogue UI, dynamic level transitions, audio, and DX12 rendering. |

---

## Prerequisites

- **OS:** Windows 10 (1903+) / Windows 11 (x86_64) with DirectX 12 (Feature Level 12_0 / Shader Model 6.6 capable GPU) and WASAPI audio.
- **Toolchain:** Rust stable with Edition 2024 support (Rust 1.85+; verified here with 1.98.0), the MSVC x64 linker and Windows SDK from Visual Studio Build Tools.
- **Runtime shaders:** DirectX Shader Compiler (`dxc.exe` and its companion DLLs) from the Windows SDK, discoverable via `DXC_PATH` (full executable path), SDK locations, or `PATH`. Required for rendered execution; headless mode skips DX12.
- **Game Assets (for live Zone play):** Standard S.T.A.L.K.E.R. Anomaly 1.5.x / Call of Pripyat `fsgame.ltx`, `db/` archives, and/or `gamedata/` directory. *(All workspace unit & integration tests synthesize self-contained in-memory/temp assets and require no external files to pass.)*

---

## Single-Command Build & Verification

Build the optimized release executable (`crates/target/release/xray-engine.exe`, thin LTO, `opt-level = 3`):

```powershell
cargo build --release --locked --manifest-path crates/Cargo.toml -p xray-engine --bin xray-engine
```

From `crates/`, the equivalent short command is `cargo build --release -p xray-engine`. Use `--offline` only with dependencies already cached. Headless execution below exercises the runtime loop without GPU rendering or a loaded game level.

Verify headless execution (60 frames):

```powershell
crates/target/release/xray-engine.exe --headless --max-frames 60
```

Run the full workspace test suite & strict Clippy verification:

```powershell
cargo clippy --offline --locked --manifest-path crates/Cargo.toml -p xray-audio -p xray-engine -p xray-core -p xray-vfs -p xray-formats -p xray-net --all-targets --no-deps -- -D warnings
cargo clippy --offline --locked --manifest-path crates/Cargo.toml -p xray-render-dx12 --lib --no-deps -- -D warnings
cargo test --offline --locked --manifest-path crates/Cargo.toml
```

---

## CLI Usage Guide

`crates/xray-engine/src/main.rs` supports classic X-Ray launch flags alongside standalone runtime options:

```text
xray-engine [OPTIONS]

OPTIONS:
  -fsltx <path>         Path to fsgame.ltx (default: ./fsgame.ltx if present)
  -level <name>         Level name or vpath (e.g. l01_escape or levels/l01_escape)
  -save <name>          Saved game name or .scop vpath (e.g. quicksave)
  -connect <ip[:port]>  Connect to co-op host (default port 5445)
  --host [port]         Listen as co-op host (default port 5445)
  --nickname <name>     Player nickname for co-op sessions (default: Stalker)
  --width <px>          Window width (default: 1280)
  --height <px>         Window height (default: 720)
  --no-vsync            Disable VSync swapchain present
  --headless            Create hidden window without DX12 initialization
  --max-frames <n>      Exit cleanly after rendering <n> frames
  -h, --help            Print help message
```

### Common Launch Examples

```powershell
# Launch Cordon from an existing Anomaly installation
crates\target\release\xray-engine.exe -fsltx D:\Anomaly\fsgame.ltx -level l01_escape

# Load a saved .scop game at 1920x1080 without VSync
crates\target\release\xray-engine.exe -fsltx D:\Anomaly\fsgame.ltx -level l01_escape -save quicksave --width 1920 --height 1080 --no-vsync

# Host a 2+ player co-op session in Zaton on UDP port 5445
crates\target\release\xray-engine.exe -fsltx D:\Anomaly\fsgame.ltx -level zaton --host 5445 --nickname Falcon

# Join the current CLI host over IPv4
crates\target\release\xray-engine.exe -fsltx D:\Anomaly\fsgame.ltx -level zaton -connect 192.168.1.42:5445 --nickname Honey
```

---

## Controls

| Input | Action |
| --- | --- |
| `W` / `A` / `S` / `D` | Move |
| `Shift` / `Ctrl` / `Space` | Sprint / Crouch / Jump |
| `LMB` / `RMB` | Fire / Aim Down Sights (ADS with FOV zoom & crosshair auto-hide) |
| `R` / `1`–`6` | Reload / Swap Weapon Slots 1–6 |
| `Tab` or `I` | Toggle Inventory & Equipment Grid (drag-and-drop equip, right-click consumables) |
| `P` | Toggle PDA (level minimap, co-op peer radar, active story quests) |
| `E` | Interact (NPC/Trader dialogue, item pickup, confirm Level Transition) |
| `Esc` | Dismiss active modal / menu |

---

## Native P2P Co-Op Multiplayer

`xray-net` provides built-in P2P networking with **zero third-party virtual LAN drivers (no Radmin/Hamachi required)**:

The library exposes three transport tiers; the shipped CLI currently wires direct sessions only:

1. **Tier 1 — IPv6 direct UDP:** supported by `TransportConfig` and the session API. An IPv6 client is selected by an IPv6 `-connect` address, but `--host` currently calls `HostSession::new(port, false)` and binds IPv4. IPv6 hosting requires library integration.
2. **Tier 2 — STUN / UDP hole punching:** `HolePunchCoordinator` implements endpoint discovery and punching. The binary does not invoke it or expose rendezvous configuration. Success depends on NAT behavior; arbitrary CGNAT traversal is not guaranteed.
3. **Tier 3 — XRRL relay:** `RelayServer` / `RelayClient` implement relay datagrams. A reachable relay and library integration are required; the binary has no relay flag or automatic fallback.

For the available CLI path, host with `--host 5445` and join with `-connect <host IPv4>:5445`, using reachable UDP endpoints and matching game assets. Internet connectivity may require existing firewall/router configuration; no public rendezvous or relay service is bundled. WAN traversal was not verified by the headless smoke check.

Co-op gameplay synchronization includes:
- Reliable sequenced & 32-bit ACK-bitmask delivery alongside unreliable entity transform deltas (`EntityDelta`).
- Cross-player `.scoc` inventory/avatar state (`RemotePeerAvatar` blips on the PDA radar and rendered in-world avatars).
- Synchronous **Level Transition Handshake** (`PacketType::LevelChangeNotice`: `PrepareTransition` → `ReadyAtDestination` → `CommitSpawn` / `Abort`) so host and clients save `.scoc`/`.scop` state, load destination assets, and spawn together.

---

## Visual & Audio Subsystems

### DirectX 12 7-Stage Render Graph (`xray-render-dx12` + `render_bridge.rs`)
- **SM 6.6 Bindless Descriptor Indexing**: Zero per-draw descriptor table churn across world/dynamic meshes and textures (`DXT1`, `DXT5`, `ATI1`/`BC4`, `ATI2`/`BC5`, `BGRA8`).
- **Pass Pipeline**: Early-Z Depth Prepass → G-Buffer MRT → 3-Split Cascaded Shadow Map (`CascadedShadowAtlas`) → Deferred Combine (**SSFX 16-step volumetric sun shafts** + dynamic `GpuLight`s) → Filmic Postprocess (`Bloom` + `Uncharted2`/`ACES`/`Reinhard`) → Depth-tested Forward Sky/Clouds/Grass Details/Bullet Wallmarks/Rain/Sparks + dedicated **HUD projection weapon pass** → `UiCanvas` overlay & Loading Splash screen.

### Pure-Rust 3D Spatial Audio (`xray-audio`)
- **Streaming Vorbis Decoder**: Pure-Rust `OggS` demuxer + Vorbis I decoder (Floor 0/1, Residue 0/1/2, polar coupling, FFT IMDCT) reading directly from `xray-vfs` plus `xr_sound` OGG comment chunks (`min_dist`, `max_dist`, `base_volume`, `game_type`).
- **3D Spatial Mixer**: Listener-relative distance attenuation, equal-power stereo panning, `level.som` acoustic occlusion, Doppler shift, and soft `tanh` limiting mixed at 48 kHz onto Windows **WASAPI** shared mode (with automatic `HeadlessNull` fallback for CI/headless runs).
- **Gameplay Triggers**: Material-aware footsteps/landings (`gamemtl.xr`), weather wind/rain/3D thunder, `level.snd_static` ambient loops, 3D weapon discharge & bullet impact audio, and UI/consumable feedback.

---

## Clean-Room Verification & Legacy Provenance

- The standalone dependency graph uses Rust workspace crates and Microsoft's `windows` crate family; `xray-core-ffi` is not an engine dependency. No `build.rs`, `.c`, or `.cpp` files were found under `crates/`. The repository still contains legacy C++ source outside that workspace.
- These checks establish build/link separation, not clean-room provenance. Establishing independent authorship requires an auditable specification/source-access history; it cannot be inferred from Rust syntax or a successful build. Existing code and documentation reference canonical C++ behavior.
- Runtime independence still includes Windows system DLLs and DXC for shader compilation; game assets are separate and are not bundled. See the [Phase E verification record](docs/PHASE_E_VERIFICATION.md) for the measured checks and packaging limits.
