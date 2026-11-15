# Visual diagnostics

The standalone Rust renderer lives under `crates/` in this repository.

## Live Anomaly data

Set `XRAY_ANOMALY_ROOT` to the directory containing the game archives and
`fsgame.ltx`, then run from the repository root:

```powershell
$env:XRAY_ANOMALY_ROOT = 'D:\path\to\Anomaly'
cargo test --manifest-path crates/Cargo.toml -p xray-engine --test visual_diagnostics live_material_and_sky_vfs_decode_upload -- --ignored --nocapture
```

This explicitly requested test fails if the data or DX12 device is unavailable.
It loads l01_escape, prints unlinked mesh counts and actor spawn position,
resolves current weather sky paths, and checks every unique diffuse texture and
both sky textures through the same VFS/decode/upload function used by the render
bridge. Failures identify path lookup, archive read, DDS decode, or GPU upload.
Successful uploads do not prove correct UVs, cubemap interpretation, camera
placement, or final level presentation. Runtime fallback warnings are limited to
32 distinct paths per process to prevent repeated per-frame log flooding.

The older `live_level_diagnostics` tests are now explicitly ignored by default
and require this same environment variable; they previously silently returned
success when a hardcoded `C:\Games` directory was absent. The new visual test
does not load savegames.

## Verified locally, 2026-11-15

- Translated/rotated camera projection: near/far D3D depth, positive forward W,
  negative behind-camera W, and correct left/right screen coordinates passed.
- Generated BC1 texture: real DX12 upload, alias normalization/cache identity,
  and separate missing-path versus decode errors passed.
- Camera lifetime: replaced forward-pass camera allocations remain retained
  until submission retirement. The bridge can replace these during HUD drawing.
- Existing sky GPU readbacks passed: procedural center `[10,38,127,255]`,
  dual-texture blend `[127,0,127,255]`, foreground occlusion center green and
  exposed corner red.
- Existing synthetic seven-stage render/physics integration passed.

No Anomaly asset root was supplied on this host; the live asset test was not run.
The reported all-gray materials, black live sky, and sliced live polygons are
therefore not claimed fixed. Do not change near planes or spawn coordinates
based only on these synthetic checks. The current camera matrix upload already
contains the earlier correction in commit `95a7775f`.

## Follow-up on bfbfb5d3 hardware feedback

The tested revision predates the UI/font/depth/sky changes in `f363839e`,
projection changes in `95a7775f`, and camera resource lifetime correction in
`12fcb00a`. The current `matrix_mul_44(proj, view)` is intentional: this helper
implements Fmatrix's reversed-argument multiplication, producing row-vector
`view * proj`. Reversing the call is not a justified black-screen fix.

Weapon sounds now read inherited `snd_shoot`, `snd_reload`, and `snd_holster`
from the mounted `configs/system.ltx` include tree. Layered shoot aliases use
`snd_1_layer` for the muzzle report; additional layers and random variations
are not implemented. Optional CSV volume/delay fields are not interpreted.
Inventory spec replacement reapplies these mappings; unchanged slots avoid
per-frame config resolution. Missing qualified sound paths no longer select
an unrelated PM/AK report. Synthetic sound fallbacks emit a diagnostic once per
cached fallback, including ambient candidates, so missing assets can be
separated from mixer problems. A failed config load emits a warning and retains
the default paths. DLTX override discovery is not added by this change.

Config inheritance, alias-cycle termination, mounted includes, and inventory
resynchronization have synthetic regression coverage. Local checks do not
establish correct final visuals or authentic audio on the Radeon R5 M330.
A current-build live run with real assets remains required; capture texture
failure and audio fallback diagnostics alongside screenshots and the active
weapon section. No ambient tuning is justified from the description alone.

Validation for the follow-up: 153 format, 45 renderer, 8 audio, 14 engine library,
and 3 synthetic DX12 integration/visual tests passed. The live asset test was
explicitly ignored. Release build passed. Strict Clippy remains blocked by 8
existing warnings in game_loop.rs, ui_inventory.rs, and world_loader.rs; those
unrelated files were not changed.
