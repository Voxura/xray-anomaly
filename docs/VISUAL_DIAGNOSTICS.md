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
