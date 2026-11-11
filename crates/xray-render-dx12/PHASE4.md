# Phase 4 geometry ABI

`include/xray_render_dx12.h` defines the C/C++ boundary. Calls on one renderer
must be serialized. Pointers must refer to live memory of the documented size;
panic containment cannot make dangling pointers safe.

- `XRayVertex`: 44 bytes; POSITION at 0, NORMAL at 12, TEXCOORD at 24,
  TANGENT at 32. No tangent handedness component.
- Indices: use DXGI enum values **57** (u16) or **42** (u32), not 16/32.
  Empty geometry, incompatible stride, view-size overflow and out-of-range
  vertex indices return `XR_DX12_ERR_INVALID_ARGUMENT`.
- Mesh handles: opaque process-local u64, zero invalid. Stale and foreign
  handles return `XR_DX12_ERR_INVALID_HANDLE`. No handle recycling.
- `CameraConstants`: size/alignment 256; offsets 0/64/128/192/208.
  Matrices are column-major. The baseline vertex shader uses `view_proj`;
  the caller supplies the combined matrix. Camera updates are immutable snapshots.
- Root parameter 0: camera root CBV b0, all shader stages. Parameter 1:
  16 SRVs t0..t15, pixel-visible. Static sampler s0: linear wrap U/V/W.

Call create renderer, attach window, create mesh, update camera, begin frame,
clear, draw indexed, end frame, then eventually destroy mesh and renderer.
Drawing requires an active frame and a previously supplied camera. A second
begin, attach or resize during recording is rejected.

The first draw lazily compiles `shaders/geometry.hlsl` using the existing DXC
compiler discovery and creates a baseline PSO. This requires DXC and SM 6.6
support. Missing compiler, compilation or driver failures return the existing
`XR_DX12_ERR_D3D12`. This PSO draws triangle lists to the RGBA8 backbuffer with
normal visualization, no depth test, no culling and no texture sampling. The 16
texture descriptors are initialized null SRVs; texture/material upload remains
future work. It does not replace the older deferred/bindless signature or shaders.

Vertex/index data uses immutable upload-heap buffers (correctness-first; no
default-heap staging optimization yet). Draws retain mesh and camera COM
references until their submission fence completes, so destroying a mesh or
updating the camera during recording cannot invalidate an earlier draw.
Submission tracking precedes Present so presentation failure cannot prematurely
release resources. A failed Signal conservatively retains its resources until
renderer teardown.

## Verification

Full crate and hardware checks require the locked Cargo dependencies, DXC and
DX12 hardware. `cargo test --offline --locked --manifest-path crates/Cargo.toml
-p xray-render-dx12` is the normal check from the repository root.

The dependency-free `src/geometry_abi.rs` tests can also be compiled directly
with `rustc --edition 2024 --test`. They cover every field offset, alignment,
both index widths, invalid sizes/formats, index bounds, and stale/foreign handles.
The Phase 4 hardware test checks draw output by GPU readback, including camera
replacement and mesh destruction before submission; it must pass before claiming
the raster path works on hardware.
