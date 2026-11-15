# Phase E release verification

Verified 2026-11-15 on Windows x64, NVIDIA GeForce RTX 4060 present; Rust/Cargo 1.98.0. Source revision before documentation changes: aa95dd80.

## Release checks

- `cargo build --release --manifest-path crates/Cargo.toml -p xray-engine --bin xray-engine`: exit 0, finished in 1m 24s, no compiler warnings emitted. Existing profile: opt-level 3, thin LTO, one codegen unit, debuginfo stripped.
- `crates/target/release/xray-engine.exe --headless --max-frames 60`: exit 0, 60 frames stepped, no warning lines emitted. No game assets or DX12 rendering exercised; timing is not a GPU benchmark.
- Executable: 2,638,848 bytes; SHA-256 `1caf6923b0a5ad9232967648c046a958185e18d2e76973bfc442fec0de5fe9b5`.
- PE import-directory inspection found Windows/DirectX/Winsock/COM, UCRT API sets and `VCRUNTIME140.dll`; no legacy xr*, Lua or OpenAL imports. This is a static import check, not a complete runtime dependency trace.
- Workspace source inventory found no build.rs, .c or .cpp files. `cargo tree -p xray-engine --edges normal` contains local Rust crates, the windows crate family and Rust proc-macro dependencies; no xray-core-ffi or legacy C++ target.
- No new full-suite or rendered-level validation is claimed by this documentation change. Falcon's reported suite results are separate evidence.

## Standalone packaging

Local artifact: `crates/target/release/xray-engine-standalone-win64.zip`, containing the executable, README, Architecture Guide and this record. Build output archives are not committed or published by this step.

Extract to a writable directory and run `xray-engine.exe --headless --max-frames 60`. For rendered use, provide Windows DX12 / SM 6.6 support, the Microsoft Visual C++ x64 runtime (VCRUNTIME140.dll), DXC and its companion DLLs via Windows SDK/PATH or DXC_PATH, and separately obtained compatible game assets using `-fsltx`. No game assets, DXC, Microsoft redistributables or legacy engine DLLs are bundled. Consult repository license/provenance terms before redistribution.

A fresh-machine rendered installation and WAN NAT traversal remain unverified. The archive is a runtime/documentation bundle, not a self-contained installer. Build/link separation does not certify clean-room authorship; see the Architecture Guide.
