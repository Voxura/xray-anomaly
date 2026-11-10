# Phase 1 Migration Plan: Read-Only LTX Configuration Subsystem

## 1. Executive Summary & Scope Boundary
Phase 1 focuses **exclusively on read-only LTX configuration parsing** behind an opt-in runtime flag (`-use_rust_ini`).
- **Included in Phase 1:** Standalone zero-copy LTX parser, section inheritance, `#include` resolution, safe C ABI export layer.
- **Explicitly Deferred to Phase 2:** VFS archive (`.db`/`.xdb`) decompression, asset loading pipeline, and global `shared_str` table replacement.

```
┌────────────────────────────────────────────────────────┐
│                   X-Ray Monolith (C++)                 │
│         xrCore `CInifile` (Default Engine Path)         │
│   ┌─────────────────────────────────────────────────┐  │
│   │ Opt-in Switch: `-use_rust_ini` (Runtime Flag)    │  │
│   └────────────────────────┬────────────────────────┘  │
└────────────────────────────┼───────────────────────────┘
                             │ (Safe C ABI: Opaque Handles)
┌────────────────────────────▼───────────────────────────┐
│                   xr_core_ffi Layer                    │
│   xr_config_open() • xr_config_read() • catch_unwind   │
└────────────────────────────┬───────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────┐
│               crates/xray-core (Pure Rust)             │
│   Zero-Copy LTX Parser • Slice Borrowing • Arenas      │
└────────────────────────────────────────────────────────┘
```

---

## 2. Adopted Acceptance Gates (Strict Verification Protocol)

### Gate 1: Baseline Build & Verification
- Record a clean, reproducible legacy C++ engine build and smoke test prior to touching any engine source files.

### Gate 2: Differential Compatibility Suite
- Differential parity test across all 46 cataloged fixtures in `docs/benchmarks/fixtures_manifest.json`.
- Must produce byte-for-byte or DOM-equivalent key-value resolution for:
  - Section inheritance: `[child]:parent`
  - Recursive includes: `#include "path.ltx"`
  - Inline and full-line comments (`;` and `//`)
  - Duplicate section/key resolution matching X-Ray's last-write-wins policy
  - Malformed and truncated inputs gracefully rejecting without crashing.

### Gate 3: Explicit C ABI Contract
- **Opaque Handles:** Engine consumes an opaque `xr_config_t*`.
- **Ownership Rules:** Rust allocates and frees its own heap storage. C++ never calls `free()` on Rust-allocated memory.
- **Safety Preconditions:** All pointer arguments checked for null before dereferencing.

### Gate 4: Zero-Panic & Fault Containment
- All exported functions wrapped in `std::panic::catch_unwind`.
- Any internal panic is trapped and returns an explicit negative status code (`XR_STATUS_ERROR_PANIC`), preventing cross-boundary stack unwinding.

### Gate 5: Reversible Dual-Path Isolation
- The legacy C++ `CInifile` implementation remains the default path.
- The Rust parser is activated solely via `-use_rust_ini` command line argument.
- Disabling the flag restores 100% legacy behavior immediately.

### Gate 6: Empirical Profiling & Memory Budgets
- Benchmark cold-load and warm-load latency against the legacy C++ parser using the established measurement harness.
- Verify peak memory and allocation counts meet or exceed the agreed performance budgets.

---

## 3. Road to Phase 2
Only after all 6 gates are fulfilled and verified will we advance to **Phase 2**:
- **Phase 2 Scope:** VFS `.db`/`.xdb` decompression, package mounting, and modern resource streaming.
