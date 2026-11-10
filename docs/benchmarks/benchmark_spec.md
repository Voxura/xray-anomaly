# X-Ray Anomaly: Benchmark Harness & Architecture Specification

## 1. Executive Summary & Core Objective
The objective of this project is the complete rewrite of the X-Ray engine into pure, modern Rust (named **x-ray anomaly**).
The motivation is not language replacement for its own sake, but unlocking modern systems architecture:
- Data-Oriented Design (DoD) / Cache-coherent Component layouts
- Fearless multi-threaded CPU command generation
- Explicit modern graphics backend (DirectX 12 / low-level API)
- Sub-millisecond frame pacing with zero GC/heap fragmentation pauses

C/C++ interoperability (FFI) is strictly temporary scaffolding during phased migration, with explicit exit criteria for every bridge.

---

## 2. Metric Schema & Statistical Rigor
Every benchmark run must export structured metrics conforming to the following schema:

```csv
run_id,frame_id,phase,frame_interval_ms,cpu_frame_ms,cpu_submit_ms,gpu_ms,alloc_count,alloc_bytes,peak_live_heap_bytes,workload_counters
```

### Acceptance & Statistical Rules:
1. **Sample Size:** Minimum 10,000 frames per run across 5 repeated runs.
2. **Thermal & Order Skew:** Alternate baseline (C++) and candidate (Rust) execution order.
3. **Pacing Focus:** Report Mean, p95, and p99 frame times. Retain and isolate stall spikes.
4. **Disentangled Metrics:** Never add overlapping CPU and GPU execution times. Differentiate cold OS disk cache from warm memory reads.

---

## 3. Workload Categories
1. **Asset & VFS Pipeline:**
   - Manifest: Fixed distribution of `.ltx`, `.xml`, textures, and compressed archives.
   - Decomposition: Measure I/O read, decompression, and parsing/AST construction separately.
2. **Spatial Queries (`xrCDB` / Physics):**
   - Regimes: Sparse, Typical, and High-Density raycast batches.
   - Parameters: Deterministic PRNG seed, static vs dynamic geometry, hit/miss tracking.
3. **Render Submission (DirectX 12):**
   - Multithreaded command list recording throughput.
   - Draw call count variation under identical scene geometry to isolate CPU driver overhead.
