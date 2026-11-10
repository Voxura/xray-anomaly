//! # xray-render-dx12
//!
//! High-performance pure Rust DirectX 12 render backend for the X-Ray Monolith Engine.
//! Implements Shader Model 6.6 bindless descriptor indexing, an automated Render Graph DAG pass
//! compiler with resource barrier scheduling, and multi-threaded command list recording.

pub mod allocators;
pub mod bindless;
pub mod deferred;
pub mod device;
pub mod ffi;
pub mod graph;
pub mod heaps;
pub mod pipeline;
pub mod renderer;
pub mod shaders;
pub mod swapchain;
