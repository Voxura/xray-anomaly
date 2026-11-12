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
pub mod geometry;
pub mod geometry_abi;
mod geometry_ffi;
pub mod heaps;
pub mod lighting;
pub mod particle;
pub mod pipeline;
pub mod renderer;
pub mod shaders;
pub mod shadow;
pub mod swapchain;
pub mod texture;
pub mod ui_canvas;

pub use geometry::PushDrawConstants;
pub use geometry_abi::{CameraConstants, MeshHandle, XRayVertex};
pub use lighting::{DynamicLight, GpuLight, LightManager, LightType};
pub use particle::{GpuParticleInstance, ParticleSystemRenderer};
pub use shadow::DirectionalShadowMap;
pub use ui_canvas::{GlyphQuad, GpuUiQuad, UiCanvas, UiCanvasPipeline, UiPushConstants};
