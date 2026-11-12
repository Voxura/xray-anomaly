//! # xray-render-dx12
//!
//! High-performance pure Rust DirectX 12 render backend for the X-Ray Monolith Engine.
//! Implements Shader Model 6.6 bindless descriptor indexing, an automated Render Graph DAG pass
//! compiler with resource barrier scheduling, and multi-threaded command list recording.

#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::type_complexity)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::chunks_exact_to_as_chunks)]

pub mod allocators;
pub mod bindless;
pub mod clouds;
pub mod deferred;
pub mod details;
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
pub mod postprocess;
pub mod rain;
pub mod renderer;
pub mod shaders;
pub mod shadow;
pub mod sky;
pub mod swapchain;
pub mod texture;
pub mod ui_canvas;
pub mod wallmarks;

pub use clouds::{CloudRenderer, CloudSimulation, CloudsPushConstants};
pub use deferred::{
    DeferredCombinePipeline, DeferredPipeline, GBufferManager, GBufferMeshPipeline,
    GBufferResources, GBufferTargets, GpuLightParams, PushCombineConstants,
};
pub use details::{
    DetailManager, DetailPushConstants, DetailRenderer, GpuDetailInstance, GpuDetailVertex,
};
pub use geometry::PushDrawConstants;
pub use geometry_abi::{CameraConstants, MeshHandle, XRayVertex};
pub use lighting::{DynamicLight, GpuLight, LightManager, LightType};
pub use particle::{GpuParticleInstance, ParticleSystemRenderer};
pub use postprocess::{
    BloomAdaptState, BloomConfig, ImageCorrectionConfig, ImageCorrectionGpuParams,
    PostprocessManager, PostprocessPushConstants, PostprocessRenderer, TonemapGpuParams,
    TonemapParams,
};
pub use rain::{GpuRainDrop, RainManager, RainPushConstants, RainRenderer};
pub use renderer::{DeviceState, Dx12Renderer, RenderStats};
pub use shadow::{CascadedShadowAtlas, DirectionalShadowMap};
pub use sky::{SkyPushConstants, SkyRenderer};
pub use ui_canvas::{GlyphQuad, GpuUiQuad, UiCanvas, UiCanvasPipeline, UiPushConstants};
pub use wallmarks::{
    GpuWallmarkVertex, WallmarkPushConstants, WallmarkRenderer, WallmarksManager,
};
