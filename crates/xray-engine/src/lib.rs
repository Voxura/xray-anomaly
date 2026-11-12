//! X-Ray Monolith Standalone Pure-Rust Engine
//!
//! Integrates the core subsystems:
//! - Window management & native Win32 message pump (`window`)
//! - Action mapping & keyboard/mouse edge detection (`input`)
//! - First-person camera with X-Ray left-handed projection (`camera`)
//! - High-precision sub-microsecond timer and 60Hz fixed accumulator (`timer`)
//! - Main game loop runtime coordinator (`game_loop`)
//! - Stalker capsule character controller & BVH collision physics (`physics`)
//! - DirectX 12 render graph bridge (`render_bridge`)
//! - Level world streaming, BVH collision & weather simulation (`world_loader`)
//! - Live ALife entity spawn manager & `.scop`/`.scoc` co-op state sync (`alife_manager`)

pub mod alife_manager;
pub mod camera;
pub mod game_loop;
pub mod input;
pub mod physics;
pub mod render_bridge;
pub mod timer;
pub mod window;
pub mod world_loader;

pub use alife_manager::{AlifeManager, EntityRole, LiveAlifeEntity, RemotePeerAvatar};
pub use camera::{FirstPersonCamera, DEFAULT_FAR_Z, DEFAULT_FOV_DEG, DEFAULT_NEAR_Z};
pub use game_loop::{EngineConfig, EngineRuntime, NetworkSession};
pub use input::{Action, EngineInput, MouseButton};
pub use physics::{CapsuleConfig, StalkerController};
pub use render_bridge::{FrameRenderStats, HudOverlayState, RenderBridge, RenderBridgeConfig};
pub use timer::EngineTimer;
pub use window::{EngineWindow, WindowConfig, WindowEvent};
pub use world_loader::{
    LevelLoadConfig, LoadedDynamicModel, LoadedVisualMesh, LoadedWorld, WorldLoadError,
    WorldLoadSummary, WorldLoader, WorldWeatherState,
};
