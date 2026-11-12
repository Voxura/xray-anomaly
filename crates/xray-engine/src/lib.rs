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
//! - First-person HUD weapons, skeletal viewmodel rig & ballistic combat (`hud_weapon`)
//! - Interactive 2D grid Inventory, Equipment & NPC/Trader dialogue (`ui_inventory`)
//! - Interactive PDA level map, co-op peer radar & story quest tracker (`ui_pda`)
//! - Dynamic level transitions, multi-stage loading screen & co-op sync (`level_transition`)
//! - Pure-Rust 3D spatial audio subsystem (`xray_audio`)

pub mod alife_manager;
pub mod camera;
pub mod game_loop;
pub mod hud_weapon;
pub mod input;
pub mod level_transition;
pub mod physics;
pub mod render_bridge;
pub mod timer;
pub mod ui_inventory;
pub mod ui_pda;
pub mod window;
pub mod world_loader;

pub use xray_audio;
pub use xray_audio::{AudioListener, AudioManager, AudioManagerConfig, SurfaceMaterialCategory};

pub use alife_manager::{AlifeManager, EntityRole, LiveAlifeEntity, RemotePeerAvatar};
pub use camera::{FirstPersonCamera, DEFAULT_FAR_Z, DEFAULT_FOV_DEG, DEFAULT_NEAR_Z};
pub use game_loop::{EngineConfig, EngineRuntime, NetworkSession};
pub use hud_weapon::{
    hud_projection_matrix, BallisticImpact, HudDrawPart, HudWeaponController, ShotReport,
    WeaponFireMode, WeaponSpec, WeaponState,
};
pub use input::{Action, EngineInput, MouseButton};
pub use level_transition::{
    canonical_level_name, lookup_level_metadata, resolve_level_vfs_dir, ActiveTransition,
    LevelChangerVolume, LevelTransitionManager, TransitionState, TransitionStepOutput,
    TransitionUiLayout, TravelPrompt, ZoneLevelMetadata,
};
pub use physics::{CapsuleConfig, StalkerController};
pub use render_bridge::{FrameRenderStats, HudOverlayState, RenderBridge, RenderBridgeConfig};
pub use timer::EngineTimer;
pub use ui_inventory::{
    DialoguePromptState, DragState, EquipmentSlot, InventoryItem, InventoryLayout,
    InventoryManager, ItemCategory, ItemUseEffect, UiFontResource, UseTarget,
};
pub use ui_pda::{MapBlip, MapBlipKind, PdaLayout, PdaManager, PdaTab, QuestTrackerEntry};
pub use window::{EngineWindow, WindowConfig, WindowEvent};
pub use world_loader::{
    LevelLoadConfig, LoadedDynamicModel, LoadedVisualMesh, LoadedWorld, WorldLoadError,
    WorldLoadSummary, WorldLoader, WorldWeatherState,
};
