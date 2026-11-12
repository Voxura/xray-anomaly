//! X-Ray Monolith Standalone Pure-Rust Engine
//!
//! Integrates the core subsystems:
//! - Window management & native Win32 message pump (`window`)
//! - Action mapping & keyboard/mouse edge detection (`input`)
//! - First-person camera with X-Ray left-handed projection (`camera`)
//! - High-precision sub-microsecond timer and 60Hz fixed accumulator (`timer`)
//! - Main game loop runtime coordinator (`game_loop`)

pub mod camera;
pub mod game_loop;
pub mod input;
pub mod timer;
pub mod window;

pub use camera::{FirstPersonCamera, DEFAULT_FAR_Z, DEFAULT_FOV_DEG, DEFAULT_NEAR_Z};
pub use game_loop::{EngineConfig, EngineRuntime, NetworkSession};
pub use input::{Action, EngineInput, MouseButton};
pub use timer::EngineTimer;
pub use window::{EngineWindow, WindowConfig, WindowEvent};
