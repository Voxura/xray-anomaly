//! Engine Game Loop & Subsystem Coordinator
//!
//! Ties together Win32 window event polling, high-precision timing, camera updates,
//! fixed-timestep simulation ticks (physics/ALife/network), and DirectX 12 frame rendering.

use crate::alife_manager::AlifeManager;
use crate::camera::FirstPersonCamera;
use crate::input::{Action, EngineInput};
use crate::timer::EngineTimer;
use crate::window::{EngineWindow, WindowConfig};
use crate::world_loader::{LevelLoadConfig, LoadedWorld, WorldLoader};
use xray_net::{ClientSession, EntityDelta, HostSession};
use xray_render_dx12::Dx12Renderer;
use xray_vfs::VirtualFileSystem;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub window: WindowConfig,
    pub fixed_dt: f32,
    pub vsync: bool,
    pub move_speed: f32,
    pub sprint_speed: f32,
    pub crouch_speed: f32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            window: WindowConfig::default(),
            fixed_dt: 1.0 / 60.0,
            vsync: true,
            move_speed: 5.0,
            sprint_speed: 9.0,
            crouch_speed: 2.5,
        }
    }
}

pub enum NetworkSession {
    Host(HostSession),
    Client(ClientSession),
}

impl NetworkSession {
    pub fn update(&mut self) -> std::io::Result<Vec<EntityDelta>> {
        match self {
            Self::Host(h) => h.update(),
            Self::Client(c) => c.update(),
        }
    }

    pub fn send_entity_delta(&mut self, delta: &EntityDelta) -> std::io::Result<()> {
        match self {
            Self::Host(h) => h.send_entity_delta(delta),
            Self::Client(c) => c.send_entity_delta(delta),
        }
    }
}

pub struct EngineRuntime {
    pub config: EngineConfig,
    pub window: EngineWindow,
    pub timer: EngineTimer,
    pub input: EngineInput,
    pub camera: FirstPersonCamera,
    pub renderer: Option<Dx12Renderer>,
    pub network: Option<NetworkSession>,
    pub world: Option<LoadedWorld>,
    pub alife: Option<AlifeManager>,
    pub running: bool,
}

impl EngineRuntime {
    /// Initialize full runtime, optionally creating and binding the DirectX 12 renderer
    pub fn new(
        config: EngineConfig,
        init_renderer: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let window = EngineWindow::new(&config.window)?;
        let timer = EngineTimer::new();
        let input = EngineInput::new();
        let camera = FirstPersonCamera::new(config.window.width, config.window.height);

        let renderer = if init_renderer {
            let mut r = Dx12Renderer::new(1)?;
            r.attach_window(window.hwnd(), config.window.width, config.window.height)?;
            Some(r)
        } else {
            None
        };

        Ok(Self {
            config,
            window,
            timer,
            input,
            camera,
            renderer,
            network: None,
            world: None,
            alife: None,
            running: true,
        })
    }

    /// Attach a peer-to-peer co-op multiplayer session
    pub fn attach_network(&mut self, session: NetworkSession) {
        self.network = Some(session);
    }

    /// Attach a streamed level world bundle
    pub fn attach_world(&mut self, mut world: LoadedWorld) {
        if let Some(alife) = &self.alife {
            world.weather.apply_scop_weather(&alife.weather_snapshot);
        }
        self.world = Some(world);
    }

    /// Attach an ALife simulation manager and synchronize camera & weather initial state
    pub fn attach_alife(&mut self, alife: AlifeManager) {
        if let Some(actor) = alife.host_actor() {
            self.camera.position = actor.position;
            self.camera.pitch = actor.angle[0];
            self.camera.yaw = actor.angle[1];
        }
        if let Some(world) = &mut self.world {
            world.weather.apply_scop_weather(&alife.weather_snapshot);
        }
        self.alife = Some(alife);
    }

    /// Stream level world assets and optional `.scop` savegame from `VirtualFileSystem`
    pub fn load_world_and_save_from_vfs(
        &mut self,
        vfs: &VirtualFileSystem,
        level_config: &LevelLoadConfig,
        scop_vpath: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let world = WorldLoader::load_from_vfs(vfs, level_config)?;
        let mut alife = match scop_vpath {
            Some(path) => AlifeManager::load_from_vfs(vfs, path)?,
            None => AlifeManager::default(),
        };
        if alife.entities.is_empty()
            && let Some(spawn) = &world.level_spawn
        {
            alife.seed_from_level_spawn(&world.level_dir, spawn);
        }
        self.attach_world(world);
        self.attach_alife(alife);
        Ok(())
    }

    /// Advance one frame of the game loop. Returns `false` when window closes or user quits.
    pub fn step_frame(&mut self) -> Result<bool, Box<dyn std::error::Error>> {
        if !self.running {
            return Ok(false);
        }

        // 1. Process OS window messages
        let active = self.window.poll_events(&mut self.input);
        if !active || self.input.is_action_just_pressed(Action::Menu) {
            self.running = false;
            return Ok(false);
        }

        // 2. Handle window resize
        if self.camera.aspect
            != (self.window.height() as f32 / self.window.width() as f32)
        {
            self.camera.resize(self.window.width(), self.window.height());
            if let Some(r) = &mut self.renderer {
                r.resize(self.window.width(), self.window.height())?;
            }
        }

        // 3. Update timer
        let dt = self.timer.tick();

        // 4. Update camera rotation and movement
        if self.input.is_cursor_locked() {
            let (dx, dy) = self.input.mouse_delta();
            if dx != 0.0 || dy != 0.0 {
                self.camera.rotate(dx, dy, self.input.sensitivity());
            }
        }

        let prev_pos = self.camera.position;
        let intent = self.input.movement_intent();
        let mut speed = self.config.move_speed;
        if self.input.is_action_down(Action::Sprint) {
            speed = self.config.sprint_speed;
        } else if self.input.is_action_down(Action::Crouch) {
            speed = self.config.crouch_speed;
        }
        self.camera.move_relative(intent, speed, dt);

        if let Some(alife) = &mut self.alife {
            let inv_dt = if dt > 1e-6 { 1.0 / dt } else { 0.0 };
            let vel = [
                (self.camera.position[0] - prev_pos[0]) * inv_dt,
                (self.camera.position[1] - prev_pos[1]) * inv_dt,
                (self.camera.position[2] - prev_pos[2]) * inv_dt,
            ];
            alife.set_host_camera_transform(
                self.camera.position,
                self.camera.yaw,
                self.camera.pitch,
                vel,
            );
        }

        // 5. Fixed-timestep simulation steps (physics, ALife, network sync)
        let fixed_dt = self.config.fixed_dt;
        let net = &mut self.network;
        let alife = &mut self.alife;
        self.timer.step_fixed(fixed_dt, |f_dt| {
            if let Some(session) = net.as_mut()
                && let Ok(deltas) = session.update()
                && let Some(mgr) = alife.as_mut()
            {
                for delta in &deltas {
                    mgr.apply_network_delta(delta);
                }
            }
            if let Some(mgr) = alife.as_mut() {
                mgr.step(f_dt);
                if let Some(session) = net.as_mut() {
                    for delta in mgr.collect_outgoing_deltas() {
                        let _ = session.send_entity_delta(&delta);
                    }
                }
            }
        });

        if let Some(world) = &mut self.world {
            world.step_weather(
                dt,
                self.camera.position,
                self.camera.forward(),
                self.camera.far_z,
            );
        }

        // 6. Direct3D 12 frame presentation
        if let Some(r) = &mut self.renderer {
            let _camera_constants = self.camera.to_render_constants();
            r.begin_frame()?;
            // In the combined pipeline, deferred shading / sky / geometry passes execute here.
            r.end_frame(self.config.vsync)?;
        }

        Ok(true)
    }

    /// Run engine main loop until window close
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        while self.running {
            if !self.step_frame()? {
                break;
            }
        }

        if let Some(r) = &mut self.renderer {
            let _ = r.wait_for_gpu();
        }

        Ok(())
    }
}
