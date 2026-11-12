//! Engine Game Loop & Subsystem Coordinator
//!
//! Ties together Win32 window event polling, high-precision timing, camera updates,
//! fixed-timestep simulation ticks (capsule physics/ALife/network), and the 7-stage
//! DirectX 12 render graph frame pipeline.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::alife_manager::{AlifeManager, EntityRole};
use crate::camera::FirstPersonCamera;
use crate::input::{Action, EngineInput};
use crate::physics::StalkerController;
use crate::render_bridge::{HudOverlayState, RenderBridge, RenderBridgeConfig};
use crate::timer::EngineTimer;
use crate::window::{EngineWindow, WindowConfig};
use crate::world_loader::{LevelLoadConfig, LoadedWorld, WorldLoader};
use xray_audio::{AudioListener, AudioManager, AudioManagerConfig};
use xray_formats::cform::CformBvh;
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
    pub physics: StalkerController,
    pub renderer: Option<Dx12Renderer>,
    pub render_bridge: Option<RenderBridge>,
    pub audio: AudioManager,
    pub hud: HudOverlayState,
    pub network: Option<NetworkSession>,
    pub world: Option<LoadedWorld>,
    pub alife: Option<AlifeManager>,
    pub vfs: Option<Arc<VirtualFileSystem>>,
    pub elapsed_sec: f32,
    pub running: bool,
}

impl EngineRuntime {
    /// Initialize full runtime, optionally creating and binding the DirectX 12 renderer
    /// and render graph pipeline bridge.
    pub fn new(
        config: EngineConfig,
        init_renderer: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let window = EngineWindow::new(&config.window)?;
        let timer = EngineTimer::new();
        let input = EngineInput::new();
        let camera = FirstPersonCamera::new(config.window.width, config.window.height);
        let mut physics = StalkerController::default();
        physics.sync_from_eye(camera.position);

        let (renderer, render_bridge) = if init_renderer {
            let mut r = Dx12Renderer::new(1)?;
            r.attach_window(window.hwnd(), config.window.width, config.window.height)?;
            let rb = RenderBridge::new(&mut r, RenderBridgeConfig::default())?;
            (Some(r), Some(rb))
        } else {
            (None, None)
        };

        let audio = AudioManager::new(AudioManagerConfig {
            prefer_hardware_output: init_renderer && config.window.visible,
            ..Default::default()
        });

        Ok(Self {
            config,
            window,
            timer,
            input,
            camera,
            physics,
            renderer,
            render_bridge,
            audio,
            hud: HudOverlayState::default(),
            network: None,
            world: None,
            alife: None,
            vfs: None,
            elapsed_sec: 0.0,
            running: true,
        })
    }

    /// Attach a shared [`VirtualFileSystem`] for on-demand dynamic model/texture/audio streaming.
    pub fn attach_vfs(&mut self, vfs: Arc<VirtualFileSystem>) {
        self.audio.attach_vfs(Arc::clone(&vfs));
        self.vfs = Some(vfs);
    }

    /// Attach a peer-to-peer co-op multiplayer session.
    pub fn attach_network(&mut self, session: NetworkSession) {
        self.network = Some(session);
    }

    /// Attach a streamed level world bundle.
    pub fn attach_world(&mut self, mut world: LoadedWorld) {
        if let Some(alife) = &self.alife {
            world.weather.apply_scop_weather(&alife.weather_snapshot);
        }
        if let Some(rb) = &mut self.render_bridge {
            rb.invalidate_world();
        }
        self.audio
            .set_occlusion_mesh(world.sound_occlusion.clone().map(Arc::new));
        self.world = Some(world);
    }

    /// Attach an ALife simulation manager and synchronize camera, physics & weather initial state.
    pub fn attach_alife(&mut self, alife: AlifeManager) {
        if let Some(actor) = alife.host_actor() {
            self.camera.position = actor.position;
            self.camera.pitch = actor.angle[0];
            self.camera.yaw = actor.angle[1];
            self.physics.sync_from_eye(self.camera.position);
            self.hud.health = actor.health.clamp(0.0, 1.0);
        }
        if let Some(world) = &mut self.world {
            world.weather.apply_scop_weather(&alife.weather_snapshot);
        }
        self.alife = Some(alife);
    }

    /// Synchronize currently attached world geometry, textures, lights, and grass details onto GPU.
    pub fn sync_world_to_gpu(
        &mut self,
        vfs: Option<&VirtualFileSystem>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let (Some(r), Some(rb), Some(w)) =
            (&mut self.renderer, &mut self.render_bridge, &self.world)
        {
            rb.sync_world(r, w, vfs)?;
        }
        Ok(())
    }

    /// Stream level world assets and optional `.scop` savegame from `VirtualFileSystem`.
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
        self.sync_world_to_gpu(Some(vfs))?;
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
        if self.camera.aspect != (self.window.height() as f32 / self.window.width() as f32) {
            self.camera.resize(self.window.width(), self.window.height());
            if let Some(r) = &mut self.renderer {
                r.resize(self.window.width(), self.window.height())?;
            }
        }

        // 3. Update timer
        let dt = self.timer.tick();
        self.elapsed_sec += dt;

        // 4. Update camera orientation from mouse delta & compute world-space XZ wish vector
        if self.input.is_cursor_locked() {
            let (dx, dy) = self.input.mouse_delta();
            if dx != 0.0 || dy != 0.0 {
                self.camera.rotate(dx, dy, self.input.sensitivity());
            }
        }

        let intent = self.input.movement_intent();
        let crouch_held = self.input.is_action_down(Action::Crouch);
        let mut speed = self.config.move_speed;
        if self.input.is_action_down(Action::Sprint) && !crouch_held {
            speed = self.config.sprint_speed;
        } else if crouch_held {
            speed = self.config.crouch_speed;
        }

        let forward_horiz = [self.camera.yaw.sin(), self.camera.yaw.cos()];
        let right_full = self.camera.right();
        let wish_xz = [
            right_full[0] * intent[0] + forward_horiz[0] * intent[1],
            right_full[2] * intent[0] + forward_horiz[1] * intent[1],
        ];
        let mut jump_pending = self.input.is_action_just_pressed(Action::Jump);

        // Build BVH acceleration structure for `level.cform` if loaded
        let cform_bvh = self
            .world
            .as_ref()
            .and_then(|w| w.cform.as_ref())
            .map(CformBvh::build);
        let gamemtl_ref = self.world.as_ref().and_then(|w| w.gamemtl.as_ref());

        // 5. Fixed-timestep simulation steps (capsule physics, footsteps, ALife, network sync)
        let fixed_dt = self.config.fixed_dt;
        let physics = &mut self.physics;
        let camera = &mut self.camera;
        let net = &mut self.network;
        let alife = &mut self.alife;
        let hud = &mut self.hud;
        let audio = &mut self.audio;
        self.timer.step_fixed(fixed_dt, |f_dt| {
            let jump_req = std::mem::take(&mut jump_pending);
            physics.step(
                wish_xz,
                speed,
                jump_req,
                crouch_held,
                f_dt,
                cform_bvh.as_ref(),
            );
            camera.position = physics.eye_position();

            let horiz_speed = physics.velocity[0].hypot(physics.velocity[2]);
            audio.step_footsteps(
                f_dt,
                physics.feet_position,
                horiz_speed,
                physics.grounded,
                physics.crouching,
                physics.ground_material,
                gamemtl_ref,
            );

            if let Some(session) = net.as_mut()
                && let Ok(deltas) = session.update()
                && let Some(mgr) = alife.as_mut()
            {
                for delta in &deltas {
                    mgr.apply_network_delta(delta);
                }
            }
            if let Some(mgr) = alife.as_mut() {
                mgr.set_host_camera_transform(
                    camera.position,
                    camera.yaw,
                    camera.pitch,
                    physics.velocity,
                );
                if let Some(actor) = mgr.host_actor() {
                    hud.health = actor.health.clamp(0.0, 1.0);
                }
                mgr.step(f_dt);
                if let Some(session) = net.as_mut() {
                    for delta in mgr.collect_outgoing_deltas() {
                        let _ = session.send_entity_delta(&delta);
                    }
                }
            }
        });

        // Keep host ALife actor orientation & 3D audio listener synced every frame
        if let Some(mgr) = &mut self.alife {
            mgr.set_host_camera_transform(
                self.camera.position,
                self.camera.yaw,
                self.camera.pitch,
                self.physics.velocity,
            );
        }
        self.audio.update_listener(AudioListener::from_basis(
            self.camera.position,
            self.camera.forward(),
            self.camera.up(),
            self.camera.right(),
            self.physics.velocity,
        ));

        if self.input.is_action_just_pressed(Action::PrimaryFire) {
            self.audio.trigger_weapon_sound(
                "weapons\\ak74_shoot",
                self.camera.position,
                self.physics.velocity,
            );
        }

        if let Some(world) = &mut self.world {
            world.step_weather(
                dt,
                self.camera.position,
                self.camera.forward(),
                self.camera.far_z,
            );

            let (wind_vel, rain_dens) = world
                .weather
                .current_env
                .as_ref()
                .map(|e| (e.wind_velocity, e.rain_density))
                .unwrap_or((1.5, 0.0));
            let thunder_trigger = world.weather.last_thunderbolt.as_ref().and_then(|tb| {
                tb.triggered_sound
                    .as_deref()
                    .map(|snd| (snd, tb.triggered_sound_pos, tb.triggered_distance.max(450.0)))
            });
            self.audio.update_weather(
                wind_vel,
                rain_dens,
                world.weather.wetness.factor,
                self.camera.position,
                thunder_trigger,
            );

            if let Some(static_sounds) = &world.static_sounds {
                let hour_24 = world.weather.cycle_time_sec / 3600.0;
                self.audio.sync_static_sounds(static_sounds, hour_24);
            }
        }

        if let Some(mgr) = &self.alife {
            let mut active_keys = BTreeSet::new();
            for entity in mgr.entities.values() {
                if !entity.online || entity.role == EntityRole::LocalHostActor {
                    continue;
                }
                let tracked = matches!(
                    entity.role,
                    EntityRole::Stalker
                        | EntityRole::Mutant
                        | EntityRole::Trader
                        | EntityRole::RemotePeerActor { .. }
                );
                if tracked {
                    let key = u64::from(entity.id);
                    active_keys.insert(key);
                    let speed_sq = entity.velocity[0] * entity.velocity[0]
                        + entity.velocity[2] * entity.velocity[2];
                    let moving = speed_sq > 0.05 || entity.anim_state != 0;
                    self.audio.update_alife_emitter(
                        key,
                        entity.position,
                        entity.velocity,
                        moving,
                        dt,
                    );
                }
            }
            self.audio.retain_alife_emitters(&active_keys);
        }

        // 6. Direct3D 12 7-stage render graph presentation
        if let (Some(r), Some(rb)) = (&mut self.renderer, &mut self.render_bridge) {
            r.begin_frame()?;
            rb.render_frame(
                r,
                &self.camera,
                self.world.as_ref(),
                self.alife.as_ref(),
                self.vfs.as_deref(),
                &self.hud,
                dt,
                self.elapsed_sec,
            )?;
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
