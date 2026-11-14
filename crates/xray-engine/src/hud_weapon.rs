//! First-Person HUD Weapons, Skeletal Bone Rig & Ballistic Combat Mechanics (`xray-engine`)
//!
//! Implements the complete first-person combat subsystem:
//! - Separate HUD projection matrix (`hud_projection_matrix` using `R_VIEWPORT_NEAR = 0.005`) to prevent
//!   viewmodel clipping into nearby world walls, alongside smooth ADS camera FOV zooming.
//! - On-demand VFS streaming for weapon & hand `.ogf` models (plus optional `.omf` animations),
//!   evaluating per-bone transform palettes (`BonePalette` / `skin_ogf_hierarchy`) and clean
//!   procedural viewmodel fallbacks when raw `.ogf` files are not mounted.
//! - Complete weapon state machine (`Idle`, `Aiming`, `Firing`, `Reloading`, `Holstering`,
//!   `Unholstering`) driven by `PrimaryFire` (LMB), `SecondaryFire` (RMB ADS), `Reload` (`R`),
//!   and `Slot1..Slot6` weapon swapping.
//! - Ballistic hitscan trajectory raycasting against `CformBvh` & live `AlifeManager` entity
//!   hitboxes with `gamemtl.xr` penetration, distance damage falloff, `WallmarksManager` bullet
//!   hole decals, spark/debris particles, dynamic muzzle-flash light, and 3D `AudioManager`
//!   weapon discharge + bullet impact SFX.

use std::collections::BTreeMap;

use xray_audio::{resolve_surface_category, AudioManager, SurfaceMaterialCategory};
use xray_formats::camera::{
    hud_projection_matrix as fmt_hud_projection_matrix, Dx12CameraConstants, DEFAULT_HUD_FOV_FACTOR,
};
use xray_formats::cform::CformBvh;
use xray_formats::gamemtl::{gamemtl_flags, GameMaterialLibrary};
use xray_formats::hom::{frustum_planes, ClipFrustum};
use xray_formats::lanims::pack_argb;
use xray_formats::ogf::{
    matrix_mul_43, matrix_mul_44, matrix_set_xyz_translate, vector_generate_orthonormal_basis,
    BonePoseMatrices, OgfModel, OgfTextureLink, StaticVertex, XRayMatrix,
};
use xray_formats::particles::{
    project_static_wallmark_on_cform, StaticWallmarkConfig, StaticWallmarkRecord, WallmarkVertex,
};
use xray_formats::skinning::{skin_ogf_hierarchy, BonePalette};
use xray_formats::thm::ResolvedTextureLinks;
use xray_formats::vfs_bridge::{load_ogf_from_vfs, resolve_ogf_texture_links};
use xray_render_dx12::geometry_abi::{CameraConstants, XRayVertex};
use xray_render_dx12::lighting::GpuLight;
use xray_render_dx12::particle::GpuParticleInstance;
use xray_render_dx12::wallmarks::{GpuWallmarkVertex, WallmarksManager};
use xray_vfs::{normalize_virtual_path, VirtualFileSystem};

use crate::alife_manager::{AlifeManager, EntityRole};
use crate::camera::{FirstPersonCamera, DEFAULT_FOV_DEG};
use crate::input::{Action, EngineInput};
use crate::world_loader::LoadedWorld;

/// Far clipping plane for first-person HUD geometry (`hud_projection_matrix`).
pub const DEFAULT_HUD_FAR_PLANE: f32 = 150.0;
/// Maximum number of material layers a single projectile may penetrate.
pub const MAX_PENETRATION_STEPS: usize = 4;

/// High-level firing behavior of a weapon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeaponFireMode {
    Melee,
    SemiAuto,
    FullAuto,
    BoltAction,
}

/// First-person weapon state machine phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeaponState {
    Idle,
    Aiming,
    Firing,
    Reloading,
    Holstering,
    Unholstering,
}

/// Configuration & runtime ammo state for a single inventory weapon slot (`Slot1..=Slot6`).
#[derive(Debug, Clone, PartialEq)]
pub struct WeaponSpec {
    pub slot: u8,
    pub section: String,
    pub weapon_hud_ogf: String,
    pub hands_hud_ogf: String,
    pub shoot_sound: String,
    pub reload_sound: String,
    pub holster_sound: String,
    pub rpm: f32,
    pub fire_mode: WeaponFireMode,
    pub clip_capacity: u32,
    pub ammo_in_clip: u32,
    pub reserve_ammo: u32,
    pub reload_duration_sec: f32,
    pub holster_duration_sec: f32,
    pub unholster_duration_sec: f32,
    pub ads_fov_deg: f32,
    pub ads_zoom_speed: f32,
    pub hip_offset: [f32; 3],
    pub ads_offset: [f32; 3],
    pub muzzle_local_offset: [f32; 3],
    pub base_damage: f32,
    pub range_meters: f32,
    pub penetration_power: f32,
    pub recoil_pitch_kick: f32,
    pub wallmark_size: f32,
}

impl WeaponSpec {
    #[inline]
    pub fn fire_interval_sec(&self) -> f32 {
        60.0 / self.rpm.max(15.0)
    }

    pub fn default_loadout() -> [Self; 6] {
        [
            Self {
                slot: 1,
                section: "wpn_knife".to_string(),
                weapon_hud_ogf: "dynamics\\weapons\\wpn_knife\\wpn_knife_hud.ogf".to_string(),
                hands_hud_ogf: "actors\\stalker_hero\\hand_stalker_hud.ogf".to_string(),
                shoot_sound: "weapons\\knife\\knife_1".to_string(),
                reload_sound: "weapons\\generic_draw".to_string(),
                holster_sound: "weapons\\generic_holster".to_string(),
                rpm: 90.0,
                fire_mode: WeaponFireMode::Melee,
                clip_capacity: 0,
                ammo_in_clip: 0,
                reserve_ammo: 0,
                reload_duration_sec: 0.4,
                holster_duration_sec: 0.18,
                unholster_duration_sec: 0.22,
                ads_fov_deg: 62.0,
                ads_zoom_speed: 10.0,
                hip_offset: [0.16, -0.18, 0.36],
                ads_offset: [0.10, -0.15, 0.32],
                muzzle_local_offset: [0.0, 0.0, 0.45],
                base_damage: 65.0,
                range_meters: 2.3,
                penetration_power: 0.2,
                recoil_pitch_kick: 0.008,
                wallmark_size: 0.04,
            },
            Self {
                slot: 2,
                section: "wpn_pm".to_string(),
                weapon_hud_ogf: "dynamics\\weapons\\wpn_pm\\wpn_pm_hud.ogf".to_string(),
                hands_hud_ogf: "actors\\stalker_hero\\hand_stalker_hud.ogf".to_string(),
                shoot_sound: "weapons\\pm\\pm_shoot".to_string(),
                reload_sound: "weapons\\pm\\pm_reload".to_string(),
                holster_sound: "weapons\\generic_holster".to_string(),
                rpm: 320.0,
                fire_mode: WeaponFireMode::SemiAuto,
                clip_capacity: 8,
                ammo_in_clip: 8,
                reserve_ammo: 64,
                reload_duration_sec: 1.55,
                holster_duration_sec: 0.2,
                unholster_duration_sec: 0.24,
                ads_fov_deg: 55.0,
                ads_zoom_speed: 14.0,
                hip_offset: [0.14, -0.14, 0.34],
                ads_offset: [0.0, -0.095, 0.27],
                muzzle_local_offset: [0.0, 0.025, 0.32],
                base_damage: 34.0,
                range_meters: 110.0,
                penetration_power: 0.55,
                recoil_pitch_kick: 0.015,
                wallmark_size: 0.045,
            },
            Self {
                slot: 3,
                section: "wpn_ak74".to_string(),
                weapon_hud_ogf: "dynamics\\weapons\\wpn_ak74\\wpn_ak74_hud.ogf".to_string(),
                hands_hud_ogf: "actors\\stalker_hero\\hand_stalker_hud.ogf".to_string(),
                shoot_sound: "weapons\\ak74\\ak74_shoot".to_string(),
                reload_sound: "weapons\\ak74_reload".to_string(),
                holster_sound: "weapons\\generic_holster".to_string(),
                rpm: 600.0,
                fire_mode: WeaponFireMode::FullAuto,
                clip_capacity: 30,
                ammo_in_clip: 30,
                reserve_ammo: 180,
                reload_duration_sec: 2.1,
                holster_duration_sec: 0.25,
                unholster_duration_sec: 0.30,
                ads_fov_deg: 50.0,
                ads_zoom_speed: 12.0,
                hip_offset: [0.14, -0.15, 0.36],
                ads_offset: [0.0, -0.105, 0.25],
                muzzle_local_offset: [0.0, 0.02, 0.58],
                base_damage: 44.0,
                range_meters: 350.0,
                penetration_power: 1.25,
                recoil_pitch_kick: 0.014,
                wallmark_size: 0.055,
            },
            Self {
                slot: 4,
                section: "wpn_mp5".to_string(),
                weapon_hud_ogf: "dynamics\\weapons\\wpn_mp5\\wpn_mp5_hud.ogf".to_string(),
                hands_hud_ogf: "actors\\stalker_hero\\hand_stalker_hud.ogf".to_string(),
                shoot_sound: "weapons\\mp5\\mp5_shoot".to_string(),
                reload_sound: "weapons\\mp5_reload".to_string(),
                holster_sound: "weapons\\generic_holster".to_string(),
                rpm: 750.0,
                fire_mode: WeaponFireMode::FullAuto,
                clip_capacity: 30,
                ammo_in_clip: 30,
                reserve_ammo: 180,
                reload_duration_sec: 1.85,
                holster_duration_sec: 0.22,
                unholster_duration_sec: 0.26,
                ads_fov_deg: 52.0,
                ads_zoom_speed: 13.0,
                hip_offset: [0.13, -0.15, 0.34],
                ads_offset: [0.0, -0.102, 0.24],
                muzzle_local_offset: [0.0, 0.02, 0.46],
                base_damage: 32.0,
                range_meters: 180.0,
                penetration_power: 0.72,
                recoil_pitch_kick: 0.011,
                wallmark_size: 0.048,
            },
            Self {
                slot: 5,
                section: "wpn_svd".to_string(),
                weapon_hud_ogf: "dynamics\\weapons\\wpn_svd\\wpn_svd_hud.ogf".to_string(),
                hands_hud_ogf: "actors\\stalker_hero\\hand_stalker_hud.ogf".to_string(),
                shoot_sound: "weapons\\svd\\svd_shoot".to_string(),
                reload_sound: "weapons\\svd_reload".to_string(),
                holster_sound: "weapons\\generic_holster".to_string(),
                rpm: 120.0,
                fire_mode: WeaponFireMode::SemiAuto,
                clip_capacity: 10,
                ammo_in_clip: 10,
                reserve_ammo: 50,
                reload_duration_sec: 2.5,
                holster_duration_sec: 0.32,
                unholster_duration_sec: 0.36,
                ads_fov_deg: 28.0,
                ads_zoom_speed: 10.0,
                hip_offset: [0.15, -0.16, 0.38],
                ads_offset: [0.0, -0.098, 0.23],
                muzzle_local_offset: [0.0, 0.025, 0.74],
                base_damage: 95.0,
                range_meters: 600.0,
                penetration_power: 1.75,
                recoil_pitch_kick: 0.035,
                wallmark_size: 0.068,
            },
            Self {
                slot: 6,
                section: "wpn_bolt".to_string(),
                weapon_hud_ogf: "dynamics\\devices\\dev_bolt\\dev_bolt_hud.ogf".to_string(),
                hands_hud_ogf: "actors\\stalker_hero\\hand_stalker_hud.ogf".to_string(),
                shoot_sound: "weapons\\generic_draw".to_string(),
                reload_sound: "weapons\\generic_draw".to_string(),
                holster_sound: "weapons\\generic_holster".to_string(),
                rpm: 75.0,
                fire_mode: WeaponFireMode::BoltAction,
                clip_capacity: 1,
                ammo_in_clip: 1,
                reserve_ammo: 999,
                reload_duration_sec: 0.65,
                holster_duration_sec: 0.18,
                unholster_duration_sec: 0.22,
                ads_fov_deg: 64.0,
                ads_zoom_speed: 9.0,
                hip_offset: [0.15, -0.17, 0.32],
                ads_offset: [0.08, -0.14, 0.30],
                muzzle_local_offset: [0.0, 0.0, 0.25],
                base_damage: 8.0,
                range_meters: 15.0,
                penetration_power: 0.1,
                recoil_pitch_kick: 0.004,
                wallmark_size: 0.03,
            },
        ]
    }
}

/// Cached `.ogf` asset streamed from VFS for first-person HUD weapon or hand geometry.
#[derive(Debug, Clone)]
pub struct CachedHudOgfAsset {
    pub vpath: String,
    pub model: OgfModel,
    pub texture_links: Vec<ResolvedTextureLinks>,
    pub bind_poses: Option<Vec<BonePoseMatrices>>,
}

/// CPU-evaluated render part ready for GPU upload in Stage (e.7) HUD pass.
#[derive(Debug, Clone)]
pub struct HudDrawPart {
    pub key: String,
    pub texture_vpath: Option<String>,
    pub vertices: Vec<XRayVertex>,
    pub indices: Vec<u16>,
    pub world_transform: XRayMatrix,
}

/// Single projectile impact result along a ballistic ray trace.
#[derive(Debug, Clone, PartialEq)]
pub struct BallisticImpact {
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub distance: f32,
    pub damage: f32,
    pub material_id: Option<u16>,
    pub surface: SurfaceMaterialCategory,
    pub tri_index: usize,
    pub suppress_wm: bool,
    pub hit_entity_id: Option<u16>,
}

/// Telemetry & combat summary for one fired round.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShotReport {
    pub slot: u8,
    pub muzzle_world_pos: [f32; 3],
    pub impacts: Vec<BallisticImpact>,
    pub penetrated_surfaces: usize,
    pub wallmarks_spawned: usize,
    pub sparks_spawned: usize,
}

#[derive(Debug, Clone)]
struct SparkParticle {
    position: [f32; 3],
    velocity: [f32; 3],
    color: [f32; 4],
    size: [f32; 2],
    age_sec: f32,
    lifetime_sec: f32,
    additive: bool,
}

/// Compute the X-Ray first-person HUD perspective projection matrix (`hud_projection_matrix`)
/// with `R_VIEWPORT_NEAR = 0.005` so HUD geometry avoids near-wall clipping.
#[inline]
pub fn hud_projection_matrix(hud_fov_factor: f32, aspect: f32, far_plane: f32) -> XRayMatrix {
    fmt_hud_projection_matrix(hud_fov_factor, aspect, far_plane)
}

/// Coordinates HUD projection/FOV zoom, weapon & hand OGF skeletal transforms, firing/reload/holster
/// state machine, ballistic penetration raycasting, wallmarks, particles, and 3D combat audio.
pub struct HudWeaponController {
    pub base_world_fov_deg: f32,
    pub base_hud_fov_factor: f32,
    pub hud_far_plane: f32,
    pub slots: [WeaponSpec; 6],
    pub active_slot: u8,
    pub state: WeaponState,
    pub state_timer_sec: f32,
    pub fire_cooldown_sec: f32,
    pub pending_slot: Option<u8>,
    pub ads_blend: f32,
    pub recoil_kick: f32,
    pub sway_time_sec: f32,
    pub muzzle_flash_timer_sec: f32,
    pub last_muzzle_world_pos: [f32; 3],
    pub visible: bool,

    pub wallmarks: WallmarksManager,
    ogf_cache: BTreeMap<String, Option<CachedHudOgfAsset>>,
    active_sparks: Vec<SparkParticle>,

    pub shots_fired: u64,
    pub reloads_completed: u64,
    pub slots_swapped: u64,
    pub wallmarks_spawned: u64,
    pub sparks_spawned: u64,
    pub last_shot_report: Option<ShotReport>,
}

impl Default for HudWeaponController {
    fn default() -> Self {
        Self {
            base_world_fov_deg: DEFAULT_FOV_DEG,
            base_hud_fov_factor: DEFAULT_HUD_FOV_FACTOR,
            hud_far_plane: DEFAULT_HUD_FAR_PLANE,
            slots: WeaponSpec::default_loadout(),
            active_slot: 3, // AK-74M active out of the box
            state: WeaponState::Idle,
            state_timer_sec: 0.0,
            fire_cooldown_sec: 0.0,
            pending_slot: None,
            ads_blend: 0.0,
            recoil_kick: 0.0,
            sway_time_sec: 0.0,
            muzzle_flash_timer_sec: 0.0,
            last_muzzle_world_pos: [0.0, 1.6, 0.5],
            visible: true,
            wallmarks: WallmarksManager::new(),
            ogf_cache: BTreeMap::new(),
            active_sparks: Vec::new(),
            shots_fired: 0,
            reloads_completed: 0,
            slots_swapped: 0,
            wallmarks_spawned: 0,
            sparks_spawned: 0,
            last_shot_report: None,
        }
    }
}

impl HudWeaponController {
    /// Active equipped [`WeaponSpec`].
    #[inline]
    pub fn active_weapon(&self) -> &WeaponSpec {
        let idx = (self.active_slot.saturating_sub(1) as usize).min(5);
        &self.slots[idx]
    }

    /// Mutable access to active equipped [`WeaponSpec`].
    #[inline]
    pub fn active_weapon_mut(&mut self) -> &mut WeaponSpec {
        let idx = (self.active_slot.saturating_sub(1) as usize).min(5);
        &mut self.slots[idx]
    }

    /// Current HUD FOV factor interpolated smoothly between hip and ADS framing.
    #[inline]
    pub fn current_hud_fov_factor(&self) -> f32 {
        let target_ads = (self.base_hud_fov_factor * 0.82).max(0.28);
        self.base_hud_fov_factor + (target_ads - self.base_hud_fov_factor) * self.ads_blend
    }

    /// Compute DX12 [`CameraConstants`] using `hud_projection_matrix` so first-person weapon/hand
    /// geometry renders in HUD depth range `[0.005 .. hud_far_plane]` without wall clipping.
    pub fn build_hud_camera_constants(&self, camera: &FirstPersonCamera) -> CameraConstants {
        let view = camera.view_matrix();
        let proj = hud_projection_matrix(
            self.current_hud_fov_factor(),
            camera.aspect,
            self.hud_far_plane,
        );
        let c = Dx12CameraConstants::from_xray_matrices(&view, &proj, camera.position);
        CameraConstants {
            view: c.view,
            proj: c.proj,
            view_proj: c.view_proj,
            cam_pos: c.cam_pos,
            padding: c.padding,
        }
    }

    /// Active muzzle flash point light when firing, ready for `RenderBridge` dynamic lighting.
    pub fn muzzle_flash_light(&self) -> Option<GpuLight> {
        if self.muzzle_flash_timer_sec <= 0.0 {
            return None;
        }
        let intensity = (self.muzzle_flash_timer_sec / 0.055).clamp(0.0, 1.0) * 2.6;
        Some(GpuLight {
            position: self.last_muzzle_world_pos,
            range: 7.5,
            color: [1.0, 0.72, 0.32, intensity],
            direction: [0.0, 0.0, 1.0],
            light_type: 0, // Point light
            spot_params: [1.0, 0.5],
            attenuation: [1.0, 0.35],
        })
    }

    /// Active spark & muzzle particles ready for DX12 billboard instancing in Stage (e.6).
    pub fn active_gpu_particles(&self) -> Vec<GpuParticleInstance> {
        self.active_sparks
            .iter()
            .filter_map(|p| {
                if p.lifetime_sec <= 1e-5 {
                    return None;
                }
                let life_left = (1.0 - p.age_sec / p.lifetime_sec).clamp(0.0, 1.0);
                if life_left <= 0.0 {
                    return None;
                }
                let mut color = p.color;
                color[3] *= life_left;
                Some(GpuParticleInstance {
                    position: p.position,
                    rotation: 0.0,
                    size: [
                        p.size[0] * (0.45 + 0.55 * life_left),
                        p.size[1] * (0.45 + 0.55 * life_left),
                    ],
                    color,
                    blend_mode: u32::from(p.additive),
                    ..Default::default()
                })
            })
            .collect()
    }

    /// Gather visible bullet-hole wallmark vertices inside camera view frustum for Stage (e.4).
    pub fn gather_wallmark_vertices(
        &mut self,
        camera: &FirstPersonCamera,
        elapsed_sec: f32,
    ) -> Vec<GpuWallmarkVertex> {
        let view = camera.view_matrix();
        let proj = camera.projection_matrix();
        let vp = matrix_mul_44(&proj, &view);
        let frustum = ClipFrustum::from_view_proj_dx(&vp, frustum_planes::ALL);
        let batches = self
            .wallmarks
            .collect_visible_batches(elapsed_sec, camera.position, &frustum, 0.0);
        let mut out = Vec::new();
        for batch in batches {
            for v in &batch.vertices {
                out.push(GpuWallmarkVertex::from(v));
            }
        }
        out
    }

    /// Stream and cache an `.ogf` HUD model (weapon or hands) from VFS on demand.
    pub fn ensure_ogf_cached(
        &mut self,
        vfs: &VirtualFileSystem,
        vpath: &str,
    ) -> Option<&CachedHudOgfAsset> {
        let key = normalize_virtual_path(vpath)
            .unwrap_or_else(|_| vpath.replace('/', "\\").to_ascii_lowercase());
        if !self.ogf_cache.contains_key(&key) {
            let mut candidates = vec![
                key.clone(),
                if key.starts_with("meshes\\") { key.clone() } else { format!("meshes\\{key}") },
            ];
            if key.contains("hand") {
                candidates.push("meshes\\anomaly_weapons\\hands\\wpn_hand_00.ogf".to_string());
                candidates.push("anomaly_weapons\\hands\\wpn_hand_00.ogf".to_string());
                candidates.push("meshes\\actors\\stalker_hero\\hand_stalker_hud.ogf".to_string());
                candidates.push("actors\\stalker_hero\\hand_stalker_hud.ogf".to_string());
            }
            if let Some(rest) = key.strip_prefix("dynamics\\weapons\\") {
                candidates.push(format!("meshes\\anomaly_weapons\\{rest}"));
                candidates.push(format!("anomaly_weapons\\{rest}"));
                candidates.push(format!("meshes\\dynamics\\weapons\\{rest}"));
            } else if let Some(rest) = key.strip_prefix("meshes\\dynamics\\weapons\\") {
                candidates.push(format!("meshes\\anomaly_weapons\\{rest}"));
                candidates.push(format!("anomaly_weapons\\{rest}"));
                candidates.push(format!("dynamics\\weapons\\{rest}"));
            } else if let Some(rest) = key.strip_prefix("anomaly_weapons\\") {
                candidates.push(format!("meshes\\dynamics\\weapons\\{rest}"));
                candidates.push(format!("dynamics\\weapons\\{rest}"));
                candidates.push(format!("meshes\\anomaly_weapons\\{rest}"));
            } else if let Some(rest) = key.strip_prefix("meshes\\anomaly_weapons\\") {
                candidates.push(format!("meshes\\dynamics\\weapons\\{rest}"));
                candidates.push(format!("dynamics\\weapons\\{rest}"));
                candidates.push(format!("anomaly_weapons\\{rest}"));
            }
            let mut resolved_vpath = key.clone();
            for c in &candidates {
                if vfs.exists(c) {
                    resolved_vpath = c.clone();
                    break;
                }
            }
            let loaded = load_ogf_from_vfs(vfs, &resolved_vpath).ok().map(|model| {
                let mut texture_links = Vec::new();
                if let Some(link) = extract_ogf_texture_link(&model)
                    && let Ok(resolved) = resolve_ogf_texture_links(vfs, &link)
                {
                    texture_links.push(resolved);
                }
                let bind_poses = model
                    .skeleton
                    .as_ref()
                    .and_then(|sk| sk.compute_bind_pose_matrices());
                CachedHudOgfAsset {
                    vpath: resolved_vpath,
                    model,
                    texture_links,
                    bind_poses,
                }
            });
            self.ogf_cache.insert(key.clone(), loaded);
        }
        self.ogf_cache.get(&key).and_then(Option::as_ref)
    }

    /// Preload active weapon & hand `.ogf` assets from `vfs` when available.
    pub fn ensure_active_ogfs_loaded(&mut self, vfs: &VirtualFileSystem) {
        let w_path = self.active_weapon().weapon_hud_ogf.clone();
        let h_path = self.active_weapon().hands_hud_ogf.clone();
        let _ = self.ensure_ogf_cached(vfs, &w_path);
        let _ = self.ensure_ogf_cached(vfs, &h_path);
    }

    /// Compute world-space HUD root matrix anchored to `camera` with ADS blend, holster dip,
    /// reload tilt, and recoil kickback.
    pub fn compute_hud_root_transform(&self, camera: &FirstPersonCamera) -> XRayMatrix {
        let spec = self.active_weapon();
        let right = camera.right();
        let forward = camera.forward();
        // True camera up orthogonalized against forward and right: up = cross(forward, right) in LH
        let cam_up = normalize3([
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ]);

        let t = self.ads_blend.clamp(0.0, 1.0);
        let hip = spec.hip_offset;
        let ads = spec.ads_offset;
        let sway_amp = (1.0 - t * 0.78) * 0.004;
        let sway_x = (self.sway_time_sec * 1.9).sin() * sway_amp;
        let sway_y = (self.sway_time_sec * 3.8).cos() * sway_amp * 0.7;

        let state_drop = match self.state {
            WeaponState::Holstering => {
                let dur = spec.holster_duration_sec.max(0.05);
                let frac = (1.0 - self.state_timer_sec / dur).clamp(0.0, 1.0);
                frac * 0.38
            }
            WeaponState::Unholstering => {
                let dur = spec.unholster_duration_sec.max(0.05);
                let frac = (self.state_timer_sec / dur).clamp(0.0, 1.0);
                frac * 0.38
            }
            WeaponState::Reloading => {
                let dur = spec.reload_duration_sec.max(0.1);
                let phase = 1.0 - (self.state_timer_sec / dur).clamp(0.0, 1.0);
                (phase * std::f32::consts::PI).sin() * 0.16
            }
            _ => 0.0,
        };

        let local_x = hip[0] + (ads[0] - hip[0]) * t + sway_x;
        let local_y = (hip[1] + (ads[1] - hip[1]) * t + sway_y - state_drop + self.recoil_kick * 0.25).max(-0.25);
        let local_z = (hip[2] + (ads[2] - hip[2]) * t - self.recoil_kick + 0.16).max(0.36);

        let world_pos = [
            camera.position[0] + right[0] * local_x + cam_up[0] * local_y + forward[0] * local_z,
            camera.position[1] + right[1] * local_x + cam_up[1] * local_y + forward[1] * local_z,
            camera.position[2] + right[2] * local_x + cam_up[2] * local_y + forward[2] * local_z,
        ];

        let pitch_kick = (self.recoil_kick * 0.9 - state_drop * 1.4).clamp(-0.45, 0.45);
        let (sk, ck) = pitch_kick.sin_cos();
        let pitched_up = [
            cam_up[0] * ck - forward[0] * sk,
            cam_up[1] * ck - forward[1] * sk,
            cam_up[2] * ck - forward[2] * sk,
        ];
        let pitched_fwd = [
            cam_up[0] * sk + forward[0] * ck,
            cam_up[1] * sk + forward[1] * ck,
            cam_up[2] * sk + forward[2] * ck,
        ];

        [
            [right[0], right[1], right[2], 0.0],
            [pitched_up[0], pitched_up[1], pitched_up[2], 0.0],
            [pitched_fwd[0], pitched_fwd[1], pitched_fwd[2], 0.0],
            [world_pos[0], world_pos[1], world_pos[2], 1.0],
        ]
    }

    /// Compute world-space muzzle tip position for muzzle-flash light, spark emitter, and 3D gunfire audio.
    pub fn compute_muzzle_world_pos(&self, camera: &FirstPersonCamera) -> [f32; 3] {
        let root = self.compute_hud_root_transform(camera);
        let m = self.active_weapon().muzzle_local_offset;
        transform_point_mat(&root, m)
    }

    /// Evaluate animated bone matrices (`BonePalette`) for a cached HUD skeleton based on active state.
    pub fn evaluate_asset_bone_palette(&self, asset: &CachedHudOgfAsset) -> Option<BonePalette> {
        let skel = asset.model.skeleton.as_ref()?;
        let bind_poses = asset.bind_poses.as_ref()?;
        let mut model_transforms: Vec<XRayMatrix> =
            bind_poses.iter().map(|b| b.model_bind).collect();

        // Apply state-driven procedural offsets to root and secondary bones (bolt/receiver/mag)
        if let Some(root_m) = model_transforms.first_mut() {
            let kick_offset = matrix_set_xyz_translate(
                [-self.recoil_kick * 0.5, 0.0, 0.0],
                [0.0, 0.0, -self.recoil_kick * 0.8],
            );
            *root_m = matrix_mul_43(root_m, &kick_offset);
        }
        for (idx, m) in model_transforms.iter_mut().enumerate().skip(1) {
            let reload_dip = if self.state == WeaponState::Reloading && idx == 1 {
                -0.08 * (self.sway_time_sec * 6.0).sin().abs()
            } else {
                0.0
            };
            let bolt_travel = if self.recoil_kick > 0.001 && idx <= 2 {
                -self.recoil_kick * 1.2
            } else {
                0.0
            };
            if reload_dip != 0.0 || bolt_travel != 0.0 {
                let delta = matrix_set_xyz_translate([0.0, 0.0, 0.0], [0.0, reload_dip, bolt_travel]);
                *m = matrix_mul_43(m, &delta);
            }
        }

        BonePalette::from_model_transforms(skel, &model_transforms).ok()
    }

    /// Gather skinned/posed HUD draw parts for both the active weapon & hands OGF (or clean fallback).
    pub fn collect_hud_draw_parts(
        &mut self,
        camera: &FirstPersonCamera,
        vfs: Option<&VirtualFileSystem>,
    ) -> Vec<HudDrawPart> {
        if !self.visible {
            return Vec::new();
        }
        if let Some(vfs) = vfs {
            self.ensure_active_ogfs_loaded(vfs);
        }

        let root_transform = self.compute_hud_root_transform(camera);
        let weapon_key = self.active_weapon().weapon_hud_ogf.clone();
        let hands_key = self.active_weapon().hands_hud_ogf.clone();

        let mut parts = Vec::new();
        self.append_ogf_parts(&weapon_key, root_transform, &mut parts);
        self.append_ogf_parts(&hands_key, root_transform, &mut parts);

        if parts.is_empty() {
            build_fallback_viewmodel_parts(self.active_slot, root_transform, &mut parts);
        }
        parts
    }

    fn append_ogf_parts(
        &self,
        ogf_path: &str,
        root_transform: XRayMatrix,
        out: &mut Vec<HudDrawPart>,
    ) {
        let key = normalize_virtual_path(ogf_path)
            .unwrap_or_else(|_| ogf_path.replace('/', "\\").to_ascii_lowercase());
        let Some(Some(asset)) = self.ogf_cache.get(&key) else {
            return;
        };

        if let Some(palette) = self.evaluate_asset_bone_palette(asset)
            && let Ok(skinned_parts) = skin_ogf_hierarchy(&asset.model, &palette)
            && !skinned_parts.is_empty()
        {
            for (idx, sp) in skinned_parts.into_iter().enumerate() {
                if sp.vertices.is_empty() || sp.indices.is_empty() {
                    continue;
                }
                let verts: Vec<XRayVertex> = sp
                    .vertices
                    .into_iter()
                    .map(|sv| {
                        let px = if sv.position[0].is_nan() || sv.position[0].is_infinite() { 0.0 } else { sv.position[0] };
                        let py = if sv.position[1].is_nan() || sv.position[1].is_infinite() { 0.0 } else { sv.position[1] };
                        let pz = if sv.position[2].is_nan() || sv.position[2].is_infinite() { 0.0 } else { sv.position[2].max(-0.32) };
                        let normal = normalize3([
                            if sv.normal[0].is_nan() { 0.0 } else { sv.normal[0] },
                            if sv.normal[1].is_nan() { 1.0 } else { sv.normal[1] },
                            if sv.normal[2].is_nan() { 0.0 } else { sv.normal[2] },
                        ]);
                        let (tangent, _) = vector_generate_orthonormal_basis(normal);
                        XRayVertex {
                            position: [px, py, pz],
                            normal,
                            uv: [
                                if sv.uv[0].is_nan() { 0.0 } else { sv.uv[0] },
                                if sv.uv[1].is_nan() { 0.0 } else { sv.uv[1] },
                            ],
                            tangent,
                        }
                    })
                    .collect();
                let tex = sp
                    .texture
                    .as_ref()
                    .map(|t| t.texture.clone())
                    .or_else(|| asset.texture_links.first().map(|l| l.diffuse_dds.clone()));
                out.push(HudDrawPart {
                    key: format!("{key}#sk_{idx}"),
                    texture_vpath: tex,
                    vertices: verts,
                    indices: sp.indices,
                    world_transform: root_transform,
                });
            }
            return;
        }

        // Static / rigid OGF child hierarchy fallback
        let default_tex = asset.texture_links.first().map(|l| l.diffuse_dds.clone());
        collect_static_ogf_parts(&asset.model, &key, root_transform, default_tex.as_deref(), out);
    }

    /// Advance the weapon state machine, smooth camera ADS FOV zoom, timers, particle sparks,
    /// and handle ballistic firing + hitscan wallmarks + 3D audio for this frame.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        dt_sec: f32,
        input: &EngineInput,
        camera: &mut FirstPersonCamera,
        player_velocity: [f32; 3],
        elapsed_sec: f32,
        world: Option<&LoadedWorld>,
        alife: Option<&mut AlifeManager>,
        audio: Option<&mut AudioManager>,
    ) -> Option<ShotReport> {
        let dt = dt_sec.max(0.0);
        self.sway_time_sec += dt;
        self.fire_cooldown_sec = (self.fire_cooldown_sec - dt).max(0.0);
        self.muzzle_flash_timer_sec = (self.muzzle_flash_timer_sec - dt).max(0.0);
        self.recoil_kick = (self.recoil_kick * (-dt * 18.0).exp()).max(0.0);

        self.step_sparks(dt);
        self.wallmarks.prune_expired(elapsed_sec);

        // 1. Check weapon slot selection (`Slot1..Slot6`)
        let requested_slot = requested_slot_from_input(input);
        if let Some(target_slot) = requested_slot
            && target_slot != self.active_slot
            && !matches!(self.state, WeaponState::Holstering)
        {
            self.pending_slot = Some(target_slot);
            self.state = WeaponState::Holstering;
            self.state_timer_sec = self.active_weapon().holster_duration_sec;
            if let Some(aud) = audio {
                let pos = camera.position;
                let snd = self.active_weapon().holster_sound.clone();
                aud.trigger_weapon_sound(&snd, pos, player_velocity);
            }
            // Early exit from aiming while swapping
            self.update_fov_and_ads(dt, false, camera);
            return None;
        }

        // 2. Process timed state transitions (Holstering -> Unholstering -> Reloading)
        if self.state_timer_sec > 0.0 {
            self.state_timer_sec = (self.state_timer_sec - dt).max(0.0);
        }

        match self.state {
            WeaponState::Holstering => {
                if self.state_timer_sec <= 0.0 {
                    if let Some(next_slot) = self.pending_slot.take() {
                        self.active_slot = next_slot.clamp(1, 6);
                        self.slots_swapped += 1;
                    }
                    self.state = WeaponState::Unholstering;
                    self.state_timer_sec = self.active_weapon().unholster_duration_sec;
                }
                self.update_fov_and_ads(dt, false, camera);
                return None;
            }
            WeaponState::Unholstering => {
                if self.state_timer_sec <= 0.0 {
                    self.state = if input.is_action_down(Action::SecondaryFire) {
                        WeaponState::Aiming
                    } else {
                        WeaponState::Idle
                    };
                }
                self.update_fov_and_ads(dt, false, camera);
                return None;
            }
            WeaponState::Reloading => {
                if self.state_timer_sec <= 0.0 {
                    let spec = self.active_weapon_mut();
                    let needed = spec.clip_capacity.saturating_sub(spec.ammo_in_clip);
                    let transfer = needed.min(spec.reserve_ammo);
                    spec.ammo_in_clip += transfer;
                    spec.reserve_ammo -= transfer;
                    self.reloads_completed += 1;
                    self.state = if input.is_action_down(Action::SecondaryFire) {
                        WeaponState::Aiming
                    } else {
                        WeaponState::Idle
                    };
                }
                self.update_fov_and_ads(dt, false, camera);
                return None;
            }
            _ => {}
        }

        // 3. Reload trigger (`R`)
        let can_reload = {
            let w = self.active_weapon();
            w.clip_capacity > 0 && w.ammo_in_clip < w.clip_capacity && w.reserve_ammo > 0
        };
        if input.is_action_just_pressed(Action::Reload) && can_reload {
            self.state = WeaponState::Reloading;
            self.state_timer_sec = self.active_weapon().reload_duration_sec;
            if let Some(aud) = audio {
                let snd = self.active_weapon().reload_sound.clone();
                aud.trigger_weapon_sound(&snd, camera.position, player_velocity);
            }
            self.update_fov_and_ads(dt, false, camera);
            return None;
        }

        // 4. ADS aiming & FOV zoom
        let aiming_held = input.is_action_down(Action::SecondaryFire);
        self.update_fov_and_ads(dt, aiming_held, camera);

        // 5. Firing input & RPM cadence
        let fire_requested = match self.active_weapon().fire_mode {
            WeaponFireMode::FullAuto => input.is_action_down(Action::PrimaryFire),
            WeaponFireMode::SemiAuto | WeaponFireMode::BoltAction | WeaponFireMode::Melee => {
                input.is_action_just_pressed(Action::PrimaryFire)
            }
        };

        if fire_requested && self.fire_cooldown_sec <= 0.0 {
            let has_ammo = {
                let w = self.active_weapon();
                w.clip_capacity == 0 || w.ammo_in_clip > 0
            };
            if has_ammo {
                let report = self.fire_active_round(
                    camera,
                    player_velocity,
                    elapsed_sec,
                    world,
                    alife,
                    audio,
                );
                self.last_shot_report = Some(report.clone());
                return Some(report);
            } else if can_reload {
                // Auto-reload when pulling trigger on empty clip with reserve ammo
                self.state = WeaponState::Reloading;
                self.state_timer_sec = self.active_weapon().reload_duration_sec;
                if let Some(aud) = audio {
                    let snd = self.active_weapon().reload_sound.clone();
                    aud.trigger_weapon_sound(&snd, camera.position, player_velocity);
                }
                return None;
            }
        }

        // Transition out of Firing back to Aiming / Idle once cadence completes
        if self.fire_cooldown_sec <= 0.0 {
            self.state = if aiming_held {
                WeaponState::Aiming
            } else {
                WeaponState::Idle
            };
        }
        None
    }

    fn update_fov_and_ads(&mut self, dt: f32, aiming: bool, camera: &mut FirstPersonCamera) {
        let ads_speed = self.active_weapon().ads_zoom_speed.max(1.0);
        let alpha = 1.0 - (-ads_speed * dt).exp();
        let target_blend = if aiming { 1.0 } else { 0.0 };
        self.ads_blend += (target_blend - self.ads_blend) * alpha;

        let target_fov = self.base_world_fov_deg
            + (self.active_weapon().ads_fov_deg - self.base_world_fov_deg) * self.ads_blend;
        camera.fov_deg = target_fov;
    }

    fn fire_active_round(
        &mut self,
        camera: &mut FirstPersonCamera,
        player_velocity: [f32; 3],
        elapsed_sec: f32,
        world: Option<&LoadedWorld>,
        mut alife: Option<&mut AlifeManager>,
        mut audio: Option<&mut AudioManager>,
    ) -> ShotReport {
        // Consume clip ammo, enter Firing state, apply recoil kick & muzzle flash
        let spec_snapshot = {
            let w = self.active_weapon_mut();
            if w.clip_capacity > 0 && w.ammo_in_clip > 0 {
                w.ammo_in_clip -= 1;
            }
            w.clone()
        };

        self.state = WeaponState::Firing;
        self.fire_cooldown_sec = spec_snapshot.fire_interval_sec();
        self.recoil_kick = (self.recoil_kick + 0.045).min(0.11);
        self.muzzle_flash_timer_sec = 0.055;
        self.shots_fired += 1;

        let muzzle_world_pos = self.compute_muzzle_world_pos(camera);
        self.last_muzzle_world_pos = muzzle_world_pos;

        // Spawn additive muzzle flash sparks
        let forward = camera.forward();
        self.spawn_muzzle_sparks(muzzle_world_pos, forward);

        // Apply slight upward camera recoil pitch kick
        let pitch_kick = spec_snapshot.recoil_pitch_kick * (1.0 - 0.35 * self.ads_blend);
        camera.pitch = (camera.pitch + pitch_kick).clamp(
            -crate::camera::PITCH_LIMIT,
            crate::camera::PITCH_LIMIT,
        );

        // Trigger 3D gunshot report
        if let Some(aud) = audio.as_deref_mut() {
            aud.play_weapon_fire_3d(&spec_snapshot.shoot_sound, muzzle_world_pos, player_velocity);
        }

        // Ballistic raycast vs CformBvh (with penetration) + live ALife entity hitboxes
        let gamemtl = world.and_then(|w| w.gamemtl.as_ref());
        let cform = world.and_then(|w| w.cform.as_ref());
        let mut impacts = Vec::new();
        let mut wallmarks_spawned = 0usize;
        let mut sparks_spawned = 2usize; // 2 muzzle flash particles

        if let Some(model) = cform {
            let bvh = CformBvh::build(model);
            let mut cur_origin = camera.position;
            let mut traveled = 0.0f32;
            let mut rem_pen = spec_snapshot.penetration_power;

            for _step in 0..MAX_PENETRATION_STEPS {
                let rem_range = spec_snapshot.range_meters - traveled;
                if rem_range <= 0.05 || rem_pen <= 0.05 {
                    break;
                }
                let Some(hit) = bvh.raycast_nearest(cur_origin, forward, rem_range, false) else {
                    break;
                };
                let abs_dist = traveled + hit.range;

                // Check whether an ALife entity stands in front of this static hit
                if let Some(mgr) = alife.as_deref_mut()
                    && let Some((ent_id, ent_dist, ent_pos)) =
                        raycast_alife_entities(mgr, cur_origin, forward, hit.range)
                {
                    let total_ent_dist = traveled + ent_dist;
                    let dmg = compute_damage_falloff(
                        spec_snapshot.base_damage,
                        total_ent_dist,
                        spec_snapshot.range_meters,
                        rem_pen / spec_snapshot.penetration_power.max(0.01),
                    );
                    if let Some(target) = mgr.entities.get_mut(&ent_id) {
                        target.health = (target.health - dmg * 0.01).clamp(0.0, 1.0);
                        target.dirty_net_sync = true;
                    }
                    impacts.push(BallisticImpact {
                        point: ent_pos,
                        normal: [-forward[0], -forward[1], -forward[2]],
                        distance: total_ent_dist,
                        damage: dmg,
                        material_id: None,
                        surface: SurfaceMaterialCategory::Dirt,
                        tri_index: usize::MAX,
                        suppress_wm: true,
                        hit_entity_id: Some(ent_id),
                    });
                    sparks_spawned += self.spawn_impact_sparks(
                        ent_pos,
                        [-forward[0], -forward[1], -forward[2]],
                        SurfaceMaterialCategory::Dirt,
                    );
                }

                let tri_normal = compute_oriented_tri_normal(hit.verts, forward);
                let (surface, _) = resolve_surface_category(Some(hit.material), gamemtl);
                let falloff_scale = rem_pen / spec_snapshot.penetration_power.max(0.01);
                let damage = compute_damage_falloff(
                    spec_snapshot.base_damage,
                    abs_dist,
                    spec_snapshot.range_meters,
                    falloff_scale,
                );

                // Trigger bullet impact 3D sound at surface point
                if let Some(aud) = audio.as_deref_mut() {
                    aud.play_bullet_impact_3d(surface, hit.point, [0.0; 3]);
                }

                // Spawn surface wallmark (bullet hole) via WallmarksManager unless suppressed
                let mtl_suppress_wm = gamemtl
                    .and_then(|lib| {
                        lib.material_by_idx(usize::from(hit.material))
                            .or_else(|| lib.material_by_id(u32::from(hit.material)))
                    })
                    .is_some_and(|m| m.has_flag(gamemtl_flags::FL_SUPPRESS_WALLMARKS));

                if !hit.suppress_wm && !mtl_suppress_wm {
                    let wm_cfg = StaticWallmarkConfig {
                        shader: "effects\\wallmark".to_string(),
                        texture: surface_wallmark_texture(surface).to_string(),
                        size: spec_snapshot.wallmark_size,
                        ttl: 60.0,
                        rotation_deg: ((self.shots_fired * 37) % 360) as f32,
                        time_now: elapsed_sec,
                        ignore_range_opt: true,
                        camera_pos: Some(camera.position),
                        respect_suppress_wm: true,
                    };
                    let wm_record = project_static_wallmark_on_cform(
                        model,
                        &bvh,
                        hit.tri_index,
                        hit.point,
                        &wm_cfg,
                    )
                    .or_else(|| build_quad_wallmark_fallback(hit.point, tri_normal, &wm_cfg));
                    if let Some(rec) = wm_record {
                        self.wallmarks.add(rec);
                        self.wallmarks_spawned += 1;
                        wallmarks_spawned += 1;
                    }
                }

                // Spawn debris / spark particles at impact
                sparks_spawned += self.spawn_impact_sparks(hit.point, tri_normal, surface);

                impacts.push(BallisticImpact {
                    point: hit.point,
                    normal: tri_normal,
                    distance: abs_dist,
                    damage,
                    material_id: Some(hit.material),
                    surface,
                    tri_index: hit.tri_index,
                    suppress_wm: hit.suppress_wm || mtl_suppress_wm,
                    hit_entity_id: None,
                });

                // Evaluate penetration through this material via gamemtl.xr `shoot_factor`
                let pen_cost = material_penetration_cost(hit.material, surface, gamemtl);
                if pen_cost < rem_pen {
                    rem_pen = (rem_pen - pen_cost).max(0.0);
                    traveled = abs_dist + 0.04;
                    cur_origin = [
                        hit.point[0] + forward[0] * 0.04,
                        hit.point[1] + forward[1] * 0.04,
                        hit.point[2] + forward[2] * 0.04,
                    ];
                } else {
                    break;
                }
            }
        } else if let Some(mgr) = alife {
            // Even without level.cform, allow hitting live ALife targets in open arenas
            if let Some((ent_id, ent_dist, ent_pos)) =
                raycast_alife_entities(mgr, camera.position, forward, spec_snapshot.range_meters)
            {
                let dmg = compute_damage_falloff(
                    spec_snapshot.base_damage,
                    ent_dist,
                    spec_snapshot.range_meters,
                    1.0,
                );
                if let Some(target) = mgr.entities.get_mut(&ent_id) {
                    target.health = (target.health - dmg * 0.01).clamp(0.0, 1.0);
                    target.dirty_net_sync = true;
                }
                if let Some(aud) = audio {
                    aud.play_bullet_impact_3d(SurfaceMaterialCategory::Dirt, ent_pos, [0.0; 3]);
                }
                sparks_spawned += self.spawn_impact_sparks(
                    ent_pos,
                    [-forward[0], -forward[1], -forward[2]],
                    SurfaceMaterialCategory::Dirt,
                );
                impacts.push(BallisticImpact {
                    point: ent_pos,
                    normal: [-forward[0], -forward[1], -forward[2]],
                    distance: ent_dist,
                    damage: dmg,
                    material_id: None,
                    surface: SurfaceMaterialCategory::Dirt,
                    tri_index: usize::MAX,
                    suppress_wm: true,
                    hit_entity_id: Some(ent_id),
                });
            }
        }

        let penetrated_surfaces = impacts.len().saturating_sub(1);
        ShotReport {
            slot: spec_snapshot.slot,
            muzzle_world_pos,
            impacts,
            penetrated_surfaces,
            wallmarks_spawned,
            sparks_spawned,
        }
    }

    fn spawn_muzzle_sparks(&mut self, muzzle_pos: [f32; 3], forward: [f32; 3]) {
        for i in 0..2 {
            let spread = if i == 0 { 0.0 } else { 0.35 };
            self.active_sparks.push(SparkParticle {
                position: [
                    muzzle_pos[0] + forward[0] * (0.04 + 0.03 * i as f32),
                    muzzle_pos[1] + forward[1] * (0.04 + 0.03 * i as f32),
                    muzzle_pos[2] + forward[2] * (0.04 + 0.03 * i as f32),
                ],
                velocity: [
                    forward[0] * (3.5 + spread),
                    forward[1] * (3.5 + spread) + 0.25,
                    forward[2] * (3.5 + spread),
                ],
                color: [1.0, 0.78, 0.35, 0.95],
                size: [0.16, 0.16],
                age_sec: 0.0,
                lifetime_sec: 0.07,
                additive: true,
            });
            self.sparks_spawned += 1;
        }
    }

    fn spawn_impact_sparks(
        &mut self,
        point: [f32; 3],
        normal: [f32; 3],
        surface: SurfaceMaterialCategory,
    ) -> usize {
        let (t1, t2) = vector_generate_orthonormal_basis(normal);
        let (color, count) = match surface {
            SurfaceMaterialCategory::Metal => ([1.0, 0.82, 0.35, 1.0], 5usize),
            SurfaceMaterialCategory::Concrete => ([0.82, 0.80, 0.76, 0.9], 4usize),
            SurfaceMaterialCategory::Wood => ([0.62, 0.44, 0.26, 0.9], 4usize),
            SurfaceMaterialCategory::Water => ([0.75, 0.88, 0.98, 0.85], 4usize),
            _ => ([0.52, 0.42, 0.32, 0.85], 3usize),
        };
        let additive = matches!(surface, SurfaceMaterialCategory::Metal);

        for i in 0..count {
            let a = (i as f32) * std::f32::consts::TAU / (count as f32);
            let radial = 1.4 + (i % 3) as f32 * 0.45;
            let vel = [
                normal[0] * 2.4 + (t1[0] * a.cos() + t2[0] * a.sin()) * radial,
                normal[1] * 2.4 + (t1[1] * a.cos() + t2[1] * a.sin()) * radial + 0.9,
                normal[2] * 2.4 + (t1[2] * a.cos() + t2[2] * a.sin()) * radial,
            ];
            self.active_sparks.push(SparkParticle {
                position: [
                    point[0] + normal[0] * 0.015,
                    point[1] + normal[1] * 0.015,
                    point[2] + normal[2] * 0.015,
                ],
                velocity: vel,
                color,
                size: [0.075, 0.075],
                age_sec: 0.0,
                lifetime_sec: if additive { 0.24 } else { 0.36 },
                additive,
            });
            self.sparks_spawned += 1;
        }
        count
    }

    fn step_sparks(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        for p in &mut self.active_sparks {
            p.age_sec += dt;
            if !p.additive {
                p.velocity[1] -= 9.81 * dt;
            }
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.position[2] += p.velocity[2] * dt;
        }
        self.active_sparks.retain(|p| p.age_sec < p.lifetime_sec);
    }
}

fn requested_slot_from_input(input: &EngineInput) -> Option<u8> {
    if input.is_action_just_pressed(Action::Slot1) {
        Some(1)
    } else if input.is_action_just_pressed(Action::Slot2) {
        Some(2)
    } else if input.is_action_just_pressed(Action::Slot3) {
        Some(3)
    } else if input.is_action_just_pressed(Action::Slot4) {
        Some(4)
    } else if input.is_action_just_pressed(Action::Slot5) {
        Some(5)
    } else if input.is_action_just_pressed(Action::Slot6) {
        Some(6)
    } else {
        None
    }
}

fn extract_ogf_texture_link(model: &OgfModel) -> Option<OgfTextureLink> {
    if let Some(link) = &model.texture {
        return Some(link.clone());
    }
    for child in &model.children {
        if let Some(link) = extract_ogf_texture_link(child) {
            return Some(link);
        }
    }
    None
}

fn collect_static_ogf_parts(
    model: &OgfModel,
    key_prefix: &str,
    world_transform: XRayMatrix,
    fallback_tex: Option<&str>,
    out: &mut Vec<HudDrawPart>,
) {
    if let Some(verts) = &model.vertices
        && !model.indices.is_empty()
    {
        let static_verts = verts.to_static_attributes();
        let gpu_verts: Vec<XRayVertex> = static_verts
            .iter()
            .map(static_vertex_to_xray_vertex)
            .collect();
        let tex = model
            .texture
            .as_ref()
            .map(|t| t.texture.clone())
            .or_else(|| fallback_tex.map(str::to_string));
        out.push(HudDrawPart {
            key: format!("{key_prefix}#st_{}", out.len()),
            texture_vpath: tex,
            vertices: gpu_verts,
            indices: model.indices.clone(),
            world_transform,
        });
    }
    for child in &model.children {
        collect_static_ogf_parts(child, key_prefix, world_transform, fallback_tex, out);
    }
}

fn static_vertex_to_xray_vertex(v: &StaticVertex) -> XRayVertex {
    let pos = [
        if v.position[0].is_nan() { 0.0 } else { v.position[0] },
        if v.position[1].is_nan() { 0.0 } else { v.position[1] },
        if v.position[2].is_nan() { 0.0 } else { v.position[2] },
    ];
    let normal = normalize3(v.normal);
    let (tangent, _) = vector_generate_orthonormal_basis(normal);
    XRayVertex {
        position: pos,
        normal,
        uv: [
            if v.uv0[0].is_nan() { 0.0 } else { v.uv0[0] },
            if v.uv0[1].is_nan() { 0.0 } else { v.uv0[1] },
        ],
        tangent,
    }
}

/// Clean procedural first-person viewmodel mesh (weapon body + left/right hand sleeves) used
/// automatically when no `.ogf` HUD assets are mounted in `VirtualFileSystem`.
fn build_fallback_viewmodel_parts(
    slot: u8,
    root_transform: XRayMatrix,
    out: &mut Vec<HudDrawPart>,
) {
    // Receiver / barrel box sized subtly by slot class, strictly with positive Z in front of eye
    let barrel_len = match slot {
        1 => 0.22,
        2 => 0.24,
        5 => 0.62,
        _ => 0.44,
    };
    let (w_verts, w_idx) = make_box_mesh([-0.025, -0.03, 0.02], [0.025, 0.035, barrel_len + 0.05]);
    out.push(HudDrawPart {
        key: format!("__hud_fallback_weapon_{slot}"),
        texture_vpath: None,
        vertices: w_verts,
        indices: w_idx,
        world_transform: root_transform,
    });

    // Left support hand & right grip sleeve proxies
    let (lh_v, lh_i) = make_box_mesh([-0.06, -0.07, 0.14], [-0.015, -0.015, 0.26]);
    out.push(HudDrawPart {
        key: "__hud_fallback_hand_l".to_string(),
        texture_vpath: None,
        vertices: lh_v,
        indices: lh_i,
        world_transform: root_transform,
    });
    let (rh_v, rh_i) = make_box_mesh([0.01, -0.08, 0.03], [0.055, -0.015, 0.12]);
    out.push(HudDrawPart {
        key: "__hud_fallback_hand_r".to_string(),
        texture_vpath: None,
        vertices: rh_v,
        indices: rh_i,
        world_transform: root_transform,
    });
}

fn make_box_mesh(min: [f32; 3], max: [f32; 3]) -> (Vec<XRayVertex>, Vec<u16>) {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;
    let verts = vec![
        // Front (+Z)
        XRayVertex { position: [x0, y0, z1], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [x1, y0, z1], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [x1, y1, z1], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [x0, y1, z1], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        // Back (-Z)
        XRayVertex { position: [x1, y0, z0], normal: [0.0, 0.0, -1.0], uv: [0.0, 1.0], tangent: [-1.0, 0.0, 0.0] },
        XRayVertex { position: [x0, y0, z0], normal: [0.0, 0.0, -1.0], uv: [1.0, 1.0], tangent: [-1.0, 0.0, 0.0] },
        XRayVertex { position: [x0, y1, z0], normal: [0.0, 0.0, -1.0], uv: [1.0, 0.0], tangent: [-1.0, 0.0, 0.0] },
        XRayVertex { position: [x1, y1, z0], normal: [0.0, 0.0, -1.0], uv: [0.0, 0.0], tangent: [-1.0, 0.0, 0.0] },
    ];
    let indices = vec![
        0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 5, 0, 3, 5, 3, 6, 1, 4, 7, 1, 7, 2, 3, 2, 7, 3, 7, 6,
        5, 4, 1, 5, 1, 0,
    ];
    (verts, indices)
}

fn surface_wallmark_texture(surface: SurfaceMaterialCategory) -> &'static str {
    match surface {
        SurfaceMaterialCategory::Concrete => "wm\\wm_bullet_concrete",
        SurfaceMaterialCategory::Metal => "wm\\wm_bullet_metal",
        SurfaceMaterialCategory::Wood => "wm\\wm_bullet_wood",
        SurfaceMaterialCategory::Dirt
        | SurfaceMaterialCategory::Grass
        | SurfaceMaterialCategory::Gravel
        | SurfaceMaterialCategory::Water => "wm\\wm_bullet_ground",
    }
}

fn material_penetration_cost(
    cform_material_id: u16,
    surface: SurfaceMaterialCategory,
    gamemtl: Option<&GameMaterialLibrary>,
) -> f32 {
    if let Some(lib) = gamemtl
        && let Some(mtl) = lib
            .material_by_idx(usize::from(cform_material_id))
            .or_else(|| lib.material_by_id(u32::from(cform_material_id)))
    {
        if mtl.has_flag(gamemtl_flags::FL_PASSABLE) {
            return 0.05;
        }
        if mtl.shoot_factor > 0.0 {
            return mtl.shoot_factor.clamp(0.05, 2.0);
        }
    }
    match surface {
        SurfaceMaterialCategory::Grass | SurfaceMaterialCategory::Water => 0.15,
        SurfaceMaterialCategory::Wood => 0.42,
        SurfaceMaterialCategory::Dirt | SurfaceMaterialCategory::Gravel => 0.70,
        SurfaceMaterialCategory::Metal => 0.90,
        SurfaceMaterialCategory::Concrete => 1.35,
    }
}

fn compute_damage_falloff(
    base_damage: f32,
    distance: f32,
    max_range: f32,
    penetration_scale: f32,
) -> f32 {
    let ratio = (distance / max_range.max(1.0)).clamp(0.0, 1.0);
    let dist_atten = 1.0 - 0.65 * ratio;
    (base_damage * dist_atten * penetration_scale.clamp(0.1, 1.0)).max(0.0)
}

fn build_quad_wallmark_fallback(
    point: [f32; 3],
    normal: [f32; 3],
    cfg: &StaticWallmarkConfig,
) -> Option<StaticWallmarkRecord> {
    let sz = cfg.size.max(0.005);
    let (u_ax, v_ax) = vector_generate_orthonormal_basis(normal);
    let color = pack_argb(255, 48, 48, 48);
    let offset = [
        point[0] + normal[0] * 0.002,
        point[1] + normal[1] * 0.002,
        point[2] + normal[2] * 0.002,
    ];
    let corners = [
        ([-1.0f32, -1.0f32], [0.0f32, 1.0f32]),
        ([1.0, -1.0], [1.0, 1.0]),
        ([1.0, 1.0], [1.0, 0.0]),
        ([-1.0, 1.0], [0.0, 0.0]),
    ];
    let mk = |(c, uv): ([f32; 2], [f32; 2])| WallmarkVertex {
        position: [
            offset[0] + (u_ax[0] * c[0] + v_ax[0] * c[1]) * sz,
            offset[1] + (u_ax[1] * c[0] + v_ax[1] * c[1]) * sz,
            offset[2] + (u_ax[2] * c[0] + v_ax[2] * c[1]) * sz,
        ],
        color,
        uv,
    };
    let v0 = mk(corners[0]);
    let v1 = mk(corners[1]);
    let v2 = mk(corners[2]);
    let v3 = mk(corners[3]);
    let time_end = if cfg.ttl < 0.0 { -1.0 } else { cfg.ttl.max(1.0) };
    Some(StaticWallmarkRecord {
        bounds_center: offset,
        bounds_radius: sz * std::f32::consts::SQRT_2,
        shader: cfg.shader.clone(),
        texture: cfg.texture.clone(),
        time_start: cfg.time_now,
        time_end,
        vertices: vec![v0, v1, v2, v0, v2, v3],
    })
}

fn raycast_alife_entities(
    alife: &AlifeManager,
    origin: [f32; 3],
    dir: [f32; 3],
    max_dist: f32,
) -> Option<(u16, f32, [f32; 3])> {
    let mut best: Option<(u16, f32, [f32; 3])> = None;
    for ent in alife.entities.values() {
        if !ent.online
            || ent.health <= 0.0
            || !matches!(
                ent.role,
                EntityRole::Stalker | EntityRole::Mutant | EntityRole::Trader
            )
        {
            continue;
        }
        // Bounding sphere around torso center (feet.y + 1.0m, radius 0.65m)
        let center = [ent.position[0], ent.position[1] + 1.0, ent.position[2]];
        let oc = [
            center[0] - origin[0],
            center[1] - origin[1],
            center[2] - origin[2],
        ];
        let proj = oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2];
        if proj < 0.0 || proj > max_dist {
            continue;
        }
        let d2 = (oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2]) - proj * proj;
        let r = 0.65f32;
        if d2 <= r * r {
            let t = (proj - (r * r - d2).sqrt()).max(0.0);
            if best.is_none_or(|b| t < b.1) {
                let pt = [
                    origin[0] + dir[0] * t,
                    origin[1] + dir[1] * t,
                    origin[2] + dir[2] * t,
                ];
                best = Some((ent.id, t, pt));
            }
        }
    }
    best
}

fn compute_oriented_tri_normal(verts: [[f32; 3]; 3], incoming_dir: [f32; 3]) -> [f32; 3] {
    let e0 = [
        verts[1][0] - verts[0][0],
        verts[1][1] - verts[0][1],
        verts[1][2] - verts[0][2],
    ];
    let e1 = [
        verts[2][0] - verts[0][0],
        verts[2][1] - verts[0][1],
        verts[2][2] - verts[0][2],
    ];
    let mut n = normalize3([
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ]);
    let dot = n[0] * incoming_dir[0] + n[1] * incoming_dir[1] + n[2] * incoming_dir[2];
    if dot > 0.0 {
        n = [-n[0], -n[1], -n[2]];
    }
    n
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if l2 <= 1e-12 {
        return [0.0, 1.0, 0.0];
    }
    let inv = 1.0 / l2.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

#[inline]
fn transform_point_mat(m: &XRayMatrix, p: [f32; 3]) -> [f32; 3] {
    [
        p[0] * m[0][0] + p[1] * m[1][0] + p[2] * m[2][0] + m[3][0],
        p[0] * m[0][1] + p[1] * m[1][1] + p[2] * m[2][1] + m[3][1],
        p[0] * m[0][2] + p[1] * m[1][2] + p[2] * m[2][2] + m[3][2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use xray_formats::cform::{CformModel, CformTriangle};
    use xray_formats::chunk::ChunkWriter;
    use xray_formats::gamemtl::{gamemtl_chunks, gamemtl_root_chunks, GameMaterial};
    use xray_formats::ogf::{ogf_chunks, skel_vert_type, OgfVisualType, VertBoned1W};

    struct TempDirGuard(PathBuf);
    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn make_temp_dir(tag: &str) -> TempDirGuard {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("xray_hud_weapon_{tag}_{nonce}"));
        fs::create_dir_all(&dir).expect("create temp test dir");
        TempDirGuard(dir)
    }

    fn write_gamemtl_blob(materials: &[GameMaterial]) -> Vec<u8> {
        let mut mtls_chunk = ChunkWriter::new();
        for (i, m) in materials.iter().enumerate() {
            let mut mw = ChunkWriter::new();
            let mut main = Vec::new();
            main.extend_from_slice(&m.id.to_le_bytes());
            main.extend_from_slice(m.name.as_bytes());
            main.push(0);
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_MAIN, &main);
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FLAGS, &m.flags.to_le_bytes());

            let mut phys = Vec::new();
            for v in [
                m.ph_friction,
                m.ph_damping,
                m.ph_spring,
                m.ph_bounce_start_velocity,
                m.ph_bouncing,
            ] {
                phys.extend_from_slice(&v.to_le_bytes());
            }
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_PHYSICS, &phys);

            let mut fac = Vec::new();
            for v in [
                m.shoot_factor,
                m.bounce_damage_factor,
                m.vis_transparency_factor,
                m.snd_occlusion_factor,
            ] {
                fac.extend_from_slice(&v.to_le_bytes());
            }
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FACTORS, &fac);
            mtls_chunk.write_chunk(i as u32, &mw.into_bytes());
        }

        let mut root = ChunkWriter::new();
        root.write_chunk(
            gamemtl_root_chunks::GAMEMTLS_CHUNK_VERSION,
            &1u16.to_le_bytes(),
        );
        let mut counts = Vec::new();
        counts.extend_from_slice(&(materials.len() as u32).to_le_bytes());
        counts.extend_from_slice(&0u32.to_le_bytes());
        root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_AUTOINC, &counts);
        root.write_chunk(
            gamemtl_root_chunks::GAMEMTLS_CHUNK_MTLS,
            &mtls_chunk.into_bytes(),
        );
        root.into_bytes()
    }

    fn build_synthetic_skeletal_ogf_bytes() -> Vec<u8> {
        let mut ogf = ChunkWriter::new();
        let mut hdr = vec![4u8, OgfVisualType::SkeletonAnim as u8];
        hdr.extend_from_slice(&0u16.to_le_bytes());
        for f in [-0.2f32, -0.2, -0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.4] {
            hdr.extend_from_slice(&f.to_le_bytes());
        }
        ogf.write_chunk(ogf_chunks::OGF_HEADER, &hdr);

        // Build 3-vertex 1W skinned triangle mesh bound to bone 0
        let mut verts_bytes = Vec::new();
        verts_bytes.extend_from_slice(&skel_vert_type::FVF_1L.to_le_bytes());
        verts_bytes.extend_from_slice(&3u32.to_le_bytes());
        let v_positions = [[0.0f32, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.2]];
        for p in v_positions {
            let v = VertBoned1W {
                position: p,
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                binormal: [0.0, 0.0, 1.0],
                u: 0.0,
                v: 0.0,
                matrix: 0,
            };
            let raw = unsafe {
                std::slice::from_raw_parts(
                    (&v as *const VertBoned1W).cast::<u8>(),
                    std::mem::size_of::<VertBoned1W>(),
                )
            };
            verts_bytes.extend_from_slice(raw);
        }
        ogf.write_chunk(ogf_chunks::OGF_VERTICES, &verts_bytes);

        let mut idx_bytes = Vec::new();
        idx_bytes.extend_from_slice(&3u32.to_le_bytes());
        for i in [0u16, 1, 2] {
            idx_bytes.extend_from_slice(&i.to_le_bytes());
        }
        ogf.write_chunk(ogf_chunks::OGF_INDICES, &idx_bytes);

        // Skeleton with 2 bones (`wpn_body` root -> `wpn_bolt` child)
        let mut bones_bytes = Vec::new();
        bones_bytes.extend_from_slice(&2u32.to_le_bytes());
        for (name, parent) in [("wpn_body", ""), ("wpn_bolt", "wpn_body")] {
            bones_bytes.extend_from_slice(name.as_bytes());
            bones_bytes.push(0);
            bones_bytes.extend_from_slice(parent.as_bytes());
            bones_bytes.push(0);
            // Fobb: 15 floats (rotate[9] + translate[3] + halfsize[3]) = 60 bytes
            bones_bytes.extend_from_slice(&[0u8; 60]);
        }
        ogf.write_chunk(ogf_chunks::OGF_S_BONE_NAMES, &bones_bytes);
        ogf.into_bytes()
    }

    #[test]
    fn test_hud_projection_and_weapon_state_machine() {
        let mut ctrl = HudWeaponController::default();
        let mut cam = FirstPersonCamera::new(1280, 720);
        let world_proj = cam.projection_matrix();
        let hud_proj = hud_projection_matrix(ctrl.current_hud_fov_factor(), cam.aspect, 150.0);
        assert_ne!(world_proj, hud_proj);

        let mut input = EngineInput::new();
        let initial_ammo = ctrl.active_weapon().ammo_in_clip;
        assert_eq!(ctrl.active_slot, 3);
        assert_eq!(ctrl.state, WeaponState::Idle);

        // Hold RMB -> smooth ADS zoom & Aiming state
        input.on_mouse_down(crate::input::MouseButton::Right);
        for _ in 0..20 {
            ctrl.step(1.0 / 60.0, &input, &mut cam, [0.0; 3], 1.0, None, None, None);
        }
        assert_eq!(ctrl.state, WeaponState::Aiming);
        assert!(cam.fov_deg < DEFAULT_FOV_DEG - 4.0, "ADS must zoom camera FOV");
        assert!(ctrl.current_hud_fov_factor() < ctrl.base_hud_fov_factor);

        // Fire 2 rounds under RPM cadence while holding LMB on full-auto AK-74
        input.on_mouse_down(crate::input::MouseButton::Left);
        let shot1 = ctrl.step(1.0 / 60.0, &input, &mut cam, [0.0; 3], 1.05, None, None, None);
        assert!(shot1.is_some());
        assert_eq!(ctrl.active_weapon().ammo_in_clip, initial_ammo - 1);
        assert!(ctrl.muzzle_flash_light().is_some());

        // Advance >= 0.1s (600 RPM cadence) to fire next automatic round
        let mut saw_second = false;
        for step in 0..8 {
            input.begin_frame();
            if ctrl
                .step(
                    1.0 / 60.0,
                    &input,
                    &mut cam,
                    [0.0; 3],
                    1.05 + step as f32 * (1.0 / 60.0),
                    None,
                    None,
                    None,
                )
                .is_some()
            {
                saw_second = true;
                break;
            }
        }
        assert!(saw_second, "full-auto cadence must fire next round after cooldown");
        input.on_mouse_up(crate::input::MouseButton::Left);
        input.on_mouse_up(crate::input::MouseButton::Right);

        // Trigger Reload (`R`) and advance through clip timer
        input.begin_frame();
        input.on_key_down(EngineInput::VK_KEY_R);
        ctrl.step(1.0 / 60.0, &input, &mut cam, [0.0; 3], 1.2, None, None, None);
        assert_eq!(ctrl.state, WeaponState::Reloading);
        for _ in 0..140 {
            input.begin_frame();
            ctrl.step(1.0 / 60.0, &input, &mut cam, [0.0; 3], 3.5, None, None, None);
        }
        assert_eq!(ctrl.active_weapon().ammo_in_clip, 30);
        assert_eq!(ctrl.reloads_completed, 1);

        // Swap slot to Slot2 (`PM` pistol) via Holstering -> Unholstering
        input.begin_frame();
        input.on_key_down(EngineInput::VK_KEY_2);
        ctrl.step(1.0 / 60.0, &input, &mut cam, [0.0; 3], 3.6, None, None, None);
        assert_eq!(ctrl.state, WeaponState::Holstering);
        for _ in 0..45 {
            input.begin_frame();
            ctrl.step(1.0 / 60.0, &input, &mut cam, [0.0; 3], 4.5, None, None, None);
        }
        assert_eq!(ctrl.active_slot, 2);
        assert_eq!(ctrl.state, WeaponState::Idle);
    }

    #[test]
    fn test_ballistic_penetration_wallmarks_sparks_and_audio() {
        // Two walls in front of +Z:
        // Wall 1 (Wood, mat 0) at z = 6.0m (penetrable!)
        // Wall 2 (Concrete, mat 1) at z = 12.0m (stops round!)
        let verts = vec![
            // Wood wall triangles
            [-3.0f32, 0.0, 6.0],
            [3.0, 0.0, 6.0],
            [3.0, 4.0, 6.0],
            [-3.0, 4.0, 6.0],
            // Concrete wall triangles
            [-4.0, 0.0, 12.0],
            [4.0, 0.0, 12.0],
            [4.0, 4.0, 12.0],
            [-4.0, 4.0, 12.0],
        ];
        let tris = vec![
            CformTriangle::from_raw([0, 1, 2], 0),
            CformTriangle::from_raw([0, 2, 3], 0),
            CformTriangle::from_raw([4, 5, 6], 1),
            CformTriangle::from_raw([4, 6, 7], 1),
        ];
        let cform = CformModel::from_parts(verts, tris).expect("cform");

        let wood = GameMaterial {
            id: 0,
            name: "materials\\wood_solid".to_string(),
            shoot_factor: 0.45,
            ..Default::default()
        };
        let concrete = GameMaterial {
            id: 1,
            name: "materials\\concrete".to_string(),
            shoot_factor: 1.35,
            ..Default::default()
        };
        let gamemtl = GameMaterialLibrary::parse(&write_gamemtl_blob(&[wood, concrete])).expect("gamemtl");

        let world = LoadedWorld {
            level_dir: "levels\\test".to_string(),
            level_data: xray_formats::level::LevelData {
                xrlc_version: 14,
                xrlc_quality: 0,
                shaders: Vec::new(),
                visuals: Vec::new(),
                sectors: Vec::new(),
                portals: Vec::new(),
                glows: Vec::new(),
                glows_raw_len: 0,
                lights: Vec::new(),
            },
            level_geom: None,
            level_geomx: None,
            cform: Some(cform),
            hom: None,
            details: None,
            fog_volumes: None,
            static_particles: None,
            level_ai: None,
            level_gct: None,
            level_spawn: None,
            level_game: None,
            gamemtl: Some(gamemtl),
            sound_occlusion: None,
            static_sounds: None,
            visual_meshes: Vec::new(),
            dynamic_models: BTreeMap::new(),
            weather: crate::world_loader::WorldWeatherState::default(),
        };

        let mut ctrl = HudWeaponController::default();
        let mut cam = FirstPersonCamera::new(1280, 720);
        cam.position = [0.0, 1.65, 0.0];
        let mut audio = AudioManager::new_headless();
        audio.update_listener(xray_audio::AudioListener::from_yaw_pitch(
            cam.position,
            cam.yaw,
            cam.pitch,
            [0.0; 3],
        ));

        let mut input = EngineInput::new();
        input.on_mouse_down(crate::input::MouseButton::Left);

        let report = ctrl
            .step(
                1.0 / 60.0,
                &input,
                &mut cam,
                [0.0; 3],
                2.0,
                Some(&world),
                None,
                Some(&mut audio),
            )
            .expect("shot report");

        assert_eq!(report.impacts.len(), 2, "bullet must penetrate wood wall and hit concrete");
        assert_eq!(report.impacts[0].surface, SurfaceMaterialCategory::Wood);
        assert_eq!(report.impacts[1].surface, SurfaceMaterialCategory::Concrete);
        assert!(report.impacts[0].damage > report.impacts[1].damage);
        assert_eq!(ctrl.wallmarks.len(), 2, "two bullet-hole decals added to WallmarksManager");
        assert!(report.sparks_spawned >= 8);
        assert!(!ctrl.active_gpu_particles().is_empty());

        let wm_verts = ctrl.gather_wallmark_vertices(&cam, 2.0);
        assert!(!wm_verts.is_empty());

        let astats = audio.stats();
        assert_eq!(astats.weapon_sounds_triggered, 1);
        assert_eq!(astats.bullet_impacts_triggered, 2);
    }

    #[test]
    fn test_hud_vfs_ogf_streaming_and_bone_palette_evaluation() {
        let dir = make_temp_dir("ogf_rig");
        let wpn_dir = dir.0.join("meshes\\dynamics\\weapons\\wpn_ak74");
        let hand_dir = dir.0.join("meshes\\actors\\stalker_hero");
        fs::create_dir_all(&wpn_dir).unwrap();
        fs::create_dir_all(&hand_dir).unwrap();

        let ogf_bytes = build_synthetic_skeletal_ogf_bytes();
        fs::write(wpn_dir.join("wpn_ak74_hud.ogf"), &ogf_bytes).unwrap();
        fs::write(hand_dir.join("hand_stalker_hud.ogf"), &ogf_bytes).unwrap();

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_loose_dir(&dir.0);

        let mut ctrl = HudWeaponController::default();
        let cam = FirstPersonCamera::new(1280, 720);
        ctrl.recoil_kick = 0.05;
        let parts = ctrl.collect_hud_draw_parts(&cam, Some(&vfs));
        assert_eq!(parts.len(), 2, "both weapon HUD OGF and hand HUD OGF must stream & skin");
        assert!(parts.iter().all(|p| p.vertices.len() == 3 && p.indices.len() == 3));
    }
}
