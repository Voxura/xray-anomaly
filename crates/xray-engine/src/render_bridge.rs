//! DirectX 12 Render Graph Pipeline Bridge (`xray-engine`)
//!
//! Coordinates [`WorldLoader`] (`LoadedWorld`) and [`AlifeManager`] geometry/textures/lighting/weather
//! into the 7-stage [`Dx12Renderer`] frame pass sequence inside `EngineRuntime::step_frame`:
//!   a) Early-Z Depth Prepass (`draw_depth_prepass`)
//!   b) G-Buffer MRT Pass (`begin_gbuffer_pass` -> `draw_gbuffer_mesh` -> `end_gbuffer_pass`)
//!   c) 3-Split Cascaded Shadow Map Pass (`build_3split_csm_atlas` + `CascadedShadowAtlas` + `draw_csm_shadow_mesh`)
//!   d) Deferred Lighting Combine (`execute_deferred_combine_csm` with CSM PCF, SSFX volumetrics & dynamic lights)
//!   f) Filmic Postprocessing Resolve (`draw_postprocess`: bloom, exposure, Uncharted2/ACES tonemapping -> backbuffer)
//!   e) Forward Alpha & Environment Effects (Early-Z occluded `draw_sky`, `draw_clouds`, `draw_details`,
//!      `draw_wallmarks`, `draw_rain`, `draw_particles`)
//!   g) UI Canvas HUD Overlay (`ui_draw_crosshair`, health/stamina `ui_draw_meter_bar`, `ui_flush`)
//!
//! Note on Pass Ordering `(f)` before `(e)`:
//! `PostprocessRenderer`'s pipeline outputs opaque fullscreen HDR->LDR (`BlendEnable = false`,
//! `DepthEnable = false`) onto the swapchain RTV, while all forward renderers (`SkyRenderer`,
//! `CloudRenderer`, `DetailRenderer`, `WallmarkRenderer`, `RainRenderer`, `ParticleSystemRenderer`)
//! render directly to the swapchain RTV tested against Stage (a)'s Early-Z depth buffer (`sc.dsv_handle()`).
//! Resolving the deferred HDR target to the backbuffer right before forward passes ensures opaque
//! deferred geometry is properly tonemapped AND preserves Early-Z sky/rain/foliage occlusion without
//! wiping forward effects.

use std::collections::BTreeMap;

use windows::core::Result as WinResult;

use crate::alife_manager::{AlifeManager, EntityRole};
use crate::camera::FirstPersonCamera;
use crate::hud_weapon::{HudWeaponController, WeaponState};
use crate::level_transition::LevelTransitionManager;
use crate::ui_inventory::{InventoryManager, UiFontResource};
use crate::ui_pda::PdaManager;
use crate::world_loader::{LoadedDynamicModel, LoadedWorld};
use xray_formats::details::{LevelDetails, DETAIL_ID_EMPTY, DETAIL_SLOT_SIZE};
use xray_formats::hom::{frustum_planes, ClipFrustum};
use xray_formats::ogf::{
    matrix_mul_44, matrix_set_xyz_translate, vector_generate_orthonormal_basis, OgfModel,
    OgfVisualType, StaticVertex, XRAY_IDENTITY_MATRIX,
};
use xray_formats::sun::{build_3split_csm_atlas, CsmAtlasPacking, SunCascadeConfig};
use xray_formats::thm::{texture_stem_to_dds_vpath, ResolvedTextureLinks};
use xray_formats::vfs_bridge::{load_font_from_vfs, load_ltx_tree_from_vfs};
use xray_formats::weather::MixedEnvironment;
use xray_render_dx12::bindless::DescriptorIndex;
use xray_render_dx12::clouds::{CloudSimulation, CloudsPushConstants};
use xray_render_dx12::deferred::GpuLightParams;
use xray_render_dx12::details::{DetailManager, DetailPushConstants, GpuDetailInstance, GpuDetailVertex};
use xray_render_dx12::geometry_abi::{CameraConstants, MeshHandle, XRayVertex};
use xray_render_dx12::lighting::GpuLight;
use xray_render_dx12::particle::GpuParticleInstance;
use xray_render_dx12::postprocess::PostprocessPushConstants;
use xray_render_dx12::rain::RainPushConstants;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::shadow::CascadedShadowAtlas;
use xray_render_dx12::sky::SkyPushConstants;
use xray_render_dx12::wallmarks::{GpuWallmarkVertex, WallmarkPushConstants};
use xray_vfs::VirtualFileSystem;

/// Maximum dynamic lights forwarded to `execute_deferred_combine_csm` per frame.
pub const MAX_ACTIVE_FRAME_LIGHTS: usize = 64;
/// Maximum grass detail instances gathered around camera per batch.
pub const MAX_DETAIL_INSTANCES_PER_BATCH: usize = 512;

/// Configuration knobs for [`RenderBridge`].
#[derive(Debug, Clone, PartialEq)]
pub struct RenderBridgeConfig {
    /// Tile resolution (pixels per side) for each of the 3 CSM shadow cascades (default `1024`).
    pub csm_tile_size: u32,
    /// Cross-cascade boundary fade fraction (`0.0..=0.5`).
    pub csm_blend_band: f32,
    /// Enable SSFX 16-step volumetric sunlight shafts in deferred combine.
    pub volumetric_shafts: bool,
    /// Tonemap operator (`0` = Uncharted2 Filmic, `1` = ACES, `2` = Reinhard, `3` = Pass-through).
    pub tonemap_mode: u32,
    /// Bloom mix factor applied in filmic postprocess.
    pub bloom_factor: f32,
    /// Draw grass details from `level.details` when present.
    pub render_details: bool,
    /// Draw crosshair & health/stamina HUD overlay.
    pub render_hud: bool,
    /// Cascade distances and depth bias config.
    pub sun_cascades: SunCascadeConfig,
}

impl Default for RenderBridgeConfig {
    fn default() -> Self {
        Self {
            csm_tile_size: 512,
            csm_blend_band: 0.1,
            volumetric_shafts: false,
            tonemap_mode: 0,
            bloom_factor: 0.25,
            render_details: true,
            render_hud: true,
            sun_cascades: SunCascadeConfig::default(),
        }
    }
}

/// Player HUD overlay state rendered in Stage (g) UI Canvas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HudOverlayState {
    /// Normalized actor health in `[0.0, 1.0]`.
    pub health: f32,
    /// Normalized actor stamina in `[0.0, 1.0]`.
    pub stamina: f32,
    /// Crosshair visibility flag.
    pub show_crosshair: bool,
}

impl Default for HudOverlayState {
    fn default() -> Self {
        Self {
            health: 1.0,
            stamina: 1.0,
            show_crosshair: true,
        }
    }
}

/// Telemetry counters recorded across the 7 render stages of each GPU frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameRenderStats {
    pub early_z_draws: u32,
    pub gbuffer_static_draws: u32,
    pub gbuffer_dynamic_draws: u32,
    pub csm_cascade_draws: [u32; 3],
    pub combine_light_count: u32,
    pub combine_csm_active: bool,
    pub postprocess_executed: bool,
    pub sky_drawn: bool,
    pub clouds_drawn: bool,
    pub detail_instances_drawn: u32,
    pub wallmark_verts_drawn: u32,
    pub rain_drops_drawn: u32,
    pub particle_instances_drawn: u32,
    pub hud_weapon_draws: u32,
    pub ui_elements_drawn: u32,
}

#[derive(Debug, Clone)]
struct GpuWorldMesh {
    mesh_handle: MeshHandle,
    fastpath_handle: Option<MeshHandle>,
    albedo_tex: DescriptorIndex,
    normal_tex: DescriptorIndex,
    material_tex: DescriptorIndex,
    bsphere_center: [f32; 3],
    bsphere_radius: f32,
    is_static_caster: bool,
}

#[derive(Debug, Clone)]
struct GpuOgfSubMesh {
    mesh_handle: MeshHandle,
    albedo_tex: DescriptorIndex,
    normal_tex: DescriptorIndex,
    material_tex: DescriptorIndex,
}

#[derive(Debug, Clone)]
struct DynamicDrawInstance {
    submeshes: Vec<GpuOgfSubMesh>,
    transform_index: u32,
}

/// Bridges streamed world visuals, dynamic ALife entities, weather, and HUD into [`Dx12Renderer`].
pub struct RenderBridge {
    pub config: RenderBridgeConfig,
    pub csm_atlas: CascadedShadowAtlas,
    pub cloud_sim: CloudSimulation,
    pub wallmark_vertices: Vec<GpuWallmarkVertex>,
    pub frame_particles: Vec<GpuParticleInstance>,
    pub last_stats: FrameRenderStats,
    pub world_synced: bool,

    fallback_albedo: DescriptorIndex,
    fallback_normal: DescriptorIndex,
    fallback_material: DescriptorIndex,
    fallback_white: DescriptorIndex,

    world_meshes: Vec<GpuWorldMesh>,
    world_lights: Vec<GpuLight>,
    detail_meshes: Vec<Vec<GpuDetailVertex>>,
    detail_textures: Vec<Option<DescriptorIndex>>,
    dynamic_ogf_cache: BTreeMap<String, Vec<GpuOgfSubMesh>>,
    hud_mesh_cache: BTreeMap<String, (MeshHandle, DescriptorIndex)>,
    default_cube_mesh: MeshHandle,
    ui_font: Option<UiFontResource>,
    ui_font_attempted: bool,
    ui_hud_tex: Option<DescriptorIndex>,
    ui_equipment_tex: Option<DescriptorIndex>,
}

impl RenderBridge {
    /// Allocate fallback textures, CSM shadow atlas, and default placeholder mesh on `renderer`.
    pub fn new(renderer: &mut Dx12Renderer, mut config: RenderBridgeConfig) -> WinResult<Self> {
        config.csm_tile_size = config.csm_tile_size.max(32);

        let fallback_albedo = load_solid_dds(renderer, "__engine_fallback_albedo.dds", 210, 210, 210, 255)?;
        let fallback_normal = load_solid_dds(renderer, "__engine_fallback_normal.dds", 128, 128, 255, 128)?;
        let fallback_material = load_solid_dds(renderer, "__engine_fallback_material.dds", 0, 128, 64, 1)?;
        let fallback_white = load_solid_dds(renderer, "__engine_fallback_white.dds", 255, 255, 255, 255)?;

        let csm_atlas = CascadedShadowAtlas::new(
            &renderer.device_ctx.device,
            config.csm_tile_size,
            CsmAtlasPacking::Row3x1,
            &mut renderer.bindless_allocator,
            &renderer.bindless_heap,
            renderer.srv_descriptor_size,
        )?;

        let default_cube_mesh = create_unit_actor_box_mesh(renderer)?;

        Ok(Self {
            config,
            csm_atlas,
            cloud_sim: CloudSimulation::default(),
            wallmark_vertices: Vec::new(),
            frame_particles: Vec::new(),
            last_stats: FrameRenderStats::default(),
            world_synced: false,
            fallback_albedo,
            fallback_normal,
            fallback_material,
            fallback_white,
            world_meshes: Vec::new(),
            world_lights: Vec::new(),
            detail_meshes: Vec::new(),
            detail_textures: Vec::new(),
            dynamic_ogf_cache: BTreeMap::new(),
            hud_mesh_cache: BTreeMap::new(),
            default_cube_mesh,
            ui_font: None,
            ui_font_attempted: false,
            ui_hud_tex: None,
            ui_equipment_tex: None,
        })
    }

    /// Mark world GPU caches dirty so the next frame re-uploads static geometry/lights.
    pub fn invalidate_world(&mut self) {
        self.world_meshes.clear();
        self.world_lights.clear();
        self.detail_meshes.clear();
        self.detail_textures.clear();
        self.world_synced = false;
    }

    /// Queue static wallmark decal vertices to draw in Stage (e) until cleared.
    pub fn set_wallmarks(&mut self, vertices: Vec<GpuWallmarkVertex>) {
        self.wallmark_vertices = vertices;
    }

    /// Enqueue a billboard particle instance for the upcoming Stage (e) particle pass.
    pub fn push_particle(&mut self, particle: GpuParticleInstance) {
        self.frame_particles.push(particle);
    }

    /// Upload all static level meshes, textures, detail foliage meshes, and lights from `world`.
    pub fn sync_world(
        &mut self,
        renderer: &mut Dx12Renderer,
        world: &LoadedWorld,
        vfs: Option<&VirtualFileSystem>,
    ) -> WinResult<()> {
        self.world_meshes.clear();
        self.world_lights.clear();
        self.detail_meshes.clear();
        self.detail_textures.clear();

        // 1. Static visual meshes + fastpath shadow/depth buffers + bindless textures
        for visual in &world.visual_meshes {
            let Some(mesh_handle) = upload_resolved_mesh(renderer, &visual.mesh.vertices, &visual.mesh.indices)? else {
                continue;
            };
            let fastpath_handle = match &visual.fastpath {
                Some(fp) => upload_resolved_mesh(renderer, &fp.vertices, &fp.indices)?,
                None => None,
            };

            let (albedo_tex, normal_tex, material_tex) =
                self.resolve_material_textures(renderer, vfs, &visual.texture_links);

            let is_static_caster = !matches!(visual.visual_type, OgfVisualType::Lod);
            self.world_meshes.push(GpuWorldMesh {
                mesh_handle,
                fastpath_handle,
                albedo_tex,
                normal_tex,
                material_tex,
                bsphere_center: visual.bsphere.center,
                bsphere_radius: visual.bsphere.radius,
                is_static_caster,
            });
        }

        // 2. Dynamic lights from `level`
        for lvl_light in &world.level_data.lights {
            let light_type = if lvl_light.is_spot() {
                1
            } else if lvl_light.is_directional() {
                2
            } else {
                0
            };
            let spot_cos = lvl_light.spot_cos_half_angles();
            self.world_lights.push(GpuLight {
                position: lvl_light.position,
                range: lvl_light.range.max(0.1),
                color: lvl_light.diffuse,
                direction: lvl_light.normalized_direction(),
                light_type,
                spot_params: spot_cos,
                attenuation: [lvl_light.attenuation[0], lvl_light.attenuation[1]],
            });
        }

        // 3. Detail grass object meshes (`level.details`)
        if let Some(details) = &world.details {
            for obj in &details.objects {
                let gpu_verts = DetailManager::build_gpu_mesh(obj);
                let tex = if !obj.texture.is_empty() {
                    let vpath = texture_stem_to_dds_vpath(&obj.texture);
                    try_load_vfs_texture(renderer, vfs, &vpath)
                } else {
                    None
                };
                self.detail_meshes.push(gpu_verts);
                self.detail_textures.push(tex);
            }
        }

        self.world_synced = true;
        Ok(())
    }

    fn resolve_material_textures(
        &self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        links: &[ResolvedTextureLinks],
    ) -> (DescriptorIndex, DescriptorIndex, DescriptorIndex) {
        let Some(first) = links.first() else {
            return (self.fallback_albedo, self.fallback_normal, self.fallback_material);
        };
        let albedo = try_load_vfs_texture(renderer, vfs, &first.diffuse_dds)
            .unwrap_or(self.fallback_albedo);
        let normal_vpath = first
            .ext_normal_map_dds
            .as_deref()
            .or_else(|| first.bump.as_ref().map(|b| b.bump_dds.as_str()));
        let normal = normal_vpath
            .and_then(|p| try_load_vfs_texture(renderer, vfs, p))
            .unwrap_or(self.fallback_normal);
        let material = first
            .bump
            .as_ref()
            .and_then(|b| try_load_vfs_texture(renderer, vfs, &b.bump_hash_dds))
            .unwrap_or(self.fallback_material);
        (albedo, normal, material)
    }

    fn ensure_dynamic_ogf_cached(
        &mut self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        key: &str,
        dyn_model: Option<&LoadedDynamicModel>,
    ) -> WinResult<Vec<GpuOgfSubMesh>> {
        if let Some(cached) = self.dynamic_ogf_cache.get(key) {
            return Ok(cached.clone());
        }
        let mut submeshes = Vec::new();
        if let Some(dm) = dyn_model {
            let fallback_mats = self.resolve_material_textures(renderer, vfs, &dm.texture_links);
            self.collect_ogf_submeshes(renderer, vfs, &dm.model, fallback_mats, &mut submeshes)?;
        }
        if submeshes.is_empty() {
            submeshes.push(GpuOgfSubMesh {
                mesh_handle: self.default_cube_mesh,
                albedo_tex: self.fallback_albedo,
                normal_tex: self.fallback_normal,
                material_tex: self.fallback_material,
            });
        }
        self.dynamic_ogf_cache.insert(key.to_string(), submeshes.clone());
        Ok(submeshes)
    }

    fn collect_ogf_submeshes(
        &self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        model: &OgfModel,
        default_mats: (DescriptorIndex, DescriptorIndex, DescriptorIndex),
        out: &mut Vec<GpuOgfSubMesh>,
    ) -> WinResult<()> {
        if let Some(verts) = &model.vertices
            && !model.indices.is_empty()
        {
            let static_verts = verts.to_static_attributes();
            if let Some(mesh_handle) = upload_resolved_mesh(renderer, &static_verts, &model.indices)? {
                let albedo = model
                    .texture
                    .as_ref()
                    .and_then(|link| {
                        let dds = texture_stem_to_dds_vpath(&link.texture);
                        try_load_vfs_texture(renderer, vfs, &dds)
                    })
                    .unwrap_or(default_mats.0);
                out.push(GpuOgfSubMesh {
                    mesh_handle,
                    albedo_tex: albedo,
                    normal_tex: default_mats.1,
                    material_tex: default_mats.2,
                });
            }
        }
        for child in &model.children {
            self.collect_ogf_submeshes(renderer, vfs, child, default_mats, out)?;
        }
        Ok(())
    }

    fn prepare_dynamic_draws(
        &mut self,
        renderer: &mut Dx12Renderer,
        cam_pos: [f32; 3],
        frustum: &ClipFrustum,
        world: Option<&LoadedWorld>,
        alife: Option<&AlifeManager>,
        vfs: Option<&VirtualFileSystem>,
    ) -> WinResult<Vec<DynamicDrawInstance>> {
        let mut matrices: Vec<[[f32; 4]; 4]> = vec![XRAY_IDENTITY_MATRIX];
        let mut draws = Vec::new();

        let Some(alife) = alife else {
            renderer.update_transforms(&matrices)?;
            return Ok(draws);
        };

        const MAX_ENT_VIS_DIST_SQ: f32 = 280.0 * 280.0;

        // Online NPCs / dynamic objects / actors (excluding first-person host actor)
        for ent in alife.entities.values() {
            if !ent.online || ent.role == EntityRole::LocalHostActor {
                continue;
            }
            if distance_sq(cam_pos, ent.position) > MAX_ENT_VIS_DIST_SQ {
                continue;
            }
            if !frustum.test_sphere(ent.position, 2.5) {
                continue;
            }
            let xf = matrix_set_xyz_translate(ent.angle, ent.position);
            let tf_idx = matrices.len() as u32;
            matrices.push(xf);

            let visual_key = ent
                .visual_name
                .as_deref()
                .unwrap_or("__default_stalker_proxy");
            let dyn_model = world.and_then(|w| {
                ent.visual_name
                    .as_deref()
                    .and_then(|v| find_dynamic_model(w, v))
            });
            let submeshes = self.ensure_dynamic_ogf_cached(renderer, vfs, visual_key, dyn_model)?;
            draws.push(DynamicDrawInstance {
                submeshes,
                transform_index: tf_idx,
            });
        }

        // Connected co-op peers (`RemotePeerAvatar`)
        for peer in alife.remote_peers.values() {
            if distance_sq(cam_pos, peer.position) > MAX_ENT_VIS_DIST_SQ {
                continue;
            }
            if !frustum.test_sphere(peer.position, 2.5) {
                continue;
            }
            let xf = matrix_set_xyz_translate([peer.pitch, peer.yaw, 0.0], peer.position);
            let tf_idx = matrices.len() as u32;
            matrices.push(xf);

            let vpath = peer.visual_path();
            let visual_key = vpath.unwrap_or("__default_coop_peer_proxy");
            let dyn_model = world.and_then(|w| vpath.and_then(|v| find_dynamic_model(w, v)));
            let submeshes = self.ensure_dynamic_ogf_cached(renderer, vfs, visual_key, dyn_model)?;
            draws.push(DynamicDrawInstance {
                submeshes,
                transform_index: tf_idx,
            });
        }

        renderer.update_transforms(&matrices)?;
        Ok(draws)
    }

    /// Execute all 7 DX12 render graph passes for one frame (called between `begin_frame()` and `end_frame()`).
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &mut self,
        renderer: &mut Dx12Renderer,
        camera: &FirstPersonCamera,
        world: Option<&LoadedWorld>,
        alife: Option<&AlifeManager>,
        vfs: Option<&VirtualFileSystem>,
        hud: &HudOverlayState,
        dt: f32,
        elapsed_sec: f32,
    ) -> WinResult<FrameRenderStats> {
        self.render_frame_with_hud(renderer, camera, world, alife, vfs, hud, None, dt, elapsed_sec)
    }

    /// Execute all 7 DX12 render graph passes plus optional first-person HUD weapon pass & combat FX.
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame_with_hud(
        &mut self,
        renderer: &mut Dx12Renderer,
        camera: &FirstPersonCamera,
        world: Option<&LoadedWorld>,
        alife: Option<&AlifeManager>,
        vfs: Option<&VirtualFileSystem>,
        hud: &HudOverlayState,
        hud_weapon: Option<&mut HudWeaponController>,
        dt: f32,
        elapsed_sec: f32,
    ) -> WinResult<FrameRenderStats> {
        self.render_frame_with_ui(
            renderer,
            camera,
            world,
            alife,
            vfs,
            hud,
            hud_weapon,
            None,
            None,
            dt,
            elapsed_sec,
        )
    }

    /// Execute all 7 DX12 render graph passes plus first-person HUD weapon, Inventory, PDA, and Dialogue UI.
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame_with_ui(
        &mut self,
        renderer: &mut Dx12Renderer,
        camera: &FirstPersonCamera,
        world: Option<&LoadedWorld>,
        alife: Option<&AlifeManager>,
        vfs: Option<&VirtualFileSystem>,
        hud: &HudOverlayState,
        hud_weapon: Option<&mut HudWeaponController>,
        inventory: Option<&InventoryManager>,
        pda: Option<&mut PdaManager>,
        dt: f32,
        elapsed_sec: f32,
    ) -> WinResult<FrameRenderStats> {
        self.render_frame_with_transition(
            renderer,
            camera,
            world,
            alife,
            vfs,
            hud,
            hud_weapon,
            inventory,
            pda,
            None,
            dt,
            elapsed_sec,
        )
    }

    /// Execute all 7 DX12 render graph passes plus Inventory, PDA, Dialogue, Travel Prompt & Loading Screen UI.
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame_with_transition(
        &mut self,
        renderer: &mut Dx12Renderer,
        camera: &FirstPersonCamera,
        world: Option<&LoadedWorld>,
        alife: Option<&AlifeManager>,
        vfs: Option<&VirtualFileSystem>,
        hud: &HudOverlayState,
        mut hud_weapon: Option<&mut HudWeaponController>,
        inventory: Option<&InventoryManager>,
        pda: Option<&mut PdaManager>,
        level_transition: Option<&LevelTransitionManager>,
        dt: f32,
        elapsed_sec: f32,
    ) -> WinResult<FrameRenderStats> {
        if let Some(w) = world
            && !self.world_synced
        {
            self.sync_world(renderer, w, vfs)?;
        }

        let mut stats = FrameRenderStats::default();
        let cam_constants = camera.to_render_constants();
        renderer.update_camera(&cam_constants)?;

        let view_mat = camera.view_matrix();
        let proj_mat = camera.projection_matrix();
        let vp_xray = matrix_mul_44(&proj_mat, &view_mat);
        let frustum = ClipFrustum::from_view_proj_dx(&vp_xray, frustum_planes::ALL);
        const MAX_VIS_DIST: f32 = 280.0;

        let dynamic_draws = self.prepare_dynamic_draws(
            renderer,
            camera.position,
            &frustum,
            world,
            alife,
            vfs,
        )?;

        // ---------------------------------------------------------------------
        // Stage (a): Early-Z Depth Prepass -> primes swapchain DSV for sky/FX occlusion
        // ---------------------------------------------------------------------
        renderer.clear_backbuffer([0.04, 0.05, 0.08, 1.0]);
        for mesh in &self.world_meshes {
            if mesh.bsphere_radius > 1e-4 {
                let dist_sq = distance_sq(camera.position, mesh.bsphere_center);
                if dist_sq > (MAX_VIS_DIST + mesh.bsphere_radius).powi(2) {
                    continue;
                }
                if !frustum.test_sphere(mesh.bsphere_center, mesh.bsphere_radius) {
                    continue;
                }
                // Early-Z occluder filtering: skip distant tiny objects that do not occlude geometry
                if mesh.bsphere_radius < 0.8 && dist_sq > 40.0 * 40.0 {
                    continue;
                }
            }
            let handle = mesh.fastpath_handle.unwrap_or(mesh.mesh_handle);
            renderer.draw_depth_prepass(handle, u32::MAX)?;
            stats.early_z_draws += 1;
        }
        for inst in &dynamic_draws {
            for sub in &inst.submeshes {
                renderer.draw_depth_prepass(sub.mesh_handle, inst.transform_index)?;
                stats.early_z_draws += 1;
            }
        }

        // ---------------------------------------------------------------------
        // Stage (b): G-Buffer MRT Pass (Albedo, Normal, Material + Depth)
        // ---------------------------------------------------------------------
        renderer.begin_gbuffer_pass()?;
        for mesh in &self.world_meshes {
            if mesh.bsphere_radius > 1e-4 {
                if distance_sq(camera.position, mesh.bsphere_center)
                    > (MAX_VIS_DIST + mesh.bsphere_radius).powi(2)
                {
                    continue;
                }
                if !frustum.test_sphere(mesh.bsphere_center, mesh.bsphere_radius) {
                    continue;
                }
            }
            renderer.draw_gbuffer_mesh(
                mesh.mesh_handle,
                Some(mesh.albedo_tex),
                Some(mesh.normal_tex),
                Some(mesh.material_tex),
                u32::MAX,
            )?;
            stats.gbuffer_static_draws += 1;
        }
        for inst in &dynamic_draws {
            for sub in &inst.submeshes {
                renderer.draw_gbuffer_mesh(
                    sub.mesh_handle,
                    Some(sub.albedo_tex),
                    Some(sub.normal_tex),
                    Some(sub.material_tex),
                    inst.transform_index,
                )?;
                stats.gbuffer_dynamic_draws += 1;
            }
        }
        renderer.end_gbuffer_pass()?;

        // ---------------------------------------------------------------------
        // Stage (c): 3-Split Cascaded Shadow Map (CSM) Pass
        // ---------------------------------------------------------------------
        let env = world.and_then(|w| w.weather.current_env.as_ref());
        let (sun_dir_to_surface, sun_rgb, ambient_rgb) = extract_lighting_env(env);
        let sun_pos = [
            camera.position[0] - sun_dir_to_surface[0] * 500.0,
            camera.position[1] - sun_dir_to_surface[1] * 500.0,
            camera.position[2] - sun_dir_to_surface[2] * 500.0,
        ];

        if let Some(plan) = build_3split_csm_atlas(
            &self.config.sun_cascades,
            camera.position,
            camera.forward(),
            &view_mat,
            &proj_mat,
            sun_pos,
            sun_dir_to_surface,
            self.csm_atlas.tile_size,
            CsmAtlasPacking::Row3x1,
            self.config.csm_blend_band,
            self.config.volumetric_shafts,
        ) {
            self.csm_atlas.update_plan(plan.clone())?;
            if let Some(cmd) = renderer.current_cmd_list.clone() {
                self.csm_atlas.begin_atlas_pass(&cmd);
                for (cascade_idx, slot) in plan.slots.iter().enumerate() {
                    let csm_cam = CameraConstants {
                        view: XRAY_IDENTITY_MATRIX,
                        proj: slot.view_proj,
                        view_proj: slot.view_proj,
                        cam_pos: [camera.position[0], camera.position[1], camera.position[2], 1.0],
                        padding: [0.0; 12],
                    };
                    renderer.update_camera(&csm_cam)?;

                    for mesh in &self.world_meshes {
                        if !mesh.is_static_caster {
                            continue;
                        }
                        if mesh.bsphere_radius > 1e-4 {
                            let dist_sq = distance_sq(camera.position, mesh.bsphere_center);
                            if dist_sq > (slot.split_distance + mesh.bsphere_radius + 32.0).powi(2) {
                                continue;
                            }
                            if !slot.cull_frustum.planes.is_empty()
                                && !slot.cull_frustum.test_sphere(mesh.bsphere_center, mesh.bsphere_radius)
                            {
                                continue;
                            }
                            // Far cascade detail culling: skip clutter smaller than 2.0m in cascade 2
                            if cascade_idx == 2 && mesh.bsphere_radius < 2.0 {
                                continue;
                            }
                            // Mid cascade detail culling: skip tiny clutter smaller than 0.5m in cascade 1
                            if cascade_idx == 1 && mesh.bsphere_radius < 0.5 {
                                continue;
                            }
                        }
                        let caster = mesh.fastpath_handle.unwrap_or(mesh.mesh_handle);
                        renderer.draw_csm_shadow_mesh(&self.csm_atlas, cascade_idx, caster, u32::MAX)?;
                        stats.csm_cascade_draws[cascade_idx] += 1;
                    }
                    for inst in &dynamic_draws {
                        for sub in &inst.submeshes {
                            renderer.draw_csm_shadow_mesh(
                                &self.csm_atlas,
                                cascade_idx,
                                sub.mesh_handle,
                                inst.transform_index,
                            )?;
                            stats.csm_cascade_draws[cascade_idx] += 1;
                        }
                    }
                }
                self.csm_atlas.end_atlas_pass(&cmd);
                // Restore main player camera CBV before deferred combine & forward stages
                renderer.update_camera(&cam_constants)?;
            }
        }

        // ---------------------------------------------------------------------
        // Stage (d): Deferred Lighting Combine (CSM PCF + SSFX Volumetrics + Dynamic Lights)
        // ---------------------------------------------------------------------
        let mut active_lights = gather_visible_lights(
            &self.world_lights,
            camera.position,
            MAX_ACTIVE_FRAME_LIGHTS.saturating_sub(1),
        );
        if let Some(muzzle_lt) = hud_weapon.as_ref().and_then(|hw| hw.muzzle_flash_light()) {
            active_lights.push(muzzle_lt);
        }
        renderer.update_lights(&active_lights)?;
        stats.combine_light_count = active_lights.len() as u32;
        stats.combine_csm_active = self.csm_atlas.plan.is_some();

        let light_params = GpuLightParams {
            sun_direction: [
                -sun_dir_to_surface[0],
                -sun_dir_to_surface[1],
                -sun_dir_to_surface[2],
                0.0,
            ],
            sun_color: [sun_rgb[0], sun_rgb[1], sun_rgb[2], 1.0],
            ambient_color: [ambient_rgb[0], ambient_rgb[1], ambient_rgb[2], 1.0],
            ..Default::default()
        };

        let hdr_srv = renderer.execute_deferred_combine_csm(
            &light_params,
            &self.csm_atlas,
            stats.combine_light_count,
            self.config.volumetric_shafts,
        )?;

        // ---------------------------------------------------------------------
        // Stage (f): Filmic Postprocessing Resolve (HDR Accumulation -> Backbuffer LDR)
        // ---------------------------------------------------------------------
        let width = renderer.width.max(1);
        let height = renderer.height.max(1);
        if let Some(e) = env {
            let pp = renderer.postprocess_manager_mut();
            pp.tonemap.exposure = e.bloom_exposure.max(0.5);
        }
        let post_push = PostprocessPushConstants {
            screen_res_inv: [1.0 / width as f32, 1.0 / height as f32],
            time_seconds: elapsed_sec,
            bloom_factor: self.config.bloom_factor,
            scene_texture_index: hdr_srv.0,
            bloom_texture_index: u32::MAX,
            tonemap_mode: self.config.tonemap_mode,
            color_grading_enabled: 0,
        };
        renderer.draw_postprocess(&post_push)?;
        stats.postprocess_executed = true;

        // ---------------------------------------------------------------------
        // Stage (e): Forward Alpha & Atmospheric Effects (Early-Z Depth Occluded)
        // ---------------------------------------------------------------------
        // e.1 Sky Dome
        let (sky_tex0, sky_tex1, sky_blend, sky_color, sun_glow) = if let Some(e) = env {
            let (s0, s1) = e.sky_dds_vpaths();
            let idx0 = s0
                .as_deref()
                .and_then(|p| try_load_vfs_texture(renderer, vfs, p))
                .map(|d| d.0)
                .unwrap_or(u32::MAX);
            let idx1 = s1
                .as_deref()
                .and_then(|p| try_load_vfs_texture(renderer, vfs, p))
                .map(|d| d.0)
                .unwrap_or(idx0);
            (
                idx0,
                idx1,
                e.weight,
                e.sky_tint_rgba(),
                e.sun_glow_rgba(e.sun_shafts_intensity.max(0.15)),
            )
        } else {
            (
                u32::MAX,
                u32::MAX,
                0.0,
                [0.85, 0.92, 1.0, 1.0],
                [sun_rgb[0], sun_rgb[1], sun_rgb[2], 0.25],
            )
        };
        let sky_push = SkyPushConstants {
            sky_texture0: sky_tex0,
            sky_texture1: sky_tex1,
            blend_factor: sky_blend,
            is_cubemap: 0,
            sky_color,
            sun_color: sun_glow,
            sun_dir: [
                -sun_dir_to_surface[0],
                -sun_dir_to_surface[1],
                -sun_dir_to_surface[2],
                0.0,
            ],
        };
        renderer.draw_sky(&sky_push)?;
        stats.sky_drawn = true;

        // e.2 Scrolling Clouds Dome
        let (wind_vel, wind_dir, clouds_color, clouds_rot, clouds_tex0, clouds_tex1, clouds_blend) =
            if let Some(e) = env {
                let (c0, c1) = e.clouds_dds_vpaths();
                let i0 = c0
                    .as_deref()
                    .and_then(|p| try_load_vfs_texture(renderer, vfs, p))
                    .map(|d| d.0)
                    .unwrap_or(u32::MAX);
                let i1 = c1
                    .as_deref()
                    .and_then(|p| try_load_vfs_texture(renderer, vfs, p))
                    .map(|d| d.0)
                    .unwrap_or(i0);
                (
                    e.wind_velocity,
                    e.wind_direction,
                    e.clouds_color_rgba(),
                    e.sky_rotation,
                    i0,
                    i1,
                    e.weight,
                )
            } else {
                (
                    120.0,
                    0.35,
                    [0.9, 0.92, 0.96, 0.5],
                    0.0,
                    u32::MAX,
                    u32::MAX,
                    0.0,
                )
            };
        self.cloud_sim.step(dt.max(0.0), wind_vel, wind_dir);
        let clouds_push = CloudsPushConstants {
            clouds_texture0: clouds_tex0,
            clouds_texture1: clouds_tex1,
            blend_factor: clouds_blend,
            clouds_rotation: clouds_rot,
            clouds_color,
            sun_color: sun_glow,
            sun_dir: [
                -sun_dir_to_surface[0],
                -sun_dir_to_surface[1],
                -sun_dir_to_surface[2],
                0.0,
            ],
            uv_offset0: self.cloud_sim.uv_offset0,
            uv_offset1: self.cloud_sim.uv_offset1,
            ..Default::default()
        };
        renderer.draw_clouds(&clouds_push)?;
        stats.clouds_drawn = true;

        // e.3 Instanced Grass Details (`level.details`)
        if self.config.render_details
            && let Some(details) = world.and_then(|w| w.details.as_ref())
        {
            stats.detail_instances_drawn = self.draw_level_details(
                renderer,
                details,
                camera.position,
                wind_dir,
                wind_vel,
                elapsed_sec,
            )?;
        }

        // e.4 Static Wallmark Decals
        let mut combined_wallmarks = self.wallmark_vertices.clone();
        if let Some(hw) = hud_weapon.as_deref_mut() {
            let mut hw_wm = hw.gather_wallmark_vertices(camera, elapsed_sec);
            combined_wallmarks.append(&mut hw_wm);
        }
        if !combined_wallmarks.is_empty() {
            let wm_push = WallmarkPushConstants {
                texture_index: self.fallback_white.0,
                ..Default::default()
            };
            renderer.draw_wallmarks(&combined_wallmarks, &wm_push)?;
            stats.wallmark_verts_drawn = combined_wallmarks.len() as u32;
        }

        // e.5 Instanced Rain Streaks
        let (rain_density, rain_color) = env
            .map(|e| (e.rain_density, e.rain_color_rgba()))
            .unwrap_or((0.0, [0.8, 0.85, 0.9, 0.75]));
        renderer.step_rain(camera.position, wind_dir, wind_vel, rain_density, dt.max(0.0));
        let active_rain = renderer.rain_manager_mut().active_count();
        if active_rain > 0 {
            let rain_push = RainPushConstants {
                rain_texture: self.fallback_white.0,
                rain_density,
                rain_color,
                ..Default::default()
            };
            renderer.draw_rain(&rain_push)?;
            stats.rain_drops_drawn = active_rain;
        }

        // e.6 Billboard Particles (alpha debris + additive muzzle flash / metal sparks)
        if let Some(hw) = hud_weapon.as_ref() {
            self.frame_particles.extend(hw.active_gpu_particles());
        }
        if !self.frame_particles.is_empty() {
            let mut alpha_particles = Vec::new();
            let mut additive_particles = Vec::new();
            for p in self.frame_particles.drain(..) {
                if p.blend_mode == 1 {
                    additive_particles.push(p);
                } else {
                    alpha_particles.push(p);
                }
            }
            if !alpha_particles.is_empty() {
                renderer.draw_particles(&alpha_particles, false)?;
                stats.particle_instances_drawn += alpha_particles.len() as u32;
            }
            if !additive_particles.is_empty() {
                renderer.draw_particles(&additive_particles, true)?;
                stats.particle_instances_drawn += additive_particles.len() as u32;
            }
        }

        // e.7 First-Person HUD Weapon & Hand OGF Pass (`hud_projection_matrix` + depth-only clear)
        let is_aiming_ads = hud_weapon
            .as_ref()
            .is_some_and(|hw| hw.state == WeaponState::Aiming);
        if let Some(hw) = hud_weapon.as_deref_mut() {
            let hud_parts = hw.collect_hud_draw_parts(camera, vfs);
            if !hud_parts.is_empty() {
                renderer.clear_depth_stencil(1.0, 0);
                let hud_cam = hw.build_hud_camera_constants(camera);
                renderer.update_camera(&hud_cam)?;

                let mut hud_matrices = Vec::with_capacity(hud_parts.len());
                let mut draw_items = Vec::with_capacity(hud_parts.len());
                for part in hud_parts {
                    let tf_idx = hud_matrices.len() as u32;
                    hud_matrices.push(part.world_transform);
                    let tex_idx = part
                        .texture_vpath
                        .as_deref()
                        .and_then(|vpath| {
                            let dds = texture_stem_to_dds_vpath(vpath);
                            try_load_vfs_texture(renderer, vfs, &dds)
                                .or_else(|| try_load_vfs_texture(renderer, vfs, vpath))
                        })
                        .unwrap_or(self.fallback_albedo);
                    let mesh_handle = if let Some((existing_mesh, _)) = self.hud_mesh_cache.get(&part.key) {
                        *existing_mesh
                    } else {
                        let mh = renderer.create_mesh(&part.vertices, &part.indices)?;
                        self.hud_mesh_cache.insert(part.key.clone(), (mh, tex_idx));
                        mh
                    };
                    draw_items.push((mesh_handle, tex_idx, tf_idx));
                }

                renderer.update_transforms(&hud_matrices)?;
                for (mesh_handle, tex_idx, tf_idx) in draw_items {
                    renderer.draw_mesh_with_lights(
                        mesh_handle,
                        Some(tex_idx),
                        tf_idx,
                        active_lights.len() as u32,
                    )?;
                    stats.hud_weapon_draws += 1;
                }
            }
        }

        // ---------------------------------------------------------------------
        // Stage (g): UI Canvas Overlay (Crosshair, Stalker HUD, Inventory & PDA)
        // ---------------------------------------------------------------------
        if self.config.render_hud {
            if !self.ui_font_attempted
                && let Some(vfs_ref) = vfs
            {
                self.ui_font_attempted = true;
                self.ui_font = try_load_ui_font(renderer, vfs_ref, height);
                self.ui_hud_tex = try_load_vfs_texture(renderer, Some(vfs_ref), "textures\\ui\\ui_hud.dds");
                self.ui_equipment_tex = try_load_vfs_texture(renderer, Some(vfs_ref), "textures\\ui\\ui_icon_equipment.dds");
            }
            let font_ref = self.ui_font.as_ref();

            if let Some(lt) = level_transition
                && lt.is_loading()
            {
                stats.ui_elements_drawn += lt.render_ui(renderer, vfs, width, height, font_ref);
            } else if let Some(inv) = inventory {
                let draw_crosshair = hud.show_crosshair && !is_aiming_ads;
                let hw_ref = hud_weapon.as_deref();
                stats.ui_elements_drawn += inv.render_ui(
                    renderer,
                    hw_ref,
                    width,
                    height,
                    draw_crosshair,
                    hud.stamina,
                    font_ref,
                    self.ui_hud_tex,
                    self.ui_equipment_tex,
                );
                if let Some(pda_mgr) = pda {
                    let level_name = alife
                        .map(|a| a.level_name.as_str())
                        .or_else(|| world.map(|w| w.level_dir.as_str()))
                        .unwrap_or("l01_escape");
                    stats.ui_elements_drawn +=
                        pda_mgr.render_ui(renderer, vfs, width, height, level_name, font_ref);
                }
                if let Some(lt) = level_transition {
                    stats.ui_elements_drawn += lt.render_ui(renderer, vfs, width, height, font_ref);
                }
            } else {
                let fw = width as f32;
                let fh = height as f32;
                if hud.show_crosshair && !is_aiming_ads {
                    renderer.ui_draw_crosshair(
                        [fw * 0.5, fh * 0.5],
                        10.0,
                        4.0,
                        2.0,
                        [0.95, 0.95, 0.95, 0.85],
                    );
                    stats.ui_elements_drawn += 4;
                }
                // Bottom-left Stalker health & stamina bars
                let bar_w = (fw * 0.18).clamp(80.0, 260.0);
                let bar_h = (fh * 0.016).clamp(6.0, 14.0);
                let margin = 16.0;
                renderer.ui_draw_meter_bar(
                    [margin, fh - margin - bar_h * 2.5, bar_w, bar_h],
                    hud.health.clamp(0.0, 1.0),
                    [0.08, 0.08, 0.08, 0.75],
                    [0.82, 0.16, 0.16, 0.95],
                );
                renderer.ui_draw_meter_bar(
                    [margin, fh - margin - bar_h, bar_w, bar_h * 0.75],
                    hud.stamina.clamp(0.0, 1.0),
                    [0.08, 0.08, 0.08, 0.75],
                    [0.2, 0.68, 0.92, 0.95],
                );
                stats.ui_elements_drawn += 4;
                if let Some(lt) = level_transition {
                    stats.ui_elements_drawn += lt.render_ui(renderer, vfs, width, height, font_ref);
                }
            }
            renderer.ui_flush(width, height)?;
        }

        self.last_stats = stats;
        Ok(stats)
    }

    fn draw_level_details(
        &mut self,
        renderer: &mut Dx12Renderer,
        details: &LevelDetails,
        cam_pos: [f32; 3],
        wind_dir: f32,
        wind_vel: f32,
        elapsed_sec: f32,
    ) -> WinResult<u32> {
        if self.detail_meshes.is_empty() || details.slots.is_empty() {
            return Ok(0);
        }
        let (csx, csz) = LevelDetails::world_to_slot_coords(cam_pos[0], cam_pos[2]);
        let radius_slots: i32 = 8;

        // Bucket instances per detail object prototype ID
        let mut buckets: BTreeMap<usize, Vec<GpuDetailInstance>> = BTreeMap::new();
        let mut total_gathered = 0usize;

        'outer: for dz in -radius_slots..=radius_slots {
            for dx in -radius_slots..=radius_slots {
                let sx = csx + dx;
                let sz = csz + dz;
                let Some(slot) = details.query_slot(sx, sz) else {
                    continue;
                };
                if slot.is_empty() {
                    continue;
                }
                let y = slot.y_base_meters() + 0.5 * slot.y_height_meters();
                let hemi = slot.hemi_light();
                let sun = slot.dir_light();
                let rgb = slot.rgb_light();

                for (sub_i, &obj_id) in slot.ids.iter().enumerate() {
                    if obj_id == DETAIL_ID_EMPTY {
                        continue;
                    }
                    let idx = obj_id as usize;
                    if idx >= self.detail_meshes.len() || self.detail_meshes[idx].is_empty() {
                        continue;
                    }
                    let ox = if (sub_i & 1) == 0 { 0.5 } else { 1.5 };
                    let oz = if (sub_i & 2) == 0 { 0.5 } else { 1.5 };
                    let x = (sx as f32) * DETAIL_SLOT_SIZE + ox;
                    let z = (sz as f32) * DETAIL_SLOT_SIZE + oz;
                    buckets.entry(idx).or_default().push(GpuDetailInstance {
                        position: [x, y, z],
                        scale: 1.0,
                        rotation_y: ((sub_i as f32) * 1.57) + ((sx ^ sz) & 3) as f32 * 0.785,
                        alpha: 1.0,
                        hemi_light: hemi,
                        sun_light: sun,
                        color_tint: [rgb[0].max(0.4), rgb[1].max(0.4), rgb[2].max(0.4), 1.0],
                    });
                    total_gathered += 1;
                    if total_gathered >= MAX_DETAIL_INSTANCES_PER_BATCH {
                        break 'outer;
                    }
                }
            }
        }

        let mut drawn = 0u32;
        for (obj_idx, instances) in buckets {
            if instances.is_empty() {
                continue;
            }
            let mesh_verts = &self.detail_meshes[obj_idx];
            if mesh_verts.is_empty() {
                continue;
            }
            let tex_idx = self
                .detail_textures
                .get(obj_idx)
                .and_then(|t| *t)
                .unwrap_or(self.fallback_white);
            let mgr = renderer.detail_manager_mut();
            mgr.clear_instances();
            for inst in &instances {
                mgr.add_instance(*inst);
            }
            let push = DetailPushConstants {
                wind_direction: [wind_dir.cos(), wind_dir.sin()],
                wind_speed: (wind_vel * 0.01).clamp(0.5, 4.0),
                wind_strength: 0.15,
                time_seconds: elapsed_sec,
                texture_index: tex_idx.0,
                alpha_threshold: 0.3,
                custom_param: 0.0,
            };
            renderer.draw_details(mesh_verts, &push)?;
            drawn += instances.len() as u32;
        }

        Ok(drawn)
    }
}

fn find_dynamic_model<'a>(world: &'a LoadedWorld, visual_name: &str) -> Option<&'a LoadedDynamicModel> {
    if let Some(m) = world.dynamic_models.get(visual_name) {
        return Some(m);
    }
    let norm = visual_name.replace('/', "\\").to_ascii_lowercase();
    world.dynamic_models.get(&norm).or_else(|| {
        let with_meshes = format!("meshes\\{norm}");
        world
            .dynamic_models
            .get(&with_meshes)
            .or_else(|| world.dynamic_models.get(&format!("{with_meshes}.ogf")))
    })
}

fn upload_resolved_mesh(
    renderer: &mut Dx12Renderer,
    static_verts: &[StaticVertex],
    indices: &[u16],
) -> WinResult<Option<MeshHandle>> {
    if static_verts.is_empty() || indices.is_empty() {
        return Ok(None);
    }
    let gpu_verts: Vec<XRayVertex> = static_verts
        .iter()
        .map(|v| {
            let n_len_sq = v.normal[0] * v.normal[0] + v.normal[1] * v.normal[1] + v.normal[2] * v.normal[2];
            let normal = if n_len_sq > 1e-10 {
                let inv = 1.0 / n_len_sq.sqrt();
                [v.normal[0] * inv, v.normal[1] * inv, v.normal[2] * inv]
            } else {
                [0.0, 1.0, 0.0]
            };
            let (tangent, _) = vector_generate_orthonormal_basis(normal);
            XRayVertex {
                position: v.position,
                normal,
                uv: v.uv0,
                tangent,
            }
        })
        .collect();
    let handle = renderer.create_mesh(&gpu_verts, indices)?;
    Ok(Some(handle))
}

fn try_load_vfs_texture(
    renderer: &mut Dx12Renderer,
    vfs: Option<&VirtualFileSystem>,
    vpath: &str,
) -> Option<DescriptorIndex> {
    if let Some(existing) = renderer.get_texture(vpath) {
        return Some(existing.descriptor_index);
    }
    let vfs = vfs?;
    let bytes = vfs.read_bytes(vpath).ok()?;
    renderer.load_texture_from_dds(vpath, &bytes).ok()
}

fn try_load_ui_font(
    renderer: &mut Dx12Renderer,
    vfs: &VirtualFileSystem,
    screen_height: u32,
) -> Option<UiFontResource> {
    let fonts_doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/fonts.ltx", &[]).ok()?;
    for section in [
        "ui_font_letterica16_russian",
        "ui_font_arial_14",
        "ui_font_letterica18_russian",
        "hud_font_medium",
    ] {
        if let Ok((cfg, atlas)) = load_font_from_vfs(vfs, &fonts_doc, section, screen_height)
            && let Some(tex_idx) = try_load_vfs_texture(renderer, Some(vfs), &cfg.texture_dds_vpath)
        {
            let size = renderer
                .get_texture_by_index(tex_idx)
                .map(|t| [t.width.max(1), t.height.max(1)])
                .unwrap_or([512, 256]);
            return Some(UiFontResource {
                atlas,
                texture_index: tex_idx,
                texture_size: size,
            });
        }
    }
    None
}

fn load_solid_dds(
    renderer: &mut Dx12Renderer,
    name: &str,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> WinResult<DescriptorIndex> {
    let width = 4u32;
    let height = 4u32;
    let pixel_count = (width * height) as usize;
    let mut dds = vec![0u8; 128 + pixel_count * 4];
    dds[0..4].copy_from_slice(b"DDS ");
    dds[4..8].copy_from_slice(&124u32.to_le_bytes());
    dds[8..12].copy_from_slice(&(0x1u32 | 0x2 | 0x4 | 0x1000).to_le_bytes());
    dds[12..16].copy_from_slice(&height.to_le_bytes());
    dds[16..20].copy_from_slice(&width.to_le_bytes());
    dds[76..80].copy_from_slice(&32u32.to_le_bytes());
    dds[80..84].copy_from_slice(&(0x40u32 | 0x1).to_le_bytes());
    dds[88..92].copy_from_slice(&32u32.to_le_bytes());
    dds[92..96].copy_from_slice(&0x0000_00FFu32.to_le_bytes());
    dds[96..100].copy_from_slice(&0x0000_FF00u32.to_le_bytes());
    dds[100..104].copy_from_slice(&0x00FF_0000u32.to_le_bytes());
    dds[104..108].copy_from_slice(&0xFF00_0000u32.to_le_bytes());

    for i in 0..pixel_count {
        let offset = 128 + i * 4;
        dds[offset] = r;
        dds[offset + 1] = g;
        dds[offset + 2] = b;
        dds[offset + 3] = a;
    }

    renderer.load_texture_from_dds(name, &dds)
}

fn create_unit_actor_box_mesh(renderer: &mut Dx12Renderer) -> WinResult<MeshHandle> {
    // 0.7m x 1.8m x 0.7m Stalker proxy capsule box anchored at feet (y = 0..1.8)
    let hw = 0.35f32;
    let h = 1.8f32;
    let verts = [
        // Front (+Z)
        XRayVertex { position: [-hw, 0.0, hw], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [hw, 0.0, hw], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [hw, h, hw], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [-hw, h, hw], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        // Back (-Z)
        XRayVertex { position: [hw, 0.0, -hw], normal: [0.0, 0.0, -1.0], uv: [0.0, 1.0], tangent: [-1.0, 0.0, 0.0] },
        XRayVertex { position: [-hw, 0.0, -hw], normal: [0.0, 0.0, -1.0], uv: [1.0, 1.0], tangent: [-1.0, 0.0, 0.0] },
        XRayVertex { position: [-hw, h, -hw], normal: [0.0, 0.0, -1.0], uv: [1.0, 0.0], tangent: [-1.0, 0.0, 0.0] },
        XRayVertex { position: [hw, h, -hw], normal: [0.0, 0.0, -1.0], uv: [0.0, 0.0], tangent: [-1.0, 0.0, 0.0] },
    ];
    let indices: [u16; 36] = [
        0, 1, 2, 0, 2, 3, // Front
        4, 5, 6, 4, 6, 7, // Back
        5, 0, 3, 5, 3, 6, // Left
        1, 4, 7, 1, 7, 2, // Right
        3, 2, 7, 3, 7, 6, // Top
        5, 4, 1, 5, 1, 0, // Bottom
    ];
    renderer.create_mesh(&verts, &indices)
}

fn extract_lighting_env(env: Option<&MixedEnvironment>) -> ([f32; 3], [f32; 3], [f32; 3]) {
    if let Some(e) = env {
        let len_sq =
            e.sun_dir[0] * e.sun_dir[0] + e.sun_dir[1] * e.sun_dir[1] + e.sun_dir[2] * e.sun_dir[2];
        let dir = if len_sq > 1e-8 {
            let inv = 1.0 / len_sq.sqrt();
            [e.sun_dir[0] * inv, e.sun_dir[1] * inv, e.sun_dir[2] * inv]
        } else {
            [0.45, -0.78, 0.43]
        };
        let mut amb = e.effective_ambient_rgb(0.5);
        amb[0] = amb[0].max(0.08);
        amb[1] = amb[1].max(0.08);
        amb[2] = amb[2].max(0.10);
        (dir, e.sun_color, amb)
    } else {
        ([0.45, -0.78, 0.43], [1.0, 0.94, 0.85], [0.22, 0.25, 0.30])
    }
}

fn gather_visible_lights(lights: &[GpuLight], cam_pos: [f32; 3], limit: usize) -> Vec<GpuLight> {
    if lights.is_empty() || limit == 0 {
        return Vec::new();
    }
    let mut ranked: Vec<(f32, &GpuLight)> = lights
        .iter()
        .filter_map(|l| {
            let d2 = distance_sq(cam_pos, l.position);
            let max_dist = l.range + 80.0;
            if d2 <= max_dist * max_dist {
                Some((d2, l))
            } else {
                None
            }
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked.into_iter().take(limit).map(|(_, l)| *l).collect()
}

#[inline]
fn distance_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}
