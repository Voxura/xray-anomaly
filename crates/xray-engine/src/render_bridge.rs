//! DirectX 12 Render Graph Pipeline Bridge (`xray-engine`)
//!
//! Coordinates [`WorldLoader`] (`LoadedWorld`) and [`AlifeManager`] geometry/textures/lighting/weather
//! into the [`Dx12Renderer`] frame pass sequence inside `EngineRuntime::step_frame`:
//!   a) Static visibility and target clear
//!   b) G-Buffer MRT Pass (`begin_gbuffer_pass` -> `draw_gbuffer_mesh` -> `end_gbuffer_pass`)
//!   c) 3-Split Cascaded Shadow Map Pass (`build_3split_csm_atlas` + `CascadedShadowAtlas` + `draw_csm_shadow_mesh`)
//!   d) Deferred Lighting Combine (`execute_deferred_combine_csm` with CSM PCF, SSFX volumetrics & dynamic lights)
//!   f) Filmic Postprocessing Resolve (`draw_postprocess`: bloom, exposure, Uncharted2/ACES tonemapping -> backbuffer)
//!   e) Forward Alpha & Environment Effects (scene-depth occluded `draw_sky`, `draw_clouds`,
//!      `draw_wallmarks`, `draw_rain`, `draw_particles`)
//!   g) UI Canvas HUD Overlay (`ui_draw_crosshair`, health/stamina `ui_draw_meter_bar`, `ui_flush`)
//!
//! Note on Pass Ordering `(f)` before `(e)`:
//! `PostprocessRenderer`'s pipeline outputs opaque fullscreen HDR->LDR (`BlendEnable = false`,
//! `DepthEnable = false`) onto the swapchain RTV, while all forward renderers (`SkyRenderer`,
//! `CloudRenderer`, `DetailRenderer`, `WallmarkRenderer`, `RainRenderer`, `ParticleSystemRenderer`)
//! render directly to the swapchain RTV using the depth copied from the G-Buffer.
//! Resolving the deferred HDR target to the backbuffer right before forward passes ensures opaque
//! deferred geometry is properly tonemapped and preserves sky/rain/foliage occlusion without
//! wiping forward effects.

use std::collections::BTreeMap;
use std::sync::Arc;

use windows::core::Result as WinResult;

use crate::alife_manager::{AlifeManager, EntityRole};
use crate::camera::FirstPersonCamera;
use crate::hud_weapon::{HudDrawPart, HudWeaponController, WeaponState};
use crate::level_transition::{LevelTransitionManager, canonical_level_name};
use crate::ui_inventory::{InventoryManager, UiFontResource};
use crate::ui_pda::PdaManager;
use crate::world_loader::{LoadedVisualMesh, LoadedWorld, resolve_level_texture_vpath};
use xray_formats::cform::RaycastOptions;
use xray_formats::details::{
    DETAIL_DEFAULT_PS_SSA_DISCARD, DETAIL_DEFAULT_RADIUS, DetailCacheGrid, DetailDecompressConfig,
    DetailVisibilityConfig, LevelDetails, compute_r_ssa_discard,
};
use xray_formats::hom::{frustum_planes, ClipFrustum};
use xray_formats::lighting::{HEMI_SAMPLE_DIRS, LightTrackConfig, ObjectLightTracker};
use xray_formats::ogf::{
    matrix_mul_44, matrix_set_xyz_translate, matrix_transform_point, vector_generate_orthonormal_basis, OgfModel,
    OgfVisualType, StaticVertex, XRayMatrix, XRAY_IDENTITY_MATRIX,
};
use xray_formats::postprocess::{LuminanceConfig, TonemapParams};
use xray_formats::shaders_xr::{EngineShaderLibrary, R4BlenderSpec};
use xray_formats::spawns::INVALID_OBJECT_ID;
use xray_formats::sun::{build_3split_csm_atlas, CsmAtlasPacking, CsmAtlasPlan, SunCascadeConfig};
use xray_formats::thm::{texture_stem_to_dds_vpath, ResolvedTextureLinks};
use xray_formats::vfs_bridge::{load_font_from_vfs, load_ltx_tree_from_vfs, load_ogf_from_vfs, load_shaders_xr_from_vfs, load_thm_from_vfs, load_ui_texture_atlases_from_vfs, load_xml_from_vfs, resolve_ogf_texture_links};
use xray_formats::weather::MixedEnvironment;
use xray_formats::xml_ui::UiTextureAtlasMap;
use xray_render_dx12::bindless::DescriptorIndex;
use xray_render_dx12::clouds::{CloudSimulation, CloudsPushConstants};
use xray_render_dx12::deferred::{GBufferSurface, GpuLightParams};
use xray_render_dx12::details::{DetailManager, DetailPushConstants, GpuDetailBatch, GpuDetailInstance, GpuDetailMesh};
use xray_render_dx12::geometry_abi::{CameraConstants, MeshHandle, XRayVertex};
use xray_render_dx12::geometry::{HudBlend, StaticMeshRange};
use xray_render_dx12::lighting::GpuLight;
use xray_render_dx12::particle::GpuParticleInstance;
use xray_render_dx12::postprocess::{MONOLITH_SDR_TONEMAP, PostprocessPushConstants};
use xray_render_dx12::rain::RainPushConstants;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::shadow::CascadedShadowAtlas;
use xray_render_dx12::sky::SkyPushConstants;
use xray_render_dx12::wallmarks::{GpuWallmarkVertex, WallmarkPushConstants};
use xray_vfs::{VirtualFileSystem, normalize_virtual_path};

/// Maximum dynamic lights forwarded to `execute_deferred_combine_csm` per frame.
pub const MAX_ACTIVE_FRAME_LIGHTS: usize = 64;

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
    pub scene_depth_copied: bool,
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
    surface: GBufferSurface,
    bsphere_center: [f32; 3],
    bsphere_radius: f32,
    is_static_caster: bool,
}

struct DetailPrototype {
    mesh: GpuDetailMesh,
    texture: DescriptorIndex,
}

struct LevelDetailState {
    cache: DetailCacheGrid,
    config: DetailDecompressConfig,
    prototypes: Vec<Option<DetailPrototype>>,
}

#[derive(Debug, Clone)]
struct GpuOgfSubMesh {
    mesh_handle: MeshHandle,
    albedo_tex: DescriptorIndex,
    normal_tex: DescriptorIndex,
    material_tex: DescriptorIndex,
}

struct DynamicDrawInstance {
    submeshes: Arc<[GpuOgfSubMesh]>,
    transform_index: u32,
    visibility: DynamicVisibility,
}

struct DynamicVisibility {
    camera: bool,
    shadows: [bool; 3],
}

fn classify_dynamic_visibility(
    position: [f32; 3],
    cam_pos: [f32; 3],
    frustum: &ClipFrustum,
    sun_plan: Option<&CsmAtlasPlan>,
) -> Option<DynamicVisibility> {
    const MAX_ENT_VIS_DIST_SQ: f32 = 280.0 * 280.0;
    const ENTITY_RADIUS: f32 = 2.5;
    if distance_sq(cam_pos, position) > MAX_ENT_VIS_DIST_SQ {
        return None;
    }
    let camera = frustum.test_sphere(position, ENTITY_RADIUS);
    // Offscreen models can cast visible shadows. Query the sun's caster volume
    // for each cascade, like r_dsgraph_render_subspace in render_sun_cascade.
    let shadows = std::array::from_fn(|cascade| {
        sun_plan.is_some_and(|plan| plan.slots[cascade].cull_frustum.test_sphere(position, ENTITY_RADIUS))
    });
    (camera || shadows.into_iter().any(|visible| visible))
        .then_some(DynamicVisibility { camera, shadows })
}

#[cfg(test)]
#[test]
fn offscreen_dynamic_caster_uses_the_sun_volume_for_each_cascade() {
    use xray_formats::hom::Plane;

    let camera = FirstPersonCamera::new(1280, 720);
    let mut plan = build_3split_csm_atlas(
        &SunCascadeConfig::default(),
        camera.position,
        camera.forward(),
        &camera.view_matrix(),
        &camera.projection_matrix(),
        [0.0, 400.0, -300.0],
        [0.0, -0.8, 0.6],
        1024,
        CsmAtlasPacking::Row3x1,
        0.1,
        false,
    ).unwrap();
    let exclude = ClipFrustum { planes: vec![Plane { n: [1.0, 0.0, 0.0], d: 0.0 }] };
    let include = ClipFrustum { planes: vec![Plane { n: [-1.0, 0.0, 0.0], d: 0.0 }] };
    plan.slots[0].cull_frustum = exclude.clone();
    plan.slots[1].cull_frustum = include.clone();
    plan.slots[2].cull_frustum = exclude.clone();

    let position = [8.0, 1.0, 15.0];
    let offscreen = classify_dynamic_visibility(position, camera.position, &exclude, Some(&plan)).unwrap();
    let onscreen = classify_dynamic_visibility(position, camera.position, &include, Some(&plan)).unwrap();
    assert!(!offscreen.camera);
    assert!(onscreen.camera);
    assert_eq!(offscreen.shadows, [false, true, false]);
    assert_eq!(onscreen.shadows, offscreen.shadows);
    assert!(classify_dynamic_visibility(position, camera.position, &exclude, None).is_none());
}

#[derive(Clone, Copy, Default)]
struct HudMaterial {
    blend: HudBlend,
    alpha_ref: Option<u8>,
    strict_sorting: bool,
}

impl HudMaterial {
    fn from_model(spec: R4BlenderSpec) -> Option<Self> {
        let R4BlenderSpec::DefferModel { strict_sorting, alpha_blend, alpha_ref, .. } = spec else {
            return None;
        };
        // R4 CBlender_deffer_model::Compile chooses forward blending or an alpha test.
        let forward = strict_sorting || (alpha_blend && alpha_ref < 16);
        Some(Self {
            blend: if forward { HudBlend::Alpha } else { HudBlend::Opaque },
            alpha_ref: (forward || alpha_blend)
                .then_some(if forward { alpha_ref.clamp(0, 255) as u8 } else { 200 }),
            strict_sorting,
        })
    }

    fn sort_order(self, distance_sq: f32) -> (bool, f32) {
        // r_dsgraph_render_hud traverses mapHUD front-to-back and mapHUDSorted back-to-front.
        (self.strict_sorting, if self.strict_sorting { -distance_sq } else { distance_sq })
    }
}

#[derive(Clone, Copy)]
struct HudTextures {
    albedo: DescriptorIndex,
    normal: Option<DescriptorIndex>,
    normal_error: Option<DescriptorIndex>,
    packed_bump: bool,
    material_id: f32,
}

struct HudDrawItem {
    part: HudDrawPart,
    textures: HudTextures,
    material: HudMaterial,
    transform_index: u32,
    order: (bool, f32),
}

struct HudDrawBatch {
    camera: CameraConstants,
    transforms: Vec<XRayMatrix>,
    draws: Vec<HudDrawItem>,
}

#[cfg(test)]
#[test]
fn hud_sort_order_draws_strict_glass_back_to_front_after_opaque() {
    let opaque = HudMaterial::default();
    let glass = HudMaterial { strict_sorting: true, ..HudMaterial::default() };
    assert!(opaque.sort_order(10.0) < glass.sort_order(20.0));
    assert!(opaque.sort_order(1.0) < opaque.sort_order(2.0));
    assert!(glass.sort_order(2.0) < glass.sort_order(1.0));
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
    level_details: Option<LevelDetailState>,
    dynamic_ogf_cache: BTreeMap<String, Arc<[GpuOgfSubMesh]>>,
    retired_world_meshes: Vec<MeshHandle>,
    hud_mesh_cache: BTreeMap<String, MeshHandle>,
    hud_textures: BTreeMap<String, HudTextures>,
    hud_light_tracker: ObjectLightTracker,
    hud_light_config: LightTrackConfig,
    ambient_scale: f32,
    hemisphere_scale: f32,
    hud_materials: Option<BTreeMap<String, HudMaterial>>,
    proxy_submeshes: Arc<[GpuOgfSubMesh]>,
    ui_font: Option<UiFontResource>,
    pause_font_height: u32,
    pause_fonts: BTreeMap<&'static str, Option<UiFontResource>>,
    native_tonemap: Option<TonemapParams>,
    native_middle_gray: [f32; 4],
    geometry_lod: f32,
    inventory_fonts: BTreeMap<&'static str, UiFontResource>,
    ui_font_attempted: bool,
    ui_equipment_tex: Option<DescriptorIndex>,
    ui_grid_tex: Option<DescriptorIndex>,
    inventory_atlas: Option<UiTextureAtlasMap>,
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
            level_details: None,
            dynamic_ogf_cache: BTreeMap::new(),
            retired_world_meshes: Vec::new(),
            hud_mesh_cache: BTreeMap::new(),
            hud_textures: BTreeMap::new(),
            hud_light_tracker: ObjectLightTracker::default(),
            hud_light_config: LightTrackConfig::default(),
            ambient_scale: 1.0,
            hemisphere_scale: 1.0,
            hud_materials: None,
            proxy_submeshes: Arc::from([GpuOgfSubMesh {
                mesh_handle: default_cube_mesh,
                albedo_tex: fallback_albedo,
                normal_tex: fallback_normal,
                material_tex: fallback_material,
            }]),
            ui_font: None,
            pause_font_height: 0,
            pause_fonts: BTreeMap::new(),
            native_tonemap: None,
            native_middle_gray: [1.0, 0.0, 1.0, 0.0],
            geometry_lod: 0.75,
            inventory_fonts: BTreeMap::new(),
            ui_font_attempted: false,
            ui_equipment_tex: None,
            ui_grid_tex: None,
            inventory_atlas: None,
        })
    }

    /// Mark world GPU caches dirty so the next frame re-uploads static geometry/lights.
    pub fn invalidate_world(&mut self) {
        self.retire_static_meshes();
        for submeshes in std::mem::take(&mut self.dynamic_ogf_cache).into_values() {
            self.retired_world_meshes.extend(submeshes.iter().map(|mesh| mesh.mesh_handle));
        }
        self.world_lights.clear();
        self.level_details = None;
        self.hud_light_tracker = ObjectLightTracker::default();
        self.world_synced = false;
    }

    fn retire_static_meshes(&mut self) {
        for mesh in self.world_meshes.drain(..) {
            self.retired_world_meshes.push(mesh.mesh_handle);
            self.retired_world_meshes.extend(mesh.fastpath_handle);
        }
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
        self.retire_static_meshes();
        for handle in self.retired_world_meshes.drain(..) {
            renderer.destroy_mesh(handle);
        }
        self.world_lights.clear();
        self.level_details = None;
        self.world_synced = false;
        renderer.with_asset_uploads(|renderer| self.upload_world(renderer, world, vfs))?;
        self.world_synced = true;
        Ok(())
    }

    fn upload_world<'world>(
        &mut self,
        renderer: &mut Dx12Renderer,
        world: &'world LoadedWorld,
        vfs: Option<&VirtualFileSystem>,
    ) -> WinResult<()> {
        const GEOMETRY_PAGE_BYTES: usize = 8 * 1024 * 1024;

        // 1. Static visual meshes + fastpath shadow/depth buffers + bindless textures
        let shader_library = vfs.and_then(|vfs| load_shaders_xr_from_vfs(vfs).ok());
        let mut world_materials = BTreeMap::new();
        let mut upload_page = |renderer: &mut Dx12Renderer,
                               vertices: &[XRayVertex],
                               indices: &[u16],
                               ranges: &[StaticMeshRange],
                               visuals: &[(&'world LoadedVisualMesh, bool)]| -> WinResult<()> {
            if visuals.is_empty() {
                return Ok(());
            }
            let mut handles = renderer.create_static_mesh_page(vertices, indices, ranges)?.into_iter();
            for &(visual, has_fastpath) in visuals {
                let mesh_handle = handles.next().expect("one range per visual");
                let fastpath_handle = has_fastpath.then(|| handles.next().expect("one range per fastpath"));

                let shader = visual.shader.as_ref();
                let ((albedo_tex, normal_tex, material_tex), surface) = *world_materials
                    .entry(shader.map_or("", |shader| shader.raw.as_str()))
                    .or_insert_with(|| {
                        let spec = shader.zip(shader_library.as_ref())
                            .and_then(|(shader, library)| library.find_blender(&shader.shader))
                            .and_then(R4BlenderSpec::from_blender_shader);
                        (
                            self.resolve_material_textures(renderer, vfs, &visual.texture_links),
                            Self::resolve_world_surface(renderer, vfs, &world.level_dir, spec, &visual.texture_links),
                        )
                    });

                self.world_meshes.push(GpuWorldMesh {
                    mesh_handle,
                    fastpath_handle,
                    albedo_tex,
                    normal_tex,
                    material_tex,
                    surface,
                    bsphere_center: visual.bsphere.center,
                    bsphere_radius: visual.bsphere.radius,
                    is_static_caster: !matches!(visual.visual_type, OgfVisualType::Lod),
                });
            }
            Ok(())
        };
        let (mut vertices, mut indices, mut ranges, mut pending) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for visual in &world.visual_meshes {
            if visual.mesh.vertices.is_empty() || visual.mesh.indices.is_empty() {
                continue;
            }
            let fastpath = visual.fastpath.as_ref()
                .filter(|mesh| !mesh.vertices.is_empty() && !mesh.indices.is_empty());
            for mesh in std::iter::once(&visual.mesh).chain(fastpath) {
                let vertex_start = vertices.len();
                let index_start = indices.len();
                vertices.extend(mesh.vertices.iter().map(world_vertex));
                indices.extend_from_slice(&mesh.indices);
                ranges.push(StaticMeshRange {
                    vertices: vertex_start..vertices.len(),
                    indices: index_start..indices.len(),
                });
            }
            pending.push((visual, fastpath.is_some()));
            if std::mem::size_of_val(vertices.as_slice()) + std::mem::size_of_val(indices.as_slice())
                >= GEOMETRY_PAGE_BYTES
            {
                upload_page(renderer, &vertices, &indices, &ranges, &pending)?;
                vertices.clear();
                indices.clear();
                ranges.clear();
                pending.clear();
            }
        }
        upload_page(renderer, &vertices, &indices, &ranges, &pending)?;

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
        let (radius, detail_config) = load_detail_config(vfs);
        if let Some(details) = &world.details
            && world.cform.is_some()
            && radius > 0
            && !details.objects.is_empty()
        {
            let mut prototypes = Vec::with_capacity(details.objects.len());
            for obj in &details.objects {
                let gpu_verts = DetailManager::build_gpu_mesh(obj);
                let texture = if !obj.texture.is_empty() {
                    let vpath = vfs
                        .and_then(|vfs| resolve_level_texture_vpath(vfs, &world.level_dir, &obj.texture))
                        .unwrap_or_else(|| texture_stem_to_dds_vpath(&obj.texture));
                    try_load_vfs_texture(renderer, vfs, &vpath)
                } else {
                    Some(self.fallback_white)
                };
                let prototype = if let Some(texture) = texture {
                    renderer.create_detail_mesh(&gpu_verts)?.map(|mesh| DetailPrototype { mesh, texture })
                } else {
                    None
                };
                prototypes.push(prototype);
            }
            self.level_details = Some(LevelDetailState {
                cache: DetailCacheGrid::new(details, radius),
                config: detail_config,
                prototypes,
            });
        }

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

    fn resolve_world_surface(
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        level_dir: &str,
        spec: Option<R4BlenderSpec>,
        links: &[ResolvedTextureLinks],
    ) -> GBufferSurface {
        match spec {
            Some(R4BlenderSpec::BmmD { r_detail, g_detail, b_detail, a_detail, .. }) => {
                let (Some(vfs), Some(base)) = (vfs, links.first()) else {
                    return GBufferSurface::Opaque;
                };
                let mut load = |stem: &str| {
                    let path = resolve_level_texture_vpath(vfs, level_dir, stem)?;
                    try_load_vfs_texture(renderer, Some(vfs), &path)
                };
                let Some(mask) = load(&format!("{}_mask", base.diffuse_name)) else {
                    return GBufferSurface::Opaque;
                };
                GBufferSurface::Terrain {
                    mask,
                    layers: [r_detail, g_detail, b_detail, a_detail].map(|layer| load(&layer)),
                    detail_scale: base.detail_scale,
                }
            }
            // The deferred base aref shader uses def_aref (200/255). The
            // stored CBlender_deffer_aref threshold applies to its forward blend pass.
            Some(R4BlenderSpec::Tree { alpha_blend: true, .. })
            | Some(R4BlenderSpec::DefferAref { alpha_blend: false, .. }) => GBufferSurface::AlphaTest(200),
            _ => GBufferSurface::Opaque,
        }
    }

    fn ensure_dynamic_ogf_cached(
        &mut self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        visual_name: &str,
        world: Option<&LoadedWorld>,
        allow_proxy: bool,
    ) -> WinResult<Arc<[GpuOgfSubMesh]>> {
        let Ok(key) = normalize_virtual_path(visual_name) else {
            return Ok(Arc::from([]));
        };
        let proxy = |submeshes: &Arc<[GpuOgfSubMesh]>| {
            if submeshes.is_empty() && (vfs.is_none() || allow_proxy) {
                Arc::clone(&self.proxy_submeshes)
            } else {
                Arc::clone(submeshes)
            }
        };
        if let Some(cached) = self.dynamic_ogf_cache.get(&key) {
            return Ok(proxy(cached));
        }
        let mut submeshes = Vec::new();
        let fallback = (self.fallback_albedo, self.fallback_normal, self.fallback_material);
        for path in model_search_paths(&key, world.map(|world| world.level_dir.as_str())) {
            if let Some(model) = world.and_then(|world| world.dynamic_models.get(&path)) {
                let material = self.resolve_material_textures(renderer, vfs, &model.texture_links);
                self.collect_ogf_submeshes(renderer, vfs, &model.model, material, &mut submeshes)?;
            } else if let Some(vfs) = vfs && vfs.exists(&path) {
                match load_ogf_from_vfs(vfs, &path) {
                    Ok(model) => self.collect_ogf_submeshes(renderer, Some(vfs), &model, fallback, &mut submeshes)?,
                    Err(error) => eprintln!("[model] {path}: {error}"),
                }
            }
            if !submeshes.is_empty() {
                break;
            }
        }
        let submeshes: Arc<[GpuOgfSubMesh]> = submeshes.into();
        let result = proxy(&submeshes);
        self.dynamic_ogf_cache.insert(key, Arc::clone(&submeshes));
        Ok(result)
    }

    fn collect_ogf_submeshes(
        &self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        model: &OgfModel,
        default_mats: (DescriptorIndex, DescriptorIndex, DescriptorIndex),
        out: &mut Vec<GpuOgfSubMesh>,
    ) -> WinResult<()> {
        let indices = model.progressive_indices(1.0, &[]);
        if let Some(verts) = &model.vertices
            && !indices.is_empty()
        {
            let static_verts = verts.to_static_attributes();
            if let Some(mesh_handle) = upload_resolved_mesh(renderer, &static_verts, indices)? {
                let (albedo_tex, normal_tex, material_tex) = vfs.zip(model.texture.as_ref())
                    .and_then(|(vfs, link)| resolve_ogf_texture_links(vfs, link).ok())
                    .map(|link| self.resolve_material_textures(renderer, vfs, &[link]))
                    .unwrap_or(default_mats);
                out.push(GpuOgfSubMesh {
                    mesh_handle,
                    albedo_tex,
                    normal_tex,
                    material_tex,
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
        sun_plan: Option<&CsmAtlasPlan>,
        world: Option<&LoadedWorld>,
        alife: Option<&AlifeManager>,
        vfs: Option<&VirtualFileSystem>,
        inventory: Option<&InventoryManager>,
    ) -> WinResult<Vec<DynamicDrawInstance>> {
        let mut matrices: Vec<[[f32; 4]; 4]> = vec![XRAY_IDENTITY_MATRIX];
        let mut draws = Vec::new();

        let Some(alife) = alife else {
            renderer.update_transforms(&matrices)?;
            return Ok(draws);
        };

        let current_level_id = alife.game_graph.as_ref().zip(world).and_then(|(graph, world)| {
            let name = canonical_level_name(&world.level_dir);
            graph.header.levels.iter().find(|level| level.name.eq_ignore_ascii_case(&name)).map(|level| level.id)
        });
        // Only unowned entities from this level can be drawn as independent world objects.
        for ent in alife.entities.values() {
            if !ent.online || ent.parent_id != INVALID_OBJECT_ID || ent.role == EntityRole::LocalHostActor {
                continue;
            }
            if let (Some(graph), Some(level_id), Some(_)) = (&alife.game_graph, current_level_id, &ent.source_scop)
                && !graph.vertices.get(usize::from(ent.game_vertex_id)).is_some_and(|vertex| vertex.level_id == level_id)
            {
                continue;
            }
            let Some(visibility) = classify_dynamic_visibility(ent.position, cam_pos, frustum, sun_plan) else {
                continue;
            };
            let configured_visual = inventory.and_then(|inventory| inventory.world_visual_for_section(&ent.section));
            let Some(visual_key) = ent.visual_name.as_deref().filter(|name| !name.is_empty()).or(configured_visual) else {
                continue;
            };
            let mut submeshes = self.ensure_dynamic_ogf_cached(renderer, vfs, visual_key, world, false)?;
            if submeshes.is_empty() && let Some(configured) = configured_visual.filter(|name| *name != visual_key) {
                submeshes = self.ensure_dynamic_ogf_cached(renderer, vfs, configured, world, false)?;
            }
            if submeshes.is_empty() {
                continue;
            }
            let xf = matrix_set_xyz_translate(ent.angle, ent.position);
            let tf_idx = matrices.len() as u32;
            matrices.push(xf);

            draws.push(DynamicDrawInstance {
                submeshes,
                transform_index: tf_idx,
                visibility,
            });
        }

        // Connected co-op peers (`RemotePeerAvatar`)
        for peer in alife.remote_peers.values() {
            let Some(visibility) = classify_dynamic_visibility(peer.position, cam_pos, frustum, sun_plan) else {
                continue;
            };
            let xf = matrix_set_xyz_translate([peer.pitch, peer.yaw, 0.0], peer.position);
            let tf_idx = matrices.len() as u32;
            matrices.push(xf);

            let vpath = peer.visual_path();
            let visual_key = vpath.unwrap_or("__default_coop_peer_proxy");
            let submeshes = self.ensure_dynamic_ogf_cached(renderer, vfs, visual_key, world, true)?;
            draws.push(DynamicDrawInstance {
                submeshes,
                transform_index: tf_idx,
                visibility,
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

    /// Monolith replaces level rendering with a separate menu pass.
    pub(crate) fn render_pause_menu(
        &mut self,
        renderer: &mut Dx12Renderer,
        menu: &crate::ui_pause::PauseMenu,
        vfs: Option<&VirtualFileSystem>,
        single_player: bool,
    ) -> WinResult<()> {
        let (width, height) = (renderer.stats().width, renderer.stats().height);
        if self.pause_font_height != height {
            self.pause_font_height = height;
            self.pause_fonts.clear();
        }
        let sections = menu.font_sections();
        if let Some(vfs) = vfs && sections.iter().any(|section| !self.pause_fonts.contains_key(section)) {
            let fonts = load_ltx_tree_from_vfs(vfs, "$game_configs$/fonts.ltx", &[]).ok();
            for section in sections {
                self.pause_fonts.entry(section).or_insert_with(|| {
                    load_ui_font(renderer, vfs, fonts.as_ref()?, section, height)
                });
            }
        }
        let fallback = self.ui_font.as_ref().or_else(|| self.pause_fonts.values().find_map(Option::as_ref));
        let fonts = sections.map(|section| self.pause_fonts.get(section).and_then(Option::as_ref).or(fallback));
        renderer.clear_backbuffer([0.025, 0.032, 0.033, 1.0]);
        menu.render(renderer, fonts, [width, height], single_player);
        renderer.ui_flush(width, height)
    }

    pub(crate) fn reset_pause_fonts(&mut self) {
        self.pause_fonts.clear();
    }

    pub(crate) fn load_render_profile(&mut self, profile: &str) {
        self.geometry_lod = 0.75;
        self.ambient_scale = 1.0;
        self.hemisphere_scale = 1.0;
        self.hud_light_config = LightTrackConfig::default();
        let mut tonemap = TonemapParams::default();
        let mut luminance = LuminanceConfig::default();
        let mut has_native_settings = false;
        for line in profile.lines() {
            let mut words = line.split_whitespace();
            let (Some(cvar), Some(value)) = (words.next(), words.next()) else { continue };
            if cvar == "r2_tonemap" {
                match value {
                    "on" | "1" => luminance.tonemap_enabled = true,
                    "off" | "0" => luminance.tonemap_enabled = false,
                    _ => {}
                }
                has_native_settings = true;
                continue;
            }
            let Ok(value) = value.parse::<f32>() else { continue };
            match cvar {
                "r__geometry_lod" if (0.1..=1.5).contains(&value) => {
                    self.geometry_lod = value;
                    continue;
                }
                "r2_sun_lumscale_amb" if (0.0..=3.0).contains(&value) => {
                    self.ambient_scale = value;
                    continue;
                }
                "r2_sun_lumscale_hemi" if (0.0..=3.0).contains(&value) => {
                    self.hemisphere_scale = value;
                    continue;
                }
                "r2_dhemi_sky_scale" if (0.0..=100.0).contains(&value) => {
                    self.hud_light_config.dhemi_sky_scale = value;
                    continue;
                }
                _ => {}
            }
            if cvar.starts_with("r2_tonemap_") {
                match cvar {
                    "r2_tonemap_middlegray" if (0.0..=2.0).contains(&value) => luminance.middle_gray = value,
                    "r2_tonemap_lowlum" if (0.0001..=1.0).contains(&value) => luminance.low_lum = value,
                    "r2_tonemap_amount" if (0.0..=1.0).contains(&value) => luminance.amount = value,
                    _ => continue,
                }
                has_native_settings = true;
                continue;
            }
            let Some(parameter) = cvar.strip_prefix("r2_tnmp_") else { continue };
            if !(0.0..=20.0).contains(&value) {
                continue;
            }
            match parameter {
                "a" => tonemap.a = value,
                "b" => tonemap.b = value,
                "c" => tonemap.c = value,
                "d" => tonemap.d = value,
                "e" => tonemap.e = value,
                "f" => tonemap.f = value,
                "w" => tonemap.w = value,
                "exposure" => tonemap.exposure = value,
                "gamma" => tonemap.gamma = value,
                "onoff" if value <= 1.0 => tonemap.enabled = value == 1.0,
                _ => continue,
            }
            has_native_settings = true;
        }
        self.native_tonemap = has_native_settings.then_some(tonemap);
        let amount = if luminance.tonemap_enabled { luminance.amount } else { 0.0 };
        self.native_middle_gray = [
            1.0 + (luminance.middle_gray - 1.0) * amount,
            amount,
            1.0 + (luminance.low_lum - 1.0) * amount,
            0.0,
        ];
    }

    fn prepare_hud_draws(
        &mut self,
        renderer: &mut Dx12Renderer,
        weapon: &mut HudWeaponController,
        camera: &FirstPersonCamera,
        vfs: Option<&VirtualFileSystem>,
    ) -> Option<HudDrawBatch> {
        let parts = weapon.collect_hud_draw_parts(camera, vfs);
        if parts.is_empty() {
            return None;
        }
        if self.hud_materials.is_none() && let Some(vfs) = vfs {
            self.hud_materials = Some(load_hud_materials(vfs));
        }
        let mut batch = HudDrawBatch {
            camera: weapon.build_hud_camera_constants(camera),
            transforms: Vec::with_capacity(parts.len()),
            draws: Vec::with_capacity(parts.len()),
        };
        for part in parts {
            let transform_index = batch.transforms.len() as u32;
            batch.transforms.push(part.world_transform);
            let material = part.shader.as_deref()
                .and_then(|name| self.hud_materials.as_ref()?.get(name))
                .copied()
                .unwrap_or_default();
            let center = matrix_transform_point(&part.world_transform, part.visual_center);
            let order = material.sort_order(distance_sq(camera.position, center));
            let textures = self.resolve_hud_textures(
                renderer, vfs, part.texture_vpath.as_deref(), material.blend,
            );
            batch.draws.push(HudDrawItem { part, textures, material, transform_index, order });
        }
        batch.draws.sort_by(|a, b| a.order.0.cmp(&b.order.0).then_with(|| a.order.1.total_cmp(&b.order.1)));
        Some(batch)
    }

    fn resolve_hud_textures(
        &mut self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        texture: Option<&str>,
        blend: HudBlend,
    ) -> HudTextures {
        let mut textures = HudTextures {
            albedo: self.fallback_albedo,
            normal: None,
            normal_error: None,
            packed_bump: false,
            material_id: 1.0,
        };
        let (Some(texture), Some(vfs)) = (texture, vfs) else { return textures };
        let key = texture_stem_to_dds_vpath(texture);
        if blend == HudBlend::Opaque && let Some(cached) = self.hud_textures.get(&key) {
            return *cached;
        }
        textures.albedo = try_load_vfs_texture(renderer, Some(vfs), &key)
            .unwrap_or(self.fallback_albedo);
        if blend == HudBlend::Opaque {
            if let Ok(Some(params)) = load_thm_from_vfs(vfs, texture) {
                let links = params.resolve_texture_links(texture);
                textures.material_id = links.material_id;
                if let Some(bump) = links.bump {
                    textures.normal = try_load_vfs_texture(renderer, Some(vfs), &bump.bump_dds);
                    if textures.normal.is_some() {
                        textures.packed_bump = true;
                        textures.normal_error = try_load_vfs_texture(renderer, Some(vfs), &bump.bump_hash_dds);
                    }
                }
                if textures.normal.is_none() && let Some(path) = links.ext_normal_map_dds {
                    textures.normal = try_load_vfs_texture(renderer, Some(vfs), &path);
                }
            }
            self.hud_textures.insert(key, textures);
        }
        textures
    }

    /// A light count selects the forward pass for blended materials.
    fn draw_hud_pass(
        &mut self,
        renderer: &mut Dx12Renderer,
        batch: &HudDrawBatch,
        forward_lights: Option<u32>,
    ) -> WinResult<u32> {
        let mut draws = batch.draws.iter()
            .filter(|draw| (draw.material.blend != HudBlend::Opaque) == forward_lights.is_some())
            .peekable();
        if draws.peek().is_none() {
            return Ok(0);
        }
        renderer.update_camera(&batch.camera)?;
        renderer.update_transforms(&batch.transforms)?;
        let mut count = 0;
        for draw in draws {
            let part = &draw.part;
            let mesh = if let Some(cached) = self.hud_mesh_cache.get(&part.key).filter(|_| !part.transient) {
                *cached
            } else {
                let mesh = renderer.create_hud_mesh(&part.vertices, &part.indices)?;
                if !part.transient {
                    self.hud_mesh_cache.insert(part.key.clone(), mesh);
                }
                mesh
            };
            let textures = draw.textures;
            let material = draw.material;
            let result = match forward_lights {
                Some(lights) => renderer.draw_hud_mesh_with_material(
                    mesh, Some(textures.albedo), draw.transform_index, lights,
                    material.blend, material.alpha_ref,
                ),
                None => renderer.draw_gbuffer_surface(
                    mesh, Some(textures.albedo), textures.normal, textures.normal_error,
                    draw.transform_index, GBufferSurface::Hud {
                        alpha_ref: material.alpha_ref,
                        packed_bump: textures.packed_bump,
                        material_id: textures.material_id,
                    },
                ),
            };
            if part.transient {
                // Draw recording retains the resources until the submission fence completes.
                renderer.destroy_mesh(mesh);
            }
            result?;
            count += 1;
        }
        renderer.restore_normal_viewport();
        Ok(count)
    }

    fn set_hud_environment(
        &mut self,
        renderer: &mut Dx12Renderer,
        world: &LoadedWorld,
        weather: &MixedEnvironment,
        camera: &FirstPersonCamera,
        vfs: Option<&VirtualFileSystem>,
        dt: f32,
        light_params: &mut GpuLightParams,
    ) {
        let (a, b) = weather.sky_env_dds_vpaths();
        let [first, second] = [a, b].map(|path| {
            let index = try_load_vfs_texture(renderer, vfs, path.as_deref()?)?;
            renderer.get_texture_by_index(index)?.is_cubemap.then_some(index.0)
        });
        let Some(index) = first.or(second) else { return };
        light_params.hud_environment_indices = [first.unwrap_or(index), second.unwrap_or(index), u32::MAX, u32::MAX];

        let tracker = &mut self.hud_light_tracker;
        let config = self.hud_light_config;
        if let Some(collision) = &world.cform {
            let bvh = collision.build_bvh();
            let first_sample = tracker.result_count == 0;
            // CROS traces five static hemisphere rays per update. The camera
            // supplies the first-person sample origin until the actor visual is available.
            for sample in tracker.next_hemi_ray_indices(config.dhemi_count) {
                let direction = xray_formats::skinning::normalize_vec3(HEMI_SAMPLE_DIRS[sample]);
                tracker.ray_clear[sample] = bvh.raycast(
                    camera.position,
                    direction,
                    50.0,
                    RaycastOptions { only_first: true, ..Default::default() },
                ).is_empty();
            }
            tracker.accumulate_sky_hemi(config.dhemi_sky_scale);
            tracker.step_smooth(if first_sample { 1.0 } else { dt }, config.lt_smooth);
        }
        let hemi = tracker.to_hemi_gpu_params(0.0, [0.0; 4], [0.0; 4]);
        light_params.hud_hemi_positive = hemi.pos_faces;
        light_params.hud_hemi_negative = hemi.neg_faces;
        for axis in 0..3 {
            // R4 phase_combine binds ambient separately from the environment tint.
            light_params.hud_environment_color[axis] = (2.0 * weather.hemi_color[axis] + 1e-5)
                * 2.0 * self.hemisphere_scale;
            light_params.hud_ambient_hemi[axis] =
                (2.0 * weather.ambient_color[axis]).max(0.001) * self.ambient_scale;
        }
        light_params.hud_environment_color[3] = weather.weight;
        light_params.hud_ambient_hemi[3] = hemi.material[0];
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
        let ssa_discard = self.ssa_discard(renderer, camera);
        const MAX_VIS_DIST: f32 = 280.0;

        let env = world.and_then(|w| w.weather.current_env.as_ref());
        let (sun_dir_to_surface, sun_rgb, ambient_rgb) = extract_lighting_env(env);
        let sun_pos = std::array::from_fn(|axis| {
            camera.position[axis] - sun_dir_to_surface[axis] * 500.0
        });
        let sun_plan = build_3split_csm_atlas(
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
        );
        let is_aiming_ads = hud_weapon.as_ref().is_some_and(|hw| hw.state == WeaponState::Aiming);
        let hud_draws = hud_weapon.as_deref_mut()
            .and_then(|weapon| self.prepare_hud_draws(renderer, weapon, camera, vfs));

        // ---------------------------------------------------------------------
        // Stage (a): Clear the swapchain targets
        // ---------------------------------------------------------------------
        renderer.clear_backbuffer([0.04, 0.05, 0.08, 1.0]);
        // ---------------------------------------------------------------------
        // Stage (b): G-Buffer MRT Pass (Albedo, Normal, Material + Depth)
        // ---------------------------------------------------------------------
        renderer.begin_gbuffer_pass()?;
        if let Some(batch) = &hud_draws {
            stats.hud_weapon_draws = self.draw_hud_pass(renderer, batch, None)?;
            if stats.hud_weapon_draws > 0 {
                renderer.update_camera(&cam_constants)?;
            }
        }
        let dynamic_draws = self.prepare_dynamic_draws(
            renderer,
            camera.position,
            &frustum,
            sun_plan.as_ref(),
            world,
            alife,
            vfs,
            inventory,
        )?;
        for mesh in &self.world_meshes {
            if mesh.bsphere_radius > 1e-4 {
                let dist_sq = distance_sq(camera.position, mesh.bsphere_center);
                if dist_sq > (MAX_VIS_DIST + mesh.bsphere_radius).powi(2)
                    || below_ssa_discard(mesh.bsphere_radius, dist_sq, ssa_discard)
                {
                    continue;
                }
                if !frustum.test_sphere(mesh.bsphere_center, mesh.bsphere_radius) {
                    continue;
                }
            }
            renderer.draw_gbuffer_surface(
                mesh.mesh_handle,
                Some(mesh.albedo_tex),
                Some(mesh.normal_tex),
                Some(mesh.material_tex),
                u32::MAX,
                mesh.surface,
            )?;
            stats.gbuffer_static_draws += 1;
        }
        for inst in dynamic_draws.iter().filter(|instance| instance.visibility.camera) {
            for sub in inst.submeshes.iter() {
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
        if self.config.render_details && let Some(world) = world {
            stats.detail_instances_drawn = self.draw_level_details(
                renderer, world, camera, &frustum, &vp_xray, ssa_discard,
                env.map_or(0.35, |weather| weather.wind_direction),
                env.map_or(120.0, |weather| weather.wind_velocity),
                elapsed_sec,
            )?;
        }
        renderer.end_gbuffer_pass()?;
        // R4 reuses the opaque scene depth for forward rendering.
        renderer.copy_scene_depth_to_backbuffer()?;
        stats.scene_depth_copied = true;

        // ---------------------------------------------------------------------
        // Stage (c): 3-Split Cascaded Shadow Map (CSM) Pass
        // ---------------------------------------------------------------------
        if let Some(plan) = sun_plan {
            self.csm_atlas.update_plan(plan)?;
            if let Some(cmd) = renderer.current_cmd_list.clone() {
                self.csm_atlas.begin_atlas_pass(&cmd);
                let plan = self.csm_atlas.plan.as_ref().unwrap();
                let mut shadow_draws = Vec::new();
                for (cascade_idx, slot) in plan.slots.iter().enumerate() {
                    shadow_draws.clear();
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
                            if dist_sq > (slot.split_distance + mesh.bsphere_radius + 32.0).powi(2)
                                || below_ssa_discard(mesh.bsphere_radius, dist_sq, ssa_discard)
                            {
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
                        shadow_draws.push((caster, u32::MAX));
                    }
                    for inst in dynamic_draws.iter().filter(|instance| instance.visibility.shadows[cascade_idx]) {
                        for sub in inst.submeshes.iter() {
                            shadow_draws.push((sub.mesh_handle, inst.transform_index));
                        }
                    }
                    renderer.draw_csm_shadow_batch(&self.csm_atlas, cascade_idx, &shadow_draws)?;
                    stats.csm_cascade_draws[cascade_idx] = shadow_draws.len() as u32;
                }
                self.csm_atlas.end_atlas_pass(&cmd);
                // Restore main player camera CBV before deferred combine & forward stages
                renderer.update_camera(&cam_constants)?;
            }
        } else {
            self.csm_atlas.plan = None;
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

        let mut light_params = GpuLightParams {
            sun_direction: [
                -sun_dir_to_surface[0],
                -sun_dir_to_surface[1],
                -sun_dir_to_surface[2],
                0.0,
            ],
            sun_color: [sun_rgb[0], sun_rgb[1], sun_rgb[2], 1.0],
            ambient_color: [ambient_rgb[0], ambient_rgb[1], ambient_rgb[2], 1.0],
            fog_color: env.map_or([0.0; 4], |weather| weather.fog_color_rgba()),
            fog_range: env.map_or([0.0; 4], |weather| [weather.fog_near, weather.fog_far, 0.0, 0.0]),
            hud_projection: hud_draws.as_ref().map_or([0.0; 4], |batch| {
                let p = batch.camera.proj;
                [p[0][0], p[1][1], p[2][2], p[3][2]]
            }),
            ..Default::default()
        };
        if stats.hud_weapon_draws > 0 && let (Some(world), Some(weather)) = (world, env) {
            self.set_hud_environment(renderer, world, weather, camera, vfs, dt, &mut light_params);
        }

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
        let native_tonemap = self.native_tonemap.filter(|_| self.config.tonemap_mode == 0);
        if let Some(tonemap) = native_tonemap {
            let settings = renderer.postprocess_manager_mut();
            settings.tonemap = tonemap;
            settings.middle_gray = self.native_middle_gray;
        } else if let Some(e) = env {
            let pp = renderer.postprocess_manager_mut();
            pp.tonemap.exposure = e.bloom_exposure.max(0.5);
        }
        let post_push = PostprocessPushConstants {
            screen_res_inv: [1.0 / width as f32, 1.0 / height as f32],
            time_seconds: elapsed_sec,
            bloom_factor: self.config.bloom_factor,
            scene_texture_index: hdr_srv.0,
            bloom_texture_index: u32::MAX,
            tonemap_mode: native_tonemap.map_or(self.config.tonemap_mode, |_| MONOLITH_SDR_TONEMAP),
            color_grading_enabled: 0,
        };
        renderer.draw_postprocess(&post_push)?;
        stats.postprocess_executed = true;

        // ---------------------------------------------------------------------
        // Stage (e): Forward Alpha & Atmospheric Effects (scene depth occlusion)
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
            is_cubemap: u32::from(renderer.get_texture_by_index(DescriptorIndex(sky_tex0)).is_some_and(|t| t.is_cubemap))
                | (u32::from(renderer.get_texture_by_index(DescriptorIndex(sky_tex1)).is_some_and(|t| t.is_cubemap)) << 1),
            sky_color,
            sun_color: sun_glow,
            sun_dir: [
                -sun_dir_to_surface[0],
                -sun_dir_to_surface[1],
                -sun_dir_to_surface[2],
                0.0,
            ],
            sky_rotation: env.map_or(0.0, |e| e.sky_rotation),
            ..Default::default()
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

        // e.7 Forward HUD passes for scope glass and additive reticles.
        if let Some(batch) = &hud_draws {
            stats.hud_weapon_draws += self.draw_hud_pass(renderer, batch, Some(stats.combine_light_count))?;
        }

        // ---------------------------------------------------------------------
        // Stage (g): UI Canvas Overlay (Crosshair, Stalker HUD, Inventory & PDA)
        // ---------------------------------------------------------------------
        if self.config.render_hud {
            if !self.ui_font_attempted
                && let Some(vfs_ref) = vfs
            {
                self.ui_font_attempted = true;
                if let Ok(fonts) = load_ltx_tree_from_vfs(vfs_ref, "$game_configs$/fonts.ltx", &[]) {
                    self.ui_font = ["ui_font_letterica16_russian", "ui_font_arial_14", "ui_font_letterica18_russian", "hud_font_medium"]
                        .into_iter().find_map(|section| load_ui_font(renderer, vfs_ref, &fonts, section, height));
                    self.inventory_fonts = [
                        ("letterica18", "ui_font_letterica18_russian"),
                        ("graffiti22", "ui_font_graffiti22_russian"),
                    ].into_iter().filter_map(|(name, section)| {
                        Some((name, load_ui_font(renderer, vfs_ref, &fonts, section, height)?))
                    }).collect();
                }
                self.ui_equipment_tex = try_load_vfs_texture(renderer, Some(vfs_ref), "textures\\ui\\ui_icon_equipment.dds");
                self.ui_grid_tex = try_load_vfs_texture(renderer, Some(vfs_ref), "textures\\ui\\ui_grid.dds");
                self.inventory_atlas = load_ui_texture_atlases_from_vfs(vfs_ref).ok().or_else(|| {
                    let doc = load_xml_from_vfs(vfs_ref, "configs/ui/textures_descr/ui_actor_menu.xml").ok()?;
                    let mut map = UiTextureAtlasMap::default();
                    map.merge_doc(&doc);
                    Some(map)
                });
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
                    &self.inventory_fonts,
                    self.ui_equipment_tex,
                    self.ui_grid_tex,
                    self.inventory_atlas.as_ref(),
                    vfs,
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

    fn ssa_discard(&self, renderer: &Dx12Renderer, camera: &FirstPersonCamera) -> f32 {
        compute_r_ssa_discard(
            renderer.width,
            renderer.height,
            camera.fov_deg,
            self.geometry_lod,
            DETAIL_DEFAULT_PS_SSA_DISCARD,
        )
    }

    fn draw_level_details(
        &mut self,
        renderer: &mut Dx12Renderer,
        world: &LoadedWorld,
        camera: &FirstPersonCamera,
        frustum: &ClipFrustum,
        view_proj: &XRayMatrix,
        ssa_discard: f32,
        wind_dir: f32,
        wind_vel: f32,
        elapsed_sec: f32,
    ) -> WinResult<u32> {
        let (Some(state), Some(details), Some(cform)) = (&mut self.level_details, &world.details, &world.cform) else {
            return Ok(0);
        };
        let position = camera.position;
        let (sx, sz) = LevelDetails::world_to_cache_center(position[0], position[2]);
        // Monolith fills the initial grid, then decompresses up to seven new cells per update.
        state.cache.update_cache_with_cform(
            details, sx, sz, position, 7, &state.config, &cform.build_bvh(), cform, world.gamemtl.as_ref(),
        );
        let visible = state.cache.collect_visible(
            position, frustum, None, view_proj, renderer.frame_index as u32,
            &DetailVisibilityConfig {
                ssa_discard,
                ..Default::default()
            },
        );
        let mut instances = Vec::new();
        let mut batches = Vec::new();
        for (groups, wind_strength) in [(&visible.still, 0.0), (&visible.wave1, 0.15), (&visible.wave2, 0.15)] {
            for (group, prototype) in groups.iter().zip(&state.prototypes) {
                let Some(prototype) = prototype.as_ref().filter(|_| !group.is_empty()) else {
                    continue;
                };
                let start = instances.len();
                instances.extend(group.iter().map(|item| GpuDetailInstance {
                    position: item.position,
                    scale: item.scale_calculated,
                    rotation_y: item.xform[2][0].atan2(item.xform[0][0]),
                    alpha: item.alpha_target,
                    hemi_light: item.c_hemi,
                    sun_light: item.c_sun,
                    color_tint: [1.0; 4],
                }));
                batches.push(GpuDetailBatch {
                    mesh: &prototype.mesh,
                    instances: start..instances.len(),
                    push: DetailPushConstants {
                        wind_direction: [wind_dir.cos(), wind_dir.sin()],
                        wind_speed: (wind_vel * 0.01).clamp(0.5, 4.0),
                        wind_strength,
                        time_seconds: elapsed_sec,
                        texture_index: prototype.texture.0,
                        alpha_threshold: 200.0 / 255.0,
                        custom_param: 0.0,
                    },
                });
            }
        }
        renderer.draw_gbuffer_details(&instances, &batches)?;
        Ok(instances.len() as u32)
    }
}

fn load_detail_config(vfs: Option<&VirtualFileSystem>) -> (u32, DetailDecompressConfig) {
    let mut radius = DETAIL_DEFAULT_RADIUS;
    let mut config = DetailDecompressConfig::default();
    if let Some(commands) = vfs.and_then(|vfs| vfs.read_to_string_lossy("user.ltx").ok()) {
        for line in commands.lines() {
            let mut words = line.split_whitespace();
            let (Some(command), Some(value)) = (words.next(), words.next()) else {
                continue;
            };
            match command {
                "r__detail_radius" => {
                    if let Ok(value) = value.parse::<u32>() { radius = value.min(250); }
                }
                "r__detail_density" => {
                    if let Ok(value) = value.parse::<f32>() { config.density = value.clamp(0.04, 1.0); }
                }
                "r__detail_height" => {
                    if let Ok(value) = value.parse::<f32>() { config.detail_height = value.clamp(0.5, 2.0); }
                }
                "ssfx_terrain_grass_slope" => {
                    if let Ok(value) = value.parse::<f32>() { config.terrain_grass_slope = value.clamp(0.0, 1.0); }
                }
                _ => {}
            }
        }
    }
    (radius, config)
}

fn model_search_paths(key: &str, level_dir: Option<&str>) -> impl Iterator<Item = String> {
    let file = if key.rsplit('\\').next().is_some_and(|name| name.contains('.')) {
        key.to_owned()
    } else {
        format!("{key}.ogf")
    };
    // CModelPool tries the explicit path, the level and then $game_meshes$.
    [
        Some(key.to_owned()),
        (key != file).then(|| file.clone()),
        level_dir.and_then(|dir| normalize_virtual_path(&format!("{dir}\\{file}")).ok()),
        (!file.starts_with("meshes\\")).then(|| format!("meshes\\{file}")),
    ].into_iter().flatten()
}

fn upload_resolved_mesh(
    renderer: &mut Dx12Renderer,
    static_verts: &[StaticVertex],
    indices: &[u16],
) -> WinResult<Option<MeshHandle>> {
    if static_verts.is_empty() || indices.is_empty() {
        return Ok(None);
    }
    let gpu_verts: Vec<XRayVertex> = static_verts.iter().map(world_vertex).collect();
    let handle = renderer.create_mesh(&gpu_verts, indices)?;
    Ok(Some(handle))
}

fn world_vertex(v: &StaticVertex) -> XRayVertex {
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
}

fn load_hud_materials(vfs: &VirtualFileSystem) -> BTreeMap<String, HudMaterial> {
    let mut materials: BTreeMap<_, _> = vfs.read_bytes("shaders.xr").ok()
        .and_then(|bytes| EngineShaderLibrary::parse(&bytes).ok())
        .into_iter()
        .flat_map(|library| library.blenders)
        .filter_map(|shader| {
            let material = HudMaterial::from_model(R4BlenderSpec::from_blender_shader(&shader)?)?;
            Some((shader.desc.name, material))
        })
        .collect();
    // The installed r3 Lua pass uses srcalpha/one, alpha ref 0 and depth test without writes.
    materials.insert("models\\lfo_light_dot_weapons_old".to_owned(), HudMaterial {
        blend: HudBlend::Additive,
        alpha_ref: Some(0),
        strict_sorting: true,
    });
    materials
}

fn try_load_vfs_texture(
    renderer: &mut Dx12Renderer,
    vfs: Option<&VirtualFileSystem>,
    vpath: &str,
) -> Option<DescriptorIndex> {
    match load_vfs_texture(renderer, vfs, vpath) {
        Ok(index) => Some(index),
        Err(error) => {
            // Bound diagnostics and report each failed path once, rather than every frame.
            static REPORTED: std::sync::Mutex<std::collections::BTreeSet<String>> =
                std::sync::Mutex::new(std::collections::BTreeSet::new());
            if let Ok(mut reported) = REPORTED.lock() {
                if reported.len() < 32 && reported.insert(vpath.to_owned()) {
                    eprintln!("[texture] {vpath}: {error}");
                }
            }
            None
        }
    }
}

/// Load through the same path as material/sky binding, preserving the failure stage.
/// Also accepts X-Ray aliases and bare texture stems for diagnostic callers.
pub fn load_vfs_texture(
    renderer: &mut Dx12Renderer,
    vfs: Option<&VirtualFileSystem>,
    vpath: &str,
) -> Result<DescriptorIndex, String> {
    let normalized = xray_vfs::normalize_virtual_path(vpath)
        .map_err(|e| format!("path normalization: {e:?}"))?;
    // Level-local lightmaps/terrain paths must retain their levels/ prefix.
    let key = if normalized.starts_with("levels\\") {
        normalized
    } else {
        texture_stem_to_dds_vpath(&normalized)
    };
    if let Some(existing) = renderer.get_texture(&key) {
        return Ok(existing.descriptor_index);
    }
    let vfs = vfs.ok_or_else(|| "VFS not attached".to_owned())?;
    let source = vfs.resolve(&key).map_err(|e| format!("VFS lookup {key}: {e:?}"))?;
    let bytes = vfs.read_bytes(&key).map_err(|e| format!("VFS read {source:?}: {e:?}"))?;
    let dds = xray_render_dx12::texture::parse_dds(&bytes)
        .map_err(|e| format!("DDS decode {key} from {source:?}: {e:?}"))?;
    renderer.load_parsed_dds_texture(&key, &dds)
        .map_err(|e| format!("GPU upload {key} from {source:?}: {e:?}"))
}

fn load_ui_font(
    renderer: &mut Dx12Renderer,
    vfs: &VirtualFileSystem,
    fonts: &xray_core::ltx_loader::OwnedLtxDocument,
    section: &str,
    screen_height: u32,
) -> Option<UiFontResource> {
    let (cfg, atlas) = load_font_from_vfs(vfs, fonts, section, screen_height).ok()?;
    let texture_index = try_load_vfs_texture(renderer, Some(vfs), &cfg.texture_dds_vpath)?;
    let texture = renderer.get_texture_by_index(texture_index)?;
    Some(UiFontResource { atlas, texture_index, texture_size: [texture.width, texture.height] })
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

fn below_ssa_discard(radius: f32, distance_sq: f32, threshold: f32) -> bool {
    // R4's CalcSSA uses the sphere radius divided by squared camera distance.
    radius <= threshold * (distance_sq + 1e-5)
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
        // R4 phase_combine doubles the environment ambient and uses a 0.001 floor.
        // Keep the current hemispherical estimate until the environment probes are sampled here.
        let amb = std::array::from_fn(|axis| {
            (2.0 * e.ambient_color[axis]).max(0.001) + 0.5 * e.hemi_color[axis]
        });
        (dir, e.sun_color, amb)
    } else {
        ([0.45, -0.78, 0.43], [1.0, 0.94, 0.85], [0.28, 0.30, 0.34])
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
