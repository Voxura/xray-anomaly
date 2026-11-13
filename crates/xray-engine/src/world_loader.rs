//! Level World Loader & Atmospheric Weather Stepper (`xray-engine`).
//!
//! Streams a complete X-Ray level bundle from [`VirtualFileSystem`] archives and loose overrides:
//! - Core scene (`level`), geometry containers (`level.geom` / optional `level.geomx`)
//! - Resolved visual hierarchy & static meshes with `.thm`/`.dds` texture links
//! - On-demand dynamic `.ogf` model cache (characters, weapons, props)
//! - Pure-Rust `level.cform` BVH collision & ground snapping
//! - Optional `level.hom` occlusion rasterizer + portal/sector traversal
//! - Optional `level.details`, `level.env_mod`, `level.fog_vol`, `level.ps_static`, `level.ai`, `level.gct`, `level.spawn`, `level.game`
//! - Live sky/weather cycle evaluation (`WeatherLibrary`, `WetnessState`, `WindAnimState`, `RainEffectState`, `ThunderboltEffectState`)

use std::collections::BTreeMap;
use std::fmt;

use xray_formats::ai_graph::{LevelCrossTable, LevelGraph};
use xray_formats::cform::{CformModel, RaycastHit};
use xray_formats::chunk::FormatError;
use xray_formats::details::LevelDetails;
use xray_formats::gamemtl::GameMaterialLibrary;
use xray_formats::hom::{frustum_planes, ClipFrustum, OccRasterizer, HomMesh};
use xray_formats::level::{
    LevelData, LevelGeometry, LevelShaderRef, PortalTraversalOptions, PortalTraversalResult,
    ResolvedLevelMesh,
};
use xray_formats::ogf::{matrix_mul_44, OgfBBox, OgfBSphere, OgfModel, OgfTextureLink, OgfVisualType, XRayMatrix};
use xray_formats::scop::ScopWeatherState;
use xray_formats::shaders_xr::{LevelEnvModifiers, LevelFogVolumes, LevelStaticParticles};
use xray_formats::sound_env::{LevelStaticSounds, SoundOcclusionMesh};
use xray_formats::spawns::{LevelGameData, LevelSpawnData};
use xray_formats::thm::ResolvedTextureLinks;
use xray_formats::vfs_bridge::{
    load_cform_from_vfs, load_details_from_vfs, load_env_mods_from_vfs, load_fog_volumes_from_vfs,
    load_gamemtl_from_vfs, load_hom_from_vfs, load_level_ai_from_vfs, load_level_from_vfs,
    load_level_game_from_vfs, load_level_gct_from_vfs, load_level_geometry_from_vfs,
    load_level_spawn_from_vfs, load_ogf_from_vfs, load_ps_static_from_vfs, load_som_from_vfs,
    load_static_sounds_from_vfs, load_weather_library_from_vfs, resolve_ogf_texture_links,
    AssetLoadError,
};
use xray_formats::weather::{
    normalize_time, EnvBoost, EnvGlobalParams, MixedEnvironment, RainEffectState, RainFrameOutput,
    ThunderboltEffectState, ThunderboltFrameOutput, WeatherLibrary, WetnessState, WindAnimState,
    DEFAULT_WETNESS_MULTIPLIERS,
};
use xray_vfs::{normalize_virtual_path, VfsError, VirtualFileSystem};

/// Error produced while streaming level geometry, visuals, or environment assets from VFS.
#[derive(Debug)]
pub enum WorldLoadError {
    Asset(AssetLoadError),
    MissingRequiredFile { level_dir: String, file: &'static str },
    InvalidMesh { visual_index: usize, reason: FormatError },
}

impl fmt::Display for WorldLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Asset(e) => write!(f, "asset load failed: {e}"),
            Self::MissingRequiredFile { level_dir, file } => {
                write!(f, "missing required level file '{file}' in '{level_dir}'")
            }
            Self::InvalidMesh { visual_index, reason } => {
                write!(f, "failed to resolve level visual {visual_index}: {reason:?}")
            }
        }
    }
}

impl std::error::Error for WorldLoadError {}

impl From<AssetLoadError> for WorldLoadError {
    fn from(err: AssetLoadError) -> Self {
        Self::Asset(err)
    }
}

impl From<FormatError> for WorldLoadError {
    fn from(err: FormatError) -> Self {
        Self::Asset(AssetLoadError::Format(err))
    }
}

impl From<VfsError> for WorldLoadError {
    fn from(err: VfsError) -> Self {
        Self::Asset(AssetLoadError::Vfs(err))
    }
}

/// Configuration controlling which level subsystems [`WorldLoader`] streams on load.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelLoadConfig {
    /// Virtual path of the level directory (e.g. `"levels/l01_escape"` or `"$game_levels$/zaton"`).
    pub level_dir: String,
    /// Decode static level visual vertices/indices using `level.geom` when available.
    pub resolve_visual_meshes: bool,
    /// Resolve `.thm`/`.dds` diffuse, bump, and detail texture links for visual shaders.
    pub resolve_textures: bool,
    /// Also attempt to load `level.geomx` fastpath Z-prepass/shadow containers if present.
    pub load_fastpath_geom: bool,
    /// Load optional auxiliary files (`level.hom`, `level.details`, `level.env_mod`,
    /// `level.fog_vol`, `level.ps_static`, `level.ai`, `level.gct`, `level.spawn`, `level.game`).
    pub load_optional_aux: bool,
    /// Load `$game_configs$/environment/` cycles into [`WorldWeatherState`].
    pub load_weather: bool,
}

impl Default for LevelLoadConfig {
    fn default() -> Self {
        Self {
            level_dir: "$game_levels$/l01_escape".to_string(),
            resolve_visual_meshes: true,
            resolve_textures: true,
            load_fastpath_geom: false,
            load_optional_aux: true,
            load_weather: true,
        }
    }
}

impl LevelLoadConfig {
    pub fn minimal(level_dir: impl Into<String>) -> Self {
        Self {
            level_dir: level_dir.into(),
            resolve_visual_meshes: true,
            resolve_textures: false,
            load_fastpath_geom: false,
            load_optional_aux: false,
            load_weather: false,
        }
    }
}

/// Resolved visual entry extracted from `level` + `level.geom` + `.thm`/`.dds`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedVisualMesh {
    pub visual_index: usize,
    pub visual_type: OgfVisualType,
    pub bbox: OgfBBox,
    pub bsphere: OgfBSphere,
    pub shader: Option<LevelShaderRef>,
    pub mesh: ResolvedLevelMesh,
    pub fastpath: Option<ResolvedLevelMesh>,
    pub texture_links: Vec<ResolvedTextureLinks>,
}

/// Dynamic `.ogf` asset and resolved texture chain streamed on demand for NPCs, props, or weapons.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedDynamicModel {
    pub vpath: String,
    pub model: OgfModel,
    pub texture_links: Vec<ResolvedTextureLinks>,
}

/// Live atmospheric & weather simulation state synced between `WeatherLibrary` and `ScopWeatherState`.
#[derive(Debug, Clone)]
pub struct WorldWeatherState {
    pub weather_lib: Option<WeatherLibrary>,
    pub env_mods: Option<LevelEnvModifiers>,
    pub active_cycle: String,
    pub next_cycle: String,
    pub weather_fx: String,
    pub cycle_time_sec: f32,
    /// Ratio of game-time speed to real-time seconds (matches X-Ray `alife.ltx` `time_factor`).
    pub time_factor: f32,
    pub wetness: WetnessState,
    pub wind: WindAnimState,
    pub rain: RainEffectState,
    pub thunderbolt: ThunderboltEffectState,
    pub global_params: EnvGlobalParams,
    pub boost: EnvBoost,
    pub current_env: Option<MixedEnvironment>,
    pub last_rain_frame: Option<RainFrameOutput>,
    pub last_thunderbolt: Option<ThunderboltFrameOutput>,
    pub surge_active: bool,
    pub psi_storm_active: bool,
}

impl Default for WorldWeatherState {
    fn default() -> Self {
        Self {
            weather_lib: None,
            env_mods: None,
            active_cycle: "default".to_string(),
            next_cycle: "default".to_string(),
            weather_fx: String::new(),
            cycle_time_sec: 9.0 * 3600.0,
            time_factor: 10.0,
            wetness: WetnessState::default(),
            wind: WindAnimState::default(),
            rain: RainEffectState::default(),
            thunderbolt: ThunderboltEffectState::default(),
            global_params: EnvGlobalParams::default(),
            boost: EnvBoost::default(),
            current_env: None,
            last_rain_frame: None,
            last_thunderbolt: None,
            surge_active: false,
            psi_storm_active: false,
        }
    }
}

impl WorldWeatherState {
    /// Apply persisted weather scalars and active cycle selection from a host `.scop` Chunk 3.
    pub fn apply_scop_weather(&mut self, scop: &ScopWeatherState) {
        if !scop.current_weather.is_empty() {
            self.active_cycle = scop.current_weather.clone();
        }
        self.next_cycle = if scop.next_weather.is_empty() {
            self.active_cycle.clone()
        } else {
            scop.next_weather.clone()
        };
        self.weather_fx = scop.weather_fx.clone();
        self.cycle_time_sec = normalize_time(scop.cycle_time_sec);
        self.wetness = scop.wetness_state();
        self.surge_active = scop.surge_active;
        self.psi_storm_active = scop.psi_storm_active;
    }

    /// Snapshot live atmospheric scalars back into a `.scop` Chunk 3 weather struct.
    pub fn to_scop_weather(&self) -> ScopWeatherState {
        let mut out = ScopWeatherState {
            current_weather: self.active_cycle.clone(),
            next_weather: self.next_cycle.clone(),
            weather_fx: self.weather_fx.clone(),
            weather_fx_time: 0.0,
            cycle_time_sec: self.cycle_time_sec,
            rain_wetness: self.wetness.factor,
            surge_active: self.surge_active,
            psi_storm_active: self.psi_storm_active,
            ..Default::default()
        };
        if let Some(mixed) = &self.current_env {
            out.apply_mixed_environment(mixed);
        }
        out.rain_wetness = self.wetness.factor;
        out
    }

    /// Resolve a cycle name in `weather_lib`, falling back gracefully to `"default"` or the first loaded cycle.
    pub fn resolved_cycle_name(&self) -> Option<&str> {
        let lib = self.weather_lib.as_ref()?;
        if lib.cycles.contains_key(&self.active_cycle) {
            return Some(self.active_cycle.as_str());
        }
        if lib.cycles.contains_key("default") {
            return Some("default");
        }
        lib.cycles.keys().next().map(String::as_str)
    }

    /// Advance sky/weather cycle blending, wind animation, surface wetness, and rain particles.
    pub fn step(
        &mut self,
        dt_sec: f32,
        camera_pos: [f32; 3],
        camera_dir: [f32; 3],
        vis_distance: f32,
        cform: Option<&CformModel>,
    ) -> Option<&MixedEnvironment> {
        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        self.cycle_time_sec = normalize_time(self.cycle_time_sec + dt * self.time_factor);

        let cycle_name = self.resolved_cycle_name()?.to_string();
        let lib = self.weather_lib.as_ref()?;
        let mixed = lib.sample_at(
            &cycle_name,
            self.cycle_time_sec,
            self.env_mods.as_ref(),
            camera_pos,
            vis_distance,
            &self.boost,
        )?;

        self.wind
            .step(dt, mixed.wind_velocity, mixed.wind_direction, 0.2);
        self.wetness.step(
            dt,
            mixed.rain_density,
            DEFAULT_WETNESS_MULTIPLIERS[0],
            DEFAULT_WETNESS_MULTIPLIERS[1],
        );

        let (rain_out, tb_out) = if let Some(model) = cform {
            let bvh = model.build_bvh();
            let rain_frame = self.rain.step_with_raycast(
                dt,
                &mixed,
                camera_pos,
                camera_dir,
                None,
                mixed.hemi_color[1].max(0.0),
                |orig, dir, max_dist| bvh.raycast_nearest(orig, dir, max_dist, true).map(|h| h.range),
            );
            let tb_frame = self.thunderbolt.step_with_raycast(
                dt,
                &mixed,
                &self.global_params,
                lib,
                None,
                camera_pos,
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                |orig, dir, max_dist| bvh.raycast_nearest(orig, dir, max_dist, true).map(|h| h.range),
            );
            (rain_frame, tb_frame)
        } else {
            let rain_frame = self.rain.step_no_collision(
                dt,
                &mixed,
                camera_pos,
                camera_dir,
                mixed.hemi_color[1].max(0.0),
            );
            let tb_frame = self.thunderbolt.step_with_raycast(
                dt,
                &mixed,
                &self.global_params,
                lib,
                None,
                camera_pos,
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                |_orig, _dir, _max_dist| None,
            );
            (rain_frame, tb_frame)
        };
        self.last_rain_frame = Some(rain_out);
        self.last_thunderbolt = Some(tb_out);
        self.current_env = Some(mixed);
        self.current_env.as_ref()
    }
}

/// Summary statistics for diagnostics, logging, and telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldLoadSummary {
    pub level_dir: String,
    pub shader_count: usize,
    pub raw_visual_count: usize,
    pub resolved_mesh_count: usize,
    pub sector_count: usize,
    pub portal_count: usize,
    pub light_count: usize,
    pub glow_count: usize,
    pub cform_vertex_count: usize,
    pub cform_triangle_count: usize,
    pub hom_triangle_count: usize,
    pub detail_slot_count: usize,
    pub ai_vertex_count: usize,
    pub spawn_entity_count: usize,
    pub weather_cycle_count: usize,
}

/// Complete streamed X-Ray level world bundle ready for rendering, collision queries, and ALife spawning.
#[derive(Debug, Clone)]
pub struct LoadedWorld {
    pub level_dir: String,
    pub level_data: LevelData,
    pub level_geom: Option<LevelGeometry>,
    pub level_geomx: Option<LevelGeometry>,
    pub cform: Option<CformModel>,
    pub hom: Option<HomMesh>,
    pub details: Option<LevelDetails>,
    pub fog_volumes: Option<LevelFogVolumes>,
    pub static_particles: Option<LevelStaticParticles>,
    pub level_ai: Option<LevelGraph>,
    pub level_gct: Option<LevelCrossTable>,
    pub level_spawn: Option<LevelSpawnData>,
    pub level_game: Option<LevelGameData>,
    pub gamemtl: Option<GameMaterialLibrary>,
    pub sound_occlusion: Option<SoundOcclusionMesh>,
    pub static_sounds: Option<LevelStaticSounds>,
    pub visual_meshes: Vec<LoadedVisualMesh>,
    pub dynamic_models: BTreeMap<String, LoadedDynamicModel>,
    pub weather: WorldWeatherState,
}

impl LoadedWorld {
    /// Stream a single `.ogf` dynamic model (and optional `.thm`/`.dds` links) from VFS with caching.
    pub fn stream_ogf_model(
        &mut self,
        vfs: &VirtualFileSystem,
        ogf_vpath: &str,
        resolve_textures: bool,
    ) -> Result<&LoadedDynamicModel, WorldLoadError> {
        let canonical_key = normalize_virtual_path(ogf_vpath)?;
        if !self.dynamic_models.contains_key(&canonical_key) {
            let model = load_ogf_from_vfs(vfs, &canonical_key)?;
            let mut texture_links = Vec::new();
            if resolve_textures
                && let Some(link) = extract_ogf_texture_link(&model)
            {
                texture_links.push(resolve_ogf_texture_links(vfs, &link)?);
            }
            self.dynamic_models.insert(
                canonical_key.clone(),
                LoadedDynamicModel {
                    vpath: canonical_key.clone(),
                    model,
                    texture_links,
                },
            );
        }
        Ok(self
            .dynamic_models
            .get(&canonical_key)
            .expect("inserted dynamic model must exist"))
    }

    /// Perform an AABB BVH raycast against static `level.cform` world collision geometry.
    pub fn raycast_cform(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        max_dist: f32,
        cull_backfaces: bool,
    ) -> Option<RaycastHit> {
        let model = self.cform.as_ref()?;
        let bvh = model.build_bvh();
        bvh.raycast_nearest(origin, dir, max_dist, cull_backfaces)
    }

    /// Cast a downward ray from `[x, y + 0.5, z]` to find walkable ground Y height in `level.cform`.
    pub fn ground_height_at(&self, pos: [f32; 3], max_drop: f32) -> Option<f32> {
        let origin = [pos[0], pos[1] + 0.5, pos[2]];
        let hit = self.raycast_cform(origin, [0.0, -1.0, 0.0], max_drop + 0.5, true)?;
        Some(hit.point[1])
    }

    /// Run portal & optional HOM occlusion visibility traversal from `camera_pos`.
    pub fn query_visible_sectors(
        &self,
        camera_pos: [f32; 3],
        view_matrix: &XRayMatrix,
        proj_matrix: &XRayMatrix,
        options: &PortalTraversalOptions,
    ) -> PortalTraversalResult {
        let full_transform = matrix_mul_44(proj_matrix, view_matrix);
        let frustum = ClipFrustum::from_view_proj(&full_transform, frustum_planes::ALL);
        let rasterizer = self.hom.as_ref().map(|hom_mesh| {
            let mut r = OccRasterizer::new();
            r.render_mesh(hom_mesh, &full_transform, camera_pos);
            r.propagate(&hom_mesh.triangles);
            r
        });
        let bvh = self.cform.as_ref().map(CformModel::build_bvh);
        let start_sector = self
            .level_data
            .detect_sector(camera_pos, bvh.as_ref())
            .unwrap_or(0);
        self.level_data.traverse_portals(
            start_sector,
            &frustum,
            camera_pos,
            &full_transform,
            rasterizer.as_ref(),
            options,
        )
    }

    /// Step the level's sky/weather simulation by `dt_sec`.
    pub fn step_weather(
        &mut self,
        dt_sec: f32,
        camera_pos: [f32; 3],
        camera_dir: [f32; 3],
        vis_distance: f32,
    ) -> Option<&MixedEnvironment> {
        let cform_ref = self.cform.as_ref();
        self.weather
            .step(dt_sec, camera_pos, camera_dir, vis_distance, cform_ref)
    }

    /// Build a compact diagnostic summary of streamed level contents.
    pub fn summary(&self) -> WorldLoadSummary {
        WorldLoadSummary {
            level_dir: self.level_dir.clone(),
            shader_count: self.level_data.shaders.len(),
            raw_visual_count: self.level_data.visuals.len(),
            resolved_mesh_count: self.visual_meshes.len(),
            sector_count: self.level_data.sectors.len(),
            portal_count: self.level_data.portals.len(),
            light_count: self.level_data.lights.len(),
            glow_count: self.level_data.glows.len(),
            cform_vertex_count: self.cform.as_ref().map_or(0, |c| c.vertices.len()),
            cform_triangle_count: self.cform.as_ref().map_or(0, |c| c.triangles.len()),
            hom_triangle_count: self.hom.as_ref().map_or(0, |h| h.triangles.len()),
            detail_slot_count: self.details.as_ref().map_or(0, |d| d.slots.len()),
            ai_vertex_count: self.level_ai.as_ref().map_or(0, |a| a.vertices.len()),
            spawn_entity_count: self.level_spawn.as_ref().map_or(0, |s| s.spawns.len()),
            weather_cycle_count: self
                .weather
                .weather_lib
                .as_ref()
                .map_or(0, |w| w.cycles.len()),
        }
    }
}

/// Loader coordinator streaming level scenes, collision, visuals, and weather from `VirtualFileSystem`.
pub struct WorldLoader;

impl WorldLoader {
    pub fn load_from_vfs(
        vfs: &VirtualFileSystem,
        config: &LevelLoadConfig,
    ) -> Result<LoadedWorld, WorldLoadError> {
        let level_data = match load_level_from_vfs(vfs, &config.level_dir) {
            Ok(data) => data,
            Err(AssetLoadError::Vfs(VfsError::FileNotFound)) => {
                return Err(WorldLoadError::MissingRequiredFile {
                    level_dir: config.level_dir.clone(),
                    file: "level",
                });
            }
            Err(e) => return Err(WorldLoadError::Asset(e)),
        };

        let level_geom = load_optional(|| load_level_geometry_from_vfs(vfs, &config.level_dir))?;
        let level_geomx = if config.load_fastpath_geom {
            let geomx_path = format!(
                "{}/level.geomx",
                config.level_dir.trim_end_matches(['/', '\\'])
            );
            load_optional(|| load_level_geometry_from_vfs(vfs, &geomx_path))?
        } else {
            None
        };
        let mut cform = load_optional(|| load_cform_from_vfs(vfs, &config.level_dir))?;
        if let Some(c) = &mut cform {
            c.build_spatial_index();
        }

        let level_leaf = config
            .level_dir
            .trim_end_matches(['/', '\\'])
            .rsplit(['/', '\\'])
            .next()
            .filter(|s| !s.is_empty() && !s.starts_with('$'));

        let (
            hom,
            details,
            env_mods,
            fog_volumes,
            static_particles,
            level_ai,
            level_gct,
            level_spawn,
            level_game,
            gamemtl,
            sound_occlusion,
            static_sounds,
        ) = if config.load_optional_aux {
            (
                load_optional(|| load_hom_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_details_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_env_mods_from_vfs(vfs, &config.level_dir, level_leaf))?,
                load_optional(|| load_fog_volumes_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_ps_static_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_level_ai_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_level_gct_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_level_spawn_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_level_game_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_gamemtl_from_vfs(vfs))?,
                load_optional(|| load_som_from_vfs(vfs, &config.level_dir))?,
                load_optional(|| load_static_sounds_from_vfs(vfs, &config.level_dir))?,
            )
        } else {
            (None, None, None, None, None, None, None, None, None, None, None, None)
        };

        let visual_meshes = if config.resolve_visual_meshes {
            resolve_level_visuals(
                vfs,
                &config.level_dir,
                &level_data,
                level_geom.as_ref(),
                level_geomx.as_ref(),
                config.resolve_textures,
            )?
        } else {
            Vec::new()
        };

        let weather_lib = if config.load_weather {
            load_optional(|| load_weather_library_from_vfs(vfs, level_leaf))?
        } else {
            None
        };

        let weather = WorldWeatherState {
            weather_lib,
            env_mods,
            ..Default::default()
        };

        Ok(LoadedWorld {
            level_dir: config.level_dir.clone(),
            level_data,
            level_geom,
            level_geomx,
            cform,
            hom,
            details,
            fog_volumes,
            static_particles,
            level_ai,
            level_gct,
            level_spawn,
            level_game,
            gamemtl,
            sound_occlusion,
            static_sounds,
            visual_meshes,
            dynamic_models: BTreeMap::new(),
            weather,
        })
    }
}

fn load_optional<T>(
    f: impl FnOnce() -> Result<T, AssetLoadError>,
) -> Result<Option<T>, WorldLoadError> {
    match f() {
        Ok(val) => Ok(Some(val)),
        Err(AssetLoadError::Vfs(VfsError::FileNotFound)) => Ok(None),
        Err(e) => Err(WorldLoadError::Asset(e)),
    }
}

fn resolve_level_visuals(
    vfs: &VirtualFileSystem,
    level_dir: &str,
    level_data: &LevelData,
    level_geom: Option<&LevelGeometry>,
    level_geomx: Option<&LevelGeometry>,
    resolve_textures: bool,
) -> Result<Vec<LoadedVisualMesh>, WorldLoadError> {
    let empty_geom = LevelGeometry {
        vertex_buffers: Vec::new(),
        index_buffers: Vec::new(),
        swis: Vec::new(),
    };
    let geom = level_geom.unwrap_or(&empty_geom);

    let mut out = Vec::new();
    for (visual_index, visual) in level_data.visuals.iter().enumerate() {
        let has_geometry =
            visual.gcontainer.is_some() || visual.vcontainer.is_some() || visual.vertices.is_some();
        if !has_geometry {
            continue;
        }
        let mesh = level_data
            .resolve_visual_mesh(visual_index, geom)
            .map_err(|reason| WorldLoadError::InvalidMesh {
                visual_index,
                reason,
            })?;
        let fastpath = match level_geomx {
            Some(gx) => level_data
                .resolve_fastpath_mesh(visual_index, gx)
                .map_err(|reason| WorldLoadError::InvalidMesh {
                    visual_index,
                    reason,
                })?,
            None => None,
        };

        let mut texture_links = Vec::new();
        if resolve_textures
            && let Some(shader_ref) = &mesh.shader
        {
            for tex_name in &shader_ref.textures {
                let link = OgfTextureLink {
                    shader: shader_ref.shader.clone(),
                    texture: tex_name.clone(),
                };
                let mut resolved = resolve_ogf_texture_links(vfs, &link)?;
                if !vfs.exists(&resolved.diffuse_dds)
                    && let Some(level_tex) =
                        resolve_level_texture_vpath(vfs, level_dir, &resolved.diffuse_name)
                {
                    resolved.diffuse_dds = level_tex;
                }
                texture_links.push(resolved);
            }
        }

        out.push(LoadedVisualMesh {
            visual_index,
            visual_type: visual.header.visual_type,
            bbox: visual.header.bbox,
            bsphere: visual.header.bsphere,
            shader: mesh.shader.clone(),
            mesh,
            fastpath,
            texture_links,
        });
    }

    Ok(out)
}

/// Resolve a level-specific texture path (such as per-level terrain or lightmaps)
/// when the global `textures/<stem>.dds` path does not exist in VFS.
pub fn resolve_level_texture_vpath(
    vfs: &VirtualFileSystem,
    level_dir: &str,
    tex_name: &str,
) -> Option<String> {
    let clean = tex_name.replace('/', "\\");
    let stem = clean.trim_matches('\\');

    // Candidate 1: Standard global textures directory `textures\<stem>.dds`
    let cand_textures = if stem.to_ascii_lowercase().starts_with("textures\\") {
        format!("{stem}.dds")
    } else {
        format!("textures\\{stem}.dds")
    };
    if vfs.exists(&cand_textures) {
        return Some(cand_textures);
    }

    // Candidate 2: Under level directory `<level_dir>\<stem>.dds` (e.g. `levels\l01_escape\terrain\terrain_escape.dds`)
    let clean_level = level_dir.trim_matches('\\');
    let cand_level_stem = format!("{clean_level}\\{stem}.dds");
    if vfs.exists(&cand_level_stem) {
        return Some(cand_level_stem);
    }

    // Candidate 3: Under level directory with stripped prefix (e.g. `terrain\terrain_escape` -> `levels\l01_escape\terrain_escape.dds`)
    if let Some((_, file_part)) = stem.rsplit_once('\\') {
        let cand_level_file = format!("{clean_level}\\{file_part}.dds");
        if vfs.exists(&cand_level_file) {
            return Some(cand_level_file);
        }
        let cand_level_terrain = format!("{clean_level}\\terrain\\{file_part}.dds");
        if vfs.exists(&cand_level_terrain) {
            return Some(cand_level_terrain);
        }
    }

    None
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
