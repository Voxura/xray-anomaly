//! X-Ray Monolith / S.T.A.L.K.E.R. Anomaly Weather, Environment & Atmospheric Mixers
//!
//! Mirrors `CEnvironment`, `CEnvDescriptor`, `CEnvDescriptorMixer`, `CEnvAmbient`,
//! `CLensFlareDescriptor`, and `SThunderboltDesc`/`SThunderboltCollection`
//! (`src/xrEngine/Environment.cpp`, `Environment_misc.cpp`, `thunderbolt.cpp`,
//! `xr_efflensflare.cpp`):
//! - Weather cycles (`configs\environment\weathers\*.ltx`) & weather effects (`weather_effects\*.ltx`)
//! - Global environment settings (`environment.ltx`) & hourly sun paths (`sun_positions.ltx` + dynamic solar model)
//! - Ambients (`ambients.ltx` + level-specific overrides), `sound_channels.ltx`, `effects.ltx`
//! - Lens flares / suns (`suns.ltx`) & thunderbolt lightning presets (`thunderbolts.ltx`, `thunderbolt_collections.ltx`)
//! - Circular 24-hour (`86400s`) descriptor selection & linear mixer blending with `LevelEnvModifiers` (`EnvModAccum`).

use std::collections::BTreeMap;

use xray_core::ltx_loader::OwnedLtxDocument;

use crate::chunk::FormatError;
use crate::ogf::{matrix_mul_43, matrix_rotate_y, XRayMatrix, XRAY_IDENTITY_MATRIX};
use crate::shaders_xr::{env_mod_flags, EnvModAccum, LevelEnvModifiers};
use crate::thm::texture_stem_to_dds_vpath;

pub const DAY_LENGTH: f32 = 86_400.0;
pub const WFX_TRANS_TIME: f32 = 5.0;

// ============================================================================
// Skybox Half-Box, Clouds Hemisphere, Wind & Rain Render Constants
// ============================================================================

/// Number of unique vertices in X-Ray's skybox half-box (`dxEnvironmentRender.cpp:15-36`).
pub const SKYBOX_HBOX_VERTEX_COUNT: usize = 12;
/// Number of triangles in X-Ray's skybox half-box (`dxEnvironmentRender.cpp:37-59`).
pub const SKYBOX_HBOX_TRIANGLE_COUNT: usize = 20;
/// Number of indices (`20 * 3`) in X-Ray's skybox half-box.
pub const SKYBOX_HBOX_INDEX_COUNT: usize = 60;

/// Single skybox half-box vertex (`v_skybox` position + 3D cubemap texcoord).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyboxVertex {
    pub pos: [f32; 3],
    pub texcoord: [f32; 3],
}

/// 12 skybox half-box vertex positions matching `hbox_verts[v * 2]` (`dxEnvironmentRender.cpp:15-36`).
pub const SKYBOX_HBOX_POSITIONS: [[f32; 3]; SKYBOX_HBOX_VERTEX_COUNT] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 0.0, -1.0],
    [1.0, 0.0, -1.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
];

/// 12 skybox half-box 3D cubemap lookup coordinates matching `hbox_verts[v * 2 + 1]`
/// (note bottom 4 vertices use `-1.01` Y for horizon seam compensation).
pub const SKYBOX_HBOX_TEXCOORDS: [[f32; 3]; SKYBOX_HBOX_VERTEX_COUNT] = [
    [-1.0, -1.01, -1.0],
    [1.0, -1.01, -1.0],
    [-1.0, -1.01, 1.0],
    [1.0, -1.01, 1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0],
    [-1.0, -1.0, 1.0],
];

/// 60 indices (`20 * 3`) matching `hbox_faces` (`dxEnvironmentRender.cpp:37-59`).
pub const SKYBOX_HBOX_INDICES: [u16; SKYBOX_HBOX_INDEX_COUNT] = [
    0, 2, 3, 3, 1, 0, 4, 5, 7, 7, 6, 4, 0, 1, 9, 9, 8, 0, 8, 9, 5, 5, 4, 8, 1, 3, 10, 10, 9, 1,
    9, 10, 7, 7, 5, 9, 3, 2, 11, 11, 10, 3, 10, 11, 6, 6, 7, 10, 2, 0, 8, 8, 11, 2, 11, 8, 4, 4,
    6, 11,
];

/// Construct the interleaved 12-vertex / 60-index skybox half-box mesh.
pub fn skybox_hbox_mesh() -> ([SkyboxVertex; SKYBOX_HBOX_VERTEX_COUNT], [u16; SKYBOX_HBOX_INDEX_COUNT])
{
    let mut verts = [SkyboxVertex {
        pos: [0.0; 3],
        texcoord: [0.0; 3],
    }; SKYBOX_HBOX_VERTEX_COUNT];
    for (i, slot) in verts.iter_mut().enumerate() {
        slot.pos = SKYBOX_HBOX_POSITIONS[i];
        slot.texcoord = SKYBOX_HBOX_TEXCOORDS[i];
    }
    (verts, SKYBOX_HBOX_INDICES)
}

/// Number of vertices in X-Ray's tessellated clouds dome (`HEMI2_VERTS` in `xrHemisphere.cpp`).
pub const CLOUDS_HEMI_VERTEX_COUNT: usize = 91;
/// Number of triangles in X-Ray's tessellated clouds dome (`HEMI2_FACES` in `xrHemisphere.cpp`).
pub const CLOUDS_HEMI_TRIANGLE_COUNT: usize = 160;
/// Number of indices (`160 * 3`) in `CLOUDS_HEMI_INDICES`.
pub const CLOUDS_HEMI_INDEX_COUNT: usize = 480;

/// 91 unit-hemisphere vertex positions for cloud dome rendering (`hemi_2` in `xrHemisphere.cpp:86-179`).
pub const CLOUDS_HEMI_POSITIONS: [[f32; 3]; CLOUDS_HEMI_VERTEX_COUNT] = [
    [0.273268, 0.961937, -0.000000],
    [0.084444, 0.961937, 0.259892],
    [0.000000, 1.000000, -0.000000],
    [0.525730, 0.850655, -0.000000],
    [0.361805, 0.894424, 0.262865],
    [0.162460, 0.850655, 0.500000],
    [-0.221077, 0.961937, 0.160622],
    [-0.138196, 0.894424, 0.425324],
    [-0.425324, 0.850655, 0.309015],
    [-0.221077, 0.961937, -0.160623],
    [-0.447212, 0.894424, -0.000001],
    [-0.425324, 0.850655, -0.309017],
    [0.084445, 0.961937, -0.259892],
    [-0.138196, 0.894424, -0.425327],
    [0.162460, 0.850655, -0.500000],
    [0.361805, 0.894424, -0.262865],
    [0.447212, 0.723610, 0.525730],
    [0.638195, 0.723610, 0.262865],
    [0.688194, 0.525730, 0.500000],
    [-0.361801, 0.723610, 0.587784],
    [-0.052786, 0.723610, 0.688187],
    [-0.262865, 0.525730, 0.809013],
    [-0.670822, 0.723610, -0.162460],
    [-0.670822, 0.723610, 0.162458],
    [-0.850647, 0.525730, -0.000001],
    [-0.052786, 0.723610, -0.688194],
    [-0.361801, 0.723610, -0.587788],
    [-0.262865, 0.525730, -0.809021],
    [0.638195, 0.723610, -0.262865],
    [0.447212, 0.723610, -0.525734],
    [0.688194, 0.525730, -0.500000],
    [0.738174, 0.674606, -0.000000],
    [0.822617, 0.505722, -0.259892],
    [0.894424, 0.447212, -0.000000],
    [0.822617, 0.505722, 0.259892],
    [0.228109, 0.674606, 0.702049],
    [0.501377, 0.505722, 0.702049],
    [0.276394, 0.447212, 0.850647],
    [0.007032, 0.505722, 0.862671],
    [-0.597195, 0.674606, 0.433887],
    [-0.512752, 0.505722, 0.693779],
    [-0.723602, 0.447212, 0.525730],
    [-0.818275, 0.505722, 0.273266],
    [-0.597195, 0.674606, -0.433887],
    [-0.818275, 0.505722, -0.273268],
    [-0.723602, 0.447212, -0.525734],
    [-0.512753, 0.505722, -0.693779],
    [0.228109, 0.674606, -0.702049],
    [0.007032, 0.505722, -0.862671],
    [0.276394, 0.447212, -0.850655],
    [0.501377, 0.505722, -0.702049],
    [0.861801, 0.276394, 0.425323],
    [0.959251, 0.232454, 0.160622],
    [0.951057, 0.000000, 0.309017],
    [0.449184, 0.232454, 0.862671],
    [0.670822, 0.276394, 0.688187],
    [0.587788, 0.000000, 0.809013],
    [-0.138196, 0.276394, 0.951057],
    [0.143665, 0.232454, 0.961937],
    [0.000000, 0.000000, 1.000000],
    [-0.681641, 0.232454, 0.693779],
    [-0.447212, 0.276394, 0.850647],
    [-0.587784, 0.000000, 0.809013],
    [-0.947212, 0.276394, 0.162458],
    [-0.870460, 0.232454, 0.433887],
    [-0.951057, 0.000000, 0.309015],
    [-0.870460, 0.232454, -0.433887],
    [-0.947212, 0.276394, -0.162460],
    [-0.951057, 0.000000, -0.309017],
    [-0.447212, 0.276394, -0.850655],
    [-0.681641, 0.232454, -0.693779],
    [-0.587784, 0.000000, -0.809021],
    [0.143665, 0.232454, -0.961937],
    [-0.138196, 0.276394, -0.951057],
    [0.000000, 0.000000, -1.000000],
    [0.670822, 0.276394, -0.688194],
    [0.449188, 0.232454, -0.862671],
    [0.587788, 0.000000, -0.809013],
    [0.959251, 0.232454, -0.160623],
    [0.861801, 0.276394, -0.425327],
    [0.951057, 0.000000, -0.309017],
    [1.000000, 0.000000, -0.000000],
    [0.809021, 0.000000, 0.587784],
    [0.309017, 0.000000, 0.951057],
    [-0.309015, 0.000000, 0.951057],
    [-0.809013, 0.000000, 0.587784],
    [-1.000000, 0.000000, -0.000001],
    [-0.809013, 0.000000, -0.587788],
    [-0.309015, 0.000000, -0.951057],
    [0.309017, 0.000000, -0.951057],
    [0.809021, 0.000000, -0.587788],
];

/// 480 triangle indices (`160 * 3`) for cloud dome rendering (`hemi_2v` in `xrHemisphere.cpp:180-342`).
pub const CLOUDS_HEMI_INDICES: [u16; CLOUDS_HEMI_INDEX_COUNT] = [
    0, 1, 2, 3, 4, 0, 4, 1, 0, 4, 5, 1, 1, 6, 2, 5, 7, 1, 7, 6, 1, 7, 8, 6, 6, 9, 2, 8, 10, 6,
    10, 9, 6, 10, 11, 9, 9, 12, 2, 11, 13, 9, 13, 12, 9, 13, 14, 12, 12, 0, 2, 14, 15, 12, 15, 0,
    12, 15, 3, 0, 16, 17, 18, 5, 4, 16, 4, 17, 16, 4, 3, 17, 19, 20, 21, 8, 7, 19, 7, 20, 19, 7,
    5, 20, 22, 23, 24, 11, 10, 22, 10, 23, 22, 10, 8, 23, 25, 26, 27, 14, 13, 25, 13, 26, 25, 13,
    11, 26, 28, 29, 30, 3, 15, 28, 15, 29, 28, 15, 14, 29, 28, 31, 3, 30, 32, 28, 32, 31, 28, 32,
    33, 31, 31, 17, 3, 33, 34, 31, 34, 17, 31, 34, 18, 17, 16, 35, 5, 18, 36, 16, 36, 35, 16, 36,
    37, 35, 35, 20, 5, 37, 38, 35, 38, 20, 35, 38, 21, 20, 19, 39, 8, 21, 40, 19, 40, 39, 19, 40,
    41, 39, 39, 23, 8, 41, 42, 39, 42, 23, 39, 42, 24, 23, 22, 43, 11, 24, 44, 22, 44, 43, 22, 44,
    45, 43, 43, 26, 11, 45, 46, 43, 46, 26, 43, 46, 27, 26, 25, 47, 14, 27, 48, 25, 48, 47, 25, 48,
    49, 47, 47, 29, 14, 49, 50, 47, 50, 29, 47, 50, 30, 29, 51, 52, 53, 18, 34, 51, 34, 52, 51, 34,
    33, 52, 54, 55, 56, 37, 36, 54, 36, 55, 54, 36, 18, 55, 57, 58, 59, 21, 38, 57, 38, 58, 57, 38,
    37, 58, 60, 61, 62, 41, 40, 60, 40, 61, 60, 40, 21, 61, 63, 64, 65, 24, 42, 63, 42, 64, 63, 42,
    41, 64, 66, 67, 68, 45, 44, 66, 44, 67, 66, 44, 24, 67, 69, 70, 71, 27, 46, 69, 46, 70, 69, 46,
    45, 70, 72, 73, 74, 49, 48, 72, 48, 73, 72, 48, 27, 73, 75, 76, 77, 30, 50, 75, 50, 76, 75, 50,
    49, 76, 78, 79, 80, 33, 32, 78, 32, 79, 78, 32, 30, 79, 78, 52, 33, 80, 81, 78, 81, 52, 78, 81,
    53, 52, 51, 55, 18, 53, 82, 51, 82, 55, 51, 82, 56, 55, 54, 58, 37, 56, 83, 54, 83, 58, 54, 83,
    59, 58, 57, 61, 21, 59, 84, 57, 84, 61, 57, 84, 62, 61, 60, 64, 41, 62, 85, 60, 85, 64, 60, 85,
    65, 64, 63, 67, 24, 65, 86, 63, 86, 67, 63, 86, 68, 67, 66, 70, 45, 68, 87, 66, 87, 70, 66, 87,
    71, 70, 69, 73, 27, 71, 88, 69, 88, 73, 69, 88, 74, 73, 72, 76, 49, 74, 89, 72, 89, 76, 72, 89,
    77, 76, 75, 79, 30, 77, 90, 75, 90, 79, 75, 90, 80, 79,
];

/// Rain emitter & streak geometry constants (`Rain.cpp:20-32`, `dxRainRender.cpp:7-18`).
pub const RAIN_MAX_DESIRED_ITEMS: u32 = 2500;
pub const RAIN_SOURCE_RADIUS: f32 = 15.0;
pub const RAIN_CPU_SOURCE_RADIUS: f32 = 12.5;
pub const RAIN_SOURCE_OFFSET: f32 = 20.0;
pub const RAIN_MAX_DISTANCE: f32 = RAIN_SOURCE_OFFSET * 1.5;
pub const RAIN_SINK_OFFSET: f32 = -(RAIN_MAX_DISTANCE - RAIN_SOURCE_OFFSET);
pub const RAIN_DROP_LENGTH: f32 = 5.0;
pub const RAIN_DROP_WIDTH: f32 = 0.30;
pub const RAIN_DROP_MAX_ANGLE_RAD: f32 = 35.0 * (std::f32::consts::PI / 180.0);
pub const RAIN_DROP_SPEED_MIN: f32 = 40.0;
pub const RAIN_DROP_SPEED_MAX: f32 = 80.0;

/// Accumulating wind animation state (`CEnvironment::wind_anim`, `Environment.cpp:617-628`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WindAnimState {
    pub anim: [f32; 4],
}

impl WindAnimState {
    /// Advance `wind_anim` by `dt_sec` (`Environment.cpp:617-628`):
    /// - `anim.xy`: integrated `(vel * cos, vel * sin)` along `PI_DIV_2 - wind_direction_rad`
    /// - `anim.z`: accumulated clamped gust speed
    /// - `anim.w`: elapsed time in seconds
    pub fn step(
        &mut self,
        dt_sec: f32,
        wind_velocity: f32,
        wind_direction_rad: f32,
        tree_wind_min: f32,
    ) {
        if dt_sec <= 0.0 || !dt_sec.is_finite() {
            return;
        }
        let vel = wind_velocity.max(tree_wind_min * 1000.0).max(200.0) * 0.001;
        let dir = -wind_direction_rad + std::f32::consts::FRAC_PI_2;
        let (ws, wc) = dir.sin_cos();
        self.anim[0] += vel * wc * dt_sec;
        self.anim[1] += vel * ws * dt_sec;
        self.anim[2] += (vel * 1.33).clamp(0.0, 1.0) * dt_sec;
        self.anim[3] += dt_sec;
    }
}

/// Global atmospheric parameters from `environment\environment.ltx` `[environment]`.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvGlobalParams {
    pub altitude_rad: f32,
    pub delta_longitude_rad: f32,
    pub min_dist_factor: f32,
    pub tilt_rad: f32,
    /// Probability of a follow-up lightning strike (`second_propability` INI key in X-Ray).
    pub second_probability: f32,
    pub sky_color: f32,
    pub sun_color: f32,
    pub fog_color: f32,
}

impl Default for EnvGlobalParams {
    fn default() -> Self {
        Self {
            altitude_rad: 20.0_f32.to_radians(),
            delta_longitude_rad: 30.0_f32.to_radians(),
            min_dist_factor: 0.9,
            tilt_rad: 15.0_f32.to_radians(),
            second_probability: 0.3,
            sky_color: 0.1,
            sun_color: 0.5,
            fog_color: 0.25,
        }
    }
}

impl EnvGlobalParams {
    pub fn from_ltx(doc: &OwnedLtxDocument) -> Self {
        let def = Self::default();
        if !doc.sections.contains_key("environment") {
            return def;
        }
        Self {
            altitude_rad: doc
                .read_f32("environment", "altitude", def.altitude_rad.to_degrees())
                .to_radians(),
            delta_longitude_rad: doc
                .read_f32(
                    "environment",
                    "delta_longitude",
                    def.delta_longitude_rad.to_degrees(),
                )
                .to_radians(),
            min_dist_factor: doc
                .read_f32("environment", "min_dist_factor", def.min_dist_factor)
                .min(0.95),
            tilt_rad: doc
                .read_f32("environment", "tilt", def.tilt_rad.to_degrees())
                .to_radians(),
            second_probability: doc
                .read_f32(
                    "environment",
                    "second_propability",
                    doc.read_f32("environment", "second_probability", def.second_probability),
                )
                .clamp(0.0, 1.0),
            sky_color: doc.read_f32("environment", "sky_color", def.sky_color),
            sun_color: doc.read_f32("environment", "sun_color", def.sun_color),
            fog_color: doc.read_f32("environment", "fog_color", def.fog_color),
        }
    }
}

/// 24 hourly `(altitude_deg, longitude_deg)` entries from `environment\sun_positions.ltx`.
#[derive(Debug, Clone, PartialEq)]
pub struct SunPositionsConfig {
    pub hourly_alt_long_deg: [[f32; 2]; 24],
}

impl Default for SunPositionsConfig {
    fn default() -> Self {
        Self {
            hourly_alt_long_deg: [[-35.0, 0.0]; 24],
        }
    }
}

impl SunPositionsConfig {
    pub fn from_ltx(doc: &OwnedLtxDocument) -> Result<Self, FormatError> {
        let mut hourly = [[0.0f32; 2]; 24];
        for (hour, slot) in hourly.iter_mut().enumerate() {
            let key = format!("{hour:02}:00:00");
            if !doc.sections.contains_key(&key) {
                return Err(FormatError::InvalidValue("missing hourly sun_positions section"));
            }
            let alt = doc.read_f32(&key, "sun_altitude", 0.0);
            let lon = doc.read_f32(&key, "sun_longitude", 0.0);
            if !alt.is_finite() || !lon.is_finite() {
                return Err(FormatError::InvalidValue("non-finite sun angle"));
            }
            *slot = [alt, lon];
        }
        Ok(Self {
            hourly_alt_long_deg: hourly,
        })
    }

    /// Evaluate normalized X-Ray sun vector (`setHP(deg2rad(alt), deg2rad(long))`) at `game_time_sec`.
    /// Matches `CEnvironment::calculate_config_sun_dir` (`Environment.cpp:702-736`).
    pub fn sun_dir_at_time(&self, game_time_sec: f32) -> [f32; 3] {
        let gt = normalize_time(game_time_sec);
        let current_time = gt / (DAY_LENGTH / 24.0);
        let weather_time = (current_time.floor() as usize).min(23);
        let s_weight = current_time - (weather_time as f32);

        let [s_alt, s_long] = self.hourly_alt_long_deg[weather_time];
        let (real_alt, real_long) = if s_weight > 0.0 {
            let next_hour = if weather_time == 23 { 0 } else { weather_time + 1 };
            let [s_alt2, s_long2] = self.hourly_alt_long_deg[next_hour];
            (
                s_alt + (s_alt2 - s_alt) * s_weight,
                s_long + (s_long2 - s_long) * s_weight,
            )
        } else {
            (s_alt, s_long)
        };

        dir_from_hp(real_alt.to_radians(), real_long.to_radians())
    }
}

/// Convert X-Ray yaw/heading `h` and pitch `p` (radians) into a unit direction (`_vector3::setHP`).
pub fn dir_from_hp(h: f32, p: f32) -> [f32; 3] {
    let ch = h.cos();
    let cp = p.cos();
    let sh = h.sin();
    let sp = p.sin();
    [-cp * sh, sp, cp * ch]
}

/// Dynamic sun model (`CEnvironment::calculate_dynamic_sun_dir`, `Environment.cpp:738-803`).
/// Returns `(sun_dir, sun_color_blend)`.
pub fn calculate_dynamic_sun_dir(game_time_sec: f32) -> ([f32; 3], f32) {
    let g = ((360.0 / 365.25) * (180.0 + game_time_sec / DAY_LENGTH)).to_radians();
    let d_deg = 0.396372 - 22.91327 * g.cos() + 4.02543 * g.sin()
        - 0.387205 * (2.0 * g).cos()
        + 0.051967 * (2.0 * g).sin()
        - 0.154527 * (3.0 * g).cos()
        + 0.084798 * (3.0 * g).sin();

    let tc = 0.004297 + 0.107029 * g.cos()
        - 1.837877 * g.sin()
        - 0.837378 * (2.0 * g).cos()
        - 2.340475 * (2.0 * g).sin();

    let longitude = -30.4_f32;
    let mut sha = (game_time_sec / (DAY_LENGTH / 24.0) - 12.0) * 15.0 + longitude + tc;
    if sha > 180.0 {
        sha -= 360.0;
    }
    if sha < -180.0 {
        sha += 360.0;
    }

    let lat_r = 50.27_f32.to_radians();
    let d_r = d_deg.to_radians();
    let cos_sza = (lat_r.sin() * d_r.sin() + lat_r.cos() * d_r.cos() * sha.to_radians().cos())
        .clamp(-1.0, 1.0);
    let sza = cos_sza.acos();
    let mut sea = std::f32::consts::FRAC_PI_2 - sza;

    let denom = sza.sin() * lat_r.cos();
    let cos_az = if denom.abs() > 1e-6 {
        ((d_r.sin() - lat_r.sin() * cos_sza) / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let mut az = cos_az.acos() + std::f32::consts::PI;

    let min_x = 1.0_f32.to_radians();
    let min_y = 3.0_f32.to_radians();
    if sea < min_x {
        sea = min_x;
    }
    let blend = ((sea - min_x) / (min_y - min_x)).clamp(0.0, 1.0);
    sea = -sea;

    if sha < 0.0 {
        az = std::f32::consts::TAU - az;
    }

    (dir_from_hp(az, sea), blend)
}

/// Normalize game time in seconds to `[0.0, DAY_LENGTH]`.
pub fn normalize_time(tm: f32) -> f32 {
    if tm < 0.0 {
        let wrapped = tm.rem_euclid(DAY_LENGTH);
        if wrapped == 0.0 {
            0.0
        } else {
            wrapped
        }
    } else if tm > DAY_LENGTH {
        tm % DAY_LENGTH
    } else {
        tm
    }
}

/// Forward distance along the 24h cycle from `prev` to `cur` (`CEnvironment::TimeDiff`).
pub fn time_diff(prev: f32, cur: f32) -> f32 {
    if prev > cur {
        (DAY_LENGTH - prev) + cur
    } else {
        cur - prev
    }
}

/// Interpolation weight `[0..1]` for `val` inside `[min_t, max_t]` (handling midnight wraparound).
pub fn time_weight(val: f32, min_t: f32, max_t: f32) -> f32 {
    let length = time_diff(min_t, max_t);
    if length <= 1e-5 {
        return 0.0;
    }
    let mut weight = 0.0;
    if min_t > max_t {
        if val >= min_t || val <= max_t {
            weight = time_diff(min_t, val) / length;
        }
    } else if val >= min_t && val <= max_t {
        weight = time_diff(min_t, val) / length;
    }
    weight.clamp(0.0, 1.0)
}

/// Parse `"HH:MM:SS"` keyframe section names into seconds (`0..=86400`).
pub fn parse_hhmmss_seconds(ident: &str) -> Result<f32, FormatError> {
    let trimmed = ident.trim();
    if trimmed == "24:00:00" {
        return Ok(DAY_LENGTH);
    }
    let mut parts = trimmed.split(':');
    let h = parts
        .next()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .ok_or(FormatError::InvalidValue("invalid weather time hour"))?;
    let m = parts
        .next()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .ok_or(FormatError::InvalidValue("invalid weather time minute"))?;
    let s = parts
        .next()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .ok_or(FormatError::InvalidValue("invalid weather time second"))?;
    if parts.next().is_some() || h >= 24 || m >= 60 || s >= 60 {
        return Err(FormatError::InvalidValue("weather time out of range"));
    }
    Ok((h as f32) * 3600.0 + (m as f32) * 60.0 + (s as f32))
}

fn parse_f32_2(raw: Option<&str>, default: [f32; 2]) -> [f32; 2] {
    let Some(s) = raw else {
        return default;
    };
    let mut it = s.split(',').filter_map(|p| p.trim().parse::<f32>().ok());
    [
        it.next().unwrap_or(default[0]),
        it.next().unwrap_or(default[1]),
    ]
}

fn parse_f32_3(raw: Option<&str>, default: [f32; 3]) -> [f32; 3] {
    let Some(s) = raw else {
        return default;
    };
    let mut it = s.split(',').filter_map(|p| p.trim().parse::<f32>().ok());
    [
        it.next().unwrap_or(default[0]),
        it.next().unwrap_or(default[1]),
        it.next().unwrap_or(default[2]),
    ]
}

fn parse_f32_4(raw: Option<&str>, default: [f32; 4]) -> [f32; 4] {
    let Some(s) = raw else {
        return default;
    };
    let mut it = s.split(',').filter_map(|p| p.trim().parse::<f32>().ok());
    [
        it.next().unwrap_or(default[0]),
        it.next().unwrap_or(default[1]),
        it.next().unwrap_or(default[2]),
        it.next().unwrap_or(default[3]),
    ]
}

fn parse_csv_strings(raw: Option<&str>) -> Vec<String> {
    raw.unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Ambient sound channel definition (`sound_channels.ltx`).
#[derive(Debug, Clone, PartialEq)]
pub struct EnvSoundChannelDef {
    pub name: String,
    pub min_distance: f32,
    pub max_distance: f32,
    pub period: [i32; 4],
    pub sounds: Vec<String>,
}

impl EnvSoundChannelDef {
    pub fn from_ltx(doc: &OwnedLtxDocument, section: &str) -> Result<Self, FormatError> {
        let min_distance = doc.read_f32(section, "min_distance", 1.0);
        let max_distance = doc.read_f32(section, "max_distance", min_distance + 1.0);
        let period = [
            doc.read_i32(section, "period0", 1000),
            doc.read_i32(section, "period1", 5000),
            doc.read_i32(section, "period2", 1000),
            doc.read_i32(section, "period3", 5000),
        ];
        if period[0] > period[1] || period[2] > period[3] || max_distance <= min_distance {
            return Err(FormatError::InvalidValue("invalid env sound channel bounds"));
        }
        let sounds = parse_csv_strings(doc.read_str(section, "sounds"));
        if sounds.is_empty() {
            return Err(FormatError::InvalidValue("empty env sound channel sounds list"));
        }
        Ok(Self {
            name: section.to_string(),
            min_distance,
            max_distance,
            period,
            sounds,
        })
    }
}

/// Ambient particle/wind-blast effect (`effects.ltx`).
#[derive(Debug, Clone, PartialEq)]
pub struct EnvAmbientEffectDef {
    pub name: String,
    pub life_time_ms: u32,
    pub particles: String,
    pub offset: [f32; 3],
    pub wind_gust_factor: f32,
    pub sound: Option<String>,
    pub wind_blast_strength: f32,
    pub wind_blast_direction: [f32; 3],
    pub wind_blast_in_time: f32,
    pub wind_blast_out_time: f32,
}

impl EnvAmbientEffectDef {
    pub fn from_ltx(doc: &OwnedLtxDocument, section: &str) -> Result<Self, FormatError> {
        let life_time_ms = (doc.read_f32(section, "life_time", 1.0) * 1000.0).floor().max(0.0) as u32;
        let particles = doc
            .read_str(section, "particles")
            .unwrap_or("")
            .trim()
            .to_string();
        if particles.is_empty() {
            return Err(FormatError::InvalidValue("missing env ambient effect particles"));
        }
        let offset = parse_f32_3(doc.read_str(section, "offset"), [0.0; 3]);
        let wind_gust_factor = doc.read_f32(section, "wind_gust_factor", 0.0);
        let sound = doc
            .read_str(section, "sound")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);

        let (wind_blast_strength, wind_blast_direction, wind_blast_in_time, wind_blast_out_time) =
            if doc.read_str(section, "wind_blast_strength").is_some() {
                let strength = doc.read_f32(section, "wind_blast_strength", 0.0);
                let lon = doc.read_f32(section, "wind_blast_longitude", 0.0).to_radians();
                let dir = dir_from_hp(lon, 0.0);
                let in_t = doc.read_f32(section, "wind_blast_in_time", 0.0);
                let out_t = doc.read_f32(section, "wind_blast_out_time", 0.0);
                (strength, dir, in_t, out_t)
            } else {
                (0.0, [0.0, 0.0, 1.0], 0.0, 0.0)
            };

        Ok(Self {
            name: section.to_string(),
            life_time_ms,
            particles,
            offset,
            wind_gust_factor,
            sound,
            wind_blast_strength,
            wind_blast_direction,
            wind_blast_in_time,
            wind_blast_out_time,
        })
    }
}

/// Environment ambient preset (`ambients.ltx` & `ambients\<level>.ltx`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvAmbientDef {
    pub name: String,
    pub sound_channels: Vec<String>,
    pub min_effect_period_ms: u32,
    pub max_effect_period_ms: u32,
    pub effects: Vec<String>,
}

impl EnvAmbientDef {
    pub fn from_ltx(doc: &OwnedLtxDocument, section: &str) -> Result<Self, FormatError> {
        let sound_channels = parse_csv_strings(doc.read_str(section, "sound_channels"));
        let min_effect_period_ms =
            (doc.read_f32(section, "min_effect_period", 1.0) * 1000.0).floor().max(0.0) as u32;
        let max_effect_period_ms =
            (doc.read_f32(section, "max_effect_period", 1.0) * 1000.0).floor().max(0.0) as u32;
        let effects = parse_csv_strings(doc.read_str(section, "effects"));
        if sound_channels.is_empty() && effects.is_empty() {
            return Err(FormatError::InvalidValue("empty env ambient channels and effects"));
        }
        Ok(Self {
            name: section.to_string(),
            sound_channels,
            min_effect_period_ms,
            max_effect_period_ms,
            effects,
        })
    }
}

/// Individual sun/flare billboard element (`CLensFlareDescriptor::SFlare`).
#[derive(Debug, Clone, PartialEq)]
pub struct LensFlareElement {
    pub radius: f32,
    pub opacity: f32,
    pub position: f32,
    pub texture: String,
    pub shader: String,
}

/// Lens flare / sun visual definition from `environment\suns.ltx`.
#[derive(Debug, Clone, PartialEq)]
pub struct LensFlareDef {
    pub name: String,
    pub has_source: bool,
    pub has_flares: bool,
    pub has_gradient: bool,
    pub source_radius: f32,
    pub source_ignore_color: bool,
    pub source_texture: String,
    pub source_shader: String,
    pub gradient_radius: f32,
    pub gradient_opacity: f32,
    pub gradient_texture: String,
    pub gradient_shader: String,
    pub flares: Vec<LensFlareElement>,
    pub blend_rise_speed: f32,
    pub blend_down_speed: f32,
}

impl LensFlareDef {
    pub fn from_ltx(doc: &OwnedLtxDocument, section: &str) -> Self {
        let has_source = doc.read_bool(section, "sun", false);
        let (source_radius, source_ignore_color, source_texture, source_shader) = if has_source {
            (
                doc.read_f32(section, "sun_radius", 0.1),
                doc.read_bool(section, "sun_ignore_color", false),
                doc.read_str(section, "sun_texture").unwrap_or("").trim().to_string(),
                doc.read_str(section, "sun_shader").unwrap_or("").trim().to_string(),
            )
        } else {
            (0.0, false, String::new(), String::new())
        };

        let has_flares = doc.read_bool(section, "flares", false);
        let mut flares = Vec::new();
        if has_flares {
            let shader = doc.read_str(section, "flare_shader").unwrap_or("").trim().to_string();
            let textures = parse_csv_strings(doc.read_str(section, "flare_textures"));
            let radii: Vec<f32> = parse_csv_strings(doc.read_str(section, "flare_radius"))
                .into_iter()
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect();
            let opacities: Vec<f32> = parse_csv_strings(doc.read_str(section, "flare_opacity"))
                .into_iter()
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect();
            let positions: Vec<f32> = parse_csv_strings(doc.read_str(section, "flare_position"))
                .into_iter()
                .map(|s| s.parse::<f32>().unwrap_or(0.0))
                .collect();

            for (i, texture) in textures.into_iter().enumerate() {
                flares.push(LensFlareElement {
                    radius: radii.get(i).copied().unwrap_or(0.0),
                    opacity: opacities.get(i).copied().unwrap_or(0.0),
                    position: positions.get(i).copied().unwrap_or(0.0),
                    texture,
                    shader: shader.clone(),
                });
            }
        }

        let has_gradient = doc.read_bool(section, "gradient", false);
        let (gradient_radius, gradient_opacity, gradient_texture, gradient_shader) = if has_gradient
        {
            (
                doc.read_f32(section, "gradient_radius", 0.0),
                doc.read_f32(section, "gradient_opacity", 0.0),
                doc.read_str(section, "gradient_texture")
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                doc.read_str(section, "gradient_shader")
                    .unwrap_or("")
                    .trim()
                    .to_string(),
            )
        } else {
            (0.0, 0.0, String::new(), String::new())
        };

        let eps_s = 1e-5_f32;
        let blend_rise_speed = 1.0 / (doc.read_f32(section, "blend_rise_time", 1.0).max(0.0) + eps_s);
        let blend_down_speed = 1.0 / (doc.read_f32(section, "blend_down_time", 1.0).max(0.0) + eps_s);

        Self {
            name: section.to_string(),
            has_source,
            has_flares,
            has_gradient,
            source_radius,
            source_ignore_color,
            source_texture,
            source_shader,
            gradient_radius,
            gradient_opacity,
            gradient_texture,
            gradient_shader,
            flares,
            blend_rise_speed,
            blend_down_speed,
        }
    }
}

/// Thunderbolt gradient flare (`SThunderboltDesc::SFlare`).
#[derive(Debug, Clone, PartialEq)]
pub struct ThunderboltFlareDef {
    pub shader: String,
    pub texture: String,
    pub radius: [f32; 2],
    pub opacity: f32,
}

/// Lightning bolt preset (`SThunderboltDesc`, `environment\thunderbolts.ltx`).
#[derive(Debug, Clone, PartialEq)]
pub struct ThunderboltDescDef {
    pub name: String,
    pub gradient_top: ThunderboltFlareDef,
    pub gradient_center: ThunderboltFlareDef,
    pub color_anim: String,
    pub lightning_model: String,
    pub sound: Option<String>,
}

impl ThunderboltDescDef {
    pub fn from_ltx(doc: &OwnedLtxDocument, section: &str) -> Self {
        let gradient_top = ThunderboltFlareDef {
            shader: doc
                .read_str(section, "gradient_top_shader")
                .unwrap_or("")
                .trim()
                .to_string(),
            texture: doc
                .read_str(section, "gradient_top_texture")
                .unwrap_or("")
                .trim()
                .to_string(),
            radius: parse_f32_2(doc.read_str(section, "gradient_top_radius"), [1.0, 1.0]),
            opacity: doc.read_f32(section, "gradient_top_opacity", 1.0),
        };
        let gradient_center = ThunderboltFlareDef {
            shader: doc
                .read_str(section, "gradient_center_shader")
                .unwrap_or("")
                .trim()
                .to_string(),
            texture: doc
                .read_str(section, "gradient_center_texture")
                .unwrap_or("")
                .trim()
                .to_string(),
            radius: parse_f32_2(doc.read_str(section, "gradient_center_radius"), [1.0, 1.0]),
            opacity: doc.read_f32(section, "gradient_center_opacity", 1.0),
        };
        Self {
            name: section.to_string(),
            gradient_top,
            gradient_center,
            color_anim: doc.read_str(section, "color_anim").unwrap_or("").trim().to_string(),
            lightning_model: doc
                .read_str(section, "lightning_model")
                .unwrap_or("")
                .trim()
                .to_string(),
            sound: doc
                .read_str(section, "sound")
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        }
    }
}

/// Collection of thunderbolt presets (`SThunderboltCollection`, `thunderbolt_collections.ltx`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThunderboltCollectionDef {
    pub name: String,
    pub thunderbolts: Vec<String>,
}

impl ThunderboltCollectionDef {
    pub fn from_ltx(doc: &OwnedLtxDocument, section: &str) -> Self {
        let mut thunderbolts: Vec<String> = doc
            .sections
            .get(section)
            .map(|sec| sec.entries.keys().cloned().collect())
            .unwrap_or_default();
        thunderbolts.sort();
        Self {
            name: section.to_string(),
            thunderbolts,
        }
    }
}

/// Weather keyframe descriptor (`CEnvDescriptor`, `Environment_misc.cpp:267-360`).
#[derive(Debug, Clone, PartialEq)]
pub struct EnvDescriptor {
    pub identifier: String,
    pub exec_time: f32,
    pub exec_time_loaded: f32,
    pub sky_texture: String,
    pub sky_texture_env: String,
    pub clouds_texture: String,
    pub clouds_color: [f32; 4],
    pub sky_color: [f32; 3],
    pub sky_rotation: f32,
    pub far_plane: f32,
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    pub fog_distance: f32,
    pub rain_density: f32,
    pub rain_color: [f32; 3],
    pub bolt_period: f32,
    pub bolt_duration: f32,
    pub wind_velocity: f32,
    pub wind_direction: f32,
    pub volumetric_intensity_factor: f32,
    pub volumetric_distance_factor: f32,
    pub ambient_color: [f32; 3],
    pub hemi_color: [f32; 4],
    pub sun_color: [f32; 3],
    pub sun_dir: [f32; 3],
    pub sun_shafts_intensity: f32,
    pub water_intensity: f32,
    pub hemi_vibrance: f32,
    pub hemi_contrast: f32,
    pub wet_surfaces: f32,
    pub tree_amplitude_intensity: f32,
    pub bloom_threshold: f32,
    pub bloom_exposure: f32,
    pub bloom_sky_intensity: f32,
    pub sun_id: String,
    pub thunderbolt_collection_id: String,
    pub env_ambient_id: Option<String>,
}

impl Default for EnvDescriptor {
    fn default() -> Self {
        Self {
            identifier: "00:00:00".to_string(),
            exec_time: 0.0,
            exec_time_loaded: 0.0,
            sky_texture: String::new(),
            sky_texture_env: String::new(),
            clouds_texture: String::new(),
            clouds_color: [1.0, 1.0, 1.0, 1.0],
            sky_color: [1.0, 1.0, 1.0],
            sky_rotation: 0.0,
            far_plane: 400.0,
            fog_color: [1.0, 1.0, 1.0],
            fog_density: 0.0,
            fog_distance: 400.0,
            rain_density: 0.0,
            rain_color: [0.0, 0.0, 0.0],
            bolt_period: 0.0,
            bolt_duration: 0.0,
            wind_velocity: 0.0,
            wind_direction: 0.0,
            volumetric_intensity_factor: 1.0,
            volumetric_distance_factor: 1.0,
            ambient_color: [0.0, 0.0, 0.0],
            hemi_color: [1.0, 1.0, 1.0, 1.0],
            sun_color: [1.0, 1.0, 1.0],
            sun_dir: [0.0, -1.0, 0.0],
            sun_shafts_intensity: 0.0,
            water_intensity: 1.0,
            hemi_vibrance: 1.0,
            hemi_contrast: 1.0,
            wet_surfaces: 0.0,
            tree_amplitude_intensity: 0.01,
            bloom_threshold: 3.5,
            bloom_exposure: 3.0,
            bloom_sky_intensity: 0.6,
            sun_id: String::new(),
            thunderbolt_collection_id: String::new(),
            env_ambient_id: None,
        }
    }
}

impl EnvDescriptor {
    pub fn from_ltx(doc: &OwnedLtxDocument, identifier: &str) -> Result<Self, FormatError> {
        let exec_time = parse_hhmmss_seconds(identifier)?;
        let sky_texture = doc
            .read_str(identifier, "sky_texture")
            .unwrap_or("")
            .trim()
            .to_string();
        let sky_texture_env = if sky_texture.is_empty() {
            String::new()
        } else {
            format!("{sky_texture}#small")
        };
        let clouds_texture = doc
            .read_str(identifier, "clouds_texture")
            .unwrap_or("")
            .trim()
            .to_string();

        // `clouds_color = r, g, b, a, multiplier` (`Environment_misc.cpp:283-286`)
        let clouds_color = if let Some(raw) = doc.read_str(identifier, "clouds_color") {
            let mut vals = [1.0, 1.0, 1.0, 1.0, 2.0];
            for (i, tok) in raw.split(',').take(5).enumerate() {
                if let Ok(v) = tok.trim().parse::<f32>() {
                    vals[i] = v;
                }
            }
            let mult = 0.5 * vals[4];
            [vals[0] * mult, vals[1] * mult, vals[2] * mult, vals[3]]
        } else {
            [1.0, 1.0, 1.0, 1.0]
        };

        let thunderbolt_collection_id = doc
            .read_str(identifier, "thunderbolt_collection")
            .unwrap_or("")
            .trim()
            .to_string();
        let (bolt_period, bolt_duration) = if !thunderbolt_collection_id.is_empty() {
            (
                doc.read_f32(identifier, "thunderbolt_period", 0.0),
                doc.read_f32(identifier, "thunderbolt_duration", 0.0),
            )
        } else {
            (0.0, 0.0)
        };

        Ok(Self {
            identifier: identifier.to_string(),
            exec_time,
            exec_time_loaded: exec_time,
            sky_texture,
            sky_texture_env,
            clouds_texture,
            clouds_color,
            sky_color: parse_f32_3(doc.read_str(identifier, "sky_color"), [1.0; 3]),
            sky_rotation: doc.read_f32(identifier, "sky_rotation", 0.0).to_radians(),
            far_plane: doc.read_f32(identifier, "far_plane", 400.0),
            fog_color: parse_f32_3(doc.read_str(identifier, "fog_color"), [1.0; 3]),
            fog_density: doc.read_f32(identifier, "fog_density", 0.0),
            fog_distance: doc.read_f32(identifier, "fog_distance", 400.0),
            rain_density: doc.read_f32(identifier, "rain_density", 0.0).clamp(0.0, 1.0),
            rain_color: parse_f32_3(doc.read_str(identifier, "rain_color"), [0.0; 3]),
            bolt_period,
            bolt_duration,
            wind_velocity: doc.read_f32(identifier, "wind_velocity", 0.0),
            wind_direction: doc.read_f32(identifier, "wind_direction", 0.0).to_radians(),
            volumetric_intensity_factor: doc.read_f32(
                identifier,
                "volumetric_intensity_factor",
                1.0,
            ),
            volumetric_distance_factor: doc.read_f32(identifier, "volumetric_distance_factor", 1.0),
            ambient_color: parse_f32_3(doc.read_str(identifier, "ambient_color"), [0.0; 3]),
            hemi_color: parse_f32_4(doc.read_str(identifier, "hemisphere_color"), [1.0; 4]),
            sun_color: parse_f32_3(doc.read_str(identifier, "sun_color"), [1.0; 3]),
            sun_dir: [0.0, -1.0, 0.0],
            sun_shafts_intensity: doc.read_f32(identifier, "sun_shafts_intensity", 0.0),
            water_intensity: doc.read_f32(identifier, "water_intensity", 1.0),
            hemi_vibrance: doc.read_f32(identifier, "hemi_vibrance", 1.0),
            hemi_contrast: doc.read_f32(identifier, "hemi_contrast", 1.0),
            wet_surfaces: doc.read_f32(identifier, "wet_surface_factor", 0.0),
            tree_amplitude_intensity: doc.read_f32(identifier, "tree_amplitude_intensity", 0.01),
            bloom_threshold: doc.read_f32(identifier, "bloom_threshold", 3.5),
            bloom_exposure: doc.read_f32(identifier, "bloom_exposure", 3.0),
            bloom_sky_intensity: doc.read_f32(identifier, "bloom_sky_intensity", 0.6),
            sun_id: doc.read_str(identifier, "sun").unwrap_or("").trim().to_string(),
            thunderbolt_collection_id,
            env_ambient_id: doc
                .read_str(identifier, "ambient")
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        })
    }
}

/// Optional additive environment brightness boosts (`env_boost` in Anomaly).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EnvBoost {
    pub sky_color: f32,
    pub clouds_color: f32,
    pub ambient: f32,
    pub hemi: f32,
    pub sun_color: f32,
    pub rain_color: f32,
    pub fog_color: f32,
}

/// Blended atmospheric state output by `lerp_descriptors` (`CEnvDescriptorMixer`).
#[derive(Debug, Clone, PartialEq)]
pub struct MixedEnvironment {
    pub weight: f32,
    pub from_id: String,
    pub to_id: String,
    pub sky_texture_a: String,
    pub sky_texture_b: String,
    pub sky_texture_env_a: String,
    pub sky_texture_env_b: String,
    pub clouds_texture_a: String,
    pub clouds_texture_b: String,
    pub clouds_color: [f32; 4],
    pub sky_color: [f32; 3],
    pub sky_rotation: f32,
    pub far_plane: f32,
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    pub fog_distance: f32,
    pub fog_near: f32,
    pub fog_far: f32,
    pub rain_density: f32,
    pub rain_color: [f32; 3],
    pub bolt_period: f32,
    pub bolt_duration: f32,
    pub wind_velocity: f32,
    pub wind_direction: f32,
    pub ambient_color: [f32; 3],
    pub hemi_color: [f32; 4],
    pub sun_color: [f32; 3],
    pub sun_dir: [f32; 3],
    pub sun_shafts_intensity: f32,
    pub water_intensity: f32,
    pub hemi_vibrance: f32,
    pub hemi_contrast: f32,
    pub wet_surfaces: f32,
    pub volumetric_intensity_factor: f32,
    pub volumetric_distance_factor: f32,
    pub tree_amplitude_intensity: f32,
    pub bloom_threshold: f32,
    pub bloom_exposure: f32,
    pub bloom_sky_intensity: f32,
    pub active_sun_id: String,
    pub active_thunderbolt_id: String,
}

impl MixedEnvironment {
    #[inline]
    pub fn sun_color_rgba(&self) -> [f32; 4] {
        [self.sun_color[0], self.sun_color[1], self.sun_color[2], 1.0]
    }

    #[inline]
    pub fn ambient_color_rgba(&self) -> [f32; 4] {
        [
            self.ambient_color[0],
            self.ambient_color[1],
            self.ambient_color[2],
            1.0,
        ]
    }

    #[inline]
    pub fn fog_color_rgba(&self) -> [f32; 4] {
        [
            self.fog_color[0],
            self.fog_color[1],
            self.fog_color[2],
            self.fog_density,
        ]
    }

    #[inline]
    pub fn sky_color_rgba(&self) -> [f32; 4] {
        [self.sky_color[0], self.sky_color[1], self.sky_color[2], 1.0]
    }

    #[inline]
    pub fn sun_dir_vec4(&self) -> [f32; 4] {
        [self.sun_dir[0], self.sun_dir[1], self.sun_dir[2], 0.0]
    }

    /// Unit direction pointing TOWARD the sun (`-sun_dir`), matching sky shader glow dot products.
    #[inline]
    pub fn sun_direction_to_sun_vec4(&self) -> [f32; 4] {
        [-self.sun_dir[0], -self.sun_dir[1], -self.sun_dir[2], 0.0]
    }

    /// Skybox vertex tint RGBA (`[sky_color.r, sky_color.g, sky_color.b, weight]`, `dxEnvironmentRender.cpp:300-301`).
    #[inline]
    pub fn sky_tint_rgba(&self) -> [f32; 4] {
        [
            self.sky_color[0],
            self.sky_color[1],
            self.sky_color[2],
            self.weight,
        ]
    }

    /// Sun RGB combined with an explicit flare/glow alpha for `SkyPushConstants::sun_color`.
    #[inline]
    pub fn sun_glow_rgba(&self, glow_intensity: f32) -> [f32; 4] {
        [
            self.sun_color[0],
            self.sun_color[1],
            self.sun_color[2],
            glow_intensity.max(0.0),
        ]
    }

    /// Blended clouds tint RGBA (`clouds_color` with `0.5 * multiplier` already baked in).
    #[inline]
    pub fn clouds_color_rgba(&self) -> [f32; 4] {
        self.clouds_color
    }

    /// Two normalized `[0..1]` UV wind drift direction pairs `(wd0_xz, wd1_xz)` matching
    /// `dxEnvironmentRender.cpp:368-373`.
    #[inline]
    pub fn clouds_wind_uv_directions(&self) -> ([f32; 2], [f32; 2]) {
        let wd0 = dir_from_hp(std::f32::consts::FRAC_PI_4, 0.0);
        let wd1 = dir_from_hp(
            std::f32::consts::FRAC_PI_4 + std::f32::consts::FRAC_PI_8,
            0.0,
        );
        (
            [0.5 * wd0[0] + 0.5, 0.5 * wd0[2] + 0.5],
            [0.5 * wd1[0] + 0.5, 0.5 * wd1[2] + 0.5],
        )
    }

    /// Normalized `[0..1]` clouds wind direction vector `[wd0.x, wd0.z, wd1.x, wd1.z]`.
    ///
    /// Note: C++ `dxEnvironmentRender.cpp:374` packs `color_rgba(wind_dir.x, wind_dir.y, wind_dir.w, wind_dir.z)`,
    /// swapping components `2` and `3` for DX9 fixed-function `D3DCOLOR`.
    #[inline]
    pub fn clouds_wind_rgba(&self) -> [f32; 4] {
        let (d0, d1) = self.clouds_wind_uv_directions();
        [d0[0], d0[1], d1[0], d1[1]]
    }

    /// Rain vertex tint RGBA (`[rain_color.rgb, rain_density * 0.5 + 0.5]`, `dxRainRender.cpp:94-96`).
    #[inline]
    pub fn rain_color_rgba(&self) -> [f32; 4] {
        [
            self.rain_color[0],
            self.rain_color[1],
            self.rain_color[2],
            (self.rain_density * 0.5 + 0.5).clamp(0.0, 1.0),
        ]
    }

    /// Target rain streak instance count for `max_particles` (`dxRainRender.cpp:87`).
    #[inline]
    pub fn rain_desired_count(&self, max_particles: u32) -> u32 {
        if self.rain_density <= 1e-5 {
            0
        } else {
            (0.01 * (1.0 + self.rain_density.clamp(0.0, 1.0) * 99.0) * (max_particles as f32))
                .floor()
                .max(0.0) as u32
        }
    }

    /// Rain fall direction unit vector and horizontal XZ spawn offset matching `CEffect_Rain::Prepare`
    /// (`Rain.cpp:81-105`, without per-frame Perlin gust).
    pub fn rain_streak_axis_and_offset(&self) -> ([f32; 3], [f32; 2]) {
        let wind_vel = (self.wind_velocity * 0.001).clamp(0.0, 1.0);
        let wind_dir = -self.wind_direction;
        let pitch = RAIN_DROP_MAX_ANGLE_RAD * wind_vel;
        let axis = dir_from_hp(wind_dir, pitch - std::f32::consts::FRAC_PI_2);
        let denom = (std::f32::consts::FRAC_PI_2 - pitch).sin();
        let dist = if denom.abs() > 1e-5 {
            pitch.sin() * RAIN_SOURCE_OFFSET / denom
        } else {
            0.0
        };
        let fix_north = wind_dir - std::f32::consts::FRAC_PI_2;
        (axis, [dist * fix_north.cos(), dist * fix_north.sin()])
    }

    /// Canonical VFS `.dds` paths `(sky_a, sky_b)` for primary skybox cubemaps.
    #[inline]
    pub fn sky_dds_vpaths(&self) -> (Option<String>, Option<String>) {
        (
            (!self.sky_texture_a.is_empty()).then(|| texture_stem_to_dds_vpath(&self.sky_texture_a)),
            (!self.sky_texture_b.is_empty()).then(|| texture_stem_to_dds_vpath(&self.sky_texture_b)),
        )
    }

    /// Canonical VFS `.dds` paths `(sky_env_a, sky_env_b)` for `#small` irradiance cubemaps.
    #[inline]
    pub fn sky_env_dds_vpaths(&self) -> (Option<String>, Option<String>) {
        (
            (!self.sky_texture_env_a.is_empty())
                .then(|| texture_stem_to_dds_vpath(&self.sky_texture_env_a)),
            (!self.sky_texture_env_b.is_empty())
                .then(|| texture_stem_to_dds_vpath(&self.sky_texture_env_b)),
        )
    }

    /// Canonical VFS `.dds` paths `(clouds_a, clouds_b)` for scrolling cloud textures.
    #[inline]
    pub fn clouds_dds_vpaths(&self) -> (Option<String>, Option<String>) {
        (
            (!self.clouds_texture_a.is_empty())
                .then(|| texture_stem_to_dds_vpath(&self.clouds_texture_a)),
            (!self.clouds_texture_b.is_empty())
                .then(|| texture_stem_to_dds_vpath(&self.clouds_texture_b)),
        )
    }

    /// Skybox world transform `rotateY(sky_rotation) + translate_over(camera_pos)`
    /// (`dxEnvironmentRender.cpp:296-297`).
    #[inline]
    pub fn sky_world_transform(&self, camera_pos: [f32; 3]) -> XRayMatrix {
        let mut m = matrix_rotate_y(self.sky_rotation);
        m[3][0] = camera_pos[0];
        m[3][1] = camera_pos[1];
        m[3][2] = camera_pos[2];
        m[3][3] = 1.0;
        m
    }

    /// Clouds dome world transform `rotateY(sky_rotation) * scale(10, 0.4, 10) + translate_over(camera_pos)`
    /// (`dxEnvironmentRender.cpp:362-366`).
    #[inline]
    pub fn clouds_world_transform(&self, camera_pos: [f32; 3]) -> XRayMatrix {
        let mut m_scale = XRAY_IDENTITY_MATRIX;
        m_scale[0][0] = 10.0;
        m_scale[1][1] = 0.4;
        m_scale[2][2] = 10.0;
        let rot = matrix_rotate_y(self.sky_rotation);
        let mut m = matrix_mul_43(&rot, &m_scale);
        m[3][0] = camera_pos[0];
        m[3][1] = camera_pos[1];
        m[3][2] = camera_pos[2];
        m[3][3] = 1.0;
        m
    }

    #[inline]
    pub fn effective_ambient_rgb(&self, hemi_weight: f32) -> [f32; 3] {
        let hw = hemi_weight.max(0.0);
        [
            self.ambient_color[0] + self.hemi_color[0] * hw,
            self.ambient_color[1] + self.hemi_color[1] * hw,
            self.ambient_color[2] + self.hemi_color[2] * hw,
        ]
    }
}

#[inline]
fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

#[inline]
fn lerp_3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp_f32(a[0], b[0], t),
        lerp_f32(a[1], b[1], t),
        lerp_f32(a[2], b[2], t),
    ]
}

#[inline]
fn lerp_4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        lerp_f32(a[0], b[0], t),
        lerp_f32(a[1], b[1], t),
        lerp_f32(a[2], b[2], t),
        lerp_f32(a[3], b[3], t),
    ]
}

#[inline]
fn boost_clamp_3(v: &mut [f32; 3], add: f32) {
    if add != 0.0 {
        for c in v {
            *c = (*c + add).clamp(0.0, 1.0);
        }
    }
}

/// Blend two `EnvDescriptor` keyframes at weight `f` (`[0..1]`), applying accumulated `EnvModAccum`
/// and optional `EnvBoost` (`CEnvDescriptorMixer::lerp`, `Environment_misc.cpp:437-592`).
#[allow(clippy::too_many_arguments)]
pub fn lerp_descriptors(
    a: &EnvDescriptor,
    b: &EnvDescriptor,
    f: f32,
    mdf: &EnvModAccum,
    vis_distance: f32,
    boost: &EnvBoost,
    sun_shafts_min: f32,
    sun_shafts_value: f32,
) -> MixedEnvironment {
    let f = f.clamp(0.0, 1.0);
    let modif_power = 1.0 / (mdf.total_power + 1.0);

    let mut clouds_color = lerp_4(a.clouds_color, b.clouds_color, f);
    let sky_rotation = lerp_f32(a.sky_rotation, b.sky_rotation, f);

    let far_plane = if (mdf.active_flags & env_mod_flags::VIEW_DIST) != 0 {
        (lerp_f32(a.far_plane, b.far_plane, f) + mdf.far_plane) * vis_distance * modif_power
    } else {
        lerp_f32(a.far_plane, b.far_plane, f) * vis_distance
    };

    let mut fog_color = lerp_3(a.fog_color, b.fog_color, f);
    if (mdf.active_flags & env_mod_flags::FOG_COLOR) != 0 {
        for (c, m) in fog_color.iter_mut().zip(mdf.fog_color.iter()) {
            *c = (*c + *m) * modif_power;
        }
    }

    let mut fog_density = lerp_f32(a.fog_density, b.fog_density, f);
    if (mdf.active_flags & env_mod_flags::FOG_DENSITY) != 0 {
        fog_density = (fog_density + mdf.fog_density) * modif_power;
    }

    let max_fog_dist = (far_plane - 10.0).max(1.0);
    let fog_distance = lerp_f32(a.fog_distance, b.fog_distance, f).clamp(1.0, max_fog_dist);
    let fog_near = (1.0 - fog_density) * 0.85 * fog_distance;
    let fog_far = 0.99 * fog_distance;

    let rain_density = lerp_f32(a.rain_density, b.rain_density, f);
    let mut rain_color = lerp_3(a.rain_color, b.rain_color, f);

    let bolt_period = lerp_f32(a.bolt_period, b.bolt_period, f);
    let bolt_duration = lerp_f32(a.bolt_duration, b.bolt_duration, f);
    let wind_velocity = lerp_f32(a.wind_velocity, b.wind_velocity, f);
    let wind_direction = lerp_f32(a.wind_direction, b.wind_direction, f);

    let mut sun_shafts = lerp_f32(a.sun_shafts_intensity, b.sun_shafts_intensity, f);
    sun_shafts *= 1.0 - sun_shafts_min;
    sun_shafts += sun_shafts_min;
    sun_shafts *= sun_shafts_value;
    sun_shafts = sun_shafts.clamp(0.0, 1.0);

    let water_intensity = lerp_f32(a.water_intensity, b.water_intensity, f);
    let hemi_vibrance = lerp_f32(a.hemi_vibrance, b.hemi_vibrance, f);
    let hemi_contrast = lerp_f32(a.hemi_contrast, b.hemi_contrast, f);
    let wet_surfaces = lerp_f32(a.wet_surfaces, b.wet_surfaces, f);
    let volumetric_intensity_factor = lerp_f32(
        a.volumetric_intensity_factor,
        b.volumetric_intensity_factor,
        f,
    );
    let volumetric_distance_factor =
        lerp_f32(a.volumetric_distance_factor, b.volumetric_distance_factor, f);
    let tree_amplitude_intensity =
        lerp_f32(a.tree_amplitude_intensity, b.tree_amplitude_intensity, f);
    let bloom_threshold = lerp_f32(a.bloom_threshold, b.bloom_threshold, f);
    let bloom_exposure = lerp_f32(a.bloom_exposure, b.bloom_exposure, f);
    let bloom_sky_intensity = lerp_f32(a.bloom_sky_intensity, b.bloom_sky_intensity, f);

    let mut sky_color = lerp_3(a.sky_color, b.sky_color, f);
    if (mdf.active_flags & env_mod_flags::SKY_COLOR) != 0 {
        for (c, m) in sky_color.iter_mut().zip(mdf.sky_color.iter()) {
            *c = (*c + *m) * modif_power;
        }
    }

    let mut ambient_color = lerp_3(a.ambient_color, b.ambient_color, f);
    if (mdf.active_flags & env_mod_flags::AMBIENT_COLOR) != 0 {
        for (c, m) in ambient_color.iter_mut().zip(mdf.ambient.iter()) {
            *c = (*c + *m) * modif_power;
        }
    }

    let mut hemi_color = lerp_4(a.hemi_color, b.hemi_color, f);
    if (mdf.active_flags & env_mod_flags::HEMI_COLOR) != 0 {
        hemi_color[0] = (hemi_color[0] + mdf.hemi_color[0]) * modif_power;
        hemi_color[1] = (hemi_color[1] + mdf.hemi_color[1]) * modif_power;
        hemi_color[2] = (hemi_color[2] + mdf.hemi_color[2]) * modif_power;
    }

    let mut sun_color = lerp_3(a.sun_color, b.sun_color, f);

    // Apply brightness boosts (Environment_misc.cpp:537-592)
    boost_clamp_3(&mut sky_color, boost.sky_color);
    if boost.clouds_color != 0.0 {
        for c in &mut clouds_color {
            *c = (*c + boost.clouds_color).clamp(0.0, 1.0);
        }
    }
    boost_clamp_3(&mut ambient_color, boost.ambient);
    if boost.hemi != 0.0 {
        for c in &mut hemi_color[..3] {
            *c = (*c + boost.hemi).clamp(0.0, 1.0);
        }
    }
    boost_clamp_3(&mut sun_color, boost.sun_color);
    boost_clamp_3(&mut rain_color, boost.rain_color);
    boost_clamp_3(&mut fog_color, boost.fog_color);

    let (active_sun_id, active_thunderbolt_id) = if f < 0.5 {
        (a.sun_id.clone(), a.thunderbolt_collection_id.clone())
    } else {
        (b.sun_id.clone(), b.thunderbolt_collection_id.clone())
    };

    MixedEnvironment {
        weight: f,
        from_id: a.identifier.clone(),
        to_id: b.identifier.clone(),
        sky_texture_a: a.sky_texture.clone(),
        sky_texture_b: b.sky_texture.clone(),
        sky_texture_env_a: a.sky_texture_env.clone(),
        sky_texture_env_b: b.sky_texture_env.clone(),
        clouds_texture_a: a.clouds_texture.clone(),
        clouds_texture_b: b.clouds_texture.clone(),
        clouds_color,
        sky_color,
        sky_rotation,
        far_plane,
        fog_color,
        fog_density,
        fog_distance,
        fog_near,
        fog_far,
        rain_density,
        rain_color,
        bolt_period,
        bolt_duration,
        wind_velocity,
        wind_direction,
        ambient_color,
        hemi_color,
        sun_color,
        sun_dir: a.sun_dir,
        sun_shafts_intensity: sun_shafts,
        water_intensity,
        hemi_vibrance,
        hemi_contrast,
        wet_surfaces,
        volumetric_intensity_factor,
        volumetric_distance_factor,
        tree_amplitude_intensity,
        bloom_threshold,
        bloom_exposure,
        bloom_sky_intensity,
        active_sun_id,
        active_thunderbolt_id,
    }
}

/// Ordered 24-hour cycle of weather keyframes (`weathers\<cycle>.ltx`).
#[derive(Debug, Clone, PartialEq)]
pub struct WeatherCycle {
    pub name: String,
    pub descriptors: Vec<EnvDescriptor>,
}

impl WeatherCycle {
    pub fn from_ltx(name: &str, doc: &OwnedLtxDocument) -> Result<Self, FormatError> {
        let mut descriptors = Vec::with_capacity(doc.sections.len());
        for sec in doc.sections.keys() {
            descriptors.push(EnvDescriptor::from_ltx(doc, sec)?);
        }
        descriptors.sort_by(|a, b| a.exec_time_loaded.total_cmp(&b.exec_time_loaded));
        Ok(Self {
            name: name.to_string(),
            descriptors,
        })
    }

    /// Find bracketing descriptors `(prev, next, weight)` for `game_time_sec` (`CEnvironment::SelectEnvs`).
    pub fn select_at(&self, game_time_sec: f32) -> Option<(&EnvDescriptor, &EnvDescriptor, f32)> {
        if self.descriptors.len() < 2 {
            return self
                .descriptors
                .first()
                .map(|d| (d, d, 0.0));
        }
        let gt = normalize_time(game_time_sec);
        let idx = self.descriptors.partition_point(|d| d.exec_time < gt);
        let (e0, e1) = if idx == self.descriptors.len() || idx == 0 {
            (&self.descriptors[self.descriptors.len() - 1], &self.descriptors[0])
        } else {
            (&self.descriptors[idx - 1], &self.descriptors[idx])
        };
        let w = time_weight(gt, e0.exec_time, e1.exec_time);
        Some((e0, e1, w))
    }
}

/// Complete environment & weather preset library loaded from `$game_configs$\environment\`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WeatherLibrary {
    pub global_params: EnvGlobalParams,
    pub sun_positions: Option<SunPositionsConfig>,
    pub cycles: BTreeMap<String, WeatherCycle>,
    pub effects_cycles: BTreeMap<String, WeatherCycle>,
    pub ambients: BTreeMap<String, EnvAmbientDef>,
    pub sound_channels: BTreeMap<String, EnvSoundChannelDef>,
    pub ambient_effects: BTreeMap<String, EnvAmbientEffectDef>,
    pub lens_flares: BTreeMap<String, LensFlareDef>,
    pub thunderbolts: BTreeMap<String, ThunderboltDescDef>,
    pub thunderbolt_collections: BTreeMap<String, ThunderboltCollectionDef>,
}

impl WeatherLibrary {
    /// Sample and blend a named weather cycle at `game_time_sec`.
    pub fn sample_at(
        &self,
        cycle_name: &str,
        game_time_sec: f32,
        env_mods: Option<&LevelEnvModifiers>,
        camera_pos: [f32; 3],
        vis_distance: f32,
        boost: &EnvBoost,
    ) -> Option<MixedEnvironment> {
        let cycle = self.cycles.get(cycle_name)?;
        let (a, b, w) = cycle.select_at(game_time_sec)?;
        let accum = env_mods
            .map(|m| m.evaluate_at(camera_pos))
            .unwrap_or(EnvModAccum {
                total_power: 0.0,
                active_flags: 0,
                far_plane: 0.0,
                fog_color: [0.0; 3],
                fog_density: 0.0,
                ambient: [0.0; 3],
                sky_color: [0.0; 3],
                hemi_color: [0.0; 3],
            });
        let mut mixed = lerp_descriptors(a, b, w, &accum, vis_distance, boost, 0.0, 1.0);
        if let Some(sun_pos) = &self.sun_positions {
            mixed.sun_dir = sun_pos.sun_dir_at_time(game_time_sec);
        }
        Some(mixed)
    }

    pub fn parse_sound_channels(&mut self, doc: &OwnedLtxDocument) -> Result<(), FormatError> {
        for sec in doc.sections.keys() {
            if doc.read_str(sec, "sounds").is_some() {
                self.sound_channels
                    .insert(sec.clone(), EnvSoundChannelDef::from_ltx(doc, sec)?);
            }
        }
        Ok(())
    }

    pub fn parse_ambient_effects(&mut self, doc: &OwnedLtxDocument) -> Result<(), FormatError> {
        for sec in doc.sections.keys() {
            if doc.read_str(sec, "particles").is_some() {
                self.ambient_effects
                    .insert(sec.clone(), EnvAmbientEffectDef::from_ltx(doc, sec)?);
            }
        }
        Ok(())
    }

    pub fn parse_ambients(&mut self, doc: &OwnedLtxDocument) -> Result<(), FormatError> {
        for sec in doc.sections.keys() {
            if doc.read_str(sec, "sound_channels").is_some()
                || doc.read_str(sec, "effects").is_some()
            {
                self.ambients
                    .insert(sec.clone(), EnvAmbientDef::from_ltx(doc, sec)?);
            }
        }
        Ok(())
    }

    pub fn parse_suns(&mut self, doc: &OwnedLtxDocument) {
        for sec in doc.sections.keys() {
            if doc.read_str(sec, "sun").is_some() {
                self.lens_flares
                    .insert(sec.clone(), LensFlareDef::from_ltx(doc, sec));
            }
        }
    }

    pub fn parse_thunderbolts(&mut self, doc: &OwnedLtxDocument) {
        for sec in doc.sections.keys() {
            if doc.read_str(sec, "lightning_model").is_some() {
                self.thunderbolts
                    .insert(sec.clone(), ThunderboltDescDef::from_ltx(doc, sec));
            }
        }
    }

    pub fn parse_thunderbolt_collections(&mut self, doc: &OwnedLtxDocument) {
        for sec in doc.sections.keys() {
            self.thunderbolt_collections
                .insert(sec.clone(), ThunderboltCollectionDef::from_ltx(doc, sec));
        }
    }
}

/// Parse an `[...]` LTX string into an `OwnedLtxDocument` cleanly without external files.
pub fn parse_ltx_str(text: &str) -> OwnedLtxDocument {
    use xray_core::ltx_loader::LoadedLtxSource;
    OwnedLtxDocument::from_sources(&[LoadedLtxSource {
        virtual_path: "<memory>".to_string(),
        depth: 0,
        text: text.to_string(),
    }])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders_xr::EnvModifier;

    #[test]
    fn test_weather_cycle_wraparound_selection_and_lerp() {
        let cycle_ltx = r#"
[00:00:00]
sky_texture = sky\sky_night
clouds_texture = sky\sky_oblaka
clouds_color = 0.2, 0.2, 0.4, 0.5, 2.0
sky_color = 0.1, 0.1, 0.2
sky_rotation = 90
far_plane = 400
fog_color = 0.1, 0.1, 0.2
fog_density = 0.4
fog_distance = 200
rain_density = 0.0
rain_color = 0.0, 0.0, 0.0
wind_velocity = 100.0
wind_direction = 0.0
ambient_color = 0.05, 0.05, 0.05
hemisphere_color = 0.2, 0.2, 0.3, 1.0
sun_color = 0.0, 0.0, 0.0
sun = none
thunderbolt_collection = tb_coll_1
thunderbolt_period = 20.0
thunderbolt_duration = 0.4
ambient = env_night
bloom_threshold = 3.0

[06:00:00]
sky_texture = sky\sky_dawn
clouds_texture = sky\sky_oblaka
clouds_color = 0.8, 0.6, 0.4, 1.0, 1.0
sky_color = 0.7, 0.5, 0.4
sky_rotation = 0
far_plane = 800
fog_color = 0.5, 0.4, 0.3
fog_density = 0.2
fog_distance = 400
rain_density = 0.6
rain_color = 0.3, 0.3, 0.3
wind_velocity = 300.0
wind_direction = 180.0
ambient_color = 0.25, 0.25, 0.25
hemisphere_color = 0.6, 0.6, 0.6, 1.0
sun_color = 1.0, 0.8, 0.6
sun = sun_clear
thunderbolt_collection =
ambient = env_dawn
bloom_threshold = 4.0
"#;
        let doc = parse_ltx_str(cycle_ltx);
        let cycle = WeatherCycle::from_ltx("w_clear", &doc).unwrap();
        assert_eq!(cycle.descriptors.len(), 2);
        assert_eq!(cycle.descriptors[0].sky_texture_env, "sky\\sky_night#small");
        // clouds_color: 0.5 * multiplier (=2.0) * rgb = 1.0 * [0.2,0.2,0.4], alpha preserved 0.5
        assert!((cycle.descriptors[0].clouds_color[0] - 0.2).abs() < 1e-6);
        assert!((cycle.descriptors[0].clouds_color[3] - 0.5).abs() < 1e-6);

        // At 03:00:00 -> exact 0.5 blend between 00:00 and 06:00
        let (a, b, w) = cycle.select_at(3.0 * 3600.0).unwrap();
        assert_eq!(a.identifier, "00:00:00");
        assert_eq!(b.identifier, "06:00:00");
        assert!((w - 0.5).abs() < 1e-5);

        // At 18:00:00 -> wraps 06:00:00 -> 00:00:00 (18h interval: 12/18 = 0.6666...)
        let (wa, wb, ww) = cycle.select_at(18.0 * 3600.0).unwrap();
        assert_eq!(wa.identifier, "06:00:00");
        assert_eq!(wb.identifier, "00:00:00");
        assert!((ww - (12.0 / 18.0)).abs() < 1e-4);

        // Blend with a local EnvModifier
        let env_mods = LevelEnvModifiers {
            version: 0x0016,
            modifiers: vec![EnvModifier {
                name: Some("fog_zone".to_string()),
                position: [0.0, 0.0, 0.0],
                radius: 100.0,
                power: 1.0,
                far_plane: 100.0,
                fog_color: [1.0, 1.0, 1.0],
                fog_density: 0.7,
                ambient: [0.0; 3],
                sky_color: [0.0; 3],
                hemi_color: [0.0; 3],
                use_flags: env_mod_flags::FOG_DENSITY | env_mod_flags::VIEW_DIST,
            }],
        };

        let mut lib = WeatherLibrary::default();
        lib.cycles.insert("w_clear".to_string(), cycle);
        let sample = lib
            .sample_at(
                "w_clear",
                3.0 * 3600.0,
                Some(&env_mods),
                [0.0, 0.0, 0.0],
                1.0,
                &EnvBoost::default(),
            )
            .unwrap();
        // Base far_plane midpoint = 600; + 100 mod = 700; * 1 / (1+1) = 350
        assert!((sample.far_plane - 350.0).abs() < 1e-4);
        assert!(sample.fog_near < sample.fog_far);
        assert!((sample.sun_color_rgba()[0] - 0.5).abs() < 1e-5);
        assert_eq!(sample.sun_color_rgba()[3], 1.0);
        assert!((sample.ambient_color_rgba()[0] - 0.15).abs() < 1e-5);
        assert_eq!(sample.fog_color_rgba()[3], sample.fog_density);
        assert!((sample.sky_color_rgba()[0] - 0.4).abs() < 1e-5);
        assert_eq!(sample.sun_dir_vec4()[3], 0.0);
        // ambient midpoint = 0.15, hemi midpoint = 0.4 -> 0.15 + 0.5 * 0.4 = 0.35
        assert!((sample.effective_ambient_rgb(0.5)[0] - 0.35).abs() < 1e-5);
    }

    #[test]
    fn test_dynamic_sun_and_presets_parsing() {
        let (dir_noon, blend_noon) = calculate_dynamic_sun_dir(12.0 * 3600.0);
        assert!(dir_noon[1] < 0.0, "noon sun ray should point downward");
        assert!((blend_noon - 1.0).abs() < 1e-5);

        let suns_doc = parse_ltx_str(
            r#"
[sun_clear]
sun = true
sun_shader = effects\sun
sun_texture = fx\fx_sun
sun_radius = 0.08
sun_ignore_color = false
flares = true
flare_shader = effects\flare
flare_textures = fx\fx_flare1, fx\fx_flare2
flare_radius = 0.1, 0.2
flare_opacity = 0.5, 0.25
flare_position = 0.4, 0.8
gradient = true
gradient_shader = effects\flare
gradient_texture = fx\fx_gradient
gradient_radius = 0.5
gradient_opacity = 0.6
blend_rise_time = 2.0
blend_down_time = 4.0
"#,
        );
        let mut lib = WeatherLibrary::default();
        lib.parse_suns(&suns_doc);
        let sun = &lib.lens_flares["sun_clear"];
        assert!(sun.has_source && sun.has_flares && sun.has_gradient);
        assert_eq!(sun.flares.len(), 2);
        assert!((sun.flares[1].opacity - 0.25).abs() < 1e-6);

        let env_doc = parse_ltx_str(
            r#"
[environment]
altitude = 25.0
delta_longitude = 30.0
min_dist_factor = 0.99
tilt = 15.0
second_propability = 0.35
sky_color = 0.1
sun_color = 0.5
fog_color = 0.25
"#,
        );
        let gp = EnvGlobalParams::from_ltx(&env_doc);
        assert!((gp.min_dist_factor - 0.95).abs() < 1e-6);
        assert!((gp.second_probability - 0.35).abs() < 1e-6);
    }

    #[test]
    fn test_skybox_clouds_wind_rain_and_vpath_helpers() {
        // 1. Half-box & hemisphere geometry invariants
        let (verts, idx) = skybox_hbox_mesh();
        assert_eq!(verts.len(), SKYBOX_HBOX_VERTEX_COUNT);
        assert_eq!(idx.len(), SKYBOX_HBOX_INDEX_COUNT);
        assert!((verts[0].texcoord[1] + 1.01).abs() < 1e-6);
        assert!(idx.iter().all(|&i| (i as usize) < SKYBOX_HBOX_VERTEX_COUNT));

        assert_eq!(CLOUDS_HEMI_POSITIONS.len(), CLOUDS_HEMI_VERTEX_COUNT);
        assert_eq!(CLOUDS_HEMI_INDICES.len(), CLOUDS_HEMI_INDEX_COUNT);
        assert!(CLOUDS_HEMI_INDICES.iter().all(|&i| (i as usize) < CLOUDS_HEMI_VERTEX_COUNT));
        for p in &CLOUDS_HEMI_POSITIONS {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4);
            assert!(p[1] >= 0.0);
        }

        // 2. Empty sky_texture -> empty sky_texture_env & None DDS VFS path
        let empty_doc = parse_ltx_str("[00:00:00]\nsky_color = 1, 1, 1\n");
        let empty_desc = EnvDescriptor::from_ltx(&empty_doc, "00:00:00").unwrap();
        assert!(empty_desc.sky_texture.is_empty());
        assert!(empty_desc.sky_texture_env.is_empty());

        // 3. MixedEnvironment helpers
        let cycle_doc = parse_ltx_str(
            r#"
[00:00:00]
sky_texture = sky\sky_night
clouds_texture = sky\sky_oblaka
sky_color = 0.2, 0.4, 0.6
sky_rotation = 90
rain_density = 0.5
rain_color = 0.3, 0.3, 0.4
wind_velocity = 500.0
wind_direction = 90.0
sun_color = 1.0, 0.8, 0.6

[12:00:00]
sky_texture = sky\sky_noon
clouds_texture = sky\sky_oblaka_noon
sky_color = 0.6, 0.8, 1.0
sky_rotation = 90
rain_density = 0.5
rain_color = 0.3, 0.3, 0.4
wind_velocity = 500.0
wind_direction = 90.0
sun_color = 1.0, 0.8, 0.6
"#,
        );
        let cycle = WeatherCycle::from_ltx("w_test", &cycle_doc).unwrap();
        let (a, b, w) = cycle.select_at(6.0 * 3600.0).unwrap();
        let mut mixed = lerp_descriptors(
            a,
            b,
            w,
            &EnvModAccum {
                total_power: 0.0,
                active_flags: 0,
                far_plane: 0.0,
                fog_color: [0.0; 3],
                fog_density: 0.0,
                ambient: [0.0; 3],
                sky_color: [0.0; 3],
                hemi_color: [0.0; 3],
            },
            1.0,
            &EnvBoost::default(),
            0.0,
            1.0,
        );
        mixed.sun_dir = [0.0, -0.6, 0.8];

        assert_eq!(mixed.sky_texture_env_a, "sky\\sky_night#small");
        assert_eq!(mixed.sky_texture_env_b, "sky\\sky_noon#small");
        assert_eq!(
            mixed.sky_dds_vpaths(),
            (
                Some("textures\\sky\\sky_night.dds".to_string()),
                Some("textures\\sky\\sky_noon.dds".to_string())
            )
        );
        assert_eq!(
            mixed.sky_env_dds_vpaths(),
            (
                Some("textures\\sky\\sky_night#small.dds".to_string()),
                Some("textures\\sky\\sky_noon#small.dds".to_string())
            )
        );
        assert_eq!(
            mixed.clouds_dds_vpaths(),
            (
                Some("textures\\sky\\sky_oblaka.dds".to_string()),
                Some("textures\\sky\\sky_oblaka_noon.dds".to_string())
            )
        );

        let tint = mixed.sky_tint_rgba();
        assert!((tint[0] - 0.4).abs() < 1e-5);
        assert!((tint[3] - 0.5).abs() < 1e-5);
        assert!((mixed.rain_color_rgba()[3] - 0.75).abs() < 1e-5);
        assert_eq!(mixed.rain_desired_count(RAIN_MAX_DESIRED_ITEMS), 1262);
        assert_eq!(mixed.sun_direction_to_sun_vec4(), [0.0, 0.6, -0.8, 0.0]);
        assert_eq!(mixed.sun_glow_rgba(0.4), [1.0, 0.8, 0.6, 0.4]);

        let cam = [10.0, 20.0, 30.0];
        let sky_m = mixed.sky_world_transform(cam);
        let sky_pt = crate::ogf::matrix_transform_point(&sky_m, [1.0, 0.0, 0.0]);
        assert!((sky_pt[0] - 10.0).abs() < 1e-5);
        assert!((sky_pt[2] - 29.0).abs() < 1e-5);

        let clouds_m = mixed.clouds_world_transform(cam);
        let clouds_pt = crate::ogf::matrix_transform_point(&clouds_m, [1.0, 1.0, 0.0]);
        assert!((clouds_pt[0] - 10.0).abs() < 1e-4);
        assert!((clouds_pt[1] - 20.4).abs() < 1e-4);
        assert!((clouds_pt[2] - 20.0).abs() < 1e-4);

        let (rain_axis, rain_off) = mixed.rain_streak_axis_and_offset();
        assert!(rain_axis[1] < -0.5);
        assert!(rain_off[0].hypot(rain_off[1]) > 1.0);

        let mut wind = WindAnimState::default();
        wind.step(0.5, 400.0, 0.0, 0.0);
        assert!((wind.anim[0] - 0.0).abs() < 1e-5);
        assert!((wind.anim[1] - 0.2).abs() < 1e-5);
        assert!((wind.anim[3] - 0.5).abs() < 1e-5);
    }
}
