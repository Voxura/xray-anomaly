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
use crate::details::{DetailMesh, DetailMeshVertexOut};
use crate::lanims::{unpack_argb_f32, LightAnimItem, LightAnimLibrary};
use crate::ogf::{
    matrix_mul_43, matrix_mul_44, matrix_rotate_y, matrix_set_xyzi_translate,
    matrix_transform_point, matrix_transform_projected, XRayMatrix, XRAY_IDENTITY_MATRIX,
};
use crate::particle_sim::ParticleRng;
use crate::shaders_xr::{env_mod_flags, EnvModAccum, LevelEnvModifiers};
use crate::skinning::{matrix_transform_dir, normalize_vec3};
use crate::thm::texture_stem_to_dds_vpath;

pub const DAY_LENGTH: f32 = 86_400.0;
pub const WFX_TRANS_TIME: f32 = 5.0;
/// Seconds to wet or dry surfaces at multiplier `1.0` (`WETNESS_CHANGE_TIME`, `Environment_misc.cpp:20`).
pub const WETNESS_CHANGE_TIME: f32 = 166.0;
/// Default world camera near clip plane (`VIEWPORT_NEAR`, `device.h:21, 110`).
pub const DEFAULT_VIEWPORT_NEAR: f32 = 0.2;
/// HUD / camera-attached near clip plane (`R_VIEWPORT_NEAR`, `device.h:22`).
pub const R_VIEWPORT_NEAR: f32 = 0.005;
/// Default `[wet_mult, dry_mult]` console multipliers (`ssfx_wetness_multiplier`, `xr_ioc_cmd.cpp:1050`).
pub const DEFAULT_WETNESS_MULTIPLIERS: [f32; 2] = [1.0, 0.3];

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

/// Rain emitter & streak geometry constants (`Rain.cpp:20-35`, `dxRainRender.cpp:7-22`).
pub const RAIN_MAX_DESIRED_ITEMS: u32 = 2500;
pub const RAIN_SOURCE_RADIUS: f32 = 15.0;
pub const RAIN_CPU_SOURCE_RADIUS: f32 = 12.5;
pub const RAIN_SOURCE_OFFSET: f32 = 20.0;
pub const RAIN_MAX_DISTANCE: f32 = RAIN_SOURCE_OFFSET * 1.5;
pub const RAIN_SINK_OFFSET: f32 = -(RAIN_MAX_DISTANCE - RAIN_SOURCE_OFFSET);
pub const RAIN_DROP_LENGTH: f32 = 5.0;
pub const RAIN_DROP_WIDTH: f32 = 0.30;
pub const RAIN_DROP_ANGLE_RAD: f32 = 15.0 * (std::f32::consts::PI / 180.0);
pub const RAIN_DROP_MAX_ANGLE_RAD: f32 = 35.0 * (std::f32::consts::PI / 180.0);
pub const RAIN_DROP_SPEED_MIN: f32 = 40.0;
pub const RAIN_DROP_SPEED_MAX: f32 = 80.0;
pub const RAIN_MAX_PARTICLES: usize = 1000;
pub const RAIN_SPLASH_TIME_SEC: f32 = 0.3;

/// Standard 6-index pattern emitted per quad (`CBackend::CreateQuadIB`, `R_Backend.cpp:42-48`).
pub const QUAD_INDEX_PATTERN: [u16; 6] = [0, 1, 2, 3, 2, 1];

/// Append one quad's 6 `QuadIB` indices anchored at `vertex_base`.
#[inline]
pub fn append_quad_indices(indices: &mut Vec<u16>, vertex_base: u16) {
    indices.extend(QUAD_INDEX_PATTERN.map(|o| vertex_base + o));
}

/// Shared lit billboard vertex (`FVF::F_LIT`: position, linear RGBA color, UV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvLitVertex {
    pub pos: [f32; 3],
    pub color_rgba: [f32; 4],
    pub uv: [f32; 2],
}

/// Alternating UV patterns used by `dxRainRender::Render` streaks (`dxRainRender.cpp:213-216`).
pub const RAIN_STREAK_UVS: [[[f32; 2]; 4]; 2] = [
    [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]],
    [[1.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
];

/// Unpacked `v_clouds` dome vertex (`dxEnvironmentRender.cpp:79-91`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudsDomeVertex {
    pub pos: [f32; 3],
    pub wind_rgba: [f32; 4],
    pub color_rgba: [f32; 4],
}

/// Construct the 91-vertex / 480-index clouds dome mesh tinted for `env`
/// (`dxEnvironmentRender.cpp:374-387`).
pub fn clouds_hemi_mesh(
    env: &MixedEnvironment,
) -> (
    [CloudsDomeVertex; CLOUDS_HEMI_VERTEX_COUNT],
    [u16; CLOUDS_HEMI_INDEX_COUNT],
) {
    let wind_rgba = env.clouds_wind_rgba();
    let color_rgba = env.clouds_color_rgba();
    let mut verts = [CloudsDomeVertex {
        pos: [0.0; 3],
        wind_rgba: [0.0; 4],
        color_rgba: [0.0; 4],
    }; CLOUDS_HEMI_VERTEX_COUNT];
    for (slot, pos) in verts.iter_mut().zip(CLOUDS_HEMI_POSITIONS.iter()) {
        slot.pos = *pos;
        slot.wind_rgba = wind_rgba;
        slot.color_rgba = color_rgba;
    }
    (verts, CLOUDS_HEMI_INDICES)
}

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

    /// Dual-layer scrolling UV offsets `((u0, v0), (u1, v1))` derived from elapsed `anim[3]`
    /// along X-Ray's cloud wind basis (`PI_DIV_4` & `3*PI_DIV_8`, `dxEnvironmentRender.cpp:370-371`).
    #[inline]
    pub fn clouds_uv_scroll(&self, layer0_speed: f32, layer1_speed: f32) -> ([f32; 2], [f32; 2]) {
        let wd0 = dir_from_hp(std::f32::consts::FRAC_PI_4, 0.0);
        let wd1 = dir_from_hp(
            std::f32::consts::FRAC_PI_4 + std::f32::consts::FRAC_PI_8,
            0.0,
        );
        let t = self.anim[3];
        (
            [wd0[0] * t * layer0_speed, wd0[2] * t * layer0_speed],
            [wd1[0] * t * layer1_speed, wd1[2] * t * layer1_speed],
        )
    }
}

// ============================================================================
// 1D Perlin Noise (`src/xrEngine/perlin.cpp`)
// ============================================================================

const PERLIN_1D_SAMPLE_SIZE: usize = 256;

/// Classic 1D Perlin noise matching `CPerlinNoise1D` (`perlin.h`, `perlin.cpp:29-111`).
#[derive(Debug, Clone)]
pub struct PerlinNoise1D {
    octaves: usize,
    frequency: f32,
    amplitude: f32,
    perm: Vec<usize>,
    g1: Vec<f32>,
    prev_continuous_time: f32,
    times: Vec<f32>,
}

impl PerlinNoise1D {
    pub fn new(seed: u32, octaves: usize, frequency: f32, amplitude: f32) -> Self {
        let b = PERLIN_1D_SAMPLE_SIZE;
        let mut perm = vec![0usize; b + b + 2];
        let mut g1 = vec![0.0f32; b + b + 2];

        let mut lcg_state: u32 = seed;
        let mut msvcrt_rand = || -> usize {
            lcg_state = lcg_state.wrapping_mul(214_013).wrapping_add(2_531_011);
            ((lcg_state >> 16) & 0x7FFF) as usize
        };

        for i in 0..b {
            perm[i] = i;
            let r = (msvcrt_rand() % (b + b)) as i32 - (b as i32);
            g1[i] = (r as f32) / (b as f32);
        }
        for i in (1..b).rev() {
            let j = msvcrt_rand() % b;
            perm.swap(i, j);
        }
        for i in 0..(b + 2) {
            perm[b + i] = perm[i];
            g1[b + i] = g1[i];
        }

        Self {
            octaves,
            frequency,
            amplitude,
            perm,
            g1,
            prev_continuous_time: 0.0,
            times: vec![0.0; octaves],
        }
    }

    pub fn set_params(&mut self, octaves: usize, frequency: f32, amplitude: f32) {
        self.octaves = octaves;
        self.frequency = frequency;
        self.amplitude = amplitude;
        self.times.resize(octaves, 0.0);
    }

    fn noise1d(&self, arg: f32) -> f32 {
        let t = (arg as f64) + 4096.0;
        let t_floor = t.floor();
        let b0 = (t_floor as i64).rem_euclid(PERLIN_1D_SAMPLE_SIZE as i64) as usize;
        let b1 = (b0 + 1) & (PERLIN_1D_SAMPLE_SIZE - 1);
        let rx0 = (t - t_floor) as f32;
        let rx1 = rx0 - 1.0;
        let sx = rx0 * rx0 * (3.0 - 2.0 * rx0);
        let u = rx0 * self.g1[self.perm[b0]];
        let v = rx1 * self.g1[self.perm[b1]];
        u + sx * (v - u)
    }

    /// Evaluate multi-octave 1D noise at absolute coordinate `v` (`CPerlinNoise1D::Get`).
    pub fn get(&self, v: f32) -> f32 {
        if !v.is_finite() || self.octaves == 0 {
            return 0.0;
        }
        let mut result = 0.0;
        let mut amp = self.amplitude;
        let mut coord = v * self.frequency;
        for _ in 0..self.octaves {
            result += self.noise1d(coord) * amp;
            coord *= 2.0;
            amp *= 0.5;
        }
        result
    }

    /// Evaluate smooth delta-accumulated 1D noise (`CPerlinNoise1D::GetContinious`).
    pub fn get_continuous(&mut self, v: f32) -> f32 {
        if !v.is_finite() || self.octaves == 0 {
            return 0.0;
        }
        let mut dv = v;
        if self.prev_continuous_time != 0.0 {
            dv -= self.prev_continuous_time;
        }
        self.prev_continuous_time = v;

        let mut result = 0.0;
        let mut amp = self.amplitude;
        dv *= self.frequency;
        for i in 0..self.octaves {
            let next_t = self.times[i] + dv;
            self.times[i] = next_t;
            result += self.noise1d(next_t) * amp;
            dv *= 2.0;
            amp *= 0.5;
        }
        result
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
            let key_alt = format!("{hour}:00:00");
            let active_key = if doc.sections.contains_key(&key) {
                key
            } else if doc.sections.contains_key(&key_alt) {
                key_alt
            } else {
                return Err(FormatError::InvalidValue("missing hourly sun_positions section"));
            };
            let alt = doc.read_f32(&active_key, "sun_altitude", 0.0);
            let lon = doc.read_f32(&active_key, "sun_longitude", 0.0);
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

/// Parse `"HH:MM:SS"` or `"HH:MM"` keyframe section names into seconds (`0..=86400`).
pub fn parse_hhmmss_seconds(ident: &str) -> Result<f32, FormatError> {
    let trimmed = ident.trim();
    if trimmed == "24:00:00" || trimmed == "24:00" {
        return Ok(DAY_LENGTH);
    }
    let parts: Vec<&str> = trimmed.split(':').map(str::trim).collect();
    if parts.len() < 2 || parts.len() > 3 {
        return Err(FormatError::InvalidValue("invalid weather time format"));
    }
    let h = parts[0]
        .parse::<u32>()
        .map_err(|_| FormatError::InvalidValue("invalid weather time hour"))?;
    let m = parts[1]
        .parse::<u32>()
        .map_err(|_| FormatError::InvalidValue("invalid weather time minute"))?;
    let s = if parts.len() == 3 {
        parts[2]
            .parse::<u32>()
            .map_err(|_| FormatError::InvalidValue("invalid weather time second"))?
    } else {
        0
    };
    if h == 24 && m == 0 && s == 0 {
        return Ok(DAY_LENGTH);
    }
    if h >= 24 || m >= 60 || s >= 60 {
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
        let mut min_distance = doc.read_f32(section, "min_distance", 1.0).max(0.0);
        let mut max_distance = doc.read_f32(section, "max_distance", min_distance + 1.0).max(0.0);
        if max_distance < min_distance {
            std::mem::swap(&mut min_distance, &mut max_distance);
        }
        if (max_distance - min_distance).abs() < 1e-3 {
            max_distance = min_distance + 1.0;
        }
        let mut p0 = doc.read_i32(section, "period0", 1000).max(0);
        let mut p1 = doc.read_i32(section, "period1", 5000).max(0);
        let mut p2 = doc.read_i32(section, "period2", 1000).max(0);
        let mut p3 = doc.read_i32(section, "period3", 5000).max(0);
        if p0 > p1 {
            std::mem::swap(&mut p0, &mut p1);
        }
        if p2 > p3 {
            std::mem::swap(&mut p2, &mut p3);
        }
        let period = [p0, p1, p2, p3];
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

/// Persistent surface wetness accumulator matching `CEnvironment::wetness_factor`
/// (`src/xrEngine/Environment_misc.cpp:526-531`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WetnessState {
    pub factor: f32,
}

impl WetnessState {
    pub fn new(initial_factor: f32) -> Self {
        Self {
            factor: initial_factor.clamp(0.0, 1.0),
        }
    }

    /// Advance surface wetness by `dt_sec` using current `rain_density` and
    /// `[wet_mult, dry_mult]` (`DEFAULT_WETNESS_MULTIPLIERS`).
    pub fn step(&mut self, dt_sec: f32, rain_density: f32, wet_mult: f32, dry_mult: f32) -> f32 {
        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        if rain_density > 0.0 {
            self.factor += rain_density * wet_mult * dt / WETNESS_CHANGE_TIME;
        } else {
            self.factor -= dry_mult * dt / WETNESS_CHANGE_TIME;
        }
        self.factor = self.factor.clamp(0.0, 1.0);
        self.factor
    }
}

/// Packed 368-byte standard per-frame global shader constants matching
/// `Blender_Recorder_StandartBinding.cpp:107-324, 506-760`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardFrameUniforms {
    /// Linear fog view-plane vector (`dot(w_pos, fog_plane)` in `common_functions.h:54`).
    pub fog_plane: [f32; 4],
    /// `(-fog_near * r, fog_near, fog_far, r)` where `r = 1 / (fog_far - fog_near)`.
    pub fog_params: [f32; 4],
    /// `(fog_color.r, fog_color.g, fog_color.b, fog_density)`.
    pub fog_color: [f32; 4],
    /// `(wind_direction_rad, wind_velocity, 0, 0)`.
    pub wind_params: [f32; 4],
    /// `(time_global, time_global * 10, time_global / 10, sin(time_global))`.
    pub timers: [f32; 4],
    /// `(game_time, game_time / DAY_LENGTH, game_time / 3600, floor(game_time / 3600))`.
    pub timers_game: [f32; 4],
    /// `(width, height, 1 / width, 1 / height)`.
    pub screen_res: [f32; 4],
    /// `(fov_deg, aspect, tan(deg2rad(fov_deg) / 2), far_plane * 0.75)`.
    pub ogse_c_screen: [f32; 4],
    /// `(viewport_near, far_plane, 0, 0)`.
    pub near_far_plane: [f32; 4],
    /// `(rain_density, wetness_factor, 0, 0)`.
    pub rain_params: [f32; 4],
    /// `(cam_pos.xyz, 1.0)`.
    pub eye_position: [f32; 4],
    /// `(cam_dir.xyz, 0.0)`.
    pub eye_direction: [f32; 4],
    /// `(cam_top.xyz, 0.0)`.
    pub eye_normal: [f32; 4],
    /// `(sun_color.rgb, 0.0)`.
    pub l_sun_color: [f32; 4],
    /// `(sun_dir_world.xyz, 0.0)`.
    pub l_sun_dir_w: [f32; 4],
    /// `(normalize(mul_dir(mView, sun_dir)).xyz, 0.0)`.
    pub l_sun_dir_e: [f32; 4],
    /// `hemi_color` RGBA (`CEnvDescriptor::hemi_color`).
    pub l_hemi_color: [f32; 4],
    /// `(ambient.rgb, mixer_weight)`.
    pub l_ambient: [f32; 4],
    /// `(sky_color.rgb, sky_rotation)`.
    pub sky_color: [f32; 4],
    /// DX11 texel-adjusted view-projection texgen (`mVPTexgen`).
    pub m_vp_texgen: XRayMatrix,
}

const _: () = assert!(std::mem::size_of::<StandardFrameUniforms>() == 368);

impl StandardFrameUniforms {
    /// Compute all standard per-frame global bindings from the active [`MixedEnvironment`],
    /// camera basis/matrices, frame resolution, and global/game clocks.
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        env: &MixedEnvironment,
        view: &XRayMatrix,
        full_transform: &XRayMatrix,
        camera_pos: [f32; 3],
        camera_dir: [f32; 3],
        camera_top: [f32; 3],
        fov_deg: f32,
        aspect: f32,
        viewport_near: f32,
        screen_w: u32,
        screen_h: u32,
        time_global_sec: f32,
        game_time_sec: f32,
        wetness_factor: f32,
    ) -> Self {
        let px = -(full_transform[0][3] + full_transform[0][2]);
        let py = -(full_transform[1][3] + full_transform[1][2]);
        let pz = -(full_transform[2][3] + full_transform[2][2]);
        let pw = -(full_transform[3][3] + full_transform[3][2]);
        let len = (px * px + py * py + pz * pz).sqrt();
        let plane = if len > 1e-12 && len.is_finite() {
            let denom = -1.0 / len;
            [px * denom, py * denom, pz * denom, pw * denom]
        } else {
            [0.0, 0.0, 1.0, 0.0]
        };

        let fog_span = env.fog_far - env.fog_near;
        let b = if fog_span.abs() > 1e-6 {
            1.0 / fog_span
        } else {
            0.0
        };
        let fog_plane = [
            -plane[0] * b,
            -plane[1] * b,
            -plane[2] * b,
            1.0 - (plane[3] - env.fog_near) * b,
        ];
        let fog_params = [-env.fog_near * b, env.fog_near, env.fog_far, b];

        let w = screen_w.max(1) as f32;
        let h = screen_h.max(1) as f32;
        let hour_len = DAY_LENGTH / 24.0;

        let sun_dir_view_raw = matrix_transform_dir(view, env.sun_dir);
        let sun_dir_view = normalize_vec3(sun_dir_view_raw);
        let sun_dir_e = if sun_dir_view[0] * sun_dir_view[0]
            + sun_dir_view[1] * sun_dir_view[1]
            + sun_dir_view[2] * sun_dir_view[2]
            > 1e-12
        {
            sun_dir_view
        } else {
            [0.0, -1.0, 0.0]
        };

        let m_texel_adjust: XRayMatrix = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0, 1.0],
        ];
        let m_vp_texgen = matrix_mul_44(&m_texel_adjust, full_transform);

        Self {
            fog_plane,
            fog_params,
            fog_color: [
                env.fog_color[0],
                env.fog_color[1],
                env.fog_color[2],
                env.fog_density,
            ],
            wind_params: [env.wind_direction, env.wind_velocity, 0.0, 0.0],
            timers: [
                time_global_sec,
                time_global_sec * 10.0,
                time_global_sec / 10.0,
                time_global_sec.sin(),
            ],
            timers_game: [
                game_time_sec,
                game_time_sec / DAY_LENGTH,
                game_time_sec / hour_len,
                (game_time_sec / hour_len).floor(),
            ],
            screen_res: [w, h, 1.0 / w, 1.0 / h],
            ogse_c_screen: [
                fov_deg,
                aspect,
                (fov_deg.to_radians() * 0.5).tan(),
                env.far_plane * 0.75,
            ],
            near_far_plane: [viewport_near, env.far_plane, 0.0, 0.0],
            rain_params: [env.rain_density, wetness_factor, 0.0, 0.0],
            eye_position: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
            eye_direction: [camera_dir[0], camera_dir[1], camera_dir[2], 0.0],
            eye_normal: [camera_top[0], camera_top[1], camera_top[2], 0.0],
            l_sun_color: [env.sun_color[0], env.sun_color[1], env.sun_color[2], 0.0],
            l_sun_dir_w: [env.sun_dir[0], env.sun_dir[1], env.sun_dir[2], 0.0],
            l_sun_dir_e: [sun_dir_e[0], sun_dir_e[1], sun_dir_e[2], 0.0],
            l_hemi_color: env.hemi_color,
            l_ambient: [
                env.ambient_color[0],
                env.ambient_color[1],
                env.ambient_color[2],
                env.weight,
            ],
            sky_color: [
                env.sky_color[0],
                env.sky_color[1],
                env.sky_color[2],
                env.sky_rotation,
            ],
            m_vp_texgen,
        }
    }
}

/// Default interpolation factor for `eye_position_lerp` / `eye_direction_lerp`
/// (`scope_fog_interp` in `src/xrEngine/xr_ioc_cmd.cpp:1036`).
pub const DEFAULT_SCOPE_FOG_INTERP: f32 = 0.15;

/// Persistent interpolated eye pose state (`cl_eye_PL` / `cl_eye_DL`,
/// `Blender_Recorder_StandartBinding.cpp:275-311`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScopeEyeLerpState {
    initialized: bool,
    pub eye_position_lerp: [f32; 3],
    pub eye_direction_lerp: [f32; 3],
}

impl ScopeEyeLerpState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_initial(pos: [f32; 3], dir: [f32; 3]) -> Self {
        Self {
            initialized: true,
            eye_position_lerp: pos,
            eye_direction_lerp: dir,
        }
    }

    /// Advance smoothed scope eye `(position, direction)` using `interp` (`tV.lerp(tV, V, interp)`)
    /// and return `(eye_position_lerp_vec4, eye_direction_lerp_vec4)`.
    ///
    /// Note: snaps to the initial camera pose on the very first call instead of lerping from world origin.
    pub fn step(
        &mut self,
        cam_pos: [f32; 3],
        cam_dir: [f32; 3],
        interp: f32,
    ) -> ([f32; 4], [f32; 4]) {
        if !self.initialized {
            self.eye_position_lerp = cam_pos;
            self.eye_direction_lerp = cam_dir;
            self.initialized = true;
        } else {
            let t = interp.clamp(0.0, 1.0);
            for i in 0..3 {
                self.eye_position_lerp[i] =
                    self.eye_position_lerp[i] * (1.0 - t) + cam_pos[i] * t;
                self.eye_direction_lerp[i] =
                    self.eye_direction_lerp[i] * (1.0 - t) + cam_dir[i] * t;
            }
        }
        (
            [
                self.eye_position_lerp[0],
                self.eye_position_lerp[1],
                self.eye_position_lerp[2],
                1.0,
            ],
            [
                self.eye_direction_lerp[0],
                self.eye_direction_lerp[1],
                self.eye_direction_lerp[2],
                0.0,
            ],
        )
    }
}

/// Console defaults for Crookr fake-scope optics (`src/xrEngine/xr_ioc_cmd.cpp:1036-1048`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FakeScopeConfig {
    pub fog_interp: f32,
    pub fog_travel: f32,
    pub fog_attack: f32,
    pub fog_mattack: f32,
    pub ca: f32,
    pub outer_blur: f32,
    pub inner_blur: f32,
    pub scroll_power: f32,
    pub brightness: f32,
    pub radius: f32,
    pub fog_radius: f32,
    pub fog_sharp: f32,
}

impl Default for FakeScopeConfig {
    fn default() -> Self {
        Self {
            fog_interp: DEFAULT_SCOPE_FOG_INTERP,
            fog_travel: 0.25,
            fog_attack: 0.66,
            fog_mattack: 0.25,
            ca: 0.003,
            outer_blur: 1.0,
            inner_blur: 0.1,
            scroll_power: 0.66,
            brightness: 1.0,
            radius: 0.0,
            fog_radius: 1.25,
            fog_sharp: 4.0,
        }
    }
}

impl FakeScopeConfig {
    #[inline]
    pub fn to_uniforms(&self) -> FakeScopeUniforms {
        FakeScopeUniforms::from(self)
    }
}

/// 48-byte packed Crookr fake-scope constant block (`cl_fakescope_params`, `cl_fakescope_ca`,
/// `cl_fakescope_params3` in `Blender_Recorder_StandartBinding.cpp:326-364`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FakeScopeUniforms {
    /// `[scroll_power, inner_blur, outer_blur, brightness]`.
    pub fakescope_params: [f32; 4],
    /// `[ca, fog_attack, fog_mattack, fog_travel]`.
    pub fakescope_ca: [f32; 4],
    /// `[radius, fog_radius, fog_sharp, 0.0]`.
    pub fakescope_params3: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<FakeScopeUniforms>() == 48);

impl Default for FakeScopeUniforms {
    fn default() -> Self {
        FakeScopeConfig::default().to_uniforms()
    }
}

impl From<&FakeScopeConfig> for FakeScopeUniforms {
    fn from(cfg: &FakeScopeConfig) -> Self {
        Self {
            fakescope_params: [
                cfg.scroll_power,
                cfg.inner_blur,
                cfg.outer_blur,
                cfg.brightness,
            ],
            fakescope_ca: [cfg.ca, cfg.fog_attack, cfg.fog_mattack, cfg.fog_travel],
            fakescope_params3: [cfg.radius, cfg.fog_radius, cfg.fog_sharp, 0.0],
        }
    }
}

/// 128-byte 3D shader scope, reticle mark-switch, and reflex-sight uniforms
/// (`Blender_Recorder_StandartBinding.cpp:366-442`, `xrRender_console.cpp:388-399`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct S3dsScopeUniforms {
    pub markswitch_current: i32,
    pub markswitch_count: i32,
    pub _pad: [i32; 2],
    pub markswitch_color: [f32; 4],
    pub s3ds_param_1: [f32; 4],
    pub s3ds_param_2: [f32; 4],
    pub s3ds_param_3: [f32; 4],
    pub s3ds_param_4: [f32; 4],
    pub reflex_param: [f32; 4],
    pub _tail_pad: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<S3dsScopeUniforms>() == 128);

impl Default for S3dsScopeUniforms {
    fn default() -> Self {
        Self {
            markswitch_current: 0,
            markswitch_count: 0,
            _pad: [0; 2],
            markswitch_color: [0.0; 4],
            s3ds_param_1: [0.0; 4],
            s3ds_param_2: [0.0; 4],
            s3ds_param_3: [0.0; 4],
            s3ds_param_4: [0.0; 4],
            reflex_param: [0.0; 4],
            _tail_pad: [0.0; 4],
        }
    }
}

impl S3dsScopeUniforms {
    #[inline]
    pub fn markswitch_current_vec4(&self) -> [f32; 4] {
        [self.markswitch_current as f32, 0.0, 0.0, 0.0]
    }

    #[inline]
    pub fn markswitch_count_vec4(&self) -> [f32; 4] {
        [self.markswitch_count as f32, 0.0, 0.0, 0.0]
    }
}

/// Actor status shader binding (`IGame_Persistent::act_dat` / `cl_actor_params`,
/// `Blender_Recorder_StandartBinding.cpp:692-706`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ActorShaderData {
    pub health: f32,
    pub stamina: f32,
    pub bleeding: f32,
    pub helmet: bool,
}

impl ActorShaderData {
    #[inline]
    pub fn to_vec4(&self) -> [f32; 4] {
        [
            self.health,
            self.stamina,
            self.bleeding,
            if self.helmet { 1.0 } else { 0.0 },
        ]
    }
}

/// PDA display shader binding (`IGame_Persistent::pda_data` / `cl_pda_params`,
/// `Blender_Recorder_StandartBinding.cpp:734-747`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PdaShaderParams {
    pub display_factor: f32,
    pub psy_influence: f32,
    pub display_brightness: f32,
}

impl PdaShaderParams {
    #[inline]
    pub fn to_vec4(&self) -> [f32; 4] {
        [
            self.display_factor,
            self.psy_influence,
            self.display_brightness,
            0.0,
        ]
    }
}

/// Global engine shader constants (`CGShaderConstants` / `g_pGamePersistent->m_pGShaderConstants`,
/// `Blender_Recorder_StandartBinding.cpp:646-676`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GlobalShaderConstants {
    pub hud_params: [f32; 4],
    pub hud_fov_params: [f32; 4],
    pub m_script_params: [f32; 4],
    pub m_blender_mode: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GlobalShaderConstants>() == 64);

/// Developer shader tweak constants `shader_param_1..8` (`ps_dev_param_1..8`,
/// `Blender_Recorder_StandartBinding.cpp:810-881`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DevDebugParams {
    pub params: [[f32; 4]; 8],
}

const _: () = assert!(std::mem::size_of::<DevDebugParams>() == 128);

/// Remaining SSFX surface, flora, wind, POM, HUD drop & timing console settings
/// (`xrRender_console.cpp:406-475`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxSurfaceConfig {
    pub floravariation: [f32; 4],
    pub pom: [f32; 4],
    pub terrain_pom: [f32; 4],
    pub terrain_offset: [f32; 4],
    pub wind_grass: [f32; 4],
    pub wind_trees: [f32; 4],
    pub florafixes_1: [f32; 4],
    pub florafixes_2: [f32; 4],
    pub wetsurfaces_1: [f32; 4],
    pub wetsurfaces_2: [f32; 4],
    pub is_underground: bool,
    pub gloss_factor: f32,
    pub gloss_minmax: [f32; 3],
    pub lightsetup_1: [f32; 4],
    pub hud_drops_1: [f32; 4],
    pub hud_drops_2: [f32; 4],
    pub hud_hemi: f32,
    pub blood_decals: [f32; 4],
}

impl Default for SsfxSurfaceConfig {
    fn default() -> Self {
        Self {
            floravariation: [0.025, 0.1, 0.025, 0.05],
            pom: [16.0, 12.0, 0.035, 0.4],
            terrain_pom: [12.0, 20.0, 0.04, 1.0],
            terrain_offset: [0.0; 4],
            wind_grass: [9.5, 1.4, 1.5, 0.4],
            wind_trees: [11.0, 0.15, 0.5, 0.15],
            florafixes_1: [0.3, 0.21, 0.3, 0.21],
            florafixes_2: [2.0, 1.0, 0.0, 0.0],
            wetsurfaces_1: [1.0; 4],
            wetsurfaces_2: [1.0; 4],
            is_underground: false,
            gloss_factor: 0.5,
            gloss_minmax: [0.0, 0.92, 0.0],
            lightsetup_1: [0.35, 0.5, 0.0, 0.0],
            hud_drops_1: [1.0; 4],
            hud_drops_2: [1.5, 0.85, 0.0, 2.0],
            hud_hemi: 0.15,
            blood_decals: [0.6, 0.6, 0.0, 0.0],
        }
    }
}

/// 304-byte packed GPU uniform block for SSFX surface/flora/HUD/timing constants
/// (`Blender_Recorder_StandartBinding.cpp:899-1227`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SsfxSurfaceUniforms {
    pub ssfx_gloss: [f32; 4],
    pub ssfx_wetsurfaces_1: [f32; 4],
    pub ssfx_wetsurfaces_2: [f32; 4],
    pub ssfx_floravariation: [f32; 4],
    pub ssfx_florafixes_1: [f32; 4],
    pub ssfx_florafixes_2: [f32; 4],
    pub ssfx_wsetup_grass: [f32; 4],
    pub ssfx_wsetup_trees: [f32; 4],
    pub ssfx_pom: [f32; 4],
    pub ssfx_terrain_pom: [f32; 4],
    pub ssfx_terrain_offset: [f32; 4],
    pub ssfx_lightsetup_1: [f32; 4],
    pub ssfx_hud_drops_1: [f32; 4],
    pub ssfx_hud_drops_2: [f32; 4],
    pub ssfx_hud_hemi: [f32; 4],
    pub ssfx_blood_decals: [f32; 4],
    pub ssfx_is_underground: [f32; 4],
    pub ssfx_issvp: [f32; 4],
    pub ssfx_timedelta: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SsfxSurfaceUniforms>() == 304);

impl SsfxSurfaceConfig {
    #[inline]
    pub fn to_gpu_uniforms(
        &self,
        is_svp_frame: bool,
        time_delta_sec: f32,
        frame_index: u32,
    ) -> SsfxSurfaceUniforms {
        SsfxSurfaceUniforms {
            ssfx_gloss: [
                self.gloss_minmax[0],
                self.gloss_minmax[1],
                self.gloss_factor,
                0.0,
            ],
            ssfx_wetsurfaces_1: self.wetsurfaces_1,
            ssfx_wetsurfaces_2: self.wetsurfaces_2,
            ssfx_floravariation: self.floravariation,
            ssfx_florafixes_1: self.florafixes_1,
            ssfx_florafixes_2: self.florafixes_2,
            ssfx_wsetup_grass: self.wind_grass,
            ssfx_wsetup_trees: self.wind_trees,
            ssfx_pom: self.pom,
            ssfx_terrain_pom: self.terrain_pom,
            ssfx_terrain_offset: self.terrain_offset,
            ssfx_lightsetup_1: self.lightsetup_1,
            ssfx_hud_drops_1: self.hud_drops_1,
            ssfx_hud_drops_2: self.hud_drops_2,
            ssfx_hud_hemi: [self.hud_hemi, 0.0, 0.0, 0.0],
            ssfx_blood_decals: self.blood_decals,
            ssfx_is_underground: [if self.is_underground { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            ssfx_issvp: [if is_svp_frame { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
            ssfx_timedelta: [time_delta_sec, frame_index as f32, 0.0, 0.0],
        }
    }
}

/// Tracks active `$user$cmap0` / `$user$cmap1` color-grading texture names and detects changes
/// matching `ColorMapManager` (`ColorMapManager.cpp:13-51`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColorMapCacheState {
    slots: [String; 2],
}

impl ColorMapCacheState {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn slot_names(&self) -> [&str; 2] {
        [&self.slots[0], &self.slots[1]]
    }

    /// Update a single `$user$cmap{0,1}` slot, returning `true` when the bound texture changed
    /// (`ColorMapManager::UpdateTexture`, `ColorMapManager.cpp:20-51`).
    pub fn update_slot(&mut self, index: usize, tex_name: &str) -> bool {
        if index >= 2 || self.slots[index] == tex_name {
            return false;
        }
        self.slots[index] = tex_name.to_string();
        true
    }

    /// Update both color-map slots together (`ColorMapManager::SetTextures`), returning per-slot
    /// dirty flags `[cmap0_changed, cmap1_changed]`.
    #[inline]
    pub fn set_textures(&mut self, tex0: &str, tex1: &str) -> [bool; 2] {
        [self.update_slot(0, tex0), self.update_slot(1, tex1)]
    }
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
            if !sec.contains(':') {
                continue;
            }
            match EnvDescriptor::from_ltx(doc, sec) {
                Ok(desc) => descriptors.push(desc),
                Err(err) => {
                    eprintln!("[weather] skipping invalid descriptor section '{sec}' in '{name}': {err}");
                }
            }
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
                match EnvSoundChannelDef::from_ltx(doc, sec) {
                    Ok(channel) => {
                        self.sound_channels.insert(sec.clone(), channel);
                    }
                    Err(err) => {
                        eprintln!("[weather] ignoring invalid sound channel '{sec}': {err}");
                    }
                }
            }
        }
        Ok(())
    }

    pub fn parse_ambient_effects(&mut self, doc: &OwnedLtxDocument) -> Result<(), FormatError> {
        for sec in doc.sections.keys() {
            if doc.read_str(sec, "particles").is_some() {
                match EnvAmbientEffectDef::from_ltx(doc, sec) {
                    Ok(effect) => {
                        self.ambient_effects.insert(sec.clone(), effect);
                    }
                    Err(err) => {
                        eprintln!("[weather] ignoring invalid ambient effect '{sec}': {err}");
                    }
                }
            }
        }
        Ok(())
    }

    pub fn parse_ambients(&mut self, doc: &OwnedLtxDocument) -> Result<(), FormatError> {
        for sec in doc.sections.keys() {
            if doc.read_str(sec, "sound_channels").is_some()
                || doc.read_str(sec, "effects").is_some()
            {
                match EnvAmbientDef::from_ltx(doc, sec) {
                    Ok(ambient) => {
                        self.ambients.insert(sec.clone(), ambient);
                    }
                    Err(err) => {
                        eprintln!("[weather] ignoring invalid ambient '{sec}': {err}");
                    }
                }
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

// ============================================================================
// Pure-Rust Runtime Simulators: Rain, Thunderbolt & LensFlare
// ============================================================================

#[inline]
fn rng_range(rng: &mut ParticleRng, lo: f32, hi: f32) -> f32 {
    lo + (hi - lo) * rng.unit_f32()
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn random_dir_cone(rng: &mut ParticleRng, cone_axis: [f32; 3], cone_angle: f32) -> [f32; 3] {
    let cos_theta = (rng.unit_f32() * std::f32::consts::PI).cos();
    let phi = rng.unit_f32() * std::f32::consts::TAU;
    let r = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let (sa, ca) = phi.sin_cos();
    let rnd = [r * ca, r * sa, cos_theta];
    let scale = rng.unit_f32() * cone_angle.tan();
    normalize_vec3([
        cone_axis[0] + rnd[0] * scale,
        cone_axis[1] + rnd[1] * scale,
        cone_axis[2] + rnd[2] * scale,
    ])
}

/// Extract X-Ray yaw/heading `h` and pitch `p` (radians) from direction vector (`_vector3::getHP`).
#[inline]
pub fn hp_from_dir(v: [f32; 3]) -> (f32, f32) {
    let eps = 1e-5_f32;
    let x = v[0];
    let y = v[1];
    let z = v[2];
    if x.abs() < eps && z.abs() < eps {
        let p = if y.abs() >= eps {
            if y > 0.0 {
                std::f32::consts::FRAC_PI_2
            } else {
                -std::f32::consts::FRAC_PI_2
            }
        } else {
            0.0
        };
        (0.0, p)
    } else {
        let h = if z.abs() < eps {
            if x > 0.0 {
                -std::f32::consts::FRAC_PI_2
            } else {
                std::f32::consts::FRAC_PI_2
            }
        } else if z < 0.0 {
            -(x / z).atan() + std::f32::consts::PI
        } else {
            -(x / z).atan()
        };
        let hyp = (x * x + z * z).sqrt();
        let p = if hyp.abs() < eps {
            if y > 0.0 {
                std::f32::consts::FRAC_PI_2
            } else {
                -std::f32::consts::FRAC_PI_2
            }
        } else {
            (y / hyp).atan()
        };
        (h, p)
    }
}

/// Configuration for `RainEffectState` (`Rain.cpp` / `dxRainRender.cpp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RainEffectConfig {
    pub max_particles: u32,
    pub source_radius: f32,
    pub source_offset: f32,
    pub max_distance: f32,
    pub sink_offset: f32,
    pub drop_length: f32,
    pub drop_width: f32,
    pub drop_speed_scale: f32,
    pub drop_angle_rad: f32,
    pub drop_max_angle_rad: f32,
    pub drop_speed_min: f32,
    pub drop_speed_max: f32,
    pub max_splashes: usize,
    pub splash_lifetime_sec: f32,
}

impl Default for RainEffectConfig {
    fn default() -> Self {
        Self {
            max_particles: RAIN_MAX_DESIRED_ITEMS,
            source_radius: RAIN_SOURCE_RADIUS,
            source_offset: RAIN_SOURCE_OFFSET,
            max_distance: RAIN_MAX_DISTANCE,
            sink_offset: RAIN_SINK_OFFSET,
            drop_length: RAIN_DROP_LENGTH,
            drop_width: RAIN_DROP_WIDTH,
            drop_speed_scale: 1.0,
            drop_angle_rad: RAIN_DROP_ANGLE_RAD,
            drop_max_angle_rad: RAIN_DROP_MAX_ANGLE_RAD,
            drop_speed_min: RAIN_DROP_SPEED_MIN,
            drop_speed_max: RAIN_DROP_SPEED_MAX,
            max_splashes: RAIN_MAX_PARTICLES,
            splash_lifetime_sec: RAIN_SPLASH_TIME_SEC,
        }
    }
}

/// Single active rain streak particle (`CEffect_Rain::Item`, `Rain.h:25-39`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RainDropItem {
    pub pos: [f32; 3],
    pub pos_hit: [f32; 3],
    pub dir: [f32; 3],
    pub speed: f32,
    pub life_remaining_sec: f32,
    pub hit_remaining_sec: f32,
    pub hit_pending: bool,
    pub uv_set: u8,
}

impl RainDropItem {
    #[inline]
    pub fn invalidate(&mut self) {
        self.life_remaining_sec = -1.0;
    }
}

/// Active rain impact splash state (`CEffect_Rain::Particle`, `Rain.h:41-47`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RainSplashParticle {
    pub xform: XRayMatrix,
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
    pub time_remaining_sec: f32,
}

/// Per-frame splash draw instance scaled by remaining life (`dxRainRender.cpp:293-295`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RainSplashInstance {
    pub xform: XRayMatrix,
    pub color_rgba: [f32; 4],
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
}

/// Output emitted by `RainEffectState::step_with_raycast`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RainFrameOutput {
    pub active: bool,
    pub desired_items: u32,
    pub active_items: u32,
    pub audio_volume: f32,
    pub rain_hemi: f32,
    pub rain_color_rgba: [f32; 4],
    /// Contiguous groups of 4 vertices per streak quad, indexed by `QUAD_INDEX_PATTERN`.
    pub streak_vertices: Vec<EnvLitVertex>,
    pub splashes: Vec<RainSplashInstance>,
}

/// Pure-Rust CPU rain streak & splash simulator (`CEffect_Rain` + `dxRainRender`).
#[derive(Debug, Clone)]
pub struct RainEffectState {
    pub config: RainEffectConfig,
    pub items: Vec<RainDropItem>,
    pub splashes: Vec<RainSplashParticle>,
    pub current_items: usize,
    pub audio_active: bool,
    pub audio_volume: f32,
    pub rain_hemi: f32,
    hemi_factor: f32,
    time_sec: f32,
    perlin: PerlinNoise1D,
    rng: ParticleRng,
}

impl Default for RainEffectState {
    fn default() -> Self {
        Self::new(1, RainEffectConfig::default())
    }
}

impl RainEffectState {
    pub fn new(seed: u64, config: RainEffectConfig) -> Self {
        Self {
            config,
            items: Vec::new(),
            splashes: Vec::new(),
            current_items: 0,
            audio_active: false,
            audio_volume: 0.0,
            rain_hemi: 0.0,
            hemi_factor: 0.0,
            time_sec: 0.0,
            perlin: PerlinNoise1D::new((seed & 0xFFFF) as u32, 2, 1.0, 0.66666),
            rng: ParticleRng::new(seed),
        }
    }

    /// Reset all active drops, splashes, and audio volume (`CEffect_Rain::InvalidateState`).
    pub fn invalidate_state(&mut self) {
        self.audio_active = false;
        self.audio_volume = 0.0;
        self.current_items = 0;
        self.items.clear();
        self.splashes.clear();
    }

    /// Compute wind-gust-perturbed horizontal spawn offset & fall axis (`CEffect_Rain::Prepare`, `Rain.cpp:81-105`).
    pub fn prepare_axis_and_offset(
        &mut self,
        wind_vel_01: f32,
        wind_dir_rad: f32,
    ) -> ([f32; 2], [f32; 3]) {
        let wind_gust = self.perlin.get_continuous(self.time_sec * 0.3) * 2.0;
        let wind_velocity = (wind_vel_01 + wind_gust).clamp(0.0, 1.0);
        let pitch = self.config.drop_max_angle_rad * wind_velocity;
        let axis = dir_from_hp(wind_dir_rad, pitch - std::f32::consts::FRAC_PI_2);

        let denom = (std::f32::consts::FRAC_PI_2 - pitch).sin();
        let dist = if denom.abs() > 1e-5 {
            pitch.sin() * self.config.source_offset / denom
        } else {
            0.0
        };
        let fix_north = wind_dir_rad - std::f32::consts::FRAC_PI_2;
        ([dist * fix_north.cos(), dist * fix_north.sin()], axis)
    }

    fn renew_item(&mut self, item: &mut RainDropItem, height: f32, b_hit: bool) {
        item.uv_set = (self.rng.next_u64() & 1) as u8;
        let travel_sec = if item.speed > 1e-5 {
            height.max(0.0) / item.speed
        } else {
            0.0
        };
        if b_hit {
            item.life_remaining_sec = travel_sec;
            item.hit_remaining_sec = travel_sec;
            item.hit_pending = true;
            item.pos_hit = [
                item.pos[0] + item.dir[0] * height,
                item.pos[1] + item.dir[1] * height,
                item.pos[2] + item.dir[2] * height,
            ];
        } else {
            item.life_remaining_sec = travel_sec;
            item.hit_remaining_sec = 2.0 * travel_sec;
            item.hit_pending = false;
            item.pos_hit = item.pos;
        }
    }

    fn born_item<F>(
        &mut self,
        env: &MixedEnvironment,
        camera_pos: [f32; 3],
        camera_dir: [f32; 3],
        ray_pick: &mut F,
    ) -> RainDropItem
    where
        F: FnMut([f32; 3], [f32; 3], f32) -> Option<f32>,
    {
        let wind_dir = -env.wind_direction;
        let wind_vel = (env.wind_velocity * 0.001).clamp(0.0, 1.0);
        let (rain_offset, rain_axis) = self.prepare_axis_and_offset(wind_vel, wind_dir);

        let r = self.config.source_radius * 0.5;
        let rx = rng_range(&mut self.rng, -r, r);
        let rz = rng_range(&mut self.rng, -r, r);

        let final_view = [
            camera_pos[0] + camera_dir[0] * 5.0,
            camera_pos[1] + camera_dir[1] * 5.0,
            camera_pos[2] + camera_dir[2] * 5.0,
        ];
        let angle =
            rng_range(&mut self.rng, -self.config.drop_angle_rad, self.config.drop_angle_rad)
                * (1.5 - wind_vel);
        let dir = random_dir_cone(&mut self.rng, rain_axis, angle);
        let pos = [
            rain_offset[0] + final_view[0] + rx,
            self.config.source_offset + camera_pos[1],
            rain_offset[1] + final_view[2] + rz,
        ];
        let speed = rng_range(&mut self.rng, self.config.drop_speed_min, self.config.drop_speed_max)
            * self.config.drop_speed_scale
            * (wind_vel * 1.5).clamp(0.5, 1.0);

        let mut item = RainDropItem {
            pos,
            pos_hit: pos,
            dir,
            speed,
            life_remaining_sec: 0.0,
            hit_remaining_sec: 0.0,
            hit_pending: false,
            uv_set: 0,
        };
        if let Some(hit_dist) = ray_pick(pos, dir, self.config.max_distance) {
            let clamped = hit_dist.clamp(0.0, self.config.max_distance);
            self.renew_item(&mut item, clamped, true);
        } else {
            self.renew_item(&mut item, self.config.max_distance, false);
        }
        item
    }

    fn spawn_splash(&mut self, pos: [f32; 3], splash_bounds: Option<([f32; 3], f32)>) {
        if (self.rng.next_u64() & 1) != 0 {
            return;
        }
        if self.splashes.len() >= self.config.max_splashes {
            return;
        }
        let Some((b_center, b_radius)) = splash_bounds else {
            return;
        };
        let mut xform = matrix_rotate_y(rng_range(&mut self.rng, 0.0, std::f32::consts::TAU));
        xform[3][0] = pos[0];
        xform[3][1] = pos[1];
        xform[3][2] = pos[2];
        xform[3][3] = 1.0;
        let world_center = matrix_transform_point(&xform, b_center);
        self.splashes.push(RainSplashParticle {
            xform,
            bounds_center: world_center,
            bounds_radius: b_radius,
            time_remaining_sec: self.config.splash_lifetime_sec,
        });
    }

    fn update_audio(
        &mut self,
        dt_sec: f32,
        rain_density: f32,
        wind_velocity: f32,
        hemi_luminance: f32,
    ) {
        let wind_vel = (wind_velocity * 0.001).clamp(0.0, 1.0)
            * if rain_density > 0.0 { 1.0 } else { 0.0 };
        let factor = rain_density * 0.5 + wind_vel * 0.5;
        let t = dt_sec.clamp(0.001, 1.0);
        let hemi = hemi_luminance.max(0.0);
        self.hemi_factor = self.hemi_factor * (1.0 - t) + hemi * t;
        self.rain_hemi = hemi;

        if factor < 1e-5 {
            self.audio_active = false;
            self.audio_volume = 0.0;
        } else {
            self.audio_active = true;
            self.audio_volume = (factor * self.hemi_factor).clamp(0.1, 1.0);
        }
    }

    /// Advance rain particles, cylinder wrapping, raycast hits, splashes, audio, and streak
    /// billboards for one frame (`Rain.cpp:177-480`, `dxRainRender.cpp:60-335`).
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_raycast<F>(
        &mut self,
        dt_sec: f32,
        env: &MixedEnvironment,
        camera_pos: [f32; 3],
        camera_dir: [f32; 3],
        splash_bounds: Option<([f32; 3], f32)>,
        hemi_luminance: f32,
        mut ray_pick: F,
    ) -> RainFrameOutput
    where
        F: FnMut([f32; 3], [f32; 3], f32) -> Option<f32>,
    {
        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        self.time_sec += dt;
        self.update_audio(dt, env.rain_density, env.wind_velocity, hemi_luminance);

        let rain_color_rgba = env.rain_color_rgba();
        let factor = env.rain_density;
        if factor < 1e-5 {
            return RainFrameOutput {
                active: false,
                desired_items: 0,
                active_items: 0,
                audio_volume: self.audio_volume,
                rain_hemi: self.rain_hemi,
                rain_color_rgba,
                streak_vertices: Vec::new(),
                splashes: Vec::new(),
            };
        }

        let desired_items = env.rain_desired_count(self.config.max_particles) as usize;
        if self.current_items < desired_items {
            self.current_items = desired_items;
        }
        while self.items.len() < self.current_items {
            let item = self.born_item(env, camera_pos, camera_dir, &mut ray_pick);
            self.items.push(item);
        }

        let wrap_radius_sq = (self.config.source_radius * 1.5).powi(2);
        let upper_y = camera_pos[1] + self.config.source_offset;
        let factor_visual = factor * 0.5 + 0.5;
        let mut streak_vertices = Vec::with_capacity(self.current_items * 4);

        let mut i = 0usize;
        while i < self.current_items {
            let mut hit_pos_to_spawn = None;
            let mut needs_born = false;

            {
                let one = &mut self.items[i];
                if one.hit_pending && one.hit_remaining_sec < 0.0 {
                    one.hit_pending = false;
                    hit_pos_to_spawn = Some(one.pos_hit);
                    if self.current_items > desired_items {
                        self.current_items -= 1;
                    }
                }
                if one.life_remaining_sec < 0.0 {
                    needs_born = true;
                    if self.current_items > desired_items {
                        self.current_items -= 1;
                    }
                }
            }

            if let Some(hit_p) = hit_pos_to_spawn {
                self.spawn_splash(hit_p, splash_bounds);
            }
            if needs_born {
                self.items[i] = self.born_item(env, camera_pos, camera_dir, &mut ray_pick);
                if i >= self.current_items {
                    break;
                }
            }

            let mut wrap_query: Option<(usize, [f32; 3], [f32; 3], f32)> = None;
            {
                let one = &mut self.items[i];
                let step_len = one.speed * dt;
                one.pos[0] += one.dir[0] * step_len;
                one.pos[1] += one.dir[1] * step_len;
                one.pos[2] += one.dir[2] * step_len;
                one.life_remaining_sec -= dt;
                one.hit_remaining_sec -= dt;

                let wx = one.pos[0] - camera_pos[0];
                let wz = one.pos[2] - camera_pos[2];
                let wlen_sq = wx * wx + wz * wz;
                if wlen_sq > wrap_radius_sq {
                    if (one.pos[1] - camera_pos[1]) < self.config.sink_offset {
                        one.invalidate();
                    } else {
                        let wlen = wlen_sq.sqrt();
                        let inv_wlen = 1.0 / wlen;
                        let shift = wlen + self.config.source_radius;
                        one.pos[0] -= (wx * inv_wlen) * shift;
                        one.pos[2] -= (wz * inv_wlen) * shift;

                        // Plane at y = upper_y with normal = (0, -1, 0), ray -one.dir:
                        let denom = one.dir[1];
                        if denom.abs() > 1e-5 {
                            let t_plane = (one.pos[1] - upper_y) / denom;
                            if t_plane >= 0.0 {
                                let src_p = [
                                    one.pos[0] - one.dir[0] * t_plane,
                                    one.pos[1] - one.dir[1] * t_plane,
                                    one.pos[2] - one.dir[2] * t_plane,
                                ];
                                wrap_query = Some((i, src_p, one.dir, t_plane));
                            } else {
                                one.invalidate();
                            }
                        } else {
                            one.invalidate();
                        }
                    }
                }
            }

            if let Some((idx, src_p, dir, dist_to_plane)) = wrap_query {
                if let Some(hit_h) = ray_pick(src_p, dir, self.config.max_distance) {
                    if hit_h <= dist_to_plane {
                        self.items[idx].invalidate();
                    } else {
                        let mut item = self.items[idx];
                        self.renew_item(&mut item, hit_h - dist_to_plane, true);
                        self.items[idx] = item;
                    }
                } else if self.config.max_distance > dist_to_plane {
                    let mut item = self.items[idx];
                    self.renew_item(&mut item, self.config.max_distance - dist_to_plane, false);
                    self.items[idx] = item;
                } else {
                    self.items[idx].invalidate();
                }
            }

            let one = &self.items[i];
            let pos_head = one.pos;
            let trail_len = self.config.drop_length * factor_visual;
            let pos_trail = [
                pos_head[0] - one.dir[0] * trail_len,
                pos_head[1] - one.dir[1] * trail_len,
                pos_head[2] - one.dir[2] * trail_len,
            ];
            let delta = [
                pos_head[0] - pos_trail[0],
                pos_head[1] - pos_trail[1],
                pos_head[2] - pos_trail[2],
            ];
            let line_d = normalize_vec3(delta);
            let mid = [
                pos_trail[0] + delta[0] * 0.5,
                pos_trail[1] + delta[1] * 0.5,
                pos_trail[2] + delta[2] * 0.5,
            ];
            let cam_dir = normalize_vec3([
                mid[0] - camera_pos[0],
                mid[1] - camera_pos[1],
                mid[2] - camera_pos[2],
            ]);
            let line_top = cross3(cam_dir, line_d);
            let w = self.config.drop_width;
            let uvs = RAIN_STREAK_UVS[(one.uv_set & 1) as usize];

            streak_vertices.push(EnvLitVertex {
                pos: [
                    pos_trail[0] - line_top[0] * w,
                    pos_trail[1] - line_top[1] * w,
                    pos_trail[2] - line_top[2] * w,
                ],
                color_rgba: rain_color_rgba,
                uv: uvs[0],
            });
            streak_vertices.push(EnvLitVertex {
                pos: [
                    pos_trail[0] + line_top[0] * w,
                    pos_trail[1] + line_top[1] * w,
                    pos_trail[2] + line_top[2] * w,
                ],
                color_rgba: rain_color_rgba,
                uv: uvs[1],
            });
            streak_vertices.push(EnvLitVertex {
                pos: [
                    pos_head[0] - line_top[0] * w,
                    pos_head[1] - line_top[1] * w,
                    pos_head[2] - line_top[2] * w,
                ],
                color_rgba: rain_color_rgba,
                uv: uvs[2],
            });
            streak_vertices.push(EnvLitVertex {
                pos: [
                    pos_head[0] + line_top[0] * w,
                    pos_head[1] + line_top[1] * w,
                    pos_head[2] + line_top[2] * w,
                ],
                color_rgba: rain_color_rgba,
                uv: uvs[3],
            });

            i += 1;
        }

        // Advance active splashes (`dxRainRender.cpp:275-321`)
        let life_denom = self.config.splash_lifetime_sec.max(1e-5);
        let mut splashes_out = Vec::with_capacity(self.splashes.len());
        self.splashes.retain_mut(|p| {
            p.time_remaining_sec -= dt;
            if p.time_remaining_sec < 0.0 {
                return false;
            }
            let s = (p.time_remaining_sec / life_denom).clamp(0.0, 1.0);
            let m_scale = [
                [s, 0.0, 0.0, 0.0],
                [0.0, s, 0.0, 0.0],
                [0.0, 0.0, s, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            splashes_out.push(RainSplashInstance {
                xform: matrix_mul_43(&p.xform, &m_scale),
                color_rgba: rain_color_rgba,
                bounds_center: p.bounds_center,
                bounds_radius: p.bounds_radius,
            });
            true
        });

        RainFrameOutput {
            active: true,
            desired_items: desired_items as u32,
            active_items: self.current_items as u32,
            audio_volume: self.audio_volume,
            rain_hemi: self.rain_hemi,
            rain_color_rgba,
            streak_vertices,
            splashes: splashes_out,
        }
    }

    /// Convenience wrapper of `step_with_raycast` without level collision raycasts.
    pub fn step_no_collision(
        &mut self,
        dt_sec: f32,
        env: &MixedEnvironment,
        camera_pos: [f32; 3],
        camera_dir: [f32; 3],
        hemi_luminance: f32,
    ) -> RainFrameOutput {
        self.step_with_raycast(
            dt_sec,
            env,
            camera_pos,
            camera_dir,
            None,
            hemi_luminance,
            |_, _, _| None,
        )
    }

    /// Transform a `.dm` splash mesh across all active splash instances (`dxRainRender.cpp:298`),
    /// stopping cleanly before exceeding `u16::MAX` vertex indices.
    pub fn emit_splash_mesh(
        dm: &DetailMesh,
        output: &RainFrameOutput,
    ) -> (Vec<DetailMeshVertexOut>, Vec<u16>) {
        let mut verts = Vec::with_capacity(output.splashes.len() * dm.vertices.len());
        let mut indices = Vec::with_capacity(output.splashes.len() * dm.indices.len());
        for inst in &output.splashes {
            let Ok(base) = u16::try_from(verts.len()) else {
                break;
            };
            let Some((v, idx)) = dm.transfer(&inst.xform, inst.color_rgba, 0.0, 0.0, base) else {
                break;
            };
            verts.extend(v);
            indices.extend(idx);
        }
        (verts, indices)
    }
}

/// Playback state of `ThunderboltEffectState` (`CEffect_Thunderbolt::EState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThunderboltPlayState {
    #[default]
    Idle,
    Working,
}

/// Camera-facing top/center flare quad emitted for an active lightning bolt (`dxThunderboltRender.cpp:61-98`).
#[derive(Debug, Clone, PartialEq)]
pub struct ThunderboltGradientQuad {
    pub flare: ThunderboltFlareDef,
    pub vertices: [EnvLitVertex; 4],
}

/// Active lightning strike mesh & flare draw payload (`dxThunderboltRender::Render`).
#[derive(Debug, Clone, PartialEq)]
pub struct ThunderboltBoltDraw {
    pub lightning_model: String,
    pub xform: XRayMatrix,
    pub color_rgba: [f32; 4],
    pub dv: f32,
    pub center: [f32; 3],
    pub size: f32,
    pub top_gradient: ThunderboltGradientQuad,
    pub center_gradient: ThunderboltGradientQuad,
}

/// Per-frame output from `ThunderboltEffectState::step_with_raycast`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThunderboltFrameOutput {
    pub active: bool,
    pub lightning_phase: f32,
    pub flash_color: [f32; 3],
    pub current_direction: [f32; 3],
    pub triggered_sound: Option<String>,
    pub triggered_sound_pos: [f32; 3],
    pub triggered_distance: f32,
    pub bolt_draw: Option<ThunderboltBoltDraw>,
}

#[derive(Debug, Clone, Copy, Default)]
struct ThunderboltFlashSnapshot {
    applied: bool,
    base_sky: [f32; 3],
    base_sun: [f32; 3],
    base_fog: [f32; 3],
    base_sun_dir: [f32; 3],
    flash_sky: [f32; 3],
    flash_sun: [f32; 3],
    flash_fog: [f32; 3],
    flash_sun_dir: [f32; 3],
}

/// Pure-Rust lightning bolt scheduler & atmospheric flash simulator
/// (`CEffect_Thunderbolt` + `dxThunderboltRender`).
#[derive(Debug, Clone)]
pub struct ThunderboltEffectState {
    pub state: ThunderboltPlayState,
    pub enabled: bool,
    pub current_desc: Option<ThunderboltDescDef>,
    pub current_xform: XRayMatrix,
    pub current_direction: [f32; 3],
    pub lightning_center: [f32; 3],
    pub lightning_size: f32,
    pub lightning_phase: f32,
    pub life_time: f32,
    pub current_time: f32,
    pub time_global: f32,
    pub next_lightning_time: f32,
    flash_snapshot: ThunderboltFlashSnapshot,
    rng: ParticleRng,
}

impl Default for ThunderboltEffectState {
    fn default() -> Self {
        Self::new(1)
    }
}

impl ThunderboltEffectState {
    pub fn new(seed: u64) -> Self {
        Self {
            state: ThunderboltPlayState::Idle,
            enabled: false,
            current_desc: None,
            current_xform: XRAY_IDENTITY_MATRIX,
            current_direction: [0.0, -1.0, 0.0],
            lightning_center: [0.0; 3],
            lightning_size: 0.0,
            lightning_phase: 0.0,
            life_time: 0.0,
            current_time: 0.0,
            time_global: 0.0,
            next_lightning_time: 0.0,
            flash_snapshot: ThunderboltFlashSnapshot::default(),
            rng: ParticleRng::new(seed),
        }
    }

    /// Restore un-flashed sky/sun/fog values on `env` if unchanged since the previous flash
    /// (`thunderbolt.cpp:201-205, 272-279`).
    pub fn undo_flash(&mut self, env: &mut MixedEnvironment) {
        if self.flash_snapshot.applied {
            if env.sky_color == self.flash_snapshot.flash_sky {
                env.sky_color = self.flash_snapshot.base_sky;
            }
            if env.sun_color == self.flash_snapshot.flash_sun {
                env.sun_color = self.flash_snapshot.base_sun;
            }
            if env.fog_color == self.flash_snapshot.flash_fog {
                env.fog_color = self.flash_snapshot.base_fog;
            }
            if env.sun_dir == self.flash_snapshot.flash_sun_dir {
                env.sun_dir = self.flash_snapshot.base_sun_dir;
            }
            self.flash_snapshot.applied = false;
        }
    }

    /// Apply atmospheric lightning flash color/direction modulation onto `env`
    /// (`thunderbolt.cpp:307-333`).
    pub fn apply_flash(
        &mut self,
        env: &mut MixedEnvironment,
        global_params: &EnvGlobalParams,
        output: &ThunderboltFrameOutput,
    ) {
        self.flash_snapshot.applied = false;
        if !output.active {
            return;
        }
        self.flash_snapshot.base_sky = env.sky_color;
        self.flash_snapshot.base_sun = env.sun_color;
        self.flash_snapshot.base_fog = env.fog_color;
        self.flash_snapshot.base_sun_dir = env.sun_dir;

        for (c, f) in env.sky_color.iter_mut().zip(output.flash_color.iter()) {
            *c = (*c + f * global_params.sky_color).clamp(0.0, 1.0);
        }
        for (c, f) in env.sun_color.iter_mut().zip(output.flash_color.iter()) {
            *c += f * global_params.sun_color;
        }
        for (c, f) in env.fog_color.iter_mut().zip(output.flash_color.iter()) {
            *c += f * global_params.fog_color;
        }
        if output.current_direction[1] < 0.0 {
            env.sun_dir = output.current_direction;
        }

        self.flash_snapshot.flash_sky = env.sky_color;
        self.flash_snapshot.flash_sun = env.sun_color;
        self.flash_snapshot.flash_fog = env.fog_color;
        self.flash_snapshot.flash_sun_dir = env.sun_dir;
        self.flash_snapshot.applied = true;
    }

    /// Transform `dm` (`lightning_model`) with `draw.xform` and animated UV shift `(0.0, draw.dv)`.
    pub fn emit_bolt_mesh(
        dm: &DetailMesh,
        draw: &ThunderboltBoltDraw,
    ) -> Option<(Vec<DetailMeshVertexOut>, Vec<u16>)> {
        dm.transfer(&draw.xform, draw.color_rgba, 0.0, draw.dv, 0)
    }

    fn build_gradient_quad(
        pos: [f32; 3],
        flare: &ThunderboltFlareDef,
        lightning_size: f32,
        lightning_phase: f32,
        camera_right: [f32; 3],
        camera_top: [f32; 3],
    ) -> ThunderboltGradientQuad {
        let c_val = (flare.opacity * lightning_phase).clamp(0.0, 1.0);
        let color_rgba = [c_val, c_val, c_val, c_val];
        let sx = [
            camera_right[0] * (flare.radius[0] * lightning_size),
            camera_right[1] * (flare.radius[0] * lightning_size),
            camera_right[2] * (flare.radius[0] * lightning_size),
        ];
        let sy = [
            camera_top[0] * (-flare.radius[1] * lightning_size),
            camera_top[1] * (-flare.radius[1] * lightning_size),
            camera_top[2] * (-flare.radius[1] * lightning_size),
        ];
        let vtx = |sign_x: f32, sign_y: f32, u: f32, v: f32| EnvLitVertex {
            pos: [
                pos[0] + sign_x * sx[0] + sign_y * sy[0],
                pos[1] + sign_x * sx[1] + sign_y * sy[1],
                pos[2] + sign_x * sx[2] + sign_y * sy[2],
            ],
            color_rgba,
            uv: [u, v],
        };
        ThunderboltGradientQuad {
            flare: flare.clone(),
            vertices: [
                vtx(1.0, -1.0, 0.0, 0.0),
                vtx(1.0, 1.0, 0.0, 1.0),
                vtx(-1.0, -1.0, 1.0, 0.0),
                vtx(-1.0, 1.0, 1.0, 1.0),
            ],
        }
    }

    /// Advance lightning scheduling, active bolt animation, and draw payload
    /// (`CEffect_Thunderbolt::OnFrame`, `Bolt`, `RayPick`, `dxThunderboltRender::Render`).
    ///
    /// Note: in `SThunderboltDesc::load` (`thunderbolt.cpp:67`), C++ overrides `color_anim.fps = frame_count as f32`
    /// so sampling `color_anim` at `current_time / life_time` sweeps the full timeline over `[0, 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_raycast<F>(
        &mut self,
        dt_sec: f32,
        env: &MixedEnvironment,
        global_params: &EnvGlobalParams,
        weather_lib: &WeatherLibrary,
        light_anims: Option<&LightAnimLibrary>,
        camera_pos: [f32; 3],
        camera_right: [f32; 3],
        camera_top: [f32; 3],
        mut ray_pick: F,
    ) -> ThunderboltFrameOutput
    where
        F: FnMut([f32; 3], [f32; 3], f32) -> Option<f32>,
    {
        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        self.time_global += dt;

        let mut output = ThunderboltFrameOutput::default();
        let collection_id = env.active_thunderbolt_id.as_str();
        let period = env.bolt_period;
        let duration = env.bolt_duration;
        let now_enabled = !collection_id.is_empty();

        if self.enabled != now_enabled {
            self.enabled = now_enabled;
            self.next_lightning_time =
                self.time_global + period + rng_range(&mut self.rng, -period * 0.5, period * 0.5);
        } else if self.enabled
            && self.time_global > self.next_lightning_time
            && self.state == ThunderboltPlayState::Idle
            && let Some(coll) = weather_lib.thunderbolt_collections.get(collection_id)
            && !coll.thunderbolts.is_empty()
        {
            let pick_idx = self.rng.range_usize(coll.thunderbolts.len());
            let desc_id = &coll.thunderbolts[pick_idx];
            if let Some(desc) = weather_lib.thunderbolts.get(desc_id).cloned() {
                self.state = ThunderboltPlayState::Working;
                self.life_time =
                    duration + rng_range(&mut self.rng, -duration * 0.5, duration * 0.5);
                self.life_time = self.life_time.max(1e-4);
                self.current_time = 0.0;

                let (sun_h, _) = hp_from_dir(env.sun_dir);
                let alt = global_params.altitude_rad;
                let lng = rng_range(
                    &mut self.rng,
                    sun_h - global_params.delta_longitude_rad + std::f32::consts::PI,
                    sun_h + global_params.delta_longitude_rad + std::f32::consts::PI,
                );
                let min_d = env.far_plane * global_params.min_dist_factor;
                let max_d = env.far_plane * 0.95;
                let dist = rng_range(&mut self.rng, min_d.min(max_d), max_d.max(min_d));
                self.current_direction = dir_from_hp(lng, alt);
                let pos = [
                    camera_pos[0] + self.current_direction[0] * dist,
                    camera_pos[1] + self.current_direction[1] * dist,
                    camera_pos[2] + self.current_direction[2] * dist,
                ];

                let dev = [
                    rng_range(&mut self.rng, -global_params.tilt_rad, global_params.tilt_rad),
                    rng_range(&mut self.rng, 0.0, std::f32::consts::TAU),
                    rng_range(&mut self.rng, -global_params.tilt_rad, global_params.tilt_rad),
                ];
                let mut xf = matrix_set_xyzi_translate(dev, [0.0, 0.0, 0.0]);
                let light_dir = matrix_transform_dir(&xf, [0.0, -1.0, 0.0]);

                let mut size = env.far_plane * 2.0;
                if let Some(hit) = ray_pick(pos, light_dir, size) {
                    size = hit.clamp(0.0, size);
                } else if light_dir[1].abs() > 1e-5 {
                    // Plane y = 0 with normal (0, -1, 0): dist = pos.y / -light_dir.y
                    let plane_dist = -pos[1] / light_dir[1];
                    if plane_dist >= 0.0 && plane_dist <= size {
                        size = plane_dist;
                    }
                }
                self.lightning_size = size;
                self.lightning_center = [
                    pos[0] + light_dir[0] * (size * 0.5),
                    pos[1] + light_dir[1] * (size * 0.5),
                    pos[2] + light_dir[2] * (size * 0.5),
                ];

                let mut scale_m = XRAY_IDENTITY_MATRIX;
                scale_m[0][0] = size;
                scale_m[1][1] = size;
                scale_m[2][2] = size;
                xf[3][0] = pos[0];
                xf[3][1] = pos[1];
                xf[3][2] = pos[2];
                self.current_xform = matrix_mul_43(&xf, &scale_m);

                if self.rng.unit_f32() < global_params.second_probability {
                    self.next_lightning_time = self.time_global + duration + 1e-5;
                } else {
                    self.next_lightning_time = self.time_global
                        + period
                        + rng_range(&mut self.rng, -period * 0.3, period * 0.3);
                    output.triggered_sound = desc.sound.clone();
                    output.triggered_sound_pos = pos;
                    output.triggered_distance = dist;
                }

                self.current_direction = [
                    -self.current_direction[0],
                    -self.current_direction[1],
                    -self.current_direction[2],
                ];
                self.current_desc = Some(desc);
            }
        }

        if self.state == ThunderboltPlayState::Working {
            if self.current_time > self.life_time {
                self.state = ThunderboltPlayState::Idle;
                return output;
            }
            self.current_time += dt;
            let norm_t = if self.life_time > 1e-5 {
                self.current_time / self.life_time
            } else {
                0.0
            };
            self.lightning_phase = (1.5 * norm_t).clamp(0.0, 1.0);

            let desc = match &self.current_desc {
                Some(d) => d,
                None => {
                    self.state = ThunderboltPlayState::Idle;
                    return output;
                }
            };

            let flash_rgb = if let Some(lib) = light_anims {
                if let Some(anim) = lib.find_item(&desc.color_anim).or_else(|| lib.find_item_ci(&desc.color_anim)) {
                    let mut normalized_anim = LightAnimItem {
                        fps: anim.frame_count.max(1) as f32,
                        ..anim.clone()
                    };
                    let (argb, _) = normalized_anim.calculate_rgb(norm_t);
                    normalized_anim.fps = anim.fps;
                    let rgba = unpack_argb_f32(argb);
                    [
                        rgba[0].clamp(0.0, 1.0),
                        rgba[1].clamp(0.0, 1.0),
                        rgba[2].clamp(0.0, 1.0),
                    ]
                } else {
                    [1.0, 1.0, 1.0]
                }
            } else {
                [1.0, 1.0, 1.0]
            };

            let dv = if self.lightning_phase > 0.5 {
                ((self.rng.next_u64() & 1) as f32) * 0.5
            } else {
                self.lightning_phase * 0.5
            };

            let top_pos = [
                self.current_xform[3][0],
                self.current_xform[3][1],
                self.current_xform[3][2],
            ];
            let top_gradient = Self::build_gradient_quad(
                top_pos,
                &desc.gradient_top,
                self.lightning_size,
                self.lightning_phase,
                camera_right,
                camera_top,
            );
            // Note: GSC `dxThunderboltRender.cpp:82` copy-pasted `m_GradientTop->fOpacity`; we evaluate
            // `gradient_center.opacity` cleanly for `center_gradient`.
            let center_gradient = Self::build_gradient_quad(
                self.lightning_center,
                &desc.gradient_center,
                self.lightning_size,
                self.lightning_phase,
                camera_right,
                camera_top,
            );

            output.active = true;
            output.lightning_phase = self.lightning_phase;
            output.flash_color = flash_rgb;
            output.current_direction = self.current_direction;
            output.bolt_draw = Some(ThunderboltBoltDraw {
                lightning_model: desc.lightning_model.clone(),
                xform: self.current_xform,
                color_rgba: [1.0, 1.0, 1.0, 1.0],
                dv,
                center: self.lightning_center,
                size: self.lightning_size,
                top_gradient,
                center_gradient,
            });
        }

        output
    }
}

/// Constants matching `xr_efflensflare.cpp` (`BLEND_INC_SPEED`, `BLEND_DEC_SPEED`, `MAX_RAYS`, `RayDeltas`).
pub const LENS_FLARE_BLEND_INC_SPEED: f32 = 8.0;
pub const LENS_FLARE_BLEND_DEC_SPEED: f32 = 4.0;
pub const LENS_FLARE_MAX_RAYS: usize = 5;
pub const LENS_FLARE_RAY_DELTAS: [[f32; 2]; LENS_FLARE_MAX_RAYS] =
    [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]];
pub const LENS_FLARE_RAY_SCALE: f32 = 0.02;
pub const LENS_FLARE_RAY_MAX_DIST: f32 = 1000.0;

#[inline]
fn blend_lerp(cur: &mut f32, tgt: f32, speed: f32, dt: f32) {
    let diff = tgt - *cur;
    let diff_a = diff.abs();
    if diff_a < 1e-5 {
        return;
    }
    let mot = (speed * dt).min(diff_a);
    *cur += (diff / diff_a) * mot;
}

/// Lens flare crossfade state (`CLensFlare::LFState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LensFlarePlayState {
    #[default]
    None,
    Idle,
    Hide,
    Show,
}

/// Type of billboard emitted by `LensFlareState` (`dxLensFlareRender::Render`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensFlareQuadKind {
    Source,
    Flare(usize),
    Gradient,
}

/// Single ordered lens flare billboard quad (`QuadIB` vertex layout).
#[derive(Debug, Clone, PartialEq)]
pub struct LensFlareDrawQuad {
    pub kind: LensFlareQuadKind,
    pub shader: String,
    pub texture: String,
    pub vertices: [EnvLitVertex; 4],
}

/// Per-frame output from `LensFlareState::step_with_raycast`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LensFlareFrameOutput {
    pub render_active: bool,
    pub state_blend: f32,
    pub visibility_blend: f32,
    pub gradient_value: f32,
    pub sun_world_pos: [f32; 3],
    pub quads: Vec<LensFlareDrawQuad>,
}

/// Pure-Rust sun disc, lens flare & halo gradient runtime (`CLensFlare` + `dxLensFlareRender`).
#[derive(Debug, Clone, Default)]
pub struct LensFlareState {
    pub state: LensFlarePlayState,
    pub current_id: String,
    pub state_blend: f32,
    pub f_blend: f32,
    pub f_gradient_value: f32,
}

impl LensFlareState {
    pub fn new() -> Self {
        Self::default()
    }

    fn build_flare_quad(
        kind: LensFlareQuadKind,
        shader: &str,
        texture: &str,
        center: [f32; 3],
        sx: [f32; 3],
        sy: [f32; 3],
        color_rgba: [f32; 4],
    ) -> LensFlareDrawQuad {
        let vtx = |sign_x: f32, sign_y: f32, u: f32, v: f32| EnvLitVertex {
            pos: [
                center[0] + sign_x * sx[0] + sign_y * sy[0],
                center[1] + sign_x * sx[1] + sign_y * sy[1],
                center[2] + sign_x * sx[2] + sign_y * sy[2],
            ],
            color_rgba,
            uv: [u, v],
        };
        LensFlareDrawQuad {
            kind,
            shader: shader.to_string(),
            texture: texture.to_string(),
            vertices: [
                vtx(1.0, -1.0, 0.0, 0.0),
                vtx(1.0, 1.0, 0.0, 1.0),
                vtx(-1.0, -1.0, 1.0, 0.0),
                vtx(-1.0, 1.0, 1.0, 1.0),
            ],
        }
    }

    /// Update lens flare transitions, 5-ray occlusion sampling, screen-space gradient attenuation,
    /// and ordered quad emission (`xr_efflensflare.cpp:270-513`, `dxLensFlareRender.cpp:30-144`).
    ///
    /// `ray_transmittance(origin, ray_dir, max_dist)` should return `0.0` when occluded, `1.0` when
    /// unobstructed, or `(0..1)` for partially transparent materials.
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_raycast<F>(
        &mut self,
        dt_sec: f32,
        time_factor: f32,
        paused: bool,
        env: &MixedEnvironment,
        weather_lib: &WeatherLibrary,
        camera_pos: [f32; 3],
        camera_right: [f32; 3],
        camera_top: [f32; 3],
        camera_dir: [f32; 3],
        full_transform: Option<&XRayMatrix>,
        draw_flags: (bool, bool, bool),
        mut ray_transmittance: F,
    ) -> LensFlareFrameOutput
    where
        F: FnMut([f32; 3], [f32; 3], f32) -> f32,
    {
        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        let tf = if time_factor.is_finite() && time_factor >= 0.0 {
            time_factor
        } else {
            1.0
        };

        let next_id = if env.active_sun_id.is_empty()
            || !weather_lib.lens_flares.contains_key(&env.active_sun_id)
        {
            ""
        } else {
            env.active_sun_id.as_str()
        };

        match self.state {
            LensFlarePlayState::None => {
                self.state = LensFlarePlayState::Show;
                self.current_id = next_id.to_string();
            }
            LensFlarePlayState::Idle => {
                if next_id != self.current_id {
                    self.state = LensFlarePlayState::Hide;
                }
            }
            LensFlarePlayState::Show => {
                if let Some(cur) = weather_lib.lens_flares.get(&self.current_id) {
                    self.state_blend += cur.blend_rise_speed * dt * tf;
                } else {
                    self.state_blend = 1.0 + 1e-4;
                }
                if self.state_blend >= 1.0 || paused {
                    self.state_blend = 1.0;
                    self.state = LensFlarePlayState::Idle;
                }
            }
            LensFlarePlayState::Hide => {
                if let Some(cur) = weather_lib.lens_flares.get(&self.current_id) {
                    self.state_blend -= cur.blend_down_speed * dt * tf;
                } else {
                    self.state_blend = -1e-4;
                }
                if self.state_blend <= 0.0 || paused {
                    self.state = LensFlarePlayState::Show;
                    self.current_id = next_id.to_string();
                    self.state_blend = weather_lib
                        .lens_flares
                        .get(&self.current_id)
                        .map(|d| d.blend_rise_speed * dt * tf)
                        .unwrap_or(0.0);
                }
            }
        }
        self.state_blend = self.state_blend.clamp(0.0, 1.0);

        let light_mag_sq = env.sun_color[0] * env.sun_color[0]
            + env.sun_color[1] * env.sun_color[1]
            + env.sun_color[2] * env.sun_color[2];
        let Some(desc) = weather_lib.lens_flares.get(&self.current_id) else {
            return LensFlareFrameOutput::default();
        };
        if light_mag_sq <= 0.0 {
            return LensFlareFrameOutput::default();
        }

        let v_sun_dir = [-env.sun_dir[0], -env.sun_dir[1], -env.sun_dir[2]];
        let vec_dir = normalize_vec3(camera_dir);
        let vec_sun_norm = normalize_vec3(v_sun_dir);
        let f_dot = dot3(vec_sun_norm, vec_dir);
        if f_dot <= 0.01 {
            return LensFlareFrameOutput {
                render_active: false,
                state_blend: self.state_blend,
                visibility_blend: self.f_blend,
                gradient_value: self.f_gradient_value,
                sun_world_pos: [0.0; 3],
                quads: Vec::new(),
            };
        }

        let f_distance = env.far_plane * 0.75;
        let vec_center = [
            camera_pos[0] + vec_dir[0] * f_distance,
            camera_pos[1] + vec_dir[1] * f_distance,
            camera_pos[2] + vec_dir[2] * f_distance,
        ];
        let sun_dist = f_distance / f_dot;
        let vec_light = [
            camera_pos[0] + vec_sun_norm[0] * sun_dist,
            camera_pos[1] + vec_sun_norm[1] * sun_dist,
            camera_pos[2] + vec_sun_norm[2] * sun_dist,
        ];
        let vec_axis = [
            vec_light[0] - vec_center[0],
            vec_light[1] - vec_center[1],
            vec_light[2] - vec_center[2],
        ];
        let vec_x = normalize_vec3(camera_right);
        let vec_y = cross3(vec_x, vec_dir);
        let _ = camera_top; // orthonormal screen basis derived from right x dir per xr_efflensflare.cpp:376

        let vec_sx = [
            vec_x[0] * LENS_FLARE_RAY_SCALE,
            vec_x[1] * LENS_FLARE_RAY_SCALE,
            vec_x[2] * LENS_FLARE_RAY_SCALE,
        ];
        let vec_sy = [
            vec_y[0] * LENS_FLARE_RAY_SCALE,
            vec_y[1] * LENS_FLARE_RAY_SCALE,
            vec_y[2] * LENS_FLARE_RAY_SCALE,
        ];

        let mut f_vis_result = 0.0f32;
        for delta in &LENS_FLARE_RAY_DELTAS {
            let ray_d = [
                v_sun_dir[0] + vec_sx[0] * delta[0] + vec_sy[0] * delta[1],
                v_sun_dir[1] + vec_sx[1] * delta[0] + vec_sy[1] * delta[1],
                v_sun_dir[2] + vec_sx[2] * delta[0] + vec_sy[2] * delta[1],
            ];
            f_vis_result +=
                ray_transmittance(camera_pos, ray_d, LENS_FLARE_RAY_MAX_DIST).clamp(0.0, 1.0);
        }
        f_vis_result *= 1.0 / (LENS_FLARE_MAX_RAYS as f32);

        blend_lerp(&mut self.f_blend, f_vis_result, LENS_FLARE_BLEND_DEC_SPEED, dt);
        self.f_blend = self.f_blend.clamp(0.0, 1.0);

        if desc.has_gradient {
            let (kx, ky, in_range) = if let Some(ft) = full_transform {
                if let Some(mut scr) = matrix_transform_projected(ft, vec_light) {
                    scr[1] *= -1.0;
                    let sun_blend = 0.5f32;
                    let sun_max = 2.5f32;
                    let ax = scr[0].abs();
                    let ay = scr[1].abs();
                    if ax > sun_max || ay > sun_max {
                        (0.0, 0.0, false)
                    } else {
                        let kx = if ax > sun_blend {
                            (sun_max - ax) / (sun_max - sun_blend)
                        } else {
                            1.0
                        };
                        let ky = if ay > sun_blend {
                            (sun_max - ay) / (sun_max - sun_blend)
                        } else {
                            1.0
                        };
                        (kx, ky, true)
                    }
                } else {
                    (0.0, 0.0, false)
                }
            } else {
                (1.0, 1.0, true)
            };
            if in_range {
                let op = self.state_blend * desc.gradient_opacity;
                self.f_gradient_value = kx * ky * op * self.f_blend;
            } else {
                self.f_gradient_value = 0.0;
            }
        }

        let (draw_sun, draw_flares, draw_gradient) = draw_flags;
        let mut quads = Vec::new();
        let dw_light = [env.sun_color[0], env.sun_color[1], env.sun_color[2], 1.0];

        if draw_sun && desc.has_source {
            let r = desc.source_radius * f_distance;
            let sx = [vec_x[0] * r, vec_x[1] * r, vec_x[2] * r];
            let sy = [vec_y[0] * r, vec_y[1] * r, vec_y[2] * r];
            let mut color = if desc.source_ignore_color {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                dw_light
            };
            color[3] *= self.state_blend;
            quads.push(Self::build_flare_quad(
                LensFlareQuadKind::Source,
                &desc.source_shader,
                &desc.source_texture,
                vec_light,
                sx,
                sy,
                color,
            ));
        }

        if self.f_blend >= 1e-5 {
            if draw_flares && desc.has_flares {
                let axis_sq = dot3(vec_axis, vec_axis);
                let vec_dx = if axis_sq > 1e-12 {
                    let inv = 1.0 / axis_sq.sqrt();
                    [vec_axis[0] * inv, vec_axis[1] * inv, vec_axis[2] * inv]
                } else {
                    vec_x
                };
                let vec_dy = cross3(vec_dx, vec_dir);
                for (idx, flare) in desc.flares.iter().enumerate() {
                    let center = [
                        vec_center[0] + vec_axis[0] * flare.position,
                        vec_center[1] + vec_axis[1] * flare.position,
                        vec_center[2] + vec_axis[2] * flare.position,
                    ];
                    let r = flare.radius * f_distance;
                    let sx = [vec_dx[0] * r, vec_dx[1] * r, vec_dx[2] * r];
                    let sy = [vec_dy[0] * r, vec_dy[1] * r, vec_dy[2] * r];
                    let cl = flare.opacity * self.f_blend * self.state_blend;
                    let color = [
                        dw_light[0] * cl,
                        dw_light[1] * cl,
                        dw_light[2] * cl,
                        dw_light[3] * cl,
                    ];
                    quads.push(Self::build_flare_quad(
                        LensFlareQuadKind::Flare(idx),
                        &flare.shader,
                        &flare.texture,
                        center,
                        sx,
                        sy,
                        color,
                    ));
                }
            }

            if draw_gradient && self.f_gradient_value >= 1e-5 && desc.has_gradient {
                let r = desc.gradient_radius * self.f_gradient_value * f_distance;
                let sx = [vec_x[0] * r, vec_x[1] * r, vec_x[2] * r];
                let sy = [vec_y[0] * r, vec_y[1] * r, vec_y[2] * r];
                let cl = self.f_gradient_value * self.state_blend;
                let color = [
                    dw_light[0] * cl,
                    dw_light[1] * cl,
                    dw_light[2] * cl,
                    dw_light[3] * cl,
                ];
                quads.push(Self::build_flare_quad(
                    LensFlareQuadKind::Gradient,
                    &desc.gradient_shader,
                    &desc.gradient_texture,
                    vec_light,
                    sx,
                    sy,
                    color,
                ));
            }
        }

        LensFlareFrameOutput {
            render_active: true,
            state_blend: self.state_blend,
            visibility_blend: self.f_blend,
            gradient_value: self.f_gradient_value,
            sun_world_pos: vec_light,
            quads,
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
        let (uv0, uv1) = wind.clouds_uv_scroll(0.1, 0.2);
        assert!((uv0[0].hypot(uv0[1]) - 0.05).abs() < 1e-5);
        assert!((uv1[0].hypot(uv1[1]) - 0.1).abs() < 1e-5);

        let (cloud_verts, cloud_idx) = clouds_hemi_mesh(&mixed);
        assert_eq!(cloud_verts.len(), CLOUDS_HEMI_VERTEX_COUNT);
        assert_eq!(cloud_idx.len(), CLOUDS_HEMI_INDEX_COUNT);
        assert_eq!(cloud_verts[0].color_rgba, mixed.clouds_color_rgba());
    }

    #[test]
    fn test_perlin_1d_and_rain_effect_simulator() {
        let p_a = PerlinNoise1D::new(42, 2, 1.0, 0.66666);
        let p_b = PerlinNoise1D::new(42, 2, 1.0, 0.66666);
        assert!((p_a.get(1.37) - p_b.get(1.37)).abs() < 1e-7);

        let mut p_c = PerlinNoise1D::new(42, 2, 1.0, 0.66666);
        let c1 = p_c.get_continuous(0.3);
        let c2 = p_c.get_continuous(0.6);
        assert!(c1.is_finite() && c2.is_finite() && (c1 - c2).abs() > 1e-6);

        let cycle_doc = parse_ltx_str(
            r#"
[00:00:00]
rain_density = 0.5
rain_color = 0.4, 0.5, 0.6
wind_velocity = 400.0
wind_direction = 45.0
"#,
        );
        let cycle = WeatherCycle::from_ltx("w_rain", &cycle_doc).unwrap();
        let (a, b, w) = cycle.select_at(0.0).unwrap();
        let env = lerp_descriptors(
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

        let cfg = RainEffectConfig {
            max_particles: 20,
            splash_lifetime_sec: 0.3,
            ..RainEffectConfig::default()
        };
        let mut rain = RainEffectState::new(7, cfg);
        let cam_pos = [0.0, 2.0, 0.0];
        let cam_dir = [0.0, 0.0, 1.0];

        // Force frequent ground hits so splash lifecycle triggers deterministically
        let mut out = RainFrameOutput::default();
        for _ in 0..8 {
            out = rain.step_with_raycast(
                0.08,
                &env,
                cam_pos,
                cam_dir,
                Some(([0.0, 0.0, 0.0], 0.2)),
                0.8,
                |_, _, _| Some(1.0),
            );
        }
        assert!(out.active);
        assert_eq!(out.desired_items, 10);
        assert!(out.active_items >= 10);
        assert_eq!(out.streak_vertices.len(), (out.active_items as usize) * 4);
        assert!(out.audio_volume >= 0.1 && out.audio_volume <= 1.0);
        assert!(!out.splashes.is_empty());

        let mut dry = env.clone();
        dry.rain_density = 0.0;
        let dry_out = rain.step_no_collision(0.1, &dry, cam_pos, cam_dir, 0.8);
        assert!(!dry_out.active);
        assert_eq!(dry_out.audio_volume, 0.0);
        assert!(dry_out.streak_vertices.is_empty());

        let mut wetness = WetnessState::default();
        assert_eq!(
            wetness.step(
                WETNESS_CHANGE_TIME * 0.5,
                1.0,
                DEFAULT_WETNESS_MULTIPLIERS[0],
                DEFAULT_WETNESS_MULTIPLIERS[1]
            ),
            0.5
        );
        let dried = wetness.step(
            WETNESS_CHANGE_TIME * 0.5,
            0.0,
            DEFAULT_WETNESS_MULTIPLIERS[0],
            DEFAULT_WETNESS_MULTIPLIERS[1],
        );
        assert!((dried - 0.35).abs() < 1e-6);

        let proj = crate::camera::matrix_build_projection(
            std::f32::consts::FRAC_PI_2,
            9.0 / 16.0,
            DEFAULT_VIEWPORT_NEAR,
            env.far_plane,
        );
        let view = crate::camera::matrix_build_camera_dir(cam_pos, cam_dir, [0.0, 1.0, 0.0]);
        let full = matrix_mul_44(&proj, &view);
        let uni = StandardFrameUniforms::compute(
            &env,
            &view,
            &full,
            cam_pos,
            cam_dir,
            [0.0, 1.0, 0.0],
            90.0,
            9.0 / 16.0,
            DEFAULT_VIEWPORT_NEAR,
            1920,
            1080,
            12.5,
            3.5 * 3600.0,
            dried,
        );
        assert_eq!(uni.screen_res, [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0]);
        assert!((uni.ogse_c_screen[2] - 1.0).abs() < 1e-5);
        assert_eq!(uni.near_far_plane, [DEFAULT_VIEWPORT_NEAR, env.far_plane, 0.0, 0.0]);
        assert!((uni.rain_params[1] - dried).abs() < 1e-6);
        assert_eq!(uni.timers_game[3], 3.0);
        let fog_at_near = uni.fog_plane[0] * 0.0
            + uni.fog_plane[1] * 2.0
            + uni.fog_plane[2] * env.fog_near
            + uni.fog_plane[3];
        let fog_at_far = uni.fog_plane[0] * 0.0
            + uni.fog_plane[1] * 2.0
            + uni.fog_plane[2] * env.fog_far
            + uni.fog_plane[3];
        assert!(fog_at_near > fog_at_far + 0.9);
    }

    #[test]
    fn test_thunderbolt_and_lens_flare_step() {
        let mut lib = WeatherLibrary::default();
        lib.parse_thunderbolts(&parse_ltx_str(
            r#"
[tb_0]
gradient_top_shader = effects\flare
gradient_top_texture = fx\fx_thunder_top
gradient_top_radius = 0.05, 0.05
gradient_top_opacity = 0.7
gradient_center_shader = effects\flare
gradient_center_texture = fx\fx_thunder_center
gradient_center_radius = 0.04, 0.04
gradient_center_opacity = 0.6
color_anim = tb_flash
lightning_model = dm\lightning.dm
sound = nature\thunder1
"#,
        ));
        lib.parse_thunderbolt_collections(&parse_ltx_str("[tb_coll]\ntb_0 =\n"));
        lib.parse_suns(&parse_ltx_str(
            r#"
[sun_clear]
sun = true
sun_shader = effects\sun
sun_texture = fx\fx_sun
sun_radius = 0.06
sun_ignore_color = false
flares = true
flare_shader = effects\flare
flare_textures = fx\fx_flare1, fx\fx_flare2
flare_radius = 0.1, 0.2
flare_opacity = 0.5, 0.3
flare_position = 0.4, 0.8
gradient = true
gradient_shader = effects\flare
gradient_texture = fx\fx_grad
gradient_radius = 0.4
gradient_opacity = 0.6
blend_rise_time = 0.5
blend_down_time = 0.5
"#,
        ));

        let mut anim = LightAnimItem::new("tb_flash");
        anim.frame_count = 4;
        anim.keys.clear();
        anim.insert_key(0, crate::lanims::pack_argb(0xFF, 0xFF, 0xFF, 0xFF))
            .unwrap();
        anim.insert_key(4, crate::lanims::pack_argb(0xFF, 0x20, 0x20, 0x40))
            .unwrap();
        let mut anim_lib = LightAnimLibrary::default();
        anim_lib.append_item(anim).unwrap();

        let desc_doc = parse_ltx_str(
            r#"
[00:00:00]
sky_color = 0.2, 0.2, 0.3
sun_color = 0.8, 0.7, 0.6
fog_color = 0.1, 0.1, 0.2
far_plane = 400.0
thunderbolt_collection = tb_coll
thunderbolt_period = 0.2
thunderbolt_duration = 0.6
sun = sun_clear
"#,
        );
        let cycle = WeatherCycle::from_ltx("w_tb", &desc_doc).unwrap();
        let (a, b, w) = cycle.select_at(0.0).unwrap();
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
        mixed.sun_dir = normalize_vec3([0.0, -0.5, -1.0]);

        let gp = EnvGlobalParams::default();
        let mut tb = ThunderboltEffectState::new(9);
        let mut active_out = None;
        for _ in 0..20 {
            let out = tb.step_with_raycast(
                0.05,
                &mixed,
                &gp,
                &lib,
                Some(&anim_lib),
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                |_, _, _| Some(120.0),
            );
            if out.active {
                tb.apply_flash(&mut mixed, &gp, &out);
                active_out = Some(out);
                break;
            }
        }
        let tb_out = active_out.expect("thunderbolt should fire within period");
        assert!(tb_out.bolt_draw.is_some());
        assert!(mixed.sky_color[0] > 0.2);
        tb.undo_flash(&mut mixed);
        assert!((mixed.sky_color[0] - 0.2).abs() < 1e-5);

        // Lens flare looking straight toward the sun (-sun_dir = +0.5Y, +Z normalized)
        mixed.sun_dir = normalize_vec3([0.0, -0.2, -1.0]);
        let mut lf = LensFlareState::new();
        let proj = crate::camera::matrix_build_projection(
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.2,
            500.0,
        );
        let view = crate::camera::matrix_build_camera_dir(
            [0.0, 2.0, 0.0],
            [0.0, 0.2, 1.0],
            [0.0, 1.0, 0.0],
        );
        let full_xf = matrix_mul_43(&proj, &view);
        let mut lf_out = LensFlareFrameOutput::default();
        for _ in 0..10 {
            lf_out = lf.step_with_raycast(
                0.1,
                1.0,
                false,
                &mixed,
                &lib,
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                normalize_vec3([0.0, 0.2, 1.0]),
                Some(&full_xf),
                (true, true, true),
                |_, _, _| 1.0,
            );
        }
        assert!(lf_out.render_active);
        assert!(lf_out.state_blend > 0.9);
        assert!(lf_out.visibility_blend > 0.5);
        assert!(lf_out.quads.len() >= 3);
        assert_eq!(lf_out.quads[0].kind, LensFlareQuadKind::Source);

        // Looking away from sun disables flare rendering
        let away = lf.step_with_raycast(
            0.1,
            1.0,
            false,
            &mixed,
            &lib,
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            Some(&full_xf),
            (true, true, true),
            |_, _, _| 1.0,
        );
        assert!(!away.render_active);
        assert!(away.quads.is_empty());
    }

    #[test]
    fn scope_actor_and_pda_shader_uniforms() {
        let mut eye = ScopeEyeLerpState::new();
        let (p0, d0) = eye.step([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], DEFAULT_SCOPE_FOG_INTERP);
        assert_eq!(p0, [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(d0, [0.0, 0.0, 1.0, 0.0]);
        let (p1, _) = eye.step([11.0, 2.0, 3.0], [1.0, 0.0, 0.0], 0.25);
        assert!((p1[0] - 3.5).abs() < 1e-6);

        let fake_cfg = FakeScopeConfig::default();
        let fake_u = fake_cfg.to_uniforms();
        assert_eq!(std::mem::size_of::<FakeScopeUniforms>(), 48);
        assert_eq!(fake_u.fakescope_params, [0.66, 0.1, 1.0, 1.0]);
        assert_eq!(fake_u.fakescope_ca, [0.003, 0.66, 0.25, 0.25]);
        assert_eq!(fake_u.fakescope_params3, [0.0, 1.25, 4.0, 0.0]);

        let s3ds = S3dsScopeUniforms {
            markswitch_current: 2,
            markswitch_count: 5,
            markswitch_color: [1.0, 0.2, 0.1, 1.0],
            ..S3dsScopeUniforms::default()
        };
        assert_eq!(std::mem::size_of::<S3dsScopeUniforms>(), 128);
        assert_eq!(s3ds.markswitch_current_vec4(), [2.0, 0.0, 0.0, 0.0]);
        assert_eq!(s3ds.markswitch_count_vec4(), [5.0, 0.0, 0.0, 0.0]);

        let actor = ActorShaderData {
            health: 0.85,
            stamina: 0.5,
            bleeding: 0.1,
            helmet: true,
        };
        assert_eq!(actor.to_vec4(), [0.85, 0.5, 0.1, 1.0]);

        let pda = PdaShaderParams {
            display_factor: 0.75,
            psy_influence: 0.25,
            display_brightness: 0.9,
        };
        assert_eq!(pda.to_vec4(), [0.75, 0.25, 0.9, 0.0]);

        let g_consts = GlobalShaderConstants {
            hud_params: [1.0, 0.5, 0.0, 1.0],
            ..GlobalShaderConstants::default()
        };
        assert_eq!(std::mem::size_of::<GlobalShaderConstants>(), 64);
        assert_eq!(g_consts.hud_params[1], 0.5);

        let dev = DevDebugParams::default();
        assert_eq!(std::mem::size_of::<DevDebugParams>(), 128);
        assert_eq!(dev.params[7], [0.0; 4]);

        let ssfx = SsfxSurfaceConfig {
            is_underground: true,
            ..SsfxSurfaceConfig::default()
        }
        .to_gpu_uniforms(true, 0.016, 42);
        assert_eq!(std::mem::size_of::<SsfxSurfaceUniforms>(), 304);
        assert_eq!(ssfx.ssfx_gloss, [0.0, 0.92, 0.5, 0.0]);
        assert_eq!(ssfx.ssfx_hud_hemi, [0.15, 0.0, 0.0, 0.0]);
        assert_eq!(ssfx.ssfx_is_underground, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(ssfx.ssfx_issvp, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(ssfx.ssfx_timedelta, [0.016, 42.0, 0.0, 0.0]);

        let mut cmap = ColorMapCacheState::new();
        assert_eq!(cmap.set_textures("fx\\cmap_day", ""), [true, false]);
        assert_eq!(cmap.set_textures("fx\\cmap_day", "fx\\cmap_night"), [false, true]);
        assert_eq!(cmap.set_textures("", "fx\\cmap_night"), [true, false]);
        assert_eq!(cmap.slot_names(), ["", "fx\\cmap_night"]);
    }
}
