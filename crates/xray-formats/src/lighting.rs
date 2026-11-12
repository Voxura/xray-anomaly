//! Pure Rust Dynamic Lighting Pipeline, Shadow-Map Atlas Allocator & Object LightTrack
//!
//! Grounds X-Ray's local light volumes, adaptive shadow-map sizing, SMAP atlas packing,
//! projective accumulation transforms, and object hemisphere ambient tracking directly from:
//! - `src/Layers/xrRenderPC_R4/SMAP_Allocator.h` & `r2_R_lights.cpp`
//! - `src/Layers/xrRenderPC_R4/Light_Render_Direct_ComputeXFS.cpp`
//! - `src/Layers/xrRender/light.cpp` & `src/Layers/xrRenderPC_R4/r2_types.h` (`u_diffuse2s`)
//! - `src/Layers/xrRenderPC_R4/r4_rendertarget_accum_spot.cpp`, `accum_point.cpp`, `accum_reflected.cpp`, & `enable_scissor.cpp`
//! - `src/Layers/xrRender/Light_Package.h` & `Light_Package.cpp`
//! - `src/Layers/xrRenderPC_R4/light_GI.cpp` & `r2_R_lights.cpp` (`render_indirect`)
//! - `src/Layers/xrRender/LightTrack.h` & `LightTrack.cpp`

use std::f32::consts::FRAC_1_SQRT_2;

use crate::camera::{matrix_build_camera_dir, matrix_build_projection, matrix_build_projection_ortho};
use crate::cform::{CformBvh, RaycastOptions};
use crate::hom::{frustum_planes, ClipFrustum, FrustumVisibility};
use crate::level::{d3d_light_type, LevelLight};
use crate::ogf::{
    matrix_mul_43, matrix_mul_44, matrix_scale, matrix_transform_point,
    matrix_transform_projected, XRayMatrix, XRAY_IDENTITY_MATRIX,
};
use crate::skinning::matrix_transform_dir;

pub const SMAP_NEAR_PLANE: f32 = 0.1;
pub const SMAP_ADAPT_MIN: u32 = 768;
pub const SMAP_ADAPT_OPTIMAL: u32 = 768;
pub const SMAP_ADAPT_MAX: u32 = 1536;
pub const TEX_JITTER_SIZE: u32 = 64;

pub const LIGHT_SSA_GLOD_START_DEFAULT: f32 = 256.0;
pub const LIGHT_SSA_GLOD_END_DEFAULT: f32 = 64.0;
pub const LIGHT_SLIGHT_FADE_DEFAULT: f32 = 0.5;
pub const LIGHT_LS_SQUALITY_DEFAULT: f32 = 1.0;

pub const LT_HEMISAMPLES: usize = 26;
pub const LT_INC: f32 = 4.0;
pub const LT_DEC: f32 = 2.0;

/// Cube face indices used by `CROS_impl::hemi_cube` (`LightTrack.h:17-26`).
pub mod hemi_cube_faces {
    pub const POS_X: usize = 0;
    pub const POS_Y: usize = 1;
    pub const POS_Z: usize = 2;
    pub const NEG_X: usize = 3;
    pub const NEG_Y: usize = 4;
    pub const NEG_Z: usize = 5;
    pub const NUM_FACES: usize = 6;
}

/// 6 cubemap up hints (`cmNorm`) used when expanding shadowed point lights (`light.cpp:354-356`).
pub const OMNIPART_CUBE_UP_HINTS: [[f32; 3]; 6] = [
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
];

/// 6 cubemap forward directions (`cmDir`: `+X, -X, +Y, -Y, +Z, -Z`, `light.cpp:357-359`).
pub const OMNIPART_CUBE_DIRS: [[f32; 3]; 6] = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
];

/// 26 stratified hemisphere ray directions (`hdir[26][3]`, `LightTrack.cpp:107-140`).
pub const HEMI_SAMPLE_DIRS: [[f32; 3]; LT_HEMISAMPLES] = [
    [-0.26287, 0.52573, 0.80902],
    [0.27639, 0.44721, 0.85065],
    [-0.95106, 0.00000, 0.30902],
    [-0.95106, 0.00000, -0.30902],
    [0.58779, 0.00000, -0.80902],
    [0.58779, 0.00000, 0.80902],
    [-0.00000, 0.00000, 1.00000],
    [0.52573, 0.85065, 0.00000],
    [-0.26287, 0.52573, -0.80902],
    [-0.42533, 0.85065, 0.30902],
    [0.95106, 0.00000, 0.30902],
    [0.95106, 0.00000, -0.30902],
    [0.00000, 1.00000, 0.00000],
    [-0.58779, 0.00000, 0.80902],
    [-0.72361, 0.44721, 0.52573],
    [-0.72361, 0.44721, -0.52573],
    [-0.58779, 0.00000, -0.80902],
    [0.16246, 0.85065, -0.50000],
    [0.89443, 0.44721, 0.00000],
    [-0.85065, 0.52573, -0.00000],
    [0.16246, 0.85065, 0.50000],
    [0.68819, 0.52573, -0.50000],
    [0.27639, 0.44721, -0.85065],
    [0.00000, 0.00000, -1.00000],
    [-0.42533, 0.85065, -0.30902],
    [0.68819, 0.52573, 0.50000],
];

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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
fn normalize3_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let sq = dot3(v, v);
    if sq > 1e-12 && sq.is_finite() {
        let inv = 1.0 / sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

/// Renderer light classification (`enum IRender_Light::LT` in `Render.h:42-49`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RenderLightType {
    Direct = 0,
    Point = 1,
    Spot = 2,
    Omnipart = 3,
    Reflected = 4,
}

impl RenderLightType {
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::Direct),
            1 => Some(Self::Point),
            2 => Some(Self::Spot),
            3 => Some(Self::Omnipart),
            4 => Some(Self::Reflected),
            _ => None,
        }
    }

    pub fn from_d3d_light_type(d3d_type: u32) -> Self {
        match d3d_type {
            d3d_light_type::DIRECTIONAL => Self::Direct,
            d3d_light_type::SPOT => Self::Spot,
            _ => Self::Point,
        }
    }
}

/// Console gloss parameters for `u_diffuse2s` (`r2_types.h:174-195`, `xrRender_console.cpp:466-533`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlossConfig {
    pub r2_gloss_factor: f32,
    pub r2_gloss_min: f32,
    pub ssfx_gloss_method: i32,
    pub ssfx_gloss_factor: f32,
    pub ssfx_gloss_minmax: [f32; 3],
}

impl Default for GlossConfig {
    fn default() -> Self {
        Self {
            r2_gloss_factor: 4.0,
            r2_gloss_min: 0.0,
            ssfx_gloss_method: 0,
            ssfx_gloss_factor: 0.5,
            ssfx_gloss_minmax: [0.0, 0.92, 0.0],
        }
    }
}

/// Specular-from-diffuse factor matching `u_diffuse2s` (`r2_types.h:181-195`).
#[inline]
pub fn compute_u_diffuse2s(rgb: [f32; 3], cfg: &GlossConfig) -> f32 {
    if cfg.ssfx_gloss_method == 0 {
        let v = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
        let scaled = if v < 1.0 {
            v.max(0.0).powf(2.0 / 3.0)
        } else {
            v
        };
        cfg.r2_gloss_min + cfg.r2_gloss_factor * scaled
    } else {
        cfg.ssfx_gloss_minmax[0]
            + (cfg.ssfx_gloss_minmax[1] - cfg.ssfx_gloss_minmax[0]).clamp(0.0, 1.0)
                * cfg.ssfx_gloss_factor
    }
}

/// Inclusive texel rectangle inside a square shadow-map atlas (`struct SMAP_Rect`, `SMAP_Allocator.h:3-34`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SmapRect {
    pub min: [i32; 2],
    pub max: [i32; 2],
}

impl SmapRect {
    #[inline]
    pub fn setup(origin: [i32; 2], size: u32) -> Self {
        let span = size.saturating_sub(1) as i32;
        Self {
            min: origin,
            max: [origin[0] + span, origin[1] + span],
        }
    }

    #[inline]
    pub fn intersect(&self, other: &Self) -> bool {
        !(self.max[0] < other.min[0]
            || self.max[1] < other.min[1]
            || self.min[0] > other.max[0]
            || self.min[1] > other.max[1])
    }

    #[inline]
    pub fn valid(&self) -> bool {
        self.min[0] != self.max[0] && self.min[1] != self.max[1]
    }

    #[inline]
    pub fn critical_points(&self) -> ([i32; 2], [i32; 2]) {
        (
            [self.max[0] + 1, self.min[1]],
            [self.min[0], self.max[1] + 1],
        )
    }

    #[inline]
    pub fn size(&self) -> u32 {
        (self.max[0] - self.min[0] + 1).max(0) as u32
    }
}

/// Critical-point first-fit square shadow-map atlas allocator (`SMAP_Allocator`, `SMAP_Allocator.h:36-98`).
#[derive(Debug, Clone, Default)]
pub struct SmapAllocator {
    pool_size: u32,
    stack: Vec<SmapRect>,
    cpoint: Vec<[i32; 2]>,
}

impl SmapAllocator {
    pub fn new(pool_size: u32) -> Self {
        let mut alloc = Self::default();
        alloc.initialize(pool_size);
        alloc
    }

    pub fn initialize(&mut self, pool_size: u32) {
        self.pool_size = pool_size;
        self.stack.clear();
        self.cpoint.clear();
    }

    #[inline]
    pub fn pool_size(&self) -> u32 {
        self.pool_size
    }

    #[inline]
    pub fn allocated(&self) -> &[SmapRect] {
        &self.stack
    }

    fn add_rect(&mut self, rect: SmapRect) {
        self.stack.push(rect);
        let (p0, p1) = rect.critical_points();
        let ps = self.pool_size as i32;
        if p0[0] < ps && p0[1] < ps {
            self.cpoint.push(p0);
        }
        if p1[0] < ps && p1[1] < ps {
            self.cpoint.push(p1);
        }
    }

    /// Reserve a `tile_size x tile_size` square region in the atlas (`tile_size > 4 && tile_size <= pool_size`).
    pub fn push(&mut self, tile_size: u32) -> Option<SmapRect> {
        if tile_size <= 4 || tile_size > self.pool_size {
            return None;
        }
        if self.stack.is_empty() {
            let rect = SmapRect::setup([0, 0], tile_size);
            self.add_rect(rect);
            return Some(rect);
        }

        let ps = self.pool_size as i32;
        for it in 0..self.cpoint.len() {
            let cand = SmapRect::setup(self.cpoint[it], tile_size);
            if cand.max[0] >= ps || cand.max[1] >= ps {
                continue;
            }
            if self.stack.iter().any(|existing| existing.intersect(&cand)) {
                continue;
            }
            self.cpoint.remove(it);
            self.add_rect(cand);
            return Some(cand);
        }
        None
    }
}

/// Assignment output produced by `pack_smap_requests`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmapPackAssignment {
    pub item_index: usize,
    pub smap_id: u16,
    pub rect: SmapRect,
}

/// Greedy descending multi-page shadow-map atlas packer (`CRender::render_lights`, `r2_R_lights.cpp:52-83`).
pub fn pack_smap_requests(pool_size: u32, requests: &[(usize, u32)]) -> Vec<SmapPackAssignment> {
    if pool_size <= 4 || requests.is_empty() {
        return Vec::new();
    }
    let mut remaining: Vec<(usize, u32)> = requests
        .iter()
        .copied()
        .filter(|&(_, size)| size > 4 && size <= pool_size)
        .collect();
    // Reverse size sort matches `pred_area` (`a0 > a1`).
    remaining.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut packed = Vec::with_capacity(remaining.len());
    let mut pool = SmapAllocator::default();
    let mut smap_id: u16 = 0;

    while !remaining.is_empty() {
        pool.initialize(pool_size);
        let before = remaining.len();
        let mut next_remaining = Vec::new();
        for (item_index, tile_size) in remaining {
            if let Some(rect) = pool.push(tile_size) {
                packed.push(SmapPackAssignment {
                    item_index,
                    smap_id,
                    rect,
                });
            } else {
                next_remaining.push((item_index, tile_size));
            }
        }
        if next_remaining.len() == before {
            break;
        }
        remaining = next_remaining;
        smap_id = smap_id.saturating_add(1);
    }

    packed
}

/// Compute `(r_ssaGLOD_start, r_ssaGLOD_end)` matching `CRender::Calculate` (`r2_R_calculate.cpp:18-25`).
#[inline]
pub fn compute_r_ssa_glod_bounds(
    viewport_width: u32,
    viewport_height: u32,
    fov_deg: f32,
    r_lod: f32,
    glod_ssa_start: f32,
    glod_ssa_end: f32,
) -> (f32, f32) {
    let fov_factor = (90.0 / fov_deg.max(1.0)).powi(2);
    let g_f_screen = (viewport_width.max(1) as f32)
        * (viewport_height.max(1) as f32)
        * fov_factor
        * (1e-7 + r_lod.max(0.0));
    let start = (glod_ssa_start / 3.0).powi(2) / g_f_screen.max(1.0);
    let end = (glod_ssa_end / 3.0).powi(2) / g_f_screen.max(1.0);
    (start, end)
}

/// Configuration for `DynamicLight::compute_smap_xform` (`Light_Render_Direct_ComputeXFS.cpp`,
/// `xrRender_console.cpp:228,449`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSmapConfig {
    pub smap_size: u32,
    pub ls_squality: f32,
    pub ssfx_min_res: u32,
    pub ssfx_max_res: u32,
}

impl Default for LightSmapConfig {
    fn default() -> Self {
        Self {
            smap_size: 2048,
            ls_squality: LIGHT_LS_SQUALITY_DEFAULT,
            ssfx_min_res: 256,
            ssfx_max_res: SMAP_ADAPT_MAX,
        }
    }
}

/// Computed shadow-map transform & resolved square tile size for a shadowed light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightSmapXform {
    pub size: u32,
    pub tan_shift: f32,
    pub view: XRayMatrix,
    pub project: XRayMatrix,
    pub combine: XRayMatrix,
}

/// Cubemap face sub-light descriptor generated by `DynamicLight::expand_omnipart_faces`
/// (`light::export_`, `light.cpp:367-420`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OmnipartFaceLight {
    pub face_index: u8,
    pub light: DynamicLight,
}

/// Pure-Rust representation of X-Ray's dynamic `class light` (`light.h`, `light.cpp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicLight {
    pub light_type: RenderLightType,
    pub is_static: bool,
    pub is_active: bool,
    pub is_shadowed: bool,
    pub is_volumetric: bool,
    pub is_ortho: bool,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub right: [f32; 3],
    pub range: f32,
    pub cone_rad: f32,
    pub virtual_size: f32,
    pub color_rgb: [f32; 3],
    pub attenuation0: f32,
    pub attenuation1: f32,
    pub attenuation2: f32,
    pub falloff: f32,
    pub cached_smap_size: u32,
}

impl Default for DynamicLight {
    fn default() -> Self {
        Self {
            light_type: RenderLightType::Point,
            is_static: false,
            is_active: true,
            is_shadowed: false,
            is_volumetric: false,
            is_ortho: false,
            position: [0.0, -1000.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            right: [0.0, 0.0, 0.0],
            range: 8.0,
            cone_rad: 60.0f32.to_radians(),
            virtual_size: SMAP_NEAR_PLANE,
            color_rgb: [1.0, 1.0, 1.0],
            attenuation0: 1.0,
            attenuation1: 0.0,
            attenuation2: 0.0,
            falloff: 1.0,
            cached_smap_size: SMAP_ADAPT_MAX,
        }
    }
}

impl DynamicLight {
    /// Convert a static/scene [`LevelLight`] into a [`DynamicLight`].
    pub fn from_level_light(src: &LevelLight, is_shadowed: bool) -> Self {
        let light_type = RenderLightType::from_d3d_light_type(src.light_type);
        let cone_rad = if src.phi > 1e-4 {
            src.phi
        } else {
            60.0f32.to_radians()
        };
        Self {
            light_type,
            is_static: true,
            is_active: true,
            is_shadowed,
            is_volumetric: false,
            is_ortho: false,
            position: src.position,
            direction: src.normalized_direction(),
            right: [0.0, 0.0, 0.0],
            range: src.range.max(0.01),
            cone_rad,
            virtual_size: SMAP_NEAR_PLANE,
            color_rgb: [src.diffuse[0], src.diffuse[1], src.diffuse[2]],
            attenuation0: src.attenuation[0],
            attenuation1: src.attenuation[1],
            attenuation2: src.attenuation[2],
            falloff: src.falloff,
            cached_smap_size: SMAP_ADAPT_MAX,
        }
    }

    /// Build orthonormal `(right, up, dir)` axes matching `light::xform_calc` (`light.cpp:276-304`).
    pub fn ortho_basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let l_dir = normalize3_or(self.direction, [0.0, 0.0, 1.0]);
        if dot3(self.right, self.right) > 1e-5 {
            let r0 = normalize3_or(self.right, [1.0, 0.0, 0.0]);
            let l_up = normalize3_or(cross3(l_dir, r0), [0.0, 1.0, 0.0]);
            let l_right = normalize3_or(cross3(l_up, l_dir), [1.0, 0.0, 0.0]);
            (l_right, l_up, l_dir)
        } else {
            let mut l_up = [0.0, 1.0, 0.0];
            if dot3(l_up, l_dir).abs() > 0.99 {
                l_up = [0.0, 0.0, 1.0];
            }
            let l_right = normalize3_or(cross3(l_up, l_dir), [1.0, 0.0, 0.0]);
            let l_up = normalize3_or(cross3(l_dir, l_right), [0.0, 1.0, 0.0]);
            (l_right, l_up, l_dir)
        }
    }

    /// Minimal enclosing world-space bounding sphere `(center, radius)` (`light::spatial_move`, `light.cpp:203-242`).
    pub fn spatial_sphere(&self) -> ([f32; 3], f32) {
        let dir = normalize3_or(self.direction, [0.0, -1.0, 0.0]);
        match self.light_type {
            RenderLightType::Direct => (self.position, self.range),
            RenderLightType::Reflected | RenderLightType::Point => (self.position, self.range),
            RenderLightType::Spot => {
                let half_cone = (self.cone_rad * 0.5).clamp(1e-4, 1.56);
                if self.cone_rad >= std::f32::consts::FRAC_PI_2 {
                    let center = [
                        self.position[0] + dir[0] * self.range,
                        self.position[1] + dir[1] * self.range,
                        self.position[2] + dir[2] * self.range,
                    ];
                    (center, self.range * half_cone.tan())
                } else {
                    let cos_half = half_cone.cos();
                    let r = self.range / (2.0 * cos_half * cos_half).max(1e-6);
                    let center = [
                        self.position[0] + dir[0] * r,
                        self.position[1] + dir[1] * r,
                        self.position[2] + dir[2] * r,
                    ];
                    (center, r)
                }
            }
            RenderLightType::Omnipart => {
                let r = self.range * FRAC_1_SQRT_2;
                let center = [
                    self.position[0] + dir[0] * r,
                    self.position[1] + dir[1] * r,
                    self.position[2] + dir[2] * r,
                ];
                (center, r)
            }
        }
    }

    /// World-space light volume stencil/shader mesh transform (`light::xform_calc`, `light.cpp:307-350`).
    pub fn volume_world_xform(&self) -> XRayMatrix {
        let (l_right, l_up, l_dir) = self.ortho_basis();
        let m_r: XRayMatrix = [
            [l_right[0], l_right[1], l_right[2], 0.0],
            [l_up[0], l_up[1], l_up[2], 0.0],
            [l_dir[0], l_dir[1], l_dir[2], 0.0],
            [self.position[0], self.position[1], self.position[2], 1.0],
        ];
        let m_scale = match self.light_type {
            RenderLightType::Reflected | RenderLightType::Point => {
                matrix_scale(self.range, self.range, self.range)
            }
            RenderLightType::Spot => {
                let s = 2.0 * self.range * (self.cone_rad * 0.5).tan();
                matrix_scale(s, s, self.range)
            }
            RenderLightType::Omnipart => {
                let r2 = 2.0 * self.range;
                matrix_scale(r2, r2, r2)
            }
            RenderLightType::Direct => return XRAY_IDENTITY_MATRIX,
        };
        matrix_mul_43(&m_r, &m_scale)
    }

    /// Expand a shadowed point light into 6 `OMNIPART` face sub-lights (`light::export_`, `light.cpp:367-420`).
    pub fn expand_omnipart_faces(&self) -> [OmnipartFaceLight; 6] {
        std::array::from_fn(|face| {
            let dir = OMNIPART_CUBE_DIRS[face];
            let norm = OMNIPART_CUBE_UP_HINTS[face];
            let right = cross3(norm, dir);
            OmnipartFaceLight {
                face_index: face as u8,
                light: Self {
                    light_type: RenderLightType::Omnipart,
                    is_shadowed: true,
                    direction: dir,
                    right,
                    cone_rad: std::f32::consts::FRAC_PI_2,
                    ..*self
                },
            }
        })
    }

    /// Distance/area LOD fade factor `[0.0, 1.0]` (`light::get_LOD`, `light.cpp:454-462`).
    pub fn compute_lod(
        &self,
        camera_pos: [f32; 3],
        slight_fade: f32,
        r_ssa_glod_start: f32,
        r_ssa_glod_end: f32,
    ) -> f32 {
        if !self.is_shadowed {
            return 1.0;
        }
        let (sphere_p, sphere_r) = self.spatial_sphere();
        let dx = camera_pos[0] - sphere_p[0];
        let dy = camera_pos[1] - sphere_p[1];
        let dz = camera_pos[2] - sphere_p[2];
        let dist_sq = dx * dx + dy * dy + dz * dz + 1e-3;
        let ssa = slight_fade * sphere_r / dist_sq;
        let denom = r_ssa_glod_start - r_ssa_glod_end;
        if denom.abs() <= 1e-12 {
            return if ssa >= r_ssa_glod_start { 1.0 } else { 0.0 };
        }
        ((ssa - r_ssa_glod_end) / denom).clamp(0.0, 1.0).sqrt()
    }

    /// Compute adaptive shadow-map tile resolution and view/projection matrices
    /// (`CLight_Compute_XFORM_and_VIS::compute_xf_spot`, `Light_Render_Direct_ComputeXFS.cpp:4-106`).
    pub fn compute_smap_xform(
        &self,
        camera_pos: [f32; 3],
        cfg: &LightSmapConfig,
    ) -> LightSmapXform {
        let (_, l_up, l_dir) = self.ortho_basis();
        let (sphere_p, sphere_r) = self.spatial_sphere();
        let dx = camera_pos[0] - sphere_p[0];
        let dy = camera_pos[1] - sphere_p[1];
        let dz = camera_pos[2] - sphere_p[2];
        let dist = ((dx * dx + dy * dy + dz * dz).sqrt() - sphere_r).max(0.0);
        let ssa = ((self.range * self.range) / (1.0 + dist * dist)).clamp(0.0, 1.0);

        let intensity0 = (self.color_rgb[0] + self.color_rgb[1] + self.color_rgb[2]) / 3.0;
        let intensity1 =
            self.color_rgb[0] * 0.2125 + self.color_rgb[1] * 0.7154 + self.color_rgb[2] * 0.0721;
        let intensity = ((intensity0 + intensity1) * 0.5).max(0.0);

        let sizefactor = (self.range / 8.0).max(0.0);
        let widefactor = (self.cone_rad / std::f32::consts::FRAC_PI_2).max(0.0);

        let factor0 = ssa.sqrt();
        let factor1 = intensity.powf(1.0 / 16.0);
        let factor3 = sizefactor.powf(0.25);
        let factor4 = widefactor.sqrt();
        let factor = cfg.ls_squality * factor0 * factor1 * factor3 * factor4;

        let max_size = cfg.smap_size.min(cfg.ssfx_max_res).max(cfg.ssfx_min_res);
        let raw_size = (factor * SMAP_ADAPT_OPTIMAL as f32).floor() as i32;
        let clamped_size = raw_size.clamp(cfg.ssfx_min_res as i32, max_size as i32);
        let epsilon = (clamped_size as f32 * 0.01).ceil() as i32;
        let cached = self.cached_smap_size as i32;
        let final_size = if (clamped_size - cached).abs() >= epsilon {
            clamped_size
        } else {
            cached
        }
        .max(1) as u32;

        let view = matrix_build_camera_dir(self.position, l_dir, l_up);
        let tan_shift = match self.light_type {
            RenderLightType::Omnipart => 0.3,
            RenderLightType::Point => 0.2007129,
            _ => 0.0610865,
        };
        let fov_or_dim = (self.cone_rad + tan_shift).max(1e-3);
        let near = self.virtual_size.max(1e-4);
        let far = (self.range + 1e-7).max(near + 1e-4);
        let project = if self.is_ortho {
            matrix_build_projection_ortho(fov_or_dim, fov_or_dim, near, far)
        } else {
            matrix_build_projection(fov_or_dim, 1.0, near, far)
        };
        let combine = matrix_mul_44(&project, &view);

        LightSmapXform {
            size: final_size,
            tan_shift,
            view,
            project,
            combine,
        }
    }

    /// Select deferred accumulation shader element index matching `accum_point` / `accum_spot`
    /// (`SE_L_*`, `r4_rendertarget_accum_point.cpp:84-95`, `r2_types.h:151-155`).
    #[inline]
    pub fn accum_shader_element(
        &self,
        is_translucent: bool,
        smap_size: u32,
        atlas_size: u32,
    ) -> AccumShaderElement {
        if !self.is_shadowed {
            AccumShaderElement::Unshadowed
        } else if is_translucent {
            AccumShaderElement::Translucent
        } else if smap_size == atlas_size {
            AccumShaderElement::Fullsize
        } else {
            AccumShaderElement::Normal
        }
    }

    /// Compute projected NDC depth bounds `[z_min, z_max]` for `NVDBT` / depth-bounds testing
    /// when the enclosing light sphere (`R * 1.01`) is completely inside `frustum`
    /// (`CRenderTarget::enable_dbt_bounds`, `r4_rendertarget_enable_scissor.cpp:13-38`).
    pub fn compute_depth_bounds(
        &self,
        full_transform: &XRayMatrix,
        frustum: &ClipFrustum,
    ) -> Option<[f32; 2]> {
        let (sp_center, sp_radius) = self.spatial_sphere();
        if frustum.test_sphere_visibility(sp_center, sp_radius * 1.01) != FrustumVisibility::Fully {
            return None;
        }
        let min = [
            sp_center[0] - sp_radius,
            sp_center[1] - sp_radius,
            sp_center[2] - sp_radius,
        ];
        let max = [
            sp_center[0] + sp_radius,
            sp_center[1] + sp_radius,
            sp_center[2] + sp_radius,
        ];
        let mut z_min = f32::INFINITY;
        let mut z_max = f32::NEG_INFINITY;
        for mask in 0u8..8 {
            let corner = [
                if (mask & 1) != 0 { max[0] } else { min[0] },
                if (mask & 2) != 0 { max[1] } else { min[1] },
                if (mask & 4) != 0 { max[2] } else { min[2] },
            ];
            let p = matrix_transform_projected(full_transform, corner)?;
            z_min = z_min.min(p[2]);
            z_max = z_max.max(p[2]);
        }
        Some([z_min, z_max])
    }
}

/// Accumulation pass shader element index (`SE_L_*`, `src/Layers/xrRenderPC_R4/r2_types.h:150-155`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AccumShaderElement {
    Fill = 0,
    Unshadowed = 1,
    Normal = 2,
    Fullsize = 3,
    Translucent = 4,
}

/// Stencil-mask pass shader element index (`SE_MASK_*`, `src/Layers/xrRenderPC_R4/r2_types.h:157-163`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AccumMaskElement {
    Spot = 0,
    Point = 1,
    Direct = 2,
    AccumVol = 3,
    Accum2D = 4,
    Albedo = 5,
}

/// 2D screen-space texgen transform (`CRenderTarget::u_compute_texgen_screen`, `r4_rendertarget.cpp:223-239`).
#[inline]
pub fn compute_texgen_screen(wvp: &XRayMatrix) -> XRayMatrix {
    let m_texel_adjust: XRayMatrix = [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0, 1.0],
    ];
    matrix_mul_44(&m_texel_adjust, wvp)
}

/// 2D screen-space tiled jitter texgen transform (`CRenderTarget::u_compute_texgen_jitter`, `r4_rendertarget.cpp:242-261`).
#[inline]
pub fn compute_texgen_jitter(
    wvp: &XRayMatrix,
    viewport_width: u32,
    viewport_height: u32,
    jitter_size: u32,
) -> XRayMatrix {
    let base = compute_texgen_screen(wvp);
    let j = jitter_size.max(1) as f32;
    let sx = (viewport_width.max(1) as f32) / j;
    let sy = (viewport_height.max(1) as f32) / j;
    let scale = matrix_scale(sx, sy, 1.0);
    matrix_mul_44(&scale, &base)
}

/// Near-plane frustum intersection check used by `CRenderTarget::enable_scissor`
/// (`r4_rendertarget_enable_scissor.cpp:64-85`).
pub fn near_plane_intersects_sphere(
    full_transform: &XRayMatrix,
    sphere_center: [f32; 3],
    sphere_radius: f32,
) -> bool {
    let mut px = -(full_transform[0][3] + full_transform[0][2]);
    let mut py = -(full_transform[1][3] + full_transform[1][2]);
    let mut pz = -(full_transform[2][3] + full_transform[2][2]);
    let mut pw = -(full_transform[3][3] + full_transform[3][2]);
    let len = (px * px + py * py + pz * pz).sqrt();
    if len <= 1e-12 || !len.is_finite() {
        return false;
    }
    let denom = -1.0 / len;
    px *= denom;
    py *= denom;
    pz *= denom;
    pw *= denom;
    let classify = px * sphere_center[0] + py * sphere_center[1] + pz * sphere_center[2] + pw;
    (classify - sphere_radius) <= 0.0
}

/// Configuration for [`compute_spot_accum_constants`] (`r4_rendertarget_accum_spot.cpp:89-169`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotAccumConfig {
    pub atlas_size: u32,
    pub depth_scale: f32,
    pub depth_bias: f32,
    pub slight_fade: f32,
    pub r_ssa_glod_start: f32,
    pub r_ssa_glod_end: f32,
    pub gloss: GlossConfig,
}

impl Default for SpotAccumConfig {
    fn default() -> Self {
        Self {
            atlas_size: 2048,
            depth_scale: 1.00001,
            depth_bias: -0.001,
            slight_fade: LIGHT_SLIGHT_FADE_DEFAULT,
            r_ssa_glod_start: 1e-4,
            r_ssa_glod_end: 1e-5,
            gloss: GlossConfig::default(),
        }
    }
}

/// GPU shader inputs computed for a single `accum_spot` or `accum_point` pass
/// (`r4_rendertarget_accum_spot.cpp:83-169`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotAccumConstants {
    pub m_shadow: XRayMatrix,
    pub m_lmap: XRayMatrix,
    /// Transposed columns `(col0, col1)` passed to `RCache.set_ca("m_lmap", 0/1, ...)`.
    pub m_lmap_rows: [[f32; 4]; 2],
    /// `[view_x, view_y, view_z, att_factor]` (`Ldynamic_pos`).
    pub dynamic_pos: [f32; 4],
    /// `[r * lod, g * lod, b * lod, spec]` (`Ldynamic_color`).
    pub dynamic_color: [f32; 4],
    pub is_full_smap: bool,
}

/// Build `accum_spot` / `accum_point` shadow, lightmap, and attenuation constants in view space
/// (`r4_rendertarget_accum_spot.cpp:89-169`).
pub fn compute_spot_accum_constants(
    light: &DynamicLight,
    smap_xform: &LightSmapXform,
    atlas_rect: SmapRect,
    camera_pos: [f32; 3],
    view_matrix: &XRayMatrix,
    inv_view_matrix: &XRayMatrix,
    cfg: &SpotAccumConfig,
) -> SpotAccumConstants {
    let smapsize = cfg.atlas_size.max(1) as f32;
    let f_texel_offs = 0.5 / smapsize;
    let view_dim = (smap_xform.size.saturating_sub(2) as f32) / smapsize;
    let view_sx = (atlas_rect.min[0] + 1) as f32 / smapsize;
    let view_sy = (atlas_rect.min[1] + 1) as f32 / smapsize;

    let m_texel_adjust_smap: XRayMatrix = [
        [view_dim * 0.5, 0.0, 0.0, 0.0],
        [0.0, -view_dim * 0.5, 0.0, 0.0],
        [0.0, 0.0, cfg.depth_scale, 0.0],
        [
            view_dim * 0.5 + view_sx + f_texel_offs,
            view_dim * 0.5 + view_sy + f_texel_offs,
            cfg.depth_bias,
            1.0,
        ],
    ];
    let xf_project_smap = matrix_mul_44(&m_texel_adjust_smap, &smap_xform.project);
    let view_to_light = matrix_mul_44(&smap_xform.view, inv_view_matrix);
    let m_shadow_smap = matrix_mul_44(&xf_project_smap, &view_to_light);

    let m_texel_adjust_lmap: XRayMatrix = [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0],
        [0.0, 0.0, cfg.depth_scale, 0.0],
        [0.5 + f_texel_offs, 0.5 + f_texel_offs, cfg.depth_bias, 1.0],
    ];
    let xf_project_lmap = matrix_mul_44(&m_texel_adjust_lmap, &smap_xform.project);
    let m_lmap = matrix_mul_44(&xf_project_lmap, &view_to_light);

    let m_shadow = if light.is_shadowed {
        m_shadow_smap
    } else {
        m_lmap
    };
    let m_lmap_rows = [
        [m_lmap[0][0], m_lmap[1][0], m_lmap[2][0], m_lmap[3][0]],
        [m_lmap[0][1], m_lmap[1][1], m_lmap[2][1], m_lmap[3][1]],
    ];

    let lod = light.compute_lod(
        camera_pos,
        cfg.slight_fade,
        cfg.r_ssa_glod_start,
        cfg.r_ssa_glod_end,
    );
    let clr = [
        light.color_rgb[0] * lod,
        light.color_rgb[1] * lod,
        light.color_rgb[2] * lod,
    ];
    let l_spec = compute_u_diffuse2s(clr, &cfg.gloss);
    let l_pos = matrix_transform_point(view_matrix, light.position);
    let att_r = (light.range * 0.95).max(1e-4);
    let att_factor = 1.0 / (att_r * att_r);

    SpotAccumConstants {
        m_shadow,
        m_lmap,
        m_lmap_rows,
        dynamic_pos: [l_pos[0], l_pos[1], l_pos[2], att_factor],
        dynamic_color: [clr[0], clr[1], clr[2], l_spec],
        is_full_smap: smap_xform.size == cfg.atlas_size,
    }
}

/// GPU shader inputs computed for a single `CRenderTarget::accum_point` pass
/// (`r4_rendertarget_accum_point.cpp:17-107`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointAccumConstants {
    pub shader_element: AccumShaderElement,
    pub m_texgen: XRayMatrix,
    /// `[view_x, view_y, view_z, 1 / (att_r * att_r)]` with `att_r = range * 0.95` (`Ldynamic_pos`).
    pub dynamic_pos: [f32; 4],
    /// `[r, g, b, u_diffuse2s(rgb)]` (`Ldynamic_color`).
    pub dynamic_color: [f32; 4],
    /// Screen-space shadow ID (`L->sss_id`, or `-1` during dual-viewport/SVP frames).
    pub sss_id: i32,
    pub is_full_smap: bool,
    pub cull_backfaces: bool,
}

/// Build `accum_point` view-space position, attenuation, specular, `m_texgen`, `sss_id`, and
/// shader element matching `CRenderTarget::accum_point` (`r4_rendertarget_accum_point.cpp:17-107`).
#[allow(clippy::too_many_arguments)]
pub fn compute_point_accum_constants(
    light: &DynamicLight,
    smap_size: u32,
    atlas_size: u32,
    is_translucent: bool,
    is_svp_frame: bool,
    light_sss_id: i32,
    view_matrix: &XRayMatrix,
    full_transform: &XRayMatrix,
    gloss: &GlossConfig,
) -> PointAccumConstants {
    let wvp = matrix_mul_44(full_transform, &light.volume_world_xform());
    let m_texgen = compute_texgen_screen(&wvp);
    let l_pos = matrix_transform_point(view_matrix, light.position);
    let att_r = (light.range * 0.95).max(1e-4);
    let l_spec = compute_u_diffuse2s(light.color_rgb, gloss);
    PointAccumConstants {
        shader_element: light.accum_shader_element(is_translucent, smap_size, atlas_size),
        m_texgen,
        dynamic_pos: [l_pos[0], l_pos[1], l_pos[2], 1.0 / (att_r * att_r)],
        dynamic_color: [
            light.color_rgb[0],
            light.color_rgb[1],
            light.color_rgb[2],
            l_spec,
        ],
        sss_id: if is_svp_frame { -1 } else { light_sss_id },
        is_full_smap: smap_size == atlas_size,
        cull_backfaces: true,
    }
}

/// GPU shader inputs computed for a single `CRenderTarget::accum_reflected` pass
/// (`r4_rendertarget_accum_reflected.cpp:13-62`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectedAccumConstants {
    pub m_texgen: XRayMatrix,
    /// `[view_x, view_y, view_z, 1 / (range * range)]` (`Ldynamic_pos`).
    pub dynamic_pos: [f32; 4],
    /// `[r, g, b, u_diffuse2s(rgb)]` (`Ldynamic_color`).
    pub dynamic_color: [f32; 4],
    /// Unit view-space reflected bounce direction `[dx, dy, dz, 0.0]` (`direction`).
    pub direction_view: [f32; 4],
    /// True when the light's bounding sphere intersects the near plane (`CULL_CW` vs `CULL_CCW`).
    pub cull_backfaces: bool,
}

/// Build `accum_reflected` constants matching `CRenderTarget::accum_reflected`
/// (`r4_rendertarget_accum_reflected.cpp:18-61`).
///
/// Set `apply_half_texel_offset = false` for DX10/11 canonical texgen matching
/// [`compute_texgen_screen`], or `true` for the legacy half-texel offset in the C++ source.
pub fn compute_reflected_accum_constants(
    light: &DynamicLight,
    view_matrix: &XRayMatrix,
    full_transform: &XRayMatrix,
    viewport_width: u32,
    viewport_height: u32,
    apply_half_texel_offset: bool,
    gloss: &GlossConfig,
) -> ReflectedAccumConstants {
    let wvp = matrix_mul_44(full_transform, &light.volume_world_xform());
    let m_texgen = if apply_half_texel_offset {
        let ow = 0.5 / viewport_width.max(1) as f32;
        let oh = 0.5 / viewport_height.max(1) as f32;
        let m_texel_adjust: XRayMatrix = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, -0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5 + ow, 0.5 + oh, 0.0, 1.0],
        ];
        matrix_mul_44(&m_texel_adjust, &wvp)
    } else {
        compute_texgen_screen(&wvp)
    };

    let (sp_center, sp_radius) = light.spatial_sphere();
    let cull_backfaces = near_plane_intersects_sphere(full_transform, sp_center, sp_radius);

    let l_pos = matrix_transform_point(view_matrix, light.position);
    let raw_dir = matrix_transform_dir(view_matrix, light.direction);
    let l_dir = normalize3_or(raw_dir, [0.0, 0.0, 1.0]);
    let range = light.range.max(1e-4);
    let l_spec = compute_u_diffuse2s(light.color_rgb, gloss);

    ReflectedAccumConstants {
        m_texgen,
        dynamic_pos: [l_pos[0], l_pos[1], l_pos[2], 1.0 / (range * range)],
        dynamic_color: [
            light.color_rgb[0],
            light.color_rgb[1],
            light.color_rgb[2],
            l_spec,
        ],
        direction_view: [l_dir[0], l_dir[1], l_dir[2], 0.0],
        cull_backfaces,
    }
}

/// Visibility/range sort key matching `pred_light_cmp` (`Light_Package.cpp:12-24`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightPackageOrderKey {
    pub pending: bool,
    pub query_order: u32,
    pub range: f32,
}

/// Stable-sort light queue entries matching `light_Package::sort` (`Light_Package.cpp:12-32`):
/// completed queries precede pending ones; pending entries sort descending by `query_order`,
/// and completed entries sort descending by `range`.
pub fn sort_light_package_by_key<T, F>(items: &mut [T], mut key_fn: F)
where
    F: FnMut(&T) -> LightPackageOrderKey,
{
    items.sort_by(|lhs, rhs| {
        let a = key_fn(lhs);
        let b = key_fn(rhs);
        match (a.pending, b.pending) {
            (true, true) => b.query_order.cmp(&a.query_order),
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => b.range.total_cmp(&a.range),
        }
    });
}

/// Accumulate a directional sample `(dir, scale)` into X-Ray's 6-face `hemi_cube`
/// (`CROS_impl::accum_hemi`, `LightTrack.cpp:161-177`).
#[inline]
pub fn accum_hemi_cube(hemi_cube: &mut [f32; 6], dir: [f32; 3], scale: f32) {
    use hemi_cube_faces::*;
    if dir[0] > 0.0 {
        hemi_cube[POS_X] += dir[0] * scale;
    } else {
        hemi_cube[NEG_X] -= dir[0] * scale;
    }
    if dir[1] > 0.0 {
        hemi_cube[POS_Y] += dir[1] * scale;
    } else {
        hemi_cube[NEG_Y] -= dir[1] * scale;
    }
    if dir[2] > 0.0 {
        hemi_cube[POS_Z] += dir[2] * scale;
    } else {
        hemi_cube[NEG_Z] -= dir[2] * scale;
    }
}

/// Console defaults controlling `CROS_impl` (`xrRender_console.cpp:247-254`, `LightTrack.h:9-11`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightTrackConfig {
    pub dhemi_sky_scale: f32,
    pub dhemi_light_scale: f32,
    pub dhemi_light_flow: f32,
    pub dhemi_count: usize,
    pub lt_smooth: f32,
    pub lt_inc: f32,
    pub lt_dec: f32,
}

impl Default for LightTrackConfig {
    fn default() -> Self {
        Self {
            dhemi_sky_scale: 0.08,
            dhemi_light_scale: 0.2,
            dhemi_light_flow: 0.1,
            dhemi_count: 5,
            lt_smooth: 1.0,
            lt_inc: LT_INC,
            lt_dec: LT_DEC,
        }
    }
}

/// Static/dynamic light visibility entry tracked across frames by [`ObjectLightTracker`]
/// (`CROS_impl::Item` in `LightTrack.h:28-35`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackedLightEntry {
    pub id: u64,
    pub last_touched_frame: u32,
    pub position: [f32; 3],
    pub color_rgb: [f32; 3],
    pub attenuation0: f32,
    pub attenuation1: f32,
    pub attenuation2: f32,
    pub falloff: f32,
    pub is_static: bool,
    pub test: f32,
    pub energy: f32,
}

/// Pure-Rust per-object ambient, sun & hemisphere light tracker (`class CROS_impl`,
/// `LightTrack.h` / `LightTrack.cpp`).
#[derive(Debug, Clone)]
pub struct ObjectLightTracker {
    pub ray_clear: [bool; LT_HEMISAMPLES],
    pub result_count: usize,
    pub result_iterator: usize,
    pub hemi_cube: [f32; 6],
    pub hemi_cube_smooth: [f32; 6],
    pub hemi_value: f32,
    pub hemi_smooth: f32,
    pub sun_value: f32,
    pub sun_smooth: f32,
    pub approximate: [f32; 3],
    pub tracked_lights: Vec<TrackedLightEntry>,
}

impl Default for ObjectLightTracker {
    fn default() -> Self {
        Self {
            ray_clear: [false; LT_HEMISAMPLES],
            result_count: 0,
            result_iterator: 0,
            hemi_cube: [0.0; 6],
            hemi_cube_smooth: [0.0; 6],
            hemi_value: 0.5,
            hemi_smooth: 0.5,
            sun_value: 0.2,
            sun_smooth: 0.2,
            approximate: [0.0; 3],
            tracked_lights: Vec::new(),
        }
    }
}

impl ObjectLightTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return clamped scalar luminosity (`CROS_impl::get_luminocity`, `LightTrack.h:84-89`).
    #[inline]
    pub fn luminosity(&self) -> f32 {
        self.approximate[0]
            .max(self.approximate[1])
            .max(self.approximate[2])
            .clamp(0.0, 1.0)
    }

    /// Select the next batch of hemisphere ray indices to trace for this update (`calc_sky_hemi_value`).
    pub fn next_hemi_ray_indices(&mut self, count: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(count);
        for _ in 0..count {
            let sample = if self.result_count < LT_HEMISAMPLES {
                let s = self.result_count;
                self.result_count += 1;
                s
            } else {
                let s = self.result_iterator % LT_HEMISAMPLES;
                self.result_iterator = self.result_iterator.wrapping_add(1);
                s
            };
            indices.push(sample);
        }
        indices
    }

    /// Recompute `hemi_value` and sky contribution in `hemi_cube` after updating `ray_clear`
    /// (`LightTrack.cpp:435-447`).
    pub fn accumulate_sky_hemi(&mut self, sky_scale: f32) {
        self.hemi_cube = [0.0; 6];
        let mut pass = 0usize;
        for (&dir_raw, &clear) in HEMI_SAMPLE_DIRS
            .iter()
            .zip(self.ray_clear.iter())
            .take(self.result_count)
        {
            if clear {
                pass += 1;
                let dir = normalize3_or(dir_raw, [0.0, 1.0, 0.0]);
                accum_hemi_cube(&mut self.hemi_cube, dir, sky_scale);
            }
        }
        let denom = self.result_count.max(1) as f32;
        self.hemi_value = (pass as f32 / denom) * sky_scale;
    }

    /// Update light visibility cache and recompute `approximate` + `hemi_cube` (`LightTrack.cpp:216-292,489-537`).
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_lighting<F>(
        &mut self,
        frame: u32,
        dt: f32,
        object_pos: [f32; 3],
        env_ambient: [f32; 3],
        env_hemi: [f32; 3],
        env_sun: [f32; 3],
        cfg: &LightTrackConfig,
        mut ray_is_clear: F,
    ) where
        F: FnMut(u64, [f32; 3], [f32; 3], f32) -> bool,
    {
        let first_time = self.result_count == 0;
        self.accumulate_sky_hemi(cfg.dhemi_sky_scale);

        self.tracked_lights
            .retain(|entry| entry.last_touched_frame == frame);

        let mut accum = [
            env_ambient[0] + env_hemi[0] * self.hemi_smooth + env_sun[0] * self.sun_smooth,
            env_ambient[1] + env_hemi[1] * self.hemi_smooth + env_sun[1] * self.sun_smooth,
            env_ambient[2] + env_hemi[2] * self.hemi_smooth + env_sun[2] * self.sun_smooth,
        ];

        let mut hemi_cube_light = [0.0f32; 6];
        let mut lacc = [0.0f32; 3];

        for entry in &mut self.tracked_lights {
            let to_obj = [
                object_pos[0] - entry.position[0],
                object_pos[1] - entry.position[1],
                object_pos[2] - entry.position[2],
            ];
            let d = dot3(to_obj, to_obj).sqrt();
            let dir_from_light = if d > 1e-6 {
                [to_obj[0] / d, to_obj[1] / d, to_obj[2] / d]
            } else {
                [0.0, 1.0, 0.0]
            };

            let amount = if d > 1e-6 && !ray_is_clear(entry.id, entry.position, dir_from_light, d) {
                -cfg.lt_dec
            } else {
                cfg.lt_inc
            };
            entry.test = (entry.test + amount * dt.max(0.0)).clamp(-0.5, 1.0);
            entry.energy = 0.9 * entry.energy + 0.1 * entry.test;

            let intensity =
                (entry.color_rgb[0] + entry.color_rgb[1] + entry.color_rgb[2]) / 3.0;
            if entry.energy * intensity <= 1e-5 {
                continue;
            }

            let mut l_color = [
                entry.color_rgb[0] * (entry.energy * 0.5),
                entry.color_rgb[1] * (entry.energy * 0.5),
                entry.color_rgb[2] * (entry.energy * 0.5),
            ];
            if !entry.is_static {
                l_color[0] *= 0.5;
                l_color[1] *= 0.5;
                l_color[2] *= 0.5;
            }

            let denom = entry.attenuation0 + entry.attenuation1 * d + entry.attenuation2 * d * d;
            let raw_att = if denom.abs() > 1e-6 {
                1.0 / denom - d * entry.falloff
            } else {
                0.0
            };
            let a = raw_att.max(0.0) * if entry.is_static { 1.0 } else { 2.0 };

            let dir_to_light = normalize3_or([-to_obj[0], -to_obj[1], -to_obj[2]], [0.0, 1.0, 0.0]);
            let koef = (l_color[0] + l_color[1] + l_color[2]) / 3.0 * a * cfg.dhemi_light_scale;
            accum_hemi_cube(&mut hemi_cube_light, dir_to_light, koef);

            lacc[0] += l_color[0] * a;
            lacc[1] += l_color[1] * a;
            lacc[2] += l_color[2] * a;
        }

        const MIN_HEMI: f32 = 1.0 / 255.0;
        let hemi_light = (lacc[0] + lacc[1] + lacc[2]) / 3.0 * cfg.dhemi_light_scale;
        self.hemi_value = (self.hemi_value + hemi_light).max(MIN_HEMI);

        for i in 0..hemi_cube_faces::NUM_FACES {
            let opp = (i + hemi_cube_faces::NUM_FACES / 2) % hemi_cube_faces::NUM_FACES;
            self.hemi_cube[i] +=
                hemi_cube_light[i] * (1.0 - cfg.dhemi_light_flow) + cfg.dhemi_light_flow * hemi_cube_light[opp];
            self.hemi_cube[i] = self.hemi_cube[i].max(MIN_HEMI);
        }

        accum[0] += lacc[0];
        accum[1] += lacc[1];
        accum[2] += lacc[2];

        if first_time {
            self.hemi_smooth = self.hemi_value;
            self.hemi_cube_smooth = self.hemi_cube;
        }
        self.step_smooth(dt, cfg.lt_smooth);
        self.approximate = accum;
    }

    /// Exponential temporal smoothing (`CROS_impl::update_smooth`, `LightTrack.cpp:365-373`).
    pub fn step_smooth(&mut self, dt: f32, lt_smooth: f32) {
        let l_f = (dt.max(0.0) * lt_smooth).clamp(0.0, 1.0);
        let l_i = 1.0 - l_f;
        self.hemi_smooth = self.hemi_value * l_f + self.hemi_smooth * l_i;
        self.sun_smooth = self.sun_value * l_f + self.sun_smooth * l_i;
        for i in 0..hemi_cube_faces::NUM_FACES {
            self.hemi_cube_smooth[i] =
                self.hemi_cube[i] * l_f + self.hemi_cube_smooth[i] * l_i;
        }
    }

    /// Pack smoothed hemisphere irradiance and scalar material properties for `R_hemi`
    /// (`r4.h:259-301`, `R_Backend_hemi.cpp`).
    pub fn to_hemi_gpu_params(
        &self,
        material_id: f32,
        hotness: [f32; 4],
        glowing: [f32; 4],
    ) -> HemiMaterialGpuParams {
        use hemi_cube_faces::*;
        HemiMaterialGpuParams {
            pos_faces: [
                self.hemi_cube_smooth[POS_X],
                self.hemi_cube_smooth[POS_Y],
                self.hemi_cube_smooth[POS_Z],
                0.0,
            ],
            neg_faces: [
                self.hemi_cube_smooth[NEG_X],
                self.hemi_cube_smooth[NEG_Y],
                self.hemi_cube_smooth[NEG_Z],
                0.0,
            ],
            material: [
                0.75 * self.hemi_smooth,
                0.75 * self.sun_smooth,
                0.0,
                encode_hemi_material_id(material_id),
            ],
            hotness,
            glowing,
        }
    }
}

// ============================================================================
// Canonical Deferred Light Volumes (`du_sphere`, `du_cone`, `du_sphere_part`, `du_box`)
// ============================================================================

pub const DU_SPHERE_NUMVERTEX: usize = 92;
pub const DU_SPHERE_NUMFACES: usize = 180;

/// Unit sphere vertices used by point & reflected light volume passes (`du_sphere.cpp:8-101`).
pub const DU_SPHERE_VERTICES: [[f32; 3]; DU_SPHERE_NUMVERTEX] = [
    [0.0000, 1.0000, 0.0000],
    [0.8944, 0.4472, 0.0000],
    [0.2764, 0.4472, 0.8507],
    [-0.7236, 0.4472, 0.5257],
    [-0.7236, 0.4472, -0.5257],
    [0.2764, 0.4472, -0.8507],
    [0.7236, -0.4472, 0.5257],
    [-0.2764, -0.4472, 0.8507],
    [-0.8944, -0.4472, -0.0000],
    [-0.2764, -0.4472, -0.8507],
    [0.7236, -0.4472, -0.5257],
    [0.0000, -1.0000, 0.0000],
    [0.3607, 0.9327, 0.0000],
    [0.6729, 0.7397, 0.0000],
    [0.1115, 0.9327, 0.3431],
    [0.2079, 0.7397, 0.6399],
    [-0.2918, 0.9327, 0.2120],
    [-0.5444, 0.7397, 0.3955],
    [-0.2918, 0.9327, -0.2120],
    [-0.5444, 0.7397, -0.3955],
    [0.1115, 0.9327, -0.3431],
    [0.2079, 0.7397, -0.6399],
    [0.7844, 0.5168, 0.3431],
    [0.5687, 0.5168, 0.6399],
    [-0.0839, 0.5168, 0.8520],
    [-0.4329, 0.5168, 0.7386],
    [-0.8362, 0.5168, 0.1835],
    [-0.8362, 0.5168, -0.1835],
    [-0.4329, 0.5168, -0.7386],
    [-0.0839, 0.5168, -0.8520],
    [0.5687, 0.5168, -0.6399],
    [0.7844, 0.5168, -0.3431],
    [0.9647, 0.1561, 0.2120],
    [0.9051, -0.1561, 0.3955],
    [0.0965, 0.1561, 0.9830],
    [-0.0965, -0.1561, 0.9830],
    [-0.9051, 0.1561, 0.3955],
    [-0.9647, -0.1561, 0.2120],
    [-0.6558, 0.1561, -0.7386],
    [-0.4998, -0.1561, -0.8520],
    [0.4998, 0.1561, -0.8520],
    [0.6558, -0.1561, -0.7386],
    [0.9647, 0.1561, -0.2120],
    [0.9051, -0.1561, -0.3955],
    [0.4998, 0.1561, 0.8520],
    [0.6558, -0.1561, 0.7386],
    [-0.6558, 0.1561, 0.7386],
    [-0.4998, -0.1561, 0.8520],
    [-0.9051, 0.1561, -0.3955],
    [-0.9647, -0.1561, -0.2120],
    [0.0965, 0.1561, -0.9830],
    [-0.0965, -0.1561, -0.9830],
    [0.4329, -0.5168, 0.7386],
    [0.0839, -0.5168, 0.8520],
    [-0.5687, -0.5168, 0.6399],
    [-0.7844, -0.5168, 0.3431],
    [-0.7844, -0.5168, -0.3431],
    [-0.5687, -0.5168, -0.6399],
    [0.0839, -0.5168, -0.8520],
    [0.4329, -0.5168, -0.7386],
    [0.8362, -0.5168, -0.1835],
    [0.8362, -0.5168, 0.1835],
    [0.2918, -0.9327, 0.2120],
    [0.5444, -0.7397, 0.3955],
    [-0.1115, -0.9327, 0.3431],
    [-0.2079, -0.7397, 0.6399],
    [-0.3607, -0.9327, -0.0000],
    [-0.6729, -0.7397, -0.0000],
    [-0.1115, -0.9327, -0.3431],
    [-0.2079, -0.7397, -0.6399],
    [0.2918, -0.9327, -0.2120],
    [0.5444, -0.7397, -0.3955],
    [0.4795, 0.8054, 0.3484],
    [-0.1832, 0.8054, 0.5637],
    [-0.5927, 0.8054, -0.0000],
    [-0.1832, 0.8054, -0.5637],
    [0.4795, 0.8054, -0.3484],
    [0.9855, -0.1699, 0.0000],
    [0.3045, -0.1699, 0.9372],
    [-0.7973, -0.1699, 0.5792],
    [-0.7973, -0.1699, -0.5792],
    [0.3045, -0.1699, -0.9372],
    [0.7973, 0.1699, 0.5792],
    [-0.3045, 0.1699, 0.9372],
    [-0.9855, 0.1699, -0.0000],
    [-0.3045, 0.1699, -0.9372],
    [0.7973, 0.1699, -0.5792],
    [0.1832, -0.8054, 0.5637],
    [-0.4795, -0.8054, 0.3484],
    [-0.4795, -0.8054, -0.3484],
    [0.1832, -0.8054, -0.5637],
    [0.5927, -0.8054, 0.0000],
];

/// Unit sphere triangle index list (`du_sphere_faces`, 180 triangles / 540 indices).
pub const DU_SPHERE_INDICES: [u16; DU_SPHERE_NUMFACES * 3] = [
    14, 12, 0, 72, 13, 12, 14, 72, 12, 15, 72, 14, 22, 1, 13, 72, 22, 13, 23, 22, 72, 15, 23, 72,
    2, 23, 15, 16, 14, 0, 73, 15, 14, 16, 73, 14, 17, 73, 16, 24, 2, 15, 73, 24, 15, 25, 24, 73,
    17, 25, 73, 3, 25, 17, 18, 16, 0, 74, 17, 16, 18, 74, 16, 19, 74, 18, 26, 3, 17, 74, 26, 17,
    27, 26, 74, 19, 27, 74, 4, 27, 19, 20, 18, 0, 75, 19, 18, 20, 75, 18, 21, 75, 20, 28, 4, 19,
    75, 28, 19, 29, 28, 75, 21, 29, 75, 5, 29, 21, 12, 20, 0, 76, 21, 20, 12, 76, 20, 13, 76, 12,
    30, 5, 21, 76, 30, 21, 31, 30, 76, 13, 31, 76, 1, 31, 13, 32, 42, 1, 77, 43, 42, 32, 77, 42,
    33, 77, 32, 60, 10, 43, 77, 60, 43, 61, 60, 77, 33, 61, 77, 6, 61, 33, 34, 44, 2, 78, 45, 44,
    34, 78, 44, 35, 78, 34, 52, 6, 45, 78, 52, 45, 53, 52, 78, 35, 53, 78, 7, 53, 35, 36, 46, 3,
    79, 47, 46, 36, 79, 46, 37, 79, 36, 54, 7, 47, 79, 54, 47, 55, 54, 79, 37, 55, 79, 8, 55, 37,
    38, 48, 4, 80, 49, 48, 38, 80, 48, 39, 80, 38, 56, 8, 49, 80, 56, 49, 57, 56, 80, 39, 57, 80,
    9, 57, 39, 40, 50, 5, 81, 51, 50, 40, 81, 50, 41, 81, 40, 58, 9, 51, 81, 58, 51, 59, 58, 81,
    41, 59, 81, 10, 59, 41, 33, 45, 6, 82, 44, 45, 33, 82, 45, 32, 82, 33, 23, 2, 44, 82, 23, 44,
    22, 23, 82, 32, 22, 82, 1, 22, 32, 35, 47, 7, 83, 46, 47, 35, 83, 47, 34, 83, 35, 25, 3, 46,
    83, 25, 46, 24, 25, 83, 34, 24, 83, 2, 24, 34, 37, 49, 8, 84, 48, 49, 37, 84, 49, 36, 84, 37,
    27, 4, 48, 84, 27, 48, 26, 27, 84, 36, 26, 84, 3, 26, 36, 39, 51, 9, 85, 50, 51, 39, 85, 51,
    38, 85, 39, 29, 5, 50, 85, 29, 50, 28, 29, 85, 38, 28, 85, 4, 28, 38, 41, 43, 10, 86, 42, 43,
    41, 86, 43, 40, 86, 41, 31, 1, 42, 86, 31, 42, 30, 31, 86, 40, 30, 86, 5, 30, 40, 62, 64, 11,
    87, 65, 64, 62, 87, 64, 63, 87, 62, 53, 7, 65, 87, 53, 65, 52, 53, 87, 63, 52, 87, 6, 52, 63,
    64, 66, 11, 88, 67, 66, 64, 88, 66, 65, 88, 64, 55, 8, 67, 88, 55, 67, 54, 55, 88, 65, 54, 88,
    7, 54, 65, 66, 68, 11, 89, 69, 68, 66, 89, 68, 67, 89, 66, 57, 9, 69, 89, 57, 69, 56, 57, 89,
    67, 56, 89, 8, 56, 67, 68, 70, 11, 90, 71, 70, 68, 90, 70, 69, 90, 68, 59, 10, 71, 90, 59, 71,
    58, 59, 90, 69, 58, 90, 9, 58, 69, 70, 62, 11, 91, 63, 62, 70, 91, 62, 71, 91, 70, 61, 6, 63,
    91, 61, 63, 60, 61, 91, 71, 60, 91, 10, 60, 71,
];

pub const DU_CONE_NUMVERTEX: usize = 18;
pub const DU_CONE_NUMFACES: usize = 32;

/// Unit spotlight cone vertices (`z in [0, 1]`, rim radius `0.5`, `du_cone.cpp:8-28`).
pub const DU_CONE_VERTICES: [[f32; 3]; DU_CONE_NUMVERTEX] = [
    [0.0000, 0.0000, 0.0000],
    [0.5000, 0.0000, 1.0000],
    [0.4619, 0.1913, 1.0000],
    [0.3536, 0.3536, 1.0000],
    [0.1913, 0.4619, 1.0000],
    [-0.0000, 0.5000, 1.0000],
    [-0.1913, 0.4619, 1.0000],
    [-0.3536, 0.3536, 1.0000],
    [-0.4619, 0.1913, 1.0000],
    [-0.5000, -0.0000, 1.0000],
    [-0.4619, -0.1913, 1.0000],
    [-0.3536, -0.3536, 1.0000],
    [-0.1913, -0.4619, 1.0000],
    [0.0000, -0.5000, 1.0000],
    [0.1913, -0.4619, 1.0000],
    [0.3536, -0.3536, 1.0000],
    [0.4619, -0.1913, 1.0000],
    [0.0000, 0.0000, 1.0000],
];

/// Unit spotlight cone triangle index list (`du_cone_faces`, 32 triangles / 96 indices).
pub const DU_CONE_INDICES: [u16; DU_CONE_NUMFACES * 3] = [
    0, 2, 1, 0, 3, 2, 0, 4, 3, 0, 5, 4, 0, 6, 5, 0, 7, 6, 0, 8, 7, 0, 9, 8, 0, 10, 9, 0, 11, 10,
    0, 12, 11, 0, 13, 12, 0, 14, 13, 0, 15, 14, 0, 16, 15, 0, 1, 16, 17, 1, 2, 17, 2, 3, 17, 3, 4,
    17, 4, 5, 17, 5, 6, 17, 6, 7, 17, 7, 8, 17, 8, 9, 17, 9, 10, 17, 10, 11, 17, 11, 12, 17, 12,
    13, 17, 13, 14, 17, 14, 15, 17, 15, 16, 17, 16, 1,
];

pub const DU_SPHERE_PART_NUMVERTEX: usize = 82;
pub const DU_SPHERE_PART_NUMFACES: usize = 160;

/// Omnipart cubemap face frustum-cap pyramid+sphere-sector vertices (`du_sphere_part.cpp:8-92`).
pub const DU_SPHERE_PART_VERTICES: [[f32; 3]; DU_SPHERE_PART_NUMVERTEX] = [
    [-0.288675, -0.288675, 0.288675],
    [0.288675, -0.288675, 0.288675],
    [-0.288675, 0.288675, 0.288675],
    [0.288675, 0.288675, 0.288675],
    [0.000000, 0.000000, 0.500000],
    [0.000000, -0.353553, 0.353553],
    [0.353553, 0.000000, 0.353553],
    [0.000000, 0.353553, 0.353553],
    [-0.353553, 0.000000, 0.353553],
    [-0.204124, -0.204124, 0.408248],
    [0.204124, -0.204124, 0.408248],
    [0.204124, 0.204124, 0.408248],
    [-0.204124, 0.204124, 0.408248],
    [-0.166667, -0.333333, 0.333333],
    [0.000000, -0.223607, 0.447214],
    [-0.223607, 0.000000, 0.447214],
    [-0.333333, -0.166667, 0.333333],
    [0.333333, -0.166667, 0.333333],
    [0.223607, 0.000000, 0.447214],
    [0.166667, -0.333333, 0.333333],
    [0.166667, 0.333333, 0.333333],
    [0.000000, 0.223607, 0.447214],
    [0.333333, 0.166667, 0.333333],
    [-0.333333, 0.166667, 0.333333],
    [-0.166667, 0.333333, 0.333333],
    [-0.257248, -0.257248, 0.342997],
    [-0.098058, -0.294174, 0.392232],
    [-0.117851, -0.117851, 0.471405],
    [-0.294174, -0.098058, 0.392232],
    [0.257248, -0.257248, 0.342997],
    [0.294174, -0.098058, 0.392232],
    [0.117851, -0.117851, 0.471405],
    [0.098058, -0.294174, 0.392232],
    [0.257248, 0.257248, 0.342997],
    [0.098058, 0.294174, 0.392232],
    [0.117851, 0.117851, 0.471405],
    [0.294174, 0.098058, 0.392232],
    [-0.257248, 0.257248, 0.342997],
    [-0.294174, 0.098058, 0.392232],
    [-0.117851, 0.117851, 0.471405],
    [-0.098058, 0.294174, 0.392232],
    [-0.234261, -0.312348, 0.312348],
    [-0.185695, -0.278543, 0.371391],
    [-0.278543, -0.185695, 0.371391],
    [-0.312348, -0.234261, 0.312348],
    [0.000000, -0.300000, 0.400000],
    [-0.109109, -0.218218, 0.436436],
    [-0.087039, -0.348155, 0.348155],
    [-0.121268, 0.000000, 0.485071],
    [-0.218218, -0.109109, 0.436436],
    [0.000000, -0.121268, 0.485071],
    [-0.348155, -0.087039, 0.348155],
    [-0.300000, 0.000000, 0.400000],
    [0.312348, -0.234261, 0.312348],
    [0.278543, -0.185695, 0.371391],
    [0.185695, -0.278543, 0.371391],
    [0.234261, -0.312348, 0.312348],
    [0.300000, 0.000000, 0.400000],
    [0.218218, -0.109109, 0.436436],
    [0.348155, -0.087039, 0.348155],
    [0.109109, -0.218218, 0.436436],
    [0.121268, 0.000000, 0.485071],
    [0.087039, -0.348155, 0.348155],
    [0.234261, 0.312348, 0.312348],
    [0.185695, 0.278543, 0.371391],
    [0.278543, 0.185695, 0.371391],
    [0.312348, 0.234261, 0.312348],
    [0.000000, 0.300000, 0.400000],
    [0.109109, 0.218218, 0.436436],
    [0.087039, 0.348155, 0.348155],
    [0.218218, 0.109109, 0.436436],
    [0.000000, 0.121268, 0.485071],
    [0.348155, 0.087039, 0.348155],
    [-0.312348, 0.234261, 0.312348],
    [-0.278543, 0.185695, 0.371391],
    [-0.185695, 0.278543, 0.371391],
    [-0.234261, 0.312348, 0.312348],
    [-0.218218, 0.109109, 0.436436],
    [-0.348155, 0.087039, 0.348155],
    [-0.109109, 0.218218, 0.436436],
    [-0.087039, 0.348155, 0.348155],
    [0.000000, 0.000000, 0.000000],
];

/// Omnipart frustum-cap triangle indices (`du_sphere_part_faces`, 160 triangles / 480 indices).
pub const DU_SPHERE_PART_INDICES: [u16; DU_SPHERE_PART_NUMFACES * 3] = [
    0, 41, 25, 25, 44, 0, 13, 42, 25, 25, 41, 13, 9, 43, 25, 25, 42, 9, 16, 44, 25, 25, 43, 16,
    5, 45, 26, 26, 47, 5, 14, 46, 26, 26, 45, 14, 9, 42, 26, 26, 46, 9, 13, 47, 26, 26, 42, 13,
    4, 48, 27, 27, 50, 4, 15, 49, 27, 27, 48, 15, 9, 46, 27, 27, 49, 9, 14, 50, 27, 27, 46, 14,
    8, 51, 28, 28, 52, 8, 16, 43, 28, 28, 51, 16, 9, 49, 28, 28, 43, 9, 15, 52, 28, 28, 49, 15,
    1, 53, 29, 29, 56, 1, 17, 54, 29, 29, 53, 17, 10, 55, 29, 29, 54, 10, 19, 56, 29, 29, 55, 19,
    6, 57, 30, 30, 59, 6, 18, 58, 30, 30, 57, 18, 10, 54, 30, 30, 58, 10, 17, 59, 30, 30, 54, 17,
    4, 50, 31, 31, 61, 4, 14, 60, 31, 31, 50, 14, 10, 58, 31, 31, 60, 10, 18, 61, 31, 31, 58, 18,
    5, 62, 32, 32, 45, 5, 19, 55, 32, 32, 62, 19, 10, 60, 32, 32, 55, 10, 14, 45, 32, 32, 60, 14,
    3, 63, 33, 33, 66, 3, 20, 64, 33, 33, 63, 20, 11, 65, 33, 33, 64, 11, 22, 66, 33, 33, 65, 22,
    7, 67, 34, 34, 69, 7, 21, 68, 34, 34, 67, 21, 11, 64, 34, 34, 68, 11, 20, 69, 34, 34, 64, 20,
    4, 61, 35, 35, 71, 4, 18, 70, 35, 35, 61, 18, 11, 68, 35, 35, 70, 11, 21, 71, 35, 35, 68, 21,
    6, 72, 36, 36, 57, 6, 22, 65, 36, 36, 72, 22, 11, 70, 36, 36, 65, 11, 18, 57, 36, 36, 70, 18,
    2, 73, 37, 37, 76, 2, 23, 74, 37, 37, 73, 23, 12, 75, 37, 37, 74, 12, 24, 76, 37, 37, 75, 24,
    8, 52, 38, 38, 78, 8, 15, 77, 38, 38, 52, 15, 12, 74, 38, 38, 77, 12, 23, 78, 38, 38, 74, 23,
    4, 71, 39, 39, 48, 4, 21, 79, 39, 39, 71, 21, 12, 77, 39, 39, 79, 12, 15, 48, 39, 39, 77, 15,
    7, 80, 40, 40, 67, 7, 24, 75, 40, 40, 80, 24, 12, 79, 40, 40, 75, 12, 21, 67, 40, 40, 79, 21,
    41, 0, 81, 0, 44, 81, 13, 41, 81, 44, 16, 81, 5, 47, 81, 47, 13, 81, 51, 8, 81, 16, 51, 81,
    53, 1, 81, 1, 56, 81, 17, 53, 81, 56, 19, 81, 6, 59, 81, 59, 17, 81, 62, 5, 81, 19, 62, 81,
    63, 3, 81, 3, 66, 81, 20, 63, 81, 66, 22, 81, 7, 69, 81, 69, 20, 81, 72, 6, 81, 22, 72, 81,
    73, 2, 81, 2, 76, 81, 23, 73, 81, 76, 24, 81, 8, 78, 81, 78, 23, 81, 80, 7, 81, 24, 80, 81,
];

pub const DU_BOX_NUMVERTEX: usize = 8;
pub const DU_BOX_NUMFACES: usize = 12;
pub const DU_BOX_NUMVERTEX2: usize = 36;

/// Unit cube indexed corner vertices (`[-0.5, +0.5]^3`, `du_box.cpp:23-32`).
pub const DU_BOX_VERTICES: [[f32; 3]; DU_BOX_NUMVERTEX] = [
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
];

/// Unit cube indexed triangle list (`du_box_faces`, 12 triangles / 36 indices).
pub const DU_BOX_INDICES: [u16; DU_BOX_NUMFACES * 3] = [
    0, 1, 2, 2, 3, 0, 3, 2, 6, 6, 7, 3, 6, 4, 5, 6, 5, 7, 4, 1, 5, 1, 0, 5, 3, 5, 0, 3, 7, 5, 1, 4,
    6, 1, 6, 2,
];

/// Unit cube unindexed 36-vertex triangle list (`du_box_vertices2`, `du_box.cpp:8-21`).
pub const DU_BOX_VERTICES2: [[f32; 3]; DU_BOX_NUMVERTEX2] = [
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
];

pub const DU_CYLINDER_NUMVERTEX: usize = 26;
pub const DU_CYLINDER_NUMFACES: usize = 48;
pub const DU_CYLINDER_NUMLINES: usize = 30;

/// Unit Z-aligned closed cylinder vertices (radius `0.5`, height `1.0`, `du_cylinder.cpp:8-35`).
pub const DU_CYLINDER_VERTICES: [[f32; 3]; DU_CYLINDER_NUMVERTEX] = [
    [0.5000, 0.0000, 0.5000],
    [0.5000, -0.0000, -0.5000],
    [0.4330, -0.2500, -0.5000],
    [0.4330, -0.2500, 0.5000],
    [0.2500, -0.4330, -0.5000],
    [0.2500, -0.4330, 0.5000],
    [0.0000, -0.5000, -0.5000],
    [-0.0000, -0.5000, 0.5000],
    [-0.2500, -0.4330, -0.5000],
    [-0.2500, -0.4330, 0.5000],
    [-0.4330, -0.2500, -0.5000],
    [-0.4330, -0.2500, 0.5000],
    [-0.5000, -0.0000, -0.5000],
    [-0.5000, 0.0000, 0.5000],
    [-0.4330, 0.2500, -0.5000],
    [-0.4330, 0.2500, 0.5000],
    [-0.2500, 0.4330, -0.5000],
    [-0.2500, 0.4330, 0.5000],
    [0.0000, 0.5000, -0.5000],
    [-0.0000, 0.5000, 0.5000],
    [0.2500, 0.4330, -0.5000],
    [0.2500, 0.4330, 0.5000],
    [0.4330, 0.2500, -0.5000],
    [0.4330, 0.2500, 0.5000],
    [-0.0000, 0.0000, 0.5000],
    [0.0000, -0.0000, -0.5000],
];

/// Unit cylinder triangle index list (`du_cylinder_faces`, 48 triangles / 144 indices).
pub const DU_CYLINDER_INDICES: [u16; DU_CYLINDER_NUMFACES * 3] = [
    0, 2, 1, 3, 2, 0, 3, 4, 2, 5, 4, 3, 5, 6, 4, 7, 6, 5, 8, 6, 9, 9, 6, 7, 10, 8, 11, 11, 8, 9,
    12, 10, 13, 13, 10, 11, 14, 12, 15, 12, 13, 15, 16, 14, 17, 17, 14, 15, 18, 16, 19, 19, 16, 17,
    19, 20, 18, 21, 20, 19, 21, 22, 20, 23, 22, 21, 23, 1, 22, 23, 0, 1, 23, 24, 0, 21, 24, 23, 19,
    24, 21, 17, 24, 19, 15, 24, 17, 13, 24, 15, 11, 24, 13, 9, 24, 11, 7, 24, 9, 5, 24, 7, 3, 24, 5,
    0, 24, 3, 8, 25, 6, 10, 25, 8, 12, 25, 10, 14, 25, 12, 16, 25, 14, 18, 25, 16, 20, 25, 18, 22,
    25, 20, 1, 25, 22, 2, 25, 1, 4, 25, 2, 6, 25, 4,
];

/// Unit cylinder wireframe line index list (`du_cylinder_lines`, 30 lines / 60 indices).
pub const DU_CYLINDER_LINES: [u16; DU_CYLINDER_NUMLINES * 2] = [
    0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 1, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14,
    16, 16, 18, 18, 20, 20, 22, 22, 1, 0, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 17, 17,
    19, 19, 21, 21, 23, 23, 0,
];

/// Canonical slice count for volumetric spotlight proxy stack (`r4_rendertarget.h:12`).
pub const VOLUMETRIC_SLICES: usize = 100;

/// Volumetric spotlight slice quad mesh (`400` vertices, `600` indices).
#[derive(Debug, Clone, PartialEq)]
pub struct VolumetricSlicesMesh {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u16>,
}

/// Build the 100-slice `[0..1, 0..1, t]` quad stack matching
/// `CRenderTarget::accum_volumetric_geom_create` (`r4_rendertarget_accum_spot_geom.cpp:119-203`).
pub fn build_volumetric_slices_mesh() -> VolumetricSlicesMesh {
    let mut vertices = Vec::with_capacity(VOLUMETRIC_SLICES * 4);
    let mut indices = Vec::with_capacity(VOLUMETRIC_SLICES * 6);
    let mut t = 0.0f32;
    let dt = 1.0f32 / (VOLUMETRIC_SLICES - 1) as f32;
    for i in 0..VOLUMETRIC_SLICES {
        vertices.push([0.0, 0.0, t]);
        vertices.push([0.0, 1.0, t]);
        vertices.push([1.0, 0.0, t]);
        vertices.push([1.0, 1.0, t]);
        t += dt;

        let base = (i * 4) as u16;
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }
    VolumetricSlicesMesh { vertices, indices }
}

/// Encode scalar material ID for `L_material.w` matching `CRender::apply_lmaterial` (`r4.h:295`).
#[inline]
pub fn encode_hemi_material_id(mtl: f32) -> f32 {
    if mtl < 5.0 {
        (mtl + 0.5) / 4.0
    } else {
        mtl
    }
}

/// GPU constant payload matching `R_hemi` (`R_Backend_hemi.cpp`, `r4.h:291-301`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HemiMaterialGpuParams {
    pub pos_faces: [f32; 4],
    pub neg_faces: [f32; 4],
    pub material: [f32; 4],
    pub hotness: [f32; 4],
    pub glowing: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<HemiMaterialGpuParams>() == 80);

/// Configuration for `CRenderTarget::accum_volumetric` (`r4_rendertarget_accum_spot.cpp:415-489`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumetricLightConfig {
    pub volumetric_distance: f32,
    pub volumetric_quality: f32,
    pub volumetric_intensity: f32,
    /// `ps_ssfx_volumetric`: `(force_enable, intensity, quality, reserved)`.
    pub ssfx_volumetric: [f32; 4],
    pub is_player_light: bool,
    pub spot_accum: SpotAccumConfig,
}

impl Default for VolumetricLightConfig {
    fn default() -> Self {
        Self {
            volumetric_distance: 1.0,
            volumetric_quality: 1.0,
            volumetric_intensity: 1.0,
            ssfx_volumetric: [1.0, 1.0, 3.0, 1.0],
            is_player_light: false,
            spot_accum: SpotAccumConfig::default(),
        }
    }
}

/// GPU shader constants and draw slice counts for `accum_volumetric`
/// (`r4_rendertarget_accum_spot.cpp:386-626`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumetricSpotParams {
    pub m_shadow: XRayMatrix,
    pub m_lmap_rows: [[f32; 4]; 2],
    /// `[view_x, view_y, view_z, att_factor]` (`Ldynamic_pos`).
    pub dynamic_pos: [f32; 4],
    /// `[r, g, b, spec]` (`Ldynamic_color`).
    pub dynamic_color: [f32; 4],
    /// View-space AABB minimum `[x1, y1, z1, 0.0]` (`vMinBounds`).
    pub v_min_bounds: [f32; 4],
    /// View-space AABB maximum `[x2, y2, z1 + (z2 - z1) / quality_normalized, 0.0]` (`vMaxBounds`).
    pub v_max_bounds: [f32; 4],
    /// 6 camera clip-space user clip planes `FrustumClipPlane[0..6]`.
    pub frustum_clip_planes: [[f32; 4]; 6],
    pub num_slices: u32,
    pub vertex_count: u32,
    pub primitive_count: u32,
    pub quality_normalized: f32,
}

/// Transform world-space frustum clip planes to negated camera clip space matching
/// `r4_rendertarget_accum_spot.cpp:564-576` (`PlaneTransform = transpose(mInvFullTransform)`).
pub fn transform_volumetric_clip_planes(
    inv_full_transform: &XRayMatrix,
    planes: &[crate::hom::Plane],
) -> [[f32; 4]; 6] {
    let mut out = [[0.0f32, 0.0, 0.0, 1.0]; 6];
    // Transpose(inv_full_transform) * row_vec(plane): out[col] = sum_row plane[row] * inv[col][row]
    for (i, plane) in planes.iter().take(6).enumerate() {
        let p = [plane.n[0], plane.n[1], plane.n[2], plane.d];
        for col in 0..4 {
            let v = p[0] * inv_full_transform[col][0]
                + p[1] * inv_full_transform[col][1]
                + p[2] * inv_full_transform[col][2]
                + p[3] * inv_full_transform[col][3];
            out[i][col] = -v;
        }
    }
    out
}

/// Compute all volumetric spotlight shader inputs (`vMinBounds`, `vMaxBounds`, `FrustumClipPlane`,
/// slice counts & attenuated intensity) matching `r4_rendertarget_accum_spot.cpp:386-626`.
#[allow(clippy::too_many_arguments)]
pub fn compute_volumetric_spot_params(
    light: &DynamicLight,
    smap_xform: &LightSmapXform,
    atlas_rect: SmapRect,
    camera_pos: [f32; 3],
    view_matrix: &XRayMatrix,
    inv_view_matrix: &XRayMatrix,
    inv_full_transform: &XRayMatrix,
    vol_cfg: &VolumetricLightConfig,
) -> VolumetricSpotParams {
    let spot = compute_spot_accum_constants(
        light,
        smap_xform,
        atlas_rect,
        camera_pos,
        view_matrix,
        inv_view_matrix,
        &vol_cfg.spot_accum,
    );

    let mut frustum = ClipFrustum::from_view_proj(&smap_xform.combine, frustum_planes::ALL);
    if frustum.planes.len() >= 6 {
        let far_d = frustum.planes[4].d;
        let near_d = frustum.planes[5].d;
        frustum.planes[4].d -= (far_d + near_d) * (1.0 - vol_cfg.volumetric_distance);
    }
    let frustum_clip_planes = transform_volumetric_clip_planes(inv_full_transform, &frustum.planes);

    let (sp_center, sp_radius) = light.spatial_sphere();
    let scaled_radius = sp_radius * vol_cfg.volumetric_distance;
    let pt_world = [
        (sp_center[0] - light.position[0]) * vol_cfg.volumetric_distance + light.position[0],
        (sp_center[1] - light.position[1]) * vol_cfg.volumetric_distance + light.position[1],
        (sp_center[2] - light.position[2]) * vol_cfg.volumetric_distance + light.position[2],
    ];
    let pt_view = matrix_transform_point(view_matrix, pt_world);
    let aabb_min = [
        pt_view[0] - scaled_radius,
        pt_view[1] - scaled_radius,
        pt_view[2] - scaled_radius,
    ];
    let aabb_max = [
        pt_view[0] + scaled_radius,
        pt_view[1] + scaled_radius,
        pt_view[2] + scaled_radius,
    ];

    let lod = light.compute_lod(
        camera_pos,
        vol_cfg.spot_accum.slight_fade,
        vol_cfg.spot_accum.r_ssa_glod_start,
        vol_cfg.spot_accum.r_ssa_glod_end,
    );
    let mut clr = [
        light.color_rgb[0] * vol_cfg.volumetric_distance,
        light.color_rgb[1] * vol_cfg.volumetric_distance,
        light.color_rgb[2] * vol_cfg.volumetric_distance,
    ];

    let (num_slices, quality_normalized) = if vol_cfg.ssfx_volumetric[0] <= 0.0 {
        let slices = ((VOLUMETRIC_SLICES as f32 * vol_cfg.volumetric_quality) as i32).max(10) as u32;
        let q = (slices as f32 / VOLUMETRIC_SLICES as f32).max(1e-4);
        let scale = (vol_cfg.volumetric_intensity / q) * lod;
        clr[0] *= scale;
        clr[1] *= scale;
        clr[2] *= scale;
        (slices, q)
    } else {
        let q_raw = vol_cfg.ssfx_volumetric[2].max(1e-4);
        let slices = ((24.0 * q_raw) as i32).max(1) as u32;
        let intensity_mod =
            if light.light_type == RenderLightType::Omnipart || vol_cfg.is_player_light {
                0.2
            } else {
                1.0
            };
        let scale = ((vol_cfg.volumetric_intensity * intensity_mod) / q_raw) * lod;
        clr[0] *= scale;
        clr[1] *= scale;
        clr[2] *= scale;
        (slices, (slices as f32 / 120.0).max(1e-4))
    };

    let draw_slices = num_slices.min(VOLUMETRIC_SLICES as u32);
    let l_spec = compute_u_diffuse2s(clr, &vol_cfg.spot_accum.gloss);
    let l_pos = matrix_transform_point(view_matrix, light.position);
    let att_r = (vol_cfg.volumetric_distance * light.range * 0.95).max(1e-4);
    let att_factor = 1.0 / (att_r * att_r);

    VolumetricSpotParams {
        m_shadow: spot.m_shadow,
        m_lmap_rows: spot.m_lmap_rows,
        dynamic_pos: [l_pos[0], l_pos[1], l_pos[2], att_factor],
        dynamic_color: [clr[0], clr[1], clr[2], l_spec],
        v_min_bounds: [aabb_min[0], aabb_min[1], aabb_min[2], 0.0],
        v_max_bounds: [
            aabb_max[0],
            aabb_max[1],
            aabb_min[2] + (aabb_max[2] - aabb_min[2]) / quality_normalized,
            0.0,
        ],
        frustum_clip_planes,
        num_slices,
        vertex_count: draw_slices * 4,
        primitive_count: draw_slices * 2,
        quality_normalized,
    }
}

// ============================================================================
// Virtual Point-Light GI Bounce Generation (`light_GI.cpp` & `r2_R_lights.cpp`)
// ============================================================================

/// Fixed deterministic seed used by `light::gi_generate` (`light_GI.cpp:28`).
pub const GI_SEED: u32 = 0x1207_1980;
pub const GI_DEFAULT_PHOTONS: u32 = 16;
pub const GI_DEFAULT_CLIP: f32 = 0.001;
pub const GI_DEFAULT_REFL: f32 = 0.9;

/// X-Ray MSVCRT-style 31-bit LCG RNG (`class CRandom` in `src/xrCore/_random.h:21-56`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrngLcg {
    holdrand: i32,
}

impl Default for CrngLcg {
    fn default() -> Self {
        Self { holdrand: 1 }
    }
}

impl CrngLcg {
    #[inline]
    pub fn new(seed: u32) -> Self {
        Self {
            holdrand: seed as i32,
        }
    }

    #[inline]
    pub fn seed(&mut self, seed: u32) {
        self.holdrand = seed as i32;
    }

    /// Generate pseudo-random integer in `[0, 32767]` (`CRandom::randI`).
    #[inline]
    pub fn rand_i(&mut self) -> i32 {
        self.holdrand = self
            .holdrand
            .wrapping_mul(214_013)
            .wrapping_add(2_531_011);
        (self.holdrand >> 16) & 0x7fff
    }

    /// Generate pseudo-random integer in `[0, max)` (`CRandom::randI(max)`).
    #[inline]
    pub fn rand_i_max(&mut self, max: i32) -> i32 {
        if max <= 0 {
            0
        } else {
            self.rand_i() % max
        }
    }

    /// Generate pseudo-random integer in `[min, max)` (`CRandom::randI(min, max)`).
    #[inline]
    pub fn rand_i_range(&mut self, min: i32, max: i32) -> i32 {
        if max <= min {
            min
        } else {
            min + self.rand_i_max(max - min)
        }
    }

    /// Generate pseudo-random float in `[0.0, max]` (`CRandom::randF(max)`).
    #[inline]
    pub fn rand_f(&mut self, max: f32) -> f32 {
        (self.rand_i() as f32 / 32767.0) * max
    }

    /// Sample a unit vector using X-Ray's `_vector3::random_dir(CRandom&)` (`_vector3d.h:459-470`).
    #[inline]
    pub fn random_dir_sphere(&mut self) -> [f32; 3] {
        let z = self.rand_f(std::f32::consts::PI).cos();
        let a = self.rand_f(std::f32::consts::TAU);
        let r = (1.0 - z * z).max(0.0).sqrt();
        let (sa, ca) = a.sin_cos();
        [r * ca, r * sa, z]
    }

    /// Sample a unit vector around `cone_axis` using `_vector3::random_dir(axis, angle, CRandom&)`
    /// (`_vector3d.h:471-478`).
    #[inline]
    pub fn random_dir_cone(&mut self, cone_axis: [f32; 3], cone_angle_rad: f32) -> [f32; 3] {
        let rnd = self.random_dir_sphere();
        let scale = self.rand_f(cone_angle_rad.tan());
        normalize3_or(
            [
                cone_axis[0] + rnd[0] * scale,
                cone_axis[1] + rnd[1] * scale,
                cone_axis[2] + rnd[2] * scale,
            ],
            cone_axis,
        )
    }
}

/// Console configuration controlling GI virtual bounce generation (`xrRender_console.cpp:222-225`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GiConfig {
    pub enabled: bool,
    pub photons: u32,
    pub clip: f32,
    pub refl: f32,
}

impl Default for GiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            photons: GI_DEFAULT_PHOTONS,
            clip: GI_DEFAULT_CLIP,
            refl: GI_DEFAULT_REFL,
        }
    }
}

/// Reflected virtual point-light sample produced by `light::gi_generate` (`struct light_indirect`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightIndirectPhoton {
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub energy: f32,
    pub sector_id: u16,
}

/// Generate reflected virtual GI bounce photons using caller-supplied nearest-hit raycasting
/// `(origin, dir, range) -> Option<(hit_dist, unit_face_normal)>` matching `light::gi_generate`
/// (`light_GI.cpp:22-96`).
pub fn generate_indirect_photons<F>(
    light: &DynamicLight,
    sector_id: u16,
    cfg: &GiConfig,
    mut raycast_nearest: F,
) -> Vec<LightIndirectPhoton>
where
    F: FnMut([f32; 3], [f32; 3], f32) -> Option<(f32, [f32; 3])>,
{
    if !cfg.enabled || cfg.photons == 0 || light.range <= 1e-6 {
        return Vec::new();
    }

    let mut rng = CrngLcg::new(GI_SEED);
    let ray_count = cfg.photons.saturating_mul(8);
    let light_dir = normalize3_or(light.direction, [0.0, 0.0, 1.0]);
    let mut indirect = Vec::new();

    for _ in 0..ray_count {
        let dir_raw = match light.light_type {
            RenderLightType::Point => rng.random_dir_sphere(),
            RenderLightType::Spot | RenderLightType::Omnipart => {
                rng.random_dir_cone(light_dir, light.cone_rad)
            }
            RenderLightType::Direct | RenderLightType::Reflected => light_dir,
        };
        let dir = normalize3_or(dir_raw, light_dir);

        let Some((dist, tri_normal_raw)) = raycast_nearest(light.position, dir, light.range) else {
            continue;
        };
        let tri_normal = normalize3_or(tri_normal_raw, [0.0, 1.0, 0.0]);
        let cos_inc = -dot3(tri_normal, dir);
        let energy = cos_inc * (1.0 - dist / light.range);
        if energy < cfg.clip {
            continue;
        }

        let position = [
            light.position[0] + dir[0] * dist,
            light.position[1] + dir[1] * dist,
            light.position[2] + dir[2] * dist,
        ];
        // Fvector::reflect(dir, TN) = dir - 2 * dot(dir, TN) * TN
        let k = -2.0 * dot3(dir, tri_normal);
        let reflected = [
            dir[0] + k * tri_normal[0],
            dir[1] + k * tri_normal[1],
            dir[2] + k * tri_normal[2],
        ];
        indirect.push(LightIndirectPhoton {
            position,
            direction: reflected,
            energy,
            sector_id,
        });
    }

    indirect.sort_by(|a, b| b.energy.total_cmp(&a.energy));
    indirect.truncate(cfg.photons as usize);

    if !indirect.is_empty() {
        let total_e: f32 = indirect.iter().map(|p| p.energy).sum();
        if total_e > 1e-12 {
            let scale_e = cfg.refl / total_e;
            for p in &mut indirect {
                p.energy *= scale_e;
            }
        }
    }

    indirect
}

/// Generate GI bounce photons directly against a static world [`CformBvh`]
/// using `OPT_CULL | OPT_ONLYNEAREST` (`light_GI.cpp:30-69`).
pub fn generate_indirect_photons_cform(
    light: &DynamicLight,
    cform: &CformBvh<'_>,
    sector_id: u16,
    cfg: &GiConfig,
) -> Vec<LightIndirectPhoton> {
    let opts = RaycastOptions {
        cull_backfaces: true,
        only_first: false,
        only_nearest: true,
    };
    generate_indirect_photons(light, sector_id, cfg, |origin, dir, range| {
        let hit = cform.raycast(origin, dir, range, opts).into_iter().next()?;
        let v01 = [
            hit.verts[1][0] - hit.verts[0][0],
            hit.verts[1][1] - hit.verts[0][1],
            hit.verts[1][2] - hit.verts[0][2],
        ];
        let v12 = [
            hit.verts[2][0] - hit.verts[1][0],
            hit.verts[2][1] - hit.verts[1][1],
            hit.verts[2][2] - hit.verts[1][2],
        ];
        let norm = normalize3_or(cross3(v01, v12), [0.0, 1.0, 0.0]);
        Some((hit.range, norm))
    })
}

/// Emitted reflected bounce light ready for [`compute_reflected_accum_constants`]
/// (`CRender::render_indirect`, `r2_R_lights.cpp:273-316`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectedLightInstance {
    pub light: DynamicLight,
    pub sector_id: u16,
}

/// Convert cached [`LightIndirectPhoton`] samples into unshadowed [`RenderLightType::Reflected`]
/// lights matching `CRender::render_indirect` (`r2_R_lights.cpp:273-316`).
pub fn build_reflected_lights(
    parent: &DynamicLight,
    photons: &[LightIndirectPhoton],
    cfg: &GiConfig,
) -> Vec<ReflectedLightInstance> {
    if !cfg.enabled || photons.is_empty() {
        return Vec::new();
    }
    let le = (parent.color_rgb[0] + parent.color_rgb[1] + parent.color_rgb[2]) / 3.0;
    let e_min = 1.0f32 / 255.0f32;
    let mut out = Vec::with_capacity(photons.len());

    for photon in photons {
        let lie = le * photon.energy;
        if lie < cfg.clip {
            continue;
        }
        let range = (lie - e_min) / e_min;
        if range < 0.1 {
            continue;
        }
        let l_up = if photon.direction[1].abs() > 0.99 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let l_right = normalize3_or(cross3(l_up, photon.direction), [1.0, 0.0, 0.0]);
        out.push(ReflectedLightInstance {
            light: DynamicLight {
                light_type: RenderLightType::Reflected,
                is_static: parent.is_static,
                is_active: true,
                is_shadowed: false,
                is_volumetric: false,
                is_ortho: false,
                position: photon.position,
                direction: photon.direction,
                right: l_right,
                range,
                cone_rad: std::f32::consts::PI,
                virtual_size: parent.virtual_size,
                color_rgb: [
                    parent.color_rgb[0] * photon.energy,
                    parent.color_rgb[1] * photon.energy,
                    parent.color_rgb[2] * photon.energy,
                ],
                attenuation0: parent.attenuation0,
                attenuation1: parent.attenuation1,
                attenuation2: parent.attenuation2,
                falloff: parent.falloff,
                cached_smap_size: parent.cached_smap_size,
            },
            sector_id: photon.sector_id,
        });
    }

    out
}

// ============================================================================
// Light & Shadow-Map Occlusion Visibility (`light_vis.cpp`, `light_smapvis.h/.cpp`, `r2_R_lights.cpp`)
// ============================================================================

/// Default frame-delay range when camera is inside a light volume (`light_vis.cpp:5-6`).
pub const LIGHT_VIS_DELAY_SMALL_MIN: i32 = 1;
pub const LIGHT_VIS_DELAY_SMALL_MAX: i32 = 3;
/// Default frame-delay range when an occluded light proves visible (`light_vis.cpp:7-8`).
pub const LIGHT_VIS_DELAY_LARGE_MIN: i32 = 10;
pub const LIGHT_VIS_DELAY_LARGE_MAX: i32 = 20;
/// Minimum fragment count threshold marking a tested light visible (`light_vis.cpp:9`).
pub const LIGHT_VIS_CULL_FRAGMENTS: u64 = 4;
/// Default `ps_r__LightSleepFrames` before `smapvis` starts pruning static shadow casters.
pub const LIGHT_SLEEP_FRAMES_DEFAULT: u32 = 10;

/// Compute the camera near-frustum corner safety radius (`safe_area` in `light_vis.cpp:23-31`)
/// matching X-Ray's `(viewport_near, fov_deg, aspect = height / width)` parameters.
#[inline]
pub fn compute_light_vis_safe_area(viewport_near: f32, fov_deg: f32, aspect: f32) -> f32 {
    let vp = viewport_near.max(0.0);
    let a0 = (fov_deg * aspect * 0.5).to_radians().abs().min(std::f32::consts::FRAC_PI_2 - 1e-4);
    let a1 = (fov_deg * 0.5).to_radians().abs().min(std::f32::consts::FRAC_PI_2 - 1e-4);
    let x0 = vp / a0.cos().max(1e-4);
    let x1 = vp / a1.cos().max(1e-4);
    let c = (x0 * x0 + x1 * x1).sqrt();
    vp.max(x0.max(x1)).max(c)
}

/// Console flags & thresholds driving [`LightVisState::prepare`]/[`LightVisState::finish_pending`]
/// (`ps_r2_ls_flags` defaults in `xrRender_console.cpp:174`, `light_vis.cpp:9,35-38`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightVisConfig {
    /// `R2FLAG_EXP_DONT_TEST_UNSHADOWED` (ON by default in Anomaly).
    pub dont_test_unshadowed: bool,
    /// `R2FLAG_EXP_DONT_TEST_SHADOWED` (OFF by default in Anomaly).
    pub dont_test_shadowed: bool,
    /// Minimum passing fragment count (`cullfragments = 4`).
    pub cull_fragments: u64,
}

impl Default for LightVisConfig {
    fn default() -> Self {
        Self {
            dont_test_unshadowed: true,
            dont_test_shadowed: false,
            cull_fragments: LIGHT_VIS_CULL_FRAGMENTS,
        }
    }
}

/// Action requested by [`LightVisState::prepare`] for the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightVisDecision {
    /// Light stays visible (camera inside safe sphere or test disabled); no occlusion query needed.
    SkipOcclusionTest,
    /// Light exited the camera sphere and is ready for GPU occlusion testing this frame.
    IssueOcclusionQuery {
        stencil_test: bool,
        volumetric_spot: bool,
    },
}

/// Per-light visibility cache & hardware occlusion scheduling state (`light::_vis`, `light.h:62-75`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightVisState {
    pub distance: f32,
    pub visible: bool,
    pub pending: bool,
    pub frame2test: u32,
    pub query_id: u32,
    pub query_order: u32,
}

impl Default for LightVisState {
    fn default() -> Self {
        Self {
            distance: 0.0,
            visible: true,
            pending: false,
            frame2test: 0,
            query_id: crate::postprocess::OCCQ_INVALID_HANDLE,
            query_order: 0,
        }
    }
}

impl LightVisState {
    /// Force next visibility test at `frame` (`light::vis_prepare` reset).
    #[inline]
    pub fn invalidate(&mut self, frame: u32) {
        self.visible = true;
        self.pending = false;
        self.frame2test = frame;
    }

    /// Evaluate per-frame visibility test scheduling matching `light::vis_prepare` (`light_vis.cpp:11-65`).
    ///
    /// Returns `None` if re-testing is deferred until `self.frame2test` (keeping `self.visible`).
    pub fn prepare(
        &mut self,
        frame: u32,
        light: &DynamicLight,
        camera_pos: [f32; 3],
        safe_area: f32,
        cfg: &LightVisConfig,
        rng: &mut CrngLcg,
    ) -> Option<LightVisDecision> {
        if frame < self.frame2test {
            return None;
        }
        let (sc, sr) = light.spatial_sphere();
        let dx = camera_pos[0] - sc[0];
        let dy = camera_pos[1] - sc[1];
        let dz = camera_pos[2] - sc[2];
        self.distance = (dx * dx + dy * dy + dz * dz).sqrt();

        let skip_test = (cfg.dont_test_unshadowed && !light.is_shadowed)
            || (cfg.dont_test_shadowed && light.is_shadowed);
        if skip_test || self.distance <= sr * 1.01 + safe_area {
            self.visible = true;
            self.pending = false;
            self.frame2test = frame.saturating_add(
                rng.rand_i_range(LIGHT_VIS_DELAY_SMALL_MIN, LIGHT_VIS_DELAY_SMALL_MAX) as u32,
            );
            return Some(LightVisDecision::SkipOcclusionTest);
        }

        self.pending = true;
        let volumetric_spot =
            light.light_type == RenderLightType::Spot && light.is_shadowed && light.is_volumetric;
        Some(LightVisDecision::IssueOcclusionQuery {
            stencil_test: !volumetric_spot,
            volumetric_spot,
        })
    }

    /// Consume GPU occlusion query fragment count matching `light::vis_update` (`light_vis.cpp:67-90`).
    pub fn finish_pending(&mut self, frame: u32, fragments: u64, rng: &mut CrngLcg, cfg: &LightVisConfig) -> bool {
        if !self.pending {
            return self.visible;
        }
        self.visible = fragments > cfg.cull_fragments;
        self.pending = false;
        self.frame2test = if self.visible {
            frame.saturating_add(
                rng.rand_i_range(LIGHT_VIS_DELAY_LARGE_MIN, LIGHT_VIS_DELAY_LARGE_MAX) as u32,
            )
        } else {
            frame.saturating_add(1)
        };
        self.visible
    }
}

/// Shadow-map static caster pruning stage (`smapvis::state` enum in `light_smapvis.h:6-11`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum SmapVisStage {
    #[default]
    Counting = 0,
    Working = 1,
    UsingTc = 3,
}

/// Per-light incremental static shadow-caster occlusion cache (`class smapvis`, `light_smapvis.h/.cpp`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmapVisSolver {
    pub stage: SmapVisStage,
    pub invisible: Vec<usize>,
    pub frame_sleep: u32,
    pub test_count: u32,
    pub test_current: u32,
    pub test_q_visual: Option<usize>,
    pub test_q_id: u32,
    pub test_q_frame: u32,
}

impl Default for SmapVisSolver {
    fn default() -> Self {
        Self {
            stage: SmapVisStage::Counting,
            invisible: Vec::new(),
            frame_sleep: 0,
            test_count: 0,
            test_current: 0,
            test_q_visual: None,
            test_q_id: crate::postprocess::OCCQ_INVALID_HANDLE,
            test_q_frame: 0,
        }
    }
}

impl SmapVisSolver {
    /// Reset discovered invisible casters and postpone pruning for `sleep_frames` (`smapvis::invalidate`).
    pub fn invalidate(&mut self, frame: u32, sleep_frames: u32) {
        self.stage = SmapVisStage::Counting;
        self.test_q_visual = None;
        self.frame_sleep = frame.saturating_add(sleep_frames);
        self.invisible.clear();
    }

    /// Returns `true` once the cooldown window has elapsed (`smapvis::sleep()` in `light_smapvis.h:33`).
    #[inline]
    pub fn is_awake(&self, frame: u32) -> bool {
        frame > self.frame_sleep
    }

    #[inline]
    pub fn is_culled(&self, visual_index: usize) -> bool {
        self.invisible.contains(&visual_index)
    }

    /// Begin a shadow-map render pass (`smapvis::begin`, `light_smapvis.cpp:25-45`).
    ///
    /// Returns `Some(feedback_caster_index)` in [`SmapVisStage::Working`] so the caller captures the
    /// Nth drawn static caster via [`Self::record_feedback`].
    pub fn begin(&mut self) -> Option<u32> {
        match self.stage {
            SmapVisStage::Counting | SmapVisStage::UsingTc => None,
            SmapVisStage::Working => {
                self.test_q_visual = None;
                self.test_q_id = crate::postprocess::OCCQ_INVALID_HANDLE;
                Some(self.test_current)
            }
        }
    }

    /// Capture the static visual index hit at `test_current` (`smapvis::rfeedback_static`).
    #[inline]
    pub fn record_feedback(&mut self, visual_index: usize) {
        self.test_q_visual = Some(visual_index);
    }

    /// End a shadow-map render pass (`smapvis::end`, `light_smapvis.cpp:47-83`).
    ///
    /// Returns `Some(visual_index)` when in [`SmapVisStage::Working`] and a caster was captured for
    /// single-object GPU occlusion testing on `frame + 1`.
    pub fn end(&mut self, frame: u32, total_static_casters: u32) -> Option<usize> {
        match self.stage {
            SmapVisStage::Counting => {
                if self.is_awake(frame) {
                    self.test_count = total_static_casters;
                    self.test_current = 0;
                    self.stage = if total_static_casters == 0 {
                        SmapVisStage::UsingTc
                    } else {
                        SmapVisStage::Working
                    };
                }
                None
            }
            SmapVisStage::Working => {
                if let Some(vis_idx) = self.test_q_visual {
                    self.test_q_frame = frame.saturating_add(1);
                    Some(vis_idx)
                } else {
                    None
                }
            }
            SmapVisStage::UsingTc => None,
        }
    }

    /// Consume GPU occlusion fragments when `frame == test_q_frame` (`smapvis::flushoccq`,
    /// `light_smapvis.cpp:85-111`). Returns `Some(was_marked_invisible)` if flushed.
    pub fn flush_occq(&mut self, frame: u32, fragments: u64) -> Option<bool> {
        if frame != self.test_q_frame || self.stage != SmapVisStage::Working {
            return None;
        }
        let vis_idx = self.test_q_visual.take()?;
        let marked_invisible = fragments == 0;
        if marked_invisible {
            self.invisible.push(vis_idx);
            self.test_count = self.test_count.saturating_sub(1);
        } else {
            self.test_current = self.test_current.saturating_add(1);
        }
        if self.test_current >= self.test_count && self.stage == SmapVisStage::Working {
            self.stage = SmapVisStage::UsingTc;
        }
        Some(marked_invisible)
    }

    /// Pull forward an in-flight `frame + 1` query if a light becomes hidden mid-frame
    /// (`smapvis::resetoccq`, `light_smapvis.cpp:113-117`).
    #[inline]
    pub fn arm_reset_occq(&mut self, frame: u32) {
        if self.test_q_frame == frame.saturating_add(1) {
            self.test_q_frame = frame;
        }
    }
}

/// SSFX grass shadow configuration matching `ps_ssfx_grass_shadows` (`{0, 0.35, 30, 0}`,
/// `xrRender_console.cpp:482`) and `rsGrassShadow`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassShadowConfig {
    pub quality_level: f32,
    pub intensity: f32,
    pub shadow_distance: f32,
    pub rs_grass_shadow: bool,
}

impl Default for GrassShadowConfig {
    fn default() -> Self {
        Self {
            quality_level: 0.0,
            intensity: 0.35,
            shadow_distance: 30.0,
            rs_grass_shadow: false,
        }
    }
}

/// Returns `true` if `light` should cast dynamic grass shadows (`check_grass_shadow` in
/// `r2_R_lights.cpp:10-26`).
pub fn check_grass_shadow_light(
    light: &DynamicLight,
    vis_distance: f32,
    view_frustum: &ClipFrustum,
    cfg: &GrassShadowConfig,
) -> bool {
    if cfg.quality_level < 3.0 || !cfg.rs_grass_shadow {
        return false;
    }
    if vis_distance > cfg.shadow_distance {
        return false;
    }
    let mut mask = 0xffu32;
    view_frustum.test_sphere_masked(light.position, light.range * 0.6, &mut mask)
        != crate::hom::FrustumVisibility::None
}

pub const SUN_LIGHT_DISTANCE: f32 = 500.0;
pub const SUN_LIGHT_RANGE: f32 = 600.0;
pub const SUN_ADAPTED_DOWN_Y: f32 = -0.75;

/// Configuration for `CLight_DB::Update` directional sun light rig (`Light_DB.cpp:200-259`,
/// `xrRender_console.cpp:240-243`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunLightRigConfig {
    pub lumscale: f32,
    pub lumscale_color: [f32; 3],
    pub distance: f32,
    pub range: f32,
}

impl Default for SunLightRigConfig {
    fn default() -> Self {
        Self {
            lumscale: 1.0,
            lumscale_color: [1.0, 1.0, 1.0],
            distance: SUN_LIGHT_DISTANCE,
            range: SUN_LIGHT_RANGE,
        }
    }
}

/// Computed sun light pose (`direction`, `position`, `color_rgb`, `range`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunLightPose {
    pub direction: [f32; 3],
    pub position: [f32; 3],
    pub color_rgb: [f32; 3],
    pub range: f32,
}

/// Paired `sun_original` & `sun_adapted` lights matching `CLight_DB::Update`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunLightRig {
    pub sun_original: SunLightPose,
    pub sun_adapted: SunLightPose,
}

/// Compute `sun_original` & `sun_adapted` direction, origin, range, and scaled RGB from
/// environment sun parameters (`CLight_DB::Update`, `Light_DB.cpp:229-259`).
pub fn compute_sun_light_rig(
    sun_dir: [f32; 3],
    sun_color_rgb: [f32; 3],
    camera_pos: [f32; 3],
    is_sun_static: bool,
    cfg: &SunLightRigConfig,
) -> SunLightRig {
    let orig_dir = normalize3_or(sun_dir, [0.0, -1.0, 0.0]);
    let orig_pos = [
        camera_pos[0] - orig_dir[0] * cfg.distance,
        camera_pos[1] - orig_dir[1] * cfg.distance,
        camera_pos[2] - orig_dir[2] * cfg.distance,
    ];

    let mut adapt_raw = [
        sun_dir[0],
        sun_dir[1] + SUN_ADAPTED_DOWN_Y,
        sun_dir[2],
    ];
    let mut counter = 0usize;
    while dot3(adapt_raw, adapt_raw).sqrt() < 0.001 && counter < 10 {
        adapt_raw[0] += sun_dir[0];
        adapt_raw[1] += sun_dir[1];
        adapt_raw[2] += sun_dir[2];
        counter += 1;
    }
    let mut adapt_dir = normalize3_or(adapt_raw, orig_dir);
    let mut adapt_pos = [
        camera_pos[0] - adapt_dir[0] * cfg.distance,
        camera_pos[1] - adapt_dir[1] * cfg.distance,
        camera_pos[2] - adapt_dir[2] * cfg.distance,
    ];
    if !is_sun_static {
        adapt_dir = orig_dir;
        adapt_pos = orig_pos;
    }

    let adapted_color = [
        sun_color_rgb[0] * cfg.lumscale * cfg.lumscale_color[0],
        sun_color_rgb[1] * cfg.lumscale * cfg.lumscale_color[1],
        sun_color_rgb[2] * cfg.lumscale * cfg.lumscale_color[2],
    ];

    SunLightRig {
        sun_original: SunLightPose {
            direction: orig_dir,
            position: orig_pos,
            color_rgb: sun_color_rgb,
            range: cfg.range,
        },
        sun_adapted: SunLightPose {
            direction: adapt_dir,
            position: adapt_pos,
            color_rgb: adapted_color,
            range: cfg.range,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::matrix_build_camera_dir;
    use crate::sun::matrix_invert_44;

    #[test]
    fn smap_allocator_and_multipage_packer_never_overlap() {
        let mut alloc = SmapAllocator::new(1024);
        let r0 = alloc.push(512).unwrap();
        let r1 = alloc.push(512).unwrap();
        let r2 = alloc.push(256).unwrap();
        assert!(r0.valid() && r1.valid() && r2.valid());
        assert!(!r0.intersect(&r1));
        assert!(!r0.intersect(&r2));
        assert!(!r1.intersect(&r2));
        assert!(alloc.push(4).is_none());
        assert!(alloc.push(2048).is_none());

        let requests = vec![(0, 768), (1, 512), (2, 768), (3, 256), (4, 256)];
        let assignments = pack_smap_requests(1024, &requests);
        assert_eq!(assignments.len(), requests.len());
        let page_count = assignments.iter().map(|a| a.smap_id).max().unwrap() + 1;
        assert!(page_count >= 2);
        for (i, a) in assignments.iter().enumerate() {
            for b in assignments.iter().skip(i + 1) {
                if a.smap_id == b.smap_id {
                    assert!(!a.rect.intersect(&b.rect));
                }
            }
        }
    }

    #[test]
    fn dynamic_light_spheres_omnipart_lod_and_accum_constants() {
        let spot = DynamicLight {
            light_type: RenderLightType::Spot,
            position: [0.0, 2.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            range: 10.0,
            cone_rad: 60.0f32.to_radians(),
            is_shadowed: true,
            color_rgb: [1.0, 0.8, 0.6],
            ..DynamicLight::default()
        };
        let (sp_center, sp_r) = spot.spatial_sphere();
        assert!(sp_center[2] > 0.0 && sp_r > 5.0);
        let xform = spot.volume_world_xform();
        assert_eq!(xform[3][0..3], [0.0, 2.0, 0.0]);

        let point = DynamicLight {
            light_type: RenderLightType::Point,
            position: [3.0, 1.0, 4.0],
            range: 12.0,
            is_shadowed: true,
            ..DynamicLight::default()
        };
        let faces = point.expand_omnipart_faces();
        assert_eq!(faces.len(), 6);
        assert!(faces.iter().all(|f| f.light.light_type == RenderLightType::Omnipart));

        let (g_start, g_end) = compute_r_ssa_glod_bounds(1920, 1080, 67.5, 0.75, 256.0, 64.0);
        assert!(g_start > g_end);
        assert_eq!(spot.compute_lod([0.0, 2.0, 0.5], 0.5, g_start, g_end), 1.0);
        assert!(spot.compute_lod([0.0, 2.0, -400.0], 0.5, g_start, g_end) < 1.0);

        let cfg = LightSmapConfig::default();
        let xf = spot.compute_smap_xform([0.0, 2.0, -5.0], &cfg);
        assert!((256..=1536).contains(&xf.size));
        assert_eq!(xf.tan_shift, 0.0610865);

        let cam_view = matrix_build_camera_dir([0.0, 2.0, -5.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let inv_view = matrix_invert_44(&cam_view).unwrap();
        let rect = SmapRect::setup([0, 0], xf.size);
        let spot_consts = compute_spot_accum_constants(
            &spot,
            &xf,
            rect,
            [0.0, 2.0, -5.0],
            &cam_view,
            &inv_view,
            &SpotAccumConfig::default(),
        );
        assert!((spot_consts.dynamic_pos[3] - 1.0 / (9.5 * 9.5)).abs() < 1e-6);
        assert!(spot_consts.dynamic_color[3] > 0.0);

        let proj = matrix_build_projection(67.5f32.to_radians(), 9.0 / 16.0, 0.2, 200.0);
        let vp = matrix_mul_44(&proj, &cam_view);
        assert!(near_plane_intersects_sphere(&vp, [0.0, 2.0, -5.0], 2.0));
        assert!(!near_plane_intersects_sphere(&vp, [0.0, 2.0, 50.0], 1.0));

        let texgen_s = compute_texgen_screen(&vp);
        let texgen_j = compute_texgen_jitter(&vp, 1920, 1080, TEX_JITTER_SIZE);
        assert!(texgen_s[0][0].is_finite() && texgen_j[0][0] != texgen_s[0][0]);

        let gloss_ssfx = GlossConfig {
            ssfx_gloss_method: 1,
            ssfx_gloss_factor: 0.5,
            ssfx_gloss_minmax: [0.1, 0.9, 0.0],
            ..GlossConfig::default()
        };
        assert!((compute_u_diffuse2s([1.0, 1.0, 1.0], &gloss_ssfx) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn object_light_tracker_accumulates_hemi_cube_and_smooths_lights() {
        let mut tracker = ObjectLightTracker::new();
        let cfg = LightTrackConfig::default();

        let batch = tracker.next_hemi_ray_indices(26);
        assert_eq!(batch.len(), 26);
        for &idx in &batch {
            tracker.ray_clear[idx] = true;
        }
        tracker.tracked_lights.push(TrackedLightEntry {
            id: 7,
            last_touched_frame: 10,
            position: [2.0, 1.0, 0.0],
            color_rgb: [1.0, 0.8, 0.6],
            attenuation0: 1.0,
            attenuation1: 0.0,
            attenuation2: 0.25,
            falloff: 0.01,
            is_static: false,
            test: 0.5,
            energy: 0.5,
        });

        tracker.evaluate_lighting(
            10,
            0.05,
            [0.0, 1.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.5, 0.4, 0.3],
            &cfg,
            |_, _, _, _| true,
        );

        assert!((tracker.hemi_value - cfg.dhemi_sky_scale).abs() > 1e-4);
        assert!(tracker.hemi_cube[hemi_cube_faces::POS_X] > tracker.hemi_cube[hemi_cube_faces::NEG_X]);
        assert!(tracker.luminosity() > 0.1);
        assert!(tracker.tracked_lights[0].energy > 0.5);
    }

    #[test]
    fn canonical_volume_meshes_hemi_gpu_params_and_volumetric_spot() {
        assert_eq!(DU_SPHERE_VERTICES.len(), DU_SPHERE_NUMVERTEX);
        assert_eq!(DU_SPHERE_INDICES.len(), DU_SPHERE_NUMFACES * 3);
        assert!(DU_SPHERE_INDICES.iter().all(|&i| (i as usize) < DU_SPHERE_NUMVERTEX));
        for &v in &DU_SPHERE_VERTICES {
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-3);
        }

        assert_eq!(DU_CONE_VERTICES.len(), DU_CONE_NUMVERTEX);
        assert_eq!(DU_CONE_INDICES.len(), DU_CONE_NUMFACES * 3);
        assert!(DU_CONE_INDICES.iter().all(|&i| (i as usize) < DU_CONE_NUMVERTEX));

        assert_eq!(DU_SPHERE_PART_VERTICES.len(), DU_SPHERE_PART_NUMVERTEX);
        assert_eq!(DU_SPHERE_PART_INDICES.len(), DU_SPHERE_PART_NUMFACES * 3);
        assert!(DU_SPHERE_PART_INDICES.iter().all(|&i| (i as usize) < DU_SPHERE_PART_NUMVERTEX));

        assert_eq!(DU_BOX_VERTICES.len(), DU_BOX_NUMVERTEX);
        assert_eq!(DU_BOX_INDICES.len(), DU_BOX_NUMFACES * 3);
        assert!(DU_BOX_INDICES.iter().all(|&i| (i as usize) < DU_BOX_NUMVERTEX));
        assert_eq!(DU_BOX_VERTICES2.len(), DU_BOX_NUMVERTEX2);

        let slices = build_volumetric_slices_mesh();
        assert_eq!(slices.vertices.len(), VOLUMETRIC_SLICES * 4);
        assert_eq!(slices.indices.len(), VOLUMETRIC_SLICES * 6);
        assert!(slices.indices.iter().all(|&i| (i as usize) < slices.vertices.len()));
        assert_eq!(slices.vertices[0], [0.0, 0.0, 0.0]);
        assert!((slices.vertices[slices.vertices.len() - 1][2] - 1.0).abs() < 1e-4);

        assert!((encode_hemi_material_id(0.0) - 0.125).abs() < 1e-6);
        assert!((encode_hemi_material_id(1.0) - 0.375).abs() < 1e-6);
        assert_eq!(encode_hemi_material_id(5.0), 5.0);

        let mut tracker = ObjectLightTracker::new();
        tracker.hemi_smooth = 0.8;
        tracker.sun_smooth = 0.4;
        tracker.hemi_cube_smooth = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let hemi = tracker.to_hemi_gpu_params(1.0, [0.25, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.5]);
        assert_eq!(hemi.pos_faces, [0.1, 0.2, 0.3, 0.0]);
        assert_eq!(hemi.neg_faces, [0.4, 0.5, 0.6, 0.0]);
        assert!((hemi.material[0] - 0.6).abs() < 1e-6);
        assert!((hemi.material[1] - 0.3).abs() < 1e-6);
        assert!((hemi.material[3] - 0.375).abs() < 1e-6);
        assert_eq!(hemi.hotness, [0.25, 0.0, 0.0, 1.0]);
        assert_eq!(hemi.glowing, [1.0, 0.0, 0.0, 0.5]);

        let spot = DynamicLight {
            light_type: RenderLightType::Spot,
            position: [0.0, 2.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            range: 10.0,
            cone_rad: 60.0f32.to_radians(),
            is_shadowed: true,
            color_rgb: [1.0, 0.8, 0.6],
            ..DynamicLight::default()
        };
        let xf = spot.compute_smap_xform([0.0, 2.0, -5.0], &LightSmapConfig::default());
        let rect = SmapRect::setup([0, 0], xf.size);
        let cam_view = matrix_build_camera_dir([0.0, 2.0, -5.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let inv_view = matrix_invert_44(&cam_view).unwrap();
        let proj = matrix_build_projection(67.5f32.to_radians(), 9.0 / 16.0, 0.2, 200.0);
        let full = matrix_mul_44(&proj, &cam_view);
        let inv_full = matrix_invert_44(&full).unwrap();

        let vanilla_cfg = VolumetricLightConfig {
            ssfx_volumetric: [0.0, 1.0, 3.0, 1.0],
            ..VolumetricLightConfig::default()
        };
        let v_params = compute_volumetric_spot_params(
            &spot,
            &xf,
            rect,
            [0.0, 2.0, -5.0],
            &cam_view,
            &inv_view,
            &inv_full,
            &vanilla_cfg,
        );
        assert_eq!(v_params.num_slices, 100);
        assert_eq!(v_params.vertex_count, 400);
        assert_eq!(v_params.primitive_count, 200);
        assert_eq!(v_params.quality_normalized, 1.0);
        assert!(v_params
            .frustum_clip_planes
            .iter()
            .flatten()
            .all(|x| x.is_finite()));

        // A world point inside the spotlight cone maps to clip coordinates where every
        // transformed clip plane evaluates non-negative.
        let pt_world = [0.0f32, 2.0, 3.0];
        let pt_clip = [
            pt_world[0] * full[0][0] + pt_world[1] * full[1][0] + pt_world[2] * full[2][0] + full[3][0],
            pt_world[0] * full[0][1] + pt_world[1] * full[1][1] + pt_world[2] * full[2][1] + full[3][1],
            pt_world[0] * full[0][2] + pt_world[1] * full[1][2] + pt_world[2] * full[2][2] + full[3][2],
            pt_world[0] * full[0][3] + pt_world[1] * full[1][3] + pt_world[2] * full[2][3] + full[3][3],
        ];
        for cp in &v_params.frustum_clip_planes {
            let signed = pt_clip[0] * cp[0] + pt_clip[1] * cp[1] + pt_clip[2] * cp[2] + pt_clip[3] * cp[3];
            assert!(signed >= -1e-4, "expected interior clip point >= 0, got {signed}");
        }

        let ssfx_cfg = VolumetricLightConfig::default();
        let s_params = compute_volumetric_spot_params(
            &spot,
            &xf,
            rect,
            [0.0, 2.0, -5.0],
            &cam_view,
            &inv_view,
            &inv_full,
            &ssfx_cfg,
        );
        assert_eq!(s_params.num_slices, 72);
        assert_eq!(s_params.vertex_count, 288);
        assert_eq!(s_params.primitive_count, 144);
        assert!((s_params.quality_normalized - 72.0 / 120.0).abs() < 1e-6);
        assert!(s_params.v_max_bounds[2] > s_params.v_min_bounds[2]);

        let max_ssfx_cfg = VolumetricLightConfig {
            ssfx_volumetric: [1.0, 1.0, 5.0, 1.0],
            is_player_light: true,
            ..VolumetricLightConfig::default()
        };
        let p_params = compute_volumetric_spot_params(
            &spot,
            &xf,
            rect,
            [0.0, 2.0, -5.0],
            &cam_view,
            &inv_view,
            &inv_full,
            &max_ssfx_cfg,
        );
        assert_eq!(p_params.num_slices, 120);
        assert_eq!(p_params.vertex_count, (VOLUMETRIC_SLICES * 4) as u32);
        assert_eq!(p_params.primitive_count, (VOLUMETRIC_SLICES * 2) as u32);
        assert!(p_params.dynamic_color[0] < s_params.dynamic_color[0]);
    }

    #[test]
    fn point_reflected_depth_bounds_light_package_sort_and_gi_photons() {
        let cam_view = matrix_build_camera_dir([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let proj = matrix_build_projection(std::f32::consts::FRAC_PI_2, 1.0, 0.2, 100.0);
        let full = matrix_mul_44(&proj, &cam_view);
        let frustum = ClipFrustum::from_view_proj(&full, frustum_planes::ALL);

        let inside_pt = DynamicLight {
            light_type: RenderLightType::Point,
            position: [0.0, 0.0, 20.0],
            range: 2.0,
            is_shadowed: true,
            ..DynamicLight::default()
        };
        let bounds = inside_pt.compute_depth_bounds(&full, &frustum).unwrap();
        assert!(bounds[0] >= 0.0 && bounds[1] <= 1.0 && bounds[0] < bounds[1]);

        let edge_pt = DynamicLight {
            position: [0.0, 0.0, 0.5],
            range: 2.0,
            ..inside_pt
        };
        assert!(edge_pt.compute_depth_bounds(&full, &frustum).is_none());

        assert_eq!(
            inside_pt.accum_shader_element(false, 2048, 2048),
            AccumShaderElement::Fullsize
        );
        assert_eq!(
            inside_pt.accum_shader_element(false, 768, 2048),
            AccumShaderElement::Normal
        );
        assert_eq!(
            inside_pt.accum_shader_element(true, 768, 2048),
            AccumShaderElement::Translucent
        );
        let unshadowed = DynamicLight {
            is_shadowed: false,
            ..inside_pt
        };
        assert_eq!(
            unshadowed.accum_shader_element(true, 2048, 2048),
            AccumShaderElement::Unshadowed
        );

        let pt_consts = compute_point_accum_constants(
            &inside_pt,
            768,
            2048,
            false,
            false,
            3,
            &cam_view,
            &full,
            &GlossConfig::default(),
        );
        assert_eq!(pt_consts.shader_element, AccumShaderElement::Normal);
        assert_eq!(pt_consts.sss_id, 3);
        assert!(pt_consts.cull_backfaces);
        assert!((pt_consts.dynamic_pos[3] - 1.0 / (1.9 * 1.9)).abs() < 1e-6);

        let pt_svp = compute_point_accum_constants(
            &inside_pt,
            768,
            2048,
            false,
            true,
            3,
            &cam_view,
            &full,
            &GlossConfig::default(),
        );
        assert_eq!(pt_svp.sss_id, -1);

        let mut pkg = vec![
            (1u32, LightPackageOrderKey { pending: true, query_order: 2, range: 50.0 }),
            (2u32, LightPackageOrderKey { pending: false, query_order: 0, range: 5.0 }),
            (3u32, LightPackageOrderKey { pending: false, query_order: 0, range: 20.0 }),
            (4u32, LightPackageOrderKey { pending: true, query_order: 10, range: 1.0 }),
        ];
        sort_light_package_by_key(&mut pkg, |entry| entry.1);
        let ordered_ids: Vec<u32> = pkg.into_iter().map(|e| e.0).collect();
        assert_eq!(ordered_ids, vec![3, 2, 4, 1]);

        let mut rng_a = CrngLcg::new(GI_SEED);
        let mut rng_b = CrngLcg::new(GI_SEED);
        assert_eq!(rng_a.rand_i(), rng_b.rand_i());
        let cone_dir = rng_a.random_dir_cone([0.0, 0.0, 1.0], 0.5);
        assert!((dot3(cone_dir, cone_dir) - 1.0).abs() < 1e-5);

        let verts = vec![
            [-20.0, -20.0, 5.0],
            [-20.0, 20.0, 5.0],
            [20.0, 20.0, 5.0],
            [20.0, -20.0, 5.0],
        ];
        let tris = vec![
            crate::cform::CformTriangle {
                verts: [0, 1, 2],
                material: 0,
                suppress_shadows: false,
                suppress_wm: false,
                sector: 2,
            },
            crate::cform::CformTriangle {
                verts: [0, 2, 3],
                material: 0,
                suppress_shadows: false,
                suppress_wm: false,
                sector: 2,
            },
        ];
        let model = crate::cform::CformModel::from_parts(verts, tris).unwrap();
        let bvh = model.build_bvh();
        let gi_spot = DynamicLight {
            light_type: RenderLightType::Spot,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            range: 20.0,
            cone_rad: 60.0f32.to_radians(),
            color_rgb: [2.0, 1.5, 1.0],
            ..DynamicLight::default()
        };
        let gi_cfg = GiConfig {
            enabled: true,
            photons: 8,
            ..GiConfig::default()
        };
        let photons = generate_indirect_photons_cform(&gi_spot, &bvh, 2, &gi_cfg);
        assert!(!photons.is_empty() && photons.len() <= 8);
        let sum_e: f32 = photons.iter().map(|p| p.energy).sum();
        assert!((sum_e - gi_cfg.refl).abs() < 1e-4);
        assert!(photons.iter().all(|p| (p.position[2] - 5.0).abs() < 1e-3 && p.direction[2] < 0.0));

        let reflected = build_reflected_lights(&gi_spot, &photons, &gi_cfg);
        assert!(!reflected.is_empty());
        assert_eq!(reflected[0].light.light_type, RenderLightType::Reflected);
        let ref_consts = compute_reflected_accum_constants(
            &reflected[0].light,
            &cam_view,
            &full,
            1920,
            1080,
            true,
            &GlossConfig::default(),
        );
        assert_eq!(ref_consts.direction_view[3], 0.0);
        assert!((ref_consts.dynamic_pos[3] - 1.0 / (reflected[0].light.range.powi(2))).abs() < 1e-6);

        // LightVisState + compute_light_vis_safe_area + SmapVisSolver + GrassShadowConfig
        let safe_area = compute_light_vis_safe_area(0.2, 67.5, 9.0 / 16.0);
        assert!(safe_area > 0.25);
        let vis_cfg = LightVisConfig {
            dont_test_unshadowed: false,
            dont_test_shadowed: false,
            cull_fragments: 4,
        };
        let mut vis_rng = CrngLcg::new(77);
        let mut vis = LightVisState::default();
        assert_eq!(
            vis.prepare(1, &gi_spot, [0.0, 0.0, 0.0], safe_area, &vis_cfg, &mut vis_rng),
            Some(LightVisDecision::SkipOcclusionTest)
        );
        // Far frame while sleeping returns None
        assert_eq!(
            vis.prepare(1, &gi_spot, [0.0, 0.0, 50.0], safe_area, &vis_cfg, &mut vis_rng),
            None
        );
        let next_f = vis.frame2test;
        assert_eq!(
            vis.prepare(next_f, &gi_spot, [0.0, 0.0, 50.0], safe_area, &vis_cfg, &mut vis_rng),
            Some(LightVisDecision::IssueOcclusionQuery {
                stencil_test: true,
                volumetric_spot: false,
            })
        );
        assert!(!vis.finish_pending(next_f + 1, 2, &mut vis_rng, &vis_cfg));
        assert_eq!(vis.frame2test, next_f + 2);
        assert_eq!(
            vis.prepare(next_f + 2, &gi_spot, [0.0, 0.0, 50.0], safe_area, &vis_cfg, &mut vis_rng),
            Some(LightVisDecision::IssueOcclusionQuery {
                stencil_test: true,
                volumetric_spot: false,
            })
        );
        assert!(vis.finish_pending(next_f + 3, 16, &mut vis_rng, &vis_cfg));
        assert!(vis.frame2test >= next_f + 3 + LIGHT_VIS_DELAY_LARGE_MIN as u32);

        let mut smap = SmapVisSolver::default();
        smap.invalidate(5, LIGHT_SLEEP_FRAMES_DEFAULT);
        assert!(!smap.is_awake(15));
        assert!(smap.is_awake(16));
        assert_eq!(smap.end(10, 2), None);
        assert_eq!(smap.end(16, 2), None);
        assert_eq!(smap.stage, SmapVisStage::Working);
        assert_eq!(smap.begin(), Some(0));
        smap.record_feedback(10);
        assert_eq!(smap.end(16, 2), Some(10));
        assert_eq!(smap.flush_occq(17, 0), Some(true));
        assert!(smap.is_culled(10));
        assert_eq!(smap.begin(), Some(0));
        smap.record_feedback(11);
        assert_eq!(smap.end(17, 2), Some(11));
        assert_eq!(smap.flush_occq(18, 25), Some(false));
        assert_eq!(smap.stage, SmapVisStage::UsingTc);

        let gs_light = DynamicLight {
            position: [0.0, 0.0, 10.0],
            range: 8.0,
            ..DynamicLight::default()
        };
        let mut gs_cfg = GrassShadowConfig::default();
        assert!(!check_grass_shadow_light(&gs_light, 10.0, &frustum, &gs_cfg));
        gs_cfg.quality_level = 3.0;
        gs_cfg.rs_grass_shadow = true;
        assert!(check_grass_shadow_light(&gs_light, 10.0, &frustum, &gs_cfg));
        assert!(!check_grass_shadow_light(&gs_light, 35.0, &frustum, &gs_cfg));

        assert_eq!(DU_CYLINDER_VERTICES.len(), DU_CYLINDER_NUMVERTEX);
        assert_eq!(DU_CYLINDER_INDICES.len(), DU_CYLINDER_NUMFACES * 3);
        assert_eq!(DU_CYLINDER_LINES.len(), DU_CYLINDER_NUMLINES * 2);
        assert!(DU_CYLINDER_INDICES
            .iter()
            .all(|&i| (i as usize) < DU_CYLINDER_NUMVERTEX));
        assert!(DU_CYLINDER_LINES
            .iter()
            .all(|&i| (i as usize) < DU_CYLINDER_NUMVERTEX));

        let rig_cfg = SunLightRigConfig {
            lumscale: 1.5,
            lumscale_color: [1.0, 0.8, 0.5],
            ..SunLightRigConfig::default()
        };
        let rig_static = compute_sun_light_rig(
            [0.0, -0.5, 0.8660254],
            [1.0, 1.0, 1.0],
            [10.0, 5.0, -2.0],
            true,
            &rig_cfg,
        );
        assert_eq!(rig_static.sun_original.range, 600.0);
        assert!((rig_static.sun_original.position[1] - 255.0).abs() < 1e-4);
        assert!(rig_static.sun_adapted.direction[1] < rig_static.sun_original.direction[1]);
        assert!((rig_static.sun_adapted.color_rgb[1] - 1.2).abs() < 1e-6);

        let rig_dynamic = compute_sun_light_rig(
            [0.0, -0.5, 0.8660254],
            [1.0, 1.0, 1.0],
            [10.0, 5.0, -2.0],
            false,
            &rig_cfg,
        );
        assert_eq!(rig_dynamic.sun_adapted.direction, rig_dynamic.sun_original.direction);
        assert_eq!(rig_dynamic.sun_adapted.position, rig_dynamic.sun_original.position);
    }
}
