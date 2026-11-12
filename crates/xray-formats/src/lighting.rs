//! Pure Rust Dynamic Lighting Pipeline, Shadow-Map Atlas Allocator & Object LightTrack
//!
//! Grounds X-Ray's local light volumes, adaptive shadow-map sizing, SMAP atlas packing,
//! projective accumulation transforms, and object hemisphere ambient tracking directly from:
//! - `src/Layers/xrRenderPC_R4/SMAP_Allocator.h` & `r2_R_lights.cpp`
//! - `src/Layers/xrRenderPC_R4/Light_Render_Direct_ComputeXFS.cpp`
//! - `src/Layers/xrRender/light.cpp` & `src/Layers/xrRenderPC_R4/r2_types.h` (`u_diffuse2s`)
//! - `src/Layers/xrRenderPC_R4/r4_rendertarget_accum_spot.cpp` & `r4_rendertarget_enable_scissor.cpp`
//! - `src/Layers/xrRender/LightTrack.h` & `LightTrack.cpp`

use std::f32::consts::FRAC_1_SQRT_2;

use crate::camera::{matrix_build_camera_dir, matrix_build_projection, matrix_build_projection_ortho};
use crate::level::{d3d_light_type, LevelLight};
use crate::ogf::{
    matrix_mul_43, matrix_mul_44, matrix_scale, matrix_transform_point, XRayMatrix,
    XRAY_IDENTITY_MATRIX,
};

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
}
