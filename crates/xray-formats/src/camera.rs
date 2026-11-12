//! Pure Rust X-Ray Camera Matrices, `.anm`/`.anms` Playback & Postprocess Effector Mixer
//!
//! Covers:
//! - X-Ray view & projection matrix builders (`src/xrCore/_matrix.h:632-765`):
//!   `matrix_build_camera`, `matrix_build_camera_dir`, `matrix_build_projection`,
//!   `matrix_build_projection_hat`, `matrix_build_projection_ortho`
//! - `CamEffectorInfo` (`SCamEffectorInfo` in `CameraDefs.h`) & `CCameraManager` effector blending
//! - `CameraMotionPlayer` (`CObjectAnimator` / `SAnimParams` over `.anm` and `.anms`)
//! - `PostprocessPlayer` (`CPostprocessAnimator` over `.ppe`) & `mix_postprocess_effectors`
//!   matching `CCameraManager::UpdatePPEffectors` (`CameraManager.cpp:425-470`)

use crate::envelopes::{ObjectMotion, ObjectMotionLibrary, PostprocessEffect, PostprocessState};
use crate::ogf::{
    matrix_invert_43, matrix_mul_43, matrix_mul_44, matrix_set_xyzi_translate, XRayMatrix,
    XRAY_IDENTITY_MATRIX,
};
use crate::weather::R_VIEWPORT_NEAR;

pub const UI_3D_ICON_ORTHO_HEIGHT: f32 = 2.0;
pub const UI_3D_ICON_NEAR_PLANE: f32 = 0.01;
pub const UI_3D_ICON_FAR_PLANE: f32 = 100.0;
pub const UI_3D_ICON_DISTANCE: f32 = 10.0;
pub const DEFAULT_HUD_FOV_FACTOR: f32 = 0.45;
pub const HUD_BASE_FOV_DEG: f32 = 83.0;
pub const DEFAULT_PARALLAX_H: f32 = 0.02;

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
    let len_sq = dot3(v, v);
    if len_sq > 1e-12 {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

/// Build an X-Ray left-handed camera view matrix from position, forward direction, and world up
/// (`Fmatrix::build_camera_dir`, `src/xrCore/_matrix.h:730-765`).
pub fn matrix_build_camera_dir(from: [f32; 3], view: [f32; 3], world_up: [f32; 3]) -> XRayMatrix {
    let v_view = normalize3_or(view, [0.0, 0.0, 1.0]);
    let f_dot = dot3(world_up, v_view);
    let up_raw = [
        world_up[0] - v_view[0] * f_dot,
        world_up[1] - v_view[1] * f_dot,
        world_up[2] - v_view[2] * f_dot,
    ];
    let fallback_up = if v_view[1].abs() < 0.99 {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let v_up = normalize3_or(up_raw, fallback_up);
    let v_right = cross3(v_up, v_view);

    [
        [v_right[0], v_up[0], v_view[0], 0.0],
        [v_right[1], v_up[1], v_view[1], 0.0],
        [v_right[2], v_up[2], v_view[2], 0.0],
        [-dot3(from, v_right), -dot3(from, v_up), -dot3(from, v_view), 1.0],
    ]
}

/// Build an X-Ray left-handed look-at view matrix (`Fmatrix::build_camera`, `_matrix.h:688-728`).
pub fn matrix_build_camera(from: [f32; 3], at: [f32; 3], world_up: [f32; 3]) -> XRayMatrix {
    let view = [at[0] - from[0], at[1] - from[1], at[2] - from[2]];
    matrix_build_camera_dir(from, view, world_up)
}

/// Build a DX `[0, 1]` Z-depth perspective projection from half-angle tangent `hat`
/// and X-Ray aspect `height / width` (`Fmatrix::build_projection_HAT`, `_matrix.h:638-665`).
pub fn matrix_build_projection_hat(hat: f32, aspect: f32, near: f32, far: f32) -> XRayMatrix {
    let cot = 1.0 / hat.max(1e-6);
    let w = aspect * cot;
    let h = cot;
    let denom = far - near;
    let q = if denom.abs() > 1e-6 { far / denom } else { 1.0 };

    [
        [w, 0.0, 0.0, 0.0],
        [0.0, h, 0.0, 0.0],
        [0.0, 0.0, q, 1.0],
        [0.0, 0.0, -q * near, 0.0],
    ]
}

/// Build an X-Ray perspective projection from vertical FOV in radians and X-Ray aspect (`height / width`)
/// (`Fmatrix::build_projection`, `_matrix.h:632-635`).
#[inline]
pub fn matrix_build_projection(fov_rad: f32, aspect: f32, near: f32, far: f32) -> XRayMatrix {
    matrix_build_projection_hat((fov_rad * 0.5).tan(), aspect, near, far)
}

/// Build an X-Ray orthographic projection (`Fmatrix::build_projection_ortho`, `_matrix.h:667-686`).
pub fn matrix_build_projection_ortho(width: f32, height: f32, near: f32, far: f32) -> XRayMatrix {
    let denom = far - near;
    let m33 = if denom.abs() > 1e-6 { 1.0 / denom } else { 1.0 };
    let m43 = if denom.abs() > 1e-6 {
        near / (near - far)
    } else {
        0.0
    };
    [
        [2.0 / width.max(1e-6), 0.0, 0.0, 0.0],
        [0.0, 2.0 / height.max(1e-6), 0.0, 0.0],
        [0.0, 0.0, m33, 0.0],
        [0.0, 0.0, m43, 1.0],
    ]
}

/// Build the orthographic projection matrix used for 3D UI item icons (`mProject3DIcons`,
/// `CameraManager.cpp:505-506`), where `aspect` is X-Ray's `height / width`.
#[inline]
pub fn ui_3d_icon_projection(aspect: f32) -> XRayMatrix {
    matrix_build_projection_ortho(
        UI_3D_ICON_ORTHO_HEIGHT / aspect.max(1e-6),
        UI_3D_ICON_ORTHO_HEIGHT,
        UI_3D_ICON_NEAR_PLANE,
        UI_3D_ICON_FAR_PLANE,
    )
}

/// Build the orthographic projection matrix matching a 3D UI icon atlas surface (`BeginUI3DIconAtlas`,
/// `r4_R_render.cpp:698-699`).
#[inline]
pub fn ui_3d_icon_atlas_projection(atlas_width: u32, atlas_height: u32) -> XRayMatrix {
    let w = UI_3D_ICON_ORTHO_HEIGHT * (atlas_width.max(1) as f32) / (atlas_height.max(1) as f32);
    matrix_build_projection_ortho(
        w,
        UI_3D_ICON_ORTHO_HEIGHT,
        UI_3D_ICON_NEAR_PLANE,
        UI_3D_ICON_FAR_PLANE,
    )
}

/// Build the first-person HUD or camera-attached perspective projection matrix (`Device.mProjectHud` /
/// `Device.mProjectCamAttached`, `CameraManager.cpp:503-504`) using `fov_factor * 83.0` deg and
/// `R_VIEWPORT_NEAR` (`0.005`).
#[inline]
pub fn hud_projection_matrix(hud_fov_factor: f32, aspect: f32, far_plane: f32) -> XRayMatrix {
    let fov_rad = (hud_fov_factor * HUD_BASE_FOV_DEG)
        .clamp(1.0, 175.0)
        .to_radians();
    matrix_build_projection(
        fov_rad,
        aspect.max(1e-6),
        R_VIEWPORT_NEAR,
        far_plane.max(R_VIEWPORT_NEAR + 1e-3),
    )
}

/// 48-byte view-space G-buffer position decompression and parallax shader constants
/// (`cl_pos_decompress_params`, `cl_pos_decompress_params2`, `cl_parallax` in `r4.cpp:104-140`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PosDecompressParams {
    /// `[horz_tan, vert_tan, (2.0 * horz_tan) / w, (2.0 * vert_tan) / h]`.
    pub pos_decompression_params: [f32; 4],
    /// `[w, h, 1.0 / w, 1.0 / h]`.
    pub pos_decompression_params2: [f32; 4],
    /// `[parallax_h, -0.5 * parallax_h, 1.0 / r_dtex_range, 1.0 / r_dtex_range]`.
    pub parallax: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<PosDecompressParams>() == 48);

impl PosDecompressParams {
    /// Compute view-space decompression & detail-parallax constants matching `r4.cpp:104-140`.
    pub fn compute(
        fov_deg: f32,
        aspect: f32,
        screen_w: u32,
        screen_h: u32,
        parallax_h: f32,
        r_dtex_range: f32,
    ) -> Self {
        let w = screen_w.max(1) as f32;
        let h = screen_h.max(1) as f32;
        let aspect_safe = aspect.max(1e-6);
        let vert_tan = -((fov_deg.clamp(1.0, 175.0) * 0.5).to_radians().tan());
        let horz_tan = -vert_tan / aspect_safe;
        let dtex = r_dtex_range.max(1e-6);
        Self {
            pos_decompression_params: [
                horz_tan,
                vert_tan,
                (2.0 * horz_tan) / w,
                (2.0 * vert_tan) / h,
            ],
            pos_decompression_params2: [w, h, 1.0 / w, 1.0 / h],
            parallax: [parallax_h, -parallax_h * 0.5, 1.0 / dtex, 1.0 / dtex],
        }
    }
}

/// Active current + previous-frame world/view/projection transform stack with lazy inverse world
/// matrix computation (`class R_xforms` in `R_Backend_xform.h` / `R_Backend_xform.cpp:6-126`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderXforms {
    pub m_w: XRayMatrix,
    pub m_invw: XRayMatrix,
    pub m_v: XRayMatrix,
    pub m_p: XRayMatrix,
    pub m_wv: XRayMatrix,
    pub m_vp: XRayMatrix,
    pub m_wvp: XRayMatrix,
    pub m_w_prev: XRayMatrix,
    pub m_v_prev: XRayMatrix,
    pub m_p_prev: XRayMatrix,
    pub m_wv_prev: XRayMatrix,
    pub m_vp_prev: XRayMatrix,
    pub m_wvp_prev: XRayMatrix,
    inv_w_valid: bool,
}

impl Default for RenderXforms {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderXforms {
    /// Initialize all current and previous-frame matrices to identity (`R_xforms::R_xforms`).
    pub fn new() -> Self {
        Self {
            m_w: XRAY_IDENTITY_MATRIX,
            m_invw: XRAY_IDENTITY_MATRIX,
            m_v: XRAY_IDENTITY_MATRIX,
            m_p: XRAY_IDENTITY_MATRIX,
            m_wv: XRAY_IDENTITY_MATRIX,
            m_vp: XRAY_IDENTITY_MATRIX,
            m_wvp: XRAY_IDENTITY_MATRIX,
            m_w_prev: XRAY_IDENTITY_MATRIX,
            m_v_prev: XRAY_IDENTITY_MATRIX,
            m_p_prev: XRAY_IDENTITY_MATRIX,
            m_wv_prev: XRAY_IDENTITY_MATRIX,
            m_vp_prev: XRAY_IDENTITY_MATRIX,
            m_wvp_prev: XRAY_IDENTITY_MATRIX,
            inv_w_valid: true,
        }
    }

    #[inline]
    pub fn inv_w_valid(&self) -> bool {
        self.inv_w_valid
    }

    /// Update world transform `m_w` and recompute `m_wv` / `m_wvp`, invalidating lazy `m_invw`
    /// (`R_xforms::set_W`, `R_Backend_xform.cpp:6-17`).
    #[inline]
    pub fn set_w(&mut self, m: XRayMatrix) {
        self.m_w = m;
        self.m_wv = matrix_mul_43(&self.m_v, &self.m_w);
        self.m_wvp = matrix_mul_44(&self.m_p, &self.m_wv);
        self.inv_w_valid = false;
    }

    /// Update view transform `m_v` and recompute `m_wv`, `m_vp`, `m_wvp`
    /// (`R_xforms::set_V`, `R_Backend_xform.cpp:19-30`).
    #[inline]
    pub fn set_v(&mut self, m: XRayMatrix) {
        self.m_v = m;
        self.m_wv = matrix_mul_43(&self.m_v, &self.m_w);
        self.m_vp = matrix_mul_44(&self.m_p, &self.m_v);
        self.m_wvp = matrix_mul_44(&self.m_p, &self.m_wv);
    }

    /// Update projection transform `m_p` and recompute `m_vp`, `m_wvp`
    /// (`R_xforms::set_P`, `R_Backend_xform.cpp:32-42`).
    #[inline]
    pub fn set_p(&mut self, m: XRayMatrix) {
        self.m_p = m;
        self.m_vp = matrix_mul_44(&self.m_p, &self.m_v);
        self.m_wvp = matrix_mul_44(&self.m_p, &self.m_wv);
    }

    /// Update previous-frame world transform `m_w_prev` and recompute `m_wv_prev` / `m_wvp_prev`
    /// (`R_xforms::set_W_prev`, `R_Backend_xform.cpp:44-53`).
    #[inline]
    pub fn set_w_prev(&mut self, m: XRayMatrix) {
        self.m_w_prev = m;
        self.m_wv_prev = matrix_mul_43(&self.m_v_prev, &self.m_w_prev);
        self.m_wvp_prev = matrix_mul_44(&self.m_p_prev, &self.m_wv_prev);
    }

    /// Update previous-frame view transform `m_v_prev` and recompute `m_wv_prev` / `m_vp_prev` / `m_wvp_prev`
    /// (`R_xforms::set_V_prev`, `R_Backend_xform.cpp:54-65`).
    #[inline]
    pub fn set_v_prev(&mut self, m: XRayMatrix) {
        self.m_v_prev = m;
        self.m_wv_prev = matrix_mul_43(&self.m_v_prev, &self.m_w_prev);
        self.m_vp_prev = matrix_mul_44(&self.m_p_prev, &self.m_v_prev);
        self.m_wvp_prev = matrix_mul_44(&self.m_p_prev, &self.m_wv_prev);
    }

    /// Update previous-frame projection transform `m_p_prev` and recompute `m_vp_prev` / `m_wvp_prev`
    /// (`R_xforms::set_P_prev`, `R_Backend_xform.cpp:66-74`).
    #[inline]
    pub fn set_p_prev(&mut self, m: XRayMatrix) {
        self.m_p_prev = m;
        self.m_vp_prev = matrix_mul_44(&self.m_p_prev, &self.m_v_prev);
        self.m_wvp_prev = matrix_mul_44(&self.m_p_prev, &self.m_wv_prev);
    }

    /// Lazily invert `m_w` (`R_xforms::apply_invw`, `R_Backend_xform.cpp:76-87`), returning `None`
    /// if the current world matrix is singular.
    #[inline]
    pub fn apply_inv_w(&mut self) -> Option<&XRayMatrix> {
        if !self.inv_w_valid {
            self.m_invw = matrix_invert_43(&self.m_w)?;
            self.inv_w_valid = true;
        }
        Some(&self.m_invw)
    }

    /// Copy the current frame matrices into `*_prev` at end-of-frame.
    #[inline]
    pub fn snapshot_to_prev(&mut self) {
        self.m_w_prev = self.m_w;
        self.m_v_prev = self.m_v;
        self.m_p_prev = self.m_p;
        self.m_wv_prev = self.m_wv;
        self.m_vp_prev = self.m_vp;
        self.m_wvp_prev = self.m_wvp;
    }
}

// ============================================================================
// Phase 14/15 DX12 G-Buffer MRT Codecs & Depth/World-Pos Decompression
// ============================================================================

/// Transpose a 4x4 matrix (converts between X-Ray row-vector matrices and DX12 column-major
/// HLSL column-vector matrices).
#[inline]
pub fn matrix_transpose_44(m: &XRayMatrix) -> XRayMatrix {
    [
        [m[0][0], m[1][0], m[2][0], m[3][0]],
        [m[0][1], m[1][1], m[2][1], m[3][1]],
        [m[0][2], m[1][2], m[2][2], m[3][2]],
        [m[0][3], m[1][3], m[2][3], m[3][3]],
    ]
}

/// Convert a normalized `[0.0, 1.0]` float into an 8-bit `R8G8B8A8_UNORM` channel.
#[inline]
pub fn f32_to_unorm8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
}

/// Convert an 8-bit `R8G8B8A8_UNORM` channel into a normalized `[0.0, 1.0]` float.
#[inline]
pub fn unorm8_to_f32(v: u8) -> f32 {
    f32::from(v) / 255.0
}

/// Pack DX12 G-Buffer `SV_Target0` (`RGBA8_UNORM`: RGB = albedo, A = ambient occlusion)
/// matching `crates/xray-render-dx12/shaders/gbuffer.hlsl:96`.
#[inline]
pub fn pack_gbuffer_albedo_ao(albedo: [f32; 3], occlusion: f32) -> [u8; 4] {
    [
        f32_to_unorm8(albedo[0]),
        f32_to_unorm8(albedo[1]),
        f32_to_unorm8(albedo[2]),
        f32_to_unorm8(occlusion),
    ]
}

/// Unpack DX12 G-Buffer `SV_Target0` into linear `(albedo, occlusion)`.
#[inline]
pub fn unpack_gbuffer_albedo_ao(rgba: [u8; 4]) -> ([f32; 3], f32) {
    (
        [
            unorm8_to_f32(rgba[0]),
            unorm8_to_f32(rgba[1]),
            unorm8_to_f32(rgba[2]),
        ],
        unorm8_to_f32(rgba[3]),
    )
}

/// Pack DX12 G-Buffer `SV_Target1` (`RGBA8_UNORM`: XYZ = `normal * 0.5 + 0.5`, W = roughness)
/// matching `gbuffer.hlsl:97`.
#[inline]
pub fn pack_gbuffer_normal_roughness(normal: [f32; 3], roughness: f32) -> [u8; 4] {
    let n = normalize3_or(normal, [0.0, 1.0, 0.0]);
    [
        f32_to_unorm8(n[0] * 0.5 + 0.5),
        f32_to_unorm8(n[1] * 0.5 + 0.5),
        f32_to_unorm8(n[2] * 0.5 + 0.5),
        f32_to_unorm8(roughness),
    ]
}

/// Unpack DX12 G-Buffer `SV_Target1` into normalized world-space `(normal, roughness)`
/// matching `deferred_combine.hlsl:106-107`.
#[inline]
pub fn unpack_gbuffer_normal_roughness(rgba: [u8; 4]) -> ([f32; 3], f32) {
    let raw = [
        unorm8_to_f32(rgba[0]) * 2.0 - 1.0,
        unorm8_to_f32(rgba[1]) * 2.0 - 1.0,
        unorm8_to_f32(rgba[2]) * 2.0 - 1.0,
    ];
    (normalize3_or(raw, [0.0, 1.0, 0.0]), unorm8_to_f32(rgba[3]))
}

/// Pack DX12 G-Buffer `SV_Target2` (`RGBA8_UNORM`: R = metallic, G = roughness, B = specular, A = material_id)
/// matching `gbuffer.hlsl:98`.
#[inline]
pub fn pack_gbuffer_material(
    metallic: f32,
    roughness: f32,
    specular: f32,
    material_id: u8,
) -> [u8; 4] {
    [
        f32_to_unorm8(metallic),
        f32_to_unorm8(roughness),
        f32_to_unorm8(specular),
        material_id,
    ]
}

/// Unpack DX12 G-Buffer `SV_Target2` into `(metallic, roughness, specular, material_id)`.
#[inline]
pub fn unpack_gbuffer_material(rgba: [u8; 4]) -> (f32, f32, f32, u8) {
    (
        unorm8_to_f32(rgba[0]),
        unorm8_to_f32(rgba[1]),
        unorm8_to_f32(rgba[2]),
        rgba[3],
    )
}

/// Transform a sampled tangent-space normal (`[0, 1]` RGB) into world space using G-Buffer TBN basis
/// (`gbuffer.hlsl:78-81`: `bitangent = cross(normal_ws, tangent_ws.xyz) * tangent_ws.w`).
pub fn decode_tangent_space_normal(
    normal_ws: [f32; 3],
    tangent_ws: [f32; 4],
    normal_sample_rgb: [f32; 3],
) -> [f32; 3] {
    let n = normalize3_or(normal_ws, [0.0, 1.0, 0.0]);
    let t = normalize3_or([tangent_ws[0], tangent_ws[1], tangent_ws[2]], [1.0, 0.0, 0.0]);
    let b_raw = cross3(n, t);
    let b = [
        b_raw[0] * tangent_ws[3],
        b_raw[1] * tangent_ws[3],
        b_raw[2] * tangent_ws[3],
    ];
    let nm = [
        normal_sample_rgb[0] * 2.0 - 1.0,
        normal_sample_rgb[1] * 2.0 - 1.0,
        normal_sample_rgb[2] * 2.0 - 1.0,
    ];
    let combined = [
        nm[0] * t[0] + nm[1] * b[0] + nm[2] * n[0],
        nm[0] * t[1] + nm[1] * b[1] + nm[2] * n[1],
        nm[0] * t[2] + nm[1] * b[2] + nm[2] * n[2],
    ];
    normalize3_or(combined, n)
}

/// Reconstruct a unit tangent-space normal from 2-channel BC5/DXT5nm `(x, y)` normalized texture samples.
#[inline]
pub fn decode_bc5_normal_xy(x_unorm: f32, y_unorm: f32) -> [f32; 3] {
    let x = x_unorm * 2.0 - 1.0;
    let y = y_unorm * 2.0 - 1.0;
    let z = (1.0 - x * x - y * y).max(0.0).sqrt();
    normalize3_or([x, y, z], [0.0, 0.0, 1.0])
}

/// Decoded logical surface sample across DX12's 3 G-Buffer MRTs (`SV_Target0..=2`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GbufferMrtPixel {
    pub albedo: [f32; 3],
    pub occlusion: f32,
    pub normal: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub specular: f32,
    pub material_id: u8,
}

impl Default for GbufferMrtPixel {
    fn default() -> Self {
        Self {
            albedo: [0.0; 3],
            occlusion: 1.0,
            normal: [0.0, 1.0, 0.0],
            metallic: 0.0,
            roughness: 0.5,
            specular: 0.04,
            material_id: u8::MAX,
        }
    }
}

/// Packed `R8G8B8A8_UNORM` pixel triplet matching DX12 G-Buffer targets (`SV_Target0..=2`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GbufferMrtEncoded {
    pub rt0_albedo_ao: [u8; 4],
    pub rt1_normal_roughness: [u8; 4],
    pub rt2_material: [u8; 4],
}

impl GbufferMrtPixel {
    pub fn encode(&self) -> GbufferMrtEncoded {
        GbufferMrtEncoded {
            rt0_albedo_ao: pack_gbuffer_albedo_ao(self.albedo, self.occlusion),
            rt1_normal_roughness: pack_gbuffer_normal_roughness(self.normal, self.roughness),
            rt2_material: pack_gbuffer_material(
                self.metallic,
                self.roughness,
                self.specular,
                self.material_id,
            ),
        }
    }

    pub fn decode(enc: &GbufferMrtEncoded) -> Self {
        let (albedo, occlusion) = unpack_gbuffer_albedo_ao(enc.rt0_albedo_ao);
        let (normal, roughness_n) = unpack_gbuffer_normal_roughness(enc.rt1_normal_roughness);
        let (metallic, roughness_m, specular, material_id) =
            unpack_gbuffer_material(enc.rt2_material);
        let roughness = if roughness_m > 0.0 {
            roughness_m
        } else {
            roughness_n
        };
        Self {
            albedo,
            occlusion,
            normal,
            metallic,
            roughness,
            specular,
            material_id,
        }
    }
}

/// 256-byte `#[repr(C, align(256))]` camera constant buffer matching DX12 `CameraConstants`
/// (`crates/xray-render-dx12/src/geometry_abi.rs`). Matrices are column-major for HLSL column vectors.
#[repr(C, align(256))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dx12CameraConstants {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub padding: [f32; 12],
}

const _: () = assert!(std::mem::size_of::<Dx12CameraConstants>() == 256);
const _: () = assert!(std::mem::align_of::<Dx12CameraConstants>() == 256);

impl Default for Dx12CameraConstants {
    fn default() -> Self {
        Self {
            view: XRAY_IDENTITY_MATRIX,
            proj: XRAY_IDENTITY_MATRIX,
            view_proj: XRAY_IDENTITY_MATRIX,
            cam_pos: [0.0, 0.0, 0.0, 1.0],
            padding: [0.0; 12],
        }
    }
}

impl Dx12CameraConstants {
    /// Convert X-Ray row-vector `view` and `proj` matrices into DX12 column-major CBV layout
    /// matching `mul(ViewProjection, float4(pos, 1.0f))`.
    pub fn from_xray_matrices(view: &XRayMatrix, proj: &XRayMatrix, cam_pos: [f32; 3]) -> Self {
        let vp_xray = matrix_mul_44(proj, view);
        Self {
            view: matrix_transpose_44(view),
            proj: matrix_transpose_44(proj),
            view_proj: matrix_transpose_44(&vp_xray),
            cam_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 1.0],
            padding: [0.0; 12],
        }
    }
}

/// 112-byte packed DX12 depth decompression & inverse view rotation constant block for compute/pixel
/// passes (SSAO/HDAO/SSFX/Deferred).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dx12DepthDecompressParams {
    /// `[2.0, -2.0, -1.0, 1.0]` mapping `[0..1]` UV to D3D `[-1..1]` NDC XY.
    pub uv_to_ndc_scale_bias: [f32; 4],
    /// `[1.0 / proj_00, 1.0 / proj_11, q = proj_22, q_near_neg = -q * near]`.
    pub proj_inv_xy_q_qnear: [f32; 4],
    /// `[near, far, is_ortho (0.0 or 1.0), m43_or_pad]`.
    pub near_far_is_ortho_pad: [f32; 4],
    /// Column 0..2 of inverse view rotation (`right`, `up`, `forward` in `.xyz`).
    pub inv_view_right: [f32; 4],
    pub inv_view_up: [f32; 4],
    pub inv_view_forward: [f32; 4],
    /// World camera position `[x, y, z, 1.0]`.
    pub cam_pos: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Dx12DepthDecompressParams>() == 112);

impl Dx12DepthDecompressParams {
    /// Construct GPU depth decompression constants from X-Ray row-vector `view`, `proj`, and camera pose.
    pub fn from_xray_camera(
        view: &XRayMatrix,
        proj: &XRayMatrix,
        cam_pos: [f32; 3],
        near: f32,
        far: f32,
    ) -> Self {
        let is_ortho = if proj[2][3].abs() <= 1e-6 { 1.0 } else { 0.0 };
        let inv_px = if proj[0][0].abs() > 1e-9 {
            1.0 / proj[0][0]
        } else {
            1.0
        };
        let inv_py = if proj[1][1].abs() > 1e-9 {
            1.0 / proj[1][1]
        } else {
            1.0
        };
        Self {
            uv_to_ndc_scale_bias: [2.0, -2.0, -1.0, 1.0],
            proj_inv_xy_q_qnear: [inv_px, inv_py, proj[2][2], proj[3][2]],
            near_far_is_ortho_pad: [near, far, is_ortho, 0.0],
            inv_view_right: [view[0][0], view[1][0], view[2][0], 0.0],
            inv_view_up: [view[0][1], view[1][1], view[2][1], 0.0],
            inv_view_forward: [view[0][2], view[1][2], view[2][2], 0.0],
            cam_pos: [cam_pos[0], cam_pos[1], cam_pos[2], 1.0],
        }
    }
}

/// Convert `[0, 1]` screen-space texture UV to DX12 clip-space NDC `[-1, 1]` (`Y` up).
#[inline]
pub fn uv_to_ndc_dx12(uv: [f32; 2]) -> [f32; 2] {
    [uv[0] * 2.0 - 1.0, (1.0 - uv[1]) * 2.0 - 1.0]
}

/// Convert DX12 clip-space NDC `[-1, 1]` (`Y` up) to `[0, 1]` screen-space texture UV.
#[inline]
pub fn ndc_to_uv_dx12(ndc: [f32; 2]) -> [f32; 2] {
    [ndc[0] * 0.5 + 0.5, 1.0 - (ndc[1] * 0.5 + 0.5)]
}

/// Reconstruct linear view-space depth `z` from a `[0, 1]` hardware depth value under a standard
/// Direct3D left-handed perspective projection (`q = far / (far - near)`, `z_ndc = q - q * near / z`).
#[inline]
pub fn linearize_depth_perspective_lh(depth: f32, near: f32, far: f32) -> f32 {
    let denom = far - near;
    if denom.abs() <= 1e-6 {
        return depth * far;
    }
    let q = far / denom;
    let d = depth - q;
    if d.abs() <= 1e-7 {
        far
    } else {
        (-q * near) / d
    }
}

/// Reconstruct view-space depth `z` from `[0, 1]` hardware depth using an X-Ray row-vector projection matrix
/// (`proj[2][3] != 0` for perspective, `0` for orthographic).
#[inline]
pub fn linearize_depth_xray_proj(depth: f32, xray_proj: &XRayMatrix) -> f32 {
    if xray_proj[2][3].abs() > 1e-6 {
        let denom = depth - xray_proj[2][2];
        if denom.abs() <= 1e-7 {
            0.0
        } else {
            xray_proj[3][2] / denom
        }
    } else if xray_proj[2][2].abs() > 1e-7 {
        (depth - xray_proj[3][2]) / xray_proj[2][2]
    } else {
        depth
    }
}

/// Reconstruct view-space `[x, y, z]` using X-Ray R4's `PosDecompressParams` and `[0, 1]` texture UV.
#[inline]
pub fn reconstruct_view_pos_r4(
    uv: [f32; 2],
    linear_depth_z: f32,
    params: &PosDecompressParams,
) -> [f32; 3] {
    let ht = params.pos_decompression_params[0];
    let vt = params.pos_decompression_params[1];
    [
        (uv[0] * 2.0 - 1.0) * ht * linear_depth_z,
        (uv[1] * 2.0 - 1.0) * vt * linear_depth_z,
        linear_depth_z,
    ]
}

/// Reconstruct view-space `[x, y, z]` from screen UV `[0, 1]`, hardware depth `[0, 1]`,
/// and an X-Ray row-vector projection matrix (perspective or orthographic).
#[inline]
pub fn reconstruct_view_pos_from_depth_xray(
    uv: [f32; 2],
    ndc_depth: f32,
    xray_proj: &XRayMatrix,
) -> [f32; 3] {
    let ndc = uv_to_ndc_dx12(uv);
    let view_x = ndc[0]
        / if xray_proj[0][0].abs() > 1e-6 {
            xray_proj[0][0]
        } else {
            1.0
        };
    let view_y = ndc[1]
        / if xray_proj[1][1].abs() > 1e-6 {
            xray_proj[1][1]
        } else {
            1.0
        };
    let view_z = linearize_depth_xray_proj(ndc_depth, xray_proj);
    if xray_proj[2][3].abs() > 1e-6 {
        [view_x * view_z, view_y * view_z, view_z]
    } else {
        [view_x, view_y, view_z]
    }
}

/// Reconstruct world-space `[x, y, z]` from screen UV `[0, 1]`, hardware depth `[0, 1]`,
/// and X-Ray row-vector `view`, `proj`, and camera world position.
#[inline]
pub fn reconstruct_world_pos_from_depth_xray(
    uv: [f32; 2],
    ndc_depth: f32,
    xray_view: &XRayMatrix,
    xray_proj: &XRayMatrix,
    cam_pos: [f32; 3],
) -> [f32; 3] {
    let pos_vs = reconstruct_view_pos_from_depth_xray(uv, ndc_depth, xray_proj);
    // Columns of upper-left 3x3 of `xray_view` are camera basis vectors `(right, up, forward)`
    [
        xray_view[0][0] * pos_vs[0]
            + xray_view[0][1] * pos_vs[1]
            + xray_view[0][2] * pos_vs[2]
            + cam_pos[0],
        xray_view[1][0] * pos_vs[0]
            + xray_view[1][1] * pos_vs[1]
            + xray_view[1][2] * pos_vs[2]
            + cam_pos[1],
        xray_view[2][0] * pos_vs[0]
            + xray_view[2][1] * pos_vs[1]
            + xray_view[2][2] * pos_vs[2]
            + cam_pos[2],
    ]
}

/// Reconstruct world-space `[x, y, z]` directly from column-major [`Dx12CameraConstants`].
#[inline]
pub fn reconstruct_world_pos_from_dx12_camera(
    uv: [f32; 2],
    ndc_depth: f32,
    cam: &Dx12CameraConstants,
) -> [f32; 3] {
    let xray_view = matrix_transpose_44(&cam.view);
    let xray_proj = matrix_transpose_44(&cam.proj);
    reconstruct_world_pos_from_depth_xray(
        uv,
        ndc_depth,
        &xray_view,
        &xray_proj,
        [cam.cam_pos[0], cam.cam_pos[1], cam.cam_pos[2]],
    )
}

/// Active camera frame state passed through camera effectors (`SCamEffectorInfo`, `CameraDefs.h:16-42`).
#[derive(Debug, Clone, PartialEq)]
pub struct CamEffectorInfo {
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub normal: [f32; 3],
    pub right: [f32; 3],
    pub fov_deg: f32,
    pub far_plane: f32,
    /// Viewport height / width ratio passed to `Fmatrix::build_projection`.
    pub aspect: f32,
    pub dont_apply: bool,
    pub affected_on_hud: bool,
}

impl Default for CamEffectorInfo {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            fov_deg: 67.5,
            far_plane: 100.0,
            aspect: 9.0 / 16.0,
            dont_apply: false,
            affected_on_hud: true,
        }
    }
}

impl CamEffectorInfo {
    /// Orthonormalize `direction`, `normal`, `right` (`CameraManager.cpp:359-364`).
    pub fn reorthogonalize(&mut self) {
        self.direction = normalize3_or(self.direction, [0.0, 0.0, 1.0]);
        self.normal = normalize3_or(self.normal, [0.0, 1.0, 0.0]);
        self.right = normalize3_or(cross3(self.normal, self.direction), [1.0, 0.0, 0.0]);
        self.normal = normalize3_or(cross3(self.direction, self.right), [0.0, 1.0, 0.0]);
    }

    pub fn view_matrix(&self) -> XRayMatrix {
        matrix_build_camera_dir(self.position, self.direction, self.normal)
    }

    pub fn projection_matrix(&self, near: f32) -> XRayMatrix {
        matrix_build_projection(self.fov_deg.to_radians(), self.aspect, near, self.far_plane)
    }
}

/// Playback driver for `.anm` / `.anms` `ObjectMotion` clips (`CObjectAnimator`, `ObjectAnimator.cpp:86-144`).
#[derive(Debug, Clone, PartialEq)]
pub struct CameraMotionPlayer {
    pub motions: Vec<ObjectMotion>,
    pub active_index: Option<usize>,
    pub elapsed_sec: f32,
    pub speed: f32,
    pub looping: bool,
    pub playing: bool,
    pub xform: XRayMatrix,
}

impl CameraMotionPlayer {
    pub fn from_single(motion: ObjectMotion) -> Self {
        Self {
            motions: vec![motion],
            active_index: None,
            elapsed_sec: 0.0,
            speed: 1.0,
            looping: false,
            playing: false,
            xform: XRAY_IDENTITY_MATRIX,
        }
    }

    pub fn from_library(lib: ObjectMotionLibrary) -> Self {
        Self {
            motions: lib.motions,
            active_index: None,
            elapsed_sec: 0.0,
            speed: 1.0,
            looping: false,
            playing: false,
            xform: XRAY_IDENTITY_MATRIX,
        }
    }

    pub fn play(&mut self, looping: bool, name: Option<&str>) -> Option<&ObjectMotion> {
        let idx = if let Some(target) = name {
            self.motions
                .iter()
                .position(|m| m.name.eq_ignore_ascii_case(target))?
        } else if !self.motions.is_empty() {
            0
        } else {
            return None;
        };
        self.active_index = Some(idx);
        self.elapsed_sec = 0.0;
        self.looping = looping;
        self.playing = true;
        self.xform = XRAY_IDENTITY_MATRIX;
        Some(&self.motions[idx])
    }

    pub fn stop(&mut self) {
        self.playing = false;
        self.active_index = None;
        self.xform = XRAY_IDENTITY_MATRIX;
    }

    pub fn active_duration(&self) -> f32 {
        self.active_index
            .and_then(|i| self.motions.get(i))
            .map(ObjectMotion::duration)
            .unwrap_or(0.0)
    }

    /// Evaluate the active clip at current time, advance playback by `dt * speed`,
    /// and update `self.xform` matching `CObjectAnimator::Update` (`ObjectAnimator.cpp:86-96`).
    pub fn update(&mut self, dt: f32) -> Option<([f32; 3], [f32; 3])> {
        let idx = self.active_index?;
        let motion = self.motions.get(idx)?;
        if !self.playing {
            return None;
        }
        let duration = motion.duration();
        let t_eval = if duration > 0.0 && self.looping {
            self.elapsed_sec.rem_euclid(duration)
        } else if duration > 0.0 {
            self.elapsed_sec.clamp(0.0, duration)
        } else {
            self.elapsed_sec
        };

        let (pos, rot) = motion.evaluate(t_eval);
        self.xform = matrix_set_xyzi_translate(rot, pos);

        let next = self.elapsed_sec + dt.max(0.0) * self.speed;
        if duration > 0.0 {
            if self.looping {
                self.elapsed_sec = next.rem_euclid(duration);
            } else if next >= duration {
                self.elapsed_sec = duration;
                self.playing = false;
            } else {
                self.elapsed_sec = next;
            }
        } else {
            self.elapsed_sec = next;
        }
        Some((pos, rot))
    }
}

/// Camera-space `.anm` effector (`CEffectorCam` / `CAnimatorCamEffector`) transforming `CamEffectorInfo`.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraMotionEffector {
    pub effector_type: u32,
    pub player: CameraMotionPlayer,
    pub absolute_positioning: bool,
    pub amplitude: f32,
}

impl CameraMotionEffector {
    pub fn new(effector_type: u32, motion: ObjectMotion, looping: bool) -> Self {
        let mut player = CameraMotionPlayer::from_single(motion);
        player.play(looping, None);
        Self {
            effector_type,
            player,
            absolute_positioning: false,
            amplitude: 1.0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.player.playing
    }

    /// Step the `.anm` clip by `dt` and apply camera offset/orientation (`ProcessCam`),
    /// returning `false` once finished.
    pub fn process_cam(&mut self, info: &mut CamEffectorInfo, dt: f32) -> bool {
        let was_playing = self.player.playing;
        let Some((pos, _rot)) = self.player.update(dt) else {
            return false;
        };
        if !was_playing {
            return false;
        }
        let xf = self.player.xform;
        if self.absolute_positioning {
            info.position = [pos[0] * self.amplitude, pos[1] * self.amplitude, pos[2] * self.amplitude];
            info.right = [xf[0][0], xf[0][1], xf[0][2]];
            info.normal = [xf[1][0], xf[1][1], xf[1][2]];
            info.direction = [xf[2][0], xf[2][1], xf[2][2]];
        } else {
            let amp = self.amplitude;
            // Transform local delta into current camera basis
            for axis in 0..3 {
                info.position[axis] += (info.right[axis] * pos[0]
                    + info.normal[axis] * pos[1]
                    + info.direction[axis] * pos[2])
                    * amp;
            }
            let r = info.right;
            let n = info.normal;
            let d = info.direction;
            let blend_axis = |basis: [f32; 3]| -> [f32; 3] {
                let mut out = [0.0f32; 3];
                for k in 0..3 {
                    out[k] = r[k] * basis[0] + n[k] * basis[1] + d[k] * basis[2];
                }
                out
            };
            let target_r = blend_axis([xf[0][0], xf[0][1], xf[0][2]]);
            let target_n = blend_axis([xf[1][0], xf[1][1], xf[1][2]]);
            let target_d = blend_axis([xf[2][0], xf[2][1], xf[2][2]]);
            for k in 0..3 {
                info.right[k] += (target_r[k] - info.right[k]) * amp;
                info.normal[k] += (target_n[k] - info.normal[k]) * amp;
                info.direction[k] += (target_d[k] - info.direction[k]) * amp;
            }
        }
        info.reorthogonalize();
        self.player.playing
    }
}

/// Active `.ppe` postprocess effector (`CPostprocessAnimator`, `PostprocessAnimator.cpp:44-248`).
#[derive(Debug, Clone, PartialEq)]
pub struct PostprocessPlayer {
    pub effector_type: u32,
    pub effect: PostprocessEffect,
    pub cyclic: bool,
    pub overlap: bool,
    pub life_time: f32,
    pub elapsed_sec: f32,
    pub length: f32,
    pub factor: f32,
    pub dest_factor: f32,
    pub factor_speed: f32,
    pub stopping: bool,
}

impl PostprocessPlayer {
    pub fn new(effector_type: u32, effect: PostprocessEffect, cyclic: bool, overlap: bool) -> Self {
        let length = effect.length();
        Self {
            effector_type,
            effect,
            cyclic,
            overlap,
            life_time: if cyclic { 100_000.0 } else { length },
            elapsed_sec: 0.0,
            length,
            factor: 1.0,
            dest_factor: 1.0,
            factor_speed: 1.0,
            stopping: false,
        }
    }

    pub fn stop(&mut self, speed: f32) {
        if !self.stopping {
            self.stopping = true;
            self.factor_speed = speed;
        }
    }

    pub fn is_valid(&self) -> bool {
        self.cyclic || self.life_time > 0.0
    }

    /// Advance by `dt` seconds and return blended `PostprocessState`, or `None` once expired
    /// (`CPostprocessAnimator::Process`, `PostprocessAnimator.cpp:190-248`).
    pub fn step(&mut self, dt: f32) -> Option<PostprocessState> {
        if !self.is_valid() {
            return None;
        }
        if self.cyclic {
            self.life_time = 100_000.0;
        } else {
            self.life_time -= dt.max(0.0);
        }

        self.elapsed_sec += dt.max(0.0);
        if self.cyclic && self.length > 0.0 && self.elapsed_sec > self.length {
            self.elapsed_sec %= self.length;
        }

        if self.stopping {
            self.factor -= dt.max(0.0) * self.factor_speed;
        } else {
            self.factor += self.factor_speed * dt.max(0.0) * (self.dest_factor - self.factor);
        }
        self.factor = self.factor.clamp(0.0001, 1.0);

        if (self.factor - 0.0001).abs() <= 1e-6 && self.stopping {
            return None;
        }

        let evaluated = self.effect.evaluate(self.elapsed_sec);
        let id = PostprocessState::identity();
        let blended = PostprocessState::lerp(&id, &evaluated, self.factor);
        Some(blended)
    }
}

/// Mix active postprocess effectors into one final `PostprocessState`
/// (`CCameraManager::UpdatePPEffectors`, `CameraManager.cpp:425-470`).
pub fn mix_postprocess_effectors(dt: f32, effectors: &mut Vec<PostprocessPlayer>) -> PostprocessState {
    let identity = PostprocessState::identity();
    let mut pp_affected = identity.clone();
    let mut count = 0usize;
    let mut blocked_overlap = false;

    let mut i = effectors.len();
    while i > 0 {
        i -= 1;
        let maybe_pp = effectors[i].step(dt);
        if let Some(l_pp_inf) = maybe_pp {
            let overlap = effectors[i].overlap;
            count += 1;
            if !blocked_overlap {
                pp_affected.add_pp(&l_pp_inf);
                pp_affected.sub_pp(&identity);
            }
            if !overlap {
                blocked_overlap = true;
                pp_affected = l_pp_inf;
            }
        } else {
            effectors.remove(i);
        }
    }

    if count == 0 {
        pp_affected = identity.clone();
    }
    if pp_affected.noise_granularity <= 0.0 {
        pp_affected.noise_granularity = identity.noise_granularity;
    }
    pp_affected
}

/// Complete camera + postprocess manager (`CCameraManager`, `CameraManager.cpp:333-470`).
#[derive(Debug, Clone, PartialEq)]
pub struct CameraManagerState {
    pub cam_info: CamEffectorInfo,
    pub cam_inert: f32,
    pub cam_effectors: Vec<CameraMotionEffector>,
    pub pp_effectors: Vec<PostprocessPlayer>,
    pub pp_affected: PostprocessState,
}

impl Default for CameraManagerState {
    fn default() -> Self {
        Self {
            cam_info: CamEffectorInfo::default(),
            cam_inert: 0.0,
            cam_effectors: Vec::new(),
            pp_effectors: Vec::new(),
            pp_affected: PostprocessState::identity(),
        }
    }
}

impl CameraManagerState {
    pub fn add_cam_effector(&mut self, eff: CameraMotionEffector) {
        self.remove_cam_effector(eff.effector_type);
        if eff.player.playing {
            if eff.absolute_positioning {
                self.cam_effectors.insert(0, eff);
            } else {
                self.cam_effectors.push(eff);
            }
        }
    }

    pub fn remove_cam_effector(&mut self, effector_type: u32) {
        self.cam_effectors
            .retain(|e| e.effector_type != effector_type);
    }

    pub fn add_pp_effector(&mut self, eff: PostprocessPlayer) {
        self.remove_pp_effector(eff.effector_type);
        self.pp_effectors.push(eff);
    }

    pub fn remove_pp_effector(&mut self, effector_type: u32) {
        self.pp_effectors
            .retain(|e| e.effector_type != effector_type);
    }

    /// Update base camera pose with `psCamInert` inertia, FOV/far/aspect exponential blend (`10 * dt`),
    /// active camera effectors (reverse order), and postprocess stack (`CCameraManager::Update`).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dt: f32,
        target_pos: [f32; 3],
        target_dir: [f32; 3],
        target_up: [f32; 3],
        fov_dest_deg: f32,
        aspect_dest: f32,
        far_dest: f32,
        position_rigid: bool,
        direction_rigid: bool,
    ) {
        let inert = if position_rigid {
            0.0
        } else {
            self.cam_inert.clamp(0.0, 1.0)
        };
        let dir_inert = if direction_rigid {
            0.0
        } else {
            self.cam_inert.clamp(0.0, 1.0)
        };
        for axis in 0..3 {
            self.cam_info.position[axis] =
                self.cam_info.position[axis] * inert + target_pos[axis] * (1.0 - inert);
            self.cam_info.direction[axis] =
                self.cam_info.direction[axis] * dir_inert + target_dir[axis] * (1.0 - dir_inert);
            self.cam_info.normal[axis] =
                self.cam_info.normal[axis] * dir_inert + target_up[axis] * (1.0 - dir_inert);
        }
        self.cam_info.reorthogonalize();

        let src = (10.0 * dt.max(0.0)).clamp(0.0, 1.0);
        let dst = 1.0 - src;
        self.cam_info.fov_deg = self.cam_info.fov_deg * dst + fov_dest_deg * src;
        self.cam_info.far_plane = self.cam_info.far_plane * dst + far_dest * src;
        self.cam_info.aspect = self.cam_info.aspect * dst + aspect_dest * src;
        self.cam_info.dont_apply = false;

        let mut i = self.cam_effectors.len();
        while i > 0 {
            i -= 1;
            let keep = if self.cam_effectors[i].is_valid() {
                self.cam_effectors[i].process_cam(&mut self.cam_info, dt)
            } else {
                false
            };
            if !keep {
                self.cam_effectors.remove(i);
            }
        }
        self.cam_info.reorthogonalize();
        self.pp_affected = mix_postprocess_effectors(dt, &mut self.pp_effectors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelopes::{
        Envelope, EnvelopeBehavior, EnvelopeKey, EnvelopeShape, PostprocessColorCurve,
    };

    fn const_env(val: f32) -> Envelope {
        Envelope {
            behavior: [EnvelopeBehavior::Constant, EnvelopeBehavior::Constant],
            keys: vec![
                EnvelopeKey {
                    value: val,
                    time: 0.0,
                    shape: EnvelopeShape::Linear,
                    ..EnvelopeKey::default()
                },
                EnvelopeKey {
                    value: val,
                    time: 1.0,
                    shape: EnvelopeShape::Linear,
                    ..EnvelopeKey::default()
                },
            ],
        }
    }

    #[test]
    fn camera_view_and_projection_matrices_match_xray_conventions() {
        let view = matrix_build_camera_dir([10.0, 2.0, -5.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let eye_local = crate::ogf::matrix_transform_point(&view, [10.0, 2.0, -5.0]);
        assert!((eye_local[0]).abs() < 1e-6);
        assert!((eye_local[1]).abs() < 1e-6);
        assert!((eye_local[2]).abs() < 1e-6);

        let pt_ahead = crate::ogf::matrix_transform_point(&view, [13.0, 6.0, 5.0]);
        assert!((pt_ahead[0] - 3.0).abs() < 1e-6);
        assert!((pt_ahead[1] - 4.0).abs() < 1e-6);
        assert!((pt_ahead[2] - 10.0).abs() < 1e-6);

        // Build projection (90deg vertical FOV -> HAT = 1, aspect = 0.5)
        let proj = matrix_build_projection(std::f32::consts::FRAC_PI_2, 0.5, 0.2, 100.0);
        assert!((proj[0][0] - 0.5).abs() < 1e-5);
        assert!((proj[1][1] - 1.0).abs() < 1e-5);
        assert!((proj[2][3] - 1.0).abs() < 1e-6);

        let ortho = matrix_build_projection_ortho(20.0, 10.0, 2.0, 12.0);
        let near_p = crate::ogf::matrix_transform_point(&ortho, [10.0, -5.0, 2.0]);
        assert!((near_p[0] - 1.0).abs() < 1e-5);
        assert!((near_p[1] - (-1.0)).abs() < 1e-5);
        assert!((near_p[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn camera_manager_motion_and_postprocess_overlap_stacking() {
        let mut cam_mgr = CameraManagerState::default();

        // Motion translating +2 along Z over 1 second
        let motion = ObjectMotion {
            name: "head_bob".to_string(),
            frame_start: 0,
            frame_end: 30,
            fps: 30.0,
            version: 5,
            envelopes: [
                const_env(0.0),
                const_env(0.0),
                Envelope {
                    behavior: [EnvelopeBehavior::Constant, EnvelopeBehavior::Constant],
                    keys: vec![
                        EnvelopeKey {
                            value: 0.0,
                            time: 0.0,
                            shape: EnvelopeShape::Linear,
                            ..EnvelopeKey::default()
                        },
                        EnvelopeKey {
                            value: 2.0,
                            time: 1.0,
                            shape: EnvelopeShape::Linear,
                            ..EnvelopeKey::default()
                        },
                    ],
                },
                const_env(0.0),
                const_env(0.0),
                const_env(0.0),
            ],
        };
        let eff_cam = CameraMotionEffector::new(100, motion, false);
        cam_mgr.add_cam_effector(eff_cam);

        // Two additive overlapping postprocess effects (blur = 0.2 and blur = 0.3 => 0.5)
        let make_ppe = |blur_val: f32, tex: &str| PostprocessEffect {
            version: 2,
            base_color: PostprocessColorCurve {
                base: 0.0,
                red: const_env(0.0),
                green: const_env(0.0),
                blue: const_env(0.0),
            },
            add_color: PostprocessColorCurve {
                base: 0.0,
                red: const_env(0.0),
                green: const_env(0.0),
                blue: const_env(0.0),
            },
            gray_color: PostprocessColorCurve {
                base: 0.0,
                red: const_env(0.0),
                green: const_env(0.0),
                blue: const_env(0.0),
            },
            gray: const_env(0.0),
            blur: const_env(blur_val),
            duality_h: const_env(0.0),
            duality_v: const_env(0.0),
            noise_intensity: Envelope::default(),
            noise_granularity: Envelope::default(),
            noise_fps: Envelope::default(),
            cm_influence: const_env(0.4),
            cm_tex1: tex.to_string(),
        };

        cam_mgr.add_pp_effector(PostprocessPlayer::new(1, make_ppe(0.2, "fx_a"), true, true));
        cam_mgr.add_pp_effector(PostprocessPlayer::new(2, make_ppe(0.3, "fx_b"), true, true));

        cam_mgr.update(
            0.5,
            [0.0, 1.7, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            75.0,
            0.5625,
            300.0,
            true,
            true,
        );
        // After dt=0.5: player evaluated at t=0.0 first before advancing
        assert!((cam_mgr.pp_affected.blur - 0.5).abs() < 1e-4);
        assert_eq!(cam_mgr.pp_affected.cm_tex1, "fx_b");
        assert_eq!(cam_mgr.pp_affected.cm_tex2, "fx_a");

        // Second step evaluates motion at t=0.5 (pos.z = 1.0)
        cam_mgr.update(
            0.6,
            [0.0, 1.7, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            75.0,
            0.5625,
            300.0,
            true,
            true,
        );
        assert!((cam_mgr.cam_info.position[2] - 1.0).abs() < 1e-4);
        // Non-looping 1.0s motion finished after elapsed 0.5+0.6 >= 1.0
        assert!(cam_mgr.cam_effectors.is_empty());

        // Non-overlapping top effector overrides stack completely
        cam_mgr.add_pp_effector(PostprocessPlayer::new(3, make_ppe(0.9, "fx_override"), true, false));
        let pp_override = mix_postprocess_effectors(0.016, &mut cam_mgr.pp_effectors);
        assert!((pp_override.blur - 0.9).abs() < 1e-4);
        assert_eq!(pp_override.cm_tex1, "fx_override");
    }

    #[test]
    fn ui_3d_icon_hud_projection_and_pos_decompress_params() {
        let aspect = 9.0 / 16.0;
        let p_icon = ui_3d_icon_projection(aspect);
        assert!((p_icon[0][0] - (2.0 * aspect / 2.0)).abs() < 1e-6);
        assert!((p_icon[1][1] - 1.0).abs() < 1e-6);
        let p_atlas = ui_3d_icon_atlas_projection(1920, 1080);
        assert!((p_atlas[0][0] - p_icon[0][0]).abs() < 1e-5);
        assert!((p_atlas[1][1] - p_icon[1][1]).abs() < 1e-6);

        let p_hud = hud_projection_matrix(DEFAULT_HUD_FOV_FACTOR, aspect, 150.0);
        let expected_hat = (DEFAULT_HUD_FOV_FACTOR * HUD_BASE_FOV_DEG * 0.5)
            .to_radians()
            .tan();
        assert!((p_hud[1][1] - 1.0 / expected_hat).abs() < 1e-5);
        assert!((p_hud[0][0] - aspect / expected_hat).abs() < 1e-5);

        let pd = PosDecompressParams::compute(90.0, 0.5, 1920, 1080, DEFAULT_PARALLAX_H, 50.0);
        assert_eq!(std::mem::size_of::<PosDecompressParams>(), 48);
        assert!((pd.pos_decompression_params[0] - 2.0).abs() < 1e-5);
        assert!((pd.pos_decompression_params[1] - (-1.0)).abs() < 1e-5);
        assert!((pd.pos_decompression_params[2] - 4.0 / 1920.0).abs() < 1e-6);
        assert!((pd.pos_decompression_params[3] - (-2.0 / 1080.0)).abs() < 1e-6);
        assert_eq!(
            pd.pos_decompression_params2,
            [1920.0, 1080.0, 1.0 / 1920.0, 1.0 / 1080.0]
        );
        assert_eq!(pd.parallax, [0.02, -0.01, 0.02, 0.02]);
    }

    #[test]
    fn render_xforms_maintains_current_prev_and_lazy_inv_w() {
        use crate::ogf::{matrix_set_xyz_translate, matrix_transform_point, matrix_transform_projected};

        let mut xf = RenderXforms::new();
        assert!(xf.inv_w_valid());
        assert_eq!(xf.m_wvp, XRAY_IDENTITY_MATRIX);

        let w = matrix_set_xyz_translate([0.0, 0.4, 0.0], [3.0, 1.0, 8.0]);
        let v = matrix_build_camera_dir([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        let p = matrix_build_projection(std::f32::consts::FRAC_PI_2, 1.0, 0.2, 100.0);

        xf.set_p(p);
        xf.set_v(v);
        xf.set_w(w);
        assert!(!xf.inv_w_valid());

        let local_pt = [1.0, 2.0, 3.0];
        let world_pt = matrix_transform_point(&xf.m_w, local_pt);
        let view_pt = matrix_transform_point(&xf.m_v, world_pt);
        let view_pt_wv = matrix_transform_point(&xf.m_wv, local_pt);
        for k in 0..3 {
            assert!((view_pt[k] - view_pt_wv[k]).abs() < 1e-5);
        }

        let ndc_seq = matrix_transform_projected(&xf.m_vp, world_pt).unwrap();
        let ndc_wvp = matrix_transform_projected(&xf.m_wvp, local_pt).unwrap();
        for k in 0..3 {
            assert!((ndc_seq[k] - ndc_wvp[k]).abs() < 1e-5);
        }

        let inv_w = *xf.apply_inv_w().unwrap();
        assert!(xf.inv_w_valid());
        let round_trip = matrix_transform_point(&inv_w, world_pt);
        for k in 0..3 {
            assert!((round_trip[k] - local_pt[k]).abs() < 1e-5);
        }

        xf.snapshot_to_prev();
        assert_eq!(xf.m_wvp_prev, xf.m_wvp);
        let w2 = matrix_set_xyz_translate([0.0, 0.0, 0.0], [10.0, 0.0, 12.0]);
        xf.set_w(w2);
        assert_ne!(xf.m_wvp, xf.m_wvp_prev);
        xf.set_w_prev(w2);
        xf.set_v_prev(v);
        xf.set_p_prev(p);
        assert_eq!(xf.m_wvp_prev, xf.m_wvp);
    }

    #[test]
    fn gbuffer_mrt_codecs_and_dx12_depth_world_pos_reconstruction() {
        let pix = GbufferMrtPixel {
            albedo: [0.8, 0.4, 0.2],
            occlusion: 0.9,
            normal: normalize3_or([0.3, 0.9, -0.25], [0.0, 1.0, 0.0]),
            metallic: 0.75,
            roughness: 0.35,
            specular: 0.04,
            material_id: 42,
        };
        let enc = pix.encode();
        let dec = GbufferMrtPixel::decode(&enc);
        for k in 0..3 {
            assert!((dec.albedo[k] - pix.albedo[k]).abs() < 1e-2);
            assert!((dec.normal[k] - pix.normal[k]).abs() < 1.5e-2);
        }
        assert!((dec.occlusion - 0.9).abs() < 1e-2);
        assert!((dec.metallic - 0.75).abs() < 1e-2);
        assert!((dec.roughness - 0.35).abs() < 1e-2);
        assert!((dec.specular - 0.04).abs() < 1e-2);
        assert_eq!(dec.material_id, 42);

        // Tangent space normal mapping matching gbuffer.hlsl:78-81
        let ws_n = decode_tangent_space_normal(
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 1.0],
        );
        assert!((ws_n[1] - 1.0).abs() < 1e-5);
        let bc5 = decode_bc5_normal_xy(0.5, 0.5);
        assert!((bc5[2] - 1.0).abs() < 1e-5);

        // Perspective world -> clip -> UV/depth -> reconstructed world roundtrip
        let cam_pos = [12.0, 3.5, -8.0];
        let view = matrix_build_camera_dir(cam_pos, [0.2, -0.1, 1.0], [0.0, 1.0, 0.0]);
        let proj = matrix_build_projection(75.0f32.to_radians(), 9.0 / 16.0, 0.2, 200.0);
        let dx_cam = Dx12CameraConstants::from_xray_matrices(&view, &proj, cam_pos);
        assert_eq!(std::mem::size_of::<Dx12CameraConstants>(), 256);

        let target_ws = [15.0, 2.0, 9.5];
        let vp_xray = matrix_mul_44(&proj, &view);
        let ndc = crate::ogf::matrix_transform_projected(&vp_xray, target_ws).unwrap();
        let uv = ndc_to_uv_dx12([ndc[0], ndc[1]]);
        let lin_z = linearize_depth_perspective_lh(ndc[2], 0.2, 200.0);
        assert!((lin_z - crate::ogf::matrix_transform_point(&view, target_ws)[2]).abs() < 1e-3);

        let rec_ws = reconstruct_world_pos_from_dx12_camera(uv, ndc[2], &dx_cam);
        for k in 0..3 {
            assert!((rec_ws[k] - target_ws[k]).abs() < 1e-2, "axis {k}");
        }

        let dp = Dx12DepthDecompressParams::from_xray_camera(&view, &proj, cam_pos, 0.2, 200.0);
        assert_eq!(dp.near_far_is_ortho_pad[2], 0.0);
        assert_eq!(std::mem::size_of::<Dx12DepthDecompressParams>(), 112);

        // R4 PosDecompressParams reconstruction check
        let r4_pd = PosDecompressParams::compute(75.0, 9.0 / 16.0, 1920, 1080, 0.02, 50.0);
        let r4_vs = reconstruct_view_pos_r4([0.5, 0.5], 10.0, &r4_pd);
        assert!(r4_vs[0].abs() < 1e-6 && r4_vs[1].abs() < 1e-6 && (r4_vs[2] - 10.0).abs() < 1e-6);

        // Orthographic depth reconstruction
        let ortho = matrix_build_projection_ortho(20.0, 10.0, 1.0, 21.0);
        let o_ndc = crate::ogf::matrix_transform_projected(&ortho, [4.0, -2.5, 11.0]).unwrap();
        let o_uv = ndc_to_uv_dx12([o_ndc[0], o_ndc[1]]);
        let o_vs = reconstruct_view_pos_from_depth_xray(o_uv, o_ndc[2], &ortho);
        assert!((o_vs[0] - 4.0).abs() < 1e-4);
        assert!((o_vs[1] - (-2.5)).abs() < 1e-4);
        assert!((o_vs[2] - 11.0).abs() < 1e-4);
    }
}
