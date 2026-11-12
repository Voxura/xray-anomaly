//! Pure-Rust X-Ray Phase-13 Post-Processing, Bloom, Luminance, Combine, Tonemapping, Gamma LUT & HDR10 Constants
//!
//! Grounds C++ renderer and HLSL shader parameters directly from:
//! - `src/Layers/xrRenderPC_R3/r3_rendertarget_phase_PP.cpp` (`u_calc_tc_noise`)
//! - `src/Layers/xrRenderPC_R3/r3_rendertarget_phase_bloom.cpp` (`CalcGauss_k7`, `CalcGauss_wave`, UV/tap builders)
//! - `src/Layers/xrRenderPC_R4/r4_rendertarget_phase_luminance.cpp` (`phase_luminance` 256->64->8->1 & `MiddleGray`)
//! - `src/Layers/xrRender/rendertarget_phase_sunshafts.cpp` (`phase_sunshafts` 5-pass schedule)
//! - `src/Layers/xrRenderPC_R4/r4_rendertarget_phase_combine.cpp` (`combine_1`, `combine_2`, `tnmp_*`, `vignette_control`)
//! - `src/Layers/xrRender/xr_effgamma.cpp` (`CGammaControl::GenLUT`)
//! - `src/Layers/xrRender/Blender_Recorder_StandartBinding.cpp` (`hdr10_parameters1..10`)
//! - `src/Layers/xrRender/rendertarget_phase_lut.cpp` & `src/Layers/xrRenderPC_R4/blender_lut.cpp`
//! - `src/Layers/xrRender/r__occlusion.h` & `r__occlusion.cpp` (`R_occlusion`)
//! - `gamedata/shaders/r3/{hdr10.h,combine_2_naa.ps,effects_sun.ps}`

use crate::envelopes::{PostprocessGpuParams, PostprocessState};
use crate::lighting::{compute_u_diffuse2s, GlossConfig, TEX_JITTER_SIZE};
use crate::ogf::{
    matrix_mul_44, matrix_transform_point, matrix_transform_projected, XRayMatrix,
    XRAY_IDENTITY_MATRIX,
};
use crate::particle_sim::ParticleRng;
use crate::shaders_xr::{CmpFunc, CullMode, StencilOp, StencilState};
use crate::skinning::{matrix_transform_dir, normalize_vec3};
use crate::weather::MixedEnvironment;

const EPS_S: f32 = 1e-7;

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let denom = edge1 - edge0;
    if denom.abs() <= 1e-7 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / denom).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn mul_mat3_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [dot3(m[0], v), dot3(m[1], v), dot3(m[2], v)]
}

// ============================================================================
// Screen-Space PPE Noise State (`CRenderTarget::u_calc_tc_noise`)
// ============================================================================

/// Persistent animated UV jitter state for full-screen grain noise
/// (`CRenderTarget::u_calc_tc_noise`, `r3_rendertarget_phase_PP.cpp:4-35`).
#[derive(Debug, Clone)]
pub struct PostprocessNoiseState {
    pub im_noise_time: f32,
    pub im_noise_shift_w: u32,
    pub im_noise_shift_h: u32,
    rng: ParticleRng,
}

impl Default for PostprocessNoiseState {
    fn default() -> Self {
        Self::new(1)
    }
}

impl PostprocessNoiseState {
    pub fn new(seed: u64) -> Self {
        Self {
            im_noise_time: 0.0,
            im_noise_shift_w: 0,
            im_noise_shift_h: 0,
            rng: ParticleRng::new(seed),
        }
    }

    /// Advance noise timer and compute tiled noise UV rectangle `[start_u, start_v, end_u, end_v]`
    /// matching `CRenderTarget::u_calc_tc_noise` (`r3_rendertarget_phase_PP.cpp:4-35`).
    #[allow(clippy::too_many_arguments)]
    pub fn step_uv_rect(
        &mut self,
        dt_sec: f32,
        noise_tex_w: u32,
        noise_tex_h: u32,
        screen_w: u32,
        screen_h: u32,
        noise_scale: f32,
        noise_fps: f32,
    ) -> [f32; 4] {
        let scale = if noise_scale.is_finite() && noise_scale > 0.0 {
            noise_scale
        } else {
            1.0
        };
        let tw = ((noise_tex_w.max(1) as f32) * scale + EPS_S)
            .ceil()
            .max(1.0) as u32;
        let th = ((noise_tex_h.max(1) as f32) * scale + EPS_S)
            .ceil()
            .max(1.0) as u32;

        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        self.im_noise_time -= dt;
        if self.im_noise_time < 0.0 {
            self.im_noise_shift_w = self.rng.range_usize(tw as usize) as u32;
            self.im_noise_shift_h = self.rng.range_usize(th as usize) as u32;
            let fps = if noise_fps.is_finite() && noise_fps.abs() > 1e-4 {
                noise_fps.abs()
            } else {
                30.0
            };
            let fps_time = 1.0 / fps;
            let steps = ((-self.im_noise_time) / fps_time).floor() + 1.0;
            self.im_noise_time += steps * fps_time;
            if self.im_noise_time < 0.0 {
                self.im_noise_time = 0.0;
            }
        }

        let start_u = (self.im_noise_shift_w as f32 + 0.5) / tw as f32;
        let start_v = (self.im_noise_shift_h as f32 + 0.5) / th as f32;
        let cnt_w = screen_w / tw;
        let cnt_h = screen_h / th;
        let end_u = start_u + cnt_w as f32 + 1.0;
        let end_v = start_v + cnt_h as f32 + 1.0;
        [start_u, start_v, end_u, end_v]
    }
}

// ============================================================================
// Bloom Kernels, Adaptation & UV Setup (`r3_rendertarget_phase_bloom.cpp`)
// ============================================================================

/// Compute 15-tap symmetrical 1D Gaussian kernel packed as `(w0, w1)` matching
/// `CalcGauss_k7` (`r3_rendertarget_phase_bloom.cpp:33-51`):
/// - `w0 = [W[1], W[2], W[3], W[4]]` (`-1..-4`)
/// - `w1 = [W[5], W[6], W[7], W[0]]` (`-5..-7`, center `0`)
pub fn calc_gauss_k7(radius: f32, s_out: f32) -> ([f32; 4], [f32; 4]) {
    let r = if radius.is_finite() {
        radius.abs().max(1e-3)
    } else {
        3.3
    };
    let scale = if s_out.is_finite() { s_out } else { 1.0 };

    let mut w = [0.0f32; 8];
    for (neg_i, slot) in w.iter_mut().enumerate() {
        let i_f = -(neg_i as i32) as f32;
        *slot = (-(i_f * i_f) / (2.0 * r * r)).exp();
    }

    let mut mag = 0.0f32;
    for (idx, &val) in w.iter().enumerate() {
        mag += if idx != 0 { 2.0 * val } else { val };
    }
    if mag > 0.0 {
        for slot in &mut w {
            *slot = scale * *slot / mag;
        }
    }

    ([w[1], w[2], w[3], w[4]], [w[5], w[6], w[7], w[0]])
}

/// Sum two normalized `calc_gauss_k7` kernels (`CalcGauss_wave`, `r3_rendertarget_phase_bloom.cpp:53-66`).
pub fn calc_gauss_wave(r_base: f32, r_detail: f32, s_out: f32) -> ([f32; 4], [f32; 4]) {
    let (mut w0, mut w1) = calc_gauss_k7(r_base, s_out);
    let (t0, t1) = calc_gauss_k7(r_detail, s_out);
    for k in 0..4 {
        w0[k] += t0[k];
        w1[k] += t1[k];
    }
    (w0, w1)
}

/// Legacy & R3/DX12 bloom parameters (`ps_r2_ls_bloom_*` in `xrRender_console.cpp:210-217`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BloomConfig {
    pub kernel_g: f32,
    pub kernel_b: f32,
    pub speed: f32,
    pub kernel_scale: f32,
    pub threshold: f32,
    pub fast_bloom: bool,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            kernel_g: 3.0,
            kernel_b: 0.7,
            speed: 100.0,
            kernel_scale: 0.7,
            threshold: 1.0,
            fast_bloom: false,
        }
    }
}

/// 32-byte 16-byte-aligned Gaussian weight pair (`weight[0]`, `weight[1]`) for bloom blur passes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BloomFilterGpuWeights {
    pub weight0: [f32; 4],
    pub weight1: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<BloomFilterGpuWeights>() == 32);

impl BloomFilterGpuWeights {
    #[inline]
    pub fn horizontal_wave(cfg: &BloomConfig) -> Self {
        let (weight0, weight1) =
            calc_gauss_wave(cfg.kernel_g, cfg.kernel_g / 3.0, cfg.kernel_scale);
        Self { weight0, weight1 }
    }

    #[inline]
    pub fn vertical_wave(cfg: &BloomConfig, screen_w: u32, screen_h: u32) -> Self {
        let aspect = screen_h.max(1) as f32 / screen_w.max(1) as f32;
        let kernel = cfg.kernel_g * aspect;
        let (weight0, weight1) = calc_gauss_wave(kernel, kernel / 3.0, cfg.kernel_scale);
        Self { weight0, weight1 }
    }
}

/// Bloom exposure-adaptation state (`CRenderTarget::f_bloom_factor`, `r3_rendertarget_phase_bloom.cpp:136`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BloomAdaptState {
    pub bloom_factor: f32,
}

impl Default for BloomAdaptState {
    fn default() -> Self {
        Self { bloom_factor: 1.0 }
    }
}

impl BloomAdaptState {
    #[inline]
    pub fn with_factor(bloom_factor: f32) -> Self {
        Self { bloom_factor }
    }

    /// Advance exponential adaptation `0.9 * prev + 0.1 * speed * dt` and return updated `bloom_factor`.
    #[inline]
    pub fn step(&mut self, dt_sec: f32, speed: f32) -> f32 {
        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        let sp = if speed.is_finite() {
            speed.max(0.0)
        } else {
            0.0
        };
        self.bloom_factor = 0.9 * self.bloom_factor + 0.1 * sp * dt;
        self.bloom_factor
    }

    /// Shader uniform `b_params = (threshold, threshold, threshold, bloom_factor)`.
    #[inline]
    pub fn b_params(&self, threshold: f32) -> [f32; 4] {
        [threshold, threshold, threshold, self.bloom_factor]
    }
}

/// Compute initial 4-tap bloom build UV corners `(a[0..4], b[0..4])`
/// matching `CRenderTarget::phase_bloom` (`r3_rendertarget_phase_bloom.cpp:85-105`).
pub fn bloom_build_4tap_uvs(
    screen_w: u32,
    screen_h: u32,
    bloom_w: u32,
    bloom_h: u32,
) -> ([[f32; 2]; 4], [[f32; 2]; 4]) {
    let w = screen_w.max(1) as f32;
    let h = screen_h.max(1) as f32;
    let tw = bloom_w.max(1) as f32;
    let th = bloom_h.max(1) as f32;

    let half_w = w * 0.5;
    let half_h = h * 0.5;
    let aspect_w = half_w / tw;
    let aspect_h = half_h / th;

    let one = [(1.0 / w) * aspect_w, (1.0 / h) * aspect_h];
    let half = [0.5 / w, 0.5 / h];

    let a = [
        [half[0], half[1]],
        [half[0] + one[0], half[1]],
        [half[0], half[1] + one[1]],
        [half[0] + one[0], half[1] + one[1]],
    ];
    let b = [
        [1.0 + a[0][0], 1.0 + a[0][1]],
        [1.0 + a[1][0], 1.0 + a[1][1]],
        [1.0 + a[2][0], 1.0 + a[2][1]],
        [1.0 + a[3][0], 1.0 + a[3][1]],
    ];
    (a, b)
}

/// Compute the 8 `[u_neg, v_neg, v_pos, u_pos]` tap base offsets for the horizontal 15-tap bloom pass
/// (`r3_rendertarget_phase_bloom.cpp:204-216`).
pub fn bloom_filter_taps_horizontal(bloom_w: u32, bloom_h: u32) -> [[f32; 4]; 8] {
    let w = bloom_w.max(1) as f32;
    let h = bloom_h.max(1) as f32;
    let two_x = 2.0 / w;
    let one_x = 1.0 / w;
    let half = [0.5 / w, 0.5 / h];

    let mut taps = [[0.0f32; 4]; 8];
    taps[0] = [half[0], half[1], half[1], half[0]];
    taps[1] = [
        taps[0][0] - one_x - half[0],
        half[1],
        half[1],
        taps[0][3] + one_x + half[0],
    ];
    for idx in 2..8 {
        taps[idx] = [
            taps[idx - 1][0] - two_x,
            half[1],
            half[1],
            taps[idx - 1][3] + two_x,
        ];
    }
    taps
}

/// Compute the 8 `[u_neg, v_neg, v_pos, u_pos]` tap base offsets for the vertical 15-tap bloom pass
/// (`r3_rendertarget_phase_bloom.cpp:284-296`).
pub fn bloom_filter_taps_vertical(bloom_w: u32, bloom_h: u32) -> [[f32; 4]; 8] {
    let w = bloom_w.max(1) as f32;
    let h = bloom_h.max(1) as f32;
    let two_y = 2.0 / h;
    let one_y = 1.0 / h;
    let half = [0.5 / w, 0.5 / h];

    let mut taps = [[0.0f32; 4]; 8];
    taps[0] = [half[0], half[1], half[1], half[0]];
    taps[1] = [
        half[0],
        taps[0][1] - one_y - half[1],
        half[1] + one_y + taps[0][2],
        half[0],
    ];
    for idx in 2..8 {
        taps[idx] = [
            half[0],
            taps[idx - 1][1] - two_y,
            two_y + taps[idx - 1][2],
            half[0],
        ];
    }
    taps
}

/// 32-byte SSFX bloom constant block (`ps_ssfx_bloom_1`, `ps_ssfx_bloom_2`, `xrRender_console.cpp:420-422`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxBloomGpuParams {
    /// `[threshold, exposure, use_presets_flag, sky_intensity]`.
    pub bloom_1: [f32; 4],
    /// `[blur_radius, vibrance, lens, dirt]`.
    pub bloom_2: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<SsfxBloomGpuParams>() == 32);

impl Default for SsfxBloomGpuParams {
    fn default() -> Self {
        Self {
            bloom_1: [3.5, 3.0, 0.0, 0.6],
            bloom_2: [3.0, 1.5, 1.5, 1.0],
        }
    }
}

impl SsfxBloomGpuParams {
    /// Derive SSFX bloom parameters from an interpolated `MixedEnvironment` frame (`bloom_*` & `hemi_vibrance`).
    pub fn from_mixed_environment(env: &MixedEnvironment, blur_radius: f32, lens: f32, dirt: f32) -> Self {
        Self {
            bloom_1: [
                env.bloom_threshold,
                env.bloom_exposure,
                0.0,
                env.bloom_sky_intensity,
            ],
            bloom_2: [blur_radius, env.hemi_vibrance, lens, dirt],
        }
    }
}

// ============================================================================
// Tonemapping, Image Correction & Vignette (`r3_rendertarget_phase_combine.cpp`)
// ============================================================================

/// Uncharted 2 filmic tonemapper console parameters (`ps_r2_tnmp_*`, `xrRender_console.cpp:292-301`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TonemapParams {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
    pub w: f32,
    pub exposure: f32,
    pub gamma: f32,
    pub enabled: bool,
}

impl Default for TonemapParams {
    fn default() -> Self {
        Self {
            a: 0.205,
            b: 0.35,
            c: 0.55,
            d: 0.20,
            e: 0.02,
            f: 0.15,
            w: 7.5,
            exposure: 7.0,
            gamma: 0.25,
            enabled: false,
        }
    }
}

/// 48-byte GPU constant layout mirroring `tnmp_a..tnmp_onoff` (`r3_rendertarget_phase_combine.cpp:533-543`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TonemapGpuParams {
    /// `[tnmp_a, tnmp_b, tnmp_c, tnmp_d]`.
    pub abcd: [f32; 4],
    /// `[tnmp_e, tnmp_f, tnmp_w, tnmp_exposure]`.
    pub efw_exposure: [f32; 4],
    /// `[tnmp_gamma, tnmp_onoff, 0.0, 0.0]`.
    pub gamma_onoff: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<TonemapGpuParams>() == 48);

/// Scalar Hable / Uncharted 2 rational shoulder-toe curve (`hdr10.h:389-392`).
#[inline]
pub fn uncharted2_tonemap_curve(x: f32, p: &TonemapParams) -> f32 {
    let f_div = p.f.max(1e-6);
    let num = x * (p.a * x + p.c * p.b) + p.d * p.e;
    let den = x * (p.a * x + p.b) + p.d * f_div;
    if den.abs() <= 1e-12 {
        0.0
    } else {
        (num / den) - (p.e / f_div)
    }
}

impl TonemapParams {
    #[inline]
    pub fn to_gpu_params(&self) -> TonemapGpuParams {
        TonemapGpuParams {
            abcd: [self.a, self.b, self.c, self.d],
            efw_exposure: [self.e, self.f, self.w, self.exposure],
            gamma_onoff: [self.gamma, if self.enabled { 1.0 } else { 0.0 }, 0.0, 0.0],
        }
    }

    /// Reference CPU evaluation matching Uncharted 2 white-scale normalization when enabled.
    pub fn evaluate_rgb(&self, color: [f32; 3]) -> [f32; 3] {
        if !self.enabled {
            return color;
        }
        let white_val = uncharted2_tonemap_curve(self.w.max(1e-4), self).max(1e-6);
        let inv_white = 1.0 / white_val;
        [
            (uncharted2_tonemap_curve((color[0] * self.exposure).max(0.0), self) * inv_white)
                .clamp(0.0, 1.0),
            (uncharted2_tonemap_curve((color[1] * self.exposure).max(0.0), self) * inv_white)
                .clamp(0.0, 1.0),
            (uncharted2_tonemap_curve((color[2] * self.exposure).max(0.0), self) * inv_white)
                .clamp(0.0, 1.0),
        ]
    }
}

/// LDR post-combine color & vignette settings (`ps_r2_img_*` and `ps_vignette_control`,
/// `xrRender_console.cpp:350-353, 381`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageCorrectionConfig {
    pub exposure: f32,
    pub gamma: f32,
    pub saturation: f32,
    pub color_grading: [f32; 3],
    pub vignette_control: [f32; 4],
}

impl Default for ImageCorrectionConfig {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            gamma: 1.0,
            saturation: 1.0,
            color_grading: [0.0; 3],
            vignette_control: [0.0; 4],
        }
    }
}

/// 48-byte GPU constant block mirroring `pp_img_corrections`, `pp_img_cg`, and `vignette_control`
/// (`Blender_Recorder_StandartBinding.cpp:718-732, 1362-1369`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ImageCorrectionGpuParams {
    /// `[ps_r2_img_exposure, ps_r2_img_gamma, ps_r2_img_saturation, 1.0]`.
    pub img_corrections: [f32; 4],
    /// `[ps_r2_img_cg.x, ps_r2_img_cg.y, ps_r2_img_cg.z, 1.0]`.
    pub img_cg: [f32; 4],
    /// `[corner_darken, center_darken, edge_expand, reserved]`.
    pub vignette_control: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<ImageCorrectionGpuParams>() == 48);

impl ImageCorrectionConfig {
    #[inline]
    pub fn to_gpu_params(&self) -> ImageCorrectionGpuParams {
        ImageCorrectionGpuParams {
            img_corrections: [self.exposure, self.gamma, self.saturation, 1.0],
            img_cg: [
                self.color_grading[0],
                self.color_grading[1],
                self.color_grading[2],
                1.0,
            ],
            vignette_control: self.vignette_control,
        }
    }
}

/// Apply radial screen vignette matching `combine_2_naa.ps:83-101`:
/// - `vignette_control[0]`: corner darkening `[0..1]`
/// - `vignette_control[1]`: center darkening `[0..1]`
/// - `vignette_control[2]`: inner radius contraction `[0..1]`
pub fn apply_vignette_rgb(
    color: [f32; 3],
    uv: [f32; 2],
    vignette_control: [f32; 4],
) -> [f32; 3] {
    let mid = [uv[0] - 0.5, uv[1] - 0.5];
    let dist_norm = (mid[0] * mid[0] + mid[1] * mid[1]).sqrt() * 2.0;
    let edge_radius = std::f32::consts::SQRT_2 * (1.0 - vignette_control[2].clamp(0.0, 1.0));
    let t = if dist_norm > edge_radius {
        1.0
    } else if edge_radius <= 0.0 {
        0.0
    } else {
        (dist_norm / edge_radius).clamp(0.0, 1.0)
    };
    let center_mul = 1.0 - vignette_control[1];
    let corner_mul = 1.0 - vignette_control[0];
    let factor = center_mul + (corner_mul - center_mul) * t;
    [color[0] * factor, color[1] * factor, color[2] * factor]
}

// ============================================================================
// Hardware Display Gamma Curve (`CGammaControl::GenLUT`, `xr_effgamma.cpp`)
// ============================================================================

/// Display gamma / brightness / contrast / channel balance (`CGammaControl`, `xr_effgamma.cpp:25-85`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaControlConfig {
    pub gamma: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub balance: [f32; 3],
}

impl Default for GammaControlConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            brightness: 1.0,
            contrast: 1.0,
            balance: [1.0, 1.0, 1.0],
        }
    }
}

impl GammaControlConfig {
    /// Generate DXGI-style floating-point RGB LUT curve (`CGammaControl::GenLUT`, `xr_effgamma.cpp:25-52`).
    pub fn gen_lut_f32(
        &self,
        control_point_positions: &[f32],
        min_cv: f32,
        max_cv: f32,
    ) -> Vec<[f32; 3]> {
        const EPS: f32 = 1e-5;
        let delta_cv = max_cv - min_cv;
        let og = 1.0 / (self.gamma + EPS);
        let b = self.brightness * 0.5;
        let c = self.contrast * 0.5;

        let mut out = Vec::with_capacity(control_point_positions.len());
        for &pos in control_point_positions {
            let p = pos.clamp(0.0, 1.0);
            let curved =
                (c + 0.5) * p.powf(og) + (b - 0.5) * 0.5 - c * 0.5 + 0.25;
            let cv = min_cv + curved * delta_cv;
            out.push([
                (cv * self.balance[0]).clamp(min_cv, max_cv),
                (cv * self.balance[1]).clamp(min_cv, max_cv),
                (cv * self.balance[2]).clamp(min_cv, max_cv),
            ]);
        }
        out
    }

    /// Generate classic 256-entry 16-bit gamma ramp (`CGammaControl::GenLUT(D3DGAMMARAMP&)`, `xr_effgamma.cpp:73-85`).
    pub fn gen_ramp_u16(&self) -> [[u16; 3]; 256] {
        const EPS: f32 = 1e-5;
        let og = 1.0 / (self.gamma + EPS);
        let b = self.brightness * 0.5;
        let c = self.contrast * 0.5;
        let mut ramp = [[0u16; 3]; 256];
        for (idx, entry) in ramp.iter_mut().enumerate() {
            let val = (c + 0.5) * (idx as f32 / 255.0).powf(og) * 65535.0
                + (b - 0.5) * 32768.0
                - c * 32768.0
                + 16384.0;
            for (slot, &bal) in entry.iter_mut().zip(self.balance.iter()) {
                let scaled = (val * bal).floor() as i32;
                *slot = scaled.clamp(0, 65535) as u16;
            }
        }
        ramp
    }
}

// ============================================================================
// HDR10 Pipeline Constants, Colorspaces & Sun Blending (`hdr10.h`, `effects_sun.ps`)
// ============================================================================

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hdr10Colorspace {
    Rec709 = 0,
    P3D65 = 1,
    #[default]
    Rec2020 = 2,
}

impl Hdr10Colorspace {
    #[inline]
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Rec709,
            1 => Self::P3D65,
            _ => Self::Rec2020,
        }
    }

    #[inline]
    #[allow(clippy::excessive_precision)]
    pub fn luminance(self, color: [f32; 3]) -> f32 {
        match self {
            Self::Rec709 => dot3(color, [0.2126390, 0.7151687, 0.0721923]),
            Self::P3D65 => dot3(color, [0.2289746, 0.6917385, 0.0792869]),
            Self::Rec2020 => dot3(color, [0.2627002, 0.6779981, 0.0593017]),
        }
    }

    /// Transform RGB tristimulus between supported HDR10 colorspaces using Bradford matrices (`hdr10.h:133-172`).
    #[allow(clippy::excessive_precision)]
    pub fn convert(self, dst: Self, color: [f32; 3]) -> [f32; 3] {
        match (self, dst) {
            (a, b) if a == b => color,
            (Self::Rec709, Self::P3D65) => mul_mat3_vec3(
                &[
                    [0.82246197, 0.17753803, -0.00000000],
                    [0.03319420, 0.96680580, 0.00000000],
                    [0.01708263, 0.07239744, 0.91051993],
                ],
                color,
            ),
            (Self::Rec709, Self::Rec2020) => mul_mat3_vec3(
                &[
                    [0.62740390, 0.32928304, 0.04331307],
                    [0.06909729, 0.91954040, 0.01136232],
                    [0.01639144, 0.08801331, 0.89559525],
                ],
                color,
            ),
            (Self::P3D65, Self::Rec709) => mul_mat3_vec3(
                &[
                    [1.22494018, -0.22494018, -0.00000000],
                    [-0.04205695, 1.04205695, 0.00000000],
                    [-0.01963755, -0.07863605, 1.09827360],
                ],
                color,
            ),
            (Self::P3D65, Self::Rec2020) => mul_mat3_vec3(
                &[
                    [0.75383303, 0.19859737, 0.04756960],
                    [0.04574385, 0.94177722, 0.01247893],
                    [-0.00121034, 0.01760172, 0.98360862],
                ],
                color,
            ),
            (Self::Rec2020, Self::Rec709) => mul_mat3_vec3(
                &[
                    [1.66049100, -0.58764114, -0.07284986],
                    [-0.12455047, 1.13289990, -0.00834942],
                    [-0.01815076, -0.10057890, 1.11872966],
                ],
                color,
            ),
            (Self::Rec2020, Self::P3D65) => mul_mat3_vec3(
                &[
                    [1.34357825, -0.28217967, -0.06139858],
                    [-0.06529745, 1.07578792, -0.01049046],
                    [0.00282179, -0.01959849, 1.01677671],
                ],
                color,
            ),
            _ => color,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hdr10Tonemapper {
    #[default]
    AcesNarkowicz = 0,
    AcesHill = 1,
    AgXNormal = 2,
    AgXPunchy = 3,
    Uchimura = 4,
    SteveM = 5,
    Uncharted2 = 6,
    ReinhardW20 = 7,
    ReinhardW30 = 8,
}

impl Hdr10Tonemapper {
    #[inline]
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::AcesNarkowicz,
            1 => Self::AcesHill,
            2 => Self::AgXNormal,
            3 => Self::AgXPunchy,
            4 => Self::Uchimura,
            5 => Self::SteveM,
            6 => Self::Uncharted2,
            7 => Self::ReinhardW20,
            8 => Self::ReinhardW30,
            _ => Self::AcesNarkowicz,
        }
    }

    /// Shader bitmask `1u << ps_r4_hdr10_tonemapper` (`Blender_Recorder_StandartBinding.cpp:1301`).
    #[inline]
    pub fn bitmask(self) -> u32 {
        1u32 << (self as u32)
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hdr10TonemapMode {
    Luminance = 0,
    #[default]
    Color = 1,
}

impl Hdr10TonemapMode {
    #[inline]
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Luminance,
            _ => Self::Color,
        }
    }
}

/// Apply ST.2084 Perceptual Quantizer EOTF^-1 encoding to linear normalized `[0..1]` RGB (`hdr10.h:194-205`).
pub fn apply_st2084_pq_rgb(color_norm: [f32; 3]) -> [f32; 3] {
    const M1: f32 = 2610.0 / 4096.0 / 4.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;

    let encode = |v: f32| -> f32 {
        let cp = v.max(0.0).powf(M1);
        ((C1 + C2 * cp) / (1.0 + C3 * cp)).powf(M2)
    };
    [
        encode(color_norm[0]),
        encode(color_norm[1]),
        encode(color_norm[2]),
    ]
}

/// Compute HDR sun disc weight in `[0.0, 1.0]` across dawn/day/dusk/night hours
/// matching `effects_sun.ps:33-49` (`TIME_HOURSF`).
pub fn effects_sun_hdr_blend_factor(
    hour_of_day: f32,
    dawn_begin: f32,
    dawn_end: f32,
    dusk_begin: f32,
    dusk_end: f32,
) -> f32 {
    if dawn_end <= hour_of_day && hour_of_day <= dusk_begin {
        1.0
    } else if dawn_begin < hour_of_day && hour_of_day < dawn_end {
        smoothstep(0.0, 1.0, (hour_of_day - dawn_begin) / (dawn_end - dawn_begin))
    } else if dusk_begin < hour_of_day && hour_of_day < dusk_end {
        1.0 - smoothstep(0.0, 1.0, (hour_of_day - dusk_begin) / (dusk_end - dusk_begin))
    } else {
        0.0
    }
}

/// Full HDR10 display / bloom / lens-flare / sun configuration matching
/// `xrRender_console.cpp:304-348` and `Blender_Recorder_StandartBinding.cpp:1291-1359`.
#[derive(Debug, Clone, PartialEq)]
pub struct Hdr10Config {
    pub enabled: bool,
    pub whitepoint_nits: f32,
    pub ui_nits: f32,
    pub rendering_pda: bool,
    pub colorspace: Hdr10Colorspace,
    pub pda_intensity: f32,
    pub tonemapper: Hdr10Tonemapper,
    pub tonemap_mode: Hdr10TonemapMode,
    pub exposure: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub contrast_middle_gray: f32,
    pub brightness: f32,
    pub gamma: f32,
    pub ui_saturation: f32,
    pub bloom_on: bool,
    pub bloom_blur_passes: u32,
    pub bloom_blur_scale: f32,
    pub bloom_intensity: f32,
    pub flare_on: bool,
    pub flare_threshold: f32,
    pub flare_power: f32,
    pub flare_ghosts: i32,
    pub flare_ghost_dispersal: f32,
    pub flare_center_falloff: f32,
    pub flare_halo_scale: f32,
    pub flare_halo_ca: f32,
    pub flare_ghost_ca: f32,
    pub flare_blur_passes: u32,
    pub flare_blur_scale: f32,
    pub flare_ghost_intensity: f32,
    pub flare_halo_intensity: f32,
    pub flare_lens_color: [f32; 3],
    pub sun_on: bool,
    pub sun_intensity: f32,
    pub sun_inner_radius: f32,
    pub sun_outer_radius: f32,
    pub sun_dawn_begin: f32,
    pub sun_dawn_end: f32,
    pub sun_dusk_begin: f32,
    pub sun_dusk_end: f32,
}

impl Default for Hdr10Config {
    fn default() -> Self {
        Self {
            enabled: false,
            whitepoint_nits: 400.0,
            ui_nits: 400.0,
            rendering_pda: false,
            colorspace: Hdr10Colorspace::Rec2020,
            pda_intensity: 1.0,
            tonemapper: Hdr10Tonemapper::AcesNarkowicz,
            tonemap_mode: Hdr10TonemapMode::Color,
            exposure: 0.8,
            contrast: 0.0,
            saturation: 0.1,
            contrast_middle_gray: 0.5,
            brightness: 0.0,
            gamma: 1.1,
            ui_saturation: 0.5,
            bloom_on: false,
            bloom_blur_passes: 20,
            bloom_blur_scale: 1.0,
            bloom_intensity: 0.06,
            flare_on: false,
            flare_threshold: 0.0,
            flare_power: 0.04,
            flare_ghosts: 1,
            flare_ghost_dispersal: 0.6,
            flare_center_falloff: 1.1,
            flare_halo_scale: 0.47,
            flare_halo_ca: 10.0,
            flare_ghost_ca: 3.0,
            flare_blur_passes: 12,
            flare_blur_scale: 1.0,
            flare_ghost_intensity: 0.04,
            flare_halo_intensity: 0.04,
            flare_lens_color: [1.0, 0.7, 1.0],
            sun_on: false,
            sun_intensity: 80.0,
            sun_inner_radius: 0.20,
            sun_outer_radius: 0.40,
            sun_dawn_begin: 4.5,
            sun_dawn_end: 6.0,
            sun_dusk_begin: 18.5,
            sun_dusk_end: 21.0,
        }
    }
}

/// 160-byte GPU constant buffer matching `hdr10_parameters1..10` (`hdr10.h:48-105`,
/// `Blender_Recorder_StandartBinding.cpp:1291-1359`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Hdr10GpuParams {
    pub parameters1: [f32; 4],
    pub parameters2: [f32; 4],
    pub parameters3: [f32; 4],
    pub parameters4: [f32; 4],
    pub parameters5: [f32; 4],
    pub parameters6: [f32; 4],
    pub parameters7: [f32; 4],
    pub parameters8: [f32; 4],
    pub parameters9: [f32; 4],
    pub parameters10: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Hdr10GpuParams>() == 160);

impl Hdr10Config {
    pub fn to_gpu_params(&self) -> Hdr10GpuParams {
        let wp = self.whitepoint_nits.max(1e-3);
        let gamma_rcp = 1.0 / self.gamma.max(1e-3);
        Hdr10GpuParams {
            parameters1: [
                self.whitepoint_nits,
                self.ui_nits / wp,
                if self.enabled { 1.0 } else { 0.0 },
                if self.rendering_pda { 1.0 } else { 0.0 },
            ],
            parameters2: [
                (self.colorspace as u32) as f32,
                self.pda_intensity,
                self.tonemapper.bitmask() as f32,
                (self.tonemap_mode as u32) as f32,
            ],
            parameters3: [
                self.exposure,
                self.contrast + 1.0,
                self.saturation + 1.0,
                self.contrast_middle_gray,
            ],
            parameters4: [
                if self.bloom_on { 1.0 } else { 0.0 },
                self.bloom_blur_scale,
                self.bloom_intensity,
                self.sun_intensity,
            ],
            parameters5: [
                self.sun_dawn_begin,
                self.sun_dawn_end,
                self.sun_dusk_begin,
                self.sun_dusk_end,
            ],
            parameters6: [
                self.brightness,
                gamma_rcp,
                self.flare_threshold,
                self.flare_power,
            ],
            parameters7: [
                self.flare_ghosts as f32,
                self.flare_ghost_dispersal,
                self.flare_center_falloff,
                self.flare_halo_scale,
            ],
            parameters8: [
                self.flare_halo_ca,
                self.flare_ghost_ca,
                self.flare_blur_scale,
                self.ui_saturation + 1.0,
            ],
            parameters9: [
                self.flare_ghost_intensity,
                self.flare_halo_intensity,
                self.sun_inner_radius,
                self.sun_outer_radius,
            ],
            parameters10: [
                self.flare_lens_color[0],
                self.flare_lens_color[1],
                self.flare_lens_color[2],
                if self.sun_on { 1.0 } else { 0.0 },
            ],
        }
    }

    /// Generate the half-resolution Kawase downsample + ping-pong blur pass schedule matching
    /// `CRenderTarget::phase_hdr10_bloom` (`r4_rendertarget_phase_hdr10_bloom.cpp:27-76`).
    pub fn bloom_blur_schedule(&self, screen_w: u32, screen_h: u32) -> Vec<Hdr10PingPongPass> {
        build_hdr10_kawase_schedule(
            screen_w,
            screen_h,
            self.bloom_blur_passes,
            self.bloom_blur_scale,
            false,
        )
    }

    /// Generate the half-resolution Karis downsample, feature-gen & Kawase blur pass schedule
    /// matching `CRenderTarget::phase_hdr10_lens_flare` (`r4_rendertarget_phase_hdr10_lens_flare.cpp:27-92`).
    pub fn flare_blur_schedule(&self, screen_w: u32, screen_h: u32) -> Vec<Hdr10PingPongPass> {
        build_hdr10_kawase_schedule(
            screen_w,
            screen_h,
            self.flare_blur_passes,
            self.flare_blur_scale,
            true,
        )
    }
}

/// Kind of pass inside the half-res HDR10 bloom / lens-flare Kawase ping-pong chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hdr10PassKind {
    Downsample,
    FeatureGen,
    BlurHorizontal,
    BlurVertical,
}

/// Single render pass descriptor emitted by `Hdr10Config::{bloom_blur_schedule, flare_blur_schedule}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hdr10PingPongPass {
    pub kind: Hdr10PassKind,
    /// `None` means sampling full-res scene color; `Some(0|1)` samples `rt_HDR10_HalfRes[slot]`.
    pub src_halfres_slot: Option<u8>,
    pub dst_halfres_slot: u8,
    pub viewport_w: u32,
    pub viewport_h: u32,
    /// Constant value for `hdr10_bloom_sparams` / `hdr10_flare_sparams` (`[dx, dy, is_horz, 0.0]`).
    pub sparams: [f32; 4],
}

fn build_hdr10_kawase_schedule(
    screen_w: u32,
    screen_h: u32,
    blur_passes: u32,
    blur_scale: f32,
    with_feature_gen: bool,
) -> Vec<Hdr10PingPongPass> {
    let orig_w = screen_w.max(1) as f32;
    let orig_h = screen_h.max(1) as f32;
    let half_w = (screen_w / 2).max(1);
    let half_h = (screen_h / 2).max(1);
    let half_wf = half_w as f32;
    let half_hf = half_h as f32;

    let mut passes = Vec::with_capacity(
        1 + usize::from(with_feature_gen) + (blur_passes as usize) * 2,
    );
    let mut src: u8 = 0;
    let mut dst: u8 = 1;

    passes.push(Hdr10PingPongPass {
        kind: Hdr10PassKind::Downsample,
        src_halfres_slot: None,
        dst_halfres_slot: dst,
        viewport_w: half_w,
        viewport_h: half_h,
        sparams: [1.0 / orig_w, 1.0 / orig_h, 0.0, 0.0],
    });

    if with_feature_gen {
        std::mem::swap(&mut src, &mut dst);
        passes.push(Hdr10PingPongPass {
            kind: Hdr10PassKind::FeatureGen,
            src_halfres_slot: Some(src),
            dst_halfres_slot: dst,
            viewport_w: half_w,
            viewport_h: half_h,
            sparams: [1.0 / orig_w, 1.0 / orig_h, 0.0, 0.0],
        });
    }

    for pass in 0..blur_passes {
        let radius = blur_scale * (pass + 1) as f32;
        let dx = radius / half_wf;
        let dy = radius / half_hf;
        for dir in 0..2u32 {
            std::mem::swap(&mut src, &mut dst);
            let is_horz = (dir % 2) == 1;
            passes.push(Hdr10PingPongPass {
                kind: if is_horz {
                    Hdr10PassKind::BlurHorizontal
                } else {
                    Hdr10PassKind::BlurVertical
                },
                src_halfres_slot: Some(src),
                dst_halfres_slot: dst,
                viewport_w: half_w,
                viewport_h: half_h,
                sparams: [dx, dy, if is_horz { 1.0 } else { 0.0 }, 0.0],
            });
        }
    }

    passes
}

// ============================================================================
// Depth of Field, Gasmask Visor, Night/Heat Vision & SSFX Screen-Space Params
// ============================================================================

/// Depth of Field configuration (`ps_r2_dof*` in `xrRender_console.cpp:492-494`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DofConfig {
    pub near: f32,
    pub focus: f32,
    pub far: f32,
    pub sky_distance: f32,
    pub kernel_size: f32,
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            near: -1.25,
            focus: 0.0,
            far: 600.0,
            sky_distance: 30.0,
            kernel_size: 0.25,
        }
    }
}

/// 32-byte GPU constant layout for `dof_params` + `dof_kernel` (`rendertarget_phase_dof.cpp:50-51`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DofGpuParams {
    /// `[near, focus, far, ps_r2_dof_sky]`.
    pub dof_params: [f32; 4],
    /// `[0.5 / screen_w * kernel_size, 0.5 / screen_h * kernel_size, kernel_size, 0.0]`.
    pub dof_kernel: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<DofGpuParams>() == 32);

impl DofConfig {
    #[inline]
    pub fn to_gpu_params(&self, screen_w: u32, screen_h: u32) -> DofGpuParams {
        let kx = (0.5 / screen_w.max(1) as f32) * self.kernel_size;
        let ky = (0.5 / screen_h.max(1) as f32) * self.kernel_size;
        DofGpuParams {
            dof_params: [self.near, self.focus, self.far, self.sky_distance],
            dof_kernel: [kx, ky, self.kernel_size, 0.0],
        }
    }
}

/// Visor gasmask & raindrop overlay settings (`ps_r2_mask_control`, `ps_r2_drops_control`,
/// `xrRender_console.cpp:257-258`, `rendertarget_phase_gasmask_drops.cpp:47-48`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasmaskConfig {
    pub mask_control: [f32; 4],
    pub drops_control: [f32; 3],
}

impl Default for GasmaskConfig {
    fn default() -> Self {
        Self {
            mask_control: [0.0; 4],
            drops_control: [0.0, 1.15, 0.0],
        }
    }
}

/// 32-byte GPU constant layout for `mask_control` + `drops_control`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GasmaskGpuParams {
    pub mask_control: [f32; 4],
    pub drops_control: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GasmaskGpuParams>() == 32);

impl GasmaskConfig {
    #[inline]
    pub fn is_mask_active(&self) -> bool {
        self.mask_control[0] > 0.0
    }

    #[inline]
    pub fn are_drops_active(&self) -> bool {
        self.is_mask_active() && self.drops_control[0] > 0.0
    }

    #[inline]
    pub fn to_gpu_params(&self) -> GasmaskGpuParams {
        GasmaskGpuParams {
            mask_control: self.mask_control,
            drops_control: [
                self.drops_control[0],
                self.drops_control[1],
                self.drops_control[2],
                0.0,
            ],
        }
    }
}

/// Beef's shader-based night vision generations (`r__nightvision` & `night_vision.h:27-137`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NightVisionGen {
    #[default]
    Off = 0,
    Gen1 = 1,
    Gen2 = 2,
    Gen3 = 3,
    Gen1Flipped = 10,
    Gen1Centered = 11,
}

impl NightVisionGen {
    #[inline]
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Gen1,
            2 => Self::Gen2,
            3 => Self::Gen3,
            10 => Self::Gen1Flipped,
            11 => Self::Gen1Centered,
            _ => Self::Off,
        }
    }

    #[inline]
    pub fn profile(self) -> Option<NightVisionProfile> {
        match self {
            Self::Off => None,
            Self::Gen1 | Self::Gen1Flipped | Self::Gen1Centered => Some(NightVisionProfile {
                min_blur: 0.25,
                max_blur: 0.95,
                min_vision_distance: 30.0,
                max_vision_distance: 250.0,
                dim_threshold: 0.7,
                bloom_threshold: 0.11,
                bloom_multiplier: 2.5,
                crt_effect_factor: 0.05,
                noise_factor: 0.15,
                scintillation_constant: 0.999,
                saturation_color: [0.4, 1.0, 0.1],
                vignette_amount: 0.10,
                circle_radius: 0.5,
            }),
            Self::Gen2 => Some(NightVisionProfile {
                min_blur: 0.20,
                max_blur: 0.80,
                min_vision_distance: 30.0,
                max_vision_distance: 400.0,
                dim_threshold: 0.7,
                bloom_threshold: 0.18,
                bloom_multiplier: 2.0,
                crt_effect_factor: 0.20,
                noise_factor: 0.15,
                scintillation_constant: 0.9993,
                saturation_color: [0.3, 1.0, 0.3],
                vignette_amount: 0.08,
                circle_radius: 0.5,
            }),
            Self::Gen3 => Some(NightVisionProfile {
                min_blur: 0.10,
                max_blur: 0.60,
                min_vision_distance: 30.0,
                max_vision_distance: 600.0,
                dim_threshold: 0.7,
                bloom_threshold: 0.29,
                bloom_multiplier: 2.0,
                crt_effect_factor: 0.40,
                noise_factor: 0.15,
                scintillation_constant: 0.9995,
                saturation_color: [0.1, 1.0, 0.8],
                vignette_amount: 0.05,
                circle_radius: 0.5,
            }),
        }
    }

    /// Evaluate Beef's NVG circular tube lens mask in `[0.0, 1.0]` with aspect-ratio correction
    /// (`compute_lens_mask` & `aspect_ratio_correction` in `night_vision.h:97-148`).
    pub fn lens_mask(self, uv: [f32; 2], screen_w: u32, screen_h: u32) -> f32 {
        let aspect = screen_w.max(1) as f32 / screen_h.max(1) as f32;
        let corrected = [(uv[0] - 0.5) * aspect + 0.5, uv[1]];
        let in_circle = |center: [f32; 2]| -> bool {
            let dx = corrected[0] - center[0];
            let dy = corrected[1] - center[1];
            (dx * dx + dy * dy).sqrt() <= 0.5
        };
        let inside = match self {
            Self::Off => false,
            Self::Gen1 => in_circle([0.75, 0.5]),
            Self::Gen1Flipped => in_circle([0.25, 0.5]),
            Self::Gen1Centered => in_circle([0.5, 0.5]),
            Self::Gen2 => in_circle([0.25, 0.5]) || in_circle([0.75, 0.5]),
            Self::Gen3 => {
                in_circle([0.05, 0.5])
                    || in_circle([0.30, 0.5])
                    || in_circle([0.70, 0.5])
                    || in_circle([0.95, 0.5])
            }
        };
        if inside {
            1.0
        } else {
            0.0
        }
    }
}

/// Per-generation physical optic/intensifier constants (`night_vision.h:36-92`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NightVisionProfile {
    pub min_blur: f32,
    pub max_blur: f32,
    pub min_vision_distance: f32,
    pub max_vision_distance: f32,
    pub dim_threshold: f32,
    pub bloom_threshold: f32,
    pub bloom_multiplier: f32,
    pub crt_effect_factor: f32,
    pub noise_factor: f32,
    pub scintillation_constant: f32,
    pub saturation_color: [f32; 3],
    pub vignette_amount: f32,
    pub circle_radius: f32,
}

/// Thermal / HeatVision shader settings (`xrRender_console.cpp:274-282`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeatVisionConfig {
    pub enabled: bool,
    pub mode: f32,
    pub cooldown_enabled: bool,
    pub cooldown_time_ms: f32,
    pub zombie_cold: bool,
    pub steps: [f32; 4],
    pub blurring: [f32; 4],
    pub args_1: [f32; 4],
    pub args_2: [f32; 4],
}

impl Default for HeatVisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: 0.0,
            cooldown_enabled: true,
            cooldown_time_ms: 20_000.0,
            zombie_cold: false,
            steps: [0.45, 0.65, 0.76, 0.0],
            blurring: [15.0, 4.0, 60.0, 0.0],
            args_1: [0.0; 4],
            args_2: [0.0; 4],
        }
    }
}

/// 64-byte GPU constant layout for `heatvision_params1/2` and `args1/2`
/// (`Blender_Recorder_StandartBinding.cpp:473-498`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HeatVisionGpuParams {
    pub params1: [f32; 4],
    pub params2: [f32; 4],
    pub args_1: [f32; 4],
    pub args_2: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<HeatVisionGpuParams>() == 64);

impl HeatVisionConfig {
    #[inline]
    pub fn to_gpu_params(&self) -> HeatVisionGpuParams {
        HeatVisionGpuParams {
            params1: [
                if self.enabled { 1.0 } else { 0.0 },
                self.steps[0],
                self.steps[1],
                self.steps[2],
            ],
            params2: [
                self.blurring[0],
                self.blurring[1],
                self.blurring[2],
                self.mode,
            ],
            args_1: self.args_1,
            args_2: self.args_2,
        }
    }
}

/// Screen Space Shaders TAA jitter & sharpness settings (`ps_ssfx_taa`,
/// `xrRender_console.cpp:408`, `Blender_Recorder_StandartBinding.cpp:1173-1193`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxTaaConfig {
    pub enabled: f32,
    pub jitter_amount: f32,
    pub sharpness: f32,
    pub extra: f32,
}

impl Default for SsfxTaaConfig {
    fn default() -> Self {
        Self {
            enabled: 1.0,
            jitter_amount: 0.5,
            sharpness: 0.6,
            extra: 0.0,
        }
    }
}

impl SsfxTaaConfig {
    /// 4-sample cross pattern (`TAA_Offset[4]`, `Blender_Recorder_StandartBinding.cpp:1179-1185`).
    pub const TAA_OFFSETS: [[f32; 2]; 4] = [[0.0, -1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

    /// Compute `ssfx_jitter` uniform `[jitter_x * amount, jitter_y * amount, enabled, extra]`.
    #[inline]
    pub fn jitter_for_frame(
        &self,
        frame_index: u32,
        screen_w: u32,
        screen_h: u32,
        taa_feature_enabled: bool,
    ) -> [f32; 4] {
        if self.enabled > 0.0 && taa_feature_enabled {
            let offset = Self::TAA_OFFSETS[(frame_index % 4) as usize];
            let jx = (offset[0] / screen_w.max(1) as f32) * self.jitter_amount;
            let jy = (offset[1] / screen_h.max(1) as f32) * self.jitter_amount;
            [jx, jy, self.enabled, self.extra]
        } else {
            [0.0, 0.0, self.enabled, self.extra]
        }
    }

    #[inline]
    pub fn to_taa_setup(&self) -> [f32; 4] {
        [self.enabled, self.jitter_amount, self.sharpness, self.extra]
    }
}

/// SSFX fog, volumetric shadow & cascade settings (`xrRender_console.cpp:407-410, 449-454, 481-489`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxScreenConfig {
    pub fog: [f32; 4],
    pub fog_scattering: f32,
    pub motionblur: [f32; 4],
    pub lut: [f32; 4],
    pub shadows: [f32; 3],
    pub shadow_cascades: [f32; 3],
    pub shadow_bias: [f32; 3],
    pub grass_shadows: [f32; 4],
    pub volumetric: [f32; 4],
    pub wpn_dof_1: [f32; 4],
    pub wpn_dof_2: f32,
}

impl Default for SsfxScreenConfig {
    fn default() -> Self {
        Self {
            fog: [8.0, 1.3, 0.1, 0.0],
            fog_scattering: 0.6,
            motionblur: [6.0, 0.0, 0.0, 0.0],
            lut: [0.0; 4],
            shadows: [256.0, 1536.0, 0.0],
            shadow_cascades: [20.0, 40.0, 160.0],
            shadow_bias: [0.4, 0.03, 0.0],
            grass_shadows: [0.0, 0.35, 30.0, 0.0],
            volumetric: [1.0, 1.0, 3.0, 1.0],
            wpn_dof_1: [0.0; 4],
            wpn_dof_2: 1.0,
        }
    }
}

/// SSFX AO/IL/SSR/Water/Rain/SSS settings matching `xrRender_console.cpp:424-479`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxLightingConfig {
    pub sss_quality: [f32; 4],
    pub sss: [f32; 4],
    pub ao_quality: i32,
    pub ao: [f32; 4],
    pub ao_setup1: [f32; 4],
    pub il_quality: i32,
    pub il: [f32; 4],
    pub il_setup1: [f32; 4],
    pub ssr_quality: i32,
    pub ssr: [f32; 4],
    pub ssr_2: [f32; 4],
    pub water: [f32; 4],
    pub water_quality: [f32; 3],
    pub water_setup1: [f32; 4],
    pub water_setup2: [f32; 4],
    pub rain_1: [f32; 4],
    pub rain_2: [f32; 4],
    pub rain_3: [f32; 4],
    pub rain_drops_setup: [f32; 4],
}

impl Default for SsfxLightingConfig {
    fn default() -> Self {
        Self {
            sss_quality: [12.0, 4.0, 1.0, 1.0],
            sss: [1.0, 1.0, 1.0, 0.0],
            ao_quality: 4,
            ao: [1.0, 5.0, 1.0, 2.5],
            ao_setup1: [150.0, 1.0, 1.0, 0.0],
            il_quality: 32,
            il: [6.66, 1.0, 1.0, 5.0],
            il_setup1: [150.0, 1.0, 0.5, 0.0],
            ssr_quality: 0,
            ssr: [1.0, 0.2, 0.0, 0.0],
            ssr_2: [0.0, 1.3, 1.0, 0.015],
            water: [1.0, 0.8, 1.0, 0.0],
            water_quality: [1.0, 2.0, 0.0],
            water_setup1: [0.6, 3.0, 0.3, 0.05],
            water_setup2: [0.8, 6.0, 0.3, 0.5],
            rain_1: [2.0, 0.1, 0.6, 2.0],
            rain_2: [0.5, 0.1, 1.0, 0.5],
            rain_3: [0.5, 1.0, 0.0, 0.0],
            rain_drops_setup: [2500.0, 15.0, 0.0, 0.0],
        }
    }
}

/// Number of persistent point/spot light slots tracked by `phase_ssfx_sss_ext`.
pub const SSFX_SSS_EXT_MAX_SLOTS: usize = 8;

/// Input candidate light for `SsfxSssExtSlots` matching `light_Package::v_shadowed` entries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxSssCandidate {
    pub id: u64,
    pub position: [f32; 3],
    pub range: f32,
    /// Pre-sorted package distance from camera (`light::distance`).
    pub distance: f32,
    pub active: bool,
    pub is_secondary_omnipart: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TrackedSssLight {
    id: u64,
    position: [f32; 3],
    range: f32,
    distance: f32,
    active: bool,
}

/// Persistent 8-slot external light selector & view-space fadeout packer matching
/// `CRenderTarget::phase_ssfx_sss_ext` (`rendertarget_phase_blur.cpp:631-840`).
#[derive(Debug, Clone, Default)]
pub struct SsfxSssExtSlots {
    slots: [Option<TrackedSssLight>; SSFX_SSS_EXT_MAX_SLOTS],
    initialized: bool,
    next_refresh_frame: u32,
}

impl SsfxSssExtSlots {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear a light immediately when destroyed (`OnLightDestroy`).
    pub fn remove_light(&mut self, id: u64) {
        for slot in &mut self.slots {
            if slot.is_some_and(|l| l.id == id) {
                *slot = None;
            }
        }
    }

    /// Return current slot assignment IDs (index `0..8`) for debugging/SSS pass offset correlation.
    pub fn assigned_ids(&self) -> [Option<u64>; SSFX_SSS_EXT_MAX_SLOTS] {
        self.slots.map(|entry| entry.map(|l| l.id))
    }

    /// Step the persistent 8-slot tracker and return `lights_data[8]` (`[vx, vy, vz, fade]`),
    /// where `fade = 0.0` is full visibility and `1.0` is fully faded out.
    pub fn update(
        &mut self,
        frame: u32,
        camera_pos: [f32; 3],
        view_matrix: &XRayMatrix,
        shadowed_lights: &[SsfxSssCandidate],
    ) -> [[f32; 4]; SSFX_SSS_EXT_MAX_SLOTS] {
        let mut out = [[0.0f32; 4]; SSFX_SSS_EXT_MAX_SLOTS];
        let mut sorted: Vec<(u8, f32, SsfxSssCandidate)> = Vec::new();
        let check_package = !self.initialized || frame > self.next_refresh_frame;

        if check_package {
            self.initialized = true;
            self.next_refresh_frame = frame.saturating_add(2);

            for &cand in shadowed_lights {
                if !cand.is_secondary_omnipart
                    && cand.range > 1.5
                    && cand.distance < 800.0
                    && cand.active
                {
                    let dx = camera_pos[0] - cand.position[0];
                    let dy = camera_pos[1] - cand.position[1];
                    let dz = camera_pos[2] - cand.position[2];
                    let dist_lpos = (dx * dx + dy * dy + dz * dz).sqrt();
                    let priority = if dist_lpos <= cand.range { 0u8 } else { 1u8 };
                    sorted.push((priority, dist_lpos, cand));
                }
            }

            // Total ordering consistent with `SortLights`: inside-range lights first, then nearest `distance`.
            sorted.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.distance.total_cmp(&b.2.distance)));

            for &(_, _, cand) in &sorted {
                let mut add = true;
                let mut free_slot: Option<usize> = None;
                for (slot_idx, slot) in self.slots.iter().enumerate() {
                    if let Some(existing) = slot {
                        if existing.id == cand.id {
                            add = false;
                            break;
                        }
                    } else {
                        free_slot = Some(slot_idx);
                    }
                }
                if add && let Some(idx) = free_slot {
                    self.slots[idx] = Some(TrackedSssLight {
                        id: cand.id,
                        position: cand.position,
                        range: cand.range,
                        distance: cand.distance,
                        active: cand.active,
                    });
                }
            }
        }

        for (slot_idx, slot) in self.slots.iter_mut().enumerate() {
            let Some(mut tracked) = *slot else {
                continue;
            };

            let mut remove = true;
            let dist_lpos = if check_package {
                if let Some(&(_, d, cand)) = sorted.iter().find(|entry| entry.2.id == tracked.id) {
                    tracked.position = cand.position;
                    tracked.range = cand.range;
                    tracked.distance = cand.distance;
                    tracked.active = cand.active;
                    *slot = Some(tracked);
                    remove = false;
                    d
                } else {
                    f32::MAX
                }
            } else {
                if let Some(cand) = shadowed_lights.iter().find(|c| c.id == tracked.id) {
                    tracked.position = cand.position;
                    tracked.range = cand.range;
                    tracked.distance = cand.distance;
                    tracked.active = cand.active;
                    *slot = Some(tracked);
                }
                let dx = camera_pos[0] - tracked.position[0];
                let dy = camera_pos[1] - tracked.position[1];
                let dz = camera_pos[2] - tracked.position[2];
                remove = false;
                (dx * dx + dy * dy + dz * dz).sqrt()
            };

            if dist_lpos > tracked.range * 2.0 {
                remove = true;
            }
            if !tracked.active || remove {
                *slot = None;
                continue;
            }

            let vpos = matrix_transform_point(view_matrix, tracked.position);
            let max_atte = 1.0 - ((tracked.distance - 780.0) / -100.0).clamp(0.0, 1.0);
            let atte = 1.0
                - ((dist_lpos - tracked.range * 1.9) / -(tracked.range * 0.5)).clamp(0.0, 1.0);
            out[slot_idx] = [vpos[0], vpos[1], vpos[2], max_atte.max(atte)];
        }

        out
    }
}

/// Single render pass descriptor emitted for SSFX blur / AO / IL / volumetric / bloom schedules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxBlurPassDesc {
    pub viewport_w: f32,
    pub viewport_h: f32,
    pub element_index: u8,
    pub dst_slot: u8,
    pub setup: [f32; 4],
}

/// Build the 4-pass bilateral blur schedule for `phase_ssfx_ao` (`r4_rendertarget_phase_ssao.cpp:177-302`).
pub fn ssfx_ao_blur_schedule(
    screen_w: u32,
    screen_h: u32,
    scale_factor: f32,
) -> [SsfxBlurPassDesc; 4] {
    let w = screen_w.max(1) as f32;
    let h = screen_h.max(1) as f32;
    let s = scale_factor.clamp(1.0, 8.0);
    let sx = w / s;
    let sy = h / s;
    [
        SsfxBlurPassDesc {
            viewport_w: w,
            viewport_h: h,
            element_index: 1,
            dst_slot: 1,
            setup: [s, 0.25, sx, sy],
        },
        SsfxBlurPassDesc {
            viewport_w: w,
            viewport_h: h,
            element_index: 2,
            dst_slot: 0,
            setup: [1.0, 0.5, sx, sy],
        },
        SsfxBlurPassDesc {
            viewport_w: w,
            viewport_h: h,
            element_index: 1,
            dst_slot: 1,
            setup: [1.0, 0.75, sx, sy],
        },
        SsfxBlurPassDesc {
            viewport_w: w,
            viewport_h: h,
            element_index: 2,
            dst_slot: 0,
            setup: [1.0, 1.0, sx, sy],
        },
    ]
}

/// Build the 4-pass blur schedule for `phase_ssfx_il` (`r4_rendertarget_phase_ssao.cpp:319-442`).
pub fn ssfx_il_blur_schedule(
    screen_w: u32,
    screen_h: u32,
    scale_factor: f32,
) -> [SsfxBlurPassDesc; 4] {
    let w = screen_w.max(1) as f32;
    let h = screen_h.max(1) as f32;
    let s = scale_factor.clamp(1.0, 8.0);
    let sx = w / s;
    let sy = h / s;
    [
        SsfxBlurPassDesc {
            viewport_w: sx,
            viewport_h: sy,
            element_index: 4,
            dst_slot: 1,
            setup: [1.0, 0.25, sx, sy],
        },
        SsfxBlurPassDesc {
            viewport_w: sx,
            viewport_h: sy,
            element_index: 5,
            dst_slot: 0,
            setup: [1.0, 0.5, sx, sy],
        },
        SsfxBlurPassDesc {
            viewport_w: sx,
            viewport_h: sy,
            element_index: 4,
            dst_slot: 1,
            setup: [1.0, 0.75, sx, sy],
        },
        SsfxBlurPassDesc {
            viewport_w: sx,
            viewport_h: sy,
            element_index: 5,
            dst_slot: 0,
            setup: [1.0, 1.0, sx, sy],
        },
    ]
}

/// Build the 4-pass 1/8-res `phase_ssfx_volumetric_blur` schedule (`rendertarget_phase_blur.cpp:365-391`).
pub fn ssfx_volumetric_blur_schedule(screen_w: u32, screen_h: u32) -> [SsfxBlurPassDesc; 4] {
    let vw = screen_w.max(1) as f32 / 8.0;
    let vh = screen_h.max(1) as f32 / 8.0;
    let pixelsize = [0.0, 1.0, 1.0, 2.0];
    let pixelscale = [2.0, 0.5, 2.0, 0.5];
    let mut out = [SsfxBlurPassDesc {
        viewport_w: vw,
        viewport_h: vh,
        element_index: 0,
        dst_slot: 0,
        setup: [0.0; 4],
    }; 4];
    for b in 0..4 {
        out[b] = SsfxBlurPassDesc {
            viewport_w: vw,
            viewport_h: vh,
            element_index: (b % 2) as u8,
            dst_slot: (b % 2) as u8,
            setup: [vw, vh, pixelsize[b], pixelscale[b]],
        };
    }
    out
}

/// Build the 6-pass multi-mip `phase_blur` schedule (`1/2`, `1/4`, `1/8` H+V, `rendertarget_phase_blur.cpp:33-184`).
pub fn phase_blur_schedule(screen_w: u32, screen_h: u32) -> [SsfxBlurPassDesc; 6] {
    let w0 = screen_w.max(1) as f32;
    let h0 = screen_h.max(1) as f32;
    let scales = [0.5f32, 0.25, 0.125];
    let mut out = [SsfxBlurPassDesc {
        viewport_w: w0 * 0.5,
        viewport_h: h0 * 0.5,
        element_index: 0,
        dst_slot: 0,
        setup: [0.0; 4],
    }; 6];
    for (level, &scale) in scales.iter().enumerate() {
        let vw = w0 * scale;
        let vh = h0 * scale;
        out[level * 2] = SsfxBlurPassDesc {
            viewport_w: vw,
            viewport_h: vh,
            element_index: (level * 2) as u8,
            dst_slot: 0,
            setup: [1.0, 0.0, vw, vh],
        };
        out[level * 2 + 1] = SsfxBlurPassDesc {
            viewport_w: vw,
            viewport_h: vh,
            element_index: (level * 2 + 1) as u8,
            dst_slot: 1,
            setup: [0.0, 1.0, vw, vh],
        };
    }
    out
}

/// SSFX Bloom lens 2-pass quarter-res blur schedule (`r4_rendertarget_phase_bloom.cpp:436-458`).
pub fn ssfx_bloom_lens_blur_schedule(screen_w: u32, screen_h: u32) -> [SsfxBlurPassDesc; 2] {
    let qw = screen_w.max(1) as f32 / 4.0;
    let qh = screen_h.max(1) as f32 / 4.0;
    [
        SsfxBlurPassDesc {
            viewport_w: qw,
            viewport_h: qh,
            element_index: 1,
            dst_slot: 0,
            setup: [qw, qh, 0.0, 2.0],
        },
        SsfxBlurPassDesc {
            viewport_w: qw,
            viewport_h: qh,
            element_index: 2,
            dst_slot: 1,
            setup: [qw, qh, 0.0, 5.0],
        },
    ]
}

// ============================================================================
// Luminance Adaptation (`r4_rendertarget_phase_luminance.cpp`, `r2_types.h`)
// ============================================================================

/// Bloom & initial luminance downsample source texture dimensions (`BLOOM_size_X/Y`, `r2_types.h:141-142`).
pub const BLOOM_SIZE_X: u32 = 256;
pub const BLOOM_SIZE_Y: u32 = 256;

/// Eye-adaptation and middle-gray tonemap configuration (`xrRender_console.cpp:188, 206-209`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LuminanceConfig {
    pub adaptation: f32,
    pub middle_gray: f32,
    pub low_lum: f32,
    pub amount: f32,
    pub tonemap_enabled: bool,
}

impl Default for LuminanceConfig {
    fn default() -> Self {
        Self {
            adaptation: 1.0,
            middle_gray: 1.0,
            low_lum: 0.4,
            amount: 0.7,
            tonemap_enabled: true,
        }
    }
}

/// Build pass-0 (`256x256 -> 64x64`) 4-tap log-luminance UV bounds `(a[0..4], b[0..4])`
/// matching `CRenderTarget::phase_luminance` (`r4_rendertarget_phase_luminance.cpp:38-49`).
pub fn luminance_pass0_uvs(bloom_w: u32, bloom_h: u32) -> ([[f32; 2]; 4], [[f32; 2]; 4]) {
    let w = bloom_w.max(1) as f32;
    let h = bloom_h.max(1) as f32;
    let one = [2.0 / w, 2.0 / h];
    let half = [1.0 / w, 1.0 / h];
    let a = [
        [half[0], half[1]],
        [half[0] + one[0], half[1]],
        [half[0], half[1] + one[1]],
        [half[0] + one[0], half[1] + one[1]],
    ];
    let b = [
        [1.0 + a[0][0], 1.0 + a[0][1]],
        [1.0 + a[1][0], 1.0 + a[1][1]],
        [1.0 + a[2][0], 1.0 + a[2][1]],
        [1.0 + a[3][0], 1.0 + a[3][1]],
    ];
    (a, b)
}

/// Build 16-tap filter kernel `(a[0..16], b[0..16])` for luminance pass 1 (`64 -> 8`) and pass 2 (`8 -> 1`)
/// (`r4_rendertarget_phase_luminance.cpp:90-97, 127-134`).
pub fn luminance_filter_taps(src_size: u32) -> ([[f32; 2]; 16], [[f32; 2]; 16]) {
    let src = src_size.max(1) as f32;
    let mut a = [[0.0f32; 2]; 16];
    let mut b = [[0.0f32; 2]; 16];
    for k in 0..16 {
        let x = ((k * 2 + 1) % 8) as f32;
        let y = ((k / 4) * 2 + 1) as f32;
        a[k] = [x / src, y / src];
        b[k] = [a[k][0] + 1.0, a[k][1] + 1.0];
    }
    (a, b)
}

/// Pack the 4 fullscreen quad corners `(bottom_left, top_left, bottom_right, top_right)`
/// matching `v_filter::uv[8]` in `CRenderTarget::phase_luminance` (`r4_rendertarget_phase_luminance.cpp:100-112`).
pub fn luminance_filter_quad_uvs(src_size: u32) -> [[[f32; 4]; 8]; 4] {
    let (a, b) = luminance_filter_taps(src_size);
    let mut out = [[[0.0f32; 4]; 8]; 4];
    for t in 0..8 {
        out[0][t] = [a[t][0], b[t][1], b[t + 8][1], a[t + 8][0]]; // left + down
        out[1][t] = [a[t][0], a[t][1], a[t + 8][1], a[t + 8][0]]; // left + up
        out[2][t] = [b[t][0], b[t][1], b[t + 8][1], b[t + 8][0]]; // right + down
        out[3][t] = [b[t][0], a[t][1], a[t + 8][1], b[t + 8][0]]; // right + up
    }
    out
}

/// Persistent exponential eye-adaptation state (`CRenderTarget::f_luminance_adapt`, initialized `0.5`
/// in `r4_rendertarget.cpp:930`, updated in `r4_rendertarget_phase_luminance.cpp:152-161`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LuminanceAdaptState {
    pub f_luminance_adapt: f32,
}

impl Default for LuminanceAdaptState {
    fn default() -> Self {
        Self {
            f_luminance_adapt: 0.5,
        }
    }
}

impl LuminanceAdaptState {
    #[inline]
    pub fn with_adapt(f_luminance_adapt: f32) -> Self {
        Self { f_luminance_adapt }
    }

    /// Advance `f_luminance_adapt` and compute `MiddleGray` (`[rgb.x, rgb.y, rgb.z, f_luminance_adapt]`).
    #[inline]
    pub fn step_middle_gray(&mut self, dt_sec: f32, cfg: &LuminanceConfig) -> [f32; 4] {
        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        let adapt_rate = if cfg.adaptation.is_finite() {
            cfg.adaptation.max(0.0)
        } else {
            0.0
        };
        self.f_luminance_adapt = 0.9 * self.f_luminance_adapt + 0.1 * dt * adapt_rate;

        let amount = if cfg.tonemap_enabled { cfg.amount } else { 0.0 };
        let none = [1.0f32, 0.0, 1.0];
        let full = [cfg.middle_gray, 1.0, cfg.low_lum];
        [
            none[0] + (full[0] - none[0]) * amount,
            none[1] + (full[1] - none[1]) * amount,
            none[2] + (full[2] - none[2]) * amount,
            self.f_luminance_adapt,
        ]
    }
}

// ============================================================================
// Screen-Space Sunshafts (`rendertarget_phase_sunshafts.cpp`)
// ============================================================================

/// Screen-space sunshaft console parameters (`ps_r2_ss_sunshafts_*`, `xrRender_console.cpp:289-290`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunshaftsConfig {
    pub length: f32,
    pub radius: f32,
}

impl Default for SunshaftsConfig {
    fn default() -> Self {
        Self {
            length: 1.0,
            radius: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SunshaftsPassKind {
    Mask,
    BlurPass1,
    BlurPass2,
    BlurPass3,
    Blend,
}

/// Single render pass descriptor emitted by [`sunshafts_schedule`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunshaftsPassDesc {
    pub kind: SunshaftsPassKind,
    pub element_index: u8,
    /// `Some(0)` = `rt_sunshafts_0`, `Some(1)` = `rt_sunshafts_1`, `None` = `rt_Generic`.
    pub dst_slot: Option<u8>,
    /// `c_sunshafts = [intensity_scaled, length, step_z, radius]`, or `None` on mask generation.
    pub c_sunshafts: Option<[f32; 4]>,
}

/// Half-texel screen UV bounds `(p0, p1)` matching `rendertarget_phase_sunshafts.cpp:22-23`.
#[inline]
pub fn sunshafts_uv_bounds(screen_w: u32, screen_h: u32) -> ([f32; 2], [f32; 2]) {
    let w = screen_w.max(1) as f32;
    let h = screen_h.max(1) as f32;
    ([0.5 / w, 0.5 / h], [(w + 0.5) / w, (h + 0.5) / h])
}

/// Build the 5-pass screen-space sunshafts schedule matching `CRenderTarget::phase_sunshafts`
/// (`rendertarget_phase_sunshafts.cpp:6-200`). Returns `None` when `env_intensity <= 0.001`.
pub fn sunshafts_schedule(
    env_intensity: f32,
    cfg: &SunshaftsConfig,
) -> Option<[SunshaftsPassDesc; 5]> {
    if !env_intensity.is_finite() || env_intensity <= 0.001 {
        return None;
    }
    let px = if env_intensity < 1.0 {
        env_intensity * 2.0
    } else {
        1.0
    };
    let py = cfg.length;
    let pw = cfg.radius;
    Some([
        SunshaftsPassDesc {
            kind: SunshaftsPassKind::Mask,
            element_index: 0,
            dst_slot: Some(0),
            c_sunshafts: None,
        },
        SunshaftsPassDesc {
            kind: SunshaftsPassKind::BlurPass1,
            element_index: 1,
            dst_slot: Some(1),
            c_sunshafts: Some([px, py, 1.0, pw]),
        },
        SunshaftsPassDesc {
            kind: SunshaftsPassKind::BlurPass2,
            element_index: 2,
            dst_slot: Some(0),
            c_sunshafts: Some([px, py, 0.7, pw]),
        },
        SunshaftsPassDesc {
            kind: SunshaftsPassKind::BlurPass3,
            element_index: 3,
            dst_slot: Some(1),
            c_sunshafts: Some([px, py, 0.3, pw]),
        },
        SunshaftsPassDesc {
            kind: SunshaftsPassKind::Blend,
            element_index: 4,
            dst_slot: None,
            c_sunshafts: Some([px, py, 0.0, pw]),
        },
    ])
}

// ============================================================================
// Combine-1 & Combine-2 Constants (`r4_rendertarget_phase_combine.cpp`)
// ============================================================================

/// Configuration for `combine_1` deferred lighting + ambient/SSAO composition (`r4_rendertarget_phase_combine.cpp:190-296`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Combine1Config {
    pub sun_lumscale_amb: f32,
    pub sun_lumscale_hemi: f32,
    pub nv_lum_factor: f32,
    pub fov_deg: f32,
    pub tex_jitter_size: u32,
    pub gloss: GlossConfig,
}

impl Default for Combine1Config {
    fn default() -> Self {
        Self {
            sun_lumscale_amb: 1.0,
            sun_lumscale_hemi: 1.0,
            nv_lum_factor: 0.0,
            fov_deg: 67.5,
            tex_jitter_size: TEX_JITTER_SIZE,
            gloss: GlossConfig::default(),
        }
    }
}

/// 96-byte 16B-aligned GPU constant block for `combine_1` (`r4_rendertarget_phase_combine.cpp:191-296`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Combine1GpuParams {
    pub l_ambient: [f32; 4],
    pub l_dynamic_color: [f32; 4],
    pub l_dynamic_dir: [f32; 4],
    pub env_color: [f32; 4],
    pub fog_color: [f32; 4],
    /// `[ssao_noise_tile_factor, ssao_kernel_size, jitter_scale_x, jitter_scale_y]`.
    pub ssao_and_jitter: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<Combine1GpuParams>() == 96);

/// Compute `combine_1` shader uniforms from `MixedEnvironment`, adapted sun light, view matrix & viewport
/// matching `r4_rendertarget_phase_combine.cpp:190-296`.
pub fn compute_combine1_params(
    env: &MixedEnvironment,
    sun_color: [f32; 3],
    sun_dir_world: [f32; 3],
    view: &XRayMatrix,
    screen_w: u32,
    screen_h: u32,
    cfg: &Combine1Config,
) -> Combine1GpuParams {
    const MIN_AMB: f32 = 0.001;
    const XRAY_EPS: f32 = 1e-5;

    let amb_scale = cfg.sun_lumscale_amb + cfg.nv_lum_factor;
    let l_ambient = [
        (env.ambient_color[0] * 2.0).max(MIN_AMB) * amb_scale,
        (env.ambient_color[1] * 2.0).max(MIN_AMB) * amb_scale,
        (env.ambient_color[2] * 2.0).max(MIN_AMB) * amb_scale,
        0.0,
    ];

    let hemi_scale = 2.0 * cfg.sun_lumscale_hemi;
    let env_color = [
        (env.hemi_color[0] * 2.0 + XRAY_EPS) * hemi_scale,
        (env.hemi_color[1] * 2.0 + XRAY_EPS) * hemi_scale,
        (env.hemi_color[2] * 2.0 + XRAY_EPS) * hemi_scale,
        env.weight,
    ];

    let fog_color = [env.fog_color[0], env.fog_color[1], env.fog_color[2], 0.0];

    let base_tan = (67.5_f32 * 0.5).to_radians().tan();
    let cur_tan = (cfg.fov_deg.clamp(5.0, 175.0) * 0.5).to_radians().tan().max(1e-4);
    let ssao_scale = base_tan / cur_tan;

    let jitter_sz = cfg.tex_jitter_size.max(1) as f32;
    let scale_x = screen_w.max(1) as f32 / jitter_sz;
    let scale_y = screen_h.max(1) as f32 / jitter_sz;

    let l_spec = compute_u_diffuse2s(sun_color, &cfg.gloss);
    let l_dynamic_color = [sun_color[0], sun_color[1], sun_color[2], l_spec];
    let dir_view = normalize_vec3(matrix_transform_dir(view, sun_dir_world));
    let l_dynamic_dir = [dir_view[0], dir_view[1], dir_view[2], 0.0];

    Combine1GpuParams {
        l_ambient,
        l_dynamic_color,
        l_dynamic_dir,
        env_color,
        fog_color,
        ssao_and_jitter: [2.0 * ssao_scale, 150.0 * ssao_scale, scale_x, scale_y],
    }
}

/// Edge-AA and motion-blur settings for `combine_2` (`xrRender_console.cpp:218-221`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Combine2Config {
    pub aa_barrier: [f32; 3],
    pub aa_weights: [f32; 3],
    pub aa_kernel: f32,
    pub m_blur: f32,
    pub aa_enabled: bool,
    pub distort_enabled: bool,
}

impl Default for Combine2Config {
    fn default() -> Self {
        Self {
            aa_barrier: [0.8, 0.1, 0.0],
            aa_weights: [0.25, 0.25, 0.0],
            aa_kernel: 0.5,
            m_blur: 0.0,
            aa_enabled: false,
            distort_enabled: false,
        }
    }
}

/// Select shader pass index inside `s_combine` for `combine_2` (`r4_rendertarget_phase_combine.cpp:679-687`).
#[inline]
pub fn combine2_shader_element(aa_enabled: bool, distort_enabled: bool) -> u8 {
    match (aa_enabled, distort_enabled) {
        (true, true) => 3,
        (true, false) => 1,
        (false, true) => 4,
        (false, false) => 2,
    }
}

/// One corner of the `combine_2` neighbor-sampling `v_aa` vertex format (`r4_rendertarget_phase_combine.cpp:614-670`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Combine2AaCornerUvs {
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub uv3: [f32; 2],
    pub uv4: [f32; 2],
    pub uv5: [f32; 4],
    pub uv6: [f32; 4],
}

/// Build `(BL, TL, BR, TR)` quad corner neighbor UVs for `combine_2` edge AA (`r4_rendertarget_phase_combine.cpp:626-670`).
pub fn combine2_aa_quad_uvs(screen_w: u32, screen_h: u32) -> [Combine2AaCornerUvs; 4] {
    let w = screen_w.max(1) as f32;
    let h = screen_h.max(1) as f32;
    let ddw = 1.0 / w;
    let ddh = 1.0 / h;
    let p0 = [0.5 / w, 0.5 / h];
    let p1 = [(w + 0.5) / w, (h + 0.5) / h];

    let build_corner = |u: f32, v: f32| Combine2AaCornerUvs {
        uv0: [u, v],
        uv1: [u - ddw, v - ddh],
        uv2: [u + ddw, v + ddh],
        uv3: [u + ddw, v - ddh],
        uv4: [u - ddw, v + ddh],
        uv5: [u - ddw, v, v, u + ddw],
        uv6: [u, v - ddh, v + ddh, u],
    };

    [
        build_corner(p0[0], p1[1]),
        build_corner(p0[0], p0[1]),
        build_corner(p1[0], p1[1]),
        build_corner(p1[0], p0[1]),
    ]
}

/// 192-byte GPU constant block for `combine_2` (`e_barrier`, `e_weights`, `e_kernel`, `m_blur`, `m_current`, `m_previous`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Combine2GpuParams {
    pub e_barrier: [f32; 4],
    pub e_weights: [f32; 4],
    pub e_kernel: [f32; 4],
    pub m_blur: [f32; 4],
    pub m_current: XRayMatrix,
    pub m_previous: XRayMatrix,
}

const _: () = assert!(std::mem::size_of::<Combine2GpuParams>() == 192);

/// Persistent per-camera view-projection history tracker for `combine_2` motion blur & SSFX reprojection
/// (`r4_rendertarget_phase_combine.cpp:85-99`).
#[derive(Debug, Clone, PartialEq)]
pub struct Combine2MotionState {
    pub saved_viewproj: Option<XRayMatrix>,
    pub saved_position: Option<[f32; 3]>,
    pub matrix_previous: XRayMatrix,
    pub matrix_current: XRayMatrix,
    pub position_previous: [f32; 3],
}

impl Default for Combine2MotionState {
    fn default() -> Self {
        Self::new()
    }
}

impl Combine2MotionState {
    pub fn new() -> Self {
        Self {
            saved_viewproj: None,
            saved_position: None,
            matrix_previous: XRAY_IDENTITY_MATRIX,
            matrix_current: XRAY_IDENTITY_MATRIX,
            position_previous: [0.0; 3],
        }
    }

    /// Update motion-blur reprojection matrices (unless `is_second_viewport_frame`, matching
    /// `!Device.m_SecondViewport.IsSVPFrame()`) and build `Combine2GpuParams`.
    pub fn step_and_gpu_params(
        &mut self,
        camera_pos: [f32; 3],
        inv_view: &XRayMatrix,
        proj: &XRayMatrix,
        full_transform: &XRayMatrix,
        is_second_viewport_frame: bool,
        cfg: &Combine2Config,
    ) -> Combine2GpuParams {
        if !is_second_viewport_frame {
            self.position_previous = self.saved_position.unwrap_or(camera_pos);
            self.saved_position = Some(camera_pos);

            let prev_vp = self.saved_viewproj.unwrap_or(*full_transform);
            self.matrix_previous = matrix_mul_44(&prev_vp, inv_view);
            self.matrix_current = *proj;
            self.saved_viewproj = Some(*full_transform);
        }

        let mb_scale = (cfg.m_blur * 0.5) / 12.0;
        Combine2GpuParams {
            e_barrier: [cfg.aa_barrier[0], cfg.aa_barrier[1], cfg.aa_barrier[2], 0.0],
            e_weights: [cfg.aa_weights[0], cfg.aa_weights[1], cfg.aa_weights[2], 0.0],
            e_kernel: [cfg.aa_kernel, cfg.aa_kernel, cfg.aa_kernel, 0.0],
            m_blur: [mb_scale, -mb_scale, 0.0, 0.0],
            m_current: self.matrix_current,
            m_previous: self.matrix_previous,
        }
    }
}

// ============================================================================
// SSAO, HDAO Compute Dispatch, SMAA & SSFX TAA Schedules
// ============================================================================

/// Half-res SSAO noise tile & kernel parameters matching `CRenderTarget::phase_ssao` /
/// `phase_downsamp` (`r4_rendertarget_phase_ssao.cpp:38-140`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsaoParams {
    pub noise_tile_factor: f32,
    pub kernel_size: f32,
    pub half_width: f32,
    pub half_height: f32,
    pub resolution: [f32; 4],
    pub jitter_scale: [f32; 2],
    pub downsamp_size: [u32; 2],
    pub downsamp_jitter_scale: [f32; 2],
}

/// Compute `phase_ssao` and `phase_downsamp` viewport & constant parameters.
pub fn compute_ssao_params(
    screen_w: u32,
    screen_h: u32,
    fov_deg: f32,
    ssao_half_data: bool,
) -> SsaoParams {
    let base_tan = 67.5_f32.to_radians().tan();
    let cur_tan = fov_deg.clamp(5.0, 89.0).to_radians().tan().max(1e-4);
    let scale = base_tan / cur_tan;

    let half_width = (screen_w.max(1) as f32 * 0.5).max(1.0);
    let half_height = (screen_h.max(1) as f32 * 0.5).max(1.0);
    let jitter_sz = TEX_JITTER_SIZE.max(1) as f32;

    let downsamp_size = if ssao_half_data {
        [(screen_w / 2).max(1), (screen_h / 2).max(1)]
    } else {
        [screen_w.max(1), screen_h.max(1)]
    };

    SsaoParams {
        noise_tile_factor: 2.0 * scale,
        kernel_size: 150.0 * scale,
        half_width,
        half_height,
        resolution: [
            half_width,
            half_height,
            1.0 / half_width,
            1.0 / half_height,
        ],
        jitter_scale: [half_width / jitter_sz, half_height / jitter_sz],
        downsamp_size,
        downsamp_jitter_scale: [
            downsamp_size[0] as f32 / jitter_sz,
            downsamp_size[1] as f32 / jitter_sz,
        ],
    }
}

pub const HDAO_GROUP_TEXEL_DIMENSION: u32 = 56;
pub const HDAO_GROUP_TEXEL_OVERLAP: u32 = 12;
pub const HDAO_GROUP_TEXEL_STEP: u32 =
    HDAO_GROUP_TEXEL_DIMENSION - 2 * HDAO_GROUP_TEXEL_OVERLAP;

/// Compute shader threadgroup dispatch dimensions for `CRenderTarget::phase_hdao`
/// (`r4_rendertarget_phase_hdao.cpp:3-39`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdaoDispatch {
    pub groups_x: u32,
    pub groups_y: u32,
    pub groups_z: u32,
}

#[inline]
pub fn compute_hdao_dispatch(screen_w: u32, screen_h: u32) -> HdaoDispatch {
    HdaoDispatch {
        groups_x: screen_w.max(1).div_ceil(HDAO_GROUP_TEXEL_STEP),
        groups_y: screen_h.max(1).div_ceil(HDAO_GROUP_TEXEL_STEP),
        groups_z: 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmaaPassKind {
    EdgeDetection,
    BlendingWeights,
    NeighborhoodBlending,
}

/// Single render pass descriptor emitted by [`smaa_schedule`] (`rendertarget_phase_smaa.cpp:4-101`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmaaPassDesc {
    pub kind: SmaaPassKind,
    pub element_index: u8,
    pub clear_rt: bool,
    pub stencil_enable: bool,
    pub stencil_cmp_equal: bool,
    pub dst_name: &'static str,
    pub copy_to_name: Option<&'static str>,
}

/// Build the 3-pass SMAA schedule (`EdgeDetection -> BlendingWeights -> NeighborhoodBlending`).
pub fn smaa_schedule(dx10_msaa: bool) -> [SmaaPassDesc; 3] {
    let final_rt = if dx10_msaa { "rt_Generic" } else { "rt_Color" };
    [
        SmaaPassDesc {
            kind: SmaaPassKind::EdgeDetection,
            element_index: 0,
            clear_rt: true,
            stencil_enable: true,
            stencil_cmp_equal: false,
            dst_name: "rt_smaa_edgetex",
            copy_to_name: None,
        },
        SmaaPassDesc {
            kind: SmaaPassKind::BlendingWeights,
            element_index: 1,
            clear_rt: true,
            stencil_enable: true,
            stencil_cmp_equal: true,
            dst_name: "rt_smaa_blendtex",
            copy_to_name: None,
        },
        SmaaPassDesc {
            kind: SmaaPassKind::NeighborhoodBlending,
            element_index: 2,
            clear_rt: false,
            stencil_enable: false,
            stencil_cmp_equal: false,
            dst_name: final_rt,
            copy_to_name: Some("rt_Generic_0"),
        },
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsfxTaaPassKind {
    PrepareMotionVectors,
    TemporalResolve,
    Sharpen,
}

/// Single render pass descriptor emitted by [`ssfx_taa_schedule`]
/// (`rendertarget_phase_smaa.cpp:105-180`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsfxTaaPassDesc {
    pub kind: SsfxTaaPassKind,
    pub element_index: u8,
    pub dst_name: &'static str,
    pub copy_to_name: Option<&'static str>,
}

/// Build the 3-pass SSFX TAA schedule (`PrepareMotionVectors -> TemporalResolve -> Sharpen`).
pub fn ssfx_taa_schedule(dx10_msaa: bool) -> [SsfxTaaPassDesc; 3] {
    let resolve_rt = if dx10_msaa { "rt_Generic" } else { "rt_Color" };
    [
        SsfxTaaPassDesc {
            kind: SsfxTaaPassKind::PrepareMotionVectors,
            element_index: 0,
            dst_name: "rt_ssfx_accum",
            copy_to_name: Some("rt_ssfx_taa"),
        },
        SsfxTaaPassDesc {
            kind: SsfxTaaPassKind::TemporalResolve,
            element_index: 1,
            dst_name: resolve_rt,
            copy_to_name: Some("rt_ssfx_prev_frame"),
        },
        SsfxTaaPassDesc {
            kind: SsfxTaaPassKind::Sharpen,
            element_index: 2,
            dst_name: "rt_Generic_0",
            copy_to_name: None,
        },
    ]
}

// ============================================================================
// Screen PPE Bridge, LUT Phase & Hardware Occlusion Pool
// ============================================================================

/// Prepared full-screen PPE pass outputs bridging [`PostprocessState`] and [`PostprocessNoiseState`]
/// (`CRenderTarget::phase_pp`, `r4_rendertarget_phase_PP.cpp:114-172`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostprocessPassPlan {
    pub active: bool,
    pub uses_color_map: bool,
    /// Shader element index: `4` when color-grading LUT blending is active, else `0`.
    pub shader_element: u32,
    pub gpu_params: PostprocessGpuParams,
    /// `[u0, v0, u1, v1]` (`u_calc_tc_duality_ss`).
    pub right_uv: [f32; 4],
    /// `[u0, v0, u1, v1]` (`u_calc_tc_duality_ss`).
    pub left_uv: [f32; 4],
    /// `[u0, v0, u1, v1]` (`u_calc_tc_noise`, zeroed when `active == false`).
    pub noise_uv: [f32; 4],
}

/// Evaluate runtime PPE pass parameters and advance grain noise only when `state.is_active()`.
#[allow(clippy::too_many_arguments)]
pub fn build_postprocess_pass_plan(
    state: &PostprocessState,
    noise_state: &mut PostprocessNoiseState,
    dt_sec: f32,
    noise_tex_w: u32,
    noise_tex_h: u32,
    rt_width: u32,
    rt_height: u32,
    device_width: u32,
    device_height: u32,
    noise_scale: f32,
) -> PostprocessPassPlan {
    let active = state.is_active();
    let uses_color_map = state.uses_color_map();
    let (right_uv, left_uv) = state.duality_uv_rects(rt_width, rt_height, device_height);
    let noise_uv = if active {
        noise_state.step_uv_rect(
            dt_sec,
            noise_tex_w,
            noise_tex_h,
            device_width,
            device_height,
            noise_scale,
            state.noise_fps,
        )
    } else {
        [0.0; 4]
    };
    PostprocessPassPlan {
        active,
        uses_color_map,
        shader_element: if uses_color_map { 4 } else { 0 },
        gpu_params: state.to_gpu_params(),
        right_uv,
        left_uv,
        noise_uv,
    }
}

pub const LUT_SHADER_NAME: &str = "pp_lut";
pub const LUT_NOISE_TEXTURE: &str = "fx\\blue_noise";
pub const LUT_ATLAS_TEXTURE: &str = "shaders\\lut_atlas";

/// SSFX color-grading LUT console uniform `ps_ssfx_lut` (`xrRender_console.cpp:454`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SsfxLutConfig {
    pub setup: [f32; 4],
}

/// Render-target and resource bindings for `CRenderTarget::phase_lut`
/// (`rendertarget_phase_lut.cpp:25-52`, `blender_lut.cpp:15-27`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LutPassDesc {
    pub dst_name: &'static str,
    pub copy_to_name: &'static str,
    pub shader_name: &'static str,
    pub src_image_name: &'static str,
    pub noise_texture: &'static str,
    pub lut_atlas_texture: &'static str,
    pub element_index: u8,
}

/// Build the DX11 render-target/resource descriptor for `CRenderTarget::phase_lut`.
#[inline]
pub fn lut_pass_desc(dx10_msaa: bool) -> LutPassDesc {
    LutPassDesc {
        dst_name: if dx10_msaa { "rt_Generic" } else { "rt_Color" },
        copy_to_name: "rt_Generic_0",
        shader_name: LUT_SHADER_NAME,
        src_image_name: "rt_Generic_0",
        noise_texture: LUT_NOISE_TEXTURE,
        lut_atlas_texture: LUT_ATLAS_TEXTURE,
        element_index: 0,
    }
}

/// Default hardware occlusion query pool size (`occq_size = 2 * 768` in `r__occlusion.h:3`).
pub const OCCQ_DEFAULT_LIMIT: u32 = 2 * 768;
/// Sentinel handle matching `R_occlusion::iInvalidHandle` (`r__occlusion.h:24`).
pub const OCCQ_INVALID_HANDLE: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OcclusionQueryEntry {
    order: u32,
    slot: u32,
}

/// Active occlusion query allocation returned by [`OcclusionQueryPool::begin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OcclusionQueryAllocation {
    pub handle: u32,
    pub order: u32,
    pub slot: u32,
}

/// Pure-Rust query heap slot & priority order manager matching `R_occlusion`
/// (`src/Layers/xrRender/r__occlusion.h`, `r__occlusion.cpp`).
#[derive(Debug, Clone)]
pub struct OcclusionQueryPool {
    enabled: bool,
    capacity: u32,
    pool: Vec<OcclusionQueryEntry>,
    used: Vec<Option<OcclusionQueryEntry>>,
    free_ids: Vec<u32>,
}

impl Default for OcclusionQueryPool {
    fn default() -> Self {
        Self::new(OCCQ_DEFAULT_LIMIT, true)
    }
}

impl OcclusionQueryPool {
    /// Allocate a pool of `limit` ordered query slots (`R_occlusion::occq_create`).
    pub fn new(limit: u32, enabled: bool) -> Self {
        let mut pool = Vec::with_capacity(limit as usize);
        for it in 0..limit {
            pool.push(OcclusionQueryEntry { order: it, slot: it });
        }
        // Keep highest `order` at index 0 and lowest `order` at `.last()` for O(1) pop().
        pool.reverse();
        Self {
            enabled,
            capacity: limit,
            pool,
            used: Vec::with_capacity(limit as usize),
            free_ids: Vec::with_capacity(limit as usize),
        }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    #[inline]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    #[inline]
    pub fn available_count(&self) -> usize {
        self.pool.len()
    }

    #[inline]
    pub fn in_flight_count(&self) -> usize {
        (self.capacity as usize).saturating_sub(self.pool.len())
    }

    /// Look up backend query slot index for an in-flight `handle`.
    #[inline]
    pub fn slot_for_handle(&self, handle: u32) -> Option<u32> {
        if handle == OCCQ_INVALID_HANDLE {
            return None;
        }
        self.used.get(handle as usize).and_then(|e| e.map(|q| q.slot))
    }

    /// Acquire the lowest-order free query slot matching `R_occlusion::occq_begin`.
    pub fn begin(&mut self) -> Option<OcclusionQueryAllocation> {
        if !self.enabled {
            return None;
        }
        let q = self.pool.pop()?;
        let handle = if let Some(id) = self.free_ids.pop() {
            self.used[id as usize] = Some(q);
            id
        } else {
            let id = self.used.len() as u32;
            self.used.push(Some(q));
            id
        };
        Some(OcclusionQueryAllocation {
            handle,
            order: q.order,
            slot: q.slot,
        })
    }

    /// Release `handle` back into the descending-order free pool (`R_occlusion::occq_get`).
    pub fn complete(&mut self, handle: u32) -> Option<u32> {
        if !self.enabled || handle == OCCQ_INVALID_HANDLE {
            return None;
        }
        let entry = self.used.get_mut(handle as usize)?;
        let q = entry.take()?;

        let mut pos = self.pool.len();
        while pos > 0 && self.pool[pos - 1].order < q.order {
            pos -= 1;
        }
        self.pool.insert(pos, q);
        self.free_ids.push(handle);
        Some(q.slot)
    }

    /// Reset all queries to unused descending order (`0..capacity`).
    pub fn reset(&mut self) {
        self.pool.clear();
        for it in 0..self.capacity {
            self.pool.push(OcclusionQueryEntry { order: it, slot: it });
        }
        self.pool.reverse();
        self.used.clear();
        self.free_ids.clear();
    }
}

// ============================================================================
// Screen-Space FVF Quad Vertex Builders (`FVF::TL` / `FVF::TL2uv`, `r4_rendertarget*.cpp`)
// ============================================================================

const EPS_QUAD: f32 = 1e-5; // `EPS` in `src/xrCore/vector.h:52`

/// 28-byte screen-space vertex matching `FVF::TL` (`src/Layers/xrRender/FVF.h:108-147`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenTlVertex {
    pub p: [f32; 4],
    pub color: u32,
    pub uv: [f32; 2],
}
const _: () = assert!(std::mem::size_of::<ScreenTlVertex>() == 28);

/// 36-byte dual-UV screen-space vertex matching `FVF::TL2uv` (`src/Layers/xrRender/FVF.h:151-196`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenTl2uvVertex {
    pub p: [f32; 4],
    pub color: u32,
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
}
const _: () = assert!(std::mem::size_of::<ScreenTl2uvVertex>() == 36);

#[inline]
const fn quad_rgba(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

#[inline]
fn build_ndc_tl2uv_quad() -> [ScreenTl2uvVertex; 4] {
    let c = quad_rgba(255, 255, 255, 255);
    [
        ScreenTl2uvVertex {
            p: [-1.0, -1.0, 0.0, 1.0],
            color: c,
            uv0: [0.0, 1.0],
            uv1: [0.0, 0.0],
        },
        ScreenTl2uvVertex {
            p: [-1.0, 1.0, EPS_S, 1.0],
            color: c,
            uv0: [0.0, 0.0],
            uv1: [0.0, 0.0],
        },
        ScreenTl2uvVertex {
            p: [1.0, -1.0, EPS_S, 1.0],
            color: c,
            uv0: [1.0, 1.0],
            uv1: [0.0, 0.0],
        },
        ScreenTl2uvVertex {
            p: [1.0, 1.0, EPS_S, 1.0],
            color: c,
            uv0: [1.0, 0.0],
            uv1: [0.0, 0.0],
        },
    ]
}

/// Fullscreen quad for `CRenderTarget::create_minmax_SM` (`r4_rendertarget_create_minmaxSM.cpp:10-18`).
#[inline]
pub fn build_create_minmax_sm_quad() -> [ScreenTl2uvVertex; 4] {
    build_ndc_tl2uv_quad()
}

/// Fullscreen quad for `CRenderTarget::mark_msaa_edges` (`r4_rendertarget_mark_msaa_edges.cpp:25-33`).
#[inline]
pub fn build_mark_msaa_edges_quad() -> [ScreenTl2uvVertex; 4] {
    build_ndc_tl2uv_quad()
}

/// Half-texel shifted fullscreen quad for `u_stencil_optimize` / `reset_light_marker(true)`
/// (`r4_rendertarget.cpp:1505-1513`).
#[inline]
pub fn build_stencil_optimize_quad(width: u32, height: u32) -> [ScreenTlVertex; 4] {
    let w = width as f32;
    let h = height as f32;
    let c = quad_rgba(255, 255, 255, 255);
    [
        ScreenTlVertex { p: [-0.5, h - 0.5, 0.0, 1.0], color: c, uv: [0.0, 0.0] },
        ScreenTlVertex { p: [-0.5, -0.5, 0.0, 1.0], color: c, uv: [0.0, 0.0] },
        ScreenTlVertex { p: [w - 0.5, h - 0.5, 0.0, 1.0], color: c, uv: [0.0, 0.0] },
        ScreenTlVertex { p: [w - 0.5, -0.5, 0.0, 1.0], color: c, uv: [0.0, 0.0] },
    ]
}

/// Fullscreen quad for `CRenderTarget::phase_ui_3d_icons_postprocess` (`r4_rendertarget.cpp:1341-1345`).
#[inline]
pub fn build_ui_3d_icons_postprocess_quad(width: u32, height: u32) -> [ScreenTlVertex; 4] {
    let w = width as f32;
    let h = height as f32;
    let c = quad_rgba(255, 255, 255, 255);
    [
        ScreenTlVertex { p: [0.0, h, EPS_S, 1.0], color: c, uv: [0.0, 1.0] },
        ScreenTlVertex { p: [0.0, 0.0, EPS_S, 1.0], color: c, uv: [0.0, 0.0] },
        ScreenTlVertex { p: [w, h, EPS_S, 1.0], color: c, uv: [1.0, 1.0] },
        ScreenTlVertex { p: [w, 0.0, EPS_S, 1.0], color: c, uv: [1.0, 0.0] },
    ]
}

/// Fullscreen quad for `CRenderTarget::phase_ssfx_rain` (`r4_rendertarget_phase_rain.cpp:17-34`).
#[inline]
pub fn build_phase_ssfx_rain_quad(width: u32, height: u32) -> [ScreenTlVertex; 4] {
    let w = width as f32;
    let h = height as f32;
    let c = quad_rgba(0, 0, 0, 255);
    [
        ScreenTlVertex { p: [0.0, h, EPS_S, 1.0], color: c, uv: [0.0, 1.0] },
        ScreenTlVertex { p: [0.0, 0.0, EPS_S, 1.0], color: c, uv: [0.0, 0.0] },
        ScreenTlVertex { p: [w, h, EPS_S, 1.0], color: c, uv: [1.0, 1.0] },
        ScreenTlVertex { p: [w, 0.0, EPS_S, 1.0], color: c, uv: [1.0, 0.0] },
    ]
}

/// Fullscreen mask quad for `CRenderTarget::draw_rain` (`r4_rendertarget_draw_rain.cpp:10-44`).
#[inline]
pub fn build_draw_rain_mask_quad(width: u32, height: u32) -> [ScreenTlVertex; 4] {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let c = quad_rgba(255, 255, 255, 255);
    let p0 = [0.5 / w, 0.5 / h];
    let p1 = [(w + 0.5) / w, (h + 0.5) / h];
    [
        ScreenTlVertex { p: [EPS_QUAD, h + EPS_QUAD, EPS_S, 1.0], color: c, uv: [p0[0], p1[1]] },
        ScreenTlVertex { p: [EPS_QUAD, EPS_QUAD, EPS_S, 1.0], color: c, uv: [p0[0], p0[1]] },
        ScreenTlVertex { p: [w + EPS_QUAD, h + EPS_QUAD, EPS_S, 1.0], color: c, uv: [p1[0], p1[1]] },
        ScreenTlVertex { p: [w + EPS_QUAD, EPS_QUAD, EPS_S, 1.0], color: c, uv: [p1[0], p0[1]] },
    ]
}

/// Projected depth-clipped jittered quad for `CRenderTarget::draw_rain`
/// (`r4_rendertarget_draw_rain.cpp:62-70, 173-189`).
#[inline]
pub fn build_draw_rain_light_quad(
    width: u32,
    f_rain_far: f32,
    cam_pos: [f32; 3],
    cam_dir: [f32; 3],
    full_transform: &XRayMatrix,
) -> [ScreenTl2uvVertex; 4] {
    let center_pt = [
        cam_pos[0] + cam_dir[0] * f_rain_far,
        cam_pos[1] + cam_dir[1] * f_rain_far,
        cam_pos[2] + cam_dir[2] * f_rain_far,
    ];
    let d_z = matrix_transform_projected(full_transform, center_pt).map_or(EPS_S, |p| p[2]);
    let scale_x = width as f32 / TEX_JITTER_SIZE as f32;
    let c = quad_rgba(255, 255, 255, 255);
    [
        ScreenTl2uvVertex { p: [-1.0, -1.0, d_z, 1.0], color: c, uv0: [0.0, 1.0], uv1: [0.0, scale_x] },
        ScreenTl2uvVertex { p: [-1.0, 1.0, d_z, 1.0], color: c, uv0: [0.0, 0.0], uv1: [0.0, 0.0] },
        ScreenTl2uvVertex { p: [1.0, -1.0, d_z, 1.0], color: c, uv0: [1.0, 1.0], uv1: [scale_x, scale_x] },
        ScreenTl2uvVertex { p: [1.0, 1.0, d_z, 1.0], color: c, uv0: [1.0, 0.0], uv1: [scale_x, 0.0] },
    ]
}

// ============================================================================
// Additional Post-Processing, Bloom & Combine Vertex Builders + R4 RT / Phase Specs
// ============================================================================

/// 48-byte fullscreen PPE vertex matching `TL_2c3uv` (`r4_rendertarget_phase_PP.cpp:95-112`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhasePpVertex {
    pub p: [f32; 4],
    pub color0_argb: u32,
    pub color1_argb: u32,
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
}
const _: () = assert!(std::mem::size_of::<PhasePpVertex>() == 48);

impl PostprocessPassPlan {
    /// Emit 4-vertex fullscreen strip/quad matching `CRenderTarget::phase_pp` (`r4_rendertarget_phase_PP.cpp:153-163`).
    pub fn build_quad(&self, width: u32, height: u32, du: f32, dv: f32) -> [PhasePpVertex; 4] {
        let w = width as f32;
        let h = height as f32;
        let r = self.right_uv;
        let l = self.left_uv;
        let n = self.noise_uv;
        let c0 = self.gpu_params.color0_argb;
        let c1 = self.gpu_params.color1_argb;
        [
            PhasePpVertex {
                p: [du, dv + h, EPS_S, 1.0],
                color0_argb: c0,
                color1_argb: c1,
                uv0: [r[0], r[3]],
                uv1: [l[0], l[3]],
                uv2: [n[0], n[3]],
            },
            PhasePpVertex {
                p: [du, dv, EPS_S, 1.0],
                color0_argb: c0,
                color1_argb: c1,
                uv0: [r[0], r[1]],
                uv1: [l[0], l[1]],
                uv2: [n[0], n[1]],
            },
            PhasePpVertex {
                p: [du + w, dv + h, EPS_S, 1.0],
                color0_argb: c0,
                color1_argb: c1,
                uv0: [r[2], r[3]],
                uv1: [l[2], l[3]],
                uv2: [n[2], n[3]],
            },
            PhasePpVertex {
                p: [du + w, dv, EPS_S, 1.0],
                color0_argb: c0,
                color1_argb: c1,
                uv0: [r[2], r[1]],
                uv1: [l[2], l[1]],
                uv2: [n[2], n[1]],
            },
        ]
    }
}

/// 48-byte 4-tap bloom build/fast-filter vertex matching `v_build` (`r4_rendertarget_phase_bloom.cpp:7-14`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BloomBuildVertex {
    pub p: [f32; 4],
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub uv3: [f32; 2],
}
const _: () = assert!(std::mem::size_of::<BloomBuildVertex>() == 48);

/// 144-byte 8x vec4 separable bloom filter vertex matching `v_filter` (`r4_rendertarget_phase_bloom.cpp:16-27`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BloomFilterVertex {
    pub p: [f32; 4],
    pub uv0: [f32; 4],
    pub uv1: [f32; 4],
    pub uv2: [f32; 4],
    pub uv3: [f32; 4],
    pub uv4: [f32; 4],
    pub uv5: [f32; 4],
    pub uv6: [f32; 4],
    pub uv7: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<BloomFilterVertex>() == 144);

/// Build the initial bloom extraction quad (`CRenderTarget::phase_bloom` -> `rt_Bloom_1`, lines 85-131).
pub fn build_bloom_build_quad(
    device_width: u32,
    device_height: u32,
    bloom_w: u32,
    bloom_h: u32,
) -> [BloomBuildVertex; 4] {
    let w = device_width.max(1) as f32;
    let h = device_height.max(1) as f32;
    let tw = bloom_w.max(1) as f32;
    let th = bloom_h.max(1) as f32;
    let aspect_w = (w * 0.5) / tw;
    let aspect_h = (h * 0.5) / th;
    let one_x = (1.0 / w) * aspect_w;
    let one_y = (1.0 / h) * aspect_h;
    let half_x = 0.5 / w;
    let half_y = 0.5 / h;

    let a_0 = [half_x, half_y];
    let a_1 = [half_x + one_x, half_y];
    let a_2 = [half_x, half_y + one_y];
    let a_3 = [half_x + one_x, half_y + one_y];
    let b_0 = [1.0 + a_0[0], 1.0 + a_0[1]];
    let b_1 = [1.0 + a_1[0], 1.0 + a_1[1]];
    let b_2 = [1.0 + a_2[0], 1.0 + a_2[1]];
    let b_3 = [1.0 + a_3[0], 1.0 + a_3[1]];

    [
        BloomBuildVertex {
            p: [EPS_QUAD, th + EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [a_0[0], b_0[1]],
            uv1: [a_1[0], b_1[1]],
            uv2: [a_2[0], b_2[1]],
            uv3: [a_3[0], b_3[1]],
        },
        BloomBuildVertex {
            p: [EPS_QUAD, EPS_QUAD, EPS_QUAD, 1.0],
            uv0: a_0,
            uv1: a_1,
            uv2: a_2,
            uv3: a_3,
        },
        BloomBuildVertex {
            p: [tw + EPS_QUAD, th + EPS_QUAD, EPS_QUAD, 1.0],
            uv0: b_0,
            uv1: b_1,
            uv2: b_2,
            uv3: b_3,
        },
        BloomBuildVertex {
            p: [tw + EPS_QUAD, EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [b_0[0], a_0[1]],
            uv1: [b_1[0], a_1[1]],
            uv2: [b_2[0], a_2[1]],
            uv3: [b_3[0], a_3[1]],
        },
    ]
}

/// Build the fast-bloom 4-tap quad (`CRenderTarget::phase_bloom` `R2FLAG_FASTBLOOM`, lines 155-189).
pub fn build_bloom_fast_filter_quad(
    bloom_w: u32,
    bloom_h: u32,
    kernel_b: f32,
) -> [BloomBuildVertex; 4] {
    let w = bloom_w.max(1) as f32;
    let h = bloom_h.max(1) as f32;
    let ddw = (1.0 / w) * kernel_b;
    let ddh = (1.0 / h) * kernel_b;
    let p0 = [0.5 / w, 0.5 / h];
    let p1 = [(w + 0.5) / w, (h + 0.5) / h];
    [
        BloomBuildVertex {
            p: [EPS_QUAD, h + EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [p0[0] - ddw, p1[1] - ddh],
            uv1: [p0[0] + ddw, p1[1] + ddh],
            uv2: [p0[0] + ddw, p1[1] - ddh],
            uv3: [p0[0] - ddw, p1[1] + ddh],
        },
        BloomBuildVertex {
            p: [EPS_QUAD, EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [p0[0] - ddw, p0[1] - ddh],
            uv1: [p0[0] + ddw, p0[1] + ddh],
            uv2: [p0[0] + ddw, p0[1] - ddh],
            uv3: [p0[0] - ddw, p0[1] + ddh],
        },
        BloomBuildVertex {
            p: [w + EPS_QUAD, h + EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [p1[0] - ddw, p1[1] - ddh],
            uv1: [p1[0] + ddw, p1[1] + ddh],
            uv2: [p1[0] + ddw, p1[1] - ddh],
            uv3: [p1[0] - ddw, p1[1] + ddh],
        },
        BloomBuildVertex {
            p: [w + EPS_QUAD, EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [p1[0] - ddw, p0[1] - ddh],
            uv1: [p1[0] + ddw, p0[1] + ddh],
            uv2: [p1[0] + ddw, p0[1] - ddh],
            uv3: [p1[0] - ddw, p0[1] + ddh],
        },
    ]
}

fn build_bloom_filter_vertices(w: f32, h: f32, a: &[[f32; 4]; 8]) -> [BloomFilterVertex; 4] {
    [
        BloomFilterVertex {
            p: [EPS_QUAD, h + EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [a[0][0], 1.0 + a[0][1], 0.0, 0.0],
            uv1: [a[1][0], 1.0 + a[1][1], 1.0 + a[1][2], a[1][3]],
            uv2: [a[2][0], 1.0 + a[2][1], 1.0 + a[2][2], a[2][3]],
            uv3: [a[3][0], 1.0 + a[3][1], 1.0 + a[3][2], a[3][3]],
            uv4: [a[4][0], 1.0 + a[4][1], 1.0 + a[4][2], a[4][3]],
            uv5: [a[5][0], 1.0 + a[5][1], 1.0 + a[5][2], a[5][3]],
            uv6: [a[6][0], 1.0 + a[6][1], 1.0 + a[6][2], a[6][3]],
            uv7: [a[7][0], 1.0 + a[7][1], 1.0 + a[7][2], a[7][3]],
        },
        BloomFilterVertex {
            p: [EPS_QUAD, EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [a[0][0], a[0][1], 0.0, 0.0],
            uv1: a[1],
            uv2: a[2],
            uv3: a[3],
            uv4: a[4],
            uv5: a[5],
            uv6: a[6],
            uv7: a[7],
        },
        BloomFilterVertex {
            p: [w + EPS_QUAD, h + EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [1.0 + a[0][0], 1.0 + a[0][1], 0.0, 0.0],
            uv1: [1.0 + a[1][0], 1.0 + a[1][1], 1.0 + a[1][2], 1.0 + a[1][3]],
            uv2: [1.0 + a[2][0], 1.0 + a[2][1], 1.0 + a[2][2], 1.0 + a[2][3]],
            uv3: [1.0 + a[3][0], 1.0 + a[3][1], 1.0 + a[3][2], 1.0 + a[3][3]],
            uv4: [1.0 + a[4][0], 1.0 + a[4][1], 1.0 + a[4][2], 1.0 + a[4][3]],
            uv5: [1.0 + a[5][0], 1.0 + a[5][1], 1.0 + a[5][2], 1.0 + a[5][3]],
            uv6: [1.0 + a[6][0], 1.0 + a[6][1], 1.0 + a[6][2], 1.0 + a[6][3]],
            uv7: [1.0 + a[7][0], 1.0 + a[7][1], 1.0 + a[7][2], 1.0 + a[7][3]],
        },
        BloomFilterVertex {
            p: [w + EPS_QUAD, EPS_QUAD, EPS_QUAD, 1.0],
            uv0: [1.0 + a[0][0], a[0][1], 0.0, 0.0],
            uv1: [1.0 + a[1][0], a[1][1], a[1][2], 1.0 + a[1][3]],
            uv2: [1.0 + a[2][0], a[2][1], a[2][2], 1.0 + a[2][3]],
            uv3: [1.0 + a[3][0], a[3][1], a[3][2], 1.0 + a[3][3]],
            uv4: [1.0 + a[4][0], a[4][1], a[4][2], 1.0 + a[4][3]],
            uv5: [1.0 + a[5][0], a[5][1], a[5][2], 1.0 + a[5][3]],
            uv6: [1.0 + a[6][0], a[6][1], a[6][2], 1.0 + a[6][3]],
            uv7: [1.0 + a[7][0], a[7][1], a[7][2], 1.0 + a[7][3]],
        },
    ]
}

/// Build the slow-bloom horizontal filter quad (`r4_rendertarget_phase_bloom.cpp:207-271`).
pub fn build_bloom_filter_quad_horizontal(bloom_w: u32, bloom_h: u32) -> [BloomFilterVertex; 4] {
    let w = bloom_w.max(1) as f32;
    let h = bloom_h.max(1) as f32;
    let taps = bloom_filter_taps_horizontal(bloom_w, bloom_h);
    build_bloom_filter_vertices(w, h, &taps)
}

/// Build the slow-bloom vertical filter quad (`r4_rendertarget_phase_bloom.cpp:287-351`).
pub fn build_bloom_filter_quad_vertical(bloom_w: u32, bloom_h: u32) -> [BloomFilterVertex; 4] {
    let w = bloom_w.max(1) as f32;
    let h = bloom_h.max(1) as f32;
    let taps = bloom_filter_taps_vertical(bloom_w, bloom_h);
    build_bloom_filter_vertices(w, h, &taps)
}

/// Fullscreen quad for every pass inside `CRenderTarget::phase_ssfx_bloom` (`r4_rendertarget_phase_bloom.cpp:401-405`).
#[inline]
pub fn build_ssfx_bloom_quad(width: u32, height: u32) -> [ScreenTlVertex; 4] {
    let w = width as f32;
    let h = height as f32;
    [
        ScreenTlVertex { p: [0.0, h, EPS_S, 1.0], color: 0, uv: [0.0, 1.0] },
        ScreenTlVertex { p: [0.0, 0.0, EPS_S, 1.0], color: 0, uv: [0.0, 0.0] },
        ScreenTlVertex { p: [w, h, EPS_S, 1.0], color: 0, uv: [1.0, 1.0] },
        ScreenTlVertex { p: [w, 0.0, EPS_S, 1.0], color: 0, uv: [1.0, 0.0] },
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsfxBloomPassKind {
    Build,
    LensFlares,
    LensBlur,
    Downsample,
    Upsample,
    Combine,
}

/// Single render pass descriptor emitted by [`ssfx_bloom_schedule`] (`r4_rendertarget_phase_bloom.cpp:383-548`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsfxBloomPassDesc {
    pub kind: SsfxBloomPassKind,
    pub shader_name: &'static str,
    pub element_index: u8,
    pub dst_name: &'static str,
    pub viewport_w: f32,
    pub viewport_h: f32,
    pub blur_setup: Option<[f32; 4]>,
}

/// Build the complete SSFX bloom render pass sequence matching `CRenderTarget::phase_ssfx_bloom`.
pub fn ssfx_bloom_schedule(
    width: u32,
    height: u32,
    mask_control_x: f32,
    bloom_blur_scale: f32,
) -> Vec<SsfxBloomPassDesc> {
    let w = width as f32;
    let h = height as f32;
    let mut passes = Vec::with_capacity(16);

    passes.push(SsfxBloomPassDesc {
        kind: SsfxBloomPassKind::Build,
        shader_name: "ssfx_bloom",
        element_index: 0,
        dst_name: "$user$ssfx_bloom1",
        viewport_w: w * 0.5,
        viewport_h: h * 0.5,
        blur_setup: None,
    });

    if mask_control_x > 0.0 {
        passes.push(SsfxBloomPassDesc {
            kind: SsfxBloomPassKind::LensFlares,
            shader_name: "ssfx_bloom_flares",
            element_index: 0,
            dst_name: "$user$ssfx_bloom_tmp4",
            viewport_w: w * 0.25,
            viewport_h: h * 0.25,
            blur_setup: None,
        });
        let lens_targets = ["$user$ssfx_bloom_tmp4_2", "$user$ssfx_bloom_lens"];
        for (i, &dst_name) in lens_targets.iter().enumerate() {
            passes.push(SsfxBloomPassDesc {
                kind: SsfxBloomPassKind::LensBlur,
                shader_name: "ssfx_bloom_flares",
                element_index: (1 + i) as u8,
                dst_name,
                viewport_w: w * 0.25,
                viewport_h: h * 0.25,
                blur_setup: Some([w * 0.25, h * 0.25, 0.0, 2.0 + 3.0 * (i as f32)]),
            });
        }
    }

    const DOWN_RTS: [&str; 6] = [
        "$user$ssfx_bloom_tmp2",
        "$user$ssfx_bloom_tmp4",
        "$user$ssfx_bloom_tmp8",
        "$user$ssfx_bloom_tmp16",
        "$user$ssfx_bloom_tmp32",
        "$user$ssfx_bloom_tmp64",
    ];
    for (i, &dst_name) in DOWN_RTS.iter().enumerate() {
        let scale = (1u32 << (i + 1)) as f32;
        passes.push(SsfxBloomPassDesc {
            kind: SsfxBloomPassKind::Downsample,
            shader_name: "ssfx_bloom_downsample",
            element_index: i as u8,
            dst_name,
            viewport_w: w / scale,
            viewport_h: h / scale,
            blur_setup: Some([w / scale, h / scale, 0.0, bloom_blur_scale]),
        });
    }

    const UP_RTS: [&str; 5] = [
        "$user$ssfx_bloom_tmp32_2",
        "$user$ssfx_bloom_tmp16_2",
        "$user$ssfx_bloom_tmp8_2",
        "$user$ssfx_bloom_tmp4_2",
        "$user$ssfx_bloom_tmp2",
    ];
    for (i, &dst_name) in UP_RTS.iter().enumerate() {
        let scale = (1u32 << (5 - i)) as f32;
        passes.push(SsfxBloomPassDesc {
            kind: SsfxBloomPassKind::Upsample,
            shader_name: "ssfx_bloom_upsample",
            element_index: i as u8,
            dst_name,
            viewport_w: w / scale,
            viewport_h: h / scale,
            blur_setup: Some([w / scale, h / scale, 0.0, bloom_blur_scale]),
        });
    }

    // `Combine` runs before restoring device viewport (remaining at half-resolution `w / 2, h / 2`).
    passes.push(SsfxBloomPassDesc {
        kind: SsfxBloomPassKind::Combine,
        shader_name: "ssfx_bloom",
        element_index: 1,
        dst_name: "$user$ssfx_bloom1",
        viewport_w: w * 0.5,
        viewport_h: h * 0.5,
        blur_setup: None,
    });

    passes
}

/// 88-byte anti-aliasing combine vertex matching `v_aa` (`r4_rendertarget_phase_combine.cpp:614-624`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Combine2AaVertex {
    pub p: [f32; 4],
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub uv3: [f32; 2],
    pub uv4: [f32; 2],
    pub uv5: [f32; 4],
    pub uv6: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<Combine2AaVertex>() == 88);

/// Build the 4-vertex `combine_2` anti-aliasing quad (`r4_rendertarget_phase_combine.cpp:626-671`).
pub fn build_combine2_aa_quad(width: u32, height: u32) -> [Combine2AaVertex; 4] {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let corners = combine2_aa_quad_uvs(width, height);
    let positions = [
        [EPS_QUAD, h + EPS_QUAD, EPS_QUAD, 1.0],
        [EPS_QUAD, EPS_QUAD, EPS_QUAD, 1.0],
        [w + EPS_QUAD, h + EPS_QUAD, EPS_QUAD, 1.0],
        [w + EPS_QUAD, EPS_QUAD, EPS_QUAD, 1.0],
    ];
    std::array::from_fn(|i| Combine2AaVertex {
        p: positions[i],
        uv0: corners[i].uv0,
        uv1: corners[i].uv1,
        uv2: corners[i].uv2,
        uv3: corners[i].uv3,
        uv4: corners[i].uv4,
        uv5: corners[i].uv5,
        uv6: corners[i].uv6,
    })
}

/// Build the fullscreen NDC quad for `CRenderTarget::phase_combine_volumetric`
/// (`r4_rendertarget_phase_combine.cpp:892-896`).
#[inline]
pub fn build_combine_volumetric_quad() -> [ScreenTlVertex; 4] {
    [
        ScreenTlVertex { p: [-1.0, 1.0, 0.0, 1.0], color: 0, uv: [0.0, 1.0] },
        ScreenTlVertex { p: [-1.0, -1.0, 0.0, 0.0], color: 0, uv: [0.0, 0.0] },
        ScreenTlVertex { p: [1.0, 1.0, 1.0, 1.0], color: 0, uv: [1.0, 1.0] },
        ScreenTlVertex { p: [1.0, -1.0, 1.0, 0.0], color: 0, uv: [1.0, 0.0] },
    ]
}

// ============================================================================
// R4 Render Target Allocation Catalog (`CRenderTarget::CRenderTarget`, `r4_rendertarget.cpp`)
// ============================================================================

/// Logical pixel format for DX11/DX12 R4 render targets (`D3DFMT_*` -> `DXGI_FORMAT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R4TargetFormat {
    Rgba16F,
    Rgba8Unorm,
    R10G10B10A2Unorm,
    D24UnormS8Uint,
    R32F,
    R16F,
}

/// Feature flags controlling which RTs `CRenderTarget` allocates (`r4_rendertarget.cpp:487-993`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R4RenderTargetConfig {
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    pub smap_size: u32,
    pub gpu_num: u32,
    pub dx10_msaa: bool,
    pub dx11_hdr10: bool,
    pub hdr10_generic_r_on: bool,
    pub mrtmixdepth: bool,
    pub fp16_blend: bool,
    pub advancedpp: bool,
    pub ssfx_sss: bool,
    pub ssfx_bloom: bool,
    pub dx10_minmax_sm: bool,
    pub ssao_opt_data: bool,
    pub ssao_half_data: bool,
    pub is_nvidia_vendor: bool,
    pub ssao_hdao: bool,
    pub ssao_ultra: bool,
    pub atlas_enabled: bool,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub atlas_samples: u32,
}

impl Default for R4RenderTargetConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            sample_count: 1,
            smap_size: 2048,
            gpu_num: 1,
            dx10_msaa: false,
            dx11_hdr10: false,
            hdr10_generic_r_on: false,
            mrtmixdepth: true,
            fp16_blend: true,
            advancedpp: true,
            ssfx_sss: true,
            ssfx_bloom: true,
            dx10_minmax_sm: false,
            ssao_opt_data: true,
            ssao_half_data: false,
            is_nvidia_vendor: true,
            ssao_hdao: false,
            ssao_ultra: false,
            atlas_enabled: true,
            atlas_width: 1024,
            atlas_height: 1024,
            atlas_samples: 4,
        }
    }
}

/// Single render target specification emitted by [`build_r4_render_target_catalog`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R4RenderTargetSpec {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub format: R4TargetFormat,
    pub sample_count: u32,
    pub uav: bool,
}

/// Build the canonical R4/Anomaly render-target allocation list matching `CRenderTarget::CRenderTarget`.
pub fn build_r4_render_target_catalog(cfg: &R4RenderTargetConfig) -> Vec<R4RenderTargetSpec> {
    let w = cfg.width.max(1);
    let h = cfg.height.max(1);
    let samples = if cfg.dx10_msaa {
        cfg.sample_count.max(1)
    } else {
        1
    };
    let generic_fmt = if cfg.dx11_hdr10 {
        R4TargetFormat::Rgba16F
    } else {
        R4TargetFormat::Rgba8Unorm
    };
    let mut list = Vec::with_capacity(64);
    let mut push = |name: &str, rw: u32, rh: u32, format: R4TargetFormat, sc: u32, uav: bool| {
        list.push(R4RenderTargetSpec {
            name: name.to_string(),
            width: rw.max(1),
            height: rh.max(1),
            format,
            sample_count: sc.max(1),
            uav,
        });
    };

    push("$user$position", w, h, R4TargetFormat::Rgba16F, samples, false);
    if cfg.dx10_msaa {
        push("$user$msaadepth", w, h, R4TargetFormat::D24UnormS8Uint, samples, false);
    }
    push("$user$temp_zb", w, h, R4TargetFormat::D24UnormS8Uint, 1, false);

    if cfg.mrtmixdepth {
        let albedo_fmt = if cfg.dx11_hdr10 {
            R4TargetFormat::Rgba16F
        } else {
            R4TargetFormat::Rgba8Unorm
        };
        push("$user$albedo", w, h, albedo_fmt, samples, false);
        push("$user$accum", w, h, R4TargetFormat::Rgba16F, samples, false);
    } else if cfg.fp16_blend {
        push("$user$albedo", w, h, R4TargetFormat::Rgba16F, samples, false);
        push("$user$accum", w, h, R4TargetFormat::Rgba16F, samples, false);
    } else {
        push("$user$albedo", w, h, R4TargetFormat::Rgba8Unorm, samples, false);
        push("$user$accum", w, h, R4TargetFormat::Rgba16F, samples, false);
        push("$user$accum_temp", w, h, R4TargetFormat::Rgba16F, samples, false);
    }

    push("$user$generic0", w, h, generic_fmt, 1, false);
    push("$user$generic1", w, h, generic_fmt, 1, false);
    push("$user$generic", w, h, generic_fmt, 1, false);
    push("$user$scopeRT", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$heat", w, h, R4TargetFormat::Rgba8Unorm, samples, false);
    push("$user$generic_temp", w, h, generic_fmt, samples, false);
    push("$user$dof", w, h, generic_fmt, 1, false);

    let ui_fmt = if cfg.dx11_hdr10 {
        R4TargetFormat::R10G10B10A2Unorm
    } else {
        R4TargetFormat::Rgba8Unorm
    };
    push("$user$viewport2", w, h, ui_fmt, 1, false);
    push("$user$ui", w, h, ui_fmt, 1, false);

    if cfg.atlas_enabled {
        let aw = cfg.atlas_width.max(1);
        let ah = cfg.atlas_height.max(1);
        let asc = cfg.atlas_samples.max(1);
        push("$user$ui_3d_icons", aw, ah, R4TargetFormat::Rgba8Unorm, 1, false);
        push("$user$ui_3d_icons_raw", aw, ah, R4TargetFormat::Rgba8Unorm, 1, false);
        if asc > 1 {
            push("$user$ui_3d_icons_raw_msaa", aw, ah, R4TargetFormat::Rgba8Unorm, asc, false);
        }
        push("$user$ui_3d_icons_depth", aw, ah, R4TargetFormat::D24UnormS8Uint, asc, false);
    }

    if cfg.dx11_hdr10 {
        push("$user$hdr10_halfres0", w / 2, h / 2, R4TargetFormat::Rgba16F, 1, false);
        push("$user$hdr10_halfres1", w / 2, h / 2, R4TargetFormat::Rgba16F, 1, false);
    }

    push("$user$sun_shafts0", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$sun_shafts1", w, h, R4TargetFormat::Rgba8Unorm, 1, false);

    push("$user$blur_h_2", w / 2, h / 2, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$blur_2", w / 2, h / 2, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$blur_h_4", w / 4, h / 4, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$blur_4", w / 4, h / 4, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$blur_h_8", w / 8, h / 8, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$blur_8", w / 8, h / 8, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$pp_bloom", w, h, R4TargetFormat::Rgba8Unorm, 1, false);

    push("$user$ssfx_taa", w, h, R4TargetFormat::Rgba16F, samples, false);
    push("$user$ssfx_prev_frame", w, h, generic_fmt, 1, false);
    push("$user$ssfx_motion_vectors", w, h, R4TargetFormat::Rgba16F, samples, false);
    push("$user$ssfx", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_temp", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_temp2", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_temp3", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_accum", w, h, R4TargetFormat::Rgba16F, samples, false);
    push("$user$ssfx_ssr", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_water", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_ao", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_il", w, h, R4TargetFormat::Rgba8Unorm, 1, false);

    if cfg.ssfx_sss {
        push("$user$ssfx_sss", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
        push("$user$ssfx_sss_ext", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
        push("$user$ssfx_sss_ext2", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
        push("$user$ssfx_sss_tmp", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    }

    if cfg.ssfx_bloom {
        push("$user$ssfx_bloom1", w / 2, h / 2, R4TargetFormat::Rgba16F, 1, false);
        push("$user$ssfx_bloom_emissive", w, h, R4TargetFormat::Rgba8Unorm, samples, false);
        push("$user$ssfx_bloom_lens", w / 4, h / 4, R4TargetFormat::Rgba8Unorm, 1, false);
        for (name, div) in [
            ("$user$ssfx_bloom_tmp2", 2u32),
            ("$user$ssfx_bloom_tmp4", 4),
            ("$user$ssfx_bloom_tmp8", 8),
            ("$user$ssfx_bloom_tmp16", 16),
            ("$user$ssfx_bloom_tmp32", 32),
            ("$user$ssfx_bloom_tmp64", 64),
            ("$user$ssfx_bloom_tmp32_2", 32),
            ("$user$ssfx_bloom_tmp16_2", 16),
            ("$user$ssfx_bloom_tmp8_2", 8),
            ("$user$ssfx_bloom_tmp4_2", 4),
        ] {
            push(name, w / div, h / div, R4TargetFormat::Rgba16F, 1, false);
        }
    }

    push("$user$ssfx_volumetric", w / 8, h / 8, R4TargetFormat::Rgba16F, 1, false);
    push("$user$ssfx_volumetric_tmp", w / 8, h / 8, R4TargetFormat::Rgba16F, 1, false);
    push("$user$ssfx_rain", w / 8, h / 8, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_water_waves", 512, 512, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$ssfx_prev_p", w, h, R4TargetFormat::Rgba16F, samples, false);

    if cfg.dx10_msaa {
        let gr_fmt = if cfg.hdr10_generic_r_on {
            R4TargetFormat::Rgba16F
        } else {
            R4TargetFormat::Rgba8Unorm
        };
        push("$user$generic0_r", w, h, gr_fmt, samples, false);
        push("$user$generic1_r", w, h, gr_fmt, samples, false);
    }
    if cfg.advancedpp {
        push("$user$generic2", w, h, R4TargetFormat::Rgba16F, samples, false);
    }

    let smap_sz = cfg.smap_size.max(64);
    push("$user$smap_depth", smap_sz, smap_sz, R4TargetFormat::D24UnormS8Uint, 1, false);
    if cfg.dx10_minmax_sm {
        push("$user$smap_depth_minmax", smap_sz / 4, smap_sz / 4, R4TargetFormat::R32F, 1, false);
    }

    push("$user$bloom1", BLOOM_SIZE_X, BLOOM_SIZE_Y, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$bloom2", BLOOM_SIZE_X, BLOOM_SIZE_Y, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$smaa_edgetex", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$smaa_blendtex", w, h, R4TargetFormat::Rgba8Unorm, 1, false);
    push("$user$lum_t64", 64, 64, R4TargetFormat::Rgba16F, 1, false);
    push("$user$lum_t8", 8, 8, R4TargetFormat::Rgba16F, 1, false);
    for it in 0..(cfg.gpu_num.max(1) * 2) {
        push(&format!("$user$luminance_{it}"), 1, 1, R4TargetFormat::R32F, 1, false);
    }

    if cfg.ssao_opt_data {
        let (sw, sh) = if cfg.ssao_half_data {
            (w / 2, h / 2)
        } else {
            (w, h)
        };
        let fmt = if cfg.is_nvidia_vendor {
            R4TargetFormat::R32F
        } else {
            R4TargetFormat::R16F
        };
        push("$user$half_depth", sw, sh, fmt, 1, false);
    }
    if cfg.ssao_hdao && cfg.ssao_ultra {
        push("$user$ssao_temp", w, h, R4TargetFormat::R16F, 1, true);
    }

    list
}

// ============================================================================
// R4 Render-Phase Target/Depth/Clear/Stencil/Cull/ColorWrite State Machine
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R4RtSlot {
    Null,
    BaseRt,
    Position,
    Albedo,
    Accumulator,
    AccumulatorTemp,
    Heat,
    SsfxMotionVectors,
    Generic0,
    Generic1,
    Generic0Msaa,
    Generic1Msaa,
    Generic2,
    SmapSurf,
    CurrentRt0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R4DepthSlot {
    Null,
    BaseZb,
    MsaaDepth,
    SmapDepth,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct R4PhaseClearSpec {
    pub rt0_rgba: Option<[f32; 4]>,
    pub rt1_rgba: Option<[f32; 4]>,
    pub depth_value: Option<f32>,
    pub stencil_value: Option<u8>,
}

impl R4PhaseClearSpec {
    pub const NONE: Self = Self {
        rt0_rgba: None,
        rt1_rgba: None,
        depth_value: None,
        stencil_value: None,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R4RenderPhaseKind {
    ScenePrepare,
    SceneBegin,
    SceneEndAlbedoResolve,
    AccumulatorFirstClear,
    AccumulatorAppend,
    VolumetricAccumulator,
    ShadowDirect,
    ShadowDirectTshadowsClear,
    ShadowSpotClear,
    ShadowSpot,
    OcclusionQuery,
    Wallmarks,
    CombineVolumetric,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct R4PhaseStateOptions {
    pub dx10_msaa: bool,
    pub albedo_wo: bool,
    pub fp16_blend: bool,
    pub hw_smap: bool,
    pub ssfx_volumetric: bool,
}

/// Complete R4 render-phase pipeline binding & clear contract
/// (`r4_rendertarget_phase_{scene,accumulator,smap_D,smap_S,occq,combine}.cpp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct R4RenderPhaseSpec {
    pub rts: [R4RtSlot; 4],
    pub depth: R4DepthSlot,
    pub clears: R4PhaseClearSpec,
    pub cull_mode: Option<CullMode>,
    pub stencil: Option<StencilState>,
    pub color_write_enable: Option<[bool; 4]>,
}

/// Compute exact RT/DSV/clear/cull/stencil/color-write contract for an R4 render phase.
pub fn r4_phase_spec(kind: R4RenderPhaseKind, opts: R4PhaseStateOptions) -> R4RenderPhaseSpec {
    let scene_zb = if opts.dx10_msaa {
        R4DepthSlot::MsaaDepth
    } else {
        R4DepthSlot::BaseZb
    };
    let stencil_lessequal_01 = StencilState {
        enable: true,
        func: CmpFunc::LessEqual,
        mask: 0xFF,
        write_mask: 0x00,
        fail: StencilOp::Keep,
        pass: StencilOp::Keep,
        zfail: StencilOp::Keep,
        ref_value: 0x01,
    };
    match kind {
        R4RenderPhaseKind::ScenePrepare => R4RenderPhaseSpec {
            rts: [R4RtSlot::Position, R4RtSlot::Null, R4RtSlot::Null, R4RtSlot::Null],
            depth: scene_zb,
            clears: R4PhaseClearSpec {
                rt0_rgba: Some([0.0; 4]),
                rt1_rgba: Some([0.0; 4]), // `rt_Heat`
                depth_value: Some(1.0),
                stencil_value: Some(0),
            },
            cull_mode: None,
            stencil: None,
            color_write_enable: None,
        },
        R4RenderPhaseKind::SceneBegin => R4RenderPhaseSpec {
            rts: [
                R4RtSlot::Position,
                if opts.albedo_wo { R4RtSlot::Accumulator } else { R4RtSlot::Albedo },
                R4RtSlot::Heat,
                R4RtSlot::SsfxMotionVectors,
            ],
            depth: scene_zb,
            clears: R4PhaseClearSpec::NONE,
            cull_mode: Some(CullMode::Ccw),
            stencil: Some(StencilState {
                enable: true,
                func: CmpFunc::Always,
                mask: 0xFF,
                write_mask: 0x7F,
                fail: StencilOp::Keep,
                pass: StencilOp::Replace,
                zfail: StencilOp::Keep,
                ref_value: 0x01,
            }),
            color_write_enable: Some([true; 4]),
        },
        R4RenderPhaseKind::SceneEndAlbedoResolve => R4RenderPhaseSpec {
            rts: [
                if opts.albedo_wo { R4RtSlot::Albedo } else { R4RtSlot::Null },
                R4RtSlot::Null,
                R4RtSlot::Null,
                R4RtSlot::Null,
            ],
            depth: scene_zb,
            clears: R4PhaseClearSpec::NONE,
            cull_mode: if opts.albedo_wo { Some(CullMode::None) } else { None },
            stencil: if opts.albedo_wo { Some(stencil_lessequal_01) } else { None },
            color_write_enable: if opts.albedo_wo { Some([true; 4]) } else { None },
        },
        R4RenderPhaseKind::AccumulatorFirstClear => R4RenderPhaseSpec {
            rts: [R4RtSlot::Accumulator, R4RtSlot::Null, R4RtSlot::Null, R4RtSlot::Null],
            depth: scene_zb,
            clears: R4PhaseClearSpec {
                rt0_rgba: Some([0.0; 4]),
                rt1_rgba: None,
                depth_value: None,
                stencil_value: None,
            },
            cull_mode: Some(CullMode::None),
            stencil: Some(stencil_lessequal_01),
            color_write_enable: Some([true; 4]),
        },
        R4RenderPhaseKind::AccumulatorAppend => R4RenderPhaseSpec {
            rts: [
                if opts.fp16_blend { R4RtSlot::Accumulator } else { R4RtSlot::AccumulatorTemp },
                R4RtSlot::Null,
                R4RtSlot::Null,
                R4RtSlot::Null,
            ],
            depth: scene_zb,
            clears: R4PhaseClearSpec::NONE,
            cull_mode: None,
            stencil: None,
            color_write_enable: None,
        },
        R4RenderPhaseKind::VolumetricAccumulator => R4RenderPhaseSpec {
            rts: [R4RtSlot::Generic2, R4RtSlot::Null, R4RtSlot::Null, R4RtSlot::Null],
            depth: if opts.ssfx_volumetric { R4DepthSlot::Null } else { scene_zb },
            clears: R4PhaseClearSpec::NONE,
            cull_mode: Some(CullMode::None),
            stencil: Some(StencilState::default()),
            color_write_enable: Some([true; 4]),
        },
        R4RenderPhaseKind::ShadowDirect => R4RenderPhaseSpec {
            rts: [R4RtSlot::SmapSurf, R4RtSlot::Null, R4RtSlot::Null, R4RtSlot::Null],
            depth: R4DepthSlot::SmapDepth,
            clears: R4PhaseClearSpec {
                rt0_rgba: None,
                rt1_rgba: None,
                depth_value: Some(1.0),
                stencil_value: None,
            },
            cull_mode: None,
            stencil: Some(StencilState::default()),
            color_write_enable: None,
        },
        R4RenderPhaseKind::ShadowDirectTshadowsClear | R4RenderPhaseKind::ShadowSpotClear => {
            if kind == R4RenderPhaseKind::ShadowSpotClear {
                R4RenderPhaseSpec {
                    rts: [R4RtSlot::Null; 4],
                    depth: R4DepthSlot::SmapDepth,
                    clears: R4PhaseClearSpec {
                        rt0_rgba: None,
                        rt1_rgba: None,
                        depth_value: Some(1.0),
                        stencil_value: None,
                    },
                    cull_mode: None,
                    stencil: None,
                    color_write_enable: None,
                }
            } else {
                R4RenderPhaseSpec {
                    rts: [R4RtSlot::CurrentRt0, R4RtSlot::Null, R4RtSlot::Null, R4RtSlot::Null],
                    depth: R4DepthSlot::Null,
                    clears: R4PhaseClearSpec {
                        rt0_rgba: Some([1.0; 4]),
                        rt1_rgba: None,
                        depth_value: None,
                        stencil_value: None,
                    },
                    cull_mode: None,
                    stencil: None,
                    color_write_enable: Some([true; 4]),
                }
            }
        }
        R4RenderPhaseKind::ShadowSpot => R4RenderPhaseSpec {
            rts: [R4RtSlot::SmapSurf, R4RtSlot::Null, R4RtSlot::Null, R4RtSlot::Null],
            depth: R4DepthSlot::SmapDepth,
            clears: R4PhaseClearSpec::NONE,
            cull_mode: Some(CullMode::Ccw),
            stencil: Some(StencilState::default()),
            color_write_enable: if opts.hw_smap { Some([false; 4]) } else { None },
        },
        R4RenderPhaseKind::OcclusionQuery => R4RenderPhaseSpec {
            rts: [
                if opts.dx10_msaa { R4RtSlot::Null } else { R4RtSlot::BaseRt },
                R4RtSlot::Null,
                R4RtSlot::Null,
                R4RtSlot::Null,
            ],
            depth: scene_zb,
            clears: R4PhaseClearSpec::NONE,
            cull_mode: Some(CullMode::Ccw),
            stencil: Some(stencil_lessequal_01),
            color_write_enable: Some([false; 4]),
        },
        R4RenderPhaseKind::Wallmarks => R4RenderPhaseSpec {
            rts: [R4RtSlot::Albedo, R4RtSlot::Null, R4RtSlot::Null, R4RtSlot::Null],
            depth: scene_zb,
            clears: R4PhaseClearSpec::NONE,
            cull_mode: Some(CullMode::Ccw),
            stencil: Some(stencil_lessequal_01),
            color_write_enable: Some([true, true, true, false]),
        },
        R4RenderPhaseKind::CombineVolumetric => R4RenderPhaseSpec {
            rts: [
                if opts.dx10_msaa { R4RtSlot::Generic0Msaa } else { R4RtSlot::Generic0 },
                if opts.dx10_msaa { R4RtSlot::Generic1Msaa } else { R4RtSlot::Generic1 },
                R4RtSlot::Null,
                R4RtSlot::Null,
            ],
            depth: scene_zb,
            clears: R4PhaseClearSpec::NONE,
            cull_mode: None,
            stencil: None,
            color_write_enable: Some([true, true, true, false]),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postprocess_noise_state_tiles_and_steps_at_fps() {
        let mut st = PostprocessNoiseState::new(42);
        // 1920x1080 with 64x64 noise @ scale=1.0, fps=20 (0.05s period)
        let r0 = st.step_uv_rect(0.01, 64, 64, 1920, 1080, 1.0, 20.0);
        // cnt_w = 1920 / 64 = 30 => end_u = start_u + 31.0
        assert!((r0[2] - r0[0] - 31.0).abs() < 1e-5);
        // cnt_h = 1080 / 64 = 16 => end_v = start_v + 17.0
        assert!((r0[3] - r0[1] - 17.0).abs() < 1e-5);

        // Smaller dt within remaining period preserves exact integer shift
        let r1 = st.step_uv_rect(0.02, 64, 64, 1920, 1080, 1.0, 20.0);
        assert_eq!(r1, r0);
        assert!((r0[0] * 64.0 - r0[0].mul_add(64.0, 0.0)).abs() < 1e-6);

        // Large dt skips safely in O(1) without infinite loop
        let r2 = st.step_uv_rect(10.0, 64, 64, 1920, 1080, 2.0, 30.0);
        assert!((r2[2] - r2[0] - (1920 / 128) as f32 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bloom_gauss_weights_adapt_and_tap_geometry() {
        let (w0, w1) = calc_gauss_k7(3.3, 1.0);
        let sum = w1[3] + 2.0 * (w0[0] + w0[1] + w0[2] + w0[3] + w1[0] + w1[1] + w1[2]);
        assert!((sum - 1.0).abs() < 1e-5, "k7 normalized sum={sum}");
        assert!(w1[3] > w0[0] && w0[0] > w0[3] && w0[3] > w1[2]);

        let (wave0, wave1) = calc_gauss_wave(3.3, 1.0, 1.0);
        let wave_sum = wave1[3]
            + 2.0
                * (wave0[0]
                    + wave0[1]
                    + wave0[2]
                    + wave0[3]
                    + wave1[0]
                    + wave1[1]
                    + wave1[2]);
        assert!((wave_sum - 2.0).abs() < 1e-5, "wave sum={wave_sum}");

        let cfg = BloomConfig::default();
        let hw = BloomFilterGpuWeights::horizontal_wave(&cfg);
        let vw = BloomFilterGpuWeights::vertical_wave(&cfg, 1920, 1080);
        assert!(vw.weight1[3] > hw.weight1[3]);

        let mut adapt = BloomAdaptState::default();
        let f1 = adapt.step(0.05, 100.0);
        assert!((f1 - (0.9 * 1.0 + 0.1 * 100.0 * 0.05)).abs() < 1e-5);
        assert_eq!(adapt.b_params(0.8), [0.8, 0.8, 0.8, f1]);

        let (a, b) = bloom_build_4tap_uvs(1280, 720, 256, 256);
        assert!((b[0][0] - (1.0 + a[0][0])).abs() < 1e-6);

        let taps_h = bloom_filter_taps_horizontal(256, 256);
        assert!(taps_h[7][3] > taps_h[1][3] && taps_h[7][0] < taps_h[1][0]);
        let taps_v = bloom_filter_taps_vertical(256, 256);
        assert!(taps_v[7][2] > taps_v[1][2] && taps_v[7][1] < taps_v[1][1]);
    }

    #[test]
    fn tonemap_vignette_and_gamma_luts() {
        let mut tm = TonemapParams::default();
        let hdr_in = [2.0, 1.0, 0.25];
        assert_eq!(tm.evaluate_rgb(hdr_in), hdr_in);
        tm.enabled = true;
        let mapped = tm.evaluate_rgb(hdr_in);
        assert!(mapped[0] > mapped[1] && mapped[1] > mapped[2]);
        assert!(mapped.iter().all(|&c| (0.0..=1.0).contains(&c)));
        let tm_gpu = tm.to_gpu_params();
        assert_eq!(tm_gpu.abcd, [tm.a, tm.b, tm.c, tm.d]);
        assert_eq!(tm_gpu.gamma_onoff[1], 1.0);

        let img_cfg = ImageCorrectionConfig {
            exposure: 1.25,
            gamma: 1.1,
            saturation: 0.9,
            color_grading: [0.1, -0.05, 0.2],
            vignette_control: [0.8, 0.2, 0.0, 0.0],
        };
        let img_gpu = img_cfg.to_gpu_params();
        assert_eq!(img_gpu.img_corrections, [1.25, 1.1, 0.9, 1.0]);
        assert_eq!(img_gpu.img_cg, [0.1, -0.05, 0.2, 1.0]);
        assert_eq!(img_gpu.vignette_control, [0.8, 0.2, 0.0, 0.0]);

        let vig_center = apply_vignette_rgb([1.0, 0.5, 0.25], [0.5, 0.5], [0.8, 0.2, 0.0, 0.0]);
        assert!((vig_center[0] - 0.8).abs() < 1e-5);
        let vig_corner = apply_vignette_rgb([1.0, 0.5, 0.25], [0.0, 0.0], [0.8, 0.2, 0.0, 0.0]);
        assert!((vig_corner[0] - 0.2).abs() < 1e-5);

        let gamma_cfg = GammaControlConfig::default();
        let curve = gamma_cfg.gen_lut_f32(&[0.0, 0.5, 1.0], 0.0, 1.0);
        assert!((curve[0][0] - 0.0).abs() < 1e-4);
        assert!((curve[1][1] - 0.5).abs() < 1e-4);
        assert!((curve[2][2] - 1.0).abs() < 1e-4);

        let ramp = gamma_cfg.gen_ramp_u16();
        assert_eq!(ramp[0], [0, 0, 0]);
        assert_eq!(ramp[255], [65535, 65535, 65535]);
        assert!(ramp[128][0] > 32500 && ramp[128][0] < 33200);
    }

    #[test]
    fn hdr10_colorspace_roundtrip_pq_and_sun_dawn_dusk() {
        assert_eq!(Hdr10Colorspace::from_u32(0), Hdr10Colorspace::Rec709);
        assert_eq!(Hdr10Colorspace::from_u32(99), Hdr10Colorspace::Rec2020);
        assert_eq!(Hdr10Tonemapper::from_u32(6), Hdr10Tonemapper::Uncharted2);
        assert_eq!(Hdr10TonemapMode::from_u32(0), Hdr10TonemapMode::Luminance);

        let rec709 = [0.6, 0.3, 0.1];
        let rec2020 = Hdr10Colorspace::Rec709.convert(Hdr10Colorspace::Rec2020, rec709);
        let back = Hdr10Colorspace::Rec2020.convert(Hdr10Colorspace::Rec709, rec2020);
        for k in 0..3 {
            assert!((back[k] - rec709[k]).abs() < 1e-4);
        }

        let pq = apply_st2084_pq_rgb([0.0, 0.5, 1.0]);
        assert!((pq[0] - 0.0).abs() < 1e-5);
        assert!(pq[1] > 0.8 && pq[1] < 0.99);
        assert!((pq[2] - 1.0).abs() < 1e-4);

        let cfg = Hdr10Config {
            enabled: true,
            tonemapper: Hdr10Tonemapper::Uncharted2,
            ..Hdr10Config::default()
        };
        let gpu = cfg.to_gpu_params();
        assert_eq!(gpu.parameters1[2], 1.0);
        assert_eq!(gpu.parameters2[2], 64.0);
        assert_eq!(gpu.parameters10[0..3], [1.0, 0.7, 1.0]);

        assert_eq!(effects_sun_hdr_blend_factor(2.0, 4.5, 6.0, 18.5, 21.0), 0.0);
        let dawn = effects_sun_hdr_blend_factor(5.25, 4.5, 6.0, 18.5, 21.0);
        assert!((dawn - 0.5).abs() < 1e-5);
        assert_eq!(effects_sun_hdr_blend_factor(12.0, 4.5, 6.0, 18.5, 21.0), 1.0);
        let dusk = effects_sun_hdr_blend_factor(19.75, 4.5, 6.0, 18.5, 21.0);
        assert!((dusk - 0.5).abs() < 1e-5);
    }

    #[test]
    fn hdr10_kawase_schedules_alternate_ping_pong_slots() {
        let cfg = Hdr10Config {
            bloom_blur_passes: 2,
            flare_blur_passes: 1,
            ..Hdr10Config::default()
        };
        let bloom_sched = cfg.bloom_blur_schedule(1920, 1080);
        assert_eq!(bloom_sched.len(), 5);
        assert_eq!(bloom_sched[0].kind, Hdr10PassKind::Downsample);
        assert_eq!(bloom_sched[0].src_halfres_slot, None);
        assert_eq!(bloom_sched[0].dst_halfres_slot, 1);
        assert_eq!(bloom_sched[1].kind, Hdr10PassKind::BlurVertical);
        assert_eq!(bloom_sched[1].src_halfres_slot, Some(1));
        assert_eq!(bloom_sched[1].dst_halfres_slot, 0);
        assert_eq!(bloom_sched[1].sparams[2], 0.0);
        assert_eq!(bloom_sched[2].kind, Hdr10PassKind::BlurHorizontal);
        assert_eq!(bloom_sched[2].src_halfres_slot, Some(0));
        assert_eq!(bloom_sched[2].dst_halfres_slot, 1);
        assert_eq!(bloom_sched[2].sparams[2], 1.0);

        let flare_sched = cfg.flare_blur_schedule(1280, 720);
        assert_eq!(flare_sched.len(), 4);
        assert_eq!(flare_sched[0].kind, Hdr10PassKind::Downsample);
        assert_eq!(flare_sched[1].kind, Hdr10PassKind::FeatureGen);
        assert_eq!(flare_sched[1].src_halfres_slot, Some(1));
        assert_eq!(flare_sched[1].dst_halfres_slot, 0);
        assert_eq!(flare_sched[2].kind, Hdr10PassKind::BlurVertical);
        assert_eq!(flare_sched[2].src_halfres_slot, Some(0));
        assert_eq!(flare_sched[2].dst_halfres_slot, 1);
    }

    #[test]
    fn dof_gasmask_nvg_heatvision_and_ssfx_configs() {
        let dof = DofConfig::default().to_gpu_params(1920, 1080);
        assert_eq!(dof.dof_params, [-1.25, 0.0, 600.0, 30.0]);
        assert!((dof.dof_kernel[0] - (0.5 / 1920.0 * 0.25)).abs() < 1e-7);

        let mut gm = GasmaskConfig::default();
        assert!(!gm.is_mask_active());
        gm.mask_control[0] = 1.0;
        gm.drops_control[0] = 0.75;
        assert!(gm.is_mask_active() && gm.are_drops_active());
        assert_eq!(gm.to_gpu_params().drops_control, [0.75, 1.15, 0.0, 0.0]);

        assert!(NightVisionGen::Off.profile().is_none());
        let gen3 = NightVisionGen::Gen3.profile().unwrap();
        assert_eq!(gen3.max_vision_distance, 600.0);
        assert_eq!(NightVisionGen::Gen1Centered.lens_mask([0.5, 0.5], 1920, 1080), 1.0);
        assert_eq!(NightVisionGen::Gen1Centered.lens_mask([0.0, 0.0], 1920, 1080), 0.0);
        assert_eq!(NightVisionGen::Gen2.lens_mask([0.25, 0.5], 1000, 1000), 1.0);

        let hv = HeatVisionConfig {
            enabled: true,
            mode: 1.0,
            ..HeatVisionConfig::default()
        }
        .to_gpu_params();
        assert_eq!(hv.params1, [1.0, 0.45, 0.65, 0.76]);
        assert_eq!(hv.params2[3], 1.0);

        let taa = SsfxTaaConfig::default();
        let j0 = taa.jitter_for_frame(0, 1000, 1000, true);
        assert_eq!(j0, [0.0, -0.0005, 1.0, 0.0]);
        let j_off = taa.jitter_for_frame(1, 1000, 1000, false);
        assert_eq!(j_off, [0.0, 0.0, 1.0, 0.0]);

        let ssfx = SsfxScreenConfig::default();
        assert_eq!(ssfx.shadow_cascades, [20.0, 40.0, 160.0]);
        assert_eq!(ssfx.fog_scattering, 0.6);
    }

    #[test]
    fn ssfx_sss_ext_slots_and_blur_schedules() {
        let mut tracker = SsfxSssExtSlots::new();
        let cam = [0.0, 0.0, 0.0];
        let view = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let candidates = vec![
            SsfxSssCandidate {
                id: 1,
                position: [2.0, 0.0, 0.0],
                range: 10.0,
                distance: 2.0,
                active: true,
                is_secondary_omnipart: false,
            },
            SsfxSssCandidate {
                id: 2,
                position: [0.0, 0.0, 15.0],
                range: 10.0,
                distance: 15.0,
                active: true,
                is_secondary_omnipart: false,
            },
            // Ignored: secondary omnipart or range <= 1.5
            SsfxSssCandidate {
                id: 3,
                position: [1.0, 0.0, 0.0],
                range: 10.0,
                distance: 1.0,
                active: true,
                is_secondary_omnipart: true,
            },
            SsfxSssCandidate {
                id: 4,
                position: [1.0, 0.0, 0.0],
                range: 1.0,
                distance: 1.0,
                active: true,
                is_secondary_omnipart: false,
            },
        ];

        let data0 = tracker.update(1, cam, &view, &candidates);
        let assigned = tracker.assigned_ids();
        assert!(assigned.contains(&Some(1)));
        assert!(assigned.contains(&Some(2)));
        assert!(!assigned.contains(&Some(3)));
        assert!(!assigned.contains(&Some(4)));
        let slot1 = assigned.iter().position(|&id| id == Some(1)).unwrap();
        assert_eq!(data0[slot1], [2.0, 0.0, 0.0, 0.0]);

        // Move camera past `2.0 * range` of light 1 -> evicted immediately even on skipped frame 2
        let data1 = tracker.update(2, [25.0, 0.0, 0.0], &view, &candidates);
        assert!(tracker.assigned_ids()[slot1].is_none());
        assert_eq!(data1[slot1], [0.0; 4]);

        let ao = ssfx_ao_blur_schedule(1920, 1080, 2.0);
        assert_eq!(ao[0].setup, [2.0, 0.25, 960.0, 540.0]);
        assert_eq!(ao[3].setup, [1.0, 1.0, 960.0, 540.0]);

        let il = ssfx_il_blur_schedule(1920, 1080, 4.0);
        assert_eq!(il[1].viewport_w, 480.0);
        assert_eq!(il[1].setup, [1.0, 0.5, 480.0, 270.0]);

        let vol = ssfx_volumetric_blur_schedule(1600, 800);
        assert_eq!(vol[0].setup, [200.0, 100.0, 0.0, 2.0]);
        assert_eq!(vol[3].setup, [200.0, 100.0, 2.0, 0.5]);

        let mip_blur = phase_blur_schedule(1920, 1080);
        assert_eq!(mip_blur[4].setup, [1.0, 0.0, 240.0, 135.0]);
        assert_eq!(mip_blur[5].setup, [0.0, 1.0, 240.0, 135.0]);

        let lens = ssfx_bloom_lens_blur_schedule(1280, 720);
        assert_eq!(lens[0].setup, [320.0, 180.0, 0.0, 2.0]);
        assert_eq!(lens[1].setup, [320.0, 180.0, 0.0, 5.0]);

        let lighting_cfg = SsfxLightingConfig::default();
        assert_eq!(lighting_cfg.il_quality, 32);
    }

    #[test]
    fn luminance_sunshafts_and_combine_constants() {
        let (p0_a, p0_b) = luminance_pass0_uvs(BLOOM_SIZE_X, BLOOM_SIZE_Y);
        assert!((p0_a[0][0] - 1.0 / 256.0).abs() < 1e-6);
        assert!((p0_a[3][0] - 3.0 / 256.0).abs() < 1e-6);
        assert!((p0_b[0][1] - (1.0 + 1.0 / 256.0)).abs() < 1e-6);

        let quad = luminance_filter_quad_uvs(64);
        assert!((quad[0][0][0] - 1.0 / 64.0).abs() < 1e-6);
        assert!((quad[0][0][1] - (1.0 + 1.0 / 64.0)).abs() < 1e-6);
        assert!((quad[3][7][3] - (1.0 + 7.0 / 64.0)).abs() < 1e-6);

        let mut lum = LuminanceAdaptState::default();
        assert_eq!(lum.f_luminance_adapt, 0.5);
        let mg = lum.step_middle_gray(0.1, &LuminanceConfig::default());
        assert!((mg[0] - 1.0).abs() < 1e-6);
        assert!((mg[1] - 0.7).abs() < 1e-6);
        assert!((mg[2] - (0.3 + 0.7 * 0.4)).abs() < 1e-5);
        assert!((mg[3] - (0.9 * 0.5 + 0.01)).abs() < 1e-6);

        assert!(sunshafts_schedule(0.0005, &SunshaftsConfig::default()).is_none());
        let sched = sunshafts_schedule(0.4, &SunshaftsConfig::default()).unwrap();
        assert_eq!(sched[0].dst_slot, Some(0));
        assert!(sched[0].c_sunshafts.is_none());
        assert_eq!(sched[1].c_sunshafts, Some([0.8, 1.0, 1.0, 1.0]));
        assert_eq!(sched[2].c_sunshafts, Some([0.8, 1.0, 0.7, 1.0]));
        assert_eq!(sched[3].c_sunshafts, Some([0.8, 1.0, 0.3, 1.0]));
        assert_eq!(sched[4].dst_slot, None);
        assert_eq!(sched[4].c_sunshafts, Some([0.8, 1.0, 0.0, 1.0]));

        let (ss_p0, ss_p1) = sunshafts_uv_bounds(1920, 1080);
        assert!((ss_p0[0] - 0.5 / 1920.0).abs() < 1e-7);
        assert!((ss_p1[1] - 1080.5 / 1080.0).abs() < 1e-6);

        let desc = crate::weather::EnvDescriptor {
            ambient_color: [0.1, 0.2, 0.3],
            hemi_color: [0.4, 0.5, 0.6, 1.0],
            fog_color: [0.7, 0.8, 0.9],
            ..crate::weather::EnvDescriptor::default()
        };
        let mdf = crate::shaders_xr::EnvModAccum {
            total_power: 0.0,
            active_flags: 0,
            far_plane: 0.0,
            fog_color: [0.0; 3],
            fog_density: 0.0,
            ambient: [0.0; 3],
            sky_color: [0.0; 3],
            hemi_color: [0.0; 3],
        };
        let env = crate::weather::lerp_descriptors(
            &desc,
            &desc,
            0.25,
            &mdf,
            1.0,
            &crate::weather::EnvBoost::default(),
            0.0,
            0.0,
        );
        let c1 = compute_combine1_params(
            &env,
            [1.0, 0.8, 0.6],
            [0.0, -1.0, 0.0],
            &XRAY_IDENTITY_MATRIX,
            1920,
            1080,
            &Combine1Config::default(),
        );
        assert!((c1.l_ambient[0] - 0.2).abs() < 1e-5);
        assert!((c1.env_color[0] - (0.8 + 1e-5) * 2.0).abs() < 1e-5);
        assert_eq!(c1.env_color[3], 0.25);
        assert_eq!(c1.fog_color, [0.7, 0.8, 0.9, 0.0]);
        assert!((c1.ssao_and_jitter[0] - 2.0).abs() < 1e-5);
        assert!((c1.ssao_and_jitter[1] - 150.0).abs() < 1e-4);
        assert!((c1.ssao_and_jitter[2] - 30.0).abs() < 1e-5);

        assert_eq!(combine2_shader_element(true, true), 3);
        assert_eq!(combine2_shader_element(true, false), 1);
        assert_eq!(combine2_shader_element(false, true), 4);
        assert_eq!(combine2_shader_element(false, false), 2);
        let aa_uvs = combine2_aa_quad_uvs(1000, 500);
        assert!((aa_uvs[1].uv1[0] - (0.5 / 1000.0 - 1.0 / 1000.0)).abs() < 1e-7);

        let mut mstate = Combine2MotionState::new();
        let proj = crate::camera::matrix_build_projection(
            std::f32::consts::FRAC_PI_2,
            1.0,
            0.2,
            100.0,
        );
        let view = crate::camera::matrix_build_camera([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [0.0, 1.0, 0.0]);
        let full = matrix_mul_44(&proj, &view);
        let inv_view = crate::ogf::matrix_invert_43(&view).unwrap();
        let c2 = mstate.step_and_gpu_params(
            [1.0, 2.0, 3.0],
            &inv_view,
            &proj,
            &full,
            false,
            &Combine2Config {
                m_blur: 0.6,
                ..Combine2Config::default()
            },
        );
        assert!((c2.m_blur[0] - 0.025).abs() < 1e-6);
        assert!((c2.m_blur[1] + 0.025).abs() < 1e-6);
        for r in 0..4 {
            for c in 0..4 {
                assert!((c2.m_previous[r][c] - c2.m_current[r][c]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn ssao_hdao_smaa_and_ssfx_taa_schedules() {
        let ssao_half = compute_ssao_params(1920, 1080, 67.5, true);
        assert!((ssao_half.noise_tile_factor - 2.0).abs() < 1e-5);
        assert!((ssao_half.kernel_size - 150.0).abs() < 1e-4);
        assert_eq!(ssao_half.half_width, 960.0);
        assert_eq!(ssao_half.half_height, 540.0);
        assert_eq!(
            ssao_half.resolution,
            [960.0, 540.0, 1.0 / 960.0, 1.0 / 540.0]
        );
        assert_eq!(ssao_half.downsamp_size, [960, 540]);
        assert!((ssao_half.jitter_scale[0] - 15.0).abs() < 1e-5);

        let ssao_full = compute_ssao_params(1920, 1080, 67.5, false);
        assert_eq!(ssao_full.downsamp_size, [1920, 1080]);
        assert!((ssao_full.downsamp_jitter_scale[0] - 30.0).abs() < 1e-5);

        assert_eq!(HDAO_GROUP_TEXEL_STEP, 32);
        let hdao = compute_hdao_dispatch(1920, 1080);
        assert_eq!(
            hdao,
            HdaoDispatch {
                groups_x: 60,
                groups_y: 34,
                groups_z: 1,
            }
        );

        let smaa_no_msaa = smaa_schedule(false);
        assert_eq!(smaa_no_msaa[0].kind, SmaaPassKind::EdgeDetection);
        assert!(smaa_no_msaa[0].clear_rt && smaa_no_msaa[0].stencil_enable);
        assert!(!smaa_no_msaa[0].stencil_cmp_equal);
        assert!(smaa_no_msaa[1].stencil_cmp_equal);
        assert_eq!(smaa_no_msaa[2].dst_name, "rt_Color");
        assert_eq!(smaa_no_msaa[2].copy_to_name, Some("rt_Generic_0"));
        assert_eq!(smaa_schedule(true)[2].dst_name, "rt_Generic");

        let taa_no_msaa = ssfx_taa_schedule(false);
        assert_eq!(taa_no_msaa[0].kind, SsfxTaaPassKind::PrepareMotionVectors);
        assert_eq!(taa_no_msaa[0].dst_name, "rt_ssfx_accum");
        assert_eq!(taa_no_msaa[0].copy_to_name, Some("rt_ssfx_taa"));
        assert_eq!(taa_no_msaa[1].dst_name, "rt_Color");
        assert_eq!(taa_no_msaa[1].copy_to_name, Some("rt_ssfx_prev_frame"));
        assert_eq!(taa_no_msaa[2].kind, SsfxTaaPassKind::Sharpen);
        assert_eq!(taa_no_msaa[2].dst_name, "rt_Generic_0");
        assert_eq!(ssfx_taa_schedule(true)[1].dst_name, "rt_Generic");
    }

    #[test]
    fn postprocess_pass_plan_lut_and_occlusion_pool() {
        let mut noise = PostprocessNoiseState::new(7);
        let idle = PostprocessState::identity();
        let idle_plan = build_postprocess_pass_plan(
            &idle,
            &mut noise,
            0.05,
            64,
            64,
            1920,
            1080,
            1920,
            1080,
            1.0,
        );
        assert!(!idle_plan.active);
        assert!(!idle_plan.uses_color_map);
        assert_eq!(idle_plan.shader_element, 0);
        assert_eq!(idle_plan.noise_uv, [0.0; 4]);

        let active_cm = PostprocessState {
            cm_influence: 0.75,
            cm_interpolate: 0.25,
            noise_intensity: 0.4,
            noise_fps: 24.0,
            duality_h: 0.02,
            ..PostprocessState::identity()
        };
        let cm_plan = build_postprocess_pass_plan(
            &active_cm,
            &mut noise,
            0.05,
            64,
            64,
            1920,
            1080,
            1920,
            1080,
            1.0,
        );
        assert!(cm_plan.active && cm_plan.uses_color_map);
        assert_eq!(cm_plan.shader_element, 4);
        assert!(cm_plan.noise_uv[2] > cm_plan.noise_uv[0]);
        assert!(cm_plan.right_uv[2] < cm_plan.left_uv[2]);

        let lut_std = lut_pass_desc(false);
        assert_eq!(lut_std.dst_name, "rt_Color");
        assert_eq!(lut_std.copy_to_name, "rt_Generic_0");
        assert_eq!(lut_std.shader_name, LUT_SHADER_NAME);
        assert_eq!(lut_std.noise_texture, LUT_NOISE_TEXTURE);
        assert_eq!(lut_std.lut_atlas_texture, LUT_ATLAS_TEXTURE);
        assert_eq!(lut_pass_desc(true).dst_name, "rt_Generic");
        assert_eq!(SsfxLutConfig::default().setup, [0.0; 4]);

        assert_eq!(OCCQ_DEFAULT_LIMIT, 1536);
        let mut pool = OcclusionQueryPool::new(3, true);
        let q0 = pool.begin().unwrap();
        let q1 = pool.begin().unwrap();
        let q2 = pool.begin().unwrap();
        assert_eq!((q0.order, q1.order, q2.order), (0, 1, 2));
        assert!(pool.begin().is_none());
        assert_eq!(pool.in_flight_count(), 3);
        assert_eq!(pool.slot_for_handle(q1.handle), Some(q1.slot));

        // Release higher-order q1 before q0: pool keeps descending order so q0 pops first next.
        assert_eq!(pool.complete(q1.handle), Some(q1.slot));
        assert_eq!(pool.complete(q0.handle), Some(q0.slot));
        assert!(pool.complete(q0.handle).is_none());
        assert!(pool.complete(OCCQ_INVALID_HANDLE).is_none());

        let next = pool.begin().unwrap();
        assert_eq!(next.order, 0);
        assert_eq!(pool.begin().unwrap().order, 1);

        pool.set_enabled(false);
        assert!(pool.begin().is_none());

        // Screen-space TL & TL2uv quad builders
        let mm = build_create_minmax_sm_quad();
        let msaa_e = build_mark_msaa_edges_quad();
        assert_eq!(mm, msaa_e);
        assert_eq!(mm[0].p[2], 0.0);
        assert_eq!(mm[1].p[2], EPS_S);

        let opt_q = build_stencil_optimize_quad(1920, 1080);
        assert_eq!(opt_q[0].p, [-0.5, 1079.5, 0.0, 1.0]);
        assert_eq!(opt_q[3].p, [1919.5, -0.5, 0.0, 1.0]);

        let icons_q = build_ui_3d_icons_postprocess_quad(1280, 720);
        assert_eq!(icons_q[0].color, 0xFFFF_FFFF);
        let rain_bg = build_phase_ssfx_rain_quad(1280, 720);
        assert_eq!(rain_bg[0].color, 0xFF00_0000);

        let rain_mask = build_draw_rain_mask_quad(100, 50);
        assert!((rain_mask[0].uv[0] - 0.005).abs() < 1e-6);
        assert!((rain_mask[2].uv[1] - 1.01).abs() < 1e-6);

        let rain_lit = build_draw_rain_light_quad(
            1280,
            50.0,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &XRAY_IDENTITY_MATRIX,
        );
        assert!((rain_lit[0].p[2] - 50.0).abs() < 1e-5);
        assert_eq!(rain_lit[0].uv1[1], 20.0);
    }

    #[test]
    fn pp_bloom_combine_quads_rt_catalog_and_phase_specs() {
        assert_eq!(std::mem::size_of::<PhasePpVertex>(), 48);
        assert_eq!(std::mem::size_of::<BloomBuildVertex>(), 48);
        assert_eq!(std::mem::size_of::<BloomFilterVertex>(), 144);
        assert_eq!(std::mem::size_of::<Combine2AaVertex>(), 88);

        let mut noise = PostprocessNoiseState::new(3);
        let state = PostprocessState {
            gray: 0.5,
            noise_intensity: 0.2,
            cm_influence: 0.4,
            ..PostprocessState::identity()
        };
        let plan = build_postprocess_pass_plan(&state, &mut noise, 0.01, 64, 64, 1920, 1080, 1920, 1080, 1.0);
        let pp_quad = plan.build_quad(1920, 1080, 0.25, -0.5);
        assert_eq!(pp_quad[0].p, [0.25, 1079.5, EPS_S, 1.0]);
        assert_eq!(pp_quad[3].p, [1920.25, -0.5, EPS_S, 1.0]);
        assert_eq!(pp_quad[0].color0_argb, plan.gpu_params.color0_argb);
        assert_eq!(pp_quad[0].uv0, [plan.right_uv[0], plan.right_uv[3]]);

        let bloom_q = build_bloom_build_quad(1920, 1080, BLOOM_SIZE_X, BLOOM_SIZE_Y);
        assert_eq!(bloom_q[2].p, [256.0 + EPS_QUAD, 256.0 + EPS_QUAD, EPS_QUAD, 1.0]);
        assert!((bloom_q[0].uv0[1] - (1.0 + 0.5 / 1080.0)).abs() < 1e-6);

        let fast_q = build_bloom_fast_filter_quad(256, 256, 2.0);
        assert!((fast_q[1].uv0[0] - (0.5 / 256.0 - 2.0 / 256.0)).abs() < 1e-6);

        let h_q = build_bloom_filter_quad_horizontal(256, 256);
        let v_q = build_bloom_filter_quad_vertical(256, 256);
        assert_eq!(h_q[0].p[1], 256.0 + EPS_QUAD);
        assert!((v_q[1].uv0[0] - 0.5 / 256.0).abs() < 1e-6);

        let ssfx_q = build_ssfx_bloom_quad(1920, 1080);
        assert_eq!(ssfx_q[0].color, 0);
        assert_eq!(ssfx_q[2].uv, [1.0, 1.0]);

        let ssfx_no_lens = ssfx_bloom_schedule(1920, 1080, 0.0, 2.0);
        assert_eq!(ssfx_no_lens.len(), 13);
        assert_eq!(ssfx_no_lens[0].kind, SsfxBloomPassKind::Build);
        assert_eq!(ssfx_no_lens.last().unwrap().kind, SsfxBloomPassKind::Combine);
        assert_eq!(ssfx_no_lens.last().unwrap().viewport_w, 960.0);

        let ssfx_with_lens = ssfx_bloom_schedule(1920, 1080, 1.0, 2.0);
        assert_eq!(ssfx_with_lens.len(), 16);
        assert_eq!(ssfx_with_lens[1].kind, SsfxBloomPassKind::LensFlares);
        assert_eq!(ssfx_with_lens[3].dst_name, "$user$ssfx_bloom_lens");
        assert_eq!(ssfx_with_lens[3].blur_setup, Some([480.0, 270.0, 0.0, 5.0]));

        let aa_q = build_combine2_aa_quad(1000, 500);
        assert_eq!(aa_q[0].p, [EPS_QUAD, 500.0 + EPS_QUAD, EPS_QUAD, 1.0]);
        assert!((aa_q[0].uv1[0] - (0.5 / 1000.0 - 1.0 / 1000.0)).abs() < 1e-7);

        let vol_q = build_combine_volumetric_quad();
        assert_eq!(vol_q[0].p, [-1.0, 1.0, 0.0, 1.0]);
        assert_eq!(vol_q[3].p, [1.0, -1.0, 1.0, 0.0]);

        let rt_cat = build_r4_render_target_catalog(&R4RenderTargetConfig {
            dx10_msaa: true,
            sample_count: 4,
            dx11_hdr10: true,
            hdr10_generic_r_on: true,
            ssao_hdao: true,
            ssao_ultra: true,
            dx10_minmax_sm: true,
            ..R4RenderTargetConfig::default()
        });
        assert!(rt_cat.iter().any(|r| r.name == "$user$msaadepth" && r.sample_count == 4));
        assert!(rt_cat.iter().any(|r| r.name == "$user$hdr10_halfres0" && r.width == 960));
        assert!(rt_cat.iter().any(|r| r.name == "$user$ssao_temp" && r.uav));
        assert!(rt_cat.iter().any(|r| r.name == "$user$smap_depth_minmax" && r.format == R4TargetFormat::R32F));

        let sb = r4_phase_spec(
            R4RenderPhaseKind::SceneBegin,
            R4PhaseStateOptions { dx10_msaa: true, albedo_wo: true, ..R4PhaseStateOptions::default() },
        );
        assert_eq!(sb.rts[1], R4RtSlot::Accumulator);
        assert_eq!(sb.depth, R4DepthSlot::MsaaDepth);
        assert_eq!(sb.stencil.unwrap().write_mask, 0x7F);

        let wm = r4_phase_spec(R4RenderPhaseKind::Wallmarks, R4PhaseStateOptions::default());
        assert_eq!(wm.color_write_enable, Some([true, true, true, false]));
        assert_eq!(wm.stencil.unwrap().func, CmpFunc::LessEqual);
    }
}
