//! Pure-Rust X-Ray Phase-13 Post-Processing, Bloom, Tonemapping, Gamma LUT & HDR10 Constants
//!
//! Grounds C++ renderer and HLSL shader parameters directly from:
//! - `src/Layers/xrRenderPC_R3/r3_rendertarget_phase_PP.cpp` (`u_calc_tc_noise`)
//! - `src/Layers/xrRenderPC_R3/r3_rendertarget_phase_bloom.cpp` (`CalcGauss_k7`, `CalcGauss_wave`, UV/tap builders)
//! - `src/Layers/xrRenderPC_R3/r3_rendertarget_phase_combine.cpp` (`tnmp_*`, `vignette_control`)
//! - `src/Layers/xrRender/xr_effgamma.cpp` (`CGammaControl::GenLUT`)
//! - `src/Layers/xrRender/Blender_Recorder_StandartBinding.cpp` (`hdr10_parameters1..10`)
//! - `gamedata/shaders/r3/{hdr10.h,combine_2_naa.ps,effects_sun.ps}`

use crate::particle_sim::ParticleRng;
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
}
