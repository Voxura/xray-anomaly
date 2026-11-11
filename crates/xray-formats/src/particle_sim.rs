//! Pure Rust PAPI Runtime Particle Simulator & GPU Billboard Emitter
//!
//! Faithful pure-Rust implementation matching `src/xrParticles/particle_core.cpp`,
//! `particle_actions_collection.cpp`, `particle_manager.cpp`, `noise.cpp`,
//! and `src/Layers/xrRender/ParticleEffect.cpp` / `ParticleEffectDef.cpp` / `ParticleGroup.cpp`:
//! - All 11 `ParticleDomain` generators (`Point`, `Line`, `Triangle`, `Plane`, `Box`,
//!   `Sphere`, `Cylinder`, `Cone`, `Blob`, `Disc`, `Rectangle`) & `transform`/`transform_dir`
//! - Classic `fractalsum3` / `turbulence3` 3D gradient Perlin noise (`noise.cpp`, MSVCRT LCG seed 1)
//! - Deterministic `ParticleEffectInstance` simulation across `CompiledParticleAction` (`0..=31`)
//!   with fixed sub-stepping (`update_step_ms`, max 3 steps/frame), `DF_TIME_LIMIT`, and animated sprites
//! - `ParticleGroupInstance` timeline scheduling (`time0..time1`, `on_play_child`,
//!   `on_birth_child`, `on_dead_child`, `FL_ON_PLAY_CHILD_REWIND`)
//! - GPU-ready `ParticleBillboardInstance` & `QuadIB`-ordered `ParticleQuadVertex` extraction

use std::f32::consts::{PI, TAU};

use crate::lanims::pack_rgba_f32;
use crate::ogf::{
    matrix_mul_43, matrix_set_hpb, matrix_transform_point, XRayMatrix, XRAY_IDENTITY_MATRIX,
};
use crate::particles::{
    papi_flags, ped_flags, pg_effect_flags, CompiledParticleAction, ParticleActionKind,
    ParticleDomain, ParticleDomainType, ParticleEffectDef, ParticleGroupDef, ParticlesLibrary,
};
use crate::skinning::{matrix_transform_dir, normalize_vec3};

const EPS_SMALL: f32 = 0.00001;
const P_MAXFLOAT: f32 = 1.0e16;
const STEP_DEFAULT: f32 = 0.033;
const ONE_OVER_SQRT_2PI: f32 = 0.398_942_3;

// ============================================================================
// Deterministic PRNG & PAPI Normal Distribution
// ============================================================================

/// Fast deterministic SplitMix64 / Xorshift64* PRNG for CPU particle simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticleRng {
    state: u64,
}

impl ParticleRng {
    pub fn new(seed: u64) -> Self {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        if s == 0 {
            s = 0xD1B5_4A32_D192_ED03;
        }
        Self { state: s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0.0, 1.0)`, matching `drand48()`.
    #[inline]
    pub fn unit_f32(&mut self) -> f32 {
        // 24 precision bits divided by 2^24 stays strictly < 1.0 in f32 except tiny rounding
        let bits = (self.next_u64() >> 40) as u32;
        let v = (bits as f32) * (1.0 / 16_777_216.0);
        if v >= 1.0 {
            0.999_999_94
        } else {
            v
        }
    }

    /// Positive `(0.0, 1.0]` uniform sample safe for `ln()`.
    #[inline]
    fn open_unit_f32(&mut self) -> f32 {
        self.unit_f32().max(1e-7)
    }

    #[inline]
    pub fn range_usize(&mut self, max_exclusive: usize) -> usize {
        if max_exclusive <= 1 {
            return 0;
        }
        ((self.next_u64() >> 16) as usize) % max_exclusive
    }

    /// Sample Gaussian variate matching `PAPI::NRand(sigma)`.
    pub fn normal(&mut self, sigma: f32) -> f32 {
        const ONE_OVER_SIGMA_EXP: f32 = 1.0 / 0.7975;
        if sigma == 0.0 {
            return 0.0;
        }
        loop {
            let y = -self.open_unit_f32().ln();
            let threshold = (-0.5 * (y - 1.0) * (y - 1.0)).exp();
            if self.unit_f32() <= threshold {
                let sign = if (self.next_u64() & 1) != 0 { 1.0 } else { -1.0 };
                return sign * y * sigma * ONE_OVER_SIGMA_EXP;
            }
        }
    }
}

// ============================================================================
// Pure-Rust Perlin Noise (`src/xrParticles/noise.cpp`)
// ============================================================================

const NOISE_B: usize = 256;

/// Classic 3D Perlin gradient noise matching `src/xrParticles/noise.cpp` (`srand(1)` MSVCRT LCG).
#[derive(Debug, Clone)]
pub struct PerlinNoise3 {
    perm: Vec<usize>,
    grad: Vec<[f32; 3]>,
}

impl Default for PerlinNoise3 {
    fn default() -> Self {
        Self::new()
    }
}

impl PerlinNoise3 {
    pub fn new() -> Self {
        let len = NOISE_B + NOISE_B + 2;
        let mut perm = vec![0usize; len];
        let mut grad = vec![[0.0f32; 3]; len];

        let mut lcg_state: u32 = 1;
        let mut msvcrt_rand = || -> i32 {
            lcg_state = lcg_state.wrapping_mul(214_013).wrapping_add(2_531_011);
            ((lcg_state >> 16) & 0x7FFF) as i32
        };

        for g_slot in grad.iter_mut().take(NOISE_B) {
            let mut v = [0.0f32; 3];
            let mut s;
            loop {
                for comp in &mut v {
                    let rnd = msvcrt_rand();
                    *comp = ((rnd % ((NOISE_B * 2) as i32)) - NOISE_B as i32) as f32 / NOISE_B as f32;
                }
                s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if s <= 1.0 && s > 1e-12 {
                    break;
                }
            }
            let inv_s = 1.0 / s.sqrt();
            *g_slot = [v[0] * inv_s, v[1] * inv_s, v[2] * inv_s];
        }

        for (i, slot) in perm.iter_mut().enumerate().take(NOISE_B) {
            *slot = i;
        }
        let mut i = NOISE_B;
        while i > 0 {
            let j = (msvcrt_rand() as usize) % NOISE_B;
            perm.swap(i, j);
            if i < 2 {
                break;
            }
            i -= 2;
        }

        for idx in 0..(NOISE_B + 2) {
            perm[NOISE_B + idx] = perm[idx];
            grad[NOISE_B + idx] = grad[idx];
        }

        Self { perm, grad }
    }

    pub fn noise3(&self, v: [f32; 3]) -> f32 {
        #[inline]
        fn setup(val: f32) -> (usize, usize, f32, f32) {
            let t = val + 10_000.0;
            let tt = t.floor() as i32;
            let b0 = (tt as usize) & (NOISE_B - 1);
            let b1 = (b0 + 1) & (NOISE_B - 1);
            let r0 = t - (tt as f32);
            let r1 = r0 - 1.0;
            (b0, b1, r0, r1)
        }
        #[inline]
        fn s_curve(t: f32) -> f32 {
            t * t * (3.0 - 2.0 * t)
        }
        #[inline]
        fn lerp(t: f32, a: f32, b: f32) -> f32 {
            a + t * (b - a)
        }
        #[inline]
        fn at(q: [f32; 3], rx: f32, ry: f32, rz: f32) -> f32 {
            rx * q[0] + ry * q[1] + rz * q[2]
        }

        let (bx0, bx1, rx0, rx1) = setup(v[0]);
        let (by0, by1, ry0, ry1) = setup(v[1]);
        let (bz0, bz1, rz0, rz1) = setup(v[2]);

        let i = self.perm[bx0];
        let j = self.perm[bx1];
        let b00 = self.perm[i + by0];
        let b10 = self.perm[j + by0];
        let b01 = self.perm[i + by1];
        let b11 = self.perm[j + by1];

        let sx = s_curve(rx0);
        let sy = s_curve(ry0);
        let sz = s_curve(rz0);

        let u0 = at(self.grad[b00 + bz0], rx0, ry0, rz0);
        let v0 = at(self.grad[b10 + bz0], rx1, ry0, rz0);
        let a0 = lerp(sx, u0, v0);

        let u1 = at(self.grad[b01 + bz0], rx0, ry1, rz0);
        let v1 = at(self.grad[b11 + bz0], rx1, ry1, rz0);
        let b0 = lerp(sx, u1, v1);
        let c = lerp(sy, a0, b0);

        let u2 = at(self.grad[b00 + bz1], rx0, ry0, rz1);
        let v2 = at(self.grad[b10 + bz1], rx1, ry0, rz1);
        let a1 = lerp(sx, u2, v2);

        let u3 = at(self.grad[b01 + bz1], rx0, ry1, rz1);
        let v3 = at(self.grad[b11 + bz1], rx1, ry1, rz1);
        let b1 = lerp(sx, u3, v3);
        let d = lerp(sy, a1, b1);

        1.5 * lerp(sz, c, d)
    }

    pub fn fractalsum3(&self, v: [f32; 3], mut freq: f32, octaves: i32) -> f32 {
        let boost = freq;
        let mut sum = 0.0f32;
        let count = octaves.clamp(1, 16);
        for _ in 0..count {
            let scaled = [v[0] * freq, v[1] * freq, v[2] * freq];
            if freq != 0.0 {
                sum += self.noise3(scaled) / freq;
            }
            freq *= 2.059;
        }
        sum * boost
    }

    pub fn turbulence3(&self, v: [f32; 3], mut freq: f32, octaves: i32) -> f32 {
        let boost = freq;
        let mut sum = 0.0f32;
        let count = octaves.clamp(1, 16);
        for _ in 0..count {
            let scaled = [v[0] * freq, v[1] * freq, v[2] * freq];
            if freq != 0.0 {
                sum += self.noise3(scaled).abs() / freq;
            }
            freq *= 2.059;
        }
        sum * boost
    }
}

// ============================================================================
// ParticleDomain Runtime Sampling & Transform Extensions
// ============================================================================

impl ParticleDomain {
    /// Sample a uniform/Gaussian point inside this domain matching `PAPI::pDomain::Generate`.
    pub fn sample(&self, rng: &mut ParticleRng) -> [f32; 3] {
        match self.domain_type {
            ParticleDomainType::Point | ParticleDomainType::Plane => self.p1,
            ParticleDomainType::Line => {
                let t = rng.unit_f32();
                [
                    self.p1[0] + self.p2[0] * t,
                    self.p1[1] + self.p2[1] * t,
                    self.p1[2] + self.p2[2] * t,
                ]
            }
            ParticleDomainType::Box => [
                self.p1[0] + (self.p2[0] - self.p1[0]) * rng.unit_f32(),
                self.p1[1] + (self.p2[1] - self.p1[1]) * rng.unit_f32(),
                self.p1[2] + (self.p2[2] - self.p1[2]) * rng.unit_f32(),
            ],
            ParticleDomainType::Triangle => {
                let r1 = rng.unit_f32();
                let r2 = rng.unit_f32();
                let (a, b) = if r1 + r2 < 1.0 {
                    (r1, r2)
                } else {
                    (1.0 - r1, 1.0 - r2)
                };
                [
                    self.p1[0] + self.u[0] * a + self.v[0] * b,
                    self.p1[1] + self.u[1] * a + self.v[1] * b,
                    self.p1[2] + self.u[2] * a + self.v[2] * b,
                ]
            }
            ParticleDomainType::Rectangle => {
                let a = rng.unit_f32();
                let b = rng.unit_f32();
                [
                    self.p1[0] + self.u[0] * a + self.v[0] * b,
                    self.p1[1] + self.u[1] * a + self.v[1] * b,
                    self.p1[2] + self.u[2] * a + self.v[2] * b,
                ]
            }
            ParticleDomainType::Sphere => {
                let dir = normalize_vec3([
                    rng.unit_f32() - 0.5,
                    rng.unit_f32() - 0.5,
                    rng.unit_f32() - 0.5,
                ]);
                let rad = if (self.radius1 - self.radius2).abs() <= f32::EPSILON {
                    self.radius1
                } else {
                    self.radius2 + rng.unit_f32() * (self.radius1 - self.radius2)
                };
                [
                    self.p1[0] + dir[0] * rad,
                    self.p1[1] + dir[1] * rad,
                    self.p1[2] + dir[2] * rad,
                ]
            }
            ParticleDomainType::Cylinder | ParticleDomainType::Cone => {
                let dist = rng.unit_f32();
                let theta = rng.unit_f32() * TAU;
                let r = self.radius2 + rng.unit_f32() * (self.radius1 - self.radius2);
                let (sin_t, cos_t) = theta.sin_cos();
                let mut x = r * cos_t;
                let mut y = r * sin_t;
                if self.domain_type == ParticleDomainType::Cone {
                    x *= dist;
                    y *= dist;
                }
                [
                    self.p1[0] + self.p2[0] * dist + self.u[0] * x + self.v[0] * y,
                    self.p1[1] + self.p2[1] * dist + self.u[1] * x + self.v[1] * y,
                    self.p1[2] + self.p2[2] * dist + self.u[2] * x + self.v[2] * y,
                ]
            }
            ParticleDomainType::Blob => [
                self.p1[0] + rng.normal(self.radius1),
                self.p1[1] + rng.normal(self.radius1),
                self.p1[2] + rng.normal(self.radius1),
            ],
            ParticleDomainType::Disc => {
                let theta = rng.unit_f32() * TAU;
                let r = self.radius2 + rng.unit_f32() * (self.radius1 - self.radius2);
                let (sin_t, cos_t) = theta.sin_cos();
                let x = r * cos_t;
                let y = r * sin_t;
                [
                    self.p1[0] + self.u[0] * x + self.v[0] * y,
                    self.p1[1] + self.u[1] * x + self.v[1] * y,
                    self.p1[2] + self.u[2] * x + self.v[2] * y,
                ]
            }
        }
    }

    /// Transform this domain into world/parent space matching `PAPI::pDomain::transform`.
    pub fn transform(&self, m: &XRayMatrix) -> Self {
        let mut out = *self;
        match self.domain_type {
            ParticleDomainType::Box => {
                let mut mn = [f32::INFINITY; 3];
                let mut mx = [f32::NEG_INFINITY; 3];
                for sx in [self.p1[0], self.p2[0]] {
                    for sy in [self.p1[1], self.p2[1]] {
                        for sz in [self.p1[2], self.p2[2]] {
                            let pt = matrix_transform_point(m, [sx, sy, sz]);
                            for k in 0..3 {
                                mn[k] = mn[k].min(pt[k]);
                                mx[k] = mx[k].max(pt[k]);
                            }
                        }
                    }
                }
                out.p1 = mn;
                out.p2 = mx;
            }
            ParticleDomainType::Plane => {
                out.p1 = matrix_transform_point(m, self.p1);
                out.p2 = matrix_transform_dir(m, self.p2);
                out.radius1 = -dot3(out.p1, out.p2);
            }
            ParticleDomainType::Sphere | ParticleDomainType::Blob | ParticleDomainType::Point => {
                out.p1 = matrix_transform_point(m, self.p1);
            }
            ParticleDomainType::Line => {
                out.p1 = matrix_transform_point(m, self.p1);
                out.p2 = matrix_transform_dir(m, self.p2);
            }
            ParticleDomainType::Cylinder
            | ParticleDomainType::Cone
            | ParticleDomainType::Triangle
            | ParticleDomainType::Rectangle
            | ParticleDomainType::Disc => {
                out.p1 = matrix_transform_point(m, self.p1);
                out.p2 = matrix_transform_dir(m, self.p2);
                out.u = matrix_transform_dir(m, self.u);
                out.v = matrix_transform_dir(m, self.v);
                if matches!(
                    self.domain_type,
                    ParticleDomainType::Triangle | ParticleDomainType::Rectangle
                ) {
                    out.radius1 = -dot3(out.p1, out.p2);
                } else if self.domain_type == ParticleDomainType::Disc {
                    out.radius1_sqr = -dot3(out.p1, out.p2);
                }
            }
        }
        out
    }

    /// Direction-only transform matching `PAPI::pDomain::transform_dir` (zeros translation).
    pub fn transform_dir(&self, m: &XRayMatrix) -> Self {
        let mut dir_m = *m;
        dir_m[3][0] = 0.0;
        dir_m[3][1] = 0.0;
        dir_m[3][2] = 0.0;
        self.transform(&dir_m)
    }
}

// ============================================================================
// Runtime Particle State & Effect Simulation
// ============================================================================

/// Live simulated particle record matching `PAPI::Particle`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleState {
    pub pos: [f32; 3],
    pub pos_b: [f32; 3],
    pub size: [f32; 3],
    pub rot: [f32; 3],
    pub vel: [f32; 3],
    /// Normalized `[r, g, b, a]` in `0.0..=1.0`.
    pub color_rgba: [f32; 4],
    pub age: f32,
    /// Fixed-point 8-bit sub-frame index (`frame * 255`).
    pub frame_q8: u16,
    pub animate_ccw: bool,
}

/// Snapshot emitted for particle group child triggers (`on_birth_child` / `on_dead_child`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleLifeEvent {
    pub particle: ParticleState,
    pub slot_index: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ParticleStepEvents {
    pub born: Vec<ParticleLifeEvent>,
    pub died: Vec<ParticleLifeEvent>,
}

/// GPU-ready particle billboard descriptor matching `ParticleRenderStream` in `ParticleEffect.cpp`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleBillboardInstance {
    pub center: [f32; 3],
    pub rotation_rad: f32,
    pub half_size: [f32; 2],
    pub right_axis: [f32; 3],
    pub up_axis: [f32; 3],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub color_rgba: [f32; 4],
}

/// Expanded quad vertex matching X-Ray `FVF::LIT` + `CreateQuadIB` (`BL, TL, BR, TR`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleQuadVertex {
    pub position: [f32; 3],
    pub color_argb: u32,
    pub uv: [f32; 2],
}

impl ParticleBillboardInstance {
    #[inline]
    pub fn color_argb_u32(&self) -> u32 {
        pack_rgba_f32(self.color_rgba)
    }

    /// Expand this billboard into 4 quad vertices and 6 `CreateQuadIB` indices (`[0, 1, 2, 3, 2, 1]`).
    pub fn expand_quad(&self, base_vertex: u16) -> ([ParticleQuadVertex; 4], [u16; 6]) {
        let (sa, ca) = self.rotation_rad.sin_cos();
        let t = self.up_axis;
        let r = self.right_axis;
        let r1 = self.half_size[0];
        let r2 = self.half_size[1];

        let vr = [
            (t[0] * sa + r[0] * ca) * r1,
            (t[1] * sa + r[1] * ca) * r1,
            (t[2] * sa + r[2] * ca) * r1,
        ];
        let vt = [
            (t[0] * ca - r[0] * sa) * r2,
            (t[1] * ca - r[1] * sa) * r2,
            (t[2] * ca - r[2] * sa) * r2,
        ];

        let a = sub3(vt, vr);
        let b = add3(vt, vr);
        let c = [-a[0], -a[1], -a[2]];
        let d = [-b[0], -b[1], -b[2]];
        let clr = self.color_argb_u32();

        let verts = [
            ParticleQuadVertex {
                position: add3(d, self.center),
                color_argb: clr,
                uv: [self.uv_min[0], self.uv_max[1]],
            },
            ParticleQuadVertex {
                position: add3(a, self.center),
                color_argb: clr,
                uv: [self.uv_min[0], self.uv_min[1]],
            },
            ParticleQuadVertex {
                position: add3(c, self.center),
                color_argb: clr,
                uv: [self.uv_max[0], self.uv_max[1]],
            },
            ParticleQuadVertex {
                position: add3(b, self.center),
                color_argb: clr,
                uv: [self.uv_max[0], self.uv_min[1]],
            },
        ];
        let indices = [
            base_vertex,
            base_vertex + 1,
            base_vertex + 2,
            base_vertex + 3,
            base_vertex + 2,
            base_vertex + 1,
        ];
        (verts, indices)
    }
}

#[derive(Debug, Clone)]
struct RuntimeAction {
    flags: u32,
    local: ParticleActionKind,
    world: ParticleActionKind,
    source_silent: bool,
    source_parent_vel: [f32; 3],
    explosion_age: f32,
    turbulence_age: f32,
    restore_time_left: f32,
}

impl RuntimeAction {
    fn from_compiled(act: &CompiledParticleAction) -> Self {
        let restore_time_left = match &act.kind {
            ParticleActionKind::Restore { time_left } => *time_left,
            _ => 0.0,
        };
        let source_silent = (act.flags & papi_flags::PASOURCE_SILENT) != 0;
        let source_parent_vel = match &act.kind {
            ParticleActionKind::Source { parent_vel, .. } => *parent_vel,
            _ => [0.0; 3],
        };
        Self {
            flags: act.flags,
            local: act.kind.clone(),
            world: act.kind.clone(),
            source_silent,
            source_parent_vel,
            explosion_age: 0.0,
            turbulence_age: 0.0,
            restore_time_left,
        }
    }

    fn transform(&mut self, full: &XRayMatrix, translation_only: &XRayMatrix, parent_vel_arg: [f32; 3]) {
        let m = if (self.flags & papi_flags::ALLOW_ROTATE) != 0 {
            full
        } else {
            translation_only
        };
        self.world = match &self.local {
            ParticleActionKind::Avoid {
                position,
                look_ahead,
                magnitude,
                epsilon,
            } => ParticleActionKind::Avoid {
                position: position.transform(m),
                look_ahead: *look_ahead,
                magnitude: *magnitude,
                epsilon: *epsilon,
            },
            ParticleActionKind::Bounce {
                position,
                one_minus_friction,
                resilience,
                cutoff_sqr,
            } => ParticleActionKind::Bounce {
                position: position.transform(m),
                one_minus_friction: *one_minus_friction,
                resilience: *resilience,
                cutoff_sqr: *cutoff_sqr,
            },
            ParticleActionKind::Explosion {
                center,
                velocity,
                magnitude,
                stdev,
                age,
                epsilon,
            } => ParticleActionKind::Explosion {
                center: matrix_transform_point(m, *center),
                velocity: *velocity,
                magnitude: *magnitude,
                stdev: *stdev,
                age: *age,
                epsilon: *epsilon,
            },
            ParticleActionKind::Jet {
                center,
                acc,
                magnitude,
                epsilon,
                max_radius,
            } => ParticleActionKind::Jet {
                center: matrix_transform_point(m, *center),
                acc: acc.transform_dir(m),
                magnitude: *magnitude,
                epsilon: *epsilon,
                max_radius: *max_radius,
            },
            ParticleActionKind::OrbitLine {
                p,
                axis,
                magnitude,
                epsilon,
                max_radius,
            } => ParticleActionKind::OrbitLine {
                p: matrix_transform_point(m, *p),
                axis: matrix_transform_dir(m, *axis),
                magnitude: *magnitude,
                epsilon: *epsilon,
                max_radius: *max_radius,
            },
            ParticleActionKind::OrbitPoint {
                center,
                magnitude,
                epsilon,
                max_radius,
            } => ParticleActionKind::OrbitPoint {
                center: matrix_transform_point(m, *center),
                magnitude: *magnitude,
                epsilon: *epsilon,
                max_radius: *max_radius,
            },
            ParticleActionKind::RandomAccel { gen_acc } => ParticleActionKind::RandomAccel {
                gen_acc: gen_acc.transform_dir(m),
            },
            ParticleActionKind::RandomDisplace { gen_disp } => ParticleActionKind::RandomDisplace {
                gen_disp: gen_disp.transform_dir(m),
            },
            ParticleActionKind::RandomVelocity { gen_vel } => ParticleActionKind::RandomVelocity {
                gen_vel: gen_vel.transform_dir(m),
            },
            ParticleActionKind::Scatter {
                center,
                magnitude,
                epsilon,
                max_radius,
            } => ParticleActionKind::Scatter {
                center: matrix_transform_point(m, *center),
                magnitude: *magnitude,
                epsilon: *epsilon,
                max_radius: *max_radius,
            },
            ParticleActionKind::Sink {
                kill_inside,
                position,
            } => ParticleActionKind::Sink {
                kill_inside: *kill_inside,
                position: position.transform(m),
            },
            ParticleActionKind::SinkVelocity {
                kill_inside,
                velocity,
            } => ParticleActionKind::SinkVelocity {
                kill_inside: *kill_inside,
                velocity: velocity.transform_dir(m),
            },
            ParticleActionKind::Source {
                position,
                velocity,
                rot,
                size,
                color,
                alpha,
                particle_rate,
                age,
                age_sigma,
                parent_vel: _,
                parent_motion,
            } => {
                self.source_parent_vel = [
                    parent_vel_arg[0] * *parent_motion,
                    parent_vel_arg[1] * *parent_motion,
                    parent_vel_arg[2] * *parent_motion,
                ];
                ParticleActionKind::Source {
                    position: position.transform(m),
                    velocity: velocity.transform_dir(m),
                    rot: *rot,
                    size: *size,
                    color: *color,
                    alpha: *alpha,
                    particle_rate: *particle_rate,
                    age: *age,
                    age_sigma: *age_sigma,
                    parent_vel: self.source_parent_vel,
                    parent_motion: *parent_motion,
                }
            }
            ParticleActionKind::TargetVelocity {
                did_variant,
                velocity,
                scale,
            } => ParticleActionKind::TargetVelocity {
                did_variant: *did_variant,
                velocity: matrix_transform_dir(m, *velocity),
                scale: *scale,
            },
            ParticleActionKind::Vortex {
                center,
                axis,
                magnitude,
                epsilon,
                max_radius,
            } => ParticleActionKind::Vortex {
                center: matrix_transform_point(m, *center),
                axis: matrix_transform_dir(m, *axis),
                magnitude: *magnitude,
                epsilon: *epsilon,
                max_radius: *max_radius,
            },
            // Gravity, Damping, Turbulence, etc. are not transformed in PAPI C++
            other => other.clone(),
        };
    }
}

/// Stateful CPU simulator for a single `ParticleEffectDef` (`CParticleEffect`).
#[derive(Debug, Clone)]
pub struct ParticleEffectInstance {
    pub def: ParticleEffectDef,
    actions: Vec<RuntimeAction>,
    particles: Vec<ParticleState>,
    playing: bool,
    deferred_stop: bool,
    elapsed_limit: f32,
    accum_dt_ms: u32,
    initial_position: [f32; 3],
    world_xform: XRayMatrix,
    use_world_xform: bool,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    rng: ParticleRng,
    noise: PerlinNoise3,
}

impl ParticleEffectInstance {
    pub fn new(def: ParticleEffectDef, seed: u64) -> Self {
        let actions = def
            .actions
            .iter()
            .map(RuntimeAction::from_compiled)
            .collect();
        let elapsed_limit = if (def.flags & ped_flags::DF_TIME_LIMIT) != 0 {
            def.time_limit.unwrap_or(0.0)
        } else {
            0.0
        };
        let cap = (def.max_particles as usize).min(16_384);
        Self {
            def,
            actions,
            particles: Vec::with_capacity(cap),
            playing: false,
            deferred_stop: false,
            elapsed_limit,
            accum_dt_ms: 0,
            initial_position: [0.0; 3],
            world_xform: XRAY_IDENTITY_MATRIX,
            use_world_xform: false,
            bounds_min: [-0.001; 3],
            bounds_max: [0.001; 3],
            rng: ParticleRng::new(seed),
            noise: PerlinNoise3::new(),
        }
    }

    pub fn play(&mut self) {
        self.deferred_stop = false;
        self.playing = true;
        if (self.def.flags & ped_flags::DF_TIME_LIMIT) != 0 {
            self.elapsed_limit = self.def.time_limit.unwrap_or(0.0);
        }
        for act in &mut self.actions {
            match &act.local {
                ParticleActionKind::Source { .. } => act.source_silent = false,
                ParticleActionKind::Explosion { .. } => act.explosion_age = 0.0,
                ParticleActionKind::Turbulence { .. } => act.turbulence_age = 0.0,
                ParticleActionKind::Restore { time_left } => act.restore_time_left = *time_left,
                _ => {}
            }
        }
    }

    pub fn stop(&mut self, deferred: bool) {
        for act in &mut self.actions {
            if matches!(act.local, ParticleActionKind::Source { .. }) {
                act.source_silent = true;
            }
        }
        if deferred {
            self.deferred_stop = true;
        } else {
            self.playing = false;
            self.deferred_stop = false;
            self.particles.clear();
        }
    }

    #[inline]
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    #[inline]
    pub fn is_looped(&self) -> bool {
        (self.def.flags & ped_flags::DF_TIME_LIMIT) == 0
    }

    #[inline]
    pub fn particles(&self) -> &[ParticleState] {
        &self.particles
    }

    #[inline]
    pub fn bounds_aabb(&self) -> ([f32; 3], [f32; 3]) {
        (self.bounds_min, self.bounds_max)
    }

    /// Update parent transform (`CParticleEffect::UpdateParent`).
    /// When `use_world_xform` is `false` (default in X-Ray), actions themselves are transformed
    /// into world space so particles simulate in world coordinates; when `true`, particles
    /// remain in local space and are transformed during billboard emission.
    pub fn update_parent(
        &mut self,
        xform: XRayMatrix,
        velocity: [f32; 3],
        use_world_xform: bool,
    ) {
        self.use_world_xform = use_world_xform;
        if use_world_xform {
            self.world_xform = xform;
        } else {
            self.initial_position = [xform[3][0], xform[3][1], xform[3][2]];
            let mut m_t = XRAY_IDENTITY_MATRIX;
            m_t[3][0] = xform[3][0];
            m_t[3][1] = xform[3][1];
            m_t[3][2] = xform[3][2];
            for act in &mut self.actions {
                act.transform(&xform, &m_t, velocity);
            }
        }
    }

    /// Advance simulation using `update_step_ms` accumulator (max 3 substeps/frame) matching `CParticleEffect::OnFrame`.
    pub fn step(&mut self, frame_dt_ms: u32) -> ParticleStepEvents {
        let mut events = ParticleStepEvents::default();
        if !self.playing {
            self.recompute_idle_bounds();
            return events;
        }

        self.accum_dt_ms = self.accum_dt_ms.saturating_add(frame_dt_ms);
        let u_step = self.def.update_step_ms.max(1);
        let f_step = (u_step as f32) / 1000.0;
        let step_count = (self.accum_dt_ms / u_step).min(3);
        self.accum_dt_ms %= u_step;

        for _ in 0..step_count {
            if !self.step_single(f_step, &mut events) {
                break;
            }
        }
        events
    }

    /// Execute one explicit `dt_sec` substep (ignoring `update_step_ms` accumulation).
    pub fn step_fixed_dt(&mut self, dt_sec: f32) -> ParticleStepEvents {
        let mut events = ParticleStepEvents::default();
        if !self.playing || dt_sec <= 0.0 {
            self.recompute_idle_bounds();
            return events;
        }
        self.step_single(dt_sec, &mut events);
        events
    }

    fn step_single(&mut self, dt: f32, events: &mut ParticleStepEvents) -> bool {
        if (self.def.flags & ped_flags::DF_TIME_LIMIT) != 0 && !self.deferred_stop {
            self.elapsed_limit -= dt;
            if self.elapsed_limit < 0.0 {
                self.elapsed_limit = self.def.time_limit.unwrap_or(0.0);
                self.stop(true);
                return false;
            }
        }

        let mut tm_max = 1.0f32;
        for idx in 0..self.actions.len() {
            execute_action(
                &mut self.actions[idx],
                &mut self.particles,
                &self.def,
                dt,
                &mut tm_max,
                &mut self.rng,
                &self.noise,
                events,
            );
        }

        // Exact C++ flag test: `m_Flags.is(dfFramed | dfAnimated)` requires BOTH bits
        let anim_mask = ped_flags::DF_FRAMED | ped_flags::DF_ANIMATED;
        if (self.def.flags & anim_mask) == anim_mask
            && let Some(frame_def) = self.def.frame
        {
            let speed_fac = frame_def.speed * dt;
            let frame_count_f = frame_def.frame_count.max(1) as f32;
            for p in &mut self.particles {
                let dir = if p.animate_ccw { -1.0 } else { 1.0 };
                let mut f = (p.frame_q8 as f32) / 255.0 + dir * speed_fac;
                if f > frame_count_f {
                    f -= frame_count_f;
                }
                if f < 0.0 {
                    f += frame_count_f;
                }
                p.frame_q8 = (f * 255.0).floor().clamp(0.0, u16::MAX as f32) as u16;
            }
        }

        self.recompute_bounds();
        if self.deferred_stop && self.particles.is_empty() {
            self.playing = false;
            self.deferred_stop = false;
            return false;
        }
        true
    }

    fn recompute_bounds(&mut self) {
        if self.particles.is_empty() {
            self.recompute_idle_bounds();
            return;
        }
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        let mut max_size = 0.0f32;
        for p in &self.particles {
            for k in 0..3 {
                mn[k] = mn[k].min(p.pos[k]);
                mx[k] = mx[k].max(p.pos[k]);
                max_size = max_size.max(p.size[k]);
            }
        }
        for k in 0..3 {
            mn[k] -= max_size;
            mx[k] += max_size;
        }
        self.bounds_min = mn;
        self.bounds_max = mx;
    }

    fn recompute_idle_bounds(&mut self) {
        self.bounds_min = [
            self.initial_position[0] - 0.001,
            self.initial_position[1] - 0.001,
            self.initial_position[2] - 0.001,
        ];
        self.bounds_max = [
            self.initial_position[0] + 0.001,
            self.initial_position[1] + 0.001,
            self.initial_position[2] + 0.001,
        ];
    }

    /// Generate GPU-ready billboards matching `ParticleRenderStream` (`ParticleEffect.cpp:450-570`).
    pub fn emit_billboards(
        &self,
        cam_right: [f32; 3],
        cam_up: [f32; 3],
        cam_dir: [f32; 3],
    ) -> Vec<ParticleBillboardInstance> {
        if self.particles.is_empty() || (self.def.flags & ped_flags::DF_SPRITE) == 0 {
            return Vec::new();
        }
        let cam_dir_norm = normalize_vec3(cam_dir);
        let framed = (self.def.flags & ped_flags::DF_FRAMED) != 0;
        let vel_scaled = (self.def.flags & ped_flags::DF_VELOCITY_SCALE) != 0;
        let vel_scale = self.def.velocity_scale.unwrap_or([0.0; 3]);
        let align_to_path = (self.def.flags & ped_flags::DF_ALIGN_TO_PATH) != 0;
        let world_align = (self.def.flags & ped_flags::DF_WORLD_ALIGN) != 0;
        let face_align = (self.def.flags & ped_flags::DF_FACE_ALIGN) != 0;
        let default_rot = self
            .def
            .align_to_path_rotation
            .unwrap_or([-PI * 0.5, 0.0, 0.0]);

        let mut out = Vec::with_capacity(self.particles.len());
        for p in &self.particles {
            let (uv_min, uv_max) = if framed && let Some(fr) = self.def.frame {
                let frame_idx = (p.frame_q8 as f32 / 255.0).floor() as i32;
                fr.calculate_tc(frame_idx)
            } else {
                ([0.0, 0.0], [1.0, 1.0])
            };

            let mut r_x = p.size[0] * 0.5;
            let mut r_y = p.size[1] * 0.5;
            let speed = len3(p.vel);
            if vel_scaled {
                r_x += speed * vel_scale[0];
                r_y += speed * vel_scale[1];
            }

            let center = if self.use_world_xform {
                matrix_transform_point(&self.world_xform, p.pos)
            } else {
                p.pos
            };

            let (right_axis, up_axis) = if align_to_path {
                if speed < EPS_SMALL && world_align {
                    let mut m = matrix_set_hpb(default_rot[1], default_rot[0], default_rot[2]);
                    if self.use_world_xform {
                        m = matrix_mul_43(&self.world_xform, &m);
                    }
                    ([m[0][0], m[0][1], m[0][2]], [m[2][0], m[2][1], m[2][2]])
                } else if speed >= EPS_SMALL && face_align {
                    let k = scale3(p.vel, 1.0 / speed);
                    let ref_up = if dot3([0.0, 1.0, 0.0], k).abs() > 0.99 {
                        [0.0, 0.0, 1.0]
                    } else {
                        [0.0, 1.0, 0.0]
                    };
                    let mut i_ax = normalize_vec3(cross3(ref_up, k));
                    let mut j_ax = normalize_vec3(cross3(k, i_ax));
                    if self.use_world_xform {
                        i_ax = matrix_transform_dir(&self.world_xform, i_ax);
                        j_ax = matrix_transform_dir(&self.world_xform, j_ax);
                    }
                    (i_ax, j_ax)
                } else {
                    let mut dir = if speed >= EPS_SMALL {
                        scale3(p.vel, 1.0 / speed)
                    } else {
                        dir_from_hp(-default_rot[1], -default_rot[0])
                    };
                    if self.use_world_xform {
                        dir = matrix_transform_dir(&self.world_xform, dir);
                    }
                    let right = normalize_vec3(cross3(dir, cam_dir_norm));
                    (right, dir)
                }
            } else {
                (cam_right, cam_up)
            };

            out.push(ParticleBillboardInstance {
                center,
                rotation_rad: p.rot[0],
                half_size: [r_x, r_y],
                right_axis,
                up_axis,
                uv_min,
                uv_max,
                color_rgba: p.color_rgba,
            });
        }
        out
    }

    /// Emit indexed CPU quad vertices ready for direct vertex/index buffer upload.
    pub fn emit_billboard_mesh(
        &self,
        cam_right: [f32; 3],
        cam_up: [f32; 3],
        cam_dir: [f32; 3],
    ) -> (Vec<ParticleQuadVertex>, Vec<u16>) {
        let billboards = self.emit_billboards(cam_right, cam_up, cam_dir);
        let mut verts = Vec::with_capacity(billboards.len() * 4);
        let mut indices = Vec::with_capacity(billboards.len() * 6);
        for bb in billboards {
            let Ok(base) = u16::try_from(verts.len()) else {
                break;
            };
            let (qv, qi) = bb.expand_quad(base);
            verts.extend_from_slice(&qv);
            indices.extend_from_slice(&qi);
        }
        (verts, indices)
    }
}

struct SpawnParticleParams {
    pos: [f32; 3],
    pos_b: [f32; 3],
    size: [f32; 3],
    rot: [f32; 3],
    vel: [f32; 3],
    color_rgba: [f32; 4],
    age: f32,
}

fn spawn_particle(
    particles: &mut Vec<ParticleState>,
    def: &ParticleEffectDef,
    rng: &mut ParticleRng,
    events: &mut ParticleStepEvents,
    params: SpawnParticleParams,
) -> bool {
    if particles.len() >= def.max_particles as usize {
        return false;
    }
    let mut frame_q8 = 0u16;
    if (def.flags & ped_flags::DF_RANDOM_FRAME) != 0
        && let Some(fr) = def.frame
    {
        let idx = rng.range_usize(fr.frame_count.max(1) as usize);
        frame_q8 = ((idx as f32) * 255.0).floor().min(u16::MAX as f32) as u16;
    }
    let animate_ccw = (def.flags & ped_flags::DF_ANIMATED) != 0
        && (def.flags & ped_flags::DF_RANDOM_PLAYBACK) != 0
        && (rng.next_u64() & 1) != 0;

    let particle = ParticleState {
        pos: params.pos,
        pos_b: params.pos_b,
        size: params.size,
        rot: params.rot,
        vel: params.vel,
        color_rgba: params.color_rgba,
        age: params.age,
        frame_q8,
        animate_ccw,
    };
    let slot_index = particles.len();
    particles.push(particle);
    events.born.push(ParticleLifeEvent {
        particle,
        slot_index,
    });
    true
}

fn remove_particle_swap(
    particles: &mut Vec<ParticleState>,
    idx: usize,
    events: &mut ParticleStepEvents,
) {
    if idx >= particles.len() {
        return;
    }
    let removed = particles.swap_remove(idx);
    events.died.push(ParticleLifeEvent {
        particle: removed,
        slot_index: idx,
    });
}

#[allow(clippy::too_many_arguments)]
fn execute_action(
    act: &mut RuntimeAction,
    particles: &mut Vec<ParticleState>,
    def: &ParticleEffectDef,
    dt: f32,
    tm_max: &mut f32,
    rng: &mut ParticleRng,
    noise: &PerlinNoise3,
    events: &mut ParticleStepEvents,
) {
    match &act.world {
        ParticleActionKind::Avoid {
            position,
            look_ahead,
            magnitude,
            epsilon,
        } => {
            let magdt = *magnitude * dt;
            execute_avoid(particles, position, *look_ahead, magdt, *epsilon, dt);
        }
        ParticleActionKind::Bounce {
            position,
            one_minus_friction,
            resilience,
            cutoff_sqr,
        } => {
            execute_bounce(
                particles,
                position,
                *one_minus_friction,
                *resilience,
                *cutoff_sqr,
                dt,
            );
        }
        ParticleActionKind::CopyVertexB { copy_pos } => {
            if *copy_pos != 0 {
                for p in particles.iter_mut() {
                    p.pos_b = p.pos;
                }
            }
        }
        ParticleActionKind::Damping {
            damping,
            v_low_sqr,
            v_high_sqr,
        } => {
            let scale = [
                1.0 - (1.0 - damping[0]) * dt,
                1.0 - (1.0 - damping[1]) * dt,
                1.0 - (1.0 - damping[2]) * dt,
            ];
            for p in particles.iter_mut() {
                let v2 = dot3(p.vel, p.vel);
                if v2 >= *v_low_sqr && v2 <= *v_high_sqr {
                    p.vel[0] *= scale[0];
                    p.vel[1] *= scale[1];
                    p.vel[2] *= scale[2];
                }
            }
        }
        ParticleActionKind::Explosion {
            center,
            velocity,
            magnitude,
            stdev,
            age: _,
            epsilon,
        } => {
            let radius = *velocity * act.explosion_age;
            let magdt = *magnitude * dt;
            let inv_sigma = if *stdev != 0.0 { 1.0 / *stdev } else { 0.0 };
            let inexp = -0.5 * inv_sigma * inv_sigma;
            let outexp = ONE_OVER_SQRT_2PI * inv_sigma;
            for p in particles.iter_mut() {
                let dir = sub3(p.pos, *center);
                let dist_sqr = dot3(dir, dir);
                let dist = dist_sqr.sqrt();
                let dfw = radius - dist;
                let gd = (dfw * dfw * inexp).exp() * outexp;
                let denom = (dist + 1e-6) * (dist_sqr + *epsilon);
                if denom.abs() > 1e-12 {
                    let s = gd * magdt / denom;
                    p.vel[0] += dir[0] * s;
                    p.vel[1] += dir[1] * s;
                    p.vel[2] += dir[2] * s;
                }
            }
            act.explosion_age += dt;
        }
        ParticleActionKind::Follow {
            magnitude,
            epsilon,
            max_radius,
        } => {
            if particles.len() >= 2 {
                let magdt = *magnitude * dt;
                let max_r2 = *max_radius * *max_radius;
                for i in 0..(particles.len() - 1) {
                    let to_him = sub3(particles[i + 1].pos, particles[i].pos);
                    let len2 = dot3(to_him, to_him);
                    if max_r2 >= P_MAXFLOAT || len2 < max_r2 {
                        let denom = len2.sqrt() * (len2 + *epsilon);
                        if denom > 1e-9 {
                            let s = magdt / denom;
                            particles[i].vel[0] += to_him[0] * s;
                            particles[i].vel[1] += to_him[1] * s;
                            particles[i].vel[2] += to_him[2] * s;
                        }
                    }
                }
            }
        }
        ParticleActionKind::Gravitate {
            magnitude,
            epsilon,
            max_radius,
        } => {
            let magdt = *magnitude * dt;
            let max_r2 = *max_radius * *max_radius;
            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let to_him = sub3(particles[j].pos, particles[i].pos);
                    let len2 = dot3(to_him, to_him) + EPS_SMALL;
                    if max_r2 >= P_MAXFLOAT || len2 < max_r2 {
                        let denom = len2.sqrt() * (len2 + *epsilon);
                        if denom > 1e-9 {
                            let acc = scale3(to_him, magdt / denom);
                            particles[i].vel = add3(particles[i].vel, acc);
                            particles[j].vel = sub3(particles[j].vel, acc);
                        }
                    }
                }
            }
        }
        ParticleActionKind::Gravity { direction } => {
            let dv = scale3(*direction, dt);
            for p in particles.iter_mut() {
                p.vel = add3(p.vel, dv);
            }
        }
        ParticleActionKind::Jet {
            center,
            acc,
            magnitude,
            epsilon,
            max_radius,
        } => {
            let magdt = *magnitude * dt;
            let max_r2 = *max_radius * *max_radius;
            for p in particles.iter_mut() {
                let d = sub3(p.pos, *center);
                let r2 = dot3(d, d);
                if max_r2 >= P_MAXFLOAT || r2 < max_r2 {
                    let a = acc.sample(rng);
                    let denom = r2 + *epsilon;
                    if denom.abs() > 1e-12 {
                        p.vel = add3(p.vel, scale3(a, magdt / denom));
                    }
                }
            }
        }
        ParticleActionKind::KillOld {
            age_limit,
            kill_less_than,
        } => {
            *tm_max = *age_limit;
            let less_flag = *kill_less_than != 0;
            for idx in (0..particles.len()).rev() {
                let cond = (particles[idx].age < *age_limit) ^ less_flag;
                if !cond {
                    remove_particle_swap(particles, idx, events);
                }
            }
        }
        ParticleActionKind::MatchVelocity {
            magnitude,
            epsilon,
            max_radius,
        } => {
            let magdt = *magnitude * dt;
            let max_r2 = *max_radius * *max_radius;
            for i in 0..particles.len() {
                for j in (i + 1)..particles.len() {
                    let d = sub3(particles[j].pos, particles[i].pos);
                    let r2 = dot3(d, d);
                    if max_r2 >= P_MAXFLOAT || r2 < max_r2 {
                        let denom = r2 + *epsilon;
                        if denom.abs() > 1e-12 {
                            let acc = scale3(particles[j].vel, magdt / denom);
                            particles[i].vel = add3(particles[i].vel, acc);
                            particles[j].vel = sub3(particles[j].vel, acc);
                        }
                    }
                }
            }
        }
        ParticleActionKind::Move => {
            for p in particles.iter_mut() {
                p.age += dt;
                p.pos_b = p.pos;
                p.pos[0] += p.vel[0] * dt;
                p.pos[1] += p.vel[1] * dt;
                p.pos[2] += p.vel[2] * dt;
            }
        }
        ParticleActionKind::OrbitLine {
            p: line_p,
            axis,
            magnitude,
            epsilon,
            max_radius,
        } => {
            let magdt = *magnitude * dt;
            let max_r2 = *max_radius * *max_radius;
            for p in particles.iter_mut() {
                let f = sub3(p.pos, *line_p);
                let w = scale3(*axis, dot3(f, *axis));
                let into = sub3(w, f);
                let r2 = dot3(into, into);
                if max_r2 >= P_MAXFLOAT || r2 < max_r2 {
                    let denom = r2.sqrt() + r2 + *epsilon;
                    if denom > 1e-9 {
                        p.vel = add3(p.vel, scale3(into, magdt / denom));
                    }
                }
            }
        }
        ParticleActionKind::OrbitPoint {
            center,
            magnitude,
            epsilon,
            max_radius,
        } => {
            let magdt = *magnitude * dt;
            let max_r2 = *max_radius * *max_radius;
            for p in particles.iter_mut() {
                let dir = sub3(*center, p.pos);
                let r2 = dot3(dir, dir);
                if max_r2 >= P_MAXFLOAT || r2 < max_r2 {
                    let denom = r2.sqrt() + r2 + *epsilon;
                    if denom > 1e-9 {
                        p.vel = add3(p.vel, scale3(dir, magdt / denom));
                    }
                }
            }
        }
        ParticleActionKind::RandomAccel { gen_acc } => {
            for p in particles.iter_mut() {
                let a = gen_acc.sample(rng);
                p.vel = add3(p.vel, scale3(a, dt));
            }
        }
        ParticleActionKind::RandomDisplace { gen_disp } => {
            for p in particles.iter_mut() {
                let disp = gen_disp.sample(rng);
                p.pos = add3(p.pos, scale3(disp, dt));
            }
        }
        ParticleActionKind::RandomVelocity { gen_vel } => {
            for p in particles.iter_mut() {
                p.vel = gen_vel.sample(rng);
            }
        }
        ParticleActionKind::Restore { .. } => {
            if act.restore_time_left <= 0.0 {
                for p in particles.iter_mut() {
                    p.pos = p.pos_b;
                    p.vel = [0.0; 3];
                }
            } else {
                let t = act.restore_time_left;
                let t_sqr_inv_2dt = (dt * 2.0) / (t * t);
                let t_cub_inv_3dt2 = (dt * dt * 3.0) / (t * t * t);
                for p in particles.iter_mut() {
                    for k in 0..3 {
                        let b = (-2.0 * t * p.vel[k] + 3.0 * p.pos_b[k] - 3.0 * p.pos[k]) * t_sqr_inv_2dt;
                        let a = (t * p.vel[k] - 2.0 * p.pos_b[k] + 2.0 * p.pos[k]) * t_cub_inv_3dt2;
                        p.vel[k] += a + b;
                    }
                }
            }
            act.restore_time_left -= dt;
        }
        ParticleActionKind::Scatter {
            center,
            magnitude,
            epsilon,
            max_radius,
        } => {
            let magdt = *magnitude * dt;
            let max_r2 = *max_radius * *max_radius;
            for p in particles.iter_mut() {
                let dir = sub3(p.pos, *center);
                let r2 = dot3(dir, dir);
                if (max_r2 >= P_MAXFLOAT || r2 < max_r2) && r2 > 1e-12 {
                    let r = r2.sqrt();
                    let denom = r2 + *epsilon;
                    if denom.abs() > 1e-12 {
                        let acc = scale3(dir, 1.0 / r);
                        p.vel = add3(p.vel, scale3(acc, magdt / denom));
                    }
                }
            }
        }
        ParticleActionKind::Sink {
            kill_inside,
            position,
        } => {
            let kill_in = *kill_inside != 0;
            for idx in (0..particles.len()).rev() {
                let cond = position.contains(particles[idx].pos) ^ kill_in;
                if !cond {
                    remove_particle_swap(particles, idx, events);
                }
            }
        }
        ParticleActionKind::SinkVelocity {
            kill_inside,
            velocity,
        } => {
            let kill_in = *kill_inside != 0;
            for idx in (0..particles.len()).rev() {
                let cond = velocity.contains(particles[idx].vel) ^ kill_in;
                if !cond {
                    remove_particle_swap(particles, idx, events);
                }
            }
        }
        ParticleActionKind::Source {
            position,
            velocity,
            rot,
            size,
            color,
            alpha,
            particle_rate,
            age,
            age_sigma,
            parent_vel: _,
            parent_motion: _,
        } => {
            if !act.source_silent {
                let exact = *particle_rate * dt;
                let mut rate = exact.floor().max(0.0) as usize;
                if rng.unit_f32() < (exact - rate as f32) {
                    rate += 1;
                }
                let room = (def.max_particles as usize).saturating_sub(particles.len());
                rate = rate.min(room);
                let tracks_b = (act.flags & papi_flags::PASOURCE_VERTEXB_TRACKS) != 0;
                let single_size = (act.flags & papi_flags::PASOURCE_SINGLE_SIZE) != 0;

                for _ in 0..rate {
                    let pos = position.sample(rng);
                    let mut siz = size.sample(rng);
                    if single_size {
                        siz = [siz[0], siz[0], siz[0]];
                    }
                    let rt = rot.sample(rng);
                    let mut vel = velocity.sample(rng);
                    vel = add3(vel, act.source_parent_vel);
                    let col = color.sample(rng);
                    let p_age = *age + rng.normal(*age_sigma);
                    let pos_b = if tracks_b { pos } else { [0.0; 3] };
                    let rgba = [
                        col[0].clamp(0.0, 1.0),
                        col[1].clamp(0.0, 1.0),
                        col[2].clamp(0.0, 1.0),
                        (*alpha).clamp(0.0, 1.0),
                    ];
                    spawn_particle(
                        particles,
                        def,
                        rng,
                        events,
                        SpawnParticleParams {
                            pos,
                            pos_b,
                            size: siz,
                            rot: rt,
                            vel,
                            color_rgba: rgba,
                            age: p_age,
                        },
                    );
                }
            }
        }
        ParticleActionKind::SpeedLimit {
            min_speed,
            max_speed,
        } => {
            let min2 = *min_speed * *min_speed;
            let max2 = *max_speed * *max_speed;
            for p in particles.iter_mut() {
                let s2 = dot3(p.vel, p.vel);
                if *min_speed >= 0.0 && s2 < min2 && s2 > 1e-12 {
                    let s = s2.sqrt();
                    p.vel = scale3(p.vel, *min_speed / s);
                } else if *max_speed >= 0.0 && s2 > max2 && s2 > 1e-12 {
                    let s = s2.sqrt();
                    p.vel = scale3(p.vel, *max_speed / s);
                }
            }
        }
        ParticleActionKind::TargetColor {
            color,
            alpha,
            scale,
            time_from,
            time_to,
        } => {
            // Exact formula from `PATargetColor::Execute` (`particle_actions_collection.cpp:1483`)
            let coeff = STEP_DEFAULT / dt.max(1e-5);
            let scale_fac = (*scale * STEP_DEFAULT).clamp(0.0, 1.0);
            let min_age = *time_from * *tm_max;
            let max_age = *time_to * *tm_max;
            for p in particles.iter_mut() {
                if p.age < min_age || p.age > max_age {
                    continue;
                }
                let tgt = [color[0], color[1], color[2], *alpha];
                for (cur, dst) in p.color_rgba.iter_mut().zip(tgt) {
                    let c_n = *cur + (dst - *cur) * scale_fac;
                    *cur -= (*cur - c_n) / coeff;
                }
            }
        }
        ParticleActionKind::TargetSize { size, scale } => {
            let sf = [scale[0] * dt, scale[1] * dt, scale[2] * dt];
            for p in particles.iter_mut() {
                for k in 0..3 {
                    p.size[k] += (size[k] - p.size[k]) * sf[k];
                }
            }
        }
        ParticleActionKind::TargetRotate { rot, scale, .. } => {
            let scale_fac = *scale * dt;
            let r = rot[0].abs();
            for p in particles.iter_mut() {
                let sign = if p.rot[0] >= 0.0 {
                    scale_fac
                } else {
                    -scale_fac
                };
                p.rot[0] += (r - p.rot[0].abs()) * sign;
            }
        }
        ParticleActionKind::TargetVelocity {
            velocity, scale, ..
        } => {
            let scale_fac = *scale * dt;
            for p in particles.iter_mut() {
                p.vel[0] += (velocity[0] - p.vel[0]) * scale_fac;
                p.vel[1] += (velocity[1] - p.vel[1]) * scale_fac;
                p.vel[2] += (velocity[2] - p.vel[2]) * scale_fac;
            }
        }
        ParticleActionKind::Vortex {
            center,
            axis,
            magnitude,
            epsilon,
            max_radius,
        } => {
            let magdt = *magnitude * dt;
            let max_r2 = *max_radius * *max_radius;
            for p in particles.iter_mut() {
                let mut offset = sub3(p.pos, *center);
                let r2 = dot3(offset, offset);
                if (max_r2 < P_MAXFLOAT && r2 > max_r2) || r2 <= 1e-12 {
                    continue;
                }
                let r = r2.sqrt();
                let off_norm = scale3(offset, 1.0 / r);
                let axis_proj = dot3(off_norm, *axis);
                let w = scale3(*axis, axis_proj);
                let u = sub3(off_norm, w);
                let v = cross3(*axis, u);
                let theta = magdt / (r2 + *epsilon);
                let (s, c) = theta.sin_cos();
                offset = [
                    (u[0] * c + v[0] * s + w[0]) * r,
                    (u[1] * c + v[1] * s + w[1]) * r,
                    (u[2] * c + v[2] * s + w[2]) * r,
                ];
                p.pos = add3(offset, *center);
            }
        }
        ParticleActionKind::Turbulence {
            frequency,
            octaves,
            magnitude,
            epsilon,
            offset,
        } => {
            act.turbulence_age += dt;
            for p in particles.iter_mut() {
                let pv = [
                    p.pos[0] + offset[0] * act.turbulence_age,
                    p.pos[1] + offset[1] * act.turbulence_age,
                    p.pos[2] + offset[2] * act.turbulence_age,
                ];
                let d = noise.fractalsum3(pv, *frequency, *octaves);
                let delta = [
                    (noise.fractalsum3([pv[0] + *epsilon, pv[1], pv[2]], *frequency, *octaves) - d)
                        * *magnitude,
                    (noise.fractalsum3([pv[0], pv[1] + *epsilon, pv[2]], *frequency, *octaves) - d)
                        * *magnitude,
                    (noise.fractalsum3([pv[0], pv[1], pv[2] + *epsilon], *frequency, *octaves) - d)
                        * *magnitude,
                ];
                let orig_mag = len3(p.vel);
                if orig_mag > 1e-9 {
                    let cand = add3(p.vel, delta);
                    let new_mag = len3(cand);
                    if new_mag > 1e-9 {
                        p.vel = scale3(cand, orig_mag / new_mag);
                    }
                }
            }
        }
    }
}

fn execute_avoid(
    particles: &mut [ParticleState],
    position: &ParticleDomain,
    look_ahead: f32,
    magdt: f32,
    epsilon: f32,
    dt: f32,
) {
    match position.domain_type {
        ParticleDomainType::Plane => {
            for p in particles.iter_mut() {
                let dist = dot3(p.pos, position.p2) + position.radius1;
                if look_ahead >= P_MAXFLOAT || dist < look_ahead {
                    let vm = len3(p.vel);
                    if vm > 1e-9 {
                        let vn = scale3(p.vel, 1.0 / vm);
                        let push = scale3(position.p2, magdt / (dist * dist + epsilon));
                        let tmp = add3(push, vn);
                        let tlen = len3(tmp);
                        if tlen > 1e-9 {
                            p.vel = scale3(tmp, vm / tlen);
                        }
                    }
                }
            }
        }
        ParticleDomainType::Sphere => {
            let r_sqr = position.radius1 * position.radius1;
            for p in particles.iter_mut() {
                let vm = len3(p.vel);
                if vm <= 1e-9 {
                    continue;
                }
                let vn = scale3(p.vel, 1.0 / vm);
                let l = sub3(position.p1, p.pos);
                let v = dot3(l, vn);
                let disc = r_sqr - dot3(l, l) + v * v;
                if disc < 0.0 {
                    continue;
                }
                let t = v - disc.sqrt();
                if t < 0.0 || t > vm * look_ahead {
                    continue;
                }
                let c = normalize_vec3(cross3(vn, l));
                let s = cross3(vn, c);
                let tmp = add3(scale3(s, magdt / (t * t + epsilon)), vn);
                let tlen = len3(tmp);
                if tlen > 1e-9 {
                    p.vel = scale3(tmp, vm / tlen);
                }
            }
        }
        ParticleDomainType::Disc => {
            let r1_sqr = position.radius1 * position.radius1;
            let r2_sqr = position.radius2 * position.radius2;
            for p in particles.iter_mut() {
                let pnext = add3(p.pos, scale3(p.vel, dt * look_ahead));
                let dist_old = dot3(p.pos, position.p2) + position.radius1_sqr;
                let dist_new = dot3(pnext, position.p2) + position.radius1_sqr;
                if dist_old * dist_new >= 0.0 {
                    continue;
                }
                let nv = dot3(position.p2, p.vel);
                if nv.abs() <= 1e-9 {
                    continue;
                }
                let t = -dist_old / nv;
                let phit = add3(p.pos, scale3(p.vel, t));
                let offset = sub3(phit, position.p1);
                let rad2 = dot3(offset, offset);
                if rad2 > r1_sqr || rad2 < r2_sqr {
                    continue;
                }
                let s = normalize_vec3(offset);
                let vm = len3(p.vel);
                if vm > 1e-9 {
                    let vn = scale3(p.vel, 1.0 / vm);
                    let tmp = add3(scale3(s, magdt / (t * t + epsilon)), vn);
                    let tlen = len3(tmp);
                    if tlen > 1e-9 {
                        p.vel = scale3(tmp, vm / tlen);
                    }
                }
            }
        }
        ParticleDomainType::Triangle | ParticleDomainType::Rectangle => {
            let Some((s1, s2)) = plane_uv_basis(position) else {
                return;
            };
            let is_tri = position.domain_type == ParticleDomainType::Triangle;
            let un = if position.radius1_sqr != 0.0 {
                scale3(position.u, 1.0 / position.radius1_sqr)
            } else {
                [0.0; 3]
            };
            let vn_basis = if position.radius2_sqr != 0.0 {
                scale3(position.v, 1.0 / position.radius2_sqr)
            } else {
                [0.0; 3]
            };
            let fn_edge = normalize_vec3(sub3(position.v, position.u));

            for p in particles.iter_mut() {
                let pnext = add3(p.pos, scale3(p.vel, dt * look_ahead));
                let dist_old = dot3(p.pos, position.p2) + position.radius1;
                let dist_new = dot3(pnext, position.p2) + position.radius1;
                if dist_old * dist_new >= 0.0 {
                    continue;
                }
                let nv = dot3(position.p2, p.vel);
                if nv.abs() <= 1e-9 {
                    continue;
                }
                let t = -dist_old / nv;
                let phit = add3(p.pos, scale3(p.vel, t));
                let offset = sub3(phit, position.p1);
                let upos = dot3(offset, s1);
                let vpos = dot3(offset, s2);
                let outside = if is_tri {
                    upos < 0.0 || vpos < 0.0 || (upos + vpos) > 1.0
                } else {
                    upos < 0.0 || vpos < 0.0 || upos > 1.0 || vpos > 1.0
                };
                if outside {
                    continue;
                }
                let uofs = sub3(scale3(un, dot3(un, offset)), offset);
                let vofs = sub3(scale3(vn_basis, dot3(vn_basis, offset)), offset);
                let s = if is_tri {
                    let foff = sub3(offset, position.u);
                    let fofs = sub3(scale3(fn_edge, dot3(fn_edge, foff)), foff);
                    let ud = dot3(uofs, uofs);
                    let vd = dot3(vofs, vofs);
                    let fd = dot3(fofs, fofs);
                    if ud <= vd && ud <= fd {
                        normalize_vec3(uofs)
                    } else if vd <= fd {
                        normalize_vec3(vofs)
                    } else {
                        normalize_vec3(fofs)
                    }
                } else {
                    let foff = sub3(add3(position.u, position.v), offset);
                    let fofs = sub3(scale3(un, dot3(un, foff)), foff);
                    let ud = dot3(uofs, uofs);
                    let vd = dot3(vofs, vofs);
                    let fd = dot3(fofs, fofs);
                    if ud <= vd && ud <= fd {
                        normalize_vec3(uofs)
                    } else if vd <= fd {
                        normalize_vec3(vofs)
                    } else {
                        normalize_vec3(fofs)
                    }
                };
                let vm = len3(p.vel);
                if vm > 1e-9 {
                    let vn = scale3(p.vel, 1.0 / vm);
                    let tmp = add3(scale3(s, magdt / (t * t + epsilon)), vn);
                    let tlen = len3(tmp);
                    if tlen > 1e-9 {
                        p.vel = scale3(tmp, vm / tlen);
                    }
                }
            }
        }
        _ => {}
    }
}

fn execute_bounce(
    particles: &mut [ParticleState],
    position: &ParticleDomain,
    one_minus_friction: f32,
    resilience: f32,
    cutoff_sqr: f32,
    dt: f32,
) {
    match position.domain_type {
        ParticleDomainType::Plane => {
            for p in particles.iter_mut() {
                let pnext = add3(p.pos, scale3(p.vel, dt));
                let d_old = dot3(p.pos, position.p2) + position.radius1;
                let d_new = dot3(pnext, position.p2) + position.radius1;
                if d_old * d_new >= 0.0 {
                    continue;
                }
                let nmag = dot3(p.vel, position.p2);
                apply_bounce_velocity(p, position.p2, nmag, one_minus_friction, resilience, cutoff_sqr);
            }
        }
        ParticleDomainType::Sphere => {
            for p in particles.iter_mut() {
                let pnext = add3(p.pos, scale3(p.vel, dt));
                if !position.contains(pnext) {
                    continue;
                }
                let inside = position.contains(p.pos);
                let n = normalize_vec3(sub3(p.pos, position.p1));
                let nmag = dot3(p.vel, n);
                if inside {
                    if nmag < 0.0 {
                        let vn = scale3(n, nmag);
                        let vt = sub3(p.vel, vn);
                        p.vel = sub3(vt, vn);
                    }
                } else {
                    apply_bounce_velocity(p, n, nmag, one_minus_friction, resilience, cutoff_sqr);
                }
            }
        }
        ParticleDomainType::Disc => {
            let r1_sqr = position.radius1 * position.radius1;
            let r2_sqr = position.radius2 * position.radius2;
            for p in particles.iter_mut() {
                let pnext = add3(p.pos, scale3(p.vel, dt));
                let d_old = dot3(p.pos, position.p2) + position.radius1_sqr;
                let d_new = dot3(pnext, position.p2) + position.radius1_sqr;
                if d_old * d_new >= 0.0 {
                    continue;
                }
                let nv = dot3(position.p2, p.vel);
                if nv.abs() <= 1e-9 {
                    continue;
                }
                let t = -d_old / nv;
                let phit = add3(p.pos, scale3(p.vel, t));
                let rad2 = dot3(sub3(phit, position.p1), sub3(phit, position.p1));
                if rad2 > r1_sqr || rad2 < r2_sqr {
                    continue;
                }
                apply_bounce_velocity(p, position.p2, nv, one_minus_friction, resilience, cutoff_sqr);
            }
        }
        ParticleDomainType::Triangle | ParticleDomainType::Rectangle => {
            let Some((s1, s2)) = plane_uv_basis(position) else {
                return;
            };
            let is_tri = position.domain_type == ParticleDomainType::Triangle;
            for p in particles.iter_mut() {
                let pnext = add3(p.pos, scale3(p.vel, dt));
                let d_old = dot3(p.pos, position.p2) + position.radius1;
                let d_new = dot3(pnext, position.p2) + position.radius1;
                if d_old * d_new >= 0.0 {
                    continue;
                }
                let nv = dot3(position.p2, p.vel);
                if nv.abs() <= 1e-9 {
                    continue;
                }
                let t = -d_old / nv;
                let phit = add3(p.pos, scale3(p.vel, t));
                let offset = sub3(phit, position.p1);
                let upos = dot3(offset, s1);
                let vpos = dot3(offset, s2);
                let outside = if is_tri {
                    upos < 0.0 || vpos < 0.0 || (upos + vpos) > 1.0
                } else {
                    upos < 0.0 || vpos < 0.0 || upos > 1.0 || vpos > 1.0
                };
                if outside {
                    continue;
                }
                apply_bounce_velocity(p, position.p2, nv, one_minus_friction, resilience, cutoff_sqr);
            }
        }
        _ => {}
    }
}

#[inline]
fn apply_bounce_velocity(
    p: &mut ParticleState,
    normal: [f32; 3],
    nmag: f32,
    one_minus_friction: f32,
    resilience: f32,
    cutoff_sqr: f32,
) {
    let vn = scale3(normal, nmag);
    let vt = sub3(p.vel, vn);
    let vn_scaled = scale3(vn, resilience);
    if dot3(vt, vt) <= cutoff_sqr {
        p.vel = sub3(vt, vn_scaled);
    } else {
        p.vel = sub3(scale3(vt, one_minus_friction), vn_scaled);
    }
}

fn plane_uv_basis(dom: &ParticleDomain) -> Option<([f32; 3], [f32; 3])> {
    let u = dom.u;
    let v = dom.v;
    let w = cross3(u, v);
    let denom = w[2] * u[0] * v[1]
        - w[2] * u[1] * v[0]
        - u[2] * w[0] * v[1]
        - u[0] * v[2] * w[1]
        + v[2] * w[0] * u[1]
        + u[2] * v[0] * w[1];
    if denom.abs() <= 1e-12 {
        return None;
    }
    let inv = 1.0 / denom;
    let s1 = [
        (v[1] * w[2] - v[2] * w[1]) * inv,
        (v[2] * w[0] - v[0] * w[2]) * inv,
        (v[0] * w[1] - v[1] * w[0]) * inv,
    ];
    let s2 = [
        -(u[1] * w[2] - u[2] * w[1]) * inv,
        -(u[2] * w[0] - u[0] * w[2]) * inv,
        -(u[0] * w[1] - u[1] * w[0]) * inv,
    ];
    Some((s1, s2))
}

// ============================================================================
// ParticleGroupInstance (`CPGDef` runtime scheduler)
// ============================================================================

/// Draw batch of billboard instances grouped by particle effect shader & texture.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleBillboardDrawBatch {
    pub effect_name: String,
    pub shader: Option<String>,
    pub texture: Option<String>,
    pub instances: Vec<ParticleBillboardInstance>,
}

#[derive(Debug, Clone)]
struct GroupRuntimeItem {
    effect: Option<ParticleEffectInstance>,
    related_children: Vec<ParticleEffectInstance>,
    free_children: Vec<ParticleEffectInstance>,
}

impl GroupRuntimeItem {
    fn play(&mut self) {
        if let Some(eff) = &mut self.effect {
            eff.play();
        }
    }

    fn stop(&mut self, deferred: bool) {
        if let Some(eff) = &mut self.effect {
            eff.stop(deferred);
        }
        for c in &mut self.related_children {
            c.stop(deferred);
        }
        for c in &mut self.free_children {
            c.stop(deferred);
        }
        if !deferred {
            self.related_children.clear();
            self.free_children.clear();
        }
    }

    fn is_playing(&self) -> bool {
        self.effect.as_ref().is_some_and(ParticleEffectInstance::is_playing)
    }
}

/// Stateful CPU simulator for a `ParticleGroupDef` (`CParticleGroup`) including timed stages and child emitters.
#[derive(Debug, Clone)]
pub struct ParticleGroupInstance {
    pub def: ParticleGroupDef,
    items: Vec<GroupRuntimeItem>,
    playing: bool,
    deferred_stop: bool,
    current_time: f32,
    initial_position: [f32; 3],
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    child_seed_counter: u64,
}

impl ParticleGroupInstance {
    pub fn compile(
        def: ParticleGroupDef,
        lib: &ParticlesLibrary,
        seed: u64,
    ) -> Result<Self, &'static str> {
        let mut items = Vec::with_capacity(def.effects.len());
        for (idx, item_def) in def.effects.iter().enumerate() {
            let eff_def = lib
                .find_effect(&item_def.effect_name)
                .ok_or("particle group references missing effect")?
                .clone();
            let eff_inst = ParticleEffectInstance::new(eff_def, seed.wrapping_add(idx as u64 + 1));
            items.push(GroupRuntimeItem {
                effect: Some(eff_inst),
                related_children: Vec::new(),
                free_children: Vec::new(),
            });
        }
        Ok(Self {
            def,
            items,
            playing: false,
            deferred_stop: false,
            current_time: 0.0,
            initial_position: [0.0; 3],
            bounds_min: [-0.001; 3],
            bounds_max: [0.001; 3],
            child_seed_counter: seed.wrapping_add(0x1000),
        })
    }

    pub fn play(&mut self) {
        self.current_time = 0.0;
        self.deferred_stop = false;
        self.playing = true;
    }

    pub fn stop(&mut self, deferred: bool) {
        if deferred {
            self.deferred_stop = true;
        } else {
            self.playing = false;
            self.deferred_stop = false;
        }
        for item in &mut self.items {
            item.stop(deferred);
        }
    }

    #[inline]
    pub fn is_playing(&self) -> bool {
        self.playing
    }

    #[inline]
    pub fn bounds_aabb(&self) -> ([f32; 3], [f32; 3]) {
        (self.bounds_min, self.bounds_max)
    }

    pub fn particle_count(&self) -> usize {
        self.items
            .iter()
            .map(|it| {
                it.effect.as_ref().map_or(0, |e| e.particles().len())
                    + it.related_children.iter().map(|c| c.particles().len()).sum::<usize>()
                    + it.free_children.iter().map(|c| c.particles().len()).sum::<usize>()
            })
            .sum()
    }

    pub fn update_parent(&mut self, xform: XRayMatrix, velocity: [f32; 3], use_world_xform: bool) {
        self.initial_position = [xform[3][0], xform[3][1], xform[3][2]];
        for item in &mut self.items {
            if let Some(eff) = &mut item.effect {
                eff.update_parent(xform, velocity, use_world_xform);
            }
        }
    }

    pub fn step(&mut self, dt_ms: u32, lib: &ParticlesLibrary) {
        if !self.playing {
            self.bounds_min = [
                self.initial_position[0] - 0.001,
                self.initial_position[1] - 0.001,
                self.initial_position[2] - 0.001,
            ];
            self.bounds_max = [
                self.initial_position[0] + 0.001,
                self.initial_position[1] + 0.001,
                self.initial_position[2] + 0.001,
            ];
            return;
        }

        let ct = self.current_time;
        let f_dt = (dt_ms as f32) / 1000.0;
        for (item_def, runtime) in self.def.effects.iter().zip(self.items.iter_mut()) {
            if (item_def.flags & pg_effect_flags::FL_ENABLED) == 0 {
                continue;
            }
            if runtime.is_playing() {
                if ct <= item_def.time1 && (ct + f_dt) >= item_def.time1 {
                    let deferred = (item_def.flags & pg_effect_flags::FL_DEFERRED_STOP) != 0;
                    runtime.stop(deferred);
                }
            } else if !self.deferred_stop && ct <= item_def.time0 && (ct + f_dt) >= item_def.time0 {
                runtime.play();
            }
        }

        self.current_time += f_dt;
        if self.def.time_limit > 0.0
            && self.current_time > self.def.time_limit
            && !self.deferred_stop
        {
            self.stop(true);
        }

        let mut any_playing = false;
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        let mut any_bounds = false;

        for (item_def, runtime) in self.def.effects.iter().zip(self.items.iter_mut()) {
            if let Some(eff) = &mut runtime.effect {
                let parent_xform = eff.world_xform;
                let parent_use_xform = eff.use_world_xform;
                let events = eff.step(dt_ms);

                // Trigger OnDead first in case `swap_remove` shifts indices
                for ev in &events.died {
                    if (item_def.flags & pg_effect_flags::FL_ON_PLAY_CHILD) != 0
                        && ev.slot_index < runtime.related_children.len()
                    {
                        let mut stopped = runtime.related_children.swap_remove(ev.slot_index);
                        stopped.stop(true);
                        runtime.free_children.push(stopped);
                    }
                    if (item_def.flags & pg_effect_flags::FL_ON_DEAD_CHILD) != 0
                        && !item_def.on_dead_child.is_empty()
                        && let Some(cdef) = lib.find_effect(&item_def.on_dead_child)
                    {
                        self.child_seed_counter = self.child_seed_counter.wrapping_add(1);
                        let child = spawn_child_effect(
                            cdef.clone(),
                            &ev.particle,
                            parent_xform,
                            parent_use_xform,
                            self.child_seed_counter,
                        );
                        runtime.free_children.push(child);
                    }
                }

                for ev in &events.born {
                    if (item_def.flags & pg_effect_flags::FL_ON_BIRTH_CHILD) != 0
                        && !item_def.on_birth_child.is_empty()
                        && let Some(cdef) = lib.find_effect(&item_def.on_birth_child)
                    {
                        self.child_seed_counter = self.child_seed_counter.wrapping_add(1);
                        let child = spawn_child_effect(
                            cdef.clone(),
                            &ev.particle,
                            parent_xform,
                            parent_use_xform,
                            self.child_seed_counter,
                        );
                        runtime.free_children.push(child);
                    }
                    if (item_def.flags & pg_effect_flags::FL_ON_PLAY_CHILD) != 0
                        && !item_def.on_play_child.is_empty()
                        && let Some(cdef) = lib.find_effect(&item_def.on_play_child)
                    {
                        self.child_seed_counter = self.child_seed_counter.wrapping_add(1);
                        let child = spawn_child_effect(
                            cdef.clone(),
                            &ev.particle,
                            parent_xform,
                            parent_use_xform,
                            self.child_seed_counter,
                        );
                        runtime.related_children.push(child);
                    }
                }

                if eff.is_playing() {
                    any_playing = true;
                    let (bmn, bmx) = eff.bounds_aabb();
                    merge_aabb(&mut mn, &mut mx, bmn, bmx);
                    any_bounds = true;
                    if (item_def.flags & pg_effect_flags::FL_ON_PLAY_CHILD) != 0
                        && !item_def.on_play_child.is_empty()
                    {
                        for (p, child) in eff
                            .particles()
                            .iter()
                            .zip(runtime.related_children.iter_mut())
                        {
                            let child_fstep = (child.def.update_step_ms.max(1) as f32) / 1000.0;
                            let mut m = XRAY_IDENTITY_MATRIX;
                            m[3][0] = p.pos[0];
                            m[3][1] = p.pos[1];
                            m[3][2] = p.pos[2];
                            let v = scale3(sub3(p.pos, p.pos_b), 1.0 / child_fstep);
                            child.update_parent(m, v, false);
                        }
                    }
                }
            }

            let rewind = (item_def.flags & pg_effect_flags::FL_ON_PLAY_CHILD_REWIND) != 0;
            for child in &mut runtime.related_children {
                child.step(dt_ms);
                if child.is_playing() {
                    any_playing = true;
                    let (bmn, bmx) = child.bounds_aabb();
                    merge_aabb(&mut mn, &mut mx, bmn, bmx);
                    any_bounds = true;
                } else if rewind {
                    child.play();
                }
            }

            for child in &mut runtime.free_children {
                child.step(dt_ms);
                if child.is_playing() {
                    any_playing = true;
                    let (bmn, bmx) = child.bounds_aabb();
                    merge_aabb(&mut mn, &mut mx, bmn, bmx);
                    any_bounds = true;
                }
            }
            runtime.free_children.retain(ParticleEffectInstance::is_playing);
        }

        if self.deferred_stop && !any_playing {
            self.playing = false;
            self.deferred_stop = false;
        }
        if any_bounds {
            self.bounds_min = mn;
            self.bounds_max = mx;
        }
    }

    /// Collect all live billboard instances organized by effect shader/texture batch.
    pub fn emit_billboards(
        &self,
        cam_right: [f32; 3],
        cam_up: [f32; 3],
        cam_dir: [f32; 3],
    ) -> Vec<ParticleBillboardDrawBatch> {
        let mut batches = Vec::new();
        for item in &self.items {
            for eff in item
                .effect
                .iter()
                .chain(item.related_children.iter())
                .chain(item.free_children.iter())
            {
                let instances = eff.emit_billboards(cam_right, cam_up, cam_dir);
                if !instances.is_empty() {
                    batches.push(ParticleBillboardDrawBatch {
                        effect_name: eff.def.name.clone(),
                        shader: eff.def.shader.clone(),
                        texture: eff.def.texture.clone(),
                        instances,
                    });
                }
            }
        }
        batches
    }
}

fn spawn_child_effect(
    def: ParticleEffectDef,
    parent_particle: &ParticleState,
    parent_xform: XRayMatrix,
    parent_use_xform: bool,
    seed: u64,
) -> ParticleEffectInstance {
    let mut child = ParticleEffectInstance::new(def, seed);
    let child_fstep = (child.def.update_step_ms.max(1) as f32) / 1000.0;
    let mut vel = scale3(sub3(parent_particle.pos, parent_particle.pos_b), 1.0 / child_fstep);
    let mut m = if parent_use_xform {
        vel = matrix_transform_dir(&parent_xform, vel);
        parent_xform
    } else {
        XRAY_IDENTITY_MATRIX
    };
    let pos = matrix_transform_point(&m, parent_particle.pos);
    m[3][0] = pos[0];
    m[3][1] = pos[1];
    m[3][2] = pos[2];
    child.play();
    child.update_parent(m, vel, false);
    child
}

// ============================================================================
// Vector Helpers
// ============================================================================

#[inline]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

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
fn len3(a: [f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

#[inline]
fn dir_from_hp(h: f32, p: f32) -> [f32; 3] {
    let (sh, ch) = h.sin_cos();
    let (sp, cp) = p.sin_cos();
    [-cp * sh, sp, cp * ch]
}

#[inline]
fn merge_aabb(dst_min: &mut [f32; 3], dst_max: &mut [f32; 3], src_min: [f32; 3], src_max: [f32; 3]) {
    for k in 0..3 {
        dst_min[k] = dst_min[k].min(src_min[k]);
        dst_max[k] = dst_max[k].max(src_max[k]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::{GroupEffectItem, ParticleSpriteFrame};

    #[test]
    fn test_particle_domain_sampling_and_perlin_noise() {
        let mut rng = ParticleRng::new(42);
        let box_dom = ParticleDomain::from_editor_params(
            ParticleDomainType::Box,
            [-2.0, 0.0, -1.0, 2.0, 4.0, 1.0, 0.0, 0.0, 0.0],
        );
        for _ in 0..64 {
            let pt = box_dom.sample(&mut rng);
            assert!(box_dom.contains(pt));
        }

        let sphere_dom = ParticleDomain::from_editor_params(
            ParticleDomainType::Sphere,
            [1.0, 2.0, 3.0, 1.5, 3.0, 0.0, 0.0, 0.0, 0.0],
        );
        for _ in 0..64 {
            let pt = sphere_dom.sample(&mut rng);
            assert!(sphere_dom.contains(pt));
        }

        // Translated Box transform shifts bounds cleanly
        let mut xform = XRAY_IDENTITY_MATRIX;
        xform[3] = [10.0, -5.0, 2.0, 1.0];
        let translated = box_dom.transform(&xform);
        assert_eq!(translated.p1, [8.0, -5.0, 1.0]);
        assert_eq!(translated.p2, [12.0, -1.0, 3.0]);

        let noise = PerlinNoise3::new();
        let n0 = noise.fractalsum3([0.3, 1.7, -0.9], 2.0, 3);
        let n1 = noise.fractalsum3([0.3, 1.7, -0.9], 2.0, 3);
        assert_eq!(n0, n1);
        assert!(n0.is_finite());
        assert!(noise.turbulence3([1.2, 0.4, 2.1], 1.5, 2) >= 0.0);
    }

    #[test]
    fn test_particle_effect_simulation_actions_and_billboard_output() {
        let def = ParticleEffectDef {
            name: "test\\fountain".to_string(),
            flags: ped_flags::DF_SPRITE | ped_flags::DF_FRAMED | ped_flags::DF_ANIMATED,
            max_particles: 32,
            update_step_ms: 50,
            shader: Some("effects\\add".to_string()),
            texture: Some("fx\\spark".to_string()),
            frame: Some(ParticleSpriteFrame {
                tex_size: [0.5, 0.5],
                reserved: [0.0, 0.0],
                frame_dim_x: 2,
                frame_count: 4,
                speed: 20.0, // advances 1 frame per 0.05s step
            }),
            time_limit: None,
            collision: None,
            velocity_scale: None,
            align_to_path_rotation: None,
            actions: vec![
                CompiledParticleAction {
                    outer_type: 21,
                    flags: papi_flags::PASOURCE_VERTEXB_TRACKS | papi_flags::ALLOW_ROTATE,
                    inner_type: 21,
                    kind: ParticleActionKind::Source {
                        position: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [0.0; 9],
                        ),
                        velocity: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        rot: ParticleDomain::default(),
                        size: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [2.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        color: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        alpha: 1.0,
                        particle_rate: 100.0, // 5 per 50ms step
                        age: 0.0,
                        age_sigma: 0.0,
                        parent_vel: [0.0; 3],
                        parent_motion: 1.0,
                    },
                },
                CompiledParticleAction {
                    outer_type: 8,
                    flags: 0,
                    inner_type: 8,
                    kind: ParticleActionKind::Gravity {
                        direction: [0.0, -10.0, 0.0],
                    },
                },
                CompiledParticleAction {
                    outer_type: 12,
                    flags: 0,
                    inner_type: 12,
                    kind: ParticleActionKind::Move,
                },
                CompiledParticleAction {
                    outer_type: 10,
                    flags: 0,
                    inner_type: 10,
                    kind: ParticleActionKind::KillOld {
                        age_limit: 0.25,
                        kill_less_than: 0,
                    },
                },
            ],
        };

        let mut inst = ParticleEffectInstance::new(def, 7);
        let mut xform = XRAY_IDENTITY_MATRIX;
        xform[3] = [10.0, 0.0, -4.0, 1.0];
        inst.update_parent(xform, [0.0; 3], false);
        inst.play();

        let ev1 = inst.step(50);
        assert_eq!(ev1.born.len(), 5);
        assert_eq!(inst.particles().len(), 5);
        // After 1st 0.05s step: vel_y = 10 - 0.5 = 9.5, pos_y = 0.05 * 9.5 = 0.475, pos_x = 10.0
        let p0 = inst.particles()[0];
        assert!((p0.pos[0] - 10.0).abs() < 1e-4);
        assert!((p0.pos[1] - 0.475).abs() < 1e-4);
        assert!((p0.pos[2] - (-4.0)).abs() < 1e-4);
        // Animated frame advanced by 20 * 0.05 = 1 frame -> frame_q8 == 255 -> UV quad 1 [0.5..1.0, 0.0..0.5]
        assert_eq!(p0.frame_q8, 255);

        let bbs = inst.emit_billboards([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(bbs.len(), 5);
        assert_eq!(bbs[0].uv_min, [0.5, 0.0]);
        assert_eq!(bbs[0].uv_max, [1.0, 0.5]);
        assert_eq!(bbs[0].half_size, [1.0, 2.0]);

        let (qverts, qidx) = inst.emit_billboard_mesh([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(qverts.len(), 20);
        assert_eq!(qidx.len(), 30);
        assert_eq!(&qidx[0..6], &[0, 1, 2, 3, 2, 1]);

        // Advance several steps until KillOld triggers and clears old generation
        for _ in 0..8 {
            inst.step(50);
        }
        assert!(inst.particles().len() <= 25);
        inst.stop(false);
        assert!(!inst.is_playing());
        assert!(inst.particles().is_empty());
    }

    #[test]
    fn test_particle_group_timeline_and_on_birth_child() {
        let burst = ParticleEffectDef {
            name: "grp\\parent".to_string(),
            flags: ped_flags::DF_SPRITE,
            max_particles: 4,
            update_step_ms: 50,
            shader: Some("effects\\add".to_string()),
            texture: Some("fx\\flash".to_string()),
            frame: None,
            time_limit: None,
            collision: None,
            velocity_scale: None,
            align_to_path_rotation: None,
            actions: vec![
                CompiledParticleAction {
                    outer_type: 21,
                    flags: papi_flags::PASOURCE_VERTEXB_TRACKS,
                    inner_type: 21,
                    kind: ParticleActionKind::Source {
                        position: ParticleDomain::default(),
                        velocity: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        rot: ParticleDomain::default(),
                        size: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        color: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        alpha: 1.0,
                        particle_rate: 40.0, // 2 per 50ms
                        age: 0.0,
                        age_sigma: 0.0,
                        parent_vel: [0.0; 3],
                        parent_motion: 0.0,
                    },
                },
                CompiledParticleAction {
                    outer_type: 12,
                    flags: 0,
                    inner_type: 12,
                    kind: ParticleActionKind::Move,
                },
            ],
        };
        let smoke = ParticleEffectDef {
            name: "grp\\smoke".to_string(),
            flags: ped_flags::DF_SPRITE | ped_flags::DF_TIME_LIMIT,
            max_particles: 8,
            update_step_ms: 50,
            shader: Some("effects\\blend".to_string()),
            texture: Some("fx\\smoke".to_string()),
            frame: None,
            time_limit: Some(0.25),
            collision: None,
            velocity_scale: None,
            align_to_path_rotation: None,
            actions: burst.actions.clone(),
        };

        let group_def = ParticleGroupDef {
            name: "grp\\combo".to_string(),
            flags: 0,
            time_limit: 0.30,
            effects: vec![GroupEffectItem {
                effect_name: "grp\\parent".to_string(),
                on_play_child: String::new(),
                on_birth_child: "grp\\smoke".to_string(),
                on_dead_child: String::new(),
                time0: 0.0,
                time1: 0.25,
                flags: pg_effect_flags::FL_ENABLED | pg_effect_flags::FL_ON_BIRTH_CHILD,
            }],
        };

        let lib = ParticlesLibrary {
            effects: vec![burst, smoke],
            groups: vec![group_def.clone()],
        };

        let mut grp = ParticleGroupInstance::compile(group_def, &lib, 99).unwrap();
        grp.play();
        grp.step(50, &lib);
        assert!(grp.is_playing());
        // Parent birthed 2 particles and spawned 2 free child smoke effects
        let batches = grp.emit_billboards([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(batches.iter().any(|b| b.effect_name == "grp\\parent"));

        grp.step(50, &lib);
        let batches2 = grp.emit_billboards([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(batches2.iter().any(|b| b.effect_name == "grp\\smoke"));
    }
}
