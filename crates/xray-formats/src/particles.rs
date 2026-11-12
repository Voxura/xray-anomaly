//! Pure Rust X-Ray `particles.xr` / `.pe` / `.pg` Particle System & Wallmark Config Parsers
//!
//! Covers:
//! - `particles.xr` (`CPSLibrary`: `PS_CHUNK_VERSION`, `PS_CHUNK_SECONDGEN` `CPEDef`,
//!   `PS_CHUNK_THIRDGEN` `CPGDef`), plus Anomaly loose `$game_particles$/*.pe` & `*.pg` LTX overrides
//! - `CPEDef` sprite frame UV table (`SFrame`), effect flags, collision/velocity-scale/align params
//! - Full PAPI compiled action bytecode decoder/compiler (`PActionEnum` `0..=31`, `pDomain` 68B)
//! - `CPGDef` hierarchical particle groups (`SEffect`, play/birth/death child hooks, time limits)
//! - Wallmark configuration LTX parser (`BloodWallmarksConfig` / `WallmarkSlotConfig`)
//!
//! Note: X-Ray Anomaly / CoP does **not** have a binary `level.wallmarks` file on disk.
//! `CWallmarksEngine` generates static and skeleton wallmarks dynamically at runtime via geometry
//! projection against `level.cform` (respecting `CformTriangle::suppress_wm` / `GAMEMTL_SUPPRESS_WM`),
//! configured by LTX sections referencing `effects\wallmark` textures.

use std::collections::{HashMap, HashSet};
use std::f32::consts::{FRAC_PI_2, PI};

use xray_core::LtxDocument;
use xray_core::ltx_loader::OwnedLtxDocument;

use crate::camera::matrix_build_camera;
use crate::cform::{sphere_triangle_test, CformBvh, CformModel};
use crate::chunk::{ByteCursor, ChunkReader, ChunkWriter, FormatError};
use crate::hom::{frustum_planes, ClipFrustum};
use crate::lanims::pack_argb;
use crate::ogf::{
    matrix_invert_43, matrix_mul_43, matrix_rotate_z, matrix_scale, matrix_transform_point,
    XRayMatrix,
};
use crate::skinning::{
    matrix_transform_dir, skin_position_with_links, skin_vertex_pos_and_influences, BoneMask,
    BonePalette, SkeletonCollisionIndex,
};

pub const PS_VERSION: u16 = 0x0001;
pub const PED_VERSION: u16 = 0x0001;
pub const PGD_VERSION: u16 = 0x0003;

const EPS_L: f32 = 0.001;
const P_MAXFLOAT: f32 = 1.0e16;

pub mod ps_chunks {
    pub const VERSION: u32 = 0x0001;
    pub const FIRSTGEN: u32 = 0x0002;
    pub const SECONDGEN: u32 = 0x0003;
    pub const THIRDGEN: u32 = 0x0004;
}

pub mod ped_chunks {
    pub const VERSION: u32 = 0x0001;
    pub const NAME: u32 = 0x0002;
    pub const EFFECTDATA: u32 = 0x0003;
    pub const ACTIONLIST: u32 = 0x0004;
    pub const FLAGS: u32 = 0x0005;
    pub const FRAME: u32 = 0x0006;
    pub const SPRITE: u32 = 0x0007;
    pub const TIMELIMIT: u32 = 0x0008;
    pub const TIMELIMIT2: u32 = 0x0009;
    pub const SOURCETEXT_OBSOLETE: u32 = 0x0020;
    pub const COLLISION: u32 = 0x0021;
    pub const VEL_SCALE: u32 = 0x0022;
    pub const EDATA: u32 = 0x0024;
    pub const ALIGN_TO_PATH: u32 = 0x0025;
}

pub mod ped_flags {
    pub const DF_SPRITE: u32 = 1 << 0;
    pub const DF_FRAMED: u32 = 1 << 10;
    pub const DF_ANIMATED: u32 = 1 << 11;
    pub const DF_RANDOM_FRAME: u32 = 1 << 12;
    pub const DF_RANDOM_PLAYBACK: u32 = 1 << 13;
    pub const DF_TIME_LIMIT: u32 = 1 << 14;
    pub const DF_ALIGN_TO_PATH: u32 = 1 << 15;
    pub const DF_COLLISION: u32 = 1 << 16;
    pub const DF_COLLISION_DEL: u32 = 1 << 17;
    pub const DF_VELOCITY_SCALE: u32 = 1 << 18;
    pub const DF_COLLISION_DYN: u32 = 1 << 19;
    pub const DF_WORLD_ALIGN: u32 = 1 << 20;
    pub const DF_FACE_ALIGN: u32 = 1 << 21;
    pub const DF_CULLING: u32 = 1 << 22;
    pub const DF_CULL_CCW: u32 = 1 << 23;
}

pub mod pgd_chunks {
    pub const VERSION: u32 = 0x0001;
    pub const NAME: u32 = 0x0002;
    pub const FLAGS: u32 = 0x0003;
    pub const EFFECTS: u32 = 0x0004;
    pub const TIME_LIMIT: u32 = 0x0005;
    pub const EFFECTS2: u32 = 0x0007;
}

pub mod pg_effect_flags {
    pub const FL_DEFERRED_STOP: u32 = 1 << 0;
    pub const FL_ON_PLAY_CHILD: u32 = 1 << 1;
    pub const FL_ENABLED: u32 = 1 << 2;
    pub const FL_ON_PLAY_CHILD_REWIND: u32 = 1 << 4;
    pub const FL_ON_BIRTH_CHILD: u32 = 1 << 5;
    pub const FL_ON_DEAD_CHILD: u32 = 1 << 6;
}

pub mod papi_flags {
    pub const ALLOW_ROTATE: u32 = 1 << 1;
    pub const PASOURCE_SINGLE_SIZE: u32 = 1 << 29;
    pub const PASOURCE_SILENT: u32 = 1 << 30;
    pub const PASOURCE_VERTEXB_TRACKS: u32 = 1 << 31;
    pub const ED_ACTION_ENABLED: u32 = 1 << 0;
    pub const ED_ACTION_DRAW: u32 = 1 << 1;
}

pub const PARTICLE_DOMAIN_BINARY_SIZE: usize = 68;

/// PAPI spatial/parameter domain shapes (`PAPI::PDomainEnum`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ParticleDomainType {
    Point = 0,
    Line = 1,
    Triangle = 2,
    Plane = 3,
    Box = 4,
    Sphere = 5,
    Cylinder = 6,
    Cone = 7,
    Blob = 8,
    Disc = 9,
    Rectangle = 10,
}

impl ParticleDomainType {
    pub fn from_u32(v: u32) -> Result<Self, FormatError> {
        match v {
            0 => Ok(Self::Point),
            1 => Ok(Self::Line),
            2 => Ok(Self::Triangle),
            3 => Ok(Self::Plane),
            4 => Ok(Self::Box),
            5 => Ok(Self::Sphere),
            6 => Ok(Self::Cylinder),
            7 => Ok(Self::Cone),
            8 => Ok(Self::Blob),
            9 => Ok(Self::Disc),
            10 => Ok(Self::Rectangle),
            _ => Err(FormatError::InvalidValue("unknown PAPI pDomain type")),
        }
    }
}

/// 68-byte compiled `PAPI::pDomain` (`#pragma pack(push, 4)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleDomain {
    pub domain_type: ParticleDomainType,
    pub p1: [f32; 3],
    pub p2: [f32; 3],
    pub u: [f32; 3],
    pub v: [f32; 3],
    pub radius1: f32,
    pub radius2: f32,
    pub radius1_sqr: f32,
    pub radius2_sqr: f32,
}

impl Default for ParticleDomain {
    fn default() -> Self {
        Self {
            domain_type: ParticleDomainType::Point,
            p1: [0.0; 3],
            p2: [0.0; 3],
            u: [0.0; 3],
            v: [0.0; 3],
            radius1: 0.0,
            radius2: 0.0,
            radius1_sqr: 0.0,
            radius2_sqr: 0.0,
        }
    }
}

impl ParticleDomain {
    /// Read a 68-byte compiled `pDomain` struct from `ByteCursor`.
    pub fn read_binary(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let dtype = ParticleDomainType::from_u32(cur.read_u32()?)?;
        Ok(Self {
            domain_type: dtype,
            p1: cur.read_f32_3()?,
            p2: cur.read_f32_3()?,
            u: cur.read_f32_3()?,
            v: cur.read_f32_3()?,
            radius1: cur.read_f32()?,
            radius2: cur.read_f32()?,
            radius1_sqr: cur.read_f32()?,
            radius2_sqr: cur.read_f32()?,
        })
    }

    pub fn write_binary(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.domain_type as u32).to_le_bytes());
        for vec in [self.p1, self.p2, self.u, self.v] {
            for f in vec {
                out.extend_from_slice(&f.to_le_bytes());
            }
        }
        for f in [self.radius1, self.radius2, self.radius1_sqr, self.radius2_sqr] {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }

    /// Compile an editor 9-float `PDomain` representation (`type, f[0..9]`) into `PAPI::pDomain`
    /// matching `particle_core.cpp::pDomain::pDomain`.
    pub fn from_editor_params(domain_type: ParticleDomainType, f: [f32; 9]) -> Self {
        let mut d = Self {
            domain_type,
            ..Self::default()
        };
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8] = f;
        match domain_type {
            ParticleDomainType::Point => {
                d.p1 = [a0, a1, a2];
            }
            ParticleDomainType::Line => {
                d.p1 = [a0, a1, a2];
                d.p2 = [a3 - a0, a4 - a1, a5 - a2];
            }
            ParticleDomainType::Box => {
                d.p1 = [a0.min(a3), a1.min(a4), a2.min(a5)];
                d.p2 = [a0.max(a3), a1.max(a4), a2.max(a5)];
            }
            ParticleDomainType::Triangle => {
                d.p1 = [a0, a1, a2];
                d.u = [a3 - a0, a4 - a1, a5 - a2];
                d.v = [a6 - a0, a7 - a1, a8 - a2];
                d.radius1_sqr = vec3_len(d.u);
                let tu = if d.radius1_sqr > 0.0 {
                    vec3_scale(d.u, 1.0 / d.radius1_sqr)
                } else {
                    [0.0; 3]
                };
                d.radius2_sqr = vec3_len(d.v);
                let tv = if d.radius2_sqr > 0.0 {
                    vec3_scale(d.v, 1.0 / d.radius2_sqr)
                } else {
                    [0.0; 3]
                };
                d.p2 = vec3_normalize_safe(vec3_cross(tu, tv));
                d.radius1 = -vec3_dot(d.p1, d.p2);
            }
            ParticleDomainType::Rectangle => {
                d.p1 = [a0, a1, a2];
                d.u = [a3, a4, a5];
                d.v = [a6, a7, a8];
                d.radius1_sqr = vec3_len(d.u);
                let tu = if d.radius1_sqr > 0.0 {
                    vec3_scale(d.u, 1.0 / d.radius1_sqr)
                } else {
                    [0.0; 3]
                };
                d.radius2_sqr = vec3_len(d.v);
                let tv = if d.radius2_sqr > 0.0 {
                    vec3_scale(d.v, 1.0 / d.radius2_sqr)
                } else {
                    [0.0; 3]
                };
                d.p2 = vec3_normalize_safe(vec3_cross(tu, tv));
                d.radius1 = -vec3_dot(d.p1, d.p2);
            }
            ParticleDomainType::Plane => {
                d.p1 = [a0, a1, a2];
                d.p2 = vec3_normalize_safe([a3, a4, a5]);
                d.radius1 = -vec3_dot(d.p1, d.p2);
            }
            ParticleDomainType::Sphere => {
                d.p1 = [a0, a1, a2];
                d.radius1 = a3.max(a4);
                d.radius2 = a3.min(a4);
                d.radius1_sqr = d.radius1 * d.radius1;
                d.radius2_sqr = d.radius2 * d.radius2;
            }
            ParticleDomainType::Cylinder | ParticleDomainType::Cone => {
                d.p1 = [a0, a1, a2];
                d.p2 = [a3 - a0, a4 - a1, a5 - a2];
                d.radius1 = a6.max(a7);
                d.radius2 = a6.min(a7);
                d.radius1_sqr = d.radius1 * d.radius1;
                let p2l2 = vec3_dot(d.p2, d.p2);
                d.radius2_sqr = if p2l2 != 0.0 { 1.0 / p2l2 } else { 0.0 };
                let n = vec3_normalize_safe(d.p2);
                let basis = if n[0].abs() > 0.999 {
                    [0.0, 1.0, 0.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let proj = vec3_dot(basis, n);
                d.u = vec3_normalize_safe([
                    basis[0] - n[0] * proj,
                    basis[1] - n[1] * proj,
                    basis[2] - n[2] * proj,
                ]);
                d.v = vec3_cross(n, d.u);
            }
            ParticleDomainType::Blob => {
                d.p1 = [a0, a1, a2];
                d.radius1 = a3;
                let tmp = if a3 != 0.0 { 1.0 / a3 } else { 0.0 };
                d.radius2_sqr = -0.5 * tmp * tmp;
                d.radius2 = (1.0 / (2.0 * PI).sqrt()) * tmp;
            }
            ParticleDomainType::Disc => {
                d.p1 = [a0, a1, a2];
                d.p2 = vec3_normalize_safe([a3, a4, a5]);
                d.radius1 = a6.max(a7);
                d.radius2 = a6.min(a7);
                let basis = if d.p2[0].abs() > 0.999 {
                    [0.0, 1.0, 0.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let proj = vec3_dot(basis, d.p2);
                d.u = vec3_normalize_safe([
                    basis[0] - d.p2[0] * proj,
                    basis[1] - d.p2[1] * proj,
                    basis[2] - d.p2[2] * proj,
                ]);
                d.v = vec3_cross(d.p2, d.u);
                d.radius1_sqr = -vec3_dot(d.p1, d.p2);
            }
        }
        d
    }

    /// Geometric `pDomain::Within` check (`Box`, `Plane`, `Sphere`, `Cylinder`, `Cone`).
    pub fn contains(&self, pos: [f32; 3]) -> bool {
        match self.domain_type {
            ParticleDomainType::Box => {
                pos[0] >= self.p1[0]
                    && pos[0] <= self.p2[0]
                    && pos[1] >= self.p1[1]
                    && pos[1] <= self.p2[1]
                    && pos[2] >= self.p1[2]
                    && pos[2] <= self.p2[2]
            }
            ParticleDomainType::Plane => vec3_dot(pos, self.p2) >= -self.radius1,
            ParticleDomainType::Sphere => {
                let d = vec3_sub(pos, self.p1);
                let r2 = vec3_dot(d, d);
                r2 <= self.radius1_sqr && r2 >= self.radius2_sqr
            }
            ParticleDomainType::Cylinder | ParticleDomainType::Cone => {
                let x = vec3_sub(pos, self.p1);
                let dist = vec3_dot(self.p2, x) * self.radius2_sqr;
                if !(0.0..=1.0).contains(&dist) {
                    return false;
                }
                let xrad = [
                    x[0] - self.p2[0] * dist,
                    x[1] - self.p2[1] * dist,
                    x[2] - self.p2[2] * dist,
                ];
                let r_sqr = vec3_dot(xrad, xrad);
                if self.domain_type == ParticleDomainType::Cone {
                    let rmax = dist * self.radius1;
                    let rmin = dist * self.radius2;
                    r_sqr <= rmax * rmax && r_sqr >= rmin * rmin
                } else {
                    r_sqr <= self.radius1_sqr && r_sqr >= self.radius2 * self.radius2
                }
            }
            _ => false,
        }
    }
}

/// Decoded payload of a compiled PAPI action (`particle_actions_collection_io.cpp`).
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ParticleActionKind {
    Avoid {
        position: ParticleDomain,
        look_ahead: f32,
        magnitude: f32,
        epsilon: f32,
    },
    Bounce {
        position: ParticleDomain,
        one_minus_friction: f32,
        resilience: f32,
        cutoff_sqr: f32,
    },
    CopyVertexB {
        copy_pos: u32,
    },
    Damping {
        damping: [f32; 3],
        v_low_sqr: f32,
        v_high_sqr: f32,
    },
    Explosion {
        center: [f32; 3],
        velocity: f32,
        magnitude: f32,
        stdev: f32,
        age: f32,
        epsilon: f32,
    },
    Follow {
        magnitude: f32,
        epsilon: f32,
        max_radius: f32,
    },
    Gravitate {
        magnitude: f32,
        epsilon: f32,
        max_radius: f32,
    },
    Gravity {
        direction: [f32; 3],
    },
    Jet {
        center: [f32; 3],
        acc: ParticleDomain,
        magnitude: f32,
        epsilon: f32,
        max_radius: f32,
    },
    KillOld {
        age_limit: f32,
        kill_less_than: u32,
    },
    MatchVelocity {
        magnitude: f32,
        epsilon: f32,
        max_radius: f32,
    },
    Move,
    OrbitLine {
        p: [f32; 3],
        axis: [f32; 3],
        magnitude: f32,
        epsilon: f32,
        max_radius: f32,
    },
    OrbitPoint {
        center: [f32; 3],
        magnitude: f32,
        epsilon: f32,
        max_radius: f32,
    },
    RandomAccel {
        gen_acc: ParticleDomain,
    },
    RandomDisplace {
        gen_disp: ParticleDomain,
    },
    RandomVelocity {
        gen_vel: ParticleDomain,
    },
    Restore {
        time_left: f32,
    },
    Scatter {
        center: [f32; 3],
        magnitude: f32,
        epsilon: f32,
        max_radius: f32,
    },
    Sink {
        kill_inside: u32,
        position: ParticleDomain,
    },
    SinkVelocity {
        kill_inside: u32,
        velocity: ParticleDomain,
    },
    SpeedLimit {
        min_speed: f32,
        max_speed: f32,
    },
    Source {
        position: ParticleDomain,
        velocity: ParticleDomain,
        rot: ParticleDomain,
        size: ParticleDomain,
        color: ParticleDomain,
        alpha: f32,
        particle_rate: f32,
        age: f32,
        age_sigma: f32,
        parent_vel: [f32; 3],
        parent_motion: f32,
    },
    TargetColor {
        color: [f32; 3],
        alpha: f32,
        scale: f32,
        time_from: f32,
        time_to: f32,
    },
    TargetSize {
        size: [f32; 3],
        scale: [f32; 3],
    },
    TargetRotate {
        did_variant: bool,
        rot: [f32; 3],
        scale: f32,
    },
    TargetVelocity {
        did_variant: bool,
        velocity: [f32; 3],
        scale: f32,
    },
    Vortex {
        center: [f32; 3],
        axis: [f32; 3],
        magnitude: f32,
        epsilon: f32,
        max_radius: f32,
    },
    Turbulence {
        frequency: f32,
        octaves: i32,
        magnitude: f32,
        epsilon: f32,
        offset: [f32; 3],
    },
}

/// One compiled PAPI action entry in `PED_CHUNK_ACTIONLIST`.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledParticleAction {
    pub outer_type: u32,
    pub flags: u32,
    pub inner_type: u32,
    pub kind: ParticleActionKind,
}

impl CompiledParticleAction {
    pub fn action_id(&self) -> u32 {
        match &self.kind {
            ParticleActionKind::Avoid { .. } => 0,
            ParticleActionKind::Bounce { .. } => 1,
            ParticleActionKind::CopyVertexB { .. } => 3,
            ParticleActionKind::Damping { .. } => 4,
            ParticleActionKind::Explosion { .. } => 5,
            ParticleActionKind::Follow { .. } => 6,
            ParticleActionKind::Gravitate { .. } => 7,
            ParticleActionKind::Gravity { .. } => 8,
            ParticleActionKind::Jet { .. } => 9,
            ParticleActionKind::KillOld { .. } => 10,
            ParticleActionKind::MatchVelocity { .. } => 11,
            ParticleActionKind::Move => 12,
            ParticleActionKind::OrbitLine { .. } => 13,
            ParticleActionKind::OrbitPoint { .. } => 14,
            ParticleActionKind::RandomAccel { .. } => 15,
            ParticleActionKind::RandomDisplace { .. } => 16,
            ParticleActionKind::RandomVelocity { .. } => 17,
            ParticleActionKind::Restore { .. } => 18,
            ParticleActionKind::Sink { .. } => 19,
            ParticleActionKind::SinkVelocity { .. } => 20,
            ParticleActionKind::Source { .. } => 21,
            ParticleActionKind::SpeedLimit { .. } => 22,
            ParticleActionKind::TargetColor { .. } => 23,
            ParticleActionKind::TargetSize { .. } => 24,
            ParticleActionKind::TargetRotate { did_variant, .. } => {
                if *did_variant {
                    26
                } else {
                    25
                }
            }
            ParticleActionKind::TargetVelocity { did_variant, .. } => {
                if *did_variant {
                    28
                } else {
                    27
                }
            }
            ParticleActionKind::Vortex { .. } => 29,
            ParticleActionKind::Turbulence { .. } => 30,
            ParticleActionKind::Scatter { .. } => 31,
        }
    }

    pub fn write_binary(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.outer_type.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.inner_type.to_le_bytes());
        match &self.kind {
            ParticleActionKind::Avoid {
                position,
                look_ahead,
                magnitude,
                epsilon,
            } => {
                position.write_binary(out);
                write_f32s(out, &[*look_ahead, *magnitude, *epsilon]);
            }
            ParticleActionKind::Bounce {
                position,
                one_minus_friction,
                resilience,
                cutoff_sqr,
            } => {
                position.write_binary(out);
                write_f32s(out, &[*one_minus_friction, *resilience, *cutoff_sqr]);
            }
            ParticleActionKind::CopyVertexB { copy_pos } => {
                out.extend_from_slice(&copy_pos.to_le_bytes());
            }
            ParticleActionKind::Damping {
                damping,
                v_low_sqr,
                v_high_sqr,
            } => {
                write_f32s(out, damping);
                write_f32s(out, &[*v_low_sqr, *v_high_sqr]);
            }
            ParticleActionKind::Explosion {
                center,
                velocity,
                magnitude,
                stdev,
                age,
                epsilon,
            } => {
                write_f32s(out, center);
                write_f32s(out, &[*velocity, *magnitude, *stdev, *age, *epsilon]);
            }
            ParticleActionKind::Follow {
                magnitude,
                epsilon,
                max_radius,
            }
            | ParticleActionKind::Gravitate {
                magnitude,
                epsilon,
                max_radius,
            }
            | ParticleActionKind::MatchVelocity {
                magnitude,
                epsilon,
                max_radius,
            } => {
                write_f32s(out, &[*magnitude, *epsilon, *max_radius]);
            }
            ParticleActionKind::Gravity { direction } => {
                write_f32s(out, direction);
            }
            ParticleActionKind::Jet {
                center,
                acc,
                magnitude,
                epsilon,
                max_radius,
            } => {
                write_f32s(out, center);
                acc.write_binary(out);
                write_f32s(out, &[*magnitude, *epsilon, *max_radius]);
            }
            ParticleActionKind::KillOld {
                age_limit,
                kill_less_than,
            } => {
                write_f32s(out, &[*age_limit]);
                out.extend_from_slice(&kill_less_than.to_le_bytes());
            }
            ParticleActionKind::Move => {}
            ParticleActionKind::OrbitLine {
                p,
                axis,
                magnitude,
                epsilon,
                max_radius,
            } => {
                write_f32s(out, p);
                write_f32s(out, axis);
                write_f32s(out, &[*magnitude, *epsilon, *max_radius]);
            }
            ParticleActionKind::OrbitPoint {
                center,
                magnitude,
                epsilon,
                max_radius,
            }
            | ParticleActionKind::Scatter {
                center,
                magnitude,
                epsilon,
                max_radius,
            } => {
                write_f32s(out, center);
                write_f32s(out, &[*magnitude, *epsilon, *max_radius]);
            }
            ParticleActionKind::RandomAccel { gen_acc } => gen_acc.write_binary(out),
            ParticleActionKind::RandomDisplace { gen_disp } => gen_disp.write_binary(out),
            ParticleActionKind::RandomVelocity { gen_vel } => gen_vel.write_binary(out),
            ParticleActionKind::Restore { time_left } => write_f32s(out, &[*time_left]),
            ParticleActionKind::Sink {
                kill_inside,
                position,
            } => {
                out.extend_from_slice(&kill_inside.to_le_bytes());
                position.write_binary(out);
            }
            ParticleActionKind::SinkVelocity {
                kill_inside,
                velocity,
            } => {
                out.extend_from_slice(&kill_inside.to_le_bytes());
                velocity.write_binary(out);
            }
            ParticleActionKind::SpeedLimit {
                min_speed,
                max_speed,
            } => {
                write_f32s(out, &[*min_speed, *max_speed]);
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
                parent_vel,
                parent_motion,
            } => {
                position.write_binary(out);
                velocity.write_binary(out);
                rot.write_binary(out);
                size.write_binary(out);
                color.write_binary(out);
                write_f32s(out, &[*alpha, *particle_rate, *age, *age_sigma]);
                write_f32s(out, parent_vel);
                write_f32s(out, &[*parent_motion]);
            }
            ParticleActionKind::TargetColor {
                color,
                alpha,
                scale,
                time_from,
                time_to,
            } => {
                write_f32s(out, color);
                write_f32s(out, &[*alpha, *scale, *time_from, *time_to]);
            }
            ParticleActionKind::TargetSize { size, scale } => {
                write_f32s(out, size);
                write_f32s(out, scale);
            }
            ParticleActionKind::TargetRotate { rot, scale, .. } => {
                write_f32s(out, rot);
                write_f32s(out, &[*scale]);
            }
            ParticleActionKind::TargetVelocity {
                velocity, scale, ..
            } => {
                write_f32s(out, velocity);
                write_f32s(out, &[*scale]);
            }
            ParticleActionKind::Vortex {
                center,
                axis,
                magnitude,
                epsilon,
                max_radius,
            } => {
                write_f32s(out, center);
                write_f32s(out, axis);
                write_f32s(out, &[*magnitude, *epsilon, *max_radius]);
            }
            ParticleActionKind::Turbulence {
                frequency,
                octaves,
                magnitude,
                epsilon,
                offset,
            } => {
                write_f32s(out, &[*frequency]);
                out.extend_from_slice(&octaves.to_le_bytes());
                write_f32s(out, &[*magnitude, *epsilon]);
                write_f32s(out, offset);
            }
        }
    }
}

/// Parse `PED_CHUNK_ACTIONLIST` (`CParticleManager::LoadActions`).
pub fn parse_compiled_action_list(bytes: &[u8]) -> Result<Vec<CompiledParticleAction>, FormatError> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let mut cur = ByteCursor::new(bytes);
    let count = cur.read_u32()? as usize;
    let mut actions = Vec::with_capacity(count.min(4096));
    for _ in 0..count {
        let outer_type = cur.read_u32()?;
        if outer_type == u32::MAX {
            continue;
        }
        let flags = cur.read_u32()?;
        let inner_type = cur.read_u32()?;
        let kind = match outer_type {
            0 => ParticleActionKind::Avoid {
                position: ParticleDomain::read_binary(&mut cur)?,
                look_ahead: cur.read_f32()?,
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
            },
            1 => ParticleActionKind::Bounce {
                position: ParticleDomain::read_binary(&mut cur)?,
                one_minus_friction: cur.read_f32()?,
                resilience: cur.read_f32()?,
                cutoff_sqr: cur.read_f32()?,
            },
            3 => ParticleActionKind::CopyVertexB {
                copy_pos: cur.read_u32()?,
            },
            4 => ParticleActionKind::Damping {
                damping: cur.read_f32_3()?,
                v_low_sqr: cur.read_f32()?,
                v_high_sqr: cur.read_f32()?,
            },
            5 => ParticleActionKind::Explosion {
                center: cur.read_f32_3()?,
                velocity: cur.read_f32()?,
                magnitude: cur.read_f32()?,
                stdev: cur.read_f32()?,
                age: cur.read_f32()?,
                epsilon: cur.read_f32()?,
            },
            6 => ParticleActionKind::Follow {
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                max_radius: cur.read_f32()?,
            },
            7 => ParticleActionKind::Gravitate {
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                max_radius: cur.read_f32()?,
            },
            8 => ParticleActionKind::Gravity {
                direction: cur.read_f32_3()?,
            },
            9 => ParticleActionKind::Jet {
                center: cur.read_f32_3()?,
                acc: ParticleDomain::read_binary(&mut cur)?,
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                max_radius: cur.read_f32()?,
            },
            10 => ParticleActionKind::KillOld {
                age_limit: cur.read_f32()?,
                kill_less_than: cur.read_u32()?,
            },
            11 => ParticleActionKind::MatchVelocity {
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                max_radius: cur.read_f32()?,
            },
            12 => ParticleActionKind::Move,
            13 => ParticleActionKind::OrbitLine {
                p: cur.read_f32_3()?,
                axis: cur.read_f32_3()?,
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                max_radius: cur.read_f32()?,
            },
            14 => ParticleActionKind::OrbitPoint {
                center: cur.read_f32_3()?,
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                max_radius: cur.read_f32()?,
            },
            15 => ParticleActionKind::RandomAccel {
                gen_acc: ParticleDomain::read_binary(&mut cur)?,
            },
            16 => ParticleActionKind::RandomDisplace {
                gen_disp: ParticleDomain::read_binary(&mut cur)?,
            },
            17 => ParticleActionKind::RandomVelocity {
                gen_vel: ParticleDomain::read_binary(&mut cur)?,
            },
            18 => ParticleActionKind::Restore {
                time_left: cur.read_f32()?,
            },
            19 => ParticleActionKind::Sink {
                kill_inside: cur.read_u32()?,
                position: ParticleDomain::read_binary(&mut cur)?,
            },
            20 => ParticleActionKind::SinkVelocity {
                kill_inside: cur.read_u32()?,
                velocity: ParticleDomain::read_binary(&mut cur)?,
            },
            21 => ParticleActionKind::Source {
                position: ParticleDomain::read_binary(&mut cur)?,
                velocity: ParticleDomain::read_binary(&mut cur)?,
                rot: ParticleDomain::read_binary(&mut cur)?,
                size: ParticleDomain::read_binary(&mut cur)?,
                color: ParticleDomain::read_binary(&mut cur)?,
                alpha: cur.read_f32()?,
                particle_rate: cur.read_f32()?,
                age: cur.read_f32()?,
                age_sigma: cur.read_f32()?,
                parent_vel: cur.read_f32_3()?,
                parent_motion: cur.read_f32()?,
            },
            22 => ParticleActionKind::SpeedLimit {
                min_speed: cur.read_f32()?,
                max_speed: cur.read_f32()?,
            },
            23 => ParticleActionKind::TargetColor {
                color: cur.read_f32_3()?,
                alpha: cur.read_f32()?,
                scale: cur.read_f32()?,
                time_from: cur.read_f32()?,
                time_to: cur.read_f32()?,
            },
            24 => ParticleActionKind::TargetSize {
                size: cur.read_f32_3()?,
                scale: cur.read_f32_3()?,
            },
            25 | 26 => ParticleActionKind::TargetRotate {
                did_variant: outer_type == 26,
                rot: cur.read_f32_3()?,
                scale: cur.read_f32()?,
            },
            27 | 28 => ParticleActionKind::TargetVelocity {
                did_variant: outer_type == 28,
                velocity: cur.read_f32_3()?,
                scale: cur.read_f32()?,
            },
            29 => ParticleActionKind::Vortex {
                center: cur.read_f32_3()?,
                axis: cur.read_f32_3()?,
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                max_radius: cur.read_f32()?,
            },
            30 => ParticleActionKind::Turbulence {
                frequency: cur.read_f32()?,
                octaves: cur.read_i32()?,
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                offset: cur.read_f32_3()?,
            },
            31 => ParticleActionKind::Scatter {
                center: cur.read_f32_3()?,
                magnitude: cur.read_f32()?,
                epsilon: cur.read_f32()?,
                max_radius: cur.read_f32()?,
            },
            _ => {
                return Err(FormatError::InvalidValue(
                    "unsupported PAPI action opcode in PED_CHUNK_ACTIONLIST",
                ));
            }
        };
        actions.push(CompiledParticleAction {
            outer_type,
            flags,
            inner_type,
            kind,
        });
    }
    Ok(actions)
}

/// Encode compiled PAPI actions into `PED_CHUNK_ACTIONLIST` bytecode.
pub fn encode_compiled_action_list(actions: &[CompiledParticleAction]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(actions.len() as u32).to_le_bytes());
    for act in actions {
        act.write_binary(&mut out);
    }
    out
}

/// 28-byte sprite animation sheet layout (`PS::SFrame` in `ParticleEffectDef.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleSpriteFrame {
    pub tex_size: [f32; 2],
    pub reserved: [f32; 2],
    pub frame_dim_x: i32,
    pub frame_count: i32,
    pub speed: f32,
}

impl Default for ParticleSpriteFrame {
    fn default() -> Self {
        Self {
            tex_size: [32.0 / 256.0, 64.0 / 128.0],
            reserved: [0.0, 0.0],
            frame_dim_x: 8,
            frame_count: 16,
            speed: 24.0,
        }
    }
}

impl ParticleSpriteFrame {
    pub fn read_binary(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        Ok(Self {
            tex_size: cur.read_f32_2()?,
            reserved: cur.read_f32_2()?,
            frame_dim_x: cur.read_i32()?,
            frame_count: cur.read_i32()?,
            speed: cur.read_f32()?,
        })
    }

    pub fn to_bytes(&self) -> [u8; 28] {
        let mut out = [0u8; 28];
        out[0..4].copy_from_slice(&self.tex_size[0].to_le_bytes());
        out[4..8].copy_from_slice(&self.tex_size[1].to_le_bytes());
        out[8..12].copy_from_slice(&self.reserved[0].to_le_bytes());
        out[12..16].copy_from_slice(&self.reserved[1].to_le_bytes());
        out[16..20].copy_from_slice(&self.frame_dim_x.to_le_bytes());
        out[20..24].copy_from_slice(&self.frame_count.to_le_bytes());
        out[24..28].copy_from_slice(&self.speed.to_le_bytes());
        out
    }

    /// Compute `(uv_min, uv_max)` for `frame` (`SFrame::CalculateTC`).
    pub fn calculate_tc(&self, frame: i32) -> ([f32; 2], [f32; 2]) {
        let dim_x = self.frame_dim_x.max(1);
        let f = frame.max(0);
        let col = (f % dim_x) as f32;
        let row = (f / dim_x) as f32;
        let lt = [col * self.tex_size[0], row * self.tex_size[1]];
        let rb = [lt[0] + self.tex_size[0], lt[1] + self.tex_size[1]];
        (lt, rb)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleCollisionParams {
    pub one_minus_friction: f32,
    pub resilience: f32,
    pub sqr_cutoff: f32,
}

/// Particle Effect Definition (`PS::CPEDef`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleEffectDef {
    pub name: String,
    pub flags: u32,
    pub max_particles: u32,
    pub update_step_ms: u32,
    pub shader: Option<String>,
    pub texture: Option<String>,
    pub frame: Option<ParticleSpriteFrame>,
    pub time_limit: Option<f32>,
    pub collision: Option<ParticleCollisionParams>,
    pub velocity_scale: Option<[f32; 3]>,
    pub align_to_path_rotation: Option<[f32; 3]>,
    pub actions: Vec<CompiledParticleAction>,
}

impl ParticleEffectDef {
    /// Parse a single binary `CPEDef` chunk payload (as stored in `particles.xr` `PS_CHUNK_SECONDGEN`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);

        let ver_bytes = reader.require_chunk(ped_chunks::VERSION)?;
        let version = ByteCursor::new(&ver_bytes).read_u16()?;
        if version != PED_VERSION {
            return Err(FormatError::InvalidValue("unsupported CPEDef version"));
        }

        let name_bytes = reader.require_chunk(ped_chunks::NAME)?;
        let name = ByteCursor::new(&name_bytes).read_zstring()?;

        let eff_bytes = reader.require_chunk(ped_chunks::EFFECTDATA)?;
        let max_particles = ByteCursor::new(&eff_bytes).read_u32()?;

        let act_bytes = reader.require_chunk(ped_chunks::ACTIONLIST)?;
        let actions = parse_compiled_action_list(&act_bytes)?;

        let flags = if let Some(fl_bytes) = reader.open_chunk(ped_chunks::FLAGS)? {
            ByteCursor::new(&fl_bytes).read_u32()?
        } else {
            0
        };

        let (shader, texture) = if (flags & ped_flags::DF_SPRITE) != 0 {
            let spr = reader.require_chunk(ped_chunks::SPRITE)?;
            let mut cur = ByteCursor::new(&spr);
            (Some(cur.read_zstring()?), Some(cur.read_zstring()?))
        } else {
            (None, None)
        };

        let frame = if (flags & ped_flags::DF_FRAMED) != 0 {
            let fr = reader.require_chunk(ped_chunks::FRAME)?;
            Some(ParticleSpriteFrame::read_binary(&mut ByteCursor::new(&fr))?)
        } else {
            None
        };

        let time_limit = if (flags & ped_flags::DF_TIME_LIMIT) != 0 {
            let tl = reader.require_chunk(ped_chunks::TIMELIMIT)?;
            Some(ByteCursor::new(&tl).read_f32()?)
        } else {
            None
        };

        let collision = if (flags & ped_flags::DF_COLLISION) != 0 {
            let col = reader.require_chunk(ped_chunks::COLLISION)?;
            let mut cur = ByteCursor::new(&col);
            Some(ParticleCollisionParams {
                one_minus_friction: cur.read_f32()?,
                resilience: cur.read_f32()?,
                sqr_cutoff: cur.read_f32()?,
            })
        } else {
            None
        };

        let velocity_scale = if (flags & ped_flags::DF_VELOCITY_SCALE) != 0 {
            let vs = reader.require_chunk(ped_chunks::VEL_SCALE)?;
            Some(ByteCursor::new(&vs).read_f32_3()?)
        } else {
            None
        };

        let align_to_path_rotation = if (flags & ped_flags::DF_ALIGN_TO_PATH) != 0 {
            if let Some(atp) = reader.open_chunk(ped_chunks::ALIGN_TO_PATH)? {
                Some(ByteCursor::new(&atp).read_f32_3()?)
            } else {
                Some([-FRAC_PI_2, 0.0, 0.0])
            }
        } else {
            None
        };

        Ok(Self {
            name,
            flags,
            max_particles,
            update_step_ms: 33,
            shader,
            texture,
            frame,
            time_limit,
            collision,
            velocity_scale,
            align_to_path_rotation,
            actions,
        })
    }

    /// Encode this `CPEDef` into X-Ray binary chunk format (`particles.xr` subchunk).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = ChunkWriter::new();
        w.write_chunk(ped_chunks::VERSION, &PED_VERSION.to_le_bytes());

        let mut name_buf = self.name.as_bytes().to_vec();
        name_buf.push(0);
        w.write_chunk(ped_chunks::NAME, &name_buf);

        w.write_chunk(ped_chunks::EFFECTDATA, &self.max_particles.to_le_bytes());
        w.write_chunk(
            ped_chunks::ACTIONLIST,
            &encode_compiled_action_list(&self.actions),
        );
        w.write_chunk(ped_chunks::FLAGS, &self.flags.to_le_bytes());

        if (self.flags & ped_flags::DF_SPRITE) != 0 {
            let mut spr = Vec::new();
            if let Some(s) = &self.shader {
                spr.extend_from_slice(s.as_bytes());
            }
            spr.push(0);
            if let Some(t) = &self.texture {
                spr.extend_from_slice(t.as_bytes());
            }
            spr.push(0);
            w.write_chunk(ped_chunks::SPRITE, &spr);
        }

        if (self.flags & ped_flags::DF_FRAMED) != 0
            && let Some(fr) = &self.frame
        {
            w.write_chunk(ped_chunks::FRAME, &fr.to_bytes());
        }

        if (self.flags & ped_flags::DF_TIME_LIMIT) != 0
            && let Some(tl) = self.time_limit
        {
            w.write_chunk(ped_chunks::TIMELIMIT, &tl.to_le_bytes());
        }

        if (self.flags & ped_flags::DF_COLLISION) != 0
            && let Some(c) = &self.collision
        {
            let mut col = Vec::with_capacity(12);
            write_f32s(&mut col, &[c.one_minus_friction, c.resilience, c.sqr_cutoff]);
            w.write_chunk(ped_chunks::COLLISION, &col);
        }

        if (self.flags & ped_flags::DF_VELOCITY_SCALE) != 0
            && let Some(v) = &self.velocity_scale
        {
            let mut buf = Vec::with_capacity(12);
            write_f32s(&mut buf, v);
            w.write_chunk(ped_chunks::VEL_SCALE, &buf);
        }

        if (self.flags & ped_flags::DF_ALIGN_TO_PATH) != 0
            && let Some(r) = &self.align_to_path_rotation
        {
            let mut buf = Vec::with_capacity(12);
            write_f32s(&mut buf, r);
            w.write_chunk(ped_chunks::ALIGN_TO_PATH, &buf);
        }

        w.into_bytes()
    }

    /// Parse an Anomaly loose `.pe` LTX file (`CPEDef::Load2`) and compile its enabled editor actions.
    pub fn from_pe_ltx(name: &str, ltx_text: &str) -> Result<Self, FormatError> {
        let doc = LtxDocument::parse(ltx_text);
        let view = LtxView::Borrowed(&doc);
        Self::from_pe_view(name, &view)
    }

    /// Parse an Anomaly `.pe` from an `OwnedLtxDocument` tree.
    pub fn from_pe_doc(name: &str, doc: &OwnedLtxDocument) -> Result<Self, FormatError> {
        let view = LtxView::Owned(doc);
        Self::from_pe_view(name, &view)
    }

    fn from_pe_view(name: &str, doc: &LtxView<'_>) -> Result<Self, FormatError> {
        if !doc.contains_section("_effect") {
            return Err(FormatError::InvalidValue("missing [_effect] section in .pe"));
        }

        let update_step_ms = doc.read_u32("_effect", "update_step").unwrap_or(33);
        let max_particles = doc
            .read_u32("_effect", "max_particles")
            .ok_or(FormatError::InvalidValue("missing max_particles in .pe"))?;
        let flags = doc
            .read_u32("_effect", "flags")
            .ok_or(FormatError::InvalidValue("missing flags in .pe"))?;

        let (shader, texture) = if (flags & ped_flags::DF_SPRITE) != 0 {
            (
                Some(doc.read_str("sprite", "shader").unwrap_or("").to_string()),
                Some(doc.read_str("sprite", "texture").unwrap_or("").to_string()),
            )
        } else {
            (None, None)
        };

        let frame = if (flags & ped_flags::DF_FRAMED) != 0 {
            Some(ParticleSpriteFrame {
                tex_size: doc.read_vec2("frame", "tex_size").unwrap_or([1.0, 1.0]),
                reserved: doc.read_vec2("frame", "reserved").unwrap_or([0.0, 0.0]),
                frame_dim_x: doc.read_i32("frame", "dim_x").unwrap_or(1),
                frame_count: doc.read_i32("frame", "frame_count").unwrap_or(1),
                speed: doc.read_f32("frame", "speed").unwrap_or(0.0),
            })
        } else {
            None
        };

        let time_limit = if (flags & ped_flags::DF_TIME_LIMIT) != 0 {
            Some(doc.read_f32("timelimit", "value").unwrap_or(0.0))
        } else {
            None
        };

        let collision = if (flags & ped_flags::DF_COLLISION) != 0 {
            Some(ParticleCollisionParams {
                one_minus_friction: doc.read_f32("collision", "one_minus_friction").unwrap_or(1.0),
                resilience: doc
                    .read_f32("collision", "collide_resilence")
                    .or_else(|| doc.read_f32("collision", "collide_resilience"))
                    .unwrap_or(0.0),
                sqr_cutoff: doc.read_f32("collision", "collide_sqr_cutoff").unwrap_or(0.0),
            })
        } else {
            None
        };

        let velocity_scale = if (flags & ped_flags::DF_VELOCITY_SCALE) != 0 {
            Some(doc.read_vec3("velocity_scale", "value").unwrap_or([0.0; 3]))
        } else {
            None
        };

        let align_to_path_rotation = if (flags & ped_flags::DF_ALIGN_TO_PATH) != 0 {
            Some(
                doc.read_vec3("align_to_path", "default_rotation")
                    .unwrap_or([-FRAC_PI_2, 0.0, 0.0]),
            )
        } else {
            None
        };

        let action_count = doc.read_u32("_effect", "action_count").unwrap_or(0);
        let mut actions = Vec::new();
        for action_id in 0..action_count {
            let sect = format!("action_{action_id:04}");
            if !doc.contains_section(&sect) {
                continue;
            }
            if let Some(compiled) = parse_and_compile_editor_action(doc, &sect)? {
                actions.push(compiled);
            }
        }

        Ok(Self {
            name: name.to_string(),
            flags,
            max_particles,
            update_step_ms,
            shader,
            texture,
            frame,
            time_limit,
            collision,
            velocity_scale,
            align_to_path_rotation,
            actions,
        })
    }
}

/// Child effect entry inside a `ParticleGroupDef` (`CPGDef::SEffect`).
#[derive(Debug, Clone, PartialEq)]
pub struct GroupEffectItem {
    pub effect_name: String,
    pub on_play_child: String,
    pub on_birth_child: String,
    pub on_dead_child: String,
    pub time0: f32,
    pub time1: f32,
    pub flags: u32,
}

/// Particle Group Definition (`PS::CPGDef`).
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleGroupDef {
    pub name: String,
    pub flags: u32,
    pub time_limit: f32,
    pub effects: Vec<GroupEffectItem>,
}

impl ParticleGroupDef {
    /// Parse a single binary `CPGDef` chunk payload (`particles.xr` `PS_CHUNK_THIRDGEN`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);

        let ver_bytes = reader.require_chunk(pgd_chunks::VERSION)?;
        let version = ByteCursor::new(&ver_bytes).read_u16()?;
        if version != PGD_VERSION {
            return Err(FormatError::InvalidValue("unsupported CPGDef version"));
        }

        let name_bytes = reader.require_chunk(pgd_chunks::NAME)?;
        let name = ByteCursor::new(&name_bytes).read_zstring()?;

        let flags = if let Some(fl_bytes) = reader.open_chunk(pgd_chunks::FLAGS)? {
            ByteCursor::new(&fl_bytes).read_u32()?
        } else {
            0
        };

        let mut time_limit = if let Some(tl_bytes) = reader.open_chunk(pgd_chunks::TIME_LIMIT)? {
            ByteCursor::new(&tl_bytes).read_f32()?
        } else {
            0.0
        };
        let dont_calc_timelimit = time_limit > 0.0;

        let mut effects = Vec::new();
        if let Some(eff_bytes) = reader.open_chunk(pgd_chunks::EFFECTS)? {
            let mut cur = ByteCursor::new(&eff_bytes);
            let count = cur.read_u32()? as usize;
            effects.reserve(count.min(1024));
            for _ in 0..count {
                let effect_name = cur.read_zstring()?;
                let on_play_child = cur.read_zstring()?;
                let on_birth_child = cur.read_zstring()?;
                let on_dead_child = cur.read_zstring()?;
                let time0 = cur.read_f32()?;
                let time1 = cur.read_f32()?;
                let eff_flags = cur.read_u32()?;
                if !dont_calc_timelimit {
                    time_limit = time_limit.max(time1);
                }
                effects.push(GroupEffectItem {
                    effect_name,
                    on_play_child,
                    on_birth_child,
                    on_dead_child,
                    time0,
                    time1,
                    flags: eff_flags,
                });
            }
        }

        Ok(Self {
            name,
            flags,
            time_limit,
            effects,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = ChunkWriter::new();
        w.write_chunk(pgd_chunks::VERSION, &PGD_VERSION.to_le_bytes());

        let mut name_buf = self.name.as_bytes().to_vec();
        name_buf.push(0);
        w.write_chunk(pgd_chunks::NAME, &name_buf);
        w.write_chunk(pgd_chunks::FLAGS, &self.flags.to_le_bytes());

        let mut effs = Vec::new();
        effs.extend_from_slice(&(self.effects.len() as u32).to_le_bytes());
        for eff in &self.effects {
            for s in [
                &eff.effect_name,
                &eff.on_play_child,
                &eff.on_birth_child,
                &eff.on_dead_child,
            ] {
                effs.extend_from_slice(s.as_bytes());
                effs.push(0);
            }
            write_f32s(&mut effs, &[eff.time0, eff.time1]);
            effs.extend_from_slice(&eff.flags.to_le_bytes());
        }
        w.write_chunk(pgd_chunks::EFFECTS, &effs);
        w.write_chunk(pgd_chunks::TIME_LIMIT, &self.time_limit.to_le_bytes());

        w.into_bytes()
    }

    /// Parse an Anomaly loose `.pg` LTX file (`CPGDef::Load2`).
    pub fn from_pg_ltx(name: &str, ltx_text: &str) -> Result<Self, FormatError> {
        let doc = LtxDocument::parse(ltx_text);
        let view = LtxView::Borrowed(&doc);
        Self::from_pg_view(name, &view)
    }

    pub fn from_pg_doc(name: &str, doc: &OwnedLtxDocument) -> Result<Self, FormatError> {
        let view = LtxView::Owned(doc);
        Self::from_pg_view(name, &view)
    }

    fn from_pg_view(name: &str, doc: &LtxView<'_>) -> Result<Self, FormatError> {
        if !doc.contains_section("_group") {
            return Err(FormatError::InvalidValue("missing [_group] section in .pg"));
        }

        let flags = doc.read_u32("_group", "flags").unwrap_or(0);
        let effects_count = doc.read_u32("_group", "effects_count").unwrap_or(0);
        let time_limit = doc.read_f32("_group", "timelimit").unwrap_or(0.0);

        let mut effects = Vec::with_capacity(effects_count as usize);
        for idx in 0..effects_count {
            let sect = format!("effect_{idx:04}");
            effects.push(GroupEffectItem {
                effect_name: doc
                    .read_str(&sect, "effect_name")
                    .unwrap_or("")
                    .to_string(),
                on_play_child: doc
                    .read_str(&sect, "on_play_child")
                    .unwrap_or("")
                    .to_string(),
                on_birth_child: doc
                    .read_str(&sect, "on_birth_child")
                    .unwrap_or("")
                    .to_string(),
                on_dead_child: doc
                    .read_str(&sect, "on_death_child")
                    .unwrap_or("")
                    .to_string(),
                time0: doc.read_f32(&sect, "time0").unwrap_or(0.0),
                time1: doc.read_f32(&sect, "time1").unwrap_or(0.0),
                flags: doc.read_u32(&sect, "flags").unwrap_or(0),
            });
        }

        Ok(Self {
            name: name.to_string(),
            flags,
            time_limit,
            effects,
        })
    }
}

/// Complete particle library (`CPSLibrary`), sorted by name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ParticlesLibrary {
    pub effects: Vec<ParticleEffectDef>,
    pub groups: Vec<ParticleGroupDef>,
}

impl ParticlesLibrary {
    /// Parse binary `$game_data$/particles.xr` (`CPSLibrary::Load`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let root = ChunkReader::new(data);
        let ver_bytes = root.require_chunk(ps_chunks::VERSION)?;
        let ver = ByteCursor::new(&ver_bytes).read_u16()?;
        if ver != PS_VERSION {
            return Err(FormatError::InvalidValue("unsupported particles.xr version"));
        }

        let mut effects = Vec::new();
        if let Some(sec_bytes) = root.open_chunk(ps_chunks::SECONDGEN)? {
            for entry in ChunkReader::new(&sec_bytes).iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                effects.push(ParticleEffectDef::parse(&payload)?);
            }
        }

        let mut groups = Vec::new();
        if let Some(thd_bytes) = root.open_chunk(ps_chunks::THIRDGEN)? {
            for entry in ChunkReader::new(&thd_bytes).iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                groups.push(ParticleGroupDef::parse(&payload)?);
            }
        }

        effects.sort_by(|a, b| a.name.cmp(&b.name));
        groups.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Self { effects, groups })
    }

    /// Encode this library into binary `particles.xr` format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut root = ChunkWriter::new();
        root.write_chunk(ps_chunks::VERSION, &PS_VERSION.to_le_bytes());

        let mut effs = ChunkWriter::new();
        for (i, eff) in self.effects.iter().enumerate() {
            effs.write_chunk(i as u32, &eff.to_bytes());
        }
        root.write_chunk(ps_chunks::SECONDGEN, &effs.into_bytes());

        let mut grps = ChunkWriter::new();
        for (i, grp) in self.groups.iter().enumerate() {
            grps.write_chunk(i as u32, &grp.to_bytes());
        }
        root.write_chunk(ps_chunks::THIRDGEN, &grps.into_bytes());

        root.into_bytes()
    }

    /// Merge loose `.pe`/`.pg` overrides on top of `particles.xr` (`CPSLibrary::Load` precedence:
    /// loose definitions with the same name replace compiled ones, then sorted by name).
    pub fn merge_loose(
        &mut self,
        loose_effects: Vec<ParticleEffectDef>,
        loose_groups: Vec<ParticleGroupDef>,
    ) {
        if !loose_effects.is_empty() {
            let mut by_name: HashMap<String, ParticleEffectDef> =
                HashMap::with_capacity(self.effects.len() + loose_effects.len());
            for eff in self.effects.drain(..) {
                by_name.insert(eff.name.clone(), eff);
            }
            for eff in loose_effects {
                by_name.insert(eff.name.clone(), eff);
            }
            self.effects = by_name.into_values().collect();
        }
        if !loose_groups.is_empty() {
            let mut by_name: HashMap<String, ParticleGroupDef> =
                HashMap::with_capacity(self.groups.len() + loose_groups.len());
            for grp in self.groups.drain(..) {
                by_name.insert(grp.name.clone(), grp);
            }
            for grp in loose_groups {
                by_name.insert(grp.name.clone(), grp);
            }
            self.groups = by_name.into_values().collect();
        }
        self.effects.sort_by(|a, b| a.name.cmp(&b.name));
        self.groups.sort_by(|a, b| a.name.cmp(&b.name));
    }

    pub fn find_effect(&self, name: &str) -> Option<&ParticleEffectDef> {
        self.effects
            .binary_search_by(|e| e.name.as_str().cmp(name))
            .ok()
            .map(|i| &self.effects[i])
    }

    pub fn find_group(&self, name: &str) -> Option<&ParticleGroupDef> {
        self.groups
            .binary_search_by(|g| g.name.as_str().cmp(name))
            .ok()
            .map(|i| &self.groups[i])
    }
}

/// Wallmark texture slot configuration (CSV textures rendered via `effects\wallmark`).
#[derive(Debug, Clone, PartialEq)]
pub struct WallmarkSlotConfig {
    pub shader: String,
    pub textures: Vec<String>,
}

/// Entity blood-splatter & blood-drop wallmark LTX configuration (`CEntityAlive::LoadBloodyWallmarks`).
#[derive(Debug, Clone, PartialEq)]
pub struct BloodWallmarksConfig {
    pub wallmarks: WallmarkSlotConfig,
    pub blood_drops: WallmarkSlotConfig,
    pub min_size: f32,
    pub max_size: f32,
    pub dist: f32,
    pub nominal_hit: f32,
    pub start_blood_size: f32,
    pub stop_blood_size: f32,
    pub blood_drop_size: f32,
}

impl BloodWallmarksConfig {
    pub fn from_ltx_str(ltx_text: &str, section: &str) -> Result<Self, FormatError> {
        let doc = LtxDocument::parse(ltx_text);
        let view = LtxView::Borrowed(&doc);
        Self::from_ltx_view(&view, section)
    }

    pub fn from_ltx_doc(doc: &OwnedLtxDocument, section: &str) -> Result<Self, FormatError> {
        let view = LtxView::Owned(doc);
        Self::from_ltx_view(&view, section)
    }

    fn from_ltx_view(doc: &LtxView<'_>, section: &str) -> Result<Self, FormatError> {
        if !doc.contains_section(section) {
            return Err(FormatError::InvalidValue("missing wallmark config section"));
        }
        let split_csv = |s: &str| -> Vec<String> {
            s.split(',')
                .map(|part| part.trim().trim_matches('"').to_string())
                .filter(|part| !part.is_empty())
                .collect()
        };

        let wm_csv = doc.read_str(section, "wallmarks").unwrap_or("");
        let drop_csv = doc.read_str(section, "blood_drops").unwrap_or("");

        Ok(Self {
            wallmarks: WallmarkSlotConfig {
                shader: "effects\\wallmark".to_string(),
                textures: split_csv(wm_csv),
            },
            blood_drops: WallmarkSlotConfig {
                shader: "effects\\wallmark".to_string(),
                textures: split_csv(drop_csv),
            },
            min_size: doc.read_f32(section, "min_size").unwrap_or(0.5),
            max_size: doc.read_f32(section, "max_size").unwrap_or(1.5),
            dist: doc.read_f32(section, "dist").unwrap_or(2.0),
            nominal_hit: doc.read_f32(section, "nominal_hit").unwrap_or(0.5),
            start_blood_size: doc.read_f32(section, "start_blood_size").unwrap_or(0.4),
            stop_blood_size: doc.read_f32(section, "stop_blood_size").unwrap_or(0.025),
            blood_drop_size: doc.read_f32(section, "blood_drop_size").unwrap_or(0.1),
        })
    }
}

/// Runtime static decal polygon record (`CWallmarksEngine::static_wallmark`).
#[derive(Debug, Clone, PartialEq)]
pub struct StaticWallmarkRecord {
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
    pub shader: String,
    pub texture: String,
    pub time_start: f32,
    pub time_end: f32,
    pub vertices: Vec<WallmarkVertex>,
}

/// Projected wallmark vertex (`FVF::LIT`: position, packed ARGB diffuse, UV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallmarkVertex {
    pub position: [f32; 3],
    pub color: u32,
    pub uv: [f32; 2],
}

pub const WALLMARK_RANGE_STATIC: f32 = 100.0;
pub const WALLMARK_RANGE_SKELETON: f32 = 50.0;
pub const WALLMARK_DEFAULT_TTL: f32 = 50.0;
pub const WALLMARK_SHIFT_PP: f32 = 0.0001;
pub const WALLMARK_SHIFT_V: f32 = 0.0001;
pub const WALLMARK_COS_88_DEG: f32 = 0.034899;
pub const WALLMARK_DUPLICATE_TOLERANCE: f32 = 0.02;
const WALLMARK_PACK_EPS: f32 = 1e-5;

/// Parameters controlling static wallmark decal projection (`CWallmarksEngine::AddStaticWallmark`).
#[derive(Debug, Clone, PartialEq)]
pub struct StaticWallmarkConfig {
    pub shader: String,
    pub texture: String,
    /// Decal half-extent in meters (`sz`).
    pub size: f32,
    /// Lifetime in seconds: `0.0` uses `WALLMARK_DEFAULT_TTL * 15.0` (750s), `< 0.0` is infinite (`-1.0`).
    pub ttl: f32,
    /// Random or explicit decal roll around contact normal in degrees.
    pub rotation_deg: f32,
    /// Creation timestamp (`Device.fTimeGlobal`).
    pub time_now: f32,
    /// Skip `WALLMARK_RANGE_STATIC` (100m) distance cull when `true`.
    pub ignore_range_opt: bool,
    /// Optional camera world position for distance optimization culling.
    pub camera_pos: Option<[f32; 3]>,
    /// Skip triangles with `CformTriangle::suppress_wm == true`.
    pub respect_suppress_wm: bool,
}

impl Default for StaticWallmarkConfig {
    fn default() -> Self {
        Self {
            shader: "effects\\wallmark".to_string(),
            texture: String::new(),
            size: 0.25,
            ttl: 0.0,
            rotation_deg: 0.0,
            time_now: 0.0,
            ignore_range_opt: false,
            camera_pos: None,
            respect_suppress_wm: true,
        }
    }
}

/// Weld triangle vertex positions within `eps` matching `CDB::Collector::add_face_packed_D`.
pub fn pack_triangle_vertices(
    tris: &[([f32; 3], [f32; 3], [f32; 3])],
    eps: f32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let tol = eps.max(1e-6);
    let cell_size = tol * 4.0;
    let inv_cell = 1.0 / cell_size;
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::with_capacity(tris.len() * 2);
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(tris.len() * 2);
    let mut out_tris = Vec::with_capacity(tris.len());

    let mut intern = |p: [f32; 3]| -> u32 {
        let cx = (p[0] * inv_cell).floor() as i64;
        let cy = (p[1] * inv_cell).floor() as i64;
        let cz = (p[2] * inv_cell).floor() as i64;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &idx in bucket {
                            let q = verts[idx as usize];
                            if (q[0] - p[0]).abs() <= tol
                                && (q[1] - p[1]).abs() <= tol
                                && (q[2] - p[2]).abs() <= tol
                            {
                                return idx;
                            }
                        }
                    }
                }
            }
        }
        let idx = verts.len() as u32;
        verts.push(p);
        grid.entry((cx, cy, cz)).or_default().push(idx);
        idx
    };

    for &(p0, p1, p2) in tris {
        out_tris.push([intern(p0), intern(p1), intern(p2)]);
    }
    (verts, out_tris)
}

/// Build per-triangle edge adjacency matching `CDB::Collector::calc_adjacency` (`xrCDB_Collector.cpp`).
pub fn calc_triangle_adjacency(tris: &[[u32; 3]]) -> Vec<[Option<u32>; 3]> {
    #[derive(Clone, Copy)]
    struct EdgeRec {
        v0: u32,
        v1: u32,
        face_id: u32,
        edge_id: u8,
    }

    let mut edges = Vec::with_capacity(tris.len() * 3);
    for (face_idx, &tri) in tris.iter().enumerate() {
        let face_id = face_idx as u32;
        for e in 0..3 {
            let a = tri[e];
            let b = tri[(e + 1) % 3];
            let (v0, v1) = if a <= b { (a, b) } else { (b, a) };
            edges.push(EdgeRec {
                v0,
                v1,
                face_id,
                edge_id: e as u8,
            });
        }
    }

    edges.sort_unstable_by(|a, b| {
        a.v0.cmp(&b.v0)
            .then_with(|| a.v1.cmp(&b.v1))
            .then_with(|| a.face_id.cmp(&b.face_id))
    });

    let mut adj = vec![[None; 3]; tris.len()];
    for pair in edges.windows(2) {
        let e0 = pair[0];
        let e1 = pair[1];
        if e0.v0 == e1.v0 && e0.v1 == e1.v1 && e0.face_id != e1.face_id {
            adj[e0.face_id as usize][e0.edge_id as usize] = Some(e1.face_id);
            adj[e1.face_id as usize][e1.edge_id as usize] = Some(e0.face_id);
        }
    }
    adj
}

/// Build the orthographic decal projection matrix matching `CWallmarksEngine::BuildMatrix`
/// + Z-rotation (`WallmarksEngine.cpp:193-206, 244-247`).
pub fn build_wallmark_view_matrix(
    normal: [f32; 3],
    size: f32,
    contact_point: [f32; 3],
    rotation_deg: f32,
) -> XRayMatrix {
    let n = vec3_normalize_safe(normal);
    let n = if vec3_len(n) <= 1e-6 {
        [0.0, 0.0, 1.0]
    } else {
        n
    };
    let inv_sz = 1.0 / size.max(1e-4);
    let at = vec3_sub(contact_point, n);
    let y = if n[1].abs() > 0.99 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let right = vec3_cross(y, n);
    let up = vec3_cross(n, right);

    let mut m_view = matrix_build_camera(contact_point, at, up);
    m_view = matrix_mul_43(&matrix_scale(inv_sz, inv_sz, inv_sz), &m_view);
    if rotation_deg.abs() > 1e-6 {
        m_view = matrix_mul_43(&matrix_rotate_z(rotation_deg.to_radians()), &m_view);
    }
    m_view
}

#[inline]
fn tri_mknormal(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    let e0 = vec3_sub(p1, p0);
    let e1 = vec3_sub(p2, p1);
    vec3_normalize_safe(vec3_cross(e0, e1))
}

fn project_packed_wallmark(
    verts: &[[f32; 3]],
    tris: &[[u32; 3]],
    seed_tri_idx: usize,
    contact_point: [f32; 3],
    cfg: &StaticWallmarkConfig,
) -> Option<StaticWallmarkRecord> {
    if cfg.size <= 1e-4 || seed_tri_idx >= tris.len() {
        return None;
    }
    if !cfg.ignore_range_opt
        && let Some(cam) = cfg.camera_pos
    {
        let d = vec3_sub(contact_point, cam);
        if vec3_dot(d, d) > WALLMARK_RANGE_STATIC * WALLMARK_RANGE_STATIC {
            return None;
        }
    }

    let seed = tris[seed_tri_idx];
    let sml_normal = tri_mknormal(
        verts[seed[0] as usize],
        verts[seed[1] as usize],
        verts[seed[2] as usize],
    );
    if vec3_len(sml_normal) <= 1e-6 {
        return None;
    }

    let m_view = build_wallmark_view_matrix(sml_normal, cfg.size, contact_point, cfg.rotation_deg);
    let clipper = ClipFrustum::from_view_proj(&m_view, frustum_planes::LRTB);
    let adj = calc_triangle_adjacency(tris);

    let mut visited = vec![false; tris.len()];
    let mut stack = Vec::with_capacity(32);
    stack.push(seed_tri_idx);
    let mut out_verts = Vec::with_capacity(24);
    let base_color = pack_argb(0, 128, 128, 128);

    while let Some(t) = stack.pop() {
        if visited[t] {
            continue;
        }
        visited[t] = true;

        let tri = tris[t];
        let poly = [
            verts[tri[0] as usize],
            verts[tri[1] as usize],
            verts[tri[2] as usize],
        ];
        if let Some(clipped) = clipper.clip_poly(&poly) {
            let p0 = clipped[0];
            let uv0 = matrix_transform_point(&m_view, p0);
            let v0 = WallmarkVertex {
                position: p0,
                color: base_color,
                uv: [(1.0 + uv0[0]) * 0.5, (1.0 - uv0[1]) * 0.5],
            };

            let p1 = clipped[1];
            let uv1 = matrix_transform_point(&m_view, p1);
            let mut v1 = WallmarkVertex {
                position: p1,
                color: base_color,
                uv: [(1.0 + uv1[0]) * 0.5, (1.0 - uv1[1]) * 0.5],
            };

            for &pi in &clipped[2..] {
                let uvi = matrix_transform_point(&m_view, pi);
                let v2 = WallmarkVertex {
                    position: pi,
                    color: base_color,
                    uv: [(1.0 + uvi[0]) * 0.5, (1.0 - uvi[1]) * 0.5],
                };
                out_verts.push(v0);
                out_verts.push(v1);
                out_verts.push(v2);
                v1 = v2;
            }

            for adj_idx in adj[t].into_iter().flatten() {
                let idx = adj_idx as usize;
                if visited[idx] {
                    continue;
                }
                let nt = tris[idx];
                let test_normal = tri_mknormal(
                    verts[nt[0] as usize],
                    verts[nt[1] as usize],
                    verts[nt[2] as usize],
                );
                if vec3_dot(test_normal, sml_normal) >= WALLMARK_COS_88_DEG {
                    stack.push(idx);
                }
            }
        }
    }

    if out_verts.len() < 3 {
        return None;
    }

    let mut bmin = out_verts[0].position;
    let mut bmax = out_verts[0].position;
    for v in &out_verts[1..] {
        for axis in 0..3 {
            bmin[axis] = bmin[axis].min(v.position[axis]);
            bmax[axis] = bmax[axis].max(v.position[axis]);
        }
    }
    let bounds_center = [
        (bmin[0] + bmax[0]) * 0.5,
        (bmin[1] + bmax[1]) * 0.5,
        (bmin[2] + bmax[2]) * 0.5,
    ];
    let half_diag = vec3_sub(bmax, bounds_center);
    let bounds_radius = vec3_len(half_diag);
    let time_end = if cfg.ttl < 0.0 {
        -1.0
    } else if cfg.ttl == 0.0 {
        WALLMARK_DEFAULT_TTL * 15.0
    } else {
        cfg.ttl
    };

    Some(StaticWallmarkRecord {
        bounds_center,
        bounds_radius,
        shader: cfg.shader.clone(),
        texture: cfg.texture.clone(),
        time_start: cfg.time_now,
        time_end,
        vertices: out_verts,
    })
}

/// Project a static decal polygon onto an arbitrary triangle list starting from `seed_tri_index`.
pub fn project_static_wallmark_from_tris(
    tris: &[([f32; 3], [f32; 3], [f32; 3])],
    seed_tri_index: usize,
    contact_point: [f32; 3],
    cfg: &StaticWallmarkConfig,
) -> Option<StaticWallmarkRecord> {
    if seed_tri_index >= tris.len() {
        return None;
    }
    let (verts, packed_tris) = pack_triangle_vertices(tris, WALLMARK_PACK_EPS);
    project_packed_wallmark(&verts, &packed_tris, seed_tri_index, contact_point, cfg)
}

/// Project a static decal polygon against a `level.cform` BVH (`CWallmarksEngine::AddWallmark_internal`).
pub fn project_static_wallmark_on_cform(
    model: &CformModel,
    bvh: &CformBvh<'_>,
    seed_tri_index: usize,
    contact_point: [f32; 3],
    cfg: &StaticWallmarkConfig,
) -> Option<StaticWallmarkRecord> {
    let seed_tri = model.triangles.get(seed_tri_index)?;
    if cfg.respect_suppress_wm && seed_tri.suppress_wm {
        return None;
    }
    if cfg.size <= 1e-4 {
        return None;
    }
    if !cfg.ignore_range_opt
        && let Some(cam) = cfg.camera_pos
    {
        let d = vec3_sub(contact_point, cam);
        if vec3_dot(d, d) > WALLMARK_RANGE_STATIC * WALLMARK_RANGE_STATIC {
            return None;
        }
    }

    let half_q = cfg.size * 2.5;
    let hits = bvh.box_query(contact_point, [half_q, half_q, half_q], true, false);
    if hits.is_empty() {
        return None;
    }

    let mut raw_tris = Vec::with_capacity(hits.len() + 1);
    raw_tris.push((
        model.vertices[seed_tri.verts[0] as usize],
        model.vertices[seed_tri.verts[1] as usize],
        model.vertices[seed_tri.verts[2] as usize],
    ));
    for h in hits {
        if h.tri_index == seed_tri_index {
            continue;
        }
        if cfg.respect_suppress_wm && model.triangles[h.tri_index].suppress_wm {
            continue;
        }
        raw_tris.push((h.verts[0], h.verts[1], h.verts[2]));
    }

    let (verts, packed_tris) = pack_triangle_vertices(&raw_tris, WALLMARK_PACK_EPS);
    project_packed_wallmark(&verts, &packed_tris, 0, contact_point, cfg)
}

/// Flattened static wallmark render batch grouped by `(shader, texture)`.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticWallmarkBatch {
    pub shader: String,
    pub texture: String,
    pub vertices: Vec<WallmarkVertex>,
}

/// Active static decal pool managing slots, duplicate merge, TTL expiry, and frustum/SSA emission.
#[derive(Debug, Clone, Default)]
pub struct StaticWallmarkPool {
    items: Vec<StaticWallmarkRecord>,
}

impl StaticWallmarkPool {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    pub fn items(&self) -> &[StaticWallmarkRecord] {
        &self.items
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Add or replace an overlapping wallmark within `0.02m` (`WallmarksEngine.cpp:280-298`).
    pub fn add(&mut self, record: StaticWallmarkRecord) {
        for existing in &mut self.items {
            if existing.shader == record.shader
                && existing.texture == record.texture
                && (existing.bounds_center[0] - record.bounds_center[0]).abs()
                    < WALLMARK_DUPLICATE_TOLERANCE
                && (existing.bounds_center[1] - record.bounds_center[1]).abs()
                    < WALLMARK_DUPLICATE_TOLERANCE
                && (existing.bounds_center[2] - record.bounds_center[2]).abs()
                    < WALLMARK_DUPLICATE_TOLERANCE
            {
                *existing = record;
                return;
            }
        }
        self.items.push(record);
    }

    /// Remove finite-lifetime wallmarks whose age ratio reached `1.0`, returning count pruned.
    pub fn prune_expired(&mut self, time_now: f32) -> usize {
        let before = self.items.len();
        self.items.retain(|wm| {
            if wm.time_end < 0.0 {
                return true;
            }
            let w = (time_now - wm.time_start) / wm.time_end.max(1e-6);
            w < 1.0
        });
        before - self.items.len()
    }

    /// Frustum + SSA cull (`ssa >= ssa_discard / 4`), prune expired decals, and emit vertex batches
    /// matching `CWallmarksEngine::Render` (`WallmarksEngine.cpp:407-456`).
    ///
    /// `ssa_discard` is the normalized `r_ssaDISCARD` threshold (`ps_r__ssaDISCARD^2 / g_fSCREEN`,
    /// e.g. via [`crate::details::compute_r_ssa_discard`]).
    pub fn collect_visible_batches(
        &mut self,
        time_now: f32,
        camera_pos: [f32; 3],
        frustum: &ClipFrustum,
        ssa_discard: f32,
    ) -> Vec<StaticWallmarkBatch> {
        let ssa_clip = ssa_discard * 0.25;
        let mut batches: Vec<StaticWallmarkBatch> = Vec::new();
        let mut idx = 0usize;

        while idx < self.items.len() {
            let wm = &self.items[idx];
            if frustum.test_sphere(wm.bounds_center, wm.bounds_radius) {
                let d = vec3_sub(camera_pos, wm.bounds_center);
                let dist_sq = vec3_dot(d, d).max(1e-6);
                let ssa = (wm.bounds_radius * wm.bounds_radius) / dist_sq;
                if ssa >= ssa_clip {
                    let a = if wm.time_end < 0.0 {
                        0.0
                    } else {
                        ((time_now - wm.time_start) / wm.time_end.max(1e-6)).clamp(0.0, 1.0)
                    };
                    let a_c = ((a * 255.0).floor() as i32).clamp(0, 255) as u8;
                    let color = pack_argb(a_c, 128, 128, 128);

                    let batch_idx = if let Some(pos) = batches
                        .iter()
                        .position(|b| b.shader == wm.shader && b.texture == wm.texture)
                    {
                        pos
                    } else {
                        batches.push(StaticWallmarkBatch {
                            shader: wm.shader.clone(),
                            texture: wm.texture.clone(),
                            vertices: Vec::with_capacity(wm.vertices.len()),
                        });
                        batches.len() - 1
                    };
                    let out = &mut batches[batch_idx].vertices;
                    for v in &wm.vertices {
                        out.push(WallmarkVertex {
                            position: v.position,
                            color,
                            uv: v.uv,
                        });
                    }
                }
            }

            if wm.time_end >= 0.0 {
                let w = (time_now - wm.time_start) / wm.time_end.max(1e-6);
                if w >= 1.0 {
                    self.items.swap_remove(idx);
                    continue;
                }
            }
            idx += 1;
        }

        batches
    }
}

/// Bind-pose skinned wallmark triangle (`CSkeletonWallmark::WMFace` in `SkeletonCustom.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkeletonWallmarkFace {
    /// Original unskinned bind-pose vertex positions.
    pub verts: [[f32; 3]; 3],
    /// Sentinel-padded per-vertex bone indices (`SkeletonX.cpp:580-724`).
    pub bone_ids: [[u16; 4]; 3],
    /// Per-vertex blend weights (unused trailing entries zero-filled).
    pub weights: [[f32; 3]; 3],
    /// Explicit skinning link counts per corner (`0..=4`) preventing sentinel ambiguity on repeated bone IDs.
    pub link_counts: [u8; 3],
    /// Projected decal texture coordinates computed at impact pose.
    pub uvs: [[f32; 2]; 3],
}

/// Parameters controlling skeletal decal projection (`CKinematics::AddWallmark`).
#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonWallmarkConfig {
    /// Optional caller entity identifier isolating duplicate merge and pose lookup across multiple actors.
    pub object_id: u64,
    pub shader: String,
    pub texture: String,
    /// Decal sphere radius in meters (`sz`).
    pub size: f32,
    /// Lifetime in seconds: `0.0` uses `WALLMARK_DEFAULT_TTL` (50s, matching `ps_r__WallmarkTTL`),
    /// `< 0.0` is infinite (`-1.0`).
    pub ttl: f32,
    /// Decal roll angle around projection normal in degrees (`Random[-20..20]` in C++).
    pub rotation_deg: f32,
    /// Creation timestamp (`Device.fTimeGlobal`).
    pub time_now: f32,
    /// Skip `WALLMARK_RANGE_SKELETON` (50m) distance cull when `true`.
    pub ignore_range_opt: bool,
    /// Optional camera world position for 50m distance optimization culling.
    pub camera_pos: Option<[f32; 3]>,
    /// Maximum ray distance for bone picking (`flt_max` in C++).
    pub max_dist: f32,
}

impl Default for SkeletonWallmarkConfig {
    fn default() -> Self {
        Self {
            object_id: 0,
            shader: "effects\\wallmark".to_string(),
            texture: String::new(),
            size: 0.25,
            ttl: 0.0,
            rotation_deg: 0.0,
            time_now: 0.0,
            ignore_range_opt: false,
            camera_pos: None,
            max_dist: f32::MAX,
        }
    }
}

/// Active skeletal decal attached to a rigged model in bind pose (`CSkeletonWallmark`).
#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonWallmarkRecord {
    pub object_id: u64,
    /// Model-space contact sphere center (`m_LocalBounds.P`).
    pub local_center: [f32; 3],
    /// Model-space bounding sphere radius (`m_LocalBounds.R`, `size * 2.0`).
    pub local_radius: f32,
    pub shader: String,
    pub texture: String,
    pub time_start: f32,
    pub time_end: f32,
    pub faces: Vec<SkeletonWallmarkFace>,
}

impl SkeletonWallmarkRecord {
    /// World-space bounding sphere `(m_Bounds.P, m_Bounds.R)` transformed by current `parent_xform`.
    #[inline]
    pub fn world_bounds(&self, parent_xform: &XRayMatrix) -> ([f32; 3], f32) {
        (
            matrix_transform_point(parent_xform, self.local_center),
            self.local_radius,
        )
    }

    /// True if this decal matches `(object_id, shader, texture)` and sits within `eps` per axis
    /// of `cp` (`CSkeletonWallmark::Similar`, `SkeletonCustom.cpp:779`).
    #[inline]
    pub fn similar(
        &self,
        object_id: u64,
        shader: &str,
        texture: &str,
        cp: [f32; 3],
        eps: f32,
    ) -> bool {
        self.object_id == object_id
            && self.shader == shader
            && self.texture == texture
            && (self.local_center[0] - cp[0]).abs() < eps
            && (self.local_center[1] - cp[1]).abs() < eps
            && (self.local_center[2] - cp[2]).abs() < eps
    }
}

/// Pick and project a dynamic decal onto an animated skeletal model matching `CKinematics::AddWallmark`
/// (`SkeletonCustom.cpp:714-813`) and `CSkeletonX::_FillVerticesSoft1W..4W` (`SkeletonX.cpp:568-743`).
pub fn project_skeleton_wallmark(
    index: &SkeletonCollisionIndex<'_>,
    palette: &BonePalette,
    parent_xform: &XRayMatrix,
    visible_bones: &BoneMask,
    start_world: [f32; 3],
    dir_world: [f32; 3],
    cfg: &SkeletonWallmarkConfig,
) -> Option<SkeletonWallmarkRecord> {
    if cfg.size <= 1e-4 {
        return None;
    }

    if !cfg.ignore_range_opt
        && let Some(cam) = cfg.camera_pos
    {
        let origin = [parent_xform[3][0], parent_xform[3][1], parent_xform[3][2]];
        let d = vec3_sub(origin, cam);
        if vec3_dot(d, d) > WALLMARK_RANGE_SKELETON * WALLMARK_RANGE_SKELETON {
            return None;
        }
    }

    let inv_parent = matrix_invert_43(parent_xform)?;
    let s = matrix_transform_point(&inv_parent, start_world);
    let d = matrix_transform_dir(&inv_parent, dir_world);

    let (hit_opt, cache_obbs) =
        index.pick_nearest_model_space_with_obbs(palette, visible_bones, cfg.max_dist, s, d);
    let hit = hit_opt?;

    // Contact point in model space: cp = S + D * dist
    let cp = [
        s[0] + d[0] * hit.dist,
        s[1] + d[1] * hit.dist,
        s[2] + d[2] * hit.dist,
    ];

    // Collect visible pickable bones whose animated OBB overlaps sphere(cp, size)
    let mut test_bones = Vec::new();
    for (k, obb) in cache_obbs.iter().enumerate() {
        let bone_id = k as u16;
        if visible_bones.is_visible(bone_id)
            && index.is_bone_pickable(bone_id)
            && obb.test_sphere(cp, cfg.size)
        {
            test_bones.push(bone_id);
        }
    }
    if test_bones.is_empty() {
        return None;
    }

    // Projection normal = normalize(hit_normal + normalize(-D)) matching SkeletonCustom.cpp:794-796
    let neg_d = vec3_normalize_safe([-d[0], -d[1], -d[2]]);
    let combined = [
        hit.normal[0] + neg_d[0],
        hit.normal[1] + neg_d[1],
        hit.normal[2] + neg_d[2],
    ];
    let proj_normal = {
        let n = vec3_normalize_safe(combined);
        if vec3_len(n) <= 1e-6 {
            hit.normal
        } else {
            n
        }
    };
    if vec3_len(proj_normal) <= 1e-6 {
        return None;
    }

    let m_view = build_wallmark_view_matrix(proj_normal, 0.9 * cfg.size, cp, cfg.rotation_deg);

    let mut faces = Vec::new();
    let mut seen_faces: HashSet<(usize, u32)> = HashSet::new();
    for (leaf_idx, leaf) in index.leaves.iter().enumerate() {
        for &bone_id in &test_bones {
            let Some(bone_slots) = index.bone_child_faces.get(bone_id as usize) else {
                continue;
            };
            let bone_faces = &bone_slots[leaf_idx];
            for &face_u32 in bone_faces {
                if !seen_faces.insert((leaf_idx, face_u32)) {
                    continue;
                }
                let base = (face_u32 as usize) * 3;
                let mut skinned_p = [[0.0f32; 3]; 3];
                let mut bind_p = [[0.0f32; 3]; 3];
                let mut bone_ids = [[0u16; 4]; 3];
                let mut weights = [[0.0f32; 3]; 3];
                let mut link_counts = [0u8; 3];
                let mut ok = true;

                for k in 0..3 {
                    let vi = leaf.active_indices[base + k] as usize;
                    match skin_vertex_pos_and_influences(
                        leaf.vertices,
                        vi,
                        &palette.render_transforms,
                    ) {
                        Ok(inf) => {
                            skinned_p[k] = inf.skinned_pos;
                            bind_p[k] = inf.bind_pos;
                            link_counts[k] = inf.link_count;
                            bone_ids[k] = inf.sentinel_bones;
                            weights[k] = inf.weights;
                        }
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }

                let test_normal = tri_mknormal(skinned_p[0], skinned_p[1], skinned_p[2]);
                if vec3_dot(test_normal, proj_normal) < 1e-5 {
                    continue;
                }
                if !sphere_triangle_test(cp, cfg.size, skinned_p[0], skinned_p[1], skinned_p[2]) {
                    continue;
                }

                let mut uvs = [[0.0f32; 2]; 3];
                for k in 0..3 {
                    let uvw = matrix_transform_point(&m_view, skinned_p[k]);
                    uvs[k] = [(1.0 + uvw[0]) * 0.5, (1.0 - uvw[1]) * 0.5];
                }

                faces.push(SkeletonWallmarkFace {
                    verts: bind_p,
                    bone_ids,
                    weights,
                    link_counts,
                    uvs,
                });
            }
        }
    }

    if faces.is_empty() {
        return None;
    }

    let time_end = if cfg.ttl < 0.0 {
        -1.0
    } else if cfg.ttl == 0.0 {
        WALLMARK_DEFAULT_TTL
    } else {
        cfg.ttl
    };

    Some(SkeletonWallmarkRecord {
        object_id: cfg.object_id,
        local_center: cp,
        local_radius: cfg.size * 2.0,
        shader: cfg.shader.clone(),
        texture: cfg.texture.clone(),
        time_start: cfg.time_now,
        time_end,
        faces,
    })
}

/// Active skeletal decal pool managing per-object decals, `0.02m` duplicate replacement, TTL
/// expiry, frustum culling, and live-pose re-skinning into `StaticWallmarkBatch` vertices.
#[derive(Debug, Clone, Default)]
pub struct SkeletonWallmarkPool {
    items: Vec<SkeletonWallmarkRecord>,
}

impl SkeletonWallmarkPool {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[inline]
    pub fn items(&self) -> &[SkeletonWallmarkRecord] {
        &self.items
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Add or replace an overlapping decal on the same `(object_id, shader, texture)` within
    /// `WALLMARK_DUPLICATE_TOLERANCE` (`0.02m`, matching `SkeletonCustom.cpp:776-786`).
    pub fn add(&mut self, record: SkeletonWallmarkRecord) {
        for existing in &mut self.items {
            if existing.similar(
                record.object_id,
                &record.shader,
                &record.texture,
                record.local_center,
                WALLMARK_DUPLICATE_TOLERANCE,
            ) {
                *existing = record;
                return;
            }
        }
        self.items.push(record);
    }

    /// Remove all skeletal wallmarks owned by `object_id`, returning count removed.
    pub fn remove_object(&mut self, object_id: u64) -> usize {
        let before = self.items.len();
        self.items.retain(|wm| wm.object_id != object_id);
        before - self.items.len()
    }

    /// Remove finite-lifetime skeletal decals whose elapsed ratio reached `1.0`.
    pub fn prune_expired(&mut self, time_now: f32) -> usize {
        let before = self.items.len();
        self.items.retain(|wm| {
            if wm.time_end < 0.0 {
                return true;
            }
            let w = (time_now - wm.time_start) / wm.time_end.max(1e-6);
            w < 1.0
        });
        before - self.items.len()
    }

    /// Prune expired decals (`w >= 1.0`), frustum-cull live world bounds (`CKinematics::CalculateWallmarks`),
    /// and re-skin visible faces with live `(parent_xform, palette)` (`CKinematics::RenderWallmark`).
    ///
    /// Matches `SkeletonCustom.cpp:829-930`: decals with `w >= 1.0` are pruned immediately without
    /// rendering, while active decals emit `WallmarkVertex` triangles grouped by `(shader, texture)`.
    pub fn collect_visible_batches<'a, F>(
        &mut self,
        time_now: f32,
        frustum: &ClipFrustum,
        mut resolve_pose: F,
    ) -> Result<Vec<StaticWallmarkBatch>, FormatError>
    where
        F: FnMut(u64) -> Option<(&'a XRayMatrix, &'a BonePalette)>,
    {
        let mut batches: Vec<StaticWallmarkBatch> = Vec::new();
        let mut idx = 0usize;

        while idx < self.items.len() {
            let expired = {
                let wm = &self.items[idx];
                if wm.time_end < 0.0 {
                    false
                } else {
                    let w = (time_now - wm.time_start) / wm.time_end.max(1e-6);
                    w >= 1.0
                }
            };
            if expired {
                self.items.swap_remove(idx);
                continue;
            }

            let wm = &self.items[idx];
            if let Some((parent_xform, palette)) = resolve_pose(wm.object_id) {
                let (world_center, world_radius) = wm.world_bounds(parent_xform);
                if frustum.test_sphere(world_center, world_radius) {
                    let w = if wm.time_end < 0.0 {
                        0.0
                    } else {
                        ((time_now - wm.time_start) / wm.time_end.max(1e-6)).clamp(0.0, 1.0)
                    };
                    let a_c = ((w * 255.0).floor() as i32).clamp(0, 255) as u8;
                    let color = pack_argb(a_c, 128, 128, 128);

                    let batch_idx = if let Some(pos) = batches
                        .iter()
                        .position(|b| b.shader == wm.shader && b.texture == wm.texture)
                    {
                        pos
                    } else {
                        batches.push(StaticWallmarkBatch {
                            shader: wm.shader.clone(),
                            texture: wm.texture.clone(),
                            vertices: Vec::with_capacity(wm.faces.len() * 3),
                        });
                        batches.len() - 1
                    };

                    let out = &mut batches[batch_idx].vertices;
                    for face in &wm.faces {
                        for k in 0..3 {
                            let model_pos = skin_position_with_links(
                                face.verts[k],
                                face.link_counts[k],
                                face.bone_ids[k],
                                face.weights[k],
                                &palette.render_transforms,
                            )?;
                            let world_pos = matrix_transform_point(parent_xform, model_pos);
                            out.push(WallmarkVertex {
                                position: world_pos,
                                color,
                                uv: face.uvs[k],
                            });
                        }
                    }
                }
            }

            idx += 1;
        }

        Ok(batches)
    }
}

// ============================================================================
// Internal .pe EParticleAction -> CompiledParticleAction Compiler
// ============================================================================

enum LtxView<'a> {
    Borrowed(&'a LtxDocument<'a>),
    Owned(&'a OwnedLtxDocument),
}

impl<'a> LtxView<'a> {
    fn contains_section(&self, section: &str) -> bool {
        match self {
            Self::Borrowed(d) => d.contains_section(section),
            Self::Owned(d) => d.sections.contains_key(section),
        }
    }

    fn read_str(&self, section: &str, key: &str) -> Option<&str> {
        match self {
            Self::Borrowed(d) => d.read_str(section, key),
            Self::Owned(d) => d.read_str(section, key),
        }
    }

    fn read_f32(&self, section: &str, key: &str) -> Option<f32> {
        self.read_str(section, key)?.parse::<f32>().ok()
    }

    fn read_i32(&self, section: &str, key: &str) -> Option<i32> {
        let s = self.read_str(section, key)?;
        if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            i32::from_str_radix(hex, 16).ok()
        } else {
            s.parse::<i32>().ok()
        }
    }

    fn read_u32(&self, section: &str, key: &str) -> Option<u32> {
        let s = self.read_str(section, key)?;
        if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            u32::from_str_radix(hex, 16).ok()
        } else {
            s.parse::<u32>().ok()
        }
    }

    fn read_bool(&self, section: &str, key: &str) -> Option<bool> {
        let s = self.read_str(section, key)?;
        match s.to_ascii_lowercase().as_str() {
            "true" | "1" | "on" | "yes" => Some(true),
            "false" | "0" | "off" | "no" => Some(false),
            _ => None,
        }
    }

    fn read_vec2(&self, section: &str, key: &str) -> Option<[f32; 2]> {
        let s = self.read_str(section, key)?;
        let mut it = s.split(',').filter_map(|p| p.trim().parse::<f32>().ok());
        Some([it.next()?, it.next()?])
    }

    fn read_vec3(&self, section: &str, key: &str) -> Option<[f32; 3]> {
        let s = self.read_str(section, key)?;
        let mut it = s.split(',').filter_map(|p| p.trim().parse::<f32>().ok());
        Some([it.next()?, it.next()?, it.next()?])
    }
}

fn read_editor_domain(doc: &LtxView<'_>, action_sect: &str, idx: usize) -> Result<ParticleDomain, FormatError> {
    let dom_sect = format!("domain_{action_sect}_{idx:04}");
    let dtype = ParticleDomainType::from_u32(
        doc.read_u32(&dom_sect, "type")
            .ok_or(FormatError::InvalidValue("missing domain type in .pe"))?,
    )?;
    let v0 = doc.read_vec3(&dom_sect, "v0").unwrap_or([0.0; 3]);
    let v1 = doc.read_vec3(&dom_sect, "v1").unwrap_or([0.0; 3]);
    let v2 = doc.read_vec3(&dom_sect, "v2").unwrap_or([0.0; 3]);
    Ok(ParticleDomain::from_editor_params(
        dtype,
        [v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]],
    ))
}

fn read_flt(doc: &LtxView<'_>, sect: &str, idx: usize, ver: u32, default: f32) -> Result<f32, FormatError> {
    let key = format!("flt_{idx:04}");
    match doc.read_f32(sect, &key) {
        Some(v) => Ok(v),
        None if ver == 0 => Ok(default),
        None => Err(FormatError::InvalidValue("missing flt parameter in .pe action")),
    }
}

fn read_vec(doc: &LtxView<'_>, sect: &str, idx: usize, default: [f32; 3]) -> [f32; 3] {
    let key = format!("vec_{idx:04}");
    doc.read_vec3(sect, &key).unwrap_or(default)
}

fn read_b(doc: &LtxView<'_>, sect: &str, idx: usize, default: bool) -> bool {
    let key = format!("bool_{idx:04}");
    doc.read_bool(sect, &key).unwrap_or(default)
}

fn read_i(doc: &LtxView<'_>, sect: &str, idx: usize, default: i32) -> i32 {
    let key = format!("int_{idx:04}");
    doc.read_i32(sect, &key).unwrap_or(default)
}

fn parse_and_compile_editor_action(
    doc: &LtxView<'_>,
    sect: &str,
) -> Result<Option<CompiledParticleAction>, FormatError> {
    let action_type = doc
        .read_u32(sect, "action_type")
        .ok_or(FormatError::InvalidValue("missing action_type in .pe"))?;
    let ver = doc.read_u32(sect, "version").unwrap_or(1);
    let ed_flags = doc
        .read_u32(sect, "flags")
        .unwrap_or(papi_flags::ED_ACTION_ENABLED);

    if (ed_flags & papi_flags::ED_ACTION_ENABLED) == 0 {
        return Ok(None);
    }

    let rotate_flags = |allow_rot: bool| -> u32 {
        if allow_rot {
            papi_flags::ALLOW_ROTATE
        } else {
            0
        }
    };

    // Parameter order matches ASCII std::map<AnsiString, ...> key order in ParticleEffectActions.cpp
    let (flags, kind) = match action_type {
        // 0: EPAAvoid -> floats: [Epsilon, Look Ahead, Magnitude], domains: [Position], bools: [Allow Rotate]
        0 => {
            let eps = read_flt(doc, sect, 0, ver, 0.0)?;
            let look = read_flt(doc, sect, 1, ver, 0.0)?;
            let mag = read_flt(doc, sect, 2, ver, 0.0)?;
            let pos = read_editor_domain(doc, sect, 0)?;
            let rot = read_b(doc, sect, 0, true);
            (
                rotate_flags(rot),
                ParticleActionKind::Avoid {
                    position: pos,
                    look_ahead: look,
                    magnitude: mag,
                    epsilon: eps,
                },
            )
        }
        // 1: EPABounce -> floats: [Cutoff, Friction, Resilience], domains: [Position], bools: [Allow Rotate]
        1 => {
            let cutoff = read_flt(doc, sect, 0, ver, 1.0)?;
            let friction = read_flt(doc, sect, 1, ver, 0.5)?;
            let resilience = read_flt(doc, sect, 2, ver, 0.1)?;
            let pos = read_editor_domain(doc, sect, 0)?;
            let rot = read_b(doc, sect, 0, true);
            (
                rotate_flags(rot),
                ParticleActionKind::Bounce {
                    position: pos,
                    one_minus_friction: 1.0 - friction,
                    resilience,
                    cutoff_sqr: cutoff * cutoff,
                },
            )
        }
        // 3: EPACopyVertexB -> bools: [Copy Position]
        3 => {
            let copy_pos = u32::from(read_b(doc, sect, 0, true));
            (0, ParticleActionKind::CopyVertexB { copy_pos })
        }
        // 4: EPADamping -> vectors: [Damping], floats: [V High, V Low]
        4 => {
            let damping = read_vec(doc, sect, 0, [0.0; 3]);
            let v_high = read_flt(doc, sect, 0, ver, P_MAXFLOAT)?;
            let v_low = read_flt(doc, sect, 1, ver, 0.0)?;
            (
                0,
                ParticleActionKind::Damping {
                    damping,
                    v_low_sqr: v_low * v_low,
                    v_high_sqr: v_high * v_high,
                },
            )
        }
        // 5: EPAExplosion -> vectors: [Center], floats: [Age, Epsilon, Magnitude, Standart Dev, Velocity], bools: [Allow Rotate]
        5 => {
            let center = read_vec(doc, sect, 0, [0.0; 3]);
            let age = read_flt(doc, sect, 0, ver, 0.0)?;
            let eps = read_flt(doc, sect, 1, ver, EPS_L)?;
            let mag = read_flt(doc, sect, 2, ver, 2.0)?;
            let stdev = read_flt(doc, sect, 3, ver, 3.0)?;
            let vel = read_flt(doc, sect, 4, ver, 1.0)?;
            let rot = read_b(doc, sect, 0, true);
            (
                rotate_flags(rot),
                ParticleActionKind::Explosion {
                    center,
                    velocity: vel,
                    magnitude: mag,
                    stdev,
                    age,
                    epsilon: if eps < 0.0 { EPS_L } else { eps },
                },
            )
        }
        // 6: EPAFollow -> floats: [Epsilon, Magnitude, Max Radius]
        6 => (
            0,
            ParticleActionKind::Follow {
                epsilon: read_flt(doc, sect, 0, ver, EPS_L)?,
                magnitude: read_flt(doc, sect, 1, ver, 0.0)?,
                max_radius: read_flt(doc, sect, 2, ver, P_MAXFLOAT)?,
            },
        ),
        // 7: EPAGravitate -> floats: [Epsilon, Magnitude, Max Radius]
        7 => (
            0,
            ParticleActionKind::Gravitate {
                epsilon: read_flt(doc, sect, 0, ver, 0.001)?,
                magnitude: read_flt(doc, sect, 1, ver, 1.0)?,
                max_radius: read_flt(doc, sect, 2, ver, 10.0)?,
            },
        ),
        // 8: EPAGravity -> vectors: [Direction], bools: [Allow Rotate]
        8 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::Gravity {
                direction: read_vec(doc, sect, 0, [0.0, -9.8, 0.0]),
            },
        ),
        // 9: EPAJet -> domains: [Accelerate], vectors: [Center], floats: [Epsilon, Magnitude, Max Radius], bools: [Allow Rotate]
        9 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::Jet {
                acc: read_editor_domain(doc, sect, 0)?,
                center: read_vec(doc, sect, 0, [0.0; 3]),
                epsilon: read_flt(doc, sect, 0, ver, EPS_L)?,
                magnitude: read_flt(doc, sect, 1, ver, 0.0)?,
                max_radius: read_flt(doc, sect, 2, ver, P_MAXFLOAT)?,
            },
        ),
        // 10: EPAKillOld -> floats: [Age Limit], bools: [Kill Less Than]
        10 => (
            0,
            ParticleActionKind::KillOld {
                age_limit: read_flt(doc, sect, 0, ver, 5.0)?,
                kill_less_than: u32::from(read_b(doc, sect, 0, false)),
            },
        ),
        // 11: EPAMatchVelocity -> floats: [Epsilon, Magnitude, Max Radius]
        11 => (
            0,
            ParticleActionKind::MatchVelocity {
                epsilon: read_flt(doc, sect, 0, ver, EPS_L)?,
                magnitude: read_flt(doc, sect, 1, ver, 0.0)?,
                max_radius: read_flt(doc, sect, 2, ver, P_MAXFLOAT)?,
            },
        ),
        // 12: EPAMove
        12 => (0, ParticleActionKind::Move),
        // 13: EPAOrbitLine -> vectors: [Axis, Position], floats: [Epsilon, Magnitude, Max Radius], bools: [Allow Rotate]
        13 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::OrbitLine {
                axis: vec3_normalize_safe(read_vec(doc, sect, 0, [0.0; 3])),
                p: read_vec(doc, sect, 1, [0.0; 3]),
                epsilon: read_flt(doc, sect, 0, ver, EPS_L)?,
                magnitude: read_flt(doc, sect, 1, ver, 1.0)?,
                max_radius: read_flt(doc, sect, 2, ver, P_MAXFLOAT)?,
            },
        ),
        // 14: EPAOrbitPoint -> vectors: [Center], floats: [Epsilon, Magnitude, Max Radius], bools: [Allow Rotate]
        14 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::OrbitPoint {
                center: read_vec(doc, sect, 0, [0.0; 3]),
                epsilon: read_flt(doc, sect, 0, ver, 0.1)?,
                magnitude: read_flt(doc, sect, 1, ver, 400.0)?,
                max_radius: read_flt(doc, sect, 2, ver, 100.0)?,
            },
        ),
        // 15: EPARandomAccel
        15 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::RandomAccel {
                gen_acc: read_editor_domain(doc, sect, 0)?,
            },
        ),
        // 16: EPARandomDisplace
        16 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::RandomDisplace {
                gen_disp: read_editor_domain(doc, sect, 0)?,
            },
        ),
        // 17: EPARandomVelocity
        17 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::RandomVelocity {
                gen_vel: read_editor_domain(doc, sect, 0)?,
            },
        ),
        // 18: EPARestore -> floats: [Time]
        18 => (
            0,
            ParticleActionKind::Restore {
                time_left: read_flt(doc, sect, 0, ver, 0.0)?,
            },
        ),
        // 19: EPASink -> domains: [Domain], bools: [Allow Rotate, Kill Inside]
        19 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::Sink {
                position: read_editor_domain(doc, sect, 0)?,
                kill_inside: u32::from(read_b(doc, sect, 1, true)),
            },
        ),
        // 20: EPASinkVelocity -> domains: [Domain], bools: [Allow Rotate, Kill Inside]
        20 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::SinkVelocity {
                velocity: read_editor_domain(doc, sect, 0)?,
                kill_inside: u32::from(read_b(doc, sect, 1, true)),
            },
        ),
        // 21: EPASource ->
        //   floats: [Age Sigma, Color\Alpha, Parent Motion, Rate, Starting Age]
        //   domains: [Color, Domain, Rotation, Size, Velocity]
        //   bools: [Allow Rotate, Single Size]
        21 => {
            let age_sigma = read_flt(doc, sect, 0, ver, 0.0)?;
            let alpha = read_flt(doc, sect, 1, ver, 0.0)?;
            let parent_motion = read_flt(doc, sect, 2, ver, 0.0)?;
            let particle_rate = read_flt(doc, sect, 3, ver, 100.0)?;
            let age = read_flt(doc, sect, 4, ver, 0.0)?;
            let color = read_editor_domain(doc, sect, 0)?;
            let position = read_editor_domain(doc, sect, 1)?;
            let rot = read_editor_domain(doc, sect, 2)?;
            let size = read_editor_domain(doc, sect, 3)?;
            let velocity = read_editor_domain(doc, sect, 4)?;
            let allow_rotate = read_b(doc, sect, 0, false);
            let single_size = read_b(doc, sect, 1, false);
            let f = (if single_size {
                papi_flags::PASOURCE_SINGLE_SIZE
            } else {
                0
            }) | papi_flags::PASOURCE_VERTEXB_TRACKS
                | rotate_flags(allow_rotate);
            (
                f,
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
                    parent_vel: [0.0; 3],
                    parent_motion,
                },
            )
        }
        // 22: EPASpeedLimit -> floats: [Max Speed, Min Speed]
        22 => (
            0,
            ParticleActionKind::SpeedLimit {
                max_speed: read_flt(doc, sect, 0, ver, 15.0)?,
                min_speed: read_flt(doc, sect, 1, ver, -1.0)?,
            },
        ),
        // 23: EPATargetColor -> vectors: [Color], floats: [Alpha, Scale, TimeFrom, TimeTo]
        23 => (
            0,
            ParticleActionKind::TargetColor {
                color: read_vec(doc, sect, 0, [1.0, 1.0, 1.0]),
                alpha: read_flt(doc, sect, 0, ver, 1.0)?,
                scale: read_flt(doc, sect, 1, ver, 1.0)?,
                time_from: read_flt(doc, sect, 2, ver, 0.0)?,
                time_to: read_flt(doc, sect, 3, ver, 1.0)?,
            },
        ),
        // 24: EPATargetSize -> vectors: [Scale, Size]
        24 => (
            0,
            ParticleActionKind::TargetSize {
                scale: read_vec(doc, sect, 0, [1.0, 1.0, 0.0]),
                size: read_vec(doc, sect, 1, [2.0, 2.0, 0.001]),
            },
        ),
        // 25/26: EPATargetRotate -> vectors: [Rotation], floats: [Scale]
        25 | 26 => (
            0,
            ParticleActionKind::TargetRotate {
                did_variant: action_type == 26,
                rot: read_vec(doc, sect, 0, [0.0; 3]),
                scale: read_flt(doc, sect, 0, ver, 1.0)?,
            },
        ),
        // 27/28: EPATargetVelocity -> vectors: [Velocity], floats: [Scale], bools: [Allow Rotate]
        27 | 28 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::TargetVelocity {
                did_variant: action_type == 28,
                velocity: read_vec(doc, sect, 0, [0.0; 3]),
                scale: read_flt(doc, sect, 0, ver, 1.0)?,
            },
        ),
        // 29: EPAVortex -> vectors: [Axis, Center], floats: [Epsilon, Magnitude, Max Radius], bools: [Allow Rotate]
        29 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::Vortex {
                axis: vec3_normalize_safe(read_vec(doc, sect, 0, [0.0, 1.0, 0.0])),
                center: read_vec(doc, sect, 1, [0.0; 3]),
                epsilon: read_flt(doc, sect, 0, ver, EPS_L)?,
                magnitude: read_flt(doc, sect, 1, ver, 1.0)?,
                max_radius: read_flt(doc, sect, 2, ver, 1.0)?,
            },
        ),
        // 30: EPATurbulence -> floats: [Delta, Frequency, Magnitude], ints: [Octaves], vectors: [Movement]
        30 => (
            0,
            ParticleActionKind::Turbulence {
                epsilon: read_flt(doc, sect, 0, ver, 0.01)?,
                frequency: read_flt(doc, sect, 1, ver, 2.0)?,
                magnitude: read_flt(doc, sect, 2, ver, 10.0)?,
                octaves: read_i(doc, sect, 0, 1),
                offset: read_vec(doc, sect, 0, [1.0, 1.0, 1.0]),
            },
        ),
        // 31: EPAScatter -> vectors: [Center], floats: [Epsilon, Magnitude, Max Radius], bools: [Allow Rotate]
        31 => (
            rotate_flags(read_b(doc, sect, 0, true)),
            ParticleActionKind::Scatter {
                center: read_vec(doc, sect, 0, [0.0; 3]),
                epsilon: read_flt(doc, sect, 0, ver, EPS_L)?,
                magnitude: read_flt(doc, sect, 1, ver, 0.0)?,
                max_radius: read_flt(doc, sect, 2, ver, P_MAXFLOAT)?,
            },
        ),
        _ => return Err(FormatError::InvalidValue("unknown .pe action_type")),
    };

    // Note: C++ pCreateEAction maps PATargetRotateDID -> PATargetRotateID during EPATargetRotate Compile
    let compiled_id = match action_type {
        26 => 25,
        28 => 27,
        other => other,
    };

    Ok(Some(CompiledParticleAction {
        outer_type: compiled_id,
        flags,
        inner_type: compiled_id,
        kind,
    }))
}

#[inline]
fn write_f32s(out: &mut Vec<u8>, vals: &[f32]) {
    for &f in vals {
        out.extend_from_slice(&f.to_le_bytes());
    }
}

#[inline]
fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn vec3_scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn vec3_len(a: [f32; 3]) -> f32 {
    vec3_dot(a, a).sqrt()
}

#[inline]
fn vec3_normalize_safe(a: [f32; 3]) -> [f32; 3] {
    let l2 = vec3_dot(a, a);
    if l2 > 1e-12 {
        let inv = 1.0 / l2.sqrt();
        [a[0] * inv, a[1] * inv, a[2] * inv]
    } else {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_particle_domain_compile_and_within_checks() {
        let sphere = ParticleDomain::from_editor_params(
            ParticleDomainType::Sphere,
            [1.0, 2.0, 3.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(sphere.radius1, 5.0);
        assert_eq!(sphere.radius2, 2.0);
        assert!(sphere.contains([4.0, 2.0, 3.0])); // r = 3 in [2, 5]
        assert!(!sphere.contains([1.5, 2.0, 3.0])); // r = 0.5 < 2
        assert!(!sphere.contains([7.0, 2.0, 3.0])); // r = 6 > 5

        let box_dom = ParticleDomain::from_editor_params(
            ParticleDomainType::Box,
            [2.0, -1.0, 0.0, -2.0, 3.0, 4.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(box_dom.p1, [-2.0, -1.0, 0.0]);
        assert_eq!(box_dom.p2, [2.0, 3.0, 4.0]);
        assert!(box_dom.contains([0.0, 1.0, 2.0]));
    }

    #[test]
    fn test_particles_xr_binary_roundtrip_and_actions() {
        let src = CompiledParticleAction {
            outer_type: 21,
            flags: papi_flags::PASOURCE_VERTEXB_TRACKS | papi_flags::ALLOW_ROTATE,
            inner_type: 21,
            kind: ParticleActionKind::Source {
                position: ParticleDomain::from_editor_params(
                    ParticleDomainType::Box,
                    [-0.5, 0.0, -0.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0],
                ),
                velocity: ParticleDomain::from_editor_params(
                    ParticleDomainType::Point,
                    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                rot: ParticleDomain::default(),
                size: ParticleDomain::from_editor_params(
                    ParticleDomainType::Point,
                    [0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                color: ParticleDomain::from_editor_params(
                    ParticleDomainType::Point,
                    [1.0, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                alpha: 0.8,
                particle_rate: 50.0,
                age: 2.0,
                age_sigma: 0.2,
                parent_vel: [0.0; 3],
                parent_motion: 0.1,
            },
        };
        let grav = CompiledParticleAction {
            outer_type: 8,
            flags: papi_flags::ALLOW_ROTATE,
            inner_type: 8,
            kind: ParticleActionKind::Gravity {
                direction: [0.0, -9.8, 0.0],
            },
        };
        let eff = ParticleEffectDef {
            name: "pfx\\sparks".to_string(),
            flags: ped_flags::DF_SPRITE | ped_flags::DF_FRAMED | ped_flags::DF_TIME_LIMIT,
            max_particles: 128,
            update_step_ms: 33,
            shader: Some("effects\\add".to_string()),
            texture: Some("fx\\fx_spark".to_string()),
            frame: Some(ParticleSpriteFrame {
                tex_size: [0.5, 0.5],
                reserved: [0.0, 0.0],
                frame_dim_x: 2,
                frame_count: 4,
                speed: 12.0,
            }),
            time_limit: Some(3.5),
            collision: None,
            velocity_scale: None,
            align_to_path_rotation: None,
            actions: vec![src, grav],
        };
        let grp = ParticleGroupDef {
            name: "anomaly\\fire_group".to_string(),
            flags: 0,
            time_limit: 4.0,
            effects: vec![GroupEffectItem {
                effect_name: "pfx\\sparks".to_string(),
                on_play_child: String::new(),
                on_birth_child: String::new(),
                on_dead_child: String::new(),
                time0: 0.0,
                time1: 4.0,
                flags: pg_effect_flags::FL_ENABLED,
            }],
        };
        let lib = ParticlesLibrary {
            effects: vec![eff.clone()],
            groups: vec![grp.clone()],
        };
        let raw = lib.to_bytes();
        let parsed = ParticlesLibrary::parse(&raw).unwrap();
        assert_eq!(parsed.find_effect("pfx\\sparks").unwrap().actions.len(), 2);
        let (lt, rb) = parsed.effects[0].frame.unwrap().calculate_tc(3);
        assert_eq!(lt, [0.5, 0.5]);
        assert_eq!(rb, [1.0, 1.0]);
        assert_eq!(parsed.find_group("anomaly\\fire_group").unwrap().effects.len(), 1);
    }

    #[test]
    fn test_loose_pe_pg_ltx_parsing_and_override_precedence() {
        let pe_ltx = r#"
[_effect]
version = 1
max_particles = 64
update_step = 20
flags = 0x4011
action_count = 2

[sprite]
shader = effects\blend
texture = fx\fx_smoke

[timelimit]
value = 5.0

[action_0000]
version = 1
action_type = 21
action_name = Source
flags = 1
flt_0000 = 0.1
flt_0001 = 0.75
flt_0002 = 0.25
flt_0003 = 30.0
flt_0004 = 1.5
bool_0000 = true
bool_0001 = true

[domain_action_0000_0000]
type = 0
v0 = 1.0, 1.0, 1.0
v1 = 0,0,0
v2 = 0,0,0

[domain_action_0000_0001]
type = 4
v0 = -1,-1,-1
v1 = 1,1,1
v2 = 0,0,0

[domain_action_0000_0002]
type = 0
v0 = 0,0,0
v1 = 0,0,0
v2 = 0,0,0

[domain_action_0000_0003]
type = 0
v0 = 0.5,0.5,0.5
v1 = 0,0,0
v2 = 0,0,0

[domain_action_0000_0004]
type = 0
v0 = 0,2,0
v1 = 0,0,0
v2 = 0,0,0

[action_0001]
version = 1
action_type = 8
action_name = DisabledGravity
flags = 0
vec_0000 = 0, -5, 0
bool_0000 = true
"#;
        let pe = ParticleEffectDef::from_pe_ltx("pfx\\sparks", pe_ltx).unwrap();
        assert_eq!(pe.max_particles, 64);
        assert_eq!(pe.update_step_ms, 20);
        // action_0001 has flEnabled=0 so only action_0000 compiles
        assert_eq!(pe.actions.len(), 1);
        match &pe.actions[0].kind {
            ParticleActionKind::Source {
                alpha,
                particle_rate,
                age,
                age_sigma,
                parent_motion,
                velocity,
                ..
            } => {
                assert_eq!((*alpha, *particle_rate, *age, *age_sigma, *parent_motion), (0.75, 30.0, 1.5, 0.1, 0.25));
                assert_eq!(velocity.p1, [0.0, 2.0, 0.0]);
            }
            other => panic!("unexpected action {other:?}"),
        }

        let pg_ltx = r#"
[_group]
flags = 0
effects_count = 1
timelimit = 2.5

[effect_0000]
effect_name = pfx\sparks
on_play_child =
on_birth_child = pfx\child
on_death_child =
time0 = 0.0
time1 = 2.5
flags = 4
"#;
        let pg = ParticleGroupDef::from_pg_ltx("custom\\g", pg_ltx).unwrap();
        assert_eq!(pg.effects[0].on_birth_child, "pfx\\child");

        let mut lib = ParticlesLibrary {
            effects: vec![ParticleEffectDef {
                name: "pfx\\sparks".to_string(),
                flags: 0,
                max_particles: 10,
                update_step_ms: 33,
                shader: None,
                texture: None,
                frame: None,
                time_limit: None,
                collision: None,
                velocity_scale: None,
                align_to_path_rotation: None,
                actions: vec![],
            }],
            groups: vec![],
        };
        lib.merge_loose(vec![pe], vec![pg]);
        assert_eq!(lib.find_effect("pfx\\sparks").unwrap().max_particles, 64);
        assert_eq!(lib.find_group("custom\\g").unwrap().effects.len(), 1);
    }

    #[test]
    fn test_blood_wallmarks_ltx_config() {
        let ltx = r#"
[m_blood_marks]
wallmarks = wm\wm_blood_1, wm\wm_blood_2
min_size = 0.3
max_size = 1.2
dist = 2.4
nominal_hit = 0.5
blood_drops = wm\wm_blood_drop_1
start_blood_size = 0.4
stop_blood_size = 0.02
blood_drop_size = 0.1
"#;
        let cfg = BloodWallmarksConfig::from_ltx_str(ltx, "m_blood_marks").unwrap();
        assert_eq!(cfg.wallmarks.shader, "effects\\wallmark");
        assert_eq!(cfg.wallmarks.textures, vec!["wm\\wm_blood_1", "wm\\wm_blood_2"]);
        assert_eq!(cfg.blood_drops.textures, vec!["wm\\wm_blood_drop_1"]);
        assert!((cfg.nominal_hit - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_static_wallmark_projection_cform_adjacency_crease_and_pool() {
        use crate::camera::{matrix_build_camera, matrix_build_projection};
        use crate::cform::CformTriangle;
        use crate::ogf::matrix_mul_44;

        // Build a 4-triangle mesh sharing vertices:
        // - Tri 0 & 1 form a front wall on z = 0 facing -Z (normal = [0, 0, -1]) across x=[-1..1], y=[-1..1]
        //   Wait: (p1-p0)x(p2-p1) for p0=(-1,-1,0), p1=(-1,1,0), p2=(1,1,0) => (0,2,0)x(2,0,0) = (0,0,-4) -> [0,0,-1].
        // - Tri 2 is a perpendicular side wall at x=1 turning +Z (normal = [1, 0, 0]): dot([1,0,0], [0,0,-1]) = 0 < 0.034899 (90 deg crease!)
        // - Tri 3 shares top edge y=1 with Tri 1 & 0, tilted slightly (same plane / small 15 deg tilt), marked suppress_wm=true when on cform.
        let model = CformModel::from_parts(
            vec![
                [-1.0, -1.0, 0.0], // 0
                [-1.0, 1.0, 0.0],  // 1
                [1.0, 1.0, 0.0],   // 2
                [1.0, -1.0, 0.0],  // 3
                [1.0, -1.0, 2.0],  // 4
                [1.0, 1.0, 2.0],   // 5
                [-1.0, 2.0, 0.25], // 6
                [1.0, 2.0, 0.25],  // 7
            ],
            vec![
                CformTriangle {
                    verts: [0, 1, 2],
                    material: 0,
                    suppress_shadows: false,
                    suppress_wm: false,
                    sector: 0,
                },
                CformTriangle {
                    verts: [0, 2, 3],
                    material: 0,
                    suppress_shadows: false,
                    suppress_wm: false,
                    sector: 0,
                },
                // 90 deg crease wall sharing edge (3, 2)
                CformTriangle {
                    verts: [3, 2, 5],
                    material: 0,
                    suppress_shadows: false,
                    suppress_wm: false,
                    sector: 0,
                },
                // Top lip sharing edge (1, 2) with suppress_wm = true
                CformTriangle {
                    verts: [1, 6, 7],
                    material: 0,
                    suppress_shadows: false,
                    suppress_wm: true,
                    sector: 0,
                },
            ],
        )
        .unwrap();
        let bvh = model.build_bvh();

        // Project a 0.5m radius decal at (0.0, 0.0, 0.0) across the diagonal shared by Tri 0 & Tri 1
        let mut cfg = StaticWallmarkConfig {
            shader: "effects\\wallmark".to_string(),
            texture: "wm\\wm_bullet".to_string(),
            size: 0.5,
            ttl: 10.0,
            rotation_deg: 0.0,
            time_now: 2.0,
            ignore_range_opt: false,
            camera_pos: Some([0.0, 0.0, -5.0]),
            respect_suppress_wm: true,
        };
        let wm = project_static_wallmark_on_cform(&model, &bvh, 0, [0.0, 0.0, 0.0], &cfg).unwrap();
        assert!(wm.vertices.len() >= 6);
        assert_eq!(wm.vertices.len() % 3, 0);
        // Every projected vertex stays on z = 0 wall (90 deg side wall & suppress_wm excluded) and within [-0.5, 0.5]
        for v in &wm.vertices {
            assert!(v.position[2].abs() < 1e-4, "vertex leaked off front wall: {:?}", v.position);
            assert!(v.position[0] >= -0.501 && v.position[0] <= 0.501);
            assert!(v.position[1] >= -0.501 && v.position[1] <= 0.501);
            assert!(v.uv[0] >= -1e-4 && v.uv[0] <= 1.0 + 1e-4);
            assert!(v.uv[1] >= -1e-4 && v.uv[1] <= 1.0 + 1e-4);
        }
        assert!((wm.bounds_center[0]).abs() < 1e-4);
        assert!((wm.bounds_center[1]).abs() < 1e-4);

        // Seed triangle with suppress_wm=true must produce None
        assert!(project_static_wallmark_on_cform(&model, &bvh, 3, [0.0, 1.5, 0.1], &cfg).is_none());

        // Range culling (>100m away) returns None unless ignore_range_opt = true
        cfg.camera_pos = Some([0.0, 0.0, -200.0]);
        assert!(project_static_wallmark_on_cform(&model, &bvh, 0, [0.0, 0.0, 0.0], &cfg).is_none());
        cfg.ignore_range_opt = true;
        assert!(project_static_wallmark_on_cform(&model, &bvh, 0, [0.0, 0.0, 0.0], &cfg).is_some());

        // Pool duplicate replacement (< 0.02m) + visibility emission + TTL expiry
        let mut pool = StaticWallmarkPool::new();
        pool.add(wm.clone());
        assert_eq!(pool.len(), 1);
        let mut replacement = wm.clone();
        replacement.bounds_center[0] += 0.005;
        replacement.time_start = 4.0;
        pool.add(replacement);
        assert_eq!(pool.len(), 1);

        let view = matrix_build_camera([0.0, 0.0, -4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = matrix_build_projection(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);
        let full = matrix_mul_44(&proj, &view);
        let frustum = ClipFrustum::from_view_proj(&full, frustum_planes::LRTB | frustum_planes::FAR);

        // At t=9.0, elapsed = 5.0 / 10.0 = 0.5 -> alpha ~ 127
        let batches = pool.collect_visible_batches(9.0, [0.0, 0.0, -4.0], &frustum, 0.01);
        assert_eq!(batches.len(), 1);
        let a_byte = (batches[0].vertices[0].color >> 24) & 0xFF;
        assert!((126..=128).contains(&a_byte), "expected ~127 alpha, got {a_byte}");

        // At t=15.0, elapsed = 11.0 / 10.0 >= 1.0 -> rendered on frame of death & pruned
        let batches_end = pool.collect_visible_batches(15.0, [0.0, 0.0, -4.0], &frustum, 0.01);
        assert_eq!(batches_end.len(), 1);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_skeleton_wallmark_projection_animated_deformation_and_pool() {
        use crate::camera::{matrix_build_camera, matrix_build_projection};
        use crate::chunk::ChunkWriter;
        use crate::hom::{frustum_planes, ClipFrustum};
        use crate::ogf::ogf_chunks::*;
        use crate::ogf::{
            matrix_mul_44, matrix_set_xyz_translate, skel_vert_type, BoneIkData, BoneIkSection,
            BoneShape, BoneShapeType, JointLimit, OgfBone, OgfModel, OgfObb, OgfVertices,
            OgfVisualType, VertBoned3W, XRAY_IDENTITY_MATRIX,
        };
        use crate::skinning::{BoneMask, BonePalette, SkeletonCollisionIndex};

        fn wv3(out: &mut Vec<u8>, v: [f32; 3]) {
            for c in v {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }

        let mut ogf_w = ChunkWriter::new();
        let mut hdr = vec![4u8, OgfVisualType::SkeletonAnim as u8, 0, 0];
        hdr.extend_from_slice(&[0u8; 40]);
        ogf_w.write_chunk(OGF_HEADER, &hdr);
        let id_obb = OgfObb {
            rotate: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translate: [0.0; 3],
            halfsize: [1.5, 1.5, 1.5],
        };
        let mut bn = Vec::new();
        bn.extend_from_slice(&2u32.to_le_bytes());
        bn.extend_from_slice(b"root\0\0");
        for r in id_obb.rotate {
            wv3(&mut bn, r);
        }
        wv3(&mut bn, [0.0; 3]);
        wv3(&mut bn, id_obb.halfsize);
        bn.extend_from_slice(b"arm\0root\0");
        for r in id_obb.rotate {
            wv3(&mut bn, r);
        }
        wv3(&mut bn, [0.0; 3]);
        wv3(&mut bn, id_obb.halfsize);
        ogf_w.write_chunk(OGF_S_BONE_NAMES, &bn);

        let mut ik = Vec::new();
        for bt in [[0.0f32, 0.0, 0.0], [0.0, 2.0, 0.0]] {
            ik.extend_from_slice(&1u32.to_le_bytes());
            ik.extend_from_slice(b"default\0");
            ik.extend_from_slice(&(BoneShapeType::Box as u16).to_le_bytes());
            ik.extend_from_slice(&0u16.to_le_bytes());
            for _ in 0..27 {
                ik.extend_from_slice(&0.0f32.to_le_bytes());
            }
            ik.extend_from_slice(&0u32.to_le_bytes());
            for _ in 0..3 {
                ik.extend_from_slice(&[0u8; 16]);
            }
            ik.extend_from_slice(&[0u8; 24]);
            wv3(&mut ik, [0.0; 3]);
            wv3(&mut ik, bt);
            ik.extend_from_slice(&5.0f32.to_le_bytes());
            wv3(&mut ik, [0.0; 3]);
        }
        ogf_w.write_chunk(OGF_S_IKDATA, &ik);

        let mut model = OgfModel::parse(&ogf_w.into_bytes()).unwrap();
        let make_bone = |id: u16, name: &str, p: Option<u16>, bind: [f32; 3]| OgfBone {
            id,
            name: name.to_string(),
            parent_name: if p.is_some() { "root".into() } else { String::new() },
            parent_id: p,
            children: if id == 0 { vec![1] } else { vec![] },
            obb: id_obb,
            ik: Some(BoneIkSection {
                game_mtl_name: "default".into(),
                shape: BoneShape {
                    shape_type: BoneShapeType::Box as u16,
                    flags: 0,
                    box_obb: id_obb,
                    sphere_center: [0.0; 3],
                    sphere_radius: 0.0,
                    cylinder_center: [0.0; 3],
                    cylinder_direction: [0.0, 1.0, 0.0],
                    cylinder_height: 0.0,
                    cylinder_radius: 0.0,
                },
                ik_data: BoneIkData {
                    version: 1,
                    joint_type: 0,
                    limits: [JointLimit {
                        limit: [0.0, 0.0],
                        spring_factor: 0.0,
                        damping_factor: 0.0,
                    }; 3],
                    spring_factor: 0.0,
                    damping_factor: 0.0,
                    ik_flags: 0,
                    break_force: 0.0,
                    break_torque: 0.0,
                    friction: 0.0,
                },
                bind_rotation_xyz: [0.0; 3],
                bind_translation: bind,
                mass: 1.0,
                center_of_mass: [0.0; 3],
            }),
        };
        model.skeleton.as_mut().unwrap().bones = vec![
            make_bone(0, "root", None, [0.0, 0.0, 0.0]),
            make_bone(1, "arm", Some(0), [0.0, 2.0, 0.0]),
        ];

        // Child 0 (1W): tri 0 front-facing (-Z) at (0, 2, 0) on bone 1; tri 1 backfacing (+Z); tri 2 outside sphere at y=10
        let mut c0_w = ChunkWriter::new();
        let mut chdr = vec![4u8, OgfVisualType::SkeletonGeomdefSt as u8, 0, 0];
        chdr.extend_from_slice(&[0u8; 40]);
        c0_w.write_chunk(OGF_HEADER, &chdr);
        c0_w.write_chunk(OGF_TEXTURE, b"act\\stalker\0models\\model\0");
        let verts_1w = [
            // tri 0 (bone 1): front-facing -Z
            (1u32, [-0.4f32, 1.8, 0.0]),
            (1, [0.0, 2.4, 0.0]),
            (1, [0.4, 1.8, 0.0]),
            // tri 1 (bone 1): opposite winding (backfacing) at same coords
            (1, [-0.4, 1.8, 0.0]),
            (1, [0.4, 1.8, 0.0]),
            (1, [0.0, 2.4, 0.0]),
            // tri 2 (bone 1): far outside size=0.6 sphere
            (1, [-0.4, 9.8, 0.0]),
            (1, [0.0, 10.4, 0.0]),
            (1, [0.4, 9.8, 0.0]),
        ];
        let mut vb0 = Vec::new();
        vb0.extend_from_slice(&skel_vert_type::FVF_1L.to_le_bytes());
        vb0.extend_from_slice(&(verts_1w.len() as u32).to_le_bytes());
        for (b, pos) in verts_1w {
            wv3(&mut vb0, pos);
            wv3(&mut vb0, [0.0, 0.0, -1.0]);
            wv3(&mut vb0, [1.0, 0.0, 0.0]);
            wv3(&mut vb0, [0.0, 1.0, 0.0]);
            vb0.extend_from_slice(&0.0f32.to_le_bytes());
            vb0.extend_from_slice(&0.0f32.to_le_bytes());
            vb0.extend_from_slice(&b.to_le_bytes());
        }
        c0_w.write_chunk(OGF_VERTICES, &vb0);
        let mut ib0 = Vec::new();
        ib0.extend_from_slice(&(9u32).to_le_bytes());
        for i in 0u16..9 {
            ib0.extend_from_slice(&i.to_le_bytes());
        }
        c0_w.write_chunk(OGF_INDICES, &ib0);
        model.children.push(OgfModel::parse(&c0_w.into_bytes()).unwrap());

        // Child 1 (3W repeated bone IDs [0, 1, 1]): tests both bone 0 & bone 1 dedup AND repeated-bone skinning accuracy!
        let mut c1_w = ChunkWriter::new();
        c1_w.write_chunk(OGF_HEADER, &chdr);
        c1_w.write_chunk(OGF_TEXTURE, b"act\\stalker\0models\\model\0");
        let mut c1_model = OgfModel::parse(&c1_w.into_bytes()).unwrap();
        c1_model.vertices = Some(OgfVertices::Skinned3W(vec![
            VertBoned3W {
                matrices: [0, 1, 1],
                position: [-0.2, 1.9, 0.0],
                normal: [0.0, 0.0, -1.0],
                tangent: [1.0, 0.0, 0.0],
                binormal: [0.0, 1.0, 0.0],
                weights: [0.25, 0.50], // bone 0 gets 0.25, bone 1 gets 0.75
                u: 0.0,
                v: 0.0,
            },
            VertBoned3W {
                matrices: [0, 1, 1],
                position: [0.0, 2.2, 0.0],
                normal: [0.0, 0.0, -1.0],
                tangent: [1.0, 0.0, 0.0],
                binormal: [0.0, 1.0, 0.0],
                weights: [0.25, 0.50],
                u: 0.0,
                v: 0.0,
            },
            VertBoned3W {
                matrices: [0, 1, 1],
                position: [0.2, 1.9, 0.0],
                normal: [0.0, 0.0, -1.0],
                tangent: [1.0, 0.0, 0.0],
                binormal: [0.0, 1.0, 0.0],
                weights: [0.25, 0.50],
                u: 0.0,
                v: 0.0,
            },
        ]));
        c1_model.indices = vec![0, 1, 2];
        model.children.push(c1_model);

        let index = SkeletonCollisionIndex::from_ogf(&model).unwrap();
        let bind_pal = BonePalette::bind_pose(index.skeleton).unwrap();
        let all_vis = BoneMask::all(2);
        let impact_xform = matrix_set_xyz_translate([0.0, 0.0, 0.0], [5.0, 0.0, 0.0]);

        let mut cfg = SkeletonWallmarkConfig {
            object_id: 42,
            shader: "effects\\wallmark".to_string(),
            texture: "wm\\wm_blood_1".to_string(),
            size: 0.6,
            ttl: 0.0, // defaults to WALLMARK_DEFAULT_TTL (50s)
            rotation_deg: 0.0,
            time_now: 1.0,
            ignore_range_opt: false,
            camera_pos: Some([5.0, 0.0, -4.0]),
            max_dist: 100.0,
        };
        let wm = project_skeleton_wallmark(
            &index,
            &bind_pal,
            &impact_xform,
            &all_vis,
            [5.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
            &cfg,
        )
        .unwrap();
        assert_eq!(wm.time_end, WALLMARK_DEFAULT_TTL);
        // Only tri 0 of child 0 + tri 0 of child 1 (deduped across bones 0 & 1) = 2 faces!
        assert_eq!(wm.faces.len(), 2);
        for f in &wm.faces {
            for uv in &f.uvs {
                assert!(uv[0] >= -1e-3 && uv[0] <= 1.001);
                assert!(uv[1] >= -1e-3 && uv[1] <= 1.001);
            }
        }

        // Range cull beyond WALLMARK_RANGE_SKELETON (50m)
        cfg.camera_pos = Some([5.0, 0.0, -60.0]);
        assert!(project_skeleton_wallmark(
            &index,
            &bind_pal,
            &impact_xform,
            &all_vis,
            [5.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
            &cfg,
        )
        .is_none());

        // Pool duplicate replacement (<0.02m) + separate object isolation
        let mut pool = SkeletonWallmarkPool::new();
        pool.add(wm.clone());
        let mut near_dup = wm.clone();
        near_dup.local_center[0] += 0.005;
        near_dup.time_start = 2.0;
        pool.add(near_dup);
        assert_eq!(pool.len(), 1);
        let mut other_obj = wm.clone();
        other_obj.object_id = 99;
        pool.add(other_obj);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.remove_object(99), 1);
        assert_eq!(pool.len(), 1);

        // Animate pose (+3.0 X on arm bone 1) and move character to [8, 1, 0]
        let moved_models = [
            XRAY_IDENTITY_MATRIX,
            matrix_set_xyz_translate([0.0, 0.0, 0.0], [3.0, 2.0, 0.0]),
        ];
        let moved_pal = BonePalette::from_model_transforms(index.skeleton, &moved_models).unwrap();
        let live_xform = matrix_set_xyz_translate([0.0, 0.0, 0.0], [8.0, 1.0, 0.0]);

        let view = matrix_build_camera([10.0, 3.0, -6.0], [10.0, 3.0, 0.0], [0.0, 1.0, 0.0]);
        let proj = matrix_build_projection(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);
        let full = matrix_mul_44(&proj, &view);
        let frustum = ClipFrustum::from_view_proj(&full, frustum_planes::LRTB | frustum_planes::FAR);

        // Elapsed (27.0 - 2.0) / 50.0 = 0.5 -> alpha ~ 127
        let batches = pool
            .collect_visible_batches(27.0, &frustum, |_| Some((&live_xform, &moved_pal)))
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].vertices.len(), 6);
        let a_byte = (batches[0].vertices[0].color >> 24) & 0xFF;
        assert!((126..=128).contains(&a_byte));

        // 1W corner 1 of face 0 was bind (0.0, 2.4, 0.0) on bone 1 -> model (3.0, 2.4, 0.0) -> world (11.0, 3.4, 0.0)
        assert!((batches[0].vertices[1].position[0] - 11.0).abs() < 1e-4);
        assert!((batches[0].vertices[1].position[1] - 3.4).abs() < 1e-4);
        // 3W corner 1 of face 1 had 0.75 weight on bone 1 -> model (2.25, 2.2, 0.0) -> world (10.25, 3.2, 0.0)
        assert!((batches[0].vertices[4].position[0] - 10.25).abs() < 1e-4);
        assert!((batches[0].vertices[4].position[1] - 3.2).abs() < 1e-4);

        // Frustum looking away culls emission without removing the wallmark
        let away_view = matrix_build_camera([10.0, 3.0, -6.0], [10.0, 3.0, -10.0], [0.0, 1.0, 0.0]);
        let away_frustum =
            ClipFrustum::from_view_proj(&matrix_mul_44(&proj, &away_view), frustum_planes::LRTB);
        let culled = pool
            .collect_visible_batches(27.0, &away_frustum, |_| Some((&live_xform, &moved_pal)))
            .unwrap();
        assert!(culled.is_empty());
        assert_eq!(pool.len(), 1);

        // Expiry at time >= time_start + time_end prunes immediately (SkeletonCustom.cpp:830)
        let expired = pool
            .collect_visible_batches(52.0, &frustum, |_| Some((&live_xform, &moved_pal)))
            .unwrap();
        assert!(expired.is_empty());
        assert_eq!(pool.len(), 0);
    }
}
