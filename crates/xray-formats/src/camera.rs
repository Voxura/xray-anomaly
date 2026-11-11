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
use crate::ogf::{matrix_set_xyzi_translate, XRayMatrix, XRAY_IDENTITY_MATRIX};

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
}
