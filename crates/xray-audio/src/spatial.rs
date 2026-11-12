//! 3D Spatial Audio Math, Attenuation Curves, Stereo Panning & `level.som` Occlusion
//!
//! Coordinates listener transforms (`FirstPersonCamera` vectors), X-Ray `.ogg` falloff
//! parameters (`min_dist`, `max_dist`, `base_volume`), inverse-distance attenuation with smooth
//! horizon fadeout, equal-power stereo panning, and optional `SoundOcclusionMesh` ray transmission.

use xray_formats::sound_env::{OggXRayParams, SoundOcclusionMesh};

/// Speed of sound in air (m/s) used for mild physical Doppler calculations.
pub const SPEED_OF_SOUND_MPS: f32 = 343.3;

/// Distance attenuation curve model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttenuationModel {
    /// X-Ray OpenAL/XAudio inverse-clamped model with smooth tail fade reaching 0 at `max_dist`.
    InverseClamped { rolloff: f32 },
    /// Linear ramp from 1.0 at `min_dist` to 0.0 at `max_dist`.
    Linear,
    /// Fixed volume regardless of distance (for 2D HUD / global wind / thunder fallback).
    None,
}

impl Default for AttenuationModel {
    fn default() -> Self {
        Self::InverseClamped { rolloff: 1.0 }
    }
}

/// World-space 3D audio listener (synced each frame with `FirstPersonCamera`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioListener {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub right: [f32; 3],
    pub velocity: [f32; 3],
}

impl Default for AudioListener {
    fn default() -> Self {
        Self {
            position: [0.0, 1.8, 0.0],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        }
    }
}

impl AudioListener {
    /// Construct a normalized listener from camera basis vectors.
    pub fn from_basis(
        position: [f32; 3],
        forward: [f32; 3],
        up: [f32; 3],
        right: [f32; 3],
        velocity: [f32; 3],
    ) -> Self {
        let fwd = normalize_or(forward, [0.0, 0.0, 1.0]);
        let rgt = normalize_or(right, [1.0, 0.0, 0.0]);
        let upv = normalize_or(up, [0.0, 1.0, 0.0]);
        Self {
            position,
            forward: fwd,
            up: upv,
            right: rgt,
            velocity,
        }
    }

    /// Construct a listener directly from left-handed yaw/pitch angles matching `FirstPersonCamera`.
    pub fn from_yaw_pitch(position: [f32; 3], yaw: f32, pitch: f32, velocity: [f32; 3]) -> Self {
        let cos_p = pitch.cos();
        let fwd = [cos_p * yaw.sin(), pitch.sin(), cos_p * yaw.cos()];
        let right = [yaw.cos(), 0.0, -yaw.sin()];
        let up = cross(fwd, right);
        Self::from_basis(position, fwd, up, right, velocity)
    }
}

/// World-space 3D sound emitter properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioEmitter3D {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub min_dist: f32,
    pub max_dist: f32,
    pub base_volume: f32,
    pub gain: f32,
    pub pitch: f32,
    pub spatialized: bool,
    pub occlusion_enabled: bool,
    pub doppler_factor: f32,
    pub attenuation: AttenuationModel,
}

impl Default for AudioEmitter3D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            min_dist: 1.0,
            max_dist: 30.0,
            base_volume: 1.0,
            gain: 1.0,
            pitch: 1.0,
            spatialized: true,
            occlusion_enabled: true,
            doppler_factor: 0.25,
            attenuation: AttenuationModel::default(),
        }
    }
}

impl AudioEmitter3D {
    /// Populate distance falloff & base volume from parsed X-Ray `.ogg` comment metadata.
    pub fn from_xray_params(position: [f32; 3], params: &OggXRayParams) -> Self {
        Self {
            position,
            min_dist: params.min_dist.max(0.1),
            max_dist: params.max_dist.max(params.min_dist.max(0.1) + 0.1),
            base_volume: params.base_volume.clamp(0.0, 4.0),
            ..Self::default()
        }
    }
}

/// Output stereo channel gains and effective playback pitch computed by [`compute_spatial_gains`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialMixGains {
    pub left_gain: f32,
    pub right_gain: f32,
    pub distance: f32,
    pub distance_attenuation: f32,
    pub occlusion_transmission: f32,
    pub pan: f32,
    pub effective_pitch: f32,
}

/// Compute 3D spatial stereo gains, occlusion transmission, and Doppler pitch.
pub fn compute_spatial_gains(
    listener: &AudioListener,
    emitter: &AudioEmitter3D,
    occlusion: Option<&SoundOcclusionMesh>,
) -> SpatialMixGains {
    let base_amp = (emitter.base_volume * emitter.gain).max(0.0);

    if !emitter.spatialized {
        let eq = std::f32::consts::FRAC_1_SQRT_2 * base_amp;
        return SpatialMixGains {
            left_gain: eq,
            right_gain: eq,
            distance: 0.0,
            distance_attenuation: 1.0,
            occlusion_transmission: 1.0,
            pan: 0.0,
            effective_pitch: emitter.pitch.clamp(0.25, 4.0),
        };
    }

    let dx = emitter.position[0] - listener.position[0];
    let dy = emitter.position[1] - listener.position[1];
    let dz = emitter.position[2] - listener.position[2];
    let dist_sq = dx * dx + dy * dy + dz * dz;
    let dist = dist_sq.sqrt();

    let min_d = emitter.min_dist.max(0.05);
    let max_d = emitter.max_dist.max(min_d + 0.05);

    let distance_attenuation = match emitter.attenuation {
        AttenuationModel::None => 1.0,
        AttenuationModel::Linear => {
            if dist <= min_d {
                1.0
            } else if dist >= max_d {
                0.0
            } else {
                1.0 - (dist - min_d) / (max_d - min_d)
            }
        }
        AttenuationModel::InverseClamped { rolloff } => {
            if dist <= min_d {
                1.0
            } else if dist >= max_d {
                0.0
            } else {
                let r = rolloff.max(0.01);
                let inv = min_d / (min_d + r * (dist - min_d));
                // Smooth horizon taper across the outer 15% of [min_d, max_d] so distant sources
                // reach exact 0.0 cleanly instead of snapping from non-zero inverse gain.
                let t = (dist - min_d) / (max_d - min_d);
                let tail = if t > 0.85 {
                    let s = ((1.0 - t) / 0.15).clamp(0.0, 1.0);
                    s * s * (3.0 - 2.0 * s)
                } else {
                    1.0
                };
                (inv * tail).clamp(0.0, 1.0)
            }
        }
    };

    let occlusion_transmission = if emitter.occlusion_enabled && distance_attenuation > 1e-4 {
        occlusion
            .map(|som| som.query_occlusion(listener.position, emitter.position).clamp(0.0, 1.0))
            .unwrap_or(1.0)
    } else {
        1.0
    };

    // Equal-power stereo panning scaled down gracefully inside `min_dist` so close sounds stay centered.
    let pan = if dist > 1e-5 {
        let inv_d = 1.0 / dist;
        let dir = [dx * inv_d, dy * inv_d, dz * inv_d];
        let raw_pan = dot(dir, listener.right).clamp(-1.0, 1.0);
        let proximity = (dist / min_d).clamp(0.0, 1.0);
        raw_pan * proximity
    } else {
        0.0
    };

    // Angle mapping: pan in [-1, 1] -> theta in [0, pi/2] => L = cos(theta), R = sin(theta)
    let theta = (pan + 1.0) * 0.25 * std::f32::consts::PI;
    let total_amp = base_amp * distance_attenuation * occlusion_transmission;
    let left_gain = theta.cos() * total_amp;
    let right_gain = theta.sin() * total_amp;

    // Subtle physical Doppler factor
    let effective_pitch = if emitter.doppler_factor > 0.0 && dist > 1e-4 {
        let inv_d = 1.0 / dist;
        let dir_to_emitter = [dx * inv_d, dy * inv_d, dz * inv_d];
        let v_listener = dot(listener.velocity, dir_to_emitter);
        let v_emitter = dot(emitter.velocity, dir_to_emitter);
        let scale = emitter.doppler_factor.clamp(0.0, 1.0);
        let num = (SPEED_OF_SOUND_MPS + v_listener * scale).clamp(50.0, 600.0);
        let den = (SPEED_OF_SOUND_MPS + v_emitter * scale).clamp(50.0, 600.0);
        (emitter.pitch * (num / den)).clamp(0.25, 4.0)
    } else {
        emitter.pitch.clamp(0.25, 4.0)
    };

    SpatialMixGains {
        left_gain,
        right_gain,
        distance: dist,
        distance_attenuation,
        occlusion_transmission,
        pan,
        effective_pitch,
    }
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = dot(v, v);
    if len_sq > 1e-10 {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_attenuation_and_stereo_panning() {
        let listener = AudioListener::from_yaw_pitch([0.0, 1.65, 0.0], 0.0, 0.0, [0.0; 3]);

        // Sound directly to the right (+X)
        let emitter_right = AudioEmitter3D {
            position: [5.0, 1.65, 0.0],
            min_dist: 1.0,
            max_dist: 30.0,
            ..Default::default()
        };
        let gains_r = compute_spatial_gains(&listener, &emitter_right, None);
        assert!(gains_r.pan > 0.99);
        assert!(gains_r.right_gain > gains_r.left_gain * 5.0);
        assert!(gains_r.distance_attenuation > 0.1 && gains_r.distance_attenuation < 0.3);

        // Sound directly to the left (-X)
        let mut emitter_left = emitter_right;
        emitter_left.position = [-5.0, 1.65, 0.0];
        let gains_l = compute_spatial_gains(&listener, &emitter_left, None);
        assert!(gains_l.pan < -0.99);
        assert!(gains_l.left_gain > gains_l.right_gain * 5.0);

        // Sound beyond max_dist fades cleanly to silence
        let mut emitter_far = emitter_right;
        emitter_far.position = [40.0, 1.65, 0.0];
        let gains_far = compute_spatial_gains(&listener, &emitter_far, None);
        assert_eq!(gains_far.distance_attenuation, 0.0);
        assert_eq!(gains_far.left_gain, 0.0);
        assert_eq!(gains_far.right_gain, 0.0);
    }
}
