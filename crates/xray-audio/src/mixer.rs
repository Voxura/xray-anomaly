//! Pure-Rust Multi-Voice Spatial Software Audio Mixer (`xray-audio`)
//!
//! Resamples mono/stereo clips to the mixer sample rate (`48000 Hz`), applies smoothed
//! 3D spatial stereo gains per voice, respects X-Ray sound categories, and sums active voices
//! with a soft master limiter.

use std::collections::BTreeMap;
use std::sync::Arc;

use xray_formats::sound_env::{OggXRayParams, SoundOcclusionMesh};

use crate::spatial::{compute_spatial_gains, AudioEmitter3D, AudioListener};

/// Default shared mixer output sample rate matching Windows WASAPI shared mode.
pub const DEFAULT_MIXER_SAMPLE_RATE: u32 = 48_000;

/// Unique opaque identifier for a playing or queued voice inside [`SpatialAudioMixer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VoiceId(pub u64);

/// Logical mixing bus categories for independent volume control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AudioCategory {
    Sfx,
    Footstep,
    Weapon,
    Ambient,
    Weather,
    Dialogue,
}

/// Decoded PCM audio clip ready for fast lock-free-style mixing across voices.
#[derive(Debug, Clone)]
pub struct SoundClip {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u8,
    /// Interleaved `f32` PCM samples normalized to `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    pub xray_params: OggXRayParams,
}

impl SoundClip {
    pub fn frame_count(&self) -> usize {
        let ch = usize::from(self.channels.max(1));
        self.samples.len() / ch
    }

    pub fn duration_seconds(&self) -> f32 {
        self.frame_count() as f32 / self.sample_rate.max(1) as f32
    }

    /// Sample clip cleanly using linear interpolation across fractional source frame position.
    #[inline]
    fn sample_mono_at(&self, frame_pos: f64) -> f32 {
        let frames = self.frame_count();
        if frames == 0 {
            return 0.0;
        }
        let ch = usize::from(self.channels.max(1));
        let idx0 = (frame_pos.floor() as usize).min(frames - 1);
        let idx1 = (idx0 + 1).min(frames - 1);
        let frac = (frame_pos - frame_pos.floor()) as f32;

        let s0 = if ch == 1 {
            self.samples[idx0]
        } else {
            0.5 * (self.samples[idx0 * ch] + self.samples[idx0 * ch + 1])
        };
        let s1 = if ch == 1 {
            self.samples[idx1]
        } else {
            0.5 * (self.samples[idx1 * ch] + self.samples[idx1 * ch + 1])
        };
        s0 + (s1 - s0) * frac
    }

    #[inline]
    fn sample_stereo_at(&self, frame_pos: f64) -> [f32; 2] {
        let frames = self.frame_count();
        if frames == 0 {
            return [0.0, 0.0];
        }
        let ch = usize::from(self.channels.max(1));
        if ch == 1 {
            let s = self.sample_mono_at(frame_pos);
            return [s, s];
        }
        let idx0 = (frame_pos.floor() as usize).min(frames - 1);
        let idx1 = (idx0 + 1).min(frames - 1);
        let frac = (frame_pos - frame_pos.floor()) as f32;
        let l0 = self.samples[idx0 * ch];
        let r0 = self.samples[idx0 * ch + 1];
        let l1 = self.samples[idx1 * ch];
        let r1 = self.samples[idx1 * ch + 1];
        [l0 + (l1 - l0) * frac, r0 + (r1 - r0) * frac]
    }
}

/// Active voice entry managed by [`SpatialAudioMixer`].
#[derive(Debug, Clone)]
pub struct ActiveVoice {
    pub id: VoiceId,
    pub category: AudioCategory,
    pub clip: Arc<SoundClip>,
    pub emitter: AudioEmitter3D,
    pub looping: bool,
    pub paused: bool,
    pub frame_cursor: f64,
    pub target_left_gain: f32,
    pub target_right_gain: f32,
    pub current_left_gain: f32,
    pub current_right_gain: f32,
    pub effective_pitch: f32,
}

/// Pure-Rust real-time stereo software spatial mixer.
pub struct SpatialAudioMixer {
    pub sample_rate: u32,
    pub master_volume: f32,
    pub category_volumes: BTreeMap<AudioCategory, f32>,
    pub listener: AudioListener,
    pub occlusion_mesh: Option<Arc<SoundOcclusionMesh>>,
    pub total_frames_mixed: u64,
    next_voice_id: u64,
    voices: BTreeMap<VoiceId, ActiveVoice>,
}

impl Default for SpatialAudioMixer {
    fn default() -> Self {
        Self::new(DEFAULT_MIXER_SAMPLE_RATE)
    }
}

impl SpatialAudioMixer {
    pub fn new(sample_rate: u32) -> Self {
        let mut category_volumes = BTreeMap::new();
        category_volumes.insert(AudioCategory::Sfx, 1.0);
        category_volumes.insert(AudioCategory::Footstep, 0.85);
        category_volumes.insert(AudioCategory::Weapon, 1.0);
        category_volumes.insert(AudioCategory::Ambient, 0.75);
        category_volumes.insert(AudioCategory::Weather, 0.8);
        category_volumes.insert(AudioCategory::Dialogue, 1.0);

        Self {
            sample_rate: sample_rate.max(8_000),
            master_volume: 1.0,
            category_volumes,
            listener: AudioListener::default(),
            occlusion_mesh: None,
            total_frames_mixed: 0,
            next_voice_id: 1,
            voices: BTreeMap::new(),
        }
    }

    pub fn set_category_volume(&mut self, cat: AudioCategory, volume: f32) {
        self.category_volumes.insert(cat, volume.clamp(0.0, 2.0));
    }

    pub fn category_volume(&self, cat: AudioCategory) -> f32 {
        self.category_volumes.get(&cat).copied().unwrap_or(1.0)
    }

    pub fn active_voice_count(&self) -> usize {
        self.voices.len()
    }

    /// Spawn a new playing voice and evaluate its initial spatial stereo gains immediately.
    pub fn play(
        &mut self,
        clip: Arc<SoundClip>,
        category: AudioCategory,
        emitter: AudioEmitter3D,
        looping: bool,
    ) -> VoiceId {
        let id = VoiceId(self.next_voice_id);
        self.next_voice_id += 1;

        let spatial = compute_spatial_gains(&self.listener, &emitter, self.occlusion_mesh.as_deref());
        let cat_vol = self.category_volume(category) * self.master_volume.clamp(0.0, 2.0);
        let init_l = spatial.left_gain * cat_vol;
        let init_r = spatial.right_gain * cat_vol;

        self.voices.insert(
            id,
            ActiveVoice {
                id,
                category,
                clip,
                emitter,
                looping,
                paused: false,
                frame_cursor: 0.0,
                target_left_gain: init_l,
                target_right_gain: init_r,
                current_left_gain: init_l,
                current_right_gain: init_r,
                effective_pitch: spatial.effective_pitch,
            },
        );
        id
    }

    pub fn get_voice(&self, id: VoiceId) -> Option<&ActiveVoice> {
        self.voices.get(&id)
    }

    pub fn update_voice_emitter(&mut self, id: VoiceId, emitter: AudioEmitter3D) -> bool {
        if let Some(v) = self.voices.get_mut(&id) {
            v.emitter = emitter;
            true
        } else {
            false
        }
    }

    pub fn stop_voice(&mut self, id: VoiceId) -> bool {
        self.voices.remove(&id).is_some()
    }

    pub fn stop_all(&mut self) {
        self.voices.clear();
    }

    /// Recompute target spatial gains for all active voices from the current [`AudioListener`].
    pub fn refresh_spatial_gains(&mut self) {
        let som = self.occlusion_mesh.as_deref();
        let master = self.master_volume.clamp(0.0, 2.0);
        let listener = &self.listener;

        for voice in self.voices.values_mut() {
            let cat_vol = self
                .category_volumes
                .get(&voice.category)
                .copied()
                .unwrap_or(1.0);
            let gains = compute_spatial_gains(listener, &voice.emitter, som);
            voice.target_left_gain = gains.left_gain * cat_vol * master;
            voice.target_right_gain = gains.right_gain * cat_vol * master;
            voice.effective_pitch = gains.effective_pitch;
        }
    }

    /// Mix `out_stereo.len() / 2` interleaved `[L, R, L, R, ...]` float32 frames cleanly.
    pub fn mix_frames(&mut self, out_stereo: &mut [f32]) {
        out_stereo.fill(0.0);
        let out_frames = out_stereo.len() / 2;
        if out_frames == 0 {
            return;
        }

        self.refresh_spatial_gains();

        let mix_sr = self.sample_rate as f64;
        let inv_frames = 1.0 / out_frames as f32;
        let mut finished = Vec::new();

        for (&id, voice) in &mut self.voices {
            if voice.paused {
                continue;
            }
            let src_frames = voice.clip.frame_count();
            if src_frames == 0 {
                finished.push(id);
                continue;
            }

            let step = (voice.clip.sample_rate as f64 / mix_sr) * voice.effective_pitch as f64;
            let start_l = voice.current_left_gain;
            let start_r = voice.current_right_gain;
            let delta_l = (voice.target_left_gain - start_l) * inv_frames;
            let delta_r = (voice.target_right_gain - start_r) * inv_frames;

            // Optimization: skip audible mixing if voice is completely inaudible on both ends,
            // while still advancing its playback cursor cleanly.
            let silent = start_l.abs() < 1e-6
                && start_r.abs() < 1e-6
                && voice.target_left_gain.abs() < 1e-6
                && voice.target_right_gain.abs() < 1e-6;

            if silent {
                voice.frame_cursor += step * out_frames as f64;
                if voice.frame_cursor >= src_frames as f64 {
                    if voice.looping {
                        voice.frame_cursor %= src_frames as f64;
                    } else {
                        finished.push(id);
                    }
                }
                voice.current_left_gain = voice.target_left_gain;
                voice.current_right_gain = voice.target_right_gain;
                continue;
            }

            let spatial_non_spatial = !voice.emitter.spatialized && voice.clip.channels >= 2;
            for frame_idx in 0..out_frames {
                if voice.frame_cursor >= src_frames as f64 {
                    if voice.looping {
                        voice.frame_cursor %= src_frames as f64;
                    } else {
                        finished.push(id);
                        break;
                    }
                }

                let gl = start_l + delta_l * frame_idx as f32;
                let gr = start_r + delta_r * frame_idx as f32;

                let (sl, sr) = if spatial_non_spatial {
                    let s = voice.clip.sample_stereo_at(voice.frame_cursor);
                    (s[0], s[1])
                } else {
                    let m = voice.clip.sample_mono_at(voice.frame_cursor);
                    (m, m)
                };

                let base = frame_idx * 2;
                out_stereo[base] += sl * gl;
                out_stereo[base + 1] += sr * gr;

                voice.frame_cursor += step;
            }

            voice.current_left_gain = voice.target_left_gain;
            voice.current_right_gain = voice.target_right_gain;
        }

        for id in finished {
            self.voices.remove(&id);
        }

        // Smooth soft-knee master limiter to prevent harsh clipping when simultaneous shots/thunder fire
        for sample in out_stereo.iter_mut() {
            if sample.abs() > 0.9 {
                *sample = sample.tanh();
            }
        }

        self.total_frames_mixed = self.total_frames_mixed.saturating_add(out_frames as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_spatial_panning_resampling_and_oneshot_retirement() {
        let mut mixer = SpatialAudioMixer::new(48_000);
        mixer.listener = AudioListener::from_yaw_pitch([0.0, 1.65, 0.0], 0.0, 0.0, [0.0; 3]);

        // 24 kHz sine wave (0.01s = 240 frames) placed to the right (+X)
        let samples: Vec<f32> = (0..240).map(|i| (i as f32 * 0.1).sin()).collect();
        let clip = Arc::new(SoundClip {
            name: "test_tone".to_string(),
            sample_rate: 24_000,
            channels: 1,
            samples,
            xray_params: OggXRayParams::default(),
        });

        let id = mixer.play(
            clip,
            AudioCategory::Sfx,
            AudioEmitter3D {
                position: [3.0, 1.65, 0.0],
                min_dist: 1.0,
                max_dist: 20.0,
                ..Default::default()
            },
            false,
        );
        assert_eq!(mixer.active_voice_count(), 1);

        // First 240 output frames at 48 kHz = 0.005s = half the clip
        let mut buf = vec![0.0f32; 240 * 2];
        mixer.mix_frames(&mut buf);
        assert_eq!(mixer.active_voice_count(), 1);

        let energy_l: f32 = buf.iter().step_by(2).map(|x| x * x).sum();
        let energy_r: f32 = buf.iter().skip(1).step_by(2).map(|x| x * x).sum();
        assert!(energy_r > 0.1);
        assert!(energy_r > energy_l * 10.0, "right channel should dominate");

        // Mix remaining > 0.005s -> one-shot completes cleanly
        let mut buf2 = vec![0.0f32; 400 * 2];
        mixer.mix_frames(&mut buf2);
        assert_eq!(mixer.active_voice_count(), 0);
        assert!(mixer.get_voice(id).is_none());
    }
}
