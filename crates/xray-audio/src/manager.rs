//! High-Level Gameplay Audio Coordinator (`xray-audio`)
//!
//! Coordinates `SpatialAudioMixer`, `AudioOutputBackend` (Windows WASAPI / headless fallback),
//! VFS `.ogg` streaming cache (`decode_ogg_from_vfs` + `OggXRayParams`), surface-specific
//! `CformBvh` + `gamemtl.xr` footsteps, dynamic weather loops (wind, rain, 3D thunderclaps),
//! `level.snd_static` ambient point sources, `level.som` acoustic occlusion, and ALife
//! stalker / weapon spatial sound emitters.

use std::collections::{BTreeMap, BTreeSet};
use std::f32::consts::TAU;
use std::sync::{Arc, Mutex};

use xray_formats::gamemtl::{gamemtl_flags, GameMaterialLibrary};
use xray_formats::sound_env::{LevelStaticSounds, OggXRayParams, SoundOcclusionMesh};
use xray_vfs::VirtualFileSystem;

use crate::backend::AudioOutputBackend;
use crate::mixer::{
    AudioCategory, SoundClip, SpatialAudioMixer, VoiceId, DEFAULT_MIXER_SAMPLE_RATE,
};
use crate::ogg_vorbis::decode_ogg_from_vfs;
use crate::spatial::{AttenuationModel, AudioEmitter3D, AudioListener};

/// Physical surface material buckets used for X-Ray footstep & landing audio selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SurfaceMaterialCategory {
    Dirt,
    Grass,
    Gravel,
    Wood,
    Metal,
    Water,
    Concrete,
}

impl SurfaceMaterialCategory {
    pub fn key(self) -> &'static str {
        match self {
            Self::Dirt => "dirt",
            Self::Grass => "grass",
            Self::Gravel => "gravel",
            Self::Wood => "wood",
            Self::Metal => "metal",
            Self::Water => "water",
            Self::Concrete => "concrete",
        }
    }

    /// Standard X-Ray footstep `.ogg` candidates in `$game_sounds$/material/human/step/`.
    pub fn default_step_paths(self) -> &'static [&'static str] {
        match self {
            Self::Dirt => &[
                "material\\human\\step\\earth01",
                "material\\human\\step\\earth02",
                "material\\human\\step\\dirt01",
            ],
            Self::Grass => &[
                "material\\human\\step\\grass01",
                "material\\human\\step\\grass02",
            ],
            Self::Gravel => &[
                "material\\human\\step\\gravel01",
                "material\\human\\step\\gravel02",
                "material\\human\\step\\sand01",
            ],
            Self::Wood => &[
                "material\\human\\step\\wood01",
                "material\\human\\step\\wood02",
            ],
            Self::Metal => &[
                "material\\human\\step\\metal01",
                "material\\human\\step\\tin01",
            ],
            Self::Water => &[
                "material\\human\\step\\water01",
                "material\\human\\step\\water02",
            ],
            Self::Concrete => &[
                "material\\human\\step\\asphalt01",
                "material\\human\\step\\concrete01",
                "material\\human\\step\\stone01",
            ],
        }
    }
}

/// Map an X-Ray material identifier (e.g. `"materials\\wood_solid"`, `"materials\\bush"`) to
/// a normalized footstep audio bucket.
pub fn classify_material_name(name: &str) -> SurfaceMaterialCategory {
    let lower = name.to_ascii_lowercase();
    if lower.contains("water") || lower.contains("puddle") || lower.contains("swamp") || lower.contains("liquid") {
        SurfaceMaterialCategory::Water
    } else if lower.contains("metal") || lower.contains("iron") || lower.contains("steel") || lower.contains("tin") || lower.contains("grate") || lower.contains("barrel") {
        SurfaceMaterialCategory::Metal
    } else if lower.contains("wood") || lower.contains("plank") || lower.contains("tree") || lower.contains("bark") {
        SurfaceMaterialCategory::Wood
    } else if lower.contains("gravel") || lower.contains("pebble") || lower.contains("rubble") || lower.contains("sand") || lower.contains("crushed") {
        SurfaceMaterialCategory::Gravel
    } else if lower.contains("grass") || lower.contains("bush") || lower.contains("leaf") || lower.contains("leaves") || lower.contains("moss") || lower.contains("reed") {
        SurfaceMaterialCategory::Grass
    } else if lower.contains("concrete") || lower.contains("asphalt") || lower.contains("brick") || lower.contains("stone") || lower.contains("tile") || lower.contains("slate") || lower.contains("glass") {
        SurfaceMaterialCategory::Concrete
    } else {
        SurfaceMaterialCategory::Dirt
    }
}

/// Resolve a `CformTriangle::material` ID using `gamemtl.xr`, returning both the high-level
/// [`SurfaceMaterialCategory`] and any explicit VFS `step_sounds` listed on the material pair.
pub fn resolve_surface_category(
    cform_material_id: Option<u16>,
    gamemtl: Option<&GameMaterialLibrary>,
) -> (SurfaceMaterialCategory, Vec<String>) {
    let Some(id) = cform_material_id else {
        return (SurfaceMaterialCategory::Concrete, Vec::new());
    };
    let Some(lib) = gamemtl else {
        // Without gamemtl.xr loaded, deterministic modulo fallback keeps tests & minimal scenes expressive
        return (SurfaceMaterialCategory::Dirt, Vec::new());
    };

    let mat = lib
        .material_by_idx(usize::from(id))
        .or_else(|| lib.material_by_id(u32::from(id)));
    let Some(mat) = mat else {
        return (SurfaceMaterialCategory::Dirt, Vec::new());
    };

    let cat = if mat.has_flag(gamemtl_flags::FL_LIQUID) {
        SurfaceMaterialCategory::Water
    } else {
        classify_material_name(&mat.name)
    };

    // Look up step sounds between `default_actor` / `actor` / `human` and this surface
    let mut step_paths = Vec::new();
    let actor_id = lib
        .material_by_name("actor")
        .or_else(|| lib.material_by_name("default_object"))
        .or_else(|| lib.material_by_name("default"))
        .map(|m| m.id);
    if let Some(aid) = actor_id
        && let Some(pair) = lib.find_pair(aid, mat.id)
    {
        for s in pair.step_sounds() {
            let t = s.trim();
            if !t.is_empty() {
                step_paths.push(t.to_string());
            }
        }
    }

    (cat, step_paths)
}

/// Configuration for [`AudioManager`].
#[derive(Debug, Clone)]
pub struct AudioManagerConfig {
    pub sample_rate: u32,
    pub prefer_hardware_output: bool,
    pub master_volume: f32,
}

impl Default for AudioManagerConfig {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_MIXER_SAMPLE_RATE,
            prefer_hardware_output: true,
            master_volume: 1.0,
        }
    }
}

/// Runtime telemetry snapshot exported by [`AudioManager::stats`].
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEngineStats {
    pub hardware_backend_active: bool,
    pub sample_rate: u32,
    pub active_voices: usize,
    pub cached_clips: usize,
    pub footsteps_triggered: u64,
    pub landings_triggered: u64,
    pub last_footstep_surface: Option<SurfaceMaterialCategory>,
    pub thunder_triggered: u64,
    pub weapon_sounds_triggered: u64,
    pub wind_gain: f32,
    pub rain_gain: f32,
    pub active_static_sounds: usize,
    pub tracked_alife_emitters: usize,
    pub total_frames_mixed: u64,
}

#[derive(Debug, Clone)]
struct AlifeEmitterState {
    position: [f32; 3],
    velocity: [f32; 3],
    voice_id: Option<VoiceId>,
    step_accum_m: f32,
}

/// Complete high-level spatial audio manager integrated into `EngineRuntime`.
pub struct AudioManager {
    mixer: Arc<Mutex<SpatialAudioMixer>>,
    backend: AudioOutputBackend,
    vfs: Option<Arc<VirtualFileSystem>>,
    clip_cache: BTreeMap<String, Arc<SoundClip>>,

    // Footstep kinematics state
    footstep_distance_m: f32,
    was_grounded: bool,
    air_time_sec: f32,
    step_parity: u64,
    footsteps_triggered: u64,
    landings_triggered: u64,
    last_footstep_surface: Option<SurfaceMaterialCategory>,

    // Weather looping voices
    wind_voice: Option<VoiceId>,
    rain_voice: Option<VoiceId>,
    wind_gain: f32,
    rain_gain: f32,
    thunder_triggered: u64,

    // Level static & ALife emitters
    static_voices: BTreeMap<usize, VoiceId>,
    alife_emitters: BTreeMap<u64, AlifeEmitterState>,
    weapon_sounds_triggered: u64,
}

impl AudioManager {
    /// Initialize `AudioManager` with WASAPI hardware output (falling back cleanly to headless
    /// software mode when `--headless` is set or no audio device is present).
    pub fn new(config: AudioManagerConfig) -> Self {
        let mut mixer_inner = SpatialAudioMixer::new(config.sample_rate);
        mixer_inner.master_volume = config.master_volume;
        let mixer = Arc::new(Mutex::new(mixer_inner));
        let backend = AudioOutputBackend::open_default(
            Arc::clone(&mixer),
            config.prefer_hardware_output,
        );

        Self {
            mixer,
            backend,
            vfs: None,
            clip_cache: BTreeMap::new(),
            footstep_distance_m: 0.0,
            was_grounded: true,
            air_time_sec: 0.0,
            step_parity: 0,
            footsteps_triggered: 0,
            landings_triggered: 0,
            last_footstep_surface: None,
            wind_voice: None,
            rain_voice: None,
            wind_gain: 0.0,
            rain_gain: 0.0,
            thunder_triggered: 0,
            static_voices: BTreeMap::new(),
            alife_emitters: BTreeMap::new(),
            weapon_sounds_triggered: 0,
        }
    }

    pub fn new_headless() -> Self {
        Self::new(AudioManagerConfig {
            prefer_hardware_output: false,
            ..Default::default()
        })
    }

    pub fn attach_vfs(&mut self, vfs: Arc<VirtualFileSystem>) {
        self.vfs = Some(vfs);
    }

    pub fn set_occlusion_mesh(&mut self, som: Option<Arc<SoundOcclusionMesh>>) {
        if let Ok(mut m) = self.mixer.lock() {
            m.occlusion_mesh = som;
        }
    }

    pub fn update_listener(&self, listener: AudioListener) {
        if let Ok(mut m) = self.mixer.lock() {
            m.listener = listener;
        }
    }

    pub fn set_master_volume(&self, volume: f32) {
        if let Ok(mut m) = self.mixer.lock() {
            m.master_volume = volume.clamp(0.0, 2.0);
        }
    }

    pub fn set_category_volume(&self, category: AudioCategory, volume: f32) {
        if let Ok(mut m) = self.mixer.lock() {
            m.set_category_volume(category, volume);
        }
    }

    pub fn register_clip(&mut self, key: impl Into<String>, clip: Arc<SoundClip>) {
        self.clip_cache.insert(key.into().to_ascii_lowercase(), clip);
    }

    /// Mix `frames` stereo frames synchronously into an interleaved `f32` buffer
    /// (primarily for `--headless` determinism and offline verification tests).
    pub fn mix_headless_frames(&self, frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; frames * 2];
        if let Ok(mut m) = self.mixer.lock() {
            m.mix_frames(&mut out);
        }
        out
    }

    pub fn stats(&self) -> AudioEngineStats {
        let (active_voices, total_frames_mixed) = self
            .mixer
            .lock()
            .map(|m| (m.active_voice_count(), m.total_frames_mixed))
            .unwrap_or((0, 0));

        AudioEngineStats {
            hardware_backend_active: self.backend.is_hardware_active(),
            sample_rate: self.backend.sample_rate(),
            active_voices,
            cached_clips: self.clip_cache.len(),
            footsteps_triggered: self.footsteps_triggered,
            landings_triggered: self.landings_triggered,
            last_footstep_surface: self.last_footstep_surface,
            thunder_triggered: self.thunder_triggered,
            weapon_sounds_triggered: self.weapon_sounds_triggered,
            wind_gain: self.wind_gain,
            rain_gain: self.rain_gain,
            active_static_sounds: self.static_voices.len(),
            tracked_alife_emitters: self.alife_emitters.len(),
            total_frames_mixed,
        }
    }

    /// Resolve or decode an `.ogg` clip from VFS, caching it cleanly. Returns `None` if unavailable.
    pub fn load_clip_from_vfs(&mut self, sound_ref: &str) -> Option<Arc<SoundClip>> {
        let key = sound_ref.trim().to_ascii_lowercase();
        if key.is_empty() {
            return None;
        }
        if let Some(hit) = self.clip_cache.get(&key) {
            return Some(Arc::clone(hit));
        }
        let vfs = self.vfs.as_ref()?;
        if let Ok(decoded) = decode_ogg_from_vfs(vfs, sound_ref) {
            let clip = decoded.into_sound_clip(sound_ref);
            self.clip_cache.insert(key, Arc::clone(&clip));
            return Some(clip);
        }
        None
    }

    fn resolve_or_synthesize<F>(&mut self, candidates: &[&str], synth_key: &str, synth: F) -> Arc<SoundClip>
    where
        F: FnOnce() -> SoundClip,
    {
        for &cand in candidates {
            if let Some(clip) = self.load_clip_from_vfs(cand) {
                return clip;
            }
        }
        let skey = format!("__synth__:{synth_key}");
        if let Some(existing) = self.clip_cache.get(&skey) {
            return Arc::clone(existing);
        }
        let generated = Arc::new(synth());
        self.clip_cache.insert(skey, Arc::clone(&generated));
        generated
    }

    /// Play any `.ogg` sound path (or fallback tone if missing in minimal VFS) in 3D space,
    /// honoring its parsed `OggXRayParams` distance falloff & base volume.
    pub fn play_sound_3d(
        &mut self,
        sound_ref: &str,
        category: AudioCategory,
        position: [f32; 3],
        velocity: [f32; 3],
        looping: bool,
        gain_scale: f32,
    ) -> VoiceId {
        let clip = self.resolve_or_synthesize(&[sound_ref], sound_ref, || {
            synthesize_generic_sfx(sound_ref)
        });
        let mut emitter = AudioEmitter3D::from_xray_params(position, &clip.xray_params);
        emitter.velocity = velocity;
        emitter.gain = gain_scale.max(0.0);

        self.play_clip_internal(clip, category, emitter, looping)
    }

    fn play_clip_internal(
        &self,
        clip: Arc<SoundClip>,
        category: AudioCategory,
        emitter: AudioEmitter3D,
        looping: bool,
    ) -> VoiceId {
        if let Ok(mut m) = self.mixer.lock() {
            m.play(clip, category, emitter, looping)
        } else {
            VoiceId(0)
        }
    }

    /// Step footstep & landing audio from player locomotion + `CformBvh` / `gamemtl.xr` surface.
    #[allow(clippy::too_many_arguments)]
    pub fn step_footsteps(
        &mut self,
        dt_sec: f32,
        feet_pos: [f32; 3],
        horizontal_speed_mps: f32,
        grounded: bool,
        crouched: bool,
        ground_material: Option<u16>,
        gamemtl: Option<&GameMaterialLibrary>,
    ) -> Option<VoiceId> {
        let (surface, pair_sounds) = resolve_surface_category(ground_material, gamemtl);

        if !grounded {
            self.air_time_sec += dt_sec.max(0.0);
            self.was_grounded = false;
            return None;
        }

        // Landing impact thud after noticeable airborne time
        let just_landed = !self.was_grounded && self.air_time_sec > 0.12;
        self.was_grounded = true;
        let air_dur = self.air_time_sec;
        self.air_time_sec = 0.0;

        if just_landed {
            self.landings_triggered += 1;
            self.footstep_distance_m = 0.0;
            let impact_gain = (0.55 + air_dur * 0.7).clamp(0.5, 1.2);
            return Some(self.trigger_footstep_on_surface(
                surface,
                &pair_sounds,
                feet_pos,
                impact_gain,
                0.86,
            ));
        }

        if horizontal_speed_mps < 0.45 {
            // Gently decay stride accumulator while stationary so stopping & moving feels crisp
            self.footstep_distance_m *= 0.85;
            return None;
        }

        let stride_len = if crouched {
            1.35
        } else if horizontal_speed_mps > 4.8 {
            2.15
        } else {
            1.80
        };

        self.footstep_distance_m += horizontal_speed_mps * dt_sec.max(0.0);
        if self.footstep_distance_m >= stride_len {
            self.footstep_distance_m -= stride_len;
            let gain = if crouched {
                0.42
            } else if horizontal_speed_mps > 4.8 {
                0.95
            } else {
                0.75
            };
            let pitch = if (self.step_parity & 1) == 0 {
                0.97
            } else {
                1.03
            };
            self.step_parity += 1;
            return Some(self.trigger_footstep_on_surface(
                surface,
                &pair_sounds,
                feet_pos,
                gain,
                pitch,
            ));
        }

        None
    }

    pub fn trigger_footstep_on_surface(
        &mut self,
        surface: SurfaceMaterialCategory,
        pair_sounds: &[String],
        feet_pos: [f32; 3],
        gain: f32,
        pitch: f32,
    ) -> VoiceId {
        let idx = self.step_parity as usize;
        let clip = if !pair_sounds.is_empty() {
            let pick = pair_sounds[idx % pair_sounds.len()].as_str();
            self.load_clip_from_vfs(pick)
        } else {
            None
        }
        .unwrap_or_else(|| {
            let defaults = surface.default_step_paths();
            let chosen = defaults[idx % defaults.len()];
            self.resolve_or_synthesize(
                &[chosen],
                &format!("step_{}", surface.key()),
                || synthesize_footstep(surface),
            )
        });

        let mut emitter = AudioEmitter3D::from_xray_params(feet_pos, &clip.xray_params);
        emitter.min_dist = 0.8;
        emitter.max_dist = emitter.max_dist.max(18.0);
        emitter.gain = gain;
        emitter.pitch = pitch;
        emitter.occlusion_enabled = false;

        self.footsteps_triggered += 1;
        self.last_footstep_surface = Some(surface);
        self.play_clip_internal(clip, AudioCategory::Footstep, emitter, false)
    }

    /// Modulate ambient wind & rain loops from `WeatherLibrary` parameters and fire
    /// 3D positional thunderclaps whenever a thunderbolt triggers.
    pub fn update_weather(
        &mut self,
        wind_velocity: f32,
        rain_density: f32,
        wetness: f32,
        listener_pos: [f32; 3],
        thunder: Option<(&str, [f32; 3], f32)>,
    ) {
        let target_wind = (0.08 + wind_velocity.max(0.0) * 0.045).clamp(0.05, 0.75);
        let target_rain = ((rain_density.max(0.0) * 0.8 + wetness.clamp(0.0, 1.0) * 0.25) * 0.9)
            .clamp(0.0, 0.85);
        self.wind_gain = target_wind;
        self.rain_gain = target_rain;

        // Ensure wind loop exists & update gain
        let wind_emitter = AudioEmitter3D {
            position: listener_pos,
            gain: target_wind,
            spatialized: false,
            occlusion_enabled: false,
            attenuation: AttenuationModel::None,
            ..Default::default()
        };
        if let Some(vid) = self.wind_voice
            && let Ok(mut m) = self.mixer.lock()
            && !m.update_voice_emitter(vid, wind_emitter)
        {
            self.wind_voice = None;
        }
        if self.wind_voice.is_none() && target_wind > 0.01 {
            let clip = self.resolve_or_synthesize(
                &["ambient\\wind_01", "nature\\wind_1"],
                "weather_wind",
                synthesize_wind_loop,
            );
            self.wind_voice =
                Some(self.play_clip_internal(clip, AudioCategory::Weather, wind_emitter, true));
        }

        // Ensure rain patter loop exists / scales cleanly with rain_density & wetness
        let rain_emitter = AudioEmitter3D {
            position: listener_pos,
            gain: target_rain,
            spatialized: false,
            occlusion_enabled: false,
            attenuation: AttenuationModel::None,
            ..Default::default()
        };
        if let Some(vid) = self.rain_voice
            && let Ok(mut m) = self.mixer.lock()
        {
            if target_rain <= 0.005 {
                m.stop_voice(vid);
                self.rain_voice = None;
            } else if !m.update_voice_emitter(vid, rain_emitter) {
                self.rain_voice = None;
            }
        }
        if self.rain_voice.is_none() && target_rain > 0.005 {
            let clip = self.resolve_or_synthesize(
                &["ambient\\rain", "nature\\rain_1"],
                "weather_rain",
                synthesize_rain_loop,
            );
            self.rain_voice =
                Some(self.play_clip_internal(clip, AudioCategory::Weather, rain_emitter, true));
        }

        // 3D positional thunderclap from ThunderboltEffectState
        if let Some((sound_name, sound_pos, max_dist)) = thunder {
            let clip = self.resolve_or_synthesize(
                &[sound_name, "nature\\thunder-1", "ambient\\thunder_1"],
                "weather_thunder",
                synthesize_thunderclap,
            );
            let mut emitter = AudioEmitter3D::from_xray_params(sound_pos, &clip.xray_params);
            emitter.min_dist = 35.0;
            emitter.max_dist = max_dist.max(450.0);
            emitter.gain = 1.05;
            emitter.occlusion_enabled = false;
            self.thunder_triggered += 1;
            self.play_clip_internal(clip, AudioCategory::Weather, emitter, false);
        }
    }

    /// Synchronize looping `level.snd_static` ambient point sources according to current in-game hour.
    pub fn sync_static_sounds(
        &mut self,
        static_sounds: &LevelStaticSounds,
        game_hour_24: f32,
    ) {
        let level_time_ms = (game_hour_24.rem_euclid(24.0) * 3_600_000.0) as u32;
        let mut still_active = BTreeSet::new();
        for (idx, ss) in static_sounds.sounds.iter().enumerate() {
            let is_active = ss.is_active_at(level_time_ms);
            if is_active {
                still_active.insert(idx);
                let already_playing = self
                    .static_voices
                    .get(&idx)
                    .and_then(|&vid| {
                        self.mixer
                            .lock()
                            .ok()
                            .and_then(|m| m.get_voice(vid).map(|_| vid))
                    });
                if already_playing.is_none() {
                    let clip = self.resolve_or_synthesize(
                        &[ss.wav_name.as_str()],
                        &format!("static_{idx}"),
                        || synthesize_ambient_hum(&ss.wav_name),
                    );
                    let mut emitter = AudioEmitter3D::from_xray_params(ss.position, &clip.xray_params);
                    emitter.gain = ss.volume.clamp(0.0, 2.0);
                    emitter.pitch = ss.frequency.clamp(0.25, 4.0);
                    let vid = self.play_clip_internal(clip, AudioCategory::Ambient, emitter, true);
                    self.static_voices.insert(idx, vid);
                }
            }
        }

        // Stop any static voices outside their scheduled hours
        let stale: Vec<usize> = self
            .static_voices
            .keys()
            .copied()
            .filter(|idx| !still_active.contains(idx))
            .collect();
        for idx in stale {
            if let Some(vid) = self.static_voices.remove(&idx)
                && let Ok(mut m) = self.mixer.lock()
            {
                m.stop_voice(vid);
            }
        }
    }

    /// Track an ALife stalker entity's 3D position/velocity, updating active voice spatialization
    /// and emitting rhythmic patrol footsteps when moving within hearing distance.
    pub fn update_alife_emitter(
        &mut self,
        entity_key: u64,
        position: [f32; 3],
        velocity: [f32; 3],
        moving: bool,
        dt_sec: f32,
    ) {
        let speed = (velocity[0] * velocity[0] + velocity[2] * velocity[2]).sqrt();
        let effective_speed = if moving && speed < 0.5 { 2.2 } else { speed };

        let state = self
            .alife_emitters
            .entry(entity_key)
            .or_insert_with(|| AlifeEmitterState {
                position,
                velocity,
                voice_id: None,
                step_accum_m: 0.0,
            });
        state.position = position;
        state.velocity = velocity;

        if let Some(vid) = state.voice_id
            && let Ok(mut m) = self.mixer.lock()
        {
            if let Some(existing) = m.get_voice(vid).cloned() {
                let mut emitter = existing.emitter;
                emitter.position = position;
                emitter.velocity = velocity;
                m.update_voice_emitter(vid, emitter);
            } else {
                state.voice_id = None;
            }
        }

        if moving {
            state.step_accum_m += effective_speed * dt_sec.max(0.0);
            if state.step_accum_m >= 1.9 {
                state.step_accum_m -= 1.9;
                let clip = self.resolve_or_synthesize(
                    &["material\\human\\step\\earth01"],
                    "step_dirt",
                    || synthesize_footstep(SurfaceMaterialCategory::Dirt),
                );
                let mut emitter = AudioEmitter3D::from_xray_params(position, &clip.xray_params);
                emitter.velocity = velocity;
                emitter.min_dist = 1.5;
                emitter.max_dist = emitter.max_dist.max(25.0);
                emitter.gain = 0.65;
                let vid = self.play_clip_internal(clip, AudioCategory::Footstep, emitter, false);
                if let Some(st) = self.alife_emitters.get_mut(&entity_key) {
                    st.voice_id = Some(vid);
                }
            }
        }
    }

    /// Prune ALife audio trackers for entities no longer present in the world.
    pub fn retain_alife_emitters(&mut self, active_keys: &BTreeSet<u64>) {
        self.alife_emitters.retain(|k, _| active_keys.contains(k));
    }

    /// Trigger a spatialized 3D weapon shot / reload sound at `muzzle_pos`.
    pub fn trigger_weapon_sound(
        &mut self,
        sound_ref: &str,
        muzzle_pos: [f32; 3],
        velocity: [f32; 3],
    ) -> VoiceId {
        let clip = self.resolve_or_synthesize(
            &[sound_ref, "weapons\\ak74_shoot"],
            &format!("weapon_{sound_ref}"),
            synthesize_weapon_shot,
        );
        let mut emitter = AudioEmitter3D::from_xray_params(muzzle_pos, &clip.xray_params);
        emitter.velocity = velocity;
        emitter.min_dist = emitter.min_dist.max(4.0);
        emitter.max_dist = emitter.max_dist.max(120.0);
        emitter.gain = 1.1;

        self.weapon_sounds_triggered += 1;
        self.play_clip_internal(clip, AudioCategory::Weapon, emitter, false)
    }
}

// ---------------------------------------------------------------------------
// Deterministic Procedural Synthesis Fallbacks (used when .ogg missing in test/minimal VFS)
// ---------------------------------------------------------------------------

fn hash_noise(seed: u32, idx: usize) -> f32 {
    let mut x = seed ^ (idx as u32).wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn synthesize_footstep(surface: SurfaceMaterialCategory) -> SoundClip {
    let sr = 24_000u32;
    let len = 3_600usize; // 0.15s crisp transient
    let mut samples = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / sr as f32;
        let env = (-t * 28.0).exp();
        let n = hash_noise(surface as u32 + 101, i);
        let s = match surface {
            SurfaceMaterialCategory::Dirt => (t * 95.0 * TAU).sin() * 0.45 + n * 0.35,
            SurfaceMaterialCategory::Grass => n * 0.45 + (t * 1200.0 * TAU).sin() * 0.12,
            SurfaceMaterialCategory::Gravel => {
                let crunch = ((t * 65.0).sin().abs()).powi(2);
                n * (0.4 + 0.5 * crunch)
            }
            SurfaceMaterialCategory::Wood => (t * 145.0 * TAU).sin() * 0.55 + n * 0.25,
            SurfaceMaterialCategory::Metal => {
                (t * 420.0 * TAU).sin() * 0.4 + (t * 785.0 * TAU).sin() * 0.25 + n * 0.15
            }
            SurfaceMaterialCategory::Water => n * 0.5 * (t * 320.0 * TAU).cos().abs(),
            SurfaceMaterialCategory::Concrete => (t * 180.0 * TAU).sin() * 0.4 + n * 0.35,
        };
        samples.push((s * env * 0.75).clamp(-1.0, 1.0));
    }
    SoundClip {
        name: format!("synth_step_{}", surface.key()),
        sample_rate: sr,
        channels: 1,
        samples,
        xray_params: OggXRayParams {
            min_dist: 0.8,
            max_dist: 18.0,
            base_volume: 1.0,
            ..Default::default()
        },
    }
}

fn synthesize_wind_loop() -> SoundClip {
    let sr = 24_000u32;
    let len = 24_000usize; // 1.0s seamless filtered wind loop
    let mut samples = Vec::with_capacity(len);
    let mut lp = 0.0f32;
    for i in 0..len {
        let t = i as f32 / sr as f32;
        let mod_amp = 0.65 + 0.35 * (t * TAU).sin();
        let raw = hash_noise(777, i);
        lp += 0.03 * (raw - lp);
        samples.push((lp * mod_amp * 0.8).clamp(-1.0, 1.0));
    }
    SoundClip {
        name: "synth_wind_loop".to_string(),
        sample_rate: sr,
        channels: 1,
        samples,
        xray_params: OggXRayParams::default(),
    }
}

fn synthesize_rain_loop() -> SoundClip {
    let sr = 24_000u32;
    let len = 24_000usize; // 1.0s patter noise
    let mut samples = Vec::with_capacity(len);
    let mut prev = 0.0f32;
    for i in 0..len {
        let n = hash_noise(999, i);
        // Gentle high-pass patter texture
        let hp = n - 0.65 * prev;
        prev = n;
        samples.push((hp * 0.32).clamp(-1.0, 1.0));
    }
    SoundClip {
        name: "synth_rain_loop".to_string(),
        sample_rate: sr,
        channels: 1,
        samples,
        xray_params: OggXRayParams::default(),
    }
}

fn synthesize_thunderclap() -> SoundClip {
    let sr = 24_000u32;
    let len = 28_800usize; // 1.2s rolling rumble
    let mut samples = Vec::with_capacity(len);
    let mut lp = 0.0f32;
    for i in 0..len {
        let t = i as f32 / sr as f32;
        let env = (t * 18.0).min(1.0) * (-t * 2.2).exp();
        let crack = hash_noise(31337, i);
        lp += 0.045 * (crack - lp);
        let sub = (t * 48.0 * TAU).sin() * 0.45;
        samples.push(((lp * 0.8 + sub) * env).clamp(-1.0, 1.0));
    }
    SoundClip {
        name: "synth_thunder".to_string(),
        sample_rate: sr,
        channels: 1,
        samples,
        xray_params: OggXRayParams {
            min_dist: 35.0,
            max_dist: 500.0,
            base_volume: 1.0,
            ..Default::default()
        },
    }
}

fn synthesize_weapon_shot() -> SoundClip {
    let sr = 24_000u32;
    let len = 6_000usize; // 0.25s crisp gunshot report
    let mut samples = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / sr as f32;
        let crack_env = (-t * 36.0).exp();
        let tail_env = (-t * 11.0).exp();
        let noise = hash_noise(4242, i);
        let boom = (t * 110.0 * TAU).sin();
        samples.push((noise * 0.7 * crack_env + boom * 0.55 * tail_env).clamp(-1.0, 1.0));
    }
    SoundClip {
        name: "synth_weapon_shot".to_string(),
        sample_rate: sr,
        channels: 1,
        samples,
        xray_params: OggXRayParams {
            min_dist: 4.0,
            max_dist: 120.0,
            base_volume: 1.0,
            ..Default::default()
        },
    }
}

fn synthesize_ambient_hum(name: &str) -> SoundClip {
    let sr = 24_000u32;
    let len = 12_000usize;
    let samples: Vec<f32> = (0..len)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (t * 72.0 * TAU).sin() * 0.25 + hash_noise(name.len() as u32 + 5, i) * 0.1
        })
        .collect();
    SoundClip {
        name: name.to_string(),
        sample_rate: sr,
        channels: 1,
        samples,
        xray_params: OggXRayParams::default(),
    }
}

fn synthesize_generic_sfx(name: &str) -> SoundClip {
    let sr = 24_000u32;
    let len = 4_800usize;
    let samples: Vec<f32> = (0..len)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (t * 260.0 * TAU).sin() * (-t * 14.0).exp() * 0.5
        })
        .collect();
    SoundClip {
        name: name.to_string(),
        sample_rate: sr,
        channels: 1,
        samples,
        xray_params: OggXRayParams::default(),
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_classification_and_footstep_cadence() {
        assert_eq!(classify_material_name("materials\\earth"), SurfaceMaterialCategory::Dirt);
        assert_eq!(classify_material_name("materials\\grass_tall"), SurfaceMaterialCategory::Grass);
        assert_eq!(classify_material_name("materials\\gravel"), SurfaceMaterialCategory::Gravel);
        assert_eq!(classify_material_name("materials\\wood_solid"), SurfaceMaterialCategory::Wood);
        assert_eq!(classify_material_name("materials\\tin"), SurfaceMaterialCategory::Metal);
        assert_eq!(classify_material_name("materials\\water"), SurfaceMaterialCategory::Water);
        assert_eq!(classify_material_name("materials\\concrete"), SurfaceMaterialCategory::Concrete);

        let mut audio = AudioManager::new_headless();
        audio.update_listener(AudioListener::from_yaw_pitch([0.0, 1.65, 0.0], 0.0, 0.0, [0.0; 3]));

        // Simulate 2 seconds of walking at 3.8 m/s on Wood (stride 1.8m => ~4 footsteps)
        let mut gm = GameMaterialLibrary {
            version: 1,
            material_index: 1,
            material_pair_index: 0,
            materials: vec![Default::default()],
            material_pairs: Vec::new(),
        };
        gm.materials[0].name = "materials\\wood_plank".to_string();

        for _ in 0..120 {
            audio.step_footsteps(
                1.0 / 60.0,
                [0.0, 0.0, 0.0],
                3.8,
                true,
                false,
                Some(0),
                Some(&gm),
            );
        }
        let stats = audio.stats();
        assert!(stats.footsteps_triggered >= 3);
        assert_eq!(stats.last_footstep_surface, Some(SurfaceMaterialCategory::Wood));

        // Simulate jump airborne + landing thud
        for _ in 0..15 {
            audio.step_footsteps(1.0 / 60.0, [0.0, 1.0, 0.0], 2.0, false, false, Some(0), Some(&gm));
        }
        audio.step_footsteps(1.0 / 60.0, [0.0, 0.0, 0.0], 2.0, true, false, Some(0), Some(&gm));
        assert_eq!(audio.stats().landings_triggered, 1);
    }

    #[test]
    fn test_weather_audio_rain_scaling_and_3d_thunder_panning() {
        let mut audio = AudioManager::new_headless();
        audio.update_listener(AudioListener::from_yaw_pitch([0.0, 1.65, 0.0], 0.0, 0.0, [0.0; 3]));

        // Dry windy frame -> no rain loop active
        audio.update_weather(5.0, 0.0, 0.0, [0.0, 1.65, 0.0], None);
        let dry_rain = audio.stats().rain_gain;
        assert!(dry_rain <= 0.01);

        // Storm frame with heavy rain + thunderclap to the right (+X = 120m)
        audio.update_weather(
            12.0,
            0.9,
            0.85,
            [0.0, 1.65, 0.0],
            Some(("nature\\thunder-1", [120.0, 60.0, 0.0], 500.0)),
        );
        let stats = audio.stats();
        assert!(stats.wind_gain > 0.3);
        assert!(stats.rain_gain > 0.5);
        assert_eq!(stats.thunder_triggered, 1);

        let mixed = audio.mix_headless_frames(1024);
        let total_energy: f32 = mixed.iter().map(|x| x * x).sum();
        assert!(total_energy > 0.05, "storm audio mix must be non-silent");
    }

    #[test]
    fn test_alife_and_weapon_3d_panning() {
        let mut audio = AudioManager::new_headless();
        audio.update_listener(AudioListener::from_yaw_pitch([0.0, 1.65, 0.0], 0.0, 0.0, [0.0; 3]));

        // Fire weapon to the LEFT (-X = -10m)
        audio.trigger_weapon_sound("weapons\\ak74_shoot", [-10.0, 1.5, 0.0], [0.0; 3]);
        assert_eq!(audio.stats().weapon_sounds_triggered, 1);

        let pcm = audio.mix_headless_frames(1024);
        let l_energy: f32 = pcm.iter().step_by(2).map(|x| x * x).sum();
        let r_energy: f32 = pcm.iter().skip(1).step_by(2).map(|x| x * x).sum();
        assert!(l_energy > r_energy * 8.0, "left gunshot must dominate L channel");
    }
}
