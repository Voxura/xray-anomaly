//! Pure-Rust 3D Spatial Audio Engine for X-Ray Monolith (`xray-audio`)
//!
//! Provides:
//! - Pure-Rust `.ogg` Vorbis I container demuxing & decoding streaming from `xray-vfs` (`ogg_vorbis`)
//! - X-Ray `xr_sound` comment chunk metadata reader (`OggXRayParams`: `min_dist`, `max_dist`, `base_volume`, `game_type`)
//! - 3D listener transform, distance attenuation curves, stereo panning, and `level.som` occlusion (`spatial`)
//! - Multi-voice resampling software spatial mixer (`mixer`)
//! - Windows WASAPI shared-mode hardware audio output with headless fallback (`backend`)
//! - High-level X-Ray gameplay audio coordinator: material footsteps, weather wind/rain/thunder, ALife & weapons (`manager`)

pub mod backend;
pub mod manager;
pub mod mixer;
pub mod ogg_vorbis;
pub mod spatial;

pub use backend::{AudioOutputBackend, WasapiAudioOutput};
pub use manager::{
    classify_material_name, AudioEngineStats, AudioManager, AudioManagerConfig,
    SurfaceMaterialCategory,
};
pub use mixer::{AudioCategory, SoundClip, SpatialAudioMixer, VoiceId, DEFAULT_MIXER_SAMPLE_RATE};
pub use ogg_vorbis::{
    decode_ogg_bytes, decode_ogg_from_vfs, open_ogg_stream, resolve_sound_vpath, DecodedOggAudio,
    OggVorbisError, VorbisStreamDecoder,
};
pub use spatial::{
    compute_spatial_gains, AttenuationModel, AudioEmitter3D, AudioListener, SpatialMixGains,
};
