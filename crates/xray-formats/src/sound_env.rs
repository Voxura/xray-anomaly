//! Pure Rust X-Ray Sound Environments, Occlusion & Static Sounds
//!
//! Covers:
//! - `senvironment.xr`: EAX/EFX reverb environment library (`CSoundRender_Environment`, v4 mB conversion & v5 extended + `lerp`)
//! - `level.som`: Sound Occlusion Mesh (44-byte `SOM_poly` + BVH multiplicative occlusion ray query)
//! - `level.snd_env`: Level sound-environment zone geometry (`CFORM` v4 + front/back environment names via `[-Y]` probe)
//! - `level.snd_static`: Ambient level point sounds with day-time schedule (`CSound_Static`)

use crate::cform::{ray_options, CformModel, CformTriangle, RaycastOptions};
use crate::chunk::{ByteCursor, ChunkReader, FormatError};

pub const SDEF_ENV_VERSION: u32 = 4;
pub const SNDENV_VER_EXTENDED: u32 = 5;
pub const SOM_POLY_SIZE: usize = 44;
/// X-Ray `.ogg` Vorbis comment payload version (`Sound.h:23`).
pub const OGG_COMMENT_VERSION: u32 = 0x0003;

#[inline]
fn mb_to_gain(mb: f32) -> f32 {
    10.0f32.powf(mb / 2000.0)
}

/// X-Ray EAX/EFX reverb descriptor (`CSoundRender_Environment`).
#[derive(Debug, Clone, PartialEq)]
pub struct SoundEnvironment {
    pub version: u32,
    pub name: String,
    pub room: f32,
    pub room_hf: f32,
    pub room_rolloff_factor: f32,
    pub decay_time: f32,
    pub decay_hf_ratio: f32,
    pub reflections: f32,
    pub reflections_delay: f32,
    pub reverb: f32,
    pub reverb_delay: f32,
    pub environment_size: f32,
    pub environment_diffusion: f32,
    pub air_absorption_hf: f32,
    pub environment: u32,
    // Extended v5 EFX fields
    pub room_lf: f32,
    pub reflections_pan: f32,
    pub reverb_pan: f32,
    pub decay_hf_limit: u32,
    pub echo_time: f32,
    pub echo_depth: f32,
    pub decay_lf_ratio: f32,
    pub modulation_time: f32,
    pub modulation_depth: f32,
    pub hf_reference: f32,
    pub lf_reference: f32,
    pub density: f32,
}

impl Default for SoundEnvironment {
    fn default() -> Self {
        Self {
            version: SDEF_ENV_VERSION,
            name: String::new(),
            room: 0.0,
            room_hf: 0.891_25,
            room_rolloff_factor: 0.0,
            decay_time: 1.49,
            decay_hf_ratio: 0.83,
            reflections: 0.05,
            reflections_delay: 0.007,
            reverb: 1.258_925,
            reverb_delay: 0.011,
            environment_size: 7.5,
            environment_diffusion: 1.0,
            air_absorption_hf: 0.994_26,
            environment: 0,
            room_lf: 1.0,
            reflections_pan: 0.0,
            reverb_pan: 0.0,
            decay_hf_limit: 1,
            echo_time: 0.25,
            echo_depth: 0.0,
            decay_lf_ratio: 1.0,
            modulation_time: 0.25,
            modulation_depth: 0.0,
            hf_reference: 5000.0,
            lf_reference: 250.0,
            density: 1.0,
        }
    }
}

impl SoundEnvironment {
    /// Parse one `CSoundRender_Environment` chunk payload (`SoundRender_Environment.cpp:105`).
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(bytes);
        let version = cur.read_u32()?;
        let name = cur.read_zstring()?;

        let mut env = Self {
            version,
            name,
            ..Default::default()
        };

        let raw_room = cur.read_f32()?;
        let raw_room_hf = cur.read_f32()?;
        if version <= SDEF_ENV_VERSION {
            env.room = mb_to_gain(raw_room);
            env.room_hf = mb_to_gain(raw_room_hf);
        } else {
            env.room = raw_room;
            env.room_hf = raw_room_hf;
        }

        env.room_rolloff_factor = cur.read_f32()?;
        env.decay_time = cur.read_f32()?;
        env.decay_hf_ratio = cur.read_f32()?;
        env.reflections = cur.read_f32()?;
        env.reflections_delay = cur.read_f32()?;

        let raw_reverb = cur.read_f32()?;
        env.reverb = if version <= SDEF_ENV_VERSION {
            mb_to_gain(raw_reverb)
        } else {
            raw_reverb
        };

        env.reverb_delay = cur.read_f32()?;
        env.environment_size = cur.read_f32()?;
        env.environment_diffusion = cur.read_f32()?;

        let raw_air = cur.read_f32()?;
        env.air_absorption_hf = if version <= SDEF_ENV_VERSION {
            mb_to_gain(raw_air)
        } else {
            raw_air
        };

        env.environment = cur.read_u32()?;

        if version == SNDENV_VER_EXTENDED {
            env.room_lf = cur.read_f32()?;
            env.reflections_pan = cur.read_f32()?;
            env.reverb_pan = cur.read_f32()?;
            env.decay_hf_limit = cur.read_u32()?;
            env.echo_time = cur.read_f32()?;
            env.echo_depth = cur.read_f32()?;
            env.reverb_delay = cur.read_f32()?;
            env.decay_lf_ratio = cur.read_f32()?;
            env.modulation_time = cur.read_f32()?;
            env.modulation_depth = cur.read_f32()?;
            env.hf_reference = cur.read_f32()?;
            env.lf_reference = cur.read_f32()?;
            env.density = cur.read_f32()?;
        }

        Ok(env)
    }

    /// Blend two environments with clamped OpenAL EFX bounds (`CSoundRender_Environment::lerp`).
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let f = t.clamp(0.0, 1.0);
        let fi = 1.0 - f;
        Self {
            version: self.version.max(other.version),
            name: self.name.clone(),
            room: (fi * self.room + f * other.room).clamp(0.0, 1.0),
            room_hf: (fi * self.room_hf + f * other.room_hf).clamp(0.0, 1.0),
            room_rolloff_factor: (fi * self.room_rolloff_factor + f * other.room_rolloff_factor)
                .clamp(0.0, 10.0),
            decay_time: (fi * self.decay_time + f * other.decay_time).clamp(0.1, 20.0),
            decay_hf_ratio: (fi * self.decay_hf_ratio + f * other.decay_hf_ratio).clamp(0.1, 2.0),
            reflections: (fi * self.reflections + f * other.reflections).clamp(0.0, 3.16),
            reflections_delay: (fi * self.reflections_delay + f * other.reflections_delay)
                .clamp(0.0, 0.3),
            reverb: (fi * self.reverb + f * other.reverb).clamp(0.0, 10.0),
            reverb_delay: (fi * self.reverb_delay + f * other.reverb_delay).clamp(0.0, 0.1),
            environment_size: fi * self.environment_size + f * other.environment_size,
            environment_diffusion: (fi * self.environment_diffusion
                + f * other.environment_diffusion)
                .clamp(0.0, 1.0),
            air_absorption_hf: (fi * self.air_absorption_hf + f * other.air_absorption_hf)
                .clamp(0.892, 1.0),
            environment: if f >= 0.5 {
                other.environment
            } else {
                self.environment
            },
            room_lf: (fi * self.room_lf + f * other.room_lf).clamp(0.0, 1.0),
            reflections_pan: fi * self.reflections_pan + f * other.reflections_pan,
            reverb_pan: fi * self.reverb_pan + f * other.reverb_pan,
            decay_hf_limit: if f >= 0.5 {
                other.decay_hf_limit
            } else {
                self.decay_hf_limit
            },
            echo_time: (fi * self.echo_time + f * other.echo_time).clamp(0.075, 0.25),
            echo_depth: (fi * self.echo_depth + f * other.echo_depth).clamp(0.0, 1.0),
            decay_lf_ratio: (fi * self.decay_lf_ratio + f * other.decay_lf_ratio).clamp(0.1, 2.0),
            modulation_time: (fi * self.modulation_time + f * other.modulation_time)
                .clamp(0.04, 4.0),
            modulation_depth: (fi * self.modulation_depth + f * other.modulation_depth)
                .clamp(0.0, 1.0),
            hf_reference: (fi * self.hf_reference + f * other.hf_reference).clamp(1000.0, 20000.0),
            lf_reference: (fi * self.lf_reference + f * other.lf_reference).clamp(20.0, 1000.0),
            density: (fi * self.density + f * other.density).clamp(0.0, 1.0),
        }
    }
}

/// Parsed `senvironment.xr` library (`SoundEnvironment_LIB`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SoundEnvironmentLibrary {
    pub environments: Vec<SoundEnvironment>,
}

impl SoundEnvironmentLibrary {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let mut environments = Vec::new();
        for chunk in reader.iter() {
            let chunk = chunk?;
            let payload = chunk.decode()?;
            environments.push(SoundEnvironment::parse(&payload)?);
        }
        Ok(Self { environments })
    }

    pub fn get(&self, id: usize) -> Option<&SoundEnvironment> {
        self.environments.get(id)
    }

    pub fn find_by_name(&self, name: &str) -> Option<&SoundEnvironment> {
        self.environments
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(name))
    }
}

/// Single `level.som` acoustic occlusion polygon (`SOM_poly` = 44B).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SomPolygon {
    pub verts: [[f32; 3]; 3],
    pub two_sided: bool,
    /// Transmission factor `[0..1]` (`0.0` blocks sound completely, `1.0` unattenuated).
    pub occ: f32,
}

/// Parsed `level.som` sound occlusion mesh with BVH ray transmission evaluation.
#[derive(Debug, Clone)]
pub struct SoundOcclusionMesh {
    pub version: u32,
    pub polygons: Vec<SomPolygon>,
    cform: CformModel,
    tri_transmission: Vec<f32>,
}

impl SoundOcclusionMesh {
    /// Parse binary `level.som` (`CSoundRender_Core::i_load_som`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let ver_chunk = reader.require_chunk(0)?;
        let mut vcur = ByteCursor::new(&ver_chunk);
        let version = vcur.read_u32()?;
        if version != 0 {
            return Err(FormatError::InvalidValue(
                "unsupported level.som version (expected 0)",
            ));
        }

        let poly_chunk = reader.require_chunk(1)?;
        if !poly_chunk.len().is_multiple_of(SOM_POLY_SIZE) {
            return Err(FormatError::InvalidValue(
                "level.som chunk 1 size is not a multiple of 44 bytes",
            ));
        }
        let count = poly_chunk.len() / SOM_POLY_SIZE;
        let mut polygons = Vec::with_capacity(count);
        let mut cur = ByteCursor::new(&poly_chunk);
        for _ in 0..count {
            let v0 = cur.read_f32_3()?;
            let v1 = cur.read_f32_3()?;
            let v2 = cur.read_f32_3()?;
            let b2sided = cur.read_u32()? != 0;
            let occ = cur.read_f32()?;
            polygons.push(SomPolygon {
                verts: [v0, v1, v2],
                two_sided: b2sided,
                occ,
            });
        }

        Self::from_polygons(version, polygons)
    }

    pub fn from_polygons(version: u32, polygons: Vec<SomPolygon>) -> Result<Self, FormatError> {
        let mut vertices = Vec::with_capacity(polygons.len() * 3);
        let mut triangles = Vec::with_capacity(polygons.len() * 2);
        let mut tri_transmission = Vec::with_capacity(polygons.len() * 2);

        for poly in &polygons {
            let base = vertices.len() as u32;
            vertices.extend_from_slice(&poly.verts);
            triangles.push(CformTriangle::from_raw([base, base + 1, base + 2], 0));
            tri_transmission.push(poly.occ);
            if poly.two_sided {
                // Reverse winding matches CSoundRender_Core::i_load_som two-sided insertion.
                triangles.push(CformTriangle::from_raw([base + 2, base + 1, base], 0));
                tri_transmission.push(poly.occ);
            }
        }

        let cform = CformModel::from_parts(vertices, triangles)?;
        Ok(Self {
            version,
            polygons,
            cform,
            tri_transmission,
        })
    }

    /// Compute multiplicative acoustic occlusion `[0..1]` between `listener_pos` and `source_pos`
    /// (`CSoundRender_Core::get_occlusion`). Returns `1.0` when completely unobstructed.
    pub fn query_occlusion(&self, listener_pos: [f32; 3], source_pos: [f32; 3]) -> f32 {
        let dx = source_pos[0] - listener_pos[0];
        let dy = source_pos[1] - listener_pos[1];
        let dz = source_pos[2] - listener_pos[2];
        let range = (dx * dx + dy * dy + dz * dz).sqrt();
        if range <= 1e-6 || self.polygons.is_empty() {
            return 1.0;
        }
        let dir = [dx / range, dy / range, dz / range];
        let bvh = self.cform.build_bvh();
        let hits = bvh.raycast(
            listener_pos,
            dir,
            range,
            RaycastOptions::from_flags(ray_options::OPT_CULL),
        );

        let mut occ_value = 1.0f32;
        for hit in hits {
            if let Some(&factor) = self.tri_transmission.get(hit.tri_index) {
                occ_value *= factor;
            }
        }
        occ_value
    }
}

/// Parsed `level.snd_env` spatial reverb-zone geometry (`CSoundRender_Core::i_create_sdef_geometry`).
#[derive(Debug, Clone)]
pub struct LevelSoundEnv {
    pub env_names: Vec<String>,
    cform: CformModel,
    /// Full 16-bit `(front_id, back_id)` indices preserved from raw `CDB::TRI::dummy`.
    pub env_pairs: Vec<(u16, u16)>,
}

impl LevelSoundEnv {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let names_chunk = reader.require_chunk(0)?;
        let cform_chunk = reader.require_chunk(1)?;

        let mut env_names = Vec::new();
        let mut cur = ByteCursor::new(&names_chunk);
        while !cur.is_eof() {
            env_names.push(cur.read_zstring()?);
        }

        let cform = CformModel::parse(&cform_chunk)?;
        // Extract full 16-bit front_id (`dummy & 0xFFFF`) without losing CFORM bits 14..15.
        let mut fc = ByteCursor::new(&cform_chunk);
        let _ver = fc.read_u32()?;
        let vc = fc.read_u32()? as usize;
        let tc = fc.read_u32()? as usize;
        let _ = fc.read_bytes(24)?;
        let _ = fc.read_bytes(vc * 12)?;
        let mut env_pairs = Vec::with_capacity(tc);
        for _ in 0..tc {
            let _ = fc.read_bytes(12)?;
            let dummy = fc.read_u32()?;
            env_pairs.push(((dummy & 0xFFFF) as u16, (dummy >> 16) as u16));
        }

        Ok(Self {
            env_names,
            cform,
            env_pairs,
        })
    }

    /// Probe directly downward `-Y` (`1000m`) at `pos` to determine active sound environment name.
    pub fn query_env_name_at(&self, pos: [f32; 3]) -> Option<&str> {
        let bvh = self.cform.build_bvh();
        let dir = [0.0, -1.0, 0.0];
        let hit = bvh.raycast_nearest(pos, dir, 1000.0, false)?;
        let (front_id, back_id) = *self.env_pairs.get(hit.tri_index)?;

        let [v0, v1, v2] = hit.verts;
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        // Normal Y-component against downward ray (dot with [0, -1, 0] == -n_y < 0 <=> n_y > 0)
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let env_idx = if (-ny) < 0.0 {
            front_id as usize
        } else {
            back_id as usize
        };
        self.env_names.get(env_idx).map(String::as_str)
    }
}

/// Static ambient level sound emitter (`level.snd_static`).
#[derive(Debug, Clone, PartialEq)]
pub struct StaticSound {
    pub wav_name: String,
    pub position: [f32; 3],
    pub volume: f32,
    pub frequency: f32,
    /// Active time-of-day window `(start_ms, end_ms)` (`[0, 0]` = always active).
    pub active_time: [u32; 2],
    pub play_time: [u32; 2],
    pub pause_time: [u32; 2],
}

impl StaticSound {
    #[inline]
    pub fn is_active_at(&self, level_time_ms: u32) -> bool {
        let [start, end] = self.active_time;
        if start == 0 && end == 0 {
            return true;
        }
        if start <= end {
            level_time_ms >= start && level_time_ms <= end
        } else {
            level_time_ms >= start || level_time_ms <= end
        }
    }
}

/// Parsed `level.snd_static` collection (`CLevelSoundManager::Load`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LevelStaticSounds {
    pub sounds: Vec<StaticSound>,
}

impl LevelStaticSounds {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let mut sounds = Vec::new();
        for chunk in reader.iter() {
            let chunk = chunk?;
            let payload = chunk.decode()?;
            let sub = ChunkReader::new(&payload);
            let obj = sub.require_chunk(0)?;
            let mut cur = ByteCursor::new(&obj);
            let wav_name = cur.read_zstring()?;
            let position = cur.read_f32_3()?;
            let volume = cur.read_f32()?;
            let frequency = cur.read_f32()?;
            let active_time = [cur.read_u32()?, cur.read_u32()?];
            let play_time = [cur.read_u32()?, cur.read_u32()?];
            let pause_time = [cur.read_u32()?, cur.read_u32()?];
            sounds.push(StaticSound {
                wav_name,
                position,
                volume,
                frequency,
                active_time,
                play_time,
                pause_time,
            });
        }
        Ok(Self { sounds })
    }
}

/// X-Ray spatial & gameplay sound metadata stored inside `.ogg` Vorbis comment 0
/// (`CSoundRender_Source::LoadWave`, `src/xrSound/SoundRender_Source_loader.cpp:108-153`).
#[derive(Debug, Clone, PartialEq)]
pub struct OggXRayParams {
    pub version: u32,
    pub min_dist: f32,
    pub max_dist: f32,
    pub base_volume: f32,
    pub game_type: u32,
    pub max_ai_dist: f32,
}

impl Default for OggXRayParams {
    fn default() -> Self {
        Self {
            version: OGG_COMMENT_VERSION,
            min_dist: 1.0,
            max_dist: 30.0,
            base_volume: 1.0,
            game_type: 0,
            max_ai_dist: 30.0,
        }
    }
}

impl OggXRayParams {
    /// Parse binary comment 0 payload (`0x0001`, `0x0002`, or `0x0003` = `OGG_COMMENT_VERSION`).
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(bytes);
        let version = cur.read_u32()?;
        match version {
            0x0001 => {
                let min_dist = cur.read_f32()?;
                let max_dist = cur.read_f32()?;
                let game_type = cur.read_u32()?;
                Ok(Self {
                    version,
                    min_dist,
                    max_dist,
                    base_volume: 1.0,
                    game_type,
                    max_ai_dist: max_dist,
                })
            }
            0x0002 => {
                let min_dist = cur.read_f32()?;
                let max_dist = cur.read_f32()?;
                let base_volume = cur.read_f32()?;
                let game_type = cur.read_u32()?;
                Ok(Self {
                    version,
                    min_dist,
                    max_dist,
                    base_volume,
                    game_type,
                    max_ai_dist: max_dist,
                })
            }
            OGG_COMMENT_VERSION => {
                let min_dist = cur.read_f32()?;
                let max_dist = cur.read_f32()?;
                let base_volume = cur.read_f32()?;
                let game_type = cur.read_u32()?;
                let max_ai_dist = cur.read_f32()?;
                Ok(Self {
                    version,
                    min_dist,
                    max_dist,
                    base_volume,
                    game_type,
                    max_ai_dist,
                })
            }
            _ => Err(FormatError::InvalidValue(
                "unsupported X-Ray .ogg comment version (expected 1, 2, or 3)",
            )),
        }
    }

    /// Serialize in `OGG_COMMENT_VERSION` (`0x0003`) 24-byte layout.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24);
        out.extend_from_slice(&OGG_COMMENT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.min_dist.to_le_bytes());
        out.extend_from_slice(&self.max_dist.to_le_bytes());
        out.extend_from_slice(&self.base_volume.to_le_bytes());
        out.extend_from_slice(&self.game_type.to_le_bytes());
        out.extend_from_slice(&self.max_ai_dist.to_le_bytes());
        out
    }
}

/// Pure-Rust Ogg Vorbis container + X-Ray sound source metadata (`CSoundRender_Source`).
#[derive(Debug, Clone, PartialEq)]
pub struct OggSoundMetadata {
    pub channels: u8,
    pub sample_rate: u32,
    pub bitrate_maximum: i32,
    pub bitrate_nominal: i32,
    pub bitrate_minimum: i32,
    pub total_samples: Option<u64>,
    pub duration_seconds: Option<f32>,
    pub pcm_16bit_bytes_total: Option<u64>,
    pub vendor: String,
    pub xray_params: Option<OggXRayParams>,
    pub raw_comments: Vec<Vec<u8>>,
    pub tags: Vec<(String, String)>,
}

impl OggSoundMetadata {
    /// Parse an Ogg Vorbis stream, decoding Vorbis identification (`0x01vorbis`),
    /// Vorbis comment (`0x03vorbis`) with X-Ray `OGG_COMMENT_VERSION` metadata,
    /// and page-level `granule_position` duration.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut pos = 0usize;
        let mut serial_filter: Option<u32> = None;
        let mut last_granule: Option<u64> = None;
        let mut packet_buf = Vec::new();
        let mut packets: Vec<Vec<u8>> = Vec::new();
        let mut pages_seen = 0usize;

        while pos < data.len() {
            if pos + 27 > data.len() {
                return Err(FormatError::UnexpectedEof);
            }
            if &data[pos..pos + 4] != b"OggS" {
                return Err(FormatError::InvalidValue("expected OggS page magic"));
            }
            let stream_version = data[pos + 4];
            if stream_version != 0 {
                return Err(FormatError::InvalidValue("unsupported Ogg stream version"));
            }
            let header_type = data[pos + 5];
            let granule = u64::from_le_bytes(data[pos + 6..pos + 14].try_into().unwrap());
            let serial = u32::from_le_bytes(data[pos + 14..pos + 18].try_into().unwrap());
            let page_segments = data[pos + 26] as usize;
            let seg_table_start = pos + 27;
            let seg_table_end = seg_table_start + page_segments;
            if seg_table_end > data.len() {
                return Err(FormatError::UnexpectedEof);
            }
            let seg_table = &data[seg_table_start..seg_table_end];
            let payload_len: usize = seg_table.iter().map(|&b| usize::from(b)).sum();
            let payload_start = seg_table_end;
            let payload_end = payload_start + payload_len;
            if payload_end > data.len() {
                return Err(FormatError::UnexpectedEof);
            }

            if serial_filter.is_none() {
                serial_filter = Some(serial);
            }
            pages_seen += 1;

            if Some(serial) == serial_filter {
                if (granule as i64) >= 0 {
                    last_granule = Some(granule);
                }

                // Reassemble packets only until both Vorbis ID & Comment headers are extracted.
                if packets.len() < 2 {
                    if (header_type & 0x01) == 0 && !packet_buf.is_empty() {
                        packet_buf.clear();
                    }
                    let mut pcur = payload_start;
                    for &seg_len_u8 in seg_table {
                        let seg_len = usize::from(seg_len_u8);
                        packet_buf.extend_from_slice(&data[pcur..pcur + seg_len]);
                        pcur += seg_len;
                        if seg_len < 255 {
                            packets.push(core::mem::take(&mut packet_buf));
                        }
                    }
                }
            }

            pos = payload_end;
        }

        if pages_seen == 0 || packets.is_empty() {
            return Err(FormatError::InvalidValue("empty or truncated Ogg stream"));
        }

        let id_pkt = &packets[0];
        if id_pkt.len() < 30 || id_pkt[0] != 0x01 || &id_pkt[1..7] != b"vorbis" {
            return Err(FormatError::InvalidValue("expected Vorbis identification header"));
        }
        let mut id_cur = ByteCursor::new(&id_pkt[7..]);
        let vorbis_version = id_cur.read_u32()?;
        if vorbis_version != 0 {
            return Err(FormatError::InvalidValue("unsupported Vorbis codec version"));
        }
        let channels = id_cur.read_u8()?;
        let sample_rate = id_cur.read_u32()?;
        if channels == 0 || sample_rate == 0 {
            return Err(FormatError::InvalidValue("invalid Vorbis channel/sample rate"));
        }
        let bitrate_maximum = id_cur.read_i32()?;
        let bitrate_nominal = id_cur.read_i32()?;
        let bitrate_minimum = id_cur.read_i32()?;
        let _blocksize = id_cur.read_u8()?;
        let framing = id_cur.read_u8()?;
        if (framing & 1) == 0 {
            return Err(FormatError::InvalidValue("Vorbis identification framing bit is zero"));
        }

        let mut vendor = String::new();
        let mut raw_comments = Vec::new();
        let mut tags = Vec::new();
        let mut xray_params = None;

        if let Some(cmt_pkt) = packets.get(1)
            && cmt_pkt.len() >= 15
            && cmt_pkt[0] == 0x03
            && &cmt_pkt[1..7] == b"vorbis"
        {
            let mut ccur = ByteCursor::new(&cmt_pkt[7..]);
            let vlen = ccur.read_u32()? as usize;
            let vbytes = ccur.read_bytes(vlen)?;
            vendor = String::from_utf8_lossy(vbytes).into_owned();
            let count = ccur.read_u32()? as usize;
            raw_comments.reserve(count.min(256));
            for idx in 0..count {
                let clen = ccur.read_u32()? as usize;
                let cbytes = ccur.read_bytes(clen)?.to_vec();
                if idx == 0 && cbytes.len() >= 16 {
                    let vers = u32::from_le_bytes([cbytes[0], cbytes[1], cbytes[2], cbytes[3]]);
                    if matches!(vers, 0x0001 | 0x0002 | OGG_COMMENT_VERSION)
                        && let Ok(parsed) = OggXRayParams::parse(&cbytes)
                    {
                        xray_params = Some(parsed);
                    }
                }
                if let Ok(text) = core::str::from_utf8(&cbytes)
                    && let Some((k, v)) = text.split_once('=')
                    && !k.is_empty()
                    && !k.contains('\0')
                {
                    tags.push((k.to_string(), v.to_string()));
                }
                raw_comments.push(cbytes);
            }
        }

        let duration_seconds = last_granule.map(|samples| (samples as f32) / (sample_rate as f32));
        let pcm_16bit_bytes_total =
            last_granule.map(|samples| samples.saturating_mul(u64::from(channels) * 2));

        Ok(Self {
            channels,
            sample_rate,
            bitrate_maximum,
            bitrate_nominal,
            bitrate_minimum,
            total_samples: last_granule,
            duration_seconds,
            pcm_16bit_bytes_total,
            vendor,
            xray_params,
            raw_comments,
            tags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;

    #[test]
    fn parse_senvironment_v4_and_v5_and_lerp() {
        let mut env4 = Vec::new();
        env4.extend_from_slice(&SDEF_ENV_VERSION.to_le_bytes());
        env4.extend_from_slice(b"cave\0");
        env4.extend_from_slice(&(-1000.0f32).to_le_bytes()); // room mB -> 10^(-0.5)
        env4.extend_from_slice(&0.0f32.to_le_bytes()); // room_hf mB -> 1.0
        for f in [1.0f32, 2.0, 0.5, 0.1, 0.02] {
            env4.extend_from_slice(&f.to_le_bytes());
        }
        env4.extend_from_slice(&0.0f32.to_le_bytes()); // reverb mB -> 1.0
        for f in [0.05f32, 12.0, 0.8] {
            env4.extend_from_slice(&f.to_le_bytes());
        }
        env4.extend_from_slice(&0.0f32.to_le_bytes()); // air_absorption_hf -> 1.0
        env4.extend_from_slice(&4u32.to_le_bytes()); // environment id

        let mut env5 = Vec::new();
        env5.extend_from_slice(&SNDENV_VER_EXTENDED.to_le_bytes());
        env5.extend_from_slice(b"tunnel\0");
        for f in [
            0.5f32, 0.5, 0.5, 4.0, 0.9, 0.2, 0.01, 0.8, 0.03, 5.0, 0.6, 0.95,
        ] {
            env5.extend_from_slice(&f.to_le_bytes());
        }
        env5.extend_from_slice(&9u32.to_le_bytes());
        // 13 extended v5 fields
        for f in [0.4f32, 0.1, 0.2] {
            env5.extend_from_slice(&f.to_le_bytes());
        }
        env5.extend_from_slice(&1u32.to_le_bytes());
        for f in [0.15f32, 0.25, 0.04, 1.2, 0.5, 0.3, 6000.0, 200.0, 0.9] {
            env5.extend_from_slice(&f.to_le_bytes());
        }

        let mut cw = ChunkWriter::new();
        cw.write_chunk(0, &env4);
        cw.write_chunk(1, &env5);
        let lib = SoundEnvironmentLibrary::parse(&cw.into_bytes()).unwrap();

        assert_eq!(lib.environments.len(), 2);
        let cave = lib.find_by_name("cave").unwrap();
        assert!((cave.room - mb_to_gain(-1000.0)).abs() < 1e-4);
        let tunnel = lib.find_by_name("tunnel").unwrap();
        assert!((tunnel.reverb_delay - 0.04).abs() < 1e-6);
        assert!((tunnel.density - 0.9).abs() < 1e-6);

        let blended = cave.lerp(tunnel, 0.5);
        assert!((blended.decay_time - 3.0).abs() < 1e-5);
    }

    #[test]
    fn parse_level_som_and_query_occlusion() {
        let mut poly_bytes = Vec::new();
        // Wall at z=5 blocking listener (0,0,0) -> source (0,0,10) with two-sided 0.25 transmission
        let tri = [[-5.0f32, -5.0, 5.0], [-5.0, 5.0, 5.0], [5.0, 0.0, 5.0]];
        for p in tri {
            for f in p {
                poly_bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        poly_bytes.extend_from_slice(&1u32.to_le_bytes()); // b2sided = true
        poly_bytes.extend_from_slice(&0.25f32.to_le_bytes()); // occ = 0.25

        let mut cw = ChunkWriter::new();
        cw.write_chunk(0, &0u32.to_le_bytes());
        cw.write_chunk(1, &poly_bytes);

        let som = SoundOcclusionMesh::parse(&cw.into_bytes()).unwrap();
        assert!((som.query_occlusion([0.0, 0.0, 0.0], [0.0, 0.0, 10.0]) - 0.25).abs() < 1e-5);
        assert!((som.query_occlusion([0.0, 0.0, 10.0], [0.0, 0.0, 0.0]) - 0.25).abs() < 1e-5);
        assert!((som.query_occlusion([0.0, 0.0, 0.0], [0.0, 0.0, 3.0]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn parse_level_snd_env_and_static_sounds() {
        // Chunk 0: packed names; Chunk 1: CFORM v4 quad at y=0 facing +Y
        let mut names = Vec::new();
        names.extend_from_slice(b"outdoor\0indoor\0");

        let mut cform = Vec::new();
        cform.extend_from_slice(&4u32.to_le_bytes());
        cform.extend_from_slice(&3u32.to_le_bytes());
        cform.extend_from_slice(&1u32.to_le_bytes());
        for f in [-5.0f32, 0.0, -5.0, 5.0, 0.0, 5.0] {
            cform.extend_from_slice(&f.to_le_bytes());
        }
        for p in [[-5.0f32, 0.0, -5.0], [-5.0, 0.0, 5.0], [5.0, 0.0, -5.0]] {
            for f in p {
                cform.extend_from_slice(&f.to_le_bytes());
            }
        }
        // triangle indices (0,1,2) in right-handed/XRay CW -> n=(v1-v0)x(v2-v0)
        // v1-v0 = (0,0,10), v2-v0 = (10,0,0) => cross = (0, 100, 0) => +Y => ny > 0 => -ny < 0 => front_id
        cform.extend_from_slice(&0u32.to_le_bytes());
        cform.extend_from_slice(&1u32.to_le_bytes());
        cform.extend_from_slice(&2u32.to_le_bytes());
        // front_id = 1 ("indoor"), back_id = 0 ("outdoor")
        let dummy: u32 = 1u32;
        cform.extend_from_slice(&dummy.to_le_bytes());

        let mut cw = ChunkWriter::new();
        cw.write_chunk(0, &names);
        cw.write_chunk(1, &cform);
        let snd_env = LevelSoundEnv::parse(&cw.into_bytes()).unwrap();
        assert_eq!(snd_env.query_env_name_at([0.0, 2.0, 0.0]), Some("indoor"));

        // Static sound chunk
        let mut snd_obj = Vec::new();
        snd_obj.extend_from_slice(b"ambient\\wind_1\0");
        for f in [10.0f32, 2.0, -5.0, 0.85, 1.0] {
            snd_obj.extend_from_slice(&f.to_le_bytes());
        }
        for u in [6000u32, 18000, 1000, 3000, 500, 1500] {
            snd_obj.extend_from_slice(&u.to_le_bytes());
        }
        let mut sub_cw = ChunkWriter::new();
        sub_cw.write_chunk(0, &snd_obj);
        let mut root_cw = ChunkWriter::new();
        root_cw.write_chunk(0, &sub_cw.into_bytes());

        let static_sounds = LevelStaticSounds::parse(&root_cw.into_bytes()).unwrap();
        assert_eq!(static_sounds.sounds.len(), 1);
        assert_eq!(static_sounds.sounds[0].wav_name, "ambient\\wind_1");
        assert!(static_sounds.sounds[0].is_active_at(12000));
        assert!(!static_sounds.sounds[0].is_active_at(20000));
    }

    fn build_ogg_page(serial: u32, seq: u32, header_type: u8, granule: u64, segs: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"OggS");
        out.push(0); // stream structure version
        out.push(header_type);
        out.extend_from_slice(&granule.to_le_bytes());
        out.extend_from_slice(&serial.to_le_bytes());
        out.extend_from_slice(&seq.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // CRC (not verified by parser)
        out.push(segs.len() as u8);
        for s in segs {
            out.push(s.len() as u8);
        }
        for s in segs {
            out.extend_from_slice(s);
        }
        out
    }

    #[test]
    fn parse_ogg_vorbis_and_xray_comment_versions() {
        // Test OggXRayParams v1 and v2 backward-compat defaults
        let mut v1 = Vec::new();
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&2.0f32.to_le_bytes());
        v1.extend_from_slice(&45.0f32.to_le_bytes());
        v1.extend_from_slice(&7u32.to_le_bytes());
        let p1 = OggXRayParams::parse(&v1).unwrap();
        assert_eq!(p1.base_volume, 1.0);
        assert_eq!(p1.max_ai_dist, 45.0);

        let mut v2 = Vec::new();
        v2.extend_from_slice(&2u32.to_le_bytes());
        v2.extend_from_slice(&1.5f32.to_le_bytes());
        v2.extend_from_slice(&60.0f32.to_le_bytes());
        v2.extend_from_slice(&0.8f32.to_le_bytes());
        v2.extend_from_slice(&12u32.to_le_bytes());
        let p2 = OggXRayParams::parse(&v2).unwrap();
        assert_eq!(p2.base_volume, 0.8);
        assert_eq!(p2.max_ai_dist, 60.0);

        // Build full Ogg stream with Vorbis ID + multi-page Comment + Audio EOS page
        let mut id_pkt = Vec::new();
        id_pkt.push(0x01);
        id_pkt.extend_from_slice(b"vorbis");
        id_pkt.extend_from_slice(&0u32.to_le_bytes()); // version
        id_pkt.push(2); // stereo
        id_pkt.extend_from_slice(&44_100u32.to_le_bytes());
        id_pkt.extend_from_slice(&0i32.to_le_bytes());
        id_pkt.extend_from_slice(&128_000i32.to_le_bytes());
        id_pkt.extend_from_slice(&0i32.to_le_bytes());
        id_pkt.push(0xB8); // blocksize 0/1
        id_pkt.push(0x01); // framing bit

        let xray_c = OggXRayParams {
            version: OGG_COMMENT_VERSION,
            min_dist: 3.0,
            max_dist: 55.0,
            base_volume: 0.9,
            game_type: 0x0100_0001,
            max_ai_dist: 35.0,
        };
        let mut cmt_pkt = Vec::new();
        cmt_pkt.push(0x03);
        cmt_pkt.extend_from_slice(b"vorbis");
        let vendor = b"X-Ray Sound Encoder";
        cmt_pkt.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        cmt_pkt.extend_from_slice(vendor);
        cmt_pkt.extend_from_slice(&2u32.to_le_bytes());
        let c0 = xray_c.to_bytes();
        cmt_pkt.extend_from_slice(&(c0.len() as u32).to_le_bytes());
        cmt_pkt.extend_from_slice(&c0);
        let c1 = b"TITLE=Campfire Guitar";
        cmt_pkt.extend_from_slice(&(c1.len() as u32).to_le_bytes());
        cmt_pkt.extend_from_slice(c1);
        cmt_pkt.push(0x01); // framing bit

        // Split comment packet across two Ogg pages if desired (test 255-byte continuation)
        let mut pad_cmt = cmt_pkt;
        pad_cmt.resize(260, 0);
        let serial = 0x1234_5678;
        let mut ogg = Vec::new();
        ogg.extend_from_slice(&build_ogg_page(serial, 0, 0x02, 0, &[&id_pkt]));
        ogg.extend_from_slice(&build_ogg_page(serial, 1, 0x00, 0, &[&pad_cmt[..255]]));
        ogg.extend_from_slice(&build_ogg_page(serial, 2, 0x01, 0, &[&pad_cmt[255..]]));
        ogg.extend_from_slice(&build_ogg_page(serial, 3, 0x04, 88_200, &[b"\x05dummy"]));

        let meta = OggSoundMetadata::parse(&ogg).unwrap();
        assert_eq!(meta.channels, 2);
        assert_eq!(meta.sample_rate, 44_100);
        assert_eq!(meta.bitrate_nominal, 128_000);
        assert_eq!(meta.total_samples, Some(88_200));
        assert!((meta.duration_seconds.unwrap() - 2.0).abs() < 1e-6);
        assert_eq!(meta.pcm_16bit_bytes_total, Some(88_200 * 4));
        assert_eq!(meta.vendor, "X-Ray Sound Encoder");
        assert_eq!(meta.xray_params, Some(xray_c));
        assert_eq!(meta.tags, vec![("TITLE".to_string(), "Campfire Guitar".to_string())]);
    }
}
