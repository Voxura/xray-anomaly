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
}
