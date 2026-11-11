//! Pure Rust X-Ray `gamemtl.xr` Material & Material-Pair Library Parser
//!
//! Faithful pure-Rust implementation matching `src/xrEngine/GameMtlLib.h`,
//! `src/xrEngine/GameMtlLib.cpp`, and `src/xrEngine/GameMtlLib_Engine.cpp`
//! (including Anomaly's `materials/materials.ltx` & `materials/material_pairs.ltx` overlays).

use xray_core::ltx_loader::OwnedLtxDocument;

use crate::chunk::{ByteCursor, ChunkReader, FormatError};

pub const GAMEMTL_CURRENT_VERSION: u16 = 0x0001;
pub const GAMEMTL_NONE_ID: u32 = u32::MAX;

/// Top-level `gamemtl.xr` chunk IDs (`GameMtlLib.h:10-13`).
pub mod gamemtl_root_chunks {
    pub const GAMEMTLS_CHUNK_VERSION: u32 = 0x1000;
    pub const GAMEMTLS_CHUNK_AUTOINC: u32 = 0x1001;
    pub const GAMEMTLS_CHUNK_MTLS: u32 = 0x1002;
    pub const GAMEMTLS_CHUNK_MTLS_PAIR: u32 = 0x1003;
}

/// Per-material child chunk IDs (`GameMtlLib.h:15-23`).
pub mod gamemtl_chunks {
    pub const GAMEMTL_CHUNK_MAIN: u32 = 0x1000;
    pub const GAMEMTL_CHUNK_FLAGS: u32 = 0x1001;
    pub const GAMEMTL_CHUNK_PHYSICS: u32 = 0x1002;
    pub const GAMEMTL_CHUNK_FACTORS: u32 = 0x1003;
    pub const GAMEMTL_CHUNK_FLOTATION: u32 = 0x1004;
    pub const GAMEMTL_CHUNK_DESC: u32 = 0x1005;
    pub const GAMEMTL_CHUNK_INJURIOUS: u32 = 0x1006;
    pub const GAMEMTL_CHUNK_DENSITY: u32 = 0x1007;
    pub const GAMEMTL_CHUNK_FACTORS_MP: u32 = 0x1008;
}

/// Per-material-pair child chunk IDs (`GameMtlLib.h:25-30`).
pub mod gamemtl_pair_chunks {
    pub const GAMEMTLPAIR_CHUNK_PAIR: u32 = 0x1000;
    pub const GAMEMTLPAIR_CHUNK_BREAKING: u32 = 0x1002;
    pub const GAMEMTLPAIR_CHUNK_STEP: u32 = 0x1003;
    pub const GAMEMTLPAIR_CHUNK_COLLIDE: u32 = 0x1005;
}

/// `SGameMtl` behavior & collision flags (`GameMtlLib.h:80-103`).
pub mod gamemtl_flags {
    pub const FL_BREAKABLE: u32 = 1 << 0;
    pub const FL_BOUNCEABLE: u32 = 1 << 2;
    pub const FL_SKIDMARK: u32 = 1 << 3;
    pub const FL_BLOODMARK: u32 = 1 << 4;
    pub const FL_CLIMABLE: u32 = 1 << 5;
    pub const FL_PASSABLE: u32 = 1 << 7;
    pub const FL_DYNAMIC: u32 = 1 << 8;
    pub const FL_LIQUID: u32 = 1 << 9;
    pub const FL_SUPPRESS_SHADOWS: u32 = 1 << 10;
    pub const FL_SUPPRESS_WALLMARKS: u32 = 1 << 11;
    pub const FL_ACTOR_OBSTACLE: u32 = 1 << 12;
    pub const FL_NO_RICOCHET: u32 = 1 << 13;
    pub const FL_INJURIOUS: u32 = 1 << 28;
    pub const FL_SHOOTABLE: u32 = 1 << 29;
    pub const FL_TRANSPARENT: u32 = 1 << 30;
    pub const FL_SLOW_DOWN: u32 = 1 << 31;
}

/// `SGameMtlPair::OwnProps` flags (`GameMtlLib.h:172-181`).
pub mod gamemtl_pair_flags {
    pub const FL_BREAKING_SOUNDS: u32 = 1 << 1;
    pub const FL_STEP_SOUNDS: u32 = 1 << 2;
    pub const FL_COLLIDE_SOUNDS: u32 = 1 << 4;
    pub const FL_COLLIDE_PARTICLES: u32 = 1 << 5;
    pub const FL_COLLIDE_MARKS: u32 = 1 << 6;
}

/// Individual physical/surface material (`SGameMtl`).
#[derive(Debug, Clone, PartialEq)]
pub struct GameMaterial {
    pub id: u32,
    pub name: String,
    pub desc: String,
    pub flags: u32,
    pub ph_friction: f32,
    pub ph_damping: f32,
    pub ph_spring: f32,
    pub ph_bounce_start_velocity: f32,
    pub ph_bouncing: f32,
    pub flotation_factor: f32,
    pub shoot_factor: f32,
    pub shoot_factor_mp: f32,
    pub bounce_damage_factor: f32,
    pub injurious_speed: f32,
    pub vis_transparency_factor: f32,
    pub snd_occlusion_factor: f32,
    pub density_factor: f32,
    pub collision_sound_vel: f32,
}

impl Default for GameMaterial {
    fn default() -> Self {
        Self {
            id: GAMEMTL_NONE_ID,
            name: "unknown".to_string(),
            desc: String::new(),
            flags: 0,
            ph_friction: 1.0,
            ph_damping: 1.0,
            ph_spring: 1.0,
            ph_bounce_start_velocity: 0.0,
            ph_bouncing: 0.1,
            flotation_factor: 1.0,
            shoot_factor: 0.0,
            shoot_factor_mp: 0.0,
            bounce_damage_factor: 1.0,
            injurious_speed: 0.0,
            vis_transparency_factor: 0.0,
            snd_occlusion_factor: 0.0,
            density_factor: 0.0,
            collision_sound_vel: 10.0,
        }
    }
}

impl GameMaterial {
    #[inline]
    pub fn has_flag(&self, bit: u32) -> bool {
        (self.flags & bit) != 0
    }

    #[inline]
    pub fn set_flag(&mut self, bit: u32, enabled: bool) {
        if enabled {
            self.flags |= bit;
        } else {
            self.flags &= !bit;
        }
    }

    fn parse(data: &[u8]) -> Result<Self, FormatError> {
        use gamemtl_chunks::*;
        let reader = ChunkReader::new(data);
        let mut mtl = Self::default();

        let main_bytes = reader.require_chunk(GAMEMTL_CHUNK_MAIN)?;
        let mut mc = ByteCursor::new(&main_bytes);
        mtl.id = mc.read_u32()?;
        mtl.name = mc.read_zstring()?;

        if let Some(desc_bytes) = reader.open_chunk(GAMEMTL_CHUNK_DESC)? {
            let mut dc = ByteCursor::new(&desc_bytes);
            mtl.desc = dc.read_zstring()?;
        }

        let flag_bytes = reader.require_chunk(GAMEMTL_CHUNK_FLAGS)?;
        let mut fc = ByteCursor::new(&flag_bytes);
        mtl.flags = fc.read_u32()?;

        let phys_bytes = reader.require_chunk(GAMEMTL_CHUNK_PHYSICS)?;
        let mut pc = ByteCursor::new(&phys_bytes);
        mtl.ph_friction = pc.read_f32()?;
        mtl.ph_damping = pc.read_f32()?;
        mtl.ph_spring = pc.read_f32()?;
        mtl.ph_bounce_start_velocity = pc.read_f32()?;
        mtl.ph_bouncing = pc.read_f32()?;

        let fac_bytes = reader.require_chunk(GAMEMTL_CHUNK_FACTORS)?;
        let mut fac = ByteCursor::new(&fac_bytes);
        mtl.shoot_factor = fac.read_f32()?;
        mtl.bounce_damage_factor = fac.read_f32()?;
        mtl.vis_transparency_factor = fac.read_f32()?;
        mtl.snd_occlusion_factor = fac.read_f32()?;

        mtl.shoot_factor_mp = if let Some(mp_bytes) = reader.open_chunk(GAMEMTL_CHUNK_FACTORS_MP)? {
            let mut c = ByteCursor::new(&mp_bytes);
            c.read_f32()?
        } else {
            mtl.shoot_factor
        };

        if let Some(flot_bytes) = reader.open_chunk(GAMEMTL_CHUNK_FLOTATION)? {
            let mut c = ByteCursor::new(&flot_bytes);
            mtl.flotation_factor = c.read_f32()?;
        }
        if let Some(inj_bytes) = reader.open_chunk(GAMEMTL_CHUNK_INJURIOUS)? {
            let mut c = ByteCursor::new(&inj_bytes);
            mtl.injurious_speed = c.read_f32()?;
        }
        if let Some(dens_bytes) = reader.open_chunk(GAMEMTL_CHUNK_DENSITY)? {
            let mut c = ByteCursor::new(&dens_bytes);
            mtl.density_factor = c.read_f32()?;
        }

        Ok(mtl)
    }
}

/// Interaction pair between two game materials (`SGameMtlPair`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameMaterialPair {
    pub mtl0: u32,
    pub mtl1: u32,
    pub id: u32,
    pub id_parent: u32,
    pub own_props: u32,
    pub breaking_sounds_raw: String,
    pub step_sounds_raw: String,
    pub collide_sounds_raw: String,
    pub collide_particles_raw: String,
    pub collide_marks_raw: String,
}

impl Default for GameMaterialPair {
    fn default() -> Self {
        Self {
            mtl0: GAMEMTL_NONE_ID,
            mtl1: GAMEMTL_NONE_ID,
            id: GAMEMTL_NONE_ID,
            id_parent: GAMEMTL_NONE_ID,
            own_props: u32::MAX,
            breaking_sounds_raw: String::new(),
            step_sounds_raw: String::new(),
            collide_sounds_raw: String::new(),
            collide_particles_raw: String::new(),
            collide_marks_raw: String::new(),
        }
    }
}

impl GameMaterialPair {
    #[inline]
    pub fn matches_ids(&self, a: u32, b: u32) -> bool {
        (self.mtl0 == a && self.mtl1 == b) || (self.mtl0 == b && self.mtl1 == a)
    }

    pub fn breaking_sounds(&self) -> Vec<&str> {
        split_csv_items(&self.breaking_sounds_raw)
    }

    pub fn step_sounds(&self) -> Vec<&str> {
        split_csv_items(&self.step_sounds_raw)
    }

    pub fn collide_sounds(&self) -> Vec<&str> {
        split_csv_items(&self.collide_sounds_raw)
    }

    pub fn collide_particles(&self) -> Vec<&str> {
        split_csv_items(&self.collide_particles_raw)
    }

    pub fn collide_marks(&self) -> Vec<&str> {
        split_csv_items(&self.collide_marks_raw)
    }

    fn parse(data: &[u8]) -> Result<Self, FormatError> {
        use gamemtl_pair_chunks::*;
        let reader = ChunkReader::new(data);

        let pair_bytes = reader.require_chunk(GAMEMTLPAIR_CHUNK_PAIR)?;
        let mut pc = ByteCursor::new(&pair_bytes);
        let mtl0 = pc.read_u32()?;
        let mtl1 = pc.read_u32()?;
        let id = pc.read_u32()?;
        let id_parent = pc.read_u32()?;
        let own_props = pc.read_u32()?;

        let brk_bytes = reader.require_chunk(GAMEMTLPAIR_CHUNK_BREAKING)?;
        let mut bc = ByteCursor::new(&brk_bytes);
        let breaking_sounds_raw = bc.read_zstring()?;

        let step_bytes = reader.require_chunk(GAMEMTLPAIR_CHUNK_STEP)?;
        let mut sc = ByteCursor::new(&step_bytes);
        let step_sounds_raw = sc.read_zstring()?;

        let col_bytes = reader.require_chunk(GAMEMTLPAIR_CHUNK_COLLIDE)?;
        let mut cc = ByteCursor::new(&col_bytes);
        let collide_sounds_raw = cc.read_zstring()?;
        let collide_particles_raw = cc.read_zstring()?;
        let collide_marks_raw = cc.read_zstring()?;

        Ok(Self {
            mtl0,
            mtl1,
            id,
            id_parent,
            own_props,
            breaking_sounds_raw,
            step_sounds_raw,
            collide_sounds_raw,
            collide_particles_raw,
            collide_marks_raw,
        })
    }
}

/// Complete X-Ray game material & interaction pair database (`CGameMtlLibrary`).
#[derive(Debug, Clone, PartialEq)]
pub struct GameMaterialLibrary {
    pub version: u16,
    pub material_index: u32,
    pub material_pair_index: u32,
    pub materials: Vec<GameMaterial>,
    pub material_pairs: Vec<GameMaterialPair>,
}

impl GameMaterialLibrary {
    /// Parse a binary `gamemtl.xr` stream (`CGameMtlLibrary::Load` in `GameMtlLib.cpp:70-235`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        use gamemtl_root_chunks::*;
        let reader = ChunkReader::new(data);

        let ver_bytes = reader.require_chunk(GAMEMTLS_CHUNK_VERSION)?;
        let mut vc = ByteCursor::new(&ver_bytes);
        let version = vc.read_u16()?;
        if version != GAMEMTL_CURRENT_VERSION {
            return Err(FormatError::InvalidValue("unsupported gamemtl.xr version"));
        }

        let inc_bytes = reader.require_chunk(GAMEMTLS_CHUNK_AUTOINC)?;
        let mut ic = ByteCursor::new(&inc_bytes);
        let material_index = ic.read_u32()?;
        let material_pair_index = ic.read_u32()?;

        let mut materials = Vec::new();
        if let Some(mtls_bytes) = reader.open_chunk(GAMEMTLS_CHUNK_MTLS)? {
            let sub = ChunkReader::new(&mtls_bytes);
            for entry in sub.iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                materials.push(GameMaterial::parse(&payload)?);
            }
        }

        let mut material_pairs = Vec::new();
        if let Some(pairs_bytes) = reader.open_chunk(GAMEMTLS_CHUNK_MTLS_PAIR)? {
            let sub = ChunkReader::new(&pairs_bytes);
            for entry in sub.iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                material_pairs.push(GameMaterialPair::parse(&payload)?);
            }
        }

        Ok(Self {
            version,
            material_index,
            material_pair_index,
            materials,
            material_pairs,
        })
    }

    pub fn material_by_id(&self, id: u32) -> Option<&GameMaterial> {
        self.materials.iter().find(|m| m.id == id)
    }

    /// Look up material by name (case-insensitive matching `CGameMtlLibrary::GetMaterialIt`).
    pub fn material_by_name(&self, name: &str) -> Option<&GameMaterial> {
        self.materials
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(name))
    }

    pub fn material_index_by_id(&self, id: u32) -> Option<usize> {
        self.materials.iter().position(|m| m.id == id)
    }

    pub fn material_index_by_name(&self, name: &str) -> Option<usize> {
        self.materials
            .iter()
            .position(|m| m.name.eq_ignore_ascii_case(name))
    }

    /// Symmetric lookup for material interaction pair `(mtl0_id, mtl1_id)`.
    pub fn find_pair(&self, mtl0_id: u32, mtl1_id: u32) -> Option<&GameMaterialPair> {
        self.material_pairs
            .iter()
            .find(|p| p.matches_ids(mtl0_id, mtl1_id))
    }

    pub fn find_pair_by_names(&self, name0: &str, name1: &str) -> Option<&GameMaterialPair> {
        let m0 = self.material_by_name(name0)?.id;
        let m1 = self.material_by_name(name1)?.id;
        self.find_pair(m0, m1)
    }

    /// Apply Anomaly `materials/materials.ltx` additions/overrides (`GameMtlLib.cpp:117-185`).
    pub fn apply_materials_ltx(&mut self, ltx: &OwnedLtxDocument) {
        let mut biggest_id = self
            .materials
            .iter()
            .map(|m| m.id as i64)
            .max()
            .unwrap_or(-1);

        let mut names: Vec<&String> = ltx.sections.keys().collect();
        names.sort_unstable();

        for sec_name in names {
            let idx = if let Some(existing) = self
                .materials
                .iter()
                .position(|m| m.name.eq_ignore_ascii_case(sec_name.as_str()))
            {
                existing
            } else {
                biggest_id += 1;
                let created = GameMaterial {
                    id: biggest_id as u32,
                    name: sec_name.clone(),
                    ..Default::default()
                };
                self.materials.push(created);
                self.materials.len() - 1
            };

            let m = &mut self.materials[idx];
            let s = sec_name.as_str();
            if let Some(desc) = ltx.read_str(s, "desc") {
                m.desc = desc.to_string();
            }

            const FLAG_KEYS: &[(&str, u32)] = &[
                ("flag_breakable", gamemtl_flags::FL_BREAKABLE),
                ("flag_bounceable", gamemtl_flags::FL_BOUNCEABLE),
                ("flag_skidmark", gamemtl_flags::FL_SKIDMARK),
                ("flag_bloodmark", gamemtl_flags::FL_BLOODMARK),
                ("flag_climable", gamemtl_flags::FL_CLIMABLE),
                ("flag_passable", gamemtl_flags::FL_PASSABLE),
                ("flag_dynamic", gamemtl_flags::FL_DYNAMIC),
                ("flag_liquid", gamemtl_flags::FL_LIQUID),
                ("flag_suppress_shadows", gamemtl_flags::FL_SUPPRESS_SHADOWS),
                ("flag_suppress_wallmarks", gamemtl_flags::FL_SUPPRESS_WALLMARKS),
                ("flag_actor_obstacle", gamemtl_flags::FL_ACTOR_OBSTACLE),
                ("flag_bullet_no_ricochet", gamemtl_flags::FL_NO_RICOCHET),
                ("flag_injurious", gamemtl_flags::FL_INJURIOUS),
                ("flag_shootable", gamemtl_flags::FL_SHOOTABLE),
                ("flag_transparent", gamemtl_flags::FL_TRANSPARENT),
                ("flag_slowdown", gamemtl_flags::FL_SLOW_DOWN),
            ];
            for &(key, bit) in FLAG_KEYS {
                if ltx.read_str(s, key).is_some() {
                    m.set_flag(bit, ltx.read_bool(s, key, m.has_flag(bit)));
                }
            }

            if ltx.read_str(s, "friction").is_some() {
                m.ph_friction = ltx.read_f32(s, "friction", m.ph_friction);
            }
            if ltx.read_str(s, "damping").is_some() {
                m.ph_damping = ltx.read_f32(s, "damping", m.ph_damping);
            }
            if ltx.read_str(s, "spring").is_some() {
                m.ph_spring = ltx.read_f32(s, "spring", m.ph_spring);
            }
            if ltx.read_str(s, "bounce_start_velocity").is_some() {
                m.ph_bounce_start_velocity =
                    ltx.read_f32(s, "bounce_start_velocity", m.ph_bounce_start_velocity);
            }
            if ltx.read_str(s, "bouncing").is_some() {
                m.ph_bouncing = ltx.read_f32(s, "bouncing", m.ph_bouncing);
            }
            if ltx.read_str(s, "shoot_factor").is_some() {
                m.shoot_factor = ltx.read_f32(s, "shoot_factor", m.shoot_factor);
            }
            if ltx.read_str(s, "shoot_factor_mp").is_some() {
                m.shoot_factor_mp = ltx.read_f32(s, "shoot_factor_mp", m.shoot_factor_mp);
            }
            if ltx.read_str(s, "bounce_damage_factor").is_some() {
                m.bounce_damage_factor =
                    ltx.read_f32(s, "bounce_damage_factor", m.bounce_damage_factor);
            }
            if ltx.read_str(s, "vis_transparency_factor").is_some() {
                m.vis_transparency_factor =
                    ltx.read_f32(s, "vis_transparency_factor", m.vis_transparency_factor);
            }
            if ltx.read_str(s, "sound_occlusion_factor").is_some() {
                m.snd_occlusion_factor =
                    ltx.read_f32(s, "sound_occlusion_factor", m.snd_occlusion_factor);
            }
            if ltx.read_str(s, "flotation_factor").is_some() {
                m.flotation_factor = ltx.read_f32(s, "flotation_factor", m.flotation_factor);
            }
            if ltx.read_str(s, "injurious_factor").is_some() {
                m.injurious_speed = ltx.read_f32(s, "injurious_factor", m.injurious_speed);
            }
            if ltx.read_str(s, "density_factor").is_some() {
                m.density_factor = ltx.read_f32(s, "density_factor", m.density_factor);
            }
            if ltx.read_str(s, "collision_sound_vel").is_some() {
                m.collision_sound_vel = ltx.read_f32(s, "collision_sound_vel", m.collision_sound_vel);
            }
        }
    }

    /// Apply Anomaly `materials/material_pairs.ltx` (`[matA@matB]` sections) additions/overrides
    /// (`GameMtlLib.cpp:238-324`).
    pub fn apply_material_pairs_ltx(&mut self, ltx: &OwnedLtxDocument) {
        let mut biggest_id = self
            .material_pairs
            .iter()
            .map(|p| p.id as i64)
            .max()
            .unwrap_or(-1);

        let mut names: Vec<&String> = ltx.sections.keys().collect();
        names.sort_unstable();

        for sec_name in names {
            let parts: Vec<&str> = sec_name.split('@').collect();
            if parts.len() < 2 {
                continue;
            }
            let Some(m0) = self.material_by_name(parts[0].trim()).map(|m| m.id) else {
                continue;
            };
            let Some(m1) = self.material_by_name(parts[1].trim()).map(|m| m.id) else {
                continue;
            };

            let pair_idx = if let Some(existing) = self
                .material_pairs
                .iter()
                .position(|p| p.matches_ids(m0, m1))
            {
                existing
            } else {
                biggest_id += 1;
                self.material_pairs.push(GameMaterialPair {
                    mtl0: m0,
                    mtl1: m1,
                    id: biggest_id as u32,
                    id_parent: GAMEMTL_NONE_ID,
                    own_props: u32::MAX,
                    ..GameMaterialPair::default()
                });
                self.material_pairs.len() - 1
            };

            let pair = &mut self.material_pairs[pair_idx];
            let s = sec_name.as_str();
            if let Some(v) = ltx.read_str(s, "breaking_sounds") {
                pair.breaking_sounds_raw = v.to_string();
                pair.own_props |= gamemtl_pair_flags::FL_BREAKING_SOUNDS;
            }
            if let Some(v) = ltx.read_str(s, "step_sounds") {
                pair.step_sounds_raw = v.to_string();
                pair.own_props |= gamemtl_pair_flags::FL_STEP_SOUNDS;
            }
            if let Some(v) = ltx.read_str(s, "collide_sounds") {
                pair.collide_sounds_raw = v.to_string();
                pair.own_props |= gamemtl_pair_flags::FL_COLLIDE_SOUNDS;
            }
            if let Some(v) = ltx.read_str(s, "collide_particles") {
                pair.collide_particles_raw = v.to_string();
                pair.own_props |= gamemtl_pair_flags::FL_COLLIDE_PARTICLES;
            }
            if let Some(v) = ltx.read_str(s, "collide_marks") {
                pair.collide_marks_raw = v.to_string();
                pair.own_props |= gamemtl_pair_flags::FL_COLLIDE_MARKS;
            }
        }
    }
}

fn split_csv_items(raw: &str) -> Vec<&str> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use xray_core::ltx_loader::LoadedLtxSource;

    fn build_material_chunk(id: u32, name: &str, flags: u32, shoot_factor: f32) -> Vec<u8> {
        use gamemtl_chunks::*;
        let mut mw = ChunkWriter::new();
        let mut main = id.to_le_bytes().to_vec();
        main.extend_from_slice(name.as_bytes());
        main.push(0);
        mw.write_chunk(GAMEMTL_CHUNK_MAIN, &main);
        mw.write_chunk(GAMEMTL_CHUNK_DESC, b"test material description\0");
        mw.write_chunk(GAMEMTL_CHUNK_FLAGS, &flags.to_le_bytes());

        let mut phys = Vec::new();
        for f in [0.9f32, 0.2, 1.0, 0.5, 0.3] {
            phys.extend_from_slice(&f.to_le_bytes());
        }
        mw.write_chunk(GAMEMTL_CHUNK_PHYSICS, &phys);

        let mut fac = Vec::new();
        for f in [shoot_factor, 1.0, 0.0, 0.4] {
            fac.extend_from_slice(&f.to_le_bytes());
        }
        mw.write_chunk(GAMEMTL_CHUNK_FACTORS, &fac);
        mw.write_chunk(GAMEMTL_CHUNK_FLOTATION, &0.85f32.to_le_bytes());
        mw.write_chunk(GAMEMTL_CHUNK_DENSITY, &7.8f32.to_le_bytes());
        mw.into_bytes()
    }

    fn build_pair_chunk(mtl0: u32, mtl1: u32, id: u32) -> Vec<u8> {
        use gamemtl_pair_chunks::*;
        let mut pw = ChunkWriter::new();
        let mut pair = Vec::new();
        for v in [mtl0, mtl1, id, GAMEMTL_NONE_ID, gamemtl_pair_flags::FL_STEP_SOUNDS] {
            pair.extend_from_slice(&v.to_le_bytes());
        }
        pw.write_chunk(GAMEMTLPAIR_CHUNK_PAIR, &pair);
        pw.write_chunk(GAMEMTLPAIR_CHUNK_BREAKING, b"material\\metal_break\0");
        pw.write_chunk(
            GAMEMTLPAIR_CHUNK_STEP,
            b"material\\human\\step\\metal_1, material\\human\\step\\metal_2\0",
        );
        let mut col = Vec::new();
        col.extend_from_slice(b"material\\bullet_metal\0");
        col.extend_from_slice(b"hit_fx\\hit_metal_01, hit_fx\\hit_metal_02\0");
        col.extend_from_slice(b"wm\\wm_bullet_metal_1\0");
        pw.write_chunk(GAMEMTLPAIR_CHUNK_COLLIDE, &col);
        pw.into_bytes()
    }

    #[test]
    fn test_gamemtl_parse_lookup_and_ltx_overlays() {
        use gamemtl_root_chunks::*;
        let mut mtls_w = ChunkWriter::new();
        mtls_w.write_chunk(
            0,
            &build_material_chunk(
                0,
                "default",
                gamemtl_flags::FL_SHOOTABLE | gamemtl_flags::FL_BOUNCEABLE,
                0.5,
            ),
        );
        mtls_w.write_chunk(
            1,
            &build_material_chunk(
                5,
                "objects\\metal",
                gamemtl_flags::FL_BOUNCEABLE | gamemtl_flags::FL_ACTOR_OBSTACLE,
                0.2,
            ),
        );

        let mut pairs_w = ChunkWriter::new();
        pairs_w.write_chunk(0, &build_pair_chunk(0, 5, 10));

        let mut root = ChunkWriter::new();
        root.write_chunk(GAMEMTLS_CHUNK_VERSION, &GAMEMTL_CURRENT_VERSION.to_le_bytes());
        let mut autoinc = Vec::new();
        autoinc.extend_from_slice(&6u32.to_le_bytes());
        autoinc.extend_from_slice(&11u32.to_le_bytes());
        root.write_chunk(GAMEMTLS_CHUNK_AUTOINC, &autoinc);
        root.write_chunk(GAMEMTLS_CHUNK_MTLS, &mtls_w.into_bytes());
        root.write_chunk(GAMEMTLS_CHUNK_MTLS_PAIR, &pairs_w.into_bytes());

        let mut lib = GameMaterialLibrary::parse(&root.into_bytes()).expect("gamemtl.xr must parse");
        assert_eq!(lib.materials.len(), 2);
        let metal = lib.material_by_name("OBJECTS\\METAL").unwrap();
        assert_eq!(metal.id, 5);
        assert_eq!(metal.shoot_factor_mp, 0.2); // defaulted from shoot_factor
        assert!((metal.density_factor - 7.8).abs() < 1e-5);

        // Symmetric pair lookup works regardless of argument order
        let pair = lib.find_pair_by_names("objects\\metal", "default").unwrap();
        assert_eq!(
            pair.step_sounds(),
            vec!["material\\human\\step\\metal_1", "material\\human\\step\\metal_2"]
        );
        assert_eq!(
            pair.collide_particles(),
            vec!["hit_fx\\hit_metal_01", "hit_fx\\hit_metal_02"]
        );

        // Apply loose LTX overlays
        let mtl_ltx = OwnedLtxDocument::from_sources(&[LoadedLtxSource {
            virtual_path: "materials\\materials.ltx".to_string(),
            depth: 0,
            text: "[objects\\metal]\nshoot_factor = 0.12\n\n[objects\\ceramic]\nflag_breakable = true\nfriction = 0.75\n".to_string(),
        }]);
        lib.apply_materials_ltx(&mtl_ltx);
        assert_eq!(lib.material_by_name("objects\\metal").unwrap().shoot_factor, 0.12);
        let ceramic = lib.material_by_name("objects\\ceramic").unwrap();
        assert_eq!(ceramic.id, 6);
        assert!(ceramic.has_flag(gamemtl_flags::FL_BREAKABLE));

        let pair_ltx = OwnedLtxDocument::from_sources(&[LoadedLtxSource {
            virtual_path: "materials\\material_pairs.ltx".to_string(),
            depth: 0,
            text: "[objects\\ceramic@objects\\metal]\ncollide_sounds = material\\ceramic_clink\ncollide_marks = wm\\wm_ceramic\n".to_string(),
        }]);
        lib.apply_material_pairs_ltx(&pair_ltx);
        let new_pair = lib
            .find_pair_by_names("objects\\metal", "objects\\ceramic")
            .unwrap();
        assert_eq!(new_pair.id, 11);
        assert_eq!(new_pair.collide_sounds(), vec!["material\\ceramic_clink"]);
        assert_eq!(new_pair.collide_marks(), vec!["wm\\wm_ceramic"]);

        // Error checks: unsupported version and missing version chunk
        let mut bad_ver = ChunkWriter::new();
        bad_ver.write_chunk(GAMEMTLS_CHUNK_VERSION, &99u16.to_le_bytes());
        assert!(matches!(
            GameMaterialLibrary::parse(&bad_ver.into_bytes()),
            Err(FormatError::InvalidValue(_))
        ));
        assert!(matches!(
            GameMaterialLibrary::parse(&[]),
            Err(FormatError::MissingChunk(GAMEMTLS_CHUNK_VERSION))
        ));
    }
}
