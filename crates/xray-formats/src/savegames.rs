//! X-Ray Monolith / S.T.A.L.K.E.R. Anomaly `.scop` ALife Savegame Parser & Writer
//!
//! Mirrors `CALifeStorageManager`, `CSavedGameWrapper`, `CALifeSimulatorHeader`,
//! `CALifeTimeManager`, `CALifeSpawnRegistry`, `CALifeObjectRegistry`,
//! `CALifeRegistryContainer`, and `date_time.cpp`:
//! - Outer compressed container: `[u32 magic = 0xFFFF_FFFF][u32 alife_version][u32 uncompressed_size][lzo1x payload]`
//! - Inner chunked stream:
//!   - `ALIFE_CHUNK_DATA` (`0x0000`): ALife version (`u32`)
//!   - `SPAWN_CHUNK_DATA` (`0x0001`): subchunk `0` (`spawn_name`, 16-byte `spawn_guid`, optional Anomaly `level_id` + `level_name`), subchunk `1` (`save_updates` vertex chunk stream)
//!   - `OBJECT_CHUNK_DATA` (`0x0002`): `[u32 count]` dynamic objects (`[u16 spawn_len][M_SPAWN NET_Packet][u16 update_len][M_UPDATE NET_Packet]`)
//!   - `GAME_TIME_CHUNK_DATA` (`0x0005`): `[u64 game_time_ms][f32 time_factor][f32 normal_time_factor]`
//!   - `REGISTRY_CHUNK_DATA` (`0x0009`): sequential ALife registries (`info_portions`, `character_relations`, `game_news`, `specific_characters`, `map_locations`, `game_tasks`, `actor_statistics`) plus optional mod trailing bytes.

use crate::chunk::{ByteCursor, ChunkReader, ChunkWriter, FormatError};
use crate::spawns::{AlifeObjectState, SpawnPacket, M_UPDATE};

pub const SAVE_EXTENSION: &str = ".scop";
pub const SAVE_MAGIC: u32 = u32::MAX;
pub const ALIFE_VERSION: u32 = 0x0006;

pub const ALIFE_CHUNK_DATA: u32 = 0x0000;
pub const SPAWN_CHUNK_DATA: u32 = 0x0001;
pub const OBJECT_CHUNK_DATA: u32 = 0x0002;
pub const GAME_TIME_CHUNK_DATA: u32 = 0x0005;
pub const REGISTRY_CHUNK_DATA: u32 = 0x0009;

/// Unpacked Gregorian game date/time matching `generate_time` / `split_time` (`src/xrGame/date_time.cpp`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GameDateTime {
    pub year: u32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub millisecond: u32,
}

impl GameDateTime {
    /// Pack into X-Ray's 64-bit millisecond counter (`ALife::_TIME_ID`, `generate_time_impl`).
    pub fn to_game_time_ms(&self) -> u64 {
        let years = self.year.max(1);
        let extra_day_count: u64 =
            if (years.is_multiple_of(400)) || (years.is_multiple_of(4) && !years.is_multiple_of(100)) {
                1
            } else {
                0
            };
        let years_minus_1 = u64::from(years - 1);
        let mut result = years_minus_1 * 365 + years_minus_1 / 4 - years_minus_1 / 100 + years_minus_1 / 400;
        if self.month > 1 {
            result += 31;
        }
        if self.month > 2 {
            result += 28 + extra_day_count;
        }
        if self.month > 3 {
            result += 31;
        }
        if self.month > 4 {
            result += 30;
        }
        if self.month > 5 {
            result += 31;
        }
        if self.month > 6 {
            result += 30;
        }
        if self.month > 7 {
            result += 31;
        }
        if self.month > 8 {
            result += 31;
        }
        if self.month > 9 {
            result += 30;
        }
        if self.month > 10 {
            result += 31;
        }
        if self.month > 11 {
            result += 30;
        }
        result += u64::from(self.day.saturating_sub(1));
        result = result * 24 + u64::from(self.hour);
        result = result * 60 + u64::from(self.minute);
        result = result * 60 + u64::from(self.second);
        result = result * 1000 + u64::from(self.millisecond);
        result
    }

    /// Unpack an X-Ray 64-bit millisecond counter (`ALife::_TIME_ID`, `split_time_impl`).
    pub fn from_game_time_ms(mut time: u64) -> Self {
        let millisecond = (time % 1000) as u32;
        time /= 1000;
        let second = (time % 60) as u32;
        time /= 60;
        let minute = (time % 60) as u32;
        time /= 60;
        let hour = (time % 24) as u32;
        time /= 24;

        let p0 = time / (400 * 365 + 100 - 4 + 1);
        time -= p0 * (400 * 365 + 100 - 4 + 1);
        let p1 = time / (100 * 365 + 25 - 1);
        time -= p1 * (100 * 365 + 25 - 1);
        let p2 = time / (4 * 365 + 1);
        time -= p2 * (4 * 365 + 1);
        let p3 = ((time as u32) / 365).min(3) as u64;
        time -= p3 * 365;
        let year = (400 * p0 + 100 * p1 + 4 * p2 + p3 + 1) as u32;
        time += 1;

        let extra_day_count: u64 =
            if (year.is_multiple_of(400)) || (year.is_multiple_of(4) && !year.is_multiple_of(100)) {
                1
            } else {
                0
            };

        let mut month = 1u32;
        let month_lengths = [
            31u64,
            28 + extra_day_count,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
        ];
        for len in month_lengths {
            if time > len {
                month += 1;
                time -= len;
            } else {
                break;
            }
        }
        let day = time as u32;

        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            millisecond,
        }
    }

    /// Parse `HH:MM:SS` and `DD.MM.YYYY` INI strings (`CALifeTimeManager::init`).
    pub fn from_ini_strings(time_str: &str, date_str: &str) -> Result<Self, FormatError> {
        let mut t_parts = time_str.trim().split(':');
        let hour = t_parts
            .next()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .ok_or(FormatError::InvalidValue("invalid start_time hour"))?;
        let minute = t_parts
            .next()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .ok_or(FormatError::InvalidValue("invalid start_time minute"))?;
        let second = t_parts
            .next()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .ok_or(FormatError::InvalidValue("invalid start_time second"))?;

        let mut d_parts = date_str.trim().split('.');
        let day = d_parts
            .next()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .ok_or(FormatError::InvalidValue("invalid start_date day"))?;
        let month = d_parts
            .next()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .ok_or(FormatError::InvalidValue("invalid start_date month"))?;
        let year = d_parts
            .next()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .ok_or(FormatError::InvalidValue("invalid start_date year"))?;

        if year == 0 || !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour >= 24 || minute >= 60 || second >= 60 {
            return Err(FormatError::InvalidValue("start date/time out of range"));
        }

        Ok(Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            millisecond: 0,
        })
    }
}

/// Validates an outer `.scop` header matching `CSavedGameWrapper::valid_saved_game`.
/// Returns `Some((alife_version, uncompressed_size))` if valid.
pub fn validate_scop_header(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 12 {
        return None;
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().ok()?);
    let version = u32::from_le_bytes(data[4..8].try_into().ok()?);
    let uncompressed_size = u32::from_le_bytes(data[8..12].try_into().ok()?);
    if magic == SAVE_MAGIC && version >= ALIFE_VERSION {
        Some((version, uncompressed_size))
    } else {
        None
    }
}

/// Decompress an outer `.scop` container into `(header_alife_version, inner_chunk_bytes)`.
pub fn decompress_scop(data: &[u8]) -> Result<(u32, Vec<u8>), FormatError> {
    if data.len() < 12 {
        return Err(FormatError::UnexpectedEof);
    }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != SAVE_MAGIC {
        return Err(FormatError::InvalidValue("invalid .scop magic header"));
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version < ALIFE_VERSION {
        return Err(FormatError::InvalidValue("unsupported .scop ALife version"));
    }
    let uncompressed_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    if uncompressed_size > 512 * 1024 * 1024 {
        return Err(FormatError::InvalidValue(
            ".scop uncompressed size exceeds sanity bound",
        ));
    }
    let payload = &data[12..];
    let mut inner = vec![0u8; uncompressed_size];
    let written = xray_vfs::decompress_lzo1x(payload, &mut inner)
        .map_err(|_| FormatError::InvalidValue("LZO1X decompression failed for .scop savegame"))?;
    inner.truncate(written);
    Ok((version, inner))
}

/// Compress an inner ALife stream into the `.scop` outer LZO1X container.
pub fn compress_scop(alife_version: u32, inner_stream: &[u8]) -> Vec<u8> {
    let compressed = xray_vfs::compress_lzo1x(inner_stream);
    let mut out = Vec::with_capacity(12 + compressed.len());
    out.extend_from_slice(&SAVE_MAGIC.to_le_bytes());
    out.extend_from_slice(&alife_version.to_le_bytes());
    out.extend_from_slice(&(inner_stream.len() as u32).to_le_bytes());
    out.extend_from_slice(&compressed);
    out
}

/// ALife game time header (`GAME_TIME_CHUNK_DATA`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SaveGameTime {
    pub game_time_ms: u64,
    pub time_factor: f32,
    pub normal_time_factor: f32,
}

impl Default for SaveGameTime {
    fn default() -> Self {
        Self {
            game_time_ms: 0,
            time_factor: 10.0,
            normal_time_factor: 10.0,
        }
    }
}

impl SaveGameTime {
    pub fn date_time(&self) -> GameDateTime {
        GameDateTime::from_game_time_ms(self.game_time_ms)
    }

    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        Ok(Self {
            game_time_ms: cur.read_u64()?,
            time_factor: cur.read_f32()?,
            normal_time_factor: cur.read_f32()?,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&self.game_time_ms.to_le_bytes());
        out.extend_from_slice(&self.time_factor.to_le_bytes());
        out.extend_from_slice(&self.normal_time_factor.to_le_bytes());
        out
    }
}

/// Spawn reference & optional level-pack fast-UI info stored in `SPAWN_CHUNK_DATA`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveSpawnHeader {
    pub spawn_name: String,
    pub spawn_guid: [u8; 16],
    /// Anomaly level-pack fast-UI extension (`saved_game_wrapper.cpp:153-158`).
    pub level_id: Option<u8>,
    pub level_name: Option<String>,
    /// Per-spawn-vertex update subchunks in `SPAWN_CHUNK_DATA` chunk `1`.
    pub spawn_updates: Vec<(u32, Vec<u8>)>,
}

impl Default for SaveSpawnHeader {
    fn default() -> Self {
        Self {
            spawn_name: "all".to_string(),
            spawn_guid: [0u8; 16],
            level_id: None,
            level_name: None,
            spawn_updates: Vec::new(),
        }
    }
}

impl SaveSpawnHeader {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let sub0 = reader
            .open_chunk(0)?
            .ok_or(FormatError::MissingChunk(0))?;
        let mut cur = ByteCursor::new(&sub0);
        let spawn_name = cur.read_zstring()?;
        let guid_bytes = cur.read_bytes(16)?;
        let mut spawn_guid = [0u8; 16];
        spawn_guid.copy_from_slice(guid_bytes);

        let (level_id, level_name) = if !cur.is_eof() {
            let id = cur.read_u8()?;
            let name = cur.read_zstring()?;
            (Some(id), Some(name))
        } else {
            (None, None)
        };

        let mut spawn_updates = Vec::new();
        if let Some(sub1) = reader.open_chunk(1)? {
            for entry in ChunkReader::new(&sub1).iter() {
                let entry = entry?;
                spawn_updates.push((entry.id, entry.decode()?.into_owned()));
            }
        }

        Ok(Self {
            spawn_name,
            spawn_guid,
            level_id,
            level_name,
            spawn_updates,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut root = ChunkWriter::new();

        let mut sub0 = Vec::with_capacity(self.spawn_name.len() + 32);
        sub0.extend_from_slice(self.spawn_name.as_bytes());
        sub0.push(0);
        sub0.extend_from_slice(&self.spawn_guid);
        if let Some(level_id) = self.level_id {
            sub0.push(level_id);
            if let Some(level_name) = &self.level_name {
                sub0.extend_from_slice(level_name.as_bytes());
            }
            sub0.push(0);
        }
        root.write_chunk(0, &sub0);

        let mut sub1 = ChunkWriter::new();
        for (vertex_id, payload) in &self.spawn_updates {
            sub1.write_chunk(*vertex_id, payload);
        }
        root.write_chunk(1, &sub1.into_bytes());

        root.into_bytes()
    }
}

/// Dynamic ALife object record inside `OBJECT_CHUNK_DATA`.
#[derive(Debug, Clone, PartialEq)]
pub struct SaveObjectRecord {
    pub spawn: SpawnPacket,
    pub update_msg_id: u16,
    /// Raw payload bytes after the 2-byte `M_UPDATE` header.
    pub update_payload: Vec<u8>,
}

impl SaveObjectRecord {
    /// Decode the `CSE_ALifeObject` header prefix when present.
    pub fn alife_state(&self) -> Result<Option<AlifeObjectState>, FormatError> {
        if self.spawn.state_payload.is_empty() {
            Ok(None)
        } else {
            AlifeObjectState::parse(self.spawn.version, &self.spawn.state_payload).map(Some)
        }
    }

    /// Decode `CSE_ALifeCreatureAbstract::fHealth` for the actor (`id == 0`) or creature subclasses
    /// whose `M_UPDATE` starts with `fHealth: f32` (`xrServer_Objects_ALife_Monsters.cpp:1123`).
    pub fn creature_update_health(&self) -> Option<f32> {
        if self.update_payload.len() < 4 {
            return None;
        }
        Some(f32::from_le_bytes([
            self.update_payload[0],
            self.update_payload[1],
            self.update_payload[2],
            self.update_payload[3],
        ]))
    }

    pub fn parse_list(data: &[u8]) -> Result<Vec<Self>, FormatError> {
        let mut cur = ByteCursor::new(data);
        let count = cur.read_u32()? as usize;
        if count > 1_000_000 {
            return Err(FormatError::InvalidValue("object count exceeds sanity bound"));
        }
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let spawn_len = cur.read_u16()? as usize;
            let spawn_bytes = cur.read_bytes(spawn_len)?;
            let spawn = SpawnPacket::parse(spawn_bytes)?;

            let update_len = cur.read_u16()? as usize;
            let update_bytes = cur.read_bytes(update_len)?;
            if update_bytes.len() < 2 {
                return Err(FormatError::InvalidValue("truncated M_UPDATE packet"));
            }
            let update_msg_id = u16::from_le_bytes([update_bytes[0], update_bytes[1]]);
            if update_msg_id != M_UPDATE {
                return Err(FormatError::InvalidValue("expected M_UPDATE packet ID (0)"));
            }
            let update_payload = update_bytes[2..].to_vec();

            out.push(Self {
                spawn,
                update_msg_id,
                update_payload,
            });
        }
        Ok(out)
    }

    pub fn encode_list(objects: &[Self]) -> Result<Vec<u8>, FormatError> {
        let mut out = Vec::with_capacity(4 + objects.len() * 64);
        out.extend_from_slice(&(objects.len() as u32).to_le_bytes());
        for obj in objects {
            let spawn_bytes = obj.spawn.to_bytes();
            let spawn_len = u16::try_from(spawn_bytes.len())
                .map_err(|_| FormatError::InvalidValue("spawn packet exceeds u16"))?;
            out.extend_from_slice(&spawn_len.to_le_bytes());
            out.extend_from_slice(&spawn_bytes);

            let update_total = obj.update_payload.len() + 2;
            let update_len = u16::try_from(update_total)
                .map_err(|_| FormatError::InvalidValue("update packet exceeds u16"))?;
            out.extend_from_slice(&update_len.to_le_bytes());
            out.extend_from_slice(&obj.update_msg_id.to_le_bytes());
            out.extend_from_slice(&obj.update_payload);
        }
        Ok(out)
    }
}

/// Character relation goodwill data (`RELATION_DATA`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelationData {
    pub personal: Vec<(u16, i32)>,
    pub communities: Vec<(i32, i32)>,
}

/// PDA news item (`GAME_NEWS_DATA` in `src/xrGame/game_news.h`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameNewsEntry {
    pub news_type: u32,
    pub caption: String,
    pub text: String,
    pub receive_time: u64,
    pub texture_name: String,
}

/// Saved PDA map spot entry (`SLocationKey` + `CMapLocation`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedMapLocation {
    pub object_id: u16,
    pub spot_type: String,
    pub reserved_flag: u8,
    pub hint: String,
    pub flags: u32,
    pub owner_task_id: String,
}

/// Saved Lua script helper lists inside `CGameTask` (`SScriptTaskHelper`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SavedTaskScriptHelper {
    pub complete_lua_functions: Vec<String>,
    pub fail_lua_functions: Vec<String>,
    pub lua_functions_on_complete: Vec<String>,
    pub lua_functions_on_fail: Vec<String>,
}

/// Saved mission/task entry (`SGameTaskKey` + `CGameTask::save_task`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedGameTask {
    pub task_id: String,
    pub task_state: u32,
    pub task_type: u32,
    pub receive_time: u64,
    pub finish_time: u64,
    pub time_to_complete: u64,
    pub timer_finish: u64,
    pub title: String,
    pub description: String,
    pub script_helper: SavedTaskScriptHelper,
    pub icon_texture_name: String,
    pub map_hint: String,
    pub map_location: String,
    pub map_object_id: u16,
    pub priority: u32,
}

/// Task registry container (`CGameTaskRegistry`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GameTaskRegistryData {
    pub tasks_by_owner: Vec<(u16, Vec<SavedGameTask>)>,
    pub active_task_id: String,
}

/// Detail line in actor statistics (`SStatDetailBData`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorStatDetail {
    pub key: String,
    pub int_count: i32,
    pub int_points: i32,
    pub str_value: String,
}

/// Section entry in actor statistics (`SStatSectionData`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActorStatSection {
    pub details: Vec<ActorStatDetail>,
    pub key: String,
}

/// Sequential ALife registries in `REGISTRY_CHUNK_DATA` (`0x0009`, `CALifeRegistryContainer`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SaveRegistries {
    pub info_portions: Vec<(u16, Vec<String>)>,
    pub character_relations: Vec<(u16, RelationData)>,
    pub game_news: Vec<(u16, Vec<GameNewsEntry>)>,
    pub specific_characters: Vec<(String, i32)>,
    pub map_locations: Vec<(u16, Vec<SavedMapLocation>)>,
    pub game_tasks: GameTaskRegistryData,
    pub actor_statistics: Vec<(u16, Vec<ActorStatSection>)>,
    /// Any trailing bytes appended after the standard 7 registries.
    pub trailing_bytes: Vec<u8>,
}

fn read_str_vec(cur: &mut ByteCursor<'_>) -> Result<Vec<String>, FormatError> {
    let len = cur.read_u32()? as usize;
    if len > 1_000_000 {
        return Err(FormatError::InvalidValue("string list count exceeds sanity bound"));
    }
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(cur.read_zstring()?);
    }
    Ok(out)
}

fn write_zstring(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
    out.push(0);
}

fn write_str_vec(out: &mut Vec<u8>, items: &[String]) {
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for s in items {
        write_zstring(out, s);
    }
}

impl SaveRegistries {
    pub fn parse(data: &[u8], alife_version: u32) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);

        // 1. m_info_portions
        let info_count = cur.read_u32()? as usize;
        let mut info_portions = Vec::with_capacity(info_count);
        for _ in 0..info_count {
            let id = cur.read_u16()?;
            let list = read_str_vec(&mut cur)?;
            info_portions.push((id, list));
        }

        // 2. m_character_relations
        let rel_count = cur.read_u32()? as usize;
        let mut character_relations = Vec::with_capacity(rel_count);
        for _ in 0..rel_count {
            let id = cur.read_u16()?;
            let p_len = cur.read_u32()? as usize;
            let mut personal = Vec::with_capacity(p_len);
            for _ in 0..p_len {
                personal.push((cur.read_u16()?, cur.read_i32()?));
            }
            let c_len = cur.read_u32()? as usize;
            let mut communities = Vec::with_capacity(c_len);
            for _ in 0..c_len {
                communities.push((cur.read_i32()?, cur.read_i32()?));
            }
            character_relations.push((
                id,
                RelationData {
                    personal,
                    communities,
                },
            ));
        }

        // 3. m_game_news
        let news_owners = cur.read_u32()? as usize;
        let mut game_news = Vec::with_capacity(news_owners);
        for _ in 0..news_owners {
            let id = cur.read_u16()?;
            let n_len = cur.read_u32()? as usize;
            let mut items = Vec::with_capacity(n_len);
            for _ in 0..n_len {
                let news_type = cur.read_u32()?;
                let caption = cur.read_zstring()?;
                let text = cur.read_zstring()?;
                let receive_time = cur.read_u64()?;
                let texture_name = cur.read_zstring()?;
                items.push(GameNewsEntry {
                    news_type,
                    caption,
                    text,
                    receive_time,
                    texture_name,
                });
            }
            game_news.push((id, items));
        }

        // 4. m_specific_characters
        let sc_count = cur.read_u32()? as usize;
        let mut specific_characters = Vec::with_capacity(sc_count);
        for _ in 0..sc_count {
            specific_characters.push((cur.read_zstring()?, cur.read_i32()?));
        }

        // 5. m_map_locations
        let map_owners = cur.read_u32()? as usize;
        let mut map_locations = Vec::with_capacity(map_owners);
        for _ in 0..map_owners {
            let id = cur.read_u16()?;
            let l_len = cur.read_u32()? as usize;
            let mut locs = Vec::with_capacity(l_len);
            for _ in 0..l_len {
                let object_id = cur.read_u16()?;
                let spot_type = cur.read_zstring()?;
                let reserved_flag = cur.read_u8()?;
                let hint = cur.read_zstring()?;
                let flags = cur.read_u32()?;
                let owner_task_id = cur.read_zstring()?;
                locs.push(SavedMapLocation {
                    object_id,
                    spot_type,
                    reserved_flag,
                    hint,
                    flags,
                    owner_task_id,
                });
            }
            map_locations.push((id, locs));
        }

        // 6. m_game_tasks
        let task_owners = cur.read_u32()? as usize;
        let mut tasks_by_owner = Vec::with_capacity(task_owners);
        for _ in 0..task_owners {
            let id = cur.read_u16()?;
            let t_len = cur.read_u32()? as usize;
            let mut tasks = Vec::with_capacity(t_len);
            for _ in 0..t_len {
                let task_id = cur.read_zstring()?;
                let task_state = cur.read_u32()?;
                let task_type = cur.read_u32()?;
                let receive_time = cur.read_u64()?;
                let finish_time = cur.read_u64()?;
                let time_to_complete = cur.read_u64()?;
                let timer_finish = cur.read_u64()?;
                let title = cur.read_zstring()?;
                let description = cur.read_zstring()?;
                let script_helper = SavedTaskScriptHelper {
                    complete_lua_functions: read_str_vec(&mut cur)?,
                    fail_lua_functions: read_str_vec(&mut cur)?,
                    lua_functions_on_complete: read_str_vec(&mut cur)?,
                    lua_functions_on_fail: read_str_vec(&mut cur)?,
                };
                let icon_texture_name = cur.read_zstring()?;
                let map_hint = cur.read_zstring()?;
                let map_location = cur.read_zstring()?;
                let map_object_id = cur.read_u16()?;
                let priority = cur.read_u32()?;
                tasks.push(SavedGameTask {
                    task_id,
                    task_state,
                    task_type,
                    receive_time,
                    finish_time,
                    time_to_complete,
                    timer_finish,
                    title,
                    description,
                    script_helper,
                    icon_texture_name,
                    map_hint,
                    map_location,
                    map_object_id,
                    priority,
                });
            }
            tasks_by_owner.push((id, tasks));
        }
        let active_task_id = cur.read_zstring()?;

        // 7. m_actor_statistics
        let stat_owners = cur.read_u32()? as usize;
        let mut actor_statistics = Vec::with_capacity(stat_owners);
        for _ in 0..stat_owners {
            let id = cur.read_u16()?;
            let sec_len = cur.read_u32()? as usize;
            let mut sections = Vec::with_capacity(sec_len);
            for _ in 0..sec_len {
                let det_len = cur.read_u32()? as usize;
                let mut details = Vec::with_capacity(det_len);
                for _ in 0..det_len {
                    let key = cur.read_zstring()?;
                    let int_count = cur.read_i32()?;
                    let int_points = cur.read_i32()?;
                    let str_value = if alife_version > 0x0002 {
                        cur.read_zstring()?
                    } else {
                        String::new()
                    };
                    details.push(ActorStatDetail {
                        key,
                        int_count,
                        int_points,
                        str_value,
                    });
                }
                let key = if alife_version == 0x0002 {
                    let legacy_tok = cur.read_i32()?;
                    let _legacy_total = cur.read_i32()?;
                    match legacy_tok {
                        100 => "total",
                        1 => "stalkerkills",
                        2 => "monsterkills",
                        3 => "quests",
                        4 => "artefacts",
                        5 => "reputation",
                        _ => "foo",
                    }
                    .to_string()
                } else {
                    cur.read_zstring()?
                };
                sections.push(ActorStatSection { details, key });
            }
            actor_statistics.push((id, sections));
        }

        let trailing_bytes = if cur.is_eof() {
            Vec::new()
        } else {
            data[cur.position()..].to_vec()
        };

        Ok(Self {
            info_portions,
            character_relations,
            game_news,
            specific_characters,
            map_locations,
            game_tasks: GameTaskRegistryData {
                tasks_by_owner,
                active_task_id,
            },
            actor_statistics,
            trailing_bytes,
        })
    }

    pub fn to_bytes(&self, alife_version: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);

        // 1. m_info_portions
        out.extend_from_slice(&(self.info_portions.len() as u32).to_le_bytes());
        for (id, list) in &self.info_portions {
            out.extend_from_slice(&id.to_le_bytes());
            write_str_vec(&mut out, list);
        }

        // 2. m_character_relations
        out.extend_from_slice(&(self.character_relations.len() as u32).to_le_bytes());
        for (id, rel) in &self.character_relations {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(rel.personal.len() as u32).to_le_bytes());
            for (target, goodwill) in &rel.personal {
                out.extend_from_slice(&target.to_le_bytes());
                out.extend_from_slice(&goodwill.to_le_bytes());
            }
            out.extend_from_slice(&(rel.communities.len() as u32).to_le_bytes());
            for (comm, goodwill) in &rel.communities {
                out.extend_from_slice(&comm.to_le_bytes());
                out.extend_from_slice(&goodwill.to_le_bytes());
            }
        }

        // 3. m_game_news
        out.extend_from_slice(&(self.game_news.len() as u32).to_le_bytes());
        for (id, items) in &self.game_news {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(items.len() as u32).to_le_bytes());
            for item in items {
                out.extend_from_slice(&item.news_type.to_le_bytes());
                write_zstring(&mut out, &item.caption);
                write_zstring(&mut out, &item.text);
                out.extend_from_slice(&item.receive_time.to_le_bytes());
                write_zstring(&mut out, &item.texture_name);
            }
        }

        // 4. m_specific_characters
        out.extend_from_slice(&(self.specific_characters.len() as u32).to_le_bytes());
        for (name, val) in &self.specific_characters {
            write_zstring(&mut out, name);
            out.extend_from_slice(&val.to_le_bytes());
        }

        // 5. m_map_locations
        out.extend_from_slice(&(self.map_locations.len() as u32).to_le_bytes());
        for (id, locs) in &self.map_locations {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(locs.len() as u32).to_le_bytes());
            for loc in locs {
                out.extend_from_slice(&loc.object_id.to_le_bytes());
                write_zstring(&mut out, &loc.spot_type);
                out.push(loc.reserved_flag);
                write_zstring(&mut out, &loc.hint);
                out.extend_from_slice(&loc.flags.to_le_bytes());
                write_zstring(&mut out, &loc.owner_task_id);
            }
        }

        // 6. m_game_tasks
        out.extend_from_slice(&(self.game_tasks.tasks_by_owner.len() as u32).to_le_bytes());
        for (id, tasks) in &self.game_tasks.tasks_by_owner {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(tasks.len() as u32).to_le_bytes());
            for t in tasks {
                write_zstring(&mut out, &t.task_id);
                out.extend_from_slice(&t.task_state.to_le_bytes());
                out.extend_from_slice(&t.task_type.to_le_bytes());
                out.extend_from_slice(&t.receive_time.to_le_bytes());
                out.extend_from_slice(&t.finish_time.to_le_bytes());
                out.extend_from_slice(&t.time_to_complete.to_le_bytes());
                out.extend_from_slice(&t.timer_finish.to_le_bytes());
                write_zstring(&mut out, &t.title);
                write_zstring(&mut out, &t.description);
                write_str_vec(&mut out, &t.script_helper.complete_lua_functions);
                write_str_vec(&mut out, &t.script_helper.fail_lua_functions);
                write_str_vec(&mut out, &t.script_helper.lua_functions_on_complete);
                write_str_vec(&mut out, &t.script_helper.lua_functions_on_fail);
                write_zstring(&mut out, &t.icon_texture_name);
                write_zstring(&mut out, &t.map_hint);
                write_zstring(&mut out, &t.map_location);
                out.extend_from_slice(&t.map_object_id.to_le_bytes());
                out.extend_from_slice(&t.priority.to_le_bytes());
            }
        }
        write_zstring(&mut out, &self.game_tasks.active_task_id);

        // 7. m_actor_statistics
        out.extend_from_slice(&(self.actor_statistics.len() as u32).to_le_bytes());
        for (id, sections) in &self.actor_statistics {
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
            for sec in sections {
                out.extend_from_slice(&(sec.details.len() as u32).to_le_bytes());
                for d in &sec.details {
                    write_zstring(&mut out, &d.key);
                    out.extend_from_slice(&d.int_count.to_le_bytes());
                    out.extend_from_slice(&d.int_points.to_le_bytes());
                    if alife_version > 0x0002 {
                        write_zstring(&mut out, &d.str_value);
                    }
                }
                write_zstring(&mut out, &sec.key);
            }
        }

        out.extend_from_slice(&self.trailing_bytes);
        out
    }
}

/// Complete parsed `.scop` savegame file.
#[derive(Debug, Clone, PartialEq)]
pub struct SavegameFile {
    pub alife_version: u32,
    pub time: SaveGameTime,
    pub spawn_info: SaveSpawnHeader,
    pub objects: Vec<SaveObjectRecord>,
    pub registries: SaveRegistries,
    /// Additional chunks preserved verbatim for round-trip compatibility.
    pub extra_chunks: Vec<(u32, Vec<u8>)>,
}

impl SavegameFile {
    /// Decompress and parse a full `.scop` savegame buffer.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let (_hdr_version, inner) = decompress_scop(data)?;
        Self::parse_inner(&inner)
    }

    /// Parse an already-decompressed ALife save stream.
    pub fn parse_inner(inner: &[u8]) -> Result<Self, FormatError> {
        let root = ChunkReader::new(inner);

        let alife_chunk = root
            .open_chunk(ALIFE_CHUNK_DATA)?
            .ok_or(FormatError::MissingChunk(ALIFE_CHUNK_DATA))?;
        let mut alife_cur = ByteCursor::new(&alife_chunk);
        let alife_version = alife_cur.read_u32()?;
        if alife_version < ALIFE_VERSION {
            return Err(FormatError::InvalidValue("unsupported inner .scop ALife version"));
        }

        let time_chunk = root
            .open_chunk(GAME_TIME_CHUNK_DATA)?
            .ok_or(FormatError::MissingChunk(GAME_TIME_CHUNK_DATA))?;
        let time = SaveGameTime::parse(&time_chunk)?;

        let spawn_chunk = root
            .open_chunk(SPAWN_CHUNK_DATA)?
            .ok_or(FormatError::MissingChunk(SPAWN_CHUNK_DATA))?;
        let spawn_info = SaveSpawnHeader::parse(&spawn_chunk)?;

        let objects_chunk = root
            .open_chunk(OBJECT_CHUNK_DATA)?
            .ok_or(FormatError::MissingChunk(OBJECT_CHUNK_DATA))?;
        let objects = SaveObjectRecord::parse_list(&objects_chunk)?;

        let registry_chunk = root
            .open_chunk(REGISTRY_CHUNK_DATA)?
            .ok_or(FormatError::MissingChunk(REGISTRY_CHUNK_DATA))?;
        let registries = SaveRegistries::parse(&registry_chunk, alife_version)?;

        let mut extra_chunks = Vec::new();
        for entry in root.iter() {
            let entry = entry?;
            if !matches!(
                entry.id,
                ALIFE_CHUNK_DATA
                    | SPAWN_CHUNK_DATA
                    | OBJECT_CHUNK_DATA
                    | GAME_TIME_CHUNK_DATA
                    | REGISTRY_CHUNK_DATA
            ) {
                extra_chunks.push((entry.id, entry.decode()?.into_owned()));
            }
        }

        Ok(Self {
            alife_version,
            time,
            spawn_info,
            objects,
            registries,
            extra_chunks,
        })
    }

    /// Encode the uncompressed ALife stream (`CALifeStorageManager::save` chunk order).
    pub fn to_inner_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let mut root = ChunkWriter::new();
        root.write_chunk(ALIFE_CHUNK_DATA, &self.alife_version.to_le_bytes());
        root.write_chunk(GAME_TIME_CHUNK_DATA, &self.time.to_bytes());
        root.write_chunk(SPAWN_CHUNK_DATA, &self.spawn_info.to_bytes());
        root.write_chunk(OBJECT_CHUNK_DATA, &SaveObjectRecord::encode_list(&self.objects)?);
        root.write_chunk(
            REGISTRY_CHUNK_DATA,
            &self.registries.to_bytes(self.alife_version),
        );
        for (id, payload) in &self.extra_chunks {
            root.write_chunk(*id, payload);
        }
        Ok(root.into_bytes())
    }

    /// Compress and serialize into an `.scop` file.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let inner = self.to_inner_bytes()?;
        Ok(compress_scop(self.alife_version, &inner))
    }

    /// Return the actor (`ID == 0`) object record, if present.
    pub fn actor(&self) -> Option<&SaveObjectRecord> {
        self.objects.iter().find(|o| o.spawn.id == 0)
    }

    /// Convenience summary helper matching `CSavedGameWrapper` fast save-slot metadata.
    pub fn actor_health(&self) -> Option<f32> {
        self.actor().and_then(SaveObjectRecord::creature_update_health)
    }

    /// Look up an ALife object by entity ID.
    pub fn find_object(&self, id: u16) -> Option<&SaveObjectRecord> {
        self.objects.iter().find(|o| o.spawn.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spawns::{cse_spawn_flags, m_spawn_flags, AlifeObjectState, SPAWN_VERSION};

    #[test]
    fn test_date_time_gregorian_leap_years_roundtrip() {
        let cases = [
            GameDateTime {
                year: 1,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
                millisecond: 0,
            },
            GameDateTime {
                year: 1600,
                month: 2,
                day: 29,
                hour: 12,
                minute: 34,
                second: 56,
                millisecond: 789,
            },
            GameDateTime {
                year: 1900,
                month: 12,
                day: 31,
                hour: 23,
                minute: 59,
                second: 59,
                millisecond: 999,
            },
            GameDateTime {
                year: 2012,
                month: 2,
                day: 29,
                hour: 5,
                minute: 30,
                second: 1,
                millisecond: 42,
            },
            GameDateTime {
                year: 2018,
                month: 6,
                day: 15,
                hour: 8,
                minute: 15,
                second: 0,
                millisecond: 0,
            },
        ];

        for dt in cases {
            let packed = dt.to_game_time_ms();
            let unpacked = GameDateTime::from_game_time_ms(packed);
            assert_eq!(unpacked, dt);
        }

        // Verify 1900 was not a leap year (Mar 1 follows Feb 28 directly -> 1 day delta)
        let feb28 = GameDateTime {
            year: 1900,
            month: 2,
            day: 28,
            ..Default::default()
        }
        .to_game_time_ms();
        let mar1 = GameDateTime {
            year: 1900,
            month: 3,
            day: 1,
            ..Default::default()
        }
        .to_game_time_ms();
        assert_eq!(mar1 - feb28, 24 * 60 * 60 * 1000);

        let ini_dt = GameDateTime::from_ini_strings("09:00:00", "01.05.2012").unwrap();
        assert_eq!(ini_dt.year, 2012);
        assert_eq!(ini_dt.month, 5);
        assert_eq!(ini_dt.day, 1);
        assert_eq!(ini_dt.hour, 9);
    }

    #[test]
    fn test_scop_savegame_roundtrip_all_registries() {
        let actor_state = AlifeObjectState {
            game_vertex_id: 12,
            distance: 3.5,
            direct_control: true,
            level_vertex_id: 4096,
            flags: 0x0000_00BF,
            custom_data: "[actor]\ngod=false".to_string(),
            story_id: 0,
            spawn_story_id: u32::MAX,
            subclass_payload: vec![1, 2, 3, 4],
        };
        let actor_spawn = SpawnPacket {
            section: "actor".to_string(),
            name_replace: "actor".to_string(),
            game_id_legacy: 0,
            rp: 0xFE,
            position: [10.0, 2.0, -30.0],
            angle: [0.0, 1.5, 0.0],
            respawn_time: 0,
            id: 0,
            parent_id: u16::MAX,
            phantom_id: u16::MAX,
            flags: m_spawn_flags::OBJECT_LOCAL | m_spawn_flags::VERSION,
            version: SPAWN_VERSION,
            game_type: 1,
            script_version: 12,
            client_data: Vec::new(),
            spawn_id: 0,
            state_payload: actor_state.to_bytes(),
            trailing_update_payload: Vec::new(),
        };

        let mut actor_update = Vec::new();
        actor_update.extend_from_slice(&0.875f32.to_le_bytes()); // CSE_ALifeCreatureAbstract::fHealth
        actor_update.extend_from_slice(&[0xAA, 0xBB, 0xCC]);

        let save = SavegameFile {
            alife_version: ALIFE_VERSION,
            time: SaveGameTime {
                game_time_ms: GameDateTime {
                    year: 2012,
                    month: 8,
                    day: 14,
                    hour: 13,
                    minute: 45,
                    second: 10,
                    millisecond: 250,
                }
                .to_game_time_ms(),
                time_factor: 6.0,
                normal_time_factor: 10.0,
            },
            spawn_info: SaveSpawnHeader {
                spawn_name: "all".to_string(),
                spawn_guid: [0x42; 16],
                level_id: Some(3),
                level_name: Some("l01_escape".to_string()),
                spawn_updates: vec![(0, Vec::new()), (1, vec![7, 8])],
            },
            objects: vec![SaveObjectRecord {
                spawn: actor_spawn,
                update_msg_id: M_UPDATE,
                update_payload: actor_update,
            }],
            registries: SaveRegistries {
                info_portions: vec![(0, vec!["esc_trader_met".to_string(), "pda_ tutorial".to_string()])],
                character_relations: vec![(
                    0,
                    RelationData {
                        personal: vec![(42, 250)],
                        communities: vec![(1, 500), (2, -1000)],
                    },
                )],
                game_news: vec![(
                    0,
                    vec![GameNewsEntry {
                        news_type: 0,
                        caption: "st_tip".to_string(),
                        text: "st_hello_zone".to_string(),
                        receive_time: 123456,
                        texture_name: "ui_iconsTotal_found_thing".to_string(),
                    }],
                )],
                specific_characters: vec![("actor_profile".to_string(), 1)],
                map_locations: vec![(
                    0,
                    vec![SavedMapLocation {
                        object_id: 99,
                        spot_type: "green_location".to_string(),
                        reserved_flag: 0,
                        hint: "sidorovich".to_string(),
                        flags: 1,
                        owner_task_id: "esc_task_1".to_string(),
                    }],
                )],
                game_tasks: GameTaskRegistryData {
                    tasks_by_owner: vec![(
                        0,
                        vec![SavedGameTask {
                            task_id: "esc_task_1".to_string(),
                            task_state: 1,
                            task_type: 0,
                            receive_time: 1000,
                            finish_time: 0,
                            time_to_complete: 86400000,
                            timer_finish: 0,
                            title: "Meet Sidorovich".to_string(),
                            description: "Visit the bunker.".to_string(),
                            script_helper: SavedTaskScriptHelper {
                                complete_lua_functions: vec!["task_done".to_string()],
                                ..Default::default()
                            },
                            icon_texture_name: "ui_inGame2_PDA".to_string(),
                            map_hint: "Bunker".to_string(),
                            map_location: "green_location".to_string(),
                            map_object_id: 99,
                            priority: 10,
                        }],
                    )],
                    active_task_id: "esc_task_1".to_string(),
                },
                actor_statistics: vec![(
                    0,
                    vec![ActorStatSection {
                        details: vec![ActorStatDetail {
                            key: "bandits".to_string(),
                            int_count: 5,
                            int_points: 50,
                            str_value: String::new(),
                        }],
                        key: "stalkerkills".to_string(),
                    }],
                )],
                trailing_bytes: vec![0xDE, 0xAD],
            },
            extra_chunks: vec![(99, vec![1, 2, 3])],
        };

        let _ = cse_spawn_flags::ENABLED;
        let bytes = save.to_bytes().expect("encode .scop");
        assert_eq!(validate_scop_header(&bytes).map(|v| v.0), Some(ALIFE_VERSION));

        let parsed = SavegameFile::parse(&bytes).expect("parse .scop");
        assert_eq!(parsed, save);
        assert_eq!(parsed.actor_health(), Some(0.875));
        assert_eq!(parsed.time.date_time().year, 2012);
        assert_eq!(parsed.time.date_time().hour, 13);
        assert_eq!(parsed.spawn_info.level_name.as_deref(), Some("l01_escape"));
        assert_eq!(
            parsed.registries.game_tasks.active_task_id,
            "esc_task_1"
        );
    }
}
