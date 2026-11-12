//! X-Ray Monolith / Call of Pripyat / S.T.A.L.K.E.R. Anomaly `.scop` Host ALife & World Savegame Codec
//!
//! Pairs with [`crate::scoc`] (peer script/Lua persistence) to load and serialize
//! the authoritative host ALife simulation & world state (`ScopSaveFile`), supporting:
//! - Outer LZO1X compressed container (`SAVE_MAGIC = 0xFFFF_FFFF`, `alife_version >= 0x0006`,
//!   `uncompressed_size`, `LZO1X` payload) via [`crate::savegames`]
//! - **Chunk 0 (`SCOP_CHUNK_HEADER`)**: Save header (`alife_version`, format version,
//!   spawn name/GUID, level ID/name, `SaveGameTime`, and `ScopActorStats`)
//! - **Chunk 1 (`SCOP_CHUNK_ALIFE_OBJECTS`)**: Typed ALife simulation objects
//!   (`ScopAlifeEntity` / `ScopAlifePayload` covering `Actor`, `Stalker`, `Mutant`,
//!   `SmartTerrain`, `PhysicsObject`, `SpaceRestrictor`, `InventoryBox`, `LevelChanger`,
//!   `AnomalousZone`, `SmartCover`, `OnlineOfflineGroup`, `Trader`, `Weapon`,
//!   `InventoryItem`, and fallback `CSE_ALifeDynamicObject`)
//! - **Chunk 2 (`SCOP_CHUNK_GRAPH_QUESTS`)**: Game graph vertex/level mappings
//!   (`ScopGraphMapping`), story quests (`ScopStoryQuest`), active task & `info_portions`,
//!   and full `SaveRegistries` compatibility
//! - **Chunk 3 (`SCOP_CHUNK_WEATHER_ENV`)**: Environment weather cycle, FX, rain/wetness,
//!   wind, surge/psi-storm, and atmospheric state (`ScopWeatherState`)
//!
//! Also seamlessly auto-detects and converts canonical `CALifeStorageManager` `.scop`
//! archives (`0x0000` / `0x0001` / `0x0002` / `0x0005` / `0x0009` chunks from [`crate::savegames`]).

use std::collections::BTreeMap;

use crate::ai_graph::{INVALID_AI_VERTEX, INVALID_GAME_VERTEX};
use crate::chunk::{ByteCursor, ChunkReader, ChunkWriter, FormatError};
use crate::savegames::{
    compress_scop, decompress_scop, ActorStatSection, GameDateTime, GameTaskRegistryData,
    SaveGameTime, SaveObjectRecord, SaveRegistries, SaveSpawnHeader, SavedGameTask, SavegameFile,
    ALIFE_VERSION, SAVE_EXTENSION, SAVE_MAGIC,
};
use crate::spawns::{
    alife_object_flags, m_spawn_flags, AlifeAnomalousZoneState, AlifeCreatureActorState,
    AlifeHumanStalkerState, AlifeInventoryBoxState, AlifeInventoryItemState,
    AlifeItemWeaponState, AlifeLevelChangerState, AlifeMonsterBaseState, AlifeObjectPhysicState,
    AlifeObjectState, AlifeOnlineOfflineGroupState, AlifeSmartCoverState, AlifeSmartZoneState,
    AlifeSpaceRestrictorState, AlifeTraderState, ServerVisualState, SpawnPacket, INVALID_OBJECT_ID,
    INVALID_SPAWN_STORY_ID, INVALID_STORY_ID, M_UPDATE, SPAWN_VERSION,
};
use crate::weather::{MixedEnvironment, WetnessState, DAY_LENGTH};

pub const SCOP_EXTENSION: &str = SAVE_EXTENSION;
pub const SCOP_MAGIC: u32 = SAVE_MAGIC;
pub const SCOP_ALIFE_VERSION: u32 = ALIFE_VERSION;
pub const SCOP_FORMAT_VERSION: u32 = 1;

/// Magic signature inside structured Chunk 0 distinguishing `ScopSaveFile` Chunk 0..3
/// from canonical `CALifeStorageManager` Chunk `0x0000` (which is 4 bytes `u32 alife_version`).
pub const SCOP_HEADER_SIGNATURE: [u8; 4] = *b"SCOP";

/// Chunk 0: Save header (`alife_version`, level metadata, game time, actor stats).
pub const SCOP_CHUNK_HEADER: u32 = 0;
/// Chunk 1: Typed ALife dynamic objects (`ScopAlifeEntity` list).
pub const SCOP_CHUNK_ALIFE_OBJECTS: u32 = 1;
/// Chunk 2: Game graph vertex mappings, story quests, active task, info portions & registries.
pub const SCOP_CHUNK_GRAPH_QUESTS: u32 = 2;
/// Chunk 3: Environment & atmospheric weather state (`ScopWeatherState`).
pub const SCOP_CHUNK_WEATHER_ENV: u32 = 3;

/// Optional weather chunk ID when appending weather state alongside canonical `CALifeStorageManager` chunks.
pub const SCOP_CANONICAL_WEATHER_CHUNK: u32 = 0x000A;

const MAX_SCOP_OBJECTS: usize = 1_000_000;
const MAX_SCOP_STRINGS: usize = 1_000_000;

fn write_zstring(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(s.as_bytes());
    out.push(0);
}

fn write_f32_3(out: &mut Vec<u8>, v: [f32; 3]) {
    out.extend_from_slice(&v[0].to_le_bytes());
    out.extend_from_slice(&v[1].to_le_bytes());
    out.extend_from_slice(&v[2].to_le_bytes());
}

fn write_u16_vec(out: &mut Vec<u8>, items: &[u16]) {
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for &id in items {
        out.extend_from_slice(&id.to_le_bytes());
    }
}

fn read_u16_vec(cur: &mut ByteCursor<'_>) -> Result<Vec<u16>, FormatError> {
    let count = cur.read_u32()? as usize;
    if count > MAX_SCOP_OBJECTS {
        return Err(FormatError::InvalidValue(
            "u16 list count exceeds sanity bound in .scop",
        ));
    }
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(cur.read_u16()?);
    }
    Ok(out)
}

fn write_bytes_vec(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn read_bytes_vec(cur: &mut ByteCursor<'_>) -> Result<Vec<u8>, FormatError> {
    let len = cur.read_u32()? as usize;
    if len > 16 * 1024 * 1024 {
        return Err(FormatError::InvalidValue(
            "byte blob length exceeds sanity bound in .scop",
        ));
    }
    Ok(cur.read_bytes(len)?.to_vec())
}

// ============================================================================
// Chunk 0: Savegame Header & Actor Stats
// ============================================================================

/// Summary of the player actor saved inside `.scop` Chunk 0 (`ScopHeader::actor_stats`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScopActorStats {
    pub actor_name: String,
    pub profile_name: String,
    pub community_index: i32,
    pub health: f32,
    pub radiation: f32,
    pub stamina: f32,
    pub bleeding: f32,
    pub money: u32,
    pub rank: i32,
    pub reputation: i32,
    pub position: [f32; 3],
    pub angles: [f32; 3],
    pub game_vertex_id: u16,
    pub level_vertex_id: u32,
    pub holder_id: u16,
    pub statistics: Vec<ActorStatSection>,
}

impl Default for ScopActorStats {
    fn default() -> Self {
        Self {
            actor_name: "Marked One".to_string(),
            profile_name: "actor".to_string(),
            community_index: 0,
            health: 1.0,
            radiation: 0.0,
            stamina: 1.0,
            bleeding: 0.0,
            money: 0,
            rank: 0,
            reputation: 0,
            position: [0.0; 3],
            angles: [0.0; 3],
            game_vertex_id: INVALID_GAME_VERTEX,
            level_vertex_id: INVALID_AI_VERTEX,
            holder_id: INVALID_OBJECT_ID,
            statistics: Vec::new(),
        }
    }
}

impl ScopActorStats {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        Self::parse_from_cursor(&mut cur)
    }

    pub fn parse_from_cursor(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let actor_name = cur.read_zstring()?;
        let profile_name = cur.read_zstring()?;
        let community_index = cur.read_i32()?;
        let health = cur.read_f32()?;
        let radiation = cur.read_f32()?;
        let stamina = cur.read_f32()?;
        let bleeding = cur.read_f32()?;
        let money = cur.read_u32()?;
        let rank = cur.read_i32()?;
        let reputation = cur.read_i32()?;
        let position = cur.read_f32_3()?;
        let angles = cur.read_f32_3()?;
        let game_vertex_id = cur.read_u16()?;
        let level_vertex_id = cur.read_u32()?;
        let holder_id = cur.read_u16()?;

        let sec_count = cur.read_u32()? as usize;
        if sec_count > MAX_SCOP_STRINGS {
            return Err(FormatError::InvalidValue(
                "actor statistic section count exceeds sanity bound",
            ));
        }
        let mut statistics = Vec::with_capacity(sec_count);
        for _ in 0..sec_count {
            let key = cur.read_zstring()?;
            let det_count = cur.read_u32()? as usize;
            if det_count > MAX_SCOP_STRINGS {
                return Err(FormatError::InvalidValue(
                    "actor statistic detail count exceeds sanity bound",
                ));
            }
            let mut details = Vec::with_capacity(det_count);
            for _ in 0..det_count {
                details.push(crate::savegames::ActorStatDetail {
                    key: cur.read_zstring()?,
                    int_count: cur.read_i32()?,
                    int_points: cur.read_i32()?,
                    str_value: cur.read_zstring()?,
                });
            }
            statistics.push(ActorStatSection { details, key });
        }

        Ok(Self {
            actor_name,
            profile_name,
            community_index,
            health,
            radiation,
            stamina,
            bleeding,
            money,
            rank,
            reputation,
            position,
            angles,
            game_vertex_id,
            level_vertex_id,
            holder_id,
            statistics,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        write_zstring(out, &self.actor_name);
        write_zstring(out, &self.profile_name);
        out.extend_from_slice(&self.community_index.to_le_bytes());
        out.extend_from_slice(&self.health.to_le_bytes());
        out.extend_from_slice(&self.radiation.to_le_bytes());
        out.extend_from_slice(&self.stamina.to_le_bytes());
        out.extend_from_slice(&self.bleeding.to_le_bytes());
        out.extend_from_slice(&self.money.to_le_bytes());
        out.extend_from_slice(&self.rank.to_le_bytes());
        out.extend_from_slice(&self.reputation.to_le_bytes());
        write_f32_3(out, self.position);
        write_f32_3(out, self.angles);
        out.extend_from_slice(&self.game_vertex_id.to_le_bytes());
        out.extend_from_slice(&self.level_vertex_id.to_le_bytes());
        out.extend_from_slice(&self.holder_id.to_le_bytes());

        out.extend_from_slice(&(self.statistics.len() as u32).to_le_bytes());
        for sec in &self.statistics {
            write_zstring(out, &sec.key);
            out.extend_from_slice(&(sec.details.len() as u32).to_le_bytes());
            for d in &sec.details {
                write_zstring(out, &d.key);
                out.extend_from_slice(&d.int_count.to_le_bytes());
                out.extend_from_slice(&d.int_points.to_le_bytes());
                write_zstring(out, &d.str_value);
            }
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128);
        self.write_to(&mut out);
        out
    }

    /// Look up the integer count for a named statistic entry (e.g. `"stalkerkills"`).
    pub fn find_stat_count(&self, section_key: &str, detail_key: &str) -> Option<i32> {
        self.statistics
            .iter()
            .find(|s| s.key == section_key)
            .and_then(|s| s.details.iter().find(|d| d.key == detail_key))
            .map(|d| d.int_count)
    }
}

/// Chunk 0 header for a `.scop` ALife world savegame.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopHeader {
    pub alife_version: u32,
    pub format_version: u32,
    pub spawn_name: String,
    pub spawn_guid: [u8; 16],
    pub level_id: Option<u8>,
    pub level_name: String,
    pub game_time: SaveGameTime,
    pub actor_stats: ScopActorStats,
}

impl Default for ScopHeader {
    fn default() -> Self {
        Self {
            alife_version: SCOP_ALIFE_VERSION,
            format_version: SCOP_FORMAT_VERSION,
            spawn_name: "all".to_string(),
            spawn_guid: [0u8; 16],
            level_id: Some(0),
            level_name: "l01_escape".to_string(),
            game_time: SaveGameTime::default(),
            actor_stats: ScopActorStats::default(),
        }
    }
}

impl ScopHeader {
    pub fn date_time(&self) -> GameDateTime {
        self.game_time.date_time()
    }

    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let sig = cur.read_bytes(4)?;
        if sig != SCOP_HEADER_SIGNATURE {
            return Err(FormatError::InvalidValue("invalid structured .scop header signature"));
        }
        let format_version = cur.read_u32()?;
        if format_version == 0 || format_version > SCOP_FORMAT_VERSION {
            return Err(FormatError::InvalidValue("unsupported .scop Chunk 0 format version"));
        }
        let alife_version = cur.read_u32()?;
        if alife_version < ALIFE_VERSION {
            return Err(FormatError::InvalidValue("unsupported .scop ALife version"));
        }
        let spawn_name = cur.read_zstring()?;
        let guid_bytes = cur.read_bytes(16)?;
        let mut spawn_guid = [0u8; 16];
        spawn_guid.copy_from_slice(guid_bytes);

        let has_level_id = cur.read_u8()? != 0;
        let level_id = if has_level_id {
            Some(cur.read_u8()?)
        } else {
            None
        };
        let level_name = cur.read_zstring()?;

        let time_bytes = cur.read_bytes(16)?;
        let game_time = SaveGameTime::parse(time_bytes)?;
        let actor_stats = ScopActorStats::parse_from_cursor(&mut cur)?;

        Ok(Self {
            alife_version,
            format_version,
            spawn_name,
            spawn_guid,
            level_id,
            level_name,
            game_time,
            actor_stats,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(192);
        out.extend_from_slice(&SCOP_HEADER_SIGNATURE);
        out.extend_from_slice(&self.format_version.to_le_bytes());
        out.extend_from_slice(&self.alife_version.to_le_bytes());
        write_zstring(&mut out, &self.spawn_name);
        out.extend_from_slice(&self.spawn_guid);
        match self.level_id {
            Some(id) => {
                out.push(1);
                out.push(id);
            }
            None => out.push(0),
        }
        write_zstring(&mut out, &self.level_name);
        out.extend_from_slice(&self.game_time.to_bytes());
        self.actor_stats.write_to(&mut out);
        out
    }
}

// ============================================================================
// Chunk 1: ALife Dynamic Objects & Typed Entity Subclasses
// ============================================================================

/// High-level ALife entity classification stored in `.scop` Chunk 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScopAlifeKind {
    Actor = 0,
    Stalker = 1,
    Mutant = 2,
    SmartTerrain = 3,
    PhysicsObject = 4,
    SpaceRestrictor = 5,
    InventoryBox = 6,
    LevelChanger = 7,
    AnomalousZone = 8,
    SmartCover = 9,
    OnlineOfflineGroup = 10,
    Trader = 11,
    Weapon = 12,
    InventoryItem = 13,
    DynamicObject = 14,
}

impl ScopAlifeKind {
    pub fn from_u8(v: u8) -> Result<Self, FormatError> {
        match v {
            0 => Ok(Self::Actor),
            1 => Ok(Self::Stalker),
            2 => Ok(Self::Mutant),
            3 => Ok(Self::SmartTerrain),
            4 => Ok(Self::PhysicsObject),
            5 => Ok(Self::SpaceRestrictor),
            6 => Ok(Self::InventoryBox),
            7 => Ok(Self::LevelChanger),
            8 => Ok(Self::AnomalousZone),
            9 => Ok(Self::SmartCover),
            10 => Ok(Self::OnlineOfflineGroup),
            11 => Ok(Self::Trader),
            12 => Ok(Self::Weapon),
            13 => Ok(Self::InventoryItem),
            14 => Ok(Self::DynamicObject),
            _ => Err(FormatError::InvalidValue("unknown .scop ALife entity kind tag")),
        }
    }

    #[inline]
    pub fn is_physics_or_restrictor(self) -> bool {
        matches!(
            self,
            Self::PhysicsObject | Self::SpaceRestrictor | Self::LevelChanger | Self::AnomalousZone
        )
    }
}

/// Typed `CSE_ALifeSmartZone` / Anomaly `se_smart_terrain` simulation state.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopSmartTerrainState {
    pub zone: AlifeSmartZoneState,
    pub smart_name: String,
    pub max_population: u16,
    pub squad_id: u16,
    pub staying_time_ms: u64,
    pub arriving_npc_ids: Vec<u16>,
    pub assigned_npc_ids: Vec<u16>,
    pub script_state: Vec<u8>,
}

impl ScopSmartTerrainState {
    pub fn new(smart_name: impl Into<String>, zone: AlifeSmartZoneState) -> Self {
        let smart_name = smart_name.into();
        let max_population = parse_custom_data_u16(&zone.alife.custom_data, "max_population")
            .unwrap_or(0);
        let squad_id = parse_custom_data_u16(&zone.alife.custom_data, "squad_id").unwrap_or(0);
        let script_state = zone.trailing.clone();
        Self {
            zone,
            smart_name,
            max_population,
            squad_id,
            staying_time_ms: 0,
            arriving_npc_ids: Vec::new(),
            assigned_npc_ids: Vec::new(),
            script_state,
        }
    }

    pub fn parse_ext(cur: &mut ByteCursor<'_>, smart_name: String, zone: AlifeSmartZoneState) -> Result<Self, FormatError> {
        let max_population = cur.read_u16()?;
        let squad_id = cur.read_u16()?;
        let staying_time_ms = cur.read_u64()?;
        let arriving_npc_ids = read_u16_vec(cur)?;
        let assigned_npc_ids = read_u16_vec(cur)?;
        let script_state = read_bytes_vec(cur)?;
        Ok(Self {
            zone,
            smart_name,
            max_population,
            squad_id,
            staying_time_ms,
            arriving_npc_ids,
            assigned_npc_ids,
            script_state,
        })
    }

    pub fn write_ext(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.max_population.to_le_bytes());
        out.extend_from_slice(&self.squad_id.to_le_bytes());
        out.extend_from_slice(&self.staying_time_ms.to_le_bytes());
        write_u16_vec(out, &self.arriving_npc_ids);
        write_u16_vec(out, &self.assigned_npc_ids);
        write_bytes_vec(out, &self.script_state);
    }

    #[inline]
    pub fn current_population(&self) -> usize {
        self.assigned_npc_ids.len() + self.arriving_npc_ids.len()
    }
}

/// Decoded typed state for a `.scop` ALife dynamic object (`CSE_ALifeDynamicObject` hierarchy).
#[derive(Debug, Clone, PartialEq)]
pub enum ScopAlifePayload {
    Actor(AlifeCreatureActorState),
    Stalker(AlifeHumanStalkerState),
    Mutant(AlifeMonsterBaseState),
    SmartTerrain(ScopSmartTerrainState),
    PhysicsObject(AlifeObjectPhysicState),
    SpaceRestrictor(AlifeSpaceRestrictorState),
    InventoryBox(AlifeInventoryBoxState),
    LevelChanger(AlifeLevelChangerState),
    AnomalousZone(AlifeAnomalousZoneState),
    SmartCover(AlifeSmartCoverState),
    OnlineOfflineGroup(AlifeOnlineOfflineGroupState),
    Trader(AlifeTraderState),
    Weapon(AlifeItemWeaponState),
    InventoryItem(AlifeInventoryItemState),
    DynamicObject(AlifeObjectState),
}

impl ScopAlifePayload {
    pub fn kind(&self) -> ScopAlifeKind {
        match self {
            Self::Actor(_) => ScopAlifeKind::Actor,
            Self::Stalker(_) => ScopAlifeKind::Stalker,
            Self::Mutant(_) => ScopAlifeKind::Mutant,
            Self::SmartTerrain(_) => ScopAlifeKind::SmartTerrain,
            Self::PhysicsObject(_) => ScopAlifeKind::PhysicsObject,
            Self::SpaceRestrictor(_) => ScopAlifeKind::SpaceRestrictor,
            Self::InventoryBox(_) => ScopAlifeKind::InventoryBox,
            Self::LevelChanger(_) => ScopAlifeKind::LevelChanger,
            Self::AnomalousZone(_) => ScopAlifeKind::AnomalousZone,
            Self::SmartCover(_) => ScopAlifeKind::SmartCover,
            Self::OnlineOfflineGroup(_) => ScopAlifeKind::OnlineOfflineGroup,
            Self::Trader(_) => ScopAlifeKind::Trader,
            Self::Weapon(_) => ScopAlifeKind::Weapon,
            Self::InventoryItem(_) => ScopAlifeKind::InventoryItem,
            Self::DynamicObject(_) => ScopAlifeKind::DynamicObject,
        }
    }

    /// Encode this typed payload back into `SPAWN_VERSION` (128) `M_SPAWN` state bytes.
    pub fn to_state_bytes(&self) -> Vec<u8> {
        match self {
            Self::Actor(s) => s.to_bytes(),
            Self::Stalker(s) => s.to_bytes(),
            Self::Mutant(s) => s.to_bytes(),
            Self::SmartTerrain(s) => s.zone.to_bytes(),
            Self::PhysicsObject(s) => s.to_bytes(),
            Self::SpaceRestrictor(s) => s.to_bytes(),
            Self::InventoryBox(s) => s.to_bytes(),
            Self::LevelChanger(s) => s.to_bytes(),
            Self::AnomalousZone(s) => s.to_bytes(),
            Self::SmartCover(s) => s.to_bytes(),
            Self::OnlineOfflineGroup(s) => s.to_bytes(),
            Self::Trader(s) => s.to_bytes(),
            Self::Weapon(s) => s.to_bytes(),
            Self::InventoryItem(s) => s.to_bytes(),
            Self::DynamicObject(s) => s.to_bytes(),
        }
    }

    /// Borrow the base `CSE_ALifeObject` header fields when available inline.
    pub fn alife_object(&self) -> &AlifeObjectState {
        match self {
            Self::Actor(s) => &s.creature.base.alife,
            Self::Stalker(s) => &s.human.monster.creature.base.alife,
            Self::Mutant(s) => &s.monster.creature.base.alife,
            Self::SmartTerrain(s) => &s.zone.alife,
            Self::PhysicsObject(s) => &s.base.alife,
            Self::SpaceRestrictor(s) => &s.alife,
            Self::InventoryBox(s) => &s.base.alife,
            Self::LevelChanger(s) => &s.restrictor.alife,
            Self::AnomalousZone(s) => &s.zone.restrictor.alife,
            Self::SmartCover(s) => &s.alife,
            Self::OnlineOfflineGroup(s) => &s.alife,
            Self::Trader(s) => &s.base.alife,
            Self::Weapon(s) => &s.item.base.alife,
            Self::InventoryItem(s) => &s.base.alife,
            Self::DynamicObject(s) => s,
        }
    }

    /// Return the creature/breakable health encoded in the `M_SPAWN` state payload, if any.
    pub fn state_health(&self) -> Option<f32> {
        match self {
            Self::Actor(s) => Some(s.creature.health),
            Self::Stalker(s) => Some(s.human.monster.creature.health),
            Self::Mutant(s) => Some(s.monster.creature.health),
            _ => None,
        }
    }

    /// Return the visual model path (`ServerVisualState`), if this subclass carries a visual.
    pub fn visual(&self) -> Option<&ServerVisualState> {
        match self {
            Self::Actor(s) => Some(&s.creature.base.visual),
            Self::Stalker(s) => Some(&s.human.monster.creature.base.visual),
            Self::Mutant(s) => Some(&s.monster.creature.base.visual),
            Self::PhysicsObject(s) => Some(&s.base.visual),
            Self::InventoryBox(s) => Some(&s.base.visual),
            Self::Trader(s) => Some(&s.base.visual),
            Self::Weapon(s) => Some(&s.item.base.visual),
            Self::InventoryItem(s) => Some(&s.base.visual),
            _ => None,
        }
    }
}

/// Typed ALife simulation entity inside `.scop` Chunk 1.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopAlifeEntity {
    pub spawn: SpawnPacket,
    pub update_msg_id: u16,
    pub update_payload: Vec<u8>,
    pub payload: ScopAlifePayload,
}

impl ScopAlifeEntity {
    /// Construct a new typed `.scop` entity from high-level fields and an explicit `ScopAlifePayload`.
    pub fn new(
        id: u16,
        section: impl Into<String>,
        name_replace: impl Into<String>,
        position: [f32; 3],
        angle: [f32; 3],
        payload: ScopAlifePayload,
    ) -> Self {
        let state_payload = payload.to_state_bytes();
        let spawn = SpawnPacket {
            section: section.into(),
            name_replace: name_replace.into(),
            game_id_legacy: 0,
            rp: 0xFE,
            position,
            angle,
            respawn_time: 0,
            id,
            parent_id: INVALID_OBJECT_ID,
            phantom_id: INVALID_OBJECT_ID,
            flags: m_spawn_flags::OBJECT_LOCAL | m_spawn_flags::VERSION,
            version: SPAWN_VERSION,
            game_type: u16::MAX,
            script_version: 10,
            client_data: Vec::new(),
            spawn_id: INVALID_OBJECT_ID,
            state_payload,
            trailing_update_payload: Vec::new(),
        };
        let update_payload = match payload.state_health() {
            Some(health) => health.to_le_bytes().to_vec(),
            None => Vec::new(),
        };
        Self {
            spawn,
            update_msg_id: M_UPDATE,
            update_payload,
            payload,
        }
    }

    /// Re-serialize `self.payload` into `self.spawn.state_payload`.
    pub fn sync_state_payload(&mut self) {
        self.spawn.state_payload = self.payload.to_state_bytes();
    }

    /// Convert a canonical [`SaveObjectRecord`] into a typed [`ScopAlifeEntity`].
    pub fn from_save_object(record: &SaveObjectRecord) -> Result<Self, FormatError> {
        let kind = classify_alife_section(record.spawn.id, &record.spawn.section);
        let payload = decode_payload_with_fallback(&record.spawn, kind)?;
        Ok(Self {
            spawn: record.spawn.clone(),
            update_msg_id: record.update_msg_id,
            update_payload: record.update_payload.clone(),
            payload,
        })
    }

    /// Convert back into a canonical [`SaveObjectRecord`] (synchronizing state payload if edited).
    pub fn to_save_object(&self) -> SaveObjectRecord {
        let mut spawn = self.spawn.clone();
        spawn.state_payload = self.payload.to_state_bytes();
        SaveObjectRecord {
            spawn,
            update_msg_id: self.update_msg_id,
            update_payload: self.update_payload.clone(),
        }
    }

    #[inline]
    pub fn id(&self) -> u16 {
        self.spawn.id
    }

    #[inline]
    pub fn parent_id(&self) -> u16 {
        self.spawn.parent_id
    }

    #[inline]
    pub fn section(&self) -> &str {
        &self.spawn.section
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.spawn.name_replace
    }

    #[inline]
    pub fn position(&self) -> [f32; 3] {
        self.spawn.position
    }

    #[inline]
    pub fn angle(&self) -> [f32; 3] {
        self.spawn.angle
    }

    #[inline]
    pub fn kind(&self) -> ScopAlifeKind {
        self.payload.kind()
    }

    #[inline]
    pub fn alife(&self) -> &AlifeObjectState {
        self.payload.alife_object()
    }

    #[inline]
    pub fn game_vertex_id(&self) -> u16 {
        self.alife().game_vertex_id
    }

    #[inline]
    pub fn level_vertex_id(&self) -> u32 {
        self.alife().level_vertex_id
    }

    #[inline]
    pub fn story_id(&self) -> u32 {
        self.alife().story_id
    }

    #[inline]
    pub fn custom_data(&self) -> &str {
        &self.alife().custom_data
    }

    #[inline]
    pub fn is_online(&self) -> bool {
        (self.alife().flags & alife_object_flags::SWITCH_ONLINE) != 0
    }

    /// Decode health from `M_UPDATE` (when `>= 4` bytes for creatures) or `M_SPAWN` state payload.
    pub fn health(&self) -> Option<f32> {
        match &self.payload {
            ScopAlifePayload::Actor(_)
            | ScopAlifePayload::Stalker(_)
            | ScopAlifePayload::Mutant(_) => {
                if self.update_payload.len() >= 4 {
                    Some(f32::from_le_bytes([
                        self.update_payload[0],
                        self.update_payload[1],
                        self.update_payload[2],
                        self.update_payload[3],
                    ]))
                } else {
                    self.payload.state_health()
                }
            }
            _ => None,
        }
    }

    /// Parse a Chunk 1 ALife objects payload (`Vec<ScopAlifeEntity>`).
    pub fn parse_list(data: &[u8]) -> Result<Vec<Self>, FormatError> {
        let mut cur = ByteCursor::new(data);
        let count = cur.read_u32()? as usize;
        if count > MAX_SCOP_OBJECTS {
            return Err(FormatError::InvalidValue(
                "ALife object count exceeds sanity bound in .scop Chunk 1",
            ));
        }
        let mut entities = Vec::with_capacity(count);
        for _ in 0..count {
            let kind = ScopAlifeKind::from_u8(cur.read_u8()?)?;
            let spawn_len = usize::from(cur.read_u16()?);
            let spawn_bytes = cur.read_bytes(spawn_len)?;
            let spawn = SpawnPacket::parse(spawn_bytes)?;

            let update_len = usize::from(cur.read_u16()?);
            let update_bytes = cur.read_bytes(update_len)?;
            if update_bytes.len() < 2 {
                return Err(FormatError::InvalidValue("truncated M_UPDATE packet in .scop Chunk 1"));
            }
            let update_msg_id = u16::from_le_bytes([update_bytes[0], update_bytes[1]]);
            if update_msg_id != M_UPDATE {
                return Err(FormatError::InvalidValue(
                    "expected M_UPDATE packet ID (0) in .scop Chunk 1",
                ));
            }
            let update_payload = update_bytes[2..].to_vec();

            let payload = if kind == ScopAlifeKind::SmartTerrain {
                let zone = AlifeSmartZoneState::parse(spawn.version, &spawn.state_payload)?;
                ScopAlifePayload::SmartTerrain(ScopSmartTerrainState::parse_ext(
                    &mut cur,
                    spawn.name_replace.clone(),
                    zone,
                )?)
            } else {
                decode_payload_strict_or_fallback(&spawn, kind)?
            };

            entities.push(Self {
                spawn,
                update_msg_id,
                update_payload,
                payload,
            });
        }
        Ok(entities)
    }

    /// Serialize Chunk 1 ALife objects payload.
    pub fn encode_list(entities: &[Self]) -> Result<Vec<u8>, FormatError> {
        let mut out = Vec::with_capacity(4 + entities.len() * 96);
        out.extend_from_slice(&(entities.len() as u32).to_le_bytes());
        for entity in entities {
            out.push(entity.kind() as u8);

            let mut spawn = entity.spawn.clone();
            spawn.state_payload = entity.payload.to_state_bytes();
            let spawn_bytes = spawn.to_bytes();
            let spawn_len = u16::try_from(spawn_bytes.len())
                .map_err(|_| FormatError::InvalidValue("spawn packet exceeds u16 in .scop Chunk 1"))?;
            out.extend_from_slice(&spawn_len.to_le_bytes());
            out.extend_from_slice(&spawn_bytes);

            let update_total = entity.update_payload.len() + 2;
            let update_len = u16::try_from(update_total)
                .map_err(|_| FormatError::InvalidValue("update packet exceeds u16 in .scop Chunk 1"))?;
            out.extend_from_slice(&update_len.to_le_bytes());
            out.extend_from_slice(&entity.update_msg_id.to_le_bytes());
            out.extend_from_slice(&entity.update_payload);

            if let ScopAlifePayload::SmartTerrain(smart) = &entity.payload {
                smart.write_ext(&mut out);
            }
        }
        Ok(out)
    }
}

/// Classify a section/object ID into a [`ScopAlifeKind`].
pub fn classify_alife_section(object_id: u16, section: &str) -> ScopAlifeKind {
    let s = section.trim().to_ascii_lowercase();
    if object_id == 0 || s == "actor" || s == "mp_actor" {
        return ScopAlifeKind::Actor;
    }
    if s == "smart_terrain" || s.contains("smart_terrain") || s.ends_with("_smart") {
        return ScopAlifeKind::SmartTerrain;
    }
    if s == "inventory_box"
        || s.starts_with("inventory_box")
        || s.starts_with("m_inventory_box")
        || s.contains("stash")
        || s.starts_with("taynik")
    {
        return ScopAlifeKind::InventoryBox;
    }
    if s == "level_changer" || s.starts_with("level_changer") {
        return ScopAlifeKind::LevelChanger;
    }
    if s == "smart_cover" || s.starts_with("smart_cover") {
        return ScopAlifeKind::SmartCover;
    }
    if s == "online_offline_group" || s.starts_with("sim_squad") || s.contains("_squad") {
        return ScopAlifeKind::OnlineOfflineGroup;
    }
    if s == "m_trader" || s.ends_with("_trader") {
        return ScopAlifeKind::Trader;
    }
    if s.contains("stalker") || s.starts_with("sim_default_") {
        return ScopAlifeKind::Stalker;
    }
    if is_mutant_section(&s) {
        return ScopAlifeKind::Mutant;
    }
    if s.starts_with("physic_")
        || s.contains("destroyable_object")
        || s == "o_physic"
        || s == "search_light"
    {
        return ScopAlifeKind::PhysicsObject;
    }
    if s.starts_with("zone_") || s.contains("anomaly") {
        return ScopAlifeKind::AnomalousZone;
    }
    if s == "space_restrictor"
        || s.contains("restrictor")
        || s == "script_zone"
        || s.starts_with("camp_zone")
    {
        return ScopAlifeKind::SpaceRestrictor;
    }
    if s.starts_with("wpn_") && !s.starts_with("wpn_addon_") {
        return ScopAlifeKind::Weapon;
    }
    if is_inventory_item_section(&s) {
        return ScopAlifeKind::InventoryItem;
    }
    ScopAlifeKind::DynamicObject
}

fn is_mutant_section(s: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "m_bloodsucker",
        "bloodsucker",
        "m_boar",
        "boar_",
        "m_burer",
        "burer_",
        "m_cat",
        "cat_",
        "m_chimera",
        "chimera_",
        "m_controller",
        "controller_",
        "m_dog",
        "dog_",
        "m_flesh",
        "flesh_",
        "m_fracture",
        "fracture_",
        "m_gigant",
        "gigant_",
        "m_izlom",
        "m_karlik",
        "m_poltergeist",
        "poltergeist_",
        "m_pseudodog",
        "pseudodog_",
        "psy_dog",
        "psysucker",
        "m_rat",
        "rat_",
        "m_snork",
        "snork_",
        "m_tushkano",
        "tushkano_",
        "m_zombie",
        "zombie_",
        "m_crow",
        "mutant_",
        "mut_",
    ];
    PREFIXES.iter().any(|p| s.starts_with(p))
}

fn is_inventory_item_section(s: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "medkit",
        "bandage",
        "antirad",
        "vodka",
        "bread",
        "kolbasa",
        "conserva",
        "energy_drink",
        "af_",
        "ammo_",
        "grenade_",
        "detector_",
        "device_torch",
        "device_pda",
        "bolt",
        "wpn_addon_",
        "helm_",
        "novice_outfit",
        "stalker_outfit",
        "exo_outfit",
        "outfit_",
    ];
    PREFIXES.iter().any(|p| s.starts_with(p)) || s.ends_with("_outfit")
}

fn decode_payload_exact(
    spawn: &SpawnPacket,
    kind: ScopAlifeKind,
) -> Result<ScopAlifePayload, FormatError> {
    let v = spawn.version;
    let b = &spawn.state_payload;
    match kind {
        ScopAlifeKind::Actor => Ok(ScopAlifePayload::Actor(AlifeCreatureActorState::parse(v, b)?)),
        ScopAlifeKind::Stalker => {
            Ok(ScopAlifePayload::Stalker(AlifeHumanStalkerState::parse(v, b)?))
        }
        ScopAlifeKind::Mutant => Ok(ScopAlifePayload::Mutant(AlifeMonsterBaseState::parse(v, b)?)),
        ScopAlifeKind::SmartTerrain => {
            let zone = AlifeSmartZoneState::parse(v, b)?;
            Ok(ScopAlifePayload::SmartTerrain(ScopSmartTerrainState::new(
                spawn.name_replace.clone(),
                zone,
            )))
        }
        ScopAlifeKind::PhysicsObject => {
            Ok(ScopAlifePayload::PhysicsObject(AlifeObjectPhysicState::parse(v, b)?))
        }
        ScopAlifeKind::SpaceRestrictor => Ok(ScopAlifePayload::SpaceRestrictor(
            AlifeSpaceRestrictorState::parse(v, b)?,
        )),
        ScopAlifeKind::InventoryBox => {
            Ok(ScopAlifePayload::InventoryBox(AlifeInventoryBoxState::parse(v, b)?))
        }
        ScopAlifeKind::LevelChanger => {
            Ok(ScopAlifePayload::LevelChanger(AlifeLevelChangerState::parse(v, b)?))
        }
        ScopAlifeKind::AnomalousZone => Ok(ScopAlifePayload::AnomalousZone(
            AlifeAnomalousZoneState::parse(v, b)?,
        )),
        ScopAlifeKind::SmartCover => {
            Ok(ScopAlifePayload::SmartCover(AlifeSmartCoverState::parse(v, b)?))
        }
        ScopAlifeKind::OnlineOfflineGroup => Ok(ScopAlifePayload::OnlineOfflineGroup(
            AlifeOnlineOfflineGroupState::parse(v, b)?,
        )),
        ScopAlifeKind::Trader => Ok(ScopAlifePayload::Trader(AlifeTraderState::parse(v, b)?)),
        ScopAlifeKind::Weapon => Ok(ScopAlifePayload::Weapon(AlifeItemWeaponState::parse(v, b)?)),
        ScopAlifeKind::InventoryItem => Ok(ScopAlifePayload::InventoryItem(
            AlifeInventoryItemState::parse(v, b)?,
        )),
        ScopAlifeKind::DynamicObject => {
            Ok(ScopAlifePayload::DynamicObject(AlifeObjectState::parse(v, b)?))
        }
    }
}

fn decode_payload_with_fallback(
    spawn: &SpawnPacket,
    kind: ScopAlifeKind,
) -> Result<ScopAlifePayload, FormatError> {
    match decode_payload_exact(spawn, kind) {
        Ok(payload) => Ok(payload),
        Err(_) if kind != ScopAlifeKind::DynamicObject => {
            // Canonical Anomaly / minimal synthetic saves may have mod overrides or bare
            // `CSE_ALifeObject` payloads; gracefully preserve as `DynamicObject`.
            Ok(ScopAlifePayload::DynamicObject(AlifeObjectState::parse(
                spawn.version,
                &spawn.state_payload,
            )?))
        }
        Err(e) => Err(e),
    }
}

fn decode_payload_strict_or_fallback(
    spawn: &SpawnPacket,
    kind: ScopAlifeKind,
) -> Result<ScopAlifePayload, FormatError> {
    decode_payload_with_fallback(spawn, kind)
}

fn parse_custom_data_u16(custom_data: &str, key: &str) -> Option<u16> {
    for line in custom_data.lines() {
        let trimmed = line.split(';').next().unwrap_or("").trim();
        if let Some((lhs, rhs)) = trimmed.split_once('=')
            && lhs.trim().eq_ignore_ascii_case(key)
        {
            return rhs.trim().parse::<u16>().ok();
        }
    }
    None
}

// ============================================================================
// Chunk 2: Game Graph Mappings & Story Quests
// ============================================================================

/// Game graph vertex / level / spawn update mapping entry inside `.scop` Chunk 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopGraphMapping {
    pub entity_id: u16,
    pub spawn_id: u16,
    pub game_vertex_id: u16,
    pub level_vertex_id: u32,
    pub level_id: u8,
    pub level_name: String,
    pub story_id: u32,
    pub spawn_vertex_id: Option<u32>,
    pub spawn_update_bytes: Vec<u8>,
}

impl ScopGraphMapping {
    pub fn parse_from_cursor(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let entity_id = cur.read_u16()?;
        let spawn_id = cur.read_u16()?;
        let game_vertex_id = cur.read_u16()?;
        let level_vertex_id = cur.read_u32()?;
        let level_id = cur.read_u8()?;
        let level_name = cur.read_zstring()?;
        let story_id = cur.read_u32()?;
        let has_sv = cur.read_u8()? != 0;
        let spawn_vertex_id = if has_sv { Some(cur.read_u32()?) } else { None };
        let spawn_update_bytes = read_bytes_vec(cur)?;
        Ok(Self {
            entity_id,
            spawn_id,
            game_vertex_id,
            level_vertex_id,
            level_id,
            level_name,
            story_id,
            spawn_vertex_id,
            spawn_update_bytes,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.entity_id.to_le_bytes());
        out.extend_from_slice(&self.spawn_id.to_le_bytes());
        out.extend_from_slice(&self.game_vertex_id.to_le_bytes());
        out.extend_from_slice(&self.level_vertex_id.to_le_bytes());
        out.push(self.level_id);
        write_zstring(out, &self.level_name);
        out.extend_from_slice(&self.story_id.to_le_bytes());
        match self.spawn_vertex_id {
            Some(sv) => {
                out.push(1);
                out.extend_from_slice(&sv.to_le_bytes());
            }
            None => out.push(0),
        }
        write_bytes_vec(out, &self.spawn_update_bytes);
    }
}

/// High-level mission / story quest status (`CGameTask::ETaskState` in CoP / Anomaly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopQuestStatus {
    Failed,
    InProgress,
    Completed,
    Skipped,
    Unknown(u32),
}

impl ScopQuestStatus {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Failed,
            1 => Self::InProgress,
            2 => Self::Completed,
            3 => Self::Skipped,
            other => Self::Unknown(other),
        }
    }

    pub fn to_raw(self) -> u32 {
        match self {
            Self::Failed => 0,
            Self::InProgress => 1,
            Self::Completed => 2,
            Self::Skipped => 3,
            Self::Unknown(v) => v,
        }
    }
}

/// Typed story quest entry wrapping owner entity ID and [`SavedGameTask`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopStoryQuest {
    pub owner_id: u16,
    pub task: SavedGameTask,
}

impl ScopStoryQuest {
    #[inline]
    pub fn status(&self) -> ScopQuestStatus {
        ScopQuestStatus::from_raw(self.task.task_state)
    }

    #[inline]
    pub fn task_id(&self) -> &str {
        &self.task.task_id
    }

    #[inline]
    pub fn title(&self) -> &str {
        &self.task.title
    }
}

/// Chunk 2 container holding game graph mappings, story quests, active quest,
/// info portions, and canonical ALife registries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScopGraphAndQuests {
    pub graph_mappings: Vec<ScopGraphMapping>,
    pub story_quests: Vec<ScopStoryQuest>,
    pub active_quest_id: String,
    pub info_portions: Vec<(u16, Vec<String>)>,
    pub registries: SaveRegistries,
}

impl ScopGraphAndQuests {
    pub fn parse(data: &[u8], alife_version: u32) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let map_count = cur.read_u32()? as usize;
        if map_count > MAX_SCOP_OBJECTS {
            return Err(FormatError::InvalidValue(
                "graph mapping count exceeds sanity bound in .scop Chunk 2",
            ));
        }
        let mut graph_mappings = Vec::with_capacity(map_count);
        for _ in 0..map_count {
            graph_mappings.push(ScopGraphMapping::parse_from_cursor(&mut cur)?);
        }

        let reg_bytes = read_bytes_vec(&mut cur)?;
        let registries = SaveRegistries::parse(&reg_bytes, alife_version)?;

        let mut story_quests = Vec::new();
        for (owner_id, tasks) in &registries.game_tasks.tasks_by_owner {
            for task in tasks {
                story_quests.push(ScopStoryQuest {
                    owner_id: *owner_id,
                    task: task.clone(),
                });
            }
        }
        let active_quest_id = registries.game_tasks.active_task_id.clone();
        let info_portions = registries.info_portions.clone();

        Ok(Self {
            graph_mappings,
            story_quests,
            active_quest_id,
            info_portions,
            registries,
        })
    }

    /// Add a story quest entry and keep `registries.game_tasks` synchronized.
    pub fn add_quest(&mut self, quest: ScopStoryQuest) {
        self.story_quests.push(quest);
        self.sync_registries();
    }

    /// Set the active quest ID and keep `registries.game_tasks.active_task_id` synchronized.
    pub fn set_active_quest(&mut self, quest_id: impl Into<String>) {
        self.active_quest_id = quest_id.into();
        self.registries.game_tasks.active_task_id = self.active_quest_id.clone();
    }

    /// Synchronize `info_portions`, `story_quests`, and `active_quest_id` back into `self.registries`.
    pub fn sync_registries(&mut self) {
        self.registries = self.synced_registries();
    }

    /// Synchronize `info_portions`, `story_quests`, and `active_quest_id` back into `SaveRegistries`.
    pub fn synced_registries(&self) -> SaveRegistries {
        let mut reg = self.registries.clone();
        if !self.info_portions.is_empty() || reg.info_portions.is_empty() {
            reg.info_portions = self.info_portions.clone();
        }
        if !self.story_quests.is_empty() || !self.active_quest_id.is_empty() {
            let mut by_owner: BTreeMap<u16, Vec<SavedGameTask>> = BTreeMap::new();
            for q in &self.story_quests {
                by_owner.entry(q.owner_id).or_default().push(q.task.clone());
            }
            reg.game_tasks = GameTaskRegistryData {
                tasks_by_owner: by_owner.into_iter().collect(),
                active_task_id: self.active_quest_id.clone(),
            };
        }
        reg
    }

    pub fn to_bytes(&self, alife_version: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        out.extend_from_slice(&(self.graph_mappings.len() as u32).to_le_bytes());
        for mapping in &self.graph_mappings {
            mapping.write_to(&mut out);
        }
        let reg_bytes = self.synced_registries().to_bytes(alife_version);
        write_bytes_vec(&mut out, &reg_bytes);
        out
    }
}

// ============================================================================
// Chunk 3: Environment & Weather State
// ============================================================================

/// Chunk 3 environment & atmospheric weather state persisted alongside ALife.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopWeatherState {
    pub current_weather: String,
    pub next_weather: String,
    pub weather_fx: String,
    pub weather_fx_time: f32,
    pub cycle_time_sec: f32,
    pub rain_factor: f32,
    pub rain_wetness: f32,
    pub wind_velocity: f32,
    pub wind_direction_rad: f32,
    pub fog_density: f32,
    pub thunder_active: bool,
    pub surge_active: bool,
    pub psi_storm_active: bool,
}

impl Default for ScopWeatherState {
    fn default() -> Self {
        Self {
            current_weather: "w_clear1".to_string(),
            next_weather: "w_clear1".to_string(),
            weather_fx: String::new(),
            weather_fx_time: 0.0,
            cycle_time_sec: 9.0 * 3600.0,
            rain_factor: 0.0,
            rain_wetness: 0.0,
            wind_velocity: 0.0,
            wind_direction_rad: 0.0,
            fog_density: 0.0,
            thunder_active: false,
            surge_active: false,
            psi_storm_active: false,
        }
    }
}

impl ScopWeatherState {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let version = cur.read_u32()?;
        if version != SCOP_FORMAT_VERSION {
            return Err(FormatError::InvalidValue("unsupported .scop weather Chunk 3 version"));
        }
        Ok(Self {
            current_weather: cur.read_zstring()?,
            next_weather: cur.read_zstring()?,
            weather_fx: cur.read_zstring()?,
            weather_fx_time: cur.read_f32()?,
            cycle_time_sec: cur.read_f32()?,
            rain_factor: cur.read_f32()?,
            rain_wetness: cur.read_f32()?,
            wind_velocity: cur.read_f32()?,
            wind_direction_rad: cur.read_f32()?,
            fog_density: cur.read_f32()?,
            thunder_active: cur.read_u8()? != 0,
            surge_active: cur.read_u8()? != 0,
            psi_storm_active: cur.read_u8()? != 0,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64);
        out.extend_from_slice(&SCOP_FORMAT_VERSION.to_le_bytes());
        write_zstring(&mut out, &self.current_weather);
        write_zstring(&mut out, &self.next_weather);
        write_zstring(&mut out, &self.weather_fx);
        out.extend_from_slice(&self.weather_fx_time.to_le_bytes());
        out.extend_from_slice(&self.cycle_time_sec.to_le_bytes());
        out.extend_from_slice(&self.rain_factor.to_le_bytes());
        out.extend_from_slice(&self.rain_wetness.to_le_bytes());
        out.extend_from_slice(&self.wind_velocity.to_le_bytes());
        out.extend_from_slice(&self.wind_direction_rad.to_le_bytes());
        out.extend_from_slice(&self.fog_density.to_le_bytes());
        out.push(u8::from(self.thunder_active));
        out.push(u8::from(self.surge_active));
        out.push(u8::from(self.psi_storm_active));
        out
    }

    #[inline]
    pub fn wetness_state(&self) -> WetnessState {
        WetnessState::new(self.rain_wetness)
    }

    /// Synchronize atmospheric scalars from a blended [`MixedEnvironment`].
    pub fn apply_mixed_environment(&mut self, mixed: &MixedEnvironment) {
        self.rain_factor = mixed.rain_density.clamp(0.0, 1.0);
        self.rain_wetness = mixed.wet_surfaces.clamp(0.0, 1.0);
        self.wind_velocity = mixed.wind_velocity;
        self.wind_direction_rad = mixed.wind_direction;
        self.fog_density = mixed.fog_density;
        self.thunder_active = !mixed.active_thunderbolt_id.is_empty();
    }

    /// Compute normalized cycle time `[0, DAY_LENGTH)` from a [`GameDateTime`].
    pub fn cycle_seconds_from_date_time(dt: GameDateTime) -> f32 {
        let sec = (dt.hour * 3600 + dt.minute * 60 + dt.second) as f32
            + (dt.millisecond as f32) * 0.001;
        sec.rem_euclid(DAY_LENGTH)
    }
}

// ============================================================================
// Complete Structured ScopSaveFile (Chunks 0..3 + Canonical .scop Conversion)
// ============================================================================

/// Complete typed `.scop` ALife & world savegame file (`Chunks 0..3` + canonical `CALifeStorageManager` interop).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScopSaveFile {
    pub header: ScopHeader,
    pub entities: Vec<ScopAlifeEntity>,
    pub graph_and_quests: ScopGraphAndQuests,
    pub weather: ScopWeatherState,
    /// Preserved unrecognized chunks verbatim during roundtrip.
    pub extra_chunks: Vec<(u32, Vec<u8>)>,
}

impl ScopSaveFile {
    /// Decompress and parse either a structured Chunk `0..3` `.scop` file or a canonical
    /// `CALifeStorageManager` `.scop` savegame.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let (hdr_version, inner) = decompress_scop(data)?;
        Self::parse_inner(&inner, hdr_version)
    }

    /// Parse an already-decompressed `.scop` inner chunk stream.
    pub fn parse_inner(inner: &[u8], hdr_version: u32) -> Result<Self, FormatError> {
        let root = ChunkReader::new(inner);

        if let Some(chunk0) = root.open_chunk(SCOP_CHUNK_HEADER)?
            && chunk0.len() >= 4
            && chunk0[..4] == SCOP_HEADER_SIGNATURE
        {
            let header = ScopHeader::parse(&chunk0)?;

            let chunk1 = root
                .open_chunk(SCOP_CHUNK_ALIFE_OBJECTS)?
                .ok_or(FormatError::MissingChunk(SCOP_CHUNK_ALIFE_OBJECTS))?;
            let entities = ScopAlifeEntity::parse_list(&chunk1)?;

            let chunk2 = root
                .open_chunk(SCOP_CHUNK_GRAPH_QUESTS)?
                .ok_or(FormatError::MissingChunk(SCOP_CHUNK_GRAPH_QUESTS))?;
            let graph_and_quests = ScopGraphAndQuests::parse(&chunk2, header.alife_version)?;

            let chunk3 = root
                .open_chunk(SCOP_CHUNK_WEATHER_ENV)?
                .ok_or(FormatError::MissingChunk(SCOP_CHUNK_WEATHER_ENV))?;
            let weather = ScopWeatherState::parse(&chunk3)?;

            let mut extra_chunks = Vec::new();
            for entry in root.iter() {
                let entry = entry?;
                if !matches!(
                    entry.id,
                    SCOP_CHUNK_HEADER
                        | SCOP_CHUNK_ALIFE_OBJECTS
                        | SCOP_CHUNK_GRAPH_QUESTS
                        | SCOP_CHUNK_WEATHER_ENV
                ) {
                    extra_chunks.push((entry.id, entry.decode()?.into_owned()));
                }
            }

            return Ok(Self {
                header,
                entities,
                graph_and_quests,
                weather,
                extra_chunks,
            });
        }

        // Canonical `CALifeStorageManager` `.scop` layout (`0x0000`, `0x0001`, `0x0002`, `0x0005`, `0x0009`).
        let mut save = SavegameFile::parse_inner(inner)?;
        if save.alife_version < hdr_version {
            save.alife_version = hdr_version;
        }
        Self::from_savegame(&save)
    }

    /// Encode the uncompressed inner Chunk `0..3` stream.
    pub fn to_inner_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let mut root = ChunkWriter::new();
        root.write_chunk(SCOP_CHUNK_HEADER, &self.header.to_bytes());
        root.write_chunk(
            SCOP_CHUNK_ALIFE_OBJECTS,
            &ScopAlifeEntity::encode_list(&self.entities)?,
        );
        root.write_chunk(
            SCOP_CHUNK_GRAPH_QUESTS,
            &self.graph_and_quests.to_bytes(self.header.alife_version),
        );
        root.write_chunk(SCOP_CHUNK_WEATHER_ENV, &self.weather.to_bytes());
        for (id, payload) in &self.extra_chunks {
            root.write_chunk(*id, payload);
        }
        Ok(root.into_bytes())
    }

    /// Compress and serialize into an LZO1X `.scop` savegame file using Chunk `0..3` layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let inner = self.to_inner_bytes()?;
        Ok(compress_scop(self.header.alife_version, &inner))
    }

    /// Convert a canonical [`SavegameFile`] (`savegames.rs`) into a typed [`ScopSaveFile`].
    pub fn from_savegame(save: &SavegameFile) -> Result<Self, FormatError> {
        let entities: Vec<ScopAlifeEntity> = save
            .objects
            .iter()
            .map(ScopAlifeEntity::from_save_object)
            .collect::<Result<_, _>>()?;

        let mut actor_stats = ScopActorStats::default();
        if let Some((_, secs)) = save.registries.actor_statistics.iter().find(|(id, _)| *id == 0) {
            actor_stats.statistics = secs.clone();
        }
        if let Some(actor_ent) = entities.iter().find(|e| e.id() == 0) {
            actor_stats.position = actor_ent.position();
            actor_stats.angles = actor_ent.angle();
            actor_stats.game_vertex_id = actor_ent.game_vertex_id();
            actor_stats.level_vertex_id = actor_ent.level_vertex_id();
            if let Some(hp) = actor_ent.health() {
                actor_stats.health = hp;
            }
            if let ScopAlifePayload::Actor(act) = &actor_ent.payload {
                actor_stats.actor_name = if act.trader.character_name.is_empty() {
                    actor_ent.name().to_string()
                } else {
                    act.trader.character_name.clone()
                };
                actor_stats.profile_name = act.trader.character_profile.clone();
                actor_stats.community_index = act.trader.community_index;
                actor_stats.money = act.trader.money;
                actor_stats.rank = act.trader.rank;
                actor_stats.reputation = act.trader.reputation;
                actor_stats.holder_id = act.holder_id;
            } else {
                actor_stats.actor_name = actor_ent.name().to_string();
            }
        }

        let level_name = save
            .spawn_info
            .level_name
            .clone()
            .unwrap_or_default();

        let header = ScopHeader {
            alife_version: save.alife_version,
            format_version: SCOP_FORMAT_VERSION,
            spawn_name: save.spawn_info.spawn_name.clone(),
            spawn_guid: save.spawn_info.spawn_guid,
            level_id: save.spawn_info.level_id,
            level_name: level_name.clone(),
            game_time: save.time,
            actor_stats,
        };

        let mut graph_mappings = Vec::with_capacity(
            save.spawn_info.spawn_updates.len().max(entities.len()),
        );
        let default_level_id = save.spawn_info.level_id.unwrap_or(0);
        for (idx, (spawn_vertex_id, update_bytes)) in
            save.spawn_info.spawn_updates.iter().enumerate()
        {
            let matched = entities.get(idx);
            graph_mappings.push(ScopGraphMapping {
                entity_id: matched.map_or(INVALID_OBJECT_ID, ScopAlifeEntity::id),
                spawn_id: matched.map_or(INVALID_OBJECT_ID, |e| e.spawn.spawn_id),
                game_vertex_id: matched.map_or(INVALID_GAME_VERTEX, ScopAlifeEntity::game_vertex_id),
                level_vertex_id: matched
                    .map_or(INVALID_AI_VERTEX, ScopAlifeEntity::level_vertex_id),
                level_id: default_level_id,
                level_name: level_name.clone(),
                story_id: matched.map_or(INVALID_STORY_ID, ScopAlifeEntity::story_id),
                spawn_vertex_id: Some(*spawn_vertex_id),
                spawn_update_bytes: update_bytes.clone(),
            });
        }
        if graph_mappings.is_empty() {
            for ent in &entities {
                graph_mappings.push(ScopGraphMapping {
                    entity_id: ent.id(),
                    spawn_id: ent.spawn.spawn_id,
                    game_vertex_id: ent.game_vertex_id(),
                    level_vertex_id: ent.level_vertex_id(),
                    level_id: default_level_id,
                    level_name: level_name.clone(),
                    story_id: ent.story_id(),
                    spawn_vertex_id: None,
                    spawn_update_bytes: Vec::new(),
                });
            }
        }

        let mut story_quests = Vec::new();
        for (owner_id, tasks) in &save.registries.game_tasks.tasks_by_owner {
            for task in tasks {
                story_quests.push(ScopStoryQuest {
                    owner_id: *owner_id,
                    task: task.clone(),
                });
            }
        }

        let graph_and_quests = ScopGraphAndQuests {
            graph_mappings,
            story_quests,
            active_quest_id: save.registries.game_tasks.active_task_id.clone(),
            info_portions: save.registries.info_portions.clone(),
            registries: save.registries.clone(),
        };

        let mut weather = ScopWeatherState {
            cycle_time_sec: ScopWeatherState::cycle_seconds_from_date_time(save.time.date_time()),
            ..Default::default()
        };
        let mut extra_chunks = Vec::new();
        for (id, payload) in &save.extra_chunks {
            if *id == SCOP_CANONICAL_WEATHER_CHUNK
                && let Ok(parsed_w) = ScopWeatherState::parse(payload)
            {
                weather = parsed_w;
            } else {
                extra_chunks.push((*id, payload.clone()));
            }
        }

        Ok(Self {
            header,
            entities,
            graph_and_quests,
            weather,
            extra_chunks,
        })
    }

    /// Convert back into a canonical `CALifeStorageManager` [`SavegameFile`].
    pub fn to_savegame(&self) -> Result<SavegameFile, FormatError> {
        let spawn_updates: Vec<(u32, Vec<u8>)> = self
            .graph_and_quests
            .graph_mappings
            .iter()
            .filter_map(|m| {
                m.spawn_vertex_id
                    .map(|sv| (sv, m.spawn_update_bytes.clone()))
            })
            .collect();

        let level_name = if self.header.level_name.is_empty() && self.header.level_id.is_none() {
            None
        } else {
            Some(self.header.level_name.clone())
        };

        let spawn_info = SaveSpawnHeader {
            spawn_name: self.header.spawn_name.clone(),
            spawn_guid: self.header.spawn_guid,
            level_id: self.header.level_id,
            level_name,
            spawn_updates,
        };

        let objects: Vec<SaveObjectRecord> = self
            .entities
            .iter()
            .map(ScopAlifeEntity::to_save_object)
            .collect();

        let mut registries = self.graph_and_quests.synced_registries();
        if !self.header.actor_stats.statistics.is_empty()
            && !registries.actor_statistics.iter().any(|(id, _)| *id == 0)
        {
            registries
                .actor_statistics
                .push((0, self.header.actor_stats.statistics.clone()));
        }

        let mut extra_chunks = self.extra_chunks.clone();
        extra_chunks.push((SCOP_CANONICAL_WEATHER_CHUNK, self.weather.to_bytes()));

        Ok(SavegameFile {
            alife_version: self.header.alife_version,
            time: self.header.game_time,
            spawn_info,
            objects,
            registries,
            extra_chunks,
        })
    }

    /// Serialize in canonical `CALifeStorageManager` `.scop` LZO1X format (plus weather chunk `0x000A`).
    pub fn to_anomaly_scop_bytes(&self) -> Result<Vec<u8>, FormatError> {
        self.to_savegame()?.to_bytes()
    }

    #[inline]
    pub fn dynamic_objects(&self) -> &[ScopAlifeEntity] {
        &self.entities
    }

    pub fn actor(&self) -> Option<&ScopAlifeEntity> {
        self.entities
            .iter()
            .find(|e| e.id() == 0 || e.kind() == ScopAlifeKind::Actor)
    }

    pub fn find_entity(&self, id: u16) -> Option<&ScopAlifeEntity> {
        self.entities.iter().find(|e| e.id() == id)
    }

    pub fn find_by_story_id(&self, story_id: u32) -> Option<&ScopAlifeEntity> {
        if story_id == INVALID_STORY_ID {
            return None;
        }
        self.entities.iter().find(|e| e.story_id() == story_id)
    }

    pub fn stalkers(&self) -> impl Iterator<Item = &ScopAlifeEntity> {
        self.entities
            .iter()
            .filter(|e| e.kind() == ScopAlifeKind::Stalker)
    }

    pub fn mutants(&self) -> impl Iterator<Item = &ScopAlifeEntity> {
        self.entities
            .iter()
            .filter(|e| e.kind() == ScopAlifeKind::Mutant)
    }

    pub fn smart_terrains(&self) -> impl Iterator<Item = &ScopAlifeEntity> {
        self.entities
            .iter()
            .filter(|e| e.kind() == ScopAlifeKind::SmartTerrain)
    }

    pub fn physics_objects(&self) -> impl Iterator<Item = &ScopAlifeEntity> {
        self.entities
            .iter()
            .filter(|e| e.kind() == ScopAlifeKind::PhysicsObject)
    }

    pub fn space_restrictors(&self) -> impl Iterator<Item = &ScopAlifeEntity> {
        self.entities
            .iter()
            .filter(|e| e.kind() == ScopAlifeKind::SpaceRestrictor)
    }

    /// Return all physics objects, space restrictors, level changers, and anomalous zones.
    pub fn physics_restrictors(&self) -> impl Iterator<Item = &ScopAlifeEntity> {
        self.entities
            .iter()
            .filter(|e| e.kind().is_physics_or_restrictor())
    }

    pub fn inventory_boxes(&self) -> impl Iterator<Item = &ScopAlifeEntity> {
        self.entities
            .iter()
            .filter(|e| e.kind() == ScopAlifeKind::InventoryBox)
    }

    /// Return all ALife objects whose `parent_id == owner_id` (e.g., inside an inventory box or NPC).
    pub fn entities_owned_by(&self, owner_id: u16) -> impl Iterator<Item = &ScopAlifeEntity> {
        self.entities
            .iter()
            .filter(move |e| e.parent_id() == owner_id)
    }

    pub fn active_quest(&self) -> Option<&ScopStoryQuest> {
        let active_id = &self.graph_and_quests.active_quest_id;
        if active_id.is_empty() {
            return None;
        }
        self.graph_and_quests
            .story_quests
            .iter()
            .find(|q| q.task_id() == active_id)
    }

    pub fn has_info_portion(&self, owner_id: u16, portion: &str) -> bool {
        self.graph_and_quests
            .info_portions
            .iter()
            .find(|(id, _)| *id == owner_id)
            .is_some_and(|(_, list)| list.iter().any(|p| p == portion))
    }
}

/// Suppress unused constant warnings while exporting `INVALID_SPAWN_STORY_ID` alignment.
#[allow(dead_code)]
const _SCOP_SENTINELS: (u16, u32, u32) =
    (INVALID_OBJECT_ID, INVALID_STORY_ID, INVALID_SPAWN_STORY_ID);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::savegames::{
        validate_scop_header, ActorStatDetail, GameNewsEntry, RelationData, SavedMapLocation,
        SavedTaskScriptHelper,
    };
    use crate::spawns::{
        ph_skeleton_flags, AlifeCreatureAbstractState, AlifeDynamicVisualState,
        AlifeHumanAbstractState, AlifeMonsterAbstractState, AlifeTraderAbstractState,
        ServerPhSkeletonState, ServerShapeDef,
    };

    fn sample_base_alife(gvid: u16, lvid: u32, story_id: u32, custom_data: &str) -> AlifeObjectState {
        AlifeObjectState {
            game_vertex_id: gvid,
            distance: 2.5,
            direct_control: true,
            level_vertex_id: lvid,
            flags: alife_object_flags::SWITCH_ONLINE
                | alife_object_flags::CAN_SAVE
                | alife_object_flags::USED_AI_LOCATIONS,
            custom_data: custom_data.to_string(),
            story_id,
            spawn_story_id: INVALID_SPAWN_STORY_ID,
            subclass_payload: Vec::new(),
        }
    }

    fn sample_dyn_vis(
        gvid: u16,
        lvid: u32,
        story_id: u32,
        visual_name: &str,
        custom_data: &str,
    ) -> AlifeDynamicVisualState {
        AlifeDynamicVisualState {
            alife: sample_base_alife(gvid, lvid, story_id, custom_data),
            visual: ServerVisualState {
                visual_name: visual_name.to_string(),
                flags: 0,
            },
            trailing: Vec::new(),
        }
    }

    fn sample_skeleton() -> ServerPhSkeletonState {
        ServerPhSkeletonState {
            startup_animation: "idle".to_string(),
            flags: ph_skeleton_flags::ACTIVE,
            source_id: INVALID_OBJECT_ID,
            saved_bones: None,
        }
    }

    fn build_rich_sample_save() -> ScopSaveFile {
        let dt = GameDateTime {
            year: 2018,
            month: 7,
            day: 14,
            hour: 16,
            minute: 45,
            second: 12,
            millisecond: 500,
        };

        let actor_state = AlifeCreatureActorState {
            creature: AlifeCreatureAbstractState {
                base: sample_dyn_vis(42, 12000, 0, "actors\\stalker_hero\\stalker_hero_1", ""),
                team: 0,
                squad: 0,
                group: 0,
                health: 0.92,
                dynamic_out_restrictions: Vec::new(),
                dynamic_in_restrictions: Vec::new(),
                killer_id: INVALID_OBJECT_ID,
                game_death_time: 0,
                trailing: Vec::new(),
            },
            trader: AlifeTraderAbstractState {
                money: 48500,
                specific_character: "actor".to_string(),
                trader_flags: 1,
                character_profile: "actor".to_string(),
                community_index: 0,
                rank: 1450,
                reputation: 320,
                character_name: "Zilla".to_string(),
                deadbody_can_take: true,
                deadbody_closed: false,
            },
            skeleton: sample_skeleton(),
            holder_id: INVALID_OBJECT_ID,
            trailing: Vec::new(),
        };

        let stalker_state = AlifeHumanStalkerState {
            human: AlifeHumanAbstractState {
                trader: AlifeTraderAbstractState {
                    money: 3200,
                    specific_character: "esc_2_12_stalker_wolf".to_string(),
                    trader_flags: 1,
                    character_profile: "esc_2_12_stalker_wolf".to_string(),
                    community_index: 1,
                    rank: 950,
                    reputation: 500,
                    character_name: "Wolf".to_string(),
                    deadbody_can_take: true,
                    deadbody_closed: false,
                },
                monster: AlifeMonsterAbstractState {
                    creature: AlifeCreatureAbstractState {
                        base: sample_dyn_vis(
                            45,
                            14300,
                            6,
                            "actors\\stalker_neutral\\stalker_neutral_3",
                            "[logic]\ncfg = scripts\\escape\\esc_wolf.ltx\n",
                        ),
                        team: 1,
                        squad: 2,
                        group: 1,
                        health: 1.0,
                        dynamic_out_restrictions: vec![30],
                        dynamic_in_restrictions: Vec::new(),
                        killer_id: INVALID_OBJECT_ID,
                        game_death_time: 0,
                        trailing: Vec::new(),
                    },
                    out_space_restrictors: String::new(),
                    in_space_restrictors: "esc_rookie_village_sr".to_string(),
                    smart_terrain_id: 20,
                    task_reached: true,
                    trailing: Vec::new(),
                },
                equipment_prefs: vec![1, 2, 1, 2, 0],
                main_weapon_prefs: vec![2, 1, 0, 1],
                trailing: Vec::new(),
            },
            skeleton: sample_skeleton(),
            trailing: Vec::new(),
        };

        let mutant_state = AlifeMonsterBaseState {
            monster: AlifeMonsterAbstractState {
                creature: AlifeCreatureAbstractState {
                    base: sample_dyn_vis(
                        60,
                        21500,
                        INVALID_STORY_ID,
                        "monsters\\bloodsucker\\bloodsucker",
                        "",
                    ),
                    team: 5,
                    squad: 0,
                    group: 0,
                    health: 0.78,
                    dynamic_out_restrictions: Vec::new(),
                    dynamic_in_restrictions: Vec::new(),
                    killer_id: INVALID_OBJECT_ID,
                    game_death_time: 0,
                    trailing: Vec::new(),
                },
                out_space_restrictors: String::new(),
                in_space_restrictors: String::new(),
                smart_terrain_id: 20,
                task_reached: false,
                trailing: Vec::new(),
            },
            skeleton: sample_skeleton(),
            spec_object_id: INVALID_OBJECT_ID,
            trailing: Vec::new(),
        };

        let mut smart_state = ScopSmartTerrainState::new(
            "esc_smart_terrain_2_12",
            AlifeSmartZoneState {
                alife: sample_base_alife(
                    44,
                    14000,
                    105,
                    "[smart_terrain]\ncfg = scripts\\escape\\smart\\esc_smart_terrain_2_12.ltx\nmax_population = 12\nsquad_id = 3\n",
                ),
                shapes: vec![ServerShapeDef::Sphere {
                    center: [0.0, 0.0, 0.0],
                    radius: 35.0,
                }],
                restrictor_type: 0,
                trailing: Vec::new(),
            },
        );
        smart_state.staying_time_ms = 86_400_000;
        smart_state.assigned_npc_ids = vec![10];
        smart_state.arriving_npc_ids = vec![15];
        smart_state.script_state = vec![0xCA, 0xFE];

        let restrictor_state = AlifeSpaceRestrictorState {
            alife: sample_base_alife(
                44,
                14050,
                INVALID_STORY_ID,
                "[logic]\ncfg = scripts\\escape\\esc_safe_zone.ltx\n",
            ),
            shapes: vec![ServerShapeDef::Box([
                10.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
            ])],
            restrictor_type: 2,
            trailing: Vec::new(),
        };

        let physics_state = AlifeObjectPhysicState {
            base: sample_dyn_vis(
                44,
                14100,
                INVALID_STORY_ID,
                "dynamics\\box\\box_metall_01",
                "[collide]\nignore_static\n",
            ),
            skeleton: sample_skeleton(),
            physic_type: 3,
            mass: 18.5,
            fixed_bones: "link".to_string(),
            trailing: Vec::new(),
        };

        let stash_state = AlifeInventoryBoxState {
            base: sample_dyn_vis(
                44,
                14120,
                501,
                "dynamics\\equipment_cache\\equipment_rucksack_01_case",
                "",
            ),
            can_take: true,
            closed: false,
            tip_text: "st_search_treasure".to_string(),
            trailing: Vec::new(),
        };

        let dyn_obj = sample_base_alife(44, 14130, 999, "[custom]\nflag = 1\n");

        let entities = vec![
            ScopAlifeEntity::new(
                0,
                "actor",
                "single_player",
                [-180.5, 2.1, -155.0],
                [0.0, 1.2, 0.0],
                ScopAlifePayload::Actor(actor_state),
            ),
            ScopAlifeEntity::new(
                10,
                "sim_default_stalker_veteran",
                "esc_wolf",
                [-170.0, 2.0, -140.0],
                [0.0, 0.4, 0.0],
                ScopAlifePayload::Stalker(stalker_state),
            ),
            ScopAlifeEntity::new(
                15,
                "bloodsucker_normal",
                "esc_bloodsucker_1",
                [-95.0, 0.5, -20.0],
                [0.0, -1.1, 0.0],
                ScopAlifePayload::Mutant(mutant_state),
            ),
            ScopAlifeEntity::new(
                20,
                "smart_terrain",
                "esc_smart_terrain_2_12",
                [-175.0, 2.0, -145.0],
                [0.0; 3],
                ScopAlifePayload::SmartTerrain(smart_state),
            ),
            ScopAlifeEntity::new(
                30,
                "space_restrictor",
                "esc_rookie_village_sr",
                [-175.0, 2.0, -145.0],
                [0.0; 3],
                ScopAlifePayload::SpaceRestrictor(restrictor_state),
            ),
            ScopAlifeEntity::new(
                31,
                "physic_object",
                "esc_crate_01",
                [-173.0, 2.0, -147.0],
                [0.0; 3],
                ScopAlifePayload::PhysicsObject(physics_state),
            ),
            ScopAlifeEntity::new(
                40,
                "inventory_box",
                "esc_wolf_stash",
                [-176.2, 2.5, -143.8],
                [0.0; 3],
                ScopAlifePayload::InventoryBox(stash_state),
            ),
            ScopAlifeEntity::new(
                50,
                "custom_alife_prop",
                "esc_radio_beacon",
                [-174.0, 3.0, -146.0],
                [0.0; 3],
                ScopAlifePayload::DynamicObject(dyn_obj),
            ),
        ];

        let actor_stat_sections = vec![ActorStatSection {
            key: "total".to_string(),
            details: vec![ActorStatDetail {
                key: "stalkerkills".to_string(),
                int_count: 14,
                int_points: 280,
                str_value: "14".to_string(),
            }],
        }];

        let quest = SavedGameTask {
            task_id: "esc_main_find_sidorovich".to_string(),
            task_state: 1,
            task_type: 0,
            receive_time: dt.to_game_time_ms(),
            finish_time: 0,
            time_to_complete: 0,
            timer_finish: 0,
            title: "Talk to Sidorovich".to_string(),
            description: "Visit the bunker trader in Cordon.".to_string(),
            script_helper: SavedTaskScriptHelper::default(),
            icon_texture_name: "ui_inGame2_PD_trader".to_string(),
            map_hint: "Sidorovich".to_string(),
            map_location: "blue_location".to_string(),
            map_object_id: 20,
            priority: 10,
        };

        let registries = SaveRegistries {
            info_portions: vec![(0, vec!["esc_trader_met".to_string(), "coop_host_ready".to_string()])],
            character_relations: vec![(0, RelationData { personal: vec![(10, 650)], communities: vec![(1, 200)] })],
            game_news: vec![(
                0,
                vec![GameNewsEntry {
                    news_type: 1,
                    caption: "Wolf".to_string(),
                    text: "Co-op relay online near the village.".to_string(),
                    receive_time: dt.to_game_time_ms(),
                    texture_name: "ui_inGame2_Hero".to_string(),
                }],
            )],
            specific_characters: vec![("esc_wolf".to_string(), 1)],
            map_locations: vec![(
                0,
                vec![SavedMapLocation {
                    object_id: 40,
                    spot_type: "treasure".to_string(),
                    reserved_flag: 0,
                    hint: "Wolf Stash".to_string(),
                    flags: 1,
                    owner_task_id: String::new(),
                }],
            )],
            game_tasks: GameTaskRegistryData {
                tasks_by_owner: vec![(0, vec![quest.clone()])],
                active_task_id: "esc_main_find_sidorovich".to_string(),
            },
            actor_statistics: vec![(0, actor_stat_sections.clone())],
            trailing_bytes: Vec::new(),
        };

        ScopSaveFile {
            header: ScopHeader {
                alife_version: SCOP_ALIFE_VERSION,
                format_version: SCOP_FORMAT_VERSION,
                spawn_name: "all".to_string(),
                spawn_guid: [0x42; 16],
                level_id: Some(1),
                level_name: "l01_escape".to_string(),
                game_time: SaveGameTime {
                    game_time_ms: dt.to_game_time_ms(),
                    time_factor: 10.0,
                    normal_time_factor: 10.0,
                },
                actor_stats: ScopActorStats {
                    actor_name: "Zilla".to_string(),
                    profile_name: "actor".to_string(),
                    community_index: 0,
                    health: 0.92,
                    radiation: 0.04,
                    stamina: 0.87,
                    bleeding: 0.0,
                    money: 48500,
                    rank: 1450,
                    reputation: 320,
                    position: [-180.5, 2.1, -155.0],
                    angles: [0.0, 1.2, 0.0],
                    game_vertex_id: 42,
                    level_vertex_id: 12000,
                    holder_id: INVALID_OBJECT_ID,
                    statistics: actor_stat_sections,
                },
            },
            entities,
            graph_and_quests: ScopGraphAndQuests {
                graph_mappings: vec![ScopGraphMapping {
                    entity_id: 0,
                    spawn_id: 1,
                    game_vertex_id: 42,
                    level_vertex_id: 12000,
                    level_id: 1,
                    level_name: "l01_escape".to_string(),
                    story_id: 0,
                    spawn_vertex_id: Some(42),
                    spawn_update_bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
                }],
                story_quests: vec![ScopStoryQuest {
                    owner_id: 0,
                    task: quest,
                }],
                active_quest_id: "esc_main_find_sidorovich".to_string(),
                info_portions: vec![(0, vec!["esc_trader_met".to_string(), "coop_host_ready".to_string()])],
                registries,
            },
            weather: ScopWeatherState {
                current_weather: "w_storm2".to_string(),
                next_weather: "w_cloudy1".to_string(),
                weather_fx: "fx_surge_day_3".to_string(),
                weather_fx_time: 12.5,
                cycle_time_sec: 16.0 * 3600.0 + 45.0 * 60.0 + 12.5,
                rain_factor: 0.65,
                rain_wetness: 0.42,
                wind_velocity: 7.5,
                wind_direction_rad: 1.05,
                fog_density: 0.28,
                thunder_active: true,
                surge_active: true,
                psi_storm_active: false,
            },
            extra_chunks: Vec::new(),
        }
    }

    #[test]
    fn test_scop_chunk0_to_3_lzo_roundtrip_and_queries() {
        let original = build_rich_sample_save();
        let bytes = original.to_bytes().unwrap();

        let (hdr_ver, uncomp_len) = validate_scop_header(&bytes).unwrap();
        assert_eq!(hdr_ver, SCOP_ALIFE_VERSION);
        assert!(uncomp_len > 0);

        let parsed = ScopSaveFile::parse(&bytes).unwrap();
        assert_eq!(parsed, original);

        // Typed query helpers
        assert_eq!(parsed.header.level_name, "l01_escape");
        assert_eq!(parsed.header.actor_stats.money, 48500);
        assert_eq!(
            parsed
                .header
                .actor_stats
                .find_stat_count("total", "stalkerkills"),
            Some(14)
        );
        assert_eq!(parsed.actor().unwrap().id(), 0);
        assert_eq!(parsed.stalkers().count(), 1);
        assert_eq!(parsed.mutants().count(), 1);
        assert_eq!(parsed.smart_terrains().count(), 1);
        assert_eq!(parsed.physics_restrictors().count(), 2);
        assert_eq!(parsed.space_restrictors().count(), 1);
        assert_eq!(parsed.physics_objects().count(), 1);
        assert_eq!(parsed.inventory_boxes().count(), 1);
        assert_eq!(parsed.find_by_story_id(501).unwrap().name(), "esc_wolf_stash");

        let smart = match &parsed.smart_terrains().next().unwrap().payload {
            ScopAlifePayload::SmartTerrain(st) => st,
            other => panic!("expected SmartTerrain, got {other:?}"),
        };
        assert_eq!(smart.max_population, 12);
        assert_eq!(smart.squad_id, 3);
        assert_eq!(smart.current_population(), 2);

        assert_eq!(
            parsed.active_quest().unwrap().status(),
            ScopQuestStatus::InProgress
        );
        assert!(parsed.has_info_portion(0, "coop_host_ready"));
        assert!((parsed.weather.wetness_state().factor - 0.42).abs() < 1e-6);
        assert!(parsed.weather.thunder_active && parsed.weather.surge_active);
    }

    #[test]
    fn test_scop_canonical_anomaly_savegame_conversion_roundtrip() {
        let original = build_rich_sample_save();

        // Serialize as canonical `CALifeStorageManager` LZO .scop (0x0000/0x0001/0x0002/0x0005/0x0009 + 0x000A)
        let anomaly_bytes = original.to_anomaly_scop_bytes().unwrap();

        // Parse via legacy `SavegameFile` and also auto-detect via `ScopSaveFile::parse`
        let raw_save = SavegameFile::parse(&anomaly_bytes).unwrap();
        assert_eq!(raw_save.objects.len(), 8);
        assert_eq!(raw_save.spawn_info.level_name.as_deref(), Some("l01_escape"));

        let converted = ScopSaveFile::parse(&anomaly_bytes).unwrap();
        assert_eq!(converted.header.level_name, original.header.level_name);
        assert_eq!(converted.header.actor_stats.money, 48500);
        assert_eq!(converted.header.actor_stats.actor_name, "Zilla");
        assert_eq!(converted.entities.len(), 8);
        assert_eq!(converted.stalkers().count(), 1);
        assert_eq!(converted.mutants().count(), 1);
        assert_eq!(converted.smart_terrains().count(), 1);
        assert_eq!(converted.physics_restrictors().count(), 2);
        assert_eq!(converted.inventory_boxes().count(), 1);
        assert_eq!(converted.weather, original.weather);
        assert_eq!(converted.graph_and_quests.active_quest_id, "esc_main_find_sidorovich");
        assert_eq!(converted.graph_and_quests.graph_mappings[0].spawn_vertex_id, Some(42));
    }

    #[test]
    fn test_scop_minimal_dynamic_object_fallback_and_errors() {
        // Construct a canonical SavegameFile whose stalker only carries bare AlifeObjectState bytes
        let bare_state = sample_base_alife(10, 500, 77, "[logic]\n").to_bytes();
        let pkt = SpawnPacket {
            section: "sim_default_stalker_0".to_string(),
            name_replace: "bare_stalker".to_string(),
            game_id_legacy: 0,
            rp: 0xFE,
            position: [1.0, 2.0, 3.0],
            angle: [0.0; 3],
            respawn_time: 0,
            id: 12,
            parent_id: INVALID_OBJECT_ID,
            phantom_id: INVALID_OBJECT_ID,
            flags: m_spawn_flags::VERSION,
            version: SPAWN_VERSION,
            game_type: u16::MAX,
            script_version: 10,
            client_data: Vec::new(),
            spawn_id: 9,
            state_payload: bare_state,
            trailing_update_payload: Vec::new(),
        };
        let save = SavegameFile {
            alife_version: SCOP_ALIFE_VERSION,
            time: SaveGameTime::default(),
            spawn_info: SaveSpawnHeader::default(),
            objects: vec![SaveObjectRecord {
                spawn: pkt,
                update_msg_id: M_UPDATE,
                update_payload: vec![0, 0, 0x80, 0x3F],
            }],
            registries: SaveRegistries::default(),
            extra_chunks: Vec::new(),
        };
        let scop_bytes = save.to_bytes().unwrap();
        let loaded = ScopSaveFile::parse(&scop_bytes).unwrap();
        assert_eq!(loaded.entities.len(), 1);
        assert!(matches!(
            loaded.entities[0].payload,
            ScopAlifePayload::DynamicObject(_)
        ));
        assert_eq!(loaded.entities[0].story_id(), 77);

        // Reject invalid header & missing chunks
        assert!(ScopSaveFile::parse(&[0u8; 8]).is_err());
        let bad_inner = {
            let mut cw = ChunkWriter::new();
            cw.write_chunk(SCOP_CHUNK_HEADER, &ScopHeader::default().to_bytes());
            cw.into_bytes()
        };
        let bad_outer = compress_scop(SCOP_ALIFE_VERSION, &bad_inner);
        assert!(ScopSaveFile::parse(&bad_outer).is_err());
    }

    #[test]
    fn test_scop_additional_subclasses_ownership_and_weather_sync() {
        use crate::spawns::{AlifeCustomZoneState, ServerShapeDef};

        let mut base = sample_base_alife(44, 14000, INVALID_STORY_ID, "");
        base.flags |= alife_object_flags::CAN_SAVE;

        let level_changer = AlifeLevelChangerState {
            restrictor: AlifeSpaceRestrictorState {
                alife: base.clone(),
                shapes: vec![ServerShapeDef::Sphere {
                    center: [0.0; 3],
                    radius: 3.0,
                }],
                restrictor_type: 0,
                trailing: Vec::new(),
            },
            next_graph_vertex: 312,
            next_level_vertex: 99000,
            next_position: [5.0, 1.0, -12.0],
            next_angles: [0.0, 1.57, 0.0],
            level_to_change: "l02_garbage".to_string(),
            level_point_to_change: "gar_start".to_string(),
            silent_mode: false,
            trailing: Vec::new(),
        };

        let anomaly = AlifeAnomalousZoneState {
            zone: AlifeCustomZoneState {
                restrictor: AlifeSpaceRestrictorState {
                    alife: base.clone(),
                    shapes: vec![ServerShapeDef::Sphere {
                        center: [0.0; 3],
                        radius: 4.5,
                    }],
                    restrictor_type: 1,
                    trailing: Vec::new(),
                },
                max_power: 2.2,
                owner_id: u32::MAX,
                enabled_time: 100,
                disabled_time: 50,
                start_time_shift: 10,
                trailing: Vec::new(),
            },
            offline_interactive_radius: 30.0,
            artefact_spawn_count: 8,
            artefact_position_offset: 32,
            trailing: Vec::new(),
        };

        let smart_cover = AlifeSmartCoverState {
            alife: base.clone(),
            shapes: vec![ServerShapeDef::Sphere {
                center: [0.0; 3],
                radius: 1.5,
            }],
            description: "combat_front".to_string(),
            hold_position_time: 8.0,
            enter_min_enemy_distance: 5.0,
            exit_min_enemy_distance: 2.0,
            is_combat_cover: true,
            can_fire: true,
            trailing: Vec::new(),
        };

        let squad = AlifeOnlineOfflineGroupState {
            alife: base.clone(),
            member_ids: vec![10],
            trailing: Vec::new(),
        };

        let mut weapon = ScopAlifeEntity::new(
            60,
            "wpn_ak74",
            "wpn_ak74_stash",
            [0.0; 3],
            [0.0; 3],
            ScopAlifePayload::Weapon(AlifeItemWeaponState {
                item: AlifeInventoryItemState {
                    base: sample_dyn_vis(44, 14000, INVALID_STORY_ID, "weapons\\ak74", ""),
                    condition: 0.88,
                    upgrades: vec!["up_firsta_ak74".to_string()],
                    trailing: Vec::new(),
                },
                ammo_current: 90,
                ammo_elapsed: 30,
                wpn_state: 0,
                addon_flags: 1,
                ammo_type: 0,
                ammo_elapsed_grenades: 0,
                trailing: Vec::new(),
            }),
        );
        weapon.spawn.parent_id = 40;

        let mut medkit = ScopAlifeEntity::new(
            61,
            "medkit_army",
            "medkit_army_stash",
            [0.0; 3],
            [0.0; 3],
            ScopAlifePayload::InventoryItem(AlifeInventoryItemState {
                base: sample_dyn_vis(44, 14000, INVALID_STORY_ID, "dynamics\\devices\\dev_aptechka", ""),
                condition: 1.0,
                upgrades: Vec::new(),
                trailing: Vec::new(),
            }),
        );
        medkit.spawn.parent_id = 40;

        let mut save = build_rich_sample_save();
        save.entities.push(ScopAlifeEntity::new(
            51,
            "level_changer",
            "esc_to_garbage",
            [0.0, 0.0, 250.0],
            [0.0; 3],
            ScopAlifePayload::LevelChanger(level_changer),
        ));
        save.entities.push(ScopAlifeEntity::new(
            52,
            "zone_mosquito_bald",
            "esc_anomaly_1",
            [-50.0, 1.0, 20.0],
            [0.0; 3],
            ScopAlifePayload::AnomalousZone(anomaly),
        ));
        save.entities.push(ScopAlifeEntity::new(
            53,
            "smart_cover",
            "esc_cover_01",
            [-170.0, 2.0, -142.0],
            [0.0; 3],
            ScopAlifePayload::SmartCover(smart_cover),
        ));
        save.entities.push(ScopAlifeEntity::new(
            54,
            "sim_squad_novice",
            "esc_village_squad",
            [-175.0, 2.0, -145.0],
            [0.0; 3],
            ScopAlifePayload::OnlineOfflineGroup(squad),
        ));
        save.entities.push(weapon);
        save.entities.push(medkit);

        let bytes = save.to_bytes().unwrap();
        let parsed = ScopSaveFile::parse(&bytes).unwrap();
        assert_eq!(parsed, save);
        assert_eq!(parsed.physics_restrictors().count(), 4);
        assert_eq!(parsed.entities_owned_by(40).count(), 2);

        let mixed = MixedEnvironment {
            weight: 0.5,
            from_id: "06:00:00".to_string(),
            to_id: "07:00:00".to_string(),
            sky_texture_a: String::new(),
            sky_texture_b: String::new(),
            sky_texture_env_a: String::new(),
            sky_texture_env_b: String::new(),
            clouds_texture_a: String::new(),
            clouds_texture_b: String::new(),
            clouds_color: [1.0; 4],
            sky_color: [1.0; 3],
            sky_rotation: 0.0,
            far_plane: 500.0,
            fog_color: [0.5; 3],
            fog_density: 0.35,
            fog_distance: 400.0,
            fog_near: 10.0,
            fog_far: 500.0,
            rain_density: 0.8,
            rain_color: [1.0; 3],
            bolt_period: 10.0,
            bolt_duration: 0.5,
            wind_velocity: 12.0,
            wind_direction: 0.75,
            ambient_color: [0.2; 3],
            hemi_color: [0.3; 4],
            sun_color: [1.0; 3],
            sun_dir: [0.0, -1.0, 0.0],
            sun_shafts_intensity: 0.5,
            water_intensity: 1.0,
            hemi_vibrance: 1.0,
            hemi_contrast: 1.0,
            wet_surfaces: 0.62,
            volumetric_intensity_factor: 1.0,
            volumetric_distance_factor: 1.0,
            tree_amplitude_intensity: 0.01,
            bloom_threshold: 1.0,
            bloom_exposure: 1.0,
            bloom_sky_intensity: 1.0,
            active_sun_id: "sun_default".to_string(),
            active_thunderbolt_id: "thunder_01".to_string(),
        };
        let mut w = ScopWeatherState::default();
        w.apply_mixed_environment(&mixed);
        assert!((w.rain_factor - 0.8).abs() < 1e-6);
        assert!((w.rain_wetness - 0.62).abs() < 1e-6);
        assert!((w.wind_velocity - 12.0).abs() < 1e-6);
        assert!((w.fog_density - 0.35).abs() < 1e-6);
        assert!(w.thunder_active);
    }
}
