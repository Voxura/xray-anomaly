//! ALife World Entity Spawn Manager & Savegame State Synchronizer (`xray-engine`).
//!
//! Wires host `.scop` world saves ([`ScopSaveFile`]), level static spawns ([`LevelSpawnData`]),
//! and co-op peer `.scoc` inventory/state saves ([`ScocFile`] / [`ScocPeerState`]) into the live
//! fixed-timestep simulation loop, generating and consuming [`xray_net::EntityDelta`] packets.

use std::collections::BTreeMap;

use crate::world_loader::WorldWeatherState;
use xray_formats::chunk::FormatError;
use xray_formats::savegames::GameDateTime;
use xray_formats::scoc::{ScocFile, ScocInventoryItem, ScocPeerState};
use xray_formats::scop::{
    classify_alife_section, ScopActorStats, ScopAlifeEntity, ScopAlifeKind, ScopAlifePayload,
    ScopGraphAndQuests, ScopHeader, ScopSaveFile, ScopWeatherState,
};
use xray_formats::spawns::{
    alife_object_flags, AlifeObjectState, LevelSpawnData, INVALID_OBJECT_ID, INVALID_STORY_ID,
};
use xray_formats::vfs_bridge::{load_scoc_from_vfs, load_scop_save_from_vfs, AssetLoadError};
use xray_net::EntityDelta;
use xray_vfs::VirtualFileSystem;

/// High-level simulation role of a live entity inside [`AlifeManager`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityRole {
    LocalHostActor,
    RemotePeerActor { slot: u8, nickname: String },
    Stalker,
    Mutant,
    SmartTerrain,
    PhysicsObject,
    SpaceRestrictor,
    InventoryBox,
    LevelChanger,
    AnomalousZone,
    SmartCover,
    OnlineOfflineGroup,
    Trader,
    Weapon,
    InventoryItem,
    DynamicObject,
}

impl EntityRole {
    pub fn from_scop_kind(kind: ScopAlifeKind) -> Self {
        match kind {
            ScopAlifeKind::Actor => Self::LocalHostActor,
            ScopAlifeKind::Stalker => Self::Stalker,
            ScopAlifeKind::Mutant => Self::Mutant,
            ScopAlifeKind::SmartTerrain => Self::SmartTerrain,
            ScopAlifeKind::PhysicsObject => Self::PhysicsObject,
            ScopAlifeKind::SpaceRestrictor => Self::SpaceRestrictor,
            ScopAlifeKind::InventoryBox => Self::InventoryBox,
            ScopAlifeKind::LevelChanger => Self::LevelChanger,
            ScopAlifeKind::AnomalousZone => Self::AnomalousZone,
            ScopAlifeKind::SmartCover => Self::SmartCover,
            ScopAlifeKind::OnlineOfflineGroup => Self::OnlineOfflineGroup,
            ScopAlifeKind::Trader => Self::Trader,
            ScopAlifeKind::Weapon => Self::Weapon,
            ScopAlifeKind::InventoryItem => Self::InventoryItem,
            ScopAlifeKind::DynamicObject => Self::DynamicObject,
        }
    }
}

/// Live runtime entity spawned from `.scop`, `level.spawn`, or co-op `.scoc` peer connection.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveAlifeEntity {
    pub id: u16,
    pub parent_id: u16,
    pub story_id: u32,
    pub name: String,
    pub section: String,
    pub role: EntityRole,
    pub kind: ScopAlifeKind,
    pub position: [f32; 3],
    pub angle: [f32; 3],
    pub velocity: [f32; 3],
    pub health: f32,
    pub online: bool,
    pub game_vertex_id: u16,
    pub level_vertex_id: u32,
    pub anim_state: u16,
    pub weapon_state: u16,
    pub visual_name: Option<String>,
    pub dirty_net_sync: bool,
    pub source_scop: Option<ScopAlifeEntity>,
}

impl LiveAlifeEntity {
    /// Convert this entity's current transform, health, and anim state into a network delta.
    #[inline]
    pub fn to_entity_delta(&self, tick: u32) -> EntityDelta {
        EntityDelta {
            tick,
            entity_id: self.id,
            pos: self.position,
            yaw: self.angle[1],
            pitch: self.angle[0],
            vel: self.velocity,
            anim_state: self.anim_state,
            weapon_state: self.weapon_state,
            health: self.health,
        }
    }

    /// Construct a `LiveAlifeEntity` from a decoded host `.scop` ALife record.
    pub fn from_scop_entity(ent: &ScopAlifeEntity) -> Self {
        let visual_name = ent
            .payload
            .visual()
            .map(|v| v.visual_name.clone())
            .filter(|s| !s.is_empty());
        Self {
            id: ent.id(),
            parent_id: ent.parent_id(),
            story_id: ent.story_id(),
            name: ent.name().to_string(),
            section: ent.section().to_string(),
            role: EntityRole::from_scop_kind(ent.kind()),
            kind: ent.kind(),
            position: ent.position(),
            angle: ent.angle(),
            velocity: [0.0; 3],
            health: ent.health().unwrap_or(1.0),
            online: ent.is_online(),
            game_vertex_id: ent.game_vertex_id(),
            level_vertex_id: ent.level_vertex_id(),
            anim_state: 0,
            weapon_state: 0,
            visual_name,
            dirty_net_sync: false,
            source_scop: Some(ent.clone()),
        }
    }

    /// Synchronize mutated transform/health back into the preserved [`ScopAlifeEntity`] record.
    pub fn to_scop_entity(&self) -> Option<ScopAlifeEntity> {
        let mut cloned = self.source_scop.clone()?;
        cloned.spawn.position = self.position;
        cloned.spawn.angle = self.angle;
        cloned.spawn.parent_id = self.parent_id;
        match &mut cloned.payload {
            ScopAlifePayload::Actor(s) => s.creature.health = self.health,
            ScopAlifePayload::Stalker(s) => s.human.monster.creature.health = self.health,
            ScopAlifePayload::Mutant(s) => s.monster.creature.health = self.health,
            _ => {}
        }
        cloned.sync_state_payload();
        Some(cloned)
    }
}

/// Connected co-op peer state backed by a peer `.scoc` save (`ScocPeerState`).
#[derive(Debug, Clone, PartialEq)]
pub struct RemotePeerAvatar {
    pub slot: u8,
    pub entity_id: u16,
    pub nickname: String,
    pub lan_id: u64,
    pub peer_state: ScocPeerState,
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub velocity: [f32; 3],
    pub health: f32,
    pub anim_state: u16,
    pub weapon_state: u16,
    pub last_sync_tick: u32,
}

impl RemotePeerAvatar {
    /// Count peer inventory items loaded from `.scoc`.
    #[inline]
    pub fn inventory(&self) -> &[ScocInventoryItem] {
        &self.peer_state.inventory
    }

    /// Preferred `.ogf` visual path from `head_tbl.visual` (if present in peer `.scoc`).
    #[inline]
    pub fn visual_path(&self) -> Option<&str> {
        self.peer_state.head_tbl.visual.as_deref()
    }
}

/// Live ALife world state, host actor stats, quests, spawned objects, and co-op peer registry.
#[derive(Debug, Clone)]
pub struct AlifeManager {
    pub tick: u32,
    pub alife_version: u32,
    pub format_version: u32,
    pub spawn_name: String,
    pub spawn_guid: [u8; 16],
    pub level_id: Option<u8>,
    pub level_name: String,
    pub game_time_ms: u64,
    pub time_factor: f32,
    pub normal_time_factor: f32,
    pub host_actor_id: Option<u16>,
    pub actor_stats: ScopActorStats,
    pub graph_and_quests: ScopGraphAndQuests,
    pub weather_snapshot: ScopWeatherState,
    pub extra_chunks: Vec<(u32, Vec<u8>)>,
    pub entities: BTreeMap<u16, LiveAlifeEntity>,
    pub story_id_index: BTreeMap<u32, u16>,
    pub remote_peers: BTreeMap<u16, RemotePeerAvatar>,
    next_dynamic_id: u16,
}

impl Default for AlifeManager {
    fn default() -> Self {
        Self {
            tick: 0,
            alife_version: xray_formats::scop::SCOP_ALIFE_VERSION,
            format_version: xray_formats::scop::SCOP_FORMAT_VERSION,
            spawn_name: "all".to_string(),
            spawn_guid: [0u8; 16],
            level_id: Some(0),
            level_name: "l01_escape".to_string(),
            game_time_ms: 0,
            time_factor: 10.0,
            normal_time_factor: 10.0,
            host_actor_id: None,
            actor_stats: ScopActorStats::default(),
            graph_and_quests: ScopGraphAndQuests::default(),
            weather_snapshot: ScopWeatherState::default(),
            extra_chunks: Vec::new(),
            entities: BTreeMap::new(),
            story_id_index: BTreeMap::new(),
            remote_peers: BTreeMap::new(),
            next_dynamic_id: 50_000,
        }
    }
}

impl AlifeManager {
    /// Construct a fresh manager from a parsed host [`ScopSaveFile`].
    pub fn from_scop(scop: &ScopSaveFile) -> Self {
        let mut mgr = Self {
            tick: 0,
            alife_version: scop.header.alife_version,
            format_version: scop.header.format_version,
            spawn_name: scop.header.spawn_name.clone(),
            spawn_guid: scop.header.spawn_guid,
            level_id: scop.header.level_id,
            level_name: scop.header.level_name.clone(),
            game_time_ms: scop.header.game_time.game_time_ms,
            time_factor: scop.header.game_time.time_factor,
            normal_time_factor: scop.header.game_time.normal_time_factor,
            host_actor_id: None,
            actor_stats: scop.header.actor_stats.clone(),
            graph_and_quests: scop.graph_and_quests.clone(),
            weather_snapshot: scop.weather.clone(),
            extra_chunks: scop.extra_chunks.clone(),
            entities: BTreeMap::new(),
            story_id_index: BTreeMap::new(),
            remote_peers: BTreeMap::new(),
            next_dynamic_id: 50_000,
        };

        for ent in &scop.entities {
            let live = LiveAlifeEntity::from_scop_entity(ent);
            mgr.insert_entity(live);
        }

        mgr
    }

    /// Parse compressed or uncompressed `.scop` bytes into a live [`AlifeManager`].
    pub fn from_scop_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        let scop = ScopSaveFile::parse(bytes)?;
        Ok(Self::from_scop(&scop))
    }

    /// Stream a `.scop` host savegame directly from [`VirtualFileSystem`].
    pub fn load_from_vfs(
        vfs: &VirtualFileSystem,
        scop_vpath: &str,
    ) -> Result<Self, AssetLoadError> {
        let scop = load_scop_save_from_vfs(vfs, scop_vpath)?;
        Ok(Self::from_scop(&scop))
    }

    /// Insert or replace an entity while maintaining `host_actor_id`, `story_id_index`, and `next_dynamic_id`.
    pub fn insert_entity(&mut self, entity: LiveAlifeEntity) {
        if matches!(entity.role, EntityRole::LocalHostActor) {
            self.host_actor_id = Some(entity.id);
        }
        if entity.story_id != INVALID_STORY_ID {
            self.story_id_index.insert(entity.story_id, entity.id);
        }
        if entity.id >= self.next_dynamic_id && entity.id < INVALID_OBJECT_ID {
            self.next_dynamic_id = entity.id.saturating_add(1);
        }
        self.entities.insert(entity.id, entity);
    }

    /// Allocate the next free entity ID below [`INVALID_OBJECT_ID`] (`0xFFFF`).
    pub fn allocate_entity_id(&mut self) -> u16 {
        let mut candidate = self.next_dynamic_id.max(1);
        while self.entities.contains_key(&candidate) || candidate == INVALID_OBJECT_ID {
            candidate = candidate.wrapping_add(1);
            if candidate == 0 {
                candidate = 1;
            }
        }
        self.next_dynamic_id = candidate.saturating_add(1);
        candidate
    }

    /// Seed initial world entities from a level's static `level.spawn` packets when starting without `.scop`.
    pub fn seed_from_level_spawn(&mut self, level_name: &str, spawn: &LevelSpawnData) {
        self.level_name = level_name.to_string();
        for pkt in &spawn.spawns {
            let kind = classify_alife_section(pkt.id, &pkt.section);
            let decoded_state = pkt.decode_alife_state().unwrap_or(AlifeObjectState {
                game_vertex_id: 0,
                distance: 0.0,
                direct_control: true,
                level_vertex_id: 0,
                flags: alife_object_flags::VISIBLE_FOR_AI | alife_object_flags::SWITCH_ONLINE,
                custom_data: String::new(),
                story_id: INVALID_STORY_ID,
                spawn_story_id: INVALID_STORY_ID,
                subclass_payload: Vec::new(),
            });
            let entity_id = if pkt.id == INVALID_OBJECT_ID || self.entities.contains_key(&pkt.id) {
                self.allocate_entity_id()
            } else {
                pkt.id
            };
            let online = (decoded_state.flags & alife_object_flags::SWITCH_ONLINE) != 0
                || kind == ScopAlifeKind::Actor;
            let live = LiveAlifeEntity {
                id: entity_id,
                parent_id: pkt.parent_id,
                story_id: decoded_state.story_id,
                name: pkt.name_replace.clone(),
                section: pkt.section.clone(),
                role: EntityRole::from_scop_kind(kind),
                kind,
                position: pkt.position,
                angle: pkt.angle,
                velocity: [0.0; 3],
                health: 1.0,
                online,
                game_vertex_id: decoded_state.game_vertex_id,
                level_vertex_id: decoded_state.level_vertex_id,
                anim_state: 0,
                weapon_state: 0,
                visual_name: None,
                dirty_net_sync: false,
                source_scop: None,
            };
            self.insert_entity(live);
        }
    }

    /// Attach a connected co-op peer's `.scoc` state and spawn/update their `RemotePeerActor` entity.
    pub fn attach_peer_scoc(
        &mut self,
        slot: u8,
        nickname: impl Into<String>,
        lan_id: u64,
        scoc: &ScocFile,
    ) -> u16 {
        let nickname = nickname.into();
        let peer_state = scoc.extract_peer_state();

        // Reuse existing avatar entity if this co-op slot reconnects.
        if let Some((&existing_id, _)) = self
            .remote_peers
            .iter()
            .find(|(_, p)| p.slot == slot)
        {
            if let Some(avatar) = self.remote_peers.get_mut(&existing_id) {
                avatar.nickname = nickname.clone();
                avatar.lan_id = lan_id;
                avatar.peer_state = peer_state.clone();
                avatar.last_sync_tick = self.tick;
            }
            if let Some(ent) = self.entities.get_mut(&existing_id) {
                ent.name = nickname.clone();
                ent.role = EntityRole::RemotePeerActor {
                    slot,
                    nickname,
                };
                ent.visual_name = peer_state.head_tbl.visual.clone();
                ent.online = true;
            }
            return existing_id;
        }

        let entity_id = self.allocate_entity_id();
        let spawn_pos = self
            .host_actor()
            .map(|a| [a.position[0] + 1.5 * f32::from(slot), a.position[1], a.position[2]])
            .unwrap_or([0.0, 0.0, 0.0]);
        let visual_name = peer_state.head_tbl.visual.clone();

        let avatar = RemotePeerAvatar {
            slot,
            entity_id,
            nickname: nickname.clone(),
            lan_id,
            peer_state,
            position: spawn_pos,
            yaw: 0.0,
            pitch: 0.0,
            velocity: [0.0; 3],
            health: 1.0,
            anim_state: 0,
            weapon_state: 0,
            last_sync_tick: self.tick,
        };
        self.remote_peers.insert(entity_id, avatar);

        let entity = LiveAlifeEntity {
            id: entity_id,
            parent_id: INVALID_OBJECT_ID,
            story_id: INVALID_STORY_ID,
            name: nickname.clone(),
            section: "actor_coop_peer".to_string(),
            role: EntityRole::RemotePeerActor { slot, nickname },
            kind: ScopAlifeKind::Stalker,
            position: spawn_pos,
            angle: [0.0; 3],
            velocity: [0.0; 3],
            health: 1.0,
            online: true,
            game_vertex_id: self.host_actor().map_or(0, |a| a.game_vertex_id),
            level_vertex_id: self.host_actor().map_or(0, |a| a.level_vertex_id),
            anim_state: 0,
            weapon_state: 0,
            visual_name,
            dirty_net_sync: true,
            source_scop: None,
        };
        self.insert_entity(entity);
        entity_id
    }

    /// Parse raw `.scoc` bytes and attach the peer avatar into this manager.
    pub fn attach_peer_scoc_bytes(
        &mut self,
        slot: u8,
        nickname: impl Into<String>,
        lan_id: u64,
        bytes: &[u8],
    ) -> Result<u16, FormatError> {
        let scoc = ScocFile::parse(bytes)?;
        Ok(self.attach_peer_scoc(slot, nickname, lan_id, &scoc))
    }

    /// Stream a `.scoc` file from VFS and attach the peer avatar into this manager.
    pub fn attach_peer_scoc_from_vfs(
        &mut self,
        vfs: &VirtualFileSystem,
        scoc_vpath: &str,
        slot: u8,
        nickname: impl Into<String>,
        lan_id: u64,
    ) -> Result<u16, AssetLoadError> {
        let scoc = load_scoc_from_vfs(vfs, scoc_vpath)?;
        Ok(self.attach_peer_scoc(slot, nickname, lan_id, &scoc))
    }

    /// Update the local host actor transform from the first-person camera controller.
    pub fn set_host_camera_transform(
        &mut self,
        position: [f32; 3],
        yaw: f32,
        pitch: f32,
        velocity: [f32; 3],
    ) {
        self.actor_stats.position = position;
        self.actor_stats.angles = [pitch, yaw, 0.0];
        let Some(actor_id) = self.host_actor_id else {
            return;
        };
        if let Some(actor) = self.entities.get_mut(&actor_id) {
            let moved = (actor.position[0] - position[0]).abs() > 1e-4
                || (actor.position[1] - position[1]).abs() > 1e-4
                || (actor.position[2] - position[2]).abs() > 1e-4
                || (actor.angle[1] - yaw).abs() > 1e-4
                || (actor.angle[0] - pitch).abs() > 1e-4;
            actor.position = position;
            actor.angle = [pitch, yaw, 0.0];
            actor.velocity = velocity;
            if moved {
                actor.dirty_net_sync = true;
            }
        }
    }

    /// Drain all dirty entity transforms ready to transmit via [`xray_net::HostSession`] or `ClientSession`.
    pub fn collect_outgoing_deltas(&mut self) -> Vec<EntityDelta> {
        let tick = self.tick;
        let mut deltas = Vec::new();
        for ent in self.entities.values_mut() {
            if ent.dirty_net_sync {
                deltas.push(ent.to_entity_delta(tick));
                ent.dirty_net_sync = false;
            }
        }
        deltas
    }

    /// Apply an incoming network [`EntityDelta`] to a remote peer avatar or live ALife entity.
    pub fn apply_network_delta(&mut self, delta: &EntityDelta) -> bool {
        let mut applied = false;
        if let Some(avatar) = self.remote_peers.get_mut(&delta.entity_id) {
            avatar.position = delta.pos;
            avatar.yaw = delta.yaw;
            avatar.pitch = delta.pitch;
            avatar.velocity = delta.vel;
            avatar.health = delta.health;
            avatar.anim_state = delta.anim_state;
            avatar.weapon_state = delta.weapon_state;
            avatar.last_sync_tick = delta.tick;
            applied = true;
        }
        if let Some(ent) = self.entities.get_mut(&delta.entity_id) {
            ent.position = delta.pos;
            ent.angle = [delta.pitch, delta.yaw, 0.0];
            ent.velocity = delta.vel;
            ent.health = delta.health;
            ent.anim_state = delta.anim_state;
            ent.weapon_state = delta.weapon_state;
            applied = true;
        }
        applied
    }

    /// Advance the fixed-timestep ALife simulation tick and game clock.
    pub fn step(&mut self, fixed_dt: f32) {
        self.tick = self.tick.wrapping_add(1);
        if fixed_dt.is_finite() && fixed_dt > 0.0 {
            let delta_game_ms = (fixed_dt * self.time_factor * 1000.0).round().max(0.0) as u64;
            self.game_time_ms = self.game_time_ms.saturating_add(delta_game_ms);
        }

        for ent in self.entities.values_mut() {
            if !ent.online {
                continue;
            }
            if matches!(ent.role, EntityRole::LocalHostActor) {
                continue;
            }
            if ent.velocity != [0.0; 3] && fixed_dt.is_finite() && fixed_dt > 0.0 {
                ent.position[0] += ent.velocity[0] * fixed_dt;
                ent.position[1] += ent.velocity[1] * fixed_dt;
                ent.position[2] += ent.velocity[2] * fixed_dt;
                if !matches!(ent.role, EntityRole::RemotePeerActor { .. }) {
                    ent.dirty_net_sync = true;
                }
            }
        }
    }

    /// Current calendar date/time derived from `game_time_ms`.
    pub fn game_date_time(&self) -> GameDateTime {
        GameDateTime::from_game_time_ms(self.game_time_ms)
    }

    /// Borrow the local host actor entity (`ID = 0` or `ScopAlifeKind::Actor`).
    pub fn host_actor(&self) -> Option<&LiveAlifeEntity> {
        self.host_actor_id.and_then(|id| self.entities.get(&id))
    }

    /// Borrow any live entity by ALife object ID.
    pub fn entity(&self, id: u16) -> Option<&LiveAlifeEntity> {
        self.entities.get(&id)
    }

    /// Lookup an entity by its unique story ID.
    pub fn find_by_story_id(&self, story_id: u32) -> Option<&LiveAlifeEntity> {
        let id = *self.story_id_index.get(&story_id)?;
        self.entities.get(&id)
    }

    /// Lookup a connected co-op peer avatar by assigned network slot.
    pub fn peer_by_slot(&self, slot: u8) -> Option<&RemotePeerAvatar> {
        self.remote_peers.values().find(|p| p.slot == slot)
    }

    /// Count currently online ALife entities.
    pub fn online_entity_count(&self) -> usize {
        self.entities.values().filter(|e| e.online).count()
    }

    /// Return all live entities matching `kind`.
    pub fn entities_by_kind(&self, kind: ScopAlifeKind) -> Vec<&LiveAlifeEntity> {
        self.entities.values().filter(|e| e.kind == kind).collect()
    }

    /// Export live world, host actor, quest, and weather state back into a [`ScopSaveFile`].
    pub fn export_to_scop(&self, weather: Option<&WorldWeatherState>) -> ScopSaveFile {
        let mut header = ScopHeader {
            alife_version: self.alife_version,
            format_version: self.format_version,
            spawn_name: self.spawn_name.clone(),
            spawn_guid: self.spawn_guid,
            level_id: self.level_id,
            level_name: self.level_name.clone(),
            game_time: xray_formats::savegames::SaveGameTime {
                game_time_ms: self.game_time_ms,
                time_factor: self.time_factor,
                normal_time_factor: self.normal_time_factor,
            },
            actor_stats: self.actor_stats.clone(),
        };
        if let Some(actor) = self.host_actor() {
            header.actor_stats.position = actor.position;
            header.actor_stats.angles = actor.angle;
            header.actor_stats.health = actor.health;
        }

        let mut scop_entities = Vec::new();
        for ent in self.entities.values() {
            if matches!(ent.role, EntityRole::RemotePeerActor { .. }) {
                // Co-op peer state persists to each client's `.scoc`, not host world `.scop`.
                continue;
            }
            if let Some(updated) = ent.to_scop_entity() {
                scop_entities.push(updated);
            } else {
                let fallback_alife = AlifeObjectState {
                    game_vertex_id: ent.game_vertex_id,
                    distance: 0.0,
                    direct_control: true,
                    level_vertex_id: ent.level_vertex_id,
                    flags: if ent.online {
                        alife_object_flags::SWITCH_ONLINE | alife_object_flags::VISIBLE_FOR_AI
                    } else {
                        alife_object_flags::VISIBLE_FOR_AI
                    },
                    custom_data: String::new(),
                    story_id: ent.story_id,
                    spawn_story_id: INVALID_STORY_ID,
                    subclass_payload: Vec::new(),
                };
                scop_entities.push(ScopAlifeEntity::new(
                    ent.id,
                    ent.section.clone(),
                    ent.name.clone(),
                    ent.position,
                    ent.angle,
                    ScopAlifePayload::DynamicObject(fallback_alife),
                ));
            }
        }

        let weather_state = weather
            .map(WorldWeatherState::to_scop_weather)
            .unwrap_or_else(|| self.weather_snapshot.clone());

        let mut quests = self.graph_and_quests.clone();
        quests.sync_registries();

        ScopSaveFile {
            header,
            entities: scop_entities,
            graph_and_quests: quests,
            weather: weather_state,
            extra_chunks: self.extra_chunks.clone(),
        }
    }

    /// Export a connected peer's live [`ScocPeerState`] back into a [`ScocFile`].
    pub fn export_peer_scoc(&self, entity_id: u16) -> Option<ScocFile> {
        let peer = self.remote_peers.get(&entity_id)?;
        Some(ScocFile::from_peer_state(&peer.peer_state))
    }
}
