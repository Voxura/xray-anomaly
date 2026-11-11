//! Pure Rust X-Ray `level.game` (RPoints & PatrolPaths), `patrol_paths.ltx`, `level.spawn`,
//! `all.spawn` (`CALifeSpawnRegistry`), and `NET_Packet` Spawn/ALife State Parsers
//!
//! Covers:
//! - `level.game` (`RPOINT_CHUNK = 0x2000` incl. `rptItemSpawn` `rp_profile`, `WAY_PATROLPATH_CHUNK = 0x1000`
//!   `WAYOBJECT_VERSION = 0x0013` points/links)
//! - Anomaly `$game_configs$/patrol_paths.ltx` section parser (`points`, `<pt>:name/position/flags/level_vertex_id/game_vertex_id/links`)
//! - `level.spawn` raw `M_SPAWN` packet list (`ChunkReader` over chunks `0..`)
//! - `NET_Packet` `SpawnPacket` (`M_SPAWN`, `CSE_Abstract`, `AlifeObjectState`, `GraphPointState`, legacy v1..v128)
//! - `all.spawn` (`CALifeSpawnRegistry`): `SpawnHeader` (Chunk 0), `SPAWN_GRAPH` (`CGraphAbstractSerialize` w/ `CServerEntityWrapper`
//!   Chunk 1), `artefact_spawn_positions` (Chunk 2, `GameDeathPoint`), `CPatrolPathStorage` (Chunk 3 w/ `u32` vertex IDs),
//!   and embedded `GameGraph` (Chunk 4)

use std::collections::HashMap;

use xray_core::LtxDocument;
use xray_core::ltx_loader::OwnedLtxDocument;

use crate::ai_graph::{
    GameDeathPoint, GameGraph, INVALID_AI_VERTEX, INVALID_GAME_VERTEX, XRAI_CURRENT_VERSION,
    XRAI_LARGE_VERSION,
};
use crate::chunk::{ByteCursor, ChunkReader, ChunkWriter, FormatError};

pub const WAY_BASE: u32 = 0x1000;
pub const POINT_BASE: u32 = 0x2000;

pub const WAY_PATROLPATH_CHUNK: u32 = WAY_BASE;
pub const RPOINT_CHUNK: u32 = POINT_BASE;
pub const WAYOBJECT_VERSION: u16 = 0x0013;

pub const M_UPDATE: u16 = 0;
pub const M_SPAWN: u16 = 1;
pub const SPAWN_VERSION: u16 = 128;

pub const INVALID_OBJECT_ID: u16 = u16::MAX;
pub const INVALID_STORY_ID: u32 = u32::MAX;
pub const INVALID_SPAWN_STORY_ID: u32 = u32::MAX;

pub mod wayobject_chunks {
    pub const VERSION: u32 = 0x0001;
    pub const POINTS: u32 = 0x0002;
    pub const LINKS: u32 = 0x0003;
    pub const TYPE: u32 = 0x0004;
    pub const NAME: u32 = 0x0005;
}

/// Network `CSE_Abstract::s_flags` (`xrMessages.h:270-290`).
pub mod m_spawn_flags {
    pub const OBJECT_LOCAL: u16 = 1 << 0;
    pub const OBJECT_HASUPDATE: u16 = 1 << 2;
    pub const OBJECT_ASPLAYER: u16 = 1 << 3;
    pub const OBJECT_PHANTOM: u16 = 1 << 4;
    pub const VERSION: u16 = 1 << 5;
    pub const UPDATE: u16 = 1 << 6;
    pub const TIME: u16 = 1 << 7;
    pub const DENIED: u16 = 1 << 8;
}

/// `CSE_Abstract::ESpawnFlags` (`xrServer_Object_Base.h:54-62`).
pub mod cse_spawn_flags {
    pub const ENABLED: u32 = 1 << 0;
    pub const ON_SURGE_ONLY: u32 = 1 << 1;
    pub const SINGLE_ITEM_ONLY: u32 = 1 << 2;
    pub const IF_DESTROYED_ONLY: u32 = 1 << 3;
    pub const INFINITE_COUNT: u32 = 1 << 4;
    pub const DESTROY_ON_SPAWN: u32 = 1 << 5;
}

/// `CSE_ALifeObject` behavior flags (`xrServer_Objects_ALife.h:114-129`).
pub mod alife_object_flags {
    pub const USE_SWITCHES: u32 = 1 << 0;
    pub const SWITCH_ONLINE: u32 = 1 << 1;
    pub const SWITCH_OFFLINE: u32 = 1 << 2;
    pub const INTERACTIVE: u32 = 1 << 3;
    pub const VISIBLE_FOR_AI: u32 = 1 << 4;
    pub const USEFUL_FOR_AI: u32 = 1 << 5;
    pub const OFFLINE_NO_MOVE: u32 = 1 << 6;
    pub const USED_AI_LOCATIONS: u32 = 1 << 7;
    pub const GROUP_BEHAVIOUR: u32 = 1 << 8;
    pub const CAN_SAVE: u32 = 1 << 9;
    pub const VISIBLE_FOR_MAP: u32 = 1 << 10;
    pub const USE_SMART_TERRAINS: u32 = 1 << 11;
    pub const CHECK_FOR_SEPARATOR: u32 = 1 << 12;
}

/// `ERPpointType` (`LevelGameDef.h:24-30`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RPointType {
    ActorSpawn = 0,
    ArtefactSpawn = 1,
    ItemSpawn = 2,
}

impl RPointType {
    pub fn from_u8(v: u8) -> Result<Self, FormatError> {
        match v {
            0 => Ok(Self::ActorSpawn),
            1 => Ok(Self::ArtefactSpawn),
            2 => Ok(Self::ItemSpawn),
            _ => Err(FormatError::InvalidValue("unknown RPoint type")),
        }
    }
}

/// Level respawn / multiplayer spawn / item spawn marker in `level.game` (`RPOINT_CHUNK`).
#[derive(Debug, Clone, PartialEq)]
pub struct RPoint {
    pub pos: [f32; 3],
    pub rot: [f32; 3],
    pub team: u8,
    pub point_type: RPointType,
    pub game_type: u16,
    pub rp_profile: Option<String>,
}

/// Single waypoint along a `PatrolPath`.
#[derive(Debug, Clone, PartialEq)]
pub struct PatrolPoint {
    pub vertex_id: u32,
    pub name: String,
    pub pos: [f32; 3],
    pub flags: u32,
    pub level_vertex_id: u32,
    pub game_vertex_id: u16,
}

impl PatrolPoint {
    #[inline]
    pub fn has_level_vertex(&self) -> bool {
        self.level_vertex_id != INVALID_AI_VERTEX
    }

    #[inline]
    pub fn has_game_vertex(&self) -> bool {
        self.game_vertex_id != INVALID_GAME_VERTEX
    }
}

/// Directed weighted transition between two patrol points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PatrolEdge {
    pub from: u32,
    pub to: u32,
    pub probability: f32,
}

/// Patrol waypoint graph (`CPatrolPath`), sourced from `level.game`, `patrol_paths.ltx`, or `all.spawn` Chunk 3.
#[derive(Debug, Clone, PartialEq)]
pub struct PatrolPath {
    pub name: String,
    pub level_override: Option<String>,
    pub way_type: u32,
    pub points: Vec<PatrolPoint>,
    pub edges: Vec<PatrolEdge>,
}

impl PatrolPath {
    /// Parse a single subchunk inside `level.game` `WAY_PATROLPATH_CHUNK`.
    pub fn from_raw_chunk(chunk_bytes: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(chunk_bytes);
        let ver_bytes = reader.require_chunk(wayobject_chunks::VERSION)?;
        let version = ByteCursor::new(&ver_bytes).read_u16()?;
        if version != WAYOBJECT_VERSION {
            return Err(FormatError::InvalidValue("unsupported WAYOBJECT version"));
        }

        let name_bytes = reader.require_chunk(wayobject_chunks::NAME)?;
        let name = ByteCursor::new(&name_bytes).read_zstring()?;

        let way_type = if let Some(t_bytes) = reader.open_chunk(wayobject_chunks::TYPE)? {
            ByteCursor::new(&t_bytes).read_u32()?
        } else {
            0
        };

        let pts_bytes = reader.require_chunk(wayobject_chunks::POINTS)?;
        let mut pcur = ByteCursor::new(&pts_bytes);
        let pcount = usize::from(pcur.read_u16()?);
        let mut points = Vec::with_capacity(pcount);
        for idx in 0..pcount {
            let pos = pcur.read_f32_3()?;
            let flags = pcur.read_u32()?;
            let pt_name = pcur.read_zstring()?;
            points.push(PatrolPoint {
                vertex_id: idx as u32,
                name: pt_name,
                pos,
                flags,
                level_vertex_id: INVALID_AI_VERTEX,
                game_vertex_id: INVALID_GAME_VERTEX,
            });
        }

        let links_bytes = reader.require_chunk(wayobject_chunks::LINKS)?;
        let mut lcur = ByteCursor::new(&links_bytes);
        let lcount = usize::from(lcur.read_u16()?);
        let mut edges = Vec::with_capacity(lcount);
        for _ in 0..lcount {
            let from = u32::from(lcur.read_u16()?);
            let to = u32::from(lcur.read_u16()?);
            let probability = lcur.read_f32()?;
            edges.push(PatrolEdge {
                from,
                to,
                probability,
            });
        }

        Ok(Self {
            name,
            level_override: None,
            way_type,
            points,
            edges,
        })
    }

    /// Encode this path in `level.game` `WAY_PATROLPATH_CHUNK` subchunk format.
    pub fn to_raw_chunk(&self) -> Vec<u8> {
        let mut w = ChunkWriter::new();
        w.write_chunk(wayobject_chunks::VERSION, &WAYOBJECT_VERSION.to_le_bytes());

        let mut name_buf = self.name.as_bytes().to_vec();
        name_buf.push(0);
        w.write_chunk(wayobject_chunks::NAME, &name_buf);
        w.write_chunk(wayobject_chunks::TYPE, &self.way_type.to_le_bytes());

        let mut index_by_vid = HashMap::with_capacity(self.points.len());
        let mut pts = Vec::new();
        pts.extend_from_slice(&(self.points.len() as u16).to_le_bytes());
        for (idx, pt) in self.points.iter().enumerate() {
            index_by_vid.insert(pt.vertex_id, idx as u16);
            for f in pt.pos {
                pts.extend_from_slice(&f.to_le_bytes());
            }
            pts.extend_from_slice(&pt.flags.to_le_bytes());
            pts.extend_from_slice(pt.name.as_bytes());
            pts.push(0);
        }
        w.write_chunk(wayobject_chunks::POINTS, &pts);

        let mut links = Vec::new();
        links.extend_from_slice(&(self.edges.len() as u16).to_le_bytes());
        for edge in &self.edges {
            let f_idx = index_by_vid.get(&edge.from).copied().unwrap_or(edge.from as u16);
            let t_idx = index_by_vid.get(&edge.to).copied().unwrap_or(edge.to as u16);
            links.extend_from_slice(&f_idx.to_le_bytes());
            links.extend_from_slice(&t_idx.to_le_bytes());
            links.extend_from_slice(&edge.probability.to_le_bytes());
        }
        w.write_chunk(wayobject_chunks::LINKS, &links);

        w.into_bytes()
    }

    /// Parse `all.spawn` Chunk 3 `CGraphAbstractSerialize<CPatrolPoint, float, u32>` payload.
    pub fn from_serialized_chunks(name: String, graph_bytes: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(graph_bytes);
        let cnt_bytes = reader.require_chunk(0)?;
        let expected_count = ByteCursor::new(&cnt_bytes).read_u32()? as usize;

        let verts_bytes = reader.require_chunk(1)?;
        let mut points = Vec::with_capacity(expected_count.min(4096));
        for v_entry in ChunkReader::new(&verts_bytes).iter() {
            let v_entry = v_entry?;
            let v_data = v_entry.decode()?;
            let v_reader = ChunkReader::new(&v_data);

            let vid_bytes = v_reader.require_chunk(0)?;
            let vertex_id = ByteCursor::new(&vid_bytes).read_u32()?;

            let pt_bytes = v_reader.require_chunk(1)?;
            let mut pcur = ByteCursor::new(&pt_bytes);
            let pt_name = pcur.read_zstring()?;
            let pos = pcur.read_f32_3()?;
            let flags = pcur.read_u32()?;
            let level_vertex_id = pcur.read_u32()?;
            let game_vertex_id = pcur.read_u16()?;

            points.push(PatrolPoint {
                vertex_id,
                name: pt_name,
                pos,
                flags,
                level_vertex_id,
                game_vertex_id,
            });
        }

        let mut edges = Vec::new();
        if let Some(edge_bytes) = reader.open_chunk(2)? {
            let mut ecur = ByteCursor::new(&edge_bytes);
            while !ecur.is_eof() {
                let from = ecur.read_u32()?;
                let n = ecur.read_u32()? as usize;
                for _ in 0..n {
                    let to = ecur.read_u32()?;
                    let probability = ecur.read_f32()?;
                    edges.push(PatrolEdge {
                        from,
                        to,
                        probability,
                    });
                }
            }
        }

        Ok(Self {
            name,
            level_override: None,
            way_type: 0,
            points,
            edges,
        })
    }

    /// Encode this path in `all.spawn` Chunk 3 `CGraphAbstractSerialize<CPatrolPoint, float, u32>` format.
    pub fn to_serialized_chunks(&self) -> Vec<u8> {
        let mut gw = ChunkWriter::new();
        gw.write_chunk(0, &(self.points.len() as u32).to_le_bytes());

        let mut verts_w = ChunkWriter::new();
        for (i, pt) in self.points.iter().enumerate() {
            let mut vw = ChunkWriter::new();
            vw.write_chunk(0, &pt.vertex_id.to_le_bytes());

            let mut pt_buf = Vec::new();
            pt_buf.extend_from_slice(pt.name.as_bytes());
            pt_buf.push(0);
            for f in pt.pos {
                pt_buf.extend_from_slice(&f.to_le_bytes());
            }
            pt_buf.extend_from_slice(&pt.flags.to_le_bytes());
            pt_buf.extend_from_slice(&pt.level_vertex_id.to_le_bytes());
            pt_buf.extend_from_slice(&pt.game_vertex_id.to_le_bytes());
            vw.write_chunk(1, &pt_buf);

            verts_w.write_chunk(i as u32, &vw.into_bytes());
        }
        gw.write_chunk(1, &verts_w.into_bytes());

        let mut edges_by_from: HashMap<u32, Vec<&PatrolEdge>> = HashMap::new();
        for edge in &self.edges {
            edges_by_from.entry(edge.from).or_default().push(edge);
        }
        let mut edge_buf = Vec::new();
        for pt in &self.points {
            if let Some(list) = edges_by_from.get(&pt.vertex_id)
                && !list.is_empty()
            {
                edge_buf.extend_from_slice(&pt.vertex_id.to_le_bytes());
                edge_buf.extend_from_slice(&(list.len() as u32).to_le_bytes());
                for e in list {
                    edge_buf.extend_from_slice(&e.to.to_le_bytes());
                    edge_buf.extend_from_slice(&e.probability.to_le_bytes());
                }
            }
        }
        gw.write_chunk(2, &edge_buf);

        gw.into_bytes()
    }

    /// Parse a single section of `patrol_paths.ltx` (`CPatrolPath::load_from_config`).
    pub fn from_ltx_section(patrol_name: &str, ltx_text: &str) -> Result<Self, FormatError> {
        let doc = LtxDocument::parse(ltx_text);
        let view = LtxView::Borrowed(&doc);
        Self::from_ltx_view(patrol_name, &view)
    }

    /// Parse all `[<path_name>]` sections in a `patrol_paths.ltx` tree (sorted by path name).
    pub fn parse_all_ltx_doc(doc: &OwnedLtxDocument) -> Result<Vec<Self>, FormatError> {
        let view = LtxView::Owned(doc);
        let mut names: Vec<&str> = doc.sections.keys().map(String::as_str).collect();
        names.sort_unstable();
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            out.push(Self::from_ltx_view(name, &view)?);
        }
        Ok(out)
    }

    /// Parse all sections in a raw `patrol_paths.ltx` string (sorted by path name).
    pub fn parse_all_ltx_str(ltx_text: &str) -> Result<Vec<Self>, FormatError> {
        let doc = LtxDocument::parse(ltx_text);
        let view = LtxView::Borrowed(&doc);
        let mut names: Vec<&str> = doc.sections.keys().copied().collect();
        names.sort_unstable();
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            out.push(Self::from_ltx_view(name, &view)?);
        }
        Ok(out)
    }

    fn from_ltx_view(patrol_name: &str, doc: &LtxView<'_>) -> Result<Self, FormatError> {
        let level_override = doc
            .read_str(patrol_name, "level")
            .map(|s| s.trim_matches('"').trim().to_string())
            .filter(|s| !s.is_empty());

        let points_csv = doc
            .read_str(patrol_name, "points")
            .ok_or(FormatError::InvalidValue("missing 'points' in patrol path section"))?;

        let point_keys: Vec<&str> = points_csv
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();

        let mut vertex_ids_by_key = HashMap::with_capacity(point_keys.len());
        let mut points = Vec::with_capacity(point_keys.len());

        for (idx, &pt_key) in point_keys.iter().enumerate() {
            let name_key = format!("{pt_key}:name");
            let pt_name = doc
                .read_str(patrol_name, &name_key)
                .ok_or(FormatError::InvalidValue("missing <pt>:name in patrol_paths.ltx"))?
                .trim_matches('"')
                .to_string();

            let pos_key = format!("{pt_key}:position");
            let pos = doc
                .read_vec3(patrol_name, &pos_key)
                .ok_or(FormatError::InvalidValue("missing <pt>:position in patrol_paths.ltx"))?;

            let lvid_key = format!("{pt_key}:level_vertex_id");
            let gvid_key = format!("{pt_key}:game_vertex_id");
            let (level_vertex_id, game_vertex_id) =
                match (doc.read_u32(patrol_name, &lvid_key), doc.read_u16(patrol_name, &gvid_key)) {
                    (Some(lv), Some(gv)) => (lv, gv),
                    _ if level_override.is_some() => (INVALID_AI_VERTEX, INVALID_GAME_VERTEX),
                    _ => {
                        return Err(FormatError::InvalidValue(
                            "patrol point missing level_vertex_id/game_vertex_id without 'level'",
                        ));
                    }
                };

            let flags_key = format!("{pt_key}:flags");
            let flags = doc.read_u32(patrol_name, &flags_key).unwrap_or(0);

            vertex_ids_by_key.insert(pt_key, idx as u32);
            points.push(PatrolPoint {
                vertex_id: idx as u32,
                name: pt_name,
                pos,
                flags,
                level_vertex_id,
                game_vertex_id,
            });
        }

        let mut edges = Vec::new();
        for (idx, &pt_key) in point_keys.iter().enumerate() {
            let links_key = format!("{pt_key}:links");
            let Some(links_csv) = doc.read_str(patrol_name, &links_key) else {
                continue;
            };
            for link in links_csv.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let open = link
                    .find('(')
                    .ok_or(FormatError::InvalidValue("invalid patrol point link syntax"))?;
                let close = link
                    .rfind(')')
                    .filter(|&c| c > open)
                    .ok_or(FormatError::InvalidValue("invalid patrol point link syntax"))?;
                let target_key = link[..open].trim();
                let weight_str = link[open + 1..close].trim();
                let probability: f32 = weight_str
                    .parse()
                    .map_err(|_| FormatError::InvalidValue("invalid patrol point link weight"))?;
                let &target_vid = vertex_ids_by_key
                    .get(target_key)
                    .ok_or(FormatError::InvalidValue("patrol link target not in 'points'"))?;
                edges.push(PatrolEdge {
                    from: idx as u32,
                    to: target_vid,
                    probability,
                });
            }
        }

        Ok(Self {
            name: patrol_name.to_string(),
            level_override,
            way_type: 0,
            points,
            edges,
        })
    }

    pub fn point_by_vertex_id(&self, vertex_id: u32) -> Option<&PatrolPoint> {
        self.points.iter().find(|p| p.vertex_id == vertex_id)
    }

    pub fn point_by_name(&self, name: &str) -> Option<&PatrolPoint> {
        self.points.iter().find(|p| p.name == name)
    }

    pub fn outgoing_edges(&self, from_vertex_id: u32) -> impl Iterator<Item = &PatrolEdge> {
        self.edges.iter().filter(move |e| e.from == from_vertex_id)
    }
}

/// Parsed `level.game` container holding `RPOINT_CHUNK` and `WAY_PATROLPATH_CHUNK`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LevelGameData {
    pub rpoints: Vec<RPoint>,
    pub patrol_paths: Vec<PatrolPath>,
}

impl LevelGameData {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let root = ChunkReader::new(data);

        let mut rpoints = Vec::new();
        if let Some(rp_bytes) = root.open_chunk(RPOINT_CHUNK)? {
            for entry in ChunkReader::new(&rp_bytes).iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                let mut cur = ByteCursor::new(&payload);
                let pos = cur.read_f32_3()?;
                let rot = cur.read_f32_3()?;
                let team = cur.read_u8()?;
                let point_type = RPointType::from_u8(cur.read_u8()?)?;
                let game_type = cur.read_u16()?;
                let rp_profile = if point_type == RPointType::ItemSpawn && !cur.is_eof() {
                    Some(cur.read_zstring()?)
                } else {
                    None
                };
                rpoints.push(RPoint {
                    pos,
                    rot,
                    team,
                    point_type,
                    game_type,
                    rp_profile,
                });
            }
        }

        let mut patrol_paths = Vec::new();
        if let Some(way_bytes) = root.open_chunk(WAY_PATROLPATH_CHUNK)? {
            for entry in ChunkReader::new(&way_bytes).iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                patrol_paths.push(PatrolPath::from_raw_chunk(&payload)?);
            }
        }

        Ok(Self {
            rpoints,
            patrol_paths,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut root = ChunkWriter::new();

        if !self.rpoints.is_empty() {
            let mut rp_w = ChunkWriter::new();
            for (i, rp) in self.rpoints.iter().enumerate() {
                let mut buf = Vec::with_capacity(32);
                for f in rp.pos {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                for f in rp.rot {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                buf.push(rp.team);
                buf.push(rp.point_type as u8);
                buf.extend_from_slice(&rp.game_type.to_le_bytes());
                if rp.point_type == RPointType::ItemSpawn {
                    if let Some(prof) = &rp.rp_profile {
                        buf.extend_from_slice(prof.as_bytes());
                    }
                    buf.push(0);
                }
                rp_w.write_chunk(i as u32, &buf);
            }
            root.write_chunk(RPOINT_CHUNK, &rp_w.into_bytes());
        }

        if !self.patrol_paths.is_empty() {
            let mut way_w = ChunkWriter::new();
            for (i, path) in self.patrol_paths.iter().enumerate() {
                way_w.write_chunk(i as u32, &path.to_raw_chunk());
            }
            root.write_chunk(WAY_PATROLPATH_CHUNK, &way_w.into_bytes());
        }

        root.into_bytes()
    }

    pub fn find_patrol_path(&self, name: &str) -> Option<&PatrolPath> {
        self.patrol_paths.iter().find(|p| p.name == name)
    }

    pub fn rpoints_by_type(&self, point_type: RPointType) -> impl Iterator<Item = &RPoint> {
        self.rpoints
            .iter()
            .filter(move |r| r.point_type == point_type)
    }
}

/// Decoded `CSE_ALifeObject` header prefix of an `M_SPAWN` state payload (`xrServer_Objects_ALife.cpp:439`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeObjectState {
    pub game_vertex_id: u16,
    pub distance: f32,
    pub direct_control: bool,
    pub level_vertex_id: u32,
    pub flags: u32,
    pub custom_data: String,
    pub story_id: u32,
    pub spawn_story_id: u32,
    pub subclass_payload: Vec<u8>,
}

impl AlifeObjectState {
    /// Parse `CSE_ALifeObject::STATE_Read` prefix and preserve remaining subclass state bytes.
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut game_vertex_id = INVALID_GAME_VERTEX;
        let mut distance = 0.0;
        let mut direct_control = true;
        let mut level_vertex_id = INVALID_AI_VERTEX;
        let mut flags = !alife_object_flags::OFFLINE_NO_MOVE;
        let mut custom_data = String::new();
        let mut story_id = INVALID_STORY_ID;
        let mut spawn_story_id = INVALID_SPAWN_STORY_ID;

        if version >= 1 {
            if version > 24 {
                if version < 83 {
                    let _ = cur.read_f32()?;
                }
            } else {
                let _ = cur.read_u8()?;
            }
            if version < 83 {
                let _ = cur.read_u32()?;
            }
            if version < 4 {
                let _ = cur.read_u16()?;
            }
            game_vertex_id = cur.read_u16()?;
            distance = cur.read_f32()?;
        }

        if version >= 4 {
            direct_control = cur.read_u32()? != 0;
        }
        if version >= 8 {
            level_vertex_id = cur.read_u32()?;
        }
        if version > 22 && version <= 79 {
            let _ = cur.read_u16()?;
        }
        if version > 23 && version < 84 {
            let _ = cur.read_zstring()?;
        }
        if version > 49 {
            flags = cur.read_u32()?;
        }
        if version > 57 {
            custom_data = cur.read_zstring()?;
        }
        if version > 61 {
            story_id = cur.read_u32()?;
        }
        if version > 111 {
            spawn_story_id = cur.read_u32()?;
        }

        let subclass_payload = state_bytes[cur.position()..].to_vec();

        Ok(Self {
            game_vertex_id,
            distance,
            direct_control,
            level_vertex_id,
            flags,
            custom_data,
            story_id,
            spawn_story_id,
            subclass_payload,
        })
    }

    /// Encode in `SPAWN_VERSION` (128) wire layout (`CSE_ALifeObject::STATE_Write`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.custom_data.len() + self.subclass_payload.len());
        out.extend_from_slice(&self.game_vertex_id.to_le_bytes());
        out.extend_from_slice(&self.distance.to_le_bytes());
        out.extend_from_slice(&u32::from(self.direct_control).to_le_bytes());
        out.extend_from_slice(&self.level_vertex_id.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(self.custom_data.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.story_id.to_le_bytes());
        out.extend_from_slice(&self.spawn_story_id.to_le_bytes());
        out.extend_from_slice(&self.subclass_payload);
        out
    }
}

/// Decoded `CSE_ALifeGraphPoint::STATE_Read` payload (`section == "graph_point"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPointState {
    pub connection_point_name: String,
    pub connection_level_name: String,
    pub legacy_level_id: Option<u32>,
    pub locations: [u8; 4],
}

impl GraphPointState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let connection_point_name = cur.read_zstring()?;
        let (connection_level_name, legacy_level_id) = if version < 33 {
            (String::new(), Some(cur.read_u32()?))
        } else {
            (cur.read_zstring()?, None)
        };
        let locs = cur.read_bytes(4)?;
        Ok(Self {
            connection_point_name,
            connection_level_name,
            legacy_level_id,
            locations: [locs[0], locs[1], locs[2], locs[3]],
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(self.connection_point_name.as_bytes());
        out.push(0);
        out.extend_from_slice(self.connection_level_name.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.locations);
        out
    }
}

/// Parsed `M_SPAWN` entity creation packet (`CSE_Abstract::Spawn_Read`).
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnPacket {
    pub section: String,
    pub name_replace: String,
    pub game_id_legacy: u8,
    pub rp: u8,
    pub position: [f32; 3],
    pub angle: [f32; 3],
    pub respawn_time: u16,
    pub id: u16,
    pub parent_id: u16,
    pub phantom_id: u16,
    pub flags: u16,
    pub version: u16,
    pub game_type: u16,
    pub script_version: u16,
    pub client_data: Vec<u8>,
    pub spawn_id: u16,
    pub state_payload: Vec<u8>,
    pub trailing_update_payload: Vec<u8>,
}

impl SpawnPacket {
    /// Parse a raw `M_SPAWN` `NET_Packet` byte buffer.
    pub fn parse(packet_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(packet_bytes);
        let msg_id = cur.read_u16()?;
        if msg_id != M_SPAWN {
            return Err(FormatError::InvalidValue("expected M_SPAWN packet ID (1)"));
        }

        let section = cur.read_zstring()?;
        let name_replace = cur.read_zstring()?;
        let game_id_legacy = cur.read_u8()?;
        let rp = cur.read_u8()?;
        let position = cur.read_f32_3()?;
        let angle = cur.read_f32_3()?;
        let respawn_time = cur.read_u16()?;
        let id = cur.read_u16()?;
        let parent_id = cur.read_u16()?;
        let phantom_id = cur.read_u16()?;
        let flags = cur.read_u16()?;

        let version = if (flags & m_spawn_flags::VERSION) != 0 {
            cur.read_u16()?
        } else {
            return Err(FormatError::InvalidValue(
                "M_SPAWN packet missing M_SPAWN_VERSION flag",
            ));
        };
        if version == 0 {
            return Err(FormatError::InvalidValue("M_SPAWN packet has version 0"));
        }

        let game_type = if version > 120 {
            cur.read_u16()?
        } else {
            u16::MAX
        };
        let script_version = if version > 69 { cur.read_u16()? } else { 0 };

        let client_data = if version > 70 {
            let cd_size = if version > 93 {
                usize::from(cur.read_u16()?)
            } else {
                usize::from(cur.read_u8()?)
            };
            if cd_size > 0 {
                cur.read_bytes(cd_size)?.to_vec()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let spawn_id = if version > 79 {
            cur.read_u16()?
        } else {
            INVALID_OBJECT_ID
        };

        if version < 112 {
            if version > 82 {
                let _ = cur.read_f32()?;
            }
            if version > 83 {
                let _ = cur.read_u32()?;
                let _ = cur.read_zstring()?;
                let _ = cur.read_u32()?;
                let _ = cur.read_u32()?;
                let _ = cur.read_u64()?;
            }
            if version > 84 {
                let _ = cur.read_u64()?;
                let _ = cur.read_u64()?;
            }
        }

        let state_size = cur.read_u16()?;
        if state_size < 2 {
            return Err(FormatError::InvalidValue(
                "M_SPAWN state_size must include its 2-byte length prefix (>= 2)",
            ));
        }
        let state_payload = cur.read_bytes(usize::from(state_size - 2))?.to_vec();
        let trailing_update_payload = packet_bytes[cur.position()..].to_vec();

        Ok(Self {
            section,
            name_replace,
            game_id_legacy,
            rp,
            position,
            angle,
            respawn_time,
            id,
            parent_id,
            phantom_id,
            flags,
            version,
            game_type,
            script_version,
            client_data,
            spawn_id,
            state_payload,
            trailing_update_payload,
        })
    }

    /// Encode this `SpawnPacket` in `SPAWN_VERSION` (128) wire format (`CSE_Abstract::Spawn_Write`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&M_SPAWN.to_le_bytes());
        out.extend_from_slice(self.section.as_bytes());
        out.push(0);
        out.extend_from_slice(self.name_replace.as_bytes());
        out.push(0);
        out.push(self.game_id_legacy);
        out.push(self.rp);
        for f in self.position {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.angle {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&self.respawn_time.to_le_bytes());
        out.extend_from_slice(&self.id.to_le_bytes());
        out.extend_from_slice(&self.parent_id.to_le_bytes());
        out.extend_from_slice(&self.phantom_id.to_le_bytes());
        out.extend_from_slice(&(self.flags | m_spawn_flags::VERSION).to_le_bytes());

        let ver = if self.version == 0 {
            SPAWN_VERSION
        } else {
            self.version
        };
        out.extend_from_slice(&ver.to_le_bytes());
        if ver > 120 {
            out.extend_from_slice(&self.game_type.to_le_bytes());
        }
        if ver > 69 {
            out.extend_from_slice(&self.script_version.to_le_bytes());
        }
        if ver > 70 {
            if ver > 93 {
                out.extend_from_slice(&(self.client_data.len() as u16).to_le_bytes());
            } else {
                out.push(self.client_data.len().min(u8::MAX as usize) as u8);
            }
            out.extend_from_slice(&self.client_data);
        }
        if ver > 79 {
            out.extend_from_slice(&self.spawn_id.to_le_bytes());
        }

        let state_total = (2 + self.state_payload.len()) as u16;
        out.extend_from_slice(&state_total.to_le_bytes());
        out.extend_from_slice(&self.state_payload);
        out.extend_from_slice(&self.trailing_update_payload);
        out
    }

    #[inline]
    pub fn is_graph_point(&self) -> bool {
        self.section.eq_ignore_ascii_case("graph_point")
            || self.section.eq_ignore_ascii_case("ai_graph_point")
    }

    pub fn decode_alife_state(&self) -> Result<AlifeObjectState, FormatError> {
        AlifeObjectState::parse(self.version, &self.state_payload)
    }

    pub fn decode_graph_point_state(&self) -> Result<GraphPointState, FormatError> {
        GraphPointState::parse(self.version, &self.state_payload)
    }
}

/// SDK/level-editor `level.spawn` container (each chunk = one raw `M_SPAWN` packet).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LevelSpawnData {
    pub spawns: Vec<SpawnPacket>,
}

impl LevelSpawnData {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut spawns = Vec::new();
        for entry in ChunkReader::new(data).iter() {
            let entry = entry?;
            let payload = entry.decode()?;
            spawns.push(SpawnPacket::parse(&payload)?);
        }
        Ok(Self { spawns })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = ChunkWriter::new();
        for (i, sp) in self.spawns.iter().enumerate() {
            w.write_chunk(i as u32, &sp.to_bytes());
        }
        w.into_bytes()
    }
}

/// Header of compiled `all.spawn` (`CALifeSpawnHeader`, Chunk 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnHeader {
    pub version: u32,
    pub guid: [u8; 16],
    pub graph_guid: [u8; 16],
    pub spawn_count: u32,
    pub level_count: u32,
}

/// Single server entity entry in `all.spawn` Chunk 1 (`CServerEntityWrapper`).
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnVertex {
    pub spawn_id: u16,
    pub spawn: SpawnPacket,
    pub update_packet: Vec<u8>,
}

/// Probability edge between spawn vertices in `all.spawn` Chunk 1 (`SPAWN_GRAPH`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpawnEdge {
    pub from: u16,
    pub to: u16,
    pub weight: f32,
}

/// Complete compiled world spawn archive (`all.spawn` / `CALifeSpawnRegistry`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeSpawnArchive {
    pub header: SpawnHeader,
    pub spawns: Vec<SpawnVertex>,
    pub edges: Vec<SpawnEdge>,
    pub artefact_spawn_positions: Vec<GameDeathPoint>,
    pub patrol_paths: Vec<PatrolPath>,
    pub game_graph: GameGraph,
}

impl AlifeSpawnArchive {
    /// Parse a binary `all.spawn` archive (`CALifeSpawnRegistry::load`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let root = ChunkReader::new(data);

        // Chunk 0: CALifeSpawnHeader
        let hdr_bytes = root.require_chunk(0)?;
        let mut hcur = ByteCursor::new(&hdr_bytes);
        let version = hcur.read_u32()?;
        if version != XRAI_CURRENT_VERSION && version != XRAI_LARGE_VERSION {
            return Err(FormatError::InvalidValue(
                "unsupported all.spawn header version (expected 10 or 13)",
            ));
        }
        let mut guid = [0u8; 16];
        guid.copy_from_slice(hcur.read_bytes(16)?);
        let mut graph_guid = [0u8; 16];
        graph_guid.copy_from_slice(hcur.read_bytes(16)?);
        let spawn_count = hcur.read_u32()?;
        let level_count = hcur.read_u32()?;
        let header = SpawnHeader {
            version,
            guid,
            graph_guid,
            spawn_count,
            level_count,
        };

        // Chunk 1: CGraphAbstractSerialize<CServerEntityWrapper*, float, u16>
        let graph_chunk = root.require_chunk(1)?;
        let sg_reader = ChunkReader::new(&graph_chunk);
        let cnt_bytes = sg_reader.require_chunk(0)?;
        let sg_count = ByteCursor::new(&cnt_bytes).read_u32()? as usize;
        let sg_verts_bytes = sg_reader.require_chunk(1)?;
        let mut spawns = Vec::with_capacity(sg_count.min(65_536));

        for entry in ChunkReader::new(&sg_verts_bytes).iter() {
            let entry = entry?;
            let vdata = entry.decode()?;
            let vreader = ChunkReader::new(&vdata);

            let vid_bytes = vreader.require_chunk(0)?;
            let spawn_id = ByteCursor::new(&vid_bytes).read_u16()?;

            let wrap_bytes = vreader.require_chunk(1)?;
            let wrap_reader = ChunkReader::new(&wrap_bytes);

            let sp_chunk = wrap_reader.require_chunk(0)?;
            let mut sp_cur = ByteCursor::new(&sp_chunk);
            let sp_len = usize::from(sp_cur.read_u16()?);
            let sp_bytes = sp_cur.read_bytes(sp_len)?;
            let spawn = SpawnPacket::parse(sp_bytes)?;

            let update_packet = if let Some(upd_chunk) = wrap_reader.open_chunk(1)? {
                let mut ucur = ByteCursor::new(&upd_chunk);
                let u_len = usize::from(ucur.read_u16()?);
                let raw_upd = ucur.read_bytes(u_len)?;
                if raw_upd.len() >= 2 {
                    let msg = u16::from_le_bytes([raw_upd[0], raw_upd[1]]);
                    if msg != M_UPDATE {
                        return Err(FormatError::InvalidValue(
                            "expected M_UPDATE packet ID (0) in spawn update chunk",
                        ));
                    }
                }
                raw_upd.to_vec()
            } else {
                Vec::new()
            };

            spawns.push(SpawnVertex {
                spawn_id,
                spawn,
                update_packet,
            });
        }

        let mut edges = Vec::new();
        if let Some(edge_bytes) = sg_reader.open_chunk(2)? {
            let mut ecur = ByteCursor::new(&edge_bytes);
            while !ecur.is_eof() {
                let from = ecur.read_u16()?;
                let n = ecur.read_u32()? as usize;
                for _ in 0..n {
                    let to = ecur.read_u16()?;
                    let weight = ecur.read_f32()?;
                    edges.push(SpawnEdge { from, to, weight });
                }
            }
        }

        // Chunk 2: ARTEFACT_SPAWNS (Vec<GameDeathPoint>)
        let art_bytes = root.require_chunk(2)?;
        let mut acur = ByteCursor::new(&art_bytes);
        let art_count = acur.read_u32()? as usize;
        let need_art = art_count.checked_mul(20).ok_or(FormatError::UnexpectedEof)?;
        let raw_art = acur.read_bytes(need_art)?;
        let mut artefact_spawn_positions = Vec::with_capacity(art_count);
        for rec in raw_art.as_chunks::<20>().0 {
            let mut dc = ByteCursor::new(rec);
            artefact_spawn_positions.push(GameDeathPoint {
                point: dc.read_f32_3()?,
                level_vertex_id: dc.read_u32()?,
                distance: dc.read_f32()?,
            });
        }

        // Chunk 3: CPatrolPathStorage
        let pp_chunk = root.require_chunk(3)?;
        let pp_reader = ChunkReader::new(&pp_chunk);
        let pp_cnt_bytes = pp_reader.require_chunk(0)?;
        let pp_count = ByteCursor::new(&pp_cnt_bytes).read_u32()? as usize;
        let pp_list_bytes = pp_reader.require_chunk(1)?;
        let mut patrol_paths = Vec::with_capacity(pp_count.min(16_384));
        for entry in ChunkReader::new(&pp_list_bytes).iter() {
            let entry = entry?;
            let item_bytes = entry.decode()?;
            let item_reader = ChunkReader::new(&item_bytes);
            let name_bytes = item_reader.require_chunk(0)?;
            let name = ByteCursor::new(&name_bytes).read_zstring()?;
            let graph_bytes = item_reader.require_chunk(1)?;
            patrol_paths.push(PatrolPath::from_serialized_chunks(name, &graph_bytes)?);
        }

        // Chunk 4: embedded CGameGraph
        let gg_bytes = root.require_chunk(4)?;
        let game_graph = GameGraph::parse(&gg_bytes)?;

        Ok(Self {
            header,
            spawns,
            edges,
            artefact_spawn_positions,
            patrol_paths,
            game_graph,
        })
    }

    /// Encode this `AlifeSpawnArchive` into binary `all.spawn` wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut root = ChunkWriter::new();

        // Chunk 0
        let mut hdr = Vec::with_capacity(44);
        hdr.extend_from_slice(&self.header.version.to_le_bytes());
        hdr.extend_from_slice(&self.header.guid);
        hdr.extend_from_slice(&self.header.graph_guid);
        hdr.extend_from_slice(&(self.spawns.len() as u32).to_le_bytes());
        hdr.extend_from_slice(&(self.game_graph.header.levels.len() as u32).to_le_bytes());
        root.write_chunk(0, &hdr);

        // Chunk 1
        let mut sg_w = ChunkWriter::new();
        sg_w.write_chunk(0, &(self.spawns.len() as u32).to_le_bytes());

        let mut verts_w = ChunkWriter::new();
        for (i, sv) in self.spawns.iter().enumerate() {
            let mut vw = ChunkWriter::new();
            vw.write_chunk(0, &sv.spawn_id.to_le_bytes());

            let mut wrap_w = ChunkWriter::new();
            let sp_raw = sv.spawn.to_bytes();
            let mut sp_pkt = Vec::with_capacity(2 + sp_raw.len());
            sp_pkt.extend_from_slice(&(sp_raw.len() as u16).to_le_bytes());
            sp_pkt.extend_from_slice(&sp_raw);
            wrap_w.write_chunk(0, &sp_pkt);

            let upd_bytes = if sv.update_packet.is_empty() {
                M_UPDATE.to_le_bytes().to_vec()
            } else {
                sv.update_packet.clone()
            };
            let mut upd_pkt = Vec::with_capacity(2 + upd_bytes.len());
            upd_pkt.extend_from_slice(&(upd_bytes.len() as u16).to_le_bytes());
            upd_pkt.extend_from_slice(&upd_bytes);
            wrap_w.write_chunk(1, &upd_pkt);

            vw.write_chunk(1, &wrap_w.into_bytes());
            verts_w.write_chunk(i as u32, &vw.into_bytes());
        }
        sg_w.write_chunk(1, &verts_w.into_bytes());

        let mut edges_by_from: HashMap<u16, Vec<&SpawnEdge>> = HashMap::new();
        for edge in &self.edges {
            edges_by_from.entry(edge.from).or_default().push(edge);
        }
        let mut edge_buf = Vec::new();
        for sv in &self.spawns {
            if let Some(list) = edges_by_from.get(&sv.spawn_id)
                && !list.is_empty()
            {
                edge_buf.extend_from_slice(&sv.spawn_id.to_le_bytes());
                edge_buf.extend_from_slice(&(list.len() as u32).to_le_bytes());
                for e in list {
                    edge_buf.extend_from_slice(&e.to.to_le_bytes());
                    edge_buf.extend_from_slice(&e.weight.to_le_bytes());
                }
            }
        }
        sg_w.write_chunk(2, &edge_buf);
        root.write_chunk(1, &sg_w.into_bytes());

        // Chunk 2
        let mut art_buf = Vec::with_capacity(4 + self.artefact_spawn_positions.len() * 20);
        art_buf.extend_from_slice(&(self.artefact_spawn_positions.len() as u32).to_le_bytes());
        for dp in &self.artefact_spawn_positions {
            for f in dp.point {
                art_buf.extend_from_slice(&f.to_le_bytes());
            }
            art_buf.extend_from_slice(&dp.level_vertex_id.to_le_bytes());
            art_buf.extend_from_slice(&dp.distance.to_le_bytes());
        }
        root.write_chunk(2, &art_buf);

        // Chunk 3
        let mut pp_w = ChunkWriter::new();
        pp_w.write_chunk(0, &(self.patrol_paths.len() as u32).to_le_bytes());
        let mut pp_items = ChunkWriter::new();
        for (i, path) in self.patrol_paths.iter().enumerate() {
            let mut item_w = ChunkWriter::new();
            let mut nbuf = path.name.as_bytes().to_vec();
            nbuf.push(0);
            item_w.write_chunk(0, &nbuf);
            item_w.write_chunk(1, &path.to_serialized_chunks());
            pp_items.write_chunk(i as u32, &item_w.into_bytes());
        }
        pp_w.write_chunk(1, &pp_items.into_bytes());
        root.write_chunk(3, &pp_w.into_bytes());

        // Chunk 4
        root.write_chunk(4, &self.game_graph.to_bytes());

        root.into_bytes()
    }

    pub fn find_spawn(&self, spawn_id: u16) -> Option<&SpawnVertex> {
        self.spawns.iter().find(|s| s.spawn_id == spawn_id)
    }

    pub fn find_spawn_by_name(&self, name: &str) -> Option<&SpawnVertex> {
        self.spawns.iter().find(|s| s.spawn.name_replace == name)
    }

    pub fn find_patrol_path(&self, name: &str) -> Option<&PatrolPath> {
        self.patrol_paths.iter().find(|p| p.name == name)
    }
}

enum LtxView<'a> {
    Borrowed(&'a LtxDocument<'a>),
    Owned(&'a OwnedLtxDocument),
}

impl<'a> LtxView<'a> {
    fn read_str(&self, section: &str, key: &str) -> Option<&str> {
        match self {
            Self::Borrowed(d) => d.read_str(section, key),
            Self::Owned(d) => d.read_str(section, key),
        }
    }

    fn read_u32(&self, section: &str, key: &str) -> Option<u32> {
        let s = self.read_str(section, key)?.trim().trim_matches('"');
        if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            u32::from_str_radix(hex, 16).ok()
        } else {
            s.parse::<u32>().ok()
        }
    }

    fn read_u16(&self, section: &str, key: &str) -> Option<u16> {
        let s = self.read_str(section, key)?.trim().trim_matches('"');
        if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
            u16::from_str_radix(hex, 16).ok()
        } else {
            s.parse::<u16>().ok()
        }
    }

    fn read_vec3(&self, section: &str, key: &str) -> Option<[f32; 3]> {
        let s = self.read_str(section, key)?;
        let mut it = s.split(',').filter_map(|p| p.trim().parse::<f32>().ok());
        Some([it.next()?, it.next()?, it.next()?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_graph::{GameEdge, GameGraphHeader, GameLevelInfo, GameVertex};

    #[test]
    fn test_level_game_rpoints_and_patrol_paths_roundtrip() {
        let game = LevelGameData {
            rpoints: vec![
                RPoint {
                    pos: [1.0, 2.0, 3.0],
                    rot: [0.0, 1.57, 0.0],
                    team: 1,
                    point_type: RPointType::ActorSpawn,
                    game_type: 0xFFFF,
                    rp_profile: None,
                },
                RPoint {
                    pos: [10.0, 0.0, -4.0],
                    rot: [0.0, 0.0, 0.0],
                    team: 0,
                    point_type: RPointType::ItemSpawn,
                    game_type: 0x0002,
                    rp_profile: Some("medkit_mp".to_string()),
                },
            ],
            patrol_paths: vec![PatrolPath {
                name: "esc_bridge_walk".to_string(),
                level_override: None,
                way_type: 0,
                points: vec![
                    PatrolPoint {
                        vertex_id: 0,
                        name: "wp00".to_string(),
                        pos: [0.0, 0.0, 0.0],
                        flags: 0x1,
                        level_vertex_id: INVALID_AI_VERTEX,
                        game_vertex_id: INVALID_GAME_VERTEX,
                    },
                    PatrolPoint {
                        vertex_id: 1,
                        name: "wp01".to_string(),
                        pos: [5.0, 0.0, 0.0],
                        flags: 0x2,
                        level_vertex_id: INVALID_AI_VERTEX,
                        game_vertex_id: INVALID_GAME_VERTEX,
                    },
                ],
                edges: vec![PatrolEdge {
                    from: 0,
                    to: 1,
                    probability: 0.75,
                }],
            }],
        };

        let raw = game.to_bytes();
        let parsed = LevelGameData::parse(&raw).unwrap();
        assert_eq!(parsed.rpoints.len(), 2);
        assert_eq!(parsed.rpoints[1].rp_profile.as_deref(), Some("medkit_mp"));
        let path = parsed.find_patrol_path("esc_bridge_walk").unwrap();
        assert_eq!(path.points.len(), 2);
        assert_eq!(path.edges[0], PatrolEdge { from: 0, to: 1, probability: 0.75 });
    }

    #[test]
    fn test_patrol_paths_ltx_parsing() {
        let ltx = r#"
[gar_custom_guard]
points = p0, p1
p0:name = wp00
p0:flags = 0x10
p0:position = 12.5, 1.0, -3.0
p0:level_vertex_id = 4200
p0:game_vertex_id = 150
p0:links = p1(1)
p1:name = wp01
p1:position = 20.0, 1.0, -3.0
p1:level_vertex_id = 4250
p1:game_vertex_id = 151

[zaton_overlay_path]
level = zaton
points = p0, p1
p0:name = wp00
p0:position = 1, 2, 3
p0:links = p1(0.5)
p1:name = wp01
p1:position = 4, 2, 6
"#;
        let paths = PatrolPath::parse_all_ltx_str(ltx).unwrap();
        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].name, "gar_custom_guard");
        assert_eq!(paths[0].points[0].flags, 0x10);
        assert_eq!(paths[0].points[0].level_vertex_id, 4200);
        assert_eq!(paths[0].edges[0].probability, 1.0);

        assert_eq!(paths[1].name, "zaton_overlay_path");
        assert_eq!(paths[1].level_override.as_deref(), Some("zaton"));
        assert!(!paths[1].points[0].has_level_vertex());
        assert_eq!(paths[1].edges[0].probability, 0.5);
    }

    #[test]
    fn test_spawn_packet_alife_and_graph_point_decode() {
        let alife_state = AlifeObjectState {
            game_vertex_id: 42,
            distance: 7.5,
            direct_control: true,
            level_vertex_id: 12345,
            flags: alife_object_flags::SWITCH_ONLINE | alife_object_flags::USED_AI_LOCATIONS,
            custom_data: "[logic]\nactive = walker\n".to_string(),
            story_id: 777,
            spawn_story_id: 888,
            subclass_payload: vec![0xDE, 0xAD],
        };
        let stalker_pkt = SpawnPacket {
            section: "sim_default_stalker_0".to_string(),
            name_replace: "esc_wolf".to_string(),
            game_id_legacy: 0,
            rp: 0xFE,
            position: [1.0, 2.0, 3.0],
            angle: [0.0, 0.5, 0.0],
            respawn_time: 0,
            id: INVALID_OBJECT_ID,
            parent_id: INVALID_OBJECT_ID,
            phantom_id: INVALID_OBJECT_ID,
            flags: m_spawn_flags::OBJECT_LOCAL | m_spawn_flags::VERSION,
            version: SPAWN_VERSION,
            game_type: u16::MAX,
            script_version: 10,
            client_data: vec![1, 2, 3],
            spawn_id: 5,
            state_payload: alife_state.to_bytes(),
            trailing_update_payload: Vec::new(),
        };

        let gp_state = GraphPointState {
            connection_point_name: "gar_start".to_string(),
            connection_level_name: "l02_garbage".to_string(),
            legacy_level_id: None,
            locations: [1, 2, 0, 0],
        };
        let gp_pkt = SpawnPacket {
            section: "graph_point".to_string(),
            name_replace: "esc_to_gar".to_string(),
            game_id_legacy: 0,
            rp: 0xFE,
            position: [0.0, 0.0, 100.0],
            angle: [0.0; 3],
            respawn_time: 0,
            id: INVALID_OBJECT_ID,
            parent_id: INVALID_OBJECT_ID,
            phantom_id: INVALID_OBJECT_ID,
            flags: m_spawn_flags::VERSION,
            version: SPAWN_VERSION,
            game_type: u16::MAX,
            script_version: 10,
            client_data: Vec::new(),
            spawn_id: 6,
            state_payload: gp_state.to_bytes(),
            trailing_update_payload: Vec::new(),
        };

        let level_spawn = LevelSpawnData {
            spawns: vec![stalker_pkt, gp_pkt],
        };
        let parsed_ls = LevelSpawnData::parse(&level_spawn.to_bytes()).unwrap();
        assert_eq!(parsed_ls.spawns.len(), 2);

        let decoded_alife = parsed_ls.spawns[0].decode_alife_state().unwrap();
        assert_eq!(decoded_alife.game_vertex_id, 42);
        assert_eq!(decoded_alife.level_vertex_id, 12345);
        assert_eq!(decoded_alife.story_id, 777);
        assert_eq!(decoded_alife.subclass_payload, vec![0xDE, 0xAD]);

        assert!(parsed_ls.spawns[1].is_graph_point());
        let decoded_gp = parsed_ls.spawns[1].decode_graph_point_state().unwrap();
        assert_eq!(decoded_gp.connection_level_name, "l02_garbage");
        assert_eq!(decoded_gp.locations, [1, 2, 0, 0]);
    }

    #[test]
    fn test_all_spawn_archive_full_roundtrip() {
        let gg = GameGraph {
            header: GameGraphHeader {
                version: 10,
                vertex_count: 1,
                edge_count: 1,
                death_point_count: 1,
                guid: [0x11, 0x22],
                levels: vec![GameLevelInfo {
                    name: "l01_escape".to_string(),
                    offset: [0.0; 3],
                    id: 1,
                    section: "level01".to_string(),
                    guid: [0x33, 0x44],
                }],
            },
            vertices: vec![GameVertex {
                local_point: [0.0; 3],
                global_point: [0.0; 3],
                level_id: 1,
                level_vertex_id: 77,
                vertex_types: [0, 0, 0, 0],
                edge_offset: 42,
                point_offset: 48,
                neighbour_count: 1,
                death_point_count: 1,
            }],
            edges: vec![GameEdge {
                target_vertex_id: 0,
                distance: 1.0,
            }],
            death_points: vec![GameDeathPoint {
                point: [1.0, 0.0, 1.0],
                level_vertex_id: 77,
                distance: 0.5,
            }],
            cross_tables: Vec::new(),
        };

        let alife_bytes = AlifeObjectState {
            game_vertex_id: 0,
            distance: 0.0,
            direct_control: true,
            level_vertex_id: 77,
            flags: alife_object_flags::CAN_SAVE,
            custom_data: String::new(),
            story_id: 100,
            spawn_story_id: 200,
            subclass_payload: Vec::new(),
        }
        .to_bytes();

        let archive = AlifeSpawnArchive {
            header: SpawnHeader {
                version: XRAI_CURRENT_VERSION,
                guid: [0xAA; 16],
                graph_guid: [0x11u64, 0x22].into_iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>().try_into().unwrap(),
                spawn_count: 2,
                level_count: 1,
            },
            spawns: vec![
                SpawnVertex {
                    spawn_id: 0,
                    spawn: SpawnPacket {
                        section: "actor".to_string(),
                        name_replace: "single_player".to_string(),
                        game_id_legacy: 0,
                        rp: 0xFE,
                        position: [0.0, 1.0, 0.0],
                        angle: [0.0; 3],
                        respawn_time: 0,
                        id: INVALID_OBJECT_ID,
                        parent_id: INVALID_OBJECT_ID,
                        phantom_id: INVALID_OBJECT_ID,
                        flags: m_spawn_flags::VERSION,
                        version: SPAWN_VERSION,
                        game_type: u16::MAX,
                        script_version: 10,
                        client_data: Vec::new(),
                        spawn_id: 0,
                        state_payload: alife_bytes.clone(),
                        trailing_update_payload: Vec::new(),
                    },
                    update_packet: M_UPDATE.to_le_bytes().to_vec(),
                },
                SpawnVertex {
                    spawn_id: 5,
                    spawn: SpawnPacket {
                        section: "medkit".to_string(),
                        name_replace: "esc_medkit_01".to_string(),
                        game_id_legacy: 0,
                        rp: 0xFE,
                        position: [2.0, 1.0, 0.0],
                        angle: [0.0; 3],
                        respawn_time: 0,
                        id: INVALID_OBJECT_ID,
                        parent_id: INVALID_OBJECT_ID,
                        phantom_id: INVALID_OBJECT_ID,
                        flags: m_spawn_flags::VERSION,
                        version: SPAWN_VERSION,
                        game_type: u16::MAX,
                        script_version: 10,
                        client_data: Vec::new(),
                        spawn_id: 5,
                        state_payload: alife_bytes,
                        trailing_update_payload: Vec::new(),
                    },
                    update_packet: M_UPDATE.to_le_bytes().to_vec(),
                },
            ],
            edges: vec![SpawnEdge {
                from: 0,
                to: 5,
                weight: 0.8,
            }],
            artefact_spawn_positions: vec![GameDeathPoint {
                point: [5.0, 0.0, 5.0],
                level_vertex_id: 99,
                distance: 1.5,
            }],
            patrol_paths: vec![PatrolPath {
                name: "esc_patrol_u32".to_string(),
                level_override: None,
                way_type: 0,
                points: vec![PatrolPoint {
                    vertex_id: 100_000,
                    name: "wp00".to_string(),
                    pos: [1.0, 0.0, 2.0],
                    flags: 0,
                    level_vertex_id: 77,
                    game_vertex_id: 0,
                }],
                edges: Vec::new(),
            }],
            game_graph: gg,
        };

        let raw = archive.to_bytes();
        let parsed = AlifeSpawnArchive::parse(&raw).unwrap();
        assert_eq!(parsed.header.guid, [0xAA; 16]);
        assert_eq!(parsed.find_spawn_by_name("esc_medkit_01").unwrap().spawn_id, 5);
        assert_eq!(parsed.edges, vec![SpawnEdge { from: 0, to: 5, weight: 0.8 }]);
        assert_eq!(parsed.artefact_spawn_positions.len(), 1);
        let pp = parsed.find_patrol_path("esc_patrol_u32").unwrap();
        assert_eq!(pp.points[0].vertex_id, 100_000);
        assert_eq!(parsed.game_graph.vertices[0].level_vertex_id, 77);
    }
}
