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
use crate::lanims::{unpack_argb_f32, LightAnimLibrary};
use crate::level::{d3d_light_type, LevelLight};
use crate::ogf::{matrix_forward_dir, matrix_set_xyz_translate};

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
    /// Parse `CSE_ALifeObject::STATE_Read` prefix from a shared cursor (`subclass_payload` is left empty).
    pub fn parse_prefix_from_cursor(
        version: u16,
        cur: &mut ByteCursor<'_>,
    ) -> Result<Self, FormatError> {
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

        Ok(Self {
            game_vertex_id,
            distance,
            direct_control,
            level_vertex_id,
            flags,
            custom_data,
            story_id,
            spawn_story_id,
            subclass_payload: Vec::new(),
        })
    }

    /// Encode only the `CSE_ALifeObject` header fields into `out` (omitting `subclass_payload`).
    pub fn write_prefix_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.game_vertex_id.to_le_bytes());
        out.extend_from_slice(&self.distance.to_le_bytes());
        out.extend_from_slice(&u32::from(self.direct_control).to_le_bytes());
        out.extend_from_slice(&self.level_vertex_id.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(self.custom_data.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.story_id.to_le_bytes());
        out.extend_from_slice(&self.spawn_story_id.to_le_bytes());
    }

    /// Parse `CSE_ALifeObject::STATE_Read` prefix and preserve remaining subclass state bytes.
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_prefix_from_cursor(version, &mut cur)?;
        parsed.subclass_payload = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    /// Encode in `SPAWN_VERSION` (128) wire layout (`CSE_ALifeObject::STATE_Write`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.custom_data.len() + self.subclass_payload.len());
        self.write_prefix_to(&mut out);
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
        let vals = s.split(',').filter_map(|p| p.trim().parse::<f32>().ok());
        let mut it = vals;
        Some([it.next()?, it.next()?, it.next()?])
    }
}

/// Collision shape attached to space restrictors / anomaly zones (`CSE_Shape`, `xrServer_Objects.cpp:24-85`).
#[derive(Debug, Clone, PartialEq)]
pub enum ServerShapeDef {
    Sphere { center: [f32; 3], radius: f32 },
    /// 4x3 transform matrix (`i`, `j`, `k`, `c`) as serialized by `NET_Packet::r_matrix` / `w_matrix`.
    Box([f32; 12]),
}

impl ServerShapeDef {
    /// Parse a `CSE_Shape::cform_read` sequence from `cur`.
    pub fn parse_cform_shapes(cur: &mut ByteCursor<'_>) -> Result<Vec<Self>, FormatError> {
        let count = usize::from(cur.read_u8()?);
        let mut shapes = Vec::with_capacity(count);
        for _ in 0..count {
            let kind = cur.read_u8()?;
            match kind {
                0 => {
                    let center = cur.read_f32_3()?;
                    let radius = cur.read_f32()?;
                    shapes.push(Self::Sphere { center, radius });
                }
                1 => {
                    let mut m = [0.0f32; 12];
                    for slot in &mut m {
                        *slot = cur.read_f32()?;
                    }
                    shapes.push(Self::Box(m));
                }
                _ => return Err(FormatError::InvalidValue("unknown CSE_Shape type")),
            }
        }
        Ok(shapes)
    }

    /// Encode a `CSE_Shape::cform_write` sequence into `out`.
    pub fn encode_cform_shapes(shapes: &[Self], out: &mut Vec<u8>) {
        out.push(shapes.len().min(u8::MAX as usize) as u8);
        for shape in shapes.iter().take(u8::MAX as usize) {
            match shape {
                Self::Sphere { center, radius } => {
                    out.push(0);
                    for f in center {
                        out.extend_from_slice(&f.to_le_bytes());
                    }
                    out.extend_from_slice(&radius.to_le_bytes());
                }
                Self::Box(m) => {
                    out.push(1);
                    for f in m {
                        out.extend_from_slice(&f.to_le_bytes());
                    }
                }
            }
        }
    }
}

/// Visual reference embedded in `CSE_Visual::visual_read` (`xrServer_Objects_Abstract.cpp:50-61`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerVisualState {
    pub visual_name: String,
    pub flags: u8,
}

impl ServerVisualState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let visual_name = cur.read_zstring()?;
        let flags = if version > 103 { cur.read_u8()? } else { 0 };
        Ok(Self { visual_name, flags })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.visual_name.as_bytes());
        out.push(0);
        out.push(self.flags);
    }
}

/// Decoded `CSE_ALifeDynamicObjectVisual::STATE_Read` (`xrServer_Objects_ALife.cpp:741-746`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeDynamicVisualState {
    pub alife: AlifeObjectState,
    pub visual: ServerVisualState,
    pub trailing: Vec<u8>,
}

impl AlifeDynamicVisualState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let alife = AlifeObjectState::parse_prefix_from_cursor(version, cur)?;
        let visual = if version > 31 {
            ServerVisualState::parse_from_cursor(version, cur)?
        } else {
            ServerVisualState::default()
        };
        Ok(Self {
            alife,
            visual,
            trailing: Vec::new(),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.alife.write_prefix_to(out);
        self.visual.write_to(out);
        out.extend_from_slice(&self.trailing);
    }

    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_from_cursor(version, &mut cur)?;
        parsed.trailing = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }
}

/// Decoded `CSE_ALifeSpaceRestrictor::STATE_Read` (`xrServer_Objects_ALife.cpp:862-875`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeSpaceRestrictorState {
    pub alife: AlifeObjectState,
    pub shapes: Vec<ServerShapeDef>,
    pub restrictor_type: u8,
    pub trailing: Vec<u8>,
}

impl AlifeSpaceRestrictorState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let alife = AlifeObjectState::parse_prefix_from_cursor(version, cur)?;
        let shapes = ServerShapeDef::parse_cform_shapes(cur)?;
        let restrictor_type = if version > 74 { cur.read_u8()? } else { 0 };
        Ok(Self {
            alife,
            shapes,
            restrictor_type,
            trailing: Vec::new(),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.alife.write_prefix_to(out);
        ServerShapeDef::encode_cform_shapes(&self.shapes, out);
        out.push(self.restrictor_type);
        out.extend_from_slice(&self.trailing);
    }

    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_from_cursor(version, &mut cur)?;
        parsed.trailing = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }
}

/// Decoded `CSE_ALifeLevelChanger::STATE_Read` (`xrServer_Objects_ALife.cpp:926-951`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeLevelChangerState {
    pub restrictor: AlifeSpaceRestrictorState,
    pub next_graph_vertex: u16,
    pub next_level_vertex: u32,
    pub next_position: [f32; 3],
    pub next_angles: [f32; 3],
    pub level_to_change: String,
    pub level_point_to_change: String,
    pub silent_mode: bool,
    pub trailing: Vec<u8>,
}

impl AlifeLevelChangerState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut restrictor = AlifeSpaceRestrictorState::parse(version, state_bytes)?;
        let payload = std::mem::take(&mut restrictor.trailing);
        let mut cur = ByteCursor::new(&payload);

        let (next_graph_vertex, next_level_vertex, next_position, next_angles) = if version < 34 {
            let _ = cur.read_u32()?;
            let _ = cur.read_u32()?;
            (INVALID_GAME_VERTEX, INVALID_AI_VERTEX, [0.0; 3], [0.0; 3])
        } else {
            let gvid = cur.read_u16()?;
            let lvid = cur.read_u32()?;
            let pos = cur.read_f32_3()?;
            let ang = if version <= 53 {
                [0.0, cur.read_f32()?, 0.0]
            } else {
                cur.read_f32_3()?
            };
            (gvid, lvid, pos, ang)
        };

        let level_to_change = cur.read_zstring()?;
        let level_point_to_change = cur.read_zstring()?;
        let silent_mode = if version > 116 {
            cur.read_u8()? != 0
        } else {
            false
        };
        let trailing = payload[cur.position()..].to_vec();

        Ok(Self {
            restrictor,
            next_graph_vertex,
            next_level_vertex,
            next_position,
            next_angles,
            level_to_change,
            level_point_to_change,
            silent_mode,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut rest = Vec::new();
        rest.extend_from_slice(&self.next_graph_vertex.to_le_bytes());
        rest.extend_from_slice(&self.next_level_vertex.to_le_bytes());
        for f in self.next_position {
            rest.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.next_angles {
            rest.extend_from_slice(&f.to_le_bytes());
        }
        rest.extend_from_slice(self.level_to_change.as_bytes());
        rest.push(0);
        rest.extend_from_slice(self.level_point_to_change.as_bytes());
        rest.push(0);
        rest.push(u8::from(self.silent_mode));
        rest.extend_from_slice(&self.trailing);

        let mut r = self.restrictor.clone();
        r.trailing = rest;
        r.to_bytes()
    }
}

/// Decoded `CSE_ALifeItem::STATE_Read` (`CSE_ALifeDynamicObjectVisual` + `CSE_ALifeInventoryItem`,
/// `xrServer_Objects_ALife_Items.cpp:94-106,359-369`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeInventoryItemState {
    pub base: AlifeDynamicVisualState,
    pub condition: f32,
    pub upgrades: Vec<String>,
    pub trailing: Vec<u8>,
}

impl AlifeInventoryItemState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let base = AlifeDynamicVisualState::parse_from_cursor(version, cur)?;
        let condition = if version > 52 { cur.read_f32()? } else { 1.0 };
        let mut upgrades = Vec::new();
        if version > 123 {
            let count = cur.read_u32()? as usize;
            upgrades.reserve(count.min(4096));
            for _ in 0..count {
                upgrades.push(cur.read_zstring()?);
            }
        }
        Ok(Self {
            base,
            condition,
            upgrades,
            trailing: Vec::new(),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.base.write_to(out);
        out.extend_from_slice(&self.condition.to_le_bytes());
        out.extend_from_slice(&(self.upgrades.len() as u32).to_le_bytes());
        for u in &self.upgrades {
            out.extend_from_slice(u.as_bytes());
            out.push(0);
        }
        out.extend_from_slice(&self.trailing);
    }

    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_from_cursor(version, &mut cur)?;
        parsed.trailing = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }
}

/// Decoded `CSE_ALifeItemWeapon::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:566-592`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeItemWeaponState {
    pub item: AlifeInventoryItemState,
    pub ammo_current: u16,
    pub ammo_elapsed: u16,
    pub wpn_state: u8,
    pub addon_flags: u8,
    pub ammo_type: u8,
    pub ammo_elapsed_grenades: u8,
    pub trailing: Vec<u8>,
}

impl AlifeItemWeaponState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut item = AlifeInventoryItemState::parse(version, state_bytes)?;
        let payload = std::mem::take(&mut item.trailing);
        let mut cur = ByteCursor::new(&payload);

        let ammo_current = cur.read_u16()?;
        let ammo_elapsed = cur.read_u16()?;
        let wpn_state = cur.read_u8()?;
        let addon_flags = if version > 40 { cur.read_u8()? } else { 0 };
        let ammo_type = if version > 46 { cur.read_u8()? } else { 0 };
        let ammo_elapsed_grenades = if version > 122 { cur.read_u8()? } else { 0 };
        let trailing = payload[cur.position()..].to_vec();

        Ok(Self {
            item,
            ammo_current,
            ammo_elapsed,
            wpn_state,
            addon_flags,
            ammo_type,
            ammo_elapsed_grenades,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut tail = Vec::new();
        tail.extend_from_slice(&self.ammo_current.to_le_bytes());
        tail.extend_from_slice(&self.ammo_elapsed.to_le_bytes());
        tail.push(self.wpn_state);
        tail.push(self.addon_flags);
        tail.push(self.ammo_type);
        tail.push(self.ammo_elapsed_grenades);
        tail.extend_from_slice(&self.trailing);

        let mut it = self.item.clone();
        it.trailing = tail;
        it.to_bytes()
    }
}

/// Decoded `CSE_ALifeTraderAbstract::STATE_Read` tail (`xrServer_Objects_ALife_Monsters.cpp:218-300`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AlifeTraderAbstractState {
    pub money: u32,
    pub specific_character: String,
    pub trader_flags: u32,
    pub character_profile: String,
    pub community_index: i32,
    pub rank: i32,
    pub reputation: i32,
    pub character_name: String,
    pub deadbody_can_take: bool,
    pub deadbody_closed: bool,
}

impl AlifeTraderAbstractState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let mut money = 0u32;
        let mut specific_character = String::new();
        let mut trader_flags = 0u32;
        let mut character_profile = String::new();
        let mut community_index = -1i32;
        let mut rank = i32::MIN;
        let mut reputation = i32::MIN;
        let mut character_name = String::new();

        if version > 19 {
            if version < 108 {
                let _ = cur.read_u32()?;
            }
            if version < 36 {
                let n = cur.read_u32()? as usize;
                let _ = cur.read_bytes(n.saturating_mul(2))?;
            }
            if version > 62 {
                money = cur.read_u32()?;
            }
            if version > 75 && version < 98 {
                let _ = cur.read_i32()?;
            } else if version >= 98 {
                specific_character = cur.read_zstring()?;
            }
            if version > 77 {
                trader_flags = cur.read_u32()?;
            }
            if version > 81 && version < 96 {
                let _ = cur.read_i32()?;
            } else if version > 95 {
                character_profile = cur.read_zstring()?;
            }
            if version > 85 {
                community_index = cur.read_i32()?;
            }
            if version > 86 {
                rank = cur.read_i32()?;
                reputation = cur.read_i32()?;
            }
            if version > 104 {
                character_name = cur.read_zstring()?;
            }
        }

        let (deadbody_can_take, deadbody_closed) = if version > 124 {
            let can_take = cur.read_u8()? == 1;
            let closed = cur.read_u8()? == 1;
            (can_take, closed)
        } else {
            (true, false)
        };

        Ok(Self {
            money,
            specific_character,
            trader_flags,
            character_profile,
            community_index,
            rank,
            reputation,
            character_name,
            deadbody_can_take,
            deadbody_closed,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.money.to_le_bytes());
        out.extend_from_slice(self.specific_character.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.trader_flags.to_le_bytes());
        out.extend_from_slice(self.character_profile.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.community_index.to_le_bytes());
        out.extend_from_slice(&self.rank.to_le_bytes());
        out.extend_from_slice(&self.reputation.to_le_bytes());
        out.extend_from_slice(self.character_name.as_bytes());
        out.push(0);
        out.push(u8::from(self.deadbody_can_take));
        out.push(u8::from(self.deadbody_closed));
    }
}

/// Animation reference embedded by `CSE_Motion` (`xrServer_Objects_Abstract.cpp:106-114`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerMotionState {
    pub motion_name: String,
}

impl ServerMotionState {
    pub fn parse_from_cursor(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        Ok(Self {
            motion_name: cur.read_zstring()?,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.motion_name.as_bytes());
        out.push(0);
    }
}

/// Bitflags on `CSE_PHSkeleton::_flags` (`xrServer_Objects.h:194-200`).
pub mod ph_skeleton_flags {
    pub const ACTIVE: u8 = 1 << 0;
    pub const SPAWN_COPY: u8 = 1 << 1;
    pub const SAVED_DATA: u8 = 1 << 2;
    pub const NOT_SAVE: u8 = 1 << 3;
}

/// Quantized ragdoll/skeleton bone state (`SPHNetState`, `PHNetState.cpp:8-71, 183-210`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ServerPhBoneState {
    pub pos_q8: [u8; 3],
    pub quat_q8: [u8; 4],
    pub enabled: bool,
}

impl ServerPhBoneState {
    #[inline]
    pub fn quantize_float_q8(val: f32, min: f32, max: f32) -> u8 {
        let span = max - min;
        if span.abs() <= f32::EPSILON {
            return 0;
        }
        let clamped = val.clamp(min, max);
        let q = (clamped - min) / span;
        (q * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8
    }

    #[inline]
    pub fn dequantize_float_q8(q: u8, min: f32, max: f32) -> f32 {
        (f32::from(q) / 255.0 * (max - min) + min).clamp(min, max)
    }

    pub fn from_world_position_quat(
        pos: [f32; 3],
        quat_xyzw: [f32; 4],
        min: [f32; 3],
        max: [f32; 3],
        enabled: bool,
    ) -> Self {
        Self {
            pos_q8: [
                Self::quantize_float_q8(pos[0], min[0], max[0]),
                Self::quantize_float_q8(pos[1], min[1], max[1]),
                Self::quantize_float_q8(pos[2], min[2], max[2]),
            ],
            quat_q8: [
                Self::quantize_float_q8(quat_xyzw[0], -1.0, 1.0),
                Self::quantize_float_q8(quat_xyzw[1], -1.0, 1.0),
                Self::quantize_float_q8(quat_xyzw[2], -1.0, 1.0),
                Self::quantize_float_q8(quat_xyzw[3], -1.0, 1.0),
            ],
            enabled,
        }
    }

    pub fn decode_position(&self, min: [f32; 3], max: [f32; 3]) -> [f32; 3] {
        [
            Self::dequantize_float_q8(self.pos_q8[0], min[0], max[0]),
            Self::dequantize_float_q8(self.pos_q8[1], min[1], max[1]),
            Self::dequantize_float_q8(self.pos_q8[2], min[2], max[2]),
        ]
    }

    pub fn decode_quaternion(&self) -> [f32; 4] {
        [
            Self::dequantize_float_q8(self.quat_q8[0], -1.0, 1.0),
            Self::dequantize_float_q8(self.quat_q8[1], -1.0, 1.0),
            Self::dequantize_float_q8(self.quat_q8[2], -1.0, 1.0),
            Self::dequantize_float_q8(self.quat_q8[3], -1.0, 1.0),
        ]
    }

    pub fn parse_from_cursor(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let p = cur.read_bytes(3)?;
        let q = cur.read_bytes(4)?;
        let enabled = cur.read_u8()? != 0;
        Ok(Self {
            pos_q8: [p[0], p[1], p[2]],
            quat_q8: [q[0], q[1], q[2], q[3]],
            enabled,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.pos_q8);
        out.extend_from_slice(&self.quat_q8);
        out.push(u8::from(self.enabled));
    }
}

/// Physics skeleton saved bone state (`SPHBonesData`, `PHNetState.cpp:236-279`).
#[derive(Debug, Clone, PartialEq)]
pub struct ServerPhBonesData {
    pub bones_mask: u64,
    pub root_bone: u16,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub bones: Vec<ServerPhBoneState>,
}

impl Default for ServerPhBonesData {
    fn default() -> Self {
        Self {
            bones_mask: u64::MAX,
            root_bone: 0,
            min: [-100.0; 3],
            max: [100.0; 3],
            bones: Vec::new(),
        }
    }
}

impl ServerPhBonesData {
    pub fn parse_from_cursor(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let bones_mask = cur.read_u64()?;
        let root_bone = cur.read_u16()?;
        let min = cur.read_f32_3()?;
        let max = cur.read_f32_3()?;
        let count = usize::from(cur.read_u16()?);
        let mut bones = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            bones.push(ServerPhBoneState::parse_from_cursor(cur)?);
        }
        Ok(Self {
            bones_mask,
            root_bone,
            min,
            max,
            bones,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.bones_mask.to_le_bytes());
        out.extend_from_slice(&self.root_bone.to_le_bytes());
        for f in self.min {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.max {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&(self.bones.len().min(u16::MAX as usize) as u16).to_le_bytes());
        for bone in self.bones.iter().take(u16::MAX as usize) {
            bone.write_to(out);
        }
    }
}

/// Decoded `CSE_PHSkeleton::STATE_Read` (`xrServer_Objects.cpp:233-258`).
#[derive(Debug, Clone, PartialEq)]
pub struct ServerPhSkeletonState {
    pub startup_animation: String,
    pub flags: u8,
    pub source_id: u16,
    pub saved_bones: Option<ServerPhBonesData>,
}

impl Default for ServerPhSkeletonState {
    fn default() -> Self {
        Self {
            startup_animation: String::new(),
            flags: 0,
            source_id: INVALID_OBJECT_ID,
            saved_bones: None,
        }
    }
}

impl ServerPhSkeletonState {
    pub fn parse_from_cursor(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let startup_animation = cur.read_zstring()?;
        let flags = cur.read_u8()?;
        let source_id = cur.read_u16()?;
        let saved_bones = if (flags & ph_skeleton_flags::SAVED_DATA) != 0 {
            Some(ServerPhBonesData::parse_from_cursor(cur)?)
        } else {
            None
        };
        Ok(Self {
            startup_animation,
            flags,
            source_id,
            saved_bones,
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.startup_animation.as_bytes());
        out.push(0);
        let effective_flags = if self.saved_bones.is_some() {
            self.flags | ph_skeleton_flags::SAVED_DATA
        } else {
            self.flags & !ph_skeleton_flags::SAVED_DATA
        };
        out.push(effective_flags);
        out.extend_from_slice(&self.source_id.to_le_bytes());
        if let Some(bones) = &self.saved_bones {
            bones.write_to(out);
        }
    }
}

/// Decoded `CSE_ALifeCustomZone::STATE_Read` (`xrServer_Objects_ALife.cpp:723-763`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeCustomZoneState {
    pub restrictor: AlifeSpaceRestrictorState,
    pub max_power: f32,
    pub owner_id: u32,
    pub enabled_time: u32,
    pub disabled_time: u32,
    pub start_time_shift: u32,
    pub trailing: Vec<u8>,
}

impl AlifeCustomZoneState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let restrictor = AlifeSpaceRestrictorState::parse_from_cursor(version, cur)?;
        let max_power = cur.read_f32()?;
        if version < 113 {
            let _ = cur.read_f32()?;
            let _ = cur.read_u32()?;
        }
        if version > 66 && version < 118 {
            let _ = cur.read_u32()?;
        }
        let owner_id = if version > 102 {
            cur.read_u32()?
        } else {
            u32::MAX
        };
        let (enabled_time, disabled_time) = if version > 105 {
            (cur.read_u32()?, cur.read_u32()?)
        } else {
            (0, 0)
        };
        let start_time_shift = if version > 106 { cur.read_u32()? } else { 0 };
        Ok(Self {
            restrictor,
            max_power,
            owner_id,
            enabled_time,
            disabled_time,
            start_time_shift,
            trailing: Vec::new(),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.restrictor.write_to(out);
        out.extend_from_slice(&self.max_power.to_le_bytes());
        out.extend_from_slice(&self.owner_id.to_le_bytes());
        out.extend_from_slice(&self.enabled_time.to_le_bytes());
        out.extend_from_slice(&self.disabled_time.to_le_bytes());
        out.extend_from_slice(&self.start_time_shift.to_le_bytes());
        out.extend_from_slice(&self.trailing);
    }

    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_from_cursor(version, &mut cur)?;
        parsed.trailing = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }
}

/// Decoded `CSE_ALifeAnomalousZone::STATE_Read` (`xrServer_Objects_ALife.cpp:830-888`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeAnomalousZoneState {
    pub zone: AlifeCustomZoneState,
    pub offline_interactive_radius: f32,
    pub artefact_spawn_count: u16,
    pub artefact_position_offset: u32,
    pub trailing: Vec<u8>,
}

impl AlifeAnomalousZoneState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let zone = AlifeCustomZoneState::parse_from_cursor(version, cur)?;
        let offline_interactive_radius = if version > 21 {
            let r = cur.read_f32()?;
            if version < 113 {
                let _ = cur.read_f32()?;
                let n = cur.read_u16()?;
                for _ in 0..n {
                    let _ = cur.read_zstring()?;
                    let _ = cur.read_u32()?;
                }
            }
            r
        } else {
            30.0
        };
        let (artefact_spawn_count, artefact_position_offset) = if version > 25 {
            (cur.read_u16()?, cur.read_u32()?)
        } else {
            (32, 0)
        };
        if version > 27 && version < 67 {
            let _ = cur.read_u32()?;
        }
        if version > 38 && version < 113 {
            let _ = cur.read_f32()?;
        }
        if version > 78 && version < 113 {
            let _ = cur.read_f32()?;
            let _ = cur.read_f32()?;
            let _ = cur.read_f32()?;
        }
        if version == 102 {
            let _ = cur.read_u32()?;
        }
        Ok(Self {
            zone,
            offline_interactive_radius,
            artefact_spawn_count,
            artefact_position_offset,
            trailing: Vec::new(),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.zone.write_to(out);
        out.extend_from_slice(&self.offline_interactive_radius.to_le_bytes());
        out.extend_from_slice(&self.artefact_spawn_count.to_le_bytes());
        out.extend_from_slice(&self.artefact_position_offset.to_le_bytes());
        out.extend_from_slice(&self.trailing);
    }

    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_from_cursor(version, &mut cur)?;
        parsed.trailing = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }
}

/// Decoded `CSE_ALifeTorridZone::STATE_Read` (`xrServer_Objects_ALife.cpp:927-938`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeTorridZoneState {
    pub zone: AlifeCustomZoneState,
    pub motion: ServerMotionState,
    pub trailing: Vec<u8>,
}

impl AlifeTorridZoneState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let zone = AlifeCustomZoneState::parse_from_cursor(version, &mut cur)?;
        let motion = ServerMotionState::parse_from_cursor(&mut cur)?;
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            zone,
            motion,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.zone.write_to(&mut out);
        self.motion.write_to(&mut out);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeZoneVisual::STATE_Read` (`xrServer_Objects_ALife.cpp:979-993`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeZoneVisualState {
    pub anomaly: AlifeAnomalousZoneState,
    pub visual: ServerVisualState,
    pub startup_animation: String,
    pub attack_animation: String,
    pub trailing: Vec<u8>,
}

impl AlifeZoneVisualState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let anomaly = AlifeAnomalousZoneState::parse_from_cursor(version, &mut cur)?;
        let visual = ServerVisualState::parse_from_cursor(version, &mut cur)?;
        let startup_animation = cur.read_zstring()?;
        let attack_animation = cur.read_zstring()?;
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            anomaly,
            visual,
            startup_animation,
            attack_animation,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.anomaly.write_to(&mut out);
        self.visual.write_to(&mut out);
        out.extend_from_slice(self.startup_animation.as_bytes());
        out.push(0);
        out.extend_from_slice(self.attack_animation.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// `CSE_ALifeSmartZone::STATE_Read` (`xrServer_Objects_ALife.cpp:2463-2471`) is wire-identical to `CSE_ALifeSpaceRestrictor`.
pub type AlifeSmartZoneState = AlifeSpaceRestrictorState;

/// Decoded `CSE_ALifeTeamBaseZone::STATE_Read` (`xrServer_Objects_ALife.cpp:2403-2413`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeTeamBaseZoneState {
    pub restrictor: AlifeSpaceRestrictorState,
    pub team: u8,
    pub trailing: Vec<u8>,
}

impl AlifeTeamBaseZoneState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let restrictor = AlifeSpaceRestrictorState::parse_from_cursor(version, &mut cur)?;
        let team = cur.read_u8()?;
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            restrictor,
            team,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.restrictor.write_to(&mut out);
        out.push(self.team);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeObjectPhysic::STATE_Read` (`xrServer_Objects_ALife.cpp:1030-1079`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeObjectPhysicState {
    pub base: AlifeDynamicVisualState,
    pub skeleton: ServerPhSkeletonState,
    /// Physics type (`0=Box`, `1=FixedChain`, `2=FreeChain`, `3=Skeleton`).
    pub physic_type: u32,
    pub mass: f32,
    pub fixed_bones: String,
    pub trailing: Vec<u8>,
}

impl AlifeObjectPhysicState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let base = if version >= 14 {
            let alife = AlifeObjectState::parse_prefix_from_cursor(version, &mut cur)?;
            let visual = ServerVisualState::parse_from_cursor(version, &mut cur)?;
            AlifeDynamicVisualState {
                alife,
                visual,
                trailing: Vec::new(),
            }
        } else {
            AlifeDynamicVisualState {
                alife: AlifeObjectState {
                    game_vertex_id: INVALID_GAME_VERTEX,
                    distance: 0.0,
                    direct_control: true,
                    level_vertex_id: INVALID_AI_VERTEX,
                    flags: 0,
                    custom_data: String::new(),
                    story_id: INVALID_STORY_ID,
                    spawn_story_id: INVALID_SPAWN_STORY_ID,
                    subclass_payload: Vec::new(),
                },
                visual: ServerVisualState::default(),
                trailing: Vec::new(),
            }
        };

        let mut skeleton = if version >= 64 {
            ServerPhSkeletonState::parse_from_cursor(&mut cur)?
        } else {
            ServerPhSkeletonState::default()
        };

        let physic_type = cur.read_u32()?;
        let mass = cur.read_f32()?;
        let fixed_bones = if version > 9 {
            cur.read_zstring()?
        } else {
            String::new()
        };

        if version < 65 && version > 28 {
            skeleton.startup_animation = cur.read_zstring()?;
        }
        if version < 64 {
            if version > 39 {
                skeleton.flags = cur.read_u8()?;
            }
            if version > 56 {
                skeleton.source_id = cur.read_u16()?;
            }
            if version > 60 && (skeleton.flags & ph_skeleton_flags::SAVED_DATA) != 0 {
                skeleton.saved_bones = Some(ServerPhBonesData::parse_from_cursor(&mut cur)?);
            }
        }

        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            base,
            skeleton,
            physic_type,
            mass,
            fixed_bones,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.base.write_to(&mut out);
        self.skeleton.write_to(&mut out);
        out.extend_from_slice(&self.physic_type.to_le_bytes());
        out.extend_from_slice(&self.mass.to_le_bytes());
        out.extend_from_slice(self.fixed_bones.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeInventoryBox::STATE_Read` (`xrServer_Objects_ALife.cpp:2518-2540`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeInventoryBoxState {
    pub base: AlifeDynamicVisualState,
    pub can_take: bool,
    pub closed: bool,
    pub tip_text: String,
    pub trailing: Vec<u8>,
}

impl AlifeInventoryBoxState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let base = AlifeDynamicVisualState::parse_from_cursor(version, &mut cur)?;
        let (can_take, closed, tip_text) = if version > 124 {
            let can_take = cur.read_u8()? == 1;
            let closed = cur.read_u8()? == 1;
            let tip_text = cur.read_zstring()?;
            (can_take, closed, tip_text)
        } else {
            (true, false, "inventory_box_use".to_string())
        };
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            base,
            can_take,
            closed,
            tip_text,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.base.write_to(&mut out);
        out.push(u8::from(self.can_take));
        out.push(u8::from(self.closed));
        out.extend_from_slice(self.tip_text.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeItemAmmo::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:860-870`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeItemAmmoState {
    pub item: AlifeInventoryItemState,
    pub ammo_left: u16,
    pub trailing: Vec<u8>,
}

impl AlifeItemAmmoState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let item = AlifeInventoryItemState::parse_from_cursor(version, &mut cur)?;
        let ammo_left = cur.read_u16()?;
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            item,
            ammo_left,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.item.write_to(&mut out);
        out.extend_from_slice(&self.ammo_left.to_le_bytes());
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeItemPDA::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:1011-1048`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeItemPdaState {
    pub item: AlifeInventoryItemState,
    pub original_owner: u16,
    pub specific_character: String,
    pub info_portion: String,
    pub trailing: Vec<u8>,
}

impl AlifeItemPdaState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let item = AlifeInventoryItemState::parse_from_cursor(version, &mut cur)?;
        let original_owner = if version > 58 {
            cur.read_u16()?
        } else {
            INVALID_OBJECT_ID
        };
        let (specific_character, info_portion) = if version > 89 && version < 98 {
            let _ = cur.read_i32()?;
            let _ = cur.read_i32()?;
            (String::new(), String::new())
        } else if version >= 98 {
            (cur.read_zstring()?, cur.read_zstring()?)
        } else {
            (String::new(), String::new())
        };
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            item,
            original_owner,
            specific_character,
            info_portion,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.item.write_to(&mut out);
        out.extend_from_slice(&self.original_owner.to_le_bytes());
        out.extend_from_slice(self.specific_character.as_bytes());
        out.push(0);
        out.extend_from_slice(self.info_portion.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeItemDocument::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:1079-1097`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeItemDocumentState {
    pub item: AlifeInventoryItemState,
    pub info_portion: String,
    pub trailing: Vec<u8>,
}

impl AlifeItemDocumentState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let item = AlifeInventoryItemState::parse_from_cursor(version, &mut cur)?;
        let info_portion = if version < 98 {
            let _ = cur.read_u16()?;
            String::new()
        } else {
            cur.read_zstring()?
        };
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            item,
            info_portion,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.item.write_to(&mut out);
        out.extend_from_slice(self.info_portion.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeCreatureAbstract::STATE_Read` (`xrServer_Objects_ALife_Monsters.cpp:1074-1117`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeCreatureAbstractState {
    pub base: AlifeDynamicVisualState,
    pub team: u8,
    pub squad: u8,
    pub group: u8,
    pub health: f32,
    pub dynamic_out_restrictions: Vec<u16>,
    pub dynamic_in_restrictions: Vec<u16>,
    pub killer_id: u16,
    pub game_death_time: u64,
    pub trailing: Vec<u8>,
}

impl AlifeCreatureAbstractState {
    fn read_u16_vec(cur: &mut ByteCursor<'_>) -> Result<Vec<u16>, FormatError> {
        let count = cur.read_u32()? as usize;
        let mut out = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            out.push(cur.read_u16()?);
        }
        Ok(out)
    }

    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let mut base = AlifeDynamicVisualState::parse_from_cursor(version, cur)?;
        let team = cur.read_u8()?;
        let squad = cur.read_u8()?;
        let group = cur.read_u8()?;
        let mut health = if version > 18 { cur.read_f32()? } else { 1.0 };
        if version < 115 {
            health /= 100.0;
        }
        if version < 32 {
            base.visual = ServerVisualState::parse_from_cursor(version, cur)?;
        }
        let (dynamic_out_restrictions, dynamic_in_restrictions) = if version > 87 {
            (Self::read_u16_vec(cur)?, Self::read_u16_vec(cur)?)
        } else {
            (Vec::new(), Vec::new())
        };
        let killer_id = if version > 94 {
            cur.read_u16()?
        } else {
            INVALID_OBJECT_ID
        };
        let game_death_time = if version > 115 { cur.read_u64()? } else { 0 };
        Ok(Self {
            base,
            team,
            squad,
            group,
            health,
            dynamic_out_restrictions,
            dynamic_in_restrictions,
            killer_id,
            game_death_time,
            trailing: Vec::new(),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.base.write_to(out);
        out.push(self.team);
        out.push(self.squad);
        out.push(self.group);
        out.extend_from_slice(&self.health.to_le_bytes());
        out.extend_from_slice(&(self.dynamic_out_restrictions.len() as u32).to_le_bytes());
        for id in &self.dynamic_out_restrictions {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&(self.dynamic_in_restrictions.len() as u32).to_le_bytes());
        for id in &self.dynamic_in_restrictions {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&self.killer_id.to_le_bytes());
        out.extend_from_slice(&self.game_death_time.to_le_bytes());
        out.extend_from_slice(&self.trailing);
    }

    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_from_cursor(version, &mut cur)?;
        parsed.trailing = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }
}

/// Decoded `CSE_ALifeMonsterAbstract::STATE_Read` (`xrServer_Objects_ALife_Monsters.cpp:1326-1363`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeMonsterAbstractState {
    pub creature: AlifeCreatureAbstractState,
    pub out_space_restrictors: String,
    pub in_space_restrictors: String,
    pub smart_terrain_id: u16,
    pub task_reached: bool,
    pub trailing: Vec<u8>,
}

impl AlifeMonsterAbstractState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let creature = AlifeCreatureAbstractState::parse_from_cursor(version, cur)?;
        let (out_space_restrictors, in_space_restrictors) = if version > 72 {
            let out_r = cur.read_zstring()?;
            let in_r = if version > 73 {
                cur.read_zstring()?
            } else {
                String::new()
            };
            (out_r, in_r)
        } else {
            (String::new(), String::new())
        };
        let smart_terrain_id = if version > 111 {
            cur.read_u16()?
        } else {
            INVALID_OBJECT_ID
        };
        let task_reached = if version > 113 {
            cur.read_u8()? != 0
        } else {
            false
        };
        Ok(Self {
            creature,
            out_space_restrictors,
            in_space_restrictors,
            smart_terrain_id,
            task_reached,
            trailing: Vec::new(),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.creature.write_to(out);
        out.extend_from_slice(self.out_space_restrictors.as_bytes());
        out.push(0);
        out.extend_from_slice(self.in_space_restrictors.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.smart_terrain_id.to_le_bytes());
        out.push(u8::from(self.task_reached));
        out.extend_from_slice(&self.trailing);
    }

    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_from_cursor(version, &mut cur)?;
        parsed.trailing = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }
}

/// Decoded `CSE_ALifeMonsterBase::STATE_Read` (`xrServer_Objects_ALife_Monsters.cpp:1996-2012`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeMonsterBaseState {
    pub monster: AlifeMonsterAbstractState,
    pub skeleton: ServerPhSkeletonState,
    pub spec_object_id: u16,
    pub trailing: Vec<u8>,
}

impl AlifeMonsterBaseState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let monster = AlifeMonsterAbstractState::parse_from_cursor(version, &mut cur)?;
        let skeleton = if version >= 68 {
            ServerPhSkeletonState::parse_from_cursor(&mut cur)?
        } else {
            ServerPhSkeletonState::default()
        };
        let spec_object_id = if version >= 109 {
            cur.read_u16()?
        } else {
            INVALID_OBJECT_ID
        };
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            monster,
            skeleton,
            spec_object_id,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.monster.write_to(&mut out);
        self.skeleton.write_to(&mut out);
        out.extend_from_slice(&self.spec_object_id.to_le_bytes());
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeTrader::STATE_Read` (`xrServer_Objects_ALife_Monsters.cpp:634-676`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeTraderState {
    pub base: AlifeDynamicVisualState,
    pub trader: AlifeTraderAbstractState,
    pub trailing: Vec<u8>,
}

impl AlifeTraderState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let base = AlifeDynamicVisualState::parse_from_cursor(version, &mut cur)?;
        let trader = AlifeTraderAbstractState::parse_from_cursor(version, &mut cur)?;
        if version > 35 && version < 118 {
            let _ = cur.read_u32()?;
        }
        if version > 29 && version < 118 {
            let count = cur.read_u32()?;
            for _ in 0..count {
                let _ = cur.read_zstring()?;
                let _ = cur.read_u32()?;
                let n = cur.read_u32()?;
                for _ in 0..n {
                    let _ = cur.read_zstring()?;
                    let _ = cur.read_u32()?;
                    let _ = cur.read_u32()?;
                }
            }
        }
        if version > 30 && version < 118 {
            let count = cur.read_u32()?;
            for _ in 0..count {
                let _ = cur.read_zstring()?;
                let _ = cur.read_u32()?;
                let _ = cur.read_f32()?;
                let _ = cur.read_f32()?;
            }
        }
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            base,
            trader,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.base.write_to(&mut out);
        self.trader.write_to(&mut out);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeHumanAbstract::STATE_Read` (`xrServer_Objects_ALife_Monsters.cpp:2081-2128` + `alife_human_brain.cpp:75-121`).
/// Note the C++ multiple-inheritance wire order: `CSE_ALifeTraderAbstract` comes BEFORE `CSE_ALifeMonsterAbstract`.
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeHumanAbstractState {
    pub trader: AlifeTraderAbstractState,
    pub monster: AlifeMonsterAbstractState,
    pub equipment_prefs: Vec<u8>,
    pub main_weapon_prefs: Vec<u8>,
    pub trailing: Vec<u8>,
}

impl AlifeHumanAbstractState {
    pub fn parse_from_cursor(version: u16, cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let trader = AlifeTraderAbstractState::parse_from_cursor(version, cur)?;
        let mut monster = AlifeMonsterAbstractState::parse_from_cursor(version, cur)?;

        // CALifeHumanBrain::on_state_read
        let mut equipment_prefs = Vec::new();
        let mut main_weapon_prefs = Vec::new();
        if version > 19 {
            if version < 110 {
                let n = cur.read_u32()? as usize;
                let _ = cur.read_bytes(n.saturating_mul(4))?;
                let b = cur.read_u32()? as usize;
                let _ = cur.read_bytes(b)?;
            }
            if version > 35 {
                if version < 110 {
                    let _ = cur.read_zstring()?;
                }
                if version < 118 {
                    let n = cur.read_u32()? as usize;
                    let _ = cur.read_bytes(n.saturating_mul(2))?;
                }
                let eq_len = cur.read_u32()? as usize;
                equipment_prefs = cur.read_bytes(eq_len)?.to_vec();
                let wp_len = cur.read_u32()? as usize;
                main_weapon_prefs = cur.read_bytes(wp_len)?.to_vec();
            }
        }
        if (110..112).contains(&version) {
            monster.smart_terrain_id = cur.read_u16()?;
        }

        Ok(Self {
            trader,
            monster,
            equipment_prefs,
            main_weapon_prefs,
            trailing: Vec::new(),
        })
    }

    pub fn write_to(&self, out: &mut Vec<u8>) {
        self.trader.write_to(out);
        self.monster.write_to(out);
        out.extend_from_slice(&(self.equipment_prefs.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.equipment_prefs);
        out.extend_from_slice(&(self.main_weapon_prefs.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.main_weapon_prefs);
        out.extend_from_slice(&self.trailing);
    }

    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut parsed = Self::parse_from_cursor(version, &mut cur)?;
        parsed.trailing = state_bytes[cur.position()..].to_vec();
        Ok(parsed)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write_to(&mut out);
        out
    }
}

/// Decoded `CSE_ALifeHumanStalker::STATE_Read` (`xrServer_Objects_ALife_Monsters.cpp:2172-2187`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeHumanStalkerState {
    pub human: AlifeHumanAbstractState,
    pub skeleton: ServerPhSkeletonState,
    pub trailing: Vec<u8>,
}

impl AlifeHumanStalkerState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let human = AlifeHumanAbstractState::parse_from_cursor(version, &mut cur)?;
        let skeleton = if version > 67 {
            ServerPhSkeletonState::parse_from_cursor(&mut cur)?
        } else {
            ServerPhSkeletonState::default()
        };
        if version > 90 && version < 111 {
            let _ = cur.read_u8()?;
        }
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            human,
            skeleton,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.human.write_to(&mut out);
        self.skeleton.write_to(&mut out);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeCreatureActor::STATE_Read` (`xrServer_Objects_ALife_Monsters.cpp:1474-1510`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeCreatureActorState {
    pub creature: AlifeCreatureAbstractState,
    pub trader: AlifeTraderAbstractState,
    pub skeleton: ServerPhSkeletonState,
    pub holder_id: u16,
    pub trailing: Vec<u8>,
}

impl AlifeCreatureActorState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let creature = AlifeCreatureAbstractState::parse_from_cursor(version, &mut cur)?;
        let trader = if version >= 21 {
            let t = AlifeTraderAbstractState::parse_from_cursor(version, &mut cur)?;
            if version < 32 {
                let _ = ServerVisualState::parse_from_cursor(version, &mut cur)?;
            }
            t
        } else {
            AlifeTraderAbstractState::default()
        };
        let skeleton = if version > 91 {
            ServerPhSkeletonState::parse_from_cursor(&mut cur)?
        } else {
            ServerPhSkeletonState::default()
        };
        let holder_id = if version > 88 {
            cur.read_u16()?
        } else {
            INVALID_OBJECT_ID
        };
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            creature,
            trader,
            skeleton,
            holder_id,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.creature.write_to(&mut out);
        self.trader.write_to(&mut out);
        self.skeleton.write_to(&mut out);
        out.extend_from_slice(&self.holder_id.to_le_bytes());
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// `CSE_ALifeObjectHangingLamp::flags` bitmask (`xrServer_Objects_ALife.h:463-472`).
pub mod hanging_lamp_flags {
    pub const PHYSIC: u16 = 1 << 0;
    pub const CAST_SHADOW: u16 = 1 << 1;
    pub const R1: u16 = 1 << 2;
    pub const R2: u16 = 1 << 3;
    pub const TYPE_SPOT: u16 = 1 << 4;
    pub const POINT_AMBIENT: u16 = 1 << 5;
    pub const VOLUMETRIC: u16 = 1 << 6;
}

/// Decoded `CSE_ALifeObjectHangingLamp::STATE_Read` (`xrServer_Objects_ALife.cpp:1352-1471`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeObjectHangingLampState {
    pub base: AlifeDynamicVisualState,
    pub skeleton: ServerPhSkeletonState,
    /// ARGB packed diffuse color (`0xAARRGGBB`).
    pub color: u32,
    pub brightness: f32,
    pub color_animator: String,
    pub range: f32,
    pub flags: u16,
    pub startup_animation: String,
    pub fixed_bones: String,
    pub health: f32,
    pub virtual_size: f32,
    pub ambient_radius: f32,
    pub ambient_power: f32,
    pub ambient_texture: String,
    pub light_texture: String,
    pub light_main_bone: String,
    /// Spot cone angle in radians (constructor default = `120 deg`).
    pub spot_cone_angle: f32,
    pub glow_texture: String,
    pub glow_radius: f32,
    pub light_ambient_bone: String,
    pub volumetric_quality: f32,
    pub volumetric_intensity: f32,
    pub volumetric_distance: f32,
    pub trailing: Vec<u8>,
}

impl Default for AlifeObjectHangingLampState {
    fn default() -> Self {
        Self {
            base: AlifeDynamicVisualState {
                alife: AlifeObjectState {
                    game_vertex_id: INVALID_GAME_VERTEX,
                    distance: 0.0,
                    direct_control: true,
                    level_vertex_id: INVALID_AI_VERTEX,
                    flags: 0,
                    custom_data: String::new(),
                    story_id: INVALID_STORY_ID,
                    spawn_story_id: INVALID_SPAWN_STORY_ID,
                    subclass_payload: Vec::new(),
                },
                visual: ServerVisualState::default(),
                trailing: Vec::new(),
            },
            skeleton: ServerPhSkeletonState::default(),
            color: 0xFFFF_FFFF,
            brightness: 1.0,
            color_animator: String::new(),
            range: 10.0,
            flags: hanging_lamp_flags::TYPE_SPOT | hanging_lamp_flags::R1 | hanging_lamp_flags::R2,
            startup_animation: String::new(),
            fixed_bones: String::new(),
            health: 100.0,
            virtual_size: 0.1,
            ambient_radius: 10.0,
            ambient_power: 0.1,
            ambient_texture: String::new(),
            light_texture: String::new(),
            light_main_bone: String::new(),
            spot_cone_angle: 120.0f32.to_radians(),
            glow_texture: String::new(),
            glow_radius: 0.7,
            light_ambient_bone: String::new(),
            volumetric_quality: 1.0,
            volumetric_intensity: 1.0,
            volumetric_distance: 1.0,
            trailing: Vec::new(),
        }
    }
}

impl AlifeObjectHangingLampState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let mut out = Self::default();

        if version > 20 {
            out.base = AlifeDynamicVisualState::parse_from_cursor(version, &mut cur)?;
        }
        if version >= 69 {
            out.skeleton = ServerPhSkeletonState::parse_from_cursor(&mut cur)?;
        }
        if version < 32 {
            out.base.visual = ServerVisualState::parse_from_cursor(version, &mut cur)?;
        }

        if version < 49 {
            out.color = cur.read_u32()?;
            out.color_animator = cur.read_zstring()?;
            let _ = cur.read_zstring()?;
            let _ = cur.read_zstring()?;
            out.range = cur.read_f32()?;
            let _ = cur.read_u8()?;
            if version > 10 {
                out.brightness = cur.read_f32()?;
            }
            if version > 11 {
                out.flags = cur.read_u16()?;
            }
            if version > 12 {
                let _ = cur.read_f32()?;
            }
            if version > 17 {
                out.startup_animation = cur.read_zstring()?;
            }
            if version > 42 {
                let _ = cur.read_zstring()?;
                let _ = cur.read_f32()?;
            }
            if version > 43 {
                out.fixed_bones = cur.read_zstring()?;
            }
            if version > 44 {
                out.health = cur.read_f32()?;
            }
        } else {
            out.color = cur.read_u32()?;
            out.brightness = cur.read_f32()?;
            out.color_animator = cur.read_zstring()?;
            out.range = cur.read_f32()?;
            out.flags = cur.read_u16()?;
            out.startup_animation = cur.read_zstring()?;
            out.fixed_bones = cur.read_zstring()?;
            out.health = cur.read_f32()?;
        }

        if version > 55 {
            out.virtual_size = cur.read_f32()?;
            out.ambient_radius = cur.read_f32()?;
            out.ambient_power = cur.read_f32()?;
            out.ambient_texture = cur.read_zstring()?;
            out.light_texture = cur.read_zstring()?;
            out.light_main_bone = cur.read_zstring()?;
            out.spot_cone_angle = cur.read_f32()?;
            out.glow_texture = cur.read_zstring()?;
            out.glow_radius = cur.read_f32()?;
        }

        out.light_ambient_bone = if version > 96 {
            cur.read_zstring()?
        } else {
            out.light_main_bone.clone()
        };

        if version > 118 {
            out.volumetric_quality = cur.read_f32()?;
            out.volumetric_intensity = cur.read_f32()?;
            out.volumetric_distance = cur.read_f32()?;
        }

        out.trailing = state_bytes[cur.position()..].to_vec();
        Ok(out)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.base.write_to(&mut out);
        self.skeleton.write_to(&mut out);
        out.extend_from_slice(&self.color.to_le_bytes());
        out.extend_from_slice(&self.brightness.to_le_bytes());
        out.extend_from_slice(self.color_animator.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.range.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(self.startup_animation.as_bytes());
        out.push(0);
        out.extend_from_slice(self.fixed_bones.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.health.to_le_bytes());
        out.extend_from_slice(&self.virtual_size.to_le_bytes());
        out.extend_from_slice(&self.ambient_radius.to_le_bytes());
        out.extend_from_slice(&self.ambient_power.to_le_bytes());
        out.extend_from_slice(self.ambient_texture.as_bytes());
        out.push(0);
        out.extend_from_slice(self.light_texture.as_bytes());
        out.push(0);
        out.extend_from_slice(self.light_main_bone.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.spot_cone_angle.to_le_bytes());
        out.extend_from_slice(self.glow_texture.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.glow_radius.to_le_bytes());
        out.extend_from_slice(self.light_ambient_bone.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.volumetric_quality.to_le_bytes());
        out.extend_from_slice(&self.volumetric_intensity.to_le_bytes());
        out.extend_from_slice(&self.volumetric_distance.to_le_bytes());
        out.extend_from_slice(&self.trailing);
        out
    }

    #[inline]
    pub fn is_physic(&self) -> bool {
        (self.flags & hanging_lamp_flags::PHYSIC) != 0
    }

    #[inline]
    pub fn casts_shadow(&self) -> bool {
        (self.flags & hanging_lamp_flags::CAST_SHADOW) != 0
    }

    #[inline]
    pub fn supports_r1(&self) -> bool {
        (self.flags & hanging_lamp_flags::R1) != 0
    }

    #[inline]
    pub fn supports_r2(&self) -> bool {
        (self.flags & hanging_lamp_flags::R2) != 0
    }

    #[inline]
    pub fn is_spot(&self) -> bool {
        (self.flags & hanging_lamp_flags::TYPE_SPOT) != 0
    }

    #[inline]
    pub fn has_point_ambient(&self) -> bool {
        (self.flags & hanging_lamp_flags::POINT_AMBIENT) != 0
    }

    #[inline]
    pub fn is_volumetric(&self) -> bool {
        (self.flags & hanging_lamp_flags::VOLUMETRIC) != 0
    }

    /// Evaluate linear RGBA `[r, g, b, 1.0]` matching `CHangingLamp::UpdateCL` (`HangingLamp.cpp:115-118, 295-303`).
    pub fn eval_color_rgba(
        &self,
        lanims: Option<&LightAnimLibrary>,
        t_sec: f32,
        smooth: bool,
    ) -> [f32; 4] {
        let base_rgb = if !self.color_animator.is_empty()
            && let Some(item) = lanims.and_then(|lib| lib.find_item_ci(&self.color_animator))
        {
            let c = item.sample_rgba_f32(t_sec, smooth);
            [c[0], c[1], c[2]]
        } else {
            let c = unpack_argb_f32(self.color);
            [c[0], c[1], c[2]]
        };
        [
            base_rgb[0] * self.brightness,
            base_rgb[1] * self.brightness,
            base_rgb[2] * self.brightness,
            1.0,
        ]
    }

    /// Convert primary lamp light into [`LevelLight`] using `SpawnPacket::{position, angle}` (`setXYZ(o_Angle)`).
    pub fn to_level_light(
        &self,
        position: [f32; 3],
        angles: [f32; 3],
        lanims: Option<&LightAnimLibrary>,
        t_sec: f32,
    ) -> LevelLight {
        let xf = matrix_set_xyz_translate(angles, position);
        let direction = matrix_forward_dir(&xf);
        let diffuse = self.eval_color_rgba(lanims, t_sec, true);
        let (light_type, phi, theta) = if self.is_spot() {
            (
                d3d_light_type::SPOT,
                self.spot_cone_angle,
                self.spot_cone_angle * 0.5,
            )
        } else {
            (d3d_light_type::POINT, 0.0, 0.0)
        };
        LevelLight {
            controller_id: u32::MAX,
            light_type,
            diffuse,
            specular: [0.0; 4],
            ambient: [0.0; 4],
            position,
            direction,
            range: self.range,
            falloff: 1.0,
            attenuation: [0.0; 3],
            theta,
            phi,
        }
    }

    /// Optional secondary point ambient fill light (`flPointAmbient`, `HangingLamp.cpp:143-154, 304-308`).
    pub fn to_ambient_level_light(
        &self,
        position: [f32; 3],
        lanims: Option<&LightAnimLibrary>,
        t_sec: f32,
    ) -> Option<LevelLight> {
        if !self.has_point_ambient() {
            return None;
        }
        let c = self.eval_color_rgba(lanims, t_sec, true);
        Some(LevelLight {
            controller_id: u32::MAX,
            light_type: d3d_light_type::POINT,
            diffuse: [
                c[0] * self.ambient_power,
                c[1] * self.ambient_power,
                c[2] * self.ambient_power,
                1.0,
            ],
            specular: [0.0; 4],
            ambient: [0.0; 4],
            position,
            direction: [0.0, -1.0, 0.0],
            range: self.ambient_radius,
            falloff: 1.0,
            attenuation: [0.0; 3],
            theta: 0.0,
            phi: 0.0,
        })
    }
}

/// Decoded `CSE_ALifeObjectBreakable::STATE_Read` (`xrServer_Objects_ALife.cpp:2028-2038`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeObjectBreakableState {
    pub base: AlifeDynamicVisualState,
    pub health: f32,
    pub trailing: Vec<u8>,
}

impl AlifeObjectBreakableState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let base = AlifeDynamicVisualState::parse_from_cursor(version, &mut cur)?;
        let health = cur.read_f32()?;
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            base,
            health,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.base.write_to(&mut out);
        out.extend_from_slice(&self.health.to_le_bytes());
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeObjectClimable::STATE_Read` (`xrServer_Objects_ALife.cpp:2088-2106`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeObjectClimableState {
    pub alife: AlifeObjectState,
    pub shapes: Vec<ServerShapeDef>,
    pub material: String,
    pub trailing: Vec<u8>,
}

impl AlifeObjectClimableState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let alife = if version >= 99 {
            AlifeObjectState::parse_prefix_from_cursor(version, &mut cur)?
        } else {
            AlifeObjectState {
                game_vertex_id: INVALID_GAME_VERTEX,
                distance: 0.0,
                direct_control: true,
                level_vertex_id: INVALID_AI_VERTEX,
                flags: 0,
                custom_data: String::new(),
                story_id: INVALID_STORY_ID,
                spawn_story_id: INVALID_SPAWN_STORY_ID,
                subclass_payload: Vec::new(),
            }
        };
        let shapes = ServerShapeDef::parse_cform_shapes(&mut cur)?;
        let material = if version > 126 {
            cur.read_zstring()?
        } else {
            "materials\\fake_ladders".to_string()
        };
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            alife,
            shapes,
            material,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.alife.write_prefix_to(&mut out);
        ServerShapeDef::encode_cform_shapes(&self.shapes, &mut out);
        out.extend_from_slice(self.material.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeHelicopter::STATE_Read` (`xrServer_Objects_ALife.cpp:1804-1823`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeHelicopterState {
    pub base: AlifeDynamicVisualState,
    pub motion: ServerMotionState,
    pub skeleton: ServerPhSkeletonState,
    pub startup_animation: String,
    pub engine_sound: String,
    pub trailing: Vec<u8>,
}

impl AlifeHelicopterState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let base = AlifeDynamicVisualState::parse_from_cursor(version, &mut cur)?;
        let motion = ServerMotionState::parse_from_cursor(&mut cur)?;
        let skeleton = if version >= 69 {
            ServerPhSkeletonState::parse_from_cursor(&mut cur)?
        } else {
            ServerPhSkeletonState::default()
        };
        let startup_animation = cur.read_zstring()?;
        let engine_sound = cur.read_zstring()?;
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            base,
            motion,
            skeleton,
            startup_animation,
            engine_sound,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.base.write_to(&mut out);
        self.motion.write_to(&mut out);
        self.skeleton.write_to(&mut out);
        out.extend_from_slice(self.startup_animation.as_bytes());
        out.push(0);
        out.extend_from_slice(self.engine_sound.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeCar::STATE_Read` (`xrServer_Objects_ALife.cpp:1880-1898`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeCarState {
    pub base: AlifeDynamicVisualState,
    pub skeleton: ServerPhSkeletonState,
    pub health: f32,
    pub trailing: Vec<u8>,
}

impl AlifeCarState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let base = AlifeDynamicVisualState::parse_from_cursor(version, &mut cur)?;
        let skeleton = if version > 65 {
            ServerPhSkeletonState::parse_from_cursor(&mut cur)?
        } else {
            ServerPhSkeletonState::default()
        };
        if version > 52 && version < 55 {
            let _ = cur.read_f32()?;
        }
        let mut health = if version > 92 { cur.read_f32()? } else { 1.0 };
        if health > 1.0 {
            health /= 100.0;
        }
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            base,
            skeleton,
            health,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.base.write_to(&mut out);
        self.skeleton.write_to(&mut out);
        out.extend_from_slice(&self.health.to_le_bytes());
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeStationaryMgun::STATE_Read` under `STATIONARYMGUN_NEW` (`xrServer_Objects_ALife.cpp:2281-2330`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeStationaryMgunState {
    pub base: AlifeDynamicVisualState,
    pub skeleton: ServerPhSkeletonState,
    pub ammo_type: u8,
    pub ammo_elapsed: u16,
    pub barrel_ammo_elapsed: Vec<u16>,
    pub trailing: Vec<u8>,
}

impl AlifeStationaryMgunState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let base = AlifeDynamicVisualState::parse_from_cursor(version, &mut cur)?;
        let skeleton = ServerPhSkeletonState::parse_from_cursor(&mut cur)?;
        let ammo_type = cur.read_u8()?;
        let ammo_elapsed = cur.read_u16()?;
        let barrel_count = usize::from(cur.read_u16()?);
        let mut barrel_ammo_elapsed = Vec::with_capacity(barrel_count.min(256));
        for _ in 0..barrel_count {
            barrel_ammo_elapsed.push(cur.read_u16()?);
        }
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            base,
            skeleton,
            ammo_type,
            ammo_elapsed,
            barrel_ammo_elapsed,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.base.write_to(&mut out);
        self.skeleton.write_to(&mut out);
        out.push(self.ammo_type);
        out.extend_from_slice(&self.ammo_elapsed.to_le_bytes());
        out.extend_from_slice(&(self.barrel_ammo_elapsed.len().min(u16::MAX as usize) as u16).to_le_bytes());
        for &elapsed in self.barrel_ammo_elapsed.iter().take(u16::MAX as usize) {
            out.extend_from_slice(&elapsed.to_le_bytes());
        }
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_ALifeOnlineOfflineGroup::STATE_Read` (`xrServer_Objects_ALife_Monsters.cpp:2269-2297`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeOnlineOfflineGroupState {
    pub alife: AlifeObjectState,
    pub member_ids: Vec<u16>,
    pub trailing: Vec<u8>,
}

impl AlifeOnlineOfflineGroupState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let alife = AlifeObjectState::parse_prefix_from_cursor(version, &mut cur)?;
        let count = cur.read_u32()? as usize;
        let mut member_ids = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            member_ids.push(cur.read_u16()?);
        }
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            alife,
            member_ids,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.alife.write_prefix_to(&mut out);
        out.extend_from_slice(&(self.member_ids.len() as u32).to_le_bytes());
        for &id in &self.member_ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Decoded `CSE_SmartCover::STATE_Read` (`xrServer_Objects_ALife_Smartcovers.cpp:134-163`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlifeSmartCoverState {
    pub alife: AlifeObjectState,
    pub shapes: Vec<ServerShapeDef>,
    pub description: String,
    pub hold_position_time: f32,
    pub enter_min_enemy_distance: f32,
    pub exit_min_enemy_distance: f32,
    pub is_combat_cover: bool,
    pub can_fire: bool,
    pub trailing: Vec<u8>,
}

impl AlifeSmartCoverState {
    pub fn parse(version: u16, state_bytes: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(state_bytes);
        let alife = AlifeObjectState::parse_prefix_from_cursor(version, &mut cur)?;
        let shapes = ServerShapeDef::parse_cform_shapes(&mut cur)?;
        let description = cur.read_zstring()?;
        let hold_position_time = cur.read_f32()?;
        let (enter_min_enemy_distance, exit_min_enemy_distance) = if version >= 120 {
            (cur.read_f32()?, cur.read_f32()?)
        } else {
            (0.0, 0.0)
        };
        let is_combat_cover = if version >= 122 {
            cur.read_u8()? != 0
        } else {
            false
        };
        let can_fire = if version >= 128 {
            cur.read_u8()? != 0
        } else {
            false
        };
        let trailing = state_bytes[cur.position()..].to_vec();
        Ok(Self {
            alife,
            shapes,
            description,
            hold_position_time,
            enter_min_enemy_distance,
            exit_min_enemy_distance,
            is_combat_cover,
            can_fire,
            trailing,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.alife.write_prefix_to(&mut out);
        ServerShapeDef::encode_cform_shapes(&self.shapes, &mut out);
        out.extend_from_slice(self.description.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.hold_position_time.to_le_bytes());
        out.extend_from_slice(&self.enter_min_enemy_distance.to_le_bytes());
        out.extend_from_slice(&self.exit_min_enemy_distance.to_le_bytes());
        out.push(u8::from(self.is_combat_cover));
        out.push(u8::from(self.can_fire));
        out.extend_from_slice(&self.trailing);
        out
    }
}

/// Wire-identical `CSE_ALifeMountedWeapon::STATE_Read` (`xrServer_Objects_ALife.cpp:2156`).
pub type AlifeMountedWeaponState = AlifeDynamicVisualState;
/// Wire-identical `CSE_ALifeItemTorch::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:451`).
pub type AlifeItemTorchState = AlifeInventoryItemState;
/// Wire-identical `CSE_ALifeItemDetector::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:920`).
pub type AlifeItemDetectorState = AlifeInventoryItemState;
/// Wire-identical `CSE_ALifeItemArtefact::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:960`).
pub type AlifeItemArtefactState = AlifeInventoryItemState;
/// Wire-identical `CSE_ALifeItemCustomOutfit::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:1275`).
pub type AlifeItemCustomOutfitState = AlifeInventoryItemState;
/// Wire-identical `CSE_ALifeItemHelmet::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:1320`).
pub type AlifeItemHelmetState = AlifeInventoryItemState;
/// Wire-identical `CSE_ALifeItemGrenade::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:1136`).
pub type AlifeItemGrenadeState = AlifeInventoryItemState;
/// Wire-identical `CSE_ALifeItemExplosive::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:1174`).
pub type AlifeItemExplosiveState = AlifeInventoryItemState;
/// Wire-identical `CSE_ALifeItemBolt::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:1226`).
pub type AlifeItemBoltState = AlifeInventoryItemState;
/// Wire-identical `CSE_ALifeItemWeaponShotGun::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:709`).
pub type AlifeItemWeaponShotgunState = AlifeItemWeaponState;
/// Wire-identical `CSE_ALifeItemWeaponAutoShotGun::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:746`).
pub type AlifeItemWeaponAutoShotgunState = AlifeItemWeaponState;
/// Wire-identical `CSE_ALifeItemWeaponMagazined::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:787`).
pub type AlifeItemWeaponMagazinedState = AlifeItemWeaponState;
/// Wire-identical `CSE_ALifeItemWeaponMagazinedWGL::STATE_Read` (`xrServer_Objects_ALife_Items.cpp:829`).
pub type AlifeItemWeaponMagazinedWglState = AlifeItemWeaponState;

impl SpawnPacket {
    pub fn decode_dynamic_visual_state(&self) -> Result<AlifeDynamicVisualState, FormatError> {
        AlifeDynamicVisualState::parse(self.version, &self.state_payload)
    }

    pub fn decode_space_restrictor_state(&self) -> Result<AlifeSpaceRestrictorState, FormatError> {
        AlifeSpaceRestrictorState::parse(self.version, &self.state_payload)
    }

    pub fn decode_level_changer_state(&self) -> Result<AlifeLevelChangerState, FormatError> {
        AlifeLevelChangerState::parse(self.version, &self.state_payload)
    }

    pub fn decode_inventory_item_state(&self) -> Result<AlifeInventoryItemState, FormatError> {
        AlifeInventoryItemState::parse(self.version, &self.state_payload)
    }

    pub fn decode_weapon_state(&self) -> Result<AlifeItemWeaponState, FormatError> {
        AlifeItemWeaponState::parse(self.version, &self.state_payload)
    }

    pub fn decode_custom_zone_state(&self) -> Result<AlifeCustomZoneState, FormatError> {
        AlifeCustomZoneState::parse(self.version, &self.state_payload)
    }

    pub fn decode_anomalous_zone_state(&self) -> Result<AlifeAnomalousZoneState, FormatError> {
        AlifeAnomalousZoneState::parse(self.version, &self.state_payload)
    }

    pub fn decode_torrid_zone_state(&self) -> Result<AlifeTorridZoneState, FormatError> {
        AlifeTorridZoneState::parse(self.version, &self.state_payload)
    }

    pub fn decode_zone_visual_state(&self) -> Result<AlifeZoneVisualState, FormatError> {
        AlifeZoneVisualState::parse(self.version, &self.state_payload)
    }

    pub fn decode_team_base_zone_state(&self) -> Result<AlifeTeamBaseZoneState, FormatError> {
        AlifeTeamBaseZoneState::parse(self.version, &self.state_payload)
    }

    pub fn decode_object_physic_state(&self) -> Result<AlifeObjectPhysicState, FormatError> {
        AlifeObjectPhysicState::parse(self.version, &self.state_payload)
    }

    pub fn decode_inventory_box_state(&self) -> Result<AlifeInventoryBoxState, FormatError> {
        AlifeInventoryBoxState::parse(self.version, &self.state_payload)
    }

    pub fn decode_ammo_state(&self) -> Result<AlifeItemAmmoState, FormatError> {
        AlifeItemAmmoState::parse(self.version, &self.state_payload)
    }

    pub fn decode_pda_state(&self) -> Result<AlifeItemPdaState, FormatError> {
        AlifeItemPdaState::parse(self.version, &self.state_payload)
    }

    pub fn decode_document_state(&self) -> Result<AlifeItemDocumentState, FormatError> {
        AlifeItemDocumentState::parse(self.version, &self.state_payload)
    }

    pub fn decode_creature_abstract_state(
        &self,
    ) -> Result<AlifeCreatureAbstractState, FormatError> {
        AlifeCreatureAbstractState::parse(self.version, &self.state_payload)
    }

    pub fn decode_monster_abstract_state(&self) -> Result<AlifeMonsterAbstractState, FormatError> {
        AlifeMonsterAbstractState::parse(self.version, &self.state_payload)
    }

    pub fn decode_monster_base_state(&self) -> Result<AlifeMonsterBaseState, FormatError> {
        AlifeMonsterBaseState::parse(self.version, &self.state_payload)
    }

    pub fn decode_trader_state(&self) -> Result<AlifeTraderState, FormatError> {
        AlifeTraderState::parse(self.version, &self.state_payload)
    }

    pub fn decode_human_abstract_state(&self) -> Result<AlifeHumanAbstractState, FormatError> {
        AlifeHumanAbstractState::parse(self.version, &self.state_payload)
    }

    pub fn decode_human_stalker_state(&self) -> Result<AlifeHumanStalkerState, FormatError> {
        AlifeHumanStalkerState::parse(self.version, &self.state_payload)
    }

    pub fn decode_creature_actor_state(&self) -> Result<AlifeCreatureActorState, FormatError> {
        AlifeCreatureActorState::parse(self.version, &self.state_payload)
    }

    pub fn decode_hanging_lamp_state(&self) -> Result<AlifeObjectHangingLampState, FormatError> {
        AlifeObjectHangingLampState::parse(self.version, &self.state_payload)
    }

    pub fn decode_breakable_state(&self) -> Result<AlifeObjectBreakableState, FormatError> {
        AlifeObjectBreakableState::parse(self.version, &self.state_payload)
    }

    pub fn decode_climable_state(&self) -> Result<AlifeObjectClimableState, FormatError> {
        AlifeObjectClimableState::parse(self.version, &self.state_payload)
    }

    pub fn decode_helicopter_state(&self) -> Result<AlifeHelicopterState, FormatError> {
        AlifeHelicopterState::parse(self.version, &self.state_payload)
    }

    pub fn decode_car_state(&self) -> Result<AlifeCarState, FormatError> {
        AlifeCarState::parse(self.version, &self.state_payload)
    }

    pub fn decode_stationary_mgun_state(
        &self,
    ) -> Result<AlifeStationaryMgunState, FormatError> {
        AlifeStationaryMgunState::parse(self.version, &self.state_payload)
    }

    pub fn decode_online_offline_group_state(
        &self,
    ) -> Result<AlifeOnlineOfflineGroupState, FormatError> {
        AlifeOnlineOfflineGroupState::parse(self.version, &self.state_payload)
    }

    pub fn decode_smart_cover_state(&self) -> Result<AlifeSmartCoverState, FormatError> {
        AlifeSmartCoverState::parse(self.version, &self.state_payload)
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

    #[test]
    fn test_alife_subclass_packet_codecs() {
        let base_alife = AlifeObjectState {
            game_vertex_id: 12,
            distance: 4.5,
            direct_control: true,
            level_vertex_id: 505,
            flags: alife_object_flags::CAN_SAVE | alife_object_flags::SWITCH_ONLINE,
            custom_data: "[logic]\nactive = nil\n".to_string(),
            story_id: 9001,
            spawn_story_id: INVALID_SPAWN_STORY_ID,
            subclass_payload: Vec::new(),
        };

        // Level changer (inherits SpaceRestrictor w/ Sphere + 4x3 Box shape)
        let lc = AlifeLevelChangerState {
            restrictor: AlifeSpaceRestrictorState {
                alife: base_alife.clone(),
                shapes: vec![
                    ServerShapeDef::Sphere {
                        center: [0.5, 1.0, -0.5],
                        radius: 2.5,
                    },
                    ServerShapeDef::Box([
                        1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0,
                    ]),
                ],
                restrictor_type: 2,
                trailing: Vec::new(),
            },
            next_graph_vertex: 88,
            next_level_vertex: 7777,
            next_position: [12.0, 1.0, -4.0],
            next_angles: [0.0, 1.57, 0.0],
            level_to_change: "l02_garbage".to_string(),
            level_point_to_change: "gar_start".to_string(),
            silent_mode: true,
            trailing: Vec::new(),
        };
        let lc_decoded = AlifeLevelChangerState::parse(SPAWN_VERSION, &lc.to_bytes()).unwrap();
        assert_eq!(lc_decoded, lc);

        // Inventory weapon (inherits AlifeDynamicVisualState + InventoryItem + Weapon)
        let wpn = AlifeItemWeaponState {
            item: AlifeInventoryItemState {
                base: AlifeDynamicVisualState {
                    alife: base_alife,
                    visual: ServerVisualState {
                        visual_name: "dynamics\\weapons\\wpn_ak74\\wpn_ak74".to_string(),
                        flags: 1,
                    },
                    trailing: Vec::new(),
                },
                condition: 0.85,
                upgrades: vec!["up_sect_firsta_ak74".to_string(), "up_sect_secona_ak74".to_string()],
                trailing: Vec::new(),
            },
            ammo_current: 90,
            ammo_elapsed: 30,
            wpn_state: 0,
            addon_flags: 0x03,
            ammo_type: 1,
            ammo_elapsed_grenades: 1,
            trailing: Vec::new(),
        };
        let wpn_decoded = AlifeItemWeaponState::parse(SPAWN_VERSION, &wpn.to_bytes()).unwrap();
        assert_eq!(wpn_decoded, wpn);

        // TraderAbstract section
        let trader = AlifeTraderAbstractState {
            money: 15000,
            specific_character: "esc_2_12_stalker_wolf".to_string(),
            trader_flags: 0x04,
            character_profile: "esc_2_12_stalker_wolf".to_string(),
            community_index: 2,
            rank: 450,
            reputation: 120,
            character_name: "Wolf".to_string(),
            deadbody_can_take: true,
            deadbody_closed: false,
        };
        let mut tbytes = Vec::new();
        trader.write_to(&mut tbytes);
        let mut tcur = ByteCursor::new(&tbytes);
        let trader_decoded = AlifeTraderAbstractState::parse_from_cursor(SPAWN_VERSION, &mut tcur).unwrap();
        assert_eq!(trader_decoded, trader);
        assert!(tcur.is_eof());
    }

    #[test]
    fn test_deeper_alife_zone_skeleton_stalker_and_actor_codecs() {
        let base_alife = AlifeObjectState {
            game_vertex_id: 25,
            distance: 2.0,
            direct_control: true,
            level_vertex_id: 9001,
            flags: alife_object_flags::CAN_SAVE | alife_object_flags::SWITCH_ONLINE,
            custom_data: "[smart_terrains]\nesc_smart = true\n".to_string(),
            story_id: INVALID_STORY_ID,
            spawn_story_id: INVALID_SPAWN_STORY_ID,
            subclass_payload: Vec::new(),
        };

        // Quantized bone state + CSE_PHSkeleton
        let bone = ServerPhBoneState::from_world_position_quat(
            [1.5, -2.0, 3.0],
            [0.0, std::f32::consts::FRAC_1_SQRT_2, 0.0, std::f32::consts::FRAC_1_SQRT_2],
            [-10.0, -10.0, -10.0],
            [10.0, 10.0, 10.0],
            true,
        );
        let decoded_pos = bone.decode_position([-10.0; 3], [10.0; 3]);
        assert!((decoded_pos[0] - 1.5).abs() < 0.1);
        assert!((decoded_pos[1] + 2.0).abs() < 0.1);
        assert!((decoded_pos[2] - 3.0).abs() < 0.1);

        let skel = ServerPhSkeletonState {
            startup_animation: "idle".to_string(),
            flags: ph_skeleton_flags::ACTIVE | ph_skeleton_flags::SAVED_DATA,
            source_id: 42,
            saved_bones: Some(ServerPhBonesData {
                bones_mask: 0xFFFF_FFFF_FFFF_FFFF,
                root_bone: 0,
                min: [-10.0; 3],
                max: [10.0; 3],
                bones: vec![bone],
            }),
        };

        // Anomalous zone
        let anom = AlifeAnomalousZoneState {
            zone: AlifeCustomZoneState {
                restrictor: AlifeSpaceRestrictorState {
                    alife: base_alife.clone(),
                    shapes: vec![ServerShapeDef::Sphere {
                        center: [0.0; 3],
                        radius: 4.0,
                    }],
                    restrictor_type: 1,
                    trailing: Vec::new(),
                },
                max_power: 1.5,
                owner_id: u32::MAX,
                enabled_time: 10,
                disabled_time: 5,
                start_time_shift: 2,
                trailing: Vec::new(),
            },
            offline_interactive_radius: 30.0,
            artefact_spawn_count: 16,
            artefact_position_offset: 64,
            trailing: Vec::new(),
        };
        assert_eq!(
            AlifeAnomalousZoneState::parse(SPAWN_VERSION, &anom.to_bytes()).unwrap(),
            anom
        );

        // Physic object w/ saved ragdoll bones
        let phys = AlifeObjectPhysicState {
            base: AlifeDynamicVisualState {
                alife: base_alife.clone(),
                visual: ServerVisualState {
                    visual_name: "physics\\box\\box_metall_01".to_string(),
                    flags: 0,
                },
                trailing: Vec::new(),
            },
            skeleton: skel.clone(),
            physic_type: 3,
            mass: 25.0,
            fixed_bones: "link".to_string(),
            trailing: Vec::new(),
        };
        assert_eq!(
            AlifeObjectPhysicState::parse(SPAWN_VERSION, &phys.to_bytes()).unwrap(),
            phys
        );

        // Human stalker (TraderAbstract BEFORE MonsterAbstract + HumanBrain + PHSkeleton)
        let stalker = AlifeHumanStalkerState {
            human: AlifeHumanAbstractState {
                trader: AlifeTraderAbstractState {
                    money: 5000,
                    specific_character: "esc_2_12_stalker_fanat".to_string(),
                    trader_flags: 1,
                    character_profile: "esc_2_12_stalker_fanat".to_string(),
                    community_index: 3,
                    rank: 600,
                    reputation: 250,
                    character_name: "Fanatic".to_string(),
                    deadbody_can_take: true,
                    deadbody_closed: false,
                },
                monster: AlifeMonsterAbstractState {
                    creature: AlifeCreatureAbstractState {
                        base: AlifeDynamicVisualState {
                            alife: base_alife.clone(),
                            visual: ServerVisualState {
                                visual_name: "actors\\stalker_neutral\\stalker_neutral_2"
                                    .to_string(),
                                flags: 0,
                            },
                            trailing: Vec::new(),
                        },
                        team: 0,
                        squad: 1,
                        group: 2,
                        health: 1.0,
                        dynamic_out_restrictions: vec![10, 11],
                        dynamic_in_restrictions: vec![12],
                        killer_id: INVALID_OBJECT_ID,
                        game_death_time: 0,
                        trailing: Vec::new(),
                    },
                    out_space_restrictors: String::new(),
                    in_space_restrictors: "esc_base_restrictor".to_string(),
                    smart_terrain_id: 17,
                    task_reached: true,
                    trailing: Vec::new(),
                },
                equipment_prefs: vec![1, 2, 0, 1, 2],
                main_weapon_prefs: vec![2, 1, 0, 1],
                trailing: Vec::new(),
            },
            skeleton: skel.clone(),
            trailing: Vec::new(),
        };
        assert_eq!(
            AlifeHumanStalkerState::parse(SPAWN_VERSION, &stalker.to_bytes()).unwrap(),
            stalker
        );

        // CreatureActor (CreatureAbstract -> TraderAbstract -> PHSkeleton -> holder_id)
        let actor = AlifeCreatureActorState {
            creature: stalker.human.monster.creature.clone(),
            trader: stalker.human.trader.clone(),
            skeleton: skel,
            holder_id: INVALID_OBJECT_ID,
            trailing: Vec::new(),
        };
        assert_eq!(
            AlifeCreatureActorState::parse(SPAWN_VERSION, &actor.to_bytes()).unwrap(),
            actor
        );
    }

    #[test]
    fn test_hanging_lamp_vehicles_mgun_squad_smartcover_and_items_codecs() {
        use crate::lanims::{pack_argb, LightAnimItem, LightAnimLibrary, LANIM_VERSION};

        let base_alife = AlifeObjectState {
            game_vertex_id: 100,
            distance: 1.5,
            direct_control: true,
            level_vertex_id: 2048,
            flags: alife_object_flags::CAN_SAVE | alife_object_flags::VISIBLE_FOR_MAP,
            custom_data: "[logic]\ncfg = scripts\\lamp.ltx\n".to_string(),
            story_id: INVALID_STORY_ID,
            spawn_story_id: 404,
            subclass_payload: Vec::new(),
        };
        let dyn_vis = AlifeDynamicVisualState {
            alife: base_alife.clone(),
            visual: ServerVisualState {
                visual_name: "dynamics\\light\\light_galogen_1_glass".to_string(),
                flags: 1,
            },
            trailing: Vec::new(),
        };
        let skel = ServerPhSkeletonState {
            startup_animation: "idle".to_string(),
            flags: ph_skeleton_flags::ACTIVE,
            source_id: INVALID_OBJECT_ID,
            saved_bones: None,
        };

        // 1. HangingLamp (spot + ambient + lanim)
        let lamp = AlifeObjectHangingLampState {
            base: dyn_vis.clone(),
            skeleton: skel.clone(),
            color: pack_argb(255, 255, 200, 100),
            brightness: 2.0,
            color_animator: "lights\\flicker".to_string(),
            range: 14.0,
            flags: hanging_lamp_flags::CAST_SHADOW
                | hanging_lamp_flags::R2
                | hanging_lamp_flags::TYPE_SPOT
                | hanging_lamp_flags::POINT_AMBIENT
                | hanging_lamp_flags::VOLUMETRIC,
            startup_animation: "idle".to_string(),
            fixed_bones: "link".to_string(),
            health: 100.0,
            virtual_size: 0.25,
            ambient_radius: 6.0,
            ambient_power: 0.2,
            ambient_texture: "internal\\ambient".to_string(),
            light_texture: "internal\\spot_cone".to_string(),
            light_main_bone: "bone_lamp".to_string(),
            spot_cone_angle: 90.0f32.to_radians(),
            glow_texture: "glow\\glow_01".to_string(),
            glow_radius: 0.8,
            light_ambient_bone: "bone_lamp".to_string(),
            volumetric_quality: 0.8,
            volumetric_intensity: 0.4,
            volumetric_distance: 0.9,
            trailing: Vec::new(),
        };
        let lamp_rt = AlifeObjectHangingLampState::parse(SPAWN_VERSION, &lamp.to_bytes()).unwrap();
        assert_eq!(lamp_rt, lamp);
        assert!(lamp_rt.is_spot() && lamp_rt.has_point_ambient() && lamp_rt.casts_shadow());

        let mut anim = LightAnimItem::new("lights\\flicker");
        anim.fps = 10.0;
        anim.resize(2);
        anim.insert_key(0, pack_argb(255, 0, 255, 0)).unwrap();
        let lib = LightAnimLibrary {
            version: LANIM_VERSION,
            items: vec![anim],
        };
        // setXYZ([PI/2, 0, 0]) pitches +90 deg so forward k points +Y!
        let spot_ll = lamp.to_level_light([1.0, 3.0, 2.0], [std::f32::consts::FRAC_PI_2, 0.0, 0.0], Some(&lib), 0.0);
        assert!(spot_ll.is_spot());
        assert!((spot_ll.direction[1] - 1.0).abs() < 1e-5);
        assert!((spot_ll.diffuse[1] - 2.0).abs() < 1e-5 && spot_ll.diffuse[0].abs() < 1e-5);
        let amb_ll = lamp.to_ambient_level_light([1.0, 3.0, 2.0], Some(&lib), 0.0).unwrap();
        assert!(amb_ll.is_point());
        assert!((amb_ll.diffuse[1] - 0.4).abs() < 1e-5);

        // 2. Breakable, Climable, Helicopter, Car, StationaryMgun, Squad, SmartCover
        let brk = AlifeObjectBreakableState {
            base: dyn_vis.clone(),
            health: 75.0,
            trailing: Vec::new(),
        };
        assert_eq!(AlifeObjectBreakableState::parse(SPAWN_VERSION, &brk.to_bytes()).unwrap(), brk);

        let climb = AlifeObjectClimableState {
            alife: base_alife.clone(),
            shapes: vec![ServerShapeDef::Box([
                1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.5, 0.0,
            ])],
            material: "materials\\fake_ladders".to_string(),
            trailing: Vec::new(),
        };
        assert_eq!(AlifeObjectClimableState::parse(SPAWN_VERSION, &climb.to_bytes()).unwrap(), climb);

        let heli = AlifeHelicopterState {
            base: dyn_vis.clone(),
            motion: ServerMotionState {
                motion_name: "helicopter\\patrol.anm".to_string(),
            },
            skeleton: skel.clone(),
            startup_animation: "idle".to_string(),
            engine_sound: "vehicles\\mi24_engine".to_string(),
            trailing: Vec::new(),
        };
        assert_eq!(AlifeHelicopterState::parse(SPAWN_VERSION, &heli.to_bytes()).unwrap(), heli);

        let mut raw_car = Vec::new();
        dyn_vis.write_to(&mut raw_car);
        skel.write_to(&mut raw_car);
        raw_car.extend_from_slice(&80.0f32.to_le_bytes()); // legacy > 1.0 health normalized -> 0.8
        let car = AlifeCarState::parse(SPAWN_VERSION, &raw_car).unwrap();
        assert!((car.health - 0.8).abs() < 1e-6);
        assert_eq!(AlifeCarState::parse(SPAWN_VERSION, &car.to_bytes()).unwrap(), car);

        let mgun = AlifeStationaryMgunState {
            base: dyn_vis,
            skeleton: skel,
            ammo_type: 1,
            ammo_elapsed: 200,
            barrel_ammo_elapsed: vec![100, 100],
            trailing: Vec::new(),
        };
        assert_eq!(
            AlifeStationaryMgunState::parse(SPAWN_VERSION, &mgun.to_bytes()).unwrap(),
            mgun
        );

        let squad = AlifeOnlineOfflineGroupState {
            alife: base_alife.clone(),
            member_ids: vec![10, 11, 12],
            trailing: Vec::new(),
        };
        assert_eq!(
            AlifeOnlineOfflineGroupState::parse(SPAWN_VERSION, &squad.to_bytes()).unwrap(),
            squad
        );

        let cover = AlifeSmartCoverState {
            alife: base_alife,
            shapes: vec![ServerShapeDef::Sphere {
                center: [0.0; 3],
                radius: 1.0,
            }],
            description: "combat_front".to_string(),
            hold_position_time: 12.5,
            enter_min_enemy_distance: 4.0,
            exit_min_enemy_distance: 2.0,
            is_combat_cover: true,
            can_fire: true,
            trailing: Vec::new(),
        };
        assert_eq!(
            AlifeSmartCoverState::parse(SPAWN_VERSION, &cover.to_bytes()).unwrap(),
            cover
        );

        // 3. Verify SpawnPacket helper dispatch
        let pkt = SpawnPacket {
            section: "smart_cover".to_string(),
            name_replace: "esc_cover_01".to_string(),
            game_id_legacy: 0,
            rp: 0xFE,
            position: [0.0; 3],
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
            spawn_id: 7,
            state_payload: cover.to_bytes(),
            trailing_update_payload: Vec::new(),
        };
        assert_eq!(pkt.decode_smart_cover_state().unwrap(), cover);

        // Ensure leaf item type aliases remain wired to InventoryItem/Weapon wire layouts
        let torch: AlifeItemTorchState = AlifeItemTorchState::parse(SPAWN_VERSION, &AlifeInventoryItemState {
            base: lamp.base.clone(),
            condition: 1.0,
            upgrades: Vec::new(),
            trailing: Vec::new(),
        }.to_bytes()).unwrap();
        assert_eq!(torch.condition, 1.0);
    }
}
