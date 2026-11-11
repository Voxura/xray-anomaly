//! Pure Rust X-Ray `level.ai`, `level.gct`, and `game.graph` Navigation Graph Parsers & Pathfinder
//!
//! Covers:
//! - `_compressed_normal.cpp`: `pv_compress_normal` / `pv_decompress_normal` 16-bit octahedral/projective normals
//! - `level.ai` (`XRAI_CURRENT_VERSION = 10` 23B `NodeCompressed` with 23-bit links;
//!   `XRAI_LARGE_VERSION = 13` 25B `NodeCompressed13` with 26-bit links) + A* pathfinder
//! - `level.gct` cross-table mapping level AI vertices to global `GameGraph` vertices
//! - `game.graph` global inter-level graph (levels, `CVertex`, `CEdge`, `CLevelPoint`, embedded cross-tables) + A* pathfinder

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::ogf::OgfBBox;

pub const XRAI_CURRENT_VERSION: u32 = 10;
pub const XRAI_LARGE_VERSION: u32 = 13;

pub const INVALID_AI_LINK_V10: u32 = 0x007F_FFFF;
pub const INVALID_AI_LINK_V13: u32 = 0x03FF_FFFF;
pub const INVALID_AI_VERTEX: u32 = u32::MAX;
pub const INVALID_GAME_VERTEX: u16 = u16::MAX;

const EPS_L: f32 = 0.001;

/// Decompress X-Ray's 16-bit packed normal (`pvDecompress` in `src/xrCore/_compressed_normal.cpp`).
pub fn pv_decompress_normal(packed: u16) -> [f32; 3] {
    const SIGN_X: u16 = 0x8000;
    const SIGN_Y: u16 = 0x4000;
    const SIGN_Z: u16 = 0x2000;

    let mut xbits = ((packed & 0x1F80) >> 7) as i32;
    let mut ybits = (packed & 0x007F) as i32;

    if xbits + ybits >= 127 {
        xbits = 127 - xbits;
        ybits = 127 - ybits;
    }

    let zbits = 126 - xbits - ybits;
    let x = xbits as f32;
    let y = ybits as f32;
    let z = zbits as f32;
    let len = (x * x + y * y + z * z).sqrt();
    let inv = if len > 1e-12 { 1.0 / len } else { 0.0 };

    let mut nx = x * inv;
    let mut ny = y * inv;
    let mut nz = z * inv;

    if (packed & SIGN_X) != 0 {
        nx = -nx;
    }
    if (packed & SIGN_Y) != 0 {
        ny = -ny;
    }
    if (packed & SIGN_Z) != 0 {
        nz = -nz;
    }
    [nx, ny, nz]
}

/// Compress a 3D normal into X-Ray's 16-bit packed normal (`pvCompress` in `_compressed_normal.cpp`).
pub fn pv_compress_normal(vec: [f32; 3]) -> u16 {
    let mut m_vec: u16 = 0;
    let mut tx = vec[0];
    let mut ty = vec[1];
    let mut tz = vec[2];

    if tx < 0.0 {
        m_vec |= 0x8000;
        tx = -tx;
    }
    if ty < 0.0 {
        m_vec |= 0x4000;
        ty = -ty;
    }
    if tz < 0.0 {
        m_vec |= 0x2000;
        tz = -tz;
    }

    let sum = tx + ty + tz;
    if sum <= 1e-12 {
        return m_vec;
    }
    let w = 126.0 / sum;
    let mut xbits = (tx * w).floor() as i32;
    let mut ybits = (ty * w).floor() as i32;
    xbits = xbits.clamp(0, 126);
    ybits = ybits.clamp(0, 126);

    if xbits >= 64 {
        xbits = 127 - xbits;
        ybits = 127 - ybits;
    }

    m_vec |= ((xbits as u16) & 0x3F) << 7;
    m_vec |= (ybits as u16) & 0x7F;
    m_vec
}

/// 56-byte `level.ai` header (`hdrNODES` / `CLevelGraph::CHeader`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AiGraphHeader {
    pub version: u32,
    pub vertex_count: u32,
    pub cell_size: f32,
    pub factor_y: f32,
    pub box_aabb: OgfBBox,
    pub guid: [u64; 2],
    pub row_length: u32,
    pub column_length: u32,
}

/// Unpacked AI navigation vertex (`NodeCompressed` v10 or `NodeCompressed13` v13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AiVertex {
    /// Neighbor vertex IDs in 4 cardinal directions (`u32::MAX` = no link).
    pub links: [u32; 4],
    pub cover_high: u16,
    pub cover_low: u16,
    pub plane: u16,
    pub xz: u32,
    pub y: u16,
}

impl AiVertex {
    /// Return optional neighbor index for direction `0..4`.
    #[inline]
    pub fn link(&self, dir: usize) -> Option<u32> {
        self.links
            .get(dir)
            .copied()
            .filter(|&id| id != INVALID_AI_VERTEX)
    }

    /// Return raw 4-bit `(high, low)` cover nibble `[0..15]` for direction `0..4`.
    #[inline]
    pub fn cover(&self, dir: usize) -> (u8, u8) {
        let shift = (dir & 3) * 4;
        (
            ((self.cover_high >> shift) & 0x0F) as u8,
            ((self.cover_low >> shift) & 0x0F) as u8,
        )
    }

    /// Unpack `(grid_x, grid_z)` from quantized `xz`.
    #[inline]
    pub fn grid_xz(&self, header: &AiGraphHeader) -> (u32, u32) {
        let row = header.row_length.max(1);
        (self.xz / row, self.xz % row)
    }

    /// Decode world-space 3D center position of this AI node.
    #[inline]
    pub fn world_position(&self, header: &AiGraphHeader) -> [f32; 3] {
        let (gx, gz) = self.grid_xz(header);
        let x = gx as f32 * header.cell_size + header.box_aabb.min[0];
        let y = (f32::from(self.y) / 65535.0) * header.factor_y + header.box_aabb.min[1];
        let z = gz as f32 * header.cell_size + header.box_aabb.min[2];
        [x, y, z]
    }
}

/// Parsed `level.ai` single-level AI navigation mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelGraph {
    pub header: AiGraphHeader,
    pub vertices: Vec<AiVertex>,
}

fn read_u32_le_unaligned(bytes: &[u8], byte_offset: usize) -> u32 {
    let mut buf = [0u8; 4];
    for (i, slot) in buf.iter_mut().enumerate() {
        if let Some(&b) = bytes.get(byte_offset + i) {
            *slot = b;
        }
    }
    u32::from_le_bytes(buf)
}

impl LevelGraph {
    /// Parse a binary `level.ai` buffer (versions `10` and `13`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let version = cur.read_u32()?;
        if version != XRAI_CURRENT_VERSION && version != XRAI_LARGE_VERSION {
            return Err(FormatError::InvalidValue(
                "unsupported level.ai version (expected 10 or 13)",
            ));
        }

        let vertex_count = cur.read_u32()?;
        let cell_size = cur.read_f32()?;
        let factor_y = cur.read_f32()?;
        if cell_size <= 0.0 || !cell_size.is_finite() {
            return Err(FormatError::InvalidValue("invalid level.ai cell_size"));
        }

        let box_aabb = OgfBBox {
            min: cur.read_f32_3()?,
            max: cur.read_f32_3()?,
        };
        let guid = [cur.read_u64()?, cur.read_u64()?];

        let row_length = (((box_aabb.max[2] - box_aabb.min[2]) / cell_size) + EPS_L + 1.5)
            .floor()
            .max(1.0) as u32;
        let column_length = (((box_aabb.max[0] - box_aabb.min[0]) / cell_size) + EPS_L + 1.5)
            .floor()
            .max(1.0) as u32;

        let header = AiGraphHeader {
            version,
            vertex_count,
            cell_size,
            factor_y,
            box_aabb,
            guid,
            row_length,
            column_length,
        };

        let mut vertices = Vec::with_capacity(vertex_count as usize);
        if version == XRAI_CURRENT_VERSION {
            let need = (vertex_count as usize)
                .checked_mul(23)
                .ok_or(FormatError::UnexpectedEof)?;
            let raw = cur.read_bytes(need)?;
            for rec in raw.as_chunks::<23>().0 {
                let d = &rec[0..12];
                let links = [
                    normalize_link(read_u32_le_unaligned(d, 0) & 0x007F_FFFF, INVALID_AI_LINK_V10),
                    normalize_link(
                        (read_u32_le_unaligned(d, 2) >> 7) & 0x007F_FFFF,
                        INVALID_AI_LINK_V10,
                    ),
                    normalize_link(
                        (read_u32_le_unaligned(d, 5) >> 6) & 0x007F_FFFF,
                        INVALID_AI_LINK_V10,
                    ),
                    normalize_link(
                        (read_u32_le_unaligned(d, 8) >> 5) & 0x007F_FFFF,
                        INVALID_AI_LINK_V10,
                    ),
                ];
                let cover_high = u16::from_le_bytes([rec[12], rec[13]]);
                let cover_low = u16::from_le_bytes([rec[14], rec[15]]);
                let plane = u16::from_le_bytes([rec[16], rec[17]]);
                // 5-byte NodePosition: xz[0..3] (u24 LE), y[3..5] (u16 LE)
                let xz = u32::from_le_bytes([rec[18], rec[19], rec[20], 0]);
                let y = u16::from_le_bytes([rec[21], rec[22]]);

                vertices.push(AiVertex {
                    links,
                    cover_high,
                    cover_low,
                    plane,
                    xz,
                    y,
                });
            }
        } else {
            let need = (vertex_count as usize)
                .checked_mul(25)
                .ok_or(FormatError::UnexpectedEof)?;
            let raw = cur.read_bytes(need)?;
            for rec in raw.as_chunks::<25>().0 {
                let d = &rec[0..13];
                let mut links = [INVALID_AI_VERTEX; 4];
                for (i, slot) in links.iter_mut().enumerate() {
                    let bit = i * 26;
                    let byte_off = bit / 8;
                    let shift = bit % 8;
                    let raw_id = (read_u32_le_unaligned(d, byte_off) >> shift) & 0x03FF_FFFF;
                    *slot = normalize_link(raw_id, INVALID_AI_LINK_V13);
                }
                let cover_high = u16::from_le_bytes([rec[13], rec[14]]);
                let cover_low = u16::from_le_bytes([rec[15], rec[16]]);
                let plane = u16::from_le_bytes([rec[17], rec[18]]);
                let xz = u32::from_le_bytes([rec[19], rec[20], rec[21], rec[22]]);
                let y = u16::from_le_bytes([rec[23], rec[24]]);

                vertices.push(AiVertex {
                    links,
                    cover_high,
                    cover_low,
                    plane,
                    xz,
                    y,
                });
            }
        }

        Ok(Self { header, vertices })
    }

    #[inline]
    pub fn vertex_position(&self, id: u32) -> Option<[f32; 3]> {
        self.vertices
            .get(id as usize)
            .map(|v| v.world_position(&self.header))
    }

    #[inline]
    pub fn vertex_plane_normal(&self, id: u32) -> Option<[f32; 3]> {
        self.vertices
            .get(id as usize)
            .map(|v| pv_decompress_normal(v.plane))
    }

    /// Evaluate vertical height `y` on AI node `id`'s tangent plane at world `(x, z)`.
    pub fn vertex_plane_y(&self, id: u32, x: f32, z: f32) -> Option<f32> {
        let v = self.vertices.get(id as usize)?;
        let p = v.world_position(&self.header);
        let n = pv_decompress_normal(v.plane);
        if n[1].abs() <= 1e-6 {
            return Some(p[1]);
        }
        // Plane passes through p with normal n: n.(q - p) = 0 => n_y*(y - p_y) = -(n_x*(x - p_x) + n_z*(z - p_z))
        let dy = -(n[0] * (x - p[0]) + n[2] * (z - p[2])) / n[1];
        Some(p[1] + dy)
    }

    /// Quantize a world `(x, z)` into an `xz` grid key.
    pub fn quantize_xz(&self, pos: [f32; 3]) -> Option<u32> {
        if pos[0] < self.header.box_aabb.min[0] - self.header.cell_size * 0.5
            || pos[0] > self.header.box_aabb.max[0] + self.header.cell_size * 0.5
            || pos[2] < self.header.box_aabb.min[2] - self.header.cell_size * 0.5
            || pos[2] > self.header.box_aabb.max[2] + self.header.cell_size * 0.5
        {
            return None;
        }
        let gx = ((pos[0] - self.header.box_aabb.min[0]) / self.header.cell_size + 0.5).floor()
            as i32;
        let gz = ((pos[2] - self.header.box_aabb.min[2]) / self.header.cell_size + 0.5).floor()
            as i32;
        if gx < 0 || gz < 0 || gz >= self.header.row_length as i32 {
            return None;
        }
        Some((gx as u32) * self.header.row_length + (gz as u32))
    }

    /// Locate an AI vertex covering `(pos.x, pos.z)` (picking closest `y` among multi-floor matches).
    pub fn find_vertex_by_position(&self, pos: [f32; 3]) -> Option<u32> {
        let target_xz = self.quantize_xz(pos)?;
        let mut first = self.vertices.partition_point(|v| v.xz < target_xz);

        // Fallback if input vertices aren't strictly sorted by xz
        let mut best_id: Option<u32> = None;
        let mut best_dy = f32::INFINITY;

        if first < self.vertices.len() && self.vertices[first].xz == target_xz {
            while first < self.vertices.len() && self.vertices[first].xz == target_xz {
                let p = self.vertices[first].world_position(&self.header);
                let dy = (p[1] - pos[1]).abs();
                if dy < best_dy {
                    best_dy = dy;
                    best_id = Some(first as u32);
                }
                first += 1;
            }
            return best_id;
        }

        for (idx, v) in self.vertices.iter().enumerate() {
            if v.xz == target_xz {
                let p = v.world_position(&self.header);
                let dy = (p[1] - pos[1]).abs();
                if dy < best_dy {
                    best_dy = dy;
                    best_id = Some(idx as u32);
                }
            }
        }
        best_id
    }

    /// Find the closest AI vertex in 3D Euclidean distance.
    pub fn nearest_vertex(&self, pos: [f32; 3]) -> Option<u32> {
        let mut best = None;
        let mut best_dist_sq = f32::INFINITY;
        for (idx, v) in self.vertices.iter().enumerate() {
            let p = v.world_position(&self.header);
            let dx = p[0] - pos[0];
            let dy = p[1] - pos[1];
            let dz = p[2] - pos[2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best = Some(idx as u32);
            }
        }
        best
    }

    /// Find shortest path from `start` to `goal` using Euclidean A* across 4-way node links.
    pub fn find_path_astar(&self, start: u32, goal: u32) -> Option<Vec<u32>> {
        let count = self.vertices.len() as u32;
        if start >= count || goal >= count {
            return None;
        }
        if start == goal {
            return Some(vec![start]);
        }

        let goal_pos = self.vertices[goal as usize].world_position(&self.header);
        let start_pos = self.vertices[start as usize].world_position(&self.header);

        let mut g_score: HashMap<u32, f32> = HashMap::new();
        let mut came_from: HashMap<u32, u32> = HashMap::new();
        let mut open = BinaryHeap::new();

        g_score.insert(start, 0.0);
        open.push(AStarState {
            vertex: start,
            f_cost: dist3(start_pos, goal_pos),
        });

        while let Some(AStarState { vertex, .. }) = open.pop() {
            if vertex == goal {
                let mut path = vec![goal];
                let mut cur = goal;
                while let Some(&prev) = came_from.get(&cur) {
                    path.push(prev);
                    if prev == start {
                        break;
                    }
                    cur = prev;
                }
                path.reverse();
                return Some(path);
            }

            let cur_g = g_score.get(&vertex).copied().unwrap_or(f32::INFINITY);
            let cur_pos = self.vertices[vertex as usize].world_position(&self.header);

            for dir in 0..4 {
                let Some(next) = self.vertices[vertex as usize].link(dir) else {
                    continue;
                };
                if next >= count {
                    continue;
                }
                let next_pos = self.vertices[next as usize].world_position(&self.header);
                let step = dist3(cur_pos, next_pos);
                let tentative_g = cur_g + step;
                let known_g = g_score.get(&next).copied().unwrap_or(f32::INFINITY);
                if tentative_g < known_g {
                    came_from.insert(next, vertex);
                    g_score.insert(next, tentative_g);
                    open.push(AStarState {
                        vertex: next,
                        f_cost: tentative_g + dist3(next_pos, goal_pos),
                    });
                }
            }
        }

        None
    }
}

#[inline]
fn normalize_link(raw: u32, invalid_sentinel: u32) -> u32 {
    if raw == invalid_sentinel {
        INVALID_AI_VERTEX
    } else {
        raw
    }
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[derive(Debug, Clone, Copy)]
struct AStarState<Id> {
    vertex: Id,
    f_cost: f32,
}

impl<Id: Ord> PartialEq for AStarState<Id> {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost.to_bits() == other.f_cost.to_bits() && self.vertex == other.vertex
    }
}
impl<Id: Ord> Eq for AStarState<Id> {}
impl<Id: Ord> PartialOrd for AStarState<Id> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<Id: Ord> Ord for AStarState<Id> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_cost
            .total_cmp(&self.f_cost)
            .then_with(|| other.vertex.cmp(&self.vertex))
    }
}

/// 44-byte `CGameLevelCrossTable::CHeader` (`src/xrGame/game_level_cross_table.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossTableHeader {
    pub version: u32,
    pub level_vertex_count: u32,
    pub game_vertex_count: u16,
    pub level_guid: [u64; 2],
    pub game_guid: [u64; 2],
}

/// 6-byte `CGameLevelCrossTable::CCell` (`u16 game_vertex_id + f32 distance`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossTableCell {
    pub game_vertex_id: u16,
    pub distance: f32,
}

/// Mapping from every single-level `level.ai` node index to its nearest `GameGraph` vertex.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelCrossTable {
    pub header: CrossTableHeader,
    pub cells: Vec<CrossTableCell>,
}

impl LevelCrossTable {
    /// Parse a standalone chunked `level.gct` file (`CROSS_TABLE_CHUNK_VERSION` = 0, `DATA` = 1).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let header_chunk = reader.require_chunk(0)?;
        let data_chunk = reader.require_chunk(1)?;

        let header = Self::parse_header(&header_chunk)?;
        let cells = Self::parse_cells(&data_chunk, header.level_vertex_count as usize)?;
        Ok(Self { header, cells })
    }

    /// Parse raw unchunked cross-table payload (embedded inside `game.graph`), returning `(table, bytes_read)`.
    pub fn parse_raw(data: &[u8]) -> Result<(Self, usize), FormatError> {
        if data.len() < 44 {
            return Err(FormatError::UnexpectedEof);
        }
        let header = Self::parse_header(&data[..44])?;
        let cell_bytes = (header.level_vertex_count as usize)
            .checked_mul(6)
            .ok_or(FormatError::UnexpectedEof)?;
        let total = 44usize
            .checked_add(cell_bytes)
            .ok_or(FormatError::UnexpectedEof)?;
        if data.len() < total {
            return Err(FormatError::UnexpectedEof);
        }
        let cells = Self::parse_cells(&data[44..total], header.level_vertex_count as usize)?;
        Ok((Self { header, cells }, total))
    }

    fn parse_header(bytes: &[u8]) -> Result<CrossTableHeader, FormatError> {
        let mut cur = ByteCursor::new(bytes);
        let version = cur.read_u32()?;
        let level_vertex_count = cur.read_u32()?;
        let game_vertex_count = cur.read_u32()? as u16;
        let level_guid = [cur.read_u64()?, cur.read_u64()?];
        let game_guid = [cur.read_u64()?, cur.read_u64()?];
        Ok(CrossTableHeader {
            version,
            level_vertex_count,
            game_vertex_count,
            level_guid,
            game_guid,
        })
    }

    fn parse_cells(bytes: &[u8], expected_count: usize) -> Result<Vec<CrossTableCell>, FormatError> {
        let need = expected_count
            .checked_mul(6)
            .ok_or(FormatError::UnexpectedEof)?;
        if bytes.len() < need {
            return Err(FormatError::UnexpectedEof);
        }
        let mut cells = Vec::with_capacity(expected_count);
        for rec in bytes[..need].as_chunks::<6>().0 {
            let game_vertex_id = u16::from_le_bytes([rec[0], rec[1]]);
            let distance = f32::from_le_bytes([rec[2], rec[3], rec[4], rec[5]]);
            cells.push(CrossTableCell {
                game_vertex_id,
                distance,
            });
        }
        Ok(cells)
    }

    #[inline]
    pub fn game_vertex_for_level_vertex(&self, level_vertex_id: u32) -> Option<&CrossTableCell> {
        self.cells.get(level_vertex_id as usize)
    }
}

/// Level descriptor in `game.graph` header (`GameGraph::SLevel`).
#[derive(Debug, Clone, PartialEq)]
pub struct GameLevelInfo {
    pub name: String,
    pub offset: [f32; 3],
    pub id: u8,
    pub section: String,
    pub guid: [u64; 2],
}

/// Header of `game.graph` (`GameGraph::CHeader`).
#[derive(Debug, Clone, PartialEq)]
pub struct GameGraphHeader {
    pub version: u8,
    pub vertex_count: u16,
    pub edge_count: u32,
    pub death_point_count: u32,
    pub guid: [u64; 2],
    pub levels: Vec<GameLevelInfo>,
}

/// 42-byte `GameGraph::CVertex` decoded from `game.graph`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GameVertex {
    pub local_point: [f32; 3],
    pub global_point: [f32; 3],
    pub level_id: u8,
    pub level_vertex_id: u32,
    pub vertex_types: [u8; 4],
    pub edge_offset: u32,
    pub point_offset: u32,
    pub neighbour_count: u8,
    pub death_point_count: u8,
}

/// 6-byte `GameGraph::CEdge` link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GameEdge {
    pub target_vertex_id: u16,
    pub distance: f32,
}

/// 20-byte `GameGraph::CLevelPoint` spawn/death point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GameDeathPoint {
    pub point: [f32; 3],
    pub level_vertex_id: u32,
    pub distance: f32,
}

/// Parsed global inter-level navigation `game.graph`.
#[derive(Debug, Clone, PartialEq)]
pub struct GameGraph {
    pub header: GameGraphHeader,
    pub vertices: Vec<GameVertex>,
    pub edges: Vec<GameEdge>,
    pub death_points: Vec<GameDeathPoint>,
    /// Embedded `(level_id, LevelCrossTable)` entries sorted by `level_id` (when present).
    pub cross_tables: Vec<(u8, LevelCrossTable)>,
}

impl GameGraph {
    /// Parse binary `game.graph`.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let version = cur.read_u8()?;
        let vertex_count = cur.read_u16()?;
        let edge_count = cur.read_u32()?;
        let death_point_count = cur.read_u32()?;
        let guid = [cur.read_u64()?, cur.read_u64()?];
        let level_count = cur.read_u8()? as usize;

        let mut levels = Vec::with_capacity(level_count);
        for _ in 0..level_count {
            let name = cur.read_zstring()?;
            let offset = cur.read_f32_3()?;
            let id = cur.read_u8()?;
            let section = cur.read_zstring()?;
            let lvl_guid = [cur.read_u64()?, cur.read_u64()?];
            levels.push(GameLevelInfo {
                name,
                offset,
                id,
                section,
                guid: lvl_guid,
            });
        }

        let header = GameGraphHeader {
            version,
            vertex_count,
            edge_count,
            death_point_count,
            guid,
            levels,
        };

        let vert_bytes = (vertex_count as usize)
            .checked_mul(42)
            .ok_or(FormatError::UnexpectedEof)?;
        let raw_verts = cur.read_bytes(vert_bytes)?;
        let mut vertices = Vec::with_capacity(vertex_count as usize);
        for rec in raw_verts.as_chunks::<42>().0 {
            let mut vc = ByteCursor::new(rec);
            let local_point = vc.read_f32_3()?;
            let global_point = vc.read_f32_3()?;
            let packed = vc.read_u32()?;
            let level_id = (packed & 0xFF) as u8;
            let level_vertex_id = packed >> 8;
            let vt_raw = vc.read_bytes(4)?;
            let vertex_types = [vt_raw[0], vt_raw[1], vt_raw[2], vt_raw[3]];
            let edge_offset = vc.read_u32()?;
            let point_offset = vc.read_u32()?;
            let neighbour_count = vc.read_u8()?;
            let dp_count = vc.read_u8()?;
            vertices.push(GameVertex {
                local_point,
                global_point,
                level_id,
                level_vertex_id,
                vertex_types,
                edge_offset,
                point_offset,
                neighbour_count,
                death_point_count: dp_count,
            });
        }

        let edge_bytes = (edge_count as usize)
            .checked_mul(6)
            .ok_or(FormatError::UnexpectedEof)?;
        let raw_edges = cur.read_bytes(edge_bytes)?;
        let mut edges = Vec::with_capacity(edge_count as usize);
        for rec in raw_edges.as_chunks::<6>().0 {
            edges.push(GameEdge {
                target_vertex_id: u16::from_le_bytes([rec[0], rec[1]]),
                distance: f32::from_le_bytes([rec[2], rec[3], rec[4], rec[5]]),
            });
        }

        let dp_bytes = (death_point_count as usize)
            .checked_mul(20)
            .ok_or(FormatError::UnexpectedEof)?;
        let raw_dps = cur.read_bytes(dp_bytes)?;
        let mut death_points = Vec::with_capacity(death_point_count as usize);
        for rec in raw_dps.as_chunks::<20>().0 {
            let mut dc = ByteCursor::new(rec);
            let point = dc.read_f32_3()?;
            let level_vertex_id = dc.read_u32()?;
            let distance = dc.read_f32()?;
            death_points.push(GameDeathPoint {
                point,
                level_vertex_id,
                distance,
            });
        }

        // Parse embedded per-level cross tables sorted ascending by SLevel::id (game_graph_inline.h:413-420).
        let mut cross_tables = Vec::new();
        if cur.remaining() >= 4 && !header.levels.is_empty() {
            let mut sorted_ids: Vec<u8> = header.levels.iter().map(|l| l.id).collect();
            sorted_ids.sort_unstable();
            for lvl_id in sorted_ids {
                if cur.remaining() < 4 {
                    break;
                }
                let block_size = cur.read_u32()? as usize;
                if block_size < 4 {
                    return Err(FormatError::InvalidValue(
                        "game.graph embedded cross-table block size < 4",
                    ));
                }
                let payload = cur.read_bytes(block_size - 4)?;
                let (table, _) = LevelCrossTable::parse_raw(payload)?;
                cross_tables.push((lvl_id, table));
            }
        }

        Ok(Self {
            header,
            vertices,
            edges,
            death_points,
            cross_tables,
        })
    }

    /// Return edge slice for `vertex_id` using `dwEdgeOffset` relative to the vertex array base.
    pub fn vertex_edges(&self, vertex_id: u16) -> &[GameEdge] {
        let Some(v) = self.vertices.get(vertex_id as usize) else {
            return &[];
        };
        if v.neighbour_count == 0 {
            return &[];
        }
        let verts_base = self.vertices.len() * 42;
        let off = v.edge_offset as usize;
        if off < verts_base || !(off - verts_base).is_multiple_of(6) {
            return &[];
        }
        let start = (off - verts_base) / 6;
        let end = start.saturating_add(v.neighbour_count as usize);
        self.edges.get(start..end).unwrap_or(&[])
    }

    /// Return death/spawn point slice for `vertex_id` using `dwPointOffset`.
    pub fn vertex_death_points(&self, vertex_id: u16) -> &[GameDeathPoint] {
        let Some(v) = self.vertices.get(vertex_id as usize) else {
            return &[];
        };
        if v.death_point_count == 0 {
            return &[];
        }
        let dp_base = self.vertices.len() * 42 + self.edges.len() * 6;
        let off = v.point_offset as usize;
        if off < dp_base || !(off - dp_base).is_multiple_of(20) {
            return &[];
        }
        let start = (off - dp_base) / 20;
        let end = start.saturating_add(v.death_point_count as usize);
        self.death_points.get(start..end).unwrap_or(&[])
    }

    /// Look up the embedded cross-table for `level_id` (if bundled inside `game.graph`).
    pub fn cross_table_for_level(&self, level_id: u8) -> Option<&LevelCrossTable> {
        self.cross_tables
            .iter()
            .find(|(id, _)| *id == level_id)
            .map(|(_, t)| t)
    }

    /// Find the nearest `GameVertex` on `level_id` to `local_pos` (`CGameGraph::nearest_vertex`).
    pub fn nearest_vertex_on_level(&self, level_id: u8, local_pos: [f32; 3]) -> Option<(u16, f32)> {
        let mut best: Option<(u16, f32)> = None;
        for (idx, v) in self.vertices.iter().enumerate() {
            if v.level_id != level_id {
                continue;
            }
            let d = dist3(v.local_point, local_pos);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((idx as u16, d));
            }
        }
        best
    }

    /// Shortest path between two `GameGraph` vertices via A*, returning `(vertex_path, total_distance)`.
    pub fn find_game_path(&self, start: u16, goal: u16) -> Option<(Vec<u16>, f32)> {
        let count = self.vertices.len();
        if start as usize >= count || goal as usize >= count {
            return None;
        }
        if start == goal {
            return Some((vec![start], 0.0));
        }

        let goal_pos = self.vertices[goal as usize].global_point;
        let start_pos = self.vertices[start as usize].global_point;

        let mut g_score: HashMap<u16, f32> = HashMap::new();
        let mut came_from: HashMap<u16, u16> = HashMap::new();
        let mut open = BinaryHeap::new();

        g_score.insert(start, 0.0);
        open.push(AStarState {
            vertex: start,
            f_cost: dist3(start_pos, goal_pos),
        });

        while let Some(AStarState { vertex, .. }) = open.pop() {
            if vertex == goal {
                let total_cost = g_score[&goal];
                let mut path = vec![goal];
                let mut cur = goal;
                while let Some(&prev) = came_from.get(&cur) {
                    path.push(prev);
                    if prev == start {
                        break;
                    }
                    cur = prev;
                }
                path.reverse();
                return Some((path, total_cost));
            }

            let cur_g = g_score.get(&vertex).copied().unwrap_or(f32::INFINITY);
            for edge in self.vertex_edges(vertex) {
                let next = edge.target_vertex_id;
                if next as usize >= count {
                    continue;
                }
                let tentative_g = cur_g + edge.distance.max(0.0);
                let known_g = g_score.get(&next).copied().unwrap_or(f32::INFINITY);
                if tentative_g < known_g {
                    came_from.insert(next, vertex);
                    g_score.insert(next, tentative_g);
                    let h = dist3(self.vertices[next as usize].global_point, goal_pos);
                    open.push(AStarState {
                        vertex: next,
                        f_cost: tentative_g + h,
                    });
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;

    fn pack_links_v10(links: [u32; 4]) -> [u8; 12] {
        // Inverse of read_u32_le_unaligned offsets: (0, >>0), (2, >>7), (5, >>6), (8, >>5)
        let mut bits = 0u128;
        for (i, &val) in links.iter().enumerate() {
            bits |= ((val & 0x007F_FFFF) as u128) << (i * 23);
        }
        let le = bits.to_le_bytes();
        let mut out = [0u8; 12];
        out.copy_from_slice(&le[..12]);
        out
    }

    fn pack_links_v13(links: [u32; 4]) -> [u8; 13] {
        let mut bits = 0u128;
        for (i, &val) in links.iter().enumerate() {
            bits |= ((val & 0x03FF_FFFF) as u128) << (i * 26);
        }
        let le = bits.to_le_bytes();
        let mut out = [0u8; 13];
        out.copy_from_slice(&le[..13]);
        out
    }

    fn write_ai_header(buf: &mut Vec<u8>, version: u32, count: u32, cell: f32) {
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend_from_slice(&cell.to_le_bytes());
        buf.extend_from_slice(&10.0f32.to_le_bytes()); // factor_y
        // min box (-2, 0, -2), max box (4, 10, 4)
        for f in [-2.0f32, 0.0, -2.0, 4.0, 10.0, 4.0] {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        buf.extend_from_slice(&0x8877_6655_4433_2211u64.to_le_bytes());
    }

    #[test]
    fn compressed_normal_roundtrip() {
        for raw in [
            [0.0f32, 1.0, 0.0],
            [0.577_35, 0.577_35, 0.577_35],
            [-0.4, 0.85, -0.33],
            [0.99, -0.1, 0.05],
        ] {
            let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
            let unit = [raw[0] / len, raw[1] / len, raw[2] / len];
            let packed = pv_compress_normal(unit);
            let decomp = pv_decompress_normal(packed);
            let dot = unit[0] * decomp[0] + unit[1] * decomp[1] + unit[2] * decomp[2];
            assert!(dot > 0.99, "normal roundtrip drift for {unit:?}: dot={dot}");
        }
    }

    #[test]
    fn parse_level_ai_v10_and_astar() {
        let mut buf = Vec::new();
        // 3 nodes in an L-shape: 0 -> 1 -> 2 (grid row_length = floor(6/0.7 + 0.001 + 1.5) = 10)
        // Wait with cell=1.0, min=-2..max=4 -> span=6 -> floor(6 + 0.001 + 1.5) = 7
        write_ai_header(&mut buf, XRAI_CURRENT_VERSION, 3, 1.0);
        let plane_up = pv_compress_normal([0.0, 1.0, 0.0]);
        let nodes = [
            ([1, INVALID_AI_LINK_V10, INVALID_AI_LINK_V10, INVALID_AI_LINK_V10], 0u32, 0u16),
            ([2, 0, INVALID_AI_LINK_V10, INVALID_AI_LINK_V10], 7u32, 32767u16),
            ([INVALID_AI_LINK_V10, 1, INVALID_AI_LINK_V10, INVALID_AI_LINK_V10], 8u32, 32767u16),
        ];
        for (links, xz, y) in nodes {
            buf.extend_from_slice(&pack_links_v10(links));
            buf.extend_from_slice(&0x1234u16.to_le_bytes()); // cover_high
            buf.extend_from_slice(&0xABCDu16.to_le_bytes()); // cover_low
            buf.extend_from_slice(&plane_up.to_le_bytes());
            buf.extend_from_slice(&xz.to_le_bytes()[..3]);
            buf.extend_from_slice(&y.to_le_bytes());
        }

        let lg = LevelGraph::parse(&buf).unwrap();
        assert_eq!(lg.header.row_length, 7);
        assert_eq!(lg.vertices.len(), 3);
        assert_eq!(lg.vertices[0].link(0), Some(1));
        assert_eq!(lg.vertices[0].link(1), None);
        assert_eq!(lg.vertices[0].cover(1), (0x3, 0xC));

        let pos1 = lg.vertex_position(1).unwrap();
        assert!((pos1[0] - (-1.0)).abs() < 1e-4);
        assert!((pos1[2] - (-2.0)).abs() < 1e-4);
        assert!((pos1[1] - 5.0).abs() < 1e-2);

        assert_eq!(lg.find_vertex_by_position(pos1), Some(1));
        assert_eq!(lg.find_path_astar(0, 2), Some(vec![0, 1, 2]));
        assert!((lg.vertex_plane_y(1, 3.0, 1.0).unwrap() - pos1[1]).abs() < 0.05);
    }

    #[test]
    fn parse_level_ai_v13() {
        let mut buf = Vec::new();
        write_ai_header(&mut buf, XRAI_LARGE_VERSION, 2, 1.0);
        let plane = pv_compress_normal([0.0, 1.0, 0.0]);
        let nodes = [
            ([1, INVALID_AI_LINK_V13, INVALID_AI_LINK_V13, INVALID_AI_LINK_V13], 0x0010_0001u32, 0u16),
            ([0, INVALID_AI_LINK_V13, INVALID_AI_LINK_V13, INVALID_AI_LINK_V13], 0x0010_0002u32, 1000u16),
        ];
        for (links, xz, y) in nodes {
            buf.extend_from_slice(&pack_links_v13(links));
            buf.extend_from_slice(&0u16.to_le_bytes());
            buf.extend_from_slice(&0u16.to_le_bytes());
            buf.extend_from_slice(&plane.to_le_bytes());
            buf.extend_from_slice(&xz.to_le_bytes());
            buf.extend_from_slice(&y.to_le_bytes());
        }

        let lg = LevelGraph::parse(&buf).unwrap();
        assert_eq!(lg.header.version, XRAI_LARGE_VERSION);
        assert_eq!(lg.vertices[0].xz, 0x0010_0001);
        assert_eq!(lg.vertices[0].link(0), Some(1));
        assert_eq!(lg.vertices[1].link(0), Some(0));
        assert_eq!(lg.vertices[1].link(3), None);
    }

    #[test]
    fn parse_level_gct_and_game_graph_with_embedded_cross_table() {
        // Build standalone level.gct
        let mut gct_hdr = Vec::new();
        gct_hdr.extend_from_slice(&9u32.to_le_bytes()); // version
        gct_hdr.extend_from_slice(&2u32.to_le_bytes()); // level_vertex_count
        gct_hdr.extend_from_slice(&3u32.to_le_bytes()); // game_vertex_count
        gct_hdr.extend_from_slice(&1u64.to_le_bytes());
        gct_hdr.extend_from_slice(&2u64.to_le_bytes());
        gct_hdr.extend_from_slice(&3u64.to_le_bytes());
        gct_hdr.extend_from_slice(&4u64.to_le_bytes());

        let mut gct_cells = Vec::new();
        gct_cells.extend_from_slice(&0u16.to_le_bytes());
        gct_cells.extend_from_slice(&1.25f32.to_le_bytes());
        gct_cells.extend_from_slice(&2u16.to_le_bytes());
        gct_cells.extend_from_slice(&4.5f32.to_le_bytes());

        let mut cw = ChunkWriter::new();
        cw.write_chunk(0, &gct_hdr);
        cw.write_chunk(1, &gct_cells);
        let gct_bytes = cw.into_bytes();
        let gct = LevelCrossTable::parse(&gct_bytes).unwrap();
        assert_eq!(gct.game_vertex_for_level_vertex(1).unwrap().game_vertex_id, 2);

        // Build game.graph: 3 vertices across 2 levels, edges 0->1 (5.0), 1->2 (3.0), 0->2 (20.0)
        let mut gg = Vec::new();
        gg.push(10u8); // version
        gg.extend_from_slice(&3u16.to_le_bytes()); // vertex_count
        gg.extend_from_slice(&3u32.to_le_bytes()); // edge_count
        gg.extend_from_slice(&1u32.to_le_bytes()); // death_point_count
        gg.extend_from_slice(&0xAAu64.to_le_bytes());
        gg.extend_from_slice(&0xBBu64.to_le_bytes());
        gg.push(1u8); // 1 level for simplicity of embedded cross table

        gg.extend_from_slice(b"l01_escape\0");
        for f in [0.0f32, 0.0, 0.0] {
            gg.extend_from_slice(&f.to_le_bytes());
        }
        gg.push(1u8); // id = 1
        gg.extend_from_slice(b"level01\0");
        gg.extend_from_slice(&1u64.to_le_bytes());
        gg.extend_from_slice(&2u64.to_le_bytes());

        let verts_bytes_base = 3usize * 42;
        let dp_bytes_base = verts_bytes_base + 3 * 6;
        let verts_spec = [
            ([0.0f32, 0.0, 0.0], [0.0f32, 0.0, 0.0], 1u8, 10u32, verts_bytes_base as u32, dp_bytes_base as u32, 2u8, 1u8),
            ([10.0, 0.0, 0.0], [10.0, 0.0, 0.0], 1, 20, (verts_bytes_base + 12) as u32, dp_bytes_base as u32, 1, 0),
            ([15.0, 0.0, 0.0], [15.0, 0.0, 0.0], 2, 30, (verts_bytes_base + 18) as u32, dp_bytes_base as u32, 0, 0),
        ];
        for (lp, gp, lid, lvid, eo, po, nc, dpc) in verts_spec {
            for f in lp {
                gg.extend_from_slice(&f.to_le_bytes());
            }
            for f in gp {
                gg.extend_from_slice(&f.to_le_bytes());
            }
            let packed = (lid as u32) | (lvid << 8);
            gg.extend_from_slice(&packed.to_le_bytes());
            gg.extend_from_slice(&[1, 2, 3, 4]);
            gg.extend_from_slice(&eo.to_le_bytes());
            gg.extend_from_slice(&po.to_le_bytes());
            gg.push(nc);
            gg.push(dpc);
        }

        // Edges: (1, 5.0), (2, 20.0), (2, 3.0)
        for (target, dist) in [(1u16, 5.0f32), (2, 20.0), (2, 3.0)] {
            gg.extend_from_slice(&target.to_le_bytes());
            gg.extend_from_slice(&dist.to_le_bytes());
        }

        // 1 Death point
        for f in [1.0f32, 2.0, 3.0] {
            gg.extend_from_slice(&f.to_le_bytes());
        }
        gg.extend_from_slice(&10u32.to_le_bytes());
        gg.extend_from_slice(&0.5f32.to_le_bytes());

        // Embedded cross-table block (prefixed by u32 size including itself)
        let block_len = (4 + gct_hdr.len() + gct_cells.len()) as u32;
        gg.extend_from_slice(&block_len.to_le_bytes());
        gg.extend_from_slice(&gct_hdr);
        gg.extend_from_slice(&gct_cells);

        let parsed = GameGraph::parse(&gg).unwrap();
        assert_eq!(parsed.header.levels[0].name, "l01_escape");
        assert_eq!(parsed.vertices[0].level_vertex_id, 10);
        assert_eq!(parsed.vertex_death_points(0).len(), 1);
        assert!(parsed.cross_table_for_level(1).is_some());
        assert_eq!(parsed.nearest_vertex_on_level(1, [9.5, 0.0, 0.0]), Some((1, 0.5)));

        let (path, dist) = parsed.find_game_path(0, 2).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
        assert!((dist - 8.0).abs() < 1e-5);
    }
}
