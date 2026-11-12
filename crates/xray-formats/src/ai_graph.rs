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
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use crate::chunk::{ByteCursor, ChunkReader, ChunkWriter, FormatError};
use crate::ogf::OgfBBox;

pub const DEFAULT_BUCKET_COUNT: usize = 8192;
pub const DEFAULT_MAX_VISITED_NODES: u32 = 65_500;
const TAU_3_4: f32 = 3.0 * FRAC_PI_2;

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

    /// Encode this `LevelGraph` back into binary `level.ai` wire format (`version` 10 or 13).
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let ver = self.header.version;
        if ver != XRAI_CURRENT_VERSION && ver != XRAI_LARGE_VERSION {
            return Err(FormatError::InvalidValue(
                "unsupported level.ai version for encoding (expected 10 or 13)",
            ));
        }

        let count = self.vertices.len() as u32;
        let rec_size = if ver == XRAI_CURRENT_VERSION { 23 } else { 25 };
        let mut out = Vec::with_capacity(56 + self.vertices.len() * rec_size);

        // 56-byte hdrNODES
        out.extend_from_slice(&ver.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&self.header.cell_size.to_le_bytes());
        out.extend_from_slice(&self.header.factor_y.to_le_bytes());
        for f in self.header.box_aabb.min {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in self.header.box_aabb.max {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&self.header.guid[0].to_le_bytes());
        out.extend_from_slice(&self.header.guid[1].to_le_bytes());

        if ver == XRAI_CURRENT_VERSION {
            for v in &self.vertices {
                let mut link_bits = 0u128;
                for (i, &lnk) in v.links.iter().enumerate() {
                    let encoded = if lnk == INVALID_AI_VERTEX {
                        INVALID_AI_LINK_V10
                    } else if lnk < INVALID_AI_LINK_V10 {
                        lnk
                    } else {
                        return Err(FormatError::InvalidValue(
                            "v10 level.ai neighbor link exceeds 23-bit capacity",
                        ));
                    };
                    link_bits |= (encoded as u128) << (i * 23);
                }
                if v.xz > 0x00FF_FFFF {
                    return Err(FormatError::InvalidValue(
                        "v10 level.ai vertex xz exceeds 24-bit capacity",
                    ));
                }
                let le = link_bits.to_le_bytes();
                out.extend_from_slice(&le[..12]);
                out.extend_from_slice(&v.cover_high.to_le_bytes());
                out.extend_from_slice(&v.cover_low.to_le_bytes());
                out.extend_from_slice(&v.plane.to_le_bytes());
                out.extend_from_slice(&v.xz.to_le_bytes()[..3]);
                out.extend_from_slice(&v.y.to_le_bytes());
            }
        } else {
            for v in &self.vertices {
                let mut link_bits = 0u128;
                for (i, &lnk) in v.links.iter().enumerate() {
                    let encoded = if lnk == INVALID_AI_VERTEX {
                        INVALID_AI_LINK_V13
                    } else if lnk < INVALID_AI_LINK_V13 {
                        lnk
                    } else {
                        return Err(FormatError::InvalidValue(
                            "v13 level.ai neighbor link exceeds 26-bit capacity",
                        ));
                    };
                    link_bits |= (encoded as u128) << (i * 26);
                }
                let le = link_bits.to_le_bytes();
                out.extend_from_slice(&le[..13]);
                out.extend_from_slice(&v.cover_high.to_le_bytes());
                out.extend_from_slice(&v.cover_low.to_le_bytes());
                out.extend_from_slice(&v.plane.to_le_bytes());
                out.extend_from_slice(&v.xz.to_le_bytes());
                out.extend_from_slice(&v.y.to_le_bytes());
            }
        }

        Ok(out)
    }

    /// Bilinearly interpolate 4-direction cover nibbles at `angle_yaw_rad` (`[0, TAU)`)
    /// matching `CLevelGraph::cover_in_direction` (`level_graph_vertex.cpp:513-538`).
    pub fn cover_in_direction(&self, vertex_id: u32, angle_yaw_rad: f32, high_cover: bool) -> Option<f32> {
        let v = self.vertices.get(vertex_id as usize)?;
        let c = [0, 1, 2, 3].map(|dir| {
            let (hi, lo) = v.cover(dir);
            let raw = if high_cover { hi } else { lo };
            f32::from(raw) / 15.0
        });
        Some(cover_in_direction_nibbles(angle_yaw_rad, c))
    }

    /// Check if world `(x, z)` lies inside AI node `vertex_id`'s square grid footprint (`CLevelGraph::inside`).
    #[inline]
    pub fn vertex_contains_xz(&self, vertex_id: u32, x: f32, z: f32) -> bool {
        let Some(v) = self.vertices.get(vertex_id as usize) else {
            return false;
        };
        let half = self.header.cell_size * 0.5;
        let center = v.world_position(&self.header);
        (x - center[0]).abs() <= half + 1e-4 && (z - center[2]).abs() <= half + 1e-4
    }

    /// Shortest path search using preallocated [`AiSearchWorkspace`] (`BinaryHeap` or `BucketList`)
    /// and optional [`LevelAccessMask`].
    pub fn find_path_astar_with_workspace(
        &self,
        start: u32,
        goal: u32,
        params: &AiSearchParams,
        access_mask: Option<&LevelAccessMask>,
        workspace: &mut AiSearchWorkspace,
    ) -> Option<AiSearchOutcome> {
        self.find_path_astar_internal(
            start,
            goal,
            params,
            access_mask,
            workspace,
            |_, _, base_step| base_step,
        )
    }

    /// Cover-weighted A* pathfinding preferring nodes with stronger directional cover away from `enemy_pos`.
    #[allow(clippy::too_many_arguments)]
    pub fn find_path_astar_cover_weighted(
        &self,
        start: u32,
        goal: u32,
        enemy_pos: [f32; 3],
        cover_weight: f32,
        high_cover: bool,
        params: &AiSearchParams,
        access_mask: Option<&LevelAccessMask>,
        workspace: &mut AiSearchWorkspace,
    ) -> Option<AiSearchOutcome> {
        self.find_path_astar_internal(
            start,
            goal,
            params,
            access_mask,
            workspace,
            |next_id, next_pos, base_step| {
                let dx = enemy_pos[0] - next_pos[0];
                let dz = enemy_pos[2] - next_pos[2];
                let angle = dx.atan2(dz).rem_euclid(TAU);
                let cover = self
                    .cover_in_direction(next_id, angle, high_cover)
                    .unwrap_or(0.0);
                // Exposed locations (low cover) incur extra traversal cost
                base_step + (1.0 - cover) * cover_weight.max(0.0)
            },
        )
    }

    fn find_path_astar_internal<F>(
        &self,
        start: u32,
        goal: u32,
        params: &AiSearchParams,
        access_mask: Option<&LevelAccessMask>,
        workspace: &mut AiSearchWorkspace,
        mut edge_cost: F,
    ) -> Option<AiSearchOutcome>
    where
        F: FnMut(u32, [f32; 3], f32) -> f32,
    {
        let count = self.vertices.len() as u32;
        if start >= count || goal >= count {
            return None;
        }
        if let Some(mask) = access_mask
            && (!mask.is_accessible(start) || !mask.is_accessible(goal))
        {
            return None;
        }
        if start == goal {
            return Some(AiSearchOutcome {
                path: vec![start],
                total_cost: 0.0,
                iterations: 0,
                visited_nodes: 1,
            });
        }

        let goal_pos = self.vertices[goal as usize].world_position(&self.header);
        let start_pos = self.vertices[start as usize].world_position(&self.header);
        let h_weight = params.heuristic_weight.max(0.0);

        workspace.begin(count as usize, params);
        let h0 = dist3(start_pos, goal_pos) * h_weight;
        workspace.vertices.open_start(start, h0);
        workspace.push_open(start, h0, params);

        let mut iterations = 0u32;
        while let Some((cur, _f)) = workspace.pop_open(params) {
            if iterations >= params.max_iteration_count {
                break;
            }
            iterations += 1;

            if cur == goal {
                let total_cost = workspace.vertices.g_score(goal);
                let visited_nodes = workspace.vertices.visited_count();
                let path = workspace.vertices.reconstruct_path(start, goal);
                return Some(AiSearchOutcome {
                    path,
                    total_cost,
                    iterations,
                    visited_nodes,
                });
            }

            workspace.vertices.close(cur);
            let cur_g = workspace.vertices.g_score(cur);
            let cur_pos = self.vertices[cur as usize].world_position(&self.header);

            for dir in 0..4 {
                let Some(next) = self.vertices[cur as usize].link(dir) else {
                    continue;
                };
                if next >= count || workspace.vertices.is_closed(next) {
                    continue;
                }
                if let Some(mask) = access_mask
                    && !mask.is_accessible(next)
                {
                    continue;
                }
                let next_pos = self.vertices[next as usize].world_position(&self.header);
                let base_step = dist3(cur_pos, next_pos);
                let step = edge_cost(next, next_pos, base_step);
                let tentative_g = cur_g + step;
                if tentative_g > params.max_range {
                    continue;
                }

                let was_visited = workspace.vertices.is_visited(next);
                if !was_visited
                    && workspace.vertices.visited_count() >= params.max_visited_node_count
                {
                    continue;
                }

                let h = dist3(next_pos, goal_pos) * h_weight;
                let new_f = tentative_g + h;
                if workspace.vertices.relax(next, cur, tentative_g, new_f) {
                    workspace.push_or_decrease_open(next, new_f, !was_visited, params);
                }
            }
        }

        None
    }

    /// Breadth/Dijkstra reachability flood from `start` within `params.max_range` (`SFlooder`),
    /// returning reachable `(vertex_id, distance)` ordered by increasing path cost.
    pub fn flood_level(
        &self,
        start: u32,
        params: &AiSearchParams,
        access_mask: Option<&LevelAccessMask>,
        workspace: &mut AiSearchWorkspace,
    ) -> Vec<(u32, f32)> {
        let mut out = Vec::new();
        let count = self.vertices.len() as u32;
        if start >= count {
            return out;
        }
        if let Some(mask) = access_mask
            && !mask.is_accessible(start)
        {
            return out;
        }

        workspace.begin(count as usize, params);
        workspace.vertices.open_start(start, 0.0);
        workspace.push_open(start, 0.0, params);

        let mut iterations = 0u32;
        while let Some((cur, cur_g)) = workspace.pop_open(params) {
            if iterations >= params.max_iteration_count {
                break;
            }
            iterations += 1;
            workspace.vertices.close(cur);
            out.push((cur, cur_g));

            let cur_pos = self.vertices[cur as usize].world_position(&self.header);
            for dir in 0..4 {
                let Some(next) = self.vertices[cur as usize].link(dir) else {
                    continue;
                };
                if next >= count || workspace.vertices.is_closed(next) {
                    continue;
                }
                if let Some(mask) = access_mask
                    && !mask.is_accessible(next)
                {
                    continue;
                }
                let next_pos = self.vertices[next as usize].world_position(&self.header);
                let tentative_g = cur_g + dist3(cur_pos, next_pos);
                if tentative_g > params.max_range {
                    continue;
                }
                let was_visited = workspace.vertices.is_visited(next);
                if !was_visited
                    && workspace.vertices.visited_count() >= params.max_visited_node_count
                {
                    continue;
                }
                if workspace.vertices.relax(next, cur, tentative_g, tentative_g) {
                    workspace.push_or_decrease_open(next, tentative_g, !was_visited, params);
                }
            }
        }
        out
    }

    /// Find nearest reachable vertex to `start` satisfying `predicate` via Dijkstra (`SNearestVertex`).
    pub fn find_nearest_accessible_vertex<P>(
        &self,
        start: u32,
        params: &AiSearchParams,
        access_mask: Option<&LevelAccessMask>,
        workspace: &mut AiSearchWorkspace,
        mut predicate: P,
    ) -> Option<(u32, f32)>
    where
        P: FnMut(u32, &AiVertex) -> bool,
    {
        let count = self.vertices.len() as u32;
        if start >= count {
            return None;
        }
        if let Some(mask) = access_mask
            && !mask.is_accessible(start)
        {
            return None;
        }

        workspace.begin(count as usize, params);
        workspace.vertices.open_start(start, 0.0);
        workspace.push_open(start, 0.0, params);

        let mut iterations = 0u32;
        while let Some((cur, cur_g)) = workspace.pop_open(params) {
            if iterations >= params.max_iteration_count {
                break;
            }
            iterations += 1;
            workspace.vertices.close(cur);

            if predicate(cur, &self.vertices[cur as usize]) {
                return Some((cur, cur_g));
            }

            let cur_pos = self.vertices[cur as usize].world_position(&self.header);
            for dir in 0..4 {
                let Some(next) = self.vertices[cur as usize].link(dir) else {
                    continue;
                };
                if next >= count || workspace.vertices.is_closed(next) {
                    continue;
                }
                if let Some(mask) = access_mask
                    && !mask.is_accessible(next)
                {
                    continue;
                }
                let next_pos = self.vertices[next as usize].world_position(&self.header);
                let tentative_g = cur_g + dist3(cur_pos, next_pos);
                if tentative_g > params.max_range {
                    continue;
                }
                let was_visited = workspace.vertices.is_visited(next);
                if !was_visited
                    && workspace.vertices.visited_count() >= params.max_visited_node_count
                {
                    continue;
                }
                if workspace.vertices.relax(next, cur, tentative_g, tentative_g) {
                    workspace.push_or_decrease_open(next, tentative_g, !was_visited, params);
                }
            }
        }
        None
    }

    /// Walk along linked 4-way AI node cell AABBs from `(start_vertex_id, start_pos)` toward `finish_pos`
    /// (`CLevelGraph::check_position_in_direction_slow`), returning `(reached_vertex_id, pos_on_plane)`
    /// if unobstructed by missing links or `access_mask`.
    pub fn check_position_in_direction_2d(
        &self,
        start_vertex_id: u32,
        start_pos: [f32; 3],
        finish_pos: [f32; 3],
        access_mask: Option<&LevelAccessMask>,
    ) -> Option<(u32, [f32; 3])> {
        let count = self.vertices.len() as u32;
        if start_vertex_id >= count {
            return None;
        }
        if let Some(mask) = access_mask
            && !mask.is_accessible(start_vertex_id)
        {
            return None;
        }

        if self.vertex_contains_xz(start_vertex_id, finish_pos[0], finish_pos[2]) {
            let y = self
                .vertex_plane_y(start_vertex_id, finish_pos[0], finish_pos[2])
                .unwrap_or(finish_pos[1]);
            return Some((start_vertex_id, [finish_pos[0], y, finish_pos[2]]));
        }

        let target_xz = self.quantize_xz(finish_pos)?;
        let half = self.header.cell_size * 0.5;
        let sx = start_pos[0];
        let sz = start_pos[2];
        let dx = finish_pos[0] - sx;
        let dz = finish_pos[2] - sz;

        let mut cur_id = start_vertex_id;
        let mut prev_id = INVALID_AI_VERTEX;
        let cur_p = self.vertices[cur_id as usize].world_position(&self.header);
        let mut cur_sqr = (cur_p[0] - finish_pos[0]).powi(2) + (cur_p[2] - finish_pos[2]).powi(2);

        for _ in 0..count {
            let mut stepped = false;
            for dir in 0..4 {
                let Some(next_id) = self.vertices[cur_id as usize].link(dir) else {
                    continue;
                };
                if next_id == prev_id || next_id >= count {
                    continue;
                }
                let next_v = &self.vertices[next_id as usize];
                let np = next_v.world_position(&self.header);
                let min_x = np[0] - half;
                let max_x = np[0] + half;
                let min_z = np[2] - half;
                let max_z = np[2] + half;
                if !segment_aabb_2d_intersects([sx, sz], [dx, dz], [min_x, min_z], [max_x, max_z]) {
                    continue;
                }
                let next_sqr =
                    (np[0] - finish_pos[0]).powi(2) + (np[2] - finish_pos[2]).powi(2);
                if next_sqr > cur_sqr + 1e-6 {
                    continue;
                }
                if let Some(mask) = access_mask
                    && !mask.is_accessible(next_id)
                {
                    return None;
                }
                if next_v.xz == target_xz
                    || self.vertex_contains_xz(next_id, finish_pos[0], finish_pos[2])
                {
                    let y = self
                        .vertex_plane_y(next_id, finish_pos[0], finish_pos[2])
                        .unwrap_or(finish_pos[1]);
                    return Some((next_id, [finish_pos[0], y, finish_pos[2]]));
                }
                cur_sqr = next_sqr;
                prev_id = cur_id;
                cur_id = next_id;
                stepped = true;
                break;
            }
            if !stepped {
                return None;
            }
        }
        None
    }

    /// Trace a straight 2D XZ path across `level.ai` nodes matching `CLevelGraph::create_straight_path`
    /// (`level_graph_vertex.cpp:396-511`), returning `(nodes, points_3d)` with Y projected onto node planes.
    pub fn create_straight_path_2d(
        &self,
        start_vertex_id: u32,
        start_pos: [f32; 3],
        finish_pos: [f32; 3],
        add_first_point: bool,
    ) -> Option<(Vec<u32>, Vec<[f32; 3]>)> {
        let count = self.vertices.len() as u32;
        if start_vertex_id >= count {
            return None;
        }
        let dest_xz = self.quantize_xz(finish_pos)?;
        let half = self.header.cell_size * 0.5;
        let mut nodes = Vec::new();
        let mut points = Vec::new();

        if add_first_point {
            let y0 = self
                .vertex_plane_y(start_vertex_id, start_pos[0], start_pos[2])
                .unwrap_or(start_pos[1]);
            nodes.push(start_vertex_id);
            points.push([start_pos[0], y0, start_pos[2]]);
        }

        if self.vertices[start_vertex_id as usize].xz == dest_xz
            || self.vertex_contains_xz(start_vertex_id, finish_pos[0], finish_pos[2])
        {
            return Some((nodes, points));
        }

        let sx = start_pos[0];
        let sz = start_pos[2];
        let dx = finish_pos[0] - sx;
        let dz = finish_pos[2] - sz;

        let mut cur_id = start_vertex_id;
        let mut prev_id = start_vertex_id;
        let cur_p = self.vertices[cur_id as usize].world_position(&self.header);
        let mut cur_sqr = (cur_p[0] - finish_pos[0]).powi(2) + (cur_p[2] - finish_pos[2]).powi(2);

        for _ in 0..count {
            let mut stepped = false;
            for dir in 0..4 {
                let Some(next_id) = self.vertices[cur_id as usize].link(dir) else {
                    continue;
                };
                if next_id == prev_id || next_id >= count {
                    continue;
                }
                let next_v = &self.vertices[next_id as usize];
                let np = next_v.world_position(&self.header);
                let min_x = np[0] - half;
                let max_x = np[0] + half;
                let min_z = np[2] - half;
                let max_z = np[2] + half;
                if !segment_aabb_2d_intersects([sx, sz], [dx, dz], [min_x, min_z], [max_x, max_z]) {
                    continue;
                }
                let next_sqr =
                    (np[0] - finish_pos[0]).powi(2) + (np[2] - finish_pos[2]).powi(2);
                if next_sqr > cur_sqr + 1e-6 {
                    continue;
                }

                // Exact C++ `switch (I)` boundary edges for neighbor cell `box` (`level_graph_vertex.cpp:459-484`)
                let (e0, e1) = match dir {
                    0 => ([max_x, max_z], [max_x, min_z]),
                    1 => ([min_x, min_z], [max_x, min_z]),
                    2 => ([min_x, min_z], [min_x, max_z]),
                    _ => ([max_x, max_z], [min_x, max_z]),
                };
                let Some([ix, iz]) = segment_segment_2d_intersect(
                    [sx, sz],
                    [finish_pos[0], finish_pos[2]],
                    e0,
                    e1,
                ) else {
                    continue;
                };
                let iy = self.vertex_plane_y(cur_id, ix, iz).unwrap_or(np[1]);
                nodes.push(cur_id);
                points.push([ix, iy, iz]);

                if next_v.xz == dest_xz {
                    return Some((nodes, points));
                }
                cur_sqr = next_sqr;
                prev_id = cur_id;
                cur_id = next_id;
                stepped = true;
                break;
            }
            if !stepped {
                return None;
            }
        }

        None
    }
}

/// Bilinearly interpolate 4-nibble cover array `[c0, c1, c2, c3]` at `angle_yaw_rad`
/// matching `CLevelGraph::cover_in_direction` (`level_graph_vertex.cpp:513-538`).
#[inline]
pub fn cover_in_direction_nibbles(angle_yaw_rad: f32, covers: [f32; 4]) -> f32 {
    let mut a = angle_yaw_rad.rem_euclid(TAU);
    // C++ parameter signature swaps `(b1, b0, b3, b2)` from `(cover(0), cover(1), cover(2), cover(3))`
    let [mut b1, mut b0, b3, b2] = covers;
    if a < FRAC_PI_2 {
        // first quadrant uses (b0, b1) directly
    } else if a < PI {
        a -= FRAC_PI_2;
        b0 = b1;
        b1 = b2;
    } else if a < TAU_3_4 {
        a -= PI;
        b0 = b2;
        b1 = b3;
    } else {
        a -= TAU_3_4;
        b1 = b0;
        b0 = b3;
    }
    (b1 - b0) * a / FRAC_PI_2 + b0
}

#[inline]
fn segment_aabb_2d_intersects(
    origin: [f32; 2],
    delta: [f32; 2],
    bmin: [f32; 2],
    bmax: [f32; 2],
) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = 1.0f32;
    for (o, d, lo, hi) in [
        (origin[0], delta[0], bmin[0], bmax[0]),
        (origin[1], delta[1], bmin[1], bmax[1]),
    ] {
        if d.abs() < 1e-9 {
            if o < lo || o > hi {
                return false;
            }
        } else {
            let inv = 1.0 / d;
            let mut t0 = (lo - o) * inv;
            let mut t1 = (hi - o) * inv;
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            tmin = tmin.max(t0);
            tmax = tmax.min(t1);
            if tmin > tmax + 1e-5 {
                return false;
            }
        }
    }
    true
}

#[inline]
fn segment_segment_2d_intersect(
    a0: [f32; 2],
    a1: [f32; 2],
    b0: [f32; 2],
    b1: [f32; 2],
) -> Option<[f32; 2]> {
    let r = [a1[0] - a0[0], a1[1] - a0[1]];
    let s = [b1[0] - b0[0], b1[1] - b0[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    let qp = [b0[0] - a0[0], b0[1] - a0[1]];
    if denom.abs() <= 1e-9 {
        return None;
    }
    let t = (qp[0] * s[1] - qp[1] * s[0]) / denom;
    let u = (qp[0] * r[1] - qp[1] * r[0]) / denom;
    if (-1e-5..=1.0 + 1e-5).contains(&t) && (-1e-5..=1.0 + 1e-5).contains(&u) {
        let tc = t.clamp(0.0, 1.0);
        Some([a0[0] + r[0] * tc, a0[1] + r[1] * tc])
    } else {
        None
    }
}

/// Search constraints matching `GraphEngineSpace::SBaseParameters` (`graph_engine_space.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AiSearchParams {
    pub max_range: f32,
    pub max_iteration_count: u32,
    pub max_visited_node_count: u32,
    /// When `true`, uses [`BucketPriorityQueue`] (`CDataStorageBucketList`); otherwise [`IndexedBinaryHeap`].
    pub use_bucket_queue: bool,
    pub bucket_max_value: f32,
    pub heuristic_weight: f32,
}

impl Default for AiSearchParams {
    fn default() -> Self {
        Self {
            max_range: f32::INFINITY,
            max_iteration_count: u32::MAX,
            max_visited_node_count: DEFAULT_MAX_VISITED_NODES,
            use_bucket_queue: false,
            bucket_max_value: 1000.0,
            heuristic_weight: 1.0,
        }
    }
}

/// Completed A* pathfinding outcome with search telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct AiSearchOutcome {
    pub path: Vec<u32>,
    pub total_cost: f32,
    pub iterations: u32,
    pub visited_nodes: u32,
}

/// Per-vertex traversability bitset matching `CLevelGraph::m_access_mask` (`level_graph.h`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelAccessMask {
    bits: Vec<u64>,
    len: usize,
}

impl LevelAccessMask {
    /// Allocate a mask where every vertex `0..len` starts accessible (`true`).
    pub fn all_accessible(len: usize) -> Self {
        let words = len.div_ceil(64);
        Self {
            bits: vec![u64::MAX; words],
            len,
        }
    }

    /// Allocate a mask where all vertices start blocked (`false`).
    pub fn none_accessible(len: usize) -> Self {
        let words = len.div_ceil(64);
        Self {
            bits: vec![0; words],
            len,
        }
    }

    #[inline]
    pub fn is_accessible(&self, vertex_id: u32) -> bool {
        let idx = vertex_id as usize;
        if idx >= self.len {
            return false;
        }
        (self.bits[idx / 64] & (1u64 << (idx % 64))) != 0
    }

    #[inline]
    pub fn set_accessible(&mut self, vertex_id: u32, accessible: bool) {
        let idx = vertex_id as usize;
        if idx >= self.len {
            return;
        }
        let bit = 1u64 << (idx % 64);
        if accessible {
            self.bits[idx / 64] |= bit;
        } else {
            self.bits[idx / 64] &= !bit;
        }
    }
}

/// Handle-indexed binary min-heap matching `CDataStorageBinaryHeap` (`data_storage_binary_heap_inline.h`),
/// supporting `O(log N)` `push`, `decrease_key`, and `pop_min` by dense `handle`.
#[derive(Debug, Clone)]
pub struct IndexedBinaryHeap {
    heap: Vec<(u32, f32)>,
    /// Index in `heap` + 1 (`0` = not in open heap), stamped by `epoch`.
    pos: Vec<u32>,
    item_epoch: Vec<u32>,
    epoch: u32,
}

impl Default for IndexedBinaryHeap {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexedBinaryHeap {
    pub fn new() -> Self {
        Self {
            heap: Vec::new(),
            pos: Vec::new(),
            item_epoch: Vec::new(),
            epoch: 1,
        }
    }

    pub fn clear(&mut self) {
        self.heap.clear();
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.item_epoch.fill(0);
            self.pos.fill(0);
            self.epoch = 1;
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    #[inline]
    fn ensure_handle(&mut self, handle: u32) {
        let idx = handle as usize;
        if idx >= self.pos.len() {
            let new_len = idx + 1;
            self.pos.resize(new_len, 0);
            self.item_epoch.resize(new_len, 0);
        }
        if self.item_epoch[idx] != self.epoch {
            self.item_epoch[idx] = self.epoch;
            self.pos[idx] = 0;
        }
    }

    #[inline]
    pub fn contains(&self, handle: u32) -> bool {
        let idx = handle as usize;
        idx < self.pos.len() && self.item_epoch[idx] == self.epoch && self.pos[idx] != 0
    }

    /// Push a new handle or update its key if strictly smaller.
    pub fn push_or_decrease(&mut self, handle: u32, priority: f32) {
        self.ensure_handle(handle);
        let p = self.pos[handle as usize];
        if p == 0 {
            let heap_idx = self.heap.len();
            self.heap.push((handle, priority));
            self.pos[handle as usize] = (heap_idx + 1) as u32;
            self.sift_up(heap_idx);
        } else {
            let heap_idx = (p - 1) as usize;
            if priority < self.heap[heap_idx].1 {
                self.heap[heap_idx].1 = priority;
                self.sift_up(heap_idx);
            }
        }
    }

    pub fn pop_min(&mut self) -> Option<(u32, f32)> {
        if self.heap.is_empty() {
            return None;
        }
        let min_item = self.heap[0];
        let last = self.heap.pop().unwrap();
        self.pos[min_item.0 as usize] = 0;
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.pos[last.0 as usize] = 1;
            self.sift_down(0);
        }
        Some(min_item)
    }

    #[inline]
    fn less(a: (u32, f32), b: (u32, f32)) -> bool {
        a.1 < b.1 || (a.1.to_bits() == b.1.to_bits() && a.0 < b.0)
    }

    fn sift_up(&mut self, mut idx: usize) {
        let item = self.heap[idx];
        while idx > 0 {
            let parent = (idx - 1) >> 1;
            if Self::less(item, self.heap[parent]) {
                self.heap[idx] = self.heap[parent];
                self.pos[self.heap[idx].0 as usize] = (idx + 1) as u32;
                idx = parent;
            } else {
                break;
            }
        }
        self.heap[idx] = item;
        self.pos[item.0 as usize] = (idx + 1) as u32;
    }

    fn sift_down(&mut self, mut idx: usize) {
        let item = self.heap[idx];
        let n = self.heap.len();
        loop {
            let left = (idx << 1) + 1;
            if left >= n {
                break;
            }
            let right = left + 1;
            let best = if right < n && Self::less(self.heap[right], self.heap[left]) {
                right
            } else {
                left
            };
            if Self::less(self.heap[best], item) {
                self.heap[idx] = self.heap[best];
                self.pos[self.heap[idx].0 as usize] = (idx + 1) as u32;
                idx = best;
            } else {
                break;
            }
        }
        self.heap[idx] = item;
        self.pos[item.0 as usize] = (idx + 1) as u32;
    }
}

/// 8192-bucket quantized open queue matching `CDataStorageBucketList` (`data_storage_bucket_list_inline.h`),
/// with epoch-stamped bucket heads for `O(1)` reset and `O(1)` amortized `decrease_key`/`pop_min`.
#[derive(Debug, Clone)]
pub struct BucketPriorityQueue {
    buckets: Vec<u32>,
    bucket_epoch: Vec<u32>,
    prev: Vec<u32>,
    next: Vec<u32>,
    bucket_of: Vec<u32>,
    priority: Vec<f32>,
    item_epoch: Vec<u32>,
    epoch: u32,
    min_bucket: usize,
    open_count: usize,
    min_val: f32,
    max_val: f32,
    coeff: f32,
}

impl Default for BucketPriorityQueue {
    fn default() -> Self {
        Self::new(DEFAULT_BUCKET_COUNT, 0.0, 1000.0)
    }
}

impl BucketPriorityQueue {
    pub fn new(bucket_count: usize, min_val: f32, max_val: f32) -> Self {
        let bcnt = bucket_count.max(2);
        let span = (max_val - min_val).max(1e-3);
        Self {
            buckets: vec![u32::MAX; bcnt],
            bucket_epoch: vec![0; bcnt],
            prev: Vec::new(),
            next: Vec::new(),
            bucket_of: Vec::new(),
            priority: Vec::new(),
            item_epoch: Vec::new(),
            epoch: 1,
            min_bucket: bcnt,
            open_count: 0,
            min_val,
            max_val,
            coeff: ((bcnt - 1) as f32) / span,
        }
    }

    pub fn configure_range(&mut self, min_val: f32, max_val: f32) {
        self.min_val = min_val;
        self.max_val = max_val;
        let span = (max_val - min_val).max(1e-3);
        self.coeff = ((self.buckets.len() - 1) as f32) / span;
    }

    pub fn clear(&mut self) {
        self.min_bucket = self.buckets.len();
        self.open_count = 0;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.bucket_epoch.fill(0);
            self.item_epoch.fill(0);
            self.epoch = 1;
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.open_count == 0
    }

    #[inline]
    fn bucket_for(&self, val: f32) -> usize {
        if val <= self.min_val {
            0
        } else if val >= self.max_val {
            self.buckets.len() - 1
        } else {
            (((val - self.min_val) * self.coeff) as usize).min(self.buckets.len() - 1)
        }
    }

    #[inline]
    fn ensure_handle(&mut self, handle: u32) {
        let idx = handle as usize;
        if idx >= self.item_epoch.len() {
            let new_len = idx + 1;
            self.prev.resize(new_len, u32::MAX);
            self.next.resize(new_len, u32::MAX);
            self.bucket_of.resize(new_len, u32::MAX);
            self.priority.resize(new_len, f32::INFINITY);
            self.item_epoch.resize(new_len, 0);
        }
        if self.item_epoch[idx] != self.epoch {
            self.item_epoch[idx] = self.epoch;
            self.prev[idx] = u32::MAX;
            self.next[idx] = u32::MAX;
            self.bucket_of[idx] = u32::MAX;
            self.priority[idx] = f32::INFINITY;
        }
    }

    #[inline]
    fn bucket_head(&mut self, b: usize) -> u32 {
        if self.bucket_epoch[b] != self.epoch {
            self.bucket_epoch[b] = self.epoch;
            self.buckets[b] = u32::MAX;
        }
        self.buckets[b]
    }

    fn unlink(&mut self, handle: u32) {
        let idx = handle as usize;
        let b = self.bucket_of[idx] as usize;
        if b >= self.buckets.len() {
            return;
        }
        let p = self.prev[idx];
        let n = self.next[idx];
        if p == u32::MAX {
            self.buckets[b] = n;
        } else {
            self.next[p as usize] = n;
        }
        if n != u32::MAX {
            self.prev[n as usize] = p;
        }
        self.prev[idx] = u32::MAX;
        self.next[idx] = u32::MAX;
        self.bucket_of[idx] = u32::MAX;
        self.open_count -= 1;
    }

    pub fn push_or_decrease(&mut self, handle: u32, priority: f32) {
        self.ensure_handle(handle);
        let idx = handle as usize;
        if self.bucket_of[idx] != u32::MAX {
            if priority >= self.priority[idx] {
                return;
            }
            self.unlink(handle);
        }

        let b = self.bucket_for(priority);
        self.priority[idx] = priority;
        let head = self.bucket_head(b);
        self.prev[idx] = u32::MAX;
        self.next[idx] = head;
        if head != u32::MAX {
            self.prev[head as usize] = handle;
        }
        self.buckets[b] = handle;
        self.bucket_of[idx] = b as u32;
        self.open_count += 1;
        if b < self.min_bucket {
            self.min_bucket = b;
        }
    }

    pub fn pop_min(&mut self) -> Option<(u32, f32)> {
        if self.open_count == 0 {
            return None;
        }
        let n_buckets = self.buckets.len();
        while self.min_bucket < n_buckets {
            let head = self.bucket_head(self.min_bucket);
            if head != u32::MAX {
                // Find minimum priority inside this bucket so A* optimality holds even on overflow
                let mut best = head;
                let mut best_p = self.priority[head as usize];
                let mut cur = self.next[head as usize];
                while cur != u32::MAX {
                    let cp = self.priority[cur as usize];
                    if cp < best_p {
                        best = cur;
                        best_p = cp;
                    }
                    cur = self.next[cur as usize];
                }
                self.unlink(best);
                if self.bucket_head(self.min_bucket) == u32::MAX {
                    self.min_bucket += 1;
                }
                return Some((best, best_p));
            }
            self.min_bucket += 1;
        }
        None
    }
}

/// Epoch-stamped dense per-vertex search state matching `CVertexManagerFixed` (`vertex_manager_fixed_inline.h`).
#[derive(Debug, Clone)]
pub struct FixedVertexStorage {
    epoch: u32,
    node_epoch: Vec<u32>,
    g: Vec<f32>,
    f: Vec<f32>,
    parent: Vec<u32>,
    closed: Vec<u8>,
    visited_count: u32,
}

impl Default for FixedVertexStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedVertexStorage {
    pub fn new() -> Self {
        Self {
            epoch: 1,
            node_epoch: Vec::new(),
            g: Vec::new(),
            f: Vec::new(),
            parent: Vec::new(),
            closed: Vec::new(),
            visited_count: 0,
        }
    }

    pub fn begin(&mut self, capacity: usize) {
        if self.node_epoch.len() < capacity {
            self.node_epoch.resize(capacity, 0);
            self.g.resize(capacity, f32::INFINITY);
            self.f.resize(capacity, f32::INFINITY);
            self.parent.resize(capacity, INVALID_AI_VERTEX);
            self.closed.resize(capacity, 0);
        }
        self.visited_count = 0;
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.node_epoch.fill(0);
            self.epoch = 1;
        }
    }

    #[inline]
    pub fn is_visited(&self, id: u32) -> bool {
        let i = id as usize;
        i < self.node_epoch.len() && self.node_epoch[i] == self.epoch
    }

    #[inline]
    pub fn is_closed(&self, id: u32) -> bool {
        let i = id as usize;
        i < self.node_epoch.len() && self.node_epoch[i] == self.epoch && self.closed[i] != 0
    }

    #[inline]
    pub fn visited_count(&self) -> u32 {
        self.visited_count
    }

    #[inline]
    pub fn g_score(&self, id: u32) -> f32 {
        let i = id as usize;
        if i < self.node_epoch.len() && self.node_epoch[i] == self.epoch {
            self.g[i]
        } else {
            f32::INFINITY
        }
    }

    pub fn open_start(&mut self, start: u32, f0: f32) {
        let i = start as usize;
        self.node_epoch[i] = self.epoch;
        self.g[i] = 0.0;
        self.f[i] = f0;
        self.parent[i] = INVALID_AI_VERTEX;
        self.closed[i] = 0;
        self.visited_count = 1;
    }

    #[inline]
    pub fn close(&mut self, id: u32) {
        self.closed[id as usize] = 1;
    }

    /// Relax edge `(id, new_parent)` with candidate `new_g` and `new_f`; returns `true` if improved.
    pub fn relax(&mut self, id: u32, new_parent: u32, new_g: f32, new_f: f32) -> bool {
        let i = id as usize;
        if self.node_epoch[i] != self.epoch {
            self.node_epoch[i] = self.epoch;
            self.g[i] = new_g;
            self.f[i] = new_f;
            self.parent[i] = new_parent;
            self.closed[i] = 0;
            self.visited_count += 1;
            true
        } else if new_g < self.g[i] {
            self.g[i] = new_g;
            self.f[i] = new_f;
            self.parent[i] = new_parent;
            true
        } else {
            false
        }
    }

    pub fn reconstruct_path(&self, start: u32, goal: u32) -> Vec<u32> {
        let mut path = vec![goal];
        let mut cur = goal;
        while cur != start {
            let p = self.parent[cur as usize];
            if p == INVALID_AI_VERTEX || p == cur {
                break;
            }
            path.push(p);
            cur = p;
        }
        path.reverse();
        path
    }
}

/// Fixed-capacity open-addressing hash vertex manager matching `CVertexManagerHashFixed`
/// (`vertex_manager_hash_fixed_inline.h`), indexing sparse/large graphs via epoch-stamped buckets.
#[derive(Debug, Clone)]
pub struct HashVertexStorage {
    buckets: Vec<u32>,
    bucket_epoch: Vec<u32>,
    vertex_ids: Vec<u32>,
    next_in_bucket: Vec<u32>,
    g: Vec<f32>,
    closed: Vec<u8>,
    parent_slot: Vec<u32>,
    epoch: u32,
}

impl HashVertexStorage {
    pub fn with_hash_buckets(bucket_count: usize) -> Self {
        let cap = bucket_count.next_power_of_two().max(64);
        Self {
            buckets: vec![u32::MAX; cap],
            bucket_epoch: vec![0; cap],
            vertex_ids: Vec::new(),
            next_in_bucket: Vec::new(),
            g: Vec::new(),
            closed: Vec::new(),
            parent_slot: Vec::new(),
            epoch: 1,
        }
    }

    pub fn clear(&mut self) {
        self.vertex_ids.clear();
        self.next_in_bucket.clear();
        self.g.clear();
        self.closed.clear();
        self.parent_slot.clear();
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.bucket_epoch.fill(0);
            self.epoch = 1;
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.vertex_ids.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertex_ids.is_empty()
    }

    #[inline]
    fn hash_bucket(&self, vertex_id: u32) -> usize {
        // Knuth multiplicative hash mod power-of-two bucket table
        ((vertex_id.wrapping_mul(0x9E37_79B1)) as usize) & (self.buckets.len() - 1)
    }

    pub fn find_slot(&self, vertex_id: u32) -> Option<u32> {
        let b = self.hash_bucket(vertex_id);
        if self.bucket_epoch[b] != self.epoch {
            return None;
        }
        let mut slot = self.buckets[b];
        while slot != u32::MAX {
            if self.vertex_ids[slot as usize] == vertex_id {
                return Some(slot);
            }
            slot = self.next_in_bucket[slot as usize];
        }
        None
    }

    pub fn insert_or_get_slot(&mut self, vertex_id: u32, init_g: f32, parent: u32) -> (u32, bool) {
        if let Some(slot) = self.find_slot(vertex_id) {
            return (slot, false);
        }
        let b = self.hash_bucket(vertex_id);
        let head = if self.bucket_epoch[b] == self.epoch {
            self.buckets[b]
        } else {
            self.bucket_epoch[b] = self.epoch;
            u32::MAX
        };
        let slot = self.vertex_ids.len() as u32;
        self.vertex_ids.push(vertex_id);
        self.next_in_bucket.push(head);
        self.g.push(init_g);
        self.closed.push(0);
        self.parent_slot.push(parent);
        self.buckets[b] = slot;
        (slot, true)
    }

    #[inline]
    pub fn vertex_id(&self, slot: u32) -> u32 {
        self.vertex_ids[slot as usize]
    }

    #[inline]
    pub fn g_score(&self, slot: u32) -> f32 {
        self.g[slot as usize]
    }

    #[inline]
    pub fn set_state(&mut self, slot: u32, g: f32, parent: u32) {
        self.g[slot as usize] = g;
        self.parent_slot[slot as usize] = parent;
    }

    #[inline]
    pub fn set_closed(&mut self, slot: u32) {
        self.closed[slot as usize] = 1;
    }

    #[inline]
    pub fn is_closed(&self, slot: u32) -> bool {
        self.closed[slot as usize] != 0
    }
}

/// Reusable zero-allocation workspace across A* and Dijkstra flood queries.
#[derive(Debug, Clone, Default)]
pub struct AiSearchWorkspace {
    pub heap: IndexedBinaryHeap,
    pub buckets: BucketPriorityQueue,
    pub vertices: FixedVertexStorage,
}

impl AiSearchWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    fn begin(&mut self, vertex_count: usize, params: &AiSearchParams) {
        self.vertices.begin(vertex_count);
        if params.use_bucket_queue {
            self.buckets.configure_range(0.0, params.bucket_max_value);
            self.buckets.clear();
        } else {
            self.heap.clear();
        }
    }

    #[inline]
    fn push_open(&mut self, id: u32, f: f32, params: &AiSearchParams) {
        if params.use_bucket_queue {
            self.buckets.push_or_decrease(id, f);
        } else {
            self.heap.push_or_decrease(id, f);
        }
    }

    #[inline]
    fn push_or_decrease_open(
        &mut self,
        id: u32,
        f: f32,
        _is_new: bool,
        params: &AiSearchParams,
    ) {
        self.push_open(id, f, params);
    }

    #[inline]
    fn pop_open(&mut self, params: &AiSearchParams) -> Option<(u32, f32)> {
        if params.use_bucket_queue {
            self.buckets.pop_min()
        } else {
            self.heap.pop_min()
        }
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

/// Check whether a 4-byte `GameGraph` location mask matches `vertex_types` (`255` = wildcard,
/// `CGameGraph::mask` in `game_graph_inline.h:70-82`).
#[inline]
pub fn vertex_type_matches_mask(mask: &[u8; 4], vertex_types: [u8; 4]) -> bool {
    for i in 0..4 {
        if mask[i] != vertex_types[i] && mask[i] != 255 {
            return false;
        }
    }
    true
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

    /// Encode header + cell array in raw unchunked layout (`game.graph` embedded format).
    pub fn to_raw_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(44 + self.cells.len() * 6);
        out.extend_from_slice(&self.header.version.to_le_bytes());
        out.extend_from_slice(&(self.cells.len() as u32).to_le_bytes());
        out.extend_from_slice(&u32::from(self.header.game_vertex_count).to_le_bytes());
        out.extend_from_slice(&self.header.level_guid[0].to_le_bytes());
        out.extend_from_slice(&self.header.level_guid[1].to_le_bytes());
        out.extend_from_slice(&self.header.game_guid[0].to_le_bytes());
        out.extend_from_slice(&self.header.game_guid[1].to_le_bytes());
        for cell in &self.cells {
            out.extend_from_slice(&cell.game_vertex_id.to_le_bytes());
            out.extend_from_slice(&cell.distance.to_le_bytes());
        }
        out
    }

    /// Encode this cross-table into standalone chunked `level.gct` format (Chunk `0`: 44B header,
    /// Chunk `1`: cell array matching `CGameLevelCrossTable`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut hdr = Vec::with_capacity(44);
        hdr.extend_from_slice(&self.header.version.to_le_bytes());
        hdr.extend_from_slice(&(self.cells.len() as u32).to_le_bytes());
        hdr.extend_from_slice(&u32::from(self.header.game_vertex_count).to_le_bytes());
        hdr.extend_from_slice(&self.header.level_guid[0].to_le_bytes());
        hdr.extend_from_slice(&self.header.level_guid[1].to_le_bytes());
        hdr.extend_from_slice(&self.header.game_guid[0].to_le_bytes());
        hdr.extend_from_slice(&self.header.game_guid[1].to_le_bytes());

        let mut cells_buf = Vec::with_capacity(self.cells.len() * 6);
        for cell in &self.cells {
            cells_buf.extend_from_slice(&cell.game_vertex_id.to_le_bytes());
            cells_buf.extend_from_slice(&cell.distance.to_le_bytes());
        }

        let mut cw = ChunkWriter::new();
        cw.write_chunk(0, &hdr);
        cw.write_chunk(1, &cells_buf);
        cw.into_bytes()
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

    /// Shortest path between two `GameGraph` vertices filtered by allowed `vertex_types` masks
    /// matching `CGameGraph::mask` (`game_graph_inline.h:70-82`, `255` = wildcard byte).
    pub fn find_game_path_filtered(
        &self,
        start: u16,
        goal: u16,
        allowed_type_masks: &[[u8; 4]],
    ) -> Option<(Vec<u16>, f32)> {
        let count = self.vertices.len();
        if start as usize >= count || goal as usize >= count {
            return None;
        }
        let is_allowed = |vid: u16| -> bool {
            if allowed_type_masks.is_empty() {
                return true;
            }
            let vt = self.vertices[vid as usize].vertex_types;
            allowed_type_masks
                .iter()
                .any(|m| vertex_type_matches_mask(m, vt))
        };
        let start_accessible = is_allowed(start);
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
                if next as usize >= count || (start_accessible && !is_allowed(next)) {
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

    /// Resolve the nearest `GameVertex` ID and distance for `(level_id, level_vertex_id)` using
    /// the embedded `LevelCrossTable`.
    #[inline]
    pub fn resolve_game_vertex_from_level(
        &self,
        level_id: u8,
        level_vertex_id: u32,
    ) -> Option<(u16, f32)> {
        let ct = self.cross_table_for_level(level_id)?;
        let cell = ct.game_vertex_for_level_vertex(level_vertex_id)?;
        Some((cell.game_vertex_id, cell.distance))
    }

    /// Encode this `GameGraph` into binary `game.graph` / `all.spawn` Chunk 4 wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.header.version);
        out.extend_from_slice(&(self.vertices.len() as u16).to_le_bytes());
        out.extend_from_slice(&(self.edges.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.death_points.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.header.guid[0].to_le_bytes());
        out.extend_from_slice(&self.header.guid[1].to_le_bytes());
        out.push(self.header.levels.len() as u8);

        for lvl in &self.header.levels {
            out.extend_from_slice(lvl.name.as_bytes());
            out.push(0);
            for f in lvl.offset {
                out.extend_from_slice(&f.to_le_bytes());
            }
            out.push(lvl.id);
            out.extend_from_slice(lvl.section.as_bytes());
            out.push(0);
            out.extend_from_slice(&lvl.guid[0].to_le_bytes());
            out.extend_from_slice(&lvl.guid[1].to_le_bytes());
        }

        for v in &self.vertices {
            for f in v.local_point {
                out.extend_from_slice(&f.to_le_bytes());
            }
            for f in v.global_point {
                out.extend_from_slice(&f.to_le_bytes());
            }
            let packed = u32::from(v.level_id) | (v.level_vertex_id << 8);
            out.extend_from_slice(&packed.to_le_bytes());
            out.extend_from_slice(&v.vertex_types);
            out.extend_from_slice(&v.edge_offset.to_le_bytes());
            out.extend_from_slice(&v.point_offset.to_le_bytes());
            out.push(v.neighbour_count);
            out.push(v.death_point_count);
        }

        for e in &self.edges {
            out.extend_from_slice(&e.target_vertex_id.to_le_bytes());
            out.extend_from_slice(&e.distance.to_le_bytes());
        }

        for dp in &self.death_points {
            for f in dp.point {
                out.extend_from_slice(&f.to_le_bytes());
            }
            out.extend_from_slice(&dp.level_vertex_id.to_le_bytes());
            out.extend_from_slice(&dp.distance.to_le_bytes());
        }

        let mut sorted_ct: Vec<&(u8, LevelCrossTable)> = self.cross_tables.iter().collect();
        sorted_ct.sort_by_key(|(id, _)| *id);
        for (_, ct) in sorted_ct {
            let raw = ct.to_raw_bytes();
            let block_len = (4 + raw.len()) as u32;
            out.extend_from_slice(&block_len.to_le_bytes());
            out.extend_from_slice(&raw);
        }

        out
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

        // Standalone chunked level.gct roundtrip & resolve helper
        let gct_re = LevelCrossTable::parse(&gct.to_bytes()).unwrap();
        assert_eq!(gct_re, gct);
        assert_eq!(parsed.resolve_game_vertex_from_level(1, 1), Some((2, 4.5)));
    }

    #[test]
    fn ai_queues_codecs_workspace_flood_cover_straight_and_filtered_game_graph() {
        // 1) Binary roundtrip for v10 & v13 level.ai + 4-node diamond/obstacle graph
        // Grid: cell_size=1.0, min=(0,0,0), max=(2,5,2) -> row_length = floor(2+0.001+1.5) = 3
        // Nodes: 0@(0,0), 1@(1,0) [xz=3], 2@(2,0) [xz=6], 3@(0,1) [xz=1], 4@(1,1) [xz=4], 5@(2,1) [xz=7]
        let make_links = |links: [Option<u32>; 4]| -> [u32; 4] {
            links.map(|o| o.unwrap_or(INVALID_AI_VERTEX))
        };
        let plane_up = pv_compress_normal([0.0, 1.0, 0.0]);
        let header = AiGraphHeader {
            version: XRAI_CURRENT_VERSION,
            vertex_count: 6,
            cell_size: 1.0,
            factor_y: 5.0,
            box_aabb: OgfBBox {
                min: [0.0, 0.0, 0.0],
                max: [2.0, 5.0, 2.0],
            },
            guid: [0x11, 0x22],
            row_length: 3,
            column_length: 3,
        };
        let vertices = vec![
            // dir 0:-X, 1:+Z, 2:+X, 3:-Z (per CLevelGraph::create_straight_path switch & debug2)
            AiVertex {
                links: make_links([None, Some(3), Some(1), None]),
                cover_high: 0x1234,
                cover_low: 0x0F0F, // zero high/low on exposed vs covered sides
                plane: plane_up,
                xz: 0,
                y: 0,
            },
            AiVertex {
                links: make_links([Some(0), Some(4), Some(2), None]),
                cover_high: 0x0000, // node 1 exposed
                cover_low: 0x0000,
                plane: plane_up,
                xz: 3,
                y: 0,
            },
            AiVertex {
                links: make_links([Some(1), Some(5), None, None]),
                cover_high: 0xFFFF,
                cover_low: 0xFFFF,
                plane: plane_up,
                xz: 6,
                y: 0,
            },
            AiVertex {
                links: make_links([None, None, Some(4), Some(0)]),
                cover_high: 0xFFFF, // row 1 heavily covered
                cover_low: 0xFFFF,
                plane: plane_up,
                xz: 1,
                y: 0,
            },
            AiVertex {
                links: make_links([Some(3), None, Some(5), Some(1)]),
                cover_high: 0xFFFF,
                cover_low: 0xFFFF,
                plane: plane_up,
                xz: 4,
                y: 0,
            },
            AiVertex {
                links: make_links([Some(4), None, None, Some(2)]),
                cover_high: 0xFFFF,
                cover_low: 0xFFFF,
                plane: plane_up,
                xz: 7,
                y: 0,
            },
        ];

        let lg10 = LevelGraph {
            header,
            vertices: vertices.clone(),
        };
        let raw10 = lg10.to_bytes().unwrap();
        let parsed10 = LevelGraph::parse(&raw10).unwrap();
        assert_eq!(parsed10, lg10);

        let mut lg13 = lg10.clone();
        lg13.header.version = XRAI_LARGE_VERSION;
        lg13.vertices[0].xz = 0x0100_0000;
        let raw13 = lg13.to_bytes().unwrap();
        let parsed13 = LevelGraph::parse(&raw13).unwrap();
        assert_eq!(parsed13, lg13);

        // 2) IndexedBinaryHeap & BucketPriorityQueue optimality + decrease_key + HashVertexStorage
        let mut heap = IndexedBinaryHeap::new();
        heap.push_or_decrease(5, 10.0);
        heap.push_or_decrease(2, 12.0);
        heap.push_or_decrease(2, 4.0);
        assert_eq!(heap.pop_min(), Some((2, 4.0)));
        assert_eq!(heap.pop_min(), Some((5, 10.0)));
        assert_eq!(heap.pop_min(), None);

        let mut bq = BucketPriorityQueue::new(64, 0.0, 50.0);
        bq.push_or_decrease(3, 120.0); // overflows to last bucket
        bq.push_or_decrease(1, 75.0); // also last bucket (must still yield min = 75.0)
        bq.push_or_decrease(1, 5.0);
        assert_eq!(bq.pop_min(), Some((1, 5.0)));
        assert_eq!(bq.pop_min(), Some((3, 120.0)));
        bq.clear();
        assert!(bq.is_empty());

        let mut hvs = HashVertexStorage::with_hash_buckets(64);
        let (s0, n0) = hvs.insert_or_get_slot(123_456, 1.5, u32::MAX);
        assert!(n0);
        let (s0_again, n0_again) = hvs.insert_or_get_slot(123_456, 0.0, 0);
        assert!(!n0_again && s0 == s0_again && hvs.vertex_id(s0) == 123_456);

        // 3) Workspace A* (both binary heap & bucket queue) + access mask + cover weighting
        let mut ws = AiSearchWorkspace::new();
        let heap_res = lg10
            .find_path_astar_with_workspace(0, 2, &AiSearchParams::default(), None, &mut ws)
            .unwrap();
        assert_eq!(heap_res.path, vec![0, 1, 2]);
        assert!((heap_res.total_cost - 2.0).abs() < 1e-5);

        let bq_params = AiSearchParams {
            use_bucket_queue: true,
            ..AiSearchParams::default()
        };
        let bq_res = lg10
            .find_path_astar_with_workspace(0, 2, &bq_params, None, &mut ws)
            .unwrap();
        assert_eq!(bq_res.path, vec![0, 1, 2]);

        // Block node 1 via LevelAccessMask -> detours 0 -> 3 -> 4 -> 5 -> 2
        let mut mask = LevelAccessMask::all_accessible(lg10.vertices.len());
        mask.set_accessible(1, false);
        let masked = lg10
            .find_path_astar_with_workspace(0, 2, &AiSearchParams::default(), Some(&mask), &mut ws)
            .unwrap();
        assert_eq!(masked.path, vec![0, 3, 4, 5, 2]);

        // Cover-weighted A* avoids exposed node 1 when high cover weight is applied
        let covered = lg10
            .find_path_astar_cover_weighted(
                0,
                2,
                [1.0, 0.0, -5.0],
                5.0,
                true,
                &AiSearchParams::default(),
                None,
                &mut ws,
            )
            .unwrap();
        assert_eq!(covered.path, vec![0, 3, 4, 5, 2]);

        // Flood & nearest accessible predicate
        let flooded = lg10.flood_level(
            0,
            &AiSearchParams {
                max_range: 1.5,
                ..AiSearchParams::default()
            },
            None,
            &mut ws,
        );
        assert!(flooded.iter().any(|&(id, _)| id == 1));
        assert!(!flooded.iter().any(|&(id, _)| id == 2));

        let nearest = lg10
            .find_nearest_accessible_vertex(
                0,
                &AiSearchParams::default(),
                None,
                &mut ws,
                |id, v| id != 0 && v.cover(0).0 == 15,
            )
            .unwrap();
        assert_eq!(nearest.0, 3);

        // Straight-line XZ check & path trace from 0 to 2 along +X
        let straight = lg10
            .check_position_in_direction_2d(0, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], None)
            .unwrap();
        assert_eq!(straight.0, 2);
        let (_snodes, spts) = lg10
            .create_straight_path_2d(0, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], true)
            .unwrap();
        assert!(spts.len() >= 2);

        // 4) GameGraph vertex_types mask filtering (`255` = wildcard)
        let gg = GameGraph {
            header: GameGraphHeader {
                version: 10,
                vertex_count: 3,
                edge_count: 3,
                death_point_count: 0,
                guid: [0, 0],
                levels: Vec::new(),
            },
            vertices: vec![
                GameVertex {
                    local_point: [0.0; 3],
                    global_point: [0.0, 0.0, 0.0],
                    level_id: 1,
                    level_vertex_id: 0,
                    vertex_types: [10, 0, 0, 0],
                    edge_offset: 3 * 42,
                    point_offset: 3 * 42 + 3 * 6,
                    neighbour_count: 2,
                    death_point_count: 0,
                },
                GameVertex {
                    local_point: [5.0, 0.0, 0.0],
                    global_point: [5.0, 0.0, 0.0],
                    level_id: 1,
                    level_vertex_id: 1,
                    vertex_types: [99, 0, 0, 0], // blocked unless type 99 allowed
                    edge_offset: 3 * 42 + 2 * 6,
                    point_offset: 3 * 42 + 3 * 6,
                    neighbour_count: 1,
                    death_point_count: 0,
                },
                GameVertex {
                    local_point: [10.0, 0.0, 0.0],
                    global_point: [10.0, 0.0, 0.0],
                    level_id: 1,
                    level_vertex_id: 2,
                    vertex_types: [10, 0, 0, 0],
                    edge_offset: 3 * 42 + 3 * 6,
                    point_offset: 3 * 42 + 3 * 6,
                    neighbour_count: 0,
                    death_point_count: 0,
                },
            ],
            edges: vec![
                GameEdge {
                    target_vertex_id: 1,
                    distance: 2.0,
                },
                GameEdge {
                    target_vertex_id: 2,
                    distance: 15.0,
                },
                GameEdge {
                    target_vertex_id: 2,
                    distance: 2.0,
                },
            ],
            death_points: Vec::new(),
            cross_tables: Vec::new(),
        };
        let (p_direct, d_direct) = gg.find_game_path_filtered(0, 2, &[[10, 255, 255, 255]]).unwrap();
        assert_eq!(p_direct, vec![0, 2]);
        assert!((d_direct - 15.0).abs() < 1e-5);
        let (p_all, d_all) = gg.find_game_path_filtered(0, 2, &[[255, 255, 255, 255]]).unwrap();
        assert_eq!(p_all, vec![0, 1, 2]);
        assert!((d_all - 4.0).abs() < 1e-5);
    }
}
