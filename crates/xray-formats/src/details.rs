//! Pure Rust X-Ray `level.details` & `.dm` Flora/Grass Detail Parser
//!
//! Faithful pure-Rust implementation matching `src/Layers/xrRender/DetailFormat.h`,
//! `src/Layers/xrRender/DetailModel.cpp`, `DetailManager.cpp`, and `DetailManager_cache.cpp`.
//! Supports both legacy `DETAIL_VERSION_3` (16-byte slots, 6-bit IDs) and Anomaly
//! `DETAIL_VERSION_4` (20-byte slots, 14-bit object IDs).

use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::ogf::{matrix_transform_point, OgfBBox, XRayMatrix};

pub const DETAIL_VERSION_3: u32 = 3;
pub const DETAIL_VERSION_4: u32 = 4;
pub const DETAIL_SLOT_SIZE: f32 = 2.0;
pub const DETAIL_SLOT_HALF_SIZE: f32 = 1.0;
pub const DETAIL_ID_EMPTY_V3: u16 = 0x003F;
pub const DETAIL_ID_EMPTY: u16 = 0x3FFF;
pub const DO_NO_WAVING: u32 = 0x0001;

/// `CDetail::fvfVertexIn` (20 bytes per vertex).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailVertex {
    pub position: [f32; 3],
    pub u: f32,
    pub v: f32,
}

/// Transformed detail vertex emitted by `DetailMesh::transfer` (`CDetail::fvfVertexOut`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailMeshVertexOut {
    pub position: [f32; 3],
    pub color_rgba: [f32; 4],
    pub uv: [f32; 2],
}

/// Individual grass/flora mesh loaded from `.dm` or `level.details` chunk 1 (`CDetail`).
#[derive(Debug, Clone, PartialEq)]
pub struct DetailMesh {
    pub shader: String,
    pub texture: String,
    pub flags: u32,
    pub min_scale: f32,
    pub max_scale: f32,
    pub vertices: Vec<DetailVertex>,
    pub indices: Vec<u16>,
}

impl DetailMesh {
    #[inline]
    pub fn no_waving(&self) -> bool {
        (self.flags & DO_NO_WAVING) != 0
    }

    /// Parse a binary `.dm` detail mesh payload (`CDetail::Load` in `DetailModel.cpp:82-119`).
    pub fn parse_dm(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let shader = cur.read_zstring()?;
        let texture = cur.read_zstring()?;
        let flags = cur.read_u32()?;
        let min_scale = cur.read_f32()?;
        let max_scale = cur.read_f32()?;
        let number_vertices = cur.read_u32()? as usize;
        let number_indices = cur.read_u32()? as usize;

        if !number_indices.is_multiple_of(3) {
            return Err(FormatError::InvalidValue(
                "detail mesh index count must be divisible by 3",
            ));
        }

        let vert_bytes = number_vertices
            .checked_mul(20)
            .ok_or(FormatError::UnexpectedEof)?;
        let idx_bytes = number_indices
            .checked_mul(2)
            .ok_or(FormatError::UnexpectedEof)?;

        let raw_verts = cur.read_bytes(vert_bytes)?;
        let raw_indices = cur.read_bytes(idx_bytes)?;

        let mut vertices = Vec::with_capacity(number_vertices);
        for rec in raw_verts.as_chunks::<20>().0 {
            let mut vc = ByteCursor::new(rec);
            vertices.push(DetailVertex {
                position: vc.read_f32_3()?,
                u: vc.read_f32()?,
                v: vc.read_f32()?,
            });
        }

        let mut indices = Vec::with_capacity(number_indices);
        for pair in raw_indices.as_chunks::<2>().0 {
            let idx = u16::from_le_bytes(*pair);
            if (idx as usize) >= number_vertices {
                return Err(FormatError::InvalidValue(
                    "detail mesh index references out-of-range vertex",
                ));
            }
            indices.push(idx);
        }

        Ok(Self {
            shader,
            texture,
            flags,
            min_scale,
            max_scale,
            vertices,
            indices,
        })
    }

    pub fn bbox(&self) -> Option<OgfBBox> {
        let first = self.vertices.first()?;
        let mut min = first.position;
        let mut max = first.position;
        for v in &self.vertices[1..] {
            for axis in 0..3 {
                min[axis] = min[axis].min(v.position[axis]);
                max[axis] = max[axis].max(v.position[axis]);
            }
        }
        Some(OgfBBox { min, max })
    }

    /// Bounding sphere `(center, radius)` derived from `bbox()` via `Fbox::getsphere`
    /// (`DetailModel.cpp:118`, `_fbox.h:247-252`).
    pub fn bounding_sphere(&self) -> Option<([f32; 3], f32)> {
        let bb = self.bbox()?;
        let center = [
            (bb.min[0] + bb.max[0]) * 0.5,
            (bb.min[1] + bb.max[1]) * 0.5,
            (bb.min[2] + bb.max[2]) * 0.5,
        ];
        let dx = bb.max[0] - center[0];
        let dy = bb.max[1] - center[1];
        let dz = bb.max[2] - center[2];
        Some((center, (dx * dx + dy * dy + dz * dz).sqrt()))
    }

    /// Transform vertices by `xform`, tint with `color_rgba`, offset UVs by `(du, dv)`, and shift
    /// indices by `index_offset` (`CDetail::transfer` in `DetailModel.cpp:24-80`).
    ///
    /// Returns `None` if adding `index_offset` would overflow `u16`.
    pub fn transfer(
        &self,
        xform: &XRayMatrix,
        color_rgba: [f32; 4],
        du: f32,
        dv: f32,
        index_offset: u16,
    ) -> Option<(Vec<DetailMeshVertexOut>, Vec<u16>)> {
        let max_idx = usize::from(index_offset).checked_add(self.vertices.len())?;
        if max_idx > usize::from(u16::MAX) + 1 {
            return None;
        }
        let mut out_verts = Vec::with_capacity(self.vertices.len());
        for v in &self.vertices {
            out_verts.push(DetailMeshVertexOut {
                position: matrix_transform_point(xform, v.position),
                color_rgba,
                uv: [v.u + du, v.v + dv],
            });
        }
        let mut out_indices = Vec::with_capacity(self.indices.len());
        for &idx in &self.indices {
            out_indices.push(idx.checked_add(index_offset)?);
        }
        Some((out_verts, out_indices))
    }
}

/// 24-byte `struct DetailHeader` (`DetailFormat.h:31-37`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetailHeader {
    pub version: u32,
    pub object_count: u32,
    pub offs_x: i32,
    pub offs_z: i32,
    pub size_x: u32,
    pub size_z: u32,
}

/// 4x4-bit corner alpha palette (`struct DetailPalette`, 2 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DetailPalette {
    pub a0: u8,
    pub a1: u8,
    pub a2: u8,
    pub a3: u8,
}

impl DetailPalette {
    #[inline]
    pub fn from_u16(raw: u16) -> Self {
        Self {
            a0: (raw & 0xF) as u8,
            a1: ((raw >> 4) & 0xF) as u8,
            a2: ((raw >> 8) & 0xF) as u8,
            a3: ((raw >> 12) & 0xF) as u8,
        }
    }

    #[inline]
    pub fn to_u16(self) -> u16 {
        u16::from(self.a0 & 0xF)
            | (u16::from(self.a1 & 0xF) << 4)
            | (u16::from(self.a2 & 0xF) << 8)
            | (u16::from(self.a3 & 0xF) << 12)
    }

    #[inline]
    pub fn alpha_f32(self) -> [f32; 4] {
        [
            f32::from(self.a0) / 15.0,
            f32::from(self.a1) / 15.0,
            f32::from(self.a2) / 15.0,
            f32::from(self.a3) / 15.0,
        ]
    }
}

/// Unpacked working `DetailSlot` supporting both `v3` expansion and `v4` 14-bit object IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetailSlot {
    pub ids: [u16; 4],
    pub c_dir: u8,
    pub c_hemi: u8,
    pub y_base: u16,
    pub y_height: u8,
    pub c_r: u8,
    pub c_g: u8,
    pub c_b: u8,
    pub palette: [DetailPalette; 4],
}

impl Default for DetailSlot {
    fn default() -> Self {
        Self {
            ids: [DETAIL_ID_EMPTY; 4],
            c_dir: 0,
            c_hemi: 0,
            y_base: 0,
            y_height: 0,
            c_r: 0,
            c_g: 0,
            c_b: 0,
            palette: [DetailPalette::default(); 4],
        }
    }
}

impl DetailSlot {
    /// Decode 20-byte `DETAIL_VERSION_4` slot (`DetailFormat.h:67-86`).
    pub fn from_v4_bytes(bytes: &[u8; 20]) -> Self {
        let w0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let w1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let w2 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

        let id0 = (w0 & 0x3FFF) as u16;
        let id1 = ((w0 >> 14) & 0x3FFF) as u16;
        let c_dir = ((w0 >> 28) & 0xF) as u8;

        let id2 = (w1 & 0x3FFF) as u16;
        let id3 = ((w1 >> 14) & 0x3FFF) as u16;
        let c_hemi = ((w1 >> 28) & 0xF) as u8;

        let y_base = (w2 & 0xFFF) as u16;
        let y_height = ((w2 >> 12) & 0xFF) as u8;
        let c_r = ((w2 >> 20) & 0xF) as u8;
        let c_g = ((w2 >> 24) & 0xF) as u8;
        let c_b = ((w2 >> 28) & 0xF) as u8;

        let palette = [
            DetailPalette::from_u16(u16::from_le_bytes([bytes[12], bytes[13]])),
            DetailPalette::from_u16(u16::from_le_bytes([bytes[14], bytes[15]])),
            DetailPalette::from_u16(u16::from_le_bytes([bytes[16], bytes[17]])),
            DetailPalette::from_u16(u16::from_le_bytes([bytes[18], bytes[19]])),
        ];

        Self {
            ids: [id0, id1, id2, id3],
            c_dir,
            c_hemi,
            y_base,
            y_height,
            c_r,
            c_g,
            c_b,
            palette,
        }
    }

    /// Encode into 20-byte `DETAIL_VERSION_4` wire layout.
    pub fn to_v4_bytes(self) -> [u8; 20] {
        let w0 = (u32::from(self.ids[0]) & 0x3FFF)
            | ((u32::from(self.ids[1]) & 0x3FFF) << 14)
            | (u32::from(self.c_dir & 0xF) << 28);
        let w1 = (u32::from(self.ids[2]) & 0x3FFF)
            | ((u32::from(self.ids[3]) & 0x3FFF) << 14)
            | (u32::from(self.c_hemi & 0xF) << 28);
        let w2 = (u32::from(self.y_base) & 0xFFF)
            | (u32::from(self.y_height) << 12)
            | (u32::from(self.c_r & 0xF) << 20)
            | (u32::from(self.c_g & 0xF) << 24)
            | (u32::from(self.c_b & 0xF) << 28);

        let mut out = [0u8; 20];
        out[0..4].copy_from_slice(&w0.to_le_bytes());
        out[4..8].copy_from_slice(&w1.to_le_bytes());
        out[8..12].copy_from_slice(&w2.to_le_bytes());
        for (i, p) in self.palette.into_iter().enumerate() {
            out[12 + i * 2..14 + i * 2].copy_from_slice(&p.to_u16().to_le_bytes());
        }
        out
    }

    /// Decode and expand 16-byte `DETAIL_VERSION_3` slot (`expand_v3` in `DetailFormat.h:159-167`).
    pub fn from_v3_bytes(bytes: &[u8; 16]) -> Self {
        let w0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let w1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        let y_base = (w0 & 0xFFF) as u16;
        let y_height = ((w0 >> 12) & 0xFF) as u8;
        let raw_id0 = ((w0 >> 20) & 0x3F) as u16;
        let raw_id1 = ((w0 >> 26) & 0x3F) as u16;

        let raw_id2 = (w1 & 0x3F) as u16;
        let raw_id3 = ((w1 >> 6) & 0x3F) as u16;
        let c_dir = ((w1 >> 12) & 0xF) as u8;
        let c_hemi = ((w1 >> 16) & 0xF) as u8;
        let c_r = ((w1 >> 20) & 0xF) as u8;
        let c_g = ((w1 >> 24) & 0xF) as u8;
        let c_b = ((w1 >> 28) & 0xF) as u8;

        let remap = |v: u16| if v == DETAIL_ID_EMPTY_V3 { DETAIL_ID_EMPTY } else { v };

        let palette = [
            DetailPalette::from_u16(u16::from_le_bytes([bytes[8], bytes[9]])),
            DetailPalette::from_u16(u16::from_le_bytes([bytes[10], bytes[11]])),
            DetailPalette::from_u16(u16::from_le_bytes([bytes[12], bytes[13]])),
            DetailPalette::from_u16(u16::from_le_bytes([bytes[14], bytes[15]])),
        ];

        Self {
            ids: [remap(raw_id0), remap(raw_id1), remap(raw_id2), remap(raw_id3)],
            c_dir,
            c_hemi,
            y_base,
            y_height,
            c_r,
            c_g,
            c_b,
            palette,
        }
    }

    #[inline]
    pub fn y_base_meters(&self) -> f32 {
        f32::from(self.y_base) * 0.2 - 200.0
    }

    #[inline]
    pub fn y_height_meters(&self) -> f32 {
        f32::from(self.y_height) * 0.1
    }

    /// Pack base elevation & height in meters using `DetailSlot::w_y` quantization.
    pub fn pack_y(&mut self, base_m: f32, height_m: f32) {
        let base_q = ((base_m + 200.0) / 0.2).floor().clamp(0.0, 4095.0) as u16;
        self.y_base = base_q;
        let error = base_m - self.y_base_meters();
        let height_q = ((height_m + error) / 0.1).ceil().clamp(0.0, 255.0) as u8;
        self.y_height = height_q;
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.iter().all(|&id| id == DETAIL_ID_EMPTY)
    }

    #[inline]
    pub fn dir_light(&self) -> f32 {
        f32::from(self.c_dir) / 15.0
    }

    #[inline]
    pub fn hemi_light(&self) -> f32 {
        f32::from(self.c_hemi) / 15.0
    }

    #[inline]
    pub fn rgb_light(&self) -> [f32; 3] {
        [
            f32::from(self.c_r) / 15.0,
            f32::from(self.c_g) / 15.0,
            f32::from(self.c_b) / 15.0,
        ]
    }
}

/// Complete parsed `level.details` database (`CDetailManager` backing store).
#[derive(Debug, Clone, PartialEq)]
pub struct LevelDetails {
    pub header: DetailHeader,
    pub objects: Vec<DetailMesh>,
    pub slots: Vec<DetailSlot>,
}

impl LevelDetails {
    /// Parse binary `level.details` (`CDetailManager::Load` in `DetailManager.cpp:197-229`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);

        let hdr_bytes = reader.require_chunk(0)?;
        let mut hc = ByteCursor::new(&hdr_bytes);
        let version = hc.read_u32()?;
        if version != DETAIL_VERSION_3 && version != DETAIL_VERSION_4 {
            return Err(FormatError::InvalidValue(
                "unsupported level.details version (expected 3 or 4)",
            ));
        }
        let header = DetailHeader {
            version,
            object_count: hc.read_u32()?,
            offs_x: hc.read_i32()?,
            offs_z: hc.read_i32()?,
            size_x: hc.read_u32()?,
            size_z: hc.read_u32()?,
        };

        let objs_bytes = reader.require_chunk(1)?;
        let obj_reader = ChunkReader::new(&objs_bytes);
        let mut objects = Vec::with_capacity(header.object_count as usize);
        for m_id in 0..header.object_count {
            let sub = obj_reader.require_chunk(m_id)?;
            objects.push(DetailMesh::parse_dm(&sub)?);
        }

        let slot_count = (header.size_x as usize)
            .checked_mul(header.size_z as usize)
            .ok_or(FormatError::UnexpectedEof)?;

        let slots_bytes = reader.require_chunk(2)?;
        let mut slots = Vec::with_capacity(slot_count);
        if version == DETAIL_VERSION_4 {
            let need = slot_count
                .checked_mul(20)
                .ok_or(FormatError::UnexpectedEof)?;
            if slots_bytes.len() < need {
                return Err(FormatError::UnexpectedEof);
            }
            for rec in slots_bytes[..need].as_chunks::<20>().0 {
                slots.push(DetailSlot::from_v4_bytes(rec));
            }
        } else {
            let need = slot_count
                .checked_mul(16)
                .ok_or(FormatError::UnexpectedEof)?;
            if slots_bytes.len() < need {
                return Err(FormatError::UnexpectedEof);
            }
            for rec in slots_bytes[..need].as_chunks::<16>().0 {
                slots.push(DetailSlot::from_v3_bytes(rec));
            }
        }

        Ok(Self {
            header,
            objects,
            slots,
        })
    }

    /// Direct grid lookup by zero-based database coordinates `(db_x, db_z)`.
    #[inline]
    pub fn slot_at(&self, db_x: u32, db_z: u32) -> Option<&DetailSlot> {
        if db_x < self.header.size_x && db_z < self.header.size_z {
            let idx = (db_z as usize) * (self.header.size_x as usize) + (db_x as usize);
            self.slots.get(idx)
        } else {
            None
        }
    }

    /// Query slot using signed world slot coordinates `(sx, sz)` (`CDetailManager::QueryDB`).
    #[inline]
    pub fn query_slot(&self, sx: i32, sz: i32) -> Option<&DetailSlot> {
        let db_x = sx.checked_add(self.header.offs_x)?;
        let db_z = sz.checked_add(self.header.offs_z)?;
        if db_x >= 0 && db_z >= 0 {
            self.slot_at(db_x as u32, db_z as u32)
        } else {
            None
        }
    }

    /// Convert world `(x, z)` coordinates to signed slot coordinates `(sx, sz)`.
    #[inline]
    pub fn world_to_slot_coords(world_x: f32, world_z: f32) -> (i32, i32) {
        (
            (world_x / DETAIL_SLOT_SIZE).floor() as i32,
            (world_z / DETAIL_SLOT_SIZE).floor() as i32,
        )
    }

    /// World-space bounding box of slot `(sx, sz)` matching `DetailManager_cache.cpp:59-60`.
    pub fn slot_world_bounds(&self, sx: i32, sz: i32) -> Option<OgfBBox> {
        let slot = self.query_slot(sx, sz)?;
        let x0 = (sx as f32) * DETAIL_SLOT_SIZE;
        let z0 = (sz as f32) * DETAIL_SLOT_SIZE;
        let y0 = slot.y_base_meters();
        Some(OgfBBox {
            min: [x0, y0, z0],
            max: [
                x0 + DETAIL_SLOT_SIZE,
                y0 + slot.y_height_meters(),
                z0 + DETAIL_SLOT_SIZE,
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;

    fn build_dm_bytes(shader: &str, texture: &str, flags: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(shader.as_bytes());
        out.push(0);
        out.extend_from_slice(texture.as_bytes());
        out.push(0);
        out.extend_from_slice(&flags.to_le_bytes());
        out.extend_from_slice(&0.6f32.to_le_bytes()); // min_scale
        out.extend_from_slice(&1.4f32.to_le_bytes()); // max_scale
        out.extend_from_slice(&3u32.to_le_bytes()); // 3 vertices
        out.extend_from_slice(&3u32.to_le_bytes()); // 3 indices
        for (p, uv) in [
            ([0.0f32, 0.0, 0.0], [0.0f32, 1.0]),
            ([0.2, 0.5, 0.0], [0.5, 0.0]),
            ([0.4, 0.0, 0.0], [1.0, 1.0]),
        ] {
            for f in p {
                out.extend_from_slice(&f.to_le_bytes());
            }
            out.extend_from_slice(&uv[0].to_le_bytes());
            out.extend_from_slice(&uv[1].to_le_bytes());
        }
        for i in [0u16, 1, 2] {
            out.extend_from_slice(&i.to_le_bytes());
        }
        out
    }

    #[test]
    fn test_dm_and_level_details_v4_14bit_slots() {
        let dm0 = build_dm_bytes("details\\blend", "detail\\detail_grnd_grass", 0);
        let dm1 = build_dm_bytes("details\\still", "detail\\detail_leaves", DO_NO_WAVING);
        let parsed_dm = DetailMesh::parse_dm(&dm1).unwrap();
        assert!(parsed_dm.no_waving());
        assert_eq!(parsed_dm.shader, "details\\still");
        assert_eq!(parsed_dm.texture, "detail\\detail_leaves");
        assert_eq!(parsed_dm.indices, vec![0, 1, 2]);

        let mut objs = ChunkWriter::new();
        objs.write_chunk(0, &dm0);
        objs.write_chunk(1, &dm1);

        let mut hdr = Vec::new();
        for v in [DETAIL_VERSION_4, 2] {
            hdr.extend_from_slice(&v.to_le_bytes());
        }
        hdr.extend_from_slice(&(-4i32).to_le_bytes()); // offs_x = -4
        hdr.extend_from_slice(&(-2i32).to_le_bytes()); // offs_z = -2
        hdr.extend_from_slice(&2u32.to_le_bytes()); // size_x = 2
        hdr.extend_from_slice(&1u32.to_le_bytes()); // size_z = 1

        // Slot (0,0): empty
        let s0 = DetailSlot::default();
        // Slot (1,0): 14-bit object IDs > 63 to verify v4 expansion beyond v3 limit
        let mut s1 = DetailSlot {
            ids: [1234, 9876, DETAIL_ID_EMPTY, 42],
            c_dir: 15,
            c_hemi: 6,
            y_base: 0,
            y_height: 0,
            c_r: 3,
            c_g: 12,
            c_b: 9,
            palette: [
                DetailPalette { a0: 1, a1: 2, a2: 14, a3: 15 },
                DetailPalette::default(),
                DetailPalette::default(),
                DetailPalette::default(),
            ],
        };
        s1.pack_y(12.4, 3.2);

        let mut slots_raw = Vec::new();
        slots_raw.extend_from_slice(&s0.to_v4_bytes());
        slots_raw.extend_from_slice(&s1.to_v4_bytes());

        let mut root = ChunkWriter::new();
        root.write_chunk(0, &hdr);
        root.write_chunk(1, &objs.into_bytes());
        root.write_chunk(2, &slots_raw);

        let details = LevelDetails::parse(&root.into_bytes()).unwrap();
        assert_eq!(details.header.version, DETAIL_VERSION_4);
        assert_eq!(details.objects.len(), 2);
        assert_eq!(details.slots.len(), 2);

        // World slot sx=5 -> db_x = 5 + (-4) = 1; sz=2 -> db_z = 2 + (-2) = 0
        assert!(details.query_slot(4, 2).unwrap().is_empty());
        let found = details.query_slot(5, 2).unwrap();
        assert_eq!(found.ids, [1234, 9876, DETAIL_ID_EMPTY, 42]);
        assert_eq!(found.c_dir, 15);
        assert_eq!(found.c_hemi, 6);
        assert_eq!((found.c_r, found.c_g, found.c_b), (3, 12, 9));
        assert_eq!(found.palette[0], DetailPalette { a0: 1, a1: 2, a2: 14, a3: 15 });
        assert!((found.y_base_meters() - 12.4).abs() < 0.25);
        assert!(found.y_height_meters() >= 3.0);

        let bounds = details.slot_world_bounds(5, 2).unwrap();
        assert_eq!(bounds.min[0], 10.0);
        assert_eq!(bounds.max[0], 12.0);
        assert_eq!(bounds.min[2], 4.0);
    }

    #[test]
    fn test_level_details_v3_16byte_slot_expansion() {
        let mut hdr = Vec::new();
        for v in [DETAIL_VERSION_3, 1, 0, 0, 1, 1] {
            hdr.extend_from_slice(&v.to_le_bytes());
        }
        let mut objs = ChunkWriter::new();
        objs.write_chunk(0, &build_dm_bytes("details\\blend", "detail\\grass", 0));

        // Pack 16-byte DetailSlot_v3:
        // w0: y_base=1050 (10m), y_height=25 (2.5m), id0=5, id1=0x3F (empty)
        let w0: u32 = 1050u32 | (25u32 << 12) | (5u32 << 20) | (0x3Fu32 << 26);
        // w1: id2=17, id3=0x3F, c_dir=8, c_hemi=4, c_r=15, c_g=7, c_b=2
        let w1: u32 = 17u32
            | (0x3Fu32 << 6)
            | (8u32 << 12)
            | (4u32 << 16)
            | (15u32 << 20)
            | (7u32 << 24)
            | (2u32 << 28);
        let mut s_raw = Vec::new();
        s_raw.extend_from_slice(&w0.to_le_bytes());
        s_raw.extend_from_slice(&w1.to_le_bytes());
        s_raw.extend_from_slice(&0xF841u16.to_le_bytes()); // a0=1, a1=4, a2=8, a3=15
        s_raw.extend_from_slice(&[0u8; 6]);

        let mut root = ChunkWriter::new();
        root.write_chunk(0, &hdr);
        root.write_chunk(1, &objs.into_bytes());
        root.write_chunk(2, &s_raw);

        let parsed = LevelDetails::parse(&root.into_bytes()).unwrap();
        let slot = parsed.slot_at(0, 0).unwrap();
        assert_eq!(slot.ids, [5, DETAIL_ID_EMPTY, 17, DETAIL_ID_EMPTY]);
        assert!((slot.y_base_meters() - 10.0).abs() < 1e-4);
        assert!((slot.y_height_meters() - 2.5).abs() < 1e-4);
        assert_eq!((slot.c_dir, slot.c_hemi, slot.c_r, slot.c_g, slot.c_b), (8, 4, 15, 7, 2));
        assert_eq!(slot.palette[0], DetailPalette { a0: 1, a1: 4, a2: 8, a3: 15 });
    }

    #[test]
    fn test_dm_bounding_sphere_and_transfer() {
        let dm_bytes = build_dm_bytes("details\\blend", "detail\\rain", 0);
        let dm = DetailMesh::parse_dm(&dm_bytes).unwrap();
        let (center, radius) = dm.bounding_sphere().unwrap();
        assert!((center[0] - 0.2).abs() < 1e-6);
        assert!((center[1] - 0.25).abs() < 1e-6);
        assert!((radius - (0.2_f32 * 0.2 + 0.25 * 0.25).sqrt()).abs() < 1e-5);

        let mut xf = crate::ogf::matrix_rotate_y(std::f32::consts::FRAC_PI_2);
        xf[3][0] = 10.0;
        xf[3][1] = 2.0;
        xf[3][2] = -3.0;
        let color = [0.2, 0.4, 0.6, 0.8];
        let (verts, idx) = dm.transfer(&xf, color, 0.1, 0.5, 100).unwrap();
        assert_eq!(verts.len(), 3);
        assert_eq!(idx, vec![100, 101, 102]);
        assert!((verts[2].position[0] - 10.0).abs() < 1e-5);
        assert!((verts[2].position[2] + 3.4).abs() < 1e-5);
        assert_eq!(verts[1].color_rgba, color);
        assert!((verts[1].uv[0] - 0.6).abs() < 1e-6);
        assert!((verts[1].uv[1] - 0.5).abs() < 1e-6);
        assert!(dm.transfer(&xf, color, 0.0, 0.0, u16::MAX).is_none());
    }
}
