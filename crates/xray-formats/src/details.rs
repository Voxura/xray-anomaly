//! Pure Rust X-Ray `level.details` & `.dm` Flora/Grass Detail Parser
//!
//! Faithful pure-Rust implementation matching `src/Layers/xrRender/DetailFormat.h`,
//! `src/Layers/xrRender/DetailModel.cpp`, `DetailManager.cpp`, and `DetailManager_cache.cpp`.
//! Supports both legacy `DETAIL_VERSION_3` (16-byte slots, 6-bit IDs) and Anomaly
//! `DETAIL_VERSION_4` (20-byte slots, 14-bit object IDs).

use crate::cform::{ray_triangle_test, CformBvh, CformModel};
use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::gamemtl::{gamemtl_flags, GameMaterialLibrary};
use crate::hom::{ClipFrustum, OccRasterizer};
use crate::ogf::{
    matrix_mul_43, matrix_rotate_y, matrix_scale, matrix_transform_point,
    vector_generate_orthonormal_basis, OgfBBox, XRayMatrix,
};
use crate::particle_sim::ParticleRng;
use crate::weather::PerlinNoise1D;
use xray_core::LtxDocument;
use xray_core::ltx_loader::OwnedLtxDocument;

pub const DETAIL_VERSION_3: u32 = 3;
pub const DETAIL_VERSION_4: u32 = 4;
pub const DETAIL_SLOT_SIZE: f32 = 2.0;
pub const DETAIL_SLOT_HALF_SIZE: f32 = 1.0;
pub const DETAIL_ID_EMPTY_V3: u16 = 0x003F;
pub const DETAIL_ID_EMPTY: u16 = 0x3FFF;
pub const DO_NO_WAVING: u32 = 0x0001;
pub const DETAIL_CACHE1_COUNT: usize = 4;
pub const DETAIL_OBJ_IN_SLOT: usize = 4;
pub const DETAIL_MAX_OBJECTS: usize = 16383;
pub const DETAIL_DEFAULT_RADIUS: u32 = 49;
pub const DETAIL_HW_QUANT: f32 = 16384.0;
pub const DETAIL_DEFAULT_L_AMBIENT: f32 = 0.9;
pub const DETAIL_DEFAULT_L_ANISO: f32 = 0.25;
pub const DETAIL_DEFAULT_DENSITY: f32 = 0.6;
pub const DETAIL_DEFAULT_HEIGHT: f32 = 1.0;
/// Console CVar default `ps_r__ssaDISCARD = 3.5f` (`xrRender_console.cpp:146`).
pub const DETAIL_DEFAULT_PS_SSA_DISCARD: f32 = 3.5;
/// Normalized `r_ssaDISCARD = (3.5^2) / (1920 * 1080)` (`r2_R_calculate.cpp:20`).
pub const DETAIL_DEFAULT_SSA_DISCARD: f32 = (3.5 * 3.5) / (1920.0 * 1080.0);

/// Compute normalized `r_ssaDISCARD` matching `CRender::Calculate` (`r2_R_calculate.cpp:18-20`).
#[inline]
pub fn compute_r_ssa_discard(
    viewport_width: u32,
    viewport_height: u32,
    fov_deg: f32,
    r_lod: f32,
    ps_ssa_discard: f32,
) -> f32 {
    let fov_factor = (90.0 / fov_deg.max(1.0)).powi(2);
    let g_f_screen = (viewport_width.max(1) as f32)
        * (viewport_height.max(1) as f32)
        * fov_factor
        * (1e-5 + r_lod.max(0.0));
    (ps_ssa_discard * ps_ssa_discard) / g_f_screen.max(1.0)
}

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

    /// Snap world `(x, z)` to `CDetailManager::MT_CALC` cache grid center `(s_x, s_z)`
    /// (`iFloor(EYE / dm_slot_size + 0.5)` in `DetailManager.cpp:477-478`).
    #[inline]
    pub fn world_to_cache_center(world_x: f32, world_z: f32) -> (i32, i32) {
        (
            (world_x / DETAIL_SLOT_SIZE + 0.5).floor() as i32,
            (world_z / DETAIL_SLOT_SIZE + 0.5).floor() as i32,
        )
    }
}

// ============================================================================
// Pure-Rust CDetailManager Runtime: Swing, Dither, HW Batches, Cache & Culling
// ============================================================================

/// Grass wind oscillation profile (`CDetailManager::SSwingValue` in `DetailManager.h:158-166`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailSwingValue {
    pub rot1: f32,
    pub rot2: f32,
    pub amp1: f32,
    pub amp2: f32,
    pub speed: f32,
}

impl DetailSwingValue {
    #[inline]
    pub fn lerp(self, other: Self, factor: f32) -> Self {
        let fi = 1.0 - factor;
        Self {
            amp1: fi * self.amp1 + factor * other.amp1,
            amp2: fi * self.amp2 + factor * other.amp2,
            rot1: fi * self.rot1 + factor * other.rot1,
            rot2: fi * self.rot2 + factor * other.rot2,
            speed: fi * self.speed + factor * other.speed,
        }
    }
}

/// `[details]` LTX swing preset pair (`swing_desc[2]` in `DetailManager.cpp:244-254`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailSwingConfig {
    pub normal: DetailSwingValue,
    pub fast: DetailSwingValue,
}

impl Default for DetailSwingConfig {
    fn default() -> Self {
        Self {
            normal: DetailSwingValue {
                rot1: 10.0,
                rot2: 5.0,
                amp1: 0.005,
                amp2: 0.002,
                speed: 1.0,
            },
            fast: DetailSwingValue {
                rot1: 3.0,
                rot2: 1.5,
                amp1: 0.03,
                amp2: 0.015,
                speed: 3.0,
            },
        }
    }
}

impl DetailSwingConfig {
    pub fn from_ltx_str(ltx_text: &str) -> Self {
        let doc = LtxDocument::parse(ltx_text);
        Self::from_ltx_lookup(|k| doc.read_str("details", k).and_then(|s| s.parse::<f32>().ok()))
    }

    pub fn from_ltx_doc(doc: &OwnedLtxDocument) -> Self {
        Self::from_ltx_lookup(|k| doc.read_str("details", k).and_then(|s| s.parse::<f32>().ok()))
    }

    fn from_ltx_lookup(mut f: impl FnMut(&str) -> Option<f32>) -> Self {
        let def = Self::default();
        Self {
            normal: DetailSwingValue {
                amp1: f("swing_normal_amp1").unwrap_or(def.normal.amp1),
                amp2: f("swing_normal_amp2").unwrap_or(def.normal.amp2),
                rot1: f("swing_normal_rot1").unwrap_or(def.normal.rot1),
                rot2: f("swing_normal_rot2").unwrap_or(def.normal.rot2),
                speed: f("swing_normal_speed").unwrap_or(def.normal.speed),
            },
            fast: DetailSwingValue {
                amp1: f("swing_fast_amp1").unwrap_or(def.fast.amp1),
                amp2: f("swing_fast_amp2").unwrap_or(def.fast.amp2),
                rot1: f("swing_fast_rot1").unwrap_or(def.fast.rot1),
                rot2: f("swing_fast_rot2").unwrap_or(def.fast.rot2),
                speed: f("swing_fast_speed").unwrap_or(def.fast.speed),
            },
        }
    }
}

/// GPU shader wave/wind constants matching `dx10DetailManager_VS.cpp:95-124`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailWaveConstants {
    pub consts: [f32; 4],
    pub wave0: [f32; 4],
    pub wave1: [f32; 4],
    pub dir1: [f32; 4],
    pub dir2: [f32; 4],
    pub prev_wave0: [f32; 4],
    pub prev_wave1: [f32; 4],
    pub prev_dir1: [f32; 4],
    pub prev_dir2: [f32; 4],
    pub still_consts: [f32; 4],
}

/// Time accumulator for grass wind animation (`CDetailManager::hw_Render`, `dx10DetailManager_VS.cpp:70-135`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DetailSwingState {
    pub time_rot_1: f32,
    pub time_rot_2: f32,
    pub time_pos: f32,
    pub prev_time_pos: f32,
    pub prev_dir1: [f32; 4],
    pub prev_dir2: [f32; 4],
}

impl DetailSwingState {
    pub fn step(
        &mut self,
        dt_sec: f32,
        wind_strength_factor: f32,
        cfg: &DetailSwingConfig,
        l_aniso: f32,
        l_ambient: f32,
    ) -> DetailWaveConstants {
        let dt = if !dt_sec.is_finite() || !(0.0..=1.0).contains(&dt_sec) {
            0.03
        } else {
            dt_sec
        };
        let swing = cfg.normal.lerp(cfg.fast, wind_strength_factor);
        let rot1 = if swing.rot1.abs() > 1e-5 { swing.rot1 } else { 1.0 };
        let rot2 = if swing.rot2.abs() > 1e-5 { swing.rot2 } else { 1.0 };

        self.time_rot_1 += std::f32::consts::TAU * dt / rot1;
        self.time_rot_2 += std::f32::consts::TAU * dt / rot2;
        self.time_pos += dt * swing.speed;

        // X-Ray normalizes `(sin(t), 0, cos(t), 0)` before scaling by `amp` (magnitude stays 1).
        let (s1, c1) = self.time_rot_1.sin_cos();
        let (s2, c2) = self.time_rot_2.sin_cos();
        let dir1 = [s1 * swing.amp1, 0.0, c1 * swing.amp1, 0.0];
        let dir2 = [s2 * swing.amp2, 0.0, c2 * swing.amp2, 0.0];

        let inv_tau = 1.0 / std::f32::consts::TAU;
        let inv_quant = 1.0 / DETAIL_HW_QUANT;
        let out = DetailWaveConstants {
            consts: [inv_quant, inv_quant, l_aniso, l_ambient],
            wave0: [
                (1.0 / 5.0) * inv_tau,
                (1.0 / 7.0) * inv_tau,
                (1.0 / 3.0) * inv_tau,
                self.time_pos * inv_tau,
            ],
            wave1: [
                (1.0 / 3.0) * inv_tau,
                (1.0 / 7.0) * inv_tau,
                (1.0 / 5.0) * inv_tau,
                self.time_pos * inv_tau,
            ],
            dir1,
            dir2,
            prev_wave0: [
                (1.0 / 5.0) * inv_tau,
                (1.0 / 7.0) * inv_tau,
                (1.0 / 3.0) * inv_tau,
                self.prev_time_pos * inv_tau,
            ],
            prev_wave1: [
                (1.0 / 3.0) * inv_tau,
                (1.0 / 7.0) * inv_tau,
                (1.0 / 5.0) * inv_tau,
                self.prev_time_pos * inv_tau,
            ],
            prev_dir1: self.prev_dir1,
            prev_dir2: self.prev_dir2,
            still_consts: [inv_quant, inv_quant, inv_quant, 1.0],
        };

        self.prev_time_pos = self.time_pos;
        self.prev_dir1 = dir1;
        self.prev_dir2 = dir2;
        out
    }
}

/// Ordered 16x16 Bayer-style dither table (`bwdithermap` in `DetailManager.cpp:28-59`).
pub fn detail_bwdithermap(levels: i32) -> [[i32; 16]; 16] {
    const MAGIC_4X4: [[i32; 4]; 4] = [
        [0, 14, 3, 13],
        [11, 5, 8, 6],
        [12, 2, 15, 1],
        [7, 9, 4, 10],
    ];
    let denom = (levels.max(2) - 1) as f64;
    let n = 255.0_f64 / denom;
    let magicfact = (n - 1.0) / 16.0;
    let mut out = [[0i32; 16]; 16];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                for l in 0..4 {
                    let v = 0.5
                        + (MAGIC_4X4[i][j] as f64) * magicfact
                        + ((MAGIC_4X4[k][l] as f64) / 16.0) * magicfact;
                    out[4 * k + i][4 * l + j] = v as i32;
                }
            }
        }
    }
    out
}

/// Bilinear corner density interpolation matching `Interpolate` in `DetailManager_Decompress.cpp:14-31`.
#[inline]
pub fn detail_interpolate(base: &[f32; 4], x: u32, y: u32, size: u32) -> f32 {
    if size == 0 {
        return base[0];
    }
    let f = size as f32;
    let fx = (x as f32) / f;
    let ifx = 1.0 - fx;
    let fy = (y as f32) / f;
    let ify = 1.0 - fy;

    let c01 = base[0] * ifx + base[1] * fx;
    let c23 = base[2] * ifx + base[3] * fx;
    let c02 = base[0] * ify + base[2] * fy;
    let c13 = base[1] * ify + base[3] * fy;

    let cx = ify * c01 + fy * c23;
    let cy = ifx * c02 + fx * c13;
    (cx + cy) * 0.5
}

/// Bilinear corner density interpolation + 16x16 threshold test (`InterpolateAndDither`, `DetailManager_Decompress.cpp:33-43`).
#[inline]
pub fn detail_interpolate_and_dither(
    alpha255: &[f32; 4],
    x: u32,
    y: u32,
    sx: u32,
    sy: u32,
    size: u32,
    dither: &[[i32; 16]; 16],
) -> bool {
    let max_idx = size.saturating_sub(1);
    let cx = x.min(max_idx);
    let cy = y.min(max_idx);
    let c = (detail_interpolate(alpha255, cx, cy, size.max(1)) + 0.5).floor() as i32;
    let clamped = c.clamp(0, 255);
    let row = ((cy + sy) % 16) as usize;
    let col = ((cx + sx) % 16) as usize;
    clamped > dither[col][row]
}

/// Quantize a detail UV or normalized height coordinate into signed 16-bit (`QC(v)`, `dx10DetailManager_VS.cpp:14-40`).
#[inline]
pub fn quantize_detail_coord(v: f32) -> i16 {
    (v * DETAIL_HW_QUANT).floor().clamp(-32768.0, 32767.0) as i16
}

/// Packed hardware-instanced detail vertex (`struct vertHW`, 20 bytes in `dx10DetailManager_VS.cpp:26-32`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DetailHwVertex {
    pub position: [f32; 3],
    pub u_q: i16,
    pub v_q: i16,
    pub t_q: i16,
    pub mid: i16,
}

/// Per-object batch slice in a `DetailHwBatchMesh` (`hw_Load_Geom`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetailHwObjectSpan {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub vertex_count_per_instance: u32,
    pub index_count_per_instance: u32,
}

/// Replicated static vertex/index buffer for hardware detail batching (`CDetailManager::hw_Load_Geom`).
#[derive(Debug, Clone, PartialEq)]
pub struct DetailHwBatchMesh {
    pub batch_size: u32,
    pub vertices: Vec<DetailHwVertex>,
    pub indices: Vec<u16>,
    pub object_spans: Vec<DetailHwObjectSpan>,
}

/// Packed constant-buffer per-instance record matching `dx10DetailManager_VS.cpp:305-320`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailGpuInstanceData {
    /// 3x4 transposed scale/rotation/translation matrix (`rows[0..3]`) + `(sun, sun, sun, hemi)` (`rows[3]`).
    pub rows: [[f32; 4]; 4],
    /// Terrain normal `.xyz` + dissolve alpha `.w` (`exdata` buffer).
    pub ex_data: [f32; 4],
}

impl DetailMesh {
    /// Build a replicated `batch_size`-instanced vertex/index buffer for hardware detail batching
    /// (`CDetailManager::hw_Load_Geom` in `dx10DetailManager_VS.cpp` / `DetailManager_VS.cpp`).
    ///
    /// Returns `None` if `batch_size == 0` or replicated vertex indices per object would overflow `u16`.
    pub fn build_hw_batches(objects: &[DetailMesh], batch_size: u32) -> Option<DetailHwBatchMesh> {
        let batch_size = batch_size.clamp(1, 64);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut object_spans = Vec::with_capacity(objects.len());

        for obj in objects {
            let v_count = u32::try_from(obj.vertices.len()).ok()?;
            let i_count = u32::try_from(obj.indices.len()).ok()?;
            let batch_verts = v_count.checked_mul(batch_size)?;
            if batch_verts > u32::from(u16::MAX) + 1 {
                return None;
            }
            let vertex_offset = u32::try_from(vertices.len()).ok()?;
            let index_offset = u32::try_from(indices.len()).ok()?;

            let y_span = if let Some(bb) = obj.bbox() {
                let d = bb.max[1] - bb.min[1];
                if d.abs() > 1e-6 {
                    d
                } else {
                    1.0
                }
            } else {
                1.0
            };

            for batch in 0..batch_size {
                let mid = (batch * 4) as i16;
                for v in &obj.vertices {
                    vertices.push(DetailHwVertex {
                        position: v.position,
                        u_q: quantize_detail_coord(v.u),
                        v_q: quantize_detail_coord(v.v),
                        t_q: quantize_detail_coord(v.position[1] / y_span),
                        mid,
                    });
                }
            }

            let mut base_off: u16 = 0;
            for batch in 0..batch_size {
                for &idx in &obj.indices {
                    indices.push(idx.checked_add(base_off)?);
                }
                if batch + 1 < batch_size {
                    base_off = base_off.checked_add(u16::try_from(v_count).ok()?)?;
                }
            }

            object_spans.push(DetailHwObjectSpan {
                vertex_offset,
                index_offset,
                vertex_count_per_instance: v_count,
                index_count_per_instance: i_count,
            });
        }

        Some(DetailHwBatchMesh {
            batch_size,
            vertices,
            indices,
            object_spans,
        })
    }
}

/// Deterministic X-Ray MSVCRT-LCG PRNG matching `CRandom` (`src/xrCore/_random.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetailRandom {
    holdrand: i32,
}

impl DetailRandom {
    #[inline]
    pub fn new(seed: i32) -> Self {
        Self { holdrand: seed }
    }

    #[inline]
    pub fn rand_i(&mut self) -> i32 {
        self.holdrand = self
            .holdrand
            .wrapping_mul(214_013)
            .wrapping_add(2_531_011);
        (self.holdrand >> 16) & 0x7FFF
    }

    #[inline]
    pub fn rand_i_max(&mut self, max: i32) -> i32 {
        if max <= 0 {
            0
        } else {
            self.rand_i() % max
        }
    }

    #[inline]
    pub fn rand_f(&mut self) -> f32 {
        (self.rand_i() as f32) / 32767.0
    }

    #[inline]
    pub fn rand_f_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.rand_f() * (max - min)
    }

    #[inline]
    pub fn rand_fs(&mut self, range: f32) -> f32 {
        self.rand_f_range(-range, range)
    }
}

/// Exact 64-bit spatial slot seed hash (`hash2` in `DetailManager_Decompress.cpp:74-79`).
#[inline]
pub fn detail_hash2(sx: i32, sz: i32) -> u32 {
    let a = ((sx as u32 as u64) << 32) | (sz as u32 as u64);
    let r = a
        .wrapping_mul(9_199_940_308_585_234_877)
        .wrapping_add(9_199_940_308_585_234_877);
    (r >> 32) as u32
}

/// Slot decompression parameters matching `DetailManager_Decompress.cpp` & `xrRender_console.cpp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailDecompressConfig {
    pub density: f32,
    pub detail_height: f32,
    pub terrain_grass_align: bool,
    pub terrain_grass_slope: f32,
    pub no_random: bool,
    pub use_vs: bool,
}

impl Default for DetailDecompressConfig {
    fn default() -> Self {
        Self {
            density: DETAIL_DEFAULT_DENSITY,
            detail_height: DETAIL_DEFAULT_HEIGHT,
            terrain_grass_align: false,
            terrain_grass_slope: 1.0,
            no_random: false,
            use_vs: true,
        }
    }
}

/// Per-frame visibility & SSA culling configuration (`CDetailManager::UpdateVisibleM`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailVisibilityConfig {
    pub ssa_discard: f32,
    pub no_scale: bool,
    pub fade_distance: f32,
    pub light_position: Option<[f32; 3]>,
}

impl Default for DetailVisibilityConfig {
    fn default() -> Self {
        Self {
            ssa_discard: DETAIL_DEFAULT_SSA_DISCARD,
            no_scale: false,
            fade_distance: 99999.0,
            light_position: None,
        }
    }
}

/// Single decompressed grass/detail instance (`CDetailManager::SlotItem` in `DetailManager.h:73-90`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailSlotItem {
    pub scale: f32,
    pub scale_calculated: f32,
    pub xform: XRayMatrix,
    pub vis_id: u8,
    pub c_hemi: f32,
    pub c_sun: f32,
    pub c_rgb: [f32; 3],
    pub distance_sq: f32,
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub alpha: f32,
    pub alpha_target: f32,
}

impl DetailSlotItem {
    /// Advance per-instance alpha dissolve toward `alpha_target` at `1.0 / sec`
    /// (`GoToValue` in `dx10DetailManager_VS.cpp:39-52, 288`).
    #[inline]
    pub fn step_alpha(&mut self, dt_sec: f32) -> f32 {
        let dt = dt_sec.max(0.0);
        if (self.alpha - self.alpha_target).abs() <= dt {
            self.alpha = self.alpha_target;
        } else if self.alpha < self.alpha_target {
            self.alpha += dt;
        } else {
            self.alpha -= dt;
        }
        self.alpha
    }

    /// Pack matrix and lighting row for GPU instancing (`dx10DetailManager_VS.cpp:305-320`).
    pub fn to_gpu_instance(&self, effective_scale: f32) -> DetailGpuInstanceData {
        let m = &self.xform;
        DetailGpuInstanceData {
            rows: [
                [
                    m[0][0] * effective_scale,
                    m[1][0] * effective_scale,
                    m[2][0] * effective_scale,
                    m[3][0],
                ],
                [
                    m[0][1] * effective_scale,
                    m[1][1] * effective_scale,
                    m[2][1] * effective_scale,
                    m[3][1],
                ],
                [
                    m[0][2] * effective_scale,
                    m[1][2] * effective_scale,
                    m[2][2] * effective_scale,
                    m[3][2],
                ],
                [self.c_sun, self.c_sun, self.c_sun, self.c_hemi],
            ],
            ex_data: [self.normal[0], self.normal[1], self.normal[2], self.alpha],
        }
    }
}

/// One of 4 prototype buckets in a decompressed slot (`CDetailManager::SlotPart`).
#[derive(Debug, Clone, PartialEq)]
pub struct DetailSlotPart {
    pub object_id: u16,
    pub items: Vec<DetailSlotItem>,
}

impl Default for DetailSlotPart {
    fn default() -> Self {
        Self {
            object_id: DETAIL_ID_EMPTY,
            items: Vec::new(),
        }
    }
}

/// Decompressed detail cache cell (`CDetailManager::Slot` in `DetailManager.h:112-135`).
#[derive(Debug, Clone, PartialEq)]
pub struct DetailCacheSlotState {
    pub sx: i32,
    pub sz: i32,
    pub empty: bool,
    pub pending: bool,
    pub hidden: bool,
    pub next_eval_frame: u32,
    pub bbox: OgfBBox,
    pub sphere_center: [f32; 3],
    pub sphere_radius: f32,
    pub parts: [DetailSlotPart; DETAIL_OBJ_IN_SLOT],
}

impl Default for DetailCacheSlotState {
    fn default() -> Self {
        Self {
            sx: 0,
            sz: 0,
            empty: true,
            pending: false,
            hidden: false,
            next_eval_frame: 0,
            bbox: OgfBBox::default(),
            sphere_center: [0.0; 3],
            sphere_radius: 0.0,
            parts: [
                DetailSlotPart::default(),
                DetailSlotPart::default(),
                DetailSlotPart::default(),
                DetailSlotPart::default(),
            ],
        }
    }
}

/// 4x4 coarse hierarchical visibility cell (`CDetailManager::CacheSlot1`).
#[derive(Debug, Clone, PartialEq)]
pub struct DetailCacheLevel1 {
    pub empty: bool,
    pub bbox: OgfBBox,
    pub sphere_center: [f32; 3],
    pub sphere_radius: f32,
}

impl Default for DetailCacheLevel1 {
    fn default() -> Self {
        Self {
            empty: true,
            bbox: OgfBBox::default(),
            sphere_center: [0.0; 3],
            sphere_radius: 0.0,
        }
    }
}

/// Terrain intersection sample returned to `DetailCacheGrid` slot decompression.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailSurfaceHit {
    pub height_y: f32,
    pub normal: [f32; 3],
}

/// Per-frame culled detail instances partitioned by wave bucket (`0=still`, `1=wave1`, `2=wave2`) and object ID.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DetailVisibleBatches {
    pub still: Vec<Vec<DetailSlotItem>>,
    pub wave1: Vec<Vec<DetailSlotItem>>,
    pub wave2: Vec<Vec<DetailSlotItem>>,
}

/// Pure-Rust sliding spatial cache & visibility culler (`CDetailManager` runtime).
#[derive(Debug, Clone)]
pub struct DetailCacheGrid {
    dm_size: usize,
    dm_cache1_line: usize,
    dm_cache_line: usize,
    dm_fade: f32,
    cache_cx: i32,
    cache_cz: i32,
    dither: [[i32; 16]; 16],
    slots: Vec<DetailCacheSlotState>,
    level1: Vec<DetailCacheLevel1>,
    pending_indices: Vec<usize>,
    wave_rng: DetailRandom,
    object_bounds_radius: Vec<f32>,
}

fn bbox_to_sphere(bb: &OgfBBox) -> ([f32; 3], f32) {
    let center = [
        (bb.min[0] + bb.max[0]) * 0.5,
        (bb.min[1] + bb.max[1]) * 0.5,
        (bb.min[2] + bb.max[2]) * 0.5,
    ];
    let dx = bb.max[0] - center[0];
    let dy = bb.max[1] - center[1];
    let dz = bb.max[2] - center[2];
    (center, (dx * dx + dy * dy + dz * dz).sqrt())
}

impl DetailCacheGrid {
    /// Initialize cache grid around `(0, 0)` matching `CDetailManager::cache_Initialize`.
    pub fn new(details: &LevelDetails, detail_radius: u32) -> Self {
        let dm_size = (((detail_radius.max(4) as f32) / 4.0).floor() as usize * 2).max(2);
        let dm_cache1_line = (dm_size * 2) / DETAIL_CACHE1_COUNT;
        let dm_cache_line = dm_size + 1 + dm_size;
        let dm_fade = (2 * dm_size) as f32 - 0.5;
        let cache_size = dm_cache_line * dm_cache_line;

        let mut object_bounds_radius = Vec::with_capacity(details.objects.len());
        for obj in &details.objects {
            let r = obj.bounding_sphere().map(|(_, rad)| rad).unwrap_or(0.5);
            object_bounds_radius.push(r);
        }

        let mut grid = Self {
            dm_size,
            dm_cache1_line,
            dm_cache_line,
            dm_fade,
            cache_cx: 0,
            cache_cz: 0,
            dither: detail_bwdithermap(2),
            slots: vec![DetailCacheSlotState::default(); cache_size],
            level1: vec![DetailCacheLevel1::default(); dm_cache1_line * dm_cache1_line],
            pending_indices: Vec::with_capacity(cache_size),
            wave_rng: DetailRandom::new(0x1234_5678),
            object_bounds_radius,
        };

        for gz in 0..dm_cache_line {
            for gx in 0..dm_cache_line {
                let idx = gz * dm_cache_line + gx;
                grid.assign_task(details, gx, gz, idx);
            }
        }
        grid.rebuild_level1_bounds();
        grid
    }

    #[inline]
    pub fn dm_size(&self) -> usize {
        self.dm_size
    }

    #[inline]
    pub fn cache_line(&self) -> usize {
        self.dm_cache_line
    }

    #[inline]
    pub fn cache_size(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn cache1_line(&self) -> usize {
        self.dm_cache1_line
    }

    #[inline]
    pub fn fade_distance_slots(&self) -> f32 {
        self.dm_fade
    }

    #[inline]
    pub fn center(&self) -> (i32, i32) {
        (self.cache_cx, self.cache_cz)
    }

    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending_indices.len()
    }

    #[inline]
    pub fn cg2w_x(&self, gx: usize) -> i32 {
        self.cache_cx - (self.dm_size as i32) + (gx as i32)
    }

    #[inline]
    pub fn cg2w_z(&self, gz: usize) -> i32 {
        self.cache_cz - (self.dm_size as i32) + (self.dm_cache_line as i32 - 1 - gz as i32)
    }

    #[inline]
    pub fn w2cg_x(&self, sx: i32) -> isize {
        (sx - self.cache_cx + self.dm_size as i32) as isize
    }

    #[inline]
    pub fn w2cg_z(&self, sz: i32) -> isize {
        (self.cache_cz - self.dm_size as i32 + (self.dm_cache_line as i32 - 1 - sz)) as isize
    }

    #[inline]
    pub fn slot_at_grid(&self, gx: usize, gz: usize) -> Option<&DetailCacheSlotState> {
        if gx < self.dm_cache_line && gz < self.dm_cache_line {
            self.slots.get(gz * self.dm_cache_line + gx)
        } else {
            None
        }
    }

    /// Relative slot lookup matching `CDetailManager::cache_Query`.
    pub fn query_slot_rel(&self, rel_x: i32, rel_z: i32) -> Option<&DetailCacheSlotState> {
        let gx = self.w2cg_x(rel_x + self.cache_cx);
        let gz = self.w2cg_z(rel_z + self.cache_cz);
        if gx >= 0 && gz >= 0 {
            self.slot_at_grid(gx as usize, gz as usize)
        } else {
            None
        }
    }

    /// Verify every grid slot matches `(cg2w_x(gx), cg2w_z(gz))` (`CDetailManager::cache_Validate`).
    pub fn validate_coords(&self) -> bool {
        for gz in 0..self.dm_cache_line {
            for gx in 0..self.dm_cache_line {
                let s = &self.slots[gz * self.dm_cache_line + gx];
                if s.sx != self.cg2w_x(gx) || s.sz != self.cg2w_z(gz) {
                    return false;
                }
            }
        }
        true
    }

    fn assign_task(&mut self, details: &LevelDetails, gx: usize, gz: usize, idx: usize) {
        let sx = self.cg2w_x(gx);
        let sz = self.cg2w_z(gz);
        let ds = details.query_slot(sx, sz).copied().unwrap_or_default();

        let slot = &mut self.slots[idx];
        let was_pending = slot.pending;
        slot.sx = sx;
        slot.sz = sz;
        slot.empty = ds.is_empty();
        slot.pending = true;
        slot.hidden = false;
        slot.next_eval_frame = 0;

        let x0 = (sx as f32) * DETAIL_SLOT_SIZE;
        let z0 = (sz as f32) * DETAIL_SLOT_SIZE;
        let y0 = ds.y_base_meters();
        let eps_l = 0.001_f32;
        slot.bbox = OgfBBox {
            min: [x0 - eps_l, y0 - eps_l, z0 - eps_l],
            max: [
                x0 + DETAIL_SLOT_SIZE + eps_l,
                y0 + ds.y_height_meters() + eps_l,
                z0 + DETAIL_SLOT_SIZE + eps_l,
            ],
        };
        let (sc, sr) = bbox_to_sphere(&slot.bbox);
        slot.sphere_center = sc;
        slot.sphere_radius = sr;

        for (i, part) in slot.parts.iter_mut().enumerate() {
            part.object_id = ds.ids[i];
            part.items.clear();
        }

        if !was_pending {
            self.pending_indices.push(idx);
        }
    }

    fn rebuild_level1_bounds(&mut self) {
        let l1 = self.dm_cache1_line;
        for mz1 in 0..l1 {
            for mx1 in 0..l1 {
                let mut empty = true;
                let mut bmin = [f32::INFINITY; 3];
                let mut bmax = [f32::NEG_INFINITY; 3];
                for z in 0..DETAIL_CACHE1_COUNT {
                    for x in 0..DETAIL_CACHE1_COUNT {
                        let gx = mx1 * DETAIL_CACHE1_COUNT + x;
                        let gz = mz1 * DETAIL_CACHE1_COUNT + z;
                        let s = &self.slots[gz * self.dm_cache_line + gx];
                        for a in 0..3 {
                            bmin[a] = bmin[a].min(s.bbox.min[a]);
                            bmax[a] = bmax[a].max(s.bbox.max[a]);
                        }
                        if !s.empty {
                            empty = false;
                        }
                    }
                }
                let bb = if bmin[0].is_finite() {
                    OgfBBox { min: bmin, max: bmax }
                } else {
                    OgfBBox::default()
                };
                let (center, radius) = bbox_to_sphere(&bb);
                self.level1[mz1 * l1 + mx1] = DetailCacheLevel1 {
                    empty,
                    bbox: bb,
                    sphere_center: center,
                    sphere_radius: radius,
                };
            }
        }
    }

    fn shift_grid(&mut self, details: &LevelDetails, target_cx: i32, target_cz: i32) -> bool {
        let moved = self.cache_cx != target_cx || self.cache_cz != target_cz;
        let line = self.dm_cache_line;

        while self.cache_cx != target_cx {
            if target_cx > self.cache_cx {
                self.cache_cx += 1;
                for gz in 0..line {
                    let row = gz * line;
                    self.slots[row..row + line].rotate_left(1);
                    self.assign_task(details, line - 1, gz, row + line - 1);
                }
            } else {
                self.cache_cx -= 1;
                for gz in 0..line {
                    let row = gz * line;
                    self.slots[row..row + line].rotate_right(1);
                    self.assign_task(details, 0, gz, row);
                }
            }
        }

        while self.cache_cz != target_cz {
            if target_cz > self.cache_cz {
                self.cache_cz += 1;
                for gx in 0..line {
                    let saved = self.slots[(line - 1) * line + gx].clone();
                    for gz in (1..line).rev() {
                        self.slots[gz * line + gx] = self.slots[(gz - 1) * line + gx].clone();
                    }
                    self.slots[gx] = saved;
                    self.assign_task(details, gx, 0, gx);
                }
            } else {
                self.cache_cz -= 1;
                for gx in 0..line {
                    let saved = self.slots[gx].clone();
                    for gz in 1..line {
                        self.slots[(gz - 1) * line + gx] = self.slots[gz * line + gx].clone();
                    }
                    self.slots[(line - 1) * line + gx] = saved;
                    self.assign_task(details, gx, line - 1, (line - 1) * line + gx);
                }
            }
        }

        if moved {
            self.pending_indices.clear();
            for (idx, s) in self.slots.iter().enumerate() {
                if s.pending {
                    self.pending_indices.push(idx);
                }
            }
        }
        moved
    }

    /// Shift the sliding grid to `(target_cx, target_cz)` and decompress up to `decompress_limit`
    /// nearest pending slots via `sampler` (`cache_Update` + `cache_Decompress`).
    #[allow(clippy::too_many_arguments)]
    pub fn update_cache_with_sampler<F>(
        &mut self,
        details: &LevelDetails,
        target_cx: i32,
        target_cz: i32,
        camera_pos: [f32; 3],
        decompress_limit: usize,
        cfg: &DetailDecompressConfig,
        mut sampler: F,
    ) where
        F: FnMut(f32, f32, &OgfBBox) -> Option<DetailSurfaceHit>,
    {
        let need_mega_update = self.shift_grid(details, target_cx, target_cz);

        let full_unpack = self.pending_indices.len() == self.slots.len();
        let limit = if full_unpack {
            self.slots.len()
        } else {
            decompress_limit
        };

        for _ in 0..limit {
            if self.pending_indices.is_empty() {
                break;
            }
            let best_pos = if full_unpack {
                self.pending_indices.len() - 1
            } else {
                let mut best_i = 0;
                let mut best_dist = f32::INFINITY;
                for (pos, &slot_idx) in self.pending_indices.iter().enumerate() {
                    let s = &self.slots[slot_idx];
                    let dx = camera_pos[0] - s.sphere_center[0];
                    let dy = camera_pos[1] - s.sphere_center[1];
                    let dz = camera_pos[2] - s.sphere_center[2];
                    let d = dx * dx + dy * dy + dz * dz;
                    if d < best_dist {
                        best_dist = d;
                        best_i = pos;
                    }
                }
                best_i
            };

            let slot_idx = self.pending_indices.swap_remove(best_pos);
            self.decompress_slot_at(details, slot_idx, cfg, &mut sampler);
        }

        if need_mega_update || full_unpack {
            self.rebuild_level1_bounds();
        }
    }

    /// Convenience wrapper of `update_cache_with_sampler` using a `CformBvh` (`DetailManager_Decompress.cpp:102-240`),
    /// skipping passable triangles (`gamemtl_flags::FL_PASSABLE`) when `mtl_lib` is supplied.
    #[allow(clippy::too_many_arguments)]
    pub fn update_cache_with_cform(
        &mut self,
        details: &LevelDetails,
        target_cx: i32,
        target_cz: i32,
        camera_pos: [f32; 3],
        decompress_limit: usize,
        cfg: &DetailDecompressConfig,
        bvh: &CformBvh<'_>,
        model: &CformModel,
        mtl_lib: Option<&GameMaterialLibrary>,
    ) {
        self.update_cache_with_sampler(
            details,
            target_cx,
            target_cz,
            camera_pos,
            decompress_limit,
            cfg,
            |px, pz, slot_box| {
                let center = [
                    (slot_box.min[0] + slot_box.max[0]) * 0.5,
                    (slot_box.min[1] + slot_box.max[1]) * 0.5,
                    (slot_box.min[2] + slot_box.max[2]) * 0.5,
                ];
                let half = [
                    (slot_box.max[0] - slot_box.min[0]) * 0.5,
                    (slot_box.max[1] - slot_box.min[1]) * 0.5,
                    (slot_box.max[2] - slot_box.min[2]) * 0.5,
                ];
                let hits = bvh.box_query(center, half, true, false);
                if hits.is_empty() {
                    return None;
                }
                let origin = [px, slot_box.max[1], pz];
                let dir = [0.0_f32, -1.0, 0.0];
                let mut best_y = slot_box.min[1] - 5.0;
                let mut best_normal = None;

                for h in hits {
                    if let Some(lib) = mtl_lib
                        && let Some(mtl) = model.triangle_material(h.tri_index, lib)
                        && mtl.has_flag(gamemtl_flags::FL_PASSABLE)
                    {
                        continue;
                    }
                    let [p0, p1, p2] = h.verts;
                    if let Some((range, _, _)) = ray_triangle_test(origin, dir, p0, p1, p2, true)
                        && range >= 0.0
                    {
                        let y_test = origin[1] - range;
                        if y_test > best_y {
                            best_y = y_test;
                            let e0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                            let e1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
                            let n = [
                                e0[1] * e1[2] - e0[2] * e1[1],
                                e0[2] * e1[0] - e0[0] * e1[2],
                                e0[0] * e1[1] - e0[1] * e1[0],
                            ];
                            let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
                            let norm = if len_sq > 1e-12 {
                                let inv = 1.0 / len_sq.sqrt();
                                [n[0] * inv, n[1] * inv, n[2] * inv]
                            } else {
                                [0.0, 1.0, 0.0]
                            };
                            best_normal = Some(norm);
                        }
                    }
                }

                best_normal.map(|normal| DetailSurfaceHit {
                    height_y: best_y,
                    normal,
                })
            },
        );
    }

    fn decompress_slot_at<F>(
        &mut self,
        details: &LevelDetails,
        slot_idx: usize,
        cfg: &DetailDecompressConfig,
        sampler: &mut F,
    ) where
        F: FnMut(f32, f32, &OgfBBox) -> Option<DetailSurfaceHit>,
    {
        let (sx, sz, empty, slot_box) = {
            let s = &mut self.slots[slot_idx];
            s.pending = false;
            if s.empty {
                return;
            }
            (s.sx, s.sz, s.empty, s.bbox)
        };
        if empty {
            return;
        }

        let ds = match details.query_slot(sx, sz) {
            Some(ds) if !ds.is_empty() => *ds,
            _ => return,
        };

        let mut alpha255 = [[0.0_f32; 4]; DETAIL_OBJ_IN_SLOT];
        for (dst, pal) in alpha255.iter_mut().zip(ds.palette.iter()) {
            dst[0] = 255.0 * f32::from(pal.a0) / 15.0;
            dst[1] = 255.0 * f32::from(pal.a1) / 15.0;
            dst[2] = 255.0 * f32::from(pal.a2) / 15.0;
            dst[3] = 255.0 * f32::from(pal.a3) / 15.0;
        }

        let density = cfg.density.max(0.05);
        let jitter = density / 1.7;
        let d_size = ((DETAIL_SLOT_SIZE / density).ceil() as u32).max(1);
        let seed = (0x1207_1980u32 ^ detail_hash2(sx, sz)) as i32;
        let mut r_selection = DetailRandom::new(seed);
        let mut r_jitter = DetailRandom::new(seed);
        let mut r_yaw = DetailRandom::new(seed);
        let mut r_scale = DetailRandom::new(seed);

        let min_x = (sx as f32) * DETAIL_SLOT_SIZE;
        let min_z = (sz as f32) * DETAIL_SLOT_SIZE;
        let min_y = ds.y_base_meters();
        let c_hemi = ds.hemi_light();
        let c_sun = ds.dir_light();
        let c_rgb = ds.rgb_light();

        let mut bmin = [f32::INFINITY; 3];
        let mut bmax = [f32::NEG_INFINITY; 3];
        let mut any_bounds = false;
        let mut out_parts: [Vec<DetailSlotItem>; DETAIL_OBJ_IN_SLOT] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for z in 0..=d_size {
            for x in 0..=d_size {
                let (shift_x, shift_z) = if cfg.no_random {
                    (8u32, 8u32)
                } else {
                    (
                        r_jitter.rand_i_max(16) as u32,
                        r_jitter.rand_i_max(16) as u32,
                    )
                };

                let mut selected = [0usize; DETAIL_OBJ_IN_SLOT];
                let mut sel_len = 0usize;
                for (slot_part, alpha_row) in alpha255.iter().enumerate() {
                    if ds.ids[slot_part] != DETAIL_ID_EMPTY
                        && (cfg.no_random
                            || detail_interpolate_and_dither(
                                alpha_row,
                                x,
                                z,
                                shift_x,
                                shift_z,
                                d_size,
                                &self.dither,
                            ))
                    {
                        selected[sel_len] = slot_part;
                        sel_len += 1;
                    }
                }

                if sel_len == 0 {
                    continue;
                }
                let bucket_idx = if cfg.no_random || sel_len == 1 {
                    selected[0]
                } else {
                    selected[r_selection.rand_i_max(sel_len as i32) as usize]
                };

                let obj_id = ds.ids[bucket_idx];
                let Some(dobj) = details.objects.get(obj_id as usize) else {
                    continue;
                };

                let rx = ((x as f32) / (d_size as f32)) * DETAIL_SLOT_SIZE + min_x;
                let rz = ((z as f32) / (d_size as f32)) * DETAIL_SLOT_SIZE + min_z;
                let (px, pz) = if cfg.no_random {
                    (rx, rz)
                } else {
                    (rx + r_jitter.rand_fs(jitter), rz + r_jitter.rand_fs(jitter))
                };

                let slope_thresh =
                    -(1.0 - r_scale.rand_f_range(cfg.terrain_grass_slope * 0.8, cfg.terrain_grass_slope));
                let Some(hit) = sampler(px, pz, &slot_box) else {
                    continue;
                };
                if -hit.normal[1] > slope_thresh || hit.height_y < min_y {
                    continue;
                }

                let mut scale = if cfg.no_random {
                    (dobj.min_scale * 0.5 + dobj.max_scale * 0.9) * 0.5
                } else {
                    r_scale.rand_f_range(dobj.min_scale * 0.5, dobj.max_scale * 0.9)
                };
                scale *= cfg.detail_height;

                let yaw = if cfg.no_random {
                    0.0
                } else {
                    r_yaw.rand_f_range(0.0, std::f32::consts::TAU)
                };
                let mut m_rot_y = matrix_rotate_y(yaw);
                if cfg.terrain_grass_align {
                    let (axis_i, axis_k) = vector_generate_orthonormal_basis(hit.normal);
                    let aligned = [
                        [axis_i[0], axis_i[1], axis_i[2], 0.0],
                        [hit.normal[0], hit.normal[1], hit.normal[2], 0.0],
                        [axis_k[0], axis_k[1], axis_k[2], 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ];
                    // C++ `Item.mRotY.mulB_43(CurrMatrix)` computes `mul_43(aligned, CurrMatrix)`:
                    // first local yaw (`m_rot_y`), then terrain normal alignment (`aligned`).
                    m_rot_y = matrix_mul_43(&aligned, &m_rot_y);
                }
                m_rot_y[3][0] = px;
                m_rot_y[3][1] = hit.height_y;
                m_rot_y[3][2] = pz;

                let m_xform = matrix_mul_43(&m_rot_y, &matrix_scale(scale, scale, scale));
                if let Some(obj_bb) = dobj.bbox() {
                    let corners = [
                        [obj_bb.min[0], obj_bb.min[1], obj_bb.min[2]],
                        [obj_bb.min[0], obj_bb.min[1], obj_bb.max[2]],
                        [obj_bb.max[0], obj_bb.min[1], obj_bb.max[2]],
                        [obj_bb.max[0], obj_bb.min[1], obj_bb.min[2]],
                        [obj_bb.min[0], obj_bb.max[1], obj_bb.min[2]],
                        [obj_bb.min[0], obj_bb.max[1], obj_bb.max[2]],
                        [obj_bb.max[0], obj_bb.max[1], obj_bb.max[2]],
                        [obj_bb.max[0], obj_bb.max[1], obj_bb.min[2]],
                    ];
                    for c in corners {
                        let tc = matrix_transform_point(&m_xform, c);
                        for a in 0..3 {
                            bmin[a] = bmin[a].min(tc[a]);
                            bmax[a] = bmax[a].max(tc[a]);
                        }
                        any_bounds = true;
                    }
                }

                let vis_id = if cfg.no_random || !cfg.use_vs || dobj.no_waving() {
                    0
                } else if self.wave_rng.rand_i_max(3) == 0 {
                    2
                } else {
                    1
                };

                out_parts[bucket_idx].push(DetailSlotItem {
                    scale,
                    scale_calculated: scale,
                    xform: m_rot_y,
                    vis_id,
                    c_hemi,
                    c_sun,
                    c_rgb,
                    distance_sq: 0.0,
                    position: [px, hit.height_y, pz],
                    normal: hit.normal,
                    alpha: 1.0,
                    alpha_target: 1.0,
                });
            }
        }

        let slot = &mut self.slots[slot_idx];
        for (i, items) in out_parts.into_iter().enumerate() {
            slot.parts[i].object_id = ds.ids[i];
            slot.parts[i].items = items;
        }
        if any_bounds {
            let real_bb = OgfBBox { min: bmin, max: bmax };
            let (sc, sr) = bbox_to_sphere(&real_bb);
            slot.bbox = real_bb;
            slot.sphere_center = sc;
            slot.sphere_radius = sr;
        }
    }

    /// Frustum, optional HOM occlusion, fade, and SSA culling (`CDetailManager::UpdateVisibleM`).
    pub fn collect_visible(
        &mut self,
        camera_pos: [f32; 3],
        frustum: &ClipFrustum,
        hom: Option<&OccRasterizer>,
        full_transform: &XRayMatrix,
        frame: u32,
        cfg: &DetailVisibilityConfig,
    ) -> DetailVisibleBatches {
        let obj_count = self.object_bounds_radius.len();
        let mut out = DetailVisibleBatches {
            still: vec![Vec::new(); obj_count],
            wave1: vec![Vec::new(); obj_count],
            wave2: vec![Vec::new(); obj_count],
        };

        let fade_limit = self.dm_fade * self.dm_fade;
        let fade_start = 1.0_f32;
        let fade_range = (fade_limit - fade_start).max(1e-4);
        let r_ssa_cheap = 16.0 * cfg.ssa_discard;
        let l1 = self.dm_cache1_line;
        let line = self.dm_cache_line;

        for mz1 in 0..l1 {
            for mx1 in 0..l1 {
                let ms = &self.level1[mz1 * l1 + mx1];
                if ms.empty || !frustum.test_sphere(ms.sphere_center, ms.sphere_radius) {
                    continue;
                }

                for z in 0..DETAIL_CACHE1_COUNT {
                    for x in 0..DETAIL_CACHE1_COUNT {
                        let gx = mx1 * DETAIL_CACHE1_COUNT + x;
                        let gz = mz1 * DETAIL_CACHE1_COUNT + z;
                        let slot = &mut self.slots[gz * line + gx];
                        if slot.empty
                            || !frustum.test_sphere(slot.sphere_center, slot.sphere_radius)
                        {
                            continue;
                        }
                        if let Some(occ) = hom
                            && !occ.test_aabb(&slot.bbox, full_transform, camera_pos)
                        {
                            continue;
                        }

                        let dx = camera_pos[0] - slot.sphere_center[0];
                        let dy = camera_pos[1] - slot.sphere_center[1];
                        let dz = camera_pos[2] - slot.sphere_center[2];
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        if dist_sq > fade_limit {
                            slot.hidden = true;
                            continue;
                        }

                        if frame > slot.next_eval_frame {
                            slot.next_eval_frame =
                                frame.saturating_add(15 + (self.wave_rng.rand_i_max(15) as u32));
                        }

                        let alpha = if dist_sq < fade_start {
                            0.0
                        } else {
                            (dist_sq - fade_start) / fade_range
                        };
                        let alpha_i = (1.0 - alpha).max(0.0);
                        let dist_sq_rcp = 1.0 / dist_sq.max(1e-6);

                        for part in &mut slot.parts {
                            let obj_idx = part.object_id as usize;
                            if part.object_id == DETAIL_ID_EMPTY || obj_idx >= obj_count {
                                continue;
                            }
                            let r = self.object_bounds_radius[obj_idx];
                            let rq_drcp = r * r * dist_sq_rcp;

                            for item in &mut part.items {
                                let mut scale = if cfg.no_scale {
                                    item.scale
                                } else {
                                    item.scale * alpha_i
                                };
                                item.scale_calculated = scale;

                                let ssa = if cfg.no_scale {
                                    scale
                                } else {
                                    scale * scale * rq_drcp
                                };
                                if ssa < cfg.ssa_discard {
                                    item.alpha_target = 0.0;
                                    continue;
                                }

                                if let Some(lpos) = cfg.light_position {
                                    let ldx = item.position[0] - lpos[0];
                                    let ldz = item.position[2] - lpos[2];
                                    scale *= 1.0 - (ldx * ldx + ldz * ldz) * 0.005;
                                } else if dist_sq > cfg.fade_distance {
                                    scale *= 1.0 - (dist_sq - cfg.fade_distance).abs() * 0.005;
                                }
                                if scale <= 0.0 {
                                    continue;
                                }

                                if slot.hidden {
                                    item.alpha = 0.0;
                                    slot.hidden = false;
                                }
                                item.alpha_target = 1.0;
                                item.distance_sq = dist_sq;

                                let vis_id = if ssa > r_ssa_cheap { item.vis_id } else { 0 };
                                let mut copied = *item;
                                copied.scale_calculated = scale;
                                match vis_id {
                                    1 => out.wave1[obj_idx].push(copied),
                                    2 => out.wave2[obj_idx].push(copied),
                                    _ => out.still[obj_idx].push(copied),
                                }
                            }
                        }
                    }
                }
            }
        }

        out
    }
}

// ============================================================================
// SSFX Interactive Grass Benders (`IGame_Persistent.h/.cpp`, `device.cpp`, `FTreeVisual.cpp`)
// ============================================================================

/// Total GPU constant slots for interactive grass benders (slot `0` = player, `1..16` = NPCs/shots/anomalies).
pub const GRASS_BENDERS_MAX_SLOTS: usize = 16;

/// Interactive grass bender animation mode (`IGame_Persistent::GrassBenders_Anim`, `IGame_Persistent.h:71-79`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrassBenderAnim {
    Explosion = 0,
    #[default]
    Default = 1,
    Wavy = 2,
    Suck = 3,
    Blow = 4,
    Pulse = 5,
}

/// Per-entity tracking handle passed into [`GrassBenderPool::update_character`] (`data_idx`, `data_frame`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GrassBenderHandle {
    pub slot: u8,
    pub next_check_frame: u32,
}

/// Interactive grass configuration matching `ps_ssfx_grass_interactive` & `ps_ssfx_int_grass_params_1/2`
/// (`xrRender_console.cpp:484-486`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassBendersConfig {
    /// `ps_ssfx_grass_interactive.x > 0`
    pub player_enabled: bool,
    /// `ps_ssfx_grass_interactive.y` (clamped to `0..=15` so `1 + max_benders <= 16`).
    pub max_benders: usize,
    /// `ps_ssfx_grass_interactive.z` (squared XZ distance limit from camera).
    pub max_distance_sq: f32,
    /// `ps_ssfx_int_grass_params_1` (`benders_setup`).
    pub params_1: [f32; 4],
    /// `ps_ssfx_int_grass_params_2`.
    pub params_2: [f32; 4],
}

impl Default for GrassBendersConfig {
    fn default() -> Self {
        Self {
            player_enabled: false,
            max_benders: 0,
            max_distance_sq: 2000.0,
            params_1: [1.0, 1.0, 1.0, 25.0],
            params_2: [1.0, 5.0, 1.0, 1.0],
        }
    }
}

impl GrassBendersConfig {
    #[inline]
    pub fn clamped_max_benders(&self) -> usize {
        self.max_benders.min(GRASS_BENDERS_MAX_SLOTS - 1)
    }
}

/// State for one interactive grass bender slot (`IGame_Persistent::grass_data`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassBenderSlot {
    pub id: u16,
    pub anim: GrassBenderAnim,
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub radius: f32,
    pub radius_curr: f32,
    pub str: f32,
    pub str_target: f32,
    pub time: f32,
    pub fade: f32,
    pub speed: f32,
    pub prev_pos: [f32; 4],
    pub prev_dir: [f32; 4],
}

impl Default for GrassBenderSlot {
    fn default() -> Self {
        Self {
            id: 0,
            anim: GrassBenderAnim::Default,
            pos: [0.0; 3],
            dir: [0.0, -99.0, 0.0],
            radius: 0.0,
            radius_curr: 0.0,
            str: 0.0,
            str_target: 0.0,
            time: 0.0,
            fade: 0.0,
            speed: 0.0,
            prev_pos: [0.0; 4],
            prev_dir: [0.0, -99.0, 0.0, 0.0],
        }
    }
}

/// GPU constant payload packed identically to `FTreeVisual.cpp:191-242` (`benders_pos`, `benders_prevpos`, `benders_setup`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassBendersGpuData {
    /// Active bender slot count `_min(16, max_benders + 1)` when `max_benders > 0`, else `0`.
    pub active_count: usize,
    pub benders_setup: [f32; 4],
    /// `[0..16]` = `[pos.x, pos.y, pos.z, radius_curr]`, `[16..32]` = `[dir.x, dir.y, dir.z, str]`.
    pub benders_pos: [[f32; 4]; 32],
    /// Previous frame `[0..16]` positions+radii and `[16..32]` directions+strengths.
    pub benders_prevpos: [[f32; 4]; 32],
}

impl Default for GrassBendersGpuData {
    fn default() -> Self {
        Self {
            active_count: 0,
            benders_setup: [1.0, 1.0, 1.0, 25.0],
            benders_pos: [[0.0; 4]; 32],
            benders_prevpos: [[0.0; 4]; 32],
        }
    }
}

/// Interactive grass bender simulator & GPU constant packer (`IGame_Persistent::GrassBenders*` & `device.cpp:459-468`).
#[derive(Debug, Clone)]
pub struct GrassBenderPool {
    pub alloc_index: u8,
    pub slots: [GrassBenderSlot; GRASS_BENDERS_MAX_SLOTS],
    perlin: PerlinNoise1D,
    rng: ParticleRng,
}

impl Default for GrassBenderPool {
    fn default() -> Self {
        Self::new(0x1357)
    }
}

impl GrassBenderPool {
    /// Construct a new bender pool matching `IGame_Persistent::IGame_Persistent` (`octaves = 2`, `freq = 1.0`, `amp = 0.66666`).
    pub fn new(seed: u32) -> Self {
        Self {
            alloc_index: 0,
            slots: [GrassBenderSlot::default(); GRASS_BENDERS_MAX_SLOTS],
            perlin: PerlinNoise1D::new(seed & 0xFFFF, 2, 1.0, 0.66666),
            rng: ParticleRng::new(u64::from(seed).max(1)),
        }
    }

    #[inline]
    fn rand_i(&mut self, min_inclusive: u32, max_exclusive: u32) -> u32 {
        let span = max_exclusive.saturating_sub(min_inclusive).max(1) as usize;
        min_inclusive + (self.rng.range_usize(span) as u32)
    }

    /// Assign parameters into slot `idx` (`GrassBendersSet`, `IGame_Persistent.cpp:618-635`).
    #[allow(clippy::too_many_arguments)]
    pub fn set_slot(
        &mut self,
        idx: usize,
        id: u16,
        position: [f32; 3],
        dir: [f32; 3],
        fade: f32,
        speed: f32,
        intensity: f32,
        radius: f32,
        anim: GrassBenderAnim,
        reset_time: bool,
    ) {
        if idx >= GRASS_BENDERS_MAX_SLOTS {
            return;
        }
        let s = &mut self.slots[idx];
        s.anim = anim;
        s.pos = position;
        s.id = id;
        s.radius = radius;
        s.str = intensity;
        s.fade = fade;
        s.speed = speed;
        s.dir = dir;
        if reset_time {
            s.radius_curr = 0.01;
            s.time = 0.0;
        }
    }

    /// Reset one slot (`GrassBendersReset`, `IGame_Persistent.cpp:611-616`).
    pub fn reset_slot(&mut self, idx: usize) {
        if idx >= GRASS_BENDERS_MAX_SLOTS {
            return;
        }
        self.set_slot(
            idx,
            0,
            [0.0; 3],
            [0.0, -99.0, 0.0],
            0.0,
            0.0,
            0.0,
            0.0,
            GrassBenderAnim::Default,
            true,
        );
        self.slots[idx].str_target = 0.0;
    }

    /// Reset and unlink by handle (`GrassBendersRemoveByIndex`, `IGame_Persistent.cpp:594-601`).
    pub fn remove_by_handle(&mut self, handle: &mut GrassBenderHandle) {
        if handle.slot != 0 {
            self.reset_slot(handle.slot as usize);
            handle.slot = 0;
        }
    }

    /// Reset any active benders owned by `id` (`GrassBendersRemoveById`, `IGame_Persistent.cpp:603-609`).
    pub fn remove_by_id(&mut self, id: u16, cfg: &GrassBendersConfig) {
        let limit = cfg.clamped_max_benders();
        for idx in 1..=limit {
            if self.slots[idx].id == id {
                self.reset_slot(idx);
            }
        }
    }

    /// Update or claim a slot for a moving entity (`GrassBendersUpdate`, `IGame_Persistent.cpp:350-421`).
    #[allow(clippy::too_many_arguments)]
    pub fn update_character(
        &mut self,
        id: u16,
        handle: &mut GrassBenderHandle,
        frame: u32,
        position: [f32; 3],
        camera_pos: [f32; 3],
        frustum: &ClipFrustum,
        init_radius: f32,
        init_str: f32,
        check_distance: bool,
        cfg: &GrassBendersConfig,
    ) {
        let limit = cfg.clamped_max_benders();
        if limit < 1 || id == 0 {
            return;
        }

        if handle.slot != 0 {
            let slot_idx = handle.slot as usize;
            if slot_idx >= GRASS_BENDERS_MAX_SLOTS || self.slots[slot_idx].id != id {
                handle.slot = 0;
                handle.next_check_frame = frame.saturating_add(self.rand_i(10, 35));
            } else {
                self.slots[slot_idx].pos = position;
            }
        }

        if frame < handle.next_check_frame {
            return;
        }
        handle.next_check_frame = frame.saturating_add(self.rand_i(10, 35));

        if check_distance {
            let dx = position[0] - camera_pos[0];
            let dz = position[2] - camera_pos[2];
            if dx * dx + dz * dz > cfg.max_distance_sq {
                self.remove_by_handle(handle);
                return;
            }
        }

        let rad = if handle.slot == 0 {
            1.0
        } else {
            (self.slots[handle.slot as usize].radius_curr + 0.5).max(1.0)
        };
        if !frustum.test_sphere(position, rad) {
            self.remove_by_handle(handle);
            return;
        }

        if handle.slot == 0 {
            let idx = (self.alloc_index as usize) + 1;
            if idx < GRASS_BENDERS_MAX_SLOTS && self.slots[idx].id == 0 {
                handle.slot = idx as u8;
                self.set_slot(
                    idx,
                    id,
                    position,
                    [0.0, -99.0, 0.0],
                    0.0,
                    0.0,
                    0.0,
                    init_radius,
                    GrassBenderAnim::Default,
                    true,
                );
                self.slots[idx].str_target = init_str;
                self.slots[idx].radius_curr = init_radius;
            }
            self.alloc_index = if idx < limit { idx as u8 } else { 0 };
        } else {
            handle.next_check_frame = handle.next_check_frame.saturating_add(60);
            self.slots[handle.slot as usize].pos = position;
        }
    }

    /// Trigger an explosion shockwave bender (`GrassBendersAddExplosion`, `IGame_Persistent.cpp:423-439`).
    #[allow(clippy::too_many_arguments)]
    pub fn add_explosion(
        &mut self,
        id: u16,
        position: [f32; 3],
        dir: [f32; 3],
        fade: f32,
        speed: f32,
        intensity: f32,
        radius: f32,
        cfg: &GrassBendersConfig,
    ) {
        let limit = cfg.clamped_max_benders();
        if limit < 1 {
            return;
        }
        let exp_id = id.wrapping_add(99).max(1);
        for idx in 1..=limit {
            if self.slots[idx].anim != GrassBenderAnim::Explosion {
                self.set_slot(
                    idx,
                    exp_id,
                    position,
                    dir,
                    fade,
                    speed,
                    intensity,
                    radius,
                    GrassBenderAnim::Explosion,
                    true,
                );
                self.slots[idx].str_target = intensity;
                break;
            }
        }
    }

    /// Add or accumulate a muzzle/impact shot bender (`GrassBendersAddShot`, `IGame_Persistent.cpp:441-479`).
    #[allow(clippy::too_many_arguments)]
    pub fn add_shot(
        &mut self,
        id: u16,
        position: [f32; 3],
        camera_pos: [f32; 3],
        dir: [f32; 3],
        fade: f32,
        speed: f32,
        intensity: f32,
        radius: f32,
        cfg: &GrassBendersConfig,
    ) {
        let limit = cfg.clamped_max_benders();
        if limit < 1 || intensity <= 0.0 || id == 0 {
            return;
        }
        let dx = position[0] - camera_pos[0];
        let dz = position[2] - camera_pos[2];
        if dx * dx + dz * dz > cfg.max_distance_sq {
            return;
        }

        let mut add_at: Option<usize> = None;
        for idx in 1..=limit {
            if self.slots[idx].id == id {
                let current_str = self.slots[idx].str;
                self.set_slot(
                    idx,
                    id,
                    position,
                    dir,
                    fade,
                    speed,
                    current_str,
                    radius,
                    GrassBenderAnim::Explosion,
                    false,
                );
                self.slots[idx].str_target += intensity;
                add_at = None;
                break;
            } else if add_at.is_none() && self.slots[idx].radius == 0.0 {
                add_at = Some(idx);
            }
        }

        if let Some(idx) = add_at {
            self.set_slot(
                idx,
                id,
                position,
                dir,
                fade,
                speed,
                0.001,
                radius,
                GrassBenderAnim::Explosion,
                true,
            );
            self.slots[idx].str_target = intensity;
        }
    }

    /// Advance all active grass bender animations by `dt_sec` (`GrassBendersUpdateAnimations`, `IGame_Persistent.cpp:481-592`).
    pub fn update_animations(&mut self, dt_sec: f32, cfg: &GrassBendersConfig) {
        let dt = if dt_sec.is_finite() && dt_sec > 0.0 {
            dt_sec
        } else {
            0.0
        };
        let limit = cfg.clamped_max_benders();
        for idx in 1..=limit {
            if self.slots[idx].id == 0 {
                continue;
            }
            match self.slots[idx].anim {
                GrassBenderAnim::Explosion => {
                    let s = &mut self.slots[idx];
                    s.time += dt * s.speed;
                    s.radius_curr = s.radius * s.time.min(1.0);
                    s.str_target = s.str_target.min(1.0);

                    let diff = (s.str - s.str_target).abs().max(0.1);
                    if s.str_target <= s.str {
                        s.str -= dt * s.fade * diff;
                    } else {
                        s.str += dt * s.speed * diff;
                        if s.str >= s.str_target {
                            s.str_target = 0.0;
                        }
                    }

                    if s.str < 0.0 {
                        self.reset_slot(idx);
                    }
                }
                GrassBenderAnim::Wavy => {
                    let s = &mut self.slots[idx];
                    s.time += dt * 1.5 * s.speed;
                    let curve = s.time.sin();
                    s.str = curve * (curve * 1.4).cos() * 1.8 * s.str_target;
                }
                GrassBenderAnim::Suck => {
                    self.slots[idx].time += dt * self.slots[idx].speed;
                    let t = self.slots[idx].time;
                    let curve = -(self.perlin.get_continuous(t) + 0.5).clamp(0.0, 1.0);
                    self.slots[idx].str = curve * self.slots[idx].str_target;
                }
                GrassBenderAnim::Blow => {
                    self.slots[idx].time += dt * 1.2 * self.slots[idx].speed;
                    let t = self.slots[idx].time;
                    let curve = (self.perlin.get_continuous(t) + 1.0).clamp(0.0, 2.0) * 0.25;
                    self.slots[idx].str = curve * self.slots[idx].str_target;
                }
                GrassBenderAnim::Pulse => {
                    let s = &mut self.slots[idx];
                    s.time += dt * s.speed;
                    s.radius_curr = s.radius * s.time.min(1.0);
                    if s.radius_curr >= s.radius {
                        s.str = step_grass_bender_value(s.str, 0.0, s.speed * 0.6, true, dt);
                    }
                    if s.str <= 0.0 {
                        s.str = s.str_target;
                        s.radius_curr = 0.0;
                        s.time = 0.0;
                    }
                }
                GrassBenderAnim::Default => {
                    let s = &mut self.slots[idx];
                    s.str = step_grass_bender_value(s.str, s.str_target, 2.0, true, dt);
                }
            }
        }
    }

    /// Snapshot current bender positions/strengths into `prev_pos`/`prev_dir` (`device.cpp:459-468`).
    pub fn snapshot_prev_frame(&mut self, camera_pos: [f32; 3], cfg: &GrassBendersConfig) {
        self.slots[0].prev_pos = [camera_pos[0], camera_pos[1], camera_pos[2], -1.0];
        self.slots[0].prev_dir = [0.0, -99.0, 0.0, 1.0];
        let limit = cfg.clamped_max_benders();
        for idx in 1..=limit {
            let s = &mut self.slots[idx];
            s.prev_pos = [s.pos[0], s.pos[1], s.pos[2], s.radius_curr];
            s.prev_dir = [s.dir[0], s.dir[1], s.dir[2], s.str];
        }
    }

    /// Pack GPU constant arrays `(benders_pos, benders_prevpos, benders_setup)` matching `FTreeVisual.cpp:188-242`.
    pub fn pack_gpu_data(
        &self,
        camera_pos: [f32; 3],
        cfg: &GrassBendersConfig,
    ) -> GrassBendersGpuData {
        if cfg.max_benders == 0 {
            return GrassBendersGpuData {
                active_count: 0,
                benders_setup: cfg.params_1,
                ..GrassBendersGpuData::default()
            };
        }

        let limit = cfg.clamped_max_benders();
        let active_count = (limit + 1).min(GRASS_BENDERS_MAX_SLOTS);
        let mut out = GrassBendersGpuData {
            active_count,
            benders_setup: cfg.params_1,
            benders_pos: [[0.0; 4]; 32],
            benders_prevpos: [[0.0; 4]; 32],
        };

        out.benders_pos[0] = if cfg.player_enabled {
            [camera_pos[0], camera_pos[1], camera_pos[2], -1.0]
        } else {
            [0.0; 4]
        };
        out.benders_pos[16] = [0.0, -99.0, 0.0, 1.0];

        out.benders_prevpos[0] = self.slots[0].prev_pos;
        out.benders_prevpos[16] = self.slots[0].prev_dir;

        for idx in 1..active_count {
            let s = &self.slots[idx];
            out.benders_pos[idx] = [s.pos[0], s.pos[1], s.pos[2], s.radius_curr];
            out.benders_pos[idx + 16] = [s.dir[0], s.dir[1], s.dir[2], s.str];
            out.benders_prevpos[idx] = s.prev_pos;
            out.benders_prevpos[idx + 16] = s.prev_dir;
        }

        out
    }
}

#[inline]
fn step_grass_bender_value(
    current: f32,
    go_to: f32,
    intensity: f32,
    use_easing: bool,
    dt: f32,
) -> f32 {
    let diff = (current - go_to).abs();
    let step = dt * intensity * if use_easing { diff.min(0.5) } else { 1.0 };
    if diff - step <= 0.0 {
        go_to
    } else if current < go_to {
        current + step
    } else {
        current - step
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

    #[test]
    fn test_detail_swing_config_and_step() {
        let ltx = r#"
[details]
swing_normal_amp1 = 0.01
swing_normal_amp2 = 0.004
swing_normal_rot1 = 8.0
swing_normal_rot2 = 4.0
swing_normal_speed = 1.5
swing_fast_amp1 = 0.05
swing_fast_amp2 = 0.02
swing_fast_rot1 = 2.0
swing_fast_rot2 = 1.0
swing_fast_speed = 3.5
"#;
        let cfg = DetailSwingConfig::from_ltx_str(ltx);
        assert!((cfg.normal.amp1 - 0.01).abs() < 1e-6);
        assert!((cfg.fast.speed - 3.5).abs() < 1e-6);

        let mut state = DetailSwingState::default();
        let w1 = state.step(0.05, 0.5, &cfg, 0.3, 0.85);
        let expected_amp1 = 0.5 * (0.01 + 0.05);
        let expected_amp2 = 0.5 * (0.004 + 0.02);
        let len1 = (w1.dir1[0] * w1.dir1[0] + w1.dir1[2] * w1.dir1[2]).sqrt();
        let len2 = (w1.dir2[0] * w1.dir2[0] + w1.dir2[2] * w1.dir2[2]).sqrt();
        assert!((len1 - expected_amp1).abs() < 1e-6);
        assert!((len2 - expected_amp2).abs() < 1e-6);
        assert!((w1.consts[0] - 1.0 / DETAIL_HW_QUANT).abs() < 1e-8);
        assert!((w1.consts[2] - 0.3).abs() < 1e-6);
        assert!((w1.consts[3] - 0.85).abs() < 1e-6);

        let w2 = state.step(0.05, 0.5, &cfg, 0.3, 0.85);
        assert_eq!(w2.prev_dir1, w1.dir1);
        assert_eq!(w2.prev_dir2, w1.dir2);
        assert!((w2.prev_wave0[3] - w1.wave0[3]).abs() < 1e-7);
    }

    #[test]
    fn test_detail_dither_hw_batches_and_gpu_instance() {
        let dither = detail_bwdithermap(2);
        let min_v = dither.iter().flatten().copied().min().unwrap();
        let max_v = dither.iter().flatten().copied().max().unwrap();
        assert_eq!(min_v, 0);
        // `bwdithermap(2)` yields max value `floor(0.5 + 15 * 15.875 + (15/16) * 15.875) == 253`
        assert_eq!(max_v, 253);

        let r_ssa = compute_r_ssa_discard(1920, 1080, 90.0, 1.0, DETAIL_DEFAULT_PS_SSA_DISCARD);
        assert!((r_ssa - DETAIL_DEFAULT_SSA_DISCARD).abs() < 1e-7);

        let corners = [0.0_f32, 255.0, 255.0, 255.0];
        assert!((detail_interpolate(&corners, 0, 0, 4) - 0.0).abs() < 1e-6);
        assert!((detail_interpolate(&corners, 4, 4, 4) - 255.0).abs() < 1e-6);
        assert!(!detail_interpolate_and_dither(&[0.0; 4], 1, 1, 2, 3, 4, &dither));
        assert!(detail_interpolate_and_dither(&[255.0; 4], 1, 1, 2, 3, 4, &dither));

        let dm0 = DetailMesh::parse_dm(&build_dm_bytes("details\\blend", "detail\\g0", 0)).unwrap();
        let dm1 = DetailMesh::parse_dm(&build_dm_bytes("details\\still", "detail\\g1", DO_NO_WAVING))
            .unwrap();
        let hw = DetailMesh::build_hw_batches(&[dm0, dm1], 4).unwrap();
        assert_eq!(hw.batch_size, 4);
        assert_eq!(hw.vertices.len(), 24);
        assert_eq!(hw.indices.len(), 24);
        assert_eq!(hw.vertices[0].mid, 0);
        assert_eq!(hw.vertices[3].mid, 4);
        assert_eq!(hw.vertices[6].mid, 8);
        assert_eq!(hw.vertices[9].mid, 12);
        assert_eq!(hw.object_spans[0].vertex_offset, 0);
        assert_eq!(hw.object_spans[1].vertex_offset, 12);
        assert_eq!(hw.object_spans[1].index_offset, 12);

        let item = DetailSlotItem {
            scale: 1.0,
            scale_calculated: 0.75,
            xform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [4.0, 5.0, 6.0, 1.0],
            ],
            vis_id: 1,
            c_hemi: 0.5,
            c_sun: 0.8,
            c_rgb: [0.0; 3],
            distance_sq: 10.0,
            position: [4.0, 5.0, 6.0],
            normal: [0.0, 1.0, 0.0],
            alpha: 0.9,
            alpha_target: 1.0,
        };
        let gpu = item.to_gpu_instance(item.scale_calculated);
        assert_eq!(gpu.rows[0], [0.75, 0.0, 0.0, 4.0]);
        assert_eq!(gpu.rows[1], [0.0, 0.75, 0.0, 5.0]);
        assert_eq!(gpu.rows[2], [0.0, 0.0, 0.75, 6.0]);
        assert_eq!(gpu.rows[3], [0.8, 0.8, 0.8, 0.5]);
        assert_eq!(gpu.ex_data, [0.0, 1.0, 0.0, 0.9]);
    }

    #[test]
    fn test_detail_cache_grid_cform_alignment_shift_and_visibility() {
        use crate::gamemtl::{gamemtl_chunks, gamemtl_root_chunks, GameMaterialLibrary};
        use crate::hom::{frustum_planes, OccRasterizer};
        use crate::ogf::matrix_mul_44;

        // Build LevelDetails covering slot (0,0) with an object mesh
        let dm0 = DetailMesh::parse_dm(&build_dm_bytes("details\\blend", "detail\\g0", 0)).unwrap();
        let mut s = DetailSlot {
            ids: [0, DETAIL_ID_EMPTY, DETAIL_ID_EMPTY, DETAIL_ID_EMPTY],
            c_dir: 15,
            c_hemi: 10,
            y_base: 0,
            y_height: 0,
            c_r: 8,
            c_g: 8,
            c_b: 8,
            palette: [
                DetailPalette { a0: 15, a1: 15, a2: 15, a3: 15 },
                DetailPalette::default(),
                DetailPalette::default(),
                DetailPalette::default(),
            ],
        };
        s.pack_y(0.0, 5.0);
        let details = LevelDetails {
            header: DetailHeader {
                version: DETAIL_VERSION_4,
                object_count: 1,
                offs_x: 0,
                offs_z: 0,
                size_x: 1,
                size_z: 1,
            },
            objects: vec![dm0],
            slots: vec![s],
        };

        // Gamemtl library: idx 0 solid, idx 1 FL_PASSABLE
        let make_mtl = |id: u32, name: &str, flags: u32| {
            let mut mw = ChunkWriter::new();
            let mut main = id.to_le_bytes().to_vec();
            main.extend_from_slice(name.as_bytes());
            main.push(0);
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_MAIN, &main);
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FLAGS, &flags.to_le_bytes());
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_PHYSICS, &[0u8; 20]);
            mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FACTORS, &[0u8; 16]);
            mw.into_bytes()
        };
        let mut mtls = ChunkWriter::new();
        mtls.write_chunk(0, &make_mtl(0, "ground", gamemtl_flags::FL_SHOOTABLE));
        mtls.write_chunk(1, &make_mtl(1, "bush", gamemtl_flags::FL_PASSABLE));
        let mut root = ChunkWriter::new();
        root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_VERSION, &1u16.to_le_bytes());
        root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_AUTOINC, &[0u8; 8]);
        root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_MTLS, &mtls.into_bytes());
        let gmlib = GameMaterialLibrary::parse(&root.into_bytes()).unwrap();

        // Sloped solid floor quad (slope y = 0.5 + 0.5*x) plus higher passable cover quad at y = 4.0
        // Note: downward ray (0,-1,0) hits p0=(0,0.5,0), p1=(0,0.5,2), p2=(2,1.5,0) with +Y-up normal.
        let cform = CformModel::from_parts(
            vec![
                [-0.5, 0.25, -0.5],
                [-0.5, 0.25, 2.5],
                [2.5, 1.75, 2.5],
                [2.5, 1.75, -0.5],
                // Passable overhead quad at y=4.0
                [-0.5, 4.0, -0.5],
                [-0.5, 4.0, 2.5],
                [2.5, 4.0, 2.5],
                [2.5, 4.0, -0.5],
            ],
            vec![
                crate::cform::CformTriangle {
                    verts: [0, 1, 2],
                    material: 0,
                    suppress_shadows: false,
                    suppress_wm: false,
                    sector: 0,
                },
                crate::cform::CformTriangle {
                    verts: [0, 2, 3],
                    material: 0,
                    suppress_shadows: false,
                    suppress_wm: false,
                    sector: 0,
                },
                crate::cform::CformTriangle {
                    verts: [4, 5, 6],
                    material: 1,
                    suppress_shadows: false,
                    suppress_wm: false,
                    sector: 0,
                },
                crate::cform::CformTriangle {
                    verts: [4, 6, 7],
                    material: 1,
                    suppress_shadows: false,
                    suppress_wm: false,
                    sector: 0,
                },
            ],
        )
        .unwrap();
        let bvh = cform.build_bvh();

        let mut grid = DetailCacheGrid::new(&details, 8);
        assert!(grid.validate_coords());
        assert_eq!(LevelDetails::world_to_cache_center(1.2, -0.8), (1, 0));

        let cfg = DetailDecompressConfig {
            density: 0.6,
            detail_height: 1.0,
            terrain_grass_align: true,
            terrain_grass_slope: 1.0,
            no_random: false,
            use_vs: true,
        };
        grid.update_cache_with_cform(
            &details,
            0,
            0,
            [1.0, 1.5, -3.0],
            grid.cache_size(),
            &cfg,
            &bvh,
            &cform,
            Some(&gmlib),
        );
        assert_eq!(grid.pending_count(), 0);

        let slot = grid.query_slot_rel(0, 0).unwrap();
        assert!(!slot.empty);
        let expected_items = slot.parts[0].items.len();
        assert!(expected_items > 0);
        for item in &slot.parts[0].items {
            // Must land on sloped terrain (y ~ 0.5 + 0.5*x), NOT on y=4.0 passable roof
            assert!(item.position[1] < 2.0, "expected terrain height, got {}", item.position[1]);
            let expected_y = 0.5 + 0.5 * item.position[0];
            assert!((item.position[1] - expected_y).abs() < 1e-4);
            // Aligned row 1 (m.j) must equal terrain normal (-1/sqrt(5), 2/sqrt(5), 0)
            let inv_sqrt5 = 1.0 / 5.0_f32.sqrt();
            assert!((item.normal[0] + inv_sqrt5).abs() < 1e-4);
            assert!((item.normal[1] - 2.0 * inv_sqrt5).abs() < 1e-4);
            assert!((item.xform[1][0] - item.normal[0]).abs() < 1e-5);
            assert!((item.xform[1][1] - item.normal[1]).abs() < 1e-5);
            assert!((item.xform[1][2] - item.normal[2]).abs() < 1e-5);
        }

        // Collect visible from front camera -> should emit items
        let cam_pos = [1.0, 1.5, -3.0];
        let view = crate::camera::matrix_build_camera(cam_pos, [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]);
        let proj = crate::camera::matrix_build_projection(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);
        let full = matrix_mul_44(&proj, &view);
        let frustum = ClipFrustum::from_view_proj(&full, frustum_planes::LRTB | frustum_planes::FAR);

        let vis = grid.collect_visible(cam_pos, &frustum, None, &full, 1, &DetailVisibilityConfig::default());
        let total_vis = vis.still[0].len() + vis.wave1[0].len() + vis.wave2[0].len();
        assert_eq!(total_vis, expected_items);

        // Camera looking away -> frustum culls all
        let back_view = crate::camera::matrix_build_camera(cam_pos, [1.0, 1.0, -10.0], [0.0, 1.0, 0.0]);
        let back_full = matrix_mul_44(&proj, &back_view);
        let back_frustum =
            ClipFrustum::from_view_proj(&back_full, frustum_planes::LRTB | frustum_planes::FAR);
        let vis_back = grid.collect_visible(
            cam_pos,
            &back_frustum,
            None,
            &back_full,
            2,
            &DetailVisibilityConfig::default(),
        );
        assert_eq!(
            vis_back.still[0].len() + vis_back.wave1[0].len() + vis_back.wave2[0].len(),
            0
        );

        // HOM occlusion wall directly between cam_pos and slot -> hides all
        let wall_mesh = crate::hom::HomMesh::from_raw_triangles(&[
            ([-5.0, -5.0, -1.0], [-5.0, 5.0, -1.0], [5.0, 5.0, -1.0], 1),
            ([-5.0, -5.0, -1.0], [5.0, 5.0, -1.0], [5.0, -5.0, -1.0], 1),
        ]);
        let mut occ = OccRasterizer::new();
        occ.render_mesh(&wall_mesh, &full, cam_pos);
        let vis_hom = grid.collect_visible(
            cam_pos,
            &frustum,
            Some(&occ),
            &full,
            3,
            &DetailVisibilityConfig::default(),
        );
        assert_eq!(
            vis_hom.still[0].len() + vis_hom.wave1[0].len() + vis_hom.wave2[0].len(),
            0
        );

        // Shift grid far away -> validate_coords still exact
        grid.update_cache_with_cform(&details, 3, -2, [6.0, 1.0, -4.0], 4, &cfg, &bvh, &cform, Some(&gmlib));
        assert!(grid.validate_coords());
        assert_eq!(grid.center(), (3, -2));
    }

    #[test]
    fn test_ssfx_grass_bender_pool_lifecycle_animations_and_gpu_pack() {
        use crate::hom::frustum_planes;
        use crate::ogf::matrix_mul_44;

        let mut pool = GrassBenderPool::new(77);
        let cfg = GrassBendersConfig {
            player_enabled: true,
            max_benders: 4,
            max_distance_sq: 100.0,
            ..GrassBendersConfig::default()
        };

        let cam_pos = [0.0, 1.5, -5.0];
        let view = crate::camera::matrix_build_camera(cam_pos, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let proj =
            crate::camera::matrix_build_projection(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);
        let full = matrix_mul_44(&proj, &view);
        let frustum =
            ClipFrustum::from_view_proj(&full, frustum_planes::LRTB | frustum_planes::FAR);

        // Claim character slot 1 in front of camera
        let mut handle = GrassBenderHandle::default();
        pool.update_character(
            42,
            &mut handle,
            0,
            [0.0, 0.0, 0.0],
            cam_pos,
            &frustum,
            1.25,
            0.8,
            true,
            &cfg,
        );
        assert_eq!(handle.slot, 1);
        assert_eq!(pool.slots[1].id, 42);
        assert_eq!(pool.slots[1].radius_curr, 1.25);

        // Explosion scans `1..=limit` for first non-Explosion slot, overtaking Default slot 1
        pool.add_explosion(10, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 2.0, 4.0, 0.9, 3.0, &cfg);
        assert_eq!(pool.slots[1].id, 109);
        assert_eq!(pool.slots[1].anim, GrassBenderAnim::Explosion);

        // Character detects its old slot was overtaken, resets handle, then re-claims slot 2
        let check_evicted = handle.next_check_frame;
        pool.update_character(
            42,
            &mut handle,
            check_evicted,
            [0.0, 0.0, 0.0],
            cam_pos,
            &frustum,
            1.25,
            0.8,
            true,
            &cfg,
        );
        assert_eq!(handle.slot, 0);
        let check_reclaim = handle.next_check_frame;
        pool.update_character(
            42,
            &mut handle,
            check_reclaim,
            [0.0, 0.0, 0.0],
            cam_pos,
            &frustum,
            1.25,
            0.8,
            true,
            &cfg,
        );
        assert_eq!(handle.slot, 2);
        assert_eq!(pool.slots[2].id, 42);

        // Add shot twice with same id -> claims slot 3 and accumulates target intensity
        pool.add_shot(55, [2.0, 0.0, 0.0], cam_pos, [1.0, 0.0, 0.0], 1.5, 5.0, 0.4, 2.0, &cfg);
        assert_eq!(pool.slots[3].id, 55);
        pool.add_shot(55, [2.0, 0.0, 0.0], cam_pos, [1.0, 0.0, 0.0], 1.5, 5.0, 0.3, 2.0, &cfg);
        assert!((pool.slots[3].str_target - 0.7).abs() < 1e-6);

        // Perlin Suck animation in slot 4 remains finite & non-positive
        pool.set_slot(
            4,
            88,
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            1.0,
            2.0,
            0.0,
            1.5,
            GrassBenderAnim::Suck,
            true,
        );
        pool.slots[4].str_target = 1.0;

        // Snapshot previous frame, advance animations, and pack GPU constant buffers
        pool.snapshot_prev_frame(cam_pos, &cfg);
        pool.update_animations(0.1, &cfg);
        assert!((pool.slots[1].radius_curr - 3.0 * 0.4).abs() < 1e-5);
        assert!(pool.slots[2].str > 0.0 && pool.slots[2].str <= 0.8);
        assert!(pool.slots[4].str <= 0.0 && pool.slots[4].str >= -1.0);

        let gpu = pool.pack_gpu_data(cam_pos, &cfg);
        assert_eq!(gpu.active_count, 5);
        assert_eq!(gpu.benders_pos[0], [0.0, 1.5, -5.0, -1.0]);
        assert_eq!(gpu.benders_pos[16], [0.0, -99.0, 0.0, 1.0]);
        assert_eq!(gpu.benders_prevpos[0], [0.0, 1.5, -5.0, -1.0]);
        assert!((gpu.benders_pos[1][3] - 1.2).abs() < 1e-5);
        assert_eq!(gpu.benders_prevpos[2][3], 1.25);

        // Distance eviction cleans character's slot 2
        let check_far = handle.next_check_frame;
        pool.update_character(
            42,
            &mut handle,
            check_far,
            [50.0, 0.0, 0.0],
            cam_pos,
            &frustum,
            1.25,
            0.8,
            true,
            &cfg,
        );
        assert_eq!(handle.slot, 0);
        assert_eq!(pool.slots[2].id, 0);

        // Verify Wavy, Blow, and Pulse animation variants advance and reset cleanly
        let mut anim_pool = GrassBenderPool::new(12);
        anim_pool.set_slot(1, 1, [0.0; 3], [0.0, 1.0, 0.0], 1.0, 1.0, 0.0, 1.0, GrassBenderAnim::Wavy, true);
        anim_pool.slots[1].str_target = 0.5;
        anim_pool.set_slot(2, 2, [0.0; 3], [0.0, 1.0, 0.0], 1.0, 1.0, 0.0, 1.0, GrassBenderAnim::Blow, true);
        anim_pool.slots[2].str_target = 0.8;
        anim_pool.set_slot(3, 3, [0.0; 3], [0.0, 1.0, 0.0], 1.0, 2.0, 0.6, 1.0, GrassBenderAnim::Pulse, true);
        anim_pool.slots[3].str_target = 0.6;
        anim_pool.update_animations(0.25, &cfg);
        assert!(anim_pool.slots[1].str.is_finite() && anim_pool.slots[1].str > 0.0);
        assert!(anim_pool.slots[2].str >= 0.0 && anim_pool.slots[2].str <= 0.4 + 1e-6);
        assert!((anim_pool.slots[3].radius_curr - 0.5).abs() < 1e-5);
        anim_pool.update_animations(0.5, &cfg);
        anim_pool.update_animations(1.0, &cfg);
        assert!(anim_pool.slots[3].str >= 0.0 && anim_pool.slots[3].str.is_finite());

        let mut fade_item = DetailSlotItem {
            scale: 1.0,
            scale_calculated: 1.0,
            xform: crate::ogf::XRAY_IDENTITY_MATRIX,
            vis_id: 0,
            c_hemi: 1.0,
            c_sun: 0.5,
            c_rgb: [1.0; 3],
            distance_sq: 1.0,
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            alpha: 0.0,
            alpha_target: 1.0,
        };
        assert!((fade_item.step_alpha(0.4) - 0.4).abs() < 1e-6);
        assert!((fade_item.step_alpha(0.8) - 1.0).abs() < 1e-6);
        fade_item.alpha_target = 0.0;
        assert!((fade_item.step_alpha(0.25) - 0.75).abs() < 1e-6);
    }
}
