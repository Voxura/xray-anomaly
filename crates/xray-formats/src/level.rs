//! Pure Rust X-Ray `level` & `level.geom` / `level.geomx` World Scene Parser
//!
//! Faithful pure-Rust implementation matching `src/xrEngine/xrLevel.h`,
//! `src/Layers/xrRenderPC_R4/r4_loader.cpp`, `src/Layers/xrRender/r__sector.cpp`,
//! `src/Layers/xrRender/FVisual.cpp`, and `src/Layers/xrRender/Light_DB.cpp`.

use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::lanims::LightAnimItem;
use crate::ogf::{OgfModel, OgfSlideWindows, SlideWindow, StaticVertex};
use xray_core::ltx_loader::OwnedLtxDocument;

/// D3D9 `D3DLIGHTTYPE` constants used in `fsL_LIGHT_DYNAMIC` (`LevelLight::light_type`).
pub mod d3d_light_type {
    pub const POINT: u32 = 1;
    pub const SPOT: u32 = 2;
    pub const DIRECTIONAL: u32 = 3;
}

/// Compiler `R_Light::type` constants used in `$level$\build.lights` (`R_light.h`).
pub mod build_light_type {
    pub const DIRECT: u16 = 0;
    pub const POINT: u16 = 1;
    pub const SECONDARY: u16 = 2;
}

/// Wire byte stride of each static glow entry inside `fsL_GLOWS` (`CGlowManager::Load`).
pub const LEVEL_GLOW_RECORD_SIZE: usize = 18;

/// Wire byte stride of each `R_Light` entry inside `build.lights` (`sizeof(R_Light)`).
pub const BUILD_LIGHT_RECORD_SIZE: usize = 104;

/// Top-level chunks inside `level` and `level.geom`/`level.geomx` (`enum fsL_Chunks`).
pub mod level_chunks {
    pub const FSL_HEADER: u32 = 1;
    pub const FSL_SHADERS: u32 = 2;
    pub const FSL_VISUALS: u32 = 3;
    pub const FSL_PORTALS: u32 = 4;
    pub const FSL_LIGHT_DYNAMIC: u32 = 6;
    pub const FSL_GLOWS: u32 = 7;
    pub const FSL_SECTORS: u32 = 8;
    pub const FSL_VB: u32 = 9;
    pub const FSL_IB: u32 = 10;
    pub const FSL_SWIS: u32 = 11;
}

/// Child chunks inside each `fsL_SECTORS` entry (`enum fsESectorChunks`).
pub mod sector_chunks {
    pub const FSP_PORTALS: u32 = 1;
    pub const FSP_ROOT: u32 = 2;
}

/// `D3DDECLTYPE` constants stored in `D3DVERTEXELEMENT9::Type`.
pub mod d3d_decl_type {
    pub const FLOAT1: u8 = 0;
    pub const FLOAT2: u8 = 1;
    pub const FLOAT3: u8 = 2;
    pub const FLOAT4: u8 = 3;
    pub const D3DCOLOR: u8 = 4;
    pub const UBYTE4: u8 = 5;
    pub const SHORT2: u8 = 6;
    pub const SHORT4: u8 = 7;
    pub const UBYTE4N: u8 = 8;
    pub const SHORT2N: u8 = 9;
    pub const SHORT4N: u8 = 10;
    pub const USHORT2N: u8 = 11;
    pub const USHORT4N: u8 = 12;
    pub const UDEC3: u8 = 13;
    pub const DEC3N: u8 = 14;
    pub const FLOAT16_2: u8 = 15;
    pub const FLOAT16_4: u8 = 16;
    pub const UNUSED: u8 = 17;
}

/// `D3DDECLUSAGE` constants stored in `D3DVERTEXELEMENT9::Usage`.
pub mod d3d_decl_usage {
    pub const POSITION: u8 = 0;
    pub const BLENDWEIGHT: u8 = 1;
    pub const BLENDINDICES: u8 = 2;
    pub const NORMAL: u8 = 3;
    pub const PSIZE: u8 = 4;
    pub const TEXCOORD: u8 = 5;
    pub const TANGENT: u8 = 6;
    pub const BINORMAL: u8 = 7;
    pub const TESSFACTOR: u8 = 8;
    pub const POSITIONT: u8 = 9;
    pub const COLOR: u8 = 10;
    pub const FOG: u8 = 11;
    pub const DEPTH: u8 = 12;
    pub const SAMPLE: u8 = 13;
}

/// 8-byte wire `D3DVERTEXELEMENT9` vertex declarator attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexElement9 {
    pub stream: u16,
    pub offset: u16,
    pub decl_type: u8,
    pub method: u8,
    pub usage: u8,
    pub usage_index: u8,
}

impl VertexElement9 {
    pub const END: Self = Self {
        stream: 0x00FF,
        offset: 0,
        decl_type: d3d_decl_type::UNUSED,
        method: 0,
        usage: 0,
        usage_index: 0,
    };

    #[inline]
    pub fn is_end(self) -> bool {
        self.stream == 0x00FF || self.decl_type == d3d_decl_type::UNUSED
    }

    #[inline]
    pub fn type_byte_size(decl_type: u8) -> Result<usize, FormatError> {
        use d3d_decl_type::*;
        match decl_type {
            FLOAT1 | D3DCOLOR | UBYTE4 | SHORT2 | UBYTE4N | SHORT2N | USHORT2N | UDEC3 | DEC3N
            | FLOAT16_2 => Ok(4),
            FLOAT2 | SHORT4 | SHORT4N | USHORT4N | FLOAT16_4 => Ok(8),
            FLOAT3 => Ok(12),
            FLOAT4 => Ok(16),
            UNUSED => Ok(0),
            _ => Err(FormatError::InvalidValue("unknown D3DDECLTYPE in vertex declarator")),
        }
    }
}

/// Complete vertex declaration ending with `D3DDECL_END()` plus stream-0 stride.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexDeclarator {
    pub elements: Vec<VertexElement9>,
    pub stride: usize,
}

impl VertexDeclarator {
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let mut elements = Vec::new();
        let mut stride = 0usize;

        loop {
            let elem = VertexElement9 {
                stream: cur.read_u16()?,
                offset: cur.read_u16()?,
                decl_type: cur.read_u8()?,
                method: cur.read_u8()?,
                usage: cur.read_u8()?,
                usage_index: cur.read_u8()?,
            };
            if elem.is_end() {
                break;
            }
            if elem.stream == 0 {
                let sz = VertexElement9::type_byte_size(elem.decl_type)?;
                let end = (elem.offset as usize)
                    .checked_add(sz)
                    .ok_or(FormatError::InvalidValue("D3DVERTEXELEMENT9 offset overflow"))?;
                stride = stride.max(end);
            }
            elements.push(elem);
            if elements.len() > 64 {
                return Err(FormatError::InvalidValue("vertex declarator exceeds 64 elements"));
            }
        }

        Ok(Self { elements, stride })
    }
}

/// Shared static level vertex buffer from `level.geom` / `level.geomx` (`fsL_VB`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelVertexBuffer {
    pub declarator: VertexDeclarator,
    pub vertex_count: u32,
    pub raw_bytes: Vec<u8>,
}

impl LevelVertexBuffer {
    /// Decode a slice `[base .. base + count)` into normalized `StaticVertex` attributes,
    /// handling both full floats and X-Ray packed `D3DCOLOR` normals / `SHORT2`/`SHORT4` UVs /
    /// `FLOAT16` attributes.
    pub fn decode_static_vertices(
        &self,
        base: u32,
        count: u32,
    ) -> Result<Vec<StaticVertex>, FormatError> {
        let end = (base as usize)
            .checked_add(count as usize)
            .ok_or(FormatError::InvalidValue("VB slice overflow"))?;
        if end > self.vertex_count as usize || self.declarator.stride == 0 {
            return Err(FormatError::InvalidValue("VB slice exceeds vertex buffer bounds"));
        }

        let stride = self.declarator.stride;
        let mut out = Vec::with_capacity(count as usize);

        for v_idx in (base as usize)..end {
            let v_bytes = &self.raw_bytes[v_idx * stride..(v_idx + 1) * stride];
            let mut vertex = StaticVertex {
                position: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
                diffuse: 0xFFFF_FFFF,
                specular: 0,
                uv0: [0.0; 2],
                uv1: [0.0; 2],
            };

            for elem in &self.declarator.elements {
                if elem.stream != 0 {
                    continue;
                }
                let off = elem.offset as usize;
                let sz = VertexElement9::type_byte_size(elem.decl_type)?;
                if off + sz > v_bytes.len() {
                    return Err(FormatError::UnexpectedEof);
                }
                let attr = &v_bytes[off..off + sz];

                match (elem.usage, elem.usage_index) {
                    (d3d_decl_usage::POSITION, 0) | (d3d_decl_usage::POSITIONT, 0) => {
                        vertex.position = decode_vec3_attr(elem.decl_type, attr)?;
                    }
                    (d3d_decl_usage::NORMAL, 0) => {
                        vertex.normal = decode_normal_attr(elem.decl_type, attr)?;
                    }
                    (d3d_decl_usage::COLOR, 0) => {
                        if attr.len() >= 4 {
                            vertex.diffuse = u32::from_le_bytes([attr[0], attr[1], attr[2], attr[3]]);
                        }
                    }
                    (d3d_decl_usage::TEXCOORD, 0) => {
                        vertex.uv0 = decode_uv_attr(elem.decl_type, attr, 1.0 / 1024.0)?;
                    }
                    (d3d_decl_usage::TEXCOORD, 1) => {
                        vertex.uv1 = decode_uv_attr(elem.decl_type, attr, 1.0 / 32768.0)?;
                    }
                    _ => {}
                }
            }

            out.push(vertex);
        }

        Ok(out)
    }
}

/// Shared level geometry buffers parsed from `level.geom` or `level.geomx`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LevelGeometry {
    pub vertex_buffers: Vec<LevelVertexBuffer>,
    pub index_buffers: Vec<Vec<u16>>,
    pub swis: Vec<OgfSlideWindows>,
}

impl LevelGeometry {
    /// Parse `level.geom` or `level.geomx` (`CRender::LoadBuffers` + `LoadSWIs`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        use level_chunks::*;
        let reader = ChunkReader::new(data);

        let vb_bytes = reader.require_chunk(FSL_VB)?;
        let mut vc = ByteCursor::new(&vb_bytes);
        let vb_count = vc.read_u32()? as usize;
        let mut vertex_buffers = Vec::with_capacity(vb_count);
        for _ in 0..vb_count {
            let declarator = VertexDeclarator::parse(&mut vc)?;
            let vertex_count = vc.read_u32()?;
            let byte_len = (vertex_count as usize)
                .checked_mul(declarator.stride)
                .ok_or(FormatError::UnexpectedEof)?;
            let raw_bytes = vc.read_bytes(byte_len)?.to_vec();
            vertex_buffers.push(LevelVertexBuffer {
                declarator,
                vertex_count,
                raw_bytes,
            });
        }

        let mut index_buffers = Vec::new();
        if let Some(ib_bytes) = reader.open_chunk(FSL_IB)? {
            let mut ic = ByteCursor::new(&ib_bytes);
            let ib_count = ic.read_u32()? as usize;
            index_buffers.reserve(ib_count);
            for _ in 0..ib_count {
                let idx_count = ic.read_u32()? as usize;
                let need = idx_count
                    .checked_mul(2)
                    .ok_or(FormatError::UnexpectedEof)?;
                let raw = ic.read_bytes(need)?;
                let mut indices = Vec::with_capacity(idx_count);
                for pair in raw.as_chunks::<2>().0 {
                    indices.push(u16::from_le_bytes(*pair));
                }
                index_buffers.push(indices);
            }
        }

        let mut swis = Vec::new();
        if let Some(swi_bytes) = reader.open_chunk(FSL_SWIS)? {
            let mut sc = ByteCursor::new(&swi_bytes);
            let item_count = sc.read_u32()? as usize;
            swis.reserve(item_count);
            for _ in 0..item_count {
                let reserved = [
                    sc.read_u32()?,
                    sc.read_u32()?,
                    sc.read_u32()?,
                    sc.read_u32()?,
                ];
                let count = sc.read_u32()? as usize;
                let need = count
                    .checked_mul(8)
                    .ok_or(FormatError::UnexpectedEof)?;
                let raw = sc.read_bytes(need)?;
                let mut windows = Vec::with_capacity(count);
                for rec in raw.as_chunks::<8>().0 {
                    let mut rc = ByteCursor::new(rec);
                    windows.push(SlideWindow {
                        offset: rc.read_u32()?,
                        num_tris: rc.read_u16()?,
                        num_verts: rc.read_u16()?,
                    });
                }
                swis.push(OgfSlideWindows { reserved, windows });
            }
        }

        Ok(Self {
            vertex_buffers,
            index_buffers,
            swis,
        })
    }

    pub fn slice_vertices(
        &self,
        vb_id: u32,
        vb_base: u32,
        vb_count: u32,
    ) -> Result<Vec<StaticVertex>, FormatError> {
        let vb = self
            .vertex_buffers
            .get(vb_id as usize)
            .ok_or(FormatError::InvalidValue("OGF_GCONTAINER vb_id out of range"))?;
        vb.decode_static_vertices(vb_base, vb_count)
    }

    pub fn slice_indices(
        &self,
        ib_id: u32,
        ib_base: u32,
        ib_count: u32,
    ) -> Result<&[u16], FormatError> {
        let ib = self
            .index_buffers
            .get(ib_id as usize)
            .ok_or(FormatError::InvalidValue("OGF_GCONTAINER ib_id out of range"))?;
        let end = (ib_base as usize)
            .checked_add(ib_count as usize)
            .ok_or(FormatError::InvalidValue("IB slice overflow"))?;
        ib.get((ib_base as usize)..end)
            .ok_or(FormatError::InvalidValue("OGF_GCONTAINER index slice out of bounds"))
    }
}

/// Shader/texture palette entry from `level` chunk `fsL_SHADERS`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelShaderRef {
    pub raw: String,
    pub shader: String,
    pub textures: Vec<String>,
}

impl LevelShaderRef {
    pub fn parse(raw: String) -> Self {
        if raw.is_empty() {
            return Self {
                raw,
                shader: String::new(),
                textures: Vec::new(),
            };
        }
        let (shader, tlist) = match raw.split_once('/') {
            Some((s, t)) => (s.to_string(), t),
            None => (raw.clone(), ""),
        };
        let textures = tlist
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        Self {
            raw,
            shader,
            textures,
        }
    }
}

/// Sector-connecting portal polygon (`struct b_portal`, 80 bytes on disk).
#[derive(Debug, Clone, PartialEq)]
pub struct LevelPortal {
    pub sector_front: u16,
    pub sector_back: u16,
    pub vertices: Vec<[f32; 3]>,
}

impl LevelPortal {
    /// Compute the portal's enclosing sphere `(center, radius)` (`CPortal::Setup`).
    pub fn bounding_sphere(&self) -> Option<([f32; 3], f32)> {
        if self.vertices.is_empty() {
            return None;
        }
        let mut min = self.vertices[0];
        let mut max = self.vertices[0];
        for v in &self.vertices[1..] {
            for i in 0..3 {
                min[i] = min[i].min(v[i]);
                max[i] = max[i].max(v[i]);
            }
        }
        let center = [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ];
        let mut max_r2 = 0.0f32;
        for v in &self.vertices {
            let dx = v[0] - center[0];
            let dy = v[1] - center[1];
            let dz = v[2] - center[2];
            max_r2 = max_r2.max(dx * dx + dy * dy + dz * dz);
        }
        Some((center, max_r2.sqrt()))
    }

    /// Compute the portal's averaged unit normal and signed plane distance `(normal, d)`
    /// such that `dot(normal, p) + d = signed_distance` (`CPortal::Setup`).
    pub fn plane(&self) -> Option<([f32; 3], f32)> {
        if self.vertices.len() < 3 {
            return None;
        }
        let v0 = self.vertices[0];
        let mut accum = [0.0f32; 3];
        let mut count = 0u32;
        for i in 2..self.vertices.len() {
            let v1 = self.vertices[i - 1];
            let v2 = self.vertices[i];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-6 {
                accum[0] += n[0] / len;
                accum[1] += n[1] / len;
                accum[2] += n[2] / len;
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let mag = (accum[0] * accum[0] + accum[1] * accum[1] + accum[2] * accum[2]).sqrt();
        if mag <= 1e-6 {
            return None;
        }
        let normal = [accum[0] / mag, accum[1] / mag, accum[2] / mag];
        let d = -(normal[0] * v0[0] + normal[1] * v0[1] + normal[2] * v0[2]);
        Some((normal, d))
    }

    /// Signed distance from `point` to the portal plane (positive = front side).
    pub fn signed_distance(&self, point: [f32; 3]) -> Option<f32> {
        let (n, d) = self.plane()?;
        Some(n[0] * point[0] + n[1] * point[1] + n[2] * point[2] + d)
    }

    /// Sector on the opposite side of this portal from `from_sector`.
    #[inline]
    pub fn opposite_sector(&self, from_sector: u16) -> Option<u16> {
        if from_sector == self.sector_front {
            Some(self.sector_back)
        } else if from_sector == self.sector_back {
            Some(self.sector_front)
        } else {
            None
        }
    }

    /// Sector facing `view_point` (`CPortal::getSectorFacing`), or `None` if plane degenerate.
    #[inline]
    pub fn sector_facing(&self, view_point: [f32; 3]) -> Option<u16> {
        self.signed_distance(view_point)
            .map(|sd| if sd > 0.0 { self.sector_front } else { self.sector_back })
    }

    /// Sector behind the portal relative to `view_point` (`CPortal::getSectorBack`).
    #[inline]
    pub fn sector_behind(&self, view_point: [f32; 3]) -> Option<u16> {
        self.signed_distance(view_point)
            .map(|sd| if sd > 0.0 { self.sector_back } else { self.sector_front })
    }
}

/// Occlusion sector from `level` chunk `fsL_SECTORS` (`CSector`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelSector {
    pub id: u32,
    pub portals: Vec<u16>,
    pub root_visual_id: Option<u32>,
}

/// Static glow corona record (`18` bytes in `fsL_GLOWS`, `CGlowManager::Load`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelGlow {
    pub position: [f32; 3],
    pub radius: f32,
    pub shader_id: u16,
}

impl LevelGlow {
    /// Parse all `18`-byte glow entries inside `fsL_GLOWS`.
    pub fn parse_list(bytes: &[u8]) -> Result<Vec<Self>, FormatError> {
        if !bytes.len().is_multiple_of(LEVEL_GLOW_RECORD_SIZE) {
            return Err(FormatError::InvalidValue(
                "fsL_GLOWS size must be a multiple of 18 bytes",
            ));
        }
        let mut out = Vec::with_capacity(bytes.len() / LEVEL_GLOW_RECORD_SIZE);
        for rec in bytes.as_chunks::<LEVEL_GLOW_RECORD_SIZE>().0 {
            let mut c = ByteCursor::new(rec);
            out.push(Self {
                position: c.read_f32_3()?,
                radius: c.read_f32()?,
                shader_id: c.read_u16()?,
            });
        }
        Ok(out)
    }
}

/// Static/dynamic level light record (`u32 controller + Flight`, 108 bytes in `fsL_LIGHT_DYNAMIC`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelLight {
    pub controller_id: u32,
    pub light_type: u32,
    pub diffuse: [f32; 4],
    pub specular: [f32; 4],
    pub ambient: [f32; 4],
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub range: f32,
    pub falloff: f32,
    pub attenuation: [f32; 3],
    pub theta: f32,
    pub phi: f32,
}

impl LevelLight {
    #[inline]
    pub fn is_point(&self) -> bool {
        self.light_type == d3d_light_type::POINT
    }

    #[inline]
    pub fn is_spot(&self) -> bool {
        self.light_type == d3d_light_type::SPOT
    }

    #[inline]
    pub fn is_directional(&self) -> bool {
        self.light_type == d3d_light_type::DIRECTIONAL
    }

    /// Safe unit direction vector (falls back to `(0, -1, 0)` when zero-length).
    #[inline]
    pub fn normalized_direction(&self) -> [f32; 3] {
        let len_sq = self.direction[0] * self.direction[0]
            + self.direction[1] * self.direction[1]
            + self.direction[2] * self.direction[2];
        if len_sq <= 1e-12 {
            return [0.0, -1.0, 0.0];
        }
        let inv = 1.0 / len_sq.sqrt();
        [
            self.direction[0] * inv,
            self.direction[1] * inv,
            self.direction[2] * inv,
        ]
    }

    /// Minimal enclosing sphere `(center, radius)` matching `light::spatial_move()` (`light.cpp:205-230`).
    pub fn bounding_sphere(&self) -> ([f32; 3], f32) {
        let r = self.range.max(0.0);
        if self.is_spot() {
            let dir = self.normalized_direction();
            let cone = self.phi.max(self.theta).clamp(0.0, 120.0f32.to_radians());
            if cone >= std::f32::consts::FRAC_PI_2 {
                (
                    [
                        self.position[0] + dir[0] * r,
                        self.position[1] + dir[1] * r,
                        self.position[2] + dir[2] * r,
                    ],
                    r * (0.5 * cone).tan(),
                )
            } else {
                let half_cos = (0.5 * cone).cos().max(1e-4);
                let sr = r / (2.0 * half_cos * half_cos);
                (
                    [
                        self.position[0] + dir[0] * sr,
                        self.position[1] + dir[1] * sr,
                        self.position[2] + dir[2] * sr,
                    ],
                    sr,
                )
            }
        } else {
            (self.position, r)
        }
    }

    /// Distance attenuation factor matching `LightTrack.cpp:238` (`1 / (a0 + a1*d + a2*d^2) - d*falloff`),
    /// with smooth linear `(1 - d/range)` fallback when D3D attenuation constants are all zero.
    pub fn evaluate_attenuation(&self, distance: f32) -> f32 {
        if self.is_directional() {
            return 1.0;
        }
        if distance < 0.0 {
            return 0.0;
        }
        if self.range > 0.0 && distance >= self.range {
            return 0.0;
        }
        let denom = self.attenuation[0]
            + self.attenuation[1] * distance
            + self.attenuation[2] * distance * distance;
        if denom > 1e-6 {
            (1.0 / denom - distance * self.falloff).max(0.0)
        } else if self.range > 1e-6 {
            (1.0 - distance / self.range).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Inner/outer spot cone half-angle cosines `[cos(theta / 2), cos(phi / 2)]`
    /// matching D3D9 / GPU `spot_params`.
    #[inline]
    pub fn spot_cos_half_angles(&self) -> [f32; 2] {
        let outer = self.phi.max(self.theta).clamp(0.0, std::f32::consts::PI);
        let inner = self.theta.min(outer).max(0.0);
        [(0.5 * inner).cos(), (0.5 * outer).cos()]
    }

    /// Direct3D9 spot-cone factor evaluated at `world_pos` (`theta` inner, `phi` outer, `falloff`).
    pub fn evaluate_spot_factor(&self, world_pos: [f32; 3]) -> f32 {
        if !self.is_spot() {
            return 1.0;
        }
        let to_pt = [
            world_pos[0] - self.position[0],
            world_pos[1] - self.position[1],
            world_pos[2] - self.position[2],
        ];
        let dist_sq = to_pt[0] * to_pt[0] + to_pt[1] * to_pt[1] + to_pt[2] * to_pt[2];
        if dist_sq <= 1e-12 {
            return 1.0;
        }
        let inv = 1.0 / dist_sq.sqrt();
        let dir = self.normalized_direction();
        let cos_alpha =
            (to_pt[0] * dir[0] + to_pt[1] * dir[1] + to_pt[2] * dir[2]) * inv;
        let [cos_inner, cos_outer] = self.spot_cos_half_angles();
        if cos_alpha <= cos_outer {
            0.0
        } else if cos_alpha >= cos_inner || (cos_inner - cos_outer).abs() <= 1e-6 {
            1.0
        } else {
            let t = ((cos_alpha - cos_outer) / (cos_inner - cos_outer)).clamp(0.0, 1.0);
            t.powf(self.falloff.max(0.0))
        }
    }

    /// Evaluate animated diffuse RGBA by modulating `self.diffuse` with an optional `LightAnimItem` (`lanims.xr`).
    pub fn evaluate_diffuse_with_anim(
        &self,
        anim: Option<&LightAnimItem>,
        t_sec: f32,
        smooth: bool,
    ) -> [f32; 4] {
        match anim {
            Some(item) => {
                let c = item.sample_rgba_f32(t_sec, smooth);
                [
                    self.diffuse[0] * c[0],
                    self.diffuse[1] * c[1],
                    self.diffuse[2] * c[2],
                    self.diffuse[3] * c[3],
                ]
            }
            None => self.diffuse,
        }
    }

    /// Combined scalar attenuation × spot factor × animated diffuse RGBA at `world_pos`.
    pub fn evaluate_contribution_at(
        &self,
        world_pos: [f32; 3],
        anim: Option<&LightAnimItem>,
        t_sec: f32,
        smooth: bool,
    ) -> [f32; 4] {
        let att = if self.is_directional() {
            1.0
        } else {
            let dx = world_pos[0] - self.position[0];
            let dy = world_pos[1] - self.position[1];
            let dz = world_pos[2] - self.position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            self.evaluate_attenuation(dist) * self.evaluate_spot_factor(world_pos)
        };
        if att <= 0.0 {
            return [0.0; 4];
        }
        let col = self.evaluate_diffuse_with_anim(anim, t_sec, smooth);
        [col[0] * att, col[1] * att, col[2] * att, col[3]]
    }
}

/// Compiler `R_Light` record inside `$level$\build.lights` (`104` bytes, `R_light.h:7-30`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildLightRecord {
    pub type_id: u16,
    pub gi_level: u16,
    pub diffuse: [f32; 3],
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub range: f32,
    pub range_sq: f32,
    pub falloff: f32,
    pub attenuation: [f32; 3],
    pub energy: f32,
    pub tri: [[f32; 3]; 3],
}

impl BuildLightRecord {
    /// Convert this static/hemi compiler light into a runtime [`LevelLight`] (`Light_DB.cpp:128-149`).
    pub fn to_level_light(&self) -> LevelLight {
        let light_type = match self.type_id {
            build_light_type::DIRECT => d3d_light_type::DIRECTIONAL,
            _ => d3d_light_type::POINT,
        };
        LevelLight {
            controller_id: u32::MAX,
            light_type,
            diffuse: [self.diffuse[0], self.diffuse[1], self.diffuse[2], 1.0],
            specular: [0.0, 0.0, 0.0, 1.0],
            ambient: [0.0, 0.0, 0.0, 1.0],
            position: self.position,
            direction: self.direction,
            range: self.range,
            falloff: self.falloff,
            attenuation: self.attenuation,
            theta: 0.0,
            phi: 0.0,
        }
    }
}

/// Parsed `$level$\build.lights` direct & hemispheric light tables (`CLight_DB::LoadHemi`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuildLights {
    pub direct: Vec<BuildLightRecord>,
    pub hemi: Vec<BuildLightRecord>,
}

impl BuildLights {
    fn parse_records(bytes: &[u8]) -> Result<Vec<BuildLightRecord>, FormatError> {
        if !bytes.len().is_multiple_of(BUILD_LIGHT_RECORD_SIZE) {
            return Err(FormatError::InvalidValue(
                "build.lights chunk size must be a multiple of 104 bytes",
            ));
        }
        let mut out = Vec::with_capacity(bytes.len() / BUILD_LIGHT_RECORD_SIZE);
        for rec in bytes.as_chunks::<BUILD_LIGHT_RECORD_SIZE>().0 {
            let mut c = ByteCursor::new(rec);
            out.push(BuildLightRecord {
                type_id: c.read_u16()?,
                gi_level: c.read_u16()?,
                diffuse: c.read_f32_3()?,
                position: c.read_f32_3()?,
                direction: c.read_f32_3()?,
                range: c.read_f32()?,
                range_sq: c.read_f32()?,
                falloff: c.read_f32()?,
                attenuation: c.read_f32_3()?,
                energy: c.read_f32()?,
                tri: [c.read_f32_3()?, c.read_f32_3()?, c.read_f32_3()?],
            });
        }
        Ok(out)
    }

    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let direct = match reader.open_chunk(0)? {
            Some(bytes) => Self::parse_records(&bytes)?,
            None => Vec::new(),
        };
        let hemi = match reader.open_chunk(1)? {
            Some(bytes) => Self::parse_records(&bytes)?,
            None => Vec::new(),
        };
        Ok(Self { direct, hemi })
    }
}

/// Parsed `[torch_definition]` LTX lighting parameters (`CTorch::LoadLightParams`, `Torch.cpp:202-262`).
#[derive(Debug, Clone, PartialEq)]
pub struct TorchLightConfig {
    pub section: String,
    pub light_type: u32,
    pub omni_type: u32,
    pub color: [f32; 4],
    pub range: f32,
    pub omni_color: [f32; 4],
    pub omni_range: f32,
    pub spot_angle_deg: f32,
    pub spot_texture: Option<String>,
    pub color_animator: String,
    pub guide_bone: String,
    pub glow_texture: String,
    pub glow_radius: f32,
    pub volumetric: bool,
    pub volumetric_quality: f32,
    pub volumetric_intensity: f32,
    pub volumetric_distance: f32,
}

impl Default for TorchLightConfig {
    fn default() -> Self {
        Self {
            section: "torch_definition".to_string(),
            light_type: d3d_light_type::SPOT,
            omni_type: d3d_light_type::POINT,
            color: [0.6, 0.55, 0.55, 0.8],
            range: 25.0,
            omni_color: [0.25, 0.25, 0.25, 0.3],
            omni_range: 1.5,
            spot_angle_deg: 60.0,
            spot_texture: None,
            color_animator: String::new(),
            guide_bone: "lights_bone".to_string(),
            glow_texture: String::new(),
            glow_radius: 0.3,
            volumetric: false,
            volumetric_quality: 1.0,
            volumetric_intensity: 1.0,
            volumetric_distance: 1.0,
        }
    }
}

fn parse_ltx_fcolor(raw: Option<&str>, default: [f32; 4]) -> [f32; 4] {
    let Some(raw) = raw else {
        return default;
    };
    let mut out = default;
    for (idx, tok) in raw.split(',').take(4).enumerate() {
        if let Ok(val) = tok.trim().parse::<f32>() {
            out[idx] = val;
        }
    }
    out
}

impl TorchLightConfig {
    pub fn from_ltx(
        doc: &OwnedLtxDocument,
        section: &str,
        prefer_r2: bool,
    ) -> Result<Self, FormatError> {
        if !doc.sections.contains_key(section) {
            return Err(FormatError::InvalidValue("missing torch definition LTX section"));
        }
        let color_key = if prefer_r2 && doc.read_str(section, "color_r2").is_some() {
            "color_r2"
        } else {
            "color"
        };
        let range_key = if prefer_r2 && doc.read_str(section, "range_r2").is_some() {
            "range_r2"
        } else {
            "range"
        };
        let omni_color_key = if prefer_r2 && doc.read_str(section, "omni_color_r2").is_some() {
            "omni_color_r2"
        } else {
            "omni_color"
        };
        let omni_range_key = if prefer_r2 && doc.read_str(section, "omni_range_r2").is_some() {
            "omni_range_r2"
        } else {
            "omni_range"
        };
        let def = Self::default();
        Ok(Self {
            section: section.to_string(),
            light_type: doc.read_i32(section, "type", def.light_type as i32).max(0) as u32,
            omni_type: doc
                .read_i32(section, "omni_type", def.omni_type as i32)
                .max(0) as u32,
            color: parse_ltx_fcolor(doc.read_str(section, color_key), def.color),
            range: doc.read_f32(section, range_key, def.range),
            omni_color: parse_ltx_fcolor(doc.read_str(section, omni_color_key), def.omni_color),
            omni_range: doc.read_f32(section, omni_range_key, def.omni_range),
            spot_angle_deg: doc.read_f32(section, "spot_angle", def.spot_angle_deg),
            spot_texture: doc
                .read_str(section, "spot_texture")
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
            color_animator: doc
                .read_str(section, "color_animator")
                .unwrap_or("")
                .trim()
                .to_string(),
            guide_bone: doc
                .read_str(section, "guide_bone")
                .unwrap_or(&def.guide_bone)
                .trim()
                .to_string(),
            glow_texture: doc
                .read_str(section, "glow_texture")
                .unwrap_or("")
                .trim()
                .to_string(),
            glow_radius: doc.read_f32(section, "glow_radius", def.glow_radius),
            volumetric: doc.read_bool(section, "volumetric", false),
            volumetric_quality: doc.read_f32(section, "volumetric_quality", 1.0),
            volumetric_intensity: doc.read_f32(section, "volumetric_intensity", 1.0),
            volumetric_distance: doc.read_f32(section, "volumetric_distance", 1.0),
        })
    }

    /// Build `(primary_spot_or_point, secondary_omni)` [`LevelLight`] instances at `position` & `direction`.
    pub fn to_level_light_pair(
        &self,
        position: [f32; 3],
        direction: [f32; 3],
    ) -> (LevelLight, LevelLight) {
        let phi = self.spot_angle_deg.to_radians();
        let primary = LevelLight {
            controller_id: u32::MAX,
            light_type: self.light_type,
            diffuse: self.color,
            specular: [0.0; 4],
            ambient: [0.0; 4],
            position,
            direction,
            range: self.range,
            falloff: 1.0,
            attenuation: [0.0, 0.0, 0.0],
            theta: phi * 0.5,
            phi,
        };
        let omni = LevelLight {
            controller_id: u32::MAX,
            light_type: self.omni_type,
            diffuse: self.omni_color,
            specular: [0.0; 4],
            ambient: [0.0; 4],
            position,
            direction,
            range: self.omni_range,
            falloff: 1.0,
            attenuation: [0.0, 0.0, 0.0],
            theta: 0.0,
            phi: 0.0,
        };
        (primary, omni)
    }
}

/// Resolved drawable mesh extracted from a level visual + `LevelGeometry`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedLevelMesh {
    pub shader: Option<LevelShaderRef>,
    pub vertices: Vec<StaticVertex>,
    pub indices: Vec<u16>,
}

impl ResolvedLevelMesh {
    /// Extract compact XYZ vertex positions suitable for Z-prepass or directional shadow depth passes.
    pub fn to_position_only(&self) -> Vec<[f32; 3]> {
        self.vertices.iter().map(|v| v.position).collect()
    }

    /// Sub-slice this mesh using an `FSlideWindow` progressive LOD record.
    pub fn slice_window(&self, window: SlideWindow) -> Result<Self, FormatError> {
        let start = window.offset as usize;
        let len = (window.num_tris as usize) * 3;
        let end = start
            .checked_add(len)
            .ok_or(FormatError::InvalidValue("SWI window index overflow"))?;
        let indices = self
            .indices
            .get(start..end)
            .ok_or(FormatError::InvalidValue("SWI window index slice out of bounds"))?
            .to_vec();
        let max_ref = indices.iter().copied().max().map(usize::from).unwrap_or(0);
        let vert_count = (window.num_verts as usize).min(self.vertices.len());
        let vertices = if vert_count > 0 && max_ref < vert_count {
            self.vertices[..vert_count].to_vec()
        } else {
            self.vertices.clone()
        };
        Ok(Self {
            shader: self.shader.clone(),
            vertices,
            indices,
        })
    }
}

/// Parsed `level` scene description (`hdrLEVEL`, shaders, visuals, portals, sectors, lights, glows).
#[derive(Debug, Clone, PartialEq)]
pub struct LevelData {
    pub xrlc_version: u16,
    pub xrlc_quality: u16,
    pub shaders: Vec<LevelShaderRef>,
    pub visuals: Vec<OgfModel>,
    pub portals: Vec<LevelPortal>,
    pub sectors: Vec<LevelSector>,
    pub lights: Vec<LevelLight>,
    pub glows: Vec<LevelGlow>,
    pub glows_raw_len: usize,
}

impl LevelData {
    /// Parse a binary `level` scene file (`CRender::level_Load`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        use level_chunks::*;
        let reader = ChunkReader::new(data);

        let (xrlc_version, xrlc_quality) = match reader.open_chunk(FSL_HEADER)? {
            Some(bytes) => {
                let mut hc = ByteCursor::new(&bytes);
                (hc.read_u16()?, hc.read_u16()?)
            }
            None => (14, 0),
        };

        let shaders = match reader.open_chunk(FSL_SHADERS)? {
            Some(bytes) => {
                let mut sc = ByteCursor::new(&bytes);
                let count = sc.read_u32()? as usize;
                let mut out = Vec::with_capacity(count);
                for _ in 0..count {
                    out.push(LevelShaderRef::parse(sc.read_zstring()?));
                }
                out
            }
            None => Vec::new(),
        };

        let mut visuals = Vec::new();
        if let Some(vis_bytes) = reader.open_chunk(FSL_VISUALS)? {
            let vreader = ChunkReader::new(&vis_bytes);
            for entry in vreader.iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                visuals.push(OgfModel::parse(&payload)?);
            }
        }

        let portals = match reader.open_chunk(FSL_PORTALS)? {
            Some(bytes) => {
                if bytes.len() % 80 != 0 {
                    return Err(FormatError::InvalidValue(
                        "fsL_PORTALS size must be a multiple of 80 bytes",
                    ));
                }
                let mut out = Vec::with_capacity(bytes.len() / 80);
                for rec in bytes.as_chunks::<80>().0 {
                    let mut pc = ByteCursor::new(rec);
                    let sector_front = pc.read_u16()?;
                    let sector_back = pc.read_u16()?;
                    let mut verts = [[0.0f32; 3]; 6];
                    for v in &mut verts {
                        *v = pc.read_f32_3()?;
                    }
                    let count = pc.read_u32()? as usize;
                    if count > 6 {
                        return Err(FormatError::InvalidValue(
                            "level portal vertex count exceeds 6",
                        ));
                    }
                    out.push(LevelPortal {
                        sector_front,
                        sector_back,
                        vertices: verts[..count].to_vec(),
                    });
                }
                out
            }
            None => Vec::new(),
        };

        let mut sectors = Vec::new();
        if let Some(sec_bytes) = reader.open_chunk(FSL_SECTORS)? {
            let sreader = ChunkReader::new(&sec_bytes);
            for entry in sreader.iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                let sr = ChunkReader::new(&payload);

                let portals = match sr.open_chunk(sector_chunks::FSP_PORTALS)? {
                    Some(pb) => {
                        if pb.len() % 2 != 0 {
                            return Err(FormatError::InvalidValue("fsP_Portals size must be even"));
                        }
                        pb.as_chunks::<2>()
                            .0
                            .iter()
                            .map(|pair| u16::from_le_bytes(*pair))
                            .collect()
                    }
                    None => Vec::new(),
                };

                let root_visual_id = match sr.open_chunk(sector_chunks::FSP_ROOT)? {
                    Some(rb) => {
                        let mut rc = ByteCursor::new(&rb);
                        Some(rc.read_u32()?)
                    }
                    None => None,
                };

                sectors.push(LevelSector {
                    id: entry.id,
                    portals,
                    root_visual_id,
                });
            }
        }

        let lights = match reader.open_chunk(FSL_LIGHT_DYNAMIC)? {
            Some(bytes) => {
                if bytes.len() % 108 != 0 {
                    return Err(FormatError::InvalidValue(
                        "fsL_LIGHT_DYNAMIC size must be a multiple of 108 bytes",
                    ));
                }
                let mut out = Vec::with_capacity(bytes.len() / 108);
                for rec in bytes.as_chunks::<108>().0 {
                    let mut lc = ByteCursor::new(rec);
                    let controller_id = lc.read_u32()?;
                    let light_type = lc.read_u32()?;
                    let diffuse = lc.read_f32_4()?;
                    let specular = lc.read_f32_4()?;
                    let ambient = lc.read_f32_4()?;
                    let position = lc.read_f32_3()?;
                    let direction = lc.read_f32_3()?;
                    let range = lc.read_f32()?;
                    let falloff = lc.read_f32()?;
                    let attenuation = [lc.read_f32()?, lc.read_f32()?, lc.read_f32()?];
                    let theta = lc.read_f32()?;
                    let phi = lc.read_f32()?;
                    out.push(LevelLight {
                        controller_id,
                        light_type,
                        diffuse,
                        specular,
                        ambient,
                        position,
                        direction,
                        range,
                        falloff,
                        attenuation,
                        theta,
                        phi,
                    });
                }
                out
            }
            None => Vec::new(),
        };

        let (glows, glows_raw_len) = match reader.open_chunk(FSL_GLOWS)? {
            Some(bytes) => (LevelGlow::parse_list(&bytes)?, bytes.len()),
            None => (Vec::new(), 0),
        };

        Ok(Self {
            xrlc_version,
            xrlc_quality,
            shaders,
            visuals,
            portals,
            sectors,
            lights,
            glows,
            glows_raw_len,
        })
    }

    #[inline]
    pub fn visual_shader(&self, visual: &OgfModel) -> Option<&LevelShaderRef> {
        self.shaders.get(visual.header.shader_id as usize)
    }

    #[inline]
    pub fn glow_shader(&self, glow: &LevelGlow) -> Option<&LevelShaderRef> {
        self.shaders.get(glow.shader_id as usize)
    }

    #[inline]
    pub fn sector_by_id(&self, sector_id: u32) -> Option<&LevelSector> {
        self.sectors.iter().find(|s| s.id == sector_id)
    }

    /// Collect indices into `self.portals` referenced by `sector_id` (plus any portal connecting it).
    pub fn portals_for_sector(&self, sector_id: u16) -> Vec<usize> {
        if let Some(sec) = self.sector_by_id(u32::from(sector_id)) {
            return sec
                .portals
                .iter()
                .map(|&p| p as usize)
                .filter(|&idx| idx < self.portals.len())
                .collect();
        }
        self.portals
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                (p.sector_front == sector_id || p.sector_back == sector_id).then_some(idx)
            })
            .collect()
    }

    /// Collect all sector IDs reachable from `start_sector` across portals whose bounding spheres
    /// intersect `(sphere_center, sphere_radius)` — useful for light/geometry portal culling.
    pub fn collect_reachable_sectors_within_sphere(
        &self,
        start_sector: u16,
        sphere_center: [f32; 3],
        sphere_radius: f32,
    ) -> Vec<u16> {
        let r_lim = sphere_radius.max(0.0);
        let mut visited_sectors = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        let mut seen = std::collections::HashSet::new();
        seen.insert(start_sector);
        queue.push_back(start_sector);

        while let Some(sec_id) = queue.pop_front() {
            visited_sectors.push(sec_id);
            for portal_idx in self.portals_for_sector(sec_id) {
                let portal = &self.portals[portal_idx];
                let Some(next) = portal.opposite_sector(sec_id) else {
                    continue;
                };
                if seen.contains(&next) {
                    continue;
                }
                if let Some((pc, pr)) = portal.bounding_sphere() {
                    let dx = pc[0] - sphere_center[0];
                    let dy = pc[1] - sphere_center[1];
                    let dz = pc[2] - sphere_center[2];
                    let max_d = pr + r_lim;
                    if dx * dx + dy * dy + dz * dz > max_d * max_d {
                        continue;
                    }
                }
                seen.insert(next);
                queue.push_back(next);
            }
        }
        visited_sectors
    }

    /// Resolve vertices & indices for any level visual (inline OGF, `OGF_GCONTAINER`,
    /// or `OGF_VCONTAINER`/`OGF_ICONTAINER` reference).
    pub fn resolve_visual_mesh(
        &self,
        visual_index: usize,
        geom: &LevelGeometry,
    ) -> Result<ResolvedLevelMesh, FormatError> {
        let visual = self
            .visuals
            .get(visual_index)
            .ok_or(FormatError::InvalidValue("level visual index out of range"))?;
        let shader = self.visual_shader(visual).cloned();

        if let Some(gc) = visual.gcontainer {
            let vertices = geom.slice_vertices(gc.vb_id, gc.vb_base, gc.vb_count)?;
            let indices = geom.slice_indices(gc.ib_id, gc.ib_base, gc.ib_count)?.to_vec();
            return Ok(ResolvedLevelMesh {
                shader,
                vertices,
                indices,
            });
        }

        if let Some(vc) = visual.vcontainer {
            let vertices = geom.slice_vertices(vc.id, vc.base, vc.count)?;
            let indices = if let Some(ic) = visual.icontainer {
                geom.slice_indices(ic.id, ic.base, ic.count)?.to_vec()
            } else {
                visual.indices.clone()
            };
            return Ok(ResolvedLevelMesh {
                shader,
                vertices,
                indices,
            });
        }

        if let Some(ref inline_verts) = visual.vertices {
            return Ok(ResolvedLevelMesh {
                shader,
                vertices: inline_verts.to_static_attributes(),
                indices: visual.indices.clone(),
            });
        }

        Err(FormatError::InvalidValue(
            "level visual contains neither inline vertices nor OGF container references",
        ))
    }

    /// Resolve `OGF_FASTPATH` shadow/Z-prepass geometry from `level.geomx` if present on the visual.
    pub fn resolve_fastpath_mesh(
        &self,
        visual_index: usize,
        geomx: &LevelGeometry,
    ) -> Result<Option<ResolvedLevelMesh>, FormatError> {
        let visual = self
            .visuals
            .get(visual_index)
            .ok_or(FormatError::InvalidValue("level visual index out of range"))?;
        let Some(gc) = visual.fastpath_gcontainer else {
            return Ok(None);
        };
        let shader = self.visual_shader(visual).cloned();
        let vertices = geomx.slice_vertices(gc.vb_id, gc.vb_base, gc.vb_count)?;
        let indices = geomx.slice_indices(gc.ib_id, gc.ib_base, gc.ib_count)?.to_vec();
        Ok(Some(ResolvedLevelMesh {
            shader,
            vertices,
            indices,
        }))
    }

    /// Resolve vertices & indices for a visual at a given progressive `lod` (`0.0..=1.0`),
    /// honoring inline `OGF_SWIDATA` or `OGF_SWICONTAINER` (`level.geom` `FSL_SWIS`).
    pub fn resolve_visual_mesh_lod(
        &self,
        visual_index: usize,
        geom: &LevelGeometry,
        lod: f32,
    ) -> Result<ResolvedLevelMesh, FormatError> {
        let visual = self
            .visuals
            .get(visual_index)
            .ok_or(FormatError::InvalidValue("level visual index out of range"))?;
        let win = visual
            .resolve_swi(&geom.swis, false)
            .and_then(|s| s.select_window(lod));
        let Some(sw) = win else {
            return self.resolve_visual_mesh(visual_index, geom);
        };

        let shader = self.visual_shader(visual).cloned();
        if let Some(gc) = visual.gcontainer {
            let ib_start = gc
                .ib_base
                .checked_add(sw.offset)
                .ok_or(FormatError::InvalidValue("SWI ib_base overflow"))?;
            let indices = geom
                .slice_indices(gc.ib_id, ib_start, (sw.num_tris as u32) * 3)?
                .to_vec();
            let all_verts = geom.slice_vertices(gc.vb_id, gc.vb_base, gc.vb_count)?;
            let max_ref = indices.iter().copied().max().map(usize::from).unwrap_or(0);
            let vertices = if sw.num_verts > 0 && max_ref < (sw.num_verts as usize) && (sw.num_verts as usize) <= all_verts.len() {
                all_verts[..sw.num_verts as usize].to_vec()
            } else {
                all_verts
            };
            return Ok(ResolvedLevelMesh {
                shader,
                vertices,
                indices,
            });
        }

        if let Some(vc) = visual.vcontainer {
            let all_verts = geom.slice_vertices(vc.id, vc.base, vc.count)?;
            let indices = if let Some(ic) = visual.icontainer {
                let ib_start = ic
                    .base
                    .checked_add(sw.offset)
                    .ok_or(FormatError::InvalidValue("SWI icontainer base overflow"))?;
                geom.slice_indices(ic.id, ib_start, (sw.num_tris as u32) * 3)?
                    .to_vec()
            } else {
                visual.progressive_indices(lod, &geom.swis).to_vec()
            };
            return Ok(ResolvedLevelMesh {
                shader,
                vertices: all_verts,
                indices,
            });
        }

        self.resolve_visual_mesh(visual_index, geom)?.slice_window(sw)
    }

    /// Resolve `OGF_FASTPATH` Z-prepass / directional shadow mesh at progressive `lod` (`0.0..=1.0`),
    /// matching `FProgressive::Render` (`PHASE_SMAP` using `fastpath_swi`).
    pub fn resolve_fastpath_mesh_lod(
        &self,
        visual_index: usize,
        geomx: &LevelGeometry,
        lod: f32,
    ) -> Result<Option<ResolvedLevelMesh>, FormatError> {
        let visual = self
            .visuals
            .get(visual_index)
            .ok_or(FormatError::InvalidValue("level visual index out of range"))?;
        let Some(gc) = visual.fastpath_gcontainer else {
            return Ok(None);
        };
        let win = visual
            .resolve_swi(&geomx.swis, true)
            .and_then(|s| s.select_window(lod));
        let Some(sw) = win else {
            return self.resolve_fastpath_mesh(visual_index, geomx);
        };

        let shader = self.visual_shader(visual).cloned();
        let ib_start = gc
            .ib_base
            .checked_add(sw.offset)
            .ok_or(FormatError::InvalidValue("fastpath SWI ib_base overflow"))?;
        let indices = geomx
            .slice_indices(gc.ib_id, ib_start, (sw.num_tris as u32) * 3)?
            .to_vec();
        let all_verts = geomx.slice_vertices(gc.vb_id, gc.vb_base, gc.vb_count)?;
        let max_ref = indices.iter().copied().max().map(usize::from).unwrap_or(0);
        let vertices = if sw.num_verts > 0 && max_ref < (sw.num_verts as usize) && (sw.num_verts as usize) <= all_verts.len() {
            all_verts[..sw.num_verts as usize].to_vec()
        } else {
            all_verts
        };
        Ok(Some(ResolvedLevelMesh {
            shader,
            vertices,
            indices,
        }))
    }

    /// Collect all leaf visual indices reachable from `root_visual_index` (e.g. a sector root)
    /// across `OGF_CHILDREN_L` hierarchy links, guarding against cycles.
    pub fn collect_hierarchy_leaves(&self, root_visual_index: usize) -> Vec<usize> {
        if root_visual_index >= self.visuals.len() {
            return Vec::new();
        }
        let mut visited = vec![false; self.visuals.len()];
        let mut leaves = Vec::new();
        let mut stack = vec![root_visual_index];

        while let Some(idx) = stack.pop() {
            if idx >= self.visuals.len() || visited[idx] {
                continue;
            }
            visited[idx] = true;
            let vis = &self.visuals[idx];
            if vis.child_links.is_empty() {
                leaves.push(idx);
            } else {
                for &child_id in vis.child_links.iter().rev() {
                    stack.push(child_id as usize);
                }
            }
        }
        leaves
    }
}

fn decode_vec3_attr(decl_type: u8, attr: &[u8]) -> Result<[f32; 3], FormatError> {
    use d3d_decl_type::*;
    let mut c = ByteCursor::new(attr);
    match decl_type {
        FLOAT3 | FLOAT4 => c.read_f32_3(),
        FLOAT16_4 => Ok([
            f16_to_f32(c.read_u16()?),
            f16_to_f32(c.read_u16()?),
            f16_to_f32(c.read_u16()?),
        ]),
        SHORT4 => Ok([
            f32::from(c.read_i16()?),
            f32::from(c.read_i16()?),
            f32::from(c.read_i16()?),
        ]),
        _ => Err(FormatError::InvalidValue("unsupported POSITION D3DDECLTYPE")),
    }
}

fn decode_normal_attr(decl_type: u8, attr: &[u8]) -> Result<[f32; 3], FormatError> {
    use d3d_decl_type::*;
    let mut c = ByteCursor::new(attr);
    match decl_type {
        FLOAT3 | FLOAT4 => c.read_f32_3(),
        // D3DCOLOR memory layout on little-endian is [B, G, R, A] -> (.zyx - 0.5) * 2.0
        D3DCOLOR => {
            let bz = f32::from(attr[0]) / 255.0 * 2.0 - 1.0;
            let gy = f32::from(attr[1]) / 255.0 * 2.0 - 1.0;
            let rx = f32::from(attr[2]) / 255.0 * 2.0 - 1.0;
            Ok([rx, gy, bz])
        }
        UBYTE4N => Ok([
            f32::from(attr[0]) / 255.0 * 2.0 - 1.0,
            f32::from(attr[1]) / 255.0 * 2.0 - 1.0,
            f32::from(attr[2]) / 255.0 * 2.0 - 1.0,
        ]),
        SHORT4N => Ok([
            f32::from(c.read_i16()?) / 32767.0,
            f32::from(c.read_i16()?) / 32767.0,
            f32::from(c.read_i16()?) / 32767.0,
        ]),
        _ => Ok([0.0, 1.0, 0.0]),
    }
}

fn decode_uv_attr(decl_type: u8, attr: &[u8], short_scale: f32) -> Result<[f32; 2], FormatError> {
    use d3d_decl_type::*;
    let mut c = ByteCursor::new(attr);
    match decl_type {
        FLOAT2 | FLOAT3 | FLOAT4 => c.read_f32_2(),
        FLOAT16_2 | FLOAT16_4 => Ok([f16_to_f32(c.read_u16()?), f16_to_f32(c.read_u16()?)]),
        SHORT2 | SHORT4 => Ok([
            f32::from(c.read_i16()?) * short_scale,
            f32::from(c.read_i16()?) * short_scale,
        ]),
        SHORT2N | SHORT4N => Ok([
            f32::from(c.read_i16()?) / 32767.0,
            f32::from(c.read_i16()?) / 32767.0,
        ]),
        USHORT2N | USHORT4N => Ok([
            f32::from(c.read_u16()?) / 65535.0,
            f32::from(c.read_u16()?) / 65535.0,
        ]),
        _ => Ok([0.0, 0.0]),
    }
}

/// IEEE-754 half-precision float (`binary16`) to `f32`.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mant = u32::from(bits & 0x03FF);

    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };

    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use crate::ogf::ogf_chunks::*;
    use crate::ogf::OgfVisualType;

    fn write_decl_elem(out: &mut Vec<u8>, elem: VertexElement9) {
        out.extend_from_slice(&elem.stream.to_le_bytes());
        out.extend_from_slice(&elem.offset.to_le_bytes());
        out.push(elem.decl_type);
        out.push(elem.method);
        out.push(elem.usage);
        out.push(elem.usage_index);
    }

    #[test]
    fn test_level_and_geom_parsing_with_d3d9_declarator_and_gcontainer() {
        use d3d_decl_type::*;
        use d3d_decl_usage::*;
        use level_chunks::*;

        // 1. Build level.geom with packed X-Ray D3D9 vertex format (FLOAT3 pos + D3DCOLOR normal + SHORT2 uv = 20B)
        let mut vb_stream = Vec::new();
        vb_stream.extend_from_slice(&1u32.to_le_bytes()); // 1 VB
        write_decl_elem(
            &mut vb_stream,
            VertexElement9 {
                stream: 0,
                offset: 0,
                decl_type: FLOAT3,
                method: 0,
                usage: POSITION,
                usage_index: 0,
            },
        );
        write_decl_elem(
            &mut vb_stream,
            VertexElement9 {
                stream: 0,
                offset: 12,
                decl_type: D3DCOLOR,
                method: 0,
                usage: NORMAL,
                usage_index: 0,
            },
        );
        write_decl_elem(
            &mut vb_stream,
            VertexElement9 {
                stream: 0,
                offset: 16,
                decl_type: SHORT2,
                method: 0,
                usage: TEXCOORD,
                usage_index: 0,
            },
        );
        write_decl_elem(&mut vb_stream, VertexElement9::END);

        vb_stream.extend_from_slice(&4u32.to_le_bytes()); // 4 vertices (visual uses slice [1..4])
        for (idx, p) in [
            [99.0f32, 99.0, 99.0], // padding at base 0
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
        .iter()
        .enumerate()
        {
            for c in p {
                vb_stream.extend_from_slice(&c.to_le_bytes());
            }
            // D3DCOLOR normal pointing +Y => [B=128, G=255, R=128, A=255]
            vb_stream.extend_from_slice(&[128, 255, 128, 255]);
            // SHORT2 UV => idx * 512 => idx * 0.5
            vb_stream.extend_from_slice(&((idx as i16) * 512).to_le_bytes());
            vb_stream.extend_from_slice(&0i16.to_le_bytes());
        }

        let mut ib_stream = Vec::new();
        ib_stream.extend_from_slice(&1u32.to_le_bytes()); // 1 IB
        ib_stream.extend_from_slice(&5u32.to_le_bytes()); // 5 indices; visual slices [2..5]
        for i in [0u16, 0, 0, 1, 2] {
            ib_stream.extend_from_slice(&i.to_le_bytes());
        }

        let mut geom_w = ChunkWriter::new();
        geom_w.write_chunk(FSL_VB, &vb_stream);
        geom_w.write_chunk(FSL_IB, &ib_stream);
        let geom = LevelGeometry::parse(&geom_w.into_bytes()).unwrap();
        assert_eq!(geom.vertex_buffers[0].declarator.stride, 20);

        // 2. Build level file chunks: header, shaders, visuals, portals, sectors, lights
        let mut hdr = Vec::new();
        hdr.extend_from_slice(&14u16.to_le_bytes()); // XRCL_PRODUCTION_VERSION
        hdr.extend_from_slice(&2u16.to_le_bytes()); // ebqCustom

        let mut shd = Vec::new();
        shd.extend_from_slice(&2u32.to_le_bytes());
        shd.extend_from_slice(b"\0"); // shader 0 = reserved empty
        shd.extend_from_slice(b"levels\\crete_wall/crete\\crete_wall_01,lmap#1\0");

        let mut vis0_w = ChunkWriter::new();
        let mut vhdr = vec![4u8, OgfVisualType::Normal as u8];
        vhdr.extend_from_slice(&1u16.to_le_bytes()); // shader_id = 1
        vhdr.extend_from_slice(&[0u8; 40]);
        vis0_w.write_chunk(OGF_HEADER, &vhdr);
        let mut gcont = Vec::new();
        for val in [0u32, 1, 3, 0, 2, 3] {
            gcont.extend_from_slice(&val.to_le_bytes());
        }
        vis0_w.write_chunk(OGF_GCONTAINER, &gcont);
        let mut fast_inner = ChunkWriter::new();
        fast_inner.write_chunk(OGF_GCONTAINER, &gcont);
        vis0_w.write_chunk(OGF_FASTPATH, &fast_inner.into_bytes());

        let mut vis1_w = ChunkWriter::new();
        vis1_w.write_chunk(OGF_HEADER, &vhdr);
        let mut vcont = Vec::new();
        for val in [0u32, 1, 3] {
            vcont.extend_from_slice(&val.to_le_bytes());
        }
        let mut icont = Vec::new();
        for val in [0u32, 2, 3] {
            icont.extend_from_slice(&val.to_le_bytes());
        }
        vis1_w.write_chunk(OGF_VCONTAINER, &vcont);
        vis1_w.write_chunk(OGF_ICONTAINER, &icont);

        let mut vis_wrap = ChunkWriter::new();
        vis_wrap.write_chunk(0, &vis0_w.into_bytes());
        vis_wrap.write_chunk(1, &vis1_w.into_bytes());

        // Portal 0: sectors (0, 1), 4 quad corners + 2 unused slots, count = 4
        let mut port_raw = Vec::new();
        port_raw.extend_from_slice(&0u16.to_le_bytes());
        port_raw.extend_from_slice(&1u16.to_le_bytes());
        let pverts = [
            [0.0f32, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
        ];
        for pv in pverts {
            for f in pv {
                port_raw.extend_from_slice(&f.to_le_bytes());
            }
        }
        port_raw.extend_from_slice(&4u32.to_le_bytes());

        // Sector 0 referencing portal 0 & root visual 0
        let mut sec0 = ChunkWriter::new();
        sec0.write_chunk(sector_chunks::FSP_PORTALS, &0u16.to_le_bytes());
        sec0.write_chunk(sector_chunks::FSP_ROOT, &0u32.to_le_bytes());
        let mut secs_wrap = ChunkWriter::new();
        secs_wrap.write_chunk(0, &sec0.into_bytes());

        // Dynamic/static light (108 bytes)
        let mut light_raw = Vec::new();
        light_raw.extend_from_slice(&1u32.to_le_bytes()); // controller_id
        light_raw.extend_from_slice(&3u32.to_le_bytes()); // D3DLIGHT_DIRECTIONAL = 3
        for f in [
            1.0f32, 0.9, 0.8, 1.0, // diffuse
            0.0, 0.0, 0.0, 1.0, // specular
            0.2, 0.2, 0.2, 1.0, // ambient
            10.0, 20.0, 30.0, // pos
            0.0, -1.0, 0.0, // dir
            500.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, // range..phi
        ] {
            light_raw.extend_from_slice(&f.to_le_bytes());
        }

        let mut lvl_w = ChunkWriter::new();
        lvl_w.write_chunk(FSL_HEADER, &hdr);
        lvl_w.write_chunk(FSL_SHADERS, &shd);
        lvl_w.write_chunk(FSL_VISUALS, &vis_wrap.into_bytes());
        lvl_w.write_chunk(FSL_PORTALS, &port_raw);
        lvl_w.write_chunk(FSL_SECTORS, &secs_wrap.into_bytes());
        lvl_w.write_chunk(FSL_LIGHT_DYNAMIC, &light_raw);

        let level = LevelData::parse(&lvl_w.into_bytes()).unwrap();
        assert_eq!((level.xrlc_version, level.xrlc_quality), (14, 2));
        assert_eq!(level.shaders[0].raw, "");
        assert_eq!(level.shaders[1].shader, "levels\\crete_wall");
        assert_eq!(level.shaders[1].textures, vec!["crete\\crete_wall_01", "lmap#1"]);
        assert_eq!(level.portals[0].vertices.len(), 4);
        assert_eq!(level.sectors[0].portals, vec![0]);
        assert_eq!(level.sectors[0].root_visual_id, Some(0));
        assert_eq!(level.lights[0].position, [10.0, 20.0, 30.0]);

        let resolved = level.resolve_visual_mesh(0, &geom).unwrap();
        assert_eq!(resolved.vertices.len(), 3);
        assert_eq!(resolved.vertices[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(resolved.vertices[1].position, [2.0, 0.0, 0.0]);
        assert!((resolved.vertices[1].normal[1] - 1.0).abs() < 1e-4);
        assert!((resolved.vertices[1].uv0[0] - 1.0).abs() < 1e-5); // vb index 2 * 512 / 1024 = 1.0
        assert_eq!(resolved.indices, vec![0, 1, 2]);
        assert_eq!(resolved.shader.unwrap().shader, "levels\\crete_wall");

        let fastpath = level.resolve_fastpath_mesh(0, &geom).unwrap().unwrap();
        assert_eq!(fastpath.vertices.len(), 3);
        assert_eq!(fastpath.indices, vec![0, 1, 2]);

        let resolved_vc = level.resolve_visual_mesh(1, &geom).unwrap();
        assert_eq!(resolved_vc.vertices.len(), 3);
        assert_eq!(resolved_vc.indices, vec![0, 1, 2]);

        // Half-float decode smoke check: 0x3C00 = 1.0, 0xBC00 = -1.0, 0x3800 = 0.5
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        assert!((f16_to_f32(0xBC00) + 1.0).abs() < 1e-6);
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_level_swicontainer_fastpath_lod_and_hierarchy_traversal() {
        use d3d_decl_type::*;
        use d3d_decl_usage::*;
        use level_chunks::*;

        // Build geom with 1 VB (4 verts), 1 IB (9 indices = 2 progressive LOD windows: 2 tris at offset 0, 1 tri at offset 6),
        // and 1 shared SWI item in FSL_SWIS!
        let mut vb_stream = Vec::new();
        vb_stream.extend_from_slice(&1u32.to_le_bytes());
        write_decl_elem(
            &mut vb_stream,
            VertexElement9 {
                stream: 0,
                offset: 0,
                decl_type: FLOAT3,
                method: 0,
                usage: POSITION,
                usage_index: 0,
            },
        );
        write_decl_elem(&mut vb_stream, VertexElement9::END);
        vb_stream.extend_from_slice(&4u32.to_le_bytes());
        for p in [
            [0.0f32, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ] {
            for c in p {
                vb_stream.extend_from_slice(&c.to_le_bytes());
            }
        }

        let mut ib_stream = Vec::new();
        ib_stream.extend_from_slice(&1u32.to_le_bytes());
        ib_stream.extend_from_slice(&9u32.to_le_bytes());
        for i in [0u16, 1, 2, 0, 2, 3, 0, 1, 3] {
            ib_stream.extend_from_slice(&i.to_le_bytes());
        }

        let mut swis_stream = Vec::new();
        swis_stream.extend_from_slice(&1u32.to_le_bytes()); // 1 SWI item
        for r in [0u32; 4] {
            swis_stream.extend_from_slice(&r.to_le_bytes());
        }
        swis_stream.extend_from_slice(&2u32.to_le_bytes()); // 2 windows
        // Window 0 (high LOD): offset 0, 2 tris, 4 verts
        swis_stream.extend_from_slice(&0u32.to_le_bytes());
        swis_stream.extend_from_slice(&2u16.to_le_bytes());
        swis_stream.extend_from_slice(&4u16.to_le_bytes());
        // Window 1 (low LOD): offset 6, 1 tri, 3 verts
        swis_stream.extend_from_slice(&6u32.to_le_bytes());
        swis_stream.extend_from_slice(&1u16.to_le_bytes());
        swis_stream.extend_from_slice(&4u16.to_le_bytes());

        let mut geom_w = ChunkWriter::new();
        geom_w.write_chunk(FSL_VB, &vb_stream);
        geom_w.write_chunk(FSL_IB, &ib_stream);
        geom_w.write_chunk(FSL_SWIS, &swis_stream);
        let geom = LevelGeometry::parse(&geom_w.into_bytes()).unwrap();

        // Visual 0: root Hierarchy linking children [1, 2]
        let mut vis0 = ChunkWriter::new();
        let mut h0 = vec![4u8, OgfVisualType::Hierarchy as u8, 0, 0];
        h0.extend_from_slice(&[0u8; 40]);
        vis0.write_chunk(OGF_HEADER, &h0);
        let mut links = Vec::new();
        for v in [2u32, 1, 2] {
            links.extend_from_slice(&v.to_le_bytes());
        }
        vis0.write_chunk(OGF_CHILDREN_L, &links);

        // Visual 1: TreePm using OGF_SWICONTAINER(0) + OGF_FASTPATH with embedded OGF_SWIDATA
        let mut vis1 = ChunkWriter::new();
        let mut h1 = vec![4u8, OgfVisualType::TreePm as u8, 0, 0];
        h1.extend_from_slice(&[0u8; 40]);
        vis1.write_chunk(OGF_HEADER, &h1);
        let mut gcont = Vec::new();
        for v in [0u32, 0, 4, 0, 0, 9] {
            gcont.extend_from_slice(&v.to_le_bytes());
        }
        vis1.write_chunk(OGF_GCONTAINER, &gcont);
        vis1.write_chunk(OGF_SWICONTAINER, &0u32.to_le_bytes());

        let mut fp_w = ChunkWriter::new();
        fp_w.write_chunk(OGF_GCONTAINER, &gcont);
        let mut fp_swi = Vec::new();
        for r in [0u32; 4] {
            fp_swi.extend_from_slice(&r.to_le_bytes());
        }
        fp_swi.extend_from_slice(&2u32.to_le_bytes());
        fp_swi.extend_from_slice(&0u32.to_le_bytes());
        fp_swi.extend_from_slice(&2u16.to_le_bytes());
        fp_swi.extend_from_slice(&4u16.to_le_bytes());
        fp_swi.extend_from_slice(&6u32.to_le_bytes());
        fp_swi.extend_from_slice(&1u16.to_le_bytes());
        fp_swi.extend_from_slice(&4u16.to_le_bytes());
        fp_w.write_chunk(OGF_SWIDATA, &fp_swi);
        vis1.write_chunk(OGF_FASTPATH, &fp_w.into_bytes());

        // Visual 2: simple static Normal leaf
        let mut vis2 = ChunkWriter::new();
        let mut h2 = vec![4u8, OgfVisualType::Normal as u8, 0, 0];
        h2.extend_from_slice(&[0u8; 40]);
        vis2.write_chunk(OGF_HEADER, &h2);
        vis2.write_chunk(OGF_GCONTAINER, &gcont);

        let mut vis_wrap = ChunkWriter::new();
        vis_wrap.write_chunk(0, &vis0.into_bytes());
        vis_wrap.write_chunk(1, &vis1.into_bytes());
        vis_wrap.write_chunk(2, &vis2.into_bytes());

        let mut lvl_w = ChunkWriter::new();
        lvl_w.write_chunk(FSL_VISUALS, &vis_wrap.into_bytes());
        let level = LevelData::parse(&lvl_w.into_bytes()).unwrap();

        assert_eq!(level.collect_hierarchy_leaves(0), vec![1, 2]);

        let hi_lod = level.resolve_visual_mesh_lod(1, &geom, 1.0).unwrap();
        assert_eq!(hi_lod.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(hi_lod.to_position_only().len(), 4);

        let lo_lod = level.resolve_visual_mesh_lod(1, &geom, 0.0).unwrap();
        assert_eq!(lo_lod.indices, vec![0, 1, 3]);

        let fp_lo = level
            .resolve_fastpath_mesh_lod(1, &geom, 0.0)
            .unwrap()
            .unwrap();
        assert_eq!(fp_lo.indices, vec![0, 1, 3]);
    }

    #[test]
    fn test_level_glows_build_lights_torch_cfg_light_eval_and_sector_portal_culling() {
        use level_chunks::*;
        use crate::weather::parse_ltx_str;

        // 1. fsL_GLOWS (2 glow entries = 36B) + 2 portals + 3 sectors + shaders
        let mut shd = Vec::new();
        shd.extend_from_slice(&2u32.to_le_bytes());
        shd.extend_from_slice(b"\0");
        shd.extend_from_slice(b"effects\\glow/glow\\glow_white\0");

        let mut glows_raw = Vec::new();
        for (pos, r, sid) in [([1.0f32, 2.0, 3.0], 0.65f32, 1u16), ([8.0, 2.0, 0.0], 1.2, 1u16)] {
            for f in pos {
                glows_raw.extend_from_slice(&f.to_le_bytes());
            }
            glows_raw.extend_from_slice(&r.to_le_bytes());
            glows_raw.extend_from_slice(&sid.to_le_bytes());
        }

        let mut ports = Vec::new();
        let write_portal = |buf: &mut Vec<u8>, sf: u16, sb: u16, x: f32| {
            buf.extend_from_slice(&sf.to_le_bytes());
            buf.extend_from_slice(&sb.to_le_bytes());
            for pv in [
                [x, 0.0f32, 0.0],
                [x, 4.0, 0.0],
                [x, 4.0, 4.0],
                [x, 0.0, 4.0],
                [0.0; 3],
                [0.0; 3],
            ] {
                for f in pv {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            buf.extend_from_slice(&4u32.to_le_bytes());
        };
        write_portal(&mut ports, 0, 1, 5.0);
        write_portal(&mut ports, 1, 2, 50.0);

        let mut secs_w = ChunkWriter::new();
        for (sid, plist) in [(0u32, vec![0u16]), (1, vec![0u16, 1]), (2, vec![1u16])] {
            let mut sw = ChunkWriter::new();
            let mut pb = Vec::new();
            for p in plist {
                pb.extend_from_slice(&p.to_le_bytes());
            }
            sw.write_chunk(sector_chunks::FSP_PORTALS, &pb);
            secs_w.write_chunk(sid, &sw.into_bytes());
        }

        let mut lvl_w = ChunkWriter::new();
        lvl_w.write_chunk(FSL_SHADERS, &shd);
        lvl_w.write_chunk(FSL_PORTALS, &ports);
        lvl_w.write_chunk(FSL_SECTORS, &secs_w.into_bytes());
        lvl_w.write_chunk(FSL_GLOWS, &glows_raw);
        let level = LevelData::parse(&lvl_w.into_bytes()).unwrap();

        assert_eq!(level.glows.len(), 2);
        assert_eq!(level.glows_raw_len, 36);
        assert_eq!(level.glow_shader(&level.glows[0]).unwrap().shader, "effects\\glow");
        assert_eq!(level.portals[0].opposite_sector(0), Some(1));
        assert_eq!(level.portals[0].sector_facing([1.0, 2.0, 2.0]), Some(1));
        let reachable = level.collect_reachable_sectors_within_sphere(0, [2.0, 2.0, 2.0], 10.0);
        assert_eq!(reachable, vec![0, 1]);

        // 2. build.lights Chunk 1 (104-byte R_Light)
        let mut rlight = Vec::new();
        rlight.extend_from_slice(&build_light_type::POINT.to_le_bytes());
        rlight.extend_from_slice(&0u16.to_le_bytes());
        for f in [
            0.8f32, 0.7, 0.5, // diffuse
            2.0, 3.0, 4.0, // pos
            0.0, -1.0, 0.0, // dir
            12.0, 144.0, 0.0, // range, range2, falloff
            1.0, 0.1, 0.02, // attenuation
            5.0,  // energy
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, // tri[3]
        ] {
            rlight.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(rlight.len(), BUILD_LIGHT_RECORD_SIZE);
        let mut bl_w = ChunkWriter::new();
        bl_w.write_chunk(1, &rlight);
        let bl = BuildLights::parse(&bl_w.into_bytes()).unwrap();
        assert_eq!(bl.hemi.len(), 1);
        let ll = bl.hemi[0].to_level_light();
        assert!(ll.is_point());
        assert!((ll.evaluate_attenuation(0.0) - 1.0).abs() < 1e-6);
        assert!(ll.evaluate_attenuation(12.0) == 0.0);

        // 3. Spot cone + LAnim modulation + TorchLightConfig
        let mut anim = LightAnimItem::new("flicker");
        anim.fps = 10.0;
        anim.resize(2);
        anim.insert_key(0, crate::lanims::pack_argb(255, 255, 128, 0)).unwrap();

        let torch_doc = parse_ltx_str(
            r#"
[torch_definition]
type = 2
omni_type = 1
color = 0.4, 0.4, 0.4, 0.5
color_r2 = 0.8, 0.7, 0.6, 1.0
range = 15
range_r2 = 30
omni_color = 0.1, 0.1, 0.1, 0.2
omni_color_r2 = 0.3, 0.25, 0.2, 0.5
omni_range_r2 = 2.0
spot_angle = 60
spot_texture = internal\internal_light_torch_r2
color_animator = flicker
guide_bone = lights_bone
glow_texture = glow\glow_torch_r2
glow_radius = 0.35
volumetric = on
volumetric_intensity = 0.25
"#,
        );
        let torch_cfg = TorchLightConfig::from_ltx(&torch_doc, "torch_definition", true).unwrap();
        assert_eq!(torch_cfg.range, 30.0);
        assert!(torch_cfg.volumetric);
        let (spot, omni) = torch_cfg.to_level_light_pair([0.0, 2.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(spot.is_spot() && omni.is_point());
        assert!((spot.evaluate_spot_factor([0.0, 2.0, 10.0]) - 1.0).abs() < 1e-5);
        assert_eq!(spot.evaluate_spot_factor([0.0, 2.0, -5.0]), 0.0);
        let (_sc, sr) = spot.bounding_sphere();
        assert!(sr > 14.0 && sr < 25.0);
        let contrib = spot.evaluate_contribution_at([0.0, 2.0, 10.0], Some(&anim), 0.0, false);
        assert!(contrib[0] > 0.4 && contrib[2].abs() < 1e-6);
    }
}
