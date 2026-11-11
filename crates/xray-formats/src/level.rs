//! Pure Rust X-Ray `level` & `level.geom` / `level.geomx` World Scene Parser
//!
//! Faithful pure-Rust implementation matching `src/xrEngine/xrLevel.h`,
//! `src/Layers/xrRenderPC_R4/r4_loader.cpp`, `src/Layers/xrRender/r__sector.cpp`,
//! `src/Layers/xrRender/FVisual.cpp`, and `src/Layers/xrRender/Light_DB.cpp`.

use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::ogf::{OgfModel, OgfSlideWindows, SlideWindow, StaticVertex};

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

/// Occlusion sector from `level` chunk `fsL_SECTORS` (`CSector`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelSector {
    pub id: u32,
    pub portals: Vec<u16>,
    pub root_visual_id: Option<u32>,
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

/// Resolved drawable mesh extracted from a level visual + `LevelGeometry`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedLevelMesh {
    pub shader: Option<LevelShaderRef>,
    pub vertices: Vec<StaticVertex>,
    pub indices: Vec<u16>,
}

/// Parsed `level` scene description (`hdrLEVEL`, shaders, visuals, portals, sectors, lights).
#[derive(Debug, Clone, PartialEq)]
pub struct LevelData {
    pub xrlc_version: u16,
    pub xrlc_quality: u16,
    pub shaders: Vec<LevelShaderRef>,
    pub visuals: Vec<OgfModel>,
    pub portals: Vec<LevelPortal>,
    pub sectors: Vec<LevelSector>,
    pub lights: Vec<LevelLight>,
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

        let glows_raw_len = reader
            .open_chunk(FSL_GLOWS)?
            .map(|b| b.len())
            .unwrap_or(0);

        Ok(Self {
            xrlc_version,
            xrlc_quality,
            shaders,
            visuals,
            portals,
            sectors,
            lights,
            glows_raw_len,
        })
    }

    #[inline]
    pub fn visual_shader(&self, visual: &OgfModel) -> Option<&LevelShaderRef> {
        self.shaders.get(visual.header.shader_id as usize)
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
}
