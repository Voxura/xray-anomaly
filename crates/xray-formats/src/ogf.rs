//! Pure Rust X-Ray OGF v4 Static & Skeletal Mesh + Bone Parser
//!
//! Faithful pure-Rust implementation matching `src/xrEngine/Fmesh.h`,
//! `src/Layers/xrRender/FBasicVisual.cpp`, `FVisual.cpp`, `FProgressive.cpp`,
//! `FHierrarhyVisual.cpp`, `SkeletonCustom.cpp`, `SkeletonX.cpp`, `SkeletonAnimated.cpp`,
//! and `src/xrEngine/bone.h` / `bone.cpp`.

use crate::chunk::{ByteCursor, ChunkReader, FormatError};

pub const OGF_FORMAT_VERSION: u8 = 4;

/// OGF visual subtype (`enum MT` in `src/xrEngine/Fmesh.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OgfVisualType {
    Normal = 0,
    Hierarchy = 1,
    Progressive = 2,
    SkeletonAnim = 3,
    SkeletonGeomdefPm = 4,
    SkeletonGeomdefSt = 5,
    Lod = 6,
    TreeSt = 7,
    ParticleEffect = 8,
    ParticleGroup = 9,
    SkeletonRigid = 10,
    TreePm = 11,
    FluidVolume3D = 12,
}

impl TryFrom<u8> for OgfVisualType {
    type Error = FormatError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Normal),
            1 => Ok(Self::Hierarchy),
            2 => Ok(Self::Progressive),
            3 => Ok(Self::SkeletonAnim),
            4 => Ok(Self::SkeletonGeomdefPm),
            5 => Ok(Self::SkeletonGeomdefSt),
            6 => Ok(Self::Lod),
            7 => Ok(Self::TreeSt),
            8 => Ok(Self::ParticleEffect),
            9 => Ok(Self::ParticleGroup),
            10 => Ok(Self::SkeletonRigid),
            11 => Ok(Self::TreePm),
            12 => Ok(Self::FluidVolume3D),
            _ => Err(FormatError::InvalidValue("unknown OGF visual type")),
        }
    }
}

/// Chunk IDs inside an `.ogf` v4 stream (`enum OGF_Chuncks` in `Fmesh.h`).
pub mod ogf_chunks {
    pub const OGF_HEADER: u32 = 1;
    pub const OGF_TEXTURE: u32 = 2;
    pub const OGF_VERTICES: u32 = 3;
    pub const OGF_INDICES: u32 = 4;
    pub const OGF_P_MAP: u32 = 5;
    pub const OGF_SWIDATA: u32 = 6;
    pub const OGF_VCONTAINER: u32 = 7;
    pub const OGF_ICONTAINER: u32 = 8;
    pub const OGF_CHILDREN: u32 = 9;
    pub const OGF_CHILDREN_L: u32 = 10;
    pub const OGF_LODDEF2: u32 = 11;
    pub const OGF_TREEDEF2: u32 = 12;
    pub const OGF_S_BONE_NAMES: u32 = 13;
    pub const OGF_S_MOTIONS: u32 = 14;
    pub const OGF_S_SMPARAMS: u32 = 15;
    pub const OGF_S_IKDATA: u32 = 16;
    pub const OGF_S_USERDATA: u32 = 17;
    pub const OGF_S_DESC: u32 = 18;
    pub const OGF_S_MOTION_REFS: u32 = 19;
    pub const OGF_SWICONTAINER: u32 = 20;
    pub const OGF_GCONTAINER: u32 = 21;
    pub const OGF_FASTPATH: u32 = 22;
    pub const OGF_S_LODS: u32 = 23;
    pub const OGF_S_MOTION_REFS2: u32 = 24;
    pub const OGF_COLLISION_VERTICES: u32 = 25;
    pub const OGF_COLLISION_INDICES: u32 = 26;
}

/// Skeletal vertex format tags (`OGF_SkeletonVertType` + compact legacy aliases `1..=4`).
pub mod skel_vert_type {
    pub const FVF_1L: u32 = 0x1207_1980;
    pub const FVF_2L: u32 = 2 * 0x1207_1980;
    pub const FVF_NL: u32 = 3 * 0x1207_1980;
    pub const FVF_3L: u32 = 4 * 0x1207_1980;
    pub const FVF_4L: u32 = 5 * 0x1207_1980;
}

/// D3D9 FVF flags used by X-Ray static meshes (`FVisual.cpp`, `FVF.h`).
pub mod fvf_flags {
    pub const D3DFVF_POSITION_MASK: u32 = 0x400E;
    pub const D3DFVF_XYZ: u32 = 0x0002;
    pub const D3DFVF_XYZRHW: u32 = 0x0004;
    pub const D3DFVF_XYZB1: u32 = 0x0006;
    pub const D3DFVF_XYZB2: u32 = 0x0008;
    pub const D3DFVF_XYZB3: u32 = 0x000A;
    pub const D3DFVF_XYZB4: u32 = 0x000C;
    pub const D3DFVF_XYZB5: u32 = 0x000E;
    pub const D3DFVF_XYZW: u32 = 0x4002;

    pub const D3DFVF_NORMAL: u32 = 0x0010;
    pub const D3DFVF_PSIZE: u32 = 0x0020;
    pub const D3DFVF_DIFFUSE: u32 = 0x0040;
    pub const D3DFVF_SPECULAR: u32 = 0x0080;
    pub const D3DFVF_TEXCOUNT_MASK: u32 = 0x0F00;
    pub const D3DFVF_TEXCOUNT_SHIFT: u32 = 8;

    /// Standard X-Ray static geometry FVF (`XYZ | NORMAL | TEX1`).
    pub const XRAY_STATIC_FVF_STANDARD: u32 = D3DFVF_XYZ | D3DFVF_NORMAL | 0x0100;
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OgfBBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OgfBSphere {
    pub center: [f32; 3],
    pub radius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgfHeader {
    pub format_version: u8,
    pub visual_type: OgfVisualType,
    pub shader_id: u16,
    pub bbox: OgfBBox,
    pub bsphere: OgfBSphere,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OgfTextureLink {
    pub texture: String,
    pub shader: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticVertexLayout {
    pub fvf: u32,
    pub pos_floats: u8,
    pub is_rhw: bool,
    pub has_normal: bool,
    pub has_psize: bool,
    pub has_diffuse: bool,
    pub has_specular: bool,
    pub texcoord_dims: [u8; 8],
    pub texcoord_count: u8,
    pub stride: usize,
}

impl StaticVertexLayout {
    /// Decode D3D9 FVF flags into exact per-vertex layout and byte stride.
    pub fn from_fvf(fvf: u32) -> Result<Self, FormatError> {
        use fvf_flags::*;
        let pos_bits = fvf & D3DFVF_POSITION_MASK;
        let (pos_floats, is_rhw) = match pos_bits {
            D3DFVF_XYZ => (3, false),
            D3DFVF_XYZRHW => (4, true),
            D3DFVF_XYZW => (4, false),
            D3DFVF_XYZB1 => (4, false),
            D3DFVF_XYZB2 => (5, false),
            D3DFVF_XYZB3 => (6, false),
            D3DFVF_XYZB4 => (7, false),
            D3DFVF_XYZB5 => (8, false),
            _ => return Err(FormatError::InvalidValue("unsupported D3DFVF position bits")),
        };

        let has_normal = (fvf & D3DFVF_NORMAL) != 0;
        let has_psize = (fvf & D3DFVF_PSIZE) != 0;
        let has_diffuse = (fvf & D3DFVF_DIFFUSE) != 0;
        let has_specular = (fvf & D3DFVF_SPECULAR) != 0;

        let texcoord_count = ((fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT) as u8;
        if texcoord_count > 8 {
            return Err(FormatError::InvalidValue("D3DFVF texcoord count > 8"));
        }

        let mut texcoord_dims = [0u8; 8];
        let mut stride = (pos_floats as usize) * 4;
        if has_normal {
            stride += 12;
        }
        if has_psize {
            stride += 4;
        }
        if has_diffuse {
            stride += 4;
        }
        if has_specular {
            stride += 4;
        }

        for (i, dim) in texcoord_dims
            .iter_mut()
            .enumerate()
            .take(texcoord_count as usize)
        {
            let size_code = (fvf >> (16 + i * 2)) & 0b11;
            let floats = match size_code {
                0 => 2, // D3DFVF_TEXCOORDSIZE2
                1 => 3, // D3DFVF_TEXCOORDSIZE3
                2 => 4, // D3DFVF_TEXCOORDSIZE4
                3 => 1, // D3DFVF_TEXCOORDSIZE1
                _ => unreachable!(),
            };
            *dim = floats;
            stride += (floats as usize) * 4;
        }

        Ok(Self {
            fvf,
            pos_floats,
            is_rhw,
            has_normal,
            has_psize,
            has_diffuse,
            has_specular,
            texcoord_dims,
            texcoord_count,
            stride,
        })
    }

    pub fn decode_vertex(&self, bytes: &[u8]) -> Result<StaticVertex, FormatError> {
        if bytes.len() < self.stride {
            return Err(FormatError::UnexpectedEof);
        }
        let mut c = ByteCursor::new(bytes);
        let px = c.read_f32()?;
        let py = c.read_f32()?;
        let pz = c.read_f32()?;
        for _ in 3..self.pos_floats {
            let _ = c.read_f32()?;
        }
        let normal = if self.has_normal {
            c.read_f32_3()?
        } else {
            [0.0, 1.0, 0.0]
        };
        if self.has_psize {
            let _ = c.read_f32()?;
        }
        let diffuse = if self.has_diffuse {
            c.read_u32()?
        } else {
            0xFFFF_FFFF
        };
        let specular = if self.has_specular {
            c.read_u32()?
        } else {
            0
        };

        let mut uv0 = [0.0f32; 2];
        let mut uv1 = [0.0f32; 2];
        for stage in 0..(self.texcoord_count as usize) {
            let dims = self.texcoord_dims[stage] as usize;
            let mut vals = [0.0f32; 4];
            for v in vals.iter_mut().take(dims) {
                *v = c.read_f32()?;
            }
            if stage == 0 {
                uv0 = [vals[0], vals[1]];
            } else if stage == 1 {
                uv1 = [vals[0], vals[1]];
            }
        }

        Ok(StaticVertex {
            position: [px, py, pz],
            normal,
            diffuse,
            specular,
            uv0,
            uv1,
        })
    }
}

/// Normalized static vertex decoded from any D3DFVF static OGF stream.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StaticVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub diffuse: u32,
    pub specular: u32,
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
}

/// Exact 60-byte wire layout of `struct vertBoned1W` (`src/xrEngine/bone.h`).
#[repr(C, packed(2))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertBoned1W {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub binormal: [f32; 3],
    pub u: f32,
    pub v: f32,
    pub matrix: u32,
}

/// Exact 64-byte wire layout of `struct vertBoned2W` (`src/xrEngine/bone.h`).
#[repr(C, packed(2))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertBoned2W {
    pub matrix0: u16,
    pub matrix1: u16,
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub binormal: [f32; 3],
    pub weight: f32,
    pub u: f32,
    pub v: f32,
}

/// Exact 70-byte wire layout of `struct vertBoned3W` (`src/xrEngine/bone.h`).
#[repr(C, packed(2))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertBoned3W {
    pub matrices: [u16; 3],
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub binormal: [f32; 3],
    pub weights: [f32; 2],
    pub u: f32,
    pub v: f32,
}

/// Exact 76-byte wire layout of `struct vertBoned4W` (`src/xrEngine/bone.h`).
#[repr(C, packed(2))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertBoned4W {
    pub matrices: [u16; 4],
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub binormal: [f32; 3],
    pub weights: [f32; 3],
    pub u: f32,
    pub v: f32,
}

const _: () = assert!(std::mem::size_of::<VertBoned1W>() == 60);
const _: () = assert!(std::mem::size_of::<VertBoned2W>() == 64);
const _: () = assert!(std::mem::size_of::<VertBoned3W>() == 70);
const _: () = assert!(std::mem::size_of::<VertBoned4W>() == 76);

/// Unified skinned/static vertex representation for GPU upload or software skinning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnifiedSkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub binormal: [f32; 3],
    pub uv: [f32; 2],
    pub bone_indices: [u16; 4],
    pub bone_weights: [f32; 4],
}

#[derive(Debug, Clone, PartialEq)]
pub enum OgfVertices {
    Static {
        layout: StaticVertexLayout,
        vertices: Vec<StaticVertex>,
        raw_bytes: Vec<u8>,
    },
    Skinned1W(Vec<VertBoned1W>),
    Skinned2W(Vec<VertBoned2W>),
    Skinned3W(Vec<VertBoned3W>),
    Skinned4W(Vec<VertBoned4W>),
}

impl OgfVertices {
    pub fn len(&self) -> usize {
        match self {
            Self::Static { vertices, .. } => vertices.len(),
            Self::Skinned1W(v) => v.len(),
            Self::Skinned2W(v) => v.len(),
            Self::Skinned3W(v) => v.len(),
            Self::Skinned4W(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert any OGF vertex stream into unified 4-bone skinned vertices matching
    /// `xrSkin1W..4W` weight formulas (`src/xrCPU_Pipe/xrSkin2W.cpp`).
    pub fn to_unified_skinned(&self) -> Vec<UnifiedSkinnedVertex> {
        match self {
            Self::Static { vertices, .. } => vertices
                .iter()
                .map(|v| UnifiedSkinnedVertex {
                    position: v.position,
                    normal: v.normal,
                    tangent: [1.0, 0.0, 0.0],
                    binormal: [0.0, 0.0, 1.0],
                    uv: v.uv0,
                    bone_indices: [0, 0, 0, 0],
                    bone_weights: [1.0, 0.0, 0.0, 0.0],
                })
                .collect(),
            Self::Skinned1W(verts) => verts
                .iter()
                .map(|v| UnifiedSkinnedVertex {
                    position: v.position,
                    normal: v.normal,
                    tangent: v.tangent,
                    binormal: v.binormal,
                    uv: [v.u, v.v],
                    bone_indices: [v.matrix as u16, 0, 0, 0],
                    bone_weights: [1.0, 0.0, 0.0, 0.0],
                })
                .collect(),
            Self::Skinned2W(verts) => verts
                .iter()
                .map(|v| {
                    // X-Ray `xrSkin2W`: D->P.lerp(P0, P1, S->w) => (1 - w)*P0 + w*P1
                    let w1 = v.weight;
                    let w0 = 1.0 - w1;
                    UnifiedSkinnedVertex {
                        position: v.position,
                        normal: v.normal,
                        tangent: v.tangent,
                        binormal: v.binormal,
                        uv: [v.u, v.v],
                        bone_indices: [v.matrix0, v.matrix1, 0, 0],
                        bone_weights: [w0, w1, 0.0, 0.0],
                    }
                })
                .collect(),
            Self::Skinned3W(verts) => verts
                .iter()
                .map(|v| {
                    let w2 = 1.0 - v.weights[0] - v.weights[1];
                    UnifiedSkinnedVertex {
                        position: v.position,
                        normal: v.normal,
                        tangent: v.tangent,
                        binormal: v.binormal,
                        uv: [v.u, v.v],
                        bone_indices: [v.matrices[0], v.matrices[1], v.matrices[2], 0],
                        bone_weights: [v.weights[0], v.weights[1], w2, 0.0],
                    }
                })
                .collect(),
            Self::Skinned4W(verts) => verts
                .iter()
                .map(|v| {
                    let w3 = 1.0 - v.weights[0] - v.weights[1] - v.weights[2];
                    UnifiedSkinnedVertex {
                        position: v.position,
                        normal: v.normal,
                        tangent: v.tangent,
                        binormal: v.binormal,
                        uv: [v.u, v.v],
                        bone_indices: v.matrices,
                        bone_weights: [v.weights[0], v.weights[1], v.weights[2], w3],
                    }
                })
                .collect(),
        }
    }

    /// Extract flattened static vertex attributes `(position, normal, uv)` regardless of variant.
    pub fn to_static_attributes(&self) -> Vec<StaticVertex> {
        match self {
            Self::Static { vertices, .. } => vertices.clone(),
            _ => self
                .to_unified_skinned()
                .into_iter()
                .map(|u| StaticVertex {
                    position: u.position,
                    normal: u.normal,
                    diffuse: 0xFFFF_FFFF,
                    specular: 0,
                    uv0: u.uv,
                    uv1: [0.0, 0.0],
                })
                .collect(),
        }
    }
}

/// Sliding Window LOD record (`struct FSlideWindow` in `Fmesh.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlideWindow {
    pub offset: u32,
    pub num_tris: u16,
    pub num_verts: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OgfSlideWindows {
    pub reserved: [u32; 4],
    pub windows: Vec<SlideWindow>,
}

impl OgfSlideWindows {
    /// Select progressive mesh window for `lod` (`0.0..=1.0`, `1.0 = max detail`)
    /// matching `FProgressive::Render` / `FTreeVisual_PM::Render`.
    pub fn select_window(&self, lod: f32) -> Option<SlideWindow> {
        if self.windows.is_empty() {
            return None;
        }
        let clamped = lod.clamp(0.0, 1.0);
        let max_idx = (self.windows.len() - 1) as f32;
        let lod_id = ((1.0 - clamped) * max_idx + 0.5).floor() as usize;
        self.windows.get(lod_id.min(self.windows.len() - 1)).copied()
    }
}

/// Quad vertex reordering index order used by `R_dsgraph_structure::r_dsgraph_render_lods` with `QuadIB`.
pub const LOD_IMPOSTER_QUAD_ORDER: [usize; 4] = [3, 0, 2, 1];

/// Wire stride (bytes) of `FLOD::_vertex` (`Fvector v + Fvector2 t + u32 c_rgb_hemi + u8 c_sun + 3B align`).
pub const OGF_LOD_VERTEX_SIZE: usize = 28;
/// Total byte size of `OGF_LODDEF2` (`8 * 4 * 28 = 896` bytes).
pub const OGF_LODDEF2_SIZE: usize = 8 * 4 * OGF_LOD_VERTEX_SIZE;
/// Total byte size of `OGF_TREEDEF2` (`XRayMatrix (64B) + c_scale (20B) + c_bias (20B)`).
pub const OGF_TREEDEF2_SIZE: usize = 104;

/// Single imposter vertex inside an `OGF_LODDEF2` facet (`FLOD::_vertex`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgfLodVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    pub rgb_hemi: u32,
    pub sun: u8,
}

/// One of the 8 octahedral billboard facets stored in `OGF_LODDEF2` (`FLOD::_face`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgfLodFacet {
    pub vertices: [OgfLodVertex; 4],
    /// Outward-facing unit normal computed matching `FLOD::Load`.
    pub normal: [f32; 3],
}

/// Selected pair of adjacent imposter facets and blend factor (`r__dsgraph_render_lods.cpp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodImposterSample {
    /// Primary (highest dot product) facet index (`FA`).
    pub best_facet_index: usize,
    /// Secondary (second highest dot product) facet index (`FB`).
    pub next_facet_index: usize,
    /// Blend weight in `0.0..=1.0` between `next_facet_index` and `best_facet_index` (`uF / 255`).
    pub blend_factor: f32,
    /// Forward depth bias shift `-0.5 * bsphere.radius * normalize(bsphere.center - camera_pos)`.
    pub shift: [f32; 3],
}

/// Complete `OGF_LODDEF2` billboard imposter definition + spherical `lod_factor` (`FLOD`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgfLodDef {
    pub facets: [OgfLodFacet; 8],
    /// Ratio `Sf / Ss` derived from OGF bounding box & bounding sphere (`FLOD::Load`).
    pub lod_factor: f32,
}

impl OgfLodDef {
    /// Select the top 2 view-aligned facets and compute blend weight matching `r__dsgraph_render_lods.cpp`.
    pub fn sample_imposter_pair(
        &self,
        camera_pos: [f32; 3],
        bsphere: OgfBSphere,
    ) -> LodImposterSample {
        let mut l_dir = [
            bsphere.center[0] - camera_pos[0],
            bsphere.center[1] - camera_pos[1],
            bsphere.center[2] - camera_pos[2],
        ];
        let len_sq = l_dir[0] * l_dir[0] + l_dir[1] * l_dir[1] + l_dir[2] * l_dir[2];
        if len_sq > 1e-12 {
            let inv = 1.0 / len_sq.sqrt();
            l_dir[0] *= inv;
            l_dir[1] *= inv;
            l_dir[2] *= inv;
        } else {
            l_dir = [0.0, 0.0, 1.0];
        }
        let shift = [
            -0.5 * bsphere.radius * l_dir[0],
            -0.5 * bsphere.radius * l_dir[1],
            -0.5 * bsphere.radius * l_dir[2],
        ];

        let mut scored: [(f32, usize); 8] = std::array::from_fn(|s| {
            let n = self.facets[s].normal;
            (l_dir[0] * n[0] + l_dir[1] * n[1] + l_dir[2] * n[2], s)
        });
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let (f_a, best_idx) = scored[7];
        let (f_b, next_idx) = scored[6];
        let f_c = scored[5].0;
        let denom = f_a - f_c;
        let blend_factor = if denom.abs() > 1e-6 {
            (0.5 + 0.5 * (1.0 - (f_b - f_c) / denom)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        LodImposterSample {
            best_facet_index: best_idx,
            next_facet_index: next_idx,
            blend_factor,
            shift,
        }
    }

    /// Pack the 4 reordered hardware billboard vertices (`FLOD::_hw`, 76 bytes each) matching
    /// `R_dsgraph_structure::r_dsgraph_render_lods` (`r__dsgraph_render_lods.cpp:58-105`).
    pub fn build_imposter_quad(
        &self,
        camera_pos: [f32; 3],
        bsphere: OgfBSphere,
        ssa: f32,
        thresholds: &ScreenLodThresholds,
    ) -> [LodImposterHwVertex; 4] {
        let sample = self.sample_imposter_pair(camera_pos, bsphere);
        let u_a = thresholds.imposter_alpha(ssa);
        let u_f = (sample.blend_factor * 255.5).floor().clamp(0.0, 255.0) as u8;
        let fa = &self.facets[sample.best_facet_index];
        let fb = &self.facets[sample.next_facet_index];

        std::array::from_fn(|vit| {
            let id = LOD_IMPOSTER_QUAD_ORDER[vit];
            let vb = &fb.vertices[id];
            let va = &fa.vertices[id];
            // X-Ray `color_rgba(r, g, b, a)` = D3DCOLOR BGRA little-endian `(a<<24)|(r<<16)|(g<<8)|b`
            let sun_af = ((u_f as u32) << 24)
                | ((vb.sun as u32) << 16)
                | ((va.sun as u32) << 8)
                | (u_a as u32);
            LodImposterHwVertex {
                p0: [
                    vb.position[0] + sample.shift[0],
                    vb.position[1] + sample.shift[1],
                    vb.position[2] + sample.shift[2],
                ],
                p1: [
                    va.position[0] + sample.shift[0],
                    va.position[1] + sample.shift[1],
                    va.position[2] + sample.shift[2],
                ],
                n0: fb.normal,
                n1: fa.normal,
                sun_af,
                t0: vb.uv,
                t1: va.uv,
                rgbh0: vb.rgb_hemi,
                rgbh1: va.rgb_hemi,
            }
        })
    }
}

/// 76-byte hardware vertex packed by `r_dsgraph_render_lods` (`FLOD::_hw` in `FLOD.h:22-33`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LodImposterHwVertex {
    pub p0: [f32; 3],
    pub p1: [f32; 3],
    pub n0: [f32; 3],
    pub n1: [f32; 3],
    pub sun_af: u32,
    pub t0: [f32; 2],
    pub t1: [f32; 2],
    pub rgbh0: u32,
    pub rgbh1: u32,
}

const _: () = assert!(std::mem::size_of::<LodImposterHwVertex>() == 76);

/// Screen-space LOD console defaults (`xrRender_console.cpp:142-205`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenLodConfig {
    pub r_lod: f32,
    pub ssa_discard: f32,
    pub ssa_dont_sort: f32,
    pub ssa_lod_a: f32,
    pub ssa_lod_b: f32,
    pub ssa_glod_start: f32,
    pub ssa_glod_end: f32,
    pub ssa_hzb_vs_tex: f32,
    pub parallax_range: f32,
}

impl Default for ScreenLodConfig {
    fn default() -> Self {
        Self {
            r_lod: 0.75,
            ssa_discard: 3.5,
            ssa_dont_sort: 32.0,
            ssa_lod_a: 64.0,
            ssa_lod_b: 48.0,
            ssa_glod_start: 256.0,
            ssa_glod_end: 64.0,
            ssa_hzb_vs_tex: 96.0,
            parallax_range: 75.0,
        }
    }
}

/// Per-frame normalized SSA LOD thresholds computed in `CRender::Calculate`
/// (`r2_R_calculate.cpp:17-27`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenLodThresholds {
    pub g_f_screen: f32,
    pub r_ssa_discard: f32,
    pub r_ssa_dont_sort: f32,
    pub r_ssa_lod_a: f32,
    pub r_ssa_lod_b: f32,
    pub r_ssa_glod_start: f32,
    pub r_ssa_glod_end: f32,
    pub r_ssa_hzb_vs_tex: f32,
    pub r_dtex_range: f32,
}

impl ScreenLodThresholds {
    /// True when the visual should be discarded below `r_ssaDISCARD`
    /// (`r__dsgraph_build.cpp:67, 402`).
    #[inline]
    pub fn should_discard(&self, ssa: f32) -> bool {
        ssa <= self.r_ssa_discard
    }

    /// Compute normalized `0.0..=1.0` progressive mesh LOD weight from `ssa` matching `calcLOD`
    /// (`r__dsgraph_render.cpp:19-22`).
    #[inline]
    pub fn calc_lod(&self, ssa: f32) -> f32 {
        let span = self.r_ssa_glod_start - self.r_ssa_glod_end;
        if span.abs() <= 1e-12 {
            if ssa >= self.r_ssa_glod_start {
                1.0
            } else {
                0.0
            }
        } else {
            ((ssa - self.r_ssa_glod_end) / span).clamp(0.0, 1.0).sqrt()
        }
    }

    /// Compute quantized hardware mip-LOD factor (`1.0..=7.0`) matching `R_LOD::set_LOD`
    /// (`R_Backend_LOD.cpp:13`).
    #[inline]
    pub fn hardware_lod_factor(&self, ssa: f32) -> f32 {
        let lod = self.calc_lod(ssa);
        (lod.powi(5) * 8.0).ceil().clamp(1.0, 7.0)
    }

    /// Compute quantized 8-bit billboard imposter alpha `uA` matching `r__dsgraph_render_lods.cpp:59-62`.
    #[inline]
    pub fn imposter_alpha(&self, ssa: f32) -> u8 {
        let ssa_range = (self.r_ssa_lod_a - self.r_ssa_lod_b).max(1e-7);
        let ssa_diff = ssa - self.r_ssa_lod_b;
        let scale = ssa_diff / ssa_range;
        ((1.0 - scale) * 255.0).floor().clamp(0.0, 255.0) as u8
    }
}

/// Compute screen-space area proxy `(ssa, dist_sq)` matching `CalcSSA`
/// (`r__dsgraph_build.cpp:23-34`).
#[inline]
pub fn calc_ssa(cam_pos: [f32; 3], center: [f32; 3], radius: f32) -> (f32, f32) {
    const EPS: f32 = 1e-5;
    let dx = cam_pos[0] - center[0];
    let dy = cam_pos[1] - center[1];
    let dz = cam_pos[2] - center[2];
    let dist_sq = dx * dx + dy * dy + dz * dz + EPS;
    (radius / dist_sq, dist_sq)
}

/// Active renderer geometry pass (`enum { PHASE_NORMAL = 0, PHASE_SMAP = 1 }` in `r4.h:31-37`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RenderPhase {
    Normal = 0,
    Smap = 1,
}

/// Deffer shader element indices (`r2_types.h:146-148`, `r4.cpp:94`, `r__dsgraph_build.cpp:149`).
pub mod deffer_shader_elements {
    pub const SE_R2_NORMAL_HQ: usize = 0;
    pub const SE_R2_NORMAL_LQ: usize = 1;
    pub const SE_R2_SHADOW: usize = 2;
    pub const SE_R2_TERRAIN_VERY_LOW: usize = 3;
    pub const SE_R2_SPECIAL: usize = 4;
}

/// Select deffer shader element (`0..=2`) for a dynamic visual (`CRender::rimp_select_sh_dynamic`,
/// `r4.cpp:71-79`).
#[inline]
pub fn select_dynamic_shader_element(
    phase: RenderPhase,
    dist_sq: f32,
    sphere_radius: f32,
    r_dtex_range: f32,
) -> usize {
    if phase == RenderPhase::Normal {
        let sec_dist = dist_sq.max(0.0).sqrt() - sphere_radius;
        if sec_dist < r_dtex_range {
            deffer_shader_elements::SE_R2_NORMAL_HQ
        } else {
            deffer_shader_elements::SE_R2_NORMAL_LQ
        }
    } else {
        deffer_shader_elements::SE_R2_SHADOW
    }
}

/// Select deffer shader element (`0..=3`) for a static sector visual (`CRender::rimp_select_sh_static`,
/// `r4.cpp:82-102`). Default `ssfx_terrain_quality_x` is `8.0` (`80.0m` HQ terrain threshold).
#[inline]
pub fn select_static_shader_element(
    phase: RenderPhase,
    is_landscape: bool,
    dist_sq: f32,
    sphere_radius: f32,
    r_dtex_range: f32,
    ssfx_terrain_quality_x: f32,
) -> usize {
    if phase != RenderPhase::Normal {
        return deffer_shader_elements::SE_R2_SHADOW;
    }
    let sec_dist = dist_sq.max(0.0).sqrt() - sphere_radius;
    if is_landscape {
        if sec_dist > 240.0 {
            deffer_shader_elements::SE_R2_TERRAIN_VERY_LOW
        } else if sec_dist < ssfx_terrain_quality_x * 10.0 {
            deffer_shader_elements::SE_R2_NORMAL_HQ
        } else {
            deffer_shader_elements::SE_R2_NORMAL_LQ
        }
    } else if sec_dist < r_dtex_range {
        deffer_shader_elements::SE_R2_NORMAL_HQ
    } else {
        deffer_shader_elements::SE_R2_NORMAL_LQ
    }
}

/// Reference FOV used by `GetDistFromCamera` (`BASE_FOV = 67.f` in `r__dsgraph_build.cpp:695`).
pub const VISUAL_OPT_BASE_FOV: f32 = 67.0;

/// Static geometry optimization distance cutoffs per `[L1..L5]` across `[LOW, MED, HII, ULT]`
/// (`o_optimize_static_l*_dist` in `r__dsgraph_build.cpp:603-645, 677-686`).
pub const VISUAL_OPT_STATIC_DIST: [[f32; 4]; 5] = [
    [150.0, 50.0, 50.0, 35.0],
    [200.0, 150.0, 150.0, 125.0],
    [250.0, 200.0, 200.0, 175.0],
    [350.0, 300.0, 300.0, 250.0],
    [400.0, 400.0, 350.0, 300.0],
];

/// Static geometry optimization bounding-sphere volume thresholds (`o_optimize_static_l*_size`).
pub const VISUAL_OPT_STATIC_VOL: [[f32; 4]; 5] = [
    [10.0, 25.0, 50.0, 50.0],
    [100.0, 200.0, 400.0, 500.0],
    [500.0, 1000.0, 1500.0, 1750.0],
    [2500.0, 2500.0, 5000.0, 5250.0],
    [7000.0, 7000.0, 20000.0, 25000.0],
];

/// Dynamic geometry optimization distance cutoffs per `[L1..L3]` across `[LOW, MED, HII, ULT]`
/// (`o_optimize_dynamic_l*_dist` in `r__dsgraph_build.cpp:649-675, 688-693`).
pub const VISUAL_OPT_DYNAMIC_DIST: [[f32; 4]; 3] = [
    [80.0, 40.0, 30.0, 30.0],
    [150.0, 100.0, 80.0, 50.0],
    [250.0, 200.0, 150.0, 110.0],
];

/// Dynamic geometry optimization bounding-sphere volume thresholds (`o_optimize_dynamic_l*_size`).
pub const VISUAL_OPT_DYNAMIC_VOL: [[f32; 4]; 3] = [
    [1.0, 1.0, 1.4, 2.0],
    [3.0, 4.0, 4.0, 8.0],
    [4000.0, 4000.0, 4000.0, 4000.0],
];

/// Console configuration for `IsValuableToRender` geometry distance/volume culling
/// (`xrRender_console.cpp:481, 535-536`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualOptConfig {
    pub opt_static: i32,
    pub opt_dynamic: i32,
    pub opt_shadow_geom: bool,
    pub far_shadow_cascade_dist: f32,
}

impl Default for VisualOptConfig {
    fn default() -> Self {
        Self {
            opt_static: 2,
            opt_dynamic: 2,
            opt_shadow_geom: false,
            far_shadow_cascade_dist: 160.0,
        }
    }
}

/// Sphere volume `(4 * PI / 3) * radius^3` (`Fsphere::volume`, `src/xrCore/_sphere.h:191-194`).
#[inline]
pub fn sphere_volume(radius: f32) -> f32 {
    (4.0 * std::f32::consts::PI / 3.0) * (radius * radius * radius)
}

/// FOV-normalized distance matching `GetDistFromCamera` (`r__dsgraph_build.cpp:697-705`).
#[inline]
pub fn adjusted_distance_from_camera(
    cam_pos: [f32; 3],
    from_position: [f32; 3],
    fov_deg: f32,
) -> f32 {
    let dx = cam_pos[0] - from_position[0];
    let dy = cam_pos[1] - from_position[1];
    let dz = cam_pos[2] - from_position[2];
    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
    let fov_k = VISUAL_OPT_BASE_FOV / fov_deg.max(1e-3);
    distance / fov_k
}

#[inline]
fn passes_opt_tiers<const N: usize>(
    vol: f32,
    adj_dist: f32,
    col: usize,
    vols: &[[f32; 4]; N],
    dists: &[[f32; 4]; N],
) -> bool {
    for l in 0..N {
        if vol < vols[l][col] && adj_dist > dists[l][col] {
            return false;
        }
    }
    true
}

#[inline]
fn opt_level_to_col(level: i32) -> usize {
    match level {
        2 => 1,
        3 => 2,
        4 => 3,
        _ => 0,
    }
}

/// Static geometry `IsValuableToRender` test (`r__dsgraph_build.cpp:707-820`).
pub fn is_valuable_to_render_static(
    visual_type: OgfVisualType,
    sphere: OgfBSphere,
    cam_pos: [f32; 3],
    fov_deg: f32,
    is_shadow_map: bool,
    cfg: &VisualOptConfig,
) -> bool {
    if cfg.opt_static < 1 {
        return true;
    }
    let vol = sphere_volume(sphere.radius);
    let adj_dist = adjusted_distance_from_camera(cam_pos, sphere.center, fov_deg);
    if is_shadow_map && cfg.opt_shadow_geom {
        if vol < 50_000.0 && adj_dist > cfg.far_shadow_cascade_dist {
            return false;
        }
        return passes_opt_tiers(vol, adj_dist, 2, &VISUAL_OPT_STATIC_VOL, &VISUAL_OPT_STATIC_DIST);
    }
    if matches!(
        visual_type,
        OgfVisualType::Lod | OgfVisualType::TreePm | OgfVisualType::TreeSt
    ) {
        return true;
    }
    passes_opt_tiers(
        vol,
        adj_dist,
        opt_level_to_col(cfg.opt_static),
        &VISUAL_OPT_STATIC_VOL,
        &VISUAL_OPT_STATIC_DIST,
    )
}

/// Dynamic geometry `IsValuableToRender` test (`r__dsgraph_build.cpp:707-745, 822-867`).
pub fn is_valuable_to_render_dynamic(
    sphere: OgfBSphere,
    transform: Option<&XRayMatrix>,
    cam_pos: [f32; 3],
    fov_deg: f32,
    is_shadow_map: bool,
    cfg: &VisualOptConfig,
) -> bool {
    if cfg.opt_dynamic < 1 {
        return true;
    }
    let vol = sphere_volume(sphere.radius);
    let world_center = match transform {
        Some(xf) => matrix_transform_point(xf, sphere.center),
        None => sphere.center,
    };
    let adj_dist = adjusted_distance_from_camera(cam_pos, world_center, fov_deg);
    if is_shadow_map && cfg.opt_shadow_geom {
        if vol < 50_000.0 && adj_dist > cfg.far_shadow_cascade_dist {
            return false;
        }
        return passes_opt_tiers(vol, adj_dist, 2, &VISUAL_OPT_STATIC_VOL, &VISUAL_OPT_STATIC_DIST);
    }
    passes_opt_tiers(
        vol,
        adj_dist,
        opt_level_to_col(cfg.opt_dynamic),
        &VISUAL_OPT_DYNAMIC_VOL,
        &VISUAL_OPT_DYNAMIC_DIST,
    )
}

/// Destination scene-graph bucket produced by `r_dsgraph_insert_{dynamic,static}`
/// (`r__dsgraph_build.cpp:36-600`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualPrimaryBucket {
    CustomFovDistort,
    CustomFovEmissive,
    CustomFovSorted,
    CustomFovOpaque,
    HudDistort,
    HudEmissive,
    HudSorted,
    HudOpaque,
    CamAttachedEmissive,
    CamAttachedSorted,
    CamAttachedOpaque,
    ScopeHudOpaque,
    ScopeHudSorted,
    ScopeCamAttachedOpaque,
    ScopeCamAttachedSorted,
    Ui3dSorted,
    Ui3dOpaque,
    Ui3dEmissive,
    Distort,
    Sorted,
    Water,
    Wmark,
    Emissive,
    Landscape,
    NormalOpaque,
    MatrixOpaque,
}

/// Shader/context flags driving scene-graph bucket routing (`r__dsgraph_build.cpp:36-600`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VisualRoutingFlags {
    pub is_hud: bool,
    pub is_cam_attached: bool,
    pub is_ui_3d: bool,
    pub custom_fov: f32,
    pub is_invisible: bool,
    pub priority: u8,
    pub is_strict_b2f: bool,
    pub is_emissive: bool,
    pub is_distort: bool,
    pub special_priority: u8,
    pub is_wmark: bool,
    pub is_water: bool,
    pub is_landscape: bool,
    pub scope_lense: u8,
}

/// Complete scene-graph submission decision for one visual draw (`r__dsgraph_build.cpp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualRoutingDecision {
    pub ssa: f32,
    pub dist_sq: f32,
    pub shader_element: usize,
    pub primary: Option<VisualPrimaryBucket>,
    pub distort_bucket: Option<VisualPrimaryBucket>,
    pub emissive_bucket: Option<VisualPrimaryBucket>,
    pub custom_fov: f32,
}

/// Route a dynamic visual into R4 scene-graph buckets (`R_dsgraph_structure::r_dsgraph_insert_dynamic`,
/// `r__dsgraph_build.cpp:36-386`). Returns `None` if discarded by SSA or all buckets filtered out.
#[allow(clippy::too_many_arguments)]
pub fn route_dynamic_visual(
    phase: RenderPhase,
    sphere_center_world: [f32; 3],
    sphere_radius: f32,
    cam_pos: [f32; 3],
    thresholds: &ScreenLodThresholds,
    flags: &VisualRoutingFlags,
    pmask: &[bool; 2],
    pmask_wmark: bool,
    distortion_enabled: bool,
) -> Option<VisualRoutingDecision> {
    let (ssa, dist_sq) = calc_ssa(cam_pos, sphere_center_world, sphere_radius);
    if thresholds.should_discard(ssa) {
        return None;
    }

    let spec_prio = (flags.special_priority as usize / 2).min(1);
    let distort_bucket = if distortion_enabled && flags.is_distort && pmask[spec_prio] {
        Some(if flags.custom_fov != 0.0 {
            VisualPrimaryBucket::CustomFovDistort
        } else if flags.is_hud {
            VisualPrimaryBucket::HudDistort
        } else {
            VisualPrimaryBucket::Distort
        })
    } else {
        None
    };

    let shader_element =
        select_dynamic_shader_element(phase, dist_sq, sphere_radius, thresholds.r_dtex_range);
    let main_prio = (flags.priority as usize / 2).min(1);
    if !pmask[main_prio] && !flags.is_ui_3d {
        return distort_bucket.map(|_| VisualRoutingDecision {
            ssa,
            dist_sq,
            shader_element,
            primary: None,
            distort_bucket,
            emissive_bucket: None,
            custom_fov: flags.custom_fov,
        });
    }

    match flags.scope_lense {
        1 => {
            return Some(VisualRoutingDecision {
                ssa,
                dist_sq,
                shader_element,
                primary: Some(if flags.is_cam_attached {
                    VisualPrimaryBucket::CamAttachedOpaque
                } else {
                    VisualPrimaryBucket::HudOpaque
                }),
                distort_bucket,
                emissive_bucket: None,
                custom_fov: flags.custom_fov,
            });
        }
        2 => {
            return Some(VisualRoutingDecision {
                ssa,
                dist_sq,
                shader_element,
                primary: Some(if flags.is_cam_attached {
                    VisualPrimaryBucket::ScopeCamAttachedOpaque
                } else {
                    VisualPrimaryBucket::ScopeHudOpaque
                }),
                distort_bucket,
                emissive_bucket: None,
                custom_fov: flags.custom_fov,
            });
        }
        3 => {
            return Some(VisualRoutingDecision {
                ssa,
                dist_sq,
                shader_element,
                primary: Some(if flags.is_cam_attached {
                    VisualPrimaryBucket::ScopeCamAttachedSorted
                } else {
                    VisualPrimaryBucket::ScopeHudSorted
                }),
                distort_bucket,
                emissive_bucket: None,
                custom_fov: flags.custom_fov,
            });
        }
        _ => {}
    }

    if flags.is_hud {
        let emissive_bucket = if flags.is_emissive {
            Some(if flags.custom_fov != 0.0 {
                VisualPrimaryBucket::CustomFovEmissive
            } else if flags.is_cam_attached {
                VisualPrimaryBucket::CamAttachedEmissive
            } else {
                VisualPrimaryBucket::HudEmissive
            })
        } else {
            None
        };
        let primary = Some(if flags.is_strict_b2f {
            if flags.custom_fov != 0.0 {
                VisualPrimaryBucket::CustomFovSorted
            } else if flags.is_cam_attached {
                VisualPrimaryBucket::CamAttachedSorted
            } else {
                VisualPrimaryBucket::HudSorted
            }
        } else if flags.custom_fov != 0.0 {
            VisualPrimaryBucket::CustomFovOpaque
        } else if flags.is_cam_attached {
            VisualPrimaryBucket::CamAttachedOpaque
        } else {
            VisualPrimaryBucket::HudOpaque
        });
        return Some(VisualRoutingDecision {
            ssa,
            dist_sq,
            shader_element,
            primary,
            distort_bucket,
            emissive_bucket,
            custom_fov: flags.custom_fov,
        });
    }

    if flags.is_ui_3d {
        let (primary, emissive_bucket) = if flags.is_strict_b2f {
            (Some(VisualPrimaryBucket::Ui3dSorted), None)
        } else {
            (
                Some(VisualPrimaryBucket::Ui3dOpaque),
                flags.is_emissive.then_some(VisualPrimaryBucket::Ui3dEmissive),
            )
        };
        return Some(VisualRoutingDecision {
            ssa,
            dist_sq,
            shader_element,
            primary,
            distort_bucket,
            emissive_bucket,
            custom_fov: 0.0,
        });
    }

    if flags.is_invisible {
        return distort_bucket.map(|_| VisualRoutingDecision {
            ssa,
            dist_sq,
            shader_element,
            primary: None,
            distort_bucket,
            emissive_bucket: None,
            custom_fov: flags.custom_fov,
        });
    }

    if flags.is_strict_b2f {
        return Some(VisualRoutingDecision {
            ssa,
            dist_sq,
            shader_element,
            primary: Some(if flags.custom_fov != 0.0 {
                VisualPrimaryBucket::CustomFovSorted
            } else {
                VisualPrimaryBucket::Sorted
            }),
            distort_bucket,
            emissive_bucket: None,
            custom_fov: flags.custom_fov,
        });
    }

    let emissive_bucket = if flags.is_emissive {
        Some(if flags.custom_fov != 0.0 {
            VisualPrimaryBucket::CustomFovEmissive
        } else {
            VisualPrimaryBucket::Emissive
        })
    } else {
        None
    };

    let primary = Some(if flags.is_wmark && pmask_wmark {
        VisualPrimaryBucket::Wmark
    } else {
        VisualPrimaryBucket::MatrixOpaque
    });

    Some(VisualRoutingDecision {
        ssa,
        dist_sq,
        shader_element,
        primary,
        distort_bucket,
        emissive_bucket,
        custom_fov: flags.custom_fov,
    })
}

/// Route a static level visual into R4 scene-graph buckets (`R_dsgraph_structure::r_dsgraph_insert_static`,
/// `r__dsgraph_build.cpp:388-600`).
#[allow(clippy::too_many_arguments)]
pub fn route_static_visual(
    phase: RenderPhase,
    sphere: OgfBSphere,
    cam_pos: [f32; 3],
    thresholds: &ScreenLodThresholds,
    flags: &VisualRoutingFlags,
    pmask: &[bool; 2],
    pmask_wmark: bool,
    distortion_enabled: bool,
    ssfx_water_enabled: bool,
    ssfx_terrain_quality_x: f32,
) -> Option<VisualRoutingDecision> {
    let (ssa, dist_sq) = calc_ssa(cam_pos, sphere.center, sphere.radius);
    if thresholds.should_discard(ssa) {
        return None;
    }

    let spec_prio = (flags.special_priority as usize / 2).min(1);
    let distort_bucket = if distortion_enabled && flags.is_distort && pmask[spec_prio] {
        Some(VisualPrimaryBucket::Distort)
    } else {
        None
    };

    let shader_element = select_static_shader_element(
        phase,
        flags.is_landscape,
        dist_sq,
        sphere.radius,
        thresholds.r_dtex_range,
        ssfx_terrain_quality_x,
    );
    let main_prio = (flags.priority as usize / 2).min(1);
    if !pmask[main_prio] {
        return distort_bucket.map(|_| VisualRoutingDecision {
            ssa,
            dist_sq,
            shader_element,
            primary: None,
            distort_bucket,
            emissive_bucket: None,
            custom_fov: 0.0,
        });
    }

    if flags.is_water && ssfx_water_enabled {
        return Some(VisualRoutingDecision {
            ssa,
            dist_sq,
            shader_element,
            primary: Some(VisualPrimaryBucket::Water),
            distort_bucket,
            emissive_bucket: None,
            custom_fov: 0.0,
        });
    }

    if flags.is_strict_b2f {
        return Some(VisualRoutingDecision {
            ssa,
            dist_sq,
            shader_element,
            primary: Some(VisualPrimaryBucket::Sorted),
            distort_bucket,
            emissive_bucket: None,
            custom_fov: 0.0,
        });
    }

    let emissive_bucket = flags.is_emissive.then_some(VisualPrimaryBucket::Emissive);
    let primary = if flags.is_wmark && pmask_wmark {
        VisualPrimaryBucket::Wmark
    } else if flags.is_landscape && phase == RenderPhase::Normal {
        VisualPrimaryBucket::Landscape
    } else {
        VisualPrimaryBucket::NormalOpaque
    };

    Some(VisualRoutingDecision {
        ssa,
        dist_sq,
        shader_element,
        primary: Some(primary),
        distort_bucket,
        emissive_bucket,
        custom_fov: 0.0,
    })
}

/// Compute normalized SSA thresholds and parallax detail range for `(screen_w, screen_h, fov_deg)`.
pub fn compute_screen_lod_thresholds(
    screen_w: u32,
    screen_h: u32,
    fov_deg: f32,
    cfg: &ScreenLodConfig,
) -> ScreenLodThresholds {
    let fov_factor = (90.0 / fov_deg.clamp(5.0, 175.0)).powi(2);
    let g_f_screen = ((screen_w.max(1) as f32 * screen_h.max(1) as f32)
        * fov_factor
        * (1e-7 + cfg.r_lod))
        .max(1e-7);
    ScreenLodThresholds {
        g_f_screen,
        r_ssa_discard: (cfg.ssa_discard * cfg.ssa_discard) / g_f_screen,
        r_ssa_dont_sort: (cfg.ssa_dont_sort / 3.0).powi(2) / g_f_screen,
        r_ssa_lod_a: (cfg.ssa_lod_a / 3.0).powi(2) / g_f_screen,
        r_ssa_lod_b: (cfg.ssa_lod_b / 3.0).powi(2) / g_f_screen,
        r_ssa_glod_start: (cfg.ssa_glod_start / 3.0).powi(2) / g_f_screen,
        r_ssa_glod_end: (cfg.ssa_glod_end / 3.0).powi(2) / g_f_screen,
        r_ssa_hzb_vs_tex: (cfg.ssa_hzb_vs_tex / 3.0).powi(2) / g_f_screen,
        r_dtex_range: cfg.parallax_range * g_f_screen / (1024.0 * 768.0),
    }
}

/// 5-channel vertex lighting scale/bias parameters from `OGF_TREEDEF2` (`FTreeVisual::_5color`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TreeLightingColor {
    pub rgb: [f32; 3],
    pub hemi: f32,
    pub sun: f32,
}

/// Foliage placement transform and lighting parameters parsed from `OGF_TREEDEF2` (`FTreeVisual`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgfTreeDef {
    pub xform: XRayMatrix,
    /// Pre-halved diffuse/hemi/sun scale (`* 0.5` applied at load matching `FTreeVisual::Load`).
    pub c_scale: TreeLightingColor,
    /// Pre-halved diffuse/hemi/sun bias (`* 0.5` applied at load matching `FTreeVisual::Load`).
    pub c_bias: TreeLightingColor,
}

/// Individual VB or IB slice reference (`OGF_VCONTAINER` / `OGF_ICONTAINER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OgfBufferRef {
    pub id: u32,
    pub base: u32,
    pub count: u32,
}

/// Shared level geometry container reference (`OGF_GCONTAINER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OgfGContainerRef {
    pub vb_id: u32,
    pub vb_base: u32,
    pub vb_count: u32,
    pub ib_id: u32,
    pub ib_base: u32,
    pub ib_count: u32,
}

/// Bone shape flags (`enum SBoneShape::EShapeFlags` in `src/xrEngine/bone.h`).
pub mod bone_shape_flags {
    pub const SF_NO_PICKABLE: u16 = 1 << 0;
    pub const SF_REMOVE_AFTER_BREAK: u16 = 1 << 1;
    pub const SF_NO_PHYSICS: u16 = 1 << 2;
    pub const SF_NO_FOG_COLLIDER: u16 = 1 << 3;
}

/// Oriented bounding box (`_obb<float>`, 60 bytes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgfObb {
    pub rotate: [[f32; 3]; 3],
    pub translate: [f32; 3],
    pub halfsize: [f32; 3],
}

impl Default for OgfObb {
    fn default() -> Self {
        Self {
            rotate: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translate: [0.0; 3],
            halfsize: [0.0; 3],
        }
    }
}

impl OgfObb {
    /// Convert this OBB's rotation and translation into a 4x4 affine transform matrix
    /// (`_obb::xform_get` in `src/xrCore/_obb.h:73-83`).
    #[inline]
    pub fn xform_get(&self) -> XRayMatrix {
        [
            [self.rotate[0][0], self.rotate[0][1], self.rotate[0][2], 0.0],
            [self.rotate[1][0], self.rotate[1][1], self.rotate[1][2], 0.0],
            [self.rotate[2][0], self.rotate[2][1], self.rotate[2][2], 0.0],
            [self.translate[0], self.translate[1], self.translate[2], 1.0],
        ]
    }

    /// Construct an OBB from the 4x3 linear/translation columns of `m` and half-extents.
    #[inline]
    pub fn from_matrix_halfsize(m: &XRayMatrix, halfsize: [f32; 3]) -> Self {
        Self {
            rotate: [
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ],
            translate: [m[3][0], m[3][1], m[3][2]],
            halfsize,
        }
    }

    /// Transform this OBB by `m` (`dest.transform(*this, m)` matching `_obb::transform`
    /// in `src/xrCore/_obb.h:103-112`: `destR.mul_43(M, srcR)`).
    #[inline]
    pub fn transform(&self, m: &XRayMatrix) -> Self {
        let src_r = self.xform_get();
        let dest_r = matrix_mul_43(m, &src_r);
        Self::from_matrix_halfsize(&dest_r, self.halfsize)
    }

    /// True if all half-extents are non-positive.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.halfsize[0] <= 0.0 && self.halfsize[1] <= 0.0 && self.halfsize[2] <= 0.0
    }

    /// Ray-vs-OBB separating axis intersection test matching `CDB::TestRayOBB`
    /// (`src/Layers/xrRender/cl_intersect.h:817-864`).
    ///
    /// Tests the semi-infinite ray `origin + t * dir` (`t >= 0.0`).
    pub fn test_ray(&self, origin: [f32; 3], dir: [f32; 3]) -> bool {
        #[inline]
        fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }

        let k_diff = [
            origin[0] - self.translate[0],
            origin[1] - self.translate[1],
            origin[2] - self.translate[2],
        ];

        let mut f_wdu = [0.0f32; 3];
        let mut f_awdu = [0.0f32; 3];
        for i in 0..3 {
            f_wdu[i] = dot3(dir, self.rotate[i]);
            f_awdu[i] = f_wdu[i].abs();
            let f_ddu = dot3(k_diff, self.rotate[i]);
            let f_addu = f_ddu.abs();
            if f_addu > self.halfsize[i] && f_ddu * f_wdu[i] >= 0.0 {
                return false;
            }
        }

        let k_wxd = [
            dir[1] * k_diff[2] - dir[2] * k_diff[1],
            dir[2] * k_diff[0] - dir[0] * k_diff[2],
            dir[0] * k_diff[1] - dir[1] * k_diff[0],
        ];

        let rhs0 = self.halfsize[1] * f_awdu[2] + self.halfsize[2] * f_awdu[1];
        if dot3(k_wxd, self.rotate[0]).abs() > rhs0 {
            return false;
        }
        let rhs1 = self.halfsize[0] * f_awdu[2] + self.halfsize[2] * f_awdu[0];
        if dot3(k_wxd, self.rotate[1]).abs() > rhs1 {
            return false;
        }
        let rhs2 = self.halfsize[0] * f_awdu[1] + self.halfsize[1] * f_awdu[0];
        if dot3(k_wxd, self.rotate[2]).abs() > rhs2 {
            return false;
        }

        true
    }

    /// Sphere-vs-OBB overlap test matching `CDB::TestSphereOBB`
    /// (`src/Layers/xrRender/cl_intersect.h:734-814`).
    pub fn test_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        #[inline]
        fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
            a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        }

        let k_cdiff = [
            center[0] - self.translate[0],
            center[1] - self.translate[1],
            center[2] - self.translate[2],
        ];
        let f_ax = dot3(k_cdiff, self.rotate[0]).abs();
        let f_ay = dot3(k_cdiff, self.rotate[1]).abs();
        let f_az = dot3(k_cdiff, self.rotate[2]).abs();

        let f_dx = f_ax - self.halfsize[0];
        let f_dy = f_ay - self.halfsize[1];
        let f_dz = f_az - self.halfsize[2];
        let f_r_sqr = radius * radius;

        if f_ax <= self.halfsize[0] {
            if f_ay <= self.halfsize[1] {
                if f_az <= self.halfsize[2] {
                    true
                } else {
                    f_dz <= radius
                }
            } else if f_az <= self.halfsize[2] {
                f_dy <= radius
            } else {
                f_dy * f_dy + f_dz * f_dz <= f_r_sqr
            }
        } else if f_ay <= self.halfsize[1] {
            if f_az <= self.halfsize[2] {
                f_dx <= radius
            } else {
                f_dx * f_dx + f_dz * f_dz <= f_r_sqr
            }
        } else if f_az <= self.halfsize[2] {
            f_dx * f_dx + f_dy * f_dy <= f_r_sqr
        } else {
            f_dx * f_dx + f_dy * f_dy + f_dz * f_dz <= f_r_sqr
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum BoneShapeType {
    None = 0,
    Box = 1,
    Sphere = 2,
    Cylinder = 3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneShape {
    pub shape_type: u16,
    pub flags: u16,
    pub box_obb: OgfObb,
    pub sphere_center: [f32; 3],
    pub sphere_radius: f32,
    pub cylinder_center: [f32; 3],
    pub cylinder_direction: [f32; 3],
    pub cylinder_height: f32,
    pub cylinder_radius: f32,
}

impl BoneShape {
    /// True unless `SBoneShape::sfNoPickable` is set.
    #[inline]
    pub fn is_pickable(&self) -> bool {
        (self.flags & bone_shape_flags::SF_NO_PICKABLE) == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointLimit {
    pub limit: [f32; 2],
    pub spring_factor: f32,
    pub damping_factor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneIkData {
    pub version: u16,
    pub joint_type: u32,
    pub limits: [JointLimit; 3],
    pub spring_factor: f32,
    pub damping_factor: f32,
    pub ik_flags: u32,
    pub break_force: f32,
    pub break_torque: f32,
    pub friction: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoneIkSection {
    pub game_mtl_name: String,
    pub shape: BoneShape,
    pub ik_data: BoneIkData,
    pub bind_rotation_xyz: [f32; 3],
    pub bind_translation: [f32; 3],
    pub mass: f32,
    pub center_of_mass: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct OgfBone {
    pub id: u16,
    pub name: String,
    pub parent_name: String,
    pub parent_id: Option<u16>,
    pub children: Vec<u16>,
    pub obb: OgfObb,
    pub ik: Option<BoneIkSection>,
}

/// X-Ray 4x4 row-vector affine matrix matching `Fmatrix` (`src/xrCore/_matrix.h`).
pub type XRayMatrix = [[f32; 4]; 4];

pub const XRAY_IDENTITY_MATRIX: XRayMatrix = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Compute `Fmatrix::rotateX(angle)` matching `src/xrCore/_matrix.h:348-361`.
pub fn matrix_rotate_x(angle: f32) -> XRayMatrix {
    let (sina, cosa) = angle.sin_cos();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, cosa, sina, 0.0],
        [0.0, -sina, cosa, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Compute `Fmatrix::rotateY(angle)` matching `src/xrCore/_matrix.h:363-376`.
///
/// Matches X-Ray's left-handed row-vector basis (`i = (cos, 0, -sin)`, `k = (sin, 0, cos)`),
/// which rotates in the opposite direction to `matrix_set_hpb(angle, 0, 0)`.
pub fn matrix_rotate_y(angle: f32) -> XRayMatrix {
    let (sina, cosa) = angle.sin_cos();
    [
        [cosa, 0.0, -sina, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [sina, 0.0, cosa, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Compute `Fmatrix::rotateZ(angle)` matching `src/xrCore/_matrix.h:378-391`.
pub fn matrix_rotate_z(angle: f32) -> XRayMatrix {
    let (sina, cosa) = angle.sin_cos();
    [
        [cosa, sina, 0.0, 0.0],
        [-sina, cosa, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Compute `Fmatrix::scale(x, y, z)` matching `src/xrCore/_matrix.h:334-341`.
#[inline]
pub fn matrix_scale(x: f32, y: f32, z: f32) -> XRayMatrix {
    [
        [x, 0.0, 0.0, 0.0],
        [0.0, y, 0.0, 0.0],
        [0.0, 0.0, z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Compute `(up, right)` orthonormal vectors perpendicular to `dir` matching
/// `Fvector::generate_orthonormal_basis` (`src/xrCore/_vector3d.h:653-675`).
pub fn vector_generate_orthonormal_basis(dir: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let up = if dir[0].abs() >= dir[1].abs() {
        let denom = (dir[0] * dir[0] + dir[2] * dir[2]).sqrt();
        if denom > 1e-6 {
            let inv = 1.0 / denom;
            [-dir[2] * inv, 0.0, dir[0] * inv]
        } else {
            [0.0, 0.0, 1.0]
        }
    } else {
        let denom = (dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if denom > 1e-6 {
            let inv = 1.0 / denom;
            [0.0, dir[2] * inv, -dir[1] * inv]
        } else {
            [0.0, 1.0, 0.0]
        }
    };
    let right = [
        up[1] * dir[2] - up[2] * dir[1],
        up[2] * dir[0] - up[0] * dir[2],
        up[0] * dir[1] - up[1] * dir[0],
    ];
    (up, right)
}

/// Compute `Fmatrix::setHPB(h, p, b)` matching `_matrix.h:852-876`.
pub fn matrix_set_hpb(h: f32, p: f32, b: f32) -> XRayMatrix {
    let (sh, ch) = h.sin_cos();
    let (sp, cp) = p.sin_cos();
    let (sb, cb) = b.sin_cos();
    let cc = ch * cb;
    let cs = ch * sb;
    let sc = sh * cb;
    let ss = sh * sb;
    [
        [cc - sp * ss, -cp * sb, sp * cs + sc, 0.0],
        [sp * sc + cs, cp * cb, ss - sp * cc, 0.0],
        [-cp * sh, sp, cp * ch, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Compute `Fmatrix::setXYZ(x, y, z)` (`setHPB(y, x, z)`) with translation `t` (`_matrix.h:880`).
pub fn matrix_set_xyz_translate(xyz: [f32; 3], t: [f32; 3]) -> XRayMatrix {
    let mut m = matrix_set_hpb(xyz[1], xyz[0], xyz[2]);
    m[3][0] = t[0];
    m[3][1] = t[1];
    m[3][2] = t[2];
    m
}

/// Extract the forward (`+Z`, `Fmatrix::k`) basis vector from `m`.
#[inline]
pub fn matrix_forward_dir(m: &XRayMatrix) -> [f32; 3] {
    [m[2][0], m[2][1], m[2][2]]
}

/// Compute `Fmatrix::setXYZi(x, y, z)` (`setHPB(-y, -x, -z)`) with translation `t`.
pub fn matrix_set_xyzi_translate(xyz: [f32; 3], t: [f32; 3]) -> XRayMatrix {
    let mut m = matrix_set_hpb(-xyz[1], -xyz[0], -xyz[2]);
    m[3][0] = t[0];
    m[3][1] = t[1];
    m[3][2] = t[2];
    m
}

/// Compute `Fmatrix::mul(A, B)` (full 4x4 row-vector product, `_matrix.h:116-139`).
///
/// Transforms row vector by `b` first, then by `a` (`v * (b * a)`).
pub fn matrix_mul_44(a: &XRayMatrix, b: &XRayMatrix) -> XRayMatrix {
    let mut r = XRAY_IDENTITY_MATRIX;
    for row in 0..4 {
        for col in 0..4 {
            r[row][col] = a[0][col] * b[row][0]
                + a[1][col] * b[row][1]
                + a[2][col] * b[row][2]
                + a[3][col] * b[row][3];
        }
    }
    r
}

/// Compute `Fmatrix::mul_43(A, B)` (`_matrix.h:142-165`).
pub fn matrix_mul_43(a: &XRayMatrix, b: &XRayMatrix) -> XRayMatrix {
    let mut r = XRAY_IDENTITY_MATRIX;
    for row in 0..3 {
        for col in 0..3 {
            r[row][col] =
                a[0][col] * b[row][0] + a[1][col] * b[row][1] + a[2][col] * b[row][2];
        }
        r[row][3] = 0.0;
    }
    for col in 0..3 {
        r[3][col] =
            a[0][col] * b[3][0] + a[1][col] * b[3][1] + a[2][col] * b[3][2] + a[3][col];
    }
    r[3][3] = 1.0;
    r
}

/// Compute 4x3 affine matrix inverse matching `Fmatrix::invert` (`_matrix.h:195-225`).
pub fn matrix_invert_43(a: &XRayMatrix) -> Option<XRayMatrix> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() <= 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let m00 = inv * (a[1][1] * a[2][2] - a[1][2] * a[2][1]);
    let m01 = -inv * (a[0][1] * a[2][2] - a[0][2] * a[2][1]);
    let m02 = inv * (a[0][1] * a[1][2] - a[0][2] * a[1][1]);

    let m10 = -inv * (a[1][0] * a[2][2] - a[1][2] * a[2][0]);
    let m11 = inv * (a[0][0] * a[2][2] - a[0][2] * a[2][0]);
    let m12 = -inv * (a[0][0] * a[1][2] - a[0][2] * a[1][0]);

    let m20 = inv * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let m21 = -inv * (a[0][0] * a[2][1] - a[0][1] * a[2][0]);
    let m22 = inv * (a[0][0] * a[1][1] - a[0][1] * a[1][0]);

    let tx = -(a[3][0] * m00 + a[3][1] * m10 + a[3][2] * m20);
    let ty = -(a[3][0] * m01 + a[3][1] * m11 + a[3][2] * m21);
    let tz = -(a[3][0] * m02 + a[3][1] * m12 + a[3][2] * m22);

    Some([
        [m00, m01, m02, 0.0],
        [m10, m11, m12, 0.0],
        [m20, m21, m22, 0.0],
        [tx, ty, tz, 1.0],
    ])
}

/// Transform a 3D point by an `XRayMatrix` (`Fmatrix::transform_tiny`).
pub fn matrix_transform_point(m: &XRayMatrix, p: [f32; 3]) -> [f32; 3] {
    [
        p[0] * m[0][0] + p[1] * m[1][0] + p[2] * m[2][0] + m[3][0],
        p[0] * m[0][1] + p[1] * m[1][1] + p[2] * m[2][1] + m[3][1],
        p[0] * m[0][2] + p[1] * m[1][2] + p[2] * m[2][2] + m[3][2],
    ]
}

/// Transform and perspective-divide a 3D point by a 4x4 projection/view-proj matrix
/// (`Fmatrix::transform(Fvector&, const Fvector&)` in `src/xrCore/_matrix.h:815-821`).
///
/// Returns `None` if clip `w` is near zero (`<= 1e-6` in magnitude) or non-finite.
pub fn matrix_transform_projected(m: &XRayMatrix, p: [f32; 3]) -> Option<[f32; 3]> {
    let w = p[0] * m[0][3] + p[1] * m[1][3] + p[2] * m[2][3] + m[3][3];
    if w.abs() <= 1e-6 || !w.is_finite() {
        return None;
    }
    let iw = 1.0 / w;
    Some([
        (p[0] * m[0][0] + p[1] * m[1][0] + p[2] * m[2][0] + m[3][0]) * iw,
        (p[0] * m[0][1] + p[1] * m[1][1] + p[2] * m[2][1] + m[3][1]) * iw,
        (p[0] * m[0][2] + p[1] * m[1][2] + p[2] * m[2][2] + m[3][2]) * iw,
    ])
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BonePoseMatrices {
    pub local_bind: XRayMatrix,
    pub model_bind: XRayMatrix,
    /// Inverse bind-pose matrix (`CBoneData::m2b_transform`: maps model space -> bone local).
    pub m2b_transform: XRayMatrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OgfSkeleton {
    pub bones: Vec<OgfBone>,
    pub root_bone: Option<u16>,
    pub user_data: Option<String>,
    pub lod_ref: Option<String>,
    pub motion_refs: Vec<String>,
}

impl OgfSkeleton {
    /// Resolve parent/root links matching `CKinematics::Load` and compute `local_bind`,
    /// `model_bind`, and `m2b_transform` matching `CBoneData::CalculateM2B`.
    pub fn compute_bind_pose_matrices(&self) -> Option<Vec<BonePoseMatrices>> {
        let root_id = self.root_bone?;
        let count = self.bones.len();
        let mut results = vec![
            BonePoseMatrices {
                local_bind: XRAY_IDENTITY_MATRIX,
                model_bind: XRAY_IDENTITY_MATRIX,
                m2b_transform: XRAY_IDENTITY_MATRIX,
            };
            count
        ];

        fn recurse(
            bones: &[OgfBone],
            id: u16,
            parent_model: &XRayMatrix,
            out: &mut [BonePoseMatrices],
            depth: usize,
        ) -> Option<()> {
            if depth > bones.len() {
                return None; // cycle guard
            }
            let bone = bones.get(id as usize)?;
            let local_bind = match &bone.ik {
                Some(ik) => matrix_set_xyzi_translate(ik.bind_rotation_xyz, ik.bind_translation),
                None => XRAY_IDENTITY_MATRIX,
            };
            let model_bind = matrix_mul_43(parent_model, &local_bind);
            let m2b = matrix_invert_43(&model_bind)?;
            out[id as usize] = BonePoseMatrices {
                local_bind,
                model_bind,
                m2b_transform: m2b,
            };
            for &child_id in &bone.children {
                recurse(bones, child_id, &model_bind, out, depth + 1)?;
            }
            Some(())
        }

        recurse(&self.bones, root_id, &XRAY_IDENTITY_MATRIX, &mut results, 0)?;
        Some(results)
    }

    pub fn bone_id(&self, name: &str) -> Option<u16> {
        let lower = name.to_ascii_lowercase();
        self.bones
            .iter()
            .find(|b| b.name == lower)
            .map(|b| b.id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OgfDesc {
    pub source_file: String,
    pub build_name: String,
    pub build_time: u64,
    pub create_name: String,
    pub create_time: u64,
    pub modif_name: String,
    pub modif_time: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OgfModel {
    pub header: OgfHeader,
    pub texture: Option<OgfTextureLink>,
    pub vertices: Option<OgfVertices>,
    pub indices: Vec<u16>,
    pub swi: Option<OgfSlideWindows>,
    pub swi_container: Option<u32>,
    pub gcontainer: Option<OgfGContainerRef>,
    pub vcontainer: Option<OgfBufferRef>,
    pub icontainer: Option<OgfBufferRef>,
    pub fastpath_gcontainer: Option<OgfGContainerRef>,
    pub fastpath_swi: Option<OgfSlideWindows>,
    pub lod_def: Option<OgfLodDef>,
    pub tree_def: Option<OgfTreeDef>,
    pub skeleton: Option<OgfSkeleton>,
    pub desc: Option<OgfDesc>,
    pub child_links: Vec<u32>,
    pub children: Vec<OgfModel>,
}

impl OgfModel {
    /// Parse an X-Ray OGF v4 binary stream (static, progressive, hierarchy, or skeletal).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        use ogf_chunks::*;
        let reader = ChunkReader::new(data);

        let header_bytes = reader.require_chunk(OGF_HEADER)?;
        let header = parse_header(&header_bytes)?;

        let texture = match reader.open_chunk(OGF_TEXTURE)? {
            Some(bytes) => {
                let mut cur = ByteCursor::new(&bytes);
                let texture = cur.read_zstring()?;
                let shader = cur.read_zstring()?;
                Some(OgfTextureLink { texture, shader })
            }
            None => None,
        };

        let gcontainer = match reader.open_chunk(OGF_GCONTAINER)? {
            Some(bytes) => Some(parse_gcontainer(&bytes)?),
            None => None,
        };

        let vcontainer = match reader.open_chunk(OGF_VCONTAINER)? {
            Some(bytes) => Some(parse_buffer_ref(&bytes)?),
            None => None,
        };

        let icontainer = match reader.open_chunk(OGF_ICONTAINER)? {
            Some(bytes) => Some(parse_buffer_ref(&bytes)?),
            None => None,
        };

        let swi_container = match reader.open_chunk(OGF_SWICONTAINER)? {
            Some(bytes) => {
                let mut cur = ByteCursor::new(&bytes);
                Some(cur.read_u32()?)
            }
            None => None,
        };

        let (fastpath_gcontainer, fastpath_swi) = match reader.open_chunk(OGF_FASTPATH)? {
            Some(bytes) => {
                let fp_reader = ChunkReader::new(&bytes);
                let fp_g = match fp_reader.open_chunk(OGF_GCONTAINER)? {
                    Some(gb) => Some(parse_gcontainer(&gb)?),
                    None => None,
                };
                let fp_s = match fp_reader.open_chunk(OGF_SWIDATA)? {
                    Some(sb) => Some(parse_swidata(&sb)?),
                    None => None,
                };
                (fp_g, fp_s)
            }
            None => (None, None),
        };

        let vertices = match reader.open_chunk(OGF_VERTICES)? {
            Some(bytes) => Some(parse_vertices(&bytes)?),
            None => None,
        };

        let indices = match reader.open_chunk(OGF_INDICES)? {
            Some(bytes) => parse_indices(&bytes)?,
            None => Vec::new(),
        };

        if let Some(ref verts) = vertices {
            let vcount = verts.len();
            if indices.iter().any(|&idx| (idx as usize) >= vcount) {
                return Err(FormatError::InvalidValue(
                    "OGF_INDICES references out-of-range vertex index",
                ));
            }
        }

        let swi = match reader.open_chunk(OGF_SWIDATA)? {
            Some(bytes) => Some(parse_swidata(&bytes)?),
            None => None,
        };

        let desc = match reader.open_chunk(OGF_S_DESC)? {
            Some(bytes) => Some(parse_desc(&bytes)?),
            None => None,
        };

        let child_links = match reader.open_chunk(OGF_CHILDREN_L)? {
            Some(bytes) => {
                let mut cur = ByteCursor::new(&bytes);
                let cnt = cur.read_u32()? as usize;
                let mut links = Vec::with_capacity(cnt.min(cur.remaining() / 4));
                for _ in 0..cnt {
                    links.push(cur.read_u32()?);
                }
                links
            }
            None => Vec::new(),
        };

        let children = match reader.open_chunk(OGF_CHILDREN)? {
            Some(bytes) => {
                let sub_reader = ChunkReader::new(&bytes);
                let mut out = Vec::new();
                for sub_entry in sub_reader.iter() {
                    let sub_entry = sub_entry?;
                    let decoded = sub_entry.decode()?;
                    out.push(Self::parse(&decoded)?);
                }
                out
            }
            None => Vec::new(),
        };

        let lod_def = match reader.open_chunk(OGF_LODDEF2)? {
            Some(bytes) => Some(parse_loddef2(&bytes, &header)?),
            None => None,
        };

        let tree_def = match reader.open_chunk(OGF_TREEDEF2)? {
            Some(bytes) => Some(parse_treedef2(&bytes)?),
            None => None,
        };

        let skeleton = parse_skeleton_if_present(&reader)?;

        Ok(Self {
            header,
            texture,
            vertices,
            indices,
            swi,
            swi_container,
            gcontainer,
            vcontainer,
            icontainer,
            fastpath_gcontainer,
            fastpath_swi,
            lod_def,
            tree_def,
            skeleton,
            desc,
            child_links,
            children,
        })
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Resolve active `OgfSlideWindows` from either inline `OGF_SWIDATA` or `OGF_SWICONTAINER`
    /// (`use_fastpath = true` checks `fastpath_swi`).
    pub fn resolve_swi<'a>(
        &'a self,
        external_swis: &'a [OgfSlideWindows],
        use_fastpath: bool,
    ) -> Option<&'a OgfSlideWindows> {
        if use_fastpath {
            return self.fastpath_swi.as_ref();
        }
        self.swi
            .as_ref()
            .or_else(|| self.swi_container.and_then(|id| external_swis.get(id as usize)))
    }

    /// Return LOD-sliced inline indices using `OGF_SWIDATA` (or `external_swis` via `swi_container`).
    pub fn progressive_indices(&self, lod: f32, external_swis: &[OgfSlideWindows]) -> &[u16] {
        let Some(swi) = self.resolve_swi(external_swis, false) else {
            return &self.indices;
        };
        let Some(win) = swi.select_window(lod) else {
            return &self.indices;
        };
        let start = win.offset as usize;
        let len = (win.num_tris as usize) * 3;
        start
            .checked_add(len)
            .and_then(|end| self.indices.get(start..end))
            .unwrap_or(&self.indices)
    }
}

fn parse_buffer_ref(data: &[u8]) -> Result<OgfBufferRef, FormatError> {
    let mut cur = ByteCursor::new(data);
    Ok(OgfBufferRef {
        id: cur.read_u32()?,
        base: cur.read_u32()?,
        count: cur.read_u32()?,
    })
}

fn parse_gcontainer(data: &[u8]) -> Result<OgfGContainerRef, FormatError> {
    let mut cur = ByteCursor::new(data);
    Ok(OgfGContainerRef {
        vb_id: cur.read_u32()?,
        vb_base: cur.read_u32()?,
        vb_count: cur.read_u32()?,
        ib_id: cur.read_u32()?,
        ib_base: cur.read_u32()?,
        ib_count: cur.read_u32()?,
    })
}

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize_safe3(v: [f32; 3]) -> [f32; 3] {
    let m = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if m > f32::MIN_POSITIVE {
        let inv = (1.0 / m).sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        v
    }
}

#[inline]
fn mk_normal(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    normalize_safe3(cross3(sub3(p1, p0), sub3(p2, p1)))
}

fn parse_loddef2(data: &[u8], header: &OgfHeader) -> Result<OgfLodDef, FormatError> {
    if data.len() < OGF_LODDEF2_SIZE {
        return Err(FormatError::UnexpectedEof);
    }
    let mut cur = ByteCursor::new(data);
    let mut facets = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut verts = Vec::with_capacity(4);
        for _ in 0..4 {
            let position = cur.read_f32_3()?;
            let uv = cur.read_f32_2()?;
            let rgb_hemi = cur.read_u32()?;
            let sun = cur.read_u8()?;
            let _pad = cur.read_bytes(3)?;
            verts.push(OgfLodVertex {
                position,
                uv,
                rgb_hemi,
                sun,
            });
        }
        let v: [OgfLodVertex; 4] = verts.try_into().expect("exactly 4 LOD vertices");
        let n0 = mk_normal(v[0].position, v[1].position, v[2].position);
        let n1 = mk_normal(v[1].position, v[2].position, v[3].position);
        let n2 = mk_normal(v[2].position, v[3].position, v[0].position);
        let n3 = mk_normal(v[3].position, v[0].position, v[1].position);
        let avg = [
            (n0[0] + n1[0] + n2[0] + n3[0]) * 0.25,
            (n0[1] + n1[1] + n2[1] + n3[1]) * 0.25,
            (n0[2] + n1[2] + n2[2] + n3[2]) * 0.25,
        ];
        let unit = normalize_safe3(avg);
        facets.push(OgfLodFacet {
            vertices: v,
            normal: [-unit[0], -unit[1], -unit[2]],
        });
    }

    let mut half_ext = [
        (header.bbox.max[0] - header.bbox.min[0]) * 0.5,
        (header.bbox.max[1] - header.bbox.min[1]) * 0.5,
        (header.bbox.max[2] - header.bbox.min[2]) * 0.5,
    ];
    half_ext.sort_by(f32::total_cmp);
    let r = header.bsphere.radius;
    let lod_factor = if r > 1e-6 {
        let a = half_ext[1].clamp(0.0, r);
        let sf = 4.0 * (0.5 * (r * r * (a / r).asin() + a * (r * r - a * a).max(0.0).sqrt()));
        let ss = std::f32::consts::PI * r * r;
        sf / ss
    } else {
        1.0
    };

    Ok(OgfLodDef {
        facets: facets.try_into().expect("exactly 8 LOD facets"),
        lod_factor,
    })
}

fn parse_treedef2(data: &[u8]) -> Result<OgfTreeDef, FormatError> {
    if data.len() < OGF_TREEDEF2_SIZE {
        return Err(FormatError::UnexpectedEof);
    }
    let mut cur = ByteCursor::new(data);
    let xform = [
        cur.read_f32_4()?,
        cur.read_f32_4()?,
        cur.read_f32_4()?,
        cur.read_f32_4()?,
    ];
    let c_scale = TreeLightingColor {
        rgb: [
            cur.read_f32()? * 0.5,
            cur.read_f32()? * 0.5,
            cur.read_f32()? * 0.5,
        ],
        hemi: cur.read_f32()? * 0.5,
        sun: cur.read_f32()? * 0.5,
    };
    let c_bias = TreeLightingColor {
        rgb: [
            cur.read_f32()? * 0.5,
            cur.read_f32()? * 0.5,
            cur.read_f32()? * 0.5,
        ],
        hemi: cur.read_f32()? * 0.5,
        sun: cur.read_f32()? * 0.5,
    };
    Ok(OgfTreeDef {
        xform,
        c_scale,
        c_bias,
    })
}

fn parse_header(data: &[u8]) -> Result<OgfHeader, FormatError> {
    let mut cur = ByteCursor::new(data);
    let format_version = cur.read_u8()?;
    if format_version != OGF_FORMAT_VERSION {
        return Err(FormatError::InvalidValue("unsupported OGF format version (expected 4)"));
    }
    let visual_type = OgfVisualType::try_from(cur.read_u8()?)?;
    let shader_id = cur.read_u16()?;
    let bbox = OgfBBox {
        min: cur.read_f32_3()?,
        max: cur.read_f32_3()?,
    };
    let bsphere = OgfBSphere {
        center: cur.read_f32_3()?,
        radius: cur.read_f32()?,
    };
    Ok(OgfHeader {
        format_version,
        visual_type,
        shader_id,
        bbox,
        bsphere,
    })
}

fn parse_vertices(data: &[u8]) -> Result<OgfVertices, FormatError> {
    let mut cur = ByteCursor::new(data);
    let vert_type = cur.read_u32()?;
    let count = cur.read_u32()? as usize;

    match vert_type {
        skel_vert_type::FVF_1L | 1 => {
            let need = count
                .checked_mul(60)
                .ok_or(FormatError::UnexpectedEof)?;
            let raw = cur.read_bytes(need)?;
            let mut out = Vec::with_capacity(count);
            for chunk in raw.as_chunks::<60>().0 {
                let mut vc = ByteCursor::new(chunk);
                out.push(VertBoned1W {
                    position: vc.read_f32_3()?,
                    normal: vc.read_f32_3()?,
                    tangent: vc.read_f32_3()?,
                    binormal: vc.read_f32_3()?,
                    u: vc.read_f32()?,
                    v: vc.read_f32()?,
                    matrix: vc.read_u32()?,
                });
            }
            Ok(OgfVertices::Skinned1W(out))
        }
        skel_vert_type::FVF_2L | 2 => {
            let need = count
                .checked_mul(64)
                .ok_or(FormatError::UnexpectedEof)?;
            let raw = cur.read_bytes(need)?;
            let mut out = Vec::with_capacity(count);
            for chunk in raw.as_chunks::<64>().0 {
                let mut vc = ByteCursor::new(chunk);
                out.push(VertBoned2W {
                    matrix0: vc.read_u16()?,
                    matrix1: vc.read_u16()?,
                    position: vc.read_f32_3()?,
                    normal: vc.read_f32_3()?,
                    tangent: vc.read_f32_3()?,
                    binormal: vc.read_f32_3()?,
                    weight: vc.read_f32()?,
                    u: vc.read_f32()?,
                    v: vc.read_f32()?,
                });
            }
            Ok(OgfVertices::Skinned2W(out))
        }
        skel_vert_type::FVF_3L | 3 => {
            let need = count
                .checked_mul(70)
                .ok_or(FormatError::UnexpectedEof)?;
            let raw = cur.read_bytes(need)?;
            let mut out = Vec::with_capacity(count);
            for chunk in raw.as_chunks::<70>().0 {
                let mut vc = ByteCursor::new(chunk);
                out.push(VertBoned3W {
                    matrices: [vc.read_u16()?, vc.read_u16()?, vc.read_u16()?],
                    position: vc.read_f32_3()?,
                    normal: vc.read_f32_3()?,
                    tangent: vc.read_f32_3()?,
                    binormal: vc.read_f32_3()?,
                    weights: [vc.read_f32()?, vc.read_f32()?],
                    u: vc.read_f32()?,
                    v: vc.read_f32()?,
                });
            }
            Ok(OgfVertices::Skinned3W(out))
        }
        skel_vert_type::FVF_4L | 4 => {
            let need = count
                .checked_mul(76)
                .ok_or(FormatError::UnexpectedEof)?;
            let raw = cur.read_bytes(need)?;
            let mut out = Vec::with_capacity(count);
            for chunk in raw.as_chunks::<76>().0 {
                let mut vc = ByteCursor::new(chunk);
                out.push(VertBoned4W {
                    matrices: [
                        vc.read_u16()?,
                        vc.read_u16()?,
                        vc.read_u16()?,
                        vc.read_u16()?,
                    ],
                    position: vc.read_f32_3()?,
                    normal: vc.read_f32_3()?,
                    tangent: vc.read_f32_3()?,
                    binormal: vc.read_f32_3()?,
                    weights: [vc.read_f32()?, vc.read_f32()?, vc.read_f32()?],
                    u: vc.read_f32()?,
                    v: vc.read_f32()?,
                });
            }
            Ok(OgfVertices::Skinned4W(out))
        }
        fvf => {
            let layout = StaticVertexLayout::from_fvf(fvf)?;
            let need = count
                .checked_mul(layout.stride)
                .ok_or(FormatError::UnexpectedEof)?;
            let raw = cur.read_bytes(need)?;
            let mut out = Vec::with_capacity(count);
            for chunk in raw.chunks_exact(layout.stride) {
                out.push(layout.decode_vertex(chunk)?);
            }
            Ok(OgfVertices::Static {
                layout,
                vertices: out,
                raw_bytes: raw.to_vec(),
            })
        }
    }
}

fn parse_indices(data: &[u8]) -> Result<Vec<u16>, FormatError> {
    let mut cur = ByteCursor::new(data);
    let count = cur.read_u32()? as usize;
    let need = count
        .checked_mul(2)
        .ok_or(FormatError::UnexpectedEof)?;
    let raw = cur.read_bytes(need)?;
    let mut indices = Vec::with_capacity(count);
    for pair in raw.as_chunks::<2>().0 {
        indices.push(u16::from_le_bytes(*pair));
    }
    Ok(indices)
}

fn parse_swidata(data: &[u8]) -> Result<OgfSlideWindows, FormatError> {
    let mut cur = ByteCursor::new(data);
    let reserved = [
        cur.read_u32()?,
        cur.read_u32()?,
        cur.read_u32()?,
        cur.read_u32()?,
    ];
    let count = cur.read_u32()? as usize;
    let need = count
        .checked_mul(8)
        .ok_or(FormatError::UnexpectedEof)?;
    let raw = cur.read_bytes(need)?;
    let mut windows = Vec::with_capacity(count);
    for rec in raw.as_chunks::<8>().0 {
        let mut rc = ByteCursor::new(rec);
        windows.push(SlideWindow {
            offset: rc.read_u32()?,
            num_tris: rc.read_u16()?,
            num_verts: rc.read_u16()?,
        });
    }
    Ok(OgfSlideWindows { reserved, windows })
}

fn parse_desc(data: &[u8]) -> Result<OgfDesc, FormatError> {
    // Support both 32-bit time_t (x86/SDK) and 64-bit time_t (x64 MSVC) cleanly:
    // Read strings with probe or check remaining size after each string!
    let mut cur = ByteCursor::new(data);
    let source_file = cur.read_zstring()?;
    let build_name = cur.read_zstring()?;

    // Try 64-bit time_t first, falling back to 32-bit time_t if remaining length is shorter
    let save_pos = cur.position();
    if let Ok((b_t, c_n, c_t, m_n, m_t)) = try_parse_desc_times(&mut cur, true)
        && cur.is_eof()
    {
        return Ok(OgfDesc {
            source_file,
            build_name,
            build_time: b_t,
            create_name: c_n,
            create_time: c_t,
            modif_name: m_n,
            modif_time: m_t,
        });
    }

    let mut cur32 = ByteCursor::new(&data[save_pos..]);
    let (b_t, c_n, c_t, m_n, m_t) = try_parse_desc_times(&mut cur32, false)?;
    Ok(OgfDesc {
        source_file,
        build_name,
        build_time: b_t,
        create_name: c_n,
        create_time: c_t,
        modif_name: m_n,
        modif_time: m_t,
    })
}

fn try_parse_desc_times(
    cur: &mut ByteCursor<'_>,
    use_u64: bool,
) -> Result<(u64, String, u64, String, u64), FormatError> {
    let read_t = |c: &mut ByteCursor<'_>| -> Result<u64, FormatError> {
        if use_u64 {
            c.read_u64()
        } else {
            Ok(c.read_u32()? as u64)
        }
    };
    let b_t = read_t(cur)?;
    let c_n = cur.read_zstring()?;
    let c_t = read_t(cur)?;
    let m_n = cur.read_zstring()?;
    let m_t = read_t(cur)?;
    Ok((b_t, c_n, c_t, m_n, m_t))
}

fn read_obb(cur: &mut ByteCursor<'_>) -> Result<OgfObb, FormatError> {
    let r0 = cur.read_f32_3()?;
    let r1 = cur.read_f32_3()?;
    let r2 = cur.read_f32_3()?;
    let translate = cur.read_f32_3()?;
    let halfsize = cur.read_f32_3()?;
    Ok(OgfObb {
        rotate: [r0, r1, r2],
        translate,
        halfsize,
    })
}

fn read_bone_shape(cur: &mut ByteCursor<'_>) -> Result<BoneShape, FormatError> {
    let shape_type = cur.read_u16()?;
    let flags = cur.read_u16()?;
    let box_obb = read_obb(cur)?;
    let sphere_center = cur.read_f32_3()?;
    let sphere_radius = cur.read_f32()?;
    let cylinder_center = cur.read_f32_3()?;
    let cylinder_direction = cur.read_f32_3()?;
    let cylinder_height = cur.read_f32()?;
    let cylinder_radius = cur.read_f32()?;
    Ok(BoneShape {
        shape_type,
        flags,
        box_obb,
        sphere_center,
        sphere_radius,
        cylinder_center,
        cylinder_direction,
        cylinder_height,
        cylinder_radius,
    })
}

fn parse_skeleton_if_present(reader: &ChunkReader<'_>) -> Result<Option<OgfSkeleton>, FormatError> {
    use ogf_chunks::*;
    let Some(bones_bytes) = reader.open_chunk(OGF_S_BONE_NAMES)? else {
        return Ok(None);
    };

    let mut cur = ByteCursor::new(&bones_bytes);
    let count = cur.read_u32()? as usize;
    let mut bones = Vec::with_capacity(count);
    for id in 0..count {
        let name = cur.read_zstring()?.to_ascii_lowercase();
        let parent_name = cur.read_zstring()?.to_ascii_lowercase();
        let obb = read_obb(&mut cur)?;
        bones.push(OgfBone {
            id: id as u16,
            name,
            parent_name,
            parent_id: None,
            children: Vec::new(),
            obb,
            ik: None,
        });
    }

    // Wire parent_id and children indices (`SkeletonCustom.cpp:256-277`)
    let mut root_bone = None;
    for idx in 0..bones.len() {
        let parent_name = bones[idx].parent_name.clone();
        if parent_name.is_empty() {
            if root_bone.is_some() {
                return Err(FormatError::InvalidValue("multiple root bones in OGF skeleton"));
            }
            root_bone = Some(idx as u16);
        } else {
            let parent_idx = bones
                .iter()
                .position(|b| b.name == parent_name)
                .ok_or(FormatError::InvalidValue("bone parent name not found"))?;
            bones[idx].parent_id = Some(parent_idx as u16);
            bones[parent_idx].children.push(idx as u16);
        }
    }

    if let Some(ik_bytes) = reader.open_chunk(OGF_S_IKDATA)? {
        let mut ikc = ByteCursor::new(&ik_bytes);
        for bone in &mut bones {
            let version = ikc.read_u32()? as u16;
            let game_mtl_name = ikc.read_zstring()?;
            let shape = read_bone_shape(&mut ikc)?;
            let joint_type = ikc.read_u32()?;
            let mut limits = [JointLimit {
                limit: [0.0, 0.0],
                spring_factor: 1.0,
                damping_factor: 1.0,
            }; 3];
            for lim in &mut limits {
                lim.limit = ikc.read_f32_2()?;
                lim.spring_factor = ikc.read_f32()?;
                lim.damping_factor = ikc.read_f32()?;
            }
            let spring_factor = ikc.read_f32()?;
            let damping_factor = ikc.read_f32()?;
            let ik_flags = ikc.read_u32()?;
            let break_force = ikc.read_f32()?;
            let break_torque = ikc.read_f32()?;
            let friction = if version > 0 { ikc.read_f32()? } else { 0.0 };
            let bind_rotation_xyz = ikc.read_f32_3()?;
            let bind_translation = ikc.read_f32_3()?;
            let mass = ikc.read_f32()?;
            let center_of_mass = ikc.read_f32_3()?;

            bone.ik = Some(BoneIkSection {
                game_mtl_name,
                shape,
                ik_data: BoneIkData {
                    version,
                    joint_type,
                    limits,
                    spring_factor,
                    damping_factor,
                    ik_flags,
                    break_force,
                    break_torque,
                    friction,
                },
                bind_rotation_xyz,
                bind_translation,
                mass,
                center_of_mass,
            });
        }
    }

    let user_data = match reader.open_chunk(OGF_S_USERDATA)? {
        Some(bytes) => {
            let mut uc = ByteCursor::new(&bytes);
            Some(uc.read_zstring()?)
        }
        None => None,
    };

    let lod_ref = match reader.open_chunk(OGF_S_LODS)? {
        Some(bytes) => {
            let mut lc = ByteCursor::new(&bytes);
            Some(lc.read_zstring()?)
        }
        None => None,
    };

    let mut motion_refs = Vec::new();
    if let Some(bytes) = reader.open_chunk(OGF_S_MOTION_REFS2)? {
        let mut mc = ByteCursor::new(&bytes);
        let cnt = mc.read_u32()? as usize;
        for _ in 0..cnt {
            motion_refs.push(mc.read_zstring()?);
        }
    } else if let Some(bytes) = reader.open_chunk(OGF_S_MOTION_REFS)? {
        let mut mc = ByteCursor::new(&bytes);
        let raw = mc.read_zstring()?;
        for part in raw.split(',') {
            let p = part.trim();
            if !p.is_empty() {
                motion_refs.push(p.to_string());
            }
        }
    }

    Ok(Some(OgfSkeleton {
        bones,
        root_bone,
        user_data,
        lod_ref,
        motion_refs,
    }))
}

// ============================================================================
// Tree Foliage Wind & Lighting Constants (`FTreeVisual.h/.cpp`)
// ============================================================================

/// Fixed-point vertex quantization tile and divisor for `OGF_TREEDEF2` meshes
/// (`FTreeVisual_tile = 16`, `FTreeVisual_quant = 2048` in `FTreeVisual.h:67-68`).
pub const FTREE_VISUAL_TILE: u32 = 16;
pub const FTREE_VISUAL_QUANT: u32 = 32768 / FTREE_VISUAL_TILE;

/// Console defaults for tree foliage wind sway and lighting scale/bias
/// (`ps_r__Tree_*` in `xrRender_console.cpp:132-136`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeWindConfig {
    pub w_rot: f32,
    pub w_speed: f32,
    pub w_amp: f32,
    pub wave: [f32; 3],
    pub sbc: f32,
}

impl Default for TreeWindConfig {
    fn default() -> Self {
        Self {
            w_rot: 10.0,
            w_speed: 1.0,
            w_amp: 0.005,
            wave: [0.1, 0.01, 0.11],
            sbc: 1.5,
        }
    }
}

/// One frame's computed wind direction/amplitude & vertex-wave vector (`FTreeVisual_setup` in `FTreeVisual.cpp:110-149`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeWindFrame {
    pub scale: f32,
    pub wave: [f32; 4],
    pub wind: [f32; 4],
}

impl Default for TreeWindFrame {
    fn default() -> Self {
        Self {
            scale: 1.0 / FTREE_VISUAL_QUANT as f32,
            wave: [0.0; 4],
            wind: [0.0; 4],
        }
    }
}

/// Per-instance tree draw constant block (`FTreeVisual::Render`, `FTreeVisual.cpp:161-185`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeInstanceGpuConstants {
    pub m_xform: XRayMatrix,
    pub m_xform_v: XRayMatrix,
    pub consts: [f32; 4],
    pub wave: [f32; 4],
    pub wind: [f32; 4],
    pub prev_wave: [f32; 4],
    pub prev_wind: [f32; 4],
    pub c_scale: [f32; 4],
    pub c_bias: [f32; 4],
    pub c_sun: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<TreeInstanceGpuConstants>() == 256);

/// Persistent per-frame tree wind tracker preserving `(curr, prev)` across `dwFrame` changes
/// (`FTreeVisual.cpp:151-185`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeWindState {
    initialized: bool,
    pub frame: u64,
    pub curr: TreeWindFrame,
    pub prev: TreeWindFrame,
}

impl Default for TreeWindState {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeWindState {
    pub fn new() -> Self {
        Self {
            initialized: false,
            frame: 0,
            curr: TreeWindFrame::default(),
            prev: TreeWindFrame::default(),
        }
    }

    /// Update shared per-frame tree wind/wave vectors when `frame` advances.
    ///
    /// Uses `TREE_WIND_EFFECT` (`env_tree_amplitude_intensity`, falling back to `cfg.w_amp` if `None`).
    pub fn step(
        &mut self,
        frame: u64,
        time_global_sec: f32,
        env_tree_amplitude_intensity: Option<f32>,
        cfg: &TreeWindConfig,
    ) -> (&TreeWindFrame, &TreeWindFrame) {
        if !self.initialized || self.frame != frame {
            let next_curr = compute_tree_wind_frame(
                time_global_sec,
                env_tree_amplitude_intensity.unwrap_or(cfg.w_amp),
                cfg,
            );
            if self.initialized {
                self.prev = self.curr;
            } else {
                self.prev = next_curr;
                self.initialized = true;
            }
            self.curr = next_curr;
            self.frame = frame;
        }
        (&self.curr, &self.prev)
    }

    /// Build GPU shader constants for a single [`OgfTreeDef`] tree visual (`FTreeVisual::Render`).
    pub fn instance_constants(
        &self,
        tree: &OgfTreeDef,
        view: &XRayMatrix,
        cfg: &TreeWindConfig,
    ) -> TreeInstanceGpuConstants {
        let s = cfg.sbc * 1.3333;
        let m_xform_v = matrix_mul_43(view, &tree.xform);
        TreeInstanceGpuConstants {
            m_xform: tree.xform,
            m_xform_v,
            consts: [self.curr.scale, self.curr.scale, 0.0, 0.0],
            wave: self.curr.wave,
            wind: self.curr.wind,
            prev_wave: self.prev.wave,
            prev_wind: self.prev.wind,
            c_scale: [
                s * tree.c_scale.rgb[0],
                s * tree.c_scale.rgb[1],
                s * tree.c_scale.rgb[2],
                s * tree.c_scale.hemi,
            ],
            c_bias: [
                s * tree.c_bias.rgb[0],
                s * tree.c_bias.rgb[1],
                s * tree.c_bias.rgb[2],
                s * tree.c_bias.hemi,
            ],
            c_sun: [s * tree.c_scale.sun, s * tree.c_bias.sun, 0.0, 0.0],
        }
    }
}

fn compute_tree_wind_frame(
    time_global_sec: f32,
    wind_amp: f32,
    cfg: &TreeWindConfig,
) -> TreeWindFrame {
    let w_rot = if cfg.w_rot.abs() > 1e-4 {
        cfg.w_rot
    } else {
        10.0
    };
    let t = if time_global_sec.is_finite() {
        time_global_sec
    } else {
        0.0
    };
    let tm_rot = std::f32::consts::TAU * t / w_rot;
    let sx = tm_rot.sin();
    let cz = tm_rot.cos();
    let len = (sx * sx + cz * cz).sqrt().max(1e-6);
    let amp = if wind_amp.is_finite() { wind_amp } else { 0.0 };

    TreeWindFrame {
        scale: 1.0 / FTREE_VISUAL_QUANT as f32,
        wave: [
            cfg.wave[0] / std::f32::consts::TAU,
            cfg.wave[1] / std::f32::consts::TAU,
            cfg.wave[2] / std::f32::consts::TAU,
            (t * cfg.w_speed) / std::f32::consts::TAU,
        ],
        wind: [sx / len * amp, 0.0, cz / len * amp, 0.0],
    }
}

// ============================================================================
// LOD Imposter Batching & DSGraph Texture Sorting (`r__dsgraph_render_lods.cpp`, `r__dsgraph_render.cpp`)
// ============================================================================

/// Shader element indices used by tree/LOD rendering (`enum SE_R1` in `Shader.h:168-180`).
pub mod lod_shader_elements {
    pub const SE_R1_NORMAL_LQ: u32 = 1;
    pub const SE_R1_LMODELS: u32 = 4;
}

/// Select the LOD shader element index matching `r_dsgraph_render_lods` (`r__dsgraph_render_lods.cpp:24`).
#[inline]
pub const fn lod_shader_element_id(setup_zb: bool) -> u32 {
    if setup_zb {
        lod_shader_elements::SE_R1_LMODELS
    } else {
        lod_shader_elements::SE_R1_NORMAL_LQ
    }
}

/// Input entry to LOD imposter batch packing (`R_dsgraph::_LodItem`).
#[derive(Debug, Clone, Copy)]
pub struct LodRenderInput<'a> {
    pub shader_key: u64,
    pub bsphere: OgfBSphere,
    pub ssa: f32,
    pub lod_def: &'a OgfLodDef,
}

/// Contiguous shader run inside a single LOD VB batch (`lstLODgroups` in `r__dsgraph_render_lods.cpp:48-56`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LodDrawGroup {
    pub shader_key: u64,
    pub start_imposter: usize,
    pub imposter_count: usize,
}

/// Packed dynamic VB chunk of up to `max_imposters_per_batch` LOD quads and its draw groups.
#[derive(Debug, Clone, PartialEq)]
pub struct LodBatchChunk {
    pub vertices: Vec<LodImposterHwVertex>,
    pub groups: Vec<LodDrawGroup>,
}

/// Pack sorted LOD items into vertex-buffer-sized batches of imposter quads and contiguous shader
/// draw groups, matching `R_dsgraph_structure::r_dsgraph_render_lods` (`r__dsgraph_render_lods.cpp:17-109`).
///
/// Note: Unlike the outer/inner loop double-increment quirk in C++, this advances cleanly by exact
/// chunk length so every input item is emitted when multiple VB batches are needed.
pub fn pack_lod_imposter_batches(
    items: &[LodRenderInput<'_>],
    camera_pos: [f32; 3],
    thresholds: &ScreenLodThresholds,
    max_imposters_per_batch: usize,
) -> Vec<LodBatchChunk> {
    if items.is_empty() {
        return Vec::new();
    }
    let cap = max_imposters_per_batch.max(1);
    let mut chunks = Vec::new();
    let mut offset = 0usize;

    while offset < items.len() {
        let batch_len = (items.len() - offset).min(cap);
        let mut vertices = Vec::with_capacity(batch_len * 4);
        let mut groups = Vec::new();

        let mut cur_shader = items[offset].shader_key;
        let mut cur_start = 0usize;
        let mut cur_count = 0usize;

        for (j, item) in items[offset..offset + batch_len].iter().enumerate() {
            if item.shader_key == cur_shader {
                cur_count += 1;
            } else {
                groups.push(LodDrawGroup {
                    shader_key: cur_shader,
                    start_imposter: cur_start,
                    imposter_count: cur_count,
                });
                cur_shader = item.shader_key;
                cur_start = j;
                cur_count = 1;
            }

            let quad = item.lod_def.build_imposter_quad(
                camera_pos,
                item.bsphere,
                item.ssa,
                thresholds,
            );
            vertices.extend_from_slice(&quad);
        }

        groups.push(LodDrawGroup {
            shader_key: cur_shader,
            start_imposter: cur_start,
            imposter_count: cur_count,
        });

        chunks.push(LodBatchChunk { vertices, groups });
        offset += batch_len;
    }

    chunks
}

/// Texture-bucket node for DSGraph rendering (`mapNormalTextures::TNode` / `mapMatrixTextures::TNode`).
#[derive(Debug, Clone, PartialEq)]
pub struct DsgraphTextureNode<K, V> {
    pub texture_ids: K,
    pub ssa: f32,
    pub payload: V,
}

/// Sort DSGraph texture-bucket nodes matching `sort_tlist_nrm` / `sort_tlist_mat`
/// (`r__dsgraph_render.cpp:329-421`).
///
/// - When `use_ssa` is `true` and texture tuple arity `<= 1`: sorts all nodes descending by `ssa`.
/// - When `use_ssa` is `true` and arity `> 1`: puts `ssa > r_ssa_hzb_vs_tex` first (descending by
///   `ssa`), followed by remaining nodes sorted lexicographically by `texture_ids`.
/// - When `use_ssa` is `false`: sorts all nodes lexicographically by `texture_ids`.
pub fn sort_dsgraph_texture_nodes<K, V>(
    mut nodes: Vec<DsgraphTextureNode<K, V>>,
    use_ssa: bool,
    r_ssa_hzb_vs_tex: f32,
) -> Vec<DsgraphTextureNode<K, V>>
where
    K: AsRef<[u32]>,
{
    let Some(first) = nodes.first() else {
        return nodes;
    };
    let amount = first.texture_ids.as_ref().len();

    if use_ssa {
        if amount <= 1 {
            nodes.sort_by(|a, b| b.ssa.total_cmp(&a.ssa));
            return nodes;
        }
        let mut high = Vec::new();
        let mut low = Vec::new();
        for node in nodes {
            if node.ssa > r_ssa_hzb_vs_tex {
                high.push(node);
            } else {
                low.push(node);
            }
        }
        high.sort_by(|a, b| b.ssa.total_cmp(&a.ssa));
        low.sort_by(|a, b| a.texture_ids.as_ref().cmp(b.texture_ids.as_ref()));
        high.extend(low);
        high
    } else {
        nodes.sort_by(|a, b| a.texture_ids.as_ref().cmp(b.texture_ids.as_ref()));
        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use ogf_chunks::*;

    fn write_vec3(out: &mut Vec<u8>, v: [f32; 3]) {
        for c in v {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }

    fn write_obb(out: &mut Vec<u8>, obb: &OgfObb) {
        for r in obb.rotate {
            write_vec3(out, r);
        }
        write_vec3(out, obb.translate);
        write_vec3(out, obb.halfsize);
    }

    fn build_header_chunk(visual_type: OgfVisualType, shader_id: u16) -> Vec<u8> {
        let mut b = Vec::new();
        b.push(OGF_FORMAT_VERSION);
        b.push(visual_type as u8);
        b.extend_from_slice(&shader_id.to_le_bytes());
        write_vec3(&mut b, [-1.0, 0.0, -1.0]);
        write_vec3(&mut b, [1.0, 2.0, 1.0]);
        write_vec3(&mut b, [0.0, 1.0, 0.0]);
        b.extend_from_slice(&1.5f32.to_le_bytes());
        b
    }

    #[test]
    fn test_static_progressive_ogf_parse() {
        let mut w = ChunkWriter::new();
        w.write_chunk(
            OGF_HEADER,
            &build_header_chunk(OgfVisualType::Progressive, 1),
        );
        w.write_chunk(OGF_TEXTURE, b"wpn\\wpn_ak74\0models\\model\0");

        // 3 vertices in standard X-Ray static FVF (XYZ | NORMAL | TEX1 = 32B)
        let mut verts = Vec::new();
        verts.extend_from_slice(&fvf_flags::XRAY_STATIC_FVF_STANDARD.to_le_bytes());
        verts.extend_from_slice(&3u32.to_le_bytes());
        for (p, uv) in [
            ([0.0f32, 0.0, 0.0], [0.0f32, 1.0]),
            ([1.0, 0.0, 0.0], [1.0, 1.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0]),
        ] {
            write_vec3(&mut verts, p);
            write_vec3(&mut verts, [0.0, 0.0, 1.0]);
            verts.extend_from_slice(&uv[0].to_le_bytes());
            verts.extend_from_slice(&uv[1].to_le_bytes());
        }
        w.write_chunk(OGF_VERTICES, &verts);

        let mut idxs = Vec::new();
        idxs.extend_from_slice(&3u32.to_le_bytes());
        for i in [0u16, 1, 2] {
            idxs.extend_from_slice(&i.to_le_bytes());
        }
        w.write_chunk(OGF_INDICES, &idxs);

        let mut swi = Vec::new();
        for r in [0u32; 4] {
            swi.extend_from_slice(&r.to_le_bytes());
        }
        swi.extend_from_slice(&1u32.to_le_bytes()); // count
        swi.extend_from_slice(&0u32.to_le_bytes()); // offset
        swi.extend_from_slice(&1u16.to_le_bytes()); // num_tris
        swi.extend_from_slice(&3u16.to_le_bytes()); // num_verts
        w.write_chunk(OGF_SWIDATA, &swi);

        let bytes = w.into_bytes();
        let ogf = OgfModel::parse(&bytes).expect("static progressive OGF must parse");
        assert_eq!(ogf.header.visual_type, OgfVisualType::Progressive);
        assert_eq!(
            ogf.texture,
            Some(OgfTextureLink {
                texture: "wpn\\wpn_ak74".to_string(),
                shader: "models\\model".to_string(),
            })
        );
        assert_eq!(ogf.triangle_count(), 1);
        assert_eq!(ogf.swi.as_ref().unwrap().windows.len(), 1);
        let verts_ref = ogf.vertices.as_ref().unwrap();
        assert_eq!(verts_ref.len(), 3);
        let static_verts = verts_ref.to_static_attributes();
        assert_eq!(static_verts[1].position, [1.0, 0.0, 0.0]);
        assert_eq!(static_verts[2].uv0, [0.0, 0.0]);
    }

    #[test]
    fn test_skeletal_ogf_children_bones_ik_and_m2b_transform() {
        // Build child skinned mesh with 2W vertices (FVF_2L)
        let mut child = ChunkWriter::new();
        child.write_chunk(
            OGF_HEADER,
            &build_header_chunk(OgfVisualType::SkeletonGeomdefSt, 0),
        );
        child.write_chunk(OGF_TEXTURE, b"act\\act_stalker\0models\\model\0");

        let mut v2w = Vec::new();
        v2w.extend_from_slice(&skel_vert_type::FVF_2L.to_le_bytes());
        v2w.extend_from_slice(&3u32.to_le_bytes());
        for i in 0..3u16 {
            v2w.extend_from_slice(&0u16.to_le_bytes()); // matrix0
            v2w.extend_from_slice(&1u16.to_le_bytes()); // matrix1
            write_vec3(&mut v2w, [i as f32, 1.0, 0.0]); // P
            write_vec3(&mut v2w, [0.0, 1.0, 0.0]); // N
            write_vec3(&mut v2w, [1.0, 0.0, 0.0]); // T
            write_vec3(&mut v2w, [0.0, 0.0, 1.0]); // B
            v2w.extend_from_slice(&0.25f32.to_le_bytes()); // weight -> matrix0 gets 0.75, matrix1 gets 0.25
            v2w.extend_from_slice(&0.5f32.to_le_bytes());
            v2w.extend_from_slice(&0.5f32.to_le_bytes());
        }
        child.write_chunk(OGF_VERTICES, &v2w);

        let mut cidx = Vec::new();
        cidx.extend_from_slice(&3u32.to_le_bytes());
        for i in [0u16, 1, 2] {
            cidx.extend_from_slice(&i.to_le_bytes());
        }
        child.write_chunk(OGF_INDICES, &cidx);

        // Parent OGF_CHILDREN chunk wrapping child 0
        let mut children_wrap = ChunkWriter::new();
        children_wrap.write_chunk(0, &child.into_bytes());

        // Root skeleton with 2 bones: root -> arm_upper (translated +2.0 Y, +1.0 X)
        let mut bone_names = Vec::new();
        bone_names.extend_from_slice(&2u32.to_le_bytes());
        let id_obb = OgfObb {
            rotate: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translate: [0.0, 0.0, 0.0],
            halfsize: [0.5, 0.5, 0.5],
        };
        bone_names.extend_from_slice(b"bip01_pelvis\0\0");
        write_obb(&mut bone_names, &id_obb);
        bone_names.extend_from_slice(b"bip01_spine\0bip01_pelvis\0");
        write_obb(&mut bone_names, &id_obb);

        let mut ik_data = Vec::new();
        for (idx, trans) in [[0.0f32, 1.0, 0.0], [1.0, 2.0, 0.0]].iter().enumerate() {
            ik_data.extend_from_slice(&1u32.to_le_bytes()); // vers = 1 (> 0 includes friction)
            ik_data.extend_from_slice(b"default_object\0");
            // SBoneShape: u16 type, u16 flags, Fobb (60B), Fsphere (16B), Fcylinder (32B) = 112B
            ik_data.extend_from_slice(&1u16.to_le_bytes());
            ik_data.extend_from_slice(&0u16.to_le_bytes());
            write_obb(&mut ik_data, &id_obb);
            write_vec3(&mut ik_data, [0.0, 0.0, 0.0]);
            ik_data.extend_from_slice(&0.5f32.to_le_bytes());
            write_vec3(&mut ik_data, [0.0, 0.0, 0.0]);
            write_vec3(&mut ik_data, [0.0, 1.0, 0.0]);
            ik_data.extend_from_slice(&1.0f32.to_le_bytes());
            ik_data.extend_from_slice(&0.25f32.to_le_bytes());
            // SJointIKData: u32 type, 3 * SJointLimit (4*4=16B each), 2*f32, u32, 3*f32
            ik_data.extend_from_slice(&0u32.to_le_bytes());
            for _ in 0..3 {
                ik_data.extend_from_slice(&0.0f32.to_le_bytes());
                ik_data.extend_from_slice(&0.0f32.to_le_bytes());
                ik_data.extend_from_slice(&1.0f32.to_le_bytes());
                ik_data.extend_from_slice(&1.0f32.to_le_bytes());
            }
            ik_data.extend_from_slice(&1.0f32.to_le_bytes());
            ik_data.extend_from_slice(&1.0f32.to_le_bytes());
            ik_data.extend_from_slice(&0u32.to_le_bytes());
            ik_data.extend_from_slice(&0.0f32.to_le_bytes());
            ik_data.extend_from_slice(&0.0f32.to_le_bytes());
            ik_data.extend_from_slice(&0.05f32.to_le_bytes());
            write_vec3(&mut ik_data, [0.0, 0.0, 0.0]); // bind_rotation_xyz
            write_vec3(&mut ik_data, *trans); // bind_translation
            ik_data.extend_from_slice(&((idx + 1) as f32 * 5.0).to_le_bytes()); // mass
            write_vec3(&mut ik_data, [0.0, 0.0, 0.0]); // center_of_mass
        }

        let mut motion_refs = Vec::new();
        motion_refs.extend_from_slice(&2u32.to_le_bytes());
        motion_refs.extend_from_slice(b"actors\\stalker_animation\0actors\\stalker_scripts\0");

        let mut root = ChunkWriter::new();
        root.write_chunk(
            OGF_HEADER,
            &build_header_chunk(OgfVisualType::SkeletonAnim, 0),
        );
        root.write_chunk(OGF_CHILDREN, &children_wrap.into_bytes());
        root.write_compressed_chunk(OGF_S_BONE_NAMES, &bone_names);
        root.write_compressed_chunk(OGF_S_IKDATA, &ik_data);
        root.write_chunk(OGF_S_MOTION_REFS2, &motion_refs);
        root.write_chunk(OGF_S_USERDATA, b"[custom]\nkey = value\n\0");

        let ogf_bytes = root.into_bytes();
        let model = OgfModel::parse(&ogf_bytes).expect("skeletal OGF with compressed bone chunks must parse");
        assert_eq!(model.header.visual_type, OgfVisualType::SkeletonAnim);
        assert_eq!(model.children.len(), 1);

        let unified = model.children[0]
            .vertices
            .as_ref()
            .unwrap()
            .to_unified_skinned();
        assert_eq!(unified[0].bone_indices, [0, 1, 0, 0]);
        assert_eq!(unified[0].bone_weights, [0.75, 0.25, 0.0, 0.0]);

        let skel = model.skeleton.as_ref().unwrap();
        assert_eq!(skel.bones.len(), 2);
        assert_eq!(skel.root_bone, Some(0));
        assert_eq!(skel.bone_id("bip01_spine"), Some(1));
        assert_eq!(
            skel.motion_refs,
            vec![
                "actors\\stalker_animation".to_string(),
                "actors\\stalker_scripts".to_string()
            ]
        );

        let poses = skel.compute_bind_pose_matrices().unwrap();
        // Child bone world translation in bind pose = [0+1, 1+2, 0] = [1, 3, 0]
        // Applying m2b_transform to model-space [1, 3, 0] must return local origin [0, 0, 0]!
        let child_local = matrix_transform_point(&poses[1].m2b_transform, [1.0, 3.0, 0.0]);
        assert!(child_local[0].abs() < 1e-5);
        assert!(child_local[1].abs() < 1e-5);
        assert!(child_local[2].abs() < 1e-5);
    }

    #[test]
    fn test_ogf_rejects_out_of_range_indices_and_bad_version() {
        let mut bad_ver = ChunkWriter::new();
        let mut hdr = build_header_chunk(OgfVisualType::Normal, 0);
        hdr[0] = 3; // invalid old OGF v3
        bad_ver.write_chunk(OGF_HEADER, &hdr);
        assert!(OgfModel::parse(&bad_ver.into_bytes()).is_err());

        let mut bad_idx = ChunkWriter::new();
        bad_idx.write_chunk(OGF_HEADER, &build_header_chunk(OgfVisualType::Normal, 0));
        let mut verts = Vec::new();
        verts.extend_from_slice(&fvf_flags::XRAY_STATIC_FVF_STANDARD.to_le_bytes());
        verts.extend_from_slice(&1u32.to_le_bytes());
        verts.extend_from_slice(&[0u8; 32]);
        bad_idx.write_chunk(OGF_VERTICES, &verts);
        let mut idxs = Vec::new();
        idxs.extend_from_slice(&3u32.to_le_bytes());
        for i in [0u16, 1, 99] {
            idxs.extend_from_slice(&i.to_le_bytes());
        }
        bad_idx.write_chunk(OGF_INDICES, &idxs);
        assert!(matches!(
            OgfModel::parse(&bad_idx.into_bytes()),
            Err(FormatError::InvalidValue(_))
        ));
    }

    #[test]
    fn test_fvf_multi_uv_diffuse_gcontainer_and_1w_3w_4w_skinning() {
        // D3DFVF_XYZ (12) | D3DFVF_NORMAL (12) | D3DFVF_DIFFUSE (4) | TEX2 (0x0200)
        // with TEXCOORDSIZE3 for stage 1 (`1 << 18`): 12 + 12 + 4 + 8 + 12 = 48 bytes
        let fvf = fvf_flags::D3DFVF_XYZ
            | fvf_flags::D3DFVF_NORMAL
            | fvf_flags::D3DFVF_DIFFUSE
            | 0x0200
            | (1 << 18);
        let layout = StaticVertexLayout::from_fvf(fvf).unwrap();
        assert_eq!(layout.stride, 48);
        assert_eq!(layout.texcoord_dims[0], 2);
        assert_eq!(layout.texcoord_dims[1], 3);

        let mut vtx = Vec::new();
        write_vec3(&mut vtx, [1.0, 2.0, 3.0]);
        write_vec3(&mut vtx, [0.0, 1.0, 0.0]);
        vtx.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        vtx.extend_from_slice(&0.25f32.to_le_bytes());
        vtx.extend_from_slice(&0.75f32.to_le_bytes());
        write_vec3(&mut vtx, [0.1, 0.2, 0.3]);
        let decoded = layout.decode_vertex(&vtx).unwrap();
        assert_eq!(decoded.position, [1.0, 2.0, 3.0]);
        assert_eq!(decoded.diffuse, 0xDEAD_BEEF);
        assert_eq!(decoded.uv0, [0.25, 0.75]);
        assert_eq!(decoded.uv1, [0.1, 0.2]);

        // OGF_GCONTAINER model
        let mut gw = ChunkWriter::new();
        gw.write_chunk(OGF_HEADER, &build_header_chunk(OgfVisualType::Normal, 2));
        let mut gcont = Vec::new();
        for val in [1u32, 100, 32, 2, 200, 48] {
            gcont.extend_from_slice(&val.to_le_bytes());
        }
        gw.write_chunk(OGF_GCONTAINER, &gcont);
        let gmodel = OgfModel::parse(&gw.into_bytes()).unwrap();
        let gc = gmodel.gcontainer.unwrap();
        assert_eq!((gc.vb_id, gc.vb_base, gc.vb_count, gc.ib_id, gc.ib_base, gc.ib_count), (1, 100, 32, 2, 200, 48));

        // 1W, 3W, and 4W skeletal vertex streams & unified weight complement checks
        let mut w1 = ChunkWriter::new();
        w1.write_chunk(OGF_HEADER, &build_header_chunk(OgfVisualType::SkeletonGeomdefSt, 0));
        let mut b1 = Vec::new();
        b1.extend_from_slice(&skel_vert_type::FVF_1L.to_le_bytes());
        b1.extend_from_slice(&1u32.to_le_bytes());
        write_vec3(&mut b1, [1.0, 0.0, 0.0]);
        write_vec3(&mut b1, [0.0, 1.0, 0.0]);
        write_vec3(&mut b1, [1.0, 0.0, 0.0]);
        write_vec3(&mut b1, [0.0, 0.0, 1.0]);
        b1.extend_from_slice(&0.5f32.to_le_bytes());
        b1.extend_from_slice(&0.25f32.to_le_bytes());
        b1.extend_from_slice(&7u32.to_le_bytes());
        w1.write_chunk(OGF_VERTICES, &b1);
        let m1 = OgfModel::parse(&w1.into_bytes()).unwrap();
        let u1 = m1.vertices.unwrap().to_unified_skinned();
        assert_eq!(u1[0].bone_indices, [7, 0, 0, 0]);
        assert_eq!(u1[0].bone_weights, [1.0, 0.0, 0.0, 0.0]);

        let mut w3 = ChunkWriter::new();
        w3.write_chunk(OGF_HEADER, &build_header_chunk(OgfVisualType::SkeletonGeomdefSt, 0));
        let mut b3 = Vec::new();
        b3.extend_from_slice(&skel_vert_type::FVF_3L.to_le_bytes());
        b3.extend_from_slice(&1u32.to_le_bytes());
        for m in [1u16, 3, 5] {
            b3.extend_from_slice(&m.to_le_bytes());
        }
        for _ in 0..12 {
            b3.extend_from_slice(&0.0f32.to_le_bytes());
        }
        b3.extend_from_slice(&0.5f32.to_le_bytes());
        b3.extend_from_slice(&0.3f32.to_le_bytes());
        b3.extend_from_slice(&0.0f32.to_le_bytes());
        b3.extend_from_slice(&0.0f32.to_le_bytes());
        w3.write_chunk(OGF_VERTICES, &b3);
        let m3 = OgfModel::parse(&w3.into_bytes()).unwrap();
        let u3 = m3.vertices.unwrap().to_unified_skinned();
        assert_eq!(u3[0].bone_indices, [1, 3, 5, 0]);
        assert!((u3[0].bone_weights[2] - 0.2).abs() < 1e-6);

        let mut w4 = ChunkWriter::new();
        w4.write_chunk(OGF_HEADER, &build_header_chunk(OgfVisualType::SkeletonGeomdefSt, 0));
        let mut b4 = Vec::new();
        b4.extend_from_slice(&skel_vert_type::FVF_4L.to_le_bytes());
        b4.extend_from_slice(&1u32.to_le_bytes());
        for m in [2u16, 4, 6, 8] {
            b4.extend_from_slice(&m.to_le_bytes());
        }
        for _ in 0..12 {
            b4.extend_from_slice(&0.0f32.to_le_bytes());
        }
        for wt in [0.4f32, 0.3, 0.2] {
            b4.extend_from_slice(&wt.to_le_bytes());
        }
        b4.extend_from_slice(&0.0f32.to_le_bytes());
        b4.extend_from_slice(&0.0f32.to_le_bytes());
        w4.write_chunk(OGF_VERTICES, &b4);
        let m4 = OgfModel::parse(&w4.into_bytes()).unwrap();
        let u4 = m4.vertices.unwrap().to_unified_skinned();
        assert_eq!(u4[0].bone_indices, [2, 4, 6, 8]);
        assert!((u4[0].bone_weights[3] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_ogf_loddef2_treedef2_and_progressive_swi_selection() {
        // 1. OGF_LODDEF2 (8 facets * 4 vertices * 28B = 896B)
        let mut lod_raw = Vec::new();
        for f in 0..8 {
            let ang = (f as f32) * std::f32::consts::FRAC_PI_4;
            let (s, c) = ang.sin_cos();
            // Tangent plane facing (+c, 0, +s): FLOD::Load inverts cross(p1-p0, p2-p1), so build
            // quad where cross(up, right) points inward (-c, 0, -s) -> inverted normal = (+c, 0, +s)
            let right = [s, 0.0, -c];
            let up = [0.0f32, 1.0, 0.0];
            let corners = [
                [-right[0] - up[0], -up[1], -right[2] - up[2]],
                [-right[0] + up[0], up[1], -right[2] + up[2]],
                [right[0] + up[0], up[1], right[2] + up[2]],
                [right[0] - up[0], -up[1], right[2] - up[2]],
            ];
            for (k, pos) in corners.iter().enumerate() {
                write_vec3(&mut lod_raw, *pos);
                lod_raw.extend_from_slice(&(k as f32 * 0.25).to_le_bytes());
                lod_raw.extend_from_slice(&0.5f32.to_le_bytes());
                lod_raw.extend_from_slice(&0xFF80_8080u32.to_le_bytes());
                lod_raw.push(200u8);
                lod_raw.extend_from_slice(&[0u8; 3]); // C struct align tail pad
            }
        }
        assert_eq!(lod_raw.len(), OGF_LODDEF2_SIZE);

        let mut lod_w = ChunkWriter::new();
        lod_w.write_chunk(OGF_HEADER, &build_header_chunk(OgfVisualType::Lod, 0));
        lod_w.write_chunk(OGF_LODDEF2, &lod_raw);
        let lod_model = OgfModel::parse(&lod_w.into_bytes()).unwrap();
        let lod_def = lod_model.lod_def.expect("OGF_LODDEF2 must parse");
        assert!((lod_def.facets[0].normal[0] - 1.0).abs() < 1e-5);
        assert!(lod_def.lod_factor > 0.0 && lod_def.lod_factor.is_finite());

        // Camera along -X looking at sphere center at (0, 1, 0) -> Ldir = (+1, 0, 0) picks facet 0
        let sample = lod_def.sample_imposter_pair([-20.0, 1.0, 0.0], lod_model.header.bsphere);
        assert_eq!(sample.best_facet_index, 0);
        assert!(sample.blend_factor >= 0.0 && sample.blend_factor <= 1.0);
        assert!((sample.shift[0] + 0.75).abs() < 1e-4);

        // 2. OGF_TREEDEF2 (64B XRayMatrix + 20B scale + 20B bias = 104B)
        let mut tree_raw = Vec::new();
        for r in XRAY_IDENTITY_MATRIX {
            for f in r {
                tree_raw.extend_from_slice(&f.to_le_bytes());
            }
        }
        // c_scale (rgb=[2, 2, 2], hemi=1.0, sun=0.8) -> loaded halved (*0.5)
        for f in [2.0f32, 2.0, 2.0, 1.0, 0.8] {
            tree_raw.extend_from_slice(&f.to_le_bytes());
        }
        // c_bias (rgb=[0.2, 0.4, 0.6], hemi=0.1, sun=0.05) -> loaded halved (*0.5)
        for f in [0.2f32, 0.4, 0.6, 0.1, 0.05] {
            tree_raw.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(tree_raw.len(), OGF_TREEDEF2_SIZE);

        let mut tree_w = ChunkWriter::new();
        tree_w.write_chunk(OGF_HEADER, &build_header_chunk(OgfVisualType::TreeSt, 1));
        tree_w.write_chunk(OGF_TREEDEF2, &tree_raw);
        let tree_model = OgfModel::parse(&tree_w.into_bytes()).unwrap();
        let td = tree_model.tree_def.expect("OGF_TREEDEF2 must parse");
        assert_eq!(td.xform, XRAY_IDENTITY_MATRIX);
        assert!((td.c_scale.rgb[0] - 1.0).abs() < 1e-6);
        assert!((td.c_scale.sun - 0.4).abs() < 1e-6);
        assert!((td.c_bias.rgb[1] - 0.2).abs() < 1e-6);

        // 3. Progressive SWI window selection
        let swis = OgfSlideWindows {
            reserved: [0; 4],
            windows: vec![
                SlideWindow {
                    offset: 0,
                    num_tris: 4,
                    num_verts: 6,
                },
                SlideWindow {
                    offset: 12,
                    num_tris: 2,
                    num_verts: 4,
                },
                SlideWindow {
                    offset: 18,
                    num_tris: 1,
                    num_verts: 3,
                },
            ],
        };
        assert_eq!(swis.select_window(1.0).unwrap().num_tris, 4);
        assert_eq!(swis.select_window(0.5).unwrap().num_tris, 2);
        assert_eq!(swis.select_window(0.0).unwrap().num_tris, 1);
    }

    #[test]
    fn test_matrix_rotate_y_matches_xray_fmatrix() {
        let ry = matrix_rotate_y(std::f32::consts::FRAC_PI_2);
        let vx = matrix_transform_point(&ry, [1.0, 2.0, 0.0]);
        assert!((vx[0] - 0.0).abs() < 1e-6);
        assert!((vx[1] - 2.0).abs() < 1e-6);
        assert!((vx[2] + 1.0).abs() < 1e-6);

        let vz = matrix_transform_point(&ry, [0.0, -1.0, 1.0]);
        assert!((vz[0] - 1.0).abs() < 1e-6);
        assert!((vz[1] + 1.0).abs() < 1e-6);
        assert!((vz[2] - 0.0).abs() < 1e-6);

        let proj = crate::camera::matrix_build_projection(
            std::f32::consts::FRAC_PI_2,
            1.0,
            1.0,
            100.0,
        );
        let ndc = matrix_transform_projected(&proj, [2.0, -2.0, 4.0]).unwrap();
        assert!((ndc[0] - 0.5).abs() < 1e-5);
        assert!((ndc[1] + 0.5).abs() < 1e-5);
        assert!(ndc[2] > 0.0 && ndc[2] < 1.0);
        assert!(matrix_transform_projected(&proj, [0.0, 0.0, 0.0]).is_none());

        let rx = matrix_rotate_x(std::f32::consts::FRAC_PI_2);
        let vy = matrix_transform_point(&rx, [0.0, 1.0, 0.0]);
        assert!((vy[1] - 0.0).abs() < 1e-6);
        assert!((vy[2] - 1.0).abs() < 1e-6);

        let rz = matrix_rotate_z(std::f32::consts::FRAC_PI_2);
        let vx_z = matrix_transform_point(&rz, [1.0, 0.0, 0.0]);
        assert!((vx_z[0] - 0.0).abs() < 1e-6);
        assert!((vx_z[1] - 1.0).abs() < 1e-6);

        let ms = matrix_scale(2.0, 3.0, 4.0);
        assert_eq!(matrix_transform_point(&ms, [1.0, 2.0, -1.0]), [2.0, 6.0, -4.0]);

        let (up, right) = vector_generate_orthonormal_basis([0.0, 1.0, 0.0]);
        let dot_ju = up[0] * 0.0 + up[1] * 1.0 + up[2] * 0.0;
        let dot_jr = right[0] * 0.0 + right[1] * 1.0 + right[2] * 0.0;
        let dot_ur = up[0] * right[0] + up[1] * right[1] + up[2] * right[2];
        assert!(dot_ju.abs() < 1e-6 && dot_jr.abs() < 1e-6 && dot_ur.abs() < 1e-6);

        let view = crate::camera::matrix_build_camera(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        let full = matrix_mul_44(&proj, &view);
        let origin_ndc = matrix_transform_projected(&full, [0.0, 0.0, 0.0]).unwrap();
        assert!(origin_ndc[0].abs() < 1e-6 && origin_ndc[1].abs() < 1e-6);
        assert!(origin_ndc[2] > 0.0 && origin_ndc[2] < 1.0);
    }

    #[test]
    fn test_ogf_obb_transform_ray_sphere_and_bone_shape_flags() {
        let base_obb = OgfObb {
            rotate: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translate: [1.0, 0.0, 0.0],
            halfsize: [0.5, 1.0, 0.25],
        };
        let mut xform = matrix_rotate_z(std::f32::consts::FRAC_PI_2);
        xform[3] = [0.0, 3.0, 2.0, 1.0];
        let world_obb = base_obb.transform(&xform);

        // Local +X -> +Y after rotZ(+pi/2), so translate [1,0,0] becomes [0,4,2]
        assert!((world_obb.translate[0] - 0.0).abs() < 1e-5);
        assert!((world_obb.translate[1] - 4.0).abs() < 1e-5);
        assert!((world_obb.translate[2] - 2.0).abs() < 1e-5);
        assert_eq!(world_obb.halfsize, [0.5, 1.0, 0.25]);

        // Rotated half-extents around [0,4,2]: +/-1.0 along world X, +/-0.5 along world Y, +/-0.25 along world Z.
        // Ray toward center hits; ray pointing away misses; off-corner ray misses.
        assert!(world_obb.test_ray([0.0, 4.0, 0.0], [0.0, 0.0, 1.0]));
        assert!(world_obb.test_ray([0.8, 4.2, 0.0], [0.0, 0.0, 1.0]));
        assert!(!world_obb.test_ray([0.0, 4.0, 0.0], [0.0, 0.0, -1.0]));
        assert!(!world_obb.test_ray([1.2, 4.0, 0.0], [0.0, 0.0, 1.0]));

        // Sphere touching face/edge/corner vs outside
        assert!(world_obb.test_sphere([0.0, 4.0, 2.0], 0.1)); // inside
        assert!(world_obb.test_sphere([1.4, 4.0, 2.0], 0.5)); // dX = 0.4 <= 0.5
        assert!(!world_obb.test_sphere([1.6, 4.0, 2.0], 0.5)); // dX = 0.6 > 0.5
        assert!(world_obb.test_sphere([1.2, 4.7, 2.0], 0.3)); // dX=0.2, dY=0.2 -> sqrt(0.08)=0.283 <= 0.3
        assert!(!world_obb.test_sphere([1.3, 4.8, 2.0], 0.3)); // dX=0.3, dY=0.3 -> sqrt(0.18)=0.424 > 0.3

        let mut bs = BoneShape {
            shape_type: BoneShapeType::Box as u16,
            flags: 0,
            box_obb: base_obb,
            sphere_center: [0.0; 3],
            sphere_radius: 0.0,
            cylinder_center: [0.0; 3],
            cylinder_direction: [0.0, 1.0, 0.0],
            cylinder_height: 0.0,
            cylinder_radius: 0.0,
        };
        assert!(bs.is_pickable());
        bs.flags |= bone_shape_flags::SF_NO_PICKABLE;
        assert!(!bs.is_pickable());
    }

    #[test]
    fn test_tree_wind_state_and_instance_constants() {
        let cfg = TreeWindConfig::default();
        let mut wind_state = TreeWindState::new();
        let (c0, p0) = wind_state.step(1, 2.5, Some(0.4), &cfg);
        assert_eq!(c0, p0);
        assert!((c0.scale - 1.0 / 2048.0).abs() < 1e-7);
        // tm_rot = 2*pi * 2.5 / 10 = pi/2 => sin=1, cos=0 => wind.x = 0.4, wind.z = 0
        assert!((c0.wind[0] - 0.4).abs() < 1e-5);
        assert!(c0.wind[2].abs() < 1e-5);
        assert!((c0.wave[3] - 2.5 / std::f32::consts::TAU).abs() < 1e-6);

        // Same frame id keeps previous snapshot untouched; advancing shifts curr -> prev
        let prev_saved = *c0;
        let (c1_same, p1_same) = wind_state.step(1, 99.0, Some(1.0), &cfg);
        assert_eq!(*c1_same, prev_saved);
        assert_eq!(*p1_same, prev_saved);

        let (c2, p2) = wind_state.step(2, 5.0, Some(0.4), &cfg);
        assert_eq!(*p2, prev_saved);
        assert!((c2.wind[2] + 0.4).abs() < 1e-5);

        let tree = OgfTreeDef {
            xform: matrix_set_xyz_translate([0.0; 3], [10.0, 2.0, -5.0]),
            c_scale: TreeLightingColor {
                rgb: [1.0, 0.8, 0.6],
                hemi: 0.5,
                sun: 0.4,
            },
            c_bias: TreeLightingColor {
                rgb: [0.1, 0.2, 0.3],
                hemi: 0.05,
                sun: 0.02,
            },
        };
        let view = matrix_set_xyz_translate([0.0; 3], [0.0, 0.0, 3.0]);
        let gpu = wind_state.instance_constants(&tree, &view, &cfg);
        let s = 1.5 * 1.3333;
        assert!((gpu.c_scale[0] - s).abs() < 1e-5);
        assert!((gpu.c_bias[3] - s * 0.05).abs() < 1e-6);
        assert!((gpu.c_sun[0] - s * 0.4).abs() < 1e-5);
        assert!((gpu.c_sun[1] - s * 0.02).abs() < 1e-6);
        assert_eq!(gpu.m_xform_v[3], [10.0, 2.0, -2.0, 1.0]);
    }

    #[test]
    fn screen_lod_thresholds_and_imposter_quad_builder() {
        let cfg = ScreenLodConfig::default();
        let t = compute_screen_lod_thresholds(1920, 1080, 90.0, &cfg);
        let expected_g = 1920.0 * 1080.0 * (1e-7 + 0.75);
        assert!((t.g_f_screen - expected_g).abs() < 1e-1);
        assert!(t.r_ssa_glod_start > t.r_ssa_glod_end);
        assert!(t.r_ssa_lod_a > t.r_ssa_lod_b);
        assert!(t.r_ssa_lod_b > t.r_ssa_discard);
        assert!((t.r_dtex_range - 75.0 * t.g_f_screen / (1024.0 * 768.0)).abs() < 1e-4);

        assert_eq!(t.imposter_alpha(t.r_ssa_lod_b * 0.5), 255);
        assert_eq!(t.imposter_alpha(t.r_ssa_lod_a * 2.0), 0);
        let mid_alpha = t.imposter_alpha(0.5 * (t.r_ssa_lod_a + t.r_ssa_lod_b));
        assert!((126..=128).contains(&mid_alpha));

        let facets: [OgfLodFacet; 8] = std::array::from_fn(|s| {
            let angle = (s as f32) * std::f32::consts::FRAC_PI_4;
            let normal = [angle.cos(), 0.0, angle.sin()];
            let vertices = std::array::from_fn(|v| OgfLodVertex {
                position: [v as f32, s as f32, 0.0],
                uv: [v as f32 * 0.25, s as f32 * 0.125],
                rgb_hemi: 0x1020_3000 + (s as u32) * 16 + v as u32,
                sun: (10 + s * 10 + v) as u8,
            });
            OgfLodFacet { vertices, normal }
        });
        let lod = OgfLodDef {
            facets,
            lod_factor: 1.0,
        };
        let bsphere = OgfBSphere {
            center: [0.0, 0.0, 10.0],
            radius: 2.0,
        };
        let cam_pos = [0.0, 0.0, 0.0];
        let sample = lod.sample_imposter_pair(cam_pos, bsphere);
        assert_eq!(sample.shift, [0.0, 0.0, -1.0]);

        let quad = lod.build_imposter_quad(cam_pos, bsphere, t.r_ssa_lod_b, &t);
        assert_eq!(std::mem::size_of::<LodImposterHwVertex>(), 76);
        let fa = &facets[sample.best_facet_index];
        let fb = &facets[sample.next_facet_index];
        let expected_uf = (sample.blend_factor * 255.5).floor().clamp(0.0, 255.0) as u8;
        for (vit, &id) in LOD_IMPOSTER_QUAD_ORDER.iter().enumerate() {
            assert_eq!(
                quad[vit].p0,
                [
                    fb.vertices[id].position[0],
                    fb.vertices[id].position[1],
                    fb.vertices[id].position[2] - 1.0,
                ]
            );
            assert_eq!(
                quad[vit].p1,
                [
                    fa.vertices[id].position[0],
                    fa.vertices[id].position[1],
                    fa.vertices[id].position[2] - 1.0,
                ]
            );
            assert_eq!(quad[vit].n0, fb.normal);
            assert_eq!(quad[vit].n1, fa.normal);
            assert_eq!(quad[vit].t0, fb.vertices[id].uv);
            assert_eq!(quad[vit].t1, fa.vertices[id].uv);
            assert_eq!(quad[vit].rgbh0, fb.vertices[id].rgb_hemi);
            assert_eq!(quad[vit].rgbh1, fa.vertices[id].rgb_hemi);
            let expected_sun_af = ((expected_uf as u32) << 24)
                | ((fb.vertices[id].sun as u32) << 16)
                | ((fa.vertices[id].sun as u32) << 8)
                | 255;
            assert_eq!(quad[vit].sun_af, expected_sun_af);
        }
    }

    #[test]
    fn visual_shader_selection_optimization_and_dsgraph_routing() {
        let thresholds =
            compute_screen_lod_thresholds(1920, 1080, 75.0, &ScreenLodConfig::default());

        assert!(thresholds.should_discard(thresholds.r_ssa_discard));
        assert!(!thresholds.should_discard(thresholds.r_ssa_discard * 2.0));
        assert!((thresholds.calc_lod(thresholds.r_ssa_glod_start) - 1.0).abs() < 1e-6);
        assert_eq!(thresholds.calc_lod(thresholds.r_ssa_glod_end), 0.0);
        assert_eq!(thresholds.hardware_lod_factor(thresholds.r_ssa_glod_start), 7.0);
        assert_eq!(thresholds.hardware_lod_factor(thresholds.r_ssa_glod_end), 1.0);

        let (ssa, dist_sq) = calc_ssa([0.0, 0.0, 0.0], [0.0, 0.0, 10.0], 2.0);
        assert!((dist_sq - 100.0).abs() < 1e-3);
        assert!((ssa - 0.02).abs() < 1e-5);

        assert_eq!(
            select_dynamic_shader_element(RenderPhase::Smap, 100.0, 2.0, 50.0),
            deffer_shader_elements::SE_R2_SHADOW
        );
        assert_eq!(
            select_dynamic_shader_element(RenderPhase::Normal, 100.0, 2.0, 50.0),
            deffer_shader_elements::SE_R2_NORMAL_HQ
        );
        assert_eq!(
            select_dynamic_shader_element(RenderPhase::Normal, 10000.0, 2.0, 50.0),
            deffer_shader_elements::SE_R2_NORMAL_LQ
        );
        assert_eq!(
            select_static_shader_element(RenderPhase::Normal, true, 70.0 * 70.0, 1.0, 50.0, 8.0),
            deffer_shader_elements::SE_R2_NORMAL_HQ
        );
        assert_eq!(
            select_static_shader_element(RenderPhase::Normal, true, 120.0 * 120.0, 1.0, 50.0, 8.0),
            deffer_shader_elements::SE_R2_NORMAL_LQ
        );
        assert_eq!(
            select_static_shader_element(RenderPhase::Normal, true, 260.0 * 260.0, 1.0, 50.0, 8.0),
            deffer_shader_elements::SE_R2_TERRAIN_VERY_LOW
        );

        let opt = VisualOptConfig::default();
        let small_sphere = OgfBSphere {
            center: [0.0, 0.0, 120.0],
            radius: 1.0, // vol = 4.19 < 25
        };
        assert!(!is_valuable_to_render_static(
            OgfVisualType::Normal,
            small_sphere,
            [0.0, 0.0, 0.0],
            67.0,
            false,
            &opt
        ));
        assert!(is_valuable_to_render_static(
            OgfVisualType::TreeSt,
            small_sphere,
            [0.0, 0.0, 0.0],
            67.0,
            false,
            &opt
        ));

        let dyn_xf = matrix_set_xyz_translate([0.0; 3], [0.0, 0.0, 120.0]);
        assert!(!is_valuable_to_render_dynamic(
            OgfBSphere {
                center: [0.0, 0.0, 0.0],
                radius: 0.5, // vol = 0.52 < 1.0; dist = 120 > 40
            },
            Some(&dyn_xf),
            [0.0, 0.0, 0.0],
            67.0,
            false,
            &opt
        ));

        let scope_flags = VisualRoutingFlags {
            is_hud: true,
            is_cam_attached: true,
            scope_lense: 2,
            is_emissive: true,
            is_distort: true,
            ..VisualRoutingFlags::default()
        };
        let r_scope = route_dynamic_visual(
            RenderPhase::Normal,
            [0.0, 0.0, 1.0],
            1.0,
            [0.0, 0.0, 0.0],
            &thresholds,
            &scope_flags,
            &[true, true],
            true,
            true,
        )
        .unwrap();
        assert_eq!(r_scope.primary, Some(VisualPrimaryBucket::ScopeCamAttachedOpaque));
        assert_eq!(r_scope.distort_bucket, Some(VisualPrimaryBucket::HudDistort));

        let water_flags = VisualRoutingFlags {
            is_water: true,
            is_landscape: true,
            is_distort: true,
            ..VisualRoutingFlags::default()
        };
        let r_static = route_static_visual(
            RenderPhase::Normal,
            OgfBSphere {
                center: [0.0, 0.0, 20.0],
                radius: 5.0,
            },
            [0.0, 0.0, 0.0],
            &thresholds,
            &water_flags,
            &[true, true],
            true,
            true,
            true,
            8.0,
        )
        .unwrap();
        assert_eq!(r_static.primary, Some(VisualPrimaryBucket::Water));
        assert_eq!(r_static.distort_bucket, Some(VisualPrimaryBucket::Distort));

        // LOD batch packing across multiple chunks and shader runs
        assert_eq!(lod_shader_element_id(true), lod_shader_elements::SE_R1_LMODELS);
        assert_eq!(lod_shader_element_id(false), lod_shader_elements::SE_R1_NORMAL_LQ);

        let facets: [OgfLodFacet; 8] = std::array::from_fn(|i| {
            let a = (i as f32) * std::f32::consts::FRAC_PI_4;
            let n = [a.cos(), 0.0, a.sin()];
            OgfLodFacet {
                vertices: [
                    OgfLodVertex { position: [-1.0, -1.0, 0.0], uv: [0.0, 1.0], rgb_hemi: 0x102030, sun: 12 },
                    OgfLodVertex { position: [-1.0,  1.0, 0.0], uv: [0.0, 0.0], rgb_hemi: 0x102030, sun: 12 },
                    OgfLodVertex { position: [ 1.0,  1.0, 0.0], uv: [1.0, 0.0], rgb_hemi: 0x102030, sun: 12 },
                    OgfLodVertex { position: [ 1.0, -1.0, 0.0], uv: [1.0, 1.0], rgb_hemi: 0x102030, sun: 12 },
                ],
                normal: n,
            }
        });
        let lod_def = OgfLodDef { facets, lod_factor: 1.0 };
        let thresholds_lod = compute_screen_lod_thresholds(1920, 1080, 75.0, &ScreenLodConfig::default());
        let inputs = [
            LodRenderInput {
                shader_key: 10,
                bsphere: OgfBSphere { center: [0.0, 0.0, 10.0], radius: 2.0 },
                ssa: 30.0,
                lod_def: &lod_def,
            },
            LodRenderInput {
                shader_key: 10,
                bsphere: OgfBSphere { center: [5.0, 0.0, 10.0], radius: 2.0 },
                ssa: 32.0,
                lod_def: &lod_def,
            },
            LodRenderInput {
                shader_key: 20,
                bsphere: OgfBSphere { center: [10.0, 0.0, 10.0], radius: 2.0 },
                ssa: 35.0,
                lod_def: &lod_def,
            },
        ];
        let batches = pack_lod_imposter_batches(&inputs, [0.0, 0.0, 0.0], &thresholds_lod, 2);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].vertices.len(), 8);
        assert_eq!(batches[0].groups.len(), 1);
        assert_eq!(batches[0].groups[0].imposter_count, 2);
        assert_eq!(batches[1].vertices.len(), 4);
        assert_eq!(batches[1].groups[0].shader_key, 20);

        // DSGraph texture list sorting (SSA split vs lex-only)
        let tex_nodes = vec![
            DsgraphTextureNode { texture_ids: vec![5, 2], ssa: 1.0, payload: 0 },
            DsgraphTextureNode { texture_ids: vec![2, 9], ssa: 50.0, payload: 1 },
            DsgraphTextureNode { texture_ids: vec![1, 4], ssa: 2.0, payload: 2 },
            DsgraphTextureNode { texture_ids: vec![8, 1], ssa: 100.0, payload: 3 },
        ];
        let sorted_ssa = sort_dsgraph_texture_nodes(tex_nodes.clone(), true, 10.0);
        let order_ssa: Vec<usize> = sorted_ssa.iter().map(|n| n.payload).collect();
        assert_eq!(order_ssa, vec![3, 1, 2, 0]);
        let sorted_lex = sort_dsgraph_texture_nodes(tex_nodes, false, 10.0);
        let order_lex: Vec<usize> = sorted_lex.iter().map(|n| n.payload).collect();
        assert_eq!(order_lex, vec![2, 1, 0, 3]);
    }
}
