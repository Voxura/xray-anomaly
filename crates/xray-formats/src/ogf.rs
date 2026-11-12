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

/// Oriented bounding box (`_obb<float>`, 60 bytes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OgfObb {
    pub rotate: [[f32; 3]; 3],
    pub translate: [f32; 3],
    pub halfsize: [f32; 3],
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
}
