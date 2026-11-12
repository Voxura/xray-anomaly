//! Pure Rust X-Ray Skeletal Skinning, Bone Palette & Animation Blending
//!
//! Replaces legacy C++ `CKinematics::CLBone` / `xrSkin1W..4W` CPU skinning paths
//! (`SkeletonRigid.cpp`, `SkeletonCustom.cpp`, `SkeletonX.cpp`, `xrSkin2W.cpp`) with
//! deterministic pure-Rust `BonePalette` construction and `1W`/`2W`/`3W`/`4W` skinning.

use crate::cform::ray_triangle_test;
use crate::chunk::FormatError;
use crate::ogf::{
    matrix_invert_43, matrix_mul_43, matrix_transform_point, BonePoseMatrices, OgfBBox, OgfModel,
    OgfObb, OgfSkeleton, OgfTextureLink, OgfVertices, OgfVisualType, StaticVertex, XRayMatrix,
    XRAY_IDENTITY_MATRIX,
};
use crate::omf::OmfMotions;

/// Transposed 4x3 shader bone matrix (`3 x float4` rows) matching `SkeletonX.cpp:136-138`:
/// - `rows[0] = [m[0][0], m[1][0], m[2][0], m[3][0]]`
/// - `rows[1] = [m[0][1], m[1][1], m[2][1], m[3][1]]`
/// - `rows[2] = [m[0][2], m[1][2], m[2][2], m[3][2]]`
///
/// Designed for direct structured-buffer / constant-buffer upload to HLSL shaders using
/// `dot(float4(pos, 1.0), bone.rows[axis])`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneMatrix4x3 {
    pub rows: [[f32; 4]; 3],
}

const _: () = assert!(std::mem::size_of::<BoneMatrix4x3>() == 48);

impl BoneMatrix4x3 {
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    #[inline]
    pub fn from_matrix(m: &XRayMatrix) -> Self {
        Self {
            rows: [
                [m[0][0], m[1][0], m[2][0], m[3][0]],
                [m[0][1], m[1][1], m[2][1], m[3][1]],
                [m[0][2], m[1][2], m[2][2], m[3][2]],
            ],
        }
    }

    #[inline]
    pub fn to_matrix(self) -> XRayMatrix {
        let [r0, r1, r2] = self.rows;
        [
            [r0[0], r1[0], r2[0], 0.0],
            [r0[1], r1[1], r2[1], 0.0],
            [r0[2], r1[2], r2[2], 0.0],
            [r0[3], r1[3], r2[3], 1.0],
        ]
    }

    #[inline]
    pub fn transform_point(self, p: [f32; 3]) -> [f32; 3] {
        let [r0, r1, r2] = self.rows;
        [
            p[0] * r0[0] + p[1] * r0[1] + p[2] * r0[2] + r0[3],
            p[0] * r1[0] + p[1] * r1[1] + p[2] * r1[2] + r1[3],
            p[0] * r2[0] + p[1] * r2[1] + p[2] * r2[2] + r2[3],
        ]
    }

    #[inline]
    pub fn transform_dir(self, d: [f32; 3]) -> [f32; 3] {
        let [r0, r1, r2] = self.rows;
        [
            d[0] * r0[0] + d[1] * r0[1] + d[2] * r0[2],
            d[0] * r1[0] + d[1] * r1[1] + d[2] * r1[2],
            d[0] * r2[0] + d[1] * r2[1] + d[2] * r2[2],
        ]
    }
}

/// Evaluated skeletal bone palette for one frame / pose (`CKinematics` bone instances).
#[derive(Debug, Clone, PartialEq)]
pub struct BonePalette {
    /// Model-space bone world transforms (`CBoneInstance::mTransform`), one per skeleton bone.
    pub model_transforms: Vec<XRayMatrix>,
    /// Skinning matrices (`mTransform * m2b_transform`, i.e. `CBoneInstance::mRenderTransform`).
    /// Equal to identity across all bones when the skeleton is in its bind pose.
    pub render_transforms: Vec<XRayMatrix>,
}

impl BonePalette {
    #[inline]
    pub fn len(&self) -> usize {
        self.render_transforms.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.render_transforms.is_empty()
    }

    /// Construct bind-pose palette (`render_transforms` all identity, `model_transforms` from IK bind pose).
    pub fn bind_pose(skeleton: &OgfSkeleton) -> Result<Self, FormatError> {
        let bind_poses = skeleton
            .compute_bind_pose_matrices()
            .ok_or(FormatError::InvalidValue(
                "failed computing OGF bind pose matrices",
            ))?;
        let model_transforms = bind_poses.iter().map(|b| b.model_bind).collect();
        let render_transforms = vec![XRAY_IDENTITY_MATRIX; bind_poses.len()];
        Ok(Self {
            model_transforms,
            render_transforms,
        })
    }

    /// Combine supplied model-space bone transforms (`mTransform`) with the skeleton's `m2b_transform`
    /// inverse bind-pose matrices (`CBoneData::CalculateM2B`).
    pub fn from_model_transforms(
        skeleton: &OgfSkeleton,
        model_transforms: &[XRayMatrix],
    ) -> Result<Self, FormatError> {
        let bind_poses = skeleton
            .compute_bind_pose_matrices()
            .ok_or(FormatError::InvalidValue(
                "failed computing OGF bind pose matrices",
            ))?;
        Self::from_model_transforms_with_bind(&bind_poses, model_transforms)
    }

    /// Fast variant accepting precomputed `BonePoseMatrices` (avoids recomputing `m2b_transform` per frame).
    pub fn from_model_transforms_with_bind(
        bind_poses: &[BonePoseMatrices],
        model_transforms: &[XRayMatrix],
    ) -> Result<Self, FormatError> {
        if model_transforms.len() != bind_poses.len() {
            return Err(FormatError::InvalidValue(
                "model_transforms count does not match skeleton bone count",
            ));
        }

        let render_transforms = model_transforms
            .iter()
            .zip(bind_poses)
            .map(|(world, bind)| matrix_mul_43(world, &bind.m2b_transform))
            .collect();

        Ok(Self {
            model_transforms: model_transforms.to_vec(),
            render_transforms,
        })
    }

    /// Evaluate single OMF animation clip at `time_sec` and build ready-to-skin bone palette.
    pub fn from_omf_clip(
        skeleton: &OgfSkeleton,
        motions: &OmfMotions,
        clip_index: usize,
        time_sec: f32,
    ) -> Result<Self, FormatError> {
        let model_transforms = motions.evaluate_skeleton_pose(skeleton, clip_index, time_sec)?;
        Self::from_model_transforms(skeleton, &model_transforms)
    }

    /// Evaluate SLERP/LERP blend between two OMF clips and build ready-to-skin bone palette.
    pub fn from_omf_blend(
        skeleton: &OgfSkeleton,
        motions: &OmfMotions,
        clip_a: usize,
        time_a: f32,
        clip_b: usize,
        time_b: f32,
        weight_b: f32,
    ) -> Result<Self, FormatError> {
        let model_transforms =
            motions.evaluate_blended_pose(skeleton, clip_a, time_a, clip_b, time_b, weight_b)?;
        Self::from_model_transforms(skeleton, &model_transforms)
    }

    /// Export skinning matrices as packed 4x3 shader constant rows (`SkeletonX.cpp` layout).
    pub fn to_gpu_4x3_matrices(&self) -> Vec<BoneMatrix4x3> {
        self.render_transforms
            .iter()
            .map(BoneMatrix4x3::from_matrix)
            .collect()
    }

    /// Export skinning matrices as 4x4 `XRayMatrix` buffers.
    pub fn to_gpu_4x4_matrices(&self) -> Vec<XRayMatrix> {
        self.render_transforms.clone()
    }
}

/// 44-byte CPU-skinned static vertex matching the DX12 `XRayVertex` wire ABI
/// (`position[3]`, `normal[3]`, `uv[2]`, `tangent[3]`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 3],
}

const _: () = assert!(std::mem::size_of::<SkinnedVertex>() == 44);

impl SkinnedVertex {
    pub fn to_static_vertex(self) -> StaticVertex {
        StaticVertex {
            position: self.position,
            normal: self.normal,
            diffuse: 0xFFFF_FFFF,
            specular: 0,
            uv0: self.uv,
            uv1: [0.0, 0.0],
        }
    }
}

/// Skinned child or standalone mesh surface extracted from an OGF hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedMeshPart {
    pub child_index: Option<usize>,
    pub texture: Option<OgfTextureLink>,
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u16>,
}

/// Transform a 3D direction/normal vector by the 3x3 linear upper-left block of `XRayMatrix`
/// (`Fmatrix::transform_dir`).
#[inline]
pub fn matrix_transform_dir(m: &XRayMatrix, v: [f32; 3]) -> [f32; 3] {
    [
        v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0],
        v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1],
        v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2],
    ]
}

#[inline]
pub fn normalize_vec3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq <= 1e-12 {
        return [0.0, 1.0, 0.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

#[inline]
fn add_weighted_vec3(acc: &mut [f32; 3], v: [f32; 3], weight: f32) {
    acc[0] += v[0] * weight;
    acc[1] += v[1] * weight;
    acc[2] += v[2] * weight;
}

/// CPU-skin an OGF vertex stream (`Static` or `Skinned1W..4W`) with a `BonePalette`
/// using exact X-Ray weight formulas (`xrSkin1W..4W`).
pub fn skin_vertices(
    vertices: &OgfVertices,
    palette: &BonePalette,
) -> Result<Vec<SkinnedVertex>, FormatError> {
    let mats = &palette.render_transforms;

    let get_mat = |bone_id: usize| -> Result<&XRayMatrix, FormatError> {
        mats.get(bone_id)
            .ok_or(FormatError::InvalidValue("vertex references bone index outside BonePalette"))
    };

    match vertices {
        OgfVertices::Static {
            vertices: static_verts,
            ..
        } => Ok(static_verts
            .iter()
            .map(|v| SkinnedVertex {
                position: v.position,
                normal: v.normal,
                uv: v.uv0,
                tangent: [1.0, 0.0, 0.0],
            })
            .collect()),
        OgfVertices::Skinned1W(verts) => {
            let mut out = Vec::with_capacity(verts.len());
            for v in verts {
                let m = get_mat(v.matrix as usize)?;
                out.push(SkinnedVertex {
                    position: matrix_transform_point(m, v.position),
                    normal: normalize_vec3(matrix_transform_dir(m, v.normal)),
                    uv: [v.u, v.v],
                    tangent: normalize_vec3(matrix_transform_dir(m, v.tangent)),
                });
            }
            Ok(out)
        }
        OgfVertices::Skinned2W(verts) => {
            let mut out = Vec::with_capacity(verts.len());
            for v in verts {
                let m0 = get_mat(v.matrix0 as usize)?;
                let m1 = get_mat(v.matrix1 as usize)?;
                let w1 = v.weight;
                let w0 = 1.0 - w1;

                let p0 = matrix_transform_point(m0, v.position);
                let p1 = matrix_transform_point(m1, v.position);
                let n0 = matrix_transform_dir(m0, v.normal);
                let n1 = matrix_transform_dir(m1, v.normal);
                let t0 = matrix_transform_dir(m0, v.tangent);
                let t1 = matrix_transform_dir(m1, v.tangent);

                let position = [
                    p0[0] * w0 + p1[0] * w1,
                    p0[1] * w0 + p1[1] * w1,
                    p0[2] * w0 + p1[2] * w1,
                ];
                let normal = normalize_vec3([
                    n0[0] * w0 + n1[0] * w1,
                    n0[1] * w0 + n1[1] * w1,
                    n0[2] * w0 + n1[2] * w1,
                ]);
                let tangent = normalize_vec3([
                    t0[0] * w0 + t1[0] * w1,
                    t0[1] * w0 + t1[1] * w1,
                    t0[2] * w0 + t1[2] * w1,
                ]);

                out.push(SkinnedVertex {
                    position,
                    normal,
                    uv: [v.u, v.v],
                    tangent,
                });
            }
            Ok(out)
        }
        OgfVertices::Skinned3W(verts) => {
            let mut out = Vec::with_capacity(verts.len());
            for v in verts {
                let w2 = 1.0 - v.weights[0] - v.weights[1];
                let weights = [v.weights[0], v.weights[1], w2];
                let mut pos = [0.0f32; 3];
                let mut norm = [0.0f32; 3];
                let mut tang = [0.0f32; 3];

                for (&bone, &w) in v.matrices.iter().zip(&weights) {
                    let m = get_mat(bone as usize)?;
                    add_weighted_vec3(&mut pos, matrix_transform_point(m, v.position), w);
                    add_weighted_vec3(&mut norm, matrix_transform_dir(m, v.normal), w);
                    add_weighted_vec3(&mut tang, matrix_transform_dir(m, v.tangent), w);
                }

                out.push(SkinnedVertex {
                    position: pos,
                    normal: normalize_vec3(norm),
                    uv: [v.u, v.v],
                    tangent: normalize_vec3(tang),
                });
            }
            Ok(out)
        }
        OgfVertices::Skinned4W(verts) => {
            let mut out = Vec::with_capacity(verts.len());
            for v in verts {
                let w3 = 1.0 - v.weights[0] - v.weights[1] - v.weights[2];
                let weights = [v.weights[0], v.weights[1], v.weights[2], w3];
                let mut pos = [0.0f32; 3];
                let mut norm = [0.0f32; 3];
                let mut tang = [0.0f32; 3];

                for (&bone, &w) in v.matrices.iter().zip(&weights) {
                    let m = get_mat(bone as usize)?;
                    add_weighted_vec3(&mut pos, matrix_transform_point(m, v.position), w);
                    add_weighted_vec3(&mut norm, matrix_transform_dir(m, v.normal), w);
                    add_weighted_vec3(&mut tang, matrix_transform_dir(m, v.tangent), w);
                }

                out.push(SkinnedVertex {
                    position: pos,
                    normal: normalize_vec3(norm),
                    uv: [v.u, v.v],
                    tangent: normalize_vec3(tang),
                });
            }
            Ok(out)
        }
    }
}

/// Recursively skin an OGF visual (root and any child meshes) into `SkinnedMeshPart` surfaces.
pub fn skin_ogf_hierarchy(
    root: &OgfModel,
    palette: &BonePalette,
) -> Result<Vec<SkinnedMeshPart>, FormatError> {
    fn collect(
        model: &OgfModel,
        child_index: Option<usize>,
        palette: &BonePalette,
        out: &mut Vec<SkinnedMeshPart>,
    ) -> Result<(), FormatError> {
        if let Some(ref verts) = model.vertices {
            out.push(SkinnedMeshPart {
                child_index,
                texture: model.texture.clone(),
                vertices: skin_vertices(verts, palette)?,
                indices: model.indices.clone(),
            });
        }
        for (idx, child) in model.children.iter().enumerate() {
            collect(child, Some(idx), palette, out)?;
        }
        Ok(())
    }

    let mut out = Vec::new();
    collect(root, None, palette, &mut out)?;
    Ok(out)
}

/// Compute tight AABB over a slice of skinned vertices.
pub fn compute_skinned_aabb(vertices: &[SkinnedVertex]) -> Option<OgfBBox> {
    let first = vertices.first()?;
    let mut min = first.position;
    let mut max = first.position;
    for v in &vertices[1..] {
        for axis in 0..3 {
            min[axis] = min[axis].min(v.position[axis]);
            max[axis] = max[axis].max(v.position[axis]);
        }
    }
    Some(OgfBBox { min, max })
}

#[inline]
fn tri_mknormal_zero(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    let e0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
    let n = [
        e0[1] * e1[2] - e0[2] * e1[1],
        e0[2] * e1[0] - e0[0] * e1[2],
        e0[0] * e1[1] - e0[1] * e1[0],
    ];
    let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if len_sq <= 1e-12 {
        return [0.0; 3];
    }
    let inv = 1.0 / len_sq.sqrt();
    [n[0] * inv, n[1] * inv, n[2] * inv]
}

/// Per-bone visibility bitmask (`CKinematics::LL_GetBoneVisible` / `VisMask`).
///
/// Supports arbitrary bone counts (including skeletons with more than 64 bones).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BoneMask {
    bone_count: usize,
    bits: Vec<u64>,
}

impl BoneMask {
    #[inline]
    fn words_for(bone_count: usize) -> usize {
        bone_count.div_ceil(64)
    }

    #[inline]
    fn mask_tail(bits: &mut [u64], bone_count: usize) {
        let rem = bone_count % 64;
        if rem != 0 && let Some(last) = bits.last_mut() {
            *last &= (1u64 << rem) - 1;
        }
    }

    /// Mask with every bone `0..bone_count` marked visible.
    pub fn all(bone_count: usize) -> Self {
        let mut bits = vec![u64::MAX; Self::words_for(bone_count)];
        Self::mask_tail(&mut bits, bone_count);
        Self { bone_count, bits }
    }

    /// Mask with all bones hidden.
    pub fn empty(bone_count: usize) -> Self {
        Self {
            bone_count,
            bits: vec![0u64; Self::words_for(bone_count)],
        }
    }

    /// Construct from a legacy `u64` X-Ray visibility mask (`bones <= 64` fill word 0).
    pub fn from_u64(mask: u64, bone_count: usize) -> Self {
        let words = Self::words_for(bone_count);
        let mut bits = vec![0u64; words];
        if let Some(first) = bits.first_mut() {
            *first = mask;
        }
        Self::mask_tail(&mut bits, bone_count);
        Self { bone_count, bits }
    }

    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bone_count
    }

    #[inline]
    pub fn is_visible(&self, bone_id: u16) -> bool {
        let idx = bone_id as usize;
        if idx >= self.bone_count {
            return false;
        }
        let word = idx / 64;
        let bit = idx % 64;
        ((self.bits[word] >> bit) & 1) != 0
    }

    #[inline]
    pub fn set_visible(&mut self, bone_id: u16, visible: bool) {
        let idx = bone_id as usize;
        if idx >= self.bone_count {
            return;
        }
        let word = idx / 64;
        let bit = idx % 64;
        if visible {
            self.bits[word] |= 1u64 << bit;
        } else {
            self.bits[word] &= !(1u64 << bit);
        }
    }

    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.bits.first().copied().unwrap_or(0)
    }
}

/// Evaluate exact `0`/`1`/`2`/`3`/`4`-link position skinning using an explicit `link_count`
/// (avoids C++ sentinel ambiguity when adjacent bone IDs repeat inside 3W/4W vertices).
pub fn skin_position_with_links(
    position: [f32; 3],
    link_count: u8,
    bone_ids: [u16; 4],
    weights: [f32; 3],
    render_transforms: &[XRayMatrix],
) -> Result<[f32; 3], FormatError> {
    let get_mat = |bone_id: u16| -> Result<&XRayMatrix, FormatError> {
        render_transforms
            .get(bone_id as usize)
            .ok_or(FormatError::InvalidValue(
                "wallmark vertex references bone index outside BonePalette",
            ))
    };

    match link_count {
        0 => Ok(position),
        1 => Ok(matrix_transform_point(get_mat(bone_ids[0])?, position)),
        2 => {
            let p0 = matrix_transform_point(get_mat(bone_ids[0])?, position);
            let p1 = matrix_transform_point(get_mat(bone_ids[1])?, position);
            let w1 = weights[0];
            let w0 = 1.0 - w1;
            Ok([
                p0[0] * w0 + p1[0] * w1,
                p0[1] * w0 + p1[1] * w1,
                p0[2] * w0 + p1[2] * w1,
            ])
        }
        3 => {
            let p0 = matrix_transform_point(get_mat(bone_ids[0])?, position);
            let p1 = matrix_transform_point(get_mat(bone_ids[1])?, position);
            let p2 = matrix_transform_point(get_mat(bone_ids[2])?, position);
            let w0 = weights[0];
            let w1 = weights[1];
            let w2 = 1.0 - w0 - w1;
            Ok([
                p0[0] * w0 + p1[0] * w1 + p2[0] * w2,
                p0[1] * w0 + p1[1] * w1 + p2[1] * w2,
                p0[2] * w0 + p1[2] * w1 + p2[2] * w2,
            ])
        }
        4 => {
            let p0 = matrix_transform_point(get_mat(bone_ids[0])?, position);
            let p1 = matrix_transform_point(get_mat(bone_ids[1])?, position);
            let p2 = matrix_transform_point(get_mat(bone_ids[2])?, position);
            let p3 = matrix_transform_point(get_mat(bone_ids[3])?, position);
            let w0 = weights[0];
            let w1 = weights[1];
            let w2 = weights[2];
            let w3 = 1.0 - w0 - w1 - w2;
            Ok([
                p0[0] * w0 + p1[0] * w1 + p2[0] * w2 + p3[0] * w3,
                p0[1] * w0 + p1[1] * w1 + p2[1] * w2 + p3[1] * w3,
                p0[2] * w0 + p1[2] * w1 + p2[2] * w2 + p3[2] * w3,
            ])
        }
        _ => Err(FormatError::InvalidValue(
            "invalid skeletal skinning link count",
        )),
    }
}

/// Evaluate `CKinematics::RenderWallmark` sentinel bone-equality skinning (`SkeletonCustom.cpp:868-924`).
#[inline]
pub fn skin_position_with_wm_sentinel(
    position: [f32; 3],
    bone_ids: [u16; 4],
    weights: [f32; 3],
    render_transforms: &[XRayMatrix],
) -> Result<[f32; 3], FormatError> {
    let links = if bone_ids[0] == bone_ids[1] {
        1
    } else if bone_ids[1] == bone_ids[2] {
        2
    } else if bone_ids[2] == bone_ids[3] {
        3
    } else {
        4
    };
    skin_position_with_links(position, links, bone_ids, weights, render_transforms)
}

/// Result of evaluating a single vertex's posed position and raw skinning influences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinnedVertexInfluence {
    pub skinned_pos: [f32; 3],
    pub bind_pos: [f32; 3],
    pub link_count: u8,
    pub sentinel_bones: [u16; 4],
    pub weights: [f32; 3],
}

/// Skin a single vertex position from `OgfVertices` and return its posed position plus bind-pose skinning influences.
pub fn skin_vertex_pos_and_influences(
    vertices: &OgfVertices,
    vertex_idx: usize,
    render_transforms: &[XRayMatrix],
) -> Result<SkinnedVertexInfluence, FormatError> {
    match vertices {
        OgfVertices::Static { vertices, .. } => {
            let v = vertices
                .get(vertex_idx)
                .ok_or(FormatError::InvalidValue("static vertex index out of bounds"))?;
            Ok(SkinnedVertexInfluence {
                skinned_pos: v.position,
                bind_pos: v.position,
                link_count: 0,
                sentinel_bones: [0; 4],
                weights: [0.0; 3],
            })
        }
        OgfVertices::Skinned1W(verts) => {
            let v = verts
                .get(vertex_idx)
                .ok_or(FormatError::InvalidValue("1W vertex index out of bounds"))?;
            let b = v.matrix as u16;
            let bone_ids = [b, b, b, b];
            let weights = [0.0; 3];
            let skinned =
                skin_position_with_links(v.position, 1, bone_ids, weights, render_transforms)?;
            Ok(SkinnedVertexInfluence {
                skinned_pos: skinned,
                bind_pos: v.position,
                link_count: 1,
                sentinel_bones: bone_ids,
                weights,
            })
        }
        OgfVertices::Skinned2W(verts) => {
            let v = verts
                .get(vertex_idx)
                .ok_or(FormatError::InvalidValue("2W vertex index out of bounds"))?;
            let bone_ids = [v.matrix0, v.matrix1, v.matrix1, v.matrix1];
            let weights = [v.weight, 0.0, 0.0];
            let skinned =
                skin_position_with_links(v.position, 2, bone_ids, weights, render_transforms)?;
            Ok(SkinnedVertexInfluence {
                skinned_pos: skinned,
                bind_pos: v.position,
                link_count: 2,
                sentinel_bones: bone_ids,
                weights,
            })
        }
        OgfVertices::Skinned3W(verts) => {
            let v = verts
                .get(vertex_idx)
                .ok_or(FormatError::InvalidValue("3W vertex index out of bounds"))?;
            let bone_ids = [v.matrices[0], v.matrices[1], v.matrices[2], v.matrices[2]];
            let weights = [v.weights[0], v.weights[1], 0.0];
            let skinned =
                skin_position_with_links(v.position, 3, bone_ids, weights, render_transforms)?;
            Ok(SkinnedVertexInfluence {
                skinned_pos: skinned,
                bind_pos: v.position,
                link_count: 3,
                sentinel_bones: bone_ids,
                weights,
            })
        }
        OgfVertices::Skinned4W(verts) => {
            let v = verts
                .get(vertex_idx)
                .ok_or(FormatError::InvalidValue("4W vertex index out of bounds"))?;
            let bone_ids = v.matrices;
            let weights = v.weights;
            let skinned =
                skin_position_with_links(v.position, 4, bone_ids, weights, render_transforms)?;
            Ok(SkinnedVertexInfluence {
                skinned_pos: skinned,
                bind_pos: v.position,
                link_count: 4,
                sentinel_bones: bone_ids,
                weights,
            })
        }
    }
}

/// Borrowed mesh surface inside an OGF hierarchy participating in skeletal picking / wallmarks.
#[derive(Debug, Clone)]
pub struct SkeletonLeafInfo<'a> {
    pub child_index: Option<usize>,
    pub model: &'a OgfModel,
    pub vertices: &'a OgfVertices,
    /// Active triangle index slice (`swi.windows[0]` max LOD for `SkeletonGeomdefPm` / `Progressive`).
    pub active_indices: &'a [u16],
    pub tri_count: usize,
}

/// Per-bone face index mapping (`CBoneData::child_faces` + `CSkeletonX::_CollectBoneFaces`).
#[derive(Debug, Clone)]
pub struct SkeletonCollisionIndex<'a> {
    pub skeleton: &'a OgfSkeleton,
    pub leaves: Vec<SkeletonLeafInfo<'a>>,
    /// Indexed as `bone_child_faces[bone_id][leaf_idx] -> sorted unique face indices`.
    pub bone_child_faces: Vec<Vec<Vec<u32>>>,
}

impl<'a> SkeletonCollisionIndex<'a> {
    /// Collect all active mesh leaves and build per-bone face lists (`CSkeletonX_ST`/`PM::AfterLoad`).
    pub fn from_ogf(root: &'a OgfModel) -> Result<Self, FormatError> {
        let skeleton = root
            .skeleton
            .as_ref()
            .ok_or(FormatError::MissingChunk(crate::ogf::ogf_chunks::OGF_S_BONE_NAMES))?;
        let bone_count = skeleton.bones.len();

        let mut leaves = Vec::new();
        fn gather_leaves<'m>(
            node: &'m OgfModel,
            child_index: Option<usize>,
            out: &mut Vec<SkeletonLeafInfo<'m>>,
        ) {
            if let Some(ref verts) = node.vertices
                && !node.indices.is_empty()
            {
                let active_indices = if matches!(
                    node.header.visual_type,
                    OgfVisualType::SkeletonGeomdefPm | OgfVisualType::Progressive
                ) && let Some(swi) = &node.swi
                    && let Some(max_lod) = swi.windows.first()
                {
                    let start = max_lod.offset as usize;
                    let count = (max_lod.num_tris as usize) * 3;
                    let end = start.saturating_add(count).min(node.indices.len());
                    if start <= end {
                        let slice = &node.indices[start..end];
                        let aligned = slice.len() - (slice.len() % 3);
                        &slice[..aligned]
                    } else {
                        &[]
                    }
                } else {
                    let aligned = node.indices.len() - (node.indices.len() % 3);
                    &node.indices[..aligned]
                };

                let tri_count = active_indices.len() / 3;
                if tri_count > 0 {
                    out.push(SkeletonLeafInfo {
                        child_index,
                        model: node,
                        vertices: verts,
                        active_indices,
                        tri_count,
                    });
                }
            }
            for (idx, child) in node.children.iter().enumerate() {
                gather_leaves(child, Some(idx), out);
            }
        }
        gather_leaves(root, None, &mut leaves);

        let mut bone_child_faces = vec![vec![Vec::new(); leaves.len()]; bone_count];

        for (leaf_idx, leaf) in leaves.iter().enumerate() {
            let vcount = leaf.vertices.len();
            for tri_idx in 0..leaf.tri_count {
                let face_u32 = tri_idx as u32;
                let base = tri_idx * 3;
                for k in 0..3 {
                    let vi = leaf.active_indices[base + k] as usize;
                    if vi >= vcount {
                        return Err(FormatError::InvalidValue(
                            "skeletal index references out-of-bounds vertex",
                        ));
                    }
                    match leaf.vertices {
                        OgfVertices::Static { .. } => {
                            if bone_count > 0 {
                                bone_child_faces[0][leaf_idx].push(face_u32);
                            }
                        }
                        OgfVertices::Skinned1W(verts) => {
                            let b = verts[vi].matrix as usize;
                            let slots = bone_child_faces.get_mut(b).ok_or(
                                FormatError::InvalidValue(
                                    "1W vertex references bone index outside OgfSkeleton",
                                ),
                            )?;
                            slots[leaf_idx].push(face_u32);
                        }
                        OgfVertices::Skinned2W(verts) => {
                            for b in [verts[vi].matrix0 as usize, verts[vi].matrix1 as usize] {
                                let slots = bone_child_faces.get_mut(b).ok_or(
                                    FormatError::InvalidValue(
                                        "2W vertex references bone index outside OgfSkeleton",
                                    ),
                                )?;
                                slots[leaf_idx].push(face_u32);
                            }
                        }
                        OgfVertices::Skinned3W(verts) => {
                            for &bm in &verts[vi].matrices {
                                let slots = bone_child_faces.get_mut(bm as usize).ok_or(
                                    FormatError::InvalidValue(
                                        "3W vertex references bone index outside OgfSkeleton",
                                    ),
                                )?;
                                slots[leaf_idx].push(face_u32);
                            }
                        }
                        OgfVertices::Skinned4W(verts) => {
                            for &bm in &verts[vi].matrices {
                                let slots = bone_child_faces.get_mut(bm as usize).ok_or(
                                    FormatError::InvalidValue(
                                        "4W vertex references bone index outside OgfSkeleton",
                                    ),
                                )?;
                                slots[leaf_idx].push(face_u32);
                            }
                        }
                    }
                }
            }

            for bone_slots in &mut bone_child_faces {
                let faces = &mut bone_slots[leaf_idx];
                faces.sort_unstable();
                faces.dedup();
            }
        }

        Ok(Self {
            skeleton,
            leaves,
            bone_child_faces,
        })
    }

    /// True if `bone_id` exists and does not carry `SBoneShape::sfNoPickable`.
    #[inline]
    pub fn is_bone_pickable(&self, bone_id: u16) -> bool {
        self.skeleton
            .bones
            .get(bone_id as usize)
            .is_some_and(|bone| bone.ik.as_ref().map(|ik| ik.shape.is_pickable()).unwrap_or(true))
    }

    /// Build animated model-space OBBs (`BD.obb.transform(mTransform)`) for all bones.
    pub fn build_cache_obbs(&self, palette: &BonePalette) -> Vec<OgfObb> {
        self.skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(idx, bone)| {
                if let Some(m) = palette.model_transforms.get(idx) {
                    bone.obb.transform(m)
                } else {
                    bone.obb
                }
            })
            .collect()
    }

    /// Pick the nearest triangle owned by `bone_id` in model space (`CSkeletonX::PickBone`).
    pub fn pick_single_bone_model_space(
        &self,
        palette: &BonePalette,
        bone_id: u16,
        max_dist: f32,
        origin_model: [f32; 3],
        dir_model: [f32; 3],
    ) -> Option<BonePickHit> {
        let bone_slots = self.bone_child_faces.get(bone_id as usize)?;
        let mut best_dist = max_dist;
        let mut best_hit: Option<BonePickHit> = None;

        for (leaf_idx, leaf) in self.leaves.iter().enumerate() {
            let faces = &bone_slots[leaf_idx];
            if faces.is_empty() {
                continue;
            }

            for &face_u32 in faces {
                let face_idx = face_u32 as usize;
                let base = face_idx * 3;
                let mut tri = [[0.0f32; 3]; 3];
                let mut ok = true;
                for (k, slot) in tri.iter_mut().enumerate() {
                    let vi = leaf.active_indices[base + k] as usize;
                    match skin_vertex_pos_and_influences(
                        leaf.vertices,
                        vi,
                        &palette.render_transforms,
                    ) {
                        Ok(inf) => *slot = inf.skinned_pos,
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }

                if let Some((t, u, v)) =
                    ray_triangle_test(origin_model, dir_model, tri[0], tri[1], tri[2], true)
                    && t >= 0.0
                    && t < best_dist
                {
                    let normal = tri_mknormal_zero(tri[0], tri[1], tri[2]);
                    best_dist = t;
                    best_hit = Some(BonePickHit {
                        bone_id,
                        leaf_index: leaf_idx,
                        face_index: face_idx,
                        dist: t,
                        barycentric_uv: [u, v],
                        normal,
                        tri,
                    });
                }
            }
        }

        best_hit
    }

    /// Prune visible pickable bones by animated OBB ray test and return the nearest triangle hit
    /// together with the per-bone animated `cache_obbs` (`CKinematics::AddWallmark` lines 728-753).
    pub fn pick_nearest_model_space_with_obbs(
        &self,
        palette: &BonePalette,
        visible_bones: &BoneMask,
        max_dist: f32,
        origin_model: [f32; 3],
        dir_model: [f32; 3],
    ) -> (Option<BonePickHit>, Vec<OgfObb>) {
        let mut cache_obbs = vec![OgfObb::default(); self.skeleton.bones.len()];
        let mut best_dist = max_dist;
        let mut best_hit: Option<BonePickHit> = None;

        for (k, bone) in self.skeleton.bones.iter().enumerate() {
            let bone_id = k as u16;
            if !visible_bones.is_visible(bone_id) || !self.is_bone_pickable(bone_id) {
                continue;
            }
            let Some(m) = palette.model_transforms.get(k) else {
                continue;
            };
            let obb = bone.obb.transform(m);
            cache_obbs[k] = obb;

            if obb.test_ray(origin_model, dir_model)
                && let Some(hit) = self.pick_single_bone_model_space(
                    palette,
                    bone_id,
                    best_dist,
                    origin_model,
                    dir_model,
                )
                && hit.dist < best_dist
            {
                best_dist = hit.dist;
                best_hit = Some(hit);
            }
        }

        (best_hit, cache_obbs)
    }
}

/// Ray-triangle pick hit against a skeleton (`IKinematics::pick_result`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BonePickHit {
    pub bone_id: u16,
    pub leaf_index: usize,
    pub face_index: usize,
    pub dist: f32,
    pub barycentric_uv: [f32; 2],
    pub normal: [f32; 3],
    pub tri: [[f32; 3]; 3],
}

/// Transform a world-space ray into model space and test a specific bone (`CKinematics::PickBone`).
pub fn pick_ogf_single_bone(
    index: &SkeletonCollisionIndex<'_>,
    palette: &BonePalette,
    parent_xform: &XRayMatrix,
    bone_id: u16,
    max_dist: f32,
    origin_world: [f32; 3],
    dir_world: [f32; 3],
) -> Option<BonePickHit> {
    let inv_parent = matrix_invert_43(parent_xform)?;
    let s = matrix_transform_point(&inv_parent, origin_world);
    let d = matrix_transform_dir(&inv_parent, dir_world);

    let mut hit = index.pick_single_bone_model_space(palette, bone_id, max_dist, s, d)?;
    hit.tri = [
        matrix_transform_point(parent_xform, hit.tri[0]),
        matrix_transform_point(parent_xform, hit.tri[1]),
        matrix_transform_point(parent_xform, hit.tri[2]),
    ];
    hit.normal = tri_mknormal_zero(hit.tri[0], hit.tri[1], hit.tri[2]);
    Some(hit)
}

/// Find the nearest pickable skeletal hit in world space using OBB pruning across all visible bones.
pub fn pick_ogf_bone(
    index: &SkeletonCollisionIndex<'_>,
    palette: &BonePalette,
    parent_xform: &XRayMatrix,
    visible_bones: &BoneMask,
    max_dist: f32,
    origin_world: [f32; 3],
    dir_world: [f32; 3],
) -> Option<BonePickHit> {
    let inv_parent = matrix_invert_43(parent_xform)?;
    let s = matrix_transform_point(&inv_parent, origin_world);
    let d = matrix_transform_dir(&inv_parent, dir_world);

    let (mut hit, _) = index.pick_nearest_model_space_with_obbs(palette, visible_bones, max_dist, s, d);
    if let Some(ref mut h) = hit {
        h.tri = [
            matrix_transform_point(parent_xform, h.tri[0]),
            matrix_transform_point(parent_xform, h.tri[1]),
            matrix_transform_point(parent_xform, h.tri[2]),
        ];
        h.normal = tri_mknormal_zero(h.tri[0], h.tri[1], h.tri[2]);
    }
    hit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use crate::ogf::ogf_chunks::*;
    use crate::ogf::{
        skel_vert_type, OgfObb, OgfVisualType, VertBoned1W, VertBoned2W, VertBoned3W, VertBoned4W,
    };
    use crate::omf::{motion_def_flags, motion_key_flags};
    use std::f32::consts::FRAC_PI_2;

    fn write_vec3(out: &mut Vec<u8>, v: [f32; 3]) {
        for c in v {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }

    fn write_ik_bone(ik_data: &mut Vec<u8>, bind_trans: [f32; 3]) {
        ik_data.extend_from_slice(&1u32.to_le_bytes());
        ik_data.extend_from_slice(b"default\0");
        ik_data.extend_from_slice(&0u16.to_le_bytes()); // shape none
        ik_data.extend_from_slice(&0u16.to_le_bytes());
        for _ in 0..27 {
            ik_data.extend_from_slice(&0.0f32.to_le_bytes());
        }
        ik_data.extend_from_slice(&0u32.to_le_bytes()); // joint_type
        for _ in 0..3 {
            ik_data.extend_from_slice(&[0u8; 16]);
        }
        ik_data.extend_from_slice(&[0u8; 24]); // spring, damping, flags, break_force, break_torque, friction
        write_vec3(ik_data, [0.0, 0.0, 0.0]);
        write_vec3(ik_data, bind_trans);
        ik_data.extend_from_slice(&5.0f32.to_le_bytes());
        write_vec3(ik_data, [0.0, 0.0, 0.0]);
    }

    fn build_two_bone_skeleton_and_omf() -> (OgfSkeleton, OmfMotions) {
        let mut ogf_w = ChunkWriter::new();
        let mut hdr = vec![4u8, OgfVisualType::SkeletonAnim as u8, 0, 0];
        hdr.extend_from_slice(&[0u8; 40]);
        ogf_w.write_chunk(OGF_HEADER, &hdr);

        let id_obb = OgfObb {
            rotate: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translate: [0.0, 0.0, 0.0],
            halfsize: [0.2, 0.2, 0.2],
        };
        let mut bn = Vec::new();
        bn.extend_from_slice(&2u32.to_le_bytes());
        bn.extend_from_slice(b"root\0\0");
        for r in id_obb.rotate {
            write_vec3(&mut bn, r);
        }
        write_vec3(&mut bn, id_obb.translate);
        write_vec3(&mut bn, id_obb.halfsize);
        bn.extend_from_slice(b"arm\0root\0");
        for r in id_obb.rotate {
            write_vec3(&mut bn, r);
        }
        write_vec3(&mut bn, id_obb.translate);
        write_vec3(&mut bn, id_obb.halfsize);
        ogf_w.write_chunk(OGF_S_BONE_NAMES, &bn);

        let mut ik = Vec::new();
        write_ik_bone(&mut ik, [0.0, 0.0, 0.0]);
        // In bind pose arm sits at (0, 2, 0) in model space.
        write_ik_bone(&mut ik, [0.0, 2.0, 0.0]);
        ogf_w.write_chunk(OGF_S_IKDATA, &ik);

        let model = OgfModel::parse(&ogf_w.into_bytes()).unwrap();
        let skel = model.skeleton.unwrap();

        // Build OMF with 2 clips:
        // clip 0: root identity, arm sits at (0, 2, 0) -> bind pose equivalent
        // clip 1: root rotated +90 deg around Z (x,y,z,w = 0,0,sin(45),cos(45)), arm keeps local (0, 2, 0)
        // Result of parent +90deg Z rotation (XRay right/up rule: +Y goes to -X in XRay mk_xform check):
        // Using exact `BoneKeyframe::to_matrix` for rot(Z, +pi/2) -> model(arm) = (-2, 0, 0) or (+2, 0, 0).
        let mut mp = Vec::new();
        mp.extend_from_slice(&4u16.to_le_bytes());
        mp.extend_from_slice(&1u16.to_le_bytes());
        mp.extend_from_slice(b"default\0");
        mp.extend_from_slice(&2u16.to_le_bytes());
        mp.extend_from_slice(b"root\0");
        mp.extend_from_slice(&0u32.to_le_bytes());
        mp.extend_from_slice(b"arm\0");
        mp.extend_from_slice(&1u32.to_le_bytes());
        mp.extend_from_slice(&2u16.to_le_bytes());
        for (idx, name) in ["bind", "rot90"].iter().enumerate() {
            mp.extend_from_slice(name.as_bytes());
            mp.push(0);
            mp.extend_from_slice(&(motion_def_flags::ESM_IDLE as u32).to_le_bytes());
            mp.extend_from_slice(&0u16.to_le_bytes());
            mp.extend_from_slice(&(idx as u16).to_le_bytes());
            for f in [1.0f32, 1.0, 1.0, 0.5] {
                mp.extend_from_slice(&f.to_le_bytes());
            }
            mp.extend_from_slice(&0u32.to_le_bytes());
        }

        let make_clip = |name: &str, root_q: [i16; 4]| -> Vec<u8> {
            let mut clip = Vec::new();
            clip.extend_from_slice(name.as_bytes());
            clip.push(0);
            clip.extend_from_slice(&1u32.to_le_bytes()); // 1 frame
            // Track 0 (root): static q, origin t
            clip.push(motion_key_flags::FL_RKEY_ABSENT);
            for q in root_q {
                clip.extend_from_slice(&q.to_le_bytes());
            }
            write_vec3(&mut clip, [0.0, 0.0, 0.0]);
            // Track 1 (arm): identity q, static [0, 2, 0]
            clip.push(motion_key_flags::FL_RKEY_ABSENT);
            for q in [0i16, 0, 0, 32767] {
                clip.extend_from_slice(&q.to_le_bytes());
            }
            write_vec3(&mut clip, [0.0, 2.0, 0.0]);
            clip
        };

        let q90 = ((FRAC_PI_2 * 0.5).sin() * 32767.0).round() as i16; // ~23170
        let mut ms = ChunkWriter::new();
        ms.write_chunk(0, &2u32.to_le_bytes());
        ms.write_chunk(1, &make_clip("bind", [0, 0, 0, 32767]));
        ms.write_chunk(2, &make_clip("rot90", [0, 0, q90, q90]));

        let mut omf_w = ChunkWriter::new();
        omf_w.write_chunk(OGF_S_SMPARAMS, &mp);
        omf_w.write_chunk(OGF_S_MOTIONS, &ms.into_bytes());
        let motions = OmfMotions::parse(&omf_w.into_bytes()).unwrap();
        (skel, motions)
    }

    #[test]
    fn test_skinning_bind_pose_omf_pivot_and_1w_2w_3w_4w() {
        let (skel, motions) = build_two_bone_skeleton_and_omf();

        // 1. Bind pose leaves vertices completely unchanged and 4x3 matrices equal identity
        let bind_pal = BonePalette::bind_pose(&skel).unwrap();
        let gpu4x3 = bind_pal.to_gpu_4x3_matrices();
        assert_eq!(gpu4x3, vec![BoneMatrix4x3::IDENTITY, BoneMatrix4x3::IDENTITY]);

        let v2w = OgfVertices::Skinned2W(vec![VertBoned2W {
            matrix0: 0,
            matrix1: 1,
            position: [0.0, 2.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            binormal: [0.0, 0.0, 1.0],
            weight: 0.5,
            u: 0.25,
            v: 0.75,
        }]);
        let skinned_bind = skin_vertices(&v2w, &bind_pal).unwrap();
        assert_eq!(skinned_bind[0].position, [0.0, 2.0, 0.0]);
        assert_eq!(skinned_bind[0].uv, [0.25, 0.75]);

        // 2. Clip 1 pivots entire skeleton by root rotation (+90 deg around Z in mk_xform)
        // Because `arm` is at (0, 2, 0) in both bind pose and clip 1 local space, a vertex glued
        // 100% to `arm` at model-space bind coords (0, 2, 0) (local 0,0,0 to `arm`) moves solely
        // with `arm`'s world origin, AND `BoneMatrix4x3::transform_point` matches CPU skinning!
        let rot_pal = BonePalette::from_omf_clip(&skel, &motions, 1, 0.0).unwrap();
        let v1w_arm = OgfVertices::Skinned1W(vec![VertBoned1W {
            matrix: 1,
            position: [0.0, 2.0, 0.0], // coincides with `arm` bind origin
            normal: [1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            binormal: [0.0, 0.0, 1.0],
            u: 0.0,
            v: 0.0,
        }]);
        let skinned_arm = skin_vertices(&v1w_arm, &rot_pal).unwrap();
        let expected_arm_pos = rot_pal.model_transforms[1][3];
        assert!((skinned_arm[0].position[0] - expected_arm_pos[0]).abs() < 1e-4);
        assert!((skinned_arm[0].position[1] - expected_arm_pos[1]).abs() < 1e-4);
        assert!((skinned_arm[0].position[2] - expected_arm_pos[2]).abs() < 1e-4);
        // X-Ray `mk_xform` rotates X normal toward -Y (or +Y): check orthogonality & GPU 4x3 equivalence
        let gpu_rot = rot_pal.to_gpu_4x3_matrices();
        let gpu_p = gpu_rot[1].transform_point([0.0, 2.0, 0.0]);
        let gpu_n = gpu_rot[1].transform_dir([1.0, 0.0, 0.0]);
        assert!((gpu_p[0] - skinned_arm[0].position[0]).abs() < 1e-5);
        assert!((gpu_p[1] - skinned_arm[0].position[1]).abs() < 1e-5);
        assert!((gpu_n[0] - skinned_arm[0].normal[0]).abs() < 1e-5);
        assert!((gpu_n[1] - skinned_arm[0].normal[1]).abs() < 1e-5);
        assert!(skinned_arm[0].normal[0].abs() < 1e-3);
        assert!(skinned_arm[0].normal[1].abs() > 0.999);

        // 3. 50/50 SLERP blend between bind (0deg) and rot90 (90deg) produces ~45deg rotated normal
        let blend_pal = BonePalette::from_omf_blend(&skel, &motions, 0, 0.0, 1, 0.0, 0.5).unwrap();
        let skinned_blend = skin_vertices(&v1w_arm, &blend_pal).unwrap();
        let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert!((skinned_blend[0].normal[0] - inv_sqrt2).abs() < 1e-3);
        assert!((skinned_blend[0].normal[1].abs() - inv_sqrt2).abs() < 1e-3);

        // 4. 3W and 4W complement weights test + AABB helper + OOB bone rejection
        let v3w = OgfVertices::Skinned3W(vec![VertBoned3W {
            matrices: [0, 1, 1],
            position: [0.0, 2.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            binormal: [0.0, 0.0, 1.0],
            weights: [0.25, 0.50], // w2 = 0.25 -> bone 0 gets 0.25, bone 1 gets 0.75
            u: 0.1,
            v: 0.2,
        }]);
        let s3 = skin_vertices(&v3w, &rot_pal).unwrap();
        let v4w = OgfVertices::Skinned4W(vec![VertBoned4W {
            matrices: [0, 1, 1, 1],
            position: [0.0, 2.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            tangent: [0.0, 1.0, 0.0],
            binormal: [0.0, 0.0, 1.0],
            weights: [0.25, 0.25, 0.25], // w3 = 0.25 -> same 0.25 / 0.75 split
            u: 0.1,
            v: 0.2,
        }]);
        let s4 = skin_vertices(&v4w, &rot_pal).unwrap();
        assert!((s3[0].position[0] - s4[0].position[0]).abs() < 1e-5);
        assert!((s3[0].position[1] - s4[0].position[1]).abs() < 1e-5);

        let aabb = compute_skinned_aabb(&s3).unwrap();
        assert_eq!(aabb.min, s3[0].position);
        assert_eq!(aabb.max, s3[0].position);

        let oob = OgfVertices::Skinned1W(vec![VertBoned1W {
            matrix: 99,
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            binormal: [0.0, 0.0, 1.0],
            u: 0.0,
            v: 0.0,
        }]);
        assert!(matches!(
            skin_vertices(&oob, &bind_pal),
            Err(FormatError::InvalidValue(_))
        ));

        // Hierarchy helper sanity check on child mesh
        let mut child_w = ChunkWriter::new();
        let mut chdr = vec![4u8, OgfVisualType::SkeletonGeomdefSt as u8, 0, 0];
        chdr.extend_from_slice(&[0u8; 40]);
        child_w.write_chunk(OGF_HEADER, &chdr);
        child_w.write_chunk(OGF_TEXTURE, b"act\\hero\0models\\model\0");
        let mut vb = Vec::new();
        vb.extend_from_slice(&skel_vert_type::FVF_1L.to_le_bytes());
        vb.extend_from_slice(&1u32.to_le_bytes());
        write_vec3(&mut vb, [0.0, 2.0, 0.0]);
        write_vec3(&mut vb, [1.0, 0.0, 0.0]);
        write_vec3(&mut vb, [0.0, 1.0, 0.0]);
        write_vec3(&mut vb, [0.0, 0.0, 1.0]);
        vb.extend_from_slice(&0.0f32.to_le_bytes());
        vb.extend_from_slice(&0.0f32.to_le_bytes());
        vb.extend_from_slice(&1u32.to_le_bytes());
        child_w.write_chunk(OGF_VERTICES, &vb);
        child_w.write_chunk(OGF_INDICES, &0u32.to_le_bytes());
        let child = OgfModel::parse(&child_w.into_bytes()).unwrap();
        let parts = skin_ogf_hierarchy(&child, &bind_pal).unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].vertices[0].position, [0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_bone_mask_arbitrary_counts_and_legacy_u64() {
        let mut small = BoneMask::all(3);
        assert!(small.is_visible(0) && small.is_visible(2) && !small.is_visible(3));
        assert_eq!(small.as_u64(), 0b111);
        small.set_visible(1, false);
        assert!(!small.is_visible(1));
        assert_eq!(small.as_u64(), 0b101);

        let mut wide = BoneMask::all(70);
        assert!(wide.is_visible(69) && !wide.is_visible(70));
        wide.set_visible(67, false);
        assert!(!wide.is_visible(67) && wide.is_visible(68));

        let from_bits = BoneMask::from_u64(0x5, 4);
        assert!(from_bits.is_visible(0) && !from_bits.is_visible(1) && from_bits.is_visible(2));
        assert!(!BoneMask::empty(10).is_visible(0));
    }

    #[test]
    fn test_skeleton_collision_index_picking_animated_obbs_flags_and_swi() {
        use crate::ogf::{bone_shape_flags, matrix_set_xyz_translate, OgfBone, BoneIkData, BoneIkSection, BoneShape, BoneShapeType};

        fn make_obb_bone(
            id: u16,
            name: &str,
            parent_name: &str,
            parent_id: Option<u16>,
            bind_trans: [f32; 3],
            obb_center_local: [f32; 3],
            no_pickable: bool,
        ) -> OgfBone {
            let obb = OgfObb {
                rotate: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                translate: obb_center_local,
                halfsize: [0.8, 0.8, 0.8],
            };
            OgfBone {
                id,
                name: name.to_string(),
                parent_name: parent_name.to_string(),
                parent_id,
                children: if id == 0 { vec![1] } else { Vec::new() },
                obb,
                ik: Some(BoneIkSection {
                    game_mtl_name: "default".to_string(),
                    shape: BoneShape {
                        shape_type: BoneShapeType::Box as u16,
                        flags: if no_pickable {
                            bone_shape_flags::SF_NO_PICKABLE
                        } else {
                            0
                        },
                        box_obb: obb,
                        sphere_center: [0.0; 3],
                        sphere_radius: 0.0,
                        cylinder_center: [0.0; 3],
                        cylinder_direction: [0.0, 1.0, 0.0],
                        cylinder_height: 0.0,
                        cylinder_radius: 0.0,
                    },
                    ik_data: BoneIkData {
                        version: 1,
                        joint_type: 0,
                        limits: [crate::ogf::JointLimit {
                            limit: [0.0, 0.0],
                            spring_factor: 0.0,
                            damping_factor: 0.0,
                        }; 3],
                        spring_factor: 0.0,
                        damping_factor: 0.0,
                        ik_flags: 0,
                        break_force: 0.0,
                        break_torque: 0.0,
                        friction: 0.0,
                    },
                    bind_rotation_xyz: [0.0; 3],
                    bind_translation: bind_trans,
                    mass: 1.0,
                    center_of_mass: [0.0; 3],
                }),
            }
        }

        // Write a valid SkeletonAnim parent with child SkeletonGeomdefPm whose SWI max LOD cuts off tri #1
        let mut ogf_w = ChunkWriter::new();
        let mut hdr = vec![4u8, OgfVisualType::SkeletonAnim as u8, 0, 0];
        hdr.extend_from_slice(&[0u8; 40]);
        ogf_w.write_chunk(OGF_HEADER, &hdr);
        let id_obb = OgfObb {
            rotate: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translate: [0.0; 3],
            halfsize: [0.8, 0.8, 0.8],
        };
        let mut bn = Vec::new();
        bn.extend_from_slice(&2u32.to_le_bytes());
        bn.extend_from_slice(b"root\0\0");
        for r in id_obb.rotate {
            write_vec3(&mut bn, r);
        }
        write_vec3(&mut bn, [0.0; 3]);
        write_vec3(&mut bn, id_obb.halfsize);
        bn.extend_from_slice(b"arm\0root\0");
        for r in id_obb.rotate {
            write_vec3(&mut bn, r);
        }
        write_vec3(&mut bn, [0.0; 3]);
        write_vec3(&mut bn, id_obb.halfsize);
        ogf_w.write_chunk(OGF_S_BONE_NAMES, &bn);
        let mut ik = Vec::new();
        write_ik_bone(&mut ik, [0.0, 0.0, 0.0]);
        write_ik_bone(&mut ik, [0.0, 2.0, 0.0]);
        ogf_w.write_chunk(OGF_S_IKDATA, &ik);

        let mut model = OgfModel::parse(&ogf_w.into_bytes()).unwrap();
        // Replace bones with explicit OBB extents covering their triangles
        let skel = model.skeleton.as_mut().unwrap();
        skel.bones[0] = make_obb_bone(0, "root", "", None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], false);
        skel.bones[1] = make_obb_bone(1, "arm", "root", Some(0), [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], false);

        // Child mesh with 3 triangles:
        // tri 0 (bone 0, z=0 facing -Z): (-0.5, -0.5, 0), (0.0, 0.5, 0), (0.5, -0.5, 0)
        // tri 1 (bone 1, z=0 facing -Z): (-0.5, 1.5, 0), (0.0, 2.5, 0), (0.5, 1.5, 0)
        // tri 2 (bone 1, z=-1): outside SWI windows[0] -> must NEVER be picked!
        let mut child_w = ChunkWriter::new();
        let mut chdr = vec![4u8, OgfVisualType::SkeletonGeomdefPm as u8, 0, 0];
        chdr.extend_from_slice(&[0u8; 40]);
        child_w.write_chunk(OGF_HEADER, &chdr);
        child_w.write_chunk(OGF_TEXTURE, b"act\\hero\0models\\model\0");
        let tri_verts = [
            // tri 0 -> bone 0
            (0u32, [-0.5f32, -0.5, 0.0]),
            (0, [0.0, 0.5, 0.0]),
            (0, [0.5, -0.5, 0.0]),
            // tri 1 -> bone 1
            (1, [-0.5, 1.5, 0.0]),
            (1, [0.0, 2.5, 0.0]),
            (1, [0.5, 1.5, 0.0]),
            // tri 2 -> bone 1 (outside SWI[0])
            (1, [-0.5, 1.5, -1.0]),
            (1, [0.0, 2.5, -1.0]),
            (1, [0.5, 1.5, -1.0]),
        ];
        let mut vb = Vec::new();
        vb.extend_from_slice(&skel_vert_type::FVF_1L.to_le_bytes());
        vb.extend_from_slice(&(tri_verts.len() as u32).to_le_bytes());
        for (b, pos) in tri_verts {
            write_vec3(&mut vb, pos);
            write_vec3(&mut vb, [0.0, 0.0, -1.0]);
            write_vec3(&mut vb, [1.0, 0.0, 0.0]);
            write_vec3(&mut vb, [0.0, 1.0, 0.0]);
            vb.extend_from_slice(&0.0f32.to_le_bytes());
            vb.extend_from_slice(&0.0f32.to_le_bytes());
            vb.extend_from_slice(&b.to_le_bytes());
        }
        child_w.write_chunk(OGF_VERTICES, &vb);
        let mut ib = Vec::new();
        ib.extend_from_slice(&(9u32).to_le_bytes());
        for i in 0u16..9 {
            ib.extend_from_slice(&i.to_le_bytes());
        }
        child_w.write_chunk(OGF_INDICES, &ib);
        let mut swi_b = Vec::new();
        swi_b.extend_from_slice(&[0u8; 16]);
        swi_b.extend_from_slice(&1u32.to_le_bytes()); // 1 window covering tri 0 & tri 1 (2 tris)
        swi_b.extend_from_slice(&0u32.to_le_bytes());
        swi_b.extend_from_slice(&2u16.to_le_bytes());
        swi_b.extend_from_slice(&6u16.to_le_bytes());
        child_w.write_chunk(OGF_SWIDATA, &swi_b);
        let child = OgfModel::parse(&child_w.into_bytes()).unwrap();
        model.children.push(child);

        let index = SkeletonCollisionIndex::from_ogf(&model).unwrap();
        assert_eq!(index.bone_child_faces[0][0], vec![0]);
        assert_eq!(index.bone_child_faces[1][0], vec![1]);

        let bind_pal = BonePalette::bind_pose(index.skeleton).unwrap();
        let all_vis = BoneMask::all(2);
        let parent_xform = matrix_set_xyz_translate([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);

        // Hit bone 1 in world space at x=10, y=2
        let hit = pick_ogf_bone(
            &index,
            &bind_pal,
            &parent_xform,
            &all_vis,
            100.0,
            [10.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
        )
        .unwrap();
        assert_eq!(hit.bone_id, 1);
        assert!((hit.dist - 5.0).abs() < 1e-4);
        assert!((hit.normal[2] + 1.0).abs() < 1e-4);

        // Hiding bone 1 in BoneMask makes `pick_ogf_bone` miss y=2
        let mut only_root = BoneMask::all(2);
        only_root.set_visible(1, false);
        assert!(pick_ogf_bone(
            &index,
            &bind_pal,
            &parent_xform,
            &only_root,
            100.0,
            [10.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
        )
        .is_none());

        // Shifting bone 1 to X=+3 moves both bone 1's OBB and triangle from X=0 to X=3 (world X=13)
        let moved_models = [
            XRAY_IDENTITY_MATRIX,
            matrix_set_xyz_translate([0.0, 0.0, 0.0], [3.0, 2.0, 0.0]),
        ];
        let moved_pal = BonePalette::from_model_transforms(index.skeleton, &moved_models).unwrap();
        assert!(pick_ogf_bone(
            &index,
            &moved_pal,
            &parent_xform,
            &all_vis,
            100.0,
            [10.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
        )
        .is_none());
        let moved_hit = pick_ogf_bone(
            &index,
            &moved_pal,
            &parent_xform,
            &all_vis,
            100.0,
            [13.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
        )
        .unwrap();
        assert_eq!(moved_hit.bone_id, 1);

        // SF_NO_PICKABLE skips bone 1 in `pick_ogf_bone` while `pick_ogf_single_bone` still checks explicit bone
        model.skeleton.as_mut().unwrap().bones[1]
            .ik
            .as_mut()
            .unwrap()
            .shape
            .flags |= bone_shape_flags::SF_NO_PICKABLE;
        let index_nopick = SkeletonCollisionIndex::from_ogf(&model).unwrap();
        assert!(pick_ogf_bone(
            &index_nopick,
            &bind_pal,
            &parent_xform,
            &all_vis,
            100.0,
            [10.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
        )
        .is_none());
        assert!(pick_ogf_single_bone(
            &index_nopick,
            &bind_pal,
            &parent_xform,
            1,
            100.0,
            [10.0, 2.0, -5.0],
            [0.0, 0.0, 1.0],
        )
        .is_some());

        // Explicit `link_count` handles repeated consecutive bone IDs in 3W without C++ sentinel mis-dispatch
        let p_links = skin_position_with_links(
            [0.0, 2.0, 0.0],
            3,
            [0, 1, 1, 1],
            [0.25, 0.5, 0.0],
            &moved_pal.render_transforms,
        )
        .unwrap();
        assert!((p_links[0] - 2.25).abs() < 1e-4);
    }
}
