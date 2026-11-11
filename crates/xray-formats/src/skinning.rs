//! Pure Rust X-Ray Skeletal Skinning, Bone Palette & Animation Blending
//!
//! Replaces legacy C++ `CKinematics::CLBone` / `xrSkin1W..4W` CPU skinning paths
//! (`SkeletonRigid.cpp`, `SkeletonCustom.cpp`, `SkeletonX.cpp`, `xrSkin2W.cpp`) with
//! deterministic pure-Rust `BonePalette` construction and `1W`/`2W`/`3W`/`4W` skinning.

use crate::chunk::FormatError;
use crate::ogf::{
    matrix_mul_43, matrix_transform_point, BonePoseMatrices, OgfBBox, OgfModel, OgfSkeleton,
    OgfTextureLink, OgfVertices, StaticVertex, XRayMatrix, XRAY_IDENTITY_MATRIX,
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
}
