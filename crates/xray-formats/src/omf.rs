//! Pure Rust X-Ray `.omf` / Embedded OGF Skeletal Motion Clips & Parameters Parser
//!
//! Faithful pure-Rust implementation matching `src/xrEngine/SkeletonMotions.h`,
//! `src/xrEngine/SkeletonMotions.cpp`, `src/xrEngine/SkeletonMotionDefs.h`,
//! `src/xrEngine/motion.h`, and `src/Layers/xrRender/AnimationKeyCalculate.h`.
//!
//! Works identically for standalone `.omf` files referenced by `OGF_S_MOTION_REFS`
//! and embedded `OGF_S_SMPARAMS` / `OGF_S_MOTIONS` chunks inside `.ogf` v4 models.

use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::ogf::ogf_chunks::{OGF_S_MOTIONS, OGF_S_SMPARAMS};
use crate::ogf::{matrix_mul_43, OgfSkeleton, XRayMatrix};

/// Maximum `OGF_S_SMPARAMS` format version supported by X-Ray Anomaly (`Fmesh.h:86`).
pub const XR_OGF_SM_PARAMS_VERSION: u16 = 4;
/// X-Ray skeletal sample rate (`SkeletonMotionDefs.h:8-12`).
pub const SAMPLE_FPS: f32 = 30.0;
pub const SAMPLE_SPF: f32 = 1.0 / SAMPLE_FPS;
/// Quaternion 16-bit dequantization factor (`KEY_Quant = 32767.0`).
pub const KEY_QUANT: f32 = 32767.0;
pub const KEY_QUANT_INV: f32 = 1.0 / KEY_QUANT;
/// Motion parameter accrue/falloff multiplier (`fQuantizerRangeExt` in `SkeletonMotions.h:129`).
pub const MOTION_QUANTIZER_RANGE_EXT: f32 = 1.5;

/// `ESMFlags` (`src/xrEngine/motion.h:148-158`).
pub mod motion_def_flags {
    pub const ESM_FX: u16 = 1 << 0;
    pub const ESM_STOP_AT_END: u16 = 1 << 1;
    pub const ESM_NO_MIX: u16 = 1 << 2;
    pub const ESM_SYNC_PART: u16 = 1 << 3;
    pub const ESM_USE_FOOT_STEPS: u16 = 1 << 4;
    pub const ESM_ROOT_MOVER: u16 = 1 << 5;
    pub const ESM_IDLE: u16 = 1 << 6;
    pub const ESM_USE_WEAPON_BONE: u16 = 1 << 7;
}

/// Bone keyframe channel presence flags (`src/xrEngine/SkeletonMotions.h:18-23`).
pub mod motion_key_flags {
    pub const FL_TKEY_PRESENT: u8 = 1 << 0;
    pub const FL_RKEY_ABSENT: u8 = 1 << 1;
    pub const FL_TKEY_16_IS_BIT: u8 = 1 << 2;
}

/// Named bone entry inside an OMF partition (`SkeletonMotions.cpp:118-121`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionPartitionBone {
    pub name: String,
    /// Sequential motion-track bone index stored in `OGF_S_MOTIONS` (`rm_bones` index).
    pub motion_bone_index: u32,
}

/// Skeletal partition grouping (`CPartDef` in `SkeletonMotions.h:174`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MotionPartition {
    pub name: String,
    pub bones: Vec<MotionPartitionBone>,
}

/// Time interval mark attached to a v4 skeletal animation clip (`motion_marks`).
#[derive(Debug, Clone, PartialEq)]
pub struct MotionMark {
    pub name: String,
    pub intervals: Vec<(f32, f32)>,
}

impl MotionMark {
    pub fn pick_mark(&self, time_sec: f32) -> Option<(f32, f32)> {
        for &(start, end) in &self.intervals {
            if start <= time_sec && time_sec <= end {
                return Some((start, end));
            }
            if start > time_sec {
                break;
            }
        }
        None
    }
}

/// Motion definition & blend metadata (`CMotionDef` in `SkeletonMotions.h:131`).
#[derive(Debug, Clone, PartialEq)]
pub struct MotionDef {
    pub name: String,
    pub bone_or_part: u16,
    pub motion: u16,
    pub speed: f32,
    pub power: f32,
    pub accrue: f32,
    pub falloff: f32,
    pub flags: u16,
    pub marks: Vec<MotionMark>,
}

impl MotionDef {
    #[inline]
    pub fn is_fx(&self) -> bool {
        (self.flags & motion_def_flags::ESM_FX) != 0
    }

    #[inline]
    pub fn stop_at_end(&self) -> bool {
        (self.flags & motion_def_flags::ESM_STOP_AT_END) != 0
    }

    /// Quantize float to X-Ray `u16` (`[0.0, 100.0]`) matching `CMotionDef::Quantize`.
    #[inline]
    pub fn quantize_param(val: f32) -> u16 {
        let scaled = (val * 655.35).floor();
        if scaled <= 0.0 {
            0
        } else if scaled >= 65535.0 {
            65535
        } else {
            scaled as u16
        }
    }

    #[inline]
    pub fn dequantize_param(q: u16) -> f32 {
        (q as f32) / 655.35
    }

    /// Engine-quantized playback speed (`CMotionDef::Speed()`).
    pub fn quantized_speed(&self) -> f32 {
        Self::dequantize_param(Self::quantize_param(self.speed))
    }

    /// Engine-quantized blend power (`CMotionDef::Power()`).
    pub fn quantized_power(&self) -> f32 {
        Self::dequantize_param(Self::quantize_param(self.power))
    }

    /// Engine-quantized accrue blend rate (`CMotionDef::Accrue()`).
    pub fn effective_accrue(&self) -> f32 {
        MOTION_QUANTIZER_RANGE_EXT * Self::dequantize_param(Self::quantize_param(self.accrue))
    }

    /// Engine-quantized falloff blend rate (`CMotionDef::Falloff()`), matching X-Ray's
    /// cycle rule clamping `falloff < accrue`.
    pub fn effective_falloff(&self) -> f32 {
        let accrue_q = Self::quantize_param(self.accrue);
        let mut falloff_q = Self::quantize_param(self.falloff);
        if !self.is_fx() && falloff_q >= accrue_q {
            falloff_q = accrue_q.saturating_sub(1);
        }
        MOTION_QUANTIZER_RANGE_EXT * Self::dequantize_param(falloff_q)
    }
}

/// Quantized 16-bit quaternion key (`CKeyQR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedRotKey {
    pub x: i16,
    pub y: i16,
    pub z: i16,
    pub w: i16,
}

impl QuantizedRotKey {
    /// Dequantize into `[x, y, z, w]` matching `QR2Quat` (`AnimationKeyCalculate.h:60-66`).
    #[inline]
    pub fn to_quat_xyzw(self) -> [f32; 4] {
        [
            (self.x as f32) * KEY_QUANT_INV,
            (self.y as f32) * KEY_QUANT_INV,
            (self.z as f32) * KEY_QUANT_INV,
            (self.w as f32) * KEY_QUANT_INV,
        ]
    }
}

/// Quantized bone translation channel (`_keysT8` / `_keysT16` / static `_initT`).
#[derive(Debug, Clone, PartialEq)]
pub enum BoneTranslationKeys {
    Static {
        init_t: [f32; 3],
    },
    Quantized8 {
        crc32: u32,
        keys: Vec<[i8; 3]>,
        size_t: [f32; 3],
        init_t: [f32; 3],
    },
    Quantized16 {
        crc32: u32,
        keys: Vec<[i16; 3]>,
        size_t: [f32; 3],
        init_t: [f32; 3],
    },
}

/// Dequantized single-bone keyframe `(rotation_xyzw, translation)` (`struct CKey`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoneKeyframe {
    pub rotation_xyzw: [f32; 4],
    pub translation: [f32; 3],
}

impl BoneKeyframe {
    /// Blend two keyframes by `t` in `[0.0, 1.0]` using shortest-arc quaternion SLERP
    /// and linear translation interpolation (`AnimationBlend` / `MixChannels`).
    pub fn blend(a: Self, b: Self, t: f32) -> Self {
        let weight = t.clamp(0.0, 1.0);
        Self {
            rotation_xyzw: slerp_quat_xyzw(a.rotation_xyzw, b.rotation_xyzw, weight),
            translation: [
                a.translation[0] + (b.translation[0] - a.translation[0]) * weight,
                a.translation[1] + (b.translation[1] - a.translation[1]) * weight,
                a.translation[2] + (b.translation[2] - a.translation[2]) * weight,
            ],
        }
    }

    /// Convert dequantized keyframe into local bone transform matrix (`Fmatrix::mk_xform`).
    pub fn to_matrix(self) -> XRayMatrix {
        let [x, y, z, w] = normalize_quat_xyzw(self.rotation_xyzw);
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        // X-Ray `src/xrCore/vector.h::mk_xform(Fquaternion, Fvector)` exact layout
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
            [self.translation[0], self.translation[1], self.translation[2], 1.0],
        ]
    }
}

/// Per-bone animation keyframe track inside a clip (`class CMotion`).
#[derive(Debug, Clone, PartialEq)]
pub struct BoneMotionTrack {
    pub flags: u8,
    pub frame_count: u32,
    pub rot_crc32: Option<u32>,
    pub rot_keys: Vec<QuantizedRotKey>,
    pub translation: BoneTranslationKeys,
}

impl BoneMotionTrack {
    #[inline]
    pub fn duration_seconds(&self) -> f32 {
        (self.frame_count as f32) * SAMPLE_SPF
    }

    /// Evaluate un-interpolated quantized keyframe at integer `frame` (wraps mod `frame_count`).
    pub fn sample_at_frame(&self, frame: u32) -> BoneKeyframe {
        let count = self.frame_count.max(1) as usize;
        let idx = (frame as usize) % count;

        let rotation_xyzw = if (self.flags & motion_key_flags::FL_RKEY_ABSENT) != 0 {
            self.rot_keys
                .first()
                .map(|k| k.to_quat_xyzw())
                .unwrap_or([0.0, 0.0, 0.0, 1.0])
        } else {
            self.rot_keys
                .get(idx)
                .map(|k| k.to_quat_xyzw())
                .unwrap_or([0.0, 0.0, 0.0, 1.0])
        };

        let translation = match &self.translation {
            BoneTranslationKeys::Static { init_t } => *init_t,
            BoneTranslationKeys::Quantized8 {
                keys,
                size_t,
                init_t,
                ..
            } => {
                let k = keys.get(idx).copied().unwrap_or([0, 0, 0]);
                [
                    (k[0] as f32) * size_t[0] + init_t[0],
                    (k[1] as f32) * size_t[1] + init_t[1],
                    (k[2] as f32) * size_t[2] + init_t[2],
                ]
            }
            BoneTranslationKeys::Quantized16 {
                keys,
                size_t,
                init_t,
                ..
            } => {
                let k = keys.get(idx).copied().unwrap_or([0, 0, 0]);
                [
                    (k[0] as f32) * size_t[0] + init_t[0],
                    (k[1] as f32) * size_t[1] + init_t[1],
                    (k[2] as f32) * size_t[2] + init_t[2],
                ]
            }
        };

        BoneKeyframe {
            rotation_xyzw,
            translation,
        }
    }

    /// Evaluate smooth interpolated keyframe at `time_sec` matching `AnimationKeyCalculate::Dequantize`.
    pub fn sample_at_time(&self, time_sec: f32) -> BoneKeyframe {
        let time_frames = time_sec.max(0.0) * SAMPLE_FPS;
        let frame = time_frames.floor() as u32;
        let delta = (time_frames - (frame as f32)).clamp(0.0, 1.0);

        if (self.flags & motion_key_flags::FL_RKEY_ABSENT) != 0
            && matches!(self.translation, BoneTranslationKeys::Static { .. })
        {
            return self.sample_at_frame(0);
        }

        let k0 = self.sample_at_frame(frame);
        let k1 = self.sample_at_frame(frame.saturating_add(1));

        let rotation_xyzw = if (self.flags & motion_key_flags::FL_RKEY_ABSENT) != 0 {
            k0.rotation_xyzw
        } else {
            slerp_quat_xyzw(k0.rotation_xyzw, k1.rotation_xyzw, delta)
        };

        let translation = [
            k0.translation[0] + (k1.translation[0] - k0.translation[0]) * delta,
            k0.translation[1] + (k1.translation[1] - k0.translation[1]) * delta,
            k0.translation[2] + (k1.translation[2] - k0.translation[2]) * delta,
        ];

        BoneKeyframe {
            rotation_xyzw,
            translation,
        }
    }
}

/// Complete animation clip containing one `BoneMotionTrack` per motion bone (`OGF_S_MOTIONS[m_idx + 1]`).
#[derive(Debug, Clone, PartialEq)]
pub struct MotionClip {
    pub name: String,
    pub frame_count: u32,
    pub tracks: Vec<BoneMotionTrack>,
}

impl MotionClip {
    #[inline]
    pub fn duration_seconds(&self) -> f32 {
        (self.frame_count as f32) * SAMPLE_SPF
    }
}

/// Full parsed `.omf` / embedded OGF skeletal motion container (`motions_value`).
#[derive(Debug, Clone, PartialEq)]
pub struct OmfMotions {
    pub sm_params_version: u16,
    pub partitions: Vec<MotionPartition>,
    pub defs: Vec<MotionDef>,
    pub clips: Vec<MotionClip>,
}

impl OmfMotions {
    /// Parse standalone `.omf` bytes or embedded `OGF_S_SMPARAMS` + `OGF_S_MOTIONS` inside an `.ogf`.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let mp_bytes = reader.require_chunk(OGF_S_SMPARAMS)?;
        let ms_bytes = reader.require_chunk(OGF_S_MOTIONS)?;
        Self::parse_chunks(&mp_bytes, &ms_bytes)
    }

    /// Return `Ok(None)` if `data` has neither `OGF_S_SMPARAMS` nor `OGF_S_MOTIONS` (e.g. static OGF or `OGF_S_MOTION_REFS`),
    /// or `Ok(Some(..))` if embedded skeletal motion chunks exist.
    pub fn parse_if_present(data: &[u8]) -> Result<Option<Self>, FormatError> {
        let reader = ChunkReader::new(data);
        let Some(mp_bytes) = reader.open_chunk(OGF_S_SMPARAMS)? else {
            return Ok(None);
        };
        let ms_bytes = reader.require_chunk(OGF_S_MOTIONS)?;
        Ok(Some(Self::parse_chunks(&mp_bytes, &ms_bytes)?))
    }

    fn parse_chunks(mp_bytes: &[u8], ms_bytes: &[u8]) -> Result<Self, FormatError> {
        let (sm_params_version, partitions, defs) = parse_sm_params(mp_bytes)?;
        let expected_bones: usize = partitions.iter().map(|p| p.bones.len()).sum();
        let clips = parse_motions_chunk(ms_bytes, expected_bones)?;

        Ok(Self {
            sm_params_version,
            partitions,
            defs,
            clips,
        })
    }

    /// Total number of bones referenced across all partitions.
    pub fn partition_bone_count(&self) -> usize {
        self.partitions.iter().map(|p| p.bones.len()).sum()
    }

    /// Look up motion definition by name (ASCII case-insensitive).
    pub fn find_def(&self, name: &str) -> Option<(usize, &MotionDef)> {
        self.defs
            .iter()
            .enumerate()
            .find(|(_, d)| d.name.eq_ignore_ascii_case(name))
    }

    /// Look up motion clip by name (ASCII case-insensitive).
    pub fn find_clip(&self, name: &str) -> Option<&MotionClip> {
        self.clips.iter().find(|c| c.name.eq_ignore_ascii_case(name))
    }

    /// Build `rm_bones` remap table (`motion_bone_index -> OgfBone::id`) matching
    /// `motions_value::load` in `SkeletonMotions.cpp:96-140`.
    pub fn map_bones_to_skeleton(&self, skeleton: &OgfSkeleton) -> Result<Vec<u16>, FormatError> {
        let expected = skeleton.bones.len();
        let mut rm_bones = vec![u16::MAX; expected];

        for part in &self.partitions {
            for pbone in &part.bones {
                let skel_id = skeleton
                    .bone_id(&pbone.name)
                    .ok_or(FormatError::InvalidValue("OMF bone missing from OGF skeleton"))?;
                let m_idx = pbone.motion_bone_index as usize;
                let slot = rm_bones
                    .get_mut(m_idx)
                    .ok_or(FormatError::InvalidValue("OMF bone remap index out of range"))?;
                *slot = skel_id;
            }
        }

        if rm_bones.contains(&u16::MAX) {
            return Err(FormatError::InvalidValue(
                "incomplete OMF partition bone coverage for OGF skeleton",
            ));
        }
        Ok(rm_bones)
    }

    /// Evaluate model-space pose matrices for every bone in `skeleton` at `clip_index` and `time_sec`,
    /// returning one `XRayMatrix` per `skeleton.bones` index.
    pub fn evaluate_skeleton_pose(
        &self,
        skeleton: &OgfSkeleton,
        clip_index: usize,
        time_sec: f32,
    ) -> Result<Vec<XRayMatrix>, FormatError> {
        let clip = self
            .clips
            .get(clip_index)
            .ok_or(FormatError::InvalidValue("OMF clip index out of range"))?;
        let rm_bones = self.map_bones_to_skeleton(skeleton)?;
        if clip.tracks.len() != rm_bones.len() {
            return Err(FormatError::InvalidValue(
                "OMF clip track count does not match skeleton bone count",
            ));
        }

        let mut local_by_skel_id = vec![crate::ogf::XRAY_IDENTITY_MATRIX; skeleton.bones.len()];
        for (track_idx, track) in clip.tracks.iter().enumerate() {
            let skel_id = rm_bones[track_idx] as usize;
            let key = track.sample_at_time(time_sec);
            local_by_skel_id[skel_id] = key.to_matrix();
        }

        accumulate_world_pose(skeleton, &local_by_skel_id)
    }

    /// Evaluate blended model-space pose matrices across two clips using per-bone quaternion SLERP
    /// + translation LERP before hierarchical FK composition (`CKinematicsAnimated::LL_BoneMatrixBuild`).
    pub fn evaluate_blended_pose(
        &self,
        skeleton: &OgfSkeleton,
        clip_a: usize,
        time_a: f32,
        clip_b: usize,
        time_b: f32,
        weight_b: f32,
    ) -> Result<Vec<XRayMatrix>, FormatError> {
        let a = self
            .clips
            .get(clip_a)
            .ok_or(FormatError::InvalidValue("OMF clip_a index out of range"))?;
        let b = self
            .clips
            .get(clip_b)
            .ok_or(FormatError::InvalidValue("OMF clip_b index out of range"))?;
        let rm_bones = self.map_bones_to_skeleton(skeleton)?;
        if a.tracks.len() != rm_bones.len() || b.tracks.len() != rm_bones.len() {
            return Err(FormatError::InvalidValue(
                "OMF clip track count does not match skeleton bone count",
            ));
        }

        let mut local_by_skel_id = vec![crate::ogf::XRAY_IDENTITY_MATRIX; skeleton.bones.len()];
        for (track_idx, (track_a, track_b)) in a.tracks.iter().zip(&b.tracks).enumerate() {
            let skel_id = rm_bones[track_idx] as usize;
            let ka = track_a.sample_at_time(time_a);
            let kb = track_b.sample_at_time(time_b);
            local_by_skel_id[skel_id] = BoneKeyframe::blend(ka, kb, weight_b).to_matrix();
        }

        accumulate_world_pose(skeleton, &local_by_skel_id)
    }
}

fn accumulate_world_pose(
    skeleton: &OgfSkeleton,
    local_by_skel_id: &[XRayMatrix],
) -> Result<Vec<XRayMatrix>, FormatError> {
    let Some(root_id) = skeleton.root_bone else {
        return Ok(local_by_skel_id.to_vec());
    };

    let mut world = vec![crate::ogf::XRAY_IDENTITY_MATRIX; skeleton.bones.len()];
    fn recurse(
        skeleton: &OgfSkeleton,
        bone_id: u16,
        parent_world: &XRayMatrix,
        local_mats: &[XRayMatrix],
        world_out: &mut [XRayMatrix],
        depth: usize,
    ) -> Result<(), FormatError> {
        if depth > skeleton.bones.len() {
            return Err(FormatError::InvalidValue("cycle in OGF skeleton hierarchy"));
        }
        let bone = skeleton
            .bones
            .get(bone_id as usize)
            .ok_or(FormatError::InvalidValue("invalid OGF bone id"))?;
        let local = &local_mats[bone_id as usize];
        let cur_world = matrix_mul_43(parent_world, local);
        world_out[bone_id as usize] = cur_world;
        for &child in &bone.children {
            recurse(skeleton, child, &cur_world, local_mats, world_out, depth + 1)?;
        }
        Ok(())
    }

    recurse(
        skeleton,
        root_id,
        &crate::ogf::XRAY_IDENTITY_MATRIX,
        local_by_skel_id,
        &mut world,
        0,
    )?;
    Ok(world)
}

fn parse_sm_params(data: &[u8]) -> Result<(u16, Vec<MotionPartition>, Vec<MotionDef>), FormatError> {
    let mut cur = ByteCursor::new(data);
    let version = cur.read_u16()?;
    if version > XR_OGF_SM_PARAMS_VERSION {
        return Err(FormatError::InvalidValue(
            "unsupported OGF/OMF SMParams version (> 4)",
        ));
    }

    let part_count = cur.read_u16()? as usize;
    let mut partitions = Vec::with_capacity(part_count);
    for _ in 0..part_count {
        let name = cur.read_zstring()?.to_ascii_lowercase();
        let bone_count = cur.read_u16()? as usize;
        let mut bones = Vec::with_capacity(bone_count.min(cur.remaining() / 5));
        for _ in 0..bone_count {
            let bname = cur.read_zstring()?.to_ascii_lowercase();
            let motion_bone_index = cur.read_u32()?;
            bones.push(MotionPartitionBone {
                name: bname,
                motion_bone_index,
            });
        }
        partitions.push(MotionPartition { name, bones });
    }

    let mot_count = cur.read_u16()? as usize;
    let mut defs = Vec::with_capacity(mot_count);
    for _ in 0..mot_count {
        let name = cur.read_zstring()?.to_ascii_lowercase();
        let flags = cur.read_u32()? as u16;
        let bone_or_part = cur.read_u16()?;
        let motion = cur.read_u16()?;
        let speed = cur.read_f32()?;
        let power = cur.read_f32()?;
        let accrue = cur.read_f32()?;
        let falloff = cur.read_f32()?;

        let mut marks = Vec::new();
        if version >= 4 {
            let mark_count = cur.read_u32()? as usize;
            marks.reserve(mark_count.min(cur.remaining() / 4));
            for _ in 0..mark_count {
                let mark_name = cur.read_line_string()?;
                let interval_count = cur.read_u32()? as usize;
                let need = interval_count
                    .checked_mul(8)
                    .ok_or(FormatError::UnexpectedEof)?;
                if cur.remaining() < need {
                    return Err(FormatError::UnexpectedEof);
                }
                let mut intervals = Vec::with_capacity(interval_count);
                for _ in 0..interval_count {
                    intervals.push((cur.read_f32()?, cur.read_f32()?));
                }
                marks.push(MotionMark {
                    name: mark_name,
                    intervals,
                });
            }
        }

        defs.push(MotionDef {
            name,
            bone_or_part,
            motion,
            speed,
            power,
            accrue,
            falloff,
            flags,
            marks,
        });
    }

    Ok((version, partitions, defs))
}

fn parse_motions_chunk(ms_bytes: &[u8], bone_count: usize) -> Result<Vec<MotionClip>, FormatError> {
    let reader = ChunkReader::new(ms_bytes);
    let cnt_bytes = reader.require_chunk(0)?;
    let mut cnt_cur = ByteCursor::new(&cnt_bytes);
    let clip_count = cnt_cur.read_u32()?;
    if clip_count >= 0x3FFF {
        return Err(FormatError::InvalidValue(
            "OMF motion count exceeds 14-bit MotionID limit",
        ));
    }

    let mut clips = Vec::with_capacity(clip_count as usize);
    for m_idx in 0..clip_count {
        let chunk_bytes = reader.require_chunk(m_idx + 1)?;
        let mut cur = ByteCursor::new(&chunk_bytes);
        let name = cur.read_zstring()?.to_ascii_lowercase();
        let frame_count = cur.read_u32()?;
        if frame_count == 0 {
            return Err(FormatError::InvalidValue("OMF motion frame count must be > 0"));
        }

        let mut tracks = Vec::with_capacity(bone_count);
        for _ in 0..bone_count {
            let flags = cur.read_u8()?;

            let (rot_crc32, rot_keys) = if (flags & motion_key_flags::FL_RKEY_ABSENT) != 0 {
                let k = QuantizedRotKey {
                    x: cur.read_i16()?,
                    y: cur.read_i16()?,
                    z: cur.read_i16()?,
                    w: cur.read_i16()?,
                };
                (None, vec![k])
            } else {
                let crc = cur.read_u32()?;
                let n = frame_count as usize;
                let need = n.checked_mul(8).ok_or(FormatError::UnexpectedEof)?;
                if cur.remaining() < need {
                    return Err(FormatError::UnexpectedEof);
                }
                let mut keys = Vec::with_capacity(n);
                for _ in 0..n {
                    keys.push(QuantizedRotKey {
                        x: cur.read_i16()?,
                        y: cur.read_i16()?,
                        z: cur.read_i16()?,
                        w: cur.read_i16()?,
                    });
                }
                (Some(crc), keys)
            };

            let translation = if (flags & motion_key_flags::FL_TKEY_PRESENT) != 0 {
                let crc32 = cur.read_u32()?;
                let n = frame_count as usize;
                if (flags & motion_key_flags::FL_TKEY_16_IS_BIT) != 0 {
                    let need = n.checked_mul(6).ok_or(FormatError::UnexpectedEof)?;
                    if cur.remaining() < need {
                        return Err(FormatError::UnexpectedEof);
                    }
                    let mut keys = Vec::with_capacity(n);
                    for _ in 0..n {
                        keys.push([cur.read_i16()?, cur.read_i16()?, cur.read_i16()?]);
                    }
                    let size_t = cur.read_f32_3()?;
                    let init_t = cur.read_f32_3()?;
                    BoneTranslationKeys::Quantized16 {
                        crc32,
                        keys,
                        size_t,
                        init_t,
                    }
                } else {
                    let need = n.checked_mul(3).ok_or(FormatError::UnexpectedEof)?;
                    if cur.remaining() < need {
                        return Err(FormatError::UnexpectedEof);
                    }
                    let mut keys = Vec::with_capacity(n);
                    for _ in 0..n {
                        keys.push([cur.read_i8()?, cur.read_i8()?, cur.read_i8()?]);
                    }
                    let size_t = cur.read_f32_3()?;
                    let init_t = cur.read_f32_3()?;
                    BoneTranslationKeys::Quantized8 {
                        crc32,
                        keys,
                        size_t,
                        init_t,
                    }
                }
            } else {
                let init_t = cur.read_f32_3()?;
                BoneTranslationKeys::Static { init_t }
            };

            tracks.push(BoneMotionTrack {
                flags,
                frame_count,
                rot_crc32,
                rot_keys,
                translation,
            });
        }

        clips.push(MotionClip {
            name,
            frame_count,
            tracks,
        });
    }

    Ok(clips)
}

#[inline]
fn normalize_quat_xyzw(q: [f32; 4]) -> [f32; 4] {
    let mag_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if mag_sq <= 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = 1.0 / mag_sq.sqrt();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

/// Shortest-arc quaternion SLERP (`[x, y, z, w]`).
pub fn slerp_quat_xyzw(q0: [f32; 4], q1: [f32; 4], tm: f32) -> [f32; 4] {
    let mut cosom = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    let sign = if cosom < 0.0 {
        cosom = -cosom;
        -1.0
    } else {
        1.0
    };

    let (scale0, scale1) = if (1.0 - cosom) > 1e-5 {
        let omega = cosom.clamp(-1.0, 1.0).acos();
        let sinom = omega.sin();
        if sinom.abs() > 1e-6 {
            let i_sinom = 1.0 / sinom;
            let t_omega = tm * omega;
            (
                (omega - t_omega).sin() * i_sinom,
                t_omega.sin() * i_sinom * sign,
            )
        } else {
            (1.0 - tm, tm * sign)
        }
    } else {
        (1.0 - tm, tm * sign)
    };

    [
        scale0 * q0[0] + scale1 * q1[0],
        scale0 * q0[1] + scale1 * q1[1],
        scale0 * q0[2] + scale1 * q1[2],
        scale0 * q0[3] + scale1 * q1[3],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use crate::ogf::ogf_chunks::{OGF_HEADER, OGF_S_BONE_NAMES, OGF_S_IKDATA};
    use crate::ogf::{OgfModel, OgfObb, OgfVisualType};
    use std::f32::consts::FRAC_PI_2;

    fn write_f32_3(out: &mut Vec<u8>, v: [f32; 3]) {
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }

    #[test]
    fn test_omf_v4_compressed_motions_dequantize_and_marks() {
        // Build OGF_S_SMPARAMS v4 with 2 bones (root + arm) and 1 clip with motion marks
        let mut mp = Vec::new();
        mp.extend_from_slice(&4u16.to_le_bytes()); // vers = 4
        mp.extend_from_slice(&1u16.to_le_bytes()); // 1 partition
        mp.extend_from_slice(b"default\0");
        mp.extend_from_slice(&2u16.to_le_bytes()); // 2 bones in partition
        // Remap order intentionally reversed relative to skeleton order:
        mp.extend_from_slice(b"bip01_spine\0");
        mp.extend_from_slice(&0u32.to_le_bytes()); // track 0 -> bip01_spine
        mp.extend_from_slice(b"bip01_pelvis\0");
        mp.extend_from_slice(&1u32.to_le_bytes()); // track 1 -> bip01_pelvis

        mp.extend_from_slice(&1u16.to_le_bytes()); // 1 motion def
        mp.extend_from_slice(b"idle_0\0");
        mp.extend_from_slice(&(motion_def_flags::ESM_IDLE as u32).to_le_bytes());
        mp.extend_from_slice(&0u16.to_le_bytes()); // bone_or_part
        mp.extend_from_slice(&0u16.to_le_bytes()); // motion index
        mp.extend_from_slice(&1.5f32.to_le_bytes()); // speed
        mp.extend_from_slice(&1.0f32.to_le_bytes()); // power
        mp.extend_from_slice(&2.0f32.to_le_bytes()); // accrue
        mp.extend_from_slice(&1.0f32.to_le_bytes()); // falloff
        // v4 motion marks (1 mark with CRLF IWriter::w_string format)
        mp.extend_from_slice(&1u32.to_le_bytes());
        mp.extend_from_slice(b"RightHand\r\n");
        mp.extend_from_slice(&1u32.to_le_bytes());
        mp.extend_from_slice(&0.10f32.to_le_bytes());
        mp.extend_from_slice(&0.45f32.to_le_bytes());

        // Build OGF_S_MOTIONS with 2-frame animation:
        // - Track 0 (bip01_spine): R present (frame 0 identity, frame 1 +90deg Z), T 16-bit present
        // - Track 1 (bip01_pelvis): R absent (static identity), T static
        let mut clip_payload = Vec::new();
        clip_payload.extend_from_slice(b"idle_0\0");
        clip_payload.extend_from_slice(&2u32.to_le_bytes()); // 2 frames

        // Track 0: FL_TKEY_PRESENT | FL_TKEY_16_IS_BIT
        clip_payload.push(motion_key_flags::FL_TKEY_PRESENT | motion_key_flags::FL_TKEY_16_IS_BIT);
        clip_payload.extend_from_slice(&0x1122_3344u32.to_le_bytes()); // rot crc32
        // Frame 0: identity quat (0, 0, 0, 32767)
        for v in [0i16, 0, 0, 32767] {
            clip_payload.extend_from_slice(&v.to_le_bytes());
        }
        // Frame 1: 90 deg around Z -> sin(pi/4)=cos(pi/4)=0.7071068 -> ~23170
        for v in [0i16, 0, 23170, 23170] {
            clip_payload.extend_from_slice(&v.to_le_bytes());
        }
        clip_payload.extend_from_slice(&0x5566_7788u32.to_le_bytes()); // trans crc32
        // T16 frame 0: [0, 0, 0] -> init_t = [1, 2, 3]
        for v in [0i16, 0, 0] {
            clip_payload.extend_from_slice(&v.to_le_bytes());
        }
        // T16 frame 1: [100, 200, 0] with size_t = [0.01, 0.02, 0.0] -> +[1, 4, 0] => [2, 6, 3]
        for v in [100i16, 200, 0] {
            clip_payload.extend_from_slice(&v.to_le_bytes());
        }
        write_f32_3(&mut clip_payload, [0.01, 0.02, 0.0]); // _sizeT FIRST
        write_f32_3(&mut clip_payload, [1.0, 2.0, 3.0]); // _initT SECOND

        // Track 1: FL_RKEY_ABSENT + static init_t = [0, 1.5, 0]
        clip_payload.push(motion_key_flags::FL_RKEY_ABSENT);
        for v in [0i16, 0, 0, 32767] {
            clip_payload.extend_from_slice(&v.to_le_bytes());
        }
        write_f32_3(&mut clip_payload, [0.0, 1.5, 0.0]);

        let mut ms_inner = ChunkWriter::new();
        ms_inner.write_chunk(0, &1u32.to_le_bytes());
        ms_inner.write_compressed_chunk(1, &clip_payload);

        let mut omf_writer = ChunkWriter::new();
        omf_writer.write_compressed_chunk(OGF_S_SMPARAMS, &mp);
        omf_writer.write_chunk(OGF_S_MOTIONS, &ms_inner.into_bytes());
        let omf_bytes = omf_writer.into_bytes();

        let omf = OmfMotions::parse(&omf_bytes).expect("OMF must parse cleanly");
        assert_eq!(omf.sm_params_version, 4);
        assert_eq!(omf.partition_bone_count(), 2);
        let (_, def) = omf.find_def("IDLE_0").unwrap();
        assert!((def.quantized_speed() - 1.5).abs() < 0.01);
        assert_eq!(def.marks.len(), 1);
        assert_eq!(def.marks[0].name, "RightHand");
        assert!(def.marks[0].pick_mark(0.25).is_some());

        let clip = omf.find_clip("idle_0").unwrap();
        assert_eq!(clip.frame_count, 2);
        // Sample track 0 halfway between frame 0 and frame 1 (0.5 / 30s)
        let mid_key = clip.tracks[0].sample_at_time(0.5 * SAMPLE_SPF);
        assert!((mid_key.translation[0] - 1.5).abs() < 1e-4);
        assert!((mid_key.translation[1] - 4.0).abs() < 1e-4);
        assert!((mid_key.translation[2] - 3.0).abs() < 1e-4);
        // Halfway slerp between 0 and pi/2 around Z => angle = pi/4 => z ~ sin(pi/8) = 0.38268
        assert!((mid_key.rotation_xyzw[2] - (FRAC_PI_2 * 0.25).sin()).abs() < 1e-3);

        // Build matching OGF skeleton (bip01_pelvis -> bip01_spine) and verify pose evaluation remap
        let mut bone_names = Vec::new();
        bone_names.extend_from_slice(&2u32.to_le_bytes());
        let id_obb = OgfObb {
            rotate: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translate: [0.0, 0.0, 0.0],
            halfsize: [0.5, 0.5, 0.5],
        };
        let write_obb = |out: &mut Vec<u8>, o: &OgfObb| {
            for r in o.rotate {
                write_f32_3(out, r);
            }
            write_f32_3(out, o.translate);
            write_f32_3(out, o.halfsize);
        };
        bone_names.extend_from_slice(b"bip01_pelvis\0\0");
        write_obb(&mut bone_names, &id_obb);
        bone_names.extend_from_slice(b"bip01_spine\0bip01_pelvis\0");
        write_obb(&mut bone_names, &id_obb);

        let mut ogf_w = ChunkWriter::new();
        let mut hdr = vec![4u8, OgfVisualType::SkeletonAnim as u8, 0, 0];
        hdr.extend_from_slice(&[0u8; 40]);
        ogf_w.write_chunk(OGF_HEADER, &hdr);
        ogf_w.write_chunk(OGF_S_BONE_NAMES, &bone_names);
        // Unused IK chunk not needed for hierarchy-only mapping check
        let _ = OGF_S_IKDATA;
        let model = OgfModel::parse(&ogf_w.into_bytes()).unwrap();
        let skel = model.skeleton.unwrap();

        let world_mats = omf
            .evaluate_skeleton_pose(&skel, 0, SAMPLE_SPF)
            .expect("pose evaluation must remap OMF tracks onto OGF skeleton");
        // pelvis local Y = 1.5; spine frame 1 local = [2, 6, 3] => spine world = [2, 7.5, 3]
        assert!((world_mats[0][3][1] - 1.5).abs() < 1e-4);
        assert!((world_mats[1][3][0] - 2.0).abs() < 1e-4);
        assert!((world_mats[1][3][1] - 7.5).abs() < 1e-4);
        assert!((world_mats[1][3][2] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_omf_8bit_translation_and_empty_mark_line() {
        let mut mp = Vec::new();
        mp.extend_from_slice(&4u16.to_le_bytes());
        mp.extend_from_slice(&1u16.to_le_bytes());
        mp.extend_from_slice(b"all\0");
        mp.extend_from_slice(&1u16.to_le_bytes());
        mp.extend_from_slice(b"root\0");
        mp.extend_from_slice(&0u32.to_le_bytes());
        mp.extend_from_slice(&1u16.to_le_bytes());
        mp.extend_from_slice(b"walk\0");
        mp.extend_from_slice(&(motion_def_flags::ESM_STOP_AT_END as u32).to_le_bytes());
        mp.extend_from_slice(&0u16.to_le_bytes());
        mp.extend_from_slice(&0u16.to_le_bytes());
        mp.extend_from_slice(&1.0f32.to_le_bytes());
        mp.extend_from_slice(&1.0f32.to_le_bytes());
        mp.extend_from_slice(&0.5f32.to_le_bytes());
        mp.extend_from_slice(&0.8f32.to_le_bytes()); // >= accrue on cycle -> clamped < accrue
        mp.extend_from_slice(&1u32.to_le_bytes());
        mp.extend_from_slice(b"\r\n"); // empty line mark
        mp.extend_from_slice(&0u32.to_le_bytes());

        let mut clip = Vec::new();
        clip.extend_from_slice(b"walk\0");
        clip.extend_from_slice(&2u32.to_le_bytes());
        clip.push(motion_key_flags::FL_TKEY_PRESENT | motion_key_flags::FL_RKEY_ABSENT);
        for v in [0i16, 0, 0, 32767] {
            clip.extend_from_slice(&v.to_le_bytes());
        }
        clip.extend_from_slice(&0xCAFE_BABEu32.to_le_bytes());
        clip.extend_from_slice(&[-10i8 as u8, 0u8, 10u8]);
        clip.extend_from_slice(&[20i8 as u8, 5u8, 10u8]);
        write_f32_3(&mut clip, [0.5, 1.0, 2.0]);
        write_f32_3(&mut clip, [5.0, 0.0, 0.0]);

        let mut ms = ChunkWriter::new();
        ms.write_chunk(0, &1u32.to_le_bytes());
        ms.write_chunk(1, &clip);
        let mut omf_w = ChunkWriter::new();
        omf_w.write_chunk(OGF_S_SMPARAMS, &mp);
        omf_w.write_chunk(OGF_S_MOTIONS, &ms.into_bytes());

        let omf = OmfMotions::parse(&omf_w.into_bytes()).unwrap();
        assert!(omf.defs[0].stop_at_end());
        assert!(omf.defs[0].effective_falloff() < omf.defs[0].effective_accrue());
        assert_eq!(omf.defs[0].marks[0].name, "");

        let k1 = omf.clips[0].tracks[0].sample_at_frame(1);
        assert_eq!(k1.translation, [15.0, 5.0, 20.0]);

        // Verify BoneKeyframe::to_matrix matches X-Ray `vector.h::mk_xform` & `matrix_set_hpb`:
        // +90deg bank (roll around +Z axis): q = [0, 0, sin(pi/4), cos(pi/4)]
        let qz90 = BoneKeyframe {
            rotation_xyzw: [0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2],
            translation: [3.0, 4.0, 5.0],
        };
        let mut expected = crate::ogf::matrix_set_hpb(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        expected[3][0] = 3.0;
        expected[3][1] = 4.0;
        expected[3][2] = 5.0;
        let actual = qz90.to_matrix();
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    (actual[r][c] - expected[r][c]).abs() < 1e-5,
                    "mismatch at [{r}][{c}]: actual={} vs expected={}",
                    actual[r][c],
                    expected[r][c]
                );
            }
        }
    }
}
