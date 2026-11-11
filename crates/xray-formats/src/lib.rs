//! # xray-formats
//!
//! Pure Rust X-Ray Monolith binary asset & model format parsers:
//! - `chunk`: Zero-copy X-Ray Chunked File Format reader (`ID + size + payload` + `CFS_CompressMark` LZHuf)
//! - `ogf`: X-Ray OGF v4 static, progressive, hierarchy, and skeletal mesh + bone/IK parser
//! - `omf`: X-Ray `.omf` / embedded `OGF_S_MOTIONS` skeletal animation clips, partitions & marks parser
//! - `thm`: X-Ray `.thm` / `.ltx` texture descriptor parser (`_bump` / `_bump#` / detail / material links)
//! - `gamemtl`: X-Ray `gamemtl.xr` physical/surface material & interaction pair library parser
//! - `vfs_bridge`: Direct `xray-vfs` streaming integration for `.ogf`, `.omf`, `.thm`, `gamemtl.xr` & DLTX `.ltx` trees

pub mod chunk;
pub mod gamemtl;
pub mod ogf;
pub mod omf;
pub mod thm;
pub mod vfs_bridge;

pub use chunk::{ByteCursor, ChunkEntry, ChunkIter, ChunkReader, ChunkWriter, FormatError};
pub use gamemtl::{
    gamemtl_chunks, gamemtl_flags, gamemtl_pair_chunks, gamemtl_pair_flags,
    gamemtl_root_chunks, GameMaterial, GameMaterialLibrary, GameMaterialPair,
    GAMEMTL_CURRENT_VERSION, GAMEMTL_NONE_ID,
};
pub use ogf::{
    fvf_flags, matrix_invert_43, matrix_mul_43, matrix_set_hpb, matrix_set_xyzi_translate,
    matrix_transform_point, ogf_chunks, skel_vert_type, BoneIkData, BoneIkSection,
    BonePoseMatrices, BoneShape, BoneShapeType, OgfBBox, OgfBSphere, OgfBone, OgfDesc,
    OgfGContainerRef, OgfHeader, OgfModel, OgfObb, OgfSkeleton, OgfSlideWindows,
    OgfTextureLink, OgfVertices, OgfVisualType, SlideWindow, StaticVertex, StaticVertexLayout,
    UnifiedSkinnedVertex, VertBoned1W, VertBoned2W, VertBoned3W, VertBoned4W, XRayMatrix,
};
pub use omf::{
    motion_def_flags, motion_key_flags, BoneKeyframe, BoneMotionTrack, BoneTranslationKeys,
    MotionClip, MotionDef, MotionMark, MotionPartition, MotionPartitionBone, OmfMotions,
    QuantizedRotKey, KEY_QUANT, KEY_QUANT_INV, MOTION_QUANTIZER_RANGE_EXT, SAMPLE_FPS,
    SAMPLE_SPF, XR_OGF_SM_PARAMS_VERSION,
};
pub use thm::{
    strip_texture_extension, texture_flags, texture_stem_to_dds_vpath, texture_stem_to_thm_vpath,
    thm_chunks, BumpTextureLinks, ResolvedTextureLinks, TextureBumpMode, TextureFormat,
    TextureMaterial, TextureParams, TextureType,
};
pub use vfs_bridge::{
    load_gamemtl_from_vfs, load_ltx_tree_from_vfs, load_ogf_from_vfs, load_omf_from_vfs,
    load_thm_from_vfs, resolve_ogf_texture_links, AssetLoadError, VfsLtxProvider,
};
