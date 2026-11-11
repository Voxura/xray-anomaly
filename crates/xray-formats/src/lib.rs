//! # xray-formats
//!
//! Pure Rust X-Ray Monolith binary asset & model format parsers:
//! - `cform`: `level.cform` collision database, pure-Rust AABB BVH raycast (`OPT_CULL`/`OPT_ONLYFIRST`/`OPT_ONLYNEAREST`) & SAT box queries
//! - `chunk`: Zero-copy X-Ray Chunked File Format reader (`ID + size + payload` + `CFS_CompressMark` LZHuf)
//! - `details`: `.dm` & `level.details` (`DETAIL_VERSION_3` / `DETAIL_VERSION_4`) flora/grass slots
//! - `gamemtl`: X-Ray `gamemtl.xr` physical/surface material & interaction pair library parser
//! - `level`: `level` & `level.geom`/`level.geomx` scene, visuals, sectors, portals, lights & `D3DVERTEXELEMENT9` VB/IB/SWIS decoders
//! - `ogf`: X-Ray OGF v4 static, progressive, hierarchy, and skeletal mesh + bone/IK parser
//! - `omf`: X-Ray `.omf` / embedded `OGF_S_MOTIONS` skeletal animation clips, partitions, blending & marks parser
//! - `skinning`: Pure-Rust OGF `m2b_transform` bone palette, OMF pose evaluation & `1W`–`4W` CPU skinning
//! - `thm`: X-Ray `.thm` / `.ltx` texture descriptor parser (`_bump` / `_bump#` / detail / material links)
//! - `vfs_bridge`: Direct `xray-vfs` streaming integration for `.ogf`, `.omf`, `.thm`, `gamemtl.xr`, `level*`, `.dm` & DLTX `.ltx` trees

pub mod cform;
pub mod chunk;
pub mod details;
pub mod gamemtl;
pub mod level;
pub mod ogf;
pub mod omf;
pub mod skinning;
pub mod thm;
pub mod vfs_bridge;

pub use cform::{
    ray_options, BoxQueryHit, CformBvh, CformHeader, CformModel, CformTriangle, RaycastHit,
    RaycastOptions, CFORM_CURRENT_VERSION,
};
pub use chunk::{ByteCursor, ChunkEntry, ChunkIter, ChunkReader, ChunkWriter, FormatError};
pub use details::{
    DetailHeader, DetailMesh, DetailPalette, DetailSlot, DetailVertex, LevelDetails,
    DETAIL_ID_EMPTY, DETAIL_ID_EMPTY_V3, DETAIL_SLOT_HALF_SIZE, DETAIL_SLOT_SIZE, DETAIL_VERSION_3,
    DETAIL_VERSION_4, DO_NO_WAVING,
};
pub use gamemtl::{
    gamemtl_chunks, gamemtl_flags, gamemtl_pair_chunks, gamemtl_pair_flags,
    gamemtl_root_chunks, GameMaterial, GameMaterialLibrary, GameMaterialPair,
    GAMEMTL_CURRENT_VERSION, GAMEMTL_NONE_ID,
};
pub use level::{
    d3d_decl_type, d3d_decl_usage, level_chunks, sector_chunks, LevelData, LevelGeometry,
    LevelLight, LevelPortal, LevelSector, LevelShaderRef, LevelVertexBuffer, ResolvedLevelMesh,
    VertexDeclarator, VertexElement9,
};
pub use ogf::{
    fvf_flags, matrix_invert_43, matrix_mul_43, matrix_set_hpb, matrix_set_xyzi_translate,
    matrix_transform_point, ogf_chunks, skel_vert_type, BoneIkData, BoneIkSection,
    BonePoseMatrices, BoneShape, BoneShapeType, OgfBBox, OgfBSphere, OgfBone, OgfBufferRef,
    OgfDesc, OgfGContainerRef, OgfHeader, OgfModel, OgfObb, OgfSkeleton, OgfSlideWindows,
    OgfTextureLink, OgfVertices, OgfVisualType, SlideWindow, StaticVertex, StaticVertexLayout,
    UnifiedSkinnedVertex, VertBoned1W, VertBoned2W, VertBoned3W, VertBoned4W, XRayMatrix,
    XRAY_IDENTITY_MATRIX,
};
pub use omf::{
    motion_def_flags, motion_key_flags, slerp_quat_xyzw, BoneKeyframe, BoneMotionTrack,
    BoneTranslationKeys, MotionClip, MotionDef, MotionMark, MotionPartition,
    MotionPartitionBone, OmfMotions, QuantizedRotKey, KEY_QUANT, KEY_QUANT_INV,
    MOTION_QUANTIZER_RANGE_EXT, SAMPLE_FPS, SAMPLE_SPF, XR_OGF_SM_PARAMS_VERSION,
};
pub use skinning::{
    compute_skinned_aabb, matrix_transform_dir, normalize_vec3, skin_ogf_hierarchy,
    skin_vertices, BoneMatrix4x3, BonePalette, SkinnedMeshPart, SkinnedVertex,
};
pub use thm::{
    strip_texture_extension, texture_flags, texture_stem_to_dds_vpath, texture_stem_to_thm_vpath,
    thm_chunks, BumpTextureLinks, ResolvedTextureLinks, TextureBumpMode, TextureFormat,
    TextureMaterial, TextureParams, TextureType,
};
pub use vfs_bridge::{
    load_cform_from_vfs, load_details_from_vfs, load_dm_from_vfs, load_gamemtl_from_vfs,
    load_level_from_vfs, load_level_geometry_from_vfs, load_ltx_tree_from_vfs, load_ogf_from_vfs,
    load_omf_from_vfs, load_thm_from_vfs, resolve_ogf_texture_links, AssetLoadError,
    VfsLtxProvider,
};
