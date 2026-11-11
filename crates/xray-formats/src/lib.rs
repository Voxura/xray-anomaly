//! # xray-formats
//!
//! Pure Rust X-Ray Monolith binary asset & model format parsers:
//! - `chunk`: Zero-copy X-Ray Chunked File Format reader (`ID + size + payload` + `CFS_CompressMark` LZHuf)
//! - `ogf`: X-Ray OGF v4 static, progressive, hierarchy, and skeletal mesh + bone/IK parser
//! - `vfs_bridge`: Direct `xray-vfs` streaming integration for `.ogf` models & `#include`/DLTX `.ltx` trees

pub mod chunk;
pub mod ogf;
pub mod vfs_bridge;

pub use chunk::{ByteCursor, ChunkEntry, ChunkIter, ChunkReader, ChunkWriter, FormatError};
pub use ogf::{
    fvf_flags, matrix_invert_43, matrix_mul_43, matrix_set_hpb, matrix_set_xyzi_translate,
    matrix_transform_point, ogf_chunks, skel_vert_type, BoneIkData, BoneIkSection,
    BonePoseMatrices, BoneShape, BoneShapeType, OgfBBox, OgfBSphere, OgfBone, OgfDesc,
    OgfGContainerRef, OgfHeader, OgfModel, OgfObb, OgfSkeleton, OgfSlideWindows,
    OgfTextureLink, OgfVertices, OgfVisualType, SlideWindow, StaticVertex, StaticVertexLayout,
    UnifiedSkinnedVertex, VertBoned1W, VertBoned2W, VertBoned3W, VertBoned4W, XRayMatrix,
};
pub use vfs_bridge::{load_ltx_tree_from_vfs, load_ogf_from_vfs, AssetLoadError, VfsLtxProvider};
