//! # xray-vfs
//!
//! High-performance pure Rust Virtual File System (VFS) and archive decompressor
//! for the X-Ray Monolith Engine (`.db` and `.xdb` formats).
//! Supports native Win32 memory-mapped zero-copy archives, Chunk 666 header resolution,
//! priority `gamedata/` -> `.xdb*` -> `.db*` layering, `$game_*$` path aliases,
//! bounds-checked LZO1X + LZHuf decompression, and parallel multi-threaded extraction.

pub mod archive;
pub mod lzhuf;
pub mod lzo;
pub mod mmap;
pub mod vfs;

pub use archive::{
    compute_crc32, ArchiveBacking, ArchiveEntry, ArchiveHeader, MountedArchive, VfsError,
    XRayArchive, CFS_COMPRESS_MARK, CFS_HEADER_CHUNK_ID,
};
pub use lzhuf::{compress_lzhuf, decompress_lzhuf, decompress_lzhuf_with_limit, LzhufError};
pub use lzo::{compress_lzo1x, decompress_lzo1x, LzoError};
pub use mmap::MappedFile;
pub use vfs::{normalize_virtual_path, normalize_virtual_prefix, VfsSource, VirtualFileSystem};
