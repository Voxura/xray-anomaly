//! # xray-vfs
//!
//! High-performance pure Rust Virtual File System (VFS) and archive decompressor
//! for the X-Ray Monolith Engine (`.db` and `.xdb` formats).
//! Supports memory-mapped zero-copy access, bounds-checked LZO1X decompression,
//! and parallel multi-threaded file extraction.

pub mod archive;
pub mod lzo;

pub use archive::{ArchiveEntry, XRayArchive, VfsError};
pub use lzo::{decompress_lzo1x, LzoError};
