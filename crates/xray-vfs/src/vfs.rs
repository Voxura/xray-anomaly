//! Priority-Layered Virtual File System (VFS) & Parallel Batch Streamer
//!
//! Mirrors X-Ray Anomaly's `CLocatorAPI` resolution order:
//! 1. Loose `gamedata/` directories on disk (highest priority, last mounted wins)
//! 2. Patch / Mod archives (`.xdb*`, sorted lexicographically, later wins)
//! 3. Base game archives (`.db*`, sorted lexicographically, later wins)
//!
//! Archive reads stay zero-copy on top of Win32 memory-mapped `MappedFile`s and
//! distribute parallel LZO1X decompression + CRC32 verification across worker threads
//! via `std::thread::scope`.

use std::collections::{BTreeSet, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use crate::archive::{MountedArchive, VfsError};

const ALIAS_MAP: &[(&str, &str)] = &[
    ("$fs_root$", ""),
    ("$game_data$", ""),
    ("$game_configs$", "configs\\"),
    ("$game_config$", "configs\\"),
    ("$game_textures$", "textures\\"),
    ("$game_meshes$", "meshes\\"),
    ("$game_shaders$", "shaders\\"),
    ("$game_sounds$", "sounds\\"),
    ("$game_scripts$", "scripts\\"),
    ("$game_levels$", "levels\\"),
    ("$game_spawns$", "spawns\\"),
    ("$game_anims$", "anims\\"),
];

/// Source backing a resolved virtual path lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VfsSource {
    LooseFile(PathBuf),
    Archive { archive_name: String, entry_name: String },
}

/// Normalize an X-Ray engine path or `$game_*$` alias into canonical lowercase segments.
fn normalize_virtual_segments(raw: &str) -> Result<Vec<String>, VfsError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(VfsError::InvalidPath("empty path".to_string()));
    }

    let mut work = trimmed.replace('/', "\\");
    // Strip optional leading $alias$
    if work.starts_with('$') {
        if let Some(end_dollar) = work[1..].find('$') {
            let alias = &work[..end_dollar + 2];
            let lower_alias = alias.to_ascii_lowercase();
            let rest = work[end_dollar + 2..].trim_start_matches('\\');
            if let Some(&(_, mapped)) = ALIAS_MAP.iter().find(|&&(k, _)| k == lower_alias) {
                work = if mapped.is_empty() {
                    rest.to_string()
                } else if rest.is_empty() {
                    mapped.trim_end_matches('\\').to_string()
                } else {
                    format!("{mapped}{rest}")
                };
            } else {
                return Err(VfsError::InvalidPath(format!(
                    "unknown X-Ray path alias '{alias}'"
                )));
            }
        } else {
            return Err(VfsError::InvalidPath(format!(
                "unterminated path alias in '{raw}'"
            )));
        }
    }

    let mut segments: Vec<String> = Vec::new();
    for part in work.split('\\') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err(VfsError::InvalidPath(format!(
                "path traversal '..' is not allowed: '{raw}'"
            )));
        }
        segments.push(part.to_ascii_lowercase());
    }

    // Strip leading "gamedata" if someone passes "gamedata/configs/system.ltx"
    if segments.first().is_some_and(|s| s == "gamedata") {
        segments.remove(0);
    }

    Ok(segments)
}

/// Normalize an X-Ray engine file path or `$game_*$` alias into a canonical gamedata-relative
/// lowercase key separated by backslashes (`\`): e.g. `textures\ui\cursor_cross.dds`.
pub fn normalize_virtual_path(raw: &str) -> Result<String, VfsError> {
    let segments = normalize_virtual_segments(raw)?;
    if segments.is_empty() {
        return Err(VfsError::InvalidPath(format!(
            "path '{raw}' resolves to empty virtual path"
        )));
    }
    Ok(segments.join("\\"))
}

/// Normalize an X-Ray directory prefix (allowing `""`, `"."`, `"gamedata"`, or `$game_data$` as root prefix `""`,
/// and otherwise returning `"dir\\"` with a trailing backslash).
pub fn normalize_virtual_prefix(raw: &str) -> Result<String, VfsError> {
    if raw.trim().is_empty() {
        return Ok(String::new());
    }
    let segments = normalize_virtual_segments(raw)?;
    if segments.is_empty() {
        Ok(String::new())
    } else {
        Ok(format!("{}\\", segments.join("\\")))
    }
}

/// Thread-safe, priority-layered X-Ray Virtual File System.
pub struct VirtualFileSystem {
    /// Loose `gamedata/` directories searched in reverse order (last mounted wins).
    loose_dirs: Vec<PathBuf>,
    /// Mounted `.db`/`.xdb` archives searched in reverse order (last mounted wins).
    archives: Vec<Arc<MountedArchive>>,
}

impl Default for VirtualFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualFileSystem {
    pub fn new() -> Self {
        Self {
            loose_dirs: Vec::new(),
            archives: Vec::new(),
        }
    }

    /// Mount an Anomaly root directory: scans `.db*` (root & `db/`), `.xdb*` (root & `mods/`),
    /// and loose `gamedata/` override directory cleanly in X-Ray priority order.
    pub fn mount_game_root(root_dir: impl AsRef<Path>) -> Result<Self, VfsError> {
        let root = root_dir.as_ref();
        let mut vfs = Self::new();

        let mut db_paths = Vec::new();
        collect_archives_with_ext(root, "db", &mut db_paths)?;
        collect_archives_with_ext(&root.join("db"), "db", &mut db_paths)?;
        db_paths.sort();
        for p in db_paths {
            let mounted = MountedArchive::open_mmap(&p)?;
            if mounted.archive.header.auto_load {
                vfs.archives.push(Arc::new(mounted));
            }
        }

        let mut xdb_paths = Vec::new();
        collect_archives_with_ext(root, "xdb", &mut xdb_paths)?;
        collect_archives_with_ext(&root.join("mods"), "xdb", &mut xdb_paths)?;
        xdb_paths.sort();
        for p in xdb_paths {
            let mounted = MountedArchive::open_mmap(&p)?;
            if mounted.archive.header.auto_load {
                vfs.archives.push(Arc::new(mounted));
            }
        }

        let gamedata = root.join("gamedata");
        if gamedata.is_dir() {
            vfs.mount_loose_dir(gamedata);
        }

        Ok(vfs)
    }

    /// Mount a loose `gamedata/` directory. Loose mounts always override archives,
    /// and later loose mounts override earlier ones.
    pub fn mount_loose_dir(&mut self, dir: impl AsRef<Path>) {
        self.loose_dirs.push(dir.as_ref().to_path_buf());
    }

    /// Mount a zero-copy memory-mapped `.db`/`.xdb` archive file (higher priority than earlier archives).
    pub fn mount_archive_mmap(&mut self, path: impl AsRef<Path>) -> Result<(), VfsError> {
        let mounted = MountedArchive::open_mmap(path)?;
        self.archives.push(Arc::new(mounted));
        Ok(())
    }

    /// Mount an in-memory archive byte buffer (higher priority than earlier archives).
    pub fn mount_archive_bytes(
        &mut self,
        name: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Result<(), VfsError> {
        let mounted = MountedArchive::from_bytes(name, bytes)?;
        self.archives.push(Arc::new(mounted));
        Ok(())
    }

    #[inline]
    pub fn loose_mount_count(&self) -> usize {
        self.loose_dirs.len()
    }

    #[inline]
    pub fn archive_count(&self) -> usize {
        self.archives.len()
    }

    /// Total unique indexed entries across mounted archives (loose directory files are resolved dynamically).
    pub fn archive_entry_count(&self) -> usize {
        let mut keys = HashSet::new();
        for arch in &self.archives {
            for k in arch.archive.entries.keys() {
                keys.insert(k.as_str());
            }
        }
        keys.len()
    }

    /// Resolve which layer (`LooseFile` or `Archive`) wins for `virtual_path`.
    pub fn resolve(&self, virtual_path: &str) -> Result<VfsSource, VfsError> {
        let norm = normalize_virtual_path(virtual_path)?;
        self.resolve_normalized(&norm).ok_or(VfsError::FileNotFound)
    }

    fn resolve_normalized(&self, norm: &str) -> Option<VfsSource> {
        let rel_fs = norm.replace('\\', std::path::MAIN_SEPARATOR_STR);
        for root in self.loose_dirs.iter().rev() {
            let candidate = root.join(&rel_fs);
            if is_clean_subpath(&rel_fs) && candidate.is_file() {
                return Some(VfsSource::LooseFile(candidate));
            }
        }

        for arch in self.archives.iter().rev() {
            if let Some(entry) = arch.archive.entries.get(norm) {
                return Some(VfsSource::Archive {
                    archive_name: arch.name.clone(),
                    entry_name: entry.name.clone(),
                });
            }
        }

        None
    }

    /// Check whether a virtual path exists in any mounted layer.
    pub fn exists(&self, virtual_path: &str) -> bool {
        self.resolve(virtual_path).is_ok()
    }

    /// Query uncompressed size in bytes without reading/decompressing the payload.
    pub fn file_size(&self, virtual_path: &str) -> Result<u64, VfsError> {
        let norm = normalize_virtual_path(virtual_path)?;
        let rel_fs = norm.replace('\\', std::path::MAIN_SEPARATOR_STR);

        if is_clean_subpath(&rel_fs) {
            for root in self.loose_dirs.iter().rev() {
                let candidate = root.join(&rel_fs);
                if let Ok(meta) = std::fs::metadata(&candidate)
                    && meta.is_file()
                {
                    return Ok(meta.len());
                }
            }
        }

        for arch in self.archives.iter().rev() {
            if let Some(entry) = arch.archive.entries.get(&norm) {
                return Ok(entry.uncompressed_size as u64);
            }
        }

        Err(VfsError::FileNotFound)
    }

    /// Read and verify/decompress a single virtual file.
    pub fn read_bytes(&self, virtual_path: &str) -> Result<Vec<u8>, VfsError> {
        let norm = normalize_virtual_path(virtual_path)?;
        let rel_fs = norm.replace('\\', std::path::MAIN_SEPARATOR_STR);

        if is_clean_subpath(&rel_fs) {
            for root in self.loose_dirs.iter().rev() {
                let candidate = root.join(&rel_fs);
                if candidate.is_file() {
                    return Ok(std::fs::read(&candidate)?);
                }
            }
        }

        for arch in self.archives.iter().rev() {
            if let Some(entry) = arch.archive.entries.get(&norm) {
                return arch.read_entry(entry);
            }
        }

        Err(VfsError::FileNotFound)
    }

    /// Read a virtual file as UTF-8 text (lossy replacement on legacy Windows-1251 comments).
    pub fn read_to_string_lossy(&self, virtual_path: &str) -> Result<String, VfsError> {
        let bytes = self.read_bytes(virtual_path)?;
        let s = match String::from_utf8(bytes) {
            Ok(valid) => valid,
            Err(err) => String::from_utf8_lossy(err.as_bytes()).into_owned(),
        };
        Ok(s)
    }

    /// List unique sorted virtual paths under an optional normalized prefix (`""`, `"gamedata"`, or `"$game_data$"` lists everything).
    pub fn list_files(&self, prefix: &str) -> Result<Vec<String>, VfsError> {
        let norm_prefix = normalize_virtual_prefix(prefix)?;

        let mut results = BTreeSet::new();

        for arch in &self.archives {
            for key in arch.archive.entries.keys() {
                if norm_prefix.is_empty() || key.starts_with(&norm_prefix) {
                    results.insert(key.clone());
                }
            }
        }

        for root in &self.loose_dirs {
            walk_loose_dir(root, root, &norm_prefix, &mut results)?;
        }

        Ok(results.into_iter().collect())
    }

    /// Stream and verify/decompress a batch of virtual paths in parallel across worker threads.
    ///
    /// Preserves request order in the returned vector. Safe without locking because mounted
    /// archives are immutable read-only memory mappings after mount time.
    pub fn read_batch<'a>(&self, paths: &[&'a str]) -> Vec<(&'a str, Result<Vec<u8>, VfsError>)> {
        if paths.is_empty() {
            return Vec::new();
        }
        if paths.len() <= 2 {
            return paths.iter().map(|&p| (p, self.read_bytes(p))).collect();
        }

        let max_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, paths.len());
        let chunk_size = paths.len().div_ceil(max_workers);

        let chunk_results: Vec<Vec<(&'a str, Result<Vec<u8>, VfsError>)>> =
            std::thread::scope(|scope| {
                let mut handles = Vec::new();
                for slice in paths.chunks(chunk_size) {
                    handles.push(scope.spawn(move || {
                        slice.iter().map(|&p| (p, self.read_bytes(p))).collect()
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("VFS batch worker panicked"))
                    .collect()
            });

        chunk_results.into_iter().flatten().collect()
    }
}

fn is_clean_subpath(rel: &str) -> bool {
    Path::new(rel)
        .components()
        .all(|c| matches!(c, Component::Normal(_)))
}

fn collect_archives_with_ext(
    dir: &Path,
    ext_prefix: &str,
    out: &mut Vec<PathBuf>,
) -> Result<(), VfsError> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            let lower = ext.to_ascii_lowercase();
            // Matches `.db`, `.db0`, `.dba` or `.xdb`, `.xdb0` etc.
            if lower.starts_with(ext_prefix) {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn walk_loose_dir(
    root: &Path,
    dir: &Path,
    norm_prefix: &str,
    out: &mut BTreeSet<String>,
) -> Result<(), VfsError> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_loose_dir(root, &path, norm_prefix, out)?;
        } else if path.is_file()
            && let Ok(rel) = path.strip_prefix(root)
        {
            let vpath = rel
                .to_string_lossy()
                .replace('/', "\\")
                .to_ascii_lowercase();
            if norm_prefix.is_empty() || vpath.starts_with(norm_prefix) {
                out.insert(vpath);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::compute_crc32;
    use crate::lzo::compress_lzo1x;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn build_synthetic_db(entries: &[(&str, &[u8], bool)]) -> Vec<u8> {
        struct Prepared<'a> {
            name: &'a str,
            raw: &'a [u8],
            stored: Vec<u8>,
            crc: u32,
        }

        let prepared: Vec<Prepared<'_>> = entries
            .iter()
            .map(|&(name, raw, compress)| Prepared {
                name,
                raw,
                stored: if compress {
                    compress_lzo1x(raw)
                } else {
                    raw.to_vec()
                },
                crc: compute_crc32(raw),
            })
            .collect();

        // Determine exact Chunk 1 byte length first
        let mut table_len = 0usize;
        for p in &prepared {
            let rec_size = 16 + p.name.len();
            table_len += 2 + rec_size;
        }
        let mut cursor = 8 + table_len;

        let mut chunk1 = Vec::with_capacity(table_len);
        for p in &prepared {
            let rec_size = (16 + p.name.len()) as u16;
            chunk1.extend_from_slice(&rec_size.to_le_bytes());
            chunk1.extend_from_slice(&(p.raw.len() as u32).to_le_bytes());
            chunk1.extend_from_slice(&(p.stored.len() as u32).to_le_bytes());
            chunk1.extend_from_slice(&p.crc.to_le_bytes());
            chunk1.extend_from_slice(p.name.as_bytes());
            chunk1.extend_from_slice(&(cursor as u32).to_le_bytes());
            cursor += p.stored.len();
        }

        let mut out = Vec::with_capacity(cursor);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
        out.extend_from_slice(&chunk1);
        for p in &prepared {
            out.extend_from_slice(&p.stored);
        }
        out
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!("xray_vfs_test_{}_{}_{}", std::process::id(), nanos, tag));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn test_path_normalization_aliases_and_traversal_guard() {
        assert_eq!(
            normalize_virtual_path("$game_textures$/ui//cursor_cross.dds").unwrap(),
            "textures\\ui\\cursor_cross.dds"
        );
        assert_eq!(
            normalize_virtual_path("gamedata\\Configs\\System.ltx").unwrap(),
            "configs\\system.ltx"
        );
        assert_eq!(
            normalize_virtual_path("$game_meshes$/weapons/ak74.ogf").unwrap(),
            "meshes\\weapons\\ak74.ogf"
        );
        assert!(matches!(
            normalize_virtual_path("configs/../../etc/passwd"),
            Err(VfsError::InvalidPath(_))
        ));
        assert!(matches!(
            normalize_virtual_path("$unknown_alias$/foo.ltx"),
            Err(VfsError::InvalidPath(_))
        ));
    }

    #[test]
    fn test_priority_layering_and_parallel_batch_mmap() {
        let root = temp_dir("priority");
        let db_dir = root.join("db");
        let gamedata_dir = root.join("gamedata");
        std::fs::create_dir_all(&db_dir).unwrap();
        std::fs::create_dir_all(gamedata_dir.join("configs")).unwrap();

        let base_db = build_synthetic_db(&[
            ("configs\\system.ltx", b"from_base_db", false),
            ("configs\\weapons\\ak74.ltx", b"ak74_base_v1", true),
            (
                "textures\\ui\\test.dds",
                b"DDS_BASE_TEXTURE_REPEATED_REPEATED_REPEATED_REPEATED",
                true,
            ),
        ]);
        let patch_xdb = build_synthetic_db(&[
            ("configs\\system.ltx", b"from_patch_xdb", false),
            ("configs\\weapons\\ak74.ltx", b"ak74_patch_v2_rebalanced", true),
        ]);

        std::fs::write(db_dir.join("base.db0"), &base_db).unwrap();
        std::fs::write(root.join("patch.xdb0"), &patch_xdb).unwrap();
        // Loose gamedata overrides both archives for system.ltx
        std::fs::write(gamedata_dir.join("configs\\system.ltx"), b"from_loose_gamedata").unwrap();

        let vfs = VirtualFileSystem::mount_game_root(&root).expect("mount_game_root should succeed");
        assert_eq!(vfs.archive_count(), 2);
        assert_eq!(vfs.loose_mount_count(), 1);
        assert_eq!(vfs.archive_entry_count(), 3);

        // Verify override precedence
        assert_eq!(
            vfs.read_bytes("$game_configs$/system.ltx").unwrap(),
            b"from_loose_gamedata"
        );
        assert_eq!(
            vfs.read_bytes("configs/weapons/ak74.ltx").unwrap(),
            b"ak74_patch_v2_rebalanced"
        );
        assert_eq!(
            vfs.read_bytes("$game_textures$/ui/test.dds").unwrap(),
            b"DDS_BASE_TEXTURE_REPEATED_REPEATED_REPEATED_REPEATED"
        );
        assert_eq!(
            vfs.file_size("$game_configs$/weapons/ak74.ltx").unwrap(),
            b"ak74_patch_v2_rebalanced".len() as u64
        );

        // Verify source metadata & directory listing
        assert!(matches!(
            vfs.resolve("configs/system.ltx").unwrap(),
            VfsSource::LooseFile(_)
        ));
        assert!(matches!(
            vfs.resolve("textures/ui/test.dds").unwrap(),
            VfsSource::Archive { .. }
        ));
        assert_eq!(
            vfs.list_files("configs").unwrap(),
            vec![
                "configs\\system.ltx".to_string(),
                "configs\\weapons\\ak74.ltx".to_string()
            ]
        );
        assert_eq!(
            vfs.list_files("$game_configs$").unwrap(),
            vfs.list_files("configs").unwrap()
        );
        assert_eq!(vfs.list_files("$game_data$").unwrap().len(), 3);

        // Parallel batch read across loose + compressed xdb + compressed db concurrently
        let batch = vfs.read_batch(&[
            "$game_config$/system.ltx",
            "configs/weapons/ak74.ltx",
            "$game_textures$/ui/test.dds",
            "missing/nope.ogf",
        ]);
        assert_eq!(batch.len(), 4);
        assert_eq!(batch[0].1.as_ref().unwrap(), b"from_loose_gamedata");
        assert_eq!(batch[1].1.as_ref().unwrap(), b"ak74_patch_v2_rebalanced");
        assert_eq!(
            batch[2].1.as_ref().unwrap(),
            b"DDS_BASE_TEXTURE_REPEATED_REPEATED_REPEATED_REPEATED"
        );
        assert_eq!(batch[3].1, Err(VfsError::FileNotFound));

        drop(vfs);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn test_real_repo_gamedata_streaming_if_present() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let cursor_path = repo_root.join("gamedata/textures/ui/cursor_cross.dds");
        if cursor_path.is_file() {
            let vfs = VirtualFileSystem::mount_game_root(repo_root)
                .expect("mounting repo root with real gamedata/ must succeed");
            let cursor_bytes = vfs
                .read_bytes("$game_textures$/ui/cursor_cross.dds")
                .expect("must stream cursor_cross.dds from gamedata");
            assert!(cursor_bytes.starts_with(b"DDS "));
            let knife_ltx = vfs
                .read_to_string_lossy("$game_configs$/mod_system_knife.ltx")
                .expect("must stream mod_system_knife.ltx from gamedata");
            assert!(knife_ltx.contains("[wpn_knife]"));
        }
    }
}
