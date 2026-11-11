//! X-Ray Monolith .db / .xdb Chunked Archive Reader & Parallel VFS Decompressor
//!
//! Implements chunk-based header discovery (Chunk 666), file table loading (Chunk 1)
//! including `CFS_CompressMark` (LZHuf) chunk decompression, zero-copy memory-mapped
//! archive backing, and bounds-checked LZO1X entry decompression with CRC32 verification.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::lzhuf::{decompress_lzhuf, LzhufError};
use crate::lzo::{decompress_lzo1x, LzoError};
use crate::mmap::MappedFile;

/// Chunk ID flag bit indicating LZHuf (`_compressLZ`) compression on an X-Ray FS chunk.
pub const CFS_COMPRESS_MARK: u32 = 1 << 31;
/// Chunk ID carrying optional archive header INI metadata (`[header] entry_point`, `auto_load`).
pub const CFS_HEADER_CHUNK_ID: u32 = 666;
/// Chunk ID carrying the `.db`/`.xdb` packed file index table.
pub const XRAY_ARCHIVE_FILE_TABLE_CHUNK_ID: u32 = 1;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum VfsError {
    InvalidChunkHeader,
    MissingFileTable,
    FileNotFound,
    CorruptedData,
    CrcMismatch { expected: u32, computed: u32 },
    DecompressionFailed(LzoError),
    ChunkDecompressionFailed(LzhufError),
    InvalidPath(String),
    Io(String),
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for VfsError {}

impl From<std::io::Error> for VfsError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveEntry {
    pub name: String,
    pub uncompressed_size: u32,
    pub compressed_size: u32,
    pub crc: u32,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveHeader {
    pub entry_point: String,
    pub auto_load: bool,
}

impl Default for ArchiveHeader {
    fn default() -> Self {
        Self {
            entry_point: "gamedata".to_string(),
            auto_load: true,
        }
    }
}

impl ArchiveHeader {
    /// Parse Chunk 666 INI header text matching `CLocatorAPI::LoadArchive` (`LocatorAPI.cpp`).
    pub fn parse_ini(raw_text: &str) -> Self {
        let mut header = Self::default();
        let mut in_header_section = false;

        for raw_line in raw_text.lines() {
            let line = raw_line
                .split(';')
                .next()
                .unwrap_or("")
                .trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                let sec = line[1..line.len() - 1].trim();
                in_header_section = sec.eq_ignore_ascii_case("header");
                continue;
            }
            if !in_header_section {
                continue;
            }
            if let Some(eq) = line.find('=') {
                let key = line[..eq].trim();
                let val = line[eq + 1..].trim().trim_matches('"');
                if key.eq_ignore_ascii_case("entry_point") {
                    header.entry_point = val.to_string();
                } else if key.eq_ignore_ascii_case("auto_load") {
                    header.auto_load = matches!(
                        val.to_ascii_lowercase().as_str(),
                        "true" | "1" | "on" | "yes"
                    );
                }
            }
        }

        header
    }

    /// Translate the archive `entry_point` into a gamedata-relative prefix (`""` or `"sub\\"`).
    pub fn gamedata_prefix(&self) -> String {
        let ep = self
            .entry_point
            .trim()
            .replace('/', "\\")
            .to_ascii_lowercase();
        let trimmed = ep.trim_matches('\\');

        if trimmed.is_empty() || trimmed == "gamedata" || trimmed == "$game_data$" || trimmed == "$fs_root$" {
            return String::new();
        }
        if let Some(rest) = trimmed.strip_prefix("$game_data$\\") {
            return with_trailing_backslash(rest);
        }
        if let Some(rest) = trimmed.strip_prefix("gamedata\\") {
            return with_trailing_backslash(rest);
        }
        if let Some(rest) = trimmed.strip_prefix("$fs_root$\\gamedata\\") {
            return with_trailing_backslash(rest);
        }
        if trimmed == "$game_config$" || trimmed == "$game_configs$" {
            return "configs\\".to_string();
        }
        if let Some(rest) = trimmed.strip_prefix("$game_config$\\") {
            return format!("configs\\{}", with_trailing_backslash(rest));
        }
        if let Some(rest) = trimmed.strip_prefix("$game_configs$\\") {
            return format!("configs\\{}", with_trailing_backslash(rest));
        }
        if let Some((alias, rest)) = map_known_alias(trimmed) {
            return format!("{alias}{rest}");
        }
        with_trailing_backslash(trimmed)
    }
}

fn with_trailing_backslash(s: &str) -> String {
    let clean = s.trim_matches('\\');
    if clean.is_empty() {
        String::new()
    } else {
        format!("{clean}\\")
    }
}

fn map_known_alias(lower: &str) -> Option<(&'static str, String)> {
    const ALIASES: &[(&str, &str)] = &[
        ("$game_meshes$", "meshes\\"),
        ("$game_textures$", "textures\\"),
        ("$game_shaders$", "shaders\\"),
        ("$game_sounds$", "sounds\\"),
        ("$game_scripts$", "scripts\\"),
        ("$game_levels$", "levels\\"),
        ("$game_spawns$", "spawns\\"),
        ("$game_anims$", "anims\\"),
    ];

    for &(alias, target) in ALIASES {
        if lower == alias {
            return Some((target, String::new()));
        }
        if let Some(rest) = lower.strip_prefix(alias).and_then(|s| s.strip_prefix('\\')) {
            return Some((target, with_trailing_backslash(rest)));
        }
    }
    None
}

pub struct XRayArchive {
    pub header: ArchiveHeader,
    pub entries: HashMap<String, ArchiveEntry>,
}

impl XRayArchive {
    /// Parse archive chunk structure: optional Chunk 666 (header INI) and Chunk 1 (file table),
    /// transparently decompressing `CFS_CompressMark` (LZHuf) chunks if flagged.
    pub fn parse(data: &[u8]) -> Result<Self, VfsError> {
        let mut pos = 0usize;
        let data_len = data.len();
        let mut header_chunk: Option<Cow<'_, [u8]>> = None;
        let mut file_table_chunk: Option<Cow<'_, [u8]>> = None;

        while pos + 8 <= data_len {
            let raw_id = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
            pos += 8;

            if pos.checked_add(chunk_size).is_none_or(|end| end > data_len) {
                if file_table_chunk.is_some() {
                    // Trailing bytes after Chunk 1 are raw file payloads not wrapped in Chunk 0.
                    break;
                }
                return Err(VfsError::InvalidChunkHeader);
            }

            let compressed = (raw_id & CFS_COMPRESS_MARK) != 0;
            let chunk_id = raw_id & !CFS_COMPRESS_MARK;
            if file_table_chunk.is_some() && chunk_id != CFS_HEADER_CHUNK_ID && chunk_id != 0 {
                // Stop if unchunked payload bytes immediately follow Chunk 1.
                break;
            }
            let chunk_slice = &data[pos..pos + chunk_size];

            if chunk_id == CFS_HEADER_CHUNK_ID {
                header_chunk = Some(decode_chunk_payload(chunk_slice, compressed)?);
            } else if chunk_id == XRAY_ARCHIVE_FILE_TABLE_CHUNK_ID {
                file_table_chunk = Some(decode_chunk_payload(chunk_slice, compressed)?);
                if header_chunk.is_some() {
                    break;
                }
            }

            pos += chunk_size;
        }

        let header = match header_chunk {
            Some(bytes) => ArchiveHeader::parse_ini(&String::from_utf8_lossy(&bytes)),
            None => ArchiveHeader::default(),
        };
        let prefix = header.gamedata_prefix();

        let table_bytes = file_table_chunk.ok_or(VfsError::MissingFileTable)?;
        let table_data: &[u8] = &table_bytes;
        let mut entries = HashMap::new();
        let mut table_pos = 0usize;
        let table_len = table_data.len();

        while table_pos + 2 <= table_len {
            let record_size =
                u16::from_le_bytes(table_data[table_pos..table_pos + 2].try_into().unwrap()) as usize;
            table_pos += 2;

            if table_pos.checked_add(record_size).is_none_or(|end| end > table_len) {
                return Err(VfsError::InvalidChunkHeader);
            }

            let record = &table_data[table_pos..table_pos + record_size];
            table_pos += record_size;

            if record.len() < 16 {
                continue;
            }

            let uncompressed_size = u32::from_le_bytes(record[0..4].try_into().unwrap());
            let compressed_size = u32::from_le_bytes(record[4..8].try_into().unwrap());
            let crc = u32::from_le_bytes(record[8..12].try_into().unwrap());

            let name_len = record.len() - 16;
            let name_bytes = &record[12..12 + name_len];
            let raw_name = String::from_utf8_lossy(name_bytes)
                .trim_matches('\0')
                .replace('/', "\\");
            let trimmed_name = raw_name.trim_start_matches('\\');
            let full_name = if prefix.is_empty() {
                trimmed_name.to_string()
            } else {
                format!("{prefix}{trimmed_name}")
            };

            let offset =
                u32::from_le_bytes(record[12 + name_len..16 + name_len].try_into().unwrap()) as usize;

            // Entries with offset == 0 and uncompressed_size == 0 are directory entries in X-Ray tables
            if uncompressed_size == 0 && compressed_size == 0 && offset == 0 && raw_name.ends_with('\\') {
                continue;
            }

            let key = full_name.to_ascii_lowercase();
            entries.insert(
                key,
                ArchiveEntry {
                    name: full_name,
                    uncompressed_size,
                    compressed_size,
                    crc,
                    offset,
                },
            );
        }

        Ok(Self { header, entries })
    }

    /// Look up metadata for a normalized or relative virtual path.
    pub fn get_entry(&self, file_path: &str) -> Option<&ArchiveEntry> {
        let normalized = file_path
            .trim_start_matches(['/', '\\'])
            .replace('/', "\\")
            .to_ascii_lowercase();
        self.entries.get(&normalized)
    }

    /// Read and decompress a specific file from the archive byte slice.
    pub fn read_file(&self, file_path: &str, archive_data: &[u8]) -> Result<Vec<u8>, VfsError> {
        let entry = self.get_entry(file_path).ok_or(VfsError::FileNotFound)?;
        self.read_entry(entry, archive_data)
    }

    /// Read, decompress (LZO1X if compressed), and CRC32-verify a resolved `ArchiveEntry`.
    pub fn read_entry(&self, entry: &ArchiveEntry, archive_data: &[u8]) -> Result<Vec<u8>, VfsError> {
        let comp_len = entry.compressed_size as usize;
        let end = entry
            .offset
            .checked_add(comp_len)
            .ok_or(VfsError::CorruptedData)?;
        if end > archive_data.len() {
            return Err(VfsError::CorruptedData);
        }

        let compressed_slice = &archive_data[entry.offset..end];
        let mut output = vec![0u8; entry.uncompressed_size as usize];

        if entry.compressed_size == entry.uncompressed_size {
            output.copy_from_slice(compressed_slice);
        } else {
            let decompressed = decompress_lzo1x(compressed_slice, &mut output)
                .map_err(VfsError::DecompressionFailed)?;

            if decompressed != entry.uncompressed_size as usize {
                return Err(VfsError::CorruptedData);
            }
        }

        let computed_crc = compute_crc32(&output);
        if computed_crc != entry.crc {
            return Err(VfsError::CrcMismatch {
                expected: entry.crc,
                computed: computed_crc,
            });
        }

        Ok(output)
    }
}

fn decode_chunk_payload(slice: &[u8], compressed: bool) -> Result<Cow<'_, [u8]>, VfsError> {
    if compressed {
        let decoded = decompress_lzhuf(slice).map_err(VfsError::ChunkDecompressionFailed)?;
        Ok(Cow::Owned(decoded))
    } else {
        Ok(Cow::Borrowed(slice))
    }
}

/// Backing storage for a mounted `.db`/`.xdb` archive.
pub enum ArchiveBacking {
    Mapped(Arc<MappedFile>),
    Bytes(Arc<Vec<u8>>),
}

impl ArchiveBacking {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(m) => m.as_slice(),
            Self::Bytes(b) => b.as_slice(),
        }
    }
}

/// Memory-mapped or buffer-backed parsed X-Ray archive ready for lock-free concurrent reads.
pub struct MountedArchive {
    pub name: String,
    pub backing: ArchiveBacking,
    pub archive: XRayArchive,
}

impl MountedArchive {
    /// Open and zero-copy memory-map a `.db` or `.xdb` archive from disk.
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self, VfsError> {
        let path_ref = path.as_ref();
        let mapped = Arc::new(MappedFile::open(path_ref)?);
        let archive = XRayArchive::parse(mapped.as_slice())?;
        Ok(Self {
            name: path_ref.to_string_lossy().into_owned(),
            backing: ArchiveBacking::Mapped(mapped),
            archive,
        })
    }

    /// Wrap an in-memory archive byte buffer.
    pub fn from_bytes(name: impl Into<String>, bytes: Vec<u8>) -> Result<Self, VfsError> {
        let arc = Arc::new(bytes);
        let archive = XRayArchive::parse(arc.as_slice())?;
        Ok(Self {
            name: name.into(),
            backing: ArchiveBacking::Bytes(arc),
            archive,
        })
    }

    #[inline]
    pub fn read_entry(&self, entry: &ArchiveEntry) -> Result<Vec<u8>, VfsError> {
        self.archive.read_entry(entry, self.backing.as_slice())
    }

    #[inline]
    pub fn read_file(&self, file_path: &str) -> Result<Vec<u8>, VfsError> {
        self.archive.read_file(file_path, self.backing.as_slice())
    }
}

const fn generate_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            c = if (c & 1) != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

const CRC32_TABLE: [u32; 256] = generate_crc32_table();

/// Compute IEEE 802.3 standard CRC32 checksum (table-accelerated).
pub fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = ((crc ^ (byte as u32)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lzhuf::compress_lzhuf;

    #[test]
    fn test_crc32_computation() {
        assert_eq!(compute_crc32(b""), 0x0000_0000);
        assert_eq!(compute_crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            compute_crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn test_synthetic_archive_parsing_and_read() {
        let payload = b"[game_version]\nversion = 1.5.2\n";
        let crc = compute_crc32(payload);

        let file_name = b"configs\\system.ltx";
        let record_size = (16 + file_name.len()) as u16;

        let mut chunk1 = Vec::new();
        chunk1.extend_from_slice(&record_size.to_le_bytes());
        chunk1.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        chunk1.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        chunk1.extend_from_slice(&crc.to_le_bytes());
        chunk1.extend_from_slice(file_name);

        let chunk1_header_len = 8;
        let data_offset = chunk1_header_len + chunk1.len() + 4;
        chunk1.extend_from_slice(&(data_offset as u32).to_le_bytes());

        let mut archive_bytes = Vec::new();
        archive_bytes.extend_from_slice(&1u32.to_le_bytes());
        archive_bytes.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
        archive_bytes.extend_from_slice(&chunk1);
        archive_bytes.extend_from_slice(payload);

        let archive = XRayArchive::parse(&archive_bytes).expect("Failed to parse synthetic archive");
        assert_eq!(archive.entries.len(), 1);

        let data = archive
            .read_file("configs/system.ltx", &archive_bytes)
            .expect("Failed to read file from archive");

        assert_eq!(data, payload);
    }

    #[test]
    fn test_lzo1x_compressed_archive_entry_parsing_and_read() {
        use crate::lzo::compress_lzo1x;

        let mut original_payload = Vec::new();
        for i in 0..50 {
            original_payload.extend_from_slice(
                format!(
                    "[stalker_profile_{}]\nrank = master\ncommunity = monolith\nhealth = 100.0\n",
                    i
                )
                .as_bytes(),
            );
        }

        let compressed_payload = compress_lzo1x(&original_payload);
        assert!(compressed_payload.len() < original_payload.len());
        let crc = compute_crc32(&original_payload);

        let file_name = b"configs\\creatures\\stalkers.ltx";
        let record_size = (16 + file_name.len()) as u16;

        let mut chunk1 = Vec::new();
        chunk1.extend_from_slice(&record_size.to_le_bytes());
        chunk1.extend_from_slice(&(original_payload.len() as u32).to_le_bytes());
        chunk1.extend_from_slice(&(compressed_payload.len() as u32).to_le_bytes());
        chunk1.extend_from_slice(&crc.to_le_bytes());
        chunk1.extend_from_slice(file_name);

        let chunk1_header_len = 8;
        let data_offset = chunk1_header_len + chunk1.len() + 4;
        chunk1.extend_from_slice(&(data_offset as u32).to_le_bytes());

        let mut archive_bytes = Vec::new();
        archive_bytes.extend_from_slice(&1u32.to_le_bytes());
        archive_bytes.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
        archive_bytes.extend_from_slice(&chunk1);
        archive_bytes.extend_from_slice(&compressed_payload);

        let archive = XRayArchive::parse(&archive_bytes).expect("Failed to parse compressed archive");
        assert_eq!(archive.entries.len(), 1);

        let data = archive
            .read_file("configs/creatures/stalkers.ltx", &archive_bytes)
            .expect("Decompression and CRC32 verification must pass");

        assert_eq!(data, original_payload);
    }

    #[test]
    fn test_archive_chunk666_header_and_compressed_file_table() {
        let header_ini = b"[header]\nauto_load = true\nentry_point = $game_meshes$\\weapons\n";
        let header_comp = compress_lzhuf(header_ini);

        let payload = b"OGF_FAKE_MODEL_PAYLOAD_0123456789";
        let crc = compute_crc32(payload);
        let entry_name = b"ak74\\wpn_ak74.ogf";
        let record_size = (16 + entry_name.len()) as u16;

        // First compute exact table chunk compressed bytes so we know final data offset!
        let dummy_offset = 0u32;
        let mut table_plain = Vec::new();
        table_plain.extend_from_slice(&record_size.to_le_bytes());
        table_plain.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        table_plain.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        table_plain.extend_from_slice(&crc.to_le_bytes());
        table_plain.extend_from_slice(entry_name);
        table_plain.extend_from_slice(&dummy_offset.to_le_bytes());

        // Rebuild with actual offset matching archive size before payload
        let chunk_header_len = 8 + header_comp.len();
        let probe_comp = compress_lzhuf(&table_plain);
        let mut final_offset = (chunk_header_len + 8 + probe_comp.len()) as u32;
        loop {
            let len = table_plain.len();
            table_plain[len - 4..len].copy_from_slice(&final_offset.to_le_bytes());
            let table_comp = compress_lzhuf(&table_plain);
            let new_offset = (chunk_header_len + 8 + table_comp.len()) as u32;
            if new_offset == final_offset {
                let mut archive_bytes = Vec::new();
                archive_bytes.extend_from_slice(&(CFS_HEADER_CHUNK_ID | CFS_COMPRESS_MARK).to_le_bytes());
                archive_bytes.extend_from_slice(&(header_comp.len() as u32).to_le_bytes());
                archive_bytes.extend_from_slice(&header_comp);

                archive_bytes.extend_from_slice(&(1u32 | CFS_COMPRESS_MARK).to_le_bytes());
                archive_bytes.extend_from_slice(&(table_comp.len() as u32).to_le_bytes());
                archive_bytes.extend_from_slice(&table_comp);
                archive_bytes.extend_from_slice(payload);

                let archive =
                    XRayArchive::parse(&archive_bytes).expect("compressed Chunk 666 + Chunk 1 must parse");
                assert_eq!(archive.header.entry_point, "$game_meshes$\\weapons");
                assert!(archive.header.auto_load);

                let loaded = archive
                    .read_file("meshes\\weapons\\ak74\\wpn_ak74.ogf", &archive_bytes)
                    .expect("entry should be resolved under entry_point alias prefix");
                assert_eq!(loaded, payload);
                break;
            }
            final_offset = new_offset;
        }
    }
}
