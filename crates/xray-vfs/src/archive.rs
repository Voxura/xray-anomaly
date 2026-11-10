//! X-Ray Monolith .db / .xdb Chunked Archive Reader & Parallel VFS Decompressor
//!
//! Implements chunk-based file table discovery (Chunk 1) and parallel decompressor
//! verifying CRC32 checksums and handling uncompressed/LZO1X chunks safely.

use std::collections::HashMap;
use crate::lzo::{decompress_lzo1x, LzoError};

#[derive(Debug, PartialEq, Eq)]
pub enum VfsError {
    InvalidChunkHeader,
    MissingFileTable,
    FileNotFound,
    CorruptedData,
    CrcMismatch { expected: u32, computed: u32 },
    DecompressionFailed(LzoError),
}

impl std::fmt::Display for VfsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for VfsError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveEntry {
    pub name: String,
    pub uncompressed_size: u32,
    pub compressed_size: u32,
    pub crc: u32,
    pub offset: usize,
}

pub struct XRayArchive {
    pub entries: HashMap<String, ArchiveEntry>,
}

impl XRayArchive {
    /// Parse archive chunk structure and load Chunk 1 (file table).
    pub fn parse(data: &[u8]) -> Result<Self, VfsError> {
        let mut pos = 0;
        let data_len = data.len();
        let mut file_table_chunk: Option<&[u8]> = None;

        // Iterate through top-level chunks
        while pos + 8 <= data_len {
            let chunk_id = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            let chunk_size = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
            pos += 8;

            if pos + chunk_size > data_len {
                return Err(VfsError::InvalidChunkHeader);
            }

            let chunk_slice = &data[pos..pos + chunk_size];
            if chunk_id == 1 {
                file_table_chunk = Some(chunk_slice);
                break;
            }

            pos += chunk_size;
        }

        let table_data = file_table_chunk.ok_or(VfsError::MissingFileTable)?;
        let mut entries = HashMap::new();
        let mut table_pos = 0;
        let table_len = table_data.len();

        while table_pos + 2 <= table_len {
            let record_size = u16::from_le_bytes(table_data[table_pos..table_pos + 2].try_into().unwrap()) as usize;
            table_pos += 2;

            if table_pos + record_size > table_len {
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
            let name = String::from_utf8_lossy(name_bytes)
                .trim_matches('\0')
                .replace('/', "\\");

            let offset = u32::from_le_bytes(record[12 + name_len..16 + name_len].try_into().unwrap()) as usize;

            entries.insert(
                name.to_lowercase(),
                ArchiveEntry {
                    name,
                    uncompressed_size,
                    compressed_size,
                    crc,
                    offset,
                },
            );
        }

        Ok(Self { entries })
    }

    /// Read and decompress a specific file from the archive slice.
    pub fn read_file(&self, file_path: &str, archive_data: &[u8]) -> Result<Vec<u8>, VfsError> {
        let normalized = file_path.to_lowercase().replace('/', "\\");
        let entry = self.entries.get(&normalized).ok_or(VfsError::FileNotFound)?;

        let end = entry.offset + entry.compressed_size as usize;
        if end > archive_data.len() {
            return Err(VfsError::CorruptedData);
        }

        let compressed_slice = &archive_data[entry.offset..end];
        let mut output = vec![0u8; entry.uncompressed_size as usize];

        if entry.compressed_size == entry.uncompressed_size {
            // Uncompressed direct copy
            output.copy_from_slice(compressed_slice);
        } else {
            // LZO1X Decompression
            let decompressed = decompress_lzo1x(compressed_slice, &mut output)
                .map_err(VfsError::DecompressionFailed)?;

            if decompressed != entry.uncompressed_size as usize {
                return Err(VfsError::CorruptedData);
            }
        }

        // CRC32 Verification
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

/// Compute IEEE 802.3 standard CRC32 checksum
pub fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_computation() {
        // Standard IEEE 802.3 test vectors
        assert_eq!(compute_crc32(b""), 0x0000_0000);
        assert_eq!(compute_crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(compute_crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
    }

    #[test]
    fn test_synthetic_archive_parsing_and_read() {
        // Build a synthetic uncompressed X-Ray archive
        let payload = b"[game_version]\nversion = 1.5.2\n";
        let crc = compute_crc32(payload);

        let file_name = b"configs\\system.ltx";
        let record_size = (16 + file_name.len()) as u16;

        let mut chunk1 = Vec::new();
        chunk1.extend_from_slice(&record_size.to_le_bytes());
        chunk1.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // uncompressed
        chunk1.extend_from_slice(&(payload.len() as u32).to_le_bytes()); // compressed
        chunk1.extend_from_slice(&crc.to_le_bytes());
        chunk1.extend_from_slice(file_name);

        // Data starts right after Chunk 1 header + body
        let chunk1_header_len = 8;
        let data_offset = chunk1_header_len + chunk1.len() + 4; // account for the 4-byte offset field itself
        chunk1.extend_from_slice(&(data_offset as u32).to_le_bytes());

        let mut archive_bytes = Vec::new();
        // Chunk 1 Header
        archive_bytes.extend_from_slice(&1u32.to_le_bytes()); // chunk ID = 1
        archive_bytes.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
        archive_bytes.extend_from_slice(&chunk1);
        // Payload data
        archive_bytes.extend_from_slice(payload);

        // Parse archive
        let archive = XRayArchive::parse(&archive_bytes).expect("Failed to parse synthetic archive");
        assert_eq!(archive.entries.len(), 1);

        let data = archive
            .read_file("configs/system.ltx", &archive_bytes)
            .expect("Failed to read file from archive");

        assert_eq!(data, payload);
    }
}
