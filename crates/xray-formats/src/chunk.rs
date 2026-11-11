//! Zero-Copy X-Ray Chunked File Format Reader & Cursor (`FS.h` / `FS_impl.h`)
//!
//! Handles sequential and random-access reading of `[u32 id_and_flags][u32 size]` chunks,
//! transparent `CFS_CompressMark` (`1 << 31`) LZHuf decompression via `xray_vfs`,
//! and bounds-checked little-endian primitive/zstring decoding with zero panics.

use std::borrow::Cow;
use std::fmt;

use xray_vfs::{compress_lzhuf, decompress_lzhuf, LzhufError, CFS_COMPRESS_MARK};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    UnexpectedEof,
    InvalidChunkBounds {
        chunk_id: u32,
        offset: usize,
        size: usize,
        total_len: usize,
    },
    DecompressionFailed(LzhufError),
    InvalidString(&'static str),
    InvalidValue(&'static str),
    MissingChunk(u32),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FormatError {}

/// Metadata & raw payload slice for a single X-Ray FS chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkEntry<'a> {
    pub raw_id: u32,
    pub id: u32,
    pub compressed: bool,
    pub offset: usize,
    pub payload: &'a [u8],
}

impl<'a> ChunkEntry<'a> {
    /// Return raw uncompressed bytes (zero-copy borrow) or transparently LZHuf-decompress
    /// when `CFS_CompressMark` is set.
    pub fn decode(&self) -> Result<Cow<'a, [u8]>, FormatError> {
        if self.compressed {
            let bytes =
                decompress_lzhuf(self.payload).map_err(FormatError::DecompressionFailed)?;
            Ok(Cow::Owned(bytes))
        } else {
            Ok(Cow::Borrowed(self.payload))
        }
    }

    /// Return decoded payload wrapped in a `ChunkReader` for nested sub-chunks.
    pub fn into_reader(self) -> Result<OwnedOrBorrowedReader<'a>, FormatError> {
        Ok(OwnedOrBorrowedReader {
            data: self.decode()?,
        })
    }
}

/// Container holding either a zero-copy borrowed chunk or owned decompressed bytes.
#[derive(Debug, Clone)]
pub struct OwnedOrBorrowedReader<'a> {
    pub data: Cow<'a, [u8]>,
}

impl<'a> OwnedOrBorrowedReader<'a> {
    #[inline]
    pub fn chunks(&self) -> ChunkReader<'_> {
        ChunkReader::new(self.data.as_ref())
    }

    #[inline]
    pub fn cursor(&self) -> ByteCursor<'_> {
        ByteCursor::new(self.data.as_ref())
    }
}

/// Zero-copy iterator over top-level X-Ray chunks (`[u32 raw_id][u32 size][payload...]`).
#[derive(Debug, Clone, Copy)]
pub struct ChunkReader<'a> {
    data: &'a [u8],
}

impl<'a> ChunkReader<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    #[inline]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Iterate over all top-level chunks, validating chunk header bounds cleanly.
    pub fn iter(&self) -> ChunkIter<'a> {
        ChunkIter {
            data: self.data,
            pos: 0,
        }
    }

    /// Collect all validated chunks into a vector, failing on truncated chunk boundaries.
    pub fn collect_all(&self) -> Result<Vec<ChunkEntry<'a>>, FormatError> {
        self.iter().collect()
    }

    /// Locate a chunk matching `target_id` (ignoring `CFS_CompressMark`).
    pub fn find_chunk(&self, target_id: u32) -> Result<Option<ChunkEntry<'a>>, FormatError> {
        for entry in self.iter() {
            let entry = entry?;
            if entry.id == target_id {
                return Ok(Some(entry));
            }
        }
        Ok(None)
    }

    /// Locate and decode a chunk matching `target_id` (transparently handling `CFS_CompressMark`).
    pub fn open_chunk(&self, target_id: u32) -> Result<Option<Cow<'a, [u8]>>, FormatError> {
        match self.find_chunk(target_id)? {
            Some(entry) => Ok(Some(entry.decode()?)),
            None => Ok(None),
        }
    }

    /// Locate and decode a required chunk, returning `FormatError::MissingChunk` if absent.
    pub fn require_chunk(&self, target_id: u32) -> Result<Cow<'a, [u8]>, FormatError> {
        self.open_chunk(target_id)?
            .ok_or(FormatError::MissingChunk(target_id))
    }
}

pub struct ChunkIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = Result<ChunkEntry<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        if self.pos + 8 > self.data.len() {
            let err = FormatError::InvalidChunkBounds {
                chunk_id: 0,
                offset: self.pos,
                size: 8,
                total_len: self.data.len(),
            };
            self.pos = self.data.len();
            return Some(Err(err));
        }

        let raw_id = u32::from_le_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
        let size =
            u32::from_le_bytes(self.data[self.pos + 4..self.pos + 8].try_into().unwrap()) as usize;
        let id = raw_id & !CFS_COMPRESS_MARK;
        let compressed = (raw_id & CFS_COMPRESS_MARK) != 0;

        let payload_start = self.pos + 8;
        let payload_end = match payload_start.checked_add(size) {
            Some(end) if end <= self.data.len() => end,
            _ => {
                let err = FormatError::InvalidChunkBounds {
                    chunk_id: id,
                    offset: payload_start,
                    size,
                    total_len: self.data.len(),
                };
                self.pos = self.data.len();
                return Some(Err(err));
            }
        };

        self.pos = payload_end;
        Some(Ok(ChunkEntry {
            raw_id,
            id,
            compressed,
            offset: payload_start,
            payload: &self.data[payload_start..payload_end],
        }))
    }
}

/// Bounds-checked little-endian cursor for reading X-Ray binary structures (`IReader` parity).
#[derive(Debug, Clone, Copy)]
pub struct ByteCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    #[inline]
    pub fn is_eof(&self) -> bool {
        self.pos >= self.data.len()
    }

    pub fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], FormatError> {
        let end = self.pos.checked_add(len).ok_or(FormatError::UnexpectedEof)?;
        if end > self.data.len() {
            return Err(FormatError::UnexpectedEof);
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    #[inline]
    pub fn read_u8(&mut self) -> Result<u8, FormatError> {
        Ok(self.read_bytes(1)?[0])
    }

    #[inline]
    pub fn read_i8(&mut self) -> Result<i8, FormatError> {
        Ok(self.read_bytes(1)?[0] as i8)
    }

    #[inline]
    pub fn read_u16(&mut self) -> Result<u16, FormatError> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    #[inline]
    pub fn read_i16(&mut self) -> Result<i16, FormatError> {
        Ok(i16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }

    #[inline]
    pub fn read_u32(&mut self) -> Result<u32, FormatError> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    #[inline]
    pub fn read_i32(&mut self) -> Result<i32, FormatError> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    #[inline]
    pub fn read_u64(&mut self) -> Result<u64, FormatError> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    #[inline]
    pub fn read_i64(&mut self) -> Result<i64, FormatError> {
        Ok(i64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    #[inline]
    pub fn read_f32(&mut self) -> Result<f32, FormatError> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    #[inline]
    pub fn read_f32_2(&mut self) -> Result<[f32; 2], FormatError> {
        Ok([self.read_f32()?, self.read_f32()?])
    }

    #[inline]
    pub fn read_f32_3(&mut self) -> Result<[f32; 3], FormatError> {
        Ok([self.read_f32()?, self.read_f32()?, self.read_f32()?])
    }

    #[inline]
    pub fn read_f32_4(&mut self) -> Result<[f32; 4], FormatError> {
        Ok([
            self.read_f32()?,
            self.read_f32()?,
            self.read_f32()?,
            self.read_f32()?,
        ])
    }

    /// Read a NUL-terminated C string matching `IReader::r_stringZ` (tolerates EOF without NUL).
    pub fn read_zstring(&mut self) -> Result<String, FormatError> {
        let start = self.pos;
        let mut end = start;
        while end < self.data.len() && self.data[end] != 0 {
            end += 1;
        }
        let s = String::from_utf8_lossy(&self.data[start..end]).into_owned();
        if end < self.data.len() {
            self.pos = end + 1; // consume terminating NUL
        } else {
            self.pos = end;
        }
        Ok(s)
    }

    /// Read a `\r`/`\n`-terminated line matching `IReader::r_string` (`motion_marks`).
    pub fn read_line_string(&mut self) -> Result<String, FormatError> {
        if self.is_eof() {
            return Err(FormatError::UnexpectedEof);
        }
        let start = self.pos;
        while self.pos < self.data.len() {
            let b = self.data[self.pos];
            if b == b'\r' || b == b'\n' {
                break;
            }
            self.pos += 1;
        }
        let line = String::from_utf8_lossy(&self.data[start..self.pos]).into_owned();
        if self.pos < self.data.len() {
            if self.data[self.pos] == b'\r' {
                self.pos += 1;
                if self.pos < self.data.len() && self.data[self.pos] == b'\n' {
                    self.pos += 1;
                }
            } else if self.data[self.pos] == b'\n' {
                self.pos += 1;
            }
        }
        Ok(line)
    }
}

/// Chunk stream builder matching `IWriter::open_chunk` / `w_compressed`.
#[derive(Debug, Default)]
pub struct ChunkWriter {
    buf: Vec<u8>,
}

impl ChunkWriter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn write_chunk(&mut self, id: u32, payload: &[u8]) {
        let clean_id = id & !CFS_COMPRESS_MARK;
        self.buf.extend_from_slice(&clean_id.to_le_bytes());
        self.buf
            .extend_from_slice(&(payload.len() as u32).to_le_bytes());
        self.buf.extend_from_slice(payload);
    }

    pub fn write_compressed_chunk(&mut self, id: u32, payload: &[u8]) {
        let raw_id = (id & !CFS_COMPRESS_MARK) | CFS_COMPRESS_MARK;
        let compressed = compress_lzhuf(payload);
        self.buf.extend_from_slice(&raw_id.to_le_bytes());
        self.buf
            .extend_from_slice(&(compressed.len() as u32).to_le_bytes());
        self.buf.extend_from_slice(&compressed);
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_writer_reader_and_compressed_chunks() {
        let mut w = ChunkWriter::new();
        w.write_chunk(1, b"plain_header\0");
        w.write_compressed_chunk(7, b"repeated_chunk_data_repeated_chunk_data_repeated_chunk_data");
        let raw = w.into_bytes();

        let reader = ChunkReader::new(&raw);
        let chunks = reader.collect_all().unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].id, 1);
        assert!(!chunks[0].compressed);
        assert_eq!(chunks[1].id, 7);
        assert!(chunks[1].compressed);

        let decoded_header = reader.require_chunk(1).unwrap();
        let mut cur = ByteCursor::new(&decoded_header);
        assert_eq!(cur.read_zstring().unwrap(), "plain_header");
        assert!(cur.is_eof());

        let decoded_comp = reader.require_chunk(7).unwrap();
        assert_eq!(
            decoded_comp.as_ref(),
            b"repeated_chunk_data_repeated_chunk_data_repeated_chunk_data"
        );
    }

    #[test]
    fn test_chunk_bounds_rejection() {
        // Chunk claims 100 bytes in a 10-byte buffer
        let mut bad = Vec::new();
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(&100u32.to_le_bytes());
        bad.extend_from_slice(&[0xAA, 0xBB]);

        let reader = ChunkReader::new(&bad);
        assert!(matches!(
            reader.find_chunk(1),
            Err(FormatError::InvalidChunkBounds { .. })
        ));
    }

    #[test]
    fn test_byte_cursor_signed_and_line_strings() {
        let mut raw = Vec::new();
        raw.push((-42i8) as u8);
        raw.extend_from_slice(&(-1234i16).to_le_bytes());
        raw.extend_from_slice(b"mark_alpha\r\n");
        raw.extend_from_slice(b"\r\n"); // empty line
        raw.extend_from_slice(b"mark_lf\n");
        raw.extend_from_slice(b"tail_eof");

        let mut cur = ByteCursor::new(&raw);
        assert_eq!(cur.read_i8().unwrap(), -42);
        assert_eq!(cur.read_i16().unwrap(), -1234);
        assert_eq!(cur.read_line_string().unwrap(), "mark_alpha");
        assert_eq!(cur.read_line_string().unwrap(), "");
        assert_eq!(cur.read_line_string().unwrap(), "mark_lf");
        assert_eq!(cur.read_line_string().unwrap(), "tail_eof");
        assert!(matches!(cur.read_line_string(), Err(FormatError::UnexpectedEof)));
    }
}
