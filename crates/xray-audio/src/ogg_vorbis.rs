//! Pure-Rust Ogg Vorbis I Container Demuxer, Stream Decoder & X-Ray Sound Metadata Reader
//!
//! Decodes `.ogg` Vorbis I streams directly from memory or `xray-vfs::VirtualFileSystem`
//! without external third-party audio codecs, while preserving X-Ray's `xr_sound` comment
//! metadata (`OggXRayParams`: `min_dist`, `max_dist`, `base_volume`, `game_type`, `max_ai_dist`).
//!
//! Implements the full Vorbis I specification:
//! - OggS page & segment packet reassembly (`OggDemuxer`)
//! - Vorbis identification (`0x01vorbis`), comment (`0x03vorbis`), and setup (`0x05vorbis`) headers
//! - Codebook canonical Huffman binary trees & VQ lookup types 0 / 1 / 2
//! - Floor 0 (Bark + LSP) & Floor 1 piecewise-linear dB envelope synthesis
//! - Residue types 0, 1, and 2 (MSB-first classword partitioning & interleaved Type 2)
//! - Submap channel mappings & polar stereo inverse coupling
//! - Fast `O(N log N)` split-radix/FFT-based IMDCT with Vorbis slope overlap-add

use std::collections::VecDeque;
use std::f64::consts::{FRAC_PI_2, PI};
use std::fmt;
use std::sync::Arc;

use xray_formats::sound_env::OggXRayParams;
use xray_vfs::{normalize_virtual_path, VfsError, VirtualFileSystem};

use crate::mixer::SoundClip;

fn ensure_ogg_extension(trimmed: &str) -> String {
    if trimmed.len() >= 4 && trimmed[trimmed.len() - 4..].eq_ignore_ascii_case(".ogg") {
        trimmed.to_string()
    } else {
        format!("{trimmed}.ogg")
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum OggVorbisError {
    Vfs(VfsError),
    InvalidContainer(&'static str),
    InvalidHeader(&'static str),
    InvalidCodebook(&'static str),
    InvalidFloor(&'static str),
    InvalidResidue(&'static str),
    InvalidMapping(&'static str),
    InvalidPacket(&'static str),
    Unsupported(&'static str),
}

impl fmt::Display for OggVorbisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vfs(e) => write!(f, "VFS audio error: {e}"),
            Self::InvalidContainer(s) => write!(f, "invalid Ogg container: {s}"),
            Self::InvalidHeader(s) => write!(f, "invalid Vorbis header: {s}"),
            Self::InvalidCodebook(s) => write!(f, "invalid Vorbis codebook: {s}"),
            Self::InvalidFloor(s) => write!(f, "invalid Vorbis floor: {s}"),
            Self::InvalidResidue(s) => write!(f, "invalid Vorbis residue: {s}"),
            Self::InvalidMapping(s) => write!(f, "invalid Vorbis mapping: {s}"),
            Self::InvalidPacket(s) => write!(f, "invalid Vorbis audio packet: {s}"),
            Self::Unsupported(s) => write!(f, "unsupported Vorbis feature: {s}"),
        }
    }
}

impl std::error::Error for OggVorbisError {}

impl From<VfsError> for OggVorbisError {
    fn from(value: VfsError) -> Self {
        Self::Vfs(value)
    }
}

/// Complete decoded audio buffer + X-Ray `.ogg` metadata.
#[derive(Debug, Clone)]
pub struct DecodedOggAudio {
    pub sample_rate: u32,
    pub channels: u8,
    pub vendor: String,
    pub comments: Vec<Vec<u8>>,
    pub xray_params: OggXRayParams,
    /// Interleaved `f32` PCM samples normalized to `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
}

impl DecodedOggAudio {
    pub fn into_sound_clip(self, name: impl Into<String>) -> Arc<SoundClip> {
        Arc::new(SoundClip {
            name: name.into(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            samples: self.samples,
            xray_params: self.xray_params,
        })
    }
}

// ---------------------------------------------------------------------------
// VFS Path Resolution & Public Entrypoints
// ---------------------------------------------------------------------------

/// Resolve an X-Ray sound reference (`"weapons\\ak74_shot"`, `"sounds\\ambient\\wind.ogg"`,
/// or `"$game_sounds$/..."`) to a VFS virtual path present inside `vfs`.
pub fn resolve_sound_vpath(vfs: &VirtualFileSystem, sound_name_or_path: &str) -> Result<String, OggVorbisError> {
    let trimmed = sound_name_or_path.trim();
    if trimmed.is_empty() {
        return Err(OggVorbisError::InvalidContainer("empty sound path"));
    }
    let with_ext = ensure_ogg_extension(trimmed);
    if with_ext.starts_with('$') {
        if vfs.exists(&with_ext) {
            return Ok(with_ext);
        }
        // Fallback: strip `$game_sounds$/` if mounted archives only contain `sounds\...`
        if let Some(rest) = with_ext
            .strip_prefix("$game_sounds$/")
            .or_else(|| with_ext.strip_prefix("$game_sounds$\\"))
        {
            let norm = normalize_virtual_path(&format!("sounds\\{rest}"))
                .map_err(|_| OggVorbisError::InvalidContainer("invalid sound virtual path"))?;
            if vfs.exists(&norm) {
                return Ok(norm);
            }
        }
        return Ok(with_ext);
    }

    let norm = normalize_virtual_path(&with_ext)
        .map_err(|_| OggVorbisError::InvalidContainer("invalid sound virtual path"))?;
    if let Some(rest) = norm.strip_prefix("sounds\\") {
        if vfs.exists(&norm) {
            return Ok(norm);
        }
        let alias = format!("$game_sounds$/{rest}");
        if vfs.exists(&alias) {
            return Ok(alias);
        }
        return Ok(norm);
    }

    let candidate = format!("sounds\\{norm}");
    if vfs.exists(&candidate) {
        return Ok(candidate);
    }
    let alias = format!("$game_sounds$/{norm}");
    if vfs.exists(&alias) {
        return Ok(alias);
    }
    Ok(candidate)
}

/// Open an `.ogg` Vorbis I stream directly from `VirtualFileSystem`, decoding the three setup
/// headers immediately so `sample_rate`, `channels`, and `xray_params` are readable up front.
pub fn open_ogg_stream(
    vfs: &VirtualFileSystem,
    sound_name_or_path: &str,
) -> Result<VorbisStreamDecoder, OggVorbisError> {
    let vpath = resolve_sound_vpath(vfs, sound_name_or_path)?;
    let bytes = vfs.read_bytes(&vpath)?;
    VorbisStreamDecoder::from_owned_bytes(bytes)
}

/// Stream and decode an `.ogg` Vorbis I sound file directly from `VirtualFileSystem`.
pub fn decode_ogg_from_vfs(
    vfs: &VirtualFileSystem,
    sound_name_or_path: &str,
) -> Result<DecodedOggAudio, OggVorbisError> {
    let mut decoder = open_ogg_stream(vfs, sound_name_or_path)?;
    let samples = decoder.decode_all()?;
    Ok(DecodedOggAudio {
        sample_rate: decoder.sample_rate(),
        channels: decoder.channels(),
        vendor: decoder.vendor().to_string(),
        comments: decoder.comments().to_vec(),
        xray_params: decoder.xray_params().clone(),
        samples,
    })
}

/// Decode in-memory `.ogg` Vorbis I bytes into interleaved `f32` PCM + X-Ray metadata.
pub fn decode_ogg_bytes(data: &[u8]) -> Result<DecodedOggAudio, OggVorbisError> {
    let mut decoder = VorbisStreamDecoder::from_owned_bytes(data.to_vec())?;
    let samples = decoder.decode_all()?;
    Ok(DecodedOggAudio {
        sample_rate: decoder.sample_rate(),
        channels: decoder.channels(),
        vendor: decoder.vendor().to_string(),
        comments: decoder.comments().to_vec(),
        xray_params: decoder.xray_params().clone(),
        samples,
    })
}

// ---------------------------------------------------------------------------
// OggS Container Demuxer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OggPacket {
    pub bytes: Vec<u8>,
    pub granule_position: i64,
    pub is_eos: bool,
}

/// Incremental OggS page demuxer reassembling laced segments into Vorbis packets.
pub struct OggDemuxer {
    data: Vec<u8>,
    cursor: usize,
    target_serial: Option<u32>,
    pending_packet: Vec<u8>,
    queued_packets: VecDeque<OggPacket>,
    eos_reached: bool,
}

impl OggDemuxer {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            cursor: 0,
            target_serial: None,
            pending_packet: Vec::new(),
            queued_packets: VecDeque::new(),
            eos_reached: false,
        }
    }

    pub fn next_packet(&mut self) -> Result<Option<OggPacket>, OggVorbisError> {
        while self.queued_packets.is_empty() {
            if self.eos_reached || self.cursor >= self.data.len() {
                if !self.pending_packet.is_empty() {
                    let tail = std::mem::take(&mut self.pending_packet);
                    return Ok(Some(OggPacket {
                        bytes: tail,
                        granule_position: -1,
                        is_eos: true,
                    }));
                }
                return Ok(None);
            }
            self.read_next_page()?;
        }
        Ok(self.queued_packets.pop_front())
    }

    fn read_next_page(&mut self) -> Result<(), OggVorbisError> {
        loop {
            if self.cursor >= self.data.len() {
                self.eos_reached = true;
                return Ok(());
            }
            if self.cursor + 27 > self.data.len() {
                return Err(OggVorbisError::InvalidContainer("truncated OggS page header"));
            }
            if &self.data[self.cursor..self.cursor + 4] != b"OggS" {
                return Err(OggVorbisError::InvalidContainer("expected OggS page magic"));
            }
            let version = self.data[self.cursor + 4];
            if version != 0 {
                return Err(OggVorbisError::InvalidContainer("unsupported Ogg page version"));
            }
            let header_type = self.data[self.cursor + 5];
            let granule =
                i64::from_le_bytes(self.data[self.cursor + 6..self.cursor + 14].try_into().unwrap());
            let serial =
                u32::from_le_bytes(self.data[self.cursor + 14..self.cursor + 18].try_into().unwrap());
            let page_segments = usize::from(self.data[self.cursor + 26]);
            let seg_start = self.cursor + 27;
            let seg_end = seg_start + page_segments;
            if seg_end > self.data.len() {
                return Err(OggVorbisError::InvalidContainer("truncated Ogg segment table"));
            }
            let payload_len: usize = self.data[seg_start..seg_end]
                .iter()
                .map(|&s| usize::from(s))
                .sum();
            let payload_start = seg_end;
            let payload_end = payload_start + payload_len;
            if payload_end > self.data.len() {
                return Err(OggVorbisError::InvalidContainer("truncated Ogg page payload"));
            }

            let target = *self.target_serial.get_or_insert(serial);
            self.cursor = payload_end;

            // Skip multiplexed secondary logical bitstreams cleanly
            if serial != target {
                continue;
            }

            let continued = (header_type & 0x01) != 0;
            let is_eos_page = (header_type & 0x04) != 0;
            if !continued && !self.pending_packet.is_empty() {
                // Previous packet wasn't cleanly terminated before new non-continued page
                let leftover = std::mem::take(&mut self.pending_packet);
                self.queued_packets.push_back(OggPacket {
                    bytes: leftover,
                    granule_position: -1,
                    is_eos: false,
                });
            }

            // Collect packet indices completing on this page so only the LAST completing packet
            // carries the page granule_position per Ogg spec.
            let mut offset = payload_start;
            let mut completed_here = 0usize;
            for seg_idx in 0..page_segments {
                let seg_len = usize::from(self.data[seg_start + seg_idx]);
                self.pending_packet
                    .extend_from_slice(&self.data[offset..offset + seg_len]);
                offset += seg_len;
                if seg_len < 255 {
                    let pkt = std::mem::take(&mut self.pending_packet);
                    self.queued_packets.push_back(OggPacket {
                        bytes: pkt,
                        granule_position: -1,
                        is_eos: false,
                    });
                    completed_here += 1;
                }
            }

            if completed_here > 0 {
                if let Some(last) = self.queued_packets.back_mut() {
                    last.granule_position = granule;
                    last.is_eos = is_eos_page;
                }
            } else if is_eos_page && !self.pending_packet.is_empty() {
                let pkt = std::mem::take(&mut self.pending_packet);
                self.queued_packets.push_back(OggPacket {
                    bytes: pkt,
                    granule_position: granule,
                    is_eos: true,
                });
            }

            if is_eos_page {
                self.eos_reached = true;
            }
            return Ok(());
        }
    }
}

// ---------------------------------------------------------------------------
// BitReader (LSB-first Vorbis I)
// ---------------------------------------------------------------------------

#[inline]
const fn ilog(x: u32) -> u32 {
    32 - x.leading_zeros()
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
    eop: bool,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
            eop: false,
        }
    }

    #[inline]
    fn is_eop(&self) -> bool {
        self.eop
    }

    #[inline]
    fn read_bit_opt(&mut self) -> Option<u32> {
        if self.eop || self.byte_pos >= self.data.len() {
            self.eop = true;
            return None;
        }
        let bit = u32::from((self.data[self.byte_pos] >> self.bit_pos) & 1);
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit)
    }

    #[inline]
    fn read_bits_opt(&mut self, count: u32) -> Option<u32> {
        if count == 0 {
            return Some(0);
        }
        if count > 32 {
            self.eop = true;
            return None;
        }
        let mut out = 0u32;
        for shift in 0..count {
            let b = self.read_bit_opt()?;
            out |= b << shift;
        }
        Some(out)
    }

    fn read_bit_hdr(&mut self, err: &'static str) -> Result<u32, OggVorbisError> {
        self.read_bit_opt().ok_or(OggVorbisError::InvalidHeader(err))
    }

    fn read_bits_hdr(&mut self, count: u32, err: &'static str) -> Result<u32, OggVorbisError> {
        self.read_bits_opt(count)
            .ok_or(OggVorbisError::InvalidHeader(err))
    }

    fn read_u32_aligned_bytes(&mut self) -> Result<u32, OggVorbisError> {
        if self.bit_pos != 0 {
            return Err(OggVorbisError::InvalidHeader("expected byte alignment in comment header"));
        }
        if self.byte_pos + 4 > self.data.len() {
            return Err(OggVorbisError::InvalidHeader("unexpected EOF in comment header"));
        }
        let val = u32::from_le_bytes(
            self.data[self.byte_pos..self.byte_pos + 4]
                .try_into()
                .unwrap(),
        );
        self.byte_pos += 4;
        Ok(val)
    }

    fn read_bytes(&mut self, len: usize) -> Result<Vec<u8>, OggVorbisError> {
        if self.bit_pos != 0 {
            return Err(OggVorbisError::InvalidHeader("expected byte alignment for byte slice"));
        }
        if self.byte_pos.saturating_add(len) > self.data.len() {
            return Err(OggVorbisError::InvalidHeader("unexpected EOF reading byte slice"));
        }
        let slice = self.data[self.byte_pos..self.byte_pos + len].to_vec();
        self.byte_pos += len;
        Ok(slice)
    }
}

#[inline]
fn float32_unpack(x: u32) -> f32 {
    let mut mantissa = (x & 0x001f_ffff) as f64;
    let sign = (x & 0x8000_0000) != 0;
    let exponent = ((x & 0x7fe0_0000) >> 21) as i32;
    if sign {
        mantissa = -mantissa;
    }
    (mantissa * 2.0_f64.powi(exponent - 788)) as f32
}

fn lookup1_values(entries: usize, dimensions: usize) -> usize {
    if dimensions == 0 || entries == 0 {
        return 0;
    }
    let mut r = ((entries as f64).powf(1.0 / dimensions as f64)).floor() as usize;
    while (r + 1).checked_pow(dimensions as u32).is_some_and(|p| p <= entries) {
        r += 1;
    }
    while r.checked_pow(dimensions as u32).is_none_or(|p| p > entries) {
        r = r.saturating_sub(1);
    }
    r
}

// ---------------------------------------------------------------------------
// Vorbis Codebook (Huffman Binary Tree + VQ Lookup)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct HuffNode {
    /// Positive => index into `nodes`; negative => bitwise NOT of entry index; `-1`? We use `EMPTY = i32::MIN`.
    child: [i32; 2],
}

const HUFF_EMPTY: i32 = i32::MIN;

#[derive(Debug, Clone)]
struct Codebook {
    dimensions: usize,
    huff_nodes: Vec<HuffNode>,
    single_entry: Option<usize>,
    /// Flattened `[entries * dimensions]` VQ vectors for lookup_type 1 or 2 (`empty` for type 0).
    vq_table: Vec<f32>,
}

impl Codebook {
    fn parse(br: &mut BitReader<'_>) -> Result<Self, OggVorbisError> {
        let sync = br
            .read_bits_opt(24)
            .ok_or(OggVorbisError::InvalidCodebook("truncated codebook sync"))?;
        if sync != 0x564342 {
            return Err(OggVorbisError::InvalidCodebook("invalid codebook sync pattern"));
        }
        let dimensions = br
            .read_bits_opt(16)
            .ok_or(OggVorbisError::InvalidCodebook("truncated codebook dimensions"))?
            as usize;
        let entries = br
            .read_bits_opt(24)
            .ok_or(OggVorbisError::InvalidCodebook("truncated codebook entries"))?
            as usize;
        if entries > 2_000_000 {
            return Err(OggVorbisError::InvalidCodebook("excessive codebook entries"));
        }

        let ordered = br
            .read_bit_opt()
            .ok_or(OggVorbisError::InvalidCodebook("truncated ordered flag"))?
            != 0;
        let mut lengths = vec![0u8; entries];

        if !ordered {
            let sparse = br
                .read_bit_opt()
                .ok_or(OggVorbisError::InvalidCodebook("truncated sparse flag"))?
                != 0;
            for len in &mut lengths {
                let present = if sparse {
                    br.read_bit_opt()
                        .ok_or(OggVorbisError::InvalidCodebook("truncated sparse flag"))?
                        != 0
                } else {
                    true
                };
                if present {
                    let l = br
                        .read_bits_opt(5)
                        .ok_or(OggVorbisError::InvalidCodebook("truncated codeword length"))?
                        as u8
                        + 1;
                    *len = l;
                }
            }
        } else {
            let mut current_entry = 0usize;
            let mut current_length = br
                .read_bits_opt(5)
                .ok_or(OggVorbisError::InvalidCodebook("truncated initial length"))?
                as u8
                + 1;
            while current_entry < entries {
                let remaining = (entries - current_entry) as u32;
                let number = br
                    .read_bits_opt(ilog(remaining))
                    .ok_or(OggVorbisError::InvalidCodebook("truncated ordered run count"))?
                    as usize;
                if current_entry + number > entries {
                    return Err(OggVorbisError::InvalidCodebook("ordered run exceeds entries"));
                }
                if current_length > 32 && number > 0 {
                    return Err(OggVorbisError::InvalidCodebook("codeword length exceeds 32"));
                }
                for l in &mut lengths[current_entry..current_entry + number] {
                    *l = current_length;
                }
                current_entry += number;
                current_length = current_length.saturating_add(1);
            }
        }

        let (huff_nodes, single_entry) = build_vorbis_huffman_tree(&lengths)?;

        let lookup_type = br
            .read_bits_opt(4)
            .ok_or(OggVorbisError::InvalidCodebook("truncated VQ lookup_type"))?;
        let mut vq_table = Vec::new();
        if lookup_type > 2 {
            return Err(OggVorbisError::InvalidCodebook("unsupported VQ lookup_type > 2"));
        }
        if lookup_type == 1 || lookup_type == 2 {
            let minimum_value = float32_unpack(
                br.read_bits_opt(32)
                    .ok_or(OggVorbisError::InvalidCodebook("truncated VQ minimum_value"))?,
            );
            let delta_value = float32_unpack(
                br.read_bits_opt(32)
                    .ok_or(OggVorbisError::InvalidCodebook("truncated VQ delta_value"))?,
            );
            let value_bits = br
                .read_bits_opt(4)
                .ok_or(OggVorbisError::InvalidCodebook("truncated VQ value_bits"))?
                + 1;
            let sequence_p = br
                .read_bit_opt()
                .ok_or(OggVorbisError::InvalidCodebook("truncated VQ sequence_p"))?
                != 0;
            let lookup_values = if lookup_type == 1 {
                lookup1_values(entries, dimensions)
            } else {
                entries.saturating_mul(dimensions)
            };
            let mut multiplicands = Vec::with_capacity(lookup_values);
            for _ in 0..lookup_values {
                let m = br
                    .read_bits_opt(value_bits)
                    .ok_or(OggVorbisError::InvalidCodebook("truncated VQ multiplicand"))?;
                multiplicands.push(m);
            }

            let table_len = entries.saturating_mul(dimensions);
            if table_len > 4_000_000 {
                return Err(OggVorbisError::InvalidCodebook("excessive VQ table size"));
            }
            vq_table.resize(table_len, 0.0);
            if lookup_type == 1 && lookup_values > 0 && dimensions > 0 {
                for entry in 0..entries {
                    let mut last = 0.0f32;
                    let mut index_divisor = 1usize;
                    let base = entry * dimensions;
                    for d in 0..dimensions {
                        let off = (entry / index_divisor) % lookup_values;
                        let val = multiplicands[off] as f32 * delta_value + minimum_value + last;
                        vq_table[base + d] = val;
                        if sequence_p {
                            last = val;
                        }
                        index_divisor = index_divisor.saturating_mul(lookup_values).max(1);
                    }
                }
            } else if lookup_type == 2 && dimensions > 0 {
                for entry in 0..entries {
                    let mut last = 0.0f32;
                    let base = entry * dimensions;
                    for d in 0..dimensions {
                        let idx = base + d;
                        let mult = multiplicands.get(idx).copied().unwrap_or(0);
                        let val = mult as f32 * delta_value + minimum_value + last;
                        vq_table[idx] = val;
                        if sequence_p {
                            last = val;
                        }
                    }
                }
            }
        }

        Ok(Self {
            dimensions,
            huff_nodes,
            single_entry,
            vq_table,
        })
    }

    #[inline]
    fn decode_scalar(&self, br: &mut BitReader<'_>) -> Option<usize> {
        if let Some(single) = self.single_entry {
            return Some(single);
        }
        if self.huff_nodes.is_empty() {
            return None;
        }
        let mut node_idx = 0i32;
        loop {
            let bit = br.read_bit_opt()? as usize;
            let next = self.huff_nodes[node_idx as usize].child[bit];
            if next == HUFF_EMPTY {
                return None;
            }
            if next < 0 {
                return Some((!next) as usize);
            }
            node_idx = next;
        }
    }

    #[inline]
    fn vq_slice(&self, entry: usize) -> Option<&[f32]> {
        let start = entry.checked_mul(self.dimensions)?;
        let end = start.checked_add(self.dimensions)?;
        self.vq_table.get(start..end)
    }
}

/// Build the Vorbis I canonical Huffman binary tree (`marker[len]` prefix allocation).
/// Returns `(nodes, single_entry)`.
fn build_vorbis_huffman_tree(lengths: &[u8]) -> Result<(Vec<HuffNode>, Option<usize>), OggVorbisError> {
    let used_count = lengths.iter().filter(|&&l| l > 0).count();
    if used_count == 0 {
        return Ok((Vec::new(), Some(0)));
    }
    if used_count == 1 {
        let (idx, &len) = lengths
            .iter()
            .enumerate()
            .find(|(_, l)| **l > 0)
            .expect("used_count == 1");
        if len != 1 {
            // Degenerate single-entry codebooks with length 1 require 0 bits conceptually in some decoders;
            // still build the explicit 1-bit or multi-bit tree so real bitstreams consume exact bits.
            let mut nodes = vec![HuffNode {
                child: [HUFF_EMPTY, HUFF_EMPTY],
            }];
            insert_codeword_recursive(&mut nodes, 0, len, idx)?;
            return Ok((nodes, None));
        }
        let nodes = vec![HuffNode {
            child: [!(idx as i32), HUFF_EMPTY],
        }];
        return Ok((nodes, None));
    }

    let mut nodes = vec![HuffNode {
        child: [HUFF_EMPTY, HUFF_EMPTY],
    }];

    for (entry_idx, &len) in lengths.iter().enumerate() {
        if len == 0 {
            continue;
        }
        if len > 32 {
            return Err(OggVorbisError::InvalidCodebook("codeword length exceeds 32"));
        }
        if !insert_codeword_recursive(&mut nodes, 0, len, entry_idx)? {
            return Err(OggVorbisError::InvalidCodebook("oversubscribed Huffman tree"));
        }
    }

    Ok((nodes, None))
}

fn insert_codeword_recursive(
    nodes: &mut Vec<HuffNode>,
    node_idx: usize,
    depth_remaining: u8,
    entry_idx: usize,
) -> Result<bool, OggVorbisError> {
    if depth_remaining == 1 {
        for side in 0..2 {
            if nodes[node_idx].child[side] == HUFF_EMPTY {
                nodes[node_idx].child[side] = !(entry_idx as i32);
                return Ok(true);
            }
        }
        return Ok(false);
    }

    for side in 0..2 {
        let ch = nodes[node_idx].child[side];
        if ch == HUFF_EMPTY {
            let new_idx = nodes.len();
            if new_idx > 262_144 {
                return Err(OggVorbisError::InvalidCodebook("Huffman tree node limit exceeded"));
            }
            nodes.push(HuffNode {
                child: [HUFF_EMPTY, HUFF_EMPTY],
            });
            nodes[node_idx].child[side] = new_idx as i32;
            if insert_codeword_recursive(nodes, new_idx, depth_remaining - 1, entry_idx)? {
                return Ok(true);
            }
        } else if ch >= 0
            && insert_codeword_recursive(nodes, ch as usize, depth_remaining - 1, entry_idx)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

// ---------------------------------------------------------------------------
// Vorbis Floors (Floor 0 & Floor 1)
// ---------------------------------------------------------------------------

#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const FLOOR1_FROMDB_LOOKUP: [f32; 256] = [
    1.0649863e-07, 1.1341951e-07, 1.2079015e-07, 1.2863978e-07,
    1.3699951e-07, 1.4590251e-07, 1.5538408e-07, 1.6548181e-07,
    1.7623575e-07, 1.8768855e-07, 1.9988561e-07, 2.1287530e-07,
    2.2670913e-07, 2.4144197e-07, 2.5713223e-07, 2.7384213e-07,
    2.9163793e-07, 3.1059021e-07, 3.3077411e-07, 3.5226968e-07,
    3.7516214e-07, 3.9954229e-07, 4.2550680e-07, 4.5315863e-07,
    4.8260743e-07, 5.1396998e-07, 5.4737065e-07, 5.8294187e-07,
    6.2082472e-07, 6.6116941e-07, 7.0413592e-07, 7.4989464e-07,
    7.9862701e-07, 8.5052630e-07, 9.0579828e-07, 9.6466216e-07,
    1.0273513e-06, 1.0941144e-06, 1.1652161e-06, 1.2409384e-06,
    1.3215816e-06, 1.4074654e-06, 1.4989305e-06, 1.5963394e-06,
    1.7000785e-06, 1.8105592e-06, 1.9282195e-06, 2.0535261e-06,
    2.1869758e-06, 2.3290978e-06, 2.4804557e-06, 2.6416497e-06,
    2.8133190e-06, 2.9961443e-06, 3.1908506e-06, 3.3982101e-06,
    3.6190449e-06, 3.8542308e-06, 4.1047004e-06, 4.3714470e-06,
    4.6555282e-06, 4.9580707e-06, 5.2802740e-06, 5.6234160e-06,
    5.9888572e-06, 6.3780469e-06, 6.7925283e-06, 7.2339451e-06,
    7.7040476e-06, 8.2047000e-06, 8.7378876e-06, 9.3057248e-06,
    9.9104632e-06, 1.0554501e-05, 1.1240392e-05, 1.1970856e-05,
    1.2748789e-05, 1.3577278e-05, 1.4459606e-05, 1.5399272e-05,
    1.6400004e-05, 1.7465768e-05, 1.8600792e-05, 1.9809576e-05,
    2.1096914e-05, 2.2467911e-05, 2.3928002e-05, 2.5482978e-05,
    2.7139006e-05, 2.8902651e-05, 3.0780908e-05, 3.2781225e-05,
    3.4911534e-05, 3.7180282e-05, 3.9596466e-05, 4.2169667e-05,
    4.4910090e-05, 4.7828601e-05, 5.0936773e-05, 5.4246931e-05,
    5.7772202e-05, 6.1526565e-05, 6.5524908e-05, 6.9783085e-05,
    7.4317983e-05, 7.9147585e-05, 8.4291040e-05, 8.9768747e-05,
    9.5602426e-05, 0.00010181521, 0.00010843174, 0.00011547824,
    0.00012298267, 0.00013097477, 0.00013948625, 0.00014855085,
    0.00015820453, 0.00016848555, 0.00017943469, 0.00019109536,
    0.00020351382, 0.00021673929, 0.00023082423, 0.00024582449,
    0.00026179955, 0.00027881276, 0.00029693158, 0.00031622787,
    0.00033677814, 0.00035866388, 0.00038197188, 0.00040679456,
    0.00043323036, 0.00046138411, 0.00049136745, 0.00052329927,
    0.00055730621, 0.00059352311, 0.00063209358, 0.00067317058,
    0.00071691700, 0.00076350630, 0.00081312324, 0.00086596457,
    0.00092223983, 0.00098217216, 0.0010459992,  0.0011139742,
    0.0011863665,  0.0012634633,  0.0013455702,  0.0014330129,
    0.0015261382,  0.0016253153,  0.0017309374,  0.0018434235,
    0.0019632195,  0.0020908006,  0.0022266726,  0.0023713743,
    0.0025254795,  0.0026895994,  0.0028643847,  0.0030505286,
    0.0032487691,  0.0034598925,  0.0036847358,  0.0039241906,
    0.0041792066,  0.0044507950,  0.0047400328,  0.0050480668,
    0.0053761186,  0.0057254891,  0.0060975636,  0.0064938176,
    0.0069158225,  0.0073652516,  0.0078438871,  0.0083536271,
    0.0088964928,  0.009474637,   0.010090352,   0.010746080,
    0.011444421,   0.012188144,   0.012980198,   0.013823725,
    0.014722068,   0.015678791,   0.016697687,   0.017782797,
    0.018938423,   0.020169149,   0.021479854,   0.022875735,
    0.024362330,   0.025945531,   0.027631618,   0.029427276,
    0.031339626,   0.033376252,   0.035545228,   0.037855157,
    0.040315199,   0.042935108,   0.045725273,   0.048696758,
    0.051861348,   0.055231591,   0.058820850,   0.062643361,
    0.066714279,   0.071049749,   0.075666962,   0.080584227,
    0.085821044,   0.091398179,   0.097337747,   0.10366330,
    0.11039993,    0.11757434,    0.12521498,    0.13335215,
    0.14201813,    0.15124727,    0.16107617,    0.17154380,
    0.18269168,    0.19456402,    0.20720788,    0.22067342,
    0.23501402,    0.25028656,    0.26655159,    0.28387361,
    0.30232132,    0.32196786,    0.34289114,    0.36517414,
    0.38890521,    0.41417847,    0.44109412,    0.46975890,
    0.50028648,    0.53279791,    0.56742212,    0.60429640,
    0.64356699,    0.68538959,    0.72993007,    0.77736504,
    0.82788260,    0.88168307,    0.9389798,     1.0,
];

#[derive(Debug, Clone)]
struct Floor0 {
    order: usize,
    rate: f64,
    bark_map_size: usize,
    amplitude_bits: u32,
    amplitude_offset: u32,
    books: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Floor1Class {
    dimensions: usize,
    subclasses: u32,
    masterbook: usize,
    subclass_books: Vec<i32>,
}

#[derive(Debug, Clone)]
struct Floor1 {
    partition_classes: Vec<usize>,
    classes: Vec<Floor1Class>,
    multiplier: i32,
    range_bits: u32,
    x_list: Vec<i32>,
    sorted_indices: Vec<usize>,
    neighbors: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
enum Floor {
    Type0(Floor0),
    Type1(Floor1),
}

impl Floor {
    fn parse(br: &mut BitReader<'_>, num_books: usize) -> Result<Self, OggVorbisError> {
        let floor_type = br
            .read_bits_opt(16)
            .ok_or(OggVorbisError::InvalidFloor("truncated floor_type"))?;
        match floor_type {
            0 => {
                let order = br
                    .read_bits_opt(8)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor0 order"))?
                    as usize;
                let rate = br
                    .read_bits_opt(16)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor0 rate"))?
                    as f64;
                let bark_map_size = br
                    .read_bits_opt(16)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor0 bark_map_size"))?
                    as usize;
                let amplitude_bits = br
                    .read_bits_opt(6)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor0 amplitude_bits"))?;
                let amplitude_offset = br
                    .read_bits_opt(8)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor0 amplitude_offset"))?;
                let number_of_books = br
                    .read_bits_opt(4)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor0 number_of_books"))?
                    as usize
                    + 1;
                let mut books = Vec::with_capacity(number_of_books);
                for _ in 0..number_of_books {
                    let b = br
                        .read_bits_opt(8)
                        .ok_or(OggVorbisError::InvalidFloor("truncated floor0 book"))?
                        as usize;
                    if b >= num_books {
                        return Err(OggVorbisError::InvalidFloor("floor0 book index out of range"));
                    }
                    books.push(b);
                }
                Ok(Self::Type0(Floor0 {
                    order,
                    rate,
                    bark_map_size: bark_map_size.max(1),
                    amplitude_bits,
                    amplitude_offset,
                    books,
                }))
            }
            1 => {
                let partitions = br
                    .read_bits_opt(5)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor1 partitions"))?
                    as usize;
                let mut partition_classes = Vec::with_capacity(partitions);
                let mut max_class = None::<usize>;
                for _ in 0..partitions {
                    let c = br
                        .read_bits_opt(4)
                        .ok_or(OggVorbisError::InvalidFloor("truncated floor1 partition class"))?
                        as usize;
                    max_class = Some(max_class.map_or(c, |m| m.max(c)));
                    partition_classes.push(c);
                }
                let class_count = max_class.map_or(0, |m| m + 1);
                let mut classes = Vec::with_capacity(class_count);
                for _ in 0..class_count {
                    let dimensions = br
                        .read_bits_opt(3)
                        .ok_or(OggVorbisError::InvalidFloor("truncated floor1 class dim"))?
                        as usize
                        + 1;
                    let subclasses = br
                        .read_bits_opt(2)
                        .ok_or(OggVorbisError::InvalidFloor("truncated floor1 subclasses"))?;
                    let masterbook = if subclasses > 0 {
                        let mb = br
                            .read_bits_opt(8)
                            .ok_or(OggVorbisError::InvalidFloor("truncated floor1 masterbook"))?
                            as usize;
                        if mb >= num_books {
                            return Err(OggVorbisError::InvalidFloor("floor1 masterbook out of range"));
                        }
                        mb
                    } else {
                        0
                    };
                    let sub_count = 1usize << subclasses;
                    let mut subclass_books = Vec::with_capacity(sub_count);
                    for _ in 0..sub_count {
                        let raw = br
                            .read_bits_opt(8)
                            .ok_or(OggVorbisError::InvalidFloor("truncated floor1 subclass book"))?
                            as i32
                            - 1;
                        if raw >= num_books as i32 {
                            return Err(OggVorbisError::InvalidFloor("floor1 subclass book out of range"));
                        }
                        subclass_books.push(raw);
                    }
                    classes.push(Floor1Class {
                        dimensions,
                        subclasses,
                        masterbook,
                        subclass_books,
                    });
                }

                let multiplier = br
                    .read_bits_opt(2)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor1 multiplier"))?
                    as i32
                    + 1;
                let range_bits = br
                    .read_bits_opt(4)
                    .ok_or(OggVorbisError::InvalidFloor("truncated floor1 range_bits"))?;

                let mut x_list = Vec::with_capacity(65);
                x_list.push(0);
                x_list.push(1i32 << range_bits);
                for &cls in &partition_classes {
                    let dim = classes[cls].dimensions;
                    for _ in 0..dim {
                        if x_list.len() > 65 {
                            return Err(OggVorbisError::InvalidFloor("floor1 X_list exceeded 65"));
                        }
                        let x = br
                            .read_bits_opt(range_bits)
                            .ok_or(OggVorbisError::InvalidFloor("truncated floor1 X coord"))?
                            as i32;
                        x_list.push(x);
                    }
                }

                let values = x_list.len();
                let mut neighbors = vec![(0usize, 0usize); values];
                for (i, slot) in neighbors.iter_mut().enumerate().skip(2) {
                    *slot = (low_neighbor(&x_list, i), high_neighbor(&x_list, i));
                }

                let mut sorted_indices: Vec<usize> = (0..values).collect();
                sorted_indices.sort_by_key(|&idx| x_list[idx]);

                Ok(Self::Type1(Floor1 {
                    partition_classes,
                    classes,
                    multiplier,
                    range_bits,
                    x_list,
                    sorted_indices,
                    neighbors,
                }))
            }
            _ => Err(OggVorbisError::InvalidFloor("unsupported floor_type > 1")),
        }
    }

    /// Decode and evaluate floor curve into `out_floor[0..half_n]`.
    /// Returns `true` if channel is active (`nonzero`), `false` if silent.
    fn decode_and_render(
        &self,
        br: &mut BitReader<'_>,
        codebooks: &[Codebook],
        half_n: usize,
        out_floor: &mut [f32],
    ) -> bool {
        match self {
            Self::Type0(f0) => f0.decode_and_render(br, codebooks, half_n, out_floor),
            Self::Type1(f1) => f1.decode_and_render(br, codebooks, half_n, out_floor),
        }
    }
}

#[inline]
fn bark(f: f64) -> f64 {
    13.1 * (0.00074 * f).atan() + 2.24 * (0.000_000_018_5 * f * f).atan() + 1.0e-4 * f
}

impl Floor0 {
    fn decode_and_render(
        &self,
        br: &mut BitReader<'_>,
        codebooks: &[Codebook],
        half_n: usize,
        out_floor: &mut [f32],
    ) -> bool {
        let Some(amplitude) = br.read_bits_opt(self.amplitude_bits) else {
            return false;
        };
        if amplitude == 0 {
            return false;
        }
        let book_bits = ilog((self.books.len() as u32).saturating_sub(1));
        let Some(book_idx) = br.read_bits_opt(book_bits) else {
            return false;
        };
        let Some(&cb_id) = self.books.get(book_idx as usize) else {
            return false;
        };
        let cb = &codebooks[cb_id];
        if cb.dimensions == 0 {
            return false;
        }

        let mut coeffs = Vec::with_capacity(self.order + cb.dimensions);
        let mut last = 0.0f32;
        while coeffs.len() < self.order {
            let Some(entry) = cb.decode_scalar(br) else {
                return false;
            };
            let Some(vq) = cb.vq_slice(entry) else {
                return false;
            };
            for &v in vq {
                coeffs.push(v + last);
            }
            if let Some(&tail) = coeffs.last() {
                last = tail;
            }
        }
        coeffs.truncate(self.order);

        let amp_max = ((1u64 << self.amplitude_bits).saturating_sub(1)).max(1) as f64;
        let db_linear =
            amplitude as f64 * (self.amplitude_offset as f64 / amp_max);
        let foobar_max = bark(0.5 * self.rate) / self.bark_map_size as f64;

        for (i, out_slot) in out_floor.iter_mut().enumerate().take(half_n) {
            let mut bark_idx = ((bark(0.5 * self.rate * i as f64 / half_n as f64) / foobar_max)
                .floor() as usize)
                .min(self.bark_map_size.saturating_sub(1));
            if i == half_n - 1 && self.bark_map_size > 1 {
                bark_idx = bark_idx.min(self.bark_map_size - 1);
            }
            let omega = PI * bark_idx as f64 / self.bark_map_size as f64;
            let cos_w = omega.cos();
            let mut p = 1.0f64;
            let mut q = 1.0f64;
            for j in (0..self.order).step_by(2) {
                let c0 = f64::from(coeffs[j]).cos();
                let c1 = if j + 1 < self.order {
                    f64::from(coeffs[j + 1]).cos()
                } else {
                    0.0
                };
                q *= 4.0 * (c0 - cos_w) * (c0 - cos_w);
                if j + 1 < self.order {
                    p *= 4.0 * (c1 - cos_w) * (c1 - cos_w);
                }
            }
            if (self.order & 1) == 1 {
                p *= 2.0 * (1.0 - cos_w * cos_w);
            } else {
                p *= 2.0 * (1.0 - cos_w);
                q *= 2.0 * (1.0 + cos_w);
            }
            let lsp = (p + q).max(1e-18).sqrt();
            let val = (0.115_129_25 * (db_linear / lsp - f64::from(self.amplitude_offset))).exp();
            *out_slot = val as f32;
        }
        true
    }
}

#[inline]
fn low_neighbor(v: &[i32], x: usize) -> usize {
    let mut best = 0usize;
    let mut best_val = i32::MIN;
    let target = v[x];
    for (i, &vi) in v[..x].iter().enumerate() {
        if vi < target && vi > best_val {
            best = i;
            best_val = vi;
        }
    }
    best
}

#[inline]
fn high_neighbor(v: &[i32], x: usize) -> usize {
    let mut best = 1usize;
    let mut best_val = i32::MAX;
    let target = v[x];
    for (i, &vi) in v[..x].iter().enumerate() {
        if vi > target && vi < best_val {
            best = i;
            best_val = vi;
        }
    }
    best
}

#[inline]
fn render_point(x0: i32, y0: i32, x1: i32, y1: i32, x: i32) -> i32 {
    let dy = y1 - y0;
    let adx = x1 - x0;
    if adx == 0 {
        return y0;
    }
    let ady = dy.abs();
    let err = ady * (x - x0);
    let off = err / adx;
    if dy < 0 {
        y0 - off
    } else {
        y0 + off
    }
}

#[inline]
fn render_line(x0: i32, y0: i32, x1: i32, y1: i32, out: &mut [f32]) {
    let dy = y1 - y0;
    let adx = (x1 - x0).max(1);
    let ady = dy.abs();
    let base = dy / adx;
    let sy = if dy < 0 { base - 1 } else { base + 1 };
    let ady_rem = ady - base.abs() * adx;
    let mut err = 0i32;
    let mut x = x0;
    let mut y = y0;

    if x >= 0 && (x as usize) < out.len() {
        out[x as usize] = FLOOR1_FROMDB_LOOKUP[(y.clamp(0, 255)) as usize];
    }
    while x + 1 < x1 {
        x += 1;
        err += ady_rem;
        if err >= adx {
            err -= adx;
            y += sy;
        } else {
            y += base;
        }
        if x >= 0 && (x as usize) < out.len() {
            out[x as usize] = FLOOR1_FROMDB_LOOKUP[(y.clamp(0, 255)) as usize];
        }
    }
}

impl Floor1 {
    fn decode_and_render(
        &self,
        br: &mut BitReader<'_>,
        codebooks: &[Codebook],
        half_n: usize,
        out_floor: &mut [f32],
    ) -> bool {
        let Some(nonzero) = br.read_bit_opt() else {
            return false;
        };
        if nonzero == 0 {
            return false;
        }

        let range = [256i32, 128, 86, 64][(self.multiplier - 1).clamp(0, 3) as usize];
        let y_bits = ilog((range - 1) as u32);
        let values = self.x_list.len();
        let mut y_vals = [0i32; 66];
        let Some(y0) = br.read_bits_opt(y_bits) else {
            return false;
        };
        let Some(y1) = br.read_bits_opt(y_bits) else {
            return false;
        };
        y_vals[0] = y0 as i32;
        y_vals[1] = y1 as i32;

        let mut offset = 2usize;
        for &cls_idx in &self.partition_classes {
            let cls = &self.classes[cls_idx];
            let cdim = cls.dimensions;
            let cbits = cls.subclasses;
            let csub = (1usize << cbits) - 1;
            let mut cval = if cbits > 0 {
                let Some(v) = codebooks[cls.masterbook].decode_scalar(br) else {
                    return false;
                };
                v
            } else {
                0
            };

            for j in 0..cdim {
                let book = cls.subclass_books[cval & csub];
                cval >>= cbits;
                if book >= 0 {
                    let Some(y) = codebooks[book as usize].decode_scalar(br) else {
                        return false;
                    };
                    y_vals[offset + j] = y as i32;
                } else {
                    y_vals[offset + j] = 0;
                }
            }
            offset += cdim;
        }

        // Step 1: unwrap amplitude predictions
        let mut step2_flag = [false; 66];
        step2_flag[0] = true;
        step2_flag[1] = true;
        let mut final_y = [0i32; 66];
        final_y[0] = y_vals[0];
        final_y[1] = y_vals[1];

        for i in 2..values {
            let (low_idx, high_idx) = self.neighbors[i];
            let predicted = render_point(
                self.x_list[low_idx],
                final_y[low_idx],
                self.x_list[high_idx],
                final_y[high_idx],
                self.x_list[i],
            );
            let val = y_vals[i];
            let highroom = range - predicted;
            let lowroom = predicted;
            let room = highroom.min(lowroom) * 2;
            if val != 0 {
                step2_flag[low_idx] = true;
                step2_flag[high_idx] = true;
                step2_flag[i] = true;
                if val >= room {
                    final_y[i] = if highroom > lowroom {
                        val - lowroom + predicted
                    } else {
                        predicted - val + highroom - 1
                    };
                } else if (val & 1) != 0 {
                    final_y[i] = predicted - ((val + 1) >> 1);
                } else {
                    final_y[i] = predicted + (val >> 1);
                }
            } else {
                step2_flag[i] = false;
                final_y[i] = predicted;
            }
        }

        // Step 2: curve synthesis in sorted X order
        out_floor[..half_n].fill(0.0);
        let first_idx = self.sorted_indices[0];
        let mut hx = 0i32;
        let mut lx = 0i32;
        let mut ly = final_y[first_idx] * self.multiplier;

        for &idx in &self.sorted_indices[1..] {
            if step2_flag[idx] {
                let hy = final_y[idx] * self.multiplier;
                hx = self.x_list[idx];
                render_line(lx, ly, hx, hy, &mut out_floor[..half_n]);
                lx = hx;
                ly = hy;
            }
        }
        if (hx as usize) < half_n {
            render_line(hx, ly, half_n as i32, ly, &mut out_floor[..half_n]);
        }
        let _ = self.range_bits;
        true
    }
}

// ---------------------------------------------------------------------------
// Vorbis Residues (Types 0, 1, and 2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Residue {
    residue_type: u16,
    begin: usize,
    end: usize,
    partition_size: usize,
    classifications: usize,
    classbook: usize,
    books: Vec<[i32; 8]>,
}

impl Residue {
    fn parse(br: &mut BitReader<'_>, num_books: usize) -> Result<Self, OggVorbisError> {
        let residue_type = br
            .read_bits_opt(16)
            .ok_or(OggVorbisError::InvalidResidue("truncated residue_type"))?
            as u16;
        if residue_type > 2 {
            return Err(OggVorbisError::InvalidResidue("unsupported residue_type > 2"));
        }
        let begin = br
            .read_bits_opt(24)
            .ok_or(OggVorbisError::InvalidResidue("truncated residue begin"))?
            as usize;
        let end = br
            .read_bits_opt(24)
            .ok_or(OggVorbisError::InvalidResidue("truncated residue end"))?
            as usize;
        let partition_size = br
            .read_bits_opt(24)
            .ok_or(OggVorbisError::InvalidResidue("truncated partition_size"))?
            as usize
            + 1;
        let classifications = br
            .read_bits_opt(6)
            .ok_or(OggVorbisError::InvalidResidue("truncated classifications"))?
            as usize
            + 1;
        let classbook = br
            .read_bits_opt(8)
            .ok_or(OggVorbisError::InvalidResidue("truncated classbook"))?
            as usize;
        if classbook >= num_books {
            return Err(OggVorbisError::InvalidResidue("residue classbook out of range"));
        }

        let mut cascade = Vec::with_capacity(classifications);
        for _ in 0..classifications {
            let low = br
                .read_bits_opt(3)
                .ok_or(OggVorbisError::InvalidResidue("truncated cascade low"))?;
            let bitflag = br
                .read_bit_opt()
                .ok_or(OggVorbisError::InvalidResidue("truncated cascade bitflag"))?;
            let high = if bitflag != 0 {
                br.read_bits_opt(5)
                    .ok_or(OggVorbisError::InvalidResidue("truncated cascade high"))?
            } else {
                0
            };
            cascade.push((high << 3) | low);
        }

        let mut books = vec![[-1i32; 8]; classifications];
        for (i, &casc) in cascade.iter().enumerate() {
            for (bit, book_slot) in books[i].iter_mut().enumerate() {
                if (casc & (1 << bit)) != 0 {
                    let b = br
                        .read_bits_opt(8)
                        .ok_or(OggVorbisError::InvalidResidue("truncated residue book"))?
                        as usize;
                    if b >= num_books {
                        return Err(OggVorbisError::InvalidResidue("residue book out of range"));
                    }
                    *book_slot = b as i32;
                }
            }
        }

        Ok(Self {
            residue_type,
            begin,
            end,
            partition_size,
            classifications,
            classbook,
            books,
        })
    }

    fn decode(
        &self,
        br: &mut BitReader<'_>,
        codebooks: &[Codebook],
        do_not_decode: &[bool],
        half_n: usize,
        out_vecs: &mut [&mut [f32]],
    ) {
        let ch = out_vecs.len();
        if ch == 0 || br.is_eop() {
            return;
        }
        if self.residue_type == 2 {
            // Spec 8.6.2: If all channels are unused, skip residue decode altogether.
            if do_not_decode.iter().all(|&d| d) {
                return;
            }
            let mut flat = vec![0.0f32; half_n * ch];
            self.decode_vectors(br, codebooks, &[false], half_n * ch, &mut [&mut flat]);
            for (c_idx, out) in out_vecs.iter_mut().enumerate() {
                if !do_not_decode[c_idx] {
                    for (i, dst) in out[..half_n].iter_mut().enumerate() {
                        *dst += flat[i * ch + c_idx];
                    }
                }
            }
            return;
        }

        self.decode_vectors(br, codebooks, do_not_decode, half_n, out_vecs);
    }

    fn decode_vectors(
        &self,
        br: &mut BitReader<'_>,
        codebooks: &[Codebook],
        do_not_decode: &[bool],
        vec_len: usize,
        out_vecs: &mut [&mut [f32]],
    ) {
        let actual_end = self.end.min(vec_len);
        if self.begin >= actual_end || self.partition_size == 0 {
            return;
        }
        let n_to_read = actual_end - self.begin;
        let partitions_to_read = n_to_read / self.partition_size;
        if partitions_to_read == 0 {
            return;
        }

        let ch = out_vecs.len();
        let classbook = &codebooks[self.classbook];
        let classwords_per_codeword = classbook.dimensions.max(1);
        let mut class_labels = vec![vec![0usize; partitions_to_read]; ch];

        'passes: for pass in 0..8 {
            let mut partition_count = 0usize;
            while partition_count < partitions_to_read {
                if pass == 0 {
                    for j in 0..ch {
                        if do_not_decode[j] {
                            continue;
                        }
                        if partition_count.is_multiple_of(classwords_per_codeword) {
                            let Some(mut temp) = classbook.decode_scalar(br) else {
                                break 'passes;
                            };
                            // MSB-first base-classifications extraction per Vorbis spec
                            for d in (0..classwords_per_codeword).rev() {
                                let p_idx = partition_count + d;
                                let label = temp % self.classifications;
                                temp /= self.classifications;
                                if p_idx < partitions_to_read {
                                    class_labels[j][p_idx] = label;
                                }
                            }
                        }
                    }
                }

                for _d in 0..classwords_per_codeword {
                    if partition_count >= partitions_to_read {
                        break;
                    }
                    let offset = self.begin + partition_count * self.partition_size;
                    for j in 0..ch {
                        if do_not_decode[j] {
                            continue;
                        }
                        let vqclass = class_labels[j][partition_count];
                        let vqbook = self.books[vqclass][pass];
                        if vqbook >= 0 {
                            let cb = &codebooks[vqbook as usize];
                            if cb.dimensions == 0 {
                                continue;
                            }
                            if self.residue_type == 0 {
                                let step = (self.partition_size / cb.dimensions).max(1);
                                for i in 0..step {
                                    let Some(entry) = cb.decode_scalar(br) else {
                                        break 'passes;
                                    };
                                    let Some(vq) = cb.vq_slice(entry) else {
                                        break 'passes;
                                    };
                                    for (k, &val) in vq.iter().enumerate() {
                                        let idx = offset + i + k * step;
                                        if idx < actual_end {
                                            out_vecs[j][idx] += val;
                                        }
                                    }
                                }
                            } else {
                                // Type 1 and flattened Type 2
                                let mut i = 0usize;
                                while i < self.partition_size {
                                    let Some(entry) = cb.decode_scalar(br) else {
                                        break 'passes;
                                    };
                                    let Some(vq) = cb.vq_slice(entry) else {
                                        break 'passes;
                                    };
                                    for &val in vq {
                                        let idx = offset + i;
                                        if i < self.partition_size && idx < actual_end {
                                            out_vecs[j][idx] += val;
                                        }
                                        i += 1;
                                    }
                                }
                            }
                        }
                    }
                    partition_count += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mapping & Mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mapping {
    coupling_steps: Vec<(usize, usize)>,
    mux: Vec<usize>,
    submaps: Vec<(usize, usize)>, // (floor_idx, residue_idx)
}

impl Mapping {
    fn parse(
        br: &mut BitReader<'_>,
        channels: usize,
        num_floors: usize,
        num_residues: usize,
    ) -> Result<Self, OggVorbisError> {
        let mapping_type = br
            .read_bits_opt(16)
            .ok_or(OggVorbisError::InvalidMapping("truncated mapping_type"))?;
        if mapping_type != 0 {
            return Err(OggVorbisError::InvalidMapping("unsupported mapping_type != 0"));
        }

        let submap_flag = br
            .read_bit_opt()
            .ok_or(OggVorbisError::InvalidMapping("truncated submap flag"))?
            != 0;
        let submaps_count = if submap_flag {
            br.read_bits_opt(4)
                .ok_or(OggVorbisError::InvalidMapping("truncated submaps count"))?
                as usize
                + 1
        } else {
            1
        };

        let coupling_flag = br
            .read_bit_opt()
            .ok_or(OggVorbisError::InvalidMapping("truncated coupling flag"))?
            != 0;
        let mut coupling_steps = Vec::new();
        if coupling_flag {
            let steps = br
                .read_bits_opt(8)
                .ok_or(OggVorbisError::InvalidMapping("truncated coupling_steps"))?
                as usize
                + 1;
            let ch_bits = ilog((channels as u32).saturating_sub(1));
            for _ in 0..steps {
                let mag = br
                    .read_bits_opt(ch_bits)
                    .ok_or(OggVorbisError::InvalidMapping("truncated coupling mag"))?
                    as usize;
                let ang = br
                    .read_bits_opt(ch_bits)
                    .ok_or(OggVorbisError::InvalidMapping("truncated coupling ang"))?
                    as usize;
                if mag == ang || mag >= channels || ang >= channels {
                    return Err(OggVorbisError::InvalidMapping("invalid channel coupling pair"));
                }
                coupling_steps.push((mag, ang));
            }
        }

        let reserved = br
            .read_bits_opt(2)
            .ok_or(OggVorbisError::InvalidMapping("truncated reserved field"))?;
        if reserved != 0 {
            return Err(OggVorbisError::InvalidMapping("non-zero reserved mapping bits"));
        }

        let mut mux = vec![0usize; channels];
        if submaps_count > 1 {
            for m in &mut mux {
                let val = br
                    .read_bits_opt(4)
                    .ok_or(OggVorbisError::InvalidMapping("truncated mapping mux"))?
                    as usize;
                if val >= submaps_count {
                    return Err(OggVorbisError::InvalidMapping("mapping mux >= submaps"));
                }
                *m = val;
            }
        }

        let mut submaps = Vec::with_capacity(submaps_count);
        for _ in 0..submaps_count {
            let _time = br
                .read_bits_opt(8)
                .ok_or(OggVorbisError::InvalidMapping("truncated submap time"))?;
            let floor_idx = br
                .read_bits_opt(8)
                .ok_or(OggVorbisError::InvalidMapping("truncated submap floor"))?
                as usize;
            let res_idx = br
                .read_bits_opt(8)
                .ok_or(OggVorbisError::InvalidMapping("truncated submap residue"))?
                as usize;
            if floor_idx >= num_floors || res_idx >= num_residues {
                return Err(OggVorbisError::InvalidMapping("submap floor or residue index out of range"));
            }
            submaps.push((floor_idx, res_idx));
        }

        Ok(Self {
            coupling_steps,
            mux,
            submaps,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct Mode {
    blockflag: bool,
    mapping: usize,
}

// ---------------------------------------------------------------------------
// Fast O(N log N) Vorbis IMDCT + Overlap-Add Windowing
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct Cpx {
    re: f64,
    im: f64,
}

impl Cpx {
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Unnormalized complex FFT with selectable sign (`inverse = true` computes `+2*pi*m*q/N`).
fn fft_inplace(buf: &mut [Cpx], inverse: bool) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while (j & bit) != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2usize;
    while len <= n {
        let half = len >> 1;
        let angle = sign * (2.0 * PI) / len as f64;
        let wlen = Cpx::new(angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let mut w = Cpx::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[start + k];
                let v = buf[start + k + half].mul(w);
                buf[start + k] = Cpx::new(u.re + v.re, u.im + v.im);
                buf[start + k + half] = Cpx::new(u.re - v.re, u.im - v.im);
                w = w.mul(wlen);
            }
        }
        len <<= 1;
    }
}

/// Fast `O(N log N)` Vorbis I Inverse MDCT (`N` power of 2 in `8..=8192`).
/// Computes `y[k] = sum_{i=0}^{N/2-1} X[i] * cos(pi / (2*N) * (2*k + 1 + N/2) * (2*i + 1))`.
pub fn vorbis_imdct_fast(spectrum: &[f32], out: &mut [f32]) {
    let n = out.len();
    debug_assert!(n.is_power_of_two() && n >= 8 && spectrum.len() >= n / 2);
    let m = n / 2;
    let n4 = n / 4;
    let two_n = (2 * n) as f64;

    let mut work = vec![Cpx::default(); n4];
    for (idx, w) in work.iter_mut().enumerate() {
        let x_even = f64::from(spectrum[2 * idx]);
        let x_odd = f64::from(spectrum[m - 1 - 2 * idx]);
        let theta = PI * (4 * idx + 1) as f64 / two_n;
        let pre = Cpx::new(theta.cos(), theta.sin());
        *w = Cpx::new(x_even, -x_odd).mul(pre);
    }

    fft_inplace(&mut work, true);

    let mut dct4 = vec![0.0f32; m];
    for (q, &v) in work.iter().enumerate() {
        let theta = (2.0 * PI * q as f64) / n as f64;
        let w = v.mul(Cpx::new(theta.cos(), theta.sin()));
        dct4[2 * q] = w.re as f32;
        dct4[m - 1 - 2 * q] = w.im as f32;
    }

    // Fold DCT-IV into full Vorbis IMDCT vector using half-window symmetries:
    // - [0..N/4)    = +dct4[N/4..N/2)
    // - [N/4..N/2)  = -out[N/2-1-k]
    // - [N/2..3N/4) = -dct4[3N/4-1-k]
    // - [3N/4..N)   = +out[N/2 + (N-1-k)]
    for k in 0..n4 {
        let v0 = dct4[n4 + k];
        out[k] = v0;
        out[m - 1 - k] = -v0;

        let v1 = -dct4[n4 - 1 - k];
        out[m + k] = v1;
        out[n - 1 - k] = v1;
    }
}

fn build_vorbis_slope_window(half_n: usize) -> Vec<f32> {
    let mut win = Vec::with_capacity(half_n);
    for i in 0..half_n {
        let s = ((i as f64 + 0.5) / (2.0 * half_n as f64) * PI).sin();
        let w = (FRAC_PI_2 * s * s).sin();
        win.push(w as f32);
    }
    win
}

// ---------------------------------------------------------------------------
// Complete Vorbis I Stream Decoder
// ---------------------------------------------------------------------------

pub struct VorbisStreamDecoder {
    demuxer: OggDemuxer,
    channels: u8,
    sample_rate: u32,
    blocksize_0: usize,
    blocksize_1: usize,
    vendor: String,
    comments: Vec<Vec<u8>>,
    xray_params: OggXRayParams,
    codebooks: Vec<Codebook>,
    floors: Vec<Floor>,
    residues: Vec<Residue>,
    mappings: Vec<Mapping>,
    modes: Vec<Mode>,
    window_short: Vec<f32>,
    window_long: Vec<f32>,
    overlap_bufs: Vec<Vec<f32>>,
    prev_blocksize: Option<usize>,
    finished: bool,
}

impl VorbisStreamDecoder {
    pub fn from_owned_bytes(bytes: Vec<u8>) -> Result<Self, OggVorbisError> {
        let mut demuxer = OggDemuxer::new(bytes);

        let pkt0 = demuxer
            .next_packet()?
            .ok_or(OggVorbisError::InvalidHeader("missing Vorbis identification packet"))?;
        let (channels, sample_rate, blocksize_0, blocksize_1) = parse_id_packet(&pkt0.bytes)?;

        let pkt1 = demuxer
            .next_packet()?
            .ok_or(OggVorbisError::InvalidHeader("missing Vorbis comment packet"))?;
        let (vendor, comments, xray_params) = parse_comment_packet(&pkt1.bytes)?;

        let pkt2 = demuxer
            .next_packet()?
            .ok_or(OggVorbisError::InvalidHeader("missing Vorbis setup packet"))?;
        let (codebooks, floors, residues, mappings, modes) =
            parse_setup_packet(&pkt2.bytes, usize::from(channels))?;

        let window_short = build_vorbis_slope_window(blocksize_0 / 2);
        let window_long = build_vorbis_slope_window(blocksize_1 / 2);
        let overlap_bufs = vec![Vec::new(); usize::from(channels)];

        Ok(Self {
            demuxer,
            channels,
            sample_rate,
            blocksize_0,
            blocksize_1,
            vendor,
            comments,
            xray_params,
            codebooks,
            floors,
            residues,
            mappings,
            modes,
            window_short,
            window_long,
            overlap_bufs,
            prev_blocksize: None,
            finished: false,
        })
    }

    #[inline]
    pub fn channels(&self) -> u8 {
        self.channels
    }

    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    pub fn vendor(&self) -> &str {
        &self.vendor
    }

    #[inline]
    pub fn comments(&self) -> &[Vec<u8>] {
        &self.comments
    }

    #[inline]
    pub fn xray_params(&self) -> &OggXRayParams {
        &self.xray_params
    }

    /// Decode the next audio packet, appending interleaved stereo/mono `f32` frames to `out_interleaved`.
    /// Returns `Ok(false)` once end-of-stream is reached.
    pub fn decode_next_packet(
        &mut self,
        out_interleaved: &mut Vec<f32>,
    ) -> Result<bool, OggVorbisError> {
        if self.finished {
            return Ok(false);
        }

        let Some(pkt) = self.demuxer.next_packet()? else {
            self.finished = true;
            return Ok(false);
        };
        if pkt.bytes.is_empty() {
            if pkt.is_eos {
                self.finished = true;
            }
            return Ok(true);
        }

        let mut br = BitReader::new(&pkt.bytes);
        let Some(packet_type) = br.read_bit_opt() else {
            return Ok(true);
        };
        if packet_type != 0 {
            return Err(OggVorbisError::InvalidPacket("non-audio packet inside audio section"));
        }

        let mode_bits = ilog((self.modes.len() as u32).saturating_sub(1));
        let Some(mode_idx) = br.read_bits_opt(mode_bits) else {
            return Ok(true);
        };
        let mode = *self
            .modes
            .get(mode_idx as usize)
            .ok_or(OggVorbisError::InvalidPacket("audio mode index out of range"))?;

        let n = if mode.blockflag {
            self.blocksize_1
        } else {
            self.blocksize_0
        };
        let half_n = n / 2;

        let (prev_window_flag, next_window_flag) = if mode.blockflag {
            let p = br.read_bit_opt().unwrap_or(0) != 0;
            let nx = br.read_bit_opt().unwrap_or(0) != 0;
            (p, nx)
        } else {
            (false, false)
        };

        let ch = usize::from(self.channels);
        let mapping = &self.mappings[mode.mapping];

        let mut floors_buf = vec![vec![0.0f32; half_n]; ch];
        let mut nonzero = vec![false; ch];
        for c in 0..ch {
            let submap_idx = mapping.mux[c];
            let floor_idx = mapping.submaps[submap_idx].0;
            nonzero[c] = self.floors[floor_idx].decode_and_render(
                &mut br,
                &self.codebooks,
                half_n,
                &mut floors_buf[c],
            );
        }

        // Spec 4.3.2 step 4: propagate nonzero across polar-coupled channel pairs
        for &(mag, ang) in &mapping.coupling_steps {
            if nonzero[mag] || nonzero[ang] {
                nonzero[mag] = true;
                nonzero[ang] = true;
            }
        }

        let mut spectra = vec![vec![0.0f32; half_n]; ch];
        for (submap_idx, &(_, residue_idx)) in mapping.submaps.iter().enumerate() {
            let mut ch_indices = Vec::new();
            let mut do_not_decode = Vec::new();
            for (c, &nz) in nonzero.iter().enumerate() {
                if mapping.mux[c] == submap_idx {
                    ch_indices.push(c);
                    do_not_decode.push(!nz);
                }
            }
            if ch_indices.is_empty() {
                continue;
            }
            // Extract mutable borrows per submap channel
            let mut sub_vecs = vec![vec![0.0f32; half_n]; ch_indices.len()];
            let mut slices: Vec<&mut [f32]> = sub_vecs.iter_mut().map(|v| v.as_mut_slice()).collect();
            self.residues[residue_idx].decode(
                &mut br,
                &self.codebooks,
                &do_not_decode,
                half_n,
                &mut slices,
            );
            for (local_i, &c) in ch_indices.iter().enumerate() {
                spectra[c].copy_from_slice(&sub_vecs[local_i]);
            }
        }

        // Spec 4.3.5: inverse coupling in reverse coupling step order
        for &(mag, ang) in mapping.coupling_steps.iter().rev() {
            let (left_slice, right_slice) = if mag < ang {
                let (lo, hi) = spectra.split_at_mut(ang);
                (&mut lo[mag], &mut hi[0])
            } else {
                let (lo, hi) = spectra.split_at_mut(mag);
                (&mut hi[0], &mut lo[ang])
            };
            for (ms, as_) in left_slice[..half_n].iter_mut().zip(&mut right_slice[..half_n]) {
                let m = *ms;
                let a = *as_;
                let (new_m, new_a) = if m > 0.0 {
                    if a > 0.0 { (m, m - a) } else { (m + a, m) }
                } else if a > 0.0 {
                    (m, m + a)
                } else {
                    (m - a, m)
                };
                *ms = new_m;
                *as_ = new_a;
            }
        }

        // Floor dot product & fast O(N log N) IMDCT
        let mut pcm_blocks = vec![vec![0.0f32; n]; ch];
        for (c, &nz) in nonzero.iter().enumerate() {
            if !nz {
                continue;
            }
            for (s, &f) in spectra[c][..half_n].iter_mut().zip(&floors_buf[c][..half_n]) {
                *s *= f;
            }
            vorbis_imdct_fast(&spectra[c], &mut pcm_blocks[c]);
        }

        // Apply asymmetric/symmetric Vorbis slope window
        let (lw_start, _lw_end, left_win) = if mode.blockflag && !prev_window_flag {
            let s = n / 4 - self.blocksize_0 / 4;
            let e = n / 4 + self.blocksize_0 / 4;
            (s, e, self.window_short.as_slice())
        } else {
            let w = if mode.blockflag {
                self.window_long.as_slice()
            } else {
                self.window_short.as_slice()
            };
            (0, half_n, w)
        };
        let (rw_start, rw_end, right_win) = if mode.blockflag && !next_window_flag {
            let s = 3 * n / 4 - self.blocksize_0 / 4;
            let e = 3 * n / 4 + self.blocksize_0 / 4;
            (s, e, self.window_short.as_slice())
        } else {
            let w = if mode.blockflag {
                self.window_long.as_slice()
            } else {
                self.window_short.as_slice()
            };
            (half_n, n, w)
        };

        for blk in &mut pcm_blocks {
            blk[..lw_start].fill(0.0);
            for (k, &w) in left_win.iter().enumerate() {
                blk[lw_start + k] *= w;
            }
            let rlen = right_win.len();
            for (k, &w) in right_win.iter().enumerate() {
                let idx = rw_start + rlen - 1 - k;
                blk[idx] *= w;
            }
            blk[rw_end..n].fill(0.0);
        }

        // Overlap-add against previous packet's right half
        if let Some(prev_n) = self.prev_blocksize {
            let cur_start = if n > prev_n { (n - prev_n) / 4 } else { 0 };
            let cur_end = half_n;
            let prev_start = if prev_n > n { (prev_n - n) / 4 } else { 0 };
            let prev_half = prev_n / 2;
            let overlap_len = (prev_half - prev_start).min(cur_end - cur_start);

            let out_frames = if prev_n > n {
                (prev_n - n) / 4 + n / 2
            } else if n > prev_n {
                (n - prev_n) / 4 + prev_n / 2
            } else {
                half_n
            };
            let base_out = out_interleaved.len();
            out_interleaved.resize(base_out + out_frames * ch, 0.0);

            // Leading un-overlapped samples from previous long frame if switching long -> short
            let lead_prev = prev_start;
            for (c, (ov, blk)) in self.overlap_bufs.iter().zip(&pcm_blocks).enumerate() {
                for (i, &val) in ov[..lead_prev].iter().enumerate() {
                    out_interleaved[base_out + i * ch + c] = val;
                }
                for (i, (&ov_val, &blk_val)) in ov[prev_start..prev_start + overlap_len]
                    .iter()
                    .zip(&blk[cur_start..cur_start + overlap_len])
                    .enumerate()
                {
                    out_interleaved[base_out + (lead_prev + i) * ch + c] =
                        (ov_val + blk_val).clamp(-1.5, 1.5);
                }
                // Trailing un-overlapped left samples from current long frame if switching short -> long
                let trail_start = lead_prev + overlap_len;
                for (offset, &val) in blk[(cur_start + overlap_len)..half_n].iter().enumerate() {
                    let out_i = trail_start + offset;
                    if out_i < out_frames {
                        out_interleaved[base_out + out_i * ch + c] = val;
                    }
                }
            }
        }

        // Save right half of current frame for next packet overlap
        for (ov, blk) in self.overlap_bufs.iter_mut().zip(&pcm_blocks) {
            ov.clear();
            ov.extend_from_slice(&blk[half_n..n]);
        }
        self.prev_blocksize = Some(n);

        if pkt.is_eos {
            self.finished = true;
        }
        Ok(true)
    }

    /// Decode all remaining Vorbis packets in the stream to interleaved `f32` PCM.
    pub fn decode_all(&mut self) -> Result<Vec<f32>, OggVorbisError> {
        let mut pcm = Vec::new();
        while self.decode_next_packet(&mut pcm)? {}
        Ok(pcm)
    }
}

// ---------------------------------------------------------------------------
// Header Packet Parsers
// ---------------------------------------------------------------------------

fn parse_id_packet(pkt: &[u8]) -> Result<(u8, u32, usize, usize), OggVorbisError> {
    if pkt.len() < 30 || pkt[0] != 0x01 || &pkt[1..7] != b"vorbis" {
        return Err(OggVorbisError::InvalidHeader("expected 0x01vorbis identification packet"));
    }
    let version = u32::from_le_bytes(pkt[7..11].try_into().unwrap());
    if version != 0 {
        return Err(OggVorbisError::Unsupported("unsupported Vorbis version != 0"));
    }
    let channels = pkt[11];
    if channels == 0 || channels > 8 {
        return Err(OggVorbisError::Unsupported("unsupported Vorbis channel count"));
    }
    let sample_rate = u32::from_le_bytes(pkt[12..16].try_into().unwrap());
    if sample_rate == 0 {
        return Err(OggVorbisError::InvalidHeader("Vorbis sample rate is 0"));
    }
    let bs_byte = pkt[28];
    let exp0 = u32::from(bs_byte & 0x0f);
    let exp1 = u32::from((bs_byte >> 4) & 0x0f);
    if !(6..=13).contains(&exp0) || !(6..=13).contains(&exp1) || exp0 > exp1 {
        return Err(OggVorbisError::InvalidHeader("invalid Vorbis blocksize exponents"));
    }
    if (pkt[29] & 1) == 0 {
        return Err(OggVorbisError::InvalidHeader("missing framing bit in ID header"));
    }
    Ok((channels, sample_rate, 1usize << exp0, 1usize << exp1))
}

fn parse_comment_packet(pkt: &[u8]) -> Result<(String, Vec<Vec<u8>>, OggXRayParams), OggVorbisError> {
    if pkt.len() < 7 || pkt[0] != 0x03 || &pkt[1..7] != b"vorbis" {
        return Err(OggVorbisError::InvalidHeader("expected 0x03vorbis comment packet"));
    }
    let mut br = BitReader::new(&pkt[7..]);
    let vendor_len = br.read_u32_aligned_bytes()? as usize;
    let vendor_bytes = br.read_bytes(vendor_len)?;
    let vendor = String::from_utf8_lossy(&vendor_bytes).into_owned();
    let count = br.read_u32_aligned_bytes()? as usize;
    let mut comments = Vec::with_capacity(count.min(256));
    let mut xray_params = OggXRayParams::default();

    for i in 0..count {
        let len = br.read_u32_aligned_bytes()? as usize;
        let payload = br.read_bytes(len)?;
        if i == 0
            && let Ok(parsed) = OggXRayParams::parse(&payload)
        {
            xray_params = parsed;
        }
        comments.push(payload);
    }

    Ok((vendor, comments, xray_params))
}

type VorbisSetupTables = (
    Vec<Codebook>,
    Vec<Floor>,
    Vec<Residue>,
    Vec<Mapping>,
    Vec<Mode>,
);

fn parse_setup_packet(
    pkt: &[u8],
    channels: usize,
) -> Result<VorbisSetupTables, OggVorbisError> {
    if pkt.len() < 7 || pkt[0] != 0x05 || &pkt[1..7] != b"vorbis" {
        return Err(OggVorbisError::InvalidHeader("expected 0x05vorbis setup packet"));
    }
    let mut br = BitReader::new(&pkt[7..]);

    // 1. Codebooks
    let codebook_count = br.read_bits_hdr(8, "truncated codebook count")? as usize + 1;
    let mut codebooks = Vec::with_capacity(codebook_count);
    for _ in 0..codebook_count {
        codebooks.push(Codebook::parse(&mut br)?);
    }

    // 2. Time-domain transforms (must be zeros in Vorbis I)
    let time_count = br.read_bits_hdr(6, "truncated time count")? as usize + 1;
    for _ in 0..time_count {
        let val = br.read_bits_hdr(16, "truncated time transform value")?;
        if val != 0 {
            return Err(OggVorbisError::InvalidHeader("non-zero time domain transform"));
        }
    }

    // 3. Floors
    let floor_count = br.read_bits_hdr(6, "truncated floor count")? as usize + 1;
    let mut floors = Vec::with_capacity(floor_count);
    for _ in 0..floor_count {
        floors.push(Floor::parse(&mut br, codebooks.len())?);
    }

    // 4. Residues
    let residue_count = br.read_bits_hdr(6, "truncated residue count")? as usize + 1;
    let mut residues = Vec::with_capacity(residue_count);
    for _ in 0..residue_count {
        residues.push(Residue::parse(&mut br, codebooks.len())?);
    }

    // 5. Mappings
    let mapping_count = br.read_bits_hdr(6, "truncated mapping count")? as usize + 1;
    let mut mappings = Vec::with_capacity(mapping_count);
    for _ in 0..mapping_count {
        mappings.push(Mapping::parse(&mut br, channels, floors.len(), residues.len())?);
    }

    // 6. Modes
    let mode_count = br.read_bits_hdr(6, "truncated mode count")? as usize + 1;
    let mut modes = Vec::with_capacity(mode_count);
    for _ in 0..mode_count {
        let blockflag = br.read_bit_hdr("truncated mode blockflag")? != 0;
        let _windowtype = br.read_bits_hdr(16, "truncated mode windowtype")?;
        let _transformtype = br.read_bits_hdr(16, "truncated mode transformtype")?;
        let mapping = br.read_bits_hdr(8, "truncated mode mapping")? as usize;
        if mapping >= mappings.len() {
            return Err(OggVorbisError::InvalidHeader("mode mapping index out of range"));
        }
        modes.push(Mode { blockflag, mapping });
    }

    let framing = br.read_bit_hdr("missing setup framing bit")?;
    if framing == 0 {
        return Err(OggVorbisError::InvalidHeader("zero setup framing bit"));
    }

    Ok((codebooks, floors, residues, mappings, modes))
}

// ---------------------------------------------------------------------------
// Comprehensive Tests (IMDCT Reference Match, Synthetic Vorbis I Stream, VFS)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct BitWriter {
        bytes: Vec<u8>,
        cur_byte: u8,
        bit_pos: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                cur_byte: 0,
                bit_pos: 0,
            }
        }

        fn write_bit(&mut self, bit: u32) {
            if (bit & 1) != 0 {
                self.cur_byte |= 1 << self.bit_pos;
            }
            self.bit_pos += 1;
            if self.bit_pos == 8 {
                self.bytes.push(self.cur_byte);
                self.cur_byte = 0;
                self.bit_pos = 0;
            }
        }

        fn write_bits(&mut self, val: u32, count: u32) {
            for s in 0..count {
                self.write_bit((val >> s) & 1);
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.bit_pos > 0 {
                self.bytes.push(self.cur_byte);
            }
            self.bytes
        }
    }

    fn make_ogg_page(seq: u32, header_type: u8, granule: i64, serial: u32, packet: &[u8]) -> Vec<u8> {
        let mut page = Vec::new();
        page.extend_from_slice(b"OggS");
        page.push(0);
        page.push(header_type);
        page.extend_from_slice(&granule.to_le_bytes());
        page.extend_from_slice(&serial.to_le_bytes());
        page.extend_from_slice(&seq.to_le_bytes());
        page.extend_from_slice(&0u32.to_le_bytes()); // CRC optional for decoder

        let mut segs = Vec::new();
        let mut rem = packet.len();
        while rem >= 255 {
            segs.push(255u8);
            rem -= 255;
        }
        segs.push(rem as u8);
        page.push(segs.len() as u8);
        page.extend_from_slice(&segs);
        page.extend_from_slice(packet);
        page
    }

    /// Quantize a positive float to Vorbis `float32_pack` representation with integer mantissa.
    fn float32_pack_simple(val: i32) -> u32 {
        // value = val * 2^(788 - 788) -> exponent = 788, mantissa = abs(val)
        let sign = if val < 0 { 1u32 << 31 } else { 0 };
        let abs = val.unsigned_abs() & 0x001f_ffff;
        sign | (788u32 << 21) | abs
    }

    /// Naive O(N^2) reference Vorbis I IMDCT equation (`Spec 1.1.2 / 4.3.7`).
    fn naive_vorbis_imdct(spectrum: &[f32], n: usize) -> Vec<f32> {
        let m = n / 2;
        let mut out = vec![0.0f32; n];
        for (k, dst) in out.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (i, &x) in spectrum[..m].iter().enumerate() {
                let arg = PI / (2 * n) as f64 * (2 * k + 1 + m) as f64 * (2 * i + 1) as f64;
                sum += f64::from(x) * arg.cos();
            }
            *dst = sum as f32;
        }
        out
    }

    #[test]
    fn test_fast_imdct_matches_naive_vorbis_reference() {
        for &n in &[64usize, 128, 256, 512, 1024] {
            let m = n / 2;
            let spectrum: Vec<f32> = (0..m)
                .map(|i| (i as f32 * 0.37).sin() * 2.5 - (i as f32 * 0.11).cos())
                .collect();
            let expected = naive_vorbis_imdct(&spectrum, n);
            let mut fast = vec![0.0f32; n];
            vorbis_imdct_fast(&spectrum, &mut fast);

            let max_err = expected
                .iter()
                .zip(fast.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-3,
                "IMDCT mismatch at N={n}: max_err={max_err}"
            );
        }
    }

    #[test]
    fn test_decode_synthetic_vorbis_stream_and_xray_metadata_from_vfs() {
        // Build a complete, valid Vorbis I stream (64-sample blocksize_0 & blocksize_1, 1ch, 44100 Hz)
        // exercising Codebook Huffman + VQ, Floor1 piecewise-linear dB curve, Residue1, IMDCT, and overlap-add!
        let mut id_pkt = Vec::new();
        id_pkt.push(0x01);
        id_pkt.extend_from_slice(b"vorbis");
        id_pkt.extend_from_slice(&0u32.to_le_bytes()); // version 0
        id_pkt.push(1); // channels = 1
        id_pkt.extend_from_slice(&44_100u32.to_le_bytes()); // sample_rate
        id_pkt.extend_from_slice(&0i32.to_le_bytes());
        id_pkt.extend_from_slice(&128_000i32.to_le_bytes());
        id_pkt.extend_from_slice(&0i32.to_le_bytes());
        id_pkt.push((6u8 << 4) | 6u8); // blocksize_0=64, blocksize_1=64
        id_pkt.push(1); // framing

        let xray_c = OggXRayParams {
            version: 0x0003,
            min_dist: 2.5,
            max_dist: 45.0,
            base_volume: 0.9,
            game_type: 12,
            max_ai_dist: 45.0,
        };
        let mut cmt_pkt = Vec::new();
        cmt_pkt.push(0x03);
        cmt_pkt.extend_from_slice(b"vorbis");
        let vendor = b"X-Ray Monolith Rust";
        cmt_pkt.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        cmt_pkt.extend_from_slice(vendor);
        cmt_pkt.extend_from_slice(&1u32.to_le_bytes()); // 1 comment
        let xr_bytes = xray_c.to_bytes();
        cmt_pkt.extend_from_slice(&(xr_bytes.len() as u32).to_le_bytes());
        cmt_pkt.extend_from_slice(&xr_bytes);
        cmt_pkt.push(1);

        // Setup packet:
        // Book 0 (Floor1 / Classbook scalar): 2 entries, dim=1, lengths=[1, 1] -> codes 0 ('0'), 1 ('1'), lookup_type=0
        // Book 1 (Residue VQ): 4 entries, dim=2, lengths=[2, 2, 2, 2] -> codes '00','01','10','11',
        // lookup_type=2, min=0, delta=4, value_bits=1, mults = [[0,0],[1,0],[0,1],[1,1]] => entry 1 = [4.0, 0.0], entry 3 = [4.0, 4.0]
        let mut sb = BitWriter::new();
        sb.write_bits(1, 8); // codebook_count - 1 = 1 (2 books)

        // Book 0
        sb.write_bits(0x564342, 24);
        sb.write_bits(1, 16); // dim = 1
        sb.write_bits(2, 24); // entries = 2
        sb.write_bit(0); // unordered
        sb.write_bit(0); // not sparse
        sb.write_bits(0, 5); // len = 1
        sb.write_bits(0, 5); // len = 1
        sb.write_bits(0, 4); // lookup_type = 0

        // Book 1
        sb.write_bits(0x564342, 24);
        sb.write_bits(2, 16); // dim = 2
        sb.write_bits(4, 24); // entries = 4
        sb.write_bit(0); // unordered
        sb.write_bit(0); // not sparse
        for _ in 0..4 {
            sb.write_bits(1, 5); // len = 2
        }
        sb.write_bits(2, 4); // lookup_type = 2
        sb.write_bits(float32_pack_simple(0), 32);
        sb.write_bits(float32_pack_simple(4), 32);
        sb.write_bits(0, 4); // value_bits = 1
        sb.write_bit(0); // sequence_p = false
        for &m in &[0u32, 0, 1, 0, 0, 1, 1, 1] {
            sb.write_bits(m, 1);
        }

        // Time transforms
        sb.write_bits(0, 6); // 1 transform
        sb.write_bits(0, 16);

        // Floor 0: Floor1 with 0 partitions -> only endpoints Y0, Y1
        sb.write_bits(0, 6); // 1 floor
        sb.write_bits(1, 16); // type 1
        sb.write_bits(0, 5); // partitions = 0
        sb.write_bits(0, 2); // multiplier = 1 (range=256 => y_bits=8)
        sb.write_bits(5, 4); // range_bits = 5 (X = [0, 32])

        // Residue 0: type 1, begin=0, end=8, partition_size=4, classifications=1, classbook=0
        sb.write_bits(0, 6); // 1 residue
        sb.write_bits(1, 16); // type 1
        sb.write_bits(0, 24); // begin = 0
        sb.write_bits(8, 24); // end = 8
        sb.write_bits(3, 24); // partition_size = 4
        sb.write_bits(0, 6); // classifications = 1
        sb.write_bits(0, 8); // classbook = 0
        sb.write_bits(1, 3); // cascade low = bit 0 set (pass 0 only)
        sb.write_bit(0); // bitflag = 0
        sb.write_bits(1, 8); // book[0][0] = Book 1

        // Mapping 0
        sb.write_bits(0, 6); // 1 mapping
        sb.write_bits(0, 16); // type 0
        sb.write_bit(0); // submaps = 1
        sb.write_bit(0); // no coupling
        sb.write_bits(0, 2); // reserved
        sb.write_bits(0, 8); // time
        sb.write_bits(0, 8); // floor 0
        sb.write_bits(0, 8); // residue 0

        // Mode 0
        sb.write_bits(0, 6); // 1 mode
        sb.write_bit(0); // blockflag = 0
        sb.write_bits(0, 16); // window
        sb.write_bits(0, 16); // transform
        sb.write_bits(0, 8); // mapping 0
        sb.write_bit(1); // framing

        let mut setup_pkt = vec![0x05];
        setup_pkt.extend_from_slice(b"vorbis");
        setup_pkt.extend_from_slice(&sb.finish());

        // Helper to emit 1 audio packet:
        // Floor1 nonzero=1, Y0=255, Y1=255 (linear gain = 1.0 across spectrum!)
        // Classword 0 => class 0 (code '0' in Book 0)
        // Partition 0 (bins 0..4): Book 1 entry 3 ('11' = [4,4]), entry 0 ('00' = [0,0]) -> [4, 4, 0, 0]
        // Partition 1 (bins 4..8): Book 1 entry 0 ('00' = [0,0]), entry 0 ('00' = [0,0])
        let make_audio_pkt = || -> Vec<u8> {
            let mut aw = BitWriter::new();
            aw.write_bit(0); // audio packet
            // 1 mode -> 0 mode bits
            aw.write_bit(1); // floor1 nonzero
            aw.write_bits(255, 8); // Y0 = 255 => 1.0
            aw.write_bits(255, 8); // Y1 = 255 => 1.0
            // classword partition 0 (classbook dim=1 => read twice across 2 partitions)
            aw.write_bit(0); // class 0
            // partition 0 VQ: 2 codewords in Book 1 (MSB-first tree paths '1','1' and '0','0')
            aw.write_bit(1);
            aw.write_bit(1); // entry 3 => [4.0, 4.0]
            aw.write_bit(0);
            aw.write_bit(0); // entry 0 => [0.0, 0.0]
            // classword partition 1
            aw.write_bit(0); // class 0
            // partition 1 VQ
            aw.write_bit(0);
            aw.write_bit(0);
            aw.write_bit(0);
            aw.write_bit(0);
            aw.finish()
        };

        let mut ogg = Vec::new();
        ogg.extend_from_slice(&make_ogg_page(0, 0x02, 0, 99, &id_pkt));
        ogg.extend_from_slice(&make_ogg_page(1, 0x00, 0, 99, &cmt_pkt));
        ogg.extend_from_slice(&make_ogg_page(2, 0x00, 0, 99, &setup_pkt));
        ogg.extend_from_slice(&make_ogg_page(3, 0x00, 0, 99, &make_audio_pkt()));
        ogg.extend_from_slice(&make_ogg_page(4, 0x00, 32, 99, &make_audio_pkt()));
        ogg.extend_from_slice(&make_ogg_page(5, 0x04, 64, 99, &make_audio_pkt()));

        // Pack into a synthetic X-Ray DB archive (chunk 1 index table) and stream via VirtualFileSystem
        let vpath = "sounds\\weapons\\synthetic_shot.ogg";
        let crc = xray_vfs::compute_crc32(&ogg);
        let rec_len = (16 + vpath.len()) as u16;
        let table_len = 2 + usize::from(rec_len);
        let data_offset = (8 + table_len) as u32;
        let mut chunk1 = Vec::new();
        chunk1.extend_from_slice(&rec_len.to_le_bytes());
        chunk1.extend_from_slice(&(ogg.len() as u32).to_le_bytes());
        chunk1.extend_from_slice(&(ogg.len() as u32).to_le_bytes());
        chunk1.extend_from_slice(&crc.to_le_bytes());
        chunk1.extend_from_slice(vpath.as_bytes());
        chunk1.extend_from_slice(&data_offset.to_le_bytes());

        let mut archive = Vec::new();
        archive.extend_from_slice(&1u32.to_le_bytes());
        archive.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
        archive.extend_from_slice(&chunk1);
        archive.extend_from_slice(&ogg);

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("audio_test.db0", archive).unwrap();

        let decoded = decode_ogg_from_vfs(&vfs, "weapons\\synthetic_shot").unwrap();
        assert_eq!(decoded.sample_rate, 44_100);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.vendor, "X-Ray Monolith Rust");
        assert_eq!(decoded.xray_params, xray_c);
        // 3 packets with N=64 -> 2 overlap-add blocks * 32 samples = 64 non-zero PCM samples!
        assert_eq!(decoded.samples.len(), 64);
        let energy: f32 = decoded.samples.iter().map(|x| x * x).sum();
        assert!(energy > 1.0, "expected audible decoded PCM energy, got {energy}");
    }
}
