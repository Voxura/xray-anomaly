//! Pure Rust Bounds-Checked X-Ray LZHuf Decompressor & Compressor
//!
//! Thread-safe, zero-global-state implementation of X-Ray Monolith Engine's
//! adaptive Huffman + LZSS chunk compression (`src/xrCore/LzHuf.cpp`,
//! `_compressLZ` / `_decompressLZ`), used for `CFS_CompressMark` (`1 << 31`)
//! chunks in `.db`/`.xdb` archives and serialized engine streams.

use std::fmt;

const N: usize = 4096;
const F: usize = 60;
const THRESHOLD: usize = 2;
const NIL: usize = N;

const N_CHAR: usize = 256 - THRESHOLD + F; // 314
const T: usize = N_CHAR * 2 - 1; // 627
const R: usize = T - 1; // 626
const MAX_FREQ: u32 = 0x4000;

/// Hard upper bound to protect against corrupt 32-bit uncompressed length headers (512 MiB).
const DEFAULT_MAX_OUTPUT_SIZE: usize = 512 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LzhufError {
    UnexpectedEof,
    CorruptStream,
    OutputLimitExceeded { declared: usize, max_allowed: usize },
}

impl fmt::Display for LzhufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for LzhufError {}

#[rustfmt::skip]
const P_LEN: [u8; 64] = [
    0x03, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
];

#[rustfmt::skip]
const P_CODE: [u8; 64] = [
    0x00, 0x20, 0x30, 0x40, 0x50, 0x58, 0x60, 0x68,
    0x70, 0x78, 0x80, 0x88, 0x90, 0x94, 0x98, 0x9C,
    0xA0, 0xA4, 0xA8, 0xAC, 0xB0, 0xB4, 0xB8, 0xBC,
    0xC0, 0xC2, 0xC4, 0xC6, 0xC8, 0xCA, 0xCC, 0xCE,
    0xD0, 0xD2, 0xD4, 0xD6, 0xD8, 0xDA, 0xDC, 0xDE,
    0xE0, 0xE2, 0xE4, 0xE6, 0xE8, 0xEA, 0xEC, 0xEE,
    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
    0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,
];

#[rustfmt::skip]
const D_CODE: [u8; 256] = [
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x09,
    0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A,
    0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B,
    0x0C, 0x0C, 0x0C, 0x0C, 0x0D, 0x0D, 0x0D, 0x0D,
    0x0E, 0x0E, 0x0E, 0x0E, 0x0F, 0x0F, 0x0F, 0x0F,
    0x10, 0x10, 0x10, 0x10, 0x11, 0x11, 0x11, 0x11,
    0x12, 0x12, 0x12, 0x12, 0x13, 0x13, 0x13, 0x13,
    0x14, 0x14, 0x14, 0x14, 0x15, 0x15, 0x15, 0x15,
    0x16, 0x16, 0x16, 0x16, 0x17, 0x17, 0x17, 0x17,
    0x18, 0x18, 0x19, 0x19, 0x1A, 0x1A, 0x1B, 0x1B,
    0x1C, 0x1C, 0x1D, 0x1D, 0x1E, 0x1E, 0x1F, 0x1F,
    0x20, 0x20, 0x21, 0x21, 0x22, 0x22, 0x23, 0x23,
    0x24, 0x24, 0x25, 0x25, 0x26, 0x26, 0x27, 0x27,
    0x28, 0x28, 0x29, 0x29, 0x2A, 0x2A, 0x2B, 0x2B,
    0x2C, 0x2C, 0x2D, 0x2D, 0x2E, 0x2E, 0x2F, 0x2F,
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,
];

#[rustfmt::skip]
const D_LEN: [u8; 256] = [
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
    0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
];

struct HuffmanModel {
    freq: [u32; T + 1],
    prnt: [usize; T + N_CHAR + 1],
    son: [usize; T],
}

impl HuffmanModel {
    fn new() -> Self {
        let mut model = Self {
            freq: [0; T + 1],
            prnt: [0; T + N_CHAR + 1],
            son: [0; T],
        };
        model.start_huff();
        model
    }

    fn start_huff(&mut self) {
        for i in 0..N_CHAR {
            self.freq[i] = 1;
            self.son[i] = i + T;
            self.prnt[i + T] = i;
        }
        let mut i = 0;
        let mut j = N_CHAR;
        while j <= R {
            self.freq[j] = self.freq[i] + self.freq[i + 1];
            self.son[j] = i;
            self.prnt[i] = j;
            self.prnt[i + 1] = j;
            i += 2;
            j += 1;
        }
        self.freq[T] = 0xFFFF;
        self.prnt[R] = 0;
    }

    fn reconst(&mut self) {
        let mut j = 0usize;
        for i in 0..T {
            if self.son[i] >= T {
                self.freq[j] = (self.freq[i] + 1) / 2;
                self.son[j] = self.son[i];
                j += 1;
            }
        }

        let mut i = 0usize;
        for j in N_CHAR..T {
            let k = i + 1;
            let f = self.freq[i] + self.freq[k];
            self.freq[j] = f;

            let mut ins = j;
            while ins > 0 && f < self.freq[ins - 1] {
                ins -= 1;
            }

            if ins < j {
                self.freq.copy_within(ins..j, ins + 1);
                self.son.copy_within(ins..j, ins + 1);
            }
            self.freq[ins] = f;
            self.son[ins] = i;

            i += 2;
        }

        for i in 0..T {
            let k = self.son[i];
            self.prnt[k] = i;
            if k < T {
                self.prnt[k + 1] = i;
            }
        }
    }

    fn update(&mut self, sym: usize) {
        if self.freq[R] == MAX_FREQ {
            self.reconst();
        }
        let mut c = self.prnt[sym + T];
        loop {
            self.freq[c] += 1;
            let k = self.freq[c];

            let mut l = c + 1;
            if l < T && k > self.freq[l] {
                while l + 1 < T && k > self.freq[l + 1] {
                    l += 1;
                }
                self.freq[c] = self.freq[l];
                self.freq[l] = k;

                let i = self.son[c];
                self.prnt[i] = l;
                if i < T {
                    self.prnt[i + 1] = l;
                }

                let j = self.son[l];
                self.son[l] = i;

                self.prnt[j] = c;
                if j < T {
                    self.prnt[j + 1] = c;
                }
                self.son[c] = j;

                c = l;
            }

            c = self.prnt[c];
            if c == 0 {
                break;
            }
        }
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    getbuf: u16,
    getlen: u8,
    eof_padding_bits: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            getbuf: 0,
            getlen: 0,
            eof_padding_bits: 0,
        }
    }

    fn next_byte_or_pad(&mut self) -> Result<u16, LzhufError> {
        if self.pos < self.data.len() {
            let b = self.data[self.pos] as u16;
            self.pos += 1;
            Ok(b)
        } else {
            // Matching LzHuf.cpp: trailing bits past input end pad with 0 up to 16 bits
            // to flush the final shift-register bits safely; excessive EOF reads fail.
            self.eof_padding_bits += 8;
            if self.eof_padding_bits > 24 {
                Err(LzhufError::UnexpectedEof)
            } else {
                Ok(0)
            }
        }
    }

    #[inline]
    fn get_bit(&mut self) -> Result<usize, LzhufError> {
        while self.getlen <= 8 {
            let b = self.next_byte_or_pad()?;
            self.getbuf |= b << (8 - self.getlen);
            self.getlen += 8;
        }
        let bit = ((self.getbuf & 0x8000) >> 15) as usize;
        self.getbuf <<= 1;
        self.getlen -= 1;
        Ok(bit)
    }

    #[inline]
    fn get_byte(&mut self) -> Result<usize, LzhufError> {
        while self.getlen <= 8 {
            let b = self.next_byte_or_pad()?;
            self.getbuf |= b << (8 - self.getlen);
            self.getlen += 8;
        }
        let byte = ((self.getbuf & 0xFF00) >> 8) as usize;
        self.getbuf <<= 8;
        self.getlen -= 8;
        Ok(byte)
    }
}

struct BitWriter {
    out: Vec<u8>,
    putbuf: u16,
    putlen: u8,
}

impl BitWriter {
    fn with_capacity(cap: usize) -> Self {
        Self {
            out: Vec::with_capacity(cap),
            putbuf: 0,
            putlen: 0,
        }
    }

    #[inline]
    fn put_code(&mut self, len: u8, code_hi: u16) {
        self.putbuf |= code_hi >> self.putlen;
        self.putlen += len;
        if self.putlen >= 8 {
            self.out.push((self.putbuf >> 8) as u8);
            self.putlen -= 8;
            if self.putlen >= 8 {
                self.out.push((self.putbuf & 0xFF) as u8);
                self.putlen -= 8;
                self.putbuf = code_hi.wrapping_shl((len - self.putlen) as u32);
            } else {
                self.putbuf <<= 8;
            }
        }
    }

    #[inline]
    fn flush(&mut self) {
        if self.putlen > 0 {
            self.out.push((self.putbuf >> 8) as u8);
            self.putlen = 0;
            self.putbuf = 0;
        }
    }
}

/// Decompress an X-Ray LZHuf stream (`_compressLZ` output) with a default 512 MiB cap.
pub fn decompress_lzhuf(input: &[u8]) -> Result<Vec<u8>, LzhufError> {
    decompress_lzhuf_with_limit(input, DEFAULT_MAX_OUTPUT_SIZE)
}

/// Decompress an X-Ray LZHuf stream with an explicit maximum decompressed byte cap.
pub fn decompress_lzhuf_with_limit(
    input: &[u8],
    max_output_size: usize,
) -> Result<Vec<u8>, LzhufError> {
    if input.len() < 4 {
        return Err(LzhufError::UnexpectedEof);
    }

    let declared_size = u32::from_le_bytes([input[0], input[1], input[2], input[3]]) as usize;
    if declared_size > max_output_size {
        return Err(LzhufError::OutputLimitExceeded {
            declared: declared_size,
            max_allowed: max_output_size,
        });
    }

    if declared_size == 0 {
        return Ok(Vec::new());
    }

    let mut reader = BitReader::new(&input[4..]);
    let mut model = HuffmanModel::new();
    let mut text_buf = [0x20u8; N + F];
    let mut output = Vec::with_capacity(declared_size);
    let mut r = N - F;

    while output.len() < declared_size {
        let c = decode_char(&mut reader, &mut model)?;
        if c < 256 {
            let byte = c as u8;
            output.push(byte);
            text_buf[r] = byte;
            r = (r + 1) & (N - 1);
        } else {
            let pos_offset = decode_position(&mut reader)?;
            let mut i = (r.wrapping_sub(pos_offset).wrapping_sub(1)) & (N - 1);
            let match_len = c - 255 + THRESHOLD;
            for _ in 0..match_len {
                if output.len() >= declared_size {
                    break;
                }
                let byte = text_buf[i];
                output.push(byte);
                text_buf[r] = byte;
                r = (r + 1) & (N - 1);
                i = (i + 1) & (N - 1);
            }
        }
    }

    Ok(output)
}

fn decode_char(reader: &mut BitReader<'_>, model: &mut HuffmanModel) -> Result<usize, LzhufError> {
    let mut c = model.son[R];
    while c < T {
        let bit = reader.get_bit()?;
        let idx = c + bit;
        if idx >= T {
            return Err(LzhufError::CorruptStream);
        }
        c = model.son[idx];
    }
    let sym = c - T;
    if sym >= N_CHAR {
        return Err(LzhufError::CorruptStream);
    }
    model.update(sym);
    Ok(sym)
}

fn decode_position(reader: &mut BitReader<'_>) -> Result<usize, LzhufError> {
    let mut i = reader.get_byte()?;
    let upper = (D_CODE[i] as usize) << 6;
    let extra_bits = D_LEN[i];
    if extra_bits < 2 {
        return Err(LzhufError::CorruptStream);
    }
    for _ in 0..(extra_bits - 2) {
        i = (i << 1) + reader.get_bit()?;
    }
    Ok(upper | (i & 0x3F))
}

/// Compress bytes into an X-Ray LZHuf stream compatible with `_decompressLZ` (`LzHuf.cpp`).
pub fn compress_lzhuf(input: &[u8]) -> Vec<u8> {
    let textsize = u32::try_from(input.len()).expect("input exceeds u32 length");
    let mut writer = BitWriter::with_capacity(input.len() / 2 + 16);
    writer.out.extend_from_slice(&textsize.to_le_bytes());
    if textsize == 0 {
        return writer.out;
    }

    let mut model = HuffmanModel::new();
    let mut tree = LzssTree::new(input);
    let mut s = 0usize;
    let mut r = N - F;
    for b in &mut tree.text_buf[s..r] {
        *b = 0x20;
    }

    let mut in_pos = 0usize;
    let mut len = 0usize;
    while len < F && in_pos < input.len() {
        tree.text_buf[r + len] = input[in_pos];
        in_pos += 1;
        len += 1;
    }

    for i in 1..=F {
        tree.insert_node(r - i);
    }
    tree.insert_node(r);

    while len > 0 {
        let mut match_length = tree.match_length.min(len);
        if match_length <= THRESHOLD {
            match_length = 1;
            encode_char(&mut writer, &mut model, tree.text_buf[r] as usize);
        } else {
            encode_char(&mut writer, &mut model, 255 - THRESHOLD + match_length);
            encode_position(&mut writer, tree.match_position);
        }

        let last_match_length = match_length;
        let mut i = 0usize;
        while i < last_match_length && in_pos < input.len() {
            tree.delete_node(s);
            let byte = input[in_pos];
            in_pos += 1;
            tree.text_buf[s] = byte;
            if s < F - 1 {
                tree.text_buf[s + N] = byte;
            }
            s = (s + 1) & (N - 1);
            r = (r + 1) & (N - 1);
            tree.insert_node(r);
            i += 1;
        }
        while i < last_match_length {
            i += 1;
            tree.delete_node(s);
            s = (s + 1) & (N - 1);
            r = (r + 1) & (N - 1);
            len -= 1;
            if len > 0 {
                tree.insert_node(r);
            }
        }
    }

    writer.flush();
    writer.out
}

fn encode_char(writer: &mut BitWriter, model: &mut HuffmanModel, sym: usize) {
    let mut code = 0u16;
    let mut j = 0u8;
    let mut k = model.prnt[sym + T];

    loop {
        code >>= 1;
        if (k & 1) != 0 {
            code |= 0x8000;
        }
        j += 1;
        k = model.prnt[k];
        if k == R {
            break;
        }
    }

    writer.put_code(j, code);
    model.update(sym);
}

fn encode_position(writer: &mut BitWriter, pos: usize) {
    let upper = pos >> 6;
    writer.put_code(P_LEN[upper], (P_CODE[upper] as u16) << 8);
    writer.put_code(6, ((pos & 0x3F) as u16) << 10);
}

struct LzssTree {
    text_buf: Vec<u8>,
    lson: Vec<usize>,
    rson: Vec<usize>,
    dad: Vec<usize>,
    match_position: usize,
    match_length: usize,
}

impl LzssTree {
    fn new(_input: &[u8]) -> Self {
        let mut rson = vec![0; N + 257];
        for slot in &mut rson[N + 1..=N + 256] {
            *slot = NIL;
        }
        Self {
            text_buf: vec![0; N + F],
            lson: vec![0; N + 1],
            rson,
            dad: vec![NIL; N + 1],
            match_position: 0,
            match_length: 0,
        }
    }

    fn insert_node(&mut self, r: usize) {
        let mut cmp = 1i32;
        let mut p = N + 1 + self.text_buf[r] as usize;
        self.rson[r] = NIL;
        self.lson[r] = NIL;
        self.match_length = 0;

        loop {
            if cmp >= 0 {
                if self.rson[p] != NIL {
                    p = self.rson[p];
                } else {
                    self.rson[p] = r;
                    self.dad[r] = p;
                    return;
                }
            } else if self.lson[p] != NIL {
                p = self.lson[p];
            } else {
                self.lson[p] = r;
                self.dad[r] = p;
                return;
            }

            let mut i = 1usize;
            while i < F {
                cmp = self.text_buf[r + i] as i32 - self.text_buf[p + i] as i32;
                if cmp != 0 {
                    break;
                }
                i += 1;
            }

            if i > THRESHOLD {
                if i > self.match_length {
                    self.match_position = (r.wrapping_sub(p) & (N - 1)).wrapping_sub(1);
                    self.match_length = i;
                    if self.match_length >= F {
                        break;
                    }
                } else if i == self.match_length {
                    let c = (r.wrapping_sub(p) & (N - 1)).wrapping_sub(1);
                    if c < self.match_position {
                        self.match_position = c;
                    }
                }
            }
        }

        self.dad[r] = self.dad[p];
        self.lson[r] = self.lson[p];
        self.rson[r] = self.rson[p];
        self.dad[self.lson[p]] = r;
        self.dad[self.rson[p]] = r;
        if self.rson[self.dad[p]] == p {
            self.rson[self.dad[p]] = r;
        } else {
            self.lson[self.dad[p]] = r;
        }
        self.dad[p] = NIL;
    }

    fn delete_node(&mut self, p: usize) {
        if self.dad[p] == NIL {
            return;
        }
        let q = if self.rson[p] == NIL {
            self.lson[p]
        } else if self.lson[p] == NIL {
            self.rson[p]
        } else {
            let mut q = self.lson[p];
            if self.rson[q] != NIL {
                while self.rson[q] != NIL {
                    q = self.rson[q];
                }
                self.rson[self.dad[q]] = self.lson[q];
                self.dad[self.lson[q]] = self.dad[q];
                self.lson[q] = self.lson[p];
                self.dad[self.lson[p]] = q;
            }
            self.rson[q] = self.rson[p];
            self.dad[self.rson[p]] = q;
            q
        };

        self.dad[q] = self.dad[p];
        if self.rson[self.dad[p]] == p {
            self.rson[self.dad[p]] = q;
        } else {
            self.lson[self.dad[p]] = q;
        }
        self.dad[p] = NIL;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lzhuf_empty_and_small_round_trip() {
        let compressed_empty = compress_lzhuf(b"");
        assert_eq!(compressed_empty, vec![0, 0, 0, 0]);
        assert_eq!(decompress_lzhuf(&compressed_empty), Ok(Vec::new()));

        let small = b"xray";
        let comp = compress_lzhuf(small);
        assert_eq!(decompress_lzhuf(&comp), Ok(small.to_vec()));
    }

    #[test]
    fn test_lzhuf_large_repetitive_round_trip() {
        let mut payload = Vec::new();
        for idx in 0..250 {
            payload.extend_from_slice(
                format!("[section_{}]\ntexture = wpn\\wpn_ak74\nvalue = {}\n", idx, idx * 7)
                    .as_bytes(),
            );
        }
        let compressed = compress_lzhuf(&payload);
        assert!(compressed.len() < payload.len());
        let restored = decompress_lzhuf(&compressed).expect("lzhuf decode must succeed");
        assert_eq!(restored, payload);
    }

    #[test]
    fn test_lzhuf_guards_malformed_and_oversized_inputs() {
        assert_eq!(decompress_lzhuf(&[1, 2, 3]), Err(LzhufError::UnexpectedEof));
        // Declared size > limit
        let huge = [0xFF, 0xFF, 0xFF, 0x7F, 0x00];
        assert!(matches!(
            decompress_lzhuf_with_limit(&huge, 1024),
            Err(LzhufError::OutputLimitExceeded { .. })
        ));
        // Truncated stream body
        let truncated = [64, 0, 0, 0];
        assert_eq!(decompress_lzhuf(&truncated), Err(LzhufError::UnexpectedEof));
    }
}
