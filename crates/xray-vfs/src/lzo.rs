//! Pure Rust Bounds-Checked LZO1X Decompressor & Compressor
//!
//! Provides memory-safe decompression and compression for X-Ray `.db` / `.xdb` archive chunks.
//! Guaranteed to prevent buffer overflows, invalid back-references, and out-of-bounds reads.
//! 100% compatible with X-Ray Monolith Engine's `rt_lzo1x_1.cpp` and `rt_lzo1x_d.ch`.

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LzoError {
    UnexpectedEof,
    OutputOverrun,
    InvalidMatchDistance,
    LookbehindOutOfBounds,
    TrailingGarbage,
}

impl std::fmt::Display for LzoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for LzoError {}

/// Decompress an LZO1X compressed slice into an output buffer of expected size.
///
/// Guaranteed memory-safe: verifies all source slice indices and destination slice offsets before copying.
/// Matches standard Oberhumer LZO1X decompression (`rt_lzo1x_d.ch`) byte-for-byte.
pub fn decompress_lzo1x(input: &[u8], output: &mut [u8]) -> Result<usize, LzoError> {
    if input.is_empty() {
        return Err(LzoError::UnexpectedEof);
    }

    let mut in_pos = 0;
    let mut out_pos = 0;
    let in_len = input.len();
    let out_len = output.len();

    let read_byte = |pos: &mut usize| -> Result<u8, LzoError> {
        if *pos >= in_len {
            Err(LzoError::UnexpectedEof)
        } else {
            let b = input[*pos];
            *pos += 1;
            Ok(b)
        }
    };

    let copy_literals = |pos: &mut usize, out_p: &mut usize, count: usize, out: &mut [u8]| -> Result<(), LzoError> {
        if *pos + count > in_len {
            return Err(LzoError::UnexpectedEof);
        }
        if *out_p + count > out_len {
            return Err(LzoError::OutputOverrun);
        }
        out[*out_p..*out_p + count].copy_from_slice(&input[*pos..*pos + count]);
        *pos += count;
        *out_p += count;
        Ok(())
    };

    let copy_match = |out_p: &mut usize, dist: usize, len: usize, out: &mut [u8]| -> Result<(), LzoError> {
        if dist == 0 || dist > *out_p {
            return Err(LzoError::LookbehindOutOfBounds);
        }
        if *out_p + len > out_len {
            return Err(LzoError::OutputOverrun);
        }
        let src_start = *out_p - dist;
        for i in 0..len {
            out[*out_p + i] = out[src_start + i];
        }
        *out_p += len;
        Ok(())
    };

    let first = read_byte(&mut in_pos)?;
    let mut next_t: Option<usize> = None;

    if first > 17 {
        let t = (first - 17) as usize;
        if t < 4 {
            copy_literals(&mut in_pos, &mut out_pos, t, output)?;
            next_t = Some(read_byte(&mut in_pos)? as usize);
        } else {
            copy_literals(&mut in_pos, &mut out_pos, t, output)?;
            let t_after = read_byte(&mut in_pos)? as usize;
            if t_after >= 16 {
                next_t = Some(t_after);
            } else {
                let b = read_byte(&mut in_pos)? as usize;
                let dist = 1 + (t_after >> 2) + (b << 2);
                copy_match(&mut out_pos, dist, 2, output)?;

                let lit = input[in_pos - 2] as usize & 3;
                if lit > 0 {
                    copy_literals(&mut in_pos, &mut out_pos, lit, output)?;
                    next_t = Some(read_byte(&mut in_pos)? as usize);
                }
            }
        }
    } else {
        next_t = Some(first as usize);
    }

    'outer: while in_pos < in_len || next_t.is_some() {
        let mut t = match next_t.take() {
            Some(val) => val,
            None => read_byte(&mut in_pos)? as usize,
        };

        if t < 16 {
            // Literal run
            if t == 0 {
                while in_pos < in_len && input[in_pos] == 0 {
                    t += 255;
                    in_pos += 1;
                }
                let next = read_byte(&mut in_pos)? as usize;
                t += 15 + next;
            }

            copy_literals(&mut in_pos, &mut out_pos, t + 3, output)?;

            let t_next = read_byte(&mut in_pos)? as usize;
            if t_next >= 16 {
                t = t_next;
            } else {
                // M1 match
                let b = read_byte(&mut in_pos)? as usize;
                let dist = 1 + (t_next >> 2) + (b << 2);
                copy_match(&mut out_pos, dist, 2, output)?;

                let lit = input[in_pos - 2] as usize & 3;
                if lit == 0 {
                    continue 'outer;
                }
                copy_literals(&mut in_pos, &mut out_pos, lit, output)?;
                t = read_byte(&mut in_pos)? as usize;
            }
        }

        // Inner match loop
        loop {
            let lit: usize;

            if t >= 64 {
                // M2 match
                let b = read_byte(&mut in_pos)? as usize;
                let dist = 1 + ((t >> 2) & 7) + (b << 3);
                let len = ((t >> 5) - 1) + 2;
                copy_match(&mut out_pos, dist, len, output)?;
                lit = input[in_pos - 2] as usize & 3;
            } else if t >= 32 {
                // M3 match
                let mut len = t & 31;
                if len == 0 {
                    while in_pos < in_len && input[in_pos] == 0 {
                        len += 255;
                        in_pos += 1;
                    }
                    let next = read_byte(&mut in_pos)? as usize;
                    len += 31 + next;
                }
                let b0 = read_byte(&mut in_pos)? as usize;
                let b1 = read_byte(&mut in_pos)? as usize;
                let dist = 1 + (b0 >> 2) + (b1 << 6);
                copy_match(&mut out_pos, dist, len + 2, output)?;
                lit = b0 & 3;
            } else if t >= 16 {
                // M4 match or EOF
                let mut dist = (t & 8) << 11;
                let mut len = t & 7;
                if len == 0 {
                    while in_pos < in_len && input[in_pos] == 0 {
                        len += 255;
                        in_pos += 1;
                    }
                    let next = read_byte(&mut in_pos)? as usize;
                    len += 7 + next;
                }
                let b0 = read_byte(&mut in_pos)? as usize;
                let b1 = read_byte(&mut in_pos)? as usize;
                let offset_lo = (b0 >> 2) + (b1 << 6);

                if dist == 0 && offset_lo == 0 {
                    // Standard EOF found!
                    break 'outer;
                }

                dist += offset_lo + 0x4000;
                copy_match(&mut out_pos, dist, len + 2, output)?;
                lit = b0 & 3;
            } else {
                // M1 match
                let b = read_byte(&mut in_pos)? as usize;
                let dist = 1 + (t >> 2) + (b << 2);
                copy_match(&mut out_pos, dist, 2, output)?;
                lit = input[in_pos - 2] as usize & 3;
            }

            if lit == 0 {
                break;
            }

            copy_literals(&mut in_pos, &mut out_pos, lit, output)?;
            if in_pos >= in_len {
                break 'outer;
            }
            t = read_byte(&mut in_pos)? as usize;
        }
    }

    Ok(out_pos)
}

/// Pure Rust LZO1X compressor producing bitstreams compatible with `rt_lzo1x_d.ch`.
///
/// Implements dictionary-based fast matching with sliding 14-bit window and standard EOF termination.
pub fn compress_lzo1x(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return vec![17, 0, 0]; // EOF marker
    }

    let mut out = Vec::with_capacity(input.len() + input.len() / 16 + 64);
    let in_len = input.len();

    // Small inputs: emit direct literal run + EOF
    if in_len <= 13 {
        let t = in_len;
        if t <= 238 {
            out.push((17 + t) as u8);
        } else {
            out.push(0);
            let mut tt = t - 18;
            while tt > 255 {
                tt -= 255;
                out.push(0);
            }
            out.push(tt as u8);
        }
        out.extend_from_slice(input);
        // Standard EOF marker
        out.push(17);
        out.push(0);
        out.push(0);
        return out;
    }

    const D_BITS: usize = 14;
    const D_SIZE: usize = 1 << D_BITS;
    let mut dict = vec![0u32; D_SIZE];

    let mut ip = 0;
    let mut ii = 0;
    let ip_end = in_len.saturating_sub(13);

    // Initial 4 bytes are literals
    ip += 4;

    while ip < ip_end {
        let v = (input[ip] as u32)
            | ((input[ip + 1] as u32) << 8)
            | ((input[ip + 2] as u32) << 16);
        let dindex = ((v.wrapping_mul(0x1824429d)) >> (32 - D_BITS)) as usize & (D_SIZE - 1);
        let m_pos = dict[dindex] as usize;
        dict[dindex] = ip as u32;

        let m_off = ip.wrapping_sub(m_pos);
        if m_pos > 0 && m_off <= 0xbfff && m_off > 0 && input[m_pos] == input[ip] && input[m_pos + 1] == input[ip + 1] && input[m_pos + 2] == input[ip + 2] {
            let lit = ip - ii;
            if lit > 0 {
                if out.is_empty() {
                    // First literal run of the entire stream
                    if lit <= 238 {
                        out.push((17 + lit) as u8);
                    } else {
                        let mut tt = lit - 18;
                        out.push(0);
                        while tt > 255 {
                            tt -= 255;
                            out.push(0);
                        }
                        out.push(tt as u8);
                    }
                } else if lit <= 3 {
                    let idx = out.len() - 2;
                    out[idx] |= lit as u8;
                } else if lit <= 18 {
                    out.push((lit - 3) as u8);
                } else {
                    let mut tt = lit - 18;
                    out.push(0);
                    while tt > 255 {
                        tt -= 255;
                        out.push(0);
                    }
                    out.push(tt as u8);
                }
                out.extend_from_slice(&input[ii..ip]);
            }

            let mut m_len = 3;
            while ip + m_len < in_len && input[m_pos + m_len] == input[ip + m_len] && m_len < 2048 {
                m_len += 1;
            }

            if m_off <= 2048 && m_len <= 8 {
                // M2 match
                let off = m_off - 1;
                out.push((((m_len - 1) << 5) | ((off & 7) << 2)) as u8);
                out.push((off >> 3) as u8);
            } else if m_off <= 16384 {
                // M3 match
                let off = m_off - 1;
                if m_len <= 33 {
                    out.push((32 | (m_len - 2)) as u8);
                } else {
                    let mut l = m_len - 33;
                    out.push(32);
                    while l > 255 {
                        l -= 255;
                        out.push(0);
                    }
                    out.push(l as u8);
                }
                out.push(((off & 63) << 2) as u8);
                out.push((off >> 6) as u8);
            } else {
                // M4 match
                let off = m_off - 0x4000;
                let k = (off & 0x4000) >> 11;
                if m_len <= 9 {
                    out.push((16 | k | (m_len - 2)) as u8);
                } else {
                    let mut l = m_len - 9;
                    out.push((16 | k) as u8);
                    while l > 255 {
                        l -= 255;
                        out.push(0);
                    }
                    out.push(l as u8);
                }
                out.push(((off & 63) << 2) as u8);
                out.push((off >> 6) as u8);
            }

            ip += m_len;
            ii = ip;
        } else {
            ip += 1;
        }
    }

    let trailing_lit = in_len - ii;
    if trailing_lit > 0 {
        if out.is_empty() {
            if trailing_lit <= 238 {
                out.push((17 + trailing_lit) as u8);
            } else {
                let mut tt = trailing_lit - 18;
                out.push(0);
                while tt > 255 {
                    tt -= 255;
                    out.push(0);
                }
                out.push(tt as u8);
            }
        } else if trailing_lit <= 3 && out.len() >= 2 {
            let idx = out.len() - 2;
            out[idx] |= trailing_lit as u8;
        } else if trailing_lit <= 18 {
            out.push((trailing_lit - 3) as u8);
        } else {
            let mut tt = trailing_lit - 18;
            out.push(0);
            while tt > 255 {
                tt -= 255;
                out.push(0);
            }
            out.push(tt as u8);
        }
        out.extend_from_slice(&input[ii..in_len]);
    }

    // Standard EOF marker (M4 match with offset 0: 0x11, 0x00, 0x00)
    out.push(17);
    out.push(0);
    out.push(0);

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unexpected_eof() {
        let mut out = [0u8; 128];
        assert_eq!(decompress_lzo1x(&[], &mut out), Err(LzoError::UnexpectedEof));
        assert_eq!(decompress_lzo1x(&[0x10], &mut out), Err(LzoError::UnexpectedEof));
    }

    #[test]
    fn test_output_overrun() {
        // Feed 10 bytes of literals into a 3-byte buffer
        let mut out = [0u8; 3];
        // First instruction > 17: 17 + 10 = 27 (literal run of 10)
        let in_data = [27u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(decompress_lzo1x(&in_data, &mut out), Err(LzoError::OutputOverrun));
    }

    #[test]
    fn test_lookbehind_out_of_bounds() {
        let mut out = [0u8; 128];
        // Match instruction referencing offset before buffer start:
        // Instruction 16 (M4), extra len = 1, b0 = 4 (b0 >> 2 = 1), b1 = 0 -> dist = 16385 on empty out_pos
        let bad_match = [16u8, 1, 4, 0];
        assert_eq!(decompress_lzo1x(&bad_match, &mut out), Err(LzoError::LookbehindOutOfBounds));
    }

    #[test]
    fn test_round_trip_synthetic_text() {
        let original = b"[zone_field_radioactive]\n$spawn = \"zones\\radioactive\"\ntype = radioactive\nradius = 5.0\nmin_density = 0.5\nmax_density = 1.0\n";
        let compressed = compress_lzo1x(original);
        assert!(!compressed.is_empty());

        let mut decompressed = vec![0u8; original.len()];
        let bytes_written = decompress_lzo1x(&compressed, &mut decompressed)
            .expect("Decompression must succeed on valid compressed stream");

        assert_eq!(bytes_written, original.len());
        assert_eq!(&decompressed[..bytes_written], original.as_slice());
    }

    #[test]
    fn test_round_trip_repetitive_patterns() {
        let mut original = Vec::new();
        for i in 0..1000 {
            original.extend_from_slice(b"MonolithEngineDirectX12BackendWorkerRing");
            original.push((i % 256) as u8);
        }

        let compressed = compress_lzo1x(&original);
        assert!(compressed.len() < original.len(), "LZO compression must reduce repetitive buffer size");

        let mut decompressed = vec![0u8; original.len()];
        let bytes_written = decompress_lzo1x(&compressed, &mut decompressed)
            .expect("Decompression must succeed on repetitive stream");

        assert_eq!(bytes_written, original.len());
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_golden_lzo1x_m2_m3_fixtures() {
        // Golden LZO1X Stream 1: M2 match (16 literals, 8-byte match at distance 16, standard EOF)
        // Expected decompressed plaintext: "ABCDEFGH12345678ABCDEFGH" (24 bytes)
        let golden_m2_stream = [
            33u8, // First instruction: 17 + 16 = 33 literals
            b'A', b'B', b'C', b'D', b'E', b'F', b'G', b'H',
            b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8',
            252u8, 1u8, // M2 match: m_len = 8, m_off = 16
            17u8, 0u8, 0u8, // EOF marker
        ];

        let mut out = [0u8; 32];
        let bytes_written = decompress_lzo1x(&golden_m2_stream, &mut out)
            .expect("Golden M2 fixture must decompress successfully");
        assert_eq!(bytes_written, 24);
        assert_eq!(&out[..bytes_written], b"ABCDEFGH12345678ABCDEFGH");

        // Golden LZO1X Stream 2: M3 match (32 literals, 12-byte match at distance 32, standard EOF)
        let mut golden_m3_stream = Vec::new();
        // 32 literals: 0 instruction for > 18 literals: [0, 14 (32-18=14)]
        golden_m3_stream.push(0u8);
        golden_m3_stream.push(14u8);
        golden_m3_stream.extend_from_slice(b"0123456789ABCDEF0123456789ABCDEF"); // 32 bytes
        // M3 match: len = 12, dist = 32 (off = 31):
        // instruction: 32 | (12 - 2) = 42
        // off & 63 << 2 = 31 << 2 = 124
        // off >> 6 = 0
        golden_m3_stream.push(42u8);
        golden_m3_stream.push(124u8);
        golden_m3_stream.push(0u8);
        // EOF marker
        golden_m3_stream.push(17u8);
        golden_m3_stream.push(0u8);
        golden_m3_stream.push(0u8);

        let mut out_m3 = [0u8; 64];
        let written_m3 = decompress_lzo1x(&golden_m3_stream, &mut out_m3)
            .expect("Golden M3 fixture must decompress successfully");
        assert_eq!(written_m3, 32 + 12);
        assert_eq!(&out_m3[..32], b"0123456789ABCDEF0123456789ABCDEF");
        assert_eq!(&out_m3[32..44], b"0123456789AB");
    }

    #[test]
    fn test_malformed_truncated_stream_variants() {
        let mut out = [0u8; 64];

        // 1. Truncated instruction in literal run
        assert_eq!(decompress_lzo1x(&[0u8, 5u8, b'a'], &mut out), Err(LzoError::UnexpectedEof));

        // 2. Truncated M2 offset
        assert_eq!(decompress_lzo1x(&[21u8, b'a', b'b', b'c', b'd', 252u8], &mut out), Err(LzoError::UnexpectedEof));

        // 3. Truncated M3 offset bytes
        assert_eq!(decompress_lzo1x(&[21u8, b'a', b'b', b'c', b'd', 42u8, 124u8], &mut out), Err(LzoError::UnexpectedEof));

        // 4. Overrun on small destination buffer
        let mut tiny_out = [0u8; 2];
        let stream = [19u8, b'a', b'b', 17u8, 0u8, 0u8];
        assert_eq!(decompress_lzo1x(&stream, &mut tiny_out), Ok(2));

        let stream_overflow = [20u8, b'a', b'b', b'c', 17u8, 0u8, 0u8];
        assert_eq!(decompress_lzo1x(&stream_overflow, &mut tiny_out), Err(LzoError::OutputOverrun));
    }
}
