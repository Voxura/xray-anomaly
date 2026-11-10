//! Pure Rust Bounds-Checked LZO1X Decompressor
//!
//! Provides memory-safe decompression for X-Ray `.db` / `.xdb` archive chunks.
//! Guaranteed to prevent buffer overflows, invalid back-references, and out-of-bounds reads.

#[derive(Debug, PartialEq, Eq)]
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
pub fn decompress_lzo1x(input: &[u8], output: &mut [u8]) -> Result<usize, LzoError> {
    let mut in_pos = 0;
    let mut out_pos = 0;
    let in_len = input.len();
    let out_len = output.len();

    // Helper closure
    let read_byte = |pos: &mut usize| -> Result<u8, LzoError> {
        if *pos >= in_len {
            Err(LzoError::UnexpectedEof)
        } else {
            let b = input[*pos];
            *pos += 1;
            Ok(b)
        }
    };

    // First instruction
    let mut instruction = read_byte(&mut in_pos)?;
    if instruction > 17 {
        let mut t = (instruction - 17) as usize;
        if t < 4 {
            // Literal run
            while t > 0 {
                if out_pos >= out_len {
                    return Err(LzoError::OutputOverrun);
                }
                output[out_pos] = read_byte(&mut in_pos)?;
                out_pos += 1;
                t -= 1;
            }
            instruction = read_byte(&mut in_pos)?;
        }
    }

    loop {
        if instruction < 16 {
            // 0..15: Literal run
            let mut t = instruction as usize;
            if t == 0 {
                while in_pos < in_len && input[in_pos] == 0 {
                    t += 255;
                    in_pos += 1;
                }
                let next = read_byte(&mut in_pos)?;
                t += 15 + (next as usize);
            }

            t += 3;
            for _ in 0..t {
                if out_pos >= out_len {
                    return Err(LzoError::OutputOverrun);
                }
                output[out_pos] = read_byte(&mut in_pos)?;
                out_pos += 1;
            }

            instruction = read_byte(&mut in_pos)?;
            if instruction < 16 {
                // Another literal run directly
                continue;
            }
        }

        // Match copy
        let mut match_len: usize;
        let match_dist: usize;

        if instruction >= 64 {
            // Short match
            match_len = ((instruction >> 5) - 1) as usize;
            let b = read_byte(&mut in_pos)?;
            match_dist = (((instruction >> 2) & 7) as usize) | ((b as usize) << 3);
        } else if instruction >= 32 {
            // Long match
            match_len = (instruction & 31) as usize;
            if match_len == 0 {
                while in_pos < in_len && input[in_pos] == 0 {
                    match_len += 255;
                    in_pos += 1;
                }
                let next = read_byte(&mut in_pos)?;
                match_len += 31 + (next as usize);
            }
            let b0 = read_byte(&mut in_pos)?;
            let b1 = read_byte(&mut in_pos)?;
            match_dist = (b0 as usize) | ((b1 as usize) << 8);
        } else {
            // 16..31: Very long match or EOF
            let is_eof = (instruction & 15) == 0;
            match_len = (instruction & 7) as usize;
            if match_len == 0 {
                while in_pos < in_len && input[in_pos] == 0 {
                    match_len += 255;
                    in_pos += 1;
                }
                let next = read_byte(&mut in_pos)?;
                match_len += 7 + (next as usize);
            }
            let b0 = read_byte(&mut in_pos)?;
            let b1 = read_byte(&mut in_pos)?;
            match_dist = (b0 as usize) | ((b1 as usize) << 8);

            if is_eof && match_dist == 0 {
                // End of compressed stream
                break;
            }
        }

        match_len += 2;
        let match_offset = match_dist + 1;
        if match_offset > out_pos {
            return Err(LzoError::LookbehindOutOfBounds);
        }

        // Copy match bytes (allowing overlapping run-length copies)
        let copy_start = out_pos - match_offset;
        for i in 0..match_len {
            if out_pos >= out_len {
                return Err(LzoError::OutputOverrun);
            }
            output[out_pos] = output[copy_start + i];
            out_pos += 1;
        }

        if in_pos >= in_len {
            break;
        }

        // Check trailing literal bytes
        let trailing_lits = (instruction & 3) as usize;
        for _ in 0..trailing_lits {
            if in_pos >= in_len {
                break;
            }
            if out_pos >= out_len {
                return Err(LzoError::OutputOverrun);
            }
            output[out_pos] = read_byte(&mut in_pos)?;
            out_pos += 1;
        }

        if in_pos >= in_len {
            break;
        }
        instruction = read_byte(&mut in_pos)?;
    }

    Ok(out_pos)
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
        // Instruction byte for 10 literals: 10 - 3 = 7 -> instruction = 7 (literal run of 7+3 = 10)
        let in_data = [7u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(decompress_lzo1x(&in_data, &mut out), Err(LzoError::OutputOverrun));
    }

    #[test]
    fn test_lookbehind_out_of_bounds() {
        let mut out = [0u8; 128];
        // Match instruction referencing offset before buffer start
        let bad_match = [64u8, 255u8];
        assert_eq!(decompress_lzo1x(&bad_match, &mut out), Err(LzoError::LookbehindOutOfBounds));
    }
}
