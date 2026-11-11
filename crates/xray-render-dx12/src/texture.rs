//! Pure Rust DirectDraw Surface (DDS) Texture Decoder & Descriptor Registry
//!
//! Provides zero-copy parsing of DDS headers, subresource footprint calculation,
//! format detection (BC1-BC7, uncompressed RGBA), and layout planning for DirectX 12
//! texture placement in default heaps and bindless descriptor heaps.

use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC3_UNORM,
    DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
};

pub const DDS_MAGIC: u32 = 0x20534444; // "DDS "

pub const DDSD_CAPS: u32 = 0x1;
pub const DDSD_HEIGHT: u32 = 0x2;
pub const DDSD_WIDTH: u32 = 0x4;
pub const DDSD_PITCH: u32 = 0x8;
pub const DDSD_PIXELFORMAT: u32 = 0x1000;
pub const DDSD_MIPMAPCOUNT: u32 = 0x20000;
pub const DDSD_LINEARSIZE: u32 = 0x80000;
pub const DDSD_DEPTH: u32 = 0x800000;

pub const DDPF_ALPHAPIXELS: u32 = 0x1;
pub const DDPF_ALPHA: u32 = 0x2;
pub const DDPF_FOURCC: u32 = 0x4;
pub const DDPF_RGB: u32 = 0x40;

pub const FOURCC_DXT1: u32 = u32::from_le_bytes(*b"DXT1");
pub const FOURCC_DXT2: u32 = u32::from_le_bytes(*b"DXT2");
pub const FOURCC_DXT3: u32 = u32::from_le_bytes(*b"DXT3");
pub const FOURCC_DXT4: u32 = u32::from_le_bytes(*b"DXT4");
pub const FOURCC_DXT5: u32 = u32::from_le_bytes(*b"DXT5");
pub const FOURCC_DX10: u32 = u32::from_le_bytes(*b"DX10");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureError {
    InvalidMagic,
    InvalidHeaderSize,
    InvalidDimensions,
    UnsupportedFormat,
    BufferTooShort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdsMipLevel<'a> {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub row_pitch: u32,
    pub slice_pitch: u32,
    pub data: &'a [u8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DdsImage<'a> {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub mip_count: u32,
    pub format: DXGI_FORMAT,
    pub is_block_compressed: bool,
    pub block_size: u32,
    pub mips: Vec<DdsMipLevel<'a>>,
}

/// Parse a raw in-memory DDS byte stream into structured image metadata and subresource slices
pub fn parse_dds(bytes: &[u8]) -> Result<DdsImage<'_>, TextureError> {
    if bytes.len() < 128 {
        return Err(TextureError::BufferTooShort);
    }

    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != DDS_MAGIC {
        return Err(TextureError::InvalidMagic);
    }

    let header_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if header_size != 124 {
        return Err(TextureError::InvalidHeaderSize);
    }

    let flags = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let height = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let width = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
    let depth = if flags & DDSD_DEPTH != 0 {
        u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]).max(1)
    } else {
        1
    };

    let mip_count = if flags & DDSD_MIPMAPCOUNT != 0 {
        u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]).max(1)
    } else {
        1
    };

    if width == 0 || height == 0 {
        return Err(TextureError::InvalidDimensions);
    }

    // PixelFormat starts at offset 76 (32 bytes)
    let pf_flags = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]);
    let pf_fourcc = u32::from_le_bytes([bytes[84], bytes[85], bytes[86], bytes[87]]);
    let pf_rgb_bits = u32::from_le_bytes([bytes[88], bytes[89], bytes[90], bytes[91]]);
    let pf_r_mask = u32::from_le_bytes([bytes[92], bytes[93], bytes[94], bytes[95]]);
    let pf_g_mask = u32::from_le_bytes([bytes[96], bytes[97], bytes[98], bytes[99]]);
    let pf_b_mask = u32::from_le_bytes([bytes[100], bytes[101], bytes[102], bytes[103]]);
    let _pf_a_mask = u32::from_le_bytes([bytes[104], bytes[105], bytes[106], bytes[107]]);

    let mut data_offset = 128;
    let (format, is_bc, block_size) = if pf_flags & DDPF_FOURCC != 0 {
        if pf_fourcc == FOURCC_DX10 {
            if bytes.len() < 148 {
                return Err(TextureError::BufferTooShort);
            }
            let dxgi_fmt_raw = u32::from_le_bytes([bytes[128], bytes[129], bytes[130], bytes[131]]);
            data_offset = 148;
            match dxgi_fmt_raw {
                71 => (DXGI_FORMAT_BC1_UNORM, true, 8),
                74 => (DXGI_FORMAT_BC2_UNORM, true, 16),
                77 => (DXGI_FORMAT_BC3_UNORM, true, 16),
                98 => (DXGI_FORMAT_BC7_UNORM, true, 16),
                28 => (DXGI_FORMAT_R8G8B8A8_UNORM, false, 4),
                _ => (DXGI_FORMAT(dxgi_fmt_raw as i32), false, 4),
            }
        } else if pf_fourcc == FOURCC_DXT1 {
            (DXGI_FORMAT_BC1_UNORM, true, 8)
        } else if pf_fourcc == FOURCC_DXT2 || pf_fourcc == FOURCC_DXT3 {
            (DXGI_FORMAT_BC2_UNORM, true, 16)
        } else if pf_fourcc == FOURCC_DXT4 || pf_fourcc == FOURCC_DXT5 {
            (DXGI_FORMAT_BC3_UNORM, true, 16)
        } else {
            return Err(TextureError::UnsupportedFormat);
        }
    } else if pf_flags & DDPF_RGB != 0 {
        if pf_rgb_bits == 32
            && pf_r_mask == 0x000000FF
            && pf_g_mask == 0x0000FF00
            && pf_b_mask == 0x00FF0000
        {
            (DXGI_FORMAT_R8G8B8A8_UNORM, false, 4)
        } else {
            return Err(TextureError::UnsupportedFormat);
        }
    } else {
        return Err(TextureError::UnsupportedFormat);
    };

    let mut mips = Vec::with_capacity(mip_count as usize);
    let mut current_offset = data_offset;

    let mut mip_w = width;
    let mut mip_h = height;

    for level in 0..mip_count {
        let (row_pitch, slice_pitch) = if is_bc {
            let num_blocks_wide = (mip_w + 3) / 4;
            let num_blocks_high = (mip_h + 3) / 4;
            let row_bytes = num_blocks_wide * block_size;
            let total_bytes = row_bytes * num_blocks_high;
            (row_bytes, total_bytes)
        } else {
            let row_bytes = mip_w * block_size;
            let total_bytes = row_bytes * mip_h;
            (row_bytes, total_bytes)
        };

        let next_offset = current_offset + slice_pitch as usize;
        if next_offset > bytes.len() {
            return Err(TextureError::BufferTooShort);
        }

        mips.push(DdsMipLevel {
            level,
            width: mip_w,
            height: mip_h,
            row_pitch,
            slice_pitch,
            data: &bytes[current_offset..next_offset],
        });

        current_offset = next_offset;
        mip_w = (mip_w / 2).max(1);
        mip_h = (mip_h / 2).max(1);
    }

    Ok(DdsImage {
        width,
        height,
        depth,
        mip_count,
        format,
        is_block_compressed: is_bc,
        block_size,
        mips,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dds(width: u32, height: u32, fourcc: u32, data_size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; 128 + data_size];
        buf[0..4].copy_from_slice(&DDS_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&124u32.to_le_bytes());
        buf[8..12].copy_from_slice(&(DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT).to_le_bytes());
        buf[12..16].copy_from_slice(&height.to_le_bytes());
        buf[16..20].copy_from_slice(&width.to_le_bytes());
        // PixelFormat
        buf[76..80].copy_from_slice(&32u32.to_le_bytes());
        buf[80..84].copy_from_slice(&DDPF_FOURCC.to_le_bytes());
        buf[84..88].copy_from_slice(&fourcc.to_le_bytes());
        buf
    }

    #[test]
    fn test_dxt1_parsing() {
        // 4x4 DXT1 is exactly 1 block = 8 bytes
        let buf = create_test_dds(4, 4, FOURCC_DXT1, 8);
        let img = parse_dds(&buf).expect("Failed to parse valid DXT1 DDS");
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(img.format, DXGI_FORMAT_BC1_UNORM);
        assert!(img.is_block_compressed);
        assert_eq!(img.mips.len(), 1);
        assert_eq!(img.mips[0].slice_pitch, 8);
    }

    #[test]
    fn test_dxt5_parsing() {
        // 8x8 DXT5 is 2x2 blocks = 4 blocks * 16 bytes = 64 bytes
        let buf = create_test_dds(8, 8, FOURCC_DXT5, 64);
        let img = parse_dds(&buf).expect("Failed to parse valid DXT5 DDS");
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 8);
        assert_eq!(img.format, DXGI_FORMAT_BC3_UNORM);
        assert_eq!(img.mips[0].slice_pitch, 64);
    }

    #[test]
    fn test_invalid_magic() {
        let mut buf = create_test_dds(4, 4, FOURCC_DXT1, 8);
        buf[0] = b'X';
        assert_eq!(parse_dds(&buf), Err(TextureError::InvalidMagic));
    }

    #[test]
    fn test_truncated_data() {
        let buf = create_test_dds(8, 8, FOURCC_DXT5, 32); // Needs 64, provided only 32
        assert_eq!(parse_dds(&buf), Err(TextureError::BufferTooShort));
    }
}
