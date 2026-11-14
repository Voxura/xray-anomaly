//! Pure Rust DirectDraw Surface (DDS) Texture Decoder & Descriptor Registry
//!
//! Provides zero-copy parsing of DDS headers, subresource footprint calculation,
//! format detection (BC1-BC7, uncompressed RGBA), layout planning for DirectX 12
//! texture placement in default heaps, staging buffer uploading, and bindless SRV registration.

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use windows::core::{Result as WinResult, HRESULT};
use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12GraphicsCommandList, ID3D12Resource,
    D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_UPLOAD,
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT, D3D12_RANGE,
    D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_DIMENSION_TEXTURE2D,
    D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ,
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_TRANSITION_BARRIER,
    D3D12_SHADER_RESOURCE_VIEW_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC_0,
    D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_TEX2D_SRV,
    D3D12_TEXTURE_COPY_LOCATION, D3D12_TEXTURE_COPY_LOCATION_0,
    D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_TEXTURE_LAYOUT_UNKNOWN,
    D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_A8_UNORM, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC2_UNORM,
    DXGI_FORMAT_BC2_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM_SRGB,
    DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_SNORM, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_SNORM,
    DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};

use crate::bindless::DescriptorIndex;

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
pub const FOURCC_ATI1: u32 = u32::from_le_bytes(*b"ATI1");
pub const FOURCC_BC4U: u32 = u32::from_le_bytes(*b"BC4U");
pub const FOURCC_BC4S: u32 = u32::from_le_bytes(*b"BC4S");
pub const FOURCC_ATI2: u32 = u32::from_le_bytes(*b"ATI2");
pub const FOURCC_BC5U: u32 = u32::from_le_bytes(*b"BC5U");
pub const FOURCC_BC5S: u32 = u32::from_le_bytes(*b"BC5S");
pub const FOURCC_DX10: u32 = u32::from_le_bytes(*b"DX10");

fn failure() -> windows::core::Error {
    windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

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
                72 => (DXGI_FORMAT_BC1_UNORM_SRGB, true, 8),
                74 => (DXGI_FORMAT_BC2_UNORM, true, 16),
                75 => (DXGI_FORMAT_BC2_UNORM_SRGB, true, 16),
                77 => (DXGI_FORMAT_BC3_UNORM, true, 16),
                78 => (DXGI_FORMAT_BC3_UNORM_SRGB, true, 16),
                80 => (DXGI_FORMAT_BC4_UNORM, true, 8),
                81 => (DXGI_FORMAT_BC4_SNORM, true, 8),
                83 => (DXGI_FORMAT_BC5_UNORM, true, 16),
                84 => (DXGI_FORMAT_BC5_SNORM, true, 16),
                87 => (DXGI_FORMAT_B8G8R8A8_UNORM, false, 4),
                91 => (DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, false, 4),
                98 => (DXGI_FORMAT_BC7_UNORM, true, 16),
                99 => (DXGI_FORMAT_BC7_UNORM_SRGB, true, 16),
                28 => (DXGI_FORMAT_R8G8B8A8_UNORM, false, 4),
                29 => (DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, false, 4),
                _ => (DXGI_FORMAT(dxgi_fmt_raw as i32), false, 4),
            }
        } else if pf_fourcc == FOURCC_DXT1 {
            (DXGI_FORMAT_BC1_UNORM, true, 8)
        } else if pf_fourcc == FOURCC_DXT2 || pf_fourcc == FOURCC_DXT3 {
            (DXGI_FORMAT_BC2_UNORM, true, 16)
        } else if pf_fourcc == FOURCC_DXT4 || pf_fourcc == FOURCC_DXT5 {
            (DXGI_FORMAT_BC3_UNORM, true, 16)
        } else if pf_fourcc == FOURCC_ATI1 || pf_fourcc == FOURCC_BC4U {
            (DXGI_FORMAT_BC4_UNORM, true, 8)
        } else if pf_fourcc == FOURCC_BC4S {
            (DXGI_FORMAT_BC4_SNORM, true, 8)
        } else if pf_fourcc == FOURCC_ATI2 || pf_fourcc == FOURCC_BC5U {
            (DXGI_FORMAT_BC5_UNORM, true, 16)
        } else if pf_fourcc == FOURCC_BC5S {
            (DXGI_FORMAT_BC5_SNORM, true, 16)
        } else {
            return Err(TextureError::UnsupportedFormat);
        }
    } else if pf_flags & DDPF_RGB != 0 {
        if pf_rgb_bits == 32 {
            if pf_r_mask == 0x000000FF && pf_g_mask == 0x0000FF00 && pf_b_mask == 0x00FF0000 {
                // RGBA
                (DXGI_FORMAT_R8G8B8A8_UNORM, false, 4)
            } else if pf_r_mask == 0x00FF0000 && pf_g_mask == 0x0000FF00 && pf_b_mask == 0x000000FF {
                // BGRA (D3DFMT_A8R8G8B8 / D3DFMT_X8R8G8B8 common in S.T.A.L.K.E.R. UI textures)
                if pf_flags & DDPF_ALPHAPIXELS != 0 && _pf_a_mask != 0 {
                    (DXGI_FORMAT_B8G8R8A8_UNORM, false, 4)
                } else {
                    (DXGI_FORMAT_B8G8R8X8_UNORM, false, 4)
                }
            } else {
                return Err(TextureError::UnsupportedFormat);
            }
        } else {
            return Err(TextureError::UnsupportedFormat);
        }
    } else if (pf_flags & 0x00000002 != 0 || pf_flags & 0x00000001 != 0) && pf_rgb_bits == 8 {
        // D3DFMT_A8 (DDPF_ALPHA = 0x2 / DDPF_ALPHAPIXELS = 0x1) 8-bit uncompressed alpha texture
        (DXGI_FORMAT_A8_UNORM, false, 1)
    } else if pf_flags & 0x00020000 != 0 && pf_rgb_bits == 8 {
        // DDPF_LUMINANCE 8-bit texture
        (DXGI_FORMAT_R8_UNORM, false, 1)
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
            if !mips.is_empty() {
                // Buffer ended before all declared mip levels; keep valid decoded mips
                break;
            }
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

    let actual_mip_count = mips.len() as u32;
    Ok(DdsImage {
        width,
        height,
        depth,
        mip_count: actual_mip_count,
        format,
        is_block_compressed: is_bc,
        block_size,
        mips,
    })
}

/// GPU Texture placed in high-performance VRAM (D3D12_HEAP_TYPE_DEFAULT)
#[derive(Clone)]
pub struct GpuTexture {
    pub resource: ID3D12Resource,
    pub descriptor_index: DescriptorIndex,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub format: DXGI_FORMAT,
}

/// Helper that allocates a Default heap texture, a linear staging buffer, and records
/// the copy commands and transitions onto the supplied command list.
pub fn upload_dds_texture(
    device: &ID3D12Device,
    cmd: &ID3D12GraphicsCommandList,
    dds: &DdsImage,
) -> WinResult<(ID3D12Resource, ID3D12Resource)> {
    let mip_count = dds.mips.len().max(1);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: dds.width as u64,
        Height: dds.height,
        DepthOrArraySize: dds.depth as u16,
        MipLevels: mip_count as u16,
        Format: dds.format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };

    let default_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };

    let mut texture_res: Option<ID3D12Resource> = None;
    unsafe {
        let res = device.CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut texture_res,
        );
        if let Err(e) = res {
            eprintln!(
                "[upload_dds_texture] Failed to allocate texture resource ({}x{}, format {:?}, mips {}): {:?}",
                dds.width, dds.height, dds.format, mip_count, e
            );
            return Err(e);
        }
    }
    let texture_res = texture_res.ok_or_else(|| {
        eprintln!("[upload_dds_texture] texture_res is null after CreateCommittedResource");
        failure()
    })?;

    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); mip_count];
    let mut num_rows = vec![0u32; mip_count];
    let mut row_sizes = vec![0u64; mip_count];
    let mut total_bytes = 0u64;

    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            mip_count as u32,
            0,
            Some(layouts.as_mut_ptr()),
            Some(num_rows.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total_bytes),
        );
    }

    let upload_heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let aligned_upload_size = ((total_bytes + 255) & !255).max(256);
    let upload_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: aligned_upload_size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };

    let mut upload_res: Option<ID3D12Resource> = None;
    unsafe {
        let res = device.CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &upload_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
            &mut upload_res,
        );
        if let Err(e) = res {
            eprintln!(
                "[upload_dds_texture] Failed to allocate upload staging buffer (size {} bytes): {:?}",
                aligned_upload_size, e
            );
            return Err(e);
        }
    }
    let upload_res = upload_res.ok_or_else(|| {
        eprintln!("[upload_dds_texture] upload_res is null after CreateCommittedResource");
        failure()
    })?;

    unsafe {
        let mut mapped_ptr: *mut u8 = std::ptr::null_mut();
        upload_res.Map(
            0,
            Some(&D3D12_RANGE { Begin: 0, End: 0 }),
            Some((&mut mapped_ptr as *mut *mut u8).cast()),
        )?;

        for (i, mip) in dds.mips.iter().enumerate() {
            let layout = &layouts[i];
            let subresource_dst = mapped_ptr.add(layout.Offset as usize);
            let dst_pitch = layout.Footprint.RowPitch as usize;
            let rows = num_rows[i] as usize;
            let src_pitch = mip.row_pitch as usize;
            let copy_row_bytes = (row_sizes[i] as usize).min(src_pitch);

            for row in 0..rows {
                let src_offset = row * src_pitch;
                if src_offset + copy_row_bytes <= mip.data.len() {
                    let src_row = mip.data.as_ptr().add(src_offset);
                    let dst_row = subresource_dst.add(row * dst_pitch);
                    std::ptr::copy_nonoverlapping(src_row, dst_row, copy_row_bytes);
                }
            }
        }

        upload_res.Unmap(
            0,
            Some(&D3D12_RANGE {
                Begin: 0,
                End: total_bytes as usize,
            }),
        );

        // Record CopyTextureRegion for each subresource
        for (i, layout) in layouts.iter().enumerate() {
            let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&texture_res),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let src_loc = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&upload_res),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *layout,
                },
            };
            cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
        }

        // Transition texture from COPY_DEST to PIXEL_SHADER_RESOURCE
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&texture_res),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                    StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                }),
            },
        };
        cmd.ResourceBarrier(&[barrier]);
    }

    Ok((texture_res, upload_res))
}

/// Register a Shader Resource View into the bindless descriptor heap slot
pub fn register_srv(
    device: &ID3D12Device,
    texture: &ID3D12Resource,
    format: DXGI_FORMAT,
    mip_count: u32,
    cpu_handle: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: mip_count,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe {
        device.CreateShaderResourceView(Some(texture), Some(&srv_desc), cpu_handle);
    }
}

/// Manages loaded textures and staging buffer retirement across GPU fence completions
#[derive(Default)]
pub struct TextureRegistry {
    by_name: HashMap<String, GpuTexture>,
    by_index: HashMap<u32, GpuTexture>,
    recording_staging: Vec<ID3D12Resource>,
    submitted_staging: Vec<(u64, Vec<ID3D12Resource>)>,
}

impl TextureRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: String, texture: GpuTexture) {
        self.by_index.insert(texture.descriptor_index.0, texture.clone());
        self.by_name.insert(name, texture);
    }

    pub fn get_by_name(&self, name: &str) -> Option<&GpuTexture> {
        self.by_name.get(name)
    }

    pub fn get_by_index(&self, index: DescriptorIndex) -> Option<&GpuTexture> {
        self.by_index.get(&index.0)
    }

    pub fn track_staging(&mut self, staging: ID3D12Resource) {
        self.recording_staging.push(staging);
    }

    pub fn tag_submission(&mut self, fence_value: u64) {
        if !self.recording_staging.is_empty() {
            let list = std::mem::take(&mut self.recording_staging);
            self.submitted_staging.push((fence_value, list));
        }
    }

    pub fn retire(&mut self, completed_fence: u64) {
        self.submitted_staging.retain(|(fence, _)| *fence > completed_fence);
    }
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

    #[test]
    fn test_bgra8_ui_texture_parsing() {
        // 4x4 BGRA8 (D3DFMT_A8R8G8B8) = 16 pixels * 4 bytes = 64 bytes
        let mut buf = vec![0u8; 128 + 64];
        buf[0..4].copy_from_slice(&DDS_MAGIC.to_le_bytes());
        buf[4..8].copy_from_slice(&124u32.to_le_bytes());
        buf[8..12].copy_from_slice(&(DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT).to_le_bytes());
        buf[12..16].copy_from_slice(&4u32.to_le_bytes());
        buf[16..20].copy_from_slice(&4u32.to_le_bytes());
        // PixelFormat
        buf[76..80].copy_from_slice(&32u32.to_le_bytes());
        buf[80..84].copy_from_slice(&(DDPF_RGB | DDPF_ALPHAPIXELS).to_le_bytes());
        buf[88..92].copy_from_slice(&32u32.to_le_bytes()); // 32 bits
        buf[92..96].copy_from_slice(&0x00FF0000u32.to_le_bytes()); // R mask
        buf[96..100].copy_from_slice(&0x0000FF00u32.to_le_bytes()); // G mask
        buf[100..104].copy_from_slice(&0x000000FFu32.to_le_bytes()); // B mask
        buf[104..108].copy_from_slice(&0xFF000000u32.to_le_bytes()); // A mask

        let img = parse_dds(&buf).expect("Failed to parse valid BGRA8 DDS");
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(img.format, DXGI_FORMAT_B8G8R8A8_UNORM);
        assert!(!img.is_block_compressed);
        assert_eq!(img.mips[0].slice_pitch, 64);
    }

    #[test]
    fn test_registry_tracking_and_retirement() {
        let mut registry = TextureRegistry::new();
        assert!(registry.get_by_name("act_stalker").is_none());
        assert!(registry.get_by_index(DescriptorIndex(0)).is_none());
        registry.retire(100);
    }

    #[test]
    fn test_ati1_bc4_parsing() {
        // 4x4 ATI1 is 1 block * 8 bytes = 8 bytes
        let buf = create_test_dds(4, 4, FOURCC_ATI1, 8);
        let img = parse_dds(&buf).expect("Failed to parse valid ATI1/BC4 DDS");
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert_eq!(img.format, DXGI_FORMAT_BC4_UNORM);
        assert!(img.is_block_compressed);
        assert_eq!(img.block_size, 8);
    }

    #[test]
    fn test_ati2_bc5_parsing() {
        // 8x8 ATI2 (3Dc/BC5 normal map) is 2x2 blocks = 4 blocks * 16 bytes = 64 bytes
        let buf = create_test_dds(8, 8, FOURCC_ATI2, 64);
        let img = parse_dds(&buf).expect("Failed to parse valid ATI2/BC5 DDS");
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 8);
        assert_eq!(img.format, DXGI_FORMAT_BC5_UNORM);
        assert!(img.is_block_compressed);
        assert_eq!(img.block_size, 16);
    }
}
