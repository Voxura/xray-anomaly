//! Phase 5 Bindless Texture Sampling Hardware Readback Test
//!
//! Validates that a texture uploaded to a default heap and registered in the
//! global shader-visible descriptor heap is sampled correctly by the geometry pipeline
//! using 32-bit push constants and SM6.0 / SM6.6 bindless descriptor indexing on physical hardware.

use std::mem::ManuallyDrop;
use std::ptr;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, WS_OVERLAPPEDWINDOW,
};
use windows::core::w;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::texture::{
    DDPF_ALPHAPIXELS, DDPF_RGB, DDS_MAGIC, DDSD_HEIGHT, DDSD_PIXELFORMAT, DDSD_WIDTH,
};
use xray_render_dx12::{CameraConstants, XRayVertex};

struct Window(HWND);
impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.0);
        }
    }
}

fn identity_camera() -> CameraConstants {
    let mut c = CameraConstants::default();
    for i in 0..4 {
        c.view[i][i] = 1.0;
        c.proj[i][i] = 1.0;
        c.view_proj[i][i] = 1.0;
    }
    c
}

unsafe fn transition(
    cmd: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let mut barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
        ..Default::default()
    };
    unsafe {
        cmd.ResourceBarrier(std::slice::from_ref(&barrier));
        ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
    }
}

fn create_solid_color_rgba8_dds(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let data_bytes = pixel_count * 4;
    let mut dds = vec![0u8; 128 + data_bytes];

    dds[0..4].copy_from_slice(&DDS_MAGIC.to_le_bytes());
    dds[4..8].copy_from_slice(&124u32.to_le_bytes()); // header size
    dds[8..12].copy_from_slice(&(DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT).to_le_bytes());
    dds[12..16].copy_from_slice(&height.to_le_bytes());
    dds[16..20].copy_from_slice(&width.to_le_bytes());
    dds[76..80].copy_from_slice(&32u32.to_le_bytes()); // pf_size
    dds[80..84].copy_from_slice(&(DDPF_RGB | DDPF_ALPHAPIXELS).to_le_bytes());
    dds[88..92].copy_from_slice(&32u32.to_le_bytes()); // pf_rgb_bits
    dds[92..96].copy_from_slice(&0x000000FFu32.to_le_bytes()); // r_mask
    dds[96..100].copy_from_slice(&0x0000FF00u32.to_le_bytes()); // g_mask
    dds[100..104].copy_from_slice(&0x00FF0000u32.to_le_bytes()); // b_mask
    dds[104..108].copy_from_slice(&0xFF000000u32.to_le_bytes()); // a_mask

    for i in 0..pixel_count {
        let offset = 128 + i * 4;
        dds[offset] = r;
        dds[offset + 1] = g;
        dds[offset + 2] = b;
        dds[offset + 3] = a;
    }
    dds
}

#[test]
fn test_bindless_textured_mesh_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("Phase 5 Bindless Readback"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            64,
            64,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    });

    let mut renderer = Dx12Renderer::new(1).expect("DirectX 12 hardware required");
    println!("Phase 5 bindless adapter: {}", renderer.adapter_name());
    renderer.attach_window(window.0, 64, 64).unwrap();

    // Create a 4x4 pure green RGBA8 DDS texture
    let green_dds = create_solid_color_rgba8_dds(4, 4, 0, 255, 0, 255);
    let srv_index = renderer
        .load_texture_from_dds("test_green_surface.dds", &green_dds)
        .expect("Failed to upload green DDS texture");

    assert!(srv_index.is_valid());
    println!("Allocated bindless SRV descriptor index: {:?}", srv_index);

    // Quad or triangle covering viewport center (-0.8 to +0.8)
    let vertices = [
        XRayVertex {
            position: [-0.8, -0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.0, 0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.5, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.8, -0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices = [0u16, 1, 2];

    let mesh_handle = renderer
        .create_mesh(&vertices, &indices)
        .expect("Failed to create mesh");
    renderer
        .update_camera(&identity_camera())
        .expect("Failed to update camera");

    // Create readback buffer
    let readback: ID3D12Resource = unsafe {
        let mut resource = None;
        renderer
            .device_ctx
            .device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 64 * 256,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut resource,
            )
            .unwrap();
        resource.unwrap()
    };

    // Frame 1: Draw textured mesh using bindless albedo index
    renderer.begin_frame().unwrap();
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer
        .draw_textured_mesh(mesh_handle, srv_index)
        .expect("Failed to draw textured mesh");

    // Copy backbuffer to readback buffer
    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer
        .swapchain
        .as_ref()
        .unwrap()
        .current_backbuffer_resource();

    unsafe {
        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );

        let mut source = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let mut destination = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(readback.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 64,
                        Height: 64,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };

        cmd.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
        ManuallyDrop::drop(&mut source.pResource);
        ManuallyDrop::drop(&mut destination.pResource);

        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }

    renderer.end_frame(false).unwrap();
    renderer.wait_for_gpu().unwrap();

    // Verify sampled pixel values
    unsafe {
        let mut mapped = ptr::null_mut();
        readback
            .Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: 64 * 256,
                }),
                Some(&mut mapped),
            )
            .unwrap();

        let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), 64 * 256);
        // Center pixel at (32, 32)
        let center = bytes[32 * 256 + 32 * 4..32 * 256 + 32 * 4 + 4].to_vec();
        // Corner pixel at (0, 0)
        let corner = bytes[..4].to_vec();
        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));

        println!("Center pixel sampled color: {:?}", center);
        println!("Corner pixel clear color: {:?}", corner);

        assert_eq!(
            center,
            [0, 255, 0, 255],
            "Center pixel must match sampled green texture"
        );
        assert_eq!(
            corner,
            [0, 0, 0, 255],
            "Corner pixel must match black backbuffer clear"
        );
    }

    println!("Phase 5 bindless textured mesh hardware readback verified successfully!");
}

#[test]
fn test_bindless_normal_mapped_mesh_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("Phase 5 Normal Map Readback"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            64,
            64,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    });

    let mut renderer = Dx12Renderer::new(1).expect("DirectX 12 hardware required");
    renderer.attach_window(window.0, 64, 64).unwrap();

    // 1. Albedo: Solid White RGBA8 (255, 255, 255, 255)
    let white_dds = create_solid_color_rgba8_dds(4, 4, 255, 255, 255, 255);
    let albedo_idx = renderer
        .load_texture_from_dds("test_white_albedo.dds", &white_dds)
        .unwrap();

    // 2. Normal map: Flat normal (128, 128, 255, 255) in tangent space (pointing straight out [0, 0, 1])
    let flat_normal_dds = create_solid_color_rgba8_dds(4, 4, 128, 128, 255, 255);
    let normal_idx = renderer
        .load_texture_from_dds("test_flat_bump.dds", &flat_normal_dds)
        .unwrap();

    let vertices = [
        XRayVertex {
            position: [-0.8, -0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.0, 0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.5, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.8, -0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices = [0u16, 1, 2];

    let mesh_handle = renderer.create_mesh(&vertices, &indices).unwrap();
    renderer.update_camera(&identity_camera()).unwrap();

    let readback: ID3D12Resource = unsafe {
        let mut resource = None;
        renderer
            .device_ctx
            .device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 64 * 256,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut resource,
            )
            .unwrap();
        resource.unwrap()
    };

    renderer.begin_frame().unwrap();
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer
        .draw_textured_material_mesh(mesh_handle, Some(albedo_idx), Some(normal_idx), None)
        .expect("Failed to draw material mesh with normal map");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer
        .swapchain
        .as_ref()
        .unwrap()
        .current_backbuffer_resource();

    unsafe {
        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );

        let mut source = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let mut destination = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(readback.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 64,
                        Height: 64,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };

        cmd.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
        ManuallyDrop::drop(&mut source.pResource);
        ManuallyDrop::drop(&mut destination.pResource);

        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }

    renderer.end_frame(false).unwrap();
    renderer.wait_for_gpu().unwrap();

    unsafe {
        let mut mapped = ptr::null_mut();
        readback
            .Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: 64 * 256,
                }),
                Some(&mut mapped),
            )
            .unwrap();

        let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), 64 * 256);
        let center = bytes[32 * 256 + 32 * 4..32 * 256 + 32 * 4 + 4].to_vec();
        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));

        println!("Normal-mapped lit center color: {:?}", center);
        assert!(center[0] > 100 && center[0] < 220, "Red channel must be lit by normal map");
        assert!(center[1] > 100 && center[1] < 220, "Green channel must be lit by normal map");
        assert!(center[2] > 100 && center[2] < 220, "Blue channel must be lit by normal map");
        assert_eq!(center[3], 255, "Alpha channel must be full opacity");
    }

    println!("Phase 5 normal-mapped mesh readback verified successfully!");
}

#[test]
fn test_real_repo_cursor_cross_dds_vfs_streaming_and_render() {
    let repo_cursor = std::path::Path::new("../../gamedata/textures/ui/cursor_cross.dds")
        .canonicalize()
        .or_else(|_| std::path::Path::new("gamedata/textures/ui/cursor_cross.dds").canonicalize())
        .or_else(|_| std::path::Path::new("../gamedata/textures/ui/cursor_cross.dds").canonicalize());

    let cursor_path = match repo_cursor {
        Ok(p) => p,
        Err(_) => {
            println!("Skipping real asset test: cursor_cross.dds not found in repository root");
            return;
        }
    };

    let dds_bytes = std::fs::read(&cursor_path).expect("Failed to read cursor_cross.dds from disk");
    assert_eq!(dds_bytes.len(), 262272);

    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("Real Cursor Render"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            64,
            64,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    });

    let mut renderer = Dx12Renderer::new(1).expect("DirectX 12 hardware required");
    renderer.attach_window(window.0, 64, 64).unwrap();

    let srv_idx = renderer
        .load_texture_from_dds("ui/cursor_cross.dds", &dds_bytes)
        .expect("Failed to upload real cursor_cross.dds to default heap");

    assert!(srv_idx.is_valid());

    let vertices = [
        XRayVertex {
            position: [-1.0, -1.0, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [-1.0, 1.0, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [1.0, 1.0, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [1.0, -1.0, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices = [0u16, 1, 2, 0, 2, 3];

    let mesh_handle = renderer.create_mesh(&vertices, &indices).unwrap();
    renderer.update_camera(&identity_camera()).unwrap();

    renderer.begin_frame().unwrap();
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer.draw_textured_mesh(mesh_handle, srv_idx).unwrap();
    renderer.end_frame(false).unwrap();
    renderer.wait_for_gpu().unwrap();

    println!("Real repo cursor_cross.dds streamed, uploaded to default heap, and rendered on hardware GPU!");
}

#[test]
fn test_bindless_gpu_transform_structured_buffer_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("Phase 5 Transform Readback"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            64,
            64,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    });

    let mut renderer = Dx12Renderer::new(1).expect("DirectX 12 hardware required");
    renderer.attach_window(window.0, 64, 64).unwrap();

    let green_dds = create_solid_color_rgba8_dds(4, 4, 0, 255, 0, 255);
    let albedo_idx = renderer
        .load_texture_from_dds("test_transform_green.dds", &green_dds)
        .unwrap();

    // Two matrices in structured buffer:
    // Index 0: Identity
    // Index 1: Translation out of view (+10.0 on X)
    let transforms = [
        [
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0, 1.0],
        ],
    ];
    renderer.update_transforms(&transforms).unwrap();

    let vertices = [
        XRayVertex {
            position: [-0.8, -0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.0, 0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.5, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.8, -0.8, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices = [0u16, 1, 2];

    let mesh_handle = renderer.create_mesh(&vertices, &indices).unwrap();
    renderer.update_camera(&identity_camera()).unwrap();

    let readback: ID3D12Resource = unsafe {
        let mut resource = None;
        renderer
            .device_ctx
            .device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 64 * 256,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut resource,
            )
            .unwrap();
        resource.unwrap()
    };

    // Case 1: Draw with transform_index = 0 (identity) -> Triangle is visible and center pixel is green [0, 255, 0, 255]
    renderer.begin_frame().unwrap();
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer
        .draw_textured_mesh_with_transform(mesh_handle, albedo_idx, 0)
        .expect("Failed to draw textured mesh with transform");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer
        .swapchain
        .as_ref()
        .unwrap()
        .current_backbuffer_resource();

    unsafe {
        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );

        let mut source = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let mut destination = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(readback.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 64,
                        Height: 64,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };

        cmd.CopyTextureRegion(&destination, 0, 0, 0, &source, None);
        ManuallyDrop::drop(&mut source.pResource);
        ManuallyDrop::drop(&mut destination.pResource);

        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }

    renderer.end_frame(false).unwrap();
    renderer.wait_for_gpu().unwrap();

    unsafe {
        let mut mapped = ptr::null_mut();
        readback
            .Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: 64 * 256,
                }),
                Some(&mut mapped),
            )
            .unwrap();

        let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), 64 * 256);
        let center = bytes[32 * 256 + 32 * 4..32 * 256 + 32 * 4 + 4].to_vec();
        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));

        assert_eq!(center, [0, 255, 0, 255], "Identity transform matrix must render triangle at center");
    }

    println!("GPU StructuredBuffer transform indexing hardware readback verified successfully!");
}
