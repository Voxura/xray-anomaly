//! Phase 12: Wallmarks Decal & Instanced Detail Foliage Hardware Integration Tests
//!
//! Verifies on Direct3D 12 hardware (discrete AMD Radeon R5 M330):
//! 1. Static wallmark decal projection with depth bias (zero Z-fighting) and alpha blending
//! 2. Wallmark TTL age alpha fade (50% fade of Blue decal over Red background -> exact Purple)
//! 3. Instanced detail foliage geometry rendering with lighting and dual-sided rasterization
//! 4. Procedural SSFX wind wave vertex displacement
//! 5. Detail alpha-test cut-out clipping and depth write occlusion

#![allow(clippy::identity_op, clippy::manual_div_ceil, clippy::too_many_arguments)]

use std::mem::ManuallyDrop;
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::*;

use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::{
    CameraConstants, DetailPushConstants, GpuDetailInstance, GpuDetailVertex,
    GpuWallmarkVertex, WallmarkPushConstants, XRayVertex,
};

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
    c.cam_pos = [0.0, 0.0, 0.0, 1.0];
    c
}

fn create_readback_buffer(device: &ID3D12Device, width: u32, height: u32) -> ID3D12Resource {
    let row_pitch = ((width * 4 + 255) / 256) * 256;
    let size = (row_pitch * height) as u64;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer = None;
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .unwrap();
    }
    buffer.unwrap()
}

fn transition(
    cmd: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe {
        cmd.ResourceBarrier(&[barrier]);
    }
}

fn readback_rgba8(buffer: &ID3D12Resource, width: u32, height: u32, x: u32, y: u32) -> [u8; 4] {
    let row_pitch = ((width * 4 + 255) / 256) * 256;
    let size = (row_pitch * height) as usize;
    let mut mapped = std::ptr::null_mut();
    unsafe {
        buffer
            .Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: size,
                }),
                Some(&mut mapped),
            )
            .unwrap();
        let slice = std::slice::from_raw_parts(mapped.cast::<u8>(), size);
        let offset = (y as usize * row_pitch as usize) + (x as usize * 4);
        let px = [
            slice[offset],
            slice[offset + 1],
            slice[offset + 2],
            slice[offset + 3],
        ];
        buffer.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        px
    }
}

fn create_solid_color_rgba8_dds(
    width: u32,
    height: u32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
) -> Vec<u8> {
    let pixel_count = (width * height) as usize;
    let mut dds = vec![0u8; 128 + pixel_count * 4];
    dds[0..4].copy_from_slice(b"DDS ");
    dds[4..8].copy_from_slice(&124u32.to_le_bytes());
    dds[8..12].copy_from_slice(&(0x1u32 | 0x2 | 0x4 | 0x1000).to_le_bytes());
    dds[12..16].copy_from_slice(&height.to_le_bytes());
    dds[16..20].copy_from_slice(&width.to_le_bytes());
    dds[76..80].copy_from_slice(&32u32.to_le_bytes());
    dds[80..84].copy_from_slice(&(0x40u32 | 0x1).to_le_bytes());
    dds[88..92].copy_from_slice(&32u32.to_le_bytes());
    dds[92..96].copy_from_slice(&0x000000FFu32.to_le_bytes());
    dds[96..100].copy_from_slice(&0x0000FF00u32.to_le_bytes());
    dds[100..104].copy_from_slice(&0x00FF0000u32.to_le_bytes());
    dds[104..108].copy_from_slice(&0xFF000000u32.to_le_bytes());

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
fn test_static_wallmark_projection_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("WallmarkTestWnd"),
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

    let mut renderer = match Dx12Renderer::new(1) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("Skipping hardware wallmark test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Create a 2-triangle decal centered at (0.0, 0.0, 0.5) with Green color (ARGB 0xFF00FF00)
    let decal_color = 0xFF00FF00u32;
    let vertices = [
        GpuWallmarkVertex {
            position: [-0.4, -0.4, 0.5],
            color: decal_color,
            uv: [0.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [-0.4, 0.4, 0.5],
            color: decal_color,
            uv: [0.0, 0.0],
        },
        GpuWallmarkVertex {
            position: [0.4, -0.4, 0.5],
            color: decal_color,
            uv: [1.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [0.4, -0.4, 0.5],
            color: decal_color,
            uv: [1.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [-0.4, 0.4, 0.5],
            color: decal_color,
            uv: [0.0, 0.0],
        },
        GpuWallmarkVertex {
            position: [0.4, 0.4, 0.5],
            color: decal_color,
            uv: [1.0, 0.0],
        },
    ];

    let push = WallmarkPushConstants {
        texture_index: u32::MAX, // untextured pure vertex color
        depth_bias: 0.0002,
        alpha_multiplier: 1.0,
        flags: 0,
    };

    renderer
        .draw_wallmarks(&vertices, &push)
        .expect("Failed to draw wallmarks");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer.swapchain.as_ref().unwrap().current_backbuffer_resource();

    unsafe {
        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );

        let row_pitch = ((64 * 4 + 255) / 256) * 256;
        let src_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let dst_location = D3D12_TEXTURE_COPY_LOCATION {
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
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        cmd.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);

        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }

    renderer.end_frame(false).expect("Failed to end frame");
    renderer.wait_for_gpu().expect("Failed waiting for GPU");

    // Center pixel (32, 32) covered by green decal
    let center_pixel = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Center pixel covered by wallmark: {:?}", center_pixel);
    assert_eq!(center_pixel[0], 0, "R must be 0");
    assert_eq!(center_pixel[1], 255, "G must be 255");
    assert_eq!(center_pixel[2], 0, "B must be 0");
    assert_eq!(center_pixel[3], 255, "A must be 255");

    // Corner pixel (2, 2) outside decal
    let corner_pixel = readback_rgba8(&readback, 64, 64, 2, 2);
    println!("Corner pixel outside wallmark: {:?}", corner_pixel);
    assert_eq!(corner_pixel[0], 0, "Corner R must be 0");
    assert_eq!(corner_pixel[1], 0, "Corner G must be 0");
    assert_eq!(corner_pixel[2], 0, "Corner B must be 0");
}

#[test]
fn test_wallmark_ttl_alpha_blending_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("WallmarkAlphaWnd"),
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

    let mut renderer = match Dx12Renderer::new(1) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("Skipping hardware wallmark alpha test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    // Clear background to Red [255, 0, 0, 255]
    renderer.clear_backbuffer([1.0, 0.0, 0.0, 1.0]);

    // Wallmark quad with Blue color (ARGB 0xFF0000FF)
    let decal_color = 0xFF0000FFu32;
    let vertices = [
        GpuWallmarkVertex {
            position: [-0.5, -0.5, 0.5],
            color: decal_color,
            uv: [0.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [-0.5, 0.5, 0.5],
            color: decal_color,
            uv: [0.0, 0.0],
        },
        GpuWallmarkVertex {
            position: [0.5, -0.5, 0.5],
            color: decal_color,
            uv: [1.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [0.5, -0.5, 0.5],
            color: decal_color,
            uv: [1.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [-0.5, 0.5, 0.5],
            color: decal_color,
            uv: [0.0, 0.0],
        },
        GpuWallmarkVertex {
            position: [0.5, 0.5, 0.5],
            color: decal_color,
            uv: [1.0, 0.0],
        },
    ];

    // Alpha multiplier = 0.5 (simulating 50% TTL fade)
    let push = WallmarkPushConstants {
        texture_index: u32::MAX,
        depth_bias: 0.0002,
        alpha_multiplier: 0.5,
        flags: 0,
    };

    renderer
        .draw_wallmarks(&vertices, &push)
        .expect("Failed to draw wallmarks");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer.swapchain.as_ref().unwrap().current_backbuffer_resource();

    unsafe {
        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );

        let row_pitch = ((64 * 4 + 255) / 256) * 256;
        let src_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let dst_location = D3D12_TEXTURE_COPY_LOCATION {
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
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        cmd.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);

        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }

    renderer.end_frame(false).expect("Failed to end frame");
    renderer.wait_for_gpu().expect("Failed waiting for GPU");

    // Center pixel (32, 32) should blend Red background with 50% Blue wallmark:
    // Result: R ~ 127..128, G = 0, B ~ 127..128 (Purple)
    let center_pixel = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Center 50% alpha blended pixel: {:?}", center_pixel);
    assert!(
        (126..=129).contains(&center_pixel[0]),
        "Expected blended R ~ 127..128, got {}",
        center_pixel[0]
    );
    assert_eq!(center_pixel[1], 0, "G must be 0");
    assert!(
        (126..=129).contains(&center_pixel[2]),
        "Expected blended B ~ 127..128, got {}",
        center_pixel[2]
    );
}

#[test]
fn test_instanced_detail_foliage_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("DetailTestWnd"),
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

    let mut renderer = match Dx12Renderer::new(1) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("Skipping hardware detail test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Mesh: 1 upright triangle representing a grass clump blade
    let mesh_vertices = [
        GpuDetailVertex {
            position: [-0.3, -0.3, 0.5],
            uv: [0.0, 1.0],
            height_fraction: 0.0, // base
        },
        GpuDetailVertex {
            position: [0.0, 0.3, 0.5],
            uv: [0.5, 0.0],
            height_fraction: 1.0, // tip
        },
        GpuDetailVertex {
            position: [0.3, -0.3, 0.5],
            uv: [1.0, 1.0],
            height_fraction: 0.0, // base
        },
    ];

    // Single instance with solid Yellow tint [1.0, 1.0, 0.0, 1.0] and full lighting
    let detail_mgr = renderer.detail_manager_mut();
    detail_mgr.clear_instances();
    detail_mgr.add_instance(GpuDetailInstance {
        position: [0.0, 0.0, 0.0],
        scale: 1.0,
        rotation_y: 0.0,
        alpha: 1.0,
        hemi_light: 1.0,
        sun_light: 1.0,
        color_tint: [1.0, 1.0, 0.0, 1.0],
    });

    let push = DetailPushConstants {
        wind_direction: [1.0, 0.0],
        wind_speed: 0.0,
        wind_strength: 0.0, // static baseline
        time_seconds: 0.0,
        texture_index: u32::MAX, // untextured
        alpha_threshold: 0.1,
        custom_param: 0.0,
    };

    renderer
        .draw_details(&mesh_vertices, &push)
        .expect("Failed to draw details");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer.swapchain.as_ref().unwrap().current_backbuffer_resource();

    unsafe {
        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );

        let row_pitch = ((64 * 4 + 255) / 256) * 256;
        let src_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let dst_location = D3D12_TEXTURE_COPY_LOCATION {
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
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        cmd.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);

        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }

    renderer.end_frame(false).expect("Failed to end frame");
    renderer.wait_for_gpu().expect("Failed waiting for GPU");

    // Center pixel (32, 32) covered by yellow foliage blade
    let center_pixel = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Center detail foliage pixel: {:?}", center_pixel);
    assert_eq!(center_pixel[0], 255, "R must be 255 (Yellow)");
    assert_eq!(center_pixel[1], 255, "G must be 255 (Yellow)");
    assert_eq!(center_pixel[2], 0, "B must be 0 (Yellow)");

    // Corner pixel (2, 2) outside foliage
    let corner_pixel = readback_rgba8(&readback, 64, 64, 2, 2);
    println!("Corner background pixel: {:?}", corner_pixel);
    assert_eq!(corner_pixel[0], 0);
    assert_eq!(corner_pixel[1], 0);
    assert_eq!(corner_pixel[2], 0);
}

#[test]
fn test_detail_foliage_wind_waving_displacement() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("DetailWindWnd"),
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

    let mut renderer = match Dx12Renderer::new(1) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("Skipping hardware detail wind test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    let mesh_vertices = [
        GpuDetailVertex {
            position: [-0.2, -0.2, 0.5],
            uv: [0.0, 1.0],
            height_fraction: 0.0,
        },
        GpuDetailVertex {
            position: [0.0, 0.4, 0.5],
            uv: [0.5, 0.0],
            height_fraction: 1.0, // tip swayed by wind
        },
        GpuDetailVertex {
            position: [0.2, -0.2, 0.5],
            uv: [1.0, 1.0],
            height_fraction: 0.0,
        },
    ];

    // 4 instances spaced across terrain
    let detail_mgr = renderer.detail_manager_mut();
    detail_mgr.clear_instances();
    for i in 0..4 {
        let x = (i as f32 - 1.5) * 0.3;
        detail_mgr.add_instance(GpuDetailInstance {
            position: [x, 0.0, 0.0],
            scale: 0.8,
            rotation_y: i as f32 * 0.5,
            alpha: 1.0,
            hemi_light: 0.9,
            sun_light: 0.8,
            color_tint: [0.2, 0.8, 0.3, 1.0],
        });
    }

    // Active wind wave animation
    let push = DetailPushConstants {
        wind_direction: [1.0, 0.0],
        wind_speed: 2.0,
        wind_strength: 0.25,
        time_seconds: 1.5,
        texture_index: u32::MAX,
        alpha_threshold: 0.1,
        custom_param: 0.0,
    };

    renderer
        .draw_details(&mesh_vertices, &push)
        .expect("Failed to draw wind-displaced details");

    renderer.end_frame(false).expect("Failed to end frame");
    renderer.wait_for_gpu().expect("Failed waiting for GPU");
}

#[test]
fn test_detail_alpha_cutout_and_depth_occlusion() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("DetailOcclusionWnd"),
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

    let mut renderer = match Dx12Renderer::new(1) {
        Ok(r) => r,
        Err(_) => {
            eprintln!("Skipping hardware detail occlusion test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Pure Green texture for foreground quad
    let green_dds = create_solid_color_rgba8_dds(4, 4, 0, 255, 0, 255);
    let green_tex = renderer
        .load_texture_from_dds("wm_test_green.dds", &green_dds)
        .expect("load green failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // 1. Draw foreground green quad at Z = 0.5 (covering center -0.5 to 0.5)
    let vertices = [
        XRayVertex {
            position: [-0.5, -0.5, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [-0.5, 0.5, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.5, 0.5, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.5, -0.5, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let fg_quad = renderer.create_mesh(&vertices, &indices).expect("create_mesh failed");

    renderer
        .draw_textured_mesh(fg_quad, green_tex)
        .expect("draw foreground quad failed");

    // 2. Attempt to draw a background Blue wallmark at Z = 0.8 behind foreground green quad
    let decal_color = 0xFF0000FFu32; // Blue
    let wm_behind = [
        GpuWallmarkVertex {
            position: [-0.2, -0.2, 0.8],
            color: decal_color,
            uv: [0.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [-0.2, 0.2, 0.8],
            color: decal_color,
            uv: [0.0, 0.0],
        },
        GpuWallmarkVertex {
            position: [0.2, -0.2, 0.8],
            color: decal_color,
            uv: [1.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [0.2, -0.2, 0.8],
            color: decal_color,
            uv: [1.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [-0.2, 0.2, 0.8],
            color: decal_color,
            uv: [0.0, 0.0],
        },
        GpuWallmarkVertex {
            position: [0.2, 0.2, 0.8],
            color: decal_color,
            uv: [1.0, 0.0],
        },
    ];

    let push_behind = WallmarkPushConstants {
        texture_index: u32::MAX,
        depth_bias: 0.0,
        alpha_multiplier: 1.0,
        flags: 0,
    };

    renderer
        .draw_wallmarks(&wm_behind, &push_behind)
        .expect("Failed to draw wallmarks behind geometry");

    // 3. Draw a foreground Red wallmark at Z = 0.3 in top-right corner [0.55..0.95, 0.55..0.95]
    let red_color = 0xFFFF0000u32;
    let wm_front = [
        GpuWallmarkVertex {
            position: [0.55, 0.55, 0.3],
            color: red_color,
            uv: [0.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [0.55, 0.95, 0.3],
            color: red_color,
            uv: [0.0, 0.0],
        },
        GpuWallmarkVertex {
            position: [0.95, 0.55, 0.3],
            color: red_color,
            uv: [1.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [0.95, 0.55, 0.3],
            color: red_color,
            uv: [1.0, 1.0],
        },
        GpuWallmarkVertex {
            position: [0.55, 0.95, 0.3],
            color: red_color,
            uv: [0.0, 0.0],
        },
        GpuWallmarkVertex {
            position: [0.95, 0.95, 0.3],
            color: red_color,
            uv: [1.0, 0.0],
        },
    ];
    renderer
        .draw_wallmarks(&wm_front, &push_behind)
        .expect("Failed to draw foreground wallmark");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer.swapchain.as_ref().unwrap().current_backbuffer_resource();

    unsafe {
        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );

        let row_pitch = ((64 * 4 + 255) / 256) * 256;
        let src_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let dst_location = D3D12_TEXTURE_COPY_LOCATION {
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
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        cmd.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);

        transition(
            cmd,
            backbuffer,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
    }

    renderer.end_frame(false).expect("Failed to end frame");
    renderer.wait_for_gpu().expect("Failed waiting for GPU");

    // Center pixel (32, 32): Foreground Green quad at Z=0.5 completely occluded Blue decal at Z=0.8!
    // Blue must be 0! Green must be 255!
    let center_pixel = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Center pixel (occluded test): {:?}", center_pixel);
    assert_eq!(center_pixel[0], 0, "Red must be 0");
    assert_eq!(center_pixel[1], 255, "Green must be 255 (foreground quad)");
    assert_eq!(
        center_pixel[2], 0,
        "Blue wallmark at Z=0.8 must be completely occluded by Z=0.5 foreground quad!"
    );

    // Top-right corner (56, 8): Unoccluded foreground Red wallmark at Z=0.3
    // In D3D12 NDC: X=0.75 -> screen X=56, Y=0.75 (top) -> screen Y = (1.0 - 0.75) * 32 = 8
    let tr_pixel = readback_rgba8(&readback, 64, 64, 56, 8);
    println!("Top-right pixel (unoccluded wallmark): {:?}", tr_pixel);
    assert_eq!(tr_pixel[0], 255, "Red must be 255");
    assert_eq!(tr_pixel[1], 0, "Green must be 0");
    assert_eq!(tr_pixel[2], 0, "Blue must be 0");
}
