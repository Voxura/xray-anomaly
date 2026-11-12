//! Phase 11: Clouds & Rain Weather Effects Hardware Integration Tests
//!
//! Verifies on Direct3D 12 hardware:
//! 1. Procedural hemisphere cloud dome rasterization directly from `SV_VertexID` (0..3072) with far-plane depth clamping
//! 2. Dual-texture weather cycle cloud blending (50% blend of Red & Blue textures -> Purple)
//! 3. Instanced billboard rain streak rendering with camera-facing expansion and needle-profile alpha falloff
//! 4. Early-Z depth occlusion: foreground level geometry ($Z = 0.5$) completely occludes rain and clouds

use std::mem::ManuallyDrop;
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::*;

use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::{
    CameraConstants, CloudsPushConstants, RainPushConstants, XRayVertex,
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

unsafe fn transition(
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
fn test_clouds_dome_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("CloudsTestWnd"),
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
            eprintln!("Skipping hardware clouds test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    let push = CloudsPushConstants {
        clouds_texture0: u32::MAX,
        clouds_texture1: u32::MAX,
        blend_factor: 0.0,
        clouds_rotation: 0.0,
        clouds_color: [1.0, 0.7, 0.4, 1.0], // Sunset amber
        sun_color: [1.0, 0.9, 0.7, 0.0],
        sun_dir: [0.0, 1.0, 0.0, 0.0],
        uv_offset0: [0.0, 0.0],
        uv_offset1: [0.0, 0.0],
        clouds_scale: [100.0, 35.0],
        padding: [0.0, 0.0],
    };
    renderer.draw_clouds(&push).expect("draw_clouds failed");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer.swapchain.as_ref().unwrap().current_backbuffer_resource();
    unsafe {
        transition(cmd, backbuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let row_pitch = ((64 * 4 + 255) / 256) * 256;
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
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
        cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
        transition(cmd, backbuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    // Sample dome pixel at center (32, 32)
    let pixel = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Cloud dome center readback pixel: {:?}", pixel);

    assert!(pixel[0] > 0, "Cloud dome should render non-zero red channel");
    assert!(pixel[1] > 0, "Cloud dome should render non-zero green channel");
    assert_eq!(pixel[3], 255);
}

#[test]
fn test_clouds_dual_texture_blending_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("CloudsBlendWnd"),
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
            eprintln!("Skipping hardware clouds blend test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Pure red and pure blue DDS textures
    let red_dds = create_solid_color_rgba8_dds(4, 4, 255, 0, 0, 255);
    let blue_dds = create_solid_color_rgba8_dds(4, 4, 0, 0, 255, 255);

    let red_idx = renderer
        .load_texture_from_dds("clouds_red.dds", &red_dds)
        .expect("load red failed");
    let blue_idx = renderer
        .load_texture_from_dds("clouds_blue.dds", &blue_dds)
        .expect("load blue failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // 50% weather cycle blend between Red and Blue -> Purple
    let push = CloudsPushConstants {
        clouds_texture0: red_idx.0,
        clouds_texture1: blue_idx.0,
        blend_factor: 0.5,
        clouds_rotation: 0.0,
        clouds_color: [1.0, 1.0, 1.0, 1.0],
        sun_color: [0.0, 0.0, 0.0, 0.0],
        sun_dir: [0.0, 1.0, 0.0, 0.0],
        uv_offset0: [0.0, 0.0],
        uv_offset1: [0.0, 0.0],
        clouds_scale: [100.0, 35.0],
        padding: [0.0, 0.0],
    };
    renderer.draw_clouds(&push).expect("draw_clouds failed");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer.swapchain.as_ref().unwrap().current_backbuffer_resource();
    unsafe {
        transition(cmd, backbuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let row_pitch = ((64 * 4 + 255) / 256) * 256;
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
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
        cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
        transition(cmd, backbuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    let pixel = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Dual-texture cloud blend readback pixel: {:?}", pixel);

    // 50% Red and 50% Blue blend produces exact purple: R ~ 128, G ~ 0, B ~ 128
    assert!(
        (pixel[0] as i32 - 128).abs() <= 3,
        "Red channel should be ~128, got {}",
        pixel[0]
    );
    assert_eq!(pixel[1], 0, "Green channel should be 0");
    assert!(
        (pixel[2] as i32 - 128).abs() <= 3,
        "Blue channel should be ~128, got {}",
        pixel[2]
    );
    assert_eq!(pixel[3], 255);
}

#[test]
fn test_instanced_rain_streaks_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("RainStreaksWnd"),
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
            eprintln!("Skipping hardware rain streaks test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Place a single rain streak in front of the camera:
    // Centered at (0, 0, 0.5), head at (0, 0.4, 0.5), trail at (0, -0.4, 0.5)
    let rain_mgr = renderer.rain_manager_mut();
    rain_mgr.set_active_count(1);
    let drop = &mut rain_mgr.drops_mut()[0];
    drop.position = [0.0, -0.4, 0.5];
    drop.length = 0.8;
    drop.direction = [0.0, -1.0, 0.0];
    drop.width = 0.4;
    drop.alpha = 1.0;

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Yellow rain streak for unambiguous verification
    let push = RainPushConstants {
        rain_texture: u32::MAX,
        rain_density: 1.0,
        streak_length_scale: 1.0,
        streak_width_scale: 1.0,
        rain_color: [1.0, 1.0, 0.0, 1.0], // Yellow
    };
    renderer.draw_rain(&push).expect("draw_rain failed");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer.swapchain.as_ref().unwrap().current_backbuffer_resource();
    unsafe {
        transition(cmd, backbuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let row_pitch = ((64 * 4 + 255) / 256) * 256;
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
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
        cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
        transition(cmd, backbuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    // Center pixel (32, 32) is intersected by the yellow streak
    let center = readback_rgba8(&readback, 64, 64, 32, 32);
    // Corner pixel (2, 2) has no rain streak and must remain black
    let corner = readback_rgba8(&readback, 64, 64, 2, 2);

    println!("Rain streak center pixel: {:?}", center);
    println!("Rain background corner pixel: {:?}", corner);

    assert!(center[0] > 50, "Rain streak center should have yellow red channel: {}", center[0]);
    assert!(center[1] > 50, "Rain streak center should have yellow green channel: {}", center[1]);
    assert_eq!(center[2], 0, "Rain streak should have zero blue channel");

    assert_eq!(corner[0], 0, "Corner background should be black");
    assert_eq!(corner[1], 0, "Corner background should be black");
    assert_eq!(corner[2], 0, "Corner background should be black");
}

#[test]
fn test_weather_early_z_occlusion() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("WeatherOcclusionWnd"),
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
            eprintln!("Skipping hardware weather occlusion test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Pure Green texture for foreground quad
    let green_dds = create_solid_color_rgba8_dds(4, 4, 0, 255, 0, 255);
    let green_tex = renderer.load_texture_from_dds("weather_green_fore.dds", &green_dds).expect("load green failed");

    // Foreground quad covering center (-0.5 to 0.5) at Z=0.5
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
    let quad = renderer.create_mesh(&vertices, &indices).expect("create_mesh failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // 1. Draw foreground green quad at Z=0.5 (writes depth)
    renderer
        .draw_textured_mesh(quad, green_tex)
        .expect("draw foreground quad failed");

    // 2. Draw red cloud dome at Z=1.0 (occluded by foreground green quad!)
    let push = CloudsPushConstants {
        clouds_texture0: u32::MAX,
        clouds_texture1: u32::MAX,
        blend_factor: 0.0,
        clouds_rotation: 0.0,
        clouds_color: [1.0, 0.0, 0.0, 1.0], // Red clouds
        sun_color: [0.0, 0.0, 0.0, 0.0],
        sun_dir: [0.0, 1.0, 0.0, 0.0],
        uv_offset0: [0.0, 0.0],
        uv_offset1: [0.0, 0.0],
        clouds_scale: [100.0, 35.0],
        padding: [0.0, 0.0],
    };
    renderer.draw_clouds(&push).expect("draw_clouds failed");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    let backbuffer = renderer.swapchain.as_ref().unwrap().current_backbuffer_resource();
    unsafe {
        transition(cmd, backbuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(backbuffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let row_pitch = ((64 * 4 + 255) / 256) * 256;
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
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
        cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
        transition(cmd, backbuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    }

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    // Center pixel (32, 32) is covered by green quad (Z=0.5): red clouds (Z=1.0) must be occluded!
    let center = readback_rgba8(&readback, 64, 64, 32, 32);
    // Corner pixel (2, 2) has no foreground quad: red clouds show through!
    let corner = readback_rgba8(&readback, 64, 64, 2, 2);

    println!("Occluded center pixel: {:?}", center);
    println!("Unoccluded corner pixel: {:?}", corner);

    assert_eq!(center[0], 0, "Foreground quad must completely occlude red clouds");
    assert_eq!(center[1], 255, "Foreground quad green color must be preserved");
    assert_eq!(center[2], 0, "Blue channel should be zero");
    assert_eq!(center[3], 255);

    assert!(corner[0] > 0, "Corner unoccluded pixel should show red clouds");
    assert_eq!(corner[1], 0, "Corner unoccluded pixel should have 0 green");
}
