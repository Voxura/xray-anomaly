//! Phase 10: Infinite Skybox & Atmosphere Hardware Readback Integration Tests
//!
//! Verifies on Direct3D 12 hardware:
//! 1. Procedural unit cube rasterization directly from `SV_VertexID` (0..36) at the far plane (Z=1.0)
//! 2. Dual-texture weather cycle blending (lerp between texture0 and texture1)
//! 3. Early-Z depth occlusion: foreground geometry occludes the skybox

use std::mem::ManuallyDrop;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::w;

use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::{CameraConstants, SkyPushConstants, XRayVertex};

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
fn test_skybox_procedural_cube_hardware_readback() {
    unsafe {
        let mut debug: Option<ID3D12Debug> = None;
        if D3D12GetDebugInterface(&mut debug).is_ok() {
            if let Some(d) = debug {
                d.EnableDebugLayer();
            }
        }
    }
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("SkyboxTest"),
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
            eprintln!("Skipping hardware skybox test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Sky with custom sky_color tint: Blue [0.2, 0.5, 1.0, 1.0]
    let push = SkyPushConstants {
        sky_color: [0.2, 0.5, 1.0, 1.0],
        ..Default::default()
    };
    renderer.draw_sky(&push).expect("draw_sky failed");

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

    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Skybox Procedural Readback center = {:?}", center_px);

    // Default procedural sky has color0=[0.2, 0.3, 0.5] tinted by sky_color=[0.2, 0.5, 1.0]
    // Blue channel must be dominant
    assert!(center_px[2] > center_px[0], "Sky blue channel ({}) must exceed red ({})", center_px[2], center_px[0]);
    assert!(center_px[3] > 0, "Sky alpha must be non-zero");
}

#[test]
fn test_skybox_dual_texture_blending_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("SkyboxBlendTest"),
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
            eprintln!("Skipping hardware skybox blend test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Texture 0: Pure Red (255, 0, 0, 255)
    let red_dds = create_solid_color_rgba8_dds(4, 4, 255, 0, 0, 255);
    let tex0 = renderer.load_texture_from_dds("sky_red.dds", &red_dds).expect("load red failed");

    // Texture 1: Pure Blue (0, 0, 255, 255)
    let blue_dds = create_solid_color_rgba8_dds(4, 4, 0, 0, 255, 255);
    let tex1 = renderer.load_texture_from_dds("sky_blue.dds", &blue_dds).expect("load blue failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Blend 50% between Red and Blue -> Purple [127, 0, 127, 255]
    let push = SkyPushConstants {
        sky_texture0: tex0.0,
        sky_texture1: tex1.0,
        blend_factor: 0.5,
        is_cubemap: 0,
        sky_color: [1.0, 1.0, 1.0, 1.0],
        ..Default::default()
    };
    renderer.draw_sky(&push).expect("draw_sky failed");

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

    let px = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Skybox 50% Dual Blend Readback: {:?}", px);

    // 50% blend of Red (255, 0, 0) and Blue (0, 0, 255) yields Purple [127, 0, 127, 255]
    assert!((px[0] as i32 - 127).abs() <= 5, "Red channel must be ~127, got {}", px[0]);
    assert_eq!(px[1], 0, "Green channel must be 0");
    assert!((px[2] as i32 - 127).abs() <= 5, "Blue channel must be ~127, got {}", px[2]);
}

#[test]
fn test_skybox_occlusion_by_foreground_geometry() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("SkyOcclusionTest"),
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
            eprintln!("Skipping hardware sky occlusion test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Pure Green texture for foreground quad
    let green_dds = create_solid_color_rgba8_dds(4, 4, 0, 255, 0, 255);
    let green_tex = renderer.load_texture_from_dds("green_fore.dds", &green_dds).expect("load green failed");

    // Pure Red texture for skybox
    let red_dds = create_solid_color_rgba8_dds(4, 4, 255, 0, 0, 255);
    let red_tex = renderer.load_texture_from_dds("red_sky.dds", &red_dds).expect("load red failed");

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

    // 2. Draw infinite red skybox at Z=1.0 (occluded by foreground green quad!)
    let push = SkyPushConstants {
        sky_texture0: red_tex.0,
        sky_color: [1.0, 1.0, 1.0, 1.0],
        ..Default::default()
    };
    renderer.draw_sky(&push).expect("draw_sky failed");

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

    // Center pixel (32, 32): Occluded by foreground quad -> MUST be Green [0, 255, 0, 255]
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    // Corner pixel (4, 4): Outside foreground quad -> MUST be Red sky [255, 0, 0, 255]
    let corner_px = readback_rgba8(&readback, 64, 64, 4, 4);

    println!(
        "Sky Occlusion Readback: center = {:?}, corner = {:?}",
        center_px, corner_px
    );

    assert_eq!(center_px[1], 255, "Center pixel must be green foreground quad");
    assert_eq!(center_px[0], 0, "Center pixel red channel must be 0 (sky occluded)");

    assert_eq!(corner_px[0], 255, "Corner pixel must be red sky");
    assert_eq!(corner_px[1], 0, "Corner pixel green channel must be 0");
}
