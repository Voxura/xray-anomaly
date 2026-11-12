//! Phase 13: Post-Processing, HDR10 Tonemapping, Bloom & Color Grading Integration Tests
//!
//! Verifies on Direct3D 12 hardware (discrete AMD Radeon R5 M330):
//! 1. Procedural fullscreen triangle rasterization covering entire NDC [-1, 1] without vertex/index buffers
//! 2. Uncharted 2 / Filmic tonemapping curve matching mathematical model
//! 3. ACES Filmic tonemapping hardware execution
//! 4. Radial screen vignette corner darkening according to X-Ray combine_2_naa.ps
//! 5. Additive bloom composition with scene backbuffer

#![allow(clippy::identity_op, clippy::manual_div_ceil, clippy::too_many_arguments)]

use std::mem::ManuallyDrop;
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::*;

use xray_formats::postprocess::{uncharted2_tonemap_curve, TonemapParams};
use xray_render_dx12::postprocess::PostprocessPushConstants;
use xray_render_dx12::renderer::Dx12Renderer;

struct Window(HWND);
impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.0);
        }
    }
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
fn test_postprocess_fullscreen_triangle_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("PostprocessTestWnd1"),
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
            eprintln!("Skipping hardware postprocess test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let dds = create_solid_color_rgba8_dds(64, 64, 180, 90, 45, 255);
    let scene_tex_idx = renderer
        .load_texture_from_dds("scene_tex1.dds", &dds)
        .expect("load_texture_from_dds failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Setup pass-through postprocess configuration (mode 3 = pass-through/clamped)
    {
        let mgr = renderer.postprocess_manager_mut();
        mgr.image_correction.exposure = 1.0;
        mgr.image_correction.gamma = 1.0;
        mgr.image_correction.saturation = 1.0;
        mgr.image_correction.vignette_control = [0.0, 0.0, 0.0, 0.0];
    }

    let push = PostprocessPushConstants {
        screen_res_inv: [1.0 / 64.0, 1.0 / 64.0],
        time_seconds: 0.0,
        bloom_factor: 0.0,
        scene_texture_index: scene_tex_idx.0,
        bloom_texture_index: u32::MAX,
        tonemap_mode: 3, // Pass-through / clamped
        color_grading_enabled: 0,
    };

    renderer.draw_postprocess(&push).expect("draw_postprocess failed");

    // Copy to readback buffer
    let backbuffer = renderer
        .swapchain
        .as_ref()
        .unwrap()
        .current_backbuffer_resource();
    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );

    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(backbuffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let row_pitch = ((64 * 4 + 255) / 256) * 256;
    let dst = D3D12_TEXTURE_COPY_LOCATION {
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
    unsafe {
        cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    }
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    // Verify center and all 4 corners match exact texture color [180, 90, 45, 255]
    let center = readback_rgba8(&readback, 64, 64, 32, 32);
    let top_left = readback_rgba8(&readback, 64, 64, 0, 0);
    let top_right = readback_rgba8(&readback, 64, 64, 63, 0);
    let bottom_left = readback_rgba8(&readback, 64, 64, 0, 63);
    let bottom_right = readback_rgba8(&readback, 64, 64, 63, 63);

    assert_eq!(center, [180, 90, 45, 255], "Center color mismatch");
    assert_eq!(top_left, [180, 90, 45, 255], "Top-left corner mismatch");
    assert_eq!(top_right, [180, 90, 45, 255], "Top-right corner mismatch");
    assert_eq!(bottom_left, [180, 90, 45, 255], "Bottom-left corner mismatch");
    assert_eq!(bottom_right, [180, 90, 45, 255], "Bottom-right corner mismatch");
}

#[test]
fn test_postprocess_uncharted2_tonemap_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("PostprocessTestWnd2"),
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
            eprintln!("Skipping hardware tonemap test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    // Texture color: 200/255 = ~0.784
    let dds = create_solid_color_rgba8_dds(64, 64, 200, 200, 200, 255);
    let scene_tex_idx = renderer
        .load_texture_from_dds("scene_tex2.dds", &dds)
        .expect("load_texture_from_dds failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Setup Uncharted 2 tonemap parameters
    let tm_params = TonemapParams {
        a: 0.22,
        b: 0.30,
        c: 0.10,
        d: 0.20,
        e: 0.01,
        f: 0.30,
        w: 11.2,
        exposure: 2.0,
        gamma: 1.0,
        enabled: true,
    };

    {
        let mgr = renderer.postprocess_manager_mut();
        mgr.tonemap = tm_params;
        mgr.image_correction.exposure = 1.0;
        mgr.image_correction.gamma = 1.0;
        mgr.image_correction.saturation = 1.0;
        mgr.image_correction.vignette_control = [0.0, 0.0, 0.0, 0.0];
    }

    let push = PostprocessPushConstants {
        screen_res_inv: [1.0 / 64.0, 1.0 / 64.0],
        time_seconds: 0.0,
        bloom_factor: 0.0,
        scene_texture_index: scene_tex_idx.0,
        bloom_texture_index: u32::MAX,
        tonemap_mode: 0, // Uncharted 2 / Filmic
        color_grading_enabled: 0,
    };

    renderer.draw_postprocess(&push).expect("draw_postprocess failed");

    // Copy to readback buffer
    let backbuffer = renderer
        .swapchain
        .as_ref()
        .unwrap()
        .current_backbuffer_resource();
    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );

    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(backbuffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let row_pitch = ((64 * 4 + 255) / 256) * 256;
    let dst = D3D12_TEXTURE_COPY_LOCATION {
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
    unsafe {
        cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    }
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    let px = readback_rgba8(&readback, 64, 64, 32, 32);

    // Theoretical expected value:
    // in = 200/255 = 0.7843137
    // in * tm_params.exposure = 1.5686274
    // uncharted2_tonemap_curve(1.5686274) / uncharted2_tonemap_curve(11.2)
    let in_hdr = 200.0 / 255.0;
    let curr = uncharted2_tonemap_curve(in_hdr * tm_params.exposure, &tm_params);
    let white = uncharted2_tonemap_curve(tm_params.w, &tm_params);
    let expected_ldr = (curr / white).clamp(0.0, 1.0);
    let expected_u8 = (expected_ldr * 255.0).round() as u8;

    let diff = (px[0] as i32 - expected_u8 as i32).abs();
    assert!(
        diff <= 2,
        "GPU tonemapped pixel {} did not match expected {} within tolerance (diff={})",
        px[0],
        expected_u8,
        diff
    );
}

#[test]
fn test_postprocess_radial_vignette_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("PostprocessTestWnd3"),
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
            eprintln!("Skipping hardware vignette test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    // Pure white scene texture
    let dds = create_solid_color_rgba8_dds(64, 64, 255, 255, 255, 255);
    let scene_tex_idx = renderer
        .load_texture_from_dds("scene_tex3.dds", &dds)
        .expect("load_texture_from_dds failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Vignette: corner darkening 0.8 (80%), center darkening 0.0
    {
        let mgr = renderer.postprocess_manager_mut();
        mgr.image_correction.exposure = 1.0;
        mgr.image_correction.gamma = 1.0;
        mgr.image_correction.saturation = 1.0;
        mgr.image_correction.vignette_control = [0.8, 0.0, 0.0, 0.0];
    }

    let push = PostprocessPushConstants {
        screen_res_inv: [1.0 / 64.0, 1.0 / 64.0],
        time_seconds: 0.0,
        bloom_factor: 0.0,
        scene_texture_index: scene_tex_idx.0,
        bloom_texture_index: u32::MAX,
        tonemap_mode: 3, // Pass-through
        color_grading_enabled: 0,
    };

    renderer.draw_postprocess(&push).expect("draw_postprocess failed");

    // Copy to readback buffer
    let backbuffer = renderer
        .swapchain
        .as_ref()
        .unwrap()
        .current_backbuffer_resource();
    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );

    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(backbuffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let row_pitch = ((64 * 4 + 255) / 256) * 256;
    let dst = D3D12_TEXTURE_COPY_LOCATION {
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
    unsafe {
        cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    }
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    let center = readback_rgba8(&readback, 64, 64, 32, 32);
    let corner = readback_rgba8(&readback, 64, 64, 0, 0);

    // Center pixel (32, 32) is half-pixel offset from (32.0, 32.0), so factor = 1.0 - 0.8 * (1/64) = 0.9875 -> ~252
    assert!(
        center[0] >= 250,
        "Center should remain close to 255 (got {})",
        center[0]
    );

    // Corner (0, 0) is at t ~ 0.984 -> factor ~ 0.2125 -> ~54
    assert!(
        corner[0] <= 60,
        "Corner pixel must be heavily darkened (got {})",
        corner[0]
    );
    assert!(
        center[0] > corner[0] + 180,
        "Center ({}) must be far brighter than corner ({})",
        center[0],
        corner[0]
    );
}

#[test]
fn test_postprocess_bloom_additive_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("PostprocessTestWnd4"),
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
            eprintln!("Skipping hardware bloom test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    // Scene texture: solid 60
    let scene_dds = create_solid_color_rgba8_dds(64, 64, 60, 60, 60, 255);
    let scene_tex_idx = renderer
        .load_texture_from_dds("scene_tex4.dds", &scene_dds)
        .expect("load_texture_from_dds failed");

    // Bloom texture: solid 100
    let bloom_dds = create_solid_color_rgba8_dds(64, 64, 100, 100, 100, 255);
    let bloom_tex_idx = renderer
        .load_texture_from_dds("bloom_tex4.dds", &bloom_dds)
        .expect("load_texture_from_dds failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // bloom_factor = 0.5: expected = 60 + 100 * 0.5 = 110
    {
        let mgr = renderer.postprocess_manager_mut();
        mgr.image_correction.exposure = 1.0;
        mgr.image_correction.gamma = 1.0;
        mgr.image_correction.saturation = 1.0;
        mgr.image_correction.vignette_control = [0.0, 0.0, 0.0, 0.0];
    }

    let push = PostprocessPushConstants {
        screen_res_inv: [1.0 / 64.0, 1.0 / 64.0],
        time_seconds: 0.0,
        bloom_factor: 0.5,
        scene_texture_index: scene_tex_idx.0,
        bloom_texture_index: bloom_tex_idx.0,
        tonemap_mode: 3, // Pass-through
        color_grading_enabled: 0,
    };

    renderer.draw_postprocess(&push).expect("draw_postprocess failed");

    // Copy to readback buffer
    let backbuffer = renderer
        .swapchain
        .as_ref()
        .unwrap()
        .current_backbuffer_resource();
    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );

    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(backbuffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let row_pitch = ((64 * 4 + 255) / 256) * 256;
    let dst = D3D12_TEXTURE_COPY_LOCATION {
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
    unsafe {
        cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    }
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    let px = readback_rgba8(&readback, 64, 64, 32, 32);

    // Expected: 60/255 + (100/255) * 0.5 = 110/255
    let expected = 110;
    let diff = (px[0] as i32 - expected).abs();
    assert!(
        diff <= 2,
        "Additive bloom pixel {} did not match expected {} within tolerance (diff={})",
        px[0],
        expected,
        diff
    );
}

#[test]
fn test_postprocess_aces_filmic_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("PostprocessTestWnd5"),
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
            eprintln!("Skipping hardware ACES test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    // Flat 255 input -> x = 1.0
    let dds = create_solid_color_rgba8_dds(64, 64, 255, 255, 255, 255);
    let scene_tex_idx = renderer
        .load_texture_from_dds("scene_tex5.dds", &dds)
        .expect("load_texture_from_dds failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    {
        let mgr = renderer.postprocess_manager_mut();
        mgr.image_correction.exposure = 1.0;
        mgr.image_correction.gamma = 1.0;
        mgr.image_correction.saturation = 1.0;
        mgr.image_correction.vignette_control = [0.0, 0.0, 0.0, 0.0];
    }

    let push = PostprocessPushConstants {
        screen_res_inv: [1.0 / 64.0, 1.0 / 64.0],
        time_seconds: 0.0,
        bloom_factor: 0.0,
        scene_texture_index: scene_tex_idx.0,
        bloom_texture_index: u32::MAX,
        tonemap_mode: 1, // ACES Filmic
        color_grading_enabled: 0,
    };

    renderer.draw_postprocess(&push).expect("draw_postprocess failed");

    // Copy to readback buffer
    let backbuffer = renderer
        .swapchain
        .as_ref()
        .unwrap()
        .current_backbuffer_resource();
    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );

    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(backbuffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let row_pitch = ((64 * 4 + 255) / 256) * 256;
    let dst = D3D12_TEXTURE_COPY_LOCATION {
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
    unsafe {
        cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
    }
    transition(
        cmd,
        backbuffer,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    let px = readback_rgba8(&readback, 64, 64, 32, 32);

    // ACES curve: x = 1.0: (1.0 * (2.51 * 1.0 + 0.03)) / (1.0 * (2.43 * 1.0 + 0.59) + 0.14) = 2.54 / 3.16 = 0.803797
    let expected = ((2.54f32 / 3.16f32) * 255.0f32).round() as u8;
    let diff = (px[0] as i32 - expected as i32).abs();
    assert!(
        diff <= 2,
        "ACES Filmic pixel {} did not match expected {} within tolerance (diff={})",
        px[0],
        expected,
        diff
    );
}
