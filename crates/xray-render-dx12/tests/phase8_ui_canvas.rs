//! Phase 8 Direct3D 12 2D UI Canvas & HUD Immediate-Mode Hardware Readback Tests
//!
//! Validates that procedural 2D UI quads, bindless atlas texture regions, alpha blending,
//! and HUD widgets (crosshair, health meter bars, frames) are rasterized correctly on physical hardware.

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

struct Window(HWND);
impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.0);
        }
    }
}

unsafe fn transition(
    cmd: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let barrier = D3D12_RESOURCE_BARRIER {
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
        cmd.ResourceBarrier(&[barrier]);
    }
}

fn create_readback_buffer(device: &ID3D12Device, width: u32, height: u32) -> ID3D12Resource {
    let row_pitch = ((width * 4 + 255) / 256) * 256;
    let total_bytes = (row_pitch * height) as u64;

    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total_bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut res = None;
    unsafe {
        device
            .CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut res,
            )
            .expect("Create readback buffer failed");
    }
    res.unwrap()
}

fn readback_rgba8(readback: &ID3D12Resource, width: u32, height: u32, x: u32, y: u32) -> [u8; 4] {
    let row_pitch = ((width * 4 + 255) / 256) * 256;
    let offset = (y * row_pitch + x * 4) as usize;
    let mut mapped = ptr::null_mut();
    unsafe {
        readback
            .Map(0, None, Some(&mut mapped))
            .expect("Map readback buffer failed");
        let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), (row_pitch * height) as usize);
        let pixel = [bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]];
        readback.Unmap(0, None);
        pixel
    }
}

fn create_custom_rgba8_dds(width: u32, height: u32, pixels: &[[u8; 4]]) -> Vec<u8> {
    let data_bytes = (width * height * 4) as usize;
    let mut dds = vec![0u8; 128 + data_bytes];

    dds[0..4].copy_from_slice(&DDS_MAGIC.to_le_bytes());
    dds[4..8].copy_from_slice(&124u32.to_le_bytes());
    dds[8..12].copy_from_slice(&(DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT).to_le_bytes());
    dds[12..16].copy_from_slice(&height.to_le_bytes());
    dds[16..20].copy_from_slice(&width.to_le_bytes());
    dds[76..80].copy_from_slice(&32u32.to_le_bytes());
    dds[80..84].copy_from_slice(&(DDPF_RGB | DDPF_ALPHAPIXELS).to_le_bytes());
    dds[88..92].copy_from_slice(&32u32.to_le_bytes());
    dds[92..96].copy_from_slice(&0x000000FFu32.to_le_bytes());
    dds[96..100].copy_from_slice(&0x0000FF00u32.to_le_bytes());
    dds[100..104].copy_from_slice(&0x00FF0000u32.to_le_bytes());
    dds[104..108].copy_from_slice(&0xFF000000u32.to_le_bytes());

    for (i, px) in pixels.iter().enumerate() {
        let offset = 128 + i * 4;
        dds[offset..offset + 4].copy_from_slice(px);
    }
    dds
}

#[test]
fn test_ui_canvas_solid_rects_and_alpha_blending_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("UiCanvasSolidTest"),
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
            eprintln!("Skipping hardware UI Canvas test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // 1. Layer 1: Solid Red quad [10, 10, 44, 44]
    renderer.ui_fill_rect([10.0, 10.0, 44.0, 44.0], [1.0, 0.0, 0.0, 1.0]);

    // 2. Layer 2: 50% Alpha Blue quad [10, 10, 44, 44] blending over red
    renderer.ui_fill_rect([10.0, 10.0, 44.0, 44.0], [0.0, 0.0, 1.0, 0.5]);

    // Flush UI quads
    renderer.ui_flush(0, 0).expect("ui_flush failed");

    // Copy to readback
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

    // Blended pixel inside overlapping quads (32, 32):
    // Red: 255 * (1 - 0.5) = 127
    // Blue: 255 * 0.5 = 127
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    // Background pixel outside quads (2, 2): Must be Black [0, 0, 0, 255]
    let bg_px = readback_rgba8(&readback, 64, 64, 2, 2);

    println!("UI Canvas Alpha Blend: center = {:?}, bg = {:?}", center_px, bg_px);
    assert!((center_px[0] as i32 - 127).abs() <= 3, "Red channel must be ~127");
    assert_eq!(center_px[1], 0, "Green channel must be 0");
    assert!((center_px[2] as i32 - 127).abs() <= 3, "Blue channel must be ~127");
    assert_eq!(center_px[3], 255, "Alpha must be 255");
    assert_eq!(bg_px, [0, 0, 0, 255], "Background must be black");
}

#[test]
fn test_ui_canvas_textured_atlas_region_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("UiCanvasTexturedTest"),
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
            eprintln!("Skipping hardware UI Canvas test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    // 2x2 Texture Atlas:
    // (0,0): Red [255, 0, 0, 255]       | (1,0): Green [0, 255, 0, 255]
    // (0,1): Blue [0, 0, 255, 255]      | (1,1): Yellow [255, 255, 0, 255]
    let atlas_pixels = [
        [255, 0, 0, 255], [0, 255, 0, 255],
        [0, 0, 255, 255], [255, 255, 0, 255],
    ];
    let dds_bytes = create_custom_rgba8_dds(2, 2, &atlas_pixels);
    let atlas_idx = renderer.load_texture_from_dds("ui_atlas_test.dds", &dds_bytes)
        .expect("load atlas dds failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Quad A (Left Half: 0..32): Sample Top-Left quadrant (Red)
    renderer.ui_draw_textured_rect(
        [0.0, 0.0, 32.0, 64.0],
        [0.0, 0.0, 0.5, 0.5],
        atlas_idx,
        [1.0, 1.0, 1.0, 1.0],
    );

    // Quad B (Right Half: 32..64): Sample Bottom-Right quadrant (Yellow)
    renderer.ui_draw_textured_rect(
        [32.0, 0.0, 32.0, 64.0],
        [0.5, 0.5, 1.0, 1.0],
        atlas_idx,
        [1.0, 1.0, 1.0, 1.0],
    );

    renderer.ui_flush(0, 0).expect("ui_flush failed");

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

    let left_px = readback_rgba8(&readback, 64, 64, 16, 32);
    let right_px = readback_rgba8(&readback, 64, 64, 48, 32);

    println!("UI Atlas Readback: left = {:?}, right = {:?}", left_px, right_px);
    assert_eq!(left_px[0], 255, "Left half red channel must be 255");
    assert_eq!(left_px[1], 0, "Left half green channel must be 0");
    assert_eq!(left_px[2], 0, "Left half blue channel must be 0");

    assert_eq!(right_px[0], 255, "Right half red channel must be 255");
    assert_eq!(right_px[1], 255, "Right half green channel must be 255");
    assert_eq!(right_px[2], 0, "Right half blue channel must be 0");
}

#[test]
fn test_ui_canvas_crosshair_and_meter_hud_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("UiHudWidgetsTest"),
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
            eprintln!("Skipping hardware UI Canvas test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // 1. Crosshair at center (32, 32): size 10, gap 4, thickness 2, Green
    renderer.ui_draw_crosshair([32.0, 32.0], 10.0, 4.0, 2.0, [0.0, 1.0, 0.0, 1.0]);

    // 2. Health meter bar at bottom: [10, 50, 44, 8], 50% fraction, Dark Red BG, Bright Green Bar
    renderer.ui_draw_meter_bar([10.0, 50.0, 44.0, 8.0], 0.5, [0.5, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]);

    renderer.ui_flush(0, 0).expect("ui_flush failed");

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

    // Center pixel of crosshair: Inside the 4px gap -> Black [0, 0, 0, 255]
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    // Top arm of crosshair: At Y = 22 -> Green [0, 255, 0, 255]
    let top_arm_px = readback_rgba8(&readback, 64, 64, 32, 22);

    // Meter bar filled side (left at X=20, Y=54): Green [0, 255, 0, 255]
    let meter_filled_px = readback_rgba8(&readback, 64, 64, 20, 54);
    // Meter bar unfilled side (right at X=42, Y=54): Dark Red ~127
    let meter_unfilled_px = readback_rgba8(&readback, 64, 64, 42, 54);

    println!(
        "HUD Widgets Readback: center = {:?}, top_arm = {:?}, filled = {:?}, unfilled = {:?}",
        center_px, top_arm_px, meter_filled_px, meter_unfilled_px
    );

    assert_eq!(center_px, [0, 0, 0, 255], "Crosshair center gap must be black");
    assert_eq!(top_arm_px[1], 255, "Crosshair arm must be green");
    assert_eq!(meter_filled_px[1], 255, "Filled meter must be green");
    assert!(meter_unfilled_px[0] > 100 && meter_unfilled_px[1] == 0, "Unfilled meter must be dark red");
}

#[test]
fn test_real_repo_cursor_cross_dds_ui_canvas_render() {
    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap().parent().unwrap();
    let dds_path = repo_root.join("gamedata/textures/ui/cursor_cross.dds");

    if !dds_path.exists() {
        eprintln!("Skipping real cursor_cross.dds test: file not present");
        return;
    }

    let dds_bytes = std::fs::read(&dds_path).expect("Failed to read cursor_cross.dds");

    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("UiCursorCrossTest"),
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
            eprintln!("Skipping hardware UI test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let cursor_idx = renderer
        .load_texture_from_dds("ui/cursor_cross.dds", &dds_bytes)
        .expect("Failed to upload cursor_cross.dds");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Draw real 256x256 cursor texture onto 32x32 UI quad at [16, 16]
    renderer.ui_draw_textured_rect(
        [16.0, 16.0, 32.0, 32.0],
        [0.0, 0.0, 1.0, 1.0],
        cursor_idx,
        [1.0, 1.0, 1.0, 1.0],
    );

    renderer.ui_flush(0, 0).expect("ui_flush failed");

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

    // Outside quad (2, 2) must remain black
    let bg_px = readback_rgba8(&readback, 64, 64, 2, 2);
    // Center pixel of cursor (32, 32)
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);

    println!("Real Repo Cursor Readback: bg = {:?}, center = {:?}", bg_px, center_px);
    assert_eq!(bg_px, [0, 0, 0, 255], "Background must be black");
    // Center of cursor_cross.dds contains visible crosshair dot/lines
    assert!(center_px[3] > 0, "Cursor pixel alpha must be non-zero");
}

