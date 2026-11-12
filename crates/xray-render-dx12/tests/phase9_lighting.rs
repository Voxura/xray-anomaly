//! Phase 9: Dynamic Lighting Hardware Readback Integration Tests
//!
//! Verifies on Direct3D 12 hardware:
//! 1. Point light distance falloff and quadratic attenuation on 3D geometry
//! 2. Spot light cone falloff (inner/outer angular attenuation)
//! 3. Animated light color streaming via `xray_formats::lanims::LightAnimItem`

use std::mem::ManuallyDrop;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::w;

use xray_formats::lanims::LightAnimItem;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::{CameraConstants, DynamicLight, XRayVertex};

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

/// Helper to generate a solid RGBA8 uncompressed DDS file in memory
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
    dds[4..8].copy_from_slice(&124u32.to_le_bytes()); // header size
    dds[8..12].copy_from_slice(&(0x1u32 | 0x2 | 0x4 | 0x1000).to_le_bytes()); // flags: CAPS | HEIGHT | WIDTH | PIXELFORMAT
    dds[12..16].copy_from_slice(&height.to_le_bytes());
    dds[16..20].copy_from_slice(&width.to_le_bytes());
    dds[76..80].copy_from_slice(&32u32.to_le_bytes()); // pf_size
    dds[80..84].copy_from_slice(&(0x40u32 | 0x1).to_le_bytes()); // DDPF_RGB | DDPF_ALPHAPIXELS
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

/// Helper to build a centered XY quad at Z=0.5 with normal pointing towards -Z
fn create_test_quad(renderer: &mut Dx12Renderer) -> xray_render_dx12::MeshHandle {
    let vertices = [
        XRayVertex {
            position: [-0.8, -0.8, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [-0.8, 0.8, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.8, 0.8, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.8, -0.8, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    renderer
        .create_mesh(&vertices, &indices)
        .expect("create_mesh failed")
}

#[test]
fn test_point_light_attenuation_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("PointLightTest"),
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
            eprintln!("Skipping hardware point light test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let mesh = create_test_quad(&mut renderer);
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Point light positioned right in front of the quad at [0.0, 0.0, 0.1]
    let light = DynamicLight::new_point([0.0, 0.0, 0.1], 1.5, [1.0, 1.0, 1.0, 2.0]);
    let gpu_lights = [light.evaluate_at(0.0)];
    renderer.update_lights(&gpu_lights).expect("update_lights failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    renderer
        .draw_mesh_with_lights(mesh, None, u32::MAX, 1)
        .expect("draw_mesh_with_lights failed");

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

    // Center pixel (32, 32): Directly under point light
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    // Near-corner pixel (10, 10): Farther away from point light
    let corner_px = readback_rgba8(&readback, 64, 64, 10, 10);

    println!(
        "Point Light Readback: center = {:?}, corner = {:?}",
        center_px, corner_px
    );

    // Center pixel should receive significantly more light than corner pixel due to distance attenuation
    assert!(
        center_px[2] > corner_px[2],
        "Center intensity ({}) must be greater than corner intensity ({}) due to distance falloff",
        center_px[2],
        corner_px[2]
    );
}

#[test]
fn test_spot_light_cone_falloff_hardware_readback() {
    use std::f32::consts::PI;

    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("SpotLightTest"),
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
            eprintln!("Skipping hardware spot light test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let mesh = create_test_quad(&mut renderer);
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Spot light pointing along +Z directly at center [0, 0, 0.5] with narrow cone
    let spot = DynamicLight::new_spot(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        2.0,
        PI / 9.0, // 20 deg inner
        PI / 6.0, // 30 deg outer
        [1.0, 1.0, 1.0, 3.0],
    );
    let gpu_lights = [spot.evaluate_at(0.0)];
    renderer.update_lights(&gpu_lights).expect("update_lights failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    renderer
        .draw_mesh_with_lights(mesh, None, u32::MAX, 1)
        .expect("draw_mesh_with_lights failed");

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
    let corner_px = readback_rgba8(&readback, 64, 64, 10, 10);

    println!(
        "Spot Light Readback: center = {:?}, corner = {:?}",
        center_px, corner_px
    );

    assert!(
        center_px[2] > corner_px[2],
        "Center within cone ({}) must be brighter than corner outside cone ({})",
        center_px[2],
        corner_px[2]
    );
}

#[test]
fn test_animated_light_hardware_readback() {
    use std::collections::BTreeMap;

    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("AnimLightTest"),
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
            eprintln!("Skipping hardware anim light test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    // Upload white albedo texture so lit color directly shows light color
    let white_dds = create_solid_color_rgba8_dds(4, 4, 255, 255, 255, 255);
    let tex_idx = renderer
        .load_texture_from_dds("white_light_surf.dds", &white_dds)
        .expect("load white texture failed");

    let mesh = create_test_quad(&mut renderer);
    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Keyframe 0: Green (0xFF00_FF00)
    // Keyframe 1: Blue  (0xFF00_00FF)
    let mut keys = BTreeMap::new();
    keys.insert(0, 0xFF00_FF00);
    keys.insert(1, 0xFF00_00FF);

    let anim = LightAnimItem {
        name: "color_cycle".to_string(),
        fps: 1.0,
        default_fps: 1.0,
        frame_count: 2,
        keys,
    };

    let light = DynamicLight::new_point([0.0, 0.0, 0.1], 1.5, [1.0, 1.0, 1.0, 2.0]).with_anim(anim);

    // Frame 1: at t = 0.0, should be Green
    let gpu_lights_t0 = [light.evaluate_at(0.0)];
    renderer.update_lights(&gpu_lights_t0).expect("update_lights t0 failed");

    let readback_t0 = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer
        .draw_mesh_with_lights(mesh, Some(tex_idx), u32::MAX, 1)
        .expect("draw t0 failed");

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
            pResource: ManuallyDrop::new(Some(readback_t0.clone())),
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
    renderer.end_frame(false).expect("end_frame t0 failed");
    renderer.wait_for_gpu().expect("wait_for_gpu t0 failed");

    let px_t0 = readback_rgba8(&readback_t0, 64, 64, 32, 32);
    println!("Animated Light t=0.0 Readback: {:?}", px_t0);
    assert!(px_t0[1] > px_t0[2], "Green channel ({}) must exceed blue ({}) at t=0", px_t0[1], px_t0[2]);

    // Frame 2: at t = 1.0, should be Blue
    let gpu_lights_t1 = [light.evaluate_at(1.0)];
    renderer.update_lights(&gpu_lights_t1).expect("update_lights t1 failed");

    let readback_t1 = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer
        .draw_mesh_with_lights(mesh, Some(tex_idx), u32::MAX, 1)
        .expect("draw t1 failed");

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
            pResource: ManuallyDrop::new(Some(readback_t1.clone())),
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
    renderer.end_frame(false).expect("end_frame t1 failed");
    renderer.wait_for_gpu().expect("wait_for_gpu t1 failed");

    let px_t1 = readback_rgba8(&readback_t1, 64, 64, 32, 32);
    println!("Animated Light t=1.0 Readback: {:?}", px_t1);
    assert!(px_t1[2] > px_t1[1], "Blue channel ({}) must exceed green ({}) at t=1", px_t1[2], px_t1[1]);
}
