//! Phase 14: Deferred Combine & G-Buffer Hardware Readback Integration Tests
//!
//! Verifies on Direct3D 12 hardware (AMD Radeon R5 M330):
//! 1. G-Buffer MRT rasterization (Albedo, Normal, Material, Depth)
//! 2. Deferred Lighting Combine pass with Directional Sunlight + Ambient
//! 3. Dynamic point light distance attenuation in deferred combine pass
//! 4. Full end-to-end Deferred Shading + Phase 13 Tonemapping pipeline

use std::mem::ManuallyDrop;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::w;

use xray_render_dx12::postprocess::PostprocessPushConstants;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::{CameraConstants, DynamicLight, GpuLightParams, XRayVertex};

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

/// Helper quad covering [-0.5, 0.5] centered in NDC with normal pointing towards -Z
fn create_centered_quad(renderer: &mut Dx12Renderer) -> xray_render_dx12::MeshHandle {
    let vertices = [
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
        XRayVertex {
            position: [-0.5, -0.5, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    renderer
        .create_mesh(&vertices, &indices)
        .expect("create_mesh failed")
}

/// Fullscreen quad covering entire [-1, 1] screen area
fn create_fullscreen_quad(renderer: &mut Dx12Renderer) -> xray_render_dx12::MeshHandle {
    let vertices = [
        XRayVertex {
            position: [-1.0, 1.0, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [1.0, 1.0, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [1.0, -1.0, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [-1.0, -1.0, 0.5],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    renderer
        .create_mesh(&vertices, &indices)
        .expect("create_mesh failed")
}

#[test]
fn test_gbuffer_mrt_rasterization_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("GBufferTestWnd1"),
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
            eprintln!("Skipping hardware G-Buffer test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let mesh_handle = create_centered_quad(&mut renderer);

    // Orange Albedo texture (220, 110, 40)
    let albedo_dds = create_solid_color_rgba8_dds(4, 4, 220, 110, 40, 255);
    let albedo_idx = renderer
        .load_texture_from_dds("albedo_orange.dds", &albedo_dds)
        .expect("load albedo failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.update_camera(&identity_camera()).expect("update_camera failed");
    renderer.begin_frame().expect("begin_frame failed");

    renderer.begin_gbuffer_pass().expect("begin_gbuffer_pass failed");
    renderer
        .draw_gbuffer_mesh(mesh_handle, Some(albedo_idx), None, None, u32::MAX)
        .expect("draw_gbuffer_mesh failed");
    renderer.end_gbuffer_pass().expect("end_gbuffer_pass failed");

    // Copy GBuffer Albedo target to readback buffer
    let albedo_res = {
        let gbuffer = renderer.gbuffer_manager_mut().unwrap();
        gbuffer.targets.as_ref().unwrap().albedo.clone()
    };

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    transition(
        cmd,
        &albedo_res,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );

    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(albedo_res.clone())),
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
        &albedo_res,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    );

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");

    // Center pixel (32, 32) inside the quad should contain the orange albedo
    let center = readback_rgba8(&readback, 64, 64, 32, 32);
    assert!(
        (center[0] as i32 - 220).abs() <= 3,
        "Center R should match albedo 220, got {}",
        center[0]
    );
    assert!(
        (center[1] as i32 - 110).abs() <= 3,
        "Center G should match albedo 110, got {}",
        center[1]
    );
    assert!(
        (center[2] as i32 - 40).abs() <= 3,
        "Center B should match albedo 40, got {}",
        center[2]
    );

    // Corner pixel (2, 2) outside the quad should be black (cleared background [0, 0, 0, 255])
    let corner = readback_rgba8(&readback, 64, 64, 2, 2);
    assert_eq!(
        corner,
        [0, 0, 0, 255],
        "Corner outside quad should be cleared black, got {:?}",
        corner
    );
}

#[test]
fn test_deferred_combine_directional_sunlight() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("GBufferTestWnd2"),
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
            eprintln!("Skipping hardware sunlight combine test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let mesh_handle = create_fullscreen_quad(&mut renderer);

    // Pure white Albedo texture
    let albedo_dds = create_solid_color_rgba8_dds(4, 4, 255, 255, 255, 255);
    let albedo_idx = renderer
        .load_texture_from_dds("albedo_white.dds", &albedo_dds)
        .expect("load albedo failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.update_camera(&identity_camera()).expect("update_camera failed");
    renderer.begin_frame().expect("begin_frame failed");

    // 1. G-Buffer MRT pass
    renderer.begin_gbuffer_pass().expect("begin_gbuffer_pass failed");
    renderer
        .draw_gbuffer_mesh(mesh_handle, Some(albedo_idx), None, None, u32::MAX)
        .expect("draw_gbuffer_mesh failed");
    renderer.end_gbuffer_pass().expect("end_gbuffer_pass failed");

    // 2. Deferred Combine pass: Directional Sunlight pointing along -Z towards normal [0, 0, -1]
    let light_params = GpuLightParams {
        sun_direction: [0.0, 0.0, -1.0, 0.0],
        sun_color: [1.0, 1.0, 1.0, 1.0],
        ambient_color: [0.2, 0.2, 0.2, 1.0],
        ..Default::default()
    };
    let hdr_srv = renderer
        .execute_deferred_combine(&light_params, None, 0)
        .expect("execute_deferred_combine failed");

    // 3. Postprocess pass: pass-through to backbuffer
    let post_push = PostprocessPushConstants {
        screen_res_inv: [1.0 / 64.0, 1.0 / 64.0],
        time_seconds: 0.0,
        bloom_factor: 0.0,
        scene_texture_index: hdr_srv.0,
        bloom_texture_index: u32::MAX,
        tonemap_mode: 3, // Pass-through clamped
        color_grading_enabled: 0,
    };
    renderer
        .draw_postprocess(&post_push)
        .expect("draw_postprocess failed");

    // Copy backbuffer to readback
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
    // Sun diffuse dot(N, L) = dot([0, 0, -1], [0, 0, -1]) = 1.0. SunColor = 1.0, Ambient = 0.2.
    // Total HDR = 1.0 + 0.2 = 1.2, clamped by pass-through tonemap to 1.0 (255)
    assert!(
        center[0] >= 250,
        "Center R should be near max 255 from direct sun, got {}",
        center[0]
    );
    assert!(
        center[1] >= 250,
        "Center G should be near max 255 from direct sun, got {}",
        center[1]
    );
    assert!(
        center[2] >= 250,
        "Center B should be near max 255 from direct sun, got {}",
        center[2]
    );
}

#[test]
fn test_deferred_combine_dynamic_point_light_attenuation() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("GBufferTestWnd3"),
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
            eprintln!("Skipping hardware point light combine test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let mesh_handle = create_fullscreen_quad(&mut renderer);

    let albedo_dds = create_solid_color_rgba8_dds(4, 4, 255, 255, 255, 255);
    let albedo_idx = renderer
        .load_texture_from_dds("albedo_white_pt.dds", &albedo_dds)
        .expect("load albedo failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.update_camera(&identity_camera()).expect("update_camera failed");

    // Dynamic point light at NDC center (0, 0, 0.2) in front of quad (z=0.5)
    // Pure green light with range 1.0
    let green_light = DynamicLight::new_point([0.0, 0.0, 0.2], 1.0, [0.0, 1.0, 0.0, 1.0]);
    let gpu_lights = [green_light.evaluate_at(0.0)];
    renderer.update_lights(&gpu_lights).expect("update_lights failed");

    renderer.begin_frame().expect("begin_frame failed");

    // 1. G-Buffer MRT pass
    renderer.begin_gbuffer_pass().expect("begin_gbuffer_pass failed");
    renderer
        .draw_gbuffer_mesh(mesh_handle, Some(albedo_idx), None, None, u32::MAX)
        .expect("draw_gbuffer_mesh failed");
    renderer.end_gbuffer_pass().expect("end_gbuffer_pass failed");

    // 2. Deferred combine: No sun, no ambient (only dynamic point light)
    let light_params = GpuLightParams {
        sun_direction: [0.0, 0.0, -1.0, 0.0],
        sun_color: [0.0, 0.0, 0.0, 0.0],
        ambient_color: [0.0, 0.0, 0.0, 0.0],
        ..Default::default()
    };
    let hdr_srv = renderer
        .execute_deferred_combine(&light_params, None, 1)
        .expect("execute_deferred_combine failed");

    // 3. Postprocess pass
    let post_push = PostprocessPushConstants {
        screen_res_inv: [1.0 / 64.0, 1.0 / 64.0],
        time_seconds: 0.0,
        bloom_factor: 0.0,
        scene_texture_index: hdr_srv.0,
        bloom_texture_index: u32::MAX,
        tonemap_mode: 3,
        color_grading_enabled: 0,
    };
    renderer
        .draw_postprocess(&post_push)
        .expect("draw_postprocess failed");

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

    // Center pixel (32, 32) is close to the point light (distance = 0.3) -> strongly lit green
    let center = readback_rgba8(&readback, 64, 64, 32, 32);
    assert_eq!(center[0], 0, "R channel should be 0 for pure green light");
    assert!(
        center[1] >= 100,
        "Center G should be strongly lit (> 100), got {}",
        center[1]
    );
    assert_eq!(center[2], 0, "B channel should be 0 for pure green light");

    // Corner pixel (2, 2) has distance ~ 1.45, exceeding light range 1.0 -> 0 illumination
    let corner = readback_rgba8(&readback, 64, 64, 2, 2);
    assert_eq!(
        corner[1], 0,
        "Corner G should be 0 outside point light range, got {}",
        corner[1]
    );
    assert!(
        center[1] > corner[1] + 100,
        "Center must be significantly brighter than corner"
    );
}

#[test]
fn test_deferred_full_pipeline_gbuffer_combine_and_tonemap() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("GBufferTestWnd4"),
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
            eprintln!("Skipping hardware full pipeline test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let mesh_handle = create_fullscreen_quad(&mut renderer);

    // Sky blue albedo
    let albedo_dds = create_solid_color_rgba8_dds(4, 4, 100, 150, 220, 255);
    let albedo_idx = renderer
        .load_texture_from_dds("albedo_blue.dds", &albedo_dds)
        .expect("load albedo failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.update_camera(&identity_camera()).expect("update_camera failed");
    renderer.begin_frame().expect("begin_frame failed");

    // Step 1: Render into GBuffer MRTs
    renderer.begin_gbuffer_pass().expect("begin_gbuffer_pass failed");
    renderer
        .draw_gbuffer_mesh(mesh_handle, Some(albedo_idx), None, None, u32::MAX)
        .expect("draw_gbuffer_mesh failed");
    renderer.end_gbuffer_pass().expect("end_gbuffer_pass failed");

    // Step 2: Combine Lighting into HDR
    let light_params = GpuLightParams {
        sun_direction: [0.0, 0.0, -1.0, 0.0],
        sun_color: [1.2, 1.2, 1.0, 1.0],
        ambient_color: [0.15, 0.15, 0.2, 1.0],
        ..Default::default()
    };
    let hdr_srv = renderer
        .execute_deferred_combine(&light_params, None, 0)
        .expect("execute_deferred_combine failed");

    // Step 3: Phase 13 Uncharted 2 Tonemap + Vignette postprocess
    let post_push = PostprocessPushConstants {
        screen_res_inv: [1.0 / 64.0, 1.0 / 64.0],
        time_seconds: 0.0,
        bloom_factor: 0.0,
        scene_texture_index: hdr_srv.0,
        bloom_texture_index: u32::MAX,
        tonemap_mode: 0, // Uncharted 2 Filmic
        color_grading_enabled: 0,
    };
    renderer
        .draw_postprocess(&post_push)
        .expect("draw_postprocess failed");

    // Step 4: Readback final composited backbuffer
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
    let corner = readback_rgba8(&readback, 64, 64, 2, 2);

    // Vignette darkens corners compared to center
    assert!(
        center[2] > corner[2],
        "Center blue {} should be brighter than vignette corner blue {}",
        center[2],
        corner[2]
    );
    // Preserves color hue: Blue is dominant
    assert!(
        center[2] > center[0] && center[2] > center[1],
        "Blue component should dominate for sky blue albedo"
    );
}
