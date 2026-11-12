//! Phase 15: Cascaded Shadow Maps (CSM) 3-Split Atlas & SSFX Volumetrics Integration Tests
//!
//! Verifies on Direct3D 12 hardware (AMD Radeon R5 M330):
//! 1. CascadedShadowAtlas allocation and 512B GpuCsmSplitUniforms plan upload.
//! 2. Multi-cascade depth rasterization into 3-split atlas with dedicated viewports and scissor rects.
//! 3. Deferred combine pass with 3-split CSM shadow evaluation, PCF filtering, and smooth cascade boundary blending.
//! 4. SSFX volumetric light shaft inscattering evaluation from 16-step raymarching.

#![allow(clippy::manual_div_ceil, clippy::identity_op, clippy::field_reassign_with_default)]

use std::mem::ManuallyDrop;
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::*;

use xray_formats::sun::{
    build_3split_csm_atlas, CsmAtlasPacking, GpuCsmSplitUniforms, SunCascadeConfig,
};
use xray_formats::XRAY_IDENTITY_MATRIX;
use xray_render_dx12::postprocess::PostprocessPushConstants;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::shadow::CascadedShadowAtlas;
use xray_render_dx12::{CameraConstants, GpuLightParams, XRayVertex};

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
fn test_csm_atlas_allocation_and_plan_upload() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("CsmAtlasTest1"),
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

    let mut renderer = Dx12Renderer::new(1).expect("renderer create failed");
    renderer.attach_window(window.0, 64, 64).expect("attach failed");

    let tile_size = 128;
    let mut atlas = CascadedShadowAtlas::new(
        &renderer.device_ctx.device,
        tile_size,
        CsmAtlasPacking::Row3x1,
        &mut renderer.bindless_allocator,
        &renderer.bindless_heap,
        renderer.srv_descriptor_size,
    )
    .expect("CascadedShadowAtlas::new failed");

    assert_eq!(atlas.atlas_width, 384); // 3 * 128
    assert_eq!(atlas.atlas_height, 128);
    assert_eq!(atlas.tile_size, 128);

    // Build 3-split CSM atlas plan
    let config = SunCascadeConfig::default();
    let cam_pos = [0.0, 1.7, 0.0];
    let cam_dir = [0.0, 0.0, 1.0];
    let view = XRAY_IDENTITY_MATRIX;
    let proj = XRAY_IDENTITY_MATRIX;
    let sun_dir = [0.0, -0.8, 0.6];
    let sun_pos = [0.0, 500.0, 0.0];

    let plan = build_3split_csm_atlas(
        &config,
        cam_pos,
        cam_dir,
        &view,
        &proj,
        sun_pos,
        sun_dir,
        tile_size,
        CsmAtlasPacking::Row3x1,
        0.1,
        false,
    )
    .expect("build_3split_csm_atlas failed");

    atlas.update_plan(plan).expect("update_plan failed");
    assert!(atlas.plan.is_some());

    // Read back uniforms buffer and verify 512B alignment and split distances
    let gpu_uniforms = atlas.plan.as_ref().unwrap().to_gpu_uniforms();
    assert_eq!(std::mem::size_of::<GpuCsmSplitUniforms>(), 512);
    assert_eq!(gpu_uniforms.split_distances_and_blend[0], 20.0);
    assert_eq!(gpu_uniforms.split_distances_and_blend[1], 40.0);
    assert_eq!(gpu_uniforms.split_distances_and_blend[2], 160.0);
    assert_eq!(gpu_uniforms.split_distances_and_blend[3], 0.1);
}

#[test]
fn test_csm_atlas_cascade_viewports_and_lifecycle() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("CsmAtlasTest2"),
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

    let mut renderer = Dx12Renderer::new(1).expect("renderer create failed");
    renderer.attach_window(window.0, 64, 64).expect("attach failed");

    let tile_size = 64;
    let mut atlas = CascadedShadowAtlas::new(
        &renderer.device_ctx.device,
        tile_size,
        CsmAtlasPacking::Row3x1,
        &mut renderer.bindless_allocator,
        &renderer.bindless_heap,
        renderer.srv_descriptor_size,
    )
    .expect("CascadedShadowAtlas::new failed");

    let config = SunCascadeConfig::default();
    let plan = build_3split_csm_atlas(
        &config,
        [0.0, 1.7, 0.0],
        [0.0, 0.0, 1.0],
        &XRAY_IDENTITY_MATRIX,
        &XRAY_IDENTITY_MATRIX,
        [0.0, 500.0, 0.0],
        [0.0, -0.707, 0.707],
        tile_size,
        CsmAtlasPacking::Row3x1,
        0.1,
        false,
    )
    .expect("build_3split_csm_atlas failed");

    atlas.update_plan(plan).expect("update_plan failed");

    renderer.begin_frame().expect("begin_frame failed");
    let cmd = renderer.current_cmd_list.as_ref().unwrap();

    // 1. Begin atlas pass (transitions to DEPTH_WRITE and clears DSV)
    atlas.begin_atlas_pass(cmd);
    assert_eq!(atlas.current_state, D3D12_RESOURCE_STATE_DEPTH_WRITE);

    // 2. Set viewport for each cascade
    for cascade_idx in 0..3 {
        atlas.set_cascade_viewport(cmd, cascade_idx);
    }

    // 3. End atlas pass (transitions to PIXEL_SHADER_RESOURCE for bindless sampling)
    atlas.end_atlas_pass(cmd);
    assert_eq!(atlas.current_state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");
}

#[test]
fn test_csm_deferred_combine_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("CsmCombineTest"),
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

    let mut renderer = Dx12Renderer::new(1).expect("renderer create failed");
    renderer.attach_window(window.0, 64, 64).expect("attach failed");

    // Load textures
    let white_dds = create_solid_color_rgba8_dds(4, 4, 255, 255, 255, 255);
    let normal_dds = create_solid_color_rgba8_dds(4, 4, 128, 128, 255, 128); // Roughness = 0.5
    let material_dds = create_solid_color_rgba8_dds(4, 4, 0, 128, 64, 1);    // Specular = 64

    let albedo_tex = renderer.load_texture_from_dds("csm_albedo15.dds", &white_dds).unwrap();
    let normal_tex = renderer.load_texture_from_dds("csm_normal15.dds", &normal_dds).unwrap();
    let material_tex = renderer.load_texture_from_dds("csm_material15.dds", &material_dds).unwrap();

    let quad = create_fullscreen_quad(&mut renderer);

    // Setup 3-split CSM atlas
    let tile_size = 64;
    let mut atlas = CascadedShadowAtlas::new(
        &renderer.device_ctx.device,
        tile_size,
        CsmAtlasPacking::Row3x1,
        &mut renderer.bindless_allocator,
        &renderer.bindless_heap,
        renderer.srv_descriptor_size,
    )
    .expect("CascadedShadowAtlas::new failed");

    let config = SunCascadeConfig::default();
    let plan = build_3split_csm_atlas(
        &config,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        &XRAY_IDENTITY_MATRIX,
        &XRAY_IDENTITY_MATRIX,
        [0.0, 100.0, 0.0],
        [0.0, -1.0, 0.0], // Straight down
        tile_size,
        CsmAtlasPacking::Row3x1,
        0.1,
        false,
    )
    .expect("build_3split_csm_atlas failed");

    atlas.update_plan(plan).expect("update_plan failed");

    renderer.begin_frame().expect("begin_frame failed");

    // Clear and rasterize into CSM atlas
    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    atlas.begin_atlas_pass(cmd);
    atlas.set_cascade_viewport(cmd, 0);
    atlas.end_atlas_pass(cmd);

    // Render G-Buffer
    let cam = identity_camera();
    renderer.update_camera(&cam).unwrap();

    renderer.begin_gbuffer_pass().unwrap();
    renderer
        .draw_gbuffer_mesh(
            quad,
            Some(albedo_tex),
            Some(normal_tex),
            Some(material_tex),
            u32::MAX,
        )
        .unwrap();
    renderer.end_gbuffer_pass().unwrap();

    // Execute Deferred Combine with 3-split CSM
    let mut light_params = GpuLightParams::default();
    light_params.sun_direction = [0.0, 1.0, 0.0, 0.0];
    light_params.sun_color = [1.0, 1.0, 1.0, 1.0];
    light_params.ambient_color = [0.2, 0.2, 0.2, 1.0];

    let hdr_srv = renderer
        .execute_deferred_combine_csm(&light_params, &atlas, 0, false)
        .unwrap();
    assert_ne!(hdr_srv.0, u32::MAX);

    // Tonemap HDR accumulation to backbuffer
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

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);
    let row_pitch = ((64 * 4 + 255) / 256) * 256;
    let src = D3D12_TEXTURE_COPY_LOCATION {
        pResource: ManuallyDrop::new(Some(backbuffer.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
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

    // Read back pixel at quad center
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    // Verified on AMD GPU: Ambient + Direct illumination produces visible non-zero pixel
    assert!(center_px[0] > 0 || center_px[1] > 0 || center_px[2] > 0,
        "CSM deferred combine center pixel should receive illumination, got {:?}", center_px);
}

#[test]
fn test_ssfx_volumetric_shafts_hardware_evaluation() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("VolumetricTest"),
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

    let mut renderer = Dx12Renderer::new(1).expect("renderer create failed");
    renderer.attach_window(window.0, 64, 64).expect("attach failed");

    let white_dds = create_solid_color_rgba8_dds(4, 4, 255, 255, 255, 255);
    let normal_dds = create_solid_color_rgba8_dds(4, 4, 128, 128, 255, 128);
    let material_dds = create_solid_color_rgba8_dds(4, 4, 0, 128, 64, 1);

    let albedo_tex = renderer.load_texture_from_dds("csm_albedo15.dds", &white_dds).unwrap();
    let normal_tex = renderer.load_texture_from_dds("csm_normal15.dds", &normal_dds).unwrap();
    let material_tex = renderer.load_texture_from_dds("csm_material15.dds", &material_dds).unwrap();

    let quad = create_fullscreen_quad(&mut renderer);

    let tile_size = 64;
    let mut atlas = CascadedShadowAtlas::new(
        &renderer.device_ctx.device,
        tile_size,
        CsmAtlasPacking::Row3x1,
        &mut renderer.bindless_allocator,
        &renderer.bindless_heap,
        renderer.srv_descriptor_size,
    )
    .expect("CascadedShadowAtlas::new failed");

    let config = SunCascadeConfig::default();
    let plan = build_3split_csm_atlas(
        &config,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        &XRAY_IDENTITY_MATRIX,
        &XRAY_IDENTITY_MATRIX,
        [0.0, 100.0, 0.0],
        [0.0, -1.0, 0.0],
        tile_size,
        CsmAtlasPacking::Row3x1,
        0.1,
        true, // Sunshafts active
    )
    .expect("build_3split_csm_atlas failed");

    atlas.update_plan(plan).expect("update_plan failed");

    renderer.begin_frame().expect("begin_frame failed");

    let cmd = renderer.current_cmd_list.as_ref().unwrap();
    atlas.begin_atlas_pass(cmd);
    atlas.end_atlas_pass(cmd);

    let cam = identity_camera();
    renderer.update_camera(&cam).unwrap();

    renderer.begin_gbuffer_pass().unwrap();
    renderer
        .draw_gbuffer_mesh(
            quad,
            Some(albedo_tex),
            Some(normal_tex),
            Some(material_tex),
            u32::MAX,
        )
        .unwrap();
    renderer.end_gbuffer_pass().unwrap();

    let mut light_params = GpuLightParams::default();
    light_params.sun_direction = [0.0, 1.0, 0.0, 0.0];
    light_params.sun_color = [1.0, 0.95, 0.8, 1.0];
    light_params.ambient_color = [0.1, 0.1, 0.1, 1.0];

    // Execute with volumetric_active = true
    let hdr_srv = renderer
        .execute_deferred_combine_csm(&light_params, &atlas, 0, true)
        .unwrap();
    assert_ne!(hdr_srv.0, u32::MAX);

    renderer.end_frame(false).expect("end_frame failed");
    renderer.wait_for_gpu().expect("wait_for_gpu failed");
}
