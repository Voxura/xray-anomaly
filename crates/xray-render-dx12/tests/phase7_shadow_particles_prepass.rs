//! Phase 7 Depth-Prepass, Directional Sunlight Shadow Mapping, and Particle Billboard Instancing Tests
//!
//! Validates on AMD Radeon R5 M330 hardware:
//! 1. Hardware Depth-Prepass (null pixel shader) + EQUAL depth rejection test (zero pixel overdraw).
//! 2. Real-time Directional Sunlight Shadow Mapping pass + bindless sampling in geometry pixel shader.
//! 3. Ultra-low-overhead procedural quad Particle Billboard Instancing with Honey's `ParticleSpriteFrame` UVs.

use std::mem::ManuallyDrop;
use std::ptr;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, WS_OVERLAPPEDWINDOW,
};
use windows::core::w;
use xray_formats::particles::ParticleSpriteFrame;
use xray_render_dx12::particle::GpuParticleInstance;
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

    for i in 0..pixel_count {
        let offset = 128 + i * 4;
        dds[offset] = r;
        dds[offset + 1] = g;
        dds[offset + 2] = b;
        dds[offset + 3] = a;
    }
    dds
}

fn create_2x2_atlas_rgba8_dds() -> Vec<u8> {
    let mut dds = vec![0u8; 128 + 16];
    dds[0..4].copy_from_slice(&DDS_MAGIC.to_le_bytes());
    dds[4..8].copy_from_slice(&124u32.to_le_bytes());
    dds[8..12].copy_from_slice(&(DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT).to_le_bytes());
    dds[12..16].copy_from_slice(&2u32.to_le_bytes());
    dds[16..20].copy_from_slice(&2u32.to_le_bytes());
    dds[76..80].copy_from_slice(&32u32.to_le_bytes());
    dds[80..84].copy_from_slice(&(DDPF_RGB | DDPF_ALPHAPIXELS).to_le_bytes());
    dds[88..92].copy_from_slice(&32u32.to_le_bytes());
    dds[92..96].copy_from_slice(&0x000000FFu32.to_le_bytes());
    dds[96..100].copy_from_slice(&0x0000FF00u32.to_le_bytes());
    dds[100..104].copy_from_slice(&0x00FF0000u32.to_le_bytes());
    dds[104..108].copy_from_slice(&0xFF000000u32.to_le_bytes());

    // Row 0: Pixel 0 (Red), Pixel 1 (Cyan)
    dds[128..132].copy_from_slice(&[255, 0, 0, 255]);
    dds[132..136].copy_from_slice(&[0, 255, 255, 255]);
    // Row 1: Pixel 2 (Yellow), Pixel 3 (Magenta)
    dds[136..140].copy_from_slice(&[255, 255, 0, 255]);
    dds[140..144].copy_from_slice(&[255, 0, 255, 255]);

    dds
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
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    unsafe {
        let mut res: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut res,
            )
            .expect("Create readback buffer failed");
        res.unwrap()
    }
}

fn readback_rgba8(
    readback: &ID3D12Resource,
    width: u32,
    _height: u32,
    x: u32,
    y: u32,
) -> [u8; 4] {
    let row_pitch = ((width * 4 + 255) / 256) * 256;
    unsafe {
        let mut mapped = ptr::null_mut();
        readback.Map(0, None, Some(&mut mapped)).expect("Map readback failed");
        let offset = (y * row_pitch + x * 4) as usize;
        let mut pixel = [0u8; 4];
        ptr::copy_nonoverlapping((mapped as *const u8).add(offset), pixel.as_mut_ptr(), 4);
        readback.Unmap(0, None);
        pixel
    }
}

fn make_quad_vertices(z: f32) -> [XRayVertex; 4] {
    [
        XRayVertex {
            position: [-0.5, -0.5, z],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [-0.5, 0.5, z],
            normal: [0.0, 0.0, -1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.5, 0.5, z],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.5, -0.5, z],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ]
}

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

#[test]
fn test_depth_prepass_early_z_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("DepthPrepassEarlyZTest"),
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
            eprintln!("Skipping hardware depth prepass test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    let green_dds = create_solid_color_rgba8_dds(4, 4, 0, 255, 0, 255);
    let red_dds = create_solid_color_rgba8_dds(4, 4, 255, 0, 0, 255);
    let green_idx = renderer.load_texture_from_dds("green_prepass.dds", &green_dds).expect("load green failed");
    let red_idx = renderer.load_texture_from_dds("red_back.dds", &red_dds).expect("load red failed");

    // Quad A (Front): Z = 0.3
    let quad_a = make_quad_vertices(0.3);
    let quad_a_handle = renderer.create_mesh(&quad_a, &QUAD_INDICES).expect("create quad a failed");

    // Quad B (Back): Z = 0.7
    let quad_b = make_quad_vertices(0.7);
    let quad_b_handle = renderer.create_mesh(&quad_b, &QUAD_INDICES).expect("create quad b failed");

    let cam = identity_camera();
    renderer.update_camera(&cam).expect("update camera failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // 1. Prepass: Render front Quad A into depth buffer with null pixel shader (writes Z = 0.3)
    renderer.draw_depth_prepass(quad_a_handle, u32::MAX).expect("depth prepass failed");

    // 2. Early-Z Main Pass with EQUAL depth testing:
    // Front quad (Z = 0.3) matches depth prepass -> passes test and shades green!
    renderer.draw_mesh_equal_depth(quad_a_handle, Some(green_idx), None, u32::MAX, None)
        .expect("draw quad a equal depth failed");

    // Back quad (Z = 0.7) does NOT match depth prepass (0.3) -> 100% rejected by early-Z test!
    renderer.draw_mesh_equal_depth(quad_b_handle, Some(red_idx), None, u32::MAX, None)
        .expect("draw quad b equal depth failed");

    // Copy backbuffer to readback
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

    // Verify center pixel: Must be Green [0, 255, 0, 255], proving red was rejected by early-Z prepass
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Early-Z Prepass Test center pixel: {:?}", center_px);
    assert_eq!(center_px[0], 0, "Red must be 0 (rejected by early-Z)");
    assert_eq!(center_px[1], 255, "Green must be 255 (front quad shaded)");
    assert_eq!(center_px[3], 255, "Alpha must be 255");
}

#[test]
fn test_directional_shadow_map_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("ShadowMapHardwareTest"),
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
            eprintln!("Skipping hardware shadow test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    // Initialize 128x128 directional shadow map
    let shadow_idx = renderer.init_shadow_map(128).expect("init_shadow_map failed");

    // White texture for receiver ground
    let white_dds = create_solid_color_rgba8_dds(4, 4, 255, 255, 255, 255);
    let white_idx = renderer.load_texture_from_dds("white_receiver.dds", &white_dds).expect("load white failed");

    // Shadow Caster: Quad covering center of screen at Z = 0.3
    let caster_vertices = [
        XRayVertex { position: [-0.3, -0.3, 0.3], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [-0.3,  0.3, 0.3], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [ 0.3,  0.3, 0.3], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [ 0.3, -0.3, 0.3], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0], tangent: [1.0, 0.0, 0.0] },
    ];
    let caster_handle = renderer.create_mesh(&caster_vertices, &QUAD_INDICES).expect("create caster failed");

    // Receiver Ground: Large Quad covering entire screen at Z = 0.8
    let receiver_vertices = [
        XRayVertex { position: [-1.0, -1.0, 0.8], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [-1.0,  1.0, 0.8], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [ 1.0,  1.0, 0.8], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0], tangent: [1.0, 0.0, 0.0] },
        XRayVertex { position: [ 1.0, -1.0, 0.8], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0], tangent: [1.0, 0.0, 0.0] },
    ];
    let receiver_handle = renderer.create_mesh(&receiver_vertices, &QUAD_INDICES).expect("create receiver failed");

    let cam = identity_camera();
    renderer.update_camera(&cam).expect("update camera failed");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");

    // 1. Pass 1: Render caster into shadow depth map
    renderer.begin_shadow_pass().expect("begin shadow pass failed");
    renderer.draw_shadow_mesh(caster_handle, u32::MAX).expect("draw shadow mesh failed");
    renderer.end_shadow_pass().expect("end shadow pass failed");

    // 2. Pass 2: Main Color Pass rendering receiver quad with shadow testing
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer.draw_textured_mesh_with_shadow(receiver_handle, white_idx, shadow_idx)
        .expect("draw receiver with shadow failed");

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

    // Center pixel (under caster) must be shadowed
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    // Corner pixel (outside caster) must be unshadowed
    let corner_px = readback_rgba8(&readback, 64, 64, 5, 5);

    println!("Shadow Test: center (shadowed) = {:?}, corner (lit) = {:?}", center_px, corner_px);
    assert!(center_px[0] < corner_px[0], "Center pixel intensity ({}) must be less than corner pixel intensity ({}) due to shadow occlusion", center_px[0], corner_px[0]);
}

#[test]
fn test_particle_billboard_instancing_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("ParticleInstancingTest"),
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
            eprintln!("Skipping hardware particle test: D3D12 device unavailable");
            return;
        }
    };
    renderer.attach_window(window.0, 64, 64).expect("attach_window failed");

    // 2x2 sprite sheet atlas: frame 1 is Cyan [0, 255, 255, 255]
    let atlas_dds = create_2x2_atlas_rgba8_dds();
    let atlas_idx = renderer.load_texture_from_dds("particle_atlas.dds", &atlas_dds).expect("load atlas failed");

    // Use Honey's ParticleSpriteFrame parser struct
    let sprite_frame = ParticleSpriteFrame {
        tex_size: [0.5, 0.5],
        reserved: [0.0, 0.0],
        frame_dim_x: 2,
        frame_count: 4,
        speed: 12.0,
    };
    // Calculate UV coordinates for Frame index 1 (Cyan frame)
    let (uv_min, uv_max) = sprite_frame.calculate_tc(1);

    let cam = identity_camera();
    renderer.update_camera(&cam).expect("update camera failed");

    // Create 1 particle billboard at center of screen
    let instance = GpuParticleInstance {
        position: [0.0, 0.0, 0.5],
        rotation: 0.0,
        size: [1.2, 1.2],
        uv_min,
        uv_max,
        pad0: [0.0, 0.0],
        color: [1.0, 1.0, 1.0, 1.0],
        texture_index: atlas_idx.0,
        blend_mode: 0, // Alpha blend
        pad1: [0, 0],
    };

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    renderer.begin_frame().expect("begin_frame failed");
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    // Render procedurally generated particle billboard quad
    renderer.draw_particles(&[instance], false).expect("draw_particles failed");

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

    // Center pixel must match Cyan [0, 255, 255, 255] from Frame 1
    let center_px = readback_rgba8(&readback, 64, 64, 32, 32);
    println!("Particle Instancing Test center pixel: {:?}", center_px);
    assert!(center_px[0] <= 5, "Red must be near 0 for Cyan, got {}", center_px[0]);
    assert!(center_px[1] >= 250, "Green must be near 255 for Cyan, got {}", center_px[1]);
    assert!(center_px[2] >= 250, "Blue must be near 255 for Cyan, got {}", center_px[2]);
    assert_eq!(center_px[3], 255, "Alpha must be 255");
}
