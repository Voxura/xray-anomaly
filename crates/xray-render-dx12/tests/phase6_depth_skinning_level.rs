//! Phase 6 Depth-Stencil, Skeletal Bone Palette, and Level Geometry Hardware Tests
//!
//! Validates:
//! 1. Direct3D 12 hardware D32_FLOAT Depth-Stencil buffer allocation, clearing,
//!    and LESS_EQUAL depth rejection on AMD Radeon R5 M330 hardware.
//! 2. Parsing and hardware rendering of `ResolvedLevelMesh` from `xray-formats`.
//! 3. Skeletal skinning `BonePalette` / `BoneMatrix4x3` evaluation and dynamic
//!    StructuredBuffer transform upload and vertex shader execution.

use std::mem::ManuallyDrop;
use std::ptr;
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, WS_OVERLAPPEDWINDOW,
};
use windows::core::w;
use xray_formats::level::{LevelShaderRef, ResolvedLevelMesh};
use xray_formats::ogf::{StaticVertex, XRAY_IDENTITY_MATRIX};
use xray_formats::skinning::{BoneMatrix4x3, SkinnedVertex};
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

fn create_readback_buffer(device: &ID3D12Device, width: u32, height: u32) -> ID3D12Resource {
    let row_pitch = ((width * 4 + 255) / 256) * 256;
    let buffer_size = (row_pitch * height) as u64;
    unsafe {
        let mut resource = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: buffer_size,
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
    }
}

#[test]
fn test_depth_buffer_hardware_readback_and_occlusion() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("Phase 6 Depth Occlusion"),
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

    let mut renderer = Dx12Renderer::new(1).expect("Discrete DX12 GPU required");
    renderer.attach_window(window.0, 64, 64).unwrap();
    renderer
        .update_camera(&identity_camera())
        .expect("Failed to update camera");

    // 1. Upload green texture (closer front mesh) and red texture (further back mesh)
    let green_dds = create_solid_color_rgba8_dds(16, 16, 0, 255, 0, 255);
    let red_dds = create_solid_color_rgba8_dds(16, 16, 255, 0, 0, 255);

    let green_srv = renderer
        .load_texture_from_dds("depth_front_green.dds", &green_dds)
        .expect("Failed to upload green DDS");
    let red_srv = renderer
        .load_texture_from_dds("depth_back_red.dds", &red_dds)
        .expect("Failed to upload red DDS");

    // Mesh A (Closer): Quad covering [-0.5, 0.5] at depth Z = 0.4
    let mesh_front_verts = [
        XRayVertex {
            position: [-0.5, -0.5, 0.4],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [-0.5, 0.5, 0.4],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.5, 0.5, 0.4],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.5, -0.5, 0.4],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let indices = [0u16, 1, 2, 0, 2, 3];
    let mesh_front = renderer
        .create_mesh(&mesh_front_verts, &indices)
        .expect("Failed to create front mesh");

    // Mesh B (Further back): Larger quad covering [-0.85, 0.85] at depth Z = 0.8
    let mesh_back_verts = [
        XRayVertex {
            position: [-0.85, -0.85, 0.8],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [-0.85, 0.85, 0.8],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.85, 0.85, 0.8],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        XRayVertex {
            position: [0.85, -0.85, 0.8],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];
    let mesh_back = renderer
        .create_mesh(&mesh_back_verts, &indices)
        .expect("Failed to create back mesh");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    // Frame: Draw front green quad (Z=0.4), then draw back red quad (Z=0.8).
    // Because depth testing is enabled with LESS_EQUAL:
    // - In the center region [-0.5, 0.5], Z=0.4 is in the depth buffer.
    // - The red quad at Z=0.8 MUST be occluded and rejected!
    // - In the perimeter region where Z was cleared to 1.0, the red quad (Z=0.8 <= 1.0) MUST render!
    renderer.begin_frame().unwrap();
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);

    renderer
        .draw_textured_mesh(mesh_front, green_srv)
        .expect("Draw front");
    renderer
        .draw_textured_mesh(mesh_back, red_srv)
        .expect("Draw back");

    // Copy to readback buffer
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

    // Verify sampled pixels
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
        // Center pixel at (32, 32) -> MUST be Green [0, 255, 0, 255] (occluded red)
        let center = bytes[32 * 256 + 32 * 4..32 * 256 + 32 * 4 + 4].to_vec();
        // Perimeter pixel at (54, 54) -> outside front mesh, inside back mesh -> MUST be Red [255, 0, 0, 255]
        let perimeter = bytes[54 * 256 + 54 * 4..54 * 256 + 54 * 4 + 4].to_vec();
        // Corner pixel at (0, 0) -> outside both -> Black [0, 0, 0, 255]
        let corner = bytes[..4].to_vec();

        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));

        println!("Center pixel (occluded): {:?}", center);
        println!("Perimeter pixel (visible): {:?}", perimeter);
        println!("Corner pixel (cleared): {:?}", corner);

        assert_eq!(
            center,
            [0, 255, 0, 255],
            "Center pixel must be green: red quad behind it was correctly occluded by D32_FLOAT depth buffer!"
        );
        assert_eq!(
            perimeter,
            [255, 0, 0, 255],
            "Perimeter pixel must be red: back quad is visible where front quad is absent!"
        );
        assert_eq!(corner, [0, 0, 0, 255], "Corner pixel must be black clear color");
    }
}

#[test]
fn test_resolved_level_mesh_hardware_draw_and_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("Phase 6 ResolvedLevelMesh"),
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

    let mut renderer = Dx12Renderer::new(1).expect("Discrete DX12 GPU required");
    renderer.attach_window(window.0, 64, 64).unwrap();
    renderer
        .update_camera(&identity_camera())
        .expect("Failed to update camera");

    // 1. Construct a ResolvedLevelMesh matching xray-formats level parser output
    let level_mesh = ResolvedLevelMesh {
        shader: Some(LevelShaderRef::parse("default/level_wood_floor".into())),
        vertices: vec![
            StaticVertex {
                position: [-0.7, -0.7, 0.5],
                normal: [0.0, 0.0, 1.0],
                diffuse: 0xFFFF_FFFF,
                specular: 0,
                uv0: [0.0, 1.0],
                uv1: [0.0, 0.0],
            },
            StaticVertex {
                position: [-0.7, 0.7, 0.5],
                normal: [0.0, 0.0, 1.0],
                diffuse: 0xFFFF_FFFF,
                specular: 0,
                uv0: [0.0, 0.0],
                uv1: [0.0, 0.0],
            },
            StaticVertex {
                position: [0.7, 0.7, 0.5],
                normal: [0.0, 0.0, 1.0],
                diffuse: 0xFFFF_FFFF,
                specular: 0,
                uv0: [1.0, 0.0],
                uv1: [0.0, 0.0],
            },
            StaticVertex {
                position: [0.7, -0.7, 0.5],
                normal: [0.0, 0.0, 1.0],
                diffuse: 0xFFFF_FFFF,
                specular: 0,
                uv0: [1.0, 1.0],
                uv1: [0.0, 0.0],
            },
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
    };

    // 2. Convert StaticVertex to XRayVertex
    let vertices: Vec<XRayVertex> = level_mesh
        .vertices
        .iter()
        .map(|v| XRayVertex {
            position: v.position,
            normal: v.normal,
            uv: v.uv0,
            tangent: [1.0, 0.0, 0.0],
        })
        .collect();

    let mesh_handle = renderer
        .create_mesh(&vertices, &level_mesh.indices)
        .expect("Failed to create level mesh");

    // Upload cyan texture [0, 255, 255, 255]
    let cyan_dds = create_solid_color_rgba8_dds(16, 16, 0, 255, 255, 255);
    let cyan_srv = renderer
        .load_texture_from_dds("level_cyan_tex.dds", &cyan_dds)
        .expect("Failed to upload cyan DDS");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    // Render frame
    renderer.begin_frame().unwrap();
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer
        .draw_textured_mesh(mesh_handle, cyan_srv)
        .expect("Draw textured level mesh");

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

    // Verify readback
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
        let corner = bytes[..4].to_vec();
        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));

        println!("ResolvedLevelMesh center pixel: {:?}", center);
        println!("ResolvedLevelMesh corner pixel: {:?}", corner);

        assert_eq!(center, [0, 255, 255, 255], "ResolvedLevelMesh center must be cyan");
        assert_eq!(corner, [0, 0, 0, 255], "ResolvedLevelMesh corner must be black");
    }
}

#[test]
fn test_skeletal_bone_palette_gpu_transform_hardware_readback() {
    let window = Window(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("Phase 6 Skeletal Skinning"),
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

    let mut renderer = Dx12Renderer::new(1).expect("Discrete DX12 GPU required");
    renderer.attach_window(window.0, 64, 64).unwrap();
    renderer
        .update_camera(&identity_camera())
        .expect("Failed to update camera");

    // 1. SkinnedVertex matching xray-formats skinning ABI (44 bytes)
    let skinned_verts = [
        SkinnedVertex {
            position: [-0.25, -0.25, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
        SkinnedVertex {
            position: [0.0, 0.25, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [0.5, 0.0],
            tangent: [1.0, 0.0, 0.0],
        },
        SkinnedVertex {
            position: [0.25, -0.25, 0.5],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
            tangent: [1.0, 0.0, 0.0],
        },
    ];

    let vertices: Vec<XRayVertex> = skinned_verts
        .iter()
        .map(|v| XRayVertex {
            position: v.position,
            normal: v.normal,
            uv: v.uv,
            tangent: v.tangent,
        })
        .collect();
    let indices = [0u16, 1, 2];

    let mesh_handle = renderer
        .create_mesh(&vertices, &indices)
        .expect("Failed to create skinned mesh");

    // 2. Bone palette: evaluate bone 0 = identity, bone 1 = translated by (+0.5, +0.5)
    let mut m_bone1 = XRAY_IDENTITY_MATRIX;
    m_bone1[3][0] = 0.5; // X translation
    m_bone1[3][1] = 0.5; // Y translation

    // Verify BoneMatrix4x3 packing matches transposed HLSL layout
    let bone4x3 = BoneMatrix4x3::from_matrix(&m_bone1);
    assert_eq!(bone4x3.rows[0][3], 0.5);
    assert_eq!(bone4x3.rows[1][3], 0.5);

    renderer
        .update_transforms(&[XRAY_IDENTITY_MATRIX, m_bone1])
        .expect("Failed to update bone transforms");

    // Upload yellow texture [255, 255, 0, 255]
    let yellow_dds = create_solid_color_rgba8_dds(16, 16, 255, 255, 0, 255);
    let yellow_srv = renderer
        .load_texture_from_dds("skeletal_yellow_tex.dds", &yellow_dds)
        .expect("Failed to upload yellow DDS");

    let readback = create_readback_buffer(&renderer.device_ctx.device, 64, 64);

    // Render frame with transform_index = 1 (shifting triangle from center to top-right quadrant)
    renderer.begin_frame().unwrap();
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer
        .draw_textured_mesh_with_transform(mesh_handle, yellow_srv, 1)
        .expect("Draw mesh with transform index 1");

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

    // Verify readback
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
        // Original center (32, 32) is now empty -> Black [0, 0, 0, 255]
        let center = bytes[32 * 256 + 32 * 4..32 * 256 + 32 * 4 + 4].to_vec();
        // Transformed center at (+0.5, +0.5) NDC -> D3D screen coordinates (X=48, Y=16)
        let transformed = bytes[16 * 256 + 48 * 4..16 * 256 + 48 * 4 + 4].to_vec();
        let corner = bytes[..4].to_vec();

        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));

        println!("Untransformed center pixel: {:?}", center);
        println!("Transformed bone position pixel: {:?}", transformed);
        println!("Corner pixel: {:?}", corner);

        assert_eq!(
            center,
            [0, 0, 0, 255],
            "Original center must be black: mesh was translated away by bone transform"
        );
        assert_eq!(
            transformed,
            [255, 255, 0, 255],
            "Transformed position must be yellow: bone matrix translated vertex in GPU StructuredBuffer!"
        );
        assert_eq!(corner, [0, 0, 0, 255], "Corner must remain black clear color");
    }
}
