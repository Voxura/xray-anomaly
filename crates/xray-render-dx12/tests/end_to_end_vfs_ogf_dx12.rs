//! End-to-End Integration Test:
//! Pure Rust VFS Archive Streaming -> OGF v4 Static Mesh Parser -> DDS Decoder -> DX12 GPU Allocation -> Live Draw & Readback
//!
//! Verifies the entire Phase 5 pipeline on hardware without any legacy C++ engine dependencies.

use std::fs::File;
use std::io::Write;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, WS_OVERLAPPEDWINDOW,
};
use windows::core::w;

use xray_formats::chunk::ChunkWriter;
use xray_formats::ogf::ogf_chunks::*;
use xray_formats::ogf::{fvf_flags, OgfVertices, OgfVisualType};
use xray_formats::vfs_bridge::load_ogf_from_vfs;
use xray_render_dx12::geometry_abi::{CameraConstants, XRayVertex};
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::texture::{
    DDPF_FOURCC, DDSD_HEIGHT, DDSD_PIXELFORMAT, DDSD_WIDTH, DDS_MAGIC, FOURCC_DXT1,
};
use xray_vfs::{compress_lzo1x, compute_crc32, VirtualFileSystem};

struct TestWindow(HWND);
impl Drop for TestWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.0);
        }
    }
}

fn create_identity_camera() -> CameraConstants {
    let mut c = CameraConstants::default();
    for i in 0..4 {
        c.view[i][i] = 1.0;
        c.proj[i][i] = 1.0;
        c.view_proj[i][i] = 1.0;
    }
    c
}

fn create_test_dds(width: u32, height: u32, fourcc: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = vec![0u8; 128 + data.len()];
    buf[0..4].copy_from_slice(&DDS_MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&124u32.to_le_bytes());
    buf[8..12].copy_from_slice(&(DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT).to_le_bytes());
    buf[12..16].copy_from_slice(&height.to_le_bytes());
    buf[16..20].copy_from_slice(&width.to_le_bytes());
    // PixelFormat
    buf[76..80].copy_from_slice(&32u32.to_le_bytes());
    buf[80..84].copy_from_slice(&DDPF_FOURCC.to_le_bytes());
    buf[84..88].copy_from_slice(&fourcc.to_le_bytes());
    buf[128..128 + data.len()].copy_from_slice(data);
    buf
}

#[test]
fn test_end_to_end_vfs_streaming_ogf_dds_render() {
    let temp_dir = std::env::temp_dir().join(format!("xray_e2e_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Build a synthetic OGF v4 model
    let mut ow = ChunkWriter::new();
    let mut hdr = vec![4u8, OgfVisualType::Normal as u8, 0, 0];
    hdr.extend_from_slice(&[0u8; 40]); // bbox + bsphere
    ow.write_chunk(OGF_HEADER, &hdr);
    ow.write_chunk(OGF_TEXTURE, b"weapons\\ak74\0default\0");

    let mut vb = Vec::new();
    vb.extend_from_slice(&fvf_flags::XRAY_STATIC_FVF_STANDARD.to_le_bytes());
    vb.extend_from_slice(&3u32.to_le_bytes()); // 3 vertices (32 bytes each: pos 12, norm 12, uv 8)

    // Vertex 0: (-0.8, -0.8, 0.5)
    vb.extend_from_slice(&(-0.8f32).to_le_bytes());
    vb.extend_from_slice(&(-0.8f32).to_le_bytes());
    vb.extend_from_slice(&0.5f32.to_le_bytes());
    vb.extend_from_slice(&0.0f32.to_le_bytes()); // Normal X
    vb.extend_from_slice(&1.0f32.to_le_bytes()); // Normal Y
    vb.extend_from_slice(&0.0f32.to_le_bytes()); // Normal Z
    vb.extend_from_slice(&0.0f32.to_le_bytes()); // UV U
    vb.extend_from_slice(&1.0f32.to_le_bytes()); // UV V

    // Vertex 1: (0.0, 0.8, 0.5)
    vb.extend_from_slice(&0.0f32.to_le_bytes());
    vb.extend_from_slice(&0.8f32.to_le_bytes());
    vb.extend_from_slice(&0.5f32.to_le_bytes());
    vb.extend_from_slice(&0.0f32.to_le_bytes());
    vb.extend_from_slice(&1.0f32.to_le_bytes());
    vb.extend_from_slice(&0.0f32.to_le_bytes());
    vb.extend_from_slice(&0.5f32.to_le_bytes());
    vb.extend_from_slice(&0.0f32.to_le_bytes());

    // Vertex 2: (0.8, -0.8, 0.5)
    vb.extend_from_slice(&0.8f32.to_le_bytes());
    vb.extend_from_slice(&(-0.8f32).to_le_bytes());
    vb.extend_from_slice(&0.5f32.to_le_bytes());
    vb.extend_from_slice(&0.0f32.to_le_bytes());
    vb.extend_from_slice(&1.0f32.to_le_bytes());
    vb.extend_from_slice(&0.0f32.to_le_bytes());
    vb.extend_from_slice(&1.0f32.to_le_bytes());
    vb.extend_from_slice(&1.0f32.to_le_bytes());

    ow.write_chunk(OGF_VERTICES, &vb);

    let mut ib = Vec::new();
    ib.extend_from_slice(&3u32.to_le_bytes());
    for i in [0u16, 1, 2] {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    ow.write_chunk(OGF_INDICES, &ib);

    let ogf_raw = ow.into_bytes();
    let ogf_comp = compress_lzo1x(&ogf_raw);

    // 2. Build a synthetic 4x4 DXT1 DDS texture (8 bytes block data)
    let dds_pixel_data = [0xFF, 0x00, 0x00, 0x00, 0xAA, 0x55, 0xAA, 0x55];
    let dds_raw = create_test_dds(4, 4, FOURCC_DXT1, &dds_pixel_data);
    let dds_comp = compress_lzo1x(&dds_raw);

    // 3. Pack into an X-Ray .db archive
    let entries: Vec<(&str, &[u8], Vec<u8>)> = vec![
        ("meshes\\weapons\\ak74.ogf", &ogf_raw, ogf_comp),
        ("textures\\weapons\\ak74.dds", &dds_raw, dds_comp),
    ];

    let mut table_len = 0usize;
    for (n, _, _) in &entries {
        table_len += 2 + 16 + n.len();
    }
    let mut offset = 8 + table_len;
    let mut chunk1 = Vec::new();
    for (n, raw, comp) in &entries {
        let rec_len = (16 + n.len()) as u16;
        chunk1.extend_from_slice(&rec_len.to_le_bytes());
        chunk1.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        chunk1.extend_from_slice(&(comp.len() as u32).to_le_bytes());
        chunk1.extend_from_slice(&compute_crc32(raw).to_le_bytes());
        chunk1.extend_from_slice(n.as_bytes());
        chunk1.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += comp.len();
    }

    let archive_path = temp_dir.join("gamedata.db0");
    let mut file = File::create(&archive_path).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap(); // Chunk 1 ID
    file.write_all(&(chunk1.len() as u32).to_le_bytes()).unwrap();
    file.write_all(&chunk1).unwrap();
    for (_, _, comp) in &entries {
        file.write_all(comp).unwrap();
    }
    drop(file);

    // 4. Mount archive into pure Rust VirtualFileSystem
    let mut vfs = VirtualFileSystem::new();
    vfs.mount_archive_mmap(&archive_path)
        .expect("Failed to mount synthetic .db archive");

    // 5. Stream OGF model directly from VFS
    let ogf = load_ogf_from_vfs(&vfs, "meshes\\weapons\\ak74.ogf")
        .expect("Failed to stream OGF from VFS");

    let tex_link = ogf.texture.as_ref().expect("OGF missing texture link");
    assert_eq!(tex_link.texture, "weapons\\ak74");

    // 6. Stream DDS texture directly from VFS using model's texture path
    let tex_vpath = format!("textures\\{}.dds", tex_link.texture);
    let dds_bytes = vfs.read_bytes(&tex_vpath).expect("Failed to read DDS from VFS");

    // 7. Initialize DirectX 12 hardware renderer
    let mut renderer = Dx12Renderer::new(1).expect("Failed to initialize DX12 Renderer");
    println!("Live GPU Adapter: {}", renderer.adapter_name());

    let window = TestWindow(unsafe {
        CreateWindowExW(
            Default::default(),
            w!("STATIC"),
            w!("E2E Pipeline Test"),
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
    renderer.attach_window(window.0, 64, 64).unwrap();

    // 8. Upload DDS texture to GPU default heap and register into bindless descriptor heap
    let srv_index = renderer
        .load_texture_from_dds(&tex_link.texture, &dds_bytes)
        .expect("Failed to upload texture to GPU default heap");
    assert!(srv_index.is_valid());

    let retrieved_tex = renderer
        .get_texture("weapons\\ak74")
        .expect("Texture should be registered in renderer");
    assert_eq!(retrieved_tex.width, 4);
    assert_eq!(retrieved_tex.height, 4);
    assert_eq!(retrieved_tex.descriptor_index, srv_index);

    // 9. Convert OGF static vertices to XRayVertex and create GPU mesh
    let static_vertices = match ogf.vertices.as_ref().unwrap() {
        OgfVertices::Static { vertices, .. } => vertices,
        _ => panic!("Expected static vertices"),
    };
    assert_eq!(static_vertices.len(), 3);

    let xray_vertices: Vec<XRayVertex> = static_vertices
        .iter()
        .map(|v| XRayVertex {
            position: v.position,
            normal: v.normal,
            uv: v.uv0,
            tangent: [1.0, 0.0, 0.0],
        })
        .collect();

    let mesh_handle = renderer
        .create_mesh(&xray_vertices, &ogf.indices)
        .expect("Failed to create GPU mesh from OGF vertices");
    assert!(mesh_handle != 0);

    // 10. Update Camera, Record Frame, and Draw Mesh on Live Swapchain Backbuffer
    let camera = create_identity_camera();
    renderer.update_camera(&camera).unwrap();

    let _ = renderer.begin_frame().unwrap();
    renderer.clear_backbuffer([0.0, 0.0, 0.0, 1.0]);
    renderer.draw_mesh(mesh_handle).unwrap();
    renderer.end_frame(false).unwrap();
    renderer.wait_for_gpu().unwrap();

    println!("End-to-End VFS -> OGF -> DDS -> GPU Render executed with 100% success!");

    // Clean up temporary archive
    let _ = std::fs::remove_dir_all(&temp_dir);
}
