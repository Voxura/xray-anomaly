use xray_engine::camera::FirstPersonCamera;
use xray_engine::render_bridge::load_vfs_texture;
use xray_engine::{LevelLoadConfig, WorldLoader};
use xray_render_dx12::renderer::Dx12Renderer;
use xray_vfs::VirtualFileSystem;

// Simulate column-major HLSL mul(M, v) from the actual uploaded bytes.
fn clip(m: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 4] {
    std::array::from_fn(|col| p[0] * m[0][col] + p[1] * m[1][col] + p[2] * m[2][col] + m[3][col])
}

#[test]
fn translated_rotated_camera_maps_near_far_and_screen_halves() {
    for yaw in [0.0, 0.7, 2.8] {
        let mut cam = FirstPersonCamera::new(1280, 720);
        cam.position = [-32.84, -6.42, -16.87];
        cam.yaw = yaw;
        cam.pitch = 0.23;
        let constants = cam.to_render_constants();
        let f = cam.forward();
        let r = cam.right();
        for (distance, depth) in [(cam.near_z, 0.0), (cam.far_z, 1.0)] {
            let p = std::array::from_fn(|i| cam.position[i] + f[i] * distance);
            let c = clip(constants.view_proj, p);
            assert!(c[3] > 0.0, "forward points must have positive clip W");
            assert!((c[2] / c[3] - depth).abs() < 0.001);
            assert!((c[0] / c[3]).abs() < 0.001);
            assert!((c[1] / c[3]).abs() < 0.001);
        }
        for side in [-1.0, 1.0] {
            let p = std::array::from_fn(|i| cam.position[i] + f[i] * 10.0 + r[i] * side);
            let c = clip(constants.view_proj, p);
            assert_eq!(c[0].signum(), side);
            assert!(c[3] > 0.0);
        }
        let behind = std::array::from_fn(|i| cam.position[i] - f[i]);
        assert!(clip(constants.view_proj, behind)[3] < 0.0);
    }
}

#[test]
fn texture_binding_alias_cache_and_failure_stages_on_dx12() {
    let mut renderer = Dx12Renderer::new(2).expect("DX12 device required");
    let root = std::env::temp_dir().join(format!("xray_visual_diag_{}", std::process::id()));
    let textures = root.join("textures");
    std::fs::create_dir_all(&textures).unwrap();
    let mut vfs = VirtualFileSystem::new();
    vfs.mount_loose_dir(&root);
    let missing = load_vfs_texture(&mut renderer, Some(&vfs), "missing").unwrap_err();
    assert!(missing.contains("VFS lookup"), "{missing}");
    std::fs::write(textures.join("diagnostic.dds"), b"invalid DDS").unwrap();
    let decode = load_vfs_texture(&mut renderer, Some(&vfs), "diagnostic").unwrap_err();
    assert!(decode.contains("DDS decode"), "{decode}");
    let mut dds = vec![0u8; 136]; // one 4x4 BC1 block
    for (offset, value) in [
        (0, 0x20534444u32),
        (4, 124),
        (8, 0x1006),
        (12, 4),
        (16, 4),
        (76, 32),
        (80, 4),
        (84, 0x31545844),
    ] {
        dds[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    std::fs::write(textures.join("diagnostic.dds"), &dds).unwrap();
    let index = load_vfs_texture(&mut renderer, Some(&vfs), "$game_textures$/DIAGNOSTIC.dds")
        .expect("alias lookup and GPU upload");
    assert_eq!(
        load_vfs_texture(&mut renderer, Some(&vfs), "textures/diagnostic.dds").unwrap(),
        index
    );
    assert_eq!(renderer.get_texture_by_index(index).unwrap().width, 4);
    renderer.wait_for_gpu().unwrap();
    std::fs::remove_file(textures.join("diagnostic.dds")).unwrap();
    std::fs::remove_dir(&textures).unwrap();
    std::fs::remove_dir(&root).unwrap();
}

#[test]
#[ignore = "requires XRAY_ANOMALY_ROOT and DX12; never silently skips"]
fn live_material_and_sky_vfs_decode_upload() {
    let root = std::env::var_os("XRAY_ANOMALY_ROOT").expect("set XRAY_ANOMALY_ROOT");
    let vfs = VirtualFileSystem::mount_game_root(root).expect("mount game archives");
    assert!(vfs.archive_count() > 0, "no archives mounted");
    let mut world =
        WorldLoader::load_from_vfs(&vfs, &LevelLoadConfig::default()).expect("load l01_escape");
    assert!(!world.visual_meshes.is_empty());
    let mut paths = std::collections::BTreeSet::new();
    let mut unlinked = 0;
    for mesh in &world.visual_meshes {
        if mesh.texture_links.is_empty() {
            unlinked += 1;
        }
        for link in &mesh.texture_links {
            paths.insert(link.diffuse_dds.clone());
        }
    }
    println!(
        "meshes={}, unlinked={}, unique diffuse={}",
        world.visual_meshes.len(),
        unlinked,
        paths.len()
    );
    assert!(!paths.is_empty(), "no material links resolved");
    let (pos, _) = world.default_actor_spawn().expect("actor spawn");
    let env = world
        .step_weather(0.0, pos, [0.0, 0.0, 1.0], 1000.0)
        .expect("weather environment");
    let (a, b) = env.sky_dds_vpaths();
    let a = a.expect("first sky texture");
    let b = b.expect("second sky texture");
    println!("spawn={pos:?}, sky=({a}, {b})");
    paths.extend([a, b]);
    let mut renderer = Dx12Renderer::new(2).expect("DX12 device");
    let mut failures = Vec::new();
    for path in paths {
        match load_vfs_texture(&mut renderer, Some(&vfs), &path) {
            Ok(index) => {
                assert!(renderer.get_texture_by_index(index).is_some());
            }
            Err(e) => failures.push(format!("{path}: {e}")),
        }
    }
    renderer.wait_for_gpu().expect("texture uploads complete");
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
