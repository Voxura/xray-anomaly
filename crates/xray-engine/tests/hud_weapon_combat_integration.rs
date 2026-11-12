//! End-to-end EngineRuntime Combat & First-Person HUD Weapon Integration Test
//!
//! Verifies through `EngineRuntime::step_frame`:
//! - Separate HUD projection matrix (`hud_projection_matrix`) & VFS weapon/hand `.ogf` streaming
//! - Weapon state machine (`Idle`, `Aiming` ADS FOV zoom, `Firing` RPM cadence, `Reloading`, `Holstering`)
//! - Ballistic `CformBvh` penetration raycast (`Wood` -> `Concrete`), ALife target hit damage,
//!   `WallmarksManager` bullet-hole decals, muzzle/impact spark particles, and 3D `AudioManager`
//!   weapon fire + bullet impact sound generation.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use xray_engine::alife_manager::{AlifeManager, EntityRole, LiveAlifeEntity};
use xray_engine::game_loop::{EngineConfig, EngineRuntime};
use xray_engine::hud_weapon::{hud_projection_matrix, WeaponState};
use xray_engine::input::{EngineInput, MouseButton};
use xray_engine::window::{WindowConfig, WindowEvent};
use xray_engine::world_loader::{LevelLoadConfig, WorldLoader};
use xray_engine::SurfaceMaterialCategory;
use xray_formats::cform::{CformModel, CformTriangle};
use xray_formats::chunk::ChunkWriter;
use xray_formats::gamemtl::{gamemtl_chunks, gamemtl_root_chunks, GameMaterial};
use xray_formats::level::{d3d_decl_type, d3d_decl_usage, level_chunks, VertexElement9};
use xray_formats::ogf::{ogf_chunks, skel_vert_type, OgfVisualType, VertBoned1W};
use xray_formats::scop::ScopAlifeKind;
use xray_vfs::VirtualFileSystem;

struct TempDirGuard(PathBuf);
impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn make_temp_dir(tag: &str) -> TempDirGuard {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("xray_combat_{tag}_{nonce}"));
    fs::create_dir_all(&dir).expect("create temp test gamedata");
    TempDirGuard(dir)
}

fn write_decl_elem(out: &mut Vec<u8>, elem: VertexElement9) {
    out.extend_from_slice(&elem.stream.to_le_bytes());
    out.extend_from_slice(&elem.offset.to_le_bytes());
    out.push(elem.decl_type);
    out.push(elem.method);
    out.push(elem.usage);
    out.push(elem.usage_index);
}

fn write_minimal_level_and_geom(level_dir: &std::path::Path) {
    let mut vb_stream = Vec::new();
    vb_stream.extend_from_slice(&1u32.to_le_bytes());
    write_decl_elem(
        &mut vb_stream,
        VertexElement9 {
            stream: 0,
            offset: 0,
            decl_type: d3d_decl_type::FLOAT3,
            method: 0,
            usage: d3d_decl_usage::POSITION,
            usage_index: 0,
        },
    );
    write_decl_elem(&mut vb_stream, VertexElement9::END);
    vb_stream.extend_from_slice(&3u32.to_le_bytes());
    for p in [[-5.0f32, 0.0, -5.0], [5.0, 0.0, -5.0], [0.0, 0.0, 5.0]] {
        for c in p {
            vb_stream.extend_from_slice(&c.to_le_bytes());
        }
    }

    let mut ib_stream = Vec::new();
    ib_stream.extend_from_slice(&1u32.to_le_bytes());
    ib_stream.extend_from_slice(&3u32.to_le_bytes());
    for i in [0u16, 1, 2] {
        ib_stream.extend_from_slice(&i.to_le_bytes());
    }

    let mut geom_w = ChunkWriter::new();
    geom_w.write_chunk(level_chunks::FSL_VB, &vb_stream);
    geom_w.write_chunk(level_chunks::FSL_IB, &ib_stream);
    fs::write(level_dir.join("level.geom"), geom_w.into_bytes()).unwrap();

    let mut shd = Vec::new();
    shd.extend_from_slice(&1u32.to_le_bytes());
    shd.extend_from_slice(b"\0");
    let mut lvl_w = ChunkWriter::new();
    let hdr = vec![14u8, 0, 0, 0, 0, 0, 0, 0];
    lvl_w.write_chunk(level_chunks::FSL_HEADER, &hdr);
    lvl_w.write_chunk(level_chunks::FSL_SHADERS, &shd);
    lvl_w.write_chunk(level_chunks::FSL_VISUALS, &ChunkWriter::new().into_bytes());
    fs::write(level_dir.join("level"), lvl_w.into_bytes()).unwrap();
}

fn write_gamemtl(path: &std::path::Path) {
    let wood = GameMaterial {
        id: 0,
        name: "materials\\wood_plank".to_string(),
        shoot_factor: 0.42,
        ..Default::default()
    };

    let concrete = GameMaterial {
        id: 1,
        name: "materials\\concrete".to_string(),
        shoot_factor: 1.40,
        ..Default::default()
    };

    let mut mtls_chunk = ChunkWriter::new();
    for (i, m) in [wood, concrete].iter().enumerate() {
        let mut mw = ChunkWriter::new();
        let mut main = Vec::new();
        main.extend_from_slice(&m.id.to_le_bytes());
        main.extend_from_slice(m.name.as_bytes());
        main.push(0);
        mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_MAIN, &main);
        mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FLAGS, &m.flags.to_le_bytes());

        let mut phys = Vec::new();
        for v in [
            m.ph_friction,
            m.ph_damping,
            m.ph_spring,
            m.ph_bounce_start_velocity,
            m.ph_bouncing,
        ] {
            phys.extend_from_slice(&v.to_le_bytes());
        }
        mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_PHYSICS, &phys);

        let mut fac = Vec::new();
        for v in [
            m.shoot_factor,
            m.bounce_damage_factor,
            m.vis_transparency_factor,
            m.snd_occlusion_factor,
        ] {
            fac.extend_from_slice(&v.to_le_bytes());
        }
        mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FACTORS, &fac);
        mtls_chunk.write_chunk(i as u32, &mw.into_bytes());
    }

    let mut root = ChunkWriter::new();
    root.write_chunk(
        gamemtl_root_chunks::GAMEMTLS_CHUNK_VERSION,
        &1u16.to_le_bytes(),
    );
    let mut counts = Vec::new();
    counts.extend_from_slice(&2u32.to_le_bytes());
    counts.extend_from_slice(&0u32.to_le_bytes());
    root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_AUTOINC, &counts);
    root.write_chunk(
        gamemtl_root_chunks::GAMEMTLS_CHUNK_MTLS,
        &mtls_chunk.into_bytes(),
    );
    fs::write(path, root.into_bytes()).unwrap();
}

fn write_combat_cform(path: &std::path::Path) {
    // Floor slab + Wood cover wall (z=5.0m, mat 0) + Concrete backdrop wall (z=11.0m, mat 1)
    let verts = vec![
        // Floor [-20..20, y=0]
        [-20.0f32, 0.0, -20.0],
        [20.0, 0.0, -20.0],
        [20.0, 0.0, 20.0],
        [-20.0, 0.0, 20.0],
        // Wood cover at z=5.0
        [-3.0, 0.0, 5.0],
        [3.0, 0.0, 5.0],
        [3.0, 4.0, 5.0],
        [-3.0, 4.0, 5.0],
        // Concrete wall at z=11.0
        [-5.0, 0.0, 11.0],
        [5.0, 0.0, 11.0],
        [5.0, 5.0, 11.0],
        [-5.0, 5.0, 11.0],
    ];
    let tris = vec![
        CformTriangle::from_raw([0, 1, 2], 1),
        CformTriangle::from_raw([0, 2, 3], 1),
        CformTriangle::from_raw([4, 5, 6], 0), // wood
        CformTriangle::from_raw([4, 6, 7], 0), // wood
        CformTriangle::from_raw([8, 9, 10], 1), // concrete
        CformTriangle::from_raw([8, 10, 11], 1), // concrete
    ];
    let cform = CformModel::from_parts(verts, tris).expect("cform");
    let mut buf = Vec::new();
    buf.extend_from_slice(&cform.header.version.to_le_bytes());
    buf.extend_from_slice(&(cform.vertices.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(cform.triangles.len() as u32).to_le_bytes());
    for f in cform.header.aabb.min.into_iter().chain(cform.header.aabb.max) {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    for v in &cform.vertices {
        for c in v {
            buf.extend_from_slice(&c.to_le_bytes());
        }
    }
    for t in &cform.triangles {
        for idx in t.verts {
            buf.extend_from_slice(&idx.to_le_bytes());
        }
        buf.extend_from_slice(&t.to_raw_dummy().to_le_bytes());
    }
    fs::write(path, buf).unwrap();
}

fn build_skeletal_hud_ogf() -> Vec<u8> {
    let mut ogf = ChunkWriter::new();
    let mut hdr = vec![4u8, OgfVisualType::SkeletonAnim as u8];
    hdr.extend_from_slice(&0u16.to_le_bytes());
    for f in [-0.15f32, -0.15, -0.15, 0.15, 0.15, 0.35, 0.0, 0.0, 0.0, 0.4] {
        hdr.extend_from_slice(&f.to_le_bytes());
    }
    ogf.write_chunk(ogf_chunks::OGF_HEADER, &hdr);

    let mut vb = Vec::new();
    vb.extend_from_slice(&skel_vert_type::FVF_1L.to_le_bytes());
    vb.extend_from_slice(&3u32.to_le_bytes());
    for p in [[0.0f32, 0.0, 0.0], [0.05, 0.0, 0.1], [0.0, 0.05, 0.2]] {
        let v = VertBoned1W {
            position: p,
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            binormal: [0.0, 0.0, 1.0],
            u: 0.0,
            v: 0.0,
            matrix: 0,
        };
        let raw = unsafe {
            std::slice::from_raw_parts(
                (&v as *const VertBoned1W).cast::<u8>(),
                std::mem::size_of::<VertBoned1W>(),
            )
        };
        vb.extend_from_slice(raw);
    }
    ogf.write_chunk(ogf_chunks::OGF_VERTICES, &vb);

    let mut ib = Vec::new();
    ib.extend_from_slice(&3u32.to_le_bytes());
    for i in [0u16, 1, 2] {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    ogf.write_chunk(ogf_chunks::OGF_INDICES, &ib);

    let mut bones = Vec::new();
    bones.extend_from_slice(&1u32.to_le_bytes());
    bones.extend_from_slice(b"root\0\0");
    bones.extend_from_slice(&[0u8; 60]);
    ogf.write_chunk(ogf_chunks::OGF_S_BONE_NAMES, &bones);
    ogf.into_bytes()
}

#[test]
fn test_engine_hud_weapon_ads_ballistics_wallmarks_and_audio_integration() {
    let temp = make_temp_dir("engine_combat");
    let root = &temp.0;
    let lvl_dir = root.join("levels\\combat_range");
    let wpn_dir = root.join("meshes\\dynamics\\weapons\\wpn_ak74");
    let hand_dir = root.join("meshes\\actors\\stalker_hero");
    fs::create_dir_all(&lvl_dir).unwrap();
    fs::create_dir_all(&wpn_dir).unwrap();
    fs::create_dir_all(&hand_dir).unwrap();

    write_minimal_level_and_geom(&lvl_dir);
    write_gamemtl(&root.join("gamemtl.xr"));
    write_combat_cform(&lvl_dir.join("level.cform"));

    let ogf_bytes = build_skeletal_hud_ogf();
    fs::write(wpn_dir.join("wpn_ak74_hud.ogf"), &ogf_bytes).unwrap();
    fs::write(hand_dir.join("hand_stalker_hud.ogf"), &ogf_bytes).unwrap();

    let mut vfs = VirtualFileSystem::new();
    vfs.mount_loose_dir(root);
    let vfs = Arc::new(vfs);

    let load_cfg = LevelLoadConfig {
        load_optional_aux: true,
        ..LevelLoadConfig::minimal("levels\\combat_range")
    };
    let world = WorldLoader::load_from_vfs(&vfs, &load_cfg).expect("load world");

    let mut alife = AlifeManager::default();
    alife.entities.insert(
        42,
        LiveAlifeEntity {
            id: 42,
            parent_id: 0xFFFF,
            story_id: u32::MAX,
            name: "bandit_target".to_string(),
            section: "stalker".to_string(),
            role: EntityRole::Stalker,
            kind: ScopAlifeKind::Stalker,
            position: [0.0, 0.65, 8.0], // Behind wood cover (z=5), in front of concrete (z=11)
            angle: [0.0, 0.0, 0.0],
            velocity: [0.0; 3],
            health: 1.0,
            online: true,
            game_vertex_id: 0,
            level_vertex_id: 0,
            anim_state: 0,
            weapon_state: 0,
            visual_name: None,
            dirty_net_sync: false,
            source_scop: None,
        },
    );

    let mut cfg = EngineConfig {
        window: WindowConfig {
            title: "HUD Combat Integration".to_string(),
            width: 1280,
            height: 720,
            visible: false,
        },
        ..Default::default()
    };
    cfg.fixed_dt = 1.0 / 60.0;

    let mut runtime = EngineRuntime::new(cfg, false).expect("init headless runtime");
    runtime.attach_vfs(vfs);
    runtime.attach_world(world);
    runtime.attach_alife(alife);

    // Position player at [0, 1.65, 0] looking straight along +Z toward walls + bandit target
    runtime.physics.sync_from_eye([0.0, 1.65, 0.0]);
    runtime.camera.position = [0.0, 1.65, 0.0];
    runtime.camera.yaw = 0.0;
    runtime.camera.pitch = 0.0;

    // 1. Hold RMB for 15 frames -> ADS FOV zoom + crosshair auto-hide
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseDown {
            button: MouseButton::Right,
        });
    for _ in 0..15 {
        runtime.timer.simulate_elapsed(1.0 / 60.0);
        runtime.step_frame().expect("step ADS");
    }
    assert_eq!(runtime.hud_weapon.state, WeaponState::Aiming);
    assert!(!runtime.hud.show_crosshair);
    assert!(runtime.camera.fov_deg < 60.0, "ADS should tighten camera FOV");
    let hud_proj = hud_projection_matrix(
        runtime.hud_weapon.current_hud_fov_factor(),
        runtime.camera.aspect,
        150.0,
    );
    assert_ne!(hud_proj, runtime.camera.projection_matrix());

    // Verify HUD OGF models streamed from VFS and skinned properly
    let hud_parts = runtime
        .hud_weapon
        .collect_hud_draw_parts(&runtime.camera, runtime.vfs.as_deref());
    assert_eq!(hud_parts.len(), 2);

    // 2. Fire LMB -> penetrates wood wall (z=5), hits bandit at z=8, stops at concrete wall (z=11)
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseDown {
            button: MouseButton::Left,
        });
    runtime.timer.simulate_elapsed(1.0 / 60.0);
    runtime.step_frame().expect("step fire shot");
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseUp {
            button: MouseButton::Left,
        });
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseUp {
            button: MouseButton::Right,
        });

    let report = runtime
        .hud_weapon
        .last_shot_report
        .clone()
        .expect("shot report recorded");
    assert_eq!(report.slot, 3);
    assert!(
        report
            .impacts
            .iter()
            .any(|i| i.surface == SurfaceMaterialCategory::Wood),
        "must impact wood wall"
    );
    assert!(
        report
            .impacts
            .iter()
            .any(|i| i.surface == SurfaceMaterialCategory::Concrete),
        "must penetrate wood and hit concrete wall behind it"
    );
    assert!(
        report.impacts.iter().any(|i| i.hit_entity_id == Some(42)),
        "must hit bandit entity 42 behind wood wall"
    );
    let target_health = runtime.alife.as_ref().unwrap().entities[&42].health;
    assert!(target_health < 1.0, "bandit health must drop on hit");
    assert_eq!(runtime.hud_weapon.wallmarks.len(), 2);
    assert!(runtime.hud_weapon.sparks_spawned >= 8);

    let astats = runtime.audio.stats();
    assert_eq!(astats.weapon_sounds_triggered, 1);
    assert_eq!(astats.bullet_impacts_triggered, 2);

    // 3. Reload ('R') -> clip restores to 30, then swap to Slot 2 (`PM` pistol)
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyDown {
            vk: EngineInput::VK_KEY_R,
        });
    runtime.timer.simulate_elapsed(1.0 / 60.0);
    runtime.step_frame().expect("step reload start");
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyUp {
            vk: EngineInput::VK_KEY_R,
        });
    assert_eq!(runtime.hud_weapon.state, WeaponState::Reloading);

    for _ in 0..140 {
        runtime.timer.simulate_elapsed(1.0 / 60.0);
        runtime.step_frame().expect("step reload");
    }
    assert_eq!(runtime.hud_weapon.active_weapon().ammo_in_clip, 30);

    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyDown {
            vk: EngineInput::VK_KEY_2,
        });
    runtime.timer.simulate_elapsed(1.0 / 60.0);
    runtime.step_frame().expect("step holster");
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyUp {
            vk: EngineInput::VK_KEY_2,
        });
    for _ in 0..45 {
        runtime.timer.simulate_elapsed(1.0 / 60.0);
        runtime.step_frame().expect("step swap");
    }
    assert_eq!(runtime.hud_weapon.active_slot, 2);
    assert_eq!(runtime.hud_weapon.state, WeaponState::Idle);
}
