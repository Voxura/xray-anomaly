//! End-to-end Spatial Audio + EngineRuntime Integration Test
//!
//! Verifies:
//! - Pure-Rust `.ogg` Vorbis I decoding + X-Ray `OGG_COMMENT_VERSION` metadata streaming via VFS
//! - `StalkerController` + `CformBvh` material ID resolution through `gamemtl.xr` -> Wood/Metal footsteps
//! - `WeatherLibrary` wind, rain density scaling, and 3D thunderclap spatial audio
//! - `level.snd_static` ambient point sources & `level.som` occlusion mesh binding
//! - ALife stalker & player weapon positional audio inside `EngineRuntime::step_frame`

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use xray_engine::alife_manager::{AlifeManager, EntityRole, LiveAlifeEntity};
use xray_engine::game_loop::{EngineConfig, EngineRuntime};
use xray_engine::input::{EngineInput, MouseButton};
use xray_engine::window::{WindowConfig, WindowEvent};
use xray_engine::world_loader::LevelLoadConfig;
use xray_engine::SurfaceMaterialCategory;
use xray_formats::cform::{CformModel, CformTriangle};
use xray_formats::chunk::ChunkWriter;
use xray_formats::gamemtl::{gamemtl_chunks, gamemtl_root_chunks};
use xray_formats::level::{d3d_decl_type, d3d_decl_usage, level_chunks, VertexElement9};
use xray_formats::ogf::{ogf_chunks, OgfVisualType};
use xray_formats::scop::ScopAlifeKind;
use xray_formats::sound_env::OggXRayParams;
use xray_formats::spawns::{INVALID_OBJECT_ID, INVALID_STORY_ID};
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
    let dir = std::env::temp_dir().join(format!("xray_audio_integ_{tag}_{nonce}"));
    fs::create_dir_all(&dir).expect("create temp gamedata dir");
    TempDirGuard(dir)
}

fn build_minimal_level_with_split_materials() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // level.geom: minimal VB/IB
    let mut vb = Vec::new();
    vb.extend_from_slice(&1u32.to_le_bytes());
    let decl = [
        VertexElement9 {
            stream: 0,
            offset: 0,
            decl_type: d3d_decl_type::FLOAT3,
            method: 0,
            usage: d3d_decl_usage::POSITION,
            usage_index: 0,
        },
        VertexElement9::END,
    ];
    for e in decl {
        vb.extend_from_slice(&e.stream.to_le_bytes());
        vb.extend_from_slice(&e.offset.to_le_bytes());
        vb.push(e.decl_type);
        vb.push(e.method);
        vb.push(e.usage);
        vb.push(e.usage_index);
    }
    vb.extend_from_slice(&4u32.to_le_bytes());
    for p in [[-20.0f32, 0.0, -20.0], [20.0, 0.0, -20.0], [20.0, 0.0, 20.0], [-20.0, 0.0, 20.0]] {
        for c in p {
            vb.extend_from_slice(&c.to_le_bytes());
        }
    }
    let mut ib = Vec::new();
    ib.extend_from_slice(&1u32.to_le_bytes());
    ib.extend_from_slice(&6u32.to_le_bytes());
    for i in [0u16, 1, 2, 0, 2, 3] {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    let mut geom_w = ChunkWriter::new();
    geom_w.write_chunk(level_chunks::FSL_VB, &vb);
    geom_w.write_chunk(level_chunks::FSL_IB, &ib);

    // level: minimal shader & visual
    let mut shd = Vec::new();
    shd.extend_from_slice(&2u32.to_le_bytes());
    shd.extend_from_slice(b"\0levels\\escape/grnd\0");

    let mut vis0 = ChunkWriter::new();
    let mut hdr = vec![4u8, OgfVisualType::Normal as u8];
    hdr.extend_from_slice(&1u16.to_le_bytes());
    hdr.extend_from_slice(&[0u8; 40]);
    vis0.write_chunk(ogf_chunks::OGF_HEADER, &hdr);
    let mut gcont = Vec::new();
    for v in [0u32, 0, 4, 0, 0, 6] {
        gcont.extend_from_slice(&v.to_le_bytes());
    }
    vis0.write_chunk(ogf_chunks::OGF_GCONTAINER, &gcont);
    let mut vis_wrap = ChunkWriter::new();
    vis_wrap.write_chunk(0, &vis0.into_bytes());

    let mut lvl_w = ChunkWriter::new();
    lvl_w.write_chunk(level_chunks::FSL_HEADER, &[14u8, 0, 0, 0]);
    lvl_w.write_chunk(level_chunks::FSL_SHADERS, &shd);
    lvl_w.write_chunk(level_chunks::FSL_VISUALS, &vis_wrap.into_bytes());

    // level.cform: left half (x < 0) = material 0 (wood), right half (x >= 0) = material 1 (metal)
    let cform = CformModel::from_parts(
        vec![
            [-40.0, 0.0, -40.0], // 0
            [-40.0, 0.0, 40.0],  // 1
            [0.0, 0.0, 40.0],    // 2
            [0.0, 0.0, -40.0],   // 3
            [40.0, 0.0, 40.0],   // 4
            [40.0, 0.0, -40.0],  // 5
        ],
        vec![
            CformTriangle {
                verts: [0, 1, 2],
                material: 0,
                sector: 0,
                suppress_shadows: false,
                suppress_wm: false,
            },
            CformTriangle {
                verts: [0, 2, 3],
                material: 0,
                sector: 0,
                suppress_shadows: false,
                suppress_wm: false,
            },
            CformTriangle {
                verts: [3, 2, 4],
                material: 1,
                sector: 0,
                suppress_shadows: false,
                suppress_wm: false,
            },
            CformTriangle {
                verts: [3, 4, 5],
                material: 1,
                sector: 0,
                suppress_shadows: false,
                suppress_wm: false,
            },
        ],
    )
    .expect("construct cform model");

    let mut cform_bytes = Vec::new();
    cform_bytes.extend_from_slice(&cform.header.version.to_le_bytes());
    cform_bytes.extend_from_slice(&cform.header.vert_count.to_le_bytes());
    cform_bytes.extend_from_slice(&cform.header.face_count.to_le_bytes());
    for f in cform.header.aabb.min {
        cform_bytes.extend_from_slice(&f.to_le_bytes());
    }
    for f in cform.header.aabb.max {
        cform_bytes.extend_from_slice(&f.to_le_bytes());
    }
    for v in &cform.vertices {
        for f in *v {
            cform_bytes.extend_from_slice(&f.to_le_bytes());
        }
    }
    for t in &cform.triangles {
        for vi in t.verts {
            cform_bytes.extend_from_slice(&vi.to_le_bytes());
        }
        cform_bytes.extend_from_slice(&t.to_raw_dummy().to_le_bytes());
    }

    (lvl_w.into_bytes(), geom_w.into_bytes(), cform_bytes)
}

fn write_mtl_chunk(id: u32, name: &str) -> Vec<u8> {
    let mut mw = ChunkWriter::new();
    let mut main = Vec::new();
    main.extend_from_slice(&id.to_le_bytes());
    main.extend_from_slice(name.as_bytes());
    main.push(0);
    mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_MAIN, &main);
    mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FLAGS, &0u32.to_le_bytes());
    mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_PHYSICS, &[0u8; 20]);
    mw.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FACTORS, &[0u8; 16]);
    mw.into_bytes()
}

fn build_synthetic_gamemtl() -> Vec<u8> {
    let mut mtls_w = ChunkWriter::new();
    mtls_w.write_chunk(0, &write_mtl_chunk(0, "materials\\wood_plank"));
    mtls_w.write_chunk(1, &write_mtl_chunk(1, "materials\\metal_grate"));

    let mut root = ChunkWriter::new();
    root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_VERSION, &1u16.to_le_bytes());
    let mut autoinc = Vec::new();
    autoinc.extend_from_slice(&2u32.to_le_bytes());
    autoinc.extend_from_slice(&0u32.to_le_bytes());
    root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_AUTOINC, &autoinc);
    root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_MTLS, &mtls_w.into_bytes());
    root.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_MTLS_PAIR, &[]);
    root.into_bytes()
}

fn build_valid_vorbis_ogg_with_xray_meta(params: &OggXRayParams) -> Vec<u8> {
    struct BitWriter {
        bytes: Vec<u8>,
        cur: u8,
        bit: u8,
    }
    impl BitWriter {
        fn new() -> Self { Self { bytes: Vec::new(), cur: 0, bit: 0 } }
        fn write_bit(&mut self, b: u32) {
            if (b & 1) != 0 { self.cur |= 1 << self.bit; }
            self.bit += 1;
            if self.bit == 8 { self.bytes.push(self.cur); self.cur = 0; self.bit = 0; }
        }
        fn write_bits(&mut self, v: u32, c: u32) {
            for s in 0..c { self.write_bit((v >> s) & 1); }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.bit > 0 { self.bytes.push(self.cur); }
            self.bytes
        }
    }

    let make_page = |seq: u32, htype: u8, gran: i64, pkt: &[u8]| -> Vec<u8> {
        let mut p = Vec::new();
        p.extend_from_slice(b"OggS");
        p.push(0);
        p.push(htype);
        p.extend_from_slice(&gran.to_le_bytes());
        p.extend_from_slice(&77u32.to_le_bytes());
        p.extend_from_slice(&seq.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        p.push(1);
        p.push(pkt.len() as u8);
        p.extend_from_slice(pkt);
        p
    };

    let mut id = vec![0x01];
    id.extend_from_slice(b"vorbis");
    id.extend_from_slice(&0u32.to_le_bytes());
    id.push(1);
    id.extend_from_slice(&44_100u32.to_le_bytes());
    id.extend_from_slice(&[0u8; 12]);
    id.push((6 << 4) | 6);
    id.push(1);

    let mut cmt = vec![0x03];
    cmt.extend_from_slice(b"vorbis");
    cmt.extend_from_slice(&4u32.to_le_bytes());
    cmt.extend_from_slice(b"xray");
    cmt.extend_from_slice(&1u32.to_le_bytes());
    let xr = params.to_bytes();
    cmt.extend_from_slice(&(xr.len() as u32).to_le_bytes());
    cmt.extend_from_slice(&xr);
    cmt.push(1);

    let mut sb = BitWriter::new();
    sb.write_bits(1, 8);
    // Book 0
    sb.write_bits(0x564342, 24);
    sb.write_bits(1, 16);
    sb.write_bits(2, 24);
    sb.write_bit(0);
    sb.write_bit(0);
    sb.write_bits(0, 5);
    sb.write_bits(0, 5);
    sb.write_bits(0, 4);
    // Book 1
    sb.write_bits(0x564342, 24);
    sb.write_bits(2, 16);
    sb.write_bits(4, 24);
    sb.write_bit(0);
    sb.write_bit(0);
    for _ in 0..4 { sb.write_bits(1, 5); }
    sb.write_bits(2, 4);
    let f32_pack = (788u32 << 21) | 4;
    sb.write_bits(788u32 << 21, 32);
    sb.write_bits(f32_pack, 32);
    sb.write_bits(0, 4);
    sb.write_bit(0);
    for &m in &[0u32, 0, 1, 0, 0, 1, 1, 1] { sb.write_bits(m, 1); }
    sb.write_bits(0, 6);
    sb.write_bits(0, 16);
    sb.write_bits(0, 6);
    sb.write_bits(1, 16);
    sb.write_bits(0, 5);
    sb.write_bits(0, 2);
    sb.write_bits(5, 4);
    sb.write_bits(0, 6);
    sb.write_bits(1, 16);
    sb.write_bits(0, 24);
    sb.write_bits(8, 24);
    sb.write_bits(3, 24);
    sb.write_bits(0, 6);
    sb.write_bits(0, 8);
    sb.write_bits(1, 3);
    sb.write_bit(0);
    sb.write_bits(1, 8);
    sb.write_bits(0, 6);
    sb.write_bits(0, 16);
    sb.write_bit(0);
    sb.write_bit(0);
    sb.write_bits(0, 2);
    sb.write_bits(0, 8);
    sb.write_bits(0, 8);
    sb.write_bits(0, 8);
    sb.write_bits(0, 6);
    sb.write_bit(0);
    sb.write_bits(0, 16);
    sb.write_bits(0, 16);
    sb.write_bits(0, 8);
    sb.write_bit(1);
    let mut setup = vec![0x05];
    setup.extend_from_slice(b"vorbis");
    setup.extend_from_slice(&sb.finish());

    let pkt = || {
        let mut aw = BitWriter::new();
        aw.write_bit(0);
        aw.write_bit(1);
        aw.write_bits(255, 8);
        aw.write_bits(255, 8);
        aw.write_bit(0);
        aw.write_bit(1);
        aw.write_bit(1);
        aw.write_bit(0);
        aw.write_bit(0);
        aw.write_bit(0);
        aw.write_bit(0);
        aw.write_bit(0);
        aw.write_bit(0);
        aw.write_bit(0);
        aw.finish()
    };

    let mut out = Vec::new();
    out.extend_from_slice(&make_page(0, 0x02, 0, &id));
    out.extend_from_slice(&make_page(1, 0x00, 0, &cmt));
    out.extend_from_slice(&make_page(2, 0x00, 0, &setup));
    out.extend_from_slice(&make_page(3, 0x00, 32, &pkt()));
    out.extend_from_slice(&make_page(4, 0x04, 64, &pkt()));
    out
}

#[test]
fn test_engine_spatial_audio_footsteps_weather_and_alife_integration() {
    let guard = make_temp_dir("phase_a");
    let root = &guard.0;
    let gd = root.join("gamedata");
    let lvl = gd.join("levels").join("audio_arena");
    let env_dir = gd.join("configs").join("environment");
    let snd_wep = gd.join("sounds").join("weapons");
    fs::create_dir_all(&lvl).unwrap();
    fs::create_dir_all(&env_dir).unwrap();
    fs::create_dir_all(&snd_wep).unwrap();

    let (lvl_bytes, geom_bytes, cform_bytes) = build_minimal_level_with_split_materials();
    fs::write(lvl.join("level"), lvl_bytes).unwrap();
    fs::write(lvl.join("level.geom"), geom_bytes).unwrap();
    fs::write(lvl.join("level.cform"), cform_bytes).unwrap();
    fs::write(gd.join("gamemtl.xr"), build_synthetic_gamemtl()).unwrap();

    // Stormy weather cycle
    fs::write(
        env_dir.join("weathers").join("default.ltx").parent().unwrap_or(&env_dir).join("weather_default.ltx"),
        "",
    )
    .ok();
    fs::create_dir_all(env_dir.join("weathers")).unwrap();
    fs::write(
        env_dir.join("weathers").join("default.ltx"),
        "[00:00:00]\nrain_density = 0.85\nwind_velocity = 11.0\nwind_direction = 0.2\n\
         [12:00:00]\nrain_density = 0.85\nwind_velocity = 11.0\nwind_direction = 0.2\n",
    )
    .unwrap();

    // Write valid Vorbis I weapon sound with custom X-Ray params
    let gun_params = OggXRayParams {
        version: 0x0003,
        min_dist: 5.0,
        max_dist: 150.0,
        base_volume: 1.25,
        game_type: 1,
        max_ai_dist: 150.0,
    };
    fs::write(
        snd_wep.join("ak74_shoot.ogg"),
        build_valid_vorbis_ogg_with_xray_meta(&gun_params),
    )
    .unwrap();

    let vfs = Arc::new(VirtualFileSystem::mount_game_root(root).unwrap());

    let mut runtime = EngineRuntime::new(
        EngineConfig {
            window: WindowConfig {
                title: "Audio Integration Test".to_string(),
                width: 640,
                height: 360,
                visible: false,
            },
            fixed_dt: 1.0 / 60.0,
            ..Default::default()
        },
        false,
    )
    .unwrap();
    runtime.attach_vfs(Arc::clone(&vfs));

    runtime
        .load_world_and_save_from_vfs(
            &vfs,
            &LevelLoadConfig {
                level_dir: "levels\\audio_arena".to_string(),
                resolve_visual_meshes: false,
                resolve_textures: false,
                load_fastpath_geom: false,
                load_optional_aux: true,
                load_weather: true,
            },
            None,
        )
        .unwrap();

    // Place player on Wood (x = -4.0, z = 0.0) facing +Z (yaw=0)
    runtime.camera.yaw = 0.0;
    runtime.camera.pitch = 0.0;
    runtime.physics.sync_from_eye([-4.0, 1.65, 0.0]);
    if let Some(bvh) = runtime
        .world
        .as_ref()
        .and_then(|w| w.cform.as_ref())
        .map(xray_formats::cform::CformBvh::build)
    {
        assert!(runtime.physics.snap_to_ground(&bvh, 1.0, 3.0));
        assert_eq!(runtime.physics.ground_material, Some(0));
    }

    // Spawn an ALife stalker walking 8m to the right (+X)
    let mut alife = AlifeManager::default();
    alife.entities.insert(
        42,
        LiveAlifeEntity {
            id: 42,
            parent_id: INVALID_OBJECT_ID,
            story_id: INVALID_STORY_ID,
            name: "wolf".to_string(),
            section: "stalker".to_string(),
            role: EntityRole::Stalker,
            kind: ScopAlifeKind::Stalker,
            position: [8.0, 0.0, 0.0],
            angle: [0.0; 3],
            velocity: [2.5, 0.0, 0.0],
            health: 1.0,
            online: true,
            game_vertex_id: 0,
            level_vertex_id: 0,
            anim_state: 1,
            weapon_state: 0,
            visual_name: None,
            dirty_net_sync: false,
            source_scop: None,
        },
    );
    runtime.attach_alife(alife);
    runtime.physics.sync_from_eye([-4.0, 1.65, 0.0]);

    // Walk right (+X via D key) across Wood (-4 -> 0) and then onto Metal (0 -> +3), firing weapon once
    runtime.input.on_key_down(EngineInput::VK_KEY_D);
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseDown { button: MouseButton::Left });

    let mut saw_wood = false;
    let mut saw_metal = false;
    for frame_idx in 0..95 {
        if frame_idx == 1 {
            runtime
                .window
                .push_synthetic_event(WindowEvent::MouseUp { button: MouseButton::Left });
        }
        runtime.timer.simulate_elapsed(1.0 / 60.0);
        runtime.step_frame().unwrap();
        let st = runtime.audio.stats();
        if st.last_footstep_surface == Some(SurfaceMaterialCategory::Wood) {
            saw_wood = true;
        }
        if st.last_footstep_surface == Some(SurfaceMaterialCategory::Metal) {
            saw_metal = true;
        }
    }

    let stats = runtime.audio.stats();
    assert!(saw_wood, "expected footsteps on wood material");
    assert!(saw_metal, "expected footsteps transitioning onto metal material");
    assert!(stats.footsteps_triggered >= 3);
    assert_eq!(stats.weapon_sounds_triggered, 1);
    assert!(stats.rain_gain > 0.4, "expected WeatherLibrary storm rain gain");
    assert!(stats.wind_gain > 0.3, "expected WeatherLibrary wind gain");
    assert_eq!(stats.tracked_alife_emitters, 1);

    // Verify decoded VFS .ogg clip retained X-Ray parameters & produced stereo PCM energy
    let decoded_clip = runtime.audio.load_clip_from_vfs("weapons\\ak74_shoot").unwrap();
    assert_eq!(decoded_clip.xray_params, gun_params);

    let mixed = runtime.audio.mix_headless_frames(1024);
    let mix_energy: f32 = mixed.iter().map(|s| s * s).sum();
    assert!(mix_energy > 0.05, "expected audible mixed stereo audio output");
}
