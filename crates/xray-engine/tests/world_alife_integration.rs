//! Integration tests for `world_loader`, `alife_manager`, and `EngineRuntime` VFS/co-op wiring.

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use xray_engine::alife_manager::{AlifeManager, EntityRole};
use xray_engine::game_loop::{EngineConfig, EngineRuntime, NetworkSession};
use xray_engine::input::EngineInput;
use xray_engine::window::WindowConfig;
use xray_engine::world_loader::{LevelLoadConfig, WorldLoader};
use xray_formats::cform::{CformModel, CformTriangle};
use xray_formats::chunk::ChunkWriter;
use xray_formats::level::{d3d_decl_type, d3d_decl_usage, level_chunks, VertexElement9};
use xray_formats::ogf::{ogf_chunks, OgfVisualType};
use xray_formats::savegames::{SavedGameTask, SavedTaskScriptHelper};
use xray_formats::scoc::{ScocFile, ScocInventoryItem, ScocPeerState};
use xray_formats::scop::{
    ScopActorStats, ScopAlifeEntity, ScopAlifeKind, ScopAlifePayload, ScopGraphAndQuests,
    ScopHeader, ScopSaveFile, ScopStoryQuest, ScopWeatherState,
};
use xray_formats::spawns::{
    alife_object_flags, ph_skeleton_flags, AlifeCreatureAbstractState, AlifeCreatureActorState,
    AlifeDynamicVisualState, AlifeHumanAbstractState, AlifeHumanStalkerState,
    AlifeMonsterAbstractState, AlifeObjectState, AlifeTraderAbstractState, ServerPhSkeletonState,
    ServerVisualState, INVALID_OBJECT_ID, INVALID_SPAWN_STORY_ID,
};
use xray_net::{ClientSession, EntityDelta, HostSession};
use xray_vfs::VirtualFileSystem;

struct TempDirGuard(PathBuf);

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn make_temp_gamedata(tag: &str) -> TempDirGuard {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("xray_engine_test_{tag}_{nonce}"));
    fs::create_dir_all(&dir).expect("create temp test gamedata dir");
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

fn build_synthetic_level_bytes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // 1. level.geom: 1 VB (4 vertices), 1 IB (6 indices = quad)
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
    vb_stream.extend_from_slice(&4u32.to_le_bytes());
    for p in [
        [-20.0f32, 2.5, -20.0],
        [20.0, 2.5, -20.0],
        [20.0, 2.5, 20.0],
        [-20.0, 2.5, 20.0],
    ] {
        for c in p {
            vb_stream.extend_from_slice(&c.to_le_bytes());
        }
    }

    let mut ib_stream = Vec::new();
    ib_stream.extend_from_slice(&1u32.to_le_bytes());
    ib_stream.extend_from_slice(&6u32.to_le_bytes());
    for i in [0u16, 1, 2, 0, 2, 3] {
        ib_stream.extend_from_slice(&i.to_le_bytes());
    }

    let mut geom_w = ChunkWriter::new();
    geom_w.write_chunk(level_chunks::FSL_VB, &vb_stream);
    geom_w.write_chunk(level_chunks::FSL_IB, &ib_stream);
    let geom_bytes = geom_w.into_bytes();

    // 2. level: 2 shaders, 1 visual referencing VB 0 / IB 0 with shader_id 1
    let mut shd = Vec::new();
    shd.extend_from_slice(&2u32.to_le_bytes());
    shd.extend_from_slice(b"\0");
    shd.extend_from_slice(b"levels\\escape_ground/grnd\\grnd_earth_01\0");

    let mut vis0 = ChunkWriter::new();
    let mut hdr = vec![4u8, OgfVisualType::Normal as u8];
    hdr.extend_from_slice(&1u16.to_le_bytes()); // shader_id = 1
    hdr.extend_from_slice(&[0u8; 40]); // bbox + bsphere
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
    let level_bytes = lvl_w.into_bytes();

    // 3. level.cform: walkable horizontal ground quad at Y = 2.5 (counter-clockwise upward normal)
    let cform = CformModel::from_parts(
        vec![
            [-50.0, 2.5, -50.0],
            [-50.0, 2.5, 50.0],
            [50.0, 2.5, 50.0],
            [50.0, 2.5, -50.0],
        ],
        vec![
            CformTriangle {
                verts: [0, 1, 2],
                material: 1,
                suppress_shadows: false,
                suppress_wm: false,
                sector: 0,
            },
            CformTriangle {
                verts: [0, 2, 3],
                material: 1,
                suppress_shadows: false,
                suppress_wm: false,
                sector: 0,
            },
        ],
    )
    .expect("build synthetic cform");

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

    (level_bytes, geom_bytes, cform_bytes)
}

fn build_sample_scop_and_scoc() -> (ScopSaveFile, ScocFile) {
    let sample_alife = |gvid, lvid, story_id| AlifeObjectState {
        game_vertex_id: gvid,
        distance: 1.0,
        direct_control: true,
        level_vertex_id: lvid,
        flags: alife_object_flags::SWITCH_ONLINE | alife_object_flags::CAN_SAVE,
        custom_data: String::new(),
        story_id,
        spawn_story_id: INVALID_SPAWN_STORY_ID,
        subclass_payload: Vec::new(),
    };
    let sample_dyn = |gvid, lvid, story_id, vis: &str| AlifeDynamicVisualState {
        alife: sample_alife(gvid, lvid, story_id),
        visual: ServerVisualState {
            visual_name: vis.to_string(),
            flags: 0,
        },
        trailing: Vec::new(),
    };
    let sample_skel = || ServerPhSkeletonState {
        startup_animation: "idle".to_string(),
        flags: ph_skeleton_flags::ACTIVE,
        source_id: INVALID_OBJECT_ID,
        saved_bones: None,
    };

    let actor_state = AlifeCreatureActorState {
        creature: AlifeCreatureAbstractState {
            base: sample_dyn(12, 4200, 0, "actors\\stalker_hero\\stalker_hero_1"),
            team: 0,
            squad: 0,
            group: 0,
            health: 0.95,
            dynamic_out_restrictions: Vec::new(),
            dynamic_in_restrictions: Vec::new(),
            killer_id: INVALID_OBJECT_ID,
            game_death_time: 0,
            trailing: Vec::new(),
        },
        trader: AlifeTraderAbstractState {
            money: 25000,
            specific_character: "actor".to_string(),
            trader_flags: 1,
            character_profile: "actor".to_string(),
            community_index: 0,
            rank: 1200,
            reputation: 150,
            character_name: "Zilla".to_string(),
            deadbody_can_take: true,
            deadbody_closed: false,
        },
        skeleton: sample_skel(),
        holder_id: INVALID_OBJECT_ID,
        trailing: Vec::new(),
    };

    let wolf_state = AlifeHumanStalkerState {
        human: AlifeHumanAbstractState {
            trader: AlifeTraderAbstractState {
                money: 1500,
                specific_character: "esc_2_12_stalker_wolf".to_string(),
                trader_flags: 1,
                character_profile: "esc_2_12_stalker_wolf".to_string(),
                community_index: 1,
                rank: 900,
                reputation: 400,
                character_name: "Wolf".to_string(),
                deadbody_can_take: true,
                deadbody_closed: false,
            },
            monster: AlifeMonsterAbstractState {
                creature: AlifeCreatureAbstractState {
                    base: sample_dyn(14, 5100, 6, "actors\\stalker_neutral\\stalker_neutral_3"),
                    team: 1,
                    squad: 1,
                    group: 1,
                    health: 1.0,
                    dynamic_out_restrictions: Vec::new(),
                    dynamic_in_restrictions: Vec::new(),
                    killer_id: INVALID_OBJECT_ID,
                    game_death_time: 0,
                    trailing: Vec::new(),
                },
                    out_space_restrictors: String::new(),
                    in_space_restrictors: String::new(),
                    smart_terrain_id: INVALID_OBJECT_ID,
                    task_reached: false,
                    trailing: Vec::new(),
                },
                equipment_prefs: vec![1, 2, 1],
                main_weapon_prefs: vec![2, 1],
                trailing: Vec::new(),
            },
            skeleton: sample_skel(),
            trailing: Vec::new(),
        };

    let mut quests = ScopGraphAndQuests::default();
    quests.add_quest(ScopStoryQuest {
        owner_id: 0,
        task: SavedGameTask {
            task_id: "esc_coop_sync".to_string(),
            task_state: 1,
            task_type: 0,
            receive_time: 1000,
            finish_time: 0,
            time_to_complete: 0,
            timer_finish: 0,
            title: "Meet Falcon at Rookie Village".to_string(),
            description: "Test story quest".to_string(),
            script_helper: SavedTaskScriptHelper::default(),
            icon_texture_name: "ui_iconsTotal_grouping".to_string(),
            map_hint: "Rookie Village".to_string(),
            map_location: "green_location".to_string(),
            map_object_id: 1,
            priority: 1,
        },
    });
    quests.set_active_quest("esc_coop_sync");

    let scop = ScopSaveFile {
        header: ScopHeader {
            level_id: Some(0),
            level_name: "l01_escape".to_string(),
            actor_stats: ScopActorStats {
            position: [12.0, 4.3, -8.0],
            angles: [0.0, 1.25, 0.0],
            money: 25000,
            rank: 1200,
            reputation: 150,
            health: 0.95,
            actor_name: "Zilla".to_string(),
            ..Default::default()
        },
            ..Default::default()
        },
        entities: vec![
            ScopAlifeEntity::new(
                0,
                "actor",
                "single_player",
                [12.0, 4.3, -8.0],
                [0.0, 1.25, 0.0],
                ScopAlifePayload::Actor(actor_state),
            ),
            ScopAlifeEntity::new(
                1,
                "stalker",
                "esc_2_12_stalker_wolf",
                [18.0, 2.5, -2.0],
                [0.0, 0.0, 0.0],
                ScopAlifePayload::Stalker(wolf_state),
            ),
        ],
        graph_and_quests: quests,
        weather: ScopWeatherState {
            current_weather: "w_clear1".to_string(),
            next_weather: "w_clear1".to_string(),
            cycle_time_sec: 32_400.0,
            ..Default::default()
        },
        extra_chunks: Vec::new(),
    };

    let mut peer_state = ScocPeerState::default();
    peer_state.head_tbl.visual = Some("actors\\stalker_merc\\stalker_merc_2.ogf".to_string());
    peer_state.head_tbl.faction = Some("stalker".to_string());
    peer_state.inventory.push(ScocInventoryItem::new("wpn_ak74m", 0.92));
    peer_state.inventory.push(ScocInventoryItem::new("medkit_army", 1.0));
    let scoc = ScocFile::from_peer_state(&peer_state);

    (scop, scoc)
}

#[test]
fn test_world_loader_and_weather_cycle_from_vfs() {
    let temp = make_temp_gamedata("world_loader");
    let level_dir = temp.0.join("levels").join("l01_escape");
    let weather_dir = temp.0.join("configs").join("environment").join("weathers");
    fs::create_dir_all(&level_dir).unwrap();
    fs::create_dir_all(&weather_dir).unwrap();

    let (level_bytes, geom_bytes, cform_bytes) = build_synthetic_level_bytes();
    fs::write(level_dir.join("level"), level_bytes).unwrap();
    fs::write(level_dir.join("level.geom"), geom_bytes).unwrap();
    fs::write(level_dir.join("level.cform"), cform_bytes).unwrap();
    fs::write(
        weather_dir.join("w_clear1.ltx"),
        "[09:00:00]\nambient_color = 0.25, 0.25, 0.25\nsky_color = 0.7, 0.8, 1.0\nsun_color = 1.0, 0.95, 0.85\nrain_density = 0.0\nwind_velocity = 3.5\nwind_direction = 45.0\n",
    )
    .unwrap();

    let mut vfs = VirtualFileSystem::new();
    vfs.mount_loose_dir(&temp.0);

    let mut world = WorldLoader::load_from_vfs(
        &vfs,
        &LevelLoadConfig {
            level_dir: "$game_levels$/l01_escape".to_string(),
            resolve_visual_meshes: true,
            resolve_textures: true,
            load_fastpath_geom: false,
            load_optional_aux: false,
            load_weather: true,
        },
    )
    .expect("load world from VFS");

    let summary = world.summary();
    assert_eq!(summary.raw_visual_count, 1);
    assert_eq!(summary.resolved_mesh_count, 1);
    assert_eq!(summary.cform_triangle_count, 2);
    assert_eq!(summary.weather_cycle_count, 1);
    assert_eq!(world.visual_meshes[0].mesh.vertices.len(), 4);
    assert_eq!(
        world.visual_meshes[0].texture_links[0].diffuse_dds,
        "textures\\grnd\\grnd_earth_01.dds"
    );

    let ground_y = world
        .ground_height_at([0.0, 10.0, 0.0], 20.0)
        .expect("ground hit expected");
    assert!((ground_y - 2.5).abs() < 1e-4);

    let env = world
        .step_weather(0.5, [0.0, 4.3, 0.0], [0.0, 0.0, 1.0], 450.0)
        .expect("weather cycle should evaluate");
    assert!((env.sky_color[2] - 1.0).abs() < 0.05);
}

#[test]
fn test_alife_manager_scop_scoc_and_coop_runtime_loop() {
    let temp = make_temp_gamedata("alife_runtime");
    let level_dir = temp.0.join("levels").join("l01_escape");
    let save_dir = temp.0.join("savedgames");
    fs::create_dir_all(&level_dir).unwrap();
    fs::create_dir_all(&save_dir).unwrap();

    let (level_bytes, geom_bytes, cform_bytes) = build_synthetic_level_bytes();
    fs::write(level_dir.join("level"), level_bytes).unwrap();
    fs::write(level_dir.join("level.geom"), geom_bytes).unwrap();
    fs::write(level_dir.join("level.cform"), cform_bytes).unwrap();

    let (scop, scoc) = build_sample_scop_and_scoc();
    fs::write(save_dir.join("zilla_host.scop"), scop.to_bytes().unwrap()).unwrap();
    let scoc_bytes = scoc.to_bytes().unwrap();

    let mut vfs = VirtualFileSystem::new();
    vfs.mount_loose_dir(&temp.0);

    // Initialize headless runtime & load world + host .scop from VFS
    let mut runtime = EngineRuntime::new(
        EngineConfig {
            window: WindowConfig {
                title: "TestEngineWorldAlife".to_string(),
                width: 320,
                height: 240,
                visible: false,
            },
            ..Default::default()
        },
        false,
    )
    .expect("create runtime");

    runtime
        .load_world_and_save_from_vfs(
            &vfs,
            &LevelLoadConfig::minimal("$game_levels$/l01_escape"),
            Some("savedgames/zilla_host.scop"),
        )
        .expect("load world and scop savegame");

    // Verify camera synchronized to host actor's saved position
    assert!((runtime.camera.position[0] - 12.0).abs() < 1e-4);
    assert!((runtime.camera.position[2] + 8.0).abs() < 1e-4);

    // Attach Falcon's .scoc peer inventory onto the live ALife manager
    let peer_entity_id = runtime
        .alife
        .as_mut()
        .unwrap()
        .attach_peer_scoc_bytes(1, "Falcon", 0xFA1C_0401, &scoc_bytes)
        .expect("attach peer scoc");

    {
        let mgr = runtime.alife.as_ref().unwrap();
        assert_eq!(mgr.online_entity_count(), 3); // Host + Wolf + Falcon peer
        assert_eq!(
            mgr.find_by_story_id(6).map(|e| e.name.as_str()),
            Some("esc_2_12_stalker_wolf")
        );
        let falcon_ent = mgr.entity(peer_entity_id).unwrap();
        assert!(matches!(
            falcon_ent.role,
            EntityRole::RemotePeerActor { slot: 1, .. }
        ));
        assert_eq!(
            falcon_ent.visual_name.as_deref(),
            Some("actors\\stalker_merc\\stalker_merc_2.ogf")
        );
        let falcon_avatar = mgr.peer_by_slot(1).unwrap();
        assert_eq!(falcon_avatar.inventory().len(), 2);
        assert_eq!(falcon_avatar.inventory()[0].section, "wpn_ak74m");
    }

    // Bind live Host/Client UDP loopback sessions so step_frame exercises full co-op delta path
    let host_sess = HostSession::new(0, false).expect("bind host udp");
    let host_addr = host_sess.local_addr().unwrap();
    let connect_addr: std::net::SocketAddr =
        format!("127.0.0.1:{}", host_addr.port()).parse().unwrap();
    let mut client_sess =
        ClientSession::connect(connect_addr, 0xFA1C_0401, "Falcon", false).expect("connect client");
    runtime.attach_network(NetworkSession::Host(host_sess));

    // Complete handshake across a few updates
    for _ in 0..16 {
        runtime.step_frame().expect("runtime step");
        client_sess.update().expect("client poll");
        if client_sess.state() == xray_net::ClientState::Connected {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    // Move host forward via EngineInput + verify host transform syncs back into ALifeManager
    runtime.input.begin_frame();
    runtime.input.on_key_down(EngineInput::VK_KEY_W);
    runtime.step_frame().expect("runtime step moving");

    // Client pushes Falcon's updated transform over UDP -> Host applies it into AlifeManager
    let peer_delta = EntityDelta {
        tick: 42,
        entity_id: peer_entity_id,
        pos: [24.0, 2.5, 6.0],
        yaw: 0.75,
        pitch: -0.1,
        vel: [1.5, 0.0, 0.0],
        anim_state: 3,
        weapon_state: 1,
        health: 0.88,
    };
    client_sess.send_entity_delta(&peer_delta).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(15));
    for _ in 0..4 {
        runtime.step_frame().expect("runtime recv delta");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }

    // Check roundtrip export to both host .scop and peer .scoc
    let mgr: &AlifeManager = runtime.alife.as_ref().unwrap();
    assert_eq!(mgr.entities_by_kind(ScopAlifeKind::Stalker).len(), 2);
    let exported_scop = mgr.export_to_scop(runtime.world.as_ref().map(|w| &w.weather));
    let re_parsed_scop = ScopSaveFile::parse(&exported_scop.to_bytes().unwrap()).unwrap();
    assert_eq!(re_parsed_scop.header.level_name, "l01_escape");
    assert_eq!(re_parsed_scop.entities.len(), 2); // Remote peer excluded from host .scop

    let exported_scoc = mgr.export_peer_scoc(peer_entity_id).unwrap();
    let re_parsed_peer = exported_scoc.extract_peer_state();
    assert_eq!(re_parsed_peer.inventory.len(), 2);
    assert_eq!(re_parsed_peer.inventory[0].section, "wpn_ak74m");

    // Verify Falcon avatar transform was applied if UDP packet arrived in time (apply_network_delta directly too)
    let mut direct_check = mgr.clone();
    assert!(direct_check.apply_network_delta(&peer_delta));
    let updated_falcon = direct_check.peer_by_slot(1).unwrap();
    assert_eq!(updated_falcon.position, [24.0, 2.5, 6.0]);
    assert!((updated_falcon.health - 0.88).abs() < 1e-4);
}
