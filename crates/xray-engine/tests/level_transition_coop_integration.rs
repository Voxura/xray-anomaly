//! End-to-End Dynamic Level Transition, Loading Screen & Co-Op Sync Integration Test (`xray-engine`).
//!
//! Verifies Phase D end-to-end:
//! - Automatic extraction of `AlifeLevelChangerState` trigger volumes from `level.spawn`
//! - Player overlap detection, `TravelPrompt` modal overlay, cursor unlock & `[E]` confirmation
//! - Full 5-stage loading state machine (`Idle -> SavingPlayerState -> UnloadLevel -> LoadAssets -> Ready -> Idle`)
//! - Cross-level `.scoc` inventory + weapon slot carry-over & `.scop` quest preservation
//! - Destination `WorldLoader` geometry + `level.cform` BVH terrain elevation update (`y = 0.0 -> 2.0`)
//! - Live `xray-net` `LevelChangeNotice` handshake (`PrepareTransition -> ReadyAtDestination -> CommitSpawn`)
//!   between co-op Host and Client `EngineRuntime` instances over loopback UDP
//! - Live DirectX 12 RTX 4060 `UiCanvas` loading splash screen + multi-stage progress meter rendering

use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use xray_engine::game_loop::{EngineConfig, EngineRuntime, NetworkSession};
use xray_engine::input::EngineInput;
use xray_engine::level_transition::TransitionState;
use xray_engine::ui_inventory::EquipmentSlot;
use xray_engine::window::{WindowConfig, WindowEvent};
use xray_engine::world_loader::{LevelLoadConfig, WorldLoader};
use xray_formats::chunk::ChunkWriter;
use xray_formats::cform::{CformTriangle, CFORM_CURRENT_VERSION};
use xray_formats::level::{d3d_decl_type, d3d_decl_usage, level_chunks, VertexElement9};
use xray_formats::savegames::{SavedGameTask, SavedTaskScriptHelper};
use xray_formats::scoc::{ScocFile, ScocInventoryItem, ScocPeerState};
use xray_formats::scop::{ScopGraphAndQuests, ScopStoryQuest};
use xray_formats::spawns::{
    alife_object_flags, m_spawn_flags, AlifeLevelChangerState, AlifeObjectState,
    AlifeSpaceRestrictorState, LevelSpawnData, ServerShapeDef, SpawnPacket, INVALID_OBJECT_ID,
    INVALID_SPAWN_STORY_ID, INVALID_STORY_ID, SPAWN_VERSION,
};
use xray_net::{ClientSession, ClientState, HostSession};
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
    let dir = std::env::temp_dir().join(format!("xray_phase_d_{tag}_{nonce}"));
    fs::create_dir_all(&dir).expect("create temp test dir");
    TempDirGuard(dir)
}

fn make_solid_4x4_rgba_dds(rgba: [u8; 4]) -> Vec<u8> {
    let mut dds = vec![0u8; 128 + 16 * 4];
    dds[0..4].copy_from_slice(b"DDS ");
    dds[4..8].copy_from_slice(&124u32.to_le_bytes());
    dds[8..12].copy_from_slice(&(0x1u32 | 0x2 | 0x4 | 0x1000).to_le_bytes());
    dds[12..16].copy_from_slice(&4u32.to_le_bytes());
    dds[16..20].copy_from_slice(&4u32.to_le_bytes());
    dds[76..80].copy_from_slice(&32u32.to_le_bytes());
    dds[80..84].copy_from_slice(&(0x40u32 | 0x1).to_le_bytes());
    dds[88..92].copy_from_slice(&32u32.to_le_bytes());
    dds[92..96].copy_from_slice(&0x0000_00FFu32.to_le_bytes());
    dds[96..100].copy_from_slice(&0x0000_FF00u32.to_le_bytes());
    dds[100..104].copy_from_slice(&0x00FF_0000u32.to_le_bytes());
    dds[104..108].copy_from_slice(&0xFF00_0000u32.to_le_bytes());
    for px in dds[128..].as_chunks_mut::<4>().0 {
        *px = rgba;
    }
    dds
}

fn build_cform_plane_bytes(y_height: f32) -> Vec<u8> {
    let verts = [
        [-200.0f32, y_height, -200.0],
        [200.0, y_height, -200.0],
        [200.0, y_height, 200.0],
        [-200.0, y_height, 200.0],
    ];
    let tris = [
        CformTriangle {
            verts: [0, 2, 1],
            material: 0,
            suppress_shadows: false,
            suppress_wm: false,
            sector: 0,
        },
        CformTriangle {
            verts: [0, 3, 2],
            material: 0,
            suppress_shadows: false,
            suppress_wm: false,
            sector: 0,
        },
    ];
    let mut out = Vec::with_capacity(36 + verts.len() * 12 + tris.len() * 16);
    out.extend_from_slice(&CFORM_CURRENT_VERSION.to_le_bytes());
    out.extend_from_slice(&(verts.len() as u32).to_le_bytes());
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for f in [-200.0f32, y_height - 1.0, -200.0, 200.0, y_height + 1.0, 200.0] {
        out.extend_from_slice(&f.to_le_bytes());
    }
    for v in &verts {
        for c in v {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    for t in &tris {
        for vi in &t.verts {
            out.extend_from_slice(&vi.to_le_bytes());
        }
        out.extend_from_slice(&t.to_raw_dummy().to_le_bytes());
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn build_level_changer_spawn_packet(
    entity_id: u16,
    name: &str,
    trigger_pos: [f32; 3],
    radius: f32,
    dest_level: &str,
    dest_point: &str,
    dest_pos: [f32; 3],
    dest_angles: [f32; 3],
) -> SpawnPacket {
    let lc_state = AlifeLevelChangerState {
        restrictor: AlifeSpaceRestrictorState {
            alife: AlifeObjectState {
                game_vertex_id: 10,
                distance: 0.0,
                direct_control: true,
                level_vertex_id: 100,
                flags: alife_object_flags::SWITCH_ONLINE | alife_object_flags::VISIBLE_FOR_AI,
                custom_data: String::new(),
                story_id: INVALID_STORY_ID,
                spawn_story_id: INVALID_SPAWN_STORY_ID,
                subclass_payload: Vec::new(),
            },
            shapes: vec![ServerShapeDef::Sphere {
                center: [0.0, 0.0, 0.0],
                radius,
            }],
            restrictor_type: 3,
            trailing: Vec::new(),
        },
        next_graph_vertex: 412,
        next_level_vertex: 98_765,
        next_position: dest_pos,
        next_angles: dest_angles,
        level_to_change: dest_level.to_string(),
        level_point_to_change: dest_point.to_string(),
        silent_mode: false,
        trailing: Vec::new(),
    };
    SpawnPacket {
        section: "level_changer".to_string(),
        name_replace: name.to_string(),
        game_id_legacy: 0,
        rp: 0,
        position: trigger_pos,
        angle: [0.0; 3],
        respawn_time: 0,
        id: entity_id,
        parent_id: INVALID_OBJECT_ID,
        phantom_id: INVALID_OBJECT_ID,
        flags: m_spawn_flags::OBJECT_LOCAL | m_spawn_flags::VERSION,
        version: SPAWN_VERSION,
        game_type: u16::MAX,
        script_version: 8,
        client_data: Vec::new(),
        spawn_id: entity_id,
        state_payload: lc_state.to_bytes(),
        trailing_update_payload: Vec::new(),
    }
}

fn write_level_dir(
    level_dir: &Path,
    ground_y: f32,
    spawns: Vec<SpawnPacket>,
) {
    fs::create_dir_all(level_dir).unwrap();

    let mut vb = Vec::new();
    vb.extend_from_slice(&1u32.to_le_bytes());
    for e in [
        VertexElement9 {
            stream: 0,
            offset: 0,
            decl_type: d3d_decl_type::FLOAT3,
            method: 0,
            usage: d3d_decl_usage::POSITION,
            usage_index: 0,
        },
        VertexElement9::END,
    ] {
        vb.extend_from_slice(&e.stream.to_le_bytes());
        vb.extend_from_slice(&e.offset.to_le_bytes());
        vb.push(e.decl_type);
        vb.push(e.method);
        vb.push(e.usage);
        vb.push(e.usage_index);
    }
    vb.extend_from_slice(&3u32.to_le_bytes());
    for p in [
        [-5.0f32, ground_y, -5.0],
        [5.0, ground_y, -5.0],
        [0.0, ground_y, 5.0],
    ] {
        for c in p {
            vb.extend_from_slice(&c.to_le_bytes());
        }
    }
    let mut ib = Vec::new();
    ib.extend_from_slice(&1u32.to_le_bytes());
    ib.extend_from_slice(&3u32.to_le_bytes());
    for i in [0u16, 1, 2] {
        ib.extend_from_slice(&i.to_le_bytes());
    }
    let mut geom_w = ChunkWriter::new();
    geom_w.write_chunk(level_chunks::FSL_VB, &vb);
    geom_w.write_chunk(level_chunks::FSL_IB, &ib);
    fs::write(level_dir.join("level.geom"), geom_w.into_bytes()).unwrap();

    let mut shd = Vec::new();
    shd.extend_from_slice(&1u32.to_le_bytes());
    shd.extend_from_slice(b"\0");
    let mut lvl_w = ChunkWriter::new();
    lvl_w.write_chunk(level_chunks::FSL_HEADER, &[14u8, 0, 0, 0, 0, 0, 0, 0]);
    lvl_w.write_chunk(level_chunks::FSL_SHADERS, &shd);
    lvl_w.write_chunk(level_chunks::FSL_VISUALS, &ChunkWriter::new().into_bytes());
    fs::write(level_dir.join("level"), lvl_w.into_bytes()).unwrap();

    fs::write(level_dir.join("level.cform"), build_cform_plane_bytes(ground_y)).unwrap();
    let spawn_data = LevelSpawnData { spawns };
    fs::write(level_dir.join("level.spawn"), spawn_data.to_bytes()).unwrap();
}

fn build_multi_level_vfs(root: &Path) -> Arc<VirtualFileSystem> {
    let cfg_dir = root.join("configs");
    let intro_dir = root.join("textures").join("intro");
    let ui_dir = root.join("textures").join("ui");
    fs::create_dir_all(&cfg_dir).unwrap();
    fs::create_dir_all(&intro_dir).unwrap();
    fs::create_dir_all(&ui_dir).unwrap();

    // Level 1: Cordon (`y = 0.0`), changer at [12.0, 0.9, 20.0] -> `l02_garbage` at [-80.0, 3.65, -140.0]
    let esc_to_gar = build_level_changer_spawn_packet(
        700,
        "esc_level_changer_to_garbage",
        [12.0, 0.9, 20.0],
        4.5,
        "l02_garbage",
        "gar_start_from_escape",
        [-80.0, 3.65, -140.0],
        [0.0, 1.25, 0.0],
    );
    write_level_dir(&root.join("levels").join("l01_escape"), 0.0, vec![esc_to_gar]);

    // Level 2: Garbage (`y = 2.0`), return changer at [-100.0, 2.9, -180.0] -> `l01_escape`
    let gar_to_esc = build_level_changer_spawn_packet(
        701,
        "gar_level_changer_to_escape",
        [-100.0, 2.9, -180.0],
        4.5,
        "l01_escape",
        "esc_start_from_garbage",
        [0.0, 1.65, 0.0],
        [0.0, 0.0, 0.0],
    );
    write_level_dir(&root.join("levels").join("l02_garbage"), 2.0, vec![gar_to_esc]);

    fs::write(
        cfg_dir.join("game_levels.ltx"),
        r#"[levels]
level01 =
level02 =

[level01]
name = l01_escape
caption = Cordon
offset = 0.0, 0.0, 0.0
id = 1

[level02]
name = l02_garbage
caption = Garbage
offset = 0.0, 0.0, 1000.0
id = 2
"#,
    )
    .unwrap();

    fs::write(
        cfg_dir.join("fonts.ltx"),
        "[ui_font_letterica16_russian]\nshader = font\ntexture = ui\\ui_font_letter_16_1024\nsize = 16\n",
    )
    .unwrap();
    let mut ini = String::from("[symbol_coords]\nheight = 16\n");
    for code in 32..=126u32 {
        ini.push_str(&format!("{code:03} = 0, 0, 8\n"));
    }
    fs::write(ui_dir.join("ui_font_letter_16_1024.ini"), ini).unwrap();
    fs::write(
        ui_dir.join("ui_font_letter_16_1024.dds"),
        make_solid_4x4_rgba_dds([240, 240, 240, 255]),
    )
    .unwrap();
    fs::write(
        intro_dir.join("intro_l02_garbage.dds"),
        make_solid_4x4_rgba_dds([60, 85, 70, 255]),
    )
    .unwrap();

    let mut vfs = VirtualFileSystem::new();
    vfs.mount_loose_dir(root);
    Arc::new(vfs)
}

#[test]
fn test_coop_level_transition_loading_screen_and_state_sync() {
    let tmp = make_temp_dir("coop_transit");
    let vfs = build_multi_level_vfs(&tmp.0);

    // 1. Set up real HostSession + ClientSession over loopback UDP
    let host_net = HostSession::new(0, false).expect("bind IPv4 host session");
    let bound_port = host_net.local_addr().unwrap().port();
    let host_addr: SocketAddr = format!("127.0.0.1:{bound_port}").parse().unwrap();
    let client_net =
        ClientSession::connect(host_addr, 7777, "Falcon", false).expect("connect co-op client");

    let mut host_rt = EngineRuntime::new(
        EngineConfig {
            window: WindowConfig {
                title: "PhaseDHost".to_string(),
                width: 640,
                height: 480,
                visible: false,
            },
            vsync: false,
            ..Default::default()
        },
        false,
    )
    .unwrap();
    let mut client_rt = EngineRuntime::new(
        EngineConfig {
            window: WindowConfig {
                title: "PhaseDClient".to_string(),
                width: 640,
                height: 480,
                visible: false,
            },
            vsync: false,
            ..Default::default()
        },
        false,
    )
    .unwrap();

    host_rt.attach_vfs(Arc::clone(&vfs));
    client_rt.attach_vfs(Arc::clone(&vfs));
    host_rt.attach_network(NetworkSession::Host(host_net));
    client_rt.attach_network(NetworkSession::Client(client_net));

    // Complete co-op UDP handshake before starting
    for _ in 0..40 {
        host_rt.step_frame().unwrap();
        client_rt.step_frame().unwrap();
        let connected = matches!(
            (&host_rt.network, &client_rt.network),
            (
                Some(NetworkSession::Host(h)),
                Some(NetworkSession::Client(c))
            ) if h.has_peer() && c.state() == ClientState::Connected
        );
        if connected {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(
        host_rt.network.as_ref().unwrap().is_connected_to_peer()
            && client_rt.network.as_ref().unwrap().is_connected_to_peer(),
        "Host and Client co-op sessions must be connected"
    );

    // 2. Load initial level `l01_escape` on both Host & Client + attach co-op peer & quests/items
    let esc_cfg = LevelLoadConfig {
        level_dir: "levels/l01_escape".to_string(),
        resolve_visual_meshes: false,
        resolve_textures: false,
        load_fastpath_geom: false,
        load_optional_aux: true,
        load_weather: false,
    };
    host_rt.level_transition.default_load_config = esc_cfg.clone();
    client_rt.level_transition.default_load_config = esc_cfg.clone();
    host_rt
        .load_world_and_save_from_vfs(&vfs, &esc_cfg, None)
        .unwrap();
    client_rt
        .load_world_and_save_from_vfs(&vfs, &esc_cfg, None)
        .unwrap();

    // Verify `level_changer` was automatically extracted from `l01_escape/level.spawn`
    assert_eq!(host_rt.level_transition.changers.len(), 1);
    assert_eq!(host_rt.level_transition.changers[0].target_level, "l02_garbage");

    // Populate persisted Host inventory (.scoc), active quest (.scop), & connected peer avatar
    let mut host_peer_state = ScocPeerState::default();
    let mut wpn = ScocInventoryItem::new("wpn_vintorez", 0.91);
    wpn.id = Some(1001);
    wpn.ammo_elapsed = Some(10);
    wpn.slot = Some(2);
    let mut med = ScocInventoryItem::new("medkit_army", 1.0);
    med.id = Some(1002);
    med.remaining_uses = Some(3);
    host_peer_state.inventory = vec![wpn, med];
    host_peer_state.item_remaining_uses.insert(1002, 3);
    let host_scoc = ScocFile::from_peer_state(&host_peer_state);
    host_rt.attach_host_scoc(&host_scoc);
    assert!(
        host_rt
            .inventory
            .item_in_slot(EquipmentSlot::PrimaryWeapon)
            .is_some_and(|i| i.section == "wpn_vintorez")
    );

    if let Some(mgr) = host_rt.alife.as_mut() {
        let peer_scoc = ScocFile::default();
        mgr.attach_peer_scoc(1, "Falcon", 7777, &peer_scoc);
        mgr.graph_and_quests = ScopGraphAndQuests {
            active_quest_id: "q_visit_garbage".to_string(),
            story_quests: vec![ScopStoryQuest {
                owner_id: 0,
                task: SavedGameTask {
                    task_id: "q_visit_garbage".to_string(),
                    task_state: 1,
                    task_type: 0,
                    receive_time: 0,
                    finish_time: 0,
                    time_to_complete: 0,
                    timer_finish: 0,
                    title: "Reach the Garbage checkpoint".to_string(),
                    description: "Move north from Cordon".to_string(),
                    script_helper: SavedTaskScriptHelper::default(),
                    icon_texture_name: String::new(),
                    map_hint: "To Garbage".to_string(),
                    map_location: "l02_garbage".to_string(),
                    map_object_id: INVALID_OBJECT_ID,
                    priority: 10,
                },
            }],
            ..Default::default()
        };
    }

    // 3. Move Host into `esc_level_changer_to_garbage` trigger volume ([12.0, 0.9, 20.0]) -> Travel Prompt opens
    host_rt.physics.feet_position = [12.0, 0.0, 20.0];
    host_rt.camera.position = host_rt.physics.eye_position();
    host_rt.step_frame().unwrap();

    let prompt = host_rt
        .level_transition
        .active_prompt
        .as_ref()
        .expect("stepping inside level_changer must open TravelPrompt");
    assert_eq!(prompt.target_level, "l02_garbage");
    assert_eq!(prompt.target_title, "Garbage");
    assert!(!host_rt.input.is_cursor_locked(), "prompt must unlock cursor");
    assert_eq!(host_rt.level_transition.state, TransitionState::Idle);

    // 4. Press `[E]` on Host to confirm travel -> both Host & Client synchronously advance through
    //    `SavingPlayerState -> UnloadLevel -> LoadAssets -> Ready -> Idle`
    host_rt
        .window
        .push_synthetic_event(WindowEvent::KeyDown { vk: EngineInput::VK_KEY_E });
    host_rt.step_frame().unwrap();
    host_rt
        .window
        .push_synthetic_event(WindowEvent::KeyUp { vk: EngineInput::VK_KEY_E });

    let mut host_seen_states = Vec::new();
    let mut client_seen_states = Vec::new();
    host_seen_states.push(host_rt.level_transition.state);
    assert_eq!(host_rt.level_transition.state, TransitionState::SavingPlayerState);

    for _ in 0..40 {
        client_rt.step_frame().unwrap();
        client_seen_states.push(client_rt.level_transition.state);
        host_rt.step_frame().unwrap();
        host_seen_states.push(host_rt.level_transition.state);

        let both_done = host_rt.level_transition.completed_transitions == 1
            && client_rt.level_transition.completed_transitions == 1
            && host_rt.level_transition.state == TransitionState::Idle
            && client_rt.level_transition.state == TransitionState::Idle;
        if both_done {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    // Both peers saw all 4 loading stages and returned synchronously to `Idle`
    for st in [
        TransitionState::SavingPlayerState,
        TransitionState::UnloadLevel,
        TransitionState::LoadAssets,
        TransitionState::Ready,
    ] {
        assert!(
            host_seen_states.contains(&st),
            "host should traverse stage {st:?}; observed {host_seen_states:?}"
        );
        assert!(
            client_seen_states.contains(&st),
            "client should synchronously traverse stage {st:?}; observed {client_seen_states:?}"
        );
    }
    assert_eq!(host_rt.level_transition.completed_transitions, 1);
    assert_eq!(client_rt.level_transition.completed_transitions, 1);
    assert_eq!(host_rt.level_transition.current_level, "l02_garbage");
    assert_eq!(client_rt.level_transition.current_level, "l02_garbage");

    // 5. Validate destination level geometry, Cform BVH (`y=2.0`), `.scoc` inventory & `.scop` quest carry-over
    assert!(
        host_rt
            .world
            .as_ref()
            .unwrap()
            .level_dir
            .contains("l02_garbage")
    );
    assert!(
        client_rt
            .world
            .as_ref()
            .unwrap()
            .level_dir
            .contains("l02_garbage")
    );
    assert!(
        (host_rt.physics.feet_position[0] + 80.0).abs() < 1.0
            && (host_rt.physics.feet_position[1] - 2.0).abs() < 0.25,
        "host capsule must settle on Garbage `y=2.0` Cform surface, got {:?}",
        host_rt.physics.feet_position
    );
    assert!(
        (client_rt.physics.feet_position[0] + 80.0).abs() < 1.0
            && (client_rt.physics.feet_position[1] - 2.0).abs() < 0.25,
        "client capsule must synchronously spawn at Garbage `y=2.0`, got {:?}",
        client_rt.physics.feet_position
    );
    assert_eq!(host_rt.level_transition.changers.len(), 1);
    assert_eq!(
        host_rt.level_transition.changers[0].target_level,
        "l01_escape",
        "Garbage's return changer to Cordon should be discovered automatically"
    );
    assert!(
        host_rt
            .inventory
            .item_in_slot(EquipmentSlot::PrimaryWeapon)
            .is_some_and(|i| i.section == "wpn_vintorez")
            && host_rt
                .inventory
                .items
                .iter()
                .any(|i| i.section == "medkit_army" && i.remaining_uses == Some(3)),
        "equipped weapon and medkit uses must survive .scoc carry-over"
    );
    assert_eq!(
        host_rt.hud_weapon.active_weapon().section, "wpn_vintorez",
        "HudWeaponController must remain synced after destination load"
    );
    assert_eq!(
        host_rt
            .alife
            .as_ref()
            .unwrap()
            .graph_and_quests
            .active_quest_id,
        "q_visit_garbage",
        "active ScopStoryQuest must carry over across level transition"
    );
    assert_eq!(
        host_rt.alife.as_ref().unwrap().remote_peers.len(),
        1,
        "co-op peer avatar must remain attached across transition"
    );

    // 6. Live DirectX 12 GPU Loading Screen UI Verification (when GPU adapter is available)
    if let Ok(mut gpu_rt) = EngineRuntime::new(
        EngineConfig {
            window: WindowConfig {
                title: "PhaseDDx12Test".to_string(),
                width: 640,
                height: 480,
                visible: false,
            },
            vsync: false,
            ..Default::default()
        },
        true,
    ) {
        gpu_rt.attach_vfs(Arc::clone(&vfs));
        let world1 = WorldLoader::load_from_vfs(&vfs, &esc_cfg).unwrap();
        gpu_rt.level_transition.default_load_config = esc_cfg;
        gpu_rt.attach_world(world1);
        gpu_rt.sync_world_to_gpu(Some(&vfs)).unwrap();

        // Trigger prompt UI draw on GPU
        gpu_rt.physics.feet_position = [12.0, 0.0, 20.0];
        gpu_rt.step_frame().unwrap();
        assert!(gpu_rt.level_transition.active_prompt.is_some());
        let prompt_quads = gpu_rt
            .render_bridge
            .as_ref()
            .unwrap()
            .last_stats
            .ui_elements_drawn;
        assert!(prompt_quads >= 10, "expected TravelPrompt panel & buttons");

        // Confirm with `[E]` -> observe full-screen `UiCanvas` splash + 4-stage pips + progress bar across loading frames
        gpu_rt
            .window
            .push_synthetic_event(WindowEvent::KeyDown { vk: EngineInput::VK_KEY_E });
        gpu_rt.step_frame().unwrap();
        let loading_quads = gpu_rt
            .render_bridge
            .as_ref()
            .unwrap()
            .last_stats
            .ui_elements_drawn;
        assert!(
            loading_quads >= 10,
            "expected Loading Screen splash, pips, progress bar & font glyphs, got {loading_quads}"
        );

        for _ in 0..16 {
            gpu_rt.step_frame().unwrap();
            if !gpu_rt.level_transition.is_loading() {
                break;
            }
        }
        assert_eq!(gpu_rt.level_transition.completed_transitions, 1);
        assert_eq!(gpu_rt.level_transition.current_level, "l02_garbage");
    }
}
