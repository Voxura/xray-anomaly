//! End-to-end GPU DirectX 12 + Stalker Capsule Physics + VFS Render Graph integration test.
//!
//! Exercises the full 7-stage `RenderBridge` pipeline on live DirectX 12 hardware alongside
//! `StalkerController` BVH collision, `WorldLoader` streaming, and `AlifeManager` entity sync.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use xray_engine::alife_manager::AlifeManager;
use xray_engine::game_loop::{EngineConfig, EngineRuntime};
use xray_engine::input::EngineInput;
use xray_engine::window::WindowConfig;
use xray_engine::world_loader::LevelLoadConfig;
use xray_formats::cform::{CformModel, CformTriangle};
use xray_formats::chunk::ChunkWriter;
use xray_formats::level::{d3d_decl_type, d3d_decl_usage, d3d_light_type, level_chunks, VertexElement9};
use xray_formats::ogf::{fvf_flags, ogf_chunks, OgfVisualType};
use xray_formats::scoc::{ScocFile, ScocInventoryItem, ScocPeerState};
use xray_formats::scop::{
    ScopActorStats, ScopAlifeEntity, ScopAlifePayload, ScopGraphAndQuests, ScopHeader,
    ScopSaveFile, ScopWeatherState,
};
use xray_formats::spawns::{
    alife_object_flags, ph_skeleton_flags, AlifeCreatureAbstractState, AlifeCreatureActorState,
    AlifeDynamicVisualState, AlifeHumanAbstractState, AlifeHumanStalkerState,
    AlifeMonsterAbstractState, AlifeObjectState, AlifeTraderAbstractState, ServerPhSkeletonState,
    ServerVisualState, INVALID_OBJECT_ID, INVALID_SPAWN_STORY_ID,
};
use xray_render_dx12::particle::GpuParticleInstance;
use xray_render_dx12::wallmarks::GpuWallmarkVertex;
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
    let dir = std::env::temp_dir().join(format!("xray_gpu_pipeline_{tag}_{nonce}"));
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

fn build_spawn_visual() -> Vec<u8> {
    let header = |kind: OgfVisualType| {
        let mut bytes = vec![4, kind as u8, 0, 0];
        for value in [-1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1.2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    };
    let mut vertices = Vec::new();
    vertices.extend_from_slice(&fvf_flags::XRAY_STATIC_FVF_STANDARD.to_le_bytes());
    vertices.extend_from_slice(&3u32.to_le_bytes());
    for x in [-1.0f32, 0.0, 1.0] {
        for value in [x, 1.0 - x.abs(), 0.0, 0.0, 0.0, -1.0, 0.0, 0.0] {
            vertices.extend_from_slice(&value.to_le_bytes());
        }
    }
    let mut child = ChunkWriter::new();
    child.write_chunk(ogf_chunks::OGF_HEADER, &header(OgfVisualType::Normal));
    child.write_chunk(ogf_chunks::OGF_VERTICES, &vertices);
    child.write_chunk(ogf_chunks::OGF_INDICES, &[3, 0, 0, 0, 0, 0, 1, 0, 2, 0]);
    let child = child.into_bytes();

    let mut children = ChunkWriter::new();
    children.write_chunk(0, &child);
    children.write_chunk(1, &child);
    let mut root = ChunkWriter::new();
    root.write_chunk(ogf_chunks::OGF_HEADER, &header(OgfVisualType::Hierarchy));
    root.write_chunk(ogf_chunks::OGF_CHILDREN, &children.into_bytes());
    root.into_bytes()
}

fn build_gpu_test_level_bytes() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // 1. level.geom: ground quad [-20..20, y=0.0, -20..20]
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
        [-20.0f32, 0.0, -20.0],
        [20.0, 0.0, -20.0],
        [20.0, 0.0, 20.0],
        [-20.0, 0.0, 20.0],
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

    // 2. level: shader table + 1 static visual + 1 dynamic point light
    let mut shd = Vec::new();
    shd.extend_from_slice(&2u32.to_le_bytes());
    shd.extend_from_slice(b"\0");
    shd.extend_from_slice(b"levels\\escape_ground/grnd\\grnd_earth_01\0");

    let mut vis0 = ChunkWriter::new();
    let mut hdr = vec![4u8, OgfVisualType::Normal as u8];
    hdr.extend_from_slice(&1u16.to_le_bytes()); // shader_id = 1
    // bbox min (-20, 0, -20), bbox max (20, 0.1, 20), bsphere center (0, 0, 0), radius 30
    for f in [
        -20.0f32, 0.0, -20.0, 20.0, 0.1, 20.0, 0.0, 0.0, 0.0, 30.0,
    ] {
        hdr.extend_from_slice(&f.to_le_bytes());
    }
    vis0.write_chunk(ogf_chunks::OGF_HEADER, &hdr);
    let mut gcont = Vec::new();
    for v in [0u32, 0, 4, 0, 0, 6] {
        gcont.extend_from_slice(&v.to_le_bytes());
    }
    vis0.write_chunk(ogf_chunks::OGF_GCONTAINER, &gcont);

    let mut vis_wrap = ChunkWriter::new();
    vis_wrap.write_chunk(0, &vis0.into_bytes());

    // 1 dynamic point light (u32 controller + 104-byte D3DLIGHT9 = 108 bytes)
    let mut lights = Vec::new();
    lights.extend_from_slice(&0u32.to_le_bytes()); // controller_id
    lights.extend_from_slice(&d3d_light_type::POINT.to_le_bytes());
    for f in [1.0f32, 0.75, 0.45, 1.0] {
        lights.extend_from_slice(&f.to_le_bytes()); // diffuse
    }
    for f in [0.0f32, 0.0, 0.0, 1.0] {
        lights.extend_from_slice(&f.to_le_bytes()); // specular
    }
    for f in [0.1f32, 0.1, 0.1, 1.0] {
        lights.extend_from_slice(&f.to_le_bytes()); // ambient
    }
    for f in [2.0f32, 2.5, 4.0] {
        lights.extend_from_slice(&f.to_le_bytes()); // position
    }
    for f in [0.0f32, -1.0, 0.0] {
        lights.extend_from_slice(&f.to_le_bytes()); // direction
    }
    lights.extend_from_slice(&18.0f32.to_le_bytes()); // range
    lights.extend_from_slice(&1.0f32.to_le_bytes()); // falloff
    lights.extend_from_slice(&1.0f32.to_le_bytes()); // atten0
    lights.extend_from_slice(&0.1f32.to_le_bytes()); // atten1
    lights.extend_from_slice(&0.02f32.to_le_bytes()); // atten2
    lights.extend_from_slice(&0.5f32.to_le_bytes()); // theta
    lights.extend_from_slice(&0.9f32.to_le_bytes()); // phi

    let mut lvl_w = ChunkWriter::new();
    lvl_w.write_chunk(level_chunks::FSL_HEADER, &[14u8, 0, 0, 0]);
    lvl_w.write_chunk(level_chunks::FSL_SHADERS, &shd);
    lvl_w.write_chunk(level_chunks::FSL_VISUALS, &vis_wrap.into_bytes());
    lvl_w.write_chunk(level_chunks::FSL_LIGHT_DYNAMIC, &lights);
    let level_bytes = lvl_w.into_bytes();

    // 3. level.cform: walkable floor at y=0 + front barrier wall along z=4.0..5.0
    let cform = CformModel::from_parts(
        vec![
            [-30.0, 0.0, -30.0],
            [-30.0, 0.0, 30.0],
            [30.0, 0.0, 30.0],
            [30.0, 0.0, -30.0],
            // Wall front face at z = 4.0 (two tris facing -Z)
            [-10.0, 0.0, 4.0],
            [10.0, 0.0, 4.0],
            [10.0, 3.0, 4.0],
            [-10.0, 3.0, 4.0],
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
            CformTriangle {
                verts: [4, 5, 6],
                material: 1,
                suppress_shadows: false,
                suppress_wm: false,
                sector: 0,
            },
            CformTriangle {
                verts: [4, 6, 7],
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

#[test]
fn native_save_restores_equipped_items_and_quick_slot_references() {
    use xray_engine::ui_inventory::{EquipmentSlot, InventoryManager};
    use xray_formats::spawns::{AlifeInventoryItemState, AlifeItemAmmoState};

    let (mut save, _) = build_test_save_and_scoc();
    let actor = &mut save.entities[0];
    let ScopAlifePayload::Actor(actor_state) = &actor.payload else { unreachable!() };
    let base_item = AlifeInventoryItemState {
        base: actor_state.creature.base.clone(),
        condition: 1.0,
        upgrades: Vec::new(),
        trailing: Vec::new(),
    };
    // Physics and alive flags, conditions with one wound, and one timed booster.
    let mut client = vec![2, 1];
    client.extend([0; 16]);
    client.push(1);
    client.extend([0; 12 + 46]);
    client.push(1);
    client.extend([0; 9]);
    client.extend(b"\x03\0Player\0\0\0\0\0\0\x01\x01\x01\x01bandage\0\0\0\0");
    actor.spawn.client_data = client;

    let held_item = |id, section: &str, place: u16| {
        let mut item = ScopAlifeEntity::new(
            id, section, "", [0.0; 3], [0.0; 3], ScopAlifePayload::InventoryItem(base_item.clone()),
        );
        item.spawn.parent_id = 0;
        item.spawn.client_data.push(2);
        item.spawn.client_data.extend(place.to_le_bytes());
        item.spawn.client_data.extend(0.75f32.to_le_bytes());
        item.spawn.client_data.push(0);
        item
    };
    let rifle = held_item(2, "wpn_ak74", 0x0c31);
    let pistol = held_item(3, "wpn_pb", 0x0823);
    let bandage = held_item(4, "bandage", 3);
    let mut ammo = held_item(5, "ammo_5.45x39_fmj", 3);
    ammo.spawn.state_payload = AlifeItemAmmoState {
        item: base_item.clone(), ammo_left: 17, trailing: Vec::new(),
    }.to_bytes();
    save.entities.extend([rifle, pistol, bandage, ammo]);

    let mut inventory = InventoryManager::new_starter();
    inventory.load_from_scop(&save);
    assert_eq!(inventory.items.len(), 4);
    assert_eq!(inventory.item_in_slot(EquipmentSlot::PrimaryWeapon).unwrap().section, "wpn_ak74");
    assert!(inventory.item_in_slot(EquipmentSlot::Pistol).is_none());
    assert!(inventory.items.iter().any(|item| item.condition == 0.75 && item.ammo_elapsed == Some(17)));
    assert!(inventory.items.iter().any(|item| item.section == "bandage" && item.grid_pos.is_some()));
    assert!(inventory.use_quick_slot(0, None, None, None).is_some());
    assert!(inventory.use_quick_slot(0, None, None, None).is_none());
}

fn build_test_save_and_scoc() -> (ScopSaveFile, ScocFile) {
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
            health: 0.9,
            dynamic_out_restrictions: Vec::new(),
            dynamic_in_restrictions: Vec::new(),
            killer_id: INVALID_OBJECT_ID,
            game_death_time: 0,
            trailing: Vec::new(),
        },
        trader: AlifeTraderAbstractState {
            money: 10000,
            specific_character: "actor".to_string(),
            trader_flags: 1,
            character_profile: "actor".to_string(),
            community_index: 0,
            rank: 1000,
            reputation: 100,
            character_name: "Honey".to_string(),
            deadbody_can_take: true,
            deadbody_closed: false,
        },
        skeleton: sample_skel(),
        holder_id: INVALID_OBJECT_ID,
        trailing: Vec::new(),
    };

    let npc_state = AlifeHumanStalkerState {
        human: AlifeHumanAbstractState {
            trader: AlifeTraderAbstractState {
                money: 500,
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

    // Start player slightly above floor (eye y=2.5 -> feet y=0.85) so gravity settles onto y=0
    let scop = ScopSaveFile {
        header: ScopHeader {
            level_id: Some(0),
            level_name: "l01_escape".to_string(),
            actor_stats: ScopActorStats {
                position: [0.0, 2.5, 0.0],
                angles: [0.0, 0.0, 0.0],
                money: 10000,
                rank: 1000,
                reputation: 100,
                health: 0.9,
                actor_name: "Honey".to_string(),
                ..Default::default()
            },
            ..Default::default()
        },
        entities: vec![
            ScopAlifeEntity::new(
                0,
                "actor",
                "single_player",
                [0.0, 2.5, 0.0],
                [0.0, 0.0, 0.0],
                ScopAlifePayload::Actor(actor_state),
            ),
            ScopAlifeEntity::new(
                1,
                "stalker",
                "esc_2_12_stalker_wolf",
                [3.0, 0.0, 6.0],
                [0.0, 0.0, 0.0],
                ScopAlifePayload::Stalker(npc_state),
            ),
        ],
        graph_and_quests: ScopGraphAndQuests::default(),
        weather: ScopWeatherState {
            current_weather: "w_storm".to_string(),
            next_weather: "w_storm".to_string(),
            cycle_time_sec: 43_200.0,
            ..Default::default()
        },
        extra_chunks: Vec::new(),
    };

    let mut peer = ScocPeerState::default();
    peer.inventory.push(ScocInventoryItem::new("wpn_val", 1.0));
    let scoc = ScocFile::from_peer_state(&peer);
    (scop, scoc)
}

#[test]
fn test_full_dx12_7stage_render_graph_and_capsule_physics_loop() {
    let temp = make_temp_dir("full_pipeline");
    let level_dir = temp.0.join("levels").join("l01_escape");
    let weather_dir = temp.0.join("configs").join("environment").join("weathers");
    let save_dir = temp.0.join("savedgames");
    let mesh_dir = temp.0.join("meshes/actors/stalker_neutral");
    fs::create_dir_all(&level_dir).unwrap();
    fs::create_dir_all(&weather_dir).unwrap();
    fs::create_dir_all(&save_dir).unwrap();
    fs::create_dir_all(&mesh_dir).unwrap();
    fs::write(mesh_dir.join("stalker_neutral_3.ogf"), build_spawn_visual()).unwrap();

    let (level_bytes, geom_bytes, cform_bytes) = build_gpu_test_level_bytes();
    fs::write(level_dir.join("level"), level_bytes).unwrap();
    fs::write(level_dir.join("level.geom"), geom_bytes).unwrap();
    fs::write(level_dir.join("level.cform"), cform_bytes).unwrap();
    fs::write(
        weather_dir.join("w_storm.ltx"),
        "[12:00:00]\nambient_color = 0.25, 0.28, 0.32\nsky_color = 0.55, 0.65, 0.85\nsun_color = 1.0, 0.9, 0.75\nrain_density = 0.8\nwind_velocity = 12.0\nwind_direction = 0.6\n",
    )
    .unwrap();

    let (scop, scoc) = build_test_save_and_scoc();
    fs::write(save_dir.join("honey_host.scop"), scop.to_bytes().unwrap()).unwrap();
    let scoc_bytes = scoc.to_bytes().unwrap();

    let mut vfs = VirtualFileSystem::new();
    vfs.mount_loose_dir(&temp.0);

    let mut runtime = EngineRuntime::new(
        EngineConfig {
            window: WindowConfig {
                title: "TestDx12FullRenderPipeline".to_string(),
                width: 640,
                height: 360,
                visible: false,
            },
            vsync: false,
            ..Default::default()
        },
        true,
    )
    .expect("create DX12 runtime on RTX 4060");

    let vfs = Arc::new(vfs);
    runtime.attach_vfs(Arc::clone(&vfs));
    runtime
        .load_world_and_save_from_vfs(
            &vfs,
            &LevelLoadConfig {
                level_dir: "levels/l01_escape".to_string(),
                resolve_visual_meshes: true,
                resolve_textures: true,
                load_fastpath_geom: false,
                load_optional_aux: false,
                load_weather: true,
            },
            Some("savedgames/honey_host.scop"),
        )
        .expect("stream world and .scop onto DX12 runtime");

    let peer_id = runtime
        .alife
        .as_mut()
        .unwrap()
        .attach_peer_scoc_bytes(1, "Falcon", 0xFA1C_0001, &scoc_bytes)
        .expect("attach peer avatar");
    if let Some(peer) = runtime.alife.as_mut().unwrap().remote_peers.get_mut(&peer_id) {
        peer.position = [1.5, 0.0, 5.0];
    }

    // Feed wallmark & particle buffers into RenderBridge so Stage (e) covers decals & particles too
    if let Some(rb) = &mut runtime.render_bridge {
        rb.set_wallmarks(vec![
            GpuWallmarkVertex {
                position: [-0.5, 0.02, 2.0],
                color: 0xCC22_2222,
                uv: [0.0, 0.0],
            },
            GpuWallmarkVertex {
                position: [0.5, 0.02, 2.0],
                color: 0xCC22_2222,
                uv: [1.0, 0.0],
            },
            GpuWallmarkVertex {
                position: [0.0, 0.02, 2.8],
                color: 0xCC22_2222,
                uv: [0.5, 1.0],
            },
        ]);
        rb.push_particle(GpuParticleInstance {
            position: [0.5, 1.2, 3.0],
            rotation: 0.0,
            size: [0.4, 0.4],
            color: [1.0, 0.85, 0.6, 0.7],
            texture_index: u32::MAX,
            blend_mode: 0,
            ..Default::default()
        });
    }

    // Hold W + D to walk toward barrier z=4.0 while gravity settles player onto y=0 floor
    runtime.input.begin_frame();
    runtime.input.on_key_down(EngineInput::VK_KEY_W);
    runtime.input.on_key_down(EngineInput::VK_KEY_D);

    let mut settled_on_floor = false;
    for _ in 0..45 {
        std::thread::sleep(std::time::Duration::from_millis(16));
        runtime.step_frame().expect("step DX12 frame");
        if runtime.physics.grounded && runtime.physics.feet_position[1].abs() < 0.02 {
            settled_on_floor = true;
        }
    }

    assert!(
        settled_on_floor,
        "capsule controller should fall onto y=0 floor via gravity: feet={:?}, grounded={}",
        runtime.physics.feet_position,
        runtime.physics.grounded
    );
    assert!(
        (runtime.camera.position[1] - 1.65).abs() < 0.05,
        "eye height should match grounded capsule (1.65m): cam={:?}",
        runtime.camera.position
    );

    let stats = runtime
        .render_bridge
        .as_ref()
        .expect("render bridge present")
        .last_stats;

    assert!(stats.scene_depth_copied, "expected opaque scene depth to be reused by forward effects");
    assert_eq!(stats.early_z_draws, 0, "expected one geometry pass to fill scene depth");
    assert!(stats.gbuffer_static_draws >= 1, "expected G-Buffer static draw");
    assert!(
        stats.gbuffer_dynamic_draws >= 3,
        "expected two submeshes from the mounted NPC model and one co-op proxy, got {}",
        stats.gbuffer_dynamic_draws
    );
    assert_eq!(stats.csm_cascade_draws.len(), 3);
    assert!(
        stats.csm_cascade_draws.iter().sum::<u32>() >= 3,
        "expected 3-split CSM shadow draws, got {:?}",
        stats.csm_cascade_draws
    );
    assert!(stats.combine_csm_active, "expected CSM active in deferred combine");
    assert!(stats.combine_light_count >= 1, "expected dynamic point light in combine");
    assert!(stats.postprocess_executed, "expected filmic postprocess resolve");
    assert!(stats.sky_drawn, "expected sky dome draw");
    assert!(stats.clouds_drawn, "expected cloud dome draw");
    assert!(
        stats.hud_weapon_draws >= 3,
        "expected first-person HUD weapon + hand draws, got {}",
        stats.hud_weapon_draws
    );
    assert!(stats.ui_elements_drawn >= 1, "expected UI HUD overlay draws");

    // Verify clean GPU fence sync
    runtime
        .renderer
        .as_mut()
        .unwrap()
        .wait_for_gpu()
        .expect("DX12 wait_for_gpu clean");

    // Ensure host ALifeManager entity was kept synced with capsule position
    let actor = runtime.alife.as_ref().unwrap().host_actor().unwrap();
    assert!((actor.position[1] - 1.65).abs() < 0.05);
    assert!(AlifeManager::default().online_entity_count() == 0);
}
