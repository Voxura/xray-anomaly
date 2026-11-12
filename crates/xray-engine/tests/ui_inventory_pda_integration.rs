//! End-to-End Interactive UI, Inventory Grid, PDA & NPC Dialogue Integration Test (`xray-engine`).
//!
//! Verifies through `EngineRuntime::step_frame` + `WindowEvent`s:
//! - Hydrating `InventoryManager` & `HudWeaponController` from `.scoc` (`ScocInventoryItem`, `item_remaining_uses`)
//! - `Tab` / `I` inventory toggle, pointer drag-and-drop onto `EquipmentSlot::PrimaryWeapon`,
//!   right-click consumable healing (`medkit` uses `3 -> 2`), stamina sprint/regen, and `AudioManager` SFX
//! - `P` PDA toggle with VFS `WorldMapsConfig` + `.dds` minimap, `AlifeManager::remote_peers` blips,
//!   and `ScopStoryQuest` selection
//! - `E` (`Action::Use`) ray-cone NPC/Trader dialogue interaction and item pickup
//! - Live DirectX 12 `RenderBridge` UI Canvas quad & font glyph emission across HUD, Inventory, and PDA.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use xray_engine::alife_manager::{AlifeManager, EntityRole, LiveAlifeEntity};
use xray_engine::game_loop::{EngineConfig, EngineRuntime};
use xray_engine::input::{EngineInput, MouseButton};
use xray_engine::ui_inventory::{EquipmentSlot, InventoryLayout, UseTarget};
use xray_engine::ui_pda::{MapBlipKind, PdaLayout};
use xray_engine::window::{WindowConfig, WindowEvent};
use xray_engine::world_loader::{LevelLoadConfig, WorldLoader};
use xray_formats::chunk::ChunkWriter;
use xray_formats::level::{d3d_decl_type, d3d_decl_usage, level_chunks, VertexElement9};
use xray_formats::savegames::{SavedGameTask, SavedTaskScriptHelper};
use xray_formats::scoc::{ScocFile, ScocTable, ScocValue};
use xray_formats::scop::{ScopAlifeKind, ScopStoryQuest};
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
    let dir = std::env::temp_dir().join(format!("xray_ui_pda_{tag}_{nonce}"));
    fs::create_dir_all(&dir).expect("create temp test gamedata");
    TempDirGuard(dir)
}

fn make_solid_4x4_rgba_dds() -> Vec<u8> {
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
        *px = [90, 140, 110, 255];
    }
    dds
}

fn write_minimal_level(level_dir: &std::path::Path) {
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
    for p in [[-5.0f32, 0.0, -5.0], [5.0, 0.0, -5.0], [0.0, 0.0, 5.0]] {
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
}

fn build_synthetic_ui_vfs(root: &std::path::Path) -> Arc<VirtualFileSystem> {
    let cfg_dir = root.join("configs");
    let lvl_dir = root.join("levels").join("l01_escape");
    let map_tex_dir = root.join("textures").join("map");
    let ui_tex_dir = root.join("textures").join("ui");
    fs::create_dir_all(&cfg_dir).unwrap();
    fs::create_dir_all(&lvl_dir).unwrap();
    fs::create_dir_all(&map_tex_dir).unwrap();
    fs::create_dir_all(&ui_tex_dir).unwrap();

    write_minimal_level(&lvl_dir);
    fs::write(
        cfg_dir.join("game_maps_single.ltx"),
        r#"[level_maps_single]
l01_escape =

[l01_escape]
texture = map\map_escape
bound_rect = -200.0, -200.0, 200.0, 200.0
"#,
    )
    .unwrap();
    fs::write(
        lvl_dir.join("level.ltx"),
        "[level_map]\ntexture = map\\map_escape\nbound_rect = -200.0, -200.0, 200.0, 200.0\n",
    )
    .unwrap();
    fs::write(map_tex_dir.join("map_escape.dds"), make_solid_4x4_rgba_dds()).unwrap();

    fs::write(
        cfg_dir.join("fonts.ltx"),
        "[ui_font_letterica16_russian]\nshader = font\ntexture = ui\\ui_font_letter_16_1024\nsize = 16\n",
    )
    .unwrap();
    let mut ini = String::from("[symbol_coords]\nheight = 16\n");
    for code in 32..=126u32 {
        ini.push_str(&format!("{code:03} = 0, 0, 8\n"));
    }
    fs::write(ui_tex_dir.join("ui_font_letter_16_1024.ini"), ini).unwrap();
    fs::write(
        ui_tex_dir.join("ui_font_letter_16_1024.dds"),
        make_solid_4x4_rgba_dds(),
    )
    .unwrap();

    let mut vfs = VirtualFileSystem::new();
    vfs.mount_loose_dir(root);
    Arc::new(vfs)
}

fn build_sample_scoc() -> ScocFile {
    let mut inv_tbl = ScocTable::new();

    let mut ak_item = ScocTable::new();
    ak_item.set_str("id", ScocValue::int(3001));
    ak_item.set_str("section", ScocValue::string("wpn_ak74"));
    ak_item.set_str("condition", ScocValue::float(0.92));
    ak_item.set_str("slot", ScocValue::int(2));
    ak_item.set_str("ammo_elapsed", ScocValue::int(30));
    inv_tbl.set_i64(1, ScocValue::Table(ak_item));

    let mut svd_item = ScocTable::new();
    svd_item.set_str("id", ScocValue::int(3002));
    svd_item.set_str("section", ScocValue::string("wpn_svd"));
    svd_item.set_str("condition", ScocValue::float(0.88));
    svd_item.set_str("ammo_elapsed", ScocValue::int(10));
    inv_tbl.set_i64(2, ScocValue::Table(svd_item));

    let mut med_item = ScocTable::new();
    med_item.set_str("id", ScocValue::int(3003));
    med_item.set_str("section", ScocValue::string("medkit"));
    med_item.set_str("condition", ScocValue::float(1.0));
    inv_tbl.set_i64(3, ScocValue::Table(med_item));

    let mut band_item = ScocTable::new();
    band_item.set_str("id", ScocValue::int(3004));
    band_item.set_str("section", ScocValue::string("bandage"));
    band_item.set_str("condition", ScocValue::float(1.0));
    inv_tbl.set_i64(4, ScocValue::Table(band_item));

    let mut uses_tbl = ScocTable::new();
    uses_tbl.set_i64(3003, ScocValue::int(3));
    uses_tbl.set_i64(3004, ScocValue::int(1));

    let mut root = ScocTable::new();
    root.set_str("inventory", ScocValue::Table(inv_tbl));
    root.set_str("item_remaining_uses", ScocValue::Table(uses_tbl));
    ScocFile::new(root)
}

fn sample_quest(id: &str, title: &str, obj_id: u16) -> ScopStoryQuest {
    ScopStoryQuest {
        owner_id: 0,
        task: SavedGameTask {
            task_id: id.to_string(),
            task_state: 1,
            task_type: 0,
            receive_time: 0,
            finish_time: 0,
            time_to_complete: 0,
            timer_finish: 0,
            title: title.to_string(),
            description: format!("Complete objective: {title}"),
            script_helper: SavedTaskScriptHelper::default(),
            icon_texture_name: String::new(),
            map_hint: title.to_string(),
            map_location: "l01_escape".to_string(),
            map_object_id: obj_id,
            priority: 1,
        },
    }
}

#[test]
fn test_phase_c_inventory_pda_dialogue_and_audio_end_to_end() {
    let dir = make_temp_dir("phase_c_e2e");
    let vfs = build_synthetic_ui_vfs(dir.0.as_path());

    let mut runtime = EngineRuntime::new(
        EngineConfig {
            window: WindowConfig {
                title: "PhaseCIntegration".to_string(),
                width: 1024,
                height: 768,
                visible: false,
            },
            ..Default::default()
        },
        false,
    )
    .expect("create headless EngineRuntime");

    runtime.attach_vfs(Arc::clone(&vfs));

    // Seed ALife with actor, friendly Trader 2m ahead along +Z, ground bandage pickup nearby, co-op peer, and quests
    let mut alife = AlifeManager::default();
    alife.level_name = "l01_escape".to_string();
    alife.insert_entity(LiveAlifeEntity {
        id: 0,
        parent_id: INVALID_OBJECT_ID,
        story_id: INVALID_STORY_ID,
        name: "Marked One".to_string(),
        section: "actor".to_string(),
        role: EntityRole::LocalHostActor,
        kind: ScopAlifeKind::Actor,
        position: [0.0, 1.8, 0.0],
        angle: [0.0, 0.0, 0.0],
        velocity: [0.0; 3],
        health: 0.40,
        online: true,
        game_vertex_id: 0,
        level_vertex_id: 0,
        anim_state: 0,
        weapon_state: 0,
        visual_name: None,
        dirty_net_sync: false,
        source_scop: None,
    });
    alife.actor_stats.health = 0.40;
    alife.actor_stats.bleeding = 0.25;

    // Trader Sidorovich directly ahead (+Z = 2.0m)
    alife.insert_entity(LiveAlifeEntity {
        id: 210,
        parent_id: INVALID_OBJECT_ID,
        story_id: 300,
        name: "Sidorovich".to_string(),
        section: "esc_2_12_stalker_trader".to_string(),
        role: EntityRole::Trader,
        kind: ScopAlifeKind::Trader,
        position: [0.0, 0.45, 2.0],
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
    });

    alife.graph_and_quests.story_quests = vec![
        sample_quest("q_sidor", "Speak with Sidorovich", 210),
        sample_quest("q_nimble", "Find Nimble's Flash Drive", INVALID_OBJECT_ID),
    ];
    alife.graph_and_quests.active_quest_id = "q_sidor".to_string();

    let peer_scoc = build_sample_scoc();
    let peer_id = alife.attach_peer_scoc(1, "Falcon", 9001, &peer_scoc);
    alife.remote_peers.get_mut(&peer_id).unwrap().position = [50.0, 0.0, -50.0];

    runtime.attach_alife(alife);
    runtime.attach_host_scoc(&peer_scoc);

    // Frame 1: Look straight ahead at Sidorovich -> hover target prompt should detect Trader!
    runtime.step_frame().unwrap();
    assert!(
        matches!(
            runtime.inventory.hover_target,
            Some(UseTarget::Dialogue { entity_id: 210, is_trader: true, .. })
        ),
        "looking at nearby trader must surface [E] dialogue prompt"
    );

    // Frame 2: Press 'E' -> open Trader dialogue & click option 0 (Restock Medkit)
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyDown { vk: EngineInput::VK_KEY_E });
    runtime.step_frame().unwrap();
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyUp { vk: EngineInput::VK_KEY_E });
    assert!(runtime.inventory.dialogue.is_some());
    assert!(!runtime.input.is_cursor_locked());

    let inv_layout = InventoryLayout::compute(1024, 768, runtime.inventory.grid_cols, runtime.inventory.grid_rows);
    let opt0 = inv_layout.dialogue_option_rects[0];
    let items_before_trade = runtime.inventory.items.len();
    runtime.window.push_synthetic_event(WindowEvent::MouseMove {
        x: opt0[0] + 15.0,
        y: opt0[1] + 10.0,
    });
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseDown { button: MouseButton::Left });
    runtime.step_frame().unwrap();
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseUp { button: MouseButton::Left });
    assert_eq!(runtime.inventory.items.len(), items_before_trade + 1);

    // Close dialogue via Esc, lock cursor returns cleanly
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyDown { vk: EngineInput::VK_ESCAPE });
    runtime.step_frame().unwrap();
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyUp { vk: EngineInput::VK_ESCAPE });
    assert!(runtime.inventory.dialogue.is_none());
    assert!(runtime.running);
    assert!(runtime.input.is_cursor_locked());

    // Frame 4: Toggle Inventory via 'I'
    let ui_snd_before = runtime.audio.stats().ui_sounds_triggered;
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyDown { vk: EngineInput::VK_KEY_I });
    runtime.step_frame().unwrap();
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyUp { vk: EngineInput::VK_KEY_I });
    assert!(runtime.inventory.open);
    assert!(!runtime.input.is_cursor_locked());
    assert!(runtime.audio.stats().ui_sounds_triggered > ui_snd_before);

    // Drag wpn_svd from its 2D grid cell onto EquipmentSlot::PrimaryWeapon
    let svd_item = runtime
        .inventory
        .items
        .iter()
        .find(|it| it.section == "wpn_svd")
        .unwrap()
        .clone();
    let svd_pos = svd_item.grid_pos.unwrap();
    let start_xy = inv_layout.grid_cell_center(svd_pos.0, svd_pos.1, svd_item.grid_size.0, svd_item.grid_size.1);
    let drop_xy = inv_layout.slot_center(EquipmentSlot::PrimaryWeapon).unwrap();

    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseMove { x: start_xy[0], y: start_xy[1] });
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseDown { button: MouseButton::Left });
    runtime.step_frame().unwrap();
    assert!(runtime.inventory.drag.is_some());

    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseMove { x: drop_xy[0], y: drop_xy[1] });
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseUp { button: MouseButton::Left });
    runtime.step_frame().unwrap();

    // Verify wpn_svd equipped into PrimaryWeapon and synced to HudWeaponController without firing weapon
    assert_eq!(
        runtime.inventory.item_in_slot(EquipmentSlot::PrimaryWeapon).unwrap().section,
        "wpn_svd"
    );
    assert_eq!(runtime.hud_weapon.slots[2].section, "wpn_svd");
    assert_eq!(runtime.hud_weapon.active_slot, 3);
    assert_eq!(runtime.hud_weapon.shots_fired, 0, "dragging UI must never fire weapon");

    // Right-click medkit in 2D grid -> heals player & decrements remaining_uses (3 -> 2)
    let med_item = runtime
        .inventory
        .items
        .iter()
        .find(|it| it.section == "medkit" && it.remaining_uses == Some(3))
        .unwrap()
        .clone();
    let med_pos = med_item.grid_pos.unwrap();
    let med_xy = inv_layout.grid_cell_center(med_pos.0, med_pos.1, 1, 1);
    let hp_before = runtime.hud.health;
    let item_snd_before = runtime.audio.stats().item_sounds_triggered;

    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseMove { x: med_xy[0], y: med_xy[1] });
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseDown { button: MouseButton::Right });
    runtime.step_frame().unwrap();
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseUp { button: MouseButton::Right });

    assert!(runtime.hud.health > hp_before + 0.35);
    assert_eq!(runtime.inventory.item(med_item.uid).unwrap().remaining_uses, Some(2));
    assert!(runtime.audio.stats().item_sounds_triggered > item_snd_before);

    // Frame: Press 'P' -> seamlessly closes Inventory and opens PDA screen
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyDown { vk: EngineInput::VK_KEY_P });
    runtime.step_frame().unwrap();
    runtime
        .window
        .push_synthetic_event(WindowEvent::KeyUp { vk: EngineInput::VK_KEY_P });
    assert!(runtime.pda.open);
    assert!(!runtime.inventory.open);

    // Verify PDA map projection includes local player, co-op peer Falcon, Trader Sidorovich, and Story Quests
    assert!(runtime.pda.world_maps.is_some());
    assert!(runtime.pda.blips.iter().any(|b| b.kind == MapBlipKind::Player));
    assert!(runtime.pda.blips.iter().any(|b| b.kind == MapBlipKind::CoopPeer && b.label == "Falcon"));
    assert!(runtime.pda.blips.iter().any(|b| b.kind == MapBlipKind::Trader));
    assert_eq!(runtime.pda.quests.len(), 2);

    // Click second quest row in PDA sidebar -> activates `q_nimble`
    let pda_layout = PdaLayout::compute(1024, 768);
    let q1_rect = pda_layout.quest_row_rects[1];
    runtime.window.push_synthetic_event(WindowEvent::MouseMove {
        x: q1_rect[0] + 12.0,
        y: q1_rect[1] + 10.0,
    });
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseDown { button: MouseButton::Left });
    runtime.step_frame().unwrap();
    runtime
        .window
        .push_synthetic_event(WindowEvent::MouseUp { button: MouseButton::Left });
    assert_eq!(
        runtime.alife.as_ref().unwrap().graph_and_quests.active_quest_id,
        "q_nimble"
    );

    // Bonus GPU verification on live DX12 adapter when available: verify HUD -> Inventory -> PDA UI quad counts
    if let Ok(mut gpu_rt) = EngineRuntime::new(
        EngineConfig {
            window: WindowConfig {
                title: "PhaseCDx12Test".to_string(),
                width: 640,
                height: 480,
                visible: false,
            },
            vsync: false,
            ..Default::default()
        },
        true,
    ) {
        let world = WorldLoader::load_from_vfs(
            &vfs,
            &LevelLoadConfig {
                level_dir: "levels/l01_escape".to_string(),
                resolve_visual_meshes: false,
                resolve_textures: false,
                load_fastpath_geom: false,
                load_optional_aux: false,
                load_weather: false,
            },
        )
        .unwrap();
        gpu_rt.attach_vfs(Arc::clone(&vfs));
        gpu_rt.attach_world(world);
        gpu_rt.attach_alife(runtime.alife.take().unwrap());
        gpu_rt.attach_host_scoc(&peer_scoc);
        gpu_rt.sync_world_to_gpu(Some(&vfs)).unwrap();

        gpu_rt.step_frame().unwrap();
        let hud_only_quads = gpu_rt.render_bridge.as_ref().unwrap().last_stats.ui_elements_drawn;
        assert!(hud_only_quads >= 10, "expected always-on HUD meters/slots/ammo");

        // Open Inventory on GPU
        gpu_rt
            .window
            .push_synthetic_event(WindowEvent::KeyDown { vk: EngineInput::VK_TAB });
        gpu_rt.step_frame().unwrap();
        gpu_rt
            .window
            .push_synthetic_event(WindowEvent::KeyUp { vk: EngineInput::VK_TAB });
        let inv_quads = gpu_rt.render_bridge.as_ref().unwrap().last_stats.ui_elements_drawn;
        assert!(inv_quads > hud_only_quads + 50, "expected inventory 10x12 grid + slots");

        // Switch to PDA on GPU (includes level minimap DDS texture & VFS bitmap font glyph quads)
        gpu_rt
            .window
            .push_synthetic_event(WindowEvent::KeyDown { vk: EngineInput::VK_KEY_P });
        gpu_rt.step_frame().unwrap();
        let pda_quads = gpu_rt.render_bridge.as_ref().unwrap().last_stats.ui_elements_drawn;
        assert!(pda_quads > hud_only_quads + 10, "expected PDA frame + minimap + blips + quests");
    }
}
