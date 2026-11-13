use std::path::Path;
use xray_engine::world_loader::{LevelLoadConfig, WorldLoader};
use xray_vfs::VirtualFileSystem;

#[test]
fn test_verify_world_loader_resolves_terrain_textures() {
    let games_root = Path::new(r"C:\Games");
    if !games_root.join("fsgame.ltx").exists() {
        return;
    }

    let vfs = VirtualFileSystem::mount_game_root(games_root).expect("mount VFS");
    let level_cfg = LevelLoadConfig {
        level_dir: "levels\\l01_escape".to_string(),
        resolve_visual_meshes: true,
        resolve_textures: true,
        load_fastpath_geom: false,
        load_optional_aux: true,
        load_weather: true,
    };

    let world = WorldLoader::load_from_vfs(&vfs, &level_cfg).expect("load l01_escape");

    let mut total_meshes = 0usize;
    let mut meshes_with_texture_links = 0usize;
    let mut total_texture_links = 0usize;
    let mut total_textures_valid = 0usize;
    let mut total_textures_missing = 0usize;

    let mut missing_texture_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    let mut parsed_ok = 0usize;
    let mut parsed_err = 0usize;
    let mut parse_errors: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();

    let mut unique_textures: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for visual in &world.visual_meshes {
        total_meshes += 1;
        if !visual.texture_links.is_empty() {
            meshes_with_texture_links += 1;
        }
        for link in &visual.texture_links {
            total_texture_links += 1;
            if vfs.exists(&link.diffuse_dds) {
                total_textures_valid += 1;
                unique_textures.insert(link.diffuse_dds.clone());
            } else {
                total_textures_missing += 1;
                missing_texture_names.insert(link.diffuse_name.clone());
            }
        }
    }

    println!(
        "[diagnostic] Total meshes: {}, With links: {}, Total links: {}, Valid: {}, Missing: {}",
        total_meshes, meshes_with_texture_links, total_texture_links, total_textures_valid, total_textures_missing
    );
    println!("[diagnostic] Unique valid textures to parse: {}", unique_textures.len());

    for path in &unique_textures {
        match vfs.read_bytes(path) {
            Ok(bytes) => {
                match xray_render_dx12::texture::parse_dds(&bytes) {
                    Ok(_) => {
                        parsed_ok += 1;
                    }
                    Err(e) => {
                        parsed_err += 1;
                        *parse_errors.entry(format!("{:?} on {}", e, path)).or_insert(0) += 1;
                    }
                }
            }
            Err(e) => {
                parsed_err += 1;
                *parse_errors.entry(format!("read_bytes failed: {:?} on {}", e, path)).or_insert(0) += 1;
            }
        }
    }

    println!(
        "[diagnostic] DDS read results: {} ok, {} errors",
        parsed_ok, parsed_err
    );

    // Test spawn positions in level:
    if let Some(ref spawn) = world.level_spawn {
        println!("[diagnostic] Spawn packets in level: {}", spawn.spawns.len());
        for pkt in &spawn.spawns {
            if pkt.section.to_ascii_lowercase().contains("actor") || pkt.id == 0 {
                println!(
                    "[diagnostic] Actor packet found: id={}, section='{}', pos={:?}, angle={:?}",
                    pkt.id, pkt.section, pkt.position, pkt.angle
                );
            }
        }
    } else {
        println!("[diagnostic] world.level_spawn is NONE!");
    }
    if !parse_errors.is_empty() {
        println!("[diagnostic] DDS Parse errors:");
        for (err, count) in parse_errors.iter().take(15) {
            println!("   {} times: {}", count, err);
        }
    }
}

#[test]
fn test_verify_frustum_culling_against_level_meshes() {
    use xray_formats::hom::{frustum_planes, ClipFrustum};
    use xray_formats::ogf::matrix_mul_44;
    use xray_engine::camera::FirstPersonCamera;

    let games_root = Path::new(r"C:\Games");
    if !games_root.join("fsgame.ltx").exists() {
        return;
    }

    let vfs = VirtualFileSystem::mount_game_root(games_root).expect("mount VFS");
    let level_cfg = LevelLoadConfig {
        level_dir: "levels\\l01_escape".to_string(),
        resolve_visual_meshes: true,
        resolve_textures: false,
        load_fastpath_geom: false,
        load_optional_aux: false,
        load_weather: true,
    };

    let mut world = WorldLoader::load_from_vfs(&vfs, &level_cfg).expect("load l01_escape");

    let mut cam = FirstPersonCamera::new(1280, 720);
    cam.position = [-32.844086, -6.419943, -16.873344]; // Spawn point in Sidorovich bunker
    cam.pitch = 0.0;
    cam.yaw = 0.0;

    let view = cam.view_matrix();
    let proj = cam.projection_matrix();

    let vp_pv = matrix_mul_44(&proj, &view);
    let vp_vp = matrix_mul_44(&view, &proj);

    let frustum_pv = ClipFrustum::from_view_proj_dx(&vp_pv, frustum_planes::ALL);
    let frustum_vp = ClipFrustum::from_view_proj_dx(&vp_vp, frustum_planes::ALL);

    let mut pass_pv = 0usize;
    let mut pass_vp = 0usize;
    let mut total_with_bsphere = 0usize;

    for visual in &world.visual_meshes {
        let r = visual.bsphere.radius;
        let c = visual.bsphere.center;
        if r > 1e-4 {
            total_with_bsphere += 1;
            if frustum_pv.test_sphere(c, r) {
                pass_pv += 1;
            }
            if frustum_vp.test_sphere(c, r) {
                pass_vp += 1;
            }
        }
    }

    println!(
        "[diagnostic] Frustum test from spawn [-32.84, -6.42, -16.87]: total_with_bsphere={}, pass_pv(proj*view)={}, pass_vp(view*proj)={}",
        total_with_bsphere, pass_pv, pass_vp
    );

    // Also test spawn from outside, e.g. rookie village [30.43, 0.48, -326.10]
    cam.position = [30.43, 0.48, -326.10];
    let view2 = cam.view_matrix();
    let proj2 = cam.projection_matrix();
    let vp_pv2 = matrix_mul_44(&proj2, &view2);
    let vp_vp2 = matrix_mul_44(&view2, &proj2);
    let frustum_pv2 = ClipFrustum::from_view_proj_dx(&vp_pv2, frustum_planes::ALL);
    let frustum_vp2 = ClipFrustum::from_view_proj_dx(&vp_vp2, frustum_planes::ALL);

    let mut pass_pv2 = 0usize;
    let mut pass_vp2 = 0usize;
    for visual in &world.visual_meshes {
        let r = visual.bsphere.radius;
        let c = visual.bsphere.center;
        if r > 1e-4 {
            if frustum_pv2.test_sphere(c, r) {
                pass_pv2 += 1;
            }
            if frustum_vp2.test_sphere(c, r) {
                pass_vp2 += 1;
            }
        }
    }

    println!(
        "[diagnostic] Frustum test from [30.43, 0.48, -326.10]: pass_pv(proj*view)={}, pass_vp(view*proj)={}",
        pass_pv2, pass_vp2
    );

    // Also check lighting env:
    let env_step = world.step_weather(0.016, cam.position, cam.forward(), cam.far_z);
    println!("[diagnostic] weather after step_weather is Some: {}", env_step.is_some());
    if let Some(e) = env_step {
        println!("[diagnostic] sun_color: {:?}, ambient_color: {:?}, hemi_color: {:?}, sun_dir: {:?}", e.sun_color, e.ambient_color, e.hemi_color, e.sun_dir);
        println!("[diagnostic] effective_ambient_rgb(0.5): {:?}", e.effective_ambient_rgb(0.5));
    }

    // Audio diagnostic:
    println!("[diagnostic] Check weapon sounds in VFS:");
    println!("[diagnostic] Check ambient sound files in VFS:");
    if let Ok(files) = vfs.list_files("sounds\\ambient") {
        println!("[diagnostic] sounds\\ambient files count: {}, first 10: {:?}", files.len(), &files[..files.len().min(10)]);
    }

    // UI textures in VFS:
    let ui_checks = [
        "textures\\ui\\ui_hud.dds",
        "textures\\ui\\ui_icon_equipment.dds",
        "textures\\ui\\ui_common.dds",
        "textures\\ui\\ui_font_letterica16_1024.dds",
        "textures\\ui\\ui_font_letterica18_1024.dds",
        "textures\\ui\\ui_font_graffiti19_1024.dds",
        "configs\\fonts.ltx",
        "configs\\ui\\ui_hud.xml",
    ];
    println!("[diagnostic] Check UI assets in VFS:");
    for p in ui_checks {
        println!("   {} exists: {}", p, vfs.exists(p));
    }

    use xray_formats::vfs_bridge::load_font_from_vfs;
    match xray_formats::vfs_bridge::load_ltx_tree_from_vfs(&vfs, "$game_configs$/fonts.ltx", &[]) {
        Ok(fonts_doc) => {
            let sections: Vec<&str> = fonts_doc.sections.keys().map(|s| s.as_str()).collect();
            println!("[diagnostic] fonts_doc sections count: {}, first 10: {:?}", sections.len(), &sections[..sections.len().min(10)]);
            for sec in [
                "ui_font_letterica16_russian",
                "ui_font_arial_14",
                "ui_font_letterica18_russian",
                "hud_font_medium",
                "ui_font_graffiti19_russian",
                "ui_font_graff_19",
            ] {
                match load_font_from_vfs(&vfs, &fonts_doc, sec, 720) {
                    Ok((cfg, atlas)) => {
                        let dds_exists = vfs.exists(&cfg.texture_dds_vpath);
                        println!(
                            "[diagnostic] font '{}': resolved={}, texture_dds_vpath={}, exists={}, line_height={}",
                            sec, cfg.resolved_texture, cfg.texture_dds_vpath, dds_exists, atlas.line_height
                        );
                    }
                    Err(e) => println!("[diagnostic] font '{}' load error: {:?}", sec, e),
                }
            }

            let font_bytes = vfs.read_bytes("textures\\ui\\ui_font_letter_16_1024.dds").expect("read font dds");
            let pf_flags = u32::from_le_bytes([font_bytes[80], font_bytes[81], font_bytes[82], font_bytes[83]]);
            let pf_fourcc = u32::from_le_bytes([font_bytes[84], font_bytes[85], font_bytes[86], font_bytes[87]]);
            let pf_rgb_bits = u32::from_le_bytes([font_bytes[88], font_bytes[89], font_bytes[90], font_bytes[91]]);
            let pf_r_mask = u32::from_le_bytes([font_bytes[92], font_bytes[93], font_bytes[94], font_bytes[95]]);
            let pf_g_mask = u32::from_le_bytes([font_bytes[96], font_bytes[97], font_bytes[98], font_bytes[99]]);
            let pf_b_mask = u32::from_le_bytes([font_bytes[100], font_bytes[101], font_bytes[102], font_bytes[103]]);
            let pf_a_mask = u32::from_le_bytes([font_bytes[104], font_bytes[105], font_bytes[106], font_bytes[107]]);
            let fourcc_str = String::from_utf8_lossy(&font_bytes[84..88]);
            println!(
                "[diagnostic] Font DDS header: flags=0x{:X}, fourcc=0x{:X} ('{}'), bits={}, masks=(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X})",
                pf_flags, pf_fourcc, fourcc_str, pf_rgb_bits, pf_r_mask, pf_g_mask, pf_b_mask, pf_a_mask
            );
            let parsed_font = xray_render_dx12::texture::parse_dds(&font_bytes);
            println!("[diagnostic] parse_dds on font: {:?}", parsed_font.as_ref().map(|d| (d.width, d.height, d.format)));

            let hud_bytes = vfs.read_bytes("textures\\ui\\ui_hud.dds").expect("read ui_hud");
            let parsed_hud = xray_render_dx12::texture::parse_dds(&hud_bytes);
            println!("[diagnostic] parse_dds on ui_hud: {:?}", parsed_hud.as_ref().map(|d| (d.width, d.height, d.format)));

            let eq_bytes = vfs.read_bytes("textures\\ui\\ui_icon_equipment.dds").expect("read eq");
            let parsed_eq = xray_render_dx12::texture::parse_dds(&eq_bytes);
            println!("[diagnostic] parse_dds on ui_icon_equipment: {:?}", parsed_eq.as_ref().map(|d| (d.width, d.height, d.format)));
        }
        Err(e) => println!("[diagnostic] failed to load fonts.ltx: {:?}", e),
    }
}
