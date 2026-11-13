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
