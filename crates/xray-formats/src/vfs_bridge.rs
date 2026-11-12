//! Bridges between `xray-vfs` (`VirtualFileSystem`), `xray-core` (`.ltx` / DLTX),
//! and `xray-formats` (`OgfModel`, `OmfMotions`, `TextureParams`, `GameMaterialLibrary`,
//! `CformModel`, `LevelDetails`, `LevelData`, `LevelGeometry`).

use xray_core::ltx_loader::{collect_ltx_sources, glob_matches_ascii, LtxFileProvider, OwnedLtxDocument};
use xray_vfs::{normalize_virtual_path, normalize_virtual_prefix, VfsError, VirtualFileSystem};

use crate::ai_graph::{GameGraph, LevelCrossTable, LevelGraph};
use crate::cform::CformModel;
use crate::chunk::FormatError;
use crate::details::{DetailMesh, LevelDetails};
use crate::envelopes::{ObjectMotion, ObjectMotionLibrary, PostprocessEffect};
use crate::fonts::{font_atlas_ini_vpath, resolve_font_section_owned, FontAtlas, FontSectionConfig};
use crate::gamemtl::GameMaterialLibrary;
use crate::hom::HomMesh;
use crate::lanims::LightAnimLibrary;
use crate::level::{BuildLights, LevelData, LevelGeometry, TorchLightConfig};
use crate::ogf::{OgfModel, OgfTextureLink};
use crate::omf::{BonePartitionSet, OmfMotions};
use crate::particles::{
    BloodWallmarksConfig, ParticleEffectDef, ParticleGroupDef, ParticlesLibrary,
};
use crate::savegames::SavegameFile;
use crate::shaders_xr::{
    EngineShaderLibrary, LevelEnvModifiers, LevelFogVolumes, LevelStaticParticles,
    ShaderXrlcLibrary,
};
use crate::sound_env::{
    LevelSoundEnv, LevelStaticSounds, OggSoundMetadata, SoundEnvironmentLibrary, SoundOcclusionMesh,
};
use crate::spawn_overlays::SpawnOverlaysConfig;
use crate::spawns::{AlifeSpawnArchive, LevelGameData, LevelSpawnData, PatrolPath};
use crate::thm::{
    strip_texture_extension, texture_stem_to_dds_vpath, texture_stem_to_thm_vpath,
    DdsCubemapLayout, ResolvedTextureLinks, TextureParams,
};
use crate::weather::{EnvGlobalParams, SunPositionsConfig, WeatherCycle, WeatherLibrary};
use crate::world_maps::{GameLevelsConfig, WorldMapsConfig};
use crate::xml_ui::{
    LocalizationTable, UiColorDefs, UiTextureAtlasMap, UiViewport, XmlDocument, XmlFileProvider,
};

/// Adapter implementing `xray_core::ltx_loader::LtxFileProvider` over a mounted `VirtualFileSystem`.
pub struct VfsLtxProvider<'a>(pub &'a VirtualFileSystem);

impl<'a> LtxFileProvider for VfsLtxProvider<'a> {
    type Error = VfsError;

    fn read_ltx_text(&self, virtual_path: &str) -> Result<String, Self::Error> {
        self.0.read_to_string_lossy(virtual_path)
    }

    fn list_ltx_matches(
        &self,
        dir_prefix: &str,
        filename_pattern: &str,
    ) -> Result<Vec<String>, Self::Error> {
        let prefix = normalize_virtual_prefix(dir_prefix)?;
        let all = self.0.list_files(&prefix)?;

        let mut matches = Vec::new();
        for path in all {
            if let Some(rest) = path.strip_prefix(&prefix)
                && !rest.contains('\\')
                && glob_matches_ascii(filename_pattern, rest)
            {
                matches.push(path);
            }
        }
        Ok(matches)
    }
}

/// Load and merge an entire `.ltx` include tree + DLTX `mod_*.ltx` overrides straight from `VirtualFileSystem`.
pub fn load_ltx_tree_from_vfs(
    vfs: &VirtualFileSystem,
    root_virtual_path: &str,
    dltx_mod_paths: &[&str],
) -> Result<OwnedLtxDocument, VfsError> {
    let norm_root = normalize_virtual_path(root_virtual_path)?;
    let norm_mods: Vec<String> = dltx_mod_paths
        .iter()
        .map(|p| normalize_virtual_path(p))
        .collect::<Result<_, _>>()?;
    let mod_refs: Vec<&str> = norm_mods.iter().map(String::as_str).collect();

    let provider = VfsLtxProvider(vfs);
    let sources = collect_ltx_sources(&provider, &norm_root, &mod_refs)?;
    Ok(OwnedLtxDocument::from_sources(&sources))
}

#[derive(Debug)]
pub enum AssetLoadError {
    Vfs(VfsError),
    Format(FormatError),
}

impl std::fmt::Display for AssetLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AssetLoadError {}

impl From<VfsError> for AssetLoadError {
    fn from(err: VfsError) -> Self {
        Self::Vfs(err)
    }
}

impl From<FormatError> for AssetLoadError {
    fn from(err: FormatError) -> Self {
        Self::Format(err)
    }
}

/// Stream an `.ogf` asset from `VirtualFileSystem` (archive or loose override) and parse into `OgfModel`.
pub fn load_ogf_from_vfs(
    vfs: &VirtualFileSystem,
    virtual_path: &str,
) -> Result<OgfModel, AssetLoadError> {
    let bytes = vfs.read_bytes(virtual_path)?;
    Ok(OgfModel::parse(&bytes)?)
}

/// Stream a standalone `.omf` clip library from `VirtualFileSystem`.
///
/// Accepts either a full VFS path (`$game_meshes$/actors/stalker_animation.omf`) or an
/// `OGF_S_MOTION_REFS` reference (`actors\stalker_animation`).
pub fn load_omf_from_vfs(
    vfs: &VirtualFileSystem,
    reference_or_path: &str,
) -> Result<OmfMotions, AssetLoadError> {
    let trimmed = reference_or_path.trim();
    let has_omf_ext = trimmed.len() >= 4 && trimmed[trimmed.len() - 4..].eq_ignore_ascii_case(".omf");
    let with_ext = if has_omf_ext {
        trimmed.to_string()
    } else {
        format!("{trimmed}.omf")
    };

    let candidate_path = if with_ext.starts_with('$') {
        with_ext
    } else {
        let norm = normalize_virtual_path(&with_ext)?;
        if norm.starts_with("meshes\\") {
            norm
        } else {
            format!("meshes\\{norm}")
        }
    };

    let bytes = vfs.read_bytes(&candidate_path)?;
    Ok(OmfMotions::parse(&bytes)?)
}

/// Stream optional `.thm` (or `.ltx` `[texture]` fallback) metadata from `VirtualFileSystem`.
///
/// Returns `Ok(None)` when neither `<tex>.thm` nor `<tex>.ltx` exists in VFS.
pub fn load_thm_from_vfs(
    vfs: &VirtualFileSystem,
    texture_name: &str,
) -> Result<Option<TextureParams>, AssetLoadError> {
    let thm_path = texture_stem_to_thm_vpath(texture_name);
    if vfs.exists(&thm_path) {
        let bytes = vfs.read_bytes(&thm_path)?;
        return Ok(Some(TextureParams::parse(&bytes)?));
    }

    let ltx_path = format!(
        "{}.ltx",
        strip_texture_extension(&thm_path)
    );
    if vfs.exists(&ltx_path) {
        let text = vfs.read_to_string_lossy(&ltx_path)?;
        match TextureParams::from_ltx_fallback(texture_name, &text) {
            Ok(params) => return Ok(Some(params)),
            Err(FormatError::MissingChunk(_)) => return Ok(None),
            Err(e) => return Err(AssetLoadError::Format(e)),
        }
    }

    Ok(None)
}

/// Resolve an `OgfTextureLink`'s diffuse `.dds` and optional `.thm` bump (`_bump` / `_bump#`),
/// detail, and external normal map paths from `VirtualFileSystem`.
pub fn resolve_ogf_texture_links(
    vfs: &VirtualFileSystem,
    link: &OgfTextureLink,
) -> Result<ResolvedTextureLinks, AssetLoadError> {
    if let Some(params) = load_thm_from_vfs(vfs, &link.texture)? {
        return Ok(params.resolve_texture_links(&link.texture));
    }

    let clean_name = strip_texture_extension(&link.texture).replace('/', "\\");
    Ok(ResolvedTextureLinks {
        diffuse_dds: texture_stem_to_dds_vpath(&clean_name),
        diffuse_name: clean_name,
        bump: None,
        detail_dds: None,
        detail_scale: 1.0,
        use_diffuse_detail: false,
        use_bump_detail: false,
        ext_normal_map_dds: None,
        material_id: 1.0,
    })
}

/// Stream `$game_data$/gamemtl.xr` from `VirtualFileSystem` and apply optional Anomaly loose LTX overlays
/// (`materials\materials.ltx` and `materials\material_pairs.ltx`) when present.
pub fn load_gamemtl_from_vfs(
    vfs: &VirtualFileSystem,
) -> Result<GameMaterialLibrary, AssetLoadError> {
    let bytes = vfs.read_bytes("$game_data$/gamemtl.xr")?;
    let mut lib = GameMaterialLibrary::parse(&bytes)?;

    if vfs.exists("$game_data$/materials/materials.ltx") {
        let mtls_ltx = load_ltx_tree_from_vfs(vfs, "$game_data$/materials/materials.ltx", &[])?;
        lib.apply_materials_ltx(&mtls_ltx);
    }

    if vfs.exists("$game_data$/materials/material_pairs.ltx") {
        let pairs_ltx =
            load_ltx_tree_from_vfs(vfs, "$game_data$/materials/material_pairs.ltx", &[])?;
        lib.apply_material_pairs_ltx(&pairs_ltx);
    }

    Ok(lib)
}

fn resolve_level_file_path(dir_or_file: &str, filename: &str) -> String {
    let trimmed = dir_or_file.trim().trim_end_matches(['/', '\\']);
    let leaf = trimmed.rsplit(['/', '\\']).next().unwrap_or("");
    if leaf.eq_ignore_ascii_case(filename) {
        trimmed.to_string()
    } else if trimmed.is_empty() {
        filename.to_string()
    } else {
        format!("{trimmed}/{filename}")
    }
}

/// Stream `level.cform` (accepting either full path or level directory like `"levels/l01_escape"`).
pub fn load_cform_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<CformModel, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.cform");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(CformModel::parse(&bytes)?)
}

/// Stream `level.details` (accepting either full path or level directory).
pub fn load_details_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelDetails, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.details");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelDetails::parse(&bytes)?)
}

/// Stream a standalone `.dm` flora/detail mesh from `VirtualFileSystem`.
pub fn load_dm_from_vfs(
    vfs: &VirtualFileSystem,
    virtual_path: &str,
) -> Result<DetailMesh, AssetLoadError> {
    let bytes = vfs.read_bytes(virtual_path)?;
    Ok(DetailMesh::parse_dm(&bytes)?)
}

/// Stream `level.geom` or `level.geomx` from `VirtualFileSystem`.
pub fn load_level_geometry_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelGeometry, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.geom");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelGeometry::parse(&bytes)?)
}

/// Stream `level` scene data (accepting either `"levels/l01_escape/level"` or `"levels/l01_escape"`).
pub fn load_level_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelData, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelData::parse(&bytes)?)
}

/// Stream `level.hom` occlusion mesh (accepting full path or level directory).
pub fn load_hom_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<HomMesh, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.hom");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(HomMesh::parse(&bytes)?)
}

/// Stream `level.ai` navigation mesh (accepting full path or level directory).
pub fn load_level_ai_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelGraph, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.ai");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelGraph::parse(&bytes)?)
}

/// Stream `level.gct` AI-to-game-graph cross-table (accepting full path or level directory).
pub fn load_level_gct_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelCrossTable, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.gct");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelCrossTable::parse(&bytes)?)
}

/// Stream global `game.graph` (defaults to `$game_data$/game.graph` when `path_override` is `None`).
pub fn load_game_graph_from_vfs(
    vfs: &VirtualFileSystem,
    path_override: Option<&str>,
) -> Result<GameGraph, AssetLoadError> {
    let vpath = path_override.unwrap_or("$game_data$/game.graph");
    let bytes = vfs.read_bytes(vpath)?;
    Ok(GameGraph::parse(&bytes)?)
}

/// Stream `$game_data$/senvironment.xr` acoustic reverb library.
pub fn load_sound_environments_from_vfs(
    vfs: &VirtualFileSystem,
) -> Result<SoundEnvironmentLibrary, AssetLoadError> {
    let bytes = vfs.read_bytes("$game_data$/senvironment.xr")?;
    Ok(SoundEnvironmentLibrary::parse(&bytes)?)
}

/// Stream `level.som` sound occlusion mesh (accepting full path or level directory).
pub fn load_som_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<SoundOcclusionMesh, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.som");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(SoundOcclusionMesh::parse(&bytes)?)
}

/// Stream `level.snd_env` spatial reverb zone mesh (accepting full path or level directory).
pub fn load_level_sound_env_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelSoundEnv, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.snd_env");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelSoundEnv::parse(&bytes)?)
}

/// Stream `level.snd_static` ambient point sounds (accepting full path or level directory).
pub fn load_static_sounds_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelStaticSounds, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.snd_static");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelStaticSounds::parse(&bytes)?)
}

/// Stream `$game_data$/shaders_xrlc.xr` compiler shader library.
pub fn load_shaders_xrlc_from_vfs(
    vfs: &VirtualFileSystem,
) -> Result<ShaderXrlcLibrary, AssetLoadError> {
    let bytes = vfs.read_bytes("$game_data$/shaders_xrlc.xr")?;
    Ok(ShaderXrlcLibrary::parse(&bytes)?)
}

/// Stream `$game_data$/shaders.xr` engine constant/matrix/blender library.
pub fn load_shaders_xr_from_vfs(
    vfs: &VirtualFileSystem,
) -> Result<EngineShaderLibrary, AssetLoadError> {
    let bytes = vfs.read_bytes("$game_data$/shaders.xr")?;
    Ok(EngineShaderLibrary::parse(&bytes)?)
}

/// Stream level environment modifiers, preferring Anomaly `$game_configs$/environment/env_mods/<level>.ltx`
/// when `level_name` matches a mounted LTX config, otherwise reading binary `level.env_mod`.
pub fn load_env_mods_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
    level_name: Option<&str>,
) -> Result<LevelEnvModifiers, AssetLoadError> {
    if let Some(name) = level_name {
        for base in [
            format!("$game_configs$/environment/env_mods/{name}.ltx"),
            format!("$game_config$/environment/env_mods/{name}.ltx"),
        ] {
            if vfs.exists(&base) {
                let doc = load_ltx_tree_from_vfs(vfs, &base, &[])?;
                return Ok(LevelEnvModifiers::from_owned_ltx(&doc));
            }
        }
    }
    let vpath = resolve_level_file_path(level_dir_or_path, "level.env_mod");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelEnvModifiers::parse(&bytes)?)
}

/// Stream `level.ps_static` placed particle emitters (accepting full path or level directory).
pub fn load_ps_static_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelStaticParticles, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.ps_static");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelStaticParticles::parse(&bytes)?)
}

/// Stream `level.fog_vol` volumetric fog regions (accepting full path or level directory).
pub fn load_fog_volumes_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelFogVolumes, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.fog_vol");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelFogVolumes::parse(&bytes)?)
}

fn ensure_extension_ci(trimmed: &str, ext: &[u8]) -> String {
    if trimmed.len() >= ext.len()
        && trimmed.as_bytes()[trimmed.len() - ext.len()..].eq_ignore_ascii_case(ext)
    {
        trimmed.to_string()
    } else {
        let ext_str = std::str::from_utf8(ext).unwrap_or("");
        format!("{trimmed}{ext_str}")
    }
}

/// Stream `$game_data$/particles.xr` and merge loose `particles\*.pe` / `*.pg` LTX definitions
/// with Anomaly precedence (loose `.pe`/`.pg` files override compiled definitions).
pub fn load_particles_from_vfs(vfs: &VirtualFileSystem) -> Result<ParticlesLibrary, AssetLoadError> {
    let has_xr = vfs.exists("$game_data$/particles.xr");
    let mut lib = if has_xr {
        let bytes = vfs.read_bytes("$game_data$/particles.xr")?;
        ParticlesLibrary::parse(&bytes)?
    } else {
        ParticlesLibrary::default()
    };

    let particle_files = vfs.list_files("particles")?;
    let mut loose_effects = Vec::new();
    let mut loose_groups = Vec::new();

    for vpath in particle_files {
        let rel = vpath.strip_prefix("particles\\").unwrap_or(&vpath);
        if rel.len() > 3 && rel.as_bytes()[rel.len() - 3..].eq_ignore_ascii_case(b".pe") {
            let name = &rel[..rel.len() - 3];
            let text = vfs.read_to_string_lossy(&vpath)?;
            loose_effects.push(ParticleEffectDef::from_pe_ltx(name, &text)?);
        } else if rel.len() > 3 && rel.as_bytes()[rel.len() - 3..].eq_ignore_ascii_case(b".pg") {
            let name = &rel[..rel.len() - 3];
            let text = vfs.read_to_string_lossy(&vpath)?;
            loose_groups.push(ParticleGroupDef::from_pg_ltx(name, &text)?);
        }
    }

    if !has_xr && loose_effects.is_empty() && loose_groups.is_empty() {
        return Err(AssetLoadError::Vfs(VfsError::FileNotFound));
    }

    lib.merge_loose(loose_effects, loose_groups);
    Ok(lib)
}

/// Load a blood wallmarks LTX section (e.g. `[m_blood_marks]`) via `load_ltx_tree_from_vfs`.
pub fn load_blood_wallmarks_from_vfs(
    vfs: &VirtualFileSystem,
    ltx_virtual_path: &str,
    section: &str,
) -> Result<BloodWallmarksConfig, AssetLoadError> {
    let doc = load_ltx_tree_from_vfs(vfs, ltx_virtual_path, &[])?;
    Ok(BloodWallmarksConfig::from_ltx_doc(&doc, section)?)
}

/// Stream a camera / object `.anm` motion file from `$game_anims$/` (or explicit VFS path).
pub fn load_anm_from_vfs(
    vfs: &VirtualFileSystem,
    reference_or_path: &str,
) -> Result<ObjectMotion, AssetLoadError> {
    let trimmed = reference_or_path.trim();
    let with_ext = ensure_extension_ci(trimmed, b".anm");
    let candidate = if with_ext.starts_with('$') {
        with_ext
    } else {
        let norm = normalize_virtual_path(&with_ext)?;
        if norm.starts_with("anims\\") {
            norm
        } else {
            format!("anims\\{norm}")
        }
    };
    let bytes = vfs.read_bytes(&candidate)?;
    Ok(ObjectMotion::parse(&bytes)?)
}

/// Stream a `.ppe` postprocess effect (`CPostprocessAnimator::Load`: checks `level_dir` first, then `$game_anims$/`).
pub fn load_ppe_from_vfs(
    vfs: &VirtualFileSystem,
    reference_or_path: &str,
    level_dir: Option<&str>,
) -> Result<PostprocessEffect, AssetLoadError> {
    let trimmed = reference_or_path.trim();
    let with_ext = ensure_extension_ci(trimmed, b".ppe");

    if with_ext.starts_with('$') {
        let bytes = vfs.read_bytes(&with_ext)?;
        return Ok(PostprocessEffect::parse(&bytes)?);
    }

    if let Some(dir) = level_dir {
        let dir_clean = dir.trim().trim_end_matches(['/', '\\']);
        if !dir_clean.is_empty() {
            let rel_clean = with_ext.trim_start_matches(['/', '\\']);
            let level_candidate = format!("{dir_clean}/{rel_clean}");
            if vfs.exists(&level_candidate) {
                let bytes = vfs.read_bytes(&level_candidate)?;
                return Ok(PostprocessEffect::parse(&bytes)?);
            }
        }
    }

    let norm = normalize_virtual_path(&with_ext)?;
    let anim_candidate = if norm.starts_with("anims\\") {
        norm
    } else {
        format!("anims\\{norm}")
    };
    let bytes = vfs.read_bytes(&anim_candidate)?;
    Ok(PostprocessEffect::parse(&bytes)?)
}

/// Stream `level.game` (RPoints & PatrolPaths; accepts full path or level directory).
pub fn load_level_game_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelGameData, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.game");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelGameData::parse(&bytes)?)
}

/// Stream Anomaly `$game_configs$/patrol_paths.ltx` (+ optional DLTX overrides) into `Vec<PatrolPath>`.
pub fn load_patrol_paths_ltx_from_vfs(
    vfs: &VirtualFileSystem,
    dltx_mod_paths: &[&str],
) -> Result<Vec<PatrolPath>, AssetLoadError> {
    let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/patrol_paths.ltx", dltx_mod_paths)?;
    Ok(PatrolPath::parse_all_ltx_doc(&doc)?)
}

/// Stream `level.spawn` (`M_SPAWN` entity list; accepts full path or level directory).
pub fn load_level_spawn_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<LevelSpawnData, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "level.spawn");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(LevelSpawnData::parse(&bytes)?)
}

/// Stream compiled `all.spawn` (or `<name>.spawn` under `$game_spawns$/`).
pub fn load_all_spawn_from_vfs(
    vfs: &VirtualFileSystem,
    spawn_name_or_path: &str,
) -> Result<AlifeSpawnArchive, AssetLoadError> {
    let trimmed = spawn_name_or_path.trim();
    let with_ext = if trimmed.is_empty() {
        "all.spawn".to_string()
    } else {
        ensure_extension_ci(trimmed, b".spawn")
    };
    let candidate = if with_ext.starts_with('$') {
        with_ext
    } else {
        let norm = normalize_virtual_path(&with_ext)?;
        if norm.starts_with("spawns\\") {
            norm
        } else {
            format!("spawns\\{norm}")
        }
    };
    let bytes = vfs.read_bytes(&candidate)?;
    Ok(AlifeSpawnArchive::parse(&bytes)?)
}

/// Adapter implementing `crate::xml_ui::XmlFileProvider` over a mounted `VirtualFileSystem`.
pub struct VfsXmlProvider<'a>(pub &'a VirtualFileSystem);

impl<'a> XmlFileProvider for VfsXmlProvider<'a> {
    type Error = VfsError;

    fn read_xml_bytes(&self, virtual_path: &str) -> Result<Vec<u8>, Self::Error> {
        self.0.read_bytes(virtual_path)
    }

    fn list_xml_matches(
        &self,
        dir_prefix: &str,
        filename_pattern: &str,
    ) -> Result<Vec<String>, Self::Error> {
        let prefix = normalize_virtual_prefix(dir_prefix)?;
        let all = self.0.list_files(&prefix)?;

        let mut matches = Vec::new();
        for path in all {
            if let Some(rest) = path.strip_prefix(&prefix)
                && !rest.contains('\\')
                && glob_matches_ascii(filename_pattern, rest)
            {
                matches.push(path);
            }
        }
        Ok(matches)
    }
}

/// Stream an Anomaly `.scop` ALife savegame archive from `VirtualFileSystem`.
pub fn load_scop_from_vfs(
    vfs: &VirtualFileSystem,
    virtual_path: &str,
) -> Result<SavegameFile, AssetLoadError> {
    let bytes = vfs.read_bytes(virtual_path)?;
    Ok(SavegameFile::parse(&bytes)?)
}

fn load_weather_cycles_dir(
    vfs: &VirtualFileSystem,
    dir_prefix: &str,
) -> Result<Vec<(String, WeatherCycle)>, AssetLoadError> {
    let norm_prefix = normalize_virtual_prefix(dir_prefix)?;
    let mut files = vfs.list_files(&norm_prefix)?;
    files.sort_by_key(|p| p.to_ascii_lowercase());

    let mut out = Vec::new();
    for vpath in files {
        let rest = vpath.strip_prefix(&norm_prefix).unwrap_or(&vpath);
        if rest.contains('\\') || rest.len() <= 4 || !rest.as_bytes()[rest.len() - 4..].eq_ignore_ascii_case(b".ltx") {
            continue;
        }
        let stem = &rest[..rest.len() - 4];
        let doc = load_ltx_tree_from_vfs(vfs, &vpath, &[])?;
        out.push((stem.to_owned(), WeatherCycle::from_ltx(stem, &doc)?));
    }
    Ok(out)
}

/// Stream `$game_configs$/environment/` weather cycles, effects, sun paths, ambients, lens flares,
/// and thunderbolt configs (plus optional level-specific `ambients\<level>.ltx`).
pub fn load_weather_library_from_vfs(
    vfs: &VirtualFileSystem,
    level_name: Option<&str>,
) -> Result<WeatherLibrary, AssetLoadError> {
    let mut lib = WeatherLibrary::default();
    let mut any_loaded = false;

    if vfs.exists("$game_configs$/environment/environment.ltx") {
        let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/environment/environment.ltx", &[])?;
        lib.global_params = EnvGlobalParams::from_ltx(&doc);
        any_loaded = true;
    }

    if vfs.exists("$game_configs$/environment/sun_positions.ltx") {
        let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/environment/sun_positions.ltx", &[])?;
        lib.sun_positions = Some(SunPositionsConfig::from_ltx(&doc)?);
        any_loaded = true;
    }

    if vfs.exists("$game_configs$/environment/sound_channels.ltx") {
        let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/environment/sound_channels.ltx", &[])?;
        lib.parse_sound_channels(&doc)?;
        any_loaded = true;
    }

    if vfs.exists("$game_configs$/environment/effects.ltx") {
        let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/environment/effects.ltx", &[])?;
        lib.parse_ambient_effects(&doc)?;
        any_loaded = true;
    }

    if vfs.exists("$game_configs$/environment/ambients.ltx") {
        let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/environment/ambients.ltx", &[])?;
        lib.parse_ambients(&doc)?;
        any_loaded = true;
    }

    if let Some(level) = level_name {
        let lvl_amb = format!("$game_configs$/environment/ambients/{level}.ltx");
        if vfs.exists(&lvl_amb) {
            let doc = load_ltx_tree_from_vfs(vfs, &lvl_amb, &[])?;
            lib.parse_ambients(&doc)?;
            any_loaded = true;
        }
    }

    if vfs.exists("$game_configs$/environment/suns.ltx") {
        let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/environment/suns.ltx", &[])?;
        lib.parse_suns(&doc);
        any_loaded = true;
    }

    if vfs.exists("$game_configs$/environment/thunderbolts.ltx") {
        let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/environment/thunderbolts.ltx", &[])?;
        lib.parse_thunderbolts(&doc);
        any_loaded = true;
    }

    if vfs.exists("$game_configs$/environment/thunderbolt_collections.ltx") {
        let doc = load_ltx_tree_from_vfs(
            vfs,
            "$game_configs$/environment/thunderbolt_collections.ltx",
            &[],
        )?;
        lib.parse_thunderbolt_collections(&doc);
        any_loaded = true;
    }

    for (name, cycle) in load_weather_cycles_dir(vfs, "$game_configs$/environment/weathers")? {
        lib.cycles.insert(name, cycle);
        any_loaded = true;
    }

    for (name, effect_cycle) in
        load_weather_cycles_dir(vfs, "$game_configs$/environment/weather_effects")?
    {
        lib.effects_cycles.insert(name, effect_cycle);
        any_loaded = true;
    }

    if !any_loaded {
        return Err(AssetLoadError::Vfs(VfsError::FileNotFound));
    }

    Ok(lib)
}

/// Stream and parse an X-Ray XML document (decoding Windows-1251/UTF-8 and expanding `#include "..."` directives).
pub fn load_xml_from_vfs(
    vfs: &VirtualFileSystem,
    virtual_path: &str,
) -> Result<XmlDocument, AssetLoadError> {
    let norm = normalize_virtual_path(virtual_path)?;
    let base_dir = if norm.starts_with("configs\\") {
        "configs"
    } else if let Some(idx) = norm.rfind('\\') {
        &norm[..idx]
    } else {
        ""
    };
    let provider = VfsXmlProvider(vfs);
    Ok(XmlDocument::load_with_provider(&provider, base_dir, &norm)?)
}

fn collect_top_level_xml_in_dir(
    vfs: &VirtualFileSystem,
    dir_vpath: &str,
) -> Result<Vec<String>, VfsError> {
    let prefix = normalize_virtual_prefix(dir_vpath)?;
    let mut files = vfs.list_files(&prefix)?;
    files.sort_by_key(|p| p.to_ascii_lowercase());
    Ok(files
        .into_iter()
        .filter(|path| {
            let rest = path.strip_prefix(&prefix).unwrap_or(path);
            !rest.contains('\\')
                && rest.len() > 4
                && rest.as_bytes()[rest.len() - 4..].eq_ignore_ascii_case(b".xml")
        })
        .collect())
}

/// Stream `$game_configs$/text/<language>/*.xml` into a `LocalizationTable`, pre-loading
/// English (`text\eng\*.xml`) first for missing translation fallback (`CStringTable::Init`).
pub fn load_localization_table_from_vfs(
    vfs: &VirtualFileSystem,
    language: &str,
) -> Result<LocalizationTable, AssetLoadError> {
    let clean_lang = language.trim();
    if clean_lang.is_empty() {
        return Err(AssetLoadError::Vfs(VfsError::InvalidPath(
            "empty language".to_string(),
        )));
    }

    let mut table = LocalizationTable::new(clean_lang);
    let mut file_count = 0usize;

    if !clean_lang.eq_ignore_ascii_case("eng") {
        for xml_vpath in collect_top_level_xml_in_dir(vfs, "$game_configs$/text/eng")? {
            let doc = load_xml_from_vfs(vfs, &xml_vpath)?;
            table.merge_doc(&doc);
            file_count += 1;
        }
    }

    let lang_dir = format!("$game_configs$/text/{clean_lang}");
    for xml_vpath in collect_top_level_xml_in_dir(vfs, &lang_dir)? {
        let doc = load_xml_from_vfs(vfs, &xml_vpath)?;
        table.merge_doc(&doc);
        file_count += 1;
    }

    if file_count == 0 {
        return Err(AssetLoadError::Vfs(VfsError::FileNotFound));
    }

    Ok(table)
}

/// Stream all `$game_configs$/ui/textures_descr/*.xml` UI sprite atlas sheets into `UiTextureAtlasMap`.
pub fn load_ui_texture_atlases_from_vfs(
    vfs: &VirtualFileSystem,
) -> Result<UiTextureAtlasMap, AssetLoadError> {
    let files = collect_top_level_xml_in_dir(vfs, "$game_configs$/ui/textures_descr")?;
    if files.is_empty() {
        return Err(AssetLoadError::Vfs(VfsError::FileNotFound));
    }
    let mut map = UiTextureAtlasMap::default();
    for xml_vpath in files {
        let doc = load_xml_from_vfs(vfs, &xml_vpath)?;
        map.merge_doc(&doc);
    }
    Ok(map)
}

/// Stream `$game_configs$/ui/color_defs.xml` named color palette (`UiColorDefs`).
pub fn load_color_defs_from_vfs(vfs: &VirtualFileSystem) -> Result<UiColorDefs, AssetLoadError> {
    let doc = load_xml_from_vfs(vfs, "$game_configs$/ui/color_defs.xml")?;
    Ok(UiColorDefs::from_doc(&doc))
}

/// Stream `$game_data$/lanims.xr` keyframed light/UI color animations (`LightAnimLibrary`).
pub fn load_lanims_from_vfs(vfs: &VirtualFileSystem) -> Result<LightAnimLibrary, AssetLoadError> {
    let bytes = vfs.read_bytes("$game_data$/lanims.xr")?;
    Ok(LightAnimLibrary::parse(&bytes)?)
}

/// Stream a multi-clip `.anms` motion library from `$game_anims$/` (or explicit VFS path).
pub fn load_anms_from_vfs(
    vfs: &VirtualFileSystem,
    reference_or_path: &str,
) -> Result<ObjectMotionLibrary, AssetLoadError> {
    let trimmed = reference_or_path.trim();
    let with_ext = ensure_extension_ci(trimmed, b".anms");
    let candidate = if with_ext.starts_with('$') {
        with_ext
    } else {
        let norm = normalize_virtual_path(&with_ext)?;
        if norm.starts_with("anims\\") {
            norm
        } else {
            format!("anims\\{norm}")
        }
    };
    let bytes = vfs.read_bytes(&candidate)?;
    Ok(ObjectMotionLibrary::parse(&bytes)?)
}

/// Stream skeletal bone partition LTX overrides (`CPartition::load`, `SkeletonMotions.cpp:27-80`).
///
/// When `force_hands` is `true`, loads `$game_meshes$/anomaly_weapons/hands/bone_parts.ltx`;
/// otherwise replaces the model extension with `.ltx` under `$game_meshes$/`.
pub fn load_bone_parts_ltx_from_vfs(
    vfs: &VirtualFileSystem,
    model_or_ltx_path: &str,
    force_hands: bool,
) -> Result<BonePartitionSet, AssetLoadError> {
    let candidate = if force_hands {
        "$game_meshes$/anomaly_weapons/hands/bone_parts.ltx".to_string()
    } else {
        let trimmed = model_or_ltx_path.trim();
        let stem = strip_texture_extension(trimmed);
        let with_ext = format!("{stem}.ltx");
        if with_ext.starts_with('$') {
            with_ext
        } else {
            let norm = normalize_virtual_path(&with_ext)?;
            if norm.starts_with("meshes\\") {
                norm
            } else {
                format!("meshes\\{norm}")
            }
        }
    };
    let text = vfs.read_to_string_lossy(&candidate)?;
    let doc = xray_core::LtxDocument::parse(&text);
    Ok(BonePartitionSet::from_ltx(&doc))
}

/// Stream `.ogg` sound metadata (`CSoundRender_Source::LoadWave`) from `$game_sounds$/`.
pub fn load_ogg_sound_metadata_from_vfs(
    vfs: &VirtualFileSystem,
    sound_name_or_path: &str,
) -> Result<OggSoundMetadata, AssetLoadError> {
    let trimmed = sound_name_or_path.trim();
    let with_ext = ensure_extension_ci(trimmed, b".ogg");
    let candidate = if with_ext.starts_with('$') {
        with_ext
    } else {
        let norm = normalize_virtual_path(&with_ext)?;
        if norm.starts_with("sounds\\") {
            norm
        } else {
            format!("sounds\\{norm}")
        }
    };
    let bytes = vfs.read_bytes(&candidate)?;
    Ok(OggSoundMetadata::parse(&bytes)?)
}

/// Resolve `[section]` font texture rung + `.ini`/`.dds` paths from `config_ltx` (`fonts.ltx` / `system.ltx`),
/// probing VFS existence of candidate `<resolved_texture>.ini` atlases (`GetFontTextureName`).
pub fn resolve_font_config_from_vfs(
    vfs: &VirtualFileSystem,
    config_ltx: &OwnedLtxDocument,
    section: &str,
    screen_height: u32,
) -> Result<FontSectionConfig, AssetLoadError> {
    Ok(resolve_font_section_owned(
        config_ltx,
        section,
        screen_height,
        |ini_vpath| vfs.exists(ini_vpath),
    )?)
}

/// Stream a `<font_texture>.ini` glyph metric atlas (`FontAtlas`) from `$game_textures$/` (or explicit VFS path).
pub fn load_font_atlas_from_vfs(
    vfs: &VirtualFileSystem,
    resolved_texture_or_ini: &str,
) -> Result<FontAtlas, AssetLoadError> {
    let trimmed = resolved_texture_or_ini.trim();
    let candidate = if trimmed.starts_with('$') {
        if trimmed.len() >= 4 && trimmed[trimmed.len() - 4..].eq_ignore_ascii_case(".ini") {
            trimmed.to_string()
        } else {
            let stem = strip_texture_extension(trimmed);
            format!("{stem}.ini")
        }
    } else {
        font_atlas_ini_vpath(trimmed)
    };
    let text = vfs.read_to_string_lossy(&candidate)?;
    Ok(FontAtlas::parse_ini_str(&text)?)
}

/// Resolve and stream both `[section]` font config (`FontSectionConfig`) and its parsed `FontAtlas` in one call.
pub fn load_font_from_vfs(
    vfs: &VirtualFileSystem,
    config_ltx: &OwnedLtxDocument,
    section: &str,
    screen_height: u32,
) -> Result<(FontSectionConfig, FontAtlas), AssetLoadError> {
    let cfg = resolve_font_config_from_vfs(vfs, config_ltx, section, screen_height)?;
    let atlas = load_font_atlas_from_vfs(vfs, &cfg.atlas_ini_vpath)?;
    Ok((cfg, atlas))
}

/// Discover `$game_configs$/mod_spawn_overlays_*.ltx` DLTX overlay files sorted lexicographically.
pub fn discover_spawn_overlay_mods_from_vfs(
    vfs: &VirtualFileSystem,
) -> Result<Vec<String>, AssetLoadError> {
    let provider = VfsLtxProvider(vfs);
    let mut mods = provider.list_ltx_matches("$game_configs$", "mod_spawn_overlays_*.ltx")?;
    mods.sort_by_key(|s| s.to_ascii_lowercase());
    Ok(mods)
}

/// Stream `$game_configs$/spawn_overlays.ltx` merged with DLTX `mod_spawn_overlays_*.ltx` files.
///
/// When `dltx_mod_paths` is `None`, auto-discovers all `$game_configs$/mod_spawn_overlays_*.ltx` files.
pub fn load_spawn_overlays_from_vfs(
    vfs: &VirtualFileSystem,
    dltx_mod_paths: Option<&[&str]>,
) -> Result<SpawnOverlaysConfig, AssetLoadError> {
    let owned_mods;
    let mod_refs: Vec<&str> = match dltx_mod_paths {
        Some(explicit) => explicit.to_vec(),
        None => {
            owned_mods = discover_spawn_overlay_mods_from_vfs(vfs)?;
            owned_mods.iter().map(String::as_str).collect()
        }
    };
    let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/spawn_overlays.ltx", &mod_refs)?;
    Ok(SpawnOverlaysConfig::from_owned_ltx(&doc))
}

/// Stream `$game_configs$/game_levels.ltx` (and any DLTX mods) into `GameLevelsConfig`.
pub fn load_game_levels_from_vfs(
    vfs: &VirtualFileSystem,
    dltx_mod_paths: &[&str],
) -> Result<GameLevelsConfig, AssetLoadError> {
    let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/game_levels.ltx", dltx_mod_paths)?;
    Ok(GameLevelsConfig::from_owned_ltx(&doc))
}

/// Stream `$game_configs$/game_maps_single.ltx` into `WorldMapsConfig`, optionally applying
/// per-level `$game_levels$/<level>/level.ltx` `[level_map]` overrides present in VFS.
pub fn load_world_maps_from_vfs(
    vfs: &VirtualFileSystem,
    dltx_mod_paths: &[&str],
    apply_level_ltx_overrides: bool,
) -> Result<WorldMapsConfig, AssetLoadError> {
    let doc = load_ltx_tree_from_vfs(vfs, "$game_configs$/game_maps_single.ltx", dltx_mod_paths)?;
    let mut cfg = WorldMapsConfig::from_owned_ltx(&doc)?;
    if apply_level_ltx_overrides {
        let level_names: Vec<String> = cfg.level_maps.keys().cloned().collect();
        for name in level_names {
            let level_ltx_vpath = format!("$game_levels$/{name}/level.ltx");
            if vfs.exists(&level_ltx_vpath) {
                let text = vfs.read_to_string_lossy(&level_ltx_vpath)?;
                let ltx = xray_core::LtxDocument::parse(&text);
                cfg.apply_level_ltx_override(&name, &ltx);
            }
        }
    }
    Ok(cfg)
}

/// Resolve and stream a UI XML file using aspect-ratio suffixes (`_21.xml` -> `_16.xml` -> `.xml`)
/// from `$game_configs$/ui/` (or an explicit VFS path).
pub fn load_ui_xml_for_viewport_from_vfs(
    vfs: &VirtualFileSystem,
    xml_name_or_vpath: &str,
    viewport: &UiViewport,
) -> Result<XmlDocument, AssetLoadError> {
    let trimmed = xml_name_or_vpath.trim();
    let has_prefix = trimmed.starts_with('$')
        || trimmed.contains('/')
        || trimmed.contains('\\');
    for cand in viewport.xml_variant_candidates(trimmed) {
        let vpath = if has_prefix {
            cand
        } else {
            format!("$game_configs$/ui/{cand}")
        };
        if vfs.exists(&vpath) {
            return load_xml_from_vfs(vfs, &vpath);
        }
    }
    Err(AssetLoadError::Vfs(VfsError::FileNotFound))
}

/// Stream `$level$/build.lights` (accepting either full path or level directory like `"$game_levels$/l01_escape"`).
pub fn load_build_lights_from_vfs(
    vfs: &VirtualFileSystem,
    level_dir_or_path: &str,
) -> Result<BuildLights, AssetLoadError> {
    let vpath = resolve_level_file_path(level_dir_or_path, "build.lights");
    let bytes = vfs.read_bytes(&vpath)?;
    Ok(BuildLights::parse(&bytes)?)
}

/// Stream a `[torch_definition]` lighting preset LTX section (+ optional DLTX overrides) from VFS.
pub fn load_torch_light_config_from_vfs(
    vfs: &VirtualFileSystem,
    ltx_vpath: &str,
    dltx_mod_paths: &[&str],
    section: &str,
    prefer_r2: bool,
) -> Result<TorchLightConfig, AssetLoadError> {
    let doc = load_ltx_tree_from_vfs(vfs, ltx_vpath, dltx_mod_paths)?;
    Ok(TorchLightConfig::from_ltx(&doc, section, prefer_r2)?)
}

/// Stream a skybox or irradiance (`#small`) `.dds` cubemap from VFS and compute its
/// zero-copy 6-face x mip subresource slice layout (`DdsCubemapLayout`).
pub fn load_cubemap_dds_from_vfs(
    vfs: &VirtualFileSystem,
    texture_name_or_vpath: &str,
) -> Result<(Vec<u8>, DdsCubemapLayout), AssetLoadError> {
    let trimmed = texture_name_or_vpath.trim();
    let vpath = if trimmed.starts_with('$') {
        ensure_extension_ci(trimmed, b".dds")
    } else {
        texture_stem_to_dds_vpath(trimmed)
    };
    let bytes = vfs.read_bytes(&vpath)?;
    let layout = DdsCubemapLayout::parse(&bytes)?;
    Ok((bytes, layout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use crate::gamemtl::{gamemtl_chunks, gamemtl_flags, gamemtl_root_chunks};
    use crate::ogf::ogf_chunks::*;
    use crate::ogf::{fvf_flags, OgfVisualType};
    use crate::omf::{motion_def_flags, motion_key_flags};
    use crate::thm::{thm_chunks, TextureBumpMode, TextureFormat};
    use std::path::PathBuf;
    use xray_vfs::{compress_lzo1x, compute_crc32};

    fn pack_synthetic_db(entries: Vec<(&str, Vec<u8>, Vec<u8>)>) -> Vec<u8> {
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
        let mut db = Vec::new();
        db.extend_from_slice(&1u32.to_le_bytes());
        db.extend_from_slice(&(chunk1.len() as u32).to_le_bytes());
        db.extend_from_slice(&chunk1);
        for (_, _, comp) in &entries {
            db.extend_from_slice(comp);
        }
        db
    }

    #[test]
    fn test_stream_ogf_and_ltx_from_vfs_archive() {
        // Build minimal OGF v4 binary
        let mut ow = ChunkWriter::new();
        let mut hdr = vec![4u8, OgfVisualType::Normal as u8, 0, 0];
        hdr.extend_from_slice(&[0u8; 40]); // bbox (24) + bsphere (16)
        ow.write_chunk(OGF_HEADER, &hdr);
        ow.write_chunk(OGF_TEXTURE, b"wpn\\test_tex\0models\\model\0");
        let mut vb = Vec::new();
        vb.extend_from_slice(&fvf_flags::XRAY_STATIC_FVF_STANDARD.to_le_bytes());
        vb.extend_from_slice(&3u32.to_le_bytes());
        vb.extend_from_slice(&[0u8; 3 * 32]);
        ow.write_chunk(OGF_VERTICES, &vb);
        let mut ib = Vec::new();
        ib.extend_from_slice(&3u32.to_le_bytes());
        for i in [0u16, 1, 2] {
            ib.extend_from_slice(&i.to_le_bytes());
        }
        ow.write_chunk(OGF_INDICES, &ib);
        let ogf_raw = ow.into_bytes();
        let ogf_comp = compress_lzo1x(&ogf_raw);

        // Pack OGF + LTX configs inside synthetic archive (including mutual root ref cycle check)
        let root_ltx = b"#include \"weapons_*.ltx\"\n[wpn_base]\ndamage = 10\n";
        let sub_ltx = b"#include \"weapons.ltx\"\n[wpn_ak]:wpn_base\ncaliber = 5.45\n";
        let mod_ltx = b"![wpn_ak]\ndamage = 35\n";

        let entries = vec![
            ("meshes\\weapons\\ak.ogf", ogf_raw.clone(), ogf_comp),
            ("configs\\weapons.ltx", root_ltx.to_vec(), root_ltx.to_vec()),
            ("configs\\weapons_ak.ltx", sub_ltx.to_vec(), sub_ltx.to_vec()),
            ("configs\\mod_weapons_balance.ltx", mod_ltx.to_vec(), mod_ltx.to_vec()),
        ];

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("test.db0", pack_synthetic_db(entries))
            .unwrap();

        let model = load_ogf_from_vfs(&vfs, "$game_meshes$/weapons/ak.ogf").unwrap();
        assert_eq!(model.triangle_count(), 1);
        assert_eq!(model.texture.as_ref().unwrap().texture, "wpn\\test_tex");

        let cfg = load_ltx_tree_from_vfs(
            &vfs,
            "$game_configs$/weapons.ltx",
            &["$game_configs$/mod_weapons_balance.ltx"],
        )
        .unwrap();
        assert_eq!(cfg.read_i32("wpn_ak", "damage", 0), 35);
        assert_eq!(cfg.read_str("wpn_ak", "caliber"), Some("5.45"));
    }

    #[test]
    fn test_real_gamedata_cursor_cross_dds_and_configs_vfs_streaming() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let gamedata_dir = repo_root.join("gamedata");
        assert!(
            gamedata_dir.is_dir(),
            "repo gamedata directory must exist at {}",
            gamedata_dir.display()
        );

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_loose_dir(&gamedata_dir);

        // Stream real `gamedata/textures/ui/cursor_cross.dds` (plus cursor_dot/cursor_plus)
        for tex in [
            "$game_textures$/ui/cursor_cross.dds",
            "textures\\ui\\cursor_dot.dds",
            "textures/ui/cursor_plus.dds",
        ] {
            let dds = vfs
                .read_bytes(tex)
                .unwrap_or_else(|e| panic!("failed streaming {tex}: {e:?}"));
            // Validate exact uncompressed BGRA8 256x256 DDS header (128 + 256*256*4 = 262272B)
            assert_eq!(&dds[0..4], b"DDS ", "{tex} missing DDS magic");
            let hdr_size = u32::from_le_bytes(dds[4..8].try_into().unwrap());
            let height = u32::from_le_bytes(dds[12..16].try_into().unwrap());
            let width = u32::from_le_bytes(dds[16..20].try_into().unwrap());
            let pf_flags = u32::from_le_bytes(dds[80..84].try_into().unwrap());
            let bpp = u32::from_le_bytes(dds[88..92].try_into().unwrap());
            let r_mask = u32::from_le_bytes(dds[92..96].try_into().unwrap());
            let g_mask = u32::from_le_bytes(dds[96..100].try_into().unwrap());
            let b_mask = u32::from_le_bytes(dds[100..104].try_into().unwrap());
            let a_mask = u32::from_le_bytes(dds[104..108].try_into().unwrap());

            assert_eq!(hdr_size, 124);
            assert_eq!((width, height), (256, 256));
            assert_eq!(pf_flags, 0x41); // DDPF_ALPHAPIXELS | DDPF_RGB
            assert_eq!(bpp, 32);
            assert_eq!(
                (r_mask, g_mask, b_mask, a_mask),
                (0x00FF_0000, 0x0000_FF00, 0x0000_00FF, 0xFF00_0000)
            );
            assert_eq!(dds.len(), 128 + 256 * 256 * 4);
            // Must contain drawn crosshair texels (non-zero alpha channel byte in payload)
            assert!(
                dds[128..].as_chunks::<4>().0.iter().any(|bgra| bgra[3] > 0),
                "{tex} expected non-empty alpha crosshair texels"
            );
        }

        // Verify real config streaming from mounted gamedata/ via VFS
        let knife_cfg = load_ltx_tree_from_vfs(
            &vfs,
            "$game_configs$/mod_system_knife.ltx",
            &[],
        )
        .unwrap();
        assert!(!knife_cfg.read_bool("wpn_knife", "firepos", true));

        let real_patrols = load_patrol_paths_ltx_from_vfs(&vfs, &[]).unwrap();
        assert!(real_patrols.is_empty());

        // Verify real Anomaly localization tables (English + Windows-1251 Russian Cyrillic)
        let loc_eng = load_localization_table_from_vfs(&vfs, "eng").unwrap();
        assert_eq!(loc_eng.translate("st_continue"), "Continue");
        assert_eq!(loc_eng.translate("ui_mm_menu_modded_exes"), "Modded Exes");

        let loc_rus = load_localization_table_from_vfs(&vfs, "rus").unwrap();
        assert_eq!(loc_rus.translate("st_continue"), "Продолжить");
        assert_eq!(loc_rus.translate("st_disconnect"), "Вернуться в меню");
        assert!(
            loc_rus
                .translate("st_lua_fatal_error")
                .contains("«Busy Hands»"),
            "CP1251 guillemets and Cyrillic should decode cleanly"
        );

        // Verify real UI XML `#include "ui\ui_ctrl_hdr_16.xml"` expansion
        let hdr_xml = load_xml_from_vfs(&vfs, "$game_configs$/ui/ui_ctrl_hdr.xml").unwrap();
        assert_eq!(hdr_xml.read_attrib_f32("ctrl_hdr", 0, "x", 0.0), 115.0);
        assert_eq!(
            hdr_xml.read_str("ctrl_hdr:cap_main:text", 0, ""),
            "ui_mm_video_basic_lighting"
        );

        // Verify real UI texture atlases (`textures_descr/*.xml`)
        let atlases = load_ui_texture_atlases_from_vfs(&vfs).unwrap();
        let white_rect = atlases.get("ui_inGame2_white_rect").unwrap();
        assert_eq!(white_rect.pixel_rect(), [1021.0, 1021.0, 1023.0, 1023.0]);
        assert_eq!(white_rect.dds_vpath(), "textures\\ui\\ui_common.dds");
        let opt_bg = atlases.get("ui_inGame2_opt_background").unwrap();
        assert_eq!(opt_bg.pixel_rect(), [0.0, 0.0, 1024.0, 768.0]);
        assert_eq!(
            opt_bg.dds_vpath(),
            "textures\\ui\\ui_actor_options_background.dds"
        );
    }

    #[test]
    fn test_stream_omf_thm_gamemtl_and_real_dds_from_mmap_archive() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let cursor_cross_bytes =
            std::fs::read(repo_root.join("gamedata/textures/ui/cursor_cross.dds")).unwrap();

        // 1. Minimal THM pointing cursor_cross -> ui\cursor_cross_bump
        let mut thm_w = ChunkWriter::new();
        let mut tp = Vec::new();
        tp.extend_from_slice(&(TextureFormat::Rgba as u32).to_le_bytes());
        for v in [1u32, 0, 0, 0, 0, 256, 256] {
            tp.extend_from_slice(&v.to_le_bytes());
        }
        thm_w.write_chunk(thm_chunks::THM_CHUNK_TEXTUREPARAM, &tp);
        let mut bump = Vec::new();
        bump.extend_from_slice(&0.05f32.to_le_bytes());
        bump.extend_from_slice(&(TextureBumpMode::Use as u32).to_le_bytes());
        bump.extend_from_slice(b"ui\\cursor_cross_bump\0");
        thm_w.write_chunk(thm_chunks::THM_CHUNK_BUMP, &bump);
        let thm_raw = thm_w.into_bytes();

        // 2. OGF referencing ui\cursor_cross + external actors\stalker_anim.omf
        let mut ogf_w = ChunkWriter::new();
        let mut hdr = vec![4u8, OgfVisualType::SkeletonAnim as u8, 0, 0];
        hdr.extend_from_slice(&[0u8; 40]);
        ogf_w.write_chunk(OGF_HEADER, &hdr);
        ogf_w.write_chunk(OGF_TEXTURE, b"ui\\cursor_cross\0hud\\default\0");
        let mut bn = Vec::new();
        bn.extend_from_slice(&1u32.to_le_bytes());
        bn.extend_from_slice(b"bip01\0\0");
        bn.extend_from_slice(&[0u8; 60]);
        ogf_w.write_chunk(OGF_S_BONE_NAMES, &bn);
        ogf_w.write_chunk(OGF_S_MOTION_REFS, b"actors\\stalker_anim\0");
        let ogf_raw = ogf_w.into_bytes();

        // 3. Matching OMF for actors\stalker_anim
        let mut mp = Vec::new();
        mp.extend_from_slice(&4u16.to_le_bytes());
        mp.extend_from_slice(&1u16.to_le_bytes());
        mp.extend_from_slice(b"default\0");
        mp.extend_from_slice(&1u16.to_le_bytes());
        mp.extend_from_slice(b"bip01\0");
        mp.extend_from_slice(&0u32.to_le_bytes());
        mp.extend_from_slice(&1u16.to_le_bytes());
        mp.extend_from_slice(b"stand\0");
        mp.extend_from_slice(&(motion_def_flags::ESM_IDLE as u32).to_le_bytes());
        for v in [0u16, 0] {
            mp.extend_from_slice(&v.to_le_bytes());
        }
        for f in [1.0f32, 1.0, 1.0, 0.5] {
            mp.extend_from_slice(&f.to_le_bytes());
        }
        mp.extend_from_slice(&0u32.to_le_bytes());

        let mut clip = Vec::new();
        clip.extend_from_slice(b"stand\0");
        clip.extend_from_slice(&1u32.to_le_bytes());
        clip.push(motion_key_flags::FL_RKEY_ABSENT);
        for v in [0i16, 0, 0, 32767] {
            clip.extend_from_slice(&v.to_le_bytes());
        }
        for f in [0.0f32, 2.0, 0.0] {
            clip.extend_from_slice(&f.to_le_bytes());
        }
        let mut ms = ChunkWriter::new();
        ms.write_chunk(0, &1u32.to_le_bytes());
        ms.write_chunk(1, &clip);
        let mut omf_w = ChunkWriter::new();
        omf_w.write_chunk(OGF_S_SMPARAMS, &mp);
        omf_w.write_chunk(OGF_S_MOTIONS, &ms.into_bytes());
        let omf_raw = omf_w.into_bytes();

        // 4. Minimal gamemtl.xr + loose materials.ltx overlay
        let mut mtl0 = ChunkWriter::new();
        let mut m_main = 0u32.to_le_bytes().to_vec();
        m_main.extend_from_slice(b"default_object\0");
        mtl0.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_MAIN, &m_main);
        mtl0.write_chunk(
            gamemtl_chunks::GAMEMTL_CHUNK_FLAGS,
            &gamemtl_flags::FL_SHOOTABLE.to_le_bytes(),
        );
        mtl0.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_PHYSICS, &[0u8; 20]);
        mtl0.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FACTORS, &[0u8; 16]);
        let mut mtls = ChunkWriter::new();
        mtls.write_chunk(0, &mtl0.into_bytes());
        let mut gm = ChunkWriter::new();
        gm.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_VERSION, &1u16.to_le_bytes());
        gm.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_AUTOINC, &[0u8; 8]);
        gm.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_MTLS, &mtls.into_bytes());
        let gm_raw = gm.into_bytes();

        let entries = vec![
            (
                "textures\\ui\\cursor_cross.dds",
                cursor_cross_bytes.clone(),
                compress_lzo1x(&cursor_cross_bytes),
            ),
            (
                "textures\\ui\\cursor_cross.thm",
                thm_raw.clone(),
                compress_lzo1x(&thm_raw),
            ),
            (
                "meshes\\actors\\hero.ogf",
                ogf_raw.clone(),
                compress_lzo1x(&ogf_raw),
            ),
            (
                "meshes\\actors\\stalker_anim.omf",
                omf_raw.clone(),
                compress_lzo1x(&omf_raw),
            ),
            ("gamemtl.xr", gm_raw.clone(), compress_lzo1x(&gm_raw)),
            (
                "materials\\materials.ltx",
                b"[default_object]\nfriction = 0.42\n".to_vec(),
                b"[default_object]\nfriction = 0.42\n".to_vec(),
            ),
        ];

        let temp_dir = std::env::temp_dir().join(format!("xray_formats_vfs_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let archive_path = temp_dir.join("assets.db0");
        std::fs::write(&archive_path, pack_synthetic_db(entries)).unwrap();

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_mmap(&archive_path).unwrap();

        let ogf = load_ogf_from_vfs(&vfs, "$game_meshes$/actors/hero.ogf").unwrap();
        let tex_links = resolve_ogf_texture_links(&vfs, ogf.texture.as_ref().unwrap()).unwrap();
        assert_eq!(tex_links.diffuse_dds, "textures\\ui\\cursor_cross.dds");
        assert_eq!(
            tex_links.bump.as_ref().unwrap().bump_dds,
            "textures\\ui\\cursor_cross_bump.dds"
        );
        assert_eq!(
            tex_links.bump.as_ref().unwrap().bump_hash_dds,
            "textures\\ui\\cursor_cross_bump#.dds"
        );
        assert_eq!(vfs.read_bytes(&tex_links.diffuse_dds).unwrap().len(), 262_272);

        let skel = ogf.skeleton.as_ref().unwrap();
        let omf = load_omf_from_vfs(&vfs, &skel.motion_refs[0]).unwrap();
        let pose = omf.evaluate_skeleton_pose(skel, 0, 0.0).unwrap();
        assert_eq!(pose[0][3], [0.0, 2.0, 0.0, 1.0]);

        let gmlib = load_gamemtl_from_vfs(&vfs).unwrap();
        let mat = gmlib.material_by_name("default_object").unwrap();
        assert!((mat.ph_friction - 0.42).abs() < 1e-6);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_stream_level_bundle_cform_details_and_skinning_from_vfs() {
        use crate::cform::{CformTriangle, CFORM_CURRENT_VERSION};
        use crate::details::{DetailPalette, DetailSlot, DETAIL_ID_EMPTY, DETAIL_VERSION_4};
        use crate::level::{d3d_decl_type, d3d_decl_usage, level_chunks, sector_chunks, VertexElement9};
        use crate::skinning::{skin_vertices, BonePalette};

        // 1. Minimal gamemtl.xr with 1 material at index 0
        let mut mtl0 = ChunkWriter::new();
        let mut m_main = 0u32.to_le_bytes().to_vec();
        m_main.extend_from_slice(b"default_floor\0");
        mtl0.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_MAIN, &m_main);
        mtl0.write_chunk(
            gamemtl_chunks::GAMEMTL_CHUNK_FLAGS,
            &gamemtl_flags::FL_SHOOTABLE.to_le_bytes(),
        );
        mtl0.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_PHYSICS, &[0u8; 20]);
        mtl0.write_chunk(gamemtl_chunks::GAMEMTL_CHUNK_FACTORS, &[0u8; 16]);
        let mut mtls = ChunkWriter::new();
        mtls.write_chunk(0, &mtl0.into_bytes());
        let mut gm = ChunkWriter::new();
        gm.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_VERSION, &1u16.to_le_bytes());
        gm.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_AUTOINC, &[0u8; 8]);
        gm.write_chunk(gamemtl_root_chunks::GAMEMTLS_CHUNK_MTLS, &mtls.into_bytes());
        let gm_raw = gm.into_bytes();

        // 2. Minimal level.cform: horizontal floor quad tri on y = 0.0
        let mut cform_raw = Vec::new();
        cform_raw.extend_from_slice(&CFORM_CURRENT_VERSION.to_le_bytes());
        cform_raw.extend_from_slice(&3u32.to_le_bytes()); // vertcount
        cform_raw.extend_from_slice(&1u32.to_le_bytes()); // facecount
        for f in [-5.0f32, 0.0, -5.0, 5.0, 0.0, 5.0] {
            cform_raw.extend_from_slice(&f.to_le_bytes());
        }
        for v in [[0.0f32, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]] {
            for f in v {
                cform_raw.extend_from_slice(&f.to_le_bytes());
            }
        }
        let tri = CformTriangle {
            verts: [0, 1, 2],
            material: 0,
            suppress_shadows: false,
            suppress_wm: false,
            sector: 0,
        };
        for idx in tri.verts {
            cform_raw.extend_from_slice(&idx.to_le_bytes());
        }
        cform_raw.extend_from_slice(&tri.to_raw_dummy().to_le_bytes());

        // 3. Minimal level.details (v4, 1x1)
        let mut dm_raw = Vec::new();
        dm_raw.extend_from_slice(b"details\\blend\0detail\\gr\0");
        dm_raw.extend_from_slice(&0u32.to_le_bytes());
        dm_raw.extend_from_slice(&0.8f32.to_le_bytes());
        dm_raw.extend_from_slice(&1.2f32.to_le_bytes());
        dm_raw.extend_from_slice(&3u32.to_le_bytes());
        dm_raw.extend_from_slice(&3u32.to_le_bytes());
        dm_raw.extend_from_slice(&[0u8; 3 * 20]);
        for i in [0u16, 1, 2] {
            dm_raw.extend_from_slice(&i.to_le_bytes());
        }
        let mut det_objs = ChunkWriter::new();
        det_objs.write_chunk(0, &dm_raw);
        let mut det_hdr = Vec::new();
        for v in [DETAIL_VERSION_4, 1, 0, 0, 1, 1] {
            det_hdr.extend_from_slice(&v.to_le_bytes());
        }
        let slot = DetailSlot {
            ids: [0, DETAIL_ID_EMPTY, DETAIL_ID_EMPTY, DETAIL_ID_EMPTY],
            c_dir: 15,
            c_hemi: 15,
            y_base: 1000,
            y_height: 10,
            c_r: 15,
            c_g: 15,
            c_b: 15,
            palette: [DetailPalette { a0: 15, a1: 15, a2: 15, a3: 15 }; 4],
        };
        let mut det_w = ChunkWriter::new();
        det_w.write_chunk(0, &det_hdr);
        det_w.write_chunk(1, &det_objs.into_bytes());
        det_w.write_chunk(2, &slot.to_v4_bytes());
        let details_raw = det_w.into_bytes();

        // 4. Minimal level + level.geom (1 visual referencing VB 0 / IB 0)
        let write_elem = |out: &mut Vec<u8>, e: VertexElement9| {
            out.extend_from_slice(&e.stream.to_le_bytes());
            out.extend_from_slice(&e.offset.to_le_bytes());
            out.push(e.decl_type);
            out.push(e.method);
            out.push(e.usage);
            out.push(e.usage_index);
        };
        let mut vb_raw = Vec::new();
        vb_raw.extend_from_slice(&1u32.to_le_bytes());
        write_elem(
            &mut vb_raw,
            VertexElement9 {
                stream: 0,
                offset: 0,
                decl_type: d3d_decl_type::FLOAT3,
                method: 0,
                usage: d3d_decl_usage::POSITION,
                usage_index: 0,
            },
        );
        write_elem(&mut vb_raw, VertexElement9::END);
        vb_raw.extend_from_slice(&3u32.to_le_bytes());
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
            for f in p {
                vb_raw.extend_from_slice(&f.to_le_bytes());
            }
        }
        let mut ib_raw = Vec::new();
        ib_raw.extend_from_slice(&1u32.to_le_bytes());
        ib_raw.extend_from_slice(&3u32.to_le_bytes());
        for i in [0u16, 1, 2] {
            ib_raw.extend_from_slice(&i.to_le_bytes());
        }
        let mut geom_w = ChunkWriter::new();
        geom_w.write_chunk(level_chunks::FSL_VB, &vb_raw);
        geom_w.write_chunk(level_chunks::FSL_IB, &ib_raw);
        let geom_raw = geom_w.into_bytes();

        let mut vis0 = ChunkWriter::new();
        let mut vhdr = vec![4u8, OgfVisualType::Normal as u8, 1, 0];
        vhdr.extend_from_slice(&[0u8; 40]);
        vis0.write_chunk(OGF_HEADER, &vhdr);
        let mut gcont = Vec::new();
        for v in [0u32, 0, 3, 0, 0, 3] {
            gcont.extend_from_slice(&v.to_le_bytes());
        }
        vis0.write_chunk(OGF_GCONTAINER, &gcont);
        let mut vis_wrap = ChunkWriter::new();
        vis_wrap.write_chunk(0, &vis0.into_bytes());

        let mut shd_raw = Vec::new();
        shd_raw.extend_from_slice(&2u32.to_le_bytes());
        shd_raw.extend_from_slice(b"\0levels\\ground/terrain\\ter\0");
        let mut sec0 = ChunkWriter::new();
        sec0.write_chunk(sector_chunks::FSP_ROOT, &0u32.to_le_bytes());
        let mut secs_wrap = ChunkWriter::new();
        secs_wrap.write_chunk(0, &sec0.into_bytes());

        let mut lvl_w = ChunkWriter::new();
        lvl_w.write_chunk(level_chunks::FSL_HEADER, &[14, 0, 1, 0]);
        lvl_w.write_chunk(level_chunks::FSL_SHADERS, &shd_raw);
        lvl_w.write_chunk(level_chunks::FSL_VISUALS, &vis_wrap.into_bytes());
        lvl_w.write_chunk(level_chunks::FSL_SECTORS, &secs_wrap.into_bytes());
        let lvl_raw = lvl_w.into_bytes();

        // 5. Minimal skeletal OGF with inline vertices + OMF for BonePalette verification
        let mut skel_ogf_w = ChunkWriter::new();
        let mut shdr = vec![4u8, OgfVisualType::SkeletonAnim as u8, 0, 0];
        shdr.extend_from_slice(&[0u8; 40]);
        skel_ogf_w.write_chunk(OGF_HEADER, &shdr);
        let mut skel_vb = Vec::new();
        skel_vb.extend_from_slice(&crate::ogf::skel_vert_type::FVF_1L.to_le_bytes());
        skel_vb.extend_from_slice(&1u32.to_le_bytes());
        for f in [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] {
            skel_vb.extend_from_slice(&f.to_le_bytes());
        }
        skel_vb.extend_from_slice(&0u32.to_le_bytes());
        skel_ogf_w.write_chunk(OGF_VERTICES, &skel_vb);
        let mut bn = Vec::new();
        bn.extend_from_slice(&1u32.to_le_bytes());
        bn.extend_from_slice(b"bip01\0\0");
        bn.extend_from_slice(&[0u8; 60]);
        skel_ogf_w.write_chunk(OGF_S_BONE_NAMES, &bn);
        let skel_ogf_raw = skel_ogf_w.into_bytes();

        let entries = vec![
            ("gamemtl.xr", gm_raw.clone(), compress_lzo1x(&gm_raw)),
            ("levels\\test_map\\level.cform", cform_raw.clone(), compress_lzo1x(&cform_raw)),
            ("levels\\test_map\\level.details", details_raw.clone(), compress_lzo1x(&details_raw)),
            ("levels\\test_map\\level.geom", geom_raw.clone(), compress_lzo1x(&geom_raw)),
            ("levels\\test_map\\level", lvl_raw.clone(), compress_lzo1x(&lvl_raw)),
            ("meshes\\actors\\box_skel.ogf", skel_ogf_raw.clone(), compress_lzo1x(&skel_ogf_raw)),
        ];

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("level_pack.db0", pack_synthetic_db(entries))
            .unwrap();

        let gmlib = load_gamemtl_from_vfs(&vfs).unwrap();
        let cform = load_cform_from_vfs(&vfs, "levels/test_map").unwrap();
        let bvh = cform.build_bvh();
        let hit = bvh
            .raycast_nearest([1.0, 5.0, 1.0], [0.0, -1.0, 0.0], 20.0, false)
            .unwrap();
        assert!((hit.range - 5.0).abs() < 1e-5);
        assert_eq!(bvh.hit_material(&hit, &gmlib).unwrap().name, "default_floor");

        let details = load_details_from_vfs(&vfs, "levels\\test_map").unwrap();
        assert_eq!(details.query_slot(0, 0).unwrap().ids[0], 0);

        let level = load_level_from_vfs(&vfs, "levels/test_map").unwrap();
        let geom = load_level_geometry_from_vfs(&vfs, "levels/test_map").unwrap();
        let mesh = level.resolve_visual_mesh(0, &geom).unwrap();
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.shader.unwrap().textures, vec!["terrain\\ter"]);

        let skel_model = load_ogf_from_vfs(&vfs, "meshes\\actors\\box_skel.ogf").unwrap();
        let bind_pal = BonePalette::bind_pose(skel_model.skeleton.as_ref().unwrap()).unwrap();
        let skinned = skin_vertices(skel_model.vertices.as_ref().unwrap(), &bind_pal).unwrap();
        assert_eq!(skinned.len(), 1);
    }

    #[test]
    fn test_stream_hom_ai_sound_and_shader_assets_from_vfs() {
        use crate::ai_graph::{pv_compress_normal, INVALID_AI_LINK_V10, XRAI_CURRENT_VERSION};
        use crate::ogf::XRAY_IDENTITY_MATRIX;
        use crate::shaders_xr::{env_mod_flags, shader_xrlc_flags};

        // 1. level.hom
        let mut hom_poly = Vec::new();
        for p in [[-4.0f32, -4.0, 5.0], [-4.0, 4.0, 5.0], [4.0, 4.0, 5.0]] {
            for f in p {
                hom_poly.extend_from_slice(&f.to_le_bytes());
            }
        }
        hom_poly.extend_from_slice(&1u32.to_le_bytes());
        let mut hom_w = ChunkWriter::new();
        hom_w.write_chunk(1, &hom_poly);
        let hom_raw = hom_w.into_bytes();

        // 2. level.ai (1 vertex v10)
        let mut ai_raw = Vec::new();
        ai_raw.extend_from_slice(&XRAI_CURRENT_VERSION.to_le_bytes());
        ai_raw.extend_from_slice(&1u32.to_le_bytes());
        ai_raw.extend_from_slice(&0.7f32.to_le_bytes());
        ai_raw.extend_from_slice(&10.0f32.to_le_bytes());
        for f in [-5.0f32, 0.0, -5.0, 5.0, 10.0, 5.0] {
            ai_raw.extend_from_slice(&f.to_le_bytes());
        }
        ai_raw.extend_from_slice(&1u64.to_le_bytes());
        ai_raw.extend_from_slice(&2u64.to_le_bytes());
        let bits = (INVALID_AI_LINK_V10 as u128)
            | ((INVALID_AI_LINK_V10 as u128) << 23)
            | ((INVALID_AI_LINK_V10 as u128) << 46)
            | ((INVALID_AI_LINK_V10 as u128) << 69);
        ai_raw.extend_from_slice(&bits.to_le_bytes()[..12]);
        ai_raw.extend_from_slice(&0u16.to_le_bytes());
        ai_raw.extend_from_slice(&0u16.to_le_bytes());
        ai_raw.extend_from_slice(&pv_compress_normal([0.0, 1.0, 0.0]).to_le_bytes());
        ai_raw.extend_from_slice(&0u32.to_le_bytes()[..3]);
        ai_raw.extend_from_slice(&0u16.to_le_bytes());

        // 3. level.gct
        let mut gct_hdr = Vec::new();
        gct_hdr.extend_from_slice(&9u32.to_le_bytes());
        gct_hdr.extend_from_slice(&1u32.to_le_bytes());
        gct_hdr.extend_from_slice(&1u32.to_le_bytes());
        gct_hdr.extend_from_slice(&[0u8; 32]);
        let mut gct_cells = Vec::new();
        gct_cells.extend_from_slice(&0u16.to_le_bytes());
        gct_cells.extend_from_slice(&0.25f32.to_le_bytes());
        let mut gct_w = ChunkWriter::new();
        gct_w.write_chunk(0, &gct_hdr);
        gct_w.write_chunk(1, &gct_cells);
        let gct_raw = gct_w.into_bytes();

        // 4. game.graph
        let mut gg_raw = Vec::new();
        gg_raw.push(10u8);
        gg_raw.extend_from_slice(&1u16.to_le_bytes());
        gg_raw.extend_from_slice(&0u32.to_le_bytes());
        gg_raw.extend_from_slice(&0u32.to_le_bytes());
        gg_raw.extend_from_slice(&[0u8; 16]);
        gg_raw.push(1u8);
        gg_raw.extend_from_slice(b"test_lvl\0");
        gg_raw.extend_from_slice(&[0u8; 12]);
        gg_raw.push(1u8);
        gg_raw.extend_from_slice(b"sec_test\0");
        gg_raw.extend_from_slice(&[0u8; 16]);
        gg_raw.extend_from_slice(&[0u8; 24]); // local + global
        gg_raw.extend_from_slice(&1u32.to_le_bytes()); // level_id = 1, node = 0
        gg_raw.extend_from_slice(&[0u8; 14]); // vertex_types + offsets + counts

        // 5. senvironment.xr + som + snd_env + snd_static
        let mut senv_p = Vec::new();
        senv_p.extend_from_slice(&4u32.to_le_bytes());
        senv_p.extend_from_slice(b"room\0");
        senv_p.extend_from_slice(&[0u8; 52]);
        let mut senv_w = ChunkWriter::new();
        senv_w.write_chunk(0, &senv_p);
        let senv_raw = senv_w.into_bytes();

        let mut som_p = Vec::new();
        for p in [[-2.0f32, -2.0, 2.0], [-2.0, 2.0, 2.0], [2.0, 0.0, 2.0]] {
            for f in p {
                som_p.extend_from_slice(&f.to_le_bytes());
            }
        }
        som_p.extend_from_slice(&1u32.to_le_bytes());
        som_p.extend_from_slice(&0.5f32.to_le_bytes());
        let mut som_w = ChunkWriter::new();
        som_w.write_chunk(0, &0u32.to_le_bytes());
        som_w.write_chunk(1, &som_p);
        let som_raw = som_w.into_bytes();

        let mut snd_env_cf = Vec::new();
        snd_env_cf.extend_from_slice(&4u32.to_le_bytes());
        snd_env_cf.extend_from_slice(&3u32.to_le_bytes());
        snd_env_cf.extend_from_slice(&1u32.to_le_bytes());
        snd_env_cf.extend_from_slice(&[0u8; 24]);
        for p in [[-5.0f32, 0.0, -5.0], [-5.0, 0.0, 5.0], [5.0, 0.0, -5.0]] {
            for f in p {
                snd_env_cf.extend_from_slice(&f.to_le_bytes());
            }
        }
        for i in [0u32, 1, 2, 0] {
            snd_env_cf.extend_from_slice(&i.to_le_bytes());
        }
        let mut snde_w = ChunkWriter::new();
        snde_w.write_chunk(0, b"room\0");
        snde_w.write_chunk(1, &snd_env_cf);
        let snde_raw = snde_w.into_bytes();

        let mut snd_obj = Vec::new();
        snd_obj.extend_from_slice(b"amb\\hum\0");
        snd_obj.extend_from_slice(&[0u8; 44]);
        let mut snd_sub = ChunkWriter::new();
        snd_sub.write_chunk(0, &snd_obj);
        let mut snds_w = ChunkWriter::new();
        snds_w.write_chunk(0, &snd_sub.into_bytes());
        let snds_raw = snds_w.into_bytes();

        // 6. shaders_xrlc.xr + shaders.xr + env_mod + ltx + ps_static + fog_vol
        let mut xrlc_raw = vec![0u8; 144];
        xrlc_raw[..4].copy_from_slice(b"wall");
        xrlc_raw[128..132].copy_from_slice(&shader_xrlc_flags::COLLISION.to_le_bytes());

        let mut c0 = Vec::new();
        c0.extend_from_slice(b"c0\0");
        c0.extend_from_slice(&[0u8; 80]);
        let mut shd_w = ChunkWriter::new();
        shd_w.write_chunk(0, &c0);
        let shd_raw = shd_w.into_bytes();

        let mut em_p = Vec::new();
        for f in [0.0f32, 0.0, 0.0, 10.0, 1.0, 200.0] {
            em_p.extend_from_slice(&f.to_le_bytes());
        }
        em_p.extend_from_slice(&[0u8; 52]);
        em_p.extend_from_slice(&env_mod_flags::VIEW_DIST.to_le_bytes());
        let mut em_w = ChunkWriter::new();
        em_w.write_chunk(0, &0x0016u32.to_le_bytes());
        em_w.write_chunk(1, &em_p);
        let em_raw = em_w.into_bytes();

        let em_ltx = b"[zone_fog]\nposition = 0,0,0\nradius = 25\npower = 1.0\nfog_density = 0.75\n";

        let mut ps_entry = Vec::new();
        ps_entry.extend_from_slice(&0xFFFFu16.to_le_bytes());
        ps_entry.extend_from_slice(b"pfx\\sparks\0");
        for r in XRAY_IDENTITY_MATRIX {
            for c in r {
                ps_entry.extend_from_slice(&c.to_le_bytes());
            }
        }
        let mut ps_w = ChunkWriter::new();
        ps_w.write_chunk(0, &1u32.to_le_bytes());
        ps_w.write_chunk(1, &ps_entry);
        let ps_raw = ps_w.into_bytes();

        let mut fv_raw = Vec::new();
        fv_raw.extend_from_slice(&2u16.to_le_bytes());
        fv_raw.extend_from_slice(&1u32.to_le_bytes());
        for r in XRAY_IDENTITY_MATRIX {
            for c in r {
                fv_raw.extend_from_slice(&c.to_le_bytes());
            }
        }
        fv_raw.extend_from_slice(&0u32.to_le_bytes());

        let entries = vec![
            ("levels\\test_lvl\\level.hom", hom_raw.clone(), compress_lzo1x(&hom_raw)),
            ("levels\\test_lvl\\level.ai", ai_raw.clone(), compress_lzo1x(&ai_raw)),
            ("levels\\test_lvl\\level.gct", gct_raw.clone(), compress_lzo1x(&gct_raw)),
            ("game.graph", gg_raw.clone(), compress_lzo1x(&gg_raw)),
            ("senvironment.xr", senv_raw.clone(), compress_lzo1x(&senv_raw)),
            ("levels\\test_lvl\\level.som", som_raw.clone(), compress_lzo1x(&som_raw)),
            ("levels\\test_lvl\\level.snd_env", snde_raw.clone(), compress_lzo1x(&snde_raw)),
            ("levels\\test_lvl\\level.snd_static", snds_raw.clone(), compress_lzo1x(&snds_raw)),
            ("shaders_xrlc.xr", xrlc_raw.clone(), compress_lzo1x(&xrlc_raw)),
            ("shaders.xr", shd_raw.clone(), compress_lzo1x(&shd_raw)),
            ("levels\\test_lvl\\level.env_mod", em_raw.clone(), compress_lzo1x(&em_raw)),
            ("configs\\environment\\env_mods\\ltx_map.ltx", em_ltx.to_vec(), em_ltx.to_vec()),
            ("levels\\test_lvl\\level.ps_static", ps_raw.clone(), compress_lzo1x(&ps_raw)),
            ("levels\\test_lvl\\level.fog_vol", fv_raw.clone(), compress_lzo1x(&fv_raw)),
        ];

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("world_extras.db0", pack_synthetic_db(entries))
            .unwrap();

        assert_eq!(load_hom_from_vfs(&vfs, "levels/test_lvl").unwrap().triangles.len(), 1);
        assert_eq!(load_level_ai_from_vfs(&vfs, "levels/test_lvl").unwrap().vertices.len(), 1);
        assert_eq!(load_level_gct_from_vfs(&vfs, "levels/test_lvl").unwrap().cells.len(), 1);
        assert_eq!(load_game_graph_from_vfs(&vfs, None).unwrap().vertices.len(), 1);
        assert!(load_sound_environments_from_vfs(&vfs).unwrap().find_by_name("room").is_some());
        assert!(
            (load_som_from_vfs(&vfs, "levels/test_lvl")
                .unwrap()
                .query_occlusion([0.0, 0.0, 0.0], [0.0, 0.0, 5.0])
                - 0.5)
                .abs()
                < 1e-5
        );
        assert_eq!(
            load_level_sound_env_from_vfs(&vfs, "levels/test_lvl")
                .unwrap()
                .query_env_name_at([0.0, 1.0, 0.0]),
            Some("room")
        );
        assert_eq!(
            load_static_sounds_from_vfs(&vfs, "levels/test_lvl")
                .unwrap()
                .sounds[0]
                .wav_name,
            "amb\\hum"
        );
        assert!(load_shaders_xrlc_from_vfs(&vfs).unwrap().find("wall").is_some());
        assert!(load_shaders_xr_from_vfs(&vfs).unwrap().find_constant("c0").is_some());

        let bin_mods = load_env_mods_from_vfs(&vfs, "levels/test_lvl", None).unwrap();
        assert!((bin_mods.evaluate_at([0.0, 0.0, 0.0]).far_plane - 200.0).abs() < 1e-4);

        let ltx_mods = load_env_mods_from_vfs(&vfs, "levels/test_lvl", Some("ltx_map")).unwrap();
        assert!((ltx_mods.evaluate_at([0.0, 0.0, 0.0]).fog_density - 0.75).abs() < 1e-4);

        assert_eq!(
            load_ps_static_from_vfs(&vfs, "levels/test_lvl")
                .unwrap()
                .particles[0]
                .ref_name,
            "pfx\\sparks"
        );
        assert_eq!(
            load_fog_volumes_from_vfs(&vfs, "levels/test_lvl")
                .unwrap()
                .volumes
                .len(),
            1
        );
    }

    #[test]
    fn test_stream_particles_envelopes_and_spawns_from_vfs() {
        use crate::ai_graph::{
            GameDeathPoint, GameGraph, GameGraphHeader, GameLevelInfo, GameVertex,
            XRAI_CURRENT_VERSION,
        };
        use crate::envelopes::{
            Envelope, EnvelopeBehavior, EnvelopeKey, EnvelopeShape, ObjectMotion,
            PostprocessColorCurve, PostprocessEffect, EOBJ_OMOTION_VERSION,
            POSTPROCESS_FILE_VERSION,
        };
        use crate::particles::{ped_flags, ParticleEffectDef, ParticlesLibrary};
        use crate::spawns::{
            alife_object_flags, cse_spawn_flags, m_spawn_flags, AlifeObjectState,
            AlifeSpawnArchive, LevelGameData, PatrolEdge, PatrolPoint, RPoint, RPointType,
            SpawnHeader, SpawnPacket, SpawnVertex, INVALID_SPAWN_STORY_ID, INVALID_STORY_ID,
            M_UPDATE, SPAWN_VERSION,
        };

        // 1. particles.xr + loose .pe override + .pg group + wallmarks.ltx
        let base_eff = ParticleEffectDef {
            name: "pfx\\sparks".to_string(),
            flags: ped_flags::DF_SPRITE,
            max_particles: 16,
            update_step_ms: 33,
            shader: Some("effects\\add".to_string()),
            texture: Some("fx\\fx_spark".to_string()),
            frame: None,
            time_limit: None,
            collision: None,
            velocity_scale: None,
            align_to_path_rotation: None,
            actions: vec![],
        };
        let pxr_raw = ParticlesLibrary {
            effects: vec![base_eff],
            groups: vec![],
        }
        .to_bytes();

        let pe_override = b"[_effect]\nversion = 1\nmax_particles = 96\nupdate_step = 25\nflags = 0\naction_count = 0\n";
        let pg_ltx = b"[_group]\nflags = 0\neffects_count = 1\ntimelimit = 3.0\n\n[effect_0000]\neffect_name = pfx\\sparks\non_play_child =\non_birth_child =\non_death_child =\ntime0 = 0.0\ntime1 = 3.0\nflags = 4\n";
        let wm_ltx = b"[m_blood_marks]\nwallmarks = wm\\wm_blood_1\nmin_size = 0.4\nmax_size = 1.1\ndist = 2.0\nnominal_hit = 0.5\nblood_drops = wm\\wm_blood_drop_1\nstart_blood_size = 0.3\nstop_blood_size = 0.02\nblood_drop_size = 0.1\n";

        // 2. .anm and .ppe (level override + global $game_anims$)
        let lin = |time: f32, value: f32| EnvelopeKey {
            value,
            time,
            shape: EnvelopeShape::Linear,
            ..EnvelopeKey::default()
        };
        let env_const = |v: f32| Envelope {
            behavior: [EnvelopeBehavior::Constant, EnvelopeBehavior::Constant],
            keys: vec![lin(0.0, v)],
        };
        let anm_raw = ObjectMotion {
            name: "cam_cutscene".to_string(),
            frame_start: 0,
            frame_end: 30,
            fps: 30.0,
            version: EOBJ_OMOTION_VERSION,
            envelopes: [
                env_const(4.0),
                env_const(5.0),
                env_const(6.0),
                env_const(0.5),
                env_const(0.25),
                env_const(0.0),
            ],
        }
        .to_bytes();

        let ppe_global = PostprocessEffect {
            version: POSTPROCESS_FILE_VERSION,
            base_color: PostprocessColorCurve::default(),
            add_color: PostprocessColorCurve::default(),
            gray_color: PostprocessColorCurve::default(),
            gray: env_const(0.2),
            blur: Envelope::default(),
            duality_h: Envelope::default(),
            duality_v: Envelope::default(),
            noise_intensity: Envelope::default(),
            noise_granularity: Envelope::default(),
            noise_fps: Envelope::default(),
            cm_influence: Envelope::default(),
            cm_tex1: String::new(),
        }
        .to_bytes();

        let mut ppe_level_eff = PostprocessEffect::parse(&ppe_global).unwrap();
        ppe_level_eff.gray = env_const(0.9);
        let ppe_level = ppe_level_eff.to_bytes();

        // 3. level.game + patrol_paths.ltx (+ DLTX override) + level.spawn + all.spawn
        let patrol = PatrolPath {
            name: "esc_guard_walk".to_string(),
            level_override: None,
            way_type: 0,
            points: vec![
                PatrolPoint {
                    vertex_id: 0,
                    name: "wp00".to_string(),
                    pos: [0.0, 0.0, 0.0],
                    flags: 1,
                    level_vertex_id: 10,
                    game_vertex_id: 0,
                },
                PatrolPoint {
                    vertex_id: 1,
                    name: "wp01".to_string(),
                    pos: [10.0, 0.0, 0.0],
                    flags: 2,
                    level_vertex_id: 11,
                    game_vertex_id: 0,
                },
            ],
            edges: vec![PatrolEdge {
                from: 0,
                to: 1,
                probability: 1.0,
            }],
        };
        let level_game_raw = LevelGameData {
            rpoints: vec![RPoint {
                pos: [1.0, 2.0, 3.0],
                rot: [0.0, 0.5, 0.0],
                team: 1,
                point_type: RPointType::ItemSpawn,
                game_type: u16::MAX,
                rp_profile: Some("medkit".to_string()),
            }],
            patrol_paths: vec![patrol.clone()],
        }
        .to_bytes();

        let patrol_ltx = b"[esc_guard_walk]\npoints = p0\np0:name = wp00\np0:position = 1,2,3\np0:level_vertex_id = 5\np0:game_vertex_id = 0\n";
        let patrol_dltx = b"![esc_guard_walk]\np0:flags = 7\n";

        let state_raw = AlifeObjectState {
            game_vertex_id: 0,
            distance: 0.0,
            direct_control: true,
            level_vertex_id: 10,
            flags: cse_spawn_flags::ENABLED | alife_object_flags::USED_AI_LOCATIONS,
            custom_data: "[logic]\nactive = walker\n".to_string(),
            story_id: INVALID_STORY_ID,
            spawn_story_id: INVALID_SPAWN_STORY_ID,
            subclass_payload: vec![],
        }
        .to_bytes();
        let spawn_pkt = SpawnPacket {
            section: "stalker".to_string(),
            name_replace: "esc_wolf".to_string(),
            game_id_legacy: 0,
            rp: 0xFE,
            position: [1.0, 0.0, 2.0],
            angle: [0.0; 3],
            respawn_time: 0,
            id: 0xFFFF,
            parent_id: 0xFFFF,
            phantom_id: 0xFFFF,
            flags: m_spawn_flags::OBJECT_LOCAL | m_spawn_flags::VERSION,
            version: SPAWN_VERSION,
            game_type: u16::MAX,
            script_version: 10,
            client_data: vec![],
            spawn_id: 0,
            state_payload: state_raw,
            trailing_update_payload: vec![],
        };
        let level_spawn_raw = crate::spawns::LevelSpawnData {
            spawns: vec![spawn_pkt.clone()],
        }
        .to_bytes();

        let gg = GameGraph {
            header: GameGraphHeader {
                version: XRAI_CURRENT_VERSION as u8,
                vertex_count: 1,
                edge_count: 0,
                death_point_count: 0,
                guid: [0x11, 0x22],
                levels: vec![GameLevelInfo {
                    name: "l01_escape".to_string(),
                    offset: [0.0; 3],
                    id: 1,
                    section: "level01".to_string(),
                    guid: [0x33, 0x44],
                }],
            },
            vertices: vec![GameVertex {
                local_point: [0.0; 3],
                global_point: [0.0; 3],
                level_id: 1,
                level_vertex_id: 10,
                vertex_types: [0; 4],
                edge_offset: 0,
                point_offset: 0,
                neighbour_count: 0,
                death_point_count: 0,
            }],
            edges: vec![],
            death_points: vec![],
            cross_tables: vec![],
        };
        let mut graph_guid = [0u8; 16];
        graph_guid[0..8].copy_from_slice(&0x11u64.to_le_bytes());
        graph_guid[8..16].copy_from_slice(&0x22u64.to_le_bytes());
        let all_spawn_raw = AlifeSpawnArchive {
            header: SpawnHeader {
                version: XRAI_CURRENT_VERSION,
                guid: [0xAA; 16],
                graph_guid,
                spawn_count: 1,
                level_count: 1,
            },
            spawns: vec![SpawnVertex {
                spawn_id: 0,
                spawn: spawn_pkt,
                update_packet: M_UPDATE.to_le_bytes().to_vec(),
            }],
            edges: vec![],
            artefact_spawn_positions: vec![GameDeathPoint {
                point: [3.0, 0.0, 4.0],
                level_vertex_id: 10,
                distance: 1.5,
            }],
            patrol_paths: vec![patrol],
            game_graph: gg,
        }
        .to_bytes();

        let entries = vec![
            ("particles.xr", pxr_raw.clone(), compress_lzo1x(&pxr_raw)),
            ("particles\\pfx\\sparks.pe", pe_override.to_vec(), pe_override.to_vec()),
            ("particles\\anom\\fire.pg", pg_ltx.to_vec(), pg_ltx.to_vec()),
            ("configs\\creatures\\blood.ltx", wm_ltx.to_vec(), wm_ltx.to_vec()),
            ("anims\\cam_cutscene.anm", anm_raw.clone(), compress_lzo1x(&anm_raw)),
            ("anims\\fade.ppe", ppe_global.clone(), compress_lzo1x(&ppe_global)),
            ("levels\\test_lvl\\fade.ppe", ppe_level.clone(), compress_lzo1x(&ppe_level)),
            ("levels\\test_lvl\\level.game", level_game_raw.clone(), compress_lzo1x(&level_game_raw)),
            ("configs\\patrol_paths.ltx", patrol_ltx.to_vec(), patrol_ltx.to_vec()),
            ("configs\\mod_patrol_paths.ltx", patrol_dltx.to_vec(), patrol_dltx.to_vec()),
            ("levels\\test_lvl\\level.spawn", level_spawn_raw.clone(), compress_lzo1x(&level_spawn_raw)),
            ("spawns\\all.spawn", all_spawn_raw.clone(), compress_lzo1x(&all_spawn_raw)),
        ];

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("assets_p2.db0", pack_synthetic_db(entries))
            .unwrap();

        let plib = load_particles_from_vfs(&vfs).unwrap();
        assert_eq!(plib.find_effect("pfx\\sparks").unwrap().max_particles, 96);
        assert_eq!(plib.find_group("anom\\fire").unwrap().effects[0].effect_name, "pfx\\sparks");

        let wm = load_blood_wallmarks_from_vfs(&vfs, "$game_configs$/creatures/blood.ltx", "m_blood_marks").unwrap();
        assert_eq!(wm.wallmarks.textures, vec!["wm\\wm_blood_1"]);

        let mot = load_anm_from_vfs(&vfs, "cam_cutscene").unwrap();
        assert_eq!(mot.evaluate(0.0), ([4.0, 5.0, 6.0], [0.25, 0.5, 0.0]));

        // Level-local fade.ppe overrides $game_anims$/fade.ppe; without level_dir uses $game_anims$
        let ppe_l = load_ppe_from_vfs(&vfs, "fade.ppe", Some("levels/test_lvl")).unwrap();
        assert!((ppe_l.evaluate(0.0).gray - 0.9).abs() < 1e-5);
        let ppe_g = load_ppe_from_vfs(&vfs, "fade.ppe", None).unwrap();
        assert!((ppe_g.evaluate(0.0).gray - 0.2).abs() < 1e-5);

        let lgame = load_level_game_from_vfs(&vfs, "levels/test_lvl").unwrap();
        assert_eq!(lgame.rpoints[0].rp_profile.as_deref(), Some("medkit"));
        assert_eq!(lgame.find_patrol_path("esc_guard_walk").unwrap().points.len(), 2);

        let ppaths = load_patrol_paths_ltx_from_vfs(&vfs, &["$game_configs$/mod_patrol_paths.ltx"]).unwrap();
        assert_eq!(ppaths[0].points[0].flags, 7);

        let lspawn = load_level_spawn_from_vfs(&vfs, "levels/test_lvl").unwrap();
        assert_eq!(lspawn.spawns[0].decode_alife_state().unwrap().level_vertex_id, 10);

        let aspawn = load_all_spawn_from_vfs(&vfs, "all").unwrap();
        assert_eq!(aspawn.spawns[0].spawn.name_replace, "esc_wolf");
        assert_eq!(aspawn.artefact_spawn_positions.len(), 1);
        assert_eq!(aspawn.game_graph.header.levels[0].name, "l01_escape");
    }

    #[test]
    fn test_stream_scop_weather_and_xml_ui_from_vfs_archive() {
        use crate::savegames::{
            GameDateTime, GameTaskRegistryData, SaveGameTime, SaveObjectRecord, SaveRegistries,
            SaveSpawnHeader, SavegameFile, ALIFE_VERSION,
        };
        use crate::spawns::{cse_spawn_flags, m_spawn_flags, AlifeObjectState, SpawnPacket, M_UPDATE, SPAWN_VERSION};
        use crate::weather::EnvBoost;
        use crate::xml_ui::{color_argb, UiWindowDef};

        // 1. Build a compressed `.scop` ALife savegame with an actor at 0.85 health
        let actor_state = AlifeObjectState {
            game_vertex_id: 1,
            distance: 0.0,
            direct_control: true,
            level_vertex_id: 42,
            flags: cse_spawn_flags::ENABLED,
            custom_data: String::new(),
            story_id: u32::MAX,
            spawn_story_id: u32::MAX,
            subclass_payload: vec![],
        }
        .to_bytes();
        let actor_spawn = SpawnPacket {
            section: "actor".to_string(),
            name_replace: "single_player".to_string(),
            game_id_legacy: 0,
            rp: 0xFE,
            position: [12.0, 1.0, -5.0],
            angle: [0.0; 3],
            respawn_time: 0,
            id: 0,
            parent_id: u16::MAX,
            phantom_id: u16::MAX,
            flags: m_spawn_flags::OBJECT_LOCAL | m_spawn_flags::VERSION,
            version: SPAWN_VERSION,
            game_type: u16::MAX,
            script_version: 10,
            client_data: vec![],
            spawn_id: 0,
            state_payload: actor_state,
            trailing_update_payload: vec![],
        };
        let save = SavegameFile {
            alife_version: ALIFE_VERSION,
            time: SaveGameTime {
                game_time_ms: GameDateTime {
                    year: 2012,
                    month: 6,
                    day: 15,
                    hour: 9,
                    minute: 30,
                    second: 0,
                    millisecond: 0,
                }
                .to_game_time_ms(),
                time_factor: 10.0,
                normal_time_factor: 10.0,
            },
            spawn_info: SaveSpawnHeader {
                spawn_name: "all".to_string(),
                spawn_guid: [0x5A; 16],
                level_id: Some(1),
                level_name: Some("l01_escape".to_string()),
                spawn_updates: vec![],
            },
            objects: vec![SaveObjectRecord {
                spawn: actor_spawn,
                update_msg_id: M_UPDATE,
                update_payload: 0.85f32.to_le_bytes().to_vec(),
            }],
            registries: SaveRegistries {
                game_tasks: GameTaskRegistryData::default(),
                ..Default::default()
            },
            extra_chunks: vec![],
        };
        let scop_bytes = save.to_bytes().unwrap();

        // 2. Weather LTX suite under configs\environment\
        let env_ltx = b"[environment]\naltitude = 20\ndelta_longitude = 30\nmin_dist_factor = 0.8\ntilt = 15\nsecond_propability = 0.25\nsky_color = 0.2\nsun_color = 0.6\nfog_color = 0.3\n";
        let clear_cycle_ltx = b"[00:00:00]\nambient_color = 0.2, 0.2, 0.2\nsky_color = 0.2, 0.2, 0.3\nhemi_color = 0.3, 0.3, 0.35, 1.0\nfog_distance = 300\nfog_density = 0.4\nfar_plane = 400\n[12:00:00]\nambient_color = 0.8, 0.8, 0.8\nsky_color = 0.8, 0.9, 1.0\nhemi_color = 0.9, 0.9, 1.0, 1.0\nfog_distance = 500\nfog_density = 0.1\nfar_plane = 800\n";
        let wfx_surge_ltx = b"[00:00:00]\nfog_density = 0.9\nwind_velocity = 20.0\n";

        // 3. UI XML + color_defs + textures_descr + eng/rus CP1251 localization
        let colors_xml = b"<colors><color name=\"hud_green\" r=\"30\" g=\"220\" b=\"60\" a=\"240\"/></colors>";
        let layout_root_xml = b"<w>\n#include \"ui\\sub_panel.xml\"\n</w>";
        let layout_sub_xml = b"<hud_panel x=\"50\" y=\"60\" width=\"300\" height=\"80\" stretch=\"1\"><caption x=\"0\" y=\"0\" width=\"300\" height=\"20\"><text color=\"hud_green\" font=\"letterica16\" align=\"c\">st_hud_ready</text></caption></hud_panel>";
        let tex_descr_xml = b"<w><file name=\"ui\\ui_hud\"><texture id=\"ui_hud_bar\" x=\"4\" y=\"8\" width=\"120\" height=\"16\"/></file></w>";
        let eng_xml = b"<?xml version=\"1.0\" encoding=\"windows-1251\"?>\n<string_table>\n\t<string id=\"st_hud_ready\"><text>Ready</text></string>\n\t<string id=\"st_only_english\"><text>Fallback $$ACTION_USE$$</text></string>\n</string_table>";
        // CP1251 Russian string for st_hud_ready ("Готово")
        let mut rus_xml = b"<?xml version=\"1.0\" encoding=\"windows-1251\"?>\n<string_table>\n\t<string id=\"st_hud_ready\"><text>".to_vec();
        rus_xml.extend_from_slice(b"\xC3\xEE\xF2\xEE\xE2\xEE");
        rus_xml.extend_from_slice(b"</text></string>\n</string_table>");

        let entries = vec![
            ("savedgames\\quicksave.scop", scop_bytes.clone(), scop_bytes),
            ("configs\\environment\\environment.ltx", env_ltx.to_vec(), env_ltx.to_vec()),
            ("configs\\environment\\weathers\\default_clear.ltx", clear_cycle_ltx.to_vec(), clear_cycle_ltx.to_vec()),
            ("configs\\environment\\weather_effects\\fx_surge.ltx", wfx_surge_ltx.to_vec(), wfx_surge_ltx.to_vec()),
            ("configs\\ui\\color_defs.xml", colors_xml.to_vec(), colors_xml.to_vec()),
            ("configs\\ui\\main_hud.xml", layout_root_xml.to_vec(), layout_root_xml.to_vec()),
            ("configs\\ui\\sub_panel.xml", layout_sub_xml.to_vec(), layout_sub_xml.to_vec()),
            ("configs\\ui\\textures_descr\\hud.xml", tex_descr_xml.to_vec(), tex_descr_xml.to_vec()),
            ("configs\\text\\eng\\st_hud.xml", eng_xml.to_vec(), eng_xml.to_vec()),
            ("configs\\text\\rus\\st_hud.xml", rus_xml.clone(), rus_xml),
        ];

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("assets_p3.db0", pack_synthetic_db(entries))
            .unwrap();

        let loaded_save = load_scop_from_vfs(&vfs, "savedgames/quicksave.scop").unwrap();
        assert_eq!(loaded_save.spawn_info.level_name.as_deref(), Some("l01_escape"));
        assert!((loaded_save.actor_health().unwrap() - 0.85).abs() < 1e-6);
        assert_eq!(loaded_save.time.date_time().hour, 9);

        let wlib = load_weather_library_from_vfs(&vfs, None).unwrap();
        assert!((wlib.global_params.second_probability - 0.25).abs() < 1e-6);
        assert!(wlib.effects_cycles.contains_key("fx_surge"));
        let noon = wlib
            .sample_at("default_clear", 21_600.0, None, [0.0; 3], 1.0, &EnvBoost::default())
            .unwrap();
        assert!((noon.ambient_color[0] - 0.5).abs() < 1e-4);

        let colors = load_color_defs_from_vfs(&vfs).unwrap();
        let ui_doc = load_xml_from_vfs(&vfs, "$game_configs$/ui/main_hud.xml").unwrap();
        let panel = UiWindowDef::from_xml_node(ui_doc.navigate("hud_panel", 0).unwrap(), &colors);
        let caption = panel.find_child("caption").unwrap();
        assert_eq!(caption.text.as_ref().unwrap().color_argb, color_argb(240, 30, 220, 60));

        let atlases = load_ui_texture_atlases_from_vfs(&vfs).unwrap();
        assert_eq!(atlases.get("ui_hud_bar").unwrap().dds_vpath(), "textures\\ui\\ui_hud.dds");

        let loc_rus = load_localization_table_from_vfs(&vfs, "rus").unwrap();
        assert_eq!(loc_rus.translate("st_hud_ready"), "Готово");
        let mut keys = std::collections::BTreeMap::new();
        keys.insert("USE".into(), "E".into());
        assert_eq!(loc_rus.translate_formatted("st_only_english", &keys), "Fallback E");
    }

    #[test]
    fn test_stream_lanims_and_simulate_particle_billboards_from_vfs() {
        use crate::lanims::{pack_argb, LightAnimItem, LightAnimLibrary, LANIM_VERSION};
        use crate::particle_sim::ParticleGroupInstance;
        use crate::particles::{
            papi_flags, ped_flags, pg_effect_flags, CompiledParticleAction, GroupEffectItem,
            ParticleActionKind, ParticleDomain, ParticleDomainType, ParticleEffectDef,
            ParticleGroupDef, ParticlesLibrary,
        };

        let mut pulse = LightAnimItem::new("lights\\campfire");
        pulse.fps = 10.0;
        pulse.frame_count = 10;
        pulse.keys.clear();
        pulse.insert_key(0, pack_argb(0xFF, 0xFF, 0x80, 0x00)).unwrap();
        pulse.insert_key(10, pack_argb(0xFF, 0xFF, 0xFF, 0x40)).unwrap();
        let lanim_raw = LightAnimLibrary {
            version: LANIM_VERSION,
            items: vec![pulse],
        }
        .to_bytes();

        let spark_eff = ParticleEffectDef {
            name: "fire\\sparks".to_string(),
            flags: ped_flags::DF_SPRITE,
            max_particles: 8,
            // Binary particles.xr (`CPEDef::Load`) always decodes with `m_uStep = 33` ms
            update_step_ms: 33,
            shader: Some("effects\\add".to_string()),
            texture: Some("fx\\fx_spark".to_string()),
            frame: None,
            time_limit: None,
            collision: None,
            velocity_scale: None,
            align_to_path_rotation: None,
            actions: vec![
                CompiledParticleAction {
                    outer_type: 21,
                    flags: papi_flags::PASOURCE_VERTEXB_TRACKS,
                    inner_type: 21,
                    kind: ParticleActionKind::Source {
                        position: ParticleDomain::default(),
                        velocity: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        rot: ParticleDomain::default(),
                        size: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        color: ParticleDomain::from_editor_params(
                            ParticleDomainType::Point,
                            [1.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        alpha: 1.0,
                        particle_rate: 100.0,
                        age: 0.0,
                        age_sigma: 0.0,
                        parent_vel: [0.0; 3],
                        parent_motion: 0.0,
                    },
                },
                CompiledParticleAction {
                    outer_type: 12,
                    flags: 0,
                    inner_type: 12,
                    kind: ParticleActionKind::Move,
                },
            ],
        };
        let campfire_grp = ParticleGroupDef {
            name: "fire\\campfire".to_string(),
            flags: 0,
            time_limit: 5.0,
            effects: vec![GroupEffectItem {
                effect_name: "fire\\sparks".to_string(),
                on_play_child: String::new(),
                on_birth_child: String::new(),
                on_dead_child: String::new(),
                time0: 0.0,
                time1: 5.0,
                flags: pg_effect_flags::FL_ENABLED,
            }],
        };
        let pxr_raw = ParticlesLibrary {
            effects: vec![spark_eff],
            groups: vec![campfire_grp],
        }
        .to_bytes();

        let entries = vec![
            ("lanims.xr", lanim_raw.clone(), compress_lzo1x(&lanim_raw)),
            ("particles.xr", pxr_raw.clone(), compress_lzo1x(&pxr_raw)),
        ];
        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("fx_pack.db0", pack_synthetic_db(entries))
            .unwrap();

        let lalib = load_lanims_from_vfs(&vfs).unwrap();
        let campfire = lalib.find_item("lights\\campfire").unwrap();
        assert_eq!(campfire.calculate_rgb(0.5).1, 5);

        let plib = load_particles_from_vfs(&vfs).unwrap();
        let grp_def = plib.find_group("fire\\campfire").unwrap().clone();
        let mut grp_inst = ParticleGroupInstance::compile(grp_def, &plib, 123).unwrap();
        grp_inst.play();
        grp_inst.step(66, &plib);
        let batches = grp_inst.emit_billboards([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(batches.len(), 1);
        // 2 ticks of 33ms * 100 particles/sec = 6.6 -> 6..=8 particles capped by max_particles=8
        assert!((6..=8).contains(&batches[0].instances.len()));
        let (verts, idx) = batches[0].instances[0].expand_quad(0);
        assert_eq!(verts.len(), 4);
        assert_eq!(idx, [0, 1, 2, 3, 2, 1]);
    }

    #[test]
    fn test_stream_anms_bone_parts_and_ogg_metadata_from_vfs() {
        use crate::envelopes::{
            Envelope, EnvelopeBehavior, EnvelopeKey, EnvelopeShape, ObjectMotion,
            ObjectMotionLibrary,
        };
        use crate::sound_env::{OggXRayParams, OGG_COMMENT_VERSION};

        let make_env = |val: f32| Envelope {
            behavior: [EnvelopeBehavior::Constant, EnvelopeBehavior::Constant],
            keys: vec![EnvelopeKey {
                value: val,
                time: 0.0,
                shape: EnvelopeShape::Linear,
                ..EnvelopeKey::default()
            }],
        };
        let anms_raw = ObjectMotionLibrary {
            motions: vec![
                ObjectMotion {
                    name: "cam_hit".to_string(),
                    frame_start: 0,
                    frame_end: 30,
                    fps: 30.0,
                    version: 5,
                    envelopes: [
                        make_env(1.0),
                        make_env(0.0),
                        make_env(0.0),
                        make_env(0.0),
                        make_env(0.0),
                        make_env(0.0),
                    ],
                },
                ObjectMotion {
                    name: "cam_walk".to_string(),
                    frame_start: 0,
                    frame_end: 60,
                    fps: 30.0,
                    version: 5,
                    envelopes: [
                        make_env(0.0),
                        make_env(0.5),
                        make_env(0.0),
                        make_env(0.0),
                        make_env(0.0),
                        make_env(0.0),
                    ],
                },
            ],
        }
        .to_bytes();

        let bone_parts_ltx = b"[part_0]\nbip01 =\npartition_name = default\n[part_1]\nl_hand =\npartition_name = left_hand\n";

        // Minimal 2-page Ogg Vorbis stream with X-Ray OGG_COMMENT_VERSION 0x0003 metadata
        let make_ogg_page = |seq: u32, hdr_type: u8, granule: u64, pkt: &[u8]| -> Vec<u8> {
            let mut p = Vec::new();
            p.extend_from_slice(b"OggS");
            p.push(0);
            p.push(hdr_type);
            p.extend_from_slice(&granule.to_le_bytes());
            p.extend_from_slice(&42u32.to_le_bytes());
            p.extend_from_slice(&seq.to_le_bytes());
            p.extend_from_slice(&0u32.to_le_bytes());
            p.push(1);
            p.push(pkt.len() as u8);
            p.extend_from_slice(pkt);
            p
        };
        let mut id_pkt = vec![0x01];
        id_pkt.extend_from_slice(b"vorbis");
        id_pkt.extend_from_slice(&0u32.to_le_bytes());
        id_pkt.push(1);
        id_pkt.extend_from_slice(&44_100u32.to_le_bytes());
        id_pkt.extend_from_slice(&0i32.to_le_bytes());
        id_pkt.extend_from_slice(&64_000i32.to_le_bytes());
        id_pkt.extend_from_slice(&0i32.to_le_bytes());
        id_pkt.push(0xB8);
        id_pkt.push(1);

        let xray_c = OggXRayParams {
            version: OGG_COMMENT_VERSION,
            min_dist: 2.0,
            max_dist: 40.0,
            base_volume: 0.75,
            game_type: 9,
            max_ai_dist: 25.0,
        };
        let mut cmt_pkt = vec![0x03];
        cmt_pkt.extend_from_slice(b"vorbis");
        cmt_pkt.extend_from_slice(&4u32.to_le_bytes());
        cmt_pkt.extend_from_slice(b"XRay");
        cmt_pkt.extend_from_slice(&1u32.to_le_bytes());
        let c0 = xray_c.to_bytes();
        cmt_pkt.extend_from_slice(&(c0.len() as u32).to_le_bytes());
        cmt_pkt.extend_from_slice(&c0);
        cmt_pkt.push(1);

        let mut ogg = Vec::new();
        ogg.extend_from_slice(&make_ogg_page(0, 0x02, 0, &id_pkt));
        ogg.extend_from_slice(&make_ogg_page(1, 0x00, 0, &cmt_pkt));
        ogg.extend_from_slice(&make_ogg_page(2, 0x04, 44_100, b"\x02ok"));

        let entries = vec![
            ("anims\\camera_pack.anms", anms_raw.clone(), compress_lzo1x(&anms_raw)),
            (
                "meshes\\anomaly_weapons\\hands\\bone_parts.ltx",
                bone_parts_ltx.to_vec(),
                bone_parts_ltx.to_vec(),
            ),
            ("sounds\\weapons\\ak_shot.ogg", ogg.clone(), ogg),
        ];
        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("audio_cam.db0", pack_synthetic_db(entries))
            .unwrap();

        let anms = load_anms_from_vfs(&vfs, "camera_pack").unwrap();
        assert_eq!(anms.motions.len(), 2);
        assert!((anms.find("cam_hit").unwrap().evaluate(0.0).0[0] - 1.0).abs() < 1e-6);

        let parts = load_bone_parts_ltx_from_vfs(&vfs, "", true).unwrap();
        assert_eq!(parts.part_id("default"), Some(0));
        assert_eq!(parts.part_id("left_hand"), Some(1));

        let ogg_meta = load_ogg_sound_metadata_from_vfs(&vfs, "weapons\\ak_shot").unwrap();
        assert_eq!(ogg_meta.channels, 1);
        assert_eq!(ogg_meta.sample_rate, 44_100);
        assert!((ogg_meta.duration_seconds.unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(ogg_meta.xray_params, Some(xray_c));
    }

    #[test]
    fn test_stream_fonts_spawn_overlays_and_alife_subclasses_from_vfs() {
        use crate::ai_graph::{
            GameDeathPoint, GameEdge, GameGraphHeader, GameLevelInfo, GameVertex,
            XRAI_CURRENT_VERSION,
        };
        use crate::fonts::{FontAlignment, FontDrawParams};
        use crate::spawns::{
            alife_object_flags, m_spawn_flags, AlifeDynamicVisualState, AlifeInventoryItemState,
            AlifeItemWeaponState, AlifeLevelChangerState, AlifeObjectState,
            AlifeSpaceRestrictorState, ServerShapeDef, ServerVisualState, SpawnHeader, SpawnPacket,
            SpawnVertex, INVALID_OBJECT_ID, INVALID_SPAWN_STORY_ID, INVALID_STORY_ID, M_UPDATE,
            SPAWN_VERSION,
        };

        let fonts_ltx = br#"[string_table]
font_prefix = _west

[ui_font_letterica18_russian]
shader = font
texture = ui\ui_font_letter_18_1024
texture1600 = ui\ui_font_letter_18_1600
size = 18
interval = 0.9, 1.0
"#;
        let font_ini = br#"[symbol_coords]
height = 18
065 = 8, 16, 20
066 = 24, 16, 38
"#;

        let overlays_base = br#"[settings]
link_tolerance = 2.0
sync_level_offsets = true

[graph_links]
"#;
        let overlays_mod = br#"![graph_links@demo]
l01_escape:0.0,0.0,0.0 = l02_garbage:0.0,0.0,0.0

![spawn_patch@esc_ak74]
story_id = 5500
"#;

        let gg = GameGraph {
            header: GameGraphHeader {
                version: 10,
                vertex_count: 2,
                edge_count: 0,
                death_point_count: 1,
                guid: [0x10, 0x20],
                levels: vec![
                    GameLevelInfo {
                        name: "l01_escape".into(),
                        offset: [0.0; 3],
                        id: 1,
                        section: "level01".into(),
                        guid: [1, 1],
                    },
                    GameLevelInfo {
                        name: "l02_garbage".into(),
                        offset: [0.0, 0.0, 150.0],
                        id: 2,
                        section: "level02".into(),
                        guid: [2, 2],
                    },
                ],
            },
            vertices: vec![
                GameVertex {
                    local_point: [0.0; 3],
                    global_point: [0.0; 3],
                    level_id: 1,
                    level_vertex_id: 10,
                    vertex_types: [0; 4],
                    edge_offset: 84,
                    point_offset: 84,
                    neighbour_count: 0,
                    death_point_count: 1,
                },
                GameVertex {
                    local_point: [0.0; 3],
                    global_point: [0.0, 0.0, 150.0],
                    level_id: 2,
                    level_vertex_id: 20,
                    vertex_types: [0; 4],
                    edge_offset: 84,
                    point_offset: 104,
                    neighbour_count: 0,
                    death_point_count: 0,
                },
            ],
            edges: Vec::<GameEdge>::new(),
            death_points: vec![GameDeathPoint {
                point: [0.0; 3],
                level_vertex_id: 10,
                distance: 1.0,
            }],
            cross_tables: Vec::new(),
        };

        let base_alife = AlifeObjectState {
            game_vertex_id: 0,
            distance: 0.0,
            direct_control: true,
            level_vertex_id: 10,
            flags: alife_object_flags::CAN_SAVE,
            custom_data: String::new(),
            story_id: INVALID_STORY_ID,
            spawn_story_id: INVALID_SPAWN_STORY_ID,
            subclass_payload: Vec::new(),
        };
        let lc_state = AlifeLevelChangerState {
            restrictor: AlifeSpaceRestrictorState {
                alife: base_alife.clone(),
                shapes: vec![ServerShapeDef::Sphere {
                    center: [0.0; 3],
                    radius: 2.0,
                }],
                restrictor_type: 0,
                trailing: Vec::new(),
            },
            next_graph_vertex: 1,
            next_level_vertex: 20,
            next_position: [0.0; 3],
            next_angles: [0.0; 3],
            level_to_change: "l02_garbage".into(),
            level_point_to_change: "gar_wp".into(),
            silent_mode: false,
            trailing: Vec::new(),
        };
        let wpn_state = AlifeItemWeaponState {
            item: AlifeInventoryItemState {
                base: AlifeDynamicVisualState {
                    alife: base_alife,
                    visual: ServerVisualState {
                        visual_name: "dynamics\\weapons\\wpn_ak74\\wpn_ak74".into(),
                        flags: 0,
                    },
                    trailing: Vec::new(),
                },
                condition: 0.9,
                upgrades: vec!["up_sect_ak74".into()],
                trailing: Vec::new(),
            },
            ammo_current: 60,
            ammo_elapsed: 30,
            wpn_state: 0,
            addon_flags: 1,
            ammo_type: 0,
            ammo_elapsed_grenades: 0,
            trailing: Vec::new(),
        };

        let archive = AlifeSpawnArchive {
            header: SpawnHeader {
                version: XRAI_CURRENT_VERSION,
                guid: [0x55; 16],
                graph_guid: [0x10u64, 0x20]
                    .into_iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                spawn_count: 2,
                level_count: 2,
            },
            spawns: vec![
                SpawnVertex {
                    spawn_id: 0,
                    spawn: SpawnPacket {
                        section: "level_changer".into(),
                        name_replace: "lc_esc_gar".into(),
                        game_id_legacy: 0,
                        rp: 0xFE,
                        position: [0.0; 3],
                        angle: [0.0; 3],
                        respawn_time: 0,
                        id: INVALID_OBJECT_ID,
                        parent_id: INVALID_OBJECT_ID,
                        phantom_id: INVALID_OBJECT_ID,
                        flags: m_spawn_flags::VERSION,
                        version: SPAWN_VERSION,
                        game_type: u16::MAX,
                        script_version: 10,
                        client_data: Vec::new(),
                        spawn_id: 0,
                        state_payload: lc_state.to_bytes(),
                        trailing_update_payload: Vec::new(),
                    },
                    update_packet: M_UPDATE.to_le_bytes().to_vec(),
                },
                SpawnVertex {
                    spawn_id: 1,
                    spawn: SpawnPacket {
                        section: "wpn_ak74".into(),
                        name_replace: "esc_ak74".into(),
                        game_id_legacy: 0,
                        rp: 0xFE,
                        position: [3.0, 0.0, 1.0],
                        angle: [0.0; 3],
                        respawn_time: 0,
                        id: INVALID_OBJECT_ID,
                        parent_id: INVALID_OBJECT_ID,
                        phantom_id: INVALID_OBJECT_ID,
                        flags: m_spawn_flags::VERSION,
                        version: SPAWN_VERSION,
                        game_type: u16::MAX,
                        script_version: 10,
                        client_data: Vec::new(),
                        spawn_id: 1,
                        state_payload: wpn_state.to_bytes(),
                        trailing_update_payload: Vec::new(),
                    },
                    update_packet: M_UPDATE.to_le_bytes().to_vec(),
                },
            ],
            edges: Vec::new(),
            artefact_spawn_positions: Vec::new(),
            patrol_paths: Vec::new(),
            game_graph: gg,
        }
        .to_bytes();

        let entries = vec![
            ("configs\\fonts.ltx", fonts_ltx.to_vec(), fonts_ltx.to_vec()),
            (
                "textures\\ui\\ui_font_letter_18_1600_west.ini",
                font_ini.to_vec(),
                font_ini.to_vec(),
            ),
            (
                "configs\\spawn_overlays.ltx",
                overlays_base.to_vec(),
                overlays_base.to_vec(),
            ),
            (
                "configs\\mod_spawn_overlays_demo.ltx",
                overlays_mod.to_vec(),
                overlays_mod.to_vec(),
            ),
            ("spawns\\all.spawn", archive.clone(), compress_lzo1x(&archive)),
        ];
        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("ui_spawn_pack.db0", pack_synthetic_db(entries))
            .unwrap();

        let fonts_doc = load_ltx_tree_from_vfs(&vfs, "$game_configs$/fonts.ltx", &[]).unwrap();
        let (fcfg, atlas) =
            load_font_from_vfs(&vfs, &fonts_doc, "ui_font_letterica18_russian", 1440).unwrap();
        assert_eq!(fcfg.resolved_texture, "ui\\ui_font_letter_18_1600_west");
        let quads = atlas.build_string_quads(
            [512, 256],
            [10.0, 20.0],
            "AB",
            &FontDrawParams {
                current_height: 18.0,
                align: FontAlignment::Left,
                ..FontDrawParams::default()
            },
        );
        assert_eq!(quads.len(), 2);

        let overlays = load_spawn_overlays_from_vfs(&vfs, None).unwrap();
        assert_eq!(overlays.graph_links.len(), 1);
        let mut loaded_spawn = load_all_spawn_from_vfs(&vfs, "all").unwrap();
        assert_eq!(
            loaded_spawn.spawns[0].spawn.decode_level_changer_state().unwrap(),
            lc_state
        );
        assert_eq!(
            loaded_spawn.spawns[1].spawn.decode_weapon_state().unwrap(),
            wpn_state
        );
        let report = overlays.apply_to_spawn_archive(&mut loaded_spawn, false, &[]);
        assert_eq!(report.graph.edges_added, 2);
        assert_eq!(report.spawns_patched, 1);
        assert_eq!(
            loaded_spawn.spawns[1]
                .spawn
                .decode_alife_state()
                .unwrap()
                .story_id,
            5500
        );
    }

    #[test]
    fn test_stream_world_maps_viewport_ui_xml_and_stalker_alife_from_vfs() {
        use crate::ai_graph::{
            GameGraphHeader, GameLevelInfo, GameVertex, XRAI_CURRENT_VERSION,
        };
        use crate::fonts::{FontAlignment, FontDrawParams};
        use crate::spawns::{
            alife_object_flags, m_spawn_flags, ph_skeleton_flags, AlifeAnomalousZoneState,
            AlifeCreatureAbstractState, AlifeCustomZoneState, AlifeDynamicVisualState,
            AlifeHumanAbstractState, AlifeHumanStalkerState, AlifeMonsterAbstractState,
            AlifeObjectState, AlifeSpaceRestrictorState, AlifeTraderAbstractState,
            ServerPhBoneState, ServerPhBonesData, ServerPhSkeletonState, ServerShapeDef,
            ServerVisualState, SpawnHeader, SpawnPacket, SpawnVertex, INVALID_OBJECT_ID,
            INVALID_SPAWN_STORY_ID, INVALID_STORY_ID, M_UPDATE, SPAWN_VERSION,
        };
        use crate::xml_ui::{color_argb, UiWindowDef};

        let game_levels_ltx = br#"[levels]
level01 =

[level01]
name = l01_escape
caption = "Cordon"
offset = 0.0, 0.0, 0.0
id = 1
"#;
        let game_maps_ltx = br#"[global_map]
texture = ui\ui_global_map
bound_rect = 0.0, 0.0, 1024.0, 2634.0
max_zoom = 6.0

[level_maps_single]
l01_escape =

[l01_escape]
texture = map\map_escape
bound_rect = -100.0, -100.0, 100.0, 100.0
global_rect = 300.0, 2000.0, 500.0, 2200.0
label = "st_cordon"
label_color = 255, 220, 150
"#;
        let level_ltx = br#"[level_map]
texture = map\map_escape_hd
bound_rect = -200.0, -200.0, 200.0, 200.0
"#;
        let color_defs_xml = br#"<colors><color name="hud_amber" r="255" g="180" b="50" a="255"/></colors>"#;
        let pda_ui_16_xml = br#"<w>
    <pda_caption x="50" y="100" width="200" height="40">
        <text font="letterica18" align="l" vert_align="c" color="hud_amber">Esc %c[default]Ok</text>
    </pda_caption>
</w>"#;
        let fonts_ltx = br#"[ui_font_letterica18_russian]
shader = font
texture = ui\ui_font_letter_18_1024
size = 18
"#;
        let font_ini = br#"[symbol_coords]
height = 18
032 = 0, 0, 6
069 = 8, 0, 20
079 = 24, 0, 36
099 = 40, 0, 50
107 = 52, 0, 62
115 = 64, 0, 74
"#;

        let base_alife = AlifeObjectState {
            game_vertex_id: 0,
            distance: 0.0,
            direct_control: true,
            level_vertex_id: 1,
            flags: alife_object_flags::CAN_SAVE,
            custom_data: String::new(),
            story_id: INVALID_STORY_ID,
            spawn_story_id: INVALID_SPAWN_STORY_ID,
            subclass_payload: Vec::new(),
        };
        let stalker = AlifeHumanStalkerState {
            human: AlifeHumanAbstractState {
                trader: AlifeTraderAbstractState {
                    money: 1200,
                    specific_character: "esc_stalker".into(),
                    trader_flags: 0,
                    character_profile: "esc_stalker".into(),
                    community_index: 1,
                    rank: 300,
                    reputation: 0,
                    character_name: "Petruha".into(),
                    deadbody_can_take: true,
                    deadbody_closed: false,
                },
                monster: AlifeMonsterAbstractState {
                    creature: AlifeCreatureAbstractState {
                        base: AlifeDynamicVisualState {
                            alife: base_alife.clone(),
                            visual: ServerVisualState {
                                visual_name: "actors\\stalker_neutral\\stalker_neutral_1".into(),
                                flags: 0,
                            },
                            trailing: Vec::new(),
                        },
                        team: 0,
                        squad: 0,
                        group: 0,
                        health: 1.0,
                        dynamic_out_restrictions: Vec::new(),
                        dynamic_in_restrictions: Vec::new(),
                        killer_id: INVALID_OBJECT_ID,
                        game_death_time: 0,
                        trailing: Vec::new(),
                    },
                    out_space_restrictors: String::new(),
                    in_space_restrictors: String::new(),
                    smart_terrain_id: 3,
                    task_reached: false,
                    trailing: Vec::new(),
                },
                equipment_prefs: vec![1, 0, 1, 2, 0],
                main_weapon_prefs: vec![2, 2, 1, 0],
                trailing: Vec::new(),
            },
            skeleton: ServerPhSkeletonState {
                startup_animation: "$editor".into(),
                flags: ph_skeleton_flags::SAVED_DATA,
                source_id: INVALID_OBJECT_ID,
                saved_bones: Some(ServerPhBonesData {
                    bones_mask: 1,
                    root_bone: 0,
                    min: [-5.0; 3],
                    max: [5.0; 3],
                    bones: vec![ServerPhBoneState::from_world_position_quat(
                        [0.0; 3],
                        [0.0, 0.0, 0.0, 1.0],
                        [-5.0; 3],
                        [5.0; 3],
                        true,
                    )],
                }),
            },
            trailing: Vec::new(),
        };
        let zone = AlifeAnomalousZoneState {
            zone: AlifeCustomZoneState {
                restrictor: AlifeSpaceRestrictorState {
                    alife: base_alife,
                    shapes: vec![ServerShapeDef::Sphere {
                        center: [0.0; 3],
                        radius: 3.5,
                    }],
                    restrictor_type: 1,
                    trailing: Vec::new(),
                },
                max_power: 2.0,
                owner_id: u32::MAX,
                enabled_time: 0,
                disabled_time: 0,
                start_time_shift: 0,
                trailing: Vec::new(),
            },
            offline_interactive_radius: 30.0,
            artefact_spawn_count: 8,
            artefact_position_offset: 0,
            trailing: Vec::new(),
        };

        let gg = GameGraph {
            header: GameGraphHeader {
                version: 10,
                vertex_count: 1,
                edge_count: 0,
                death_point_count: 0,
                guid: [0x11, 0x22],
                levels: vec![GameLevelInfo {
                    name: "l01_escape".into(),
                    offset: [0.0; 3],
                    id: 1,
                    section: "level01".into(),
                    guid: [1, 1],
                }],
            },
            vertices: vec![GameVertex {
                local_point: [0.0; 3],
                global_point: [0.0; 3],
                level_id: 1,
                level_vertex_id: 1,
                vertex_types: [0; 4],
                edge_offset: 48,
                point_offset: 48,
                neighbour_count: 0,
                death_point_count: 0,
            }],
            edges: Vec::new(),
            death_points: Vec::new(),
            cross_tables: Vec::new(),
        };
        let archive = AlifeSpawnArchive {
            header: SpawnHeader {
                version: XRAI_CURRENT_VERSION,
                guid: [0x22; 16],
                graph_guid: [0x11u64, 0x22]
                    .into_iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                spawn_count: 2,
                level_count: 1,
            },
            spawns: vec![
                SpawnVertex {
                    spawn_id: 0,
                    spawn: SpawnPacket {
                        section: "stalker".into(),
                        name_replace: "esc_petruha".into(),
                        game_id_legacy: 0,
                        rp: 0xFE,
                        position: [0.0; 3],
                        angle: [0.0; 3],
                        respawn_time: 0,
                        id: INVALID_OBJECT_ID,
                        parent_id: INVALID_OBJECT_ID,
                        phantom_id: INVALID_OBJECT_ID,
                        flags: m_spawn_flags::VERSION,
                        version: SPAWN_VERSION,
                        game_type: u16::MAX,
                        script_version: 10,
                        client_data: Vec::new(),
                        spawn_id: 0,
                        state_payload: stalker.to_bytes(),
                        trailing_update_payload: Vec::new(),
                    },
                    update_packet: M_UPDATE.to_le_bytes().to_vec(),
                },
                SpawnVertex {
                    spawn_id: 1,
                    spawn: SpawnPacket {
                        section: "zone_mine_gravitational_weak".into(),
                        name_replace: "esc_anom_01".into(),
                        game_id_legacy: 0,
                        rp: 0xFE,
                        position: [5.0, 0.0, 5.0],
                        angle: [0.0; 3],
                        respawn_time: 0,
                        id: INVALID_OBJECT_ID,
                        parent_id: INVALID_OBJECT_ID,
                        phantom_id: INVALID_OBJECT_ID,
                        flags: m_spawn_flags::VERSION,
                        version: SPAWN_VERSION,
                        game_type: u16::MAX,
                        script_version: 10,
                        client_data: Vec::new(),
                        spawn_id: 1,
                        state_payload: zone.to_bytes(),
                        trailing_update_payload: Vec::new(),
                    },
                    update_packet: M_UPDATE.to_le_bytes().to_vec(),
                },
            ],
            edges: Vec::new(),
            artefact_spawn_positions: Vec::new(),
            patrol_paths: Vec::new(),
            game_graph: gg,
        }
        .to_bytes();

        let entries = vec![
            ("configs\\game_levels.ltx", game_levels_ltx.to_vec(), game_levels_ltx.to_vec()),
            ("configs\\game_maps_single.ltx", game_maps_ltx.to_vec(), game_maps_ltx.to_vec()),
            ("levels\\l01_escape\\level.ltx", level_ltx.to_vec(), level_ltx.to_vec()),
            ("configs\\ui\\color_defs.xml", color_defs_xml.to_vec(), color_defs_xml.to_vec()),
            ("configs\\ui\\pda_16.xml", pda_ui_16_xml.to_vec(), pda_ui_16_xml.to_vec()),
            ("configs\\fonts.ltx", fonts_ltx.to_vec(), fonts_ltx.to_vec()),
            (
                "textures\\ui\\ui_font_letter_18_1024.ini",
                font_ini.to_vec(),
                font_ini.to_vec(),
            ),
            ("spawns\\all.spawn", archive.clone(), compress_lzo1x(&archive)),
        ];
        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("world_ui_alife.db0", pack_synthetic_db(entries))
            .unwrap();

        let levels = load_game_levels_from_vfs(&vfs, &[]).unwrap();
        assert_eq!(levels.find_by_name("l01_escape").unwrap().caption, "Cordon");

        let maps = load_world_maps_from_vfs(&vfs, &[], true).unwrap();
        let esc = maps.get_level("l01_escape").unwrap();
        assert_eq!(esc.texture, "map\\map_escape_hd");
        assert_eq!(esc.bound_rect, [-200.0, -200.0, 200.0, 200.0]);
        assert_eq!(esc.label_color_argb, color_argb(255, 255, 220, 150));
        let uv = maps.world_xz_to_level_uv("l01_escape", [0.0, 0.0]).unwrap();
        assert!((uv[0] - 0.5).abs() < 1e-6);
        assert!((uv[1] - 0.5).abs() < 1e-6);

        let vp = UiViewport::new(1920.0, 1080.0);
        let ui_doc = load_ui_xml_for_viewport_from_vfs(&vfs, "pda.xml", &vp).unwrap();
        let colors = load_color_defs_from_vfs(&vfs).unwrap();
        let wnd = UiWindowDef::from_xml_node(ui_doc.navigate("pda_caption", 0).unwrap(), &colors);
        let fonts_doc = load_ltx_tree_from_vfs(&vfs, "$game_configs$/fonts.ltx", &[]).unwrap();
        let (_fcfg, atlas) =
            load_font_from_vfs(&vfs, &fonts_doc, "ui_font_letterica18_russian", 768).unwrap();
        let quads = atlas.build_ui_window_text_quads(
            [512, 256],
            &wnd,
            &vp,
            Some(&colors),
            Some("E %c[255,0,255,0]s %c[default]Ok"),
            FontDrawParams::default(),
        );
        assert_eq!(quads.len(), 6); // 'E', ' ', 's', ' ', 'O', 'k'
        assert_eq!(quads[0].color_top_argb, color_argb(255, 255, 180, 50));
        assert_eq!(quads[2].color_top_argb, color_argb(255, 0, 255, 0));
        assert_eq!(quads[5].color_top_argb, color_argb(255, 255, 180, 50));
        assert_eq!(
            FontAlignment::parse_xml_align(&wnd.text.as_ref().unwrap().align),
            FontAlignment::Left
        );

        let loaded_spawn = load_all_spawn_from_vfs(&vfs, "all").unwrap();
        assert_eq!(
            loaded_spawn.spawns[0]
                .spawn
                .decode_human_stalker_state()
                .unwrap(),
            stalker
        );
        assert_eq!(
            loaded_spawn.spawns[1]
                .spawn
                .decode_anomalous_zone_state()
                .unwrap(),
            zone
        );
    }

    #[test]
    fn test_stream_lighting_glows_build_lights_torch_and_hanging_lamp_from_vfs() {
        use crate::lanims::{pack_argb, LightAnimItem, LightAnimLibrary, LANIM_VERSION};
        use crate::level::{build_light_type, level_chunks, BUILD_LIGHT_RECORD_SIZE};
        use crate::spawns::{
            hanging_lamp_flags, AlifeDynamicVisualState, AlifeObjectHangingLampState,
            AlifeObjectState, ServerPhSkeletonState, ServerVisualState,
        };

        // 1. lanims.xr
        let mut anim = LightAnimItem::new("lights\\candle");
        anim.fps = 10.0;
        anim.resize(2);
        anim.insert_key(0, pack_argb(255, 255, 128, 64)).unwrap();
        let lanims_raw = LightAnimLibrary {
            version: LANIM_VERSION,
            items: vec![anim],
        }
        .to_bytes();

        // 2. build.lights (1 hemi point light)
        let mut bl_rec = Vec::new();
        bl_rec.extend_from_slice(&build_light_type::POINT.to_le_bytes());
        bl_rec.extend_from_slice(&0u16.to_le_bytes());
        for f in [
            0.5f32, 0.5, 0.4, 2.0, 3.0, 4.0, 0.0, -1.0, 0.0, 10.0, 100.0, 0.0, 1.0, 0.1, 0.01,
            2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        ] {
            bl_rec.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(bl_rec.len(), BUILD_LIGHT_RECORD_SIZE);
        let mut bl_w = ChunkWriter::new();
        bl_w.write_chunk(1, &bl_rec);
        let build_lights_raw = bl_w.into_bytes();

        // 3. level (shader + 1 glow)
        let mut shd = Vec::new();
        shd.extend_from_slice(&2u32.to_le_bytes());
        shd.extend_from_slice(b"\0effects\\glow/glow\\glow_white\0");
        let mut glow_raw = Vec::new();
        for f in [4.0f32, 2.5, 1.0, 0.5] {
            glow_raw.extend_from_slice(&f.to_le_bytes());
        }
        glow_raw.extend_from_slice(&1u16.to_le_bytes());
        let mut lvl_w = ChunkWriter::new();
        lvl_w.write_chunk(level_chunks::FSL_SHADERS, &shd);
        lvl_w.write_chunk(level_chunks::FSL_GLOWS, &glow_raw);
        let level_raw = lvl_w.into_bytes();

        // 4. light_night.ltx [torch_definition]
        let torch_ltx = b"[torch_definition]\ntype = 2\nomni_type = 1\ncolor_r2 = 0.9, 0.85, 0.7, 1.0\nrange_r2 = 28\nspot_angle = 55\ncolor_animator = lights\\candle\n";

        let entries = vec![
            ("lanims.xr", lanims_raw.clone(), lanims_raw),
            ("levels\\l01_escape\\build.lights", build_lights_raw.clone(), build_lights_raw),
            ("levels\\l01_escape\\level", level_raw.clone(), level_raw),
            ("configs\\models\\objects\\light_night.ltx", torch_ltx.to_vec(), torch_ltx.to_vec()),
        ];
        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("lights.db0", pack_synthetic_db(entries))
            .unwrap();

        let lanims = load_lanims_from_vfs(&vfs).unwrap();
        let build_lights = load_build_lights_from_vfs(&vfs, "$game_levels$/l01_escape").unwrap();
        let level = load_level_from_vfs(&vfs, "$game_levels$/l01_escape").unwrap();
        let torch = load_torch_light_config_from_vfs(
            &vfs,
            "$game_configs$/models/objects/light_night.ltx",
            &[],
            "torch_definition",
            true,
        )
        .unwrap();

        assert_eq!(build_lights.hemi.len(), 1);
        assert_eq!(level.glows.len(), 1);
        assert_eq!(level.glow_shader(&level.glows[0]).unwrap().shader, "effects\\glow");
        assert_eq!(torch.range, 28.0);

        let lamp = AlifeObjectHangingLampState {
            base: AlifeDynamicVisualState {
                alife: AlifeObjectState {
                    game_vertex_id: 0,
                    distance: 0.0,
                    direct_control: true,
                    level_vertex_id: 1,
                    flags: 0,
                    custom_data: String::new(),
                    story_id: u32::MAX,
                    spawn_story_id: u32::MAX,
                    subclass_payload: Vec::new(),
                },
                visual: ServerVisualState::default(),
                trailing: Vec::new(),
            },
            skeleton: ServerPhSkeletonState::default(),
            brightness: 1.5,
            color_animator: "lights\\candle".to_string(),
            range: 12.0,
            flags: hanging_lamp_flags::TYPE_SPOT | hanging_lamp_flags::R2,
            ..Default::default()
        };
        let ll = lamp.to_level_light([0.0, 2.0, 0.0], [0.0; 3], Some(&lanims), 0.0);
        assert!((ll.diffuse[0] - 1.5).abs() < 1e-4);
        assert!((ll.evaluate_spot_factor([0.0, 2.0, 5.0]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_phase10_sky_cubemaps_env_mods_use_flags_and_level_ambients_over_vfs() {
        use crate::shaders_xr::env_mod_flags;
        use crate::thm::{
            dds_caps2, thm_chunks, CubemapFace, DdsPixelFormatKind, TextureType, DDS_MAGIC,
        };
        use crate::weather::EnvBoost;

        fn build_dxt1_cubemap(dim: u32, fill_seed: u8) -> Vec<u8> {
            let mut dds = vec![0u8; 128];
            dds[0..4].copy_from_slice(&DDS_MAGIC.to_le_bytes());
            dds[4..8].copy_from_slice(&124u32.to_le_bytes());
            dds[8..12].copy_from_slice(&0x1007u32.to_le_bytes());
            dds[12..16].copy_from_slice(&dim.to_le_bytes());
            dds[16..20].copy_from_slice(&dim.to_le_bytes());
            dds[28..32].copy_from_slice(&1u32.to_le_bytes());
            dds[76..80].copy_from_slice(&32u32.to_le_bytes());
            dds[80..84].copy_from_slice(&0x4u32.to_le_bytes()); // DDPF_FOURCC
            dds[84..88].copy_from_slice(b"DXT1");
            dds[108..112].copy_from_slice(&0x0000_1008u32.to_le_bytes());
            dds[112..116]
                .copy_from_slice(&(dds_caps2::CUBEMAP | dds_caps2::CUBEMAP_ALLFACES).to_le_bytes());
            let face_bytes = (dim.div_ceil(4) * dim.div_ceil(4) * 8) as usize;
            for face in 0..6u8 {
                dds.resize(dds.len() + face_bytes, fill_seed.wrapping_add(face));
            }
            dds
        }

        let sky_dds = build_dxt1_cubemap(8, 0x10);
        let small_dds = build_dxt1_cubemap(4, 0x80);

        let mut thm_w = ChunkWriter::new();
        let mut tp = vec![0u8; 32];
        tp[24..28].copy_from_slice(&8u32.to_le_bytes());
        tp[28..32].copy_from_slice(&8u32.to_le_bytes());
        thm_w.write_chunk(thm_chunks::THM_CHUNK_TEXTUREPARAM, &tp);
        thm_w.write_chunk(
            thm_chunks::THM_CHUNK_TEXTURE_TYPE,
            &(TextureType::CubeMap as u32).to_le_bytes(),
        );
        let sky_thm = thm_w.into_bytes();

        let weather_ltx = br#"
[00:00:00]
sky_texture = sky\sky_night
clouds_texture = sky\sky_oblaka
sky_color = 0.2, 0.3, 0.4
sky_rotation = 90
far_plane = 400
fog_color = 0.2, 0.2, 0.2
fog_density = 0.25
fog_distance = 200
ambient_color = 0.1, 0.1, 0.1
hemisphere_color = 0.4, 0.4, 0.4, 1.0
sun_color = 0.0, 0.0, 0.0
ambient = env_night
[12:00:00]
sky_texture = sky\sky_night
clouds_texture = sky\sky_oblaka
sky_color = 0.2, 0.3, 0.4
sky_rotation = 90
far_plane = 400
fog_color = 0.2, 0.2, 0.2
fog_density = 0.25
fog_distance = 200
ambient_color = 0.1, 0.1, 0.1
hemisphere_color = 0.4, 0.4, 0.4, 1.0
sun_color = 0.0, 0.0, 0.0
ambient = env_night
"#;
        let base_amb = b"[env_night]\nsound_channels = wind_quiet\n";
        let lvl_amb = b"[env_night]\nsound_channels = bunker_hum\n";
        let env_mods_ltx = br#"
[bunker_fog]
position = 0, 0, 0
radius = 50
power = 1.0
far_plane = 100
fog_color = 0.8, 0.6, 0.4
fog_density = 0.9
ambient = 0.5, 0.5, 0.5
use_flags = 0x02
"#;

        let entries = vec![
            ("textures\\sky\\sky_night.dds", sky_dds.clone(), sky_dds),
            (
                "textures\\sky\\sky_night#small.dds",
                small_dds.clone(),
                small_dds,
            ),
            ("textures\\sky\\sky_night.thm", sky_thm.clone(), sky_thm),
            (
                "configs\\environment\\weathers\\w_clear.ltx",
                weather_ltx.to_vec(),
                weather_ltx.to_vec(),
            ),
            (
                "configs\\environment\\ambients.ltx",
                base_amb.to_vec(),
                base_amb.to_vec(),
            ),
            (
                "configs\\environment\\ambients\\l01_escape.ltx",
                lvl_amb.to_vec(),
                lvl_amb.to_vec(),
            ),
            (
                "configs\\environment\\env_mods\\l01_escape.ltx",
                env_mods_ltx.to_vec(),
                env_mods_ltx.to_vec(),
            ),
        ];

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("sky.db0", pack_synthetic_db(entries))
            .unwrap();

        let lib = load_weather_library_from_vfs(&vfs, Some("l01_escape")).unwrap();
        assert_eq!(lib.ambients["env_night"].sound_channels, vec!["bunker_hum"]);

        let mods = load_env_mods_from_vfs(&vfs, "$game_levels$/l01_escape", Some("l01_escape"))
            .unwrap();
        assert_eq!(mods.modifiers[0].use_flags, env_mod_flags::FOG_COLOR);

        let mixed = lib
            .sample_at(
                "w_clear",
                6.0 * 3600.0,
                Some(&mods),
                [0.0, 0.0, 0.0],
                1.0,
                &EnvBoost::default(),
            )
            .unwrap();
        // use_flags = 0x02 must modify fog_color ONLY and leave fog_density & far_plane untouched!
        assert!((mixed.fog_color[0] - 0.5).abs() < 1e-5);
        assert!((mixed.fog_density - 0.25).abs() < 1e-5);
        assert!((mixed.far_plane - 400.0).abs() < 1e-4);

        let thm = load_thm_from_vfs(&vfs, &mixed.sky_texture_a).unwrap().unwrap();
        assert!(thm.is_cubemap());

        let (sky_a_vpath, _) = mixed.sky_dds_vpaths();
        let (env_a_vpath, _) = mixed.sky_env_dds_vpaths();
        let (sky_raw, sky_cube) = load_cubemap_dds_from_vfs(&vfs, sky_a_vpath.as_deref().unwrap())
            .unwrap();
        let (env_raw, env_cube) = load_cubemap_dds_from_vfs(&vfs, env_a_vpath.as_deref().unwrap())
            .unwrap();

        assert_eq!(sky_cube.format, DdsPixelFormatKind::Bc1Unorm);
        assert_eq!(
            sky_cube
                .subresource(CubemapFace::PositiveZ, 0)
                .unwrap()
                .slice_bytes(&sky_raw)
                .unwrap(),
            &[0x14; 32]
        );
        assert_eq!(
            env_cube
                .subresource(CubemapFace::NegativeY, 0)
                .unwrap()
                .slice_bytes(&env_raw)
                .unwrap(),
            &[0x83; 8]
        );
    }
}
