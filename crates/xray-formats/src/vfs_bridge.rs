//! Bridges between `xray-vfs` (`VirtualFileSystem`), `xray-core` (`.ltx` / DLTX),
//! and `xray-formats` (`OgfModel`, `OmfMotions`, `TextureParams`, `GameMaterialLibrary`,
//! `CformModel`, `LevelDetails`, `LevelData`, `LevelGeometry`).

use xray_core::ltx_loader::{collect_ltx_sources, glob_matches_ascii, LtxFileProvider, OwnedLtxDocument};
use xray_vfs::{normalize_virtual_path, normalize_virtual_prefix, VfsError, VirtualFileSystem};

use crate::cform::CformModel;
use crate::chunk::FormatError;
use crate::details::{DetailMesh, LevelDetails};
use crate::gamemtl::GameMaterialLibrary;
use crate::level::{LevelData, LevelGeometry};
use crate::ogf::{OgfModel, OgfTextureLink};
use crate::omf::OmfMotions;
use crate::thm::{
    strip_texture_extension, texture_stem_to_dds_vpath, texture_stem_to_thm_vpath,
    ResolvedTextureLinks, TextureParams,
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
}
