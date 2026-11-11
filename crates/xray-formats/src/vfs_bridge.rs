//! Bridges between `xray-vfs` (`VirtualFileSystem`), `xray-core` (`.ltx` / DLTX),
//! and `xray-formats` (`OgfModel` / `ChunkReader`).

use xray_core::ltx_loader::{collect_ltx_sources, glob_matches_ascii, LtxFileProvider, OwnedLtxDocument};
use xray_vfs::{normalize_virtual_path, normalize_virtual_prefix, VfsError, VirtualFileSystem};

use crate::chunk::FormatError;
use crate::ogf::OgfModel;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use crate::ogf::ogf_chunks::*;
    use crate::ogf::{fvf_flags, OgfVisualType};
    use xray_vfs::{compress_lzo1x, compute_crc32};

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

        let entries: Vec<(&str, &[u8], Vec<u8>)> = vec![
            ("meshes\\weapons\\ak.ogf", &ogf_raw, ogf_comp),
            ("configs\\weapons.ltx", root_ltx, root_ltx.to_vec()),
            ("configs\\weapons_ak.ltx", sub_ltx, sub_ltx.to_vec()),
            ("configs\\mod_weapons_balance.ltx", mod_ltx, mod_ltx.to_vec()),
        ];

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

        let mut vfs = VirtualFileSystem::new();
        vfs.mount_archive_bytes("test.db0", db).unwrap();

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
}
