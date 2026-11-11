//! Pure Rust X-Ray `.thm` Texture Metadata & `_bump` / `_bump#` Link Parser
//!
//! Faithful pure-Rust implementation matching `src/Layers/xrRender/ETextureParams.h`,
//! `src/Layers/xrRender/ETextureParams.cpp`, and `src/Layers/xrRender/TextureDescrManager.cpp`
//! (including companion `[texture]` `.ltx` fallback metadata).

use xray_core::LtxDocument;

use crate::chunk::{ByteCursor, ChunkReader, FormatError};

/// Chunk IDs inside a `.thm` texture metadata file (`ETextureParams.h:199-208`).
pub mod thm_chunks {
    pub const THM_CHUNK_VERSION: u32 = 0x0810;
    pub const THM_CHUNK_DATA: u32 = 0x0811;
    pub const THM_CHUNK_TEXTUREPARAM: u32 = 0x0812;
    pub const THM_CHUNK_TYPE: u32 = 0x0813;
    pub const THM_CHUNK_TEXTURE_TYPE: u32 = 0x0814;
    pub const THM_CHUNK_DETAIL_EXT: u32 = 0x0815;
    pub const THM_CHUNK_MATERIAL: u32 = 0x0816;
    pub const THM_CHUNK_BUMP: u32 = 0x0817;
    pub const THM_CHUNK_EXT_NORMALMAP: u32 = 0x0818;
    pub const THM_CHUNK_FADE_DELAY: u32 = 0x0819;
}

/// Texture flags bitmask (`STextureParams` enum flags in `ETextureParams.h:77-95`).
pub mod texture_flags {
    pub const FL_GENERATE_MIP_MAPS: u32 = 1 << 0;
    pub const FL_BINARY_ALPHA: u32 = 1 << 1;
    pub const FL_ALPHA_BORDER: u32 = 1 << 4;
    pub const FL_COLOR_BORDER: u32 = 1 << 5;
    pub const FL_FADE_TO_COLOR: u32 = 1 << 6;
    pub const FL_FADE_TO_ALPHA: u32 = 1 << 7;
    pub const FL_DITHER_COLOR: u32 = 1 << 8;
    pub const FL_DITHER_EACH_MIP_LEVEL: u32 = 1 << 9;
    pub const FL_DIFFUSE_DETAIL: u32 = 1 << 23;
    pub const FL_IMPLICIT_LIGHTED: u32 = 1 << 24;
    pub const FL_HAS_ALPHA: u32 = 1 << 25;
    pub const FL_BUMP_DETAIL: u32 = 1 << 26;
}

/// Texture asset class (`STextureParams::ETType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TextureType {
    #[default]
    Image = 0,
    CubeMap = 1,
    BumpMap = 2,
    NormalMap = 3,
    Terrain = 4,
}

impl TryFrom<u32> for TextureType {
    type Error = FormatError;

    fn try_from(val: u32) -> Result<Self, Self::Error> {
        match val {
            0 => Ok(Self::Image),
            1 => Ok(Self::CubeMap),
            2 => Ok(Self::BumpMap),
            3 => Ok(Self::NormalMap),
            4 => Ok(Self::Terrain),
            _ => Err(FormatError::InvalidValue("unknown THM ETType value")),
        }
    }
}

/// Texture encoding format tag (`STextureParams::ETFormat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TextureFormat {
    #[default]
    Dxt1 = 0,
    Adxt1 = 1,
    Dxt3 = 2,
    Dxt5 = 3,
    Rgba4444 = 4,
    Argb1555 = 5,
    Rgb565 = 6,
    Rgb = 7,
    Rgba = 8,
    Nvhs = 9,
    Nvhu = 10,
    A8 = 11,
    L8 = 12,
    A8L8 = 13,
}

impl TryFrom<u32> for TextureFormat {
    type Error = FormatError;

    fn try_from(val: u32) -> Result<Self, Self::Error> {
        match val {
            0 => Ok(Self::Dxt1),
            1 => Ok(Self::Adxt1),
            2 => Ok(Self::Dxt3),
            3 => Ok(Self::Dxt5),
            4 => Ok(Self::Rgba4444),
            5 => Ok(Self::Argb1555),
            6 => Ok(Self::Rgb565),
            7 => Ok(Self::Rgb),
            8 => Ok(Self::Rgba),
            9 => Ok(Self::Nvhs),
            10 => Ok(Self::Nvhu),
            11 => Ok(Self::A8),
            12 => Ok(Self::L8),
            13 => Ok(Self::A8L8),
            _ => Err(FormatError::InvalidValue("unknown THM ETFormat value")),
        }
    }
}

impl TextureFormat {
    #[inline]
    pub fn has_alpha_channel(self) -> bool {
        matches!(
            self,
            Self::Adxt1
                | Self::Dxt3
                | Self::Dxt5
                | Self::Rgba4444
                | Self::Argb1555
                | Self::Rgba
        )
    }
}

/// Bump mapping mode (`STextureParams::ETBumpMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TextureBumpMode {
    Reserved = 0,
    #[default]
    None = 1,
    Use = 2,
    UseParallax = 3,
}

impl TryFrom<u32> for TextureBumpMode {
    type Error = FormatError;

    fn try_from(val: u32) -> Result<Self, Self::Error> {
        match val {
            // `ETextureParams.cpp:100`: `if (bump_mode < tbmNone) bump_mode = tbmNone;`
            0 | 1 => Ok(Self::None),
            2 => Ok(Self::Use),
            3 => Ok(Self::UseParallax),
            _ => Err(FormatError::InvalidValue("unknown THM ETBumpMode value")),
        }
    }
}

/// Surface shading BRDF pair (`STextureParams::ETMaterial`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TextureMaterial {
    OrenNayarBlin = 0,
    #[default]
    BlinPhong = 1,
    PhongMetal = 2,
    MetalOrenNayar = 3,
}

/// Resolved VFS paths for companion bump & normal-error (`_bump` / `_bump#`) textures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BumpTextureLinks {
    /// Raw stem (e.g. `wpn\ak74_bump`).
    pub bump_name: String,
    /// Primary bump texture path (`textures\wpn\ak74_bump.dds`).
    pub bump_dds: String,
    /// Companion high-bit normal/parallax error texture (`textures\wpn\ak74_bump#.dds`).
    pub bump_hash_dds: String,
    pub steep_parallax: bool,
}

/// Complete resolved VFS texture asset set for an `OgfTextureLink` or surface texture name.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedTextureLinks {
    pub diffuse_name: String,
    pub diffuse_dds: String,
    pub bump: Option<BumpTextureLinks>,
    pub detail_dds: Option<String>,
    pub detail_scale: f32,
    pub use_diffuse_detail: bool,
    pub use_bump_detail: bool,
    pub ext_normal_map_dds: Option<String>,
    pub material_id: f32,
}

/// Parsed `.thm` texture metadata (`STextureParams` in `ETextureParams.h`).
#[derive(Debug, Clone, PartialEq)]
pub struct TextureParams {
    pub thm_version: Option<u16>,
    pub object_type: Option<u32>,
    pub has_thumbnail: bool,
    pub fmt: TextureFormat,
    pub flags: u32,
    pub border_color: u32,
    pub fade_color: u32,
    pub fade_amount: u32,
    pub fade_delay: u8,
    pub mip_filter: u32,
    pub width: u32,
    pub height: u32,
    pub detail_name: String,
    pub detail_scale: f32,
    pub texture_type: TextureType,
    pub material: u32,
    pub material_weight: f32,
    pub bump_virtual_height: f32,
    pub bump_mode: TextureBumpMode,
    pub bump_name: String,
    pub ext_normal_map_name: String,
}

impl Default for TextureParams {
    fn default() -> Self {
        Self {
            thm_version: None,
            object_type: None,
            has_thumbnail: false,
            fmt: TextureFormat::Dxt1,
            flags: texture_flags::FL_GENERATE_MIP_MAPS | texture_flags::FL_DITHER_COLOR,
            border_color: 0,
            fade_color: 0,
            fade_amount: 0,
            fade_delay: 0,
            mip_filter: 0, // kMIPFilterBox
            width: 0,
            height: 0,
            detail_name: String::new(),
            detail_scale: 1.0,
            texture_type: TextureType::Image,
            material: TextureMaterial::BlinPhong as u32,
            material_weight: 0.0,
            bump_virtual_height: 0.05,
            bump_mode: TextureBumpMode::None,
            bump_name: String::new(),
            ext_normal_map_name: String::new(),
        }
    }
}

impl TextureParams {
    /// Parse binary `.thm` bytes matching `STextureParams::Load` (`ETextureParams.cpp:67-112`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        use thm_chunks::*;
        let reader = ChunkReader::new(data);
        let mut params = Self::default();

        if let Some(ver_bytes) = reader.open_chunk(THM_CHUNK_VERSION)? {
            let mut vc = ByteCursor::new(&ver_bytes);
            params.thm_version = Some(vc.read_u16()?);
        }

        if let Some(type_bytes) = reader.open_chunk(THM_CHUNK_TYPE)? {
            let mut tc = ByteCursor::new(&type_bytes);
            params.object_type = Some(tc.read_u32()?);
        }

        params.has_thumbnail = reader.find_chunk(THM_CHUNK_DATA)?.is_some();

        let tp_bytes = reader.require_chunk(THM_CHUNK_TEXTUREPARAM)?;
        let mut tp = ByteCursor::new(&tp_bytes);
        params.fmt = TextureFormat::try_from(tp.read_u32()?)?;
        params.flags = tp.read_u32()?;
        params.border_color = tp.read_u32()?;
        params.fade_color = tp.read_u32()?;
        params.fade_amount = tp.read_u32()?;
        params.mip_filter = tp.read_u32()?;
        params.width = tp.read_u32()?;
        params.height = tp.read_u32()?;

        if let Some(tt_bytes) = reader.open_chunk(THM_CHUNK_TEXTURE_TYPE)? {
            let mut tc = ByteCursor::new(&tt_bytes);
            params.texture_type = TextureType::try_from(tc.read_u32()?)?;
        }

        if let Some(det_bytes) = reader.open_chunk(THM_CHUNK_DETAIL_EXT)? {
            let mut dc = ByteCursor::new(&det_bytes);
            params.detail_name = dc.read_zstring()?;
            params.detail_scale = dc.read_f32()?;
        }

        if let Some(mtl_bytes) = reader.open_chunk(THM_CHUNK_MATERIAL)? {
            let mut mc = ByteCursor::new(&mtl_bytes);
            params.material = mc.read_u32()?;
            params.material_weight = mc.read_f32()?;
        }

        if let Some(bump_bytes) = reader.open_chunk(THM_CHUNK_BUMP)? {
            let mut bc = ByteCursor::new(&bump_bytes);
            params.bump_virtual_height = bc.read_f32()?;
            params.bump_mode = TextureBumpMode::try_from(bc.read_u32()?)?;
            params.bump_name = bc.read_zstring()?;
        }

        if let Some(nm_bytes) = reader.open_chunk(THM_CHUNK_EXT_NORMALMAP)? {
            let mut nc = ByteCursor::new(&nm_bytes);
            params.ext_normal_map_name = nc.read_zstring()?;
        }

        if let Some(fd_bytes) = reader.open_chunk(THM_CHUNK_FADE_DELAY)? {
            let mut fc = ByteCursor::new(&fd_bytes);
            params.fade_delay = fc.read_u8()?;
        }

        Ok(params)
    }

    /// Parse companion per-texture `.ltx` metadata (`textures/<name>.ltx` `[texture]` section),
    /// matching `CTextureDescrMngr::LoadLTX` (`TextureDescrManager.cpp:126-157`).
    pub fn from_ltx_fallback(base_texture_name: &str, ltx_text: &str) -> Result<Self, FormatError> {
        let doc = LtxDocument::parse(ltx_text);
        if !doc.contains_section("texture") {
            return Err(FormatError::MissingChunk(thm_chunks::THM_CHUNK_TEXTUREPARAM));
        }

        let mode = doc.read_str("texture", "bump_mode").unwrap_or("none");
        let (bump_mode, steep) = if mode.eq_ignore_ascii_case("none") {
            (TextureBumpMode::None, false)
        } else if mode.eq_ignore_ascii_case("use") {
            (TextureBumpMode::Use, false)
        } else if mode.eq_ignore_ascii_case("parallax") {
            (TextureBumpMode::UseParallax, true)
        } else {
            return Err(FormatError::InvalidValue("invalid bump_mode in texture .ltx"));
        };
        let _ = steep;

        let base_stem = strip_texture_extension(base_texture_name);
        let bump_name = if bump_mode == TextureBumpMode::None {
            String::new()
        } else {
            match doc.read_str("texture", "bump_name") {
                Some(cfg) if !cfg.trim().is_empty() => {
                    strip_texture_extension(cfg.trim()).to_string()
                }
                _ => format!("{base_stem}_bump"),
            }
        };

        let mat_val = doc.read_f32("texture", "material", 1.0);
        let mat_idx = mat_val.floor().max(0.0) as u32;
        let mat_weight = (mat_val - (mat_idx as f32)).clamp(0.0, 1.0);

        Ok(Self {
            bump_mode,
            bump_name,
            material: mat_idx,
            material_weight: mat_weight,
            ..Self::default()
        })
    }

    #[inline]
    pub fn has_alpha(&self) -> bool {
        (self.flags & texture_flags::FL_HAS_ALPHA) != 0
    }

    #[inline]
    pub fn has_diffuse_detail(&self) -> bool {
        (self.flags & texture_flags::FL_DIFFUSE_DETAIL) != 0 && !self.detail_name.is_empty()
    }

    #[inline]
    pub fn has_bump_detail(&self) -> bool {
        (self.flags & texture_flags::FL_BUMP_DETAIL) != 0 && !self.detail_name.is_empty()
    }

    /// Packed material ID + weight (`TextureDescrManager.cpp:93`: `material + (material < 4 ? material_weight : 0)`).
    #[inline]
    pub fn effective_material_id(&self) -> f32 {
        (self.material as f32) + if self.material < 4 { self.material_weight } else { 0.0 }
    }

    /// Resolve companion `_bump` & `_bump#` VFS `.dds` paths if bump mapping is active.
    pub fn bump_links(&self) -> Option<BumpTextureLinks> {
        let steep_parallax = match self.bump_mode {
            TextureBumpMode::Use => false,
            TextureBumpMode::UseParallax => true,
            TextureBumpMode::Reserved | TextureBumpMode::None => return None,
        };
        let raw = self.bump_name.trim();
        if raw.is_empty() {
            return None;
        }
        let stem = strip_texture_extension(raw).replace('/', "\\");
        let bump_dds = texture_stem_to_dds_vpath(&stem);
        let bump_hash_dds = texture_stem_to_dds_vpath(&format!("{stem}#"));
        Some(BumpTextureLinks {
            bump_name: stem,
            bump_dds,
            bump_hash_dds,
            steep_parallax,
        })
    }

    /// Resolve all VFS texture links associated with `diffuse_name` (diffuse + `_bump` + `_bump#` + detail + normal map).
    pub fn resolve_texture_links(&self, diffuse_name: &str) -> ResolvedTextureLinks {
        let clean_name = strip_texture_extension(diffuse_name).replace('/', "\\");
        let diffuse_dds = texture_stem_to_dds_vpath(&clean_name);

        let detail_dds = if self.has_diffuse_detail() || self.has_bump_detail() {
            Some(texture_stem_to_dds_vpath(&self.detail_name))
        } else {
            None
        };

        let ext_normal_map_dds = if !self.ext_normal_map_name.trim().is_empty() {
            Some(texture_stem_to_dds_vpath(&self.ext_normal_map_name))
        } else {
            None
        };

        ResolvedTextureLinks {
            diffuse_name: clean_name,
            diffuse_dds,
            bump: self.bump_links(),
            detail_dds,
            detail_scale: self.detail_scale,
            use_diffuse_detail: self.has_diffuse_detail(),
            use_bump_detail: self.has_bump_detail(),
            ext_normal_map_dds,
            material_id: self.effective_material_id(),
        }
    }
}

/// Strip known X-Ray image/descriptor extensions (`TextureDescrManager::fix_texture_thm_name`).
pub fn strip_texture_extension(name: &str) -> &str {
    const EXTS: &[&str] = &[".tga", ".thm", ".ltx", ".dds", ".bmp", ".ogm", ".gif"];
    let trimmed = name.trim();
    if let Some(dot) = trimmed.rfind('.') {
        let ext = &trimmed[dot..];
        if EXTS.iter().any(|e| ext.eq_ignore_ascii_case(e)) {
            return &trimmed[..dot];
        }
    }
    trimmed
}

/// Normalize a bare texture reference (e.g. `wpn\ak74` or `textures/ui/cursor_cross.dds`)
/// into a canonical VFS path `textures\<stem>.dds`.
pub fn texture_stem_to_dds_vpath(tex_name: &str) -> String {
    let mut work = strip_texture_extension(tex_name)
        .replace('/', "\\")
        .trim_start_matches('\\')
        .to_ascii_lowercase();
    if let Some(rest) = work.strip_prefix("gamedata\\") {
        work = rest.to_string();
    }
    if work.starts_with("textures\\") {
        format!("{work}.dds")
    } else {
        format!("textures\\{work}.dds")
    }
}

/// Normalize a bare texture reference into `textures\<stem>.thm`.
pub fn texture_stem_to_thm_vpath(tex_name: &str) -> String {
    let mut work = strip_texture_extension(tex_name)
        .replace('/', "\\")
        .trim_start_matches('\\')
        .to_ascii_lowercase();
    if let Some(rest) = work.strip_prefix("gamedata\\") {
        work = rest.to_string();
    }
    if work.starts_with("textures\\") {
        format!("{work}.thm")
    } else {
        format!("textures\\{work}.thm")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;

    #[test]
    fn test_thm_parse_bump_detail_material_and_links() {
        use thm_chunks::*;
        let mut w = ChunkWriter::new();
        w.write_chunk(THM_CHUNK_VERSION, &0x0012u16.to_le_bytes());
        w.write_chunk(THM_CHUNK_TYPE, &1u32.to_le_bytes());
        w.write_chunk(THM_CHUNK_DATA, &[0xAB; 16]); // thumbnail placeholder

        let mut tp = Vec::new();
        tp.extend_from_slice(&(TextureFormat::Dxt5 as u32).to_le_bytes());
        let flags = texture_flags::FL_GENERATE_MIP_MAPS
            | texture_flags::FL_HAS_ALPHA
            | texture_flags::FL_DIFFUSE_DETAIL
            | texture_flags::FL_BUMP_DETAIL;
        tp.extend_from_slice(&flags.to_le_bytes());
        tp.extend_from_slice(&0u32.to_le_bytes()); // border_color
        tp.extend_from_slice(&0u32.to_le_bytes()); // fade_color
        tp.extend_from_slice(&0u32.to_le_bytes()); // fade_amount
        tp.extend_from_slice(&0u32.to_le_bytes()); // mip_filter
        tp.extend_from_slice(&1024u32.to_le_bytes());
        tp.extend_from_slice(&1024u32.to_le_bytes());
        w.write_chunk(THM_CHUNK_TEXTUREPARAM, &tp);

        w.write_chunk(THM_CHUNK_TEXTURE_TYPE, &(TextureType::Image as u32).to_le_bytes());

        let mut det = Vec::new();
        det.extend_from_slice(b"detail\\detail_metal_det1\0");
        det.extend_from_slice(&6.5f32.to_le_bytes());
        w.write_chunk(THM_CHUNK_DETAIL_EXT, &det);

        let mut mtl = Vec::new();
        mtl.extend_from_slice(&(TextureMaterial::PhongMetal as u32).to_le_bytes());
        mtl.extend_from_slice(&0.35f32.to_le_bytes());
        w.write_chunk(THM_CHUNK_MATERIAL, &mtl);

        let mut bump = Vec::new();
        bump.extend_from_slice(&0.04f32.to_le_bytes());
        bump.extend_from_slice(&(TextureBumpMode::UseParallax as u32).to_le_bytes());
        bump.extend_from_slice(b"wpn\\ak74_bump\0");
        w.write_chunk(THM_CHUNK_BUMP, &bump);

        w.write_chunk(THM_CHUNK_EXT_NORMALMAP, b"wpn\\ak74_nmap\0");
        w.write_chunk(THM_CHUNK_FADE_DELAY, &[3u8]);

        let parsed = TextureParams::parse(&w.into_bytes()).expect("THM must parse");
        assert_eq!(parsed.thm_version, Some(0x0012));
        assert!(parsed.has_thumbnail);
        assert!(parsed.has_alpha());
        assert_eq!(parsed.width, 1024);
        assert_eq!(parsed.height, 1024);
        assert_eq!(parsed.fade_delay, 3);
        assert!((parsed.effective_material_id() - 2.35).abs() < 1e-6);

        let links = parsed.resolve_texture_links("wpn\\ak74");
        assert_eq!(links.diffuse_dds, "textures\\wpn\\ak74.dds");
        let bump_links = links.bump.expect("parallax bump links must resolve");
        assert!(bump_links.steep_parallax);
        assert_eq!(bump_links.bump_dds, "textures\\wpn\\ak74_bump.dds");
        assert_eq!(bump_links.bump_hash_dds, "textures\\wpn\\ak74_bump#.dds");
        assert_eq!(
            links.detail_dds.as_deref(),
            Some("textures\\detail\\detail_metal_det1.dds")
        );
        assert_eq!(
            links.ext_normal_map_dds.as_deref(),
            Some("textures\\wpn\\ak74_nmap.dds")
        );
    }

    #[test]
    fn test_thm_reserved_bump_mode_clamps_to_none_and_ltx_fallback() {
        let mut w = ChunkWriter::new();
        let mut tp = Vec::new();
        tp.extend_from_slice(&(TextureFormat::Dxt1 as u32).to_le_bytes());
        for v in [1u32, 0, 0, 0, 0, 512, 512] {
            tp.extend_from_slice(&v.to_le_bytes());
        }
        w.write_chunk(thm_chunks::THM_CHUNK_TEXTUREPARAM, &tp);
        let mut bump = Vec::new();
        bump.extend_from_slice(&0.05f32.to_le_bytes());
        bump.extend_from_slice(&0u32.to_le_bytes()); // tbmResereved -> coerced to tbmNone
        bump.extend_from_slice(b"unused_bump\0");
        w.write_chunk(thm_chunks::THM_CHUNK_BUMP, &bump);

        let parsed = TextureParams::parse(&w.into_bytes()).unwrap();
        assert_eq!(parsed.bump_mode, TextureBumpMode::None);
        assert!(parsed.bump_links().is_none());

        let ltx = "[texture]\nbump_mode = use\nmaterial = 1.75\n";
        let fb = TextureParams::from_ltx_fallback("act\\act_stalker.dds", ltx).unwrap();
        assert_eq!(fb.bump_mode, TextureBumpMode::Use);
        assert!((fb.effective_material_id() - 1.75).abs() < 1e-6);
        let fb_links = fb.bump_links().unwrap();
        assert_eq!(fb_links.bump_dds, "textures\\act\\act_stalker_bump.dds");
        assert_eq!(fb_links.bump_hash_dds, "textures\\act\\act_stalker_bump#.dds");

        // Error checks: missing TEXTUREPARAM or invalid ETFormat
        let empty = ChunkWriter::new();
        assert!(matches!(
            TextureParams::parse(&empty.into_bytes()),
            Err(FormatError::MissingChunk(thm_chunks::THM_CHUNK_TEXTUREPARAM))
        ));
        let mut bad_fmt = ChunkWriter::new();
        let mut bad_tp = vec![0u8; 32];
        bad_tp[0..4].copy_from_slice(&999u32.to_le_bytes());
        bad_fmt.write_chunk(thm_chunks::THM_CHUNK_TEXTUREPARAM, &bad_tp);
        assert!(matches!(
            TextureParams::parse(&bad_fmt.into_bytes()),
            Err(FormatError::InvalidValue(_))
        ));
        assert!(TextureFormat::Dxt5.has_alpha_channel());
        assert!(!TextureFormat::Dxt1.has_alpha_channel());
    }
}
