//! Pure Rust X-Ray `.thm` Texture Metadata & `_bump` / `_bump#` Link Parser
//!
//! Faithful pure-Rust implementation matching `src/Layers/xrRender/ETextureParams.h`,
//! `src/Layers/xrRender/ETextureParams.cpp`, and `src/Layers/xrRender/TextureDescrManager.cpp`
//! (including companion `[texture]` `.ltx` fallback metadata).

use std::collections::BTreeMap;

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
    pub fn is_cubemap(&self) -> bool {
        self.texture_type == TextureType::CubeMap
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

/// Default `r__dtex_range` detail-texture fade distance (`TextureDescrManager.cpp:9`).
pub const DTEX_RANGE_DEFAULT: f32 = 50.0;

/// Detail usage bitmask stored in [`TextureAssoc::usage`] (`TextureDescrManager.cpp:83-87, 235-237`).
pub mod detail_usage {
    pub const DIFFUSE: u8 = 1 << 0;
    pub const BUMP: u8 = 1 << 1;
}

/// Detail-texture scale binder (`class cl_dt_scaler` in `TextureDescrManager.cpp:11-24`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailScaler {
    pub scale: f32,
}

impl Default for DetailScaler {
    fn default() -> Self {
        Self { scale: 1.0 }
    }
}

impl DetailScaler {
    /// Evaluate `dt_params = [scale, scale, scale, 1.0 / r_dtex_range]` (`TextureDescrManager.cpp:22`).
    #[inline]
    pub fn to_dt_params(self, r_dtex_range: f32) -> [f32; 4] {
        let inv_range = if r_dtex_range.abs() > 1e-6 && r_dtex_range.is_finite() {
            1.0 / r_dtex_range
        } else {
            1.0 / DTEX_RANGE_DEFAULT
        };
        [self.scale, self.scale, self.scale, inv_range]
    }
}

/// Detail-map association (`texture_assoc` in `TextureDescrManager.h`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureAssoc {
    pub detail_name: String,
    pub usage: u8,
}

/// Bump/parallax & surface-material association (`texture_spec` in `TextureDescrManager.h`).
#[derive(Debug, Clone, PartialEq)]
pub struct TextureSpec {
    pub bump_name: String,
    pub material: f32,
    pub use_steep_parallax: bool,
}

/// Combined descriptor record (`texture_desc` in `TextureDescrManager.h`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextureDescrEntry {
    pub assoc: Option<TextureAssoc>,
    pub spec: Option<TextureSpec>,
}

#[inline]
fn normalize_descr_key(tex_name: &str) -> String {
    strip_texture_extension(tex_name)
        .replace('/', "\\")
        .trim_start_matches('\\')
        .to_ascii_lowercase()
}

/// Multi-texture descriptor lookup table matching `CTextureDescrMngr`
/// (`src/Layers/xrRender/TextureDescrManager.cpp:40-257`).
///
/// Mirrors X-Ray's load priority: `$game_textures$` `.thm` -> `$level$` `.thm` -> `.ltx` fallback
/// (without overwriting entries already loaded from `.thm`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextureDescrTable {
    entries: BTreeMap<String, TextureDescrEntry>,
    detail_scalers: BTreeMap<String, DetailScaler>,
}

impl TextureDescrTable {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert or update an entry from a parsed `.thm` (`CTextureDescrMngr::LoadTHM`,
    /// `TextureDescrManager.cpp:62-105`).
    ///
    /// Returns `true` if `params.texture_type` is registered (`Image`, `Terrain`, `NormalMap`).
    pub fn insert_thm(&mut self, tex_name: &str, params: &TextureParams) -> bool {
        if !matches!(
            params.texture_type,
            TextureType::Image | TextureType::Terrain | TextureType::NormalMap
        ) {
            return false;
        }
        let key = normalize_descr_key(tex_name);
        let entry = self.entries.entry(key.clone()).or_default();

        let has_diffuse_or_bump_flag = (params.flags
            & (texture_flags::FL_DIFFUSE_DETAIL | texture_flags::FL_BUMP_DETAIL))
            != 0;
        if !params.detail_name.is_empty() && has_diffuse_or_bump_flag {
            let mut usage = 0u8;
            if (params.flags & texture_flags::FL_DIFFUSE_DETAIL) != 0 {
                usage |= detail_usage::DIFFUSE;
            }
            if (params.flags & texture_flags::FL_BUMP_DETAIL) != 0 {
                usage |= detail_usage::BUMP;
            }
            entry.assoc = Some(TextureAssoc {
                detail_name: params.detail_name.clone(),
                usage,
            });
            self.detail_scalers.insert(
                key,
                DetailScaler {
                    scale: params.detail_scale,
                },
            );
        }

        let (bump_name, use_steep_parallax) = match params.bump_mode {
            TextureBumpMode::Use => (params.bump_name.clone(), false),
            TextureBumpMode::UseParallax => (params.bump_name.clone(), true),
            TextureBumpMode::Reserved | TextureBumpMode::None => (String::new(), false),
        };
        entry.spec = Some(TextureSpec {
            bump_name,
            material: params.effective_material_id(),
            use_steep_parallax,
        });
        true
    }

    /// Insert a companion `[texture]` `.ltx` fallback if `tex_name` is not already present
    /// (`CTextureDescrMngr::LoadLTX`, `TextureDescrManager.cpp:109-158`).
    ///
    /// Returns `Ok(true)` if inserted, `Ok(false)` if already present (`.thm` priority) or missing
    /// the `[texture]` section.
    pub fn insert_ltx_fallback(&mut self, tex_name: &str, ltx_text: &str) -> Result<bool, FormatError> {
        let key = normalize_descr_key(tex_name);
        if self.entries.contains_key(&key) {
            return Ok(false);
        }
        let params = match TextureParams::from_ltx_fallback(&key, ltx_text) {
            Ok(tp) => tp,
            Err(FormatError::MissingChunk(_)) => return Ok(false),
            Err(err) => return Err(err),
        };
        let material = params.effective_material_id();
        let use_steep_parallax = params.bump_mode == TextureBumpMode::UseParallax;
        self.entries.insert(
            key,
            TextureDescrEntry {
                assoc: None,
                spec: Some(TextureSpec {
                    bump_name: params.bump_name,
                    material,
                    use_steep_parallax,
                }),
            },
        );
        Ok(true)
    }

    /// Look up raw descriptor entry by texture stem or filename.
    #[inline]
    pub fn get(&self, tex_name: &str) -> Option<&TextureDescrEntry> {
        let key = normalize_descr_key(tex_name);
        self.entries.get(&key)
    }

    /// Matching `CTextureDescrMngr::GetBumpName` (`TextureDescrManager.cpp:189-200`).
    #[inline]
    pub fn get_bump_name(&self, tex_name: &str) -> &str {
        self.get(tex_name)
            .and_then(|e| e.spec.as_ref())
            .map(|s| s.bump_name.as_str())
            .unwrap_or("")
    }

    /// Matching `CTextureDescrMngr::UseSteepParallax` (`TextureDescrManager.cpp:202-213`).
    #[inline]
    pub fn use_steep_parallax(&self, tex_name: &str) -> bool {
        self.get(tex_name)
            .and_then(|e| e.spec.as_ref())
            .is_some_and(|s| s.use_steep_parallax)
    }

    /// Matching `CTextureDescrMngr::GetMaterial` (`TextureDescrManager.cpp:215-226`, default `1.0`).
    #[inline]
    pub fn get_material(&self, tex_name: &str) -> f32 {
        self.get(tex_name)
            .and_then(|e| e.spec.as_ref())
            .map(|s| s.material)
            .unwrap_or(1.0)
    }

    /// Matching `CTextureDescrMngr::GetTextureUsage` (`TextureDescrManager.cpp:228-240`).
    ///
    /// Returns `(has_diffuse_detail, has_bump_detail)`.
    #[inline]
    pub fn get_texture_usage(&self, tex_name: &str) -> (bool, bool) {
        self.get(tex_name)
            .and_then(|e| e.assoc.as_ref())
            .map(|a| {
                (
                    (a.usage & detail_usage::DIFFUSE) != 0,
                    (a.usage & detail_usage::BUMP) != 0,
                )
            })
            .unwrap_or((false, false))
    }

    /// Matching `CTextureDescrMngr::GetDetailTexture` (`TextureDescrManager.cpp:242-257`).
    #[inline]
    pub fn get_detail_texture(&self, tex_name: &str) -> Option<(&str, Option<DetailScaler>)> {
        let key = normalize_descr_key(tex_name);
        let assoc = self.entries.get(&key)?.assoc.as_ref()?;
        let scaler = self.detail_scalers.get(&key).copied();
        Some((assoc.detail_name.as_str(), scaler))
    }
}

impl From<TextureSpec> for TextureDescrEntry {
    fn from(spec: TextureSpec) -> Self {
        Self {
            assoc: None,
            spec: Some(spec),
        }
    }
}

// ============================================================================
// Pure-Rust DDS Cubemap Header & Subresource Slicer
// ============================================================================

/// Little-endian `"DDS "` file magic (`0x20534444`).
pub const DDS_MAGIC: u32 = 0x2053_4444;
/// Size of `"DDS "` magic (4B) + `DDS_HEADER` (124B).
pub const DDS_HEADER_BYTES: usize = 128;
/// Size of `"DDS "` magic (4B) + `DDS_HEADER` (124B) + `DDS_HEADER_DXT10` (20B).
pub const DDS_DX10_HEADER_BYTES: usize = 148;

const DDSD_MIPMAPCOUNT: u32 = 0x0002_0000;
const DDPF_ALPHAPIXELS: u32 = 0x0000_0001;
const DDPF_FOURCC: u32 = 0x0000_0004;
const DDPF_RGB: u32 = 0x0000_0040;

const FOURCC_DXT1: u32 = u32::from_le_bytes(*b"DXT1");
const FOURCC_DXT2: u32 = u32::from_le_bytes(*b"DXT2");
const FOURCC_DXT3: u32 = u32::from_le_bytes(*b"DXT3");
const FOURCC_DXT4: u32 = u32::from_le_bytes(*b"DXT4");
const FOURCC_DXT5: u32 = u32::from_le_bytes(*b"DXT5");
const FOURCC_ATI1: u32 = u32::from_le_bytes(*b"ATI1");
const FOURCC_BC4U: u32 = u32::from_le_bytes(*b"BC4U");
const FOURCC_BC4S: u32 = u32::from_le_bytes(*b"BC4S");
const FOURCC_ATI2: u32 = u32::from_le_bytes(*b"ATI2");
const FOURCC_BC5U: u32 = u32::from_le_bytes(*b"BC5U");
const FOURCC_BC5S: u32 = u32::from_le_bytes(*b"BC5S");
const FOURCC_DX10: u32 = u32::from_le_bytes(*b"DX10");

/// DDS `dwCaps2` and `DDS_HEADER_DXT10::miscFlag` bitmasks for cubemaps.
pub mod dds_caps2 {
    pub const CUBEMAP: u32 = 0x0000_0200;
    pub const CUBEMAP_POSITIVEX: u32 = 0x0000_0400;
    pub const CUBEMAP_NEGATIVEX: u32 = 0x0000_0800;
    pub const CUBEMAP_POSITIVEY: u32 = 0x0000_1000;
    pub const CUBEMAP_NEGATIVEY: u32 = 0x0000_2000;
    pub const CUBEMAP_POSITIVEZ: u32 = 0x0000_4000;
    pub const CUBEMAP_NEGATIVEZ: u32 = 0x0000_8000;
    pub const CUBEMAP_ALLFACES: u32 = 0x0000_FC00;
    pub const DX10_MISC_TEXTURECUBE: u32 = 0x0000_0004;
}

/// Canonical Direct3D / DDS cubemap face order (`+X, -X, +Y, -Y, +Z, -Z`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CubemapFace {
    PositiveX = 0,
    NegativeX = 1,
    PositiveY = 2,
    NegativeY = 3,
    PositiveZ = 4,
    NegativeZ = 5,
}

impl CubemapFace {
    /// All 6 cubemap faces in DDS and D3D12 array-slice order.
    pub const ALL: [Self; 6] = [
        Self::PositiveX,
        Self::NegativeX,
        Self::PositiveY,
        Self::NegativeY,
        Self::PositiveZ,
        Self::NegativeZ,
    ];

    #[inline]
    pub const fn as_index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::PositiveX),
            1 => Some(Self::NegativeX),
            2 => Some(Self::PositiveY),
            3 => Some(Self::NegativeY),
            4 => Some(Self::PositiveZ),
            5 => Some(Self::NegativeZ),
            _ => None,
        }
    }

    #[inline]
    pub const fn dds_caps2_bit(self) -> u32 {
        match self {
            Self::PositiveX => dds_caps2::CUBEMAP_POSITIVEX,
            Self::NegativeX => dds_caps2::CUBEMAP_NEGATIVEX,
            Self::PositiveY => dds_caps2::CUBEMAP_POSITIVEY,
            Self::NegativeY => dds_caps2::CUBEMAP_NEGATIVEY,
            Self::PositiveZ => dds_caps2::CUBEMAP_POSITIVEZ,
            Self::NegativeZ => dds_caps2::CUBEMAP_NEGATIVEZ,
        }
    }

    /// D3D12 subresource index (`D3D12CalcSubresource(mip, face, 0, mip_count, 6)`).
    #[inline]
    pub const fn d3d12_subresource_index(self, mip_level: u32, mip_count: u32) -> u32 {
        mip_level + (self as u32) * mip_count
    }
}

/// Pure-Rust DDS pixel format descriptor mapped cleanly to numeric `DXGI_FORMAT` IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DdsPixelFormatKind {
    Bc1Unorm,
    Bc1UnormSrgb,
    Bc2Unorm,
    Bc2UnormSrgb,
    Bc3Unorm,
    Bc3UnormSrgb,
    Bc4Unorm,
    Bc4Snorm,
    Bc5Unorm,
    Bc5Snorm,
    Bc6hUf16,
    Bc6hSf16,
    Bc7Unorm,
    Bc7UnormSrgb,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Bgrx8Unorm,
    Bgrx8UnormSrgb,
    R16G16B16A16Float,
    R32G32B32A32Float,
    R16G16B16A16Unorm,
    R10G10B10A2Unorm,
    R11G11B10Float,
}

impl DdsPixelFormatKind {
    /// Numeric `DXGI_FORMAT` enum value (`dxgiformat.h`).
    #[inline]
    pub const fn dxgi_format(self) -> u32 {
        match self {
            Self::R32G32B32A32Float => 2,
            Self::R16G16B16A16Float => 10,
            Self::R16G16B16A16Unorm => 11,
            Self::R10G10B10A2Unorm => 24,
            Self::R11G11B10Float => 26,
            Self::Rgba8Unorm => 28,
            Self::Rgba8UnormSrgb => 29,
            Self::Bc1Unorm => 71,
            Self::Bc1UnormSrgb => 72,
            Self::Bc2Unorm => 74,
            Self::Bc2UnormSrgb => 75,
            Self::Bc3Unorm => 77,
            Self::Bc3UnormSrgb => 78,
            Self::Bc4Unorm => 80,
            Self::Bc4Snorm => 81,
            Self::Bc5Unorm => 83,
            Self::Bc5Snorm => 84,
            Self::Bgra8Unorm => 87,
            Self::Bgrx8Unorm => 88,
            Self::Bgra8UnormSrgb => 91,
            Self::Bgrx8UnormSrgb => 93,
            Self::Bc6hUf16 => 95,
            Self::Bc6hSf16 => 96,
            Self::Bc7Unorm => 98,
            Self::Bc7UnormSrgb => 99,
        }
    }

    #[inline]
    pub const fn from_dxgi_format(dxgi_format: u32) -> Option<Self> {
        match dxgi_format {
            2 => Some(Self::R32G32B32A32Float),
            10 => Some(Self::R16G16B16A16Float),
            11 => Some(Self::R16G16B16A16Unorm),
            24 => Some(Self::R10G10B10A2Unorm),
            26 => Some(Self::R11G11B10Float),
            28 => Some(Self::Rgba8Unorm),
            29 => Some(Self::Rgba8UnormSrgb),
            71 => Some(Self::Bc1Unorm),
            72 => Some(Self::Bc1UnormSrgb),
            74 => Some(Self::Bc2Unorm),
            75 => Some(Self::Bc2UnormSrgb),
            77 => Some(Self::Bc3Unorm),
            78 => Some(Self::Bc3UnormSrgb),
            80 => Some(Self::Bc4Unorm),
            81 => Some(Self::Bc4Snorm),
            83 => Some(Self::Bc5Unorm),
            84 => Some(Self::Bc5Snorm),
            87 => Some(Self::Bgra8Unorm),
            88 => Some(Self::Bgrx8Unorm),
            91 => Some(Self::Bgra8UnormSrgb),
            93 => Some(Self::Bgrx8UnormSrgb),
            95 => Some(Self::Bc6hUf16),
            96 => Some(Self::Bc6hSf16),
            98 => Some(Self::Bc7Unorm),
            99 => Some(Self::Bc7UnormSrgb),
            _ => None,
        }
    }

    #[inline]
    pub const fn is_block_compressed(self) -> bool {
        matches!(
            self,
            Self::Bc1Unorm
                | Self::Bc1UnormSrgb
                | Self::Bc2Unorm
                | Self::Bc2UnormSrgb
                | Self::Bc3Unorm
                | Self::Bc3UnormSrgb
                | Self::Bc4Unorm
                | Self::Bc4Snorm
                | Self::Bc5Unorm
                | Self::Bc5Snorm
                | Self::Bc6hUf16
                | Self::Bc6hSf16
                | Self::Bc7Unorm
                | Self::Bc7UnormSrgb
        )
    }

    /// Bytes per 4x4 block (if block-compressed) or bytes per pixel (if uncompressed).
    #[inline]
    pub const fn bytes_per_block_or_pixel(self) -> u32 {
        match self {
            Self::Bc1Unorm | Self::Bc1UnormSrgb | Self::Bc4Unorm | Self::Bc4Snorm => 8,
            Self::Bc2Unorm
            | Self::Bc2UnormSrgb
            | Self::Bc3Unorm
            | Self::Bc3UnormSrgb
            | Self::Bc5Unorm
            | Self::Bc5Snorm
            | Self::Bc6hUf16
            | Self::Bc6hSf16
            | Self::Bc7Unorm
            | Self::Bc7UnormSrgb => 16,
            Self::Rgba8Unorm
            | Self::Rgba8UnormSrgb
            | Self::Bgra8Unorm
            | Self::Bgra8UnormSrgb
            | Self::Bgrx8Unorm
            | Self::Bgrx8UnormSrgb
            | Self::R10G10B10A2Unorm
            | Self::R11G11B10Float => 4,
            Self::R16G16B16A16Float | Self::R16G16B16A16Unorm => 8,
            Self::R32G32B32A32Float => 16,
        }
    }
}

/// Byte slice coordinates for a single `(face, mip_level)` inside a DDS cubemap buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdsSubresourceSlice {
    pub face: CubemapFace,
    pub mip_level: u32,
    pub width: u32,
    pub height: u32,
    pub row_pitch: u32,
    pub slice_pitch: u32,
    pub offset: usize,
    pub byte_len: usize,
}

impl DdsSubresourceSlice {
    /// Zero-copy view of this face/mip payload within `dds_bytes`.
    #[inline]
    pub fn slice_bytes<'a>(&self, dds_bytes: &'a [u8]) -> Option<&'a [u8]> {
        dds_bytes.get(self.offset..self.offset.saturating_add(self.byte_len))
    }
}

#[inline]
fn read_le_u32(bytes: &[u8], off: usize) -> Result<u32, FormatError> {
    let slice = bytes.get(off..off + 4).ok_or(FormatError::UnexpectedEof)?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Zero-copy DDS cubemap layout & face/mip offset table (`+X, -X, +Y, -Y, +Z, -Z` x `0..mip_count`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DdsCubemapLayout {
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub format: DdsPixelFormatKind,
    pub dxgi_format: u32,
    pub is_dx10: bool,
    pub present_faces_mask: u32,
    pub data_offset: usize,
    pub total_payload_bytes: usize,
    /// All `6 * mip_count` subresources stored in face-major order (`face * mip_count + mip`),
    /// matching both DDS file layout and `D3D12CalcSubresource`.
    pub subresources: Vec<DdsSubresourceSlice>,
}

impl DdsCubemapLayout {
    /// Fast header check returning `true` if `dds_bytes` starts with a valid cubemap DDS header.
    pub fn is_dds_cubemap(dds_bytes: &[u8]) -> bool {
        if dds_bytes.len() < DDS_HEADER_BYTES {
            return false;
        }
        let Ok(magic) = read_le_u32(dds_bytes, 0) else {
            return false;
        };
        let Ok(header_size) = read_le_u32(dds_bytes, 4) else {
            return false;
        };
        if magic != DDS_MAGIC || header_size != 124 {
            return false;
        }
        let Ok(pf_flags) = read_le_u32(dds_bytes, 80) else {
            return false;
        };
        let Ok(pf_fourcc) = read_le_u32(dds_bytes, 84) else {
            return false;
        };
        let Ok(caps2) = read_le_u32(dds_bytes, 112) else {
            return false;
        };
        if (pf_flags & DDPF_FOURCC) != 0 && pf_fourcc == FOURCC_DX10 {
            if dds_bytes.len() < DDS_DX10_HEADER_BYTES {
                return false;
            }
            let Ok(misc_flag) = read_le_u32(dds_bytes, 136) else {
                return false;
            };
            (misc_flag & dds_caps2::DX10_MISC_TEXTURECUBE) != 0
                || (caps2 & dds_caps2::CUBEMAP) != 0
        } else {
            (caps2 & dds_caps2::CUBEMAP) != 0
        }
    }

    /// Parse and validate a legacy (`128B`) or DX10 (`148B`) DDS cubemap buffer, computing
    /// exact byte offsets and D3D12-aligned subresource slices for all 6 faces and mip levels.
    pub fn parse(dds_bytes: &[u8]) -> Result<Self, FormatError> {
        if dds_bytes.len() < DDS_HEADER_BYTES {
            return Err(FormatError::UnexpectedEof);
        }
        if read_le_u32(dds_bytes, 0)? != DDS_MAGIC {
            return Err(FormatError::InvalidValue("invalid DDS magic"));
        }
        if read_le_u32(dds_bytes, 4)? != 124 {
            return Err(FormatError::InvalidValue("invalid DDS_HEADER dwSize (expected 124)"));
        }

        let flags = read_le_u32(dds_bytes, 8)?;
        let height = read_le_u32(dds_bytes, 12)?;
        let width = read_le_u32(dds_bytes, 16)?;
        if width == 0 || height == 0 {
            return Err(FormatError::InvalidValue("DDS cubemap has zero dimension"));
        }
        let mip_count = if (flags & DDSD_MIPMAPCOUNT) != 0 {
            read_le_u32(dds_bytes, 28)?.max(1)
        } else {
            1
        };

        let pf_flags = read_le_u32(dds_bytes, 80)?;
        let pf_fourcc = read_le_u32(dds_bytes, 84)?;
        let pf_rgb_bits = read_le_u32(dds_bytes, 88)?;
        let pf_r_mask = read_le_u32(dds_bytes, 92)?;
        let pf_g_mask = read_le_u32(dds_bytes, 96)?;
        let pf_b_mask = read_le_u32(dds_bytes, 100)?;
        let pf_a_mask = read_le_u32(dds_bytes, 104)?;
        let caps2 = read_le_u32(dds_bytes, 112)?;

        let mut is_dx10 = false;
        let mut data_offset = DDS_HEADER_BYTES;
        let present_faces_mask: u32;

        let format = if (pf_flags & DDPF_FOURCC) != 0 && pf_fourcc == FOURCC_DX10 {
            is_dx10 = true;
            data_offset = DDS_DX10_HEADER_BYTES;
            if dds_bytes.len() < DDS_DX10_HEADER_BYTES {
                return Err(FormatError::UnexpectedEof);
            }
            let dxgi_fmt = read_le_u32(dds_bytes, 128)?;
            let resource_dim = read_le_u32(dds_bytes, 132)?;
            let misc_flag = read_le_u32(dds_bytes, 136)?;
            let array_size = read_le_u32(dds_bytes, 140)?;
            if resource_dim != 3 {
                return Err(FormatError::InvalidValue(
                    "DX10 DDS cubemap must have Texture2D resourceDimension (3)",
                ));
            }
            if (misc_flag & dds_caps2::DX10_MISC_TEXTURECUBE) == 0
                && (caps2 & dds_caps2::CUBEMAP) == 0
            {
                return Err(FormatError::InvalidValue("DDS file is not marked as a cubemap"));
            }
            if array_size > 1 {
                return Err(FormatError::InvalidValue(
                    "cubemap arrays (arraySize > 1) are not supported by DdsCubemapLayout",
                ));
            }
            present_faces_mask = dds_caps2::CUBEMAP_ALLFACES;
            DdsPixelFormatKind::from_dxgi_format(dxgi_fmt)
                .ok_or(FormatError::InvalidValue("unsupported DX10 cubemap DXGI_FORMAT"))?
        } else {
            if (caps2 & dds_caps2::CUBEMAP) == 0 {
                return Err(FormatError::InvalidValue("DDS file is not marked as a cubemap"));
            }
            let face_bits = caps2 & dds_caps2::CUBEMAP_ALLFACES;
            if face_bits != 0 && face_bits != dds_caps2::CUBEMAP_ALLFACES {
                return Err(FormatError::InvalidValue(
                    "incomplete legacy DDS cubemap (expected all 6 faces)",
                ));
            }
            present_faces_mask = dds_caps2::CUBEMAP_ALLFACES;

            if (pf_flags & DDPF_FOURCC) != 0 {
                match pf_fourcc {
                    FOURCC_DXT1 => DdsPixelFormatKind::Bc1Unorm,
                    FOURCC_DXT2 | FOURCC_DXT3 => DdsPixelFormatKind::Bc2Unorm,
                    FOURCC_DXT4 | FOURCC_DXT5 => DdsPixelFormatKind::Bc3Unorm,
                    FOURCC_ATI1 | FOURCC_BC4U => DdsPixelFormatKind::Bc4Unorm,
                    FOURCC_BC4S => DdsPixelFormatKind::Bc4Snorm,
                    FOURCC_ATI2 | FOURCC_BC5U => DdsPixelFormatKind::Bc5Unorm,
                    FOURCC_BC5S => DdsPixelFormatKind::Bc5Snorm,
                    // D3DFMT_A16B16G16R16 (36), A16B16G16R16F (113), A32B32G32R32F (116)
                    36 => DdsPixelFormatKind::R16G16B16A16Unorm,
                    113 => DdsPixelFormatKind::R16G16B16A16Float,
                    116 => DdsPixelFormatKind::R32G32B32A32Float,
                    _ => return Err(FormatError::InvalidValue("unsupported DDS cubemap FourCC")),
                }
            } else if (pf_flags & DDPF_RGB) != 0 && pf_rgb_bits == 32 {
                if pf_r_mask == 0x0000_00FF
                    && pf_g_mask == 0x0000_FF00
                    && pf_b_mask == 0x00FF_0000
                {
                    DdsPixelFormatKind::Rgba8Unorm
                } else if pf_r_mask == 0x00FF_0000
                    && pf_g_mask == 0x0000_FF00
                    && pf_b_mask == 0x0000_00FF
                {
                    if (pf_flags & DDPF_ALPHAPIXELS) != 0 && pf_a_mask != 0 {
                        DdsPixelFormatKind::Bgra8Unorm
                    } else {
                        DdsPixelFormatKind::Bgrx8Unorm
                    }
                } else {
                    return Err(FormatError::InvalidValue("unsupported DDS cubemap RGB masks"));
                }
            } else {
                return Err(FormatError::InvalidValue("unsupported DDS cubemap pixel format"));
            }
        };

        let is_bc = format.is_block_compressed();
        let bpp_or_block = format.bytes_per_block_or_pixel();
        let mut subresources = Vec::with_capacity(6 * (mip_count as usize));
        let mut cursor = data_offset;

        for face in CubemapFace::ALL {
            let mut mip_w = width;
            let mut mip_h = height;
            for mip_level in 0..mip_count {
                let (row_pitch, slice_pitch) = if is_bc {
                    let bw = mip_w.div_ceil(4);
                    let bh = mip_h.div_ceil(4);
                    let rp = bw
                        .checked_mul(bpp_or_block)
                        .ok_or(FormatError::InvalidValue("DDS row pitch overflow"))?;
                    let sp = rp
                        .checked_mul(bh)
                        .ok_or(FormatError::InvalidValue("DDS slice pitch overflow"))?;
                    (rp, sp)
                } else {
                    let rp = mip_w
                        .checked_mul(bpp_or_block)
                        .ok_or(FormatError::InvalidValue("DDS row pitch overflow"))?;
                    let sp = rp
                        .checked_mul(mip_h)
                        .ok_or(FormatError::InvalidValue("DDS slice pitch overflow"))?;
                    (rp, sp)
                };

                let byte_len = slice_pitch as usize;
                let end = cursor
                    .checked_add(byte_len)
                    .ok_or(FormatError::InvalidValue("DDS offset overflow"))?;
                if end > dds_bytes.len() {
                    return Err(FormatError::UnexpectedEof);
                }

                subresources.push(DdsSubresourceSlice {
                    face,
                    mip_level,
                    width: mip_w,
                    height: mip_h,
                    row_pitch,
                    slice_pitch,
                    offset: cursor,
                    byte_len,
                });

                cursor = end;
                mip_w = (mip_w / 2).max(1);
                mip_h = (mip_h / 2).max(1);
            }
        }

        Ok(Self {
            width,
            height,
            mip_count,
            format,
            dxgi_format: format.dxgi_format(),
            is_dx10,
            present_faces_mask,
            data_offset,
            total_payload_bytes: cursor - data_offset,
            subresources,
        })
    }

    #[inline]
    pub fn subresource(&self, face: CubemapFace, mip_level: u32) -> Option<&DdsSubresourceSlice> {
        if mip_level >= self.mip_count {
            return None;
        }
        let idx = face.as_index() * (self.mip_count as usize) + (mip_level as usize);
        self.subresources.get(idx)
    }

    #[inline]
    pub fn face_mips(&self, face: CubemapFace) -> &[DdsSubresourceSlice] {
        let m = self.mip_count as usize;
        let start = face.as_index() * m;
        self.subresources.get(start..start + m).unwrap_or(&[])
    }

    #[inline]
    pub const fn d3d12_subresource_index(&self, face: CubemapFace, mip_level: u32) -> Option<u32> {
        if mip_level < self.mip_count {
            Some(face.d3d12_subresource_index(mip_level, self.mip_count))
        } else {
            None
        }
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

    fn build_synthetic_cubemap_dds(
        width: u32,
        height: u32,
        mip_count: u32,
        fourcc: u32,
        dx10_fmt: Option<u32>,
        caps2: u32,
        dx10_misc: u32,
    ) -> Vec<u8> {
        let mut hdr = vec![0u8; DDS_HEADER_BYTES];
        hdr[0..4].copy_from_slice(&DDS_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&124u32.to_le_bytes());
        let flags = 0x1 | 0x2 | 0x4 | 0x1000 | if mip_count > 1 { DDSD_MIPMAPCOUNT } else { 0 };
        hdr[8..12].copy_from_slice(&flags.to_le_bytes());
        hdr[12..16].copy_from_slice(&height.to_le_bytes());
        hdr[16..20].copy_from_slice(&width.to_le_bytes());
        hdr[28..32].copy_from_slice(&mip_count.to_le_bytes());
        hdr[76..80].copy_from_slice(&32u32.to_le_bytes()); // ddspf size
        hdr[80..84].copy_from_slice(&DDPF_FOURCC.to_le_bytes());
        hdr[84..88].copy_from_slice(&fourcc.to_le_bytes());
        hdr[108..112].copy_from_slice(&0x0040_1008u32.to_le_bytes()); // COMPLEX | MIPMAP | TEXTURE
        hdr[112..116].copy_from_slice(&caps2.to_le_bytes());

        if let Some(dxgi) = dx10_fmt {
            hdr.extend_from_slice(&dxgi.to_le_bytes());
            hdr.extend_from_slice(&3u32.to_le_bytes()); // Texture2D
            hdr.extend_from_slice(&dx10_misc.to_le_bytes());
            hdr.extend_from_slice(&1u32.to_le_bytes()); // arraySize = 1 cubemap
            hdr.extend_from_slice(&0u32.to_le_bytes()); // miscFlags2
        }

        let fmt = dx10_fmt
            .and_then(DdsPixelFormatKind::from_dxgi_format)
            .unwrap_or(DdsPixelFormatKind::Bc1Unorm);
        let bpb = fmt.bytes_per_block_or_pixel();

        for face in 0..6u8 {
            let mut mw = width;
            let mut mh = height;
            for mip in 0..mip_count {
                let sz = if fmt.is_block_compressed() {
                    mw.div_ceil(4) * mh.div_ceil(4) * bpb
                } else {
                    mw * mh * bpb
                } as usize;
                let tag = (face << 4) | (mip as u8);
                hdr.resize(hdr.len() + sz, tag);
                mw = (mw / 2).max(1);
                mh = (mh / 2).max(1);
            }
        }
        hdr
    }

    #[test]
    fn test_thm_is_cubemap_and_dds_cubemap_legacy_dx10_slicing() {
        // 1. Legacy DXT1 8x8 cubemap with 2 mips: mip0 = 4 blocks * 8 = 32B, mip1 = 1 block * 8 = 8B -> 40B/face, 240B total
        let legacy = build_synthetic_cubemap_dds(
            8,
            8,
            2,
            FOURCC_DXT1,
            None,
            dds_caps2::CUBEMAP | dds_caps2::CUBEMAP_ALLFACES,
            0,
        );
        assert!(DdsCubemapLayout::is_dds_cubemap(&legacy));
        let layout = DdsCubemapLayout::parse(&legacy).unwrap();
        assert!(!layout.is_dx10);
        assert_eq!(layout.format, DdsPixelFormatKind::Bc1Unorm);
        assert_eq!(layout.dxgi_format, 71);
        assert_eq!(layout.mip_count, 2);
        assert_eq!(layout.data_offset, 128);
        assert_eq!(layout.total_payload_bytes, 240);
        assert_eq!(layout.subresources.len(), 12);

        let neg_z_mip1 = layout.subresource(CubemapFace::NegativeZ, 1).unwrap();
        assert_eq!(
            layout.d3d12_subresource_index(CubemapFace::NegativeZ, 1),
            Some(11)
        );
        assert_eq!(neg_z_mip1.slice_bytes(&legacy).unwrap(), &[0x51; 8]);
        assert_eq!(layout.face_mips(CubemapFace::PositiveY).len(), 2);

        // 2. DX10 BC6H_UF16 4x4 cubemap with 1 mip marked only via DX10 miscFlag (TEXTURECUBE)
        let dx10 = build_synthetic_cubemap_dds(
            4,
            4,
            1,
            FOURCC_DX10,
            Some(95),
            0,
            dds_caps2::DX10_MISC_TEXTURECUBE,
        );
        assert!(DdsCubemapLayout::is_dds_cubemap(&dx10));
        let dx10_layout = DdsCubemapLayout::parse(&dx10).unwrap();
        assert!(dx10_layout.is_dx10);
        assert_eq!(dx10_layout.format, DdsPixelFormatKind::Bc6hUf16);
        assert_eq!(dx10_layout.dxgi_format, 95);
        assert_eq!(dx10_layout.total_payload_bytes, 6 * 16);
        assert_eq!(
            dx10_layout
                .subresource(CubemapFace::PositiveY, 0)
                .unwrap()
                .slice_bytes(&dx10)
                .unwrap(),
            &[0x20; 16]
        );

        // 3. Reject non-cubemap 2D DDS & truncated buffers
        let flat_2d = build_synthetic_cubemap_dds(8, 8, 1, FOURCC_DXT1, None, 0, 0);
        assert!(!DdsCubemapLayout::is_dds_cubemap(&flat_2d));
        assert!(DdsCubemapLayout::parse(&flat_2d).is_err());

        let truncated = &legacy[..legacy.len() - 4];
        assert!(matches!(
            DdsCubemapLayout::parse(truncated),
            Err(FormatError::UnexpectedEof)
        ));

        // 4. THM is_cubemap check
        let mut thm_w = ChunkWriter::new();
        let mut tp = vec![0u8; 32];
        tp[24..28].copy_from_slice(&512u32.to_le_bytes());
        tp[28..32].copy_from_slice(&512u32.to_le_bytes());
        thm_w.write_chunk(thm_chunks::THM_CHUNK_TEXTUREPARAM, &tp);
        thm_w.write_chunk(
            thm_chunks::THM_CHUNK_TEXTURE_TYPE,
            &(TextureType::CubeMap as u32).to_le_bytes(),
        );
        let thm = TextureParams::parse(&thm_w.into_bytes()).unwrap();
        assert!(thm.is_cubemap());
    }

    #[test]
    fn texture_descr_table_thm_priority_ltx_fallback_and_detail_scaler() {
        let mut table = TextureDescrTable::new();
        assert!(table.is_empty());
        assert_eq!(table.get_material("missing\\tex"), 1.0);
        assert_eq!(table.get_bump_name("missing\\tex"), "");
        assert!(!table.use_steep_parallax("missing\\tex"));
        assert_eq!(table.get_texture_usage("missing\\tex"), (false, false));

        let wall_thm = TextureParams {
            texture_type: TextureType::Image,
            flags: texture_flags::FL_DIFFUSE_DETAIL | texture_flags::FL_BUMP_DETAIL,
            detail_name: "detail\\detail_brick".to_string(),
            detail_scale: 4.5,
            material: TextureMaterial::PhongMetal as u32,
            material_weight: 0.25,
            bump_mode: TextureBumpMode::UseParallax,
            bump_name: "wall\\wall_brick_bump".to_string(),
            ..TextureParams::default()
        };
        assert!(table.insert_thm("wall/wall_brick.dds", &wall_thm));

        // Cubemap THM entries are skipped by LoadTHM
        let sky_thm = TextureParams {
            texture_type: TextureType::CubeMap,
            ..TextureParams::default()
        };
        assert!(!table.insert_thm("sky\\sky_01", &sky_thm));

        // .ltx cannot overwrite an existing .thm entry
        let ltx_override_attempt = "[texture]\nbump_mode = use\nbump_name = wrong_bump\nmaterial = 0.5\n";
        assert!(!table.insert_ltx_fallback("wall\\wall_brick", ltx_override_attempt).unwrap());

        assert_eq!(table.get_bump_name("WALL\\wall_brick.TGA"), "wall\\wall_brick_bump");
        assert!(table.use_steep_parallax("wall\\wall_brick"));
        assert!((table.get_material("wall\\wall_brick") - 2.25).abs() < 1e-6);
        assert_eq!(table.get_texture_usage("wall\\wall_brick"), (true, true));

        let (det_name, scaler) = table.get_detail_texture("wall\\wall_brick").unwrap();
        assert_eq!(det_name, "detail\\detail_brick");
        let dt = scaler.unwrap().to_dt_params(DTEX_RANGE_DEFAULT);
        assert_eq!(dt, [4.5, 4.5, 4.5, 1.0 / 50.0]);

        // .ltx fallback populates missing entry
        let fallback_ltx = "[texture]\nbump_mode = parallax\nmaterial = 1.75\n";
        assert!(table.insert_ltx_fallback("floor\\tile_01", fallback_ltx).unwrap());
        assert_eq!(table.get_bump_name("floor\\tile_01"), "floor\\tile_01_bump");
        assert!(table.use_steep_parallax("floor\\tile_01"));
        assert!((table.get_material("floor\\tile_01") - 1.75).abs() < 1e-6);
    }
}
