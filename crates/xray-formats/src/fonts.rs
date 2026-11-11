//! Pure Rust X-Ray `CGameFont`, `GetFontTextureName`, `MbHelpers`, and `dxFontRender` Layout Parsers
//!
//! Mirrors `src/xrEngine/GameFont.{h,cpp}`, `src/xrEngine/MbHelpers.{h,cpp}`,
//! `src/Layers/xrRender/dxFontRender.cpp`, and `src/xrGame/HUDManager.cpp`:
//! - Resolution rung selection (`texture_h<N>` + legacy `texture800`/`texture`/`texture1600`/`texture2160`)
//! - Language atlas suffix resolution (`[string_table] font_prefix`, exempting DI HUD/console fonts)
//! - All 4 font atlas `.ini` layouts: `[mb_symbol_coords]`, `[symbol_coords]`, `[char widths]`, `[font_size]`
//! - UTF-8 / dumb-fallback multibyte decoding (`mbh_multi2wide`), CJK/word-wrap predicates,
//!   string width measurement (`size_of`), wrapping (`split_by_width`, `cut_length_pos`),
//!   and screen-space glyph quad emission (`build_string_quads`) for `UiCanvas` DX12 rendering.

use std::cmp::Ordering;
use std::collections::HashMap;

use xray_core::ltx_loader::OwnedLtxDocument;
use xray_core::LtxDocument;

use crate::chunk::FormatError;
use crate::thm::{strip_texture_extension, texture_stem_to_dds_vpath};
use crate::xml_ui::{
    argb_to_rgba_f32, color_argb, subst_alpha, UiColorDefs, UiTextDef, UiViewport, UiWindowDef,
};

/// Standard font configuration section names loaded by `CFontManager::InitializeFonts`.
pub const STANDARD_FONT_SECTIONS: [&str; 11] = [
    "hud_font_medium",
    "hud_font_di",
    "ui_font_arial_14",
    "ui_font_graffiti19_russian",
    "ui_font_graffiti22_russian",
    "ui_font_letterica16_russian",
    "ui_font_letterica18_russian",
    "ui_font_graff_32",
    "ui_font_graff_50",
    "ui_font_letter_25",
    "stat_font",
];

/// `CGameFont` bitflags (`src/xrEngine/GameFont.h:54-63`).
pub mod font_flags {
    pub const GRADIENT: u32 = 1 << 0;
    pub const DEVICE_INDEPENDENT: u32 = 1 << 1;
    pub const VALID: u32 = 1 << 2;
    pub const MULTIBYTE: u32 = 1 << 3;
}

/// Horizontal text alignment (`CGameFont::EAligment`, `GameFont.h:13-18`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FontAlignment {
    #[default]
    Left = 0,
    Right = 1,
    Center = 2,
}

#[derive(Debug, Clone, Copy)]
struct LegacyFontRung {
    key: &'static str,
    lo: u32,
    hi: u32,
    authored: u32,
}

const LEGACY_RUNGS: [LegacyFontRung; 4] = [
    LegacyFontRung {
        key: "texture800",
        lo: 0,
        hi: 601,
        authored: 600,
    },
    LegacyFontRung {
        key: "texture",
        lo: 601,
        hi: 1024,
        authored: 768,
    },
    LegacyFontRung {
        key: "texture1600",
        lo: 1024,
        hi: 1440,
        authored: 1200,
    },
    LegacyFontRung {
        key: "texture2160",
        lo: 1440,
        hi: 0,
        authored: 2160,
    },
];

const LEGACY_DEFAULT_IDX: usize = 1;
const RUNG_PREFIX: &str = "texture_h";

fn parse_rung_height(key: &str) -> Option<u32> {
    let rest = key.strip_prefix(RUNG_PREFIX)?;
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let val = rest.parse::<u32>().ok()?;
    (val > 0).then_some(val)
}

/// Ranked texture rung candidate computed by `rank_font_candidates` (`GetFontTextureName`).
#[derive(Debug, Clone, PartialEq)]
pub struct FontRungCandidate {
    pub key: String,
    pub texture: String,
    pub resolved_texture: String,
    pub atlas_ini_vpath: String,
    pub authored_height: u32,
    pub log_distance: f32,
    pub explicit_rung: bool,
}

/// Resolved font section metadata (`fonts.ltx` / `system.ltx` `[<font_section>]`).
#[derive(Debug, Clone, PartialEq)]
pub struct FontSectionConfig {
    pub section: String,
    pub shader: String,
    pub texture: String,
    pub resolved_texture: String,
    pub atlas_ini_vpath: String,
    pub texture_dds_vpath: String,
    pub size: Option<f32>,
    pub interval: [f32; 2],
}

/// Apply `[string_table] font_prefix` suffix unless font is device-independent HUD/console
/// (`resolve_font_texture`, `GameFont.cpp:56-66`).
pub fn resolve_font_texture_name(base: &str, font_prefix: Option<&str>) -> String {
    let is_di = base.contains("ui_font_hud_01")
        || base.contains("ui_font_hud_02")
        || base.contains("ui_font_console_02");
    if !is_di
        && let Some(prefix) = font_prefix
        && !prefix.is_empty()
    {
        return format!("{base}{prefix}");
    }
    base.to_string()
}

/// Convert a resolved font texture name into its `$game_textures$` `.ini` virtual path.
pub fn font_atlas_ini_vpath(resolved_texture: &str) -> String {
    let mut stem = strip_texture_extension(resolved_texture).replace('/', "\\");
    if stem.len() >= 4 && stem[stem.len() - 4..].eq_ignore_ascii_case(".ini") {
        stem.truncate(stem.len() - 4);
    }
    let trimmed = stem.trim_start_matches('\\');
    let clean = if trimmed.len() >= 9 && trimmed[..9].eq_ignore_ascii_case("gamedata\\") {
        &trimmed[9..]
    } else {
        trimmed
    };
    if clean.len() >= 9 && clean[..9].eq_ignore_ascii_case("textures\\") {
        format!("{clean}.ini")
    } else {
        format!("textures\\{clean}.ini")
    }
}

/// Decode a UTF-8 or single-byte fallback string into X-Ray `wide_char` (`u16`) code units +
/// byte start positions (`mbhMulti2Wide`, `src/xrEngine/MbHelpers.cpp:69-159`).
///
/// Returns `(wide_chars, byte_positions)` where `byte_positions.len() == wide_chars.len() + 1`
/// and `byte_positions[wide_chars.len()] == text.len()` (empty slice if `text` is empty).
pub fn mbh_multi2wide(text: &str) -> (Vec<u16>, Vec<usize>) {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut wides = Vec::with_capacity(bytes.len());
    let mut positions = Vec::with_capacity(bytes.len() + 1);
    let mut spos = 0usize;

    while spos < bytes.len() {
        positions.push(spos);
        let b1 = bytes[spos];
        spos += 1;

        let wc = if (b1 & 0x80) == 0x00 {
            u16::from(b1)
        } else if (b1 & 0xE0) == 0xC0 {
            if spos >= bytes.len() {
                return mbh_multi2wide_dumb(bytes);
            }
            let b2 = bytes[spos];
            spos += 1;
            if (b2 & 0xC0) != 0x80 {
                return mbh_multi2wide_dumb(bytes);
            }
            ((u16::from(b1) & 0x1F) << 6) | (u16::from(b2) & 0x3F)
        } else if (b1 & 0xF0) == 0xE0 {
            if spos + 1 >= bytes.len() {
                return mbh_multi2wide_dumb(bytes);
            }
            let b2 = bytes[spos];
            spos += 1;
            if (b2 & 0xC0) != 0x80 {
                return mbh_multi2wide_dumb(bytes);
            }
            let b3 = bytes[spos];
            spos += 1;
            if (b3 & 0xC0) != 0x80 {
                return mbh_multi2wide_dumb(bytes);
            }
            ((u16::from(b1) & 0x0F) << 12)
                | ((u16::from(b2) & 0x3F) << 6)
                | (u16::from(b3) & 0x3F)
        } else {
            return mbh_multi2wide_dumb(bytes);
        };

        wides.push(wc);
    }

    positions.push(spos);
    (wides, positions)
}

fn mbh_multi2wide_dumb(bytes: &[u8]) -> (Vec<u16>, Vec<usize>) {
    let mut wides = Vec::with_capacity(bytes.len());
    let mut positions = Vec::with_capacity(bytes.len() + 1);
    for (idx, &b) in bytes.iter().enumerate() {
        positions.push(idx);
        wides.push(u16::from(b));
    }
    positions.push(bytes.len());
    (wides, positions)
}

/// `IsNeedSpaceCharacter` (`src/xrEngine/MbHelpers.h:11-31`).
#[inline]
pub fn is_need_space_character(wc: u16) -> bool {
    matches!(
        wc,
        0x0020
            | 0x3000
            | 0xFF01
            | 0xFF0C
            | 0xFF0E
            | 0xFF1A
            | 0xFF1B
            | 0xFF1F
            | 0x2026
            | 0x3002
            | 0x3001
    )
}

/// `IsBadStartCharacter` (`src/xrEngine/MbHelpers.h:33-49`).
#[inline]
pub fn is_bad_start_character(wc: u16) -> bool {
    is_need_space_character(wc)
        || matches!(
            wc,
            0x0021 | 0x002C | 0x002E | 0x003A | 0x003B | 0x003F | 0x0029 | 0xFF09
        )
}

/// `IsBadEndCharacter` (`src/xrEngine/MbHelpers.h:51-60`).
#[inline]
pub fn is_bad_end_character(wc: u16) -> bool {
    matches!(wc, 0x0028 | 0xFF08 | 0x4E00)
}

/// `IsAlphaCharacter` (`src/xrEngine/MbHelpers.h:62-73`).
#[inline]
pub fn is_alpha_character(wc: u16) -> bool {
    matches!(
        wc,
        0x0030..=0x0039
            | 0x0041..=0x005A
            | 0x0061..=0x007A
            | 0xFF10..=0xFF19
            | 0xFF21..=0xFF3A
            | 0xFF41..=0xFF5A
    )
}

/// Pixel-space glyph bounding coordinates in the font atlas (`Fvector TCMap[i]` in `GameFont.cpp`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FontGlyphCoord {
    pub x: f32,
    pub y: f32,
    pub width: f32,
}

/// Atlas encoding layout detected from the font `.ini` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FontAtlasKind {
    MbSymbolCoords,
    SymbolCoords,
    CharWidths,
    FontSizeGrid,
}

/// Parsed font glyph metric table (`<font_texture>.ini`).
#[derive(Debug, Clone, PartialEq)]
pub struct FontAtlas {
    pub kind: FontAtlasKind,
    pub is_multibyte: bool,
    pub line_height: f32,
    pub x_step: f32,
    pub y_step: f32,
    pub default_glyph: FontGlyphCoord,
    pub sb_glyphs: Vec<FontGlyphCoord>,
    pub mb_glyphs: HashMap<u16, FontGlyphCoord>,
}

/// Text rendering state passed to `FontAtlas::build_string_quads` (`dxFontRender::OnRender`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FontDrawParams {
    pub current_height: f32,
    pub interval: [f32; 2],
    pub scale: [f32; 2],
    pub align: FontAlignment,
    pub color_argb: u32,
    pub gradient: bool,
    /// Apply DX10+ `-0.5` screen-space pixel offset (`dxFontRender.cpp:109-113`).
    pub dx_half_pixel_offset: bool,
}

impl Default for FontDrawParams {
    fn default() -> Self {
        Self {
            current_height: 16.0,
            interval: [1.0, 1.0],
            scale: [1.0, 1.0],
            align: FontAlignment::Left,
            color_argb: 0xFFFF_FFFF,
            gradient: false,
            dx_half_pixel_offset: true,
        }
    }
}

/// Positioned glyph quad ready to upload into DX12 UI Canvas buffers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FontGlyphQuad {
    pub code: u16,
    pub min_xy: [f32; 2],
    pub max_xy: [f32; 2],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub color_top_argb: u32,
    pub color_bottom_argb: u32,
}

impl FontGlyphQuad {
    /// Bounding box as `[x, y, width, height]` matching `xray_render_dx12::ui_canvas::GlyphQuad::screen_rect`.
    #[inline]
    pub fn rect_xywh(&self) -> [f32; 4] {
        [
            self.min_xy[0],
            self.min_xy[1],
            self.max_xy[0] - self.min_xy[0],
            self.max_xy[1] - self.min_xy[1],
        ]
    }

    /// Normalized atlas UVs `[u1, v1, u2, v2]` matching `xray_render_dx12::ui_canvas::GlyphQuad::uv_rect`.
    #[inline]
    pub fn uv_rect(&self) -> [f32; 4] {
        [self.uv_min[0], self.uv_min[1], self.uv_max[0], self.uv_max[1]]
    }

    /// Combined `(screen_rect_xywh, uv_rect_u1v1u2v2)` tuple ready for `UiCanvas::draw_glyph_quads`.
    #[inline]
    pub fn screen_and_uv_rects(&self) -> ([f32; 4], [f32; 4]) {
        (self.rect_xywh(), self.uv_rect())
    }

    /// Normalized `[r, g, b, a]` color for the top vertices (`dxFontRender` tint).
    #[inline]
    pub fn top_color_rgba_f32(&self) -> [f32; 4] {
        argb_to_rgba_f32(self.color_top_argb)
    }

    /// Normalized `[r, g, b, a]` color for the bottom vertices.
    #[inline]
    pub fn bottom_color_rgba_f32(&self) -> [f32; 4] {
        argb_to_rgba_f32(self.color_bottom_argb)
    }
}

/// Vertical alignment inside a UI control (`EVTextAlignment`, `src/xrGame/ui/UILines.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiTextVerticalAlign {
    #[default]
    Top = 0,
    Center = 1,
    Bottom = 2,
}

impl FontAlignment {
    /// Parse XML `align="l"|"c"|"r"` attribute (`CUIXmlInit::InitText`).
    pub fn parse_xml_align(s: &str) -> Self {
        let t = s.trim();
        if t.eq_ignore_ascii_case("c") || t.eq_ignore_ascii_case("center") {
            Self::Center
        } else if t.eq_ignore_ascii_case("r") || t.eq_ignore_ascii_case("right") {
            Self::Right
        } else {
            Self::Left
        }
    }
}

impl UiTextVerticalAlign {
    /// Parse XML `vert_align="t"|"c"|"b"` attribute (`CUIXmlInit::InitText`).
    pub fn parse_xml_align(s: &str) -> Self {
        let t = s.trim();
        if t.eq_ignore_ascii_case("c") || t.eq_ignore_ascii_case("center") {
            Self::Center
        } else if t.eq_ignore_ascii_case("b") || t.eq_ignore_ascii_case("bottom") {
            Self::Bottom
        } else {
            Self::Top
        }
    }
}

/// Resolve an XML `<text font="...">` short alias (`"letterica18"`, `"graffiti19"`, `"medium"`, ...)
/// into its `fonts.ltx` section name (`CUIXmlInit::InitFont`, `UIXmlInit.cpp:811-859`).
pub fn resolve_ui_font_section(xml_font_name: &str) -> &str {
    match xml_font_name.trim() {
        "arial" | "arial_14" => "ui_font_arial_14",
        "medium" => "hud_font_medium",
        "di" => "hud_font_di",
        "small" => "stat_font",
        "graffiti19" => "ui_font_graffiti19_russian",
        "graffiti22" => "ui_font_graffiti22_russian",
        "graffiti32" => "ui_font_graff_32",
        "graffiti50" => "ui_font_graff_50",
        "letterica16" => "ui_font_letterica16_russian",
        "letterica18" => "ui_font_letterica18_russian",
        "letterica25" => "ui_font_letter_25",
        other => other,
    }
}

/// Single color-tagged substring run produced by `CUILines::ParseTextToColoredLine`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiColoredSegment {
    pub text: String,
    pub color_argb: u32,
}

/// Options controlling `FontAtlas::build_multiline_quads` (`CUILines::Draw`, `src/xrGame/ui/UILines.cpp`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UiMultilineTextOptions {
    pub draw: FontDrawParams,
    pub vert_align: UiTextVerticalAlign,
    pub complex_mode: bool,
    pub word_wrap: bool,
    pub coloring_mode: bool,
    pub recognize_newline: bool,
    pub ellipsis: bool,
    pub password_mode: bool,
    pub offset: [f32; 2],
}

impl Default for UiMultilineTextOptions {
    fn default() -> Self {
        Self {
            draw: FontDrawParams::default(),
            vert_align: UiTextVerticalAlign::Top,
            complex_mode: false,
            word_wrap: true,
            coloring_mode: true,
            recognize_newline: true,
            ellipsis: false,
            password_mode: false,
            offset: [0.0, 0.0],
        }
    }
}

fn parse_color_tag_argb(
    tag_inner: &str,
    default_color_argb: u32,
    color_defs: Option<&UiColorDefs>,
) -> Option<u32> {
    let inner = tag_inner.trim();
    if inner.eq_ignore_ascii_case("default") {
        return Some(default_color_argb);
    }
    if let Some(defs) = color_defs
        && let Some(&clr) = defs.colors.get(inner)
    {
        return Some(clr);
    }
    let mut nums = [0i32; 4];
    let mut count = 0usize;
    for part in inner.split(',') {
        if count >= 4 {
            return None;
        }
        nums[count] = part.trim().parse::<i32>().ok()?;
        count += 1;
    }
    if count == 4 {
        Some(color_argb(
            nums[0].clamp(0, 255) as u8,
            nums[1].clamp(0, 255) as u8,
            nums[2].clamp(0, 255) as u8,
            nums[3].clamp(0, 255) as u8,
        ))
    } else {
        None
    }
}

/// Parse `%c[...]` inline color tags and `\\n`/`\n` line breaks into logical lines of colored segments
/// (`CUILines::ParseTextToColoredLine` + `ProcessNewLines`, `UILines.cpp:507-618`).
pub fn parse_colored_text_lines(
    text: &str,
    default_color_argb: u32,
    color_defs: Option<&UiColorDefs>,
    coloring_mode: bool,
    recognize_newline: bool,
) -> Vec<Vec<UiColoredSegment>> {
    let normalized_nl = if recognize_newline {
        text.replace("\r\n", "\n").replace("\\n", "\n")
    } else {
        text.to_owned()
    };

    let base_alpha = ((default_color_argb >> 24) & 0xFF) as u8;
    let mut raw_segments: Vec<UiColoredSegment> = Vec::new();

    if !coloring_mode {
        raw_segments.push(UiColoredSegment {
            text: normalized_nl,
            color_argb: default_color_argb,
        });
    } else {
        let mut cursor = 0usize;
        let mut active_color = default_color_argb;
        while let Some(rel) = normalized_nl[cursor..].find("%c[") {
            let tag_start = cursor + rel;
            if tag_start > cursor {
                raw_segments.push(UiColoredSegment {
                    text: normalized_nl[cursor..tag_start].to_owned(),
                    color_argb: subst_alpha(active_color, base_alpha),
                });
            }
            let after_open = tag_start + 3;
            if let Some(close_rel) = normalized_nl[after_open..].find(']') {
                let inner = &normalized_nl[after_open..after_open + close_rel];
                if let Some(clr) = parse_color_tag_argb(inner, default_color_argb, color_defs) {
                    active_color = clr;
                    cursor = after_open + close_rel + 1;
                    continue;
                }
            }
            // Literal fallback if unclosed/malformed tag
            raw_segments.push(UiColoredSegment {
                text: "%c[".to_owned(),
                color_argb: subst_alpha(active_color, base_alpha),
            });
            cursor = after_open;
        }
        if cursor < normalized_nl.len() || raw_segments.is_empty() {
            raw_segments.push(UiColoredSegment {
                text: normalized_nl[cursor..].to_owned(),
                color_argb: subst_alpha(active_color, base_alpha),
            });
        }
    }

    if !recognize_newline {
        return vec![raw_segments];
    }

    let mut lines: Vec<Vec<UiColoredSegment>> = vec![Vec::new()];
    for seg in raw_segments {
        let mut parts = seg.text.split('\n').peekable();
        while let Some(part) = parts.next() {
            if !part.is_empty() {
                lines.last_mut().expect("non-empty").push(UiColoredSegment {
                    text: part.to_owned(),
                    color_argb: seg.color_argb,
                });
            }
            if parts.peek().is_some() {
                lines.push(Vec::new());
            }
        }
    }
    lines
}

trait LtxQueryView {
    fn has_section(&self, section: &str) -> bool;
    fn get_str(&self, section: &str, key: &str) -> Option<&str>;
    fn section_keys(&self, section: &str) -> Vec<(&str, &str)>;
}

impl<'a> LtxQueryView for LtxDocument<'a> {
    fn has_section(&self, section: &str) -> bool {
        self.contains_section(section)
    }

    fn get_str(&self, section: &str, key: &str) -> Option<&str> {
        self.read_str(section, key)
    }

    fn section_keys(&self, section: &str) -> Vec<(&str, &str)> {
        self.sections
            .get(section)
            .map(|s| s.entries.iter().map(|(&k, &v)| (k, v)).collect())
            .unwrap_or_default()
    }
}

impl LtxQueryView for OwnedLtxDocument {
    fn has_section(&self, section: &str) -> bool {
        self.sections.contains_key(section)
    }

    fn get_str(&self, section: &str, key: &str) -> Option<&str> {
        self.read_str(section, key)
    }

    fn section_keys(&self, section: &str) -> Vec<(&str, &str)> {
        self.sections
            .get(section)
            .map(|s| {
                s.entries
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

fn parse_f32_3_str(s: &str) -> Option<[f32; 3]> {
    let mut parts = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty());
    let x = parts.next()?.parse::<f32>().ok()?;
    let y = parts.next()?.parse::<f32>().ok()?;
    let z = parts.next()?.parse::<f32>().ok()?;
    Some([x, y, z])
}

fn parse_f32_2_str(s: &str) -> Option<[f32; 2]> {
    let mut parts = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty());
    let x = parts.next()?.parse::<f32>().ok()?;
    let y = parts.next()?.parse::<f32>().ok()?;
    Some([x, y])
}

fn rank_candidates_impl<V: LtxQueryView>(
    view: &V,
    section: &str,
    screen_height: u32,
) -> Result<Vec<FontRungCandidate>, FormatError> {
    if !view.has_section(section) {
        return Err(FormatError::InvalidValue("missing font config section"));
    }

    let h = screen_height.max(1);
    let font_prefix = view.get_str("string_table", "font_prefix");

    struct RawCand {
        key: String,
        texture: String,
        authored: u32,
        explicit_rung: bool,
        dist: f32,
    }

    let mut raw = Vec::new();
    for (k, v) in view.section_keys(section) {
        if let Some(authored) = parse_rung_height(k)
            && !v.trim().is_empty()
        {
            raw.push(RawCand {
                key: k.to_string(),
                texture: v.trim().to_string(),
                authored,
                explicit_rung: true,
                dist: (authored as f32 / h as f32).ln().abs(),
            });
        }
    }

    // Legacy rung resolution (`legacy_winner_idx`, `GameFont.cpp:78-96`).
    let mut idx = LEGACY_DEFAULT_IDX;
    for (i, r) in LEGACY_RUNGS.iter().enumerate() {
        if h >= r.lo && (r.hi == 0 || h < r.hi) {
            idx = i;
            break;
        }
    }
    loop {
        if let Some(tex) = view.get_str(section, LEGACY_RUNGS[idx].key)
            && !tex.trim().is_empty()
        {
            let rung = LEGACY_RUNGS[idx];
            raw.push(RawCand {
                key: rung.key.to_string(),
                texture: tex.trim().to_string(),
                authored: rung.authored,
                explicit_rung: false,
                dist: (rung.authored as f32 / h as f32).ln().abs(),
            });
            break;
        }
        if idx == 0 {
            break;
        }
        idx -= 1;
    }

    raw.sort_by(|a, b| {
        if (a.dist - b.dist).abs() > 1.0e-5 {
            a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal)
        } else if a.explicit_rung != b.explicit_rung {
            b.explicit_rung.cmp(&a.explicit_rung)
        } else {
            b.authored.cmp(&a.authored)
        }
    });

    Ok(raw
        .into_iter()
        .map(|c| {
            let resolved_texture = resolve_font_texture_name(&c.texture, font_prefix);
            let atlas_ini_vpath = font_atlas_ini_vpath(&resolved_texture);
            FontRungCandidate {
                key: c.key,
                texture: c.texture,
                resolved_texture,
                atlas_ini_vpath,
                authored_height: c.authored,
                log_distance: c.dist,
                explicit_rung: c.explicit_rung,
            }
        })
        .collect())
}

fn resolve_font_section_impl<V: LtxQueryView, F: FnMut(&str) -> bool>(
    view: &V,
    section: &str,
    screen_height: u32,
    mut ini_exists: F,
) -> Result<FontSectionConfig, FormatError> {
    let cands = rank_candidates_impl(view, section, screen_height)?;
    let font_prefix = view.get_str("string_table", "font_prefix");
    let shader = view
        .get_str(section, "shader")
        .unwrap_or("font")
        .trim()
        .to_string();
    let size = view
        .get_str(section, "size")
        .and_then(|s| s.trim().parse::<f32>().ok());
    let interval = view
        .get_str(section, "interval")
        .and_then(parse_f32_2_str)
        .unwrap_or([1.0, 1.0]);

    for cand in &cands {
        if ini_exists(&cand.atlas_ini_vpath) {
            return Ok(FontSectionConfig {
                section: section.to_string(),
                shader,
                texture: cand.texture.clone(),
                resolved_texture: cand.resolved_texture.clone(),
                atlas_ini_vpath: cand.atlas_ini_vpath.clone(),
                texture_dds_vpath: texture_stem_to_dds_vpath(&cand.resolved_texture),
                size,
                interval,
            });
        }
    }

    if let Some(default_tex) = view.get_str(section, LEGACY_RUNGS[LEGACY_DEFAULT_IDX].key) {
        let tex = default_tex.trim().to_string();
        let resolved_texture = resolve_font_texture_name(&tex, font_prefix);
        let atlas_ini_vpath = font_atlas_ini_vpath(&resolved_texture);
        return Ok(FontSectionConfig {
            section: section.to_string(),
            shader,
            texture: tex,
            texture_dds_vpath: texture_stem_to_dds_vpath(&resolved_texture),
            resolved_texture,
            atlas_ini_vpath,
            size,
            interval,
        });
    }

    if let Some(first) = cands.into_iter().next() {
        return Ok(FontSectionConfig {
            section: section.to_string(),
            shader,
            texture: first.texture,
            texture_dds_vpath: texture_stem_to_dds_vpath(&first.resolved_texture),
            resolved_texture: first.resolved_texture,
            atlas_ini_vpath: first.atlas_ini_vpath,
            size,
            interval,
        });
    }

    Err(FormatError::InvalidValue("missing font texture rung"))
}

/// Rank candidate font atlas textures for `[section]` at `screen_height` (`GetFontTextureName`).
pub fn rank_font_candidates(
    doc: &LtxDocument<'_>,
    section: &str,
    screen_height: u32,
) -> Result<Vec<FontRungCandidate>, FormatError> {
    rank_candidates_impl(doc, section, screen_height)
}

/// Rank candidate font atlas textures from a merged `OwnedLtxDocument`.
pub fn rank_font_candidates_owned(
    doc: &OwnedLtxDocument,
    section: &str,
    screen_height: u32,
) -> Result<Vec<FontRungCandidate>, FormatError> {
    rank_candidates_impl(doc, section, screen_height)
}

/// Resolve `[section]` font texture + `.ini` / `.dds` paths using `ini_exists` callback.
pub fn resolve_font_section<F: FnMut(&str) -> bool>(
    doc: &LtxDocument<'_>,
    section: &str,
    screen_height: u32,
    ini_exists: F,
) -> Result<FontSectionConfig, FormatError> {
    resolve_font_section_impl(doc, section, screen_height, ini_exists)
}

/// Resolve `[section]` font config from an `OwnedLtxDocument` using `ini_exists` callback.
pub fn resolve_font_section_owned<F: FnMut(&str) -> bool>(
    doc: &OwnedLtxDocument,
    section: &str,
    screen_height: u32,
    ini_exists: F,
) -> Result<FontSectionConfig, FormatError> {
    resolve_font_section_impl(doc, section, screen_height, ini_exists)
}

impl FontAtlas {
    /// Parse a font `.ini` text document (`[mb_symbol_coords]`, `[symbol_coords]`, `[char widths]`, `[font_size]`).
    pub fn parse_ini_str(ini_text: &str) -> Result<Self, FormatError> {
        let doc = LtxDocument::parse(ini_text);
        Self::from_ltx(&doc)
    }

    /// Build from a parsed `LtxDocument`.
    pub fn from_ltx(doc: &LtxDocument<'_>) -> Result<Self, FormatError> {
        Self::from_view(doc)
    }

    /// Build from an `OwnedLtxDocument`.
    pub fn from_owned_ltx(doc: &OwnedLtxDocument) -> Result<Self, FormatError> {
        Self::from_view(doc)
    }

    fn from_view<V: LtxQueryView>(view: &V) -> Result<Self, FormatError> {
        if view.has_section("mb_symbol_coords") {
            let line_height = view
                .get_str("mb_symbol_coords", "height")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .ok_or(FormatError::InvalidValue("mb_symbol_coords missing height"))?;
            let x_step = (line_height / 2.0).ceil();

            let mut entries = HashMap::new();
            let mut lowest_present: Option<(u16, FontGlyphCoord)> = None;
            for (k, v) in view.section_keys("mb_symbol_coords") {
                if k.len() == 5
                    && k.bytes().all(|b| b.is_ascii_digit())
                    && let Ok(code) = k.parse::<u16>()
                    && let Some(coords) = parse_f32_3_str(v)
                {
                    let glyph = FontGlyphCoord {
                        x: coords[0],
                        y: coords[1],
                        width: 1.0 + coords[2] - coords[0],
                    };
                    entries.insert(code, glyph);
                    if lowest_present.is_none_or(|(best_c, _)| code < best_c) {
                        lowest_present = Some((code, glyph));
                    }
                }
            }

            let default_glyph = entries
                .get(&9608)
                .copied()
                .or_else(|| lowest_present.map(|(_, g)| g))
                .unwrap_or_default();

            entries.insert(0x0020, FontGlyphCoord::default());
            entries.insert(0x3000, FontGlyphCoord::default());

            return Ok(Self {
                kind: FontAtlasKind::MbSymbolCoords,
                is_multibyte: true,
                line_height,
                x_step,
                y_step: 0.0,
                default_glyph,
                sb_glyphs: Vec::new(),
                mb_glyphs: entries,
            });
        }

        if view.has_section("symbol_coords") {
            let line_height = view
                .get_str("symbol_coords", "height")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .ok_or(FormatError::InvalidValue("symbol_coords missing height"))?;
            let mut sb_glyphs = vec![FontGlyphCoord::default(); 256];
            for i in 0..256u16 {
                let key = format!("{i:03}");
                if let Some(val) = view.get_str("symbol_coords", &key)
                    && let Some(v) = parse_f32_3_str(val)
                {
                    sb_glyphs[usize::from(i)] = FontGlyphCoord {
                        x: v[0],
                        y: v[1],
                        width: v[2] - v[0],
                    };
                }
            }
            return Ok(Self {
                kind: FontAtlasKind::SymbolCoords,
                is_multibyte: false,
                line_height,
                x_step: 0.0,
                y_step: 0.0,
                default_glyph: FontGlyphCoord::default(),
                sb_glyphs,
                mb_glyphs: HashMap::new(),
            });
        }

        if view.has_section("char widths") {
            let line_height = view
                .get_str("char widths", "height")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .ok_or(FormatError::InvalidValue("char widths missing height"))?;
            let cpl = 16usize;
            let mut sb_glyphs = vec![FontGlyphCoord::default(); 256];
            for (i, slot) in sb_glyphs.iter_mut().enumerate() {
                let key = i.to_string();
                if let Some(w) = view
                    .get_str("char widths", &key)
                    .and_then(|s| s.trim().parse::<f32>().ok())
                {
                    *slot = FontGlyphCoord {
                        x: (i % cpl) as f32 * line_height,
                        y: (i / cpl) as f32 * line_height,
                        width: w,
                    };
                }
            }
            return Ok(Self {
                kind: FontAtlasKind::CharWidths,
                is_multibyte: false,
                line_height,
                x_step: 0.0,
                y_step: 0.0,
                default_glyph: FontGlyphCoord::default(),
                sb_glyphs,
                mb_glyphs: HashMap::new(),
            });
        }

        if view.has_section("font_size") {
            let line_height = view
                .get_str("font_size", "height")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .ok_or(FormatError::InvalidValue("font_size missing height"))?;
            let width = view
                .get_str("font_size", "width")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .ok_or(FormatError::InvalidValue("font_size missing width"))?;
            let cpl = view
                .get_str("font_size", "cpl")
                .and_then(|s| s.trim().parse::<usize>().ok())
                .filter(|&c| c > 0)
                .ok_or(FormatError::InvalidValue("font_size missing/invalid cpl"))?;

            let mut sb_glyphs = vec![FontGlyphCoord::default(); 256];
            for (i, slot) in sb_glyphs.iter_mut().enumerate() {
                *slot = FontGlyphCoord {
                    x: (i % cpl) as f32 * width,
                    y: (i / cpl) as f32 * line_height,
                    width,
                };
            }
            return Ok(Self {
                kind: FontAtlasKind::FontSizeGrid,
                is_multibyte: false,
                line_height,
                x_step: 0.0,
                y_step: 0.0,
                default_glyph: FontGlyphCoord::default(),
                sb_glyphs,
                mb_glyphs: HashMap::new(),
            });
        }

        Err(FormatError::InvalidValue(
            "missing font atlas section (mb_symbol_coords/symbol_coords/char widths/font_size)",
        ))
    }

    /// Look up pixel glyph coordinates (`TCMap[c]`).
    #[inline]
    pub fn glyph(&self, code: u16) -> FontGlyphCoord {
        if self.is_multibyte {
            self.mb_glyphs
                .get(&code)
                .copied()
                .unwrap_or(self.default_glyph)
        } else {
            self.sb_glyphs
                .get(usize::from(code))
                .copied()
                .unwrap_or(self.default_glyph)
        }
    }

    /// Character count (`CGameFont::smart_strlen`).
    pub fn smart_strlen(&self, text: &str) -> usize {
        if self.is_multibyte {
            mbh_multi2wide(text).0.len()
        } else {
            text.len()
        }
    }

    /// Measure rendered pixel width (`CGameFont::SizeOf_`, `GameFont.cpp:475-516`).
    pub fn measure_width(&self, text: &str, current_height: f32, interval_x: f32) -> f32 {
        if text.is_empty() {
            return 0.0;
        }
        let width_scale = if self.line_height > 0.0 {
            current_height / self.line_height
        } else {
            1.0
        };

        let mut total = 0.0f32;
        if self.is_multibyte {
            let (wides, _) = mbh_multi2wide(text);
            for wc in wides {
                let mut delta = self.glyph(wc).width - 2.0;
                if is_need_space_character(wc) {
                    delta += self.x_step;
                }
                total += delta;
            }
        } else {
            for &b in text.as_bytes() {
                total += self.glyph(u16::from(b)).width;
            }
        }

        total * width_scale * interval_x
    }

    /// Find byte offset where `text` exceeds `target_width` (`CGameFont::GetCutLengthPos`).
    pub fn cut_length_pos(&self, target_width: f32, text: &str) -> usize {
        let (wides, positions) = mbh_multi2wide(text);
        if wides.is_empty() {
            return 0;
        }

        let mut cur_width = 0.0f32;
        let mut idx = 0usize;
        while idx < wides.len() {
            let wc = wides[idx];
            let mut delta = self.glyph(wc).width - 2.0;
            if is_need_space_character(wc) {
                delta += self.x_step;
            }
            if cur_width + delta > target_width {
                break;
            }
            cur_width += delta;
            idx += 1;
        }

        positions[idx]
    }

    /// Compute word/line break byte indices within `text` for width `target_width`
    /// (`CGameFont::SplitByWidth`, `GameFont.cpp:374-408`).
    pub fn split_by_width(&self, target_width: f32, text: &str) -> Vec<usize> {
        let (wides, positions) = mbh_multi2wide(text);
        if wides.is_empty() {
            return Vec::new();
        }

        let mut splits = Vec::new();
        let mut cur_width = 0.0f32;

        for i in 0..wides.len() {
            let wc = wides[i];
            let mut delta = self.glyph(wc).width - 2.0;
            if is_need_space_character(wc) {
                delta += self.x_step;
            }

            if cur_width + delta > target_width
                && !is_bad_start_character(wc)
                && (i + 1) < wides.len()
                && (i > 0 && !is_bad_end_character(wides[i - 1]))
            {
                cur_width = delta;
                splits.push(positions[i]);
            } else {
                cur_width += delta;
            }
        }

        splits
    }

    /// Emit screen-space UV glyph quads for `text` (`dxFontRender::OnRender`, `dxFontRender.cpp:73-155`).
    pub fn build_string_quads(
        &self,
        atlas_size: [u32; 2],
        pos_xy: [f32; 2],
        text: &str,
        params: &FontDrawParams,
    ) -> Vec<FontGlyphQuad> {
        if text.is_empty() || atlas_size[0] == 0 || atlas_size[1] == 0 {
            return Vec::new();
        }

        let ts_x = atlas_size[0] as f32;
        let ts_y = atlas_size[1] as f32;
        let tc_height = self.line_height / ts_y;

        let mut x = pos_xy[0].floor();
        let mut y = pos_xy[1].floor();
        let scaled_h = params.current_height * params.scale[1];
        let mut y2 = y + scaled_h;
        let hs = if self.line_height.abs() <= 1.0e-5 {
            1.0
        } else {
            params.current_height / self.line_height
        };

        if params.align != FontAlignment::Left {
            let align_width = self.measure_width(text, params.current_height, params.interval[0]);
            match params.align {
                FontAlignment::Center => {
                    x -= (align_width * 0.5).floor() * params.scale[0];
                }
                FontAlignment::Right => {
                    x -= align_width.floor();
                }
                FontAlignment::Left => {}
            }
        }

        let clr_top = params.color_argb;
        let clr_bottom = if params.gradient {
            let a = (clr_top >> 24) & 0xFF;
            let r = ((clr_top >> 16) & 0xFF) / 2;
            let g = ((clr_top >> 8) & 0xFF) / 2;
            let b = (clr_top & 0xFF) / 2;
            color_argb(a as u8, r as u8, g as u8, b as u8)
        } else {
            clr_top
        };

        if params.dx_half_pixel_offset {
            x -= 0.5;
            y -= 0.5;
            y2 -= 0.5;
        }

        let mut quads = Vec::with_capacity(text.len());
        let (codes, _positions) = if self.is_multibyte {
            mbh_multi2wide(text)
        } else {
            let bytes = text.as_bytes();
            let codes: Vec<u16> = bytes.iter().map(|&b| u16::from(b)).collect();
            (codes, Vec::new())
        };

        for code in codes {
            let g = self.glyph(code);
            let scw = g.width * hs * params.scale[0];
            if g.width.abs() > 1.0e-5 {
                let tu = g.x / ts_x;
                let tv = g.y / ts_y;
                let f_tc_width = g.width / ts_x;
                quads.push(FontGlyphQuad {
                    code,
                    min_xy: [x, y],
                    max_xy: [x + scw, y2],
                    uv_min: [tu, tv],
                    uv_max: [tu + f_tc_width, tv + tc_height],
                    color_top_argb: clr_top,
                    color_bottom_argb: clr_bottom,
                });
            }

            x += scw * params.interval[0];
            if self.is_multibyte {
                x -= 2.0;
                if is_need_space_character(code) {
                    x += self.x_step;
                }
            }
        }

        quads
    }

    /// Screen-space horizontal cursor advance matching `build_string_quads` for one code unit.
    pub fn glyph_screen_advance(&self, code: u16, params: &FontDrawParams) -> f32 {
        let hs = if self.line_height.abs() <= 1.0e-5 {
            1.0
        } else {
            params.current_height / self.line_height
        };
        let scw = self.glyph(code).width * hs * params.scale[0];
        let mut adv = scw * params.interval[0];
        if self.is_multibyte {
            adv -= 2.0;
            if is_need_space_character(code) {
                adv += self.x_step;
            }
        }
        adv
    }

    /// Screen-space horizontal width of a text run matching `build_string_quads` cursor advance.
    pub fn measure_screen_width(&self, text: &str, params: &FontDrawParams) -> f32 {
        if text.is_empty() {
            return 0.0;
        }
        if self.is_multibyte {
            let (codes, _) = mbh_multi2wide(text);
            codes
                .into_iter()
                .map(|c| self.glyph_screen_advance(c, params))
                .sum()
        } else {
            text.as_bytes()
                .iter()
                .map(|&b| self.glyph_screen_advance(u16::from(b), params))
                .sum()
        }
    }

    /// Word-wrap logical lines of colored segments to fit within `max_width_px` (`CUILines::ParseText`,
    /// `UILines.cpp:192-310`).
    pub fn wrap_colored_lines(
        &self,
        lines: &[Vec<UiColoredSegment>],
        max_width_px: f32,
        params: &FontDrawParams,
        word_wrap: bool,
    ) -> Vec<Vec<UiColoredSegment>> {
        if !word_wrap || max_width_px <= 0.0 {
            return lines.to_vec();
        }

        let mut wrapped: Vec<Vec<UiColoredSegment>> = Vec::with_capacity(lines.len());
        if self.is_multibyte {
            let hs = if self.line_height.abs() <= 1.0e-5 {
                1.0
            } else {
                params.current_height / self.line_height
            };
            let px_scale = (hs * params.scale[0] * params.interval[0]).max(1.0e-5);
            let target_unscaled = max_width_px / px_scale;
            for line in lines {
                if line.is_empty() {
                    wrapped.push(Vec::new());
                    continue;
                }
                let mut cur_line: Vec<UiColoredSegment> = Vec::new();
                for seg in line {
                    let splits = self.split_by_width(target_unscaled, &seg.text);
                    if splits.is_empty() {
                        if !seg.text.is_empty() {
                            cur_line.push(seg.clone());
                        }
                        continue;
                    }
                    let mut start = 0usize;
                    for split in splits {
                        if split > start {
                            cur_line.push(UiColoredSegment {
                                text: seg.text[start..split].to_owned(),
                                color_argb: seg.color_argb,
                            });
                        }
                        wrapped.push(std::mem::take(&mut cur_line));
                        start = split;
                    }
                    if start < seg.text.len() {
                        cur_line.push(UiColoredSegment {
                            text: seg.text[start..].to_owned(),
                            color_argb: seg.color_argb,
                        });
                    }
                }
                wrapped.push(cur_line);
            }
            return wrapped;
        }

        // Single-byte word wrapping (`UILines.cpp:253-308`): wrap on last whitespace when exceeding `max_width_px`.
        let eps = self.glyph_screen_advance(u16::from(b'o'), params);
        for line in lines {
            if line.is_empty() {
                wrapped.push(Vec::new());
                continue;
            }
            let mut cur_line: Vec<UiColoredSegment> = Vec::new();
            let mut curr_width = 0.0f32;
            for seg in line {
                let bytes = seg.text.as_bytes();
                if bytes.is_empty() {
                    continue;
                }
                let mut start = 0usize;
                let mut last_space: Option<usize> = None;
                let mut i = 0usize;
                while i < bytes.len() {
                    let b = bytes[i];
                    if b.is_ascii_whitespace() {
                        last_space = Some(i);
                    }
                    let ch_w = self.glyph_screen_advance(u16::from(b), params);
                    let over = (curr_width + ch_w + eps) > max_width_px;
                    let is_last = i + 1 == bytes.len();
                    if over {
                        let break_end = if let Some(sp) = last_space.filter(|&sp| sp >= start) {
                            last_space = None;
                            sp + 1
                        } else if i > start || !cur_line.is_empty() {
                            i
                        } else {
                            i + 1
                        };
                        if break_end > start {
                            cur_line.push(UiColoredSegment {
                                text: String::from_utf8_lossy(&bytes[start..break_end]).into_owned(),
                                color_argb: seg.color_argb,
                            });
                        }
                        wrapped.push(std::mem::take(&mut cur_line));
                        curr_width = 0.0;
                        start = break_end;
                        i = break_end;
                    } else if is_last {
                        if i + 1 > start {
                            cur_line.push(UiColoredSegment {
                                text: String::from_utf8_lossy(&bytes[start..]).into_owned(),
                                color_argb: seg.color_argb,
                            });
                        }
                        curr_width += ch_w;
                        i += 1;
                    } else {
                        curr_width += ch_w;
                        i += 1;
                    }
                }
            }
            wrapped.push(cur_line);
        }
        wrapped
    }

    /// Truncate single-line text with `".."` suffix when wider than `max_width_px`
    /// (`GetElipsisText`, `UILines.cpp:349-383`).
    pub fn elide_line(&self, text: &str, max_width_px: f32, params: &FontDrawParams) -> String {
        let total = self.measure_screen_width(text, params);
        if total < max_width_px || max_width_px <= 0.0 {
            return text.to_owned();
        }
        let el_len = self.measure_screen_width("..", params);
        let mut acc = 0.0f32;
        let mut cut = 0usize;
        for (byte_idx, ch) in text.char_indices() {
            let w = if self.is_multibyte {
                let mut buf = [0u16; 2];
                let enc = ch.encode_utf16(&mut buf);
                enc.iter()
                    .map(|&c| self.glyph_screen_advance(c, params))
                    .sum::<f32>()
            } else {
                self.glyph_screen_advance((ch as u32 & 0xFF) as u16, params)
            };
            if acc + w + el_len < max_width_px {
                acc += w;
                cut = byte_idx + ch.len_utf8();
            } else {
                break;
            }
        }
        let mut out = text[..cut].to_owned();
        out.push_str("..");
        out
    }

    /// Build glyph quads for a bounded UI text box supporting `%c[...]` colors, multi-line word wrapping,
    /// `complex_mode`, `ellipsis`, `password_mode`, and horizontal + vertical alignments (`CUILines::Draw`).
    pub fn build_multiline_quads(
        &self,
        atlas_size: [u32; 2],
        bounds_xywh: [f32; 4],
        text: &str,
        color_defs: Option<&UiColorDefs>,
        options: &UiMultilineTextOptions,
    ) -> Vec<FontGlyphQuad> {
        if text.is_empty() || atlas_size[0] == 0 || atlas_size[1] == 0 {
            return Vec::new();
        }

        let base_x = bounds_xywh[0] + options.offset[0];
        let base_y = bounds_xywh[1] + options.offset[1];
        let wnd_w = bounds_xywh[2];
        let wnd_h = bounds_xywh[3];
        let line_step_y = options.draw.current_height * options.draw.scale[1] * options.draw.interval[1];

        if !options.complex_mode {
            let masked;
            let effective_text = if options.password_mode {
                masked = "*".repeat(text.chars().count());
                masked.as_str()
            } else if options.ellipsis && wnd_w > 0.0 {
                masked = self.elide_line(text, wnd_w, &options.draw);
                masked.as_str()
            } else {
                text
            };

            let lines = parse_colored_text_lines(
                effective_text,
                options.draw.color_argb,
                color_defs,
                options.coloring_mode && !options.password_mode,
                false,
            );
            let line = lines.into_iter().next().unwrap_or_default();
            let line_w: f32 = line
                .iter()
                .map(|seg| self.measure_screen_width(&seg.text, &options.draw))
                .sum();
            let start_x = match options.draw.align {
                FontAlignment::Left => base_x,
                FontAlignment::Center => base_x + (wnd_w - line_w) * 0.5,
                FontAlignment::Right => base_x + (wnd_w - line_w),
            };
            let start_y = match options.vert_align {
                UiTextVerticalAlign::Top => base_y,
                UiTextVerticalAlign::Center => base_y + (wnd_h - line_step_y) * 0.5,
                UiTextVerticalAlign::Bottom => base_y + (wnd_h - line_step_y),
            };

            let mut out = Vec::new();
            let mut cursor_x = start_x;
            for seg in line {
                let mut seg_params = options.draw;
                seg_params.align = FontAlignment::Left;
                seg_params.color_argb = seg.color_argb;
                out.extend(self.build_string_quads(
                    atlas_size,
                    [cursor_x, start_y],
                    &seg.text,
                    &seg_params,
                ));
                cursor_x += self.measure_screen_width(&seg.text, &seg_params);
            }
            return out;
        }

        let logical = parse_colored_text_lines(
            text,
            options.draw.color_argb,
            color_defs,
            options.coloring_mode,
            options.recognize_newline,
        );
        let wrapped = self.wrap_colored_lines(&logical, wnd_w, &options.draw, options.word_wrap);
        let vis_height = line_step_y * (wrapped.len() as f32);
        let mut cursor_y = match options.vert_align {
            UiTextVerticalAlign::Top => base_y,
            UiTextVerticalAlign::Center => base_y + (wnd_h - vis_height) * 0.5,
            UiTextVerticalAlign::Bottom => base_y + (wnd_h - vis_height),
        };

        let mut out = Vec::new();
        for line in wrapped {
            let line_w: f32 = line
                .iter()
                .map(|seg| self.measure_screen_width(&seg.text, &options.draw))
                .sum();
            let mut cursor_x = match options.draw.align {
                FontAlignment::Left => base_x,
                FontAlignment::Center => base_x + (wnd_w - line_w) * 0.5,
                FontAlignment::Right => base_x + (wnd_w - line_w),
            };
            for seg in line {
                let mut seg_params = options.draw;
                seg_params.align = FontAlignment::Left;
                seg_params.color_argb = seg.color_argb;
                out.extend(self.build_string_quads(
                    atlas_size,
                    [cursor_x, cursor_y],
                    &seg.text,
                    &seg_params,
                ));
                cursor_x += self.measure_screen_width(&seg.text, &seg_params);
            }
            cursor_y += line_step_y;
        }
        out
    }

    /// Layout text quads for a parsed `UiWindowDef` + optional `UiTextDef` scaled through `UiViewport`.
    pub fn build_ui_window_text_quads(
        &self,
        atlas_size: [u32; 2],
        window: &UiWindowDef,
        viewport: &UiViewport,
        color_defs: Option<&UiColorDefs>,
        translated_text: Option<&str>,
        base_draw: FontDrawParams,
    ) -> Vec<FontGlyphQuad> {
        let Some(text_def) = window.text.as_ref() else {
            return Vec::new();
        };
        self.build_ui_text_def_quads(
            atlas_size,
            [window.x, window.y, window.width, window.height],
            text_def,
            viewport,
            color_defs,
            translated_text,
            base_draw,
        )
    }

    /// Layout text quads from a `UiTextDef` and virtual `(1024x768)` bounds `[x, y, width, height]`.
    #[allow(clippy::too_many_arguments)]
    pub fn build_ui_text_def_quads(
        &self,
        atlas_size: [u32; 2],
        virtual_bounds_xywh: [f32; 4],
        text_def: &UiTextDef,
        viewport: &UiViewport,
        color_defs: Option<&UiColorDefs>,
        translated_text: Option<&str>,
        mut base_draw: FontDrawParams,
    ) -> Vec<FontGlyphQuad> {
        let text = translated_text.unwrap_or(&text_def.text);
        if text.is_empty() {
            return Vec::new();
        }
        let scale = viewport.scale();
        let screen_bounds = viewport.virtual_to_screen_rect(virtual_bounds_xywh);
        base_draw.align = FontAlignment::parse_xml_align(&text_def.align);
        base_draw.color_argb = text_def.color_argb;
        base_draw.scale = [base_draw.scale[0] * scale[0], base_draw.scale[1] * scale[1]];

        let opts = UiMultilineTextOptions {
            draw: base_draw,
            vert_align: UiTextVerticalAlign::parse_xml_align(&text_def.vert_align),
            complex_mode: text_def.complex_mode,
            offset: [text_def.offset[0] * scale[0], text_def.offset[1] * scale[1]],
            ..UiMultilineTextOptions::default()
        };
        self.build_multiline_quads(atlas_size, screen_bounds, text, color_defs, &opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn font_rung_resolution_and_language_prefix() {
        let ltx = r#"
[string_table]
font_prefix = _west

[ui_font_letterica18_russian]
shader = font
texture800 = ui\ui_font_letter_18_800
texture = ui\ui_font_letter_18_1024
texture1600 = ui\ui_font_letter_18_1600
texture_h1440 = ui\ui_font_letter_18_1440
size = 18
interval = 0.75, 1.0

[hud_font_di]
shader = font
texture = ui\ui_font_hud_01
"#;
        let doc = LtxDocument::parse(ltx);

        // At 1440p, explicit texture_h1440 ranks #1.
        let cands = rank_font_candidates(&doc, "ui_font_letterica18_russian", 1440).unwrap();
        assert_eq!(cands[0].key, "texture_h1440");
        assert_eq!(cands[0].resolved_texture, "ui\\ui_font_letter_18_1440_west");
        assert_eq!(
            cands[0].atlas_ini_vpath,
            "textures\\ui\\ui_font_letter_18_1440_west.ini"
        );

        // Fallback when 1440 ini is missing picks texture1600 (closer to 1440 than 768).
        let cfg = resolve_font_section(&doc, "ui_font_letterica18_russian", 1440, |p| {
            p.ends_with("1600_west.ini")
        })
        .unwrap();
        assert_eq!(cfg.resolved_texture, "ui\\ui_font_letter_18_1600_west");
        assert_eq!(
            cfg.texture_dds_vpath,
            "textures\\ui\\ui_font_letter_18_1600_west.dds"
        );
        assert_eq!(cfg.size, Some(18.0));
        assert_eq!(cfg.interval, [0.75, 1.0]);

        // DI HUD fonts exempt from font_prefix suffix.
        let di = resolve_font_section(&doc, "hud_font_di", 1080, |_| true).unwrap();
        assert_eq!(di.resolved_texture, "ui\\ui_font_hud_01");
    }

    #[test]
    fn parse_all_four_font_atlas_formats_and_emit_quads() {
        // 1. symbol_coords + dxFontRender quad layout & gradient half-color
        let sym_ini = r#"
[symbol_coords]
height = 20
065 = 10, 20, 22
066 = 30, 20, 44
"#;
        let sym_atlas = FontAtlas::parse_ini_str(sym_ini).unwrap();
        assert_eq!(sym_atlas.kind, FontAtlasKind::SymbolCoords);
        assert!(!sym_atlas.is_multibyte);
        assert_eq!(
            sym_atlas.glyph(b'A' as u16),
            FontGlyphCoord {
                x: 10.0,
                y: 20.0,
                width: 12.0,
            }
        );
        assert!((sym_atlas.measure_width("AB", 20.0, 1.0) - 26.0).abs() < 1.0e-5);

        let quads = sym_atlas.build_string_quads(
            [512, 256],
            [100.0, 40.0],
            "AB",
            &FontDrawParams {
                current_height: 20.0,
                interval: [1.0, 1.0],
                scale: [1.0, 1.0],
                align: FontAlignment::Right,
                color_argb: 0xFF80_4020,
                gradient: true,
                dx_half_pixel_offset: true,
            },
        );
        assert_eq!(quads.len(), 2);
        // Right align shifts left by 26 -> 74.0, then DX10 -0.5 -> 73.5
        assert!((quads[0].min_xy[0] - 73.5).abs() < 1.0e-5);
        assert!((quads[0].uv_min[0] - 10.0 / 512.0).abs() < 1.0e-5);
        assert_eq!(quads[0].color_top_argb, 0xFF80_4020);
        assert_eq!(quads[0].color_bottom_argb, 0xFF40_2010);

        // 2. mb_symbol_coords + space/ideographic-space zeroing + split_by_width
        let mb_ini = r#"
[mb_symbol_coords]
height = 17
09608 = 0, 0, 7
00065 = 10, 0, 21
00066 = 24, 0, 35
01046 = 40, 0, 51
"#;
        let mb_atlas = FontAtlas::parse_ini_str(mb_ini).unwrap();
        assert_eq!(mb_atlas.kind, FontAtlasKind::MbSymbolCoords);
        assert!(mb_atlas.is_multibyte);
        assert_eq!(mb_atlas.x_step, 9.0);
        assert_eq!(mb_atlas.glyph(0x0020).width, 0.0);
        assert_eq!(mb_atlas.glyph(0x3000).width, 0.0);
        assert_eq!(mb_atlas.glyph(b'A' as u16).width, 12.0); // 1 + 21 - 10
        assert_eq!(mb_atlas.glyph(0x0416).width, 12.0); // Cyrillic Ж

        // `GameFont.cpp:394`: `(i < len)` never breaks right before the final character.
        assert!(mb_atlas.split_by_width(28.0, "\u{0416} A B").is_empty());

        // "Ж A BC": Ж=12-2=10, ' '=0-2+9=7, A=10, ' '=7, B=10 -> wraps before B at target=28
        let text = "\u{0416} A BC";
        assert_eq!(mb_atlas.smart_strlen(text), 6);
        let splits = mb_atlas.split_by_width(28.0, text);
        assert_eq!(splits, vec![text.len() - 2]);

        // 3. char widths
        let cw_ini = "[char widths]\nheight = 16\n65 = 11.5\n";
        let cw_atlas = FontAtlas::parse_ini_str(cw_ini).unwrap();
        assert_eq!(cw_atlas.kind, FontAtlasKind::CharWidths);
        assert_eq!(
            cw_atlas.glyph(65),
            FontGlyphCoord {
                x: 16.0,
                y: 64.0,
                width: 11.5,
            }
        );

        // 4. font_size grid
        let fs_ini = "[font_size]\nheight = 14\nwidth = 8\ncpl = 16\n";
        let fs_atlas = FontAtlas::parse_ini_str(fs_ini).unwrap();
        assert_eq!(fs_atlas.kind, FontAtlasKind::FontSizeGrid);
        assert_eq!(
            fs_atlas.glyph(17),
            FontGlyphCoord {
                x: 8.0,
                y: 14.0,
                width: 8.0,
            }
        );
    }

    #[test]
    fn mbh_multi2wide_utf8_and_dumb_cp1251_fallback() {
        let utf8 = "A\u{0416}z";
        let (w, pos) = mbh_multi2wide(utf8);
        assert_eq!(w, vec![b'A' as u16, 0x0416, b'z' as u16]);
        assert_eq!(pos, vec![0, 1, 3, 4]);

        // Raw CP1251/invalid UTF-8 bytes fall back to 1-to-1 byte widening (`MB_DUMB_CONVERSION`).
        let (w_dumb, pos_dumb) = mbh_multi2wide_dumb(b"A\xC6");
        assert_eq!(w_dumb, vec![b'A' as u16, 0x00C6]);
        assert_eq!(pos_dumb, vec![0, 1, 2]);
    }

    #[test]
    fn glyph_quad_adapters_colored_multiline_wrap_ellipsis_and_ui_window() {
        let ini = r#"
[symbol_coords]
height = 20
032 = 0, 0, 6
042 = 6, 0, 14
046 = 14, 0, 19
065 = 20, 0, 32
066 = 34, 0, 46
067 = 48, 0, 60
111 = 60, 0, 70
"#;
        let atlas = FontAtlas::parse_ini_str(ini).unwrap();
        let mut colors = UiColorDefs::default();
        colors
            .colors
            .insert("ui_red".to_string(), color_argb(255, 255, 40, 40));

        // 1. %c[...] inline coloring + \\n explicit breaks + base alpha preservation
        let parsed = parse_colored_text_lines(
            "A %c[ui_red]B%c[default]\\n%c[255,10,20,30]C",
            color_argb(128, 255, 255, 255),
            Some(&colors),
            true,
            true,
        );
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0][0].text, "A ");
        assert_eq!(parsed[0][0].color_argb, color_argb(128, 255, 255, 255));
        assert_eq!(parsed[0][1].text, "B");
        assert_eq!(parsed[0][1].color_argb, color_argb(128, 255, 40, 40));
        assert_eq!(parsed[1][0].text, "C");
        assert_eq!(parsed[1][0].color_argb, color_argb(128, 10, 20, 30));

        // 2. Single-byte word wrapping & center/bottom multi-line quad layout + rect_xywh adapters
        let quads = atlas.build_multiline_quads(
            [256, 128],
            [10.0, 20.0, 42.0, 100.0],
            "AB C",
            Some(&colors),
            &UiMultilineTextOptions {
                draw: FontDrawParams {
                    current_height: 20.0,
                    align: FontAlignment::Center,
                    color_argb: color_argb(255, 200, 100, 50),
                    dx_half_pixel_offset: false,
                    ..FontDrawParams::default()
                },
                vert_align: UiTextVerticalAlign::Bottom,
                complex_mode: true,
                ..UiMultilineTextOptions::default()
            },
        );
        // Line 1: "AB " (A=12,B=12,' '=6 => 30px + 'o'(10px)=40<=42; plus 'C'(12) => 52>42 -> wraps to line 2!)
        assert_eq!(quads.len(), 4);
        let (rect, uv) = quads[0].screen_and_uv_rects();
        assert!((rect[2] - 12.0).abs() < 1e-5);
        assert!((rect[3] - 20.0).abs() < 1e-5);
        assert!((uv[0] - 20.0 / 256.0).abs() < 1e-5);
        assert!((quads[0].min_xy[1] - 80.0).abs() < 1e-5); // Bottom aligned 2 lines (40px) inside 20..120 -> y0=80

        // 3. Ellipsis truncation
        assert_eq!(
            atlas.elide_line("ABCABC", 40.0, &FontDrawParams { current_height: 20.0, ..FontDrawParams::default() }),
            "AB.."
        );

        // 4. UiWindowDef + UiViewport (1920x1080) integration + resolve_ui_font_section
        assert_eq!(resolve_ui_font_section("letterica18"), "ui_font_letterica18_russian");
        let wnd = UiWindowDef {
            tag: "cap".into(),
            id: None,
            window_name: None,
            x: 100.0,
            y: 50.0,
            width: 200.0,
            height: 40.0,
            align: String::new(),
            alignment: String::new(),
            stretch: false,
            hint: None,
            texture: None,
            text: Some(UiTextDef {
                font: Some("letterica18".into()),
                color_argb: color_argb(255, 255, 255, 255),
                align: "l".into(),
                vert_align: "t".into(),
                complex_mode: false,
                offset: [0.0, 0.0],
                text: "AB".into(),
            }),
            attributes: Default::default(),
            children: Vec::new(),
        };
        let vp = UiViewport::new(2048.0, 1536.0);
        let ui_quads = atlas.build_ui_window_text_quads(
            [256, 128],
            &wnd,
            &vp,
            Some(&colors),
            None,
            FontDrawParams {
                current_height: 20.0,
                dx_half_pixel_offset: false,
                ..FontDrawParams::default()
            },
        );
        assert_eq!(ui_quads.len(), 2);
        // Scaled 2x from virtual (100, 50) -> (200, 100)
        assert!((ui_quads[0].min_xy[0] - 200.0).abs() < 1e-5);
        assert!((ui_quads[0].min_xy[1] - 100.0).abs() < 1e-5);
    }
}
