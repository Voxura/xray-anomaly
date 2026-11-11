//! Pure Rust X-Ray `game_levels.ltx`, `game_maps_single.ltx`, and `level.ltx [level_map]` parsers
//!
//! Mirrors `src/xrGame/Level_load.cpp`, `src/xrGame/spawn_packs.cpp`, `src/xrGame/ui/UIMap.cpp`, and
//! `src/xrGame/ui/UIMapWnd.cpp`:
//! - `GameLevelsConfig`: `[levels]` registry, `levelNN` entries (`name`, `caption`, `offset`, `weathers`, `id`),
//!   and descending `254 -> 1` allocation for `id = auto` levels matching `spawn_packs::allocate_auto_ids`
//! - `WorldMapsConfig` / `GlobalMapDef` / `LevelMapDef`: `[global_map]`, `[def_map]`, `[level_maps_single]`,
//!   `[level_maps_mp]`, per-level `texture`, `bound_rect`, `global_rect`, `music_tracks`, `weathers`,
//!   PDA map `label` / `label_scale_max` / `label_color` / `label_offset_x/y`, and `[level_map]` overrides
//! - World `(x, z)` <-> level map pixel / UV / global map coordinate transforms (with inverted engine Z
//!   and `kx` aspect compensation matching `CUICustomMap::ConvertRealToLocal` / `CUIGlobalMap::ConvertRealToLocal`).

use std::collections::{BTreeMap, BTreeSet};

use xray_core::ltx_loader::OwnedLtxDocument;
use xray_core::LtxDocument;

use crate::chunk::FormatError;
use crate::thm::texture_stem_to_dds_vpath;
use crate::xml_ui::color_argb;

/// Default PDA map label color when `label_color` is omitted (`UIMap.cpp:464`: `color_argb(180, 230, 220, 200)`).
pub const DEFAULT_MAP_LABEL_COLOR_ARGB: u32 = color_argb(180, 230, 220, 200);

trait LtxQuery {
    fn has_section(&self, section: &str) -> bool;
    fn read_str(&self, section: &str, key: &str) -> Option<&str>;
    fn section_keys(&self, section: &str) -> Vec<(&str, &str)>;
}

impl<'a> LtxQuery for LtxDocument<'a> {
    fn has_section(&self, section: &str) -> bool {
        self.contains_section(section)
    }

    fn read_str(&self, section: &str, key: &str) -> Option<&str> {
        LtxDocument::read_str(self, section, key)
    }

    fn section_keys(&self, section: &str) -> Vec<(&str, &str)> {
        self.sections
            .get(section)
            .map(|s| s.entries.iter().map(|(&k, &v)| (k, v)).collect())
            .unwrap_or_default()
    }
}

impl LtxQuery for OwnedLtxDocument {
    fn has_section(&self, section: &str) -> bool {
        self.sections.contains_key(section)
    }

    fn read_str(&self, section: &str, key: &str) -> Option<&str> {
        OwnedLtxDocument::read_str(self, section, key)
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

fn parse_f32_3(s: &str) -> Option<[f32; 3]> {
    let mut it = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse::<f32>().ok());
    Some([it.next()?, it.next()?, it.next()?])
}

fn parse_f32_4(s: &str) -> Option<[f32; 4]> {
    let mut it = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse::<f32>().ok());
    Some([it.next()?, it.next()?, it.next()?, it.next()?])
}

/// Parse `r, g, b[, a]` integer color matching `CInifile::r_color`: omitted 4th component defaults to `255`.
pub fn parse_ltx_color_argb(s: &str) -> Option<u32> {
    let mut it = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .filter_map(|p| p.parse::<i32>().ok());
    let r = it.next()?.clamp(0, 255) as u8;
    let g = it.next()?.clamp(0, 255) as u8;
    let b = it.next()?.clamp(0, 255) as u8;
    let a = it.next().unwrap_or(255).clamp(0, 255) as u8;
    Some(color_argb(a, r, g, b))
}

/// Level ID mode stored in `[levelNN] id = ...` (`game_levels.ltx`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelIdSpec {
    Explicit(u32),
    Auto,
}

/// Single level registration parsed from `game_levels.ltx` (`[levels]` -> `[levelNN]`).
#[derive(Debug, Clone, PartialEq)]
pub struct LevelListEntry {
    pub section: String,
    pub name: String,
    pub caption: String,
    pub offset: [f32; 3],
    pub weathers: Option<String>,
    pub id_spec: Option<LevelIdSpec>,
}

/// Parsed `[levels]` table from `game_levels.ltx` (`configs\game_levels.ltx`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GameLevelsConfig {
    pub levels: Vec<LevelListEntry>,
}

impl GameLevelsConfig {
    pub fn parse_ltx_str(ltx: &str) -> Self {
        let doc = LtxDocument::parse(ltx);
        Self::from_ltx(&doc)
    }

    pub fn from_ltx(doc: &LtxDocument<'_>) -> Self {
        Self::from_query(doc)
    }

    pub fn from_owned_ltx(doc: &OwnedLtxDocument) -> Self {
        Self::from_query(doc)
    }

    fn from_query<Q: LtxQuery>(q: &Q) -> Self {
        let mut levels = Vec::new();
        for (sect_key, _) in q.section_keys("levels") {
            let section = sect_key.trim();
            if section.is_empty() || !q.has_section(section) {
                continue;
            }
            let Some(name_raw) = q.read_str(section, "name") else {
                continue;
            };
            let name = name_raw.trim().trim_matches('"').to_string();
            if name.is_empty() {
                continue;
            }
            let caption = q
                .read_str(section, "caption")
                .map(|s| s.trim().trim_matches('"').to_string())
                .unwrap_or_else(|| name.clone());
            let offset = q
                .read_str(section, "offset")
                .and_then(parse_f32_3)
                .unwrap_or([0.0; 3]);
            let weathers = q
                .read_str(section, "weathers")
                .map(|s| s.trim().trim_matches('"').to_string())
                .filter(|s| !s.is_empty());
            let id_spec = q.read_str(section, "id").and_then(|raw| {
                let t = raw.trim().trim_matches('"');
                if t.eq_ignore_ascii_case("auto") {
                    Some(LevelIdSpec::Auto)
                } else {
                    t.parse::<u32>().ok().map(LevelIdSpec::Explicit)
                }
            });

            levels.push(LevelListEntry {
                section: section.to_string(),
                name,
                caption,
                offset,
                weathers,
                id_spec,
            });
        }
        Self { levels }
    }

    pub fn find_by_name(&self, name: &str) -> Option<&LevelListEntry> {
        self.levels
            .iter()
            .find(|l| l.name.eq_ignore_ascii_case(name))
    }

    /// Resolve level IDs matching `spawn_packs::allocate_auto_ids` (`src/xrGame/spawn_packs.cpp:94-145`):
    /// - Reserves `0`, every `used_graph_ids` entry, and every explicit numeric `< 255` entry.
    /// - Sorts `id = auto` levels (not already named in `existing_graph_level_names`) alphabetically by `name`,
    ///   assigning the highest free ID at or below `254`.
    pub fn allocate_level_ids(
        &self,
        used_graph_ids: &[u8],
        existing_graph_level_names: &[&str],
    ) -> BTreeMap<String, u8> {
        const INVALID_U8: u32 = 255;
        let mut reserved = [false; 256];
        reserved[0] = true;
        for &gid in used_graph_ids {
            reserved[usize::from(gid)] = true;
        }

        let existing_set: BTreeSet<String> = existing_graph_level_names
            .iter()
            .map(|n| n.to_ascii_lowercase())
            .collect();

        // Mirrors C++ `listed_section`: last matching entry wins per name when deduplicating.
        let mut last_section_for_name: BTreeMap<String, &str> = BTreeMap::new();
        for entry in &self.levels {
            last_section_for_name.insert(entry.name.to_ascii_lowercase(), &entry.section);
        }

        let mut assigned = BTreeMap::new();
        let mut automatic: Vec<&str> = Vec::new();

        for entry in &self.levels {
            if last_section_for_name.get(&entry.name.to_ascii_lowercase()) != Some(&entry.section.as_str()) {
                continue;
            }
            match entry.id_spec {
                Some(LevelIdSpec::Explicit(id)) => {
                    if id < INVALID_U8 {
                        reserved[id as usize] = true;
                        assigned.insert(entry.name.clone(), id as u8);
                    }
                }
                Some(LevelIdSpec::Auto)
                    if !existing_set.contains(&entry.name.to_ascii_lowercase()) =>
                {
                    automatic.push(&entry.name);
                }
                _ => {}
            }
        }

        automatic.sort();
        let mut next = (INVALID_U8 - 1) as usize;
        for name in automatic {
            while next > 0 && reserved[next] {
                next -= 1;
            }
            if next == 0 {
                break;
            }
            reserved[next] = true;
            assigned.insert(name.to_string(), next as u8);
        }

        assigned
    }
}

/// `[global_map]` configuration from `game_maps_single.ltx` (`CUIGlobalMap::Init_internal`).
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalMapDef {
    pub texture: String,
    pub texture_dds_vpath: String,
    /// `[x1, y1, x2, y2]` global pixel/world rect (pre-kx-scale, raw LTX values).
    pub bound_rect: [f32; 4],
    pub max_zoom: f32,
}

/// Level minimap / PDA map entry from `game_maps_single.ltx` (`[<level>]`) and optional `level.ltx [level_map]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelMapDef {
    pub level_name: String,
    pub is_multiplayer: bool,
    pub texture: String,
    pub texture_dds_vpath: String,
    /// Raw world `[min_x, min_z, max_x, max_z]` (`bound_rect` in LTX; engine labels `x1, y1, x2, y2`).
    pub bound_rect: [f32; 4],
    /// Raw `[x1, y1, x2, y2]` inside `[global_map] bound_rect` where this level map is placed.
    pub global_rect: Option<[f32; 4]>,
    pub radius: f32,
    pub weathers: Option<String>,
    pub music_tracks: Vec<String>,
    pub label: Option<String>,
    pub label_scale_max: f32,
    pub label_color_argb: u32,
    pub label_offset: [f32; 2],
}

/// Default fallback minimap definition `[def_map]` inside `game_maps_single.ltx`.
#[derive(Debug, Clone, PartialEq)]
pub struct DefaultMapDef {
    pub texture: String,
    pub texture_dds_vpath: String,
    pub bound_rect: [f32; 4],
}

/// Complete world / PDA / minimap registry parsed from `game_maps_single.ltx` + per-level `level.ltx`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct WorldMapsConfig {
    pub global_map: Option<GlobalMapDef>,
    pub def_map: Option<DefaultMapDef>,
    pub level_maps: BTreeMap<String, LevelMapDef>,
}

impl WorldMapsConfig {
    pub fn parse_ltx_str(ltx: &str) -> Result<Self, FormatError> {
        let doc = LtxDocument::parse(ltx);
        Self::from_ltx(&doc)
    }

    pub fn from_ltx(doc: &LtxDocument<'_>) -> Result<Self, FormatError> {
        Self::from_query(doc)
    }

    pub fn from_owned_ltx(doc: &OwnedLtxDocument) -> Result<Self, FormatError> {
        Self::from_query(doc)
    }

    fn from_query<Q: LtxQuery>(q: &Q) -> Result<Self, FormatError> {
        let global_map = if q.has_section("global_map") {
            let texture = q
                .read_str("global_map", "texture")
                .unwrap_or("")
                .trim()
                .trim_matches('"')
                .to_string();
            let bound_rect = q
                .read_str("global_map", "bound_rect")
                .and_then(parse_f32_4)
                .unwrap_or([0.0; 4]);
            let max_zoom = q
                .read_str("global_map", "max_zoom")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(6.0);
            Some(GlobalMapDef {
                texture_dds_vpath: texture_stem_to_dds_vpath(&texture),
                texture,
                bound_rect,
                max_zoom,
            })
        } else {
            None
        };

        let def_map = if q.has_section("def_map") {
            let texture = q
                .read_str("def_map", "texture")
                .unwrap_or("")
                .trim()
                .trim_matches('"')
                .to_string();
            let bound_rect = q
                .read_str("def_map", "bound_rect")
                .and_then(parse_f32_4)
                .unwrap_or([0.0; 4]);
            Some(DefaultMapDef {
                texture_dds_vpath: texture_stem_to_dds_vpath(&texture),
                texture,
                bound_rect,
            })
        } else {
            None
        };

        let mut level_maps = BTreeMap::new();
        for (sect_name, is_mp) in [
            ("level_maps_single", false),
            ("level_maps_mp", true),
        ] {
            if !q.has_section(sect_name) {
                continue;
            }
            for (raw_name, _) in q.section_keys(sect_name) {
                let level_name = raw_name.trim().to_ascii_lowercase();
                if level_name.is_empty() {
                    continue;
                }
                let sect_match = if q.has_section(&level_name) {
                    level_name.as_str()
                } else {
                    raw_name.trim()
                };

                let texture = q
                    .read_str(sect_match, "texture")
                    .or_else(|| def_map.as_ref().map(|d| d.texture.as_str()))
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .to_string();
                let bound_rect = q
                    .read_str(sect_match, "bound_rect")
                    .and_then(parse_f32_4)
                    .or_else(|| def_map.as_ref().map(|d| d.bound_rect))
                    .unwrap_or([0.0; 4]);
                let global_rect = q.read_str(sect_match, "global_rect").and_then(parse_f32_4);
                let radius = q
                    .read_str(sect_match, "radius")
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0);
                let weathers = q
                    .read_str(sect_match, "weathers")
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .filter(|s| !s.is_empty());
                let music_tracks = q
                    .read_str(sect_match, "music_tracks")
                    .map(|s| {
                        s.split(',')
                            .map(|t| t.trim().trim_matches('"').to_string())
                            .filter(|t| !t.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();
                let label = q
                    .read_str(sect_match, "label")
                    .map(|s| s.trim().trim_matches('"').to_string())
                    .filter(|s| !s.is_empty());
                let label_scale_max = q
                    .read_str(sect_match, "label_scale_max")
                    .and_then(|s| s.trim().parse::<f32>().ok())
                    .unwrap_or(0.0);
                let label_color_argb = q
                    .read_str(sect_match, "label_color")
                    .and_then(parse_ltx_color_argb)
                    .unwrap_or(DEFAULT_MAP_LABEL_COLOR_ARGB);
                let label_offset = [
                    q.read_str(sect_match, "label_offset_x")
                        .and_then(|s| s.trim().parse::<f32>().ok())
                        .unwrap_or(0.0),
                    q.read_str(sect_match, "label_offset_y")
                        .and_then(|s| s.trim().parse::<f32>().ok())
                        .unwrap_or(0.0),
                ];

                level_maps.insert(
                    level_name.clone(),
                    LevelMapDef {
                        level_name,
                        is_multiplayer: is_mp,
                        texture_dds_vpath: texture_stem_to_dds_vpath(&texture),
                        texture,
                        bound_rect,
                        global_rect,
                        radius,
                        weathers,
                        music_tracks,
                        label,
                        label_scale_max,
                        label_color_argb,
                        label_offset,
                    },
                );
            }
        }

        Ok(Self {
            global_map,
            def_map,
            level_maps,
        })
    }

    /// Look up a level map entry case-insensitively.
    pub fn get_level(&self, level_name: &str) -> Option<&LevelMapDef> {
        self.level_maps.get(&level_name.to_ascii_lowercase())
    }

    /// Apply `[level_map]` (`texture`, `bound_rect`) from `levels\<level_name>\level.ltx`
    /// (`CUICustomMap::Initialize`, `UIMap.cpp:42-51`).
    pub fn apply_level_ltx_override(&mut self, level_name: &str, level_ltx: &LtxDocument<'_>) {
        if !level_ltx.contains_section("level_map") {
            return;
        }
        let key = level_name.to_ascii_lowercase();
        let texture_opt = level_ltx
            .read_str("level_map", "texture")
            .map(|s| s.trim().trim_matches('"').to_string())
            .filter(|s| !s.is_empty());
        let bound_opt = level_ltx
            .read_str("level_map", "bound_rect")
            .and_then(parse_f32_4);

        if let Some(entry) = self.level_maps.get_mut(&key) {
            if let Some(tex) = texture_opt {
                entry.texture_dds_vpath = texture_stem_to_dds_vpath(&tex);
                entry.texture = tex;
            }
            if let Some(br) = bound_opt {
                entry.bound_rect = br;
            }
        } else {
            let def_tex = self
                .def_map
                .as_ref()
                .map(|d| d.texture.clone())
                .unwrap_or_default();
            let tex = texture_opt.unwrap_or(def_tex);
            let br = bound_opt
                .or_else(|| self.def_map.as_ref().map(|d| d.bound_rect))
                .unwrap_or([0.0; 4]);
            self.level_maps.insert(
                key.clone(),
                LevelMapDef {
                    level_name: key,
                    is_multiplayer: false,
                    texture_dds_vpath: texture_stem_to_dds_vpath(&tex),
                    texture: tex,
                    bound_rect: br,
                    global_rect: None,
                    radius: 0.0,
                    weathers: None,
                    music_tracks: Vec::new(),
                    label: None,
                    label_scale_max: 0.0,
                    label_color_argb: DEFAULT_MAP_LABEL_COLOR_ARGB,
                    label_offset: [0.0, 0.0],
                },
            );
        }
    }

    /// Convert world `(x, z)` meters to normalized level minimap texture `[u, v]` in `[0, 1]`
    /// (X-Ray world `+Z` maps upward on screen, so `v = 1.0` at `min_z` and `0.0` at `max_z`).
    pub fn world_xz_to_level_uv(&self, level_name: &str, world_xz: [f32; 2]) -> Option<[f32; 2]> {
        let lvl = self.get_level(level_name)?;
        let [bx1, bz1, bx2, bz2] = lvl.bound_rect;
        let w = bx2 - bx1;
        let h = bz2 - bz1;
        if w.abs() <= 1.0e-6 || h.abs() <= 1.0e-6 {
            return None;
        }
        let u = (world_xz[0] - bx1) / w;
        let v = (bz2 - world_xz[1]) / h;
        Some([u, v])
    }

    /// Convert normalized `[u, v]` inside a level minimap texture back to world `[x, z]` meters.
    pub fn level_uv_to_world_xz(&self, level_name: &str, uv: [f32; 2]) -> Option<[f32; 2]> {
        let lvl = self.get_level(level_name)?;
        let [bx1, bz1, bx2, bz2] = lvl.bound_rect;
        let x = bx1 + uv[0] * (bx2 - bx1);
        let z = bz2 - uv[1] * (bz2 - bz1);
        Some([x, z])
    }

    /// Convert world `(x, z)` meters to level map local pixels (`CUICustomMap::ConvertRealToLocal`,
    /// `UIMap.cpp:121-154`), supporting optional heading rotation around `pivot_local_px` and screen `kx`.
    pub fn world_xz_to_local_px(
        &self,
        level_name: &str,
        world_xz: [f32; 2],
        zoom: f32,
        kx: f32,
        heading_rad: Option<(f32, [f32; 2])>,
    ) -> Option<[f32; 2]> {
        let lvl = self.get_level(level_name)?;
        let [bx1, bz1, _, bz2] = lvl.bound_rect;
        let height = bz2 - bz1;

        match heading_rad {
            None => {
                let px = (world_xz[0] - bx1) * zoom * kx;
                let py = (height - (world_xz[1] - bz1)) * zoom;
                Some([px, py])
            }
            Some((angle, pivot)) => {
                let bx1_k = bx1 * kx;
                let raw_x = (world_xz[0] * kx - bx1_k) * zoom;
                let raw_y = (height - (world_xz[1] - bz1)) * zoom;
                let tx = raw_x - pivot[0];
                let ty = raw_y - pivot[1];
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let rx = (tx * cos_a + ty * sin_a) * kx + pivot[0];
                let ry = (ty * cos_a - tx * sin_a) + pivot[1];
                Some([rx, ry])
            }
        }
    }

    /// Normalized `[u1, v1, u2, v2]` sub-rectangle of `level_name`'s `global_rect` on the global PDA map texture.
    pub fn level_global_uv_rect(&self, level_name: &str) -> Option<[f32; 4]> {
        let gm = self.global_map.as_ref()?;
        let lvl = self.get_level(level_name)?;
        let gr = lvl.global_rect?;
        let [gx1, gy1, gx2, gy2] = gm.bound_rect;
        let gw = gx2 - gx1;
        let gh = gy2 - gy1;
        if gw.abs() <= 1.0e-6 || gh.abs() <= 1.0e-6 {
            return None;
        }
        Some([
            (gr[0] - gx1) / gw,
            (gr[1] - gy1) / gh,
            (gr[2] - gx1) / gw,
            (gr[3] - gy1) / gh,
        ])
    }

    /// Map a world `(x, z)` position inside `level_name` onto normalized `[u, v]` of the global PDA map texture.
    pub fn world_xz_to_global_uv(&self, level_name: &str, world_xz: [f32; 2]) -> Option<[f32; 2]> {
        let level_uv = self.world_xz_to_level_uv(level_name, world_xz)?;
        let g_rect = self.level_global_uv_rect(level_name)?;
        Some([
            g_rect[0] + level_uv[0] * (g_rect[2] - g_rect[0]),
            g_rect[1] + level_uv[1] * (g_rect[3] - g_rect[1]),
        ])
    }

    /// Map a world `(x, z)` position inside `level_name` into local pixel coordinates on `[global_map]`
    /// at zoom `global_zoom` with aspect factor `kx` (`CUIGlobalMap::ConvertRealToLocal`).
    pub fn world_xz_to_global_px(
        &self,
        level_name: &str,
        world_xz: [f32; 2],
        global_zoom: f32,
        kx: f32,
    ) -> Option<[f32; 2]> {
        let gm = self.global_map.as_ref()?;
        let lvl = self.get_level(level_name)?;
        let gr = lvl.global_rect?;
        let level_uv = self.world_xz_to_level_uv(level_name, world_xz)?;
        let gr_x1 = gr[0] * kx;
        let gr_x2 = gr[2] * kx;
        let gx1 = gm.bound_rect[0] * kx;
        let gy1 = gm.bound_rect[1];

        let target_gx = gr_x1 + level_uv[0] * (gr_x2 - gr_x1);
        let target_gy = gr[1] + level_uv[1] * (gr[3] - gr[1]);
        Some([
            (target_gx - gx1) * global_zoom,
            (target_gy - gy1) * global_zoom,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_levels_and_auto_id_allocation() {
        let ltx = r#"
[levels]
level01 =
level02 =
level03 =
level04 =

[level01]
name = l01_escape
caption = "Cordon"
offset = 0.0, 0.0, 0.0
weathers = default
id = 1

[level02]
name = z_custom_north
caption = "North"
id = auto

[level03]
name = a_custom_swamp
caption = "Swamp"
id = auto

[level04]
name = k02_trucks
caption = "Truck Cemetery"
id = 254
"#;
        let cfg = GameLevelsConfig::parse_ltx_str(ltx);
        assert_eq!(cfg.levels.len(), 4);
        assert_eq!(cfg.find_by_name("l01_escape").unwrap().caption, "Cordon");

        // `254` is explicit -> auto levels sorted alphabetically:
        // `a_custom_swamp` gets `253`, `z_custom_north` gets `252` (skipping explicit `1` and graph-used `251` if present).
        let ids = cfg.allocate_level_ids(&[1, 252], &["l01_escape"]);
        assert_eq!(ids["l01_escape"], 1);
        assert_eq!(ids["k02_trucks"], 254);
        assert_eq!(ids["a_custom_swamp"], 253);
        assert_eq!(ids["z_custom_north"], 251);
    }

    #[test]
    fn world_maps_parsing_overrides_and_coordinate_transforms() {
        let maps_ltx = r#"
[global_map]
texture = ui\ui_global_map
bound_rect = 0.0, 0.0, 1024.0, 2048.0
max_zoom = 6.0

[def_map]
texture = ui\ui_nomap2
bound_rect = 0.0, 0.0, 100.0, 100.0

[level_maps_single]
l01_escape =
l02_garbage =

[l01_escape]
texture = map\map_escape
bound_rect = -400.0, -400.0, 400.0, 400.0
global_rect = 200.0, 1400.0, 400.0, 1600.0
weathers = dynamic_default
music_tracks = l01_escape_musics
label = st_l01_escape_name
label_scale_max = 3.5
label_color = 210, 220, 230
label_offset_x = 12.0
label_offset_y = -8.0

[l02_garbage]
"#;
        let mut world = WorldMapsConfig::parse_ltx_str(maps_ltx).unwrap();
        let gm = world.global_map.as_ref().unwrap();
        assert_eq!(gm.max_zoom, 6.0);
        assert_eq!(gm.texture_dds_vpath, "textures\\ui\\ui_global_map.dds");

        let esc = world.get_level("l01_escape").unwrap();
        assert_eq!(esc.label.as_deref(), Some("st_l01_escape_name"));
        assert_eq!(esc.label_color_argb, color_argb(255, 210, 220, 230));
        assert_eq!(esc.label_offset, [12.0, -8.0]);

        // Fallback to [def_map] before level.ltx override
        let gar = world.get_level("l02_garbage").unwrap();
        assert_eq!(gar.texture, "ui\\ui_nomap2");
        assert_eq!(gar.label_color_argb, DEFAULT_MAP_LABEL_COLOR_ARGB);

        // Apply level.ltx [level_map] override
        let level_ltx = LtxDocument::parse(
            "[level_map]\ntexture = map\\map_garbage\nbound_rect = -300.0, -300.0, 300.0, 300.0\n",
        );
        world.apply_level_ltx_override("l02_garbage", &level_ltx);
        let gar2 = world.get_level("l02_garbage").unwrap();
        assert_eq!(gar2.texture, "map\\map_garbage");
        assert_eq!(gar2.bound_rect, [-300.0, -300.0, 300.0, 300.0]);

        // World (0, 0) is dead center (0.5, 0.5) of [-400, -400, 400, 400]; north (+Z = +400) is v = 0.0!
        assert_eq!(world.world_xz_to_level_uv("l01_escape", [0.0, 0.0]), Some([0.5, 0.5]));
        assert_eq!(world.world_xz_to_level_uv("l01_escape", [-400.0, 400.0]), Some([0.0, 0.0]));
        assert_eq!(world.level_uv_to_world_xz("l01_escape", [0.25, 0.75]), Some([-200.0, -200.0]));

        // Global map placement: (0.5, 0.5) of global_rect [200,1400,400,1600] -> (300, 1500) on 1024x2048
        let guv = world.world_xz_to_global_uv("l01_escape", [0.0, 0.0]).unwrap();
        assert!((guv[0] - 300.0 / 1024.0).abs() < 1e-6);
        assert!((guv[1] - 1500.0 / 2048.0).abs() < 1e-6);

        let gpx = world.world_xz_to_global_px("l01_escape", [0.0, 0.0], 2.0, 0.75).unwrap();
        assert!((gpx[0] - 300.0 * 0.75 * 2.0).abs() < 1e-5);
        assert!((gpx[1] - 1500.0 * 2.0).abs() < 1e-5);
    }
}
