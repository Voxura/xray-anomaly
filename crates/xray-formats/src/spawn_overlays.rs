//! Pure Rust X-Ray Anomaly `spawn_overlays.ltx` / `mod_spawn_overlays_*.ltx` (DLTX) Parsers & Applicators
//!
//! Mirrors `src/xrGame/spawn_overlays.cpp`, `src/xrGame/spawn_templates.cpp`, and
//! `src/xrGame/game_graph_builder.cpp`:
//! - `[settings]` (`link_tolerance`, `sync_level_offsets`)
//! - `[graph_links]` & `[graph_links@*]` (`level_a:x,y,z = level_b:x,y,z`)
//! - `[graph_unlinks]` & `[graph_unlinks@*]` (`level_a = level_b` or point-pair)
//! - `[spawn_remove]` & `[spawn_replace]` (`name [= match_position]`, 0.5 m tolerance)
//! - `[path_remove]` (named patrol paths)
//! - `[level_cut]` (isolates levels in `GameGraph`, drops non-actor templates, inbound `CSE_ALifeLevelChanger`s & level paths)
//! - `[spawn_patch@<name>]` (`match_position`, `level`, `position`, `direction`, `custom_data`, `custom_data_cfg`,
//!   `story_id`, `spawn_story_id`, `object_flags`)
//! - Runtime applicators on `GameGraph` (`apply_to_game_graph`) and `AlifeSpawnArchive` (`apply_to_spawn_archive`).

use std::collections::HashSet;

use xray_core::ltx_loader::OwnedLtxDocument;
use xray_core::LtxDocument;

use crate::ai_graph::{GameEdge, GameGraph};
use crate::spawns::{AlifeLevelChangerState, AlifeObjectState, AlifeSpawnArchive};

pub const DEFAULT_LINK_TOLERANCE: f32 = 2.0;
pub const MATCH_TOLERANCE: f32 = 0.5;

/// Parsed endpoint in `[graph_links]` / `[graph_unlinks]`: `"level_name"` or `"level_name:x,y,z"`.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayEndpoint {
    pub level_name: String,
    pub point: Option<[f32; 3]>,
}

/// Directed-pair bidirectional link rule from `[graph_links]` / `[graph_links@*]`.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayGraphLink {
    pub source_section: String,
    pub raw_key: String,
    pub a: OverlayEndpoint,
    pub b: OverlayEndpoint,
}

/// Edge/level-pair removal rule from `[graph_unlinks]` / `[graph_unlinks@*]`.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayGraphUnlink {
    pub source_section: String,
    pub raw_key: String,
    pub a: OverlayEndpoint,
    pub b: OverlayEndpoint,
}

/// Object selector in `[spawn_remove]` / `[spawn_replace]`.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayObjectSelector {
    pub name: String,
    pub match_position: Option<[f32; 3]>,
}

/// Template override rule from `[spawn_patch@<target_name>]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OverlaySpawnPatch {
    pub section: String,
    pub target_name: String,
    pub match_position: Option<[f32; 3]>,
    pub level: Option<String>,
    pub position: Option<[f32; 3]>,
    pub direction: Option<[f32; 3]>,
    pub custom_data: Option<String>,
    pub story_id: Option<u32>,
    pub spawn_story_id: Option<u32>,
    pub object_flags: Option<u32>,
}

/// Complete parsed DLTX spawn overlay ruleset (`spawn_overlays.ltx` + `mod_spawn_overlays_*.ltx`).
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnOverlaysConfig {
    pub link_tolerance: f32,
    pub sync_level_offsets: bool,
    pub graph_links: Vec<OverlayGraphLink>,
    pub graph_unlinks: Vec<OverlayGraphUnlink>,
    pub spawn_removes: Vec<OverlayObjectSelector>,
    pub spawn_replaces: Vec<OverlayObjectSelector>,
    pub path_removes: Vec<String>,
    pub level_cuts: Vec<String>,
    pub spawn_patches: Vec<OverlaySpawnPatch>,
    pub diagnostics: Vec<String>,
}

impl Default for SpawnOverlaysConfig {
    fn default() -> Self {
        Self {
            link_tolerance: DEFAULT_LINK_TOLERANCE,
            sync_level_offsets: true,
            graph_links: Vec::new(),
            graph_unlinks: Vec::new(),
            spawn_removes: Vec::new(),
            spawn_replaces: Vec::new(),
            path_removes: Vec::new(),
            level_cuts: Vec::new(),
            spawn_patches: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}

/// Summary counters returned by `SpawnOverlaysConfig::apply_to_game_graph`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverlayGraphReport {
    pub levels_moved: u32,
    pub edges_reweighted: u32,
    pub edges_added: u32,
    pub edges_removed: u32,
}

/// Summary counters returned by `SpawnOverlaysConfig::apply_to_spawn_archive`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverlayArchiveReport {
    pub graph: OverlayGraphReport,
    pub spawns_removed: u32,
    pub spawns_patched: u32,
    pub paths_removed: u32,
}

/// Parse `"level_name"` or `"level_name:x,y,z"` (`parse_endpoint`, `spawn_overlays.cpp:45-68`).
pub fn parse_endpoint(text: &str) -> Option<OverlayEndpoint> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some((lvl, coords)) = trimmed.split_once(':') {
        let level_name = lvl.trim();
        if level_name.is_empty() {
            return None;
        }
        let point = parse_vec3_str(coords)?;
        Some(OverlayEndpoint {
            level_name: level_name.to_string(),
            point: Some(point),
        })
    } else {
        Some(OverlayEndpoint {
            level_name: trimmed.to_string(),
            point: None,
        })
    }
}

/// Unescape `custom_data` inline escapes (`\n` -> newline, `\\` -> `\`, `spawn_overlays.cpp:304-323`).
pub fn unescape_custom_data(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some('n') => {
                    out.push('\n');
                    chars.next();
                    continue;
                }
                Some('\\') => {
                    out.push('\\');
                    chars.next();
                    continue;
                }
                _ => {}
            }
        }
        out.push(c);
    }
    out
}

fn parse_vec3_str(text: &str) -> Option<[f32; 3]> {
    let mut it = text.split(',').map(str::trim).filter(|s| !s.is_empty());
    let x = it.next()?.parse::<f32>().ok()?;
    let y = it.next()?.parse::<f32>().ok()?;
    let z = it.next()?.parse::<f32>().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some([x, y, z])
}

fn parse_object_flags_str(text: &str) -> Option<u32> {
    let s = text.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else if let Some(oct) = s.strip_prefix('0').filter(|rest| !rest.is_empty()) {
        u32::from_str_radix(oct, 8).ok()
    } else {
        s.parse::<u32>().ok()
    }
}

fn matching_section_names<'a>(all_names: &[&'a str], prefix: &str) -> Vec<&'a str> {
    let mut matches: Vec<&'a str> = all_names
        .iter()
        .copied()
        .filter(|&s| {
            if let Some(rest) = s.strip_prefix(prefix) {
                rest.is_empty() || rest.starts_with('@')
            } else {
                false
            }
        })
        .collect();
    matches.sort_unstable();
    matches
}

trait OverlayLtxView {
    fn section_names_sorted(&self) -> Vec<&str>;
    fn has_section(&self, section: &str) -> bool;
    fn read_str(&self, section: &str, key: &str) -> Option<&str>;
    fn sorted_entries(&self, section: &str) -> Vec<(&str, &str)>;
}

impl<'a> OverlayLtxView for LtxDocument<'a> {
    fn section_names_sorted(&self) -> Vec<&str> {
        self.section_names()
    }

    fn has_section(&self, section: &str) -> bool {
        self.contains_section(section)
    }

    fn read_str(&self, section: &str, key: &str) -> Option<&str> {
        LtxDocument::read_str(self, section, key)
    }

    fn sorted_entries(&self, section: &str) -> Vec<(&str, &str)> {
        let mut entries: Vec<(&str, &str)> = self
            .sections
            .get(section)
            .map(|s| s.entries.iter().map(|(&k, &v)| (k, v)).collect())
            .unwrap_or_default();
        entries.sort_unstable_by_key(|(k, _)| *k);
        entries
    }
}

impl OverlayLtxView for OwnedLtxDocument {
    fn section_names_sorted(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.sections.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    fn has_section(&self, section: &str) -> bool {
        self.sections.contains_key(section)
    }

    fn read_str(&self, section: &str, key: &str) -> Option<&str> {
        OwnedLtxDocument::read_str(self, section, key)
    }

    fn sorted_entries(&self, section: &str) -> Vec<(&str, &str)> {
        let mut entries: Vec<(&str, &str)> = self
            .sections
            .get(section)
            .map(|s| {
                s.entries
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .collect()
            })
            .unwrap_or_default();
        entries.sort_unstable_by_key(|(k, _)| *k);
        entries
    }
}

fn parse_bool_str(val: &str, default: bool) -> bool {
    match val.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "on" | "yes" => true,
        "false" | "0" | "off" | "no" => false,
        _ => default,
    }
}

fn parse_overlay_from_view<V: OverlayLtxView>(view: &V) -> SpawnOverlaysConfig {
    let mut cfg = SpawnOverlaysConfig::default();

    if view.has_section("settings") {
        if let Some(v) = view
            .read_str("settings", "sync_level_offsets")
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            cfg.sync_level_offsets = parse_bool_str(v, true);
        }
        if let Some(v) = view
            .read_str("settings", "link_tolerance")
            .map(str::trim)
            .filter(|s| !s.is_empty())
            && let Ok(tol) = v.parse::<f32>()
        {
            cfg.link_tolerance = tol;
        }
    }

    let all_sections = view.section_names_sorted();

    for sec in matching_section_names(&all_sections, "graph_links") {
        for (k, v) in view.sorted_entries(sec) {
            match (parse_endpoint(k), parse_endpoint(v)) {
                (Some(a), Some(b)) if a.point.is_some() && b.point.is_some() => {
                    cfg.graph_links.push(OverlayGraphLink {
                        source_section: sec.to_string(),
                        raw_key: k.to_string(),
                        a,
                        b,
                    });
                }
                _ => {
                    cfg.diagnostics.push(format!(
                        "[{sec}] {k}: expected 'level:x,y,z = level:x,y,z'"
                    ));
                }
            }
        }
    }

    for sec in matching_section_names(&all_sections, "graph_unlinks") {
        for (k, v) in view.sorted_entries(sec) {
            match (parse_endpoint(k), parse_endpoint(v)) {
                (Some(a), Some(b)) if a.point.is_some() == b.point.is_some() => {
                    cfg.graph_unlinks.push(OverlayGraphUnlink {
                        source_section: sec.to_string(),
                        raw_key: k.to_string(),
                        a,
                        b,
                    });
                }
                _ => {
                    cfg.diagnostics.push(format!(
                        "[{sec}] {k}: expected 'level = level' or 'level:x,y,z = level:x,y,z'"
                    ));
                }
            }
        }
    }

    for (k, v) in view.sorted_entries("spawn_remove") {
        let match_position = if v.trim().is_empty() {
            None
        } else {
            parse_vec3_str(v)
        };
        cfg.spawn_removes.push(OverlayObjectSelector {
            name: k.to_string(),
            match_position,
        });
    }

    for (k, v) in view.sorted_entries("spawn_replace") {
        let match_position = if v.trim().is_empty() {
            None
        } else {
            parse_vec3_str(v)
        };
        cfg.spawn_replaces.push(OverlayObjectSelector {
            name: k.to_string(),
            match_position,
        });
    }

    for (k, _) in view.sorted_entries("path_remove") {
        cfg.path_removes.push(k.to_string());
    }

    let mut seen_cuts = HashSet::new();
    for (k, _) in view.sorted_entries("level_cut") {
        let low = k.to_ascii_lowercase();
        if seen_cuts.insert(low) {
            cfg.level_cuts.push(k.to_string());
        }
    }

    for &sec in &all_sections {
        let Some(target_name) = sec.strip_prefix("spawn_patch@") else {
            continue;
        };
        if target_name.is_empty() {
            continue;
        }
        let mut patch = OverlaySpawnPatch {
            section: sec.to_string(),
            target_name: target_name.to_string(),
            ..Default::default()
        };
        for (k, v) in view.sorted_entries(sec) {
            let v_trim = v.trim();
            if v_trim.is_empty() {
                cfg.diagnostics
                    .push(format!("[{sec}] {k}: empty value"));
                continue;
            }
            match k {
                "match_position" => patch.match_position = parse_vec3_str(v_trim),
                "level" => patch.level = Some(v_trim.to_string()),
                "position" => patch.position = parse_vec3_str(v_trim),
                "direction" => patch.direction = parse_vec3_str(v_trim),
                "custom_data" => patch.custom_data = Some(unescape_custom_data(v_trim)),
                "custom_data_cfg" => {
                    patch.custom_data = Some(format!("[logic]\ncfg = {v_trim}"));
                }
                "story_id" => {
                    if let Ok(id) = v_trim.parse::<i32>() {
                        patch.story_id = Some(id as u32);
                    }
                }
                "spawn_story_id" => {
                    if let Ok(id) = v_trim.parse::<i32>() {
                        patch.spawn_story_id = Some(id as u32);
                    }
                }
                "object_flags" => patch.object_flags = parse_object_flags_str(v_trim),
                other => cfg
                    .diagnostics
                    .push(format!("[{sec}] {other}: unknown key")),
            }
        }
        cfg.spawn_patches.push(patch);
    }

    cfg
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn find_level_id(graph: &GameGraph, name: &str) -> Option<u8> {
    graph
        .header
        .levels
        .iter()
        .find(|l| l.name.eq_ignore_ascii_case(name))
        .map(|l| l.id)
}

fn unpack_graph_edges(graph: &GameGraph) -> (Vec<Vec<GameEdge>>, Vec<u32>) {
    let vcount = graph.vertices.len();
    let old_dp_base = (vcount * 42 + graph.edges.len() * 6) as u32;
    let mut adj = Vec::with_capacity(vcount);
    let mut dp_rel_offsets = Vec::with_capacity(vcount);
    for idx in 0..vcount {
        adj.push(graph.vertex_edges(idx as u16).to_vec());
        let p_off = graph.vertices[idx].point_offset;
        dp_rel_offsets.push(p_off.saturating_sub(old_dp_base));
    }
    (adj, dp_rel_offsets)
}

fn repack_graph_edges(graph: &mut GameGraph, adj: Vec<Vec<GameEdge>>, dp_rel_offsets: &[u32]) {
    let vcount = graph.vertices.len();
    let mut flat_edges = Vec::new();
    for list in &adj {
        flat_edges.extend_from_slice(list);
    }
    let verts_base = vcount * 42;
    let new_dp_base = verts_base + flat_edges.len() * 6;

    let mut cur_edge = 0usize;
    for (idx, list) in adj.iter().enumerate() {
        let v = &mut graph.vertices[idx];
        v.edge_offset = (verts_base + cur_edge * 6) as u32;
        v.neighbour_count = list.len().min(u8::MAX as usize) as u8;
        v.point_offset = new_dp_base as u32 + dp_rel_offsets[idx];
        cur_edge += list.len();
    }

    graph.header.edge_count = flat_edges.len() as u32;
    graph.edges = flat_edges;
}

impl SpawnOverlaysConfig {
    /// Parse raw `spawn_overlays.ltx` text.
    pub fn from_ltx_str(ltx_text: &str) -> Self {
        let doc = LtxDocument::parse(ltx_text);
        Self::from_ltx(&doc)
    }

    /// Extract rules from a parsed `LtxDocument` (including DLTX merged mod sources).
    pub fn from_ltx(doc: &LtxDocument<'_>) -> Self {
        parse_overlay_from_view(doc)
    }

    /// Extract rules from an `OwnedLtxDocument` (e.g. `load_ltx_tree_from_vfs`).
    pub fn from_owned_ltx(doc: &OwnedLtxDocument) -> Self {
        parse_overlay_from_view(doc)
    }

    /// Apply `[settings] sync_level_offsets`, `[graph_unlinks*]`, `[graph_links*]`, and `[level_cut]`
    /// to `graph`, keeping vertex `edge_offset`, `point_offset`, and `header.edge_count` valid for
    /// binary `to_bytes()` round-trips (`src/xrGame/spawn_overlays.cpp:787-825`).
    pub fn apply_to_game_graph(
        &self,
        graph: &mut GameGraph,
        level_offsets: &[(&str, [f32; 3])],
    ) -> OverlayGraphReport {
        let mut report = OverlayGraphReport::default();
        let (mut adj, dp_rel) = unpack_graph_edges(graph);

        // 1. sync_level_offsets (`apply_level_offsets`, `game_graph_builder.cpp:199-228`)
        if self.sync_level_offsets {
            for &(name, new_offset) in level_offsets {
                let Some(lvl_idx) = graph
                    .header
                    .levels
                    .iter()
                    .position(|l| l.name.eq_ignore_ascii_case(name))
                else {
                    continue;
                };
                let old_offset = graph.header.levels[lvl_idx].offset;
                let delta = [
                    new_offset[0] - old_offset[0],
                    new_offset[1] - old_offset[1],
                    new_offset[2] - old_offset[2],
                ];
                if delta.iter().all(|d| d.abs() <= 1.0e-5) {
                    continue;
                }
                let lvl_id = graph.header.levels[lvl_idx].id;
                graph.header.levels[lvl_idx].offset = new_offset;
                report.levels_moved += 1;

                for v in &mut graph.vertices {
                    if v.level_id == lvl_id {
                        v.global_point[0] += delta[0];
                        v.global_point[1] += delta[1];
                        v.global_point[2] += delta[2];
                    }
                }

                for (from_idx, edges) in adj.iter_mut().enumerate() {
                    let from_lvl = graph.vertices[from_idx].level_id == lvl_id;
                    let from_gp = graph.vertices[from_idx].global_point;
                    for edge in edges.iter_mut() {
                        let to_idx = usize::from(edge.target_vertex_id);
                        if to_idx >= graph.vertices.len() {
                            continue;
                        }
                        if from_lvl != (graph.vertices[to_idx].level_id == lvl_id) {
                            edge.distance = dist3(from_gp, graph.vertices[to_idx].global_point);
                            report.edges_reweighted += 1;
                        }
                    }
                }
            }
        }

        // 2. [graph_unlinks] & [graph_unlinks@*]
        for rule in &self.graph_unlinks {
            match (rule.a.point, rule.b.point) {
                (Some(pa), Some(pb)) => {
                    let Some(la) = find_level_id(graph, &rule.a.level_name) else {
                        continue;
                    };
                    let Some(lb) = find_level_id(graph, &rule.b.level_name) else {
                        continue;
                    };
                    let Some((va, da)) = graph.nearest_vertex_on_level(la, pa) else {
                        continue;
                    };
                    let Some((vb, db)) = graph.nearest_vertex_on_level(lb, pb) else {
                        continue;
                    };
                    if da > self.link_tolerance || db > self.link_tolerance || va == vb {
                        continue;
                    }
                    let before_a = adj[usize::from(va)].len();
                    adj[usize::from(va)].retain(|e| e.target_vertex_id != vb);
                    let removed_a = (before_a - adj[usize::from(va)].len()) as u32;

                    let before_b = adj[usize::from(vb)].len();
                    adj[usize::from(vb)].retain(|e| e.target_vertex_id != va);
                    let removed_b = (before_b - adj[usize::from(vb)].len()) as u32;

                    report.edges_removed += removed_a + removed_b;
                }
                (None, None) => {
                    let Some(la) = find_level_id(graph, &rule.a.level_name) else {
                        continue;
                    };
                    let Some(lb) = find_level_id(graph, &rule.b.level_name) else {
                        continue;
                    };
                    if la == lb {
                        continue;
                    }
                    for (from_idx, edges) in adj.iter_mut().enumerate() {
                        let from_lid = graph.vertices[from_idx].level_id;
                        if from_lid != la && from_lid != lb {
                            continue;
                        }
                        let other_lid = if from_lid == la { lb } else { la };
                        let before = edges.len();
                        edges.retain(|e| {
                            graph
                                .vertices
                                .get(usize::from(e.target_vertex_id))
                                .is_none_or(|tv| tv.level_id != other_lid)
                        });
                        report.edges_removed += (before - edges.len()) as u32;
                    }
                }
                _ => {}
            }
        }

        // 3. [graph_links] & [graph_links@*]
        for rule in &self.graph_links {
            let (Some(pa), Some(pb)) = (rule.a.point, rule.b.point) else {
                continue;
            };
            let Some(la) = find_level_id(graph, &rule.a.level_name) else {
                continue;
            };
            let Some(lb) = find_level_id(graph, &rule.b.level_name) else {
                continue;
            };
            let Some((va, da)) = graph.nearest_vertex_on_level(la, pa) else {
                continue;
            };
            let Some((vb, db)) = graph.nearest_vertex_on_level(lb, pb) else {
                continue;
            };
            if da > self.link_tolerance || db > self.link_tolerance || va == vb {
                continue;
            }
            let dist = dist3(
                graph.vertices[usize::from(va)].global_point,
                graph.vertices[usize::from(vb)].global_point,
            );

            let has_ab = adj[usize::from(va)].iter().any(|e| e.target_vertex_id == vb);
            let has_ba = adj[usize::from(vb)].iter().any(|e| e.target_vertex_id == va);
            if has_ab && has_ba {
                continue;
            }
            if !has_ab && adj[usize::from(va)].len() < 255 {
                adj[usize::from(va)].push(GameEdge {
                    target_vertex_id: vb,
                    distance: dist,
                });
                report.edges_added += 1;
            }
            if !has_ba && adj[usize::from(vb)].len() < 255 {
                adj[usize::from(vb)].push(GameEdge {
                    target_vertex_id: va,
                    distance: dist,
                });
                report.edges_added += 1;
            }
        }

        // 4. [level_cut] (`isolate_level`, `game_graph_builder.cpp:175-197`)
        for cut_name in &self.level_cuts {
            let Some(cut_lid) = find_level_id(graph, cut_name) else {
                continue;
            };
            for (from_idx, edges) in adj.iter_mut().enumerate() {
                let from_cut = graph.vertices[from_idx].level_id == cut_lid;
                let before = edges.len();
                edges.retain(|e| {
                    graph
                        .vertices
                        .get(usize::from(e.target_vertex_id))
                        .is_none_or(|tv| from_cut == (tv.level_id == cut_lid))
                });
                report.edges_removed += (before - edges.len()) as u32;
            }
        }

        repack_graph_edges(graph, adj, &dp_rel);
        report
    }

    /// Apply graph overlays, spawn removals/patches, patrol removals, and `[level_cut]` to `archive`.
    pub fn apply_to_spawn_archive(
        &self,
        archive: &mut AlifeSpawnArchive,
        apply_replace: bool,
        level_offsets: &[(&str, [f32; 3])],
    ) -> OverlayArchiveReport {
        let mut report = OverlayArchiveReport {
            graph: self.apply_to_game_graph(&mut archive.game_graph, level_offsets),
            ..Default::default()
        };

        // Optional [spawn_replace] (usually before pack merge) + [spawn_remove]
        let selectors = if apply_replace {
            self.spawn_replaces
                .iter()
                .chain(self.spawn_removes.iter())
                .collect::<Vec<_>>()
        } else {
            self.spawn_removes.iter().collect::<Vec<_>>()
        };

        for sel in selectors {
            if let Some(idx) = find_spawn_vertex_idx(archive, &sel.name, sel.match_position) {
                archive.spawns.remove(idx);
                report.spawns_removed += 1;
            }
        }

        // [spawn_patch@*]
        for patch in &self.spawn_patches {
            let Some(idx) = find_spawn_vertex_idx(archive, &patch.target_name, patch.match_position) else {
                continue;
            };
            let target_level_id = if let Some(lvl_name) = patch.level.as_deref() {
                let Some(lid) = find_level_id(&archive.game_graph, lvl_name) else {
                    continue;
                };
                if patch.position.is_none() {
                    continue;
                }
                Some(lid)
            } else {
                None
            };

            let vertex = &mut archive.spawns[idx];
            if let Some(pos) = patch.position {
                vertex.spawn.position = pos;
            }
            if let Some(dir) = patch.direction {
                vertex.spawn.angle = dir;
            }
            if !vertex.spawn.is_graph_point()
                && let Ok(mut alife) = AlifeObjectState::parse(
                    vertex.spawn.version,
                    &vertex.spawn.state_payload,
                )
            {
                if let Some(cd) = &patch.custom_data {
                    alife.custom_data = cd.clone();
                }
                if let Some(sid) = patch.story_id {
                    alife.story_id = sid;
                }
                if let Some(ssid) = patch.spawn_story_id {
                    alife.spawn_story_id = ssid;
                }
                if let Some(flags) = patch.object_flags {
                    alife.flags = flags;
                }
                if let Some(pos) = patch.position {
                    let cur_lid = target_level_id.or_else(|| {
                        archive
                            .game_graph
                            .vertices
                            .get(usize::from(alife.game_vertex_id))
                            .map(|v| v.level_id)
                    });
                    if let Some(lid) = cur_lid
                        && let Some((gvid, dist)) =
                            archive.game_graph.nearest_vertex_on_level(lid, pos)
                    {
                        alife.game_vertex_id = gvid;
                        alife.distance = dist;
                    }
                }
                vertex.spawn.state_payload = alife.to_bytes();
            }
            report.spawns_patched += 1;
        }

        // [path_remove]
        if !self.path_removes.is_empty() {
            let remove_set: HashSet<String> = self
                .path_removes
                .iter()
                .map(|s| s.to_ascii_lowercase())
                .collect();
            let before = archive.patrol_paths.len();
            archive
                .patrol_paths
                .retain(|p| !remove_set.contains(&p.name.to_ascii_lowercase()));
            report.paths_removed += (before - archive.patrol_paths.len()) as u32;
        }

        // [level_cut] templates + level changers + patrol paths (`cut_objects`, `spawn_overlays.cpp:560-612`)
        if !self.level_cuts.is_empty() {
            let cut_ids: HashSet<u8> = self
                .level_cuts
                .iter()
                .filter_map(|n| find_level_id(&archive.game_graph, n))
                .collect();

            if !cut_ids.is_empty() {
                let before_spawns = archive.spawns.len();
                archive.spawns.retain(|sv| {
                    if sv.spawn.section.eq_ignore_ascii_case("actor") {
                        return true;
                    }
                    if let Ok(alife) = sv.spawn.decode_alife_state() {
                        if let Some(gv) = archive
                            .game_graph
                            .vertices
                            .get(usize::from(alife.game_vertex_id))
                            && cut_ids.contains(&gv.level_id)
                        {
                            return false;
                        }
                        if sv.spawn.section.contains("level_changer")
                            && let Ok(lc) = AlifeLevelChangerState::parse(
                                sv.spawn.version,
                                &sv.spawn.state_payload,
                            )
                            && let Some(dest_lid) =
                                find_level_id(&archive.game_graph, &lc.level_to_change)
                            && cut_ids.contains(&dest_lid)
                        {
                            return false;
                        }
                    }
                    true
                });
                report.spawns_removed += (before_spawns - archive.spawns.len()) as u32;

                let before_paths = archive.patrol_paths.len();
                archive.patrol_paths.retain(|p| {
                    let on_cut_override = p.level_override.as_deref().is_some_and(|n| {
                        find_level_id(&archive.game_graph, n)
                            .is_some_and(|lid| cut_ids.contains(&lid))
                    });
                    if on_cut_override {
                        return false;
                    }
                    let any_cut_vertex = p.points.iter().any(|pt| {
                        archive
                            .game_graph
                            .vertices
                            .get(usize::from(pt.game_vertex_id))
                            .is_some_and(|gv| cut_ids.contains(&gv.level_id))
                    });
                    !any_cut_vertex
                });
                report.paths_removed += (before_paths - archive.patrol_paths.len()) as u32;
            }
        }

        // Filter dangling SPAWN_GRAPH edges
        let live_ids: HashSet<u16> = archive.spawns.iter().map(|s| s.spawn_id).collect();
        archive
            .edges
            .retain(|e| live_ids.contains(&e.from) && live_ids.contains(&e.to));

        archive.header.spawn_count = archive.spawns.len() as u32;
        report
    }
}

fn find_spawn_vertex_idx(
    archive: &AlifeSpawnArchive,
    name: &str,
    match_position: Option<[f32; 3]>,
) -> Option<usize> {
    let mut candidates: Vec<usize> = archive
        .spawns
        .iter()
        .enumerate()
        .filter(|(_, s)| s.spawn.name_replace == name)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return None;
    }

    if let Some(target_pos) = match_position {
        let mut best: Option<(usize, f32)> = None;
        for idx in candidates {
            let d = dist3(archive.spawns[idx].spawn.position, target_pos);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((idx, d));
            }
        }
        best.and_then(|(idx, d)| (d <= MATCH_TOLERANCE).then_some(idx))
    } else if candidates.len() == 1 {
        candidates.pop()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai_graph::{
        GameDeathPoint, GameGraphHeader, GameLevelInfo, GameVertex, INVALID_AI_VERTEX,
        XRAI_CURRENT_VERSION,
    };
    use crate::spawns::{
        alife_object_flags, m_spawn_flags, AlifeSpaceRestrictorState, PatrolPath, PatrolPoint,
        ServerShapeDef, SpawnEdge, SpawnHeader, SpawnPacket, SpawnVertex, INVALID_OBJECT_ID,
        INVALID_SPAWN_STORY_ID, INVALID_STORY_ID, M_UPDATE, SPAWN_VERSION,
    };
    use std::path::Path;
    use xray_core::ltx_loader::{collect_ltx_sources, LtxFileProvider, OwnedLtxDocument};

    struct FsLtxProvider;

    impl LtxFileProvider for FsLtxProvider {
        type Error = std::io::Error;

        fn read_ltx_text(&self, virtual_path: &str) -> Result<String, Self::Error> {
            let rel = virtual_path.replace('\\', "/");
            std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel))
        }

        fn list_ltx_matches(&self, _dir: &str, _pat: &str) -> Result<Vec<String>, Self::Error> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn parse_real_repo_spawn_overlays_and_dltx_mod_if_present() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../gamedata/configs/spawn_overlays.ltx");
        let mod_meadow = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../gamedata/configs/mod_spawn_overlays_meadow.ltx");
        if !root.exists() || !mod_meadow.exists() {
            return;
        }

        let sources = collect_ltx_sources(
            &FsLtxProvider,
            "gamedata\\configs\\spawn_overlays.ltx",
            &["gamedata\\configs\\mod_spawn_overlays_meadow.ltx"],
        )
        .unwrap();
        let merged = OwnedLtxDocument::from_sources(&sources);
        let cfg = SpawnOverlaysConfig::from_owned_ltx(&merged);

        assert!(cfg.sync_level_offsets);
        assert!((cfg.link_tolerance - 2.0).abs() < 1.0e-5);
        assert_eq!(cfg.graph_links.len(), 2);
        assert!(cfg.graph_links.iter().any(|l| l.a.level_name == "l01_escape" && l.b.level_name == "y04_pole"));
    }

    #[test]
    fn apply_graph_links_unlinks_patches_removes_and_level_cut() {
        let ltx = r#"
[settings]
link_tolerance = 2.0
sync_level_offsets = true

[graph_links@mod]
l01_escape:10.0,0.0,0.0 = l02_garbage:10.0,0.0,0.0

[graph_unlinks]
l01_escape = l03_agroprom

[spawn_remove]
shared_box = 5.0, 0.0, 0.0

[path_remove]
dead_patrol

[level_cut]
l03_agroprom = true

[spawn_patch@esc_generator]
position = 11.0, 0.0, 0.0
direction = 0.0, 1.5, 0.0
custom_data = [logic]\ncfg = scripts\\gen.ltx
story_id = 4242
spawn_story_id = -1
object_flags = 0x201
"#;
        let cfg = SpawnOverlaysConfig::from_ltx_str(ltx);
        assert_eq!(cfg.spawn_patches.len(), 1);
        assert_eq!(
            cfg.spawn_patches[0].custom_data.as_deref(),
            Some("[logic]\ncfg = scripts\\gen.ltx")
        );
        assert_eq!(cfg.spawn_patches[0].spawn_story_id, Some(INVALID_SPAWN_STORY_ID));
        assert_eq!(cfg.spawn_patches[0].object_flags, Some(0x201));

        // Build a 3-vertex GameGraph (one vertex per level), with initial edge 0 -> 2 (l01 <-> l03)
        let gg = GameGraph {
            header: GameGraphHeader {
                version: 10,
                vertex_count: 3,
                edge_count: 2,
                death_point_count: 1,
                guid: [1, 2],
                levels: vec![
                    GameLevelInfo { name: "l01_escape".into(), offset: [0.0; 3], id: 1, section: "level01".into(), guid: [1, 1] },
                    GameLevelInfo { name: "l02_garbage".into(), offset: [0.0, 0.0, 100.0], id: 2, section: "level02".into(), guid: [2, 2] },
                    GameLevelInfo { name: "l03_agroprom".into(), offset: [-100.0, 0.0, 0.0], id: 3, section: "level03".into(), guid: [3, 3] },
                ],
            },
            vertices: vec![
                GameVertex {
                    local_point: [10.0, 0.0, 0.0],
                    global_point: [10.0, 0.0, 0.0],
                    level_id: 1,
                    level_vertex_id: 10,
                    vertex_types: [0; 4],
                    edge_offset: 3 * 42,
                    point_offset: 3 * 42 + 2 * 6,
                    neighbour_count: 1,
                    death_point_count: 1,
                },
                GameVertex {
                    local_point: [10.0, 0.0, 0.0],
                    global_point: [10.0, 0.0, 100.0],
                    level_id: 2,
                    level_vertex_id: 20,
                    vertex_types: [0; 4],
                    edge_offset: 3 * 42 + 6,
                    point_offset: 3 * 42 + 2 * 6 + 20,
                    neighbour_count: 0,
                    death_point_count: 0,
                },
                GameVertex {
                    local_point: [0.0; 3],
                    global_point: [-100.0, 0.0, 0.0],
                    level_id: 3,
                    level_vertex_id: 30,
                    vertex_types: [0; 4],
                    edge_offset: 3 * 42 + 6,
                    point_offset: 3 * 42 + 2 * 6 + 20,
                    neighbour_count: 1,
                    death_point_count: 0,
                },
            ],
            edges: vec![
                GameEdge { target_vertex_id: 2, distance: 110.0 },
                GameEdge { target_vertex_id: 0, distance: 110.0 },
            ],
            death_points: vec![GameDeathPoint { point: [1.0; 3], level_vertex_id: 10, distance: 0.5 }],
            cross_tables: Vec::new(),
        };

        let mk_spawn = |id: u16, section: &str, name: &str, pos: [f32; 3], payload: Vec<u8>| SpawnVertex {
            spawn_id: id,
            spawn: SpawnPacket {
                section: section.to_string(),
                name_replace: name.to_string(),
                game_id_legacy: 0,
                rp: 0xFE,
                position: pos,
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
                spawn_id: id,
                state_payload: payload,
                trailing_update_payload: Vec::new(),
            },
            update_packet: M_UPDATE.to_le_bytes().to_vec(),
        };

        let box_state = AlifeObjectState {
            game_vertex_id: 0,
            distance: 0.0,
            direct_control: true,
            level_vertex_id: 10,
            flags: alife_object_flags::CAN_SAVE,
            custom_data: String::new(),
            story_id: INVALID_STORY_ID,
            spawn_story_id: 10,
            subclass_payload: Vec::new(),
        };
        let lc_state = AlifeLevelChangerState {
            restrictor: AlifeSpaceRestrictorState {
                alife: box_state.clone(),
                shapes: vec![ServerShapeDef::Sphere { center: [0.0; 3], radius: 3.0 }],
                restrictor_type: 0,
                trailing: Vec::new(),
            },
            next_graph_vertex: 2,
            next_level_vertex: 30,
            next_position: [0.0; 3],
            next_angles: [0.0; 3],
            level_to_change: "l03_agroprom".into(),
            level_point_to_change: "start_wp".into(),
            silent_mode: true,
            trailing: Vec::new(),
        };

        let mut archive = AlifeSpawnArchive {
            header: SpawnHeader {
                version: XRAI_CURRENT_VERSION,
                guid: [0x11; 16],
                graph_guid: [0x22; 16],
                spawn_count: 5,
                level_count: 3,
            },
            spawns: vec![
                mk_spawn(0, "actor", "single_player", [0.0; 3], AlifeObjectState { game_vertex_id: 2, ..box_state.clone() }.to_bytes()),
                mk_spawn(1, "physic_object", "shared_box", [5.1, 0.0, 0.0], box_state.to_bytes()),
                mk_spawn(2, "physic_object", "shared_box", [50.0, 0.0, 0.0], box_state.to_bytes()),
                mk_spawn(3, "physic_object", "esc_generator", [9.0, 0.0, 0.0], box_state.to_bytes()),
                mk_spawn(4, "level_changer", "lc_esc_to_agr", [0.0; 3], lc_state.to_bytes()),
            ],
            edges: vec![SpawnEdge { from: 0, to: 1, weight: 1.0 }],
            artefact_spawn_positions: Vec::new(),
            patrol_paths: vec![
                PatrolPath { name: "dead_patrol".into(), level_override: None, way_type: 0, points: Vec::new(), edges: Vec::new() },
                PatrolPath {
                    name: "agr_patrol".into(),
                    level_override: None,
                    way_type: 0,
                    points: vec![PatrolPoint {
                        vertex_id: 0,
                        name: "wp00".into(),
                        pos: [0.0; 3],
                        flags: 0,
                        level_vertex_id: INVALID_AI_VERTEX,
                        game_vertex_id: 2,
                    }],
                    edges: Vec::new(),
                },
                PatrolPath {
                    name: "esc_patrol".into(),
                    level_override: None,
                    way_type: 0,
                    points: vec![PatrolPoint {
                        vertex_id: 0,
                        name: "wp00".into(),
                        pos: [0.0; 3],
                        flags: 0,
                        level_vertex_id: INVALID_AI_VERTEX,
                        game_vertex_id: 0,
                    }],
                    edges: Vec::new(),
                },
            ],
            game_graph: gg,
        };

        let report = cfg.apply_to_spawn_archive(&mut archive, false, &[("l02_garbage", [0.0, 0.0, 200.0])]);
        assert_eq!(report.graph.levels_moved, 1);
        assert_eq!(report.graph.edges_removed, 2);
        assert_eq!(report.graph.edges_added, 2);
        // Removed: shared_box@5.0 + lc_esc_to_agr (target cut level l03_agroprom); kept actor!
        assert_eq!(report.spawns_removed, 2);
        assert_eq!(report.spawns_patched, 1);
        assert_eq!(report.paths_removed, 2);

        // Verify GameGraph roundtrips cleanly after offset sync + link/unlink edits!
        let re_gg = GameGraph::parse(&archive.game_graph.to_bytes()).unwrap();
        assert_eq!(re_gg.vertex_edges(0).len(), 1);
        assert_eq!(re_gg.vertex_edges(0)[0].target_vertex_id, 1);
        assert!((re_gg.vertex_edges(0)[0].distance - 200.0).abs() < 1.0e-4);
        assert_eq!(re_gg.vertex_death_points(0).len(), 1);

        // Verify patched esc_generator
        let generator = archive.find_spawn_by_name("esc_generator").unwrap();
        assert_eq!(generator.spawn.position, [11.0, 0.0, 0.0]);
        let gen_state = generator.spawn.decode_alife_state().unwrap();
        assert_eq!(gen_state.story_id, 4242);
        assert_eq!(gen_state.spawn_story_id, INVALID_SPAWN_STORY_ID);
        assert_eq!(gen_state.flags, 0x201);
        assert_eq!(gen_state.custom_data, "[logic]\ncfg = scripts\\gen.ltx");
        assert!(archive.find_spawn_by_name("single_player").is_some());
        assert!(archive.find_spawn_by_name("shared_box").is_some());
        assert_eq!(archive.patrol_paths.len(), 1);
        assert_eq!(archive.patrol_paths[0].name, "esc_patrol");
    }
}
