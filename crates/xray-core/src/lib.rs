//! # xray-core: Pure Rust foundation for X-Ray Anomaly
//!
//! Provides zero-copy, highly optimized LTX configuration parsing, multi-parent section inheritance,
//! DLTX (Differential LTX) mod override resolution, and type-safe query interfaces.

pub mod ltx_loader;
pub mod string_table;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

pub const DLTX_DELETE: &str = "DLTX_DELETE";

/// An individual raw candidate entry parsed from a base or mod configuration file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DltxItem<'a> {
    pub key: &'a str,
    pub val: &'a str,
    pub depth: i32,
    pub insertion_index: u32,
}

impl<'a> Ord for DltxItem<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        // S.T.A.L.K.E.R. Anomaly DLTX Precedence Comparator (xr_ini.cpp L413-L425):
        // 1. Key Ascending
        // 2. Depth Ascending (lower depth wins, e.g. -200 mod < 0 base < 1 include)
        // 3. Insertion Index Descending (later insertion at equal depth wins)
        self.key
            .cmp(other.key)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| other.insertion_index.cmp(&self.insertion_index))
    }
}

impl<'a> PartialOrd for DltxItem<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// An individual section containing key-value pairs and parent references.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LtxSection<'a> {
    pub name: &'a str,
    pub parents: Vec<&'a str>,
    pub entries: HashMap<&'a str, &'a str>,
}

/// A parsed LTX configuration document with slice borrowing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LtxDocument<'a> {
    pub sections: HashMap<&'a str, LtxSection<'a>>,
    pub deleted_sections: HashSet<&'a str>,
}

impl<'a> LtxDocument<'a> {
    /// Parse raw LTX text with zero allocations for keys and values (default depth 0).
    pub fn parse(input: &'a str) -> Self {
        Self::parse_with_depth(input, 0)
    }

    /// Parse raw LTX text with an explicit DLTX depth level.
    pub fn parse_with_depth(input: &'a str, depth: i32) -> Self {
        Self::parse_sources(&[(depth, input)])
    }

    /// Parse multiple `(depth, text)` slices into a single unified LTX document,
    /// applying DLTX section/key overrides and tombstones globally across all sources.
    pub fn parse_sources(sources: &[(i32, &'a str)]) -> Self {
        let mut doc = LtxDocument::default();
        let mut raw_items: HashMap<&'a str, Vec<DltxItem<'a>>> = HashMap::new();
        let mut insertion_counter: u32 = 0;

        for &(depth, raw_input) in sources {
            let input = raw_input.strip_prefix('\u{feff}').unwrap_or(raw_input);
            let mut current_sec_name: Option<&'a str> = None;

            for line in input.lines() {
                let trimmed = line.trim();
                // Skip empty lines, ';'/'//' comments, and '#' comments/directives (`#include`, `# ...`)
                if trimmed.is_empty()
                    || trimmed.starts_with(';')
                    || trimmed.starts_with("//")
                    || trimmed.starts_with('#')
                {
                    continue;
                }

                // Strip inline comments, taking care of comments inside quotes
                let clean = Self::strip_comment(trimmed);
                if clean.is_empty() {
                    continue;
                }

                // Section delete syntax: !![section]
                if clean.starts_with("!![")
                    && let Some(close_bracket) = clean.find(']')
                {
                    let sec_name = clean[3..close_bracket].trim();
                    doc.deleted_sections.insert(sec_name);
                    doc.sections.remove(sec_name);
                    current_sec_name = None;
                    continue;
                }

                // Section headers: [section], ![section] (override), @[section] (safe override), or [child]:p1,p2
                if clean.starts_with('[') || clean.starts_with("![") || clean.starts_with("@[") {
                    let header_start = if clean.starts_with('[') { 1 } else { 2 };

                    if let Some(close_bracket) = clean[header_start..].find(']') {
                        let sec_name = clean[header_start..header_start + close_bracket].trim();
                        let close_abs = header_start + close_bracket;

                        let mut parents = Vec::new();
                        if let Some(colon_pos) = clean[close_abs..].find(':') {
                            let parent_str = clean[close_abs + colon_pos + 1..].trim();
                            for p in parent_str.split(',') {
                                let p_trimmed = p.trim();
                                if !p_trimmed.is_empty() {
                                    parents.push(p_trimmed);
                                }
                            }
                        }

                        if doc.deleted_sections.contains(sec_name) {
                            current_sec_name = None;
                            continue;
                        }

                        current_sec_name = Some(sec_name);
                        let sec = doc.sections.entry(sec_name).or_insert_with(|| LtxSection {
                            name: sec_name,
                            parents: Vec::new(),
                            entries: HashMap::new(),
                        });

                        for p in parents {
                            if !sec.parents.contains(&p) {
                                sec.parents.push(p);
                            }
                        }
                        continue;
                    }
                }

                // Key-Value entry: key = value or !key = value (tombstone) or !key
                if let Some(sec_name) = current_sec_name {
                    let (is_delete, kv_slice) = if let Some(stripped) = clean.strip_prefix('!') {
                        (true, stripped.trim_start())
                    } else {
                        (false, clean)
                    };

                    let (key, val) = if let Some(eq_pos) = kv_slice.find('=') {
                        let k = kv_slice[..eq_pos].trim();
                        let v = if is_delete {
                            DLTX_DELETE
                        } else {
                            kv_slice[eq_pos + 1..].trim()
                        };
                        (k, v)
                    } else if is_delete {
                        (kv_slice.trim(), DLTX_DELETE)
                    } else {
                        continue;
                    };

                    if !key.is_empty() {
                        insertion_counter += 1;
                        raw_items.entry(sec_name).or_default().push(DltxItem {
                            key,
                            val,
                            depth,
                            insertion_index: insertion_counter,
                        });
                    }
                }
            }
        }

        // Materialize entries using S.T.A.L.K.E.R. Anomaly DLTX Precedence Comparator
        Self::materialize_raw_items(&mut doc, raw_items);

        doc
    }

    /// Merge a batch of DltxItems into doc according to DLTX precedence rules.
    pub fn materialize_raw_items(doc: &mut LtxDocument<'a>, raw_items: HashMap<&'a str, Vec<DltxItem<'a>>>) {
        for (sec_name, mut items) in raw_items {
            if doc.deleted_sections.contains(sec_name) {
                continue;
            }

            // S.T.A.L.K.E.R. Anomaly xr_ini.cpp sort order:
            // Key ASC, Depth ASC, Insertion Index DESC
            items.sort();

            if let Some(sec) = doc.sections.get_mut(sec_name) {
                let mut last_key = "";
                for item in items {
                    if item.key == last_key {
                        // Skip duplicate; the first candidate according to sort order won
                        continue;
                    }
                    last_key = item.key;

                    // Tombstone check: DLTX_DELETE suppresses key completely from materialized section
                    if item.val != DLTX_DELETE {
                        sec.entries.insert(item.key, item.val);
                    } else {
                        sec.entries.remove(item.key);
                    }
                }
            }
        }
    }

    /// Read string value with multi-parent section inheritance resolution.
    pub fn read_str(&self, section: &str, key: &str) -> Option<&'a str> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(section);

        while let Some(cur) = queue.pop_front() {
            if !visited.insert(cur) {
                // Circular inheritance loop detected; safely terminate branch
                continue;
            }

            if let Some(sec) = self.sections.get(cur) {
                if let Some(val) = sec.entries.get(key) {
                    return Some(*val);
                }
                for parent in &sec.parents {
                    queue.push_back(*parent);
                }
            }
        }

        None
    }

    /// Read integer value with fallback.
    pub fn read_i32(&self, section: &str, key: &str, default: i32) -> i32 {
        self.read_str(section, key)
            .and_then(|s| {
                if s.starts_with("0x") || s.starts_with("0X") {
                    i32::from_str_radix(&s[2..], 16).ok()
                } else {
                    s.parse::<i32>().ok()
                }
            })
            .unwrap_or(default)
    }

    /// Read float value with fallback.
    pub fn read_f32(&self, section: &str, key: &str, default: f32) -> f32 {
        self.read_str(section, key)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(default)
    }

    /// Read boolean value (supports true/false, 1/0, on/off, yes/no).
    pub fn read_bool(&self, section: &str, key: &str, default: bool) -> bool {
        self.read_str(section, key)
            .map(|s| match s.to_ascii_lowercase().as_str() {
                "true" | "1" | "on" | "yes" => true,
                "false" | "0" | "off" | "no" => false,
                _ => default,
            })
            .unwrap_or(default)
    }

    /// Read comma-separated string list (trimmed, preserving non-empty elements).
    pub fn read_csv(&self, section: &str, key: &str) -> Option<Vec<&'a str>> {
        self.read_str(section, key).map(|s| {
            s.split(',')
                .map(|part| part.trim())
                .filter(|part| !part.is_empty())
                .collect()
        })
    }

    /// Read a 3D float vector (`x, y, z`).
    pub fn read_vec3(&self, section: &str, key: &str) -> Option<[f32; 3]> {
        let vals = self.read_csv(section, key)?;
        if vals.len() < 3 {
            return None;
        }
        Some([
            vals[0].parse::<f32>().ok()?,
            vals[1].parse::<f32>().ok()?,
            vals[2].parse::<f32>().ok()?,
        ])
    }

    /// Check whether `section` exists (and has not been deleted).
    pub fn contains_section(&self, section: &str) -> bool {
        self.sections.contains_key(section)
    }

    /// Return sorted section names in the materialized document.
    pub fn section_names(&self) -> Vec<&'a str> {
        let mut names: Vec<&'a str> = self.sections.keys().copied().collect();
        names.sort_unstable();
        names
    }

    /// Strip comments while preserving comments inside quotes.
    fn strip_comment(s: &'a str) -> &'a str {
        let mut in_quote = false;
        let bytes = s.as_bytes();
        let len = bytes.len();

        for i in 0..len {
            if bytes[i] == b'"' {
                in_quote = !in_quote;
            } else if !in_quote {
                if bytes[i] == b';' {
                    return s[..i].trim();
                }
                if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
                    return s[..i].trim();
                }
            }
        }
        s.trim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parsing() {
        let ltx = r#"
        ; Comment line
        [general]
        speed = 150
        gravity = 9.81 ; inline comment
        enabled = true

        [modifier]:general
        speed = 200
        extra_flag = 0x3F
        "#;

        let doc = LtxDocument::parse(ltx);
        assert_eq!(doc.read_i32("general", "speed", 0), 150);
        assert_eq!(doc.read_f32("general", "gravity", 0.0), 9.81);
        assert!(doc.read_bool("general", "enabled", false));

        // Inheritance check
        assert_eq!(doc.read_i32("modifier", "speed", 0), 200); // overridden
        assert_eq!(doc.read_f32("modifier", "gravity", 0.0), 9.81); // inherited from general
        assert_eq!(doc.read_i32("modifier", "extra_flag", 0), 0x3F); // hex value
    }

    #[test]
    fn test_dltx_precedence_deeper_vs_shallower() {
        // Shallower depth (e.g. -200 from mod) must beat deeper depth (e.g. 0 from base)
        let mut doc = LtxDocument::default();
        doc.sections.insert("wpn_ak74", LtxSection {
            name: "wpn_ak74",
            parents: vec![],
            entries: HashMap::new(),
        });

        let mut raw_items = HashMap::new();
        raw_items.insert(
            "wpn_ak74",
            vec![
                DltxItem {
                    key: "hit_power",
                    val: "0.45",
                    depth: 0, // base file depth
                    insertion_index: 10,
                },
                DltxItem {
                    key: "hit_power",
                    val: "0.65",
                    depth: -200, // mod file depth (lower wins)
                    insertion_index: 1,
                },
            ],
        );

        LtxDocument::materialize_raw_items(&mut doc, raw_items);
        assert_eq!(doc.read_str("wpn_ak74", "hit_power"), Some("0.65"));
    }

    #[test]
    fn test_dltx_same_depth_latest_wins() {
        let ltx = r#"
        [weapon]
        damage = 50
        damage = 75
        "#;
        let doc = LtxDocument::parse(ltx);
        assert_eq!(doc.read_i32("weapon", "damage", 0), 75);
    }

    #[test]
    fn test_dltx_tombstone_suppression_with_deeper_candidate() {
        // Winning tombstone (DLTX_DELETE) at lower depth must suppress key entirely
        // and NOT fall back to deeper non-delete candidate.
        let mut doc = LtxDocument::default();
        doc.sections.insert("outfit", LtxSection {
            name: "outfit",
            parents: vec![],
            entries: HashMap::new(),
        });

        let mut raw_items = HashMap::new();
        raw_items.insert(
            "outfit",
            vec![
                DltxItem {
                    key: "nightvision_sect",
                    val: "effector_nightvision_good",
                    depth: 0, // base file candidate
                    insertion_index: 5,
                },
                DltxItem {
                    key: "nightvision_sect",
                    val: DLTX_DELETE,
                    depth: -200, // mod deletes this key
                    insertion_index: 2,
                },
            ],
        );

        LtxDocument::materialize_raw_items(&mut doc, raw_items);
        assert_eq!(doc.read_str("outfit", "nightvision_sect"), None);
    }

    #[test]
    fn test_multi_parent_inheritance() {
        let ltx = r#"
        [parent_a]
        val_a = 10
        shared = from_a

        [parent_b]
        val_b = 20
        shared = from_b

        [child]:parent_a, parent_b
        child_val = 30
        "#;

        let doc = LtxDocument::parse(ltx);
        assert_eq!(doc.read_i32("child", "child_val", 0), 30);
        assert_eq!(doc.read_i32("child", "val_a", 0), 10);
        assert_eq!(doc.read_i32("child", "val_b", 0), 20);
        // parent_a visited before parent_b in queue order
        assert_eq!(doc.read_str("child", "shared"), Some("from_a"));
    }

    #[test]
    fn test_cycle_protection() {
        let ltx = r#"
        [node_a]:node_b
        key_a = 1

        [node_b]:node_a
        key_b = 2
        "#;

        let doc = LtxDocument::parse(ltx);
        assert_eq!(doc.read_i32("node_a", "key_a", 0), 1);
        assert_eq!(doc.read_i32("node_a", "key_b", 0), 2);
        // Does not infinite loop on non-existent key
        assert_eq!(doc.read_str("node_a", "missing"), None);
    }

    #[test]
    fn test_section_and_key_delete_syntax() {
        let ltx = r#"
        [keep_sec]
        k1 = hello
        !k2 = tombstone
        !k3

        !![deleted_sec]
        bad_val = 999
        "#;

        let doc = LtxDocument::parse(ltx);
        assert_eq!(doc.read_str("keep_sec", "k1"), Some("hello"));
        assert_eq!(doc.read_str("keep_sec", "k2"), None);
        assert_eq!(doc.read_str("keep_sec", "k3"), None);
        assert_eq!(doc.sections.get("deleted_sec"), None);
    }

    #[test]
    fn test_parse_sources_bom_hash_comments_and_vec3() {
        let base = "\u{feff}[wpn_ak74]\nhit_power = 0.45\nposition = 0.1, -0.25, 1.5\n# base comment = ignore_me\n";
        let mod_override = "# Fixes to Anomaly configs\n![wpn_ak74]\nhit_power = 0.72\n";
        let doc = LtxDocument::parse_sources(&[(0, base), (-200, mod_override)]);
        assert!(doc.contains_section("wpn_ak74"));
        assert_eq!(doc.read_f32("wpn_ak74", "hit_power", 0.0), 0.72);
        assert_eq!(doc.read_vec3("wpn_ak74", "position"), Some([0.1, -0.25, 1.5]));
        assert_eq!(doc.read_str("wpn_ak74", "# base comment"), None);
    }

    #[test]
    fn test_real_repo_ltx_configs_if_present() {
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let fixes = repo_root.join("gamedata/configs/mod_system_anomaly_fixes.ltx");
        let knife = repo_root.join("gamedata/configs/mod_system_knife.ltx");
        let parts = repo_root.join("gamedata/meshes/anomaly_weapons/hands/bone_parts.ltx");
        if fixes.is_file() && knife.is_file() && parts.is_file() {
            let fixes_txt = std::fs::read_to_string(fixes).unwrap();
            let knife_txt = std::fs::read_to_string(knife).unwrap();
            let parts_txt = std::fs::read_to_string(parts).unwrap();

            let merged = LtxDocument::parse_sources(&[
                (-200, &fixes_txt),
                (-200, &knife_txt),
                (0, &parts_txt),
            ]);
            assert_eq!(merged.read_f32("ammo_base", "k_dist", 0.0), 1.0);
            assert!(!merged.read_bool("wpn_knife", "firepos", true));
            assert_eq!(merged.read_str("part_1", "partition_name"), Some("left_hand"));
            assert_eq!(merged.read_str("part_0", "bip01"), Some(""));
        }
    }
}
