//! Multi-file `#include` & DLTX (`mod_*.ltx`) configuration loader for X-Ray `.ltx` trees.
//!
//! Resolves nested `#include "path.ltx"` and wildcard `#include "prefix_*.ltx"` directives
//! cleanly through an abstract reader/listing callback so `xray-core` remains zero-dependency
//! and works identically over both `std::fs` and `xray-vfs::VirtualFileSystem`.

use std::collections::{HashMap, HashSet, VecDeque};
use crate::LtxDocument;

/// Result of resolving an `#include` directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedLtxSource {
    pub virtual_path: String,
    pub depth: i32,
    pub text: String,
}

/// Abstract virtual filesystem interface for recursive `.ltx` loading.
pub trait LtxFileProvider {
    type Error: std::fmt::Debug;

    /// Read UTF-8/lossy text of an `.ltx` file at `virtual_path`.
    fn read_ltx_text(&self, virtual_path: &str) -> Result<String, Self::Error>;

    /// List virtual paths in `dir_prefix` matching a filename pattern (supports `*` wildcards).
    fn list_ltx_matches(
        &self,
        dir_prefix: &str,
        filename_pattern: &str,
    ) -> Result<Vec<String>, Self::Error>;
}

/// Extract `#include "..."` directives from an `.ltx` document in declaration order.
pub fn extract_includes(input: &str) -> Vec<&str> {
    let trimmed_bom = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut includes = Vec::new();
    for line in trimmed_bom.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#include") {
            let arg = rest.trim();
            if let Some(quoted) = arg.strip_prefix('"')
                && let Some(end_q) = quoted.find('"')
            {
                let path = quoted[..end_q].trim();
                if !path.is_empty() {
                    includes.push(path);
                }
            }
        }
    }
    includes
}

/// Match a case-insensitive filename against a simple `*` glob pattern (`materials_*.ltx`).
pub fn glob_matches_ascii(pattern: &str, candidate: &str) -> bool {
    let pat = pattern.to_ascii_lowercase();
    let cand = candidate.to_ascii_lowercase();
    let parts: Vec<&str> = pat.split('*').collect();
    if parts.len() == 1 {
        return cand == pat;
    }

    let mut pos = 0usize;
    for (idx, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if idx == 0 {
            if !cand.starts_with(part) {
                return false;
            }
            pos = part.len();
        } else if idx == parts.len() - 1 && !pat.ends_with('*') {
            return cand[pos..].ends_with(part);
        } else if let Some(found) = cand[pos..].find(part) {
            pos += found + part.len();
        } else {
            return false;
        }
    }
    true
}

/// Collect a root `.ltx` and its transitive `#include` graph (including wildcard includes),
/// sorted deterministically, preventing cycles.
pub fn collect_ltx_sources<P: LtxFileProvider>(
    provider: &P,
    root_path: &str,
    dltx_mod_files: &[&str],
) -> Result<Vec<LoadedLtxSource>, P::Error> {
    let mut visited = HashSet::new();
    let mut out = Vec::new();

    collect_recursive(provider, root_path, 0, &mut visited, &mut out)?;

    // Collect DLTX mod override files at negative precedence depth (-200),
    // sorted by name so mod precedence is deterministic across platforms.
    let mut sorted_mods: Vec<String> = dltx_mod_files
        .iter()
        .map(|s| normalize_rel_path(s))
        .collect();
    sorted_mods.sort_by_key(|s| s.to_ascii_lowercase());

    for mod_path in sorted_mods {
        let key = mod_path.to_ascii_lowercase();
        if !visited.insert(key) {
            continue;
        }
        let text = provider.read_ltx_text(&mod_path)?;
        out.push(LoadedLtxSource {
            virtual_path: mod_path,
            depth: -200,
            text,
        });
    }

    Ok(out)
}

fn collect_recursive<P: LtxFileProvider>(
    provider: &P,
    raw_path: &str,
    depth: i32,
    visited: &mut HashSet<String>,
    out: &mut Vec<LoadedLtxSource>,
) -> Result<(), P::Error> {
    let norm = normalize_rel_path(raw_path);
    let key = norm.to_ascii_lowercase();
    if !visited.insert(key) {
        return Ok(());
    }

    let text = provider.read_ltx_text(&norm)?;
    let parent_dir = match norm.rfind('\\') {
        Some(idx) => &norm[..=idx],
        None => "",
    };

    let includes: Vec<String> = extract_includes(&text)
        .into_iter()
        .map(|inc| join_rel_path(parent_dir, inc))
        .collect();

    out.push(LoadedLtxSource {
        virtual_path: norm,
        depth,
        text,
    });

    for inc_path in includes {
        if inc_path.contains('*') {
            let (inc_dir, pattern) = match inc_path.rfind('\\') {
                Some(idx) => (&inc_path[..=idx], &inc_path[idx + 1..]),
                None => ("", inc_path.as_str()),
            };
            let mut matches = provider.list_ltx_matches(inc_dir, pattern)?;
            matches.sort_by_key(|s| s.to_ascii_lowercase());
            for matched in matches {
                collect_recursive(provider, &matched, depth + 1, visited, out)?;
            }
        } else {
            collect_recursive(provider, &inc_path, depth + 1, visited, out)?;
        }
    }

    Ok(())
}

/// Owned, merged configuration document built from multiple `LoadedLtxSource`s with full DLTX rules.
#[derive(Debug, Clone, Default)]
pub struct OwnedLtxDocument {
    pub sections: HashMap<String, OwnedLtxSection>,
    pub deleted_sections: HashSet<String>,
}

#[derive(Debug, Clone, Default)]
pub struct OwnedLtxSection {
    pub name: String,
    pub parents: Vec<String>,
    pub entries: HashMap<String, String>,
}

impl OwnedLtxDocument {
    /// Merge multiple loaded LTX sources using `LtxDocument::parse_sources` and materialize
    /// an owned document independent of source buffer lifetimes.
    pub fn from_sources(sources: &[LoadedLtxSource]) -> Self {
        let pairs: Vec<(i32, &str)> = sources
            .iter()
            .map(|s| (s.depth, s.text.as_str()))
            .collect();
        let borrowed = LtxDocument::parse_sources(&pairs);

        let mut owned = Self {
            sections: HashMap::with_capacity(borrowed.sections.len()),
            deleted_sections: borrowed
                .deleted_sections
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        };

        for (sec_name, sec) in borrowed.sections {
            owned.sections.insert(
                sec_name.to_string(),
                OwnedLtxSection {
                    name: sec.name.to_string(),
                    parents: sec.parents.into_iter().map(|p| p.to_string()).collect(),
                    entries: sec
                        .entries
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect(),
                },
            );
        }

        owned
    }

    pub fn read_str(&self, section: &str, key: &str) -> Option<&str> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(section);

        while let Some(cur) = queue.pop_front() {
            if !visited.insert(cur) {
                continue;
            }
            if let Some(sec) = self.sections.get(cur) {
                if let Some(val) = sec.entries.get(key) {
                    return Some(val.as_str());
                }
                for parent in &sec.parents {
                    queue.push_back(parent.as_str());
                }
            }
        }
        None
    }

    pub fn read_f32(&self, section: &str, key: &str, default: f32) -> f32 {
        self.read_str(section, key)
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(default)
    }

    pub fn read_i32(&self, section: &str, key: &str, default: i32) -> i32 {
        self.read_str(section, key)
            .and_then(|s| {
                if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                    i32::from_str_radix(hex, 16).ok()
                } else {
                    s.parse::<i32>().ok()
                }
            })
            .unwrap_or(default)
    }

    pub fn read_bool(&self, section: &str, key: &str, default: bool) -> bool {
        self.read_str(section, key)
            .map(|s| match s.to_ascii_lowercase().as_str() {
                "true" | "1" | "on" | "yes" => true,
                "false" | "0" | "off" | "no" => false,
                _ => default,
            })
            .unwrap_or(default)
    }
}

fn normalize_rel_path(path: &str) -> String {
    path.trim()
        .replace('/', "\\")
        .trim_start_matches('\\')
        .to_string()
}

fn join_rel_path(base_dir: &str, child: &str) -> String {
    let clean_child = normalize_rel_path(child);
    if base_dir.is_empty() || clean_child.starts_with('$') {
        clean_child
    } else {
        format!("{base_dir}{clean_child}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    struct MemoryProvider(HashMap<String, String>);

    impl LtxFileProvider for MemoryProvider {
        type Error = Infallible;

        fn read_ltx_text(&self, virtual_path: &str) -> Result<String, Self::Error> {
            let key = virtual_path.replace('/', "\\").to_ascii_lowercase();
            Ok(self.0.get(&key).cloned().unwrap_or_default())
        }

        fn list_ltx_matches(
            &self,
            dir_prefix: &str,
            filename_pattern: &str,
        ) -> Result<Vec<String>, Self::Error> {
            let dir_low = dir_prefix.replace('/', "\\").to_ascii_lowercase();
            let mut out = Vec::new();
            for key in self.0.keys() {
                if let Some(rest) = key.strip_prefix(&dir_low)
                    && !rest.contains('\\')
                    && glob_matches_ascii(filename_pattern, rest)
                {
                    out.push(key.clone());
                }
            }
            Ok(out)
        }
    }

    #[test]
    fn test_extract_includes_and_glob() {
        let doc = "\u{feff}; comment\n#include \"materials_*.ltx\"\n#include \"weapons\\ak74.ltx\"\n";
        assert_eq!(
            extract_includes(doc),
            vec!["materials_*.ltx", "weapons\\ak74.ltx"]
        );
        assert!(glob_matches_ascii("materials_*.ltx", "materials_metal.ltx"));
        assert!(!glob_matches_ascii("materials_*.ltx", "other.ltx"));
    }

    #[test]
    fn test_recursive_include_and_dltx_mod_merge() {
        let mut files = HashMap::new();
        files.insert(
            "materials\\materials.ltx".to_string(),
            "#include \"materials_*.ltx\"\n[base_mtl]\nfriction = 0.5\n".to_string(),
        );
        files.insert(
            "materials\\materials_metal.ltx".to_string(),
            "[metal]:base_mtl\nfriction = 0.25\nflag_bounceable = true\n".to_string(),
        );
        files.insert(
            "materials\\mod_materials_overhaul.ltx".to_string(),
            "@[metal]\nfriction = 0.15\n".to_string(),
        );

        let provider = MemoryProvider(files);
        let sources = collect_ltx_sources(
            &provider,
            "materials\\materials.ltx",
            &["materials\\mod_materials_overhaul.ltx"],
        )
        .unwrap();

        assert_eq!(sources.len(), 3);
        let merged = OwnedLtxDocument::from_sources(&sources);
        assert_eq!(merged.read_str("metal", "friction"), Some("0.15"));
        assert_eq!(merged.read_str("metal", "flag_bounceable"), Some("true"));
    }
}
