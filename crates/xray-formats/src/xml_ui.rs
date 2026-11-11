//! X-Ray Monolith / S.T.A.L.K.E.R. Anomaly UI XML, Localization & Texture Atlas Parsers
//!
//! Mirrors `xrXMLParser` (`CXml`), `CUIXml`, `CUIXmlInit`, `CUITextureMaster`, and
//! `CStringTable`:
//! - Pure-Rust UTF-8 / Windows-1251 decoder (`decode_xray_text`, `decode_cp1251`)
//! - Pre-parse `#include "..."` & `*.xml` directory-mask preprocessor (`expand_xml_includes`)
//! - TinyXML-compatible DOM (`XmlDocument`, `XmlNode`) with `a:b:c` colon navigation, comments,
//!   CDATA sections, processing instructions, and XML/numeric entity unescaping
//! - Localization string table (`LocalizationTable`: `<string_table><string id="..."><text>...</text>`
//!   with English base fallback + `$$ACTION_*$$` binding expansion)
//! - UI atlas descriptors (`UiTextureAtlasMap`, `UiTextureRegion`: `textures_descr\*.xml`)
//! - Color table (`UiColorDefs`: `color_defs.xml`) and typed UI window/text/texture layout nodes
//!   (`UiWindowDef`, `UiTextDef`, `UiTextureRef`).

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::chunk::FormatError;

/// Canonical Microsoft Windows-1251 mapping for bytes `0x80..=0xBF`
/// (bytes `0x00..=0x7F` are ASCII identity; bytes `0xC0..=0xFF` map linearly to U+0410..=U+044F).
#[rustfmt::skip]
const CP1251_HIGH_HALF: [char; 64] = [
    // 0x80..=0x8F
    '\u{0402}', '\u{0403}', '\u{201A}', '\u{0453}', '\u{201E}', '\u{2026}', '\u{2020}', '\u{2021}',
    '\u{20AC}', '\u{2030}', '\u{0409}', '\u{2039}', '\u{040A}', '\u{040C}', '\u{040B}', '\u{040F}',
    // 0x90..=0x9F
    '\u{0452}', '\u{2018}', '\u{2019}', '\u{201C}', '\u{201D}', '\u{2022}', '\u{2013}', '\u{2014}',
    '\u{FFFD}', '\u{2122}', '\u{0459}', '\u{203A}', '\u{045A}', '\u{045C}', '\u{045B}', '\u{045F}',
    // 0xA0..=0xAF
    '\u{00A0}', '\u{040E}', '\u{045E}', '\u{0408}', '\u{00A4}', '\u{0490}', '\u{00A6}', '\u{00A7}',
    '\u{0401}', '\u{00A9}', '\u{0404}', '\u{00AB}', '\u{00AC}', '\u{00AD}', '\u{00AE}', '\u{0407}',
    // 0xB0..=0xBF
    '\u{00B0}', '\u{00B1}', '\u{0406}', '\u{0456}', '\u{0491}', '\u{00B5}', '\u{00B6}', '\u{00B7}',
    '\u{0451}', '\u{2116}', '\u{0454}', '\u{00BB}', '\u{0458}', '\u{0405}', '\u{0455}', '\u{0457}',
];

/// Decode raw Windows-1251 (CP1251) bytes into a Rust UTF-8 `String`.
pub fn decode_cp1251(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        match b {
            0x00..=0x7F => out.push(b as char),
            0x80..=0xBF => out.push(CP1251_HIGH_HALF[(b - 0x80) as usize]),
            0xC0..=0xFF => {
                let code = 0x0410u32 + u32::from(b - 0xC0);
                out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
            }
        }
    }
    out
}

/// Decode an X-Ray XML / config text byte buffer into UTF-8.
///
/// Handles UTF-8 BOM (`EF BB BF`), native UTF-8 mod files (including files that keep a stale
/// `encoding="windows-1251"` header), and true Windows-1251 (CP1251) XML files.
pub fn decode_xray_text(bytes: &[u8]) -> String {
    if let Some(rest) = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]) {
        if let Ok(s) = std::str::from_utf8(rest) {
            return s.to_owned();
        }
        return String::from_utf8_lossy(rest).into_owned();
    }

    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_owned();
    }

    decode_cp1251(bytes)
}

/// Abstracts file reading and wildcard listing for `#include "..."` expansion (`ParseFile`).
pub trait XmlFileProvider {
    type Error: std::fmt::Debug;

    /// Read raw XML bytes from the provider (VFS path or relative path).
    fn read_xml_bytes(&self, virtual_path: &str) -> Result<Vec<u8>, Self::Error>;

    /// List virtual paths in `dir_prefix` matching a filename glob (e.g. `"*.xml"`).
    fn list_xml_matches(
        &self,
        dir_prefix: &str,
        filename_pattern: &str,
    ) -> Result<Vec<String>, Self::Error>;
}

const MAX_XML_INCLUDE_DEPTH: usize = 32;

fn split_parent_dir(vpath: &str) -> &str {
    match vpath.rfind(['\\', '/']) {
        Some(idx) => &vpath[..idx],
        None => "",
    }
}

fn join_virtual_path(dir: &str, rel: &str) -> String {
    let clean_dir = dir.trim_matches(['\\', '/']);
    let clean_rel = rel.trim_start_matches(['\\', '/']).replace('/', "\\");
    if clean_dir.is_empty() {
        clean_rel
    } else if clean_rel.is_empty() {
        clean_dir.to_string()
    } else {
        format!("{clean_dir}\\{clean_rel}")
    }
}

fn canonical_include_key(vpath: &str) -> String {
    vpath.replace('/', "\\").to_ascii_uppercase()
}

fn parse_include_directive(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let after_hash = trimmed.strip_prefix('#')?;
    let after_kw = after_hash.trim_start();
    if after_kw.len() < 7 || !after_kw[..7].eq_ignore_ascii_case("include") {
        return None;
    }
    let after_kw = &after_kw[7..];
    if !after_kw.is_empty() && !after_kw.as_bytes()[0].is_ascii_whitespace() {
        return None;
    }
    let q1 = trimmed.find('"')?;
    let rest = &trimmed[q1 + 1..];
    let q2 = rest.find('"')?;
    Some(rest[..q2].trim())
}

fn read_include_candidate<P: XmlFileProvider>(
    provider: &P,
    base_dir: &str,
    current_dir: &str,
    inc_path: &str,
) -> Result<(String, Vec<u8>), FormatError> {
    let normalized_inc = inc_path.replace('/', "\\");
    let primary = join_virtual_path(base_dir, &normalized_inc);
    if let Ok(bytes) = provider.read_xml_bytes(&primary) {
        return Ok((primary, bytes));
    }

    if !current_dir.eq_ignore_ascii_case(base_dir) {
        let fallback = join_virtual_path(current_dir, &normalized_inc);
        if let Ok(bytes) = provider.read_xml_bytes(&fallback) {
            return Ok((fallback, bytes));
        }
    }

    if let Ok(bytes) = provider.read_xml_bytes(&normalized_inc) {
        return Ok((normalized_inc, bytes));
    }

    Err(FormatError::InvalidValue("xml #include target not found"))
}

fn expand_recursive<P: XmlFileProvider>(
    provider: &P,
    base_dir: &str,
    current_file: &str,
    out: &mut String,
    active_stack: &mut HashSet<String>,
    depth: usize,
) -> Result<(), FormatError> {
    if depth >= MAX_XML_INCLUDE_DEPTH {
        return Err(FormatError::InvalidValue("xml #include depth exceeded"));
    }

    let key = canonical_include_key(current_file);
    if !active_stack.insert(key.clone()) {
        return Err(FormatError::InvalidValue("cyclic xml #include detected"));
    }

    let raw_bytes = provider
        .read_xml_bytes(current_file)
        .map_err(|_| FormatError::InvalidValue("xml file not found"))?;
    let text = decode_xray_text(&raw_bytes);
    let current_dir = split_parent_dir(current_file);

    for raw_line in text.lines() {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if let Some(inc) = parse_include_directive(line) {
            if inc.contains('*') || inc.contains('?') {
                let inc_norm = inc.replace('/', "\\");
                let (inc_sub_dir, pattern) = match inc_norm.rfind('\\') {
                    Some(idx) => (&inc_norm[..idx], &inc_norm[idx + 1..]),
                    None => ("", inc_norm.as_str()),
                };
                let search_dir = join_virtual_path(base_dir, inc_sub_dir);
                let mut matches = provider
                    .list_xml_matches(&search_dir, pattern)
                    .unwrap_or_default();
                if matches.is_empty() && !current_dir.eq_ignore_ascii_case(base_dir) {
                    let alt_dir = join_virtual_path(current_dir, inc_sub_dir);
                    matches = provider
                        .list_xml_matches(&alt_dir, pattern)
                        .unwrap_or_default();
                }
                matches.sort_by_key(|p| p.to_ascii_lowercase());
                for matched in matches {
                    expand_recursive(
                        provider,
                        base_dir,
                        &matched,
                        out,
                        active_stack,
                        depth + 1,
                    )?;
                }
            } else {
                let (resolved_path, _) =
                    read_include_candidate(provider, base_dir, current_dir, inc)?;
                expand_recursive(
                    provider,
                    base_dir,
                    &resolved_path,
                    out,
                    active_stack,
                    depth + 1,
                )?;
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }

    active_stack.remove(&key);
    Ok(())
}

/// Expand all `#include "..."` directives (matching `ParseFile` in `xrXMLParser.cpp`)
/// starting at `root_vpath`, resolving includes relative to `base_dir` (usually `"configs"`).
pub fn expand_xml_includes<P: XmlFileProvider>(
    provider: &P,
    base_dir: &str,
    root_vpath: &str,
) -> Result<String, FormatError> {
    let mut out = String::new();
    let mut active = HashSet::new();
    expand_recursive(provider, base_dir, root_vpath, &mut out, &mut active, 0)?;
    Ok(out)
}

/// Single DOM element node inside an `XmlDocument`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct XmlNode {
    pub name: String,
    pub attributes: Vec<(String, String)>,
    pub text: String,
    pub children: Vec<XmlNode>,
}

impl XmlNode {
    /// Look up an attribute value by exact name (`TiXmlElement::Attribute`).
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    pub fn attr_or<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.attr(name).unwrap_or(default)
    }

    pub fn attr_i32(&self, name: &str, default: i32) -> i32 {
        match self.attr(name) {
            Some(v) => parse_c_int(v, default),
            None => default,
        }
    }

    pub fn attr_f32(&self, name: &str, default: f32) -> f32 {
        match self.attr(name) {
            Some(v) => parse_c_float(v, default),
            None => default,
        }
    }

    pub fn attr_bool(&self, name: &str, default: bool) -> bool {
        match self.attr(name) {
            Some(raw) => {
                let t = raw.trim();
                if t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes") || t == "1" {
                    true
                } else if t.eq_ignore_ascii_case("false") || t.eq_ignore_ascii_case("no") || t == "0" {
                    false
                } else {
                    parse_c_int(t, i32::from(default)) != 0
                }
            }
            None => default,
        }
    }

    /// Iterate direct child elements with the given tag name.
    pub fn children_named<'a, 'b>(
        &'a self,
        tag_name: &'b str,
    ) -> impl Iterator<Item = &'a XmlNode> + use<'a, 'b> {
        self.children.iter().filter(move |c| c.name == tag_name)
    }

    /// Count direct children (all children when `tag_name` is `None`, else matching tags).
    pub fn nodes_num(&self, tag_name: Option<&str>) -> usize {
        match tag_name {
            Some(tag) => self.children.iter().filter(|c| c.name == tag).count(),
            None => self.children.len(),
        }
    }

    /// Navigate colon-separated path (`"parent:child:grandchild"`), applying `node_index`
    /// to the first path component (exact `CXml::NavigateToNode` semantics).
    pub fn navigate(&self, path: &str, node_index: usize) -> Option<&XmlNode> {
        if path.is_empty() {
            return Some(self);
        }
        let mut tokens = path.split(':').filter(|s| !s.is_empty());
        let first = tokens.next()?;
        let mut current = self.children_named(first).nth(node_index)?;
        for seg in tokens {
            current = current.children_named(seg).next()?;
        }
        Some(current)
    }

    /// Return the direct text content if non-empty, or `None` (`CXml::Read`).
    pub fn read_text(&self) -> Option<&str> {
        if self.text.is_empty() {
            None
        } else {
            Some(&self.text)
        }
    }

    pub fn read_i32(&self, default: i32) -> i32 {
        match self.read_text() {
            Some(s) => parse_c_int(s, default),
            None => default,
        }
    }

    pub fn read_f32(&self, default: f32) -> f32 {
        match self.read_text() {
            Some(s) => parse_c_float(s, default),
            None => default,
        }
    }
}

/// Parsed X-Ray XML document mirroring `CXml` (`src/xrXMLParser/xrXMLParser.cpp`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct XmlDocument {
    pub roots: Vec<XmlNode>,
}

impl XmlDocument {
    pub fn parse_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        let text = decode_xray_text(bytes);
        Self::parse(&text)
    }

    pub fn parse(xml: &str) -> Result<Self, FormatError> {
        let mut parser = XmlParser::new(xml);
        parser.parse_document()
    }

    pub fn load_with_provider<P: XmlFileProvider>(
        provider: &P,
        base_dir: &str,
        root_vpath: &str,
    ) -> Result<Self, FormatError> {
        let expanded = expand_xml_includes(provider, base_dir, root_vpath)?;
        Self::parse(&expanded)
    }

    /// Primary root element (`CXml::GetRoot` = `FirstChildElement`).
    pub fn root(&self) -> Option<&XmlNode> {
        self.roots.first()
    }

    /// Navigate from the primary root element (`CXml::NavigateToNode(path, index)`).
    ///
    /// Also checks direct root matching if the first segment matches a secondary root in merged XML.
    pub fn navigate(&self, path: &str, node_index: usize) -> Option<&XmlNode> {
        if path.is_empty() {
            return self.root();
        }
        for root in &self.roots {
            if let Some(found) = root.navigate(path, node_index) {
                return Some(found);
            }
        }
        None
    }

    /// Navigate to a direct child `tag_name` of `root` whose attribute `attrib` equals `value`
    /// (`CXml::NavigateToNodeWithAttribute`).
    pub fn navigate_with_attribute(
        &self,
        tag_name: &str,
        attrib: &str,
        value: &str,
    ) -> Option<&XmlNode> {
        for root in &self.roots {
            for child in root.children_named(tag_name) {
                if child.attr(attrib) == Some(value) {
                    return Some(child);
                }
            }
        }
        None
    }

    pub fn nodes_num(&self, path: &str, index: usize, tag_name: Option<&str>) -> usize {
        if path.is_empty() {
            return self.root().map_or(0, |r| r.nodes_num(tag_name));
        }
        self.navigate(path, index)
            .or_else(|| self.root())
            .map_or(0, |n| n.nodes_num(tag_name))
    }

    pub fn read_str<'a>(&'a self, path: &str, index: usize, default: &'a str) -> &'a str {
        self.navigate(path, index)
            .and_then(XmlNode::read_text)
            .unwrap_or(default)
    }

    pub fn read_i32(&self, path: &str, index: usize, default: i32) -> i32 {
        self.navigate(path, index)
            .map_or(default, |n| n.read_i32(default))
    }

    pub fn read_f32(&self, path: &str, index: usize, default: f32) -> f32 {
        self.navigate(path, index)
            .map_or(default, |n| n.read_f32(default))
    }

    pub fn read_attrib<'a>(
        &'a self,
        path: &str,
        index: usize,
        attrib: &str,
        default: &'a str,
    ) -> &'a str {
        self.navigate(path, index)
            .and_then(|n| n.attr(attrib))
            .unwrap_or(default)
    }

    pub fn read_attrib_i32(&self, path: &str, index: usize, attrib: &str, default: i32) -> i32 {
        self.navigate(path, index)
            .map_or(default, |n| n.attr_i32(attrib, default))
    }

    pub fn read_attrib_f32(&self, path: &str, index: usize, attrib: &str, default: f32) -> f32 {
        self.navigate(path, index)
            .map_or(default, |n| n.attr_f32(attrib, default))
    }
}

fn parse_c_int(raw: &str, default: i32) -> i32 {
    let s = raw.trim();
    if s.is_empty() {
        return default;
    }
    if let Ok(v) = s.parse::<i32>() {
        return v;
    }
    // `atoi`-like prefix scan for C compatibility
    let bytes = s.as_bytes();
    let mut end = if matches!(bytes[0], b'+' | b'-') { 1 } else { 0 };
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 || (end == 1 && matches!(bytes[0], b'+' | b'-')) {
        default
    } else {
        s[..end].parse::<i32>().unwrap_or(default)
    }
}

fn parse_c_float(raw: &str, default: f32) -> f32 {
    let s = raw.trim();
    if s.is_empty() {
        return default;
    }
    s.parse::<f32>().unwrap_or(default)
}

fn unescape_xml_entities(input: &str) -> String {
    if !input.contains('&') {
        return input.to_owned();
    }
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(amp_idx) = rest.find('&') {
        out.push_str(&rest[..amp_idx]);
        let after_amp = &rest[amp_idx + 1..];
        if let Some(semi_idx) = after_amp.find(';') {
            let entity = &after_amp[..semi_idx];
            if let Some(ch) = decode_entity(entity) {
                out.push(ch);
                rest = &after_amp[semi_idx + 1..];
                continue;
            }
        }
        out.push('&');
        rest = after_amp;
    }
    out.push_str(rest);
    out
}

fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "lt" => Some('<'),
        "gt" => Some('>'),
        "amp" => Some('&'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        _ => {
            let num = entity.strip_prefix('#')?;
            let code = if let Some(hex) = num.strip_prefix(['x', 'X']) {
                u32::from_str_radix(hex, 16).ok()?
            } else {
                num.parse::<u32>().ok()?
            };
            char::from_u32(code)
        }
    }
}

struct XmlParser<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> XmlParser<'a> {
    fn new(src: &'a str) -> Self {
        let start = src.strip_prefix('\u{FEFF}').map_or(0, |_| '\u{FEFF}'.len_utf8());
        Self { src, pos: start }
    }

    fn remaining(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.src.as_bytes().get(self.pos) {
            if b.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) -> Result<(), FormatError> {
        // Assumes remaining starts with `<!--`
        self.pos += 4;
        if let Some(rel) = self.remaining().find("-->") {
            self.pos += rel + 3;
            Ok(())
        } else {
            Err(FormatError::InvalidValue("unterminated XML comment"))
        }
    }

    fn skip_pi(&mut self) -> Result<(), FormatError> {
        // Assumes remaining starts with `<?`
        self.pos += 2;
        if let Some(rel) = self.remaining().find("?>") {
            self.pos += rel + 2;
            Ok(())
        } else {
            Err(FormatError::InvalidValue("unterminated XML processing instruction"))
        }
    }

    fn skip_doctype(&mut self) -> Result<(), FormatError> {
        // Assumes remaining starts with `<!` (not `--` or `[CDATA[`)
        self.pos += 2;
        let mut depth = 0usize;
        while !self.is_eof() {
            let b = self.src.as_bytes()[self.pos];
            self.pos += 1;
            match b {
                b'[' => depth += 1,
                b']' => depth = depth.saturating_sub(1),
                b'>' if depth == 0 => return Ok(()),
                _ => {}
            }
        }
        Err(FormatError::InvalidValue("unterminated DOCTYPE / declaration"))
    }

    fn read_cdata(&mut self) -> Result<&'a str, FormatError> {
        // Assumes remaining starts with `<![CDATA[`
        self.pos += 9;
        if let Some(rel) = self.remaining().find("]]>") {
            let slice = &self.src[self.pos..self.pos + rel];
            self.pos += rel + 3;
            Ok(slice)
        } else {
            Err(FormatError::InvalidValue("unterminated XML CDATA section"))
        }
    }

    fn read_name(&mut self) -> Option<&'a str> {
        let start = self.pos;
        while let Some(&b) = self.src.as_bytes().get(self.pos) {
            if b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos > start {
            Some(&self.src[start..self.pos])
        } else {
            None
        }
    }

    fn parse_document(&mut self) -> Result<XmlDocument, FormatError> {
        let mut roots = Vec::new();
        loop {
            self.skip_ws();
            if self.is_eof() {
                break;
            }
            let rem = self.remaining();
            if rem.starts_with("<!--") {
                self.skip_comment()?;
            } else if rem.starts_with("<?") {
                self.skip_pi()?;
            } else if rem.starts_with("<!") {
                self.skip_doctype()?;
            } else if rem.starts_with('<') && !rem.starts_with("</") {
                roots.push(self.parse_element()?);
            } else if let Some(next_lt) = rem.find('<') {
                // Skip stray top-level text before/after root elements (e.g. from included comments)
                let skipped = rem[..next_lt].trim();
                if !skipped.is_empty() && !rem[..next_lt].starts_with('#') {
                    return Err(FormatError::InvalidValue("unexpected top-level XML text"));
                }
                self.pos += next_lt;
            } else {
                break;
            }
        }

        if roots.is_empty() {
            return Err(FormatError::InvalidValue("no root XML element found"));
        }
        Ok(XmlDocument { roots })
    }

    fn parse_element(&mut self) -> Result<XmlNode, FormatError> {
        if !self.remaining().starts_with('<') {
            return Err(FormatError::InvalidValue("expected '<' opening XML tag"));
        }
        self.pos += 1;

        let name = self
            .read_name()
            .ok_or(FormatError::InvalidValue("empty XML element name"))?
            .to_owned();

        let mut attributes = Vec::new();
        let self_closing = loop {
            self.skip_ws();
            let rem = self.remaining();
            if rem.starts_with("/>") {
                self.pos += 2;
                break true;
            }
            if rem.starts_with('>') {
                self.pos += 1;
                break false;
            }
            if rem.is_empty() {
                return Err(FormatError::UnexpectedEof);
            }

            let attr_name = self
                .read_name()
                .ok_or(FormatError::InvalidValue("invalid XML attribute name"))?
                .to_owned();
            self.skip_ws();
            if !self.remaining().starts_with('=') {
                return Err(FormatError::InvalidValue("expected '=' after XML attribute"));
            }
            self.pos += 1;
            self.skip_ws();
            let quote = self
                .src
                .as_bytes()
                .get(self.pos)
                .copied()
                .filter(|&c| c == b'"' || c == b'\'')
                .ok_or(FormatError::InvalidValue("expected quoted XML attribute value"))?;
            self.pos += 1;
            let val_rel = self
                .remaining()
                .find(quote as char)
                .ok_or(FormatError::InvalidValue("unterminated XML attribute quote"))?;
            let raw_val = &self.src[self.pos..self.pos + val_rel];
            self.pos += val_rel + 1;
            attributes.push((attr_name, unescape_xml_entities(raw_val)));
        };

        if self_closing {
            return Ok(XmlNode {
                name,
                attributes,
                text: String::new(),
                children: Vec::new(),
            });
        }

        let mut children = Vec::new();
        let mut raw_text = String::new();

        loop {
            if self.is_eof() {
                return Err(FormatError::InvalidValue("unclosed XML element before EOF"));
            }
            let rem = self.remaining();
            if rem.starts_with("</") {
                self.pos += 2;
                let close_name = self
                    .read_name()
                    .ok_or(FormatError::InvalidValue("invalid XML closing tag"))?;
                if close_name != name {
                    return Err(FormatError::InvalidValue("mismatched XML closing tag"));
                }
                self.skip_ws();
                if !self.remaining().starts_with('>') {
                    return Err(FormatError::InvalidValue("expected '>' closing XML tag"));
                }
                self.pos += 1;
                break;
            } else if rem.starts_with("<!--") {
                self.skip_comment()?;
            } else if rem.starts_with("<?") {
                self.skip_pi()?;
            } else if rem.starts_with("<![CDATA[") {
                raw_text.push_str(self.read_cdata()?);
            } else if rem.starts_with("<!") {
                self.skip_doctype()?;
            } else if rem.starts_with('<') {
                children.push(self.parse_element()?);
            } else {
                let text_end = rem.find('<').unwrap_or(rem.len());
                raw_text.push_str(&unescape_xml_entities(&rem[..text_end]));
                self.pos += text_end;
            }
        }

        // Condense pure formatting whitespace when empty or when child elements exist
        let text = if raw_text.trim().is_empty() && (!children.is_empty() || raw_text.contains('\n')) {
            String::new()
        } else {
            raw_text
        };

        Ok(XmlNode {
            name,
            attributes,
            text,
            children,
        })
    }
}

/// Replace `$$ACTION_<name>$$` placeholders in a localized string (`CStringTable::ParseLine`).
pub fn expand_action_placeholders<F>(input: &str, mut resolve_action: F) -> String
where
    F: FnMut(&str) -> Option<String>,
{
    const PREFIX: &str = "$$ACTION_";
    if !input.contains(PREFIX) {
        return input.to_owned();
    }

    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;
    while let Some(rel_start) = input[cursor..].find(PREFIX) {
        let start = cursor + rel_start;
        out.push_str(&input[cursor..start]);
        let after_prefix = start + PREFIX.len();
        if let Some(rel_end) = input[after_prefix..].find("$$") {
            let action_name = &input[after_prefix..after_prefix + rel_end];
            if let Some(replacement) = resolve_action(action_name) {
                out.push_str(&replacement);
            } else {
                out.push_str(&input[start..after_prefix + rel_end + 2]);
            }
            cursor = after_prefix + rel_end + 2;
        } else {
            out.push_str(&input[start..]);
            cursor = input.len();
            break;
        }
    }
    if cursor < input.len() {
        out.push_str(&input[cursor..]);
    }
    out
}

/// Unescape literal `\\n` two-character sequences used inside X-Ray XML `<text>` entries.
pub fn unescape_xray_newlines(text: &str) -> String {
    if text.contains("\\n") {
        text.replace("\\n", "\n")
    } else {
        text.to_owned()
    }
}

/// X-Ray XML Localization Table (`CStringTable`: `configs\text\<lang>\*.xml`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LocalizationTable {
    pub language: String,
    pub entries: BTreeMap<String, String>,
    pub raw_key_binding_ids: BTreeSet<String>,
}

impl LocalizationTable {
    pub fn new(language: &str) -> Self {
        Self {
            language: language.to_owned(),
            entries: BTreeMap::new(),
            raw_key_binding_ids: BTreeSet::new(),
        }
    }

    /// Merge `<string id="..."><text>...</text></string>` entries from a decoded XML `XmlDocument`.
    pub fn merge_doc(&mut self, doc: &XmlDocument) -> usize {
        let mut count = 0usize;
        for root in &doc.roots {
            for string_node in root.children_named("string") {
                let Some(id) = string_node.attr("id") else {
                    continue;
                };
                let text = string_node
                    .children_named("text")
                    .next()
                    .map_or("", |n| n.text.as_str());
                if text.contains("$$ACTION_") {
                    self.raw_key_binding_ids.insert(id.to_owned());
                } else {
                    self.raw_key_binding_ids.remove(id);
                }
                self.entries.insert(id.to_owned(), text.to_owned());
                count += 1;
            }
        }
        count
    }

    pub fn merge_xml_bytes(&mut self, bytes: &[u8]) -> Result<usize, FormatError> {
        let doc = XmlDocument::parse_bytes(bytes)?;
        Ok(self.merge_doc(&doc))
    }

    pub fn merge_xml_str(&mut self, xml: &str) -> Result<usize, FormatError> {
        let doc = XmlDocument::parse(xml)?;
        Ok(self.merge_doc(&doc))
    }

    pub fn from_xml_bytes(language: &str, bytes: &[u8]) -> Result<Self, FormatError> {
        let mut table = Self::new(language);
        table.merge_xml_bytes(bytes)?;
        Ok(table)
    }

    /// Look up raw localization value; falls back to `id` (`CStringTable::translate`).
    pub fn translate<'a>(&'a self, id: &'a str) -> &'a str {
        self.entries.get(id).map_or(id, String::as_str)
    }

    pub fn get(&self, id: &str) -> Option<&str> {
        self.entries.get(id).map(String::as_str)
    }

    /// Translate and substitute `$$ACTION_<name>$$` bindings + `\n` newlines.
    pub fn translate_formatted(
        &self,
        id: &str,
        key_bindings: &BTreeMap<String, String>,
    ) -> String {
        let raw = self.translate(id);
        let with_keys = expand_action_placeholders(raw, |k| key_bindings.get(k).cloned());
        unescape_xray_newlines(&with_keys)
    }
}

/// Rectangular texture region inside a UI atlas sheet (`TEX_INFO`, `UITextureMaster.cpp`).
#[derive(Debug, Clone, PartialEq)]
pub struct UiTextureRegion {
    pub id: String,
    pub atlas_file: String,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub border_l: i16,
    pub border_t: i16,
    pub border_r: i16,
    pub border_b: i16,
}

impl UiTextureRegion {
    /// Pixel bounds `[x1, y1, x2, y2]` (`Frect` in X-Ray).
    pub fn pixel_rect(&self) -> [f32; 4] {
        [self.x, self.y, self.x + self.width, self.y + self.height]
    }

    /// Normalized `[u1, v1, u2, v2]` texture coordinates given atlas texture dimensions.
    pub fn uv_rect(&self, atlas_width: f32, atlas_height: f32) -> [f32; 4] {
        let iw = if atlas_width != 0.0 { 1.0 / atlas_width } else { 0.0 };
        let ih = if atlas_height != 0.0 { 1.0 / atlas_height } else { 0.0 };
        [
            self.x * iw,
            self.y * ih,
            (self.x + self.width) * iw,
            (self.y + self.height) * ih,
        ]
    }

    /// Corresponding VFS `.dds` path (`textures\<atlas_file>.dds`).
    pub fn dds_vpath(&self) -> String {
        let clean = self.atlas_file.replace('/', "\\");
        let with_ext = if clean.len() >= 4
            && clean[clean.len() - 4..].eq_ignore_ascii_case(".dds")
        {
            clean
        } else {
            format!("{clean}.dds")
        };
        if with_ext.to_ascii_lowercase().starts_with("textures\\") {
            with_ext
        } else {
            format!("textures\\{with_ext}")
        }
    }
}

/// UI shared texture atlas registry (`CUITextureMaster::ParseShTexInfo`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UiTextureAtlasMap {
    pub regions: BTreeMap<String, UiTextureRegion>,
}

impl UiTextureAtlasMap {
    pub fn merge_doc(&mut self, doc: &XmlDocument) -> usize {
        let mut count = 0usize;
        for root in &doc.roots {
            for file_node in root.children_named("file") {
                let atlas_file = file_node.attr_or("name", "");
                for tex_node in file_node.children_named("texture") {
                    let Some(id) = tex_node.attr("id") else {
                        continue;
                    };
                    let region = UiTextureRegion {
                        id: id.to_owned(),
                        atlas_file: atlas_file.to_owned(),
                        x: tex_node.attr_f32("x", 0.0),
                        y: tex_node.attr_f32("y", 0.0),
                        width: tex_node.attr_f32("width", 0.0),
                        height: tex_node.attr_f32("height", 0.0),
                        border_l: tex_node.attr_i32("border_l", 0) as i16,
                        border_t: tex_node.attr_i32("border_t", 0) as i16,
                        border_r: tex_node.attr_i32("border_r", 0) as i16,
                        border_b: tex_node.attr_i32("border_b", 0) as i16,
                    };
                    self.regions.insert(id.to_owned(), region);
                    count += 1;
                }
            }
        }
        count
    }

    pub fn from_xml(xml: &str) -> Result<Self, FormatError> {
        let doc = XmlDocument::parse(xml)?;
        let mut map = Self::default();
        map.merge_doc(&doc);
        Ok(map)
    }

    pub fn get(&self, id: &str) -> Option<&UiTextureRegion> {
        self.regions.get(id)
    }

    /// `CUITextureMaster::IsSh`: shared atlas IDs contain no path separator (`\` or `/`).
    pub fn is_shared_region(texture_name: &str) -> bool {
        !texture_name.contains('\\') && !texture_name.contains('/')
    }
}

/// Pack 8-bit components into X-Ray `u32` `color_argb(a, r, g, b)`.
pub const fn color_argb(a: u8, r: u8, g: u8, b: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Unpack X-Ray `color_argb` into normalized `[r, g, b, a]` floats in `0.0..=1.0`.
pub fn argb_to_rgba_f32(argb: u32) -> [f32; 4] {
    let a = ((argb >> 24) & 0xFF) as f32 / 255.0;
    let r = ((argb >> 16) & 0xFF) as f32 / 255.0;
    let g = ((argb >> 8) & 0xFF) as f32 / 255.0;
    let b = (argb & 0xFF) as f32 / 255.0;
    [r, g, b, a]
}

/// Named UI colors parsed from `configs\ui\color_defs.xml` (`CUIXmlInit::InitColorDefs`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UiColorDefs {
    pub colors: BTreeMap<String, u32>,
}

impl UiColorDefs {
    pub fn merge_doc(&mut self, doc: &XmlDocument) -> usize {
        let mut count = 0usize;
        let container = doc
            .navigate("colors", 0)
            .or_else(|| doc.root());
        if let Some(colors_node) = container {
            for cnode in colors_node.children_named("color") {
                let name = cnode.attr_or("name", "");
                if name.is_empty() {
                    continue;
                }
                let r = cnode.attr_i32("r", 0).clamp(0, 255) as u8;
                let g = cnode.attr_i32("g", 0).clamp(0, 255) as u8;
                let b = cnode.attr_i32("b", 0).clamp(0, 255) as u8;
                let a = cnode.attr_i32("a", 255).clamp(0, 255) as u8;
                self.colors.insert(name.to_owned(), color_argb(a, r, g, b));
                count += 1;
            }
        }
        count
    }

    pub fn from_doc(doc: &XmlDocument) -> Self {
        let mut defs = Self::default();
        defs.merge_doc(doc);
        defs
    }

    /// Resolve a color attribute (`color="..."` named color OR `r/g/b/a` integers matching `CUIXmlInit::GetColor`).
    pub fn resolve_node_color(&self, node: &XmlNode, default_rgb_channel: u8) -> u32 {
        if let Some(name) = node.attr("color")
            && let Some(&clr) = self.colors.get(name)
        {
            return clr;
        }
        let d = i32::from(default_rgb_channel);
        let r = node.attr_i32("r", d).clamp(0, 255) as u8;
        let g = node.attr_i32("g", d).clamp(0, 255) as u8;
        let b = node.attr_i32("b", d).clamp(0, 255) as u8;
        let a = node.attr_i32("a", 255).clamp(0, 255) as u8;
        color_argb(a, r, g, b)
    }
}

/// Parsed `<text>` child definition inside a UI XML control (`CUIXmlInit::InitText`).
#[derive(Debug, Clone, PartialEq)]
pub struct UiTextDef {
    pub font: Option<String>,
    pub color_argb: u32,
    pub align: String,
    pub vert_align: String,
    pub complex_mode: bool,
    pub offset: [f32; 2],
    pub text: String,
}

impl UiTextDef {
    pub fn from_xml_node(node: &XmlNode, colors: &UiColorDefs) -> Self {
        Self {
            font: node.attr("font").map(str::to_owned),
            color_argb: colors.resolve_node_color(node, 255),
            align: node.attr_or("align", "").to_owned(),
            vert_align: node.attr_or("vert_align", "").to_owned(),
            complex_mode: node.attr_bool("complex_mode", false),
            offset: [node.attr_f32("x", 0.0), node.attr_f32("y", 0.0)],
            text: node.text.clone(),
        }
    }
}

/// Parsed `<texture>` / `<texture_e/t/d/h>` properties inside a UI control (`CUIXmlInit::InitTexture`).
#[derive(Debug, Clone, PartialEq)]
pub struct UiTextureRef {
    pub texture: Option<String>,
    pub shader: Option<String>,
    pub rect: Option<[f32; 4]>,
    pub color_argb: u32,
    pub offset: [f32; 2],
    pub texture_e: Option<String>,
    pub texture_t: Option<String>,
    pub texture_d: Option<String>,
    pub texture_h: Option<String>,
}

/// Structured X-Ray UI layout widget/window node (`CUIXmlInit::InitWindow` / `InitStatic` hierarchy).
#[derive(Debug, Clone, PartialEq)]
pub struct UiWindowDef {
    pub tag: String,
    pub id: Option<String>,
    pub window_name: Option<String>,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub align: String,
    pub alignment: String,
    pub stretch: bool,
    pub hint: Option<String>,
    pub texture: Option<UiTextureRef>,
    pub text: Option<UiTextDef>,
    pub attributes: BTreeMap<String, String>,
    pub children: Vec<UiWindowDef>,
}

impl UiWindowDef {
    pub fn from_xml_node(node: &XmlNode, colors: &UiColorDefs) -> Self {
        let tex_node = node.children_named("texture").next();
        let tex_e = node
            .children_named("texture_e")
            .next()
            .and_then(XmlNode::read_text)
            .map(str::to_owned);
        let tex_t = node
            .children_named("texture_t")
            .next()
            .and_then(XmlNode::read_text)
            .map(str::to_owned);
        let tex_d = node
            .children_named("texture_d")
            .next()
            .and_then(XmlNode::read_text)
            .map(str::to_owned);
        let tex_h = node
            .children_named("texture_h")
            .next()
            .and_then(XmlNode::read_text)
            .map(str::to_owned);
        let offset_node = node.children_named("texture_offset").next();

        let texture = if tex_node.is_some()
            || tex_e.is_some()
            || tex_t.is_some()
            || tex_d.is_some()
            || tex_h.is_some()
        {
            let (tex_str, shader, rect, color_argb) = match tex_node {
                Some(tn) => {
                    let rx = tn.attr_f32("x", 0.0);
                    let ry = tn.attr_f32("y", 0.0);
                    let rw = tn.attr_f32("width", 0.0);
                    let rh = tn.attr_f32("height", 0.0);
                    let rect = if rw != 0.0 && rh != 0.0 {
                        Some([rx, ry, rx + rw, ry + rh])
                    } else {
                        None
                    };
                    (
                        tn.read_text().map(str::to_owned),
                        tn.attr("shader").map(str::to_owned),
                        rect,
                        colors.resolve_node_color(tn, 255),
                    )
                }
                None => (None, None, None, color_argb(255, 255, 255, 255)),
            };
            let offset = match offset_node {
                Some(on) => [on.attr_f32("x", 0.0), on.attr_f32("y", 0.0)],
                None => [0.0, 0.0],
            };
            Some(UiTextureRef {
                texture: tex_str,
                shader,
                rect,
                color_argb,
                offset,
                texture_e: tex_e,
                texture_t: tex_t,
                texture_d: tex_d,
                texture_h: tex_h,
            })
        } else {
            None
        };

        let text = node
            .children_named("text")
            .next()
            .map(|tn| UiTextDef::from_xml_node(tn, colors));

        let window_name = node
            .children_named("window_name")
            .next()
            .and_then(XmlNode::read_text)
            .map(str::to_owned);

        let mut attributes = BTreeMap::new();
        for (k, v) in &node.attributes {
            attributes.insert(k.clone(), v.clone());
        }

        let mut children = Vec::new();
        for ch in &node.children {
            if matches!(
                ch.name.as_str(),
                "texture" | "texture_offset" | "texture_e" | "texture_t" | "texture_d" | "texture_h" | "text" | "window_name"
            ) {
                continue;
            }
            children.push(Self::from_xml_node(ch, colors));
        }

        Self {
            tag: node.name.clone(),
            id: node.attr("id").map(str::to_owned),
            window_name,
            x: node.attr_f32("x", 0.0),
            y: node.attr_f32("y", 0.0),
            width: node.attr_f32("width", 0.0),
            height: node.attr_f32("height", 0.0),
            align: node.attr_or("align", "").to_owned(),
            alignment: node.attr_or("alignment", "").to_owned(),
            stretch: node.attr_bool("stretch", false),
            hint: node.attr("hint").map(str::to_owned),
            texture,
            text,
            attributes,
            children,
        }
    }

    pub fn find_child(&self, tag: &str) -> Option<&UiWindowDef> {
        self.children.iter().find(|c| c.tag == tag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MemXmlProvider(BTreeMap<String, Vec<u8>>);

    impl XmlFileProvider for MemXmlProvider {
        type Error = &'static str;

        fn read_xml_bytes(&self, virtual_path: &str) -> Result<Vec<u8>, Self::Error> {
            let key = virtual_path.replace('/', "\\").to_ascii_lowercase();
            self.0.get(&key).cloned().ok_or("missing")
        }

        fn list_xml_matches(
            &self,
            dir_prefix: &str,
            filename_pattern: &str,
        ) -> Result<Vec<String>, Self::Error> {
            let mut prefix = dir_prefix.replace('/', "\\").to_ascii_lowercase();
            if !prefix.is_empty() && !prefix.ends_with('\\') {
                prefix.push('\\');
            }
            let suffix = filename_pattern
                .strip_prefix('*')
                .unwrap_or(filename_pattern)
                .to_ascii_lowercase();
            let mut out = Vec::new();
            for k in self.0.keys() {
                if let Some(rest) = k.strip_prefix(&prefix)
                    && !rest.contains('\\')
                    && rest.ends_with(&suffix)
                {
                    out.push(k.clone());
                }
            }
            Ok(out)
        }
    }

    #[test]
    fn test_xml_dom_navigation_entities_cdata_and_cp1251() {
        let mut raw_cp1251 = b"<?xml version=\"1.0\" encoding=\"windows-1251\"?>\n<!-- \xCF\xF0\xE8\xE2\xE5\xF2 -->\n<string_table>\n\t<string id=\"st_test\">\n\t\t<text>"
            .to_vec();
        // CP1251: "Привет & Проверка №1" + literal \n + action tag
        raw_cp1251.extend_from_slice(
            b"\xCF\xF0\xE8\xE2\xE5\xF2 &amp; <![CDATA[<Tag>]]> \xB91\\n$$ACTION_USE$$!",
        );
        raw_cp1251.extend_from_slice(b"</text>\n\t</string>\n</string_table>\n");

        let table = LocalizationTable::from_xml_bytes("rus", &raw_cp1251).unwrap();
        let mut binds = BTreeMap::new();
        binds.insert("USE".to_string(), "F".to_string());
        assert_eq!(
            table.translate_formatted("st_test", &binds),
            "Привет & <Tag> №1\nF!"
        );
        assert!(table.raw_key_binding_ids.contains("st_test"));
        assert_eq!(table.translate("st_missing_fallback"), "st_missing_fallback");
    }

    #[test]
    fn test_xml_includes_wildcards_atlas_colors_and_ui_layout() {
        let mut files = BTreeMap::new();
        files.insert(
            "configs\\ui\\main.xml".into(),
            br#"<w>
#include "ui\hdr_16.xml"
#include "ui\parts\*.xml"
</w>"#
                .to_vec(),
        );
        files.insert(
            "configs\\ui\\hdr_16.xml".into(),
            br#"<ctrl_hdr x="115" y="400" width="800" height="300" stretch="1">
    <frame x="0" y="0" width="802" height="300" stretch="1">
        <texture r="0" g="0" b="0" a="150">ui_icons_PDA_tooltips</texture>
    </frame>
    <cap_main x="0" y="10" width="800" height="30">
        <text color="ui_clr_1" font="letterica18" align="c" vert_align="c">ui_title</text>
    </cap_main>
</ctrl_hdr>"#
                .to_vec(),
        );
        files.insert(
            "configs\\ui\\parts\\b_btn.xml".into(),
            br#"<btn x="10" y="20" width="100" height="30"/>"#.to_vec(),
        );
        files.insert(
            "configs\\ui\\parts\\a_btn.xml".into(),
            br#"<btn x="5" y="10" width="50" height="20"/>"#.to_vec(),
        );

        let provider = MemXmlProvider(files);
        let doc = XmlDocument::load_with_provider(&provider, "configs", "configs\\ui\\main.xml")
            .unwrap();

        assert_eq!(doc.read_attrib_f32("ctrl_hdr", 0, "x", 0.0), 115.0);
        assert_eq!(doc.read_attrib_f32("btn", 0, "x", 0.0), 5.0);
        assert_eq!(doc.read_attrib_f32("btn", 1, "x", 0.0), 10.0);
        assert_eq!(doc.read_str("ctrl_hdr:cap_main:text", 0, ""), "ui_title");

        let colors_doc = XmlDocument::parse(
            r#"<colors><color name="ui_clr_1" r="250" g="200" b="150" a="180"/></colors>"#,
        )
        .unwrap();
        let colors = UiColorDefs::from_doc(&colors_doc);
        assert_eq!(colors.colors["ui_clr_1"], color_argb(180, 250, 200, 150));

        let hdr_wnd = UiWindowDef::from_xml_node(doc.navigate("ctrl_hdr", 0).unwrap(), &colors);
        assert!(hdr_wnd.stretch);
        let frame = hdr_wnd.find_child("frame").unwrap();
        assert_eq!(
            frame.texture.as_ref().unwrap().texture.as_deref(),
            Some("ui_icons_PDA_tooltips")
        );
        assert_eq!(
            frame.texture.as_ref().unwrap().color_argb,
            color_argb(150, 0, 0, 0)
        );
        let cap_main = hdr_wnd.find_child("cap_main").unwrap();
        assert_eq!(
            cap_main.text.as_ref().unwrap().color_argb,
            color_argb(180, 250, 200, 150)
        );

        let atlas = UiTextureAtlasMap::from_xml(
            r#"<w><file name="ui\ui_common"><texture id="ui_rect" x="10" y="20" width="30" height="40" border_l="1"/></file></w>"#,
        )
        .unwrap();
        let region = atlas.get("ui_rect").unwrap();
        assert_eq!(region.pixel_rect(), [10.0, 20.0, 40.0, 60.0]);
        assert_eq!(region.border_l, 1);
        assert_eq!(region.dds_vpath(), "textures\\ui\\ui_common.dds");
    }
}
