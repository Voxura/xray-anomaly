//! X-Ray Monolith / Call of Chernobyl / S.T.A.L.K.E.R. Anomaly / xrRazom
//! `.scoc` Savegame Codec (`lmarshal.c` Binary Lua Marshal Format)
//!
//! CoC / Anomaly / xrRazom stores script-side player & mod persistence (`m_data`)
//! alongside each `.scop` ALife savegame in a `.scoc` file serialized by Lua's
//! `marshal.encode(m_data)` (`src/3rd party/lua-extensions/lmarshal.c`):
//!
//! - Byte `0`: magic `0x8e` (`MAR_MAGIC`)
//! - Byte `1`: root Lua type `0x05` (`LUA_TTABLE`)
//! - Byte `2`: table mode flags `0x02` (`MAR_TVAL`)
//! - Bytes `3..7`: little-endian `u32` payload byte length
//! - Recursive Lua values inside table `(key, value)` pairs:
//!   - `0x00`: `LUA_TNIL`
//!   - `0x01`: `LUA_TBOOLEAN` (`u8`: `0` = false, non-zero = true)
//!   - `0x03`: `LUA_TNUMBER` (8-byte little-endian IEEE 754 `f64` — LuaJIT `lua_Number`, covering floats and exact 53-bit `i64` integers)
//!   - `0x04`: `LUA_TSTRING` (`u32` little-endian length + raw bytes)
//!   - `0x05`: `LUA_TTABLE` + `u8` subtag:
//!     - `0x01` (`MAR_TREF`): `u32` 1-based preorder table back-reference
//!     - `0x02` (`MAR_TVAL`): `u32` payload length + recursive `(key, value)` stream
//!     - `0x03` (`MAR_TUSR`): `u32` payload length + persist callback table stream
//!
//! On top of the lossless binary AST (`ScocFile`, `ScocTable`, `ScocValue`), this
//! module also provides typed representations for Anomaly / CoC / xrRazom co-op peer
//! state extraction and reconstruction:
//! - Player inventory items (`ScocInventoryItem`: section, id, durability/condition, remaining uses, ammo, upgrades)
//! - `item_remaining_uses` map
//! - `visited_levels` set/map
//! - xrRazom peer tables: `skills_levels`, `psy_table`, `head_tbl`, `visited_campfires`.

use std::collections::BTreeMap;
use std::fmt;

use crate::chunk::{ByteCursor, FormatError};
use crate::xml_ui::decode_cp1251;

pub const SCOC_EXTENSION: &str = ".scoc";
pub const SCOC_MAGIC: u8 = 0x8e;

pub const LUA_TNIL: u8 = 0x00;
pub const LUA_TBOOLEAN: u8 = 0x01;
pub const LUA_TLIGHTUSERDATA: u8 = 0x02;
pub const LUA_TNUMBER: u8 = 0x03;
pub const LUA_TSTRING: u8 = 0x04;
pub const LUA_TTABLE: u8 = 0x05;
pub const LUA_TFUNCTION: u8 = 0x06;
pub const LUA_TUSERDATA: u8 = 0x07;
pub const LUA_TTHREAD: u8 = 0x08;

pub const MAR_TREF: u8 = 0x01;
pub const MAR_TVAL: u8 = 0x02;
pub const MAR_TUSR: u8 = 0x03;

/// Conservative nesting limit guarding against corrupted recursive `.scoc` payloads.
pub const SCOC_MAX_RECURSION_DEPTH: usize = 128;

const MAX_EXACT_LUA_INT: f64 = 9_007_199_254_740_992.0; // 2^53

/// Validated 7-byte header at the start of an X-Ray `.scoc` file (`lmarshal.c` root table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScocHeader {
    pub magic: u8,
    pub root_type: u8,
    pub flags: u8,
    pub payload_len: u32,
}

/// Validate a `.scoc` buffer prefix (`magic == 0x8e`, `root_type == 0x05`, `flags == 0x02`, exact `u32 payload_len`).
pub fn validate_scoc_header(bytes: &[u8]) -> Option<ScocHeader> {
    if bytes.len() < 7 {
        return None;
    }
    let magic = bytes[0];
    let root_type = bytes[1];
    let flags = bytes[2];
    let payload_len = u32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
    if magic != SCOC_MAGIC || root_type != LUA_TTABLE || flags != MAR_TVAL {
        return None;
    }
    if payload_len as usize != bytes.len() - 7 {
        return None;
    }
    Some(ScocHeader {
        magic,
        root_type,
        flags,
        payload_len,
    })
}

/// Raw lossless Lua table preserving insertion order and arbitrary key types (`String`, integer `Number`, `Bool`, etc.).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScocTable {
    pub entries: Vec<(ScocValue, ScocValue)>,
}

impl ScocTable {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get_str(&self, key: &str) -> Option<&ScocValue> {
        let target = key.as_bytes();
        self.entries.iter().find_map(|(k, v)| match k {
            ScocValue::String(bytes) if bytes.as_slice() == target => Some(v),
            _ => None,
        })
    }

    pub fn get_str_mut(&mut self, key: &str) -> Option<&mut ScocValue> {
        let target = key.as_bytes();
        self.entries.iter_mut().find_map(|(k, v)| match k {
            ScocValue::String(bytes) if bytes.as_slice() == target => Some(v),
            _ => None,
        })
    }

    pub fn get_i64(&self, key: i64) -> Option<&ScocValue> {
        self.entries.iter().find_map(|(k, v)| match k.as_i64() {
            Some(n) if n == key => Some(v),
            _ => None,
        })
    }

    pub fn get_table(&self, key: &str) -> Option<&ScocTable> {
        self.get_str(key).and_then(ScocValue::as_table)
    }

    pub fn get_table_mut(&mut self, key: &str) -> Option<&mut ScocTable> {
        self.get_str_mut(key).and_then(ScocValue::as_table_mut)
    }

    pub fn set_str(&mut self, key: impl Into<String>, value: ScocValue) {
        let key_str: String = key.into();
        if let Some(slot) = self.get_str_mut(&key_str) {
            *slot = value;
        } else {
            self.entries.push((ScocValue::string(key_str), value));
        }
    }

    pub fn set_i64(&mut self, key: i64, value: ScocValue) {
        for (k, v) in &mut self.entries {
            if k.as_i64() == Some(key) {
                *v = value;
                return;
            }
        }
        self.entries.push((ScocValue::int(key), value));
    }

    pub fn remove_str(&mut self, key: &str) -> Option<ScocValue> {
        let target = key.as_bytes();
        let pos = self.entries.iter().position(|(k, _)| match k {
            ScocValue::String(bytes) => bytes.as_slice() == target,
            _ => false,
        })?;
        Some(self.entries.remove(pos).1)
    }

    /// Collect dense/sparse positive integer-keyed entries (`1..`) sorted by Lua array index.
    pub fn sorted_array_entries(&self) -> Vec<(i64, &ScocValue)> {
        let mut items: Vec<(i64, &ScocValue)> = self
            .entries
            .iter()
            .filter_map(|(k, v)| k.as_i64().filter(|&idx| idx >= 1).map(|idx| (idx, v)))
            .collect();
        items.sort_by_key(|(idx, _)| *idx);
        items
    }

    /// Encode this table's entries without the outer `[0x05, 0x02, u32 len]` wrapper.
    pub fn encode_payload(&self, out: &mut Vec<u8>) -> Result<(), FormatError> {
        encode_table_payload_depth(self, out, 0)
    }

    /// Decode table `(key, value)` pairs directly from an inner table payload slice.
    pub fn decode_payload(payload: &[u8]) -> Result<Self, FormatError> {
        let mut seen = Vec::new();
        let mut next_idx = 1usize;
        decode_table_entries(payload, &mut seen, &mut next_idx, 0)
    }
}

/// Recursive Lua value stored inside a `.scoc` marshal stream (`lmarshal.c`).
#[derive(Debug, Clone, PartialEq)]
pub enum ScocValue {
    Nil,
    Bool(bool),
    /// Wire tag `0x03` (`LUA_TNUMBER`): 8-byte IEEE 754 `f64` covering both floats and 53-bit `i64` integers.
    Number(f64),
    String(Vec<u8>),
    Table(ScocTable),
    /// Table serialized with `MAR_TUSR` (`0x03`) `__persist` hook.
    UserTable(ScocTable),
    /// Opaque Lua closure (`LUA_TFUNCTION`) if encountered in a modded `.scoc`.
    Function {
        bytecode: Vec<u8>,
        upvalues: ScocTable,
    },
    /// Unresolved self-referential cycle back-reference (`MAR_TREF` pointing to an ancestor still on stack).
    CycleRef(u32),
}

impl ScocValue {
    #[inline]
    pub const fn nil() -> Self {
        Self::Nil
    }

    #[inline]
    pub const fn bool(v: bool) -> Self {
        Self::Bool(v)
    }

    #[inline]
    pub const fn float(v: f64) -> Self {
        Self::Number(v)
    }

    #[inline]
    pub fn int(v: i64) -> Self {
        Self::Number(v as f64)
    }

    #[inline]
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into().into_bytes())
    }

    #[inline]
    pub const fn bytes(b: Vec<u8>) -> Self {
        Self::String(b)
    }

    #[inline]
    pub const fn table(t: ScocTable) -> Self {
        Self::Table(t)
    }

    #[inline]
    pub fn is_nil(&self) -> bool {
        matches!(self, Self::Nil)
    }

    #[inline]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Return exact integer value when stored as a finite integral `lua_Number` within `[-2^53, 2^53]`.
    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(n)
                if n.is_finite()
                    && n.fract() == 0.0
                    && *n >= -MAX_EXACT_LUA_INT
                    && *n <= MAX_EXACT_LUA_INT =>
            {
                Some(*n as i64)
            }
            _ => None,
        }
    }

    #[inline]
    pub fn as_i64_round(&self) -> Option<i64> {
        match self {
            Self::Number(n) if n.is_finite() && *n >= -MAX_EXACT_LUA_INT && *n <= MAX_EXACT_LUA_INT => {
                Some(n.round() as i64)
            }
            _ => None,
        }
    }

    #[inline]
    pub fn as_u32(&self) -> Option<u32> {
        self.as_i64().and_then(|v| u32::try_from(v).ok())
    }

    #[inline]
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::String(b) => Some(b.as_slice()),
            _ => None,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(b) => std::str::from_utf8(b).ok(),
            _ => None,
        }
    }

    /// Decode string as UTF-8 when valid, otherwise Windows-1251 (standard X-Ray Cyrillic text encoding).
    #[inline]
    pub fn as_xray_string(&self) -> Option<String> {
        match self {
            Self::String(b) => match std::str::from_utf8(b) {
                Ok(s) => Some(s.to_string()),
                Err(_) => Some(decode_cp1251(b)),
            },
            _ => None,
        }
    }

    #[inline]
    pub fn as_table(&self) -> Option<&ScocTable> {
        match self {
            Self::Table(t) | Self::UserTable(t) => Some(t),
            _ => None,
        }
    }

    #[inline]
    pub fn as_table_mut(&mut self) -> Option<&mut ScocTable> {
        match self {
            Self::Table(t) | Self::UserTable(t) => Some(t),
            _ => None,
        }
    }

    #[inline]
    pub fn tag(&self) -> u8 {
        match self {
            Self::Nil => LUA_TNIL,
            Self::Bool(_) => LUA_TBOOLEAN,
            Self::Number(_) => LUA_TNUMBER,
            Self::String(_) => LUA_TSTRING,
            Self::Table(_) | Self::UserTable(_) | Self::CycleRef(_) => LUA_TTABLE,
            Self::Function { .. } => LUA_TFUNCTION,
        }
    }
}

/// Complete decoded `.scoc` file wrapping the root Lua `m_data` table.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScocFile {
    pub root: ScocTable,
}

impl ScocFile {
    pub fn new(root: ScocTable) -> Self {
        Self { root }
    }

    /// Parse a `.scoc` binary file, strictly verifying `0x8e` magic, root `0x05` table, `0x02` value tag,
    /// and exact `u32` payload length.
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let Some(hdr) = validate_scoc_header(bytes) else {
            if bytes.len() < 7 {
                return Err(FormatError::UnexpectedEof);
            }
            if bytes[0] != SCOC_MAGIC {
                return Err(FormatError::InvalidValue("invalid .scoc magic byte (expected 0x8e)"));
            }
            if bytes[1] != LUA_TTABLE {
                return Err(FormatError::InvalidValue("invalid .scoc root Lua type (expected 0x05 table)"));
            }
            if bytes[2] != MAR_TVAL {
                return Err(FormatError::InvalidValue("invalid .scoc root table flags (expected 0x02 MAR_TVAL)"));
            }
            return Err(FormatError::InvalidValue(
                ".scoc payload length does not match file size",
            ));
        };

        let payload = &bytes[7..7 + hdr.payload_len as usize];
        // Slot 1 in `lmarshal.c` is reserved for the root table before its children decode.
        let mut seen: Vec<Option<ScocValue>> = vec![None];
        let mut next_idx = 2usize;
        let root = decode_table_entries(payload, &mut seen, &mut next_idx, 1)?;
        seen[0] = Some(ScocValue::Table(root.clone()));
        Ok(Self { root })
    }

    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        Self::parse(bytes)
    }

    /// Encode into an exact `lmarshal.c`-compatible `.scoc` binary buffer:
    /// `[0x8e][0x05][0x02][u32 payload_len][entries...]`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        let mut payload = Vec::new();
        encode_table_payload_depth(&self.root, &mut payload, 1)?;
        let payload_len = u32::try_from(payload.len())
            .map_err(|_| FormatError::InvalidValue(".scoc payload exceeds u32 max length"))?;
        let mut out = Vec::with_capacity(7 + payload.len());
        out.push(SCOC_MAGIC);
        out.push(LUA_TTABLE);
        out.push(MAR_TVAL);
        out.extend_from_slice(&payload_len.to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Extract typed CoC / Anomaly / xrRazom peer state tables (`inventory`, `item_remaining_uses`,
    /// `visited_levels`, `skills_levels`, `psy_table`, `head_tbl`, `visited_campfires`) from `self.root`.
    pub fn extract_peer_state(&self) -> ScocPeerState {
        ScocPeerState::from_root_table(&self.root)
    }

    /// Update the typed CoC / Anomaly / xrRazom peer state tables in `self.root` while preserving any
    /// unrelated mod keys or raw subtables.
    pub fn apply_peer_state(&mut self, state: &ScocPeerState) {
        state.write_into_root_table(&mut self.root);
    }

    /// Construct a `ScocFile` from a typed `ScocPeerState`.
    pub fn from_peer_state(state: &ScocPeerState) -> Self {
        let mut root = ScocTable::new();
        state.write_into_root_table(&mut root);
        Self { root }
    }
}

/// Generic `marshal.encode(value)` stream encoder (`[0x8e][value...]`).
pub fn encode_marshal_value(value: &ScocValue) -> Result<Vec<u8>, FormatError> {
    let mut out = Vec::new();
    out.push(SCOC_MAGIC);
    encode_value_depth(value, &mut out, 0)?;
    Ok(out)
}

/// Generic `marshal.decode(bytes)` stream decoder (`[0x8e][value...]`).
pub fn decode_marshal_value(bytes: &[u8]) -> Result<ScocValue, FormatError> {
    if bytes.is_empty() {
        return Err(FormatError::UnexpectedEof);
    }
    if bytes[0] != SCOC_MAGIC {
        return Err(FormatError::InvalidValue("invalid lmarshal magic byte (expected 0x8e)"));
    }
    let mut cur = ByteCursor::new(&bytes[1..]);
    let mut seen = Vec::new();
    let mut next_idx = 1usize;
    let val = decode_value_cursor(&mut cur, &mut seen, &mut next_idx, 0)?;
    if cur.remaining() != 0 {
        return Err(FormatError::InvalidValue(
            "unexpected trailing bytes after lmarshal root value",
        ));
    }
    Ok(val)
}

fn encode_value_depth(
    value: &ScocValue,
    out: &mut Vec<u8>,
    depth: usize,
) -> Result<(), FormatError> {
    if depth > SCOC_MAX_RECURSION_DEPTH {
        return Err(FormatError::InvalidValue(
            ".scoc table recursion depth exceeded limit",
        ));
    }
    match value {
        ScocValue::Nil => {
            out.push(LUA_TNIL);
        }
        ScocValue::Bool(b) => {
            out.push(LUA_TBOOLEAN);
            out.push(u8::from(*b));
        }
        ScocValue::Number(n) => {
            out.push(LUA_TNUMBER);
            out.extend_from_slice(&n.to_le_bytes());
        }
        ScocValue::String(bytes) => {
            let len = u32::try_from(bytes.len())
                .map_err(|_| FormatError::InvalidValue(".scoc string exceeds u32 max length"))?;
            out.push(LUA_TSTRING);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(bytes);
        }
        ScocValue::Table(tbl) => {
            let mut inner = Vec::new();
            encode_table_payload_depth(tbl, &mut inner, depth + 1)?;
            let len = u32::try_from(inner.len())
                .map_err(|_| FormatError::InvalidValue(".scoc table payload exceeds u32 max"))?;
            out.push(LUA_TTABLE);
            out.push(MAR_TVAL);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&inner);
        }
        ScocValue::UserTable(tbl) => {
            let mut inner = Vec::new();
            encode_table_payload_depth(tbl, &mut inner, depth + 1)?;
            let len = u32::try_from(inner.len())
                .map_err(|_| FormatError::InvalidValue(".scoc user table payload exceeds u32 max"))?;
            out.push(LUA_TTABLE);
            out.push(MAR_TUSR);
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&inner);
        }
        ScocValue::Function { bytecode, upvalues } => {
            let bc_len = u32::try_from(bytecode.len())
                .map_err(|_| FormatError::InvalidValue(".scoc function bytecode exceeds u32 max"))?;
            let mut upval_buf = Vec::new();
            encode_table_payload_depth(upvalues, &mut upval_buf, depth + 1)?;
            let upval_len = u32::try_from(upval_buf.len())
                .map_err(|_| FormatError::InvalidValue(".scoc upvalue table exceeds u32 max"))?;
            out.push(LUA_TFUNCTION);
            out.push(MAR_TVAL);
            out.extend_from_slice(&bc_len.to_le_bytes());
            out.extend_from_slice(bytecode);
            out.extend_from_slice(&upval_len.to_le_bytes());
            out.extend_from_slice(&upval_buf);
        }
        ScocValue::CycleRef(ref_id) => {
            out.push(LUA_TTABLE);
            out.push(MAR_TREF);
            out.extend_from_slice(&ref_id.to_le_bytes());
        }
    }
    Ok(())
}

fn encode_table_payload_depth(
    table: &ScocTable,
    out: &mut Vec<u8>,
    depth: usize,
) -> Result<(), FormatError> {
    if depth > SCOC_MAX_RECURSION_DEPTH {
        return Err(FormatError::InvalidValue(
            ".scoc table recursion depth exceeded limit",
        ));
    }
    for (k, v) in &table.entries {
        if matches!(k, ScocValue::Nil) || matches!(v, ScocValue::Nil) {
            continue;
        }
        encode_value_depth(k, out, depth)?;
        encode_value_depth(v, out, depth)?;
    }
    Ok(())
}

fn decode_table_entries(
    payload: &[u8],
    seen: &mut Vec<Option<ScocValue>>,
    next_idx: &mut usize,
    depth: usize,
) -> Result<ScocTable, FormatError> {
    if depth > SCOC_MAX_RECURSION_DEPTH {
        return Err(FormatError::InvalidValue(
            ".scoc table recursion depth exceeded limit",
        ));
    }
    let mut cur = ByteCursor::new(payload);
    let mut entries = Vec::new();
    while cur.remaining() > 0 {
        let key = decode_value_cursor(&mut cur, seen, next_idx, depth)?;
        let val = decode_value_cursor(&mut cur, seen, next_idx, depth)?;
        entries.push((key, val));
    }
    Ok(ScocTable { entries })
}

fn decode_value_cursor(
    cur: &mut ByteCursor<'_>,
    seen: &mut Vec<Option<ScocValue>>,
    next_idx: &mut usize,
    depth: usize,
) -> Result<ScocValue, FormatError> {
    if depth > SCOC_MAX_RECURSION_DEPTH {
        return Err(FormatError::InvalidValue(
            ".scoc value recursion depth exceeded limit",
        ));
    }
    let val_type = cur.read_u8()?;
    match val_type {
        LUA_TNIL | LUA_TTHREAD => Ok(ScocValue::Nil),
        LUA_TBOOLEAN => {
            let b = cur.read_u8()?;
            Ok(ScocValue::Bool(b != 0))
        }
        LUA_TNUMBER => {
            let raw: [u8; 8] = cur
                .read_bytes(8)?
                .try_into()
                .map_err(|_| FormatError::UnexpectedEof)?;
            let n = f64::from_le_bytes(raw);
            Ok(ScocValue::Number(n))
        }
        LUA_TSTRING => {
            let len = cur.read_u32()? as usize;
            let bytes = cur.read_bytes(len)?.to_vec();
            Ok(ScocValue::String(bytes))
        }
        LUA_TTABLE => {
            let tag = cur.read_u8()?;
            match tag {
                MAR_TREF => {
                    let ref_raw = cur.read_i32()?;
                    if ref_raw < 1 {
                        return Err(FormatError::InvalidValue(
                            "invalid MAR_TREF index (< 1) in .scoc",
                        ));
                    }
                    let slot = (ref_raw - 1) as usize;
                    if slot >= seen.len() {
                        return Err(FormatError::InvalidValue(
                            "out-of-bounds MAR_TREF reference in .scoc",
                        ));
                    }
                    if let Some(resolved) = &seen[slot] {
                        Ok(resolved.clone())
                    } else {
                        Ok(ScocValue::CycleRef(ref_raw as u32))
                    }
                }
                MAR_TVAL => {
                    let len = cur.read_u32()? as usize;
                    let sub_payload = cur.read_bytes(len)?;
                    // Allocate 1-based seen slot BEFORE recursing (`lmarshal.c:345`).
                    let slot = *next_idx - 1;
                    *next_idx += 1;
                    if seen.len() <= slot {
                        seen.resize(slot + 1, None);
                    }
                    let table = decode_table_entries(sub_payload, seen, next_idx, depth + 1)?;
                    let val = ScocValue::Table(table);
                    seen[slot] = Some(val.clone());
                    Ok(val)
                }
                MAR_TUSR => {
                    let len = cur.read_u32()? as usize;
                    let sub_payload = cur.read_bytes(len)?;
                    let table = decode_table_entries(sub_payload, seen, next_idx, depth + 1)?;
                    // `MAR_TUSR` records seen index AFTER decoding its callback table (`lmarshal.c:357`).
                    let slot = *next_idx - 1;
                    *next_idx += 1;
                    if seen.len() <= slot {
                        seen.resize(slot + 1, None);
                    }
                    let val = ScocValue::UserTable(table);
                    seen[slot] = Some(val.clone());
                    Ok(val)
                }
                _ => Err(FormatError::InvalidValue(
                    "invalid LUA_TTABLE marshal subtag in .scoc",
                )),
            }
        }
        LUA_TFUNCTION => {
            let tag = cur.read_u8()?;
            if tag == MAR_TREF {
                let ref_raw = cur.read_i32()?;
                if ref_raw < 1 {
                    return Err(FormatError::InvalidValue(
                        "invalid LUA_TFUNCTION MAR_TREF index in .scoc",
                    ));
                }
                let slot = (ref_raw - 1) as usize;
                let Some(Some(resolved)) = seen.get(slot) else {
                    return Err(FormatError::InvalidValue(
                        "unresolved LUA_TFUNCTION MAR_TREF reference in .scoc",
                    ));
                };
                Ok(resolved.clone())
            } else {
                let bc_len = cur.read_u32()? as usize;
                let bytecode = cur.read_bytes(bc_len)?.to_vec();
                let slot = *next_idx - 1;
                *next_idx += 1;
                if seen.len() <= slot {
                    seen.resize(slot + 1, None);
                }
                let upval_len = cur.read_u32()? as usize;
                let upval_payload = cur.read_bytes(upval_len)?;
                let upvalues = decode_table_entries(upval_payload, seen, next_idx, depth + 1)?;
                let val = ScocValue::Function { bytecode, upvalues };
                seen[slot] = Some(val.clone());
                Ok(val)
            }
        }
        LUA_TUSERDATA => {
            let tag = cur.read_u8()?;
            match tag {
                MAR_TREF => {
                    let ref_raw = cur.read_i32()?;
                    if ref_raw < 1 {
                        return Err(FormatError::InvalidValue(
                            "invalid LUA_TUSERDATA MAR_TREF index in .scoc",
                        ));
                    }
                    let slot = (ref_raw - 1) as usize;
                    Ok(seen
                        .get(slot)
                        .and_then(|v| v.as_ref())
                        .cloned()
                        .unwrap_or(ScocValue::CycleRef(ref_raw as u32)))
                }
                MAR_TUSR => {
                    let len = cur.read_u32()? as usize;
                    let sub_payload = cur.read_bytes(len)?;
                    let table = decode_table_entries(sub_payload, seen, next_idx, depth + 1)?;
                    let slot = *next_idx - 1;
                    *next_idx += 1;
                    if seen.len() <= slot {
                        seen.resize(slot + 1, None);
                    }
                    let val = ScocValue::UserTable(table);
                    seen[slot] = Some(val.clone());
                    Ok(val)
                }
                _ => Ok(ScocValue::Nil),
            }
        }
        _ => Err(FormatError::InvalidValue("unknown Lua type tag in .scoc stream")),
    }
}

// ============================================================================
// Typed CoC / Anomaly / xrRazom Peer State Representations
// ============================================================================

/// Single player inventory item record stored in `.scoc` (section, durability/condition,
/// weapon magazine/attachments/upgrades, remaining multi-use count, and extra mod fields).
#[derive(Debug, Clone, PartialEq)]
pub struct ScocInventoryItem {
    pub id: Option<i64>,
    pub section: String,
    /// Item condition / durability normalized to `[0.0, 1.0]` (`condition` / `cond` / `durability`).
    pub condition: f64,
    pub remaining_uses: Option<i64>,
    pub slot: Option<i64>,
    pub ammo_elapsed: Option<i64>,
    pub ammo_type: Option<i64>,
    pub ammo_section: Option<String>,
    pub scope_attached: Option<bool>,
    pub silencer_attached: Option<bool>,
    pub grenade_launcher_attached: Option<bool>,
    pub addon_flags: Option<i64>,
    pub quantity: Option<i64>,
    pub upgrades: Vec<String>,
    pub custom_data: Option<String>,
    /// Preserves any unmodeled mod fields (e.g. battery charge, container weight, artifact charge) losslessly.
    pub extra: ScocTable,
}

impl Default for ScocInventoryItem {
    fn default() -> Self {
        Self {
            id: None,
            section: String::new(),
            condition: 1.0,
            remaining_uses: None,
            slot: None,
            ammo_elapsed: None,
            ammo_type: None,
            ammo_section: None,
            scope_attached: None,
            silencer_attached: None,
            grenade_launcher_attached: None,
            addon_flags: None,
            quantity: None,
            upgrades: Vec::new(),
            custom_data: None,
            extra: ScocTable::new(),
        }
    }
}

impl ScocInventoryItem {
    pub fn new(section: impl Into<String>, condition: f64) -> Self {
        Self {
            section: section.into(),
            condition: condition.clamp(0.0, 1.0),
            ..Self::default()
        }
    }

    pub fn from_scoc_table(tbl: &ScocTable, fallback_id: Option<i64>) -> Self {
        let mut item = Self {
            id: fallback_id,
            ..Self::default()
        };

        let mut fallback_name: Option<(ScocValue, ScocValue)> = None;

        for (k, v) in &tbl.entries {
            let Some(k_str) = k.as_str() else {
                item.extra.entries.push((k.clone(), v.clone()));
                continue;
            };
            match k_str {
                "id" | "object_id" => item.id = v.as_i64().or(item.id),
                "section" | "sec" | "section_name" => {
                    if let Some(s) = v.as_xray_string() {
                        item.section = s;
                    }
                }
                "name" => {
                    if fallback_name.is_none() {
                        fallback_name = Some((k.clone(), v.clone()));
                    } else {
                        item.extra.entries.push((k.clone(), v.clone()));
                    }
                }
                "condition" | "cond" | "durability" => {
                    if let Some(c) = v.as_f64() {
                        item.condition = c;
                    }
                }
                "remaining_uses" | "uses" => item.remaining_uses = v.as_i64(),
                "slot" | "active_slot" => item.slot = v.as_i64(),
                "ammo_elapsed" | "ammo" => item.ammo_elapsed = v.as_i64(),
                "ammo_type" => item.ammo_type = v.as_i64(),
                "ammo_section" => item.ammo_section = v.as_xray_string(),
                "scope" | "scope_attached" => {
                    item.scope_attached = v.as_bool().or_else(|| v.as_i64().map(|n| n != 0));
                }
                "silencer" | "silencer_attached" => {
                    item.silencer_attached = v.as_bool().or_else(|| v.as_i64().map(|n| n != 0));
                }
                "gl" | "grenade_launcher" | "grenade_launcher_attached" => {
                    item.grenade_launcher_attached =
                        v.as_bool().or_else(|| v.as_i64().map(|n| n != 0));
                }
                "addon_flags" | "addons" => item.addon_flags = v.as_i64(),
                "quantity" | "count" | "box_size" => item.quantity = v.as_i64(),
                "custom_data" => item.custom_data = v.as_xray_string(),
                "upgrades" => {
                    if let Some(up_tbl) = v.as_table() {
                        for (_, uv) in up_tbl.sorted_array_entries() {
                            if let Some(s) = uv.as_xray_string() {
                                item.upgrades.push(s);
                            }
                        }
                        if item.upgrades.is_empty() {
                            for (uk, uv) in &up_tbl.entries {
                                if uk.as_i64().is_none()
                                    && uv.as_bool() == Some(true)
                                    && let Some(s) = uk.as_xray_string()
                                {
                                    item.upgrades.push(s);
                                }
                            }
                        }
                    }
                }
                _ => item.extra.entries.push((k.clone(), v.clone())),
            }
        }
        if let Some((k, v)) = fallback_name {
            if item.section.is_empty() {
                if let Some(s) = v.as_xray_string() {
                    item.section = s;
                } else {
                    item.extra.entries.push((k, v));
                }
            } else {
                item.extra.entries.push((k, v));
            }
        }
        item
    }

    pub fn to_scoc_table(&self) -> ScocTable {
        let mut tbl = ScocTable::new();
        if let Some(id) = self.id {
            tbl.set_str("id", ScocValue::int(id));
        }
        tbl.set_str("section", ScocValue::string(self.section.clone()));
        tbl.set_str("condition", ScocValue::float(self.condition));
        if let Some(uses) = self.remaining_uses {
            tbl.set_str("remaining_uses", ScocValue::int(uses));
        }
        if let Some(slot) = self.slot {
            tbl.set_str("slot", ScocValue::int(slot));
        }
        if let Some(ammo_elapsed) = self.ammo_elapsed {
            tbl.set_str("ammo_elapsed", ScocValue::int(ammo_elapsed));
        }
        if let Some(ammo_type) = self.ammo_type {
            tbl.set_str("ammo_type", ScocValue::int(ammo_type));
        }
        if let Some(ammo_section) = &self.ammo_section {
            tbl.set_str("ammo_section", ScocValue::string(ammo_section.clone()));
        }
        if let Some(v) = self.scope_attached {
            tbl.set_str("scope", ScocValue::bool(v));
        }
        if let Some(v) = self.silencer_attached {
            tbl.set_str("silencer", ScocValue::bool(v));
        }
        if let Some(v) = self.grenade_launcher_attached {
            tbl.set_str("grenade_launcher", ScocValue::bool(v));
        }
        if let Some(flags) = self.addon_flags {
            tbl.set_str("addon_flags", ScocValue::int(flags));
        }
        if let Some(qty) = self.quantity {
            tbl.set_str("quantity", ScocValue::int(qty));
        }
        if !self.upgrades.is_empty() {
            let mut up = ScocTable::with_capacity(self.upgrades.len());
            for (idx, u) in self.upgrades.iter().enumerate() {
                up.set_i64((idx + 1) as i64, ScocValue::string(u.clone()));
            }
            tbl.set_str("upgrades", ScocValue::Table(up));
        }
        if let Some(cd) = &self.custom_data {
            tbl.set_str("custom_data", ScocValue::string(cd.clone()));
        }
        for (k, v) in &self.extra.entries {
            tbl.entries.push((k.clone(), v.clone()));
        }
        tbl
    }
}

/// Skill progression entry in xrRazom / Anomaly `m_data.skills_levels`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScocSkillLevel {
    pub current_level: i64,
    pub experience: f64,
    pub unlocked: Option<bool>,
    /// Preserves compact `{ ["skill_name"] = 4 }` scalar format when loaded as a bare number.
    pub scalar_only: bool,
    pub extra: ScocTable,
}

impl Default for ScocSkillLevel {
    fn default() -> Self {
        Self {
            current_level: 0,
            experience: 0.0,
            unlocked: None,
            scalar_only: false,
            extra: ScocTable::new(),
        }
    }
}

impl ScocSkillLevel {
    pub fn new(current_level: i64, experience: f64) -> Self {
        Self {
            current_level,
            experience,
            ..Self::default()
        }
    }

    pub fn from_scoc_value(val: &ScocValue) -> Option<Self> {
        if let Some(lvl) = val.as_i64() {
            return Some(Self {
                current_level: lvl,
                scalar_only: true,
                ..Self::default()
            });
        }
        let tbl = val.as_table()?;
        let mut out = Self::default();
        for (k, v) in &tbl.entries {
            match k.as_str() {
                Some("current_level") | Some("level") => {
                    if let Some(n) = v.as_i64() {
                        out.current_level = n;
                    }
                }
                Some("experience") | Some("exp") => {
                    if let Some(x) = v.as_f64() {
                        out.experience = x;
                    }
                }
                Some("unlocked") => out.unlocked = v.as_bool(),
                _ => out.extra.entries.push((k.clone(), v.clone())),
            }
        }
        Some(out)
    }

    pub fn to_scoc_value(&self) -> ScocValue {
        if self.scalar_only && self.extra.is_empty() && self.unlocked.is_none() && self.experience == 0.0 {
            return ScocValue::int(self.current_level);
        }
        let mut tbl = ScocTable::new();
        tbl.set_str("current_level", ScocValue::int(self.current_level));
        tbl.set_str("experience", ScocValue::float(self.experience));
        if let Some(u) = self.unlocked {
            tbl.set_str("unlocked", ScocValue::bool(u));
        }
        for (k, v) in &self.extra.entries {
            tbl.entries.push((k.clone(), v.clone()));
        }
        ScocValue::Table(tbl)
    }
}

/// xrRazom / Anomaly `m_data.psy_table` wrapper with typed accessors and lossless extra field preservation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScocPsyTable {
    pub psy_health: Option<f64>,
    pub active_phantoms: Option<i64>,
    pub in_psy_zone: Option<bool>,
    pub extra: ScocTable,
}

impl ScocPsyTable {
    pub fn from_scoc_table(tbl: &ScocTable) -> Self {
        let mut psy_health = None;
        let mut active_phantoms = None;
        let mut in_psy_zone = None;
        let mut extra = ScocTable::new();
        for (k, v) in &tbl.entries {
            match k.as_str() {
                Some("psy_health") | Some("actor_psy_health") | Some("health") => {
                    if let Some(h) = v.as_f64() {
                        psy_health = Some(h);
                    } else {
                        extra.entries.push((k.clone(), v.clone()));
                    }
                }
                Some("active_phantoms") | Some("phantom_count") => {
                    if let Some(p) = v.as_i64() {
                        active_phantoms = Some(p);
                    } else {
                        extra.entries.push((k.clone(), v.clone()));
                    }
                }
                Some("in_psy_zone") => {
                    if let Some(z) = v.as_bool() {
                        in_psy_zone = Some(z);
                    } else {
                        extra.entries.push((k.clone(), v.clone()));
                    }
                }
                _ => extra.entries.push((k.clone(), v.clone())),
            }
        }
        Self {
            psy_health,
            active_phantoms,
            in_psy_zone,
            extra,
        }
    }

    pub fn to_scoc_table(&self) -> ScocTable {
        let mut tbl = ScocTable::new();
        if let Some(h) = self.psy_health {
            tbl.set_str("psy_health", ScocValue::float(h));
        }
        if let Some(p) = self.active_phantoms {
            tbl.set_str("active_phantoms", ScocValue::int(p));
        }
        if let Some(z) = self.in_psy_zone {
            tbl.set_str("in_psy_zone", ScocValue::bool(z));
        }
        for (k, v) in &self.extra.entries {
            tbl.entries.push((k.clone(), v.clone()));
        }
        tbl
    }
}

/// xrRazom `m_data.head_tbl` peer character head / visual appearance state.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScocHeadTable {
    pub head: Option<String>,
    pub visual: Option<String>,
    pub outfit: Option<String>,
    pub faction: Option<String>,
    pub extra: ScocTable,
}

impl ScocHeadTable {
    pub fn from_scoc_table(tbl: &ScocTable) -> Self {
        let mut out = Self::default();
        for (k, v) in &tbl.entries {
            match k.as_str() {
                Some("head") => {
                    if let Some(s) = v.as_xray_string() {
                        out.head = Some(s);
                    } else {
                        out.extra.entries.push((k.clone(), v.clone()));
                    }
                }
                Some("visual") => {
                    if let Some(s) = v.as_xray_string() {
                        out.visual = Some(s);
                    } else {
                        out.extra.entries.push((k.clone(), v.clone()));
                    }
                }
                Some("outfit") => {
                    if let Some(s) = v.as_xray_string() {
                        out.outfit = Some(s);
                    } else {
                        out.extra.entries.push((k.clone(), v.clone()));
                    }
                }
                Some("faction") => {
                    if let Some(s) = v.as_xray_string() {
                        out.faction = Some(s);
                    } else {
                        out.extra.entries.push((k.clone(), v.clone()));
                    }
                }
                _ => out.extra.entries.push((k.clone(), v.clone())),
            }
        }
        out
    }

    pub fn to_scoc_table(&self) -> ScocTable {
        let mut tbl = ScocTable::new();
        if let Some(v) = &self.head {
            tbl.set_str("head", ScocValue::string(v.clone()));
        }
        if let Some(v) = &self.visual {
            tbl.set_str("visual", ScocValue::string(v.clone()));
        }
        if let Some(v) = &self.outfit {
            tbl.set_str("outfit", ScocValue::string(v.clone()));
        }
        if let Some(v) = &self.faction {
            tbl.set_str("faction", ScocValue::string(v.clone()));
        }
        for (k, v) in &self.extra.entries {
            tbl.entries.push((k.clone(), v.clone()));
        }
        tbl
    }
}

/// Identifier used inside `visited_campfires` (numeric campfire object ID or named smart/campfire key).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScocCampfireKey {
    Id(i64),
    Name(String),
}

impl fmt::Display for ScocCampfireKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Id(id) => write!(f, "{id}"),
            Self::Name(name) => f.write_str(name),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScocCampfireEntry {
    pub key: ScocCampfireKey,
    pub state: ScocValue,
}

impl ScocCampfireEntry {
    pub fn by_id(id: i64) -> Self {
        Self {
            key: ScocCampfireKey::Id(id),
            state: ScocValue::Bool(true),
        }
    }

    pub fn by_name(name: impl Into<String>) -> Self {
        Self {
            key: ScocCampfireKey::Name(name.into()),
            state: ScocValue::Bool(true),
        }
    }
}

/// Complete typed representation of CoC / Anomaly / xrRazom peer save state stored inside `.scoc`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScocPeerState {
    pub inventory: Vec<ScocInventoryItem>,
    pub item_remaining_uses: BTreeMap<i64, i64>,
    /// Extra non-integer `item_remaining_uses` entries if a mod stores string keys or nested subtables.
    pub item_remaining_uses_extra: Vec<(ScocValue, ScocValue)>,
    pub visited_levels: BTreeMap<String, ScocValue>,
    pub skills_levels: BTreeMap<String, ScocSkillLevel>,
    pub psy_table: ScocPsyTable,
    pub head_tbl: ScocHeadTable,
    pub visited_campfires: Vec<ScocCampfireEntry>,
    /// Preserves any unmodeled top-level tables (`se_object`, `game_object`, `event_queue`, etc.).
    pub extra_root_entries: Vec<(ScocValue, ScocValue)>,
}

impl ScocPeerState {
    pub fn from_root_table(root: &ScocTable) -> Self {
        let mut state = Self::default();

        // 1. Extract `item_remaining_uses` first so inventory items can cross-reference IDs.
        if let Some(uses_tbl) = root.get_table("item_remaining_uses") {
            for (k, v) in &uses_tbl.entries {
                let key_id = k
                    .as_i64()
                    .or_else(|| k.as_str().and_then(|s| s.trim().parse::<i64>().ok()));
                if let (Some(id), Some(uses)) = (key_id, v.as_i64()) {
                    state.item_remaining_uses.insert(id, uses);
                } else {
                    state.item_remaining_uses_extra.push((k.clone(), v.clone()));
                }
            }
        }

        // 2. Extract player inventory items from `inventory` (or `actor_inventory`).
        if let Some(inv_tbl) = root
            .get_table("inventory")
            .or_else(|| root.get_table("actor_inventory"))
        {
            for (k, v) in &inv_tbl.entries {
                if let Some(item_tbl) = v.as_table() {
                    let fallback_id = if item_tbl.get_str("id").is_some() {
                        None
                    } else {
                        k.as_i64().filter(|&n| n >= 1000) // preserve explicit id only when non-array slot
                    };
                    let mut item = ScocInventoryItem::from_scoc_table(item_tbl, fallback_id);
                    if item.remaining_uses.is_none()
                        && let Some(id) = item.id
                        && let Some(&uses) = state.item_remaining_uses.get(&id)
                    {
                        item.remaining_uses = Some(uses);
                    }
                    state.inventory.push(item);
                } else if let Some(sec) = v.as_xray_string() {
                    state.inventory.push(ScocInventoryItem::new(sec, 1.0));
                }
            }
        }

        // 3. Extract `visited_levels` (`{ ["l01_escape"] = true }` or array `{ [1] = "l01_escape" }`).
        if let Some(lvl_tbl) = root
            .get_table("visited_levels")
            .or_else(|| root.get_table("actor_visited_levels"))
        {
            for (k, v) in &lvl_tbl.entries {
                if let Some(name) = k.as_xray_string() {
                    state.visited_levels.insert(name, v.clone());
                } else if k.as_i64().is_some()
                    && let Some(name) = v.as_xray_string()
                {
                    state.visited_levels.insert(name, ScocValue::Bool(true));
                }
            }
        }

        // 4. Extract xrRazom `skills_levels`.
        if let Some(skills_tbl) = root.get_table("skills_levels") {
            for (k, v) in &skills_tbl.entries {
                if let Some(skill_name) = k.as_xray_string()
                    && let Some(parsed) = ScocSkillLevel::from_scoc_value(v)
                {
                    state.skills_levels.insert(skill_name, parsed);
                }
            }
        }

        // 5. Extract xrRazom `psy_table`.
        if let Some(psy_tbl) = root.get_table("psy_table") {
            state.psy_table = ScocPsyTable::from_scoc_table(psy_tbl);
        }

        // 6. Extract xrRazom `head_tbl`.
        if let Some(head_tbl) = root.get_table("head_tbl") {
            state.head_tbl = ScocHeadTable::from_scoc_table(head_tbl);
        }

        // 7. Extract xrRazom `visited_campfires` (`[id] = true`, `["name"] = true`, or array `[1] = id/"name"`).
        if let Some(cf_tbl) = root.get_table("visited_campfires") {
            for (k, v) in &cf_tbl.entries {
                if let Some(name) = k.as_xray_string() {
                    state.visited_campfires.push(ScocCampfireEntry {
                        key: ScocCampfireKey::Name(name),
                        state: v.clone(),
                    });
                } else if let Some(id) = k.as_i64() {
                    if let Some(name_val) = v.as_xray_string() {
                        state.visited_campfires.push(ScocCampfireEntry {
                            key: ScocCampfireKey::Name(name_val),
                            state: ScocValue::Bool(true),
                        });
                    } else {
                        state.visited_campfires.push(ScocCampfireEntry {
                            key: ScocCampfireKey::Id(id),
                            state: v.clone(),
                        });
                    }
                }
            }
        }

        // 8. Preserve all other root entries cleanly.
        for (k, v) in &root.entries {
            let is_known = matches!(
                k.as_str(),
                Some(
                    "inventory"
                        | "actor_inventory"
                        | "item_remaining_uses"
                        | "visited_levels"
                        | "actor_visited_levels"
                        | "skills_levels"
                        | "psy_table"
                        | "head_tbl"
                        | "visited_campfires"
                )
            );
            if !is_known {
                state.extra_root_entries.push((k.clone(), v.clone()));
            }
        }

        state
    }

    pub fn write_into_root_table(&self, root: &mut ScocTable) {
        if !self.inventory.is_empty() || root.get_str("inventory").is_some() {
            let mut inv_tbl = ScocTable::with_capacity(self.inventory.len());
            for (idx, item) in self.inventory.iter().enumerate() {
                inv_tbl.set_i64((idx + 1) as i64, ScocValue::Table(item.to_scoc_table()));
            }
            root.set_str("inventory", ScocValue::Table(inv_tbl));
        }

        if !self.item_remaining_uses.is_empty()
            || !self.item_remaining_uses_extra.is_empty()
            || root.get_str("item_remaining_uses").is_some()
        {
            let mut uses_tbl = ScocTable::new();
            for (&id, &uses) in &self.item_remaining_uses {
                uses_tbl.set_i64(id, ScocValue::int(uses));
            }
            for (k, v) in &self.item_remaining_uses_extra {
                uses_tbl.entries.push((k.clone(), v.clone()));
            }
            root.set_str("item_remaining_uses", ScocValue::Table(uses_tbl));
        }

        if !self.visited_levels.is_empty() || root.get_str("visited_levels").is_some() {
            let mut lvl_tbl = ScocTable::with_capacity(self.visited_levels.len());
            for (name, val) in &self.visited_levels {
                lvl_tbl.set_str(name.clone(), val.clone());
            }
            root.set_str("visited_levels", ScocValue::Table(lvl_tbl));
        }

        if !self.skills_levels.is_empty() || root.get_str("skills_levels").is_some() {
            let mut sk_tbl = ScocTable::with_capacity(self.skills_levels.len());
            for (name, skill) in &self.skills_levels {
                sk_tbl.set_str(name.clone(), skill.to_scoc_value());
            }
            root.set_str("skills_levels", ScocValue::Table(sk_tbl));
        }

        let psy_out = self.psy_table.to_scoc_table();
        if !psy_out.is_empty() || root.get_str("psy_table").is_some() {
            root.set_str("psy_table", ScocValue::Table(psy_out));
        }

        let head_out = self.head_tbl.to_scoc_table();
        if !head_out.is_empty() || root.get_str("head_tbl").is_some() {
            root.set_str("head_tbl", ScocValue::Table(head_out));
        }

        if !self.visited_campfires.is_empty() || root.get_str("visited_campfires").is_some() {
            let mut cf_tbl = ScocTable::with_capacity(self.visited_campfires.len());
            for entry in &self.visited_campfires {
                match &entry.key {
                    ScocCampfireKey::Id(id) => cf_tbl.set_i64(*id, entry.state.clone()),
                    ScocCampfireKey::Name(name) => cf_tbl.set_str(name.clone(), entry.state.clone()),
                }
            }
            root.set_str("visited_campfires", ScocValue::Table(cf_tbl));
        }

        for (k, v) in &self.extra_root_entries {
            let key_exists = match k {
                ScocValue::String(bytes) => std::str::from_utf8(bytes)
                    .ok()
                    .and_then(|s| root.get_str(s))
                    .is_some(),
                _ => false,
            };
            if !key_exists {
                root.entries.push((k.clone(), v.clone()));
            }
        }
    }

    #[inline]
    pub fn is_level_visited(&self, level_name: &str) -> bool {
        match self.visited_levels.get(level_name) {
            Some(ScocValue::Bool(b)) => *b,
            Some(ScocValue::Number(n)) => *n > 0.0,
            Some(ScocValue::Nil) | None => false,
            Some(_) => true,
        }
    }

    #[inline]
    pub fn mark_level_visited(&mut self, level_name: impl Into<String>) {
        self.visited_levels
            .insert(level_name.into(), ScocValue::Bool(true));
    }

    #[inline]
    pub fn is_campfire_visited_id(&self, id: i64) -> bool {
        self.visited_campfires
            .iter()
            .any(|e| matches!(e.key, ScocCampfireKey::Id(k) if k == id))
    }

    #[inline]
    pub fn is_campfire_visited_name(&self, name: &str) -> bool {
        self.visited_campfires
            .iter()
            .any(|e| matches!(&e.key, ScocCampfireKey::Name(k) if k == name))
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, FormatError> {
        ScocFile::from_peer_state(self).to_bytes()
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FormatError> {
        Ok(ScocFile::from_bytes(bytes)?.extract_peer_state())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoc_wire_format_header_and_all_primitive_types_roundtrip() {
        let mut inner = ScocTable::new();
        inner.set_i64(1, ScocValue::string("l01_escape"));
        inner.set_i64(2, ScocValue::int(-42));
        inner.set_str("active", ScocValue::bool(true));
        inner.set_str("weight", ScocValue::float(3.25));

        let mut root = ScocTable::new();
        root.set_str("flag_true", ScocValue::bool(true));
        root.set_str("flag_false", ScocValue::bool(false));
        root.set_str("float_val", ScocValue::float(0.875));
        root.set_str("int_val", ScocValue::int(123_456_789));
        root.set_str("text_val", ScocValue::string("wpn_ak74"));
        root.set_str("nested", ScocValue::Table(inner.clone()));

        let file = ScocFile::new(root);
        let bytes = file.to_bytes().expect("encode .scoc");

        // Verify exact lmarshal binary header: [0x8e, 0x05, 0x02, u32 payload_len]
        assert_eq!(bytes[0], 0x8e);
        assert_eq!(bytes[1], 0x05);
        assert_eq!(bytes[2], 0x02);
        let payload_len = u32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]) as usize;
        assert_eq!(payload_len, bytes.len() - 7);
        assert!(validate_scoc_header(&bytes).is_some());

        let decoded = ScocFile::parse(&bytes).expect("decode .scoc");
        assert_eq!(decoded, file);
        assert_eq!(decoded.root.get_str("flag_true").unwrap().as_bool(), Some(true));
        assert_eq!(decoded.root.get_str("flag_false").unwrap().as_bool(), Some(false));
        assert_eq!(decoded.root.get_str("float_val").unwrap().as_f64(), Some(0.875));
        assert_eq!(decoded.root.get_str("int_val").unwrap().as_i64(), Some(123_456_789));
        assert_eq!(decoded.root.get_str("text_val").unwrap().as_str(), Some("wpn_ak74"));
        assert_eq!(decoded.root.get_table("nested"), Some(&inner));

        // Verify generic `marshal.encode` / `marshal.decode` entrypoints for scalar roots too.
        let scalar_bytes = encode_marshal_value(&ScocValue::float(42.5)).unwrap();
        assert_eq!(
            decode_marshal_value(&scalar_bytes).unwrap(),
            ScocValue::float(42.5)
        );
    }

    #[test]
    fn scoc_resolves_mar_tref_shared_subtables_and_rejects_bad_inputs() {
        // Construct raw lmarshal stream matching C `lmarshal.c` slot numbering:
        // Slot 1 = root table
        // Slot 2 = shared subtable { ["shared"] = true }
        // Key "a" -> Slot 2 (`MAR_TVAL`), Key "b" -> `MAR_TREF(2)`
        let mut shared_payload = Vec::new();
        encode_value_depth(&ScocValue::string("shared"), &mut shared_payload, 0).unwrap();
        encode_value_depth(&ScocValue::bool(true), &mut shared_payload, 0).unwrap();

        let mut root_payload = Vec::new();
        encode_value_depth(&ScocValue::string("a"), &mut root_payload, 0).unwrap();
        root_payload.push(LUA_TTABLE);
        root_payload.push(MAR_TVAL);
        root_payload.extend_from_slice(&(shared_payload.len() as u32).to_le_bytes());
        root_payload.extend_from_slice(&shared_payload);

        encode_value_depth(&ScocValue::string("b"), &mut root_payload, 0).unwrap();
        root_payload.push(LUA_TTABLE);
        root_payload.push(MAR_TREF);
        root_payload.extend_from_slice(&2u32.to_le_bytes());

        let mut raw = Vec::new();
        raw.push(SCOC_MAGIC);
        raw.push(LUA_TTABLE);
        raw.push(MAR_TVAL);
        raw.extend_from_slice(&(root_payload.len() as u32).to_le_bytes());
        raw.extend_from_slice(&root_payload);

        let parsed = ScocFile::parse(&raw).expect("decode MAR_TREF stream");
        assert_eq!(
            parsed.root.get_table("a").and_then(|t| t.get_str("shared")).and_then(ScocValue::as_bool),
            Some(true)
        );
        assert_eq!(parsed.root.get_table("b"), parsed.root.get_table("a"));

        // Corrupt headers & truncations return structured errors rather than panicking.
        assert!(ScocFile::parse(&[]).is_err());
        assert!(ScocFile::parse(&[0x00, 0x05, 0x02, 0, 0, 0, 0]).is_err());
        assert!(ScocFile::parse(&[0x8e, 0x04, 0x02, 0, 0, 0, 0]).is_err());
        assert!(ScocFile::parse(&[0x8e, 0x05, 0x01, 0, 0, 0, 0]).is_err());
        assert!(ScocFile::parse(&[0x8e, 0x05, 0x02, 10, 0, 0, 0, 0]).is_err());
    }

    #[test]
    fn scoc_xrrazom_peer_state_roundtrip() {
        let mut peer = ScocPeerState::default();

        let mut ak74 = ScocInventoryItem::new("wpn_ak74", 0.91);
        ak74.id = Some(4096);
        ak74.slot = Some(2);
        ak74.ammo_elapsed = Some(30);
        ak74.ammo_type = Some(0);
        ak74.scope_attached = Some(true);
        ak74.silencer_attached = Some(false);
        ak74.upgrades = vec!["up_firsta_ak74".to_string(), "up_seconc_ak74".to_string()];

        let mut medkit = ScocInventoryItem::new("medkit_army", 0.75);
        medkit.id = Some(4097);
        medkit.remaining_uses = Some(2);

        peer.inventory.push(ak74);
        peer.inventory.push(medkit);
        peer.item_remaining_uses.insert(4097, 2);
        peer.item_remaining_uses.insert(5100, 4);

        peer.mark_level_visited("l01_escape");
        peer.mark_level_visited("l02_garbage");
        peer.visited_levels
            .insert("l03_agroprom".to_string(), ScocValue::int(2));

        peer.skills_levels
            .insert("scavenging".to_string(), ScocSkillLevel::new(4, 315.5));
        peer.skills_levels
            .insert("survival".to_string(), ScocSkillLevel::new(7, 940.25));

        peer.psy_table.psy_health = Some(0.84);
        peer.psy_table.active_phantoms = Some(1);
        peer.psy_table.in_psy_zone = Some(false);
        peer.psy_table
            .extra
            .set_str("psy_storm_intensity", ScocValue::float(0.35));

        peer.head_tbl.head = Some("stalker_neutral_head_3".to_string());
        peer.head_tbl.visual = Some("actors\\stalker_hero\\stalker_hero_exo".to_string());
        peer.head_tbl.faction = Some("stalker".to_string());
        peer.head_tbl.extra.set_str("skin_variant", ScocValue::int(2));

        peer.visited_campfires.push(ScocCampfireEntry::by_id(1820));
        peer.visited_campfires
            .push(ScocCampfireEntry::by_name("esc_smart_terrain_5_7_campfire"));

        peer.extra_root_entries
            .push((ScocValue::string("campaign_day"), ScocValue::int(19)));

        let bytes = peer.to_bytes().expect("encode peer .scoc");
        assert!(validate_scoc_header(&bytes).is_some());

        let decoded = ScocPeerState::from_bytes(&bytes).expect("decode peer .scoc");
        assert_eq!(decoded, peer);
        assert!(decoded.is_level_visited("l01_escape"));
        assert!(decoded.is_level_visited("l03_agroprom"));
        assert!(!decoded.is_level_visited("l12_stancia"));
        assert!(decoded.is_campfire_visited_id(1820));
        assert!(decoded.is_campfire_visited_name("esc_smart_terrain_5_7_campfire"));
        assert_eq!(decoded.inventory[0].upgrades.len(), 2);
        assert_eq!(decoded.inventory[1].remaining_uses, Some(2));
        assert_eq!(decoded.skills_levels["scavenging"].current_level, 4);
        assert_eq!(decoded.psy_table.psy_health, Some(0.84));
        assert_eq!(
            decoded.head_tbl.visual.as_deref(),
            Some("actors\\stalker_hero\\stalker_hero_exo")
        );

        // Verify raw array-style `visited_levels` and scalar `skills_levels` normalize cleanly too.
        let mut alt_root = ScocTable::new();
        let mut arr_levels = ScocTable::new();
        arr_levels.set_i64(1, ScocValue::string("jupiter"));
        alt_root.set_str("visited_levels", ScocValue::Table(arr_levels));
        let mut scalar_skills = ScocTable::new();
        scalar_skills.set_str("tinkering", ScocValue::int(5));
        alt_root.set_str("skills_levels", ScocValue::Table(scalar_skills));
        let cp1251_bytes = vec![0xCA, 0xEE, 0xF0, 0xE4, 0xEE, 0xED]; // "Кордон" in CP1251
        alt_root.set_str("zone_ru", ScocValue::bytes(cp1251_bytes));

        let alt_bytes = ScocFile::new(alt_root).to_bytes().unwrap();
        let alt_file = ScocFile::parse(&alt_bytes).unwrap();
        assert_eq!(alt_file.root.get_str("zone_ru").unwrap().as_xray_string().as_deref(), Some("Кордон"));
        let alt_peer = alt_file.extract_peer_state();
        assert!(alt_peer.is_level_visited("jupiter"));
        assert_eq!(alt_peer.skills_levels["tinkering"].current_level, 5);
        assert!(alt_peer.skills_levels["tinkering"].scalar_only);

        // Verify both `name`-first + `section` preservation and `name`-only fallback in inventory items.
        let mut with_both = ScocTable::new();
        with_both.set_str("name", ScocValue::string("wpn_ak74_m1_4096"));
        with_both.set_str("section", ScocValue::string("wpn_ak74_m1"));
        with_both.set_str("condition", ScocValue::float(0.65));
        let parsed_both = ScocInventoryItem::from_scoc_table(&with_both, None);
        assert_eq!(parsed_both.section, "wpn_ak74_m1");
        assert_eq!(
            parsed_both.extra.get_str("name").and_then(ScocValue::as_str),
            Some("wpn_ak74_m1_4096")
        );

        let mut name_only = ScocTable::new();
        name_only.set_str("name", ScocValue::string("bandage"));
        name_only.set_str("condition", ScocValue::float(1.0));
        let parsed_name_only = ScocInventoryItem::from_scoc_table(&name_only, None);
        assert_eq!(parsed_name_only.section, "bandage");
        assert!(parsed_name_only.extra.is_empty());
    }
}
