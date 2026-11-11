//! Pure Rust X-Ray `lanims.xr` Light/UI Color Animation Library Parser & Evaluator
//!
//! Faithful implementation matching `src/xrEngine/LightAnimLibrary.h` and
//! `src/xrEngine/LightAnimLibrary.cpp`:
//! - Binary `lanims.xr` format (`LANIM_VERSION = 0x0001`, v0 BGR->RGB auto-upgrade)
//! - Keyframed ARGB color splines (`CLAItem`: `fFPS`, `iFrameCount`, `Keys`)
//! - Exact frame interpolation (`InterpolateRGB`/`InterpolateBGR`) & looping playback
//!   (`CalculateRGB`/`CalculateBGR` + normalized `[r, g, b, a]` `f32` helper)

use std::collections::BTreeMap;

use crate::chunk::{ByteCursor, ChunkReader, ChunkWriter, FormatError};

pub const LANIM_VERSION: u16 = 0x0001;

pub mod lanim_chunks {
    pub const VERSION: u32 = 0x0000;
    pub const ITEM_LIST: u32 = 0x0001;
}

pub mod lanim_item_chunks {
    pub const COMMON: u32 = 0x0001;
    pub const KEYS: u32 = 0x0002;
}

/// Pack 8-bit `R, G, B, A` channels into D3DCOLOR `0xAARRGGBB` (`color_rgba` in `_color.h`).
#[inline]
pub const fn pack_argb(a: u8, r: u8, g: u8, b: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Unpack D3DCOLOR `0xAARRGGBB` into normalized linear floats `[r, g, b, a]` in `0.0..=1.0`.
#[inline]
pub fn unpack_argb_f32(argb: u32) -> [f32; 4] {
    const INV_255: f32 = 1.0 / 255.0;
    [
        ((argb >> 16) & 0xFF) as f32 * INV_255,
        ((argb >> 8) & 0xFF) as f32 * INV_255,
        (argb & 0xFF) as f32 * INV_255,
        ((argb >> 24) & 0xFF) as f32 * INV_255,
    ]
}

/// Pack normalized `[r, g, b, a]` floats into D3DCOLOR `0xAARRGGBB` matching `Fcolor::get()`.
#[inline]
pub fn pack_rgba_f32(rgba: [f32; 4]) -> u32 {
    let quant = |c: f32| -> u8 { (c * 255.0).floor().clamp(0.0, 255.0) as u8 };
    pack_argb(quant(rgba[3]), quant(rgba[0]), quant(rgba[1]), quant(rgba[2]))
}

/// Swap R and B channels while preserving A and G (`subst_alpha(bgr2rgb(c), color_get_A(c))`).
#[inline]
pub const fn swap_rb_keep_ag(color: u32) -> u32 {
    (color & 0xFF00_FF00) | ((color & 0x00FF_0000) >> 16) | ((color & 0x0000_00FF) << 16)
}

/// Single keyframed light/UI color animation (`CLAItem`).
#[derive(Debug, Clone, PartialEq)]
pub struct LightAnimItem {
    pub name: String,
    pub fps: f32,
    pub default_fps: f32,
    pub frame_count: u32,
    /// Map of `frame -> ARGB (0xAARRGGBB)`.
    pub keys: BTreeMap<u32, u32>,
}

impl Default for LightAnimItem {
    fn default() -> Self {
        let mut keys = BTreeMap::new();
        keys.insert(0, 0x0000_0000);
        Self {
            name: String::new(),
            fps: 15.0,
            default_fps: 15.0,
            frame_count: 1,
            keys,
        }
    }
}

impl LightAnimItem {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    pub fn init_default(&mut self) {
        self.keys.insert(0, 0x0000_0000);
    }

    /// Parse one `CLAItem` chunk payload (`CHUNK_ITEM_COMMON` + `CHUNK_ITEM_KEYS`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let common = reader.require_chunk(lanim_item_chunks::COMMON)?;
        let mut cur = ByteCursor::new(&common);
        let name = cur.read_zstring()?;
        let fps = cur.read_f32()?;
        let frame_count = cur.read_u32()?;

        let keys_bytes = reader.require_chunk(lanim_item_chunks::KEYS)?;
        let mut kc = ByteCursor::new(&keys_bytes);
        let key_cnt = kc.read_u32()? as usize;
        if kc.remaining() < key_cnt.saturating_mul(8) {
            return Err(FormatError::UnexpectedEof);
        }
        let mut keys = BTreeMap::new();
        for _ in 0..key_cnt {
            let frame = kc.read_u32()?;
            let color = kc.read_u32()?;
            keys.insert(frame, color);
        }

        Ok(Self {
            name,
            fps,
            default_fps: fps,
            frame_count,
            keys,
        })
    }

    /// Encode this animation into `CLAItem::Save` chunk bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = ChunkWriter::new();

        let mut common = Vec::with_capacity(self.name.len() + 9);
        common.extend_from_slice(self.name.as_bytes());
        common.push(0);
        common.extend_from_slice(&self.fps.to_le_bytes());
        common.extend_from_slice(&self.frame_count.to_le_bytes());
        w.write_chunk(lanim_item_chunks::COMMON, &common);

        let mut keys_buf = Vec::with_capacity(4 + self.keys.len() * 8);
        keys_buf.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        for (&frame, &color) in &self.keys {
            keys_buf.extend_from_slice(&frame.to_le_bytes());
            keys_buf.extend_from_slice(&color.to_le_bytes());
        }
        w.write_chunk(lanim_item_chunks::KEYS, &keys_buf);

        w.into_bytes()
    }

    #[inline]
    pub fn length_sec(&self) -> f32 {
        if self.fps > 0.0 {
            (self.frame_count as f32) / self.fps
        } else {
            0.0
        }
    }

    #[inline]
    pub fn length_ms(&self) -> u32 {
        (self.length_sec() * 1000.0).floor().max(0.0) as u32
    }

    #[inline]
    pub fn set_framerate(&mut self, fps: f32) {
        self.fps = fps;
    }

    #[inline]
    pub fn reset_framerate(&mut self) {
        self.fps = self.default_fps;
    }

    pub fn insert_key(&mut self, frame: u32, color: u32) -> Result<(), FormatError> {
        if frame > self.frame_count {
            return Err(FormatError::InvalidValue(
                "light anim key frame exceeds frame_count",
            ));
        }
        self.keys.insert(frame, color);
        Ok(())
    }

    pub fn delete_key(&mut self, frame: u32) {
        if frame == 0 || frame > self.frame_count {
            return;
        }
        self.keys.remove(&frame);
    }

    pub fn move_key(&mut self, from: u32, to: u32) -> bool {
        if from > self.frame_count || to > self.frame_count {
            return false;
        }
        if let Some(col) = self.keys.remove(&from) {
            self.keys.insert(to, col);
            true
        } else {
            false
        }
    }

    pub fn resize(&mut self, new_len: u32) {
        let new_len = new_len.max(1);
        if new_len == self.frame_count {
            return;
        }
        if new_len > self.frame_count {
            let old_len = self.frame_count;
            self.frame_count = new_len;
            self.move_key(old_len, new_len);
        } else {
            self.keys.retain(|&f, _| f < new_len);
            self.frame_count = new_len;
        }
    }

    #[inline]
    pub fn is_key(&self, frame: u32) -> bool {
        self.keys.contains_key(&frame)
    }

    #[inline]
    pub fn first_key_frame(&self) -> Option<u32> {
        self.keys.keys().next().copied()
    }

    #[inline]
    pub fn last_key_frame(&self) -> Option<u32> {
        self.keys.keys().next_back().copied()
    }

    pub fn prev_key_frame(&self, frame: u32) -> Option<u32> {
        self.keys.range(..frame).next_back().map(|(&k, _)| k).or_else(|| {
            if self.keys.contains_key(&frame) {
                Some(frame)
            } else {
                self.last_key_frame()
            }
        })
    }

    pub fn next_key_frame(&self, frame: u32) -> Option<u32> {
        self.keys
            .range((frame + 1)..)
            .next()
            .map(|(&k, _)| k)
            .or_else(|| self.last_key_frame())
    }

    /// Linearly interpolate ARGB `0xAARRGGBB` at integer `frame` matching `CLAItem::InterpolateRGB`.
    pub fn interpolate_rgb(&self, frame: u32) -> u32 {
        if self.keys.is_empty() {
            return 0;
        }
        if let Some(&exact) = self.keys.get(&frame) {
            return exact;
        }
        let Some((&a_frame, &a_col)) = self.keys.range(..frame).next_back() else {
            // Safely clamp to earliest key if frame precedes the first defined key
            return self.keys.values().next().copied().unwrap_or(0);
        };
        let Some((&b_frame, &b_col)) = self.keys.range((frame + 1)..).next() else {
            // No subsequent key -> return last key color
            return a_col;
        };

        let c0 = unpack_argb_f32(a_col);
        let c1 = unpack_argb_f32(b_col);
        let denom = (b_frame - a_frame) as f32;
        let t = if denom > 0.0 {
            (frame - a_frame) as f32 / denom
        } else {
            0.0
        };
        let lerped = [
            c0[0] + (c1[0] - c0[0]) * t,
            c0[1] + (c1[1] - c0[1]) * t,
            c0[2] + (c1[2] - c0[2]) * t,
            c0[3] + (c1[3] - c0[3]) * t,
        ];
        pack_rgba_f32(lerped)
    }

    /// Linearly interpolate BGR-swapped `0xAABBGGRR` at `frame` matching `CLAItem::InterpolateBGR`.
    #[inline]
    pub fn interpolate_bgr(&self, frame: u32) -> u32 {
        swap_rb_keep_ag(self.interpolate_rgb(frame))
    }

    /// Wrap continuous time `t_sec` into frame index and return `(argb_u32, frame)` matching
    /// `CLAItem::CalculateRGB`.
    pub fn calculate_rgb(&self, t_sec: f32) -> (u32, u32) {
        let frame = self.frame_at_time(t_sec);
        (self.interpolate_rgb(frame), frame)
    }

    /// Wrap continuous time `t_sec` into frame index and return `(abgr_u32, frame)` matching
    /// `CLAItem::CalculateBGR`.
    pub fn calculate_bgr(&self, t_sec: f32) -> (u32, u32) {
        let frame = self.frame_at_time(t_sec);
        (self.interpolate_bgr(frame), frame)
    }

    /// Continuous `f32` RGBA `[r, g, b, a]` at `t_sec` with smooth intra-frame linear blending
    /// or exact integer-frame snapping (`smooth = false` matches C++ `iFloor` step exactly).
    pub fn sample_rgba_f32(&self, t_sec: f32, smooth: bool) -> [f32; 4] {
        if self.keys.is_empty() {
            return [0.0; 4];
        }
        if !smooth || self.fps <= 0.0 || self.frame_count == 0 {
            let (argb, _) = self.calculate_rgb(t_sec);
            return unpack_argb_f32(argb);
        }
        let dur = (self.frame_count as f32) / self.fps;
        if dur <= 0.0 {
            return unpack_argb_f32(self.interpolate_rgb(0));
        }
        let local_frame = t_sec.rem_euclid(dur) * self.fps;
        let f_clamped = local_frame.clamp(0.0, self.frame_count as f32);
        let lo_idx = (f_clamped.floor() as u32).min(self.frame_count);
        let hi_idx = (lo_idx + 1).min(self.frame_count);
        let frac = f_clamped - (lo_idx as f32);
        let c0 = unpack_argb_f32(self.interpolate_rgb(lo_idx));
        let c1 = unpack_argb_f32(self.interpolate_rgb(hi_idx));
        [
            c0[0] + (c1[0] - c0[0]) * frac,
            c0[1] + (c1[1] - c0[1]) * frac,
            c0[2] + (c1[2] - c0[2]) * frac,
            c0[3] + (c1[3] - c0[3]) * frac,
        ]
    }

    fn frame_at_time(&self, t_sec: f32) -> u32 {
        if self.fps <= 0.0 || self.frame_count == 0 {
            return 0;
        }
        let dur = (self.frame_count as f32) / self.fps;
        if dur <= 0.0 || !t_sec.is_finite() {
            return 0;
        }
        let wrapped = t_sec.rem_euclid(dur);
        let f = (wrapped * self.fps + 1e-5).floor().max(0.0) as u32;
        f % self.frame_count
    }
}

/// Complete `$game_data$/lanims.xr` library (`ELightAnimLibrary`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LightAnimLibrary {
    pub version: u16,
    pub items: Vec<LightAnimItem>,
}

impl LightAnimLibrary {
    /// Parse a binary `lanims.xr` stream (`ELightAnimLibrary::Load`), automatically upgrading v0
    /// BGR key colors to v1 ARGB layout.
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let version = match reader.open_chunk(lanim_chunks::VERSION)? {
            Some(bytes) => ByteCursor::new(&bytes).read_u16()?,
            None => 0,
        };

        let mut items = Vec::new();
        if let Some(list_bytes) = reader.open_chunk(lanim_chunks::ITEM_LIST)? {
            let sub = ChunkReader::new(&list_bytes);
            for entry in sub.iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                let mut item = LightAnimItem::parse(&payload)?;
                if version == 0 {
                    for col in item.keys.values_mut() {
                        *col = swap_rb_keep_ag(*col);
                    }
                }
                items.push(item);
            }
        }

        Ok(Self { version, items })
    }

    /// Encode into `lanims.xr` binary chunk layout (always writes `LANIM_VERSION = 0x0001`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut root = ChunkWriter::new();
        root.write_chunk(lanim_chunks::VERSION, &LANIM_VERSION.to_le_bytes());

        let mut list_writer = ChunkWriter::new();
        for (idx, item) in self.items.iter().enumerate() {
            list_writer.write_chunk(idx as u32, &item.to_bytes());
        }
        root.write_chunk(lanim_chunks::ITEM_LIST, &list_writer.into_bytes());
        root.into_bytes()
    }

    pub fn find_item(&self, name: &str) -> Option<&LightAnimItem> {
        self.items.iter().find(|it| it.name == name)
    }

    pub fn find_item_ci(&self, name: &str) -> Option<&LightAnimItem> {
        self.items
            .iter()
            .find(|it| it.name.eq_ignore_ascii_case(name))
    }

    pub fn append_item(&mut self, item: LightAnimItem) -> Result<(), FormatError> {
        if self.find_item(&item.name).is_some() {
            return Err(FormatError::InvalidValue("duplicate light animation name"));
        }
        self.items.push(item);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lanims_binary_roundtrip_and_keyframe_interpolation() {
        let mut flicker = LightAnimItem::new("lights\\flicker_01");
        flicker.fps = 20.0;
        flicker.default_fps = 20.0;
        flicker.frame_count = 10;
        flicker.keys.clear();
        flicker.insert_key(0, pack_argb(0xFF, 0x00, 0x40, 0x80)).unwrap();
        flicker.insert_key(4, pack_argb(0xFF, 0x80, 0xC0, 0x00)).unwrap();
        flicker.insert_key(10, pack_argb(0xFF, 0xFF, 0x00, 0xFF)).unwrap();

        let mut pulse = LightAnimItem::new("ui\\hud_pulse");
        pulse.fps = 10.0;
        pulse.default_fps = 10.0;
        pulse.frame_count = 4;
        pulse.keys.clear();
        pulse.insert_key(0, pack_argb(0x20, 0xFF, 0x80, 0x00)).unwrap();
        pulse.insert_key(4, pack_argb(0xFF, 0xFF, 0x80, 0x00)).unwrap();

        let lib = LightAnimLibrary {
            version: LANIM_VERSION,
            items: vec![flicker, pulse],
        };
        let encoded = lib.to_bytes();
        let parsed = LightAnimLibrary::parse(&encoded).unwrap();
        assert_eq!(parsed.version, LANIM_VERSION);
        assert_eq!(parsed.items.len(), 2);

        let f = parsed.find_item("lights\\flicker_01").unwrap();
        assert_eq!(f.length_ms(), 500);
        // Exact keys
        assert_eq!(f.interpolate_rgb(0), pack_argb(0xFF, 0x00, 0x40, 0x80));
        assert_eq!(f.interpolate_rgb(4), pack_argb(0xFF, 0x80, 0xC0, 0x00));
        // Midpoint frame 2 between frame 0 and 4: R=64 (0x40), G=128 (0x80), B=64 (0x40)
        assert_eq!(f.interpolate_rgb(2), pack_argb(0xFF, 0x40, 0x80, 0x40));
        // BGR swap check keeps alpha 0xFF and G 0x80 while swapping R/B
        assert_eq!(f.interpolate_bgr(0), pack_argb(0xFF, 0x80, 0x40, 0x00));

        // Time wraparound: duration = 10 / 20 = 0.5s, so t = 0.6s wraps to 0.1s -> frame 2
        let (argb, frame) = f.calculate_rgb(0.6);
        assert_eq!(frame, 2);
        assert_eq!(argb, pack_argb(0xFF, 0x40, 0x80, 0x40));

        // Negative time safety via rem_euclid
        let (neg_argb, neg_frame) = f.calculate_rgb(-0.4);
        assert_eq!(neg_frame, 2);
        assert_eq!(neg_argb, pack_argb(0xFF, 0x40, 0x80, 0x40));

        // Resize shrinks keys beyond new_len - 1 just like CLAItem::Resize
        let mut resized = f.clone();
        resized.resize(5);
        assert!(!resized.is_key(10));
        assert!(resized.is_key(4));
    }

    #[test]
    fn test_lanims_v0_upgrades_bgr_channels_to_rgb() {
        let mut item = LightAnimItem::new("v0_torch");
        item.fps = 15.0;
        item.frame_count = 5;
        item.keys.clear();
        // Legacy v0 stored (A=0xCC, B=0x11, G=0x22, R=0x99) -> encoded bytes as 0xCC112299
        item.keys.insert(0, pack_argb(0xCC, 0x11, 0x22, 0x99));

        let mut root = ChunkWriter::new();
        root.write_chunk(lanim_chunks::VERSION, &0u16.to_le_bytes());
        let mut list = ChunkWriter::new();
        list.write_chunk(0, &item.to_bytes());
        root.write_chunk(lanim_chunks::ITEM_LIST, &list.into_bytes());

        let parsed = LightAnimLibrary::parse(&root.into_bytes()).unwrap();
        assert_eq!(parsed.version, 0);
        let upgraded = parsed.find_item("v0_torch").unwrap();
        assert_eq!(upgraded.interpolate_rgb(0), pack_argb(0xCC, 0x99, 0x22, 0x11));
    }
}
