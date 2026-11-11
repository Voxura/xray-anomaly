//! Pure Rust X-Ray Envelope Splines (`CEnvelope`), Object Motions (`.anm`), and Postprocess Effects (`.ppe`)
//!
//! Covers:
//! - LightWave `CEnvelope` spline parser (`Load_1`, `Load_2` w/ `q16` [-32..32] quantization, `LoadA` ASCII)
//! - Exact `evalEnvelope` evaluator across all 6 shapes (`TCB`, `Hermite`, `Bezier`, `Linear`, `Step`, `Bezier2D`)
//!   and all 6 pre/post extrapolation behaviors (`Reset`, `Constant`, `Repeat`, `Oscillate`, `OffsetRepeat`, `Linear`)
//! - `ObjectMotion` (`.anm`, `EOBJ_OMOTION` `0x1100`, versions `0x0003`..`0x0005` incl. v4 swapped `RotP`/`RotH`)
//! - `PostprocessEffect` (`.ppe` v1/v2 `CPostprocessAnimator`) + `PostprocessState` (`SPPInfo::identity` / `lerp`)

use crate::chunk::{ByteCursor, ChunkReader, ChunkWriter, FormatError};

pub const EOBJ_OMOTION: u32 = 0x1100;
pub const EOBJ_OMOTION_VERSION_V3: u16 = 0x0003;
pub const EOBJ_OMOTION_VERSION_V4: u16 = 0x0004;
pub const EOBJ_OMOTION_VERSION: u16 = 0x0005;

pub const POSTPROCESS_FILE_VERSION_V1: u32 = 0x0001;
pub const POSTPROCESS_FILE_VERSION: u32 = 0x0002;

const Q16_MIN: f32 = -32.0;
const Q16_MAX: f32 = 32.0;

/// Decode X-Ray's 16-bit fixed-point quantized float (`IReader::r_float_q16`).
#[inline]
pub fn decode_q16(raw: u16, min: f32, max: f32) -> f32 {
    (f32::from(raw) * (max - min)) / 65535.0 + min
}

/// Encode a float into X-Ray's 16-bit fixed-point quantized representation (`IWriter::w_float_q16`).
#[inline]
pub fn encode_q16(value: f32, min: f32, max: f32) -> u16 {
    let span = max - min;
    if span <= 0.0 || !value.is_finite() {
        return 0;
    }
    let q = ((value - min) / span).clamp(0.0, 1.0);
    (q * 65535.0 + 0.5).floor().clamp(0.0, 65535.0) as u16
}

/// Interpolation curve shape per `st_Key` (`SHAPE_*` in `envelope.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EnvelopeShape {
    Tcb = 0,
    Hermite = 1,
    Bezier = 2,
    Linear = 3,
    Step = 4,
    Bezier2D = 5,
}

impl EnvelopeShape {
    pub fn from_u8(v: u8) -> Result<Self, FormatError> {
        match v {
            0 => Ok(Self::Tcb),
            1 => Ok(Self::Hermite),
            2 => Ok(Self::Bezier),
            3 => Ok(Self::Linear),
            4 => Ok(Self::Step),
            5 => Ok(Self::Bezier2D),
            _ => Err(FormatError::InvalidValue("unknown CEnvelope key shape")),
        }
    }
}

/// Pre/post time extrapolation behavior (`BEH_*` in `envelope.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EnvelopeBehavior {
    Reset = 0,
    Constant = 1,
    Repeat = 2,
    Oscillate = 3,
    OffsetRepeat = 4,
    Linear = 5,
}

impl EnvelopeBehavior {
    pub fn from_u32(v: u32) -> Result<Self, FormatError> {
        match v {
            0 => Ok(Self::Reset),
            1 => Ok(Self::Constant),
            2 => Ok(Self::Repeat),
            3 => Ok(Self::Oscillate),
            4 => Ok(Self::OffsetRepeat),
            5 => Ok(Self::Linear),
            _ => Err(FormatError::InvalidValue("unknown CEnvelope behavior")),
        }
    }
}

/// Single keyframe in a LightWave/X-Ray `CEnvelope` (`st_Key`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvelopeKey {
    pub value: f32,
    pub time: f32,
    pub shape: EnvelopeShape,
    pub tension: f32,
    pub continuity: f32,
    pub bias: f32,
    pub param: [f32; 4],
}

impl Default for EnvelopeKey {
    fn default() -> Self {
        Self {
            value: 0.0,
            time: 0.0,
            shape: EnvelopeShape::Tcb,
            tension: 0.0,
            continuity: 0.0,
            bias: 0.0,
            param: [0.0; 4],
        }
    }
}

impl EnvelopeKey {
    /// Legacy uncompressed key layout (`st_Key::Load_1`, 36 bytes).
    pub fn read_load1(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let value = cur.read_f32()?;
        let time = cur.read_f32()?;
        let shape = EnvelopeShape::from_u8((cur.read_u32()? & 0xFF) as u8)?;
        let tension = cur.read_f32()?;
        let continuity = cur.read_f32()?;
        let bias = cur.read_f32()?;
        let param = cur.read_f32_4()?;
        Ok(Self {
            value,
            time,
            shape,
            tension,
            continuity,
            bias,
            param,
        })
    }

    /// Standard X-Ray compressed key layout (`st_Key::Load_2`, 9 or 23 bytes).
    pub fn read_load2(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let value = cur.read_f32()?;
        let time = cur.read_f32()?;
        let shape = EnvelopeShape::from_u8(cur.read_u8()?)?;
        if shape != EnvelopeShape::Step {
            let tension = decode_q16(cur.read_u16()?, Q16_MIN, Q16_MAX);
            let continuity = decode_q16(cur.read_u16()?, Q16_MIN, Q16_MAX);
            let bias = decode_q16(cur.read_u16()?, Q16_MIN, Q16_MAX);
            let p0 = decode_q16(cur.read_u16()?, Q16_MIN, Q16_MAX);
            let p1 = decode_q16(cur.read_u16()?, Q16_MIN, Q16_MAX);
            let p2 = decode_q16(cur.read_u16()?, Q16_MIN, Q16_MAX);
            let p3 = decode_q16(cur.read_u16()?, Q16_MIN, Q16_MAX);
            Ok(Self {
                value,
                time,
                shape,
                tension,
                continuity,
                bias,
                param: [p0, p1, p2, p3],
            })
        } else {
            Ok(Self {
                value,
                time,
                shape,
                tension: 0.0,
                continuity: 0.0,
                bias: 0.0,
                param: [0.0; 4],
            })
        }
    }

    /// Serialize in `st_Key::Save` (`Load_2`) binary format.
    pub fn write_load2(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.value.to_le_bytes());
        out.extend_from_slice(&self.time.to_le_bytes());
        out.push(self.shape as u8);
        if self.shape != EnvelopeShape::Step {
            for val in [
                self.tension,
                self.continuity,
                self.bias,
                self.param[0],
                self.param[1],
                self.param[2],
                self.param[3],
            ] {
                out.extend_from_slice(&encode_q16(val, Q16_MIN, Q16_MAX).to_le_bytes());
            }
        }
    }
}

/// Time-varying scalar spline curve (`CEnvelope`).
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    pub behavior: [EnvelopeBehavior; 2],
    pub keys: Vec<EnvelopeKey>,
}

impl Default for Envelope {
    fn default() -> Self {
        Self {
            behavior: [EnvelopeBehavior::Constant, EnvelopeBehavior::Constant],
            keys: Vec::new(),
        }
    }
}

impl Envelope {
    /// Read legacy `CEnvelope::Load_1` layout.
    pub fn parse_load1(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let b0 = EnvelopeBehavior::from_u32(cur.read_u32()?)?;
        let b1 = EnvelopeBehavior::from_u32(cur.read_u32()?)?;
        let count = cur.read_u32()? as usize;
        let mut keys = Vec::with_capacity(count.min(65_536));
        for _ in 0..count {
            keys.push(EnvelopeKey::read_load1(cur)?);
        }
        Ok(Self {
            behavior: [b0, b1],
            keys,
        })
    }

    /// Read standard `CEnvelope::Load_2` layout (used by `.anm` v4/v5 and `.ppe`).
    pub fn parse_load2(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let b0 = EnvelopeBehavior::from_u32(u32::from(cur.read_u8()?))?;
        let b1 = EnvelopeBehavior::from_u32(u32::from(cur.read_u8()?))?;
        let count = cur.read_u16()? as usize;
        let mut keys = Vec::with_capacity(count);
        for _ in 0..count {
            keys.push(EnvelopeKey::read_load2(cur)?);
        }
        Ok(Self {
            behavior: [b0, b1],
            keys,
        })
    }

    /// Serialize in `CEnvelope::Save` (`Load_2`) binary format.
    pub fn write_load2(&self, out: &mut Vec<u8>) {
        out.push(self.behavior[0] as u8);
        out.push(self.behavior[1] as u8);
        out.extend_from_slice(&(self.keys.len() as u16).to_le_bytes());
        for key in &self.keys {
            key.write_load2(out);
        }
    }

    /// Parse text LightWave `.mot`/ASCII envelope block (`CEnvelope::LoadA`).
    pub fn parse_ascii(text: &str) -> Result<Self, FormatError> {
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        let first = lines.next().ok_or(FormatError::UnexpectedEof)?;
        if !first.contains("{ Envelope") {
            return Err(FormatError::InvalidValue("expected '{ Envelope' header"));
        }
        let count_line = lines.next().ok_or(FormatError::UnexpectedEof)?;
        let nkeys: usize = count_line
            .split_whitespace()
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or(FormatError::InvalidValue("invalid ASCII envelope key count"))?;

        let mut keys = Vec::with_capacity(nkeys);
        for _ in 0..nkeys {
            let line = lines.next().ok_or(FormatError::UnexpectedEof)?;
            let rest = line
                .strip_prefix("Key")
                .ok_or(FormatError::InvalidValue("expected 'Key' line"))?
                .trim();
            let nums: Vec<f32> = rest
                .split_whitespace()
                .filter_map(|p| p.parse::<f32>().ok())
                .collect();
            if nums.len() < 9 {
                return Err(FormatError::InvalidValue("expected 9 floats in ASCII Key"));
            }
            let shape = EnvelopeShape::from_u8(nums[2] as u8)?;
            let mut key = EnvelopeKey {
                value: nums[0],
                time: nums[1],
                shape,
                ..EnvelopeKey::default()
            };
            if shape == EnvelopeShape::Tcb {
                key.tension = nums[3];
                key.continuity = nums[4];
                key.bias = nums[5];
            }
            if shape == EnvelopeShape::Bezier2D {
                key.param = [nums[3], nums[4], nums[5], nums[6]];
            } else {
                key.param[0] = nums[6];
                key.param[1] = nums[7];
            }
            keys.push(key);
        }

        let beh_line = lines.next().ok_or(FormatError::UnexpectedEof)?;
        let beh_rest = beh_line
            .strip_prefix("Behaviors")
            .ok_or(FormatError::InvalidValue("expected 'Behaviors' line"))?
            .trim();
        let mut beh_it = beh_rest.split_whitespace().filter_map(|p| p.parse::<u32>().ok());
        let b0 = EnvelopeBehavior::from_u32(
            beh_it
                .next()
                .ok_or(FormatError::InvalidValue("missing pre-behavior"))?,
        )?;
        let b1 = EnvelopeBehavior::from_u32(
            beh_it
                .next()
                .ok_or(FormatError::InvalidValue("missing post-behavior"))?,
        )?;

        Ok(Self {
            behavior: [b0, b1],
            keys,
        })
    }

    /// Return `(min_time, max_time)` and span duration matching `CEnvelope::GetLength`.
    pub fn time_bounds(&self) -> (f32, f32) {
        match (self.keys.first(), self.keys.last()) {
            (Some(first), Some(last)) => (first.time, last.time),
            _ => (0.0, 0.0),
        }
    }

    #[inline]
    pub fn duration(&self) -> f32 {
        let (mn, mx) = self.time_bounds();
        mx - mn
    }

    /// Evaluate the envelope at `time` (`evalEnvelope` in `src/xrEngine/interp.cpp`).
    pub fn evaluate(&self, mut time: f32) -> f32 {
        if self.keys.is_empty() {
            return 0.0;
        }
        if self.keys.len() == 1 {
            return self.keys[0].value;
        }

        let sz = self.keys.len();
        let skey = &self.keys[0];
        let ekey = &self.keys[sz - 1];
        let skey_n = &self.keys[1];
        let ekey_p = &self.keys[sz - 2];

        let mut offset = 0.0f32;

        if time < skey.time {
            match self.behavior[0] {
                EnvelopeBehavior::Reset => return 0.0,
                EnvelopeBehavior::Constant => return skey.value,
                EnvelopeBehavior::Repeat => {
                    time = wrap_range(time, skey.time, ekey.time).0;
                }
                EnvelopeBehavior::Oscillate => {
                    let (wrapped, noff) = wrap_range(time, skey.time, ekey.time);
                    time = if (noff & 1) != 0 {
                        ekey.time - skey.time - wrapped
                    } else {
                        wrapped
                    };
                }
                EnvelopeBehavior::OffsetRepeat => {
                    let (wrapped, noff) = wrap_range(time, skey.time, ekey.time);
                    time = wrapped;
                    offset = (noff as f32) * (ekey.value - skey.value);
                }
                EnvelopeBehavior::Linear => {
                    let span = skey_n.time - skey.time;
                    if span.abs() <= 1e-12 {
                        return skey.value;
                    }
                    let out_t = outgoing(None, skey, skey_n) / span;
                    return out_t * (time - skey.time) + skey.value;
                }
            }
        } else if time > ekey.time {
            match self.behavior[1] {
                EnvelopeBehavior::Reset => return 0.0,
                EnvelopeBehavior::Constant => return ekey.value,
                EnvelopeBehavior::Repeat => {
                    time = wrap_range(time, skey.time, ekey.time).0;
                }
                EnvelopeBehavior::Oscillate => {
                    let (wrapped, noff) = wrap_range(time, skey.time, ekey.time);
                    time = if (noff & 1) != 0 {
                        ekey.time - skey.time - wrapped
                    } else {
                        wrapped
                    };
                }
                EnvelopeBehavior::OffsetRepeat => {
                    let (wrapped, noff) = wrap_range(time, skey.time, ekey.time);
                    time = wrapped;
                    offset = (noff as f32) * (ekey.value - skey.value);
                }
                EnvelopeBehavior::Linear => {
                    let span = ekey.time - ekey_p.time;
                    if span.abs() <= 1e-12 {
                        return ekey.value;
                    }
                    let in_t = incoming(ekey_p, ekey, None) / span;
                    return in_t * (time - ekey.time) + ekey.value;
                }
            }
        }

        // Locate span [k, k+1], safely bounded for degenerate ranges
        let mut k = 0usize;
        while k + 2 < sz && time > self.keys[k + 1].time {
            k += 1;
        }

        let key0 = &self.keys[k];
        let key1 = &self.keys[k + 1];
        let key0_p = if k > 0 { Some(&self.keys[k - 1]) } else { None };
        let key1_n = self.keys.get(k + 2);

        if time == key0.time {
            return key0.value + offset;
        } else if time == key1.time {
            return key1.value + offset;
        }

        let dt = key1.time - key0.time;
        if dt.abs() <= 1e-12 {
            return key0.value + offset;
        }
        let t = (time - key0.time) / dt;

        match key1.shape {
            EnvelopeShape::Tcb | EnvelopeShape::Bezier | EnvelopeShape::Hermite => {
                let out_t = outgoing(key0_p, key0, key1);
                let in_t = incoming(key0, key1, key1_n);
                let (h1, h2, h3, h4) = hermite(t);
                h1 * key0.value + h2 * key1.value + h3 * out_t + h4 * in_t + offset
            }
            EnvelopeShape::Bezier2D => bez2(key0, key1, time) + offset,
            EnvelopeShape::Linear => key0.value + t * (key1.value - key0.value) + offset,
            EnvelopeShape::Step => key0.value + offset,
        }
    }
}

#[inline]
fn wrap_range(v: f32, lo: f32, hi: f32) -> (f32, i32) {
    let r = hi - lo;
    if r == 0.0 || !r.is_finite() {
        return (lo, 0);
    }
    let v2 = lo + v - r * (v / r).floor();
    let rnd = if v2 > v { 0.5 } else { -0.5 };
    let i = -(((v2 - v) / r + rnd) as i32);
    (v2, i)
}

#[inline]
fn hermite(t: f32) -> (f32, f32, f32, f32) {
    let t2 = t * t;
    let t3 = t * t2;
    let h2 = 3.0 * t2 - t3 - t3;
    let h1 = 1.0 - h2;
    let h4 = t3 - t2;
    let h3 = h4 - t2 + t;
    (h1, h2, h3, h4)
}

#[inline]
fn bezier1d(x0: f32, x1: f32, x2: f32, x3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let c = 3.0 * (x1 - x0);
    let b = 3.0 * (x2 - x1) - c;
    let a = x3 - x0 - c - b;
    a * t3 + b * t2 + c * t + x0
}

fn bez2_time(x0: f32, x1: f32, x2: f32, x3: f32, time: f32) -> f32 {
    let mut t0 = 0.0f32;
    let mut t1 = 1.0f32;
    let mut t = 0.5f32;
    for _ in 0..64 {
        t = t0 + (t1 - t0) * 0.5;
        let v = bezier1d(x0, x1, x2, x3, t);
        if (time - v).abs() <= 0.0001 {
            return t;
        }
        if v > time {
            t1 = t;
        } else {
            t0 = t;
        }
    }
    t
}

fn bez2(key0: &EnvelopeKey, key1: &EnvelopeKey, time: f32) -> f32 {
    let x = if key0.shape == EnvelopeShape::Bezier2D {
        key0.time + key0.param[2]
    } else {
        key0.time + (key1.time - key0.time) / 3.0
    };
    let t = bez2_time(key0.time, x, key1.time + key1.param[0], key1.time, time);
    let y = if key0.shape == EnvelopeShape::Bezier2D {
        key0.value + key0.param[3]
    } else {
        key0.value + key0.param[1] / 3.0
    };
    bezier1d(key0.value, y, key1.param[1] + key1.value, key1.value, t)
}

fn outgoing(key0p: Option<&EnvelopeKey>, key0: &EnvelopeKey, key1: &EnvelopeKey) -> f32 {
    match key0.shape {
        EnvelopeShape::Tcb => {
            let a = (1.0 - key0.tension) * (1.0 + key0.continuity) * (1.0 + key0.bias);
            let b = (1.0 - key0.tension) * (1.0 - key0.continuity) * (1.0 - key0.bias);
            let d = key1.value - key0.value;
            if let Some(p) = key0p {
                let denom = key1.time - p.time;
                let t = if denom.abs() > 1e-12 {
                    (key1.time - key0.time) / denom
                } else {
                    0.0
                };
                t * (a * (key0.value - p.value) + b * d)
            } else {
                b * d
            }
        }
        EnvelopeShape::Linear => {
            let d = key1.value - key0.value;
            if let Some(p) = key0p {
                let denom = key1.time - p.time;
                let t = if denom.abs() > 1e-12 {
                    (key1.time - key0.time) / denom
                } else {
                    0.0
                };
                t * (key0.value - p.value + d)
            } else {
                d
            }
        }
        EnvelopeShape::Bezier | EnvelopeShape::Hermite => {
            let mut out = key0.param[1];
            if let Some(p) = key0p {
                let denom = key1.time - p.time;
                if denom.abs() > 1e-12 {
                    out *= (key1.time - key0.time) / denom;
                }
            }
            out
        }
        EnvelopeShape::Bezier2D => {
            let out = key0.param[3] * (key1.time - key0.time);
            if key0.param[2].abs() > 1e-5 {
                out / key0.param[2]
            } else {
                out * 1e5
            }
        }
        EnvelopeShape::Step => 0.0,
    }
}

fn incoming(key0: &EnvelopeKey, key1: &EnvelopeKey, key1n: Option<&EnvelopeKey>) -> f32 {
    match key1.shape {
        EnvelopeShape::Linear => {
            let d = key1.value - key0.value;
            if let Some(n) = key1n {
                let denom = n.time - key0.time;
                let t = if denom.abs() > 1e-12 {
                    (key1.time - key0.time) / denom
                } else {
                    0.0
                };
                t * (n.value - key1.value + d)
            } else {
                d
            }
        }
        EnvelopeShape::Tcb => {
            let a = (1.0 - key1.tension) * (1.0 - key1.continuity) * (1.0 + key1.bias);
            let b = (1.0 - key1.tension) * (1.0 + key1.continuity) * (1.0 - key1.bias);
            let d = key1.value - key0.value;
            if let Some(n) = key1n {
                let denom = n.time - key0.time;
                let t = if denom.abs() > 1e-12 {
                    (key1.time - key0.time) / denom
                } else {
                    0.0
                };
                t * (b * (n.value - key1.value) + a * d)
            } else {
                a * d
            }
        }
        EnvelopeShape::Bezier | EnvelopeShape::Hermite => {
            let mut inn = key1.param[0];
            if let Some(n) = key1n {
                let denom = n.time - key0.time;
                if denom.abs() > 1e-12 {
                    inn *= (key1.time - key0.time) / denom;
                }
            }
            inn
        }
        EnvelopeShape::Bezier2D => {
            let inn = key1.param[1] * (key1.time - key0.time);
            if key1.param[0].abs() > 1e-5 {
                inn / key1.param[0]
            } else {
                inn * 1e5
            }
        }
        EnvelopeShape::Step => 0.0,
    }
}

/// Canonical 6-channel X-Ray camera / object animation (`.anm`, `COMotion`).
///
/// Channel indices follow `EChannelType`: `[PosX, PosY, PosZ, RotH, RotP, RotB]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectMotion {
    pub name: String,
    pub frame_start: i32,
    pub frame_end: i32,
    pub fps: f32,
    pub version: u16,
    pub envelopes: [Envelope; 6],
}

impl ObjectMotion {
    /// Parse a chunked `.anm` buffer (unwrapping `EOBJ_OMOTION = 0x1100`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let chunk = reader.require_chunk(EOBJ_OMOTION)?;
        Self::parse_raw(&chunk)
    }

    /// Parse unchunked `COMotion` payload (supporting versions `0x0003`, `0x0004`, `0x0005`).
    pub fn parse_raw(payload: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(payload);
        Self::parse_from_cursor(&mut cur)
    }

    /// Parse unchunked `COMotion` sequential stream (`COMotion::Load`) from a shared cursor.
    pub fn parse_from_cursor(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let name = cur.read_zstring()?;
        let frame_start = cur.read_i32()?;
        let frame_end = cur.read_i32()?;
        let fps = cur.read_f32()?;
        let version = cur.read_u16()?;

        let envelopes = match version {
            EOBJ_OMOTION_VERSION_V3 => [
                Envelope::parse_load1(cur)?,
                Envelope::parse_load1(cur)?,
                Envelope::parse_load1(cur)?,
                Envelope::parse_load1(cur)?,
                Envelope::parse_load1(cur)?,
                Envelope::parse_load1(cur)?,
            ],
            EOBJ_OMOTION_VERSION_V4 => {
                // Version 4 stored rotation channels in swapped [P, H, B] order (`motion.cpp:136-147`)
                let px = Envelope::parse_load2(cur)?;
                let py = Envelope::parse_load2(cur)?;
                let pz = Envelope::parse_load2(cur)?;
                let rp = Envelope::parse_load2(cur)?;
                let rh = Envelope::parse_load2(cur)?;
                let rb = Envelope::parse_load2(cur)?;
                [px, py, pz, rh, rp, rb]
            }
            EOBJ_OMOTION_VERSION => [
                Envelope::parse_load2(cur)?,
                Envelope::parse_load2(cur)?,
                Envelope::parse_load2(cur)?,
                Envelope::parse_load2(cur)?,
                Envelope::parse_load2(cur)?,
                Envelope::parse_load2(cur)?,
            ],
            _ => return Err(FormatError::InvalidValue("unsupported .anm COMotion version")),
        };

        Ok(Self {
            name,
            frame_start,
            frame_end,
            fps,
            version,
            envelopes,
        })
    }

    /// Append raw unchunked `COMotion` bytes (without `EOBJ_OMOTION` chunk header).
    pub fn write_raw(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.name.as_bytes());
        out.push(0);
        out.extend_from_slice(&self.frame_start.to_le_bytes());
        out.extend_from_slice(&self.frame_end.to_le_bytes());
        out.extend_from_slice(&self.fps.to_le_bytes());
        out.extend_from_slice(&EOBJ_OMOTION_VERSION.to_le_bytes());
        for env in &self.envelopes {
            env.write_load2(out);
        }
    }

    /// Serialize into standard chunked `.anm` v5 (`EOBJ_OMOTION`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut raw = Vec::new();
        self.write_raw(&mut raw);
        let mut w = ChunkWriter::new();
        w.write_chunk(EOBJ_OMOTION, &raw);
        w.into_bytes()
    }

    /// Frame count `(frame_end - frame_start + 1)`.
    #[inline]
    pub fn frame_count(&self) -> i32 {
        self.frame_end.saturating_sub(self.frame_start).saturating_add(1)
    }

    /// Clip duration in seconds (`frame_count() / fps`).
    #[inline]
    pub fn duration(&self) -> f32 {
        if self.fps > 0.0 {
            self.frame_count().max(0) as f32 / self.fps
        } else {
            0.0
        }
    }

    /// Evaluate `(translation [x, y, z], rotation [RotP, RotH, RotB])` at time `t` seconds (`COMotion::_Evaluate`).
    pub fn evaluate(&self, t: f32) -> ([f32; 3], [f32; 3]) {
        let pos = [
            self.envelopes[0].evaluate(t),
            self.envelopes[1].evaluate(t),
            self.envelopes[2].evaluate(t),
        ];
        // Note `motion.cpp:90-92`: R.x = RotP (idx 4), R.y = RotH (idx 3), R.z = RotB (idx 5)
        let rot = [
            self.envelopes[4].evaluate(t),
            self.envelopes[3].evaluate(t),
            self.envelopes[5].evaluate(t),
        ];
        (pos, rot)
    }
}

/// Multi-clip `.anms` library (`CObjectAnimator::LoadMotions`, `ObjectAnimator.cpp:60-76`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObjectMotionLibrary {
    pub motions: Vec<ObjectMotion>,
}

impl ObjectMotionLibrary {
    /// Parse a binary `.anms` buffer (`u32 count` + sequential unchunked `COMotion` records).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let count = cur.read_u32()? as usize;
        let mut motions = Vec::with_capacity(count.min(4096));
        for _ in 0..count {
            motions.push(ObjectMotion::parse_from_cursor(&mut cur)?);
        }
        motions.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { motions })
    }

    /// Serialize to binary `.anms`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.motions.len() as u32).to_le_bytes());
        for m in &self.motions {
            m.write_raw(&mut out);
        }
        out
    }

    /// Find clip by name (`CObjectAnimator::Play` lookup).
    pub fn find(&self, name: &str) -> Option<&ObjectMotion> {
        self.motions
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(name))
    }
}

/// RGB envelope triplet (`CPostProcessColor`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PostprocessColorCurve {
    pub base: f32,
    pub red: Envelope,
    pub green: Envelope,
    pub blue: Envelope,
}

impl PostprocessColorCurve {
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let base = cur.read_f32()?;
        let red = Envelope::parse_load2(cur)?;
        let green = Envelope::parse_load2(cur)?;
        let blue = Envelope::parse_load2(cur)?;
        Ok(Self {
            base,
            red,
            green,
            blue,
        })
    }

    pub fn write(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.base.to_le_bytes());
        self.red.write_load2(out);
        self.green.write_load2(out);
        self.blue.write_load2(out);
    }

    #[inline]
    pub fn evaluate(&self, t: f32) -> [f32; 3] {
        [
            self.red.evaluate(t),
            self.green.evaluate(t),
            self.blue.evaluate(t),
        ]
    }

    #[inline]
    pub fn duration(&self) -> f32 {
        self.red
            .duration()
            .max(self.green.duration())
            .max(self.blue.duration())
    }
}

/// Evaluated full-screen post-processing snapshot (`SPPInfo` in `CameraManager.h`).
#[derive(Debug, Clone, PartialEq)]
pub struct PostprocessState {
    pub color_base: [f32; 3],
    pub color_add: [f32; 3],
    pub color_gray: [f32; 3],
    pub gray: f32,
    pub blur: f32,
    pub duality_h: f32,
    pub duality_v: f32,
    pub noise_intensity: f32,
    pub noise_granularity: f32,
    pub noise_fps: f32,
    pub cm_influence: f32,
    pub cm_interpolate: f32,
    pub cm_tex1: String,
    pub cm_tex2: String,
}

impl Default for PostprocessState {
    fn default() -> Self {
        Self::identity()
    }
}

impl PostprocessState {
    /// Neutral engine baseline (`pp_identity` in `CameraManager.cpp:154-164`).
    pub fn identity() -> Self {
        Self {
            color_base: [0.5, 0.5, 0.5],
            color_add: [0.0, 0.0, 0.0],
            color_gray: [0.333, 0.333, 0.333],
            gray: 0.0,
            blur: 0.0,
            duality_h: 0.0,
            duality_v: 0.0,
            noise_intensity: 0.0,
            noise_granularity: 1.0,
            noise_fps: 30.0,
            cm_influence: 0.0,
            cm_interpolate: 1.0,
            cm_tex1: String::new(),
            cm_tex2: String::new(),
        }
    }

    /// Zero accumulator state (`pp_zero` in `CameraManager.cpp:165-171`).
    pub fn zero() -> Self {
        Self {
            color_base: [0.0; 3],
            color_add: [0.0; 3],
            color_gray: [0.0; 3],
            gray: 0.0,
            blur: 0.0,
            duality_h: 0.0,
            duality_v: 0.0,
            noise_intensity: 0.0,
            noise_granularity: 0.0,
            noise_fps: 0.0,
            cm_influence: 0.0,
            cm_interpolate: 0.0,
            cm_tex1: String::new(),
            cm_tex2: String::new(),
        }
    }

    /// Pure blend from `def` toward `to` by clamped `factor` (`[0.0..=1.0]`).
    pub fn lerp(def: &Self, to: &Self, factor: f32) -> Self {
        let f = factor.clamp(0.0, 1.0);
        let lerp3 = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
            [
                a[0] + (b[0] - a[0]) * f,
                a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f,
            ]
        };
        Self {
            color_base: lerp3(def.color_base, to.color_base),
            color_add: lerp3(def.color_add, to.color_add),
            color_gray: lerp3(def.color_gray, to.color_gray),
            gray: def.gray + (to.gray - def.gray) * f,
            blur: def.blur + (to.blur - def.blur) * f,
            duality_h: def.duality_h + (to.duality_h - def.duality_h) * f,
            duality_v: def.duality_v + (to.duality_v - def.duality_v) * f,
            noise_intensity: to.noise_intensity,
            noise_granularity: to.noise_granularity,
            noise_fps: to.noise_fps,
            cm_influence: def.cm_influence + (to.cm_influence - def.cm_influence) * f,
            cm_interpolate: def.cm_interpolate + (to.cm_interpolate - def.cm_interpolate) * f,
            cm_tex1: to.cm_tex1.clone(),
            cm_tex2: to.cm_tex2.clone(),
        }
    }

    /// In-place `SPPInfo::lerp(def, to, factor)` effector composition (`CameraManager.cpp:106`).
    pub fn accumulate_lerp(&mut self, def: &Self, to: &Self, factor: f32) {
        let blended = Self::lerp(def, to, factor);
        self.duality_h += blended.duality_h;
        self.duality_v += blended.duality_v;
        self.gray += blended.gray;
        self.blur += blended.blur;
        self.noise_intensity = blended.noise_intensity;
        self.noise_granularity = blended.noise_granularity;
        self.noise_fps = blended.noise_fps;
        self.color_base = blended.color_base;
        self.color_gray = blended.color_gray;
        self.color_add = blended.color_add;
        self.cm_tex1 = blended.cm_tex1;
        self.cm_tex2 = blended.cm_tex2;
        self.cm_influence += blended.cm_influence;
        self.cm_interpolate += blended.cm_interpolate;
    }

    /// Accumulate another postprocess snapshot matching `SPPInfo::add` (`CameraManager.cpp:25-55`).
    pub fn add_pp(&mut self, ppi: &Self) {
        self.blur += ppi.blur;
        self.gray += ppi.gray;
        self.duality_h += ppi.duality_h;
        self.duality_v += ppi.duality_v;
        self.noise_intensity = self.noise_intensity.max(ppi.noise_intensity);
        self.noise_granularity = self.noise_granularity.max(ppi.noise_granularity);
        self.noise_fps = self.noise_fps.max(ppi.noise_fps);
        for axis in 0..3 {
            self.color_base[axis] += ppi.color_base[axis];
            self.color_gray[axis] += ppi.color_gray[axis];
            self.color_add[axis] += ppi.color_add[axis];
        }
        if !ppi.cm_tex1.is_empty() {
            if !self.cm_tex1.is_empty() {
                self.cm_tex2 = ppi.cm_tex1.clone();
                let denom = self.cm_influence + ppi.cm_influence;
                self.cm_interpolate = if denom.abs() > 1e-8 {
                    1.0 - self.cm_influence / denom
                } else {
                    0.0
                };
            } else {
                self.cm_tex1 = ppi.cm_tex1.clone();
                self.cm_influence = ppi.cm_influence;
                self.cm_interpolate = 0.0;
            }
            self.cm_influence = self.cm_influence.max(ppi.cm_influence);
        }
    }

    /// Subtract additive postprocess channels matching `SPPInfo::sub` (`CameraManager.cpp:57-67`).
    pub fn sub_pp(&mut self, ppi: &Self) {
        self.blur -= ppi.blur;
        self.gray -= ppi.gray;
        self.duality_h -= ppi.duality_h;
        self.duality_v -= ppi.duality_v;
        for axis in 0..3 {
            self.color_base[axis] -= ppi.color_base[axis];
            self.color_gray[axis] -= ppi.color_gray[axis];
            self.color_add[axis] -= ppi.color_add[axis];
        }
    }
}

/// Keyframed postprocess effect asset (`.ppe`, `CPostprocessAnimator`).
#[derive(Debug, Clone, PartialEq)]
pub struct PostprocessEffect {
    pub version: u32,
    pub base_color: PostprocessColorCurve,
    pub add_color: PostprocessColorCurve,
    pub gray_color: PostprocessColorCurve,
    pub gray: Envelope,
    pub blur: Envelope,
    pub duality_h: Envelope,
    pub duality_v: Envelope,
    pub noise_intensity: Envelope,
    pub noise_granularity: Envelope,
    pub noise_fps: Envelope,
    pub cm_influence: Envelope,
    pub cm_tex1: String,
}

impl PostprocessEffect {
    /// Parse a binary `.ppe` postprocess animation (`CPostprocessAnimator::Load`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let version = cur.read_u32()?;
        let base_color = PostprocessColorCurve::parse(&mut cur)?;
        let add_color = PostprocessColorCurve::parse(&mut cur)?;
        let gray_color = PostprocessColorCurve::parse(&mut cur)?;
        let gray = Envelope::parse_load2(&mut cur)?;
        let blur = Envelope::parse_load2(&mut cur)?;
        let duality_h = Envelope::parse_load2(&mut cur)?;
        let duality_v = Envelope::parse_load2(&mut cur)?;
        let noise_intensity = Envelope::parse_load2(&mut cur)?;
        let noise_granularity = Envelope::parse_load2(&mut cur)?;
        let noise_fps = Envelope::parse_load2(&mut cur)?;

        let (cm_influence, cm_tex1) = if version >= POSTPROCESS_FILE_VERSION {
            let cm_inf = Envelope::parse_load2(&mut cur)?;
            let tex1 = if !cur.is_eof() {
                cur.read_zstring()?
            } else {
                String::new()
            };
            (cm_inf, tex1)
        } else {
            (Envelope::default(), String::new())
        };

        Ok(Self {
            version,
            base_color,
            add_color,
            gray_color,
            gray,
            blur,
            duality_h,
            duality_v,
            noise_intensity,
            noise_granularity,
            noise_fps,
            cm_influence,
            cm_tex1,
        })
    }

    /// Serialize in binary v2 `.ppe` layout.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&POSTPROCESS_FILE_VERSION.to_le_bytes());
        self.base_color.write(&mut out);
        self.add_color.write(&mut out);
        self.gray_color.write(&mut out);
        self.gray.write_load2(&mut out);
        self.blur.write_load2(&mut out);
        self.duality_h.write_load2(&mut out);
        self.duality_v.write_load2(&mut out);
        self.noise_intensity.write_load2(&mut out);
        self.noise_granularity.write_load2(&mut out);
        self.noise_fps.write_load2(&mut out);
        self.cm_influence.write_load2(&mut out);
        out.extend_from_slice(self.cm_tex1.as_bytes());
        out.push(0);
        out
    }

    /// Maximum keyframe duration across all curves (`CPostprocessAnimator::GetLength`).
    pub fn length(&self) -> f32 {
        [
            self.base_color.duration(),
            self.add_color.duration(),
            self.gray_color.duration(),
            self.gray.duration(),
            self.blur.duration(),
            self.duality_h.duration(),
            self.duality_v.duration(),
            self.noise_intensity.duration(),
            self.noise_granularity.duration(),
            self.noise_fps.duration(),
            self.cm_influence.duration(),
        ]
        .into_iter()
        .fold(0.0f32, f32::max)
    }

    /// Evaluate `.ppe` curves with runtime engine `pp_identity` offsets & noise scaling (`PostprocessAnimator.cpp:216-237`).
    pub fn evaluate(&self, t: f32) -> PostprocessState {
        let id = PostprocessState::identity();
        let cb = self.base_color.evaluate(t);
        let ca = self.add_color.evaluate(t);
        let cg = self.gray_color.evaluate(t);

        let noise_intensity = if self.noise_intensity.keys.is_empty() {
            id.noise_intensity
        } else {
            self.noise_intensity.evaluate(t)
        };
        let noise_granularity = if self.noise_granularity.keys.is_empty() {
            id.noise_granularity
        } else {
            self.noise_granularity.evaluate(t)
        };
        let noise_fps = if self.noise_fps.keys.is_empty() {
            id.noise_fps
        } else {
            self.noise_fps.evaluate(t) * 100.0
        };

        PostprocessState {
            color_base: [
                cb[0] + id.color_base[0],
                cb[1] + id.color_base[1],
                cb[2] + id.color_base[2],
            ],
            color_add: [
                ca[0] + id.color_add[0],
                ca[1] + id.color_add[1],
                ca[2] + id.color_add[2],
            ],
            color_gray: [
                cg[0] + id.color_gray[0],
                cg[1] + id.color_gray[1],
                cg[2] + id.color_gray[2],
            ],
            gray: self.gray.evaluate(t),
            blur: self.blur.evaluate(t),
            duality_h: self.duality_h.evaluate(t),
            duality_v: self.duality_v.evaluate(t),
            noise_intensity,
            noise_granularity,
            noise_fps,
            cm_influence: self.cm_influence.evaluate(t),
            cm_interpolate: 1.0,
            cm_tex1: self.cm_tex1.clone(),
            cm_tex2: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_key(time: f32, value: f32) -> EnvelopeKey {
        EnvelopeKey {
            value,
            time,
            shape: EnvelopeShape::Linear,
            ..EnvelopeKey::default()
        }
    }

    #[test]
    fn test_q16_quantization_roundtrip() {
        for sample in [-32.0f32, -12.345, 0.0, 1.0, 17.5, 32.0] {
            let q = encode_q16(sample, Q16_MIN, Q16_MAX);
            let back = decode_q16(q, Q16_MIN, Q16_MAX);
            assert!((back - sample).abs() < 0.002, "sample={sample}, back={back}");
        }
    }

    #[test]
    fn test_envelope_shapes_and_behaviors() {
        // Linear + Oscillate including negative pre-behavior odd cycles
        let env = Envelope {
            behavior: [EnvelopeBehavior::Oscillate, EnvelopeBehavior::OffsetRepeat],
            keys: vec![linear_key(0.0, 0.0), linear_key(2.0, 10.0)],
        };
        assert!((env.evaluate(1.0) - 5.0).abs() < 1e-5);
        // t = -1.0 -> noff = -1 (odd) => flipped 2 - 0 - 1 = 1.0 => 5.0
        assert!((env.evaluate(-1.0) - 5.0).abs() < 1e-5);
        // t = 5.0 -> noff = 2 => wrapped = 1.0 + 2 * 10.0 = 25.0
        assert!((env.evaluate(5.0) - 25.0).abs() < 1e-5);

        // Step & Hermite & Bezier2D
        let step_env = Envelope {
            behavior: [EnvelopeBehavior::Constant, EnvelopeBehavior::Constant],
            keys: vec![
                EnvelopeKey {
                    value: 3.0,
                    time: 0.0,
                    shape: EnvelopeShape::Step,
                    ..EnvelopeKey::default()
                },
                EnvelopeKey {
                    value: 9.0,
                    time: 4.0,
                    shape: EnvelopeShape::Step,
                    ..EnvelopeKey::default()
                },
            ],
        };
        assert_eq!(step_env.evaluate(2.5), 3.0);
        assert_eq!(step_env.evaluate(4.0), 9.0);

        let bez2_env = Envelope {
            behavior: [EnvelopeBehavior::Constant, EnvelopeBehavior::Constant],
            keys: vec![
                EnvelopeKey {
                    value: 0.0,
                    time: 0.0,
                    shape: EnvelopeShape::Bezier2D,
                    param: [0.0, 0.0, 1.0, 0.0],
                    ..EnvelopeKey::default()
                },
                EnvelopeKey {
                    value: 10.0,
                    time: 2.0,
                    shape: EnvelopeShape::Bezier2D,
                    param: [-1.0, 0.0, 0.0, 0.0],
                    ..EnvelopeKey::default()
                },
            ],
        };
        assert!((bez2_env.evaluate(1.0) - 5.0).abs() < 1e-3);

        // Binary Load_2 roundtrip
        let mut raw = Vec::new();
        step_env.write_load2(&mut raw);
        let parsed = Envelope::parse_load2(&mut ByteCursor::new(&raw)).unwrap();
        assert_eq!(parsed.keys.len(), 2);
        assert_eq!(parsed.evaluate(4.0), 9.0);

        // ASCII LoadA test
        let ascii = "{ Envelope\n2\nKey 1.0 0.0 0 0.5 0 0 0 0 0\nKey 5.0 2.0 3 0 0 0 0 0 0\nBehaviors 1 5\n";
        let a = Envelope::parse_ascii(ascii).unwrap();
        assert_eq!(a.behavior, [EnvelopeBehavior::Constant, EnvelopeBehavior::Linear]);
        assert!((a.evaluate(4.0) - 9.0).abs() < 1e-4);
    }

    #[test]
    fn test_anm_object_motion_v4_swapped_rotation_and_v5_roundtrip() {
        // Craft v4 raw payload where rotation order is [RotP, RotH, RotB]
        let mut raw_v4 = Vec::new();
        raw_v4.extend_from_slice(b"cam_intro\0");
        raw_v4.extend_from_slice(&0i32.to_le_bytes());
        raw_v4.extend_from_slice(&60i32.to_le_bytes());
        raw_v4.extend_from_slice(&30.0f32.to_le_bytes());
        raw_v4.extend_from_slice(&EOBJ_OMOTION_VERSION_V4.to_le_bytes());

        let vals = [1.0f32, 2.0, 3.0, 0.25 /*RotP*/, 1.5 /*RotH*/, -0.5 /*RotB*/];
        for v in vals {
            Envelope {
                behavior: [EnvelopeBehavior::Constant, EnvelopeBehavior::Constant],
                keys: vec![linear_key(0.0, v)],
            }
            .write_load2(&mut raw_v4);
        }

        let mot_v4 = ObjectMotion::parse_raw(&raw_v4).unwrap();
        let (pos, rot) = mot_v4.evaluate(0.5);
        assert_eq!(pos, [1.0, 2.0, 3.0]);
        // Evaluate returns [RotP, RotH, RotB]
        assert_eq!(rot, [0.25, 1.5, -0.5]);
        // Canonical envelope indices store RotH at [3], RotP at [4]
        assert_eq!(mot_v4.envelopes[3].evaluate(0.5), 1.5);
        assert_eq!(mot_v4.envelopes[4].evaluate(0.5), 0.25);

        // Save as v5 chunked .anm and re-parse
        let v5_bytes = mot_v4.to_bytes();
        let mot_v5 = ObjectMotion::parse(&v5_bytes).unwrap();
        assert_eq!(mot_v5.version, EOBJ_OMOTION_VERSION);
        assert_eq!(mot_v5.evaluate(0.5), ([1.0, 2.0, 3.0], [0.25, 1.5, -0.5]));
        assert!((mot_v5.duration() - (61.0 / 30.0)).abs() < 1e-5);
    }

    #[test]
    fn test_ppe_postprocess_v2_eval_and_lerp() {
        let ppe = PostprocessEffect {
            version: POSTPROCESS_FILE_VERSION,
            base_color: PostprocessColorCurve {
                base: 0.0,
                red: Envelope {
                    keys: vec![linear_key(0.0, 0.2)],
                    ..Envelope::default()
                },
                green: Envelope::default(),
                blue: Envelope::default(),
            },
            add_color: PostprocessColorCurve::default(),
            gray_color: PostprocessColorCurve::default(),
            gray: Envelope {
                keys: vec![linear_key(0.0, 0.0), linear_key(2.0, 0.8)],
                ..Envelope::default()
            },
            blur: Envelope::default(),
            duality_h: Envelope {
                keys: vec![linear_key(0.0, 0.01)],
                ..Envelope::default()
            },
            duality_v: Envelope::default(),
            noise_intensity: Envelope::default(), // empty -> pp_identity (0.0)
            noise_granularity: Envelope::default(), // empty -> pp_identity (1.0)
            noise_fps: Envelope {
                keys: vec![linear_key(0.0, 0.45)], // scaled * 100 -> 45.0
                ..Envelope::default()
            },
            cm_influence: Envelope {
                keys: vec![linear_key(0.0, 0.6)],
                ..Envelope::default()
            },
            cm_tex1: "fx\\color_grade_rad".to_string(),
        };

        let raw = ppe.to_bytes();
        let parsed = PostprocessEffect::parse(&raw).unwrap();
        assert!((parsed.length() - 2.0).abs() < 1e-5);

        let st = parsed.evaluate(1.0);
        assert!((st.color_base[0] - 0.7).abs() < 1e-5);
        assert!((st.gray - 0.4).abs() < 1e-5);
        assert!((st.duality_h - 0.01).abs() < 1e-5);
        assert_eq!(st.noise_granularity, 1.0);
        assert!((st.noise_fps - 45.0).abs() < 1e-4);
        assert!((st.cm_influence - 0.6).abs() < 1e-5);
        assert_eq!(st.cm_tex1, "fx\\color_grade_rad");

        let blended = PostprocessState::lerp(&PostprocessState::identity(), &st, 0.5);
        assert!((blended.gray - 0.2).abs() < 1e-5);
        assert!((blended.color_base[0] - 0.6).abs() < 1e-5);
    }
}
