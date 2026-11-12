//! Pure Rust X-Ray Shaders, Lightmap Compilers, R4/Anomaly Blender Passes, Environment Modifiers, Particles & Fog Volumes
//!
//! Covers:
//! - `shaders_xrlc.xr`: 144-byte `Shader_xrLC` collision/rendering/lightmap compiler entries
//! - `shaders.xr`: `WaveForm`, `CConstant` (chunk 0), `CMatrix` (chunk 1), `IBlender` descriptors + properties (chunk 2)
//! - R4 / DX11 / Anomaly blender compiler & `uber_deffer` / `uber_shadow` / `dxUIRender` pure-Rust pass planners
//! - `level.env_mod` & Anomaly `environment\env_mods\<level>.ltx`: `CEnvModifier` zones + spatial evaluation
//! - `level.ps_static`: Static world particle emitters (`CPGDef` / `CParticleGroup`)
//! - `level.fog_vol`: Volumetric fog & obstacle matrices (`dx103DFluidVolume` / `v2` & `v3`)

use std::f32::consts::PI;

use xray_core::ltx_loader::OwnedLtxDocument;
use xray_core::LtxDocument;

use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::ogf::{matrix_invert_43, matrix_mul_43, XRayMatrix, XRAY_IDENTITY_MATRIX};
use crate::thm::TextureDescrTable;

/// `CMatrix` evaluation mode (`src/Layers/xrRender/SH_Matrix.h:13`).
pub mod matrix_mode {
    pub const PROGRAMMABLE: u32 = 0;
    pub const TCM: u32 = 1;
    pub const S_REFL: u32 = 2;
    pub const C_REFL: u32 = 3;
    pub const DETAIL: u32 = 4;
}

/// `CMatrix` texture-coordinate modifier mask bits (`src/Layers/xrRender/SH_Matrix.h:15-21`).
pub mod tcm_flags {
    pub const SCALE: u32 = 1 << 0;
    pub const ROTATE: u32 = 1 << 1;
    pub const SCROLL: u32 = 1 << 2;
}

pub const SHADER_XRLC_ENTRY_SIZE: usize = 144;
pub const CBLENDER_DESC_SIZE: usize = 176;

/// `Shader_xrLC` compiler surface flag bits (`src/xrEngine/Shader_xrLC.h`).
pub mod shader_xrlc_flags {
    pub const COLLISION: u32 = 1 << 0;
    pub const RENDERING: u32 = 1 << 1;
    pub const OPTIMIZE_UV: u32 = 1 << 2;
    pub const LIGHT_VERTEX: u32 = 1 << 3;
    pub const LIGHT_CAST_SHADOW: u32 = 1 << 4;
    pub const LIGHT_SHARP: u32 = 1 << 5;
}

/// 144-byte `Shader_xrLC` record from `shaders_xrlc.xr`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderXrlcEntry {
    pub name: String,
    pub flags: u32,
    pub vert_translucency: f32,
    pub vert_ambient: f32,
    pub lm_density: f32,
}

/// Parsed `shaders_xrlc.xr` library.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShaderXrlcLibrary {
    pub entries: Vec<ShaderXrlcEntry>,
}

fn read_fixed_cstr(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

impl ShaderXrlcLibrary {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        if !data.len().is_multiple_of(SHADER_XRLC_ENTRY_SIZE) {
            return Err(FormatError::InvalidValue(
                "shaders_xrlc.xr size is not a multiple of 144 bytes",
            ));
        }
        let count = data.len() / SHADER_XRLC_ENTRY_SIZE;
        let mut entries = Vec::with_capacity(count);
        for rec in data.as_chunks::<144>().0 {
            let name = read_fixed_cstr(&rec[..128]);
            let flags = u32::from_le_bytes([rec[128], rec[129], rec[130], rec[131]]);
            let vert_translucency = f32::from_le_bytes([rec[132], rec[133], rec[134], rec[135]]);
            let vert_ambient = f32::from_le_bytes([rec[136], rec[137], rec[138], rec[139]]);
            let lm_density = f32::from_le_bytes([rec[140], rec[141], rec[142], rec[143]]);
            entries.push(ShaderXrlcEntry {
                name,
                flags,
                vert_translucency,
                vert_ambient,
                lm_density,
            });
        }
        Ok(Self { entries })
    }

    pub fn find(&self, name: &str) -> Option<&ShaderXrlcEntry> {
        self.entries
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(name))
    }
}

/// Periodic waveform kind matching `WaveForm::EFunction` (`src/xrEngine/WaveForm.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum WaveFunction {
    Constant = 0,
    Sin = 1,
    Triangle = 2,
    Square = 3,
    Sawtooth = 4,
    InvSawtooth = 5,
}

impl WaveFunction {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => Self::Constant,
            1 => Self::Sin,
            2 => Self::Triangle,
            3 => Self::Square,
            4 => Self::Sawtooth,
            5 => Self::InvSawtooth,
            _ => Self::Constant,
        }
    }
}

/// 20-byte `WaveForm` (`arg0 + arg1 * func((t + arg2) * arg3)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveForm {
    pub func: WaveFunction,
    pub arg: [f32; 4],
}

impl WaveForm {
    pub fn parse(cur: &mut ByteCursor<'_>) -> Result<Self, FormatError> {
        let func = WaveFunction::from_raw(cur.read_u32()?);
        let arg = [
            cur.read_f32()?,
            cur.read_f32()?,
            cur.read_f32()?,
            cur.read_f32()?,
        ];
        Ok(Self { func, arg })
    }

    pub fn evaluate(&self, t: f32) -> f32 {
        let x = (t + self.arg[2]) * self.arg[3];
        let phase = x - x.floor();
        let val = match self.func {
            WaveFunction::Constant => 0.0,
            WaveFunction::Sin => (phase * PI * 2.0).sin(),
            WaveFunction::Triangle => ((phase - 0.25) * PI * 2.0).sin().asin() / (PI * 0.5),
            WaveFunction::Square => {
                let c = (phase * PI).cos();
                if c < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            }
            WaveFunction::Sawtooth => ((phase + 0.5) * PI).tan().atan() / (PI * 0.5),
            WaveFunction::InvSawtooth => -(((phase + 0.5) * PI).tan().atan() / (PI * 0.5)),
        };
        self.arg[0] + self.arg[1] * val
    }
}

/// `CConstant` entry in `shaders.xr` chunk 0 (4 animated RGBA channels).
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderConstant {
    pub name: String,
    pub channels: [WaveForm; 4],
}

impl ShaderConstant {
    pub fn evaluate(&self, t: f32) -> [f32; 4] {
        [
            self.channels[0].evaluate(t),
            self.channels[1].evaluate(t),
            self.channels[2].evaluate(t),
            self.channels[3].evaluate(t),
        ]
    }
}

/// `CMatrix` UV-transform entry in `shaders.xr` chunk 1.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderMatrix {
    pub name: String,
    pub mode: u32,
    pub tcm: u32,
    /// `[scale_u, scale_v, rotate, scroll_u, scroll_v]` waveforms.
    pub transforms: [WaveForm; 5],
}

#[inline]
fn matrix_tc_trans(u: f32, v: f32) -> XRayMatrix {
    let mut m = XRAY_IDENTITY_MATRIX;
    m[2][0] = u;
    m[2][1] = v;
    m
}

#[inline]
fn matrix_rotate_z(angle: f32) -> XRayMatrix {
    let (sina, cosa) = angle.sin_cos();
    [
        [cosa, sina, 0.0, 0.0],
        [-sina, cosa, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Transform 2D texture coordinates `(u, v)` using X-Ray's homogeneous 3-coord `(u, v, 1)` layout (`CMatrix::tc_trans` puts translation in row 2).
#[inline]
pub fn transform_uv(m: &XRayMatrix, uv: [f32; 2]) -> [f32; 2] {
    [
        uv[0] * m[0][0] + uv[1] * m[1][0] + m[2][0],
        uv[0] * m[0][1] + uv[1] * m[1][1] + m[2][1],
    ]
}

impl ShaderMatrix {
    /// Evaluate `CMatrix::Calculate()` (`src/Layers/xrRender/SH_Matrix.cpp:4-80`), keeping `prev` untouched for `PROGRAMMABLE`/`DETAIL` modes.
    pub fn evaluate_with_prev(
        &self,
        prev: XRayMatrix,
        t_global: f32,
        view_matrix: &XRayMatrix,
    ) -> XRayMatrix {
        match self.mode {
            matrix_mode::PROGRAMMABLE | matrix_mode::DETAIL => prev,
            matrix_mode::TCM => {
                // Exact call sequence from SH_Matrix.cpp:15-44 (`tc_trans(.5,.5)` -> `mulA_43` -> `mulB_43(-0.5,-0.5)`)
                let mut s_u = 1.0f32;
                let mut s_v = 1.0f32;
                let mut xform = matrix_tc_trans(0.5, 0.5);
                if (self.tcm & tcm_flags::ROTATE) != 0 {
                    let rot = matrix_rotate_z(self.transforms[2].evaluate(t_global) * t_global);
                    xform = matrix_mul_43(&rot, &xform);
                }
                if (self.tcm & tcm_flags::SCALE) != 0 {
                    s_u = self.transforms[0].evaluate(t_global);
                    s_v = self.transforms[1].evaluate(t_global);
                    let scale = [
                        [s_u, 0.0, 0.0, 0.0],
                        [0.0, s_v, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ];
                    xform = matrix_mul_43(&scale, &xform);
                }
                if (self.tcm & tcm_flags::SCROLL) != 0 {
                    let u = self.transforms[3].evaluate(t_global) * t_global * s_u;
                    let v = self.transforms[4].evaluate(t_global) * t_global * s_v;
                    let scroll = matrix_tc_trans(u, v);
                    xform = matrix_mul_43(&scroll, &xform);
                }
                let post = matrix_tc_trans(-0.5, -0.5);
                matrix_mul_43(&xform, &post)
            }
            matrix_mode::S_REFL => {
                let ux = 0.5 * view_matrix[0][0];
                let uy = 0.5 * view_matrix[1][0];
                let uz = 0.5 * view_matrix[2][0];
                let vx = -0.5 * view_matrix[0][1];
                let vy = -0.5 * view_matrix[1][1];
                let vz = -0.5 * view_matrix[2][1];
                [
                    [ux, vx, 0.0, 0.0],
                    [uy, vy, 0.0, 0.0],
                    [uz, vz, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                ]
            }
            matrix_mode::C_REFL => {
                let mut m = *view_matrix;
                m[3][0] = 0.0;
                m[3][1] = 0.0;
                m[3][2] = 0.0;
                matrix_invert_43(&m).unwrap_or(XRAY_IDENTITY_MATRIX)
            }
            _ => prev,
        }
    }

    /// Stateless evaluation of `CMatrix::Calculate()` starting from identity.
    pub fn evaluate(&self, t_global: f32, view_matrix: &XRayMatrix) -> XRayMatrix {
        self.evaluate_with_prev(XRAY_IDENTITY_MATRIX, t_global, view_matrix)
    }
}

/// Decode X-Ray `MK_CLSID(a..h)` 64-bit value into its readable 8-char ASCII code.
pub fn clsid_to_string(clsid: u64) -> String {
    let bytes = clsid.to_be_bytes();
    let trimmed = bytes
        .iter()
        .copied()
        .map(|b| if b.is_ascii_graphic() { b } else { b' ' })
        .collect::<Vec<u8>>();
    String::from_utf8_lossy(&trimmed).trim().to_string()
}

/// 176-byte `CBlender_DESC` header at the start of each blender chunk (`shaders.xr` chunk 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlenderDesc {
    pub clsid: u64,
    pub clsid_str: String,
    pub name: String,
    pub computer: String,
    pub time: u32,
    pub version: u16,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlenderPropertyValue {
    Marker,
    Matrix(String),
    Constant(String),
    Texture(String),
    Integer { value: i32, min: i32, max: i32 },
    Float { value: f32, min: f32, max: f32 },
    Bool(bool),
    Token { selected: u32, items: Vec<(u32, String)> },
    Clsid { selected: u64, choices: Vec<u64> },
    Object(String),
    String(String),
    MarkerTemplate { typ: u32, limit: u32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlenderProperty {
    pub name: String,
    pub value: BlenderPropertyValue,
}

/// Parsed blender entry inside `shaders.xr` chunk 2.
#[derive(Debug, Clone, PartialEq)]
pub struct BlenderShader {
    pub desc: BlenderDesc,
    pub priority: i32,
    pub strict_sorting: bool,
    pub base_texture: String,
    pub base_transform: String,
    pub properties: Vec<BlenderProperty>,
    /// Unparsed trailing bytes if a legacy/custom property opcode is encountered.
    pub raw_tail: Vec<u8>,
}

/// Parsed `shaders.xr` (`CResourceManager::_Load`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EngineShaderLibrary {
    pub constants: Vec<ShaderConstant>,
    pub matrices: Vec<ShaderMatrix>,
    pub blenders: Vec<BlenderShader>,
}

impl EngineShaderLibrary {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        if data.len() >= 8 && &data[..8] == b"shENGINE" {
            return Err(FormatError::InvalidValue(
                "legacy shENGINE shaders.xr format is unsupported",
            ));
        }

        let reader = ChunkReader::new(data);
        let mut constants = Vec::new();
        if let Some(ch) = reader.open_chunk(0)? {
            let mut cur = ByteCursor::new(&ch);
            while !cur.is_eof() {
                let name = cur.read_zstring()?;
                let channels = [
                    WaveForm::parse(&mut cur)?,
                    WaveForm::parse(&mut cur)?,
                    WaveForm::parse(&mut cur)?,
                    WaveForm::parse(&mut cur)?,
                ];
                constants.push(ShaderConstant { name, channels });
            }
        }

        let mut matrices = Vec::new();
        if let Some(ch) = reader.open_chunk(1)? {
            let mut cur = ByteCursor::new(&ch);
            while !cur.is_eof() {
                let name = cur.read_zstring()?;
                let mode = cur.read_u32()?;
                let tcm = cur.read_u32()?;
                let transforms = [
                    WaveForm::parse(&mut cur)?,
                    WaveForm::parse(&mut cur)?,
                    WaveForm::parse(&mut cur)?,
                    WaveForm::parse(&mut cur)?,
                    WaveForm::parse(&mut cur)?,
                ];
                matrices.push(ShaderMatrix {
                    name,
                    mode,
                    tcm,
                    transforms,
                });
            }
        }

        let mut blenders = Vec::new();
        if let Some(ch) = reader.open_chunk(2)? {
            let sub = ChunkReader::new(&ch);
            for entry in sub.iter() {
                let entry = entry?;
                let payload = entry.decode()?;
                blenders.push(parse_blender_chunk(&payload)?);
            }
        }

        Ok(Self {
            constants,
            matrices,
            blenders,
        })
    }

    pub fn find_blender(&self, name: &str) -> Option<&BlenderShader> {
        self.blenders
            .iter()
            .find(|b| b.desc.name.eq_ignore_ascii_case(name))
    }

    pub fn find_constant(&self, name: &str) -> Option<&ShaderConstant> {
        self.constants
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
    }

    pub fn find_matrix(&self, name: &str) -> Option<&ShaderMatrix> {
        self.matrices
            .iter()
            .find(|m| m.name.eq_ignore_ascii_case(name))
    }
}

fn parse_blender_chunk(bytes: &[u8]) -> Result<BlenderShader, FormatError> {
    if bytes.len() < CBLENDER_DESC_SIZE {
        return Err(FormatError::UnexpectedEof);
    }
    let mut cur = ByteCursor::new(bytes);
    let clsid = cur.read_u64()?;
    let raw_name = cur.read_bytes(128)?;
    let raw_computer = cur.read_bytes(32)?;
    let time = cur.read_u32()?;
    let version = cur.read_u16()?;
    let _pad = cur.read_u16()?;

    let desc = BlenderDesc {
        clsid,
        clsid_str: clsid_to_string(clsid),
        name: read_fixed_cstr(raw_name),
        computer: read_fixed_cstr(raw_computer),
        time,
        version,
    };

    let mut priority = 1i32;
    let mut strict_sorting = false;
    let mut base_texture = "$base0".to_string();
    let mut base_transform = "$null".to_string();
    let mut properties = Vec::new();
    let mut raw_tail = Vec::new();

    while !cur.is_eof() {
        let save_pos = cur.position();
        let pid = cur.read_u32()?;
        let pname = cur.read_zstring()?;
        let value = match pid {
            0 => BlenderPropertyValue::Marker,
            1 => {
                let s = read_fixed_cstr(cur.read_bytes(64)?);
                if pname.eq_ignore_ascii_case("Transform") && base_transform == "$null" {
                    base_transform = s.clone();
                }
                BlenderPropertyValue::Matrix(s)
            }
            2 => BlenderPropertyValue::Constant(read_fixed_cstr(cur.read_bytes(64)?)),
            3 => {
                let s = read_fixed_cstr(cur.read_bytes(64)?);
                if pname.eq_ignore_ascii_case("Name") && base_texture == "$base0" {
                    base_texture = s.clone();
                }
                BlenderPropertyValue::Texture(s)
            }
            4 => {
                let value = cur.read_i32()?;
                let min = cur.read_i32()?;
                let max = cur.read_i32()?;
                if pname.eq_ignore_ascii_case("Priority") {
                    priority = value;
                }
                BlenderPropertyValue::Integer { value, min, max }
            }
            5 => BlenderPropertyValue::Float {
                value: cur.read_f32()?,
                min: cur.read_f32()?,
                max: cur.read_f32()?,
            },
            6 => {
                let v = cur.read_u32()? != 0;
                if pname.eq_ignore_ascii_case("Strict sorting") {
                    strict_sorting = v;
                }
                BlenderPropertyValue::Bool(v)
            }
            7 => {
                let selected = cur.read_u32()?;
                let count = cur.read_u32()? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    let id = cur.read_u32()?;
                    let text = read_fixed_cstr(cur.read_bytes(64)?);
                    items.push((id, text));
                }
                BlenderPropertyValue::Token { selected, items }
            }
            8 => {
                let selected = cur.read_u64()?;
                let count = cur.read_u32()? as usize;
                let mut choices = Vec::with_capacity(count);
                for _ in 0..count {
                    choices.push(cur.read_u64()?);
                }
                BlenderPropertyValue::Clsid { selected, choices }
            }
            9 => BlenderPropertyValue::Object(read_fixed_cstr(cur.read_bytes(64)?)),
            10 => BlenderPropertyValue::String(read_fixed_cstr(cur.read_bytes(64)?)),
            11 => BlenderPropertyValue::MarkerTemplate {
                typ: cur.read_u32()?,
                limit: cur.read_u32()?,
            },
            _ => {
                raw_tail = bytes[save_pos..].to_vec();
                break;
            }
        };
        properties.push(BlenderProperty { name: pname, value });
    }

    Ok(BlenderShader {
        desc,
        priority,
        strict_sorting,
        base_texture,
        base_transform,
        properties,
        raw_tail,
    })
}

/// Active parameter flags for `CEnvModifier` (`src/xrServerEntities/LevelGameDef.h:32-40`).
pub mod env_mod_flags {
    pub const VIEW_DIST: u16 = 1 << 0;
    pub const FOG_COLOR: u16 = 1 << 1;
    pub const FOG_DENSITY: u16 = 1 << 2;
    pub const AMBIENT_COLOR: u16 = 1 << 3;
    pub const SKY_COLOR: u16 = 1 << 4;
    pub const HEMI_COLOR: u16 = 1 << 5;
    pub const ALL: u16 = 0xFFFF;
}

/// Single localized weather/environment modifier zone (`CEnvModifier`).
#[derive(Debug, Clone, PartialEq)]
pub struct EnvModifier {
    pub name: Option<String>,
    pub position: [f32; 3],
    pub radius: f32,
    pub power: f32,
    pub far_plane: f32,
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    pub ambient: [f32; 3],
    pub sky_color: [f32; 3],
    pub hemi_color: [f32; 3],
    pub use_flags: u16,
}

/// Result of accumulating all active `EnvModifier` zones at a camera position.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvModAccum {
    pub total_power: f32,
    pub active_flags: u16,
    pub far_plane: f32,
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    pub ambient: [f32; 3],
    pub sky_color: [f32; 3],
    pub hemi_color: [f32; 3],
}

/// Parsed `level.env_mod` (and/or Anomaly `environment\env_mods\<level>.ltx`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LevelEnvModifiers {
    pub version: u32,
    pub modifiers: Vec<EnvModifier>,
}

/// Parse an `.ltx` `use_flags` integer matching C++ `strtoul(val, nullptr, 0)` (hex `0x..`, octal `0..`, or decimal).
fn parse_env_mod_use_flags(raw: &str) -> Option<u16> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    let parsed = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(hex.trim(), 16).ok()?
    } else if s.starts_with('0') && s.len() > 1 && s.bytes().skip(1).all(|b| (b'0'..=b'7').contains(&b)) {
        u32::from_str_radix(&s[1..], 8).ok()?
    } else {
        s.parse::<u32>().ok()?
    };
    Some((parsed & 0xFFFF) as u16)
}

impl LevelEnvModifiers {
    /// Parse binary `level.env_mod` (`CLevel::Load_GameSpecific_After` / `CEnvironment::Mods_Load`).
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let mut version = 0x0015u32;
        let mut modifiers = Vec::new();

        for chunk in reader.iter() {
            let chunk = chunk?;
            let payload = chunk.decode()?;
            if chunk.id == 0 && payload.len() == 4 {
                let mut vc = ByteCursor::new(&payload);
                version = vc.read_u32()?;
                continue;
            }
            let mut cur = ByteCursor::new(&payload);
            let position = cur.read_f32_3()?;
            let radius = cur.read_f32()?;
            let power = cur.read_f32()?;
            let far_plane = cur.read_f32()?;
            let fog_color = cur.read_f32_3()?;
            let fog_density = cur.read_f32()?;
            let ambient = cur.read_f32_3()?;
            let sky_color = cur.read_f32_3()?;
            let hemi_color = cur.read_f32_3()?;
            let use_flags = if version >= 0x0016 {
                cur.read_u16()?
            } else {
                env_mod_flags::ALL
            };
            modifiers.push(EnvModifier {
                name: None,
                position,
                radius,
                power,
                far_plane,
                fog_color,
                fog_density,
                ambient,
                sky_color,
                hemi_color,
                use_flags,
            });
        }

        Ok(Self { version, modifiers })
    }

    /// Parse Anomaly loose `.ltx` environment modifier table (`environment\env_mods\<level>.ltx`).
    pub fn from_ltx(doc: &LtxDocument<'_>) -> Self {
        let mut sections: Vec<&str> = doc.sections.keys().copied().collect();
        sections.sort_unstable();
        let mut modifiers = Vec::new();
        for section in sections {
            let Some(position) = doc.read_vec3(section, "position") else {
                continue;
            };
            let radius = doc.read_f32(section, "radius", 0.0);
            let power = doc.read_f32(section, "power", 1.0);
            let far_plane = doc.read_f32(section, "far_plane", 0.0);
            let fog_color = doc.read_vec3(section, "fog_color").unwrap_or([0.0; 3]);
            let fog_density = doc.read_f32(section, "fog_density", 0.0);
            let ambient = doc.read_vec3(section, "ambient").unwrap_or([0.0; 3]);
            let sky_color = doc.read_vec3(section, "sky_color").unwrap_or([0.0; 3]);
            let hemi_color = doc.read_vec3(section, "hemi_color").unwrap_or([0.0; 3]);

            let use_flags = if let Some(explicit) =
                doc.read_str(section, "use_flags").and_then(parse_env_mod_use_flags)
            {
                explicit
            } else {
                let mut inferred = 0u16;
                if doc.read_str(section, "far_plane").is_some() {
                    inferred |= env_mod_flags::VIEW_DIST;
                }
                if doc.read_str(section, "fog_color").is_some() {
                    inferred |= env_mod_flags::FOG_COLOR;
                }
                if doc.read_str(section, "fog_density").is_some() {
                    inferred |= env_mod_flags::FOG_DENSITY;
                }
                if doc.read_str(section, "ambient").is_some() {
                    inferred |= env_mod_flags::AMBIENT_COLOR;
                }
                if doc.read_str(section, "sky_color").is_some() {
                    inferred |= env_mod_flags::SKY_COLOR;
                }
                if doc.read_str(section, "hemi_color").is_some() {
                    inferred |= env_mod_flags::HEMI_COLOR;
                }
                if inferred == 0 {
                    env_mod_flags::ALL
                } else {
                    inferred
                }
            };

            modifiers.push(EnvModifier {
                name: Some(section.to_string()),
                position,
                radius,
                power,
                far_plane,
                fog_color,
                fog_density,
                ambient,
                sky_color,
                hemi_color,
                use_flags,
            });
        }
        Self {
            version: 0x0016,
            modifiers,
        }
    }

    /// Parse merged/owned Anomaly `.ltx` environment modifier tree (`OwnedLtxDocument`).
    pub fn from_owned_ltx(doc: &OwnedLtxDocument) -> Self {
        fn parse_vec3(raw: Option<&str>) -> Option<[f32; 3]> {
            let mut it = raw?
                .split(',')
                .map(|p| p.trim().parse::<f32>().ok());
            Some([it.next()??, it.next()??, it.next()??])
        }

        let mut sections: Vec<&str> = doc.sections.keys().map(String::as_str).collect();
        sections.sort_unstable();
        let mut modifiers = Vec::new();
        for section in sections {
            let Some(position) = parse_vec3(doc.read_str(section, "position")) else {
                continue;
            };
            let radius = doc.read_f32(section, "radius", 0.0);
            let power = doc.read_f32(section, "power", 1.0);
            let far_plane = doc.read_f32(section, "far_plane", 0.0);
            let fog_color = parse_vec3(doc.read_str(section, "fog_color")).unwrap_or([0.0; 3]);
            let fog_density = doc.read_f32(section, "fog_density", 0.0);
            let ambient = parse_vec3(doc.read_str(section, "ambient")).unwrap_or([0.0; 3]);
            let sky_color = parse_vec3(doc.read_str(section, "sky_color")).unwrap_or([0.0; 3]);
            let hemi_color = parse_vec3(doc.read_str(section, "hemi_color")).unwrap_or([0.0; 3]);

            let use_flags = if let Some(explicit) =
                doc.read_str(section, "use_flags").and_then(parse_env_mod_use_flags)
            {
                explicit
            } else {
                let mut inferred = 0u16;
                if doc.read_str(section, "far_plane").is_some() {
                    inferred |= env_mod_flags::VIEW_DIST;
                }
                if doc.read_str(section, "fog_color").is_some() {
                    inferred |= env_mod_flags::FOG_COLOR;
                }
                if doc.read_str(section, "fog_density").is_some() {
                    inferred |= env_mod_flags::FOG_DENSITY;
                }
                if doc.read_str(section, "ambient").is_some() {
                    inferred |= env_mod_flags::AMBIENT_COLOR;
                }
                if doc.read_str(section, "sky_color").is_some() {
                    inferred |= env_mod_flags::SKY_COLOR;
                }
                if doc.read_str(section, "hemi_color").is_some() {
                    inferred |= env_mod_flags::HEMI_COLOR;
                }
                if inferred == 0 {
                    env_mod_flags::ALL
                } else {
                    inferred
                }
            };

            modifiers.push(EnvModifier {
                name: Some(section.to_string()),
                position,
                radius,
                power,
                far_plane,
                fog_color,
                fog_density,
                ambient,
                sky_color,
                hemi_color,
                use_flags,
            });
        }
        Self {
            version: 0x0016,
            modifiers,
        }
    }

    /// Accumulate all modifier zones affecting `view_pos` (`CEnvModifier::sum`).
    pub fn evaluate_at(&self, view_pos: [f32; 3]) -> EnvModAccum {
        let mut out = EnvModAccum {
            total_power: 0.0,
            active_flags: 0,
            far_plane: 0.0,
            fog_color: [0.0; 3],
            fog_density: 0.0,
            ambient: [0.0; 3],
            sky_color: [0.0; 3],
            hemi_color: [0.0; 3],
        };

        for m in &self.modifiers {
            if m.radius <= 0.0 {
                continue;
            }
            let dx = view_pos[0] - m.position[0];
            let dy = view_pos[1] - m.position[1];
            let dz = view_pos[2] - m.position[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq >= m.radius * m.radius {
                continue;
            }
            let att = 1.0 - dist_sq.sqrt() / m.radius;
            let power = m.power * att;
            out.total_power += power;

            if (m.use_flags & env_mod_flags::VIEW_DIST) != 0 {
                out.far_plane += m.far_plane * power;
                out.active_flags |= env_mod_flags::VIEW_DIST;
            }
            if (m.use_flags & env_mod_flags::FOG_COLOR) != 0 {
                for axis in 0..3 {
                    out.fog_color[axis] += m.fog_color[axis] * power;
                }
                out.active_flags |= env_mod_flags::FOG_COLOR;
            }
            if (m.use_flags & env_mod_flags::FOG_DENSITY) != 0 {
                out.fog_density += m.fog_density * power;
                out.active_flags |= env_mod_flags::FOG_DENSITY;
            }
            if (m.use_flags & env_mod_flags::AMBIENT_COLOR) != 0 {
                for axis in 0..3 {
                    out.ambient[axis] += m.ambient[axis] * power;
                }
                out.active_flags |= env_mod_flags::AMBIENT_COLOR;
            }
            if (m.use_flags & env_mod_flags::SKY_COLOR) != 0 {
                for axis in 0..3 {
                    out.sky_color[axis] += m.sky_color[axis] * power;
                }
                out.active_flags |= env_mod_flags::SKY_COLOR;
            }
            if (m.use_flags & env_mod_flags::HEMI_COLOR) != 0 {
                for axis in 0..3 {
                    out.hemi_color[axis] += m.hemi_color[axis] * power;
                }
                out.active_flags |= env_mod_flags::HEMI_COLOR;
            }
        }

        out
    }
}

fn read_xray_matrix(cur: &mut ByteCursor<'_>) -> Result<XRayMatrix, FormatError> {
    let mut m = [[0.0f32; 4]; 4];
    for row in &mut m {
        for cell in row.iter_mut() {
            *cell = cur.read_f32()?;
        }
    }
    Ok(m)
}

/// Static particle emitter entry (`level.ps_static`).
#[derive(Debug, Clone, PartialEq)]
pub struct StaticParticle {
    pub game_type_mask: u16,
    pub ref_name: String,
    pub transform: XRayMatrix,
}

/// Parsed `level.ps_static` collection (`CPersistentUtils::LoadPS`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LevelStaticParticles {
    pub version: u32,
    pub particles: Vec<StaticParticle>,
}

impl LevelStaticParticles {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let reader = ChunkReader::new(data);
        let mut version = 0u32;
        let mut particles = Vec::new();

        for chunk in reader.iter() {
            let chunk = chunk?;
            let payload = chunk.decode()?;
            if chunk.id == 0 && payload.len() == 4 {
                let mut vc = ByteCursor::new(&payload);
                version = vc.read_u32()?;
                continue;
            }
            let mut cur = ByteCursor::new(&payload);
            let game_type_mask = if version > 0 {
                cur.read_u16()?
            } else {
                0xFFFF
            };
            let ref_name = cur.read_zstring()?;
            let transform = read_xray_matrix(&mut cur)?;
            particles.push(StaticParticle {
                game_type_mask,
                ref_name,
                transform,
            });
        }

        Ok(Self { version, particles })
    }
}

/// Volumetric fog region (`dx103DFluidVolume::Load` / `dx103DFluidData::Load`).
#[derive(Debug, Clone, PartialEq)]
pub struct FogVolume {
    pub profile: Option<String>,
    pub transform: XRayMatrix,
    pub obstacles: Vec<XRayMatrix>,
}

/// Parsed `level.fog_vol` (`version == 2` or `3`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LevelFogVolumes {
    pub version: u16,
    pub volumes: Vec<FogVolume>,
}

impl LevelFogVolumes {
    pub fn parse(data: &[u8]) -> Result<Self, FormatError> {
        let mut cur = ByteCursor::new(data);
        let version = cur.read_u16()?;
        if version != 2 && version != 3 {
            return Err(FormatError::InvalidValue(
                "unsupported level.fog_vol version (expected 2 or 3)",
            ));
        }

        let count = cur.read_u32()? as usize;
        let mut volumes = Vec::with_capacity(count);
        for _ in 0..count {
            let profile = if version >= 3 {
                Some(cur.read_line_string()?)
            } else {
                None
            };
            let transform = read_xray_matrix(&mut cur)?;
            let obs_count = cur.read_u32()? as usize;
            let mut obstacles = Vec::with_capacity(obs_count);
            for _ in 0..obs_count {
                obstacles.push(read_xray_matrix(&mut cur)?);
            }
            volumes.push(FogVolume {
                profile,
                transform,
                obstacles,
            });
        }

        Ok(Self { version, volumes })
    }
}

// ============================================================================
// Pure-Rust R4 / DX11 / Anomaly Blender Compiler & UI Render Helpers
// ============================================================================

/// Known X-Ray `IBlender` 64-bit `MK_CLSID` identifiers (`src/Layers/xrRender/blenders/Blender_CLSID.h`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlenderClsid {
    Default,
    DefaultAref,
    Vert,
    VertAref,
    LmBmmD,
    LaEmB,
    LmEbB,
    B,
    BmmD,
    Particle,
    ScreenSet,
    ScreenGray,
    Light,
    Blur,
    ShadowTex,
    ShadowWorld,
    Detail,
    Tree,
    Model,
    ModelEbB,
    EditorWire,
    EditorSel,
}

impl BlenderClsid {
    /// 8-byte `MK_CLSID` representation matching `u64::from_be_bytes`.
    pub const fn to_u64(self) -> u64 {
        let bytes: [u8; 8] = match self {
            Self::Default => *b"LM      ",
            Self::DefaultAref => *b"LM_AREF ",
            Self::Vert => *b"V       ",
            Self::VertAref => *b"V_AREF  ",
            Self::LmBmmD => *b"LmBmmD  ",
            Self::LaEmB => *b"LaEmB   ",
            Self::LmEbB => *b"LmEbB   ",
            Self::B => *b"BmmD    ",
            Self::BmmD => *b"BmmDold ",
            Self::Particle => *b"PARTICLE",
            Self::ScreenSet => *b"S_SET   ",
            Self::ScreenGray => *b"S_GRAY  ",
            Self::Light => *b"LIGHT   ",
            Self::Blur => *b"BLUR    ",
            Self::ShadowTex => *b"SH_TEX  ",
            Self::ShadowWorld => *b"SH_WORLD",
            Self::Detail => *b"D_STILL ",
            Self::Tree => *b"D_TREE  ",
            Self::Model => *b"MODEL   ",
            Self::ModelEbB => *b"MODELEbB",
            Self::EditorWire => *b"E_WIRE  ",
            Self::EditorSel => *b"E_SEL   ",
        };
        u64::from_be_bytes(bytes)
    }

    /// Decode a raw 64-bit `CLASS_ID` from [`BlenderDesc::clsid`].
    pub const fn from_u64(clsid: u64) -> Option<Self> {
        match &clsid.to_be_bytes() {
            b"LM      " => Some(Self::Default),
            b"LM_AREF " => Some(Self::DefaultAref),
            b"V       " => Some(Self::Vert),
            b"V_AREF  " => Some(Self::VertAref),
            b"LmBmmD  " => Some(Self::LmBmmD),
            b"LaEmB   " => Some(Self::LaEmB),
            b"LmEbB   " => Some(Self::LmEbB),
            b"BmmD    " => Some(Self::B),
            b"BmmDold " => Some(Self::BmmD),
            b"PARTICLE" => Some(Self::Particle),
            b"S_SET   " => Some(Self::ScreenSet),
            b"S_GRAY  " => Some(Self::ScreenGray),
            b"LIGHT   " => Some(Self::Light),
            b"BLUR    " => Some(Self::Blur),
            b"SH_TEX  " => Some(Self::ShadowTex),
            b"SH_WORLD" => Some(Self::ShadowWorld),
            b"D_STILL " => Some(Self::Detail),
            b"D_TREE  " => Some(Self::Tree),
            b"MODEL   " => Some(Self::Model),
            b"MODELEbB" => Some(Self::ModelEbB),
            b"E_WIRE  " => Some(Self::EditorWire),
            b"E_SEL   " => Some(Self::EditorSel),
            _ => None,
        }
    }

    /// Matches `IBlender::canBeDetailed()` in R4.
    pub const fn can_be_detailed(self) -> bool {
        matches!(
            self,
            Self::Default
                | Self::Vert
                | Self::DefaultAref
                | Self::VertAref
                | Self::LmBmmD
                | Self::BmmD
                | Self::Tree
                | Self::Model
        )
    }

    /// Matches `IBlender::canBeLMAPped()` in R4.
    pub const fn can_be_lmapped(self) -> bool {
        matches!(
            self,
            Self::DefaultAref | Self::LmBmmD | Self::BmmD | Self::LmEbB
        )
    }

    /// Matches `IBlender::canUseSteepParallax()` in R4.
    pub const fn can_use_steep_parallax(self) -> bool {
        matches!(
            self,
            Self::Default | Self::Vert | Self::DefaultAref | Self::VertAref
        )
    }
}

/// Hardware tessellation mode for deferred R4 shaders (`CBlender_Compile::TessMethod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum TessMethod {
    #[default]
    NoTess = 0,
    TessPn = 1,
    TessHm = 2,
    TessPnHm = 3,
}

impl TessMethod {
    pub const fn from_u32(val: u32) -> Self {
        match val {
            1 => Self::TessPn,
            2 => Self::TessHm,
            3 => Self::TessPnHm,
            _ => Self::NoTess,
        }
    }
}

/// DX10/DX11 MSAA alpha-test strategy (`CRender::MSAA_ATEST_*` in `r4.h:39-47`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Dx10MsaaAlphaTest {
    #[default]
    Off = 0,
    Dx100Atoc = 1,
    Dx101Native = 2,
    Dx101Atoc = 3,
}

/// Deferred shader element index (`SE_R2_*` and Anomaly SSFX element `3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum DefferElement {
    #[default]
    NormalHq = 0,
    NormalLq = 1,
    Shadow = 2,
    SsfxTerrainLow = 3,
}

/// Depth/stencil comparison function matching `D3DCMPFUNC` -> `D3D_COMPARISON_FUNC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CmpFunc {
    Never,
    Less,
    Equal,
    #[default]
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// Stencil operation matching `D3DSTENCILOP` -> `D3D_STENCIL_OP`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    IncrSat,
    DecrSat,
    Invert,
    Incr,
    Decr,
}

/// Per-pass stencil state (`CBlender_Compile::r_Stencil` / `r_StencilRef`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StencilState {
    pub enable: bool,
    pub func: CmpFunc,
    pub mask: u8,
    pub write_mask: u8,
    pub fail: StencilOp,
    pub pass: StencilOp,
    pub zfail: StencilOp,
    pub ref_value: u8,
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            enable: false,
            func: CmpFunc::Always,
            mask: 0x00,
            write_mask: 0x00,
            fail: StencilOp::Keep,
            pass: StencilOp::Keep,
            zfail: StencilOp::Keep,
            ref_value: 0x00,
        }
    }
}

/// Blend factor matching `D3DBLEND` -> `D3D_BLEND`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendFactor {
    Zero,
    #[default]
    One,
    SrcColor,
    InvSrcColor,
    SrcAlpha,
    InvSrcAlpha,
    DestAlpha,
    InvDestAlpha,
    DestColor,
    InvDestColor,
    SrcAlphaSat,
}

/// Per-pass blend and color-write state (`PassSET_Blend` / `r_ColorWriteEnable`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendState {
    pub enable: bool,
    pub src: BlendFactor,
    pub dst: BlendFactor,
    pub alpha_test: bool,
    pub alpha_ref: u8,
    pub alpha_to_coverage: bool,
    pub color_write_enable: [bool; 4],
}

impl Default for BlendState {
    fn default() -> Self {
        Self {
            enable: false,
            src: BlendFactor::One,
            dst: BlendFactor::Zero,
            alpha_test: false,
            alpha_ref: 0,
            alpha_to_coverage: false,
            color_write_enable: [true; 4],
        }
    }
}

/// Rasterizer cull mode matching `D3DCULL` (`r_CullMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    None,
    Cw,
    #[default]
    Ccw,
}

/// Sampler addressing mode (`D3DTEXTUREADDRESS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TexAddressMode {
    #[default]
    Wrap,
    Clamp,
    Border,
}

/// Standard DX10/DX11 sampler preset recognized by `CBlender_Compile::r_dx10Sampler`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerFilterMode {
    BaseAniso,
    Linear,
    Linear2,
    RtLinear,
    NoFilter,
    Material,
    Smap,
    Jitter,
    Point,
}

/// Texture resource binding emitted by `CBlender_Compile::r_dx10Texture`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureBindingSpec {
    pub resource_name: String,
    pub texture_name: String,
}

/// Sampler binding and overrides emitted by `CBlender_Compile::r_dx10Sampler` + `i_dx10*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplerBindingSpec {
    pub resource_name: String,
    pub filter_mode: SamplerFilterMode,
    pub address_mode: TexAddressMode,
    pub address_w: TexAddressMode,
    pub anisotropic_override: Option<bool>,
    pub border_color: Option<u32>,
}

/// Complete DX10/DX11/DX12 render state + shader + resource recipe for a single pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledPassPlan {
    pub vs: String,
    pub hs: String,
    pub ds: String,
    pub gs: String,
    pub ps: String,
    pub cs: String,
    pub fog: bool,
    pub lighting: bool,
    pub z_test: bool,
    pub z_write: bool,
    pub z_func: CmpFunc,
    pub cull_mode: CullMode,
    pub wireframe: bool,
    pub blend: BlendState,
    pub stencil: StencilState,
    pub shader_options: Vec<(String, String)>,
    pub textures: Vec<TextureBindingSpec>,
    pub samplers: Vec<SamplerBindingSpec>,
}

/// Result of compiling an R4 blender for one element index (`ShaderElement`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompiledShaderPlan {
    pub priority: i32,
    pub strict_sorting: bool,
    pub is_landscape_prepass: bool,
    pub is_landscape: bool,
    pub passes: Vec<CompiledPassPlan>,
}

/// Strip image file extensions exactly matching X-Ray's `fix_texture_name`
/// (`src/Layers/xrRenderDX10/dx10Texture.cpp:19-28`).
pub fn fix_texture_name(name: &str) -> String {
    const EXTS: &[&str] = &[".tga", ".dds", ".bmp", ".ogm", ".gif"];
    if let Some(dot) = name.rfind('.') {
        let ext = &name[dot..];
        if EXTS.iter().any(|e| ext.eq_ignore_ascii_case(e)) {
            return name[..dot].to_string();
        }
    }
    name.to_string()
}

struct PassCompiler {
    plan: CompiledShaderPlan,
    current: Option<CompiledPassPlan>,
}

impl PassCompiler {
    fn new(priority: i32, strict_sorting: bool) -> Self {
        Self {
            plan: CompiledShaderPlan {
                priority,
                strict_sorting,
                is_landscape_prepass: false,
                is_landscape: false,
                passes: Vec::new(),
            },
            current: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn r_pass_full(
        &mut self,
        vs: &str,
        gs: &str,
        ps: &str,
        fog: bool,
        z_test: bool,
        z_write: bool,
        blend_enable: bool,
        src: BlendFactor,
        dst: BlendFactor,
        alpha_test: bool,
        alpha_ref: u32,
    ) {
        let mut z_w = z_write;
        if !self.plan.passes.is_empty() {
            z_w = false;
        }
        let mut ab = blend_enable;
        if ab && src == BlendFactor::One && dst == BlendFactor::Zero {
            ab = false;
        }
        let pass = CompiledPassPlan {
            vs: vs.to_string(),
            hs: "null".to_string(),
            ds: "null".to_string(),
            gs: gs.to_string(),
            ps: ps.to_string(),
            cs: "null".to_string(),
            fog,
            lighting: false,
            z_test,
            z_write: z_w,
            z_func: if z_test { CmpFunc::LessEqual } else { CmpFunc::Always },
            cull_mode: CullMode::Ccw,
            wireframe: false,
            blend: BlendState {
                enable: ab,
                src: if ab { src } else { BlendFactor::One },
                dst: if ab { dst } else { BlendFactor::Zero },
                alpha_test,
                alpha_ref: alpha_ref.min(255) as u8,
                alpha_to_coverage: false,
                color_write_enable: [true; 4],
            },
            stencil: StencilState::default(),
            shader_options: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
        };
        self.current = Some(pass);
    }

    fn r_pass_default(&mut self, vs: &str, ps: &str, fog: bool, z_test: bool, z_write: bool) {
        self.r_pass_full(
            vs,
            "null",
            ps,
            fog,
            z_test,
            z_write,
            false,
            BlendFactor::One,
            BlendFactor::Zero,
            false,
            0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn r_tess_pass(
        &mut self,
        vs: &str,
        hs: &str,
        ds: &str,
        gs: &str,
        ps: &str,
        fog: bool,
        z_test: bool,
        z_write: bool,
    ) {
        self.r_pass_default(vs, ps, fog, z_test, z_write);
        if let Some(pass) = self.current.as_mut() {
            pass.gs = gs.to_string();
            pass.hs = hs.to_string();
            pass.ds = ds.to_string();
        }
    }

    fn r_compute_pass(&mut self, cs: &str) {
        let pass = CompiledPassPlan {
            vs: "null".to_string(),
            hs: "null".to_string(),
            ds: "null".to_string(),
            gs: "null".to_string(),
            ps: "null".to_string(),
            cs: cs.to_string(),
            fog: false,
            lighting: false,
            z_test: false,
            z_write: false,
            z_func: CmpFunc::Always,
            cull_mode: CullMode::Ccw,
            wireframe: false,
            blend: BlendState::default(),
            stencil: StencilState::default(),
            shader_options: Vec::new(),
            textures: Vec::new(),
            samplers: Vec::new(),
        };
        self.current = Some(pass);
    }

    fn pass_set_zb_full(&mut self, z_test: bool, mut z_write: bool, z_inverted: bool) {
        if !self.plan.passes.is_empty() {
            z_write = false;
        }
        if let Some(pass) = self.current.as_mut() {
            pass.z_test = z_test;
            pass.z_write = z_write;
            pass.z_func = if z_test {
                if z_inverted {
                    CmpFunc::Greater
                } else {
                    CmpFunc::LessEqual
                }
            } else {
                CmpFunc::Always
            };
        }
    }

    fn pass_set_zb(&mut self, z_test: bool, z_write: bool) {
        self.pass_set_zb_full(z_test, z_write, false);
    }

    fn set_blend_factors(&mut self, src: BlendFactor, dst: BlendFactor) {
        if let Some(pass) = self.current.as_mut() {
            pass.blend.src = src;
            pass.blend.dst = dst;
        }
    }

    fn pass_set_blend(
        &mut self,
        mut blend_enable: bool,
        src: BlendFactor,
        dst: BlendFactor,
        alpha_test: bool,
        alpha_ref: i32,
    ) {
        if blend_enable && src == BlendFactor::One && dst == BlendFactor::Zero {
            blend_enable = false;
        }
        if let Some(pass) = self.current.as_mut() {
            pass.blend.enable = blend_enable;
            pass.blend.src = if blend_enable { src } else { BlendFactor::One };
            pass.blend.dst = if blend_enable { dst } else { BlendFactor::Zero };
            pass.blend.alpha_test = alpha_test;
            pass.blend.alpha_ref = alpha_ref.clamp(0, 255) as u8;
        }
    }

    fn pass_set_light_fog(&mut self, lighting: bool, fog: bool) {
        if let Some(pass) = self.current.as_mut() {
            pass.lighting = lighting;
            pass.fog = fog;
        }
    }

    fn set_z_func(&mut self, z_func: CmpFunc) {
        if let Some(pass) = self.current.as_mut() {
            pass.z_func = z_func;
        }
    }

    fn set_wireframe(&mut self, wireframe: bool) {
        if let Some(pass) = self.current.as_mut() {
            pass.wireframe = wireframe;
        }
    }

    fn set_alpha_to_coverage(&mut self, enable: bool) {
        if let Some(pass) = self.current.as_mut() {
            pass.blend.alpha_to_coverage = enable;
        }
    }

    fn set_shader_options(&mut self, opts: Vec<(String, String)>) {
        if let Some(pass) = self.current.as_mut() {
            pass.shader_options = opts;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn r_stencil(
        &mut self,
        enable: bool,
        func: CmpFunc,
        mask: u8,
        write_mask: u8,
        fail: StencilOp,
        pass_op: StencilOp,
        zfail: StencilOp,
    ) {
        if let Some(pass) = self.current.as_mut() {
            pass.stencil.enable = enable;
            if enable {
                pass.stencil.func = func;
                pass.stencil.mask = mask;
                pass.stencil.write_mask = write_mask;
                pass.stencil.fail = fail;
                pass.stencil.pass = pass_op;
                pass.stencil.zfail = zfail;
            }
        }
    }

    fn r_stencil_ref(&mut self, ref_value: u8) {
        if let Some(pass) = self.current.as_mut() {
            pass.stencil.ref_value = ref_value;
        }
    }

    fn r_cull_mode(&mut self, cull: CullMode) {
        if let Some(pass) = self.current.as_mut() {
            pass.cull_mode = cull;
        }
    }

    fn r_color_write_enable(&mut self, r: bool, g: bool, b: bool, a: bool) {
        if let Some(pass) = self.current.as_mut() {
            pass.blend.color_write_enable = [r, g, b, a];
        }
    }

    fn r_dx10_texture(&mut self, resource_name: &str, texture_name: &str) {
        if let Some(pass) = self.current.as_mut() {
            pass.textures.push(TextureBindingSpec {
                resource_name: resource_name.to_string(),
                texture_name: fix_texture_name(texture_name),
            });
        }
    }

    fn r_dx10_sampler(&mut self, resource_name: &str) -> usize {
        let fixed = fix_texture_name(resource_name);
        let (filter_mode, address_mode, address_w, aniso) = match fixed.as_str() {
            "smp_nofilter" => (
                SamplerFilterMode::NoFilter,
                TexAddressMode::Clamp,
                TexAddressMode::Clamp,
                None,
            ),
            "smp_rtlinear" => (
                SamplerFilterMode::RtLinear,
                TexAddressMode::Clamp,
                TexAddressMode::Clamp,
                None,
            ),
            "smp_linear" => (
                SamplerFilterMode::Linear,
                TexAddressMode::Wrap,
                TexAddressMode::Wrap,
                None,
            ),
            "smp_linear2" => (
                SamplerFilterMode::Linear2,
                TexAddressMode::Wrap,
                TexAddressMode::Wrap,
                None,
            ),
            "smp_material" => (
                SamplerFilterMode::Material,
                TexAddressMode::Clamp,
                TexAddressMode::Wrap,
                None,
            ),
            "smp_smap" => (
                SamplerFilterMode::Smap,
                TexAddressMode::Clamp,
                TexAddressMode::Clamp,
                None,
            ),
            "smp_jitter" => (
                SamplerFilterMode::Jitter,
                TexAddressMode::Wrap,
                TexAddressMode::Wrap,
                None,
            ),
            "smp_point" => (
                SamplerFilterMode::Point,
                TexAddressMode::Wrap,
                TexAddressMode::Wrap,
                None,
            ),
            _ => (
                SamplerFilterMode::BaseAniso,
                TexAddressMode::Wrap,
                TexAddressMode::Wrap,
                Some(true),
            ),
        };
        if let Some(pass) = self.current.as_mut() {
            let idx = pass.samplers.len();
            pass.samplers.push(SamplerBindingSpec {
                resource_name: fixed,
                filter_mode,
                address_mode,
                address_w,
                anisotropic_override: aniso,
                border_color: None,
            });
            idx
        } else {
            usize::MAX
        }
    }

    fn i_dx10_address(&mut self, stage: usize, address: TexAddressMode) {
        if let Some(pass) = self.current.as_mut()
            && let Some(s) = pass.samplers.get_mut(stage)
        {
            s.address_mode = address;
            s.address_w = address;
        }
    }

    fn i_dx10_filter_anizo(&mut self, stage: usize, aniso: bool) {
        if let Some(pass) = self.current.as_mut()
            && let Some(s) = pass.samplers.get_mut(stage)
        {
            s.anisotropic_override = Some(aniso);
        }
    }

    fn i_dx10_border_color(&mut self, stage: usize, color: u32) {
        if let Some(pass) = self.current.as_mut()
            && let Some(s) = pass.samplers.get_mut(stage)
        {
            s.border_color = Some(color);
        }
    }

    fn jitter(&mut self) {
        self.r_dx10_texture("jitter0", "$user$jitter_0");
        self.r_dx10_texture("jitter1", "$user$jitter_1");
        self.r_dx10_texture("jitter2", "$user$jitter_2");
        self.r_dx10_texture("jitter3", "$user$jitter_3");
        self.r_dx10_texture("jitter4", "$user$jitter_4");
        self.r_dx10_texture("jitterMipped", "$user$jitter_mipped");
        self.r_dx10_sampler("smp_jitter");
    }

    fn r_end(&mut self) {
        if let Some(pass) = self.current.take() {
            self.plan.passes.push(pass);
        }
    }
}

/// Inputs for standalone `uber_deffer` evaluation (`src/Layers/xrRender/uber_deffer.cpp:7-221`).
#[derive(Debug, Clone, Default)]
pub struct UberDefferInput<'a> {
    pub hq: bool,
    pub vspec: &'a str,
    pub pspec: &'a str,
    pub aref: bool,
    pub detail_replace: Option<&'a str>,
    pub do_not_write: bool,
    pub l_textures: &'a [&'a str],
    pub base_detail_texture: &'a str,
    pub detail_diffuse: bool,
    pub detail_bump: bool,
    pub use_steep_parallax: bool,
    pub tess_method: TessMethod,
    pub dx11_enable_tessellation: bool,
    pub wireframe: bool,
}

/// Inputs for standalone `uber_shadow` evaluation (`src/Layers/xrRender/uber_deffer.cpp:224-338`).
#[derive(Debug, Clone, Default)]
pub struct UberShadowInput<'a> {
    pub vspec: &'a str,
    pub l_textures: &'a [&'a str],
    pub base_detail_texture: &'a str,
    pub detail_diffuse: bool,
    pub detail_bump: bool,
    pub tess_method: TessMethod,
    pub dx11_enable_tessellation: bool,
    pub wireframe: bool,
}

#[inline]
fn detect_lmap_slot2(l_textures: &[&str]) -> bool {
    l_textures
        .get(2)
        .is_some_and(|tex| tex.starts_with("lmap"))
}

fn build_tess_params_and_options(
    tess_method: TessMethod,
    lmap: bool,
    detail_diffuse: bool,
    detail_bump: bool,
) -> (String, Vec<(String, String)>) {
    let mut params = String::from("(");
    let mut opts = Vec::new();
    if matches!(tess_method, TessMethod::TessPn | TessMethod::TessPnHm) {
        opts.push(("TESS_PN".to_string(), "1".to_string()));
        params.push_str("TESS_PN,");
    }
    if matches!(tess_method, TessMethod::TessHm | TessMethod::TessPnHm) {
        opts.push(("TESS_HM".to_string(), "1".to_string()));
        params.push_str("TESS_HM,");
    }
    if lmap {
        opts.push(("USE_LM_HEMI".to_string(), "1".to_string()));
        params.push_str("USE_LM_HEMI,");
    }
    if detail_diffuse {
        opts.push(("USE_TDETAIL".to_string(), "1".to_string()));
        params.push_str("USE_TDETAIL,");
    }
    if detail_bump {
        opts.push(("USE_TDETAIL_BUMP".to_string(), "1".to_string()));
        params.push_str("USE_TDETAIL_BUMP,");
    }
    params.push(')');
    (params, opts)
}

fn compile_uber_deffer_into(
    compiler: &mut PassCompiler,
    input: &UberDefferInput<'_>,
    textures_descr: &TextureDescrTable,
) {
    let fname = input
        .l_textures
        .first()
        .map(|s| fix_texture_name(s))
        .unwrap_or_default();
    let bump_name = if fname.is_empty() {
        String::new()
    } else {
        textures_descr.get_bump_name(&fname).to_string()
    };
    let bump = !bump_name.is_empty();
    let lmap = detect_lmap_slot2(input.l_textures);

    let lmh = if lmap { "_lmh" } else { "" };
    let mut vs = format!("deffer_{}{lmh}", input.vspec);
    let mut ps = format!("deffer_{}{lmh}", input.pspec);
    let dt = input.detail_replace.unwrap_or(input.base_detail_texture);

    let mut tex_detail_bump = String::new();
    let mut tex_detail_bump_x = String::new();
    let mut has_detail_bump = false;
    if input.detail_bump {
        let det_bump = textures_descr.get_bump_name(dt);
        if !det_bump.is_empty() {
            has_detail_bump = true;
            tex_detail_bump = det_bump.to_string();
            tex_detail_bump_x = format!("{det_bump}#");
        }
    }

    if input.aref {
        ps.push_str("_aref");
    }

    let (fname_a, fname_b) = if !bump {
        vs.push_str("_flat");
        ps.push_str("_flat");
        if input.hq && (input.detail_diffuse || input.detail_bump) {
            vs.push_str("_d");
            ps.push_str("_d");
        }
        (String::new(), String::new())
    } else {
        vs.push_str("_bump");
        if input.hq && input.use_steep_parallax {
            ps.push_str("_steep");
        } else {
            ps.push_str("_bump");
        }
        if input.hq && (input.detail_diffuse || input.detail_bump) {
            vs.push_str("_d");
            if has_detail_bump {
                ps.push_str("_db");
            } else {
                ps.push_str("_d");
            }
        }
        (bump_name.clone(), format!("{bump_name}#"))
    };

    if bump && input.hq {
        vs.push_str("-hq");
        ps.push_str("-hq");
    }

    if bump && input.hq && input.dx11_enable_tessellation && input.tess_method != TessMethod::NoTess
    {
        let (params, opts) = build_tess_params_and_options(
            input.tess_method,
            lmap,
            input.detail_diffuse,
            input.detail_bump,
        );
        let tess_vs = format!("deffer_{}_bump{params}", input.vspec);
        let aref_suffix = if input.aref { "_aref" } else { "" };
        let tess_ps = format!("deffer_{}{aref_suffix}_bump{params}", input.pspec);
        let hs = format!("DX11\\tess{params}");
        let ds = format!("DX11\\tess{params}");
        compiler.r_tess_pass(&tess_vs, &hs, &ds, "null", &tess_ps, false, true, true);
        compiler.set_shader_options(opts);
        let stage = compiler.r_dx10_sampler("smp_bump_ds");
        if stage != usize::MAX {
            compiler.i_dx10_address(stage, TexAddressMode::Wrap);
            compiler.i_dx10_filter_anizo(stage, true);
        }
        if input.wireframe {
            compiler.set_wireframe(true);
        }
        compiler.r_dx10_texture("s_tbump", &fname_a);
        compiler.r_dx10_texture("s_tbumpX", &fname_b);
        if has_detail_bump {
            compiler.r_dx10_texture("s_tdetailBumpX", &tex_detail_bump_x);
        }
    } else if input.do_not_write {
        compiler.r_pass_default(&vs, &ps, false, true, false);
    } else {
        compiler.r_pass_default(&vs, &ps, false, true, true);
    }

    if let Some(base_tex) = input.l_textures.first() {
        compiler.r_dx10_texture("s_base", base_tex);
    }
    compiler.r_dx10_texture("s_bumpX", &fname_b);
    compiler.r_dx10_texture("s_bump", &fname_a);
    compiler.r_dx10_texture("s_bumpD", dt);
    compiler.r_dx10_texture("s_detail", dt);
    if has_detail_bump {
        compiler.r_dx10_texture("s_detailBump", &tex_detail_bump);
        compiler.r_dx10_texture("s_detailBumpX", &tex_detail_bump_x);
    }
    compiler.r_dx10_sampler("smp_base");
    compiler.r_dx10_sampler("smp_linear");
    if lmap {
        compiler.r_dx10_texture("s_hemi", input.l_textures[2]);
        compiler.r_dx10_sampler("smp_rtlinear");
    }
}

/// Evaluate standalone DX11/R4 `uber_deffer` pass setup (`uber_deffer.cpp:7-221`).
pub fn plan_uber_deffer(
    input: &UberDefferInput<'_>,
    textures_descr: &TextureDescrTable,
) -> CompiledPassPlan {
    let mut compiler = PassCompiler::new(1, false);
    compile_uber_deffer_into(&mut compiler, input, textures_descr);
    compiler.r_end();
    compiler.plan.passes.pop().expect("uber_deffer always emits one pass")
}

fn compile_uber_shadow_into(
    compiler: &mut PassCompiler,
    input: &UberShadowInput<'_>,
    textures_descr: &TextureDescrTable,
) {
    let fname = input
        .l_textures
        .first()
        .map(|s| fix_texture_name(s))
        .unwrap_or_default();
    let bump_name = if fname.is_empty() {
        String::new()
    } else {
        textures_descr.get_bump_name(&fname).to_string()
    };
    let bump = !bump_name.is_empty();
    let lmap = detect_lmap_slot2(input.l_textures);

    let dt = input.base_detail_texture;
    let mut tex_detail_bump = String::new();
    let mut tex_detail_bump_x = String::new();
    let mut has_detail_bump = false;
    if input.detail_bump {
        let det_bump = textures_descr.get_bump_name(dt);
        if !det_bump.is_empty() {
            has_detail_bump = true;
            tex_detail_bump = det_bump.to_string();
            tex_detail_bump_x = format!("{det_bump}#");
        }
    }

    let (fname_a, fname_b) = if !bump {
        (String::new(), String::new())
    } else {
        (bump_name.clone(), format!("{bump_name}#"))
    };

    if bump && input.dx11_enable_tessellation && input.tess_method != TessMethod::NoTess {
        let (params, opts) = build_tess_params_and_options(
            input.tess_method,
            lmap,
            input.detail_diffuse,
            input.detail_bump,
        );
        let vs = format!("deffer_{}_bump{params}", input.vspec);
        let hs = format!("DX11\\tess{params}");
        let ds = format!("DX11\\tess_shadow{params}");
        compiler.r_tess_pass(&vs, &hs, &ds, "null", "dumb", false, true, true);
        compiler.set_shader_options(opts);
        if let Some(base_tex) = input.l_textures.first() {
            compiler.r_dx10_texture("s_base", base_tex);
        }
        compiler.r_dx10_texture("s_bumpX", &fname_b);
        compiler.r_dx10_texture("s_bump", &fname_a);
        if has_detail_bump {
            compiler.r_dx10_texture("s_detailBump", &tex_detail_bump);
            compiler.r_dx10_texture("s_detailBumpX", &tex_detail_bump_x);
        }
        let stage = compiler.r_dx10_sampler("smp_bump_ds");
        if stage != usize::MAX {
            compiler.i_dx10_address(stage, TexAddressMode::Wrap);
            compiler.i_dx10_filter_anizo(stage, true);
        }
        if input.wireframe {
            compiler.set_wireframe(true);
        }
    } else {
        compiler.r_pass_default("shadow_direct_base", "dumb", false, true, true);
    }
}

/// Evaluate standalone DX11/R4 `uber_shadow` pass setup (`uber_deffer.cpp:224-338`).
pub fn plan_uber_shadow(
    input: &UberShadowInput<'_>,
    textures_descr: &TextureDescrTable,
) -> CompiledPassPlan {
    let mut compiler = PassCompiler::new(1, false);
    compile_uber_shadow_into(&mut compiler, input, textures_descr);
    compiler.r_end();
    compiler.plan.passes.pop().expect("uber_shadow always emits one pass")
}

/// Engine render options affecting R4/Anomaly blender pass compilation.
#[derive(Debug, Clone)]
pub struct R4CompileConfig {
    pub element: DefferElement,
    pub hud_element: bool,
    pub allow_detail: bool,
    pub advanced_pp: bool,
    pub r2_detail_bump: bool,
    pub terrain_prepass: bool,
    pub dx11_enable_tessellation: bool,
    pub wireframe: bool,
    pub msaa_alphatest: Dx10MsaaAlphaTest,
    pub ssfx_terrain: bool,
    pub ssfx_branches: bool,
    pub ssfx_glass: bool,
    /// Set `true` when `$game_textures$\{tex0}_lod_textures.dds` exists in VFS (`blender_BmmD.cpp:331`).
    pub terrain_has_lod_textures_dds: bool,
}

impl Default for R4CompileConfig {
    fn default() -> Self {
        Self {
            element: DefferElement::NormalHq,
            hud_element: false,
            allow_detail: true,
            advanced_pp: true,
            r2_detail_bump: true,
            terrain_prepass: false,
            dx11_enable_tessellation: false,
            wireframe: false,
            msaa_alphatest: Dx10MsaaAlphaTest::Off,
            ssfx_terrain: false,
            ssfx_branches: false,
            ssfx_glass: false,
            terrain_has_lod_textures_dds: false,
        }
    }
}

/// High-level R4 blender parameter block matching concrete `CBlender_*` subclasses in `r2_blenders.cpp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum R4BlenderSpec {
    DefferFlat {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        tessellation: TessMethod,
    },
    DefferAref {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        lmapped: bool,
        alpha_ref: i32,
        alpha_blend: bool,
    },
    DefferModel {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        alpha_blend: bool,
        alpha_ref: i32,
        tessellation: TessMethod,
    },
    ModelEbB {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        env_texture: String,
        alpha_blend: bool,
    },
    LmEbB {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        env_texture: String,
        alpha_blend: bool,
    },
    BmmD {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        detail_texture: String,
        r_detail: String,
        g_detail: String,
        b_detail: String,
        a_detail: String,
    },
    DetailStill {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        alpha_blend: bool,
    },
    Tree {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        alpha_blend: bool,
        not_an_tree: bool,
    },
    ScreenSet {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        blend_mode: u32,
        clamp: bool,
        alpha_ref: i32,
        z_test: bool,
        z_write: bool,
        lighting: bool,
        fog: bool,
    },
    Particle {
        priority: i32,
        strict_sorting: bool,
        base_texture: String,
        blend_mode: u32,
        clamp: bool,
    },
    EditorWire {
        priority: i32,
        strict_sorting: bool,
    },
    EditorSel {
        priority: i32,
        strict_sorting: bool,
    },
}

fn find_nth_prop<'a>(
    props: &'a [BlenderProperty],
    name: &str,
    nth: usize,
) -> Option<&'a BlenderPropertyValue> {
    let mut seen = 0usize;
    for p in props {
        if p.name.eq_ignore_ascii_case(name) {
            if seen == nth {
                return Some(&p.value);
            }
            seen += 1;
        }
    }
    None
}

fn prop_bool(props: &[BlenderProperty], name: &str, default: bool) -> bool {
    match find_nth_prop(props, name, 0) {
        Some(BlenderPropertyValue::Bool(v)) => *v,
        _ => default,
    }
}

fn prop_int(props: &[BlenderProperty], name: &str, default: i32) -> i32 {
    match find_nth_prop(props, name, 0) {
        Some(BlenderPropertyValue::Integer { value, .. }) => *value,
        _ => default,
    }
}

fn prop_token(props: &[BlenderProperty], name: &str, default: u32) -> u32 {
    match find_nth_prop(props, name, 0) {
        Some(BlenderPropertyValue::Token { selected, .. }) => *selected,
        _ => default,
    }
}

fn prop_texture_nth(props: &[BlenderProperty], name: &str, nth: usize, default: &str) -> String {
    match find_nth_prop(props, name, nth) {
        Some(BlenderPropertyValue::Texture(s)) => s.clone(),
        _ => default.to_string(),
    }
}

impl R4BlenderSpec {
    /// Convert a parsed [`BlenderShader`] entry from `shaders.xr` into its R4 compiler spec
    /// (`r2_blenders.cpp:18-46`). Returns `None` for legacy/disabled R1/R2 CLSIDs (`BLUR`, `LIGHT`, etc.).
    pub fn from_blender_shader(shader: &BlenderShader) -> Option<Self> {
        let clsid = BlenderClsid::from_u64(shader.desc.clsid)?;
        let p = &shader.properties;
        let priority = shader.priority;
        let strict_sorting = shader.strict_sorting;
        let base_texture = shader.base_texture.clone();

        match clsid {
            BlenderClsid::Default | BlenderClsid::Vert => Some(Self::DefferFlat {
                priority,
                strict_sorting,
                base_texture,
                tessellation: TessMethod::from_u32(prop_token(p, "Tessellation", 0)),
            }),
            BlenderClsid::DefaultAref | BlenderClsid::VertAref => Some(Self::DefferAref {
                priority,
                strict_sorting,
                base_texture,
                lmapped: clsid == BlenderClsid::DefaultAref,
                alpha_ref: prop_int(p, "Alpha ref", 200),
                alpha_blend: prop_bool(p, "Alpha-blend", false),
            }),
            BlenderClsid::Model => Some(Self::DefferModel {
                priority,
                strict_sorting,
                base_texture,
                alpha_blend: prop_bool(p, "Use alpha-channel", false),
                alpha_ref: prop_int(p, "Alpha ref", 32),
                tessellation: TessMethod::from_u32(prop_token(p, "Tessellation", 0)),
            }),
            BlenderClsid::ModelEbB => Some(Self::ModelEbB {
                priority,
                strict_sorting,
                base_texture,
                env_texture: prop_texture_nth(p, "Name", 1, "$null"),
                alpha_blend: prop_bool(p, "Alpha-Blend", false),
            }),
            BlenderClsid::LmEbB => Some(Self::LmEbB {
                priority,
                strict_sorting,
                base_texture,
                env_texture: prop_texture_nth(p, "Name", 1, "$null"),
                alpha_blend: prop_bool(p, "Alpha-Blend", false),
            }),
            BlenderClsid::LmBmmD | BlenderClsid::BmmD => Some(Self::BmmD {
                priority,
                strict_sorting,
                base_texture,
                detail_texture: prop_texture_nth(p, "Name", 1, "$null"),
                r_detail: prop_texture_nth(p, "R2-R", 0, "detail\\detail_grnd_grass"),
                g_detail: prop_texture_nth(p, "R2-G", 0, "detail\\detail_grnd_asphalt"),
                b_detail: prop_texture_nth(p, "R2-B", 0, "detail\\detail_grnd_earth"),
                a_detail: prop_texture_nth(p, "R2-A", 0, "detail\\detail_grnd_yantar"),
            }),
            BlenderClsid::Detail => Some(Self::DetailStill {
                priority,
                strict_sorting,
                base_texture,
                alpha_blend: prop_bool(p, "Alpha-blend", false),
            }),
            BlenderClsid::Tree => Some(Self::Tree {
                priority,
                strict_sorting,
                base_texture,
                alpha_blend: prop_bool(p, "Alpha-blend", false),
                not_an_tree: prop_bool(p, "Object LOD", false),
            }),
            BlenderClsid::ScreenSet => Some(Self::ScreenSet {
                priority,
                strict_sorting,
                base_texture,
                blend_mode: prop_token(p, "Blending", 0),
                clamp: prop_bool(p, "Texture clamp", true),
                alpha_ref: prop_int(p, "Alpha ref", 32),
                z_test: prop_bool(p, "Z-test", false),
                z_write: prop_bool(p, "Z-write", false),
                lighting: prop_bool(p, "Lighting", false),
                fog: prop_bool(p, "Fog", false),
            }),
            BlenderClsid::Particle => Some(Self::Particle {
                priority,
                strict_sorting,
                base_texture,
                blend_mode: prop_token(p, "Blending", 0),
                clamp: prop_bool(p, "Texture clamp", true),
            }),
            BlenderClsid::EditorWire => Some(Self::EditorWire {
                priority,
                strict_sorting,
            }),
            BlenderClsid::EditorSel => Some(Self::EditorSel {
                priority,
                strict_sorting,
            }),
            BlenderClsid::ScreenGray
            | BlenderClsid::Light
            | BlenderClsid::Blur
            | BlenderClsid::ShadowTex
            | BlenderClsid::ShadowWorld
            | BlenderClsid::LaEmB
            | BlenderClsid::B => None,
        }
    }

    #[inline]
    pub fn clsid(&self) -> BlenderClsid {
        match self {
            Self::DefferFlat { .. } => BlenderClsid::Default,
            Self::DefferAref { lmapped: true, .. } => BlenderClsid::DefaultAref,
            Self::DefferAref { lmapped: false, .. } => BlenderClsid::VertAref,
            Self::DefferModel { .. } => BlenderClsid::Model,
            Self::ModelEbB { .. } => BlenderClsid::ModelEbB,
            Self::LmEbB { .. } => BlenderClsid::LmEbB,
            Self::BmmD { .. } => BlenderClsid::BmmD,
            Self::DetailStill { .. } => BlenderClsid::Detail,
            Self::Tree { .. } => BlenderClsid::Tree,
            Self::ScreenSet { .. } => BlenderClsid::ScreenSet,
            Self::Particle { .. } => BlenderClsid::Particle,
            Self::EditorWire { .. } => BlenderClsid::EditorWire,
            Self::EditorSel { .. } => BlenderClsid::EditorSel,
        }
    }

    #[inline]
    pub fn base_texture(&self) -> &str {
        match self {
            Self::DefferFlat { base_texture, .. }
            | Self::DefferAref { base_texture, .. }
            | Self::DefferModel { base_texture, .. }
            | Self::ModelEbB { base_texture, .. }
            | Self::LmEbB { base_texture, .. }
            | Self::BmmD { base_texture, .. }
            | Self::DetailStill { base_texture, .. }
            | Self::Tree { base_texture, .. }
            | Self::ScreenSet { base_texture, .. }
            | Self::Particle { base_texture, .. } => base_texture,
            Self::EditorWire { .. } | Self::EditorSel { .. } => "$base0",
        }
    }

    #[inline]
    pub fn header(&self) -> (i32, bool) {
        match self {
            Self::DefferFlat { priority, strict_sorting, .. }
            | Self::DefferAref { priority, strict_sorting, .. }
            | Self::DefferModel { priority, strict_sorting, .. }
            | Self::ModelEbB { priority, strict_sorting, .. }
            | Self::LmEbB { priority, strict_sorting, .. }
            | Self::BmmD { priority, strict_sorting, .. }
            | Self::DetailStill { priority, strict_sorting, .. }
            | Self::Tree { priority, strict_sorting, .. }
            | Self::ScreenSet { priority, strict_sorting, .. }
            | Self::Particle { priority, strict_sorting, .. }
            | Self::EditorWire { priority, strict_sorting }
            | Self::EditorSel { priority, strict_sorting } => (*priority, *strict_sorting),
        }
    }
}

fn parse_base_slot_name<'a>(slot: &'a str, l_textures: &'a [&'a str]) -> Option<&'a str> {
    match slot {
        "$null" => None,
        "$base0" => l_textures.first().copied(),
        "$base1" => l_textures.get(1).copied(),
        "$base2" => l_textures.get(2).copied(),
        "$base3" => l_textures.get(3).copied(),
        "$base4" => l_textures.get(4).copied(),
        "$base5" => l_textures.get(5).copied(),
        "$base6" => l_textures.get(6).copied(),
        "$base7" => l_textures.get(7).copied(),
        other => Some(other),
    }
}

fn push_deffer_stencil_one(compiler: &mut PassCompiler) {
    compiler.r_stencil(
        true,
        CmpFunc::Always,
        0xFF,
        0x7F,
        StencilOp::Keep,
        StencilOp::Replace,
        StencilOp::Keep,
    );
    compiler.r_stencil_ref(0x01);
}

/// Compile any supported R4/Anomaly blender into its pass list (`CBlender_Compile::_cpp_Compile`
/// in `Blender_Recorder.cpp:40-136` + all `CRender::blender_create` classes in `r2_blenders.cpp`).
pub fn plan_r4_blender_compile(
    spec: &R4BlenderSpec,
    l_textures: &[&str],
    config: &R4CompileConfig,
    textures_descr: &TextureDescrTable,
) -> CompiledShaderPlan {
    let clsid = spec.clsid();
    let base_resolved = parse_base_slot_name(spec.base_texture(), l_textures);

    let mut b_detail = config.allow_detail && clsid.can_be_detailed();
    let mut base_detail_texture = "";
    if b_detail {
        if let Some(base) = base_resolved {
            if let Some((det, _scaler)) = textures_descr.get_detail_texture(base) {
                base_detail_texture = det;
            } else {
                b_detail = false;
            }
        } else {
            b_detail = false;
        }
    }

    let mut detail_diffuse = false;
    let mut detail_bump = false;
    if b_detail && let Some(base) = base_resolved {
        let (d, b) = textures_descr.get_texture_usage(base);
        detail_diffuse = d;
        detail_bump = b;
        if !(config.advanced_pp && config.r2_detail_bump) {
            detail_diffuse |= detail_bump;
            detail_bump = false;
        }
    }

    let use_steep_parallax = clsid.can_use_steep_parallax()
        && base_resolved.is_some_and(|base| textures_descr.use_steep_parallax(base));

    let (init_priority, init_strict) = spec.header();
    let mut compiler = PassCompiler::new(init_priority, init_strict);
    let tex0 = l_textures.first().copied().unwrap_or("");
    let b_use_atoc = config.msaa_alphatest == Dx10MsaaAlphaTest::Dx100Atoc;

    match spec {
        R4BlenderSpec::DefferFlat { tessellation, .. } => match config.element {
            DefferElement::NormalHq | DefferElement::NormalLq => {
                let hq = config.element == DefferElement::NormalHq;
                compile_uber_deffer_into(
                    &mut compiler,
                    &UberDefferInput {
                        hq,
                        vspec: "base",
                        pspec: "base",
                        aref: false,
                        detail_replace: None,
                        do_not_write: false,
                        l_textures,
                        base_detail_texture,
                        detail_diffuse,
                        detail_bump,
                        use_steep_parallax,
                        tess_method: *tessellation,
                        dx11_enable_tessellation: config.dx11_enable_tessellation,
                        wireframe: config.wireframe,
                    },
                    textures_descr,
                );
                push_deffer_stencil_one(&mut compiler);
                compiler.r_end();
            }
            DefferElement::Shadow => {
                compile_uber_shadow_into(
                    &mut compiler,
                    &UberShadowInput {
                        vspec: "base",
                        l_textures,
                        base_detail_texture,
                        detail_diffuse,
                        detail_bump,
                        tess_method: *tessellation,
                        dx11_enable_tessellation: config.dx11_enable_tessellation,
                        wireframe: config.wireframe,
                    },
                    textures_descr,
                );
                compiler.r_dx10_texture("s_base", tex0);
                compiler.r_dx10_sampler("smp_base");
                compiler.r_dx10_sampler("smp_linear");
                compiler.r_color_write_enable(false, false, false, false);
                compiler.r_end();
            }
            DefferElement::SsfxTerrainLow => {}
        },

        R4BlenderSpec::DefferAref {
            lmapped,
            alpha_ref,
            alpha_blend,
            ..
        } => {
            if *alpha_blend {
                match config.element {
                    DefferElement::NormalHq | DefferElement::NormalLq => {
                        if *lmapped {
                            compiler.r_pass_full(
                                "lmapE",
                                "null",
                                "lmapE",
                                true,
                                true,
                                false,
                                true,
                                BlendFactor::SrcAlpha,
                                BlendFactor::InvSrcAlpha,
                                true,
                                (*alpha_ref).max(0) as u32,
                            );
                            compiler.r_dx10_texture("s_base", tex0);
                            if let Some(t1) = l_textures.get(1) {
                                compiler.r_dx10_texture("s_lmap", t1);
                            }
                            if let Some(t2) = l_textures.get(2) {
                                compiler.r_dx10_texture("s_hemi", t2);
                            }
                            compiler.r_dx10_texture("s_env", "$user$env_s0");
                            compiler.r_dx10_sampler("smp_base");
                            compiler.r_dx10_sampler("smp_linear");
                            compiler.r_dx10_sampler("smp_rtlinear");
                            compiler.r_end();
                        } else {
                            compiler.r_pass_full(
                                "vert",
                                "null",
                                "vert",
                                true,
                                true,
                                false,
                                true,
                                BlendFactor::SrcAlpha,
                                BlendFactor::InvSrcAlpha,
                                true,
                                (*alpha_ref).max(0) as u32,
                            );
                            compiler.r_dx10_texture("s_base", tex0);
                            compiler.r_dx10_sampler("smp_base");
                            compiler.r_end();
                        }
                    }
                    DefferElement::Shadow | DefferElement::SsfxTerrainLow => {}
                }
            } else {
                compiler.plan.priority = 1;
                compiler.plan.strict_sorting = false;
                match config.element {
                    DefferElement::NormalHq | DefferElement::NormalLq => {
                        let hq = config.element == DefferElement::NormalHq;
                        if b_use_atoc {
                            compile_uber_deffer_into(
                                &mut compiler,
                                &UberDefferInput {
                                    hq,
                                    vspec: "base",
                                    pspec: "base_atoc",
                                    aref: true,
                                    detail_replace: None,
                                    do_not_write: false,
                                    l_textures,
                                    base_detail_texture,
                                    detail_diffuse,
                                    detail_bump,
                                    use_steep_parallax,
                                    tess_method: TessMethod::NoTess,
                                    dx11_enable_tessellation: config.dx11_enable_tessellation,
                                    wireframe: config.wireframe,
                                },
                                textures_descr,
                            );
                            push_deffer_stencil_one(&mut compiler);
                            compiler.r_color_write_enable(false, false, false, false);
                            compiler.set_alpha_to_coverage(true);
                            compiler.r_end();
                        }
                        compile_uber_deffer_into(
                            &mut compiler,
                            &UberDefferInput {
                                hq,
                                vspec: "base",
                                pspec: "base",
                                aref: true,
                                detail_replace: None,
                                do_not_write: false,
                                l_textures,
                                base_detail_texture,
                                detail_diffuse,
                                detail_bump,
                                use_steep_parallax,
                                tess_method: TessMethod::NoTess,
                                dx11_enable_tessellation: config.dx11_enable_tessellation,
                                wireframe: config.wireframe,
                            },
                            textures_descr,
                        );
                        push_deffer_stencil_one(&mut compiler);
                        if b_use_atoc {
                            compiler.set_z_func(CmpFunc::Equal);
                        }
                        compiler.r_end();
                    }
                    DefferElement::Shadow => {
                        compiler.r_pass_default(
                            "shadow_direct_base_aref",
                            "shadow_direct_base_aref",
                            false,
                            true,
                            true,
                        );
                        compiler.r_dx10_texture("s_base", tex0);
                        compiler.r_dx10_sampler("smp_base");
                        compiler.r_dx10_sampler("smp_linear");
                        compiler.r_color_write_enable(false, false, false, false);
                        compiler.r_end();
                    }
                    DefferElement::SsfxTerrainLow => {}
                }
            }
        }

        R4BlenderSpec::DefferModel {
            strict_sorting,
            alpha_blend,
            alpha_ref,
            tessellation,
            ..
        } => {
            let b_forward = (*alpha_blend && *alpha_ref < 16) || *strict_sorting;
            if b_forward {
                if matches!(
                    config.element,
                    DefferElement::NormalHq | DefferElement::NormalLq
                ) {
                    compiler.r_pass_full(
                        "model_def_lq",
                        "null",
                        "model_def_lq",
                        true,
                        true,
                        false,
                        true,
                        BlendFactor::SrcAlpha,
                        BlendFactor::InvSrcAlpha,
                        true,
                        (*alpha_ref).max(0) as u32,
                    );
                    compiler.r_dx10_texture("s_base", tex0);
                    compiler.r_dx10_sampler("smp_base");
                    compiler.r_end();
                }
            } else {
                let b_aref = *alpha_blend;
                let model_atoc = b_aref && b_use_atoc;
                match config.element {
                    DefferElement::NormalHq => {
                        if model_atoc {
                            compile_uber_deffer_into(
                                &mut compiler,
                                &UberDefferInput {
                                    hq: true,
                                    vspec: "model",
                                    pspec: "base_atoc",
                                    aref: b_aref,
                                    detail_replace: None,
                                    do_not_write: false,
                                    l_textures,
                                    base_detail_texture,
                                    detail_diffuse,
                                    detail_bump,
                                    use_steep_parallax,
                                    tess_method: *tessellation,
                                    dx11_enable_tessellation: config.dx11_enable_tessellation,
                                    wireframe: config.wireframe,
                                },
                                textures_descr,
                            );
                            push_deffer_stencil_one(&mut compiler);
                            compiler.r_color_write_enable(false, false, false, false);
                            compiler.set_alpha_to_coverage(true);
                            compiler.r_end();
                        }
                        if config.hud_element {
                            compile_uber_deffer_into(
                                &mut compiler,
                                &UberDefferInput {
                                    hq: true,
                                    vspec: "model_hud",
                                    pspec: "base_hud",
                                    aref: b_aref,
                                    detail_replace: None,
                                    do_not_write: false,
                                    l_textures,
                                    base_detail_texture,
                                    detail_diffuse,
                                    detail_bump,
                                    use_steep_parallax,
                                    tess_method: *tessellation,
                                    dx11_enable_tessellation: config.dx11_enable_tessellation,
                                    wireframe: config.wireframe,
                                },
                                textures_descr,
                            );
                            compiler.r_dx10_texture("s_hud_rain", "fx\\hud_rain");
                        } else {
                            compile_uber_deffer_into(
                                &mut compiler,
                                &UberDefferInput {
                                    hq: true,
                                    vspec: "model",
                                    pspec: "base",
                                    aref: b_aref,
                                    detail_replace: None,
                                    do_not_write: false,
                                    l_textures,
                                    base_detail_texture,
                                    detail_diffuse,
                                    detail_bump,
                                    use_steep_parallax,
                                    tess_method: *tessellation,
                                    dx11_enable_tessellation: config.dx11_enable_tessellation,
                                    wireframe: config.wireframe,
                                },
                                textures_descr,
                            );
                        }
                        push_deffer_stencil_one(&mut compiler);
                        if model_atoc {
                            compiler.set_z_func(CmpFunc::Equal);
                        }
                        compiler.r_end();
                    }
                    DefferElement::NormalLq => {
                        if model_atoc {
                            compile_uber_deffer_into(
                                &mut compiler,
                                &UberDefferInput {
                                    hq: false,
                                    vspec: "model",
                                    pspec: "base_atoc",
                                    aref: b_aref,
                                    detail_replace: None,
                                    do_not_write: false,
                                    l_textures,
                                    base_detail_texture,
                                    detail_diffuse,
                                    detail_bump,
                                    use_steep_parallax,
                                    tess_method: *tessellation,
                                    dx11_enable_tessellation: config.dx11_enable_tessellation,
                                    wireframe: config.wireframe,
                                },
                                textures_descr,
                            );
                            push_deffer_stencil_one(&mut compiler);
                            compiler.r_color_write_enable(false, false, false, false);
                            compiler.set_alpha_to_coverage(true);
                            compiler.r_end();
                        }
                        compile_uber_deffer_into(
                            &mut compiler,
                            &UberDefferInput {
                                hq: false,
                                vspec: "model",
                                pspec: "base",
                                aref: b_aref,
                                detail_replace: None,
                                do_not_write: false,
                                l_textures,
                                base_detail_texture,
                                detail_diffuse,
                                detail_bump,
                                use_steep_parallax,
                                tess_method: *tessellation,
                                dx11_enable_tessellation: config.dx11_enable_tessellation,
                                wireframe: config.wireframe,
                            },
                            textures_descr,
                        );
                        push_deffer_stencil_one(&mut compiler);
                        if model_atoc {
                            compiler.set_z_func(CmpFunc::Equal);
                        }
                        compiler.r_end();
                    }
                    DefferElement::Shadow => {
                        if b_aref {
                            compiler.r_pass_full(
                                "shadow_direct_model_aref",
                                "null",
                                "shadow_direct_base_aref",
                                false,
                                true,
                                true,
                                false,
                                BlendFactor::Zero,
                                BlendFactor::One,
                                true,
                                220,
                            );
                        } else {
                            compiler.r_pass_default("shadow_direct_model", "dumb", false, true, true);
                        }
                        compiler.r_dx10_texture("s_base", tex0);
                        compiler.r_dx10_sampler("smp_base");
                        compiler.r_dx10_sampler("smp_linear");
                        compiler.r_color_write_enable(false, false, false, false);
                        compiler.r_end();
                    }
                    DefferElement::SsfxTerrainLow => {}
                }
            }
        }

        R4BlenderSpec::ModelEbB {
            env_texture,
            alpha_blend,
            ..
        } => {
            if *alpha_blend {
                if matches!(
                    config.element,
                    DefferElement::NormalHq | DefferElement::NormalLq
                ) {
                    let shader_name = if config.ssfx_glass && !config.hud_element {
                        "ssfx_glass"
                    } else {
                        "model_env_lq"
                    };
                    compiler.r_pass_full(
                        shader_name,
                        "null",
                        shader_name,
                        true,
                        true,
                        false,
                        true,
                        BlendFactor::SrcAlpha,
                        BlendFactor::InvSrcAlpha,
                        true,
                        0,
                    );
                    compiler.r_dx10_texture("s_base", tex0);
                    compiler.r_dx10_texture("s_env", env_texture);
                    if config.ssfx_glass {
                        compiler.r_dx10_texture("s_accumulator", "$user$accum");
                        compiler.r_dx10_texture("s_screen", "$user$generic_temp");
                        compiler.r_dx10_texture("s_glass", "fx\\glass_normal");
                    }
                    compiler.r_dx10_sampler("smp_base");
                    compiler.r_dx10_sampler("smp_rtlinear");
                    compiler.r_end();
                }
            } else {
                match config.element {
                    DefferElement::NormalHq => {
                        if config.hud_element {
                            compile_uber_deffer_into(
                                &mut compiler,
                                &UberDefferInput {
                                    hq: true,
                                    vspec: "model_hud",
                                    pspec: "base_hud",
                                    aref: false,
                                    detail_replace: None,
                                    do_not_write: false,
                                    l_textures,
                                    base_detail_texture,
                                    detail_diffuse,
                                    detail_bump,
                                    use_steep_parallax,
                                    tess_method: TessMethod::NoTess,
                                    dx11_enable_tessellation: config.dx11_enable_tessellation,
                                    wireframe: config.wireframe,
                                },
                                textures_descr,
                            );
                            compiler.r_dx10_texture("s_hud_rain", "fx\\hud_rain");
                        } else {
                            compile_uber_deffer_into(
                                &mut compiler,
                                &UberDefferInput {
                                    hq: true,
                                    vspec: "model",
                                    pspec: "base",
                                    aref: false,
                                    detail_replace: None,
                                    do_not_write: false,
                                    l_textures,
                                    base_detail_texture,
                                    detail_diffuse,
                                    detail_bump,
                                    use_steep_parallax,
                                    tess_method: TessMethod::NoTess,
                                    dx11_enable_tessellation: config.dx11_enable_tessellation,
                                    wireframe: config.wireframe,
                                },
                                textures_descr,
                            );
                        }
                        push_deffer_stencil_one(&mut compiler);
                        compiler.r_end();
                    }
                    DefferElement::NormalLq => {
                        compile_uber_deffer_into(
                            &mut compiler,
                            &UberDefferInput {
                                hq: false,
                                vspec: "model",
                                pspec: "base",
                                aref: false,
                                detail_replace: None,
                                do_not_write: false,
                                l_textures,
                                base_detail_texture,
                                detail_diffuse,
                                detail_bump,
                                use_steep_parallax,
                                tess_method: TessMethod::NoTess,
                                dx11_enable_tessellation: config.dx11_enable_tessellation,
                                wireframe: config.wireframe,
                            },
                            textures_descr,
                        );
                        push_deffer_stencil_one(&mut compiler);
                        compiler.r_end();
                    }
                    DefferElement::Shadow => {
                        compiler.r_pass_default("shadow_direct_model", "dumb", false, true, true);
                        compiler.r_dx10_texture("s_base", tex0);
                        compiler.r_dx10_sampler("smp_base");
                        compiler.r_dx10_sampler("smp_linear");
                        compiler.r_color_write_enable(false, false, false, false);
                        compiler.r_end();
                    }
                    DefferElement::SsfxTerrainLow => {}
                }
            }
        }

        R4BlenderSpec::LmEbB {
            env_texture,
            alpha_blend,
            ..
        } => {
            if *alpha_blend {
                compiler.r_pass_full(
                    "lmapE",
                    "null",
                    "lmapE",
                    true,
                    true,
                    false,
                    true,
                    BlendFactor::SrcAlpha,
                    BlendFactor::InvSrcAlpha,
                    true,
                    0,
                );
            } else {
                compiler.r_pass_default("lmapE", "lmapE", true, true, true);
            }
            compiler.r_dx10_texture("s_base", tex0);
            compiler.r_dx10_sampler("smp_base");
            if let Some(t1) = l_textures.get(1) {
                compiler.r_dx10_texture("s_lmap", t1);
            }
            compiler.r_dx10_sampler("smp_linear");
            if let Some(t2) = l_textures.get(2) {
                compiler.r_dx10_texture("s_hemi", t2);
            }
            compiler.r_dx10_sampler("smp_rtlinear");
            compiler.r_dx10_texture("s_env", env_texture);
            compiler.r_end();
        }

        R4BlenderSpec::BmmD {
            detail_texture,
            r_detail,
            g_detail,
            b_detail,
            a_detail,
            ..
        } => {
            let mask_tex = format!("{tex0}_mask");
            let detail_rep = if detail_texture.is_empty() {
                None
            } else {
                Some(detail_texture.as_str())
            };
            let z_prepass = config.terrain_prepass;

            match config.element {
                DefferElement::NormalHq => {
                    if z_prepass {
                        compiler.plan.is_landscape_prepass = true;
                        compiler.r_pass_default(
                            "shadow_direct_base",
                            "shadow_direct_base",
                            false,
                            true,
                            true,
                        );
                        compiler.r_color_write_enable(false, false, false, false);
                        compiler.r_end();
                    }
                    let (vspec, pspec) = if config.ssfx_terrain {
                        compiler.plan.is_landscape = true;
                        ("terrain", "terrain_high")
                    } else {
                        ("impl", "impl")
                    };
                    compile_uber_deffer_into(
                        &mut compiler,
                        &UberDefferInput {
                            hq: true,
                            vspec,
                            pspec,
                            aref: false,
                            detail_replace: detail_rep,
                            do_not_write: z_prepass,
                            l_textures,
                            base_detail_texture,
                            detail_diffuse,
                            detail_bump,
                            use_steep_parallax,
                            tess_method: TessMethod::NoTess,
                            dx11_enable_tessellation: config.dx11_enable_tessellation,
                            wireframe: config.wireframe,
                        },
                        textures_descr,
                    );
                    if z_prepass {
                        compiler.set_z_func(CmpFunc::Equal);
                    }
                    compiler.r_dx10_texture("s_mask", &mask_tex);
                    if let Some(t1) = l_textures.get(1) {
                        compiler.r_dx10_texture("s_lmap", t1);
                    }
                    compiler.r_dx10_texture("s_dt_r", r_detail);
                    compiler.r_dx10_texture("s_dt_g", g_detail);
                    compiler.r_dx10_texture("s_dt_b", b_detail);
                    compiler.r_dx10_texture("s_dt_a", a_detail);
                    compiler.r_dx10_texture("s_dn_r", &format!("{r_detail}_bump"));
                    compiler.r_dx10_texture("s_dn_g", &format!("{g_detail}_bump"));
                    compiler.r_dx10_texture("s_dn_b", &format!("{b_detail}_bump"));
                    compiler.r_dx10_texture("s_dn_a", &format!("{a_detail}_bump"));
                    if config.ssfx_terrain {
                        compiler.r_dx10_texture("s_height_r", &format!("{r_detail}_height"));
                        compiler.r_dx10_texture("s_height_g", &format!("{g_detail}_height"));
                        compiler.r_dx10_texture("s_height_b", &format!("{b_detail}_height"));
                        compiler.r_dx10_texture("s_height_a", &format!("{a_detail}_height"));
                    }
                    compiler.r_dx10_texture("s_puddles_normal", "fx\\water_normal");
                    compiler.r_dx10_texture("s_puddles_perlin", "fx\\puddles_perlin");
                    compiler.r_dx10_texture("s_puddles_mask", &format!("{tex0}_puddles_mask"));
                    compiler.r_dx10_texture("s_rainsplash", "fx\\water_sbumpvolume");
                    compiler.r_dx10_sampler("smp_base");
                    compiler.r_dx10_sampler("smp_linear");
                    push_deffer_stencil_one(&mut compiler);
                    compiler.r_end();
                }
                DefferElement::NormalLq => {
                    if z_prepass {
                        compiler.plan.is_landscape_prepass = true;
                        compiler.r_pass_default(
                            "shadow_direct_base",
                            "shadow_direct_base",
                            false,
                            true,
                            true,
                        );
                        compiler.r_color_write_enable(false, false, false, false);
                        compiler.r_end();
                    }
                    let pspec = if config.ssfx_terrain {
                        compiler.plan.is_landscape = true;
                        "terrain_mid"
                    } else {
                        "impl"
                    };
                    compile_uber_deffer_into(
                        &mut compiler,
                        &UberDefferInput {
                            hq: false,
                            vspec: "base",
                            pspec,
                            aref: false,
                            detail_replace: detail_rep,
                            do_not_write: z_prepass,
                            l_textures,
                            base_detail_texture,
                            detail_diffuse,
                            detail_bump,
                            use_steep_parallax,
                            tess_method: TessMethod::NoTess,
                            dx11_enable_tessellation: config.dx11_enable_tessellation,
                            wireframe: config.wireframe,
                        },
                        textures_descr,
                    );
                    if z_prepass {
                        compiler.set_z_func(CmpFunc::Equal);
                    }
                    compiler.r_dx10_texture("s_mask", &mask_tex);
                    if config.ssfx_terrain {
                        let lod_tex = if config.terrain_has_lod_textures_dds {
                            format!("{tex0}_lod_textures")
                        } else {
                            "terrain\\default_lod_textures".to_string()
                        };
                        compiler.r_dx10_texture("s_lod_texture", &lod_tex);
                    }
                    compiler.r_dx10_sampler("smp_base");
                    compiler.r_dx10_sampler("smp_linear");
                    push_deffer_stencil_one(&mut compiler);
                    compiler.r_end();
                }
                DefferElement::SsfxTerrainLow => {
                    if z_prepass {
                        compiler.plan.is_landscape_prepass = true;
                        compiler.r_pass_default(
                            "shadow_direct_base",
                            "shadow_direct_base",
                            false,
                            true,
                            true,
                        );
                        compiler.r_color_write_enable(false, false, false, false);
                        compiler.r_end();
                    }
                    compiler.plan.is_landscape = true;
                    compile_uber_deffer_into(
                        &mut compiler,
                        &UberDefferInput {
                            hq: false,
                            vspec: "base",
                            pspec: "terrain_low",
                            aref: false,
                            detail_replace: detail_rep,
                            do_not_write: z_prepass,
                            l_textures,
                            base_detail_texture,
                            detail_diffuse,
                            detail_bump,
                            use_steep_parallax,
                            tess_method: TessMethod::NoTess,
                            dx11_enable_tessellation: config.dx11_enable_tessellation,
                            wireframe: config.wireframe,
                        },
                        textures_descr,
                    );
                    if z_prepass {
                        compiler.set_z_func(CmpFunc::Equal);
                    }
                    compiler.r_dx10_sampler("smp_linear");
                    push_deffer_stencil_one(&mut compiler);
                    compiler.r_end();
                }
                DefferElement::Shadow => {
                    let vs = if config.ssfx_terrain {
                        "shadow_direct_terrain"
                    } else {
                        "shadow_direct_base"
                    };
                    compiler.r_pass_default(vs, "dumb", false, true, true);
                    compiler.r_dx10_texture("s_base", tex0);
                    compiler.r_dx10_sampler("smp_base");
                    compiler.r_dx10_sampler("smp_linear");
                    compiler.r_color_write_enable(false, false, false, false);
                    compiler.r_end();
                }
            }
        }

        R4BlenderSpec::DetailStill { .. } => {
            if let Some(vspec) = match config.element {
                DefferElement::NormalHq => Some("detail_w"),
                DefferElement::NormalLq => Some("detail_s"),
                DefferElement::Shadow | DefferElement::SsfxTerrainLow => None,
            } {
                if b_use_atoc {
                    compile_uber_deffer_into(
                        &mut compiler,
                        &UberDefferInput {
                            hq: false,
                            vspec,
                            pspec: "base_atoc",
                            aref: true,
                            detail_replace: None,
                            do_not_write: false,
                            l_textures,
                            base_detail_texture,
                            detail_diffuse,
                            detail_bump,
                            use_steep_parallax,
                            tess_method: TessMethod::NoTess,
                            dx11_enable_tessellation: config.dx11_enable_tessellation,
                            wireframe: config.wireframe,
                        },
                        textures_descr,
                    );
                    push_deffer_stencil_one(&mut compiler);
                    compiler.r_color_write_enable(false, false, false, false);
                    compiler.r_cull_mode(CullMode::None);
                    compiler.set_alpha_to_coverage(true);
                    compiler.r_end();
                }
                compile_uber_deffer_into(
                    &mut compiler,
                    &UberDefferInput {
                        hq: false,
                        vspec,
                        pspec: "base",
                        aref: true,
                        detail_replace: None,
                        do_not_write: false,
                        l_textures,
                        base_detail_texture,
                        detail_diffuse,
                        detail_bump,
                        use_steep_parallax,
                        tess_method: TessMethod::NoTess,
                        dx11_enable_tessellation: config.dx11_enable_tessellation,
                        wireframe: config.wireframe,
                    },
                    textures_descr,
                );
                push_deffer_stencil_one(&mut compiler);
                compiler.r_cull_mode(CullMode::None);
                if b_use_atoc {
                    compiler.set_z_func(CmpFunc::Equal);
                }
                compiler.r_end();
            }
        }

        R4BlenderSpec::Tree {
            alpha_blend,
            not_an_tree,
            ..
        } => {
            let (mut tvs, mut tps) = (
                if *not_an_tree { "tree_s" } else { "tree" },
                "base",
            );
            let tvs_s = match (*not_an_tree, *alpha_blend) {
                (true, true) => "shadow_direct_tree_s_aref",
                (true, false) => "shadow_direct_tree_s",
                (false, true) => "shadow_direct_tree_aref",
                (false, false) => "shadow_direct_tree",
            };
            let tree_atoc = *alpha_blend && b_use_atoc;
            match config.element {
                DefferElement::NormalHq => {
                    if *alpha_blend && config.ssfx_branches {
                        tvs = "tree_branch";
                        tps = "tree_branch";
                    }
                    if tree_atoc {
                        let tps_atoc = format!("{tvs}_atoc");
                        compile_uber_deffer_into(
                            &mut compiler,
                            &UberDefferInput {
                                hq: true,
                                vspec: tvs,
                                pspec: &tps_atoc,
                                aref: *alpha_blend,
                                detail_replace: None,
                                do_not_write: false,
                                l_textures,
                                base_detail_texture,
                                detail_diffuse,
                                detail_bump,
                                use_steep_parallax,
                                tess_method: TessMethod::NoTess,
                                dx11_enable_tessellation: config.dx11_enable_tessellation,
                                wireframe: config.wireframe,
                            },
                            textures_descr,
                        );
                        push_deffer_stencil_one(&mut compiler);
                        compiler.r_color_write_enable(false, false, false, false);
                        compiler.set_alpha_to_coverage(true);
                        compiler.r_end();
                    }
                    compile_uber_deffer_into(
                        &mut compiler,
                        &UberDefferInput {
                            hq: true,
                            vspec: tvs,
                            pspec: tps,
                            aref: *alpha_blend,
                            detail_replace: None,
                            do_not_write: false,
                            l_textures,
                            base_detail_texture,
                            detail_diffuse,
                            detail_bump,
                            use_steep_parallax,
                            tess_method: TessMethod::NoTess,
                            dx11_enable_tessellation: config.dx11_enable_tessellation,
                            wireframe: config.wireframe,
                        },
                        textures_descr,
                    );
                    push_deffer_stencil_one(&mut compiler);
                    compiler.r_dx10_texture("s_waves", "fx\\wind_wave");
                    compiler.r_dx10_sampler("smp_linear2");
                    if tree_atoc {
                        compiler.set_z_func(CmpFunc::Equal);
                    }
                    compiler.r_end();
                }
                DefferElement::NormalLq => {
                    if tree_atoc {
                        compile_uber_deffer_into(
                            &mut compiler,
                            &UberDefferInput {
                                hq: false,
                                vspec: tvs,
                                pspec: "base_atoc",
                                aref: *alpha_blend,
                                detail_replace: None,
                                do_not_write: false,
                                l_textures,
                                base_detail_texture,
                                detail_diffuse,
                                detail_bump,
                                use_steep_parallax,
                                tess_method: TessMethod::NoTess,
                                dx11_enable_tessellation: config.dx11_enable_tessellation,
                                wireframe: config.wireframe,
                            },
                            textures_descr,
                        );
                        push_deffer_stencil_one(&mut compiler);
                        compiler.r_color_write_enable(false, false, false, false);
                        compiler.set_alpha_to_coverage(true);
                        compiler.r_end();
                    }
                    compile_uber_deffer_into(
                        &mut compiler,
                        &UberDefferInput {
                            hq: false,
                            vspec: tvs,
                            pspec: "base",
                            aref: *alpha_blend,
                            detail_replace: None,
                            do_not_write: false,
                            l_textures,
                            base_detail_texture,
                            detail_diffuse,
                            detail_bump,
                            use_steep_parallax,
                            tess_method: TessMethod::NoTess,
                            dx11_enable_tessellation: config.dx11_enable_tessellation,
                            wireframe: config.wireframe,
                        },
                        textures_descr,
                    );
                    push_deffer_stencil_one(&mut compiler);
                    compiler.r_dx10_texture("s_waves", "fx\\wind_wave");
                    compiler.r_dx10_sampler("smp_linear2");
                    if tree_atoc {
                        compiler.set_z_func(CmpFunc::Equal);
                    }
                    compiler.r_end();
                }
                DefferElement::Shadow => {
                    if *alpha_blend {
                        compiler.r_pass_full(
                            tvs_s,
                            "null",
                            "shadow_direct_base_aref",
                            false,
                            true,
                            true,
                            true,
                            BlendFactor::Zero,
                            BlendFactor::One,
                            true,
                            200,
                        );
                    } else {
                        compiler.r_pass_default(tvs_s, "shadow_direct_base", false, true, true);
                    }
                    compiler.r_dx10_texture("s_base", tex0);
                    compiler.r_dx10_sampler("smp_base");
                    compiler.r_dx10_sampler("smp_linear");
                    compiler.r_dx10_texture("s_waves", "fx\\wind_wave");
                    compiler.r_dx10_sampler("smp_linear2");
                    compiler.r_color_write_enable(false, false, false, false);
                    compiler.r_end();
                }
                DefferElement::SsfxTerrainLow => {}
            }
        }

        R4BlenderSpec::ScreenSet {
            blend_mode,
            clamp,
            alpha_ref,
            z_test,
            z_write,
            lighting,
            fog,
            ..
        } => {
            if *blend_mode == 6 {
                compiler.r_pass_default("stub_notransform_t", "stub_default_ma", false, true, true);
            } else {
                let vs = match *blend_mode {
                    9 => "stub_notransform_t_m4",
                    7 | 8 => "stub_notransform_t_m2",
                    _ => "stub_notransform_t",
                };
                compiler.r_pass_default(vs, "stub_default", false, true, true);
            }
            compiler.r_dx10_texture("s_base", tex0);
            let smp = compiler.r_dx10_sampler("smp_base");
            if *clamp && smp != usize::MAX {
                compiler.i_dx10_address(smp, TexAddressMode::Clamp);
            }
            compiler.pass_set_zb(*z_test, *z_write);
            match *blend_mode {
                0 => compiler.pass_set_blend(false, BlendFactor::One, BlendFactor::Zero, false, 0),
                1 | 8 | 9 => compiler.pass_set_blend(
                    true,
                    BlendFactor::SrcAlpha,
                    BlendFactor::InvSrcAlpha,
                    true,
                    *alpha_ref,
                ),
                2 => compiler.pass_set_blend(true, BlendFactor::One, BlendFactor::One, false, *alpha_ref),
                3 => compiler.pass_set_blend(
                    true,
                    BlendFactor::DestColor,
                    BlendFactor::Zero,
                    false,
                    *alpha_ref,
                ),
                4 | 6 => compiler.pass_set_blend(
                    true,
                    BlendFactor::DestColor,
                    BlendFactor::SrcColor,
                    false,
                    *alpha_ref,
                ),
                5 => compiler.pass_set_blend(
                    true,
                    BlendFactor::SrcAlpha,
                    BlendFactor::One,
                    true,
                    *alpha_ref,
                ),
                7 => compiler.pass_set_blend(true, BlendFactor::One, BlendFactor::Zero, true, 0),
                _ => {}
            }
            compiler.pass_set_light_fog(*lighting, *fog);
            compiler.r_end();
        }

        R4BlenderSpec::Particle {
            blend_mode, clamp, ..
        } => {
            match config.element {
                DefferElement::NormalHq | DefferElement::NormalLq => {
                    let valid = match *blend_mode {
                        0 => {
                            compiler.r_pass_full(
                                "deffer_particle",
                                "null",
                                "deffer_particle",
                                false,
                                true,
                                true,
                                false,
                                BlendFactor::One,
                                BlendFactor::Zero,
                                false,
                                200,
                            );
                            true
                        }
                        1 => {
                            compiler.r_pass_full(
                                "particle",
                                "null",
                                "particle",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::SrcAlpha,
                                BlendFactor::InvSrcAlpha,
                                true,
                                0,
                            );
                            true
                        }
                        2 => {
                            compiler.r_pass_full(
                                "particle",
                                "null",
                                "particle",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::One,
                                BlendFactor::One,
                                true,
                                0,
                            );
                            true
                        }
                        3 => {
                            compiler.r_pass_full(
                                "particle",
                                "null",
                                "particle",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::DestColor,
                                BlendFactor::Zero,
                                true,
                                0,
                            );
                            true
                        }
                        4 => {
                            compiler.r_pass_full(
                                "particle",
                                "null",
                                "particle",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::DestColor,
                                BlendFactor::SrcColor,
                                true,
                                0,
                            );
                            true
                        }
                        5 => {
                            compiler.r_pass_full(
                                "particle",
                                "null",
                                "particle",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::SrcAlpha,
                                BlendFactor::One,
                                true,
                                0,
                            );
                            true
                        }
                        _ => false,
                    };
                    if valid {
                        compiler.r_dx10_texture("s_base", tex0);
                        let smp = compiler.r_dx10_sampler("smp_base");
                        if *clamp && smp != usize::MAX {
                            compiler.i_dx10_address(smp, TexAddressMode::Clamp);
                        }
                        compiler.r_dx10_texture("s_position", "$user$position");
                        compiler.r_dx10_sampler("smp_nofilter");
                        compiler.r_end();
                    }
                }
                DefferElement::Shadow => {
                    let valid = match *blend_mode {
                        0 => {
                            compiler.r_pass_full(
                                "particle",
                                "null",
                                "particle",
                                false,
                                true,
                                true,
                                false,
                                BlendFactor::One,
                                BlendFactor::Zero,
                                true,
                                200,
                            );
                            compiler.r_color_write_enable(false, false, false, false);
                            true
                        }
                        1 => {
                            compiler.r_pass_full(
                                "particle-clip",
                                "null",
                                "particle_s-blend",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::DestColor,
                                BlendFactor::Zero,
                                true,
                                0,
                            );
                            true
                        }
                        2 => {
                            compiler.r_pass_full(
                                "particle-clip",
                                "null",
                                "particle_s-add",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::DestColor,
                                BlendFactor::Zero,
                                true,
                                0,
                            );
                            true
                        }
                        3 | 4 => {
                            compiler.r_pass_full(
                                "particle-clip",
                                "null",
                                "particle_s-mul",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::DestColor,
                                BlendFactor::Zero,
                                true,
                                0,
                            );
                            true
                        }
                        5 => {
                            compiler.r_pass_full(
                                "particle-clip",
                                "null",
                                "particle_s-aadd",
                                false,
                                true,
                                false,
                                true,
                                BlendFactor::DestColor,
                                BlendFactor::Zero,
                                true,
                                0,
                            );
                            true
                        }
                        _ => false,
                    };
                    if valid {
                        compiler.r_dx10_texture("s_base", tex0);
                        let smp = compiler.r_dx10_sampler("smp_base");
                        if *clamp && smp != usize::MAX {
                            compiler.i_dx10_address(smp, TexAddressMode::Clamp);
                        }
                        compiler.r_dx10_texture("s_position", "$user$position");
                        compiler.r_dx10_sampler("smp_nofilter");
                        compiler.r_end();
                    }
                }
                DefferElement::SsfxTerrainLow => {}
            }
        }

        R4BlenderSpec::EditorWire { .. } => {
            compiler.r_pass_default("editor", "simple_color", false, false, false);
            compiler.r_color_write_enable(true, true, true, true);
            compiler.r_end();
        }

        R4BlenderSpec::EditorSel { .. } => {
            compiler.r_pass_full(
                "editor",
                "null",
                "simple_color",
                false,
                true,
                false,
                true,
                BlendFactor::SrcAlpha,
                BlendFactor::InvSrcAlpha,
                false,
                0,
            );
            compiler.r_end();
        }
    }

    compiler.plan
}

/// Engine-internal non-serialized R4/DX11/SSFX blenders instantiated directly in
/// `CRenderTarget` and `dxEnvironmentRender` (`r4_rendertarget.cpp:387-486`,
/// `dxEnvironmentRender.h:9-48`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R4InternalBlenderKind {
    LightOccq,
    AccumDirectMask,
    AccumDirectMaskMsaa,
    AccumDirect,
    AccumDirectMsaa,
    AccumDirectVolumetricMsaa,
    AccumDirectVolumetricSunMsaa,
    AccumPoint,
    AccumPointMsaa,
    AccumSpot,
    AccumSpotMsaa,
    AccumVolumetricMsaa,
    AccumReflected,
    AccumReflectedMsaa,
    Combine,
    CombineMsaa,
    SsaoNoMsaa,
    SsaoMsaa,
    HdaoCs,
    HdaoMsaaCs,
    CreateMinMaxSm,
    MsaaEdges,
    Rain,
    RainMsaa,
    Skybox,
    BloomBuild,
    BloomBuildMsaa,
    PostprocessMsaa,
    SsfxBloomBuild,
    SsfxBloomLens,
    SsfxBloomDownsample,
    SsfxBloomUpsample,
    Blur,
    SsfxSsr,
    SsfxVolumetricBlur,
    SsfxAo,
    SsfxSss,
    SsfxSssExt,
    SsfxRain,
    SsfxWaterBlur,
    SsfxMotionBlur,
    SsfxFogScattering,
    Dof,
    GasmaskDrops,
    GasmaskDudv,
    Hdr10BloomDownsample,
    Hdr10BloomBlur,
    Hdr10BloomUpsample,
    Hdr10LensFlareDownsample,
    Hdr10LensFlareFgen,
    Hdr10LensFlareBlur,
    Hdr10LensFlareUpsample,
    Luminance,
    Lut,
    Nightvision,
    Fakescope,
    Heatvision,
    PpBloom,
    Smaa,
    SsfxTaa,
    Sunshafts,
}

/// Compilation options for [`plan_r4_internal_blender`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct R4InternalBlenderOptions {
    pub element: u32,
    pub dx10_msaa: bool,
    pub fp16_blend: bool,
    pub sunfilter: bool,
}

impl Default for R4InternalBlenderOptions {
    fn default() -> Self {
        Self {
            element: 0,
            dx10_msaa: false,
            fp16_blend: true,
            sunfilter: false,
        }
    }
}

/// Compile an internal R4/DX11/SSFX blender into a [`CompiledShaderPlan`],
/// matching the corresponding `CBlender_*::Compile` implementation verbatim.
pub fn plan_r4_internal_blender(
    kind: R4InternalBlenderKind,
    l_textures: &[&str],
    options: R4InternalBlenderOptions,
) -> CompiledShaderPlan {
    let mut c = PassCompiler::new(1, false);
    let tex0 = l_textures.first().copied().unwrap_or("");
    let elem = options.element;

    let add_combine_samplers = |c: &mut PassCompiler| {
        c.r_dx10_sampler("smp_linear");
        c.r_dx10_sampler("smp_nofilter");
        c.r_dx10_sampler("smp_rtlinear");
    };
    let add_nvg_common = |c: &mut PassCompiler| {
        c.r_dx10_texture("s_position", "$user$position");
        c.r_dx10_texture("s_image", "$user$generic0");
        c.r_dx10_texture("s_bloom_new", "$user$pp_bloom");
        c.r_dx10_texture("s_blur_2", "$user$blur_2");
        c.r_dx10_texture("s_blur_4", "$user$blur_4");
        c.r_dx10_texture("s_blur_8", "$user$blur_8");
    };
    let add_nvg_samplers = |c: &mut PassCompiler| {
        c.r_dx10_sampler("smp_base");
        c.r_dx10_sampler("smp_nofilter");
        c.r_dx10_sampler("smp_rtlinear");
    };
    let add_taa_samplers = |c: &mut PassCompiler| {
        c.r_dx10_sampler("smp_point");
        c.r_dx10_sampler("smp_linear");
        c.r_dx10_sampler("smp_rtlinear");
        c.r_dx10_sampler("smp_nofilter");
    };

    match kind {
        R4InternalBlenderKind::LightOccq => match elem {
            0 => {
                c.r_pass_default("dumb", "dumb", false, true, false);
                c.r_end();
            }
            1 => {
                c.r_pass_default("stub_notransform_t", "dumb", false, false, false);
                c.r_color_write_enable(false, false, false, false);
                c.r_cull_mode(CullMode::None);
                c.r_stencil(
                    true,
                    CmpFunc::LessEqual,
                    0xFF,
                    0x00,
                    StencilOp::Keep,
                    StencilOp::Keep,
                    StencilOp::Keep,
                );
                c.r_end();
            }
            2 => {
                c.r_pass_default("stub_notransform_t", "dumb", false, false, false);
                c.r_color_write_enable(false, false, false, false);
                c.r_cull_mode(CullMode::None);
                let wmask = if options.dx10_msaa { 0x7E } else { 0xFE };
                c.r_stencil(
                    true,
                    CmpFunc::Always,
                    0x00,
                    wmask,
                    StencilOp::Zero,
                    StencilOp::Zero,
                    StencilOp::Zero,
                );
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::AccumDirectMask | R4InternalBlenderKind::AccumDirectMaskMsaa => {
            let msaa = kind == R4InternalBlenderKind::AccumDirectMaskMsaa;
            match elem {
                0 | 1 => {
                    c.r_pass_default("accum_mask", "dumb", false, true, false);
                    c.r_color_write_enable(false, false, false, false);
                    c.r_end();
                }
                2 => {
                    c.r_pass_full(
                        "stub_notransform_t",
                        "null",
                        if msaa { "accum_sun_mask_msaa" } else { "accum_sun_mask_nomsaa" },
                        false,
                        false,
                        false,
                        true,
                        BlendFactor::Zero,
                        BlendFactor::One,
                        true,
                        1,
                    );
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_color_write_enable(false, false, false, false);
                    c.r_end();
                }
                3 => {
                    c.r_pass_default(
                        "accum_volume",
                        if msaa { "copy_p_msaa" } else { "copy_p_nomsaa" },
                        false,
                        false,
                        false,
                    );
                    c.r_dx10_texture("s_generic", "$user$accum_temp");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_end();
                }
                4 => {
                    c.r_pass_default(
                        "stub_notransform_t",
                        if msaa { "copy_msaa" } else { "copy_nomsaa" },
                        false,
                        false,
                        false,
                    );
                    c.r_dx10_texture("s_generic", "$user$accum_temp");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_end();
                }
                5 => {
                    c.r_pass_default("stub_notransform_t", "copy_nomsaa", false, false, false);
                    c.r_dx10_texture("s_generic", "$user$accum");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_end();
                }
                _ => {}
            }
        }

        R4InternalBlenderKind::AccumDirect | R4InternalBlenderKind::AccumDirectMsaa => {
            let msaa = kind == R4InternalBlenderKind::AccumDirectMsaa;
            let _ = options.sunfilter; // `blend` is hardcoded `FALSE` in both C++ paths
            let blend = false;
            let dst = BlendFactor::Zero;
            match elem {
                0 | 1 => {
                    c.r_pass_full(
                        "accum_sun",
                        "null",
                        if msaa {
                            "accum_sun_near_msaa_nominmax"
                        } else {
                            "accum_sun_near_nomsaa_nominmax"
                        },
                        false,
                        true,
                        false,
                        blend,
                        BlendFactor::One,
                        dst,
                        false,
                        0,
                    );
                    c.r_cull_mode(CullMode::None);
                    c.pass_set_zb_full(true, false, true);
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_material", "$user$material");
                    c.r_dx10_texture("s_accumulator", "$user$accum");
                    c.r_dx10_texture("s_lmap", "sunmask");
                    c.r_dx10_texture("s_smap", "$user$smap_depth");
                    if !msaa {
                        c.r_dx10_texture("s_smap_minmax", "$user$smap_depth_minmax");
                    }
                    c.r_dx10_texture("s_ssfx_sss", "$user$ssfx_sss");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.r_dx10_sampler("smp_linear");
                    c.jitter();
                    c.r_dx10_sampler("smp_smap");
                    c.r_end();
                }
                2 => {
                    c.r_pass_full(
                        "accum_sun",
                        "null",
                        if msaa { "accum_sun_far_msaa" } else { "accum_sun_far_nomsaa" },
                        false,
                        true,
                        false,
                        blend,
                        BlendFactor::One,
                        dst,
                        false,
                        0,
                    );
                    c.r_cull_mode(CullMode::None);
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_material", "$user$material");
                    c.r_dx10_texture("s_accumulator", "$user$accum");
                    c.r_dx10_texture("s_lmap", "sunmask");
                    c.r_dx10_texture("s_smap", "$user$smap_depth");
                    c.r_dx10_texture("s_ssfx_sss", "$user$ssfx_sss");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.r_dx10_sampler("smp_linear");
                    c.jitter();
                    let s = c.r_dx10_sampler("smp_smap");
                    if s != usize::MAX {
                        c.i_dx10_address(s, TexAddressMode::Border);
                        c.i_dx10_border_color(s, 0xFFFF_FFFF);
                    }
                    c.r_end();
                }
                3 => {
                    c.r_pass_default(
                        "stub_notransform_aa_AA",
                        if msaa { "accum_sun_msaa" } else { "accum_sun_nomsaa" },
                        false,
                        false,
                        false,
                    );
                    c.r_cull_mode(CullMode::None);
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_material", "$user$material");
                    c.r_dx10_texture("s_smap", "$user$generic0");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.jitter();
                    c.r_end();
                }
                4 => {
                    c.r_pass_full(
                        "accum_sun",
                        "null",
                        if msaa {
                            "accum_sun_near_msaa_minmax"
                        } else {
                            "accum_sun_near_nomsaa_minmax"
                        },
                        false,
                        true,
                        false,
                        blend,
                        BlendFactor::One,
                        dst,
                        false,
                        0,
                    );
                    c.r_cull_mode(CullMode::None);
                    c.pass_set_zb_full(true, false, true);
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_material", "$user$material");
                    c.r_dx10_texture("s_accumulator", "$user$accum");
                    c.r_dx10_texture("s_lmap", "sunmask");
                    c.r_dx10_texture("s_smap", "$user$smap_depth");
                    c.r_dx10_texture("s_smap_minmax", "$user$smap_depth_minmax");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.r_dx10_sampler("smp_linear");
                    c.jitter();
                    c.r_dx10_sampler("smp_smap");
                    c.r_end();
                }
                _ => {}
            }
        }

        R4InternalBlenderKind::AccumDirectVolumetricMsaa => {
            if elem == 0 {
                let _ = options.sunfilter;
                c.r_pass_full(
                    "accum_sun",
                    "null",
                    "accum_volumetric_sun_msaa",
                    false,
                    true,
                    false,
                    false,
                    BlendFactor::One,
                    BlendFactor::Zero,
                    false,
                    0,
                );
                c.r_dx10_texture("s_lmap", tex0);
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_texture("s_noise", "fx\\fx_noise");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_smap");
                c.r_end();
            }
        }

        R4InternalBlenderKind::AccumDirectVolumetricSunMsaa => {
            if elem == 0 {
                c.r_pass_full(
                    "accum_sun",
                    "null",
                    "accum_volumetric_sun_msaa",
                    false,
                    false,
                    false,
                    true,
                    BlendFactor::One,
                    BlendFactor::One,
                    false,
                    0,
                );
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_texture("s_position", "$user$position");
                c.jitter();
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_smap");
                c.r_end();
            }
        }

        R4InternalBlenderKind::AccumPoint | R4InternalBlenderKind::AccumPointMsaa => {
            let msaa = kind == R4InternalBlenderKind::AccumPointMsaa;
            let blend = options.fp16_blend;
            let dst = if blend { BlendFactor::One } else { BlendFactor::Zero };
            match elem {
                0 => {
                    c.r_pass_default(
                        "stub_notransform",
                        if msaa { "copy_msaa" } else { "copy_nomsaa" },
                        false,
                        false,
                        false,
                    );
                    c.r_dx10_texture("s_base", tex0);
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_end();
                }
                1 => {
                    c.r_pass_full(
                        "accum_volume",
                        "null",
                        if msaa {
                            "accum_omni_unshadowed_msaa"
                        } else {
                            "accum_omni_unshadowed_nomsaa"
                        },
                        false,
                        false,
                        false,
                        blend,
                        BlendFactor::One,
                        dst,
                        false,
                        0,
                    );
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_material", "$user$material");
                    c.r_dx10_texture("s_lmap", tex0);
                    c.r_dx10_texture("s_accumulator", "$user$accum");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.r_end();
                }
                2..=4 => {
                    let ps = match (elem, msaa) {
                        (4, true) => "accum_omni_transluent_msaa",
                        (4, false) => "accum_omni_transluent_nomsaa",
                        (_, true) => "accum_omni_normal_msaa",
                        (_, false) => "accum_omni_normal_nomsaa",
                    };
                    c.r_pass_full(
                        "accum_volume",
                        "null",
                        ps,
                        false,
                        false,
                        false,
                        blend,
                        BlendFactor::One,
                        dst,
                        false,
                        0,
                    );
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_material", "$user$material");
                    c.r_dx10_texture("s_lmap", tex0);
                    c.r_dx10_texture("s_smap", "$user$smap_depth");
                    c.r_dx10_texture("s_accumulator", "$user$accum");
                    c.r_dx10_texture("s_ssfx_sss", "$user$ssfx_sss_tmp");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.jitter();
                    c.r_dx10_sampler("smp_smap");
                    c.r_end();
                }
                _ => {}
            }
        }

        R4InternalBlenderKind::AccumSpot | R4InternalBlenderKind::AccumSpotMsaa => {
            let msaa = kind == R4InternalBlenderKind::AccumSpotMsaa;
            let blend = options.fp16_blend;
            let dst = if blend { BlendFactor::One } else { BlendFactor::Zero };
            match elem {
                0 => {
                    c.r_pass_default(
                        "stub_notransform",
                        if msaa { "copy_msaa" } else { "copy_nomsaa" },
                        false,
                        false,
                        false,
                    );
                    c.r_dx10_texture("s_base", tex0);
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_end();
                }
                1 => {
                    c.r_pass_full(
                        "accum_volume",
                        "null",
                        if msaa {
                            "accum_spot_unshadowed_msaa"
                        } else {
                            "accum_spot_unshadowed_nomsaa"
                        },
                        false,
                        false,
                        false,
                        blend,
                        BlendFactor::One,
                        dst,
                        false,
                        0,
                    );
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_material", "$user$material");
                    c.r_dx10_texture("s_lmap", tex0);
                    c.r_dx10_texture("s_accumulator", "$user$accum");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.r_end();
                }
                2..=4 => {
                    let ps = match (elem, msaa) {
                        (3 | 4, true) => "accum_spot_fullsize_msaa",
                        (3 | 4, false) => "accum_spot_fullsize_nomsaa",
                        (_, true) => "accum_spot_normal_msaa",
                        (_, false) => "accum_spot_normal_nomsaa",
                    };
                    c.r_pass_full(
                        "accum_volume",
                        "null",
                        ps,
                        false,
                        false,
                        false,
                        blend,
                        BlendFactor::One,
                        dst,
                        false,
                        0,
                    );
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_material", "$user$material");
                    c.r_dx10_texture("s_lmap", tex0);
                    c.r_dx10_texture("s_smap", "$user$smap_depth");
                    c.r_dx10_texture("s_accumulator", "$user$accum");
                    c.r_dx10_texture("s_ssfx_sss", "$user$ssfx_sss_tmp");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.jitter();
                    c.r_dx10_sampler("smp_smap");
                    c.r_end();
                }
                _ => {}
            }
        }

        R4InternalBlenderKind::AccumVolumetricMsaa => {
            if elem == 0 {
                c.r_pass_default("accum_volumetric", "accum_volumetric_msaa", false, false, false);
                c.r_dx10_texture("s_lmap", tex0);
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_texture("s_noise", "fx\\fx_noise");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_smap");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
        }

        R4InternalBlenderKind::AccumReflected | R4InternalBlenderKind::AccumReflectedMsaa => {
            let msaa = kind == R4InternalBlenderKind::AccumReflectedMsaa;
            let blend = options.fp16_blend;
            let dst = if blend { BlendFactor::One } else { BlendFactor::Zero };
            c.r_pass_full(
                "accum_volume",
                "null",
                if msaa { "accum_indirect_msaa" } else { "accum_indirect_nomsaa" },
                false,
                false,
                false,
                blend,
                BlendFactor::One,
                dst,
                false,
                0,
            );
            c.r_dx10_texture("s_position", "$user$position");
            c.r_dx10_texture("s_diffuse", "$user$albedo");
            c.r_dx10_texture("s_material", "$user$material");
            c.r_dx10_texture("s_accumulator", "$user$accum");
            c.r_dx10_sampler("smp_nofilter");
            c.r_dx10_sampler("smp_material");
            c.r_end();
        }

        R4InternalBlenderKind::Combine | R4InternalBlenderKind::CombineMsaa => {
            let msaa = kind == R4InternalBlenderKind::CombineMsaa;
            match elem {
                0 => {
                    c.r_pass_full(
                        "combine_1",
                        "null",
                        if msaa { "combine_1_msaa" } else { "combine_1_nomsaa" },
                        false,
                        false,
                        false,
                        true,
                        BlendFactor::InvSrcAlpha,
                        BlendFactor::SrcAlpha,
                        false,
                        0,
                    );
                    c.r_stencil(
                        true,
                        CmpFunc::LessEqual,
                        0xFF,
                        0x00,
                        StencilOp::Keep,
                        StencilOp::Keep,
                        StencilOp::Keep,
                    );
                    c.r_stencil_ref(0x01);
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_diffuse", "$user$albedo");
                    c.r_dx10_texture("s_accumulator", "$user$accum");
                    c.r_dx10_texture("s_depth", "$user$depth");
                    c.r_dx10_texture("s_tonemap", "$user$tonemap");
                    c.r_dx10_texture("s_material", "$user$material");
                    if msaa {
                        c.r_dx10_texture("s_occ", "$user$ssao_temp");
                        c.r_dx10_texture("s_half_depth", "$user$half_depth");
                    }
                    c.r_dx10_texture("env_s0", "$user$env_s0");
                    c.r_dx10_texture("env_s1", "$user$env_s1");
                    c.r_dx10_texture("sky_s0", "$user$sky0");
                    c.r_dx10_texture("sky_s1", "$user$sky1");
                    if !msaa {
                        c.r_dx10_texture("s_occ", "$user$ssao_temp");
                        c.r_dx10_texture("s_half_depth", "$user$half_depth");
                    }
                    c.r_dx10_texture("ssfx_ao", "$user$ssfx_temp");
                    c.r_dx10_texture("ssfx_il", "$user$ssfx_temp2");
                    c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                    c.r_dx10_texture("s_ssfx_sss", "$user$ssfx_sss");
                    c.jitter();
                    c.r_dx10_sampler("smp_linear");
                    c.r_dx10_sampler("smp_nofilter");
                    c.r_dx10_sampler("smp_material");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.r_end();
                }
                1..=4 => {
                    let ps = match elem {
                        1 => "combine_2_AA",
                        2 => "combine_2_NAA",
                        3 => "combine_2_AA_D",
                        _ => "combine_2_NAA_D",
                    };
                    let z_write = msaa && matches!(elem, 2 | 4);
                    c.r_pass_default("stub_notransform_aa_AA", ps, false, false, z_write);
                    c.r_dx10_texture("s_position", "$user$position");
                    c.r_dx10_texture("s_image", "$user$generic0");
                    c.r_dx10_texture("s_bloom", "$user$bloom1");
                    c.r_dx10_texture("s_bloom_new", "$user$pp_bloom");
                    c.r_dx10_texture(
                        "s_distort",
                        if msaa { "$user$generic1_r" } else { "$user$generic1" },
                    );
                    c.r_dx10_texture("s_blur_2", "$user$blur_2");
                    c.r_dx10_texture("s_blur_4", "$user$blur_4");
                    c.r_dx10_texture("s_blur_8", "$user$blur_8");

                    let lut_first = if msaa { elem <= 2 } else { elem == 1 };
                    if lut_first {
                        c.r_dx10_texture("s_lut_atlas", "shaders\\lut_atlas");
                        c.r_dx10_texture("s_lens_dirt", "shaders\\lens_dirt");
                        c.r_dx10_texture("s_noise_1", "shaders\\noise\\noise_tex");
                        c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                        c.r_dx10_texture("s_ssfx_sss", "$user$ssfx_sss");
                        c.r_dx10_texture("s_ssfx_bloom", "$user$ssfx_bloom1");
                    } else {
                        c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                        c.r_dx10_texture("s_ssfx_sss", "$user$ssfx_sss");
                        c.r_dx10_texture("s_ssfx_bloom", "$user$ssfx_bloom1");
                        c.r_dx10_texture("s_lut_atlas", "shaders\\lut_atlas");
                        c.r_dx10_texture("s_lens_dirt", "shaders\\lens_dirt");
                        c.r_dx10_texture("s_noise_1", "shaders\\noise\\noise_tex");
                    }
                    add_combine_samplers(&mut c);
                    c.r_end();
                }
                _ => {}
            }
        }

        R4InternalBlenderKind::SsaoNoMsaa => match elem {
            0 => {
                c.r_pass_default("combine_1", "ssao_calc_nomsaa", false, false, false);
                c.r_stencil(
                    true,
                    CmpFunc::LessEqual,
                    0xFF,
                    0x00,
                    StencilOp::Keep,
                    StencilOp::Keep,
                    StencilOp::Keep,
                );
                c.r_stencil_ref(0x01);
                c.r_cull_mode(CullMode::None);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_tonemap", "$user$tonemap");
                c.r_dx10_texture("s_half_depth", "$user$half_depth");
                c.jitter();
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            1 => {
                c.r_pass_default("combine_1", "depth_downs", false, false, false);
                c.r_cull_mode(CullMode::None);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_tonemap", "$user$tonemap");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsaoMsaa => {
            if elem == 0 {
                c.r_pass_default("combine_1", "ssao_calc_msaa", false, false, false);
                c.r_stencil(
                    true,
                    CmpFunc::Equal,
                    0x81,
                    0x00,
                    StencilOp::Keep,
                    StencilOp::Keep,
                    StencilOp::Keep,
                );
                c.r_stencil_ref(0x81);
                c.r_cull_mode(CullMode::None);
                c.r_dx10_texture("s_position", "$user$position");
                c.jitter();
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
        }

        R4InternalBlenderKind::HdaoCs | R4InternalBlenderKind::HdaoMsaaCs => {
            if elem == 0 {
                let cs = if kind == R4InternalBlenderKind::HdaoMsaaCs {
                    "ssao_hdao_msaa"
                } else {
                    "ssao_hdao"
                };
                c.r_compute_pass(cs);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
        }

        R4InternalBlenderKind::CreateMinMaxSm => {
            if elem == 0 {
                c.r_pass_default(
                    "stub_notransform_2uv",
                    "create_minmax_sm",
                    false,
                    false,
                    false,
                );
                c.pass_set_zb_full(false, false, false);
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
        }

        R4InternalBlenderKind::MsaaEdges => {
            if elem == 0 {
                c.r_pass_default(
                    "stub_notransform_2uv",
                    "mark_msaa_edges",
                    false,
                    false,
                    false,
                );
                c.pass_set_zb_full(false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
        }

        R4InternalBlenderKind::Rain => match elem {
            0 => {
                c.r_pass_default("stub_notransform_2uv", "rain_layer", false, true, false);
                c.pass_set_zb_full(true, false, true);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_material", "$user$material");
                c.r_dx10_texture("s_accumulator", "$user$accum");
                c.r_dx10_texture("s_lmap", "sunmask");
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_linear");
                c.jitter();
                c.r_dx10_sampler("smp_smap");
                c.r_dx10_texture("s_water", "water\\water_normal");
                c.r_end();
            }
            1 => {
                c.r_pass_default(
                    "stub_notransform_2uv",
                    "rain_patch_normal_nomsaa",
                    false,
                    true,
                    false,
                );
                c.pass_set_zb_full(true, false, true);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_material", "$user$material");
                c.r_dx10_texture("s_lmap", "sunmask");
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_texture("s_diffuse", "$user$albedo");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_base");
                c.jitter();
                c.r_dx10_sampler("smp_smap");
                c.r_dx10_texture("s_water", "water\\water_SBumpVolume");
                c.r_dx10_texture("s_waterFall", "water\\water_flowing_nmap");
                c.r_dx10_texture("s_hud_mask", "$user$ssfx_motion_vectors");
                c.r_end();
            }
            2 => {
                c.r_pass_default(
                    "stub_notransform_2uv",
                    "rain_apply_normal_nomsaa",
                    false,
                    true,
                    false,
                );
                c.pass_set_zb_full(true, false, true);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_material", "$user$material");
                c.r_dx10_texture("s_lmap", "sunmask");
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_linear");
                c.jitter();
                c.r_dx10_sampler("smp_smap");
                c.r_dx10_texture("s_patched_normal", "$user$accum");
                c.r_color_write_enable(true, true, false, false);
                c.r_end();
            }
            3 => {
                c.r_pass_full(
                    "stub_notransform_2uv",
                    "null",
                    "rain_apply_gloss_nomsaa",
                    false,
                    true,
                    false,
                    true,
                    BlendFactor::One,
                    BlendFactor::One,
                    false,
                    0,
                );
                c.pass_set_zb_full(true, false, true);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_material", "$user$material");
                c.r_dx10_texture("s_lmap", "sunmask");
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_linear");
                c.jitter();
                c.r_dx10_sampler("smp_smap");
                c.r_dx10_texture("s_patched_normal", "$user$accum");
                c.set_blend_factors(BlendFactor::Zero, BlendFactor::SrcColor);
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::RainMsaa => match elem {
            0 => {
                c.r_pass_default(
                    "stub_notransform_2uv",
                    "rain_patch_normal_msaa",
                    false,
                    true,
                    false,
                );
                c.pass_set_zb_full(true, false, true);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_material", "$user$material");
                c.r_dx10_texture("s_lmap", "sunmask");
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_texture("s_diffuse", "$user$albedo");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_base");
                c.jitter();
                c.r_dx10_sampler("smp_smap");
                c.r_dx10_texture("s_water", "water\\water_SBumpVolume");
                c.r_dx10_texture("s_waterFall", "water\\water_flowing_nmap");
                c.r_end();
            }
            1 => {
                c.r_pass_default(
                    "stub_notransform_2uv",
                    "rain_apply_normal_msaa",
                    false,
                    true,
                    false,
                );
                c.pass_set_zb_full(true, false, true);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_material", "$user$material");
                c.r_dx10_texture("s_lmap", "sunmask");
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_linear");
                c.jitter();
                c.r_dx10_sampler("smp_smap");
                c.r_dx10_texture("s_patched_normal", "$user$accum");
                c.r_color_write_enable(true, true, false, false);
                c.r_end();
            }
            2 => {
                c.r_pass_full(
                    "stub_notransform_2uv",
                    "null",
                    "rain_apply_gloss_msaa",
                    false,
                    true,
                    false,
                    true,
                    BlendFactor::One,
                    BlendFactor::One,
                    false,
                    0,
                );
                c.pass_set_zb_full(true, false, true);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_material", "$user$material");
                c.r_dx10_texture("s_lmap", "sunmask");
                c.r_dx10_texture("s_smap", "$user$smap_depth");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_material");
                c.r_dx10_sampler("smp_linear");
                c.jitter();
                c.r_dx10_sampler("smp_smap");
                c.r_dx10_texture("s_patched_normal", "$user$accum");
                c.set_blend_factors(BlendFactor::Zero, BlendFactor::SrcColor);
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::Skybox => match elem {
            0 => {
                c.r_pass_default("sky2", "sky2", false, true, false);
                c.r_dx10_texture("s_sky0", "$null");
                c.r_dx10_texture("s_sky1", "$null");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_texture("s_tonemap", "$user$tonemap");
                c.pass_set_zb(false, false);
                c.r_end();
            }
            1 => {
                c.r_pass_default("ssfx_sky_mv", "ssfx_sky_mv", false, true, false);
                c.pass_set_zb(false, false);
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::BloomBuild | R4InternalBlenderKind::BloomBuildMsaa => {
            match elem {
                0 => {
                    c.r_pass_full(
                        "stub_notransform_build",
                        "null",
                        "bloom_build",
                        false,
                        false,
                        false,
                        false,
                        BlendFactor::SrcAlpha,
                        BlendFactor::InvSrcAlpha,
                        false,
                        0,
                    );
                    c.r_dx10_texture("s_image", "$user$generic1");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.r_end();
                }
                1 => {
                    c.r_pass_default("stub_notransform_filter", "bloom_filter", false, false, false);
                    c.r_dx10_texture("s_bloom", "$user$bloom1");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.r_end();
                }
                2 => {
                    c.r_pass_default("stub_notransform_filter", "bloom_filter", false, false, false);
                    c.r_dx10_texture("s_bloom", "$user$bloom2");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.r_end();
                }
                3 => {
                    c.r_pass_default(
                        "stub_notransform_build",
                        "bloom_filter_f",
                        false,
                        false,
                        false,
                    );
                    c.r_dx10_texture("s_bloom", "$user$bloom1");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.r_end();
                }
                4 => {
                    c.r_pass_default(
                        "stub_notransform_build",
                        "bloom_filter_f",
                        false,
                        false,
                        false,
                    );
                    c.r_dx10_texture("s_bloom", "$user$bloom2");
                    c.r_dx10_sampler("smp_rtlinear");
                    c.r_end();
                }
                _ => {}
            }
        }

        R4InternalBlenderKind::PostprocessMsaa => match elem {
            0 => {
                c.r_pass_full(
                    "stub_notransform_postpr",
                    "null",
                    "postprocess",
                    false,
                    false,
                    false,
                    false,
                    BlendFactor::SrcAlpha,
                    BlendFactor::InvSrcAlpha,
                    false,
                    0,
                );
                c.r_dx10_texture("s_base0", "$user$generic");
                c.r_dx10_texture("s_base1", "$user$generic");
                c.r_dx10_texture("s_noise", "fx\\fx_noise2");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            4 => {
                c.r_pass_full(
                    "stub_notransform_postpr",
                    "null",
                    "postprocess_CM",
                    false,
                    false,
                    false,
                    false,
                    BlendFactor::SrcAlpha,
                    BlendFactor::InvSrcAlpha,
                    false,
                    0,
                );
                c.r_dx10_texture("s_base0", "$user$generic");
                c.r_dx10_texture("s_base1", "$user$generic");
                c.r_dx10_texture("s_noise", "fx\\fx_noise2");
                c.r_dx10_texture("s_grad0", "$user$cmap0");
                c.r_dx10_texture("s_grad1", "$user$cmap1");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxBloomBuild => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "ssfx_bloom", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_scene", "$user$blur_2");
                c.r_dx10_texture("s_emissive", "$user$ssfx_bloom_emissive");
                c.r_dx10_texture("s_hud_mask", "$user$ssfx_motion_vectors");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            1 => {
                c.r_pass_default("stub_screen_space", "ssfx_bloom_combine", false, false, false);
                c.r_dx10_texture("s_bloom", "$user$ssfx_bloom_tmp2");
                c.r_dx10_texture("s_emissive", "$user$ssfx_bloom_emissive");
                c.r_dx10_texture("s_ssfx_lensdirt", "fx\\lens_dirt");
                c.r_dx10_texture("s_starburst", "fx\\starburst");
                c.r_dx10_texture("s_bloom_lens", "$user$ssfx_bloom_lens");
                c.r_dx10_texture("s_lenscolors", "fx\\lens_colors");
                c.r_dx10_sampler("smp_point");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxBloomLens => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "ssfx_bloom_flares", false, false, false);
                c.r_dx10_texture("s_emissive", "$user$ssfx_bloom_emissive");
                c.r_dx10_texture("s_lenscolors", "fx\\lens_colors");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            1 | 2 => {
                let tex = if elem == 1 {
                    "$user$ssfx_bloom_tmp4"
                } else {
                    "$user$ssfx_bloom_tmp4_2"
                };
                c.r_pass_default("stub_screen_space", "ssfx_bloom_upsample", false, false, false);
                c.r_dx10_texture("s_bloom", tex);
                c.r_dx10_texture("s_downsample", tex);
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxBloomDownsample => {
            let src = match elem {
                0 => Some(("$user$ssfx_bloom1", true)),
                1 => Some(("$user$ssfx_bloom_tmp2", true)),
                2 => Some(("$user$ssfx_bloom_tmp4", false)),
                3 => Some(("$user$ssfx_bloom_tmp8", false)),
                4 => Some(("$user$ssfx_bloom_tmp16", false)),
                5 => Some(("$user$ssfx_bloom_tmp32", false)),
                _ => None,
            };
            if let Some((tex, with_nofilter)) = src {
                c.r_pass_default(
                    "stub_screen_space",
                    "ssfx_bloom_downsample",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture("s_bloom", tex);
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_rtlinear");
                if with_nofilter {
                    c.r_dx10_sampler("smp_nofilter");
                }
                c.r_end();
            }
        }

        R4InternalBlenderKind::SsfxBloomUpsample => {
            let pair = match elem {
                0 => Some(("$user$ssfx_bloom_tmp64", "$user$ssfx_bloom_tmp64")),
                1 => Some(("$user$ssfx_bloom_tmp32", "$user$ssfx_bloom_tmp32_2")),
                2 => Some(("$user$ssfx_bloom_tmp16", "$user$ssfx_bloom_tmp16_2")),
                3 => Some(("$user$ssfx_bloom_tmp8", "$user$ssfx_bloom_tmp8_2")),
                4 => Some(("$user$ssfx_bloom_tmp4", "$user$ssfx_bloom_tmp4_2")),
                _ => None,
            };
            if let Some((down_tex, bloom_tex)) = pair {
                c.r_pass_default("stub_screen_space", "ssfx_bloom_upsample", false, false, false);
                c.r_dx10_texture("s_downsample", down_tex);
                c.r_dx10_texture("s_bloom", bloom_tex);
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
        }

        R4InternalBlenderKind::Blur => {
            let img = match elem {
                0 | 2 | 4 => Some("$user$generic0"),
                1 => Some("$user$blur_h_2"),
                3 => Some("$user$blur_h_4"),
                5 => Some("$user$blur_h_8"),
                _ => None,
            };
            if let Some(s_image) = img {
                c.r_pass_default("stub_screen_space", "pp_blur", false, false, false);
                c.r_dx10_texture("s_image", s_image);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_lut_atlas", "shaders\\lut_atlas");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
        }

        R4InternalBlenderKind::SsfxSsr => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "ssfx_ssr", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("ssr_image", "$user$ssfx_ssr");
                c.r_dx10_texture("s_rimage", "$user$generic_temp");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_texture("s_prev_pos", "$user$ssfx_prev_p");
                c.r_dx10_texture("s_gloss_data", "$user$ssfx_temp3");
                c.r_dx10_texture("blue_noise", "fx\\blue_noise");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            1 | 2 => {
                c.r_pass_default("stub_screen_space", "ssfx_ssr_blur", false, false, false);
                c.r_dx10_texture(
                    "ssr_image",
                    if elem == 1 { "$user$ssfx" } else { "$user$ssfx_temp" },
                );
                c.r_dx10_texture("s_diffuse", "$user$albedo");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            3 => {
                c.r_pass_default("stub_screen_space", "ssfx_ssr_combine", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_rimage", "$user$generic_temp");
                c.r_dx10_texture("ssr_image", "$user$ssfx_temp2");
                c.r_dx10_texture("s_gloss_data", "$user$ssfx_temp3");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            4 => {
                c.r_pass_default("stub_screen_space", "ssfx_ssr_noblur", false, false, false);
                c.r_dx10_texture("ssr_image", "$user$ssfx");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            5 => {
                c.r_pass_default("stub_screen_space", "ssfx_ssr_gloss", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_diffuse", "$user$albedo");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_texture("env_s0", "$user$env_s0");
                c.r_dx10_texture("env_s1", "$user$env_s1");
                c.r_dx10_texture("sky_s0", "$user$sky0");
                c.r_dx10_texture("sky_s1", "$user$sky1");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxVolumetricBlur => match elem {
            0..=3 => {
                let vol_buf = match elem {
                    0 => "$user$ssfx_volumetric",
                    1 => "$user$ssfx_volumetric_tmp",
                    2 => "$user$generic2",
                    _ => "$user$ssfx_accum",
                };
                c.r_pass_default("stub_screen_space", "ssfx_volumetric_blur", false, false, false);
                c.r_dx10_texture("vol_buffer", vol_buf);
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            5 => {
                c.r_pass_default(
                    "stub_screen_space",
                    "ssfx_volumetric_combine",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture("vol_buffer", "$user$generic2");
                c.r_dx10_texture("vol_point", "$user$ssfx_volumetric");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxAo => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "ssfx_ao", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_texture("ssfx_ao", "$user$ssfx_ao");
                c.r_dx10_texture("s_prev_pos", "$user$ssfx_prev_p");
                c.r_dx10_texture("jitter0", "$user$jitter_0");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_jitter");
                c.r_end();
            }
            1 | 2 => {
                c.r_pass_default("stub_screen_space", "ssfx_ao_blur", false, false, false);
                c.r_dx10_texture(
                    "ao_image",
                    if elem == 1 { "$user$ssfx_temp" } else { "$user$ssfx_temp3" },
                );
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            3 => {
                c.r_pass_default("stub_screen_space", "ssfx_il", false, false, false);
                c.r_dx10_texture("s_accumulator", "$user$accum");
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_texture("ssfx_ao", "$user$ssfx_il");
                c.r_dx10_texture("s_prev_pos", "$user$ssfx_prev_p");
                c.r_dx10_texture("jitter0", "$user$jitter_0");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_jitter");
                c.r_end();
            }
            4 | 5 => {
                c.r_pass_default("stub_screen_space", "ssfx_il_blur", false, false, false);
                c.r_dx10_texture(
                    "ao_image",
                    if elem == 4 { "$user$ssfx_temp2" } else { "$user$ssfx_temp3" },
                );
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxSss => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "ssfx_sss", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_texture("sss_image", "$user$ssfx_sss");
                c.r_dx10_texture("s_prev_pos", "$user$ssfx_prev_p");
                c.r_dx10_texture("jitter0", "$user$jitter_0");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_jitter");
                c.r_end();
            }
            1 | 2 => {
                c.r_pass_default("stub_screen_space", "ssfx_sss_blur", false, false, false);
                c.r_dx10_texture(
                    "sss_image",
                    if elem == 1 { "$user$ssfx" } else { "$user$ssfx_temp" },
                );
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_jitter");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxSssExt => match elem {
            0 | 1 => {
                c.r_pass_default("stub_screen_space", "ssfx_sss_ext", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_texture(
                    "sss_image",
                    if elem == 0 { "$user$ssfx_sss_ext" } else { "$user$ssfx_sss_ext2" },
                );
                c.r_dx10_texture("s_prev_pos", "$user$ssfx_prev_p");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_dx10_sampler("smp_jitter");
                c.r_end();
            }
            2 => {
                c.r_pass_default(
                    "stub_screen_space",
                    "ssfx_sss_ext_combine",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture("sss_image_1", "$user$ssfx_sss_ext");
                c.r_dx10_texture("sss_image_2", "$user$ssfx_sss_ext2");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxRain => {
            if matches!(elem, 0 | 1) {
                c.r_pass_default("stub_screen_space", "ssfx_rain", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture(
                    "ssfx_color_buffer",
                    if elem == 0 { "$user$generic0" } else { "$user$generic0_r" },
                );
                c.r_dx10_texture("volumetric_buffer", "$user$ssfx_volumetric");
                c.r_dx10_texture("bloom_buffer", "$user$ssfx_bloom1");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
        }

        R4InternalBlenderKind::SsfxWaterBlur => match elem {
            0..=2 => {
                let (ps, buf) = match elem {
                    0 => ("ssfx_water_blur", "$user$ssfx_temp"),
                    1 => ("ssfx_water_blur", "$user$ssfx_temp2"),
                    _ => ("ssfx_water_noblur", "$user$ssfx_temp2"),
                };
                c.r_pass_default("stub_screen_space", ps, false, false, false);
                c.r_dx10_texture("water_buffer", buf);
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            5 => {
                c.r_pass_default("stub_screen_space", "ssfx_water_waves", false, false, false);
                c.r_dx10_texture("water_waves", "fx\\water_height");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxMotionBlur => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "ssfx_motion_blur", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_image", "$user$generic0");
                c.r_dx10_texture("ssfx_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            1 => {
                c.r_pass_default("stub_screen_space", "ssfx_motion_blur", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_image", "$user$ssfx_temp");
                c.r_dx10_texture("ssfx_motion_vectors", "$user$ssfx_motion_vectors");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxFogScattering => match elem {
            0 => {
                c.r_pass_default(
                    "stub_screen_space",
                    "ssfx_fog_scattering",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_image", "$user$generic0");
                c.r_dx10_texture("s_blur_2", "$user$blur_2");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_linear");
                c.r_end();
            }
            2 | 3 => {
                c.r_pass_default(
                    "stub_screen_space",
                    "ssfx_fog_scattering_blur",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture(
                    "s_image",
                    if elem == 2 { "$user$generic0" } else { "$user$blur_4" },
                );
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::Dof => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "depth_of_field", false, false, false);
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_image", "$user$generic0");
                c.r_dx10_texture("s_blur_2", "$user$blur_2");
                add_nvg_samplers(&mut c);
                c.r_end();
            }
            1 => {
                c.r_pass_default("stub_screen_space", "post_processing", false, false, false);
                c.r_dx10_texture("samplero_pepero", "$user$dof");
                add_nvg_samplers(&mut c);
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::GasmaskDrops => {
            c.r_pass_default("stub_screen_space", "gasmask_drops", false, false, false);
            c.r_dx10_texture("s_image", "$user$generic0");
            c.r_dx10_texture("s_noise", "shaders\\gasmasks\\mask_noise");
            c.r_dx10_sampler("smp_base");
            c.r_dx10_sampler("smp_nofilter");
            c.r_dx10_sampler("smp_rtlinear");
            c.r_dx10_sampler("smp_linear");
            c.r_end();
        }

        R4InternalBlenderKind::GasmaskDudv => {
            c.r_pass_default("stub_screen_space", "gasmask_dudv", false, false, false);
            c.r_dx10_texture("s_image", "$user$generic0");
            c.r_dx10_texture("s_mask_droplets", "shaders\\gasmasks\\mask_droplets");
            for (slot, path) in [
                ("s_mask_nm_1", "shaders\\gasmasks\\mask_nm_1"),
                ("s_mask_nm_2", "shaders\\gasmasks\\mask_nm_2"),
                ("s_mask_nm_3", "shaders\\gasmasks\\mask_nm_3"),
                ("s_mask_nm_4", "shaders\\gasmasks\\mask_nm_4"),
                ("s_mask_nm_5", "shaders\\gasmasks\\mask_nm_5"),
                ("s_mask_nm_6", "shaders\\gasmasks\\mask_nm_6"),
                ("s_mask_nm_7", "shaders\\gasmasks\\mask_nm_7"),
                ("s_mask_nm_8", "shaders\\gasmasks\\mask_nm_8"),
                ("s_mask_nm_9", "shaders\\gasmasks\\mask_nm_9"),
                ("s_mask_nm_10", "shaders\\gasmasks\\mask_nm_10"),
            ] {
                c.r_dx10_texture(slot, path);
            }
            add_nvg_samplers(&mut c);
            c.r_end();
        }

        R4InternalBlenderKind::Hdr10BloomDownsample => {
            c.r_pass_default(
                "stub_screen_space",
                "hdr10_bloom_downsample",
                false,
                false,
                false,
            );
            if elem == 0 {
                c.r_dx10_texture("s_hdr10_game", "$user$generic0");
            }
            c.r_dx10_sampler("smp_rtlinear");
            c.r_end();
        }

        R4InternalBlenderKind::Hdr10BloomBlur => {
            c.r_pass_default("stub_screen_space", "hdr10_bloom_blur", false, false, false);
            match elem {
                0 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres0"),
                1 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres1"),
                _ => {}
            }
            c.r_dx10_sampler("smp_rtlinear");
            c.r_end();
        }

        R4InternalBlenderKind::Hdr10BloomUpsample => {
            c.r_pass_default("stub_screen_space", "hdr10_bloom_upsample", false, false, false);
            match elem {
                0 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres0"),
                1 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres1"),
                _ => {}
            }
            c.r_dx10_texture("s_hdr10_game", "$user$generic0");
            c.r_dx10_sampler("smp_rtlinear");
            c.r_end();
        }

        R4InternalBlenderKind::Hdr10LensFlareDownsample => {
            c.r_pass_default(
                "stub_screen_space",
                "hdr10_lens_flare_downsample",
                false,
                false,
                false,
            );
            if elem == 0 {
                c.r_dx10_texture("s_hdr10_game", "$user$generic0");
            }
            c.r_dx10_sampler("smp_rtlinear");
            c.r_end();
        }

        R4InternalBlenderKind::Hdr10LensFlareFgen => {
            c.r_pass_default(
                "stub_screen_space",
                "hdr10_lens_flare_fgen",
                false,
                false,
                false,
            );
            match elem {
                0 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres0"),
                1 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres1"),
                _ => {}
            }
            c.r_dx10_sampler("smp_rtlinear");
            c.r_end();
        }

        R4InternalBlenderKind::Hdr10LensFlareBlur => {
            c.r_pass_default(
                "stub_screen_space",
                "hdr10_lens_flare_blur",
                false,
                false,
                false,
            );
            match elem {
                0 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres0"),
                1 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres1"),
                _ => {}
            }
            c.r_dx10_sampler("smp_rtlinear");
            c.r_end();
        }

        R4InternalBlenderKind::Hdr10LensFlareUpsample => {
            c.r_pass_default(
                "stub_screen_space",
                "hdr10_lens_flare_upsample",
                false,
                false,
                false,
            );
            match elem {
                0 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres0"),
                1 => c.r_dx10_texture("s_hdr10_halfres", "$user$hdr10_halfres1"),
                _ => {}
            }
            c.r_dx10_texture("s_hdr10_game", "$user$generic0");
            c.r_dx10_sampler("smp_rtlinear");
            c.r_end();
        }

        R4InternalBlenderKind::Luminance => match elem {
            0 => {
                c.r_pass_default("stub_notransform_build", "bloom_luminance_1", false, false, false);
                c.r_dx10_texture("s_image", "$user$bloom1");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            1 => {
                c.r_pass_default("stub_notransform_filter", "bloom_luminance_2", false, false, false);
                c.r_dx10_texture("s_image", "$user$lum_t64");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            2 => {
                c.r_pass_default("stub_notransform_filter", "bloom_luminance_3", false, false, false);
                c.r_dx10_texture("s_image", "$user$lum_t8");
                c.r_dx10_texture("s_tonemap", "$user$tonemap_src");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_dx10_sampler("smp_nofilter");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::Lut => {
            c.r_pass_default("stub_screen_space", "pp_lut", false, false, false);
            c.r_dx10_texture("s_image", "$user$generic0");
            c.r_dx10_texture("debug_noise", "fx\\blue_noise");
            c.r_dx10_texture("s_lut_atlas", "shaders\\lut_atlas");
            c.r_dx10_sampler("smp_base");
            c.r_dx10_sampler("smp_nofilter");
            c.r_dx10_sampler("smp_rtlinear");
            c.r_dx10_sampler("smp_linear");
            c.r_end();
        }

        R4InternalBlenderKind::Nightvision => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "copy_nomsaa", false, false, false);
                c.r_dx10_texture("s_generic", "$user$generic0");
                add_nvg_samplers(&mut c);
                c.r_end();
            }
            1..=3 => {
                let ps = match elem {
                    1 => "nightvision_gen_1",
                    2 => "nightvision_gen_2",
                    _ => "nightvision_gen_3",
                };
                c.r_pass_default("stub_screen_space", ps, false, false, false);
                add_nvg_common(&mut c);
                c.r_dx10_texture("s_heat", "$user$heat");
                add_nvg_samplers(&mut c);
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::Fakescope => {
            c.r_pass_default("stub_screen_space", "fakescope", false, false, false);
            add_nvg_common(&mut c);
            c.r_dx10_texture("s_scope", "$user$scopeRT");
            add_nvg_samplers(&mut c);
            c.r_end();
        }

        R4InternalBlenderKind::Heatvision => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "copy_nomsaa", false, false, false);
                c.r_dx10_texture("s_generic", "$user$generic0");
                add_nvg_samplers(&mut c);
                c.r_end();
            }
            1 => {
                c.r_pass_default("stub_screen_space", "heatvision", false, false, false);
                add_nvg_common(&mut c);
                c.r_dx10_texture("s_heat", "$user$heat");
                add_nvg_samplers(&mut c);
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::PpBloom => {
            if elem == 0 {
                c.r_pass_default("stub_screen_space", "pp_bloom", false, false, false);
                c.r_dx10_texture("s_image", "$user$generic0");
                c.r_dx10_texture("s_blur_2", "$user$blur_2");
                c.r_dx10_texture("s_blur_4", "$user$blur_4");
                c.r_dx10_texture("s_blur_8", "$user$blur_8");
                add_nvg_samplers(&mut c);
                c.r_end();
            }
        }

        R4InternalBlenderKind::Smaa => match elem {
            0 => {
                c.r_pass_default("pp_smaa_ed", "pp_smaa_ed", false, false, false);
                c.r_dx10_texture("s_image", "$user$generic0");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            1 => {
                c.r_pass_default("pp_smaa_bc", "pp_smaa_bc", false, false, false);
                c.r_dx10_texture("s_image", "$user$generic0");
                c.r_dx10_texture("s_edgetex", "$user$smaa_edgetex");
                c.r_dx10_texture("s_areatex", "shaders\\smaa\\area_tex_dx11");
                c.r_dx10_texture("s_searchtex", "shaders\\smaa\\search_tex");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            2 => {
                c.r_pass_default("pp_smaa_nb", "pp_smaa_nb", false, false, false);
                c.r_dx10_texture("s_image", "$user$generic0");
                c.r_dx10_texture("s_blendtex", "$user$smaa_blendtex");
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::SsfxTaa => match elem {
            0 => {
                c.r_pass_default("stub_screen_space", "ssfx_taa_prepare", false, false, false);
                c.r_dx10_texture("s_ssfx_taa", "$user$ssfx_taa");
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                add_taa_samplers(&mut c);
                c.r_end();
            }
            1 => {
                c.r_pass_default("stub_screen_space", "ssfx_taa", false, false, false);
                c.r_dx10_texture("s_current", "$user$generic0");
                c.r_dx10_texture("s_previous", "$user$ssfx_prev_frame");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_taa");
                c.r_dx10_texture("s_mv", "$user$ssfx_motion_vectors");
                c.r_dx10_texture("s_depth", "$user$position");
                c.r_dx10_texture("s_previous_depth", "$user$ssfx_prev_p");
                c.r_dx10_texture("s_ssfx_temp", "$user$ssfx_temp");
                add_taa_samplers(&mut c);
                c.r_end();
            }
            2 => {
                c.r_pass_default("stub_screen_space", "ssfx_taa_sharp", false, false, false);
                c.r_dx10_texture("s_current", "$user$ssfx_prev_frame");
                c.r_dx10_texture("s_motion_vectors", "$user$ssfx_motion_vectors");
                add_taa_samplers(&mut c);
                c.r_end();
            }
            _ => {}
        },

        R4InternalBlenderKind::Sunshafts => match elem {
            0 => {
                c.r_pass_default(
                    "ssss_notransform",
                    "ogse_sunshafts_mask",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_image", "$user$generic0");
                add_nvg_samplers(&mut c);
                c.r_end();
            }
            1 | 2 => {
                c.r_pass_default(
                    "ssss_notransform",
                    "ogse_sunshafts_blur",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture(
                    "s_sun_shafts",
                    if elem == 1 { "$user$sun_shafts0" } else { "$user$sun_shafts1" },
                );
                add_nvg_samplers(&mut c);
                c.r_end();
            }
            3 => {
                c.r_pass_default(
                    "ssss_notransform",
                    "ogse_sunshafts_blur",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture("s_sun_shafts", "$user$sun_shafts0");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            4 => {
                c.r_pass_default(
                    "ssss_notransform",
                    "ogse_sunshafts_final",
                    false,
                    false,
                    false,
                );
                c.r_dx10_texture("s_position", "$user$position");
                c.r_dx10_texture("s_sun_shafts", "$user$sun_shafts1");
                c.r_dx10_texture("s_image", "$user$generic0");
                c.jitter();
                c.r_dx10_sampler("smp_nofilter");
                c.r_dx10_sampler("smp_rtlinear");
                c.r_end();
            }
            _ => {}
        },
    }

    c.plan
}

/// UI vertex stream format (`dxUIRender::ePointType` in `dxUIRender.cpp:207-244`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiPointType {
    #[default]
    None,
    Lit,
    Tl,
}

/// UI primitive topology (`dxUIRender::ePrimitiveType` in `dxUIRender.cpp:273-294`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UiPrimitiveType {
    #[default]
    None,
    TriList,
    TriStrip,
    LineList,
    LineStrip,
}

/// Resolve the effective UI shader name, switching to `"hud\\movie"` when an `.ogm` video
/// companion exists (`dxUIRender::UpdateShaderName` in `dxUIRender.cpp:178-188`).
#[inline]
pub fn resolve_ui_shader_name(has_ogm_video: bool, sh_name: &str) -> &str {
    if has_ogm_video {
        "hud\\movie"
    } else {
        sh_name
    }
}

/// Compute primitive count from pushed UI vertices matching `dxUIRender::FlushPrimitive`
/// (`dxUIRender.cpp:273-294`).
#[inline]
pub const fn ui_primitive_count(prim_type: UiPrimitiveType, vertex_count: usize) -> usize {
    match prim_type {
        UiPrimitiveType::TriStrip => vertex_count.saturating_sub(2),
        UiPrimitiveType::TriList => vertex_count / 3,
        UiPrimitiveType::LineStrip => vertex_count.saturating_sub(1),
        UiPrimitiveType::LineList => vertex_count / 2,
        UiPrimitiveType::None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkWriter;
    use crate::ogf::XRAY_IDENTITY_MATRIX;

    fn write_matrix(buf: &mut Vec<u8>, m: &XRayMatrix) {
        for row in m {
            for cell in row {
                buf.extend_from_slice(&cell.to_le_bytes());
            }
        }
    }

    fn write_waveform(buf: &mut Vec<u8>, func: u32, arg: [f32; 4]) {
        buf.extend_from_slice(&func.to_le_bytes());
        for a in arg {
            buf.extend_from_slice(&a.to_le_bytes());
        }
    }

    #[test]
    fn parse_shaders_xrlc_and_shaders_xr() {
        // shaders_xrlc.xr (1 entry = 144 bytes)
        let mut xrlc = [0u8; 144];
        xrlc[..7].copy_from_slice(b"default");
        xrlc[128..132].copy_from_slice(
            &(shader_xrlc_flags::COLLISION | shader_xrlc_flags::RENDERING).to_le_bytes(),
        );
        xrlc[132..136].copy_from_slice(&0.25f32.to_le_bytes());
        xrlc[136..140].copy_from_slice(&0.1f32.to_le_bytes());
        xrlc[140..144].copy_from_slice(&2.0f32.to_le_bytes());

        let lib_xrlc = ShaderXrlcLibrary::parse(&xrlc).unwrap();
        assert_eq!(lib_xrlc.find("default").unwrap().lm_density, 2.0);

        // shaders.xr: constant, matrix, and blender chunk
        let mut c0 = Vec::new();
        c0.extend_from_slice(b"const_glow\0");
        write_waveform(&mut c0, 1, [0.5, 0.5, 0.0, 1.0]); // SIN: 0.5 + 0.5*sin(2pi*t)
        for _ in 0..3 {
            write_waveform(&mut c0, 0, [1.0, 0.0, 0.0, 1.0]);
        }

        let mut c1 = Vec::new();
        c1.extend_from_slice(b"mat_scroll\0");
        c1.extend_from_slice(&1u32.to_le_bytes());
        c1.extend_from_slice(&0u32.to_le_bytes());
        for _ in 0..5 {
            write_waveform(&mut c1, 0, [0.0, 0.0, 0.0, 1.0]);
        }

        let mut b0 = Vec::new();
        let clsid = u64::from_be_bytes(*b"LM_DEF  ");
        b0.extend_from_slice(&clsid.to_le_bytes());
        let mut bname = [0u8; 128];
        bname[..14].copy_from_slice(b"effects\\flame0");
        b0.extend_from_slice(&bname);
        b0.extend_from_slice(&[0u8; 32]); // computer
        b0.extend_from_slice(&12345u32.to_le_bytes()); // time
        b0.extend_from_slice(&2u16.to_le_bytes()); // version
        b0.extend_from_slice(&0u16.to_le_bytes()); // pack=4 tail pad
        // Standard IBlender properties
        b0.extend_from_slice(&0u32.to_le_bytes()); // xrPID_MARKER
        b0.extend_from_slice(b"General\0");
        b0.extend_from_slice(&4u32.to_le_bytes()); // xrPID_INTEGER
        b0.extend_from_slice(b"Priority\0");
        b0.extend_from_slice(&2i32.to_le_bytes());
        b0.extend_from_slice(&0i32.to_le_bytes());
        b0.extend_from_slice(&3i32.to_le_bytes());
        b0.extend_from_slice(&6u32.to_le_bytes()); // xrPID_BOOL
        b0.extend_from_slice(b"Strict sorting\0");
        b0.extend_from_slice(&1u32.to_le_bytes());
        b0.extend_from_slice(&3u32.to_le_bytes()); // xrPID_TEXTURE
        b0.extend_from_slice(b"Name\0");
        let mut tex = [0u8; 64];
        tex[..9].copy_from_slice(b"fx\\fx_ash");
        b0.extend_from_slice(&tex);

        let mut sub_cw = ChunkWriter::new();
        sub_cw.write_chunk(0, &b0);

        let mut root_cw = ChunkWriter::new();
        root_cw.write_chunk(0, &c0);
        root_cw.write_chunk(1, &c1);
        root_cw.write_chunk(2, &sub_cw.into_bytes());

        let sh_lib = EngineShaderLibrary::parse(&root_cw.into_bytes()).unwrap();
        let cg = sh_lib.find_constant("const_glow").unwrap();
        assert!((cg.evaluate(0.25)[0] - 1.0).abs() < 1e-5);
        assert_eq!(sh_lib.find_matrix("mat_scroll").unwrap().mode, 1);

        let bl = sh_lib.find_blender("effects\\flame0").unwrap();
        assert_eq!(bl.desc.clsid_str, "LM_DEF");
        assert_eq!(bl.priority, 2);
        assert!(bl.strict_sorting);
        assert_eq!(bl.base_texture, "fx\\fx_ash");
        assert!(bl.raw_tail.is_empty());

        assert!(EngineShaderLibrary::parse(b"shENGINE1234").is_err());
    }

    #[test]
    fn parse_env_mod_ltx_ps_static_and_fog_vol() {
        let mut em = Vec::new();
        for f in [0.0f32, 0.0, 0.0, 10.0, 2.0, 100.0] {
            em.extend_from_slice(&f.to_le_bytes());
        }
        for f in [0.2f32, 0.3, 0.4, 0.5] {
            em.extend_from_slice(&f.to_le_bytes());
        }
        for f in [0.1f32, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3] {
            em.extend_from_slice(&f.to_le_bytes());
        }
        em.extend_from_slice(&(env_mod_flags::FOG_COLOR | env_mod_flags::FOG_DENSITY).to_le_bytes());

        let mut cw = ChunkWriter::new();
        cw.write_chunk(0, &0x0016u32.to_le_bytes());
        cw.write_chunk(1, &em);
        let mods = LevelEnvModifiers::parse(&cw.into_bytes()).unwrap();
        let accum = mods.evaluate_at([5.0, 0.0, 0.0]); // dist=5 => att=0.5 => power=1.0
        assert!((accum.fog_density - 0.5).abs() < 1e-5);
        assert_eq!(
            accum.active_flags,
            env_mod_flags::FOG_COLOR | env_mod_flags::FOG_DENSITY
        );
        assert!((accum.ambient[0] - 0.0).abs() < 1e-6);

        let ltx = LtxDocument::parse(
            "[zone_tunnel]\nposition = 0,0,0\nradius = 20\npower = 1.0\nfog_density = 0.8\n",
        );
        let from_ltx = LevelEnvModifiers::from_ltx(&ltx);
        assert_eq!(from_ltx.modifiers.len(), 1);
        assert!((from_ltx.evaluate_at([0.0, 0.0, 0.0]).fog_density - 0.8).abs() < 1e-5);

        // ps_static
        let mut ps_entry = Vec::new();
        ps_entry.extend_from_slice(&0x0003u16.to_le_bytes());
        ps_entry.extend_from_slice(b"anomaly\\fire_camp\0");
        write_matrix(&mut ps_entry, &XRAY_IDENTITY_MATRIX);
        let mut ps_cw = ChunkWriter::new();
        ps_cw.write_chunk(0, &1u32.to_le_bytes());
        ps_cw.write_chunk(1, &ps_entry);
        let ps = LevelStaticParticles::parse(&ps_cw.into_bytes()).unwrap();
        assert_eq!(ps.particles[0].ref_name, "anomaly\\fire_camp");

        // fog_vol v3
        let mut fv = Vec::new();
        fv.extend_from_slice(&3u16.to_le_bytes());
        fv.extend_from_slice(&1u32.to_le_bytes());
        fv.extend_from_slice(b"swamp_fog\r\n");
        write_matrix(&mut fv, &XRAY_IDENTITY_MATRIX);
        fv.extend_from_slice(&1u32.to_le_bytes());
        write_matrix(&mut fv, &XRAY_IDENTITY_MATRIX);
        let fogs = LevelFogVolumes::parse(&fv).unwrap();
        assert_eq!(fogs.version, 3);
        assert_eq!(fogs.volumes[0].profile.as_deref(), Some("swamp_fog"));
        assert_eq!(fogs.volumes[0].obstacles.len(), 1);
    }

    #[test]
    fn evaluate_shader_matrix_modes() {
        let wf = |val: f32| WaveForm {
            func: WaveFunction::Constant,
            arg: [val, 0.0, 0.0, 1.0],
        };

        // 1. TCM Scale + Scroll
        let mat_scale_scroll = ShaderMatrix {
            name: "water_flow".to_string(),
            mode: matrix_mode::TCM,
            tcm: tcm_flags::SCALE | tcm_flags::SCROLL,
            transforms: [wf(2.0), wf(3.0), wf(0.0), wf(0.25), wf(-0.5)],
        };
        let m = mat_scale_scroll.evaluate(2.0, &XRAY_IDENTITY_MATRIX);
        // u_scroll = 0.25 * 2 * 2 = 1.0, v_scroll = -0.5 * 2 * 3 = -3.0
        let uv = transform_uv(&m, [0.5, 0.25]);
        assert!((uv[0] - (0.5 * 2.0 + 1.0)).abs() < 1e-5);
        assert!((uv[1] - (0.25 * 3.0 - 3.0)).abs() < 1e-5);

        // 2. TCM Rotate (90 deg at t = 1.0)
        let mat_rot = ShaderMatrix {
            name: "rot90".to_string(),
            mode: matrix_mode::TCM,
            tcm: tcm_flags::ROTATE,
            transforms: [wf(1.0), wf(1.0), wf(std::f32::consts::FRAC_PI_2), wf(0.0), wf(0.0)],
        };
        let mr = mat_rot.evaluate(1.0, &XRAY_IDENTITY_MATRIX);
        let uvr = transform_uv(&mr, [1.0, 0.0]);
        assert!((uvr[0] - 0.0).abs() < 1e-5);
        assert!((uvr[1] - 1.0).abs() < 1e-5);

        // 3. S_REFL spherical projection from view matrix
        let mut view = crate::ogf::matrix_set_hpb(0.0, 0.0, 0.0);
        view[3][0] = 12.0;
        view[3][1] = -4.0;
        view[3][2] = 7.0;
        let s_refl = ShaderMatrix {
            name: "$s_refl".to_string(),
            mode: matrix_mode::S_REFL,
            tcm: 0,
            transforms: [wf(0.0); 5],
        };
        let ms = s_refl.evaluate(0.0, &view);
        assert!((ms[0][0] - 0.5).abs() < 1e-6);
        assert!((ms[1][1] - (-0.5)).abs() < 1e-6);
        assert!((ms[3][0] - 0.5).abs() < 1e-6);
        assert!((ms[3][1] - 0.5).abs() < 1e-6);
        assert_eq!(ms[3][3], 0.0);

        // 4. C_REFL view-rotation inverse without translation
        let view_rot = crate::ogf::matrix_set_hpb(0.4, -0.2, 0.1);
        let mut view_with_trans = view_rot;
        view_with_trans[3][0] = 99.0;
        view_with_trans[3][1] = -42.0;
        view_with_trans[3][2] = 17.0;
        let c_refl = ShaderMatrix {
            name: "$c_refl".to_string(),
            mode: matrix_mode::C_REFL,
            tcm: 0,
            transforms: [wf(0.0); 5],
        };
        let mc = c_refl.evaluate(0.0, &view_with_trans);
        assert_eq!([mc[3][0], mc[3][1], mc[3][2]], [0.0, 0.0, 0.0]);
        let prod = matrix_mul_43(&view_rot, &mc);
        for r in 0..4 {
            for c in 0..4 {
                assert!((prod[r][c] - XRAY_IDENTITY_MATRIX[r][c]).abs() < 1e-5);
            }
        }

        // 5. PROGRAMMABLE preserves previous matrix
        let custom = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let prog = ShaderMatrix {
            name: "$prog".to_string(),
            mode: matrix_mode::PROGRAMMABLE,
            tcm: 0,
            transforms: [wf(0.0); 5],
        };
        assert_eq!(prog.evaluate_with_prev(custom, 3.0, &view_with_trans), custom);
    }

    #[test]
    fn test_env_mods_ltx_explicit_use_flags_hex_and_decimal() {
        use xray_core::ltx_loader::LoadedLtxSource;

        let text = r#"
[mod_fog_only]
position = 10, 0, 10
radius = 50
power = 2.0
far_plane = 300
fog_color = 0.25, 0.35, 0.45
fog_density = 0.9
ambient = 0.4, 0.4, 0.4
sky_color = 0.5, 0.5, 0.5
hemi_color = 0.6, 0.6, 0.6
use_flags = 0x02

[mod_all_hex]
position = 200, 0, 0
radius = 10
power = 1.0
use_flags = 0X3F

[mod_decimal]
position = 500, 0, 0
radius = 25
power = 1.0
far_plane = 120
fog_density = 0.65
ambient = 0.8, 0.8, 0.8
use_flags = 5
"#;
        let borrowed = LtxDocument::parse(text);
        let owned = OwnedLtxDocument::from_sources(&[LoadedLtxSource {
            virtual_path: "configs\\environment\\env_mods\\test.ltx".to_string(),
            depth: 0,
            text: text.to_string(),
        }]);

        let parsed_b = LevelEnvModifiers::from_ltx(&borrowed);
        let parsed_o = LevelEnvModifiers::from_owned_ltx(&owned);
        assert_eq!(parsed_b, parsed_o);

        // Sections sorted alphabetically: mod_all_hex, mod_decimal, mod_fog_only
        assert_eq!(
            parsed_o.modifiers[0].use_flags,
            env_mod_flags::VIEW_DIST
                | env_mod_flags::FOG_COLOR
                | env_mod_flags::FOG_DENSITY
                | env_mod_flags::AMBIENT_COLOR
                | env_mod_flags::SKY_COLOR
                | env_mod_flags::HEMI_COLOR
        );
        assert_eq!(
            parsed_o.modifiers[1].use_flags,
            env_mod_flags::VIEW_DIST | env_mod_flags::FOG_DENSITY
        );
        assert_eq!(parsed_o.modifiers[2].use_flags, env_mod_flags::FOG_COLOR);

        // At [10, 0, 10], even though fog_density/ambient/etc. have values, only FOG_COLOR activates!
        let accum = parsed_o.evaluate_at([10.0, 0.0, 10.0]);
        assert_eq!(accum.active_flags, env_mod_flags::FOG_COLOR);
        assert!((accum.fog_color[0] - 0.5).abs() < 1e-5);
        assert_eq!(accum.fog_density, 0.0);
        assert_eq!(accum.far_plane, 0.0);
        assert_eq!(accum.ambient, [0.0; 3]);
    }

    #[test]
    fn r4_uber_deffer_uber_shadow_and_blenders_match_cpp() {
        use crate::thm::{texture_flags, TextureBumpMode, TextureParams, TextureType};

        let mut thm_tbl = TextureDescrTable::new();
        let wall_thm = TextureParams {
            texture_type: TextureType::Image,
            flags: texture_flags::FL_DIFFUSE_DETAIL | texture_flags::FL_BUMP_DETAIL,
            detail_name: "detail\\detail_brick".to_string(),
            detail_scale: 3.0,
            bump_mode: TextureBumpMode::UseParallax,
            bump_name: "wall\\brick_bump".to_string(),
            ..TextureParams::default()
        };
        let det_thm = TextureParams {
            texture_type: TextureType::Image,
            bump_mode: TextureBumpMode::Use,
            bump_name: "detail\\detail_brick_bump".to_string(),
            ..TextureParams::default()
        };
        thm_tbl.insert_thm("wall\\brick", &wall_thm);
        thm_tbl.insert_thm("detail\\detail_brick", &det_thm);

        // 1. Flat / non-bump deffer without lmap
        let flat_pass = plan_uber_deffer(
            &UberDefferInput {
                hq: true,
                vspec: "base",
                pspec: "base",
                aref: false,
                l_textures: &["crete\\plain.dds"],
                ..UberDefferInput::default()
            },
            &thm_tbl,
        );
        assert_eq!(flat_pass.vs, "deffer_base_flat");
        assert_eq!(flat_pass.ps, "deffer_base_flat");
        assert_eq!(flat_pass.textures[0].texture_name, "crete\\plain");

        // 2. Bump + Steep Parallax + LMap + Detail Bump in HQ
        let steep_pass = plan_uber_deffer(
            &UberDefferInput {
                hq: true,
                vspec: "base",
                pspec: "base",
                aref: false,
                l_textures: &["wall\\brick.dds", "lmap#1_1", "lmap#1_2"],
                base_detail_texture: "detail\\detail_brick",
                detail_diffuse: true,
                detail_bump: true,
                use_steep_parallax: true,
                ..UberDefferInput::default()
            },
            &thm_tbl,
        );
        assert_eq!(steep_pass.vs, "deffer_base_lmh_bump_d-hq");
        assert_eq!(steep_pass.ps, "deffer_base_lmh_steep_db-hq");
        assert!(steep_pass
            .textures
            .iter()
            .any(|t| t.resource_name == "s_bumpX" && t.texture_name == "wall\\brick_bump#"));
        assert!(steep_pass
            .textures
            .iter()
            .any(|t| t.resource_name == "s_detailBumpX" && t.texture_name == "detail\\detail_brick_bump#"));
        assert!(steep_pass
            .textures
            .iter()
            .any(|t| t.resource_name == "s_hemi" && t.texture_name == "lmap#1_2"));

        // 3. DX11 Tessellation (PN + HM + LM_HEMI + TDETAIL + TDETAIL_BUMP)
        let tess_pass = plan_uber_deffer(
            &UberDefferInput {
                hq: true,
                vspec: "base",
                pspec: "base",
                aref: true,
                l_textures: &["wall\\brick", "lmap#1_1", "lmap#1_2"],
                base_detail_texture: "detail\\detail_brick",
                detail_diffuse: true,
                detail_bump: true,
                tess_method: TessMethod::TessPnHm,
                dx11_enable_tessellation: true,
                wireframe: true,
                ..UberDefferInput::default()
            },
            &thm_tbl,
        );
        let expected_params = "(TESS_PN,TESS_HM,USE_LM_HEMI,USE_TDETAIL,USE_TDETAIL_BUMP,)";
        assert_eq!(tess_pass.vs, format!("deffer_base_bump{expected_params}"));
        assert_eq!(tess_pass.ps, format!("deffer_base_aref_bump{expected_params}"));
        assert_eq!(tess_pass.hs, format!("DX11\\tess{expected_params}"));
        assert_eq!(tess_pass.ds, format!("DX11\\tess{expected_params}"));
        assert!(tess_pass.wireframe);
        assert_eq!(tess_pass.shader_options.len(), 5);
        assert!(tess_pass
            .textures
            .iter()
            .any(|t| t.resource_name == "s_tdetailBumpX"));
        assert!(tess_pass
            .samplers
            .iter()
            .any(|s| s.resource_name == "smp_bump_ds" && s.anisotropic_override == Some(true)));

        let sh_tess = plan_uber_shadow(
            &UberShadowInput {
                vspec: "base",
                l_textures: &["wall\\brick"],
                detail_diffuse: false,
                detail_bump: false,
                tess_method: TessMethod::TessPn,
                dx11_enable_tessellation: true,
                wireframe: false,
                ..UberShadowInput::default()
            },
            &thm_tbl,
        );
        assert_eq!(sh_tess.vs, "deffer_base_bump(TESS_PN,)");
        assert_eq!(sh_tess.ds, "DX11\\tess_shadow(TESS_PN,)");
        assert_eq!(sh_tess.ps, "dumb");

        // 4. ATOC 2-pass deffer_aref: Pass 0 ATOC depth-only, Pass 1 main with ZWRITE=false & ZFUNC=Equal
        let aref_spec = R4BlenderSpec::DefferAref {
            priority: 1,
            strict_sorting: false,
            base_texture: "$base0".to_string(),
            lmapped: true,
            alpha_ref: 200,
            alpha_blend: false,
        };
        let aref_plan = plan_r4_blender_compile(
            &aref_spec,
            &["crete\\fence"],
            &R4CompileConfig {
                element: DefferElement::NormalHq,
                msaa_alphatest: Dx10MsaaAlphaTest::Dx100Atoc,
                ..R4CompileConfig::default()
            },
            &thm_tbl,
        );
        assert_eq!(aref_plan.passes.len(), 2);
        assert_eq!(aref_plan.passes[0].ps, "deffer_base_atoc_aref_flat");
        assert!(aref_plan.passes[0].blend.alpha_to_coverage);
        assert_eq!(aref_plan.passes[0].blend.color_write_enable, [false; 4]);
        assert!(aref_plan.passes[0].z_write);
        assert_eq!(aref_plan.passes[1].ps, "deffer_base_aref_flat");
        assert!(!aref_plan.passes[1].z_write);
        assert_eq!(aref_plan.passes[1].z_func, CmpFunc::Equal);
        assert!(aref_plan.passes[1].stencil.enable);
        assert_eq!(aref_plan.passes[1].stencil.ref_value, 1);

        // 5. SSFX BmmD Terrain (prepass + terrain_high + height textures + terrain_mid _lod_texture + shadow_direct_terrain)
        let bmmd = R4BlenderSpec::BmmD {
            priority: 1,
            strict_sorting: false,
            base_texture: "$base0".to_string(),
            detail_texture: "$null".to_string(),
            r_detail: "detail\\grnd_grass".to_string(),
            g_detail: "detail\\grnd_asphalt".to_string(),
            b_detail: "detail\\grnd_earth".to_string(),
            a_detail: "detail\\grnd_yantar".to_string(),
        };
        let hq_terrain = plan_r4_blender_compile(
            &bmmd,
            &["terrain\\terrain_escape", "lmap#1_1", "lmap#1_2"],
            &R4CompileConfig {
                element: DefferElement::NormalHq,
                terrain_prepass: true,
                ssfx_terrain: true,
                allow_detail: false,
                ..R4CompileConfig::default()
            },
            &thm_tbl,
        );
        assert!(hq_terrain.is_landscape_prepass);
        assert!(hq_terrain.is_landscape);
        assert_eq!(hq_terrain.passes.len(), 2);
        assert_eq!(hq_terrain.passes[0].vs, "shadow_direct_base");
        assert_eq!(hq_terrain.passes[0].blend.color_write_enable, [false; 4]);
        assert_eq!(hq_terrain.passes[1].vs, "deffer_terrain_lmh_flat");
        assert_eq!(hq_terrain.passes[1].ps, "deffer_terrain_high_lmh_flat");
        assert_eq!(hq_terrain.passes[1].z_func, CmpFunc::Equal);
        assert!(!hq_terrain.passes[1].z_write);
        assert!(hq_terrain.passes[1]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_height_r" && t.texture_name == "detail\\grnd_grass_height"));

        let lq_terrain = plan_r4_blender_compile(
            &bmmd,
            &["terrain\\terrain_escape", "lmap#1_1"],
            &R4CompileConfig {
                element: DefferElement::NormalLq,
                ssfx_terrain: true,
                terrain_has_lod_textures_dds: false,
                allow_detail: false,
                ..R4CompileConfig::default()
            },
            &thm_tbl,
        );
        assert_eq!(lq_terrain.passes[0].ps, "deffer_terrain_mid_flat");
        assert!(lq_terrain.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_lod_texture" && t.texture_name == "terrain\\default_lod_textures"));

        let sh_terrain = plan_r4_blender_compile(
            &bmmd,
            &["terrain\\terrain_escape"],
            &R4CompileConfig {
                element: DefferElement::Shadow,
                ssfx_terrain: true,
                ..R4CompileConfig::default()
            },
            &thm_tbl,
        );
        assert_eq!(sh_terrain.passes[0].vs, "shadow_direct_terrain");

        // 6. SSFX Tree Branches & ModelEbB Glass / HUD Rain
        let tree_plan = plan_r4_blender_compile(
            &R4BlenderSpec::Tree {
                priority: 1,
                strict_sorting: false,
                base_texture: "$base0".to_string(),
                alpha_blend: true,
                not_an_tree: false,
            },
            &["trees\\leaf"],
            &R4CompileConfig {
                element: DefferElement::NormalHq,
                ssfx_branches: true,
                allow_detail: false,
                ..R4CompileConfig::default()
            },
            &thm_tbl,
        );
        assert_eq!(tree_plan.passes[0].vs, "deffer_tree_branch_flat");
        assert_eq!(tree_plan.passes[0].ps, "deffer_tree_branch_aref_flat");
        assert!(tree_plan.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_waves" && t.texture_name == "fx\\wind_wave"));

        let glass_plan = plan_r4_blender_compile(
            &R4BlenderSpec::ModelEbB {
                priority: 2,
                strict_sorting: true,
                base_texture: "$base0".to_string(),
                env_texture: "sky\\cubemap".to_string(),
                alpha_blend: true,
            },
            &["wpn\\scope_lens"],
            &R4CompileConfig {
                element: DefferElement::NormalHq,
                ssfx_glass: true,
                hud_element: false,
                ..R4CompileConfig::default()
            },
            &thm_tbl,
        );
        assert_eq!(glass_plan.passes[0].vs, "ssfx_glass");
        assert!(glass_plan.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_glass" && t.texture_name == "fx\\glass_normal"));

        let hud_plan = plan_r4_blender_compile(
            &R4BlenderSpec::DefferModel {
                priority: 1,
                strict_sorting: false,
                base_texture: "$base0".to_string(),
                alpha_blend: false,
                alpha_ref: 32,
                tessellation: TessMethod::NoTess,
            },
            &["wpn\\ak74"],
            &R4CompileConfig {
                element: DefferElement::NormalHq,
                hud_element: true,
                allow_detail: false,
                ..R4CompileConfig::default()
            },
            &thm_tbl,
        );
        assert_eq!(hud_plan.passes[0].vs, "deffer_model_hud_flat");
        assert_eq!(hud_plan.passes[0].ps, "deffer_base_hud_flat");
        assert!(hud_plan.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_hud_rain" && t.texture_name == "fx\\hud_rain"));

        // 7. Particle, ScreenSet & dxUIRender helpers
        let part_plan = plan_r4_blender_compile(
            &R4BlenderSpec::Particle {
                priority: 1,
                strict_sorting: false,
                base_texture: "$base0".to_string(),
                blend_mode: 1,
                clamp: true,
            },
            &["pfx\\smoke"],
            &R4CompileConfig::default(),
            &thm_tbl,
        );
        assert_eq!(part_plan.passes[0].ps, "particle");
        assert_eq!(part_plan.passes[0].samplers[0].address_mode, TexAddressMode::Clamp);
        assert!(part_plan.passes[0].blend.enable);
        assert!(!part_plan.passes[0].z_write);

        let screen_plan = plan_r4_blender_compile(
            &R4BlenderSpec::ScreenSet {
                priority: 1,
                strict_sorting: false,
                base_texture: "$base0".to_string(),
                blend_mode: 8,
                clamp: true,
                alpha_ref: 16,
                z_test: false,
                z_write: false,
                lighting: false,
                fog: false,
            },
            &["ui\\font"],
            &R4CompileConfig::default(),
            &thm_tbl,
        );
        assert_eq!(screen_plan.passes[0].vs, "stub_notransform_t_m2");
        assert_eq!(screen_plan.passes[0].blend.alpha_ref, 16);

        assert_eq!(BlenderClsid::from_u64(BlenderClsid::Default.to_u64()), Some(BlenderClsid::Default));
        let raw_bmmd = BlenderShader {
            desc: BlenderDesc {
                clsid: BlenderClsid::BmmD.to_u64(),
                clsid_str: "BmmDold".to_string(),
                name: "levels\\l01_escape\\terrain".to_string(),
                computer: String::new(),
                time: 0,
                version: 3,
            },
            priority: 1,
            strict_sorting: false,
            base_texture: "$base0".to_string(),
            base_transform: "$null".to_string(),
            properties: vec![
                BlenderProperty {
                    name: "Name".to_string(),
                    value: BlenderPropertyValue::Texture("$base0".to_string()),
                },
                BlenderProperty {
                    name: "Name".to_string(),
                    value: BlenderPropertyValue::Texture("terrain\\escape_det".to_string()),
                },
                BlenderProperty {
                    name: "R2-R".to_string(),
                    value: BlenderPropertyValue::Texture("detail\\r_tex".to_string()),
                },
            ],
            raw_tail: Vec::new(),
        };
        assert_eq!(
            R4BlenderSpec::from_blender_shader(&raw_bmmd),
            Some(R4BlenderSpec::BmmD {
                priority: 1,
                strict_sorting: false,
                base_texture: "$base0".to_string(),
                detail_texture: "terrain\\escape_det".to_string(),
                r_detail: "detail\\r_tex".to_string(),
                g_detail: "detail\\detail_grnd_asphalt".to_string(),
                b_detail: "detail\\detail_grnd_earth".to_string(),
                a_detail: "detail\\detail_grnd_yantar".to_string(),
            })
        );
        assert_eq!(resolve_ui_shader_name(true, "hud\\default"), "hud\\movie");
        assert_eq!(resolve_ui_shader_name(false, "hud\\default"), "hud\\default");
        assert_eq!(ui_primitive_count(UiPrimitiveType::TriStrip, 6), 4);
        assert_eq!(ui_primitive_count(UiPrimitiveType::TriList, 6), 2);
        assert_eq!(ui_primitive_count(UiPrimitiveType::LineStrip, 5), 4);
        assert_eq!(ui_primitive_count(UiPrimitiveType::LineList, 6), 3);
    }

    #[test]
    fn r4_internal_blenders_match_cpp_passes() {
        let base = |kind: R4InternalBlenderKind, element: u32, dx10_msaa: bool| {
            plan_r4_internal_blender(
                kind,
                &["lights\\lights_spot01"],
                R4InternalBlenderOptions {
                    element,
                    dx10_msaa,
                    ..R4InternalBlenderOptions::default()
                },
            )
        };

        // LightOccq: clear stencil pass switches write mask between MSAA and non-MSAA
        let occq1 = base(R4InternalBlenderKind::LightOccq, 1, false);
        assert_eq!(occq1.passes[0].cull_mode, CullMode::None);
        assert_eq!(occq1.passes[0].stencil.func, CmpFunc::LessEqual);
        assert_eq!(occq1.passes[0].blend.color_write_enable, [false; 4]);
        let occq2_no_msaa = base(R4InternalBlenderKind::LightOccq, 2, false);
        let occq2_msaa = base(R4InternalBlenderKind::LightOccq, 2, true);
        assert_eq!(occq2_no_msaa.passes[0].stencil.write_mask, 0xFE);
        assert_eq!(occq2_msaa.passes[0].stencil.write_mask, 0x7E);

        // AccumDirect / AccumDirectMsaa: inverted Z compare, minmax SM, border color
        let sun_near = base(R4InternalBlenderKind::AccumDirect, 0, false);
        assert_eq!(sun_near.passes[0].ps, "accum_sun_near_nomsaa_nominmax");
        assert_eq!(sun_near.passes[0].z_func, CmpFunc::Greater);
        assert!(sun_near.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_smap_minmax"));
        let sun_near_msaa = base(R4InternalBlenderKind::AccumDirectMsaa, 0, true);
        assert_eq!(sun_near_msaa.passes[0].ps, "accum_sun_near_msaa_nominmax");
        assert!(!sun_near_msaa.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_smap_minmax"));

        let sun_far = base(R4InternalBlenderKind::AccumDirect, 2, false);
        assert_eq!(sun_far.passes[0].ps, "accum_sun_far_nomsaa");
        let smp_smap = sun_far.passes[0]
            .samplers
            .iter()
            .find(|s| s.resource_name == "smp_smap")
            .unwrap();
        assert_eq!(smp_smap.address_mode, TexAddressMode::Border);
        assert_eq!(smp_smap.border_color, Some(0xFFFF_FFFF));

        let sun_minmax_msaa = base(R4InternalBlenderKind::AccumDirectMsaa, 4, true);
        assert_eq!(sun_minmax_msaa.passes[0].ps, "accum_sun_near_msaa_minmax");
        assert!(sun_minmax_msaa.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_smap_minmax"));

        // Volumetric sun MSAA vs non-sun MSAA
        let vol_sun = base(R4InternalBlenderKind::AccumDirectVolumetricSunMsaa, 0, true);
        assert!(!vol_sun.passes[0].z_test);
        assert!(vol_sun.passes[0].blend.enable);
        assert_eq!(vol_sun.passes[0].blend.dst, BlendFactor::One);
        assert!(vol_sun.passes[0].textures.iter().any(|t| t.resource_name == "jitter0"));

        // AccumPoint / AccumSpot / AccumReflected FP16 blend vs disabled
        let refl_no_fp16 = plan_r4_internal_blender(
            R4InternalBlenderKind::AccumReflected,
            &[],
            R4InternalBlenderOptions {
                fp16_blend: false,
                ..R4InternalBlenderOptions::default()
            },
        );
        assert!(!refl_no_fp16.passes[0].blend.enable);
        assert_eq!(refl_no_fp16.passes[0].blend.dst, BlendFactor::Zero);
        let spot_full = base(R4InternalBlenderKind::AccumSpotMsaa, 3, true);
        assert_eq!(spot_full.passes[0].ps, "accum_spot_fullsize_msaa");

        // HDAO compute passes
        let hdao = base(R4InternalBlenderKind::HdaoCs, 0, false);
        assert_eq!(hdao.passes[0].cs, "ssao_hdao");
        assert_eq!(hdao.passes[0].vs, "null");
        assert_eq!(hdao.passes[0].ps, "null");
        let hdao_msaa = base(R4InternalBlenderKind::HdaoMsaaCs, 0, true);
        assert_eq!(hdao_msaa.passes[0].cs, "ssao_hdao_msaa");

        // Combine vs CombineMsaa differences
        let comb0_no = base(R4InternalBlenderKind::Combine, 0, false);
        let comb0_msaa = base(R4InternalBlenderKind::CombineMsaa, 0, true);
        let occ_no = comb0_no.passes[0]
            .textures
            .iter()
            .position(|t| t.resource_name == "s_occ")
            .unwrap();
        let env_no = comb0_no.passes[0]
            .textures
            .iter()
            .position(|t| t.resource_name == "env_s0")
            .unwrap();
        let occ_msaa = comb0_msaa.passes[0]
            .textures
            .iter()
            .position(|t| t.resource_name == "s_occ")
            .unwrap();
        let env_msaa = comb0_msaa.passes[0]
            .textures
            .iter()
            .position(|t| t.resource_name == "env_s0")
            .unwrap();
        assert!(occ_no > env_no);
        assert!(occ_msaa < env_msaa);

        let comb2_no = base(R4InternalBlenderKind::Combine, 2, false);
        let comb2_msaa = base(R4InternalBlenderKind::CombineMsaa, 2, true);
        assert!(!comb2_no.passes[0].z_write);
        assert!(comb2_msaa.passes[0].z_write);
        assert!(comb2_msaa.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_distort" && t.texture_name == "$user$generic1_r"));
        let lut_pos_msaa2 = comb2_msaa.passes[0]
            .textures
            .iter()
            .position(|t| t.resource_name == "s_lut_atlas")
            .unwrap();
        let mv_pos_msaa2 = comb2_msaa.passes[0]
            .textures
            .iter()
            .position(|t| t.resource_name == "s_motion_vectors")
            .unwrap();
        assert!(lut_pos_msaa2 < mv_pos_msaa2);

        // Rain vs RainMsaa: s_hud_mask + blend factor override on gloss pass
        let rain1 = base(R4InternalBlenderKind::Rain, 1, false);
        let rain_msaa0 = base(R4InternalBlenderKind::RainMsaa, 0, true);
        assert!(rain1.passes[0].textures.iter().any(|t| t.resource_name == "s_hud_mask"));
        assert!(!rain_msaa0.passes[0]
            .textures
            .iter()
            .any(|t| t.resource_name == "s_hud_mask"));
        let rain3 = base(R4InternalBlenderKind::Rain, 3, false);
        assert_eq!(rain3.passes[0].blend.src, BlendFactor::Zero);
        assert_eq!(rain3.passes[0].blend.dst, BlendFactor::SrcColor);

        // Gasmask, SsfxSssExt, SsfxFogScattering, Sunshafts, SMAA, TAA, HDR10
        let gasmask = base(R4InternalBlenderKind::GasmaskDudv, 0, false);
        assert_eq!(
            gasmask.passes[0]
                .textures
                .iter()
                .filter(|t| t.resource_name.starts_with("s_mask_nm_"))
                .count(),
            10
        );
        let sss_ext0 = base(R4InternalBlenderKind::SsfxSssExt, 0, false);
        assert!(sss_ext0.passes[0].samplers.iter().any(|s| s.resource_name == "smp_jitter"));
        assert!(!sss_ext0.passes[0].textures.iter().any(|t| t.resource_name == "jitter0"));
        assert!(base(R4InternalBlenderKind::SsfxFogScattering, 1, false).passes.is_empty());
        assert_eq!(
            base(R4InternalBlenderKind::SsfxFogScattering, 3, false).passes[0].textures[0].texture_name,
            "$user$blur_4"
        );
        let sunshafts4 = base(R4InternalBlenderKind::Sunshafts, 4, false);
        assert_eq!(sunshafts4.passes[0].ps, "ogse_sunshafts_final");
        assert!(sunshafts4.passes[0].textures.iter().any(|t| t.resource_name == "jitter0"));
        assert_eq!(base(R4InternalBlenderKind::Smaa, 1, false).passes[0].ps, "pp_smaa_bc");
        assert_eq!(base(R4InternalBlenderKind::SsfxTaa, 2, false).passes[0].ps, "ssfx_taa_sharp");
        assert_eq!(
            base(R4InternalBlenderKind::Hdr10LensFlareFgen, 1, false).passes[0].textures[0]
                .texture_name,
            "$user$hdr10_halfres1"
        );
        assert_eq!(
            base(R4InternalBlenderKind::Skybox, 1, false).passes[0].ps,
            "ssfx_sky_mv"
        );
    }
}
