//! Pure Rust X-Ray Shaders, Lightmap Compilers, Environment Modifiers, Particles & Fog Volumes
//!
//! Covers:
//! - `shaders_xrlc.xr`: 144-byte `Shader_xrLC` collision/rendering/lightmap compiler entries
//! - `shaders.xr`: `WaveForm`, `CConstant` (chunk 0), `CMatrix` (chunk 1), `IBlender` descriptors + properties (chunk 2)
//! - `level.env_mod` & Anomaly `environment\env_mods\<level>.ltx`: `CEnvModifier` zones + spatial evaluation
//! - `level.ps_static`: Static world particle emitters (`CPGDef` / `CParticleGroup`)
//! - `level.fog_vol`: Volumetric fog & obstacle matrices (`dx103DFluidVolume` / `v2` & `v3`)

use std::f32::consts::PI;

use xray_core::ltx_loader::OwnedLtxDocument;
use xray_core::LtxDocument;

use crate::chunk::{ByteCursor, ChunkReader, FormatError};
use crate::ogf::{matrix_invert_43, matrix_mul_43, XRayMatrix, XRAY_IDENTITY_MATRIX};

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

            let mut use_flags = 0u16;
            if doc.read_str(section, "far_plane").is_some() {
                use_flags |= env_mod_flags::VIEW_DIST;
            }
            if doc.read_str(section, "fog_color").is_some() {
                use_flags |= env_mod_flags::FOG_COLOR;
            }
            if doc.read_str(section, "fog_density").is_some() {
                use_flags |= env_mod_flags::FOG_DENSITY;
            }
            if doc.read_str(section, "ambient").is_some() {
                use_flags |= env_mod_flags::AMBIENT_COLOR;
            }
            if doc.read_str(section, "sky_color").is_some() {
                use_flags |= env_mod_flags::SKY_COLOR;
            }
            if doc.read_str(section, "hemi_color").is_some() {
                use_flags |= env_mod_flags::HEMI_COLOR;
            }
            if use_flags == 0 {
                use_flags = env_mod_flags::ALL;
            }

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

            let mut use_flags = 0u16;
            if doc.read_str(section, "far_plane").is_some() {
                use_flags |= env_mod_flags::VIEW_DIST;
            }
            if doc.read_str(section, "fog_color").is_some() {
                use_flags |= env_mod_flags::FOG_COLOR;
            }
            if doc.read_str(section, "fog_density").is_some() {
                use_flags |= env_mod_flags::FOG_DENSITY;
            }
            if doc.read_str(section, "ambient").is_some() {
                use_flags |= env_mod_flags::AMBIENT_COLOR;
            }
            if doc.read_str(section, "sky_color").is_some() {
                use_flags |= env_mod_flags::SKY_COLOR;
            }
            if doc.read_str(section, "hemi_color").is_some() {
                use_flags |= env_mod_flags::HEMI_COLOR;
            }
            if use_flags == 0 {
                use_flags = env_mod_flags::ALL;
            }

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
}
