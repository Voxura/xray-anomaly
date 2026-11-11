//! Direct3D 12 2D UI Canvas & HUD Immediate-Mode Renderer
//!
//! Replaces legacy X-Ray fixed-function vertex streams (`FVF::F_TL`) with
//! procedural quad instancing directly from `SV_VertexID` in the vertex shader:
//! - Zero vertex buffer and zero index buffer allocations for 2D UI elements
//! - SM6.0 / SM6.6 bindless descriptor indexing for rendering multiple texture atlases in 1 draw call
//! - Resolution-independent virtual coordinate mapping (e.g. 1024x768 base to active backbuffer)
//! - Direct interop with `xray_formats::xml_ui::UiTextureRegion` and `UiWindowDef`

use std::mem::ManuallyDrop;
use windows::core::{HRESULT, Result as WinResult};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use crate::bindless::DescriptorIndex;
use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};

fn failure() -> windows::core::Error {
    windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

/// 64-byte GPU UI quad instance record aligned to 16 bytes
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(16))]
pub struct GpuUiQuad {
    /// Screen-space or virtual-space position and size `[x, y, width, height]`
    pub rect: [f32; 4],
    /// Normalized texture coordinates `[u1, v1, u2, v2]`
    pub uv: [f32; 4],
    /// Tint / vertex color multiplier `[r, g, b, a]`
    pub color: [f32; 4],
    /// Bindless SRV descriptor index, or `u32::MAX` for solid untextured color
    pub texture_index: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

const _: () = assert!(std::mem::size_of::<GpuUiQuad>() == 64);
const _: () = assert!(std::mem::align_of::<GpuUiQuad>() == 16);

impl Default for GpuUiQuad {
    fn default() -> Self {
        Self {
            rect: [0.0; 4],
            uv: [0.0, 0.0, 1.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            texture_index: u32::MAX,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }
}

/// 16-byte root push constants for UI projection
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct UiPushConstants {
    pub screen_width: u32,
    pub screen_height: u32,
    pub virtual_width: u32,
    pub virtual_height: u32,
}

/// Direct3D 12 UI Canvas graphics pipeline
pub struct UiCanvasPipeline {
    pub root_signature: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl UiCanvasPipeline {
    pub fn new(device: &ID3D12Device) -> WinResult<Self> {
        let source = include_str!("../shaders/ui.hlsl");
        let compiler = ShaderCompiler::new().map_err(|_| failure())?;

        let vs_bytes = compiler
            .compile(&ShaderCompileRequest::new(source, "vs_main", ShaderTarget::Vertex6_0))
            .map_err(|e| {
                eprintln!("[UiCanvasPipeline] vs compile failed: {:?}", e);
                failure()
            })?;
        let ps_bytes = compiler
            .compile(&ShaderCompileRequest::new(source, "ps_main", ShaderTarget::Pixel6_0))
            .map_err(|e| {
                eprintln!("[UiCanvasPipeline] ps compile failed: {:?}", e);
                failure()
            })?;

        // 1. Root Signature:
        // Param 0: 4 32-bit push constants (b0) for screen and virtual dimensions
        // Param 1: SRV Descriptor Table (t0.., space 0) for bindless textures
        // Param 2: Root SRV (t0, space 1) for StructuredBuffer<GpuUiQuad>
        let root_constants = D3D12_ROOT_CONSTANTS {
            ShaderRegister: 0,
            RegisterSpace: 0,
            Num32BitValues: 4,
        };

        let srv_range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: u32::MAX,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };

        let root_srv = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 1,
        };

        let root_parameters = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: root_constants,
                },
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &srv_range,
                    },
                },
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: root_srv,
                },
            },
        ];

        let static_sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            MaxLOD: D3D12_FLOAT32_MAX,
            ..Default::default()
        };

        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: root_parameters.len() as u32,
            pParameters: root_parameters.as_ptr(),
            NumStaticSamplers: 1,
            pStaticSamplers: &static_sampler,
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };

        let mut signature_blob = None;
        let mut error_blob = None;
        let root_signature: ID3D12RootSignature = unsafe {
            D3D12SerializeRootSignature(
                &root_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut signature_blob,
                Some(&mut error_blob),
            )?;
            let blob = signature_blob.ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()),
            )?
        };

        // 2. Alpha-blended Pipeline State
        let rt_blend = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            LogicOpEnable: false.into(),
            SrcBlend: D3D12_BLEND_SRC_ALPHA,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        };

        let mut blend_state = D3D12_BLEND_DESC::default();
        for rt in &mut blend_state.RenderTarget {
            rt.SrcBlend = D3D12_BLEND_ONE;
            rt.DestBlend = D3D12_BLEND_ZERO;
            rt.BlendOp = D3D12_BLEND_OP_ADD;
            rt.SrcBlendAlpha = D3D12_BLEND_ONE;
            rt.DestBlendAlpha = D3D12_BLEND_ZERO;
            rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
            rt.LogicOp = D3D12_LOGIC_OP_NOOP;
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        }
        blend_state.RenderTarget[0] = rt_blend;

        let rasterizer_state = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: false.into(),
            DepthClipEnable: false.into(),
            ..Default::default()
        };

        let depth_stencil_state = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            StencilEnable: false.into(),
            ..Default::default()
        };

        let mut pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_signature.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs_bytes.as_ptr().cast(),
                BytecodeLength: vs_bytes.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps_bytes.as_ptr().cast(),
                BytecodeLength: ps_bytes.len(),
            },
            BlendState: blend_state,
            SampleMask: u32::MAX,
            RasterizerState: rasterizer_state,
            DepthStencilState: depth_stencil_state,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        pso_desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        pso_desc.InputLayout = D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: std::ptr::null(),
            NumElements: 0,
        };

        let pso = unsafe { device.CreateGraphicsPipelineState(&pso_desc)? };

        Ok(Self {
            root_signature,
            pso,
        })
    }
}

/// Immediate-mode 2D UI Canvas queue
#[derive(Default)]
pub struct UiCanvas {
    pub quads: Vec<GpuUiQuad>,
}

impl UiCanvas {
    pub fn new() -> Self {
        Self { quads: Vec::new() }
    }

    /// Clear the queued draw elements for the current frame
    pub fn clear(&mut self) {
        self.quads.clear();
    }

    /// Number of queued UI quads
    pub fn len(&self) -> usize {
        self.quads.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    /// Queue a solid untextured color rectangle `[x, y, width, height]`
    pub fn fill_rect(&mut self, rect: [f32; 4], color: [f32; 4]) {
        self.quads.push(GpuUiQuad {
            rect,
            uv: [0.0, 0.0, 1.0, 1.0],
            color,
            texture_index: u32::MAX,
            ..Default::default()
        });
    }

    /// Queue a textured rectangle with custom UV coordinates
    pub fn draw_textured_rect(
        &mut self,
        rect: [f32; 4],
        uv: [f32; 4],
        texture_index: DescriptorIndex,
        color: [f32; 4],
    ) {
        self.quads.push(GpuUiQuad {
            rect,
            uv,
            color,
            texture_index: texture_index.0,
            ..Default::default()
        });
    }

    /// Queue a 2D crosshair centered at `center` with arms of length `size`, gap from center, and thickness
    pub fn draw_crosshair(
        &mut self,
        center: [f32; 2],
        size: f32,
        gap: f32,
        thickness: f32,
        color: [f32; 4],
    ) {
        let half_thick = thickness * 0.5;
        // Top arm
        self.fill_rect(
            [center[0] - half_thick, center[1] - gap - size, thickness, size],
            color,
        );
        // Bottom arm
        self.fill_rect(
            [center[0] - half_thick, center[1] + gap, thickness, size],
            color,
        );
        // Left arm
        self.fill_rect(
            [center[0] - gap - size, center[1] - half_thick, size, thickness],
            color,
        );
        // Right arm
        self.fill_rect(
            [center[0] + gap, center[1] - half_thick, size, thickness],
            color,
        );
    }

    /// Queue a horizontal progress/meter bar (e.g. Health, Stamina, Radiation)
    pub fn draw_meter_bar(
        &mut self,
        rect: [f32; 4],
        fraction: f32,
        bg_color: [f32; 4],
        bar_color: [f32; 4],
    ) {
        // Background
        self.fill_rect(rect, bg_color);
        // Filled bar
        let clamped = fraction.clamp(0.0, 1.0);
        if clamped > 0.0 {
            self.fill_rect(
                [rect[0], rect[1], rect[2] * clamped, rect[3]],
                bar_color,
            );
        }
    }

    /// Queue an outlined border frame
    pub fn draw_frame_border(&mut self, rect: [f32; 4], thickness: f32, color: [f32; 4]) {
        // Top
        self.fill_rect([rect[0], rect[1], rect[2], thickness], color);
        // Bottom
        self.fill_rect([rect[0], rect[1] + rect[3] - thickness, rect[2], thickness], color);
        // Left
        self.fill_rect([rect[0], rect[1] + thickness, thickness, rect[3] - thickness * 2.0], color);
        // Right
        self.fill_rect([rect[0] + rect[2] - thickness, rect[1] + thickness, thickness, rect[3] - thickness * 2.0], color);
    }
}
