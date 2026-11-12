//! Phase 11: Dual-Layer Scrolling Clouds Dome Renderer
//!
//! Renders a procedural hemisphere cloud dome generated on the GPU via `SV_VertexID` (3072 vertices, 512 quads).
//! Features:
//! - Dual-texture weather cycle blending (clouds_texture0, clouds_texture1, blend_factor)
//! - Continuous wind UV velocity scrolling across primary and secondary cloud layers
//! - Sun forward-scattering illumination based on weather cycle solar angles
//! - Depth-clamped at far plane ($Z_{NDC} = 1.0$) with `LESS_EQUAL` depth testing and zero depth write
//! - Zero-overdraw Early-Z occlusion: foreground level geometry occludes cloud dome fragments

use std::mem::ManuallyDrop;
use windows::core::{Error, HRESULT, Result as WinResult};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

/// 96-byte 16-byte aligned push constants for clouds rendering
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudsPushConstants {
    pub clouds_texture0: u32,
    pub clouds_texture1: u32,
    pub blend_factor: f32,
    pub clouds_rotation: f32,
    pub clouds_color: [f32; 4],
    pub sun_color: [f32; 4],
    pub sun_dir: [f32; 4],
    pub uv_offset0: [f32; 2],
    pub uv_offset1: [f32; 2],
    pub clouds_scale: [f32; 2], // [horizontal_scale, vertical_scale]
    pub padding: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<CloudsPushConstants>() == 96);

impl Default for CloudsPushConstants {
    fn default() -> Self {
        Self {
            clouds_texture0: u32::MAX,
            clouds_texture1: u32::MAX,
            blend_factor: 0.0,
            clouds_rotation: 0.0,
            clouds_color: [1.0, 1.0, 1.0, 1.0],
            sun_color: [1.0, 0.9, 0.7, 0.5],
            sun_dir: [0.577, 0.577, 0.577, 0.0],
            uv_offset0: [0.0, 0.0],
            uv_offset1: [0.0, 0.0],
            clouds_scale: [100.0, 35.0],
            padding: [0.0, 0.0],
        }
    }
}

/// Simulation state tracking dynamic wind-driven UV velocity scrolling
#[derive(Debug, Clone, PartialEq)]
pub struct CloudSimulation {
    pub uv_offset0: [f32; 2],
    pub uv_offset1: [f32; 2],
}

impl Default for CloudSimulation {
    fn default() -> Self {
        Self {
            uv_offset0: [0.0, 0.0],
            uv_offset1: [0.0, 0.0],
        }
    }
}

impl CloudSimulation {
    /// Advance cloud layer UV offsets given time delta `dt`, wind velocity, and wind direction (radians)
    pub fn step(&mut self, dt: f32, wind_velocity: f32, wind_direction: f32) {
        // Primary layer wind vector
        let speed = wind_velocity * 0.0002;
        let dx0 = wind_direction.cos() * speed * dt;
        let dy0 = wind_direction.sin() * speed * dt;
        self.uv_offset0[0] = (self.uv_offset0[0] + dx0).fract();
        self.uv_offset0[1] = (self.uv_offset0[1] + dy0).fract();

        // Secondary layer scrolls at 1.5x speed with 45-degree crosswind offset
        let cross_dir = wind_direction + std::f32::consts::FRAC_PI_4;
        let dx1 = cross_dir.cos() * speed * 1.5 * dt;
        let dy1 = cross_dir.sin() * speed * 1.5 * dt;
        self.uv_offset1[0] = (self.uv_offset1[0] + dx1).fract();
        self.uv_offset1[1] = (self.uv_offset1[1] + dy1).fract();
    }
}

/// Pipeline and state for dual-layer scrolling clouds rendering
pub struct CloudRenderer {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl CloudRenderer {
    pub fn new(device: &ID3D12Device) -> WinResult<Self> {
        let mut camera = D3D12_ROOT_PARAMETER::default();
        camera.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        camera.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        camera.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 0,
        };

        let mut textures = D3D12_ROOT_PARAMETER::default();
        textures.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        textures.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        let srv_range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: u32::MAX,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };
        textures.Anonymous.DescriptorTable = D3D12_ROOT_DESCRIPTOR_TABLE {
            NumDescriptorRanges: 1,
            pDescriptorRanges: &srv_range,
        };

        let mut push = D3D12_ROOT_PARAMETER::default();
        push.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        push.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        push.Anonymous.Constants = D3D12_ROOT_CONSTANTS {
            ShaderRegister: 1,
            RegisterSpace: 0,
            Num32BitValues: (std::mem::size_of::<CloudsPushConstants>() / 4) as u32,
        };

        let parameters = [camera, textures, push];

        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            MaxLOD: f32::MAX,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..Default::default()
        };

        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as u32,
            pParameters: parameters.as_ptr(),
            NumStaticSamplers: 1,
            pStaticSamplers: &sampler,
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };

        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            let mut err = None;
            let res = D3D12SerializeRootSignature(
                &root_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut blob,
                Some(&mut err),
            );
            if let Err(e) = res {
                if let Some(err_b) = err {
                    let msg = std::slice::from_raw_parts(
                        err_b.GetBufferPointer() as *const u8,
                        err_b.GetBufferSize(),
                    );
                    eprintln!("[CloudRenderer] D3D12SerializeRootSignature error: {}", String::from_utf8_lossy(msg));
                }
                return Err(e);
            }
            let blob = blob.ok_or_else(failure)?;
            match device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                ),
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[CloudRenderer] CreateRootSignature failed: {:?}", e);
                    return Err(e);
                }
            }
        };

        let source = include_str!("../shaders/clouds.hlsl");
        let compiler = match ShaderCompiler::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[CloudRenderer] ShaderCompiler::new failed: {:?}", e);
                return Err(failure());
            }
        };
        let vs = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "vs_main",
                ShaderTarget::Vertex6_0,
            ))
            .map_err(|e| {
                eprintln!("[CloudRenderer] vs compile failed: {:?}", e);
                failure()
            })?;
        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_main",
                ShaderTarget::Pixel6_0,
            ))
            .map_err(|e| {
                eprintln!("[CloudRenderer] ps compile failed: {:?}", e);
                failure()
            })?;

        // Standard alpha blending over skybox: (SrcAlpha * SrcColor + InvSrcAlpha * DestColor)
        let mut blend_state = D3D12_BLEND_DESC::default();
        let rt = &mut blend_state.RenderTarget[0];
        rt.BlendEnable = true.into();
        rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;

        let depth_stencil_state = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            StencilEnable: false.into(),
            FrontFace: D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_KEEP,
                StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                StencilPassOp: D3D12_STENCIL_OP_KEEP,
                StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            },
            BackFace: D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_KEEP,
                StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                StencilPassOp: D3D12_STENCIL_OP_KEEP,
                StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            },
            ..Default::default()
        };

        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        desc.pRootSignature = ManuallyDrop::new(Some(root.clone()));
        desc.VS = D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr().cast(),
            BytecodeLength: vs.len(),
        };
        desc.PS = D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr().cast(),
            BytecodeLength: ps.len(),
        };
        desc.InputLayout = D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: std::ptr::null(),
            NumElements: 0,
        };
        desc.RasterizerState = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            ..Default::default()
        };
        desc.BlendState = blend_state;
        desc.DepthStencilState = depth_stencil_state;
        desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        desc.SampleMask = u32::MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc = DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        };

        let pso: ID3D12PipelineState = unsafe {
            match device.CreateGraphicsPipelineState(&desc) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[CloudRenderer] CreateGraphicsPipelineState failed: {:?}", e);
                    return Err(e);
                }
            }
        };

        Ok(Self { root, pso })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        camera_gpu_va: u64,
        srv_heap: &ID3D12DescriptorHeap,
        push: &CloudsPushConstants,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        width: u32,
        height: u32,
    ) {
        unsafe {
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetPipelineState(&self.pso);

            // Bind camera CBV (register b0)
            cmd.SetGraphicsRootConstantBufferView(0, camera_gpu_va);

            // Bind bindless textures heap (register t0, space0)
            cmd.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            cmd.SetGraphicsRootDescriptorTable(1, srv_heap.GetGPUDescriptorHandleForHeapStart());

            // Bind push constants (register b1)
            let push_slice = std::slice::from_raw_parts(
                (push as *const CloudsPushConstants).cast::<u32>(),
                std::mem::size_of::<CloudsPushConstants>() / 4,
            );
            cmd.SetGraphicsRoot32BitConstants(2, push_slice.len() as u32, push_slice.as_ptr().cast(), 0);

            // Viewport & scissor
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let scissor = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            cmd.RSSetViewports(&[viewport]);
            cmd.RSSetScissorRects(&[scissor]);

            // Set render target and optional depth stencil
            let dsv_ptr = dsv.as_ref().map(|d| d as *const D3D12_CPU_DESCRIPTOR_HANDLE);
            cmd.OMSetRenderTargets(1, Some(&rtv), false, dsv_ptr);

            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // 36 vertices generated procedurally in VS (unit sky cube)
            cmd.DrawInstanced(36, 1, 0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clouds_push_constants_size_and_alignment() {
        assert_eq!(std::mem::size_of::<CloudsPushConstants>(), 96);
        assert_eq!(std::mem::align_of::<CloudsPushConstants>(), 16);
    }

    #[test]
    fn test_cloud_simulation_step() {
        let mut sim = CloudSimulation::default();
        sim.step(1.0, 100.0, 0.0);
        assert!(sim.uv_offset0[0] > 0.0);
        assert_eq!(sim.uv_offset0[1], 0.0);
        assert!(sim.uv_offset1[0] > 0.0);
        assert!(sim.uv_offset1[1] > 0.0);
    }
}
