//! Phase 10: Infinite Skybox & Environment Cubemap Renderer
//!
//! Faithful implementation of X-Ray Monolith Engine sky and atmosphere subsystem:
//! - Dual-texture weather cycle blending (sky_texture0, sky_texture1, blend_factor)
//! - Procedural unit cube rasterized directly from `SV_VertexID` (0..36) - ZERO vertex buffer allocations
//! - Depth-clamped at far plane ($Z_{NDC} = 1.0$) with `LESS_EQUAL` depth test and zero depth write
//! - Early-Z occlusion culling: foreground geometry occludes sky fragments with zero shader invocation
//! - Atmospheric sunlight flare / phase glow integration

use std::mem::ManuallyDrop;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Error, HRESULT, Result as WinResult};

use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

/// 64-byte 16-byte aligned push constants for sky rendering
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyPushConstants {
    pub sky_texture0: u32,
    pub sky_texture1: u32,
    pub blend_factor: f32,
    pub is_cubemap: u32,
    pub sky_color: [f32; 4],
    pub sun_color: [f32; 4],
    pub sun_dir: [f32; 4],
}

impl Default for SkyPushConstants {
    fn default() -> Self {
        Self {
            sky_texture0: u32::MAX,
            sky_texture1: u32::MAX,
            blend_factor: 0.0,
            is_cubemap: 0,
            sky_color: [1.0, 1.0, 1.0, 1.0],
            sun_color: [1.0, 0.9, 0.7, 0.0],
            sun_dir: [0.577, 0.577, 0.577, 0.0],
        }
    }
}

/// Pipeline and state for infinite skybox rendering
pub struct SkyRenderer {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl SkyRenderer {
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
            Num32BitValues: (std::mem::size_of::<SkyPushConstants>() / 4) as u32,
        };

        let parameters = [camera, textures, push];

        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
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
                    eprintln!("[SkyRenderer] D3D12SerializeRootSignature error: {}", String::from_utf8_lossy(msg));
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
                    eprintln!("[SkyRenderer] CreateRootSignature failed: {:?}", e);
                    return Err(e);
                }
            }
        };

        let source = include_str!("../shaders/sky.hlsl");
        let compiler = match ShaderCompiler::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[SkyRenderer] ShaderCompiler::new failed: {:?}", e);
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
                eprintln!("[SkyRenderer] vs compile failed: {:?}", e);
                failure()
            })?;
        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_main",
                ShaderTarget::Pixel6_0,
            ))
            .map_err(|e| {
                eprintln!("[SkyRenderer] ps compile failed: {:?}", e);
                failure()
            })?;

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

        let pso = unsafe {
            match device.CreateGraphicsPipelineState(&desc) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[SkyRenderer] CreateGraphicsPipelineState failed: {:?}", e);
                    return Err(e);
                }
            }
        };
        unsafe {
            ManuallyDrop::drop(&mut desc.pRootSignature);
        }

        Ok(Self { root, pso })
    }

    /// Record a draw call rendering the infinite skybox
    pub fn render(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        camera_gpu_va: u64,
        srv_heap: &ID3D12DescriptorHeap,
        push: &SkyPushConstants,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        width: u32,
        height: u32,
    ) {
        unsafe {
            cmd.SetPipelineState(&self.pso);
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, camera_gpu_va);
            cmd.SetGraphicsRootDescriptorTable(
                1,
                srv_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            cmd.SetGraphicsRoot32BitConstants(
                2,
                (std::mem::size_of::<SkyPushConstants>() / 4) as u32,
                (push as *const SkyPushConstants).cast(),
                0,
            );
            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            cmd.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            let dsv_ptr = dsv.as_ref().map(|d| d as *const D3D12_CPU_DESCRIPTOR_HANDLE);
            cmd.OMSetRenderTargets(1, Some(&rtv), false, dsv_ptr);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.DrawInstanced(36, 1, 0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sky_push_constants_size_and_alignment() {
        assert_eq!(std::mem::size_of::<SkyPushConstants>(), 64);
        assert_eq!(std::mem::align_of::<SkyPushConstants>(), 16);
    }
}
