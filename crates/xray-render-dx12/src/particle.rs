//! Particle Billboard Instancing Subsystem for DirectX 12
//!
//! Provides ultra-low-overhead GPU instanced particle rendering. Quads are generated
//! purely procedurally via SV_VertexID in the vertex shader, eliminating all vertex buffer
//! allocations. Particle instance records are streamed via ByteAddressBuffer with bindless textures
//! and hardware alpha / additive blending.

use std::mem::ManuallyDrop;
use windows::core::{Result as WinResult, Error, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Foundation::RECT;

use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

/// 80-byte GPU particle instance layout (5 float4s, aligned to 16 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuParticleInstance {
    pub position: [f32; 3],
    pub rotation: f32,
    pub size: [f32; 2],
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub pad0: [f32; 2],
    pub color: [f32; 4],
    pub texture_index: u32,
    pub blend_mode: u32, // 0 = alpha blend, 1 = additive
    pub pad1: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<GpuParticleInstance>() == 80);

impl Default for GpuParticleInstance {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation: 0.0,
            size: [1.0, 1.0],
            uv_min: [0.0, 0.0],
            uv_max: [1.0, 1.0],
            pad0: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            texture_index: u32::MAX,
            blend_mode: 0,
            pad1: [0, 0],
        }
    }
}

/// High-performance GPU particle system renderer
pub struct ParticleSystemRenderer {
    root: ID3D12RootSignature,
    alpha_pso: ID3D12PipelineState,
    additive_pso: ID3D12PipelineState,
}

impl ParticleSystemRenderer {
    /// Create the particle pipeline states with alpha and additive blending
    pub fn new(device: &ID3D12Device) -> WinResult<Self> {
        // Root Signature:
        // Param 0: Camera Constants CBV (register b0, space 0)
        // Param 1: Bindless texture table (register t0, space 0)
        // Param 2: Particle instance ByteAddressBuffer SRV (register t0, space 1)
        let range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: u32::MAX,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };

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
        textures.Anonymous.DescriptorTable = D3D12_ROOT_DESCRIPTOR_TABLE {
            NumDescriptorRanges: 1,
            pDescriptorRanges: &range,
        };

        let mut particles = D3D12_ROOT_PARAMETER::default();
        particles.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        particles.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        particles.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 1,
        };

        let parameters = [camera, textures, particles];

        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
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
            NumParameters: 3,
            pParameters: parameters.as_ptr(),
            NumStaticSamplers: 1,
            pStaticSamplers: &sampler,
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };

        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut blob, None)?;
            let blob = blob.ok_or_else(failure)?;
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()),
            )?
        };

        let source = include_str!("../shaders/particle.hlsl");
        let compiler = ShaderCompiler::new().map_err(|_| failure())?;
        let vs = compiler
            .compile(&ShaderCompileRequest::new(source, "vs_particle", ShaderTarget::Vertex6_0))
            .map_err(|e| {
                eprintln!("[ParticleSystemRenderer] vs compile failed: {:?}", e);
                failure()
            })?;
        let ps = compiler
            .compile(&ShaderCompileRequest::new(source, "ps_particle", ShaderTarget::Pixel6_0))
            .map_err(|e| {
                eprintln!("[ParticleSystemRenderer] ps compile failed: {:?}", e);
                failure()
            })?;

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
        // Zero vertex layout (procedural vertex ID generation)
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
        desc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO, // Particles test depth but do not write depth
            DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            StencilEnable: false.into(),
            ..Default::default()
        };
        desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        desc.SampleMask = u32::MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };

        // 1. Alpha Blend PSO
        let mut alpha_desc = desc.clone();
        alpha_desc.pRootSignature = ManuallyDrop::new(Some(root.clone()));
        alpha_desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D12_BLEND_SRC_ALPHA,
            DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let alpha_pso = unsafe { device.CreateGraphicsPipelineState(&alpha_desc)? };
        unsafe {
            ManuallyDrop::drop(&mut alpha_desc.pRootSignature);
        }

        // 2. Additive Blend PSO
        let mut add_desc = desc;
        add_desc.pRootSignature = ManuallyDrop::new(Some(root.clone()));
        add_desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ONE,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ONE,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let additive_pso = unsafe { device.CreateGraphicsPipelineState(&add_desc)? };
        unsafe {
            ManuallyDrop::drop(&mut add_desc.pRootSignature);
        }

        Ok(Self {
            root,
            alpha_pso,
            additive_pso,
        })
    }

    /// Record instanced particle billboard draw call
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        camera_gpu_va: u64,
        bindless_heap: &ID3D12DescriptorHeap,
        particle_buffer_gpu_va: u64,
        instance_count: u32,
        additive: bool,
        width: u32,
        height: u32,
    ) {
        if instance_count == 0 {
            return;
        }

        let pso = if additive {
            &self.additive_pso
        } else {
            &self.alpha_pso
        };

        unsafe {
            cmd.SetPipelineState(pso);
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetDescriptorHeaps(&[Some(bindless_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, camera_gpu_va);
            cmd.SetGraphicsRootDescriptorTable(1, bindless_heap.GetGPUDescriptorHandleForHeapStart());
            cmd.SetGraphicsRootShaderResourceView(2, particle_buffer_gpu_va);

            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            cmd.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);

            let dsv_ptr = dsv.as_ref().map(|d| d as *const D3D12_CPU_DESCRIPTOR_HANDLE);
            cmd.OMSetRenderTargets(1, Some(&rtv), false, dsv_ptr);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // 6 vertices per quad (2 triangles), fully procedurally generated from SV_VertexID
            cmd.DrawInstanced(6, instance_count, 0, 0);
        }
    }
}
