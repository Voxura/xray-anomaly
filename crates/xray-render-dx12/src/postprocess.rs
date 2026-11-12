//! DirectX 12 Post-Processing, HDR10 Tonemapping, Bloom & Color Grading Renderer
//!
//! Executes full-screen image processing via procedural triangle rasterization (SV_VertexID 0..2)
//! without any vertex or index buffer allocations, providing Uncharted 2 / ACES / Reinhard tonemapping,
//! exposure control, gamma correction, radial vignette, and bloom composition.

use windows::core::{Error, Result as WinResult, HRESULT};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};
pub use xray_formats::postprocess::{
    BloomAdaptState, BloomConfig, ImageCorrectionConfig, ImageCorrectionGpuParams,
    TonemapGpuParams, TonemapParams,
};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

/// 32-byte push constants for post-processing execution
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostprocessPushConstants {
    pub screen_res_inv: [f32; 2],
    pub time_seconds: f32,
    pub bloom_factor: f32,
    pub scene_texture_index: u32,
    pub bloom_texture_index: u32,
    pub tonemap_mode: u32,
    pub color_grading_enabled: u32,
}

const _: () = assert!(std::mem::size_of::<PostprocessPushConstants>() == 32);

impl Default for PostprocessPushConstants {
    fn default() -> Self {
        Self {
            screen_res_inv: [0.0, 0.0],
            time_seconds: 0.0,
            bloom_factor: 0.5,
            scene_texture_index: u32::MAX,
            bloom_texture_index: u32::MAX,
            tonemap_mode: 0, // Uncharted 2 / Filmic default
            color_grading_enabled: 0,
        }
    }
}

/// Direct3D 12 Post-Processing graphics pipeline
pub struct PostprocessRenderer {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl PostprocessRenderer {
    pub fn new(device: &ID3D12Device) -> WinResult<Self> {
        // Param 0: 32-bit push constants (b0, space 0)
        let mut push = D3D12_ROOT_PARAMETER::default();
        push.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        push.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        push.Anonymous.Constants = D3D12_ROOT_CONSTANTS {
            ShaderRegister: 0,
            RegisterSpace: 0,
            Num32BitValues: (std::mem::size_of::<PostprocessPushConstants>() / 4) as u32,
        };

        // Param 1: Tonemap parameters CBV (b1, space 0)
        let mut tonemap = D3D12_ROOT_PARAMETER::default();
        tonemap.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        tonemap.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        tonemap.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 1,
            RegisterSpace: 0,
        };

        // Param 2: Image correction parameters CBV (b2, space 0)
        let mut img_correct = D3D12_ROOT_PARAMETER::default();
        img_correct.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        img_correct.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        img_correct.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 2,
            RegisterSpace: 0,
        };

        // Param 3: Bindless texture table (t0, space 0)
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

        let parameters = [push, tonemap, img_correct, textures];

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
                    eprintln!(
                        "[PostprocessRenderer] D3D12SerializeRootSignature error: {}",
                        String::from_utf8_lossy(msg)
                    );
                }
                return Err(e);
            }
            let blob = blob.ok_or_else(failure)?;
            device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                ),
            )?
        };

        let source = include_str!("../shaders/postprocess.hlsl");
        let compiler = ShaderCompiler::new().map_err(|e| {
            eprintln!("[PostprocessRenderer] ShaderCompiler::new failed: {:?}", e);
            failure()
        })?;

        let vs = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "vs_postprocess",
                ShaderTarget::Vertex6_0,
            ))
            .map_err(|e| {
                eprintln!("[PostprocessRenderer] vs compile failed: {:?}", e);
                failure()
            })?;

        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_postprocess",
                ShaderTarget::Pixel6_0,
            ))
            .map_err(|e| {
                eprintln!("[PostprocessRenderer] ps compile failed: {:?}", e);
                failure()
            })?;

        let mut blend = D3D12_BLEND_DESC::default();
        let rt_blend = &mut blend.RenderTarget[0];
        rt_blend.BlendEnable = false.into(); // Direct overwrite
        rt_blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;

        let rasterizer = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: false.into(),
            ..Default::default()
        };

        let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            ..Default::default()
        };

        let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.as_ptr().cast(),
                BytecodeLength: vs.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.as_ptr().cast(),
                BytecodeLength: ps.len(),
            },
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: rasterizer,
            DepthStencilState: depth_stencil,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: [
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
            ],
            DSVFormat: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };

        let pso: ID3D12PipelineState = unsafe { device.CreateGraphicsPipelineState(&pso_desc)? };

        Ok(Self { root, pso })
    }

    /// Render post-processing fullscreen pass
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        bindless_heap: &ID3D12DescriptorHeap,
        tonemap_va: u64,
        img_correct_va: u64,
        push: &PostprocessPushConstants,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
    ) {
        unsafe {
            cmd.SetPipelineState(&self.pso);
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetDescriptorHeaps(&[Some(bindless_heap.clone())]);

            // Param 0: 32-bit push constants
            cmd.SetGraphicsRoot32BitConstants(
                0,
                (std::mem::size_of::<PostprocessPushConstants>() / 4) as u32,
                (push as *const PostprocessPushConstants).cast(),
                0,
            );

            // Param 1: Tonemap CBV
            cmd.SetGraphicsRootConstantBufferView(1, tonemap_va);

            // Param 2: Image correction CBV
            cmd.SetGraphicsRootConstantBufferView(2, img_correct_va);

            // Param 3: Bindless texture table
            cmd.SetGraphicsRootDescriptorTable(
                3,
                bindless_heap.GetGPUDescriptorHandleForHeapStart(),
            );

            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);

            cmd.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);

            cmd.OMSetRenderTargets(1, Some(&rtv), false, None);

            // Procedural full-screen triangle: 3 vertices from SV_VertexID
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
    }
}

/// High-level coordinator managing post-processing tonemap, bloom, and color correction constants
pub struct PostprocessManager {
    pub tonemap: TonemapParams,
    pub image_correction: ImageCorrectionConfig,
    pub bloom: BloomConfig,
    pub bloom_adapt: BloomAdaptState,
    tonemap_buffer: Option<ID3D12Resource>,
    img_correct_buffer: Option<ID3D12Resource>,
}

impl Default for PostprocessManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PostprocessManager {
    pub fn new() -> Self {
        Self {
            tonemap: TonemapParams::default(),
            image_correction: ImageCorrectionConfig::default(),
            bloom: BloomConfig::default(),
            bloom_adapt: BloomAdaptState::default(),
            tonemap_buffer: None,
            img_correct_buffer: None,
        }
    }

    /// Upload tonemap and image correction parameters to GPU constant buffers and return (tonemap_va, img_correct_va)
    pub fn upload_constants(&mut self, device: &ID3D12Device) -> WinResult<(u64, u64)> {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };

        // 256-byte aligned CBV for Tonemap
        if self.tonemap_buffer.is_none() {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            unsafe {
                device.CreateCommittedResource(
                    &heap_props,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut res,
                )?;
            }
            self.tonemap_buffer = Some(res.ok_or_else(failure)?);
        }

        // 256-byte aligned CBV for Image Correction
        if self.img_correct_buffer.is_none() {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut res: Option<ID3D12Resource> = None;
            unsafe {
                device.CreateCommittedResource(
                    &heap_props,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut res,
                )?;
            }
            self.img_correct_buffer = Some(res.ok_or_else(failure)?);
        }

        let gpu_tonemap = self.tonemap.to_gpu_params();
        let gpu_img = self.image_correction.to_gpu_params();

        let tb = self.tonemap_buffer.as_ref().unwrap();
        unsafe {
            let mut mapped = std::ptr::null_mut();
            tb.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                &gpu_tonemap as *const TonemapGpuParams as *const u8,
                mapped.cast(),
                std::mem::size_of::<TonemapGpuParams>(),
            );
            tb.Unmap(0, None);
        }

        let ib = self.img_correct_buffer.as_ref().unwrap();
        unsafe {
            let mut mapped = std::ptr::null_mut();
            ib.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                &gpu_img as *const ImageCorrectionGpuParams as *const u8,
                mapped.cast(),
                std::mem::size_of::<ImageCorrectionGpuParams>(),
            );
            ib.Unmap(0, None);
        }

        let t_va = unsafe { tb.GetGPUVirtualAddress() };
        let i_va = unsafe { ib.GetGPUVirtualAddress() };
        Ok((t_va, i_va))
    }
}
