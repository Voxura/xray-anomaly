//! DirectX 12 Static Wallmark Decal Projection Renderer
//!
//! Projects static decal geometry (bullet holes, blood splatters, explosion burns)
//! directly onto level geometry with depth bias, alpha blending, and distance/TTL fade.

use windows::core::{Error, Result as WinResult, HRESULT};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};
use xray_formats::hom::ClipFrustum;
use xray_formats::particles::{StaticWallmarkBatch, StaticWallmarkPool, StaticWallmarkRecord, WallmarkVertex};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

/// 24-byte GPU wallmark vertex layout matching `WallmarkVertex`
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuWallmarkVertex {
    pub position: [f32; 3],
    pub color: u32,
    pub uv: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<GpuWallmarkVertex>() == 24);

impl From<&WallmarkVertex> for GpuWallmarkVertex {
    #[inline]
    fn from(v: &WallmarkVertex) -> Self {
        Self {
            position: v.position,
            color: v.color,
            uv: v.uv,
        }
    }
}

/// 16-byte root push constants for wallmark rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallmarkPushConstants {
    pub texture_index: u32,
    pub depth_bias: f32,
    pub alpha_multiplier: f32,
    pub flags: u32,
}

const _: () = assert!(std::mem::size_of::<WallmarkPushConstants>() == 16);

impl Default for WallmarkPushConstants {
    fn default() -> Self {
        Self {
            texture_index: u32::MAX,
            depth_bias: 0.0002, // default slight NDC offset to prevent Z-fighting
            alpha_multiplier: 1.0,
            flags: 0,
        }
    }
}

/// Direct3D 12 Wallmark decal graphics pipeline
pub struct WallmarkRenderer {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl WallmarkRenderer {
    pub fn new(device: &ID3D12Device) -> WinResult<Self> {
        // Param 0: Camera Constants CBV (b0, space 0)
        let mut camera = D3D12_ROOT_PARAMETER::default();
        camera.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        camera.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        camera.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 0,
        };

        // Param 1: Bindless texture table (t0, space 0)
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

        // Param 2: 32-bit push constants (b1, space 0)
        let mut push = D3D12_ROOT_PARAMETER::default();
        push.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        push.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        push.Anonymous.Constants = D3D12_ROOT_CONSTANTS {
            ShaderRegister: 1,
            RegisterSpace: 0,
            Num32BitValues: (std::mem::size_of::<WallmarkPushConstants>() / 4) as u32,
        };

        // Param 3: Wallmark vertices ByteAddressBuffer SRV (t0, space 1)
        let mut vertices = D3D12_ROOT_PARAMETER::default();
        vertices.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        vertices.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        vertices.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 1,
        };

        let parameters = [camera, textures, push, vertices];

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
                        "[WallmarkRenderer] D3D12SerializeRootSignature error: {}",
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

        let source = include_str!("../shaders/wallmark.hlsl");
        let compiler = ShaderCompiler::new().map_err(|e| {
            eprintln!("[WallmarkRenderer] ShaderCompiler::new failed: {:?}", e);
            failure()
        })?;

        let vs = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "vs_wallmark",
                ShaderTarget::Vertex6_0,
            ))
            .map_err(|e| {
                eprintln!("[WallmarkRenderer] vs compile failed: {:?}", e);
                failure()
            })?;

        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_wallmark",
                ShaderTarget::Pixel6_0,
            ))
            .map_err(|e| {
                eprintln!("[WallmarkRenderer] ps compile failed: {:?}", e);
                failure()
            })?;

        let mut blend = D3D12_BLEND_DESC::default();
        let rt_blend = &mut blend.RenderTarget[0];
        rt_blend.BlendEnable = true.into();
        rt_blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        rt_blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        rt_blend.BlendOp = D3D12_BLEND_OP_ADD;
        rt_blend.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt_blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        rt_blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt_blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;

        let rasterizer = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthBias: -500,
            SlopeScaledDepthBias: -1.0,
            DepthClipEnable: true.into(),
            ..Default::default()
        };

        let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO, // Decals do not write depth!
            DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
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
            DSVFormat: DXGI_FORMAT_D32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };

        let pso: ID3D12PipelineState = unsafe { device.CreateGraphicsPipelineState(&pso_desc)? };

        Ok(Self { root, pso })
    }

    /// Render a batch of wallmark decal triangles
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        camera_va: u64,
        bindless_heap: &ID3D12DescriptorHeap,
        vertices_va: u64,
        vertex_count: u32,
        push: &WallmarkPushConstants,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        width: u32,
        height: u32,
    ) {
        if vertex_count == 0 {
            return;
        }

        unsafe {
            cmd.SetPipelineState(&self.pso);
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetDescriptorHeaps(&[Some(bindless_heap.clone())]);

            // Param 0: Camera CBV
            cmd.SetGraphicsRootConstantBufferView(0, camera_va);

            // Param 1: Bindless texture table
            cmd.SetGraphicsRootDescriptorTable(
                1,
                bindless_heap.GetGPUDescriptorHandleForHeapStart(),
            );

            // Param 2: 32-bit push constants
            cmd.SetGraphicsRoot32BitConstants(
                2,
                (std::mem::size_of::<WallmarkPushConstants>() / 4) as u32,
                (push as *const WallmarkPushConstants).cast(),
                0,
            );

            // Param 3: Vertex buffer SRV
            cmd.SetGraphicsRootShaderResourceView(3, vertices_va);

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

            if let Some(dsv) = dsv {
                cmd.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            } else {
                cmd.OMSetRenderTargets(1, Some(&rtv), false, None);
            }

            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.DrawInstanced(vertex_count, 1, 0, 0);
        }
    }
}

/// High-level manager coordinating active static decal pools, TTL pruning, and GPU uploads
pub struct WallmarksManager {
    pool: StaticWallmarkPool,
    upload_buffer: Option<ID3D12Resource>,
    upload_capacity_bytes: usize,
}

impl Default for WallmarksManager {
    fn default() -> Self {
        Self::new()
    }
}

impl WallmarksManager {
    pub fn new() -> Self {
        Self {
            pool: StaticWallmarkPool::new(),
            upload_buffer: None,
            upload_capacity_bytes: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.pool.clear();
    }

    #[inline]
    pub fn add(&mut self, record: StaticWallmarkRecord) {
        self.pool.add(record);
    }

    #[inline]
    pub fn prune_expired(&mut self, time_now: f32) -> usize {
        self.pool.prune_expired(time_now)
    }

    #[inline]
    pub fn pool(&self) -> &StaticWallmarkPool {
        &self.pool
    }

    #[inline]
    pub fn pool_mut(&mut self) -> &mut StaticWallmarkPool {
        &mut self.pool
    }

    /// Frustum + SSA cull, prune expired decals, and collect visible batches
    #[inline]
    pub fn collect_visible_batches(
        &mut self,
        time_now: f32,
        camera_pos: [f32; 3],
        frustum: &ClipFrustum,
        ssa_discard: f32,
    ) -> Vec<StaticWallmarkBatch> {
        self.pool
            .collect_visible_batches(time_now, camera_pos, frustum, ssa_discard)
    }

    /// Upload a contiguous slice of `GpuWallmarkVertex` records to an upload buffer and return GPU VA
    pub fn upload(
        &mut self,
        device: &ID3D12Device,
        vertices: &[GpuWallmarkVertex],
    ) -> WinResult<u64> {
        if vertices.is_empty() {
            return Ok(0);
        }

        let needed_bytes = std::mem::size_of_val(vertices).max(256);

        if self.upload_buffer.is_none() || self.upload_capacity_bytes < needed_bytes {
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: needed_bytes as u64,
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
            self.upload_buffer = Some(res.ok_or_else(failure)?);
            self.upload_capacity_bytes = needed_bytes;
        }

        let res = self.upload_buffer.as_ref().unwrap();
        unsafe {
            let mut mapped = std::ptr::null_mut();
            res.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                vertices.as_ptr() as *const u8,
                mapped.cast(),
                std::mem::size_of_val(vertices),
            );
            res.Unmap(0, None);
            Ok(res.GetGPUVirtualAddress())
        }
    }
}
