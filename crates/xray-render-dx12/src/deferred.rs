//! Modern Deferred Rendering Pipeline for DirectX 12
//!
//! Manages GBuffer MRT resource declarations (Albedo, Normal, Material, Depth),
//! Shader Model 6.6 bindless descriptor associations, GBuffer mesh rasterization,
//! and automated Deferred Combine / Lighting evaluation into an HDR accumulation target.

use std::mem::ManuallyDrop;
use std::path::Path;
use windows::core::{Error, Result as WinResult, HRESULT};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::bindless::{BindlessIndexAllocator, DescriptorIndex};
use crate::graph::{CompiledSchedule, RenderGraph, ResourceHandle, ResourceState};
use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

fn transition_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Standard GBuffer layout for X-Ray DX12 deferred shading
#[derive(Debug, Clone, Copy)]
pub struct GBufferResources {
    pub albedo: ResourceHandle,
    pub normal: ResourceHandle,
    pub material: ResourceHandle,
    pub depth: ResourceHandle,
    pub hdr_output: ResourceHandle,
    pub backbuffer: ResourceHandle,
}

/// Deferred pipeline configuration and pass graph builder
pub struct DeferredPipeline {
    pub width: u32,
    pub height: u32,
}

impl DeferredPipeline {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Build the complete Render Graph for a frame containing GBuffer, Lighting, and Presentation passes.
    pub fn build_graph(&self, graph: &mut RenderGraph) -> (GBufferResources, CompiledSchedule) {
        let albedo = graph.create_resource("GBuffer_Albedo", ResourceState::Common, true);
        let normal = graph.create_resource("GBuffer_Normal", ResourceState::Common, true);
        let material = graph.create_resource("GBuffer_Material", ResourceState::Common, true);
        let depth = graph.create_resource("GBuffer_Depth", ResourceState::Common, true);
        let hdr_output = graph.create_resource("HDR_Accumulation", ResourceState::Common, true);
        let backbuffer = graph.create_resource("BackBuffer", ResourceState::Present, false);

        let resources = GBufferResources {
            albedo,
            normal,
            material,
            depth,
            hdr_output,
            backbuffer,
        };

        // Pass 1: GBuffer Rasterization Pass (Writes Albedo, Normal, Material, Depth)
        graph
            .add_pass("GBuffer_Raster")
            .write(albedo, ResourceState::RenderTarget)
            .write(normal, ResourceState::RenderTarget)
            .write(material, ResourceState::RenderTarget)
            .write(depth, ResourceState::DepthWrite);

        // Pass 2: Deferred Lighting Compute Pass (Reads GBuffer textures & Depth via SM 6.6 bindless, writes HDR UAV)
        graph
            .add_pass("Deferred_Lighting_Compute")
            .read(albedo, ResourceState::NonPixelShaderResource)
            .read(normal, ResourceState::NonPixelShaderResource)
            .read(material, ResourceState::NonPixelShaderResource)
            .read(depth, ResourceState::NonPixelShaderResource)
            .write(hdr_output, ResourceState::UnorderedAccess);

        // Pass 3: Tonemapping and Post-Processing (Reads HDR UAV, writes to BackBuffer)
        graph
            .add_pass("Tonemap_And_Present")
            .read(hdr_output, ResourceState::PixelShaderResource)
            .write(backbuffer, ResourceState::RenderTarget);

        let schedule = graph
            .compile()
            .expect("Deferred pipeline graph compilation failed");

        (resources, schedule)
    }

    /// Compile all required deferred shaders via DXC offline compiler
    pub fn compile_shaders(
        compiler: &ShaderCompiler,
        shaders_dir: &Path,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let gbuffer_path = shaders_dir.join("gbuffer.hlsl");
        let lighting_path = shaders_dir.join("deferred_lighting.hlsl");

        let gbuffer_hlsl = std::fs::read_to_string(&gbuffer_path)
            .map_err(|e| format!("Failed to read {}: {}", gbuffer_path.display(), e))?;
        let lighting_hlsl = std::fs::read_to_string(&lighting_path)
            .map_err(|e| format!("Failed to read {}: {}", lighting_path.display(), e))?;

        let req_vs = ShaderCompileRequest::new(&gbuffer_hlsl, "vs_main", ShaderTarget::Vertex6_6);
        let vs_bc = compiler
            .compile(&req_vs)
            .map_err(|e| format!("GBuffer VS error: {:?}", e))?;

        let req_ps = ShaderCompileRequest::new(&gbuffer_hlsl, "ps_main", ShaderTarget::Pixel6_6);
        let ps_bc = compiler
            .compile(&req_ps)
            .map_err(|e| format!("GBuffer PS error: {:?}", e))?;

        let req_cs =
            ShaderCompileRequest::new(&lighting_hlsl, "cs_main", ShaderTarget::Compute6_6);
        let cs_bc = compiler
            .compile(&req_cs)
            .map_err(|e| format!("Deferred Lighting CS error: {:?}", e))?;

        Ok((vs_bc, ps_bc, cs_bc))
    }
}

/// 32-byte push constants for deferred combine & lighting pass
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PushCombineConstants {
    pub gbuffer_albedo_idx: u32,
    pub gbuffer_normal_idx: u32,
    pub gbuffer_material_idx: u32,
    pub gbuffer_depth_idx: u32,
    pub shadow_map_idx: u32,
    pub num_lights: u32,
    pub screen_res_inv: [f32; 2],
    pub csm_active: u32,
    pub volumetric_active: u32,
    pub _pad: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<PushCombineConstants>() == 48);

impl Default for PushCombineConstants {
    fn default() -> Self {
        Self {
            gbuffer_albedo_idx: u32::MAX,
            gbuffer_normal_idx: u32::MAX,
            gbuffer_material_idx: u32::MAX,
            gbuffer_depth_idx: u32::MAX,
            shadow_map_idx: u32::MAX,
            num_lights: 0,
            screen_res_inv: [0.0, 0.0],
            csm_active: 0,
            volumetric_active: 0,
            _pad: [0, 0],
        }
    }
}

/// GPU sunlight, ambient, and shadow projection constants for deferred combine
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuLightParams {
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub ambient_color: [f32; 4],
    pub sun_view_proj: [[f32; 4]; 4],
}

impl Default for GpuLightParams {
    fn default() -> Self {
        let mut sun_view_proj = [[0.0f32; 4]; 4];
        for i in 0..4 {
            sun_view_proj[i][i] = 1.0;
        }
        Self {
            sun_direction: [0.0, 1.0, 0.0, 0.0],
            sun_color: [1.0, 1.0, 1.0, 1.0],
            ambient_color: [0.1, 0.1, 0.1, 1.0],
            sun_view_proj,
        }
    }
}

/// G-Buffer MRT textures (Albedo, Normal, Material, Depth, and HDR Accumulation Target)
pub struct GBufferTargets {
    pub width: u32,
    pub height: u32,
    pub albedo: ID3D12Resource,
    pub normal: ID3D12Resource,
    pub material: ID3D12Resource,
    pub depth: ID3D12Resource,
    pub hdr_accumulation: ID3D12Resource,
    pub rtv_heap: ID3D12DescriptorHeap,
    pub dsv_heap: ID3D12DescriptorHeap,
    pub rtv_descriptor_size: u32,
    pub albedo_srv: DescriptorIndex,
    pub normal_srv: DescriptorIndex,
    pub material_srv: DescriptorIndex,
    pub depth_srv: DescriptorIndex,
    pub hdr_srv: DescriptorIndex,
}

impl GBufferTargets {
    pub fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        bindless_allocator: &mut BindlessIndexAllocator,
        bindless_heap: &ID3D12DescriptorHeap,
        srv_descriptor_size: u32,
    ) -> WinResult<Self> {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };

        let make_tex_desc = |format: DXGI_FORMAT, flags: D3D12_RESOURCE_FLAGS| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
            Alignment: 0,
        };

        let clear_val_rtv = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                Color: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let mut albedo = None;
        let mut normal = None;
        let mut material = None;
        let mut hdr_accumulation = None;
        let mut depth = None;

        unsafe {
            let desc_rtv = make_tex_desc(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            );
            device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc_rtv,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                Some(&clear_val_rtv),
                &mut albedo,
            )?;
            device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc_rtv,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                Some(&clear_val_rtv),
                &mut normal,
            )?;
            device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc_rtv,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                Some(&clear_val_rtv),
                &mut material,
            )?;

            let desc_hdr = make_tex_desc(
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            );
            let clear_hdr = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    Color: [0.0, 0.0, 0.0, 1.0],
                },
            };
            device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc_hdr,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                Some(&clear_hdr),
                &mut hdr_accumulation,
            )?;

            let desc_depth = make_tex_desc(
                DXGI_FORMAT_R32_TYPELESS,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            );
            let clear_depth = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                        Depth: 1.0,
                        Stencil: 0,
                    },
                },
            };
            device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc_depth,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                Some(&clear_depth),
                &mut depth,
            )?;
        }

        let albedo = albedo.ok_or_else(failure)?;
        let normal = normal.ok_or_else(failure)?;
        let material = material.ok_or_else(failure)?;
        let hdr_accumulation = hdr_accumulation.ok_or_else(failure)?;
        let depth = depth.ok_or_else(failure)?;

        // RTV heap for 4 targets: [albedo, normal, material, hdr_accumulation]
        let rtv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 4,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let rtv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&rtv_heap_desc)? };
        let rtv_descriptor_size = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
        };

        let rtv_start = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        for (i, res) in [&albedo, &normal, &material, &hdr_accumulation]
            .iter()
            .enumerate()
        {
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: rtv_start.ptr + i * (rtv_descriptor_size as usize),
            };
            unsafe {
                device.CreateRenderTargetView(*res, None, handle);
            }
        }

        // DSV heap for depth
        let dsv_heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let dsv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&dsv_heap_desc)? };
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            Flags: D3D12_DSV_FLAG_NONE,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 },
            },
        };
        unsafe {
            device.CreateDepthStencilView(
                &depth,
                Some(&dsv_desc),
                dsv_heap.GetCPUDescriptorHandleForHeapStart(),
            );
        }

        // Allocate 5 bindless SRVs
        let albedo_srv = bindless_allocator.allocate().ok_or_else(failure)?;
        let normal_srv = bindless_allocator.allocate().ok_or_else(failure)?;
        let material_srv = bindless_allocator.allocate().ok_or_else(failure)?;
        let depth_srv = bindless_allocator.allocate().ok_or_else(failure)?;
        let hdr_srv = bindless_allocator.allocate().ok_or_else(failure)?;

        let bindless_start = unsafe { bindless_heap.GetCPUDescriptorHandleForHeapStart() };

        let make_srv_desc = |format: DXGI_FORMAT| D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };

        for (res, idx, format) in [
            (&albedo, albedo_srv, DXGI_FORMAT_R8G8B8A8_UNORM),
            (&normal, normal_srv, DXGI_FORMAT_R8G8B8A8_UNORM),
            (&material, material_srv, DXGI_FORMAT_R8G8B8A8_UNORM),
            (&depth, depth_srv, DXGI_FORMAT_R32_FLOAT),
            (&hdr_accumulation, hdr_srv, DXGI_FORMAT_R16G16B16A16_FLOAT),
        ] {
            let dest_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: bindless_start.ptr + (idx.0 as usize) * (srv_descriptor_size as usize),
            };
            let srv_desc = make_srv_desc(format);
            unsafe {
                device.CreateShaderResourceView(Some(res), Some(&srv_desc), dest_handle);
            }
        }

        Ok(Self {
            width,
            height,
            albedo,
            normal,
            material,
            depth,
            hdr_accumulation,
            rtv_heap,
            dsv_heap,
            rtv_descriptor_size,
            albedo_srv,
            normal_srv,
            material_srv,
            depth_srv,
            hdr_srv,
        })
    }

    pub fn rtv_handle(&self, index: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let start = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + index * (self.rtv_descriptor_size as usize),
        }
    }

    pub fn gbuffer_rtvs(&self) -> [D3D12_CPU_DESCRIPTOR_HANDLE; 3] {
        [self.rtv_handle(0), self.rtv_handle(1), self.rtv_handle(2)]
    }

    pub fn hdr_rtv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        self.rtv_handle(3)
    }

    pub fn dsv_handle(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        unsafe { self.dsv_heap.GetCPUDescriptorHandleForHeapStart() }
    }

    pub fn begin_gbuffer_pass(&self, cmd: &ID3D12GraphicsCommandList) {
        let barriers = [
            transition_barrier(
                &self.albedo,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ),
            transition_barrier(
                &self.normal,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ),
            transition_barrier(
                &self.material,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            ),
            transition_barrier(
                &self.depth,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
            ),
        ];
        unsafe {
            cmd.ResourceBarrier(&barriers);
        }
    }

    pub fn clear_gbuffer(&self, cmd: &ID3D12GraphicsCommandList) {
        let rtvs = self.gbuffer_rtvs();
        let black = [0.0f32, 0.0, 0.0, 1.0];
        let dsv = self.dsv_handle();
        unsafe {
            cmd.ClearRenderTargetView(rtvs[0], &black, None);
            cmd.ClearRenderTargetView(rtvs[1], &black, None);
            cmd.ClearRenderTargetView(rtvs[2], &black, None);
            cmd.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
        }
    }

    pub fn end_gbuffer_pass(&self, cmd: &ID3D12GraphicsCommandList) {
        let barriers = [
            transition_barrier(
                &self.albedo,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ),
            transition_barrier(
                &self.normal,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ),
            transition_barrier(
                &self.material,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ),
            transition_barrier(
                &self.depth,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            ),
        ];
        unsafe {
            cmd.ResourceBarrier(&barriers);
        }
    }

    pub fn begin_combine_pass(&self, cmd: &ID3D12GraphicsCommandList) {
        let barrier = transition_barrier(
            &self.hdr_accumulation,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe {
            cmd.ResourceBarrier(&[barrier]);
        }
    }

    pub fn clear_hdr(&self, cmd: &ID3D12GraphicsCommandList) {
        let rtv = self.hdr_rtv();
        let black = [0.0f32, 0.0, 0.0, 1.0];
        unsafe {
            cmd.ClearRenderTargetView(rttv_or_ref(&rtv), &black, None);
        }
    }

    pub fn end_combine_pass(&self, cmd: &ID3D12GraphicsCommandList) {
        let barrier = transition_barrier(
            &self.hdr_accumulation,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe {
            cmd.ResourceBarrier(&[barrier]);
        }
    }
}

#[inline]
fn rttv_or_ref(rtv: &D3D12_CPU_DESCRIPTOR_HANDLE) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    *rtv
}

/// Graphics pipeline rendering meshes into the 3 G-Buffer MRTs (Albedo, Normal, Material) and Depth
pub struct GBufferMeshPipeline {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl GBufferMeshPipeline {
    pub fn new(device: &ID3D12Device) -> WinResult<Self> {
        let mut camera = D3D12_ROOT_PARAMETER::default();
        camera.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        camera.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        camera.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 0,
        };

        let range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: u32::MAX,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let mut textures = D3D12_ROOT_PARAMETER::default();
        textures.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        textures.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        textures.Anonymous.DescriptorTable = D3D12_ROOT_DESCRIPTOR_TABLE {
            NumDescriptorRanges: 1,
            pDescriptorRanges: &range,
        };

        let mut push = D3D12_ROOT_PARAMETER::default();
        push.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        push.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        push.Anonymous.Constants = D3D12_ROOT_CONSTANTS {
            ShaderRegister: 1,
            RegisterSpace: 0,
            Num32BitValues: 8,
        };

        let mut transforms = D3D12_ROOT_PARAMETER::default();
        transforms.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        transforms.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        transforms.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 1,
        };

        let mut lights = D3D12_ROOT_PARAMETER::default();
        lights.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        lights.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        lights.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 1,
            RegisterSpace: 1,
        };

        let parameters = [camera, textures, push, transforms, lights];

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
            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
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
                        "[GBufferMeshPipeline] D3D12SerializeRootSignature error: {}",
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

        let source = include_str!("../shaders/geometry.hlsl");
        let compiler = ShaderCompiler::new().map_err(|e| {
            eprintln!("[GBufferMeshPipeline] ShaderCompiler::new failed: {:?}", e);
            failure()
        })?;

        let vs = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "vs_main",
                ShaderTarget::Vertex6_0,
            ))
            .map_err(|e| {
                eprintln!("[GBufferMeshPipeline] vs compile failed: {:?}", e);
                failure()
            })?;

        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_gbuffer",
                ShaderTarget::Pixel6_0,
            ))
            .map_err(|e| {
                eprintln!("[GBufferMeshPipeline] ps_gbuffer compile failed: {:?}", e);
                failure()
            })?;

        let input_elements = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::s!("POSITION"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::s!("NORMAL"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 12,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::s!("TEXCOORD"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 24,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::s!("TANGENT"),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 32,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];

        let mut blend = D3D12_BLEND_DESC::default();
        for i in 0..3 {
            blend.RenderTarget[i].BlendEnable = false.into();
            blend.RenderTarget[i].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        }

        let rasterizer = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_BACK,
            DepthClipEnable: true.into(),
            ..Default::default()
        };

        let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
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
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: input_elements.as_ptr(),
                NumElements: input_elements.len() as u32,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 3,
            RTVFormats: [
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM,
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
}

/// Deferred Combine & Lighting fullscreen graphics pipeline
pub struct DeferredCombinePipeline {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
    light_params_buffer: Option<ID3D12Resource>,
    dummy_csm_buffer: Option<ID3D12Resource>,
}

impl DeferredCombinePipeline {
    pub fn new(device: &ID3D12Device) -> WinResult<Self> {
        // Param 0: 32-bit push constants (b0) -> 8 dwords
        let mut push = D3D12_ROOT_PARAMETER::default();
        push.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        push.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        push.Anonymous.Constants = D3D12_ROOT_CONSTANTS {
            ShaderRegister: 0,
            RegisterSpace: 0,
            Num32BitValues: (std::mem::size_of::<PushCombineConstants>() / 4) as u32,
        };

        // Param 1: CameraConstants CBV (b1)
        let mut cam = D3D12_ROOT_PARAMETER::default();
        cam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        cam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        cam.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 1,
            RegisterSpace: 0,
        };

        // Param 2: LightParams CBV (b2)
        let mut light_params = D3D12_ROOT_PARAMETER::default();
        light_params.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        light_params.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        light_params.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 2,
            RegisterSpace: 0,
        };

        // Param 3: Dynamic Lights SRV (t0, space 1)
        let mut lights = D3D12_ROOT_PARAMETER::default();
        lights.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        lights.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        lights.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 1,
        };

        // Param 4: Bindless descriptor table (t0, space 0)
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

        // Param 5: CSM Split Uniforms CBV (b3)
        let mut csm_cbv = D3D12_ROOT_PARAMETER::default();
        csm_cbv.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        csm_cbv.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        csm_cbv.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 3,
            RegisterSpace: 0,
        };

        let parameters = [push, cam, light_params, lights, textures, csm_cbv];

        let samplers = [
            D3D12_STATIC_SAMPLER_DESC {
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
            },
            D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                MaxLOD: f32::MAX,
                ShaderRegister: 1,
                RegisterSpace: 0,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                ..Default::default()
            },
        ];

        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as u32,
            pParameters: parameters.as_ptr(),
            NumStaticSamplers: samplers.len() as u32,
            pStaticSamplers: samplers.as_ptr(),
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
                        "[DeferredCombinePipeline] D3D12SerializeRootSignature error: {}",
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

        let source = include_str!("../shaders/deferred_combine.hlsl");
        let compiler = ShaderCompiler::new().map_err(|e| {
            eprintln!(
                "[DeferredCombinePipeline] ShaderCompiler::new failed: {:?}",
                e
            );
            failure()
        })?;

        let vs = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "vs_deferred_combine",
                ShaderTarget::Vertex6_0,
            ))
            .map_err(|e| {
                eprintln!("[DeferredCombinePipeline] vs compile failed: {:?}", e);
                failure()
            })?;

        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_deferred_combine",
                ShaderTarget::Pixel6_0,
            ))
            .map_err(|e| {
                eprintln!("[DeferredCombinePipeline] ps compile failed: {:?}", e);
                failure()
            })?;

        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].BlendEnable = false.into();
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;

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
                DXGI_FORMAT_R16G16B16A16_FLOAT,
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

        Ok(Self {
            root,
            pso,
            light_params_buffer: None,
            dummy_csm_buffer: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &ID3D12Device,
        cmd: &ID3D12GraphicsCommandList,
        camera_va: u64,
        lights_va: u64,
        bindless_heap: &ID3D12DescriptorHeap,
        push: &PushCombineConstants,
        light_params: &GpuLightParams,
        csm_uniforms_va: Option<u64>,
        hdr_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
    ) -> WinResult<()> {
        if self.light_params_buffer.is_none() {
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            };
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
            let mut res = None;
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
            self.light_params_buffer = Some(res.ok_or_else(failure)?);
        }

        if self.dummy_csm_buffer.is_none() {
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 512,
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
            let mut res = None;
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
            self.dummy_csm_buffer = Some(res.ok_or_else(failure)?);
        }

        let lpb = self.light_params_buffer.as_ref().unwrap();
        unsafe {
            let mut mapped = std::ptr::null_mut();
            lpb.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                light_params as *const GpuLightParams as *const u8,
                mapped.cast(),
                std::mem::size_of::<GpuLightParams>(),
            );
            lpb.Unmap(0, None);
        }

        let light_params_va = unsafe { lpb.GetGPUVirtualAddress() };
        let csm_va = if let Some(va) = csm_uniforms_va {
            va
        } else {
            unsafe { self.dummy_csm_buffer.as_ref().unwrap().GetGPUVirtualAddress() }
        };

        unsafe {
            cmd.SetPipelineState(&self.pso);
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetDescriptorHeaps(&[Some(bindless_heap.clone())]);

            cmd.SetGraphicsRoot32BitConstants(
                0,
                (std::mem::size_of::<PushCombineConstants>() / 4) as u32,
                (push as *const PushCombineConstants).cast(),
                0,
            );

            cmd.SetGraphicsRootConstantBufferView(1, camera_va);
            cmd.SetGraphicsRootConstantBufferView(2, light_params_va);
            cmd.SetGraphicsRootShaderResourceView(3, lights_va);
            cmd.SetGraphicsRootDescriptorTable(
                4,
                bindless_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            cmd.SetGraphicsRootConstantBufferView(5, csm_va);

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

            cmd.OMSetRenderTargets(1, Some(&hdr_rtv), false, None);

            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.DrawInstanced(3, 1, 0, 0);
        }

        Ok(())
    }
}

/// Unified manager coordinating GBuffer allocation, mesh rasterization, and deferred lighting combine
pub struct GBufferManager {
    pub targets: Option<GBufferTargets>,
    pub mesh_pipeline: Option<GBufferMeshPipeline>,
    pub combine_pipeline: Option<DeferredCombinePipeline>,
    pub light_params: GpuLightParams,
}

impl Default for GBufferManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GBufferManager {
    pub fn new() -> Self {
        Self {
            targets: None,
            mesh_pipeline: None,
            combine_pipeline: None,
            light_params: GpuLightParams::default(),
        }
    }

    pub fn ensure_initialized(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
        bindless_allocator: &mut BindlessIndexAllocator,
        bindless_heap: &ID3D12DescriptorHeap,
        srv_descriptor_size: u32,
    ) -> WinResult<()> {
        let needs_recreate = match &self.targets {
            Some(t) => t.width != width || t.height != height,
            None => true,
        };

        if needs_recreate {
            self.targets = Some(GBufferTargets::new(
                device,
                width,
                height,
                bindless_allocator,
                bindless_heap,
                srv_descriptor_size,
            )?);
        }

        if self.mesh_pipeline.is_none() {
            self.mesh_pipeline = Some(GBufferMeshPipeline::new(device)?);
        }

        if self.combine_pipeline.is_none() {
            self.combine_pipeline = Some(DeferredCombinePipeline::new(device)?);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deferred_pipeline_graph_compilation() {
        let pipeline = DeferredPipeline::new(1920, 1080);
        let mut graph = RenderGraph::new();

        let (resources, schedule) = pipeline.build_graph(&mut graph);

        assert_eq!(schedule.pass_order.len(), 3);

        let lighting_pass_handle = schedule.pass_order[1];
        let barriers = schedule
            .pre_pass_barriers
            .get(&lighting_pass_handle)
            .expect("Missing lighting barriers");

        let transitions: Vec<_> = barriers
            .iter()
            .filter_map(|b| match b {
                crate::graph::ResourceBarrier::Transition {
                    resource,
                    state_before,
                    state_after,
                } => Some((*resource, *state_before, *state_after)),
                _ => None,
            })
            .collect();

        assert!(transitions.contains(&(
            resources.albedo,
            ResourceState::RenderTarget,
            ResourceState::NonPixelShaderResource
        )));
        assert!(transitions.contains(&(
            resources.normal,
            ResourceState::RenderTarget,
            ResourceState::NonPixelShaderResource
        )));
        assert!(transitions.contains(&(
            resources.material,
            ResourceState::RenderTarget,
            ResourceState::NonPixelShaderResource
        )));
        assert!(transitions.contains(&(
            resources.depth,
            ResourceState::DepthWrite,
            ResourceState::NonPixelShaderResource
        )));
        assert!(transitions.contains(&(
            resources.hdr_output,
            ResourceState::Common,
            ResourceState::UnorderedAccess
        )));
    }

    #[test]
    fn test_deferred_shader_offline_compilation() {
        let compiler =
            ShaderCompiler::new().expect("Windows SDK dxc.exe not found on build host");

        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let shaders_dir = manifest_dir.join("shaders");

        let (vs_bc, ps_bc, cs_bc) = DeferredPipeline::compile_shaders(&compiler, &shaders_dir)
            .expect("Offline DXC compilation of GBuffer and Lighting shaders failed");

        assert!(!vs_bc.is_empty(), "GBuffer VS bytecode must not be empty");
        assert!(!ps_bc.is_empty(), "GBuffer PS bytecode must not be empty");
        assert!(
            !cs_bc.is_empty(),
            "Deferred Lighting CS bytecode must not be empty"
        );
        assert_eq!(
            &vs_bc[0..4],
            b"DXBC",
            "Bytecode must be standard DXIL DXBC container"
        );
        assert_eq!(
            &ps_bc[0..4],
            b"DXBC",
            "Bytecode must be standard DXIL DXBC container"
        );
        assert_eq!(
            &cs_bc[0..4],
            b"DXBC",
            "Bytecode must be standard DXIL DXBC container"
        );
    }
}
