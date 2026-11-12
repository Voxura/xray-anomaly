//! DirectX 12 Instanced Detail Foliage & Vegetation Renderer
//!
//! Renders dense instanced vegetation (grass clumps, flowers, shrubs, pebbles)
//! with procedural SSFX wind waving animation, distance dissolve, and alpha-tested cutouts.

use windows::core::{Error, Result as WinResult, HRESULT};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};
use xray_formats::details::DetailMesh;

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

/// 24-byte GPU detail mesh vertex
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuDetailVertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    /// Normalized height factor [0.0..1.0] from root to tip for wind displacement
    pub height_fraction: f32,
}

const _: () = assert!(std::mem::size_of::<GpuDetailVertex>() == 24);

/// 48-byte 16-byte aligned GPU detail instance record (3 float4s)
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuDetailInstance {
    pub position: [f32; 3],
    pub scale: f32,
    pub rotation_y: f32,
    pub alpha: f32,
    pub hemi_light: f32,
    pub sun_light: f32,
    pub color_tint: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuDetailInstance>() == 48);

impl Default for GpuDetailInstance {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            scale: 1.0,
            rotation_y: 0.0,
            alpha: 1.0,
            hemi_light: 1.0,
            sun_light: 1.0,
            color_tint: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// 32-byte push constants for detail rendering
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetailPushConstants {
    pub wind_direction: [f32; 2],
    pub wind_speed: f32,
    pub wind_strength: f32,
    pub time_seconds: f32,
    pub texture_index: u32,
    pub alpha_threshold: f32,
    pub custom_param: f32,
}

const _: () = assert!(std::mem::size_of::<DetailPushConstants>() == 32);

impl Default for DetailPushConstants {
    fn default() -> Self {
        Self {
            wind_direction: [1.0, 0.0],
            wind_speed: 1.5,
            wind_strength: 0.15,
            time_seconds: 0.0,
            texture_index: u32::MAX,
            alpha_threshold: 0.33,
            custom_param: 0.0,
        }
    }
}

/// Direct3D 12 Detail Foliage graphics pipeline
pub struct DetailRenderer {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl DetailRenderer {
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
            Num32BitValues: (std::mem::size_of::<DetailPushConstants>() / 4) as u32,
        };

        // Param 3: Mesh vertices ByteAddressBuffer SRV (t0, space 1)
        let mut vertices = D3D12_ROOT_PARAMETER::default();
        vertices.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        vertices.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        vertices.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 1,
        };

        // Param 4: Instances ByteAddressBuffer SRV (t0, space 2)
        let mut instances = D3D12_ROOT_PARAMETER::default();
        instances.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        instances.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        instances.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 2,
        };

        let parameters = [camera, textures, push, vertices, instances];

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
                    eprintln!(
                        "[DetailRenderer] D3D12SerializeRootSignature error: {}",
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

        let source = include_str!("../shaders/details.hlsl");
        let compiler = ShaderCompiler::new().map_err(|e| {
            eprintln!("[DetailRenderer] ShaderCompiler::new failed: {:?}", e);
            failure()
        })?;

        let vs = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "vs_detail",
                ShaderTarget::Vertex6_0,
            ))
            .map_err(|e| {
                eprintln!("[DetailRenderer] vs compile failed: {:?}", e);
                failure()
            })?;

        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_detail",
                ShaderTarget::Pixel6_0,
            ))
            .map_err(|e| {
                eprintln!("[DetailRenderer] ps compile failed: {:?}", e);
                failure()
            })?;

        let mut blend = D3D12_BLEND_DESC::default();
        let rt_blend = &mut blend.RenderTarget[0];
        rt_blend.BlendEnable = false.into(); // Alpha-tested cutouts write directly to depth buffer
        rt_blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;

        let rasterizer = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE, // Double-sided vegetation foliage
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

    /// Render instanced foliage objects
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        camera_va: u64,
        bindless_heap: &ID3D12DescriptorHeap,
        vertices_va: u64,
        instances_va: u64,
        vertex_count: u32,
        instance_count: u32,
        push: &DetailPushConstants,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        width: u32,
        height: u32,
    ) {
        if vertex_count == 0 || instance_count == 0 {
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
                (std::mem::size_of::<DetailPushConstants>() / 4) as u32,
                (push as *const DetailPushConstants).cast(),
                0,
            );

            // Param 3: Mesh vertices SRV
            cmd.SetGraphicsRootShaderResourceView(3, vertices_va);

            // Param 4: Instances SRV
            cmd.SetGraphicsRootShaderResourceView(4, instances_va);

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
            cmd.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
    }
}

/// High-level manager coordinating detail mesh geometry and GPU instances
pub struct DetailManager {
    mesh_buffer: Option<ID3D12Resource>,
    mesh_capacity_bytes: usize,
    instance_buffer: Option<ID3D12Resource>,
    instance_capacity_bytes: usize,
    instances: Vec<GpuDetailInstance>,
}

impl Default for DetailManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DetailManager {
    pub fn new() -> Self {
        Self {
            mesh_buffer: None,
            mesh_capacity_bytes: 0,
            instance_buffer: None,
            instance_capacity_bytes: 0,
            instances: Vec::new(),
        }
    }

    /// Convert a parsed `DetailMesh` into an indexed triangle list of `GpuDetailVertex`
    pub fn build_gpu_mesh(dm: &DetailMesh) -> Vec<GpuDetailVertex> {
        let min_y = dm
            .vertices
            .iter()
            .map(|v| v.position[1])
            .fold(f32::INFINITY, f32::min);
        let max_y = dm
            .vertices
            .iter()
            .map(|v| v.position[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let height_span = (max_y - min_y).max(0.001);

        let mut out = Vec::with_capacity(dm.indices.len());
        for &idx in &dm.indices {
            let v = &dm.vertices[idx as usize];
            let height_fraction = ((v.position[1] - min_y) / height_span).clamp(0.0, 1.0);
            out.push(GpuDetailVertex {
                position: v.position,
                uv: [v.u, v.v],
                height_fraction,
            });
        }
        out
    }

    /// Upload mesh vertices to GPU
    pub fn upload_mesh(
        &mut self,
        device: &ID3D12Device,
        vertices: &[GpuDetailVertex],
    ) -> WinResult<u64> {
        if vertices.is_empty() {
            return Ok(0);
        }
        let needed_bytes = std::mem::size_of_val(vertices).max(256);

        if self.mesh_buffer.is_none() || self.mesh_capacity_bytes < needed_bytes {
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
            self.mesh_buffer = Some(res.ok_or_else(failure)?);
            self.mesh_capacity_bytes = needed_bytes;
        }

        let res = self.mesh_buffer.as_ref().unwrap();
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

    /// Add an instance to the active batch
    pub fn add_instance(&mut self, instance: GpuDetailInstance) {
        self.instances.push(instance);
    }

    /// Clear all active instances
    pub fn clear_instances(&mut self) {
        self.instances.clear();
    }

    /// Total active instance count
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Upload active instance records to GPU
    pub fn upload_instances(&mut self, device: &ID3D12Device) -> WinResult<u64> {
        if self.instances.is_empty() {
            return Ok(0);
        }
        let needed_bytes = std::mem::size_of_val(self.instances.as_slice()).max(256);

        if self.instance_buffer.is_none() || self.instance_capacity_bytes < needed_bytes {
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
            self.instance_buffer = Some(res.ok_or_else(failure)?);
            self.instance_capacity_bytes = needed_bytes;
        }

        let res = self.instance_buffer.as_ref().unwrap();
        unsafe {
            let mut mapped = std::ptr::null_mut();
            res.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(
                self.instances.as_ptr() as *const u8,
                mapped.cast(),
                std::mem::size_of_val(self.instances.as_slice()),
            );
            res.Unmap(0, None);
            Ok(res.GetGPUVirtualAddress())
        }
    }
}
