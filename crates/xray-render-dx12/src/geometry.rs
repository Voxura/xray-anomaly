//! Phase 4 indexed geometry. Upload resources are immutable and retained until
//! the submission fence completes, including meshes destroyed during recording.

use std::mem::ManuallyDrop;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Error, HRESULT, Result as WinResult, s};

use crate::geometry_abi::{CameraConstants, MeshHandle, MeshRegistry, VERTEX_STRIDE};
use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

struct UploadBuffer(ID3D12Resource);

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushDrawConstants {
    pub albedo_index: u32,
    pub normal_index: u32,
    pub material_index: u32,
    pub transform_index: u32,
    pub custom_param0: u32,
    pub custom_param1: u32,
    pub custom_param2: u32,
    pub custom_param3: u32,
}

impl Default for PushDrawConstants {
    fn default() -> Self {
        Self {
            albedo_index: u32::MAX,
            normal_index: u32::MAX,
            material_index: u32::MAX,
            transform_index: u32::MAX,
            custom_param0: u32::MAX,
            custom_param1: u32::MAX,
            custom_param2: u32::MAX,
            custom_param3: u32::MAX,
        }
    }
}

impl UploadBuffer {
    fn new(device: &ID3D12Device, bytes: &[u8]) -> WinResult<Self> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes.len() as u64,
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
        unsafe {
            let mut resource: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut resource,
            )?;
            let resource = resource.ok_or_else(failure)?;
            let mut mapped = std::ptr::null_mut();
            resource.Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 0 }),
                Some(&mut mapped),
            )?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
            resource.Unmap(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: bytes.len(),
                }),
            );
            Ok(Self(resource))
        }
    }
}

struct Mesh {
    vertices: UploadBuffer,
    indices: UploadBuffer,
    vertex_view: D3D12_VERTEX_BUFFER_VIEW,
    index_view: D3D12_INDEX_BUFFER_VIEW,
    index_count: u32,
}

struct GeometryPipeline {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    depth_pso: ID3D12PipelineState,
    equal_depth_pso: ID3D12PipelineState,
    srv_heap: ID3D12DescriptorHeap,
}

impl GeometryPipeline {
    /// The legacy deferred signature remains separate from this Phase 4 ABI.
    fn new(device: &ID3D12Device) -> WinResult<Self> {
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
        // The contract permits clamp/wrap; choose linear wrap on all axes.
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
            D3D12SerializeRootSignature(
                &root_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut blob,
                None,
            )?;
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
        let compiler = match ShaderCompiler::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[GeometryPipeline] ShaderCompiler::new failed: {:?}", e);
                return Err(failure());
            }
        };
        let vs = match compiler.compile(&ShaderCompileRequest::new(
            source,
            "vs_main",
            ShaderTarget::Vertex6_0,
        )) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[GeometryPipeline] vs compile failed: {:?}", e);
                return Err(failure());
            }
        };
        let ps = match compiler.compile(&ShaderCompileRequest::new(
            source,
            "ps_main",
            ShaderTarget::Pixel6_0,
        )) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[GeometryPipeline] ps compile failed: {:?}", e);
                return Err(failure());
            }
        };
        let vs_depth = match compiler.compile(&ShaderCompileRequest::new(
            source,
            "vs_depth_main",
            ShaderTarget::Vertex6_0,
        )) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[GeometryPipeline] vs_depth compile failed: {:?}", e);
                return Err(failure());
            }
        };
        let layout = [
            (s!("POSITION"), DXGI_FORMAT_R32G32B32_FLOAT, 0),
            (s!("NORMAL"), DXGI_FORMAT_R32G32B32_FLOAT, 12),
            (s!("TEXCOORD"), DXGI_FORMAT_R32G32_FLOAT, 24),
            (s!("TANGENT"), DXGI_FORMAT_R32G32B32_FLOAT, 32),
        ]
        .map(|(semantic, format, offset)| D3D12_INPUT_ELEMENT_DESC {
            SemanticName: semantic,
            Format: format,
            AlignedByteOffset: offset,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            ..Default::default()
        });
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
            pInputElementDescs: layout.as_ptr(),
            NumElements: 4,
        };
        desc.RasterizerState = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            ..Default::default()
        };
        desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        desc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            StencilEnable: false.into(),
            ..Default::default()
        };
        desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        desc.SampleMask = u32::MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc = DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        };
        // Prepare Depth-only prepass / shadow mapping pipeline state (null pixel shader)
        let mut depth_desc = desc.clone();
        depth_desc.pRootSignature = ManuallyDrop::new(Some(root.clone()));
        depth_desc.VS = D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs_depth.as_ptr().cast(),
            BytecodeLength: vs_depth.len(),
        };
        depth_desc.PS = D3D12_SHADER_BYTECODE {
            pShaderBytecode: std::ptr::null(),
            BytecodeLength: 0,
        };
        depth_desc.NumRenderTargets = 0;
        depth_desc.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
        depth_desc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            StencilEnable: false.into(),
            ..Default::default()
        };

        // Prepare Equal depth-test pipeline for early-Z prepass color shading
        let mut equal_desc = desc.clone();
        equal_desc.pRootSignature = ManuallyDrop::new(Some(root.clone()));
        equal_desc.DepthStencilState = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_EQUAL,
            StencilEnable: false.into(),
            ..Default::default()
        };

        let pso = unsafe { device.CreateGraphicsPipelineState(&desc) };
        unsafe {
            ManuallyDrop::drop(&mut desc.pRootSignature);
        }
        let pso = match pso {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[GeometryPipeline] CreateGraphicsPipelineState (main) failed: {:?}", e);
                return Err(e);
            }
        };

        let depth_pso = unsafe { device.CreateGraphicsPipelineState(&depth_desc) };
        unsafe {
            ManuallyDrop::drop(&mut depth_desc.pRootSignature);
        }
        let depth_pso = match depth_pso {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[GeometryPipeline] CreateGraphicsPipelineState (depth) failed: {:?}", e);
                return Err(e);
            }
        };

        let equal_depth_pso = unsafe { device.CreateGraphicsPipelineState(&equal_desc) };
        unsafe {
            ManuallyDrop::drop(&mut equal_desc.pRootSignature);
        }
        let equal_depth_pso = match equal_depth_pso {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[GeometryPipeline] CreateGraphicsPipelineState (equal_depth) failed: {:?}", e);
                return Err(e);
            }
        };

        let srv_heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 16,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?
        };
        // Valid null SRVs until a future texture-upload API supplies real textures.
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        unsafe {
            let start = srv_heap.GetCPUDescriptorHandleForHeapStart();
            let step = device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            for i in 0..16 {
                device.CreateShaderResourceView(
                    None::<&ID3D12Resource>,
                    Some(&srv),
                    D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: start.ptr + i * step,
                    },
                );
            }
        }
        Ok(Self {
            root,
            pso,
            depth_pso,
            equal_depth_pso,
            srv_heap,
        })
    }
}

#[derive(Default)]
pub(crate) struct GeometryState {
    meshes: MeshRegistry<Mesh>,
    camera: Option<UploadBuffer>,
    transforms: Option<UploadBuffer>,
    default_transform: Option<UploadBuffer>,
    lights: Option<UploadBuffer>,
    default_lights: Option<UploadBuffer>,
    pipeline: Option<GeometryPipeline>,
    recording_resources: Vec<ID3D12Resource>,
    submitted_resources: Vec<(u64, Vec<ID3D12Resource>)>,
}

impl GeometryState {
    pub fn create_mesh(
        &mut self,
        device: &ID3D12Device,
        vertices: &[u8],
        indices: &[u8],
        count: u32,
        format: u32,
    ) -> WinResult<MeshHandle> {
        let vertices_buffer = UploadBuffer::new(device, vertices)?;
        let indices_buffer = UploadBuffer::new(device, indices)?;
        let mesh = Mesh {
            vertex_view: D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: unsafe { vertices_buffer.0.GetGPUVirtualAddress() },
                SizeInBytes: vertices.len() as u32,
                StrideInBytes: VERTEX_STRIDE,
            },
            index_view: D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: unsafe { indices_buffer.0.GetGPUVirtualAddress() },
                SizeInBytes: indices.len() as u32,
                Format: DXGI_FORMAT(format as i32),
            },
            vertices: vertices_buffer,
            indices: indices_buffer,
            index_count: count,
        };
        self.meshes.insert(mesh).ok_or_else(failure)
    }

    pub fn destroy_mesh(&mut self, handle: MeshHandle) -> bool {
        self.meshes.remove(handle).is_some()
    }

    pub fn has_mesh(&self, handle: MeshHandle) -> bool {
        self.meshes.get(handle).is_some()
    }

    pub fn has_camera(&self) -> bool {
        self.camera.is_some()
    }

    pub fn camera_gpu_va(&self) -> Option<u64> {
        self.camera.as_ref().map(|c| unsafe { c.0.GetGPUVirtualAddress() })
    }

    pub fn track_resource(&mut self, res: ID3D12Resource) {
        self.recording_resources.push(res);
    }

    pub fn update_camera(
        &mut self,
        device: &ID3D12Device,
        constants: &CameraConstants,
    ) -> WinResult<()> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (constants as *const CameraConstants).cast::<u8>(),
                std::mem::size_of::<CameraConstants>(),
            )
        };
        self.camera = Some(UploadBuffer::new(device, bytes)?);
        Ok(())
    }

    /// Update instance or bone transform matrices
    pub fn update_transforms(
        &mut self,
        device: &ID3D12Device,
        matrices: &[[[f32; 4]; 4]],
    ) -> WinResult<()> {
        if matrices.is_empty() {
            self.transforms = None;
            return Ok(());
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(
                matrices.as_ptr().cast::<u8>(),
                std::mem::size_of_val(matrices),
            )
        };
        self.transforms = Some(UploadBuffer::new(device, bytes)?);
        Ok(())
    }

    /// Update dynamic light structured buffer
    pub fn update_lights(
        &mut self,
        device: &ID3D12Device,
        lights: &[crate::lighting::GpuLight],
    ) -> WinResult<()> {
        if lights.is_empty() {
            self.lights = None;
            return Ok(());
        }
        let bytes = crate::lighting::GpuLight::as_bytes(lights);
        self.lights = Some(UploadBuffer::new(device, bytes)?);
        Ok(())
    }

    /// Retrieve GPU virtual address for dynamic lights buffer (or default dummy buffer)
    pub fn lights_gpu_va(&mut self, device: &ID3D12Device) -> WinResult<u64> {
        if self.default_lights.is_none() {
            let dummy = [0u8; 64];
            self.default_lights = Some(UploadBuffer::new(device, &dummy)?);
        }
        let default_lt = self.default_lights.as_ref().unwrap();
        let lt_buf = self.lights.as_ref().unwrap_or(default_lt);
        unsafe { Ok(lt_buf.0.GetGPUVirtualAddress()) }
    }

    /// Caller validates frame, mesh and camera before recording any commands.
    pub fn draw(
        &mut self,
        device: &ID3D12Device,
        cmd: &ID3D12GraphicsCommandList,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        width: u32,
        height: u32,
        handle: MeshHandle,
        bindless_heap: Option<&ID3D12DescriptorHeap>,
        push_constants: Option<&PushDrawConstants>,
    ) -> WinResult<()> {
        if self.pipeline.is_none() {
            self.pipeline = Some(GeometryPipeline::new(device)?);
        }
        let pipeline = self.pipeline.as_ref().unwrap();
        let mesh = self.meshes.get(handle).unwrap();
        let camera = self.camera.as_ref().unwrap();
        let srv_heap = bindless_heap.unwrap_or(&pipeline.srv_heap);
        let default_push = PushDrawConstants::default();
        let push = push_constants.unwrap_or(&default_push);

        if self.default_transform.is_none() {
            let identity: [[[f32; 4]; 4]; 1] = [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]];
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    identity.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&identity),
                )
            };
            self.default_transform = Some(UploadBuffer::new(device, bytes)?);
        }
        let default_tf = self.default_transform.as_ref().unwrap();
        let tf_buf = self.transforms.as_ref().unwrap_or(default_tf);

        if self.default_lights.is_none() {
            let dummy = [0u8; 64];
            self.default_lights = Some(UploadBuffer::new(device, &dummy)?);
        }
        let default_lt = self.default_lights.as_ref().unwrap();
        let lt_buf = self.lights.as_ref().unwrap_or(default_lt);

        // Clone BEFORE command recording so CPU destruction cannot invalidate a draw.
        self.recording_resources.extend([
            mesh.vertices.0.clone(),
            mesh.indices.0.clone(),
            camera.0.clone(),
            tf_buf.0.clone(),
            lt_buf.0.clone(),
        ]);
        unsafe {
            cmd.SetPipelineState(&pipeline.pso);
            cmd.SetGraphicsRootSignature(&pipeline.root);
            cmd.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, camera.0.GetGPUVirtualAddress());
            cmd.SetGraphicsRootDescriptorTable(
                1,
                srv_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            cmd.SetGraphicsRoot32BitConstants(
                2,
                8,
                (push as *const PushDrawConstants).cast(),
                0,
            );
            cmd.SetGraphicsRootShaderResourceView(3, tf_buf.0.GetGPUVirtualAddress());
            cmd.SetGraphicsRootShaderResourceView(4, lt_buf.0.GetGPUVirtualAddress());
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
            cmd.IASetVertexBuffers(0, Some(&[mesh.vertex_view]));
            cmd.IASetIndexBuffer(Some(&mesh.index_view));
            cmd.DrawIndexedInstanced(mesh.index_count, 1, 0, 0, 0);
        }
        Ok(())
    }

    /// Record a fast depth-only prepass or shadow pass draw call (null pixel shader)
    pub fn draw_depth(
        &mut self,
        device: &ID3D12Device,
        cmd: &ID3D12GraphicsCommandList,
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
        handle: MeshHandle,
        transform_index: u32,
    ) -> WinResult<()> {
        self.draw_depth_viewport(
            device,
            cmd,
            dsv,
            [0, 0, width, height],
            handle,
            transform_index,
        )
    }

    /// Record a fast depth-only prepass or shadow pass draw call into an explicit `[x, y, w, h]` atlas viewport
    pub fn draw_depth_viewport(
        &mut self,
        device: &ID3D12Device,
        cmd: &ID3D12GraphicsCommandList,
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
        viewport: [u32; 4],
        handle: MeshHandle,
        transform_index: u32,
    ) -> WinResult<()> {
        if self.pipeline.is_none() {
            self.pipeline = Some(GeometryPipeline::new(device)?);
        }
        let pipeline = self.pipeline.as_ref().unwrap();
        let mesh = self.meshes.get(handle).unwrap();
        let camera = self.camera.as_ref().unwrap();

        if self.default_transform.is_none() {
            let identity: [[[f32; 4]; 4]; 1] = [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]];
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    identity.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&identity),
                )
            };
            self.default_transform = Some(UploadBuffer::new(device, bytes)?);
        }
        let default_tf = self.default_transform.as_ref().unwrap();
        let tf_buf = self.transforms.as_ref().unwrap_or(default_tf);

        self.recording_resources.extend([
            mesh.vertices.0.clone(),
            mesh.indices.0.clone(),
            camera.0.clone(),
            tf_buf.0.clone(),
        ]);
        let push = PushDrawConstants {
            transform_index,
            ..Default::default()
        };
        let [vx, vy, vw, vh] = viewport;
        unsafe {
            cmd.SetPipelineState(&pipeline.depth_pso);
            cmd.SetGraphicsRootSignature(&pipeline.root);
            cmd.SetGraphicsRootConstantBufferView(0, camera.0.GetGPUVirtualAddress());
            cmd.SetGraphicsRoot32BitConstants(
                2,
                8,
                (&push as *const PushDrawConstants).cast(),
                0,
            );
            cmd.SetGraphicsRootShaderResourceView(3, tf_buf.0.GetGPUVirtualAddress());
            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: vx as f32,
                TopLeftY: vy as f32,
                Width: vw as f32,
                Height: vh as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            cmd.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: vx as i32,
                top: vy as i32,
                right: (vx + vw) as i32,
                bottom: (vy + vh) as i32,
            }]);
            cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.IASetVertexBuffers(0, Some(&[mesh.vertex_view]));
            cmd.IASetIndexBuffer(Some(&mesh.index_view));
            cmd.DrawIndexedInstanced(mesh.index_count, 1, 0, 0, 0);
        }
        Ok(())
    }

    /// Record a draw call with EQUAL depth testing (zero depth write) for early-Z shading
    pub fn draw_equal_depth(
        &mut self,
        device: &ID3D12Device,
        cmd: &ID3D12GraphicsCommandList,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
        handle: MeshHandle,
        bindless_heap: Option<&ID3D12DescriptorHeap>,
        push_constants: Option<&PushDrawConstants>,
    ) -> WinResult<()> {
        if self.pipeline.is_none() {
            self.pipeline = Some(GeometryPipeline::new(device)?);
        }
        let pipeline = self.pipeline.as_ref().unwrap();
        let mesh = self.meshes.get(handle).unwrap();
        let camera = self.camera.as_ref().unwrap();
        let srv_heap = bindless_heap.unwrap_or(&pipeline.srv_heap);
        let default_push = PushDrawConstants::default();
        let push = push_constants.unwrap_or(&default_push);

        if self.default_transform.is_none() {
            let identity: [[[f32; 4]; 4]; 1] = [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]];
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    identity.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&identity),
                )
            };
            self.default_transform = Some(UploadBuffer::new(device, bytes)?);
        }
        let default_tf = self.default_transform.as_ref().unwrap();
        let tf_buf = self.transforms.as_ref().unwrap_or(default_tf);

        if self.default_lights.is_none() {
            let dummy = [0u8; 64];
            self.default_lights = Some(UploadBuffer::new(device, &dummy)?);
        }
        let default_lt = self.default_lights.as_ref().unwrap();
        let lt_buf = self.lights.as_ref().unwrap_or(default_lt);

        self.recording_resources.extend([
            mesh.vertices.0.clone(),
            mesh.indices.0.clone(),
            camera.0.clone(),
            tf_buf.0.clone(),
            lt_buf.0.clone(),
        ]);
        unsafe {
            cmd.SetPipelineState(&pipeline.equal_depth_pso);
            cmd.SetGraphicsRootSignature(&pipeline.root);
            cmd.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, camera.0.GetGPUVirtualAddress());
            cmd.SetGraphicsRootDescriptorTable(
                1,
                srv_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            cmd.SetGraphicsRoot32BitConstants(
                2,
                8,
                (push as *const PushDrawConstants).cast(),
                0,
            );
            cmd.SetGraphicsRootShaderResourceView(3, tf_buf.0.GetGPUVirtualAddress());
            cmd.SetGraphicsRootShaderResourceView(4, lt_buf.0.GetGPUVirtualAddress());
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
            cmd.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.IASetVertexBuffers(0, Some(&[mesh.vertex_view]));
            cmd.IASetIndexBuffer(Some(&mesh.index_view));
            cmd.DrawIndexedInstanced(mesh.index_count, 1, 0, 0, 0);
        }
        Ok(())
    }

    /// Record a G-Buffer MRT draw call (writes Albedo, Normal, Material, Depth)
    pub fn draw_gbuffer(
        &mut self,
        device: &ID3D12Device,
        cmd: &ID3D12GraphicsCommandList,
        gbuffer_pso: &ID3D12PipelineState,
        gbuffer_root: &ID3D12RootSignature,
        srv_heap: &ID3D12DescriptorHeap,
        rtvs: &[D3D12_CPU_DESCRIPTOR_HANDLE; 3],
        dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
        handle: MeshHandle,
        push_constants: &PushDrawConstants,
    ) -> WinResult<()> {
        let mesh = self.meshes.get(handle).unwrap();
        let camera = self.camera.as_ref().unwrap();

        if self.default_transform.is_none() {
            let identity: [[[f32; 4]; 4]; 1] = [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]];
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    identity.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&identity),
                )
            };
            self.default_transform = Some(UploadBuffer::new(device, bytes)?);
        }
        let default_tf = self.default_transform.as_ref().unwrap();
        let tf_buf = self.transforms.as_ref().unwrap_or(default_tf);

        if self.default_lights.is_none() {
            let dummy = [0u8; 64];
            self.default_lights = Some(UploadBuffer::new(device, &dummy)?);
        }
        let default_lt = self.default_lights.as_ref().unwrap();
        let lt_buf = self.lights.as_ref().unwrap_or(default_lt);

        self.recording_resources.extend([
            mesh.vertices.0.clone(),
            mesh.indices.0.clone(),
            camera.0.clone(),
            tf_buf.0.clone(),
            lt_buf.0.clone(),
        ]);
        unsafe {
            cmd.SetPipelineState(gbuffer_pso);
            cmd.SetGraphicsRootSignature(gbuffer_root);
            cmd.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, camera.0.GetGPUVirtualAddress());
            cmd.SetGraphicsRootDescriptorTable(
                1,
                srv_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            cmd.SetGraphicsRoot32BitConstants(
                2,
                8,
                (push_constants as *const PushDrawConstants).cast(),
                0,
            );
            cmd.SetGraphicsRootShaderResourceView(3, tf_buf.0.GetGPUVirtualAddress());
            cmd.SetGraphicsRootShaderResourceView(4, lt_buf.0.GetGPUVirtualAddress());
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
            cmd.OMSetRenderTargets(3, Some(rtvs.as_ptr()), false, Some(&dsv));
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.IASetVertexBuffers(0, Some(&[mesh.vertex_view]));
            cmd.IASetIndexBuffer(Some(&mesh.index_view));
            cmd.DrawIndexedInstanced(mesh.index_count, 1, 0, 0, 0);
        }
        Ok(())
    }

    /// Called before submission. u64::MAX keeps resources alive if Signal fails.
    pub fn prepare_submission(&mut self) {
        self.submitted_resources
            .push((u64::MAX, std::mem::take(&mut self.recording_resources)));
    }

    pub fn tag_submission(&mut self, fence: u64) {
        self.submitted_resources.last_mut().unwrap().0 = fence;
    }

    pub fn retire(&mut self, completed: u64) {
        // UINT64_MAX is D3D12's device-removal sentinel, not a completed fence.
        if completed != u64::MAX {
            self.submitted_resources
                .retain(|(fence, _)| *fence > completed);
        }
    }
}
