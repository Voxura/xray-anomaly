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
    srv_heap: ID3D12DescriptorHeap,
}

impl GeometryPipeline {
    /// The legacy deferred signature remains separate from this Phase 4 ABI.
    fn new(device: &ID3D12Device) -> WinResult<Self> {
        let range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 16,
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
        let parameters = [camera, textures];
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
            NumParameters: 2,
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
        let compiler = ShaderCompiler::new().map_err(|_| failure())?;
        let vs = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "vs_main",
                ShaderTarget::Vertex6_6,
            ))
            .map_err(|_| failure())?;
        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_main",
                ShaderTarget::Pixel6_6,
            ))
            .map_err(|_| failure())?;
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
        desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        desc.DepthStencilState.FrontFace = stencil;
        desc.DepthStencilState.BackFace = stencil;
        desc.SampleMask = u32::MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc = DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        };
        let pso = unsafe { device.CreateGraphicsPipelineState(&desc) };
        unsafe {
            ManuallyDrop::drop(&mut desc.pRootSignature);
        }
        let pso = pso?;

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
            srv_heap,
        })
    }
}

#[derive(Default)]
pub(crate) struct GeometryState {
    meshes: MeshRegistry<Mesh>,
    camera: Option<UploadBuffer>,
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

    /// Caller validates frame, mesh and camera before recording any commands.
    pub fn draw(
        &mut self,
        device: &ID3D12Device,
        cmd: &ID3D12GraphicsCommandList,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
        handle: MeshHandle,
    ) -> WinResult<()> {
        if self.pipeline.is_none() {
            self.pipeline = Some(GeometryPipeline::new(device)?);
        }
        let pipeline = self.pipeline.as_ref().unwrap();
        let mesh = self.meshes.get(handle).unwrap();
        let camera = self.camera.as_ref().unwrap();
        // Clone BEFORE command recording so CPU destruction cannot invalidate a draw.
        self.recording_resources.extend([
            mesh.vertices.0.clone(),
            mesh.indices.0.clone(),
            camera.0.clone(),
        ]);
        unsafe {
            cmd.SetPipelineState(&pipeline.pso);
            cmd.SetGraphicsRootSignature(&pipeline.root);
            cmd.SetDescriptorHeaps(&[Some(pipeline.srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, camera.0.GetGPUVirtualAddress());
            cmd.SetGraphicsRootDescriptorTable(
                1,
                pipeline.srv_heap.GetGPUDescriptorHandleForHeapStart(),
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
            cmd.OMSetRenderTargets(1, Some(&rtv), false, None);
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
