//! High-Level DirectX 12 Renderer Facade
//!
//! Provides the primary runtime entry point for the X-Ray Monolith Engine,
//! integrating device context, swapchain presentation, multi-threaded command recording,
//! bindless descriptor management, transient memory aliasing, and pipeline caching.

use std::mem::ManuallyDrop;
use windows::core::{Interface, Result as WinResult, HRESULT};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12Fence, ID3D12GraphicsCommandList, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_FENCE_FLAG_NONE, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_VIEWPORT,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

use crate::allocators::MultiThreadedCommandManager;
use crate::bindless::{BindlessIndexAllocator, DescriptorIndex};
use crate::device::Dx12DeviceContext;
use crate::geometry::GeometryState;
use crate::heaps::TransientHeapManager;
use crate::pipeline::PipelineManager;
use crate::swapchain::SwapChainManager;
use crate::texture::{GpuTexture, TextureRegistry};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    Ok = 0,
    Lost = 1,
    NeedReset = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderStats {
    pub frame_index: u64,
    pub width: u32,
    pub height: u32,
    pub completed_gpu_fence: u64,
    pub last_signaled_fence: u64,
}

/// Unified DirectX 12 Renderer instance
pub struct Dx12Renderer {
    pub device_ctx: Dx12DeviceContext,
    pub swapchain: Option<SwapChainManager>,
    pub command_mgr: MultiThreadedCommandManager,
    pub pipeline_mgr: PipelineManager,
    pub bindless_allocator: BindlessIndexAllocator,
    pub bindless_heap: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap,
    pub srv_descriptor_size: u32,
    pub transient_heap: TransientHeapManager,
    pub fence: ID3D12Fence,
    pub fence_event: HANDLE,
    pub fence_value: u64,
    pub frame_index: u64,
    pub hwnd: Option<HWND>,
    pub width: u32,
    pub height: u32,
    pub in_frame: bool,
    pub current_cmd_list: Option<ID3D12GraphicsCommandList>,
    pub shadow_map: Option<crate::shadow::DirectionalShadowMap>,
    pub particle_renderer: Option<crate::particle::ParticleSystemRenderer>,
    pub sky_renderer: Option<crate::sky::SkyRenderer>,
    pub clouds_renderer: Option<crate::clouds::CloudRenderer>,
    pub rain_renderer: Option<crate::rain::RainRenderer>,
    pub rain_manager: Option<crate::rain::RainManager>,
    pub wallmark_renderer: Option<crate::wallmarks::WallmarkRenderer>,
    pub wallmarks_manager: Option<crate::wallmarks::WallmarksManager>,
    pub detail_renderer: Option<crate::details::DetailRenderer>,
    pub detail_manager: Option<crate::details::DetailManager>,
    pub postprocess_renderer: Option<crate::postprocess::PostprocessRenderer>,
    pub postprocess_manager: Option<crate::postprocess::PostprocessManager>,
    pub gbuffer: Option<crate::deferred::GBufferManager>,
    pub ui_canvas_pipeline: Option<crate::ui_canvas::UiCanvasPipeline>,
    pub ui_canvas: crate::ui_canvas::UiCanvas,
    pub(crate) geometry: GeometryState,
    pub(crate) textures: TextureRegistry,
}

impl Dx12Renderer {
    /// Initialize DirectX 12 renderer subsystems
    pub fn new(worker_count: usize) -> WinResult<Self> {
        let device_ctx = Dx12DeviceContext::new(false)?;
        let command_mgr = MultiThreadedCommandManager::new(
            &device_ctx.device,
            worker_count.max(1),
            D3D12_COMMAND_LIST_TYPE_DIRECT,
        )?;
        let pipeline_mgr = PipelineManager::new(&device_ctx.device)?;
        let bindless_allocator = BindlessIndexAllocator::new(100_000);
        let transient_heap = TransientHeapManager::new(&device_ctx.device, 128 * 1024 * 1024)?; // 128 MB

        let bindless_heap: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap = unsafe {
            device_ctx.device.CreateDescriptorHeap(
                &windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 100_000,
                    Flags: windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                },
            )?
        };
        let srv_descriptor_size = unsafe {
            device_ctx.device.GetDescriptorHandleIncrementSize(
                windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            )
        };

        let fence: ID3D12Fence = unsafe {
            device_ctx.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?
        };
        let fence_event = unsafe { CreateEventW(None, false, false, None)? };

        Ok(Self {
            device_ctx,
            swapchain: None,
            command_mgr,
            pipeline_mgr,
            bindless_allocator,
            bindless_heap,
            srv_descriptor_size,
            transient_heap,
            fence,
            fence_event,
            fence_value: 0,
            frame_index: 0,
            hwnd: None,
            width: 0,
            height: 0,
            in_frame: false,
            current_cmd_list: None,
            shadow_map: None,
            particle_renderer: None,
            sky_renderer: None,
            clouds_renderer: None,
            rain_renderer: None,
            rain_manager: None,
            wallmark_renderer: None,
            wallmarks_manager: None,
            detail_renderer: None,
            detail_manager: None,
            postprocess_renderer: None,
            postprocess_manager: None,
            gbuffer: None,
            ui_canvas_pipeline: None,
            ui_canvas: crate::ui_canvas::UiCanvas::default(),
            geometry: GeometryState::default(),
            textures: TextureRegistry::default(),
        })
    }

    /// Attach a native Win32 window and initialize presentation swapchain
    pub fn attach_window(&mut self, hwnd: HWND, width: u32, height: u32) -> WinResult<()> {
        if self.in_frame {
            return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)));
        }
        self.wait_for_gpu()?;
        let swapchain = SwapChainManager::new(
            &self.device_ctx.factory,
            &self.device_ctx.direct_queue,
            &self.device_ctx.device,
            hwnd,
            width,
            height,
        )?;

        self.hwnd = Some(hwnd);
        self.width = width;
        self.height = height;
        self.swapchain = Some(swapchain);
        Ok(())
    }

    /// Resize swapchain backbuffers upon window dimension change
    pub fn resize(&mut self, width: u32, height: u32) -> WinResult<()> {
        if self.in_frame {
            return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)));
        }
        self.wait_for_gpu()?;
        if let Some(sc) = &mut self.swapchain {
            sc.resize(&self.device_ctx.device, width, height)?;
            self.width = width;
            self.height = height;
        }
        Ok(())
    }

    /// Begin frame recording: resets command allocator and transitions backbuffer to RenderTarget
    pub fn begin_frame(&mut self) -> WinResult<&ID3D12GraphicsCommandList> {
        if self.swapchain.is_none() || self.in_frame {
            return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)));
        }

        let completed_fence = unsafe { self.fence.GetCompletedValue() };
        self.geometry.retire(completed_fence);
        self.textures.retire(completed_fence);
        let allocator = self
            .command_mgr
            .worker(0)
            .unwrap()
            .begin_frame(self.frame_index, completed_fence)
            .map_err(|_| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;

        let cmd = self.command_mgr.acquire_command_list(allocator)?;

        // Transition backbuffer from PRESENT to RENDER_TARGET
        let backbuffer = self.swapchain.as_ref().unwrap().current_backbuffer_resource();
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(backbuffer) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_PRESENT,
                    StateAfter: D3D12_RESOURCE_STATE_RENDER_TARGET,
                }),
            },
        };
        unsafe {
            cmd.ResourceBarrier(&[barrier]);
            cmd.SetDescriptorHeaps(&[Some(self.bindless_heap.clone())]);
        }

        self.current_cmd_list = Some(cmd);
        self.in_frame = true;
        Ok(self.current_cmd_list.as_ref().unwrap())
    }

    /// Clear the current backbuffer with an RGBA color and clear depth to 1.0
    pub fn clear_backbuffer(&self, color: [f32; 4]) {
        if let (Some(cmd), Some(sc)) = (&self.current_cmd_list, &self.swapchain) {
            let rtv = sc.current_rtv_handle();
            let dsv = sc.dsv_handle();
            unsafe {
                cmd.ClearRenderTargetView(rtv, &color, None);
                cmd.ClearDepthStencilView(
                    dsv,
                    windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAG_DEPTH,
                    1.0,
                    0,
                    None,
                );
            }
        }
    }

    /// Clear the depth and stencil buffers explicitly
    pub fn clear_depth_stencil(&self, depth: f32, stencil: u8) {
        if let (Some(cmd), Some(sc)) = (&self.current_cmd_list, &self.swapchain) {
            let dsv = sc.dsv_handle();
            unsafe {
                cmd.ClearDepthStencilView(
                    dsv,
                    windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAG_DEPTH
                        | windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAG_STENCIL,
                    depth,
                    stencil,
                    None,
                );
            }
        }
    }

    /// End frame: transitions backbuffer to Present, executes commands, presents swapchain, and signals fence
    pub fn end_frame(&mut self, vsync: bool) -> WinResult<()> {
        let cmd = match self.current_cmd_list.take() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };

        // Transition backbuffer from RENDER_TARGET to PRESENT
        let backbuffer = self.swapchain.as_ref().unwrap().current_backbuffer_resource();
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: unsafe { std::mem::transmute_copy(backbuffer) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: D3D12_RESOURCE_STATE_PRESENT,
                }),
            },
        };
        unsafe {
            cmd.ResourceBarrier(&[barrier]);
            cmd.Close()?;
            // Retain immutable draw buffers even if presentation or signaling fails.
            self.geometry.prepare_submission();
            self.device_ctx
                .direct_queue
                .ExecuteCommandLists(&[Some(cmd.cast()?)]);
        }

        // Signal GPU fence
        self.fence_value += 1;
        unsafe {
            self.device_ctx
                .direct_queue
                .Signal(&self.fence, self.fence_value)?;
        }
        self.command_mgr
            .worker(0)
            .unwrap()
            .tag_submission_fence(self.frame_index, self.fence_value);
        self.geometry.tag_submission(self.fence_value);
        self.textures.tag_submission(self.fence_value);

        // Retire bindless descriptors whose GPU execution has completed
        let completed = unsafe { self.fence.GetCompletedValue() };
        self.bindless_allocator.retire_fences(completed);
        self.geometry.retire(completed);
        self.textures.retire(completed);

        self.command_mgr.release_command_list(cmd);
        self.frame_index += 1;
        self.in_frame = false;
        // Submission is tracked before Present: failure must not release live buffers.
        self.swapchain.as_ref().unwrap().present(vsync)
    }

    /// Stall CPU until GPU finishes all submitted work
    pub fn wait_for_gpu(&mut self) -> WinResult<()> {
        self.fence_value += 1;
        unsafe {
            self.device_ctx
                .direct_queue
                .Signal(&self.fence, self.fence_value)?;
            self.fence
                .SetEventOnCompletion(self.fence_value, self.fence_event)?;
            WaitForSingleObject(self.fence_event, INFINITE);
        }
        let completed = unsafe { self.fence.GetCompletedValue() };
        self.geometry.retire(completed);
        self.textures.retire(completed);
        Ok(())
    }

    /// Upload a DDS texture into GPU VRAM (default heap) and register into the bindless descriptor heap.
    /// Automatically deduplicates repeated texture requests across sub-meshes and materials.
    pub fn load_texture_from_dds(&mut self, name: &str, bytes: &[u8]) -> WinResult<DescriptorIndex> {
        let normalized_name = name.to_ascii_lowercase().replace('\\', "/");
        if let Some(existing) = self.textures.get_by_name(&normalized_name) {
            return Ok(existing.descriptor_index);
        }

        let dds = crate::texture::parse_dds(bytes).map_err(|_| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;

        let descriptor_index = self
            .bindless_allocator
            .allocate()
            .ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;

        let (texture_res, _staging_res) = if let Some(cmd) = &self.current_cmd_list {
            let (tex, staging) = crate::texture::upload_dds_texture(&self.device_ctx.device, cmd, &dds)?;
            self.textures.track_staging(staging);
            (tex, ())
        } else {
            // Out of frame upload: execute synchronously
            let allocator = self.command_mgr.worker(0).unwrap().begin_frame(self.frame_index, unsafe { self.fence.GetCompletedValue() })
                .map_err(|_| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
            let cmd = self.command_mgr.acquire_command_list(allocator)?;
            let (tex, staging) = crate::texture::upload_dds_texture(&self.device_ctx.device, &cmd, &dds)?;
            unsafe {
                cmd.Close()?;
                self.device_ctx.direct_queue.ExecuteCommandLists(&[Some(cmd.cast()?)]);
            }
            self.command_mgr.release_command_list(cmd);
            self.wait_for_gpu()?;
            drop(staging);
            (tex, ())
        };

        let heap_start = unsafe { self.bindless_heap.GetCPUDescriptorHandleForHeapStart() };
        let cpu_handle = windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: heap_start.ptr + (descriptor_index.0 as usize) * (self.srv_descriptor_size as usize),
        };

        crate::texture::register_srv(
            &self.device_ctx.device,
            &texture_res,
            dds.format,
            dds.mip_count,
            cpu_handle,
        );

        let gpu_tex = GpuTexture {
            resource: texture_res,
            descriptor_index,
            width: dds.width,
            height: dds.height,
            mip_count: dds.mip_count,
            format: dds.format,
        };

        self.textures.insert(normalized_name, gpu_tex);
        Ok(descriptor_index)
    }

    /// Retrieve loaded texture metadata and GPU resource by name (case- and slash-normalized)
    pub fn get_texture(&self, name: &str) -> Option<&GpuTexture> {
        let normalized = name.to_ascii_lowercase().replace('\\', "/");
        self.textures.get_by_name(&normalized)
    }

    /// Retrieve loaded texture metadata and GPU resource by bindless descriptor index
    pub fn get_texture_by_index(&self, index: DescriptorIndex) -> Option<&GpuTexture> {
        self.textures.get_by_index(index)
    }

    /// Create a GPU vertex & index buffer mesh from typed XRay vertices and 16-bit indices
    pub fn create_mesh(
        &mut self,
        vertices: &[crate::geometry_abi::XRayVertex],
        indices: &[u16],
    ) -> WinResult<crate::geometry_abi::MeshHandle> {
        let v_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr().cast::<u8>(),
                std::mem::size_of_val(vertices),
            )
        };
        let i_bytes = unsafe {
            std::slice::from_raw_parts(
                indices.as_ptr().cast::<u8>(),
                std::mem::size_of_val(indices),
            )
        };
        self.geometry.create_mesh(
            &self.device_ctx.device,
            v_bytes,
            i_bytes,
            indices.len() as u32,
            crate::geometry_abi::INDEX_FORMAT_U16,
        )
    }

    /// Update view/projection camera constants
    pub fn update_camera(
        &mut self,
        camera: &crate::geometry_abi::CameraConstants,
    ) -> WinResult<()> {
        self.geometry.update_camera(&self.device_ctx.device, camera)
    }

    /// Update GPU instance or bone transform palette matrices
    pub fn update_transforms(
        &mut self,
        matrices: &[[[f32; 4]; 4]],
    ) -> WinResult<()> {
        self.geometry.update_transforms(&self.device_ctx.device, matrices)
    }

    /// Record a mesh draw call onto the active frame command list
    pub fn draw_mesh(&mut self, handle: crate::geometry_abi::MeshHandle) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();
        self.geometry.draw(
            &self.device_ctx.device,
            cmd,
            rtv,
            Some(dsv),
            self.width,
            self.height,
            handle,
            Some(&self.bindless_heap),
            None,
        )
    }

    /// Record a textured mesh draw call with a bindless albedo texture descriptor index
    pub fn draw_textured_mesh(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        albedo_index: crate::bindless::DescriptorIndex,
    ) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();
        let push_constants = crate::geometry::PushDrawConstants {
            albedo_index: albedo_index.0,
            ..Default::default()
        };
        self.geometry.draw(
            &self.device_ctx.device,
            cmd,
            rtv,
            Some(dsv),
            self.width,
            self.height,
            handle,
            Some(&self.bindless_heap),
            Some(&push_constants),
        )
    }

    /// Record a draw call with an albedo texture and instance/bone transform index
    pub fn draw_textured_mesh_with_transform(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        albedo_index: crate::bindless::DescriptorIndex,
        transform_index: u32,
    ) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();
        let push_constants = crate::geometry::PushDrawConstants {
            albedo_index: albedo_index.0,
            transform_index,
            ..Default::default()
        };
        self.geometry.draw(
            &self.device_ctx.device,
            cmd,
            rtv,
            Some(dsv),
            self.width,
            self.height,
            handle,
            Some(&self.bindless_heap),
            Some(&push_constants),
        )
    }

    /// Record a draw call with optional albedo, normal/bump, and material texture descriptor indices
    pub fn draw_textured_material_mesh(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        albedo_index: Option<crate::bindless::DescriptorIndex>,
        normal_index: Option<crate::bindless::DescriptorIndex>,
        material_index: Option<crate::bindless::DescriptorIndex>,
    ) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();
        let push_constants = crate::geometry::PushDrawConstants {
            albedo_index: albedo_index.map(|i| i.0).unwrap_or(u32::MAX),
            normal_index: normal_index.map(|i| i.0).unwrap_or(u32::MAX),
            material_index: material_index.map(|i| i.0).unwrap_or(u32::MAX),
            ..Default::default()
        };
        self.geometry.draw(
            &self.device_ctx.device,
            cmd,
            rtv,
            Some(dsv),
            self.width,
            self.height,
            handle,
            Some(&self.bindless_heap),
            Some(&push_constants),
        )
    }

    /// Initialize a high-resolution directional sunlight shadow map and allocate its bindless SRV
    pub fn init_shadow_map(&mut self, resolution: u32) -> WinResult<DescriptorIndex> {
        let shadow = crate::shadow::DirectionalShadowMap::new(
            &self.device_ctx.device,
            resolution,
            &mut self.bindless_allocator,
            &self.bindless_heap,
            self.srv_descriptor_size,
            self.fence_value,
        )?;
        let idx = shadow.descriptor_index();
        self.shadow_map = Some(shadow);
        Ok(idx)
    }

    /// Retrieve the allocated bindless descriptor index of the directional shadow map
    pub fn shadow_map_index(&self) -> Option<DescriptorIndex> {
        self.shadow_map.as_ref().map(|s| s.descriptor_index())
    }

    /// Update light matrices for directional sunlight projection
    pub fn set_light_camera(&mut self, light_view_proj: [[f32; 4]; 4], light_dir: [f32; 4]) {
        if let Some(shadow) = &mut self.shadow_map {
            shadow.set_light_camera(light_view_proj, light_dir);
        }
    }

    /// Update dynamic light structured buffer
    pub fn update_lights(&mut self, lights: &[crate::lighting::GpuLight]) -> WinResult<()> {
        self.geometry.update_lights(&self.device_ctx.device, lights)
    }

    /// Record a draw call with dynamic point/spot lights evaluated
    pub fn draw_mesh_with_lights(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        albedo_index: Option<crate::bindless::DescriptorIndex>,
        transform_index: u32,
        num_lights: u32,
    ) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();
        let push_constants = crate::geometry::PushDrawConstants {
            albedo_index: albedo_index.map(|i| i.0).unwrap_or(u32::MAX),
            transform_index,
            custom_param2: num_lights,
            ..Default::default()
        };
        self.geometry.draw(
            &self.device_ctx.device,
            cmd,
            rtv,
            Some(dsv),
            self.width,
            self.height,
            handle,
            Some(&self.bindless_heap),
            Some(&push_constants),
        )
    }

    /// Record a draw call rendering the infinite skybox with weather cycle texture blending
    pub fn draw_sky(&mut self, push: &crate::sky::SkyPushConstants) -> WinResult<()> {
        if self.sky_renderer.is_none() {
            self.sky_renderer = Some(crate::sky::SkyRenderer::new(&self.device_ctx.device)?);
        }
        let sky = self.sky_renderer.as_ref().unwrap();
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let camera_va = self
            .geometry
            .camera_gpu_va()
            .ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();

        sky.render(
            cmd,
            camera_va,
            &self.bindless_heap,
            push,
            rtv,
            Some(dsv),
            self.width,
            self.height,
        );
        Ok(())
    }

    /// Record a draw call rendering the dual-layer scrolling clouds dome
    pub fn draw_clouds(&mut self, push: &crate::clouds::CloudsPushConstants) -> WinResult<()> {
        if self.clouds_renderer.is_none() {
            self.clouds_renderer = Some(crate::clouds::CloudRenderer::new(&self.device_ctx.device)?);
        }
        let clouds = self.clouds_renderer.as_ref().unwrap();
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let camera_va = self
            .geometry
            .camera_gpu_va()
            .ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();

        clouds.render(
            cmd,
            camera_va,
            &self.bindless_heap,
            push,
            rtv,
            Some(dsv),
            self.width,
            self.height,
        );
        Ok(())
    }

    /// Advance rain simulation state around camera volume
    pub fn step_rain(
        &mut self,
        cam_pos: [f32; 3],
        wind_direction: f32,
        wind_velocity: f32,
        rain_density: f32,
        dt: f32,
    ) {
        if self.rain_manager.is_none() {
            self.rain_manager = Some(crate::rain::RainManager::new(crate::rain::DEFAULT_MAX_RAIN_DROPS));
        }
        let rain_mgr = self.rain_manager.as_mut().unwrap();
        rain_mgr.step(cam_pos, wind_direction, wind_velocity, rain_density, dt);
    }

    /// Access mutable reference to rain manager
    pub fn rain_manager_mut(&mut self) -> &mut crate::rain::RainManager {
        if self.rain_manager.is_none() {
            self.rain_manager = Some(crate::rain::RainManager::new(crate::rain::DEFAULT_MAX_RAIN_DROPS));
        }
        self.rain_manager.as_mut().unwrap()
    }

    /// Record a draw call rendering instanced rain streaks
    pub fn draw_rain(&mut self, push: &crate::rain::RainPushConstants) -> WinResult<()> {
        if self.rain_renderer.is_none() {
            self.rain_renderer = Some(crate::rain::RainRenderer::new(&self.device_ctx.device)?);
        }
        if self.rain_manager.is_none() {
            self.rain_manager = Some(crate::rain::RainManager::new(crate::rain::DEFAULT_MAX_RAIN_DROPS));
        }
        let num_drops = self.rain_manager.as_ref().unwrap().active_count();
        if num_drops == 0 {
            return Ok(());
        }
        let raindrops_va = self.rain_manager.as_mut().unwrap().upload(&self.device_ctx.device)?;
        let rain = self.rain_renderer.as_ref().unwrap();
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let camera_va = self
            .geometry
            .camera_gpu_va()
            .ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();

        rain.render(
            cmd,
            camera_va,
            &self.bindless_heap,
            raindrops_va,
            num_drops,
            push,
            rtv,
            Some(dsv),
            self.width,
            self.height,
        );
        Ok(())
    }

    /// Access mutable reference to wallmarks manager
    pub fn wallmarks_manager_mut(&mut self) -> &mut crate::wallmarks::WallmarksManager {
        if self.wallmarks_manager.is_none() {
            self.wallmarks_manager = Some(crate::wallmarks::WallmarksManager::new());
        }
        self.wallmarks_manager.as_mut().unwrap()
    }

    /// Record a draw call rendering static wallmark decals
    pub fn draw_wallmarks(
        &mut self,
        vertices: &[crate::wallmarks::GpuWallmarkVertex],
        push: &crate::wallmarks::WallmarkPushConstants,
    ) -> WinResult<()> {
        if vertices.is_empty() {
            return Ok(());
        }
        if self.wallmark_renderer.is_none() {
            self.wallmark_renderer = Some(crate::wallmarks::WallmarkRenderer::new(&self.device_ctx.device)?);
        }
        if self.wallmarks_manager.is_none() {
            self.wallmarks_manager = Some(crate::wallmarks::WallmarksManager::new());
        }
        let vertices_va = self
            .wallmarks_manager
            .as_mut()
            .unwrap()
            .upload(&self.device_ctx.device, vertices)?;
        let wallmark = self.wallmark_renderer.as_ref().unwrap();
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let camera_va = self
            .geometry
            .camera_gpu_va()
            .ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();

        wallmark.render(
            cmd,
            camera_va,
            &self.bindless_heap,
            vertices_va,
            vertices.len() as u32,
            push,
            rtv,
            Some(dsv),
            self.width,
            self.height,
        );
        Ok(())
    }

    /// Access mutable reference to detail manager
    pub fn detail_manager_mut(&mut self) -> &mut crate::details::DetailManager {
        if self.detail_manager.is_none() {
            self.detail_manager = Some(crate::details::DetailManager::new());
        }
        self.detail_manager.as_mut().unwrap()
    }

    /// Record a draw call rendering instanced detail foliage
    pub fn draw_details(
        &mut self,
        mesh_vertices: &[crate::details::GpuDetailVertex],
        push: &crate::details::DetailPushConstants,
    ) -> WinResult<()> {
        if mesh_vertices.is_empty() {
            return Ok(());
        }
        if self.detail_renderer.is_none() {
            self.detail_renderer = Some(crate::details::DetailRenderer::new(&self.device_ctx.device)?);
        }
        if self.detail_manager.is_none() {
            self.detail_manager = Some(crate::details::DetailManager::new());
        }
        let mgr = self.detail_manager.as_mut().unwrap();
        let instance_count = mgr.instance_count() as u32;
        if instance_count == 0 {
            return Ok(());
        }
        let vertices_va = mgr.upload_mesh(&self.device_ctx.device, mesh_vertices)?;
        let instances_va = mgr.upload_instances(&self.device_ctx.device)?;

        let detail = self.detail_renderer.as_ref().unwrap();
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let camera_va = self
            .geometry
            .camera_gpu_va()
            .ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();

        detail.render(
            cmd,
            camera_va,
            &self.bindless_heap,
            vertices_va,
            instances_va,
            mesh_vertices.len() as u32,
            instance_count,
            push,
            rtv,
            Some(dsv),
            self.width,
            self.height,
        );
        Ok(())
    }

    /// Access mutable reference to post-processing manager
    pub fn postprocess_manager_mut(&mut self) -> &mut crate::postprocess::PostprocessManager {
        if self.postprocess_manager.is_none() {
            self.postprocess_manager = Some(crate::postprocess::PostprocessManager::new());
        }
        self.postprocess_manager.as_mut().unwrap()
    }

    /// Record a draw call rendering the post-processing pass (tonemapping, bloom, vignette)
    pub fn draw_postprocess(
        &mut self,
        push: &crate::postprocess::PostprocessPushConstants,
    ) -> WinResult<()> {
        if self.postprocess_renderer.is_none() {
            self.postprocess_renderer =
                Some(crate::postprocess::PostprocessRenderer::new(&self.device_ctx.device)?);
        }
        if self.postprocess_manager.is_none() {
            self.postprocess_manager = Some(crate::postprocess::PostprocessManager::new());
        }
        let (tonemap_va, img_correct_va) = self
            .postprocess_manager
            .as_mut()
            .unwrap()
            .upload_constants(&self.device_ctx.device)?;
        let postprocess = self.postprocess_renderer.as_ref().unwrap();
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        postprocess.render(
            cmd,
            &self.bindless_heap,
            tonemap_va,
            img_correct_va,
            push,
            rtv,
            self.width,
            self.height,
        );
        Ok(())
    }

    /// Ensure G-Buffer targets and pipelines are initialized for current window dimensions
    pub fn ensure_gbuffer(&mut self) -> WinResult<()> {
        if self.gbuffer.is_none() {
            self.gbuffer = Some(crate::deferred::GBufferManager::new());
        }
        let gbuffer = self.gbuffer.as_mut().unwrap();
        gbuffer.ensure_initialized(
            &self.device_ctx.device,
            self.width,
            self.height,
            &mut self.bindless_allocator,
            &self.bindless_heap,
            self.srv_descriptor_size,
        )
    }

    /// Access mutable reference to G-Buffer manager
    pub fn gbuffer_manager_mut(&mut self) -> WinResult<&mut crate::deferred::GBufferManager> {
        self.ensure_gbuffer()?;
        Ok(self.gbuffer.as_mut().unwrap())
    }

    /// Begin G-Buffer MRT rasterization pass
    pub fn begin_gbuffer_pass(&mut self) -> WinResult<()> {
        self.ensure_gbuffer()?;
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let gbuffer = self.gbuffer.as_ref().unwrap();
        let targets = gbuffer.targets.as_ref().unwrap();

        targets.begin_gbuffer_pass(cmd);
        targets.clear_gbuffer(cmd);
        Ok(())
    }

    /// Draw a mesh into the 3 G-Buffer MRTs (Albedo, Normal, Material) and Depth
    pub fn draw_gbuffer_mesh(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        albedo_index: Option<crate::bindless::DescriptorIndex>,
        normal_index: Option<crate::bindless::DescriptorIndex>,
        material_index: Option<crate::bindless::DescriptorIndex>,
        transform_index: u32,
    ) -> WinResult<()> {
        self.ensure_gbuffer()?;
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let gbuffer = self.gbuffer.as_ref().unwrap();
        let pipeline = gbuffer.mesh_pipeline.as_ref().unwrap();
        let targets = gbuffer.targets.as_ref().unwrap();

        let rtvs = targets.gbuffer_rtvs();
        let dsv = targets.dsv_handle();

        let push = crate::geometry::PushDrawConstants {
            albedo_index: albedo_index.map(|i| i.0).unwrap_or(u32::MAX),
            normal_index: normal_index.map(|i| i.0).unwrap_or(u32::MAX),
            material_index: material_index.map(|i| i.0).unwrap_or(u32::MAX),
            transform_index,
            ..Default::default()
        };

        self.geometry.draw_gbuffer(
            &self.device_ctx.device,
            cmd,
            &pipeline.pso,
            &pipeline.root,
            &self.bindless_heap,
            &rtvs,
            dsv,
            self.width,
            self.height,
            handle,
            &push,
        )
    }

    /// Complete G-Buffer MRT rasterization pass
    pub fn end_gbuffer_pass(&mut self) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        if let Some(gbuffer) = self.gbuffer.as_ref() {
            if let Some(targets) = gbuffer.targets.as_ref() {
                targets.end_gbuffer_pass(cmd);
            }
        }
        Ok(())
    }

    /// Execute the deferred lighting combine pass into the HDR accumulation target and return its bindless SRV
    pub fn execute_deferred_combine(
        &mut self,
        light_params: &crate::deferred::GpuLightParams,
        shadow_map_idx: Option<crate::bindless::DescriptorIndex>,
        num_lights: u32,
    ) -> WinResult<crate::bindless::DescriptorIndex> {
        self.ensure_gbuffer()?;
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let camera_va = self
            .geometry
            .camera_gpu_va()
            .ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        let lt_va = self.geometry.lights_gpu_va(&self.device_ctx.device)?;

        let gbuffer = self.gbuffer.as_mut().unwrap();
        let targets = gbuffer.targets.as_ref().unwrap();

        targets.begin_combine_pass(cmd);
        targets.clear_hdr(cmd);

        let push = crate::deferred::PushCombineConstants {
            gbuffer_albedo_idx: targets.albedo_srv.0,
            gbuffer_normal_idx: targets.normal_srv.0,
            gbuffer_material_idx: targets.material_srv.0,
            gbuffer_depth_idx: targets.depth_srv.0,
            shadow_map_idx: shadow_map_idx.map(|s| s.0).unwrap_or(u32::MAX),
            num_lights,
            screen_res_inv: [1.0 / self.width as f32, 1.0 / self.height as f32],
        };

        let hdr_rtv = targets.hdr_rtv();
        let width = self.width;
        let height = self.height;
        let hdr_srv = targets.hdr_srv;

        let combine = gbuffer.combine_pipeline.as_mut().unwrap();
        combine.render(
            &self.device_ctx.device,
            cmd,
            camera_va,
            lt_va,
            &self.bindless_heap,
            &push,
            light_params,
            hdr_rtv,
            width,
            height,
        )?;

        let targets = self.gbuffer.as_ref().unwrap().targets.as_ref().unwrap();
        targets.end_combine_pass(cmd);

        Ok(hdr_srv)
    }

    /// Begin the directional sunlight shadow map rendering pass
    pub fn begin_shadow_pass(&mut self) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let shadow = match self.shadow_map.as_mut() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        shadow.begin_shadow_pass(cmd);
        Ok(())
    }

    /// Draw a mesh into the directional shadow depth map
    pub fn draw_shadow_mesh(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        transform_index: u32,
    ) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let shadow = match self.shadow_map.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let dsv = shadow.dsv_handle();
        let res = shadow.resolution;
        self.geometry.draw_depth(
            &self.device_ctx.device,
            cmd,
            dsv,
            res,
            res,
            handle,
            transform_index,
        )
    }

    /// Complete the directional sunlight shadow pass and transition shadow map to PIXEL_SHADER_RESOURCE
    pub fn end_shadow_pass(&mut self) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let shadow = match self.shadow_map.as_mut() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        shadow.end_shadow_pass(cmd);
        Ok(())
    }

    /// Record a draw call with an albedo texture and bindless shadow map index
    pub fn draw_textured_mesh_with_shadow(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        albedo_index: crate::bindless::DescriptorIndex,
        shadow_map_index: crate::bindless::DescriptorIndex,
    ) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();
        let push_constants = crate::geometry::PushDrawConstants {
            albedo_index: albedo_index.0,
            custom_param0: shadow_map_index.0,
            ..Default::default()
        };
        self.geometry.draw(
            &self.device_ctx.device,
            cmd,
            rtv,
            Some(dsv),
            self.width,
            self.height,
            handle,
            Some(&self.bindless_heap),
            Some(&push_constants),
        )
    }

    /// Record a fast depth-only prepass draw call (null pixel shader)
    pub fn draw_depth_prepass(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        transform_index: u32,
    ) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let dsv = sc.dsv_handle();
        self.geometry.draw_depth(
            &self.device_ctx.device,
            cmd,
            dsv,
            self.width,
            self.height,
            handle,
            transform_index,
        )
    }

    /// Record a draw call with EQUAL depth testing (zero depth write) for early-Z shading
    pub fn draw_mesh_equal_depth(
        &mut self,
        handle: crate::geometry_abi::MeshHandle,
        albedo_index: Option<crate::bindless::DescriptorIndex>,
        normal_index: Option<crate::bindless::DescriptorIndex>,
        transform_index: u32,
        shadow_map_index: Option<crate::bindless::DescriptorIndex>,
    ) -> WinResult<()> {
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();
        let push_constants = crate::geometry::PushDrawConstants {
            albedo_index: albedo_index.map(|i| i.0).unwrap_or(u32::MAX),
            normal_index: normal_index.map(|i| i.0).unwrap_or(u32::MAX),
            transform_index,
            custom_param0: shadow_map_index.map(|i| i.0).unwrap_or(u32::MAX),
            ..Default::default()
        };
        self.geometry.draw_equal_depth(
            &self.device_ctx.device,
            cmd,
            rtv,
            dsv,
            self.width,
            self.height,
            handle,
            Some(&self.bindless_heap),
            Some(&push_constants),
        )
    }

    /// Record an instanced particle billboard draw call with camera alignment and blending
    pub fn draw_particles(
        &mut self,
        instances: &[crate::particle::GpuParticleInstance],
        additive: bool,
    ) -> WinResult<()> {
        if instances.is_empty() {
            return Ok(());
        }
        if self.particle_renderer.is_none() {
            self.particle_renderer = Some(crate::particle::ParticleSystemRenderer::new(&self.device_ctx.device)?);
        }
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();
        let dsv = sc.dsv_handle();

        // Upload particle instance records to transient upload buffer
        let byte_len = std::mem::size_of_val(instances);
        let heap_props = windows::Win32::Graphics::Direct3D12::D3D12_HEAP_PROPERTIES {
            Type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE_UPLOAD,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let desc = windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC {
            Dimension: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: byte_len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut upload_res: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> = None;
        unsafe {
            self.device_ctx.device.CreateCommittedResource(
                &heap_props,
                windows::Win32::Graphics::Direct3D12::D3D12_HEAP_FLAG_NONE,
                &desc,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut upload_res,
            )?;
        }
        let upload_res = upload_res.ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        unsafe {
            let mut mapped = std::ptr::null_mut();
            upload_res.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(instances.as_ptr() as *const u8, mapped.cast(), byte_len);
            upload_res.Unmap(0, None);
        }

        let camera_va = self.geometry.camera_gpu_va().ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        let particle_va = unsafe { upload_res.GetGPUVirtualAddress() };

        let renderer = self.particle_renderer.as_ref().unwrap();
        renderer.draw(
            cmd,
            rtv,
            Some(dsv),
            camera_va,
            &self.bindless_heap,
            particle_va,
            instances.len() as u32,
            additive,
            self.width,
            self.height,
        );

        // Keep buffer alive across frame submission
        self.geometry.track_resource(upload_res);
        Ok(())
    }

    /// Queue a solid color 2D UI rectangle `[x, y, width, height]`
    pub fn ui_fill_rect(&mut self, rect: [f32; 4], color: [f32; 4]) {
        self.ui_canvas.fill_rect(rect, color);
    }

    /// Queue a textured 2D UI rectangle
    pub fn ui_draw_textured_rect(
        &mut self,
        rect: [f32; 4],
        uv: [f32; 4],
        texture_index: crate::bindless::DescriptorIndex,
        color: [f32; 4],
    ) {
        self.ui_canvas.draw_textured_rect(rect, uv, texture_index, color);
    }

    /// Queue a 2D crosshair centered at `center`
    pub fn ui_draw_crosshair(
        &mut self,
        center: [f32; 2],
        size: f32,
        gap: f32,
        thickness: f32,
        color: [f32; 4],
    ) {
        self.ui_canvas.draw_crosshair(center, size, gap, thickness, color);
    }

    /// Queue a horizontal progress/meter bar (e.g. Health, Stamina, Radiation)
    pub fn ui_draw_meter_bar(
        &mut self,
        rect: [f32; 4],
        fraction: f32,
        bg_color: [f32; 4],
        bar_color: [f32; 4],
    ) {
        self.ui_canvas.draw_meter_bar(rect, fraction, bg_color, bar_color);
    }

    /// Queue a batch of font glyph quads with a specific font atlas texture and text color
    pub fn ui_draw_glyph_quads(
        &mut self,
        glyphs: &[crate::ui_canvas::GlyphQuad],
        font_texture_index: crate::bindless::DescriptorIndex,
        color: [f32; 4],
    ) {
        self.ui_canvas.draw_glyph_quads(glyphs, font_texture_index, color);
    }

    /// Flush all queued UI Canvas elements and render directly to active swapchain backbuffer
    pub fn ui_flush(
        &mut self,
        virtual_width: u32,
        virtual_height: u32,
    ) -> WinResult<()> {
        if self.ui_canvas.is_empty() {
            return Ok(());
        }
        if self.ui_canvas_pipeline.is_none() {
            self.ui_canvas_pipeline = Some(crate::ui_canvas::UiCanvasPipeline::new(&self.device_ctx.device)?);
        }
        let cmd = match self.current_cmd_list.as_ref() {
            Some(c) => c,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let sc = match self.swapchain.as_ref() {
            Some(s) => s,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))),
        };
        let rtv = sc.current_rtv_handle();

        let quads = &self.ui_canvas.quads;
        let byte_len = std::mem::size_of_val(quads.as_slice());
        let heap_props = windows::Win32::Graphics::Direct3D12::D3D12_HEAP_PROPERTIES {
            Type: windows::Win32::Graphics::Direct3D12::D3D12_HEAP_TYPE_UPLOAD,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let desc = windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC {
            Dimension: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: byte_len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: windows::Win32::Graphics::Direct3D12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut upload_res: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> = None;
        unsafe {
            self.device_ctx.device.CreateCommittedResource(
                &heap_props,
                windows::Win32::Graphics::Direct3D12::D3D12_HEAP_FLAG_NONE,
                &desc,
                windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut upload_res,
            )?;
        }
        let upload_res = upload_res.ok_or_else(|| windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32)))?;
        unsafe {
            let mut mapped = std::ptr::null_mut();
            upload_res.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(quads.as_ptr() as *const u8, mapped.cast(), byte_len);
            upload_res.Unmap(0, None);
        }

        let quad_gpu_va = unsafe { upload_res.GetGPUVirtualAddress() };
        let pipeline = self.ui_canvas_pipeline.as_ref().unwrap();

        let push_constants = crate::ui_canvas::UiPushConstants {
            screen_width: self.width,
            screen_height: self.height,
            virtual_width,
            virtual_height,
        };

        unsafe {
            cmd.SetPipelineState(&pipeline.pso);
            cmd.SetGraphicsRootSignature(&pipeline.root_signature);
            cmd.SetDescriptorHeaps(&[Some(self.bindless_heap.clone())]);
            cmd.SetGraphicsRoot32BitConstants(
                0,
                4,
                (&push_constants as *const crate::ui_canvas::UiPushConstants).cast(),
                0,
            );
            cmd.SetGraphicsRootDescriptorTable(
                1,
                self.bindless_heap.GetGPUDescriptorHandleForHeapStart(),
            );
            cmd.SetGraphicsRootShaderResourceView(2, quad_gpu_va);

            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                Width: self.width as f32,
                Height: self.height as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            cmd.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: self.width as i32,
                bottom: self.height as i32,
            }]);
            cmd.OMSetRenderTargets(1, Some(&rtv), false, None);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.DrawInstanced(6, quads.len() as u32, 0, 0);
        }

        self.geometry.track_resource(upload_res);
        self.ui_canvas.clear();
        Ok(())
    }

    /// Get physical GPU adapter name
    pub fn adapter_name(&self) -> String {
        self.device_ctx.adapter_name()
    }

    /// Query current device cooperative state
    pub fn device_state(&self) -> DeviceState {
        if let Some(sc) = &self.swapchain {
            unsafe {
                let hr = sc.swapchain.Present(0, windows::Win32::Graphics::Dxgi::DXGI_PRESENT_TEST);
                if hr.is_ok() {
                    DeviceState::Ok
                } else if hr.0 == 0x887A0005u32 as i32 { // DXGI_ERROR_DEVICE_REMOVED
                    DeviceState::Lost
                } else if hr.0 == 0x887A0007u32 as i32 { // DXGI_ERROR_DEVICE_RESET
                    DeviceState::NeedReset
                } else {
                    DeviceState::Ok
                }
            }
        } else {
            DeviceState::Ok
        }
    }

    /// Query current frame runtime statistics
    pub fn stats(&self) -> RenderStats {
        let completed = unsafe { self.fence.GetCompletedValue() };
        RenderStats {
            frame_index: self.frame_index,
            width: self.width,
            height: self.height,
            completed_gpu_fence: completed,
            last_signaled_fence: self.fence_value,
        }
    }
}

impl Drop for Dx12Renderer {
    fn drop(&mut self) {
        let _ = self.wait_for_gpu();
        if !self.fence_event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.fence_event);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_initialization_and_stats() {
        if let Ok(mut renderer) = Dx12Renderer::new(2) {
            let stats = renderer.stats();
            assert_eq!(stats.frame_index, 0);
            assert_eq!(stats.width, 0);
            assert_eq!(stats.height, 0);
            assert_eq!(renderer.device_state(), DeviceState::Ok);

            // Wait for GPU on empty queue should succeed immediately
            assert!(renderer.wait_for_gpu().is_ok());
        }
    }

    #[test]
    fn test_renderer_texture_upload_and_bindless_indexing() {
        if let Ok(mut renderer) = Dx12Renderer::new(2) {
            // Create a valid 4x4 DXT1 DDS texture
            let mut dds = vec![0u8; 128 + 8];
            dds[0..4].copy_from_slice(&crate::texture::DDS_MAGIC.to_le_bytes());
            dds[4..8].copy_from_slice(&124u32.to_le_bytes());
            dds[8..12].copy_from_slice(&(crate::texture::DDSD_WIDTH | crate::texture::DDSD_HEIGHT | crate::texture::DDSD_PIXELFORMAT).to_le_bytes());
            dds[12..16].copy_from_slice(&4u32.to_le_bytes()); // height
            dds[16..20].copy_from_slice(&4u32.to_le_bytes()); // width
            dds[76..80].copy_from_slice(&32u32.to_le_bytes());
            dds[80..84].copy_from_slice(&crate::texture::DDPF_FOURCC.to_le_bytes());
            dds[84..88].copy_from_slice(&crate::texture::FOURCC_DXT1.to_le_bytes());

            let idx = renderer.load_texture_from_dds("act_stalker_albedo", &dds)
                .expect("Failed to upload DDS texture to GPU default heap");

            assert!(idx.is_valid());
            assert_eq!(idx.0, 0);

            let tex = renderer.get_texture("act_stalker_albedo").expect("Texture not found in registry");
            assert_eq!(tex.width, 4);
            assert_eq!(tex.height, 4);
            assert_eq!(tex.descriptor_index, idx);
            assert_eq!(tex.format, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC1_UNORM);

            let tex_by_idx = renderer.get_texture_by_index(idx).expect("Texture not found by descriptor index");
            assert_eq!(tex_by_idx.width, 4);
            assert_eq!(tex_by_idx.height, 4);

            // Verify deduplication: re-requesting with case or slash variations returns existing descriptor
            let idx_duplicate = renderer
                .load_texture_from_dds("ACT_STALKER_ALBEDO", &dds)
                .expect("Failed to query deduplicated texture");
            assert_eq!(idx_duplicate, idx);
            assert_eq!(renderer.bindless_allocator.allocated_count(), 1);
        }
    }
}
