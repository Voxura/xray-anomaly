//! High-Level DirectX 12 Renderer Facade
//!
//! Provides the primary runtime entry point for the X-Ray Monolith Engine,
//! integrating device context, swapchain presentation, multi-threaded command recording,
//! bindless descriptor management, transient memory aliasing, and pipeline caching.

use std::mem::ManuallyDrop;
use windows::core::{Interface, Result as WinResult, HRESULT};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Graphics::Direct3D12::{
    ID3D12Fence, ID3D12GraphicsCommandList, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_FENCE_FLAG_NONE, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
    D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET,
    D3D12_RESOURCE_TRANSITION_BARRIER,
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

    /// Clear the current backbuffer with an RGBA color
    pub fn clear_backbuffer(&self, color: [f32; 4]) {
        if let (Some(cmd), Some(sc)) = (&self.current_cmd_list, &self.swapchain) {
            let rtv = sc.current_rtv_handle();
            unsafe {
                cmd.ClearRenderTargetView(rtv, &color, None);
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
    pub fn load_texture_from_dds(&mut self, name: &str, bytes: &[u8]) -> WinResult<DescriptorIndex> {
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

        self.textures.insert(name.to_string(), gpu_tex);
        Ok(descriptor_index)
    }

    /// Retrieve loaded texture metadata and GPU resource by name
    pub fn get_texture(&self, name: &str) -> Option<&GpuTexture> {
        self.textures.get_by_name(name)
    }

    /// Retrieve loaded texture metadata and GPU resource by bindless descriptor index
    pub fn get_texture_by_index(&self, index: DescriptorIndex) -> Option<&GpuTexture> {
        self.textures.get_by_index(index)
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
        }
    }
}
