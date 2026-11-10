//! DirectX 12 SwapChain and Render Target Management
//!
//! Wraps DXGI Flip-Discard swapchains, managing RTV descriptor heaps, backbuffer resources,
//! presentation synchronization, and window resize operations.

use windows::core::{Interface, Result as WinResult};
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12CommandQueue, ID3D12DescriptorHeap, ID3D12Device, ID3D12Resource,
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    IDXGIFactory4, IDXGISwapChain3, IDXGISwapChain4, DXGI_SCALING_STRETCH,
    DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
};

use crate::device::FRAME_BUFFER_COUNT;

pub struct SwapChainManager {
    pub swapchain: IDXGISwapChain4,
    pub rtv_heap: ID3D12DescriptorHeap,
    pub rtv_descriptor_size: usize,
    pub backbuffers: [Option<ID3D12Resource>; FRAME_BUFFER_COUNT],
    pub width: u32,
    pub height: u32,
}

impl SwapChainManager {
    /// Create swapchain and allocate RTV descriptor heap for target HWND
    pub fn new(
        factory: &IDXGIFactory4,
        direct_queue: &ID3D12CommandQueue,
        device: &ID3D12Device,
        hwnd: HWND,
        width: u32,
        height: u32,
    ) -> WinResult<Self> {
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: FRAME_BUFFER_COUNT as u32,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: windows::Win32::Graphics::Dxgi::Common::DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: 0,
        };

        let swapchain_base: IDXGISwapChain3 = unsafe {
            factory
                .CreateSwapChainForHwnd(direct_queue, hwnd, &desc, None, None)?
                .cast()?
        };
        let swapchain: IDXGISwapChain4 = swapchain_base.cast()?;

        // Create RTV Descriptor Heap
        let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: FRAME_BUFFER_COUNT as u32,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
            NodeMask: 0,
        };
        let rtv_heap: ID3D12DescriptorHeap = unsafe { device.CreateDescriptorHeap(&heap_desc)? };
        let rtv_descriptor_size = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize
        };

        let mut mgr = Self {
            swapchain,
            rtv_heap,
            rtv_descriptor_size,
            backbuffers: [None, None, None],
            width,
            height,
        };

        mgr.create_render_target_views(device)?;
        Ok(mgr)
    }

    /// Retrieve backbuffer resource handles from swapchain and initialize RTV descriptors
    fn create_render_target_views(&mut self, device: &ID3D12Device) -> WinResult<()> {
        let heap_start = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };

        for i in 0..FRAME_BUFFER_COUNT {
            let backbuffer: ID3D12Resource = unsafe { self.swapchain.GetBuffer(i as u32)? };
            let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: heap_start.ptr + (i * self.rtv_descriptor_size),
            };

            unsafe {
                device.CreateRenderTargetView(&backbuffer, None, handle);
            }
            self.backbuffers[i] = Some(backbuffer);
        }
        Ok(())
    }

    /// Current active backbuffer index
    pub fn current_backbuffer_index(&self) -> usize {
        unsafe { self.swapchain.GetCurrentBackBufferIndex() as usize }
    }

    /// CPU descriptor handle for current backbuffer's Render Target View
    pub fn current_rtv_handle(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let idx = self.current_backbuffer_index();
        let heap_start = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: heap_start.ptr + (idx * self.rtv_descriptor_size),
        }
    }

    /// Current active backbuffer ID3D12Resource
    pub fn current_backbuffer_resource(&self) -> &ID3D12Resource {
        let idx = self.current_backbuffer_index();
        self.backbuffers[idx]
            .as_ref()
            .expect("Backbuffer resource not initialized")
    }

    /// Resize swapchain backbuffers upon window dimension change
    pub fn resize(&mut self, device: &ID3D12Device, width: u32, height: u32) -> WinResult<()> {
        if width == 0 || height == 0 || (width == self.width && height == self.height) {
            return Ok(());
        }

        // Release references to existing backbuffer resources before resizing
        for i in 0..FRAME_BUFFER_COUNT {
            self.backbuffers[i] = None;
        }

        unsafe {
            self.swapchain.ResizeBuffers(
                FRAME_BUFFER_COUNT as u32,
                width,
                height,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                windows::Win32::Graphics::Dxgi::DXGI_SWAP_CHAIN_FLAG(0),
            )?;
        }

        self.width = width;
        self.height = height;
        self.create_render_target_views(device)
    }

    /// Present current backbuffer to display
    pub fn present(&self, vsync: bool) -> WinResult<()> {
        let sync_interval = if vsync { 1 } else { 0 };
        unsafe {
            self.swapchain.Present(sync_interval, windows::Win32::Graphics::Dxgi::DXGI_PRESENT(0)).ok()
        }
    }
}
