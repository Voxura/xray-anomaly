//! DirectX 12 Device, Queues, and SwapChain Abstraction
//!
//! Provides the primary hardware interface for the X-Ray Monolith modern DX12 backend.
//! Configures debug layers, verifies Shader Model 6.6 and Resource Heap Tier 2 capabilities,
//! and establishes independent Direct, Compute, and Copy command queues.

use windows::core::{Interface, Result as WinResult, HRESULT};
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, D3D12GetDebugInterface, ID3D12CommandQueue, ID3D12Debug, ID3D12Device,
    D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_TYPE_DIRECT,
    D3D12_COMMAND_QUEUE_DESC, D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_FEATURE_DATA_D3D12_OPTIONS,
    D3D12_FEATURE_DATA_SHADER_MODEL, D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_SHADER_MODEL,
    D3D12_RESOURCE_BINDING_TIER, D3D12_RESOURCE_HEAP_TIER, D3D_SHADER_MODEL_6_6,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory4, IDXGISwapChain3, IDXGISwapChain4,
    DXGI_CREATE_FACTORY_DEBUG, DXGI_SCALING_STRETCH, DXGI_SWAP_CHAIN_DESC1,
    DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_USAGE_RENDER_TARGET_OUTPUT,
};

pub const FRAME_BUFFER_COUNT: usize = 3;

#[derive(Debug, Clone, Copy)]
pub struct DeviceCaps {
    pub max_feature_level: D3D_FEATURE_LEVEL,
    pub supports_sm66: bool,
    pub resource_binding_tier: D3D12_RESOURCE_BINDING_TIER,
    pub resource_heap_tier: D3D12_RESOURCE_HEAP_TIER,
}

pub struct Dx12DeviceContext {
    pub factory: IDXGIFactory4,
    pub adapter: IDXGIAdapter1,
    pub device: ID3D12Device,
    pub direct_queue: ID3D12CommandQueue,
    pub compute_queue: ID3D12CommandQueue,
    pub copy_queue: ID3D12CommandQueue,
    pub caps: DeviceCaps,
}

impl Dx12DeviceContext {
    /// Initialize DirectX 12 factory, adapter, device, and dedicated command queues.
    pub fn new(enable_debug_layer: bool) -> WinResult<Self> {
        // 1. Enable D3D12 Debug Layer if requested
        if enable_debug_layer {
            unsafe {
                let mut debug_controller: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut debug_controller).is_ok() {
                    if let Some(debug) = debug_controller {
                        debug.EnableDebugLayer();
                    }
                }
            }
        }

        // 2. Create DXGI Factory 4
        let factory_flags = if enable_debug_layer {
            DXGI_CREATE_FACTORY_DEBUG
        } else {
            windows::Win32::Graphics::Dxgi::DXGI_CREATE_FACTORY_FLAGS(0)
        };
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(factory_flags)? };

        // 3. Enumerate high-performance hardware adapter (prefer discrete GPU with largest dedicated VRAM)
        let mut adapter: Option<IDXGIAdapter1> = None;
        let mut max_dedicated_vram: usize = 0;
        let mut adapter_idx = 0;
        loop {
            let candidate: IDXGIAdapter1 = match unsafe { factory.EnumAdapters1(adapter_idx) } {
                Ok(a) => a,
                Err(_) => break,
            };

            if let Ok(desc) = unsafe { candidate.GetDesc1() } {
                // Skip software (WARP) renderer if hardware adapter exists
                let is_software = (desc.Flags & 2) != 0; // DXGI_ADAPTER_FLAG_SOFTWARE
                if !is_software {
                    if desc.DedicatedVideoMemory >= max_dedicated_vram || adapter.is_none() {
                        max_dedicated_vram = desc.DedicatedVideoMemory;
                        adapter = Some(candidate);
                    }
                }
            }
            adapter_idx += 1;
        }

        // Fallback to adapter 0 if no dedicated hardware adapter found
        let adapter = match adapter {
            Some(a) => a,
            None => unsafe { factory.EnumAdapters1(0)? },
        };

        // 4. Create D3D12 Device (targeting Feature Level 12.0 first, falling back to 11.0)
        let feature_levels = [
            D3D_FEATURE_LEVEL_12_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_0,
        ];

        let mut device_opt: Option<ID3D12Device> = None;
        let mut achieved_level = D3D_FEATURE_LEVEL_11_0;

        for &level in &feature_levels {
            let mut dev: Option<ID3D12Device> = None;
            let hr = unsafe { D3D12CreateDevice(&adapter, level, &mut dev) };
            if hr.is_ok() {
                if let Some(d) = dev {
                    device_opt = Some(d);
                    achieved_level = level;
                    break;
                }
            }
        }

        let device = match device_opt {
            Some(d) => d,
            None => return Err(windows::core::Error::from_hresult(HRESULT(0x80004005u32 as i32))), // E_FAIL
        };

        // 5. Query hardware capabilities (SM 6.6 and Resource Heap Tiers)
        let mut sm_feature = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        let sm_supported = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                &mut sm_feature as *mut _ as *mut _,
                std::mem::size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>() as u32,
            )
        }
        .is_ok()
            && sm_feature.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;

        let mut options_feature = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        let _ = unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                &mut options_feature as *mut _ as *mut _,
                std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
            )
        };

        let caps = DeviceCaps {
            max_feature_level: achieved_level,
            supports_sm66: sm_supported,
            resource_binding_tier: options_feature.ResourceBindingTier,
            resource_heap_tier: options_feature.ResourceHeapTier,
        };

        // 6. Create Direct, Compute, and Copy Command Queues
        let direct_queue = Self::create_queue(&device, D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let compute_queue = Self::create_queue(&device, D3D12_COMMAND_LIST_TYPE_COMPUTE)?;
        let copy_queue = Self::create_queue(&device, D3D12_COMMAND_LIST_TYPE_COPY)?;

        Ok(Self {
            factory,
            adapter,
            device,
            direct_queue,
            compute_queue,
            copy_queue,
            caps,
        })
    }

    fn create_queue(
        device: &ID3D12Device,
        queue_type: windows::Win32::Graphics::Direct3D12::D3D12_COMMAND_LIST_TYPE,
    ) -> WinResult<ID3D12CommandQueue> {
        let desc = D3D12_COMMAND_QUEUE_DESC {
            Type: queue_type,
            Priority: 0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        unsafe { device.CreateCommandQueue(&desc) }
    }

    /// Create a high-performance DXGI Flip-Discard swapchain for a target HWND
    pub fn create_swapchain(
        &self,
        hwnd: HWND,
        width: u32,
        height: u32,
    ) -> WinResult<IDXGISwapChain4> {
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
            self.factory
                .CreateSwapChainForHwnd(&self.direct_queue, hwnd, &desc, None, None)?
                .cast()?
        };

        swapchain_base.cast()
    }

    /// Query the physical hardware adapter description
    pub fn adapter_name(&self) -> String {
        if let Ok(desc) = unsafe { self.adapter.GetDesc1() } {
            let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            String::from_utf16_lossy(&desc.Description[..len])
        } else {
            "Unknown Adapter".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dx12_device_context_creation() {
        match Dx12DeviceContext::new(false) {
            Ok(ctx) => {
                assert!(ctx.caps.max_feature_level.0 >= D3D_FEATURE_LEVEL_11_0.0);
            }
            Err(e) => {
                // In headless/CI or environment without D3D12 driver, note gracefully
                eprintln!("D3D12 adapter init skipped: {}", e);
            }
        }
    }
}

