//! Directional Sunlight Shadow Mapping Subsystem for DirectX 12
//!
//! Allocates dedicated DXGI_FORMAT_R32_TYPELESS depth textures with D32_FLOAT DSVs
//! for shadow generation and R32_FLOAT SRVs in the bindless descriptor heap for
//! real-time sunlight shadow evaluation across terrain and skeletal meshes.

use std::mem::ManuallyDrop;
use windows::core::{Result as WinResult, Error, HRESULT};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Foundation::RECT;

use crate::bindless::{BindlessIndexAllocator, DescriptorIndex};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

/// Manages a high-resolution directional shadow depth map and its bindless descriptor
pub struct DirectionalShadowMap {
    pub resolution: u32,
    pub resource: ID3D12Resource,
    pub dsv_heap: ID3D12DescriptorHeap,
    pub srv_index: DescriptorIndex,
    pub current_state: D3D12_RESOURCE_STATES,
    pub light_view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
}

impl DirectionalShadowMap {
    /// Allocate a directional shadow map with dedicated DSV and bindless SRV
    pub fn new(
        device: &ID3D12Device,
        resolution: u32,
        bindless_allocator: &mut BindlessIndexAllocator,
        bindless_heap: &ID3D12DescriptorHeap,
        srv_descriptor_size: u32,
        _fence_value: u64,
    ) -> WinResult<Self> {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: resolution as u64,
            Height: resolution,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32_TYPELESS,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            Alignment: 0,
        };

        let clear_val = D3D12_CLEAR_VALUE {
            Format: DXGI_FORMAT_D32_FLOAT,
            Anonymous: D3D12_CLEAR_VALUE_0 {
                DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                    Depth: 1.0,
                    Stencil: 0,
                },
            },
        };

        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                Some(&clear_val),
                &mut resource,
            )?;
        }
        let resource = resource.ok_or_else(failure)?;

        // Dedicated DSV heap
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
                &resource,
                Some(&dsv_desc),
                dsv_heap.GetCPUDescriptorHandleForHeapStart(),
            );
        }

        // Allocate bindless SRV slot and populate with R32_FLOAT view
        let srv_index = bindless_allocator.allocate().ok_or_else(failure)?;
        let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
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

        unsafe {
            let start = bindless_heap.GetCPUDescriptorHandleForHeapStart();
            let dest_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (srv_index.0 as usize) * (srv_descriptor_size as usize),
            };
            device.CreateShaderResourceView(Some(&resource), Some(&srv_desc), dest_handle);
        }

        let mut identity = [[0.0f32; 4]; 4];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1.0;
        }

        Ok(Self {
            resolution,
            resource,
            dsv_heap,
            srv_index,
            current_state: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            light_view_proj: identity,
            light_dir: [0.577, 0.577, 0.577, 0.0],
        })
    }

    /// Retrieve the DSV CPU descriptor handle
    pub fn dsv_handle(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        unsafe { self.dsv_heap.GetCPUDescriptorHandleForHeapStart() }
    }

    /// Retrieve the allocated bindless SRV descriptor index
    pub fn descriptor_index(&self) -> DescriptorIndex {
        self.srv_index
    }

    /// Transition to DEPTH_WRITE, clear shadow depth buffer, and bind DSV as render target
    pub fn begin_shadow_pass(&mut self, cmd: &ID3D12GraphicsCommandList) {
        if self.current_state != D3D12_RESOURCE_STATE_DEPTH_WRITE {
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: unsafe { std::mem::transmute_copy(&self.resource) },
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: self.current_state,
                        StateAfter: D3D12_RESOURCE_STATE_DEPTH_WRITE,
                    }),
                },
            };
            unsafe {
                cmd.ResourceBarrier(&[barrier]);
            }
            self.current_state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        }

        let dsv = self.dsv_handle();
        unsafe {
            cmd.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            cmd.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: self.resolution as f32,
                Height: self.resolution as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            cmd.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: self.resolution as i32,
                bottom: self.resolution as i32,
            }]);
            cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
        }
    }

    /// Transition from DEPTH_WRITE to PIXEL_SHADER_RESOURCE for bindless shader sampling
    pub fn end_shadow_pass(&mut self, cmd: &ID3D12GraphicsCommandList) {
        if self.current_state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: unsafe { std::mem::transmute_copy(&self.resource) },
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: self.current_state,
                        StateAfter: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                    }),
                },
            };
            unsafe {
                cmd.ResourceBarrier(&[barrier]);
            }
            self.current_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        }
    }

    /// Update light matrices for directional sunlight projection
    pub fn set_light_camera(&mut self, light_view_proj: [[f32; 4]; 4], light_dir: [f32; 4]) {
        self.light_view_proj = light_view_proj;
        self.light_dir = light_dir;
    }
}
