//! Phase 11: Instanced Rain Streaks Renderer & Simulation Manager
//!
//! Faithful implementation of X-Ray Monolith Engine rain effects (`Rain.cpp` & `dxRainRender.cpp`):
//! - Camera-centered toroidal volume simulation of rain drops with wind velocity and pitch tilting
//! - Instanced billboard camera-facing quad generation purely on the GPU via `SV_VertexID` (0..5)
//! - Streaks dynamically orient along the fall / wind velocity vector and expand perpendicularly
//! - CPU-to-GPU upload streaming via `ByteAddressBuffer` with zero vertex buffer allocations
//! - Early-Z depth occlusion: foreground level geometry, roofs, and walls correctly occlude rain drops

use std::mem::ManuallyDrop;
use windows::core::{Error, HRESULT, Result as WinResult};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};

fn failure() -> Error {
    Error::from_hresult(HRESULT(0x80004005u32 as i32))
}

pub const DEFAULT_MAX_RAIN_DROPS: usize = 2500;
pub const RAIN_SOURCE_RADIUS: f32 = 15.0;
pub const RAIN_SOURCE_HEIGHT: f32 = 20.0;
pub const RAIN_SINK_OFFSET: f32 = -5.0;
pub const RAIN_DROP_LENGTH: f32 = 4.5;
pub const RAIN_DROP_WIDTH: f32 = 0.25;
pub const RAIN_DROP_SPEED_MIN: f32 = 40.0;
pub const RAIN_DROP_SPEED_MAX: f32 = 70.0;

/// 48-byte GPU rain drop layout (3 float4s, 16-byte aligned)
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuRainDrop {
    pub position: [f32; 3],
    pub length: f32,
    pub direction: [f32; 3],
    pub alpha: f32,
    pub width: f32,
    pub uv_set: u32,
    pub padding: [u32; 2],
}

const _: () = assert!(std::mem::size_of::<GpuRainDrop>() == 48);

impl Default for GpuRainDrop {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            length: RAIN_DROP_LENGTH,
            direction: [0.0, -1.0, 0.0],
            alpha: 1.0,
            width: RAIN_DROP_WIDTH,
            uv_set: 0,
            padding: [0, 0],
        }
    }
}

/// 32-byte 16-byte aligned push constants for rain rendering
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RainPushConstants {
    pub rain_texture: u32,
    pub rain_density: f32,
    pub streak_length_scale: f32,
    pub streak_width_scale: f32,
    pub rain_color: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<RainPushConstants>() == 32);

impl Default for RainPushConstants {
    fn default() -> Self {
        Self {
            rain_texture: u32::MAX,
            rain_density: 1.0,
            streak_length_scale: 1.0,
            streak_width_scale: 1.0,
            rain_color: [0.8, 0.85, 0.9, 0.75],
        }
    }
}

/// High-performance GPU rain streak renderer
pub struct RainRenderer {
    pub root: ID3D12RootSignature,
    pub pso: ID3D12PipelineState,
}

impl RainRenderer {
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
            Num32BitValues: (std::mem::size_of::<RainPushConstants>() / 4) as u32,
        };

        // Param 3: Raindrops ByteAddressBuffer SRV (t0, space 1)
        let mut raindrops = D3D12_ROOT_PARAMETER::default();
        raindrops.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        raindrops.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
        raindrops.Anonymous.Descriptor = D3D12_ROOT_DESCRIPTOR {
            ShaderRegister: 0,
            RegisterSpace: 1,
        };

        let parameters = [camera, textures, push, raindrops];

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
                    eprintln!("[RainRenderer] D3D12SerializeRootSignature error: {}", String::from_utf8_lossy(msg));
                }
                return Err(e);
            }
            let blob = blob.ok_or_else(failure)?;
            match device.CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                ),
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[RainRenderer] CreateRootSignature failed: {:?}", e);
                    return Err(e);
                }
            }
        };

        let source = include_str!("../shaders/rain.hlsl");
        let compiler = match ShaderCompiler::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[RainRenderer] ShaderCompiler::new failed: {:?}", e);
                return Err(failure());
            }
        };
        let vs = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "vs_rain",
                ShaderTarget::Vertex6_0,
            ))
            .map_err(|e| {
                eprintln!("[RainRenderer] vs compile failed: {:?}", e);
                failure()
            })?;
        let ps = compiler
            .compile(&ShaderCompileRequest::new(
                source,
                "ps_rain",
                ShaderTarget::Pixel6_0,
            ))
            .map_err(|e| {
                eprintln!("[RainRenderer] ps compile failed: {:?}", e);
                failure()
            })?;

        // Alpha blend: SrcAlpha + InvSrcAlpha
        let mut blend_state = D3D12_BLEND_DESC::default();
        let rt = &mut blend_state.RenderTarget[0];
        rt.BlendEnable = true.into();
        rt.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOp = D3D12_BLEND_OP_ADD;
        rt.SrcBlendAlpha = D3D12_BLEND_ONE;
        rt.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;

        let depth_stencil_state = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            StencilEnable: false.into(),
            FrontFace: D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_KEEP,
                StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                StencilPassOp: D3D12_STENCIL_OP_KEEP,
                StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            },
            BackFace: D3D12_DEPTH_STENCILOP_DESC {
                StencilFailOp: D3D12_STENCIL_OP_KEEP,
                StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                StencilPassOp: D3D12_STENCIL_OP_KEEP,
                StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            },
            ..Default::default()
        };

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
            pInputElementDescs: std::ptr::null(),
            NumElements: 0,
        };
        desc.RasterizerState = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            ..Default::default()
        };
        desc.BlendState = blend_state;
        desc.DepthStencilState = depth_stencil_state;
        desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        desc.SampleMask = u32::MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc = DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        };

        let pso: ID3D12PipelineState = unsafe {
            match device.CreateGraphicsPipelineState(&desc) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[RainRenderer] CreateGraphicsPipelineState failed: {:?}", e);
                    return Err(e);
                }
            }
        };

        Ok(Self { root, pso })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        camera_gpu_va: u64,
        srv_heap: &ID3D12DescriptorHeap,
        raindrops_gpu_va: u64,
        num_drops: u32,
        push: &RainPushConstants,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        width: u32,
        height: u32,
    ) {
        if num_drops == 0 {
            return;
        }

        unsafe {
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetPipelineState(&self.pso);

            // Bind camera CBV (b0)
            cmd.SetGraphicsRootConstantBufferView(0, camera_gpu_va);

            // Bind bindless textures heap (t0, space 0)
            cmd.SetDescriptorHeaps(&[Some(srv_heap.clone())]);
            cmd.SetGraphicsRootDescriptorTable(1, srv_heap.GetGPUDescriptorHandleForHeapStart());

            // Bind push constants (b1)
            let push_slice = std::slice::from_raw_parts(
                (push as *const RainPushConstants).cast::<u32>(),
                std::mem::size_of::<RainPushConstants>() / 4,
            );
            cmd.SetGraphicsRoot32BitConstants(2, push_slice.len() as u32, push_slice.as_ptr().cast(), 0);

            // Bind raindrops ByteAddressBuffer SRV (t0, space 1)
            cmd.SetGraphicsRootShaderResourceView(3, raindrops_gpu_va);

            // Viewport & scissor
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let scissor = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            cmd.RSSetViewports(&[viewport]);
            cmd.RSSetScissorRects(&[scissor]);

            // Set render target and optional depth stencil
            let dsv_ptr = dsv.as_ref().map(|d| d as *const D3D12_CPU_DESCRIPTOR_HANDLE);
            cmd.OMSetRenderTargets(1, Some(&rtv), false, dsv_ptr);

            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            // 6 vertices per drop instance (billboard quad)
            cmd.DrawInstanced(6, num_drops, 0, 0);
        }
    }
}

/// Simulation manager handling dynamic rain drop physics and GPU upload streaming
pub struct RainManager {
    drops: Vec<GpuRainDrop>,
    speeds: Vec<f32>,
    active_count: u32,
    max_capacity: usize,
    upload_buffer: Option<ID3D12Resource>,
    upload_ptr: *mut u8,
    rng_state: u64,
}

impl RainManager {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        let mut mgr = Self {
            drops: vec![GpuRainDrop::default(); cap],
            speeds: vec![50.0; cap],
            active_count: 0,
            max_capacity: cap,
            upload_buffer: None,
            upload_ptr: std::ptr::null_mut(),
            rng_state: 0x123456789ABCDEF0,
        };

        // Initialize pseudo-random initial distribution
        for i in 0..cap {
            mgr.speeds[i] = RAIN_DROP_SPEED_MIN + (mgr.rand_f32() * (RAIN_DROP_SPEED_MAX - RAIN_DROP_SPEED_MIN));
            mgr.drops[i].length = RAIN_DROP_LENGTH * (0.8 + mgr.rand_f32() * 0.4);
            mgr.drops[i].width = RAIN_DROP_WIDTH * (0.8 + mgr.rand_f32() * 0.4);
            mgr.drops[i].alpha = 0.6 + mgr.rand_f32() * 0.4;
            mgr.drops[i].uv_set = if mgr.rand_f32() > 0.5 { 1 } else { 0 };
        }
        mgr
    }

    fn rand_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        ((self.rng_state & 0x00FFFFFF) as f32) / 16777216.0
    }

    /// Advance rain drop simulation around camera volume with wind tilting
    pub fn step(
        &mut self,
        cam_pos: [f32; 3],
        wind_direction: f32,
        wind_velocity: f32,
        rain_density: f32,
        dt: f32,
    ) {
        if rain_density <= 0.001 {
            self.active_count = 0;
            return;
        }

        let desired = ((self.max_capacity as f32) * rain_density.clamp(0.0, 1.0)) as usize;
        self.active_count = desired.clamp(0, self.max_capacity) as u32;

        // Wind pitch calculation matching X-Ray Rain.cpp
        let wind_vel_norm = (wind_velocity * 0.001).clamp(0.0, 1.0);
        let max_pitch = 35.0_f32.to_radians();
        let pitch = max_pitch * wind_vel_norm;

        // Wind direction angle with fixNorth (-90 deg)
        let dir_x = (wind_direction - std::f32::consts::FRAC_PI_2).cos() * pitch.sin();
        let dir_z = (wind_direction - std::f32::consts::FRAC_PI_2).sin() * pitch.sin();
        let dir_y = -pitch.cos();
        let dir_len = (dir_x * dir_x + dir_y * dir_y + dir_z * dir_z).sqrt().max(1e-5);
        let fall_dir = [dir_x / dir_len, dir_y / dir_len, dir_z / dir_len];

        let bottom_limit = cam_pos[1] + RAIN_SINK_OFFSET;
        let top_spawn = cam_pos[1] + RAIN_SOURCE_HEIGHT;

        for i in 0..desired {
            let speed = self.speeds[i];
            let rand_angle = self.rand_f32();
            let rand_rad = self.rand_f32();
            let rand_h = self.rand_f32();
            let drop = &mut self.drops[i];

            drop.direction = fall_dir;
            drop.position[0] += fall_dir[0] * speed * dt;
            drop.position[1] += fall_dir[1] * speed * dt;
            drop.position[2] += fall_dir[2] * speed * dt;

            // When drop falls below bottom boundary, respawn in cylinder around camera
            if drop.position[1] < bottom_limit {
                let angle = rand_angle * std::f32::consts::TAU;
                let radius = rand_rad.sqrt() * RAIN_SOURCE_RADIUS;
                drop.position[0] = cam_pos[0] + angle.cos() * radius;
                drop.position[1] = top_spawn + (rand_h * 5.0);
                drop.position[2] = cam_pos[2] + angle.sin() * radius;
            }
        }
    }

    /// Stream active raindrops to the persistent GPU upload buffer
    pub fn upload(&mut self, device: &ID3D12Device) -> WinResult<u64> {
        let size = (self.max_capacity * std::mem::size_of::<GpuRainDrop>()) as u64;

        if self.upload_buffer.is_none() {
            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                ..Default::default()
            };
            let res_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_UNKNOWN,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_NONE,
                ..Default::default()
            };

            let mut resource = None;
            unsafe {
                device.CreateCommittedResource(
                    &heap_props,
                    D3D12_HEAP_FLAG_NONE,
                    &res_desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut resource,
                )?;
            }
            let res: ID3D12Resource = resource.unwrap();
            let mut ptr = std::ptr::null_mut();
            unsafe {
                res.Map(0, None, Some(&mut ptr))?;
            }
            self.upload_buffer = Some(res);
            self.upload_ptr = ptr.cast();
        }

        if self.active_count > 0 && !self.upload_ptr.is_null() {
            let count = self.active_count as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.drops.as_ptr().cast::<u8>(),
                    self.upload_ptr,
                    count * std::mem::size_of::<GpuRainDrop>(),
                );
            }
        }

        let va = unsafe { self.upload_buffer.as_ref().unwrap().GetGPUVirtualAddress() };
        Ok(va)
    }

    #[inline]
    pub fn active_count(&self) -> u32 {
        self.active_count
    }

    #[inline]
    pub fn drops_mut(&mut self) -> &mut [GpuRainDrop] {
        &mut self.drops
    }

    #[inline]
    pub fn set_active_count(&mut self, count: u32) {
        self.active_count = count.min(self.max_capacity as u32);
    }
}

impl Drop for RainManager {
    fn drop(&mut self) {
        if let Some(buf) = self.upload_buffer.take() {
            unsafe {
                buf.Unmap(0, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rain_structures_size_and_alignment() {
        assert_eq!(std::mem::size_of::<GpuRainDrop>(), 48);
        assert_eq!(std::mem::align_of::<GpuRainDrop>(), 16);
        assert_eq!(std::mem::size_of::<RainPushConstants>(), 32);
        assert_eq!(std::mem::align_of::<RainPushConstants>(), 16);
    }

    #[test]
    fn test_rain_manager_step_and_wrap() {
        let mut mgr = RainManager::new(100);
        let cam_pos = [0.0, 10.0, 0.0];
        mgr.step(cam_pos, 0.0, 100.0, 0.5, 0.1);
        assert_eq!(mgr.active_count(), 50);

        for drop in &mgr.drops[..50] {
            assert!(drop.direction[1] < 0.0);
        }
    }
}
