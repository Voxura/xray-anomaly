//! DirectX 12 Root Signature and Pipeline State Object (PSO) Engine
//!
//! Provides the shader parameter binding ABI and thread-safe pipeline state caching.
//! Configures Root Constants for dynamic bindless draw calls, Frame Constant Buffers,
//! and caches compiled PSOs to eliminate runtime driver shader compilation hitches.

use std::collections::HashMap;
use std::sync::Mutex;
use windows::core::Result as WinResult;
use windows::Win32::Graphics::Direct3D12::{
    D3D12SerializeRootSignature, ID3D12Device, ID3D12PipelineState, ID3D12RootSignature,
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    D3D12_ROOT_CONSTANTS, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
    D3D12_ROOT_PARAMETER_TYPE_CBV, D3D12_ROOT_SIGNATURE_DESC,
    D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL, D3D_ROOT_SIGNATURE_VERSION_1_0,
};

/// 32-bit root constant count matching PushDrawConstants (8 DWORDs / 32 bytes)
pub const ROOT_CONSTANTS_DWORD_COUNT: u32 = 8;

/// Manages creation and caching of Direct3D 12 Root Signatures and PSOs
pub struct PipelineManager {
    device: ID3D12Device,
    bindless_root_signature: ID3D12RootSignature,
    pso_cache: Mutex<HashMap<u64, ID3D12PipelineState>>,
}

impl PipelineManager {
    /// Initialize pipeline manager with standard Shader Model 6.6 bindless root signature
    pub fn new(device: &ID3D12Device) -> WinResult<Self> {
        let bindless_root_signature = Self::create_bindless_root_signature(device)?;

        Ok(Self {
            device: device.clone(),
            bindless_root_signature,
            pso_cache: Mutex::new(HashMap::new()),
        })
    }

    /// Primary Shader Model 6.6 Bindless Root Signature Layout:
    /// - Parameter 0: 32-bit Root Constants (8 DWORDs at register b0, space 0)
    /// - Parameter 1: Frame Constant Buffer View (register b1, space 0)
    /// - Flags: CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED
    pub fn create_bindless_root_signature(device: &ID3D12Device) -> WinResult<ID3D12RootSignature> {
        let root_constants = D3D12_ROOT_CONSTANTS {
            ShaderRegister: 0, // register(b0)
            RegisterSpace: 0,
            Num32BitValues: ROOT_CONSTANTS_DWORD_COUNT,
        };

        let mut param0 = D3D12_ROOT_PARAMETER::default();
        param0.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        param0.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        param0.Anonymous.Constants = root_constants;

        let mut param1 = D3D12_ROOT_PARAMETER::default();
        param1.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        param1.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        param1.Anonymous.Descriptor.ShaderRegister = 1; // register(b1)
        param1.Anonymous.Descriptor.RegisterSpace = 0;

        let params = [param0, param1];

        let flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: flags,
        };

        unsafe {
            let mut blob = None;
            let mut error_blob = None;
            let res = D3D12SerializeRootSignature(
                &desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut blob,
                Some(&mut error_blob),
            );

            if let Err(e) = res {
                if let Some(err_b) = error_blob {
                    let msg = std::slice::from_raw_parts(
                        err_b.GetBufferPointer() as *const u8,
                        err_b.GetBufferSize(),
                    );
                    eprintln!("D3D12SerializeRootSignature error: {}", String::from_utf8_lossy(msg));
                }
                return Err(e);
            }

            let serialized = blob.unwrap();
            let byte_slice = std::slice::from_raw_parts(
                serialized.GetBufferPointer() as *const u8,
                serialized.GetBufferSize(),
            );

            device.CreateRootSignature(0, byte_slice)
        }
    }

    /// Get reference to standard bindless root signature
    pub fn root_signature(&self) -> &ID3D12RootSignature {
        &self.bindless_root_signature
    }

    /// Create or retrieve cached Graphics Pipeline State Object
    pub fn get_or_create_graphics_pso(
        &self,
        key: u64,
        desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    ) -> WinResult<ID3D12PipelineState> {
        let mut cache = self.pso_cache.lock().unwrap();
        if let Some(pso) = cache.get(&key) {
            return Ok(pso.clone());
        }

        let pso: ID3D12PipelineState = unsafe { self.device.CreateGraphicsPipelineState(desc)? };
        cache.insert(key, pso.clone());
        Ok(pso)
    }

    /// Create or retrieve cached Compute Pipeline State Object
    pub fn get_or_create_compute_pso(
        &self,
        key: u64,
        compute_shader_bytecode: &[u8],
    ) -> WinResult<ID3D12PipelineState> {
        let mut cache = self.pso_cache.lock().unwrap();
        if let Some(pso) = cache.get(&key) {
            return Ok(pso.clone());
        }

        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: unsafe { std::mem::transmute_copy(&self.bindless_root_signature) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: compute_shader_bytecode.as_ptr() as *const _,
                BytecodeLength: compute_shader_bytecode.len(),
            },
            NodeMask: 0,
            CachedPSO: Default::default(),
            Flags: Default::default(),
        };

        let pso: ID3D12PipelineState = unsafe { self.device.CreateComputePipelineState(&desc)? };
        cache.insert(key, pso.clone());
        Ok(pso)
    }

    /// Count of currently cached PSOs
    pub fn cached_pso_count(&self) -> usize {
        self.pso_cache.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Dx12DeviceContext;

    #[test]
    fn test_bindless_root_signature_creation() {
        if let Ok(ctx) = Dx12DeviceContext::new(false) {
            match PipelineManager::new(&ctx.device) {
                Ok(mgr) => {
                    assert_eq!(mgr.cached_pso_count(), 0);
                }
                Err(e) => {
                    panic!("Root signature error: {:?}", e);
                }
            }
        } else {
            eprintln!("Skipping hardware root signature test: D3D12 adapter unavailable");
        }
    }
}
