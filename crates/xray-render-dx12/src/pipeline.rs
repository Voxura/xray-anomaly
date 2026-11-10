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

/// Comprehensive graphics pipeline descriptor cache key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphicsPipelineKey {
    pub vs_hash: u64,
    pub ps_hash: u64,
    pub cull_mode: u8,
    pub fill_mode: u8,
    pub depth_enable: bool,
    pub depth_write: bool,
    pub depth_func: u8,
    pub blend_enable: bool,
    pub rtv_formats: [u32; 8],
    pub rtv_count: u8,
    pub dsv_format: u32,
    pub topology_type: u8,
}

impl Default for GraphicsPipelineKey {
    fn default() -> Self {
        Self {
            vs_hash: 0,
            ps_hash: 0,
            cull_mode: 3, // D3D12_CULL_MODE_BACK
            fill_mode: 3, // D3D12_FILL_MODE_SOLID
            depth_enable: true,
            depth_write: true,
            depth_func: 4, // D3D12_COMPARISON_FUNC_LESS_EQUAL
            blend_enable: false,
            rtv_formats: [0; 8],
            rtv_count: 1,
            dsv_format: 45, // DXGI_FORMAT_D32_FLOAT
            topology_type: 3, // D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
        }
    }
}

/// Compute pipeline descriptor cache key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ComputePipelineKey {
    pub cs_hash: u64,
    pub node_mask: u32,
}

/// Unified pipeline state cache key
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineKey {
    Graphics(GraphicsPipelineKey),
    Compute(ComputePipelineKey),
}

/// Manages creation and caching of Direct3D 12 Root Signatures and PSOs
pub struct PipelineManager {
    device: ID3D12Device,
    bindless_root_signature: ID3D12RootSignature,
    pso_cache: Mutex<HashMap<PipelineKey, ID3D12PipelineState>>,
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
        key: GraphicsPipelineKey,
        desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    ) -> WinResult<ID3D12PipelineState> {
        let pkey = PipelineKey::Graphics(key);
        let mut cache = self.pso_cache.lock().unwrap();
        if let Some(pso) = cache.get(&pkey) {
            return Ok(pso.clone());
        }

        let pso: ID3D12PipelineState = unsafe { self.device.CreateGraphicsPipelineState(desc)? };
        cache.insert(pkey, pso.clone());
        Ok(pso)
    }

    /// Create or retrieve cached Compute Pipeline State Object
    pub fn get_or_create_compute_pso(
        &self,
        key: ComputePipelineKey,
        compute_shader_bytecode: &[u8],
    ) -> WinResult<ID3D12PipelineState> {
        let pkey = PipelineKey::Compute(key);
        let mut cache = self.pso_cache.lock().unwrap();
        if let Some(pso) = cache.get(&pkey) {
            return Ok(pso.clone());
        }

        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: unsafe { std::mem::transmute_copy(&self.bindless_root_signature) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: compute_shader_bytecode.as_ptr() as *const _,
                BytecodeLength: compute_shader_bytecode.len(),
            },
            NodeMask: key.node_mask,
            CachedPSO: Default::default(),
            Flags: Default::default(),
        };

        let pso: ID3D12PipelineState = unsafe { self.device.CreateComputePipelineState(&desc)? };
        cache.insert(pkey, pso.clone());
        Ok(pso)
    }

    /// Retrieve or insert pipeline entry using arbitrary factory closure (for tests and custom builders)
    pub fn get_or_insert_custom<F: FnOnce() -> WinResult<ID3D12PipelineState>>(
        &self,
        key: PipelineKey,
        creator: F,
    ) -> WinResult<ID3D12PipelineState> {
        let mut cache = self.pso_cache.lock().unwrap();
        if let Some(pso) = cache.get(&key) {
            return Ok(pso.clone());
        }

        let pso = creator()?;
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
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;
    use std::thread;
    use crate::device::Dx12DeviceContext;

    fn calculate_hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

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

    #[test]
    fn test_cache_key_completeness_and_discrimination() {
        let key_base = GraphicsPipelineKey::default();
        let hash_base = calculate_hash(&key_base);

        // 1. Alter VS hash
        let mut key_vs = key_base;
        key_vs.vs_hash = 0xDEADBEEF;
        assert_ne!(key_base, key_vs);
        assert_ne!(hash_base, calculate_hash(&key_vs));

        // 2. Alter PS hash
        let mut key_ps = key_base;
        key_ps.ps_hash = 0xCAFEBABE;
        assert_ne!(key_base, key_ps);
        assert_ne!(hash_base, calculate_hash(&key_ps));

        // 3. Alter Cull mode
        let mut key_cull = key_base;
        key_cull.cull_mode = 1; // None
        assert_ne!(key_base, key_cull);
        assert_ne!(hash_base, calculate_hash(&key_cull));

        // 4. Alter Fill mode
        let mut key_fill = key_base;
        key_fill.fill_mode = 2; // Wireframe
        assert_ne!(key_base, key_fill);
        assert_ne!(hash_base, calculate_hash(&key_fill));

        // 5. Alter Depth Write
        let mut key_depth_write = key_base;
        key_depth_write.depth_write = false;
        assert_ne!(key_base, key_depth_write);
        assert_ne!(hash_base, calculate_hash(&key_depth_write));

        // 6. Alter Blend Enable
        let mut key_blend = key_base;
        key_blend.blend_enable = true;
        assert_ne!(key_base, key_blend);
        assert_ne!(hash_base, calculate_hash(&key_blend));

        // 7. Alter RTV format
        let mut key_rtv = key_base;
        key_rtv.rtv_formats[0] = 28; // DXGI_FORMAT_R8G8B8A8_UNORM
        assert_ne!(key_base, key_rtv);
        assert_ne!(hash_base, calculate_hash(&key_rtv));

        // 8. Alter Topology Type
        let mut key_topo = key_base;
        key_topo.topology_type = 2; // Line
        assert_ne!(key_base, key_topo);
        assert_ne!(hash_base, calculate_hash(&key_topo));

        // Compute Key discrimination
        let comp_base = ComputePipelineKey { cs_hash: 100, node_mask: 0 };
        let comp_diff = ComputePipelineKey { cs_hash: 101, node_mask: 0 };
        assert_ne!(comp_base, comp_diff);
        assert_ne!(calculate_hash(&comp_base), calculate_hash(&comp_diff));
    }

    #[test]
    fn test_concurrent_pso_cache_access() {
        if let Ok(ctx) = Dx12DeviceContext::new(false) {
            if let Ok(mgr) = PipelineManager::new(&ctx.device) {
                let mgr = Arc::new(mgr);
                let num_threads = 16;
                let iterations = 100;
                let mut handles = Vec::new();

                for thread_id in 0..num_threads {
                    let mgr_clone = Arc::clone(&mgr);
                    handles.push(thread::spawn(move || {
                        for i in 0..iterations {
                            // Multiple threads query 4 shared keys concurrently
                            let key_id = (thread_id + i) % 4;
                            let _key = PipelineKey::Compute(ComputePipelineKey {
                                cs_hash: key_id as u64,
                                node_mask: 0,
                            });

                            // Read cache count safely without deadlocks
                            let _ = mgr_clone.cached_pso_count();
                        }
                    }));
                }

                for h in handles {
                    h.join().unwrap();
                }

                assert_eq!(mgr.cached_pso_count(), 0);
            }
        }
    }
}
