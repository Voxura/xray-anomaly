//! DirectX 12 Shader Model 6.6 DXIL Compiler & Bytecode Cache
//!
//! Integrates DirectX Shader Compiler (DXC) to compile HLSL directly into Shader Model 6.6
//! (`vs_6_6`, `ps_6_6`, `cs_6_6`) DXIL with native `ResourceDescriptorHeap` bindless indexing.
//! Provides a persistent, thread-safe memory and disk bytecode cache to eliminate runtime stutters.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ShaderError {
    DxcNotFound,
    IoError(String),
    CompilationFailed(String),
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaderError::DxcNotFound => write!(f, "DirectX Shader Compiler (dxc.exe) not found on system"),
            ShaderError::IoError(s) => write!(f, "I/O error during shader compilation: {}", s),
            ShaderError::CompilationFailed(s) => write!(f, "HLSL compilation error:\n{}", s),
        }
    }
}

impl std::error::Error for ShaderError {}

/// Shader Model 6.6 compilation target profile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderTarget {
    Vertex6_6,
    Pixel6_6,
    Compute6_6,
    Mesh6_6,
    Amplification6_6,
}

impl ShaderTarget {
    pub fn profile_str(&self) -> &'static str {
        match self {
            ShaderTarget::Vertex6_6 => "vs_6_6",
            ShaderTarget::Pixel6_6 => "ps_6_6",
            ShaderTarget::Compute6_6 => "cs_6_6",
            ShaderTarget::Mesh6_6 => "ms_6_6",
            ShaderTarget::Amplification6_6 => "as_6_6",
        }
    }
}

/// Parameters for an HLSL compile invocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderCompileRequest {
    pub source: String,
    pub entry_point: String,
    pub target: ShaderTarget,
    pub defines: Vec<(String, String)>,
    pub optimization_level: u8,
}

impl ShaderCompileRequest {
    pub fn new(source: impl Into<String>, entry_point: impl Into<String>, target: ShaderTarget) -> Self {
        Self {
            source: source.into(),
            entry_point: entry_point.into(),
            target,
            defines: Vec::new(),
            optimization_level: 3,
        }
    }

    pub fn with_define(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.defines.push((name.into(), value.into()));
        self
    }

    /// Calculate deterministic 64-bit cache hash
    pub fn calculate_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.source.hash(&mut hasher);
        self.entry_point.hash(&mut hasher);
        self.target.hash(&mut hasher);
        for (k, v) in &self.defines {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }
        self.optimization_level.hash(&mut hasher);
        hasher.finish()
    }
}

/// Invokes DirectX Shader Compiler (DXC) to produce SM 6.6 DXIL bytecode
pub struct ShaderCompiler {
    dxc_path: PathBuf,
}

impl ShaderCompiler {
    /// Discover DXC from environment variable, Windows SDK, or system PATH
    pub fn new() -> Result<Self, ShaderError> {
        if let Ok(p) = std::env::var("DXC_PATH") {
            let path = PathBuf::from(p);
            if path.exists() {
                return Ok(Self { dxc_path: path });
            }
        }

        // Standard Windows Kits locations
        let candidates = [
            r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe",
            r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe",
            r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe",
            r"C:\Program Files (x86)\Windows Kits\10\bin\x64\dxc.exe",
        ];

        for &c in &candidates {
            let path = PathBuf::from(c);
            if path.exists() {
                return Ok(Self { dxc_path: path });
            }
        }

        // Fallback to PATH
        if let Ok(output) = Command::new("where.exe").arg("dxc.exe").output() {
            if output.status.success() {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    if let Some(first_line) = text.lines().next() {
                        let path = PathBuf::from(first_line.trim());
                        if path.exists() {
                            return Ok(Self { dxc_path: path });
                        }
                    }
                }
            }
        }

        Err(ShaderError::DxcNotFound)
    }

    /// Direct constructor specifying known DXC executable path
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self { dxc_path: path.into() }
    }

    /// Compile HLSL source into SM 6.6 DXIL bytecode
    pub fn compile(&self, req: &ShaderCompileRequest) -> Result<Vec<u8>, ShaderError> {
        let temp_dir = std::env::temp_dir();
        let hash = req.calculate_hash();
        let in_file = temp_dir.join(format!("xray_shader_in_{:016x}.hlsl", hash));
        let out_file = temp_dir.join(format!("xray_shader_out_{:016x}.cso", hash));

        std::fs::write(&in_file, &req.source)
            .map_err(|e| ShaderError::IoError(format!("Failed to write temp HLSL: {}", e)))?;

        let mut cmd = Command::new(&self.dxc_path);
        cmd.arg("-T")
            .arg(req.target.profile_str())
            .arg("-E")
            .arg(&req.entry_point)
            .arg("-Qstrip_debug")
            .arg(format!("-O{}", req.optimization_level));

        for (k, v) in &req.defines {
            if v.is_empty() {
                cmd.arg(format!("-D{}", k));
            } else {
                cmd.arg(format!("-D{}={}", k, v));
            }
        }

        cmd.arg("-Fo").arg(&out_file).arg(&in_file);

        let output = cmd.output().map_err(|e| ShaderError::IoError(format!("Failed to execute dxc.exe: {}", e)))?;

        let _ = std::fs::remove_file(&in_file);

        if !output.status.success() {
            let err_text = String::from_utf8_lossy(&output.stderr);
            let out_text = String::from_utf8_lossy(&output.stdout);
            let combined = format!("{}\n{}", err_text, out_text);
            let _ = std::fs::remove_file(&out_file);
            return Err(ShaderError::CompilationFailed(combined.trim().to_string()));
        }

        let bytecode = std::fs::read(&out_file)
            .map_err(|e| ShaderError::IoError(format!("Failed to read compiled CSO: {}", e)))?;
        let _ = std::fs::remove_file(&out_file);

        Ok(bytecode)
    }
}

/// Thread-safe in-memory and disk shader bytecode cache
pub struct ShaderBytecodeCache {
    memory_cache: Mutex<HashMap<u64, Vec<u8>>>,
    disk_cache_dir: Option<PathBuf>,
}

impl ShaderBytecodeCache {
    pub fn new(disk_cache_dir: Option<PathBuf>) -> Self {
        if let Some(ref dir) = disk_cache_dir {
            let _ = std::fs::create_dir_all(dir);
        }
        Self {
            memory_cache: Mutex::new(HashMap::new()),
            disk_cache_dir,
        }
    }

    /// Retrieve bytecode from memory or disk cache; compile and store on miss
    pub fn get_or_compile(
        &self,
        req: &ShaderCompileRequest,
        compiler: &ShaderCompiler,
    ) -> Result<Vec<u8>, ShaderError> {
        let hash = req.calculate_hash();

        // 1. Check in-memory cache
        {
            let cache = self.memory_cache.lock().unwrap();
            if let Some(bc) = cache.get(&hash) {
                return Ok(bc.clone());
            }
        }

        // 2. Check disk cache
        if let Some(ref dir) = self.disk_cache_dir {
            let disk_file = dir.join(format!("{:016x}.cso", hash));
            if disk_file.exists() {
                if let Ok(bytes) = std::fs::read(&disk_file) {
                    let mut cache = self.memory_cache.lock().unwrap();
                    cache.insert(hash, bytes.clone());
                    return Ok(bytes);
                }
            }
        }

        // 3. Compile via DXC
        let bytecode = compiler.compile(req)?;

        // 4. Save to disk cache
        if let Some(ref dir) = self.disk_cache_dir {
            let disk_file = dir.join(format!("{:016x}.cso", hash));
            let _ = std::fs::write(&disk_file, &bytecode);
        }

        // 5. Save to memory cache
        let mut cache = self.memory_cache.lock().unwrap();
        cache.insert(hash, bytecode.clone());

        Ok(bytecode)
    }

    /// Count of currently cached bytecodes in memory
    pub fn cached_count(&self) -> usize {
        self.memory_cache.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_target_profiles() {
        assert_eq!(ShaderTarget::Vertex6_6.profile_str(), "vs_6_6");
        assert_eq!(ShaderTarget::Pixel6_6.profile_str(), "ps_6_6");
        assert_eq!(ShaderTarget::Compute6_6.profile_str(), "cs_6_6");
    }

    #[test]
    fn test_cache_key_generation_discrimination() {
        let req_a = ShaderCompileRequest::new("void main() {}", "main", ShaderTarget::Compute6_6);
        let req_b = ShaderCompileRequest::new("void main() {}", "main", ShaderTarget::Pixel6_6);
        let req_c = ShaderCompileRequest::new("void main() {}", "main", ShaderTarget::Compute6_6)
            .with_define("USE_SM66", "1");

        assert_ne!(req_a.calculate_hash(), req_b.calculate_hash());
        assert_ne!(req_a.calculate_hash(), req_c.calculate_hash());
    }

    #[test]
    fn test_compile_sm66_bindless_compute_shader() {
        if let Ok(compiler) = ShaderCompiler::new() {
            let hlsl = r#"
                struct PushConstants {
                    uint inIndex;
                    uint outIndex;
                    uint count;
                };
                ConstantBuffer<PushConstants> g_Push : register(b0);

                [numthreads(64, 1, 1)]
                void CSMain(uint3 id : SV_DispatchThreadID)
                {
                    if (id.x >= g_Push.count) return;
                    StructuredBuffer<float> inBuf = ResourceDescriptorHeap[g_Push.inIndex];
                    RWStructuredBuffer<float> outBuf = ResourceDescriptorHeap[g_Push.outIndex];
                    outBuf[id.x] = inBuf[id.x] * 2.0f;
                }
            "#;

            let req = ShaderCompileRequest::new(hlsl, "CSMain", ShaderTarget::Compute6_6);
            let cache = ShaderBytecodeCache::new(None);

            // First compilation: compiles via DXC
            let bc1 = cache.get_or_compile(&req, &compiler).expect("SM 6.6 compilation must succeed");
            assert!(!bc1.is_empty());
            assert_eq!(cache.cached_count(), 1);

            // Second query: retrieved from memory cache
            let bc2 = cache.get_or_compile(&req, &compiler).expect("Cache hit must succeed");
            assert_eq!(bc1, bc2);
        } else {
            eprintln!("Skipping DXC test: dxc.exe not found");
        }
    }

    #[test]
    fn test_compilation_failure_error_reporting() {
        if let Ok(compiler) = ShaderCompiler::new() {
            let invalid_hlsl = "this is invalid syntax @#$%;";
            let req = ShaderCompileRequest::new(invalid_hlsl, "CSMain", ShaderTarget::Compute6_6);
            let res = compiler.compile(&req);

            match res {
                Err(ShaderError::CompilationFailed(diagnostics)) => {
                    assert!(!diagnostics.is_empty(), "Diagnostics must report syntax error");
                }
                other => panic!("Expected CompilationFailed, got: {:?}", other),
            }
        }
    }
}
