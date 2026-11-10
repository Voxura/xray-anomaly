//! Modern Deferred Rendering Pipeline for DirectX 12
//!
//! Manages GBuffer resource declarations, Shader Model 6.6 bindless descriptor associations,
//! and automated Render Graph scheduling for the X-Ray Monolith Engine.

use crate::graph::{CompiledSchedule, RenderGraph, ResourceHandle, ResourceState};
use crate::shaders::{ShaderCompileRequest, ShaderCompiler, ShaderTarget};
use std::path::Path;

/// Standard GBuffer layout for X-Ray DX12 deferred shading
#[derive(Debug, Clone, Copy)]
pub struct GBufferResources {
    pub albedo: ResourceHandle,
    pub normal: ResourceHandle,
    pub material: ResourceHandle,
    pub depth: ResourceHandle,
    pub hdr_output: ResourceHandle,
    pub backbuffer: ResourceHandle,
}

/// Deferred pipeline configuration and pass graph builder
pub struct DeferredPipeline {
    pub width: u32,
    pub height: u32,
}

impl DeferredPipeline {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Build the complete Render Graph for a frame containing GBuffer, Lighting, and Presentation passes.
    pub fn build_graph(&self, graph: &mut RenderGraph) -> (GBufferResources, CompiledSchedule) {
        // 1. Declare GBuffer and Target Resources
        let albedo = graph.create_resource("GBuffer_Albedo", ResourceState::Common, true);
        let normal = graph.create_resource("GBuffer_Normal", ResourceState::Common, true);
        let material = graph.create_resource("GBuffer_Material", ResourceState::Common, true);
        let depth = graph.create_resource("GBuffer_Depth", ResourceState::Common, true);
        let hdr_output = graph.create_resource("HDR_Accumulation", ResourceState::Common, true);
        let backbuffer = graph.create_resource("BackBuffer", ResourceState::Present, false);

        let resources = GBufferResources {
            albedo,
            normal,
            material,
            depth,
            hdr_output,
            backbuffer,
        };

        // 2. Pass 1: GBuffer Rasterization Pass (Writes Albedo, Normal, Material, Depth)
        graph.add_pass("GBuffer_Raster")
            .write(albedo, ResourceState::RenderTarget)
            .write(normal, ResourceState::RenderTarget)
            .write(material, ResourceState::RenderTarget)
            .write(depth, ResourceState::DepthWrite);

        // 3. Pass 2: Deferred Lighting Compute Pass (Reads GBuffer textures & Depth via SM 6.6 bindless, writes HDR UAV)
        graph.add_pass("Deferred_Lighting_Compute")
            .read(albedo, ResourceState::NonPixelShaderResource)
            .read(normal, ResourceState::NonPixelShaderResource)
            .read(material, ResourceState::NonPixelShaderResource)
            .read(depth, ResourceState::NonPixelShaderResource)
            .write(hdr_output, ResourceState::UnorderedAccess);

        // 4. Pass 3: Tonemapping and Post-Processing (Reads HDR UAV, writes to BackBuffer)
        graph.add_pass("Tonemap_And_Present")
            .read(hdr_output, ResourceState::PixelShaderResource)
            .write(backbuffer, ResourceState::RenderTarget);

        // 5. Compile schedule with topological sorting and barrier calculation
        let schedule = graph.compile().expect("Deferred pipeline graph compilation failed");

        (resources, schedule)
    }

    /// Compile all required deferred shaders via DXC offline compiler
    pub fn compile_shaders(compiler: &ShaderCompiler, shaders_dir: &Path) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let gbuffer_path = shaders_dir.join("gbuffer.hlsl");
        let lighting_path = shaders_dir.join("deferred_lighting.hlsl");

        let gbuffer_hlsl = std::fs::read_to_string(&gbuffer_path)
            .map_err(|e| format!("Failed to read {}: {}", gbuffer_path.display(), e))?;
        let lighting_hlsl = std::fs::read_to_string(&lighting_path)
            .map_err(|e| format!("Failed to read {}: {}", lighting_path.display(), e))?;

        // 1. Compile GBuffer VS
        let req_vs = ShaderCompileRequest::new(&gbuffer_hlsl, "vs_main", ShaderTarget::Vertex6_6);
        let vs_bc = compiler.compile(&req_vs).map_err(|e| format!("GBuffer VS error: {:?}", e))?;

        // 2. Compile GBuffer PS
        let req_ps = ShaderCompileRequest::new(&gbuffer_hlsl, "ps_main", ShaderTarget::Pixel6_6);
        let ps_bc = compiler.compile(&req_ps).map_err(|e| format!("GBuffer PS error: {:?}", e))?;

        // 3. Compile Deferred Lighting CS
        let req_cs = ShaderCompileRequest::new(&lighting_hlsl, "cs_main", ShaderTarget::Compute6_6);
        let cs_bc = compiler.compile(&req_cs).map_err(|e| format!("Deferred Lighting CS error: {:?}", e))?;

        Ok((vs_bc, ps_bc, cs_bc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deferred_pipeline_graph_compilation() {
        let pipeline = DeferredPipeline::new(1920, 1080);
        let mut graph = RenderGraph::new();

        let (resources, schedule) = pipeline.build_graph(&mut graph);

        assert_eq!(schedule.pass_order.len(), 3);

        // Verify barriers generated for deferred lighting pass (reads GBuffer as NonPixelShaderResource, writes HDR as UAV)
        let lighting_pass_handle = schedule.pass_order[1];
        let barriers = schedule.pre_pass_barriers.get(&lighting_pass_handle).expect("Missing lighting barriers");

        let transitions: Vec<_> = barriers.iter().filter_map(|b| match b {
            crate::graph::ResourceBarrier::Transition { resource, state_before, state_after } => {
                Some((*resource, *state_before, *state_after))
            }
            _ => None,
        }).collect();

        assert!(transitions.contains(&(resources.albedo, ResourceState::RenderTarget, ResourceState::NonPixelShaderResource)));
        assert!(transitions.contains(&(resources.normal, ResourceState::RenderTarget, ResourceState::NonPixelShaderResource)));
        assert!(transitions.contains(&(resources.material, ResourceState::RenderTarget, ResourceState::NonPixelShaderResource)));
        assert!(transitions.contains(&(resources.depth, ResourceState::DepthWrite, ResourceState::NonPixelShaderResource)));
        assert!(transitions.contains(&(resources.hdr_output, ResourceState::Common, ResourceState::UnorderedAccess)));
    }

    #[test]
    fn test_deferred_shader_offline_compilation() {
        let compiler = ShaderCompiler::new()
            .expect("Windows SDK dxc.exe not found on build host");

        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let shaders_dir = manifest_dir.join("shaders");

        let (vs_bc, ps_bc, cs_bc) = DeferredPipeline::compile_shaders(&compiler, &shaders_dir)
            .expect("Offline DXC compilation of GBuffer and Lighting shaders failed");

        assert!(!vs_bc.is_empty(), "GBuffer VS bytecode must not be empty");
        assert!(!ps_bc.is_empty(), "GBuffer PS bytecode must not be empty");
        assert!(!cs_bc.is_empty(), "Deferred Lighting CS bytecode must not be empty");
        assert_eq!(&vs_bc[0..4], b"DXBC", "Bytecode must be standard DXIL DXBC container");
        assert_eq!(&ps_bc[0..4], b"DXBC", "Bytecode must be standard DXIL DXBC container");
        assert_eq!(&cs_bc[0..4], b"DXBC", "Bytecode must be standard DXIL DXBC container");
    }
}
