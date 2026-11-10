//! Render Graph (DAG) Pass Compiler for DirectX 12
//!
//! Compiles high-level rendering pass declarations into a topologically sorted Directed Acyclic Graph (DAG),
//! computes precise resource lifetimes, and automatically batches DirectX 12 transition barriers.

use std::collections::HashMap;

/// Hardware resource states matching D3D12_RESOURCE_STATES
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceState {
    Common,
    RenderTarget,
    DepthWrite,
    DepthRead,
    PixelShaderResource,
    NonPixelShaderResource,
    UnorderedAccess,
    CopyDest,
    CopySource,
    Present,
}

/// Identifies a transient or persistent GPU resource in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle(pub u32);

/// Identifies an individual render pass in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassHandle(pub u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDesc {
    pub name: String,
    pub initial_state: ResourceState,
    pub is_transient: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceBarrier {
    Transition {
        resource: ResourceHandle,
        state_before: ResourceState,
        state_after: ResourceState,
    },
    Uav {
        resource: ResourceHandle,
    },
}

#[derive(Debug, Clone)]
pub struct PassNode {
    pub name: String,
    pub reads: Vec<(ResourceHandle, ResourceState)>,
    pub writes: Vec<(ResourceHandle, ResourceState)>,
}

/// Compiled execution schedule for a frame
#[derive(Debug, Default)]
pub struct CompiledSchedule {
    pub pass_order: Vec<PassHandle>,
    pub pre_pass_barriers: HashMap<PassHandle, Vec<ResourceBarrier>>,
}

/// Builder and compiler for a single frame's render passes
#[derive(Debug, Default)]
pub struct RenderGraph {
    resources: Vec<ResourceDesc>,
    passes: Vec<PassNode>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare a resource in the graph
    pub fn create_resource(&mut self, name: &str, initial_state: ResourceState, is_transient: bool) -> ResourceHandle {
        let handle = ResourceHandle(self.resources.len() as u32);
        self.resources.push(ResourceDesc {
            name: name.to_string(),
            initial_state,
            is_transient,
        });
        handle
    }

    /// Declare a render pass
    pub fn add_pass(&mut self, name: &str) -> PassBuilder<'_> {
        let handle = PassHandle(self.passes.len() as u32);
        self.passes.push(PassNode {
            name: name.to_string(),
            reads: Vec::new(),
            writes: Vec::new(),
        });
        PassBuilder {
            graph: self,
            pass_handle: handle,
        }
    }

    /// Compile graph into execution order and generate batched resource barriers
    pub fn compile(&self) -> Result<CompiledSchedule, String> {
        let pass_count = self.passes.len();
        if pass_count == 0 {
            return Ok(CompiledSchedule::default());
        }

        // 1. Build adjacency list for topological sort
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); pass_count];
        let mut in_degree: Vec<usize> = vec![0; pass_count];

        // Map each resource to passes that produce (write) it
        let mut producers: HashMap<ResourceHandle, Vec<usize>> = HashMap::new();
        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for (res, _) in &pass.writes {
                producers.entry(*res).or_default().push(pass_idx);
            }
        }

        // For each pass reading a resource, it must depend on passes producing that resource
        for (reader_idx, pass) in self.passes.iter().enumerate() {
            for (res, _) in &pass.reads {
                if let Some(writers) = producers.get(res) {
                    for &writer_idx in writers {
                        if writer_idx != reader_idx && !adj[writer_idx].contains(&reader_idx) {
                            adj[writer_idx].push(reader_idx);
                            in_degree[reader_idx] += 1;
                        }
                    }
                }
            }
        }

        // 2. Kahn's Algorithm for Topological Sort
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut pass_order = Vec::new();
        while let Some(idx) = queue.pop_front() {
            pass_order.push(PassHandle(idx as u32));
            for &next_idx in &adj[idx] {
                in_degree[next_idx] -= 1;
                if in_degree[next_idx] == 0 {
                    queue.push_back(next_idx);
                }
            }
        }

        if pass_order.len() != pass_count {
            return Err("Cyclic dependency detected in RenderGraph passes".to_string());
        }

        // 3. Compute Resource State Transitions and UAV Barriers
        let mut current_states: HashMap<ResourceHandle, ResourceState> = HashMap::new();
        for (idx, res) in self.resources.iter().enumerate() {
            current_states.insert(ResourceHandle(idx as u32), res.initial_state);
        }

        let mut pre_pass_barriers: HashMap<PassHandle, Vec<ResourceBarrier>> = HashMap::new();

        for &pass_h in &pass_order {
            let pass = &self.passes[pass_h.0 as usize];
            let mut barriers = Vec::new();

            // Collect required states for all read and write dependencies
            let mut required: Vec<(ResourceHandle, ResourceState, bool)> = Vec::new();
            for &(res, state) in &pass.reads {
                required.push((res, state, false));
            }
            for &(res, state) in &pass.writes {
                required.push((res, state, true));
            }

            for (res, req_state, is_write) in required {
                let cur_state = current_states.get(&res).copied().unwrap_or(ResourceState::Common);
                if cur_state != req_state {
                    barriers.push(ResourceBarrier::Transition {
                        resource: res,
                        state_before: cur_state,
                        state_after: req_state,
                    });
                    current_states.insert(res, req_state);
                } else if cur_state == ResourceState::UnorderedAccess && is_write {
                    // D3D12 UAV Barrier: flushing memory cache across consecutive UAV dispatches/draws
                    barriers.push(ResourceBarrier::Uav { resource: res });
                }
            }

            if !barriers.is_empty() {
                pre_pass_barriers.insert(pass_h, barriers);
            }
        }

        Ok(CompiledSchedule {
            pass_order,
            pre_pass_barriers,
        })
    }
}

pub struct PassBuilder<'a> {
    graph: &'a mut RenderGraph,
    pass_handle: PassHandle,
}

impl<'a> PassBuilder<'a> {
    pub fn read(self, resource: ResourceHandle, state: ResourceState) -> Self {
        self.graph.passes[self.pass_handle.0 as usize].reads.push((resource, state));
        self
    }

    pub fn write(self, resource: ResourceHandle, state: ResourceState) -> Self {
        self.graph.passes[self.pass_handle.0 as usize].writes.push((resource, state));
        self
    }

    pub fn finish(self) -> PassHandle {
        self.pass_handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_graph_compilation_and_barriers() {
        let mut rg = RenderGraph::new();

        let gbuffer_albedo = rg.create_resource("GBuffer_Albedo", ResourceState::Common, true);
        let shadow_map = rg.create_resource("ShadowMap", ResourceState::Common, true);
        let back_buffer = rg.create_resource("BackBuffer", ResourceState::Present, false);

        // Pass 1: GBuffer (writes Albedo as RenderTarget)
        rg.add_pass("GBuffer")
            .write(gbuffer_albedo, ResourceState::RenderTarget)
            .finish();

        // Pass 2: Shadow Mapping (writes ShadowMap as DepthWrite)
        rg.add_pass("ShadowPass")
            .write(shadow_map, ResourceState::DepthWrite)
            .finish();

        // Pass 3: Lighting (reads GBuffer & ShadowMap, writes BackBuffer)
        rg.add_pass("Lighting")
            .read(gbuffer_albedo, ResourceState::PixelShaderResource)
            .read(shadow_map, ResourceState::PixelShaderResource)
            .write(back_buffer, ResourceState::RenderTarget)
            .finish();

        let schedule = rg.compile().expect("Graph compilation failed");
        assert_eq!(schedule.pass_order.len(), 3);

        // Lighting pass must transition gbuffer_albedo from RenderTarget to PixelShaderResource
        let lighting_pass = schedule.pass_order[2];
        let barriers = schedule.pre_pass_barriers.get(&lighting_pass).expect("Expected barriers");

        let albedo_barrier = barriers.iter().find(|b| match b {
            ResourceBarrier::Transition { resource, .. } => *resource == gbuffer_albedo,
            _ => false,
        }).unwrap();

        match albedo_barrier {
            ResourceBarrier::Transition { state_before, state_after, .. } => {
                assert_eq!(*state_before, ResourceState::RenderTarget);
                assert_eq!(*state_after, ResourceState::PixelShaderResource);
            }
            _ => panic!("Expected transition barrier"),
        }
    }

    #[test]
    fn test_render_graph_cycle_detection_rejection() {
        let mut rg = RenderGraph::new();
        let res_a = rg.create_resource("ResA", ResourceState::Common, true);
        let res_b = rg.create_resource("ResB", ResourceState::Common, true);

        // Pass 1: writes ResA, reads ResB
        rg.add_pass("Pass1")
            .read(res_b, ResourceState::PixelShaderResource)
            .write(res_a, ResourceState::RenderTarget)
            .finish();

        // Pass 2: writes ResB, reads ResA (creates cycle: Pass1 depends on Pass2, Pass2 depends on Pass1)
        rg.add_pass("Pass2")
            .read(res_a, ResourceState::PixelShaderResource)
            .write(res_b, ResourceState::RenderTarget)
            .finish();

        let result = rg.compile();
        assert!(result.is_err(), "Expected cyclic graph to be rejected");
        assert!(result.unwrap_err().contains("Cyclic dependency"));
    }

    #[test]
    fn test_render_graph_uav_barrier_generation() {
        let mut rg = RenderGraph::new();
        let uav_buffer = rg.create_resource("ComputeBuffer", ResourceState::UnorderedAccess, true);

        // Pass 1: Compute dispatch writing to UAV buffer
        rg.add_pass("ComputeStep1")
            .write(uav_buffer, ResourceState::UnorderedAccess)
            .finish();

        // Pass 2: Subsequent compute dispatch writing to the same UAV buffer
        rg.add_pass("ComputeStep2")
            .write(uav_buffer, ResourceState::UnorderedAccess)
            .finish();

        let schedule = rg.compile().expect("Graph compilation failed");
        let step2_pass = schedule.pass_order[1];
        let barriers = schedule.pre_pass_barriers.get(&step2_pass).expect("Expected UAV barrier");

        assert_eq!(barriers.len(), 1);
        match &barriers[0] {
            ResourceBarrier::Uav { resource } => assert_eq!(*resource, uav_buffer),
            _ => panic!("Expected UAV barrier for consecutive UAV writes"),
        }
    }
}
