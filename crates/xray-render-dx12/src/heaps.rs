//! DirectX 12 Transient Memory Aliasing & Resource Heap Manager
//!
//! Implements D3D12 Resource Heap Tier 2 memory suballocation and transient aliasing.
//! Non-overlapping Render Graph passes alias the same physical VRAM allocations,
//! drastically reducing GPU memory footprint for render targets and transient buffers.

use windows::core::Result as WinResult;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12Device, ID3D12Heap, ID3D12Resource,
    D3D12_HEAP_DESC, D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES,
    D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
};

/// 64 KB standard alignment for D3D12 buffers and standard textures
pub const DEFAULT_ALIGNMENT: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;

/// Lifetime descriptor for a transient render resource
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransientResourceRequest {
    pub id: u32,
    pub size_bytes: u64,
    pub alignment: u64,
    pub first_pass: u32,
    pub last_pass: u32,
}

/// Placed resource allocation metadata inside a transient heap
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacedAllocation {
    pub resource_id: u32,
    pub heap_offset: u64,
    pub size_bytes: u64,
    pub first_pass: u32,
    pub last_pass: u32,
}

/// Aliasing transition barrier generated when two resources share overlapping heap memory
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AliasingTransition {
    pub before_resource_id: Option<u32>,
    pub after_resource_id: u32,
    pub at_pass: u32,
}

/// Pure memory planner that calculates optimal VRAM offsets and aliasing transitions
#[derive(Debug, Default)]
pub struct TransientHeapPlanner {
    requests: Vec<TransientResourceRequest>,
}

impl TransientHeapPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a transient resource request with its active pass lifetime
    pub fn add_request(&mut self, id: u32, size_bytes: u64, alignment: u64, first_pass: u32, last_pass: u32) {
        assert!(first_pass <= last_pass, "first_pass must be <= last_pass");
        let align = if alignment == 0 { DEFAULT_ALIGNMENT } else { alignment };
        self.requests.push(TransientResourceRequest {
            id,
            size_bytes,
            alignment: align,
            first_pass,
            last_pass,
        });
    }

    /// Compute optimal memory placement with aliasing reuse
    ///
    /// Resources with disjoint pass lifetimes (`req_a.last_pass < req_b.first_pass`)
    /// are eligible to alias identical heap offsets.
    pub fn plan(&self) -> (Vec<PlacedAllocation>, Vec<AliasingTransition>, u64) {
        let mut sorted_requests = self.requests.clone();
        // Sort primarily by first_pass, then descending size
        sorted_requests.sort_by(|a, b| {
            a.first_pass
                .cmp(&b.first_pass)
                .then_with(|| b.size_bytes.cmp(&a.size_bytes))
        });

        let mut placements: Vec<PlacedAllocation> = Vec::new();
        let mut total_heap_size: u64 = 0;

        for req in &sorted_requests {
            // Find lowest aligned offset that doesn't conflict with any active allocation during [first_pass, last_pass]
            let mut candidate_offset: u64 = 0;

            loop {
                // Align candidate offset
                let remainder = candidate_offset % req.alignment;
                if remainder != 0 {
                    candidate_offset += req.alignment - remainder;
                }

                let candidate_end = candidate_offset + req.size_bytes;

                // Check conflict with existing placements
                let mut conflict = false;
                for existing in &placements {
                    // Check lifetime overlap
                    let lifetimes_overlap = !(req.last_pass < existing.first_pass || req.first_pass > existing.last_pass);
                    if lifetimes_overlap {
                        let existing_end = existing.heap_offset + existing.size_bytes;
                        let memory_overlaps = !(candidate_end <= existing.heap_offset || candidate_offset >= existing_end);
                        if memory_overlaps {
                            // Advance candidate offset past the conflicting allocation
                            candidate_offset = existing_end;
                            conflict = true;
                            break;
                        }
                    }
                }

                if !conflict {
                    break;
                }
            }

            let end = candidate_offset + req.size_bytes;
            if end > total_heap_size {
                total_heap_size = end;
            }

            placements.push(PlacedAllocation {
                resource_id: req.id,
                heap_offset: candidate_offset,
                size_bytes: req.size_bytes,
                first_pass: req.first_pass,
                last_pass: req.last_pass,
            });
        }

        // Align total heap size to 64 KB boundary
        let remainder = total_heap_size % DEFAULT_ALIGNMENT;
        if remainder != 0 {
            total_heap_size += DEFAULT_ALIGNMENT - remainder;
        }

        // Generate aliasing transitions: whenever an allocation starts, find if an earlier
        // disjoint allocation occupied the same offset/memory range
        let mut transitions = Vec::new();
        for i in 0..placements.len() {
            let current = &placements[i];
            let mut previous_occupant: Option<u32> = None;

            for j in 0..i {
                let prev = &placements[j];
                if prev.last_pass < current.first_pass {
                    // Check if they share overlapping memory
                    let current_end = current.heap_offset + current.size_bytes;
                    let prev_end = prev.heap_offset + prev.size_bytes;
                    if !(current_end <= prev.heap_offset || current.heap_offset >= prev_end) {
                        previous_occupant = Some(prev.resource_id);
                    }
                }
            }

            if let Some(prev_id) = previous_occupant {
                transitions.push(AliasingTransition {
                    before_resource_id: Some(prev_id),
                    after_resource_id: current.resource_id,
                    at_pass: current.first_pass,
                });
            }
        }

        (placements, transitions, total_heap_size)
    }
}

/// Hardware Direct3D 12 Transient Heap Manager
pub struct TransientHeapManager {
    device: ID3D12Device,
    heap: ID3D12Heap,
    heap_capacity: u64,
}

impl TransientHeapManager {
    /// Create hardware transient heap with D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES (Tier 2)
    pub fn new(device: &ID3D12Device, capacity_bytes: u64) -> WinResult<Self> {
        let aligned_capacity = if capacity_bytes % DEFAULT_ALIGNMENT != 0 {
            capacity_bytes + (DEFAULT_ALIGNMENT - (capacity_bytes % DEFAULT_ALIGNMENT))
        } else {
            capacity_bytes
        };

        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: aligned_capacity,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: windows::Win32::Graphics::Direct3D12::D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: windows::Win32::Graphics::Direct3D12::D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            },
            Alignment: DEFAULT_ALIGNMENT,
            Flags: D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
        };

        let mut heap: Option<ID3D12Heap> = None;
        unsafe { device.CreateHeap(&heap_desc, &mut heap)?; }
        let heap = heap.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x80004005u32 as i32)))?;

        Ok(Self {
            device: device.clone(),
            heap,
            heap_capacity: aligned_capacity,
        })
    }

    /// Instantiate a placed resource inside this transient heap at the specified offset
    pub fn create_placed_resource(
        &self,
        offset: u64,
        desc: &D3D12_RESOURCE_DESC,
        initial_state: D3D12_RESOURCE_STATES,
    ) -> WinResult<ID3D12Resource> {
        let mut resource: Option<ID3D12Resource> = None;
        unsafe {
            self.device.CreatePlacedResource(
                &self.heap,
                offset,
                desc,
                initial_state,
                None,
                &mut resource,
            )?;
        }
        resource.ok_or_else(|| windows::core::Error::from_hresult(windows::core::HRESULT(0x80004005u32 as i32)))
    }

    /// Total heap capacity in bytes
    pub fn capacity(&self) -> u64 {
        self.heap_capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disjoint_lifetime_aliasing_reuses_heap_offset() {
        let mut planner = TransientHeapPlanner::new();

        // Pass 0..1: Resource 1 (GBuffer Normals, 10 MB)
        planner.add_request(1, 10 * 1024 * 1024, DEFAULT_ALIGNMENT, 0, 1);

        // Pass 2..3: Resource 2 (Post-Process Bloom Target, 10 MB)
        // Disjoint with Resource 1: must reuse offset 0!
        planner.add_request(2, 10 * 1024 * 1024, DEFAULT_ALIGNMENT, 2, 3);

        let (placements, transitions, total_size) = planner.plan();

        assert_eq!(placements.len(), 2);
        assert_eq!(placements[0].heap_offset, 0);
        assert_eq!(placements[1].heap_offset, 0, "Disjoint resource must alias offset 0");
        assert_eq!(total_size, 10 * 1024 * 1024, "Total heap size must equal max single allocation, not sum");

        // Transition barrier must be emitted at Pass 2
        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].before_resource_id, Some(1));
        assert_eq!(transitions[0].after_resource_id, 2);
        assert_eq!(transitions[0].at_pass, 2);
    }

    #[test]
    fn test_overlapping_lifetime_allocates_separate_offsets() {
        let mut planner = TransientHeapPlanner::new();

        // Pass 0..2: Resource 1 (10 MB)
        planner.add_request(1, 10 * 1024 * 1024, DEFAULT_ALIGNMENT, 0, 2);

        // Pass 1..3: Resource 2 (10 MB) - overlaps with Resource 1 at pass 1 and 2!
        planner.add_request(2, 10 * 1024 * 1024, DEFAULT_ALIGNMENT, 1, 3);

        let (placements, transitions, total_size) = planner.plan();

        assert_eq!(placements.len(), 2);
        assert_eq!(placements[0].heap_offset, 0);
        assert_eq!(placements[1].heap_offset, 10 * 1024 * 1024, "Overlapping resources must not share offsets");
        assert_eq!(total_size, 20 * 1024 * 1024);
        assert!(transitions.is_empty(), "No aliasing barrier between non-aliased memory");
    }

    #[test]
    fn test_alignment_quantization() {
        let mut planner = TransientHeapPlanner::new();

        // Odd size not aligned to 64 KB (100 bytes)
        planner.add_request(1, 100, DEFAULT_ALIGNMENT, 0, 1);
        planner.add_request(2, 100, DEFAULT_ALIGNMENT, 0, 1);

        let (placements, _, total_size) = planner.plan();

        assert_eq!(placements[0].heap_offset, 0);
        // Second allocation must start at aligned 64 KB boundary
        assert_eq!(placements[1].heap_offset, DEFAULT_ALIGNMENT);
        assert_eq!(total_size, 2 * DEFAULT_ALIGNMENT);
    }

    #[test]
    fn test_hardware_placed_resource_and_aliasing_barrier() {
        use crate::device::Dx12DeviceContext;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_ALIASING_BARRIER,
            D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON,
            D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_FENCE_FLAG_NONE,
            ID3D12CommandList,
        };
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

        if let Ok(ctx) = Dx12DeviceContext::new(false) {
            if ctx.caps.resource_heap_tier.0 >= windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_HEAP_TIER_2.0 {
                let heap_mgr = TransientHeapManager::new(&ctx.device, 1024 * 1024)
                    .expect("Creating transient heap must succeed");

                let buf_desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Alignment: DEFAULT_ALIGNMENT,
                    Width: 64 * 1024,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: D3D12_RESOURCE_FLAG_NONE,
                };

                let res_a = heap_mgr.create_placed_resource(
                    0,
                    &buf_desc,
                    D3D12_RESOURCE_STATE_COMMON,
                ).expect("Creating placed resource A must succeed");

                let res_b = heap_mgr.create_placed_resource(
                    0,
                    &buf_desc,
                    D3D12_RESOURCE_STATE_COMMON,
                ).expect("Creating placed resource B must succeed");

                unsafe {
                    let cmd_alloc: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator =
                        ctx.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                            .expect("Creating command allocator must succeed");

                    let cmd_list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList =
                        ctx.device.CreateCommandList(
                            0,
                            D3D12_COMMAND_LIST_TYPE_DIRECT,
                            &cmd_alloc,
                            None,
                        ).expect("Creating command list must succeed");

                    let mut barrier = D3D12_RESOURCE_BARRIER::default();
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
                    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                    barrier.Anonymous.Aliasing = std::mem::ManuallyDrop::new(
                        D3D12_RESOURCE_ALIASING_BARRIER {
                            pResourceBefore: std::mem::transmute_copy(&res_a),
                            pResourceAfter: std::mem::transmute_copy(&res_b),
                        }
                    );

                    cmd_list.ResourceBarrier(&[barrier]);
                    cmd_list.Close().expect("Closing command list must succeed");

                    let command_lists = [Some(ID3D12CommandList::from(cmd_list))];
                    ctx.direct_queue.ExecuteCommandLists(&command_lists);

                    let fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence =
                        ctx.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)
                            .expect("Creating fence must succeed");
                    ctx.direct_queue.Signal(&fence, 1).expect("Signaling fence must succeed");
                    while fence.GetCompletedValue() < 1 {
                        std::thread::yield_now();
                    }
                }
            }
        }
    }
}
