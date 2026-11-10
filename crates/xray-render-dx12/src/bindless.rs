//! Shader Model 6.6 Bindless Descriptor Indexing Engine
//!
//! Provides unbounded descriptor heap management, allowing HLSL shaders to index any texture
//! or buffer dynamically via `ResourceDescriptorHeap[index]` without CPU descriptor table rebinding.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

pub const INVALID_DESCRIPTOR_INDEX: u32 = u32::MAX;
pub const MAX_BINDLESS_DESCRIPTORS: u32 = 1_000_000; // SM 6.6 Tier 2 / 3 limit

/// Strongly typed 32-bit descriptor index referencing a slot in the GPU descriptor heap
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct DescriptorIndex(pub u32);

impl DescriptorIndex {
    pub const INVALID: DescriptorIndex = DescriptorIndex(INVALID_DESCRIPTOR_INDEX);

    #[inline(always)]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_DESCRIPTOR_INDEX
    }
}

/// Standard 32-bit push constants layout passed to HLSL Root Signature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct PushDrawConstants {
    pub albedo_index: u32,
    pub normal_index: u32,
    pub material_index: u32,
    pub transform_index: u32,
    pub custom_param0: u32,
    pub custom_param1: u32,
    pub custom_param2: u32,
    pub custom_param3: u32,
}

#[derive(Debug, Clone, Copy)]
struct DeferredRelease {
    index: u32,
    fence_value: u64,
}

/// Thread-safe bindless descriptor index allocator with GPU fence-synchronized recycling.
///
/// Prevents GPU hazards by ensuring descriptor indices are never recycled until
/// the GPU command list referencing them has completed execution past its associated fence value.
pub struct BindlessIndexAllocator {
    next_index: AtomicU32,
    free_list: Mutex<Vec<u32>>,
    deferred_releases: Mutex<VecDeque<DeferredRelease>>,
    capacity: u32,
}

impl BindlessIndexAllocator {
    pub fn new(capacity: u32) -> Self {
        Self {
            next_index: AtomicU32::new(0),
            free_list: Mutex::new(Vec::new()),
            deferred_releases: Mutex::new(VecDeque::new()),
            capacity: capacity.min(MAX_BINDLESS_DESCRIPTORS),
        }
    }

    /// Allocate a 32-bit descriptor index
    pub fn allocate(&self) -> Option<DescriptorIndex> {
        // Try recycled slot first
        {
            let mut free = self.free_list.lock().unwrap();
            if let Some(idx) = free.pop() {
                return Some(DescriptorIndex(idx));
            }
        }

        // Increment monotonically
        let idx = self.next_index.fetch_add(1, Ordering::Relaxed);
        if idx < self.capacity {
            Some(DescriptorIndex(idx))
        } else {
            None // Out of descriptor heap slots
        }
    }

    /// Enqueue a descriptor for deferred release once the specified GPU fence value is signaled.
    ///
    /// This prevents in-flight GPU draw calls from reading corrupted descriptor tables.
    pub fn free_deferred(&self, index: DescriptorIndex, fence_value: u64) {
        if index.is_valid() && index.0 < self.capacity {
            let mut deferred = self.deferred_releases.lock().unwrap();
            deferred.push_back(DeferredRelease {
                index: index.0,
                fence_value,
            });
        }
    }

    /// Immediate free without fence synchronization (use ONLY for resources never submitted to GPU).
    pub fn free_immediate(&self, index: DescriptorIndex) {
        if index.is_valid() && index.0 < self.capacity {
            let mut free = self.free_list.lock().unwrap();
            free.push(index.0);
        }
    }

    /// Process deferred releases against the latest GPU completed fence value.
    ///
    /// Any descriptors associated with `fence_value <= completed_fence` are immediately
    /// returned to the active free list.
    pub fn retire_fences(&self, completed_fence: u64) -> usize {
        let mut deferred = self.deferred_releases.lock().unwrap();
        let mut retired_count = 0;

        let mut ready = Vec::new();
        while let Some(front) = deferred.front() {
            if front.fence_value <= completed_fence {
                let item = deferred.pop_front().unwrap();
                ready.push(item.index);
                retired_count += 1;
            } else {
                break; // Queue is ordered by monotonic fence submission
            }
        }

        if !ready.is_empty() {
            let mut free = self.free_list.lock().unwrap();
            free.extend(ready);
        }

        retired_count
    }

    /// Get current count of descriptors pending GPU fence retirement
    pub fn pending_deferred_count(&self) -> usize {
        self.deferred_releases.lock().unwrap().len()
    }

    /// Get current active descriptor count (allocated and not yet in free list)
    pub fn allocated_count(&self) -> u32 {
        let total = self.next_index.load(Ordering::Relaxed).min(self.capacity);
        let free = self.free_list.lock().unwrap().len() as u32;
        total.saturating_sub(free)
    }
}

impl Default for BindlessIndexAllocator {
    fn default() -> Self {
        Self::new(MAX_BINDLESS_DESCRIPTORS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bindless_allocation_and_immediate_free() {
        let allocator = BindlessIndexAllocator::new(10);
        let d0 = allocator.allocate().unwrap();
        let d1 = allocator.allocate().unwrap();
        assert_eq!(d0.0, 0);
        assert_eq!(d1.0, 1);
        assert_eq!(allocator.allocated_count(), 2);

        // Immediate free d0 and re-allocate
        allocator.free_immediate(d0);
        assert_eq!(allocator.allocated_count(), 1);

        let d_recycled = allocator.allocate().unwrap();
        assert_eq!(d_recycled.0, 0); // Reused slot 0
        assert_eq!(allocator.allocated_count(), 2);
    }

    #[test]
    fn test_bindless_deferred_fence_retirement() {
        let allocator = BindlessIndexAllocator::new(10);
        let d0 = allocator.allocate().unwrap();
        let d1 = allocator.allocate().unwrap();

        // Enqueue deferred release associated with GPU fence value 100
        allocator.free_deferred(d0, 100);
        allocator.free_deferred(d1, 105);

        assert_eq!(allocator.pending_deferred_count(), 2);

        // GPU has only reached fence 99: neither descriptor should be recycled yet
        let retired = allocator.retire_fences(99);
        assert_eq!(retired, 0);
        assert_eq!(allocator.pending_deferred_count(), 2);

        // Next allocation should NOT reuse d0 (it's still in-flight on GPU)
        let d2 = allocator.allocate().unwrap();
        assert_eq!(d2.0, 2);

        // GPU reaches fence 100: d0 is safe to retire, but d1 is still in-flight
        let retired = allocator.retire_fences(100);
        assert_eq!(retired, 1);
        assert_eq!(allocator.pending_deferred_count(), 1);

        // Now allocate should reuse d0
        let d0_reused = allocator.allocate().unwrap();
        assert_eq!(d0_reused.0, 0);

        // GPU reaches fence 105: d1 is now retired
        let retired = allocator.retire_fences(105);
        assert_eq!(retired, 1);
        assert_eq!(allocator.pending_deferred_count(), 0);

        let d1_reused = allocator.allocate().unwrap();
        assert_eq!(d1_reused.0, 1);
    }
}
