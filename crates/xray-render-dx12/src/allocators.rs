//! Multi-Threaded Command Allocator and Frame Ring Buffer Engine
//!
//! DirectX 12 command allocators cannot be reset while the GPU is executing commands recorded from them.
//! This module provides triple-buffered, per-worker ring allocators synchronized via monotonic GPU fences,
//! enabling lock-free, zero-stall parallel command recording across multiple CPU worker threads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use windows::core::Result as WinResult;
use windows::Win32::Graphics::Direct3D12::{
    ID3D12CommandAllocator, ID3D12Device, ID3D12GraphicsCommandList,
    D3D12_COMMAND_LIST_TYPE,
};

use crate::device::FRAME_BUFFER_COUNT;

/// Per-worker ring buffer holding one `ID3D12CommandAllocator` per frame in flight
pub struct WorkerCommandRing {
    pub worker_id: usize,
    allocators: [ID3D12CommandAllocator; FRAME_BUFFER_COUNT],
    frame_fences: [AtomicU64; FRAME_BUFFER_COUNT],
}

impl WorkerCommandRing {
    pub fn new(
        device: &ID3D12Device,
        worker_id: usize,
        list_type: D3D12_COMMAND_LIST_TYPE,
    ) -> WinResult<Self> {
        let a0: ID3D12CommandAllocator = unsafe { device.CreateCommandAllocator(list_type)? };
        let a1: ID3D12CommandAllocator = unsafe { device.CreateCommandAllocator(list_type)? };
        let a2: ID3D12CommandAllocator = unsafe { device.CreateCommandAllocator(list_type)? };

        Ok(Self {
            worker_id,
            allocators: [a0, a1, a2],
            frame_fences: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
        })
    }

    /// Reset and prepare the command allocator for the given frame index once its GPU fence has passed.
    ///
    /// # Safety
    /// Returns `Err` if GPU execution has not yet reached the fence value associated with this slot.
    pub fn begin_frame(
        &self,
        frame_index: u64,
        completed_gpu_fence: u64,
    ) -> Result<&ID3D12CommandAllocator, String> {
        let slot = (frame_index as usize) % FRAME_BUFFER_COUNT;
        let last_fence = self.frame_fences[slot].load(Ordering::Acquire);

        if completed_gpu_fence < last_fence {
            return Err(format!(
                "Worker {} stall: GPU completed fence {} < required fence {} for slot {}",
                self.worker_id, completed_gpu_fence, last_fence, slot
            ));
        }

        let allocator = &self.allocators[slot];
        unsafe {
            allocator
                .Reset()
                .map_err(|e| format!("Failed to reset command allocator: {}", e))?;
        }

        Ok(allocator)
    }

    /// Tag the current frame slot with the GPU fence value signaled at submission.
    pub fn tag_submission_fence(&self, frame_index: u64, fence_value: u64) {
        let slot = (frame_index as usize) % FRAME_BUFFER_COUNT;
        self.frame_fences[slot].store(fence_value, Ordering::Release);
    }
}

/// Global manager orchestrating per-thread command ring buffers
pub struct MultiThreadedCommandManager {
    workers: Vec<WorkerCommandRing>,
    device: ID3D12Device,
    list_type: D3D12_COMMAND_LIST_TYPE,
    command_list_cache: Mutex<Vec<ID3D12GraphicsCommandList>>,
}

impl MultiThreadedCommandManager {
    pub fn new(
        device: &ID3D12Device,
        worker_count: usize,
        list_type: D3D12_COMMAND_LIST_TYPE,
    ) -> WinResult<Self> {
        let mut workers = Vec::with_capacity(worker_count);
        for id in 0..worker_count {
            workers.push(WorkerCommandRing::new(device, id, list_type)?);
        }

        Ok(Self {
            workers,
            device: device.clone(),
            list_type,
            command_list_cache: Mutex::new(Vec::new()),
        })
    }

    /// Get reference to worker command ring
    pub fn worker(&self, worker_id: usize) -> Option<&WorkerCommandRing> {
        self.workers.get(worker_id)
    }

    /// Total worker threads configured
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Acquire or create an open `ID3D12GraphicsCommandList` bound to the worker's reset allocator
    pub fn acquire_command_list(
        &self,
        allocator: &ID3D12CommandAllocator,
    ) -> WinResult<ID3D12GraphicsCommandList> {
        // Recycle existing closed command list if available
        {
            let mut cache = self.command_list_cache.lock().unwrap();
            if let Some(cmd) = cache.pop() {
                unsafe {
                    cmd.Reset(allocator, None)?;
                }
                return Ok(cmd);
            }
        }

        // Otherwise create new command list
        unsafe {
            self.device
                .CreateCommandList(0, self.list_type, allocator, None)
        }
    }

    /// Return a closed command list to the pool for recycling in future frames
    pub fn release_command_list(&self, cmd: ID3D12GraphicsCommandList) {
        let mut cache = self.command_list_cache.lock().unwrap();
        cache.push(cmd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Dx12DeviceContext;
    use windows::Win32::Graphics::Direct3D12::D3D12_COMMAND_LIST_TYPE_DIRECT;

    #[test]
    fn test_worker_command_ring_fence_synchronization() {
        if let Ok(ctx) = Dx12DeviceContext::new(false) {
            let manager = MultiThreadedCommandManager::new(
                &ctx.device,
                4,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
            )
            .expect("Failed to create MultiThreadedCommandManager");

            assert_eq!(manager.worker_count(), 4);

            let worker = manager.worker(0).unwrap();

            // Frame 0: completed fence 0 >= required 0 -> succeeds
            let alloc_f0 = worker.begin_frame(0, 0);
            assert!(alloc_f0.is_ok());
            worker.tag_submission_fence(0, 100);

            // Frame 1: slot 1, completed fence 0 >= required 0 -> succeeds
            let alloc_f1 = worker.begin_frame(1, 0);
            assert!(alloc_f1.is_ok());
            worker.tag_submission_fence(1, 101);

            // Frame 2: slot 2, completed fence 0 >= required 0 -> succeeds
            let alloc_f2 = worker.begin_frame(2, 0);
            assert!(alloc_f2.is_ok());
            worker.tag_submission_fence(2, 102);

            // Frame 3 wraps to slot 0: required fence is 100!
            // GPU completed fence is only 99 -> MUST return stall error!
            let alloc_f3_stalled = worker.begin_frame(3, 99);
            assert!(alloc_f3_stalled.is_err(), "Expected GPU fence stall error");

            // GPU completed fence reaches 100 -> succeeds!
            let alloc_f3_ready = worker.begin_frame(3, 100);
            assert!(alloc_f3_ready.is_ok(), "Expected slot 0 to reset once fence reached");

            // Acquire and release a command list
            let cmd_list = manager.acquire_command_list(alloc_f3_ready.unwrap()).unwrap();
            unsafe {
                cmd_list.Close().unwrap();
            }
            manager.release_command_list(cmd_list);
        } else {
            eprintln!("Skipping hardware allocator test: D3D12 adapter unavailable");
        }
    }
}
