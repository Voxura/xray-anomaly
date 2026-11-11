//! Phase 4 wire layout and validation, independent of the Windows bindings.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub type MeshHandle = u64;
pub const INVALID_MESH_HANDLE: MeshHandle = 0;
pub const INDEX_FORMAT_U16: u32 = 57; // DXGI_FORMAT_R16_UINT
pub const INDEX_FORMAT_U32: u32 = 42; // DXGI_FORMAT_R32_UINT
pub const VERTEX_STRIDE: u32 = 44;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct XRayVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub tangent: [f32; 3],
}

/// Matrices are column-major, multiplying column vectors in HLSL.
#[repr(C, align(256))]
#[derive(Debug, Clone, Copy, Default)]
pub struct CameraConstants {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub padding: [f32; 12],
}

const _: () = assert!(std::mem::size_of::<XRayVertex>() == 44);
const _: () = assert!(std::mem::size_of::<CameraConstants>() == 256);
const _: () = assert!(std::mem::align_of::<CameraConstants>() == 256);

/// Sizes fit both Rust slices and D3D12's UINT-sized buffer views.
pub fn mesh_byte_sizes(
    vertex_count: u32,
    stride: u32,
    index_count: u32,
    index_format: u32,
) -> Option<(usize, usize)> {
    if vertex_count == 0 || index_count == 0 || stride != VERTEX_STRIDE {
        return None;
    }
    let index_size = match index_format {
        INDEX_FORMAT_U16 => 2,
        INDEX_FORMAT_U32 => 4,
        _ => return None,
    };
    let vertices = vertex_count.checked_mul(stride)? as usize;
    let indices = index_count.checked_mul(index_size)? as usize;
    if vertices > isize::MAX as usize || indices > isize::MAX as usize {
        return None;
    }
    Some((vertices, indices))
}

pub fn indices_in_bounds(indices: &[u8], format: u32, vertex_count: u32) -> bool {
    match format {
        INDEX_FORMAT_U16 => {
            indices.len() % 2 == 0
                && indices
                    .chunks_exact(2)
                    .all(|b| (u16::from_ne_bytes([b[0], b[1]]) as u32) < vertex_count)
        }
        INDEX_FORMAT_U32 => {
            indices.len() % 4 == 0
                && indices
                    .chunks_exact(4)
                    .all(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]) < vertex_count)
        }
        _ => false,
    }
}

/// Globally unique IDs prevent stale or foreign-renderer handles from aliasing.
/// Exhaustion returns None rather than recycling an ID.
#[derive(Debug)]
pub struct MeshRegistry<T>(HashMap<MeshHandle, T>);

impl<T> Default for MeshRegistry<T> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl<T> MeshRegistry<T> {
    pub fn insert(&mut self, value: T) -> Option<MeshHandle> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let handle = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .ok()?;
        self.0.insert(handle, value);
        Some(handle)
    }

    pub fn get(&self, handle: MeshHandle) -> Option<&T> {
        self.0.get(&handle)
    }

    pub fn remove(&mut self, handle: MeshHandle) -> Option<T> {
        self.0.remove(&handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn phase4_wire_offsets() {
        assert_eq!(size_of::<XRayVertex>(), 44);
        assert_eq!(offset_of!(XRayVertex, position), 0);
        assert_eq!(offset_of!(XRayVertex, normal), 12);
        assert_eq!(offset_of!(XRayVertex, uv), 24);
        assert_eq!(offset_of!(XRayVertex, tangent), 32);
        assert_eq!(size_of::<CameraConstants>(), 256);
        assert_eq!(align_of::<CameraConstants>(), 256);
        assert_eq!(offset_of!(CameraConstants, view), 0);
        assert_eq!(offset_of!(CameraConstants, proj), 64);
        assert_eq!(offset_of!(CameraConstants, view_proj), 128);
        assert_eq!(offset_of!(CameraConstants, cam_pos), 192);
        assert_eq!(offset_of!(CameraConstants, padding), 208);
    }

    #[test]
    fn phase4_rejects_invalid_sizes_and_formats() {
        assert_eq!(mesh_byte_sizes(3, 44, 3, INDEX_FORMAT_U16), Some((132, 6)));
        assert_eq!(mesh_byte_sizes(3, 44, 3, INDEX_FORMAT_U32), Some((132, 12)));
        for args in [
            (0, 44, 3, 57),
            (3, 48, 3, 57),
            (3, 44, 0, 57),
            (3, 44, 3, 16),
            (u32::MAX, 44, 3, 57),
            (3, 44, u32::MAX, 42),
        ] {
            assert_eq!(mesh_byte_sizes(args.0, args.1, args.2, args.3), None);
        }
        assert!(indices_in_bounds(&[0, 0, 2, 0], INDEX_FORMAT_U16, 3));
        assert!(!indices_in_bounds(&[3, 0], INDEX_FORMAT_U16, 3));
        assert!(!indices_in_bounds(&[0], INDEX_FORMAT_U16, 3));
        assert!(indices_in_bounds(&2u32.to_ne_bytes(), INDEX_FORMAT_U32, 3));
        assert!(!indices_in_bounds(&3u32.to_ne_bytes(), INDEX_FORMAT_U32, 3));
        assert!(!indices_in_bounds(&[0, 0], 16, 3));
    }

    #[test]
    fn phase4_handles_reject_stale_and_foreign_ids() {
        let mut first = MeshRegistry::default();
        let mut second = MeshRegistry::default();
        let old = first.insert(1).unwrap();
        let foreign = second.insert(2).unwrap();
        assert!(first.get(INVALID_MESH_HANDLE).is_none());
        assert!(first.get(foreign).is_none());
        assert_eq!(first.remove(old), Some(1));
        assert!(first.remove(old).is_none());
        let new = first.insert(3).unwrap();
        assert_ne!(old, new);
        assert!(first.get(old).is_none());
        assert_eq!(first.get(new), Some(&3));
    }
}
