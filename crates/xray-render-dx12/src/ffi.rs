//! C Application Binary Interface (ABI) for DirectX 12 Backend Interop
//!
//! Exposes an extern "C" interface matching X-Ray's `IRenderDeviceRender` and factory requirements.
//! Every entry point contains complete panic boundary containment (`std::panic::catch_unwind`),
//! pointer validation, and deterministic status codes.

use std::ffi::c_void;
use std::panic::catch_unwind;
use windows::Win32::Foundation::HWND;

use crate::renderer::{DeviceState, Dx12Renderer, RenderStats};
pub use crate::geometry_ffi::*;

pub const XR_DX12_OK: i32 = 0;
pub const XR_DX12_ERR_NULL_PTR: i32 = -1;
pub const XR_DX12_ERR_PANIC: i32 = -2;
pub const XR_DX12_ERR_D3D12: i32 = -3;
pub const XR_DX12_ERR_INVALID_STATE: i32 = -4;
pub const XR_DX12_ERR_INVALID_ARGUMENT: i32 = -5;
pub const XR_DX12_ERR_INVALID_HANDLE: i32 = -6;

/// Allocate and initialize a new DirectX 12 renderer instance
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_create(
    worker_count: u32,
    out_ptr: *mut *mut Dx12Renderer,
) -> i32 {
    let result = catch_unwind(|| {
        if out_ptr.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        match Dx12Renderer::new(worker_count as usize) {
            Ok(renderer) => {
                unsafe {
                    *out_ptr = Box::into_raw(Box::new(renderer));
                }
                XR_DX12_OK
            }
            Err(_) => XR_DX12_ERR_D3D12,
        }
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// Destroy a previously allocated DirectX 12 renderer instance
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_destroy(renderer: *mut Dx12Renderer) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }
        unsafe {
            drop(Box::from_raw(renderer));
        }
        XR_DX12_OK
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// Attach a Win32 HWND and initialize the presentation swapchain
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_attach_window(
    renderer: *mut Dx12Renderer,
    hwnd: *mut c_void,
    width: u32,
    height: u32,
) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() || hwnd.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        let r = unsafe { &mut *renderer };
        let win_hwnd = HWND(hwnd);

        match r.attach_window(win_hwnd, width, height) {
            Ok(_) => XR_DX12_OK,
            Err(_) => XR_DX12_ERR_D3D12,
        }
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// Resize swapchain backbuffers upon window dimension changes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_resize(
    renderer: *mut Dx12Renderer,
    width: u32,
    height: u32,
) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        let r = unsafe { &mut *renderer };
        match r.resize(width, height) {
            Ok(_) => XR_DX12_OK,
            Err(_) => XR_DX12_ERR_D3D12,
        }
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// Begin frame recording: resets command allocator and prepares backbuffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_begin_frame(renderer: *mut Dx12Renderer) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        let r = unsafe { &mut *renderer };
        match r.begin_frame() {
            Ok(_) => XR_DX12_OK,
            Err(_) => XR_DX12_ERR_D3D12,
        }
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// Clear active backbuffer target with RGBA values
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_clear(
    renderer: *mut Dx12Renderer,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        let rend = unsafe { &*renderer };
        rend.clear_backbuffer([r, g, b, a]);
        XR_DX12_OK
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// End frame recording: closes command list, executes on direct queue, and presents swapchain
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_end_frame(
    renderer: *mut Dx12Renderer,
    vsync: u32,
) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        let r = unsafe { &mut *renderer };
        match r.end_frame(vsync != 0) {
            Ok(_) => XR_DX12_OK,
            Err(_) => XR_DX12_ERR_D3D12,
        }
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// Stall CPU thread until all queued GPU work completes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_wait_for_gpu(renderer: *mut Dx12Renderer) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        let r = unsafe { &mut *renderer };
        match r.wait_for_gpu() {
            Ok(_) => XR_DX12_OK,
            Err(_) => XR_DX12_ERR_D3D12,
        }
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// Query hardware device cooperative status (0 = Ok, 1 = Lost, 2 = NeedReset)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_get_device_state(
    renderer: *mut Dx12Renderer,
    out_state: *mut i32,
) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() || out_state.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        let r = unsafe { &*renderer };
        let state = match r.device_state() {
            DeviceState::Ok => 0,
            DeviceState::Lost => 1,
            DeviceState::NeedReset => 2,
        };

        unsafe {
            *out_state = state;
        }
        XR_DX12_OK
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

/// Query frame and GPU fence performance counters
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_renderer_get_stats(
    renderer: *mut Dx12Renderer,
    out_stats: *mut RenderStats,
) -> i32 {
    let result = catch_unwind(|| {
        if renderer.is_null() || out_stats.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }

        let r = unsafe { &*renderer };
        unsafe {
            *out_stats = r.stats();
        }
        XR_DX12_OK
    });

    result.unwrap_or(XR_DX12_ERR_PANIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn test_c_ffi_null_pointer_checks() {
        unsafe {
            assert_eq!(xr_dx12_renderer_create(2, ptr::null_mut()), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_destroy(ptr::null_mut()), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_attach_window(ptr::null_mut(), ptr::null_mut(), 800, 600), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_resize(ptr::null_mut(), 800, 600), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_begin_frame(ptr::null_mut()), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_clear(ptr::null_mut(), 0.0, 0.0, 0.0, 1.0), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_end_frame(ptr::null_mut(), 0), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_wait_for_gpu(ptr::null_mut()), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_get_device_state(ptr::null_mut(), ptr::null_mut()), XR_DX12_ERR_NULL_PTR);
            assert_eq!(xr_dx12_renderer_get_stats(ptr::null_mut(), ptr::null_mut()), XR_DX12_ERR_NULL_PTR);
        }
    }

    #[test]
    fn test_c_ffi_renderer_lifecycle() {
        unsafe {
            let mut renderer_ptr: *mut Dx12Renderer = ptr::null_mut();
            let status = xr_dx12_renderer_create(2, &mut renderer_ptr);
            if status == XR_DX12_OK && !renderer_ptr.is_null() {
                let mut state: i32 = -1;
                assert_eq!(xr_dx12_renderer_get_device_state(renderer_ptr, &mut state), XR_DX12_OK);
                assert_eq!(state, 0); // DeviceState::Ok

                let mut stats = RenderStats::default();
                assert_eq!(xr_dx12_renderer_get_stats(renderer_ptr, &mut stats), XR_DX12_OK);
                assert_eq!(stats.frame_index, 0);

                assert_eq!(xr_dx12_renderer_wait_for_gpu(renderer_ptr), XR_DX12_OK);
                assert_eq!(xr_dx12_renderer_destroy(renderer_ptr), XR_DX12_OK);
            }
        }
    }
}
