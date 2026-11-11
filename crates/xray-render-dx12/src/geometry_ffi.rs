//! Phase 4 C ABI. See PHASE4.md for pointer and lifetime requirements.
use crate::ffi::{
    XR_DX12_ERR_D3D12, XR_DX12_ERR_INVALID_ARGUMENT, XR_DX12_ERR_INVALID_HANDLE,
    XR_DX12_ERR_INVALID_STATE, XR_DX12_ERR_NULL_PTR, XR_DX12_ERR_PANIC, XR_DX12_OK,
};
use crate::geometry::PushDrawConstants;
use crate::geometry_abi::{CameraConstants, MeshHandle, indices_in_bounds, mesh_byte_sizes};
use crate::renderer::Dx12Renderer;
use std::ffi::c_void;
use std::panic::catch_unwind;

/// Copy caller-owned geometry into immutable GPU upload buffers.
/// `idx_format` is DXGI_FORMAT_R16_UINT (57) or DXGI_FORMAT_R32_UINT (42).
/// All calls on a renderer must be serialized. Non-null pointers must refer to
/// live, adequately sized memory; arbitrary dangling pointers cannot be validated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_create_mesh(
    renderer: *mut Dx12Renderer,
    vertices: *const c_void,
    vert_count: u32,
    vert_stride: u32,
    indices: *const c_void,
    idx_count: u32,
    idx_format: u32,
    out_handle: *mut MeshHandle,
) -> i32 {
    catch_unwind(|| {
        if out_handle.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }
        unsafe {
            out_handle.write_unaligned(0);
        }
        if renderer.is_null() || vertices.is_null() || indices.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }
        let Some((vertex_bytes, index_bytes)) =
            mesh_byte_sizes(vert_count, vert_stride, idx_count, idx_format)
        else {
            return XR_DX12_ERR_INVALID_ARGUMENT;
        };
        let vertices = unsafe { std::slice::from_raw_parts(vertices.cast::<u8>(), vertex_bytes) };
        let indices = unsafe { std::slice::from_raw_parts(indices.cast::<u8>(), index_bytes) };
        if !indices_in_bounds(indices, idx_format, vert_count) {
            return XR_DX12_ERR_INVALID_ARGUMENT;
        }
        let r = unsafe { &mut *renderer };
        match r.geometry.create_mesh(
            &r.device_ctx.device,
            vertices,
            indices,
            idx_count,
            idx_format,
        ) {
            Ok(handle) => {
                unsafe {
                    out_handle.write_unaligned(handle);
                }
                XR_DX12_OK
            }
            Err(_) => XR_DX12_ERR_D3D12,
        }
    })
    .unwrap_or(XR_DX12_ERR_PANIC)
}

/// Invalidate a handle immediately; recorded/submitted draws retain their buffers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_destroy_mesh(
    renderer: *mut Dx12Renderer,
    handle: MeshHandle,
) -> i32 {
    catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }
        let r = unsafe { &mut *renderer };
        if r.geometry.destroy_mesh(handle) {
            XR_DX12_OK
        } else {
            XR_DX12_ERR_INVALID_HANDLE
        }
    })
    .unwrap_or(XR_DX12_ERR_PANIC)
}

/// Copy 256 bytes. Each update allocates a new GPU snapshot, so previous draws
/// retain the camera they were recorded with (including updates in one frame).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_update_camera(
    renderer: *mut Dx12Renderer,
    camera_constants_ptr: *const CameraConstants,
) -> i32 {
    catch_unwind(|| {
        if renderer.is_null() || camera_constants_ptr.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }
        let constants = unsafe { camera_constants_ptr.read_unaligned() };
        let r = unsafe { &mut *renderer };
        match r.geometry.update_camera(&r.device_ctx.device, &constants) {
            Ok(()) => XR_DX12_OK,
            Err(_) => XR_DX12_ERR_D3D12,
        }
    })
    .unwrap_or(XR_DX12_ERR_PANIC)
}

/// Draw a triangle list to the active backbuffer using the Phase 4 baseline PSO.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_draw_indexed(
    renderer: *mut Dx12Renderer,
    mesh_handle: MeshHandle,
) -> i32 {
    catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }
        let r = unsafe { &mut *renderer };
        if !r.geometry.has_mesh(mesh_handle) {
            return XR_DX12_ERR_INVALID_HANDLE;
        }
        let (Some(cmd), Some(sc)) = (&r.current_cmd_list, &r.swapchain) else {
            return XR_DX12_ERR_INVALID_STATE;
        };
        if !r.in_frame || !r.geometry.has_camera() {
            return XR_DX12_ERR_INVALID_STATE;
        }
        match r.geometry.draw(
            &r.device_ctx.device,
            cmd,
            sc.current_rtv_handle(),
            r.width,
            r.height,
            mesh_handle,
            Some(&r.bindless_heap),
            None,
        ) {
            Ok(()) => XR_DX12_OK,
            Err(_) => XR_DX12_ERR_D3D12,
        }
    })
    .unwrap_or(XR_DX12_ERR_PANIC)
}

/// Record an indexed draw call with a bindless albedo texture descriptor index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn xr_dx12_draw_textured(
    renderer: *mut Dx12Renderer,
    mesh_handle: MeshHandle,
    albedo_index: u32,
) -> i32 {
    catch_unwind(|| {
        if renderer.is_null() {
            return XR_DX12_ERR_NULL_PTR;
        }
        let r = unsafe { &mut *renderer };
        if !r.geometry.has_mesh(mesh_handle) {
            return XR_DX12_ERR_INVALID_HANDLE;
        }
        let (Some(cmd), Some(sc)) = (&r.current_cmd_list, &r.swapchain) else {
            return XR_DX12_ERR_INVALID_STATE;
        };
        if !r.in_frame || !r.geometry.has_camera() {
            return XR_DX12_ERR_INVALID_STATE;
        }
        let push = PushDrawConstants {
            albedo_index,
            ..Default::default()
        };
        match r.geometry.draw(
            &r.device_ctx.device,
            cmd,
            sc.current_rtv_handle(),
            r.width,
            r.height,
            mesh_handle,
            Some(&r.bindless_heap),
            Some(&push),
        ) {
            Ok(()) => XR_DX12_OK,
            Err(_) => XR_DX12_ERR_D3D12,
        }
    })
    .unwrap_or(XR_DX12_ERR_PANIC)
}
