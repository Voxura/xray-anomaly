//! Hardware End-to-End DirectX 12 Verification Test
//!
//! Executes physical GPU operations against Intel(R) HD Graphics 5500:
//! - Window creation (Win32 HWND using predefined STATIC class)
//! - Flip-Discard swapchain creation and RTV descriptor heap binding
//! - Frame recording, barrier transitions, clearing, submission, and present
//! - Swapchain resize under GPU load with resource recreation
//! - Fence tracking and retirement
//! - C ABI boundary smoke test

use std::ptr;
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, WS_OVERLAPPEDWINDOW,
};
use xray_render_dx12::ffi::*;
use xray_render_dx12::renderer::{DeviceState, Dx12Renderer, RenderStats};

fn create_test_window(width: u32, height: u32) -> HWND {
    unsafe {
        CreateWindowExW(
            windows::Win32::UI::WindowsAndMessaging::WINDOW_EX_STYLE(0),
            w!("STATIC"),
            w!("XRay DX12 Verification Window"),
            WS_OVERLAPPEDWINDOW,
            0,
            0,
            width as i32,
            height as i32,
            None,
            None,
            None,
            None,
        )
        .expect("Failed to create Win32 test window")
    }
}

#[test]
fn test_hardware_end_to_end_swapchain_and_c_abi() {
    let hwnd = create_test_window(800, 600);
    assert!(!hwnd.0.is_null());

    // 1. Rust High-Level Renderer Pipeline Test
    {
        let mut renderer = Dx12Renderer::new(2)
            .expect("Failed to initialize hardware Dx12Renderer on host adapter");

        assert_eq!(renderer.device_state(), DeviceState::Ok);

        // Attach native window and initialize Flip-Discard swapchain
        renderer
            .attach_window(hwnd, 800, 600)
            .expect("Failed to attach window and create swapchain");

        // Record & present Frame 0
        {
            let _cmd = renderer.begin_frame().expect("Frame 0 begin failed");
            renderer.clear_backbuffer([0.2, 0.4, 0.8, 1.0]);
            renderer.end_frame(false).expect("Frame 0 end/present failed");
        }

        // Record & present Frame 1
        {
            let _cmd = renderer.begin_frame().expect("Frame 1 begin failed");
            renderer.clear_backbuffer([0.1, 0.1, 0.1, 1.0]);
            renderer.end_frame(false).expect("Frame 1 end/present failed");
        }

        // Resize window to 1024x768 while resources are active
        renderer
            .resize(1024, 768)
            .expect("Resize to 1024x768 failed");

        assert_eq!(renderer.width, 1024);
        assert_eq!(renderer.height, 768);

        // Record & present Frame 2 after resize
        {
            let _cmd = renderer.begin_frame().expect("Frame 2 post-resize begin failed");
            renderer.clear_backbuffer([0.5, 0.5, 0.0, 1.0]);
            renderer.end_frame(false).expect("Frame 2 post-resize end/present failed");
        }

        renderer.wait_for_gpu().expect("GPU flush failed");

        let stats = renderer.stats();
        assert_eq!(stats.frame_index, 3);
        assert_eq!(stats.width, 1024);
        assert_eq!(stats.height, 768);
        assert!(stats.completed_gpu_fence >= 3);
    }

    // 2. C ABI Lifecycle & Smoke Test
    unsafe {
        let mut c_renderer: *mut Dx12Renderer = ptr::null_mut();
        assert_eq!(xr_dx12_renderer_create(2, &mut c_renderer), XR_DX12_OK);
        assert!(!c_renderer.is_null());

        assert_eq!(
            xr_dx12_renderer_attach_window(c_renderer, hwnd.0, 800, 600),
            XR_DX12_OK
        );

        // Frame via C ABI
        assert_eq!(xr_dx12_renderer_begin_frame(c_renderer), XR_DX12_OK);
        assert_eq!(xr_dx12_renderer_clear(c_renderer, 0.0, 0.5, 0.5, 1.0), XR_DX12_OK);
        assert_eq!(xr_dx12_renderer_end_frame(c_renderer, 0), XR_DX12_OK);

        // Resize via C ABI
        assert_eq!(xr_dx12_renderer_resize(c_renderer, 640, 480), XR_DX12_OK);

        // Frame post-resize via C ABI
        assert_eq!(xr_dx12_renderer_begin_frame(c_renderer), XR_DX12_OK);
        assert_eq!(xr_dx12_renderer_clear(c_renderer, 0.3, 0.3, 0.3, 1.0), XR_DX12_OK);
        assert_eq!(xr_dx12_renderer_end_frame(c_renderer, 0), XR_DX12_OK);

        assert_eq!(xr_dx12_renderer_wait_for_gpu(c_renderer), XR_DX12_OK);

        let mut state: i32 = -1;
        assert_eq!(xr_dx12_renderer_get_device_state(c_renderer, &mut state), XR_DX12_OK);
        assert_eq!(state, 0);

        let mut stats = RenderStats::default();
        assert_eq!(xr_dx12_renderer_get_stats(c_renderer, &mut stats), XR_DX12_OK);
        assert_eq!(stats.frame_index, 2);
        assert_eq!(stats.width, 640);
        assert_eq!(stats.height, 480);

        assert_eq!(xr_dx12_renderer_destroy(c_renderer), XR_DX12_OK);
        let _ = DestroyWindow(hwnd);
    }
}
