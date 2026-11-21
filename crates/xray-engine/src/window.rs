//! Win32 Native Application Window & Message Pump
//!
//! Provides pure-Rust window creation, message dispatching, and raw hardware
//! input event forwarding for DirectX 12 rendering and X-Ray game loop ticks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONEAREST};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    GetClientRect, GetForegroundWindow, GetWindowLongPtrW, GetWindowPlacement, PeekMessageW, PostQuitMessage,
    LoadCursorW, RegisterClassExW, SetCursor, SetWindowLongPtrW, SetWindowPlacement, SetWindowPos,
    ShowWindow, TranslateMessage, CS_HREDRAW, CS_VREDRAW, CW_USEDEFAULT, GWLP_USERDATA, MSG,
    PM_REMOVE, SW_HIDE, SW_SHOW, WM_CLOSE, WM_DESTROY, WM_KEYDOWN, WM_KEYUP, WM_KILLFOCUS,
    WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEMOVE, WM_MOUSEWHEEL,
    WM_QUIT, WM_RBUTTONDOWN, WM_RBUTTONUP, WM_SETFOCUS, WM_SIZE, WM_SYSKEYDOWN, WM_SYSKEYUP,
    HTCLIENT, IDC_ARROW, WM_SETCURSOR, WNDCLASSEXW, WS_EX_APPWINDOW, WS_OVERLAPPEDWINDOW,
    GWL_STYLE, SWP_FRAMECHANGED, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, WINDOWPLACEMENT,
};

use crate::input::{EngineInput, MouseButton};

#[cfg(test)]
#[test]
fn borderless_fullscreen_covers_the_monitor_and_restores_the_client() {
    let mut window = EngineWindow::new(&WindowConfig {
        width: 640,
        height: 480,
        visible: false,
        ..Default::default()
    }).unwrap();
    window.toggle_fullscreen().unwrap();
    let mut monitor = MONITORINFO { cbSize: size_of::<MONITORINFO>() as u32, ..Default::default() };
    unsafe { GetMonitorInfoW(MonitorFromWindow(window.hwnd(), MONITOR_DEFAULTTONEAREST), &mut monitor).ok().unwrap(); }
    assert_eq!(window.width(), (monitor.rcMonitor.right - monitor.rcMonitor.left) as u32);
    assert_eq!(window.height(), (monitor.rcMonitor.bottom - monitor.rcMonitor.top) as u32);
    window.toggle_fullscreen().unwrap();
    assert_eq!((window.width(), window.height()), (640, 480));
}

#[derive(Debug, Clone)]
pub struct WindowConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub visible: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            title: "X-Ray Monolith Pure-Rust Engine".to_string(),
            width: 1280,
            height: 720,
            visible: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowEvent {
    Resized { width: u32, height: u32 },
    CloseRequested,
    KeyDown { vk: usize },
    KeyUp { vk: usize },
    MouseMove { x: f32, y: f32 },
    MouseDown { button: MouseButton },
    MouseUp { button: MouseButton },
    MouseWheel { delta: f32 },
    FocusChanged(bool),
}

struct WindowInternalState {
    events: Mutex<Vec<WindowEvent>>,
    is_closed: AtomicBool,
    menu_cursor: AtomicBool,
}

pub struct EngineWindow {
    hwnd: HWND,
    width: u32,
    height: u32,
    windowed: Option<(isize, WINDOWPLACEMENT)>,
    state: Box<WindowInternalState>,
}

impl EngineWindow {
    /// Create and display a native Win32 engine window
    pub fn new(config: &WindowConfig) -> Result<Self, windows::core::Error> {
        let instance = unsafe { GetModuleHandleW(None)? };
        let class_name = w!("XRayMonolithEngineClass");

        // Register window class once
        static CLASS_REGISTERED: AtomicBool = AtomicBool::new(false);
        if !CLASS_REGISTERED.swap(true, Ordering::SeqCst) {
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wnd_proc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: instance.into(),
                hIcon: Default::default(),
                hCursor: Default::default(),
                hbrBackground: Default::default(),
                lpszMenuName: windows::core::PCWSTR::null(),
                lpszClassName: class_name,
                hIconSm: Default::default(),
            };
            unsafe {
                let atom = RegisterClassExW(&wc);
                if atom == 0 {
                    // Ignore error if class was already registered by previous instance
                }
            }
        }

        let mut rect = RECT {
            left: 0,
            top: 0,
            right: config.width as i32,
            bottom: config.height as i32,
        };
        unsafe {
            let _ = AdjustWindowRectEx(
                &mut rect,
                WS_OVERLAPPEDWINDOW,
                false,
                WS_EX_APPWINDOW,
            );
        }
        let total_w = rect.right - rect.left;
        let total_h = rect.bottom - rect.top;

        // Encode title into null-terminated UTF-16
        let title_utf16: Vec<u16> = config.title.encode_utf16().chain(std::iter::once(0)).collect();

        let state = Box::new(WindowInternalState {
            events: Mutex::new(Vec::with_capacity(32)),
            is_closed: AtomicBool::new(false),
            menu_cursor: AtomicBool::new(false),
        });

        let hwnd = unsafe {
            CreateWindowExW(
                WS_EX_APPWINDOW,
                class_name,
                windows::core::PCWSTR(title_utf16.as_ptr()),
                WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                total_w,
                total_h,
                None,
                None,
                Some(instance.into()),
                None,
            )?
        };

        // Attach state pointer to window user data
        let state_ptr = &*state as *const WindowInternalState as isize;
        unsafe {
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, state_ptr);
        }

        if config.visible {
            unsafe {
                let _ = ShowWindow(hwnd, SW_SHOW);
            }
        } else {
            unsafe {
                let _ = ShowWindow(hwnd, SW_HIDE);
            }
        }

        Ok(Self {
            hwnd,
            width: config.width,
            height: config.height,
            windowed: None,
            state,
        })
    }

    #[inline]
    pub fn hwnd(&self) -> HWND {
        self.hwnd
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    #[inline]
    pub fn is_closed(&self) -> bool {
        self.state.is_closed.load(Ordering::SeqCst)
    }

    pub(crate) fn is_foreground(&self) -> bool {
        unsafe { GetForegroundWindow() == self.hwnd }
    }

    /// Toggle borderless fullscreen on this window's current monitor.
    pub(crate) fn toggle_fullscreen(&mut self) -> windows::core::Result<()> {
        let hwnd = self.hwnd;
        let flags = SWP_FRAMECHANGED | SWP_NOACTIVATE | SWP_NOZORDER;
        unsafe {
            if let Some((style, placement)) = self.windowed {
                SetWindowLongPtrW(hwnd, GWL_STYLE, style);
                SetWindowPlacement(hwnd, &placement)?;
                SetWindowPos(hwnd, None, 0, 0, 0, 0, flags | SWP_NOMOVE | SWP_NOSIZE)?;
                self.windowed = None;
            } else {
                let style = GetWindowLongPtrW(hwnd, GWL_STYLE);
                let mut placement = WINDOWPLACEMENT { length: size_of::<WINDOWPLACEMENT>() as u32, ..Default::default() };
                GetWindowPlacement(hwnd, &mut placement)?;
                let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
                let mut info = MONITORINFO { cbSize: size_of::<MONITORINFO>() as u32, ..Default::default() };
                GetMonitorInfoW(monitor, &mut info).ok()?;
                let bounds = info.rcMonitor;
                SetWindowLongPtrW(hwnd, GWL_STYLE, style & !(WS_OVERLAPPEDWINDOW.0 as isize));
                if let Err(err) = SetWindowPos(
                    hwnd, None, bounds.left, bounds.top, bounds.right - bounds.left, bounds.bottom - bounds.top, flags,
                ) {
                    SetWindowLongPtrW(hwnd, GWL_STYLE, style);
                    let _ = SetWindowPlacement(hwnd, &placement);
                    return Err(err);
                }
                self.windowed = Some((style, placement));
            }
            let mut client = RECT::default();
            GetClientRect(hwnd, &mut client)?;
            self.width = client.right.max(1) as u32;
            self.height = client.bottom.max(1) as u32;
        }
        Ok(())
    }

    /// Enqueue a synthetic window event into the pending queue (useful for automated runtime tests).
    pub fn push_synthetic_event(&self, event: WindowEvent) {
        if let Ok(mut events) = self.state.events.lock() {
            events.push(event);
        }
    }

    pub(crate) fn set_menu_cursor(&self, visible: bool) {
        self.state.menu_cursor.store(visible, Ordering::Relaxed);
    }

    /// Pump Windows event queue and dispatch to `EngineInput`
    pub fn poll_events(&mut self, input: &mut EngineInput) -> bool {
        if self.is_closed() {
            return false;
        }

        input.begin_frame();

        let mut msg = MSG::default();
        unsafe {
            while PeekMessageW(&mut msg, Some(self.hwnd), 0, 0, PM_REMOVE).as_bool() {
                if msg.message == WM_QUIT {
                    self.state.is_closed.store(true, Ordering::SeqCst);
                    return false;
                }
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        // Drain collected events from window proc
        let mut events_guard = self.state.events.lock().unwrap();
        for event in events_guard.drain(..) {
            match event {
                WindowEvent::Resized { width, height } => {
                    if width > 0 && height > 0 {
                        self.width = width;
                        self.height = height;
                    }
                }
                WindowEvent::CloseRequested => {
                    self.state.is_closed.store(true, Ordering::SeqCst);
                }
                WindowEvent::KeyDown { vk } => {
                    input.on_key_down(vk);
                }
                WindowEvent::KeyUp { vk } => {
                    input.on_key_up(vk);
                }
                WindowEvent::MouseMove { x, y } => {
                    input.on_mouse_move(x, y);
                }
                WindowEvent::MouseDown { button } => {
                    input.on_mouse_down(button);
                }
                WindowEvent::MouseUp { button } => {
                    input.on_mouse_up(button);
                }
                WindowEvent::MouseWheel { delta } => {
                    input.on_mouse_wheel(delta);
                }
                WindowEvent::FocusChanged(_) => {}
            }
        }

        !self.is_closed()
    }
}

impl Drop for EngineWindow {
    fn drop(&mut self) {
        if !self.hwnd.0.is_null() {
            unsafe {
                // Clear user data before destroying
                SetWindowLongPtrW(self.hwnd, GWLP_USERDATA, 0);
                let _ = DestroyWindow(self.hwnd);
            }
        }
    }
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    let user_data = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) };
    if user_data != 0 {
        let state = unsafe { &*(user_data as *const WindowInternalState) };
        match msg {
            WM_SETCURSOR if (lparam.0 & 0xffff) as u32 == HTCLIENT
                && state.menu_cursor.load(Ordering::Relaxed) =>
            {
                if let Ok(cursor) = unsafe { LoadCursorW(None, IDC_ARROW) } {
                    unsafe { SetCursor(Some(cursor)); }
                    return LRESULT(1);
                }
            }
            WM_CLOSE => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::CloseRequested);
                return LRESULT(0);
            }
            WM_DESTROY => {
                state.is_closed.store(true, Ordering::SeqCst);
                unsafe { PostQuitMessage(0) };
                return LRESULT(0);
            }
            WM_SIZE => {
                let width = (lparam.0 & 0xFFFF) as u32;
                let height = ((lparam.0 >> 16) & 0xFFFF) as u32;
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::Resized { width, height });
                return LRESULT(0);
            }
            WM_KEYDOWN | WM_SYSKEYDOWN => {
                let vk = wparam.0 & 0xFF;
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::KeyDown { vk });
                return LRESULT(0);
            }
            WM_KEYUP | WM_SYSKEYUP => {
                let vk = wparam.0 & 0xFF;
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::KeyUp { vk });
                return LRESULT(0);
            }
            WM_MOUSEMOVE => {
                let x = (lparam.0 as i16) as f32;
                let y = ((lparam.0 >> 16) as i16) as f32;
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::MouseMove { x, y });
                return LRESULT(0);
            }
            WM_LBUTTONDOWN => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::MouseDown {
                    button: MouseButton::Left,
                });
                return LRESULT(0);
            }
            WM_LBUTTONUP => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::MouseUp {
                    button: MouseButton::Left,
                });
                return LRESULT(0);
            }
            WM_RBUTTONDOWN => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::MouseDown {
                    button: MouseButton::Right,
                });
                return LRESULT(0);
            }
            WM_RBUTTONUP => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::MouseUp {
                    button: MouseButton::Right,
                });
                return LRESULT(0);
            }
            WM_MBUTTONDOWN => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::MouseDown {
                    button: MouseButton::Middle,
                });
                return LRESULT(0);
            }
            WM_MBUTTONUP => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::MouseUp {
                    button: MouseButton::Middle,
                });
                return LRESULT(0);
            }
            WM_MOUSEWHEEL => {
                let delta = ((wparam.0 >> 16) as i16) as f32 / 120.0;
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::MouseWheel { delta });
                return LRESULT(0);
            }
            WM_SETFOCUS => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::FocusChanged(true));
                return LRESULT(0);
            }
            WM_KILLFOCUS => {
                let mut guard = state.events.lock().unwrap();
                guard.push(WindowEvent::FocusChanged(false));
                return LRESULT(0);
            }
            _ => {}
        }
    }

    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}
