//! End-to-end integration tests for X-Ray standalone engine runtime

use xray_engine::camera::{FirstPersonCamera, PITCH_LIMIT};
use xray_engine::game_loop::{EngineConfig, EngineRuntime};
use xray_engine::input::{Action, EngineInput, MouseButton};
use xray_engine::timer::EngineTimer;
use xray_engine::window::{EngineWindow, WindowConfig};

#[test]
fn test_engine_timer_stepping_and_fixed_accumulator() {
    let mut timer = EngineTimer::new();
    assert_eq!(timer.frame_count(), 0);

    // Initial tick
    let dt = timer.tick();
    assert!(dt > 0.0);
    assert!(dt <= 0.1);
    assert_eq!(timer.frame_count(), 1);

    // Fixed timestep step test
    let mut fixed_steps = 0;
    timer.step_fixed(1.0 / 60.0, |_f_dt| {
        fixed_steps += 1;
    });
    // Should run at least 0 to 4 steps
    assert!(fixed_steps <= 4);
}

#[test]
fn test_first_person_camera_math_and_clamping() {
    let mut cam = FirstPersonCamera::new(1920, 1080);
    assert_eq!(cam.position, [0.0, 1.8, 0.0]);
    assert_eq!(cam.yaw, 0.0);
    assert_eq!(cam.pitch, 0.0);

    // Forward direction at yaw=0, pitch=0 should be +Z
    let fwd = cam.forward();
    assert!((fwd[0] - 0.0).abs() < 1e-4);
    assert!((fwd[1] - 0.0).abs() < 1e-4);
    assert!((fwd[2] - 1.0).abs() < 1e-4);

    // Test pitch clamping
    cam.rotate(0.0, 1000.0, 0.01); // Try to look way up
    assert!(cam.pitch <= PITCH_LIMIT);
    assert!(cam.pitch >= -PITCH_LIMIT);

    cam.rotate(0.0, -2000.0, 0.01); // Try to look way down
    assert!(cam.pitch <= PITCH_LIMIT);
    assert!(cam.pitch >= -PITCH_LIMIT);

    // Move camera relative
    cam.pitch = 0.0;
    cam.yaw = 0.0;
    cam.move_relative([0.0, 1.0], 10.0, 0.1); // Move forward 1m
    assert!((cam.position[2] - 1.0).abs() < 1e-4);

    // Check DX12 CameraConstants conversion
    let constants = cam.to_render_constants();
    assert_eq!(constants.cam_pos[0], cam.position[0]);
    assert_eq!(constants.cam_pos[1], cam.position[1]);
    assert_eq!(constants.cam_pos[2], cam.position[2]);
    assert_eq!(constants.cam_pos[3], 1.0);
}

#[test]
fn test_engine_input_edge_detection_and_intent() {
    let mut input = EngineInput::new();

    // Frame 1: Press W
    input.begin_frame();
    input.on_key_down(EngineInput::VK_KEY_W);
    assert!(input.is_key_down(EngineInput::VK_KEY_W));
    assert!(input.is_key_just_pressed(EngineInput::VK_KEY_W));
    assert!(!input.is_key_just_released(EngineInput::VK_KEY_W));
    assert!(input.is_action_down(Action::MoveForward));
    assert!(input.is_action_just_pressed(Action::MoveForward));

    let intent = input.movement_intent();
    assert!((intent[0] - 0.0).abs() < 1e-4); // No horizontal
    assert!((intent[1] - 1.0).abs() < 1e-4); // Forward +Z

    // Frame 2: Hold W, Press A
    input.begin_frame();
    assert!(input.is_key_down(EngineInput::VK_KEY_W));
    assert!(!input.is_key_just_pressed(EngineInput::VK_KEY_W)); // Not just pressed anymore

    input.on_key_down(EngineInput::VK_KEY_A);
    assert!(input.is_action_down(Action::MoveLeft));
    assert!(input.is_action_just_pressed(Action::MoveLeft));

    let intent_diagonal = input.movement_intent();
    // Diagonal should be normalized to length 1.0
    let len = (intent_diagonal[0] * intent_diagonal[0] + intent_diagonal[1] * intent_diagonal[1]).sqrt();
    assert!((len - 1.0).abs() < 1e-4);

    // Frame 3: Release W
    input.begin_frame();
    input.on_key_up(EngineInput::VK_KEY_W);
    assert!(!input.is_key_down(EngineInput::VK_KEY_W));
    assert!(input.is_key_just_released(EngineInput::VK_KEY_W));
    assert!(!input.is_action_down(Action::MoveForward));

    // Mouse buttons
    input.begin_frame();
    input.on_mouse_down(MouseButton::Left);
    assert!(input.is_mouse_down(MouseButton::Left));
    assert!(input.is_mouse_just_pressed(MouseButton::Left));
    assert!(input.is_action_down(Action::PrimaryFire));
}

#[test]
fn test_engine_window_headless_creation_and_polling() {
    let config = WindowConfig {
        title: "TestHeadlessWindow".to_string(),
        width: 320,
        height: 240,
        visible: false, // Don't show UI on screen
    };

    let mut window = EngineWindow::new(&config).expect("failed to create window");
    assert!(!window.hwnd().0.is_null());
    assert_eq!(window.width(), 320);
    assert_eq!(window.height(), 240);
    assert!(!window.is_closed());

    let mut input = EngineInput::new();
    let still_running = window.poll_events(&mut input);
    assert!(still_running);
}

#[test]
fn test_engine_runtime_headless_lifecycle() {
    let config = EngineConfig {
        window: WindowConfig {
            title: "TestLifecycleRuntime".to_string(),
            width: 320,
            height: 240,
            visible: false,
        },
        fixed_dt: 1.0 / 60.0,
        vsync: false,
        move_speed: 5.0,
        sprint_speed: 9.0,
        crouch_speed: 2.5,
    };

    let mut runtime = EngineRuntime::new(config, false).expect("runtime init failed");
    assert!(runtime.running);
    assert!(runtime.renderer.is_none());

    // Step 10 frames
    for _ in 0..10 {
        let active = runtime.step_frame().expect("step_frame failed");
        assert!(active);
    }

    assert!(runtime.timer.frame_count() >= 10);
}
