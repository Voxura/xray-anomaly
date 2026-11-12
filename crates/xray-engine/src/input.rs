//! Input state tracking and action mapping
//!
//! Tracks keyboard and mouse states with edge detection (pressed / just pressed /
//! just released) and translates raw Win32 input events into X-Ray gameplay actions.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Jump,
    Sprint,
    Crouch,
    Use,
    Reload,
    Inventory,
    Pda,
    Menu,
    Slot1,
    Slot2,
    Slot3,
    Slot4,
    Slot5,
    Slot6,
    PrimaryFire,
    SecondaryFire,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Debug, Clone)]
pub struct EngineInput {
    // 256 virtual key state bitmaps
    key_down: [bool; 256],
    key_just_pressed: [bool; 256],
    key_just_released: [bool; 256],

    // Mouse button states
    mouse_down: [bool; 3],
    mouse_just_pressed: [bool; 3],
    mouse_just_released: [bool; 3],

    // Mouse coordinates and deltas
    mouse_pos: (f32, f32),
    mouse_delta: (f32, f32),
    wheel_delta: f32,

    // Mouse capture / lock state
    cursor_locked: bool,
    sensitivity: f32,
}

impl Default for EngineInput {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineInput {
    pub const VK_KEY_0: usize = 0x30;
    pub const VK_KEY_1: usize = 0x31;
    pub const VK_KEY_2: usize = 0x32;
    pub const VK_KEY_3: usize = 0x33;
    pub const VK_KEY_4: usize = 0x34;
    pub const VK_KEY_5: usize = 0x35;
    pub const VK_KEY_6: usize = 0x36;
    pub const VK_KEY_A: usize = 0x41;
    pub const VK_KEY_D: usize = 0x44;
    pub const VK_KEY_E: usize = 0x45;
    pub const VK_KEY_I: usize = 0x49;
    pub const VK_KEY_P: usize = 0x50;
    pub const VK_KEY_R: usize = 0x52;
    pub const VK_KEY_S: usize = 0x53;
    pub const VK_KEY_W: usize = 0x57;
    pub const VK_TAB: usize = 0x09;
    pub const VK_ESCAPE: usize = 0x1B;
    pub const VK_SPACE: usize = 0x20;
    pub const VK_SHIFT: usize = 0x10;
    pub const VK_CONTROL: usize = 0x11;

    pub fn new() -> Self {
        Self {
            key_down: [false; 256],
            key_just_pressed: [false; 256],
            key_just_released: [false; 256],
            mouse_down: [false; 3],
            mouse_just_pressed: [false; 3],
            mouse_just_released: [false; 3],
            mouse_pos: (0.0, 0.0),
            mouse_delta: (0.0, 0.0),
            wheel_delta: 0.0,
            cursor_locked: true,
            sensitivity: 0.002,
        }
    }

    /// Reset per-frame edge detections and deltas. Call at start of each frame.
    pub fn begin_frame(&mut self) {
        self.key_just_pressed = [false; 256];
        self.key_just_released = [false; 256];
        self.mouse_just_pressed = [false; 3];
        self.mouse_just_released = [false; 3];
        self.mouse_delta = (0.0, 0.0);
        self.wheel_delta = 0.0;
    }

    /// Record a key down event
    pub fn on_key_down(&mut self, vk: usize) {
        if vk < 256 {
            if !self.key_down[vk] {
                self.key_just_pressed[vk] = true;
            }
            self.key_down[vk] = true;
        }
    }

    /// Record a key up event
    pub fn on_key_up(&mut self, vk: usize) {
        if vk < 256 {
            if self.key_down[vk] {
                self.key_just_released[vk] = true;
            }
            self.key_down[vk] = false;
        }
    }

    /// Record mouse position change and accumulate delta
    pub fn on_mouse_move(&mut self, x: f32, y: f32) {
        let dx = x - self.mouse_pos.0;
        let dy = y - self.mouse_pos.1;
        self.mouse_pos = (x, y);
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Explicitly supply relative mouse motion (e.g. from RawInput or centered cursor)
    pub fn on_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Record mouse button down
    pub fn on_mouse_down(&mut self, button: MouseButton) {
        let idx = button as usize;
        if idx < 3 {
            if !self.mouse_down[idx] {
                self.mouse_just_pressed[idx] = true;
            }
            self.mouse_down[idx] = true;
        }
    }

    /// Record mouse button up
    pub fn on_mouse_up(&mut self, button: MouseButton) {
        let idx = button as usize;
        if idx < 3 {
            if self.mouse_down[idx] {
                self.mouse_just_released[idx] = true;
            }
            self.mouse_down[idx] = false;
        }
    }

    /// Record mouse wheel delta
    pub fn on_mouse_wheel(&mut self, delta: f32) {
        self.wheel_delta += delta;
    }

    /// Check if virtual key is currently held down
    #[inline]
    pub fn is_key_down(&self, vk: usize) -> bool {
        if vk < 256 {
            self.key_down[vk]
        } else {
            false
        }
    }

    /// Check if virtual key was pressed down this frame
    #[inline]
    pub fn is_key_just_pressed(&self, vk: usize) -> bool {
        if vk < 256 {
            self.key_just_pressed[vk]
        } else {
            false
        }
    }

    /// Check if virtual key was released this frame
    #[inline]
    pub fn is_key_just_released(&self, vk: usize) -> bool {
        if vk < 256 {
            self.key_just_released[vk]
        } else {
            false
        }
    }

    /// Check if mouse button is held down
    #[inline]
    pub fn is_mouse_down(&self, button: MouseButton) -> bool {
        self.mouse_down[button as usize]
    }

    /// Check if mouse button was pressed down this frame
    #[inline]
    pub fn is_mouse_just_pressed(&self, button: MouseButton) -> bool {
        self.mouse_just_pressed[button as usize]
    }

    /// Check if mouse button was released this frame
    #[inline]
    pub fn is_mouse_just_released(&self, button: MouseButton) -> bool {
        self.mouse_just_released[button as usize]
    }

    /// Current pointer coordinates in client pixels `[x, y]`
    #[inline]
    pub fn mouse_position(&self) -> (f32, f32) {
        self.mouse_pos
    }

    /// Return a cloned input snapshot with weapon firing / aiming / slot keys cleared while modal UI is active.
    pub fn without_combat_inputs(&self) -> Self {
        let mut copy = self.clone();
        copy.mouse_down = [false; 3];
        copy.mouse_just_pressed = [false; 3];
        copy.mouse_just_released = [false; 3];
        for vk in [
            Self::VK_KEY_1,
            Self::VK_KEY_2,
            Self::VK_KEY_3,
            Self::VK_KEY_4,
            Self::VK_KEY_5,
            Self::VK_KEY_6,
            Self::VK_KEY_R,
        ] {
            copy.key_down[vk] = false;
            copy.key_just_pressed[vk] = false;
        }
        copy
    }

    /// Check if action is currently active
    pub fn is_action_down(&self, action: Action) -> bool {
        match action {
            Action::MoveForward => self.is_key_down(Self::VK_KEY_W),
            Action::MoveBackward => self.is_key_down(Self::VK_KEY_S),
            Action::MoveLeft => self.is_key_down(Self::VK_KEY_A),
            Action::MoveRight => self.is_key_down(Self::VK_KEY_D),
            Action::Jump => self.is_key_down(Self::VK_SPACE),
            Action::Sprint => self.is_key_down(Self::VK_SHIFT),
            Action::Crouch => self.is_key_down(Self::VK_CONTROL),
            Action::Use => self.is_key_down(Self::VK_KEY_E),
            Action::Reload => self.is_key_down(Self::VK_KEY_R),
            Action::Inventory => self.is_key_down(Self::VK_TAB) || self.is_key_down(Self::VK_KEY_I),
            Action::Pda => self.is_key_down(Self::VK_KEY_P),
            Action::Menu => self.is_key_down(Self::VK_ESCAPE),
            Action::Slot1 => self.is_key_down(Self::VK_KEY_1),
            Action::Slot2 => self.is_key_down(Self::VK_KEY_2),
            Action::Slot3 => self.is_key_down(Self::VK_KEY_3),
            Action::Slot4 => self.is_key_down(Self::VK_KEY_4),
            Action::Slot5 => self.is_key_down(Self::VK_KEY_5),
            Action::Slot6 => self.is_key_down(Self::VK_KEY_6),
            Action::PrimaryFire => self.is_mouse_down(MouseButton::Left),
            Action::SecondaryFire => self.is_mouse_down(MouseButton::Right),
        }
    }

    /// Check if action was triggered this frame
    pub fn is_action_just_pressed(&self, action: Action) -> bool {
        match action {
            Action::MoveForward => self.is_key_just_pressed(Self::VK_KEY_W),
            Action::MoveBackward => self.is_key_just_pressed(Self::VK_KEY_S),
            Action::MoveLeft => self.is_key_just_pressed(Self::VK_KEY_A),
            Action::MoveRight => self.is_key_just_pressed(Self::VK_KEY_D),
            Action::Jump => self.is_key_just_pressed(Self::VK_SPACE),
            Action::Sprint => self.is_key_just_pressed(Self::VK_SHIFT),
            Action::Crouch => self.is_key_just_pressed(Self::VK_CONTROL),
            Action::Use => self.is_key_just_pressed(Self::VK_KEY_E),
            Action::Reload => self.is_key_just_pressed(Self::VK_KEY_R),
            Action::Inventory => {
                self.is_key_just_pressed(Self::VK_TAB) || self.is_key_just_pressed(Self::VK_KEY_I)
            }
            Action::Pda => self.is_key_just_pressed(Self::VK_KEY_P),
            Action::Menu => self.is_key_just_pressed(Self::VK_ESCAPE),
            Action::Slot1 => self.is_key_just_pressed(Self::VK_KEY_1),
            Action::Slot2 => self.is_key_just_pressed(Self::VK_KEY_2),
            Action::Slot3 => self.is_key_just_pressed(Self::VK_KEY_3),
            Action::Slot4 => self.is_key_just_pressed(Self::VK_KEY_4),
            Action::Slot5 => self.is_key_just_pressed(Self::VK_KEY_5),
            Action::Slot6 => self.is_key_just_pressed(Self::VK_KEY_6),
            Action::PrimaryFire => self.is_mouse_just_pressed(MouseButton::Left),
            Action::SecondaryFire => self.is_mouse_just_pressed(MouseButton::Right),
        }
    }

    /// Returns normalized 2D movement intent `[x, z]` where +Z is forward and +X is right
    pub fn movement_intent(&self) -> [f32; 2] {
        let mut x = 0.0f32;
        let mut z = 0.0f32;

        if self.is_action_down(Action::MoveForward) {
            z += 1.0;
        }
        if self.is_action_down(Action::MoveBackward) {
            z -= 1.0;
        }
        if self.is_action_down(Action::MoveRight) {
            x += 1.0;
        }
        if self.is_action_down(Action::MoveLeft) {
            x -= 1.0;
        }

        let len_sq = x * x + z * z;
        if len_sq > 1e-6 {
            let inv = 1.0 / len_sq.sqrt();
            [x * inv, z * inv]
        } else {
            [0.0, 0.0]
        }
    }

    #[inline]
    pub fn mouse_delta(&self) -> (f32, f32) {
        self.mouse_delta
    }

    #[inline]
    pub fn wheel_delta(&self) -> f32 {
        self.wheel_delta
    }

    #[inline]
    pub fn sensitivity(&self) -> f32 {
        self.sensitivity
    }

    pub fn set_sensitivity(&mut self, sens: f32) {
        self.sensitivity = sens.clamp(0.0001, 0.05);
    }

    #[inline]
    pub fn is_cursor_locked(&self) -> bool {
        self.cursor_locked
    }

    pub fn set_cursor_locked(&mut self, locked: bool) {
        self.cursor_locked = locked;
    }
}
