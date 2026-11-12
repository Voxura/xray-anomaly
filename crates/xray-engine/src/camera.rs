//! First-person camera system with X-Ray left-handed coordinate conventions
//!
//! Generates view and projection matrices, updates orientations from mouse deltas,
//! and packs hardware constant buffers for DirectX 12 rendering pipelines.

use std::f32::consts::FRAC_PI_2;
use xray_formats::camera::{
    matrix_build_camera_dir, matrix_build_projection, Dx12CameraConstants,
};
use xray_formats::ogf::XRayMatrix;
use xray_render_dx12::CameraConstants;

pub const PITCH_LIMIT: f32 = FRAC_PI_2 - 0.01; // ~89.4 degrees
pub const DEFAULT_FOV_DEG: f32 = 67.5;
pub const DEFAULT_NEAR_Z: f32 = 0.1;
pub const DEFAULT_FAR_Z: f32 = 1000.0;

#[derive(Debug, Clone)]
pub struct FirstPersonCamera {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_deg: f32,
    pub aspect: f32, // X-Ray convention: height / width
    pub near_z: f32,
    pub far_z: f32,
}

impl Default for FirstPersonCamera {
    fn default() -> Self {
        Self::new(1280, 720)
    }
}

impl FirstPersonCamera {
    pub fn new(width: u32, height: u32) -> Self {
        let w = width.max(1) as f32;
        let h = height.max(1) as f32;
        Self {
            position: [0.0, 1.8, 0.0], // Standard stalker eye level: 1.8m
            yaw: 0.0,
            pitch: 0.0,
            fov_deg: DEFAULT_FOV_DEG,
            aspect: h / w,
            near_z: DEFAULT_NEAR_Z,
            far_z: DEFAULT_FAR_Z,
        }
    }

    /// Update aspect ratio upon window resizing
    pub fn resize(&mut self, width: u32, height: u32) {
        let w = width.max(1) as f32;
        let h = height.max(1) as f32;
        self.aspect = h / w;
    }

    /// Rotate camera using mouse delta
    pub fn rotate(&mut self, delta_x: f32, delta_y: f32, sensitivity: f32) {
        self.yaw += delta_x * sensitivity;
        self.pitch -= delta_y * sensitivity; // Invert Y for standard FPS look

        // Normalize yaw to [0, 2pi)
        const TWO_PI: f32 = std::f32::consts::PI * 2.0;
        self.yaw = self.yaw.rem_euclid(TWO_PI);

        // Clamp pitch to avoid gimbal lock flip
        self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Forward unit vector
    #[inline]
    pub fn forward(&self) -> [f32; 3] {
        let cos_pitch = self.pitch.cos();
        [
            cos_pitch * self.yaw.sin(),
            self.pitch.sin(),
            cos_pitch * self.yaw.cos(),
        ]
    }

    /// Right unit vector (horizontal plane)
    #[inline]
    pub fn right(&self) -> [f32; 3] {
        [self.yaw.cos(), 0.0, -self.yaw.sin()]
    }

    /// Up unit vector
    #[inline]
    pub fn up(&self) -> [f32; 3] {
        [0.0, 1.0, 0.0]
    }

    /// Move camera relative to view direction (intent: [x: right, z: forward])
    pub fn move_relative(&mut self, intent: [f32; 2], speed: f32, dt: f32) {
        if intent[0] == 0.0 && intent[1] == 0.0 {
            return;
        }

        // Horizontal forward vector (ignoring pitch)
        let forward_horiz = [self.yaw.sin(), 0.0, self.yaw.cos()];
        let right_horiz = self.right();

        let move_x = right_horiz[0] * intent[0] + forward_horiz[0] * intent[1];
        let move_z = right_horiz[2] * intent[0] + forward_horiz[2] * intent[1];

        let dist = speed * dt;
        self.position[0] += move_x * dist;
        self.position[2] += move_z * dist;
    }

    /// Compute view matrix
    pub fn view_matrix(&self) -> XRayMatrix {
        matrix_build_camera_dir(self.position, self.forward(), self.up())
    }

    /// Compute perspective projection matrix
    pub fn projection_matrix(&self) -> XRayMatrix {
        matrix_build_projection(
            self.fov_deg.to_radians(),
            self.aspect,
            self.near_z,
            self.far_z,
        )
    }

    /// Build format-level constants block
    pub fn to_dx12_constants(&self) -> Dx12CameraConstants {
        let view = self.view_matrix();
        let proj = self.projection_matrix();
        Dx12CameraConstants::from_xray_matrices(&view, &proj, self.position)
    }

    /// Build renderer-level constant buffer for upload to GPU `b1`
    pub fn to_render_constants(&self) -> CameraConstants {
        let c = self.to_dx12_constants();
        CameraConstants {
            view: c.view,
            proj: c.proj,
            view_proj: c.view_proj,
            cam_pos: c.cam_pos,
            padding: c.padding,
        }
    }
}
