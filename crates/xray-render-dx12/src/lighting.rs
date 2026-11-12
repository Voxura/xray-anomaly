//! Phase 9: Pure Rust Dynamic Lighting & Point/Spot Light Evaluation
//!
//! Faithful implementation of X-Ray Monolith Engine dynamic lighting architecture:
//! - Point lights with smooth quadratic/spherical distance attenuation
//! - Spot lights with inner/outer angular cone falloff
//! - Directional sunlight integration
//! - Animated light color modulation via `xray_formats::lanims::LightAnimItem`
//! - GPU structured buffer `GpuLight` (64-byte 16-byte aligned layout)
//! - Distance culling & sphere-AABB intersection tests for visible light selection

use xray_formats::lanims::LightAnimItem;

/// Dynamic light type
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LightType {
    #[default]
    Point = 0,
    Spot = 1,
    Directional = 2,
}

/// GPU layout for a single light source (64 bytes, 16-byte aligned)
///
/// Matches HLSL:
/// ```hlsl
/// struct GpuLightData
/// {
///     float3 position;
///     float range;
///     float4 color;
///     float3 direction;
///     uint light_type;
///     float2 spot_params; // [cos_inner, cos_outer]
///     float2 attenuation; // [att0, att1]
/// };
/// ```
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuLight {
    pub position: [f32; 3],
    pub range: f32,
    pub color: [f32; 4],
    pub direction: [f32; 3],
    pub light_type: u32,
    pub spot_params: [f32; 2],
    pub attenuation: [f32; 2],
}

impl Default for GpuLight {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            range: 0.0,
            color: [0.0, 0.0, 0.0, 0.0],
            direction: [0.0, -1.0, 0.0],
            light_type: 0,
            spot_params: [1.0, 0.707], // [cos(0), cos(45 deg)]
            attenuation: [1.0, 0.0],
        }
    }
}

impl GpuLight {
    /// Return the raw byte slice for a list of GPU lights
    pub fn as_bytes(lights: &[GpuLight]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                lights.as_ptr().cast::<u8>(),
                std::mem::size_of_val(lights),
            )
        }
    }
}

/// Pure Rust Dynamic Light description
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicLight {
    pub light_type: LightType,
    pub position: [f32; 3],
    pub range: f32,
    pub direction: [f32; 3],
    pub spot_cone_angles: [f32; 2], // [inner_rad, outer_rad]
    pub color: [f32; 4],           // [r, g, b, intensity/multiplier]
    pub attenuation: [f32; 2],     // [constant_factor, linear_factor]
    pub anim_curve: Option<LightAnimItem>,
    pub enabled: bool,
}

impl DynamicLight {
    /// Create a new point light
    pub fn new_point(position: [f32; 3], range: f32, color: [f32; 4]) -> Self {
        Self {
            light_type: LightType::Point,
            position,
            range,
            direction: [0.0, -1.0, 0.0],
            spot_cone_angles: [0.0, 0.0],
            color,
            attenuation: [1.0, 0.0],
            anim_curve: None,
            enabled: true,
        }
    }

    /// Create a new spot light
    pub fn new_spot(
        position: [f32; 3],
        direction: [f32; 3],
        range: f32,
        inner_angle_rad: f32,
        outer_angle_rad: f32,
        color: [f32; 4],
    ) -> Self {
        let len = (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]).sqrt();
        let norm_dir = if len > 1.0e-5 {
            [direction[0] / len, direction[1] / len, direction[2] / len]
        } else {
            [0.0, -1.0, 0.0]
        };

        Self {
            light_type: LightType::Spot,
            position,
            range,
            direction: norm_dir,
            spot_cone_angles: [inner_angle_rad, outer_angle_rad],
            color,
            attenuation: [1.0, 0.0],
            anim_curve: None,
            enabled: true,
        }
    }

    /// Attach an animated color curve from `lanims.xr`
    pub fn with_anim(mut self, anim: LightAnimItem) -> Self {
        self.anim_curve = Some(anim);
        self
    }

    /// Set attenuation factors
    pub fn with_attenuation(mut self, att0: f32, att1: f32) -> Self {
        self.attenuation = [att0, att1];
        self
    }

    /// Evaluate GPU parameters at a specific playback time
    pub fn evaluate_at(&self, time_seconds: f32) -> GpuLight {
        let (r, g, b, a) = if let Some(anim) = &self.anim_curve {
            let sample = anim.sample_rgba_f32(time_seconds, true);
            (
                self.color[0] * sample[0],
                self.color[1] * sample[1],
                self.color[2] * sample[2],
                self.color[3] * sample[3],
            )
        } else {
            (self.color[0], self.color[1], self.color[2], self.color[3])
        };

        let spot_params = match self.light_type {
            LightType::Spot => [
                (self.spot_cone_angles[0] * 0.5).cos(),
                (self.spot_cone_angles[1] * 0.5).cos(),
            ],
            _ => [1.0, 0.0],
        };

        GpuLight {
            position: self.position,
            range: self.range,
            color: [r, g, b, a],
            direction: self.direction,
            light_type: self.light_type as u32,
            spot_params,
            attenuation: self.attenuation,
        }
    }

    /// Test if this light intersects with a given axis-aligned bounding box (min_xyz, max_xyz)
    pub fn intersects_aabb(&self, aabb_min: [f32; 3], aabb_max: [f32; 3]) -> bool {
        if !self.enabled || self.range <= 0.0 {
            return false;
        }
        let mut d_squared = 0.0f32;
        for i in 0..3 {
            let p = self.position[i];
            if p < aabb_min[i] {
                let diff = aabb_min[i] - p;
                d_squared += diff * diff;
            } else if p > aabb_max[i] {
                let diff = p - aabb_max[i];
                d_squared += diff * diff;
            }
        }
        d_squared <= (self.range * self.range)
    }
}

/// Dynamic light manager
#[derive(Debug, Default, Clone)]
pub struct LightManager {
    lights: Vec<DynamicLight>,
}

impl LightManager {
    pub fn new() -> Self {
        Self { lights: Vec::new() }
    }

    pub fn add_light(&mut self, light: DynamicLight) -> usize {
        let id = self.lights.len();
        self.lights.push(light);
        id
    }

    pub fn get_light(&self, id: usize) -> Option<&DynamicLight> {
        self.lights.get(id)
    }

    pub fn get_light_mut(&mut self, id: usize) -> Option<&mut DynamicLight> {
        self.lights.get_mut(id)
    }

    pub fn remove_light(&mut self, id: usize) -> Option<DynamicLight> {
        if id < self.lights.len() {
            Some(self.lights.remove(id))
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.lights.clear();
    }

    pub fn len(&self) -> usize {
        self.lights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lights.is_empty()
    }

    /// Build all active GPU lights evaluated at `time_seconds`
    pub fn build_gpu_lights(&self, time_seconds: f32) -> Vec<GpuLight> {
        self.lights
            .iter()
            .filter(|l| l.enabled && l.range > 0.0)
            .map(|l| l.evaluate_at(time_seconds))
            .collect()
    }

    /// Select and sort visible lights influencing a specific bounding box
    pub fn select_visible_lights(
        &self,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        max_lights: usize,
        time_seconds: f32,
    ) -> Vec<GpuLight> {
        let center = [
            (aabb_min[0] + aabb_max[0]) * 0.5,
            (aabb_min[1] + aabb_max[1]) * 0.5,
            (aabb_min[2] + aabb_max[2]) * 0.5,
        ];

        let mut candidate_lights: Vec<(f32, GpuLight)> = self
            .lights
            .iter()
            .filter(|l| l.intersects_aabb(aabb_min, aabb_max))
            .map(|l| {
                let dx = l.position[0] - center[0];
                let dy = l.position[1] - center[1];
                let dz = l.position[2] - center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;
                (dist_sq, l.evaluate_at(time_seconds))
            })
            .collect();

        // Sort by distance to bounding box center (closest lights first)
        candidate_lights.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        candidate_lights
            .into_iter()
            .take(max_lights)
            .map(|(_, gpu_light)| gpu_light)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_light_evaluation_and_aabb_intersection() {
        let light = DynamicLight::new_point([10.0, 0.0, 5.0], 15.0, [1.0, 0.8, 0.6, 2.0]);
        let gpu = light.evaluate_at(0.0);
        assert_eq!(gpu.position, [10.0, 0.0, 5.0]);
        assert_eq!(gpu.range, 15.0);
        assert_eq!(gpu.color, [1.0, 0.8, 0.6, 2.0]);
        assert_eq!(gpu.light_type, 0);

        // AABB containing [0, 0, 0] to [5, 5, 5] -> distance squared to [10, 0, 5] is (10-5)^2 = 25 <= 15^2 (225)
        assert!(light.intersects_aabb([0.0, 0.0, 0.0], [5.0, 5.0, 5.0]));

        // Distant AABB [100, 100, 100] to [110, 110, 110]
        assert!(!light.intersects_aabb([100.0, 100.0, 100.0], [110.0, 110.0, 110.0]));
    }

    #[test]
    fn test_spot_light_cone_and_direction() {
        use std::f32::consts::PI;
        let light = DynamicLight::new_spot(
            [0.0, 10.0, 0.0],
            [0.0, -2.0, 0.0],
            20.0,
            PI / 6.0, // 30 deg inner
            PI / 3.0, // 60 deg outer
            [0.5, 0.5, 1.0, 1.0],
        );
        let gpu = light.evaluate_at(0.0);
        assert_eq!(gpu.direction, [0.0, -1.0, 0.0]);
        assert_eq!(gpu.light_type, 1);
        // spot_params = [cos(15 deg), cos(30 deg)]
        let expected_cos_inner = (PI / 12.0).cos();
        let expected_cos_outer = (PI / 6.0).cos();
        assert!((gpu.spot_params[0] - expected_cos_inner).abs() < 1.0e-5);
        assert!((gpu.spot_params[1] - expected_cos_outer).abs() < 1.0e-5);
    }

    #[test]
    fn test_animated_light_curve_modulation() {
        use std::collections::BTreeMap;
        let mut keys = BTreeMap::new();
        keys.insert(0, 0xFFFF_FFFF); // White
        keys.insert(1, 0xFFFF_0000); // Red

        let anim = LightAnimItem {
            name: "test_anim".to_string(),
            fps: 1.0,
            default_fps: 1.0,
            frame_count: 2,
            keys,
        };

        let light = DynamicLight::new_point([0.0, 0.0, 0.0], 10.0, [1.0, 1.0, 1.0, 1.0])
            .with_anim(anim);

        // At t = 0.0, white
        let gpu0 = light.evaluate_at(0.0);
        assert!((gpu0.color[0] - 1.0).abs() < 0.01);
        assert!((gpu0.color[1] - 1.0).abs() < 0.01);
        assert!((gpu0.color[2] - 1.0).abs() < 0.01);

        // At t = 1.0, red
        let gpu1 = light.evaluate_at(1.0);
        assert!((gpu1.color[0] - 1.0).abs() < 0.01);
        assert!(gpu1.color[1] < 0.01);
        assert!(gpu1.color[2] < 0.01);
    }

    #[test]
    fn test_light_manager_culling_and_priority() {
        let mut mgr = LightManager::new();
        // Light 1: Close to center [0, 0, 0], range 10
        mgr.add_light(DynamicLight::new_point([1.0, 0.0, 0.0], 10.0, [1.0, 0.0, 0.0, 1.0]));
        // Light 2: Farther from center [0, 0, 0], range 20
        mgr.add_light(DynamicLight::new_point([5.0, 0.0, 0.0], 20.0, [0.0, 1.0, 0.0, 1.0]));
        // Light 3: Out of range
        mgr.add_light(DynamicLight::new_point([50.0, 0.0, 0.0], 10.0, [0.0, 0.0, 1.0, 1.0]));

        let visible = mgr.select_visible_lights([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0], 2, 0.0);
        assert_eq!(visible.len(), 2);
        // Closest light should be Light 1 at x=1.0
        assert_eq!(visible[0].position, [1.0, 0.0, 0.0]);
        // Second should be Light 2 at x=5.0
        assert_eq!(visible[1].position, [5.0, 0.0, 0.0]);
    }
}
