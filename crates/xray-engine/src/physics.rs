//! Stalker Character Controller & Capsule Physics (`xray-engine`)
//!
//! Pure-Rust kinematic capsule character controller colliding against static
//! `level.cform` [`CformBvh`] world geometry:
//! - Standard Stalker capsule (`height = 1.8m`, `radius = 0.4m`, `eye_height = 1.65m`)
//! - Gravity (`-9.81 m/s^2`), velocity accumulation, jump impulse, and ground clamping
//! - Stair/obstacle step-up (up to `0.4m`) and smooth horizontal wall sliding

use xray_formats::cform::CformBvh;

/// Physical dimensions and movement tuning for the Stalker capsule controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapsuleConfig {
    /// Total standing capsule height in meters (default `1.8`).
    pub height: f32,
    /// Crouching capsule height in meters (default `1.25`).
    pub crouch_height: f32,
    /// Capsule horizontal radius in meters (default `0.4`).
    pub radius: f32,
    /// Camera eye offset above feet while standing in meters (default `1.65`).
    pub eye_height: f32,
    /// Camera eye offset above feet while crouching in meters (default `1.1`).
    pub crouch_eye_height: f32,
    /// Maximum stair/curb height the controller can automatically step onto in meters (default `0.4`).
    pub step_height: f32,
    /// Downward gravitational acceleration magnitude in m/s^2 (default `9.81`).
    pub gravity: f32,
    /// Upward vertical velocity impulse on jump in m/s (default `5.2`).
    pub jump_impulse: f32,
    /// Maximum walkable floor slope angle in degrees (default `50.0`).
    pub max_slope_deg: f32,
    /// Downward snap distance when grounded to stick cleanly to slopes/stairs (default `0.15`).
    pub ground_snap_dist: f32,
}

impl Default for CapsuleConfig {
    fn default() -> Self {
        Self {
            height: 1.8,
            crouch_height: 1.25,
            radius: 0.4,
            eye_height: 1.65,
            crouch_eye_height: 1.1,
            step_height: 0.4,
            gravity: 9.81,
            jump_impulse: 5.2,
            max_slope_deg: 50.0,
            ground_snap_dist: 0.15,
        }
    }
}

/// Kinematic capsule character controller for the local Stalker player.
#[derive(Debug, Clone, PartialEq)]
pub struct StalkerController {
    /// World-space bottom center of the capsule (feet contact point).
    pub feet_position: [f32; 3],
    /// Current world-space linear velocity in m/s.
    pub velocity: [f32; 3],
    /// Whether the controller is currently resting on walkable ground.
    pub grounded: bool,
    /// Whether the controller is currently crouched.
    pub crouching: bool,
    /// Last detected walkable surface normal.
    pub ground_normal: [f32; 3],
    /// Material index (`CformTriangle::material`) of the walkable triangle currently supporting feet.
    pub ground_material: Option<u16>,
    /// Capsule geometry & movement parameters.
    pub config: CapsuleConfig,
    /// When `false` (default), gravity is paused if no `CformBvh` is loaded (e.g., headless mode
    /// without level collision), preventing unbounded freefall when testing without a level mesh.
    pub apply_gravity_without_cform: bool,
}

impl Default for StalkerController {
    fn default() -> Self {
        // Align initial eye position cleanly with FirstPersonCamera's default `[0.0, 1.8, 0.0]`.
        let config = CapsuleConfig::default();
        Self {
            feet_position: [0.0, 1.8 - config.eye_height, 0.0],
            velocity: [0.0, 0.0, 0.0],
            grounded: false,
            crouching: false,
            ground_normal: [0.0, 1.0, 0.0],
            ground_material: None,
            config,
            apply_gravity_without_cform: false,
        }
    }
}

impl StalkerController {
    /// Construct a controller standing at `feet_position`.
    pub fn new(feet_position: [f32; 3]) -> Self {
        Self {
            feet_position,
            ..Self::default()
        }
    }

    /// Construct a controller whose standing eye position matches `eye_pos`.
    pub fn from_eye_position(eye_pos: [f32; 3]) -> Self {
        let mut ctrl = Self::default();
        ctrl.sync_from_eye(eye_pos);
        ctrl
    }

    /// Active capsule height accounting for crouch state.
    #[inline]
    pub fn active_height(&self) -> f32 {
        if self.crouching {
            self.config.crouch_height
        } else {
            self.config.height
        }
    }

    /// Active eye offset above feet accounting for crouch state.
    #[inline]
    pub fn active_eye_height(&self) -> f32 {
        if self.crouching {
            self.config.crouch_eye_height
        } else {
            self.config.eye_height
        }
    }

    /// World-space eye position derived from `feet_position + [0, active_eye_height(), 0]`.
    #[inline]
    pub fn eye_position(&self) -> [f32; 3] {
        [
            self.feet_position[0],
            self.feet_position[1] + self.active_eye_height(),
            self.feet_position[2],
        ]
    }

    /// Reposition feet so that `eye_position()` matches `eye_pos` (e.g., on spawn or `.scop` load).
    #[inline]
    pub fn sync_from_eye(&mut self, eye_pos: [f32; 3]) {
        self.feet_position = [
            eye_pos[0],
            eye_pos[1] - self.active_eye_height(),
            eye_pos[2],
        ];
    }

    /// Snap feet onto the nearest walkable ground triangle beneath `[x, y + probe_up, z]`.
    /// Returns `true` if ground was found and feet were clamped.
    pub fn snap_to_ground(&mut self, bvh: &CformBvh<'_>, probe_up: f32, max_drop: f32) -> bool {
        let min_floor_ny = self.config.max_slope_deg.to_radians().cos();
        if let Some((ground_y, normal, material)) = sample_ground_height(
            bvh,
            self.feet_position,
            self.config.radius,
            probe_up.max(0.05),
            max_drop.max(0.05),
            min_floor_ny,
        ) {
            self.feet_position[1] = ground_y;
            self.velocity[1] = 0.0;
            self.grounded = true;
            self.ground_normal = normal;
            self.ground_material = Some(material);
            return true;
        }
        self.ground_material = None;
        false
    }

    /// Advance character controller physics by `dt` seconds against optional `CformBvh`.
    ///
    /// - `world_wish_xz`: normalized desired horizontal movement direction `[x, z]` in world space.
    /// - `move_speed`: target horizontal speed in m/s.
    /// - `jump_requested`: whether jump should trigger if currently grounded.
    /// - `crouch_requested`: whether crouch stance is active.
    pub fn step(
        &mut self,
        world_wish_xz: [f32; 2],
        move_speed: f32,
        jump_requested: bool,
        crouch_requested: bool,
        dt: f32,
        bvh: Option<&CformBvh<'_>>,
    ) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }

        // Handle crouch stand-up clearance check before expanding capsule height.
        if crouch_requested {
            self.crouching = true;
        } else if self.crouching {
            let can_stand = match bvh {
                Some(b) => !has_ceiling_obstruction(
                    b,
                    self.feet_position,
                    self.config.crouch_height,
                    self.config.height,
                    self.config.radius,
                ),
                None => true,
            };
            if can_stand {
                self.crouching = false;
            }
        }

        // Normalize wish direction safely
        let wish_len_sq =
            world_wish_xz[0] * world_wish_xz[0] + world_wish_xz[1] * world_wish_xz[1];
        let wish_xz = if wish_len_sq > 1e-6 {
            let inv = 1.0 / wish_len_sq.sqrt().max(1.0);
            [world_wish_xz[0] * inv, world_wish_xz[1] * inv]
        } else {
            [0.0, 0.0]
        };

        // Direct responsive Stalker horizontal velocity accumulation with light air preservation
        if self.grounded || bvh.is_none() {
            self.velocity[0] = wish_xz[0] * move_speed;
            self.velocity[2] = wish_xz[1] * move_speed;
        } else {
            // In-air steering retains momentum while allowing mid-air control
            let target_vx = wish_xz[0] * move_speed;
            let target_vz = wish_xz[1] * move_speed;
            let air_blend = (8.0 * dt).clamp(0.0, 1.0);
            self.velocity[0] += (target_vx - self.velocity[0]) * air_blend;
            self.velocity[2] += (target_vz - self.velocity[2]) * air_blend;
        }

        // Jump impulse
        let mut jumped_this_step = false;
        if jump_requested && self.grounded {
            self.velocity[1] = self.config.jump_impulse;
            self.grounded = false;
            jumped_this_step = true;
        }

        // Gravity accumulation
        let gravity_active = bvh.is_some() || self.apply_gravity_without_cform;
        if gravity_active && !self.grounded {
            self.velocity[1] -= self.config.gravity * dt;
        }

        let desired_dx = self.velocity[0] * dt;
        let desired_dz = self.velocity[2] * dt;
        let desired_dy = self.velocity[1] * dt;

        let Some(bvh) = bvh else {
            // Free-space step without level collision geometry
            self.feet_position[0] += desired_dx;
            self.feet_position[2] += desired_dz;
            if gravity_active {
                self.feet_position[1] += desired_dy;
            }
            return;
        };

        let height = self.active_height();
        let radius = self.config.radius;
        let min_floor_ny = self.config.max_slope_deg.to_radians().cos();

        // 1. Horizontal displacement with wall slide and optional stair step-up
        let was_grounded = self.grounded;
        let (mut new_pos, hit_wall, wall_normal_xz) = resolve_capsule_horizontal(
            bvh,
            self.feet_position,
            [desired_dx, desired_dz],
            height,
            radius,
            min_floor_ny,
        );

        let flat_dist_sq = (new_pos[0] - self.feet_position[0]).powi(2)
            + (new_pos[2] - self.feet_position[2]).powi(2);
        let desired_dist_sq = desired_dx * desired_dx + desired_dz * desired_dz;

        let mut stepped_up = false;
        if hit_wall
            && (was_grounded || !jumped_this_step && self.velocity[1] <= 0.1)
            && self.config.step_height > 1e-4
            && desired_dist_sq > 1e-10
        {
            // Verify headroom before lifting the capsule by `step_height`
            let ceiling_clear = !has_ceiling_obstruction(
                bvh,
                self.feet_position,
                height,
                height + self.config.step_height,
                radius * 0.9,
            );
            if ceiling_clear {
                let raised_start = [
                    self.feet_position[0],
                    self.feet_position[1] + self.config.step_height,
                    self.feet_position[2],
                ];
                let (raised_pos, _, _) = resolve_capsule_horizontal(
                    bvh,
                    raised_start,
                    [desired_dx, desired_dz],
                    height,
                    radius,
                    min_floor_ny,
                );
                let raised_dist_sq = (raised_pos[0] - self.feet_position[0]).powi(2)
                    + (raised_pos[2] - self.feet_position[2]).powi(2);

                if raised_dist_sq > flat_dist_sq + 1e-6 {
                    // Check for walkable floor between current feet level and raised position
                    if let Some((step_ground_y, step_normal, step_material)) = sample_ground_height(
                        bvh,
                        raised_pos,
                        radius,
                        0.05,
                        self.config.step_height + 0.05,
                        min_floor_ny,
                    ) {
                        let rise = step_ground_y - self.feet_position[1];
                        if rise > 1e-4 && rise <= self.config.step_height + 0.01 {
                            new_pos = [raised_pos[0], step_ground_y, raised_pos[2]];
                            self.grounded = true;
                            self.ground_normal = step_normal;
                            self.ground_material = Some(step_material);
                            self.velocity[1] = 0.0;
                            stepped_up = true;
                        }
                    }
                }
            }
        }

        if hit_wall && !stepped_up {
            // Project horizontal velocity onto wall tangent so sliding stays smooth
            let vn = self.velocity[0] * wall_normal_xz[0] + self.velocity[2] * wall_normal_xz[1];
            if vn < 0.0 {
                self.velocity[0] -= wall_normal_xz[0] * vn;
                self.velocity[2] -= wall_normal_xz[1] * vn;
            }
        }

        self.feet_position = new_pos;

        // 2. Vertical integration, ceiling check, and ground clamping
        if stepped_up {
            return;
        }

        if self.velocity[1] > 0.0 {
            // Rising (e.g., jumping): integrate upwards and clamp against overhead ceilings
            let next_y = self.feet_position[1] + desired_dy;
            if let Some(ceiling_feet_y) = check_ceiling_hit(
                bvh,
                [self.feet_position[0], self.feet_position[1], self.feet_position[2]],
                next_y,
                height,
                radius,
            ) {
                self.feet_position[1] = ceiling_feet_y;
                self.velocity[1] = 0.0;
            } else {
                self.feet_position[1] = next_y;
            }
            self.grounded = false;
            self.ground_material = None;
            return;
        }

        // Falling or walking on ground: integrate Y first when airborne, or probe down to clamp
        let probe_up = if was_grounded && !jumped_this_step {
            self.config.step_height.min(0.35)
        } else {
            (height * 0.4).min(0.45)
        };
        let target_y = self.feet_position[1] + desired_dy;
        let drop_allowance = if was_grounded && !jumped_this_step {
            (-desired_dy).max(0.0) + self.config.ground_snap_dist
        } else {
            (-desired_dy).max(0.0) + 0.02
        };

        if let Some((ground_y, normal, material)) = sample_ground_height(
            bvh,
            self.feet_position,
            radius,
            probe_up,
            drop_allowance,
            min_floor_ny,
        ) && ground_y >= target_y - 0.01
        {
            self.feet_position[1] = ground_y;
            self.velocity[1] = 0.0;
            self.grounded = true;
            self.ground_normal = normal;
            self.ground_material = Some(material);
            return;
        }

        // Prevent unbounded freefall through the bottom of the world
        let min_world_y = bvh.aabb().min[1] - 5.0;
        if target_y < min_world_y {
            self.feet_position[1] = min_world_y + 6.0;
            self.velocity[1] = 0.0;
            self.grounded = true;
            return;
        }

        // Remain airborne and apply vertical drop
        self.feet_position[1] = target_y;
        self.grounded = false;
        self.ground_material = None;
    }
}

/// Resolve capsule horizontal translation against steep triangles via iterative closest-point pushout.
/// Returns `(resolved_feet_pos, hit_wall, accumulated_wall_normal_xz)`.
fn resolve_capsule_horizontal(
    bvh: &CformBvh<'_>,
    start_feet: [f32; 3],
    disp_xz: [f32; 2],
    height: f32,
    radius: f32,
    min_floor_ny: f32,
) -> ([f32; 3], bool, [f32; 2]) {
    let mut pos = [start_feet[0] + disp_xz[0], start_feet[1], start_feet[2] + disp_xz[1]];
    let mut hit_wall = false;
    let mut accum_nx = 0.0f32;
    let mut accum_nz = 0.0f32;

    let half_h = height * 0.5;
    let half_extents = [radius + 0.08, half_h + 0.05, radius + 0.08];
    let y_low = radius.min(half_h);
    let y_high = (height - radius).max(y_low);

    // 4 Gauss-Seidel depenetration passes give clean corner sliding along multiple walls
    for _ in 0..4 {
        let query_center = [pos[0], pos[1] + half_h, pos[2]];
        let hits = bvh.box_query(query_center, half_extents, true, false);
        let mut pushed_this_pass = false;

        for hit in &hits {
            let [p0, p1, p2] = hit.verts;
            let tri_n = triangle_unit_normal(p0, p1, p2);
            let is_steep_wall = tri_n[1].abs() < min_floor_ny;

            let (s, q, dist_sq) =
                closest_capsule_segment_to_triangle(pos, pos[1] + y_low, pos[1] + y_high, p0, p1, p2);

            if dist_sq >= radius * radius {
                continue;
            }

            // Skip walkable ground contact beneath/at feet so floors don't snag horizontal motion
            if !is_steep_wall && q[1] <= pos[1] + radius * 0.65 {
                continue;
            }

            let dx = s[0] - q[0];
            let dz = s[2] - q[2];
            let horiz_len_sq = dx * dx + dz * dz;
            if horiz_len_sq <= 1e-10 {
                continue;
            }

            let horiz_len = horiz_len_sq.sqrt();
            let dist = dist_sq.sqrt();
            let penetration = (radius - dist).max(0.0);
            if penetration <= 1e-5 {
                continue;
            }

            let nx = dx / horiz_len;
            let nz = dz / horiz_len;
            pos[0] += nx * (penetration + 1e-4);
            pos[2] += nz * (penetration + 1e-4);
            accum_nx += nx;
            accum_nz += nz;
            hit_wall = true;
            pushed_this_pass = true;
        }

        if !pushed_this_pass {
            break;
        }
    }

    let n_len_sq = accum_nx * accum_nx + accum_nz * accum_nz;
    let wall_normal_xz = if n_len_sq > 1e-10 {
        let inv = 1.0 / n_len_sq.sqrt();
        [accum_nx * inv, accum_nz * inv]
    } else {
        [0.0, 0.0]
    };

    (pos, hit_wall, wall_normal_xz)
}

/// Sample walkable ground beneath `feet_pos` using a 9-point disk inside the capsule footprint
/// so small triangle seams, stair edges, and steep ledges are caught reliably.
fn sample_ground_height(
    bvh: &CformBvh<'_>,
    feet_pos: [f32; 3],
    radius: f32,
    probe_up: f32,
    drop_allowance: f32,
    min_floor_ny: f32,
) -> Option<(f32, [f32; 3], u16)> {
    let r = radius * 0.98;
    let d = radius * 0.70;
    let offsets = [
        [0.0, 0.0],
        [r, 0.0],
        [-r, 0.0],
        [0.0, r],
        [0.0, -r],
        [d, d],
        [d, -d],
        [-d, d],
        [-d, -d],
    ];
    let max_range = probe_up + drop_allowance;
    let mut best_y = f32::NEG_INFINITY;
    let mut best_normal = [0.0f32, 1.0, 0.0];
    let mut best_material = 0u16;

    for [ox, oz] in offsets {
        let origin = [feet_pos[0] + ox, feet_pos[1] + probe_up, feet_pos[2] + oz];
        // Cast without backface culling and accept walkable surface normals oriented toward +Y
        if let Some(hit) = bvh.raycast_nearest(origin, [0.0, -1.0, 0.0], max_range, false) {
            let mut n = triangle_unit_normal(hit.verts[0], hit.verts[1], hit.verts[2]);
            if n[1] < 0.0 {
                n = [-n[0], -n[1], -n[2]];
            }
            if n[1] >= min_floor_ny && hit.point[1] > best_y {
                best_y = hit.point[1];
                best_normal = n;
                best_material = hit.material;
            }
        }
    }

    if best_y.is_finite() {
        Some((best_y, best_normal, best_material))
    } else {
        None
    }
}

/// Returns `true` if raising the capsule head from `from_height` to `to_height` intersects geometry.
fn has_ceiling_obstruction(
    bvh: &CformBvh<'_>,
    feet_pos: [f32; 3],
    from_height: f32,
    to_height: f32,
    radius: f32,
) -> bool {
    if to_height <= from_height {
        return false;
    }
    let y_bot = feet_pos[1] + from_height.min(to_height);
    let y_top = feet_pos[1] + to_height;
    let center_y = 0.5 * (y_bot + y_top);
    let half_h = 0.5 * (y_top - y_bot) + 0.02;
    let hits = bvh.box_query(
        [feet_pos[0], center_y, feet_pos[2]],
        [radius * 0.9, half_h, radius * 0.9],
        true,
        false,
    );
    let head_center = [feet_pos[0], y_top - radius, feet_pos[2]];
    for hit in hits {
        let q = closest_point_on_triangle(head_center, hit.verts[0], hit.verts[1], hit.verts[2]);
        let d2 = dist_sq(head_center, q);
        if d2 < radius * radius * 0.95 {
            return true;
        }
    }
    false
}

/// Checks if rising from `current_feet_y` toward `next_feet_y` hits a ceiling; returns clamped `feet_y`.
fn check_ceiling_hit(
    bvh: &CformBvh<'_>,
    feet_pos: [f32; 3],
    next_feet_y: f32,
    height: f32,
    radius: f32,
) -> Option<f32> {
    let rise = next_feet_y - feet_pos[1];
    if rise <= 0.0 {
        return None;
    }
    let head_start = feet_pos[1] + height - radius;
    let ray_origin = [feet_pos[0], head_start, feet_pos[2]];
    let hit = bvh.raycast_nearest(ray_origin, [0.0, 1.0, 0.0], rise + radius, false)?;
    let clamped_feet_y = (hit.point[1] - height - 0.005).max(feet_pos[1]);
    Some(clamped_feet_y)
}

/// Closest pair `(axis_point, tri_point, dist_sq)` between vertical segment `[(x, y_low, z), (x, y_high, z)]`
/// and triangle `(a, b, c)`.
fn closest_capsule_segment_to_triangle(
    xz_ref: [f32; 3],
    y_low: f32,
    y_high: f32,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) -> ([f32; 3], [f32; 3], f32) {
    let mid_y = 0.5 * (y_low + y_high);
    let seeds = [y_low, mid_y, y_high];
    let mut best_s = [xz_ref[0], mid_y, xz_ref[2]];
    let mut best_q = a;
    let mut best_d2 = f32::INFINITY;

    for &start_y in &seeds {
        let mut ay = start_y;
        for _ in 0..3 {
            let q = closest_point_on_triangle([xz_ref[0], ay, xz_ref[2]], a, b, c);
            ay = q[1].clamp(y_low, y_high);
        }
        let s = [xz_ref[0], ay, xz_ref[2]];
        let q = closest_point_on_triangle(s, a, b, c);
        let d2 = dist_sq(s, q);
        if d2 < best_d2 {
            best_d2 = d2;
            best_s = s;
            best_q = q;
        }
    }

    (best_s, best_q, best_d2)
}

#[inline]
fn dist_sq(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[inline]
fn triangle_unit_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if len_sq > 1e-12 {
        let inv = 1.0 / len_sq.sqrt();
        [n[0] * inv, n[1] * inv, n[2] * inv]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// Exact closest point on triangle `(a, b, c)` to point `p` (Christer Ericson's Voronoi region method).
pub fn closest_point_on_triangle(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];

    let d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
    let d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
    let d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
    let d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let denom = d1 - d3;
        let v = if denom.abs() > 1e-12 { d1 / denom } else { 0.0 };
        return [a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]];
    }

    let cp = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
    let d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
    let d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let denom = d2 - d6;
        let w = if denom.abs() > 1e-12 { d2 / denom } else { 0.0 };
        return [a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]];
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let denom = (d4 - d3) + (d5 - d6);
        let w = if denom.abs() > 1e-12 {
            (d4 - d3) / denom
        } else {
            0.0
        };
        return [
            b[0] + w * (c[0] - b[0]),
            b[1] + w * (c[1] - b[1]),
            b[2] + w * (c[2] - b[2]),
        ];
    }

    let denom = va + vb + vc;
    if denom.abs() <= 1e-12 {
        return a;
    }
    let inv = 1.0 / denom;
    let v = vb * inv;
    let w = vc * inv;
    [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use xray_formats::cform::{CformModel, CformTriangle};

    fn push_quad(
        verts: &mut Vec<[f32; 3]>,
        tris: &mut Vec<CformTriangle>,
        p0: [f32; 3],
        p1: [f32; 3],
        p2: [f32; 3],
        p3: [f32; 3],
    ) {
        let base = verts.len() as u32;
        verts.extend_from_slice(&[p0, p1, p2, p3]);
        tris.push(CformTriangle {
            verts: [base, base + 1, base + 2],
            material: 0,
            suppress_shadows: false,
            suppress_wm: false,
            sector: 0,
        });
        tris.push(CformTriangle {
            verts: [base, base + 2, base + 3],
            material: 0,
            suppress_shadows: false,
            suppress_wm: false,
            sector: 0,
        });
    }

    fn push_box(verts: &mut Vec<[f32; 3]>, tris: &mut Vec<CformTriangle>, min: [f32; 3], max: [f32; 3]) {
        // Top
        push_quad(
            verts,
            tris,
            [min[0], max[1], min[2]],
            [max[0], max[1], min[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        );
        // Front (-Z)
        push_quad(
            verts,
            tris,
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
        );
        // Back (+Z)
        push_quad(
            verts,
            tris,
            [max[0], min[1], max[2]],
            [min[0], min[1], max[2]],
            [min[0], max[1], max[2]],
            [max[0], max[1], max[2]],
        );
        // Left (-X)
        push_quad(
            verts,
            tris,
            [min[0], min[1], max[2]],
            [min[0], min[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], max[1], max[2]],
        );
        // Right (+X)
        push_quad(
            verts,
            tris,
            [max[0], min[1], min[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [max[0], max[1], min[2]],
        );
    }

    fn make_cform(verts: Vec<[f32; 3]>, tris: Vec<CformTriangle>) -> CformModel {
        CformModel::from_parts(verts, tris).expect("build test CformModel")
    }

    #[test]
    fn test_capsule_gravity_and_ground_clamping() {
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        push_quad(
            &mut verts,
            &mut tris,
            [-20.0, 0.0, -20.0],
            [20.0, 0.0, -20.0],
            [20.0, 0.0, 20.0],
            [-20.0, 0.0, 20.0],
        );
        let cform = make_cform(verts, tris);
        let bvh = CformBvh::build(&cform);

        let mut ctrl = StalkerController::new([0.0, 3.0, 0.0]);
        assert!(!ctrl.grounded);

        // Simulate 2 seconds at 60Hz
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            ctrl.step([0.0, 0.0], 0.0, false, false, dt, Some(&bvh));
        }

        assert!(ctrl.grounded, "player should land on ground y=0");
        assert!(ctrl.feet_position[1].abs() < 1e-3);
        assert!((ctrl.eye_position()[1] - 1.65).abs() < 1e-3);
        assert_eq!(ctrl.velocity[1], 0.0);
    }

    #[test]
    fn test_capsule_jump_impulse_and_landing() {
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        push_quad(
            &mut verts,
            &mut tris,
            [-10.0, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [10.0, 0.0, 10.0],
            [-10.0, 0.0, 10.0],
        );
        let cform = make_cform(verts, tris);
        let bvh = CformBvh::build(&cform);

        let mut ctrl = StalkerController::new([0.0, 0.0, 0.0]);
        assert!(ctrl.snap_to_ground(&bvh, 0.5, 1.0));
        assert!(ctrl.grounded);

        let dt = 1.0 / 60.0;
        ctrl.step([0.0, 0.0], 0.0, true, false, dt, Some(&bvh));
        assert!(!ctrl.grounded);
        assert!(ctrl.velocity[1] > 4.5);

        let mut max_y = 0.0f32;
        for _ in 0..120 {
            ctrl.step([0.0, 0.0], 0.0, false, false, dt, Some(&bvh));
            max_y = max_y.max(ctrl.feet_position[1]);
        }
        assert!(max_y > 1.0, "jump apex should exceed 1.0m, got {max_y}");
        assert!(ctrl.grounded, "player should land back on floor");
        assert!(ctrl.feet_position[1].abs() < 1e-3);
    }

    #[test]
    fn test_capsule_wall_collision_and_horizontal_slide() {
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        // Floor y=0
        push_quad(
            &mut verts,
            &mut tris,
            [-20.0, 0.0, -20.0],
            [20.0, 0.0, -20.0],
            [20.0, 0.0, 20.0],
            [-20.0, 0.0, 20.0],
        );
        // Solid wall along z = 2.0..3.0 blocking +Z motion
        push_box(&mut verts, &mut tris, [-10.0, 0.0, 2.0], [10.0, 3.0, 3.0]);

        let cform = make_cform(verts, tris);
        let bvh = CformBvh::build(&cform);

        let mut ctrl = StalkerController::new([0.0, 0.0, 0.0]);
        assert!(ctrl.snap_to_ground(&bvh, 0.5, 1.0));

        // Move diagonally (+X, +Z) toward the wall
        let dt = 1.0 / 60.0;
        for _ in 0..90 {
            ctrl.step([1.0, 1.0], 5.0, false, false, dt, Some(&bvh));
        }

        // Wall front face is at z = 2.0, so capsule center must remain <= 2.0 - radius (1.6)
        assert!(
            ctrl.feet_position[2] <= 2.0 - ctrl.config.radius + 0.02,
            "capsule penetrated wall z=2.0: z={}",
            ctrl.feet_position[2]
        );
        // Meanwhile, diagonal input should smoothly slide +X along the wall
        assert!(
            ctrl.feet_position[0] > 3.5,
            "expected horizontal wall slide along +X, got x={}",
            ctrl.feet_position[0]
        );
    }

    #[test]
    fn test_capsule_stair_step_up_and_tall_blocker() {
        let mut verts = Vec::new();
        let mut tris = Vec::new();
        // Floor y=0
        push_quad(
            &mut verts,
            &mut tris,
            [-20.0, 0.0, -20.0],
            [20.0, 0.0, -20.0],
            [20.0, 0.0, 20.0],
            [-20.0, 0.0, 20.0],
        );
        // Climbable 0.3m stair block starting at z = 1.5
        push_box(&mut verts, &mut tris, [-2.0, 0.0, 1.5], [2.0, 0.3, 4.0]);
        // Unclimbable 0.7m tall step starting at z = 3.0 on top of floor (0.4m above stair = 0.7m total)
        push_box(&mut verts, &mut tris, [-2.0, 0.0, 3.0], [2.0, 0.7, 5.0]);

        let cform = make_cform(verts, tris);
        let bvh = CformBvh::build(&cform);

        let mut ctrl = StalkerController::new([0.0, 0.0, 0.5]);
        assert!(ctrl.snap_to_ground(&bvh, 0.5, 1.0));

        // Walk +Z onto the 0.3m step (should step up cleanly!)
        let dt = 1.0 / 60.0;
        for _ in 0..30 {
            ctrl.step([0.0, 1.0], 4.0, false, false, dt, Some(&bvh));
        }
        assert!(
            ctrl.feet_position[2] > 1.7 && ctrl.feet_position[2] < 3.0,
            "expected player to climb onto 0.3m stair block, z={}",
            ctrl.feet_position[2]
        );
        assert!(
            (ctrl.feet_position[1] - 0.3).abs() < 0.02,
            "expected feet on y=0.3 stair, got y={}",
            ctrl.feet_position[1]
        );

        // Now test a 0.65m tall step in front of the player (exceeds 0.4m step_height):
        let mut verts2 = Vec::new();
        let mut tris2 = Vec::new();
        push_quad(
            &mut verts2,
            &mut tris2,
            [-10.0, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [10.0, 0.0, 10.0],
            [-10.0, 0.0, 10.0],
        );
        push_box(&mut verts2, &mut tris2, [-2.0, 0.0, 1.5], [2.0, 0.65, 4.0]);
        let cform2 = make_cform(verts2, tris2);
        let bvh2 = CformBvh::build(&cform2);

        let mut blocked = StalkerController::new([0.0, 0.0, 0.5]);
        assert!(blocked.snap_to_ground(&bvh2, 0.5, 1.0));
        for _ in 0..45 {
            blocked.step([0.0, 1.0], 4.0, false, false, dt, Some(&bvh2));
        }
        assert!(
            blocked.feet_position[2] <= 1.5 - blocked.config.radius + 0.02,
            "0.65m obstacle (>0.4m step_height) should block horizontal movement, got z={}",
            blocked.feet_position[2]
        );
    }
}
