//! High-precision engine timer and frame rate monitor
//!
//! Employs Windows `QueryPerformanceCounter` and `QueryPerformanceFrequency`
//! for sub-microsecond delta time evaluation, frame-rate smoothing, and
//! fixed-timestep physics/network accumulators.

use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

pub struct EngineTimer {
    frequency: i64,
    start_time: i64,
    prev_time: i64,
    current_time: i64,
    delta_time: f32,
    total_time: f64,
    frame_count: u64,
    fps_accum_time: f32,
    fps_accum_frames: u32,
    current_fps: f32,
    accumulator: f32,
}

impl EngineTimer {
    pub fn new() -> Self {
        let mut freq = 0i64;
        unsafe {
            let _ = QueryPerformanceFrequency(&mut freq);
        }
        let mut now = 0i64;
        unsafe {
            let _ = QueryPerformanceCounter(&mut now);
        }

        Self {
            frequency: if freq > 0 { freq } else { 1 },
            start_time: now,
            prev_time: now,
            current_time: now,
            delta_time: 1.0 / 60.0,
            total_time: 0.0,
            frame_count: 0,
            fps_accum_time: 0.0,
            fps_accum_frames: 0,
            current_fps: 60.0,
            accumulator: 0.0,
        }
    }

    /// Advance time by one frame and compute smoothed delta time
    pub fn tick(&mut self) -> f32 {
        let mut now = 0i64;
        unsafe {
            let _ = QueryPerformanceCounter(&mut now);
        }
        self.current_time = now;
        let elapsed_ticks = now - self.prev_time;
        self.prev_time = now;

        let raw_dt = elapsed_ticks as f64 / self.frequency as f64;
        // Clamp delta time to maximum of 100ms (prevents large spiral on debugger pause)
        let clamped_dt = (raw_dt as f32).clamp(0.0001, 0.1);
        self.delta_time = clamped_dt;
        self.total_time = (now - self.start_time) as f64 / self.frequency as f64;
        self.frame_count += 1;
        self.accumulator += clamped_dt;

        // FPS measurement updated every 0.5 seconds
        self.fps_accum_time += clamped_dt;
        self.fps_accum_frames += 1;
        if self.fps_accum_time >= 0.5 {
            self.current_fps = self.fps_accum_frames as f32 / self.fps_accum_time;
            self.fps_accum_time = 0.0;
            self.fps_accum_frames = 0;
        }

        clamped_dt
    }

    /// Shift `prev_time` backward so the next `tick()` observes `dt_sec` elapsed cleanly in deterministic tests.
    pub fn simulate_elapsed(&mut self, dt_sec: f32) {
        let ticks = (f64::from(dt_sec.max(0.0)) * self.frequency as f64) as i64;
        self.prev_time = self.prev_time.saturating_sub(ticks);
    }

    /// Run fixed-timestep updates (e.g. 60Hz physics/network step)
    pub fn step_fixed<F: FnMut(f32)>(&mut self, fixed_dt: f32, mut update_fn: F) {
        let max_steps = 4; // Prevent spiral of death
        let mut step = 0;
        while self.accumulator >= fixed_dt && step < max_steps {
            update_fn(fixed_dt);
            self.accumulator -= fixed_dt;
            step += 1;
        }
        if self.accumulator > fixed_dt {
            self.accumulator = 0.0;
        }
    }

    #[inline]
    pub fn delta_time(&self) -> f32 {
        self.delta_time
    }

    #[inline]
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    #[inline]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    #[inline]
    pub fn fps(&self) -> f32 {
        self.current_fps
    }
}

impl Default for EngineTimer {
    fn default() -> Self {
        Self::new()
    }
}
