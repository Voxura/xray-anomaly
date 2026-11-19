use xray_core::ltx_loader::OwnedLtxDocument;
use xray_formats::envelopes::ObjectMotion;
use xray_formats::ogf::{XRayMatrix, matrix_set_xyzi_translate};
use xray_formats::vfs_bridge::{load_anm_from_vfs, load_ltx_tree_from_vfs};
use xray_vfs::VirtualFileSystem;

// Enhanced Animations schedules restoration of the active slot after the closing motion.
const CLOSE_RECOVERY_SEC: f32 = 0.55;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackpackPhase {
    Opening,
    Open,
    Closing,
}

pub(crate) struct BackpackCue {
    pub section: String,
    pub duration: f32,
    pub speed: f32,
    pub sound: Option<String>,
    camera: Option<ObjectMotion>,
}

pub(crate) struct BackpackSequence {
    pub config: OwnedLtxDocument,
    pub opening: BackpackCue,
    pub closing: BackpackCue,
    pub phase: BackpackPhase,
    pub elapsed: f32,
}

impl BackpackSequence {
    pub fn load(vfs: &VirtualFileSystem, community_index: i32) -> Option<Self> {
        let effects =
            load_ltx_tree_from_vfs(vfs, "configs/items/items/ea_addon_backpack.ltx", &[]).ok()?;
        let config =
            load_ltx_tree_from_vfs(vfs, "configs/items/items/items_anm_backpack.ltx", &[]).ok()?;
        let relations =
            load_ltx_tree_from_vfs(vfs, "configs/creatures/game_relations.ltx", &[]).ok();
        let faction = relations.as_ref().and_then(|relations| {
            let communities = relations.read_str("game_relations", "communities")?;
            let index = usize::try_from(community_index).ok()?.checked_mul(2)?;
            let id = communities.split(',').nth(index)?.trim();
            id.strip_prefix("actor_")
        });
        let cue = |key: &str| {
            let variant = faction.map(|faction| format!("{key}_{faction}"));
            let read = |field| {
                variant
                    .as_deref()
                    .and_then(|section| effects.read_str(section, field))
                    .or_else(|| effects.read_str(key, field))
            };
            let section = format!("{}_hud", read("anm")?);
            let duration = read("tm")?.parse::<f32>().ok()? * 0.001;
            let speed = read("speed")
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(1.0);
            if !duration.is_finite() || duration <= 0.0 || !speed.is_finite() || speed <= 0.0 {
                return None;
            }
            config.read_str(&section, "anm_ea_show")?;
            let camera = read("cam").and_then(|path| load_anm_from_vfs(vfs, path).ok());
            Some(BackpackCue {
                section,
                duration,
                speed,
                sound: read("snd").map(str::to_owned),
                camera,
            })
        };
        let opening = cue("backpack_open")?;
        let closing = cue("backpack_close")?;
        Some(Self {
            config,
            opening,
            closing,
            phase: BackpackPhase::Opening,
            elapsed: 0.0,
        })
    }

    pub fn advance(&mut self, dt: f32) -> bool {
        self.elapsed += dt;
        if self.phase == BackpackPhase::Opening && self.elapsed >= self.opening.duration {
            self.phase = BackpackPhase::Open;
            self.elapsed = 0.0;
            return true;
        }
        false
    }

    pub fn close(&mut self) {
        self.phase = BackpackPhase::Closing;
        self.elapsed = 0.0;
    }

    pub fn ready_to_restore(&self) -> bool {
        self.phase == BackpackPhase::Closing
            && self.elapsed >= self.closing.duration + CLOSE_RECOVERY_SEC
    }

    pub fn motion(&self) -> Option<(&BackpackCue, &str)> {
        match self.phase {
            BackpackPhase::Opening => Some((&self.opening, "anm_ea_show")),
            BackpackPhase::Open => Some((&self.opening, "anm_ea_idle")),
            BackpackPhase::Closing if self.elapsed < self.closing.duration => {
                Some((&self.closing, "anm_ea_show"))
            }
            BackpackPhase::Closing => None,
        }
    }

    pub fn camera_offset(&self) -> Option<XRayMatrix> {
        let cue = match self.phase {
            BackpackPhase::Opening => &self.opening,
            BackpackPhase::Closing => &self.closing,
            BackpackPhase::Open => return None,
        };
        let motion = cue.camera.as_ref()?;
        if self.elapsed >= motion.duration() {
            return None;
        }
        let (position, rotation) =
            motion.evaluate(motion.frame_start as f32 / motion.fps + self.elapsed);
        Some(matrix_set_xyzi_translate(rotation, position))
    }
}
