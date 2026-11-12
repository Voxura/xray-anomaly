//! Dynamic Level Transitions, Multi-Stage Loading Screen & Co-Op Level Sync (`xray-engine`).
//!
//! Coordinates Zone travel between levels (`l01_escape`, `l02_garbage`, `zaton`, `jupiter`, ...):
//! - Automatic extraction of `CSE_ALifeLevelChanger` trigger volumes (`ServerShapeDef::Sphere` /
//!   oriented `Box`) from `AlifeManager` and `LoadedWorld::level_spawn`, plus manual registration
//! - Player trigger volume overlap detection, re-arm suppression, and interactive confirmation prompt
//! - Deterministic 5-stage loading state machine:
//!   `Idle -> SavingPlayerState (.scoc/.scop) -> UnloadLevel -> LoadAssets -> Ready`
//! - `UiCanvas` loading splash screen, level title / Zone lore text, and multi-stage progress meter
//! - Synchronous multiplayer level travel via [`xray_net::LevelChangeNotice`] handshake

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::alife_manager::{AlifeManager, RemotePeerAvatar};
use crate::camera::FirstPersonCamera;
use crate::hud_weapon::HudWeaponController;
use crate::input::{Action, EngineInput, MouseButton};
use crate::physics::StalkerController;
use crate::render_bridge::RenderBridge;
use crate::ui_inventory::{InventoryManager, UiFontResource};
use crate::world_loader::{LevelLoadConfig, LoadedWorld, WorldLoader};
use xray_audio::AudioManager;
use xray_formats::scoc::ScocFile;
use xray_formats::scop::{
    classify_alife_section, ScopActorStats, ScopAlifeKind, ScopAlifePayload, ScopGraphAndQuests,
    ScopWeatherState,
};
use xray_formats::spawns::{AlifeLevelChangerState, ServerShapeDef};
use xray_formats::vfs_bridge::load_game_levels_from_vfs;
use xray_formats::world_maps::GameLevelsConfig;
use xray_net::{LevelChangeNotice, LevelChangePhase};
use xray_render_dx12::Dx12Renderer;
use xray_vfs::VirtualFileSystem;

/// Normalize a raw level identifier or path (e.g. `"levels\\l02_garbage"` -> `"l02_garbage"`).
pub fn canonical_level_name(raw: &str) -> String {
    let trimmed = raw.trim().replace('\\', "/");
    let stripped = trimmed
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or(&trimmed);
    stripped.to_ascii_lowercase()
}

/// Resolve the VFS directory path passed into [`WorldLoader::load_from_vfs`].
pub fn resolve_level_vfs_dir(vfs: Option<&VirtualFileSystem>, raw_level: &str) -> String {
    let trimmed = raw_level.trim();
    if let Some(vfs) = vfs {
        let lower = trimmed.replace('\\', "/");
        if (lower.contains('/') || lower.starts_with('$'))
            && (vfs.exists(&format!("{}/level", lower.trim_end_matches('/'))))
        {
            return lower.trim_end_matches('/').to_string();
        }
        let short = canonical_level_name(trimmed);
        let levels_cand = format!("levels/{short}");
        if vfs.exists(&format!("{levels_cand}/level")) {
            return levels_cand;
        }
        let alias_cand = format!("$game_levels$/{short}");
        if vfs.exists(&format!("{alias_cand}/level")) {
            return alias_cand;
        }
    }
    let short = canonical_level_name(trimmed);
    format!("levels/{short}")
}

/// Display title, lore summary, and VFS intro texture for a Zone level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneLevelMetadata {
    pub level_id: String,
    pub title: String,
    pub lore_text: String,
    pub intro_dds_vpath: String,
}

/// Look up curated Zone title & lore for known S.T.A.L.K.E.R. levels, optionally using `game_levels.ltx`.
pub fn lookup_level_metadata(
    raw_level: &str,
    game_levels: Option<&GameLevelsConfig>,
) -> ZoneLevelMetadata {
    let id = canonical_level_name(raw_level);
    let (default_title, lore) = match id.as_str() {
        "l01_escape" => (
            "Cordon",
            "Southern perimeter of the Zone. Military checkpoints guard the border while rookie stalkers gather near Sidorovich's bunker.",
        ),
        "l02_garbage" => (
            "Garbage",
            "Vast radioactive scrapyard north of Cordon. Bandits, mutant packs, and Duty outposts clash across twisted hills of Soviet armor.",
        ),
        "l03_agroprom" => (
            "Agroprom Research Institute",
            "Abandoned industrial complex hiding flooded catacombs and classified military archives beneath the Zone's western sector.",
        ),
        "l04_darkvalley" => (
            "Dark Valley",
            "Fog-shrouded basin of ruined factories and bandit strongholds near the sealed entrance to Lab X-18.",
        ),
        "l05_bar" => (
            "Rostok (100 Rads Bar)",
            "Fortified Duty stronghold and neutral stalker haven centered around the subterranean 100 Rads Bar.",
        ),
        "l06_rostok" => (
            "Wild Territory",
            "Collapsed railway yards and electro-anomaly corridors patrolled by mercenaries between Rostok and Yantar.",
        ),
        "l07_military" => (
            "Army Warehouses",
            "Freedom's headquarters and disputed borderlands blocking the Brain Scorcher approaches from the Red Forest.",
        ),
        "l08_yantar" => (
            "Yantar",
            "Drained toxic lake and mobile Ecologist bunker studying intense psi-emissions from Lab X-16.",
        ),
        "l10_red_forest" | "red_forest" => (
            "Red Forest",
            "Crimson pine woods choked with spatial anomalies, Monolith patrols, and the old mine roads toward Limansk.",
        ),
        "l10_radar" => (
            "Radar",
            "Decommissioned Scorcher antenna array and winding mountain checkpoints leading toward Pripyat.",
        ),
        "l11_pripyat" | "pripyat" => (
            "Pripyat",
            "Silent ghost city overlooking the Sarcophagus where Monolith squads hold the avenues around the Prometheus cinema.",
        ),
        "l12_stancia" | "l12_stancia_2" => (
            "Chernobyl Nuclear Power Plant",
            "Heart of the Zone beneath the ruined Unit 4 sarcophagus, torn by blowouts and temporal anomalies.",
        ),
        "zaton" => (
            "Zaton",
            "Swampy northern riverbeds littered with grounded barges around the Skadovsk hull and anomaly-warped plateaus.",
        ),
        "jupiter" => (
            "Jupiter",
            "Abandoned Yanov station and sprawling Jupiter plant in the northern Zone, contested by Duty, Freedom, and researchers.",
        ),
        _ => (
            "The Zone",
            "Uncharted sector beyond the Perimeter where shifting anomalies conceal rare artifacts and forgotten underground bunkers.",
        ),
    };

    let mut title = default_title.to_string();
    if let Some(cfg) = game_levels
        && let Some(entry) = cfg.levels.iter().find(|e| canonical_level_name(&e.name) == id)
        && !entry.caption.trim().is_empty()
    {
        let clean = entry.caption.trim();
        if !clean.starts_with("st_") {
            title = clean.to_string();
        }
    }
    if title == "The Zone" && !id.is_empty() {
        title = id
            .trim_start_matches("l0")
            .trim_start_matches("l1")
            .trim_start_matches('_')
            .split('_')
            .filter(|p| !p.is_empty())
            .map(|p| {
                let mut ch = p.chars();
                match ch.next() {
                    Some(first) => format!("{}{}", first.to_ascii_uppercase(), ch.as_str()),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        if title.is_empty() {
            title = "The Zone".to_string();
        }
    }

    ZoneLevelMetadata {
        intro_dds_vpath: format!("textures\\intro\\intro_{id}.dds"),
        level_id: id,
        title,
        lore_text: lore.to_string(),
    }
}

/// Level transition finite-state machine phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionState {
    Idle,
    SavingPlayerState,
    UnloadLevel,
    LoadAssets,
    Ready,
}

impl TransitionState {
    pub fn stage_index(self) -> usize {
        match self {
            Self::Idle => 0,
            Self::SavingPlayerState => 1,
            Self::UnloadLevel => 2,
            Self::LoadAssets => 3,
            Self::Ready => 4,
        }
    }

    pub fn stage_title(self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::SavingPlayerState => "Synchronizing Actor & Squad State (.scoc)",
            Self::UnloadLevel => "Unloading Previous Sector Geometry",
            Self::LoadAssets => "Streaming Level Geometry, Cform BVH, Weather & ALife",
            Self::Ready => "Synchronizing Zone Spawn",
        }
    }
}

/// Trigger volume representing an X-Ray `CSE_ALifeLevelChanger` or scripted Zone travel point.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelChangerVolume {
    pub entity_id: u16,
    pub name: String,
    pub source_level: String,
    pub target_level: String,
    pub target_point: String,
    pub origin: [f32; 3],
    pub shapes: Vec<ServerShapeDef>,
    pub fallback_radius: f32,
    pub spawn_pos: [f32; 3],
    pub spawn_angles: [f32; 3],
    pub next_graph_vertex: u16,
    pub next_level_vertex: u32,
    pub silent_mode: bool,
}

impl LevelChangerVolume {
    /// Construct a spherical trigger volume between `source_level` and `target_level`.
    #[allow(clippy::too_many_arguments)]
    pub fn spherical(
        entity_id: u16,
        name: impl Into<String>,
        source_level: impl Into<String>,
        target_level: impl Into<String>,
        trigger_pos: [f32; 3],
        radius: f32,
        spawn_pos: [f32; 3],
        spawn_angles: [f32; 3],
        silent_mode: bool,
    ) -> Self {
        let tgt = canonical_level_name(&target_level.into());
        Self {
            entity_id,
            name: name.into(),
            source_level: canonical_level_name(&source_level.into()),
            target_level: tgt.clone(),
            target_point: format!("{tgt}_start_actor"),
            origin: trigger_pos,
            shapes: vec![ServerShapeDef::Sphere {
                center: [0.0, 0.0, 0.0],
                radius: radius.max(0.25),
            }],
            fallback_radius: radius.max(0.25),
            spawn_pos,
            spawn_angles,
            next_graph_vertex: 0,
            next_level_vertex: 0,
            silent_mode,
        }
    }

    /// Parse an X-Ray `AlifeLevelChangerState` record into a runtime trigger volume.
    pub fn from_alife_state(
        entity_id: u16,
        name: impl Into<String>,
        source_level: &str,
        origin: [f32; 3],
        state: &AlifeLevelChangerState,
    ) -> Option<Self> {
        let target_level = canonical_level_name(&state.level_to_change);
        if target_level.is_empty() {
            return None;
        }
        Some(Self {
            entity_id,
            name: name.into(),
            source_level: canonical_level_name(source_level),
            target_level,
            target_point: state.level_point_to_change.clone(),
            origin,
            shapes: state.restrictor.shapes.clone(),
            fallback_radius: 2.5,
            spawn_pos: state.next_position,
            spawn_angles: state.next_angles,
            next_graph_vertex: state.next_graph_vertex,
            next_level_vertex: state.next_level_vertex,
            silent_mode: state.silent_mode,
        })
    }

    /// Test whether a world-space actor position lies inside this changer's trigger shapes.
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        if self.shapes.is_empty() {
            let dx = point[0] - self.origin[0];
            let dy = point[1] - self.origin[1];
            let dz = point[2] - self.origin[2];
            return dx * dx + dy * dy + dz * dz <= self.fallback_radius * self.fallback_radius;
        }
        self.shapes
            .iter()
            .any(|shape| point_in_server_shape(point, self.origin, shape))
    }
}

/// Evaluate whether `point` lies within a `CSE_Shape` attached at entity `origin`.
pub fn point_in_server_shape(point: [f32; 3], origin: [f32; 3], shape: &ServerShapeDef) -> bool {
    let rel = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    match shape {
        ServerShapeDef::Sphere { center, radius } => {
            let dx = rel[0] - center[0];
            let dy = rel[1] - center[1];
            let dz = rel[2] - center[2];
            let r = radius.max(0.0);
            dx * dx + dy * dy + dz * dz <= r * r
        }
        ServerShapeDef::Box(m) => {
            // `Fmatrix`: columns are i=[m0,m1,m2], j=[m3,m4,m5], k=[m6,m7,m8], offset c=[m9,m10,m11]
            // Local unit box spans [-0.5, +0.5] along each basis vector.
            let p = [rel[0] - m[9], rel[1] - m[10], rel[2] - m[11]];
            let a00 = m[0];
            let a10 = m[1];
            let a20 = m[2];
            let a01 = m[3];
            let a11 = m[4];
            let a21 = m[5];
            let a02 = m[6];
            let a12 = m[7];
            let a22 = m[8];

            let c00 = a11 * a22 - a12 * a21;
            let c01 = -(a10 * a22 - a12 * a20);
            let c02 = a10 * a21 - a11 * a20;
            let det = a00 * c00 + a01 * c01 + a02 * c02;
            if det.abs() < 1e-8 {
                return false;
            }
            let inv_det = 1.0 / det;

            let det_x = p[0] * (a11 * a22 - a12 * a21)
                - a01 * (p[1] * a22 - a12 * p[2])
                + a02 * (p[1] * a21 - a11 * p[2]);
            let det_y = a00 * (p[1] * a22 - a12 * p[2])
                - p[0] * (a10 * a22 - a12 * a20)
                + a02 * (a10 * p[2] - p[1] * a20);
            let det_z = a00 * (a11 * p[2] - p[1] * a21)
                - a01 * (a10 * p[2] - p[1] * a20)
                + p[0] * (a10 * a21 - a11 * a20);

            let u = det_x * inv_det;
            let v = det_y * inv_det;
            let w = det_z * inv_det;
            u.abs() <= 0.5001 && v.abs() <= 0.5001 && w.abs() <= 0.5001
        }
    }
}

/// Interactive confirmation dialog shown when stepping inside a non-silent level changer.
#[derive(Debug, Clone, PartialEq)]
pub struct TravelPrompt {
    pub changer_entity_id: u16,
    pub changer_name: String,
    pub source_level: String,
    pub target_level: String,
    pub target_title: String,
    pub target_point: String,
    pub spawn_pos: [f32; 3],
    pub spawn_angles: [f32; 3],
    pub next_graph_vertex: u16,
    pub next_level_vertex: u32,
}

/// Persisted cross-level state snapshot held across `UnloadLevel -> LoadAssets`.
#[derive(Debug, Clone)]
struct CarriedWorldSnapshot {
    scoc: ScocFile,
    actor_stats: ScopActorStats,
    quests: ScopGraphAndQuests,
    weather_snapshot: ScopWeatherState,
    game_time_ms: u64,
    time_factor: f32,
    normal_time_factor: f32,
    remote_peers: Vec<RemotePeerAvatar>,
}

/// Active multi-stage level transition state.
#[derive(Debug, Clone)]
pub struct ActiveTransition {
    pub transition_id: u32,
    pub initiator_slot: u8,
    pub source_level: String,
    pub target_level: String,
    pub target_level_vpath: String,
    pub target_title: String,
    pub lore_text: String,
    pub intro_dds_vpath: String,
    pub target_point: String,
    pub spawn_pos: [f32; 3],
    pub spawn_angles: [f32; 3],
    pub next_graph_vertex: u16,
    pub next_level_vertex: u32,
    pub progress: f32,
    pub stage_label: String,
    pub load_substep: u8,
    pub prepare_sent: bool,
    pub local_ready_sent: bool,
    pub peer_ready_received: bool,
    pub commit_sent: bool,
    pub commit_received: bool,
    pub elapsed_sec: f32,
    carried: Option<CarriedWorldSnapshot>,
    staged_world: Option<LoadedWorld>,
    staged_alife: Option<AlifeManager>,
}

/// Screen-space UI layout for travel prompt and multi-stage loading screen.
#[derive(Debug, Clone)]
pub struct TransitionUiLayout {
    pub prompt_panel_rect: [f32; 4],
    pub prompt_confirm_rect: [f32; 4],
    pub prompt_cancel_rect: [f32; 4],
    pub loading_card_rect: [f32; 4],
    pub loading_splash_rect: [f32; 4],
    pub loading_progress_rect: [f32; 4],
    pub stage_pip_rects: [[f32; 4]; 4],
}

impl TransitionUiLayout {
    pub fn compute(width: u32, height: u32) -> Self {
        let fw = width.max(320) as f32;
        let fh = height.max(240) as f32;

        let pw = (fw * 0.44).clamp(280.0, 520.0);
        let ph = 148.0;
        let px = (fw - pw) * 0.5;
        let py = (fh - ph) * 0.46;
        let prompt_panel_rect = [px, py, pw, ph];
        let btn_w = ((pw - 48.0) * 0.5).max(90.0);
        let prompt_confirm_rect = [px + 16.0, py + ph - 44.0, btn_w, 30.0];
        let prompt_cancel_rect = [px + pw - 16.0 - btn_w, py + ph - 44.0, btn_w, 30.0];

        let cw = (fw * 0.78).clamp(280.0, fw - 48.0);
        let ch = (fh * 0.78).clamp(200.0, fh - 48.0);
        let cx = (fw - cw) * 0.5;
        let cy = (fh - ch) * 0.5;
        let loading_card_rect = [cx, cy, cw, ch];

        let splash_h = (ch * 0.48).clamp(80.0, ch - 135.0);
        let loading_splash_rect = [cx + 20.0, cy + 56.0, cw - 40.0, splash_h];

        let bar_y = cy + ch - 46.0;
        let bar_w = cw - 40.0;
        let loading_progress_rect = [cx + 20.0, bar_y, bar_w, 18.0];

        let gap = 8.0;
        let pip_w = ((bar_w - gap * 3.0) / 4.0).max(24.0);
        let pip_y = bar_y - 24.0;
        let stage_pip_rects = [
            [cx + 20.0, pip_y, pip_w, 14.0],
            [cx + 20.0 + (pip_w + gap), pip_y, pip_w, 14.0],
            [cx + 20.0 + (pip_w + gap) * 2.0, pip_y, pip_w, 14.0],
            [cx + 20.0 + (pip_w + gap) * 3.0, pip_y, pip_w, 14.0],
        ];

        Self {
            prompt_panel_rect,
            prompt_confirm_rect,
            prompt_cancel_rect,
            loading_card_rect,
            loading_splash_rect,
            loading_progress_rect,
            stage_pip_rects,
        }
    }
}

/// Result summary from advancing one frame of [`LevelTransitionManager::step_transition`].
#[derive(Debug, Default, Clone)]
pub struct TransitionStepOutput {
    pub completed: bool,
    pub target_level_loaded: Option<String>,
    pub outgoing_notices: Vec<LevelChangeNotice>,
    pub error: Option<String>,
}

/// Coordinates level changers, multi-stage loading screens, `.scoc`/`.scop` transfer & co-op sync.
#[derive(Debug, Clone)]
pub struct LevelTransitionManager {
    pub state: TransitionState,
    pub current_level: String,
    pub changers: Vec<LevelChangerVolume>,
    pub manual_changers: Vec<LevelChangerVolume>,
    pub active_prompt: Option<TravelPrompt>,
    pub active_transition: Option<ActiveTransition>,
    pub game_levels: Option<GameLevelsConfig>,
    pub default_load_config: LevelLoadConfig,
    pub coop_sync_timeout_sec: f32,
    pub completed_transitions: u32,
    pub last_completed_level: Option<String>,
    pub last_error: Option<String>,
    next_transition_id: u32,
    suppressed_changer_id: Option<u16>,
    vfs_game_levels_attempted: bool,
}

impl Default for LevelTransitionManager {
    fn default() -> Self {
        Self::new("l01_escape")
    }
}

impl LevelTransitionManager {
    pub fn new(initial_level: impl Into<String>) -> Self {
        let lvl = canonical_level_name(&initial_level.into());
        Self {
            state: TransitionState::Idle,
            current_level: lvl.clone(),
            changers: Vec::new(),
            manual_changers: Vec::new(),
            active_prompt: None,
            active_transition: None,
            game_levels: None,
            default_load_config: LevelLoadConfig::default(),
            coop_sync_timeout_sec: 15.0,
            completed_transitions: 0,
            last_completed_level: None,
            last_error: None,
            next_transition_id: 1,
            suppressed_changer_id: None,
            vfs_game_levels_attempted: false,
        }
    }

    #[inline]
    pub fn is_loading(&self) -> bool {
        self.state != TransitionState::Idle
    }

    #[inline]
    pub fn is_modal_open(&self) -> bool {
        self.is_loading() || self.active_prompt.is_some()
    }

    /// Reset cached `game_levels.ltx` lookup flag when a new VFS is mounted.
    pub fn invalidate_vfs_cache(&mut self) {
        self.vfs_game_levels_attempted = false;
        self.game_levels = None;
    }

    /// Lazily stream `$game_configs$/game_levels.ltx` if present in `VFS`.
    pub fn ensure_game_levels_loaded(&mut self, vfs: &VirtualFileSystem) {
        if self.vfs_game_levels_attempted {
            return;
        }
        self.vfs_game_levels_attempted = true;
        if let Ok(cfg) = load_game_levels_from_vfs(vfs, &[]) {
            self.game_levels = Some(cfg);
        }
    }

    /// Register an explicit runtime or test [`LevelChangerVolume`].
    pub fn register_changer(&mut self, changer: LevelChangerVolume) {
        if let Some(existing) = self
            .manual_changers
            .iter_mut()
            .find(|c| c.entity_id == changer.entity_id && c.source_level == changer.source_level)
        {
            *existing = changer.clone();
        } else {
            self.manual_changers.push(changer.clone());
        }
        self.rebuild_active_changers();
    }

    /// Populate level changers from live [`AlifeManager`] entities and [`LoadedWorld::level_spawn`].
    pub fn discover_from_world_and_alife(
        &mut self,
        level_name: &str,
        world: Option<&LoadedWorld>,
        alife: Option<&AlifeManager>,
    ) {
        let lvl = canonical_level_name(level_name);
        self.current_level = lvl.clone();
        self.changers.clear();
        let mut seen_ids = BTreeSet::new();

        if let Some(mgr) = alife {
            for ent in mgr.entities.values() {
                if ent.kind == ScopAlifeKind::LevelChanger
                    && let Some(scop_ent) = &ent.source_scop
                    && let ScopAlifePayload::LevelChanger(state) = &scop_ent.payload
                    && let Some(vol) = LevelChangerVolume::from_alife_state(
                        ent.id,
                        ent.name.clone(),
                        &lvl,
                        ent.position,
                        state,
                    )
                {
                    seen_ids.insert(vol.entity_id);
                    self.changers.push(vol);
                }
            }
        }

        if let Some(w) = world
            && let Some(spawn) = &w.level_spawn
        {
            for pkt in &spawn.spawns {
                let kind = classify_alife_section(pkt.id, &pkt.section);
                if kind != ScopAlifeKind::LevelChanger || seen_ids.contains(&pkt.id) {
                    continue;
                }
                if let Ok(state) = AlifeLevelChangerState::parse(pkt.version, &pkt.state_payload)
                    && let Some(vol) = LevelChangerVolume::from_alife_state(
                        pkt.id,
                        pkt.name_replace.clone(),
                        &lvl,
                        pkt.position,
                        &state,
                    )
                {
                    seen_ids.insert(vol.entity_id);
                    self.changers.push(vol);
                }
            }
        }

        for manual in &self.manual_changers {
            if manual.source_level == lvl && !seen_ids.contains(&manual.entity_id) {
                seen_ids.insert(manual.entity_id);
                self.changers.push(manual.clone());
            }
        }
    }

    fn rebuild_active_changers(&mut self) {
        let mut next: Vec<LevelChangerVolume> = self
            .changers
            .iter()
            .filter(|c| {
                !self
                    .manual_changers
                    .iter()
                    .any(|m| m.entity_id == c.entity_id && m.source_level == self.current_level)
            })
            .cloned()
            .collect();
        for manual in &self.manual_changers {
            if manual.source_level == self.current_level {
                next.push(manual.clone());
            }
        }
        self.changers = next;
    }

    /// Detect player overlap against current level's trigger volumes (`TransitionState::Idle` only).
    ///
    /// Returns an optional [`LevelChangeNotice`] (`PrepareTransition`) if a silent changer triggers.
    pub fn scan_player_triggers(
        &mut self,
        player_pos: [f32; 3],
        local_slot: u8,
        networked: bool,
    ) -> Option<LevelChangeNotice> {
        if self.state != TransitionState::Idle {
            return None;
        }
        let matching = self
            .changers
            .iter()
            .find(|c| c.contains_point(player_pos))
            .cloned();

        let Some(changer) = matching else {
            self.suppressed_changer_id = None;
            self.active_prompt = None;
            return None;
        };

        if self.suppressed_changer_id == Some(changer.entity_id) {
            return None;
        }

        if changer.silent_mode {
            self.active_prompt = None;
            self.suppressed_changer_id = Some(changer.entity_id);
            return self.begin_transition_from_changer(&changer, local_slot, networked);
        }

        let meta = lookup_level_metadata(&changer.target_level, self.game_levels.as_ref());
        self.active_prompt = Some(TravelPrompt {
            changer_entity_id: changer.entity_id,
            changer_name: changer.name,
            source_level: changer.source_level,
            target_level: changer.target_level,
            target_title: meta.title,
            target_point: changer.target_point,
            spawn_pos: changer.spawn_pos,
            spawn_angles: changer.spawn_angles,
            next_graph_vertex: changer.next_graph_vertex,
            next_level_vertex: changer.next_level_vertex,
        });
        None
    }

    /// Confirm the active [`TravelPrompt`] and start the loading pipeline.
    pub fn confirm_prompt(
        &mut self,
        local_slot: u8,
        networked: bool,
        audio: Option<&mut AudioManager>,
    ) -> Option<LevelChangeNotice> {
        let prompt = self.active_prompt.take()?;
        self.suppressed_changer_id = Some(prompt.changer_entity_id);
        if let Some(aud) = audio {
            aud.play_ui_click("interface\\pda_task");
        }
        self.start_transition_internal(
            None,
            local_slot,
            &prompt.target_level,
            &prompt.target_point,
            prompt.spawn_pos,
            prompt.spawn_angles,
            prompt.next_graph_vertex,
            prompt.next_level_vertex,
            networked,
            true,
        )
    }

    /// Dismiss the active [`TravelPrompt`] and suppress re-triggering until leaving the volume.
    pub fn cancel_prompt(&mut self, audio: Option<&mut AudioManager>) {
        if let Some(prompt) = self.active_prompt.take() {
            self.suppressed_changer_id = Some(prompt.changer_entity_id);
            if let Some(aud) = audio {
                aud.play_ui_close();
            }
        }
    }

    /// Begin a level transition directly from a [`LevelChangerVolume`].
    pub fn begin_transition_from_changer(
        &mut self,
        changer: &LevelChangerVolume,
        local_slot: u8,
        networked: bool,
    ) -> Option<LevelChangeNotice> {
        self.start_transition_internal(
            None,
            local_slot,
            &changer.target_level,
            &changer.target_point,
            changer.spawn_pos,
            changer.spawn_angles,
            changer.next_graph_vertex,
            changer.next_level_vertex,
            networked,
            true,
        )
    }

    /// Begin a level transition directly to `target_level` at `(spawn_pos, spawn_angles)`.
    pub fn begin_transition(
        &mut self,
        target_level: &str,
        spawn_pos: [f32; 3],
        spawn_angles: [f32; 3],
        local_slot: u8,
        networked: bool,
    ) -> Option<LevelChangeNotice> {
        let tgt = canonical_level_name(target_level);
        self.start_transition_internal(
            None,
            local_slot,
            &tgt,
            &format!("{tgt}_start_actor"),
            spawn_pos,
            spawn_angles,
            0,
            0,
            networked,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start_transition_internal(
        &mut self,
        explicit_id: Option<u32>,
        initiator_slot: u8,
        target_level: &str,
        target_point: &str,
        spawn_pos: [f32; 3],
        spawn_angles: [f32; 3],
        next_graph_vertex: u16,
        next_level_vertex: u32,
        networked: bool,
        emit_prepare_notice: bool,
    ) -> Option<LevelChangeNotice> {
        if self.state != TransitionState::Idle {
            return None;
        }
        let transition_id = explicit_id.unwrap_or_else(|| {
            let id = self.next_transition_id;
            self.next_transition_id = self.next_transition_id.wrapping_add(1).max(1);
            id
        });
        let meta = lookup_level_metadata(target_level, self.game_levels.as_ref());
        self.state = TransitionState::SavingPlayerState;
        self.active_prompt = None;
        self.last_error = None;

        let active = ActiveTransition {
            transition_id,
            initiator_slot,
            source_level: self.current_level.clone(),
            target_level: meta.level_id.clone(),
            target_level_vpath: resolve_level_vfs_dir(None, &meta.level_id),
            target_title: meta.title,
            lore_text: meta.lore_text,
            intro_dds_vpath: meta.intro_dds_vpath,
            target_point: target_point.to_string(),
            spawn_pos,
            spawn_angles,
            next_graph_vertex,
            next_level_vertex,
            progress: 0.12,
            stage_label: TransitionState::SavingPlayerState.stage_title().to_string(),
            load_substep: 0,
            prepare_sent: networked && emit_prepare_notice,
            local_ready_sent: false,
            peer_ready_received: false,
            commit_sent: false,
            commit_received: false,
            elapsed_sec: 0.0,
            carried: None,
            staged_world: None,
            staged_alife: None,
        };
        let notice = if networked && emit_prepare_notice {
            Some(LevelChangeNotice {
                transition_id,
                sender_slot: initiator_slot,
                phase: LevelChangePhase::PrepareTransition,
                source_level: active.source_level.clone(),
                target_level: active.target_level.clone(),
                target_point: active.target_point.clone(),
                spawn_pos: active.spawn_pos,
                spawn_angles: active.spawn_angles,
                game_vertex_id: active.next_graph_vertex,
                level_vertex_id: active.next_level_vertex,
            })
        } else {
            None
        };
        self.active_transition = Some(active);
        notice
    }

    /// Process prompt keyboard/mouse interactions while [`TravelPrompt`] is open.
    pub fn handle_input(
        &mut self,
        input: &EngineInput,
        viewport_w: u32,
        viewport_h: u32,
        local_slot: u8,
        networked: bool,
        mut audio: Option<&mut AudioManager>,
    ) -> Option<LevelChangeNotice> {
        if self.state != TransitionState::Idle || self.active_prompt.is_none() {
            return None;
        }
        if input.is_action_just_pressed(Action::Menu) {
            self.cancel_prompt(audio);
            return None;
        }
        if input.is_action_just_pressed(Action::Use) {
            return self.confirm_prompt(local_slot, networked, audio);
        }
        if input.is_mouse_just_pressed(MouseButton::Left) {
            let layout = TransitionUiLayout::compute(viewport_w, viewport_h);
            let (mx, my) = input.mouse_position();
            if point_in_rect(mx, my, layout.prompt_confirm_rect) {
                return self.confirm_prompt(local_slot, networked, audio.as_deref_mut());
            }
            if point_in_rect(mx, my, layout.prompt_cancel_rect) {
                self.cancel_prompt(audio);
                return None;
            }
        }
        None
    }

    /// Consume an incoming co-op [`LevelChangeNotice`] from `xray-net`.
    pub fn handle_network_notice(&mut self, notice: &LevelChangeNotice, local_slot: u8) {
        match notice.phase {
            LevelChangePhase::PrepareTransition => {
                if self.state == TransitionState::Idle {
                    let _ = self.start_transition_internal(
                        Some(notice.transition_id),
                        local_slot,
                        &notice.target_level,
                        &notice.target_point,
                        notice.spawn_pos,
                        notice.spawn_angles,
                        notice.game_vertex_id,
                        notice.level_vertex_id,
                        true,
                        false,
                    );
                }
            }
            LevelChangePhase::ReadyAtDestination => {
                if let Some(active) = &mut self.active_transition
                    && active.transition_id == notice.transition_id
                {
                    active.peer_ready_received = true;
                }
            }
            LevelChangePhase::CommitSpawn => {
                if let Some(active) = &mut self.active_transition
                    && active.transition_id == notice.transition_id
                {
                    active.commit_received = true;
                }
            }
            LevelChangePhase::Abort => {
                if let Some(active) = &self.active_transition
                    && active.transition_id == notice.transition_id
                {
                    self.active_transition = None;
                    self.state = TransitionState::Idle;
                }
            }
        }
    }

    /// Advance the active loading state machine by one frame.
    #[allow(clippy::too_many_arguments)]
    pub fn step_transition(
        &mut self,
        dt: f32,
        vfs: Option<&VirtualFileSystem>,
        world: &mut Option<LoadedWorld>,
        alife: &mut Option<AlifeManager>,
        renderer: &mut Option<Dx12Renderer>,
        render_bridge: &mut Option<RenderBridge>,
        inventory: &mut InventoryManager,
        hud_weapon: &mut HudWeaponController,
        camera: &mut FirstPersonCamera,
        physics: &mut StalkerController,
        audio: &mut AudioManager,
        is_network_host: bool,
        has_connected_peer: bool,
        local_slot: u8,
    ) -> TransitionStepOutput {
        let mut out = TransitionStepOutput::default();
        if self.state == TransitionState::Idle {
            return out;
        }
        let Some(mut active) = self.active_transition.take() else {
            self.state = TransitionState::Idle;
            return out;
        };
        active.elapsed_sec += dt.max(0.0);

        match self.state {
            TransitionState::Idle => {
                return out;
            }
            TransitionState::SavingPlayerState => {
                let target_level_title = active.target_level.clone();
                let scoc_saved = inventory.export_to_scoc();
                let mut actor_stats = inventory.actor_stats.clone();
                actor_stats.position = active.spawn_pos;
                actor_stats.angles = active.spawn_angles;

                let (quests, weather_snapshot, game_time_ms, time_factor, normal_time_factor, peers) =
                    if let Some(mgr) = alife.as_ref() {
                        let mut q = mgr.graph_and_quests.clone();
                        q.sync_registries();
                        let weather_snap = world
                            .as_ref()
                            .map(|w| w.weather.to_scop_weather())
                            .unwrap_or_else(|| mgr.weather_snapshot.clone());
                        let peer_list: Vec<RemotePeerAvatar> =
                            mgr.remote_peers.values().cloned().collect();
                        (
                            q,
                            weather_snap,
                            mgr.game_time_ms,
                            mgr.time_factor,
                            mgr.normal_time_factor,
                            peer_list,
                        )
                    } else {
                        let weather_snap = world
                            .as_ref()
                            .map(|w| w.weather.to_scop_weather())
                            .unwrap_or_default();
                        (
                            ScopGraphAndQuests::default(),
                            weather_snap,
                            0,
                            10.0,
                            10.0,
                            Vec::new(),
                        )
                    };

                active.carried = Some(CarriedWorldSnapshot {
                    scoc: scoc_saved,
                    actor_stats,
                    quests,
                    weather_snapshot,
                    game_time_ms,
                    time_factor,
                    normal_time_factor,
                    remote_peers: peers,
                });
                active.target_level_vpath = resolve_level_vfs_dir(vfs, &target_level_title);
                active.progress = 0.25;
                active.stage_label = TransitionState::UnloadLevel.stage_title().to_string();
                self.state = TransitionState::UnloadLevel;
                self.active_transition = Some(active);
            }
            TransitionState::UnloadLevel => {
                *world = None;
                *alife = None;
                self.changers.clear();
                if let Some(rb) = render_bridge.as_mut() {
                    rb.invalidate_world();
                }
                audio.set_occlusion_mesh(None);
                audio.retain_alife_emitters(&BTreeSet::new());

                active.progress = 0.50;
                active.load_substep = 0;
                active.stage_label = TransitionState::LoadAssets.stage_title().to_string();
                self.state = TransitionState::LoadAssets;
                self.active_transition = Some(active);
            }
            TransitionState::LoadAssets => {
                match active.load_substep {
                    0 => {
                        let mut cfg = self.default_load_config.clone();
                        cfg.level_dir = resolve_level_vfs_dir(vfs, &active.target_level);
                        active.target_level_vpath = cfg.level_dir.clone();

                        if let Some(vfs_ref) = vfs {
                            self.ensure_game_levels_loaded(vfs_ref);
                            match WorldLoader::load_from_vfs(vfs_ref, &cfg) {
                                Ok(loaded_world) => {
                                    active.staged_world = Some(loaded_world);
                                }
                                Err(err) => {
                                    let msg = format!(
                                        "failed to load level '{}' ({}): {err}",
                                        active.target_level, cfg.level_dir
                                    );
                                    self.last_error = Some(msg.clone());
                                    out.error = Some(msg);
                                    if has_connected_peer {
                                        out.outgoing_notices.push(LevelChangeNotice {
                                            transition_id: active.transition_id,
                                            sender_slot: local_slot,
                                            phase: LevelChangePhase::Abort,
                                            source_level: active.source_level,
                                            target_level: active.target_level,
                                            target_point: active.target_point,
                                            spawn_pos: active.spawn_pos,
                                            spawn_angles: active.spawn_angles,
                                            game_vertex_id: active.next_graph_vertex,
                                            level_vertex_id: active.next_level_vertex,
                                        });
                                    }
                                    self.state = TransitionState::Idle;
                                    self.active_transition = None;
                                    return out;
                                }
                            }
                        } else {
                            let msg = format!(
                                "cannot stream target level '{}' without mounted VFS",
                                active.target_level
                            );
                            self.last_error = Some(msg.clone());
                            out.error = Some(msg);
                            self.state = TransitionState::Idle;
                            self.active_transition = None;
                            return out;
                        }
                        active.load_substep = 1;
                        active.progress = 0.68;
                        active.stage_label =
                            format!("Streaming {} geometry & Cform BVH...", active.target_title);
                        self.active_transition = Some(active);
                    }
                    1 => {
                        let mut staged_world = active.staged_world.take();
                        let mut new_mgr = AlifeManager::default();
                        new_mgr.level_name = active.target_level.clone();
                        if let Some(w) = staged_world.as_ref()
                            && let Some(spawn) = &w.level_spawn
                        {
                            new_mgr.seed_from_level_spawn(&active.target_level, spawn);
                        }
                        if let Some(carried) = active.carried.as_ref() {
                            new_mgr.graph_and_quests = carried.quests.clone();
                            new_mgr.weather_snapshot = carried.weather_snapshot.clone();
                            new_mgr.game_time_ms = carried.game_time_ms;
                            new_mgr.time_factor = carried.time_factor;
                            new_mgr.normal_time_factor = carried.normal_time_factor;

                            if let Some(w) = staged_world.as_mut() {
                                w.weather.apply_scop_weather(&carried.weather_snapshot);
                            }

                            inventory.load_from_scoc(&carried.scoc);
                            inventory.actor_stats = carried.actor_stats.clone();
                            inventory.sync_hud_weapon(hud_weapon);
                            new_mgr.actor_stats = inventory.actor_stats.clone();

                            for peer in &carried.remote_peers {
                                let scoc = ScocFile::from_peer_state(&peer.peer_state);
                                let id = new_mgr.attach_peer_scoc(
                                    peer.slot,
                                    peer.nickname.clone(),
                                    peer.lan_id,
                                    &scoc,
                                );
                                let offset_pos = [
                                    active.spawn_pos[0] + 1.5 * f32::from(peer.slot.max(1)),
                                    active.spawn_pos[1],
                                    active.spawn_pos[2],
                                ];
                                if let Some(avatar) = new_mgr.remote_peers.get_mut(&id) {
                                    avatar.position = offset_pos;
                                    avatar.yaw = active.spawn_angles[1];
                                    avatar.pitch = active.spawn_angles[0];
                                    avatar.velocity = [0.0; 3];
                                }
                                if let Some(ent) = new_mgr.entities.get_mut(&id) {
                                    ent.position = offset_pos;
                                    ent.angle = active.spawn_angles;
                                    ent.velocity = [0.0; 3];
                                }
                            }
                        }

                        // Position host/local actor, camera, and capsule controller at target spawn
                        camera.position = active.spawn_pos;
                        camera.pitch = active.spawn_angles[0];
                        camera.yaw = active.spawn_angles[1];
                        physics.sync_from_eye(camera.position);
                        new_mgr.set_host_camera_transform(
                            camera.position,
                            camera.yaw,
                            camera.pitch,
                            [0.0; 3],
                        );

                        if let Some(w) = staged_world.as_ref() {
                            audio.set_occlusion_mesh(w.sound_occlusion.clone().map(Arc::new));
                        }

                        *world = staged_world;
                        if let (Some(r), Some(rb), Some(w)) =
                            (renderer.as_mut(), render_bridge.as_mut(), world.as_ref())
                        {
                            let _ = rb.sync_world(r, w, vfs);
                        }
                        active.staged_alife = Some(new_mgr);
                        active.load_substep = 2;
                        active.progress = 0.85;
                        active.stage_label =
                            format!("Hydrating {} ALife & .scoc state...", active.target_title);
                        self.active_transition = Some(active);
                    }
                    _ => {
                        let staged_alife = active.staged_alife.take();
                        *alife = staged_alife;
                        self.discover_from_world_and_alife(
                            &active.target_level,
                            world.as_ref(),
                            alife.as_ref(),
                        );

                        if has_connected_peer && !active.local_ready_sent {
                            active.local_ready_sent = true;
                            out.outgoing_notices.push(LevelChangeNotice {
                                transition_id: active.transition_id,
                                sender_slot: local_slot,
                                phase: LevelChangePhase::ReadyAtDestination,
                                source_level: active.source_level.clone(),
                                target_level: active.target_level.clone(),
                                target_point: active.target_point.clone(),
                                spawn_pos: active.spawn_pos,
                                spawn_angles: active.spawn_angles,
                                game_vertex_id: active.next_graph_vertex,
                                level_vertex_id: active.next_level_vertex,
                            });
                        }

                        active.progress = 0.94;
                        active.stage_label = if has_connected_peer {
                            "Waiting for co-op squad spawn sync...".to_string()
                        } else {
                            "Ready to enter sector".to_string()
                        };
                        self.state = TransitionState::Ready;
                        self.active_transition = Some(active);
                    }
                }
            }
            TransitionState::Ready => {
                let synced = if !has_connected_peer {
                    true
                } else if is_network_host {
                    if active.peer_ready_received || active.elapsed_sec >= self.coop_sync_timeout_sec
                    {
                        if !active.commit_sent {
                            active.commit_sent = true;
                            out.outgoing_notices.push(LevelChangeNotice {
                                transition_id: active.transition_id,
                                sender_slot: local_slot,
                                phase: LevelChangePhase::CommitSpawn,
                                source_level: active.source_level.clone(),
                                target_level: active.target_level.clone(),
                                target_point: active.target_point.clone(),
                                spawn_pos: active.spawn_pos,
                                spawn_angles: active.spawn_angles,
                                game_vertex_id: active.next_graph_vertex,
                                level_vertex_id: active.next_level_vertex,
                            });
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    active.commit_received || active.elapsed_sec >= self.coop_sync_timeout_sec
                };

                if synced {
                    let target = active.target_level.clone();
                    self.current_level = target.clone();
                    self.last_completed_level = Some(target.clone());
                    self.completed_transitions = self.completed_transitions.saturating_add(1);
                    // Suppress immediate re-trigger if destination overlaps any return changer
                    self.suppressed_changer_id = self
                        .changers
                        .iter()
                        .find(|c| c.contains_point(camera.position))
                        .map(|c| c.entity_id);
                    self.active_prompt = None;
                    self.active_transition = None;
                    self.state = TransitionState::Idle;
                    out.completed = true;
                    out.target_level_loaded = Some(target);
                } else {
                    self.active_transition = Some(active);
                }
            }
        }

        out
    }

    /// Render the travel confirmation modal or multi-stage loading screen onto `UiCanvas`.
    pub fn render_ui(
        &self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        width: u32,
        height: u32,
        font: Option<&UiFontResource>,
    ) -> u32 {
        let mut drawn = 0u32;
        let fw = width.max(64) as f32;
        let fh = height.max(64) as f32;
        let layout = TransitionUiLayout::compute(width, height);

        if let Some(active) = &self.active_transition {
            // Full-screen Zone loading splash overlay
            renderer.ui_fill_rect([0.0, 0.0, fw, fh], [0.03, 0.04, 0.04, 0.98]);
            renderer.ui_fill_rect(layout.loading_card_rect, [0.09, 0.11, 0.12, 0.96]);
            renderer.ui_fill_rect(
                [
                    layout.loading_card_rect[0],
                    layout.loading_card_rect[1],
                    layout.loading_card_rect[2],
                    42.0,
                ],
                [0.14, 0.18, 0.16, 0.98],
            );
            drawn += 3;

            // Optional VFS intro splash texture or stylized Zone vector crest
            renderer.ui_fill_rect(layout.loading_splash_rect, [0.06, 0.08, 0.08, 0.96]);
            drawn += 1;
            let splash_tex = vfs.and_then(|vfs_ref| {
                for cand in [
                    active.intro_dds_vpath.as_str(),
                    "textures\\ui\\ui_load.dds",
                ] {
                    if let Some(existing) = renderer.get_texture(cand) {
                        return Some(existing.descriptor_index);
                    }
                    if let Ok(bytes) = vfs_ref.read_bytes(cand)
                        && let Ok(idx) = renderer.load_texture_from_dds(cand, &bytes)
                    {
                        return Some(idx);
                    }
                }
                None
            });
            if let Some(tex_idx) = splash_tex {
                renderer.ui_draw_textured_rect(
                    layout.loading_splash_rect,
                    [0.0, 0.0, 1.0, 1.0],
                    tex_idx,
                    [1.0, 1.0, 1.0, 0.96],
                );
                drawn += 1;
            } else {
                // Atmospheric horizon + Zone silhouette bands
                let sr = layout.loading_splash_rect;
                renderer.ui_fill_rect(
                    [sr[0] + 6.0, sr[1] + 6.0, sr[2] - 12.0, sr[3] * 0.55],
                    [0.14, 0.21, 0.18, 0.85],
                );
                renderer.ui_fill_rect(
                    [sr[0] + 6.0, sr[1] + sr[3] * 0.62, sr[2] - 12.0, sr[3] * 0.30],
                    [0.24, 0.19, 0.12, 0.85],
                );
                drawn += 2;
            }

            // 4-stage pipeline milestone pips
            let cur_stage = self.state.stage_index();
            for (idx, pip) in layout.stage_pip_rects.iter().enumerate() {
                let stage_num = idx + 1;
                let col = if stage_num < cur_stage {
                    [0.24, 0.72, 0.36, 0.95]
                } else if stage_num == cur_stage {
                    [0.88, 0.68, 0.20, 0.98]
                } else {
                    [0.18, 0.20, 0.22, 0.88]
                };
                renderer.ui_fill_rect(*pip, col);
                drawn += 1;
            }

            // Main progress bar
            renderer.ui_draw_meter_bar(
                layout.loading_progress_rect,
                active.progress.clamp(0.0, 1.0),
                [0.08, 0.09, 0.10, 0.92],
                [0.32, 0.78, 0.40, 0.98],
            );
            drawn += 2;

            if let Some(f) = font {
                let header = format!(
                    "ZONE TRANSIT: {} -> {}",
                    active.source_level.to_ascii_uppercase(),
                    active.target_title.to_ascii_uppercase()
                );
                drawn += f.draw_text(
                    renderer,
                    &header,
                    [layout.loading_card_rect[0] + 20.0, layout.loading_card_rect[1] + 12.0],
                    18.0,
                    [0.95, 0.92, 0.80, 1.0],
                );
                let lore_y = layout.loading_splash_rect[1] + layout.loading_splash_rect[3] + 12.0;
                drawn += f.draw_text(
                    renderer,
                    &active.lore_text,
                    [layout.loading_card_rect[0] + 20.0, lore_y],
                    14.0,
                    [0.82, 0.84, 0.80, 0.96],
                );
                let pct = (active.progress.clamp(0.0, 1.0) * 100.0).round() as u32;
                let stage_txt = format!("{} ({}%)", active.stage_label, pct);
                drawn += f.draw_text(
                    renderer,
                    &stage_txt,
                    [layout.loading_progress_rect[0], lore_y + 24.0],
                    14.0,
                    [0.92, 0.88, 0.56, 1.0],
                );
            }
            return drawn;
        }

        if let Some(prompt) = &self.active_prompt {
            renderer.ui_fill_rect([0.0, 0.0, fw, fh], [0.02, 0.03, 0.04, 0.42]);
            renderer.ui_fill_rect(layout.prompt_panel_rect, [0.10, 0.12, 0.13, 0.96]);
            renderer.ui_fill_rect(
                [
                    layout.prompt_panel_rect[0],
                    layout.prompt_panel_rect[1],
                    layout.prompt_panel_rect[2],
                    32.0,
                ],
                [0.17, 0.22, 0.19, 0.98],
            );
            renderer.ui_fill_rect(layout.prompt_confirm_rect, [0.22, 0.56, 0.30, 0.96]);
            renderer.ui_fill_rect(layout.prompt_cancel_rect, [0.48, 0.22, 0.22, 0.96]);
            drawn += 5;

            if let Some(f) = font {
                let title = format!("Travel to {}?", prompt.target_title);
                drawn += f.draw_text(
                    renderer,
                    &title,
                    [layout.prompt_panel_rect[0] + 16.0, layout.prompt_panel_rect[1] + 8.0],
                    16.0,
                    [0.95, 0.94, 0.86, 1.0],
                );
                let sub = format!("Destination point: {}", prompt.target_point);
                drawn += f.draw_text(
                    renderer,
                    &sub,
                    [layout.prompt_panel_rect[0] + 16.0, layout.prompt_panel_rect[1] + 52.0],
                    14.0,
                    [0.82, 0.86, 0.82, 0.96],
                );
                drawn += f.draw_text(
                    renderer,
                    "[E] Confirm Travel",
                    [layout.prompt_confirm_rect[0] + 12.0, layout.prompt_confirm_rect[1] + 7.0],
                    14.0,
                    [0.96, 0.98, 0.92, 1.0],
                );
                drawn += f.draw_text(
                    renderer,
                    "[Esc] Stay Here",
                    [layout.prompt_cancel_rect[0] + 12.0, layout.prompt_cancel_rect[1] + 7.0],
                    14.0,
                    [0.96, 0.92, 0.92, 1.0],
                );
            }
        }

        drawn
    }
}

fn point_in_rect(px: f32, py: f32, rect: [f32; 4]) -> bool {
    px >= rect[0] && py >= rect[1] && px <= rect[0] + rect[2] && py <= rect[1] + rect[3]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_volumes_and_prompt_confirmation() {
        let mut mgr = LevelTransitionManager::new("l01_escape");
        mgr.register_changer(LevelChangerVolume::spherical(
            900,
            "esc_to_garbage",
            "l01_escape",
            "l02_garbage",
            [12.0, 0.0, 240.0],
            5.0,
            [-110.0, 1.65, -260.0],
            [0.0, 0.25, 0.0],
            false,
        ));
        // Oriented box changer (`l01_escape -> zaton`) scaled 10x4x10 centered at origin + [50, 1, -20]
        mgr.register_changer(LevelChangerVolume {
            entity_id: 901,
            name: "esc_to_zaton_box".to_string(),
            source_level: "l01_escape".to_string(),
            target_level: "zaton".to_string(),
            target_point: "zat_actor_start".to_string(),
            origin: [50.0, 1.0, -20.0],
            shapes: vec![ServerShapeDef::Box([
                10.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0,
            ])],
            fallback_radius: 2.0,
            spawn_pos: [22.0, 4.0, 18.0],
            spawn_angles: [0.0, 1.57, 0.0],
            next_graph_vertex: 320,
            next_level_vertex: 12_345,
            silent_mode: false,
        });

        // Outside all volumes
        assert!(mgr.scan_player_triggers([0.0, 0.0, 0.0], 0, false).is_none());
        assert!(mgr.active_prompt.is_none());

        // Inside sphere volume -> prompts travel to Garbage
        assert!(mgr.scan_player_triggers([13.0, 1.0, 241.0], 0, false).is_none());
        let prompt = mgr.active_prompt.as_ref().expect("expected travel prompt");
        assert_eq!(prompt.target_level, "l02_garbage");
        assert_eq!(prompt.target_title, "Garbage");

        // Cancel suppresses re-opening until stepping out
        mgr.cancel_prompt(None);
        assert!(mgr.active_prompt.is_none());
        assert!(mgr.scan_player_triggers([13.0, 1.0, 241.0], 0, false).is_none());
        assert!(mgr.active_prompt.is_none());

        // Step outside then into box trigger -> prompts travel to Zaton
        assert!(mgr.scan_player_triggers([0.0, 0.0, 0.0], 0, false).is_none());
        assert!(mgr.scan_player_triggers([54.5, 2.5, -15.5], 0, false).is_none());
        let zaton_prompt = mgr.active_prompt.as_ref().expect("expected Zaton box prompt");
        assert_eq!(zaton_prompt.target_level, "zaton");
        assert_eq!(zaton_prompt.target_title, "Zaton");

        // Confirm with networking enabled -> emits PrepareTransition notice and enters SavingPlayerState
        let notice = mgr
            .confirm_prompt(0, true, None)
            .expect("confirming prompt while networked must emit PrepareTransition");
        assert_eq!(notice.phase, LevelChangePhase::PrepareTransition);
        assert_eq!(notice.target_level, "zaton");
        assert_eq!(mgr.state, TransitionState::SavingPlayerState);
    }

    #[test]
    fn test_level_metadata_and_canonical_names() {
        assert_eq!(canonical_level_name("levels\\L02_Garbage\\"), "l02_garbage");
        let meta = lookup_level_metadata("jupiter", None);
        assert_eq!(meta.title, "Jupiter");
        assert!(meta.lore_text.contains("Yanov"));
        assert_eq!(meta.intro_dds_vpath, "textures\\intro\\intro_jupiter.dds");
    }
}
