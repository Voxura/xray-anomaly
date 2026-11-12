//! Interactive Stalker PDA Screen: Level Map, Co-Op Peer Radar & Story Quest Tracker (`xray-engine`).
//!
//! Coordinates:
//! - Level minimap streaming via [`WorldMapsConfig`] (`load_world_maps_from_vfs`) and VFS `.dds`
//!   map textures (falling back cleanly to a tactical sector grid when VFS textures are absent)
//! - World `(x, z) -> [u, v]` radar projection for:
//!   - Local player position & heading indicator
//!   - Connected co-op peer avatars (`AlifeManager::remote_peers`)
//!   - Friendly Stalkers, Traders & Quest target map markers
//! - Story Quest tracker hydrated directly from `AlifeManager::graph_and_quests.story_quests`
//!   (`ScopStoryQuest`) and `active_quest_id`
//! - Toggle keybinding `P` (`Action::Pda`), interactive tabs (`Map`, `Quests`, `Contacts`), zoom/pan,
//!   and `AudioManager` PDA open/close/task SFX.

use crate::alife_manager::{AlifeManager, EntityRole};
use crate::input::{EngineInput, MouseButton};
use crate::ui_inventory::UiFontResource;
use xray_audio::AudioManager;
use xray_formats::spawns::INVALID_OBJECT_ID;
use xray_formats::vfs_bridge::load_world_maps_from_vfs;
use xray_formats::world_maps::WorldMapsConfig;
use xray_render_dx12::bindless::DescriptorIndex;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_vfs::VirtualFileSystem;

/// Active section inside the PDA interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PdaTab {
    Map,
    Quests,
    Contacts,
}

/// Radar / minimap marker classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MapBlipKind {
    Player,
    CoopPeer,
    FriendlyStalker,
    Trader,
    QuestTarget,
}

impl MapBlipKind {
    pub fn color_rgba(self) -> [f32; 4] {
        match self {
            Self::Player => [0.22, 0.94, 0.42, 1.0],
            Self::CoopPeer => [0.24, 0.68, 0.98, 1.0],
            Self::FriendlyStalker => [0.95, 0.85, 0.25, 0.95],
            Self::Trader => [0.98, 0.56, 0.16, 0.98],
            Self::QuestTarget => [0.96, 0.24, 0.22, 1.0],
        }
    }

    pub fn size_px(self) -> f32 {
        match self {
            Self::Player | Self::CoopPeer => 10.0,
            Self::QuestTarget => 11.0,
            Self::Trader => 8.5,
            Self::FriendlyStalker => 7.0,
        }
    }
}

/// Projected entity / marker blip rendered on the PDA map canvas.
#[derive(Debug, Clone, PartialEq)]
pub struct MapBlip {
    pub id: u16,
    pub label: String,
    pub kind: MapBlipKind,
    pub world_pos: [f32; 3],
    pub yaw: f32,
    pub uv: [f32; 2],
}

/// Story quest item mirrored from `ScopStoryQuest`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestTrackerEntry {
    pub task_id: String,
    pub title: String,
    pub description: String,
    pub task_state: u32,
    pub priority: u32,
    pub map_hint: String,
    pub map_location: String,
    pub map_object_id: u16,
    pub is_active: bool,
}

/// Screen-space geometry for PDA tabs, map area, and quest sidebar.
#[derive(Debug, Clone)]
pub struct PdaLayout {
    pub frame_rect: [f32; 4],
    pub tab_rects: [(PdaTab, [f32; 4]); 3],
    pub close_rect: [f32; 4],
    pub map_rect: [f32; 4],
    pub sidebar_rect: [f32; 4],
    pub quest_row_rects: Vec<[f32; 4]>,
}

impl PdaLayout {
    pub fn compute(width: u32, height: u32) -> Self {
        let fw = width.max(320) as f32;
        let fh = height.max(240) as f32;

        let frame_w = (fw * 0.84).clamp(300.0, fw - 36.0);
        let frame_h = (fh * 0.82).clamp(220.0, fh - 36.0);
        let fx = (fw - frame_w) * 0.5;
        let fy = (fh - frame_h) * 0.5;
        let frame_rect = [fx, fy, frame_w, frame_h];

        let tab_w = 104.0;
        let tab_h = 26.0;
        let tab_rects = [
            (PdaTab::Map, [fx + 16.0, fy + 8.0, tab_w, tab_h]),
            (PdaTab::Quests, [fx + 16.0 + tab_w + 8.0, fy + 8.0, tab_w, tab_h]),
            (
                PdaTab::Contacts,
                [fx + 16.0 + (tab_w + 8.0) * 2.0, fy + 8.0, tab_w, tab_h],
            ),
        ];
        let close_rect = [fx + frame_w - 38.0, fy + 8.0, 26.0, 24.0];

        let sidebar_w = (frame_w * 0.31).clamp(180.0, 320.0);
        let content_y = fy + 44.0;
        let content_h = (frame_h - 58.0).max(80.0);
        let map_w = (frame_w - sidebar_w - 42.0).max(120.0);

        let map_rect = [fx + 16.0, content_y, map_w, content_h];
        let sidebar_rect = [fx + 16.0 + map_w + 12.0, content_y, sidebar_w, content_h];
        let row_h = 34.0;
        let max_rows = ((content_h - 32.0) / (row_h + 6.0)).floor().max(1.0) as usize;
        let quest_row_rects = (0..max_rows.min(12))
            .map(|i| {
                [
                    sidebar_rect[0] + 8.0,
                    sidebar_rect[1] + 28.0 + (i as f32) * (row_h + 6.0),
                    sidebar_rect[2] - 16.0,
                    row_h,
                ]
            })
            .collect();

        Self {
            frame_rect,
            tab_rects,
            close_rect,
            map_rect,
            sidebar_rect,
            quest_row_rects,
        }
    }
}

/// Interactive PDA Screen Manager.
#[derive(Debug, Clone)]
pub struct PdaManager {
    pub open: bool,
    pub active_tab: PdaTab,
    pub world_maps: Option<WorldMapsConfig>,
    pub map_zoom: f32,
    pub map_pan: [f32; 2],
    pub blips: Vec<MapBlip>,
    pub quests: Vec<QuestTrackerEntry>,
    pub active_quest_id: String,
    pub fallback_bounds: [f32; 4],
    vfs_maps_attempted: bool,
    cached_map_texture: Option<(String, DescriptorIndex)>,
}

impl Default for PdaManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PdaManager {
    pub fn new() -> Self {
        Self {
            open: false,
            active_tab: PdaTab::Map,
            world_maps: None,
            map_zoom: 1.0,
            map_pan: [0.0, 0.0],
            blips: Vec::new(),
            quests: Vec::new(),
            active_quest_id: String::new(),
            fallback_bounds: [-150.0, -150.0, 150.0, 150.0],
            vfs_maps_attempted: false,
            cached_map_texture: None,
        }
    }

    pub fn toggle(&mut self, audio: Option<&mut AudioManager>) -> bool {
        self.set_open(!self.open, audio);
        self.open
    }

    pub fn set_open(&mut self, open: bool, audio: Option<&mut AudioManager>) {
        if self.open == open {
            return;
        }
        self.open = open;
        if let Some(aud) = audio {
            if open {
                aud.play_ui_open();
            } else {
                aud.play_ui_close();
            }
        }
    }

    /// Stream `$game_configs$/game_maps_single.ltx` + level minimap overrides from VFS once.
    pub fn ensure_world_maps_loaded(&mut self, vfs: &VirtualFileSystem) {
        if self.vfs_maps_attempted {
            return;
        }
        self.vfs_maps_attempted = true;
        if let Ok(cfg) = load_world_maps_from_vfs(vfs, &[], true) {
            self.world_maps = Some(cfg);
        }
    }

    /// Reset VFS map lookup cache (e.g. when switching worlds or mounting archives dynamically).
    pub fn invalidate_map_cache(&mut self) {
        self.vfs_maps_attempted = false;
        self.world_maps = None;
        self.cached_map_texture = None;
    }

    /// Project a 3D world position onto normalized `[u, v]` in `[0.0, 1.0]` using `WorldMapsConfig`
    /// when loaded for `level_name`, or `fallback_bounds` `[min_x, min_z, max_x, max_z]`.
    pub fn project_world_to_uv(&self, level_name: &str, world_pos: [f32; 3]) -> [f32; 2] {
        if let Some(maps) = &self.world_maps
            && let Some(uv) = maps.world_xz_to_level_uv(level_name, [world_pos[0], world_pos[2]])
        {
            return [uv[0].clamp(0.0, 1.0), uv[1].clamp(0.0, 1.0)];
        }
        let [bx1, bz1, bx2, bz2] = self.fallback_bounds;
        let w = (bx2 - bx1).max(1.0);
        let h = (bz2 - bz1).max(1.0);
        let u = ((world_pos[0] - bx1) / w).clamp(0.02, 0.98);
        let v = ((bz2 - world_pos[2]) / h).clamp(0.02, 0.98);
        [u, v]
    }

    /// Synchronize radar blips (player, co-op peers, stalkers, traders, quest markers) and
    /// [`ScopStoryQuest`] items from live `AlifeManager` & camera state.
    pub fn sync_from_world_state(
        &mut self,
        level_name: &str,
        cam_pos: [f32; 3],
        cam_yaw: f32,
        alife: Option<&AlifeManager>,
    ) {
        self.blips.clear();
        self.quests.clear();

        // Update dynamic fallback bounds so synthetic / minimal test scenes frame all blips cleanly
        let mut min_x = cam_pos[0] - 80.0;
        let mut max_x = cam_pos[0] + 80.0;
        let mut min_z = cam_pos[2] - 80.0;
        let mut max_z = cam_pos[2] + 80.0;
        if let Some(mgr) = alife {
            for peer in mgr.remote_peers.values() {
                min_x = min_x.min(peer.position[0] - 25.0);
                max_x = max_x.max(peer.position[0] + 25.0);
                min_z = min_z.min(peer.position[2] - 25.0);
                max_z = max_z.max(peer.position[2] + 25.0);
            }
            for ent in mgr.entities.values() {
                if ent.online {
                    min_x = min_x.min(ent.position[0] - 25.0);
                    max_x = max_x.max(ent.position[0] + 25.0);
                    min_z = min_z.min(ent.position[2] - 25.0);
                    max_z = max_z.max(ent.position[2] + 25.0);
                }
            }
        }
        self.fallback_bounds = [min_x, min_z, max_x, max_z];

        // 1. Local Player blip
        self.blips.push(MapBlip {
            id: 0,
            label: "You".to_string(),
            kind: MapBlipKind::Player,
            world_pos: cam_pos,
            yaw: cam_yaw,
            uv: self.project_world_to_uv(level_name, cam_pos),
        });

        let Some(mgr) = alife else {
            return;
        };

        // 2. Co-op peer blips
        for peer in mgr.remote_peers.values() {
            self.blips.push(MapBlip {
                id: peer.entity_id,
                label: if peer.nickname.is_empty() {
                    format!("Peer {}", peer.slot)
                } else {
                    peer.nickname.clone()
                },
                kind: MapBlipKind::CoopPeer,
                world_pos: peer.position,
                yaw: peer.yaw,
                uv: self.project_world_to_uv(level_name, peer.position),
            });
        }

        // 3. Nearby friendly Stalkers & Traders
        for ent in mgr.entities.values() {
            if !ent.online {
                continue;
            }
            match ent.role {
                EntityRole::Trader => {
                    self.blips.push(MapBlip {
                        id: ent.id,
                        label: if ent.name.is_empty() {
                            ent.section.clone()
                        } else {
                            ent.name.clone()
                        },
                        kind: MapBlipKind::Trader,
                        world_pos: ent.position,
                        yaw: ent.angle[1],
                        uv: self.project_world_to_uv(level_name, ent.position),
                    });
                }
                EntityRole::Stalker => {
                    self.blips.push(MapBlip {
                        id: ent.id,
                        label: if ent.name.is_empty() {
                            ent.section.clone()
                        } else {
                            ent.name.clone()
                        },
                        kind: MapBlipKind::FriendlyStalker,
                        world_pos: ent.position,
                        yaw: ent.angle[1],
                        uv: self.project_world_to_uv(level_name, ent.position),
                    });
                }
                _ => {}
            }
        }

        // 4. Story Quests from ScopGraphAndQuests + Quest target blips
        self.active_quest_id = mgr.graph_and_quests.active_quest_id.clone();
        for sq in &mgr.graph_and_quests.story_quests {
            let t = &sq.task;
            let is_active = !self.active_quest_id.is_empty() && t.task_id == self.active_quest_id;
            self.quests.push(QuestTrackerEntry {
                task_id: t.task_id.clone(),
                title: if t.title.is_empty() {
                    t.task_id.clone()
                } else {
                    t.title.clone()
                },
                description: t.description.clone(),
                task_state: t.task_state,
                priority: t.priority,
                map_hint: t.map_hint.clone(),
                map_location: t.map_location.clone(),
                map_object_id: t.map_object_id,
                is_active,
            });

            if t.map_object_id != INVALID_OBJECT_ID
                && let Some(target_ent) = mgr.entities.get(&t.map_object_id)
            {
                self.blips.push(MapBlip {
                    id: target_ent.id,
                    label: if t.map_hint.is_empty() {
                        t.title.clone()
                    } else {
                        t.map_hint.clone()
                    },
                    kind: MapBlipKind::QuestTarget,
                    world_pos: target_ent.position,
                    yaw: 0.0,
                    uv: self.project_world_to_uv(level_name, target_ent.position),
                });
            }
        }
        self.quests.sort_by(|a, b| {
            b.is_active
                .cmp(&a.is_active)
                .then_with(|| a.priority.cmp(&b.priority))
                .then_with(|| a.task_id.cmp(&b.task_id))
        });
    }

    /// Activate / focus a tracked [`ScopStoryQuest`] by `task_id`.
    pub fn select_active_quest(
        &mut self,
        task_id: &str,
        alife: Option<&mut AlifeManager>,
        audio: Option<&mut AudioManager>,
    ) -> bool {
        if !self.quests.iter().any(|q| q.task_id == task_id) {
            return false;
        }
        self.active_quest_id = task_id.to_string();
        for q in &mut self.quests {
            q.is_active = q.task_id == task_id;
        }
        if let Some(mgr) = alife {
            mgr.graph_and_quests.active_quest_id = task_id.to_string();
            mgr.graph_and_quests.sync_registries();
        }
        if let Some(aud) = audio {
            aud.play_ui_click("interface\\pda_task");
        }
        true
    }

    /// Process pointer clicks on PDA tabs, quest tracker rows, close button, and wheel zoom.
    pub fn handle_input(
        &mut self,
        input: &EngineInput,
        viewport_w: u32,
        viewport_h: u32,
        mut alife: Option<&mut AlifeManager>,
        mut audio: Option<&mut AudioManager>,
    ) {
        if !self.open {
            return;
        }
        let wheel = input.wheel_delta();
        if wheel.abs() > 1e-4 {
            self.map_zoom = (self.map_zoom + wheel * 0.15).clamp(1.0, 4.0);
        }

        if !input.is_mouse_just_pressed(MouseButton::Left) {
            return;
        }
        let (mx, my) = input.mouse_position();
        let layout = PdaLayout::compute(viewport_w, viewport_h);

        if point_in_rect(mx, my, layout.close_rect) {
            self.set_open(false, audio);
            return;
        }
        for (tab, rect) in layout.tab_rects {
            if point_in_rect(mx, my, rect) {
                self.active_tab = tab;
                if let Some(aud) = audio.as_deref_mut() {
                    aud.play_ui_click("interface\\pda_news");
                }
                return;
            }
        }
        for (idx, rect) in layout.quest_row_rects.iter().enumerate() {
            if idx < self.quests.len() && point_in_rect(mx, my, *rect) {
                let tid = self.quests[idx].task_id.clone();
                let _ = self.select_active_quest(&tid, alife.as_deref_mut(), audio.as_deref_mut());
                return;
            }
        }
    }

    fn resolve_minimap_texture(
        &mut self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        level_name: &str,
    ) -> Option<DescriptorIndex> {
        let lvl_key = level_name.to_ascii_lowercase();
        if let Some((cached_lvl, idx)) = &self.cached_map_texture
            && cached_lvl == &lvl_key
        {
            return Some(*idx);
        }
        let vpath = self
            .world_maps
            .as_ref()
            .and_then(|m| m.get_level(&lvl_key))
            .map(|l| l.texture_dds_vpath.clone())?;
        if let Some(existing) = renderer.get_texture(&vpath) {
            self.cached_map_texture = Some((lvl_key, existing.descriptor_index));
            return Some(existing.descriptor_index);
        }
        let vfs = vfs?;
        let bytes = vfs.read_bytes(&vpath).ok()?;
        let idx = renderer.load_texture_from_dds(&vpath, &bytes).ok()?;
        self.cached_map_texture = Some((lvl_key, idx));
        Some(idx)
    }

    /// Render the interactive PDA frame, level minimap (or tactical fallback), radar blips,
    /// and quest / contacts sidebar via `Dx12Renderer` `UiCanvas`.
    pub fn render_ui(
        &mut self,
        renderer: &mut Dx12Renderer,
        vfs: Option<&VirtualFileSystem>,
        width: u32,
        height: u32,
        level_name: &str,
        font: Option<&UiFontResource>,
    ) -> u32 {
        if !self.open {
            return 0;
        }
        let mut drawn = 0u32;
        let fw = width.max(64) as f32;
        let fh = height.max(64) as f32;
        let layout = PdaLayout::compute(width, height);

        // Backdrop + Rugged PDA Outer Bezel
        renderer.ui_fill_rect([0.0, 0.0, fw, fh], [0.02, 0.03, 0.04, 0.50]);
        renderer.ui_fill_rect(layout.frame_rect, [0.09, 0.11, 0.12, 0.96]);
        renderer.ui_fill_rect(
            [layout.frame_rect[0], layout.frame_rect[1], layout.frame_rect[2], 38.0],
            [0.14, 0.17, 0.19, 0.98],
        );
        renderer.ui_fill_rect(layout.close_rect, [0.70, 0.22, 0.22, 0.95]);
        drawn += 4;

        // Top Navigation Tabs
        for (tab, rect) in layout.tab_rects {
            let active = tab == self.active_tab;
            let col = if active {
                [0.24, 0.44, 0.34, 0.98]
            } else {
                [0.15, 0.18, 0.20, 0.92]
            };
            renderer.ui_fill_rect(rect, col);
            drawn += 1;
            if let Some(f) = font {
                let title = match tab {
                    PdaTab::Map => "MAP",
                    PdaTab::Quests => "QUESTS",
                    PdaTab::Contacts => "CONTACTS",
                };
                drawn += f.draw_text(renderer, title, [rect[0] + 10.0, rect[1] + 5.0], 14.0, [0.95, 0.95, 0.90, 1.0]);
            }
        }

        // Main Map Canvas (Left Area)
        renderer.ui_fill_rect(layout.map_rect, [0.05, 0.08, 0.07, 0.95]);
        drawn += 1;

        let uv_span = 1.0 / self.map_zoom.clamp(1.0, 4.0);
        let center_uv = [
            (0.5 + self.map_pan[0]).clamp(uv_span * 0.5, 1.0 - uv_span * 0.5),
            (0.5 + self.map_pan[1]).clamp(uv_span * 0.5, 1.0 - uv_span * 0.5),
        ];
        let view_u0 = center_uv[0] - uv_span * 0.5;
        let view_v0 = center_uv[1] - uv_span * 0.5;
        let view_u1 = view_u0 + uv_span;
        let view_v1 = view_v0 + uv_span;

        if let Some(tex_idx) = self.resolve_minimap_texture(renderer, vfs, level_name) {
            renderer.ui_draw_textured_rect(
                layout.map_rect,
                [view_u0, view_v0, view_u1, view_v1],
                tex_idx,
                [1.0, 1.0, 1.0, 0.96],
            );
            drawn += 1;
        } else {
            // Clean tactical grid lines when VFS minimap .dds is not mounted
            let grid_steps = 8;
            for i in 1..grid_steps {
                let frac = (i as f32) / (grid_steps as f32);
                renderer.ui_fill_rect(
                    [
                        layout.map_rect[0] + layout.map_rect[2] * frac,
                        layout.map_rect[1],
                        1.0,
                        layout.map_rect[3],
                    ],
                    [0.14, 0.28, 0.22, 0.55],
                );
                renderer.ui_fill_rect(
                    [
                        layout.map_rect[0],
                        layout.map_rect[1] + layout.map_rect[3] * frac,
                        layout.map_rect[2],
                        1.0,
                    ],
                    [0.14, 0.28, 0.22, 0.55],
                );
                drawn += 2;
            }
        }

        // Radar Blips (Player, Co-Op Peers, Stalkers, Traders, Quest Objectives)
        for blip in &self.blips {
            let rel_u = (blip.uv[0] - view_u0) / uv_span;
            let rel_v = (blip.uv[1] - view_v0) / uv_span;
            if !(0.0..=1.0).contains(&rel_u) || !(0.0..=1.0).contains(&rel_v) {
                continue;
            }
            let bx = layout.map_rect[0] + rel_u * layout.map_rect[2];
            let by = layout.map_rect[1] + rel_v * layout.map_rect[3];
            let sz = blip.kind.size_px();
            renderer.ui_fill_rect(
                [bx - sz * 0.5, by - sz * 0.5, sz, sz],
                blip.kind.color_rgba(),
            );
            drawn += 1;

            // Player heading indicator notch
            if blip.kind == MapBlipKind::Player {
                let hx = bx + blip.yaw.sin() * 9.0;
                let hy = by - blip.yaw.cos() * 9.0;
                renderer.ui_fill_rect([hx - 2.0, hy - 2.0, 4.0, 4.0], [0.95, 1.0, 0.85, 1.0]);
                drawn += 1;
            }
            if let Some(f) = font {
                drawn += f.draw_text(
                    renderer,
                    &blip.label,
                    [bx + sz * 0.6, by - 6.0],
                    12.0,
                    blip.kind.color_rgba(),
                );
            }
        }

        // Right Sidebar: Quests or Co-op / Stalker Contacts
        renderer.ui_fill_rect(layout.sidebar_rect, [0.07, 0.08, 0.10, 0.94]);
        renderer.ui_fill_rect(
            [layout.sidebar_rect[0], layout.sidebar_rect[1], layout.sidebar_rect[2], 24.0],
            [0.13, 0.17, 0.18, 0.98],
        );
        drawn += 2;

        if self.active_tab == PdaTab::Contacts {
            let contacts: Vec<&MapBlip> = self
                .blips
                .iter()
                .filter(|b| b.kind != MapBlipKind::Player && b.kind != MapBlipKind::QuestTarget)
                .collect();
            for (idx, rect) in layout.quest_row_rects.iter().enumerate() {
                if idx >= contacts.len() {
                    break;
                }
                let c = contacts[idx];
                renderer.ui_fill_rect(*rect, [0.12, 0.15, 0.17, 0.92]);
                renderer.ui_fill_rect([rect[0] + 4.0, rect[1] + 8.0, 8.0, 8.0], c.kind.color_rgba());
                drawn += 2;
                if let Some(f) = font {
                    drawn += f.draw_text(renderer, &c.label, [rect[0] + 18.0, rect[1] + 8.0], 13.0, [0.94, 0.94, 0.90, 1.0]);
                }
            }
        } else {
            for (idx, rect) in layout.quest_row_rects.iter().enumerate() {
                if idx >= self.quests.len() {
                    break;
                }
                let q = &self.quests[idx];
                let bg = if q.is_active {
                    [0.22, 0.34, 0.24, 0.96]
                } else {
                    [0.12, 0.14, 0.16, 0.90]
                };
                renderer.ui_fill_rect(*rect, bg);
                renderer.ui_fill_rect(
                    [rect[0], rect[1], 4.0, rect[3]],
                    if q.is_active {
                        [0.95, 0.80, 0.22, 1.0]
                    } else {
                        [0.42, 0.54, 0.48, 0.85]
                    },
                );
                drawn += 2;
                if let Some(f) = font {
                    drawn += f.draw_text(renderer, &q.title, [rect[0] + 10.0, rect[1] + 6.0], 13.0, [0.95, 0.94, 0.88, 1.0]);
                }
            }
        }

        drawn
    }
}

#[inline]
fn point_in_rect(px: f32, py: f32, rect: [f32; 4]) -> bool {
    px >= rect[0] && px <= rect[0] + rect[2] && py >= rect[1] && py <= rect[1] + rect[3]
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use xray_formats::savegames::{SavedGameTask, SavedTaskScriptHelper};
    use xray_formats::scoc::ScocFile;
    use xray_formats::scop::ScopStoryQuest;

    fn sample_task(task_id: &str, title: &str, map_hint: &str) -> SavedGameTask {
        SavedGameTask {
            task_id: task_id.to_string(),
            task_state: 1,
            task_type: 0,
            receive_time: 0,
            finish_time: 0,
            time_to_complete: 0,
            timer_finish: 0,
            title: title.to_string(),
            description: String::new(),
            script_helper: SavedTaskScriptHelper::default(),
            icon_texture_name: String::new(),
            map_hint: map_hint.to_string(),
            map_location: String::new(),
            map_object_id: INVALID_OBJECT_ID,
            priority: 1,
        }
    }

    #[test]
    fn test_pda_minimap_projection_peers_and_quests() {
        let maps = WorldMapsConfig::parse_ltx_str(
            r#"
[level_maps_single]
l01_escape =

[l01_escape]
texture = map\map_escape
bound_rect = -200.0, -200.0, 200.0, 200.0
"#,
        )
        .unwrap();

        let mut alife = AlifeManager::default();
        alife.graph_and_quests.story_quests = vec![
            ScopStoryQuest {
                owner_id: 0,
                task: sample_task("esc_meet_wolf", "Meet Wolf in Rookie Village", "Wolf"),
            },
            ScopStoryQuest {
                owner_id: 0,
                task: sample_task("esc_nimble", "Rescue Nimble", ""),
            },
        ];
        alife.graph_and_quests.active_quest_id = "esc_meet_wolf".to_string();

        // Attach one co-op peer avatar at (100, 0, -100)
        let scoc = ScocFile::default();
        let peer_id = alife.attach_peer_scoc(1, "Falcon", 777, &scoc);
        alife.remote_peers.get_mut(&peer_id).unwrap().position = [100.0, 0.0, -100.0];

        let mut pda = PdaManager::new();
        pda.world_maps = Some(maps);

        let mut audio = AudioManager::new_headless();
        assert!(pda.toggle(Some(&mut audio)));
        assert!(pda.open);
        assert_eq!(audio.stats().ui_sounds_triggered, 1);

        pda.sync_from_world_state("l01_escape", [0.0, 1.65, 0.0], 0.0, Some(&alife));
        assert_eq!(pda.quests.len(), 2);
        assert!(pda.quests[0].is_active);
        assert_eq!(pda.quests[0].task_id, "esc_meet_wolf");

        let player_blip = pda.blips.iter().find(|b| b.kind == MapBlipKind::Player).unwrap();
        assert!((player_blip.uv[0] - 0.5).abs() < 1e-4);
        assert!((player_blip.uv[1] - 0.5).abs() < 1e-4);

        let peer_blip = pda.blips.iter().find(|b| b.kind == MapBlipKind::CoopPeer).unwrap();
        assert_eq!(peer_blip.label, "Falcon");
        assert!((peer_blip.uv[0] - 0.75).abs() < 1e-4);
        assert!((peer_blip.uv[1] - 0.75).abs() < 1e-4);

        assert!(pda.select_active_quest("esc_nimble", Some(&mut alife), Some(&mut audio)));
        assert_eq!(alife.graph_and_quests.active_quest_id, "esc_nimble");
        assert_eq!(audio.stats().ui_sounds_triggered, 2);
    }
}
