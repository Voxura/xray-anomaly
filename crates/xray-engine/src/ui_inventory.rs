//! Interactive Stalker 2D Grid Inventory, Equipment Paperdoll & NPC/Trader Dialogue (`xray-engine`).
//!
//! Coordinates:
//! - Loading & exporting `ScocFile` / `ScocPeerState` (`ScocInventoryItem`, `item_remaining_uses`)
//!   and `ScopSaveFile` player inventory & `ScopActorStats`
//! - 10x12 2D spatial grid layout, drag-and-drop item placement, equipment slots (Primary Weapon,
//!   Secondary Weapon, Pistol, Armor, Helmet, Backpack, Artifacts 0..5, Quick Slots 0..4)
//! - Weapon equip/unequip synchronization with [`HudWeaponController`]
//! - Consumable usage (medkit, bandage, energy drink, antirad) with `ScocInventoryItem::remaining_uses`
//!   tracking, health/stamina/bleeding/radiation updates, and `AudioManager` item SFX
//! - `'E'` (`Action::Use`) ray-cone proximity checks for friendly Stalker/Trader dialogue prompts
//!   and world item pickups
//! - Direct3D 12 `UiCanvas` immediate-mode rendering for the inventory modal, always-on HUD meters
//!   (health, stamina, bleeding/radiation, active weapon ammo/quick-slots), and dialogue overlay.

use std::collections::BTreeMap;

use crate::alife_manager::{AlifeManager, EntityRole};
use crate::hud_weapon::{HudWeaponController, WeaponFireMode, WeaponSpec};
use crate::input::{EngineInput, MouseButton};
use xray_audio::AudioManager;
use xray_formats::fonts::{FontAlignment, FontAtlas, FontDrawParams};
use xray_formats::spawns::INVALID_OBJECT_ID;
use xray_formats::scoc::{ScocFile, ScocInventoryItem, ScocPeerState};
use xray_formats::scop::{ScopActorStats, ScopAlifeKind, ScopSaveFile};
use xray_render_dx12::bindless::DescriptorIndex;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::ui_canvas::GlyphQuad;

/// Default backpack grid width in cells (classic X-Ray 10-column grid).
pub const DEFAULT_GRID_COLS: u8 = 10;
/// Default backpack grid height in cells.
pub const DEFAULT_GRID_ROWS: u8 = 12;
/// Number of artifact belt slots.
pub const ARTIFACT_SLOT_COUNT: usize = 5;
/// Number of quick-access consumable slots.
pub const QUICK_SLOT_COUNT: usize = 4;

/// Optional VFS-backed bitmap font atlas for crisp `UiCanvas::draw_glyph_quads` text rendering.
#[derive(Debug, Clone)]
pub struct UiFontResource {
    pub atlas: FontAtlas,
    pub texture_index: DescriptorIndex,
    pub texture_size: [u32; 2],
}

impl UiFontResource {
    /// Draw a single line of text at `[x, y]` if `glyphs` resolve, returning quad count queued.
    pub fn draw_text(
        &self,
        renderer: &mut Dx12Renderer,
        text: &str,
        pos_xy: [f32; 2],
        line_height_px: f32,
        color: [f32; 4],
    ) -> u32 {
        if text.is_empty() {
            return 0;
        }
        let params = FontDrawParams {
            current_height: line_height_px,
            align: FontAlignment::Left,
            ..Default::default()
        };
        let font_quads = self.atlas.build_string_quads(self.texture_size, pos_xy, text, &params);
        if font_quads.is_empty() {
            return 0;
        }
        let glyphs: Vec<GlyphQuad> = font_quads
            .into_iter()
            .map(|q| GlyphQuad {
                screen_rect: [
                    q.min_xy[0],
                    q.min_xy[1],
                    (q.max_xy[0] - q.min_xy[0]).max(1.0),
                    (q.max_xy[1] - q.min_xy[1]).max(1.0),
                ],
                uv_rect: [q.uv_min[0], q.uv_min[1], q.uv_max[0], q.uv_max[1]],
            })
            .collect();
        let count = glyphs.len() as u32;
        renderer.ui_draw_glyph_quads(&glyphs, self.texture_index, color);
        count
    }
}

/// Logical equipment slots surrounding the actor's 2D backpack grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EquipmentSlot {
    PrimaryWeapon,
    SecondaryWeapon,
    Pistol,
    Armor,
    Helmet,
    Backpack,
    Artifact(usize),
    QuickSlot(usize),
}

impl EquipmentSlot {
    /// All slots in deterministic display order.
    pub fn all_slots() -> [Self; 15] {
        [
            Self::PrimaryWeapon,
            Self::SecondaryWeapon,
            Self::Pistol,
            Self::Armor,
            Self::Helmet,
            Self::Backpack,
            Self::Artifact(0),
            Self::Artifact(1),
            Self::Artifact(2),
            Self::Artifact(3),
            Self::Artifact(4),
            Self::QuickSlot(0),
            Self::QuickSlot(1),
            Self::QuickSlot(2),
            Self::QuickSlot(3),
        ]
    }

    /// Corresponding X-Ray `HUD` slot index (`1..=6`) if this is a weapon slot.
    pub fn hud_slot_number(self) -> Option<u8> {
        match self {
            Self::Pistol => Some(2),
            Self::PrimaryWeapon => Some(3),
            Self::SecondaryWeapon => Some(4),
            _ => None,
        }
    }

    /// Human-readable label for UI chrome.
    pub fn label(self) -> &'static str {
        match self {
            Self::PrimaryWeapon => "PRIMARY",
            Self::SecondaryWeapon => "SECONDARY",
            Self::Pistol => "PISTOL",
            Self::Armor => "SUIT",
            Self::Helmet => "HELMET",
            Self::Backpack => "PACK",
            Self::Artifact(_) => "ART",
            Self::QuickSlot(_) => "QUICK",
        }
    }
}

/// Broad classification controlling slot compatibility, item color coding, and use behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemCategory {
    Weapon,
    Outfit,
    Helmet,
    Backpack,
    Artifact,
    Medical,
    Consumable,
    Ammo,
    Misc,
}

impl ItemCategory {
    pub fn accent_color(self) -> [f32; 4] {
        match self {
            Self::Weapon => [0.58, 0.56, 0.48, 0.90],
            Self::Outfit => [0.36, 0.48, 0.42, 0.90],
            Self::Helmet => [0.38, 0.46, 0.52, 0.90],
            Self::Backpack => [0.44, 0.36, 0.28, 0.90],
            Self::Artifact => [0.22, 0.68, 0.65, 0.92],
            Self::Medical => [0.76, 0.24, 0.24, 0.92],
            Self::Consumable => [0.32, 0.62, 0.34, 0.90],
            Self::Ammo => [0.68, 0.56, 0.22, 0.90],
            Self::Misc => [0.42, 0.44, 0.48, 0.88],
        }
    }
}

/// Classified metadata for a known X-Ray item section (`wpn_ak74`, `medkit`, `bandage`, etc.).
#[derive(Debug, Clone)]
struct SectionItemMeta {
    category: ItemCategory,
    grid_size: (u8, u8),
    weight_kg: f32,
    display_name: String,
    default_uses: Option<i64>,
    preferred_slot: Option<EquipmentSlot>,
}

fn classify_section_meta(section: &str) -> SectionItemMeta {
    let lower = section.trim().to_ascii_lowercase();

    if lower.starts_with("wpn_pm")
        || lower.starts_with("wpn_pb")
        || lower.starts_with("wpn_fort")
        || lower.starts_with("wpn_usp")
        || lower.starts_with("wpn_beretta")
        || lower.starts_with("wpn_colt")
        || lower.starts_with("wpn_hpsa")
        || lower.starts_with("wpn_walther")
        || lower.starts_with("wpn_sig220")
        || lower.starts_with("wpn_tt")
    {
        return SectionItemMeta {
            category: ItemCategory::Weapon,
            grid_size: (2, 1),
            weight_kg: 0.85,
            display_name: prettify_section(&lower),
            default_uses: None,
            preferred_slot: Some(EquipmentSlot::Pistol),
        };
    }
    if lower.starts_with("wpn_") {
        let size = if lower.contains("mp5") || lower.contains("aks74u") || lower.contains("klin") {
            (3, 2)
        } else {
            (4, 2)
        };
        return SectionItemMeta {
            category: ItemCategory::Weapon,
            grid_size: size,
            weight_kg: if lower.contains("svd") { 4.3 } else { 3.3 },
            display_name: prettify_section(&lower),
            default_uses: None,
            preferred_slot: Some(EquipmentSlot::PrimaryWeapon),
        };
    }
    if lower.contains("outfit") || lower.contains("exo") || lower.contains("suit") || lower.starts_with("stalker_outfit") {
        return SectionItemMeta {
            category: ItemCategory::Outfit,
            grid_size: (2, 3),
            weight_kg: 5.0,
            display_name: prettify_section(&lower),
            default_uses: None,
            preferred_slot: Some(EquipmentSlot::Armor),
        };
    }
    if lower.starts_with("helm") {
        return SectionItemMeta {
            category: ItemCategory::Helmet,
            grid_size: (2, 2),
            weight_kg: 1.4,
            display_name: prettify_section(&lower),
            default_uses: None,
            preferred_slot: Some(EquipmentSlot::Helmet),
        };
    }
    if lower.starts_with("equ_small_pack")
        || lower.starts_with("equ_military_pack")
        || lower.starts_with("equ_tourist_pack")
        || lower.contains("backpack")
    {
        return SectionItemMeta {
            category: ItemCategory::Backpack,
            grid_size: (2, 2),
            weight_kg: 1.2,
            display_name: prettify_section(&lower),
            default_uses: None,
            preferred_slot: Some(EquipmentSlot::Backpack),
        };
    }
    if lower.starts_with("af_") {
        return SectionItemMeta {
            category: ItemCategory::Artifact,
            grid_size: (1, 1),
            weight_kg: 0.5,
            display_name: prettify_section(&lower),
            default_uses: None,
            preferred_slot: None,
        };
    }
    if lower.contains("medkit") || lower.contains("bandage") || lower.contains("antirad") || lower.contains("stim") {
        return SectionItemMeta {
            category: ItemCategory::Medical,
            grid_size: (1, 1),
            weight_kg: 0.3,
            display_name: prettify_section(&lower),
            default_uses: Some(1),
            preferred_slot: None,
        };
    }
    if lower.contains("drink")
        || lower.contains("water")
        || lower.contains("vodka")
        || lower.contains("bread")
        || lower.contains("kolbasa")
        || lower.contains("conserva")
    {
        return SectionItemMeta {
            category: ItemCategory::Consumable,
            grid_size: (1, 1),
            weight_kg: 0.4,
            display_name: prettify_section(&lower),
            default_uses: Some(1),
            preferred_slot: None,
        };
    }
    if lower.starts_with("ammo_") {
        return SectionItemMeta {
            category: ItemCategory::Ammo,
            grid_size: (2, 1),
            weight_kg: 0.45,
            display_name: prettify_section(&lower),
            default_uses: None,
            preferred_slot: None,
        };
    }
    SectionItemMeta {
        category: ItemCategory::Misc,
        grid_size: (1, 1),
        weight_kg: 0.25,
        display_name: prettify_section(&lower),
        default_uses: None,
        preferred_slot: None,
    }
}

fn prettify_section(section: &str) -> String {
    let trimmed = section
        .strip_prefix("wpn_")
        .or_else(|| section.strip_prefix("af_"))
        .or_else(|| section.strip_prefix("ammo_"))
        .unwrap_or(section);
    trimmed
        .split('_')
        .filter(|s| !s.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Live item entry managed inside [`InventoryManager`].
#[derive(Debug, Clone, PartialEq)]
pub struct InventoryItem {
    pub uid: u64,
    pub source_id: Option<i64>,
    pub section: String,
    pub display_name: String,
    pub category: ItemCategory,
    pub grid_size: (u8, u8),
    pub grid_pos: Option<(u8, u8)>,
    pub equipped_slot: Option<EquipmentSlot>,
    pub condition: f32,
    pub remaining_uses: Option<i64>,
    pub quantity: u32,
    pub ammo_elapsed: Option<i64>,
    pub ammo_type: Option<i64>,
    pub ammo_section: Option<String>,
    pub weight_kg: f32,
    pub source_scoc: Option<ScocInventoryItem>,
}

impl InventoryItem {
    pub fn from_section(uid: u64, section: impl Into<String>) -> Self {
        let section_str = section.into();
        let meta = classify_section_meta(&section_str);
        Self {
            uid,
            source_id: Some(uid as i64),
            section: section_str,
            display_name: meta.display_name,
            category: meta.category,
            grid_size: meta.grid_size,
            grid_pos: None,
            equipped_slot: None,
            condition: 1.0,
            remaining_uses: meta.default_uses,
            quantity: 1,
            ammo_elapsed: None,
            ammo_type: None,
            ammo_section: None,
            weight_kg: meta.weight_kg,
            source_scoc: None,
        }
    }

    pub fn from_scoc_item(
        uid: u64,
        scoc: &ScocInventoryItem,
        item_uses_map: &BTreeMap<i64, i64>,
    ) -> Self {
        let meta = classify_section_meta(&scoc.section);
        let resolved_uses = scoc.remaining_uses.or_else(|| {
            scoc.id
                .and_then(|id| item_uses_map.get(&id).copied())
                .or(meta.default_uses)
        });
        let equipped_slot = scoc.slot.and_then(|s| match s {
            1 => Some(EquipmentSlot::Pistol),
            2 => Some(EquipmentSlot::PrimaryWeapon),
            3 => Some(EquipmentSlot::SecondaryWeapon),
            6 => Some(EquipmentSlot::Armor),
            11 => Some(EquipmentSlot::Helmet),
            12 => Some(EquipmentSlot::Backpack),
            _ => None,
        });
        Self {
            uid,
            source_id: scoc.id.or(Some(uid as i64)),
            section: scoc.section.clone(),
            display_name: meta.display_name,
            category: meta.category,
            grid_size: meta.grid_size,
            grid_pos: None,
            equipped_slot,
            condition: (scoc.condition as f32).clamp(0.0, 1.0),
            remaining_uses: resolved_uses,
            quantity: scoc.quantity.unwrap_or(1).max(1) as u32,
            ammo_elapsed: scoc.ammo_elapsed,
            ammo_type: scoc.ammo_type,
            ammo_section: scoc.ammo_section.clone(),
            weight_kg: meta.weight_kg,
            source_scoc: Some(scoc.clone()),
        }
    }

    pub fn to_scoc_item(&self) -> ScocInventoryItem {
        let mut out = self
            .source_scoc
            .clone()
            .unwrap_or_else(|| ScocInventoryItem::new(self.section.clone(), f64::from(self.condition)));
        out.id = self.source_id.or(Some(self.uid as i64));
        out.section = self.section.clone();
        out.condition = f64::from(self.condition.clamp(0.0, 1.0));
        out.remaining_uses = self.remaining_uses;
        out.quantity = Some(i64::from(self.quantity));
        out.ammo_elapsed = self.ammo_elapsed;
        out.ammo_type = self.ammo_type;
        out.ammo_section = self.ammo_section.clone();
        out.slot = self.equipped_slot.and_then(|slot| match slot {
            EquipmentSlot::Pistol => Some(1),
            EquipmentSlot::PrimaryWeapon => Some(2),
            EquipmentSlot::SecondaryWeapon => Some(3),
            EquipmentSlot::Armor => Some(6),
            EquipmentSlot::Helmet => Some(11),
            EquipmentSlot::Backpack => Some(12),
            _ => None,
        });
        out
    }
}

/// Active pointer drag state inside the 2D inventory UI.
#[derive(Debug, Clone, PartialEq)]
pub struct DragState {
    pub item_uid: u64,
    pub start_pointer: [f32; 2],
    pub current_pointer: [f32; 2],
    pub origin_cell: Option<(u8, u8)>,
    pub origin_slot: Option<EquipmentSlot>,
}

/// Stat deltas returned after activating a consumable item.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ItemUseEffect {
    pub health_delta: f32,
    pub stamina_delta: f32,
    pub bleeding_delta: f32,
    pub radiation_delta: f32,
    pub consumed: bool,
}

/// Proximity `'E'` look target discovered by [`InventoryManager::scan_use_target`].
#[derive(Debug, Clone, PartialEq)]
pub enum UseTarget {
    Dialogue {
        entity_id: u16,
        name: String,
        is_trader: bool,
        distance_m: f32,
    },
    Pickup {
        entity_id: u16,
        section: String,
        display_name: String,
        distance_m: f32,
    },
}

impl UseTarget {
    pub fn prompt_text(&self) -> String {
        match self {
            Self::Dialogue {
                name, is_trader, ..
            } => {
                if *is_trader {
                    format!("[E] Trade / Talk: {name}")
                } else {
                    format!("[E] Talk to {name}")
                }
            }
            Self::Pickup { display_name, .. } => format!("[E] Pick up {display_name}"),
        }
    }
}

/// Active NPC / Trader dialogue state opened via `Action::Use` (`E`).
#[derive(Debug, Clone, PartialEq)]
pub struct DialoguePromptState {
    pub speaker_id: u16,
    pub speaker_name: String,
    pub is_trader: bool,
    pub greeting: String,
    pub selected_option: usize,
    pub options: Vec<String>,
    pub status_line: String,
}

/// Cached screen-space layout rectangles `[x, y, w, h]` for hit testing & `UiCanvas` drawing.
#[derive(Debug, Clone)]
pub struct InventoryLayout {
    pub panel_rect: [f32; 4],
    pub grid_origin: [f32; 2],
    pub cell_size: f32,
    pub grid_cols: u8,
    pub grid_rows: u8,
    pub slot_rects: Vec<(EquipmentSlot, [f32; 4])>,
    pub use_button_rect: [f32; 4],
    pub close_button_rect: [f32; 4],
    pub dialogue_rect: [f32; 4],
    pub dialogue_option_rects: Vec<[f32; 4]>,
}

impl InventoryLayout {
    pub fn compute(width: u32, height: u32, cols: u8, rows: u8) -> Self {
        let fw = (width.max(320)) as f32;
        let fh = (height.max(240)) as f32;

        let cell_size = ((fw * 0.032).min(fh * 0.052)).clamp(20.0, 44.0);
        let grid_w = f32::from(cols) * cell_size;
        let grid_h = f32::from(rows) * cell_size;

        let panel_w = (fw * 0.84).clamp(grid_w + 300.0, fw - 32.0);
        let panel_h = (fh * 0.84).clamp(grid_h + 76.0, fh - 32.0);
        let panel_x = (fw - panel_w) * 0.5;
        let panel_y = (fh - panel_h) * 0.5;
        let panel_rect = [panel_x, panel_y, panel_w, panel_h];

        let grid_origin = [panel_x + panel_w - grid_w - 20.0, panel_y + 44.0];

        // Left equipment paperdoll area
        let left_x = panel_x + 20.0;
        let top_y = panel_y + 44.0;
        let slot_unit = (cell_size * 1.15).clamp(24.0, 50.0);

        let slot_rects = vec![
            (
                EquipmentSlot::Pistol,
                [left_x, top_y, slot_unit * 2.1, slot_unit * 1.1],
            ),
            (
                EquipmentSlot::Helmet,
                [left_x + slot_unit * 2.35, top_y, slot_unit * 1.8, slot_unit * 1.4],
            ),
            (
                EquipmentSlot::Backpack,
                [left_x + slot_unit * 4.4, top_y, slot_unit * 1.8, slot_unit * 1.4],
            ),
            (
                EquipmentSlot::PrimaryWeapon,
                [left_x, top_y + slot_unit * 1.6, slot_unit * 3.8, slot_unit * 1.7],
            ),
            (
                EquipmentSlot::SecondaryWeapon,
                [left_x, top_y + slot_unit * 3.5, slot_unit * 3.8, slot_unit * 1.7],
            ),
            (
                EquipmentSlot::Armor,
                [left_x + slot_unit * 4.05, top_y + slot_unit * 1.6, slot_unit * 2.15, slot_unit * 3.6],
            ),
            (
                EquipmentSlot::Artifact(0),
                [left_x, top_y + slot_unit * 5.5, slot_unit * 1.1, slot_unit * 1.1],
            ),
            (
                EquipmentSlot::Artifact(1),
                [left_x + slot_unit * 1.25, top_y + slot_unit * 5.5, slot_unit * 1.1, slot_unit * 1.1],
            ),
            (
                EquipmentSlot::Artifact(2),
                [left_x + slot_unit * 2.5, top_y + slot_unit * 5.5, slot_unit * 1.1, slot_unit * 1.1],
            ),
            (
                EquipmentSlot::Artifact(3),
                [left_x + slot_unit * 3.75, top_y + slot_unit * 5.5, slot_unit * 1.1, slot_unit * 1.1],
            ),
            (
                EquipmentSlot::Artifact(4),
                [left_x + slot_unit * 5.0, top_y + slot_unit * 5.5, slot_unit * 1.1, slot_unit * 1.1],
            ),
            (
                EquipmentSlot::QuickSlot(0),
                [left_x, top_y + slot_unit * 6.85, slot_unit * 1.35, slot_unit * 1.15],
            ),
            (
                EquipmentSlot::QuickSlot(1),
                [left_x + slot_unit * 1.55, top_y + slot_unit * 6.85, slot_unit * 1.35, slot_unit * 1.15],
            ),
            (
                EquipmentSlot::QuickSlot(2),
                [left_x + slot_unit * 3.10, top_y + slot_unit * 6.85, slot_unit * 1.35, slot_unit * 1.15],
            ),
            (
                EquipmentSlot::QuickSlot(3),
                [left_x + slot_unit * 4.65, top_y + slot_unit * 6.85, slot_unit * 1.35, slot_unit * 1.15],
            ),
        ];

        let use_button_rect = [
            left_x,
            panel_y + panel_h - 46.0,
            130.0,
            30.0,
        ];
        let close_button_rect = [
            panel_x + panel_w - 36.0,
            panel_y + 8.0,
            26.0,
            24.0,
        ];

        let dlg_w = (fw * 0.56).clamp(320.0, 680.0);
        let dlg_h = 190.0;
        let dlg_x = (fw - dlg_w) * 0.5;
        let dlg_y = fh - dlg_h - 28.0;
        let dialogue_rect = [dlg_x, dlg_y, dlg_w, dlg_h];
        let dialogue_option_rects = (0..3)
            .map(|i| [dlg_x + 18.0, dlg_y + 68.0 + (i as f32) * 34.0, dlg_w - 36.0, 28.0])
            .collect();

        Self {
            panel_rect,
            grid_origin,
            cell_size,
            grid_cols: cols,
            grid_rows: rows,
            slot_rects,
            use_button_rect,
            close_button_rect,
            dialogue_rect,
            dialogue_option_rects,
        }
    }

    pub fn grid_cell_center(&self, col: u8, row: u8, w: u8, h: u8) -> [f32; 2] {
        [
            self.grid_origin[0] + (f32::from(col) + f32::from(w) * 0.5) * self.cell_size,
            self.grid_origin[1] + (f32::from(row) + f32::from(h) * 0.5) * self.cell_size,
        ]
    }

    pub fn slot_center(&self, slot: EquipmentSlot) -> Option<[f32; 2]> {
        self.slot_rects
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, r)| [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5])
    }

    pub fn hit_slot(&self, px: f32, py: f32) -> Option<EquipmentSlot> {
        self.slot_rects
            .iter()
            .find(|(_, r)| point_in_rect(px, py, *r))
            .map(|(s, _)| *s)
    }

    pub fn hit_grid_cell(&self, px: f32, py: f32) -> Option<(u8, u8)> {
        let rx = px - self.grid_origin[0];
        let ry = py - self.grid_origin[1];
        if rx < 0.0 || ry < 0.0 {
            return None;
        }
        let col = (rx / self.cell_size).floor() as i32;
        let row = (ry / self.cell_size).floor() as i32;
        if col >= 0
            && row >= 0
            && col < i32::from(self.grid_cols)
            && row < i32::from(self.grid_rows)
        {
            Some((col as u8, row as u8))
        } else {
            None
        }
    }
}

/// Complete runtime inventory, equipment, actor vital stats, and dialogue manager.
#[derive(Debug, Clone)]
pub struct InventoryManager {
    pub open: bool,
    pub grid_cols: u8,
    pub grid_rows: u8,
    pub items: Vec<InventoryItem>,
    pub actor_stats: ScopActorStats,
    pub selected_uid: Option<u64>,
    pub drag: Option<DragState>,
    pub hover_target: Option<UseTarget>,
    pub dialogue: Option<DialoguePromptState>,
    pub hud_sync_dirty: bool,
    pub max_carry_weight_kg: f32,
    next_uid: u64,
    item_uses_extra: Vec<(xray_formats::scoc::ScocValue, xray_formats::scoc::ScocValue)>,
}

impl Default for InventoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl InventoryManager {
    pub fn new() -> Self {
        Self {
            open: false,
            grid_cols: DEFAULT_GRID_COLS,
            grid_rows: DEFAULT_GRID_ROWS,
            items: Vec::new(),
            actor_stats: ScopActorStats::default(),
            selected_uid: None,
            drag: None,
            hover_target: None,
            dialogue: None,
            hud_sync_dirty: false,
            max_carry_weight_kg: 50.0,
            next_uid: 1,
            item_uses_extra: Vec::new(),
        }
    }

    /// Construct an inventory pre-populated with a classic Cordon starter stalker kit.
    pub fn new_starter() -> Self {
        let mut mgr = Self::new();
        let ak = mgr.add_item("wpn_ak74");
        let _ = mgr.equip_item(ak, EquipmentSlot::PrimaryWeapon, None, None);
        let pm = mgr.add_item("wpn_pm");
        let _ = mgr.equip_item(pm, EquipmentSlot::Pistol, None, None);
        let suit = mgr.add_item("stalker_outfit");
        let _ = mgr.equip_item(suit, EquipmentSlot::Armor, None, None);

        let medkit = mgr.add_item("medkit");
        let _ = mgr.equip_item(medkit, EquipmentSlot::QuickSlot(0), None, None);
        let bandage = mgr.add_item("bandage");
        let _ = mgr.equip_item(bandage, EquipmentSlot::QuickSlot(1), None, None);
        let drink = mgr.add_item("energy_drink");
        let _ = mgr.equip_item(drink, EquipmentSlot::QuickSlot(2), None, None);

        mgr.add_item("ammo_5.45x39_fmj");
        mgr.add_item("medkit_army");
        mgr.hud_sync_dirty = false;
        mgr
    }

    /// True if any modal inventory or NPC dialogue panel is capturing cursor clicks.
    #[inline]
    pub fn is_modal_open(&self) -> bool {
        self.open || self.dialogue.is_some()
    }

    /// Total carried weight across backpack & equipped items in kilograms.
    pub fn total_weight_kg(&self) -> f32 {
        self.items
            .iter()
            .map(|it| it.weight_kg * (it.quantity.max(1) as f32))
            .sum()
    }

    /// Toggle the Inventory panel (`Tab` / `I`), clearing any stale drag state and playing SFX.
    pub fn toggle(&mut self, audio: Option<&mut AudioManager>) -> bool {
        self.set_open(!self.open, audio);
        self.open
    }

    pub fn set_open(&mut self, open: bool, audio: Option<&mut AudioManager>) {
        if self.open == open {
            return;
        }
        self.open = open;
        if !open {
            self.drag = None;
        }
        if let Some(aud) = audio {
            if open {
                aud.play_ui_open();
            } else {
                aud.play_ui_close();
            }
        }
    }

    /// Hydrate inventory items and `item_remaining_uses` from a peer/actor [`ScocFile`].
    pub fn load_from_scoc(&mut self, scoc: &ScocFile) {
        let peer = scoc.extract_peer_state();
        self.load_from_scoc_peer_state(&peer);
    }

    /// Hydrate from extracted [`ScocPeerState`] (`inventory` + `item_remaining_uses`).
    pub fn load_from_scoc_peer_state(&mut self, peer: &ScocPeerState) {
        self.items.clear();
        self.drag = None;
        self.selected_uid = None;
        self.item_uses_extra = peer.item_remaining_uses_extra.clone();

        for raw in &peer.inventory {
            let uid = self.allocate_uid();
            let mut item = InventoryItem::from_scoc_item(uid, raw, &peer.item_remaining_uses);
            if let Some(slot) = item.equipped_slot
                && self.item_in_slot(slot).is_some()
            {
                item.equipped_slot = None;
            }
            if item.equipped_slot.is_none() {
                item.grid_pos = self.find_free_cell(item.grid_size.0, item.grid_size.1, None);
            }
            self.items.push(item);
        }
        self.hud_sync_dirty = true;
    }

    /// Export live inventory items and `item_remaining_uses` into a [`ScocPeerState`].
    pub fn write_to_scoc_peer_state(&self, peer: &mut ScocPeerState) {
        peer.inventory.clear();
        peer.item_remaining_uses.clear();
        peer.item_remaining_uses_extra = self.item_uses_extra.clone();

        for item in &self.items {
            let scoc_item = item.to_scoc_item();
            if let (Some(id), Some(uses)) = (scoc_item.id, scoc_item.remaining_uses) {
                peer.item_remaining_uses.insert(id, uses);
            }
            peer.inventory.push(scoc_item);
        }
    }

    /// Export live inventory directly as a [`ScocFile`].
    pub fn export_to_scoc(&self) -> ScocFile {
        let mut peer = ScocPeerState::default();
        self.write_to_scoc_peer_state(&mut peer);
        ScocFile::from_peer_state(&peer)
    }

    /// Hydrate actor stats and actor-owned inventory entities (`parent_id == actor_id`) from a [`ScopSaveFile`].
    pub fn load_from_scop(&mut self, scop: &ScopSaveFile) {
        self.actor_stats = scop.header.actor_stats.clone();
        let actor_id = scop
            .entities
            .iter()
            .find(|e| e.kind() == ScopAlifeKind::Actor)
            .map(|e| e.id())
            .unwrap_or(0);

        let owned: Vec<_> = scop
            .entities
            .iter()
            .filter(|e| {
                e.parent_id() == actor_id
                    && matches!(e.kind(), ScopAlifeKind::Weapon | ScopAlifeKind::InventoryItem)
            })
            .collect();

        if !owned.is_empty() {
            self.items.clear();
            self.drag = None;
            for ent in owned {
                let uid = self.allocate_uid();
                let mut item = InventoryItem::from_section(uid, ent.section().to_string());
                item.source_id = Some(i64::from(ent.id()));
                item.grid_pos = self.find_free_cell(item.grid_size.0, item.grid_size.1, None);
                self.items.push(item);
            }
            self.hud_sync_dirty = true;
        }
    }

    fn allocate_uid(&mut self) -> u64 {
        let id = self.next_uid;
        self.next_uid = self.next_uid.saturating_add(1);
        id
    }

    /// Add an item by section name into the first free 2D grid cell.
    pub fn add_item(&mut self, section: impl Into<String>) -> u64 {
        let uid = self.allocate_uid();
        let mut item = InventoryItem::from_section(uid, section);
        item.grid_pos = self.find_free_cell(item.grid_size.0, item.grid_size.1, None);
        self.items.push(item);
        uid
    }

    pub fn item(&self, uid: u64) -> Option<&InventoryItem> {
        self.items.iter().find(|it| it.uid == uid)
    }

    pub fn item_mut(&mut self, uid: u64) -> Option<&mut InventoryItem> {
        self.items.iter_mut().find(|it| it.uid == uid)
    }

    pub fn item_in_slot(&self, slot: EquipmentSlot) -> Option<&InventoryItem> {
        self.items.iter().find(|it| it.equipped_slot == Some(slot))
    }

    /// Return the item occupying backpack cell `(col, row)`, accounting for multi-cell items.
    pub fn item_at_grid_cell(&self, col: u8, row: u8) -> Option<&InventoryItem> {
        self.items.iter().find(|it| {
            if it.equipped_slot.is_some() {
                return false;
            }
            if let Some((gx, gy)) = it.grid_pos {
                col >= gx
                    && col < gx.saturating_add(it.grid_size.0)
                    && row >= gy
                    && row < gy.saturating_add(it.grid_size.1)
            } else {
                false
            }
        })
    }

    /// Check whether a `w x h` rectangle at `(col, row)` fits within the grid without overlapping.
    pub fn can_place_at(&self, col: u8, row: u8, w: u8, h: u8, ignore_uid: Option<u64>) -> bool {
        if col.saturating_add(w) > self.grid_cols || row.saturating_add(h) > self.grid_rows {
            return false;
        }
        let x0 = col;
        let y0 = row;
        let x1 = col + w;
        let y1 = row + h;

        for other in &self.items {
            if Some(other.uid) == ignore_uid || other.equipped_slot.is_some() {
                continue;
            }
            let Some((ox, oy)) = other.grid_pos else {
                continue;
            };
            let ox1 = ox + other.grid_size.0;
            let oy1 = oy + other.grid_size.1;
            let overlaps = x0 < ox1 && x1 > ox && y0 < oy1 && y1 > oy;
            if overlaps {
                return false;
            }
        }
        true
    }

    /// Find the first free top-left grid coordinate `(col, row)` for an item of size `(w, h)`.
    pub fn find_free_cell(&self, w: u8, h: u8, ignore_uid: Option<u64>) -> Option<(u8, u8)> {
        if w > self.grid_cols || h > self.grid_rows {
            return None;
        }
        for row in 0..=self.grid_rows - h {
            for col in 0..=self.grid_cols - w {
                if self.can_place_at(col, row, w, h, ignore_uid) {
                    return Some((col, row));
                }
            }
        }
        None
    }

    /// Move an item (either from another grid cell or unequipping a slot) to backpack cell `(col, row)`.
    pub fn move_item_to_grid(&mut self, uid: u64, col: u8, row: u8) -> bool {
        let Some(size) = self.item(uid).map(|i| i.grid_size) else {
            return false;
        };
        if !self.can_place_at(col, row, size.0, size.1, Some(uid)) {
            return false;
        }
        let was_weapon_slot = self
            .item(uid)
            .and_then(|i| i.equipped_slot)
            .and_then(|s| s.hud_slot_number());
        if let Some(it) = self.item_mut(uid) {
            it.equipped_slot = None;
            it.grid_pos = Some((col, row));
        }
        if was_weapon_slot.is_some() {
            self.hud_sync_dirty = true;
        }
        true
    }

    /// Check if `category` is allowed in `slot`.
    pub fn is_compatible(category: ItemCategory, slot: EquipmentSlot) -> bool {
        match slot {
            EquipmentSlot::PrimaryWeapon
            | EquipmentSlot::SecondaryWeapon
            | EquipmentSlot::Pistol => category == ItemCategory::Weapon,
            EquipmentSlot::Armor => category == ItemCategory::Outfit,
            EquipmentSlot::Helmet => category == ItemCategory::Helmet,
            EquipmentSlot::Backpack => category == ItemCategory::Backpack,
            EquipmentSlot::Artifact(idx) => {
                idx < ARTIFACT_SLOT_COUNT && category == ItemCategory::Artifact
            }
            EquipmentSlot::QuickSlot(idx) => {
                idx < QUICK_SLOT_COUNT
                    && matches!(category, ItemCategory::Medical | ItemCategory::Consumable)
            }
        }
    }

    /// Equip an item into `slot`, swapping any existing occupant back into the backpack grid.
    pub fn equip_item(
        &mut self,
        uid: u64,
        slot: EquipmentSlot,
        hud_weapon: Option<&mut HudWeaponController>,
        audio: Option<&mut AudioManager>,
    ) -> bool {
        let Some((cat, section)) = self.item(uid).map(|i| (i.category, i.section.clone())) else {
            return false;
        };
        if !Self::is_compatible(cat, slot) {
            return false;
        }

        // If slot is already occupied by a different item, move occupant back into free grid cell
        if let Some(existing_uid) = self.item_in_slot(slot).map(|i| i.uid)
            && existing_uid != uid
        {
            let prev_size = self.item(existing_uid).map(|i| i.grid_size).unwrap_or((1, 1));
            let free_cell = self.find_free_cell(prev_size.0, prev_size.1, Some(uid));
            if let Some(prev) = self.item_mut(existing_uid) {
                prev.equipped_slot = None;
                prev.grid_pos = free_cell;
            }
        }

        if let Some(it) = self.item_mut(uid) {
            it.equipped_slot = Some(slot);
            it.grid_pos = None;
        }
        self.hud_sync_dirty = true;
        if let Some(hw) = hud_weapon {
            self.sync_hud_weapon(hw);
            if let Some(hud_slot) = slot.hud_slot_number() {
                hw.active_slot = hud_slot;
            }
            self.hud_sync_dirty = false;
        }
        if let Some(aud) = audio {
            aud.play_item_use(&section);
        }
        true
    }

    /// Unequip the item currently occupying `slot` back to the backpack grid.
    pub fn unequip_slot(
        &mut self,
        slot: EquipmentSlot,
        hud_weapon: Option<&mut HudWeaponController>,
        audio: Option<&mut AudioManager>,
    ) -> bool {
        let Some((uid, size, section)) = self
            .item_in_slot(slot)
            .map(|i| (i.uid, i.grid_size, i.section.clone()))
        else {
            return false;
        };
        let Some(free) = self.find_free_cell(size.0, size.1, Some(uid)) else {
            return false;
        };
        if let Some(it) = self.item_mut(uid) {
            it.equipped_slot = None;
            it.grid_pos = Some(free);
        }
        self.hud_sync_dirty = true;
        if let Some(hw) = hud_weapon {
            self.sync_hud_weapon(hw);
            if let Some(hud_slot) = slot.hud_slot_number()
                && hw.active_slot == hud_slot
            {
                // Fall back to melee knife slot when unequipping currently held firearm
                hw.active_slot = 1;
            }
            self.hud_sync_dirty = false;
        }
        if let Some(aud) = audio {
            aud.play_item_use(&section);
        }
        true
    }

    /// Push equipped `PrimaryWeapon` (slot 3), `SecondaryWeapon` (slot 4), and `Pistol` (slot 2)
    /// into [`HudWeaponController`], updating OGF paths, ammo counts, and weapon stats.
    pub fn sync_hud_weapon(&mut self, hw: &mut HudWeaponController) {
        let slot_mappings = [
            (EquipmentSlot::Pistol, 2usize, "wpn_pm"),
            (EquipmentSlot::PrimaryWeapon, 3usize, "wpn_ak74"),
            (EquipmentSlot::SecondaryWeapon, 4usize, "wpn_mp5"),
        ];
        for (eq_slot, hud_idx, default_sec) in slot_mappings {
            let slot_arr_idx = hud_idx - 1;
            if let Some(item) = self.item_in_slot(eq_slot).cloned() {
                let spec = build_weapon_spec_for_section(hud_idx as u8, &item.section, item.ammo_elapsed);
                hw.slots[slot_arr_idx] = spec;
            } else {
                // Slot unequipped: mark section empty / cleared clip so HUD knows slot is vacant
                let mut empty_spec = WeaponSpec::default_loadout()[slot_arr_idx].clone();
                if eq_slot != EquipmentSlot::PrimaryWeapon && eq_slot != EquipmentSlot::Pistol {
                    empty_spec.section = String::new();
                } else {
                    empty_spec.section = format!("unequipped_{default_sec}");
                }
                empty_spec.ammo_in_clip = 0;
                empty_spec.reserve_ammo = 0;
                hw.slots[slot_arr_idx] = empty_spec;
            }
        }
        self.hud_sync_dirty = false;
    }

    /// Use / consume an item (medkit, bandage, energy drink, antirad) or equip if gear/weapon.
    pub fn use_item(
        &mut self,
        uid: u64,
        hud_weapon: Option<&mut HudWeaponController>,
        alife: Option<&mut AlifeManager>,
        audio: Option<&mut AudioManager>,
    ) -> Option<ItemUseEffect> {
        let item = self.item(uid)?.clone();
        if !matches!(item.category, ItemCategory::Medical | ItemCategory::Consumable) {
            // Double-clicking / using weapon or suit equips it into its preferred slot
            let meta = classify_section_meta(&item.section);
            if let Some(pref_slot) = meta.preferred_slot {
                let _ = self.equip_item(uid, pref_slot, hud_weapon, audio);
            }
            return None;
        }

        let lower = item.section.to_ascii_lowercase();
        let mut effect = ItemUseEffect {
            consumed: true,
            ..Default::default()
        };
        if lower.contains("medkit_scientic") {
            effect.health_delta = 0.75;
            effect.bleeding_delta = -0.5;
            effect.radiation_delta = -0.35;
        } else if lower.contains("medkit_army") {
            effect.health_delta = 0.60;
            effect.bleeding_delta = -0.6;
        } else if lower.contains("medkit") || lower.contains("stim") {
            effect.health_delta = 0.45;
            effect.bleeding_delta = -0.3;
        } else if lower.contains("bandage") {
            effect.health_delta = 0.08;
            effect.bleeding_delta = -1.0;
        } else if lower.contains("antirad") {
            effect.radiation_delta = -0.75;
        } else if lower.contains("drink") || lower.contains("water") {
            effect.stamina_delta = 0.65;
            effect.health_delta = 0.04;
        } else if lower.contains("vodka") {
            effect.radiation_delta = -0.4;
            effect.stamina_delta = -0.1;
        } else {
            effect.health_delta = 0.12;
            effect.stamina_delta = 0.25;
        }

        self.actor_stats.health = (self.actor_stats.health + effect.health_delta).clamp(0.0, 1.0);
        self.actor_stats.stamina = (self.actor_stats.stamina + effect.stamina_delta).clamp(0.0, 1.0);
        self.actor_stats.bleeding = (self.actor_stats.bleeding + effect.bleeding_delta).clamp(0.0, 1.0);
        self.actor_stats.radiation =
            (self.actor_stats.radiation + effect.radiation_delta).clamp(0.0, 1.0);

        if let Some(mgr) = alife {
            mgr.actor_stats.health = self.actor_stats.health;
            mgr.actor_stats.stamina = self.actor_stats.stamina;
            mgr.actor_stats.bleeding = self.actor_stats.bleeding;
            mgr.actor_stats.radiation = self.actor_stats.radiation;
            if let Some(actor_id) = mgr.host_actor_id
                && let Some(actor_ent) = mgr.entities.get_mut(&actor_id)
            {
                actor_ent.health = self.actor_stats.health;
                actor_ent.dirty_net_sync = true;
            }
        }

        if let Some(aud) = audio {
            aud.play_item_use(&item.section);
        }

        // Decrement remaining uses / quantity / remove when depleted
        let mut remove_item = false;
        if let Some(it) = self.item_mut(uid) {
            if let Some(uses) = it.remaining_uses {
                if uses > 1 {
                    it.remaining_uses = Some(uses - 1);
                } else if it.quantity > 1 {
                    it.quantity -= 1;
                    it.remaining_uses = Some(1);
                } else {
                    remove_item = true;
                }
            } else if it.quantity > 1 {
                it.quantity -= 1;
            } else {
                remove_item = true;
            }
        }
        if remove_item {
            self.items.retain(|it| it.uid != uid);
            if self.selected_uid == Some(uid) {
                self.selected_uid = None;
            }
        }

        Some(effect)
    }

    /// Consume the item placed in `QuickSlot(idx)` (0..4).
    pub fn use_quick_slot(
        &mut self,
        slot_idx: usize,
        hud_weapon: Option<&mut HudWeaponController>,
        alife: Option<&mut AlifeManager>,
        audio: Option<&mut AudioManager>,
    ) -> Option<ItemUseEffect> {
        let uid = self.item_in_slot(EquipmentSlot::QuickSlot(slot_idx))?.uid;
        self.use_item(uid, hud_weapon, alife, audio)
    }

    /// Scan `AlifeManager` for friendly Stalker/Trader NPCs or ground items within `'E'` use range.
    pub fn scan_use_target(
        &mut self,
        cam_pos: [f32; 3],
        cam_forward: [f32; 3],
        alife: Option<&AlifeManager>,
    ) -> Option<UseTarget> {
        let mgr = alife?;
        let max_range = 3.2f32;
        let min_dot = 0.80f32;
        let mut best: Option<(f32, UseTarget)> = None;

        for ent in mgr.entities.values() {
            if !ent.online || ent.role == EntityRole::LocalHostActor {
                continue;
            }
            // Skip items already inside somebody's inventory
            if ent.parent_id != INVALID_OBJECT_ID {
                continue;
            }

            let target_center = match ent.role {
                EntityRole::Stalker | EntityRole::Trader | EntityRole::RemotePeerActor { .. } => {
                    [ent.position[0], ent.position[1] + 1.35, ent.position[2]]
                }
                EntityRole::InventoryItem | EntityRole::Weapon => {
                    [ent.position[0], ent.position[1] + 0.15, ent.position[2]]
                }
                _ => continue,
            };

            let dx = target_center[0] - cam_pos[0];
            let dy = target_center[1] - cam_pos[1];
            let dz = target_center[2] - cam_pos[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist > max_range || dist < 1e-4 {
                continue;
            }
            let dot = (dx * cam_forward[0] + dy * cam_forward[1] + dz * cam_forward[2]) / dist;
            if dot < min_dot {
                continue;
            }

            let cand = match ent.role {
                EntityRole::Trader => UseTarget::Dialogue {
                    entity_id: ent.id,
                    name: if ent.name.is_empty() {
                        ent.section.clone()
                    } else {
                        ent.name.clone()
                    },
                    is_trader: true,
                    distance_m: dist,
                },
                EntityRole::Stalker | EntityRole::RemotePeerActor { .. } => UseTarget::Dialogue {
                    entity_id: ent.id,
                    name: if ent.name.is_empty() {
                        ent.section.clone()
                    } else {
                        ent.name.clone()
                    },
                    is_trader: false,
                    distance_m: dist,
                },
                EntityRole::InventoryItem | EntityRole::Weapon => UseTarget::Pickup {
                    entity_id: ent.id,
                    section: ent.section.clone(),
                    display_name: prettify_section(&ent.section),
                    distance_m: dist,
                },
                _ => continue,
            };

            if best.as_ref().is_none_or(|(bd, _)| dist < *bd) {
                best = Some((dist, cand));
            }
        }

        self.hover_target = best.map(|(_, t)| t);
        self.hover_target.clone()
    }

    /// Execute `Action::Use` (`E`): close/advance dialogue, open NPC/Trader dialogue, or pick up item.
    pub fn interact_with_target(
        &mut self,
        alife: Option<&mut AlifeManager>,
        mut audio: Option<&mut AudioManager>,
    ) -> bool {
        if self.dialogue.is_some() {
            self.dialogue = None;
            if let Some(aud) = audio.as_deref_mut() {
                aud.play_ui_close();
            }
            return true;
        }

        let Some(target) = self.hover_target.clone() else {
            return false;
        };

        match target {
            UseTarget::Dialogue {
                entity_id,
                name,
                is_trader,
                ..
            } => {
                let greeting = if is_trader {
                    format!("{name}: Good hunting, stalker. Got artifacts or spare gear to trade?")
                } else {
                    format!("{name}: Keep your head down out here. Anomalies shifted since sunrise.")
                };
                let options = if is_trader {
                    vec![
                        "1. Restock Medkit (+1 Medkit)".to_string(),
                        "2. What's the news around Cordon?".to_string(),
                        "3. Catch you later.".to_string(),
                    ]
                } else {
                    vec![
                        "1. Seen any mutants or artifacts nearby?".to_string(),
                        "2. Safe travels, stalker.".to_string(),
                    ]
                };
                self.dialogue = Some(DialoguePromptState {
                    speaker_id: entity_id,
                    speaker_name: name,
                    is_trader,
                    greeting,
                    selected_option: 0,
                    options,
                    status_line: String::new(),
                });
                if let Some(aud) = audio.as_deref_mut() {
                    aud.play_ui_open();
                }
                true
            }
            UseTarget::Pickup {
                entity_id,
                section,
                ..
            } => {
                if let Some(mgr) = alife
                    && let Some(ent) = mgr.entities.get_mut(&entity_id)
                {
                    ent.online = false;
                    ent.parent_id = mgr.host_actor_id.unwrap_or(0);
                    ent.dirty_net_sync = true;
                }
                let uid = self.add_item(section);
                if let Some(it) = self.item_mut(uid) {
                    it.source_id = Some(i64::from(entity_id));
                }
                self.hover_target = None;
                if let Some(aud) = audio {
                    aud.play_item_pickup();
                }
                true
            }
        }
    }

    /// Process pointer clicks, drag-and-drop, right-click use, and dialogue buttons inside this frame.
    pub fn handle_input(
        &mut self,
        input: &EngineInput,
        viewport_w: u32,
        viewport_h: u32,
        mut hud_weapon: Option<&mut HudWeaponController>,
        alife: Option<&mut AlifeManager>,
        mut audio: Option<&mut AudioManager>,
    ) {
        let layout = InventoryLayout::compute(viewport_w, viewport_h, self.grid_cols, self.grid_rows);
        let (mx, my) = input.mouse_position();

        // Dialogue modal interactions
        if let Some(mut dlg) = self.dialogue.take() {
            if input.is_mouse_just_pressed(MouseButton::Left) {
                let mut close_dialogue = false;
                for (idx, rect) in layout.dialogue_option_rects.iter().enumerate() {
                    if idx < dlg.options.len() && point_in_rect(mx, my, *rect) {
                        dlg.selected_option = idx;
                        if let Some(aud) = audio.as_deref_mut() {
                            aud.play_ui_click("interface\\pda_task");
                        }
                        let is_last = idx + 1 == dlg.options.len();
                        if is_last {
                            close_dialogue = true;
                        } else if dlg.is_trader && idx == 0 {
                            self.add_item("medkit");
                            if let Some(aud) = audio.as_deref_mut() {
                                aud.play_item_pickup();
                            }
                            dlg.status_line = "Received Medkit from trader.".to_string();
                        } else {
                            dlg.status_line =
                                "Artifact signals spotted north near the old factory.".to_string();
                        }
                        break;
                    }
                }
                if !close_dialogue {
                    self.dialogue = Some(dlg);
                } else if let Some(aud) = audio.as_deref_mut() {
                    aud.play_ui_close();
                }
            } else {
                self.dialogue = Some(dlg);
            }
            return;
        }

        if !self.open {
            return;
        }

        // Update active drag cursor
        if let Some(drag) = &mut self.drag {
            drag.current_pointer = [mx, my];
        }

        // Right click -> instant use / equip item under cursor
        if input.is_mouse_just_pressed(MouseButton::Right) {
            if let Some(slot) = layout.hit_slot(mx, my)
                && let Some(it_uid) = self.item_in_slot(slot).map(|i| i.uid)
            {
                let item_cat = self.item(it_uid).map(|i| i.category);
                if matches!(item_cat, Some(ItemCategory::Medical | ItemCategory::Consumable)) {
                    let _ = self.use_item(it_uid, hud_weapon, alife, audio);
                } else {
                    let _ = self.unequip_slot(slot, hud_weapon, audio);
                }
                return;
            }
            if let Some((col, row)) = layout.hit_grid_cell(mx, my)
                && let Some(uid) = self.item_at_grid_cell(col, row).map(|i| i.uid)
            {
                let _ = self.use_item(uid, hud_weapon, alife, audio);
                return;
            }
        }

        // Left press -> Close button, Use Selected button, or begin Drag
        if input.is_mouse_just_pressed(MouseButton::Left) {
            if point_in_rect(mx, my, layout.close_button_rect) {
                self.set_open(false, audio);
                return;
            }
            if point_in_rect(mx, my, layout.use_button_rect)
                && let Some(uid) = self.selected_uid
            {
                let _ = self.use_item(uid, hud_weapon, alife, audio);
                return;
            }
            if let Some(slot) = layout.hit_slot(mx, my)
                && let Some(uid) = self.item_in_slot(slot).map(|i| i.uid)
            {
                self.selected_uid = Some(uid);
                self.drag = Some(DragState {
                    item_uid: uid,
                    start_pointer: [mx, my],
                    current_pointer: [mx, my],
                    origin_cell: None,
                    origin_slot: Some(slot),
                });
                return;
            }
            if let Some((col, row)) = layout.hit_grid_cell(mx, my)
                && let Some(item) = self.item_at_grid_cell(col, row)
            {
                let uid = item.uid;
                let origin_cell = item.grid_pos;
                self.selected_uid = Some(uid);
                self.drag = Some(DragState {
                    item_uid: uid,
                    start_pointer: [mx, my],
                    current_pointer: [mx, my],
                    origin_cell,
                    origin_slot: None,
                });
                return;
            }
        }

        // Left release -> finish drag-and-drop into slot or grid cell
        if input.is_mouse_just_released(MouseButton::Left)
            && let Some(drag) = self.drag.take()
        {
            let dx = mx - drag.start_pointer[0];
            let dy = my - drag.start_pointer[1];
            let moved_dist = (dx * dx + dy * dy).sqrt();

            if moved_dist >= 5.0 {
                if let Some(target_slot) = layout.hit_slot(mx, my) {
                    let _ = self.equip_item(
                        drag.item_uid,
                        target_slot,
                        hud_weapon.as_deref_mut(),
                        audio.as_deref_mut(),
                    );
                } else if let Some((col, row)) = layout.hit_grid_cell(mx, my) {
                    let ok = self.move_item_to_grid(drag.item_uid, col, row);
                    if ok && let Some(hw) = hud_weapon && self.hud_sync_dirty {
                        self.sync_hud_weapon(hw);
                    }
                    if ok && let Some(aud) = audio {
                        aud.play_ui_click("interface\\inv_slot");
                    }
                }
            }
        }
    }

    /// Render the always-on Stalker HUD bars, interaction prompt, open Inventory grid/slots, and dialogue panel.
    #[allow(clippy::too_many_arguments)]
    pub fn render_ui(
        &self,
        renderer: &mut Dx12Renderer,
        hud_weapon: Option<&HudWeaponController>,
        width: u32,
        height: u32,
        show_crosshair: bool,
        stamina: f32,
        font: Option<&UiFontResource>,
    ) -> u32 {
        let mut drawn = 0u32;
        let fw = (width.max(64)) as f32;
        let fh = (height.max(64)) as f32;

        // 1. Always-on Stalker HUD (Crosshair when no modal is open, Health/Stamina/Bleeding, Ammo & Quick Slots)
        if show_crosshair && !self.is_modal_open() {
            renderer.ui_draw_crosshair([fw * 0.5, fh * 0.5], 10.0, 4.0, 2.0, [0.95, 0.95, 0.95, 0.85]);
            drawn += 4;
        }

        // Bottom-left Health, Stamina, Bleeding & Radiation bars
        let bar_w = (fw * 0.18).clamp(90.0, 260.0);
        let bar_h = (fh * 0.016).clamp(6.0, 14.0);
        let margin = 16.0;
        renderer.ui_fill_rect(
            [margin - 6.0, fh - margin - bar_h * 4.4, bar_w + 12.0, bar_h * 4.2 + 8.0],
            [0.05, 0.06, 0.07, 0.62],
        );
        drawn += 1;

        renderer.ui_draw_meter_bar(
            [margin, fh - margin - bar_h * 3.8, bar_w, bar_h],
            self.actor_stats.health.clamp(0.0, 1.0),
            [0.08, 0.08, 0.08, 0.82],
            [0.84, 0.18, 0.18, 0.96],
        );
        renderer.ui_draw_meter_bar(
            [margin, fh - margin - bar_h * 2.2, bar_w, bar_h * 0.78],
            stamina.clamp(0.0, 1.0),
            [0.08, 0.08, 0.08, 0.82],
            [0.20, 0.70, 0.94, 0.96],
        );
        drawn += 4;

        // Bleeding / Radiation indicator pips above health
        if self.actor_stats.bleeding > 0.02 {
            renderer.ui_draw_meter_bar(
                [margin, fh - margin - bar_h * 5.0, bar_w * 0.48, 5.0],
                self.actor_stats.bleeding.clamp(0.0, 1.0),
                [0.12, 0.04, 0.04, 0.7],
                [0.95, 0.15, 0.10, 0.95],
            );
            drawn += 2;
        }
        if self.actor_stats.radiation > 0.02 {
            renderer.ui_draw_meter_bar(
                [margin + bar_w * 0.52, fh - margin - bar_h * 5.0, bar_w * 0.48, 5.0],
                self.actor_stats.radiation.clamp(0.0, 1.0),
                [0.10, 0.10, 0.04, 0.7],
                [0.92, 0.85, 0.15, 0.95],
            );
            drawn += 2;
        }

        // Quick-slot mini strip (bottom-center-left)
        let q_size = 26.0;
        for q_idx in 0..QUICK_SLOT_COUNT {
            let qx = margin + bar_w + 18.0 + (q_idx as f32) * (q_size + 6.0);
            let qy = fh - margin - q_size - 4.0;
            renderer.ui_fill_rect([qx, qy, q_size, q_size], [0.07, 0.08, 0.09, 0.75]);
            drawn += 1;
            if let Some(q_item) = self.item_in_slot(EquipmentSlot::QuickSlot(q_idx)) {
                renderer.ui_fill_rect([qx + 3.0, qy + 3.0, q_size - 6.0, q_size - 6.0], q_item.category.accent_color());
                drawn += 1;
            }
        }

        // Bottom-right Weapon Ammo HUD strip
        if let Some(hw) = hud_weapon {
            let spec = hw.active_weapon();
            let aw = 148.0;
            let ah = 42.0;
            let ax = fw - margin - aw;
            let ay = fh - margin - ah;
            renderer.ui_fill_rect([ax, ay, aw, ah], [0.05, 0.06, 0.07, 0.68]);
            drawn += 1;
            let clip_frac = if spec.clip_capacity > 0 {
                (spec.ammo_in_clip as f32 / spec.clip_capacity as f32).clamp(0.0, 1.0)
            } else {
                1.0
            };
            renderer.ui_draw_meter_bar(
                [ax + 8.0, ay + ah - 14.0, aw - 16.0, 7.0],
                clip_frac,
                [0.10, 0.10, 0.10, 0.85],
                [0.90, 0.74, 0.25, 0.95],
            );
            drawn += 2;
            if let Some(f) = font {
                let ammo_txt = if spec.clip_capacity > 0 {
                    format!("{} {}/{}", spec.section, spec.ammo_in_clip, spec.reserve_ammo)
                } else {
                    spec.section.clone()
                };
                drawn += f.draw_text(renderer, &ammo_txt, [ax + 8.0, ay + 6.0], 14.0, [0.95, 0.92, 0.85, 0.95]);
            }
        }

        // Center interaction prompt when looking at NPC/Trader/Item
        if let Some(target) = &self.hover_target
            && !self.is_modal_open()
        {
            let pw = 280.0;
            let ph = 28.0;
            let px = (fw - pw) * 0.5;
            let py = fh * 0.58;
            renderer.ui_fill_rect([px, py, pw, ph], [0.06, 0.08, 0.10, 0.82]);
            renderer.ui_fill_rect([px, py, 4.0, ph], [0.90, 0.72, 0.22, 0.95]);
            drawn += 2;
            if let Some(f) = font {
                drawn += f.draw_text(renderer, &target.prompt_text(), [px + 12.0, py + 6.0], 15.0, [0.96, 0.95, 0.90, 1.0]);
            }
        }

        // 2. Open Inventory Modal
        let layout = InventoryLayout::compute(width, height, self.grid_cols, self.grid_rows);
        if self.open {
            // Backdrop dim + main container
            renderer.ui_fill_rect([0.0, 0.0, fw, fh], [0.02, 0.02, 0.03, 0.45]);
            renderer.ui_fill_rect(layout.panel_rect, [0.08, 0.09, 0.10, 0.92]);
            // Header bar & close X button
            renderer.ui_fill_rect(
                [layout.panel_rect[0], layout.panel_rect[1], layout.panel_rect[2], 32.0],
                [0.14, 0.15, 0.17, 0.98],
            );
            renderer.ui_fill_rect(layout.close_button_rect, [0.68, 0.20, 0.20, 0.95]);
            drawn += 4;

            // Equipment Slots
            for (slot, rect) in &layout.slot_rects {
                let occ = self.item_in_slot(*slot);
                let is_sel = occ.is_some_and(|i| Some(i.uid) == self.selected_uid);
                let bg = if is_sel {
                    [0.24, 0.26, 0.20, 0.95]
                } else {
                    [0.12, 0.13, 0.15, 0.92]
                };
                renderer.ui_fill_rect(*rect, bg);
                drawn += 1;

                if let Some(item) = occ
                    && self.drag.as_ref().is_none_or(|d| d.item_uid != item.uid)
                {
                    let inner = [rect[0] + 4.0, rect[1] + 4.0, (rect[2] - 8.0).max(4.0), (rect[3] - 12.0).max(4.0)];
                    renderer.ui_fill_rect(inner, item.category.accent_color());
                    renderer.ui_draw_meter_bar(
                        [rect[0] + 4.0, rect[1] + rect[3] - 6.0, (rect[2] - 8.0).max(4.0), 3.5],
                        item.condition,
                        [0.05, 0.05, 0.05, 0.85],
                        condition_bar_color(item.condition),
                    );
                    drawn += 3;
                }
                if let Some(f) = font {
                    let lbl = occ.map(|i| i.display_name.as_str()).unwrap_or(slot.label());
                    drawn += f.draw_text(renderer, lbl, [rect[0] + 6.0, rect[1] + 6.0], 13.0, [0.92, 0.92, 0.90, 0.95]);
                }
            }

            // 2D Backpack Grid cells
            for row in 0..self.grid_rows {
                for col in 0..self.grid_cols {
                    let cx = layout.grid_origin[0] + f32::from(col) * layout.cell_size;
                    let cy = layout.grid_origin[1] + f32::from(row) * layout.cell_size;
                    renderer.ui_fill_rect(
                        [cx + 1.0, cy + 1.0, layout.cell_size - 2.0, layout.cell_size - 2.0],
                        [0.11, 0.12, 0.13, 0.88],
                    );
                    drawn += 1;
                }
            }

            // Backpack Items
            for item in &self.items {
                if item.equipped_slot.is_some()
                    || self.drag.as_ref().is_some_and(|d| d.item_uid == item.uid)
                {
                    continue;
                }
                let Some((col, row)) = item.grid_pos else {
                    continue;
                };
                let ix = layout.grid_origin[0] + f32::from(col) * layout.cell_size + 2.0;
                let iy = layout.grid_origin[1] + f32::from(row) * layout.cell_size + 2.0;
                let iw = f32::from(item.grid_size.0) * layout.cell_size - 4.0;
                let ih = f32::from(item.grid_size.1) * layout.cell_size - 4.0;
                let is_sel = Some(item.uid) == self.selected_uid;
                if is_sel {
                    renderer.ui_fill_rect([ix - 1.5, iy - 1.5, iw + 3.0, ih + 3.0], [0.95, 0.78, 0.25, 0.95]);
                    drawn += 1;
                }
                renderer.ui_fill_rect([ix, iy, iw, ih], item.category.accent_color());
                renderer.ui_draw_meter_bar(
                    [ix + 2.0, iy + ih - 5.0, (iw - 4.0).max(2.0), 3.0],
                    item.condition,
                    [0.05, 0.05, 0.05, 0.85],
                    condition_bar_color(item.condition),
                );
                drawn += 3;
                if let Some(f) = font {
                    drawn += f.draw_text(renderer, &item.display_name, [ix + 4.0, iy + 3.0], 12.0, [0.95, 0.95, 0.92, 0.96]);
                }
            }

            // Selected Item Info & Use/Equip action button
            if let Some(sel) = self.selected_uid.and_then(|id| self.item(id)) {
                let btn_col = if matches!(sel.category, ItemCategory::Medical | ItemCategory::Consumable) {
                    [0.24, 0.56, 0.30, 0.96]
                } else {
                    [0.52, 0.44, 0.22, 0.96]
                };
                renderer.ui_fill_rect(layout.use_button_rect, btn_col);
                drawn += 1;
                if let Some(f) = font {
                    let info = format!(
                        "{} | {:.1}kg | {}%",
                        sel.display_name,
                        sel.weight_kg,
                        (sel.condition * 100.0).round() as u32
                    );
                    drawn += f.draw_text(
                        renderer,
                        &info,
                        [layout.use_button_rect[0] + 140.0, layout.use_button_rect[1] + 7.0],
                        14.0,
                        [0.92, 0.92, 0.88, 0.95],
                    );
                }
            }

            // Weight bar at bottom of grid
            let weight_frac = (self.total_weight_kg() / self.max_carry_weight_kg.max(1.0)).clamp(0.0, 1.0);
            let weight_color = if weight_frac > 0.9 {
                [0.88, 0.26, 0.18, 0.95]
            } else {
                [0.78, 0.68, 0.28, 0.95]
            };
            renderer.ui_draw_meter_bar(
                [
                    layout.grid_origin[0],
                    layout.panel_rect[1] + layout.panel_rect[3] - 26.0,
                    f32::from(self.grid_cols) * layout.cell_size,
                    10.0,
                ],
                weight_frac,
                [0.06, 0.06, 0.06, 0.9],
                weight_color,
            );
            drawn += 2;

            // Floating Drag Ghost following mouse cursor
            if let Some(drag) = &self.drag
                && let Some(item) = self.item(drag.item_uid)
            {
                let gw = f32::from(item.grid_size.0) * layout.cell_size;
                let gh = f32::from(item.grid_size.1) * layout.cell_size;
                let mut ghost_col = item.category.accent_color();
                ghost_col[3] = 0.72;
                renderer.ui_fill_rect(
                    [drag.current_pointer[0] - gw * 0.5, drag.current_pointer[1] - gh * 0.5, gw, gh],
                    ghost_col,
                );
                drawn += 1;
            }
        }

        // 3. Open NPC / Trader Dialogue Panel
        if let Some(dlg) = &self.dialogue {
            renderer.ui_fill_rect(layout.dialogue_rect, [0.06, 0.07, 0.09, 0.94]);
            renderer.ui_fill_rect(
                [layout.dialogue_rect[0], layout.dialogue_rect[1], layout.dialogue_rect[2], 32.0],
                [0.16, 0.18, 0.20, 0.98],
            );
            drawn += 2;

            for (idx, rect) in layout.dialogue_option_rects.iter().enumerate() {
                if idx >= dlg.options.len() {
                    break;
                }
                let col = if idx == dlg.selected_option {
                    [0.28, 0.36, 0.26, 0.95]
                } else {
                    [0.13, 0.15, 0.17, 0.92]
                };
                renderer.ui_fill_rect(*rect, col);
                drawn += 1;
                if let Some(f) = font {
                    drawn += f.draw_text(renderer, &dlg.options[idx], [rect[0] + 10.0, rect[1] + 6.0], 14.0, [0.95, 0.94, 0.88, 1.0]);
                }
            }
            if let Some(f) = font {
                drawn += f.draw_text(
                    renderer,
                    &dlg.greeting,
                    [layout.dialogue_rect[0] + 18.0, layout.dialogue_rect[1] + 40.0],
                    14.0,
                    [0.92, 0.88, 0.76, 0.98],
                );
            }
        }

        drawn
    }
}

/// Build a matching `WeaponSpec` when equipping a firearm section into slot 2/3/4.
fn build_weapon_spec_for_section(slot: u8, section: &str, saved_ammo: Option<i64>) -> WeaponSpec {
    let lower = section.to_ascii_lowercase();
    let defaults = WeaponSpec::default_loadout();
    let template = if lower.contains("svd") {
        &defaults[4]
    } else if lower.contains("mp5") || lower.contains("klin") || lower.contains("kiparis") {
        &defaults[3]
    } else if lower.contains("ak") || lower.contains("abakan") || lower.contains("lr300") || lower.contains("val") {
        &defaults[2]
    } else if lower.contains("pm") || lower.contains("pb") || lower.contains("usp") || lower.contains("fort") {
        &defaults[1]
    } else if lower.contains("knife") {
        &defaults[0]
    } else if slot == 2 {
        &defaults[1]
    } else {
        &defaults[2]
    };

    let mut spec = template.clone();
    spec.slot = slot;
    spec.section = section.to_string();
    if let Some(ammo) = saved_ammo {
        spec.ammo_in_clip = (ammo.max(0) as u32).min(spec.clip_capacity.max(1));
    }
    if lower.contains("svd") {
        spec.fire_mode = WeaponFireMode::SemiAuto;
    }
    spec
}

#[inline]
fn point_in_rect(px: f32, py: f32, rect: [f32; 4]) -> bool {
    px >= rect[0] && px <= rect[0] + rect[2] && py >= rect[1] && py <= rect[1] + rect[3]
}

#[inline]
fn condition_bar_color(cond: f32) -> [f32; 4] {
    if cond >= 0.65 {
        [0.24, 0.82, 0.34, 0.95]
    } else if cond >= 0.30 {
        [0.88, 0.72, 0.20, 0.95]
    } else {
        [0.86, 0.22, 0.18, 0.95]
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use xray_formats::scoc::{ScocTable, ScocValue};

    #[test]
    fn test_inventory_grid_placement_drag_and_equip_sync() {
        let mut inv = InventoryManager::new();
        let ak_uid = inv.add_item("wpn_ak74");
        let svd_uid = inv.add_item("wpn_svd");

        let ak_pos = inv.item(ak_uid).unwrap().grid_pos.unwrap();
        let svd_pos = inv.item(svd_uid).unwrap().grid_pos.unwrap();
        assert_ne!(ak_pos, svd_pos, "2D grid items must not overlap");
        assert!(!inv.move_item_to_grid(svd_uid, ak_pos.0, ak_pos.1));

        // Equip AK-74 into PrimaryWeapon and sync with HudWeaponController
        let mut hw = HudWeaponController::default();
        let mut audio = AudioManager::new_headless();
        assert!(inv.equip_item(ak_uid, EquipmentSlot::PrimaryWeapon, Some(&mut hw), Some(&mut audio)));
        assert_eq!(hw.active_slot, 3);
        assert_eq!(hw.slots[2].section, "wpn_ak74");
        assert_eq!(audio.stats().item_sounds_triggered, 1);

        // Equipping SVD into PrimaryWeapon swaps AK-74 back into the 2D backpack grid
        assert!(inv.equip_item(svd_uid, EquipmentSlot::PrimaryWeapon, Some(&mut hw), Some(&mut audio)));
        assert!(inv.item(ak_uid).unwrap().grid_pos.is_some());
        assert_eq!(inv.item(ak_uid).unwrap().equipped_slot, None);
        assert_eq!(hw.slots[2].section, "wpn_svd");

        // Unequipping PrimaryWeapon returns SVD to grid and falls back active HUD slot to knife
        assert!(inv.unequip_slot(EquipmentSlot::PrimaryWeapon, Some(&mut hw), Some(&mut audio)));
        assert!(inv.item(svd_uid).unwrap().grid_pos.is_some());
        assert_eq!(hw.active_slot, 1);
    }

    #[test]
    fn test_scoc_roundtrip_and_consumables_uses() {
        let mut inv_tbl = ScocTable::new();
        let mut item1 = ScocTable::new();
        item1.set_str("id", ScocValue::int(4001));
        item1.set_str("section", ScocValue::string("medkit"));
        item1.set_str("condition", ScocValue::float(0.95));
        inv_tbl.set_i64(1, ScocValue::Table(item1));

        let mut uses_tbl = ScocTable::new();
        uses_tbl.set_i64(4001, ScocValue::int(3));

        let mut root = ScocTable::new();
        root.set_str("inventory", ScocValue::Table(inv_tbl));
        root.set_str("item_remaining_uses", ScocValue::Table(uses_tbl));
        let scoc = ScocFile::new(root);

        let mut inv = InventoryManager::new();
        inv.load_from_scoc(&scoc);
        inv.actor_stats.health = 0.25;
        inv.actor_stats.bleeding = 0.80;
        inv.actor_stats.stamina = 0.20;

        let med_uid = inv.items[0].uid;
        assert_eq!(inv.items[0].remaining_uses, Some(3));

        let mut audio = AudioManager::new_headless();
        let mut alife = AlifeManager::default();
        let eff = inv
            .use_item(med_uid, None, Some(&mut alife), Some(&mut audio))
            .expect("medkit must be usable");
        assert!(eff.health_delta > 0.4);
        assert!(inv.actor_stats.health > 0.65);
        assert_eq!(inv.item(med_uid).unwrap().remaining_uses, Some(2));
        assert_eq!(audio.stats().item_sounds_triggered, 1);

        // Bandage stops bleeding completely; energy drink refills stamina
        let band_uid = inv.add_item("bandage");
        let drink_uid = inv.add_item("energy_drink");
        let _ = inv.use_item(band_uid, None, Some(&mut alife), Some(&mut audio));
        assert_eq!(inv.actor_stats.bleeding, 0.0);
        assert!(inv.item(band_uid).is_none());

        let _ = inv.use_item(drink_uid, None, Some(&mut alife), Some(&mut audio));
        assert!(inv.actor_stats.stamina > 0.8);

        // Export back to ScocPeerState preserves updated `item_remaining_uses`
        let exported = inv.export_to_scoc();
        let peer = exported.extract_peer_state();
        assert_eq!(peer.item_remaining_uses.get(&4001), Some(&2));
    }
}
