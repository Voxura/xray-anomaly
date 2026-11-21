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
use std::sync::Arc;

use crate::alife_manager::{AlifeManager, EntityRole};
use crate::hud_weapon::{HudWeaponController, WeaponFireMode, WeaponSpec};
use crate::input::{EngineInput, MouseButton};
use xray_audio::AudioManager;
use xray_core::ltx_loader::OwnedLtxDocument;
use xray_formats::fonts::{FontAlignment, FontAtlas, FontDrawParams};
use xray_formats::spawns::INVALID_OBJECT_ID;
use xray_formats::scoc::{ScocFile, ScocInventoryItem, ScocPeerState};
use xray_formats::scop::{ScopActorInventory, ScopActorStats, ScopAlifeEntity, ScopAlifeKind, ScopAlifePayload, ScopItemPlace, ScopSaveFile};
use xray_formats::spawns::AlifeItemAmmoState;
use xray_formats::vfs_bridge::{load_ltx_tree_from_vfs, load_system_ltx_from_vfs, load_ui_xml_for_viewport_from_vfs, load_xml_from_vfs};
use xray_formats::xml_ui::{UiTextureAtlasMap, UiViewport, XmlDocument, XmlNode};
use xray_render_dx12::bindless::DescriptorIndex;
use xray_render_dx12::renderer::Dx12Renderer;
use xray_render_dx12::ui_canvas::GlyphQuad;
use xray_vfs::VirtualFileSystem;

/// Default backpack grid width in cells (classic X-Ray 10-column grid).
pub const DEFAULT_GRID_COLS: u8 = 10;
/// Default backpack grid height in cells.
pub const DEFAULT_GRID_ROWS: u8 = 12;
const NATIVE_ICON_SIZE: f32 = 41.0;
const NATIVE_GRID_PITCH: f32 = NATIVE_ICON_SIZE + 2.0;
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
        self.draw_aligned(renderer, text, pos_xy, line_height_px, FontAlignment::Left, color)
    }

    fn draw_aligned(
        &self,
        renderer: &mut Dx12Renderer,
        text: &str,
        pos_xy: [f32; 2],
        line_height_px: f32,
        align: FontAlignment,
        color: [f32; 4],
    ) -> u32 {
        if text.is_empty() {
            return 0;
        }
        let params = FontDrawParams {
            current_height: line_height_px,
            align,
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

#[derive(Debug, Clone)]
struct InventoryText {
    rect: [f32; 4],
    font: String,
    alignment: FontAlignment,
}

impl InventoryText {
    fn draw(
        &self,
        renderer: &mut Dx12Renderer,
        default: Option<&UiFontResource>,
        fonts: &BTreeMap<&str, UiFontResource>,
        text: &str,
        color: [f32; 4],
    ) -> u32 {
        let Some(font) = fonts.get(self.font.as_str()).or(default) else { return 0 };
        let [x, y, width, _] = self.rect;
        let x = match self.alignment {
            FontAlignment::Left => x,
            FontAlignment::Center => x + width * 0.5,
            FontAlignment::Right => x + width,
        };
        font.draw_aligned(renderer, text, [x, y], font.atlas.line_height, self.alignment, color)
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
    NativeSlot(u8),
    Artifact(usize),
    QuickSlot(usize),
}

impl EquipmentSlot {
    fn from_native_index(index: u8) -> Option<Self> {
        Some(match index {
            2 => Self::Pistol,
            3 => Self::PrimaryWeapon,
            7 => Self::Armor,
            12 => Self::Helmet,
            13 => Self::Backpack,
            1 | 4..=6 | 8..=11 | 14..=18 => Self::NativeSlot(index),
            _ => return None,
        })
    }

    fn native_index(self) -> Option<u8> {
        match self {
            Self::Pistol => Some(2),
            Self::PrimaryWeapon => Some(3),
            Self::Armor => Some(7),
            Self::Helmet => Some(12),
            Self::Backpack => Some(13),
            Self::NativeSlot(index) => Some(index),
            _ => None,
        }
    }

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
            Self::NativeSlot(1) => Some(1),
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
            Self::NativeSlot(1) => "KNIFE",
            Self::NativeSlot(4) => "GRENADE",
            Self::NativeSlot(5) => "BINOCULARS",
            Self::NativeSlot(6) => "BOLT",
            Self::NativeSlot(8) => "PDA",
            Self::NativeSlot(9) => "DETECTOR",
            Self::NativeSlot(10) => "TORCH",
            Self::NativeSlot(_) => "SLOT",
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

fn inventory_picker_enabled(script: &str) -> bool {
    script.lines().find_map(|line| {
        let line = line.split("--").next()?;
        let (binding, value) = line.split_once('=')?;
        let mut binding = binding.split_whitespace();
        if binding.next() != Some("local") || binding.next() != Some("enable_item_picker") || binding.next().is_some() {
            return None;
        }
        Some(value.trim().trim_end_matches(';').trim() == "true")
    }).unwrap_or(false)
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
    has_upgrades: bool,
    pub remaining_uses: Option<i64>,
    pub quantity: u32,
    pub ammo_elapsed: Option<i64>,
    pub ammo_type: Option<i64>,
    pub ammo_section: Option<String>,
    pub weight_kg: f32,
    pub source_scoc: Option<ScocInventoryItem>,
}

impl InventoryItem {
    pub fn from_section_indexed(
        uid: u64,
        section: impl Into<String>,
        grid_index: Option<&ItemGridIndex>,
    ) -> Self {
        let section_str = section.into();
        let mut meta = classify_section_meta(&section_str);
        if let Some(idx) = grid_index
            && let Some((_, _, gw, gh)) = idx.get(&section_str)
        {
            meta.grid_size = (gw.max(1), gh.max(1));
        }
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
            has_upgrades: false,
            remaining_uses: meta.default_uses,
            quantity: 1,
            ammo_elapsed: None,
            ammo_type: None,
            ammo_section: None,
            weight_kg: meta.weight_kg,
            source_scoc: None,
        }
    }

    pub fn from_section(uid: u64, section: impl Into<String>) -> Self {
        Self::from_section_indexed(uid, section, None)
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
            has_upgrades: false,
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
    bag_viewport: Option<[f32; 4]>,
    picker_rect: Option<[f32; 4]>,
    pub grid_cols: u8,
    pub grid_rows: u8,
    grid_scroll_row: u8,
    pub slot_rects: Vec<(EquipmentSlot, [f32; 4])>,
    pub use_button_rect: [f32; 4],
    pub close_button_rect: [f32; 4],
    pub dialogue_rect: [f32; 4],
    pub dialogue_option_rects: Vec<[f32; 4]>,
    backgrounds: Vec<([f32; 4], String)>,
    rotated_slots: Vec<EquipmentSlot>,
    weight_caption: Option<InventoryText>,
    weight: Option<InventoryText>,
    weight_max: Option<InventoryText>,
    money: Option<InventoryText>,
    name: Option<InventoryText>,
    faction: Option<InventoryText>,
    portrait_rect: Option<[f32; 4]>,
    health_rect: Option<[f32; 4]>,
    condition_rects: BTreeMap<EquipmentSlot, [f32; 4]>,
    filter_buttons: Vec<(usize, [f32; 4], String)>,
}

impl InventoryLayout {
    pub fn compute(width: u32, height: u32, cols: u8, rows: u8) -> Self {
        let fw = (width.max(320)) as f32;
        let fh = (height.max(240)) as f32;

        let cell_size = ((fw * 0.032).min(fh * 0.052)).clamp(20.0, 44.0);
        let grid_w = f32::from(cols) * cell_size;
        let grid_h = f32::from(rows) * cell_size;

        let panel_w = (fw * 0.84).max(grid_w + 300.0).min(fw - 32.0);
        let panel_h = (fh * 0.84).max(grid_h + 76.0).min(fh - 32.0);
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
            bag_viewport: None,
            picker_rect: None,
            grid_cols: cols,
            grid_rows: rows,
            grid_scroll_row: 0,
            slot_rects,
            use_button_rect,
            close_button_rect,
            dialogue_rect,
            dialogue_option_rects,
            backgrounds: Vec::new(),
            rotated_slots: Vec::new(),
            weight_caption: None,
            weight: None,
            weight_max: None,
            money: None,
            name: None,
            faction: None,
            portrait_rect: None,
            health_rect: None,
            condition_rects: BTreeMap::new(),
            filter_buttons: Vec::new(),
        }
    }

    fn apply_xml(&mut self, xml: &XmlDocument, viewport: UiViewport) -> Option<()> {
        let player = xml.navigate("player", 0)?;
        let equipment = xml.navigate("equipment", 0)?;
        let relative = |parent: &XmlNode, origin: [f32; 2]| {
            [
                origin[0] + parent.attr_f32("x", 0.0),
                origin[1] + parent.attr_f32("y", 0.0),
                parent.attr_f32("width", 0.0),
                parent.attr_f32("height", 0.0),
            ]
        };
        let player_rect = relative(player, [0.0; 2]);
        let equipment_rect = relative(equipment, [0.0; 2]);
        let player_origin = [player_rect[0], player_rect[1]];
        let equipment_origin = [equipment_rect[0], equipment_rect[1]];
        let [x, y, w, h] = Self::native_bag_rect(xml, viewport)?;
        let pitch = (NATIVE_GRID_PITCH * viewport.scale()[1]).min(w / f32::from(self.grid_cols.max(1)));
        self.grid_origin = [x, y];
        self.cell_size = pitch;
        self.grid_rows = (h / pitch).floor().max(1.0) as u8;
        self.bag_viewport = Some([x, y, w, h]);
        self.picker_rect = equipment.navigate("cont_picker", 0)
            .map(|node| viewport.virtual_to_screen_rect(relative(node, equipment_origin)))
            .filter(|rect| rect[2] > 0.0 && rect[3] > 0.0);
        self.backgrounds = [(equipment, equipment_origin), (player, player_origin)].into_iter()
            .filter_map(|(node, origin)| {
                let background = node.navigate("background", 0)?;
                let texture = background.navigate("texture", 0)?.read_text()?.trim().to_owned();
                Some((viewport.virtual_to_screen_rect(relative(background, origin)), texture))
            }).collect();
        let text_control = |path| {
            let node = player.navigate(path, 0)?;
            let text = node.navigate("text", 0)?;
            Some(InventoryText {
                rect: viewport.virtual_to_screen_rect(relative(node, player_origin)),
                font: text.attr_or("font", "letterica16").to_owned(),
                alignment: match text.attr_or("align", "l") {
                    "r" => FontAlignment::Right,
                    "c" => FontAlignment::Center,
                    _ => FontAlignment::Left,
                },
            })
        };
        self.weight_caption = text_control("weight_caption");
        self.weight = text_control("weight");
        self.weight_max = text_control("weight_max");
        self.money = text_control("money");
        self.name = text_control("name");
        self.faction = text_control("community");
        self.portrait_rect = player.navigate("icon", 0)
            .map(|node| viewport.virtual_to_screen_rect(relative(node, player_origin)));
        self.health_rect = equipment.navigate("actor_state_info", 0).and_then(|state| {
            let [x, y, _, _] = relative(state, equipment_origin);
            let sensor = state.navigate("health_sensor", 0)?;
            let [x, y, _, _] = relative(sensor, [x, y]);
            let progress = sensor.navigate("state_progress", 0)?;
            Some(viewport.virtual_to_screen_rect(relative(progress, [x, y])))
        });
        if let Some(sorter) = equipment.navigate("sorter", 0) {
            let [x, y, _, _] = relative(sorter, equipment_origin);
            for index in 0..8 {
                if let Some(button) = sorter.navigate(&format!("btn_{}", index + 1), 0)
                    && let Some(texture) = button.navigate("texture", 0).and_then(XmlNode::read_text)
                {
                    self.filter_buttons.push((index, viewport.virtual_to_screen_rect(relative(button, [x, y])), texture.trim().to_owned()));
                }
            }
        }
        for (slot, rect) in &mut self.slot_rects {
            let (container, id) = match slot {
                EquipmentSlot::Pistol => ("cont_equ", 2),
                EquipmentSlot::PrimaryWeapon => ("cont_equ", 3),
                EquipmentSlot::SecondaryWeapon => ("cont_equ", 1),
                EquipmentSlot::Armor => ("cont_equ", 7),
                EquipmentSlot::Helmet => ("cont_equ", 12),
                EquipmentSlot::Backpack => ("cont_equ", 13),
                EquipmentSlot::NativeSlot(id) => ("cont_equ", usize::from(*id)),
                EquipmentSlot::Artifact(index) => ("cont_belt", *index + 1),
                EquipmentSlot::QuickSlot(index) => ("cont_quick", *index + 1),
            };
            let Some(container) = equipment.navigate(container, 0) else { continue };
            let container_rect = relative(container, equipment_origin);
            let Some(cell) = container.navigate(&format!("cell_{id}"), 0) else { continue };
            let [x, y, w, h] = relative(cell, [container_rect[0], container_rect[1]]);
            *rect = viewport.virtual_to_screen_rect([x, y, w, h]);
            if let Some(bar) = cell.navigate("bar", 0) {
                self.condition_rects.insert(*slot, viewport.virtual_to_screen_rect(relative(bar, [x, y])));
            }
            if cell.navigate("pic", 0).is_some_and(|pic| {
                pic.attr_bool("heading", false) && pic.attr_f32("heading_angle", 0.0).abs() == 90.0
            }) {
                self.rotated_slots.push(*slot);
            }
        }
        self.use_button_rect = viewport.virtual_to_screen_rect([16.0, 160.0, 115.0, 29.0]);
        self.close_button_rect = [0.0; 4];
        Some(())
    }

    fn native_bag_rect(xml: &XmlDocument, viewport: UiViewport) -> Option<[f32; 4]> {
        let player = xml.navigate("player", 0)?;
        let bag = player.navigate("cont_bag", 0)?;
        let width = bag.attr_f32("width", 0.0);
        let height = bag.attr_f32("height", 0.0);
        (width > 0.0 && height > 0.0).then(|| viewport.virtual_to_screen_rect([
            player.attr_f32("x", 0.0) + bag.attr_f32("x", 0.0),
            player.attr_f32("y", 0.0) + bag.attr_f32("y", 0.0),
            width,
            height,
        ]))
    }

    pub fn grid_cell_center(&self, col: u8, row: u8, w: u8, h: u8) -> [f32; 2] {
        [
            self.grid_origin[0] + (f32::from(col) + f32::from(w) * 0.5) * self.cell_size,
            self.grid_origin[1] + (f32::from(row) - f32::from(self.grid_scroll_row)
                + f32::from(h) * 0.5) * self.cell_size,
        ]
    }

    pub fn slot_center(&self, slot: EquipmentSlot) -> Option<[f32; 2]> {
        self.slot_rects
            .iter()
            .find(|(s, _)| *s == slot)
            .map(|(_, r)| [r[0] + r[2] * 0.5, r[1] + r[3] * 0.5])
    }

    fn grid_rect(&self) -> [f32; 4] {
        self.bag_viewport.unwrap_or([
            self.grid_origin[0],
            self.grid_origin[1],
            f32::from(self.grid_cols) * self.cell_size,
            f32::from(self.grid_rows) * self.cell_size,
        ])
    }

    fn scrollbar_rect(&self) -> [f32; 4] {
        let [x, y, width, height] = self.grid_rect();
        [x + width + 1.0, y, 6.0, height]
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
        if !point_in_rect(px, py, self.grid_rect()) {
            return None;
        }
        let col = (rx / self.cell_size).floor() as i32;
        let row = (ry / self.cell_size).floor() as i32;
        if col >= 0
            && row >= 0
            && col < i32::from(self.grid_cols)
            && row < i32::from(self.grid_rows) + i32::from(self.bag_viewport.is_some())
        {
            Some((col as u8, self.grid_scroll_row.saturating_add(row as u8)))
        } else {
            None
        }
    }
}

#[cfg(test)]
#[test]
fn xml_inventory_layout_maps_draw_and_hit_rectangles_together() {
    let xml = XmlDocument::parse(r#"
        <w>
            <player x="600"><background width="400" height="768"><texture>bag</texture></background>
                <cont_bag x="20" y="80" width="350" height="620"/>
            </player>
            <equipment x="320" y="15">
                <cont_equ x="10" y="5"><cell_2 x="40" y="70" width="70" height="160">
                    <pic heading="1" heading_angle="90"/>
                </cell_2></cont_equ>
            </equipment>
        </w>
    "#).unwrap();
    let mut layout = InventoryLayout::compute(1280, 720, 10, 12);
    assert!(layout.apply_xml(&xml, UiViewport::new(1280.0, 720.0)).is_some());
    assert_eq!(layout.grid_origin, [775.0, 75.0]);
    assert_eq!(layout.hit_slot(500.0, 100.0), Some(EquipmentSlot::Pistol));
    assert!(layout.rotated_slots.contains(&EquipmentSlot::Pistol));
    let [x, y] = layout.grid_cell_center(2, 3, 1, 1);
    assert_eq!(layout.hit_grid_cell(x, y), Some((2, 3)));
}

#[cfg(test)]
#[test]
fn full_backpack_grows_and_scrolls_to_the_last_item() {
    let mut inventory = InventoryManager::new();
    let full_capacity = u16::from(DEFAULT_GRID_ROWS) * u16::from(DEFAULT_GRID_COLS);
    let last = (0..=full_capacity).map(|_| inventory.add_item("bandage")).last().unwrap();
    let (col, row) = inventory.item(last).unwrap().grid_pos.unwrap();
    assert!(row >= DEFAULT_GRID_ROWS);

    inventory.set_open(true, None);
    let mut input = EngineInput::new();
    let layout = inventory.layout(1280, 720);
    let [x, y] = layout.grid_cell_center(0, 0, 1, 1);
    input.on_mouse_move(x, y);
    input.on_mouse_wheel(-100.0);
    inventory.handle_input(&input, 1280, 720, None, None, None);

    let layout = inventory.layout(1280, 720);
    let [x, y] = layout.grid_cell_center(col, row, 1, 1);
    assert_eq!(layout.hit_grid_cell(x, y), Some((col, row)));
    assert_eq!(inventory.item_at_grid_cell(col, row).unwrap().uid, last);
}

#[cfg(test)]
#[test]
fn configured_weight_and_display_stacks_preserve_individual_items() {
    use xray_core::ltx_loader::LoadedLtxSource;

    let mut inventory = InventoryManager::new();
    inventory.ui_layouts[0] = Some(Arc::new(XmlDocument::parse("<w/>").unwrap()));
    inventory.item_settings = Some(Arc::new(OwnedLtxDocument::from_sources(&[LoadedLtxSource {
        virtual_path: "items.ltx".to_owned(),
        depth: 0,
        text: "[base_food]\ninv_weight = 0.1\n[bandage]:base_food\n[stalker_outfit]\nadditional_inventory_weight = 12\n[repair_kit]\nclass = II_FOOD\nuse_condition = true\nmax_uses = 3\n[prt_i_ammo]\n[unique_supply]\ndont_stack = true\n".to_owned(),
    }])));
    let first = inventory.add_item("bandage");
    let second = inventory.add_item("bandage");
    let cell = inventory.item(first).unwrap().grid_pos.unwrap();
    assert_eq!(inventory.item(second).unwrap().grid_pos, Some(cell));
    assert_eq!(inventory.visible_bag_stacks()[&cell].1, 2);
    assert!((inventory.total_weight_kg() - 0.2).abs() < 1e-5);

    let outfit = inventory.add_item("stalker_outfit");
    assert!(inventory.equip_item(outfit, EquipmentSlot::Armor, None, None));
    assert_eq!(inventory.carry_limit_kg(), 62.0);
    assert!(inventory.move_item_to_grid(first, cell.0 + 1, cell.1));
    assert_eq!(inventory.visible_bag_stacks()[&cell].1, 1);
    assert_eq!(inventory.item_at_grid_cell(cell.0, cell.1).unwrap().uid, second);

    let component = inventory.add_item("prt_i_ammo");
    inventory.item_mut(component).unwrap().condition = 0.4;
    let another = inventory.add_item("prt_i_ammo");
    assert_eq!(inventory.item(component).unwrap().grid_pos, inventory.item(another).unwrap().grid_pos);

    let used = inventory.add_item("repair_kit");
    inventory.item_mut(used).unwrap().remaining_uses = Some(2);
    let full = inventory.add_item("repair_kit");
    assert_ne!(inventory.item(used).unwrap().grid_pos, inventory.item(full).unwrap().grid_pos);

    let unique = inventory.add_item("unique_supply");
    let other = inventory.add_item("unique_supply");
    assert_ne!(inventory.item(unique).unwrap().grid_pos, inventory.item(other).unwrap().grid_pos);

    // The native item picker bypasses the strict per-instance checks for the backpack.
    inventory.stack_all_sections = inventory_picker_enabled("-- local enable_item_picker = false\nlocal enable_item_picker\t= true -- stock setting");
    inventory.repack_bag();
    assert_eq!(inventory.item(used).unwrap().grid_pos, inventory.item(full).unwrap().grid_pos);
    assert_eq!(inventory.item(unique).unwrap().grid_pos, inventory.item(other).unwrap().grid_pos);
    assert_eq!(inventory.item(used).unwrap().remaining_uses, Some(2));
    assert_eq!(inventory.item(full).unwrap().remaining_uses, Some(3));
    assert!(!inventory_picker_enabled("local enable_item_picker = false"));

    let xml = Arc::new(XmlDocument::parse(
        r#"<w><player x="600"><cont_bag width="350" height="575"/></player>
        <equipment x="350"><cont_picker y="565" width="255" height="162"/></equipment></w>"#,
    ).unwrap());
    inventory.ui_layouts = [Some(xml.clone()), Some(xml)];
    inventory.set_open(true, None);
    let (col, row) = inventory.item(full).unwrap().grid_pos.unwrap();
    let mut pointer = EngineInput::new();
    let [x, y] = inventory.layout(1366, 768).grid_cell_center(col, row, 1, 1);
    pointer.on_mouse_move(x, y);
    pointer.on_mouse_down(MouseButton::Left);
    inventory.handle_input(&pointer, 1366, 768, None, None, None);
    pointer.begin_frame();
    pointer.on_mouse_up(MouseButton::Left);
    inventory.handle_input(&pointer, 1366, 768, None, None, None);

    let picker = inventory.picker_cells(&inventory.layout(1366, 768)).unwrap();
    let ([x, y, w, h], _) = picker.cells.into_iter().find(|(_, item)| item.uid == full).unwrap();
    pointer.begin_frame();
    pointer.on_mouse_move(x + w * 0.5, y + h * 0.5);
    pointer.on_mouse_down(MouseButton::Left);
    inventory.handle_input(&pointer, 1366, 768, None, None, None);
    assert_eq!(inventory.selected_uid, Some(full));
}

#[cfg(test)]
#[test]
fn native_slot_accepts_items_with_the_configured_base_slot() {
    use xray_core::ltx_loader::LoadedLtxSource;

    let mut inventory = InventoryManager::new();
    inventory.native_inventory = Some(ScopActorInventory { active_slot: 0, quick_slots: Default::default() });
    inventory.item_settings = Some(Arc::new(OwnedLtxDocument::from_sources(&[LoadedLtxSource {
        virtual_path: "items.ltx".to_owned(),
        depth: 0,
        text: "[bolt]\nslot = 5\n[utility]:bolt\n[loose_part]\nslot = -1\n".to_owned(),
    }])));
    inventory.ui_layouts[0] = Some(Arc::new(XmlDocument::parse("<w/>").unwrap()));
    inventory.stack_all_sections = true;
    let slot = EquipmentSlot::from_native_index(inventory.configured_base_slot("bolt").unwrap()).unwrap();
    let utility = inventory.add_item("utility");
    let replacement = inventory.add_item("utility");
    let spare = inventory.add_item("utility");
    let part = inventory.add_item("loose_part");
    assert!(inventory.equip_item(utility, slot, None, None));
    assert!(!inventory.equip_item(part, slot, None, None));
    assert_eq!(inventory.item_in_slot(slot).map(|item| item.uid), Some(utility));
    assert!(inventory.equip_item(replacement, slot, None, None));
    assert_eq!(inventory.item(utility).unwrap().grid_pos, inventory.item(spare).unwrap().grid_pos);
    assert!(inventory.unequip_slot(slot, None, None));
    assert!(inventory.item_in_slot(slot).is_none());
    assert_eq!(inventory.item(replacement).unwrap().grid_pos, inventory.item(spare).unwrap().grid_pos);
}

struct StackPickerCells<'a> {
    rect: [f32; 4],
    cells: Vec<([f32; 4], &'a InventoryItem)>,
    max_scroll: usize,
}

/// Complete runtime inventory, equipment, actor vital stats, and dialogue manager.
#[derive(Debug, Clone)]
pub struct InventoryManager {
    pub open: bool,
    pub grid_cols: u8,
    pub grid_rows: u8,
    grid_scroll_row: u8,
    bag_filter: usize,
    stack_picker: Option<((u8, u8), usize)>,
    pub items: Vec<InventoryItem>,
    pub actor_stats: ScopActorStats,
    pub selected_uid: Option<u64>,
    pub drag: Option<DragState>,
    pub hover_target: Option<UseTarget>,
    pub dialogue: Option<DialoguePromptState>,
    pub hud_sync_dirty: bool,
    pub max_carry_weight_kg: f32,
    pub grid_index: Option<Arc<ItemGridIndex>>,
    ui_layouts: [Option<Arc<XmlDocument>>; 2],
    item_settings: Option<Arc<OwnedLtxDocument>>,
    actor_profiles: BTreeMap<String, String>,
    stack_all_sections: bool,
    default_equipment: Option<(EquipmentSlot, &'static str)>,
    currency_suffix: String,
    specific_profile: Option<String>,
    pub(crate) native_inventory: Option<ScopActorInventory>,
    community_ids: Vec<String>,
    next_uid: u64,
    item_uses_extra: Vec<(xray_formats::scoc::ScocValue, xray_formats::scoc::ScocValue)>,
}

impl Default for InventoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed or VFS-loaded item icon coordinate table inside `ui_icon_equipment.dds` (8192x4096).
/// Maps section -> (inv_grid_x, inv_grid_y, inv_grid_width, inv_grid_height).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ItemGridIndex {
    pub entries: BTreeMap<String, (u32, u32, u8, u8)>,
}

impl ItemGridIndex {
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Load and index `inv_grid_x`, `inv_grid_y`, `inv_grid_width`, `inv_grid_height`
    /// from all `configs/items/**/*.ltx` files mounted in VFS.
    pub fn load_from_vfs(vfs: &VirtualFileSystem) -> Self {
        let mut index = Self::new();
        if let Ok(files) = vfs.list_files("configs\\items") {
            for f in files {
                if !f.ends_with(".ltx") {
                    continue;
                }
                if let Ok(content) = vfs.read_to_string_lossy(&f) {
                    index.parse_ltx(&content);
                }
            }
        }
        index
    }

    /// Parse LTX content for item sections and their `inv_grid_*` properties.
    pub fn parse_ltx(&mut self, text: &str) {
        let mut current_sec: Option<String> = None;
        let mut gx = None;
        let mut gy = None;
        let mut gw = None;
        let mut gh = None;

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(';') || trimmed.starts_with("//") {
                continue;
            }
            if trimmed.starts_with('[') && trimmed.contains(']') {
                if let (Some(s), Some(x), Some(y), Some(w), Some(h)) =
                    (current_sec.take(), gx.take(), gy.take(), gw.take(), gh.take())
                {
                    self.entries.entry(s).or_insert((x, y, w, h));
                }
                let inside = &trimmed[1..trimmed.find(']').unwrap_or(trimmed.len())];
                let sec_name = inside
                    .split(':')
                    .next()
                    .unwrap_or(inside)
                    .trim()
                    .to_ascii_lowercase();
                current_sec = Some(sec_name);
                gx = None;
                gy = None;
                gw = None;
                gh = None;
            } else if let Some((lhs, rhs)) = trimmed.split_once('=') {
                let k = lhs.trim().to_ascii_lowercase();
                let v = rhs.split(';').next().unwrap_or("").trim();
                match k.as_str() {
                    "inv_grid_x" => gx = v.parse::<u32>().ok(),
                    "inv_grid_y" => gy = v.parse::<u32>().ok(),
                    "inv_grid_width" => gw = v.parse::<u8>().ok(),
                    "inv_grid_height" => gh = v.parse::<u8>().ok(),
                    _ => {}
                }
            }
        }
        if let (Some(s), Some(x), Some(y), Some(w), Some(h)) = (current_sec, gx, gy, gw, gh) {
            self.entries.entry(s).or_insert((x, y, w, h));
        }
    }

    /// Look up `(gx, gy, gw, gh)` for a section, falling back to base section if variant suffix present.
    pub fn get(&self, section: &str) -> Option<(u32, u32, u8, u8)> {
        let lower = section.trim().to_ascii_lowercase();
        if let Some(entry) = self.entries.get(&lower) {
            return Some(*entry);
        }
        // Try stripping standard condition/variant suffixes
        for suffix in &["_bad", "_verybad", "_worn", "_rusty", "_custom", "_camo", "_alt"] {
            if let Some(base) = lower.strip_suffix(suffix) {
                if let Some(entry) = self.entries.get(base) {
                    return Some(*entry);
                }
            }
        }
        None
    }
}

/// Compute normalized texture UV coordinates `[u1, v1, u2, v2]` inside `ui_icon_equipment.dds` (8192x4096 atlas).
pub fn item_icon_uv(section: &str, category: ItemCategory) -> [f32; 4] {
    item_icon_uv_indexed(section, category, None)
}

/// Compute normalized texture UV coordinates `[u1, v1, u2, v2]` inside `ui_icon_equipment.dds` (8192x4096 atlas),
/// using genuine coordinates parsed from game configs when available.
pub fn item_icon_uv_indexed(
    section: &str,
    category: ItemCategory,
    grid_index: Option<&ItemGridIndex>,
) -> [f32; 4] {
    if let Some(idx) = grid_index
        && let Some((gx, gy, gw, gh)) = idx.get(section)
    {
        let u1 = (gx * 50) as f32 / 8192.0;
        let v1 = (gy * 50) as f32 / 4096.0;
        let u2 = ((gx + u32::from(gw)) * 50) as f32 / 8192.0;
        let v2 = ((gy + u32::from(gh)) * 50) as f32 / 4096.0;
        return [u1, v1, u2, v2];
    }

    let lower = section.to_ascii_lowercase();
    let (gx, gy, gw, gh): (u32, u32, u32, u32) = if lower.contains("knife") {
        (18, 8, 2, 1)
    } else if lower.contains("pm") {
        (0, 8, 2, 1)
    } else if lower.contains("ak74") {
        (0, 0, 5, 2)
    } else if lower.contains("mp5") {
        (5, 0, 3, 2)
    } else if lower.contains("svd") {
        (0, 2, 6, 2)
    } else if lower.contains("bolt") {
        (19, 8, 1, 1)
    } else if lower.contains("medkit") {
        (8, 8, 2, 1)
    } else if lower.contains("bandage") {
        (10, 8, 1, 1)
    } else if lower.contains("antirad") {
        (11, 8, 1, 1)
    } else if lower.contains("energy") {
        (12, 8, 1, 1)
    } else if lower.contains("bread") || lower.contains("kolbasa") {
        (13, 8, 1, 1)
    } else if lower.contains("ammo") {
        (14, 8, 1, 1)
    } else if lower.contains("suit") || lower.contains("outfit") {
        (0, 4, 2, 3)
    } else if lower.contains("helm") {
        (2, 4, 2, 2)
    } else if lower.contains("af_") || category == ItemCategory::Artifact {
        (17, 8, 1, 1)
    } else {
        match category {
            ItemCategory::Weapon => (0, 8, 2, 1),
            ItemCategory::Medical => (8, 8, 2, 1),
            ItemCategory::Consumable => (12, 8, 1, 1),
            ItemCategory::Ammo => (14, 8, 1, 1),
            ItemCategory::Outfit => (0, 4, 2, 3),
            ItemCategory::Helmet => (2, 4, 2, 2),
            ItemCategory::Artifact => (17, 8, 1, 1),
            _ => (19, 8, 1, 1),
        }
    };

    let u1 = (gx * 50) as f32 / 8192.0;
    let v1 = (gy * 50) as f32 / 4096.0;
    let u2 = ((gx + gw) * 50) as f32 / 8192.0;
    let v2 = ((gy + gh) * 50) as f32 / 4096.0;
    [u1, v1, u2, v2]
}

impl InventoryManager {
    pub fn new() -> Self {
        Self {
            open: false,
            grid_cols: DEFAULT_GRID_COLS,
            grid_rows: DEFAULT_GRID_ROWS,
            grid_scroll_row: 0,
            bag_filter: 0,
            stack_picker: None,
            items: Vec::new(),
            actor_stats: ScopActorStats::default(),
            selected_uid: None,
            drag: None,
            hover_target: None,
            dialogue: None,
            hud_sync_dirty: false,
            max_carry_weight_kg: 50.0,
            grid_index: None,
            ui_layouts: [None, None],
            item_settings: None,
            actor_profiles: BTreeMap::new(),
            stack_all_sections: false,
            default_equipment: None,
            currency_suffix: "RU".to_owned(),
            specific_profile: None,
            native_inventory: None,
            community_ids: Vec::new(),
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

    pub(crate) fn load_ui_layouts(&mut self, vfs: &VirtualFileSystem) {
        self.ui_layouts = [
            UiViewport::new(1024.0, 768.0),
            UiViewport::new(1366.0, 768.0),
        ].map(|viewport| {
            load_ui_xml_for_viewport_from_vfs(vfs, "configs/ui/ui_inventory.xml", &viewport)
                .ok().map(Arc::new)
        });
        self.item_settings = load_system_ltx_from_vfs(vfs).ok().map(Arc::new);
        self.stack_all_sections = vfs.read_to_string_lossy("scripts/ui_inventory.script")
            .is_ok_and(|script| inventory_picker_enabled(&script));
        // CInventoryOwner::spawn_supplies gives the actor a default bolt. Its base slot
        // is supplied by the mounted item definition and real saved occupants take precedence.
        self.default_equipment = self.configured_base_slot("bolt")
            .and_then(EquipmentSlot::from_native_index).map(|slot| (slot, "bolt"));
        let language = self.item_settings.as_ref()
            .and_then(|settings| settings.read_str("string_table", "language"))
            .unwrap_or("eng");
        self.currency_suffix = [language, "eng"].into_iter().find_map(|language| {
            let doc = load_xml_from_vfs(vfs, &format!("configs/text/{language}/ui_xrrazom.xml")).ok()?;
            let text = doc.navigate_with_attribute("string", "id", "st_xrr_currency")?
                .navigate("text", 0)?.read_text()?.trim();
            (!text.is_empty()).then(|| text.to_owned())
        }).unwrap_or_else(|| "RU".to_owned());
        if let Some(settings) = &self.item_settings {
            self.max_carry_weight_kg = settings.read_f32("inventory", "max_weight", self.max_carry_weight_kg);
            for item in &mut self.items {
                item.weight_kg = settings.read_f32(&item.section, "inv_weight", item.weight_kg);
            }
        }
        self.actor_profiles.clear();
        if let Ok(doc) = load_xml_from_vfs(vfs, "configs/gameplay/character_desc_general_actor.xml") {
            let profiles = doc.roots.iter().flat_map(|node| {
                std::iter::once(node)
                    .filter(|root| root.name == "specific_character")
                    .chain(node.children_named("specific_character"))
            });
            for profile in profiles {
                if let Some(id) = profile.attr("id")
                    && let Some(icon) = profile.navigate("icon", 0).and_then(XmlNode::read_text)
                {
                    self.actor_profiles.insert(id.to_owned(), icon.trim().to_owned());
                }
            }
        }
        self.community_ids = load_ltx_tree_from_vfs(vfs, "configs/creatures/game_relations.ltx", &[])
            .ok().and_then(|doc| {
                Some(doc.read_str("game_relations", "communities")?.split(',').step_by(2)
                    .map(|id| id.trim().to_owned()).collect())
            }).unwrap_or_default();
    }

    fn apply_item_metadata(&self, item: &mut InventoryItem) {
        if let Some(settings) = &self.item_settings {
            item.weight_kg = settings.read_f32(&item.section, "inv_weight", item.weight_kg);
        }
        if item.source_scoc.is_none() && let Some(max_uses) = self.configured_max_uses(&item.section) {
            item.remaining_uses = Some(max_uses);
        }
        if let Some((_, _, width, height)) = self.grid_index.as_ref().and_then(|index| index.get(&item.section)) {
            item.grid_size = (width.max(1), height.max(1));
        }
    }

    pub(crate) fn resolve_inventory_icons(&self, grid: &mut ItemGridIndex) {
        let Some(settings) = &self.item_settings else { return };
        for section in settings.sections.keys() {
            let layout = || {
                let x = settings.read_str(section, "inv_grid_x")?.parse::<u32>().ok()?;
                let y = settings.read_str(section, "inv_grid_y")?.parse::<u32>().ok()?;
                let width = settings.read_str(section, "inv_grid_width")?.parse::<u8>().ok()?;
                let height = settings.read_str(section, "inv_grid_height")?.parse::<u8>().ok()?;
                (width > 0 && height > 0).then_some((x, y, width, height))
            };
            if let Some(rect) = layout() {
                grid.entries.insert(section.to_ascii_lowercase(), rect);
            }
        }
    }

    fn carry_limit_kg(&self) -> f32 {
        let Some(settings) = &self.item_settings else { return self.max_carry_weight_kg };
        let gear_bonus: f32 = self.items.iter().filter_map(|item| {
            let (key, multiplier) = match item.equipped_slot? {
                // Anomaly's scripted actor menu uses GetAdditionalMaxWeight for equipment.
                EquipmentSlot::Armor | EquipmentSlot::Backpack if self.native_inventory.is_some() => {
                    ("additional_inventory_weight2", 1.0)
                }
                EquipmentSlot::Armor | EquipmentSlot::Backpack => ("additional_inventory_weight", 1.0),
                EquipmentSlot::Artifact(_) => {
                    let condition = if self.native_inventory.is_some() { 1.0 } else { item.condition };
                    ("additional_inventory_weight", condition)
                }
                _ => return None,
            };
            Some(settings.read_f32(&item.section, key, 0.0) * multiplier)
        }).sum();
        (self.max_carry_weight_kg + gear_bonus).max(1.0)
    }

    fn actor_community(&self) -> Option<&str> {
        let index = usize::try_from(self.actor_stats.community_index).ok()?;
        self.community_ids.get(index).map(String::as_str)
    }

    pub fn layout(&self, width: u32, height: u32) -> InventoryLayout {
        let visible_rows = self.grid_rows.min(DEFAULT_GRID_ROWS);
        let mut layout = InventoryLayout::compute(width, height, self.grid_cols, visible_rows);
        if self.native_inventory.is_some() {
            layout.slot_rects.retain(|(slot, _)| *slot != EquipmentSlot::SecondaryWeapon);
            layout.slot_rects.extend([1, 4, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 18]
                .into_iter().map(|id| (EquipmentSlot::NativeSlot(id), [0.0; 4])));
        }
        let viewport = UiViewport::new(width as f32, height as f32);
        if let Some(xml) = self.ui_layouts[usize::from(viewport.is_widescreen())].as_ref() {
            layout.apply_xml(xml, viewport);
        }
        layout.grid_scroll_row = self.grid_scroll_row.min(self.max_grid_scroll(layout.grid_rows));
        layout.slot_rects.retain(|(_, rect)| rect[2] > 0.0 && rect[3] > 0.0);
        layout
    }

    pub(crate) fn fit_bag_to_viewport(&mut self, width: u32, height: u32) {
        let viewport = UiViewport::new(width as f32, height as f32);
        if let Some(xml) = self.ui_layouts[usize::from(viewport.is_widescreen())].as_ref()
            && let Some([_, _, bag_width, bag_height]) = InventoryLayout::native_bag_rect(xml, viewport)
        {
            let widest = self.items.iter().map(|item| item.grid_size.0).max().unwrap_or(1);
            let pitch = NATIVE_GRID_PITCH * viewport.scale()[1];
            let columns = ((bag_width / pitch).floor() as u8).max(widest);
            let visible_rows = (bag_height / pitch.min(bag_width / f32::from(columns))).floor() as u8;
            self.grid_rows = self.grid_rows.max(visible_rows);
            if columns != self.grid_cols {
                self.grid_cols = columns;
                self.grid_scroll_row = 0;
                self.drag = None;
                self.repack_bag();
            }
        }
    }

    fn max_grid_scroll(&self, visible_rows: u8) -> u8 {
        self.items.iter().filter(|item| self.matches_bag_filter(item)).filter_map(|item| {
            let (_, row) = item.grid_pos?;
            Some(row.saturating_add(item.grid_size.1))
        }).max().unwrap_or(0).saturating_sub(visible_rows)
    }

    fn matches_bag_filter(&self, item: &InventoryItem) -> bool {
        let utility = item.category == ItemCategory::Misc
            && ["tool", "repair", "kit"].iter().any(|word| item.section.contains(word));
        match self.bag_filter {
            1 => item.category == ItemCategory::Weapon,
            2 => matches!(item.category, ItemCategory::Outfit | ItemCategory::Helmet | ItemCategory::Backpack),
            3 => item.category == ItemCategory::Ammo,
            4 => item.category == ItemCategory::Artifact,
            5 => utility,
            6 => matches!(item.category, ItemCategory::Medical | ItemCategory::Consumable),
            7 => item.category == ItemCategory::Misc && !utility,
            _ => true,
        }
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
        self.stack_picker = None;
        self.open = open;
        if !open {
            self.drag = None;
            self.grid_scroll_row = 0;
            self.bag_filter = 0;
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

    pub(crate) fn restore_saved_item_uses(&mut self, peer: &ScocPeerState) {
        self.item_uses_extra = peer.item_remaining_uses_extra.clone();
        for item in &mut self.items {
            if let Some(uses) = item.source_id.and_then(|id| peer.item_remaining_uses.get(&id)) {
                item.remaining_uses = Some(*uses);
            }
        }
        // Repack so differently used items from the same section retain separate cells.
        self.repack_bag();
    }

    /// Hydrate from extracted [`ScocPeerState`] (`inventory` + `item_remaining_uses`).
    pub fn load_from_scoc_peer_state(&mut self, peer: &ScocPeerState) {
        self.native_inventory = None;
        self.items.clear();
        self.stack_picker = None;
        self.drag = None;
        self.selected_uid = None;
        self.grid_scroll_row = 0;
        self.item_uses_extra = peer.item_remaining_uses_extra.clone();

        for raw in &peer.inventory {
            let uid = self.allocate_uid();
            let mut item = InventoryItem::from_scoc_item(uid, raw, &peer.item_remaining_uses);
            self.apply_item_metadata(&mut item);
            if let Some(slot) = item.equipped_slot
                && self.item_in_slot(slot).is_some()
            {
                item.equipped_slot = None;
            }
            if item.equipped_slot.is_none() {
                item.grid_pos = self.place_in_bag(&item);
            }
            self.items.push(item);
        }
        self.hud_sync_dirty = true;
    }

    /// Set or update the active `ItemGridIndex` and recompute item grid dimensions accordingly.
    pub fn set_grid_index(&mut self, index: Arc<ItemGridIndex>) {
        let mut changed = false;
        for item in &mut self.items {
            if let Some((_, _, gw, gh)) = index.get(&item.section) {
                let size = (gw.max(1), gh.max(1));
                changed |= item.grid_size != size;
                item.grid_size = size;
            }
        }
        self.grid_index = Some(index);
        if changed {
            self.repack_bag();
        }
    }

    fn repack_bag(&mut self) {
        self.stack_picker = None;
        for mut item in std::mem::take(&mut self.items) {
            if item.equipped_slot.is_none() {
                item.grid_pos = self.place_in_bag(&item);
            }
            self.items.push(item);
        }
    }

    pub(crate) fn set_item_picker_enabled(&mut self, enabled: bool) {
        if self.stack_all_sections != enabled {
            self.stack_all_sections = enabled;
            self.grid_scroll_row = 0;
            self.repack_bag();
        }
    }

    pub(crate) fn item_picker_enabled(&self) -> bool {
        self.stack_all_sections
    }

    /// Hydrate actor inventory from loaded `.scop` ALife entities (`parent_id == 0` or `parent_id == actor_id`).
    pub fn hydrate_from_alife(
        &mut self,
        alife: &AlifeManager,
        grid_index: Option<&ItemGridIndex>,
    ) {
        self.native_inventory = None;
        if let Some(actor) = alife.host_actor().and_then(|actor| actor.source_scop.as_ref())
            && self.hydrate_native_inventory(actor, alife.entities.values().filter_map(|entity| entity.source_scop.as_ref()), grid_index)
        {
            return;
        }
        self.specific_profile = alife.host_actor().and_then(|actor| {
            let source = actor.source_scop.as_ref()?;
            if let ScopAlifePayload::Actor(saved) = &source.payload {
                Some(saved.trader.specific_character.clone())
            } else {
                None
            }
        });
        let actor_id = alife.host_actor_id.unwrap_or(0);
        let owned_entities: Vec<_> = alife
            .entities
            .values()
            .filter(|ent| ent.parent_id == actor_id || (actor_id == 0 && ent.parent_id == 0))
            .collect();

        if owned_entities.is_empty() {
            return;
        }

        self.items.clear();
        self.stack_picker = None;
        self.drag = None;
        self.selected_uid = None;
        self.grid_scroll_row = 0;

        for ent in owned_entities {
            let uid = self.allocate_uid();
            let mut item = InventoryItem::from_section_indexed(uid, &ent.section, grid_index);
            item.source_id = Some(ent.id as i64);
            self.apply_item_metadata(&mut item);
            if let Some(source) = &ent.source_scop {
                match &source.payload {
                    ScopAlifePayload::InventoryItem(saved) => item.condition = saved.condition.clamp(0.0, 1.0),
                    ScopAlifePayload::Weapon(saved) => {
                        item.condition = saved.item.condition.clamp(0.0, 1.0);
                        item.ammo_elapsed = Some(i64::from(saved.ammo_elapsed));
                        item.ammo_type = Some(i64::from(saved.ammo_type));
                    }
                    _ => {}
                }
            }

            // Auto-equip weapon / suit / helmet / quick slots into empty slots
            match item.category {
                ItemCategory::Weapon => {
                    let meta = classify_section_meta(&item.section);
                    let is_pistol = meta.preferred_slot == Some(EquipmentSlot::Pistol);
                    if is_pistol && self.item_in_slot(EquipmentSlot::Pistol).is_none() {
                        item.equipped_slot = Some(EquipmentSlot::Pistol);
                    } else if !is_pistol && self.item_in_slot(EquipmentSlot::PrimaryWeapon).is_none() {
                        item.equipped_slot = Some(EquipmentSlot::PrimaryWeapon);
                    } else if !is_pistol && self.item_in_slot(EquipmentSlot::SecondaryWeapon).is_none() {
                        item.equipped_slot = Some(EquipmentSlot::SecondaryWeapon);
                    }
                }
                ItemCategory::Outfit => {
                    if self.item_in_slot(EquipmentSlot::Armor).is_none() {
                        item.equipped_slot = Some(EquipmentSlot::Armor);
                    }
                }
                ItemCategory::Helmet => {
                    if self.item_in_slot(EquipmentSlot::Helmet).is_none() {
                        item.equipped_slot = Some(EquipmentSlot::Helmet);
                    }
                }
                ItemCategory::Medical | ItemCategory::Consumable => {
                    for slot_idx in 0..QUICK_SLOT_COUNT {
                        let slot = EquipmentSlot::QuickSlot(slot_idx);
                        if self.item_in_slot(slot).is_none() {
                            item.equipped_slot = Some(slot);
                            break;
                        }
                    }
                }
                _ => {}
            }

            if item.equipped_slot.is_none() {
                item.grid_pos = self.place_in_bag(&item);
            }

            self.items.push(item);
        }

        self.hud_sync_dirty = true;
    }

    fn hydrate_native_inventory<'a>(
        &mut self,
        actor: &ScopAlifeEntity,
        entities: impl Iterator<Item = &'a ScopAlifeEntity>,
        grid_index: Option<&ItemGridIndex>,
    ) -> bool {
        let Some(native) = actor.actor_inventory_client_state() else {
            return false;
        };
        self.native_inventory = Some(native);
        self.specific_profile = match &actor.payload {
            ScopAlifePayload::Actor(state) => Some(state.trader.specific_character.clone()),
            _ => None,
        };
        self.items.clear();
        self.drag = None;
        self.selected_uid = None;
        self.grid_scroll_row = 0;

        let mut items = Vec::new();
        let mut belt_slot = 0;
        for entity in entities.filter(|entity| entity.id() != actor.id() && entity.parent_id() == actor.id()) {
            let mut item = InventoryItem::from_section_indexed(self.allocate_uid(), entity.section(), grid_index);
            item.source_id = Some(i64::from(entity.id()));
            self.apply_item_metadata(&mut item);
            if self.item_settings.as_ref().and_then(|settings| settings.read_str(&item.section, "class"))
                .is_some_and(|class| matches!(class, "II_FOOD" | "II_MEDKI" | "II_ANTIR" | "II_BANDG" | "II_BOTTL" | "S_FOOD" | "S_MEDKI"))
                && let Some(uses) = entity.eatable_item_remaining_uses()
            {
                item.remaining_uses = Some(i64::from(uses));
            }
            if let ScopAlifePayload::Weapon(weapon) = &entity.payload {
                item.has_upgrades = !weapon.item.upgrades.is_empty();
                item.ammo_elapsed = Some(i64::from(weapon.ammo_elapsed));
                item.ammo_type = Some(i64::from(weapon.ammo_type));
            }
            if let ScopAlifePayload::InventoryItem(saved) = &entity.payload {
                item.has_upgrades = !saved.upgrades.is_empty();
            }
            if item.category == ItemCategory::Ammo
                && let Ok(ammo) = AlifeItemAmmoState::parse(entity.spawn.version, &entity.spawn.state_payload)
            {
                item.ammo_elapsed = Some(i64::from(ammo.ammo_left));
                if let Some(settings) = &self.item_settings {
                    let box_size = settings.read_f32(&item.section, "box_size", 0.0);
                    if box_size > 0.0 {
                        item.weight_kg *= f32::from(ammo.ammo_left) / box_size;
                    }
                }
            }
            if let Some((place, condition)) = entity.inventory_item_client_state() {
                item.condition = condition;
                item.equipped_slot = match place {
                    ScopItemPlace::Slot(id) => EquipmentSlot::from_native_index(id),
                    ScopItemPlace::Belt if belt_slot < ARTIFACT_SLOT_COUNT => {
                        let slot = belt_slot;
                        belt_slot += 1;
                        Some(EquipmentSlot::Artifact(slot))
                    }
                    _ => None,
                };
            } else if let ScopAlifePayload::InventoryItem(saved) = &entity.payload {
                item.condition = saved.condition.clamp(0.0, 1.0);
            }
            items.push(item);
        }

        // Anomaly's sizekind sort uses width, height, configured kind order, and condition.
        items.sort_by_cached_key(|item| {
            let kind_order = self.item_settings.as_ref().and_then(|settings| {
                let kind = settings.read_str(&item.section, "kind")?;
                settings.read_str("item_kind_order", kind)?.parse::<u32>().ok()
            }).unwrap_or(u32::MAX);
            (std::cmp::Reverse(item.grid_size), kind_order, item.section.clone(),
                std::cmp::Reverse(item.condition.to_bits()))
        });
        for mut item in items {
            if item.equipped_slot.is_some_and(|slot| self.item_in_slot(slot).is_some()) {
                item.equipped_slot = None;
            }
            if item.equipped_slot.is_none() {
                item.grid_pos = self.place_in_bag(&item);
            }
            self.items.push(item);
        }
        self.hud_sync_dirty = true;
        true
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
        self.native_inventory = None;
        let grid_index = self.grid_index.clone();
        if let Some(actor) = scop.entities.iter().find(|entity| entity.kind() == ScopAlifeKind::Actor)
            && self.hydrate_native_inventory(actor, scop.entities.iter(), grid_index.as_deref())
        {
            return;
        }
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
            self.grid_scroll_row = 0;
            for ent in owned {
                let uid = self.allocate_uid();
                let mut item = InventoryItem::from_section(uid, ent.section().to_string());
                item.source_id = Some(i64::from(ent.id()));
                self.apply_item_metadata(&mut item);
                item.grid_pos = self.place_in_bag(&item);
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
        self.apply_item_metadata(&mut item);
        item.grid_pos = self.place_in_bag(&item);
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

    fn visible_bag_item_at(&self, col: u8, row: u8) -> Option<&InventoryItem> {
        self.item_at_grid_cell(col, row).filter(|item| self.matches_bag_filter(item))
    }

    /// Check whether a `w x h` rectangle at `(col, row)` fits within the grid without overlapping.
    pub fn can_place_at(&self, col: u8, row: u8, w: u8, h: u8, ignore_uid: Option<u64>) -> bool {
        if w == 0 || h == 0
            || col.checked_add(w).is_none_or(|end| end > self.grid_cols)
            || row.checked_add(h).is_none_or(|end| end > self.grid_rows)
        {
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
            let ox1 = ox.saturating_add(other.grid_size.0);
            let oy1 = oy.saturating_add(other.grid_size.1);
            let overlaps = x0 < ox1 && x1 > ox && y0 < oy1 && y1 > oy;
            if overlaps {
                return false;
            }
        }
        true
    }

    /// Find the first free top-left grid coordinate `(col, row)` for an item of size `(w, h)`.
    pub fn find_free_cell(&self, w: u8, h: u8, ignore_uid: Option<u64>) -> Option<(u8, u8)> {
        if w == 0 || h == 0 || w > self.grid_cols || h > self.grid_rows {
            return None;
        }
        let cols = usize::from(self.grid_cols);
        let mut occupied = vec![false; cols * usize::from(self.grid_rows)];
        for item in &self.items {
            if item.equipped_slot.is_some() || Some(item.uid) == ignore_uid {
                continue;
            }
            let Some((x, y)) = item.grid_pos else { continue };
            let end_col = x.saturating_add(item.grid_size.0).min(self.grid_cols);
            let end_row = y.saturating_add(item.grid_size.1).min(self.grid_rows);
            for row in y..end_row {
                let start = usize::from(row) * cols;
                occupied[start + usize::from(x.min(end_col))..start + usize::from(end_col)].fill(true);
            }
        }
        for row in 0..=self.grid_rows - h {
            for col in 0..=self.grid_cols - w {
                if (row..row + h).all(|y| {
                    let start = usize::from(y) * cols + usize::from(col);
                    !occupied[start..start + usize::from(w)].contains(&true)
                }) {
                    return Some((col, row));
                }
            }
        }
        None
    }

    fn find_or_grow_cell(&mut self, (w, h): (u8, u8), ignore_uid: Option<u64>) -> Option<(u8, u8)> {
        if let Some(cell) = self.find_free_cell(w, h, ignore_uid) {
            return Some(cell);
        }
        if w == 0 || h == 0 || w > self.grid_cols {
            return None;
        }
        let row = self.grid_rows;
        self.grid_rows = row.checked_add(h)?;
        Some((0, row))
    }

    fn configured_max_uses(&self, section: &str) -> Option<i64> {
        let uses = self.item_settings.as_ref()?.read_str(section, "max_uses")?.parse::<i64>().ok()?;
        (uses > 1).then_some(uses)
    }

    fn item_uses_condition(&self, item: &InventoryItem) -> bool {
        let Some(settings) = &self.item_settings else { return show_item_condition(item.category) };
        settings.read_bool(&item.section, "use_condition", false)
            || settings.read_str(&item.section, "class").map_or_else(
                || show_item_condition(item.category),
                condition_gear_class,
            )
    }

    fn counts_rounds(&self, item: &InventoryItem) -> bool {
        if let Some(settings) = &self.item_settings
            && let Some(class) = settings.read_str(&item.section, "class")
        {
            return matches!(class, "AMMO" | "AMMO_S") && !settings.read_bool(&item.section, "grenade_ammo", false);
        }
        item.category == ItemCategory::Ammo
    }

    fn compatible_stack(&self, left: &InventoryItem, right: &InventoryItem) -> bool {
        if left.section != right.section || left.grid_size != right.grid_size {
            return false;
        }
        if self.stack_all_sections {
            // UICellContainer::FindSimilar uses GetCell_SEC before any strict checks when
            // the item picker is enabled. Distinct objects remain separate inventory records.
            return true;
        }
        if let Some(settings) = &self.item_settings {
            let max_uses = self.configured_max_uses(&left.section).filter(|_| {
                settings.read_str(&left.section, "class").is_none_or(|class| class == "II_FOOD")
                    && self.item_uses_condition(left)
            });
            if settings.read_bool(&left.section, "dont_stack", false)
                || max_uses.is_some_and(|max_uses| {
                    left.remaining_uses.unwrap_or(max_uses) != max_uses
                        || right.remaining_uses.unwrap_or(max_uses) != max_uses
                })
            {
                return false;
            }
            // Anomaly's UICellContainer groups ordinary supplies by section.
            return !self.item_uses_condition(left) || (!left.has_upgrades && !right.has_upgrades
                && (max_uses.is_some()
                    || (left.condition - right.condition).abs() < 0.1));
        }
        matches!(left.category, ItemCategory::Ammo | ItemCategory::Medical | ItemCategory::Consumable)
            && (left.condition - right.condition).abs() < 0.01
            && left.remaining_uses == right.remaining_uses
            && (left.category == ItemCategory::Ammo || left.ammo_elapsed == right.ammo_elapsed)
            && left.ammo_type == right.ammo_type
            && left.ammo_section == right.ammo_section
    }

    fn place_in_bag(&mut self, item: &InventoryItem) -> Option<(u8, u8)> {
        self.place_in_bag_ignoring(item, None)
    }

    fn place_in_bag_ignoring(&mut self, item: &InventoryItem, ignore_uid: Option<u64>) -> Option<(u8, u8)> {
        let mut visited = std::collections::BTreeSet::new();
        if self.ui_layouts.iter().any(Option::is_some)
            && let Some(cell) = self.items.iter().find_map(|other| {
                let cell = other.grid_pos?;
                (Some(other.uid) != ignore_uid && other.equipped_slot.is_none() && other.section == item.section
                    && visited.insert(cell) && self.compatible_stack(item, other)).then_some(cell)
            })
        {
            return Some(cell);
        }
        self.find_or_grow_cell(item.grid_size, ignore_uid)
    }

    fn visible_bag_stacks(&self) -> BTreeMap<(u8, u8), (&InventoryItem, u32)> {
        let mut stacks = BTreeMap::<_, (&InventoryItem, u32)>::new();
        for item in &self.items {
            if item.equipped_slot.is_some() || !self.matches_bag_filter(item)
                || self.drag.as_ref().is_some_and(|drag| drag.item_uid == item.uid)
            {
                continue;
            }
            let Some(pos) = item.grid_pos else { continue };
            let stack = stacks.entry(pos).or_insert((item, 0));
            let count = if self.counts_rounds(item) {
                item.ammo_elapsed.and_then(|rounds| u32::try_from(rounds).ok()).unwrap_or(1)
            } else {
                1
            };
            stack.1 = stack.1.saturating_add(count.saturating_mul(item.quantity.max(1)));
        }
        stacks
    }

    fn stack_members(&self, cell: (u8, u8)) -> impl Iterator<Item = &InventoryItem> {
        self.items.iter().filter(move |item| item.equipped_slot.is_none() && item.grid_pos == Some(cell))
    }

    fn picker_cells(&self, layout: &InventoryLayout) -> Option<StackPickerCells<'_>> {
        let (cell, scroll) = self.stack_picker?;
        let rect = layout.picker_rect?;
        let [x, y, w, h] = rect;
        let mut members: Vec<_> = self.stack_members(cell).collect();
        if members.len() < 2 {
            return None;
        }
        if self.counts_rounds(members[0]) {
            members.sort_by_key(|item| std::cmp::Reverse(item.ammo_elapsed));
        } else {
            members.sort_by(|a, b| b.has_upgrades.cmp(&a.has_upgrades).then_with(|| b.condition.total_cmp(&a.condition)));
        }
        let (width, height) = members[0].grid_size;
        let tile_width = (f32::from(width.max(1)) * layout.cell_size).min(w);
        let tile_height = (f32::from(height.max(1)) * layout.cell_size).min(h);
        let columns = (w / tile_width).floor() as usize;
        let visible_rows = (h / tile_height).floor() as usize;
        let max_scroll = members.len().div_ceil(columns).saturating_sub(visible_rows);
        let cells = members.into_iter().skip(scroll.min(max_scroll) * columns)
            .take(visible_rows * columns).enumerate().map(|(index, item)| {
                let left = x + (index % columns) as f32 * tile_width;
                let top = y + (index / columns) as f32 * tile_height;
                ([left, top, tile_width, tile_height], item)
            }).collect();
        Some(StackPickerCells { rect, cells, max_scroll })
    }

    /// Move an item (either from another grid cell or unequipping a slot) to backpack cell `(col, row)`.
    pub fn move_item_to_grid(&mut self, uid: u64, col: u8, row: u8) -> bool {
        let Some(size) = self.item(uid).map(|i| i.grid_size) else {
            return false;
        };
        if !self.can_place_at(col, row, size.0, size.1, Some(uid)) {
            return false;
        }
        let was_hud_slot = self
            .item(uid)
            .and_then(|i| i.equipped_slot)
            .is_some_and(|s| s == EquipmentSlot::Armor || s.hud_slot_number().is_some());
        if let Some(it) = self.item_mut(uid) {
            it.equipped_slot = None;
            it.grid_pos = Some((col, row));
        }
        if was_hud_slot {
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
            EquipmentSlot::NativeSlot(1 | 4) => category == ItemCategory::Weapon,
            EquipmentSlot::NativeSlot(_) => false,
            EquipmentSlot::Artifact(idx) => {
                idx < ARTIFACT_SLOT_COUNT && category == ItemCategory::Artifact
            }
            EquipmentSlot::QuickSlot(idx) => {
                idx < QUICK_SLOT_COUNT
                    && matches!(category, ItemCategory::Medical | ItemCategory::Consumable)
            }
        }
    }

    fn configured_base_slot(&self, section: &str) -> Option<u8> {
        let index = self.item_settings.as_ref()?.read_str(section, "slot")?.parse::<i16>().ok()?;
        u8::try_from(index.checked_add(1)?).ok()
    }

    fn can_equip(&self, item: &InventoryItem, slot: EquipmentSlot) -> bool {
        if self.native_inventory.is_some()
            && let (Some(base), Some(target)) = (self.configured_base_slot(&item.section), slot.native_index())
        {
            // UIInventory.slot_cell permits weapon swaps between these native slots.
            return base != 0 && (target == base || matches!((base, target), (2, 1 | 3) | (3, 2)));
        }
        Self::is_compatible(item.category, slot)
    }

    /// Equip an item into `slot`, swapping any existing occupant back into the backpack grid.
    pub fn equip_item(
        &mut self,
        uid: u64,
        slot: EquipmentSlot,
        hud_weapon: Option<&mut HudWeaponController>,
        audio: Option<&mut AudioManager>,
    ) -> bool {
        let Some(item) = self.item(uid) else {
            return false;
        };
        if !self.can_equip(item, slot) {
            return false;
        }
        let section = item.section.clone();
        if let EquipmentSlot::QuickSlot(index) = slot && let Some(native) = &mut self.native_inventory {
            native.quick_slots[index] = section;
            return true;
        }

        // Place the old occupant in the backpack without counting the item leaving it.
        if let Some(previous) = self.item_in_slot(slot).filter(|item| item.uid != uid).cloned() {
            let Some(free_cell) = self.place_in_bag_ignoring(&previous, Some(uid)) else { return false };
            if let Some(prev) = self.item_mut(previous.uid) {
                prev.equipped_slot = None;
                prev.grid_pos = Some(free_cell);
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
        let Some(item) = self.item_in_slot(slot).cloned() else {
            return false;
        };
        let Some(free) = self.place_in_bag(&item) else {
            return false;
        };
        if let Some(it) = self.item_mut(item.uid) {
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
            aud.play_item_use(&item.section);
        }
        true
    }

    /// Push equipped `PrimaryWeapon` (slot 3), `SecondaryWeapon` (slot 4), and `Pistol` (slot 2)
    /// into [`HudWeaponController`], updating OGF paths, ammo counts, and weapon stats.
    pub fn sync_hud_weapon(&mut self, hw: &mut HudWeaponController) {
        hw.set_outfit_section(self.item_in_slot(EquipmentSlot::Armor).map(|item| item.section.as_str()));
        let slot_mappings = [
            (EquipmentSlot::Pistol, 2usize, "wpn_pm"),
            (EquipmentSlot::PrimaryWeapon, 3usize, "wpn_ak74"),
            (EquipmentSlot::SecondaryWeapon, 4usize, "wpn_mp5"),
        ];
        if let Some(knife) = self.item_in_slot(EquipmentSlot::NativeSlot(1)) {
            hw.slots[0] = build_weapon_spec_for_section(1, &knife.section, knife.ammo_elapsed);
        }
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
            let preferred = if let Some(base) = self.configured_base_slot(&item.section)
                && self.native_inventory.is_some()
            {
                EquipmentSlot::from_native_index(base)
            } else {
                classify_section_meta(&item.section).preferred_slot
            };
            if let Some(pref_slot) = preferred {
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
        } else if let Some(item) = self.item(uid)
            && let Some(cell) = item.grid_pos
            && self.items.iter().any(|other| {
                other.uid != uid && other.grid_pos == Some(cell) && !self.compatible_stack(item, other)
            })
        {
            let size = item.grid_size;
            if let Some(new_cell) = self.find_or_grow_cell(size, Some(uid))
                && let Some(item) = self.item_mut(uid)
            {
                item.grid_pos = Some(new_cell);
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
        let uid = if let Some(native) = &self.native_inventory {
            let section = native.quick_slots.get(slot_idx)?;
            self.items.iter().find(|item| item.section == *section)?.uid
        } else {
            self.item_in_slot(EquipmentSlot::QuickSlot(slot_idx))?.uid
        };
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
        let mut layout = self.layout(viewport_w, viewport_h);
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

        if let Some(picker) = self.picker_cells(&layout) {
            if point_in_rect(mx, my, picker.rect) {
                let wheel = input.wheel_delta();
                if wheel != 0.0 {
                    let max_scroll = picker.max_scroll;
                    if let Some((_, scroll)) = &mut self.stack_picker {
                        *scroll = if wheel > 0.0 { scroll.saturating_sub(1) } else { (*scroll + 1).min(max_scroll) };
                    }
                    return;
                }
                if let Some((_, item)) = picker.cells.iter().find(|(rect, _)| point_in_rect(mx, my, *rect)) {
                    let uid = item.uid;
                    if input.is_mouse_just_pressed(MouseButton::Right) {
                        self.selected_uid = Some(uid);
                        let _ = self.use_item(uid, hud_weapon, alife, audio);
                        return;
                    }
                    if input.is_mouse_just_pressed(MouseButton::Left) {
                        let origin_cell = item.grid_pos;
                        self.selected_uid = Some(uid);
                        self.drag = Some(DragState {
                            item_uid: uid, start_pointer: [mx, my], current_pointer: [mx, my],
                            origin_cell, origin_slot: None,
                        });
                        return;
                    }
                }
            } else if input.is_mouse_just_pressed(MouseButton::Left) && !point_in_rect(mx, my, layout.use_button_rect) {
                self.stack_picker = None;
            }
        }

        if input.is_mouse_just_pressed(MouseButton::Left)
            && let Some(&(index, _, _)) = layout.filter_buttons.iter().find(|(_, rect, _)| point_in_rect(mx, my, *rect))
        {
            self.bag_filter = index;
            self.grid_scroll_row = 0;
            self.selected_uid = None;
            self.drag = None;
            return;
        }

        let scroll_limit = self.max_grid_scroll(layout.grid_rows);
        if scroll_limit > 0 {
            let track = layout.scrollbar_rect();
            if input.is_mouse_just_pressed(MouseButton::Left) && point_in_rect(mx, my, track) {
                let fraction = ((my - track[1]) / track[3]).clamp(0.0, 1.0);
                self.grid_scroll_row = (f32::from(scroll_limit) * fraction).round() as u8;
                return;
            }
            let wheel = input.wheel_delta();
            if wheel != 0.0 && (point_in_rect(mx, my, layout.grid_rect()) || point_in_rect(mx, my, track)) {
                let rows = (wheel.abs().ceil() as u8).max(1);
                self.grid_scroll_row = if wheel > 0.0 {
                    layout.grid_scroll_row.saturating_sub(rows)
                } else {
                    layout.grid_scroll_row.saturating_add(rows).min(scroll_limit)
                };
                layout.grid_scroll_row = self.grid_scroll_row;
            }
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
                && let Some(uid) = self.visible_bag_item_at(col, row).map(|i| i.uid)
            {
                let _ = self.use_item(uid, hud_weapon, alife, audio);
                return;
            }
        }

        // Left press -> Close button, Use Selected button, or begin Drag
        if input.is_mouse_just_pressed(MouseButton::Left) {
            if layout.close_button_rect[2] > 0.0 && point_in_rect(mx, my, layout.close_button_rect) {
                let animated = hud_weapon.as_deref_mut().zip(audio.as_deref_mut())
                    .is_some_and(|(hud, audio)| hud.close_backpack(audio));
                self.set_open(false, if animated { None } else { audio });
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
                && let Some(item) = self.visible_bag_item_at(col, row)
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

            if moved_dist < 5.0 && self.stack_all_sections
                && let Some(cell) = drag.origin_cell
                && self.stack_picker.is_none_or(|(selected, _)| selected != cell)
                && let Some(item) = self.item(drag.item_uid)
                && (self.item_uses_condition(item) || self.counts_rounds(item))
                && self.stack_members(cell).nth(1).is_some()
            {
                self.stack_picker = Some((cell, 0));
            }

            if moved_dist >= 5.0 {
                if let Some(target_slot) = layout.hit_slot(mx, my) {
                    let _ = self.equip_item(
                        drag.item_uid,
                        target_slot,
                        hud_weapon.as_deref_mut(),
                        audio.as_deref_mut(),
                    );
                } else if let Some((col, row)) = layout.hit_grid_cell(mx, my) {
                    if self.stack_all_sections {
                        if let Some(slot) = drag.origin_slot {
                            let _ = self.unequip_slot(slot, hud_weapon, audio);
                        }
                        return;
                    }
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
        inventory_fonts: &BTreeMap<&str, UiFontResource>,
        equipment_tex: Option<DescriptorIndex>,
        grid_tex: Option<DescriptorIndex>,
        atlas: Option<&UiTextureAtlasMap>,
        vfs: Option<&VirtualFileSystem>,
    ) -> u32 {
        let mut drawn = 0u32;
        let fw = (width.max(64)) as f32;
        let fh = (height.max(64)) as f32;

        // The first-person overlay yields to the actor menu and backpack transitions.
        if !self.open && !hud_weapon.is_some_and(HudWeaponController::backpack_blocks_input) {
            if show_crosshair && !self.is_modal_open() {
                renderer.ui_fill_rect([fw * 0.5 - 1.0, fh * 0.5 - 1.0, 2.0, 2.0], [0.95; 4]);
                drawn += 1;
            }
            let scale = (fh / 768.0).max(0.5);
            let y = fh - 36.0 * scale;
            let meter = |row: f32| [16.0 * scale, y + row * 8.0 * scale, 184.0 * scale, 3.0 * scale];
            drawn += draw_segments(renderer, meter(0.0), 32, self.actor_stats.health, [0.83, 0.70, 0.44, 0.96]);
            drawn += draw_segments(renderer, meter(1.0), 32, stamina, [0.46, 0.76, 0.88, 0.96]);
            if self.actor_stats.bleeding > 0.02 {
                drawn += draw_segments(renderer, meter(2.0), 32, self.actor_stats.bleeding, [0.94, 0.22, 0.13, 0.95]);
            }
            if self.actor_stats.radiation > 0.02 {
                drawn += draw_segments(renderer, meter(3.0), 32, self.actor_stats.radiation, [0.91, 0.82, 0.23, 0.95]);
            }

            if let Some(hw) = hud_weapon && let Some(f) = font {
                let spec = hw.active_weapon();
                if spec.clip_capacity > 0 && !spec.section.starts_with("unequipped_") && !spec.section.is_empty() {
                    let mode = match spec.fire_mode {
                        WeaponFireMode::FullAuto => "A",
                        WeaponFireMode::SemiAuto | WeaponFireMode::BoltAction => "1",
                        WeaponFireMode::Melee => "",
                    };
                    let x = fw - 68.0 * scale;
                    let color = [0.94, 0.77, 0.36, 1.0];
                    drawn += f.draw_text(renderer, &spec.ammo_in_clip.to_string(), [x, fh - 52.0 * scale], 14.0 * scale, color);
                    drawn += f.draw_text(renderer, mode, [x - 36.0 * scale, y], 12.0 * scale, color);
                    let fallback_ammo = self.item_settings.as_ref().and_then(|settings| {
                        let ammo = settings.read_str(&spec.section, "ammo_class")?;
                        ammo.split(',').next().map(str::trim)
                    });
                    if let (Some(texture), Some(section)) = (equipment_tex, fallback_ammo)
                        && self.grid_index.as_ref().is_some_and(|index| index.get(section).is_some())
                    {
                        let uv = item_icon_uv_indexed(section, ItemCategory::Ammo, self.grid_index.as_deref());
                        renderer.ui_draw_textured_rect([x, y - 2.0 * scale, 43.0 * scale, 22.0 * scale], uv, texture, [1.0; 4]);
                        drawn += 1;
                    }
                }
            }
        }
        // Center interaction prompt when looking at NPC/Trader/Item
        if let Some(target) = &self.hover_target
            && !self.is_modal_open()
            && !hud_weapon.is_some_and(HudWeaponController::backpack_blocks_input)
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
        let layout = self.layout(width, height);
        if self.open {
            renderer.ui_fill_rect([0.0, 0.0, fw, fh], [0.01, 0.01, 0.02, 0.40]);
            drawn += 1;
            if layout.backgrounds.is_empty() {
                renderer.ui_fill_rect(layout.panel_rect, [0.08, 0.09, 0.10, 0.92]);
                drawn += 1;
            }
            for (rect, texture) in &layout.backgrounds {
                if !draw_atlas_region(renderer, vfs, atlas, texture, *rect) {
                    renderer.ui_fill_rect(*rect, [0.08, 0.09, 0.10, 0.94]);
                }
                drawn += 1;
            }
            let community = self.actor_community();
            let portrait = self.specific_profile.as_ref().and_then(|id| self.actor_profiles.get(id))
                .or_else(|| community.and_then(|id| self.actor_profiles.get(id)))
                .or_else(|| self.actor_profiles.get(&self.actor_stats.profile_name));
            if let (Some(icon), Some(rect)) = (portrait, layout.portrait_rect) {
                drawn += u32::from(draw_atlas_region(renderer, vfs, atlas, icon, rect));
            }
            if let Some(name) = &layout.name {
                drawn += name.draw(renderer, font, inventory_fonts, &self.actor_stats.actor_name, [0.94, 0.88, 0.74, 1.0]);
            }
            if let (Some(id), Some(field)) = (community, &layout.faction) {
                let faction = id.strip_prefix("actor_").unwrap_or(id);
                let label = match faction {
                    "stalker" => "Loner".to_owned(),
                    "csky" => "Clear Sky".to_owned(),
                    "dolg" => "Duty".to_owned(),
                    "ecolog" => "Ecologist".to_owned(),
                    "killer" => "Mercenary".to_owned(),
                    id => prettify_section(id),
                };
                drawn += field.draw(renderer, font, inventory_fonts, &label, [0.72, 0.72, 0.68, 1.0]);
            }
            if let Some(rect) = layout.health_rect {
                drawn += draw_segments(renderer, rect, 50, self.actor_stats.health, [0.52, 0.52, 0.49, 0.92]);
            }
            let filter_labels = ["All", "Arms", "Gear", "Ammo", "Arts", "Tools", "Use", "Misc"];
            for (index, rect, texture) in &layout.filter_buttons {
                if !draw_atlas_region(renderer, vfs, atlas, texture, *rect)
                    && !draw_atlas_region(renderer, vfs, atlas, &format!("{texture}_e"), *rect)
                {
                    renderer.ui_fill_rect(*rect, [0.11, 0.12, 0.12, 0.9]);
                    if let Some(f) = font {
                        drawn += f.draw_text(renderer, filter_labels[*index], [rect[0] + 2.0, rect[1] + 5.0],
                            9.0, [0.82, 0.82, 0.77, 1.0]);
                    }
                }
                drawn += 1;
                if *index == self.bag_filter {
                    renderer.ui_fill_rect([rect[0], rect[1] + rect[3] - 2.0, rect[2], 2.0], [0.67, 0.64, 0.46, 1.0]);
                    drawn += 1;
                }
            }
            if layout.close_button_rect[2] > 0.0 {
                renderer.ui_fill_rect(layout.close_button_rect, [0.23, 0.24, 0.22, 0.95]);
                drawn += 1;
                if let Some(f) = font {
                    let [x, y, _, _] = layout.close_button_rect;
                    drawn += f.draw_text(renderer, "X", [x + 8.0, y + 4.0], 14.0, [0.92, 0.90, 0.84, 1.0]);
                }
            }

            // Equipment Slots
            let native_icon_size = layout.bag_viewport.map(|_| NATIVE_ICON_SIZE * fh / 768.0);
            for (slot, rect) in &layout.slot_rects {
                if let EquipmentSlot::QuickSlot(index) = slot && let Some(native) = &self.native_inventory {
                    let section = &native.quick_slots[*index];
                    if let Some(tex) = equipment_tex
                        && let Some((_, _, w, h)) = self.grid_index.as_ref().and_then(|icons| icons.get(section))
                        && w > 0 && h > 0
                    {
                        let uv = item_icon_uv_indexed(section, ItemCategory::Consumable, self.grid_index.as_deref());
                        let icon = native_icon_size.map_or_else(
                            || fit_inventory_icon(*rect, f32::from(w) / f32::from(h)),
                            |size| native_equipment_icon(*rect, size, w, h),
                        );
                        let opacity = if self.items.iter().any(|item| item.section == *section) { 1.0 } else { 0.35 };
                        renderer.ui_draw_textured_rect(icon, uv, tex, [1.0, 1.0, 1.0, opacity]);
                        drawn += 1;
                    }
                }
                let occ = self.item_in_slot(*slot);
                if occ.is_none()
                    && let Some((default_slot, section)) = self.default_equipment
                    && default_slot == *slot
                    && let (Some(size), Some(tex), Some(icons)) = (native_icon_size, equipment_tex, self.grid_index.as_deref())
                    && let Some((_, _, w, h)) = icons.get(section)
                {
                    let icon = native_equipment_icon(*rect, size, w, h);
                    let uv = item_icon_uv_indexed(section, ItemCategory::Misc, Some(icons));
                    renderer.ui_draw_textured_rect(icon, uv, tex, [1.0; 4]);
                    drawn += 1;
                }
                let is_sel = occ.is_some_and(|i| Some(i.uid) == self.selected_uid);
                if is_sel || layout.backgrounds.is_empty() {
                    let bg = if is_sel {
                        [0.24, 0.26, 0.20, 0.95]
                    } else {
                        [0.12, 0.13, 0.15, 0.92]
                    };
                    renderer.ui_fill_rect(*rect, bg);
                    drawn += 1;
                }

                if let Some(item) = occ
                    && self.drag.as_ref().is_none_or(|d| d.item_uid != item.uid)
                {
                    let inner = [rect[0] + 4.0, rect[1] + 4.0, (rect[2] - 8.0).max(4.0), (rect[3] - 12.0).max(4.0)];
                    if let Some(tex) = equipment_tex {
                        let uv = item_icon_uv_indexed(&item.section, item.category, self.grid_index.as_deref());
                        let rotated = layout.rotated_slots.contains(slot);
                        let (w, h) = item.grid_size;
                        let (icon_w, icon_h) = if rotated { (h, w) } else { (w, h) };
                        let aspect = if rotated { f32::from(h) / f32::from(w.max(1)) }
                            else { f32::from(w) / f32::from(h.max(1)) };
                        let icon = native_icon_size.map_or_else(
                            || fit_inventory_icon(inner, aspect),
                            |size| native_equipment_icon(*rect, size, icon_w, icon_h),
                        );
                        if rotated {
                            renderer.ui_draw_rotated_rect(icon, uv, tex, [1.0; 4]);
                        } else {
                            renderer.ui_draw_textured_rect(icon, uv, tex, [1.0; 4]);
                        }
                    } else {
                        renderer.ui_fill_rect(inner, item.category.accent_color());
                    }
                    drawn += 1;
                    if self.item_uses_condition(item) {
                        let bar = layout.condition_rects.get(slot).copied()
                            .unwrap_or([rect[0] + 4.0, rect[1] + rect[3] - 5.0, (rect[2] - 8.0).max(4.0), 3.0]);
                        let power_bar = matches!(slot, EquipmentSlot::NativeSlot(8..=10));
                        let color = if power_bar { [0.34, 0.77, 0.82, 1.0] } else { condition_bar_color(item.condition) };
                        drawn += draw_segments(renderer, bar, 18, item.condition, color);
                    }
                }
                if let EquipmentSlot::QuickSlot(index) = slot && let Some(f) = font {
                    drawn += f.draw_text(renderer, &format!("F{}", index + 1),
                        [rect[0], rect[1] + rect[3] - 11.0], 10.0, [0.91, 0.90, 0.85, 1.0]);
                }
            }

            // 2D Backpack Grid cells
            for row in 0..layout.grid_rows {
                for col in 0..layout.grid_cols {
                    let cx = layout.grid_origin[0] + f32::from(col) * layout.cell_size;
                    let cy = layout.grid_origin[1] + f32::from(row) * layout.cell_size;
                    if let Some(g_tex) = grid_tex.filter(|_| layout.backgrounds.is_empty()) {
                        renderer.ui_draw_textured_rect(
                            [cx, cy, layout.cell_size, layout.cell_size],
                            [0.0, 0.0, 1.0, 1.0],
                            g_tex,
                            [0.85, 0.88, 0.92, 0.88],
                        );
                    } else {
                        renderer.ui_fill_rect(
                            [cx + 1.0, cy + 1.0, layout.cell_size - 2.0, layout.cell_size - 2.0],
                            [0.11, 0.12, 0.13, 0.43],
                        );
                    }
                    drawn += 1;
                }
            }

            // Backpack Items
            let [_, top, _, grid_height] = layout.grid_rect();
            let bottom = top + grid_height;
            for ((col, row), (item, count)) in self.visible_bag_stacks() {
                let ix = layout.grid_origin[0] + f32::from(col) * layout.cell_size + 2.0;
                let iy = top + (f32::from(row) - f32::from(layout.grid_scroll_row)) * layout.cell_size + 2.0;
                let iw = f32::from(item.grid_size.0) * layout.cell_size - 4.0;
                let ih = f32::from(item.grid_size.1) * layout.cell_size - 4.0;
                let clipped_top = iy.max(top);
                let clipped_bottom = (iy + ih).min(bottom);
                if iw <= 0.0 || clipped_bottom <= clipped_top {
                    continue;
                }
                let rect = [ix, clipped_top, iw, clipped_bottom - clipped_top];
                let is_sel = self.selected_uid.and_then(|uid| self.item(uid))
                    .is_some_and(|selected| selected.grid_pos == Some((col, row)));
                if is_sel {
                    renderer.ui_fill_rect(rect, [0.52, 0.45, 0.22, 0.8]);
                    drawn += 1;
                }
                if let Some(tex) = equipment_tex {
                    let [u0, v0, u1, v1] = item_icon_uv_indexed(&item.section, item.category, self.grid_index.as_deref());
                    let uv = [
                        u0, v0 + (v1 - v0) * (clipped_top - iy) / ih,
                        u1, v0 + (v1 - v0) * (clipped_bottom - iy) / ih,
                    ];
                    renderer.ui_draw_textured_rect(rect, uv, tex, [1.0, 1.0, 1.0, 0.98]);
                } else {
                    renderer.ui_fill_rect(rect, item.category.accent_color());
                }
                drawn += 1;
                let grouped = count > 1 && !self.counts_rounds(item);
                if self.item_uses_condition(item) && !(self.stack_all_sections && grouped)
                    && iy + ih - 4.0 >= top && iy + ih <= bottom
                {
                    drawn += draw_segments(
                        renderer,
                        [ix + 2.0, iy + ih - 4.0, (iw - 4.0).max(2.0), 2.5],
                        18, item.condition, condition_bar_color(item.condition),
                    );
                }
                if (count > 1 || (self.counts_rounds(item) && item.ammo_elapsed.is_some()))
                    && iy >= top && iy + 12.0 <= bottom && let Some(f) = font
                {
                    let text = format!("x{count}");
                    renderer.ui_fill_rect([ix, iy, 7.0 * text.len() as f32 + 3.0, 12.0], [0.10, 0.14, 0.09, 0.94]);
                    drawn += 1 + f.draw_text(renderer, &text, [ix + 2.0, iy], 11.0, [0.88, 0.91, 0.73, 1.0]);
                }
            }
            let scroll_limit = self.max_grid_scroll(layout.grid_rows);
            if scroll_limit > 0 {
                let [x, y, w, h] = layout.scrollbar_rect();
                let visible = f32::from(layout.grid_rows);
                let thumb = (h * visible / (visible + f32::from(scroll_limit))).max(12.0).min(h);
                let fraction = f32::from(layout.grid_scroll_row) / f32::from(scroll_limit);
                renderer.ui_fill_rect([x, y, w, h], [0.08, 0.09, 0.09, 0.92]);
                renderer.ui_fill_rect([x, y + fraction * (h - thumb), w, thumb], [0.46, 0.44, 0.36, 0.90]);
                drawn += 2;
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
                    let action = if matches!(sel.category, ItemCategory::Medical | ItemCategory::Consumable) {
                        "Use"
                    } else { "Equip" };
                    let [x, y, _, _] = layout.use_button_rect;
                    drawn += f.draw_text(renderer, action, [x + 8.0, y + 5.0], 14.0, [0.98, 0.96, 0.89, 1.0]);
                    let info = format!("{:.1} kg  •  {:.0}%", sel.weight_kg, sel.condition * 100.0);
                    drawn += f.draw_text(
                        renderer,
                        &sel.display_name,
                        [x + 3.0, y - 45.0],
                        17.0,
                        [0.96, 0.94, 0.87, 1.0],
                    );
                    drawn += f.draw_text(renderer, &info, [x + 3.0, y - 24.0], 13.0, [0.78, 0.78, 0.73, 1.0]);
                }
            }

            // Weight bar at bottom of grid
            let weight = self.total_weight_kg();
            let capacity = self.carry_limit_kg();
            let weight_frac = (weight / capacity.max(1.0)).clamp(0.0, 1.0);
            let weight_color = if weight > capacity {
                [0.88, 0.26, 0.18, 0.95]
            } else if weight + 10.0 > capacity {
                [0.82, 0.72, 0.38, 0.95]
            } else {
                [0.87, 0.85, 0.80, 0.95]
            };
            if let Some(field) = &layout.weight {
                drawn += field.draw(renderer, font, inventory_fonts, &format!("{weight:.1} kg"), weight_color);
                let muted = [0.72, 0.72, 0.68, 1.0];
                if let Some(caption) = &layout.weight_caption {
                    drawn += caption.draw(renderer, font, inventory_fonts, "Weight:", muted);
                }
                if let Some(max) = &layout.weight_max {
                    drawn += max.draw(renderer, font, inventory_fonts, &format!("(max {capacity:.0} kg)"), muted);
                }
            } else {
                renderer.ui_draw_meter_bar(
                    [
                        layout.grid_origin[0],
                        layout.panel_rect[1] + layout.panel_rect[3] - 26.0,
                        f32::from(self.grid_cols) * layout.cell_size,
                        10.0,
                    ],
                    weight_frac, [0.06, 0.06, 0.06, 0.9], weight_color,
                );
                drawn += 2;
            }
            if let Some(field) = &layout.money {
                let money = format!("{} {}", self.actor_stats.money, self.currency_suffix);
                drawn += field.draw(renderer, font, inventory_fonts, &money, [1.0; 4]);
            }

            if let Some(picker) = self.picker_cells(&layout) {
                renderer.ui_fill_rect(picker.rect, [0.06, 0.06, 0.06, 0.96]);
                drawn += 1;
                for ([x, y, w, h], item) in picker.cells {
                    let tint = if self.selected_uid == Some(item.uid) { [0.29, 0.32, 0.20, 0.95] }
                        else { [0.13, 0.13, 0.12, 0.9] };
                    renderer.ui_fill_rect([x, y, w - 1.0, h - 1.0], tint);
                    drawn += 1;
                    if let Some(tex) = equipment_tex {
                        let uv = item_icon_uv_indexed(&item.section, item.category, self.grid_index.as_deref());
                        renderer.ui_draw_textured_rect([x + 2.0, y + 2.0, w - 4.0, h - 4.0], uv, tex, [1.0; 4]);
                        drawn += 1;
                    }
                    if let Some(font) = font {
                        let number = if self.counts_rounds(item) { item.ammo_elapsed } else { item.remaining_uses };
                        if let Some(number) = number {
                            drawn += font.draw_text(renderer, &number.to_string(), [x + 2.0, y], 12.0, [0.88, 0.91, 0.73, 1.0]);
                        } else if self.item_uses_condition(item) {
                            drawn += draw_segments(renderer, [x + 2.0, y + h - 4.0, w - 4.0, 2.5], 18, item.condition, condition_bar_color(item.condition));
                        }
                    }
                }
            }

            // Floating Drag Ghost following mouse cursor
            if let Some(drag) = &self.drag
                && let Some(item) = self.item(drag.item_uid)
            {
                let gw = f32::from(item.grid_size.0) * layout.cell_size;
                let gh = f32::from(item.grid_size.1) * layout.cell_size;
                let drag_rect = [drag.current_pointer[0] - gw * 0.5, drag.current_pointer[1] - gh * 0.5, gw, gh];
                if let Some(tex) = equipment_tex {
                    let uv = item_icon_uv_indexed(&item.section, item.category, self.grid_index.as_deref());
                    renderer.ui_draw_textured_rect(drag_rect, uv, tex, [1.0, 1.0, 1.0, 0.85]);
                } else {
                    let mut ghost_col = item.category.accent_color();
                    ghost_col[3] = 0.72;
                    renderer.ui_fill_rect(drag_rect, ghost_col);
                }
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
    let template = if lower.contains("ump45") {
        &defaults[3]
    } else if lower.contains("svd") {
        &defaults[4]
    } else if lower.contains("mp5") || lower.contains("klin") || lower.contains("kiparis") {
        &defaults[3]
    } else if lower.contains("ak") || lower.contains("abakan") || lower.contains("lr300") || lower.contains("val") || lower.contains("m4") || lower.contains("fn2000") {
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
    spec.hands_hud_ogf = "meshes\\anomaly_weapons\\hands\\wpn_hand_00.ogf".to_string();
    if lower.contains("ump45") {
        spec.weapon_hud_ogf = "meshes\\anomaly_weapons\\wpn_ump45\\wpn_ump45_hud.ogf".to_string();
        spec.clip_capacity = 25;
        spec.rpm = 600.0;
    } else if lower.contains("ak74") {
        spec.weapon_hud_ogf = "meshes\\anomaly_weapons\\wpn_ak74\\wpn_ak74_hud.ogf".to_string();
    }
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
fn draw_atlas_region(
    renderer: &mut Dx12Renderer,
    vfs: Option<&VirtualFileSystem>,
    atlas: Option<&UiTextureAtlasMap>,
    id: &str,
    rect: [f32; 4],
) -> bool {
    let Some(region) = atlas.and_then(|atlas| atlas.get(id)) else { return false };
    let Ok(index) = crate::render_bridge::load_vfs_texture(renderer, vfs, &region.dds_vpath()) else {
        return false;
    };
    let Some(texture) = renderer.get_texture_by_index(index) else { return false };
    let uv = region.uv_rect(texture.width as f32, texture.height as f32);
    renderer.ui_draw_textured_rect(rect, uv, index, [1.0; 4]);
    true
}

fn draw_segments(
    renderer: &mut Dx12Renderer,
    [x, y, width, height]: [f32; 4],
    segments: u32,
    value: f32,
    color: [f32; 4],
) -> u32 {
    let pitch = width / segments as f32;
    let filled = (value.clamp(0.0, 1.0) * segments as f32).round() as u32;
    for segment in 0..filled {
        renderer.ui_fill_rect([x + pitch * segment as f32, y, pitch * 0.62, height], color);
    }
    filled
}

fn show_item_condition(category: ItemCategory) -> bool {
    matches!(category, ItemCategory::Weapon | ItemCategory::Outfit | ItemCategory::Helmet)
}

// Class IDs registered by the C++ factory for the IsWeapon, IsOutfit and IsHeadgear checks.
fn condition_gear_class(class: &str) -> bool {
    matches!(class,
        "E_STLK" | "EQU_STLK" | "E_HLMET" | "EQ_HLMET"
        | "WP_VINT" | "WP_AK74" | "WP_LR300" | "WP_HPSA" | "WP_PM" | "WP_SHOTG" | "WP_ASHTG"
        | "WP_BM16" | "WP_SVD" | "WP_SVU" | "WP_RG6" | "WP_RPG7" | "WP_VAL" | "WP_WALTH"
        | "WP_USP45" | "WP_GROZA" | "WP_KNIFE"
        | "W_VINT" | "W_AK74" | "W_LR300" | "W_HPSA" | "W_PM" | "W_SHOTGN" | "W_BM16"
        | "W_SVD" | "W_SVU" | "W_RG6" | "W_RPG7" | "W_VAL" | "W_WALTHR" | "W_USP45"
        | "W_GROZA" | "W_KNIFE"
    )
}

fn fit_inventory_icon([x, y, w, h]: [f32; 4], aspect: f32) -> [f32; 4] {
    let width = w.min(h * aspect);
    let height = h.min(w / aspect);
    [x + (w - width) * 0.5, y + (h - height) * 0.5, width, height]
}

fn native_equipment_icon([x, y, w, h]: [f32; 4], tile_size: f32, cols: u8, rows: u8) -> [f32; 4] {
    let [width, height] = [f32::from(cols.max(1)) * tile_size, f32::from(rows.max(1)) * tile_size];
    let scale = (w.max(tile_size) / width).min(h.max(tile_size) / height).min(1.0);
    let [width, height] = [width * scale, height * scale];
    [x + (w - width) * 0.5, y + (h - height) * 0.5, width, height]
}

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
