use crate::ui_inventory::UiFontResource;
use xray_formats::fonts::FontAlignment;
use xray_formats::vfs_bridge::{load_ui_xml_for_viewport_from_vfs, load_xml_from_vfs};
use xray_formats::xml_ui::{LocalizationTable, UiViewport, XmlDocument, XmlNode};
use xray_render_dx12::Dx12Renderer;
use xray_vfs::VirtualFileSystem;

pub(super) type Rect = [f32; 4];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum MainAction {
    Resume,
    Options,
    Quit,
}

impl MainAction {
    pub fn from_xml(node: &XmlNode) -> Option<Self> {
        match node.attr("name")? {
            "btn_ret" => Some(Self::Resume),
            "btn_options" => Some(Self::Options),
            "btn_quit" => Some(Self::Quit),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Resume => "Return to game",
            Self::Options => "Options",
            Self::Quit => "Exit to Windows",
        }
    }
}

pub(super) struct MainLayout {
    pub buttons: Vec<(MainAction, String)>,
    first: Rect,
    pitch: f32,
    focus_inset: [f32; 2],
    focus_size: [f32; 2],
    initial_y: f32,
    slider_track_height: f32,
    pub font_section: &'static str,
    pub text_color: [f32; 4],
}

impl MainLayout {
    pub fn read(
        xml: &XmlDocument,
        table: Option<&LocalizationTable>,
        single_player: bool,
    ) -> Option<Self> {
        let widget = xml.navigate("shniaga_wnd", 0)?;
        let region = widget.navigate("buttons_region", 0)?;
        let button_height = xml.navigate("button", 0)?.attr_f32("h", 0.0);
        let path = if single_player {
            "menu_main_single"
        } else {
            "menu_main_mm"
        };
        let list = xml.navigate(path, 0)?;
        let mut buttons = Vec::new();
        for node in list.children_named("btn") {
            if let Some(action) = MainAction::from_xml(node)
                && !buttons.iter().any(|(id, _)| *id == action)
            {
                buttons.push((
                    action,
                    localized(table, node.attr("caption"), action.label()),
                ));
            }
        }
        let interval = region.attr_f32("vert_interval", 0.0);
        let left = region.attr_f32("left_ident", 0.0);
        let right = region.attr_f32("right_ident", 0.0);
        let first = [
            widget.attr_f32("x", 0.0) + region.attr_f32("x", 0.0) + right,
            widget.attr_f32("y", 0.0)
                + region.attr_f32("y", 0.0)
                + region.attr_f32("top_indent", 0.0)
                + interval,
            region.attr_f32("width", 0.0) - left - right,
            button_height,
        ];
        if buttons.len() != 3 || !valid_rect(first) || !(interval.is_finite() && interval >= 0.0) {
            return None;
        }
        let slider = widget.navigate("shniaga", 0)?;
        let focus = slider.navigate("magnifire", 0)?;
        let focus_box = node_rect(focus, [slider.attr_f32("x", 0.0), 0.0]);
        let y_offset = focus
            .navigate("y_offset", 0)
            .and_then(|node| node.text.trim().parse::<f32>().ok())
            .filter(|value| value.is_finite())
            .unwrap_or(0.0);
        let focus_inset = [
            focus_box[0] + widget.attr_f32("x", 0.0) - first[0],
            y_offset,
        ];
        let initial_y = widget.attr_f32("y", 0.0) + slider.attr_f32("y", 0.0) + focus_box[1];
        let slider_track_height = widget.attr_f32("height", 0.0);
        if !valid_rect(focus_box) || !initial_y.is_finite() || slider_track_height <= 1.0 {
            return None;
        }
        let font_section = font_section(list.attr_or("font", "graffiti19"))
            .unwrap_or("ui_font_graffiti19_russian");
        let text_color = ["r", "g", "b", "a"]
            .map(|channel| list.attr_i32(channel, 255).clamp(0, 255) as f32 / 255.0);
        Some(Self {
            buttons,
            first,
            pitch: button_height + interval,
            focus_inset,
            focus_size: [focus_box[2], focus_box[3]],
            initial_y,
            slider_track_height,
            font_section,
            text_color,
        })
    }

    pub fn row(&self, row: usize) -> Rect {
        let mut rect = self.first;
        rect[1] += self.pitch * row as f32;
        rect
    }

    pub fn selector(&self, row: usize) -> Rect {
        let [x, y, _, _] = self.row(row);
        [
            x + self.focus_inset[0],
            y + self.focus_inset[1],
            self.focus_size[0],
            self.focus_size[1],
        ]
    }

    pub fn selector_start(&self) -> f32 {
        self.initial_y
    }

    pub fn selector_duration(&self, from: f32, to: f32) -> f32 {
        ((1.0 + (from - to).abs()).ln() / self.slider_track_height.ln() * 0.3).max(0.1)
    }
}

pub(super) struct OptionsLayout {
    pub panel: Rect,
    pub heading: Rect,
    pub category: Rect,
    pub area: Rect,
    pub status: [Rect; 2],
    pub rows: [Rect; 6],
    pub labels: [String; 4],
    pub notification: Rect,
    pub label_font: &'static str,
    pub button_font: &'static str,
    pub label_x: f32,
}

impl OptionsLayout {
    pub fn read(xml: &XmlDocument, table: Option<&LocalizationTable>) -> Option<Self> {
        let main = xml.navigate("main", 0)?;
        let origin = [main.attr_f32("x", 0.0), main.attr_f32("y", 0.0)];
        let rect = |name| Some(node_rect(main.navigate(name, 0)?, origin));
        let area = rect("scroll")?;
        let template = main.navigate("st", 0)?;
        let element_height = template.attr_f32("height", 0.0);
        let check = xml.navigate("elements:check", 0)?;
        let mut rows = [[0.0; 4]; 6];
        let mut status = [[0.0; 4]; 2];
        for row in 0..2 {
            let y = area[1] + row as f32 * element_height;
            rows[row] = [area[0], y, area[2], element_height];
            status[row] = node_rect(check, [area[0], y]);
        }
        let mut labels = ["Apply", "Reset", "Default", "Cancel"].map(str::to_owned);
        let paths = ["btn_accept", "btn_reset", "btn_default", "btn_cancel"];
        for (i, name) in paths.into_iter().enumerate() {
            rows[i + 2] = rect(name)?;
            let caption = main.navigate(name, 0)?.navigate("text", 0)?;
            labels[i] = localized(table, Some(caption.text.trim()), &labels[i]);
        }
        let panel = node_rect(main, [0.0, 0.0]);
        let heading = rect("cap_1")?;
        let mut category = rect("tree_1")?;
        category[3] = main.navigate("st_tree", 0)?.attr_f32("height", 0.0);
        let notification = rect("notify")?;
        if ![panel, heading, category, area, notification]
            .into_iter()
            .chain(rows)
            .chain(status)
            .all(valid_rect)
        {
            return None;
        }
        let caption = xml.navigate("elements:cap", 0)?;
        Some(Self {
            panel,
            heading,
            category,
            area,
            status,
            rows,
            labels,
            notification,
            label_font: font_section(caption.navigate("text", 0)?.attr_or("font", "letterica16"))
                .unwrap_or("ui_font_letterica16_russian"),
            button_font: font_section(
                main.navigate("btn_accept:text", 0)?
                    .attr_or("font", "letterica18"),
            )
            .unwrap_or("ui_font_letterica18_russian"),
            label_x: caption.attr_f32("x", 0.0),
        })
    }
}

#[derive(Default)]
pub(super) struct MenuLayouts {
    pub main: Option<MainLayout>,
    pub options: Option<OptionsLayout>,
    loaded_for: Option<(u8, bool)>,
    localization: Option<LocalizationTable>,
}

impl MenuLayouts {
    pub fn sync(
        &mut self,
        vfs: &VirtualFileSystem,
        viewport: UiViewport,
        single_player: bool,
        language: &str,
    ) -> bool {
        let mode = (viewport.screen_mode(), single_player);
        if self.loaded_for == Some(mode) {
            return false;
        }
        if self.loaded_for.is_none() {
            let mut table = LocalizationTable::new(language);
            let languages = ["eng", language];
            let count = if language.eq_ignore_ascii_case("eng") {
                1
            } else {
                2
            };
            for lang in &languages[..count] {
                for name in ["ui_st_mm.xml", "ui_st_options.xml"] {
                    if let Ok(doc) = load_xml_from_vfs(vfs, &format!("configs/text/{lang}/{name}"))
                    {
                        table.merge_doc(&doc);
                    }
                }
            }
            self.localization = Some(table);
        }
        let table = self.localization.as_ref();
        self.main = load_ui_xml_for_viewport_from_vfs(vfs, "ui_mm_main.xml", &viewport)
            .ok()
            .and_then(|doc| MainLayout::read(&doc, table, single_player));
        self.options = load_ui_xml_for_viewport_from_vfs(vfs, "ui_options.xml", &viewport)
            .ok()
            .and_then(|doc| OptionsLayout::read(&doc, table));
        self.loaded_for = Some(mode);
        true
    }
}

pub(super) fn font_section(alias: &str) -> Option<&'static str> {
    Some(match alias {
        "graffiti19" => "ui_font_graffiti19_russian",
        "graffiti22" => "ui_font_graffiti22_russian",
        "letterica16" => "ui_font_letterica16_russian",
        "letterica18" => "ui_font_letterica18_russian",
        "arial_14" => "ui_font_arial_14",
        "medium" => "hud_font_medium",
        _ => return None,
    })
}

fn localized(table: Option<&LocalizationTable>, id: Option<&str>, fallback: &str) -> String {
    id.zip(table)
        .map(|(id, table)| table.translate(id))
        .filter(|text| Some(*text) != id && !text.is_empty())
        .unwrap_or(fallback)
        .to_owned()
}

fn node_rect(node: &XmlNode, origin: [f32; 2]) -> Rect {
    [
        origin[0] + node.attr_f32("x", 0.0),
        origin[1] + node.attr_f32("y", 0.0),
        node.attr_f32("width", 0.0),
        node.attr_f32("height", 0.0),
    ]
}

fn valid_rect(rect: Rect) -> bool {
    rect.into_iter().all(f32::is_finite) && rect[2] > 0.0 && rect[3] > 0.0
}

pub(super) fn contains([x, y, w, h]: Rect, (mx, my): (f32, f32)) -> bool {
    (x..x + w).contains(&mx) && (y..y + h).contains(&my)
}

pub(super) fn draw_text_in(
    renderer: &mut Dx12Renderer,
    font: &UiFontResource,
    text: &str,
    [x, y, w, h]: Rect,
    align: FontAlignment,
    color: [f32; 4],
) {
    let size = font.atlas.line_height.min(h);
    let x = match align {
        FontAlignment::Left => x,
        FontAlignment::Center => x + w * 0.5,
        FontAlignment::Right => x + w,
    };
    font.draw_aligned(
        renderer,
        text,
        [x, y + (h - size) * 0.5],
        size,
        align,
        color,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn main_menu_uses_widget_insets_button_height_and_xml_order() {
        let xml = XmlDocument::parse(
            r#"<w>
                <shniaga_wnd x="300" y="380" height="200">
                    <buttons_region x="12" y="14" width="180" vert_interval="4" top_indent="2" left_ident="3" right_ident="5"/>
                    <shniaga x="7" y="-20">
                        <magnifire x="25" y="18" width="118" height="19"><y_offset>1</y_offset></magnifire>
                    </shniaga>
                </shniaga_wnd>
                <button h="20"/>
                <menu_main_single font="graffiti19" btn_height="99">
                    <btn name="btn_ret" caption="continue"/>
                    <btn name="btn_save" caption="save"/>
                    <btn name="btn_quit" caption="quit"/>
                    <btn name="btn_options" caption="options"/>
                </menu_main_single>
            </w>"#,
        ).unwrap();
        let layout = MainLayout::read(&xml, None, true).unwrap();
        let names = layout
            .buttons
            .iter()
            .map(|(_, label)| label.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["Return to game", "Exit to Windows", "Options"]);
        assert_eq!(layout.row(0), [317.0, 400.0, 172.0, 20.0]);
        assert_eq!(layout.row(2), [317.0, 448.0, 172.0, 20.0]);
        assert_eq!(layout.selector(2), [332.0, 449.0, 118.0, 19.0]);
        assert_eq!(layout.selector_start(), 378.0);

        let viewport = UiViewport::new(1366.0, 768.0);
        let [x, y, w, h] = viewport.virtual_to_screen_rect(layout.row(2));
        let hit = (x + w * 0.5, y + h * 0.5);
        assert!(contains(
            viewport.virtual_to_screen_rect(layout.row(2)),
            hit
        ));
        assert!(!contains(
            viewport.virtual_to_screen_rect(layout.row(1)),
            hit
        ));
    }
}
