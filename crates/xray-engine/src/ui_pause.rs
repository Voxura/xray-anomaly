use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use windows::Win32::UI::Input::KeyboardAndMouse::{VK_DOWN, VK_RETURN, VK_UP};
use xray_formats::{fonts::FontAlignment, xml_ui::UiViewport};
use xray_render_dx12::Dx12Renderer;
use xray_vfs::VirtualFileSystem;

use crate::input::{Action, EngineInput, MouseButton};
use crate::ui_inventory::UiFontResource;

mod layout;
use layout::{MainAction, MenuLayouts, Rect, contains, draw_text_in};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Extras {
    pub backpack_animations: bool,
    pub item_picker: bool,
}

impl Extras {
    pub fn load(path: &Path, defaults: Self) -> io::Result<Self> {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(defaults),
            Err(error) => return Err(error),
        };
        let mut settings = defaults;
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let Ok(value) = value.trim().parse() else {
                continue;
            };
            match key.trim() {
                "backpack_animations" => settings.backpack_animations = value,
                "inventory_item_picker" => settings.item_picker = value,
                _ => {}
            }
        }
        Ok(settings)
    }

    fn save(self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        let text = format!(
            "[extras]\nbackpack_animations = {}\ninventory_item_picker = {}\n",
            self.backpack_animations, self.item_picker,
        );
        let result = fs::write(&temporary, text).and_then(|()| fs::rename(&temporary, path));
        if result.is_err() {
            let _ = fs::remove_file(temporary);
        }
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    Closed,
    Main,
    Extras,
    Discard,
    Quit,
}

impl Page {
    fn heading(self, single_player: bool) -> &'static str {
        match self {
            Self::Closed => "",
            Self::Main if single_player => "Game paused",
            Self::Main => "Game menu",
            Self::Extras => "Options / Extras",
            Self::Discard => "Discard changes?",
            Self::Quit => "Quit to Windows?",
        }
    }

    fn rows(self) -> &'static [&'static str] {
        match self {
            Self::Closed => &[],
            Self::Main => &["Return to game", "Options", "Quit to Windows"],
            Self::Extras => &[
                "Backpack animations",
                "Inventory item picker",
                "Accept",
                "Reset",
                "Default",
                "Cancel",
            ],
            Self::Discard => &["No", "Discard changes"],
            Self::Quit => &["No", "Quit to Windows"],
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum MenuOutcome {
    Stay,
    Resume,
    Quit,
    Apply(Extras),
}

pub(crate) struct PauseMenu {
    page: Page,
    focused: usize,
    defaults: Extras,
    current: Extras,
    pending: Extras,
    path: Option<PathBuf>,
    save_failed: bool,
    layouts: MenuLayouts,
    selector: Option<SelectorMotion>,
}

impl Default for PauseMenu {
    fn default() -> Self {
        let defaults = Extras {
            backpack_animations: true,
            item_picker: false,
        };
        Self {
            page: Page::Closed,
            focused: 0,
            defaults,
            current: defaults,
            pending: defaults,
            path: None,
            save_failed: false,
            layouts: MenuLayouts::default(),
            selector: None,
        }
    }
}

impl PauseMenu {
    pub fn configure(&mut self, defaults: Extras, path: Option<PathBuf>) -> Extras {
        self.defaults = defaults;
        self.current =
            path.as_deref()
                .map_or(defaults, |path| match Extras::load(path, defaults) {
                    Ok(extras) => extras,
                    Err(error) => {
                        eprintln!("[options] Could not read {}: {error}", path.display());
                        defaults
                    }
                });
        self.path = path;
        self.current
    }

    pub fn is_open(&self) -> bool {
        self.page != Page::Closed
    }

    pub fn open(&mut self) {
        self.show(Page::Main);
    }

    pub fn reset_layouts(&mut self) {
        self.layouts = MenuLayouts::default();
        self.selector = None;
    }

    pub fn sync_layouts(
        &mut self,
        vfs: Option<&VirtualFileSystem>,
        size: [u32; 2],
        single_player: bool,
        language: &str,
    ) {
        if let Some(vfs) = vfs
            && self
                .layouts
                .sync(vfs, viewport(size), single_player, language)
            && self.page == Page::Main
        {
            self.show(Page::Main);
        }
    }

    pub fn font_sections(&self) -> [&'static str; 3] {
        [
            self.layouts
                .main
                .as_ref()
                .map_or("ui_font_graffiti19_russian", |main| main.font_section),
            self.layouts
                .options
                .as_ref()
                .map_or("ui_font_letterica16_russian", |options| options.label_font),
            self.layouts
                .options
                .as_ref()
                .map_or("ui_font_letterica18_russian", |options| options.button_font),
        ]
    }

    fn show(&mut self, page: Page) {
        self.page = page;
        self.focused = 0;
        self.save_failed = false;
        self.selector = if page == Page::Main {
            self.layouts.main.as_ref().map(|layout| {
                let target = layout.selector(0)[1];
                let from = layout.selector_start();
                SelectorMotion::new(from, target, layout.selector_duration(from, target))
            })
        } else {
            None
        };
    }

    fn back(&mut self) -> MenuOutcome {
        match self.page {
            Page::Main => {
                self.show(Page::Closed);
                MenuOutcome::Resume
            }
            Page::Extras if self.pending != self.current => {
                self.show(Page::Discard);
                MenuOutcome::Stay
            }
            Page::Extras | Page::Quit => {
                self.show(Page::Main);
                MenuOutcome::Stay
            }
            Page::Discard => {
                self.show(Page::Extras);
                MenuOutcome::Stay
            }
            Page::Closed => MenuOutcome::Stay,
        }
    }

    fn activate(&mut self, row: usize) -> MenuOutcome {
        if self.page == Page::Main {
            let action = self.layouts.main.as_ref().map_or(
                [MainAction::Resume, MainAction::Options, MainAction::Quit][row],
                |layout| layout.buttons[row].0,
            );
            return match action {
                MainAction::Resume => self.back(),
                MainAction::Options => {
                    self.pending = self.current;
                    self.show(Page::Extras);
                    MenuOutcome::Stay
                }
                MainAction::Quit => {
                    self.show(Page::Quit);
                    MenuOutcome::Stay
                }
            };
        }
        match (self.page, row) {
            (Page::Extras, 5) | (Page::Quit | Page::Discard, 0) => self.back(),
            (Page::Extras, 0) => {
                self.pending.backpack_animations = !self.pending.backpack_animations;
                MenuOutcome::Stay
            }
            (Page::Extras, 1) => {
                self.pending.item_picker = !self.pending.item_picker;
                MenuOutcome::Stay
            }
            (Page::Extras, 2) => {
                if self.pending == self.current {
                    self.show(Page::Main);
                    return MenuOutcome::Stay;
                }
                if let Some(path) = &self.path
                    && let Err(error) = self.pending.save(path)
                {
                    eprintln!("[options] Could not save {}: {error}", path.display());
                    self.save_failed = true;
                    return MenuOutcome::Stay;
                }
                self.current = self.pending;
                self.show(Page::Main);
                MenuOutcome::Apply(self.current)
            }
            (Page::Extras, 3) => {
                self.pending = self.current;
                self.save_failed = false;
                MenuOutcome::Stay
            }
            (Page::Extras, 4) => {
                self.pending = self.defaults;
                MenuOutcome::Stay
            }
            (Page::Discard, 1) => {
                self.pending = self.current;
                self.show(Page::Main);
                MenuOutcome::Stay
            }
            (Page::Quit, 1) => MenuOutcome::Quit,
            _ => MenuOutcome::Stay,
        }
    }

    fn button_rect(&self, row: usize, size: [u32; 2]) -> Rect {
        let rect = match self.page {
            Page::Main => self.layouts.main.as_ref().map(|layout| layout.row(row)),
            Page::Extras => self.layouts.options.as_ref().map(|layout| layout.rows[row]),
            _ => None,
        };
        rect.map_or_else(
            || FallbackLayout::new(size[0], size[1]).row(row),
            |rect| viewport(size).virtual_to_screen_rect(rect),
        )
    }

    fn focus(&mut self, row: usize) {
        if row != self.focused
            && self.page == Page::Main
            && let Some(layout) = &self.layouts.main
        {
            let from = self
                .selector
                .as_ref()
                .map_or(layout.selector(self.focused)[1], SelectorMotion::y);
            let to = layout.selector(row)[1];
            self.selector = Some(SelectorMotion::new(
                from,
                to,
                layout.selector_duration(from, to),
            ));
        }
        self.focused = row;
    }

    pub fn handle_input(&mut self, input: &EngineInput, width: u32, height: u32) -> MenuOutcome {
        if input.is_action_just_pressed(Action::Menu) {
            return self.back();
        }
        let count = self.page.rows().len();
        if count == 0 {
            return MenuOutcome::Stay;
        }
        if let Some(row) = (0..count).find(|row| {
            contains(
                self.button_rect(*row, [width, height]),
                input.mouse_position(),
            )
        }) {
            if input.is_mouse_just_pressed(MouseButton::Left) {
                self.focus(row);
                return self.activate(row);
            }
            if input.mouse_delta() != (0.0, 0.0) {
                self.focus(row);
            }
        }
        if input.is_key_just_pressed(usize::from(VK_UP.0)) {
            self.focus(self.focused.saturating_sub(1));
        }
        if input.is_key_just_pressed(usize::from(VK_DOWN.0)) {
            self.focus((self.focused + 1).min(count - 1));
        }
        if input.is_key_just_pressed(EngineInput::VK_TAB) {
            self.focus((self.focused + 1) % count);
        }
        if input.is_key_just_pressed(usize::from(VK_RETURN.0))
            || input.is_key_just_pressed(EngineInput::VK_SPACE)
        {
            return self.activate(self.focused);
        }
        MenuOutcome::Stay
    }

    pub fn render(
        &self,
        renderer: &mut Dx12Renderer,
        fonts: [Option<&UiFontResource>; 3],
        size: [u32; 2],
        single_player: bool,
    ) {
        let vp = viewport(size);
        if self.page == Page::Main
            && let Some(layout) = &self.layouts.main
        {
            self.render_main(renderer, fonts[0], vp, layout, single_player);
        } else if self.page == Page::Extras
            && let Some(layout) = &self.layouts.options
        {
            self.render_options(renderer, fonts[1], fonts[2], vp, layout);
        } else {
            self.render_fallback(renderer, fonts[1], size, single_player);
        }
    }

    fn render_main(
        &self,
        renderer: &mut Dx12Renderer,
        font: Option<&UiFontResource>,
        vp: UiViewport,
        layout: &layout::MainLayout,
        single_player: bool,
    ) {
        let [x, y, w, h] = layout.row(0);
        let heading = [x, y - h * 1.5, w, h];
        let mut selector = layout.selector(self.focused);
        if let Some(motion) = &self.selector {
            selector[1] = motion.y();
        }
        let [sx, sy, sw, sh] = vp.virtual_to_screen_rect(selector);
        renderer.ui_fill_rect([sx, sy, sw, sh], [0.17, 0.20, 0.20, 1.0]);
        renderer.ui_fill_rect([sx, sy, 3.0, sh], [0.79, 0.62, 0.31, 1.0]);
        if let Some(font) = font {
            draw_text_in(
                renderer,
                font,
                self.page.heading(single_player),
                vp.virtual_to_screen_rect(heading),
                FontAlignment::Center,
                [0.69, 0.73, 0.70, 1.0],
            );
            for (row, (_, label)) in layout.buttons.iter().enumerate() {
                draw_text_in(
                    renderer,
                    font,
                    label,
                    vp.virtual_to_screen_rect(layout.row(row)),
                    FontAlignment::Center,
                    layout.text_color,
                );
            }
        }
    }

    fn render_options(
        &self,
        renderer: &mut Dx12Renderer,
        font: Option<&UiFontResource>,
        button_font: Option<&UiFontResource>,
        vp: UiViewport,
        layout: &layout::OptionsLayout,
    ) {
        let screen = |rect| vp.virtual_to_screen_rect(rect);
        let panel = [0.09, 0.115, 0.12, 1.0];
        let field = [0.12, 0.15, 0.15, 1.0];
        let focus = [0.23, 0.26, 0.23, 1.0];
        let accent = [0.83, 0.67, 0.34, 1.0];
        let text = [0.90, 0.91, 0.89, 1.0];
        renderer.ui_fill_rect(screen(layout.panel), panel);
        renderer.ui_fill_rect(screen(layout.area), [0.065, 0.078, 0.08, 1.0]);
        renderer.ui_fill_rect(screen(layout.category), focus);
        if let Some(font) = font {
            draw_text_in(
                renderer,
                font,
                "Options",
                screen(layout.heading),
                FontAlignment::Left,
                text,
            );
            draw_text_in(
                renderer,
                font,
                "Extras",
                screen(layout.category),
                FontAlignment::Center,
                accent,
            );
        }
        for (row, mut rect) in layout.rows.into_iter().enumerate() {
            renderer.ui_fill_rect(
                screen(rect),
                if row == self.focused { focus } else { field },
            );
            if row < 2 {
                let enabled = [self.pending.backpack_animations, self.pending.item_picker][row];
                let label = self.page.rows()[row];
                let [x, y, w, h] = screen(layout.status[row]);
                let side = w.min(h) * 0.56;
                let color = if enabled {
                    accent
                } else {
                    [0.32, 0.36, 0.35, 1.0]
                };
                let [cx, cy] = [x + (w - side) * 0.5, y + (h - side) * 0.5];
                renderer.ui_fill_rect([cx, cy, side, side], color);
                if !enabled {
                    renderer.ui_fill_rect(
                        [
                            cx + 2.0,
                            cy + 2.0,
                            (side - 4.0).max(0.0),
                            (side - 4.0).max(0.0),
                        ],
                        panel,
                    );
                }
                if let Some(font) = font {
                    rect[0] += layout.label_x;
                    draw_text_in(
                        renderer,
                        font,
                        label,
                        screen(rect),
                        FontAlignment::Left,
                        text,
                    );
                }
            } else if let Some(font) = button_font {
                draw_text_in(
                    renderer,
                    font,
                    &layout.labels[row - 2],
                    screen(rect),
                    FontAlignment::Center,
                    text,
                );
            }
        }
        if self.save_failed
            && let Some(font) = font
        {
            draw_text_in(
                renderer,
                font,
                "Could not save options",
                screen(layout.notification),
                FontAlignment::Left,
                accent,
            );
        }
    }

    fn render_fallback(
        &self,
        renderer: &mut Dx12Renderer,
        font: Option<&UiFontResource>,
        size: [u32; 2],
        single_player: bool,
    ) {
        let layout = FallbackLayout::new(size[0], size[1]);
        let s = layout.scale;
        let [x, y] = layout.origin;
        let rows = self.page.rows();
        let panel_height = 105.0 + rows.len() as f32 * 49.0;
        let label_color = [0.9, 0.91, 0.89, 1.0];
        let accent = [0.83, 0.67, 0.34, 1.0];
        renderer.ui_fill_rect(
            [x, y, 420.0 * s, panel_height * s],
            [0.09, 0.115, 0.12, 1.0],
        );
        renderer.ui_fill_rect([x, y, 4.0 * s, panel_height * s], accent);
        if let Some(font) = font {
            font.draw_text(
                renderer,
                self.page.heading(single_player),
                [x + 24.0 * s, y + 22.0 * s],
                29.0 * s,
                label_color,
            );
        }
        for (row, label) in rows.iter().enumerate() {
            let [left, top, w, h] = layout.row(row);
            let active = self.focused == row;
            let fill = if active {
                [0.23, 0.26, 0.23, 1.0]
            } else {
                [0.13, 0.16, 0.16, 1.0]
            };
            renderer.ui_fill_rect([left, top, w, h], fill);
            if let Some(font) = font {
                font.draw_text(
                    renderer,
                    label,
                    [left + 12.0 * s, top + 9.0 * s],
                    18.0 * s,
                    label_color,
                );
                if self.page == Page::Extras && row < 2 {
                    let enabled = if row == 0 {
                        self.pending.backpack_animations
                    } else {
                        self.pending.item_picker
                    };
                    let status = if enabled { "On" } else { "Off" };
                    font.draw_text(
                        renderer,
                        status,
                        [left + w - 54.0 * s, top + 9.0 * s],
                        18.0 * s,
                        accent,
                    );
                }
            }
        }
        if self.save_failed
            && let Some(font) = font
        {
            font.draw_text(
                renderer,
                "Could not save options",
                [x + 24.0 * s, y + (panel_height - 29.0) * s],
                18.0 * s,
                accent,
            );
        }
    }
}

fn viewport([width, height]: [u32; 2]) -> UiViewport {
    UiViewport::new(width as f32, height as f32)
}

struct SelectorMotion {
    start: Instant,
    from: f32,
    to: f32,
    duration: f32,
}

impl SelectorMotion {
    fn new(from: f32, to: f32, duration: f32) -> Self {
        Self {
            start: Instant::now(),
            from,
            to,
            duration,
        }
    }

    fn y(&self) -> f32 {
        // The menu selector runs on real time while the simulation is frozen.
        let t = (self.start.elapsed().as_secs_f32() / self.duration).min(1.0);
        let eased = (1.0 + 10.0 * t).ln() / 11.0_f32.ln();
        self.from + (self.to - self.from) * eased
    }
}

struct FallbackLayout {
    origin: [f32; 2],
    scale: f32,
}

impl FallbackLayout {
    fn new(width: u32, height: u32) -> Self {
        let scale = (height as f32 / 720.0).min(width as f32 / 560.0);
        Self {
            origin: [width as f32 * 0.5 - 210.0 * scale, height as f32 * 0.18],
            scale,
        }
    }

    fn row(&self, row: usize) -> [f32; 4] {
        let [x, y] = self.origin;
        let s = self.scale;
        [
            x + 24.0 * s,
            y + (72.0 + row as f32 * 49.0) * s,
            372.0 * s,
            42.0 * s,
        ]
    }
}
