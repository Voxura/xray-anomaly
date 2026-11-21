use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use windows::Win32::UI::Input::KeyboardAndMouse::{VK_DOWN, VK_RETURN, VK_UP};
use xray_render_dx12::Dx12Renderer;

use crate::input::{Action, EngineInput, MouseButton};
use crate::ui_inventory::UiFontResource;

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

    fn show(&mut self, page: Page) {
        self.page = page;
        self.focused = 0;
        self.save_failed = false;
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
        match (self.page, row) {
            (Page::Main, 0) | (Page::Extras, 5) | (Page::Quit | Page::Discard, 0) => self.back(),
            (Page::Main, 1) => {
                self.pending = self.current;
                self.show(Page::Extras);
                MenuOutcome::Stay
            }
            (Page::Main, 2) => {
                self.show(Page::Quit);
                MenuOutcome::Stay
            }
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

    pub fn handle_input(&mut self, input: &EngineInput, width: u32, height: u32) -> MenuOutcome {
        if input.is_action_just_pressed(Action::Menu) {
            return self.back();
        }
        let count = self.page.rows().len();
        if count == 0 {
            return MenuOutcome::Stay;
        }
        if input.is_key_just_pressed(usize::from(VK_UP.0)) {
            self.focused = (self.focused + count - 1) % count;
        }
        if input.is_key_just_pressed(usize::from(VK_DOWN.0))
            || input.is_key_just_pressed(EngineInput::VK_TAB)
        {
            self.focused = (self.focused + 1) % count;
        }
        if input.is_key_just_pressed(usize::from(VK_RETURN.0))
            || input.is_key_just_pressed(EngineInput::VK_SPACE)
        {
            return self.activate(self.focused);
        }
        if input.is_mouse_just_pressed(MouseButton::Left)
            && let Some(row) = MenuLayout::new(width, height).hit(input.mouse_position(), count)
        {
            self.focused = row;
            return self.activate(row);
        }
        MenuOutcome::Stay
    }

    pub fn render(
        &self,
        renderer: &mut Dx12Renderer,
        font: Option<&UiFontResource>,
        size: [u32; 2],
        mouse: (f32, f32),
        single_player: bool,
    ) {
        let layout = MenuLayout::new(size[0], size[1]);
        let s = layout.scale;
        let [x, y] = layout.origin;
        let rows = self.page.rows();
        let panel_height = 105.0 + rows.len() as f32 * 49.0;
        let label_color = [0.9, 0.91, 0.89, 1.0];
        let accent = [0.83, 0.67, 0.34, 1.0];
        let hover = layout.hit(mouse, rows.len());
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
            let active = hover == Some(row) || (hover.is_none() && self.focused == row);
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

struct MenuLayout {
    origin: [f32; 2],
    scale: f32,
}

impl MenuLayout {
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

    fn hit(&self, (mx, my): (f32, f32), count: usize) -> Option<usize> {
        (0..count).find(|row| {
            let [x, y, w, h] = self.row(*row);
            (x..x + w).contains(&mx) && (y..y + h).contains(&my)
        })
    }
}
