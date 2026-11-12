//! X-Ray Monolith Standalone Engine Binary (`xray-engine`)
//!
//! Launches the pure-Rust S.T.A.L.K.E.R. engine runtime:
//! - Parses CLI flags (`-fsltx`, `-level`, `-save`, `-connect`, `--host`, `--headless`, etc.)
//! - Mounts `.db*` / `.xdb*` archives and loose `gamedata/` via [`VirtualFileSystem`]
//! - Streams world geometry, collision (`level.cform`), weather, details, and ALife `.scop` saves
//! - Executes the fixed-timestep physics + ALife + DX12 7-stage render graph loop

use std::env;
use std::fs;
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use xray_engine::{
    EngineConfig, EngineRuntime, LevelLoadConfig, NetworkSession, WindowConfig,
};
use xray_net::{ClientSession, HostSession};
use xray_vfs::VirtualFileSystem;

pub const DEFAULT_COOP_PORT: u16 = 5445;

/// Parsed command-line launch options for `xray-engine.exe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchArgs {
    pub fsltx: Option<PathBuf>,
    pub level: Option<String>,
    pub save: Option<String>,
    pub connect: Option<String>,
    pub host_port: Option<u16>,
    pub nickname: String,
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
    pub headless: bool,
    pub max_frames: Option<u64>,
    pub show_help: bool,
}

impl Default for LaunchArgs {
    fn default() -> Self {
        Self {
            fsltx: None,
            level: None,
            save: None,
            connect: None,
            host_port: None,
            nickname: "Stalker".to_string(),
            width: 1280,
            height: 720,
            vsync: true,
            headless: false,
            max_frames: None,
            show_help: false,
        }
    }
}

/// Resolved directories extracted from `fsgame.ltx` or engine working directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FsLayout {
    pub root_dir: PathBuf,
    pub game_data_dir: Option<PathBuf>,
    pub app_data_dir: Option<PathBuf>,
}

fn print_usage() {
    println!(
        "xray-engine (Pure-Rust X-Ray Monolith Standalone Runtime)\n\
         \n\
         USAGE:\n\
           xray-engine [OPTIONS]\n\
         \n\
         OPTIONS:\n\
           -fsltx <path>         Path to fsgame.ltx (default: ./fsgame.ltx if present)\n\
           -level <name>         Level name or vpath (e.g. l01_escape or levels/l01_escape)\n\
           -save <name>          Saved game name or .scop vpath (e.g. quicksave)\n\
           -connect <ip[:port]>  Connect to co-op host (default port {DEFAULT_COOP_PORT})\n\
           --host [port]         Listen as co-op host (default port {DEFAULT_COOP_PORT})\n\
           --nickname <name>     Player nickname for co-op sessions (default: Stalker)\n\
           --width <px>          Window width (default: 1280)\n\
           --height <px>         Window height (default: 720)\n\
           --no-vsync            Disable VSync swapchain present\n\
           --headless            Create hidden window without DX12 initialization\n\
           --max-frames <n>      Exit cleanly after rendering <n> frames\n\
           -h, --help            Print this help message"
    );
}

fn take_next_arg(args: &[String], idx: &mut usize, flag: &str) -> Result<String, String> {
    if *idx + 1 >= args.len() {
        return Err(format!("missing value for argument '{flag}'"));
    }
    *idx += 1;
    Ok(args[*idx].clone())
}

/// Parse X-Ray / Anomaly command-line flags (`-fsltx`, `-level`, `-save`, `-connect`, etc.).
pub fn parse_args<I, S>(iter: I) -> Result<LaunchArgs, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let raw: Vec<String> = iter.into_iter().map(Into::into).collect();
    let mut parsed = LaunchArgs::default();
    let mut i = 0;

    while i < raw.len() {
        let arg = raw[i].as_str();
        match arg {
            "-h" | "--help" | "/?" => {
                parsed.show_help = true;
            }
            "-fsltx" | "--fsltx" | "-ltx" => {
                let val = take_next_arg(&raw, &mut i, arg)?;
                parsed.fsltx = Some(PathBuf::from(val));
            }
            "-level" | "--level" => {
                let val = take_next_arg(&raw, &mut i, arg)?;
                parsed.level = Some(val);
            }
            "-save" | "--save" => {
                let val = take_next_arg(&raw, &mut i, arg)?;
                parsed.save = Some(val);
            }
            "-connect" | "--connect" => {
                let val = take_next_arg(&raw, &mut i, arg)?;
                parsed.connect = Some(val);
            }
            "--host" | "-host" => {
                if i + 1 < raw.len() && !raw[i + 1].starts_with('-') {
                    let port: u16 = raw[i + 1]
                        .parse()
                        .map_err(|_| format!("invalid host port '{}'", raw[i + 1]))?;
                    parsed.host_port = Some(port);
                    i += 1;
                } else {
                    parsed.host_port = Some(DEFAULT_COOP_PORT);
                }
            }
            "--nickname" | "-nick" => {
                parsed.nickname = take_next_arg(&raw, &mut i, arg)?;
            }
            "--width" => {
                let val = take_next_arg(&raw, &mut i, arg)?;
                parsed.width = val
                    .parse::<u32>()
                    .map_err(|_| format!("invalid width '{val}'"))?
                    .max(64);
            }
            "--height" => {
                let val = take_next_arg(&raw, &mut i, arg)?;
                parsed.height = val
                    .parse::<u32>()
                    .map_err(|_| format!("invalid height '{val}'"))?
                    .max(64);
            }
            "--no-vsync" | "-novsync" => {
                parsed.vsync = false;
            }
            "--headless" | "-headless" => {
                parsed.headless = true;
            }
            "--max-frames" => {
                let val = take_next_arg(&raw, &mut i, arg)?;
                parsed.max_frames = Some(
                    val.parse::<u64>()
                        .map_err(|_| format!("invalid frame count '{val}'"))?,
                );
            }
            unknown => {
                return Err(format!("unrecognized argument '{unknown}'"));
            }
        }
        i += 1;
    }

    if parsed.connect.is_some() && parsed.host_port.is_some() {
        return Err("cannot specify both -connect and --host".to_string());
    }

    Ok(parsed)
}

/// Resolve root and optional `$game_data$` / `$app_data_root$` paths from an `fsgame.ltx` file.
pub fn parse_fsgame_layout(fsltx_path: &Path) -> Result<FsLayout, Box<dyn std::error::Error>> {
    let abs_fsltx = if fsltx_path.is_absolute() {
        fsltx_path.to_path_buf()
    } else {
        env::current_dir()?.join(fsltx_path)
    };
    let default_root = abs_fsltx
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut layout = FsLayout {
        root_dir: default_root.clone(),
        game_data_dir: None,
        app_data_dir: None,
    };

    if !abs_fsltx.is_file() {
        return Ok(layout);
    }

    let text = fs::read_to_string(&abs_fsltx)?;
    for line in text.lines() {
        let no_comment = line.split(';').next().unwrap_or("").trim();
        if no_comment.is_empty() {
            continue;
        }
        let Some((lhs, rhs)) = no_comment.split_once('=') else {
            continue;
        };
        let key = lhs.trim().to_ascii_lowercase();
        let cols: Vec<&str> = rhs.split('|').map(str::trim).collect();
        // X-Ray fsgame.ltx format: `$key$ = recurse | notify | root_token | sub_path`
        let resolved_path = if cols.len() >= 3 {
            let root_tok = cols[2];
            let sub = cols.get(3).copied().unwrap_or("");
            let base = if root_tok.eq_ignore_ascii_case("$fs_root$") || root_tok.is_empty() {
                layout.root_dir.clone()
            } else {
                PathBuf::from(root_tok.replace(['/', '\\'], std::path::MAIN_SEPARATOR_STR))
            };
            if sub.is_empty() {
                base
            } else {
                base.join(sub.trim_end_matches(['/', '\\']))
            }
        } else {
            continue;
        };

        match key.as_str() {
            "$fs_root$" if resolved_path.is_dir() => {
                layout.root_dir = resolved_path;
            }
            "$game_data$" if resolved_path.is_dir() => {
                layout.game_data_dir = Some(resolved_path);
            }
            "$app_data_root$" if resolved_path.is_dir() => {
                layout.app_data_dir = Some(resolved_path);
            }
            _ => {}
        }
    }

    Ok(layout)
}

/// Mount `VirtualFileSystem` from `-fsltx` (or `./fsgame.ltx` / cwd fallback).
pub fn mount_vfs_from_args(args: &LaunchArgs) -> Result<VirtualFileSystem, Box<dyn std::error::Error>> {
    let fsltx_candidate = match &args.fsltx {
        Some(p) => p.clone(),
        None => PathBuf::from("fsgame.ltx"),
    };

    let layout = if args.fsltx.is_some() || fsltx_candidate.is_file() {
        parse_fsgame_layout(&fsltx_candidate)?
    } else {
        FsLayout {
            root_dir: env::current_dir()?,
            game_data_dir: None,
            app_data_dir: None,
        }
    };

    let mut vfs = VirtualFileSystem::mount_game_root(&layout.root_dir)?;
    if let Some(custom_gd) = &layout.game_data_dir
        && *custom_gd != layout.root_dir.join("gamedata")
    {
        vfs.mount_loose_dir(custom_gd);
    }
    if let Some(app_data) = &layout.app_data_dir
        && app_data.is_dir()
    {
        // Allows `savedgames/<name>.scop` stored under Anomaly's `appdata/` to resolve via VFS
        vfs.mount_loose_dir(app_data);
    }

    Ok(vfs)
}

/// Normalize `-level <name>` into a VFS directory path (e.g. `l01_escape` -> `levels\l01_escape`).
pub fn resolve_level_dir(level_arg: &str) -> String {
    let cleaned = level_arg.trim().replace('/', "\\");
    if cleaned.starts_with('$')
        || cleaned.to_ascii_lowercase().starts_with("levels\\")
        || cleaned.contains('\\')
    {
        cleaned
    } else {
        format!("levels\\{cleaned}")
    }
}

/// Candidate VFS paths to check for `-save <name>`.
pub fn resolve_save_vpath(vfs: &VirtualFileSystem, save_arg: &str) -> Option<String> {
    let cleaned = save_arg.trim().replace('/', "\\");
    let mut candidates = vec![cleaned.clone()];
    if !cleaned.to_ascii_lowercase().ends_with(".scop") {
        candidates.push(format!("{cleaned}.scop"));
    }
    if !cleaned.to_ascii_lowercase().contains("savedgames\\") {
        candidates.push(format!("savedgames\\{cleaned}"));
        if !cleaned.to_ascii_lowercase().ends_with(".scop") {
            candidates.push(format!("savedgames\\{cleaned}.scop"));
        }
    }

    candidates.into_iter().find(|c| vfs.exists(c))
}

/// Parse `-connect <ip[:port]>` into a [`SocketAddr`] (defaults to port `5445`).
pub fn parse_connect_addr(raw: &str) -> Result<SocketAddr, String> {
    let trimmed = raw.trim();
    if let Ok(addr) = trimmed.parse::<SocketAddr>() {
        return Ok(addr);
    }
    if let Ok(ip) = trimmed.parse::<IpAddr>() {
        return Ok(SocketAddr::new(ip, DEFAULT_COOP_PORT));
    }
    let with_port = if trimmed.contains(':') {
        trimmed.to_string()
    } else {
        format!("{trimmed}:{DEFAULT_COOP_PORT}")
    };
    with_port
        .to_socket_addrs()
        .map_err(|e| format!("failed to resolve connect address '{trimmed}': {e}"))?
        .next()
        .ok_or_else(|| format!("no socket address found for '{trimmed}'"))
}

fn run_engine(args: LaunchArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.show_help {
        print_usage();
        return Ok(());
    }

    let vfs = Arc::new(mount_vfs_from_args(&args)?);

    let config = EngineConfig {
        window: WindowConfig {
            title: "X-Ray Monolith Engine".to_string(),
            width: args.width,
            height: args.height,
            visible: !args.headless,
        },
        vsync: args.vsync,
        ..Default::default()
    };

    let mut runtime = EngineRuntime::new(config, !args.headless)?;
    runtime.attach_vfs(Arc::clone(&vfs));

    if let Some(connect_raw) = &args.connect {
        let addr = parse_connect_addr(connect_raw)?;
        let use_ipv6 = addr.is_ipv6();
        let client = ClientSession::connect(addr, 0x4D4F_4E4F, &args.nickname, use_ipv6)?;
        runtime.attach_network(NetworkSession::Client(client));
    } else if let Some(host_port) = args.host_port {
        let host = HostSession::new(host_port, false)?;
        runtime.attach_network(NetworkSession::Host(host));
    }

    if let Some(level_arg) = &args.level {
        let level_dir = resolve_level_dir(level_arg);
        let save_vpath = args
            .save
            .as_deref()
            .and_then(|s| resolve_save_vpath(&vfs, s));

        if args.save.is_some() && save_vpath.is_none() {
            return Err(format!(
                "requested savegame '{}' was not found in mounted VFS",
                args.save.as_deref().unwrap_or_default()
            )
            .into());
        }

        let level_cfg = LevelLoadConfig {
            level_dir,
            ..Default::default()
        };
        runtime.load_world_and_save_from_vfs(&vfs, &level_cfg, save_vpath.as_deref())?;
    }

    if let Some(max_frames) = args.max_frames {
        for _ in 0..max_frames {
            if !runtime.step_frame()? {
                break;
            }
        }
        if let Some(r) = &mut runtime.renderer {
            let _ = r.wait_for_gpu();
        }
        return Ok(());
    }

    runtime.run()
}

fn main() {
    let cli: Vec<String> = env::args().skip(1).collect();
    let args = match parse_args(cli) {
        Ok(a) => a,
        Err(err) => {
            eprintln!("xray-engine: {err}\n");
            print_usage();
            std::process::exit(2);
        }
    };

    if let Err(err) = run_engine(args) {
        eprintln!("xray-engine fatal error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_cli_arg_parsing_xray_flags() {
        let args = parse_args([
            "-fsltx",
            "D:\\Anomaly\\fsgame.ltx",
            "-level",
            "l01_escape",
            "-save",
            "quicksave",
            "-connect",
            "192.168.1.42:5445",
            "--no-vsync",
            "--width",
            "1920",
            "--height",
            "1080",
        ])
        .expect("parse valid flags");

        assert_eq!(args.fsltx, Some(PathBuf::from("D:\\Anomaly\\fsgame.ltx")));
        assert_eq!(args.level.as_deref(), Some("l01_escape"));
        assert_eq!(args.save.as_deref(), Some("quicksave"));
        assert_eq!(args.connect.as_deref(), Some("192.168.1.42:5445"));
        assert!(!args.vsync);
        assert_eq!(args.width, 1920);
        assert_eq!(args.height, 1080);

        assert_eq!(resolve_level_dir("l01_escape"), "levels\\l01_escape");
        assert_eq!(
            resolve_level_dir("$game_levels$/l05_bar"),
            "$game_levels$\\l05_bar"
        );

        assert_eq!(
            parse_connect_addr("127.0.0.1").unwrap(),
            "127.0.0.1:5445".parse().unwrap()
        );
        assert_eq!(
            parse_connect_addr("10.0.0.8:7777").unwrap(),
            "10.0.0.8:7777".parse().unwrap()
        );
    }

    #[test]
    fn test_parse_fsgame_ltx_mounts_custom_paths() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("xray_fsltx_test_{nonce}"));
        let custom_gd = root.join("custom_gd");
        let appdata = root.join("appdata");
        let saves = appdata.join("savedgames");
        fs::create_dir_all(&custom_gd).unwrap();
        fs::create_dir_all(&saves).unwrap();
        fs::write(saves.join("falcon_coop.scop"), b"SAVE").unwrap();

        let fsltx = root.join("fsgame.ltx");
        fs::write(
            &fsltx,
            "$app_data_root$ = false | true | $fs_root$ | appdata\\\n\
             $game_data$     = false | true | $fs_root$ | custom_gd\\\n",
        )
        .unwrap();

        let args = LaunchArgs {
            fsltx: Some(fsltx.clone()),
            ..Default::default()
        };
        let vfs = mount_vfs_from_args(&args).expect("mount vfs via fsgame.ltx");
        assert_eq!(
            resolve_save_vpath(&vfs, "falcon_coop").as_deref(),
            Some("savedgames\\falcon_coop.scop")
        );

        let _ = fs::remove_dir_all(&root);
    }
}
