# Single-player parity with Monolith

The C++ tree under [src](../src) provides the behavioral reference. Compare both engines with the same Anomaly data, save, weather time, camera, equipment, and display aspect. A loaded asset, a successful draw, and a matching frame are separate checkpoints.

## Sky and first-person HUD

| Reference path | Rust counterpart | Evidence and remaining differences |
| --- | --- | --- |
| [dxEnvironmentRender::RenderSky](../src/Layers/xrRender/dxEnvironmentRender.cpp) | [SkyRenderer](../crates/xray-render-dx12/src/sky.rs) and [render bridge](../crates/xray-engine/src/render_bridge.rs) | DX12 readback covers weather cubemap blending, yaw, and foreground occlusion. Rust uses the reference half-box positions and sampling coordinates. Compare sky and cloud composition, exposure, and sun flare in matched game frames. |
| [CCustomOutfit::ApplySkinModel](../src/xrGame/CustomOutfit.cpp) and [player_hud::load](../src/xrGame/player_hud.cpp) | [HUD hands selection](../crates/xray-engine/src/hud_weapon.rs) | Outfit-driven visuals, optional second visual, and opposite-arm visibility have regression coverage. An asset probe found both arms and their mapped texture for the Monolith prototype exosuit. Script model overrides and independent hand IK remain open. |
| [hud_item_measures::load, attachable_hud_item::update, and player_hud::calc_transform](../src/xrGame/player_hud.cpp) | [HUD item and attachment transforms](../crates/xray-engine/src/hud_weapon.rs) | The weapon HUD section supplies its model and hand/item offsets. The player HUD section supplies its attachment bone. The scoped UMP45 and prototype exosuit resolve the configured meshes, the `ump45_idle` pose, and all referenced diffuse textures from installed Anomaly data. |
| [player_hud::update and render_hud](../src/xrGame/player_hud.cpp) | [HUD draw parts](../crates/xray-engine/src/hud_weapon.rs) and [DX12 HUD pass](../crates/xray-engine/src/render_bridge.rs) | Rust samples frame zero of the configured idle clip. Full playback, aim rotation, reload and fire motion aliases, script layers, HUD lighting, and an in-game visual comparison remain open. |

## Reproducible checks

Run the normal engine and renderer tests in the Cargo workspace.

~~~powershell
cargo test --offline --locked --manifest-path crates/Cargo.toml --workspace
~~~

For a paired visual check, load a copy of the same save in both engines. Record the loaded actor coordinates and facing, saved weather cycle and time, the sky at the horizon and above it, and the HUD at idle and while aiming. Use the same aspect ratio. Keep the original save and its companion files available as the comparison fixture.
