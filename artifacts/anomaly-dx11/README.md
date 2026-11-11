# Phase 4 engine delivery

Local delivery of `AnomalyDX11.exe`, built from source commit `55c7faec9fa7f276f698cdeaec703a17d792603a`.

The authorized DX11 x64 v145 build completed successfully: 0 errors, 5 warnings,
461.2 seconds. Missing incremental link data caused full link-time generation of
308,960 functions. The previous 19:41:42 executable failed the hook-string check.

Verification: the delivered executable contains all four Phase 4 ABI symbol
strings and the test-mesh success message. It is an AMD64 PE whose raw section
ranges fit inside the file. See `manifest.json` for SHA-256 and the build receipt,
and `build.log` for the complete build output with trailing whitespace removed.

This verifies build completion and inclusion of hook strings. Runtime startup,
DLL compatibility, and hardware rendering have not been tested. This package
contains the engine executable only; use an existing compatible game installation
and provide the matching `xray_render_dx12.dll` for `-use_rust_dx12` / `-r5`.

Warnings include duplicate LuaJIT object input, two missing return paths, a JPEG
buffer overrun diagnostic, and an uninitialized variable in dx10HW.cpp. These
warnings were retained in the log and were not fixed as part of this delivery.
