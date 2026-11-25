# Play the browser build locally

Build and watch the game in a headed Chrome on your own machine. Game files stay outside the repo.

## Prerequisites
- Windows, Chrome (or Edge/Chromium), Python 3, Rust (cargo), CMake, Ninja.
- Emscripten (emsdk) **7.0.1**, on PATH (`emcc`, `emcmake`).
- Your own game executable and package directory (never committed).
- RAM: plan for ~20-30 GB free.

## Quick start
```powershell
.\tools\play_local.ps1 -Exe <path-to-game.exe> -GameDir <package-dir> -Port 8141 -BuildDir work/browser-boot -InputScript calibration-skip
```
This calls `tools/run_browser_boot.py --preset long --headed`: it rebuilds stale stages (recomp lift, PIC runtime lib,
guest Wasm, shader pack, staging), starts the local COOP/COEP server and launches Chrome with its own profile.
Useful switches: `-NoBuild` (fail on stale), `-TrustBuild` (reuse artifacts), `-DryRun` (print plan),
`-Chrome <path>`, `-DurationS`. Input presets: start-screen, menu-flow, calibration-skip, intro-watch,
legal-notices, chrome-menu-flow. Defaults come from `PLAYGTA6_EXE`, `GTA6_GAME_DIR`, `CHROME_PATH`.

Stop: close the Chrome window or press Ctrl+C in the script's terminal (the server ends with it).

## Build recipe (issue #687, manual)
`build-program --memory-model window-runs --indirect-dispatch shard --shard-size 2048 --shard-instructions 65536`;
runtime lib via `emcmake cmake -S runtime -G Ninja -DCMAKE_BUILD_TYPE=Release` with
`-sMEMORY64=1 -pthread -fwasm-exceptions -fPIC`; `tools/build_guest_wasm.py --split-modules 2 --split-strategy asymmetric
--wasm-exceptions --proxy-to-pthread --link-guest-boot --browser-boot-exports --pthread-pool-size 8`;
shader pack via `playgta6-shaders pack-rdna2` + `tools/stage_reflection_sidecar.py`; then `tools/stage_browser_boot.py`.
`run_browser_boot.py` does all of this for you.

## Boot options (bisecting a browser regression)
`boot-options.json` (next to the staged boot) accepts these switches. All are optional booleans and default to true
unless noted; they mirror the `playgta6_boot` harness flags, so a regression can be bisected without a rebuild.

| Option | Default | Effect when false (harness flag) |
|---|---|---|
| `present_pacing` | true | presents do not wait for the 60 Hz vblank (`--no-present-pacing`) |
| `verify_window_marks` | false | true: readers also compare unmarked window slots |
| `fence_publish_lag` | true | fences complete at once instead of at the next present (`--no-fence-lag`) |
| `defer_gpu_objects` | true | GPU objects are written at creation, not on first use (`--eager-gpu-objects`) |
| `xmem_shared_pages` | true | XMem pages mapped at several addresses get per-address copies (`--no-xmem-shared-pages`) |
| `present_frame_waits` | true | `WaitFrameEventX(PRESENT, token)` takes the pacer tick instead of waiting for the token's present |
| `audio_endpoint` | true | the MMDevice enumerator is not registered (no default render endpoint) |

## Expected timings
First frame about 30 s after load; the intro follows later. The first build takes a long time (Wasm link); later runs reuse the cache.

## Keyboard mapping
| Key | Pad |
|---|---|
| Enter / Space | A |
| Backspace | B |
| Arrow keys | D-pad |
| W A S D | Left stick |
| R | X |
| F | Y |
| Q / E | LB / RB |
| Tab | View |
| Esc | Menu |
