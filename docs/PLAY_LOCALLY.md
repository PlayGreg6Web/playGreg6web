# Play the browser build locally

Build and watch the game in a headed Chrome on your own machine. Game files stay outside the repo.

## Prerequisites
- Windows, Chrome (or Edge/Chromium), Python 3, Rust (cargo), CMake, Ninja.
- Emscripten (emsdk) **7.0.1**, on PATH (`emcc`, `emcmake`), or `EMSDK` set, or installed anywhere under `work/**/emsdk` (e.g. `work/emsdk`, `work/real/o36/emsdk`). The build tooling (`tools/emsdk_locator.py`) picks the install whose `upstream/emscripten/emscripten-version.txt` is 7.0.1, else the newest, and prints which one it uses. If none is found `play_local.ps1` prints a one-line fix (clone emsdk into `work/emsdk`, `emsdk install 7.0.1`, `emsdk activate 7.0.1`).
- Your own game executable and package directory (never committed).
- RAM: plan for ~20-30 GB free.

## Quick start
```powershell
.\tools\play_local.ps1 -Exe <path-to-game.exe> -GameDir <package-dir> -Port 8141 -BuildDir work/browser-boot
```
This calls `tools/run_browser_boot.py --preset long --headed`: it rebuilds stale stages (recomp lift, PIC runtime lib,
guest Wasm, shader pack, staging), starts the local COOP/COEP server and launches Chrome with its own profile.
Useful switches: `-NoBuild` (fail on stale), `-TrustBuild` (reuse artifacts), `-DryRun` (print plan),
`-Chrome <path>`, `-DurationS`. Input presets: start-screen, menu-flow, calibration-skip, intro-watch,
legal-notices, chrome-menu-flow, chrome-menu-flow-2, chrome-menu-flow-4, chrome-menu-flow-5, chrome-full-boot,
menu-to-ingame, chrome-to-ingame (every preset `tools/run_browser_boot.py` accepts; `chrome-to-ingame` goes through
the LEGAL NOTICES and ROCKSTAR GAMES LEGAL TERMS screens automatically, then into the Main Menu / in-game). Defaults come from the env vars `PLAYGTA6_EXE`, `GTA6_GAME_DIR`, `CHROME_PATH` and `PLAYGTA6_SHADER_PACK`.
`-InputScript` defaults to `calibration-skip` (`-InputScript none` turns it off). `-ShaderPack` defaults to
`$env:PLAYGTA6_SHADER_PACK`, else `work/splash.wgsl.pack`; if none exists the script prints the build command and stops.
Each run gets a unique name (`play-local-YYYYMMDD-HHMMSS`, override with `-Name`), so reruns never collide.

## Troubleshooting: stuck on loading artwork
- Without a shader pack nothing renders: check that the page's debug panel shows **Shader Packs** = 1 and that
  **Pack Hits** climbs. If not, build one (`cargo run --release -p playgta6-shaders -- pack-rdna2 <input-dir> --out
  work/splash.wgsl.pack`) and pass `-ShaderPack` or set `PLAYGTA6_SHADER_PACK`.
- The title waits on its first (calibration) screen and the keyboard may not get past it. Keep the default
  `-InputScript calibration-skip` (do not pass `-InputScript none`).

A shader pack is required to see anything: without a pack the panel shows "Shader Packs: 0 packs" and every draw is dropped.
Build one with `playgta6-shaders pack-rdna2 --max-wgsl-bytes 262144` plus `tools/stage_reflection_sidecar.py` (see the build recipe below); the `.reflect.json` sidecar is staged next to the pack.
"Manifest/Bootset: Missing" in the readiness panel is fine for local play and can be ignored.

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
| `fence_publish_lag` | false | true: fences complete at the next present instead of at once (`--fence-lag`) |
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

### Native build: `--host-input`
`playgta6_boot --host-input` feeds the host's first connected XInput pad and the keyboard into GameInput. The harness has
no window of its own, so the flag is opt-in and keys are read globally. It uses the table above plus I J K L (right
stick), Shift / Ctrl (L3 / R3) and Z / C (LT / RT). Raw values pass through; the title's adapter applies the deadzones;
the pad wins per axis when it is deflected more. `--input-script` takes precedence when both are given.

## Native play: one command

```powershell
.	ools\play_native.ps1            # PLAYGTA6_EXE / GTA6_GAME_DIR as for play_local.ps1
.	ools\play_native.ps1 -FreshSetup  # new empty save dir: the setup screens appear again
```

Builds what is stale (recompiler, lift, harness with the Dawn executor on, all from the content-keyed caches below),
then runs `playgta6_boot --window --audio-out --host-input --shader-pack <pack>`: a window with the live frames, WASAPI
audio, keyboard and pad.

- **Shader pack.** `-ShaderPack` wins, then `$env:PLAYGTA6_NATIVE_SHADER_PACK`, then the newest `*.wgsl.pack` found under
  `work/` (and under `$env:PLAYGTA6_SHADER_PACK_DIR` if set). The chosen file is printed (`[play] shader pack: ...`).
  With none, the run stops before building (nothing would be drawn).
- **Saves.** Persistent by default in `%LOCALAPPDATA%\playgta6\saves` (`$env:PLAYGTA6_SAVE_DIR` overrides, `-SaveDir`
  wins). `-FreshSetup` uses a new empty `%LOCALAPPDATA%\playgta6resh-<timestamp>` so the setup screens appear; the
  persistent saves are left alone.
- **Window title** shows `fps | frame N | draws/frame | ok`. **F11** toggles borderless fullscreen. Esc is not
  handled by the window: it reaches the title as Menu (see the keyboard table).
- **Guest fault.** The window stays open on the last frame with the title `GUEST FAULT ...`; the console prints the
  `tools/fault_report.py` output right away. Close the window to exit (the report is then finalised as usual).

Other options: `-InputScript none|start-screen|menu-flow|menu-flow-2|menu-flow-3|menu-flow-4|menu-to-ingame`
(scripted setup input instead of waiting for you; default `none`), `-MaxMs` (default 2 h; closes the window
when reached), `-NoBuild`, `-TrustBuild`, `-DryRun`, `-BuildDir`. The report lands in `<build-dir>/runs/<name>/`
(`boot-report.json`: presents, `audio_out` counters). Stop with Ctrl+C or by closing the window.
Same thing from Python: `python tools/run_native_boot.py --play [--fresh-setup] [--save-dir DIR] ...`
(or `--native-executor` for just the build).

## Native boot: fast, safe repeat builds

`python tools/run_native_boot.py --build-only ...` runs only the (timed) build stages
(`recomp`, `program`, `harness`) and exits before the run; each stage prints `done in N s`.
Repeat builds avoid the three expensive stages by content-keyed caches that sit beside the
object cache (`<object cache>-lift`, `<object cache>-recomp`):

| Stage | Skipped when | Key |
|---|---|---|
| `recomp` (cargo release build, cold in a fresh worktree) | a binary for the same recompiler sources is cached; or the stamp `target/release/playgta6-recomp.key` matches | sha256 of every workspace crate's `src/`, manifests, `build.rs`, `Cargo.lock`, plus `rustc -vV` |
| `program` (discovery + lift) | the lift cache has the entry, or `program/.lift-key` matches (#1475) | recompiler sources, exe, offered DLLs, shim headers, lift options |
| `harness` (~389 shards) | the object cache has every shard object | per shard: IR hash, target triple, CPU, `-O` level, PIC mode, LLVM tool version, retarget code fingerprint |

Overrides: `PLAYGTA6_RECOMP_CACHE=<dir>|off`, `PLAYGTA6_LIFT_CACHE=<dir>|off`, `--cache-dir`.
Caches hold only hashes, binaries and generated IR, never game bytes in git.

**Build-dir lock.** A run holds `<build-dir>/.build.lock` (heartbeat-kept, taken over after
120 s without heartbeat when its process died) for the whole build and run. A second run on
the same build dir fails immediately with `build directory ... is in use by another run` and
exit code 2; give each parallel run its own `--build-dir` (the default `work/native-boot` is
per checkout, so separate worktrees never collide). Never kill cargo/rustc/python to free a
lock; wait or pick another build dir.
