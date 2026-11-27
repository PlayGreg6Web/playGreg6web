# Play locally

## Quick start: play natively (Windows)

**One-time setup.** Set three user environment variables (System Properties > Environment Variables, or `setx`), then
open a **new** PowerShell window so they are picked up:

| Variable | Value |
|---|---|
| `PLAYGTA6_EXE` | full path to your game `.exe` |
| `GTA6_GAME_DIR` | your game package directory |
| `PLAYGTA6_NATIVE_SHADER_PACK` | full path to the standard `*.wgsl.pack` shader pack |

**Play.** From the repo folder, in PowerShell:

```powershell
.\tools\play_native.ps1 -Update
```

`-Update` first fast-forwards the checkout to the latest `main` (it never discards local changes), then builds what is
stale and opens the game window. Add `-FreshSetup` to see the setup screens again (uses an empty save dir).
`.\tools\play_native.ps1 -Help` prints a short summary of the switches.

If it stops or crashes, see [WHEN_IT_CRASHES.md](WHEN_IT_CRASHES.md) for where the report is and what to send.

**Play without compiling (prebuilt).** `tools\update_prebuilt.ps1` builds once into a folder (default
`%LOCALAPPDATA%\playgta6\prebuilt`, or `-Dir` / `PLAYGTA6_PREBUILT_DIR`) into a temp dir and swaps it in only when the
build succeeded, so your playable copy is never broken; it records the source commit in `prebuilt.json`. Before the swap (native) it runs a ~75 s headless smoke boot from the new copy (`--live-executor`, menu-flow, no window) and swaps only if it reached presents > 0 and ended without a fault or on a known hard stop (`tools/prebuilt_smoke.py`); otherwise the old copy stays, the new build is left as `<dir>.new` and the stop line and run dir are printed (exit 3). `-SkipSmoke` opts out; `-DryRun` prints the plan. The browser build has no headless smoke boot (its build-only check is the gate). Then
`.\tools\play_native.ps1 -Prebuilt` starts it with `-TrustBuild` semantics (never builds) and prints the build's commit and
how many commits it is behind `origin/main`. If commits since then touched `runtime/`, `recomp/`, `shaders/` or the boot harness, it also prints one "Prebuilt is N runtime commits behind" line plus the 3 newest subjects (`tools/prebuilt_stale.py`; informational only, never blocks or rebuilds). To use another folder: `$env:PLAYGTA6_PREBUILT_DIR = "D:\pg6-play"`.

**Controls** (full table and `keymap.ini` details further down):

| Key | Action |
|---|---|
| Enter / Space | A |
| Backspace | B |
| Arrow keys | D-pad |
| W A S D | Left stick |
| I J K L | Right stick |
| R / F | X / Y |
| Q / E | LB / RB |
| Z / C | LT / RT |
| Tab / Esc | View / Menu |
| F11 | Toggle borderless fullscreen |
| F12 | Save a screenshot (PNG) to `%LOCALAPPDATA%\playgta6\screenshots` |
| F8 | Release / re-capture the mouse-look cursor |
| F3 | Stats overlay on / off (fps, frame-time graph, present mode, draws, GPU ms, audio underruns, fault text); `--overlay` starts with it on |
| Ctrl+F7 / F8 / F9 / F10 | `--show-target` only: next mip / next layer / previous target / next target |

These window hotkeys live in one table (`runtime/os/window_hotkeys.h`) and are rebindable in `keymap.ini` `[hotkeys]`
(see "Rebinding keys"); `play_native.ps1 -Help` lists them. Esc stays the pad Menu button.

Rebind anything in `%LOCALAPPDATA%\playgta6\keymap.ini` (written on first run; see "Rebinding keys" below), or point at
another file with `-Keymap <file>`.

**Mouse look** (off by default): set `look = on` under `[mouse]` in `keymap.ini`. The window then captures the cursor
(Raw Input, hidden and clipped) and the mouse drives the right stick. **F8** (`release_key`) releases or re-captures it
by hand; a click in the window re-captures. Focus loss releases it automatically. Details in "Rebinding keys" below.
A connected XInput pad also gets the game's rumble.

**Overlay:** F3 draws a small box top-left (fps and frame time, a graph of the last 120 frames, present mode, draws per frame, GPU ms when the adapter has timestamps, live-audio underruns). When the guest stops on a fault the box also shows the reason, so it is readable in fullscreen where the title bar is not. It is drawn after the game's frame is copied for `--live-digest` and screenshots, so those never contain it.

**Present interval (60 vs 30 fps in menus):** the title presents with sync interval 2 in menus (30 Hz on the console) and sets
SetFrameIntervalX(33333 us, period 3). By default (`--present-interval min1`) the runtime only enforces a 1-vblank (16.7 ms) minimum
gap between presents in title-driven mode, so menus run at about 60 fps, faster than the console. `-ConsoleTiming` (`--present-interval honor`)
waits interval x vblank between presents, so those menus run at 30 fps like the console. Gameplay at interval 1 is the same in both.
The active mode shows in the window title (`interval honor|min1`), the overlay (`HONOR` after the present mode) and the boot report
(`present_pacing.present_interval`).

**Screenshots:** F12 saves the presented frame as a PNG; `--screenshot-at` does it at a chosen time in scripted runs.

**Saves tools:** `python tools/saves.py list|export|import` lists native saves or moves them as a zip (format in
[SAVE_ZIP_FORMAT.md](SAVE_ZIP_FORMAT.md)); it reads `%LOCALAPPDATA%\playgta6\saves` by default (older builds used `...\playgta6web\saves`;
`saves.py migrate --from <old> --to <new>` copies them over). The runtime itself falls back to the older folder for a run when the new one has no saves for the title yet (it prints the `migrate` command; nothing is moved; `--save-dir` always wins). `python tools/disk_report.py` is a read-only report of what `pg6*` dirs and `work/` use
(it never deletes).

**Useful switches:** `-PresentMode auto|fifo|mailbox|immediate`, `-MaxFps N` (cap the window present rate, 0 = off),
`-ConsoleTiming` (honor the title's present sync interval, `--present-interval honor`), `-Record` (save your real input to `%LOCALAPPDATA%\playgta6\recordings\`), `-Keymap <file>`, `-FreshSetup`, `-NoBuild`,
`-DryRun`, `-Name <run name>`.

**Ray tracing** is wired in the runtime (residency, BVH helper, fault readback) but stays off until the translator
emits RT shaders ([RAYTRACING_DESIGN.md](RAYTRACING_DESIGN.md)).

**What to expect today.** Setup screens, loading, then the Bink videos. After about 110 s the game stops at the
device-ID owner decision (`XSystemGetAppSpecificDeviceId`, see [HARD_STOPS.md](HARD_STOPS.md)). The window stays open
on the last frame with the title `GUEST FAULT ...`, and the console prints the fault report. Close the window to exit.
Nothing past that point has been seen yet.

**Troubleshooting**
- Use **PowerShell**, not `cmd.exe` (the `.ps1` scripts and `$env:` variables do not work in cmd).
- "Pass -Exe and -GameDir": the variables are not visible; set them and open a new PowerShell window.
- Warning that the checkout is **detached** or **behind origin/main**: run with `-Update`.
- Browser build says **emsdk not found**: see Prerequisites below (clone emsdk into `work/emsdk`, `emsdk install 7.0.1`,
  `emsdk activate 7.0.1`). The native path does not need emsdk.
- `run directory ... already exists`: that run name was used before; drop `-Name` (a unique name is generated) or pick
  another.

---

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
Useful switches: `-NoBuild` (fail on stale), `-TrustBuild` (reuse artifacts), `-DryRun` (resolve everything and print the exact `python tools/run_*_boot.py ...` command, plus the prebuilt staleness line, then exit 0 without building or launching; same in `play_native.ps1`),
`-Chrome <path>`, `-DurationS`. Input presets: start-screen, menu-flow, calibration-skip, intro-watch,
legal-notices, chrome-menu-flow, chrome-menu-flow-2, chrome-menu-flow-4, chrome-menu-flow-5, chrome-full-boot,
menu-to-ingame, chrome-to-ingame (every preset `tools/run_browser_boot.py` accepts; `chrome-to-ingame` goes through
the LEGAL NOTICES and ROCKSTAR GAMES LEGAL TERMS screens automatically, then into the Main Menu / in-game). Defaults come from the env vars `PLAYGTA6_EXE`, `GTA6_GAME_DIR`, `CHROME_PATH` and `PLAYGTA6_SHADER_PACK`.

### Prebuilt (no compiling when you just want to play)
Build once (hours the first time, incremental after; never starts Chrome), then start in seconds:
```powershell
.\tools\update_prebuilt.ps1 -Web          # builds into a temp dir, swaps it in atomically on success
.\tools\play_local.ps1 -Prebuilt         # never builds; prints the source commit and how far behind origin/main it is
```
The directory is `-Dir` (update) / `-PrebuiltDir` (play), else `$env:PLAYGTA6_PREBUILT_WEB_DIR`, else `%LOCALAPPDATA%\playgta6\prebuilt-web`; it holds
`web/`, `boot/`, `build/` (staged shader pack, caches) and `prebuilt.json` (the source commit). It reuses the lift cache and
the previous object cache, so refreshes are incremental. `tools/run_browser_boot.py --build-only` runs just the build stages
(no server, no Chrome); `--web-dir` serves a web root other than `<repo>/web`.
`-InputScript` defaults to `calibration-skip` (`-InputScript none` turns it off). `-ShaderPack` defaults to
`$env:PLAYGTA6_SHADER_PACK`, else `work/splash.wgsl.pack`; if none exists the script prints the build command and stops.
`tools/update_prebuilt.ps1 -Web` derives its browser pack from the full native pack (`-StdPack` / `PLAYGTA6_NATIVE_SHADER_PACK`) with
`tools/subset_wgsl_pack.py`: same stages, WGPK v1 (the web preloader reads v1 only and holds the pack in memory), per-shader cap
256 KiB, total cap 1.6 GB, stages of `PLAYGTA6_SHADER_PACK` kept first, plus the filtered `.reflect.json`. An explicit `-ShaderPack` wins.
A browser pack built only from `pack-rdna2 --max-wgsl-bytes` can miss shaders the title uses after the terms (black loading screen with a spinner).
**Full shader pack (default for `-Prebuilt`).** When `$env:PLAYGTA6_NATIVE_SHADER_PACK` names a WGPK **v2** std pack with its
`.reflect.json` sidecar, `-Prebuilt` serves that whole pack (no 256 KiB cap, all stages) instead of the staged v1 subset and opens the page with
`?streamShaders=1`: the worker reads the v2 TOC and fetches shaders on demand with HTTP `Range` (the local server `server/serve_local.py` answers 206).
The pack is hard-linked (else copied) into its own `build/staged-shaders-stream/` so `/shaders/` holds only that pack. `-StreamPack <file>` picks another
pack, `-StreamShaders` forces it without `-Prebuilt` (stage dir `work/staged-shaders-stream`), `-NoStreamShaders` forces the v1 subset; an explicit
`-ShaderPack` also wins. A missing/v1 pack or missing sidecar silently falls back to the subset (an explicit `-StreamPack` errors instead).
`.	ools\play_local.ps1 -Prebuilt -Exe ... -GameDir ...` is all you need.
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
`-sMEMORY64=1 -pthread -fwasm-exceptions -fPIC`; `tools/build_guest_wasm.py --split-modules 2 --split-strategy capped
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

### Rebinding keys and mouse look: `keymap.ini`
`--host-input` loads `%LOCALAPPDATA%\playgta6\keymap.ini` (written with the documented defaults on the first run) or the
file given by `--keymap <file>` (`play_native.ps1 -Keymap`). Lines are `Key = target` under `[keys]` (e.g. `Enter = a`,
`X = right_trigger`, `Mouse_Left = right_trigger`); the file is applied on top of the table above, `Key = none` unbinds,
and each key may appear once. `[mouse]` has `look = on|off` (real mouse movement drives the right stick),
`sensitivity` (stick deflection per pixel per 8 ms poll, default 0.05), `invert_y` and `release_key` (default `F8`, same as `[hotkeys] mouse_release`).
`[hotkeys]` rebinds the live window's own keys as `name = Key` or `name = Ctrl+Key`: `fullscreen` (F11), `screenshot` (F12),
`mouse_release` (F8), `overlay` (F3), `target_mip` / `target_layer` / `target_prev` / `target_next` (Ctrl+F7..F10). Two hotkeys cannot share
a key and a plain hotkey cannot reuse a key bound in `[keys]`; either is a clear load error (`[hotkeys]: hotkeys 'a' and
'b' are bound to the same key`).
With mouse look on, the live window uses Win32 Raw Input (relative deltas, only while the window is focused) and
captures the cursor: it is hidden and confined to the window, so looking never stops at the screen edge. The deflection
decays after the mouse stops, like a released stick. The capture is released on focus loss and when minimised; the
release key toggles it by hand and a click in the window re-captures. Esc stays the pad Menu button. Without a live
window the old cursor-delta path is used. A bad file stops the
run with `file: line N: ...` messages. Only real host input is mapped.

### Recording and replaying a session: `--record-input`
`playgta6_boot --host-input --record-input <file>` writes every real keyboard/pad reading (each change of the
state, nothing synthesized) in the `--input-script` format: a `# playgta6-input-script v2` line, then
`state <present> <buttons> <lx> <ly> <rx> <ry> <lt> <rt>   # <ms> ms` lines. `--input-script <file>` replays it: each
state is written once the present counter reaches its index, so the sequence is the same however fast the run is.
`play_native.ps1 -Record` records to `%LOCALAPPDATA%/playgta6/recordings/<name>.txt` (`-RecordFile` for a path).

## Native play: one command

```powershell
.\tools\play_native.ps1            # PLAYGTA6_EXE / GTA6_GAME_DIR as for play_local.ps1
.\tools\play_native.ps1 -FreshSetup  # new empty save dir: the setup screens appear again
```

Builds what is stale (recompiler, lift, harness with the Dawn executor on, all from the content-keyed caches below),
then runs `playgta6_boot --window --audio-out --host-input --shader-pack <pack>`: a window with the live frames, WASAPI
audio, keyboard and pad.

### Staying current: `-Update`

Both `play_native.ps1` and `play_local.ps1` warn at every start when the checkout is N commits behind `origin/main`.
Pass `-Update` to fast-forward: with no uncommitted tracked changes it switches to `main` if HEAD is detached, runs
`git fetch origin` and `git merge --ff-only origin/main`, and prints `old -> new`. With local changes, on another branch,
or if `main` cannot fast-forward it prints why and continues with the current code; it never resets or discards anything.
Logic: `tools/repo_update.py` (tests: `tests/test_repo_update.py`).


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

### A/B the audio thread's MMCSS class: `-AudioMmcss`

The WASAPI render thread registers with MMCSS ("Pro Audio") for glitch-free output. If `--audio-out` runs show a lower fps
or more missed vblanks than without audio, A/B the scheduling class:
`.\tools\play_native.ps1 -AudioMmcss off` (also `audio`, `pro-audio`; default `pro-audio`). Same switch: `--audio-mmcss
pro-audio|audio|off` on `playgta6_boot` / `run_native_boot.py --play`, or env `PLAYGTA6_AUDIO_MMCSS` (the flag wins).
`off` leaves the thread at normal priority (no MMCSS, no priority boost). The `audio_out` block of `boot-report.json` records `mmcss_mode`
and `mmcss_registered` (false when `off` or when avrt.dll refused), so compare fps p5 / missed vblanks per mode.

### Debugging a black or garbled screen: `--show-target`

`playgta6_boot --window ... --show-target <selector>` (from `run_native_boot.py --play`: `-- --show-target <selector>`)
shows any render target instead of the swapchain image, read-only (the game's frame is not changed). Start with
`--show-target cycle` and press **Ctrl+F10** / **Ctrl+F9** to walk the frame's render targets (the title bar names the
current one), **Ctrl+F8** for the next array layer, **Ctrl+F7** for the next mip (they only act with `--show-target`, so
plain F8 stays the mouse release; rebind via `[hotkeys]` `target_*`). Or pick directly: `pass:12` (the colour
target of the 12th render pass of the frame), `last`, `<texture id>`, `depth:<id>` (grey; add `@range=0.9:1` for
reverse-Z or tight depth), `@layer=2`, `@mip=1`. Offline: `pg6_native_exec --trace f.pg6trace --show-target pass:5
--digest-at 30` runs headless and the digest / non-black counts it prints are those of the shown target (the headless swapchain).
Details: `docs/NATIVE_EXECUTOR_DESIGN.md`, "Present any target".

## Native boot: fast, safe repeat builds

`python tools/run_native_boot.py --build-only ...` runs only the (timed) build stages
(`recomp`, `program`, `harness`) and exits before the run; each stage prints `done in N s`.
Repeat builds avoid the three expensive stages by content-keyed caches that sit beside the
object cache (`<object cache>-lift`, `<object cache>-recomp`):

| Stage | Skipped when | Key |
|---|---|---|
| `recomp` (cargo release build, cold in a fresh worktree) | a binary for the same recompiler sources is cached; or the stamp `target/release/playgta6-recomp.key` matches | sha256 of every workspace crate's `src/`, manifests, `build.rs`, `Cargo.lock`, plus `rustc -vV` |
| `program` (discovery + lift) | the lift cache has the entry, or `program/.lift-key` matches (#1475) | recompiler sources, exe, offered DLLs, shim headers, lift options |
| `program`, discovery per module (on a whole-program miss) | `<shard lift cache>/discovery` has the module's entry | discovery sources (all but the lifter) + `rustc -vV`, module and guest image bytes, seeds ([BUILD_CACHING.md](BUILD_CACHING.md)) |
| `program`, per shard (on a whole-program miss) | the shard lift cache `<object cache>-lift-shards` has the shard | per shard: its functions' CFGs and callee flag summaries, lifter source hash, target, memory model, dispatch, probes ([BUILD_CACHING.md](BUILD_CACHING.md)) |
| `harness` (~389 shards) | the object cache has every shard object | per shard: IR hash, target triple, CPU, `-O` level, PIC mode, LLVM tool version, retarget code fingerprint |

Overrides: `PLAYGTA6_RECOMP_CACHE=<dir>|off`, `PLAYGTA6_LIFT_CACHE=<dir>|off`,
`PLAYGTA6_SHARD_LIFT_CACHE=<dir>|off` (size cap `PLAYGTA6_SHARD_LIFT_CACHE_GB`, default 24), `--cache-dir`.
Shards are cut at hash-chosen anchor functions, so a recompiler change recompiles only the shard objects
whose functions it changed ([BUILD_CACHING.md](BUILD_CACHING.md)).
Caches hold only hashes, binaries and generated IR, never game bytes in git.

**Build-dir lock.** A run holds `<build-dir>/.build.lock` (heartbeat-kept, taken over after
120 s without heartbeat when its process died) for the whole build and run. A second run on
the same build dir fails immediately with `build directory ... is in use by another run` and
exit code 2; give each parallel run its own `--build-dir` (the default `work/native-boot` is
per checkout, so separate worktrees never collide). Never kill cargo/rustc/python to free a
lock; wait or pick another build dir.
