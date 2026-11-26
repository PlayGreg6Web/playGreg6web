# Status: how to play, how far it gets, what to use when it stops

Short and kept current: update the numbers below whenever a PR changes them. Details live in the linked
docs. No game files are in this repo; the executable and package stay outside (see [PLAY_LOCALLY.md](PLAY_LOCALLY.md)).

## Play natively (Windows, Dawn executor): the recommended way

```
$env:PLAYGTA6_EXE = "D:\game\game.exe"; $env:GTA6_GAME_DIR = "D:\game\package"
.	ools\play_native.ps1 -FreshSetup  # new empty save dir so the setup screens appear
.	ools\play_native.ps1              # later runs: builds what is stale, live window with audio and input
```

- Env vars: `PLAYGTA6_EXE`, `GTA6_GAME_DIR`, `PLAYGTA6_NATIVE_SHADER_PACK` (else the newest `*.wgsl.pack`
  under `work/` is picked and printed; the std pack is picked automatically). Flags: `-MaxMs`, `-SaveDir`,
  `-InputScript`, `-NoBuild`, `-DryRun`.
- Controls: keyboard and the first XInput pad; the key table is in [PLAY_LOCALLY.md](PLAY_LOCALLY.md)
  (Enter/Space = A, Backspace = B, WASD = left stick, arrows = D-pad, Esc = Menu). F11 toggles fullscreen, F12 saves a PNG screenshot (#1827).
- Mouse look is opt-in: `keymap.ini` `[mouse] look = on` captures the cursor via Raw Input; F8 releases it (all window hotkeys: keymap.ini `[hotkeys]`), a click
  re-captures (#1828). Also: key rebinding via `keymap.ini` (#1822), `-PresentMode`/`-MaxFps` (#1812), `-Record`
  (#1801), rumble forwarded to the XInput pad (#1809).
- The window scales per monitor DPI, keeps a minimum size, restores its last size and position, and the frame
  is letterboxed into it (#1736, #1739). Timer resolution is raised and `[live]` fps telemetry is printed (#1743).
- A persistent Dawn blob cache makes later starts faster (#1750). A live-mode stall is fixed (#1733).
- Window title shows `fps | frame | draws/frame`. Close the window or press Ctrl+C to stop.
- At a guest fault the window stays open on the last frame, titled `GUEST FAULT ...`, and the console prints
  the `fault_report.py` output right away. Close the window to finish the report.
- Saves persist under `%LOCALAPPDATA%\playgta6\saves`; `tools/saves.py` exports/imports them as a zip (#1811).
- Stability: concurrent-decommit faults in GPU buffer hashing and flush are fixed (#1817, #1818).
- `-Update` (native and browser scripts) fast-forwards the checkout to `origin/main` when it has no uncommitted tracked changes; otherwise it says why and continues. Every start warns when behind `origin/main`.

## Play in the browser

```
.\tools\play_local.ps1             # build if stale, local server, headed Chrome
.\tools\play_local.ps1 -InputScript chrome-to-ingame   # scripted input from the Chrome screens to in-game
```

- Needs Emscripten: the script finds `emcc` on PATH, `EMSDK`, or an install under `work/**/emsdk` and prints
  a fix hint otherwise (#1651). Also needs a shader pack (`work/splash.wgsl.pack` is picked up automatically).
- Default `-InputScript` is `calibration-skip`; `none` turns scripting off. Details and the boot options are in
  [PLAY_LOCALLY.md](PLAY_LOCALLY.md).

## How far the boot gets

How to run it: [quick start](PLAY_LOCALLY.md#quick-start-play-natively-windows).

Setup screens, loading, then the Bink videos decode and play. A headless run reaches about 110 s and
**stops on `XSystemGetAppSpecificDeviceId`** (a trapping stub on purpose, an owner decision). Nothing past it
has been seen. Music has not started yet; the audio probe counters in the boot report (#1737) are the way to
diagnose that. Std shader pack reflection covers 31,777 of 31,822 shaders (#1735).

Owner decisions (full table in [HARD_STOPS.md](HARD_STOPS.md); agents implement none of them). The decision
brief recommends, and both are still pending (options and consequences in
[HARD_STOPS.md section 7](HARD_STOPS.md#7-pending-owner-decisions-device-id-and-xuser-sign-in-state)):

1. `XSystemGetAppSpecificDeviceId` answers with an error instead of a device id (recommended).
2. `XUser` sign-in succeeds for an offline local user (recommended).

Further open items: `XUserGetTokenAndSignatureResult` (slot 25) offline HRESULT, the long-term HRESULT of
`XStoreAcquireLicenseForDurables` (slots 71/72), and the other XStore licence slots (20/21/22/65/68).

## Ray tracing

Runtime pieces are merged (residency of BVH/address sources, fault readback, WGSL helper with CPU twin;
#1802, #1808, #1815, #1823) but RT stays **off** until the translator emits RT shaders. See
[RAYTRACING_DESIGN.md](RAYTRACING_DESIGN.md).

## Speed today

- Native live window: about 55 to 60 fps in the menus.
- Native headless: about 58 fps.
- Rough, scene-dependent figures; refresh them from a boot report (`gpu_time_top.py`, `[live]` lines, window title).

## Diagnostic tools

| Tool | Use it for |
|---|---|
| `tools/fault_report.py <boot-report.json>` | Compact fault record: rip, lifted function, call stack, last dispatched entries |
| `tools/triage_fault.py <boot-report.json> --pe <exe>` | Fault to recompiler explanation of the faulting indirect site and a fix category (triage, #1688) |
| `tools/program_census.py` | Why shader programs cannot run, by reason, count and unresolved binding kind ([PROGRAM_CENSUS.md](PROGRAM_CENSUS.md), #1738) |
| `tools/disk_report.py` | Read-only disk usage of `pg6*` dirs and `work/`: size, kind, in-use marks, safe-to-delete hint plus the command to run yourself (never deletes) |
| `tools/gpu_time_top.py <boot-report.json>` | Top GPU-time pipelines from the report |
| `tools/replay_compare.py` | Same capture through two builds, to prove identical rendering ([REPLAY_COMPARE.md](REPLAY_COMPARE.md)) |
| `playgta6-recomp survey-jumps <exe>` | Census of unresolved indirect jumps, classified; recorded targets extend reachability ([INDIRECT_JUMP_SURVEY.md](INDIRECT_JUMP_SURVEY.md), #1770) |
| `playgta6-recomp opcode-census <exe>` | Instruction mix of the executable, to prioritise lifter work |
| `tools/boot_summary.py <report|run dir>` | Short human summary of a boot report (ending incl. fault symbols and hard-stop status, screens, fps percentiles, GPU errors, shader pack, drops, audio, bink, memory, prewarm); `--compare A B` shows deltas; `run_native_boot.py` / `play_native.ps1` print it after every run |
| `tools/run_history.py --root H:/ [--pattern 'pg6*'] [--since 3d] [--group-by commit] [--fail-on-regression]` | Read-only time-sorted table of past boot runs (commit from report/prebuilt.json, preset, duration, ending + hard-stop status, max screen, fps p50/p5, GPU errors, pack misses, underruns, private peak); `--group-by commit` compares commits per preset, `--fail-on-regression` exits 1 for scripted use |

Reports hold hashes, addresses and counts only, never game bytes.
