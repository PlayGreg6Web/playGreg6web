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
  (Enter/Space = A, Backspace = B, WASD = left stick, arrows = D-pad, Esc = Menu). F11 toggles fullscreen.
- The window scales per monitor DPI, keeps a minimum size, restores its last size and position, and the frame
  is letterboxed into it (#1736, #1739). Timer resolution is raised and `[live]` fps telemetry is printed (#1743).
- A persistent Dawn blob cache makes later starts faster (#1750). A live-mode stall is fixed (#1733).
- Window title shows `fps | frame | draws/frame`. Close the window or press Ctrl+C to stop.
- At a guest fault the window stays open on the last frame, titled `GUEST FAULT ...`, and the console prints
  the `fault_report.py` output right away. Close the window to finish the report.
- Saves persist under `%LOCALAPPDATA%\playgta6\saves`.

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

Setup screens, loading, then the Bink videos decode and play. A headless run reaches about 110 s and
**stops on `XSystemGetAppSpecificDeviceId`** (a trapping stub on purpose, an owner decision). Nothing past it
has been seen. Music has not started yet; the audio probe counters in the boot report (#1737) are the way to
diagnose that. Std shader pack reflection covers 31,777 of 31,822 shaders (#1735).

Owner decisions (full table in [HARD_STOPS.md](HARD_STOPS.md); agents implement none of them). The decision
brief recommends, and both are still pending:

1. `XSystemGetAppSpecificDeviceId` answers with an error instead of a device id (recommended).
2. `XUser` sign-in succeeds for an offline local user (recommended).

Further open items: `XUserGetTokenAndSignatureResult` (slot 25) offline HRESULT, the long-term HRESULT of
`XStoreAcquireLicenseForDurables` (slots 71/72), and the other XStore licence slots (20/21/22/65/68).

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

Reports hold hashes, addresses and counts only, never game bytes.
