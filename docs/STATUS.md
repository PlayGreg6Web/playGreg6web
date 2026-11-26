# Status: how to play, how far it gets, what to use when it stops

Short and kept current: update the numbers below whenever a PR changes them. Details live in the linked
docs. No game files are in this repo; the executable and package stay outside (see [PLAY_LOCALLY.md](PLAY_LOCALLY.md)).

## Play natively (Windows, Dawn executor)

```
$env:PLAYGTA6_EXE = "D:\game\game.exe"; $env:GTA6_GAME_DIR = "D:\game\package"
.\tools\play_native.ps1            # builds what is stale, opens a live window with audio and input
.\tools\play_native.ps1 -FreshSetup  # new empty save dir so the setup screens appear
```

- Env vars: `PLAYGTA6_EXE`, `GTA6_GAME_DIR`, `PLAYGTA6_NATIVE_SHADER_PACK` (else the newest `*.wgsl.pack`
  under `work/` is picked and printed). Flags: `-MaxMs`, `-SaveDir`, `-InputScript`, `-NoBuild`, `-DryRun`.
- Controls: keyboard and the first XInput pad; the key table is in [PLAY_LOCALLY.md](PLAY_LOCALLY.md)
  (Enter/Space = A, Backspace = B, WASD = left stick, arrows = D-pad, Esc = Menu). F11 toggles fullscreen.
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

Setup screens, then loading, then the run **stops at about 64 s on the device-ID hard stop**
(`XSystemGetAppSpecificDeviceId`, a trapping stub on purpose). Nothing past it has been seen.

Pending owner decisions (full table in [HARD_STOPS.md](HARD_STOPS.md); agents implement none of them):

1. Whether `XUserGetTokenAndSignatureResult` (slot 25) answers with the offline HRESULT like slots 23/24 instead of trapping.
2. What `XSystemGetAppSpecificDeviceId` returns, if anything (this is the current stop).
3. Which HRESULT `XStoreAcquireLicenseForDurables` (slots 71/72) settles on long term.
4. Any policy for the other XStore licence slots (20/21/22/65/68) if the boot reaches them.

## Speed today

- Native live window: about 40 to 55 fps.
- Native headless: about 58 fps.
- These are rough, scene-dependent figures; refresh them from a boot report (`gpu_time_top.py`, window title).

## Diagnostic tools

| Tool | Use it for |
|---|---|
| `tools/fault_report.py <boot-report.json>` | Compact fault record: rip, lifted function, call stack, last dispatched entries |
| `tools/triage_fault.py <boot-report.json> --pe <exe>` | Fault to recompiler explanation of the faulting indirect site and a fix category |
| `tools/program_census.py` | Why shader programs cannot run, by reason and count ([PROGRAM_CENSUS.md](PROGRAM_CENSUS.md)) |
| `tools/gpu_time_top.py <boot-report.json>` | Top GPU-time pipelines from the report |
| `tools/replay_compare.py` | Same capture through two builds, to prove identical rendering ([REPLAY_COMPARE.md](REPLAY_COMPARE.md)) |
| `playgta6-recomp survey-jumps <exe>` | Census of unresolved indirect jumps ([INDIRECT_JUMP_SURVEY.md](INDIRECT_JUMP_SURVEY.md)) |
| `playgta6-recomp opcode-census <exe>` | Instruction mix of the executable, to prioritise lifter work |

Reports hold hashes, addresses and counts only, never game bytes.
