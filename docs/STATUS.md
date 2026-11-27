# Status: how to play, how far it gets, what to use when it stops

Short and kept current: update the numbers below whenever a PR changes them. Details live in the linked
docs. No game files are in this repo; the executable and package stay outside (see [PLAY_LOCALLY.md](PLAY_LOCALLY.md)).

## Play natively (Windows, Dawn executor): the recommended way

```
$env:PLAYGTA6_EXE = "D:\game\game.exe"; $env:GTA6_GAME_DIR = "D:\game\package"
.\tools\play_native.ps1 -FreshSetup  # new empty save dir so the setup screens appear
.\tools\play_native.ps1              # later runs: builds what is stale, live window with audio and input
.\tools\update_prebuilt.ps1          # build once into a folder (atomic swap, prebuilt.json), incremental after
.\tools\play_native.ps1 -Prebuilt   # start that build in seconds: never compiles, prints its commit and how far behind main it is
```

- Env vars: `PLAYGTA6_EXE`, `GTA6_GAME_DIR`, `PLAYGTA6_NATIVE_SHADER_PACK` (else the newest `*.wgsl.pack`
  under `work/` is picked and printed; the std pack is picked automatically), `PLAYGTA6_PREBUILT_DIR` (native
  prebuilt, default `%LOCALAPPDATA%\playgta6\prebuilt`), `PLAYGTA6_PREBUILT_WEB_DIR` (browser prebuilt, default
  `...\prebuilt-web`). Flags: `-Prebuilt` (#1854, #1866), `-ConsoleTiming`, `-MaxMs`, `-SaveDir`, `-InputScript`, `-NoBuild`, `-DryRun`.
- `-DryRun` (#1930) prints the resolved command and exits without building or launching. `-Prebuilt` prints a
  one-line staleness hint (commits to runtime, recomp, harness or shaders since that build; never blocks; #1926).
- `update_prebuilt.ps1` boots the fresh build headless first and swaps it in only if that smoke boot passes;
  on failure the old prebuilt and the `.new` folder stay. `-SkipSmoke` opts out, `-DryRun` prints the plan (#1948).
  A prebuilt is read-only for runners: run dirs, lock and marker go outside it, deletions inside it refuse (#1950);
  each gate run uses its own smoke run name, reads only its own report, and keeps the last 3 smoke runs (#1972).
- `--audio-mmcss pro-audio|audio|off` (env `PLAYGTA6_AUDIO_MMCSS`, `play_native -AudioMmcss`) sets the WASAPI render
  thread's MMCSS class; the mode and registration result show in the `audio_out` report (#1951).
- Input: the most recently active XInput pad drives the local user (switch on activity, reconnect in another slot
  works; rumble follows it, #1965); keys held with Alt never reach the guest, so Alt+Enter no longer taps A (#1964);
  focus loss releases the pad too (#1961).
- `-ConsoleTiming` (#1861) honors the title's present sync interval: menus run at the console's 30 fps instead of
  about 60 (default `--present-interval min1`); gameplay at interval 1 is the same.
- `python tools\trim_caches.py [--apply]` reports and LRU-trims all build caches to budgets (dry run by default); builders
  auto-trim when free disk < `PLAYGTA6_MIN_FREE_GB` (default 20). See [BUILD_CACHING.md](BUILD_CACHING.md).
- One cache root (`PLAYGTA6_CACHE_ROOT`, default `%LOCALAPPDATA%\playgta6\cache`, or `auto` = drive with most free space) holds the
  object, lift, shard-lift, recomp, targets and wasm caches for every tool; legacy dirs get a one-time hint, nothing is moved.
- F3 toggles a stats overlay (fps, frame-time graph, present mode, draws, GPU ms, underruns, fault text; #1852).
  All window hotkeys sit in one table, rebindable in `keymap.ini [hotkeys]`; `--show-target` keys are Ctrl+F7..F10 (#1844).
- A GPU device loss is recovered by the native executor instead of ending the run (#1836). Headless run of the same
  live path: `run_native_boot.py --live-executor` (#1847), no window.
- Build speed: runtime libs build while shards compile, precompiled headers (#1853); the default object pass list is
  tuned (`build_native_boot.py -O t`; `-O f` is faster codegen; #1878).
- Build times (game exe, warm caches; details in [BUILD_CACHING.md](BUILD_CACHING.md)): runtime-only change ~20-60 s;
  shim/no-op ~1 min; lifter change ~1.5-2 min + cargo (discovery cache hit, few shards); discovery rule change ~3.5-4 min
  (discovery ~155 s, 1 shard); new exe or cold cache ~8-9 min (compile 281 s). Players use `-Prebuilt` and never compile.
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
- `.\tools\update_prebuilt.ps1 -Web` then `.\tools\play_local.ps1 -Prebuilt` plays a prebuilt browser build without
  compiling (#1862, #1877). With `PLAYGTA6_NATIVE_SHADER_PACK` pointing at a v2 std pack it streams the whole pack
  with HTTP Range by default (`-StreamShaders`, `-NoStreamShaders`; #1876); otherwise `update_prebuilt -Web` derives a
  v1 subset from the std pack (#1873). Dir: `PLAYGTA6_PREBUILT_WEB_DIR`.
- Default `-InputScript` is `calibration-skip`; `none` turns scripting off. Details and the boot options are in
  [PLAY_LOCALLY.md](PLAY_LOCALLY.md).

## How far the boot gets

How to run it: [quick start](PLAY_LOCALLY.md#quick-start-play-natively-windows).

Setup screens, loading, then the Bink videos decode and play. A headless run reaches about 110 s and
**stops on `XSystemGetAppSpecificDeviceId`** (a trapping stub on purpose, an owner decision). Nothing past it
has been seen. Music has not started yet. What the diagnostics (#1947, #1954, #1958) showed: the audio `.rpf` opens
happen inside the package (`PG6_AUDIO_TRACE` logs each audio-path `CreateFile`), and the music controller never gets
a start event before the device-ID stop, so the missing music is a consequence of that stop, not a separate bug. Std shader pack reflection covers 31,777 of 31,822 shaders (#1735).

Owner decisions (full table in [HARD_STOPS.md](HARD_STOPS.md); agents implement none of them). All three are
still pending; options and consequences in
[HARD_STOPS.md section 7](HARD_STOPS.md#7-pending-owner-decisions-device-id-and-xuser-sign-in-state):

1. Device ID (`XSystemGetAppSpecificDeviceId`): answer with an error, no id (recommended).
2. `XUser` sign-in state: "signed in" for the single offline local user, no tokens (recommended).
3. Store licence (`XStoreAcquireLicenseForDurables` 71/72, slots 20/21/22/65/68): the store brief recommends
   truthful offline / not-found answers and no ownership claims.

Also open: `XUserGetTokenAndSignatureResult` (slot 25) offline HRESULT. `XNetworking` slots 3/4/5/8/9/10/14 already
answer offline (network unreachable, #1872); the security/certificate ones are hard stops and never vouch for a server.

Open investigations (no owner decision needed, nothing merged that fixes them yet):

- Title job-graph stall: a run stalls with the title's own `int3` assert in its job graph; the fault record gives
  the address and call stack (see [WHEN_IT_CRASHES.md](WHEN_IT_CRASHES.md)).
- Music: expected to start only after the device-ID decision (see above); recheck once the stop is answered.
- Browser: an `INDIRECT_CALL_MISS` ending is waiting for a crash report from a real browser run
  ([WHEN_IT_CRASHES.md](WHEN_IT_CRASHES.md) says what to send).

Since 2026-11-26 (merged): a stopped or crashed run now shows an always-visible banner ("Stopped: <slot> - needs an
owner decision" vs "Crashed: <kind> at <site> - report: <dir>"), printed again at the end of `play_native` and
browser runs (#1929, #1933); `--dump-on-fault` minidumps carry the thread context and fault record (#1925); the
host-fault races in GPU hashing and shared-page views are closed (#1903, #1913, #1916); GPU: compute-written float
and stencil textures are read back into guest memory (#1937, #1940, #1944), `ExecuteIndirect` with root constants
(#1941), a per-frame budget for async pipeline creation so a slow pipeline skips the draw instead of freezing
the frame (#1936, #1943); audio output is event-driven and underruns are split into `host_starved` and
`guest_stalled` (#1907, #1924); `inet_ntoa` and `HCTraceImplMessage` are handled (#1945).

Since 2026-11-27 (merged, #1950-#1972): a prebuilt is read-only for runners and its smoke gate is race-free (#1950,
#1972); `--audio-mmcss` (#1951); `xmem` remaps a mirrored window slot in place, so guest loads never see a trap delta
or zero page mid-remap (#1952); Dawn: distinct dummy storage textures, fragment-output writeMask, failed async
pipelines dropped and logged (#1953), `--gpu-error-scopes` attributes device errors to phase, present index and shader
hashes (#1956), validation tests (#1959); the recompiler recognises unguarded `jmp [base+idx*8]` function-pointer
tables as bounded tail-call switches (#1955); discovery is faster (incremental predecessor index #1962, per-phase
`--timings` and state kept across refinement rounds #1971); harness: `frame_breakdown` (#1963, #1968),
`--profile-guest[=hz]` sampling profiler (#1967), minidumps written from a helper process plus a watchdog backstop
for a stuck report path (#1969), dstorage stall line (#1960) and `streaming_drain` report block (#1966); a dirty-page
driven GPU buffer flush cut it from about 9.0 to about 4.5 ms/frame (#1970, `--flush-crosscheck` verifies it).
Latest soak: 12/12 boots end at the `XSystemGetAppSpecificDeviceId` owner-decision stop, 0 host AVs, 0 title traps,
0 hangs, fps p50 58.8.

## Ray tracing

Runtime pieces are merged (residency of BVH/address sources, fault readback, WGSL helper with CPU twin;
#1802, #1808, #1815, #1823) but RT stays **off** until the translator emits RT shaders. See
[RAYTRACING_DESIGN.md](RAYTRACING_DESIGN.md).

## Speed today

- Native live window: about 55 to 60 fps in the menus.
- Native headless: about 58 fps (soak p50 58.8).
- The GPU buffer flush costs about 4.5 ms/frame (was 9.0, #1970); `frame_breakdown` shows where the rest goes.
- Rough, scene-dependent figures; refresh them from a boot report (`gpu_time_top.py`, `[live]` lines, window title).

## Diagnostic tools

A run stopped or crashed? Start with [WHEN_IT_CRASHES.md](WHEN_IT_CRASHES.md).

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
| `tools/boot_summary.py <report|run dir>` | Short human summary of a boot report (ending incl. fault symbols and hard-stop status, screens, fps percentiles, GPU errors, shader pack, drops, audio, bink, memory, prewarm); `--compare A B` (run dirs or report paths) shows a side-by-side A->B diff (fps, hitches, underruns by class, drops and binding fallbacks per phase, fault kind/site, wall time) with regressions and improvements marked, `--compare-last [N]` diffs the newest run against the one N before it (run_history discovery, `--root`, default H:/); `run_native_boot.py` / `play_native.ps1` print it after every run |
| `tools/run_history.py --root H:/ [--pattern 'pg6*'] [--since 3d] [--group-by commit] [--fail-on-regression]` | Read-only time-sorted table of past boot runs (commit from report/prebuilt.json, preset, duration, ending + hard-stop status, max screen, fps p50/p5, GPU errors, pack misses, underruns, private peak); `--group-by commit` compares commits per preset, `--fail-on-regression` exits 1 for scripted use |

Also: `tools/surface_survey.py` (static ranked survey of trapping COM slots and unresolved imports reachable from boot, #1870),
`tools/minidump_stack.py` (host stack of a fault minidump, #1858). Boot reports carry build/run metadata (commit, dirty,
profile, executor, preset; #1864) and a whole-run per-phase frame-time histogram with 1%/0.1% lows (`frame_times`, #1879)
that `boot_summary.py` and `run_history.py` print; both also read browser run dirs (#1874). `frame_breakdown` splits each
phase's (and each 5 s window's) mean frame into the present thread's CPU (guest code vs our D3D12 flush / append / submit),
its blocked spans (guest waits, condvars, contended locks, sleeps, frame events, pacer, executor backpressure), process CPU,
executor time and upload / shadow-compare volume; `live_pump` adds the executor queue's totals.
`--profile-guest[=hz]` adds a sampling profiler (per-thread flat and inclusive counts in `guest_profile`, collapsed
stacks for flamegraph tools; #1967); `--gpu-error-scopes` attributes Dawn errors (#1956); `streaming_drain` shows
what the title's drain-all-streaming loop waits on (#1966); recomp `--timings` gives per-phase discovery times (#1971).

Reports hold hashes, addresses and counts only, never game bytes.
