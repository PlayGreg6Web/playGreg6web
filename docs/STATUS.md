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

Setup screens, loading, then the Bink videos decode and play, then the title streams its world. Boots no longer end in
a page-table crash inside `sub_140F2C470` (a page with two owners): the root cause (our
`GetResourceAllocationInfo` size for some streamed textures) is fixed (#2101), and the boot now reaches the in-world 3D
scene (about 55-100 s). Remaining issues: low fps in the scene, black frames (under investigation), and the #2033
store-back optimisation is still out (reverted again, #2115). Music has not started yet; recheck it. What the diagnostics (#1947, #1954, #1958) showed: the audio
`.rpf` opens happen inside the package (`PG6_AUDIO_TRACE` logs each audio-path `CreateFile`). Std shader pack reflection
covers 31,777 of 31,822 shaders (#1735).

Owner decisions (full table in [HARD_STOPS.md](HARD_STOPS.md)) are implemented as recommended (#2029, #2031): the device
id answers `E_NOTIMPL`, `XUser` reports local sign-in state, `XStore` slots 28/65 answer truthfully offline; the store
licence slots stay offline / not-found with no ownership claims.

Also open: `XUserGetTokenAndSignatureResult` (slot 25) offline HRESULT. `XNetworking` slots 3/4/5/8/9/10/14 already
answer offline (network unreachable, #1872); the security/certificate ones are hard stops and never vouch for a server.

Open investigations (nothing merged that fixes them yet):

- Title job-graph stall: a run stalls with the title's own `int3` assert in its job graph; the fault record gives
  the address and call stack (see [WHEN_IT_CRASHES.md](WHEN_IT_CRASHES.md)).
- Music: has not started; recheck now that the page-table crash is fixed.
- In-scene fps is low and some frames are black; both are under investigation.
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

Since #1972 (merged, #1973-#1982): the GPU buffer flush also page-indexes relocated uniform ranges outside every tracked buffer and the report counts `gpu_work.flush_buffer_ns` / `flush_uniform_ns` (#1973); the boot report has a `guest_sleep` block (requested vs actual wall time of every guest `Sleep`, overshoot histogram) and `--timer-resolution 1ms|default` for A/B (#1975) - the histogram shows guest `Sleep` is not tick-bound; the `bink` block counts null `Bink*` export lookups (`lookup_null`, first distinct names) and logs `BinkOpenWithOptions` stream creates (`open_log`: path, flags, result) (#1982); `com_bridge` test flake fixed (#1981). Marc: GPU descriptor heap / command ring / readback gate hardening (#1977), web loading splash and HUD telemetry (#1979).

Since #1982 (merged, #1983-#2022): `--auto-quarantine` drops failed pipelines and `tools/bisect_shader_skip.py` bisects a bad shader (#1984, #1991); the boot report gets a `guest_timed_wait` histogram (finite-timeout waits split by timed out vs signalled, overshoot buckets; #1983) and `boot_summary` shows it (#1987); the minidump helper's process freeze is excluded from the sleep / timed-wait overshoot stats and outliers of 32 ms or more are logged with `frozen_by_dump` (#2017). `ExecuteIndirect` skip counters (`IndirectVertexIndexArgsSkipped`, `IndirectRootViewArgsSkipped`) are in the report (#1992); on the current boot they are all 0, so no further ExecuteIndirect work is needed. End-to-end crash tests cover host fault, watchdog hang and the stuck report path (#1988, #1994, #2012). `update_prebuilt` and the browser prebuilt swap retry the final rename with backoff and roll back a half-done swap (#2010, #2013). The indexed GPU flush visits only reported and unprimed relocated uniform ranges (#1995). Recompiler: TBAA tags split guest-state from guest-memory accesses, about 5% less wasm code (#2000, #2021). Discovery went from 51-73 s to about 17.5 s (parallel full-map scans and bulk membership #2005, bitmap leader / worklist sets and a one-pass predecessor index #2009, refinement round 2 continues through new no-return thunks #2022). CI: the Dawn/DXC FetchContent cache is keyed by repository name after the rename (#2020).

Lazy commit (#2002, Marc) regressed loading: the image loader and host `fread`/`ReadFile` wrote into guest pages that were not yet committed, giving `load_failed ReadFailed`. Fixed by materialising pages before host writes (`HostWriteScope`, #2015) and by auditing every host write path (guest buffers before host `fwrite`, crash handler last in the chain and skipped inside host probes, new `test_lazy_host_access`; #2019).

Marc, one line each: GPU sRGB+UAV and HDR compute fixes (#1978), colour/gamma frame audit CLI (#1911), `interpolate(flat)` / location patching (#1986), Wasm64 SIMD128 hot loops (#1997), `[fault-record]` delivery and `/pg6_dump` flush (#1996), UCRT path shims (#1998), WGSL discard / EXP fixes (#1999), `pdata_interior` and jump-table recovery (#2001), `render_state_packets` hardening (#2003), pooled FD budget fix (#2004), opaque alpha on the base plane (#2006), Wave-4/5 GPU hang guards, probe timeout and device-lost replay (#2007, #2016, #2018), loader Present backpressure (#2008), RT BVH prelude race and console WebGPU features (#1990), high-throughput Range server and OPFS L2 (#1980), Win32/COM and XMA2/Opus audio hardening (#1976).

Repo rename: the project is now PlayGreg6Web (`PlayGreg6Web/playGreg6web`); main history was rewritten, so old clones and branches must be re-cloned or recreated from `origin/main`. Old GitHub URLs redirect. The local save folder is still `%LOCALAPPDATA%\playgta6\saves`.
Since #2022 (merged, #2025-#2045): **the owner decisions are implemented**: the four denied Media Foundation imports are bound as failing shims (#2029); the device id answers `E_NOTIMPL`, `XUser` reports local sign-in state with find/compare, and `XStore` slots 28/65 answer truthfully offline (#2031). `XGTextureAddressComputer` slot 9 `CopyFromSubresource` (GFX10 detile into linear rows, inverse of slot 8) is implemented (#2034). GPU: address-texture LRU counters in the boot report (#2032), pooled textures are cleared GPU-side and pool chains above 64 KiB are pooled (#2036). The vmem flush serves precollected takes in one pass (#2027). Recompiler: #2033 skipped store-backs of registers that still match `%state`, but later evidence on its effect was inconclusive and it was reverted (#2040). Lifter/runtime `%state` boundary contract documented and tested (#2043); the boot report has a `blocked_waits` census (parked waiters, object state, last signaler; #2044); the shard lift-cache key now covers callee `touched_flags` / `written_flags`, fixing stale caller IR (#2045).

Marc, one line each: web ClearTexture (opcode 63) in worker and `render_state_packets` (#2042), loader `SHARED_WASM_STUB_*` TDZ fix (#2041), RDNA2 divergent-branch descriptor load defs (sampler-heap RootDescriptor leaks, #2038), async replay bind-group loss and dump caps (#2037), precompiled/reused split side modules and bounded caches (#2035).

Since #2045 (merged, #2046-#2099): import thunks and `XMem` hooks honour `unwind_pending` and sync probed arguments (#2046). The live pipeline prewarm builds only while the live pump is idle, which fixed the std-pack menu-to-ingame hang (83 -> 2509 presents, #2054). The fault/stop report never waits on a submitter stuck in a full live pump (#2057); the `prebuilt_smoke` gate passes a run ended by the harness's own bound (#2058). Shader compile moved off the queue thread: pack WGSL is scanned once and modules compile on worker threads with pipelines deferred until ready (#2060, #2074), the WGSL text is dropped after its module exists and the fullscreen pipeline wait budget went from 10 s to 1 s per frame (#2084). Texture copies into plain destinations use a `CopyTexture` packet (#2073). `GetResourceTiling` answers the 4 KiB / 256 B modes with the 64 KiB tile rule and the 64 KB answers tile each slice as the title maps them (#2079, #2082); layout answers and the buddy allocator are ruled out as the cause of the `sub_140F2C470` double-owner page (#2071). Device errors on the current boot went from 18,324 to 389: pipeline layouts fitted to the pack WGSL and async failures classified (#2086), buffer bindings fitted to device limits plus an error-kind histogram (#2095), render pipelines built with the attachment formats and sample count of their pass (#2097). The storage relocation cache keeps a frame's working set (LRU index, 64Ki entries / 1 GiB bound), which fixed a 66 s stall and took execute from 29.7 to 5.0 ms/frame (#2093). Small-resource placement follows the D3D12 rule: textures within 64 KiB, 4 KiB blocks (#2098). `update_prebuilt` runs natives through `Invoke-Native` so stderr never raises NativeCommandError (#2099). A real-Dawn mesh-to-compute test and the mesh pipeline contract are documented (#2055); recomp host-diff tests cover aliased bit/BMI forms (#2059).

GDemay, one line each: GFX10 texture copies of contiguous runs (#2088), 4 KiB sparse buffer shadows (#2089), only selected small XG texture mips are snapshotted (#2096), no redundant snapshots for shared web texture uploads (#2094), GPU clears for pooled Wasm guest textures (#2092), first-fit address reservations skip occupied prefixes (#2065), Dawn blob-cache eviction with a min-heap (#2063), web FNV64 verification (#2061), bootset accounting scans (#2062) and PGIM relocation validation (#2064) made cheaper.

Marc, one line each: deferred ranged preload shader compiles remove a 26-40 s GPU stall and device.lost (#2056), sRGB viewFormats / STORAGE_BINDING view fix (#2072), `v_nop_e64` literal consumption and FSSM `s_swappc_b64` vertex fetch inputs (#2077), 1-roundtrip HTTPFS Stat, real DStorage `file_size_` and live `ioWorkerStats` (#2075).

Since #2099 (merged, #2100-#2132): **the page-table crash root cause is fixed**: textures with an UNKNOWN layout are sized at least as their `SW_4KB_S` placement (#2101). Norm16 textures (and 3D/volume ones) are read through a float twin instead of a dummy, and float layout entries bound to unfilterable-only views get the entry's dummy: `bind_group_unfilterable_dummies` went from about 3200 to 0 in a std-pack boot (#2100, #2107, #2123). Evicted small storage relocations return their space through a free list (#2103). #2033 (skipped store-backs of synced registers) was relanded (#2104) and reverted again (#2115); it is still out. Command-list slot 101 has no call site, so the `E_NOTIMPL` stub stays (#2105). The screen timeline marks the segment that folds presents past the 500-segment cap and `boot_summary` prints it as folded (#2106). A reusable safe `--probe` facility (spec or file, JSONL hits, safe guest reads, report counters) with a function-exit hook that logs the real `rax`/`xmm0` (#2108, #2112). The menu-to-ingame input script follows the current boot flow and `tools/screen_flow.py` detects screens (#2117). 3D-scene frame cost: resource-retire scans skip bind groups and redirects that cannot match (#2121), address textures are dropped in batches at the cap (#2128), a memory bind group is rebuilt only when one of its own storage homes moved (#2132). One command buffer per pass (`--no-split-passes` to turn off), so a pass that fails Dawn validation no longer invalidates the whole frame (#2122). The boot report counts XMA frames decoded and bad frames (#2126). Texture views take the bind-group entry's `viewDimension`, and read-only buffer aliases of a writable binding are bound through a copy (#2129). Lifter contract rule 7 and the #2104 trap analysis with a differential IR check of skipped store-backs (#2131).

Marc, one line each: FSSM `@location` remap in the web path (#2113), `CopyTexture` (opcode 62) in the WebGPU worker (#2114), async pipeline prewarm queue and GPU pass timestamp bridge (#2116), per-stage WGSL binding compaction and audit for WebGPU/D3D12 limits (#2118), `wgpu_worker.js` wiring of FSSM remap, prewarm bridge and `gpu_timer` (#2124), mixed-version PGIM range downloads rejected before caching (#2125).

Since #2132 (merged, #2134-#2157): device errors are tagged by kind, so cascade errors (a `missing_entry` placeholder, an invalid CommandBuffer / BindGroup / layout that follows an earlier root error) are counted apart and the report gives `root_total` / `cascade_total` (#2134). `run_native_boot.py` takes an opt-out machine-wide boot lock (stale PIDs are broken, the wait is recorded in the report; #2135). Query heaps are counted per operation (begin / end / resolve by type, unwritten results reported; #2140), and guest query heaps now run on WebGPU query sets with real occlusion and timestamp results, ResolveQueryData writing into the destination buffer (#2157). `boot_summary` shows and compares device error root/cascade, query counters, usage-scope and alias copies, command buffers, address-texture evictions, unfilterable dummies, boot lock wait, guest probes and the screen flow (#2142). The per-pass command-buffer split of #2122 is opt-in again (`--split-passes`): in the in-world scene its passes stalled the GPU and Dawn's Submit blocked until the title's 30 s main-loop watchdog fired (#2143). Usage scopes: read-only bindings are snapshotted and render passes split on uniform|storage and sampled|storage conflicts (#2130). Writable-buffer aliasing goes through scratch twins merged back, with in-pass buffer copy kernels, storage-texture twins and a snapshot reuse cache (#2147). Copies between block-compressed and uncompressed textures of equal block size go through a GPU scratch buffer, and texture copy skips are counted by reason (#2148). A compute dispatch after a copy that ended the pass re-sets the title's compute pipeline, fixing Dawn's "No pipeline set" (#2156).

Marc, one line each: provably sound low-overhead LLVM IR lifting optimizations (#2146), post-lowering Naga IR optimization pass (#2145), staging pool and `render_state_packets` hot-path allocations removed (#2144), hardware-adaptive RAM/VRAM web profiles (#2141), RDNA2 mesh-to-compute lowering coverage (#2127), per-stage WGSL binding compaction and audit (#2118, listed above).

Current boot status: the boot reaches the in-world 3D scene. Open: a fatal stop of the title at about 99 s (being bisected); a GPU device loss ("device hung") somewhere between 110 and 230 s, with the recovery journal going over its budget; snapshot copies of about 474 GB per boot; low fps in the scene and black frames (under investigation); #2033 stays parked (out).

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
