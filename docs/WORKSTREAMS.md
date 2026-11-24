# Workstreams & Parallel Tasks

Work is split into **5 independent tracks** so many contributors (and AI agents) can work in parallel without merge conflicts. Tracks only communicate through the frozen contracts in `docs/io_manifest.md`, `docs/wasm_abi.md`, and `docs/gpu_commands.md`.

> **Rule for all contributors & AI agents:**
> - If a task needs an ABI change, open a small PR touching `docs/` + `runtime/abi/playgta6_abi.h` first and merge it before dependent feature work.
> - **Never stop just because listed items are claimed:** if every task below is `DONE` or `IN PROGRESS`, inspect `docs/END_GOAL.md` (M1 → M4) + `gh pr list --state open`, define the next non-overlapping numbered task (disjoint files from open PRs), claim it with a GitHub Issue (`gh issue create --assignee "@me"`), add it here, and start immediately!

---

## Track 1: Recompiler (`recomp/`, Rust)
**Goal:** Lift the Xbox Series x86-64 PE executable into Wasm64 (`memory64`).

- [x] **R1 — PE Parser & Import Dumper** — `DONE` (PR #21, closes #1) *(needs `GTA6_GAME_DIR` for real test; unit-testable with synthetic PE)*
  - Parse PE64 headers, sections, exports, TLS callbacks, and import tables (`object` / `goblin` crate).
  - Emit a sanitized symbol-only report (`docs/imports_required.md` — DLL + function names only, zero game bytes).
- [x] **R2 — x86-64 Decoder & Function Discovery** — `DONE` (PR #37, closes #13) *(no game files required for unit tests)*
  - Integrate `iced-x86` to decode `.text` blocks, follow calls/jumps, and build a CFG + relocation map.
- [x] **R3 — LLVM IR → Wasm64 Lifter Prototype** — `DONE` (PR #132, closes #14) *(no game files required)*
  - Lift small synthetic x86-64 test functions (arithmetic, branches, memory loads/stores, Win64 ABI calls) to LLVM IR and compile to `wasm64`.
- [x] **R4 — PE64 `.pdata` Exception Directory & Jump-Table Recovery** — `DONE` (PR #85, closes #38) *(no game files required — disjoint file `recomp/src/pdata.rs`)*
  - Parse PE64 `IMAGE_DIRECTORY_ENTRY_EXCEPTION` (`RUNTIME_FUNCTION` begin/end/unwind RVA) and `UNWIND_INFO` to recover exact function boundaries and switch/jump tables without heuristics.
- [x] **R5 — Import Thunk & Runtime Call-Bridge Generator** — `DONE` (PR #93, closes #39) *(no game files required — disjoint file `recomp/src/stubs.rs`)*
  - Given parsed PE64 imports (`R1`), generate C++/LLVM thunk glue tables binding imported Xbox/Win32 DLL symbols to `pg6_win32_*` / `pg6_gdk_*` / `d3d12_xs` runtime shims (and safe panic/log stubs for unresolved symbols).
- [x] **R6 — PE Section Image & Relocation Packer for Wasm64 Memory** — `DONE` (PR #78, closes #40) *(no game files required — disjoint file `recomp/src/image.rs`)*
  - Emit a deterministic Wasm64 static data image + base-relocation table (`.rdata`, `.data`, `.bss`, `.reloc`) so recompiled code sees the exact guest virtual address layout inside `VirtualMemoryArena`.
- [x] **R7 — Lifter Coverage: String Ops, Bit Ops, Atomics & Remaining Integer Instructions (`recomp/src/lift/`)** — `DONE` (PR #205, closes #118) *(no game files required)*
  - Lift `rep movs/stos/cmps/scas/lods`, `bt*`, `bsf/bsr/tzcnt/lzcnt/popcnt`, `cmpxchg`/`xadd`/`bswap`, `rcl/rcr`, `shld/shrd`, `loop`/`jrcxz`, `leave`/`enter`, and `lock`-prefixed atomics, verified by execution and host-CPU differential tests.
- [x] **R8 — Lifter Coverage: SSE/SSE2 Scalar & Packed Float and Integer SIMD (`recomp/src/lift/simd.rs`)** — `DONE` (PRs #175, #176, #215, #233, closes #119) *(no game files required — disjoint file `recomp/src/lift/simd.rs`)*
  - Lift the SSE/SSE2 (+ commonly emitted SSE3/SSSE3/SSE4.1) instructions MSVC x64 code uses onto the XMM slots of the guest state, verified by execution and host-CPU differential tests.
- [x] **R9 — End-to-End Recompile Pipeline Driver (`recomp/src/pipeline.rs`)** — `DONE` (PR #186, closes #120) *(no game files required — disjoint file `recomp/src/pipeline.rs`)*
  - One `playgta6-recomp build <pe> --out <dir>` command chaining parse → `.pdata`-seeded discovery → lift → import thunks → packed guest image, with a deterministic build manifest and coverage report.
- [x] **R10 — MSVC x64 Jump-Table Pattern Recovery (`recomp/src/jump_table.rs`)** — `DONE` (PR #178, closes #149) *(driven by M2 discovery: 9,969 unresolved switch tables in `game_scarlett_master.exe`)*
  - Expand `recomp/src/jump_table.rs` to recover MSVC image-base-relative (`lea rBase, [rip+...]; movsxd/mov reg, [rBase + idx*4]; add reg, rBase; jmp reg`) switch tables and bounds checks across the real binary.
- [x] **R11 — Win64 Exception Handling for Recompiled Code (`recomp/src/eh.rs` + `runtime/guest/`)** — `DONE` (PRs #204, #214, #239, #246, closes #160) *(imports `__CxxFrameHandler3/4`, `_CxxThrowException`, `__C_specific_handler`, `RtlVirtualUnwind`, `longjmp`)*
  - Parse C++ EH `FuncInfo` (v3/v4) and `__C_specific_handler` scope tables into the CFG (funclets, catch/continuation targets); implement `RaiseException`/`RtlCaptureContext`/`RtlLookupFunctionEntry`/`RtlVirtualUnwind`/`RtlUnwindEx`/`RtlRestoreContext`/`setjmp`/`longjmp` over guest state and `.pdata`; define the lifter/runtime resume protocol.
- [x] **R12 — Multi-Module Recompile: Bundled DLLs & Guest-to-Guest Import Binding (`recomp/`)** — `DONE` (PRs #242, #253, #255, closes #161) *(no game files required for repo tests)*
  - Build the executable plus the redistributable DLLs shipped next to it (`MSVCP140`, `VCRUNTIME140`, `concrt140`, …) into one guest program: per-module image bases, exports, forwarders, imports bound to another module's exports, shared function registration tables.
- [x] **R13 — Parallel Sharded LLVM IR → Wasm64 Emscripten Build Orchestrator (`tools/build_guest_wasm.py`)** — `DONE` (PR #212, closes #193) *(no game files required — disjoint files `tools/build_guest_wasm.py`, `tests/test_build_guest_wasm.py`)*
  - Compiles sharded `playgta6-recomp build` (`R9`) `.ll`/`.c`/`.cpp` outputs across CPU cores via `emcc` with SHA-256 `.o` caching, links with `libplaygta6_core.a` into `web/dist/runtime.{js,wasm}`, and validates ABI exports.
- [x] **R14 — Guest Boot Sequence & Native First-Boot Harness (`runtime/guest/boot.{h,cpp}`, `tools/build_native_boot.py`)** — `DONE` (PRs #222 + #263, closes #198) *(no game files required for repo tests — disjoint files `runtime/guest/boot.{h,cpp}`, `runtime/guest/boot_main.cpp`, `tools/build_native_boot.py`, `tests/test_guest_boot.cpp`)*
  - Runtime boot sequence (load `PGIM` image, register lifted tables, bind import slots, minimal TEB/TLS + main thread, TLS callbacks, entry-point call, structured first-fault report) and a native host harness linking host-target lifted shards with the runtime for a first boot attempt without the browser toolchain (complements `R13`/`W40`).
- [x] **R15 — Recompilation Coverage & Boot-Path Bottleneck Profiler CLI (`tools/audit_recomp_build.py`)** — `DONE` (PR #230, closes #220) *(no game files required — disjoint files `tools/audit_recomp_build.py`, `tests/test_audit_recomp_build.py`)*
  - Analyzes `R9` `build-manifest.json` / `discover --json` outputs to rank top unlifted x86-64 mnemonics and compute entry-point + TLS-callback reachable boot-path coverage.
- [x] **R16 — Interface-Call Survey Subcommand: Vtable Slot Usage per COM Interface (`recomp/src/survey.rs`)** — `DONE` (PR #249, closes #238) *(no game files required — disjoint file `recomp/src/survey.rs`)*
  - `playgta6-recomp survey-interfaces`: for interface pointers obtained from given sources (an import's out-parameter, a parent interface's vtable slot), report the vtable slots called, call-site counts and inferred argument counts (addresses, GUID constants and counts only).
- [x] **R17 — Real-Binary Boot Bring-Up: Iterate the Native Boot to the First D3D12 Device Creation** — `DONE` (PRs #274–#295, closes #271) *(M2; game files stay local — repo changes are runtime/recompiler fixes with synthetic regression tests)*
  - Run the native boot harness (`R14`) on the real program build in a fix-and-rerun loop: triage each first fault, fix boot/loader/dispatcher/shim defects with regression tests or record the missing piece for its owning task, until the title reaches its first D3D12 device creation.
- [x] **R18 — Guest-Callable Thunks for Every Catalogued Shim Export: Dynamic `GetProcAddress` Coverage (`recomp/src/stubs/`, `runtime/guest/`)** — `DONE` (PR #289, closes #283) *(no game files required — observed at boot: `GetProcAddress` returns NULL for host shims that are not statically imported)*
  - Generate guest-callable marshaling thunks for every audited, bindable shim in the catalogue, register them with the guest runtime, and make dynamic `GetProcAddress` binding return callable guest addresses.
- [x] **R19 — Windowed Guest-Memory Translation: Fit a 16 GiB CPU Window into the wasm64 Guest Pool (prototype + measurements)** — `DONE` (PRs #300, #304, closes #285; ABI proposal open as #306) *(follow-up to O27: the title's CPU window cannot be identity-mapped into the 12 GiB browser pool; option off by default, no ABI change until measured)*
  - Prototype a second translation for one registered address range in the lifter's `guest_ptr` seam (per-64 KiB page table) with pool-backed window pages in the arena, and measure code-size and run-time cost.
- [x] **R20 — Real-Binary Boot Bring-Up, Phase 2: Native Boot from Device Creation to the First Presented Frame** — `DONE` (PRs #297–#338, closes #296) *(continues R17; local game files only for local runs)*
  - Iterate the native boot past `D3D12XboxCreateDevice` until the title presents its first frame, fixing each blocker by cause with a synthetic regression test.
- [ ] **R23 — Real-Binary Boot Bring-Up, Phase 3: Native Boot from the First Present to a Sustained Frame Loop** — `IN PROGRESS` (#341, @Timon-ok) *(continues R20; local game files only for local runs)*
  - Iterate the native boot past the first presented frames (audio HAL object, remaining command-list slots) to a sustained frame loop, and add a local frame capture of the recorded GPU command stream.

---

## Track 2: Xbox / Win32 OS Shims (`runtime/os/`, C++20 + Emscripten)
**Goal:** Reimplement console OS APIs backed by browser workers, pthreads, and HTTPFS.

- [x] **O1 — Emscripten Wasm64 + Pthreads Smoke Build** — `DONE` (PR #35, closes #23) *(no game files required)*
  - Wire `runtime/main.cpp` JS glue (`web/loader.js` imports for `host_log`, `host_wgpu_submit`, `host_httpfs_*`) and SAB HTTPFS mailbox consumer in `io_worker.js`.
- [x] **O2 — Blocking HTTPFS Bridge (`io_worker.js` ↔ `runtime/os/`)** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Implement synchronous worker-to-worker `Atomics.wait` range reads over a `SharedArrayBuffer` so `host_httpfs_open`/`read`/`close` work from pthreads.
- [x] **O3 — Virtual Memory & Threading/TLS Shims** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Implement Win32/Xbox `VirtualAlloc`/`VirtualProtect` arena mapping, `TlsAlloc`, SRW locks, and `XTaskQueue` on top of Emscripten pthreads.
- [x] **O4 — `GameInput` & `/userdata` Shims** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Read `playgta6::InputState` from the input `SharedArrayBuffer` and map save paths to IndexedDB `/userdata`.
- [x] **O5 — Xbox GDK `xgameruntime` & Win32 Kernel/File/Sync C-ABI Shims** — `DONE` (PR #36, closes #16) *(no game files required)*
  - Expose C-ABI Win32 & Xbox GDK imports (`XGameRuntimeInitialize`, `XTaskQueue*`, `XUser*` offline stub, `VirtualAlloc`/`Protect`/`Free`, `Tls*`, `SRWLock*`, `CriticalSection`/`Event`, `CreateFileW`/`ReadFile`/`SetFilePointerEx`/`GetFileSizeEx`/`CloseHandle`, `QueryPerformanceCounter`/`Frequency`) backed by our OS shims.
- [x] **O6 — Win32 Synchronization & Thread Pool Shims** — `DONE` (PR #80, closes #41) *(no game files required — disjoint files `runtime/os/win32_sync.{h,cpp}`, `tests/test_win32_sync.cpp`)*
  - Implement `CreateSemaphoreW`/`ReleaseSemaphore`, `CreateMutexW`/`ReleaseMutex`, `WaitForMultipleObjects` (`bWaitAll` true/false), `SleepConditionVariableSRW`/`WakeConditionVariable`/`WakeAllConditionVariable`, and `CreateThread`/`SetThreadAffinityMask`/`GetCurrentThreadId`.
- [x] **O7 — Win32 Heap, Module Registry & `GetProcAddress` Resolver** — `DONE` (PR #91, closes #42) *(no game files required — disjoint files `runtime/os/win32_loader.{h,cpp}`, `tests/test_win32_loader.cpp`)*
  - Implement `GetProcessHeap`/`HeapCreate`/`HeapAlloc`/`HeapReAlloc`/`HeapFree`/`HeapSize`, `GetModuleHandleW`/`LoadLibraryW`, and a dynamic `GetProcAddress` symbol registry mapping DLL + export names to registered runtime shims.
- [x] **O8 — Offline Networking, Telemetry & Xbox Live Safe Stubs** — `DONE` (PR #77, closes #43) *(no game files required — disjoint files `runtime/os/xbox_offline.{h,cpp}`, `tests/test_xbox_offline.cpp`)*
  - Implement clean offline-mode stubs for `ws2_32` (`WSAStartup`, `socket`, `getaddrinfo` → offline), `winhttp`, and Xbox Live (`XblInitialize`, `XblContextCreateHandle`, `XStore*`) so single-player boot never blocks on online services.
- [x] **O9 — Guest CPU State ABI & Recompiled-Code Dispatcher (`runtime/guest/`)** — `DONE` (PRs #138 + #158, closes #115) *(no game files required — small ABI PR first, then disjoint files `runtime/guest/guest_runtime.{h,cpp}`, `tests/test_guest_runtime.cpp`)*
  - Document the lifter's guest CPU state layout + runtime helper symbols in the shared ABI, then implement `__playgta6_indirect_call`/`__playgta6_trap`/`__playgta6_unimplemented`/`__playgta6_divide_error`, the guest-address → lifted-function table, IAT-slot → import-thunk dispatch, guest stack setup, and entry-point invocation.
- [x] **O10 — `PGIM` Guest Image Loader for `VirtualMemoryArena`** — `DONE` (PR #141, closes #116) *(no game files required — disjoint files `runtime/os/image_loader.{h,cpp}`, `tests/test_image_loader.cpp`)*
  - Load the `PGIM` v1 container from `playgta6-recomp pack-image` (R6): validate, reserve at the preferred base, commit sections, copy payloads, apply DIR64/HIGHLOW relocations when rebased, and set page protections.
- [x] **O11 — Runtime Shim Export Registration Table** — `DONE` (PR #133, closes #117) *(no game files required — disjoint files `runtime/os/shim_exports.{h,cpp}`, `tests/test_shim_exports.cpp`)*
  - Register every `pg6_win32_*` / `pg6_gdk_*` C-ABI shim with the O7 module registry under its Win32/GDK export name so guest `GetProcAddress` and the R5 import bridge can resolve them, with a header-sync test.
- [x] **O12 — Xbox Scarlett `xmem.dll` / `XCRDAPI.dll` / `xg_xs.dll` / `GameInput.dll` Shims (`runtime/os/xbox_scarlett.{h,cpp}`)** — `DONE` (PR #157, closes #150) *(driven by M2 `docs/imports_required.md`)*
  - Implement C-ABI shims for the 12 `xmem.dll` physical/virtual memory APIs, `XCRDAPI.dll!DStorageGetFactory` (DirectStorage backed by HTTPFS), `xg_xs.dll` (`XGCreateTextureComputer`, `XGSuggestSwizzleMode`), and `GameInput.dll` ordinal `#1`.
- [x] **O13 — UCRT (`api-ms-win-crt-*`) & `VCRUNTIME140` C-ABI Bridge (`runtime/os/ucrt_bridge.{h,cpp}`)** — `DONE` (PR #174, closes #151) *(driven by M2 `docs/imports_required.md`: 186 C-runtime imports)*
  - Expose thin Win64/C-ABI wrappers forwarding `api-ms-win-crt-{convert,filesystem,heap,locale,math,runtime,stdio,string,time,utility}` and `VCRUNTIME140` memory/string/exception exports onto Emscripten libc so `playgta6-recomp stubs` binds ~300 imports cleanly.
- [x] **O14 — Xbox Middleware Shims: `libHttpClient.GDK`, `Bink2Scarlett`, `AcpHal`/`xapu`/`chad`, FSR & `MFPlat` (`runtime/os/xbox_middleware.{h,cpp}`)** — `DONE` (PR #165, closes #152) *(driven by M2 `docs/imports_required.md`: 104 middleware imports)*
  - Implement clean offline/non-blocking C-ABI shims for `libHttpClient.GDK.dll` (38), `Bink2Scarlett.dll` (23), `AcpHal`/`xapu`/`chad` (19), `XboxBvhBuilder` (7), `amd_fidelityfx_loader_xbox` (4), and `MFPlat`/`MFReadWrite` (7).
- [x] **O15 — Remaining Win32 Core (`api-ms-win-core-*`, `ADVAPI32`, `bcrypt`, `ntuser`) & `MSVCP140` C++ Runtime Shims (`runtime/os/win32_compat.{h,cpp}`)** — `DONE` (PR #208, closes #163) *(driven by M2 `docs/imports_required.md`)*
  - Implement the remaining ~55 Win32 core file/process/string/time/registry/bcrypt/window shims + `MSVCP140.dll` mutex/xtime/locale/iostream helpers to hit 100% import resolution for `game_scarlett_master.exe`.
- [x] **O16 — Guest COM Object Bridge & `xgameruntime.dll` `QueryApiImpl` Provider (`runtime/guest/com_bridge.{h,cpp}`)** — `DONE` (PRs #209 + #235, closes #167) *(the executable has no static `xgameruntime.dll` imports: the GDK client code loads it and calls `QueryApiImpl`/`InitializeApiImpl`)*
  - Generic guest-memory COM objects (vtables of dispatchable addresses, `IUnknown` lifetime, argument marshaling) plus the `xgameruntime.dll` provider entry points routed onto the existing `pg6_gdk_*` shims; shared by `d3d12_xs` (G24), DirectStorage and GameInput interfaces.
- [x] **O17 — Unified Boot Import Coverage Wiring (`runtime/os/shim_exports.cpp` + `tools/check_import_coverage.py`)** — `DONE` (PRs #265, #266, #267, #268, #272, #273, #276, closes #257) *(O13/O14/O15 merged; G24 registration added when it lands)*
  - Wire `pg6_xbox_scarlett_register_all`, `pg6_ucrt_register_all`, `pg6_xbox_middleware_register_all`, `pg6_win32_compat_register_all`, and `pg6_d3d12_xs_register_all` into `shim_exports.cpp` and verify 100% coverage of all non-guest-DLL imports in `docs/imports_required.md`.
- [x] **O18 — Contiguous Guest Address Space: Offset-Mapped `VirtualMemoryArena` Backing for wasm64 (`runtime/os/vmem.{h,cpp}`)** — `DONE` (PRs #206, #232, #240, closes #187) *(lifted code reads guest memory directly in linear memory; the arena keeps pages in separate host allocations)*
  - Back the arena with one contiguous region and a single guest-base translation shared by the lifter's `guest_ptr` seam, the import-bridge seam, the guest runtime and the image loader, on host builds and under Emscripten `-sMEMORY64`, keeping commit/protect bookkeeping.
- [x] **O19 — `WS2_32.dll` 23 Ordinal Shims & Registry Bindings (`runtime/os/winsock_ordinals.{h,cpp}`)** — `DONE` (PR #203, closes #190) *(driven by M2 `docs/imports_required.md`: 23 ordinal-imported Winsock2 symbols)*
  - Implements and registers all 23 `WS2_32.dll` ordinals (`#1`..`#151`) + `getaddrinfo`/`freeaddrinfo` offline-safe shims in `Win32ModuleRegistry`.
- [x] **O20 — Complete Remaining 4 `WINHTTP.dll` Shims (`runtime/os/xbox_offline.cpp`)** — `DONE` (PR #208, closes #199) *(landed alongside `O15`: `WINHTTP.dll` is now 15/15 = 100%)*
  - Implements and registers the final 4 `WINHTTP.dll` exports in `Win32ModuleRegistry` for 15/15 (100%) WinHTTP coverage.
- [x] **O21 — Guest CPU Identification Helpers: `cpuid`/`rdtsc`/`rdtscp`/`xgetbv` (`runtime/guest/cpu_helpers.{h,cpp}`)** — `DONE` (PRs #210 + #221, closes #207) *(no game files required — small ABI addition first, then disjoint files `runtime/guest/cpu_helpers.{h,cpp}`, `tests/test_cpu_helpers.cpp`)*
  - Document and implement `__playgta6_cpuid`/`__playgta6_rdtsc`/`__playgta6_rdtscp`/`__playgta6_xgetbv` (emitted by the lifter since R7) as a consistent virtual CPU whose advertised features match what the lifter implements.
- [x] **O22 — Guest Callback Thunks for CRT Exit Registration & XAsync Provider / XTaskQueue Port Slots (`runtime/guest/`, `runtime/os/`)** — `DONE` (PRs #247, #250, #258, closes #236) *(no game files required — follow-up to O16: the `xgameruntime.dll` provider's remaining XAsync/XTaskQueue slots and CRT `atexit`-family callbacks)*
  - Guest-state thunks for `_crt_atexit`/`_register_onexit_function`/`_register_thread_local_exe_atexit_callback`, XAsync-taking shims calling guest completion callbacks, and the XAsync provider (`Begin`/`Schedule`/`Complete`/`Run`/`GetResult`/`Cancel`) + XTaskQueue port/composite slots.
- [x] **O23 — DirectStorage, GameInput & XG Guest Interface Adapters on the COM Bridge (`runtime/os/xbox_scarlett_guest.{h,cpp}`)** — `DONE` (PRs #252 + #259, closes #237) *(no game files required — disjoint files `runtime/os/xbox_scarlett_guest.{h,cpp}`, `tests/test_xbox_scarlett_guest.cpp`)*
  - Wrap the host objects behind `DStorageGetFactory`, `GameInputCreate` and `XGCreateTextureComputer` as guest COM-bridge interfaces with evidence-based slot layouts, logging stubs elsewhere, and the DirectStorage read path on the runtime file layer.
- [x] **O24 — Route XMem and Guest-Facing CRT Heap Allocations into the Guest Address Space (`runtime/os/`)** — `DONE` (PR #251, closes #243) *(no game files required — follow-up to O18: the runtime heap is fenced below 4 GiB on wasm64)*
  - Back the `xmem.dll` shims and the guest-facing CRT `malloc` family with `VirtualMemoryArena` / the Win32 heap layer so the title's allocations live in the guest pool, with one consistent allocator family and tests.
- [x] **O25 — Local wasm64 Runtime Test Runner (Docker + Node) & wasm64-Only Test Fixes (`tools/run_wasm_tests.py`)** — `DONE` (PR #260, closes #254) *(no game files required — disjoint files `tools/run_wasm_tests.py`, `tests/test_run_wasm_tests.py`; fixes in the affected tests/shims)*
  - Build every runtime test for Emscripten `-sMEMORY64 -pthread` in Docker, run it under a memory64-capable Node with a per-test report, and fix the wasm64-only failures (function-table identity in `GetProcAddress` checks, image-loader tests exceeding the wasm guest pool).
- [x] **O26 — Win32 Shims Required by the Recompiled Runtime DLLs: FLS, InitOnce, SList, Pointer Encoding & Locale (`runtime/os/win32_rtl.{h,cpp}`)** — `DONE` (PR #269, closes #264) *(driven by the first real program boot: `vcruntime140.dll` `DllMain` stops at `InitializeCriticalSectionEx`)*
  - Implement and bind `Fls*`, `InitializeCriticalSectionEx`, `InitOnceExecuteOnce`, `EncodePointer`/`DecodePointer`, SList, `SwitchToThread`, `GetModuleHandleExW`, `RtlCaptureStackBackTrace`, `LCMapStringEx`/`CompareStringEx`/`GetLocaleInfoEx`/`GetStringTypeW`/`GetCPInfo`, `FindFirstFileExW`, `GetFileInformationByHandle`, `GetTempPathW` and the other Win32 imports of `vcruntime140`/`msvcp140` that have no shim.
- [x] **O27 — XMem CPU/GPU Window Model: GPU Alias Registration & Address-Space Budget for the wasm64 Guest Pool (`runtime/os/xmem_windows.{h,cpp}`)** — `DONE` (PR #284, closes #278) *(driven by the real boot: multi-GiB `XMemVirtualAlloc` reservations after `D3D12ConfigureVirtualMemoryX`; the browser guest pool is 12 GiB)*
  - Model the title's CPU window and its GPU-visible alias (register the alias with the D3D12 layer, apply queue-side page mappings) and design how multi-GiB reservations fit the wasm64 guest pool, backed by measurements of what the title reserves and commits.

---

## Track 3: GPU Command Bridge (`runtime/gpu/` + `web/workers/wgpu_worker.js`, C++20 + JS)
**Goal:** Translate `d3d12_xs` command lists to binary packets executed on WebGPU.

- [x] **G1 — Expand Command Ring & Resource Tables** — `DONE` (PR #7, closes #4) *(no game files required)*
  - Implement `CreateBuffer`, `WriteBuffer`, `CreateShaderModule`, `CreatePipeline`, and `Draw` packets across `runtime/abi/playgta6_abi.h`, `runtime/gpu/cmd_encoder.cpp`, and `web/workers/wgpu_worker.js`.
- [x] **G2 — SharedArrayBuffer Zero-Copy GPU Ring** — `DONE` (PR #22, closes #10) *(no game files required)*
  - Replace `postMessage` buffer copies with a lock-free SPSC `SharedArrayBuffer` ring buffer between Wasm64 and `wgpu_worker.js`.
- [x] **G3 — `d3d12_xs` Interface Stubs & Trace Harness** — `DONE` (PR #7, closes #4) *(no game files required)*
  - Implement minimal COM-like `ID3D12Device` / `ID3D12GraphicsCommandList` / `PresentX` shim layer that emits packets matching `docs/gpu_commands.md`.
- [x] **G4 — `WGPK` `.wgsl.pack` Binary Loader & Shader Module Cache** — `DONE` (PR #26, closes #18) *(no game files required)*
  - Parse `.wgsl.pack` binary bundles (`shaders/src/pack.rs`) in JS (`web/workers/wgsl_pack_loader.js`), resolve WGSL by 64-bit FNV-1a hash, and cache compiled `GPUShaderModule`s.
- [x] **G5 — D3D12-XS Textures, Samplers, BindGroups, Indexed Draw & Compute Dispatch** — `DONE` (PR #55, closes #24) *(no game files required)*
  - Extend GPU command ABI, `CommandEncoder`, `d3d12_xs.h`, and `wgpu_worker.js` with textures, samplers, bind groups, indexed drawing, and compute dispatch (for RDNA2 mesh-compute lowering).
- [x] **G6 — D3D12-XS Descriptor Heaps, Root Signatures & Resource State Tracker** — `DONE` (PR #110, closes #44) *(no game files required — disjoint files `runtime/gpu/descriptor_heap.{h,cpp}`, `tests/test_descriptor_heap.cpp`)*
  - Implement CPU/GPU descriptor heap tables (`CBV_SRV_UAV`, `SAMPLER`, `RTV`, `DSV`), root signature parameter mapping, and dirty-bind-group caching that compiles D3D12 descriptor tables into WebGPU `CreateBindGroup`/`SetBindGroup` calls.
- [x] **G7 — D3D12 PSO Cache, Render Targets & Viewport/Scissor State** — `DONE` (PR #109, closes #45) *(no game files required — disjoint files `runtime/gpu/pso_cache.{h,cpp}`, `tests/test_pso_cache.cpp`)*
  - Hash D3D12 graphics & compute PSO descriptors (blend/depth/rasterizer/vertex-layout/shader hashes) to deduplicate WebGPU pipeline creation and track active RTV/DSV/viewport/scissor state per command list.
- [x] **G8 — Binary GPU Command Stream Disassembler & Trace Inspector** — `DONE` (PR #49, closes #32) *(no game files required — disjoint files `tools/gpu_trace.py`, `tests/test_gpu_trace.py`)*
  - Parse recorded `playgta6` GPU command ring byte streams into human-readable/JSON frame traces with opcode histograms, byte counts, and malformed-packet diagnostics for debugging M3 frames.
- [x] **G9 — WebGPU `OffscreenCanvas` Swapchain, Depth-Stencil & Resize Manager** — `DONE` (PR #59, closes #54) *(no game files required — disjoint files `web/workers/wgpu_swapchain.js`, `tests/test_wgpu_swapchain.mjs`)*
  - Manage `OffscreenCanvas` WebGPU context config, DPR-aware dynamic resize, `depth24plus`/`depth32float` depth-stencil attachment lifecycle, `device.lost` recovery, and frame-pacing telemetry.
- [x] **G10 — BCn / Texture Format & Mip-Chain Upload Helpers** — `DONE` (PR #66, closes #57) *(no game files required — disjoint files `web/workers/texture_formats.js`, `tests/test_texture_formats.mjs`)*
  - Map DXGI/Xbox texture formats (`BC1`–`BC7`, `RGBA8`, `RGBA16F`, depth/stencil) to WebGPU `GPUTextureFormat`, compute 256-byte-aligned `bytesPerRow` + mip dimensions, and provide CPU fallback BC1/BC3 decompression when WebGPU `texture-compression-bc` is absent.
- [x] **G11 — WebGPU Pipeline & BindGroupLayout Cache Manager (`web/workers/pipeline_cache.js`)** — `DONE` (PR #75, closes #62) *(no game files required — disjoint files `web/workers/pipeline_cache.js`, `tests/test_pipeline_cache.mjs`)*
  - Deduplicate `GPURenderPipeline` and `GPUComputePipeline` creation in JS by descriptor hash, support async `createRenderPipelineAsync` warm-up, and consume shader reflection metadata (`S6`).
- [x] **G12 — Update `tools/gpu_trace.py` for G5 Opcodes `12..21`** — `DONE` (PR #81, closes #67) *(no game files required — disjoint files `tools/gpu_trace.py`, `tests/test_gpu_trace.py`)*
  - Add full typed payload decoding for `CreateTexture`..`DispatchCompute` (`12..21`), compute pass summaries, and texture/bind-group histograms matching the merged `G5` ABI.
- [x] **G13 — WebGPU Shader/Pipeline Diagnostics & Capture Replay (`web/workers/gpu_recorder.js`)** — `DONE` (PR #86, closes #72) *(no game files required — disjoint files `web/workers/gpu_recorder.js`, `tests/test_gpu_recorder.mjs`)*
  - Ring-buffered in-browser frame recorder that captures raw binary GPU command streams (`BeginFrame`..`Present`) and exports downloadable `.pg6trace` blobs compatible with `tools/gpu_trace.py`.
- [x] **G14 — WebGPU Staging Buffer Pool & Upload Ring (`web/workers/staging_pool.js`)** — `DONE` (PR #98, closes #82) *(no game files required — disjoint files `web/workers/staging_pool.js`, `tests/test_staging_pool.mjs`)*
  - Sub-allocating ring of `MAP_WRITE | COPY_SRC` / `writeBuffer` staging chunks with fence-driven recycling to eliminate per-frame buffer allocation churn in `wgpu_worker.js`.
- [x] **G15 — Live WebGPU Smoke Demo Runner Wiring (`web/demo/run_smoke_demo.js`)** — `DONE` (PR #99, closes #88) *(no game files required — disjoint files `web/demo/run_smoke_demo.js`, `tests/test_run_smoke_demo.mjs`)*
  - Wires `SmokeSceneController` (`W16`) + `GpuRingBuffer` + `WgpuSwapchain` (`G9`) + `WgpuCommandProcessor` + `GpuFrameRecorder` (`G13`) + `PerfMonitor` (`W15`) into a turnkey browser `requestAnimationFrame` driver for live WebGPU testing.
- [x] **G16 — WebGPU Timestamp Query & Pass GPU-Time Profiler (`web/workers/gpu_timer.js`)** — `DONE` (PR #104, closes #94) *(no game files required — disjoint files `web/workers/gpu_timer.js`, `tests/test_gpu_timer.mjs`)*
  - Wraps optional WebGPU `'timestamp-query'` `GPUQuerySet` resolution (gracefully degrading when absent) to measure per-pass render/compute GPU timings in milliseconds.
- [x] **G17 — S6 Reflection Manifest Loader & Auto-BindGroup Builder (`web/workers/reflection_loader.js`)** — `DONE` (PR #126, closes #101) *(no game files required — disjoint files `web/workers/reflection_loader.js`, `tests/test_reflection_loader.mjs`)*
  - Parses `S6` (`shaders/src/reflect.rs`) reflection JSON manifests in JS and builds exact `GPUBindGroupLayout` + `GPUPipelineLayout` descriptors via `WebGpuPipelineCache` (`G11`).
- [x] **G18 — WebGPU Dynamic Uniform Ring & Draw-Call Constant Batcher (`web/workers/uniform_ring.js`)** — `DONE` (PR #124, closes #106) *(no game files required — disjoint files `web/workers/uniform_ring.js`, `tests/test_uniform_ring.mjs`)*
  - 256-byte-aligned dynamic uniform buffer ring (`hasDynamicOffset: true`) for per-draw constant uploads (D3D12 root constants / CBVs) with single-upload per-frame flush.
- [x] **G19 — WebGPU Indirect Draw & Compute-Culling Buffer Helper (`web/workers/indirect_draw.js`)** — `DONE` (PR #134, closes #112) *(no game files required — disjoint files `web/workers/indirect_draw.js`, `tests/test_indirect_draw.mjs`)*
  - Encodes and validates WebGPU `drawIndirect` / `drawIndexedIndirect` / `dispatchWorkgroupsIndirect` command buffers for compute-driven RDNA2 meshlet culling (`S5`).
- [x] **G20 — Render Pipeline & Render Pass Packet ABI v2 + D3D12 State Bridge (`runtime/gpu/state_bridge`)** — `DONE` (PRs #148 + #183, closes #121) *(no game files required — small ABI PR first, then disjoint files `runtime/gpu/state_bridge.{h,cpp}`, `tests/test_state_bridge.cpp`)*
  - Add packets for full pipeline keys, render pass begin/end with load/store ops, and viewport/scissor/blend-constant/stencil-ref, then wire the G6 descriptor/root-signature tracker and G7 PSO cache/state tracker into packet emission.
- [x] **G21 — WebGPU Resource Registry & Frame-Deferred Garbage Collector (`web/workers/resource_registry.js`)** — `DONE` (PR #145, closes #127) *(no game files required — disjoint files `web/workers/resource_registry.js`, `tests/test_resource_registry.mjs`)*
  - Tracks GPU buffer/texture/view/sampler/bind-group/pipeline handles with VRAM byte accounting and frame-epoch deferred destruction after `onSubmittedWorkDone` fences.
- [x] **G22 — WebGPU Transient Render-Target Pool & Attachment Aliasing (`web/workers/rt_pool.js`)** — `DONE` (PR #144, closes #128) *(no game files required — disjoint files `web/workers/rt_pool.js`, `tests/test_rt_pool.mjs`)*
  - Pools transient color/depth/shadow/post-FX `GPUTexture` + view attachments by descriptor key with per-frame reuse and idle/budget LRU eviction.
- [x] **G23 — Wire `GpuStagingPool` (`G14`), `DynamicUniformRing` (`G18`), `GpuPassTimer` (`G16`) & `ReflectionLoader` (`G17`) into `web/workers/wgpu_worker.js`** — `DONE` (PR #180, closes #172) *(no game files required — disjoint files `web/workers/wgpu_worker.js`, `tests/test_wgpu_worker_integration.mjs`)*
  - Integrate pooled staging uploads, dynamic uniform batching, optional timestamp profiling, reflected bind-group layouts, resource registry, and render-target pool into the main WebGPU command processor.
- [x] **G24 — `d3d12_xs.dll` C-ABI Entrypoints & COM Vtable Bridge (`runtime/gpu/d3d12_xs_exports.{h,cpp}`)** — `DONE` (PRs #244, #256, #261, closes #168) *(driven by M2 `docs/imports_required.md`: `D3D12XboxCreateDevice`, `D3D12ConfigureVirtualMemoryX`, `D3D12SerializeVersionedRootSignature`)*
  - Expose the exact 3 `d3d12_xs.dll` exports with Win64-ABI COM vtables (`ID3D12Device`, `ID3D12CommandQueue`, `ID3D12GraphicsCommandList`, `ID3D12Resource`, `ID3D12DescriptorHeap`, `ID3D12PipelineState`) bridged to `G6`/`G7`/`G20`.
- [x] **G25 — Upgrade `tools/gpu_trace.py` Disassembler for GPU Packet ABI v2 Opcodes `22..34`** — `DONE` (PR #227, closes #191) *(disjoint files `tools/gpu_trace.py`, `tests/test_gpu_trace.py`)*
  - Full typed payload decoding, summaries, and validation for all ABI v2 render-pass, pipeline-key, layout, and dynamic-state packets (`22..34`).
- [x] **G26 — Worker Splash & Bink2 Frame Presentation Bridge (`web/workers/splash_frame_bridge.js`)** — `DONE` (PR #228, closes #218) *(no game files required — disjoint files `web/workers/splash_frame_bridge.js`, `tests/test_splash_frame_bridge.mjs`)*
  - Bridges raw RGBA8/YUV420p/ImageBitmap splash & Bink2 frames into `BinkVideoPresenter` (`W38`) and emits `FIRST_PRESENT` telemetry.
- [x] **G27 — Xbox-Specific D3D12 Device & Queue Slots on the Startup Path (`runtime/gpu/d3d12_xs_xbox.{h,cpp}`)** — `DONE` (PRs #270 + #277, closes #262) *(no game files required for repo tests — follow-up to G24)*
  - Establish from call-site evidence and implement the Xbox-only device/queue/command-list methods the title's device initialisation and submission thread call (device slots 49, 61/62, 70, 78–80, 101; queue 23/24/31; list 131/132), and contribute the survey improvements used.

---

## Track 4: Shader Pipeline (`shaders/`, Rust)
**Goal:** Convert Xbox RDNA2 shaders (`--mesh-compute`, `--wave64-emulate`) → SPIR-V → WGSL packs.

- [x] **S1 — `naga` SPIR-V → WGSL Batch Converter** — `DONE` (PR #9, closes #5) *(testable with synthetic SPIR-V; real corpus needs local SPIR-V)*
  - Add `naga` to `shaders/Cargo.toml`, convert a directory of `.spv` files to `.wgsl`, emit `shader-index.json`, and output a structured failure-reason report (unsupported capabilities/extensions).
- [x] **S2 — Shader Pack Bundler & WebGPU Pre-Validator** — `DONE` (PR #9, closes #5) *(no game files required)*
  - Pack lowered `.wgsl` files into compact `.wgsl.pack` bundles (like `playgta5`) for fast loading in `wgpu_worker.js`.
- [x] **S3 — WebGPU Compatibility Passes** — `DONE` (PR #25, closes #11) *(needs SPIR-V failure report from S1)*
  - Implement SPIR-V/WGSL rewrite passes for unsupported WebGPU features (bindless indexing limits, subgroup/wave64 fallbacks, mesh-compute lowering).
- [x] **S4 — DXBC/DXIL Container Inspector & Naga IR Push-Constant/BindGroup Lowering** — `DONE` (PR #56, closes #27) *(no game files required)*
  - Add pure-Rust DXBC/DXIL container parser (`shaders/src/dxbc.rs`) and Naga IR lowering passes (`shaders/src/lower.rs`: push constants → uniform buffer, >4 bind-group remapping) + `inspect-dxbc` CLI.
- [x] **S5 — Wave64/Subgroup Emulation & Mesh-Compute WGSL Lowering Library** — `DONE` (PR #137, closes #46) *(no game files required — wait for S4 or keep disjoint in `shaders/src/wave_emul.rs`)*
  - Provide reusable WGSL workgroup-shared-memory emulation helpers for RDNA2 Wave64 ballot/shuffle/prefix-sum ops and indirect compute-driven meshlet culling/draw emission when WebGPU `subgroups` is unavailable.
- [x] **S6 — Root-Signature-to-WebGPU `BindGroupLayout` Reflection & Cache Manifest** — `DONE` (PR #92, closes #47) *(no game files required — disjoint file `shaders/src/reflect.rs`)*
  - Reflect parsed `naga::Module` resource bindings, vertex inputs, and workgroup sizes into compact JSON metadata alongside `shader-index.json` so `wgpu_worker.js` can build exact `GPUBindGroupLayout`s without runtime string parsing.
- [x] **S7 — Offline `WGPK` Shader Pack Inspector, Diff & WebGPU Limit Budget CLI (`tools/inspect_wgsl_pack.py`)** — `DONE` (PR #146, closes #129) *(no game files required — disjoint files `tools/inspect_wgsl_pack.py`, `tests/test_inspect_wgsl_pack.py`)*
  - Inspects, diffs, and audits compiled `.wgsl.pack` bundles against WebGPU group/binding/workgroup limits and feature directives (`f16`, `subgroups`).
- [x] **S8 — Shader Container Format Scanner & DXBC/DXIL/RDNA2 Blob Classifier (`tools/scan_shaders.py`)** — `DONE` (PR #189, closes #181) *(no game files required — disjoint files `tools/scan_shaders.py`, `tests/test_scan_shaders.py`)*
  - Read-only local scanner classifying shader containers (`DXBC`, `DXIL`, `SPIR-V`, or Xbox Scarlett RDNA2 ISA signatures) in a local directory into a sanitized JSON summary for M2/M4 shader planning.
- [x] **S9 — Fallback `WGPK` Shader Pack & S6 Reflection Manifest Generator (`tools/gen_fallback_shaders.py`)** — `DONE` (PR #211, closes #192) *(no game files required — disjoint files `tools/gen_fallback_shaders.py`, `tests/test_gen_fallback_shaders.py`)*
  - Generates a verified `WGPK` v1 bundle + S6 `reflection-manifest.json` with splash/UI blit, 3D debug mesh, and compute fallback WGSL shaders into gitignored `local-data/shaders/`.
- [x] **S10 — Local DXIL/DXBC Corpus Extractor & `RTS0`/`PSV0` Analyzer CLI (`tools/extract_dxil_corpus.py`)** — `DONE` (PR #225, closes #200) *(driven by `S8` finding 81 real `DXIL_SM6` containers in local DLLs)*
  - Extracts and structurally analyzes local DXIL/DXBC containers (`DXIL`, `PSV0`, `RTS0`, `ISG1`, `OSG1`, `ACCC`) into gitignored `work/shaders/dxil/` with sanitized manifest output.
- [x] **S11 — Offline DXIL → SPIR-V → WGSL Conversion Harness & Coverage CLI (`tools/convert_dxil_corpus.py`)** — `DONE` (PR #231, closes #219) *(no game files required — disjoint files `tools/convert_dxil_corpus.py`, `tests/test_convert_dxil_corpus.py`)*
  - Drives `dxil-spirv` + `playgta6-shaders` in parallel over an extracted DXIL corpus (`S10`) with SHA-256 caching and structured failure classification.

---

## Track 5: Browser Shell, Streaming I/O & Audio (`web/`, `server/`, `tools/`, JS + Python + Rust)
**Goal:** Fast, resilient browser loading, streaming, caching, and audio playback.

- [x] **W1 — OPFS + LRU Range Cache in `io_worker.js`** — `DONE` (PR #15, closes #2) *(no game files required)*
  - Cache fetched HTTP `206` byte chunks in Origin Private File System (OPFS) with a configurable budget + `bootset.json` prefetch on startup.
- [x] **W2 — IndexedDB `/userdata` Save Persistence** — `DONE` (PR #15, closes #2) *(no game files required)*
  - Restore `/userdata/` files into Wasm FS before `main()` and flush writes asynchronously (mirroring `playgta5`'s `loader.js` pattern).
- [x] **W3 — `audio_worker.js` Opus / WebAudio Pipeline** — `DONE` (PR #8, closes #6) *(no game files required)*
  - Implement `AudioWorklet` ring buffer + Ogg Opus stream decoding.
- [x] **W4 — Local Environment & Diagnostics (`tools/`)** — `DONE` (PR #8, closes #6) *(uses `GTA6_GAME_DIR` when available)*
  - Extend `tools/` to generate `bootset.json`, inspect RPF9 archive headers without extracting 100+ GB, and verify local file integrity.
- [x] **W5 — Full `InputBridge` (Gamepad API + Xbox Buttons/Sticks/Triggers + Pointer Lock)** — `DONE` (PR #28, closes #17) *(no game files required)*
  - Populate the full 40-byte `playgta6::InputState` SAB from Pointer Lock mouse, keyboard, and `navigator.getGamepads()` matching `runtime/os/game_input.h`.
- [x] **W6 — Partial-Download Game Folder Scanner & Incremental Manifest CLI (`tools/`)** — `DONE` (PR #30, closes #19) *(works on partial or complete `GTA6_GAME_DIR`)*
  - Add read-only `scan` command detecting completed vs downloading files (`.!qB`/`.part`/sparse) and `--skip-incomplete` for `manifest`/`bootset`.
- [x] **W7 — Local Server Hardening (`/api/status`, ETag/If-Range) & Browser Diagnostics HUD (`server/`, `web/index.html`)** — `DONE` (PR #29, closes #20) *(no game files required)*
  - Add ETag/`If-Range`/`HEAD` + sanitized `/api/status` to `server/serve_local.py` and upgrade `web/index.html` with a live Wasm64/WebGPU/SAB HUD.
- [x] **W8 — Multi-Voice Audio Mixer & PCM/ADPCM Voice Streamer** — `DONE` (PR #51, closes #31) *(no game files required — disjoint files `web/workers/audio_mixer.js`, `tests/test_audio_mixer.mjs`)*
  - Implement a lock-free multi-voice mixer (3D pan/gain, sample-rate resampling, ADPCM/PCM decoding, soft-clip limiter) that feeds `audio_worker.js`'s SAB ring buffer for XAudio2/XMA2 voice emulation.
- [x] **W9 — Headless End-to-End Integration Harness (Local Server ↔ Workers ↔ SAB ↔ WebGPU Mock)** — `DONE` (PR #48, closes #33) *(no game files required — disjoint file `tests/test_e2e_pipeline.mjs`)*
  - Spin up `server/serve_local.py` on an ephemeral port in a Node test, exercise real HTTP `206` Range + `ETag`/`If-Range` reads through `RangeCache` → `WgslPackCache` → `InputBridge` → `GpuRingBuffer` → `WgpuCommandProcessor` end-to-end.
- [x] **W10 — Pure-Rust Xbox XVD/XVC Container Header & Layout Inspector** — `DONE` (PR #50, closes #34) *(works on local `.xvd`/`.xvc` or synthetic unit tests — disjoint files `tools/src/xvd.rs`, `tools/src/main.rs`)*
  - Read-only parser for `msft-xvd` container headers (version, content type, drive/embedded XVD sizes, XVC region presence, encryption flags) wired as `playgta6-manifest xvd <path> [--json]`.
- [x] **W11 — Streaming Browser RPF9 TOC & Virtual File System Reader** — `DONE` (PR #61, closes #52) *(no game files required — disjoint files `web/workers/rpf9_vfs.js`, `tests/test_rpf9_vfs.mjs`)*
  - Fetch only RPF header + TOC + requested sub-file byte ranges via `RangeCache`, parse directory/file entries & names table, resolve virtual paths, and stream sub-file chunks on demand.
- [x] **W12 — `AudioWorklet` SAB Ring Processor & Main-Thread WebAudio Engine** — `DONE` (PR #60, closes #53) *(no game files required — disjoint files `web/workers/audio_worklet_processor.js`, `web/audio/audio_engine.js`, `tests/test_audio_engine.mjs`)*
  - Zero-GC `AudioWorkletProcessor` pulling 128-frame stereo blocks from `AudioRingBuffer` SAB + main-thread `AudioEngine` (`AudioContext` bootstrap, user-gesture unlock, volume/mute, underrun telemetry).
- [x] **W13 — Local Preflight & Milestone Readiness Checker CLI** — `DONE` (PR #65, closes #58) *(no game files required — disjoint files `tools/preflight.py`, `tests/test_preflight.py`)*
  - Single-command `python tools/preflight.py [--json]` verifying toolchains, `.env` / `GTA6_GAME_DIR`, COOP/COEP server sanity, `web/dist/runtime.wasm`, `data-manifest.json`, `bootset.json`, and `.wgsl.pack` readiness.
- [x] **W14 — OPFS Cache Quota Inspector & Browser Storage Manager** — `DONE` (PR #70, closes #63) *(no game files required — disjoint files `web/storage/storage_manager.js`, `tests/test_storage_manager.mjs`)*
  - Query `navigator.storage.estimate()`, request persistent storage, inspect/clear OPFS range cache and IndexedDB `/userdata` slots cleanly from the browser HUD.
- [x] **W15 — Frame Telemetry Ring & Performance Profiler (`web/telemetry/perf_monitor.js`)** — `DONE` (PR #71, closes #64) *(no game files required — disjoint files `web/telemetry/perf_monitor.js`, `tests/test_perf_monitor.mjs`)*
  - Lock-free rolling frame-time, p95/p99 latency, HTTP Range throughput, GPU packet rate, and audio underrun aggregator feeding `window.playgta6Hud.updateMetrics()`.
- [x] **W16 — Browser WebGPU Smoke Demo & Built-in Intro Splash Test Scene (`web/demo/smoke_scene.js`)** — `DONE` (PR #79, closes #68) *(no game files required — disjoint files `web/demo/smoke_scene.js`, `tests/test_smoke_scene.mjs`)*
  - Self-contained binary command stream generator exercising `G5` (`CreateTexture`, `CreateSampler`, `CreateBindGroup`, `SetIndexBuffer`, `DrawIndexed`, `DispatchCompute`) + `WgpuSwapchain` so opening `http://127.0.0.1:8080` renders a live textured 3D/compute frame in Chrome/Edge even before `runtime.wasm` is compiled.
- [x] **W17 — Emscripten Wasm64 Build Script for Linux/macOS (`runtime/build_wasm.sh`) & CI Artifact Check** — `DONE` (PR #76, closes #69) *(no game files required — disjoint files `runtime/build_wasm.sh`, `tests/test_build_scripts.py`)*
  - Cross-platform POSIX shell counterpart to `runtime/build_wasm.ps1` (`--check-only`, `-sMEMORY64=1 -pthread`, `web/dist/runtime.{js,wasm}` output).
- [x] **W18 — Streaming HTTPFS Priority Scheduler & Prefetch Queue (`web/workers/io_scheduler.js`)** — `DONE` (PR #87, closes #73) *(no game files required — disjoint files `web/workers/io_scheduler.js`, `tests/test_io_scheduler.mjs`)*
  - Priority queue (critical boot/sync reads > streaming world cells > background prefetch) with cancellation, retry backoff, and concurrency throttling in front of `RangeCache`.
- [x] **W19 — Binary Wasm64 Memory64/Threads & ABI Import Validator (`tools/check_wasm_exports.py`)** — `DONE` (PR #90, closes #74) *(no game files required — disjoint files `tools/check_wasm_exports.py`, `tests/test_wasm_exports.py`)*
  - Binary `.wasm` section/import/export validator verifying a compiled `runtime.wasm` declares `memory64` (`i64` memory), shared memory (`threads`), and exact `env.host_*` ABI imports from `docs/wasm_abi.md`.
- [x] **W20 — Gamepad Remapper & Input Profile Persistence (`web/input/input_profiles.js`)** — `DONE` (PR #96, closes #83) *(no game files required — disjoint files `web/input/input_profiles.js`, `tests/test_input_profiles.mjs`)*
  - Configurable keyboard/mouse/gamepad presets, sensitivity/deadzone curves, and localStorage/IndexedDB profile persistence feeding `InputBridge`.
- [x] **W21 — Structured Crash Reporter & Runtime Diagnostic Bundle Exporter (`web/telemetry/crash_report.js`)** — `DONE` (PR #97, closes #84) *(no game files required — disjoint files `web/telemetry/crash_report.js`, `tests/test_crash_report.mjs`)*
  - Captures Wasm traps, WebGPU validation/device-lost errors, recent runtime logs, and browser capability fingerprints into sanitized shareable JSON reports (never including game data or host paths).
- [x] **W22 — Browser Emscripten Runtime Session Controller (`web/runtime/session_controller.js`)** — `DONE` (PR #100, closes #89) *(no game files required — disjoint files `web/runtime/session_controller.js`, `tests/test_session_controller.mjs`)*
  - Orchestrates worker startup (`io_worker`, `wgpu_worker`, `audio_engine`), `InputBridge`, `PerfMonitor`, `BrowserStorageManager`, and clean restart/fallback into `SmokeSceneController` when `dist/runtime.wasm` is absent.
- [x] **W23 — Local Asset Pipeline Orchestrator CLI (`tools/prepare_local.py`)** — `DONE` (PR #105, closes #95) *(no game files required — disjoint files `tools/prepare_local.py`, `tests/test_prepare_local.py`)*
  - Single-command dry-run/execute helper that runs `scan` → `xvd`/`rpf9`/`manifest`/`bootset` → `recomp inspect` → `shaders convert` against `GTA6_GAME_DIR` into gitignored `local-data/` & `work/`.
- [x] **W24 — Browser HUD Action Controls & Interactive Diagnostics Wiring (`web/ui/hud_controller.js`)** — `DONE` (PR #122, closes #102) *(no game files required — disjoint files `web/ui/hud_controller.js`, `tests/test_hud_controller.mjs`)*
  - Binds HUD buttons (Clear OPFS Cache, Export Saves, Capture GPU Trace, Export Crash Report, Switch Input Profile, Toggle Smoke Demo) to `BrowserStorageManager`, `GpuFrameRecorder`, `CrashReporter`, and `SessionController`.
- [x] **W25 — LAN / Private-Host CORS & Range Streaming Benchmark CLI (`tools/bench_range_server.py`)** — `DONE` (PR #111, closes #103) *(no game files required — disjoint files `tools/bench_range_server.py`, `tests/test_bench_range_server.py`)*
  - Measures concurrent HTTP `206` Range throughput, latency percentiles, and `If-None-Match` (`304`) overhead against `server/serve_local.py` on synthetic payloads.
- [x] **W26 — Browser Diagnostics & Smoke Demo Bootstrap Glue (`web/app_bootstrap.js`)** — `DONE` (PR #125, closes #107) *(no game files required — disjoint files `web/app_bootstrap.js`, `tests/test_app_bootstrap.mjs`)*
  - Clean ES module entrypoint connecting `RuntimeSessionController`, `SmokeDemoRunner`, `InputProfileManager`, `AudioEngine`, `BrowserStorageManager`, and `CrashReporter` for `web/index.html`.
- [x] **W27 — Multi-Track Integration Check & CI Coverage Summary Script (`tools/repo_health.py`)** — `DONE` (PR #123, closes #108) *(no game files required — disjoint files `tools/repo_health.py`, `tests/test_repo_health.py`)*
  - Verifies zero ABI drift between `runtime/abi/playgta6_abi.h`, `tools/gpu_trace.py`, `web/workers/wgpu_worker.js`, `web/input/input_bridge.js`, and `docs/WORKSTREAMS.md` in one fast stdlib check.
- [x] **W28 — Streaming Audio Bank / AWC Header Inspector (`tools/src/awc.rs`)** — `DONE` (PR #136, closes #113) *(no game files required — disjoint file `tools/src/awc.rs`)*
  - Read-only RAGE `.awc` audio bank container header parser (stream/chunk counts, codec tag PCM/IMA-ADPCM/XMA2/Opus, sample rate, channel count) without extracting or committing audio data.
- [x] **W29 — Web Worker Pool & Offscreen Task Dispatcher (`web/workers/worker_pool.js`)** — `DONE` (PR #135, closes #114) *(no game files required — disjoint files `web/workers/worker_pool.js`, `tests/test_worker_pool.mjs`)*
  - Reusable bounded Web Worker pool supporting transferable `ArrayBuffer` jobs, cancellation, and graceful crash recovery for background decompression & shader pre-parsing.
- [x] **W30 — Wire `web/index.html` to `web/app_bootstrap.js` & `web/ui/hud_controller.js` (`web/index.html`)** — `DONE` (PR #143, closes #130) *(no game files required — disjoint files `web/index.html`, `tests/test_index_html_wiring.mjs`)*
  - Adds compact interactive HUD action controls and replaces the legacy hardcoded worker bootstrap with `bootstrapPlayGta6App` + `HudController` so opening `http://127.0.0.1:8080` renders live WebGPU smoke frames or boots `runtime.wasm`.
- [x] **W31 — Browser Streaming Manifest & Chunk Integrity Verifier (`web/storage/manifest_verifier.js`)** — `DONE` (PR #142, closes #131) *(no game files required — disjoint files `web/storage/manifest_verifier.js`, `tests/test_manifest_verifier.mjs`)*
  - Validates `data-manifest.json` and `bootset.json` schemas, computes partial-download streaming readiness percentages, and probes live HTTP `206` Range / `ETag` consistency.
- [x] **W32 — Wire `IoPriorityScheduler` (`W18`) & `Rpf9Vfs` (`W11`) into `web/workers/io_worker.js`** — `DONE` (PR #166, closes #153) *(no game files required — disjoint files `web/workers/io_worker.js`, `tests/test_io_worker_integration.mjs`)*
  - Connect prioritized Range scheduling and on-demand RPF9 virtual sub-file streaming into the `HttpfsMailbox` worker loop.
- [x] **W33 — Headless Live Boot & HUD E2E Verification Harness (`tests/test_live_boot_e2e.mjs`)** — `DONE` (PR #155, closes #154) *(no game files required — disjoint file `tests/test_live_boot_e2e.mjs`)*
  - End-to-end test spinning up `server/serve_local.py`, fetching `/`, `/api/status`, and JS modules with `COOP`/`COEP` headers, and running `bootstrapPlayGta6App` + `HudController` through smoke-demo frames, trace export, and diagnostic bundle generation.
- [x] **W34 — Real RPF9 Little-Endian `0x52504639` (`"9FPR"`) Magic Support (`tools/src/rpf9.rs`, `web/workers/rpf9_vfs.js`)** — `DONE` (PR #147, closes #140) *(driven by M2 inspection of real Xbox `.rpf` archives)*
  - Accept little-endian `0x52504639` (`39 46 50 52` = `"9FPR"`) alongside legacy big-endian `"RPF9"` headers in both the Rust CLI inspector and browser streaming RPF9 VFS reader.
- [x] **W35 — Real RPF9 Header v9 Field Layout & Encryption-Flag Diagnostic (`tools/src/rpf9.rs`)** — `DONE` (PR #179, closes #171) *(driven by M2 inspection of real `appdata0.rpf`/`common.rpf`: word 4 is a 32-bit hash/flag rather than `names_length`)*
  - Parse the real Xbox Series RPF9 header words cleanly without rejecting valid archives as truncated, and report header layout/entry/encryption metadata cleanly.
- [x] **W36 — Wasm64 Memory64 Runtime & BigInt64 Host Import Integration Tests (`tests/test_wasm64_runtime.mjs`)** — `DONE` (PR #177, closes #173) *(no game files required — disjoint file `tests/test_wasm64_runtime.mjs`)*
  - Verifies synthetic Wasm64 `i64` memory marshaling with `web/loader.js` `createHostImports` and validates local `web/dist/runtime.wasm` ABI exports when present.
- [x] **W37 — Real-Binary Import Coverage Auditor CLI (`tools/check_import_coverage.py`)** — `DONE` (PR #188, closes #182) *(no game files required — disjoint files `tools/check_import_coverage.py`, `tests/test_import_coverage.py`)*
  - Parses all 68 DLLs / 587 symbols in `docs/imports_required.md`, audits live `runtime/` shim & registry coverage vs `R12` guest-recompiled DLLs, and reports exact remaining symbols.
- [x] **W38 — WebGPU Bink2 / Splash Video Frame Presenter (`web/workers/bink_presenter.js`)** — `DONE` (PR #197, closes #184) *(no game files required — disjoint files `web/workers/bink_presenter.js`, `tests/test_bink_presenter.mjs`)*
  - Double-buffered WebGPU YUV420p/YUVA/RGBA8 texture uploader + BT.709/BT.601 WGSL full-screen letterbox presenter for intro/splash frames (`Bink2Scarlett.dll`).
- [x] **W39 — `MicrosoftGame.Config` & Xbox Package Readiness Inspector CLI (`tools/inspect_game_assets.py`)** — `DONE` (PR #196, closes #185) *(no game files required — disjoint files `tools/inspect_game_assets.py`, `tests/test_inspect_game_assets.py`)*
  - Safely parses Xbox GDK `MicrosoftGame.Config` XML and verifies declared executables, bundled DLLs, and `ShellVisuals` splash assets in `GTA6_GAME_DIR` with sanitized JSON output.
- [x] **W40 — Guest Boot Milestone Tracer & Unresolved-Import Trap Collector (`web/telemetry/boot_tracer.js`)** — `DONE` (PR #202, closes #194) *(no game files required — disjoint files `web/telemetry/boot_tracer.js`, `tests/test_boot_tracer.mjs`)*
  - Tracks real guest boot milestones (`PGIM_LOAD` → `TLS_INIT` → `CRT_INITTERM` → `D3D12_CREATE_DEVICE` → `FIRST_PRESENT`) and deduplicates `__playgta6_unimplemented` / unresolved-import traps for HUD & CrashReporter.
- [x] **W41 — Streaming `PGIM` v1 Guest Image & Shader Bundle Preloader (`web/runtime/pgim_preloader.js`)** — `DONE` (PR #213, closes #195) *(no game files required — disjoint files `web/runtime/pgim_preloader.js`, `tests/test_pgim_preloader.mjs`)*
  - Streams and validates `PGIM` v1 guest images (`R6`/`O10`), `.wgsl.pack` bundles, and S6 reflection manifests via HTTP `206` Range with progress reporting before guest entry.
- [x] **W42 — Browser Local Splash Screen & Live Boot Stage Overlay (`web/ui/splash_overlay.js`)** — `DONE` (PR #223, closes #201) *(no game files required — disjoint files `web/ui/splash_overlay.js`, `tests/test_splash_overlay.mjs`)*
  - Displays the user's own local `ShellVisuals` splash image (`/game/xbs_splash.png`) with live `BootTracer` milestone progress and cross-fades cleanly to WebGPU on `FIRST_PRESENT`.
- [x] **W43 — Local Server Splash, Shader Pack & PGIM Boot-Bundle Endpoints (`server/serve_local.py`)** — `DONE` (PR #229, closes #216) *(no game files required — disjoint files `server/serve_local.py`, `tests/test_serve_local_boot.py`)*
  - Exposes `/shaders/`, allowlisted `/boot/`, and sanitized `/api/status` boot discovery (`splashImage`, `shaderPackUrl`, `reflectionManifestUrl`, `pgimUrl`) with COOP/COEP + HTTP `206` Range.
- [x] **W44 — Browser Boot Pipeline Orchestrator (`web/runtime/boot_orchestrator.js`)** — `DONE` (PR #226, closes #217) *(no game files required — disjoint files `web/runtime/boot_orchestrator.js`, `tests/test_boot_orchestrator.mjs`)*
  - Coordinates `/api/status` discovery → `SplashOverlay` → `PgimPreloader` → `BootTracer` → `FIRST_PRESENT` cross-fade or Smoke Demo fallback.
- [x] **W45 — Wire `SplashOverlay`, `BootOrchestrator`, `BootTracer` & `PgimPreloader` into `web/index.html` & `web/app_bootstrap.js`** — `DONE` (closes #301) *(no game files required)*
  - Mounts `#splash-overlay-root`, loads the UMD boot/splash modules in `web/index.html`, and connects `BootOrchestrator` + `SplashOverlayController` + `BootTracer` + `PgimPreloader` into `bootstrapPlayGta6App` with smooth `FIRST_PRESENT` cross-fade.
