# Wasm64 ↔ JS Host ABI (`runtime/` ↔ `web/loader.js`)

ABI version is defined in `runtime/abi/playgta6_abi.h` (`playgta6::kAbiVersion = 1`).

## Required Browser Capabilities
- `self.crossOriginIsolated === true` (`SharedArrayBuffer` enabled)
- `navigator.gpu` (WebGPU on `OffscreenCanvas`)
- WebAssembly `memory64` (Wasm64, built via Emscripten `-sMEMORY64=1 -pthread`)

## Host Imports (`env`)
Declared in `runtime/os/os_shim.cpp`:

| Function | Signature | Purpose |
| :--- | :--- | :--- |
| `host_log` | `(ptr: i64, len: i64) -> ()` | UTF-8 log line to console / progress channel |
| `host_httpfs_open` | `(path_ptr: i64, path_len: i64) -> i32` | Open relative `/data/` path; returns handle `>= 0` or `-1` |
| `host_httpfs_read` | `(handle: i32, offset: i64, dst_ptr: i64, len: i64) -> i64` | Range read via `io_worker.js` |
| `host_httpfs_close` | `(handle: i32) -> ()` | Close virtual file handle |
| `host_wgpu_submit` | `(cmd_ptr: i64, cmd_len: i64) -> ()` | Forward binary command buffer to `wgpu_worker.js` |

## Shared Input Block (`InputState`, 40 bytes)
Written into a `SharedArrayBuffer` by `web/input/input_bridge.js` (`Int32Array` / `Float32Array`) matching `playgta6::InputState`:
- `0..4`: `seq` (`u32`, incremented via `Atomics.add` after each update)
- `4..8`: `buttons` (`u32` bitmask)
- `8..12`: `mouse_dx` (`i32`)
- `12..16`: `mouse_dy` (`i32`)
- `16..40`: `left_stick_x/y`, `right_stick_x/y`, `left_trigger`, `right_trigger` (`6 × f32`)

## Recompiled Guest Code (`GuestState`, 688 bytes)
Lifted x86-64 functions (`recomp/src/lift`) run on a guest CPU state matching `playgta6::GuestState` (little endian, one per guest thread):

| Offset | Size | Field |
| :--- | :--- | :--- |
| `0` | `16 × 8` | `gpr[16]`: `rax rcx rdx rbx rsp rbp rsi rdi r8..r15` (x86 encoding order) |
| `128` | `8` | `rip` (reserved for the runtime; lifted code does not maintain it) |
| `136` | `8 × 1` | `flags`: `CF PF AF ZF SF OF DF pad`, each `0` or `1` |
| `144` | `8` | `fs_base` |
| `152` | `8` | `gs_base` (TEB on Win64) |
| `160` | `16 × 16` | `xmm0..xmm15` = bits `0..127` of `ymm0..ymm15` (low qword first) |
| `416` | `16 × 16` | `ymm_hi`: bits `128..255` of `ymm0..ymm15` (low qword first) |
| `672` | `4` | `mxcsr` (initialise to `0x1F80`, `kGuestMxcsrDefault`) |
| `676` | `12` | reserved (zero) |

The layout is append-only: fields are only ever added at the end (bumping the size), existing offsets never move. The runtime zero-initialises a new thread's state except `rsp`, `gs_base` and `mxcsr = 0x1F80`. Legacy-SSE instructions leave `ymm_hi` untouched; VEX-128 instructions zero the destination's `ymm_hi`, exactly like hardware.

Calling convention:
- Every lifted function is `void sub_<VA-hex>(GuestState* state)` (`LiftedFn`); arguments and results travel in the state per the Win64 ABI (`rcx rdx r8 r9` / `xmm0..3`, fifth argument at `rsp + 40` on entry, result in `rax` / `xmm0`).
- The guest stack lives in guest memory and is addressed through `rsp`. A direct `call T` pushes the return address and calls `sub_T(state)`; `ret [imm16]` pops it (unchecked) and returns, so host and guest call stacks stay in lockstep.
- Lifted code maps guest virtual addresses 1:1 onto wasm64 linear memory (`fs:`/`gs:` operands add `fs_base`/`gs_base`).

Runtime helpers (C linkage, defined by the runtime, all take the state pointer first):

| Function | Signature | Contract |
| :--- | :--- | :--- |
| `__playgta6_indirect_call` | `(state: ptr, target: i64) -> ()` | Run the guest code at `target` as if called (return address already pushed); returns when it executes `ret`. Used for indirect `call`/`jmp`, IAT calls and unresolved switch targets. |
| `__playgta6_trap` | `(state: ptr, va: i64) -> !` | Trap instruction, invalid/unmapped code, or code after an assumed non-returning call at `va`. |
| `__playgta6_unimplemented` | `(state: ptr, va: i64) -> !` | The instruction at `va` could not be lifted. |
| `__playgta6_divide_error` | `(state: ptr, va: i64) -> !` | `#DE` from `div`/`idiv` at `va`. |

CPU identification helpers take only the state pointer, `(state: ptr) -> ()`. Lifted code stores its registers and flags to the state before the call and reloads them after; the helper touches only the GPR slots and stores 32-bit results zero-extended.

| Function | In | Out |
| :--- | :--- | :--- |
| `__playgta6_cpuid` | `EAX` leaf, `ECX` sub-leaf | `EAX EBX ECX EDX` |
| `__playgta6_rdtsc` | - | `EDX:EAX` = time-stamp counter (monotonic per thread) |
| `__playgta6_rdtscp` | - | `EDX:EAX` as `rdtsc`, `ECX` = `IA32_TSC_AUX` (processor id) |
| `__playgta6_xgetbv` | `ECX` XCR index | `EDX:EAX` = `XCR[ECX]`; an index without an XCR raises `#GP` and does not return |

The advertised `cpuid` features must not exceed what the lifter implements.
