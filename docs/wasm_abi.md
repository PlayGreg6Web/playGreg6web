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

## Recompiled Guest Code (`GuestState`, 704 bytes)
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
| `688` | `8` | `resume_rip`: where a pending context restore continues |
| `696` | `1` | `unwind_pending`: `1` while a context restore (exception unwind, `longjmp`) propagates |
| `697` | `7` | reserved (zero) |

The layout is append-only: fields are only ever added at the end (bumping the size), existing offsets never move. The runtime zero-initialises a new thread's state except `rsp`, `gs_base` and `mxcsr = 0x1F80`. Legacy-SSE instructions leave `ymm_hi` untouched; VEX-128 instructions zero the destination's `ymm_hi`, exactly like hardware.

Calling convention:
- Every lifted function is `void sub_<VA-hex>(GuestState* state)` (`LiftedFn`); arguments and results travel in the state per the Win64 ABI (`rcx rdx r8 r9` / `xmm0..3`, fifth argument at `rsp + 40` on entry, result in `rax` / `xmm0`).
- The guest stack lives in guest memory and is addressed through `rsp`. A direct `call T` pushes the return address and calls `sub_T(state)`; `ret [imm16]` pops it (unchecked) and returns, so host and guest call stacks stay in lockstep.
- Lifted code turns a guest virtual address into a host address with the translation of the [Guest Address Space](#guest-address-space) (`fs:`/`gs:` operands add `fs_base`/`gs_base` to the guest address first). The `GuestState` pointer itself is a host pointer.
- Context restores (`RtlRestoreContext`, the end of `RtlUnwindEx`, `longjmp`) continue at an arbitrary `rip` inside a function while host frames of the abandoned guest frames are still active. The runtime writes the restored registers into the state, sets `resume_rip` and `unwind_pending = 1` and returns. Every lifted activation (and every runtime frame that called guest code) seeing `unwind_pending` after a call returns to its caller immediately, without writing registers back; the activation whose entry `rsp` lies above the restored `rsp` clears the flag, reloads its registers from the state and branches to the block at `resume_rip`.

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

## Guest Address Space
Guest virtual addresses (VAs) reach host memory through **one** translation, shared by lifted code, the import bridge (`pg6_guest_seam.h`, `pg6_seam_host_ptr` / `pg6_seam_guest_addr`) and the runtime (`VirtualMemoryArena`, the image loader, guest stacks, heaps):

```
host address = __playgta6_guest_base + guest VA        (64-bit wrapping add)
```

| Symbol / constant | Defined in | Meaning |
| :--- | :--- | :--- |
| `__playgta6_guest_base` | runtime (`uint64_t`, C linkage) | Translation base. Set before any guest code runs, constant while guest code runs. `0` = identity. |
| `kGuestAddressSpaceBase` | `runtime/abi/playgta6_abi.h` | `0x1'0000'0000`: lowest guest VA the runtime hands out (`VirtualAlloc`, heaps, stacks, images). |

- The runtime backs the guest address space with **one contiguous host region**: guest VAs `[kGuestAddressSpaceBase, kGuestAddressSpaceBase + size)` are contiguous host memory, so a guest pointer stays valid across page and allocation boundaries and `host_ptr(va + n) == host_ptr(va) + n`. Reserve / commit / decommit / protect / `VirtualQuery` bookkeeping (Win32 semantics, 4 KiB pages, 64 KiB allocation granularity, zero-filled commits) lives on top of it.
- Lifted code reads `__playgta6_guest_base` (an invariant load, hoisted per function) and adds it to every guest address. A module may instead be lifted against a fixed constant base; a constant `0` (identity) adds nothing to the address computation. A constant-base module must only run with a runtime whose `__playgta6_guest_base` equals that constant.
- The bridge maps null to null in both directions; every other value uses the formula above. A host pointer outside the guest region (e.g. runtime `malloc` memory) round-trips to a guest VA that lifted code can dereference.

### Browser (wasm64): identity
- `__playgta6_guest_base == 0`: a guest VA *is* the linear-memory offset; host pointers and guest pointers are interchangeable.
- Linear memory `[0, 4 GiB)` belongs to the runtime (Emscripten static data, stacks, `malloc`); `sbrk` is fenced so the runtime heap never grows past `kGuestAddressSpaceBase`. Linear memory `[4 GiB, 16 GiB)` is the guest address space (12 GiB): the game image sits at its preferred base `0x1'4000'0000` inside it.
- The memory is shared (pthreads), declared with a 16 GiB maximum (the browser limit for memory64) and starts small; it grows on demand when guest pages are committed (it covers the highest committed guest page) and never shrinks. Decommitted guest pages are zeroed again before they are reused.
- Guest VAs at or above the current memory size trap (out of bounds). Null and other VAs below 4 GiB do **not** trap: they read and write runtime memory.
- Page protections are tracked (for `VirtualQuery` / `VirtualProtect` and for the runtime's own checked accesses) but cannot be enforced on lifted code.

### Native host builds
- The runtime reserves the guest address space from the OS and commits pages on demand. The process arena is placed at its own VAs (identity) when the OS allows it; otherwise (or for additional arenas, e.g. in tests) `__playgta6_guest_base` is non-zero and the VAs below `kGuestAddressSpaceBase` are reserved inaccessible, so null and small guest pointers fault.
- Page protections are enforced with OS page protection (execute is treated as read).

### Translated CPU window (browser target)
The title reserves a 16 GiB CPU window (24 GiB on another console profile) and maps 64 KiB pool pages at arbitrary window slots; it cannot be backed inside the 12 GiB identity pool. The window is therefore placed above the wasm64 memory and every guest access to it translates through a per-slot table:

```
va in [kGuestWindowBase, kGuestWindowBase + kGuestWindowSize):
    host address = va + __playgta6_window_table[(va - kGuestWindowBase) >> 16]
otherwise:
    host address = __playgta6_guest_base + va
```

| Symbol / constant | Defined in | Meaning |
| :--- | :--- | :--- |
| `kGuestWindowBase` | `runtime/abi/playgta6_abi.h` | `0x4'0000'0000` (16 GiB): first guest VA of the window, above any wasm64 memory. |
| `kGuestWindowSize` | `runtime/abi/playgta6_abi.h` | `0x6'0000'0000` (24 GiB): window range (covers both console profiles). |
| `kGuestWindowSlotBytes` | `runtime/abi/playgta6_abi.h` | `65536`: one table slot. |
| `kGuestWindowTrapDelta` | `runtime/abi/playgta6_abi.h` | `2^63`: delta of an unmapped slot; the address lands beyond any memory (wasm: "out of bounds" trap, native: non-canonical fault). |
| `__playgta6_window_table` | runtime (`int64_t[kGuestWindowSize / 65536]`, C linkage) | Delta per slot. Changes at run time (map / unmap): lifted code reads entries with ordinary loads, never invariant ones. |

- Lifted code is built with the lifter's `MemoryModel::Windowed` at these constants, with `stack_flat` (stack-pointer-relative accesses skip the window check: guest stacks never lie in the window). An access of `n > 1` bytes whose offset in its slot exceeds `65536 - n` may straddle two slots and calls the slow-path helpers below; naturally aligned accesses (atomics) never straddle. `rep movs` / `rep stos` over a range touching the window call the segmented helpers.
- Runtime helpers (C linkage): `void* __playgta6_guest_read(u64 va, u64 n)` (copies `n <= 64` bytes into a per-thread scratch buffer and returns it), `void* __playgta6_guest_scratch()`, `void __playgta6_guest_write(u64 va, u64 n)` (copies the scratch buffer to `va`), `void __playgta6_guest_memmove(u64 dst, u64 src, u64 n)`, `void __playgta6_guest_memset(u64 dst, u8 value, u64 n)`. An unmapped slot raises a guest fault.
- Host code that receives a guest pointer that may point into the window must translate it with the runtime's shared function (`os/guest_window.h`) and must not assume a guest buffer is host-contiguous across 64 KiB slots (segmented copies or a contiguous view).
- Page protections inside the window are not tracked per slot; an unmapped slot always traps.
