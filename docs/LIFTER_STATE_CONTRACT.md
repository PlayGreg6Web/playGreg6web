# Lifter / runtime contract on the guest state (`%state`)

Lifted functions are `void sub_<VA>(GuestState* state)` (`runtime/abi/playgta6_abi.h`). Inside a
function, registers and flags live in allocas. They are synced with `*state` around calls, at
returns and at tail jumps. This document lists every place outside lifted code that reads or
writes `GuestState`. It also states what a lifter that skips store-backs would have to guarantee.
#2033 was such a lifter, and #2040 reverted it.

**Status of the #2033 evidence: inconclusive.** With #2033, the menu-to-ingame boot crashed at
116 s, and reverting it made one run reach the 900 s limit. Since then, the same crash
(page-table executor, unmapped write, about 103 s) has reproduced on `main` with #2033 reverted.
Runs from one binary are nondeterministic, and a runtime synchronisation race is now the lead.
This audit also found no runtime writer that breaks the contract below. Treat #2033 as
"not proven unsafe", not as the cause. The contract is what any store-back-skipping lifter needs.

**#2104 (the #2033 reland), reverted by #2115: still no static violation.** See
[The #2104 trap](#the-2104-trap-sub_1406395d00x9f) below. A differential IR check over every lifted
function found no store-back that #2033 skips while the slot differs from the alloca.

## What lifted code assumes today (before any skipping)

- **Direct call to `sub_T`.**
  - Before the call, store `touched(T) & live_in_state(T)`, masked by the caller's own
    `written_state`.
  - After the call, check `unwind_pending`, then reload `written(T)`.
  - Registers outside `touched(T)` stay in the caller's allocas, so `%state` keeps a stale copy
    until the caller's next full write-back.
- **Indirect call or IAT call** (`__playgta6_indirect_call`): full `sync_out()` (own
  `written_state`) before, and `sync_in(ALL_STATE, live flags)` after.
- **Tail jump, indirect jump, `ret`:** full `sync_out` (masked by the target's flag `live_in`).
- **`cpuid` / `rdtsc` / `rdtscp` / `xgetbv`:** full `sync_out` before and full `sync_in` after the
  helper.
- **Traps:**
  - Calls to `__playgta6_trap`, `__playgta6_unimplemented` and `__playgta6_divide_error` do a full
    `sync_out`, call the helper, then hit `unreachable`.
  - An unknown `resume_rip` in an owner's `unwind:` block ends in `__playgta6_trap` without a
    sync.
- **Unwind (`unwind_pending` set after any guest call):**
  - A frame that does not own the restore returns without writing back.
  - The owner (entry `rsp` above the restored `rsp`) does a full `sync_in` and switches on
    `resume_rip`.

Summaries (`recomp/src/lift/flag_liveness.rs`) give `ALL_STATE` to the following:
- any function that contains an indirect transfer, an unmodelled instruction (`cpuid`, `div`,
  `ldmxcsr`, ...), a trap, or a call outside the analysed set;
- transitively, any caller of such a function.

A precise (non-`ALL`) summary therefore only exists for pure guest code that never hands
`%state` to the runtime.

## Every runtime site that touches `GuestState`

Paths are under `runtime/` unless shown in full. A **writer** changes a field that lifted code
holds in an alloca. A **reader** observes a stale value if the field was not stored back first.

### Writers

| Site | Writes | Reached from lifted code via |
|---|---|---|
| Generated import thunks (`recomp/src/stubs.rs`), bridge thunks (`guest/com_bridge.h`, `store_result`), `sret_thunk` (`gpu/d3d12_guest_objects.h`), COM QI/AddRef/Release | `rax`, or `xmm0` (generated thunks write the low lane only; bridge thunks zero the high lane), and `rsp += 8`. Generated and bridge thunks write neither while `unwind_pending` is set | indirect call (ALL) |
| Nested host-to-guest calls: `call_guest_tagged` (`guest/guest_runtime.cpp`), `call_guest_raw` (`guest/com_bridge.cpp`) on the **caller's** state (innermost `GuestStateScope`) | `rcx rdx r8 r9 rsp` and float args in `xmm0-3`. Restores only `rsp`, so `rax r10 r11 xmm0-5 ymm_hi flags DF mxcsr` keep whatever the callback left | indirect call (ALL). Thunks that call back this way: `qsort` (void: `rax` = last comparator result), `bsearch`, `_initterm(_e)`, `InitOnceExecuteOnce`, `FlsFree` destructors, `_callnewh`, `_execute_onexit_table`, `_cexit` / `exit`, `UnhandledExceptionFilter`, GameInput device callbacks, XGameSave enumerate, the XAsync provider, `XAsyncRun`, and task-queue and XAsync completions when they run synchronously |
| `call_guest_routine` (XMem alloc/free hooks, `os/xbox_scarlett.cpp`) | Saves the whole state and restores it afterwards (`state = saved`), except while an unwind is pending: then the restored context stays | indirect call (ALL) |
| Exception / context thunks (`guest/guest_exceptions.cpp`): `RaiseException`, `RtlRaiseException` (continue-execution), `RtlUnwind(Ex)`, `RtlRestoreContext`, consolidate, `longjmp` | **all 16 GPRs including the non-volatiles**, `flags[0..6]`, `mxcsr`, `xmm0-15`, `rip`, `resume_rip`, `unwind_pending`. `longjmp` writes `rbx rsp rbp rsi rdi r12-r15 mxcsr xmm6-15`. `ymm_hi` is never restored | indirect call (ALL), then the unwind protocol |
| `RtlCaptureContext`, lookup, VirtualUnwind and filter setters | `rax`, `rsp += 8` | indirect call (ALL) |
| `cpuid` / `rdtsc` / `rdtscp` / `xgetbv` (`guest/cpu_helpers.cpp`) | `rax rbx rcx rdx` / `rax rdx` / `+rcx` / `rax rdx` | helper (ALL, full sync) |
| `__playgta6_indirect_call` | `rip` (not tracked by lifted code) | indirect call |
| Thread start (`guest_callbacks.cpp`), thread pool (`guest_rtl.cpp`), lazy thread state (`com_bridge.cpp` `current_guest_state`), boot init hook | A fresh state per host thread (`thread_local`); writes `rsp`, `gs_base`, `mxcsr` | not from a running lifted frame |

**No runtime code replaces a guest function at its VA.** `@sub_*` symbols are only defined by
the lifter. There is one definition per VA, a failed lift aborts the build, and synthetic
addresses (COM vtables, exports) are reachable only through the dispatcher. **No runtime code
writes another thread's state.** There is no fiber or `SetThreadContext` support. The bound state
and the lazy state are both `thread_local`. The host VEH handlers (lazy commit, write tracking,
`host_fault`) never touch `GuestState`. The memory slow paths (`guest/guest_memory.cpp`) take no
state and end the run on a fault.

### Readers

| Site | Reads | When |
|---|---|---|
| Import / bridge thunks | `rcx rdx r8 r9`, `xmm0-3` (low lanes), stack args at `rsp+40+8k` | inside an indirect call (after a full sync) |
| `capture_caller_context` (RtlCaptureContext / RaiseException / setjmp) | all GPRs, `flags`, `mxcsr`, `xmm0-15`, `[rsp]` | inside an indirect call |
| `boot_dispatch_observer` (`guest/boot.cpp`) | `rcx rdx r8 r9 rsp` | inside an indirect call |
| `__playgta6_probe_hit` (`--probe-functions`, inserted at function entry **after** lifting) | `rcx rdx r8 r9` | at callee entry, **during a direct call**. A probed build's summaries (`FlagSummaries::compute_with_probes`) count the probed function as reading all four, so callers sync them first. Logging only |
| `__playgta6_probe_exit` (`--probe-functions`, inserted before every normal `ret void` **after** lifting; a tail call out becomes a call plus the hook) | `rax`, `xmm0`, `rsp` | at function exit, state already stored. `rsp` is in the probe's entry reads so the runtime pairs exit with entry. Logging only; the `unwind` returns get no hook |
| Fault capture / backtrace / thread dump (`guest/boot.cpp`, `guest_runtime.cpp format_fault`) | everything (on copies). The thread dump reads another running thread's state without synchronisation | on fatal faults or a dump. Values can be stale or torn, but nothing is written |
| `__playgta6_trap` / `unimplemented` / `divide_error` | `unwind_pending`, `resume_rip`, `rsp` | after a full sync, then the run ends |

## The contract a store-back-skipping lifter needs

#2033 skipped `store alloca -> slot` when it believed the alloca still equalled the slot. That
belief was set by a sync in either direction and cleared by any write to either side or by any
call other than a pure intrinsic. It was kept across a direct call for slots outside
`written(T)` (except `rsp`). The belief is only true if the following hold:

1. **Only lifted code writes `%state` during a direct call to a precise-summary callee.**
   - Every runtime helper that receives `%state` and returns may write any field, so it may only
     appear in functions whose summary is `ALL_STATE`.
   - Read-only observers (`probe_hit`) and no-return traps are exempt.
   - True today; pinned by `recomp/src/lift/tests/state_boundary.rs`.
2. **`written(T)` covers every store the lifted callee (transitively) makes**, including VEX-128
   `ymmh` zeroing, flags (`written_flags`) and sub-register writes. A caller's summary covers its
   callees' summaries; also pinned by the test. The #2040 investigation checked this statically
   over all 435,719 lifted functions and found no violation.
3. **Every call that hands `%state` to the runtime and returns clears all facts.** After it, a
   fact may only be re-established by an explicit reload from the slot. This covers
   `__playgta6_indirect_call` (imports, callbacks, exception and context thunks, `longjmp`) and
   the `cpuid` family. Nested callbacks write `rax r10 r11 xmm0-5 ymm_hi flags mxcsr` without
   restoring them. Context restores write every GPR, including `rbx rbp rsi rdi r12-r15`.
4. **Code that is entered other than through its CFG predecessors starts with no facts:**
   resume targets (`unwind:` switch on `resume_rip`), loop heads, and the function entry when it
   has other predecessors.
5. **One thread per `GuestState`.** No host thread may write a state while its own guest thread
   runs lifted code. This holds today (`thread_local` binding), but nothing enforces it. A
   cross-thread write here would be invisible to every lifter fact. The unconditional store-backs
   would partly overwrite such a write, which can hide or move a race.
6. **Post-lift IR insertions** (`insert_entry_probe`) bypass the lifter's fact tracking. They
   must neither write `%state` nor branch into a block with established facts.

When any of these cannot be shown, **force a full sync**: clear all facts and store every written
register. This applies to:
- every indirect call, IAT call and indirect jump;
- every `cpuid` / `rdtsc` / `rdtscp` / `xgetbv`;
- every direct call to a callee whose summary is `ALL_STATE`, or that is outside the analysed set;
- every tail jump to such a target;
- the entry of every resume target.

Before #2033, rules 1-4 already held. Nothing found here shows #2033 broke one of them.

## The #2104 trap (`sub_1406395D0+0x9f`)

One std-pack boot with #2104 trapped at `sub_1406395D0+0x9f` on four worker threads at about
658 s (callers `sub_143E54A70` <- `sub_142570310` <- `sub_14221C1F0` <- `sub_142215EA0`).

**What the trap is.** `+0x9f` is a guest `int3`. The function pops an entry from a shared
lock-free free list (`lock cmpxchg` on a tagged head) and executes the `int3` when the head holds
the "list empty" sentinel. All four threads had that sentinel in `eax`, so the pool was empty.
The trap does not consume a stale register. It shows that the pool ran dry: either entries
leaked or the free list was cut short somewhere else, earlier. The trap site cannot say where.

**#2033 does not change the trapping function.** With and without #2033, `sub_1406395D0` lifts
to the same IR. Its callers and the pool's free path differ only in skipped `store ... ptr
%state.*` lines.

**Differential check over every function.** This used a local checker that is not in the repo.
It lifted every function of the executable (435,719) and of `vcruntime140`, `vcruntime140_1`,
`msvcp140` and `Bink2Scarlett`, once with and once without #2033, using whole-program summaries
and the default (flat) memory model of that boot. The checker ran a forward dataflow over each
function's emitted IR. It tracked, per register and flag, the value in the alloca and the value
in the `%state` slot. The two count as equal only when they hold the same SSA value. At joins it
intersected over the predecessors. At a direct call it overwrote the slots of `written(T)` /
`written_flags(T)`. At a call to a runtime helper that receives `%state` it overwrote every slot.
It then checked alloca == slot at every observation:
- a direct call (`touched & live_in_state`, `touched_flags & live_in`);
- a tail call;
- a returning or trapping helper that receives `%state`;
- a normal `ret`.
- **Result:** no failure that appears only with #2033 (19,387 failures with it, 19,834 without;
  these are where the checker cannot prove two SSA values equal). The checker does catch real
  mistakes: deliberately changing #2033's join to a union makes it report failures at once.
- **Rule 2 checked against the emitted IR:** for every function, the slots its IR stores itself,
  plus the transitive closure over its `call` / `tail call` targets, are a subset of
  `written` / `written_flags`. There are no exceptions.

So, under rules 1-6, #2033 leaves `%state` identical at every point where lifted code or a
runtime helper reads it. **No missing static invariant was found, and no recomp test can
reproduce one.** A unit test that "fails with #2033's logic" would need a real counterexample,
and the check above found none.

**What #2033 does change at run time.** It removes a side effect that rules 1-6 do not account
for:

7. **Unconditional store-backs repair foreign writes; skipped store-backs do not.** Suppose
   something other than the lifted callee writes slot `r` during a direct call, where
   `r ∉ written(T)`. That can be a runtime path missed by rule 1, or another host thread sharing
   the state against rule 5. Without #2033, the caller's next store-back puts the alloca value
   back into `%state`, so the foreign write survives only until then. With #2033, the store-back
   is skipped and the foreign write stays. A store-back-skipping lifter therefore needs rules 1
   and 5 to hold **dynamically**, not just in the static audit. A cheap way to test this at run
   time: a debug build of the skipping lifter that turns each skipped store into
   `load slot; icmp ne alloca; br -> __playgta6_trap`. That asserts the invariant at exactly the
   points #2033 relies on it, so it names the first foreign write or race.

Until such a debug build has run past the trap point with no assertion, treat the 658 s pool
exhaustion like the earlier 116 s crash. It is one run against twelve. It may come from the
nondeterministic runtime race described above, and #2033's lower store traffic may only change
its timing. It may also be a real violation of rule 1 or 5 that unconditional store-backs used
to hide.

## Incidental findings (pre-existing, independent of #2033)

- **Fixed: generated import thunks ignored `unwind_pending`** (`recomp/src/stubs.rs`). If a shim
  ran a nested guest call that threw, the thunk still wrote `rax` and moved the restored `rsp`
  by 8. They now check `pg6_seam_unwind_pending` after the shim and return without touching the
  state. (`call_guest_tagged` returns without restoring `rsp` while an unwind is pending, which
  is what the unwind needs.)
- **Fixed: `call_guest_routine` (XMem hooks) restored the whole state**, which also cleared
  `unwind_pending` / `resume_rip`, so an exception thrown by the guest allocator was silently
  dropped. A pending unwind now keeps the restored state (`tests/test_xmem_hooks.cpp`).
- **The main thread has no `GuestStateScope`.** Synchronous callbacks from generated imports on
  the boot thread run on a separate lazy state; on worker threads they run on the caller's
  state. Behaviour differs by thread.
- **The shard lift cache key** (`recomp/src/pipeline/shard_lift.rs` `function_key`) once left out
  callee `touched_flags` / `written_flags`, so a discovery change that only widened a callee's
  written flags reused stale caller IR with `--lift-cache`. Fixed: all six summary fields are
  hashed (`SHARD_CACHE_VERSION` 2) and `function_key_covers_every_callee_summary_field` pins it.
- **Fixed: `probe_hit` logs could show stale arguments** for callees that do not read them. A
  probed build now treats `rcx rdx r8 r9` as read at the probed entry (see Readers).
