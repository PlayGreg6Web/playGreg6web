//! Resume protocol: continuing guest execution after an exception unwind or `longjmp`.
//!
//! `RtlRestoreContext`, the end of `RtlUnwindEx` and `longjmp` continue at an arbitrary `rip`
//! inside some function (a catch continuation, an `__except` body, the instruction after a
//! `setjmp` call) with that frame's `rsp`, while the host (wasm) call stack still holds one native
//! frame per lifted activation below it. Lifted code does not track `rip` and cannot be entered
//! mid-function, so control is transferred cooperatively (runtime side:
//! `runtime/guest/guest_exceptions.h`; state fields: `docs/wasm_abi.md`):
//!
//! 1. The runtime writes the restored registers into the guest state, sets
//!    [`STATE_RESUME_RIP_OFFSET`](super::STATE_RESUME_RIP_OFFSET) and
//!    [`STATE_UNWIND_PENDING_OFFSET`](super::STATE_UNWIND_PENDING_OFFSET) and returns.
//! 2. After every call that can run guest code (direct calls and calls through
//!    `__playgta6_indirect_call`, including calls assumed not to return), a lifted function tests
//!    the pending byte. If it is set, the activation jumps to its `unwind` block instead of
//!    reloading its registers.
//! 3. In a function without resume targets the `unwind` block is `ret void`: the activation
//!    returns without writing its registers back, so the restored context survives. Tail calls
//!    (`tail call @sub_T; ret void`) need no test, they return straight away.
//! 4. In a function with resume targets ([`cfg::Function::resume_targets`], block starts found by
//!    discovery) the `unwind` block compares the restored `rsp` with the `rsp` the activation had
//!    on entry. Every frame of the activation lies below its entry `rsp`, and every callee frame
//!    below the `rsp` of its call site, so the activation that owns the restored frame is the
//!    innermost one whose entry `rsp` is above the restored `rsp`. A non-owner returns like in 3.
//!    The owner clears the flag, reloads all its registers from the state and `switch`es on the
//!    resume `rip` to the matching block. Recursion is handled because each activation compares
//!    its own entry `rsp`. An unknown `rip` sets the flag again and calls
//!    [`HELPER_TRAP`](super::HELPER_TRAP) with it; the runtime reports it as a context restore to
//!    an unknown resume target.
//!
//! The cost on the normal path is one byte load and one well-predicted branch per call return,
//! plus one `i64` load at entry in functions with resume targets. Measured when this was added:
//! on a sample of 3,562 functions of the target executable (14,792 call sites) the IR grew by
//! 1.6% and the `-O2` wasm64 object by 1.6% (6.6% / 7.8% for the functions that have resume
//! targets); a call-bound benchmark (recursive `fib(32)`, about 7M lifted calls, node 23) ran
//! within measurement noise (under 1%).
//!
//! Alternative considered: LLVM `invoke` + wasm exception handling (`catchswitch`/`catchpad`
//! lowered to `try`/`catch_all`), with the runtime throwing instead of setting a flag. Only
//! functions with resume targets would pay anything. It works with recent toolchains (a probe with
//! rustc's LLVM 23 and node 23 catches an exception thrown by an import), but it ties every
//! lifted module and the C++ runtime to the wasm EH feature: the runtime would have to be built
//! with `-fwasm-exceptions` (its frames between guest activations, e.g. `call_guest` and the
//! dispatcher, must unwind and run destructors; it is built without exceptions today), every
//! lifted module would need the EH target feature and a personality symbol, CI's LLVM 18 and the
//! browsers' transition from the legacy EH encoding to `exnref` add risk, a `catch_all` also
//! catches the runtime's own fatal errors (each landing pad must test and rethrow), and the
//! native host-diff tests would need host unwind tables. The flag protocol is plain IR that any
//! LLVM and engine accept, and its measured cost is small. It is kept behind this module's three
//! hooks — `guest_call` (emits a call that can run guest code), `unwind_entry` (function entry)
//! and `unwind_blocks` (shared blocks at the end) — so it can be swapped later.

use std::fmt::Write as _;

use super::{FunctionLifter, block_label, imm};
use crate::lift::{GUEST_STATE_TYPE, HELPER_TRAP};

/// Field indices of `%playgta6.GuestState` (see `GUEST_STATE_TYPE` in the parent module).
const FIELD_GPR: u32 = 0;
const FIELD_RESUME_RIP: u32 = 9;
const FIELD_UNWIND_PENDING: u32 = 10;
const RSP: u32 = 4;

/// Label of the shared block every pending check branches to.
const UNWIND_LABEL: &str = "unwind";
/// Value holding the activation's `rsp` on entry (functions with resume targets only).
const ENTRY_RSP: &str = "%entry_rsp";

impl FunctionLifter<'_> {
    /// Whether the function can be the target of a context restore.
    fn has_resume_targets(&self) -> bool {
        self.func
            .resume_targets
            .iter()
            .any(|t| *t != self.func.entry && self.func.blocks.contains_key(t))
    }

    /// Emits `call_text` (a call that can run guest code), then the pending test: the current
    /// block ends and lifting continues in a fresh block on the normal path.
    pub(super) fn guest_call(&mut self, call_text: &str) {
        self.emit(call_text);
        let ptr = self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 {FIELD_UNWIND_PENDING}"
        ));
        let pending = self.val(&format!("load i8, ptr {ptr}"));
        let taken = self.val(&format!("icmp ne i8 {pending}, 0"));
        let next = self.new_label("after_call");
        self.emit(&format!(
            "br i1 {taken}, label %{UNWIND_LABEL}, label %{next}"
        ));
        self.start_block(&next);
        self.unwind_sites += 1;
    }

    /// Entry block addition (after the registers were loaded): remembers the entry `rsp`.
    pub(super) fn unwind_entry(&mut self) {
        if self.unwind_sites == 0 || !self.has_resume_targets() {
            return;
        }
        let ptr = self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 {FIELD_GPR}, i64 {RSP}"
        ));
        self.emit(&format!("{ENTRY_RSP} = load i64, ptr {ptr}"));
    }

    /// The shared `unwind` block (and the resume dispatch), emitted once after the lifted blocks.
    pub(super) fn unwind_blocks(&mut self) {
        if self.unwind_sites == 0 {
            return;
        }
        self.start_block(UNWIND_LABEL);
        if !self.has_resume_targets() {
            self.emit("ret void");
            return;
        }
        let rsp_ptr = self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 {FIELD_GPR}, i64 {RSP}"
        ));
        let restored_rsp = self.val(&format!("load i64, ptr {rsp_ptr}"));
        let owner = self.val(&format!("icmp ugt i64 {ENTRY_RSP}, {restored_rsp}"));
        self.emit(&format!(
            "br i1 {owner}, label %resume, label %unwind_leave"
        ));
        self.start_block("unwind_leave");
        self.emit("ret void");

        self.start_block("resume");
        let pending = self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 {FIELD_UNWIND_PENDING}"
        ));
        self.emit(&format!("store i8 0, ptr {pending}"));
        self.sync_in();
        let rip_ptr = self.val(&format!(
            "getelementptr inbounds {GUEST_STATE_TYPE}, ptr %state, i64 0, i32 {FIELD_RESUME_RIP}"
        ));
        let rip = self.val(&format!("load i64, ptr {rip_ptr}"));
        let mut arms = String::new();
        let targets: Vec<u64> = self
            .func
            .resume_targets
            .iter()
            .copied()
            .filter(|t| *t != self.func.entry && self.func.blocks.contains_key(t))
            .collect();
        for target in targets {
            let _ = write!(
                arms,
                " i64 {}, label %{}",
                imm(target, 64),
                block_label(target)
            );
        }
        self.emit(&format!(
            "switch i64 {rip}, label %resume_unknown [{arms} ]"
        ));
        // No block at the resume rip: report it (the runtime sees the flag set again).
        self.start_block("resume_unknown");
        self.emit(&format!("store i8 1, ptr {pending}"));
        self.emit(&format!("call void @{HELPER_TRAP}(ptr %state, i64 {rip})"));
        self.emit("unreachable");
    }
}
