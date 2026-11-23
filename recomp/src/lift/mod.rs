//! x86-64 -> LLVM IR lifter for `wasm64-unknown-unknown` (memory64).
//!
//! Turns the functions found by [`discover`](crate::discover::discover) into textual LLVM IR (`.ll`).
//! No LLVM library bindings are used; [`compile`] shells out to `llc`/`clang` to produce a wasm64
//! object.
//!
//! # Guest CPU state
//! Every lifted function is `define void @sub_<VA>(ptr noalias %state)`, where `%state` points to a
//! [`GUEST_STATE_TYPE`] in linear memory (byte offsets, little endian):
//!
//! | Offset | Size | Field |
//! | :--- | :--- | :--- |
//! | [`STATE_GPR_OFFSET`] (0) | 16 x 8 | `rax rcx rdx rbx rsp rbp rsi rdi r8..r15` (x86 encoding order) |
//! | [`STATE_RIP_OFFSET`] (128) | 8 | `rip` (reserved for the runtime; lifted code does not maintain it) |
//! | [`STATE_FLAGS_OFFSET`] (136) | 8 x 1 | `CF PF AF ZF SF OF DF pad`, one byte each, always 0 or 1 |
//! | [`STATE_FS_BASE_OFFSET`] (144) | 8 | `fs` segment base |
//! | [`STATE_GS_BASE_OFFSET`] (152) | 8 | `gs` segment base (TEB on Win64) |
//! | [`STATE_XMM_OFFSET`] (160) | 16 x 16 | `xmm0..xmm15` (reserved, not lifted yet) |
//!
//! Inside a function, every GPR the function touches and every flag lives in an `alloca` that is
//! loaded from the state at entry and after each call, and stored back before each call and every
//! exit. LLVM's `mem2reg`/`SROA` turn these into SSA values; flag slots that are never read are
//! dead-store-eliminated. `%state` is `noalias`: guest code never addresses the state block itself.
//!
//! Sub-registers follow x86-64 rules: 32-bit writes zero-extend to 64 bits, 8/16-bit writes
//! preserve the untouched bits, and `AH/CH/DH/BH` address bits 8..16.
//!
//! # Flags
//! `CF PF AF ZF SF OF` are individual `i1` slots computed exactly for every supported instruction.
//! Flags the architecture leaves *undefined* for an instruction keep their previous value. `DF`
//! is assumed clear (string instructions are not supported yet).
//!
//! # Guest memory
//! Guest virtual addresses are 64-bit and map 1:1 onto wasm64 linear memory (`inttoptr`). All
//! guest loads/stores go through one place (`FunctionLifter::guest_ptr`) so the mapping policy can
//! change later. Accesses are `align 1` (x86 allows unaligned access). All addressing modes are
//! supported (`base + index*scale + disp`, RIP-relative, 32-bit address size, `fs:`/`gs:`).
//!
//! # Calls, returns and the guest stack
//! `rsp` lives in the guest state and points into guest memory, so the guest stack is laid out
//! exactly like on hardware (return addresses, Win64 shadow space, stack arguments, saved
//! registers):
//! - direct `call T`: push the return address (`next_ip`) onto the guest stack, then
//!   `call void @sub_T(ptr %state)`. Registers are synced around the call.
//! - `ret [imm16]`: pop the return address (its value is not checked) plus `imm16` bytes, then
//!   LLVM `ret void`. The host call stack and the guest stack therefore stay in lockstep.
//! - [`Edge::TailCall`](crate::cfg::Edge::TailCall) (a `jmp`, `jcc` side or fallthrough into another
//!   function): tail call `@sub_T` followed by `ret void` (the callee's `ret` pops our caller's
//!   return address, exactly like on hardware).
//! - indirect `call`/`jmp` (register or memory, e.g. `call [rip+iat]`): same as above, but the
//!   target is dispatched at run time through [`HELPER_INDIRECT_CALL`].
//! - a switch `jmp` whose table discovery resolved (successors are
//!   [`Edge::JumpTable`](crate::cfg::Edge::JumpTable)) becomes an LLVM `switch` on the computed
//!   guest target over the case blocks; the default arm (a value outside the proven targets) is the
//!   dispatcher path above, so an incomplete table proof stays correct.
//!
//! # Control flow
//! Intra-function control flow comes from each block's [`Terminator`](crate::cfg::Terminator) and
//! typed successor edges: `Fallthrough`/`Jump` branch to their single successor, `Branch`
//! evaluates the `jcc` condition and picks `[taken, not taken]`, `Return`/`IndirectJump` are
//! lowered from the instruction. `Trap` (`int3`, `ud2`, `__fastfail`, `hlt`), `Invalid`
//! (undecodable or unmapped code, including empty blocks), code after a `NoReturnCall` and direct
//! calls into memory that discovery found unmapped or non-executable end in [`HELPER_TRAP`].
//!
//! # Runtime helpers
//! The runtime must provide (all take the state pointer first):
//! - `void __playgta6_indirect_call(ptr state, i64 target)`: run the guest code at `target` as if
//!   it had been called; the return address has already been pushed by the caller. Returns when
//!   that code executes its `ret`.
//! - `void __playgta6_unimplemented(ptr state, i64 va)` (noreturn): an instruction at `va` could
//!   not be lifted.
//! - `void __playgta6_divide_error(ptr state, i64 va)` (noreturn): `#DE` raised by `div`/`idiv`
//!   at `va` (division by zero or quotient overflow).
//! - `void __playgta6_trap(ptr state, i64 va)` (noreturn): guest execution reached a trap: the
//!   trap instruction at `va`, undecodable/unmapped code at `va`, a direct call into non-code
//!   memory at `va`, or the instruction after a call that was assumed never to return.
//!
//! # Unsupported instructions
//! Unsupported instructions (or unsupported operand forms) are never mis-lifted: the lifter emits
//! a call to [`HELPER_UNIMPLEMENTED`] followed by `unreachable` and records the instruction in
//! [`LiftedFunction::unsupported`]. Malformed control-flow graphs are reported as [`LiftError`].
//! The lifter never panics on arbitrary decoded input.

pub mod compile;
mod function;

use std::collections::BTreeSet;
use std::fmt::{self, Write as _};

use crate::cfg;

/// LLVM target triple of the emitted module.
pub const WASM64_TRIPLE: &str = "wasm64-unknown-unknown";

/// LLVM data layout of the emitted module (llc/clang replace it with their own wasm64 layout).
pub const WASM64_DATALAYOUT: &str =
    "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20";

/// Name of the guest CPU state type in the emitted IR.
pub const GUEST_STATE_TYPE: &str = "%playgta6.GuestState";

/// Byte offset of the 16 general purpose registers in the guest state.
pub const STATE_GPR_OFFSET: u64 = 0;
/// Byte offset of `rip` in the guest state.
pub const STATE_RIP_OFFSET: u64 = 128;
/// Byte offset of the flag bytes (`CF PF AF ZF SF OF DF pad`) in the guest state.
pub const STATE_FLAGS_OFFSET: u64 = 136;
/// Byte offset of the `fs` segment base in the guest state.
pub const STATE_FS_BASE_OFFSET: u64 = 144;
/// Byte offset of the `gs` segment base in the guest state.
pub const STATE_GS_BASE_OFFSET: u64 = 152;
/// Byte offset of `xmm0..xmm15` in the guest state.
pub const STATE_XMM_OFFSET: u64 = 160;
/// Distance between two XMM registers in the guest state.
pub const STATE_XMM_STRIDE: u64 = 16;
/// Total size of the guest state in bytes.
pub const STATE_SIZE: u64 = 416;

/// Offset of the fifth Win64 argument from `rsp` when lifted code enters a callee: the 8-byte
/// return address pushed by the lifted `call` plus the caller's 32-byte shadow space.
pub const CALLEE_STACK_ARG_OFFSET: u64 = 8 + 32;

/// Runtime helper for indirect calls and jumps: `void (ptr state, i64 target)`.
pub const HELPER_INDIRECT_CALL: &str = "__playgta6_indirect_call";
/// Runtime trap for instructions the lifter does not support: `void (ptr state, i64 va)`.
pub const HELPER_UNIMPLEMENTED: &str = "__playgta6_unimplemented";
/// Runtime trap for the x86 divide error (`#DE`): `void (ptr state, i64 va)`.
pub const HELPER_DIVIDE_ERROR: &str = "__playgta6_divide_error";
/// Runtime trap for guest traps, invalid code and non-returning calls: `void (ptr state, i64 va)`.
pub const HELPER_TRAP: &str = "__playgta6_trap";

/// Symbol name of the lifted function for the guest address `va` (e.g. `sub_140001000`).
pub fn function_symbol(va: u64) -> String {
    format!("sub_{va:X}")
}

/// An instruction that was replaced by a call to [`HELPER_UNIMPLEMENTED`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedInstruction {
    /// Guest virtual address of the instruction.
    pub address: u64,
    /// Lower-case mnemonic (e.g. `cpuid`), for grouping.
    pub mnemonic: String,
    /// Instruction text (Intel syntax).
    pub instruction: String,
    /// Why it could not be lifted.
    pub reason: String,
}

/// A malformed input function that cannot be lifted at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiftError {
    /// The function's entry address has no basic block.
    MissingEntryBlock { entry: u64 },
    /// A basic block has no instructions but is not marked `Terminator::Invalid`.
    EmptyBlock { block: u64 },
    /// A block's key, `start` and first instruction address disagree.
    BlockStartMismatch { block: u64, first_instruction: u64 },
    /// A `jmp`/`jcc`/`ret` appears before the last instruction of a block.
    InstructionAfterTerminator { block: u64, address: u64 },
    /// An intra-function successor edge targets an address with no block in the function.
    UnknownSuccessor { block: u64, target: u64 },
    /// Two functions with the same entry address were passed to [`lift_module`].
    DuplicateFunction { entry: u64 },
}

impl fmt::Display for LiftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEntryBlock { entry } => {
                write!(f, "function {entry:#x} has no basic block at its entry")
            }
            Self::EmptyBlock { block } => write!(
                f,
                "basic block {block:#x} has no instructions and is not marked invalid"
            ),
            Self::BlockStartMismatch {
                block,
                first_instruction,
            } => write!(
                f,
                "basic block {block:#x} starts with an instruction at {first_instruction:#x}"
            ),
            Self::InstructionAfterTerminator { block, address } => write!(
                f,
                "basic block {block:#x} continues after a control transfer (instruction at {address:#x})"
            ),
            Self::UnknownSuccessor { block, target } => write!(
                f,
                "basic block {block:#x} has a successor {target:#x} that is not a block of the function"
            ),
            Self::DuplicateFunction { entry } => {
                write!(f, "function {entry:#x} was passed more than once")
            }
        }
    }
}

impl std::error::Error for LiftError {}

/// One lifted function: its IR definition plus what the module emitter needs to know about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftedFunction {
    /// Guest entry address.
    pub entry: u64,
    /// LLVM symbol (see [`function_symbol`]).
    pub symbol: String,
    /// The `define void @sub_...` text.
    pub ir: String,
    /// Guest addresses this function calls or tail-calls directly.
    pub callees: BTreeSet<u64>,
    /// Instructions replaced by [`HELPER_UNIMPLEMENTED`] traps, in address order.
    pub unsupported: Vec<UnsupportedInstruction>,
    support: BTreeSet<Support>,
}

/// Module-level declarations or internal helper definitions a lifted function relies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Support {
    Ctpop8,
    Fshl(u32),
    Fshr(u32),
    Umul128,
    Udiv128,
    Sdiv128,
}

/// A complete lifted module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftedModule {
    /// The `.ll` text.
    pub ir: String,
    /// Per-function results, sorted by entry address.
    pub functions: Vec<LiftedFunction>,
}

/// Lifts a single function. The result is a fragment; use [`emit_module`] to get a full module.
///
/// Reads `Function::entry` and `Function::blocks`; per block `start`, `end`, `instructions`,
/// `terminator`, `successors` and, for direct calls into unmapped or non-executable memory,
/// `calls` plus `diagnostics`. Blocks shared by several functions are lifted into each of them.
pub fn lift_function(func: &cfg::Function) -> Result<LiftedFunction, LiftError> {
    function::lift(func)
}

/// Lifts a set of functions into one deterministic module (see [`emit_module`]).
pub fn lift_module<'a>(
    functions: impl IntoIterator<Item = &'a cfg::Function>,
) -> Result<LiftedModule, LiftError> {
    let mut lifted = Vec::new();
    for func in functions {
        lifted.push(lift_function(func)?);
    }
    lifted.sort_by_key(|f| f.entry);
    let ir = emit_module(&lifted)?;
    Ok(LiftedModule {
        ir,
        functions: lifted,
    })
}

/// Emits a complete `.ll` module: target info, guest state type, runtime helper declarations,
/// declarations for called functions that are not part of the module, required intrinsics and
/// internal helpers, then all function definitions sorted by entry address.
///
/// The output depends only on the set of functions (not their order): same input, same bytes.
pub fn emit_module(functions: &[LiftedFunction]) -> Result<String, LiftError> {
    let mut sorted: Vec<&LiftedFunction> = functions.iter().collect();
    sorted.sort_by_key(|f| f.entry);
    let mut defined = BTreeSet::new();
    for func in &sorted {
        if !defined.insert(func.entry) {
            return Err(LiftError::DuplicateFunction { entry: func.entry });
        }
    }

    let mut external = BTreeSet::new();
    let mut support = BTreeSet::new();
    for func in &sorted {
        external.extend(func.callees.difference(&defined).copied());
        support.extend(func.support.iter().copied());
    }
    if support.contains(&Support::Sdiv128) {
        support.insert(Support::Udiv128);
    }

    let mut out = String::new();
    out.push_str("; playgta6 recomp: x86-64 guest code lifted to LLVM IR (generated file)\n");
    let _ = writeln!(out, "target datalayout = \"{WASM64_DATALAYOUT}\"");
    let _ = writeln!(out, "target triple = \"{WASM64_TRIPLE}\"");
    out.push('\n');
    let _ = writeln!(
        out,
        "; gpr[16], rip, flags[CF PF AF ZF SF OF DF pad], fs_base, gs_base, xmm[16]"
    );
    let _ = writeln!(
        out,
        "{GUEST_STATE_TYPE} = type {{ [16 x i64], i64, [8 x i8], i64, i64, [16 x [2 x i64]] }}"
    );
    out.push('\n');
    let _ = writeln!(out, "declare void @{HELPER_INDIRECT_CALL}(ptr, i64)");
    let _ = writeln!(
        out,
        "declare void @{HELPER_UNIMPLEMENTED}(ptr, i64) noreturn"
    );
    let _ = writeln!(
        out,
        "declare void @{HELPER_DIVIDE_ERROR}(ptr, i64) noreturn"
    );
    let _ = writeln!(out, "declare void @{HELPER_TRAP}(ptr, i64) noreturn");
    for va in &external {
        let _ = writeln!(out, "declare void @{}(ptr)", function_symbol(*va));
    }
    for item in &support {
        match item {
            Support::Ctpop8 => out.push_str("declare i8 @llvm.ctpop.i8(i8)\n"),
            Support::Fshl(bits) => {
                let _ = writeln!(
                    out,
                    "declare i{bits} @llvm.fshl.i{bits}(i{bits}, i{bits}, i{bits})"
                );
            }
            Support::Fshr(bits) => {
                let _ = writeln!(
                    out,
                    "declare i{bits} @llvm.fshr.i{bits}(i{bits}, i{bits}, i{bits})"
                );
            }
            Support::Umul128 | Support::Udiv128 | Support::Sdiv128 => {}
        }
    }
    for item in &support {
        let text = match item {
            Support::Umul128 => function::UMUL128_IR,
            Support::Udiv128 => function::UDIV128_IR,
            Support::Sdiv128 => function::SDIV128_IR,
            _ => continue,
        };
        out.push('\n');
        out.push_str(text);
    }
    for func in &sorted {
        out.push('\n');
        out.push_str(&func.ir);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
