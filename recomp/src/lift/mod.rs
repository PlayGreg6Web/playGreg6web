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
//! | [`STATE_XMM_OFFSET`] (160) | 16 x 16 | `xmm0..xmm15` (bits 0..127 of `ymm0..15`) |
//! | [`STATE_YMM_HI_OFFSET`] (416) | 16 x 16 | bits 128..255 of `ymm0..15` |
//! | [`STATE_MXCSR_OFFSET`] (672) | 4 | MXCSR (the runtime initialises it to [`MXCSR_DEFAULT`]) |
//! | 676 | 12 | reserved (zero) |
//! | [`STATE_RESUME_RIP_OFFSET`] (688) | 8 | `rip` a pending context restore continues at |
//! | [`STATE_UNWIND_PENDING_OFFSET`] (696) | 1 | 1 while a context restore propagates (exception unwind, `longjmp`) |
//! | 697 | 7 | reserved (zero) |
//!
//! The layout is append-only (existing offsets never move) and mirrored by
//! `runtime/abi/playgta6_abi.h` / `docs/wasm_abi.md`.
//!
//! Inside a function, every GPR the function touches and every flag lives in an `alloca` that is
//! loaded from the state at entry and after each call, and stored back before each call and every
//! exit. LLVM's `mem2reg`/`SROA` turn these into SSA values; flag slots that are never read are
//! dead-store-eliminated. `%state` is `noalias`: guest code never addresses the state block itself.
//! XMM registers are modelled the same way (one `alloca <2 x i64>` per register the function
//! uses); see `lift/simd/mod.rs` for SSE/AVX semantics, the supported
//! instruction set and the documented deviations (NaN payloads, FTZ/DAZ, MXCSR, `rcp`/`rsqrt`).
//!
//! # Target features
//! Every lifted function carries [`LIFTED_FUNCTION_ATTRIBUTES`] (`+simd128`): SSE code is lifted
//! to plain LLVM vector IR and lowers to WebAssembly SIMD (`v128`) instructions.
//!
//! Sub-registers follow x86-64 rules: 32-bit writes zero-extend to 64 bits, 8/16-bit writes
//! preserve the untouched bits, and `AH/CH/DH/BH` address bits 8..16.
//!
//! # Flags
//! `CF PF AF ZF SF OF` are individual `i1` slots computed exactly for every supported instruction.
//! Flags the architecture leaves *undefined* for an instruction keep their previous value. `DF`
//! has no slot: it is read from and written to the state byte directly (`cld`/`std`/`popfq`
//! write it; string instructions and `pushfq` read it). `pushfq` reports bit 1 and IF set and
//! TF/AC/ID/system flags clear; `popfq` loads the six arithmetic flags and DF and ignores the
//! rest (no single-stepping or alignment checks).
//!
//! # Supported instructions
//! Moves and extensions, `lea`, the ALU (`add adc sub sbb cmp and or xor test inc dec neg not`),
//! shifts and rotates (`shl shr sar rol ror rcl rcr shld shrd`, BMI2 `shlx shrx sarx rorx`),
//! multiply/divide (`mul imul div idiv mulx`), `setcc`/`cmovcc`, `cbw`..`cqo`, `push`/`pop`,
//! `pushfq`/`popfq`, `lahf`/`sahf`, `clc stc cmc cld std`, `enter` (nesting level 0) / `leave`,
//! `xchg`, `xadd`, `cmpxchg`, `cmpxchg8b`, `cmpxchg16b`, every `lock` form (see Atomics), bit
//! instructions (`bt bts btr btc bsf bsr tzcnt lzcnt popcnt bswap`, BMI1 `andn blsi blsr blsmsk
//! bextr`, BMI2 `bzhi pdep pext`), string instructions (`movs stos lods cmps scas` with
//! `rep`/`repe`/`repne`, both DF directions; see `string_ops`), `movbe`, `xlat`, `adcx`/`adox`,
//! `crc32` (CRC-32C), AMD `clzero`, the runtime-provided `cpuid rdtsc rdtscp xgetbv`, fences
//! (`mfence lfence sfence` -> `fence seq_cst`), and no-op hints (`nop` forms, `pause`,
//! `endbr64`, `prefetch*`, `clflush*`, `clwb`, `cldemote`, `monitorx`, `mwaitx`). Control flow:
//! `jmp`, `jcc`, `loop`/`loope`/`loopne`, `jrcxz`/`jecxz`, `call`, `ret`.
//!
//! Choices for architecturally undefined results: `bsf`/`bsr` with a zero source leave the whole
//! destination register unchanged; `bswap r16` writes 0; for `cmpxchg` with a register
//! destination the destination is not written on failure and the accumulator not on success
//! (both as measured on hardware); `shld`/`shrd r16` with a count of 17..31 shift the 32-bit
//! concatenation; OF after multi-bit shifts/rotates uses the count-1 formula.
//!
//! # Atomics
//! Every `lock`-prefixed instruction and `xchg` with memory lowers to an LLVM `atomicrmw` or
//! `cmpxchg` (a `cmpxchg` loop for `lock neg`) with `seq_cst` ordering and natural alignment, and
//! the flags are computed from the old value; fences become `fence seq_cst`. wasm atomics trap on
//! misaligned addresses, so a misaligned locked access traps instead of being split. `lock
//! cmpxchg16b` has no 16-byte wasm atomic and runs under a spinlock from a table shared by all
//! linked modules ([`CMPXCHG16B_LOCKS`]): it is atomic against every other `cmpxchg16b`, its
//! halves are accessed with 8-byte atomics, but it is not atomic against plain or 8-byte locked
//! stores to the same 16 bytes made concurrently. A module with atomics appends
//! [`ATOMICS_TARGET_FEATURES`] to the `#0` target features of every definition and must be
//! linked with `--shared-memory` ([`LiftedModule::requires_shared_memory`]); modules without
//! atomics keep exactly [`LIFTED_FUNCTION_ATTRIBUTES`] (a module using the bulk `rep movs`/`rep
//! stos` paths appends [`BULK_MEMORY_TARGET_FEATURES`] so `memmove`/`memset` lower to
//! `memory.copy`/`memory.fill` instead of libcalls).
//! Plain `mov` loads and stores stay non-atomic wasm accesses: code that relies on x86-TSO
//! ordering of *plain* accesses (e.g. releasing a lock with a plain store) is not guaranteed to
//! be ordered on weakly ordered hosts.
//!
//! # Guest memory
//! Guest virtual addresses are 64-bit. Every guest load/store/atomic/bulk access turns its guest
//! address into a host pointer in exactly one place, `FunctionLifter::guest_ptr`, with the shared
//! translation of `docs/wasm_abi.md` "Guest Address Space":
//!
//! ```text
//! host = guest VA + base        (64-bit wrapping add, then inttoptr)
//! ```
//!
//! where `base` is chosen per module by [`GuestBase`]:
//! - [`GuestBase::Symbol`] (default, correct for every runtime configuration): `base` is loaded
//!   from the runtime-defined `i64` global [`GUEST_BASE_SYMBOL`] (`@__playgta6_guest_base`). The
//!   load is tagged `!invariant.load` (the runtime sets the base before guest code runs and never
//!   changes it while guest code runs), so `-O2` keeps one load per function and hoists it out of
//!   loops; each access costs one `i64.add`.
//! - [`GuestBase::Constant`]: `base` is a lift-time constant. `Constant(0)` is the identity
//!   mapping (guest VA == wasm64 linear-memory address), which is what the browser runtime uses:
//!   no translation code at all. A constant-base module must only run with a runtime whose
//!   `__playgta6_guest_base` equals the constant.
//!
//! Rejected alternatives: a symbol whose *address* is the base (folds into the wasm load offset,
//! but the offset is unsigned, so it cannot express the non-zero bases native hosts need, and
//! wasm-ld cannot place a symbol at an arbitrary address); a mutable global read without
//! `!invariant.load` (every guest store could alias it, forcing a reload per access).
//!
//! The `GuestState` pointer is a host pointer and is never translated. Accesses are `align 1`
//! (x86 allows unaligned access). All addressing modes are supported (`base + index*scale +
//! disp`, RIP-relative, 32-bit address size, `fs:`/`gs:`).
//!
//! # Calls, returns and the guest stack
//! `rsp` lives in the guest state and points into guest memory, so the guest stack is laid out
//! exactly like on hardware (return addresses, Win64 shadow space, stack arguments, saved
//! registers):
//! - direct `call T`: push the return address (`next_ip`) onto the guest stack, then
//!   `call void @sub_T(ptr %state)`. Registers are synced around the call.
//! - `ret [imm16]`: pop the return address (its value is not checked) plus `imm16` bytes, then
//!   LLVM `ret void`. The host call stack and the guest stack therefore stay in lockstep.
//! - after every call that can run guest code, the resume protocol (`unwind.rs`) tests
//!   [`STATE_UNWIND_PENDING_OFFSET`]: while a context restore (exception unwind, `longjmp`) is
//!   pending, activations return without writing their registers back until the one owning the
//!   restored `rsp` resumes at the block for [`STATE_RESUME_RIP_OFFSET`]
//!   ([`cfg::Function::resume_targets`]).
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
//!   memory at `va`, the instruction after a call that was assumed never to return, or a
//!   `cmpxchg16b` at `va` whose operand is not 16-byte aligned (#GP). Called with the resume
//!   flag still set and `va` = the resume `rip`, it reports a context restore to a block that is
//!   not a resume target of the owning function.
//!
//! Instructions whose results the runtime chooses call a helper that is declared only by modules
//! using it. The lifted code stores all its registers and flags to the state before the call and
//! reloads them after it; the helper reads and writes the state's GPRs and must leave every other
//! part of the state (flags included) unchanged. 32-bit results are stored zero-extended.
//! - `void __playgta6_cpuid(ptr state)` ([`HELPER_CPUID`]): leaf `EAX`, sub-leaf `ECX`; writes
//!   `EAX EBX ECX EDX`. The reported features must not claim more than the lifter supports.
//! - `void __playgta6_rdtsc(ptr state)` ([`HELPER_RDTSC`]): writes the 64-bit time-stamp counter
//!   to `EDX:EAX` (monotonic per thread).
//! - `void __playgta6_rdtscp(ptr state)` ([`HELPER_RDTSCP`]): as `rdtsc`, plus `ECX` =
//!   `IA32_TSC_AUX` (processor id).
//! - `void __playgta6_xgetbv(ptr state)` ([`HELPER_XGETBV`]): writes `XCR[ECX]` to `EDX:EAX`; for
//!   an `ECX` without an XCR it raises #GP (does not return).
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

/// Attributes of every lifted function (`#0`): lifted SSE code is plain LLVM vector IR, and
/// WebAssembly SIMD (`simd128`) lets it lower to `v128` instructions instead of being
/// scalarised. Consumers must run the module on an engine with wasm SIMD (every engine with
/// memory64 has it).
pub const LIFTED_FUNCTION_ATTRIBUTES: &str = "\"target-features\"=\"+simd128\"";

/// [`LIFTED_FUNCTION_ATTRIBUTES`] with extra target features appended.
fn module_attributes(extra: &str) -> String {
    let base = LIFTED_FUNCTION_ATTRIBUTES.trim_end_matches('"');
    format!("{base},{extra}\"")
}

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
/// Byte offset of the upper halves (bits 128..255) of `ymm0..ymm15` in the guest state.
pub const STATE_YMM_HI_OFFSET: u64 = 416;
/// Distance between two YMM upper halves in the guest state.
pub const STATE_YMM_HI_STRIDE: u64 = 16;
/// Byte offset of the 32-bit MXCSR in the guest state.
pub const STATE_MXCSR_OFFSET: u64 = 672;
/// Byte offset of the `rip` a pending context restore continues at.
pub const STATE_RESUME_RIP_OFFSET: u64 = 688;
/// Byte offset of the byte that is 1 while a context restore propagates down the host stack.
pub const STATE_UNWIND_PENDING_OFFSET: u64 = 696;
/// Total size of the guest state in bytes.
pub const STATE_SIZE: u64 = 704;
/// MXCSR value of a new guest thread (round to nearest, all exceptions masked, FTZ/DAZ off).
pub const MXCSR_DEFAULT: u32 = 0x1F80;

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
/// Runtime `cpuid`: `void (ptr state)` (see the module docs for the contract).
pub const HELPER_CPUID: &str = "__playgta6_cpuid";
/// Runtime `rdtsc`: `void (ptr state)` (see the module docs for the contract).
pub const HELPER_RDTSC: &str = "__playgta6_rdtsc";
/// Runtime `rdtscp`: `void (ptr state)` (see the module docs for the contract).
pub const HELPER_RDTSCP: &str = "__playgta6_rdtscp";
/// Runtime `xgetbv`: `void (ptr state)` (see the module docs for the contract).
pub const HELPER_XGETBV: &str = "__playgta6_xgetbv";

/// Target features of a module that contains atomic instructions (`lock`, `xchg [mem]`,
/// fences): real wasm atomics on a shared memory. Such a module must be linked with
/// `--shared-memory` (wasm-ld rejects it otherwise). These features are appended to
/// [`LIFTED_FUNCTION_ATTRIBUTES`] only for such modules; other modules compile unchanged.
pub const ATOMICS_TARGET_FEATURES: &str = "+atomics,+bulk-memory";
/// Target features of a module that uses `llvm.memmove`/`llvm.memset` (fast `rep movs`/`rep stos`)
/// but no atomics: `memory.copy`/`memory.fill` instead of `memmove`/`memset` libcalls.
pub const BULK_MEMORY_TARGET_FEATURES: &str = "+bulk-memory";
/// Shared lock table of the `cmpxchg16b` fallback (one definition across all linked modules).
pub const CMPXCHG16B_LOCKS: &str = "__playgta6_cmpxchg16b_locks";
/// Runtime-defined `i64` global holding the guest -> host translation base
/// (`host = __playgta6_guest_base + guest VA`, see `docs/wasm_abi.md` "Guest Address Space").
pub const GUEST_BASE_SYMBOL: &str = "__playgta6_guest_base";

/// How lifted code translates guest addresses (see "Guest memory" in the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GuestBase {
    /// `host = VA + @__playgta6_guest_base` (runtime-defined, loaded once per function).
    #[default]
    Symbol,
    /// `host = VA + constant`; `Constant(0)` is the identity mapping (no translation code).
    Constant(u64),
}

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

impl LiftedFunction {
    /// Whether the function contains atomic instructions (its module needs a shared memory).
    pub fn uses_atomics(&self) -> bool {
        self.support.contains(&Support::Atomics)
    }
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
    /// An intrinsic used by lifted SSE/AVX code.
    Simd(function::simd::Decl),
    /// `llvm.{cttz,ctlz,ctpop,bswap}.iN` (never `Ctpop` at 8 bits: that is [`Support::Ctpop8`]).
    BitIntrinsic(BitIntrinsic, u32),
    /// `llvm.memmove` (needs `+bulk-memory`, otherwise wasm lowers it to a `memmove` libcall).
    Memmove,
    /// `llvm.memset` (needs `+bulk-memory`, otherwise wasm lowers it to a `memset` libcall).
    Memset,
    /// Atomic instructions or fences: the module is compiled with [`ATOMICS_TARGET_FEATURES`].
    Atomics,
    /// The internal `cmpxchg16b` helper and the shared lock table it uses.
    Cmpxchg16b,
    Pdep,
    Pext,
    Crc32c,
    Helper(RuntimeHelper),
    /// The `@__playgta6_guest_base` declaration ([`GuestBase::Symbol`]).
    GuestBase,
}

/// LLVM bit-manipulation intrinsics used by the lifter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum BitIntrinsic {
    Cttz,
    Ctlz,
    Ctpop,
    Bswap,
}

/// Optional runtime helpers, declared only by modules that use them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RuntimeHelper {
    Cpuid,
    Rdtsc,
    Rdtscp,
    Xgetbv,
}

impl RuntimeHelper {
    fn symbol(self) -> &'static str {
        match self {
            Self::Cpuid => HELPER_CPUID,
            Self::Rdtsc => HELPER_RDTSC,
            Self::Rdtscp => HELPER_RDTSCP,
            Self::Xgetbv => HELPER_XGETBV,
        }
    }
}

/// A complete lifted module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftedModule {
    /// The `.ll` text.
    pub ir: String,
    /// Per-function results, sorted by entry address.
    pub functions: Vec<LiftedFunction>,
}

impl LiftedModule {
    /// Whether the module is compiled with [`ATOMICS_TARGET_FEATURES`] and must therefore be
    /// linked against a shared memory (`wasm-ld --shared-memory`).
    pub fn requires_shared_memory(&self) -> bool {
        self.functions.iter().any(LiftedFunction::uses_atomics)
    }
}

/// Lifts a single function. The result is a fragment; use [`emit_module`] to get a full module.
///
/// Reads `Function::entry` and `Function::blocks`; per block `start`, `end`, `instructions`,
/// `terminator`, `successors` and, for direct calls into unmapped or non-executable memory,
/// `calls` plus `diagnostics`. Blocks shared by several functions are lifted into each of them.
pub fn lift_function(func: &cfg::Function) -> Result<LiftedFunction, LiftError> {
    lift_function_with(func, GuestBase::default())
}

/// [`lift_function`] with an explicit guest address translation.
pub fn lift_function_with(
    func: &cfg::Function,
    guest_base: GuestBase,
) -> Result<LiftedFunction, LiftError> {
    function::lift(func, guest_base)
}

/// Lifts a set of functions into one deterministic module (see [`emit_module`]).
pub fn lift_module<'a>(
    functions: impl IntoIterator<Item = &'a cfg::Function>,
) -> Result<LiftedModule, LiftError> {
    lift_module_with(functions, GuestBase::default())
}

/// [`lift_module`] with an explicit guest address translation.
pub fn lift_module_with<'a>(
    functions: impl IntoIterator<Item = &'a cfg::Function>,
    guest_base: GuestBase,
) -> Result<LiftedModule, LiftError> {
    let mut lifted = Vec::new();
    for func in functions {
        lifted.push(lift_function_with(func, guest_base)?);
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
        "; gpr[16], rip, flags[CF PF AF ZF SF OF DF pad], fs_base, gs_base, xmm[16], ymm_hi[16], mxcsr, reserved, resume_rip, unwind_pending, reserved"
    );
    let _ = writeln!(
        out,
        "{GUEST_STATE_TYPE} = type {{ [16 x i64], i64, [8 x i8], i64, i64, [16 x [2 x i64]], [16 x [2 x i64]], i32, [3 x i32], i64, i8, [7 x i8] }}"
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
            Support::Simd(decl) => out.push_str(&decl.declaration()),
            Support::BitIntrinsic(kind, bits) => {
                let (name, flag) = match kind {
                    BitIntrinsic::Cttz => ("cttz", ", i1"),
                    BitIntrinsic::Ctlz => ("ctlz", ", i1"),
                    BitIntrinsic::Ctpop => ("ctpop", ""),
                    BitIntrinsic::Bswap => ("bswap", ""),
                };
                let _ = writeln!(out, "declare i{bits} @llvm.{name}.i{bits}(i{bits}{flag})");
            }
            Support::Memmove => {
                out.push_str("declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1)\n");
            }
            Support::Memset => out.push_str("declare void @llvm.memset.p0.i64(ptr, i8, i64, i1)\n"),
            Support::Helper(helper) => {
                let _ = writeln!(out, "declare void @{}(ptr)", helper.symbol());
            }
            Support::GuestBase => {
                let _ = writeln!(out, "@{GUEST_BASE_SYMBOL} = external global i64, align 8");
            }
            Support::Umul128
            | Support::Udiv128
            | Support::Sdiv128
            | Support::Atomics
            | Support::Cmpxchg16b
            | Support::Pdep
            | Support::Pext
            | Support::Crc32c => {}
        }
    }
    for item in &support {
        let text = match item {
            Support::Umul128 => function::UMUL128_IR,
            Support::Udiv128 => function::UDIV128_IR,
            Support::Sdiv128 => function::SDIV128_IR,
            Support::Cmpxchg16b => function::CMPXCHG16B_IR,
            Support::Pdep => function::PDEP_IR,
            Support::Pext => function::PEXT_IR,
            Support::Crc32c => function::CRC32C_IR,
            _ => continue,
        };
        out.push('\n');
        out.push_str(text);
    }
    for func in &sorted {
        out.push('\n');
        out.push_str(&func.ir);
    }
    // Target features are a function attribute in LLVM IR. Every definition gets `#0`: the
    // lifted functions already name it, internal helpers are tagged here (the wasm backend unifies
    // the features per module anyway). Atomics and bulk memory are added only when used.
    let extra = if support.contains(&Support::Atomics) {
        Some(ATOMICS_TARGET_FEATURES)
    } else if support.contains(&Support::Memmove) || support.contains(&Support::Memset) {
        Some(BULK_MEMORY_TARGET_FEATURES)
    } else {
        None
    };
    let attributes = match extra {
        None => LIFTED_FUNCTION_ATTRIBUTES.to_string(),
        Some(extra) => {
            let mut tagged = String::with_capacity(out.len() + 64);
            for line in out.lines() {
                tagged.push_str(line);
                if line.starts_with("define ") && line.ends_with(") {") {
                    tagged.truncate(tagged.len() - 2);
                    tagged.push_str(" #0 {");
                }
                tagged.push('\n');
            }
            out = tagged;
            module_attributes(extra)
        }
    };
    out.push('\n');
    let _ = writeln!(out, "attributes #0 = {{ {attributes} }}");
    Ok(out)
}

#[cfg(test)]
mod tests;
