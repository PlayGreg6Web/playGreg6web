//! Guest memory models (task R19, prototype): how lifted code turns a guest virtual address into
//! a host pointer. See "Guest memory" in the parent module for the default ([`GuestBase`]).
//!
//! # Why a second translation
//! The title reserves a 16 GiB CPU window (24 GiB on another console profile) next to a 3.88 GiB
//! physical page pool and maps pool pages at arbitrary 64 KiB window slots
//! (`runtime/os/xmem_windows.h`). The browser's identity-mapped guest pool is 12 GiB, so the
//! window cannot be backed 1:1. Two prototypes translate through a table of per-64 KiB deltas
//! instead (`host = va + table[slot]`); a slot's delta points a whole window page at the pool page
//! mapped there, so the window costs table entries, not address space.
//!
//! # Models
//! - [`MemoryModel::Base`]: today's translation, `host = va + base` ([`GuestBase`]). Default; its
//!   output is byte for byte what the lifter emitted before R19.
//! - [`MemoryModel::Windowed`]: one range `[W, W + S)` (the CPU window, placed above the
//!   identity-mapped guest pool) translates through `@__playgta6_window_table`
//!   (`S / 64 KiB` deltas, an `i64` array in runtime memory); every other address uses the
//!   [`GuestBase`] translation. Per access: `sub` + unsigned compare + a branch that is not taken
//!   outside the window; inside, a shift, one table load and an add.
//! - [`MemoryModel::PageTable`]: every address translates through `@__playgta6_page_table`
//!   (`pages + 1` deltas covering guest VAs `[0, pages * 64 KiB)`; the index is clamped to
//!   `pages`, whose entry is the trap entry). No range branch, one table load per access (not
//!   hoistable), and the runtime gets null-page / uncommitted-page traps for free.
//!
//! # Rules shared by both table models
//! - Table entries change at run time (the title maps and unmaps pages), so entry loads are
//!   ordinary loads: never `!invariant.load`. LLVM may merge two loads of the same entry only when
//!   nothing in between may write memory; any call (every import, every `__playgta6_*` helper) and
//!   any guest store forces a reload. The table *address* is a link-time constant (the wasm load
//!   folds it into its offset); `W`, `S` and `pages` are lift-time constants.
//! - Unmapped slots hold a delta that moves the address to `>= 2^63`
//!   (`guest_window::kTrapDelta` in the runtime): beyond any wasm64 memory, so the access traps with
//!   "memory access out of bounds" exactly like a page fault would, at no extra cost; natively the
//!   address is non-canonical and faults. A poison page (reads zero, writes ignored) was rejected:
//!   it turns guest bugs into silent corruption.
//! - An access of `n` bytes whose offset in its 64 KiB slot is above `65536 - n` may straddle two
//!   slots backed by non-adjacent host pages. It takes a slow path: loads call
//!   `__playgta6_guest_read(va, n)` (returns a per-thread scratch buffer holding the bytes, copied
//!   segment by segment through the same translation), stores write the scratch buffer from
//!   `__playgta6_guest_scratch()` and then call `__playgta6_guest_write(va, n)`. The check is one
//!   `and` + compare + not-taken branch on the window path only ([`MemoryModel::Windowed`]) or on
//!   every access ([`MemoryModel::PageTable`]).
//! - Naturally aligned accesses (atomics; `lock` operands must be aligned, wasm traps otherwise)
//!   never straddle and skip the check.
//! - `rep movs` / `rep stos` bulk paths: a range that touches the window (or any range under
//!   [`MemoryModel::PageTable`]) calls `__playgta6_guest_memmove` / `__playgta6_guest_memset`
//!   (segmented); a range outside the window keeps `llvm.memmove` / `llvm.memset`.
//!
//! # Options (all measured, see below)
//! - [`WindowCheck`]: `Branch` (range branch, inline window path), `Select` (branch-free: always
//!   one table load, rejected: slowest and largest), `Outline` (window path in a `noinline` helper
//!   per width: smallest code, but no faster).
//! - [`Window::stack_flat`] / `PageTable::stack`: `rsp`-relative accesses (`[rsp + d]`, `push`,
//!   `pop`, `call`) skip the table. Sound while guest stacks never lie in the window (the runtime
//!   allocates them in the arena). Most of the measured overhead of "typical" code was stack
//!   traffic; this removes it.
//! - [`Window::contiguous_runs`]: no straddle check or slow path at all; sound only if the runtime
//!   keeps every two adjacent *mapped* slots on adjacent host pages (a window-centric page
//!   allocator: the title never dereferences its pool view, O27, so a page's bytes may live where
//!   the window needs them; merging runs costs a copy). The R19 runtime prototype does not do
//!   this yet (it aliases pool-view pages, which are adjacent only when the title maps consecutive
//!   pool pages at consecutive slots).
//!
//! # What LLVM makes of it
//! - The guest base (`!invariant.load`) is loaded once per function; `W`, `S`, `pages` are
//!   immediates; table entry loads stay in place (reloaded after every call and guest store, as
//!   required).
//! - Every access becomes 2-3 basic blocks plus a (cold) call: loops are no longer unrolled (the
//!   base model unrolls the array-sum loop 4x), nothing is vectorised in any model, and small
//!   leaf functions are no longer inlined into their callers (inline cost) unless the stack and
//!   straddle code is removed (`stack_flat`, `contiguous_runs`).
//! - The `rep movs` fast path stays `llvm.memmove` for flat ranges under `Windowed`; under
//!   `PageTable` every bulk copy is a runtime call.
//!
//! # Measurements (R19, 2026-11; node 23 / V8 on x86-64, Windows 11)
//! wasm64 object size of 3,562 real functions (every 100th of the title), relative to the identity
//! base model (8.68 MB): `Symbol` 1.03, `Windowed` 2.18, `+stack_flat` 1.56, `+contiguous_runs`
//! 1.36, `Outline+stack_flat` 1.27, `Select` 2.22, `PageTable` 1.97, `PageTable+stack+runs` 1.24.
//! Micro-benchmarks (ns per element, median of 21, data outside the window / inside it) for
//! identity -> `Windowed+stack_flat` -> `Windowed+stack_flat+runs` -> `PageTable+stack+runs`:
//! array sum 0.22 -> 0.42/0.69 -> 0.41/0.51 -> 0.60; pointer chase 3.3 -> 3.6/5.2 -> 3.6/5.2 ->
//! 5.5; copy loop 0.31 -> 1.01/1.38 -> 0.73/1.01 -> 1.19; "typical" structs + leaf call 3.0 ->
//! 3.7/4.5 -> 3.3/4.1 -> 3.8. (`tests/memory_bench.rs`; table in the R19 report.)
//!
//! # Verdict
//! The browser target can only hold the title's 16 GiB CPU window with a table model; of the
//! prototypes, `Windowed` with `stack_flat` (and `contiguous_runs` once the runtime keeps runs
//! contiguous) costs least on code that does not touch the window (+8..25% on typical code, 2-3x
//! on tight streaming loops), `PageTable` is uniformly slower but emits the least code and would
//! add null-page / uncommitted-page traps. Both stay behind this option until the project decides.

use super::GuestBase;

/// Size of one translation slot (the title's 64 KiB physical page).
pub const SLOT_BYTES: u64 = 64 << 10;
/// log2([`SLOT_BYTES`]).
pub const SLOT_SHIFT: u32 = 16;
/// Delta table of the window ([`MemoryModel::Windowed`]): `[S / 64 KiB x i64]`, runtime-defined.
pub const WINDOW_TABLE_SYMBOL: &str = "__playgta6_window_table";
/// Delta table of [`MemoryModel::PageTable`]: `[pages + 1 x i64]`, runtime-defined.
pub const PAGE_TABLE_SYMBOL: &str = "__playgta6_page_table";
/// `ptr (i64 va, i64 n)`: copies `n` guest bytes into a per-thread scratch buffer and returns it.
pub const HELPER_GUEST_READ: &str = "__playgta6_guest_read";
/// `ptr ()`: the per-thread scratch buffer a straddling store writes before `guest_write`.
pub const HELPER_GUEST_SCRATCH: &str = "__playgta6_guest_scratch";
/// `void (i64 va, i64 n)`: copies `n` bytes from the scratch buffer to guest memory.
pub const HELPER_GUEST_WRITE: &str = "__playgta6_guest_write";
/// `void (i64 dst, i64 src, i64 n)`: `memmove` between guest ranges, segment by segment.
pub const HELPER_GUEST_MEMMOVE: &str = "__playgta6_guest_memmove";
/// `void (i64 dst, i8 value, i64 n)`: `memset` of a guest range, segment by segment.
pub const HELPER_GUEST_MEMSET: &str = "__playgta6_guest_memset";
/// Largest single access the lifter emits (a 256-bit AVX operand).
pub const MAX_ACCESS_BYTES: u64 = 32;

/// The translated window of [`MemoryModel::Windowed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Addresses outside the window: today's translation.
    pub flat: GuestBase,
    /// First guest VA of the window (`W`, 64 KiB aligned).
    pub base: u64,
    /// Size of the window in bytes (`S`, a multiple of 64 KiB).
    pub size: u64,
    /// How the range check is emitted.
    pub check: WindowCheck,
    /// Translate stack-pointer-relative accesses (`[rsp + disp]`, `push`/`pop`, `call`) with
    /// `flat` and no window check. Sound while no guest stack lies inside the window (the
    /// runtime allocates guest stacks in the arena; a guest that points `rsp` into window memory
    /// itself would break this).
    pub stack_flat: bool,
    /// The runtime keeps every two adjacent *mapped* slots host-contiguous (see "Contiguous
    /// runs" in the module docs): accesses need no straddle check and no slow path.
    pub contiguous_runs: bool,
}

/// Code shape of the [`MemoryModel::Windowed`] range check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowCheck {
    /// `br (va - W) < S`: the flat path stays branch-only (no table load); the window path is a
    /// separate block with its own straddle check.
    #[default]
    Branch,
    /// Branch-free: every access loads a table entry (entry 0 outside the window) and selects
    /// between it and the flat base; one rarely taken branch for window accesses that leave their
    /// slot.
    Select,
    /// Like [`Self::Branch`], but the window path is one call to an internal `noinline` helper
    /// per access width (`@__playgta6_wload<n>_<W>` / `@__playgta6_wstore<n>_<W>`: table lookup,
    /// straddle check and slow path inside), which keeps lifted functions small.
    Outline,
}

/// How lifted code translates guest addresses (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryModel {
    /// `host = va + base` (the default; unchanged output).
    Base(GuestBase),
    /// One window range through a delta table, everything else as `flat`.
    Windowed(Window),
    /// Every address through a delta table of `pages` 64 KiB slots (plus one trap entry).
    PageTable {
        /// Number of slots covered (guest VAs `[0, pages * 64 KiB)`).
        pages: u64,
        /// `Some(base)`: stack-pointer-relative accesses use `host = va + base` instead of the
        /// table (same soundness condition as [`Window::stack_flat`]).
        stack: Option<GuestBase>,
        /// As [`Window::contiguous_runs`].
        contiguous_runs: bool,
    },
}

impl Default for MemoryModel {
    fn default() -> Self {
        Self::Base(GuestBase::default())
    }
}

/// Module-level declarations the table models need (collected like the other `Support` items).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Decl {
    WindowTable,
    PageTable,
    Read,
    Write,
    Memmove,
    Memset,
    /// Outlined window load / store of `bytes` bytes for the window at `base`
    /// ([`WindowCheck::Outline`]).
    WindowLoad {
        bytes: u8,
        base: u64,
    },
    WindowStore {
        bytes: u8,
        base: u64,
    },
}

/// Name of an outlined window access helper.
pub(crate) fn window_helper(store: bool, bytes: u64, base: u64) -> String {
    let kind = if store { "wstore" } else { "wload" };
    format!("__playgta6_{kind}{bytes}_{base:x}")
}

impl Decl {
    pub(crate) fn declaration(self) -> String {
        match self {
            Self::WindowTable => {
                "@__playgta6_window_table = external global [0 x i64], align 8\n".to_string()
            }
            Self::PageTable => {
                "@__playgta6_page_table = external global [0 x i64], align 8\n".to_string()
            }
            Self::Read => "declare ptr @__playgta6_guest_read(i64, i64)\n".to_string(),
            Self::Write => "declare ptr @__playgta6_guest_scratch()\n\
                            declare void @__playgta6_guest_write(i64, i64)\n"
                .to_string(),
            Self::Memmove => "declare void @__playgta6_guest_memmove(i64, i64, i64)\n".to_string(),
            Self::Memset => "declare void @__playgta6_guest_memset(i64, i8, i64)\n".to_string(),
            Self::WindowLoad { bytes, base } => outlined(false, u64::from(bytes), base),
            Self::WindowStore { bytes, base } => outlined(true, u64::from(bytes), base),
        }
    }
}

/// Definition of an outlined window access helper (see [`WindowCheck::Outline`]).
fn outlined(store: bool, bytes: u64, base: u64) -> String {
    let name = window_helper(store, bytes, base);
    let bits = bytes * 8;
    let sig = if store {
        format!("void @{name}(i64 %va, i{bits} %v)")
    } else {
        format!("i{bits} @{name}(i64 %va)")
    };
    let access = |ptr: &str, tag: &str| {
        if store {
            format!("  store i{bits} %v, ptr {ptr}, align 1\n  ret void\n")
        } else {
            format!("  %v{tag} = load i{bits}, ptr {ptr}, align 1\n  ret i{bits} %v{tag}\n")
        }
    };
    let mut out = format!(
        "define internal {sig} noinline {{\nentry:\n  %rel = sub i64 %va, {}\n  %idx = lshr i64 %rel, {SLOT_SHIFT}\n  %e = getelementptr i64, ptr @{WINDOW_TABLE_SYMBOL}, i64 %idx\n  %d = load i64, ptr %e, align 8\n",
        base as i64
    );
    if bytes > 1 {
        out.push_str(&format!(
            "  %off = and i64 %va, {}\n  %fit = icmp ule i64 %off, {}\n  br i1 %fit, label %fast, label %slow\nfast:\n",
            SLOT_BYTES - 1,
            SLOT_BYTES - bytes
        ));
    }
    out.push_str("  %h = add i64 %va, %d\n  %p = inttoptr i64 %h to ptr\n");
    out.push_str(&access("%p", "f"));
    if bytes > 1 {
        out.push_str("slow:\n");
        if store {
            out.push_str(&format!(
                "  %s = call ptr @{HELPER_GUEST_SCRATCH}()\n  store i{bits} %v, ptr %s, align 1\n  call void @{HELPER_GUEST_WRITE}(i64 %va, i64 {bytes})\n  ret void\n"
            ));
        } else {
            out.push_str(&format!(
                "  %s = call ptr @{HELPER_GUEST_READ}(i64 %va, i64 {bytes})\n{}",
                access("%s", "s")
            ));
        }
    }
    out.push_str("}\n");
    out
}

impl From<GuestBase> for MemoryModel {
    fn from(base: GuestBase) -> Self {
        Self::Base(base)
    }
}
