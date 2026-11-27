//! Guest memory accesses under every [`MemoryModel`] (the translation seam; models and rules:
//! `lift/memory.rs`). With [`MemoryModel::Base`] every function here emits exactly the IR the
//! lifter emitted before the table models existed.

use std::collections::BTreeMap;

use iced_x86::Instruction;

use super::{FunctionLifter, R, imm};
use crate::lift::memory::{
    Decl, HELPER_GUEST_MEMMOVE, HELPER_GUEST_MEMSET, HELPER_GUEST_READ, MemoryModel,
    PAGE_TABLE_SYMBOL, SLOT_BYTES, SLOT_SHIFT, WINDOW_DIRTY_SYMBOL, WINDOW_TABLE_SYMBOL, Window,
    WindowCheck, helper_value_type, slow_store_helper, window_helper,
};
use crate::lift::{GuestBase, Support};

/// Maximum cumulative displacement from `rsp` allowed when propagating stack-pointer provenance
/// through non-`rsp` base registers (e.g. frame pointers `mov rbp, rsp; mov [rbp - 0x20], rax`).
pub(super) const STACK_FRAME_MAX_DISP: i64 = 0x10_0000;

/// Half of [`STACK_FRAME_MAX_DISP`]: bound for merged stack-frame offsets across non-identical
/// incoming paths so subsequent `[reg + disp]` accesses with `|disp| <= STACK_FRAME_HALF` stay
/// provably within [`STACK_FRAME_MAX_DISP`] of `entry_rsp`.
pub(super) const STACK_FRAME_HALF: i64 = STACK_FRAME_MAX_DISP / 2;

/// Proven stack-pointer displacement tracked for an SSA address value in [`FunctionLifter::stack_values`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StackDisp {
    /// Exact displacement `d` from `rsp` (or `entry_rsp`).
    Exact(i64),
    /// Unknown displacement bounded by `|d| <= max_abs` (`0 <= max_abs <= STACK_FRAME_MAX_DISP`).
    Bounded(i64),
}

/// Maximum number of dynamic window translations cached simultaneously within one basic block.
const MAX_WINDOW_LOAD_CACHE: usize = 4;

/// Maximum span (in bytes) of a coalesced same-base window burst.
pub(super) const BURST_MAX_SPAN: u64 = 4096;

/// If `addr` is a lift-time constant integer literal (e.g. a RIP-relative global address from
/// `FunctionLifter::address`), returns its 64-bit guest VA.
pub(super) fn const_addr(addr: &str) -> Option<u64> {
    addr.parse::<i64>().ok().map(|v| v as u64)
}

/// Cached translation of a dynamic address from a prior `guest_load` in the same basic block
/// (`WindowCheck::Branch` with `contiguous_runs`): lets an immediate read-modify-write store
/// (e.g. `add [rcx], rax`) or repeated load reuse `delta`, `inside`, `rel`, and `ptr` without a
/// second window range check or table lookup. Cleared on block boundaries and any call/atomic/fence.
#[derive(Clone)]
pub(super) struct CachedWindowAddr {
    pub(super) addr: String,
    pub(super) inside: String,
    pub(super) rel: String,
    pub(super) ptr: String,
}

/// The last plain guest access (`guest_load`/`guest_store`): `value` is what guest memory holds at
/// `addr` (`bytes` wide, as `ty`) right after the IR emitted up to byte `at` of the body. A later
/// load of the same address takes `value` instead of reading memory again, provided nothing
/// emitted since then is a label or [`keeps_memory`] rejects it: the load is then in the same LLVM
/// block (so `value` dominates it) with no store, call, atomic or fence in between. On x86-TSO a
/// thread reads its own store back (store-to-load forwarding) and two reads with nothing between
/// them may see the same value, so this only picks one of the executions the guest allows; the
/// memory-model barriers (`LOOP_BARRIER`, `fence seq_cst`, #719/#722) end the forwarding.
pub(super) struct LastAccess {
    addr: String,
    ty: String,
    value: String,
    bytes: u64,
    at: usize,
}

/// Whether an emitted IR line leaves guest memory as it was and stays in the current LLVM block:
/// guest loads, arithmetic, intrinsics without memory effects and stores to the register/flag
/// allocas (`%rax`, `%cf`, ...) or the guest state (`%state.*`, never guest memory) qualify;
/// labels, terminators, calls, fences, atomics and every other store do not.
fn keeps_memory(line: &str) -> bool {
    const PURE_INTRINSICS: [&str; 12] = [
        "@llvm.ctpop.",
        "@llvm.ctlz.",
        "@llvm.cttz.",
        "@llvm.bswap.",
        "@llvm.fshl.",
        "@llvm.fshr.",
        "@llvm.abs.",
        "@llvm.smax.",
        "@llvm.smin.",
        "@llvm.umax.",
        "@llvm.umin.",
        ".with.overflow.",
    ];
    let Some(code) = line.strip_prefix("  ") else {
        return false;
    };
    let rhs = code.split_once(" = ").map_or(code, |(_, rhs)| rhs);
    let op = rhs.split(' ').next().unwrap_or_default();
    match op {
        "load" => !rhs.starts_with("load atomic") && !rhs.starts_with("load volatile"),
        "store" => {
            !rhs.starts_with("store atomic")
                && !rhs.starts_with("store volatile")
                && rhs.rsplit_once(", ptr %").is_some_and(|(_, ptr)| {
                    !ptr.starts_with('t') && !ptr.contains(',') && !ptr.contains(' ')
                })
        }
        "call" | "tail" => PURE_INTRINSICS.iter().any(|f| rhs.contains(f)),
        "fence" | "atomicrmw" | "cmpxchg" | "br" | "switch" | "ret" | "unreachable"
        | "indirectbr" | "invoke" | "resume" | "callbr" => false,
        _ => true,
    }
}

/// Addressing base for one entry in a coalesced window burst:
/// - `index == None`: plain `[base + disp]` (`base` is a 64-bit GPR index `0..16`).
/// - `index == Some((idx, scale))`: indexed `[base + idx*scale + disp]` (`base` and `idx` are
///   64-bit GPR indices `0..16`, `scale` is `1 | 2 | 4 | 8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct BurstKey {
    pub(super) base: usize,
    pub(super) index: Option<(usize, u32)>,
}

impl BurstKey {
    /// Bitmask of the 64-bit GPR(s) read by this addressing base (`base`, plus `idx` if indexed).
    pub(super) const fn gpr_mask(self) -> u16 {
        let mut mask = 1u16 << self.base;
        if let Some((idx, _)) = self.index {
            mask |= 1u16 << idx;
        }
        mask
    }
}

/// One base tracked inside a coalesced window burst (`[eff_base + min_d .. eff_base + max_end)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BurstBase {
    /// Plain `[base + disp]` or indexed `[base + idx*scale + disp]` key.
    pub(super) key: BurstKey,
    /// Minimum displacement relative to `eff_base`.
    pub(super) min_d: i64,
    /// Maximum `disp + access_bytes` relative to `eff_base`.
    pub(super) max_end: i64,
    /// Whether any instruction in the burst writes through this key.
    pub(super) has_write: bool,
}

/// Active per-base state while lowering a coalesced window burst.
#[derive(Clone)]
pub(super) struct BurstOverrideBase {
    /// Plain `[base + disp]` or indexed `[base + idx*scale + disp]` key.
    pub(super) key: BurstKey,
    /// Proven inclusive lower displacement relative to `eff_base`.
    pub(super) min_d: i64,
    /// Proven exclusive upper displacement relative to `eff_base`.
    pub(super) max_end: i64,
    /// Whether this base is checked and eligible for coalesced burst stores / dirty marking.
    pub(super) has_write: bool,
    /// Host pointer base (`eff_base + delta`) in fast mode; `None` in slow fallback mode.
    pub(super) base_host: Option<String>,
    /// Whether the fast path actually lowered at least one store through this base.
    pub(super) saw_store: bool,
    /// Guest address SSA values emitted by `address()` for `[eff_base + disp]` -> `disp`.
    pub(super) addrs: BTreeMap<String, i64>,
}

/// Maximum number of distinct unmodified base GPRs coalesced into a single window burst.
pub(super) const MAX_BURST_BASES: usize = 3;

/// Active override for a coalesced multi-base window burst (`[base_k + d_i]`, up to
/// [`MAX_BURST_BASES`] bases).
#[derive(Clone)]
pub(super) struct BurstOverride {
    /// True in the 99.95% hot path (flat or single 64 KiB slot per base), false in the cold
    /// boundary-crossing fallback (`__playgta6_guest_read` / `__playgta6_slow_storeN`).
    pub(super) fast: bool,
    /// Active base descriptors (1..=[`MAX_BURST_BASES`] entries).
    pub(super) bases: Vec<BurstOverrideBase>,
}

/// One coalesced multi-base access burst detected in a basic block's instruction list.
#[derive(Debug, Clone, Copy)]
pub(super) struct WindowBurst {
    /// First instruction index in `body` (inclusive).
    pub(super) start: usize,
    /// Last instruction index in `body` (inclusive).
    pub(super) end: usize,
    /// Active base descriptors (`bases[..base_count as usize]`).
    pub(super) bases: [BurstBase; MAX_BURST_BASES],
    /// Number of active base descriptors (`1..=MAX_BURST_BASES`).
    pub(super) base_count: u8,
}

impl WindowBurst {
    pub(super) fn active_bases(&self) -> &[BurstBase] {
        &self.bases[..usize::from(self.base_count)]
    }
}

/// `!prof` of a branch whose first successor is rarely taken.
pub(super) const UNLIKELY: &str = "!prof !{!\"branch_weights\", i32 1, i32 2000}";
/// `!prof` of a branch whose first successor is almost always taken.
pub(super) const LIKELY: &str = "!prof !{!\"branch_weights\", i32 2000, i32 1}";

/// One guest access: what to do with the translated address.
enum Access<'v> {
    Load { ty: &'v str },
    Store { ty: &'v str, value: &'v str },
}

/// O37: how a write through [`FunctionLifter::guest_ptr_write`] marks its window slot.
pub(super) enum WriteMark {
    /// No window (or a stack-pointer-relative access, which never reaches it).
    None,
    /// Branch-free window check: always store the mark (the spare entry outside the window).
    Always(String),
    /// Branching window check: mark slot `index` when `inside` holds.
    Inside { inside: String, index: String },
}

/// Values of the branch-free window translation ([`WindowCheck::Select`]) of one address.
struct SelectDelta {
    /// `rel < size`: the address is inside the window.
    inside: String,
    /// `addr - W`.
    rel: String,
    /// `rel >> SLOT_SHIFT`, the slot of an in-window address.
    raw: String,
    /// What to add to the address for its host address.
    delta: String,
}

impl FunctionLifter<'_> {
    /// Returns the `i64` flat base value for `base`, hoisting the invariant load of
    /// `@__playgta6_guest_base` once into `entry:` per function.
    fn flat_base_val(&mut self, base: GuestBase) -> String {
        match base {
            GuestBase::Constant(c) => imm(c, 64),
            GuestBase::Symbol => {
                self.support.insert(Support::GuestBase);
                if let Some(name) = &self.guest_base {
                    return name.clone();
                }
                let name = self.fresh_name();
                self.guest_base = Some(name.clone());
                name
            }
        }
    }

    /// `host = addr + base` as an `i64` (today's translation).
    fn flat_host(&mut self, addr: &str, base: GuestBase) -> String {
        match base {
            GuestBase::Constant(0) => addr.to_string(),
            GuestBase::Constant(base) => {
                self.val_raw(&format!("add i64 {addr}, {}", imm(base, 64)))
            }
            GuestBase::Symbol => {
                let base_val = self.flat_base_val(GuestBase::Symbol);
                if addr == "0" {
                    base_val
                } else {
                    self.val(&format!("add i64 {addr}, {base_val}"))
                }
            }
        }
    }

    /// Delta of the table slot `index` (an ordinary load: entries change at run time).
    fn table_delta(&mut self, table: &str, index: &str) -> String {
        let entry = self.val(&format!("getelementptr i64, ptr @{table}, i64 {index}"));
        self.val(&format!("load i64, ptr {entry}, align 8"))
    }

    /// Window table entry of the in-window offset `rel` ([`Self::window_offset`]).
    fn window_entry(&mut self, rel: &str) -> String {
        let offset = self.window_offset(rel);
        self.window_entry_at(&offset)
    }

    /// Byte offset of the table entry of window offset `rel`: `(rel >>a (SLOT_SHIFT - 3)) & -8`,
    /// equal to `(rel >> SLOT_SHIFT) * 8` for every `rel < 2^63` (every in-window offset; the
    /// value is only used behind the window test or discarded by a `select` on it).
    /// The scaled index `getelementptr i64, ..., rel >> 16` lowers on wasm to
    /// `(rel >> 13) & 0x7fff_ffff_ffff_fff8`, a 9-byte `i64.const` per window check. The
    /// arithmetic shift leaves the high bits unknown, so the `-8` stays a 1-byte constant, or
    /// shrinks to the range the window test proves (`0x3f_fff8`, 4 bytes) behind a branch.
    fn window_offset(&mut self, rel: &str) -> String {
        let shifted = self.val(&format!("ashr i64 {rel}, {}", SLOT_SHIFT - 3));
        self.val(&format!("and i64 {shifted}, -8"))
    }

    /// Window table entry at byte offset `offset` (a multiple of 8).
    fn window_entry_at(&mut self, offset: &str) -> String {
        let entry = self.val(&format!(
            "getelementptr i8, ptr @{WINDOW_TABLE_SYMBOL}, i64 {offset}"
        ));
        self.val(&format!("load i64, ptr {entry}, align 8"))
    }

    /// Page-table slot index of `addr`, clamped to the trap entry `pages`.
    fn page_index(&mut self, addr: &str, pages: u64) -> String {
        self.support.insert(Support::Memory(Decl::PageTable));
        let raw = self.val(&format!("lshr i64 {addr}, {SLOT_SHIFT}"));
        let inside = self.val(&format!("icmp ult i64 {raw}, {pages}"));
        self.val(&format!("select i1 {inside}, i64 {raw}, i64 {pages}"))
    }

    /// O37/R57: marks window slot `index` written (`__playgta6_window_dirty[index] = 1`, an atomic
    /// release store of one byte), skipping the atomic store when the slot is already marked.
    /// On wasm64, `store atomic i8 release` lowers to `i32.atomic.store8` (`LOCK XCHG` in V8
    /// TurboFan), while `load atomic i8 monotonic` lowers to a 1-cycle `movzx`, cutting serializing
    /// fences to at most once per 64 KiB slot per flush. Emitted after the store it reports;
    /// terminates the current block by branching to `done`.
    fn window_mark_then_br(&mut self, index: &str, done: &str) {
        self.support.insert(Support::Memory(Decl::WindowDirty));
        // Atomic load/store: the module needs the atomics target feature (shared memory).
        self.support.insert(Support::Atomics);
        let m = self.val(&format!(
            "getelementptr i8, ptr @{WINDOW_DIRTY_SYMBOL}, i64 {index}"
        ));
        let cur = self.val(&format!("load atomic i8, ptr {m} monotonic, align 1"));
        let clean = self.val(&format!("icmp eq i8 {cur}, 0"));
        let set = self.new_label("gms");
        self.emit(&format!(
            "br i1 {clean}, label %{set}, label %{done}, {UNLIKELY}"
        ));
        self.start_internal_block(&set);
        self.emit(&format!("store atomic i8 1, ptr {m} release, align 1"));
        self.emit(&format!("br label %{done}"));
    }

    /// O37/R57: marks window slot `index` written and continues in a fresh join block.
    fn window_mark(&mut self, index: &str) {
        let done = self.new_label("gmd");
        self.window_mark_then_br(index, &done);
        self.start_internal_block(&done);
    }

    /// O37: [`Self::guest_ptr`] for an access that writes (atomics, `lock`ed bit operations).
    /// After the write the caller passes the returned mark to [`Self::finish_write`], which marks
    /// the window slot written: in its own rarely taken block for the branching window check, as an
    /// unconditional store (the spare entry outside the window) for the branch-free one, nothing
    /// for models without a window.
    pub(super) fn guest_ptr_write(&mut self, addr: &str) -> (String, WriteMark) {
        match self.memory {
            MemoryModel::Windowed(w)
                if !(self.stack_flat().is_some() && self.stack_values.contains_key(addr)) =>
            {
                if let Some(va) = const_addr(addr) {
                    let rel = va.wrapping_sub(w.base);
                    if rel >= w.size {
                        return (self.guest_ptr(addr), WriteMark::None);
                    }
                    let slot = imm(rel >> SLOT_SHIFT, 64);
                    return (self.guest_ptr(addr), WriteMark::Always(slot));
                }
                let spare = w.size >> SLOT_SHIFT;
                if w.check == WindowCheck::Select {
                    // The translation's own window test and slot index feed the mark.
                    let SelectDelta {
                        inside, raw, delta, ..
                    } = self.window_select_delta(addr, w);
                    let host = self.val(&format!("add i64 {addr}, {delta}"));
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                    let index = self.val(&format!("select i1 {inside}, i64 {raw}, i64 {spare}"));
                    return (ptr, WriteMark::Always(index));
                }
                let (inside, rel) = self.window_test(addr, w);
                let raw = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                let index = self.val(&format!("select i1 {inside}, i64 {raw}, i64 {spare}"));
                let ptr = self.guest_ptr(addr);
                (ptr, WriteMark::Inside { inside, index })
            }
            _ => (self.guest_ptr(addr), WriteMark::None),
        }
    }

    /// O37: emits the mark [`Self::guest_ptr_write`] returned, after the write.
    pub(super) fn finish_write(&mut self, mark: WriteMark) {
        match mark {
            WriteMark::None => {}
            WriteMark::Always(index) => self.window_mark(&index),
            WriteMark::Inside { inside, index } => {
                let mark_block = self.new_label("gm");
                let cont = self.new_label("gmc");
                self.emit(&format!(
                    "br i1 {inside}, label %{mark_block}, label %{cont}, {UNLIKELY}"
                ));
                self.start_internal_block(&mark_block);
                self.window_mark_then_br(&index, &cont);
                self.start_internal_block(&cont);
            }
        }
    }

    /// The translation seam for accesses that never straddle a 64 KiB slot (naturally aligned:
    /// atomics, single bytes) and for address-only uses.
    pub(super) fn guest_ptr(&mut self, addr: &str) -> String {
        let host = match self.memory {
            MemoryModel::Base(base) => self.flat_host(addr, base),
            _ if self.stack_flat().is_some() && self.stack_values.contains_key(addr) => {
                let base = self.stack_flat().unwrap_or_default();
                self.flat_host(addr, base)
            }
            MemoryModel::PageTable { pages, .. } => {
                let index = if let Some(va) = const_addr(addr) {
                    self.support.insert(Support::Memory(Decl::PageTable));
                    imm((va >> SLOT_SHIFT).min(pages), 64)
                } else {
                    self.page_index(addr, pages)
                };
                let delta = self.table_delta(PAGE_TABLE_SYMBOL, &index);
                self.val(&format!("add i64 {addr}, {delta}"))
            }
            MemoryModel::Windowed(w) if const_addr(addr).is_some() => {
                let va = const_addr(addr).unwrap();
                let rel = va.wrapping_sub(w.base);
                if rel >= w.size {
                    self.flat_host(addr, w.flat)
                } else {
                    self.support.insert(Support::Memory(Decl::WindowTable));
                    let slot = imm(rel >> SLOT_SHIFT, 64);
                    let delta = self.table_delta(WINDOW_TABLE_SYMBOL, &slot);
                    self.val(&format!("add i64 {addr}, {delta}"))
                }
            }
            MemoryModel::Windowed(w) if w.check == WindowCheck::Select => {
                let SelectDelta { delta, .. } = self.window_select_delta(addr, w);
                self.val(&format!("add i64 {addr}, {delta}"))
            }
            MemoryModel::Windowed(w) => {
                let (inside, rel) = self.window_test(addr, w);
                let pred = self.current_block.clone();
                let flat_base = self.flat_base_val(w.flat);
                let win = self.new_label("gw");
                let join = self.new_label("gj");
                self.emit(&format!(
                    "br i1 {inside}, label %{win}, label %{join}, {UNLIKELY}"
                ));
                self.start_internal_block(&win);
                let delta_win = self.window_entry(&rel);
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&join);
                let delta = self.val(&format!(
                    "phi i64 [ {flat_base}, %{pred} ], [ {delta_win}, %{win} ]"
                ));
                self.val(&format!("add i64 {addr}, {delta}"))
            }
        };
        self.val(&format!("inttoptr i64 {host} to ptr"))
    }

    /// Branch-free window delta ([`WindowCheck::Select`]): the table entry of the slot (entry 0
    /// for addresses outside the window, which keeps the load in bounds) or the flat base.
    /// Its window test and slot index stay available for the write marks ([`Self::select_marks`]).
    fn window_select_delta(&mut self, addr: &str, w: Window) -> SelectDelta {
        let (inside, rel) = self.window_test(addr, w);
        let raw = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
        let offset = self.window_offset(&rel);
        let offset = self.val(&format!("select i1 {inside}, i64 {offset}, i64 0"));
        let entry = self.window_entry_at(&offset);
        let flat = self.flat_base_val(w.flat);
        let delta = self.val(&format!("select i1 {inside}, i64 {entry}, i64 {flat}"));
        SelectDelta {
            inside,
            rel,
            raw,
            delta,
        }
    }

    /// Window path of [`WindowCheck::Outline`]: one call of the module's helper for this width.
    fn window_outlined(
        &mut self,
        addr: &str,
        bytes: u64,
        base: u64,
        access: &Access<'_>,
    ) -> Option<String> {
        let vty = helper_value_type(bytes);
        let width = u8::try_from(bytes).unwrap_or(u8::MAX);
        match access {
            Access::Load { ty } => {
                if bytes > 1 {
                    self.support.insert(Support::Memory(Decl::Read));
                }
                self.support
                    .insert(Support::Memory(Decl::WindowLoad { bytes: width, base }));
                let name = window_helper(false, bytes, base);
                let raw = self.val(&format!("call {vty} @{name}(i64 {addr})"));
                Some(if *ty == vty {
                    raw
                } else {
                    self.val(&format!("bitcast {vty} {raw} to {ty}"))
                })
            }
            Access::Store { ty, value } => {
                if bytes > 1 {
                    self.support.insert(Support::Memory(Decl::Write));
                }
                self.support
                    .insert(Support::Memory(Decl::WindowStore { bytes: width, base }));
                self.support.insert(Support::Memory(Decl::WindowDirty));
                self.support.insert(Support::Atomics);
                let name = window_helper(true, bytes, base);
                let raw = if *ty == vty {
                    (*value).to_string()
                } else {
                    self.val(&format!("bitcast {ty} {value} to {vty}"))
                };
                self.emit(&format!("call void @{name}(i64 {addr}, {vty} {raw})"));
                None
            }
        }
    }

    /// `(inside, va - W)` for the window check, cached per address within the current basic block
    /// so same-address RMW / repeated accesses reuse the range check without recomputing `sub` +
    /// `icmp ult` with 64-bit immediates.
    fn window_test(&mut self, addr: &str, w: Window) -> (String, String) {
        self.support.insert(Support::Memory(Decl::WindowTable));
        if let Some((_, inside, rel)) = self.window_test_cache.iter().find(|(a, _, _)| a == addr) {
            return (inside.clone(), rel.clone());
        }
        let rel = self.val(&format!("sub i64 {addr}, {}", imm(w.base, 64)));
        let inside = self.val(&format!("icmp ult i64 {rel}, {}", imm(w.size, 64)));
        if self.window_test_cache.len() >= 8 {
            self.window_test_cache.remove(0);
        }
        self.window_test_cache
            .push((addr.to_string(), inside.clone(), rel.clone()));
        (inside, rel)
    }

    /// True if `[a1, a1 + s1)` and `[a2, a2 + s2)` are provably disjoint in 64-bit modular guest
    /// address space (including wrap-around): either both are constant VAs, or both share the
    /// same canonical affine base SSA value with non-overlapping displacements.
    fn are_addrs_disjoint(&self, a1: &str, s1: u64, a2: &str, s2: u64) -> bool {
        if s1 == 0 || s2 == 0 {
            return false;
        }
        let (off1, off2) = match (const_addr(a1), const_addr(a2)) {
            (Some(c1), Some(c2)) => (c1, c2),
            (None, None) => {
                let (b1, o1) = self.affine_root(a1);
                let (b2, o2) = self.affine_root(a2);
                if b1 != b2 {
                    return false;
                }
                (o1 as u64, o2 as u64)
            }
            _ => return false,
        };
        let d = off2.wrapping_sub(off1);
        d >= s1 && d.wrapping_neg() >= s2
    }

    /// Scalar integer width of an LLVM type (`i8`, `i16`, `i32`, `i64`), or `None` for vectors/FP.
    fn scalar_int_bits(ty: &str) -> Option<u32> {
        match ty {
            "i8" => Some(8),
            "i16" => Some(16),
            "i32" => Some(32),
            "i64" => Some(64),
            _ => None,
        }
    }

    /// Drops `mem_cache` entries invalidated by any non-memory-preserving IR line emitted since
    /// their `at` position.
    fn prune_mem_cache(&mut self) {
        let body = &self.body;
        self.mem_cache.retain(|e| {
            body.get(e.at..)
                .is_some_and(|since| since.lines().all(keeps_memory))
        });
    }

    /// Loads `ty` (`bytes` wide, `align 1`) from guest address `addr`.
    pub(super) fn guest_load(&mut self, addr: &str, ty: &str, bytes: u64) -> String {
        self.prune_mem_cache();
        // Exact match (newest first).
        if let Some(entry) = self
            .mem_cache
            .iter()
            .rev()
            .find(|e| e.addr == addr && e.ty == ty && e.bytes == bytes)
        {
            return entry.value.clone();
        }
        // Little-endian narrow scalar integer prefix match from a wider cached access at `addr`.
        if let Some(dst_bits) = Self::scalar_int_bits(ty)
            && let Some((src_ty, src_val)) = self
                .mem_cache
                .iter()
                .rev()
                .find_map(|e| {
                    (e.addr == addr
                        && e.bytes > bytes
                        && Self::scalar_int_bits(&e.ty).is_some_and(|sb| sb > dst_bits))
                    .then(|| (e.ty.clone(), e.value.clone()))
                })
        {
            return self.val(&format!("trunc {src_ty} {src_val} to {ty}"));
        }
        let value = self
            .guest_access(addr, bytes, Access::Load { ty })
            .unwrap_or_default();
        self.note_access(addr, ty, &value, bytes);
        value
    }

    /// Stores `value` of `ty` (`bytes` wide, `align 1`) to guest address `addr`.
    pub(super) fn guest_store(&mut self, addr: &str, ty: &str, value: &str, bytes: u64) {
        self.prune_mem_cache();
        // Keep only entries provably disjoint from `[addr, addr + bytes)`.
        let mut surviving: Vec<LastAccess> = Vec::with_capacity(self.mem_cache.len());
        for entry in std::mem::take(&mut self.mem_cache) {
            if self.are_addrs_disjoint(&entry.addr, entry.bytes, addr, bytes) {
                surviving.push(entry);
            }
        }
        let before = self.body.len();
        self.guest_access(addr, bytes, Access::Store { ty, value });
        let after = self.body.len();
        // Advance surviving disjoint entries ONLY if the store stayed inside the same LLVM block
        // without calls, atomics, fences, or branches (i.e. flat / stack-flat stores).
        let simple_store = self
            .body
            .get(before..after)
            .is_some_and(|emitted| {
                emitted.lines().all(|line| {
                    keeps_memory(line)
                        || line
                            .strip_prefix("  ")
                            .is_some_and(|code| {
                                code.starts_with("store ")
                                    && !code.starts_with("store atomic")
                                    && !code.starts_with("store volatile")
                            })
                })
            });
        if simple_store {
            for entry in &mut surviving {
                entry.at = after;
            }
            self.mem_cache = surviving;
        }
        self.note_access(addr, ty, value, bytes);
    }

    /// Remembers the access just emitted as a candidate for [`Self::guest_load`] forwarding.
    fn note_access(&mut self, addr: &str, ty: &str, value: &str, bytes: u64) {
        const MAX_MEM_CACHE: usize = 8;
        if value.is_empty() {
            return;
        }
        // Replace any existing entry for the exact same `(addr, bytes)` to keep the cache compact.
        self.mem_cache
            .retain(|e| !(e.addr == addr && e.bytes == bytes));
        if self.mem_cache.len() >= MAX_MEM_CACHE {
            self.mem_cache.remove(0);
        }
        self.mem_cache.push(LastAccess {
            addr: addr.to_string(),
            ty: ty.to_string(),
            value: value.to_string(),
            bytes,
            at: self.body.len(),
        });
    }

    /// Emits the access through `ptr`; the loaded value for loads.
    fn access_at(&mut self, ptr: &str, access: &Access<'_>) -> Option<String> {
        match access {
            Access::Load { ty } => Some(self.val(&format!("load {ty}, ptr {ptr}, align 1"))),
            Access::Store { ty, value } => {
                self.emit(&format!("store {ty} {value}, ptr {ptr}, align 1"));
                None
            }
        }
    }

    /// Straddle-safe slow path for loads: returns the per-thread scratch pointer holding `bytes`.
    fn slow_read_ptr(&mut self, addr: &str, bytes: u64) -> String {
        self.support.insert(Support::Memory(Decl::Read));
        self.val(&format!(
            "call ptr @{HELPER_GUEST_READ}(i64 {addr}, i64 {bytes})"
        ))
    }

    /// Straddle-safe slow path for stores: one cold direct call to the module's outlined helper.
    fn slow_store(&mut self, addr: &str, bytes: u64, ty: &str, value: &str) {
        let width = u8::try_from(bytes).unwrap_or(u8::MAX);
        self.support.insert(Support::Memory(Decl::Write));
        self.support
            .insert(Support::Memory(Decl::SlowStore { bytes: width }));
        let vty = helper_value_type(bytes);
        let raw = if ty == vty {
            value.to_string()
        } else {
            self.val(&format!("bitcast {ty} {value} to {vty}"))
        };
        let helper = slow_store_helper(bytes);
        self.emit(&format!("call void @{helper}(i64 {addr}, {vty} {raw})"));
    }

    fn guest_access(&mut self, addr: &str, bytes: u64, access: Access<'_>) -> Option<String> {
        let is_store = matches!(access, Access::Store { .. });
        if let Some(burst) = &mut self.burst_override {
            let mut matched = None;
            for slot in &mut burst.bases {
                if let Some(&disp) = slot.addrs.get(addr)
                    && disp >= slot.min_d
                    && disp
                        .checked_add(bytes as i64)
                        .is_some_and(|end| end <= slot.max_end)
                    && (!is_store || slot.has_write)
                {
                    if burst.fast && is_store {
                        slot.saw_store = true;
                    }
                    matched = Some((burst.fast, slot.base_host.clone(), disp));
                    break;
                }
            }
            if let Some((fast, base_host, disp)) = matched {
                if fast {
                    let base_host = base_host.expect("fast burst base_host");
                    let host = if disp == 0 {
                        base_host
                    } else {
                        self.val(&format!("add i64 {base_host}, {}", imm(disp as u64, 64)))
                    };
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                    return self.access_at(&ptr, &access);
                }
                return match access {
                    Access::Load { .. } => {
                        let ptr = self.slow_read_ptr(addr, bytes);
                        self.access_at(&ptr, &access)
                    }
                    Access::Store { ty, value } => {
                        self.slow_store(addr, bytes, ty, value);
                        None
                    }
                };
            }
        }
        match self.memory {
            MemoryModel::Base(_) => {
                let ptr = self.guest_ptr(addr);
                self.access_at(&ptr, &access)
            }
            _ if self.stack_flat().is_some() && self.stack_values.contains_key(addr) => {
                let base = self.stack_flat().unwrap_or_default();
                let host = self.flat_host(addr, base);
                let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                self.access_at(&ptr, &access)
            }
            MemoryModel::PageTable { pages, .. } => {
                if let Some(va) = const_addr(addr) {
                    let offset = va & (SLOT_BYTES - 1);
                    let straddles = bytes > 1 && offset > SLOT_BYTES - bytes;
                    if straddles && !self.contiguous_runs() {
                        return match access {
                            Access::Load { .. } => {
                                let ptr = self.slow_read_ptr(addr, bytes);
                                self.access_at(&ptr, &access)
                            }
                            Access::Store { ty, value } => {
                                self.slow_store(addr, bytes, ty, value);
                                None
                            }
                        };
                    }
                    self.support.insert(Support::Memory(Decl::PageTable));
                    let index = imm((va >> SLOT_SHIFT).min(pages), 64);
                    let delta = self.table_delta(PAGE_TABLE_SYMBOL, &index);
                    let host = self.val(&format!("add i64 {addr}, {delta}"));
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                    return self.access_at(&ptr, &access);
                }
                if bytes <= 1 || self.contiguous_runs() {
                    let index = self.page_index(addr, pages);
                    let delta = self.table_delta(PAGE_TABLE_SYMBOL, &index);
                    let host = self.val(&format!("add i64 {addr}, {delta}"));
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                    return self.access_at(&ptr, &access);
                }
                let fast = self.new_label("gt");
                let slow = self.new_label("gs");
                let join = self.new_label("gj");
                let offset = self.val(&format!("and i64 {addr}, {}", SLOT_BYTES - 1));
                let fits = self.val(&format!("icmp ule i64 {offset}, {}", SLOT_BYTES - bytes));
                self.emit(&format!(
                    "br i1 {fits}, label %{fast}, label %{slow}, {LIKELY}"
                ));
                self.start_internal_block(&fast);
                let index = self.page_index(addr, pages);
                let delta = self.table_delta(PAGE_TABLE_SYMBOL, &index);
                let host = self.val(&format!("add i64 {addr}, {delta}"));
                let p_fast = self.val(&format!("inttoptr i64 {host} to ptr"));
                if let Access::Store { .. } = access {
                    self.access_at(&p_fast, &access);
                }
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&slow);
                let p_slow = match access {
                    Access::Load { .. } => Some(self.slow_read_ptr(addr, bytes)),
                    Access::Store { ty, value } => {
                        self.slow_store(addr, bytes, ty, value);
                        None
                    }
                };
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&join);
                if let Some(p_slow) = p_slow {
                    let ptr = self.val(&format!(
                        "phi ptr [ {p_fast}, %{fast} ], [ {p_slow}, %{slow} ]"
                    ));
                    self.access_at(&ptr, &access)
                } else {
                    None
                }
            }
            MemoryModel::Windowed(w) if const_addr(addr).is_some() => {
                let va = const_addr(addr).unwrap();
                self.const_window_access(addr, va, bytes, w, &access)
            }
            MemoryModel::Windowed(w) if w.check == WindowCheck::Select => {
                let sel = self.window_select_delta(addr, w);
                let (inside, delta) = (sel.inside.clone(), sel.delta.clone());
                if bytes <= 1 || self.contiguous_runs() {
                    let host = self.val(&format!("add i64 {addr}, {delta}"));
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                    let v = self.access_at(&ptr, &access);
                    if let Access::Store { .. } = access {
                        self.select_marks(&sel, w, bytes);
                    }
                    return v;
                }
                // One rarely taken branch: a window access that leaves its slot.
                let offset = self.val(&format!("and i64 {addr}, {}", SLOT_BYTES - 1));
                let over = self.val(&format!("icmp ugt i64 {offset}, {}", SLOT_BYTES - bytes));
                let slow_cond = self.val(&format!("and i1 {inside}, {over}"));
                let fast = self.new_label("gt");
                let slow = self.new_label("gs");
                let join = self.new_label("gj");
                self.emit(&format!(
                    "br i1 {slow_cond}, label %{slow}, label %{fast}, {UNLIKELY}"
                ));
                self.start_internal_block(&fast);
                let host = self.val(&format!("add i64 {addr}, {delta}"));
                let p_fast = self.val(&format!("inttoptr i64 {host} to ptr"));
                if let Access::Store { .. } = access {
                    self.access_at(&p_fast, &access);
                    self.select_marks(&sel, w, 1);
                }
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&slow);
                let p_slow = match access {
                    Access::Load { .. } => Some(self.slow_read_ptr(addr, bytes)),
                    Access::Store { ty, value } => {
                        self.slow_store(addr, bytes, ty, value);
                        None
                    }
                };
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&join);
                if let Some(p_slow) = p_slow {
                    let ptr = self.val(&format!(
                        "phi ptr [ {p_fast}, %{fast} ], [ {p_slow}, %{slow} ]"
                    ));
                    self.access_at(&ptr, &access)
                } else {
                    None
                }
            }
            MemoryModel::Windowed(w) if w.check == WindowCheck::Outline => {
                let (inside, _) = self.window_test(addr, w);
                let win = self.new_label("gw");
                let flat = self.new_label("gf");
                let join = self.new_label("gj");
                self.emit(&format!(
                    "br i1 {inside}, label %{win}, label %{flat}, {UNLIKELY}"
                ));
                self.start_internal_block(&flat);
                let host = self.flat_host(addr, w.flat);
                let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                let v_flat = self.access_at(&ptr, &access);
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&win);
                let v_win = self.window_outlined(addr, bytes, w.base, &access);
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&join);
                match (access, v_flat, v_win) {
                    (Access::Load { ty }, Some(vf), Some(vw)) => {
                        Some(self.val(&format!("phi {ty} [ {vf}, %{flat} ], [ {vw}, %{win} ]")))
                    }
                    _ => None,
                }
            }
            MemoryModel::Windowed(w) if matches!(access, Access::Store { .. }) => {
                if (bytes <= 1 || self.contiguous_runs())
                    && let Some(cached) = self
                        .window_load_cache
                        .iter()
                        .find(|c| c.addr == addr)
                        .cloned()
                {
                    self.access_at(&cached.ptr, &access);
                    let mark_block = self.new_label("gm");
                    let join = self.new_label("gj");
                    self.emit(&format!(
                        "br i1 {}, label %{mark_block}, label %{join}, {UNLIKELY}",
                        cached.inside
                    ));
                    self.start_internal_block(&mark_block);
                    let index = self.val(&format!("lshr i64 {}, {SLOT_SHIFT}", cached.rel));
                    self.emit_window_store_marks(&index, &cached.rel, bytes, &join);
                    self.start_internal_block(&join);
                    return None;
                }
                self.window_store(addr, bytes, w, &access);
                None
            }
            MemoryModel::Windowed(w) => {
                if (bytes <= 1 || self.contiguous_runs())
                    && let Some(cached) = self
                        .window_load_cache
                        .iter()
                        .find(|c| c.addr == addr)
                        .cloned()
                {
                    return self.access_at(&cached.ptr, &access);
                }
                let (inside, rel) = self.window_test(addr, w);
                let pred = self.current_block.clone();
                let flat_base = self.flat_base_val(w.flat);
                if bytes <= 1 || self.contiguous_runs() {
                    let win = self.new_label("gw");
                    let join = self.new_label("gj");
                    self.emit(&format!(
                        "br i1 {inside}, label %{win}, label %{join}, {UNLIKELY}"
                    ));
                    self.start_internal_block(&win);
                    let delta_win = self.window_entry(&rel);
                    self.emit(&format!("br label %{join}"));
                    self.start_internal_block(&join);
                    let delta = self.val(&format!(
                        "phi i64 [ {flat_base}, %{pred} ], [ {delta_win}, %{win} ]"
                    ));
                    let host = self.val(&format!("add i64 {addr}, {delta}"));
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                    if self.window_load_cache.len() >= MAX_WINDOW_LOAD_CACHE {
                        self.window_load_cache.remove(0);
                    }
                    self.window_load_cache.push(CachedWindowAddr {
                        addr: addr.to_string(),
                        inside,
                        rel,
                        ptr: ptr.clone(),
                    });
                    return self.access_at(&ptr, &access);
                }
                let win = self.new_label("gw");
                let wtbl = self.new_label("gt");
                let fast = self.new_label("gf");
                let slow = self.new_label("gs");
                let join = self.new_label("gj");
                self.emit(&format!(
                    "br i1 {inside}, label %{win}, label %{fast}, {UNLIKELY}"
                ));
                self.start_internal_block(&win);
                let offset = self.val(&format!("and i64 {addr}, {}", SLOT_BYTES - 1));
                let fits = self.val(&format!("icmp ule i64 {offset}, {}", SLOT_BYTES - bytes));
                self.emit(&format!(
                    "br i1 {fits}, label %{wtbl}, label %{slow}, {LIKELY}"
                ));
                self.start_internal_block(&wtbl);
                let delta_win = self.window_entry(&rel);
                self.emit(&format!("br label %{fast}"));
                self.start_internal_block(&fast);
                let delta = self.val(&format!(
                    "phi i64 [ {flat_base}, %{pred} ], [ {delta_win}, %{wtbl} ]"
                ));
                let host = self.val(&format!("add i64 {addr}, {delta}"));
                let p_fast = self.val(&format!("inttoptr i64 {host} to ptr"));
                if let Access::Store { .. } = access {
                    self.access_at(&p_fast, &access);
                }
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&slow);
                let p_slow = match access {
                    Access::Load { .. } => Some(self.slow_read_ptr(addr, bytes)),
                    Access::Store { ty, value } => {
                        self.slow_store(addr, bytes, ty, value);
                        None
                    }
                };
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&join);
                if let Some(p_slow) = p_slow {
                    let ptr = self.val(&format!(
                        "phi ptr [ {p_fast}, %{fast} ], [ {p_slow}, %{slow} ]"
                    ));
                    self.access_at(&ptr, &access)
                } else {
                    None
                }
            }
        }
    }

    /// R58: guest access to a lift-time constant VA under [`MemoryModel::Windowed`].
    /// - Outside `[w.base, w.base + w.size)` (all `[rip + disp]` executable globals): direct flat
    ///   translation with zero window check, branch, table load, or dirty mark.
    /// - Inside `[w.base, w.base + w.size)`: direct slot lookup (or direct slow path if straddling
    ///   when `!contiguous_runs`), with static `last_slot != slot_idx` crossing check on stores.
    fn const_window_access(
        &mut self,
        addr: &str,
        va: u64,
        bytes: u64,
        w: Window,
        access: &Access<'_>,
    ) -> Option<String> {
        let rel = va.wrapping_sub(w.base);
        if rel >= w.size {
            let host = self.flat_host(addr, w.flat);
            let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
            return self.access_at(&ptr, access);
        }
        let offset = va & (SLOT_BYTES - 1);
        let straddles = bytes > 1 && offset > SLOT_BYTES - bytes;
        if straddles && !self.contiguous_runs() {
            return match *access {
                Access::Load { .. } => {
                    let ptr = self.slow_read_ptr(addr, bytes);
                    self.access_at(&ptr, access)
                }
                Access::Store { ty, value } => {
                    self.slow_store(addr, bytes, ty, value);
                    None
                }
            };
        }
        self.support.insert(Support::Memory(Decl::WindowTable));
        let slot_idx = rel >> SLOT_SHIFT;
        let slot_imm = imm(slot_idx, 64);
        let delta = self.table_delta(WINDOW_TABLE_SYMBOL, &slot_imm);
        let host = self.val(&format!("add i64 {addr}, {delta}"));
        let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
        let v = self.access_at(&ptr, access);
        if matches!(access, Access::Store { .. }) {
            self.window_mark(&slot_imm);
            if bytes > 1 && self.contiguous_runs() {
                let last_slot = rel.wrapping_add(bytes - 1) >> SLOT_SHIFT;
                if last_slot != slot_idx {
                    self.window_mark(&imm(last_slot, 64));
                }
            }
        }
        v
    }

    /// Emits the dirty mark(s) for a confirmed in-window store and branches to `join`.
    fn emit_window_store_marks(&mut self, index: &str, rel: &str, bytes: u64, join: &str) {
        if bytes > 1 && self.contiguous_runs() {
            self.window_mark(index);
            // The slot of the last byte needs its own mark only when the store crosses into it
            // (rare, ~0.005%). Emit a compact direct atomic release store inside the cold block
            // instead of another load-check-branch diamond, while keeping the primary slot's R57
            // load-before-store guard above.
            let last = self.val(&format!("add i64 {rel}, {}", bytes - 1));
            let last_index = self.val(&format!("lshr i64 {last}, {SLOT_SHIFT}"));
            let crosses = self.val(&format!("icmp ne i64 {last_index}, {index}"));
            let mark_last = self.new_label("gm");
            self.emit(&format!(
                "br i1 {crosses}, label %{mark_last}, label %{join}, {UNLIKELY}"
            ));
            self.start_internal_block(&mark_last);
            self.support.insert(Support::Memory(Decl::WindowDirty));
            self.support.insert(Support::Atomics);
            let m2 = self.val(&format!(
                "getelementptr i8, ptr @{WINDOW_DIRTY_SYMBOL}, i64 {last_index}"
            ));
            self.emit(&format!("store atomic i8 1, ptr {m2} release, align 1"));
            self.emit(&format!("br label %{join}"));
        } else {
            self.window_mark_then_br(index, join);
        }
    }

    /// O37: a store under the branching window check. The flat path stores directly; the window
    /// path stores and then marks the slot written (and, when runs are kept contiguous and the store
    /// leaves its slot, the slot of its last byte, behind a rarely taken branch); a store that may leave its slot otherwise
    /// takes the runtime helper, which marks. No hot branch beyond the window check and the
    /// straddle check the loads have too.
    fn window_store(&mut self, addr: &str, bytes: u64, w: Window, access: &Access<'_>) {
        let (inside, rel) = self.window_test(addr, w);
        let win = self.new_label("gw");
        let flat = self.new_label("gf");
        let join = self.new_label("gj");
        self.emit(&format!(
            "br i1 {inside}, label %{win}, label %{flat}, {UNLIKELY}"
        ));
        self.start_internal_block(&flat);
        let host_flat = self.flat_host(addr, w.flat);
        let p_flat = self.val(&format!("inttoptr i64 {host_flat} to ptr"));
        self.access_at(&p_flat, access);
        self.emit(&format!("br label %{join}"));
        self.start_internal_block(&win);
        let straddle_checked = bytes > 1 && !self.contiguous_runs();
        let slow = if straddle_checked {
            let table = self.new_label("gt");
            let slow = self.new_label("gs");
            let offset = self.val(&format!("and i64 {addr}, {}", SLOT_BYTES - 1));
            let fits = self.val(&format!("icmp ule i64 {offset}, {}", SLOT_BYTES - bytes));
            self.emit(&format!(
                "br i1 {fits}, label %{table}, label %{slow}, {LIKELY}"
            ));
            self.start_internal_block(&table);
            Some(slow)
        } else {
            None
        };
        let index = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
        let delta = self.window_entry(&rel);
        let host = self.val(&format!("add i64 {addr}, {delta}"));
        let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
        self.access_at(&ptr, access);
        self.emit_window_store_marks(&index, &rel, bytes, &join);
        if let (Some(slow), Access::Store { ty, value }) = (slow, access) {
            self.start_internal_block(&slow);
            self.slow_store(addr, bytes, ty, value);
            self.emit(&format!("br label %{join}"));
        }
        self.start_internal_block(&join);
    }

    /// O37: marks of a store under the branch-free window check: the slot (the spare entry
    /// outside the window) and, when runs are kept contiguous and the store is wider than a byte,
    /// the slot of its last byte. Reuses the window test and slot index of the access itself
    /// (`sel` dominates the store), so a mark costs no second `sub`/`lshr` of the address.
    fn select_marks(&mut self, sel: &SelectDelta, w: Window, bytes: u64) {
        let SelectDelta {
            inside, rel, raw, ..
        } = sel;
        let spare = w.size >> SLOT_SHIFT;
        let index = self.val(&format!("select i1 {inside}, i64 {raw}, i64 {spare}"));
        self.window_mark(&index);
        if bytes > 1 && self.contiguous_runs() {
            let last = self.val(&format!("add i64 {rel}, {}", bytes - 1));
            let last_raw = self.val(&format!("lshr i64 {last}, {SLOT_SHIFT}"));
            let last_index = self.val(&format!("select i1 {inside}, i64 {last_raw}, i64 {spare}"));
            self.window_mark(&last_index);
        }
    }

    /// True if the runtime keeps adjacent mapped slots host-contiguous (no straddle check).
    fn contiguous_runs(&self) -> bool {
        match self.memory {
            MemoryModel::Windowed(w) => w.contiguous_runs,
            MemoryModel::PageTable {
                contiguous_runs, ..
            } => contiguous_runs,
            MemoryModel::Base(_) => true,
        }
    }

    /// Flat translation of stack-pointer-relative accesses, if the model has one.
    fn stack_flat(&self) -> Option<GuestBase> {
        match self.memory {
            MemoryModel::Windowed(w) if w.stack_flat => Some(w.flat),
            MemoryModel::PageTable { stack, .. } => stack,
            _ => None,
        }
    }

    /// Notes that SSA `value` is a stack-pointer-relative guest address at `disp` bytes from `rsp`
    /// (or `entry_rsp`) (see `stack_flat`). Non-SSA integer literals are ignored so they can never
    /// alias arbitrary constant addresses.
    pub(super) fn note_stack_value(&mut self, value: &str, disp: i64) {
        if self.stack_flat().is_some() && value.starts_with('%') {
            self.stack_values
                .insert(value.to_string(), StackDisp::Exact(disp));
        }
    }

    /// Notes that SSA `value` is a stack-pointer-relative guest address within `max_abs` bytes of
    /// `entry_rsp` (`0 <= max_abs <= STACK_FRAME_MAX_DISP`).
    pub(super) fn note_stack_bounded(&mut self, value: &str, max_abs: i64) {
        if self.stack_flat().is_some()
            && value.starts_with('%')
            && (0..=STACK_FRAME_MAX_DISP).contains(&max_abs)
        {
            self.stack_values
                .insert(value.to_string(), StackDisp::Bounded(max_abs));
        }
    }

    /// Propagates bounded stack-pointer provenance from `base` to `derived = base + delta`
    /// (e.g. frame-pointer `[rbp - disp]` or high-half AVX `[rsp + disp + 16]`).
    pub(super) fn propagate_stack_offset(&mut self, base: &str, derived: &str, delta: i64) {
        if self.stack_flat().is_none() || !derived.starts_with('%') {
            return;
        }
        let next = match self.stack_values.get(base).copied() {
            Some(StackDisp::Exact(base_disp)) => base_disp
                .checked_add(delta)
                .filter(|total| total.abs() <= STACK_FRAME_MAX_DISP)
                .map(StackDisp::Exact),
            Some(StackDisp::Bounded(max_abs)) => delta
                .checked_abs()
                .and_then(|d| max_abs.checked_add(d))
                .filter(|&total| total <= STACK_FRAME_MAX_DISP)
                .map(StackDisp::Bounded),
            None => None,
        };
        if let Some(next) = next {
            self.stack_values.insert(derived.to_string(), next);
        }
    }

    /// R58/R63/R64: lifts a coalesced multi-base window burst (`insns = body[burst.start..=burst.end]`,
    /// up to [`MAX_BURST_BASES`] unmodified base GPRs) under `WindowCheck::Branch` with a single
    /// range/page check per active base for its span `[reg + min_d, reg + max_end)`:
    /// - Fast path (each base is flat outside `[W, W+S)`, or inside a single 64 KiB window slot):
    ///   one `delta` per base, direct `base_host + disp` accesses with zero intervening branches
    ///   or table loads, and at most one dirty mark per written base after all stores.
    /// - Cold fallback (any base's span straddles `W`, `W+S`, or a 64 KiB slot boundary inside
    ///   the window): compact cold runtime helpers (`guest_read` / `slow_storeN`).
    ///
    /// Returns `Ok(true)` if the burst was lifted, or `Ok(false)` (with state rolled back) if any
    /// instruction in the burst is unsupported so the caller can fall back instruction-by-instruction.
    pub(super) fn lift_window_burst(
        &mut self,
        burst: WindowBurst,
        insns: &[Instruction],
        burst_dead: &[super::FlagMask],
    ) -> R<bool> {
        let MemoryModel::Windowed(w) = self.memory else {
            return Ok(false);
        };
        if w.check != WindowCheck::Branch {
            return Ok(false);
        }
        if (w.base & (SLOT_BYTES - 1)) != 0 || (w.size & (SLOT_BYTES - 1)) != 0 {
            return Ok(false);
        }
        // Skip any plain base register already known to be stack-flat (its accesses are already
        // flat with zero overhead) or a compile-time constant whose span lies strictly outside the
        // window (`const_window_access` already emits direct flat loads/stores with zero checks);
        // indexed `[base + idx*scale + disp]` is never assumed stack-flat.
        let mut active_bases: Vec<BurstBase> = Vec::with_capacity(MAX_BURST_BASES);
        let mut span_m1s: Vec<u64> = Vec::with_capacity(MAX_BURST_BASES);
        for &b in burst.active_bases() {
            if b.key.index.is_none()
                && self.stack_flat().is_some()
                && self.is_gpr_stack_proven(b.key.base)
            {
                continue;
            }
            let Some(span) = b
                .max_end
                .checked_sub(b.min_d)
                .and_then(|s| u64::try_from(s).ok())
            else {
                return Ok(false);
            };
            if span == 0 || span > BURST_MAX_SPAN {
                return Ok(false);
            }
            let span_m1 = span - 1;
            if b.key.index.is_none()
                && let Some(va) = self.known_gpr(b.key.base).and_then(const_addr)
            {
                let lo = va.wrapping_add(b.min_d as u64);
                let hi = lo.wrapping_add(span_m1);
                let win_end = w.base.wrapping_add(w.size);
                let outside = lo <= hi
                    && w.base <= win_end
                    && (hi < w.base || lo >= win_end);
                if outside {
                    continue;
                }
            }
            if w.base < span_m1 || w.size > u64::MAX - span_m1 {
                return Ok(false);
            }
            active_bases.push(b);
            span_m1s.push(span_m1);
        }
        if active_bases.is_empty() {
            return Ok(false);
        }

        let checkpoint = self.checkpoint();
        let result: R<()> = (|| {
            let mut base_vals: Vec<String> = Vec::with_capacity(active_bases.len());
            for &b in &active_bases {
                let base_val = self.load_gpr(b.key.base)?;
                let eff_base = match b.key.index {
                    None => base_val,
                    Some((idx_reg, scale)) => {
                        let mut scaled = self.load_gpr(idx_reg)?;
                        if scale > 1 {
                            scaled = self.val(&format!(
                                "shl i64 {scaled}, {}",
                                scale.trailing_zeros()
                            ));
                        }
                        self.val(&format!("add i64 {base_val}, {scaled}"))
                    }
                };
                base_vals.push(eff_base);
            }
            let entry_known = self.known.clone();
            let entry_facts = self.block_entry_facts;
            let entry_ymmh_zeroed = self.ymmh_zeroed;
            self.support.insert(Support::Memory(Decl::WindowTable));
            let flat_base = self.flat_base_val(w.flat);
            let fast = self.new_label("gbf");
            let slow = self.new_label("gbs");
            let join = self.new_label("gbj");

            let mut deltas: Vec<String> = Vec::with_capacity(active_bases.len());
            let mut write_metas: Vec<Option<(String, String)>> =
                Vec::with_capacity(active_bases.len());
            for (idx, (&b, &span_m1)) in active_bases.iter().zip(&span_m1s).enumerate() {
                let base_val = &base_vals[idx];
                let lo = if b.min_d == 0 {
                    base_val.clone()
                } else {
                    self.val(&format!(
                        "add i64 {base_val}, {}",
                        imm(b.min_d as u64, 64)
                    ))
                };
                let ext_base = w.base - span_m1;
                let ext_size = w.size + span_m1;
                let rel_ext = self.val(&format!("sub i64 {lo}, {}", imm(ext_base, 64)));
                let touches = self.val(&format!("icmp ult i64 {rel_ext}, {}", imm(ext_size, 64)));
                let pred = self.current_block.clone();
                let win = self.new_label("gbw");
                let wtbl = self.new_label("gbt");
                let next = if idx + 1 == active_bases.len() {
                    fast.clone()
                } else {
                    self.new_label("gbc")
                };
                self.emit(&format!(
                    "br i1 {touches}, label %{win}, label %{next}, {UNLIKELY}"
                ));

                // Window check: all bytes of base `idx` lie in one 64 KiB slot inside [W, W + S).
                self.start_internal_block(&win);
                let rel = self.val(&format!("sub i64 {lo}, {}", imm(w.base, 64)));
                let in_win = self.val(&format!("icmp ult i64 {rel}, {}", imm(w.size, 64)));
                let off = self.val(&format!("and i64 {lo}, {}", SLOT_BYTES - 1));
                let fits =
                    self.val(&format!("icmp ule i64 {off}, {}", (SLOT_BYTES - 1) - span_m1));
                let ok = self.val(&format!("and i1 {in_win}, {fits}"));
                self.emit(&format!(
                    "br i1 {ok}, label %{wtbl}, label %{slow}, {LIKELY}"
                ));

                self.start_internal_block(&wtbl);
                let slot = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                let delta_win = self.window_entry(&rel);
                self.emit(&format!("br label %{next}"));

                self.start_internal_block(&next);
                let delta = self.val(&format!(
                    "phi i64 [ {flat_base}, %{pred} ], [ {delta_win}, %{wtbl} ]"
                ));
                let write_meta = if b.has_write {
                    let iw = self.val(&format!("phi i1 [ false, %{pred} ], [ true, %{wtbl} ]"));
                    let ws = self.val(&format!("phi i64 [ 0, %{pred} ], [ {slot}, %{wtbl} ]"));
                    Some((iw, ws))
                } else {
                    None
                };
                deltas.push(delta);
                write_metas.push(write_meta);
            }

            // Fast path (already positioned at `%fast`): flat outside window OR single window slot
            // for every active base.
            self.known = entry_known.clone();
            self.block_entry_facts = entry_facts;
            self.ymmh_zeroed = entry_ymmh_zeroed;
            self.clear_window_cache();
            let mut fast_override_bases = Vec::with_capacity(active_bases.len());
            for (idx, &b) in active_bases.iter().enumerate() {
                let base_host = self.val(&format!("add i64 {}, {}", base_vals[idx], deltas[idx]));
                fast_override_bases.push(BurstOverrideBase {
                    key: b.key,
                    min_d: b.min_d,
                    max_end: b.max_end,
                    has_write: b.has_write,
                    base_host: Some(base_host),
                    saw_store: false,
                    addrs: BTreeMap::new(),
                });
            }
            self.burst_override = Some(BurstOverride {
                fast: true,
                bases: fast_override_bases,
            });
            for (k, ins) in insns.iter().enumerate() {
                self.ea = None;
                self.dead_flags = burst_dead[k];
                let r = self.lift_instruction(ins);
                self.dead_flags = 0;
                r?;
            }
            let fast_known = self.known.clone();
            let fast_facts = self.block_entry_facts;
            let fast_ymmh_zeroed = self.ymmh_zeroed;
            let done_fast = self.burst_override.take();
            let mut marks: Vec<(String, String)> = Vec::new();
            if let Some(done) = done_fast {
                for (slot, meta) in done.bases.iter().zip(write_metas) {
                    if slot.saw_store
                        && let Some((iw, ws)) = meta
                    {
                        marks.push((iw, ws));
                    }
                }
            }
            if marks.is_empty() {
                self.emit(&format!("br label %{join}"));
            } else {
                for (idx, (iw, ws)) in marks.iter().enumerate() {
                    let mark = self.new_label("gbm");
                    let next = if idx + 1 == marks.len() {
                        join.clone()
                    } else {
                        self.new_label("gbn")
                    };
                    self.emit(&format!(
                        "br i1 {iw}, label %{mark}, label %{next}, {UNLIKELY}"
                    ));
                    self.start_internal_block(&mark);
                    self.window_mark_then_br(ws, &next);
                    if idx + 1 < marks.len() {
                        self.start_internal_block(&next);
                    }
                }
            }

            // Cold fallback: either base's span crosses W, W+S, or a 64 KiB window slot boundary.
            // Route both bases' `[reg + disp]` accesses through compact cold runtime helpers
            // (`guest_read` / `slow_storeN`) so the fallback adds minimal IR and handles arbitrary
            // straddles.
            self.start_internal_block(&slow);
            self.known = entry_known.clone();
            self.block_entry_facts = entry_facts;
            self.ymmh_zeroed = entry_ymmh_zeroed;
            self.clear_window_cache();
            let slow_override_bases = active_bases
                .iter()
                .map(|&b| BurstOverrideBase {
                    key: b.key,
                    min_d: b.min_d,
                    max_end: b.max_end,
                    has_write: b.has_write,
                    base_host: None,
                    saw_store: false,
                    addrs: BTreeMap::new(),
                })
                .collect();
            self.burst_override = Some(BurstOverride {
                fast: false,
                bases: slow_override_bases,
            });
            for (k, ins) in insns.iter().enumerate() {
                self.ea = None;
                self.dead_flags = burst_dead[k];
                let r = self.lift_instruction(ins);
                self.dead_flags = 0;
                r?;
            }
            let slow_known = self.known.clone();
            let slow_facts = self.block_entry_facts;
            let slow_ymmh_zeroed = self.ymmh_zeroed;
            self.burst_override = None;
            self.emit(&format!("br label %{join}"));

            // Join block: preserve `entry_known` for registers/flags untouched by both arms (keeps
            // stack provenance like frame pointers intact across bursts), intersect `ymmh_zeroed`,
            // and clear modified registers so `mem2reg` merges them via allocas.
            self.start_block(&join);
            self.restore_unchanged_known(&entry_known, &fast_known, &slow_known);
            self.restore_unchanged_block_entry_facts(&entry_facts, &fast_facts, &slow_facts);
            self.ymmh_zeroed = fast_ymmh_zeroed & slow_ymmh_zeroed;
            Ok(())
        })();

        match result {
            Ok(()) => Ok(true),
            Err(super::InsnError::Cfg(err)) => {
                self.burst_override = None;
                Err(super::InsnError::Cfg(err))
            }
            Err(super::InsnError::Unsupported(_)) => {
                self.burst_override = None;
                self.rollback(checkpoint);
                Ok(false)
            }
        }
    }

    /// True if guest accesses may create basic blocks (callers that build phis over an access
    /// must then end the access in a block of their own).
    pub(super) fn accesses_branch(&self) -> bool {
        !matches!(self.memory, MemoryModel::Base(_))
    }

    /// Guest `memmove(dst, src, len)` for the `rep movs` bulk path. `None` keeps the caller's
    /// `llvm.memmove` on translated pointers (the base model).
    pub(super) fn guest_memmove(&mut self, dst: &str, src: &str, len: &str) -> R<bool> {
        self.guest_bulk(&[dst, src], len, |s, flat| match flat {
            Some(ptrs) => {
                s.support.insert(Support::Memmove);
                s.emit(&format!(
                    "call void @llvm.memmove.p0.p0.i64(ptr align 1 {}, ptr align 1 {}, i64 {len}, i1 false)",
                    ptrs[0], ptrs[1]
                ));
            }
            None => {
                s.support.insert(Support::Memory(Decl::Memmove));
                s.emit(&format!(
                    "call void @{HELPER_GUEST_MEMMOVE}(i64 {dst}, i64 {src}, i64 {len})"
                ));
            }
        })
    }

    /// Guest `memset(dst, byte, len)` for the `rep stos` bulk path (see [`Self::guest_memmove`]).
    pub(super) fn guest_memset(&mut self, dst: &str, byte: &str, len: &str) -> R<bool> {
        self.guest_bulk(&[dst], len, |s, flat| match flat {
            Some(ptrs) => {
                s.support.insert(Support::Memset);
                s.emit(&format!(
                    "call void @llvm.memset.p0.i64(ptr align 1 {}, i8 {byte}, i64 {len}, i1 false)",
                    ptrs[0]
                ));
            }
            None => {
                s.support.insert(Support::Memory(Decl::Memset));
                s.emit(&format!(
                    "call void @{HELPER_GUEST_MEMSET}(i64 {dst}, i8 {byte}, i64 {len})"
                ));
            }
        })
    }

    /// Bulk operation over the guest ranges `[a, a + len)`: `emit(Some(host pointers))` when the
    /// ranges can use plain host pointers, `emit(None)` for the segmented runtime helper.
    /// Returns false for the base model (the caller emits the pre-R19 code itself).
    fn guest_bulk(
        &mut self,
        addrs: &[&str],
        len: &str,
        emit: impl Fn(&mut Self, Option<&[String]>),
    ) -> R<bool> {
        match self.memory {
            MemoryModel::Base(_) => Ok(false),
            MemoryModel::PageTable { .. } => {
                emit(self, None);
                Ok(true)
            }
            MemoryModel::Windowed(w) => {
                self.support.insert(Support::Memory(Decl::WindowTable));
                let end = imm(w.base.wrapping_add(w.size), 64);
                let mut touches: Option<String> = None;
                for a in addrs {
                    // [a, a + len) intersects [W, W + S): a < W + S && a + len > W.
                    let below_end = self.val(&format!("icmp ult i64 {a}, {end}"));
                    let last = self.val(&format!("add i64 {a}, {len}"));
                    let after_start =
                        self.val(&format!("icmp ugt i64 {last}, {}", imm(w.base, 64)));
                    let t = self.val(&format!("and i1 {below_end}, {after_start}"));
                    touches = Some(match touches {
                        Some(prev) => self.val(&format!("or i1 {prev}, {t}")),
                        None => t,
                    });
                }
                let touches = touches.unwrap_or_else(|| "false".to_string());
                let slow = self.new_label("gbs");
                let fast = self.new_label("gbf");
                let join = self.new_label("gbj");
                self.emit(&format!(
                    "br i1 {touches}, label %{slow}, label %{fast}, {UNLIKELY}"
                ));
                self.start_internal_block(&fast);
                let mut ptrs = Vec::new();
                for a in addrs {
                    let host = self.flat_host(a, w.flat);
                    ptrs.push(self.val(&format!("inttoptr i64 {host} to ptr")));
                }
                emit(self, Some(&ptrs));
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&slow);
                emit(self, None);
                self.emit(&format!("br label %{join}"));
                self.start_internal_block(&join);
                Ok(true)
            }
        }
    }
}
