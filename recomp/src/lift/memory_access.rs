//! Guest memory accesses under every [`MemoryModel`] (the translation seam; models and rules:
//! `lift/memory.rs`). With [`MemoryModel::Base`] every function here emits exactly the IR the
//! lifter emitted before the table models existed.

use super::{FunctionLifter, R, imm};
use crate::lift::memory::{
    Decl, HELPER_GUEST_MEMMOVE, HELPER_GUEST_MEMSET, HELPER_GUEST_READ, MemoryModel,
    PAGE_TABLE_SYMBOL, SLOT_BYTES, SLOT_SHIFT, WINDOW_DIRTY_SYMBOL, WINDOW_TABLE_SYMBOL, Window,
    WindowCheck, helper_value_type, slow_store_helper, window_helper,
};
use crate::lift::{GuestBase, Support};

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
            GuestBase::Constant(base) => self.val(&format!("add i64 {addr}, {}", imm(base, 64))),
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

    /// Page-table slot index of `addr`, clamped to the trap entry `pages`.
    fn page_index(&mut self, addr: &str, pages: u64) -> String {
        self.support.insert(Support::Memory(Decl::PageTable));
        let raw = self.val(&format!("lshr i64 {addr}, {SLOT_SHIFT}"));
        let inside = self.val(&format!("icmp ult i64 {raw}, {pages}"));
        self.val(&format!("select i1 {inside}, i64 {raw}, i64 {pages}"))
    }

    /// O37: marks window slot `index` written (`__playgta6_window_dirty[index] = 1`, an atomic
    /// release store of one byte). Emitted after the store it reports.
    fn window_mark(&mut self, index: &str) {
        self.support.insert(Support::Memory(Decl::WindowDirty));
        // An atomic store: the module needs the atomics target feature (shared memory).
        self.support.insert(Support::Atomics);
        let m = self.val(&format!(
            "getelementptr i8, ptr @{WINDOW_DIRTY_SYMBOL}, i64 {index}"
        ));
        self.emit(&format!("store atomic i8 1, ptr {m} release, align 1"));
    }

    /// O37: [`Self::guest_ptr`] for an access that writes (atomics, `lock`ed bit operations).
    /// After the write the caller passes the returned mark to [`Self::finish_write`], which marks
    /// the window slot written: in its own rarely taken block for the branching window check, as an
    /// unconditional store (the spare entry outside the window) for the branch-free one, nothing
    /// for models without a window.
    pub(super) fn guest_ptr_write(&mut self, addr: &str) -> (String, WriteMark) {
        match self.memory {
            MemoryModel::Windowed(w)
                if !(self.stack_flat().is_some() && self.stack_values.contains(addr)) =>
            {
                let (inside, rel) = self.window_test(addr, w);
                let raw = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                let spare = w.size >> SLOT_SHIFT;
                let index = self.val(&format!("select i1 {inside}, i64 {raw}, i64 {spare}"));
                let ptr = self.guest_ptr(addr);
                let mark = if w.check == WindowCheck::Select {
                    WriteMark::Always(index)
                } else {
                    WriteMark::Inside { inside, index }
                };
                (ptr, mark)
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
                self.start_block(&mark_block);
                self.window_mark(&index);
                self.emit(&format!("br label %{cont}"));
                self.start_block(&cont);
            }
        }
    }

    /// The translation seam for accesses that never straddle a 64 KiB slot (naturally aligned:
    /// atomics, single bytes) and for address-only uses.
    pub(super) fn guest_ptr(&mut self, addr: &str) -> String {
        let host = match self.memory {
            MemoryModel::Base(base) => self.flat_host(addr, base),
            _ if self.stack_flat().is_some() && self.stack_values.contains(addr) => {
                let base = self.stack_flat().unwrap_or_default();
                self.flat_host(addr, base)
            }
            MemoryModel::PageTable { pages, .. } => {
                let index = self.page_index(addr, pages);
                let delta = self.table_delta(PAGE_TABLE_SYMBOL, &index);
                self.val(&format!("add i64 {addr}, {delta}"))
            }
            MemoryModel::Windowed(w) if w.check == WindowCheck::Select => {
                let (_, delta) = self.window_select_delta(addr, w);
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
                self.start_block(&win);
                let index = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                let delta_win = self.table_delta(WINDOW_TABLE_SYMBOL, &index);
                self.emit(&format!("br label %{join}"));
                self.start_block(&join);
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
    /// Returns `(inside, delta)`.
    fn window_select_delta(&mut self, addr: &str, w: Window) -> (String, String) {
        let (inside, rel) = self.window_test(addr, w);
        let raw = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
        let index = self.val(&format!("select i1 {inside}, i64 {raw}, i64 0"));
        let entry = self.table_delta(WINDOW_TABLE_SYMBOL, &index);
        let flat = self.flat_base_val(w.flat);
        let delta = self.val(&format!("select i1 {inside}, i64 {entry}, i64 {flat}"));
        (inside, delta)
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

    /// `(inside, va - W)` for the window check.
    fn window_test(&mut self, addr: &str, w: Window) -> (String, String) {
        self.support.insert(Support::Memory(Decl::WindowTable));
        let rel = self.val(&format!("sub i64 {addr}, {}", imm(w.base, 64)));
        let inside = self.val(&format!("icmp ult i64 {rel}, {}", imm(w.size, 64)));
        (inside, rel)
    }

    /// Loads `ty` (`bytes` wide, `align 1`) from guest address `addr`.
    pub(super) fn guest_load(&mut self, addr: &str, ty: &str, bytes: u64) -> String {
        self.guest_access(addr, bytes, Access::Load { ty })
            .unwrap_or_default()
    }

    /// Stores `value` of `ty` (`bytes` wide, `align 1`) to guest address `addr`.
    pub(super) fn guest_store(&mut self, addr: &str, ty: &str, value: &str, bytes: u64) {
        self.guest_access(addr, bytes, Access::Store { ty, value });
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
        match self.memory {
            MemoryModel::Base(_) => {
                let ptr = self.guest_ptr(addr);
                self.access_at(&ptr, &access)
            }
            _ if self.stack_flat().is_some() && self.stack_values.contains(addr) => {
                let base = self.stack_flat().unwrap_or_default();
                let host = self.flat_host(addr, base);
                let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                self.access_at(&ptr, &access)
            }
            MemoryModel::PageTable { pages, .. } => {
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
                self.start_block(&fast);
                let index = self.page_index(addr, pages);
                let delta = self.table_delta(PAGE_TABLE_SYMBOL, &index);
                let host = self.val(&format!("add i64 {addr}, {delta}"));
                let p_fast = self.val(&format!("inttoptr i64 {host} to ptr"));
                if let Access::Store { .. } = access {
                    self.access_at(&p_fast, &access);
                }
                self.emit(&format!("br label %{join}"));
                self.start_block(&slow);
                let p_slow = match access {
                    Access::Load { .. } => Some(self.slow_read_ptr(addr, bytes)),
                    Access::Store { ty, value } => {
                        self.slow_store(addr, bytes, ty, value);
                        None
                    }
                };
                self.emit(&format!("br label %{join}"));
                self.start_block(&join);
                if let Some(p_slow) = p_slow {
                    let ptr = self.val(&format!(
                        "phi ptr [ {p_fast}, %{fast} ], [ {p_slow}, %{slow} ]"
                    ));
                    self.access_at(&ptr, &access)
                } else {
                    None
                }
            }
            MemoryModel::Windowed(w) if w.check == WindowCheck::Select => {
                let (inside, delta) = self.window_select_delta(addr, w);
                if bytes <= 1 || self.contiguous_runs() {
                    let host = self.val(&format!("add i64 {addr}, {delta}"));
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                    let v = self.access_at(&ptr, &access);
                    if let Access::Store { .. } = access {
                        self.select_marks(&inside, addr, w, bytes);
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
                self.start_block(&fast);
                let host = self.val(&format!("add i64 {addr}, {delta}"));
                let p_fast = self.val(&format!("inttoptr i64 {host} to ptr"));
                if let Access::Store { .. } = access {
                    self.access_at(&p_fast, &access);
                    self.select_marks(&inside, addr, w, 1);
                }
                self.emit(&format!("br label %{join}"));
                self.start_block(&slow);
                let p_slow = match access {
                    Access::Load { .. } => Some(self.slow_read_ptr(addr, bytes)),
                    Access::Store { ty, value } => {
                        self.slow_store(addr, bytes, ty, value);
                        None
                    }
                };
                self.emit(&format!("br label %{join}"));
                self.start_block(&join);
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
                self.start_block(&flat);
                let host = self.flat_host(addr, w.flat);
                let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                let v_flat = self.access_at(&ptr, &access);
                self.emit(&format!("br label %{join}"));
                self.start_block(&win);
                let v_win = self.window_outlined(addr, bytes, w.base, &access);
                self.emit(&format!("br label %{join}"));
                self.start_block(&join);
                match (access, v_flat, v_win) {
                    (Access::Load { ty }, Some(vf), Some(vw)) => {
                        Some(self.val(&format!("phi {ty} [ {vf}, %{flat} ], [ {vw}, %{win} ]")))
                    }
                    _ => None,
                }
            }
            MemoryModel::Windowed(w) if matches!(access, Access::Store { .. }) => {
                self.window_store(addr, bytes, w, &access);
                None
            }
            MemoryModel::Windowed(w) => {
                let (inside, rel) = self.window_test(addr, w);
                let pred = self.current_block.clone();
                let flat_base = self.flat_base_val(w.flat);
                if bytes <= 1 || self.contiguous_runs() {
                    let win = self.new_label("gw");
                    let join = self.new_label("gj");
                    self.emit(&format!(
                        "br i1 {inside}, label %{win}, label %{join}, {UNLIKELY}"
                    ));
                    self.start_block(&win);
                    let index = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                    let delta_win = self.table_delta(WINDOW_TABLE_SYMBOL, &index);
                    self.emit(&format!("br label %{join}"));
                    self.start_block(&join);
                    let delta = self.val(&format!(
                        "phi i64 [ {flat_base}, %{pred} ], [ {delta_win}, %{win} ]"
                    ));
                    let host = self.val(&format!("add i64 {addr}, {delta}"));
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
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
                self.start_block(&win);
                let offset = self.val(&format!("and i64 {addr}, {}", SLOT_BYTES - 1));
                let fits = self.val(&format!("icmp ule i64 {offset}, {}", SLOT_BYTES - bytes));
                self.emit(&format!(
                    "br i1 {fits}, label %{wtbl}, label %{slow}, {LIKELY}"
                ));
                self.start_block(&wtbl);
                let index = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                let delta_win = self.table_delta(WINDOW_TABLE_SYMBOL, &index);
                self.emit(&format!("br label %{fast}"));
                self.start_block(&fast);
                let delta = self.val(&format!(
                    "phi i64 [ {flat_base}, %{pred} ], [ {delta_win}, %{wtbl} ]"
                ));
                let host = self.val(&format!("add i64 {addr}, {delta}"));
                let p_fast = self.val(&format!("inttoptr i64 {host} to ptr"));
                if let Access::Store { .. } = access {
                    self.access_at(&p_fast, &access);
                }
                self.emit(&format!("br label %{join}"));
                self.start_block(&slow);
                let p_slow = match access {
                    Access::Load { .. } => Some(self.slow_read_ptr(addr, bytes)),
                    Access::Store { ty, value } => {
                        self.slow_store(addr, bytes, ty, value);
                        None
                    }
                };
                self.emit(&format!("br label %{join}"));
                self.start_block(&join);
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
        self.start_block(&flat);
        let host_flat = self.flat_host(addr, w.flat);
        let p_flat = self.val(&format!("inttoptr i64 {host_flat} to ptr"));
        self.access_at(&p_flat, access);
        self.emit(&format!("br label %{join}"));
        self.start_block(&win);
        let straddle_checked = bytes > 1 && !self.contiguous_runs();
        let slow = if straddle_checked {
            let table = self.new_label("gt");
            let slow = self.new_label("gs");
            let offset = self.val(&format!("and i64 {addr}, {}", SLOT_BYTES - 1));
            let fits = self.val(&format!("icmp ule i64 {offset}, {}", SLOT_BYTES - bytes));
            self.emit(&format!(
                "br i1 {fits}, label %{table}, label %{slow}, {LIKELY}"
            ));
            self.start_block(&table);
            Some(slow)
        } else {
            None
        };
        let index = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
        let delta = self.table_delta(WINDOW_TABLE_SYMBOL, &index);
        let host = self.val(&format!("add i64 {addr}, {delta}"));
        let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
        self.access_at(&ptr, access);
        self.window_mark(&index);
        if bytes > 1 && self.contiguous_runs() {
            // The slot of the last byte needs its own mark only when the store crosses into it
            // (rare). Re-marking the same slot is a second atomic store (an `xchg` on x86 hosts)
            // that the consumer cannot tell apart from the first: skip it behind a cold branch.
            let last = self.val(&format!("add i64 {rel}, {}", bytes - 1));
            let last_index = self.val(&format!("lshr i64 {last}, {SLOT_SHIFT}"));
            let crosses = self.val(&format!("icmp ne i64 {last_index}, {index}"));
            let mark_last = self.new_label("gm");
            self.emit(&format!(
                "br i1 {crosses}, label %{mark_last}, label %{join}, {UNLIKELY}"
            ));
            self.start_block(&mark_last);
            self.window_mark(&last_index);
        }
        self.emit(&format!("br label %{join}"));
        if let (Some(slow), Access::Store { ty, value }) = (slow, access) {
            self.start_block(&slow);
            self.slow_store(addr, bytes, ty, value);
            self.emit(&format!("br label %{join}"));
        }
        self.start_block(&join);
    }

    /// O37: marks of a store under the branch-free window check: the slot (the spare entry
    /// outside the window) and, when runs are kept contiguous and the store is wider than a byte,
    /// the slot of its last byte.
    fn select_marks(&mut self, inside: &str, addr: &str, w: Window, bytes: u64) {
        let rel = self.val(&format!("sub i64 {addr}, {}", imm(w.base, 64)));
        let raw = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
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

    /// Notes that `value` is a stack-pointer-relative guest address (see `stack_flat`).
    pub(super) fn note_stack_value(&mut self, value: &str) {
        if self.stack_flat().is_some() {
            self.stack_values.insert(value.to_string());
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
                self.start_block(&fast);
                let mut ptrs = Vec::new();
                for a in addrs {
                    let host = self.flat_host(a, w.flat);
                    ptrs.push(self.val(&format!("inttoptr i64 {host} to ptr")));
                }
                emit(self, Some(&ptrs));
                self.emit(&format!("br label %{join}"));
                self.start_block(&slow);
                emit(self, None);
                self.emit(&format!("br label %{join}"));
                self.start_block(&join);
                Ok(true)
            }
        }
    }
}
