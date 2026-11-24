//! Guest memory accesses under every [`MemoryModel`] (the translation seam; models and rules:
//! `lift/memory.rs`). With [`MemoryModel::Base`] every function here emits exactly the IR the
//! lifter emitted before the table models existed.

use super::{FunctionLifter, R, imm};
use crate::lift::memory::{
    Decl, HELPER_GUEST_MEMMOVE, HELPER_GUEST_MEMSET, HELPER_GUEST_READ, HELPER_GUEST_SCRATCH,
    HELPER_GUEST_WRITE, MemoryModel, PAGE_TABLE_SYMBOL, SLOT_BYTES, SLOT_SHIFT,
    WINDOW_TABLE_SYMBOL, Window, WindowCheck, window_helper,
};
use crate::lift::{GUEST_BASE_SYMBOL, GuestBase, Support};

/// `!prof` of a branch whose first successor is rarely taken.
const UNLIKELY: &str = "!prof !{!\"branch_weights\", i32 1, i32 2000}";
/// `!prof` of a branch whose first successor is almost always taken.
const LIKELY: &str = "!prof !{!\"branch_weights\", i32 2000, i32 1}";

/// The block all paths of an access branch to, with the phi inputs of a load.
struct Join {
    label: String,
    incoming: Vec<(String, String)>,
}

impl Join {
    fn new(label: String) -> Self {
        Self {
            label,
            incoming: Vec::new(),
        }
    }
}

/// One guest access: what to do with the translated address.
enum Access<'v> {
    Load { ty: &'v str },
    Store { ty: &'v str, value: &'v str },
}

impl FunctionLifter<'_> {
    /// `host = addr + base` as an `i64` (today's translation).
    fn flat_host(&mut self, addr: &str, base: GuestBase) -> String {
        match base {
            GuestBase::Constant(0) => addr.to_string(),
            GuestBase::Constant(base) => self.val(&format!("add i64 {addr}, {}", imm(base, 64))),
            GuestBase::Symbol => {
                self.support.insert(Support::GuestBase);
                let base = self.val(&format!(
                    "load i64, ptr @{GUEST_BASE_SYMBOL}, align 8, !invariant.load !{{}}"
                ));
                self.val(&format!("add i64 {addr}, {base}"))
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
                let win = self.new_label("gw");
                let flat = self.new_label("gf");
                let join = self.new_label("gj");
                self.emit(&format!(
                    "br i1 {inside}, label %{win}, label %{flat}, {UNLIKELY}"
                ));
                self.start_block(&flat);
                let host_flat = self.flat_host(addr, w.flat);
                self.emit(&format!("br label %{join}"));
                self.start_block(&win);
                let index = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                let delta = self.table_delta(WINDOW_TABLE_SYMBOL, &index);
                let host_win = self.val(&format!("add i64 {addr}, {delta}"));
                self.emit(&format!("br label %{join}"));
                self.start_block(&join);
                self.val(&format!(
                    "phi i64 [ {host_flat}, %{flat} ], [ {host_win}, %{win} ]"
                ))
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
        let flat = self.flat_host("0", w.flat);
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
        let int = format!("i{}", bytes * 8);
        let width = u8::try_from(bytes).unwrap_or(u8::MAX);
        if bytes > 1 {
            self.support.insert(Support::Memory(Decl::Read));
            self.support.insert(Support::Memory(Decl::Write));
        }
        match access {
            Access::Load { ty } => {
                self.support
                    .insert(Support::Memory(Decl::WindowLoad { bytes: width, base }));
                let name = window_helper(false, bytes, base);
                let raw = self.val(&format!("call {int} @{name}(i64 {addr})"));
                Some(if *ty == int {
                    raw
                } else {
                    self.val(&format!("bitcast {int} {raw} to {ty}"))
                })
            }
            Access::Store { ty, value } => {
                self.support
                    .insert(Support::Memory(Decl::WindowStore { bytes: width, base }));
                let name = window_helper(true, bytes, base);
                let raw = if *ty == int {
                    (*value).to_string()
                } else {
                    self.val(&format!("bitcast {ty} {value} to {int}"))
                };
                self.emit(&format!("call void @{name}(i64 {addr}, {int} {raw})"));
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

    /// The straddle-safe slow path: through the runtime's per-thread scratch buffer.
    fn access_slow(&mut self, addr: &str, bytes: u64, access: &Access<'_>) -> Option<String> {
        match access {
            Access::Load { .. } => {
                self.support.insert(Support::Memory(Decl::Read));
                let scratch = self.val(&format!(
                    "call ptr @{HELPER_GUEST_READ}(i64 {addr}, i64 {bytes})"
                ));
                self.access_at(&scratch, access)
            }
            Access::Store { .. } => {
                self.support.insert(Support::Memory(Decl::Write));
                let scratch = self.val(&format!("call ptr @{HELPER_GUEST_SCRATCH}()"));
                self.access_at(&scratch, access);
                self.emit(&format!(
                    "call void @{HELPER_GUEST_WRITE}(i64 {addr}, i64 {bytes})"
                ));
                None
            }
        }
    }

    /// Table path of an access: `delta` is the slot's delta. Accesses wider than one byte check
    /// whether they stay inside the slot and take the slow path otherwise. Every path ends in a
    /// branch to `join` (loads add their value to the phi inputs); `current` is the open block.
    fn table_access(
        &mut self,
        addr: &str,
        delta: &str,
        bytes: u64,
        access: &Access<'_>,
        join: &mut Join,
        current: &str,
    ) {
        let (incoming, join) = (&mut join.incoming, join.label.as_str());
        if bytes <= 1 || self.contiguous_runs() {
            let host = self.val(&format!("add i64 {addr}, {delta}"));
            let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
            if let Some(v) = self.access_at(&ptr, access) {
                incoming.push((v, current.to_string()));
            }
            self.emit(&format!("br label %{join}"));
            return;
        }
        let fast = self.new_label("gt");
        let slow = self.new_label("gs");
        let offset = self.val(&format!("and i64 {addr}, {}", SLOT_BYTES - 1));
        let fits = self.val(&format!("icmp ule i64 {offset}, {}", SLOT_BYTES - bytes));
        self.emit(&format!(
            "br i1 {fits}, label %{fast}, label %{slow}, {LIKELY}"
        ));
        self.start_block(&fast);
        let host = self.val(&format!("add i64 {addr}, {delta}"));
        let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
        if let Some(v) = self.access_at(&ptr, access) {
            incoming.push((v, fast.clone()));
        }
        self.emit(&format!("br label %{join}"));
        self.start_block(&slow);
        if let Some(v) = self.access_slow(addr, bytes, access) {
            incoming.push((v, slow.clone()));
        }
        self.emit(&format!("br label %{join}"));
    }

    fn guest_access(&mut self, addr: &str, bytes: u64, access: Access<'_>) -> Option<String> {
        let mut incoming: Vec<(String, String)> = Vec::new();
        let join = match self.memory {
            MemoryModel::Base(_) => {
                let ptr = self.guest_ptr(addr);
                return self.access_at(&ptr, &access);
            }
            _ if self.stack_flat().is_some() && self.stack_values.contains(addr) => {
                let base = self.stack_flat().unwrap_or_default();
                let host = self.flat_host(addr, base);
                let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                return self.access_at(&ptr, &access);
            }
            MemoryModel::PageTable { pages, .. } => {
                let index = self.page_index(addr, pages);
                let delta = self.table_delta(PAGE_TABLE_SYMBOL, &index);
                let pre = self.new_label("gp");
                let mut join = Join::new(self.new_label("gj"));
                self.emit(&format!("br label %{pre}"));
                self.start_block(&pre);
                self.table_access(addr, &delta, bytes, &access, &mut join, &pre);
                incoming = join.incoming;
                join.label
            }
            MemoryModel::Windowed(w) if w.check == WindowCheck::Select => {
                let (inside, delta) = self.window_select_delta(addr, w);
                if bytes <= 1 || self.contiguous_runs() {
                    let host = self.val(&format!("add i64 {addr}, {delta}"));
                    let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                    return self.access_at(&ptr, &access);
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
                let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                if let Some(v) = self.access_at(&ptr, &access) {
                    incoming.push((v, fast.clone()));
                }
                self.emit(&format!("br label %{join}"));
                self.start_block(&slow);
                if let Some(v) = self.access_slow(addr, bytes, &access) {
                    incoming.push((v, slow.clone()));
                }
                self.emit(&format!("br label %{join}"));
                join
            }
            MemoryModel::Windowed(w) => {
                let (inside, rel) = self.window_test(addr, w);
                let win = self.new_label("gw");
                let flat = self.new_label("gf");
                let join = self.new_label("gj");
                self.emit(&format!(
                    "br i1 {inside}, label %{win}, label %{flat}, {UNLIKELY}"
                ));
                self.start_block(&flat);
                let host = self.flat_host(addr, w.flat);
                let ptr = self.val(&format!("inttoptr i64 {host} to ptr"));
                if let Some(v) = self.access_at(&ptr, &access) {
                    incoming.push((v, flat.clone()));
                }
                self.emit(&format!("br label %{join}"));
                self.start_block(&win);
                if w.check == WindowCheck::Outline {
                    if let Some(v) = self.window_outlined(addr, bytes, w.base, &access) {
                        incoming.push((v, win.clone()));
                    }
                    self.emit(&format!("br label %{join}"));
                } else {
                    let index = self.val(&format!("lshr i64 {rel}, {SLOT_SHIFT}"));
                    let delta = self.table_delta(WINDOW_TABLE_SYMBOL, &index);
                    let mut paths = Join {
                        label: join.clone(),
                        incoming: std::mem::take(&mut incoming),
                    };
                    self.table_access(addr, &delta, bytes, &access, &mut paths, &win);
                    incoming = paths.incoming;
                }
                join
            }
        };
        self.start_block(&join);
        let Access::Load { ty } = access else {
            return None;
        };
        let inputs: Vec<String> = incoming
            .iter()
            .map(|(v, block)| format!("[ {v}, %{block} ]"))
            .collect();
        Some(self.val(&format!("phi {ty} {}", inputs.join(", "))))
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
