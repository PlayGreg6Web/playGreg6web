//! The dataflow engine behind [`super::survey`]: an abstract interpreter over one function's CFG.
//!
//! # Domain
//!
//! Every register holds one abstract value ([`V`]): an unknown, an integer constant, the address
//! of a global, a stack address (relative to the entry `rsp`), an interface pointer of a *key*
//! (a source plus its GUID arguments), the vtable loaded from such a pointer, or a method pointer
//! loaded from that vtable. Memory is tracked in two places: 8-byte stack slots and object fields
//! (`[reg+disp]` of an opaque incoming pointer) live in the per-block [`State`]; globals and the
//! fields of objects whose pointer is held in a global (`mov r,[g]` then `[r+disp]`) are
//! flow-insensitive facts shared by all functions ([`Facts`]).
//!
//! # Results of callees
//!
//! A function that returns an interface pointer in `rax`, or stores one through an incoming
//! argument register (`[rcx]`, `[rdx]`, ...: an out-parameter), records that as a fact keyed by
//! its entry; direct calls to it then yield the pointer in `rax` or in the stack slot / global
//! the caller passed, with [`Prov::Returned`].
//!
//! # Fixpoint
//!
//! States are joined at block entries (equal values survive, anything else becomes unknown), so a
//! register changes at most twice and the iteration terminates quickly. Results are recorded in a
//! single extra pass over the fixpoint states, never during iteration, so a value that is lost at
//! a join can never produce a result.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use iced_x86::{Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpKind, Register};

use super::{ArgKey, Config, Env, OutParam, SourceKind};
use crate::cfg::{BasicBlock, Edge, Function};
use crate::code_image::CodeImage;
use crate::discover::Analysis;

pub(super) type KeyId = u32;

/// Largest vtable slot index recognised (`disp <= 0x7ff8`).
pub const MAX_SLOT: u32 = 4096;
/// Largest argument index accepted for out-pointers and key arguments.
pub const MAX_ARG_INDEX: u8 = 15;
/// Stack-slot / field entries kept per state; further stores are not tracked.
const MAX_MEM: usize = 48;
/// Number of stack argument slots (`[rsp+0x20..]`) watched for argument-count evidence.
const STACK_ARG_SLOTS: i64 = 16;

const RAX: usize = 0;
const RCX: usize = 1;
const RDX: usize = 2;
const RSP: usize = 4;
const RBP: usize = 5;
const R8: usize = 8;
const R9: usize = 9;
const ARG_REGS: [usize; 4] = [RCX, RDX, R8, R9];
/// Volatile general-purpose registers of the Windows x64 ABI.
const VOLATILE: [usize; 7] = [0, 1, 2, 8, 9, 10, 11];

/// Where an interface value came from; the worst provenance seen wins at joins.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub(super) enum Prov {
    /// Obtained and used inside one function (stack slot, register or return value).
    Local,
    /// Loaded from a global (or a field of a global object) that some function stored the
    /// pointer to.
    Global,
    /// Returned by a called function, in `rax` or through an out-parameter.
    Returned,
    /// Passed in by a caller that was analysed with the pointer bound to an argument.
    Context,
}

/// Abstract value of a register or memory slot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum V {
    Unk,
    /// Entry value of register `n`, used as the identity of an object for field tracking.
    Obj(u8),
    Const(u64),
    /// Address of a global (`lea reg, [rip+x]`).
    Addr(u64),
    /// Stack address relative to the entry `rsp`.
    Stack(i64),
    /// Value read from the import slot at this address (a function pointer).
    Import(u64),
    /// Return value of `GetProcAddress` for proc source `n`.
    Proc(u32),
    /// Function pointer variable of proc source `n`.
    FnVar(u32),
    /// Pointer loaded from the global at this address: the identity of a global object whose
    /// fields are tracked flow-insensitively.
    GPtr(u64),
    Iface(KeyId, Prov),
    Vtbl(KeyId, Prov),
    Method(KeyId, u32, Prov),
}

fn join_v(a: V, b: V) -> V {
    if a == b {
        return a;
    }
    match (a, b) {
        (V::Iface(k, p), V::Iface(l, q)) if k == l => V::Iface(k, p.max(q)),
        (V::Vtbl(k, p), V::Vtbl(l, q)) if k == l => V::Vtbl(k, p.max(q)),
        (V::Method(k, s, p), V::Method(l, t, q)) if k == l && s == t => V::Method(k, s, p.max(q)),
        _ => V::Unk,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Loc {
    Stack(i64),
    Field(u8, i64),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct State {
    regs: [V; 16],
    /// Argument registers written since the last call: bit 0..2 `rdx`, `r8`, `r9`; bit 3..5
    /// `xmm1`..`xmm3`.
    written: u8,
    /// Bit `k`: `[rsp+0x20+8k]` written since the last call.
    stack_args: u16,
    /// Tracked memory, sorted by location.
    mem: Vec<(Loc, V)>,
}

impl State {
    fn entry(binding: &Binding) -> Self {
        let mut regs = [V::Unk; 16];
        for (i, r) in regs.iter_mut().enumerate() {
            *r = V::Obj(i as u8);
        }
        regs[RSP] = V::Stack(0);
        for (i, key) in binding.0.iter().enumerate() {
            if let Some(k) = key {
                regs[ARG_REGS[i]] = V::Iface(*k, Prov::Context);
            }
        }
        Self {
            regs,
            written: 0,
            stack_args: 0,
            mem: Vec::new(),
        }
    }

    fn mem_get(&self, loc: Loc) -> V {
        match self.mem.binary_search_by(|(l, _)| l.cmp(&loc)) {
            Ok(i) => self.mem[i].1,
            Err(_) => V::Unk,
        }
    }

    fn mem_put(&mut self, loc: Loc, v: V) {
        match self.mem.binary_search_by(|(l, _)| l.cmp(&loc)) {
            Ok(i) => {
                if v == V::Unk {
                    self.mem.remove(i);
                } else {
                    self.mem[i].1 = v;
                }
            }
            Err(i) => {
                if v != V::Unk && self.mem.len() < MAX_MEM {
                    self.mem.insert(i, (loc, v));
                }
            }
        }
    }

    /// Forget every tracked stack slot overlapping `[off, off+size)`.
    fn stack_clobber(&mut self, off: i64, size: i64) {
        self.mem.retain(|(l, _)| match l {
            Loc::Stack(o) => !(*o + 8 > off && *o < off + size),
            Loc::Field(..) => true,
        });
    }

    fn stack_store(&mut self, off: i64, v: V, size: i64) {
        self.stack_clobber(off, size);
        if size == 8 {
            self.mem_put(Loc::Stack(off), v);
        }
    }

    fn rsp(&self) -> Option<i64> {
        match self.regs[RSP] {
            V::Stack(o) => Some(o),
            _ => None,
        }
    }

    fn set_rsp(&mut self, v: V) {
        self.regs[RSP] = v;
        self.stack_args = 0;
        if !matches!(v, V::Stack(_)) {
            self.mem.retain(|(l, _)| !matches!(l, Loc::Stack(_)));
        }
    }

    /// Joins `other` into `self`; returns whether `self` changed.
    fn join_from(&mut self, other: &State) -> bool {
        let mut changed = false;
        for i in 0..16 {
            let j = join_v(self.regs[i], other.regs[i]);
            if j != self.regs[i] {
                self.regs[i] = j;
                changed = true;
            }
        }
        let w = self.written & other.written;
        let s = self.stack_args & other.stack_args;
        if w != self.written || s != self.stack_args {
            self.written = w;
            self.stack_args = s;
            changed = true;
        }
        let before = self.mem.len();
        let theirs = &other.mem;
        let mut kept = Vec::with_capacity(self.mem.len());
        for (loc, v) in &self.mem {
            if let Ok(i) = theirs.binary_search_by(|(l, _)| l.cmp(loc)) {
                let j = join_v(*v, theirs[i].1);
                if j != V::Unk {
                    if j != *v {
                        changed = true;
                    }
                    kept.push((*loc, j));
                    continue;
                }
            }
            changed = true;
        }
        if kept.len() != before {
            changed = true;
        }
        self.mem = kept;
        changed
    }
}

/// Interface pointers bound to the four register arguments of an analysed callee.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub(super) struct Binding(pub [Option<KeyId>; 4]);

/// Why a use of an interface pointer could not be followed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum Reason {
    /// The out-pointer argument of a source call is not a stack or global address.
    OutParamUntracked,
    /// A global was stored interface pointers of several keys (or a key and a function pointer).
    GlobalAmbiguous,
    /// The pointer was stored into a field of an incoming object (followed only inside the
    /// storing function).
    StoredToField,
    /// The pointer was stored through a pointer of unknown provenance.
    StoredToUnknown,
    /// The pointer was passed to an import that is not a source.
    PassedToImport,
    /// The pointer was passed to a call whose target is not known.
    PassedToIndirectCall,
    /// The pointer was passed to a function beyond the inter-procedural depth limit, or to a
    /// target that is not a discovered function.
    PassedBeyondLimit,
    /// The pointer was passed as a non-`this` argument of a vtable call.
    PassedAsArgument,
    /// The pointer was returned in `rax` by a function that no discovered call reaches.
    ReturnedToCaller,
    /// The vtable slot of a call is computed (`call [vtable+reg*8]`, unaligned or huge offset).
    ComputedSlot,
    /// The dataflow of a function did not converge within its step budget.
    DataflowBudget,
}

impl Reason {
    pub(super) const ALL: [Reason; 11] = [
        Reason::OutParamUntracked,
        Reason::GlobalAmbiguous,
        Reason::StoredToField,
        Reason::StoredToUnknown,
        Reason::PassedToImport,
        Reason::PassedToIndirectCall,
        Reason::PassedBeyondLimit,
        Reason::PassedAsArgument,
        Reason::ReturnedToCaller,
        Reason::ComputedSlot,
        Reason::DataflowBudget,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::OutParamUntracked => "out_param_untracked",
            Self::GlobalAmbiguous => "global_ambiguous",
            Self::StoredToField => "stored_to_field",
            Self::StoredToUnknown => "stored_to_unknown",
            Self::PassedToImport => "passed_to_import",
            Self::PassedToIndirectCall => "passed_to_indirect_call",
            Self::PassedBeyondLimit => "passed_beyond_limit",
            Self::PassedAsArgument => "passed_as_argument",
            Self::ReturnedToCaller => "returned_to_caller",
            Self::ComputedSlot => "computed_slot",
            Self::DataflowBudget => "dataflow_budget",
        }
    }
}

/// Where a source call stored its interface pointer.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum OutKind {
    Stack,
    Global,
    Ret,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SourceCallRec {
    pub function: u64,
    pub out: OutKind,
}

/// Argument-count evidence bits of a vtable call site.
pub(super) const EV_REGS: u8 = 1;
pub(super) const EV_XMM: u8 = 2;
pub(super) const EV_STACK: u8 = 4;
pub(super) const EV_TAIL: u8 = 8;
/// Struct-return call: `rcx` is a stack address (the hidden result pointer) and `this` is in
/// `rdx`; the site counts as confirmed when `rdx` is the interface the vtable came from.
pub(super) const EV_SRET: u8 = 16;

#[derive(Clone, Copy, Debug)]
pub(super) struct UseRec {
    pub function: u64,
    /// `rcx` was provably the interface pointer in every analysis of the site.
    pub confirmed: bool,
    /// Lower bound of the argument count including `this`.
    pub args: u8,
    pub evidence: u8,
    pub prov: Prov,
}

pub(super) struct KeyInfo {
    pub source: usize,
    pub args: Vec<ArgKey>,
}

/// Facts shared by all functions; they only ever grow, so rounds converge.
#[derive(Default)]
pub(super) struct Facts {
    pub keys: Vec<KeyInfo>,
    key_ids: BTreeMap<(usize, Vec<ArgKey>), KeyId>,
    /// Global address -> interface keys stored there.
    pub iface_globals: BTreeMap<u64, BTreeSet<KeyId>>,
    /// Global address -> proc sources whose `GetProcAddress` result was stored there.
    pub fn_vars: BTreeMap<u64, BTreeSet<u32>>,
    pub source_calls: BTreeMap<(KeyId, u64), SourceCallRec>,
    pub uses: BTreeMap<(KeyId, u32, u64), UseRec>,
    pub unresolved: BTreeSet<(Reason, KeyId, u64)>,
    pub guard_slots: BTreeSet<u64>,
    /// Functions that forward their own arguments unchanged to a source (import or proc), mapped
    /// to that source: calls to them are source calls.
    pub forwarders: BTreeMap<u64, usize>,
    /// `(global, disp)` -> interface keys stored to field `disp` of the object whose pointer is
    /// held in `global`.
    pub iface_fields: BTreeMap<(u64, i64), BTreeSet<KeyId>>,
    /// `(function, binding)` -> interface keys the function returns in `rax`.
    pub returns: BTreeMap<(u64, Binding), BTreeSet<KeyId>>,
    /// `(function, argument index)` -> interface keys the function stores through that incoming
    /// argument register (`[arg+0]`).
    pub out_args: BTreeMap<(u64, u8), BTreeSet<KeyId>>,
    /// Counts facts that can change later analyses (new globals, function-pointer variables,
    /// source calls).
    pub generation: u64,
}

/// `unresolved` entries without a key use this id.
pub(super) const NO_KEY: KeyId = u32::MAX;

impl Facts {
    fn intern(&mut self, source: usize, args: Vec<ArgKey>) -> KeyId {
        if let Some(&id) = self.key_ids.get(&(source, args.clone())) {
            return id;
        }
        let id = self.keys.len() as KeyId;
        self.keys.push(KeyInfo {
            source,
            args: args.clone(),
        });
        self.key_ids.insert((source, args), id);
        id
    }

    pub(super) fn intern_public(&mut self, source: usize, args: Vec<ArgKey>) -> KeyId {
        self.intern(source, args)
    }

    pub(super) fn add_iface_global(&mut self, va: u64, key: KeyId) {
        if self.iface_globals.entry(va).or_default().insert(key) {
            self.generation += 1;
        }
    }

    fn add_iface_field(&mut self, va: u64, disp: i64, key: KeyId) {
        if self.iface_fields.entry((va, disp)).or_default().insert(key) {
            self.generation += 1;
        }
    }

    fn add_return(&mut self, function: u64, binding: Binding, key: KeyId) {
        if self
            .returns
            .entry((function, binding))
            .or_default()
            .insert(key)
        {
            self.generation += 1;
        }
    }

    fn add_out_arg(&mut self, function: u64, arg: u8, key: KeyId) {
        if self
            .out_args
            .entry((function, arg))
            .or_default()
            .insert(key)
        {
            self.generation += 1;
        }
    }

    /// The single key `function` returns for `binding` (falling back to the analysis without
    /// bound arguments); `None` when unknown or ambiguous.
    fn returned_key(&self, function: u64, binding: Binding) -> Option<KeyId> {
        let set = self
            .returns
            .get(&(function, binding))
            .or_else(|| self.returns.get(&(function, Binding::default())))?;
        (set.len() == 1).then(|| *set.first().expect("non-empty"))
    }

    fn out_arg_key(&self, function: u64, arg: u8) -> Option<KeyId> {
        let set = self.out_args.get(&(function, arg))?;
        (set.len() == 1).then(|| *set.first().expect("non-empty"))
    }

    fn add_fn_var(&mut self, va: u64, src: u32) {
        if self.fn_vars.entry(va).or_default().insert(src) {
            self.generation += 1;
        }
    }

    fn unresolved(&mut self, reason: Reason, key: KeyId, site: u64) {
        self.unresolved.insert((reason, key, site));
    }
}

/// Source description resolved against the image.
pub(super) struct Plan {
    pub config: Config,
    /// Import slot VA -> source index (sources of kind `Import`).
    pub import_sources: HashMap<u64, usize>,
    /// Function entry -> source index.
    pub fn_sources: HashMap<u64, usize>,
    /// `GetProcAddress` slots.
    pub getproc_slots: Vec<u64>,
    /// Requested export name -> proc source index.
    pub proc_names: Vec<(String, usize)>,
    /// Every import slot, name-agnostic.
    pub import_slots: HashMap<u64, ()>,
    /// Resolved parent source of each method source.
    pub method_parent: Vec<Option<usize>>,
}

impl Plan {
    pub(super) fn is_getproc(&self, slot: u64) -> bool {
        self.getproc_slots.contains(&slot)
    }
}

#[derive(Clone, Copy, Debug)]
enum Thunk {
    /// `jmp rax`
    JmpRax,
    /// `jmp [rip+slot]`
    JmpSlot(u64),
}

/// A memory operand resolved against the abstract state.
#[derive(Clone, Copy, Debug)]
enum MemRef {
    Global(u64),
    Stack(i64),
    Field(u8, i64),
    /// `[p+disp]` where `p` was loaded from the global.
    GlobalField(u64, i64),
    /// `[vtable+disp]`.
    VtblSlot(KeyId, Prov, i64),
    /// `[iface+disp]`.
    IfaceField(KeyId, Prov, i64),
    /// `[vtable+reg*scale+disp]`.
    Indexed(KeyId),
    Unknown,
}

enum Callee {
    Fn(u64),
    Slot(u64),
    DispatchThunk,
    Value(V),
    Opaque,
}

pub(super) struct Frame {
    pub entry: u64,
    pub depth: u8,
    pub binding: Binding,
}

pub(super) struct Engine<'a> {
    pub plan: &'a Plan,
    pub env: &'a Env,
    pub image: &'a CodeImage<'a>,
    pub analysis: &'a Analysis,
    pub facts: Facts,
    thunks: HashMap<u64, Thunk>,
    /// Functions reached by at least one direct call or tail call.
    called: HashSet<u64>,
    info: InstructionInfoFactory,
    /// Callee contexts found while recording, to be analysed next.
    pub pending: Vec<(u64, Binding)>,
    pub aborted: u64,
}

fn gpr(reg: Register) -> Option<usize> {
    let full = reg.full_register();
    if full.is_gpr64() {
        Some(full.number())
    } else {
        None
    }
}

fn is_immediate(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Immediate8
            | OpKind::Immediate8_2nd
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64
    )
}

fn is_iface(v: V) -> Option<KeyId> {
    match v {
        V::Iface(k, _) => Some(k),
        _ => None,
    }
}

/// Interface pointers held in the four argument registers.
fn binding_of(st: &State) -> Binding {
    let mut b = Binding::default();
    for (i, r) in ARG_REGS.iter().enumerate() {
        b.0[i] = is_iface(st.regs[*r]);
    }
    b
}

/// Argument index (0 = `rcx`) of the register whose entry value is object `id`.
fn arg_index_of(id: u8) -> Option<u8> {
    ARG_REGS
        .iter()
        .position(|&r| r == usize::from(id))
        .map(|i| i as u8)
}

impl<'a> Engine<'a> {
    pub(super) fn new(
        plan: &'a Plan,
        env: &'a Env,
        image: &'a CodeImage<'a>,
        analysis: &'a Analysis,
    ) -> Self {
        let mut thunks = HashMap::new();
        for f in analysis.functions.values() {
            if f.blocks.len() != 1 {
                continue;
            }
            let Some(block) = f.blocks.values().next() else {
                continue;
            };
            let [ins] = block.instructions.as_slice() else {
                continue;
            };
            if ins.mnemonic() != Mnemonic::Jmp {
                continue;
            }
            match ins.op0_kind() {
                OpKind::Register if ins.op0_register() == Register::RAX => {
                    thunks.insert(f.entry, Thunk::JmpRax);
                }
                OpKind::Memory if ins.is_ip_rel_memory_operand() => {
                    thunks.insert(f.entry, Thunk::JmpSlot(ins.ip_rel_memory_address()));
                }
                _ => {}
            }
        }
        let mut called = HashSet::new();
        for f in analysis.functions.values() {
            for block in f.blocks.values() {
                for c in &block.calls {
                    if let crate::cfg::CallTarget::Direct(t) = c.target {
                        called.insert(t);
                    }
                }
                for e in &block.successors {
                    if let Edge::TailCall(t) = e {
                        called.insert(*t);
                    }
                }
            }
        }
        let mut facts = Facts::default();
        facts
            .guard_slots
            .extend(plan.config.guard_slots.iter().copied());
        Self {
            plan,
            env,
            image,
            analysis,
            facts,
            thunks,
            called,
            info: InstructionInfoFactory::new(),
            pending: Vec::new(),
            aborted: 0,
        }
    }

    /// Functions worth analysing in the current round: those that reference an import source
    /// slot, `GetProcAddress`, a known function-pointer variable or interface global, or that
    /// call a source function directly (also through a one-instruction `jmp [slot]` thunk).
    pub(super) fn seeds(&self) -> Vec<u64> {
        let mut slots: HashMap<u64, ()> = HashMap::new();
        slots.extend(self.plan.import_sources.keys().map(|&k| (k, ())));
        if !self.plan.proc_names.is_empty() {
            slots.extend(self.plan.getproc_slots.iter().map(|&k| (k, ())));
        }
        slots.extend(self.facts.fn_vars.keys().map(|&k| (k, ())));
        slots.extend(self.facts.iface_globals.keys().map(|&k| (k, ())));
        slots.extend(self.facts.iface_fields.keys().map(|&(g, _)| (g, ())));
        let mut fns: HashMap<u64, ()> = self.plan.fn_sources.keys().map(|&k| (k, ())).collect();
        fns.extend(self.facts.forwarders.keys().map(|&k| (k, ())));
        // Callers of functions that hand out interface pointers.
        fns.extend(self.facts.returns.keys().map(|&(f, _)| (f, ())));
        fns.extend(self.facts.out_args.keys().map(|&(f, _)| (f, ())));
        for (entry, thunk) in &self.thunks {
            if let Thunk::JmpSlot(g) = thunk
                && slots.contains_key(g)
            {
                fns.insert(*entry, ());
            }
        }
        let mut out = Vec::new();
        for f in self.analysis.functions.values() {
            let mut hit = fns.contains_key(&f.entry);
            'blocks: for block in f.blocks.values() {
                if !fns.is_empty() {
                    for c in &block.calls {
                        if let crate::cfg::CallTarget::Direct(t) = c.target
                            && fns.contains_key(&t)
                        {
                            hit = true;
                            break 'blocks;
                        }
                    }
                    for e in &block.successors {
                        if let Edge::TailCall(t) = e
                            && fns.contains_key(t)
                        {
                            hit = true;
                            break 'blocks;
                        }
                    }
                }
                for ins in &block.instructions {
                    if ins.is_ip_rel_memory_operand()
                        && slots.contains_key(&ins.ip_rel_memory_address())
                    {
                        hit = true;
                        break 'blocks;
                    }
                }
            }
            if hit {
                out.push(f.entry);
            }
        }
        out
    }

    /// Analyses `function` with `binding` and records its facts. Returns whether it converged.
    pub(super) fn analyze(&mut self, function: &Function, binding: Binding, depth: u8) {
        let n = function.blocks.len();
        if n == 0 {
            return;
        }
        let blocks: Vec<&BasicBlock> = function.blocks.values().collect();
        let starts: Vec<u64> = blocks.iter().map(|b| b.start).collect();
        let index_of = |va: u64| starts.binary_search(&va).ok();
        let frame = Frame {
            entry: function.entry,
            depth,
            binding,
        };
        let Some(entry_idx) = index_of(function.entry) else {
            return;
        };
        let mut ins_state: Vec<Option<State>> = vec![None; n];
        ins_state[entry_idx] = Some(State::entry(&binding));
        let mut work: BTreeSet<usize> = BTreeSet::new();
        work.insert(entry_idx);
        let mut steps = 0usize;
        let budget = n * 12 + 64;
        while let Some(i) = work.pop_first() {
            steps += 1;
            if steps > budget {
                self.aborted += 1;
                self.facts
                    .unresolved(Reason::DataflowBudget, NO_KEY, function.entry);
                return;
            }
            let Some(mut st) = ins_state[i].clone() else {
                continue;
            };
            let block = blocks[i];
            for ins in &block.instructions {
                self.step(&frame, ins, &mut st, false);
            }
            for edge in &block.successors {
                let Some(t) = edge.intra_target() else {
                    continue;
                };
                let Some(j) = index_of(t) else { continue };
                let changed = match &mut ins_state[j] {
                    slot @ None => {
                        *slot = Some(st.clone());
                        true
                    }
                    Some(existing) => existing.join_from(&st),
                };
                if changed {
                    work.insert(j);
                }
            }
        }
        // Recording pass over the fixpoint.
        for (i, block) in blocks.iter().enumerate() {
            let Some(mut st) = ins_state[i].clone() else {
                continue;
            };
            for ins in &block.instructions {
                self.step(&frame, ins, &mut st, true);
            }
            for edge in &block.successors {
                if let Edge::TailCall(t) = edge {
                    self.bind_callee(&frame, *t, &st, block.terminator_site(), true);
                    self.forward_tail_results(&frame, *t, &st);
                }
            }
        }
    }

    /// A tail call hands the callee's results to this function's caller: its return value
    /// becomes ours, and an out-parameter written through one of our own (unchanged) incoming
    /// arguments is ours too.
    fn forward_tail_results(&mut self, f: &Frame, target: u64, st: &State) {
        if let Some(k) = self.facts.returned_key(target, binding_of(st)) {
            self.facts.add_return(f.entry, f.binding, k);
        }
        for (i, &r) in ARG_REGS.iter().enumerate() {
            if let V::Obj(id) = st.regs[r]
                && let Some(own) = arg_index_of(id)
                && let Some(k) = self.facts.out_arg_key(target, i as u8)
            {
                self.facts.add_out_arg(f.entry, own, k);
            }
        }
    }

    /// Effects of a direct call to `target` that the facts know about: the returned interface in
    /// `rax` and interfaces written through out-parameters. `args` are the argument registers
    /// before the call.
    fn apply_call_results(
        &mut self,
        f: &Frame,
        target: u64,
        binding: Binding,
        args: [V; 4],
        st: &mut State,
        rec: bool,
    ) {
        if let Some(k) = self.facts.returned_key(target, binding) {
            st.regs[RAX] = V::Iface(k, Prov::Returned);
        }
        for (i, arg) in args.iter().enumerate() {
            let Some(k) = self.facts.out_arg_key(target, i as u8) else {
                continue;
            };
            match *arg {
                V::Stack(o) => st.stack_store(o, V::Iface(k, Prov::Returned), 8),
                V::Addr(g) if rec => self.facts.add_iface_global(g, k),
                V::GPtr(g) if rec => self.facts.add_iface_field(g, 0, k),
                V::Obj(id) if rec => {
                    if let Some(own) = arg_index_of(id) {
                        self.facts.add_out_arg(f.entry, own, k);
                    }
                }
                _ => {}
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Transfer function
    // ---------------------------------------------------------------------------------------

    fn write_reg(&mut self, st: &mut State, idx: usize, v: V) {
        if idx == RSP {
            st.set_rsp(v);
            return;
        }
        st.regs[idx] = v;
        let bit = match idx {
            RDX => 1,
            R8 => 2,
            R9 => 4,
            _ => 0,
        };
        // A vtable or method pointer held in an argument register is scratch, not an argument.
        if matches!(v, V::Vtbl(..) | V::Method(..)) {
            st.written &= !bit;
        } else {
            st.written |= bit;
        }
    }

    fn step(&mut self, f: &Frame, ins: &Instruction, st: &mut State, rec: bool) {
        match ins.mnemonic() {
            Mnemonic::Mov => self.do_mov(f, ins, st, rec),
            Mnemonic::Lea => self.do_lea(ins, st),
            Mnemonic::Call => self.do_call(f, ins, st, rec, false),
            Mnemonic::Jmp => {
                if ins.op0_kind() != OpKind::NearBranch64 {
                    self.do_call(f, ins, st, rec, true);
                }
            }
            Mnemonic::Ret => {
                if rec && let V::Iface(k, _) = st.regs[RAX] {
                    self.facts.add_return(f.entry, f.binding, k);
                    // Followed into the callers; only a function nobody calls directly loses it.
                    if !self.called.contains(&f.entry) {
                        self.facts.unresolved(Reason::ReturnedToCaller, k, ins.ip());
                    }
                }
            }
            Mnemonic::Push => self.do_push(f, ins, st, rec),
            Mnemonic::Pop => {
                if ins.op0_kind() == OpKind::Register
                    && let Some(r) = gpr(ins.op0_register())
                    && let Some(rsp) = st.rsp()
                {
                    let v = st.mem_get(Loc::Stack(rsp));
                    st.set_rsp(V::Stack(rsp + 8));
                    self.write_reg(st, r, v);
                } else {
                    self.fallback(ins, st);
                }
            }
            Mnemonic::Leave => {
                let rbp = st.regs[RBP];
                st.set_rsp(rbp);
                if let Some(rsp) = st.rsp() {
                    let v = st.mem_get(Loc::Stack(rsp));
                    st.set_rsp(V::Stack(rsp + 8));
                    st.regs[RBP] = v;
                } else {
                    st.regs[RBP] = V::Unk;
                }
            }
            Mnemonic::Add | Mnemonic::Sub => self.do_addsub(ins, st),
            Mnemonic::Xor => {
                if ins.op0_kind() == OpKind::Register
                    && ins.op1_kind() == OpKind::Register
                    && ins.op0_register() == ins.op1_register()
                    && let Some(r) = gpr(ins.op0_register())
                    && (ins.op0_register().is_gpr32() || ins.op0_register().is_gpr64())
                {
                    self.write_reg(st, r, V::Const(0));
                } else {
                    self.fallback(ins, st);
                }
            }
            _ => self.fallback(ins, st),
        }
    }

    /// Generic effect of an instruction that is not modelled: every written register becomes
    /// unknown and every written stack slot is forgotten.
    fn fallback(&mut self, ins: &Instruction, st: &mut State) {
        let mut regs = [false; 16];
        let mut xmm = 0u8;
        let mut mem_write = false;
        {
            let info = self.info.info(ins);
            for u in info.used_registers() {
                if !matches!(
                    u.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                ) {
                    continue;
                }
                let reg = u.register();
                if let Some(i) = gpr(reg) {
                    regs[i] = true;
                } else if reg.is_vector_register() {
                    let n = reg.number();
                    if (1..=3).contains(&n) {
                        xmm |= 1 << (n + 2);
                    }
                }
            }
            for m in info.used_memory() {
                if matches!(
                    m.access(),
                    OpAccess::Write
                        | OpAccess::CondWrite
                        | OpAccess::ReadWrite
                        | OpAccess::ReadCondWrite
                ) {
                    mem_write = true;
                }
            }
        }
        if mem_write {
            let size = ins.memory_size().size().max(1) as i64;
            match self.mem_ref(ins, st) {
                MemRef::Stack(o) => st.stack_clobber(o, size),
                MemRef::Field(id, d) => {
                    st.mem.retain(|(l, _)| match l {
                        Loc::Field(i, o) => !(*i == id && *o + 8 > d && *o < d + size),
                        Loc::Stack(_) => true,
                    });
                }
                _ => {}
            }
        }
        for (i, w) in regs.iter().enumerate() {
            if *w {
                self.write_reg(st, i, V::Unk);
            }
        }
        st.written |= xmm;
    }

    fn do_mov(&mut self, f: &Frame, ins: &Instruction, st: &mut State, rec: bool) {
        match ins.op0_kind() {
            OpKind::Register => {
                let dst = ins.op0_register();
                let Some(d) = gpr(dst) else {
                    return self.fallback(ins, st);
                };
                let wide = dst.is_gpr64();
                let v = match ins.op1_kind() {
                    OpKind::Register => {
                        let src = ins.op1_register();
                        match gpr(src) {
                            Some(s) if wide && src.is_gpr64() => st.regs[s],
                            Some(s) if dst.is_gpr32() && src.is_gpr32() => match st.regs[s] {
                                V::Const(c) => V::Const(c & 0xffff_ffff),
                                _ => V::Unk,
                            },
                            _ => V::Unk,
                        }
                    }
                    OpKind::Memory => {
                        if wide && ins.memory_size().size() == 8 {
                            self.load(ins, st, rec)
                        } else {
                            V::Unk
                        }
                    }
                    k if is_immediate(k) => {
                        if wide {
                            V::Const(ins.immediate(1))
                        } else if dst.is_gpr32() {
                            V::Const(ins.immediate(1) & 0xffff_ffff)
                        } else {
                            V::Unk
                        }
                    }
                    _ => V::Unk,
                };
                self.write_reg(st, d, v);
            }
            OpKind::Memory => {
                let size = ins.memory_size().size() as i64;
                let v = match ins.op1_kind() {
                    OpKind::Register => match gpr(ins.op1_register()) {
                        Some(s) if ins.op1_register().is_gpr64() => st.regs[s],
                        _ => V::Unk,
                    },
                    k if is_immediate(k) => V::Const(ins.immediate(1)),
                    _ => V::Unk,
                };
                self.store(f, ins, st, v, size, rec);
            }
            _ => self.fallback(ins, st),
        }
    }

    fn do_lea(&mut self, ins: &Instruction, st: &mut State) {
        let dst = ins.op0_register();
        let Some(d) = gpr(dst) else {
            return self.fallback(ins, st);
        };
        let mut v = V::Unk;
        if dst.is_gpr64() {
            if ins.is_ip_rel_memory_operand() {
                v = V::Addr(ins.ip_rel_memory_address());
            } else if ins.memory_index() == Register::None && ins.segment_prefix() == Register::None
            {
                let disp = ins.memory_displacement64() as i64;
                let base = ins.memory_base();
                if base == Register::None {
                    v = V::Const(disp as u64);
                } else if let Some(b) = gpr(base) {
                    v = match st.regs[b] {
                        V::Stack(o) => V::Stack(o.wrapping_add(disp)),
                        V::Addr(a) => V::Addr(a.wrapping_add(disp as u64)),
                        V::Const(c) => V::Const(c.wrapping_add(disp as u64)),
                        _ => V::Unk,
                    };
                }
            }
        }
        self.write_reg(st, d, v);
    }

    fn do_addsub(&mut self, ins: &Instruction, st: &mut State) {
        if ins.op0_kind() == OpKind::Register
            && ins.op0_register().is_gpr64()
            && is_immediate(ins.op1_kind())
            && let Some(r) = gpr(ins.op0_register())
        {
            let imm = ins.immediate(1) as i64;
            let delta = if ins.mnemonic() == Mnemonic::Sub {
                imm.wrapping_neg()
            } else {
                imm
            };
            let v = match st.regs[r] {
                V::Stack(o) => V::Stack(o.wrapping_add(delta)),
                V::Addr(a) => V::Addr(a.wrapping_add(delta as u64)),
                V::Const(c) => V::Const(c.wrapping_add(delta as u64)),
                _ => V::Unk,
            };
            self.write_reg(st, r, v);
        } else {
            self.fallback(ins, st);
        }
    }

    fn do_push(&mut self, f: &Frame, ins: &Instruction, st: &mut State, rec: bool) {
        let _ = (f, rec);
        let v = match ins.op0_kind() {
            OpKind::Register => match gpr(ins.op0_register()) {
                Some(r) if ins.op0_register().is_gpr64() => st.regs[r],
                _ => V::Unk,
            },
            k if is_immediate(k) => V::Const(ins.immediate(0)),
            _ => V::Unk,
        };
        match st.rsp() {
            Some(rsp) => {
                st.set_rsp(V::Stack(rsp - 8));
                st.stack_store(rsp - 8, v, 8);
            }
            None => st.set_rsp(V::Unk),
        }
    }

    // ---------------------------------------------------------------------------------------
    // Memory
    // ---------------------------------------------------------------------------------------

    fn mem_ref(&self, ins: &Instruction, st: &State) -> MemRef {
        if ins.segment_prefix() != Register::None {
            return MemRef::Unknown;
        }
        if ins.is_ip_rel_memory_operand() {
            return MemRef::Global(ins.ip_rel_memory_address());
        }
        let disp = ins.memory_displacement64() as i64;
        let base = ins.memory_base();
        let index = ins.memory_index();
        if base == Register::None {
            return if index == Register::None {
                MemRef::Global(disp as u64)
            } else {
                MemRef::Unknown
            };
        }
        let Some(b) = gpr(base) else {
            return MemRef::Unknown;
        };
        if !base.is_gpr64() {
            return MemRef::Unknown;
        }
        let bv = st.regs[b];
        if index != Register::None {
            return match bv {
                V::Vtbl(k, _) => MemRef::Indexed(k),
                _ => MemRef::Unknown,
            };
        }
        match bv {
            V::Stack(o) => MemRef::Stack(o.wrapping_add(disp)),
            V::Addr(a) => MemRef::Global(a.wrapping_add(disp as u64)),
            V::GPtr(g) => MemRef::GlobalField(g, disp),
            V::Const(c) if self.image.contains(c.wrapping_add(disp as u64)) => {
                MemRef::Global(c.wrapping_add(disp as u64))
            }
            V::Obj(id) => MemRef::Field(id, disp),
            V::Iface(k, p) => MemRef::IfaceField(k, p, disp),
            V::Vtbl(k, p) => MemRef::VtblSlot(k, p, disp),
            _ => MemRef::Unknown,
        }
    }

    fn load(&mut self, ins: &Instruction, st: &State, rec: bool) -> V {
        match self.mem_ref(ins, st) {
            MemRef::Global(g) => self.load_global(g, ins.ip(), rec),
            MemRef::Stack(o) => st.mem_get(Loc::Stack(o)),
            MemRef::Field(id, d) => st.mem_get(Loc::Field(id, d)),
            MemRef::GlobalField(g, d) => match self.facts.iface_fields.get(&(g, d)) {
                Some(keys) if keys.len() == 1 => {
                    V::Iface(*keys.first().expect("non-empty"), Prov::Global)
                }
                Some(keys) => {
                    if rec {
                        let k = *keys.first().expect("non-empty");
                        self.facts.unresolved(Reason::GlobalAmbiguous, k, ins.ip());
                    }
                    V::Unk
                }
                None => V::Unk,
            },
            MemRef::VtblSlot(k, p, disp) => {
                if disp >= 0 && disp % 8 == 0 && ((disp / 8) as u64) < u64::from(MAX_SLOT) {
                    V::Method(k, (disp / 8) as u32, p)
                } else {
                    if rec {
                        self.facts.unresolved(Reason::ComputedSlot, k, ins.ip());
                    }
                    V::Unk
                }
            }
            MemRef::IfaceField(k, p, 0) => V::Vtbl(k, p),
            MemRef::Indexed(k) => {
                if rec {
                    self.facts.unresolved(Reason::ComputedSlot, k, ins.ip());
                }
                V::Unk
            }
            MemRef::IfaceField(..) | MemRef::Unknown => V::Unk,
        }
    }

    fn load_global(&mut self, g: u64, site: u64, rec: bool) -> V {
        if self.plan.import_slots.contains_key(&g) {
            return V::Import(g);
        }
        if let Some(set) = self.facts.fn_vars.get(&g) {
            if set.len() == 1 && !self.facts.iface_globals.contains_key(&g) {
                return V::FnVar(*set.first().expect("non-empty"));
            }
            if rec {
                self.facts.unresolved(Reason::GlobalAmbiguous, NO_KEY, site);
            }
            return V::Unk;
        }
        if let Some(keys) = self.facts.iface_globals.get(&g) {
            if keys.len() == 1 {
                return V::Iface(*keys.first().expect("non-empty"), Prov::Global);
            }
            if rec {
                let k = *keys.first().expect("non-empty");
                self.facts.unresolved(Reason::GlobalAmbiguous, k, site);
            }
            return V::Unk;
        }
        // Any other pointer-sized global may point to an object whose fields hold interfaces.
        V::GPtr(g)
    }

    fn store(&mut self, f: &Frame, ins: &Instruction, st: &mut State, v: V, size: i64, rec: bool) {
        let escaped = matches!(v, V::Iface(..));
        match self.mem_ref(ins, st) {
            MemRef::GlobalField(g, d) => {
                if rec
                    && size == 8
                    && let V::Iface(k, _) = v
                {
                    self.facts.add_iface_field(g, d, k);
                }
            }
            MemRef::Global(g) => {
                if rec && size == 8 {
                    match v {
                        V::Iface(k, _) => self.facts.add_iface_global(g, k),
                        V::Proc(s) | V::FnVar(s) => self.facts.add_fn_var(g, s),
                        _ => {}
                    }
                }
            }
            MemRef::Stack(o) => {
                st.stack_store(o, v, size);
                if let Some(rsp) = st.rsp() {
                    let d = o - rsp;
                    if (0x20..0x20 + 8 * STACK_ARG_SLOTS).contains(&d) {
                        st.stack_args |= 1 << ((d - 0x20) / 8);
                    }
                }
            }
            MemRef::Field(id, d) => {
                st.mem.retain(|(l, _)| match l {
                    Loc::Field(i, o) => !(*i == id && *o + 8 > d && *o < d + size),
                    Loc::Stack(_) => true,
                });
                if size == 8 {
                    st.mem_put(Loc::Field(id, d), v);
                }
                if rec && let V::Iface(k, _) = v {
                    match arg_index_of(id) {
                        // `[arg+0]`: an out-parameter, followed into the callers.
                        Some(i) if d == 0 && size == 8 => self.facts.add_out_arg(f.entry, i, k),
                        _ => self.facts.unresolved(Reason::StoredToField, k, ins.ip()),
                    }
                }
            }
            _ => {
                if rec
                    && escaped
                    && let V::Iface(k, _) = v
                {
                    self.facts.unresolved(Reason::StoredToUnknown, k, ins.ip());
                }
            }
        }
    }

    fn arg(&self, st: &State, i: u8) -> V {
        if i < 4 {
            st.regs[ARG_REGS[i as usize]]
        } else {
            match st.rsp() {
                Some(rsp) => st.mem_get(Loc::Stack(rsp + 8 * i64::from(i))),
                None => V::Unk,
            }
        }
    }

    // ---------------------------------------------------------------------------------------
    // Calls
    // ---------------------------------------------------------------------------------------

    /// Effects of a call on the state: volatile registers, argument evidence and fields die.
    fn clobber_call(&mut self, st: &mut State) {
        // An address passed in an argument register may have been written through.
        for r in ARG_REGS {
            if let V::Stack(o) = st.regs[r] {
                st.stack_clobber(o, 8);
            }
        }
        for r in VOLATILE {
            st.regs[r] = V::Unk;
        }
        st.written = 0;
        st.stack_args = 0;
        st.mem.retain(|(l, _)| matches!(l, Loc::Stack(_)));
    }

    fn resolve_callee(&mut self, ins: &Instruction, st: &State, rec: bool) -> Callee {
        match ins.op0_kind() {
            OpKind::NearBranch64 => {
                let t = ins.near_branch_target();
                match self.thunks.get(&t) {
                    Some(Thunk::JmpSlot(g)) => Callee::Slot(*g),
                    Some(Thunk::JmpRax) => Callee::DispatchThunk,
                    None => Callee::Fn(t),
                }
            }
            OpKind::Register => match gpr(ins.op0_register()) {
                Some(r) if ins.op0_register().is_gpr64() => Callee::Value(st.regs[r]),
                _ => Callee::Opaque,
            },
            OpKind::Memory => match self.mem_ref(ins, st) {
                MemRef::Global(g) => Callee::Slot(g),
                MemRef::VtblSlot(k, p, disp) => {
                    if disp >= 0 && disp % 8 == 0 && ((disp / 8) as u64) < u64::from(MAX_SLOT) {
                        Callee::Value(V::Method(k, (disp / 8) as u32, p))
                    } else {
                        if rec {
                            self.facts.unresolved(Reason::ComputedSlot, k, ins.ip());
                        }
                        Callee::Opaque
                    }
                }
                MemRef::Indexed(k) => {
                    if rec {
                        self.facts.unresolved(Reason::ComputedSlot, k, ins.ip());
                    }
                    Callee::Opaque
                }
                MemRef::Stack(o) => Callee::Value(st.mem_get(Loc::Stack(o))),
                MemRef::Field(id, d) => Callee::Value(st.mem_get(Loc::Field(id, d))),
                _ => Callee::Opaque,
            },
            _ => Callee::Opaque,
        }
    }

    fn do_call(&mut self, f: &Frame, ins: &Instruction, st: &mut State, rec: bool, tail: bool) {
        let site = ins.ip();
        let callee = self.resolve_callee(ins, st, rec);
        match callee {
            Callee::Value(V::Method(k, s, p)) => self.vtable_call(f, ins, st, rec, tail, k, s, p),
            Callee::Value(V::Import(g)) => self.import_call(f, ins, st, rec, g),
            Callee::Value(V::FnVar(s)) => self.source_call(f, ins, st, rec, self.proc_source(s)),
            Callee::Value(_) | Callee::Opaque => self.unknown_call(site, st, rec),
            Callee::Fn(t) => {
                if let Some(&s) = self.plan.fn_sources.get(&t) {
                    self.source_call(f, ins, st, rec, s);
                } else if let Some(&s) = self.facts.forwarders.get(&t) {
                    self.source_call(f, ins, st, rec, s);
                } else {
                    if rec {
                        self.bind_callee(f, t, st, site, false);
                    }
                    let binding = binding_of(st);
                    let args = ARG_REGS.map(|r| st.regs[r]);
                    self.clobber_call(st);
                    self.apply_call_results(f, t, binding, args, st, rec);
                }
            }
            Callee::DispatchThunk => match (st.regs[RAX], st.regs[RCX]) {
                (V::Method(k, s, p), _) => self.vtable_call(f, ins, st, rec, tail, k, s, p),
                (V::FnVar(s), _) => self.source_call(f, ins, st, rec, self.proc_source(s)),
                (V::Import(g), _) => self.import_call(f, ins, st, rec, g),
                (_, V::Method(..)) => {} // control-flow-guard check: registers preserved
                _ => self.unknown_call(site, st, rec),
            },
            Callee::Slot(g) => {
                let guard = self.facts.guard_slots.contains(&g);
                if !guard && self.plan.import_slots.contains_key(&g) {
                    self.import_call(f, ins, st, rec, g);
                } else if !guard
                    && let Some(set) = self.facts.fn_vars.get(&g)
                    && set.len() == 1
                {
                    let s = *set.first().expect("non-empty");
                    self.source_call(f, ins, st, rec, self.proc_source(s));
                } else {
                    let data_slot = !self.image.section_at(g).is_some_and(|s| s.executable);
                    match (st.regs[RAX], st.regs[RCX]) {
                        (V::Method(k, s, p), _) if guard || data_slot => {
                            if rec {
                                self.facts.guard_slots.insert(g);
                            }
                            self.vtable_call(f, ins, st, rec, tail, k, s, p);
                        }
                        (V::FnVar(s), _) if guard || data_slot => {
                            if rec {
                                self.facts.guard_slots.insert(g);
                            }
                            self.source_call(f, ins, st, rec, self.proc_source(s));
                        }
                        (_, V::Method(..)) if guard || data_slot => {
                            // Control-flow-guard check call: registers are preserved.
                            if rec {
                                self.facts.guard_slots.insert(g);
                            }
                        }
                        _ => self.unknown_call(site, st, rec),
                    }
                }
            }
        }
    }

    fn proc_source(&self, proc_id: u32) -> usize {
        proc_id as usize
    }

    /// Call whose target is not understood: any interface argument escapes.
    fn unknown_call(&mut self, site: u64, st: &mut State, rec: bool) {
        if rec {
            for r in ARG_REGS {
                if let V::Iface(k, _) = st.regs[r] {
                    self.facts.unresolved(Reason::PassedToIndirectCall, k, site);
                }
            }
        }
        self.clobber_call(st);
    }

    /// Call to an import that is not a source. `GetProcAddress` is evaluated for proc sources.
    fn import_call(&mut self, f: &Frame, ins: &Instruction, st: &mut State, rec: bool, slot: u64) {
        let _ = f;
        if let Some(&s) = self.plan.import_sources.get(&slot) {
            return self.source_call(f, ins, st, rec, s);
        }
        let site = ins.ip();
        let mut proc_result = None;
        if self.plan.is_getproc(slot)
            && !self.plan.proc_names.is_empty()
            && let V::Addr(name_va) = st.regs[RDX]
            && let Some(name) = self.read_c_string(name_va)
        {
            proc_result = self
                .plan
                .proc_names
                .iter()
                .find(|(n, _)| n.as_bytes() == name)
                .map(|(_, s)| *s as u32);
        }
        if rec {
            for r in ARG_REGS {
                if let V::Iface(k, _) = st.regs[r] {
                    self.facts.unresolved(Reason::PassedToImport, k, site);
                }
            }
        }
        self.clobber_call(st);
        if let Some(s) = proc_result {
            st.regs[RAX] = V::Proc(s);
        }
    }

    /// Reads a NUL-terminated string (at most 128 bytes) used only to compare it with a requested
    /// export name; it is never reported.
    fn read_c_string(&self, va: u64) -> Option<&'a [u8]> {
        let section = self.image.section_at(va)?;
        let off = (va - section.va) as usize;
        let data: &'a [u8] = section.data;
        let bytes = data.get(off..)?;
        let end = bytes.iter().take(128).position(|&b| b == 0)?;
        Some(&bytes[..end])
    }

    /// Reads the 16 bytes at `va` if they lie in a read-only section; the only place where data
    /// bytes are read for reporting (GUID keys).
    fn read_guid(&self, va: u64) -> Option<[u8; 16]> {
        if !self
            .env
            .readonly
            .iter()
            .any(|&(lo, hi)| va >= lo && va.checked_add(16).is_some_and(|e| e <= hi))
        {
            return None;
        }
        let section = self.image.section_at(va)?;
        let off = (va - section.va) as usize;
        let bytes = section.data.get(off..off + 16)?;
        bytes.try_into().ok()
    }

    fn key_args(&self, st: &State, args: &[u8]) -> Vec<ArgKey> {
        args.iter()
            .map(|&i| match self.arg(st, i) {
                V::Addr(va) => match self.read_guid(va) {
                    Some(g) => ArgKey::Guid(g),
                    None => ArgKey::Unknown,
                },
                V::Const(c) => ArgKey::Const(c),
                _ => ArgKey::Unknown,
            })
            .collect()
    }

    /// A call to source `src`: creates the key, stores the interface pointer to the out
    /// location, records the source call.
    fn source_call(&mut self, f: &Frame, ins: &Instruction, st: &mut State, rec: bool, src: usize) {
        let site = ins.ip();
        let plan = self.plan;
        let source = &plan.config.sources[src];
        let out = source.out;
        let key_args = source.key_args.clone();
        let entry_arg =
            |i: u8| i < 4 && st.regs[ARG_REGS[i as usize]] == V::Obj(ARG_REGS[i as usize] as u8);
        if let Some(OutParam::Arg(o)) = out
            && entry_arg(o)
            && key_args.iter().all(|&k| entry_arg(k))
            && matches!(
                plan.config.sources[src].kind,
                SourceKind::Import { .. } | SourceKind::Proc { .. } | SourceKind::Function { .. }
            )
        {
            // The function passes its own arguments through: it is a forwarder for the source.
            if rec && self.facts.forwarders.insert(f.entry, src).is_none() {
                self.facts.generation += 1;
            }
            self.clobber_call(st);
            return;
        }
        let args = self.key_args(st, &key_args);
        let key = self.facts.intern(src, args);
        let out_value = match out {
            Some(OutParam::Arg(i)) => Some(self.arg(st, i)),
            _ => None,
        };
        self.clobber_call(st);
        let mut kind = None;
        match (out, out_value) {
            (Some(OutParam::Ret), _) => {
                st.regs[RAX] = V::Iface(key, Prov::Local);
                kind = Some(OutKind::Ret);
            }
            (Some(OutParam::Arg(_)), Some(V::Stack(o))) => {
                st.stack_store(o, V::Iface(key, Prov::Local), 8);
                kind = Some(OutKind::Stack);
            }
            (Some(OutParam::Arg(_)), Some(V::Addr(g))) => {
                if rec {
                    self.facts.add_iface_global(g, key);
                }
                kind = Some(OutKind::Global);
            }
            _ => {
                if rec {
                    self.facts.unresolved(Reason::OutParamUntracked, key, site);
                }
            }
        }
        if rec && let Some(out) = kind {
            let rec_new = self
                .facts
                .source_calls
                .insert(
                    (key, site),
                    SourceCallRec {
                        function: f.entry,
                        out,
                    },
                )
                .is_none();
            if rec_new {
                self.facts.generation += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn vtable_call(
        &mut self,
        f: &Frame,
        ins: &Instruction,
        st: &mut State,
        rec: bool,
        tail: bool,
        key: KeyId,
        slot: u32,
        prov: Prov,
    ) {
        let site = ins.ip();
        // Struct return: hidden result pointer (a stack address) in `rcx`, `this` in `rdx`.
        let sret = is_iface(st.regs[RCX]) != Some(key)
            && is_iface(st.regs[RDX]) == Some(key)
            && matches!(st.regs[RCX], V::Stack(_));
        let confirmed = is_iface(st.regs[RCX]) == Some(key) || sret;
        let (mut args, mut evidence) = (1u8, if sret { EV_SRET } else { 0 });
        let gpr_bits = st.written & 7;
        let xmm_bits = (st.written >> 3) & 7;
        if gpr_bits != 0 {
            args = args.max(8 - gpr_bits.leading_zeros() as u8 + 1);
            evidence |= EV_REGS;
        }
        if xmm_bits != 0 {
            args = args.max(8 - xmm_bits.leading_zeros() as u8 + 1);
            evidence |= EV_XMM;
        }
        if st.stack_args != 0 {
            let top = 16 - st.stack_args.leading_zeros() as u8;
            args = args.max(4 + top);
            evidence |= EV_STACK;
        }
        if tail {
            evidence |= EV_TAIL;
        }
        // Method sources hanging off this slot of this key's source.
        let parent = self.facts.keys[key as usize].source;
        let plan = self.plan;
        let mut children: Vec<usize> = Vec::new();
        for (i, p) in plan.method_parent.iter().enumerate() {
            if *p == Some(parent)
                && matches!(
                    plan.config.sources[i].kind,
                    SourceKind::Method { slot: s, .. } if s == slot
                )
            {
                children.push(i);
            }
        }
        let mut outs = Vec::new();
        for &c in &children {
            let src = &plan.config.sources[c];
            let key_args = src.key_args.clone();
            let out = src.out;
            let args_k = self.key_args(st, &key_args);
            let child = self.facts.intern(c, args_k);
            let out_value = match out {
                Some(OutParam::Arg(i)) => Some(self.arg(st, i)),
                _ => None,
            };
            outs.push((child, out, out_value));
        }
        if rec {
            for r in [RDX, R8, R9] {
                if sret && r == RDX {
                    continue;
                }
                if let V::Iface(k, _) = st.regs[r]
                    && !outs.iter().any(|(_, _, ov)| *ov == Some(st.regs[r]))
                {
                    self.facts.unresolved(Reason::PassedAsArgument, k, site);
                }
            }
            let entry = self.facts.uses.entry((key, slot, site));
            use std::collections::btree_map::Entry;
            match entry {
                Entry::Vacant(v) => {
                    v.insert(UseRec {
                        function: f.entry,
                        confirmed,
                        args,
                        evidence,
                        prov,
                    });
                }
                Entry::Occupied(mut o) => {
                    let u = o.get_mut();
                    u.confirmed &= confirmed;
                    u.args = u.args.min(args);
                    u.evidence |= evidence;
                    u.prov = u.prov.max(prov);
                }
            }
        }
        if tail {
            return;
        }
        self.clobber_call(st);
        for (child, out, out_value) in outs {
            let mut kind = None;
            match (out, out_value) {
                (Some(OutParam::Ret), _) => {
                    st.regs[RAX] = V::Iface(child, Prov::Local);
                    kind = Some(OutKind::Ret);
                }
                (Some(OutParam::Arg(_)), Some(V::Stack(o))) => {
                    st.stack_store(o, V::Iface(child, Prov::Local), 8);
                    kind = Some(OutKind::Stack);
                }
                (Some(OutParam::Arg(_)), Some(V::Addr(g))) => {
                    if rec {
                        self.facts.add_iface_global(g, child);
                    }
                    kind = Some(OutKind::Global);
                }
                _ => {
                    if rec {
                        self.facts
                            .unresolved(Reason::OutParamUntracked, child, site);
                    }
                }
            }
            if rec && let Some(out) = kind {
                let new = self
                    .facts
                    .source_calls
                    .insert(
                        (child, site),
                        SourceCallRec {
                            function: f.entry,
                            out,
                        },
                    )
                    .is_none();
                if new {
                    self.facts.generation += 1;
                }
            }
        }
    }

    /// A call (or tail call) to the function at `target`: interface arguments are propagated
    /// into it, up to the configured depth.
    fn bind_callee(&mut self, f: &Frame, target: u64, st: &State, site: u64, tail: bool) {
        let _ = tail;
        let mut binding = Binding::default();
        let mut any = false;
        for (i, r) in ARG_REGS.iter().enumerate() {
            if let V::Iface(k, _) = st.regs[*r] {
                binding.0[i] = Some(k);
                any = true;
            }
        }
        if !any {
            return;
        }
        if f.depth < self.plan.config.max_depth && self.analysis.functions.contains_key(&target) {
            self.pending.push((target, binding));
        } else {
            for k in binding.0.iter().flatten() {
                self.facts.unresolved(Reason::PassedBeyondLimit, *k, site);
            }
        }
    }
}

trait TerminatorSite {
    fn terminator_site(&self) -> u64;
}

impl TerminatorSite for BasicBlock {
    fn terminator_site(&self) -> u64 {
        self.instructions.last().map_or(self.start, |i| i.ip())
    }
}
