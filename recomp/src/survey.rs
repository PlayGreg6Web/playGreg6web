//! Interface-call survey: which vtable slots does the binary call on interface pointers obtained
//! from a given source, how often and with how many arguments.
//!
//! The game talks to the platform through COM-style interfaces. This module answers, from the
//! discovered CFG alone, "for the interface pointers that come out of source X, which vtable
//! slots are called?" so the runtime knows which methods it has to implement.
//!
//! # Sources
//!
//! A [`Source`] says where interface pointers come from:
//!
//! - [`SourceKind::Import`] `DLL!Name`: an import whose N-th argument is an out-pointer (or whose
//!   return value) is the interface, e.g. `d3d12_xs.dll!D3D12XboxCreateDevice` argument 3.
//! - [`SourceKind::Proc`]: a function pointer obtained from `GetProcAddress(module, "Name")`,
//!   stored to a global and called later (the `QueryApiImpl` pattern).
//! - [`SourceKind::Function`]: a function of the image at a given address.
//! - [`SourceKind::Method`] `<source>:<slot>`: a vtable slot of a previously defined source whose
//!   out-pointer yields a new interface (device -> `CreateCommandQueue` -> queue).
//! - [`SourceKind::Global`]: a global known to hold an interface pointer.
//!
//! `key=` lists argument indices (0 = `rcx`) whose values separate interfaces of one source;
//! `lea reg,[rip+x]` arguments are resolved to the 16 bytes at `x` in a read-only section and
//! printed as a GUID. That, and comparing the string passed to `GetProcAddress` with the
//! requested export name, are the only places data bytes are read; nothing but GUIDs is printed.
//!
//! # Method
//!
//! A forward abstract interpretation per function ([`engine`]) tracks interface pointers through
//! register moves, spills to and reloads from stack slots, object fields of incoming pointers
//! (until the next call) and globals (flow-insensitively, shared by all functions and iterated to
//! a fixpoint). A vtable call is `mov r,[iface]` followed by a call or jump through
//! `[r+disp]` (slot = `disp / 8`), also through a register loaded from `[vtable+disp]`, and
//! through Control Flow Guard thunks: a call through a data slot (or into a one-instruction
//! `jmp rax` / `jmp [slot]` thunk) while `rax` holds a method pointer is a dispatch, while `rcx`
//! holding it is a check call that preserves all registers. The guard slots are detected
//! automatically and reported; `--guard-slot` forces more.
//!
//! Inter-procedurally, interface pointers passed in `rcx`/`rdx`/`r8`/`r9` to a discovered
//! function (direct calls and tail calls) are propagated into it, up to [`Config::max_depth`]
//! levels. In the other direction, a function that returns an interface pointer in `rax` or
//! stores one through an incoming argument (`mov [rcx],r`: an out-parameter) hands it to its
//! direct callers (into `rax`, or into the stack slot, global or global-object field they passed),
//! and tail calls forward both. Pointers stored into a field of an object whose pointer is kept in
//! a global (`mov r,[g]; mov [r+disp],iface`) are followed like globals and listed as global
//! object fields. A function containing a vtable call is a "wrapper"; `caller_sites` counts the
//! direct calls and tail calls that reach those wrapper functions.
//!
//! Struct-returning methods take the hidden result pointer in `rcx` and `this` in `rdx`; a call
//! whose vtable came from the interface in `rdx` while `rcx` is a stack address is confirmed and
//! marked with the `sret` evidence kind.
//!
//! # Honesty
//!
//! - `rcx` must be the same interface the vtable was loaded from, otherwise the call is
//!   reported as *unconfirmed* and kept out of the confirmed counts.
//! - The inferred argument count is a lower bound: `1 +` the highest of `rdx`/`r8`/`r9`/`xmm1-3`
//!   written (and stack argument slots `[rsp+0x20..]` stored) since the last call on every path
//!   to the site. A register written for another reason inflates it, an argument set up in a
//!   callee-saved register earlier cannot be seen. Each site carries its evidence kinds.
//! - Confidence: `high` when every confirmed site used a pointer obtained locally, `medium` when
//!   some came through a global, a global-object field, a callee's result or a caller-supplied
//!   argument, `low` when only unconfirmed sites exist.
//! - Everything the analysis cannot follow is counted by reason in the unresolved bucket
//!   ([`engine::Reason`]) instead of being guessed. Stack slots are forgotten only at the exact
//!   address passed to a call, memory writes through unknown pointers do not alias tracked
//!   slots, and conditional flow merges lose a pointer that is not held on all paths.

pub mod cli;
mod engine;
pub mod report;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use crate::cfg::{CallTarget, Edge};
use crate::code_image::CodeImage;
use crate::discover::Analysis;
use crate::pe::{IMAGE_SCN_MEM_EXECUTE, IMAGE_SCN_MEM_WRITE, ImportedSymbol, Pe64};
use engine::{Binding, Engine, KeyId, NO_KEY, Plan, Prov, Reason};

pub use engine::{MAX_ARG_INDEX, MAX_SLOT};

/// Rounds of the global-fact fixpoint.
const MAX_ROUNDS: usize = 16;
/// Upper bound of analysed (function, binding) contexts per round.
const MAX_CONTEXTS: usize = 2_000_000;

/// Name of an imported symbol.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImportRef {
    Name(String),
    Ordinal(u16),
}

impl fmt::Display for ImportRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(n) => f.write_str(n),
            Self::Ordinal(o) => write!(f, "#{o}"),
        }
    }
}

/// Which value of a source call is the interface pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutParam {
    /// Out-pointer in argument `n` (0 = `rcx`; `this` for method sources).
    Arg(u8),
    /// Returned in `rax`.
    Ret,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SourceKind {
    Import { dll: String, name: ImportRef },
    Proc { name: String },
    Function { entry: u64 },
    Method { parent: String, slot: u32 },
    Global { va: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub name: String,
    pub kind: SourceKind,
    /// `None` only for [`SourceKind::Global`].
    pub out: Option<OutParam>,
    /// Argument indices whose values key the produced interface.
    pub key_args: Vec<u8>,
}

impl Source {
    /// One-line description, as printed in reports.
    pub fn describe(&self) -> String {
        let mut s = match &self.kind {
            SourceKind::Import { dll, name } => format!("import {dll}!{name}"),
            SourceKind::Proc { name } => format!("GetProcAddress \"{name}\""),
            SourceKind::Function { entry } => format!("function 0x{entry:x}"),
            SourceKind::Method { parent, slot } => format!("method {parent}:{slot}"),
            SourceKind::Global { va } => format!("global 0x{va:x}"),
        };
        match self.out {
            Some(OutParam::Arg(i)) => s.push_str(&format!(", out arg {i}")),
            Some(OutParam::Ret) => s.push_str(", out ret"),
            None => {}
        }
        if !self.key_args.is_empty() {
            let keys: Vec<String> = self.key_args.iter().map(u8::to_string).collect();
            s.push_str(&format!(", key args {}", keys.join("+")));
        }
        s
    }
}

/// Parses a hexadecimal virtual address with or without `0x`.
pub fn parse_hex(text: &str) -> Result<u64, String> {
    let digits = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))
        .unwrap_or(text);
    u64::from_str_radix(digits, 16)
        .map_err(|_| format!("expected a hexadecimal virtual address, got '{text}'"))
}

/// Parses `name=<label>,<kind>=<value>[,out=<n>|ret][,key=<i>[+<j>...]]` where `<kind>` is one
/// of `import=<dll>!<name|#ordinal>`, `proc=<export>`, `fn=<va>`, `method=<source>:<slot>`,
/// `global=<va>`.
pub fn parse_source_spec(spec: &str) -> Result<Source, String> {
    let mut name = None;
    let mut kind = None;
    let mut out = None;
    let mut key_args = Vec::new();
    let set_kind = |k: SourceKind, kind: &mut Option<SourceKind>| -> Result<(), String> {
        if kind.is_some() {
            return Err(format!(
                "source '{spec}' has several of import=, proc=, fn=, method=, global="
            ));
        }
        *kind = Some(k);
        Ok(())
    };
    for part in spec.split(',') {
        let (k, v) = part
            .split_once('=')
            .ok_or_else(|| format!("source option '{part}' is not key=value"))?;
        match k {
            "name" => name = Some(v.to_string()),
            "import" => {
                let (dll, sym) = v
                    .split_once('!')
                    .ok_or_else(|| format!("import '{v}' must be <dll>!<name>"))?;
                let name = match sym.strip_prefix('#') {
                    Some(o) => ImportRef::Ordinal(
                        o.parse()
                            .map_err(|_| format!("bad import ordinal '{sym}'"))?,
                    ),
                    None => ImportRef::Name(sym.to_string()),
                };
                set_kind(
                    SourceKind::Import {
                        dll: dll.to_string(),
                        name,
                    },
                    &mut kind,
                )?;
            }
            "proc" => set_kind(
                SourceKind::Proc {
                    name: v.to_string(),
                },
                &mut kind,
            )?,
            "fn" => set_kind(
                SourceKind::Function {
                    entry: parse_hex(v)?,
                },
                &mut kind,
            )?,
            "global" => set_kind(SourceKind::Global { va: parse_hex(v)? }, &mut kind)?,
            "method" => {
                let (parent, slot) = v
                    .rsplit_once(':')
                    .ok_or_else(|| format!("method '{v}' must be <source>:<slot>"))?;
                let slot: u32 = slot.parse().map_err(|_| format!("bad slot in '{v}'"))?;
                if slot >= MAX_SLOT {
                    return Err(format!("slot {slot} is out of range (< {MAX_SLOT})"));
                }
                set_kind(
                    SourceKind::Method {
                        parent: parent.to_string(),
                        slot,
                    },
                    &mut kind,
                )?;
            }
            "out" => {
                out = Some(if v == "ret" {
                    OutParam::Ret
                } else {
                    let i: u8 = v.parse().map_err(|_| format!("bad out argument '{v}'"))?;
                    if i > MAX_ARG_INDEX {
                        return Err(format!(
                            "out argument {i} is too large (max {MAX_ARG_INDEX})"
                        ));
                    }
                    OutParam::Arg(i)
                });
            }
            "key" => {
                for p in v.split('+') {
                    let i: u8 = p.parse().map_err(|_| format!("bad key argument '{p}'"))?;
                    if i > MAX_ARG_INDEX {
                        return Err(format!(
                            "key argument {i} is too large (max {MAX_ARG_INDEX})"
                        ));
                    }
                    key_args.push(i);
                }
            }
            other => return Err(format!("unknown source option '{other}'")),
        }
    }
    let kind = kind.ok_or_else(|| {
        format!("source '{spec}' needs one of import=, proc=, fn=, method=, global=")
    })?;
    let is_global = matches!(kind, SourceKind::Global { .. });
    if is_global && out.is_some() {
        return Err("a global source takes no out=".to_string());
    }
    if !is_global && out.is_none() {
        return Err(format!("source '{spec}' needs out=<arg index>|ret"));
    }
    let name = name.ok_or_else(|| format!("source '{spec}' needs name=<label>"))?;
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
    {
        return Err(format!("source name '{name}' must match [A-Za-z0-9_.-]+"));
    }
    Ok(Source {
        name,
        kind,
        out,
        key_args,
    })
}

/// Survey settings.
#[derive(Clone, Debug)]
pub struct Config {
    pub sources: Vec<Source>,
    /// Data slots to treat as Control Flow Guard dispatch/check slots in addition to the ones
    /// detected automatically.
    pub guard_slots: Vec<u64>,
    /// Inter-procedural depth: how many calls deep interface arguments are followed.
    pub max_depth: u8,
    /// Example call-site addresses listed per slot.
    pub examples: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sources: Vec::new(),
            guard_slots: Vec::new(),
            max_depth: 3,
            examples: 5,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportEntry {
    /// Lower-case DLL name as written in the import directory.
    pub dll: String,
    pub name: ImportRef,
    /// VA of the IAT slot.
    pub slot: u64,
}

/// What the survey needs to know about the PE besides the code image.
#[derive(Clone, Debug, Default)]
pub struct Env {
    pub imports: Vec<ImportEntry>,
    /// `[start, end)` VA ranges of non-writable, non-executable sections (GUID constants).
    pub readonly: Vec<(u64, u64)>,
}

fn dll_key(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    lower.strip_suffix(".dll").unwrap_or(&lower).to_string()
}

impl Env {
    pub fn from_pe(pe: &Pe64) -> Self {
        let base = pe.optional.image_base;
        let mut imports = Vec::new();
        let mut add = |dll: &str, iat_rva: u32, symbols: &[ImportedSymbol]| {
            if iat_rva == 0 {
                return;
            }
            for (i, s) in symbols.iter().enumerate() {
                let name = match s {
                    ImportedSymbol::ByName { name, .. } => ImportRef::Name(name.clone()),
                    ImportedSymbol::ByOrdinal { ordinal } => ImportRef::Ordinal(*ordinal),
                };
                imports.push(ImportEntry {
                    dll: dll.to_ascii_lowercase(),
                    name,
                    slot: base + u64::from(iat_rva) + 8 * i as u64,
                });
            }
        };
        for m in &pe.imports {
            add(&m.dll_name, m.iat_rva, &m.symbols);
        }
        for m in &pe.delay_imports {
            add(&m.dll_name, m.iat_rva, &m.symbols);
        }
        let readonly = pe
            .sections
            .iter()
            .filter(|s| s.characteristics & (IMAGE_SCN_MEM_WRITE | IMAGE_SCN_MEM_EXECUTE) == 0)
            .map(|s| {
                let lo = base + u64::from(s.virtual_address);
                (lo, lo + u64::from(s.virtual_size.max(s.raw_size)))
            })
            .collect();
        Self { imports, readonly }
    }

    /// IAT slot of `dll!name` (the `.dll` suffix and case of `dll` are ignored).
    pub fn find_import(&self, dll: &str, name: &ImportRef) -> Option<u64> {
        let want = dll_key(dll);
        self.imports
            .iter()
            .find(|e| dll_key(&e.dll) == want && &e.name == name)
            .map(|e| e.slot)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SurveyError {
    NoSources,
    DuplicateSource(String),
    UnknownParent { source: String, parent: String },
    ImportNotFound(String),
    NoGetProcAddress,
}

impl fmt::Display for SurveyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSources => f.write_str("no --source given"),
            Self::DuplicateSource(n) => write!(f, "source name '{n}' is defined twice"),
            Self::UnknownParent { source, parent } => write!(
                f,
                "source '{source}' refers to '{parent}', which is not defined before it"
            ),
            Self::ImportNotFound(i) => write!(f, "import '{i}' not found in the PE"),
            Self::NoGetProcAddress => f.write_str("proc= source needs a GetProcAddress import"),
        }
    }
}

impl std::error::Error for SurveyError {}

/// One component of a key: a GUID read from the image, a constant, or unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArgKey {
    Guid([u8; 16]),
    Const(u64),
    Unknown,
}

impl fmt::Display for ArgKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Guid(b) => {
                let d1 = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let d2 = u16::from_le_bytes([b[4], b[5]]);
                let d3 = u16::from_le_bytes([b[6], b[7]]);
                write!(
                    f,
                    "{{{d1:08x}-{d2:04x}-{d3:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}}}",
                    b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
                )
            }
            Self::Const(c) => write!(f, "{c:#x}"),
            Self::Unknown => f.write_str("?"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    Low,
    Medium,
    High,
}

impl Confidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Evidence kinds of the argument-count lower bound, joined with `+`.
pub fn evidence_str(bits: u8) -> String {
    let mut parts = Vec::new();
    if bits & engine::EV_REGS != 0 {
        parts.push("regs");
    }
    if bits & engine::EV_XMM != 0 {
        parts.push("xmm");
    }
    if bits & engine::EV_STACK != 0 {
        parts.push("stack");
    }
    if parts.is_empty() {
        parts.push("this_only");
    }
    if bits & engine::EV_TAIL != 0 {
        parts.push("tail");
    }
    if bits & engine::EV_SRET != 0 {
        parts.push("sret");
    }
    parts.join("+")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotReport {
    pub slot: u32,
    /// Distinct call sites whose `this` was proven.
    pub sites: u32,
    /// Distinct call sites whose `this` could not be proven.
    pub unconfirmed_sites: u32,
    /// Distinct functions containing confirmed sites.
    pub functions: u32,
    /// Direct calls and tail calls that reach those functions.
    pub caller_sites: u32,
    /// Lower bound of the argument count including `this`, minimum and maximum over the sites.
    pub min_args: u8,
    pub max_args: u8,
    /// Evidence kinds over the sites (see [`evidence_str`]).
    pub evidence: u8,
    pub confidence: Confidence,
    pub examples: Vec<u64>,
    pub unconfirmed_examples: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyReport {
    pub key_args: Vec<ArgKey>,
    pub source_calls: u32,
    pub source_functions: u32,
    pub out_stack: u32,
    pub out_global: u32,
    pub out_ret: u32,
    /// Globals that hold interface pointers of this key.
    pub globals: Vec<u64>,
    /// `(global, disp)`: fields of objects pointed to by a global that hold pointers of this key.
    pub global_fields: Vec<(u64, i64)>,
    pub slots: Vec<SlotReport>,
    pub unresolved: BTreeMap<&'static str, u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceReport {
    pub name: String,
    pub description: String,
    pub keys: Vec<KeyReport>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub keys: u32,
    pub source_calls: u32,
    /// Distinct `(key, slot)` pairs with a confirmed site.
    pub slots: u32,
    /// Distinct confirmed `(key, slot, site)` triples.
    pub sites: u32,
    pub unconfirmed_sites: u32,
    /// Distinct functions containing a confirmed site.
    pub functions: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub functions_discovered: u32,
    pub functions_analyzed: u32,
    pub contexts: u32,
    pub rounds: u32,
    /// Functions whose dataflow exceeded its step budget.
    pub aborted: u32,
    pub context_limit_hit: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Survey {
    pub sources: Vec<SourceReport>,
    pub totals: Totals,
    /// Unresolved uses by reason over all sources.
    pub unresolved: BTreeMap<&'static str, u32>,
    /// Control Flow Guard slots used for dispatch or check calls (detected and forced).
    pub guard_slots: Vec<u64>,
    pub stats: Stats,
}

fn resolve_plan(env: &Env, config: &Config) -> Result<Plan, SurveyError> {
    if config.sources.is_empty() {
        return Err(SurveyError::NoSources);
    }
    let mut names: HashMap<&str, usize> = HashMap::new();
    let mut import_sources = HashMap::new();
    let mut fn_sources = HashMap::new();
    let mut proc_names = Vec::new();
    let mut method_parent = Vec::new();
    for (i, s) in config.sources.iter().enumerate() {
        if names.insert(&s.name, i).is_some() {
            return Err(SurveyError::DuplicateSource(s.name.clone()));
        }
        let mut parent = None;
        match &s.kind {
            SourceKind::Import { dll, name } => {
                let slot = env
                    .find_import(dll, name)
                    .ok_or_else(|| SurveyError::ImportNotFound(format!("{dll}!{name}")))?;
                import_sources.insert(slot, i);
            }
            SourceKind::Proc { name } => proc_names.push((name.clone(), i)),
            SourceKind::Function { entry } => {
                fn_sources.insert(*entry, i);
            }
            SourceKind::Method { parent: p, .. } => {
                // Defined before: the chain is acyclic by construction.
                let idx = config.sources[..i].iter().position(|x| &x.name == p);
                parent = Some(idx.ok_or_else(|| SurveyError::UnknownParent {
                    source: s.name.clone(),
                    parent: p.clone(),
                })?);
            }
            SourceKind::Global { .. } => {}
        }
        method_parent.push(parent);
    }
    let getproc_slots: Vec<u64> = env
        .imports
        .iter()
        .filter(|e| e.name == ImportRef::Name("GetProcAddress".to_string()))
        .map(|e| e.slot)
        .collect();
    if !proc_names.is_empty() && getproc_slots.is_empty() {
        return Err(SurveyError::NoGetProcAddress);
    }
    Ok(Plan {
        config: config.clone(),
        import_sources,
        fn_sources,
        getproc_slots,
        proc_names,
        import_slots: env.imports.iter().map(|e| (e.slot, ())).collect(),
        method_parent,
    })
}

/// Runs the survey over `analysis`.
pub fn survey(
    env: &Env,
    image: &CodeImage<'_>,
    analysis: &Analysis,
    config: &Config,
) -> Result<Survey, SurveyError> {
    let plan = resolve_plan(env, config)?;
    let mut engine = Engine::new(&plan, env, image, analysis);
    for (i, s) in config.sources.iter().enumerate() {
        if let SourceKind::Global { va } = s.kind {
            let key = engine.facts.intern_public(i, Vec::new());
            engine.facts.add_iface_global(va, key);
        }
    }

    let mut stats = Stats {
        functions_discovered: analysis.functions.len() as u32,
        ..Stats::default()
    };
    let mut analysed: BTreeSet<u64> = BTreeSet::new();
    for round in 1..=MAX_ROUNDS {
        let before = engine.facts.generation;
        let mut queue: BTreeMap<(u64, Binding), u8> = engine
            .seeds()
            .into_iter()
            .map(|e| ((e, Binding::default()), 0))
            .collect();
        let mut visited: HashSet<(u64, Binding)> = HashSet::new();
        while let Some(((entry, binding), depth)) = queue.pop_first() {
            if !visited.insert((entry, binding)) {
                continue;
            }
            if visited.len() > MAX_CONTEXTS {
                stats.context_limit_hit = true;
                break;
            }
            let Some(function) = analysis.function(entry) else {
                continue;
            };
            analysed.insert(entry);
            engine.analyze(function, binding, depth);
            let pending: Vec<(u64, Binding)> = std::mem::take(&mut engine.pending);
            for (t, b) in pending {
                if !visited.contains(&(t, b)) {
                    queue.entry((t, b)).or_insert(depth + 1);
                }
            }
        }
        stats.rounds = round as u32;
        stats.contexts = visited.len() as u32;
        if engine.facts.generation == before {
            break;
        }
    }
    stats.functions_analyzed = analysed.len() as u32;
    stats.aborted = engine.aborted as u32;
    Ok(build_report(&plan, &engine, analysis, stats))
}

fn caller_counts(analysis: &Analysis, targets: &BTreeSet<u64>) -> BTreeMap<u64, u32> {
    let mut out: BTreeMap<u64, u32> = BTreeMap::new();
    if targets.is_empty() {
        return out;
    }
    for f in analysis.functions.values() {
        for block in f.blocks.values() {
            for c in &block.calls {
                if let CallTarget::Direct(t) = c.target
                    && targets.contains(&t)
                {
                    *out.entry(t).or_default() += 1;
                }
            }
            for e in &block.successors {
                if let Edge::TailCall(t) = e
                    && targets.contains(t)
                {
                    *out.entry(*t).or_default() += 1;
                }
            }
        }
    }
    out
}

fn build_report(plan: &Plan, engine: &Engine<'_>, analysis: &Analysis, stats: Stats) -> Survey {
    let facts = &engine.facts;
    // Keys that appear anywhere.
    let mut used: BTreeSet<KeyId> = BTreeSet::new();
    used.extend(facts.source_calls.keys().map(|(k, _)| *k));
    used.extend(facts.uses.keys().map(|(k, _, _)| *k));
    used.extend(facts.iface_globals.values().flatten().copied());
    used.extend(facts.iface_fields.values().flatten().copied());
    used.extend(
        facts
            .unresolved
            .iter()
            .map(|(_, k, _)| *k)
            .filter(|k| *k != NO_KEY),
    );

    let wrappers: BTreeSet<u64> = facts
        .uses
        .values()
        .filter(|u| u.confirmed)
        .map(|u| u.function)
        .collect();
    let callers = caller_counts(analysis, &wrappers);

    let mut totals = Totals::default();
    let mut all_unresolved: BTreeMap<&'static str, u32> = BTreeMap::new();
    for (reason, _, _) in &facts.unresolved {
        *all_unresolved.entry(reason.as_str()).or_default() += 1;
    }

    let examples = plan.config.examples;
    let mut by_source: Vec<Vec<(KeyId, KeyReport)>> = vec![Vec::new(); plan.config.sources.len()];
    for &key in &used {
        let info = &facts.keys[key as usize];
        let mut kr = KeyReport {
            key_args: info.args.clone(),
            source_calls: 0,
            source_functions: 0,
            out_stack: 0,
            out_global: 0,
            out_ret: 0,
            globals: facts
                .iface_globals
                .iter()
                .filter(|(_, ks)| ks.contains(&key))
                .map(|(g, _)| *g)
                .collect(),
            global_fields: facts
                .iface_fields
                .iter()
                .filter(|(_, ks)| ks.contains(&key))
                .map(|(f, _)| *f)
                .collect(),
            slots: Vec::new(),
            unresolved: BTreeMap::new(),
        };
        let mut sfns = BTreeSet::new();
        for (_, rec) in facts.source_calls.range((key, 0)..=(key, u64::MAX)) {
            kr.source_calls += 1;
            sfns.insert(rec.function);
            match rec.out {
                engine::OutKind::Stack => kr.out_stack += 1,
                engine::OutKind::Global => kr.out_global += 1,
                engine::OutKind::Ret => kr.out_ret += 1,
            }
        }
        kr.source_functions = sfns.len() as u32;
        for (reason, k, _) in &facts.unresolved {
            if *k == key {
                *kr.unresolved.entry(reason.as_str()).or_default() += 1;
            }
        }
        // Slots.
        let mut slots: BTreeMap<u32, Vec<(u64, &engine::UseRec)>> = BTreeMap::new();
        for ((_, slot, site), u) in facts.uses.range((key, 0, 0)..=(key, u32::MAX, u64::MAX)) {
            slots.entry(*slot).or_default().push((*site, u));
        }
        for (slot, uses) in slots {
            let confirmed: Vec<_> = uses.iter().filter(|(_, u)| u.confirmed).collect();
            let unconfirmed: Vec<_> = uses.iter().filter(|(_, u)| !u.confirmed).collect();
            let basis: Vec<_> = if confirmed.is_empty() {
                unconfirmed.iter().collect()
            } else {
                confirmed.iter().collect()
            };
            let fns: BTreeSet<u64> = confirmed.iter().map(|(_, u)| u.function).collect();
            let confidence = if confirmed.is_empty() {
                Confidence::Low
            } else if confirmed.iter().all(|(_, u)| u.prov == Prov::Local) {
                Confidence::High
            } else {
                Confidence::Medium
            };
            let sr = SlotReport {
                slot,
                sites: confirmed.len() as u32,
                unconfirmed_sites: unconfirmed.len() as u32,
                functions: fns.len() as u32,
                caller_sites: fns
                    .iter()
                    .map(|f| callers.get(f).copied().unwrap_or(0))
                    .sum(),
                min_args: basis.iter().map(|(_, u)| u.args).min().unwrap_or(0),
                max_args: basis.iter().map(|(_, u)| u.args).max().unwrap_or(0),
                evidence: basis.iter().fold(0, |a, (_, u)| a | u.evidence),
                confidence,
                examples: confirmed.iter().take(examples).map(|(s, _)| *s).collect(),
                unconfirmed_examples: unconfirmed.iter().take(examples).map(|(s, _)| *s).collect(),
            };
            totals.sites += sr.sites;
            totals.unconfirmed_sites += sr.unconfirmed_sites;
            if sr.sites > 0 {
                totals.slots += 1;
            }
            kr.slots.push(sr);
        }
        totals.keys += 1;
        totals.source_calls += kr.source_calls;
        by_source[info.source].push((key, kr));
    }
    totals.functions = wrappers.len() as u32;

    let sources = plan
        .config
        .sources
        .iter()
        .zip(by_source)
        .map(|(s, mut keys)| {
            keys.sort_by(|a, b| a.1.key_args.cmp(&b.1.key_args).then(a.0.cmp(&b.0)));
            SourceReport {
                name: s.name.clone(),
                description: s.describe(),
                keys: keys.into_iter().map(|(_, k)| k).collect(),
            }
        })
        .collect();
    let _ = Reason::ALL;
    Survey {
        sources,
        totals,
        unresolved: all_unresolved,
        guard_slots: facts.guard_slots.iter().copied().collect(),
        stats,
    }
}
