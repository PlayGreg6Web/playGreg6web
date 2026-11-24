//! Shim families, the runtime's export-registration tables and the shim audit.
//!
//! * [`SHIM_FAMILIES`] / [`shim_families_for_dll`]: which `pg6_<family>_*` shims may implement an
//!   import of a DLL, in priority order, and how names are matched inside each family.
//! * [`scan_tables`]: reads the export tables the runtime registers in the module registry
//!   (`register_export`/`register_exports` callers) from runtime sources: every `(DLL, export
//!   name, ordinal, symbol)` tuple, whatever the table's spelling (`E(TAG, "Name", symbol)`
//!   X-macros whose tag names a `#define`d DLL constant, `{kDll, "Name", proc_address(&symbol)}`
//!   arrays, `export_entry("Name", &symbol, ordinal)` lists passed to `register_exports("DLL",
//!   ...)`). These tables are what `GetProcAddress` resolves through, so they are the
//!   authoritative export-name -> symbol and ordinal -> export maps for the static bridge too.
//! * The audit tables of `runtime/os/shim_audit.h` (`PG6_SHIM_AUDIT`, `PG6_IMPORT_GAPS`).
//! * [`parse_import_list`]: the import list of `docs/imports_required.md`.

use std::collections::BTreeMap;

use super::{Tok, dll_key, tokenize_keep_directives};
use crate::pe::ImportedSymbol;

// ---------------------------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------------------------

/// How a family maps an export name to one of its shims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameRule {
    /// The export tables first; failing that, the name key ([`super::name_key`]: ASCII-lowercase,
    /// underscores dropped) of the shim name after the prefix.
    TablesThenKey,
    /// The export tables only (exact export names). For families whose export names are not
    /// derivable from the symbol (`_exit` vs `exit`, C++-decorated names, ordinals).
    TablesOnly,
}

/// One shim family of the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShimFamily {
    /// Symbol prefix (`pg6_crt_`).
    pub prefix: &'static str,
    pub rule: NameRule,
    /// Every shim of the family needs a `PG6_SHIM_AUDIT` row to bind: `ok` binds, `deny` and
    /// `pending` leave the import a logging thunk with the recorded reason.
    pub audited: bool,
    /// Where the family lives (documentation).
    pub header: &'static str,
}

pub const PREFIX_WIN32: &str = "pg6_win32_";
pub const PREFIX_GDK: &str = "pg6_gdk_";
pub const PREFIX_D3D12: &str = "pg6_d3d12_";
pub const PREFIX_CRT: &str = "pg6_crt_";
pub const PREFIX_MW: &str = "pg6_mw_";
pub const PREFIX_XBOX: &str = "pg6_xbox_";
pub const PREFIX_WS2: &str = "pg6_ws2_";

/// Every shim family the bridge knows.
pub const SHIM_FAMILIES: &[ShimFamily] = &[
    ShimFamily {
        prefix: PREFIX_WIN32,
        rule: NameRule::TablesThenKey,
        audited: false,
        header: "os/win32_*.h, os/xbox_offline.h (table: os/shim_exports.cpp)",
    },
    ShimFamily {
        prefix: PREFIX_GDK,
        rule: NameRule::TablesThenKey,
        audited: false,
        header: "os/xgameruntime.h (table: os/shim_exports.cpp)",
    },
    ShimFamily {
        prefix: PREFIX_D3D12,
        rule: NameRule::TablesThenKey,
        audited: false,
        header: "gpu/d3d12_xs.h",
    },
    ShimFamily {
        prefix: PREFIX_CRT,
        rule: NameRule::TablesOnly,
        audited: true,
        header: "os/ucrt_bridge.h (table: os/ucrt_bridge.cpp PG6_UCRT_EXPORTS)",
    },
    ShimFamily {
        prefix: PREFIX_MW,
        rule: NameRule::TablesOnly,
        audited: true,
        header: "os/xbox_middleware.h (table: os/xbox_middleware.cpp)",
    },
    ShimFamily {
        prefix: PREFIX_XBOX,
        rule: NameRule::TablesOnly,
        audited: true,
        header: "os/xbox_scarlett.h (table: os/xbox_scarlett.cpp)",
    },
    ShimFamily {
        prefix: PREFIX_WS2,
        rule: NameRule::TablesOnly,
        audited: true,
        header: "os/winsock_ordinals.h (table: os/winsock_ordinals.cpp, names and ordinals)",
    },
];

/// The family with symbol prefix `prefix`.
pub fn family(prefix: &str) -> Option<&'static ShimFamily> {
    SHIM_FAMILIES.iter().find(|f| f.prefix == prefix)
}

/// The family a shim symbol belongs to.
pub fn family_of_symbol(symbol: &str) -> Option<&'static ShimFamily> {
    SHIM_FAMILIES.iter().find(|f| symbol.starts_with(f.prefix))
}

#[derive(Debug, Clone, Copy)]
enum DllMatch {
    Exact(&'static str),
    Prefix(&'static str),
}

/// DLL (key: lowercase, `.dll` dropped) -> families in priority order. First matching row wins.
const DLL_ROUTES: &[(DllMatch, &[&str])] = &[
    // C runtime: the UCRT API sets, ucrtbase and the VC++ runtime.
    (DllMatch::Prefix("api-ms-win-crt-"), &[PREFIX_CRT]),
    (DllMatch::Exact("ucrtbase"), &[PREFIX_CRT]),
    (DllMatch::Exact("vcruntime140"), &[PREFIX_CRT]),
    (DllMatch::Exact("vcruntime140_1"), &[PREFIX_CRT]),
    // Winsock: the ordinal family first (one coherent socket / last-error state for every
    // WS2_32 import), the O5 offline sockets for names it does not export.
    (DllMatch::Exact("ws2_32"), &[PREFIX_WS2, PREFIX_WIN32]),
    (DllMatch::Exact("kernel32"), &[PREFIX_WIN32]),
    (DllMatch::Exact("kernelbase"), &[PREFIX_WIN32]),
    (DllMatch::Exact("advapi32"), &[PREFIX_WIN32]),
    (DllMatch::Exact("winhttp"), &[PREFIX_WIN32]),
    (DllMatch::Exact("bcrypt"), &[PREFIX_WIN32]),
    (DllMatch::Exact("user32"), &[PREFIX_WIN32]),
    (DllMatch::Exact("combase"), &[PREFIX_WIN32]),
    (DllMatch::Exact("ole32"), &[PREFIX_WIN32]),
    (DllMatch::Prefix("api-ms-win-"), &[PREFIX_WIN32]),
    (DllMatch::Prefix("ext-ms-win-"), &[PREFIX_WIN32]),
    (DllMatch::Exact("xgameruntime"), &[PREFIX_GDK]),
    (DllMatch::Prefix("microsoft.xbox.services"), &[PREFIX_GDK]),
    (DllMatch::Exact("d3d12"), &[PREFIX_D3D12]),
    (DllMatch::Exact("d3d12_x"), &[PREFIX_D3D12]),
    (DllMatch::Exact("d3d12_xs"), &[PREFIX_D3D12]),
    // Xbox Series system DLLs (O12).
    (DllMatch::Exact("xmem"), &[PREFIX_XBOX]),
    (DllMatch::Exact("xcrdapi"), &[PREFIX_XBOX]),
    (DllMatch::Exact("xg_xs"), &[PREFIX_XBOX]),
    (DllMatch::Exact("gameinput"), &[PREFIX_XBOX]),
    // Middleware (O14).
    (DllMatch::Exact("libhttpclient.gdk"), &[PREFIX_MW]),
    (DllMatch::Exact("bink2scarlett"), &[PREFIX_MW]),
    (DllMatch::Exact("acphal"), &[PREFIX_MW]),
    (DllMatch::Exact("xapu"), &[PREFIX_MW]),
    (DllMatch::Exact("chad"), &[PREFIX_MW]),
    (DllMatch::Exact("amd_fidelityfx_loader_xbox"), &[PREFIX_MW]),
    (DllMatch::Exact("xboxbvhbuilder"), &[PREFIX_MW]),
    (DllMatch::Exact("mfplat"), &[PREFIX_MW]),
    (DllMatch::Exact("mfreadwrite"), &[PREFIX_MW]),
];

/// Shim family prefixes that may implement imports of `dll`, in priority order (empty when the
/// runtime has no family for it). DLL names compare case-insensitively, `.dll` optional.
pub fn shim_families_for_dll(dll: &str) -> &'static [&'static str] {
    let key = dll_key(dll);
    DLL_ROUTES
        .iter()
        .find(|(m, _)| match m {
            DllMatch::Exact(name) => key == *name,
            DllMatch::Prefix(prefix) => key.starts_with(prefix),
        })
        .map_or(&[], |(_, families)| families)
}

// ---------------------------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------------------------

/// One `(DLL, export name, ordinal, symbol)` registration of a runtime export table.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExportEntry {
    /// DLL as spelled in the table.
    pub dll: String,
    pub name: String,
    pub ordinal: Option<u16>,
    pub symbol: String,
    /// Include spelling of the scanned source.
    pub source: String,
}

/// Audit verdict of one shim (`PG6_SHIM_AUDIT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verdict {
    /// Audited: safe to call from recompiled code through a marshaling thunk. Binds.
    Ok,
    /// Audited: must not be bound (the note says why); the import stays a logging thunk.
    Deny,
    /// Not audited yet; does not bind.
    Pending,
}

impl Verdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Deny => "deny",
            Self::Pending => "pending",
        }
    }

    fn parse(word: &str) -> Option<Self> {
        match word {
            "ok" => Some(Self::Ok),
            "deny" => Some(Self::Deny),
            "pending" => Some(Self::Pending),
            _ => None,
        }
    }
}

/// One `PG6_SHIM_AUDIT` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEntry {
    pub symbol: String,
    pub verdict: Verdict,
    pub note: String,
    pub source: String,
}

/// One `PG6_IMPORT_GAPS` row: an import the runtime knowingly has no shim for, with the reason
/// (and the task that will add it). `name` is an export name, `#<ordinal>` or `*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GapEntry {
    pub dll: String,
    pub name: String,
    pub reason: String,
    pub source: String,
}

impl GapEntry {
    pub fn matches(&self, dll: &str, symbol: &ImportedSymbol) -> bool {
        if dll_key(&self.dll) != dll_key(dll) {
            return false;
        }
        match symbol {
            _ if self.name == "*" => true,
            ImportedSymbol::ByName { name, .. } => *name == self.name,
            ImportedSymbol::ByOrdinal { ordinal } => self.name == format!("#{ordinal}"),
        }
    }
}

/// Everything [`scan_tables`] found in one source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceTables {
    pub exports: Vec<ExportEntry>,
    pub audit: Vec<AuditEntry>,
    pub gaps: Vec<GapEntry>,
    pub diagnostics: Vec<String>,
}

const AUDIT_MACRO: &str = "PG6_SHIM_AUDIT";
const GAPS_MACRO: &str = "PG6_IMPORT_GAPS";

fn is_dll_name(s: &str) -> bool {
    s.len() > 4 && s.to_ascii_lowercase().ends_with(".dll")
}

/// `NAME` -> string for `#define NAME "..."`, `NAME[] = "..."` and `NAME = "..."`.
fn string_constants(toks: &[Tok]) -> BTreeMap<String, String> {
    let mut consts = BTreeMap::new();
    for (i, tok) in toks.iter().enumerate() {
        let Tok::Ident(name) = tok else { continue };
        let value = if i >= 1 && toks[i - 1].is_ident("define") {
            match toks.get(i + 1) {
                Some(Tok::Str(s)) => Some(s),
                _ => None,
            }
        } else if toks.get(i + 1).is_some_and(|t| t.is_punct('['))
            && toks.get(i + 2).is_some_and(|t| t.is_punct(']'))
            && toks.get(i + 3).is_some_and(|t| t.is_punct('='))
        {
            match toks.get(i + 4) {
                Some(Tok::Str(s)) => Some(s),
                _ => None,
            }
        } else if toks.get(i + 1).is_some_and(|t| t.is_punct('=')) {
            match (toks.get(i + 2), toks.get(i + 3)) {
                (Some(Tok::Str(s)), Some(end)) if end.is_punct(';') => Some(s),
                _ => None,
            }
        } else {
            None
        };
        if let Some(value) = value {
            consts.insert(name.clone(), value.clone());
        }
    }
    consts
}

/// A string constant by name, or by a unique `<prefix>_<name>` (token pasting: `E(K32, ...)` with
/// `#define PG6_MOD_K32 "KERNEL32.dll"` and `PG6_MOD_##mod`).
fn resolve_constant<'a>(consts: &'a BTreeMap<String, String>, name: &str) -> Option<&'a str> {
    if let Some(value) = consts.get(name) {
        return Some(value);
    }
    let suffix = format!("_{name}");
    let mut found = consts
        .iter()
        .filter(|(k, _)| k.ends_with(&suffix))
        .map(|(_, v)| v.as_str());
    let first = found.next()?;
    found.next().is_none().then_some(first)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Arg {
    Dll(String),
    Name(String),
    Ordinal(Option<u16>),
    Symbol(String),
    Other,
}

fn parse_ordinal(text: &str) -> Option<Option<u16>> {
    let digits = text.trim_end_matches(['u', 'U', 'l', 'L']);
    let value = if let Some(hex) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16).ok()?
    } else {
        digits.parse::<u32>().ok()?
    };
    let ordinal = u16::try_from(value).ok()?;
    Some((ordinal != 0).then_some(ordinal))
}

/// `pg6_x`, `&pg6_x`, `f(&pg6_x)` / `f(pg6_x)` -> `pg6_x`.
fn symbol_ref(arg: &[Tok]) -> Option<String> {
    match arg {
        [Tok::Ident(s)] if s.starts_with("pg6_") => Some(s.clone()),
        [amp, Tok::Ident(s)] if amp.is_punct('&') && s.starts_with("pg6_") => Some(s.clone()),
        [Tok::Ident(_), open, inner @ .., close] if open.is_punct('(') && close.is_punct(')') => {
            symbol_ref(inner)
        }
        _ => None,
    }
}

fn classify_arg(arg: &[Tok], consts: &BTreeMap<String, String>) -> Arg {
    if let Some(symbol) = symbol_ref(arg) {
        return Arg::Symbol(symbol);
    }
    match arg {
        [Tok::Str(s)] if is_dll_name(s) => Arg::Dll(s.clone()),
        [Tok::Str(s)] if !s.is_empty() => Arg::Name(s.clone()),
        [Tok::Num(n)] => parse_ordinal(n).map_or(Arg::Other, Arg::Ordinal),
        [Tok::Ident(id)] => match resolve_constant(consts, id) {
            Some(value) if is_dll_name(value) => Arg::Dll(value.to_string()),
            _ => Arg::Other,
        },
        _ => Arg::Other,
    }
}

/// Top-level comma-separated arguments of the group opened at `open` and closed at `close`.
fn split_args(toks: &[Tok], open: usize, close: usize) -> Vec<&[Tok]> {
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut start = open + 1;
    for j in open + 1..close {
        let tok = &toks[j];
        if tok.is_punct('(') || tok.is_punct('[') || tok.is_punct('{') {
            depth += 1;
        } else if tok.is_punct(')') || tok.is_punct(']') || tok.is_punct('}') {
            depth -= 1;
        } else if depth == 0 && tok.is_punct(',') {
            args.push(&toks[start..j]);
            start = j + 1;
        }
    }
    if start < close {
        args.push(&toks[start..close]);
    }
    args
}

/// Matching closer and enclosing opener of every `(`/`{`/`[` opener.
struct Groups {
    close: Vec<Option<usize>>,
    parent: Vec<Option<usize>>,
}

fn groups(toks: &[Tok]) -> Groups {
    let mut close = vec![None; toks.len()];
    let mut parent = vec![None; toks.len()];
    let mut stack: Vec<usize> = Vec::new();
    // Stack depth at the start of the current preprocessor directive.
    let mut directive: Option<usize> = None;
    for (i, tok) in toks.iter().enumerate() {
        if tok.is_punct('(') || tok.is_punct('{') || tok.is_punct('[') {
            parent[i] = stack.last().copied();
            stack.push(i);
        } else if tok.is_punct(')') || tok.is_punct('}') || tok.is_punct(']') {
            if directive.is_none_or(|depth| stack.len() > depth)
                && let Some(open) = stack.pop()
            {
                close[open] = Some(i);
            }
        } else if tok.is_punct('#') {
            directive.get_or_insert(stack.len());
        } else if tok.is_punct('\n') {
            // End of a directive: groups opened inside it never span it.
            let depth = directive.take().unwrap_or(0);
            while stack.len() > depth {
                if let Some(open) = stack.pop() {
                    close[open] = Some(i);
                }
            }
        }
    }
    Groups { close, parent }
}

/// Token ranges `[start, end)` of `#define <MACRO>(...)` directives.
fn macro_body(toks: &[Tok], name: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 1..toks.len() {
        if toks[i - 1].is_ident("define") && toks[i].is_ident(name) {
            let end = toks[i..]
                .iter()
                .position(|t| t.is_punct('\n'))
                .map_or(toks.len(), |p| i + p);
            out.push((i + 1, end));
        }
    }
    out
}

/// Scan one runtime source (header or `.cpp`) for export tables and audit tables.
/// `source` is its include spelling (for diagnostics and provenance).
pub fn scan_tables(text: &str, source: &str) -> SourceTables {
    let toks = tokenize_keep_directives(text);
    let consts = string_constants(&toks);
    let g = groups(&toks);
    let mut out = SourceTables::default();

    // Audit and gap tables (macro bodies), excluded from the export-table scan.
    let mut excluded = vec![false; toks.len()];
    for (start, end) in macro_body(&toks, AUDIT_MACRO) {
        excluded[start..end].iter_mut().for_each(|e| *e = true);
        for open in start..end {
            let Some(close) = g.close[open].filter(|_| toks[open].is_punct('(')) else {
                continue;
            };
            match split_args(&toks, open, close).as_slice() {
                [
                    [Tok::Ident(symbol)],
                    [Tok::Ident(verdict)],
                    [Tok::Str(note)],
                ] if symbol.starts_with("pg6_") => match Verdict::parse(verdict) {
                    Some(verdict) => out.audit.push(AuditEntry {
                        symbol: symbol.clone(),
                        verdict,
                        note: note.clone(),
                        source: source.to_string(),
                    }),
                    None => out.diagnostics.push(format!(
                        "{source}: audit row for {symbol} has unknown verdict `{verdict}`"
                    )),
                },
                _ => {}
            }
        }
    }
    for (start, end) in macro_body(&toks, GAPS_MACRO) {
        excluded[start..end].iter_mut().for_each(|e| *e = true);
        for open in start..end {
            let Some(close) = g.close[open].filter(|_| toks[open].is_punct('(')) else {
                continue;
            };
            if let [[Tok::Str(dll)], [Tok::Str(name)], [Tok::Str(reason)]] =
                split_args(&toks, open, close).as_slice()
            {
                out.gaps.push(GapEntry {
                    dll: dll.clone(),
                    name: name.clone(),
                    reason: reason.clone(),
                    source: source.to_string(),
                });
            }
        }
    }

    // Export tuples.
    let mut dll_arg_memo: BTreeMap<usize, Option<String>> = BTreeMap::new();
    let mut dll_arg = |open: usize| -> Option<String> {
        if let Some(cached) = dll_arg_memo.get(&open) {
            return cached.clone();
        }
        let found = g.close[open].and_then(|close| {
            let dlls: Vec<String> = split_args(&toks, open, close)
                .into_iter()
                .filter_map(|a| match classify_arg(a, &consts) {
                    Arg::Dll(d) => Some(d),
                    _ => None,
                })
                .collect();
            (dlls.len() == 1).then(|| dlls[0].clone())
        });
        dll_arg_memo.insert(open, found.clone());
        found
    };
    for open in 0..toks.len() {
        if excluded[open] || !(toks[open].is_punct('(') || toks[open].is_punct('{')) {
            continue;
        }
        let Some(close) = g.close[open] else { continue };
        // Cheap filter: a direct `pg6_` identifier argument somewhere inside.
        if !toks[open + 1..close]
            .iter()
            .any(|t| t.ident().is_some_and(|s| s.starts_with("pg6_")))
        {
            continue;
        }
        let args: Vec<Arg> = split_args(&toks, open, close)
            .into_iter()
            .map(|a| classify_arg(a, &consts))
            .collect();
        if args.contains(&Arg::Other) {
            continue;
        }
        let count = |f: fn(&Arg) -> bool| args.iter().filter(|a| f(a)).count();
        if count(|a| matches!(a, Arg::Symbol(_))) != 1
            || count(|a| matches!(a, Arg::Name(_))) != 1
            || count(|a| matches!(a, Arg::Dll(_))) > 1
            || count(|a| matches!(a, Arg::Ordinal(_))) > 1
        {
            continue;
        }
        let mut dll = None;
        let mut name = String::new();
        let mut symbol = String::new();
        let mut ordinal = None;
        for arg in args {
            match arg {
                Arg::Dll(d) => dll = Some(d),
                Arg::Name(n) => name = n,
                Arg::Symbol(s) => symbol = s,
                Arg::Ordinal(o) => ordinal = o,
                Arg::Other => {}
            }
        }
        // No DLL in the tuple: an enclosing call's DLL argument (`register_exports("xmem.dll",
        // {...})`), else the next DLL literal in the enclosing top-level block (an entry array
        // registered after its definition: `register_exports("WS2_32.dll", entries, n)`).
        if dll.is_none() {
            let mut outer = g.parent[open];
            let mut top = open;
            while let Some(p) = outer {
                if dll.is_none() {
                    dll = dll_arg(p);
                }
                top = p;
                outer = g.parent[p];
            }
            if dll.is_none() {
                let end = g.close[top].unwrap_or(toks.len());
                dll = toks[close + 1..end].iter().find_map(|t| match t {
                    Tok::Str(s) if is_dll_name(s) => Some(s.clone()),
                    Tok::Ident(id) => resolve_constant(&consts, id)
                        .filter(|v| is_dll_name(v))
                        .map(str::to_string),
                    _ => None,
                });
            }
        }
        match dll {
            Some(dll) => out.exports.push(ExportEntry {
                dll,
                name,
                ordinal,
                symbol,
                source: source.to_string(),
            }),
            None => out.diagnostics.push(format!(
                "{source}: export entry \"{name}\" -> {symbol} has no DLL"
            )),
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Import lists
// ---------------------------------------------------------------------------------------------

/// Imports listed in a text file: the per-DLL tables of `docs/imports_required.md`
/// (`### \`DLL\` (...)` headings, `| name | \`Symbol\` | ... |` and
/// `| ordinal | \`#N\` | ... |` rows) or plain `DLL!Name` / `DLL!#N` lines. In file order.
pub fn parse_import_list(text: &str) -> Vec<(String, ImportedSymbol)> {
    let mut out = Vec::new();
    let mut dll: Option<String> = None;
    let ticked = |s: &str| -> Option<String> {
        let start = s.find('`')? + 1;
        let end = start + s[start..].find('`')?;
        Some(s[start..end].to_string())
    };
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("### ") {
            dll = ticked(rest).filter(|d| is_dll_name(d));
            continue;
        }
        if line.starts_with("#") {
            dll = None;
            continue;
        }
        if let Some(row) = line.strip_prefix('|') {
            let Some(current) = &dll else { continue };
            let cells: Vec<&str> = row.split('|').map(str::trim).collect();
            let (Some(kind), Some(cell)) = (cells.first(), cells.get(1)) else {
                continue;
            };
            let Some(symbol) = ticked(cell) else { continue };
            match *kind {
                "name" => out.push((
                    current.clone(),
                    ImportedSymbol::ByName {
                        hint: 0,
                        name: symbol,
                    },
                )),
                "ordinal" => {
                    if let Some(ordinal) = symbol.strip_prefix('#').and_then(|o| o.parse().ok()) {
                        out.push((current.clone(), ImportedSymbol::ByOrdinal { ordinal }));
                    }
                }
                _ => {}
            }
            continue;
        }
        if let Some((d, s)) = line.split_once('!')
            && is_dll_name(d)
            && !s.is_empty()
            && !s.contains(char::is_whitespace)
        {
            let symbol = match s.strip_prefix('#').and_then(|o| o.parse().ok()) {
                Some(ordinal) => ImportedSymbol::ByOrdinal { ordinal },
                None => ImportedSymbol::ByName {
                    hint: 0,
                    name: s.to_string(),
                },
            };
            out.push((d.to_string(), symbol));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dll_routes_pick_families_in_priority_order() {
        assert_eq!(
            shim_families_for_dll("api-ms-win-crt-stdio-l1-1-0.dll"),
            [PREFIX_CRT]
        );
        assert_eq!(shim_families_for_dll("VCRUNTIME140.dll"), [PREFIX_CRT]);
        assert_eq!(shim_families_for_dll("vcruntime140_1"), [PREFIX_CRT]);
        assert_eq!(shim_families_for_dll("ucrtbase.DLL"), [PREFIX_CRT]);
        assert_eq!(
            shim_families_for_dll("WS2_32.dll"),
            [PREFIX_WS2, PREFIX_WIN32]
        );
        assert_eq!(
            shim_families_for_dll("api-ms-win-core-synch-l1-2-0.dll"),
            [PREFIX_WIN32]
        );
        assert_eq!(shim_families_for_dll("libHttpClient.GDK.dll"), [PREFIX_MW]);
        assert_eq!(shim_families_for_dll("XboxBvhBuilder.dll"), [PREFIX_MW]);
        assert_eq!(shim_families_for_dll("MFPlat.DLL"), [PREFIX_MW]);
        assert_eq!(shim_families_for_dll("GameInput.dll"), [PREFIX_XBOX]);
        assert_eq!(shim_families_for_dll("xmem"), [PREFIX_XBOX]);
        assert!(shim_families_for_dll("msvcp140.dll").is_empty());
        assert!(shim_families_for_dll("oleaut32.dll").is_empty());
        assert!(shim_families_for_dll("kernel32.dll.bak").is_empty());
        for f in SHIM_FAMILIES {
            assert_eq!(family(f.prefix), Some(f));
        }
        assert_eq!(family_of_symbol("pg6_mw_bink_close"), family(PREFIX_MW));
        assert_eq!(family_of_symbol("pg6_gthunk_qsort"), None);
    }

    const XMACRO: &str = r#"
#define PG6_UCRT_MOD_RUNTIME "api-ms-win-crt-runtime-l1-1-0.dll"
#define PG6_UCRT_MOD_VCR140 "VCRUNTIME140.dll"
#define PG6_UCRT_EXPORTS(E)                                  \
    E(RUNTIME, "_exit", pg6_crt__exit)                       \
    E(RUNTIME, "exit", pg6_crt_exit)                         \
    /* comment */                                            \
    E(VCR140, "memcpy", pg6_crt_memcpy)
#define PG6_UCRT_ENTRY(mod, name, fn) \
    Pg6UcrtExport{PG6_UCRT_MOD_##mod, name, #fn, proc_address(&fn)},
static const Pg6UcrtExport kUcrtTable[] = {PG6_UCRT_EXPORTS(PG6_UCRT_ENTRY)};
"#;

    const ARRAYS: &str = r#"
constexpr const char kDllBvh[] = "XboxBvhBuilder.dll";
const MwExportSpec* table() {
    static const MwExportSpec kExports[] = {
        {kDllBvh, "??1XboxBvhBuilder@@QEAA@XZ", raw_proc_address(&pg6_mw_bvh_builder_dtor)},
        {"MFPlat.DLL", "MFShutdown", raw_proc_address(&pg6_mw_mf_shutdown)},
    };
    return kExports;
}
size_t reg() {
    size_t total = register_exports(
        "GameInput.dll",
        {
            export_entry("GameInputCreate", &pg6_xbox_GameInputCreate, 1u),
        });
    log("not an entry", 3);
    return total;
}
uint32_t pg6_winsock_register(void) {
    std::call_once(once, [] {
        const Pg6ExportEntry entries[] = {
            export_entry("closesocket", &pg6_ws2_closesocket, 3u),
            export_entry("getaddrinfo", &pg6_ws2_getaddrinfo),
        };
        registered = register_exports("WS2_32.dll", entries, 2);
    });
    return registered;
}
static const GuestExport kGuest[] = {
    {"GameInput.dll", "GameInputCreate", 1, &pg6_gthunk_game_input_create},
};
void orphan() { use(export_entry("Lonely", &pg6_xbox_Lonely)); }
"#;

    fn entries(t: &SourceTables) -> Vec<(String, String, Option<u16>, String)> {
        t.exports
            .iter()
            .map(|e| (e.dll.clone(), e.name.clone(), e.ordinal, e.symbol.clone()))
            .collect()
    }

    fn e(
        dll: &str,
        name: &str,
        ordinal: Option<u16>,
        symbol: &str,
    ) -> (String, String, Option<u16>, String) {
        (dll.into(), name.into(), ordinal, symbol.into())
    }

    #[test]
    fn scans_x_macro_tables_with_token_pasted_dll_tags() {
        let t = scan_tables(XMACRO, "os/x.cpp");
        assert_eq!(
            entries(&t),
            [
                e(
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "_exit",
                    None,
                    "pg6_crt__exit"
                ),
                e(
                    "api-ms-win-crt-runtime-l1-1-0.dll",
                    "exit",
                    None,
                    "pg6_crt_exit"
                ),
                e("VCRUNTIME140.dll", "memcpy", None, "pg6_crt_memcpy"),
            ]
        );
        assert!(t.diagnostics.is_empty(), "{:?}", t.diagnostics);
        assert_eq!(t.exports[0].source, "os/x.cpp");
    }

    #[test]
    fn scans_struct_arrays_register_calls_and_ordinals() {
        let t = scan_tables(ARRAYS, "os/y.cpp");
        assert_eq!(
            entries(&t),
            [
                e(
                    "XboxBvhBuilder.dll",
                    "??1XboxBvhBuilder@@QEAA@XZ",
                    None,
                    "pg6_mw_bvh_builder_dtor"
                ),
                e("MFPlat.DLL", "MFShutdown", None, "pg6_mw_mf_shutdown"),
                e(
                    "GameInput.dll",
                    "GameInputCreate",
                    Some(1),
                    "pg6_xbox_GameInputCreate"
                ),
                e("WS2_32.dll", "closesocket", Some(3), "pg6_ws2_closesocket"),
                e("WS2_32.dll", "getaddrinfo", None, "pg6_ws2_getaddrinfo"),
                e(
                    "GameInput.dll",
                    "GameInputCreate",
                    Some(1),
                    "pg6_gthunk_game_input_create"
                ),
            ]
        );
        assert_eq!(t.diagnostics.len(), 1, "{:?}", t.diagnostics);
        assert!(t.diagnostics[0].contains("Lonely"));
    }

    #[test]
    fn scans_audit_and_gap_tables() {
        let text = r##"
#pragma once
// X(symbol, verdict, "note")
#define PG6_SHIM_AUDIT(X)                                          \
    X(pg6_crt_strlen, ok, "pure; char* in, size_t out")            \
    X(pg6_crt_longjmp, deny, "jmp_buf belongs to guest VCRUNTIME") \
    X(pg6_crt_fopen, pending, "")                                  \
    X(pg6_crt_bogus, maybe, "x")
#define PG6_IMPORT_GAPS(X) \
    X("d3d12_xs.dll", "*", "G24") \
    X("WS2_32.dll", "#99", "none")
"##;
        let t = scan_tables(text, "os/shim_audit.h");
        let rows: Vec<(&str, Verdict)> = t
            .audit
            .iter()
            .map(|a| (a.symbol.as_str(), a.verdict))
            .collect();
        assert_eq!(
            rows,
            [
                ("pg6_crt_strlen", Verdict::Ok),
                ("pg6_crt_longjmp", Verdict::Deny),
                ("pg6_crt_fopen", Verdict::Pending)
            ]
        );
        assert_eq!(t.audit[1].note, "jmp_buf belongs to guest VCRUNTIME");
        assert_eq!(t.diagnostics.len(), 1);
        assert!(t.exports.is_empty(), "audit rows are not export entries");
        assert_eq!(t.gaps.len(), 2);
        let any = ImportedSymbol::ByName {
            hint: 0,
            name: "D3D12XboxCreateDevice".into(),
        };
        assert!(t.gaps[0].matches("D3D12_XS.DLL", &any));
        assert!(!t.gaps[0].matches("d3d12.dll", &any));
        assert!(t.gaps[1].matches("ws2_32", &ImportedSymbol::ByOrdinal { ordinal: 99 }));
        assert!(!t.gaps[1].matches("ws2_32", &ImportedSymbol::ByOrdinal { ordinal: 9 }));
    }

    #[test]
    fn parses_markdown_and_plain_import_lists() {
        let md = "# Report\n\n## Imports\n\n### `WS2_32.dll` (2 symbols)\n\n| Kind | Symbol | Hint / Ordinal |\n| --- | --- | --- |\n| ordinal | `#111` | ordinal `111` |\n| name | `getaddrinfo` | hint `7` |\n\n### `XboxBvhBuilder.dll` (1 symbol)\n\n| name | `??1XboxBvhBuilder@@QEAA@XZ` | hint `1` |\n\n## Exports\n\n| name | `NotAnImport` | x |\nKERNEL32.dll!Sleep\nGameInput.dll!#1\n";
        let list = parse_import_list(md);
        let labels: Vec<String> = list
            .iter()
            .map(|(d, s)| match s {
                ImportedSymbol::ByName { name, .. } => format!("{d}!{name}"),
                ImportedSymbol::ByOrdinal { ordinal } => format!("{d}!#{ordinal}"),
            })
            .collect();
        assert_eq!(
            labels,
            [
                "WS2_32.dll!#111",
                "WS2_32.dll!getaddrinfo",
                "XboxBvhBuilder.dll!??1XboxBvhBuilder@@QEAA@XZ",
                "KERNEL32.dll!Sleep",
                "GameInput.dll!#1"
            ]
        );
    }
}
