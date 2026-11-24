//! Import thunk and runtime call-bridge generator.
//!
//! Recompiled guest code reaches imported functions through IAT slots (`call [rip+slot]`,
//! `jmp [rip+slot]`). This module binds every import of a parsed [`Pe64`] to the C-ABI shim of
//! the runtime that implements it and emits the glue that calls that shim from guest state:
//!
//! 1. [`Catalogue::scan`] reads the runtime shim headers and records every `extern "C"`
//!    function with its parameter and return kinds.
//! 2. [`plan_bridge`] resolves each import (normal and delay-load) to a shim or to an
//!    [`Unresolved`] reason.
//! 3. [`render_bridge`] emits a C++ header + source (one thunk per import plus a table sorted by
//!    IAT slot RVA), the default guest-state seam header and a JSON manifest. Output is
//!    deterministic: identical inputs give byte-identical files.
//!
//! # Shim catalogue
//! The catalogue is scanned from the headers passed in, not from a checked-in list: the
//! declarations are the single source of truth for both the set of shims and their signatures
//! (which the marshaling depends on), so the catalogue cannot drift from the runtime. The
//! scanner is a small tokenizer, not a C++ parser. It reads `extern "C" { ... }` blocks and
//! `extern "C"` declarations at global scope, skips namespaces, class bodies and preprocessor
//! lines, and resolves global `using`/`typedef` aliases, `struct`/`enum` names and builtin
//! types to a [`TypeKind`]. Anything it cannot classify becomes [`TypeKind::Unknown`] and makes
//! that shim unbindable instead of guessing.
//!
//! # Name mapping
//! The DLL picks the shim families that may implement it, in priority order
//! ([`shim_families_for_dll`]; DLL names compare case-insensitively, `.dll` optional):
//!
//! | DLL | Families | Name rule | Audited |
//! | :--- | :--- | :--- | :--- |
//! | API sets `api-ms-win-crt-*`, `ucrtbase`, `vcruntime140`, `vcruntime140_1` | `pg6_crt_` | tables only | yes |
//! | `ws2_32` | `pg6_ws2_`, then `pg6_win32_` | tables only / tables, then key | yes / no |
//! | `kernel32`, `kernelbase`, `advapi32`, `winhttp`, `bcrypt`, `user32`, `combase`, `ole32`, other API sets `api-ms-win-*` / `ext-ms-win-*` | `pg6_win32_` | tables, then key | no |
//! | `xgameruntime`, Xbox Live (`microsoft.xbox.services*`) | `pg6_gdk_` | tables, then key | no |
//! | `d3d12`, `d3d12_x`, `d3d12_xs` | `pg6_d3d12_` | tables, then key | no |
//! | `xmem`, `xcrdapi`, `xg_xs`, `gameinput` | `pg6_xbox_` | tables only | yes |
//! | `libhttpclient.gdk`, `bink2scarlett`, `acphal`, `xapu`, `chad`, `amd_fidelityfx_loader_xbox`, `xboxbvhbuilder`, `mfplat`, `mfreadwrite` | `pg6_mw_` | tables only | yes |
//!
//! **Export tables** are authoritative. Every file passed in (headers and `.cpp` sources) is
//! scanned for the runtime's export-registration tables ([`scan_tables`]): the `(DLL, export
//! name, ordinal, symbol)` tuples the runtime registers in the module registry for
//! `GetProcAddress` (`E(TAG, "Name", symbol)` X-macros, `{kDll, "Name", proc_address(&symbol)}`
//! arrays, `export_entry("Name", &symbol, ordinal)` lists). An import binds the family symbol a
//! table registers under its exact name: the entry under the import's own DLL when there is one,
//! else the entry under another DLL of the family (`ucrtbase.dll!malloc` takes the
//! `api-ms-win-crt-heap` entry). This is exact where name keys are not (`_exit` / `exit`,
//! C++-decorated `??0XboxBvhBuilder@@QEAA@...`, `__stdio_common_vsprintf`).
//! **Name keys** (families with "tables, then key"): when no table entry exists, the import name
//! and the shim name (prefix stripped) are compared with case and underscores ignored, so
//! `CreateFileW` binds `pg6_win32_create_file_w` and `XTaskQueueCreate` binds
//! `pg6_gdk_xtask_queue_create`. Two shims with the same key make the import ambiguous.
//! **Ordinals**: an ordinal import resolves under the export name a table registers for that
//! (DLL, ordinal) ([`Catalogue::ordinal_import_name`]: `WS2_32.dll` #3 = `closesocket` from
//! `winsock_ordinals.cpp`, `GameInput.dll` #1 = `GameInputCreate`); without one it is unresolved.
//! **Families in order**: the first family with a bindable shim wins; if none has one the first
//! family's reason is reported.
//!
//! # Shim audit
//! A shim that can be marshaled is not necessarily correct when recompiled MSVC code calls it
//! (variadic `va_list` arguments, MSVC struct layouts, host pointers returned into guest memory,
//! callbacks). Every shim of an audited family needs a row in `PG6_SHIM_AUDIT`
//! (`runtime/os/shim_audit.h`): `ok` binds; `deny` and `pending` leave the import a logging
//! thunk whose message carries the recorded reason ([`Unresolved::Denied`],
//! [`Unresolved::PendingAudit`]). `PG6_IMPORT_GAPS` records imports that knowingly have no shim
//! ([`Unresolved::Gap`]). [`Unresolved::recorded`] is the explicit reason an import stays
//! unbound; the manifest carries it as `recorded`.
//!
//! # Marshaling (Win64 ABI)
//! Parameter `i < 4` comes from `rcx rdx r8 r9` (integer and pointer kinds) or from the low lane
//! of `xmm<i>` (`float`/`double`); parameter `i >= 4` is the 8-byte stack slot at
//! `rsp + 40 + 8 * (i - 4)` (return address plus 32-byte shadow space). Integer arguments are
//! truncated to the declared type (only the low bits of a Win64 argument are defined; `bool`
//! tests the low byte), pointers are translated with `pg6_seam_host_ptr`. Results go to `rax`
//! (integers sign- or zero-extended per the declared type, pointers via `pg6_seam_guest_addr`)
//! or to the low lane of `xmm0`; `void` leaves `rax` untouched. The thunk then pops the return
//! address like `ret`. Shims that cannot be called this way are not bound: variadic functions,
//! structs passed or returned by value, unknown types and host function-pointer parameters
//! (guest callbacks would need a reverse guest trampoline).
//!
//! # Guest-state thunks
//! The runtime can implement an import completely in C++ as `extern "C" void
//! pg6_gthunk_<name>(void* state)` (`runtime/guest/guest_callbacks.h`: imports that take guest
//! callbacks, such as `CreateThread`, `qsort` or `_initterm`). For an import of a DLL with a
//! shim family whose name key matches such a guest-state thunk, the thunk takes precedence over
//! the family's shim: the generated `pg6_import_<n>` just forwards the guest state to it (it
//! marshals its own arguments and pops the return address itself). When several guest-state
//! thunks share a name key (`pg6_gthunk_exit` / `pg6_gthunk__exit`), the one whose name after
//! the prefix equals the import name binds; without such a thunk the import is ambiguous. An
//! export table entry that registers a guest-state thunk under exactly the import's DLL and name
//! (`{"GameInput.dll", "GameInputCreate", 1, &pg6_gthunk_game_input_create}` in
//! `xbox_scarlett_guest.cpp`) binds first, also for a DLL without a shim family. Guest-state
//! thunks are never audit-gated: they are written for guest callers.
//!
//! # Export thunks (dynamic `GetProcAddress` coverage)
//! The thunks above exist only for a module's static imports. Guest code that resolves an API
//! with `GetModuleHandle` + `GetProcAddress` needs a guest-callable entry for every export the
//! runtime registers, imported or not: [`plan_exports`] binds each `(DLL, export name)` of the
//! runtime's export tables under exactly the rules of a static import of it (family table,
//! export tables, audit verdict, marshalability, guest-state thunk precedence) and
//! [`render_exports`] emits one table `{dll, name, ordinal, flags, shim, thunk, reason}` with one
//! marshaling thunk per distinct shim (`playgta6-recomp export-thunks`; `build-program` emits it
//! once per program under `exports/`). Exports with no bindable shim carry the reason instead of
//! a thunk. See the `exports` module docs for the output files and `guest/export_thunks.h` for
//! the runtime side.
//!
//! # Unresolved imports
//! Every unresolved import still gets a thunk. On its first call it logs
//! `unresolved import DLL!symbol (slot 0x...)` plus the reason once through `pg6_seam_log`
//! (the runtime's `playgta6::os::log`), then calls the `[[noreturn]]`
//! `pg6_seam_trap_unresolved`, which aborts by default. An unresolved import is therefore fatal
//! on first use and harmless if never called; it never returns a made-up value, because no
//! single return value is a failure for every signature.
//!
//! # Guest-state seam
//! Everything that depends on the guest CPU state layout, guest memory or the runtime is behind
//! the functions of `pg6_guest_seam.h`. The generator writes a default seam matching the
//! lifter's `%playgta6.GuestState` (16 GPRs in x86 encoding order at offset 0, `xmm0..15` at
//! offset 160 with a 16-byte stride, guest addresses translated like lifted code does:
//! `host = __playgta6_guest_base + VA`, see `docs/wasm_abi.md` "Guest Address Space"). Only
//! top-level pointer arguments and pointer results are translated; pointers a shim stores into
//! guest memory (out-parameters, structures) are host pointers and are valid guest addresses only
//! under the identity mapping (the browser build) or when they point into the guest arena. Building the
//! thunks with `-DPG6_GUEST_SEAM_HEADER="<header>"` swaps in another seam without regenerating.
//! A seam provides `pg6_seam_get_gpr`/`pg6_seam_set_gpr`, `pg6_seam_get_xmm_lo`/
//! `pg6_seam_set_xmm_lo`, `pg6_seam_load_u64`, `pg6_seam_host_ptr`/`pg6_seam_guest_addr`
//! (null must map to null), `kPg6SeamStackArgOffset`, `pg6_seam_return`, `pg6_seam_log` and
//! `pg6_seam_trap_unresolved`.
//!
//! The loader fills each IAT slot with a value its indirect-call helper
//! (`__playgta6_indirect_call(state, target)`) maps back to the slot; `pg6_find_import_thunk`
//! then returns the thunk to run. Delay-load slots are bound the same way, so the guest's
//! delay-load helper never runs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::lift::MemoryModel;
use crate::pe::{ImportedSymbol, Pe64};

/// Generated thunk table header.
pub const HEADER_FILE: &str = "pg6_import_thunks.h";
/// Generated thunks and slot table.
pub const SOURCE_FILE: &str = "pg6_import_thunks.cpp";
/// Default guest-state seam header.
pub const SEAM_FILE: &str = "pg6_guest_seam.h";
/// Machine-readable binding manifest.
pub const MANIFEST_FILE: &str = "pg6_import_manifest.json";

/// Byte offset of the 16 GPRs in the default seam's guest state (lifter layout).
pub const SEAM_GPR_OFFSET: u64 = crate::lift::STATE_GPR_OFFSET;
/// Byte offset of `xmm0..xmm15` in the default seam's guest state (lifter layout).
pub const SEAM_XMM_OFFSET: u64 = crate::lift::STATE_XMM_OFFSET;
/// Distance between two XMM registers in the default seam's guest state.
pub const SEAM_XMM_STRIDE: u64 = crate::lift::STATE_XMM_STRIDE;
/// Offset of the fifth argument from `rsp` at callee entry: return address + shadow space.
pub const STACK_ARG_OFFSET: u64 = crate::lift::CALLEE_STACK_ARG_OFFSET;

/// Win64 integer argument registers as x86 register numbers (`rcx rdx r8 r9`).
const ARG_GPRS: [u8; 4] = [1, 2, 8, 9];
const REG_RAX: u8 = 0;
const REG_RSP: u8 = 4;

mod exports;
mod tables;

pub use exports::{
    EXPORT_HEADER_FILE, EXPORT_MANIFEST_FILE, EXPORT_SOURCE_FILE, ExportBinding, ExportCounts,
    ExportOutput, ExportPlan, ExportRow, plan_exports, render_export_coverage, render_exports,
};
pub use tables::{
    AuditEntry, ExportEntry, GapEntry, NameRule, PREFIX_CRT, PREFIX_D3D12, PREFIX_GDK, PREFIX_MW,
    PREFIX_WIN32, PREFIX_WS2, PREFIX_XBOX, SHIM_FAMILIES, ShimFamily, Verdict, family,
    family_of_symbol, parse_import_list, scan_tables, shim_families_for_dll,
};

/// Runtime-written import thunks taking the guest state (`void pg6_gthunk_<name>(void* state)`).
const PREFIX_GTHUNK: &str = "pg6_gthunk_";

/// Alias resolution depth limit (guards against `using A = B; using B = A;`).
const MAX_ALIAS_DEPTH: u8 = 16;

// ---------------------------------------------------------------------------------------------
// Catalogue
// ---------------------------------------------------------------------------------------------

/// One runtime header: its `#include` spelling and its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShimHeader {
    /// How generated code includes the header (e.g. `os/win32_kernel.h`).
    pub include: String,
    /// Header source text.
    pub text: String,
}

/// Read a header. The include spelling is the path relative to `include_root` when given and
/// the header lies below it, otherwise `<parent dir>/<file name>` (the runtime's `os/...`,
/// `gpu/...` convention).
pub fn load_shim_header(path: &Path, include_root: Option<&Path>) -> io::Result<ShimHeader> {
    let text = fs::read_to_string(path)?;
    let relative = include_root.and_then(|root| path.strip_prefix(root).ok());
    let parts: Vec<String> = match relative {
        Some(rel) => rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect(),
        None => {
            let mut parts = Vec::new();
            if let Some(dir) = path.parent().and_then(Path::file_name) {
                parts.push(dir.to_string_lossy().into_owned());
            }
            if let Some(name) = path.file_name() {
                parts.push(name.to_string_lossy().into_owned());
            }
            parts
        }
    };
    Ok(ShimHeader {
        include: parts.join("/"),
        text,
    })
}

/// How a C type crosses the guest/host boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TypeKind {
    Void,
    /// Integer, `bool`, character or enum type (passed in a GPR / 8-byte stack slot).
    Int,
    /// Data pointer (a guest address, translated with `pg6_seam_host_ptr`).
    Ptr,
    F32,
    F64,
    /// Host function pointer.
    FnPtr,
    /// Struct, class or union by value.
    Struct,
    /// Anything the scanner could not classify.
    Unknown,
}

impl TypeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Void => "void",
            Self::Int => "int",
            Self::Ptr => "ptr",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::FnPtr => "fn_ptr",
            Self::Struct => "struct",
            Self::Unknown => "unknown",
        }
    }
}

/// A parameter or return type: its spelling in the header and its kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CType {
    pub spelling: String,
    pub kind: TypeKind,
}

/// One `extern "C"` function declared by a runtime header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShimDecl {
    pub name: String,
    /// Include spelling of the declaring header.
    pub include: String,
    pub ret: CType,
    pub params: Vec<CType>,
    pub variadic: bool,
}

impl ShimDecl {
    /// Why this shim cannot be called from guest state, or `None` when it can.
    pub fn unsupported_reason(&self) -> Option<String> {
        if self.variadic {
            return Some("variadic parameter list".to_string());
        }
        match self.ret.kind {
            TypeKind::Void | TypeKind::Int | TypeKind::Ptr | TypeKind::F32 | TypeKind::F64 => {}
            TypeKind::Struct => {
                return Some(format!("returns `{}` by value", self.ret.spelling));
            }
            TypeKind::FnPtr => {
                return Some(format!(
                    "returns host function pointer `{}`",
                    self.ret.spelling
                ));
            }
            TypeKind::Unknown => {
                return Some(format!("unknown return type `{}`", self.ret.spelling));
            }
        }
        for (i, param) in self.params.iter().enumerate() {
            let why = match param.kind {
                TypeKind::Int | TypeKind::Ptr | TypeKind::F32 | TypeKind::F64 => continue,
                TypeKind::FnPtr => "is a host function pointer (guest callbacks need a trampoline)",
                TypeKind::Struct => "passes a struct by value",
                TypeKind::Void | TypeKind::Unknown => "has an unknown type",
            };
            return Some(format!("parameter {i} `{}` {why}", param.spelling));
        }
        None
    }
}

/// All `extern "C"` functions declared by a set of runtime headers, plus the export tables and
/// audit rows found in the runtime sources scanned with them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Catalogue {
    shims: BTreeMap<String, ShimDecl>,
    /// Number of files scanned (headers and sources).
    pub header_count: usize,
    /// Scanner notes (e.g. a function redeclared with a different signature).
    pub diagnostics: Vec<String>,
    /// Export-table registrations ([`tables::scan_tables`]), in scan order.
    pub exports: Vec<ExportEntry>,
    /// `PG6_SHIM_AUDIT` rows by symbol.
    pub audit: BTreeMap<String, AuditEntry>,
    /// `PG6_IMPORT_GAPS` rows.
    pub gaps: Vec<GapEntry>,
}

/// Whether an include spelling names a header (scanned for declarations); any other file (a
/// `.cpp` source) is scanned for export tables only.
fn is_header_file(include: &str) -> bool {
    let lower = include.to_ascii_lowercase();
    lower.ends_with(".h") || lower.ends_with(".hpp") || lower.ends_with(".hh")
}

impl Catalogue {
    /// Scan `headers` (all aliases and type names are collected before any signature is
    /// classified, so declaration order across headers does not matter). The first declaration
    /// of a name wins; conflicting redeclarations are reported in [`Catalogue::diagnostics`].
    /// Every file (header or source) is also scanned for export and audit tables.
    pub fn scan(headers: &[ShimHeader]) -> Self {
        let mut env = TypeEnv::default();
        let mut raw = Vec::new();
        let mut catalogue = Self {
            header_count: headers.len(),
            ..Self::default()
        };
        for (index, header) in headers.iter().enumerate() {
            if is_header_file(&header.include) {
                let tokens = tokenize(&header.text);
                scan_scope(&tokens, false, index, &mut env, &mut raw);
            }
            let found = tables::scan_tables(&header.text, &header.include);
            catalogue.exports.extend(found.exports);
            catalogue.gaps.extend(found.gaps);
            catalogue.diagnostics.extend(found.diagnostics);
            for row in found.audit {
                match catalogue.audit.get(&row.symbol) {
                    Some(existing) => catalogue.diagnostics.push(format!(
                        "audit row for {} repeated in {} (keeping {})",
                        row.symbol, row.source, existing.source
                    )),
                    None => {
                        catalogue.audit.insert(row.symbol.clone(), row);
                    }
                }
            }
        }

        for decl in raw {
            let shim = ShimDecl {
                name: decl.name,
                include: headers[decl.header].include.clone(),
                ret: env.ctype(&decl.ret),
                params: decl.params.iter().map(|p| env.ctype(p)).collect(),
                variadic: decl.variadic,
            };
            match catalogue.shims.get(&shim.name) {
                Some(existing) => {
                    if existing.ret != shim.ret
                        || existing.params != shim.params
                        || existing.variadic != shim.variadic
                    {
                        catalogue.diagnostics.push(format!(
                            "`{}` redeclared in {} with a different signature (keeping {})",
                            shim.name, shim.include, existing.include
                        ));
                    }
                }
                None => {
                    catalogue.shims.insert(shim.name.clone(), shim);
                }
            }
        }
        catalogue
    }

    /// Number of declared `extern "C"` functions.
    pub fn len(&self) -> usize {
        self.shims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shims.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&ShimDecl> {
        self.shims.get(name)
    }

    /// Declarations sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = &ShimDecl> {
        self.shims.values()
    }

    /// Declarations of one shim family (`pg6_win32_`, ...), sorted by name.
    pub fn family<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = &'a ShimDecl> {
        self.shims
            .values()
            .filter(move |s| s.name.starts_with(prefix))
    }

    /// Export name of ordinal `ordinal` of `dll`, from the runtime's export tables (an entry
    /// registering that name under that ordinal, e.g. `export_entry("GameInputCreate", ...,
    /// 1u)` under `GameInput.dll`). `None` when no table declares the ordinal or two tables
    /// disagree. Ordinal imports resolve under this name.
    pub fn ordinal_import_name(&self, dll: &str, ordinal: u16) -> Option<&str> {
        let key = dll_key(dll);
        let mut names = self
            .exports
            .iter()
            .filter(|e| e.ordinal == Some(ordinal) && dll_key(&e.dll) == key)
            .map(|e| e.name.as_str());
        let first = names.next()?;
        names.all(|n| n == first).then_some(first)
    }

    /// Recorded gap reason for an import without a shim (`PG6_IMPORT_GAPS`).
    pub fn gap_for(&self, dll: &str, symbol: &ImportedSymbol) -> Option<&GapEntry> {
        self.gaps.iter().find(|g| g.matches(dll, symbol))
    }
}

// ---------------------------------------------------------------------------------------------
// Header scanner
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Ident(String),
    /// Number or character literal.
    Num(String),
    /// String literal contents (escapes kept verbatim).
    Str(String),
    Punct(char),
}

impl Tok {
    fn ident(&self) -> Option<&str> {
        match self {
            Self::Ident(s) => Some(s),
            _ => None,
        }
    }

    fn is_ident(&self, word: &str) -> bool {
        self.ident() == Some(word)
    }

    fn is_punct(&self, c: char) -> bool {
        *self == Self::Punct(c)
    }

    fn is_word(&self) -> bool {
        matches!(self, Self::Ident(_) | Self::Num(_))
    }
}

/// Split C++ source into tokens, dropping comments and preprocessor lines.
fn tokenize(src: &str) -> Vec<Tok> {
    tokenize_impl(src, false)
}

/// Like [`tokenize`], but keeps preprocessor directives (`#define` X-macro tables): a directive
/// is `Punct('#')`, its tokens (line continuations joined) and a closing `Punct('\n')`.
fn tokenize_keep_directives(src: &str) -> Vec<Tok> {
    tokenize_impl(src, true)
}

fn tokenize_impl(src: &str, keep_directives: bool) -> Vec<Tok> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let mut line_start = true;
    let mut in_directive = false;
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            if in_directive {
                out.push(Tok::Punct('\n'));
                in_directive = false;
            }
            line_start = true;
            i += 1;
            continue;
        }
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if keep_directives && c == b'\\' {
            // Line continuation inside a kept directive.
            let mut j = i + 1;
            if j < b.len() && b[j] == b'\r' {
                j += 1;
            }
            if j < b.len() && b[j] == b'\n' {
                i = j + 1;
                continue;
            }
        }
        if c == b'#' && line_start && keep_directives {
            out.push(Tok::Punct('#'));
            in_directive = true;
            line_start = false;
            i += 1;
            continue;
        }
        if c == b'#' && line_start {
            while i < b.len() && b[i] != b'\n' {
                if b[i] == b'\\' {
                    // Line continuation (`\` + LF or CRLF) keeps the directive going.
                    let mut j = i + 1;
                    if j < b.len() && b[j] == b'\r' {
                        j += 1;
                    }
                    if j < b.len() && b[j] == b'\n' {
                        i = j + 1;
                        continue;
                    }
                }
                i += 1;
            }
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && b.get(i + 1) == Some(&b'*') {
            i += 2;
            while i < b.len() && !(b[i] == b'*' && b.get(i + 1) == Some(&b'/')) {
                i += 1;
            }
            i = (i + 2).min(b.len());
            continue;
        }
        line_start = false;
        if c == b'"' || c == b'\'' {
            let start = i + 1;
            i += 1;
            while i < b.len() && b[i] != c && b[i] != b'\n' {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            let end = i.min(b.len());
            let text = String::from_utf8_lossy(&b[start..end]).into_owned();
            i = (i + 1).min(b.len());
            out.push(if c == b'"' {
                Tok::Str(text)
            } else {
                Tok::Num(text)
            });
            continue;
        }
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Tok::Ident(src[start..i].to_string()));
            continue;
        }
        if c.is_ascii_digit() {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.' || b[i] == b'\'') {
                i += 1;
            }
            out.push(Tok::Num(src[start..i].to_string()));
            continue;
        }
        if c.is_ascii() {
            out.push(Tok::Punct(char::from(c)));
            i += 1;
        } else {
            // Skip a non-ASCII character as a whole.
            let len = src[i..].chars().next().map_or(1, char::len_utf8);
            out.push(Tok::Punct('?'));
            i += len;
        }
    }
    out
}

/// Type names collected at global scope.
#[derive(Debug, Default)]
struct TypeEnv {
    aliases: BTreeMap<String, Vec<Tok>>,
    structs: BTreeSet<String>,
    enums: BTreeSet<String>,
}

/// A function declaration before its types are classified.
#[derive(Debug)]
struct RawDecl {
    name: String,
    ret: Vec<Tok>,
    params: Vec<Vec<Tok>>,
    variadic: bool,
    header: usize,
}

/// Index of the `}` matching the `{` at `open` (or the last token when unbalanced).
fn matching_brace(toks: &[Tok], open: usize) -> usize {
    let mut depth = 0usize;
    for (j, tok) in toks.iter().enumerate().skip(open) {
        if tok.is_punct('{') {
            depth += 1;
        } else if tok.is_punct('}') {
            depth -= 1;
            if depth == 0 {
                return j;
            }
        }
    }
    toks.len().saturating_sub(1)
}

/// Index of the `)`/`]` matching the opener at `open` (or `toks.len()` when unbalanced).
fn matching_paren(toks: &[Tok], open: usize) -> usize {
    let mut depth = 0usize;
    for (j, tok) in toks.iter().enumerate().skip(open) {
        if tok.is_punct('(') || tok.is_punct('[') {
            depth += 1;
        } else if tok.is_punct(')') || tok.is_punct(']') {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return j;
            }
        }
    }
    toks.len()
}

/// Walk the statements of one scope.
fn scan_scope(
    toks: &[Tok],
    extern_c: bool,
    header: usize,
    env: &mut TypeEnv,
    out: &mut Vec<RawDecl>,
) {
    let mut i = 0;
    while i < toks.len() {
        let start = i;
        let mut depth = 0i32;
        let mut block = None;
        let mut j = i;
        while j < toks.len() {
            let tok = &toks[j];
            if tok.is_punct('(') || tok.is_punct('[') {
                depth += 1;
            } else if tok.is_punct(')') || tok.is_punct(']') {
                depth -= 1;
            } else if depth <= 0 && (tok.is_punct(';') || tok.is_punct('}')) {
                break;
            } else if depth <= 0 && tok.is_punct('{') {
                block = Some((j, matching_brace(toks, j)));
                break;
            }
            j += 1;
        }
        let head = &toks[start..j.min(toks.len())];
        match block {
            None => {
                scan_statement(head, extern_c, header, env, out);
                i = j + 1;
            }
            Some((open, close)) => {
                scan_block(head, &toks[open + 1..close], extern_c, header, env, out);
                i = close + 1;
                let declarator_follows = matches!(
                    head.first().and_then(Tok::ident),
                    Some("struct" | "class" | "union" | "enum" | "typedef")
                ) || head.iter().any(|t| t.is_punct('='));
                if declarator_follows {
                    // `struct X { ... } x;` / `T y = { ... };`: skip to the end of the statement.
                    while i < toks.len() && !toks[i].is_punct(';') {
                        i += 1;
                    }
                    i += 1;
                } else if toks.get(i).is_some_and(|t| t.is_punct(';')) {
                    i += 1;
                }
            }
        }
    }
}

/// Drop leading `[[...]]` attributes.
fn strip_attributes(mut toks: &[Tok]) -> &[Tok] {
    while toks.len() >= 2 && toks[0].is_punct('[') && toks[1].is_punct('[') {
        let close = matching_paren(toks, 0);
        toks = &toks[(close + 1).min(toks.len())..];
    }
    toks
}

fn is_extern_c(toks: &[Tok]) -> bool {
    toks.len() >= 2 && toks[0].is_ident("extern") && toks[1] == Tok::Str("C".to_string())
}

/// First identifier naming a struct/enum in a head such as `struct alignas(8) Name : Base`.
fn tag_name(toks: &[Tok]) -> Option<String> {
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            Tok::Ident(w)
                if matches!(
                    w.as_str(),
                    "struct" | "class" | "union" | "enum" | "final" | "typedef"
                ) =>
            {
                i += 1;
            }
            Tok::Ident(w) if matches!(w.as_str(), "alignas" | "__declspec" | "__attribute__") => {
                i = matching_paren(toks, i + 1) + 1;
            }
            Tok::Punct('[') => i = matching_paren(toks, i) + 1,
            Tok::Ident(w) => return Some(w.clone()),
            _ => return None,
        }
    }
    None
}

fn record_tag(head: &[Tok], env: &mut TypeEnv) {
    let Some(keyword) = head.iter().find_map(Tok::ident) else {
        return;
    };
    let keyword = if keyword == "typedef" {
        head.iter().filter_map(Tok::ident).nth(1).unwrap_or("")
    } else {
        keyword
    };
    if let Some(name) = tag_name(head) {
        match keyword {
            "struct" | "class" | "union" => {
                env.structs.insert(name);
            }
            "enum" => {
                env.enums.insert(name);
            }
            _ => {}
        }
    }
}

fn scan_block(
    head: &[Tok],
    body: &[Tok],
    extern_c: bool,
    header: usize,
    env: &mut TypeEnv,
    out: &mut Vec<RawDecl>,
) {
    let head = strip_attributes(head);
    if is_extern_c(head) && head.len() == 2 {
        scan_scope(body, true, header, env, out);
        return;
    }
    match head.first().and_then(Tok::ident) {
        Some("namespace" | "inline" | "template") | None => {}
        Some("struct" | "class" | "union" | "enum" | "typedef") => record_tag(head, env),
        Some(_) => {
            // Inline function definition: inside `extern "C"` it still declares a C function.
            if extern_c || is_extern_c(head) {
                scan_statement(head, extern_c, header, env, out);
            }
        }
    }
}

fn scan_statement(
    stmt: &[Tok],
    extern_c: bool,
    header: usize,
    env: &mut TypeEnv,
    out: &mut Vec<RawDecl>,
) {
    let stmt = strip_attributes(stmt);
    if is_extern_c(stmt) {
        scan_statement(&stmt[2..], true, header, env, out);
        return;
    }
    let Some(first) = stmt.first().and_then(Tok::ident) else {
        return;
    };
    match first {
        "template" | "static_assert" | "namespace" | "friend" => {}
        "using" => {
            if let (Some(Tok::Ident(name)), Some(eq)) = (stmt.get(1), stmt.get(2))
                && eq.is_punct('=')
                && name != "namespace"
            {
                env.aliases.insert(name.clone(), stmt[3..].to_vec());
            }
        }
        "typedef" => record_typedef(&stmt[1..], env),
        "struct" | "class" | "union" | "enum"
            if stmt.len() <= 5 && !stmt.iter().any(|t| t.is_punct('(')) =>
        {
            record_tag(stmt, env);
        }
        _ => {
            if extern_c && let Some(decl) = parse_function(stmt, header) {
                out.push(decl);
            }
        }
    }
}

fn record_typedef(rest: &[Tok], env: &mut TypeEnv) {
    // Function pointer: `typedef R (*Name)(params)`.
    if let Some(p) = rest
        .windows(3)
        .position(|w| w[0].is_punct('(') && w[1].is_punct('*') && matches!(w[2], Tok::Ident(_)))
        && let Some(Tok::Ident(name)) = rest.get(p + 2)
    {
        let mut target = rest.to_vec();
        target.remove(p + 2);
        env.aliases.insert(name.clone(), target);
        return;
    }
    if let Some((Tok::Ident(name), target)) = rest.split_last()
        && !target.is_empty()
    {
        env.aliases.insert(name.clone(), target.to_vec());
    }
}

/// Keywords that can never be a parameter or function name.
fn is_type_keyword(word: &str) -> bool {
    matches!(
        word,
        "void"
            | "bool"
            | "char"
            | "short"
            | "int"
            | "long"
            | "signed"
            | "unsigned"
            | "float"
            | "double"
            | "wchar_t"
            | "char8_t"
            | "char16_t"
            | "char32_t"
            | "const"
            | "volatile"
            | "struct"
            | "class"
            | "union"
            | "enum"
            | "typename"
            | "auto"
    )
}

/// Drop storage-class keywords and `__declspec(...)`/`__attribute__((...))` from a return type.
fn clean_return_type(toks: &[Tok]) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        match &toks[i] {
            Tok::Ident(w)
                if matches!(
                    w.as_str(),
                    "extern" | "static" | "inline" | "constexpr" | "__cdecl" | "__stdcall"
                ) =>
            {
                i += 1;
            }
            Tok::Ident(w) if matches!(w.as_str(), "__declspec" | "__attribute__") => {
                i = matching_paren(toks, i + 1) + 1;
            }
            Tok::Punct('[') if toks.get(i + 1).is_some_and(|t| t.is_punct('[')) => {
                i = matching_paren(toks, i) + 1;
            }
            tok => {
                out.push(tok.clone());
                i += 1;
            }
        }
    }
    out
}

fn parse_function(stmt: &[Tok], header: usize) -> Option<RawDecl> {
    let stmt = &clean_return_type(stmt);
    let open = (1..stmt.len()).find(|&p| stmt[p].is_punct('('))?;
    if stmt[..open].iter().any(|t| t.is_punct('=')) {
        return None;
    }
    let name = stmt[open - 1].ident()?;
    if is_type_keyword(name) || matches!(name, "__declspec" | "__attribute__" | "alignas") {
        return None;
    }
    let close = matching_paren(stmt, open);
    if close >= stmt.len() {
        return None;
    }
    // Only qualifiers such as `noexcept` may follow the parameter list.
    if stmt[close + 1..]
        .iter()
        .any(|t| !matches!(t, Tok::Ident(_)))
    {
        return None;
    }
    let ret = clean_return_type(&stmt[..open - 1]);
    if ret.is_empty() {
        return None;
    }

    let mut params = Vec::new();
    let mut variadic = false;
    let inner = &stmt[open + 1..close];
    let mut depth = 0i32;
    let mut current: Vec<Tok> = Vec::new();
    for tok in inner.iter().chain(std::iter::once(&Tok::Punct(','))) {
        if tok.is_punct('(') || tok.is_punct('[') || tok.is_punct('<') {
            depth += 1;
        } else if tok.is_punct(')') || tok.is_punct(']') || tok.is_punct('>') {
            depth -= 1;
        }
        if depth == 0 && tok.is_punct(',') {
            let param = std::mem::take(&mut current);
            if param.iter().filter(|t| t.is_punct('.')).count() >= 3 {
                variadic = true;
            } else if !param.is_empty() {
                params.push(strip_param_name(&param));
            }
        } else {
            current.push(tok.clone());
        }
    }
    if params.len() == 1 && params[0].len() == 1 && params[0][0].is_ident("void") {
        params.clear();
    }
    Some(RawDecl {
        name: name.to_string(),
        ret,
        params,
        variadic,
        header,
    })
}

/// Remove a default argument and the parameter name, leaving the type.
fn strip_param_name(param: &[Tok]) -> Vec<Tok> {
    let mut depth = 0i32;
    let mut end = param.len();
    for (i, tok) in param.iter().enumerate() {
        if tok.is_punct('(') || tok.is_punct('[') {
            depth += 1;
        } else if tok.is_punct(')') || tok.is_punct(']') {
            depth -= 1;
        } else if depth == 0 && tok.is_punct('=') {
            end = i;
            break;
        }
    }
    let ty = &param[..end];
    if ty.iter().any(|t| t.is_punct('(') || t.is_punct('[')) {
        return ty.to_vec();
    }
    if let [.., before, Tok::Ident(last)] = ty
        && !is_type_keyword(last)
        && !before.ident().is_some_and(is_type_keyword_qualifier)
        && (before.is_word()
            || before.is_punct('*')
            || before.is_punct('&')
            || before.is_punct('>'))
    {
        return ty[..ty.len() - 1].to_vec();
    }
    ty.to_vec()
}

/// `struct X` / `const X`: the identifier after these is the type, not a parameter name.
fn is_type_keyword_qualifier(word: &str) -> bool {
    matches!(
        word,
        "struct" | "class" | "union" | "enum" | "typename" | "const" | "volatile"
    )
}

/// Join type tokens back into C++ (`const char16_t*`, `unsigned int`, `std::size_t`).
fn spell(toks: &[Tok]) -> String {
    let mut out = String::new();
    let mut prev_word = false;
    for tok in toks {
        let word = tok.is_word();
        if word && prev_word {
            out.push(' ');
        }
        match tok {
            Tok::Ident(s) | Tok::Num(s) => out.push_str(s),
            Tok::Str(s) => {
                let _ = write!(out, "\"{s}\"");
            }
            Tok::Punct(c) => out.push(*c),
        }
        prev_word = word;
    }
    out
}

impl TypeEnv {
    fn ctype(&self, toks: &[Tok]) -> CType {
        CType {
            spelling: spell(toks),
            kind: self.classify(toks, 0),
        }
    }

    fn classify(&self, toks: &[Tok], depth: u8) -> TypeKind {
        if depth > MAX_ALIAS_DEPTH || toks.is_empty() {
            return TypeKind::Unknown;
        }
        if toks.iter().any(|t| t.is_punct('(')) {
            return if toks.iter().any(|t| t.is_punct('*')) {
                TypeKind::FnPtr
            } else {
                TypeKind::Unknown
            };
        }
        if toks.iter().any(|t| t.is_punct('[') || t.is_punct('&')) {
            return TypeKind::Unknown;
        }
        if toks.iter().any(|t| t.is_punct('*')) {
            return TypeKind::Ptr;
        }
        // Words that name the type: drop qualifiers and namespace paths (`std::`).
        let words: Vec<&str> = toks
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                let w = t.ident()?;
                let qualified = toks.get(i + 1).is_some_and(|n| n.is_punct(':'));
                let skip = qualified
                    || matches!(
                        w,
                        "const" | "volatile" | "struct" | "class" | "union" | "enum" | "typename"
                    );
                (!skip).then_some(w)
            })
            .collect();
        if toks
            .iter()
            .any(|t| !matches!(t, Tok::Ident(_)) && !t.is_punct(':'))
        {
            return TypeKind::Unknown;
        }
        match words.as_slice() {
            ["void"] => TypeKind::Void,
            ["float"] => TypeKind::F32,
            ["double"] => TypeKind::F64,
            [single] => {
                if is_fixed_int(single) {
                    TypeKind::Int
                } else if let Some(target) = self.aliases.get(*single) {
                    self.classify(target, depth + 1)
                } else if self.enums.contains(*single) {
                    TypeKind::Int
                } else if self.structs.contains(*single) {
                    TypeKind::Struct
                } else if is_builtin_int_word(single) {
                    TypeKind::Int
                } else {
                    TypeKind::Unknown
                }
            }
            many if !many.is_empty() && many.iter().all(|w| is_builtin_int_word(w)) => {
                TypeKind::Int
            }
            _ => TypeKind::Unknown,
        }
    }
}

fn is_builtin_int_word(word: &str) -> bool {
    matches!(
        word,
        "bool"
            | "char"
            | "short"
            | "int"
            | "long"
            | "signed"
            | "unsigned"
            | "wchar_t"
            | "char8_t"
            | "char16_t"
            | "char32_t"
    )
}

fn is_fixed_int(word: &str) -> bool {
    matches!(
        word,
        "int8_t"
            | "int16_t"
            | "int32_t"
            | "int64_t"
            | "uint8_t"
            | "uint16_t"
            | "uint32_t"
            | "uint64_t"
            | "size_t"
            | "ssize_t"
            | "ptrdiff_t"
            | "intptr_t"
            | "uintptr_t"
    )
}

// ---------------------------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------------------------

/// DLL identity: ASCII-lowercased with an optional `.dll` suffix removed.
fn dll_key(dll: &str) -> String {
    let lower = dll.to_ascii_lowercase();
    match lower.strip_suffix(".dll") {
        Some(stem) => stem.to_string(),
        None => lower,
    }
}

/// First (highest-priority) shim family prefix for an imported DLL, or `None` when the runtime
/// has no family for it ([`shim_families_for_dll`] lists all of them).
pub fn shim_prefix_for_dll(dll: &str) -> Option<&'static str> {
    shim_families_for_dll(dll).first().copied()
}

/// Matching key: ASCII-lowercased with underscores removed (`CreateFileW` and `create_file_w`
/// both give `createfilew`).
pub fn name_key(name: &str) -> String {
    name.chars()
        .filter(|&c| c != '_')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Why an import has no shim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unresolved {
    /// The DLL maps to no shim family.
    UnknownDll,
    /// Imported by ordinal: there is no name to bind.
    ByOrdinal,
    /// The family has no shim with a matching name.
    NoShim { prefix: &'static str },
    /// Several shims match the name.
    Ambiguous { candidates: Vec<String> },
    /// The matching shim's signature cannot be marshaled from guest state.
    Unmarshalable { shim: String, reason: String },
    /// The shim's audit (`PG6_SHIM_AUDIT`) says it must not be called from recompiled code.
    Denied { shim: String, reason: String },
    /// The shim belongs to an audited family and is not audited yet (`note` is the `pending`
    /// row's note, `None` when the shim has no audit row at all).
    PendingAudit { shim: String, note: Option<String> },
    /// No shim, with a recorded reason (`PG6_IMPORT_GAPS`); `code`/`detail` say why it failed.
    Gap {
        code: &'static str,
        detail: String,
        reason: String,
    },
}

impl Unresolved {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnknownDll => "unknown_dll",
            Self::ByOrdinal => "ordinal",
            Self::NoShim { .. } => "no_shim",
            Self::Ambiguous { .. } => "ambiguous",
            Self::Unmarshalable { .. } => "unmarshalable",
            Self::Denied { .. } => "denied",
            Self::PendingAudit { .. } => "pending_audit",
            Self::Gap { code, .. } => code,
        }
    }

    /// Human-readable explanation.
    pub fn describe(&self) -> String {
        match self {
            Self::UnknownDll => "no runtime shim family for this DLL".to_string(),
            Self::ByOrdinal => {
                "imported by ordinal; no runtime export table declares this ordinal".to_string()
            }
            Self::NoShim { prefix } => format!("no {prefix}* shim with a matching name"),
            Self::Ambiguous { candidates } => {
                format!("ambiguous shims: {}", candidates.join(", "))
            }
            Self::Unmarshalable { shim, reason } => format!("{shim}: {reason}"),
            Self::Denied { shim, reason } => format!("{shim}: denied by the shim audit: {reason}"),
            Self::PendingAudit { shim, note } => match note {
                Some(note) if !note.is_empty() => format!("{shim}: pending audit: {note}"),
                Some(_) => format!("{shim}: pending audit"),
                None => format!("{shim}: no PG6_SHIM_AUDIT row (unaudited shims never bind)"),
            },
            Self::Gap { detail, reason, .. } => format!("{detail}; recorded gap: {reason}"),
        }
    }

    /// The explicitly recorded reason this import stays unbound: an audit verdict (`deny` or a
    /// `pending` row), the shim's signature, or a `PG6_IMPORT_GAPS` row. `None` means nobody
    /// decided it (no shim and no gap row, an ambiguity, or a shim without an audit row).
    pub fn recorded(&self) -> Option<String> {
        match self {
            Self::Denied { reason, .. } => Some(format!("denied: {reason}")),
            Self::PendingAudit {
                note: Some(note), ..
            } => Some(if note.is_empty() {
                "pending audit".to_string()
            } else {
                format!("pending audit: {note}")
            }),
            Self::Unmarshalable { reason, .. } => Some(format!("unmarshalable: {reason}")),
            Self::Gap { reason, .. } => Some(format!("gap: {reason}")),
            _ => None,
        }
    }
}

/// Outcome of resolving one import.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    Bound { shim: String },
    Unresolved(Unresolved),
}

/// The export of another guest module an import is bound to by a program build
/// (`pipeline::program`): the IAT slot is pre-filled with `va` in the guest image, so the import
/// needs no thunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuestTarget {
    /// Build module name of the exporting guest module.
    pub module: String,
    /// Guest VA of the exported function or data.
    pub va: u64,
}

/// One import with its IAT slot and binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportBinding {
    /// Position in the plan; the thunk is `pg6_import_<index>`.
    pub index: usize,
    pub dll: String,
    pub symbol: ImportedSymbol,
    /// RVA of the 8-byte IAT slot, `None` when the descriptor has no IAT.
    pub slot_rva: Option<u32>,
    pub delay_load: bool,
    /// Host shim binding (meaningless while `guest` is set).
    pub resolution: Resolution,
    /// Set when a program build binds the import to another guest module; such an import gets
    /// no thunk and no slot table entry and counts as neither bound nor unresolved.
    pub guest: Option<GuestTarget>,
}

impl ImportBinding {
    /// `DLL!name` or `DLL!#ordinal`.
    pub fn label(&self) -> String {
        match &self.symbol {
            ImportedSymbol::ByName { name, .. } => format!("{}!{name}", self.dll),
            ImportedSymbol::ByOrdinal { ordinal } => format!("{}!#{ordinal}", self.dll),
        }
    }

    pub fn bound_shim(&self) -> Option<&str> {
        match &self.resolution {
            Resolution::Bound { shim } => Some(shim),
            Resolution::Unresolved(_) => None,
        }
    }

    #[cfg(test)]
    fn thunk_name(&self) -> String {
        Names::default().thunk(self.index)
    }

    /// Bound to a host shim (and not to a guest module).
    fn host_bound(&self) -> bool {
        self.guest.is_none() && self.bound_shim().is_some()
    }

    /// Gets a thunk and a slot table entry.
    fn tabled(&self) -> bool {
        self.slot_rva.is_some() && self.guest.is_none()
    }
}

/// Bindings for every import of an image: normal imports first, then delay-load imports, each
/// in table order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgePlan {
    pub imports: Vec<ImportBinding>,
}

impl BridgePlan {
    pub fn bound_count(&self) -> usize {
        self.imports.iter().filter(|b| b.host_bound()).count()
    }

    pub fn unresolved_count(&self) -> usize {
        self.imports.len() - self.bound_count() - self.guest_count()
    }

    /// Imports bound to another guest module ([`ImportBinding::guest`]).
    pub fn guest_count(&self) -> usize {
        self.imports.iter().filter(|b| b.guest.is_some()).count()
    }

    /// Binds the import whose IAT slot is at `slot_rva` to a guest module's export. Returns
    /// false when no import uses that slot.
    pub fn bind_guest(&mut self, slot_rva: u32, target: GuestTarget) -> bool {
        match self
            .imports
            .iter_mut()
            .find(|b| b.slot_rva == Some(slot_rva))
        {
            Some(import) => {
                import.guest = Some(target);
                true
            }
            None => false,
        }
    }

    /// Per-DLL `(dll, bound, unresolved)` over the imports not bound to a guest module, sorted.
    /// Spellings of one DLL (case, `.dll` suffix) are merged under the first spelling seen.
    pub fn coverage_by_dll(&self) -> Vec<(String, usize, usize)> {
        let mut by_dll: BTreeMap<String, (String, usize, usize)> = BTreeMap::new();
        for import in self.imports.iter().filter(|i| i.guest.is_none()) {
            let entry = by_dll
                .entry(dll_key(&import.dll))
                .or_insert_with(|| (import.dll.clone(), 0, 0));
            if import.bound_shim().is_some() {
                entry.1 += 1;
            } else {
                entry.2 += 1;
            }
        }
        by_dll.into_values().collect()
    }
}

/// Shim lookup: export tables, name keys and guest-state thunks.
struct ShimIndex<'a> {
    catalogue: &'a Catalogue,
    /// `(family prefix, name key)` -> shims, for [`NameRule::TablesThenKey`] families.
    by_key: BTreeMap<(&'static str, String), Vec<&'a ShimDecl>>,
    /// Guest-state thunks by name key (only `void (void*)` declarations).
    gthunks: BTreeMap<String, Vec<&'a ShimDecl>>,
    /// Export name -> table entries whose symbol is declared.
    by_name: BTreeMap<&'a str, Vec<&'a ExportEntry>>,
}

impl<'a> ShimIndex<'a> {
    fn new(catalogue: &'a Catalogue) -> Self {
        let mut by_key: BTreeMap<(&'static str, String), Vec<&'a ShimDecl>> = BTreeMap::new();
        for family in SHIM_FAMILIES
            .iter()
            .filter(|f| f.rule == NameRule::TablesThenKey)
        {
            let prefix = family.prefix;
            for shim in catalogue.family(prefix) {
                by_key
                    .entry((prefix, name_key(&shim.name[prefix.len()..])))
                    .or_default()
                    .push(shim);
            }
        }
        let mut gthunks: BTreeMap<String, Vec<&'a ShimDecl>> = BTreeMap::new();
        for shim in catalogue
            .family(PREFIX_GTHUNK)
            .filter(|s| is_guest_state_thunk(s))
        {
            gthunks
                .entry(name_key(&shim.name[PREFIX_GTHUNK.len()..]))
                .or_default()
                .push(shim);
        }
        let mut by_name: BTreeMap<&'a str, Vec<&'a ExportEntry>> = BTreeMap::new();
        for entry in &catalogue.exports {
            if catalogue.get(&entry.symbol).is_some() {
                by_name.entry(entry.name.as_str()).or_default().push(entry);
            }
        }
        Self {
            catalogue,
            by_key,
            gthunks,
            by_name,
        }
    }

    /// Distinct declared symbols the export tables register for `name` whose symbol satisfies
    /// `accept`: those registered under `dll` itself when there are any, else all of them.
    fn table_symbols(&self, dll: &str, name: &str, accept: impl Fn(&str) -> bool) -> Vec<&'a str> {
        let Some(entries) = self.by_name.get(name) else {
            return Vec::new();
        };
        let key = dll_key(dll);
        let matching: Vec<&ExportEntry> = entries
            .iter()
            .copied()
            .filter(|e| accept(&e.symbol))
            .collect();
        let same_dll: Vec<&ExportEntry> = matching
            .iter()
            .copied()
            .filter(|e| dll_key(&e.dll) == key)
            .collect();
        let chosen = if same_dll.is_empty() {
            matching
        } else {
            same_dll
        };
        let mut symbols: Vec<&'a str> = chosen.iter().map(|e| e.symbol.as_str()).collect();
        symbols.sort_unstable();
        symbols.dedup();
        symbols
    }

    fn resolve(&self, dll: &str, symbol: &ImportedSymbol) -> Resolution {
        match self.resolve_inner(dll, symbol) {
            Resolution::Unresolved(why) if why.recorded().is_none() => {
                match self.catalogue.gap_for(dll, symbol) {
                    Some(gap) => Resolution::Unresolved(Unresolved::Gap {
                        code: why.code(),
                        detail: why.describe(),
                        reason: gap.reason.clone(),
                    }),
                    None => Resolution::Unresolved(why),
                }
            }
            other => other,
        }
    }

    fn resolve_inner(&self, dll: &str, symbol: &ImportedSymbol) -> Resolution {
        let families = shim_families_for_dll(dll);
        let name: Option<&str> = match symbol {
            ImportedSymbol::ByName { name, .. } => Some(name),
            ImportedSymbol::ByOrdinal { ordinal } => {
                self.catalogue.ordinal_import_name(dll, *ordinal)
            }
        };
        // A table entry registering a guest-state thunk under exactly this DLL and name.
        if let Some(name) = name {
            let key = dll_key(dll);
            let mut exact: Vec<&str> = self
                .by_name
                .get(name)
                .into_iter()
                .flatten()
                .filter(|e| dll_key(&e.dll) == key)
                .filter(|e| {
                    self.catalogue
                        .get(&e.symbol)
                        .is_some_and(is_guest_state_thunk)
                })
                .map(|e| e.symbol.as_str())
                .collect();
            exact.sort_unstable();
            exact.dedup();
            if let [gthunk] = exact.as_slice() {
                return Resolution::Bound {
                    shim: (*gthunk).to_string(),
                };
            }
        }
        if families.is_empty() {
            return Resolution::Unresolved(Unresolved::UnknownDll);
        }
        let Some(name) = name else {
            return Resolution::Unresolved(Unresolved::ByOrdinal);
        };
        match self.gthunks.get(&name_key(name)).map(Vec::as_slice) {
            None | Some([]) => {}
            Some([gthunk]) => {
                return Resolution::Bound {
                    shim: gthunk.name.clone(),
                };
            }
            // Names that differ only in underscores (`exit` / `_exit`): the thunk whose name
            // after the prefix is exactly the import name.
            Some(many) => {
                return match many
                    .iter()
                    .find(|s| &s.name[PREFIX_GTHUNK.len()..] == name)
                {
                    Some(gthunk) => Resolution::Bound {
                        shim: gthunk.name.clone(),
                    },
                    None => Resolution::Unresolved(Unresolved::Ambiguous {
                        candidates: many.iter().map(|s| s.name.clone()).collect(),
                    }),
                };
            }
        }
        // The families in priority order: the first one with a bindable shim wins; when none
        // has one, the first family's reason is reported.
        let mut first_failure: Option<Unresolved> = None;
        for &prefix in families {
            let Some(fam) = tables::family(prefix) else {
                continue;
            };
            let mut candidates: Vec<&'a str> =
                self.table_symbols(dll, name, |s| s.starts_with(prefix));
            if candidates.is_empty() && fam.rule == NameRule::TablesThenKey {
                candidates = self
                    .by_key
                    .get(&(prefix, name_key(name)))
                    .map(|v| v.iter().map(|s| s.name.as_str()).collect())
                    .unwrap_or_default();
            }
            let outcome = match candidates.as_slice() {
                [] => continue,
                [shim] => self.check(fam, shim),
                many => Err(Unresolved::Ambiguous {
                    candidates: many.iter().map(|s| (*s).to_string()).collect(),
                }),
            };
            match outcome {
                Ok(shim) => return Resolution::Bound { shim },
                Err(why) => {
                    first_failure.get_or_insert(why);
                }
            }
        }
        Resolution::Unresolved(first_failure.unwrap_or(Unresolved::NoShim {
            prefix: families[0],
        }))
    }

    /// Whether `shim` of `family` may bind: marshalable and, for audited families, audited `ok`.
    fn check(&self, family: &ShimFamily, shim: &str) -> Result<String, Unresolved> {
        let Some(decl) = self.catalogue.get(shim) else {
            return Err(Unresolved::NoShim {
                prefix: family.prefix,
            });
        };
        if let Some(reason) = decl.unsupported_reason() {
            return Err(Unresolved::Unmarshalable {
                shim: shim.to_string(),
                reason,
            });
        }
        if family.audited {
            match self.catalogue.audit.get(shim) {
                Some(row) if row.verdict == Verdict::Ok => {}
                Some(row) if row.verdict == Verdict::Deny => {
                    return Err(Unresolved::Denied {
                        shim: shim.to_string(),
                        reason: row.note.clone(),
                    });
                }
                row => {
                    return Err(Unresolved::PendingAudit {
                        shim: shim.to_string(),
                        note: row.map(|r| r.note.clone()),
                    });
                }
            }
        }
        Ok(shim.to_string())
    }
}

/// `void pg6_gthunk_<name>(void* state)`: a complete import thunk written by the runtime.
fn is_guest_state_thunk(decl: &ShimDecl) -> bool {
    decl.name.starts_with(PREFIX_GTHUNK)
        && !decl.variadic
        && decl.ret.kind == TypeKind::Void
        && decl.params.len() == 1
        && decl.params[0].kind == TypeKind::Ptr
}

/// Resolve every import and delay-load import of `pe` against `catalogue`.
pub fn plan_bridge(pe: &Pe64, catalogue: &Catalogue) -> BridgePlan {
    let index = ShimIndex::new(catalogue);
    let normal = pe
        .imports
        .iter()
        .map(|m| (m.dll_name.as_str(), m.iat_rva, false, m.symbols.as_slice()));
    let delayed = pe
        .delay_imports
        .iter()
        .map(|m| (m.dll_name.as_str(), m.iat_rva, true, m.symbols.as_slice()));
    let mut imports = Vec::new();
    for (dll, iat_rva, delay_load, symbols) in normal.chain(delayed) {
        for (i, symbol) in symbols.iter().enumerate() {
            let slot_rva = (iat_rva != 0)
                .then(|| {
                    u32::try_from(i * 8)
                        .ok()
                        .and_then(|o| iat_rva.checked_add(o))
                })
                .flatten();
            imports.push(ImportBinding {
                index: imports.len(),
                dll: dll.to_string(),
                symbol: symbol.clone(),
                slot_rva,
                delay_load,
                resolution: index.resolve(dll, symbol),
                guest: None,
            });
        }
    }
    BridgePlan { imports }
}

/// Resolve an import list without a PE (e.g. [`parse_import_list`] of
/// `docs/imports_required.md`): every import gets no IAT slot, so the rendered bridge has no
/// thunks and the manifest lists each import with its binding.
pub fn plan_import_list(imports: &[(String, ImportedSymbol)], catalogue: &Catalogue) -> BridgePlan {
    let index = ShimIndex::new(catalogue);
    let imports = imports
        .iter()
        .enumerate()
        .map(|(i, (dll, symbol))| ImportBinding {
            index: i,
            dll: dll.clone(),
            symbol: symbol.clone(),
            slot_rva: None,
            delay_load: false,
            resolution: index.resolve(dll, symbol),
            guest: None,
        })
        .collect();
    BridgePlan { imports }
}

/// Resolves `(dll, symbol)` pairs against a catalogue the way [`plan_bridge`] does, e.g. for an
/// import whose guest export forwards to a runtime DLL.
pub struct ShimResolver<'a> {
    index: ShimIndex<'a>,
}

impl<'a> ShimResolver<'a> {
    pub fn new(catalogue: &'a Catalogue) -> Self {
        Self {
            index: ShimIndex::new(catalogue),
        }
    }

    pub fn resolve(&self, dll: &str, symbol: &ImportedSymbol) -> Resolution {
        self.index.resolve(dll, symbol)
    }
}

// ---------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------

/// The generated files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgeOutput {
    pub header: String,
    pub source: String,
    pub seam: String,
    pub manifest: String,
}

impl BridgeOutput {
    /// Write all files into `dir` (created if missing). Returns the written paths.
    pub fn write_to(&self, dir: &Path) -> io::Result<Vec<PathBuf>> {
        fs::create_dir_all(dir)?;
        let files = [
            (HEADER_FILE, &self.header),
            (SOURCE_FILE, &self.source),
            (SEAM_FILE, &self.seam),
            (MANIFEST_FILE, &self.manifest),
        ];
        let mut written = Vec::with_capacity(files.len());
        for (name, text) in files {
            let path = dir.join(name);
            fs::write(&path, text)?;
            written.push(path);
        }
        Ok(written)
    }
}

/// Render the thunk header/source, default seam and manifest for `plan`.
pub fn render_bridge(plan: &BridgePlan, catalogue: &Catalogue) -> BridgeOutput {
    render_bridge_with(plan, catalogue, Names::default())
}

/// [`render_bridge`] for one module of a multi-module program: every generated global name
/// carries the module name, so the bridges of several modules link into one program.
///
/// | Default name | Namespaced name |
/// | :--- | :--- |
/// | `pg6_import_<n>` (thunk) | `pg6_import_<ns>_<n>` |
/// | `pg6_import_thunks`, `pg6_import_thunk_count` | `pg6_import_thunks_<ns>`, `pg6_import_thunk_count_<ns>` |
/// | `pg6_find_import_thunk` | `pg6_find_import_thunk_<ns>` |
///
/// The header guards its shared type definitions (`PG6_IMPORT_THUNK_TYPES_DEFINED`) so several
/// modules' headers can be included together (the program descriptor does). Imports bound to a
/// guest module ([`ImportBinding::guest`]) get no thunk and no table entry, and the manifest
/// records them with `"status": "guest"`.
pub fn render_bridge_namespaced(
    plan: &BridgePlan,
    catalogue: &Catalogue,
    namespace: &str,
) -> BridgeOutput {
    render_bridge_with(
        plan,
        catalogue,
        Names {
            ns: Some(namespace),
        },
    )
}

fn render_bridge_with(plan: &BridgePlan, catalogue: &Catalogue, names: Names<'_>) -> BridgeOutput {
    BridgeOutput {
        header: render_header(names),
        source: render_source(plan, catalogue, names),
        seam: render_seam(),
        manifest: render_manifest(plan, catalogue, names),
    }
}

/// Generated global names, optionally suffixed with a module namespace.
#[derive(Debug, Clone, Copy, Default)]
struct Names<'a> {
    ns: Option<&'a str>,
}

impl Names<'_> {
    fn global(&self, base: &str) -> String {
        match self.ns {
            Some(ns) => format!("{base}_{ns}"),
            None => base.to_string(),
        }
    }

    fn thunk(&self, index: usize) -> String {
        match self.ns {
            Some(ns) => format!("pg6_import_{ns}_{index}"),
            None => format!("pg6_import_{index}"),
        }
    }
}

/// C++ string literal (printable ASCII kept, everything else as octal escapes).
fn cpp_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for b in s.bytes() {
        match b {
            b'"' | b'\\' => {
                out.push('\\');
                out.push(char::from(b));
            }
            0x20..=0x7e => out.push(char::from(b)),
            _ => {
                let _ = write!(out, "\\{b:03o}");
            }
        }
    }
    out.push('"');
    out
}

/// Text safe inside a `//` comment (no line breaks, no trailing line continuation).
fn comment_text(s: &str) -> String {
    s.chars()
        .map(|c| {
            if (c.is_ascii_graphic() && c != '\\') || c == ' ' {
                c
            } else {
                '?'
            }
        })
        .collect()
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

const HEADER_TYPES: &str = "\
using Pg6ImportThunkFn = void (*)(void* state);

inline constexpr uint32_t kPg6ImportBound = 1u;      // thunk calls a runtime shim
inline constexpr uint32_t kPg6ImportDelayLoad = 2u;  // slot belongs to a delay-load IAT
inline constexpr uint32_t kPg6ImportByOrdinal = 4u;  // `ordinal` is set, `symbol` is null

struct Pg6ImportThunk {
    uint32_t slot_rva;
    uint32_t flags;
    uint32_t ordinal;
    const char* dll;
    const char* symbol;
    const char* shim;  // null when unresolved
    Pg6ImportThunkFn thunk;
};
";

fn render_header(names: Names<'_>) -> String {
    let mut out = String::new();
    match names.ns {
        None => out.push_str("// Generated by playgta6-recomp stubs. Do not edit.\n"),
        Some(ns) => {
            let _ = writeln!(
                out,
                "// Generated by playgta6-recomp stubs for module {ns}. Do not edit."
            );
        }
    }
    out.push_str(
        "//
// Import thunks: one per IAT slot of the recompiled image, called with the guest CPU state
// after the guest pushed its return address (see pg6_guest_seam.h).
#pragma once

#include <cstddef>
#include <cstdint>

",
    );
    if names.ns.is_some() {
        out.push_str(
            "#ifndef PG6_IMPORT_THUNK_TYPES_DEFINED\n#define PG6_IMPORT_THUNK_TYPES_DEFINED\n",
        );
        out.push_str(HEADER_TYPES);
        out.push_str("#endif\n");
    } else {
        out.push_str(HEADER_TYPES);
    }
    let _ = write!(
        out,
        "
// All thunks with an IAT slot, sorted by slot_rva.
extern const Pg6ImportThunk* const {thunks};
extern const std::size_t {count};

// Thunk bound to the IAT slot at `slot_rva`, or null.
const Pg6ImportThunk* {find}(uint32_t slot_rva);
",
        thunks = names.global("pg6_import_thunks"),
        count = names.global("pg6_import_thunk_count"),
        find = names.global("pg6_find_import_thunk"),
    );
    out
}

fn render_seam_text(translated_window: bool) -> String {
    format!(
        "\
// Generated by playgta6-recomp stubs. Do not edit.
//
// Guest-state seam of the import thunks. Every access to guest registers, guest memory and the
// runtime goes through these functions. This default matches the lifter's guest state:
// rax rcx rdx rbx rsp rbp rsi rdi r8..r15 (8 bytes each) at offset {gpr}, xmm0..xmm15 at
// offset {xmm} ({stride} bytes each), and guest addresses translate like lifted code does:
// host = __playgta6_guest_base + guest VA (docs/wasm_abi.md \"Guest Address Space\"; 0, the
// identity, in the browser build).
// To use another layout, build the thunks with -DPG6_GUEST_SEAM_HEADER=\"<header>\" that
// provides the same names.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include \"abi/playgta6_abi.h\"
#include \"os/os_shim.h\"

static_assert(sizeof(void*) == 8, \"guest addresses are 64-bit\");

inline constexpr uint64_t kPg6SeamGprOffset = {gpr};
inline constexpr uint64_t kPg6SeamXmmOffset = {xmm};
inline constexpr uint64_t kPg6SeamXmmStride = {stride};
// Fifth argument relative to rsp at callee entry: return address + 32-byte shadow space.
inline constexpr uint64_t kPg6SeamStackArgOffset = {stack};

// General purpose register `reg` in x86 encoding order (0 = rax, 1 = rcx, 4 = rsp, 8 = r8).
inline uint64_t pg6_seam_get_gpr(void* state, unsigned reg) {{
    uint64_t value;
    std::memcpy(&value, static_cast<unsigned char*>(state) + kPg6SeamGprOffset + 8u * reg, 8);
    return value;
}}

inline void pg6_seam_set_gpr(void* state, unsigned reg, uint64_t value) {{
    std::memcpy(static_cast<unsigned char*>(state) + kPg6SeamGprOffset + 8u * reg, &value, 8);
}}

// Low 64 bits of xmm<n>.
inline uint64_t pg6_seam_get_xmm_lo(void* state, unsigned n) {{
    uint64_t bits;
    std::memcpy(&bits, static_cast<unsigned char*>(state) + kPg6SeamXmmOffset + kPg6SeamXmmStride * n, 8);
    return bits;
}}

inline void pg6_seam_set_xmm_lo(void* state, unsigned n, uint64_t bits) {{
    std::memcpy(static_cast<unsigned char*>(state) + kPg6SeamXmmOffset + kPg6SeamXmmStride * n, &bits, 8);
}}

{translation}
inline uint64_t pg6_seam_load_u64(uint64_t guest_va) {{
    uint64_t value;
    std::memcpy(&value, pg6_seam_host_ptr(guest_va), 8);
    return value;
}}

// Leave the thunk like `ret`: pop the return address the guest caller pushed.
inline void pg6_seam_return(void* state) {{
    pg6_seam_set_gpr(state, 4, pg6_seam_get_gpr(state, 4) + 8);
}}

inline void pg6_seam_log(const char* message) {{
    playgta6::os::log(message);
}}

// Called after an unresolved import was logged. Must not return.
[[noreturn]] inline void pg6_seam_trap_unresolved(void* /*state*/) {{
    std::abort();
}}
",
        gpr = SEAM_GPR_OFFSET,
        xmm = SEAM_XMM_OFFSET,
        stride = SEAM_XMM_STRIDE,
        stack = STACK_ARG_OFFSET,
        translation = if translated_window {
            SEAM_WINDOW_TRANSLATION
        } else {
            SEAM_FLAT_TRANSLATION
        },
    )
}

/// The seam's address translation for [`MemoryModel::Base`] (the default output).
const SEAM_FLAT_TRANSLATION: &str = "\
// Guest address -> host pointer and back (null stays null): the shared translation.
inline void* pg6_seam_host_ptr(uint64_t guest_va) {
    return guest_va == 0 ? nullptr
                         : reinterpret_cast<void*>(
                               static_cast<uintptr_t>(__playgta6_guest_base + guest_va));
}

inline uint64_t pg6_seam_guest_addr(const void* host) {
    return host == nullptr
               ? 0
               : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(host)) - __playgta6_guest_base;
}
";

/// The seam's address translation for [`MemoryModel::Windowed`]: the runtime's shared
/// translation (`runtime/os/guest_window.h`), which also covers the translated CPU window.
const SEAM_WINDOW_TRANSLATION: &str = "\
// Guest address -> host pointer and back (null stays null): the shared translation. The lifted
// code uses the translated CPU window (memory model \"windowed\"), so guest addresses in the
// window translate through its slot table (os/guest_window.h); an unmapped window address gives
// a pointer that faults when dereferenced, as the guest's own access would.
#include \"os/guest_window.h\"

inline void* pg6_seam_host_ptr(uint64_t guest_va) {
    return playgta6::os::guest_window::host_or_trap(guest_va);
}

inline uint64_t pg6_seam_guest_addr(const void* host) {
    return playgta6::os::guest_window::guest_address(host);
}
";

/// Default seam ([`render_seam_for`] of the default memory model).
fn render_seam() -> String {
    render_seam_for(&MemoryModel::default())
}

/// The guest-state seam for code lifted with `model`: [`MemoryModel::Windowed`] translates
/// guest pointers through the runtime's window-aware translation; every other model keeps the
/// default seam.
pub fn render_seam_for(model: &MemoryModel) -> String {
    render_seam_text(matches!(model, MemoryModel::Windowed(_)))
}

const SOURCE_HELPERS: &str = "\
namespace {

template <typename T>
T pg6_arg_int(uint64_t raw) {
    if constexpr (std::is_same_v<T, bool>) {
        return (raw & 0xFFu) != 0;
    } else if constexpr (std::is_enum_v<T>) {
        return static_cast<T>(static_cast<std::underlying_type_t<T>>(raw));
    } else {
        return static_cast<T>(raw);
    }
}

template <typename T>
T pg6_arg_ptr(uint64_t raw) {
    return static_cast<T>(pg6_seam_host_ptr(raw));
}

template <typename T>
uint64_t pg6_ret_int(T value) {
    if constexpr (std::is_same_v<T, bool>) {
        return value ? 1u : 0u;
    } else if constexpr (std::is_enum_v<T>) {
        return static_cast<uint64_t>(static_cast<std::underlying_type_t<T>>(value));
    } else {
        return static_cast<uint64_t>(value);
    }
}

inline float pg6_f32_from_bits(uint64_t raw) {
    const uint32_t low = static_cast<uint32_t>(raw);
    float value;
    std::memcpy(&value, &low, sizeof value);
    return value;
}

inline double pg6_f64_from_bits(uint64_t raw) {
    double value;
    std::memcpy(&value, &raw, sizeof value);
    return value;
}

inline uint64_t pg6_bits_from_f32(float value) {
    uint32_t low;
    std::memcpy(&low, &value, sizeof low);
    return low;
}

inline uint64_t pg6_bits_from_f64(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

[[noreturn]] inline void pg6_unresolved(void* state, std::atomic<bool>& logged, const char* message) {
    if (!logged.exchange(true)) {
        pg6_seam_log(message);
    }
    pg6_seam_trap_unresolved(state);
}

";

/// Guest-state expression for Win64 argument `i` of kind `ty`.
fn arg_expr(i: usize, ty: &CType) -> String {
    let raw = if i < ARG_GPRS.len() {
        match ty.kind {
            TypeKind::F32 | TypeKind::F64 => format!("pg6_seam_get_xmm_lo(state, {i})"),
            _ => format!("pg6_seam_get_gpr(state, {})", ARG_GPRS[i]),
        }
    } else {
        let offset = 8 * (i - ARG_GPRS.len());
        format!("pg6_seam_load_u64(rsp + kPg6SeamStackArgOffset + {offset})")
    };
    match ty.kind {
        TypeKind::F32 => format!("pg6_f32_from_bits({raw})"),
        TypeKind::F64 => format!("pg6_f64_from_bits({raw})"),
        TypeKind::Ptr => format!("pg6_arg_ptr<{}>({raw})", ty.spelling),
        _ => format!("pg6_arg_int<{}>({raw})", ty.spelling),
    }
}

fn render_bound_thunk(out: &mut String, thunk: &str, shim: &ShimDecl) {
    let _ = writeln!(out, "void {thunk}(void* state) {{");
    if is_guest_state_thunk(shim) {
        // Marshals its own arguments and pops the return address itself.
        let _ = writeln!(out, "    {}(state);", shim.name);
        let _ = writeln!(out, "}}");
        return;
    }
    if shim.params.len() > ARG_GPRS.len() {
        let _ = writeln!(
            out,
            "    const uint64_t rsp = pg6_seam_get_gpr(state, {REG_RSP});"
        );
    }
    let call = if shim.params.is_empty() {
        format!("{}()", shim.name)
    } else {
        let args: Vec<String> = shim
            .params
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("        {}", arg_expr(i, ty)))
            .collect();
        format!("{}(\n{})", shim.name, args.join(",\n"))
    };
    match shim.ret.kind {
        TypeKind::Void => {
            let _ = writeln!(out, "    {call};");
        }
        kind => {
            let _ = writeln!(out, "    const auto result = {call};");
            let store = match kind {
                TypeKind::F32 => {
                    "pg6_seam_set_xmm_lo(state, 0, pg6_bits_from_f32(result))".to_string()
                }
                TypeKind::F64 => {
                    "pg6_seam_set_xmm_lo(state, 0, pg6_bits_from_f64(result))".to_string()
                }
                TypeKind::Ptr => {
                    format!("pg6_seam_set_gpr(state, {REG_RAX}, pg6_seam_guest_addr(result))")
                }
                _ => format!("pg6_seam_set_gpr(state, {REG_RAX}, pg6_ret_int(result))"),
            };
            let _ = writeln!(out, "    {store};");
        }
    }
    let _ = writeln!(out, "    pg6_seam_return(state);");
    let _ = writeln!(out, "}}");
}

fn slot_text(slot_rva: Option<u32>) -> String {
    slot_rva.map_or_else(|| "none".to_string(), |s| format!("0x{s:x}"))
}

fn render_source(plan: &BridgePlan, catalogue: &Catalogue, names: Names<'_>) -> String {
    let mut out = String::new();
    match names.ns {
        None => out.push_str("// Generated by playgta6-recomp stubs. Do not edit.\n"),
        Some(ns) => {
            let _ = writeln!(
                out,
                "// Generated by playgta6-recomp stubs for module {ns}. Do not edit."
            );
        }
    }
    let _ = write!(
        out,
        "// {} imports: {} bound, {} unresolved",
        plan.imports.len(),
        plan.bound_count(),
        plan.unresolved_count()
    );
    match plan.guest_count() {
        0 => out.push_str(".\n"),
        guest => {
            let _ = writeln!(out, ", {guest} bound to guest modules.");
        }
    }
    let _ = writeln!(out, "#include \"{HEADER_FILE}\"");
    out.push('\n');
    for system in ["atomic", "cstddef", "cstdint", "cstring", "type_traits"] {
        let _ = writeln!(out, "#include <{system}>");
    }
    out.push('\n');
    out.push_str("#ifdef PG6_GUEST_SEAM_HEADER\n#include PG6_GUEST_SEAM_HEADER\n#else\n");
    let _ = writeln!(out, "#include \"{SEAM_FILE}\"\n#endif");

    let includes: BTreeSet<&str> = plan
        .imports
        .iter()
        .filter(|i| i.tabled())
        .filter_map(|i| catalogue.get(i.bound_shim()?))
        .map(|s| s.include.as_str())
        .collect();
    if !includes.is_empty() {
        out.push('\n');
        for include in includes {
            let _ = writeln!(out, "#include {}", cpp_str(include));
        }
    }
    out.push('\n');
    out.push_str(SOURCE_HELPERS);

    let tabled: Vec<&ImportBinding> = plan.imports.iter().filter(|i| i.tabled()).collect();
    for import in &tabled {
        let slot = slot_text(import.slot_rva);
        let thunk = names.thunk(import.index);
        match (
            &import.resolution,
            import.bound_shim().and_then(|s| catalogue.get(s)),
        ) {
            (Resolution::Bound { .. }, Some(shim)) => {
                let _ = writeln!(
                    out,
                    "// {} -> {} (slot {slot})",
                    comment_text(&import.label()),
                    shim.name
                );
                render_bound_thunk(&mut out, &thunk, shim);
            }
            (resolution, _) => {
                let reason = match resolution {
                    Resolution::Unresolved(why) => why.describe(),
                    Resolution::Bound { shim } => format!("{shim} missing from catalogue"),
                };
                let _ = writeln!(
                    out,
                    "// {} -> unresolved (slot {slot})",
                    comment_text(&import.label())
                );
                let message = format!(
                    "unresolved import {} (slot {slot}): {reason}",
                    import.label()
                );
                let _ = writeln!(out, "void {thunk}(void* state) {{");
                out.push_str("    static std::atomic<bool> logged{false};\n");
                let _ = writeln!(
                    out,
                    "    pg6_unresolved(state, logged, {});",
                    cpp_str(&message)
                );
                out.push_str("}\n");
            }
        }
        out.push('\n');
    }

    let mut sorted = tabled;
    sorted.sort_by_key(|i| (i.slot_rva, i.index));
    let thunks = names.global("pg6_import_thunks");
    let count = names.global("pg6_import_thunk_count");
    let find = names.global("pg6_find_import_thunk");
    if sorted.is_empty() {
        let _ = writeln!(
            out,
            "}}  // namespace\n\nconst Pg6ImportThunk* const {thunks} = nullptr;"
        );
    } else {
        out.push_str("const Pg6ImportThunk kThunks[] = {\n");
        for import in &sorted {
            let mut flags = Vec::new();
            if import.bound_shim().is_some() {
                flags.push("kPg6ImportBound");
            }
            if import.delay_load {
                flags.push("kPg6ImportDelayLoad");
            }
            let (ordinal, symbol) = match &import.symbol {
                ImportedSymbol::ByName { name, .. } => (0, cpp_str(name)),
                ImportedSymbol::ByOrdinal { ordinal } => {
                    flags.push("kPg6ImportByOrdinal");
                    (*ordinal, "nullptr".to_string())
                }
            };
            let flags = if flags.is_empty() {
                "0u".to_string()
            } else {
                flags.join(" | ")
            };
            let shim = import
                .bound_shim()
                .map_or_else(|| "nullptr".to_string(), cpp_str);
            let _ = writeln!(
                out,
                "    {{0x{:08x}u, {flags}, {ordinal}u, {}, {symbol}, {shim}, &{}}},",
                import.slot_rva.unwrap_or(0),
                cpp_str(&import.dll),
                names.thunk(import.index)
            );
        }
        let _ = writeln!(
            out,
            "}};\n\n}}  // namespace\n\nconst Pg6ImportThunk* const {thunks} = kThunks;"
        );
    }
    let _ = writeln!(out, "const std::size_t {count} = {};", sorted.len());
    let _ = write!(
        out,
        "
const Pg6ImportThunk* {find}(uint32_t slot_rva) {{
    std::size_t lo = 0;
    std::size_t hi = {count};
    while (lo < hi) {{
        const std::size_t mid = lo + (hi - lo) / 2;
        if ({thunks}[mid].slot_rva < slot_rva) {{
            lo = mid + 1;
        }} else {{
            hi = mid;
        }}
    }}
    if (lo < {count} && {thunks}[lo].slot_rva == slot_rva) {{
        return &{thunks}[lo];
    }}
    return nullptr;
}}
"
    );
    out
}

fn render_manifest(plan: &BridgePlan, catalogue: &Catalogue, names: Names<'_>) -> String {
    let mut out = String::new();
    out.push_str("{\n  \"format\": \"playgta6-import-bridge\",\n  \"version\": 1,\n");
    if let Some(ns) = names.ns {
        let _ = writeln!(out, "  \"namespace\": {},", json_str(ns));
    }
    let _ = writeln!(
        out,
        "  \"catalogue\": {{\"headers\": {}, \"shims\": {}}},",
        catalogue.header_count,
        catalogue.len()
    );
    let guest = if names.ns.is_some() {
        format!(", \"guest\": {}", plan.guest_count())
    } else {
        String::new()
    };
    let _ = writeln!(
        out,
        "  \"summary\": {{\"imports\": {}, \"bound\": {}, \"unresolved\": {}{guest}}},",
        plan.imports.len(),
        plan.bound_count(),
        plan.unresolved_count()
    );
    out.push_str("  \"dlls\": [");
    let dlls = plan.coverage_by_dll();
    for (i, (dll, bound, unresolved)) in dlls.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        let _ = write!(
            out,
            "{sep}\n    {{\"dll\": {}, \"bound\": {bound}, \"unresolved\": {unresolved}}}",
            json_str(dll)
        );
    }
    out.push_str(if dlls.is_empty() { "],\n" } else { "\n  ],\n" });
    out.push_str("  \"imports\": [");
    for (i, import) in plan.imports.iter().enumerate() {
        let sep = if i == 0 { "" } else { "," };
        let (name, ordinal) = match &import.symbol {
            ImportedSymbol::ByName { name, .. } => (json_str(name), "null".to_string()),
            ImportedSymbol::ByOrdinal { ordinal } => ("null".to_string(), ordinal.to_string()),
        };
        let slot = import
            .slot_rva
            .map_or_else(|| "null".to_string(), |s| format!("\"0x{s:x}\""));
        let thunk = if import.tabled() {
            json_str(&names.thunk(import.index))
        } else {
            "null".to_string()
        };
        let _ = write!(
            out,
            "{sep}\n    {{\"index\": {}, \"dll\": {}, \"name\": {name}, \"ordinal\": {ordinal}, \
             \"slot_rva\": {slot}, \"delay_load\": {}, \"thunk\": {thunk}, ",
            import.index,
            json_str(&import.dll),
            import.delay_load
        );
        if let Some(target) = &import.guest {
            let _ = write!(
                out,
                "\"status\": \"guest\", \"module\": {}, \"target\": \"0x{:x}\"}}",
                json_str(&target.module),
                target.va
            );
            continue;
        }
        match &import.resolution {
            Resolution::Bound { shim } => {
                let decl = catalogue.get(shim);
                let params: Vec<String> = decl
                    .map(|d| d.params.iter().map(|p| json_str(p.kind.as_str())).collect())
                    .unwrap_or_default();
                let ret = decl.map_or("unknown", |d| d.ret.kind.as_str());
                let header = decl.map_or_else(|| "null".to_string(), |d| json_str(&d.include));
                let _ = write!(
                    out,
                    "\"status\": \"bound\", \"shim\": {}, \"header\": {header}, \
                     \"params\": [{}], \"ret\": {}}}",
                    json_str(shim),
                    params.join(", "),
                    json_str(ret)
                );
            }
            Resolution::Unresolved(why) => {
                let recorded = why
                    .recorded()
                    .map_or_else(|| "null".to_string(), |r| json_str(&r));
                let _ = write!(
                    out,
                    "\"status\": \"unresolved\", \"reason\": {}, \"detail\": {}, \
                     \"recorded\": {recorded}}}",
                    json_str(why.code()),
                    json_str(&why.describe())
                );
            }
        }
    }
    out.push_str(if plan.imports.is_empty() {
        "]\n}\n"
    } else {
        "\n  ]\n}\n"
    });
    out
}

/// Plain-text coverage summary (DLL and symbol names only).
pub fn render_coverage(plan: &BridgePlan, catalogue: &Catalogue) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Import bridge: {} imports, {} bound, {} unresolved ({} shims from {} headers)",
        plan.imports.len(),
        plan.bound_count(),
        plan.unresolved_count(),
        catalogue.len(),
        catalogue.header_count
    );
    let dlls = plan.coverage_by_dll();
    let width = dlls
        .iter()
        .map(|(d, _, _)| d.len())
        .max()
        .unwrap_or(3)
        .max(3);
    let _ = writeln!(
        out,
        "  {:<width$}  {:>5}  {:>10}",
        "DLL", "bound", "unresolved"
    );
    for (dll, bound, unresolved) in &dlls {
        let _ = writeln!(out, "  {dll:<width$}  {bound:>5}  {unresolved:>10}");
    }
    let unresolved: Vec<&ImportBinding> = plan
        .imports
        .iter()
        .filter(|i| i.guest.is_none() && i.bound_shim().is_none())
        .collect();
    if !unresolved.is_empty() {
        out.push_str("Unresolved:\n");
        for import in unresolved {
            if let Resolution::Unresolved(why) = &import.resolution {
                let _ = writeln!(out, "  {}: {}", import.label(), why.describe());
            }
        }
    }
    for note in &catalogue.diagnostics {
        let _ = writeln!(out, "Note: {note}");
    }
    out
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::process::Command;

    use super::*;
    use crate::pe::ImportModule;
    use crate::pe::test_util::build_synthetic_pe64;

    pub(in crate::stubs) const MINI_INCLUDE: &str = "mini/pg6_test_shims.h";

    /// Miniature shim surface: every type and declaration form the scanner must handle, plus
    /// decoys it must ignore. It is also compiled by the tool-gated tests.
    pub(in crate::stubs) const MINI_HEADER: &str = r#"// Miniature runtime shim header for generator tests.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#define PG6_MACRO_DECOY(x) \
    extern "C" void pg6_win32_macro_decoy(void);

struct Pg6Opaque;
struct Pg6Pair {
    uint32_t a;
    uint32_t b;
};
using Pg6Hr = int32_t;
typedef uint64_t Pg6Handle;
using Pg6OpaqueHandle = Pg6Opaque*;
using Pg6Cb = void (*)(void* context, bool canceled);
typedef void (*Pg6OldCb)(int);
enum class Pg6Mode : uint32_t { A = 0, B = 1 };

namespace pg6_decoy {
using Pg6Hr = double;
extern "C" void pg6_win32_namespace_decoy(void);
inline int helper() { return 1; }
}  // namespace pg6_decoy

// extern "C" void pg6_win32_comment_decoy(void);
/* extern "C" void pg6_win32_block_comment_decoy(void); */

extern "C" {

uint64_t pg6_win32_virtual_alloc(uint64_t address,
                                 uint64_t size,
                                 uint32_t allocation_type,
                                 uint32_t protect);
uint32_t pg6_win32_get_tick_count(void);
void pg6_win32_sleep(uint32_t milliseconds);
double pg6_win32_mix_args(int32_t a, double b, const char* c, float d, uint64_t e, float f,
                          int8_t g, Pg6OpaqueHandle h);
float pg6_win32_scale(float x, bool flag);
Pg6Opaque* pg6_win32_echo_ptr(Pg6Opaque* p);
Pg6Handle pg6_win32_open_thing(const char16_t* name, unsigned int flags, std::size_t len);
int pg6_win32_printf_like(const char* format, ...);
Pg6Pair pg6_win32_get_pair(void);
void pg6_win32_take_pair(Pg6Pair pair);
void pg6_win32_with_callback(Pg6OldCb callback);
void pg6_win32_mystery(std::string_view text);
int32_t pg6_win32_dup_a(void);
int32_t pg6_win32_dupa(void);

Pg6Hr pg6_gdk_xtask_queue_create(Pg6Mode work_mode, Pg6Mode completion_mode, Pg6OpaqueHandle* queue);
Pg6Hr pg6_gdk_xtask_queue_submit_callback(Pg6OpaqueHandle queue, uint32_t port, void* context,
                                          Pg6Cb callback);

Pg6Hr pg6_d3d12_d3d12_create_device(void* adapter, uint32_t feature_level, const void* riid,
                                    void** device);

}  // extern "C"

extern "C" uint32_t pg6_gdk_single_decl(uint8_t x = 0) noexcept;
"#;

    pub(in crate::stubs) fn mini_headers() -> Vec<ShimHeader> {
        vec![ShimHeader {
            include: MINI_INCLUDE.to_string(),
            text: MINI_HEADER.to_string(),
        }]
    }

    pub(in crate::stubs) fn mini_catalogue() -> Catalogue {
        Catalogue::scan(&mini_headers())
    }

    /// The shared synthetic PE with its delay-load IAT field filled in (the fixture leaves it
    /// 0): the delay descriptor sits at `.rdata + 0x180` (file offset 0x580), the IAT address
    /// at +12, and the table at the free `.rdata + 0x240` (RVA 0x2240).
    fn synthetic_pe(delay_use_va: bool) -> Pe64 {
        let mut bytes = build_synthetic_pe64(delay_use_va);
        let addr: u32 = if delay_use_va {
            0x0040_0000 + 0x2240
        } else {
            0x2240
        };
        bytes[0x58C..0x590].copy_from_slice(&addr.to_le_bytes());
        Pe64::parse(&bytes).expect("synthetic PE parses")
    }

    fn named(names: &[&str]) -> Vec<ImportedSymbol> {
        names
            .iter()
            .map(|n| ImportedSymbol::ByName {
                hint: 0,
                name: (*n).to_string(),
            })
            .collect()
    }

    const RICH_API_SET: [&str; 13] = [
        "GetTickCount",
        "Sleep",
        "MixArgs",
        "Scale",
        "EchoPtr",
        "OpenThing",
        "PrintfLike",
        "GetPair",
        "TakePair",
        "WithCallback",
        "Mystery",
        "DupA",
        "NotThere",
    ];

    /// Synthetic PE plus extra modules covering every resolution path.
    fn rich_pe() -> Pe64 {
        let mut pe = synthetic_pe(false);
        pe.imports.push(ImportModule {
            dll_name: "api-ms-win-core-test-l1-1-0.dll".to_string(),
            iat_rva: 0x3000,
            symbols: named(&RICH_API_SET),
        });
        pe.imports.push(ImportModule {
            dll_name: "XGameRuntime.DLL".to_string(),
            iat_rva: 0x3100,
            symbols: named(&["XTaskQueueSubmitCallback", "SingleDecl"]),
        });
        pe.imports.push(ImportModule {
            dll_name: "unknown.dll".to_string(),
            iat_rva: 0x3200,
            symbols: named(&["Foo"]),
        });
        pe.imports.push(ImportModule {
            dll_name: "KERNEL32".to_string(),
            iat_rva: 0,
            symbols: named(&["VirtualAlloc"]),
        });
        pe
    }

    fn find<'a>(plan: &'a BridgePlan, label: &str) -> &'a ImportBinding {
        plan.imports
            .iter()
            .find(|i| i.label() == label)
            .unwrap_or_else(|| panic!("import {label} missing from plan"))
    }

    fn kinds(shim: &ShimDecl) -> Vec<TypeKind> {
        shim.params.iter().map(|p| p.kind).collect()
    }

    pub(in crate::stubs) fn temp_dir(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!(
            "playgta6_stubs_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn scanner_reads_extern_c_declarations_and_kinds() {
        use TypeKind::*;
        let catalogue = mini_catalogue();
        let names: Vec<&str> = catalogue.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(catalogue.len(), 18, "{names:?}");
        for decoy in [
            "macro_decoy",
            "namespace_decoy",
            "comment_decoy",
            "block_comment_decoy",
        ] {
            assert!(
                catalogue.get(&format!("pg6_win32_{decoy}")).is_none(),
                "{decoy}"
            );
        }
        assert!(
            catalogue.diagnostics.is_empty(),
            "{:?}",
            catalogue.diagnostics
        );

        let alloc = catalogue.get("pg6_win32_virtual_alloc").unwrap();
        assert_eq!(alloc.include, MINI_INCLUDE);
        assert_eq!(kinds(alloc), [Int, Int, Int, Int]);
        assert_eq!(
            alloc.params[2].spelling, "uint32_t",
            "parameter name stripped"
        );

        let mix = catalogue.get("pg6_win32_mix_args").unwrap();
        assert_eq!(kinds(mix), [Int, F64, Ptr, F32, Int, F32, Int, Ptr]);
        assert_eq!(mix.ret.kind, F64);
        assert_eq!(mix.params[2].spelling, "const char*");
        assert_eq!(mix.params[7].spelling, "Pg6OpaqueHandle");

        let open = catalogue.get("pg6_win32_open_thing").unwrap();
        let spellings: Vec<&str> = open.params.iter().map(|p| p.spelling.as_str()).collect();
        assert_eq!(
            spellings,
            ["const char16_t*", "unsigned int", "std::size_t"]
        );
        assert_eq!(kinds(open), [Ptr, Int, Int]);
        assert_eq!(open.ret.kind, Int, "typedef alias resolved");

        let tick = catalogue.get("pg6_win32_get_tick_count").unwrap();
        assert!(tick.params.is_empty(), "(void) means no parameters");
        assert_eq!(catalogue.get("pg6_win32_sleep").unwrap().ret.kind, Void);
        assert_eq!(catalogue.get("pg6_win32_echo_ptr").unwrap().ret.kind, Ptr);

        let create = catalogue.get("pg6_gdk_xtask_queue_create").unwrap();
        assert_eq!(kinds(create), [Int, Int, Ptr], "enum class is an integer");
        assert_eq!(
            create.ret.kind, Int,
            "global alias wins over the namespaced decoy"
        );

        let single = catalogue.get("pg6_gdk_single_decl").unwrap();
        assert_eq!(kinds(single), [Int]);
        assert_eq!(
            single.params[0].spelling, "uint8_t",
            "default argument stripped"
        );

        let submit = catalogue
            .get("pg6_gdk_xtask_queue_submit_callback")
            .unwrap();
        assert_eq!(kinds(submit), [Ptr, Int, Ptr, FnPtr]);
        assert!(
            submit
                .unsupported_reason()
                .unwrap()
                .contains("parameter 3 `Pg6Cb`")
        );
        let callback = catalogue.get("pg6_win32_with_callback").unwrap();
        assert_eq!(kinds(callback), [FnPtr], "typedef function pointer");

        let printf = catalogue.get("pg6_win32_printf_like").unwrap();
        assert!(printf.variadic);
        assert_eq!(kinds(printf), [Ptr]);
        assert_eq!(
            printf.unsupported_reason().as_deref(),
            Some("variadic parameter list")
        );
        let pair = catalogue.get("pg6_win32_get_pair").unwrap();
        assert_eq!(pair.ret.kind, Struct);
        assert_eq!(
            pair.unsupported_reason().as_deref(),
            Some("returns `Pg6Pair` by value")
        );
        assert_eq!(
            kinds(catalogue.get("pg6_win32_take_pair").unwrap()),
            [Struct]
        );
        assert_eq!(
            kinds(catalogue.get("pg6_win32_mystery").unwrap()),
            [Unknown]
        );
        assert_eq!(alloc.unsupported_reason(), None);
        assert_eq!(mix.unsupported_reason(), None);
    }

    #[test]
    fn scanner_reports_conflicting_redeclarations() {
        let headers = [
            ShimHeader {
                include: "a.h".to_string(),
                text: "extern \"C\" int pg6_win32_f(int x);\n".to_string(),
            },
            ShimHeader {
                include: "b.h".to_string(),
                text: "extern \"C\" { int pg6_win32_f(int); long pg6_win32_g(void); }\n\
                       extern \"C\" double pg6_win32_f(double y);\n"
                    .to_string(),
            },
        ];
        let catalogue = Catalogue::scan(&headers);
        assert_eq!(catalogue.len(), 2);
        assert_eq!(catalogue.header_count, 2);
        assert_eq!(catalogue.get("pg6_win32_f").unwrap().include, "a.h");
        assert_eq!(
            catalogue.diagnostics.len(),
            1,
            "{:?}",
            catalogue.diagnostics
        );
        assert!(catalogue.diagnostics[0].contains("pg6_win32_f"));
    }

    #[test]
    fn dll_families_and_name_keys() {
        for dll in [
            "KERNEL32.dll",
            "kernel32.DLL",
            "kernel32",
            "KernelBase.dll",
            "api-ms-win-core-synch-l1-2-0.dll",
            "API-MS-WIN-CORE-FILE-L1-1-0.DLL",
            "ext-ms-win-ntuser-window-l1-1-0.dll",
            "WINHTTP.dll",
            "ADVAPI32.dll",
            "bcrypt.dll",
            "USER32.dll",
            "combase.dll",
            "ole32.dll",
        ] {
            assert_eq!(shim_prefix_for_dll(dll), Some("pg6_win32_"), "{dll}");
        }
        assert_eq!(
            shim_prefix_for_dll("WS2_32.dll"),
            Some("pg6_ws2_"),
            "the Winsock ordinal family first, pg6_win32_ second"
        );
        assert_eq!(shim_prefix_for_dll("XGameRuntime.dll"), Some("pg6_gdk_"));
        assert_eq!(
            shim_prefix_for_dll("Microsoft.Xbox.Services.141.GDK.C.Thunks.dll"),
            Some("pg6_gdk_")
        );
        assert_eq!(shim_prefix_for_dll("d3d12_xs.dll"), Some("pg6_d3d12_"));
        assert_eq!(shim_prefix_for_dll("D3D12_X.DLL"), Some("pg6_d3d12_"));
        assert_eq!(shim_prefix_for_dll("gdi32.dll"), None);
        assert_eq!(shim_prefix_for_dll("kernel32.dll.bak"), None);

        assert_eq!(name_key("CreateFileW"), name_key("create_file_w"));
        assert_eq!(name_key("GetTickCount64"), name_key("get_tick_count64"));
        assert_eq!(
            name_key("InitializeSRWLock"),
            name_key("initialize_srw_lock")
        );
        assert_eq!(name_key("XTaskQueueCreate"), name_key("xtask_queue_create"));
        assert_ne!(name_key("CreateEventA"), name_key("create_event_w"));
    }

    #[test]
    fn plan_binds_synthetic_imports_and_delay_imports() {
        let catalogue = mini_catalogue();
        for delay_use_va in [false, true] {
            let plan = plan_bridge(&synthetic_pe(delay_use_va), &catalogue);
            let summary: Vec<(String, Option<u32>, bool, Option<&str>)> = plan
                .imports
                .iter()
                .map(|i| (i.label(), i.slot_rva, i.delay_load, i.bound_shim()))
                .collect();
            assert_eq!(
                summary,
                [
                    (
                        "KERNEL32.dll!VirtualAlloc".to_string(),
                        Some(0x20C0),
                        false,
                        Some("pg6_win32_virtual_alloc")
                    ),
                    ("KERNEL32.dll!#42".to_string(), Some(0x20C8), false, None),
                    (
                        "xgameruntime.dll!XTaskQueueCreate".to_string(),
                        Some(0x20E0),
                        false,
                        Some("pg6_gdk_xtask_queue_create")
                    ),
                    (
                        "d3d12_xs.dll!D3D12CreateDevice".to_string(),
                        Some(0x2240),
                        true,
                        Some("pg6_d3d12_d3d12_create_device")
                    ),
                    ("d3d12_xs.dll!#99".to_string(), Some(0x2248), true, None),
                ],
                "delay_use_va = {delay_use_va}"
            );
            assert_eq!(
                find(&plan, "KERNEL32.dll!#42").resolution,
                Resolution::Unresolved(Unresolved::ByOrdinal)
            );
            assert_eq!(plan.bound_count(), 3);
            assert_eq!(plan.unresolved_count(), 2);
            let indices: Vec<usize> = plan.imports.iter().map(|i| i.index).collect();
            assert_eq!(indices, [0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn plan_reports_every_unresolved_reason() {
        let plan = plan_bridge(&rich_pe(), &mini_catalogue());
        let api = "api-ms-win-core-test-l1-1-0.dll";
        for (name, shim) in [
            ("GetTickCount", "pg6_win32_get_tick_count"),
            ("Sleep", "pg6_win32_sleep"),
            ("MixArgs", "pg6_win32_mix_args"),
            ("Scale", "pg6_win32_scale"),
            ("EchoPtr", "pg6_win32_echo_ptr"),
            ("OpenThing", "pg6_win32_open_thing"),
        ] {
            assert_eq!(
                find(&plan, &format!("{api}!{name}")).bound_shim(),
                Some(shim)
            );
        }
        let unresolved = |label: &str| match &find(&plan, label).resolution {
            Resolution::Unresolved(why) => why.clone(),
            Resolution::Bound { shim } => panic!("{label} unexpectedly bound to {shim}"),
        };
        for name in [
            "PrintfLike",
            "GetPair",
            "TakePair",
            "WithCallback",
            "Mystery",
        ] {
            assert_eq!(
                unresolved(&format!("{api}!{name}")).code(),
                "unmarshalable",
                "{name}"
            );
        }
        assert_eq!(
            unresolved(&format!("{api}!DupA")),
            Unresolved::Ambiguous {
                candidates: vec!["pg6_win32_dup_a".to_string(), "pg6_win32_dupa".to_string()]
            }
        );
        assert_eq!(
            unresolved(&format!("{api}!NotThere")),
            Unresolved::NoShim {
                prefix: "pg6_win32_"
            }
        );
        assert_eq!(
            unresolved("XGameRuntime.DLL!XTaskQueueSubmitCallback"),
            Unresolved::Unmarshalable {
                shim: "pg6_gdk_xtask_queue_submit_callback".to_string(),
                reason: "parameter 3 `Pg6Cb` is a host function pointer (guest callbacks need \
                         a trampoline)"
                    .to_string()
            }
        );
        assert_eq!(
            find(&plan, "XGameRuntime.DLL!SingleDecl").bound_shim(),
            Some("pg6_gdk_single_decl"),
            "DLL names are case-insensitive"
        );
        assert_eq!(unresolved("unknown.dll!Foo"), Unresolved::UnknownDll);
        let slotless = find(&plan, "KERNEL32!VirtualAlloc");
        assert_eq!(slotless.slot_rva, None);
        assert_eq!(slotless.bound_shim(), Some("pg6_win32_virtual_alloc"));
        assert_eq!(
            find(&plan, &format!("{api}!NotThere")).slot_rva,
            Some(0x3060)
        );

        assert_eq!(plan.imports.len(), 5 + 13 + 2 + 1 + 1);
        assert_eq!(plan.bound_count(), 3 + 6 + 1 + 1);
        let coverage = plan.coverage_by_dll();
        assert_eq!(
            coverage
                .iter()
                .find(|(d, _, _)| d == "KERNEL32.dll")
                .map(|(_, b, u)| (*b, *u)),
            Some((2, 1)),
            "KERNEL32.dll and KERNEL32 merge: {coverage:?}"
        );
        assert_eq!(coverage.len(), 5, "{coverage:?}");
    }

    #[test]
    fn guest_state_thunks_bind_by_export_table_and_ordinal() {
        let mut headers = mini_headers();
        headers.push(ShimHeader {
            include: "os/xbox_scarlett_guest.h".to_string(),
            text: "extern \"C\" {
                   void pg6_gthunk_dstorage_get_factory(void* state);
                   void pg6_gthunk_game_input_create(void* state);
                   void pg6_gthunk_only_in_table(void* state);
                   }
"
                .to_string(),
        });
        // The guest-callable registration table (xbox_scarlett_guest.cpp shape).
        headers.push(ShimHeader {
            include: "os/xbox_scarlett_guest.cpp".to_string(),
            text: "static const GuestExport kExports[] = {
                   {\"XCRDAPI.dll\", \"DStorageGetFactory\", 0, &pg6_gthunk_dstorage_get_factory},
                   {\"GameInput.dll\", \"GameInputCreate\", 1, &pg6_gthunk_game_input_create},
                   {\"Console.dll\", \"OnlyInTable\", 0, &pg6_gthunk_only_in_table},
                   };
                   extern \"C\" void pg6_gthunk_not_a_decl_in_a_source(void* state) {}
"
                .to_string(),
        });
        let catalogue = Catalogue::scan(&headers);
        assert!(
            catalogue.get("pg6_gthunk_not_a_decl_in_a_source").is_none(),
            "sources are scanned for tables only"
        );
        assert_eq!(
            catalogue.ordinal_import_name("GAMEINPUT", 1),
            Some("GameInputCreate")
        );
        assert_eq!(catalogue.ordinal_import_name("GameInput.dll", 2), None);
        assert_eq!(catalogue.ordinal_import_name("KERNEL32.dll", 1), None);
        let mut pe = synthetic_pe(false);
        pe.imports.push(ImportModule {
            dll_name: "GameInput.dll".to_string(),
            iat_rva: 0x3200,
            symbols: vec![
                ImportedSymbol::ByOrdinal { ordinal: 1 },
                ImportedSymbol::ByOrdinal { ordinal: 2 },
            ],
        });
        pe.imports.push(ImportModule {
            dll_name: "XCRDAPI.dll".to_string(),
            iat_rva: 0x3000,
            symbols: named(&["DStorageGetFactory", "DStorageSomethingElse"]),
        });
        pe.imports.push(ImportModule {
            dll_name: "xmem.dll".to_string(),
            iat_rva: 0x3100,
            symbols: named(&["DStorageGetFactory"]),
        });
        pe.imports.push(ImportModule {
            dll_name: "Console.dll".to_string(),
            iat_rva: 0x3300,
            symbols: named(&["OnlyInTable", "DStorageGetFactory"]),
        });
        let plan = plan_bridge(&pe, &catalogue);
        assert_eq!(
            find(&plan, "XCRDAPI.dll!DStorageGetFactory").bound_shim(),
            Some("pg6_gthunk_dstorage_get_factory")
        );
        assert_eq!(
            find(&plan, "XCRDAPI.dll!DStorageSomethingElse").resolution,
            Resolution::Unresolved(Unresolved::NoShim {
                prefix: "pg6_xbox_"
            })
        );
        assert_eq!(
            find(&plan, "xmem.dll!DStorageGetFactory").bound_shim(),
            Some("pg6_gthunk_dstorage_get_factory"),
            "a DLL with a shim family binds guest-state thunks by name key"
        );
        assert_eq!(
            find(&plan, "Console.dll!OnlyInTable").bound_shim(),
            Some("pg6_gthunk_only_in_table"),
            "a table entry binds its guest-state thunk under exactly its DLL, family or not"
        );
        assert_eq!(
            find(&plan, "Console.dll!DStorageGetFactory").resolution,
            Resolution::Unresolved(Unresolved::UnknownDll),
            "no name-key thunks for a DLL without a family"
        );
        assert_eq!(
            find(&plan, "GameInput.dll!#1").bound_shim(),
            Some("pg6_gthunk_game_input_create"),
            "GameInput ordinal 1 is GameInputCreate (from the table)"
        );
        assert_eq!(
            find(&plan, "GameInput.dll!#2").resolution,
            Resolution::Unresolved(Unresolved::ByOrdinal)
        );
    }

    #[test]
    fn guest_state_thunks_take_precedence_over_shims() {
        let mut headers = mini_headers();
        headers.push(ShimHeader {
            include: "guest/cb.h".to_string(),
            text: "extern \"C\" {\n\
                   void pg6_gthunk_xtask_queue_submit_callback(void* state);\n\
                   void pg6_gthunk_with_callback(void* state);\n\
                   int pg6_gthunk_sleep(void* state);\n\
                   void pg6_gthunk_not_imported(void* state);\n\
                   }\n"
                .to_string(),
        });
        let catalogue = Catalogue::scan(&headers);
        let plan = plan_bridge(&rich_pe(), &catalogue);
        let api = "api-ms-win-core-test-l1-1-0.dll";
        assert_eq!(
            find(&plan, "XGameRuntime.DLL!XTaskQueueSubmitCallback").bound_shim(),
            Some("pg6_gthunk_xtask_queue_submit_callback"),
            "function-pointer shim replaced by the guest-state thunk"
        );
        assert_eq!(
            find(&plan, &format!("{api}!WithCallback")).bound_shim(),
            Some("pg6_gthunk_with_callback")
        );
        assert_eq!(
            find(&plan, &format!("{api}!Sleep")).bound_shim(),
            Some("pg6_win32_sleep"),
            "only `void (void*)` declarations are guest-state thunks"
        );
        assert_eq!(plan.bound_count(), 3 + 6 + 1 + 1 + 2);

        let source = render_bridge(&plan, &catalogue).source;
        let name = find(&plan, "XGameRuntime.DLL!XTaskQueueSubmitCallback").thunk_name();
        let start = source
            .find(&format!("void {name}(void* state) {{"))
            .expect("forwarding thunk");
        let end = start + source[start..].find("\n}\n").expect("thunk end");
        assert_eq!(
            &source[start..end],
            format!(
                "void {name}(void* state) {{\n    pg6_gthunk_xtask_queue_submit_callback(state);"
            )
        );
        assert!(source.contains("#include \"guest/cb.h\""));
    }

    #[test]
    fn guest_state_thunks_sharing_a_name_key_bind_by_exact_name() {
        let mut headers = mini_headers();
        headers.push(ShimHeader {
            include: "guest/cb.h".to_string(),
            text: "extern \"C\" {\n\
                   void pg6_gthunk_exit(void* state);\n\
                   void pg6_gthunk__exit(void* state);\n\
                   void pg6_gthunk__cexit(void* state);\n\
                   void pg6_gthunk__c_exit(void* state);\n\
                   void pg6_gthunk_crt_atexit(void* state);\n\
                   void pg6_gthunk_foo_bar(void* state);\n\
                   void pg6_gthunk_foobar_(void* state);\n\
                   }\n"
                .to_string(),
        });
        let catalogue = Catalogue::scan(&headers);
        let mut pe = synthetic_pe(false);
        pe.imports.push(ImportModule {
            dll_name: "api-ms-win-crt-runtime-l1-1-0.dll".to_string(),
            iat_rva: 0x3000,
            symbols: named(&[
                "exit",
                "_exit",
                "_cexit",
                "_c_exit",
                "_crt_atexit",
                "FooBar",
            ]),
        });
        let plan = plan_bridge(&pe, &catalogue);
        let dll = "api-ms-win-crt-runtime-l1-1-0.dll";
        for (import, thunk) in [
            ("exit", "pg6_gthunk_exit"),
            ("_exit", "pg6_gthunk__exit"),
            ("_cexit", "pg6_gthunk__cexit"),
            ("_c_exit", "pg6_gthunk__c_exit"),
            ("_crt_atexit", "pg6_gthunk_crt_atexit"),
        ] {
            assert_eq!(
                find(&plan, &format!("{dll}!{import}")).bound_shim(),
                Some(thunk),
                "{import}"
            );
        }
        // The runtime's real declarations scan the same way.
        let runtime = Path::new(env!("CARGO_MANIFEST_DIR")).join("../runtime");
        let header = load_shim_header(
            &runtime.join("guest").join("guest_callbacks.h"),
            Some(&runtime),
        )
        .expect("guest_callbacks.h");
        let real = Catalogue::scan(&[header]);
        assert!(real.diagnostics.is_empty(), "{:?}", real.diagnostics);
        let real_plan = plan_bridge(&pe, &real);
        for import in ["exit", "_exit", "_cexit", "_c_exit", "_crt_atexit"] {
            let bound = find(&real_plan, &format!("{dll}!{import}"))
                .bound_shim()
                .map(str::to_string);
            assert_eq!(
                bound,
                Some(format!("pg6_gthunk_{import}").replace("pg6_gthunk__crt", "pg6_gthunk_crt")),
                "{import}"
            );
        }

        let foo = find(&plan, &format!("{dll}!FooBar"));
        assert_eq!(foo.bound_shim(), None);
        assert!(
            matches!(
                &foo.resolution,
                Resolution::Unresolved(Unresolved::Ambiguous { candidates }) if candidates.len() == 2
            ),
            "{:?}",
            foo.resolution
        );
    }

    #[test]
    fn thunks_marshal_win64_registers_stack_and_xmm() {
        let catalogue = mini_catalogue();
        let plan = plan_bridge(&rich_pe(), &catalogue);
        let source = render_bridge(&plan, &catalogue).source;
        let thunk = |label: &str| -> String {
            let name = find(&plan, label).thunk_name();
            let start = source
                .find(&format!("void {name}(void* state) {{"))
                .unwrap_or_else(|| panic!("thunk for {label} missing"));
            let end = start + source[start..].find("\n}\n").expect("thunk end");
            source[start..end].to_string()
        };
        let api = "api-ms-win-core-test-l1-1-0.dll";

        // 0 args, integer result.
        let t = thunk(&format!("{api}!GetTickCount"));
        assert!(
            t.contains("const auto result = pg6_win32_get_tick_count();"),
            "{t}"
        );
        assert!(
            t.contains("pg6_seam_set_gpr(state, 0, pg6_ret_int(result));"),
            "{t}"
        );
        assert!(t.contains("pg6_seam_return(state);"), "{t}");
        assert!(!t.contains("rsp"), "{t}");

        // 1 arg, void result leaves rax alone.
        let t = thunk(&format!("{api}!Sleep"));
        assert!(
            t.contains(
                "pg6_win32_sleep(\n        pg6_arg_int<uint32_t>(pg6_seam_get_gpr(state, 1)));"
            ),
            "{t}"
        );
        assert!(!t.contains("set_gpr(state, 0"), "{t}");

        // 4 integer args: rcx rdx r8 r9.
        let t = thunk("KERNEL32.dll!VirtualAlloc");
        for (i, reg) in [(0, 1), (1, 2), (2, 8), (3, 9)] {
            let ty = if i < 2 { "uint64_t" } else { "uint32_t" };
            assert!(
                t.contains(&format!(
                    "pg6_arg_int<{ty}>(pg6_seam_get_gpr(state, {reg}))"
                )),
                "{t}"
            );
        }
        assert!(!t.contains("rsp"), "{t}");

        // 8 mixed args: positional xmm for floats, stack slots past the shadow space.
        let t = thunk(&format!("{api}!MixArgs"));
        let expected = [
            "const uint64_t rsp = pg6_seam_get_gpr(state, 4);",
            "pg6_arg_int<int32_t>(pg6_seam_get_gpr(state, 1))",
            "pg6_f64_from_bits(pg6_seam_get_xmm_lo(state, 1))",
            "pg6_arg_ptr<const char*>(pg6_seam_get_gpr(state, 8))",
            "pg6_f32_from_bits(pg6_seam_get_xmm_lo(state, 3))",
            "pg6_arg_int<uint64_t>(pg6_seam_load_u64(rsp + kPg6SeamStackArgOffset + 0))",
            "pg6_f32_from_bits(pg6_seam_load_u64(rsp + kPg6SeamStackArgOffset + 8))",
            "pg6_arg_int<int8_t>(pg6_seam_load_u64(rsp + kPg6SeamStackArgOffset + 16))",
            "pg6_arg_ptr<Pg6OpaqueHandle>(pg6_seam_load_u64(rsp + kPg6SeamStackArgOffset + 24))",
            "pg6_seam_set_xmm_lo(state, 0, pg6_bits_from_f64(result));",
        ];
        let mut pos = 0;
        for line in expected {
            let found = t[pos..]
                .find(line)
                .unwrap_or_else(|| panic!("{line} in\n{t}"));
            pos += found + line.len();
        }

        let t = thunk(&format!("{api}!Scale"));
        assert!(
            t.contains("pg6_f32_from_bits(pg6_seam_get_xmm_lo(state, 0))"),
            "{t}"
        );
        assert!(
            t.contains("pg6_arg_int<bool>(pg6_seam_get_gpr(state, 2))"),
            "{t}"
        );
        assert!(t.contains("pg6_bits_from_f32(result)"), "{t}");

        let t = thunk(&format!("{api}!EchoPtr"));
        assert!(
            t.contains("pg6_seam_set_gpr(state, 0, pg6_seam_guest_addr(result));"),
            "{t}"
        );
    }

    #[test]
    fn unresolved_imports_get_logging_trap_stubs() {
        let catalogue = mini_catalogue();
        let plan = plan_bridge(&rich_pe(), &catalogue);
        let out = render_bridge(&plan, &catalogue);

        let foo = find(&plan, "unknown.dll!Foo");
        assert!(out.source.contains(&format!(
            "void {}(void* state) {{\n    static std::atomic<bool> logged{{false}};\n    \
             pg6_unresolved(state, logged, \"unresolved import unknown.dll!Foo (slot 0x3200): \
             no runtime shim family for this DLL\");\n}}",
            foo.thunk_name()
        )));
        assert!(out.source.contains(
            "\"unresolved import KERNEL32.dll!#42 (slot 0x20c8): imported by ordinal; no runtime \
             export table declares this ordinal\""
        ));
        assert!(
            out.source
                .contains("if (!logged.exchange(true)) {\n        pg6_seam_log(message);")
        );
        assert!(out.source.contains("pg6_seam_trap_unresolved(state);"));
        assert!(
            out.seam
                .contains("[[noreturn]] inline void pg6_seam_trap_unresolved")
        );
        assert!(out.seam.contains("std::abort();"));
        assert!(out.seam.contains("playgta6::os::log(message);"));
        // Pointers use the lifter's translation (host = base + VA), null maps to null.
        let base = crate::lift::GUEST_BASE_SYMBOL;
        assert!(out.seam.contains("#include \"abi/playgta6_abi.h\""));
        assert!(
            out.seam
                .contains(&format!("static_cast<uintptr_t>({base} + guest_va)"))
        );
        assert!(
            out.seam
                .contains(&format!("reinterpret_cast<uintptr_t>(host)) - {base}"))
        );
        assert!(out.seam.contains("return guest_va == 0 ? nullptr"));
        assert!(out.seam.contains("return host == nullptr"));

        // Imports without an IAT slot appear in the manifest only.
        let slotless = find(&plan, "KERNEL32!VirtualAlloc");
        assert!(!out.source.contains(&format!("{}(", slotless.thunk_name())));
        assert!(out.manifest.contains(&format!(
            "\"index\": {}, \"dll\": \"KERNEL32\", \"name\": \"VirtualAlloc\", \"ordinal\": null, \
             \"slot_rva\": null, \"delay_load\": false, \"thunk\": null, \"status\": \"bound\"",
            slotless.index
        )));
    }

    #[test]
    fn table_is_sorted_by_slot_and_flags_entries() {
        let catalogue = mini_catalogue();
        let mut pe = rich_pe();
        // Move a module below the others so table order differs from plan order.
        pe.imports.push(ImportModule {
            dll_name: "kernel32.dll".to_string(),
            iat_rva: 0x1800,
            symbols: named(&["Sleep"]),
        });
        let plan = plan_bridge(&pe, &catalogue);
        let source = render_bridge(&plan, &catalogue).source;
        let table = &source[source.find("const Pg6ImportThunk kThunks[] = {").unwrap()..];
        let table = &table[..table.find("};").unwrap()];
        let slots: Vec<u32> = table
            .lines()
            .filter_map(|l| l.trim().strip_prefix("{0x"))
            .map(|l| u32::from_str_radix(&l[..8], 16).unwrap())
            .collect();
        assert_eq!(
            slots.len(),
            plan.imports.iter().filter(|i| i.slot_rva.is_some()).count()
        );
        assert!(slots.windows(2).all(|w| w[0] < w[1]), "{slots:x?}");
        assert_eq!(slots[0], 0x1800);
        assert!(table.contains(
            "{0x000020c0u, kPg6ImportBound, 0u, \"KERNEL32.dll\", \"VirtualAlloc\", \
             \"pg6_win32_virtual_alloc\", &pg6_import_0},"
        ));
        assert!(table.contains(
            "{0x000020c8u, kPg6ImportByOrdinal, 42u, \"KERNEL32.dll\", nullptr, nullptr, \
             &pg6_import_1},"
        ));
        assert!(table.contains(&format!(
            "{{0x00002248u, kPg6ImportDelayLoad | kPg6ImportByOrdinal, 99u, \"d3d12_xs.dll\", \
             nullptr, nullptr, &{}}},",
            find(&plan, "d3d12_xs.dll!#99").thunk_name()
        )));
        assert!(source.contains(&format!("pg6_import_thunk_count = {};", slots.len())));
    }

    #[test]
    fn output_is_deterministic() {
        let headers = vec![
            ShimHeader {
                include: "os/a.h".to_string(),
                text: "extern \"C\" void pg6_win32_sleep(uint32_t ms);\n".to_string(),
            },
            ShimHeader {
                include: "os/b.h".to_string(),
                text: "extern \"C\" uint64_t pg6_win32_virtual_alloc(uint64_t a, uint64_t s, \
                       uint32_t t, uint32_t p);\n"
                    .to_string(),
            },
        ];
        let mut reversed = headers.clone();
        reversed.reverse();
        let pe = rich_pe();
        let render = |h: &[ShimHeader]| {
            let catalogue = Catalogue::scan(h);
            render_bridge(&plan_bridge(&pe, &catalogue), &catalogue)
        };
        let first = render(&headers);
        assert_eq!(first, render(&headers));
        assert_eq!(first, render(&reversed), "header order does not matter");
        assert!(
            first
                .source
                .contains("#include \"os/a.h\"\n#include \"os/b.h\"\n")
        );

        let dir = temp_dir("determinism");
        let written = first.write_to(&dir.join("one")).expect("write");
        let rewritten = render(&headers).write_to(&dir.join("two")).expect("write");
        assert_eq!(written.len(), 4);
        for (a, b) in written.iter().zip(&rewritten) {
            assert_eq!(a.file_name(), b.file_name());
            assert_eq!(
                fs::read(a).unwrap(),
                fs::read(b).unwrap(),
                "{}",
                a.display()
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_and_coverage_list_names_only() {
        let catalogue = mini_catalogue();
        let plan = plan_bridge(&rich_pe(), &catalogue);
        let out = render_bridge(&plan, &catalogue);
        let manifest = &out.manifest;
        assert!(manifest.starts_with("{\n  \"format\": \"playgta6-import-bridge\",\n"));
        assert!(manifest.contains("\"catalogue\": {\"headers\": 1, \"shims\": 18},"));
        assert!(
            manifest.contains("\"summary\": {\"imports\": 22, \"bound\": 11, \"unresolved\": 11},")
        );
        assert!(manifest.contains(
            "{\"index\": 0, \"dll\": \"KERNEL32.dll\", \"name\": \"VirtualAlloc\", \"ordinal\": null, \
             \"slot_rva\": \"0x20c0\", \"delay_load\": false, \"thunk\": \"pg6_import_0\", \
             \"status\": \"bound\", \"shim\": \"pg6_win32_virtual_alloc\", \
             \"header\": \"mini/pg6_test_shims.h\", \"params\": [\"int\", \"int\", \"int\", \"int\"], \
             \"ret\": \"int\"}"
        ));
        assert!(manifest.contains(&format!(
            "\"name\": null, \"ordinal\": 99, \"slot_rva\": \"0x2248\", \"delay_load\": true, \
             \"thunk\": \"{}\", \"status\": \"unresolved\", \"reason\": \"ordinal\"",
            find(&plan, "d3d12_xs.dll!#99").thunk_name()
        )));
        assert!(manifest.contains("\"reason\": \"ambiguous\""));
        assert!(manifest.contains("\"reason\": \"unknown_dll\""));
        assert!(manifest.contains("\"reason\": \"no_shim\""));
        assert!(manifest.contains("\"params\": [\"int\", \"f64\", \"ptr\", \"f32\", \"int\", \"f32\", \"int\", \"ptr\"], \"ret\": \"f64\""));
        let opens = manifest.matches('{').count();
        assert_eq!(opens, manifest.matches('}').count());
        assert_eq!(manifest.matches('[').count(), manifest.matches(']').count());

        let coverage = render_coverage(&plan, &catalogue);
        assert!(coverage.starts_with(
            "Import bridge: 22 imports, 11 bound, 11 unresolved (18 shims from 1 headers)\n"
        ));
        assert!(coverage.contains("unknown.dll!Foo: no runtime shim family for this DLL\n"));
        let row = |dll: &str| -> Vec<&str> {
            coverage
                .lines()
                .find(|l| l.trim_start().starts_with(&format!("{dll} ")))
                .map(|l| l.split_whitespace().collect())
                .unwrap_or_default()
        };
        assert_eq!(
            row("xgameruntime.dll"),
            ["xgameruntime.dll", "2", "1"],
            "{coverage}"
        );
        assert_eq!(
            row("KERNEL32.dll"),
            ["KERNEL32.dll", "2", "1"],
            "{coverage}"
        );
    }

    #[test]
    fn escapes_hostile_symbol_names() {
        let mut pe = synthetic_pe(false);
        pe.imports.push(ImportModule {
            dll_name: "evil\".dll".to_string(),
            iat_rva: 0x4000,
            symbols: named(&["a\\b\n*/\u{e9}"]),
        });
        let catalogue = mini_catalogue();
        let out = render_bridge(&plan_bridge(&pe, &catalogue), &catalogue);
        assert!(
            out.source
                .contains("// evil\".dll!a?b?*/? -> unresolved (slot 0x4000)\n")
        );
        assert!(
            out.source
                .contains("\"evil\\\".dll\", \"a\\\\b\\012*/\\303\\251\"")
        );
        assert!(
            out.manifest
                .contains("\"dll\": \"evil\\\".dll\", \"name\": \"a\\\\b\\n*/\u{e9}\"")
        );
    }

    #[test]
    fn empty_import_table_still_renders() {
        let mut pe = synthetic_pe(false);
        pe.imports.clear();
        pe.delay_imports.clear();
        let catalogue = mini_catalogue();
        let out = render_bridge(&plan_bridge(&pe, &catalogue), &catalogue);
        assert!(
            out.source
                .contains("const Pg6ImportThunk* const pg6_import_thunks = nullptr;")
        );
        assert!(out.source.contains("pg6_import_thunk_count = 0;"));
        assert!(
            out.manifest
                .contains("\"dlls\": [],\n  \"imports\": []\n}\n")
        );
    }

    // -----------------------------------------------------------------------------------------
    // Real runtime headers
    // -----------------------------------------------------------------------------------------

    pub(in crate::stubs) fn runtime_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("runtime")
    }

    fn real_headers() -> Vec<ShimHeader> {
        let root = runtime_root();
        let mut paths = Vec::new();
        for dir in ["abi", "os", "gpu"] {
            let entries = fs::read_dir(root.join(dir)).expect("runtime header dir");
            for entry in entries {
                let path = entry.expect("dir entry").path();
                if path.extension().is_some_and(|e| e == "h") {
                    paths.push(path);
                }
            }
        }
        paths.sort();
        paths
            .iter()
            .map(|p| load_shim_header(p, Some(&root)).expect("read runtime header"))
            .collect()
    }

    /// Catalogue of the real runtime headers plus a PE that imports every shim of the
    /// `pg6_win32_` / `pg6_gdk_` families by a name that keys to it.
    fn real_catalogue_and_pe() -> (Catalogue, Pe64) {
        let catalogue = Catalogue::scan(&real_headers());
        let mut pe = synthetic_pe(false);
        let mut iat = 0x10_000;
        for (dll, prefix) in [
            ("KERNEL32.dll", PREFIX_WIN32),
            ("xgameruntime.dll", PREFIX_GDK),
        ] {
            let names: Vec<String> = catalogue
                .family(prefix)
                .map(|s| s.name[prefix.len()..].replace('_', ""))
                .collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            pe.imports.push(ImportModule {
                dll_name: dll.to_string(),
                iat_rva: iat,
                symbols: named(&refs),
            });
            iat += 0x1000;
        }
        pe.imports.push(ImportModule {
            dll_name: "api-ms-win-core-synch-l1-2-0.dll".to_string(),
            iat_rva: iat,
            symbols: named(&[
                "CreateFileW",
                "GetTickCount64",
                "InitializeSRWLock",
                "AcquireSRWLockExclusive",
            ]),
        });
        pe.imports.push(ImportModule {
            dll_name: "WS2_32.dll".to_string(),
            iat_rva: iat + 0x1000,
            symbols: named(&["WSAStartup", "closesocket"]),
        });
        (catalogue, pe)
    }

    #[test]
    fn scans_real_runtime_headers() {
        use TypeKind::*;
        let (catalogue, pe) = real_catalogue_and_pe();
        let win32 = catalogue.family(PREFIX_WIN32).count();
        let gdk = catalogue.family(PREFIX_GDK).count();
        let d3d12 = catalogue.family(PREFIX_D3D12).count();
        let unsupported: Vec<String> = catalogue
            .iter()
            .filter_map(|s| Some(format!("{}: {}", s.name, s.unsupported_reason()?)))
            .collect();
        println!(
            "real runtime headers: {} headers, {} extern \"C\" shims ({win32} pg6_win32_, {gdk} \
             pg6_gdk_, {d3d12} pg6_d3d12_), {} not marshalable: {unsupported:#?}",
            catalogue.header_count,
            catalogue.len(),
            unsupported.len()
        );
        assert!(win32 >= 38, "O5 win32 shims present: {win32}");
        assert!(gdk >= 18, "O5 gdk shims present: {gdk}");
        assert!(
            catalogue.diagnostics.is_empty(),
            "{:?}",
            catalogue.diagnostics
        );

        let create = catalogue
            .get("pg6_win32_create_file_w")
            .expect("CreateFileW shim");
        assert_eq!(create.include, "os/win32_kernel.h");
        assert_eq!(kinds(create), [Ptr, Int, Int, Ptr, Int, Int, Int]);
        assert_eq!(create.ret.spelling, "Pg6Handle");
        assert_eq!(create.ret.kind, Int);
        let tls = catalogue.get("pg6_win32_tls_get_value").unwrap();
        assert_eq!((kinds(tls), tls.ret.kind), (vec![Int], Ptr));
        assert!(
            catalogue
                .get("pg6_win32_get_tick_count64")
                .unwrap()
                .params
                .is_empty()
        );
        let queue = catalogue.get("pg6_gdk_xtask_queue_create").unwrap();
        assert_eq!(queue.include, "os/xgameruntime.h");
        assert_eq!(kinds(queue), [Int, Int, Ptr]);
        assert!(
            catalogue
                .get("pg6_gdk_xsystem_get_analytics_info")
                .unwrap()
                .unsupported_reason()
                .unwrap()
                .contains("by value")
        );
        assert!(
            catalogue
                .get("pg6_gdk_xtask_queue_submit_callback")
                .unwrap()
                .unsupported_reason()
                .unwrap()
                .contains("function pointer")
        );

        // Every marshalable win32/gdk shim binds; the rest are reported, never bound.
        let plan = plan_bridge(&pe, &catalogue);
        for import in plan
            .imports
            .iter()
            .filter(|i| i.dll != "d3d12_xs.dll" && i.slot_rva.unwrap_or(0) >= 0x10_000)
        {
            match &import.resolution {
                Resolution::Bound { shim } => {
                    assert_eq!(catalogue.get(shim).unwrap().unsupported_reason(), None);
                }
                Resolution::Unresolved(why) => {
                    assert_eq!(
                        why.code(),
                        "unmarshalable",
                        "{}: {}",
                        import.label(),
                        why.describe()
                    );
                }
            }
        }
        let api = "api-ms-win-core-synch-l1-2-0.dll";
        for (name, shim) in [
            ("CreateFileW", "pg6_win32_create_file_w"),
            ("GetTickCount64", "pg6_win32_get_tick_count64"),
            ("InitializeSRWLock", "pg6_win32_initialize_srw_lock"),
            (
                "AcquireSRWLockExclusive",
                "pg6_win32_acquire_srw_lock_exclusive",
            ),
        ] {
            assert_eq!(
                find(&plan, &format!("{api}!{name}")).bound_shim(),
                Some(shim)
            );
        }
        assert_eq!(
            find(&plan, "WS2_32.dll!WSAStartup").bound_shim(),
            Some("pg6_win32_wsa_startup")
        );
        assert_eq!(
            find(&plan, "WS2_32.dll!closesocket").bound_shim(),
            Some("pg6_win32_closesocket")
        );
        assert_eq!(
            find(&plan, "xgameruntime.dll!XTaskQueueCreate").bound_shim(),
            Some("pg6_gdk_xtask_queue_create")
        );
        assert_eq!(
            find(&plan, "d3d12_xs.dll!D3D12CreateDevice").resolution,
            Resolution::Unresolved(Unresolved::NoShim {
                prefix: "pg6_d3d12_"
            }),
            "d3d12_xs exports D3D12XboxCreateDevice, not D3D12CreateDevice"
        );
    }

    /// The default shim set of this checkout (headers and export-table sources).
    pub(in crate::stubs) fn checkout_catalogue() -> Catalogue {
        let start = Path::new(env!("CARGO_MANIFEST_DIR"));
        let (paths, root) =
            crate::pipeline::checkout_shim_headers(start).expect("playgta6web checkout");
        let headers: Vec<ShimHeader> = paths
            .iter()
            .map(|p| load_shim_header(p, Some(&root)).expect("read runtime file"))
            .collect();
        Catalogue::scan(&headers)
    }

    #[test]
    fn real_export_tables_and_audit_cover_every_family_shim() {
        let catalogue = checkout_catalogue();
        assert!(
            catalogue.diagnostics.is_empty(),
            "{:#?}",
            catalogue.diagnostics
        );
        let mut per_family: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for entry in &catalogue.exports {
            if let Some(family) = family_of_symbol(&entry.symbol) {
                per_family
                    .entry(family.prefix)
                    .or_default()
                    .insert(entry.symbol.as_str());
            }
        }
        let count = |prefix: &str| per_family.get(prefix).map_or(0, BTreeSet::len);
        println!(
            "export tables: {} entries; symbols per family: {:?}",
            catalogue.exports.len(),
            per_family
                .iter()
                .map(|(k, v)| (*k, v.len()))
                .collect::<Vec<_>>()
        );
        assert!(count(PREFIX_CRT) >= 186, "ucrt_bridge table");
        assert!(count(PREFIX_MW) >= 98, "xbox_middleware table");
        assert!(count(PREFIX_XBOX) >= 16, "xbox_scarlett table");
        assert!(count(PREFIX_WS2) >= 25, "winsock_ordinals table");
        assert!(count(PREFIX_WIN32) >= 150, "shim_exports table");

        // Every exported shim of an audited family is declared and has exactly one audit row;
        // every audit row names a declared shim of an audited family.
        let mut problems = Vec::new();
        for family in SHIM_FAMILIES.iter().filter(|f| f.audited) {
            for symbol in per_family.get(family.prefix).into_iter().flatten() {
                if catalogue.get(symbol).is_none() {
                    problems.push(format!("{symbol}: exported but not declared in a header"));
                }
                if !catalogue.audit.contains_key(*symbol) {
                    problems.push(format!("{symbol}: no PG6_SHIM_AUDIT row"));
                }
            }
        }
        for row in catalogue.audit.values() {
            if catalogue.get(&row.symbol).is_none() {
                problems.push(format!("audit row {}: no such shim", row.symbol));
            }
            if !family_of_symbol(&row.symbol).is_some_and(|f| f.audited) {
                problems.push(format!("audit row {}: not an audited family", row.symbol));
            }
            if row.verdict != Verdict::Ok && row.note.trim().is_empty() {
                problems.push(format!(
                    "audit row {}: {} without a reason",
                    row.symbol,
                    row.verdict.as_str()
                ));
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));

        // A guest-state thunk registered for an export wins over the family shim registered for
        // the same export (static bridge; tests/test_shim_exports.cpp checks the registry).
        let resolver = ShimResolver::new(&catalogue);
        let mut both = 0;
        for entry in catalogue
            .exports
            .iter()
            .filter(|e| e.symbol.starts_with(PREFIX_GTHUNK))
        {
            let symbol = ImportedSymbol::ByName {
                hint: 0,
                name: entry.name.clone(),
            };
            assert_eq!(
                resolver.resolve(&entry.dll, &symbol),
                Resolution::Bound {
                    shim: entry.symbol.clone()
                },
                "{}!{}",
                entry.dll,
                entry.name
            );
            if catalogue.exports.iter().any(|e| {
                e.name == entry.name
                    && dll_key(&e.dll) == dll_key(&entry.dll)
                    && family_of_symbol(&e.symbol).is_some()
            }) {
                both += 1;
            }
        }
        assert!(
            both >= 4,
            "DStorageGetFactory, GameInputCreate, XG*: {both}"
        );
        let game_input = ImportedSymbol::ByOrdinal { ordinal: 1 };
        assert_eq!(
            resolver.resolve("GameInput.dll", &game_input),
            Resolution::Bound {
                shim: "pg6_gthunk_game_input_create".to_string()
            }
        );
        assert_eq!(
            catalogue.ordinal_import_name("WS2_32.dll", 115),
            Some("WSAStartup")
        );
    }

    /// DLLs of `docs/imports_required.md` that a program build loads as guest modules (the
    /// redistributable C++ runtime shipped with the game, R12): their imports bind to guest
    /// code, so the bridge is not asked.
    const DOC_GUEST_DLLS: [&str; 3] = ["msvcp140", "vcruntime140", "vcruntime140_1"];

    #[test]
    fn imports_required_doc_binds_or_records_a_reason() {
        let catalogue = checkout_catalogue();
        let doc = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("docs")
            .join("imports_required.md");
        let imports = parse_import_list(&fs::read_to_string(doc).expect("read imports doc"));
        assert_eq!(imports.len(), 587);
        let plan = plan_import_list(&imports, &catalogue);
        let mut by_status: BTreeMap<String, usize> = BTreeMap::new();
        let mut unrecorded = Vec::new();
        for import in &plan.imports {
            let status = if DOC_GUEST_DLLS.contains(&dll_key(&import.dll).as_str()) {
                "guest".to_string()
            } else if crate::pipeline::program::runtime_exception_api(&import.dll, &import.symbol)
                .is_some()
            {
                // Bound to R11's exception thunks by the runtime program loader.
                "runtime".to_string()
            } else {
                match &import.resolution {
                    Resolution::Bound { shim } if shim.starts_with(PREFIX_GTHUNK) => {
                        "gthunk".to_string()
                    }
                    Resolution::Bound { .. } => "host".to_string(),
                    Resolution::Unresolved(why) => {
                        if why.recorded().is_none() {
                            unrecorded.push(format!("{}: {}", import.label(), why.describe()));
                        }
                        format!("unresolved:{}", why.code())
                    }
                }
            };
            *by_status.entry(status).or_default() += 1;
        }
        println!("docs/imports_required.md: {by_status:?}");
        assert!(
            unrecorded.is_empty(),
            "imports unresolved without a recorded reason (add a shim, an audit row or a \
             PG6_IMPORT_GAPS row):\n{}",
            unrecorded.join("\n")
        );

        // The repository's coverage auditor reaches the same verdict from the bridge manifest
        // (tools/check_import_coverage.py --bridge-manifest), when a Python 3 is available.
        let dir = temp_dir("doc_manifest");
        let manifest = dir.join(MANIFEST_FILE);
        fs::write(&manifest, render_bridge(&plan, &catalogue).manifest).expect("write manifest");
        let tool = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tools")
            .join("check_import_coverage.py");
        let ran = ["python3", "python"].iter().find_map(|python| {
            Command::new(python)
                .arg(&tool)
                .arg("--bridge-manifest")
                .arg(&manifest)
                .output()
                .ok()
                .filter(|o| {
                    // Windows' `python3` app-execution alias exists but cannot run scripts.
                    !String::from_utf8_lossy(&o.stderr).contains("Microsoft Store")
                })
        });
        match ran {
            Some(output) => assert!(
                output.status.success(),
                "check_import_coverage.py --bridge-manifest failed:\n{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
            None => println!("no python found; skipping check_import_coverage.py"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------------------------
    // Tool-gated: compile (and run) the generated C++ with a host compiler when one exists.
    // -----------------------------------------------------------------------------------------

    pub(in crate::stubs) enum Cxx {
        Gnu(String),
        Msvc(String),
    }

    /// `$CXX`, else the first of `c++`, `g++`, `clang++`, `cl` that runs.
    pub(in crate::stubs) fn find_cxx() -> Option<Cxx> {
        let candidates: Vec<String> = match env::var("CXX") {
            Ok(cxx) if !cxx.trim().is_empty() => vec![cxx],
            _ => ["c++", "g++", "clang++", "cl"].map(String::from).to_vec(),
        };
        candidates.into_iter().find_map(|cxx| {
            let msvc = Path::new(&cxx)
                .file_stem()
                .is_some_and(|s| s.eq_ignore_ascii_case("cl"));
            let probe = if msvc {
                Command::new(&cxx).arg("/?").output()
            } else {
                Command::new(&cxx).arg("--version").output()
            };
            let cxx = if msvc { Cxx::Msvc(cxx) } else { Cxx::Gnu(cxx) };
            probe.is_ok_and(|o| o.status.success()).then_some(cxx)
        })
    }

    /// Compile `sources`; link into `exe` when given, otherwise check syntax only.
    pub(in crate::stubs) fn compile(
        cxx: &Cxx,
        sources: &[PathBuf],
        includes: &[PathBuf],
        exe: Option<&Path>,
        strict: bool,
        work_dir: &Path,
    ) {
        let mut cmd = match cxx {
            Cxx::Gnu(c) => {
                let mut cmd = Command::new(c);
                cmd.arg("-std=c++20");
                if strict {
                    cmd.args(["-Wall", "-Wextra", "-Werror"]);
                }
                for inc in includes {
                    cmd.arg(format!("-I{}", inc.display()));
                }
                cmd.args(sources);
                match exe {
                    Some(exe) => cmd.arg("-o").arg(exe),
                    None => cmd.arg("-fsyntax-only"),
                };
                cmd
            }
            Cxx::Msvc(c) => {
                let mut cmd = Command::new(c);
                cmd.args(["/nologo", "/std:c++20", "/EHsc"]);
                if strict {
                    cmd.args(["/W4", "/WX"]);
                }
                for inc in includes {
                    cmd.arg(format!("/I{}", inc.display()));
                }
                cmd.args(sources);
                match exe {
                    Some(exe) => cmd.arg(format!("/Fe{}", exe.display())),
                    None => cmd.arg("/Zs"),
                };
                cmd
            }
        };
        let output = cmd
            .current_dir(work_dir)
            .output()
            .expect("run C++ compiler");
        assert!(
            output.status.success(),
            "C++ compile failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    const HARNESS: &str = r#"// Executes generated import thunks against host-side fake shims.
#include "pg6_import_thunks.h"
#include "pg6_guest_seam.h"
#include "mini/pg6_test_shims.h"

#include <cstdio>
#include <cstring>
#include <string_view>
#ifdef _MSC_VER
#include <stdlib.h>
#endif

namespace playgta6::os {
void log(std::string_view message) {
    std::fprintf(stderr, "LOG: %.*s\n", static_cast<int>(message.size()), message.data());
    std::fflush(stderr);
}
}  // namespace playgta6::os

// The runtime's guest -> host translation base. Identity, like the browser build: the fake shims
// store host pointers through out-parameters (`*queue = &g_marker`), which the bridge does not
// translate (see "Guest-state seam" in stubs.rs).
extern "C" {
uint64_t __playgta6_guest_base = 0;
}

namespace {
int g_failures = 0;
#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #cond);   \
            ++g_failures;                                                             \
        }                                                                             \
    } while (0)

alignas(16) unsigned char g_state[416];
uint64_t g_stack[32];
const char g_text[] = "hi";
const char16_t g_wide[] = u"ok";
int g_marker = 0;
uint32_t g_slept = 0;
uint64_t g_alloc[4];

uint64_t addr(const void* p) { return pg6_seam_guest_addr(p); }
uint64_t f32_bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
uint64_t f64_bits(double v) { uint64_t b; std::memcpy(&b, &v, 8); return b; }
uint64_t rax() { return pg6_seam_get_gpr(g_state, 0); }

void reset() {
    std::memset(g_state, 0, sizeof g_state);
    std::memset(g_stack, 0, sizeof g_stack);
    g_stack[8] = 0x1400010FFull;  // return address pushed by the guest call
    pg6_seam_set_gpr(g_state, 4, addr(&g_stack[8]));
}

void call_slot(uint32_t slot) {
    const Pg6ImportThunk* thunk = pg6_find_import_thunk(slot);
    CHECK(thunk != nullptr);
    if (thunk == nullptr) {
        return;
    }
    const uint64_t rsp = pg6_seam_get_gpr(g_state, 4);
    thunk->thunk(g_state);
    CHECK(pg6_seam_get_gpr(g_state, 4) == rsp + 8);
}
}  // namespace

extern "C" uint64_t pg6_win32_virtual_alloc(uint64_t address, uint64_t size, uint32_t type,
                                            uint32_t protect) {
    g_alloc[0] = address;
    g_alloc[1] = size;
    g_alloc[2] = type;
    g_alloc[3] = protect;
    return 0x7000000u;
}
extern "C" uint32_t pg6_win32_get_tick_count(void) { return 0xFFFFFFF0u; }
extern "C" void pg6_win32_sleep(uint32_t milliseconds) { g_slept = milliseconds; }
extern "C" double pg6_win32_mix_args(int32_t a, double b, const char* c, float d, uint64_t e,
                                     float f, int8_t g, Pg6OpaqueHandle h) {
    CHECK(a == -5);
    CHECK(b == 2.5);
    CHECK(c == g_text);
    CHECK(d == 1.25f);
    CHECK(e == 0x1122334455667788ull);
    CHECK(f == -3.5f);
    CHECK(g == -2);
    CHECK(h == reinterpret_cast<Pg6OpaqueHandle>(&g_marker));
    return 42.75;
}
extern "C" float pg6_win32_scale(float x, bool flag) { return flag ? x * 2.0f : x; }
extern "C" Pg6Opaque* pg6_win32_echo_ptr(Pg6Opaque* p) { return p; }
extern "C" Pg6Handle pg6_win32_open_thing(const char16_t* name, unsigned int flags,
                                          std::size_t len) {
    return (name == g_wide && flags == 3u && len == 2u) ? 0x55u : 0u;
}
extern "C" Pg6Hr pg6_gdk_xtask_queue_create(Pg6Mode work_mode, Pg6Mode completion_mode,
                                            Pg6OpaqueHandle* queue) {
    CHECK(work_mode == Pg6Mode::B);
    CHECK(completion_mode == Pg6Mode::A);
    *queue = reinterpret_cast<Pg6OpaqueHandle>(&g_marker);
    return static_cast<Pg6Hr>(0x80004005u);
}
extern "C" Pg6Hr pg6_d3d12_d3d12_create_device(void* adapter, uint32_t feature_level,
                                               const void* riid, void** device) {
    CHECK(riid == nullptr);
    *device = adapter;
    return feature_level == 0xC000u ? 0 : -1;
}
extern "C" uint32_t pg6_gdk_single_decl(uint8_t x) noexcept { return x; }

int main(int argc, char** argv) {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    if (argc > 1 && std::strcmp(argv[1], "unresolved") == 0) {
        reset();
        call_slot(SLOT_UNKNOWN_FOO);
        std::puts("RETURNED");
        return 0;
    }

    // Table: sorted, searchable, flagged.
    CHECK(pg6_import_thunk_count == EXPECTED_COUNT);
    for (std::size_t i = 1; i < pg6_import_thunk_count; ++i) {
        CHECK(pg6_import_thunks[i - 1].slot_rva < pg6_import_thunks[i].slot_rva);
    }
    CHECK(pg6_find_import_thunk(0x1u) == nullptr);
    CHECK(pg6_find_import_thunk(0xFFFFFFFFu) == nullptr);
    CHECK(pg6_find_import_thunk(0x20C0u)->flags == kPg6ImportBound);
    CHECK(std::strcmp(pg6_find_import_thunk(0x20C0u)->shim, "pg6_win32_virtual_alloc") == 0);
    CHECK(pg6_find_import_thunk(0x2248u)->flags == (kPg6ImportDelayLoad | kPg6ImportByOrdinal));
    CHECK(pg6_find_import_thunk(0x2248u)->ordinal == 99u);
    CHECK(pg6_find_import_thunk(0x2248u)->symbol == nullptr);

    // 4 integer args in rcx rdx r8 r9; u32 parameters see only the low 32 bits.
    reset();
    pg6_seam_set_gpr(g_state, 1, 0x10000u);
    pg6_seam_set_gpr(g_state, 2, 0x2000u);
    pg6_seam_set_gpr(g_state, 8, 0xDEADBEEF00003000ull);
    pg6_seam_set_gpr(g_state, 9, 0xFFFFFFFF00000004ull);
    call_slot(0x20C0u);
    CHECK(g_alloc[0] == 0x10000u && g_alloc[1] == 0x2000u);
    CHECK(g_alloc[2] == 0x3000u && g_alloc[3] == 4u);
    CHECK(rax() == 0x7000000u);

    // 0 args: unsigned result zero-extended into rax.
    reset();
    pg6_seam_set_gpr(g_state, 0, 0x1234u);
    call_slot(SLOT_GET_TICK_COUNT);
    CHECK(rax() == 0xFFFFFFF0u);

    // 1 arg, void result: rax untouched.
    reset();
    pg6_seam_set_gpr(g_state, 0, 0x77u);
    pg6_seam_set_gpr(g_state, 1, 0xAAAAAAAA00000064ull);
    call_slot(SLOT_SLEEP);
    CHECK(g_slept == 100u);
    CHECK(rax() == 0x77u);

    // 8 mixed args: floats by position in xmm1/xmm3, then stack slots past the shadow space.
    reset();
    pg6_seam_set_gpr(g_state, 1, 0x12345678FFFFFFFBull);
    pg6_seam_set_gpr(g_state, 2, 0x9999u);
    pg6_seam_set_xmm_lo(g_state, 1, f64_bits(2.5));
    pg6_seam_set_gpr(g_state, 8, addr(g_text));
    pg6_seam_set_xmm_lo(g_state, 3, 0xCCCCCCCC00000000ull | f32_bits(1.25f));
    g_stack[8 + 5] = 0x1122334455667788ull;
    g_stack[8 + 6] = 0xAAAAAAAA00000000ull | f32_bits(-3.5f);
    g_stack[8 + 7] = 0x55555555555555FEull;
    g_stack[8 + 8] = addr(&g_marker);
    call_slot(SLOT_MIX_ARGS);
    CHECK(pg6_seam_get_xmm_lo(g_state, 0) == f64_bits(42.75));

    // float in xmm0, bool from the low byte of rdx, float result in xmm0.
    reset();
    pg6_seam_set_xmm_lo(g_state, 0, f32_bits(1.5f));
    pg6_seam_set_gpr(g_state, 2, 0x100u);
    call_slot(SLOT_SCALE);
    CHECK(static_cast<uint32_t>(pg6_seam_get_xmm_lo(g_state, 0)) == f32_bits(1.5f));
    reset();
    pg6_seam_set_xmm_lo(g_state, 0, f32_bits(1.5f));
    pg6_seam_set_gpr(g_state, 2, 0xFF01u);
    call_slot(SLOT_SCALE);
    CHECK(static_cast<uint32_t>(pg6_seam_get_xmm_lo(g_state, 0)) == f32_bits(3.0f));

    // Pointer in and out; null stays null.
    reset();
    pg6_seam_set_gpr(g_state, 1, addr(&g_marker));
    call_slot(SLOT_ECHO_PTR);
    CHECK(rax() == addr(&g_marker));
    reset();
    pg6_seam_set_gpr(g_state, 0, 0x99u);
    call_slot(SLOT_ECHO_PTR);
    CHECK(rax() == 0u);

    // Aliased types.
    reset();
    pg6_seam_set_gpr(g_state, 1, addr(g_wide));
    pg6_seam_set_gpr(g_state, 2, 3u);
    pg6_seam_set_gpr(g_state, 8, 2u);
    call_slot(SLOT_OPEN_THING);
    CHECK(rax() == 0x55u);

    // Enum args, out-pointer, negative HRESULT sign-extended.
    reset();
    uint64_t queue = 0;
    pg6_seam_set_gpr(g_state, 1, 1u);
    pg6_seam_set_gpr(g_state, 2, 0u);
    pg6_seam_set_gpr(g_state, 8, addr(&queue));
    call_slot(0x20E0u);
    CHECK(queue == addr(&g_marker));
    CHECK(rax() == 0xFFFFFFFF80004005ull);

    // Delay-load slot.
    reset();
    uint64_t device = 0;
    pg6_seam_set_gpr(g_state, 1, addr(&g_marker));
    pg6_seam_set_gpr(g_state, 2, 0xC000u);
    pg6_seam_set_gpr(g_state, 9, addr(&device));
    call_slot(0x2240u);
    CHECK(device == addr(&g_marker));
    CHECK(rax() == 0u);

    // uint8_t parameter with a default argument in the declaration.
    reset();
    pg6_seam_set_gpr(g_state, 1, 0x1FFu);
    call_slot(SLOT_SINGLE_DECL);
    CHECK(rax() == 0xFFu);

    if (g_failures == 0) {
        std::puts("PG6_THUNKS_OK");
    }
    return g_failures == 0 ? 0 : 1;
}
"#;

    #[test]
    fn generated_bridge_compiles_and_marshals_tool_gated() {
        let Some(cxx) = find_cxx() else {
            println!("note: no C++ compiler found ($CXX, c++, g++, clang++, cl); skipping");
            return;
        };
        let catalogue = mini_catalogue();
        let plan = plan_bridge(&rich_pe(), &catalogue);
        let dir = temp_dir("compile_run");
        let out_dir = dir.join("out");
        render_bridge(&plan, &catalogue)
            .write_to(&out_dir)
            .expect("write bridge");
        fs::create_dir_all(dir.join("mini")).unwrap();
        fs::write(dir.join(MINI_INCLUDE), MINI_HEADER).unwrap();

        let slot = |label: &str| format!("0x{:X}u", find(&plan, label).slot_rva.unwrap());
        let api = "api-ms-win-core-test-l1-1-0.dll";
        let mut harness = HARNESS.to_string();
        for (key, label) in [
            ("SLOT_UNKNOWN_FOO", "unknown.dll!Foo".to_string()),
            ("SLOT_GET_TICK_COUNT", format!("{api}!GetTickCount")),
            ("SLOT_SLEEP", format!("{api}!Sleep")),
            ("SLOT_MIX_ARGS", format!("{api}!MixArgs")),
            ("SLOT_SCALE", format!("{api}!Scale")),
            ("SLOT_ECHO_PTR", format!("{api}!EchoPtr")),
            ("SLOT_OPEN_THING", format!("{api}!OpenThing")),
            (
                "SLOT_SINGLE_DECL",
                "XGameRuntime.DLL!SingleDecl".to_string(),
            ),
        ] {
            harness = harness.replace(key, &slot(&label));
        }
        let tabled = plan.imports.iter().filter(|i| i.slot_rva.is_some()).count();
        harness = harness.replace("EXPECTED_COUNT", &format!("{tabled}u"));
        let main_cpp = dir.join("harness.cpp");
        fs::write(&main_cpp, harness).unwrap();

        let exe = dir.join(if cfg!(windows) {
            "harness.exe"
        } else {
            "harness"
        });
        compile(
            &cxx,
            &[out_dir.join(SOURCE_FILE), main_cpp],
            &[out_dir.clone(), dir.clone(), runtime_root()],
            Some(&exe),
            true,
            &dir,
        );

        let run = Command::new(&exe).output().expect("run harness");
        let stdout = String::from_utf8_lossy(&run.stdout);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(run.status.success(), "harness failed:\n{stdout}\n{stderr}");
        assert!(stdout.contains("PG6_THUNKS_OK"), "{stdout}\n{stderr}");

        // Unresolved import: logged, then trapped (never returns).
        let run = Command::new(&exe)
            .arg("unresolved")
            .output()
            .expect("run harness");
        let stdout = String::from_utf8_lossy(&run.stdout);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(!run.status.success(), "unresolved import must trap");
        assert!(!stdout.contains("RETURNED"), "{stdout}");
        assert!(
            stderr.contains(
                "LOG: unresolved import unknown.dll!Foo (slot 0x3200): no runtime shim family \
                 for this DLL"
            ),
            "{stderr}"
        );
        println!("compiled and ran generated thunks with the host C++ compiler");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn generated_bridge_for_real_headers_compiles_tool_gated() {
        let Some(cxx) = find_cxx() else {
            println!("note: no C++ compiler found ($CXX, c++, g++, clang++, cl); skipping");
            return;
        };
        let (catalogue, pe) = real_catalogue_and_pe();
        let plan = plan_bridge(&pe, &catalogue);
        let dir = temp_dir("compile_real");
        render_bridge(&plan, &catalogue)
            .write_to(&dir)
            .expect("write bridge");
        compile(
            &cxx,
            &[dir.join(SOURCE_FILE)],
            &[dir.clone(), runtime_root()],
            None,
            false,
            &dir,
        );
        println!(
            "generated thunks for {} real shims compile against runtime/ headers",
            plan.bound_count()
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The bridge for the executable's real import list (docs/imports_required.md) against the
    /// checkout's default shim set: every family header the bound shims come from is included
    /// into one translation unit, as in the program build.
    #[test]
    fn generated_bridge_for_imports_required_compiles_tool_gated() {
        let Some(cxx) = find_cxx() else {
            println!("note: no C++ compiler found ($CXX, c++, g++, clang++, cl); skipping");
            return;
        };
        let catalogue = checkout_catalogue();
        let doc = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("docs")
            .join("imports_required.md");
        let imports = parse_import_list(&fs::read_to_string(doc).expect("read imports doc"));
        let mut pe = synthetic_pe(false);
        pe.imports.clear();
        pe.delay_imports.clear();
        let mut iat = 0x10_000u32;
        for (dll, symbol) in imports {
            match pe.imports.last_mut() {
                Some(module) if module.dll_name == dll => module.symbols.push(symbol),
                _ => {
                    iat += 0x1000;
                    pe.imports.push(ImportModule {
                        dll_name: dll,
                        iat_rva: iat,
                        symbols: vec![symbol],
                    });
                }
            }
        }
        let plan = plan_bridge(&pe, &catalogue);
        let includes: BTreeSet<&str> = plan
            .imports
            .iter()
            .filter_map(|i| catalogue.get(i.bound_shim()?))
            .map(|s| s.include.as_str())
            .collect();
        assert!(includes.contains("os/ucrt_bridge.h"), "{includes:?}");
        assert!(includes.contains("os/xbox_scarlett.h"), "{includes:?}");
        let dir = temp_dir("compile_doc");
        render_bridge(&plan, &catalogue)
            .write_to(&dir)
            .expect("write bridge");
        compile(
            &cxx,
            &[dir.join(SOURCE_FILE)],
            &[dir.clone(), runtime_root()],
            None,
            false,
            &dir,
        );
        println!(
            "generated thunks for {} imports_required.md bindings compile ({} headers)",
            plan.bound_count(),
            includes.len()
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
