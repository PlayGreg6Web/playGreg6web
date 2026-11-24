//! Multi-module program builds: the executable plus the DLLs shipped next to it, each built as a
//! guest module with the single-module stages ([`super::run_module`]) and linked into one guest
//! program. `playgta6-recomp build-program` wraps [`build_program`].
//!
//! # Steps
//! 1. **Inputs.** The executable and the supplied DLLs (`--dll <file>`, every `*.dll` directly in
//!    a `--dll-dir`), keyed by canonical name (basename, ASCII-lowercased, `.dll` appended when
//!    there is no extension: Windows' rule, see [`canonical_dll`]); two files with one canonical
//!    name are an error.
//! 2. **Module selection.** Starting from the executable, every imported DLL (normal and delay
//!    load) and every forwarder target that is supplied and whose [`Role`] is guest becomes a
//!    guest module, transitively; an API-set name (`api-ms-win-crt-*`) selects the module its
//!    contract maps to ([`API_SET_RULES`]) when that is supplied. `--guest-dll` adds a supplied DLL
//!    even when nothing imports it. A dependency that is not a guest module is classified as
//!    `host` (the runtime implements it: a shim family of the import bridge or a module the
//!    runtime registers), `host_only` (supplied, but the policy keeps it on the host side) or
//!    `missing` (neither supplied nor known to the runtime).
//! 3. **Initialisation order.** Depth-first over guest-module dependencies in import-table
//!    order, executable last (its dependencies, then the `--guest-dll` modules sorted by name);
//!    dependencies come before their importers. An edge that closes a cycle is dropped from the
//!    ordering and reported under `cycles` (the order stays deterministic).
//! 4. **Base assignment** ([`rebase::assign_bases`]), in the order executable, then initialisation
//!    order: a module keeps its preferred base when `[base, base + SizeOfImage rounded up to
//!    64 KiB)` overlaps no module placed before it; otherwise it moves to the lowest 64 KiB-aligned
//!    address at or above its preferred base where it fits, below `0x7FF8_0000_0000` (the
//!    runtime's virtual module handles). The executable always keeps its base. A module with
//!    `IMAGE_FILE_RELOCS_STRIPPED` that would have to move is an error.
//! 5. **Build-time rebasing** ([`rebase::rebase_pe`]): a moved module's file bytes get its base
//!    relocations applied and `ImageBase` rewritten before anything else reads them, so
//!    discovery, the lifted `sub_<VA>` functions and their constants, `.pdata`/EH VAs, exports,
//!    IAT slots, TLS and the packed image all agree with the new base, and the runtime maps the
//!    image there without relocating it. Building a module at base A and rebasing it to B gives
//!    exactly the files of building it at B.
//! 6. **Import binding** (below), then the allocator-family check.
//! 7. **Module builds**: each module goes through [`super::run_module`] into `<out>/<module>/`
//!    (the single-module layout) with its [`ModuleLink`]: guest-bound IAT slots are pre-filled in
//!    `image/guest.pgim`, the import bridge (`stubs/`) uses the module namespace
//!    ([`stubs::render_bridge_namespaced`]) and the registration table carries the module name.
//!    Every lifted function must lie in its module's range and no `sub_<VA>` may appear in two
//!    modules (checked; a violation fails the build).
//! 8. `program-manifest.json`, `program-report.txt` and the runtime descriptor `pg6_program.cpp`.
//!
//! Module names (output directory and symbol suffix): `main` for the executable, the lower-cased
//! DLL file stem with every character outside `[a-z0-9_]` replaced by `_` for a DLL
//! ([`module_symbol`]; `libHttpClient.GDK.dll` -> `libhttpclient_gdk`).
//!
//! # Import binding policy
//! Every import of every module (the executable and the guest DLLs alike) is bound to exactly
//! one of: an export of a guest module (its IAT slot holds the export's real guest VA, so
//! `call [rip+slot]` reaches the lifted function through the dispatcher and a data import reads
//! the variable), the host shim bridge (the slot gets a synthetic address bound to a thunk at
//! load time, R5/O9), or, when the bridge has no shim, the bridge's logging thunk (unresolved).
//! Imports of DLLs that are not guest modules always go to the bridge. For a guest module's
//! export the first matching rule of the [`Policy`] decides (`dll!symbol=action`, `*`
//! wildcards; DLL patterns match canonical names after API-set mapping, symbol patterns the
//! export name or `#<ordinal>`):
//!
//! | Rule (first match wins) | Action | Why |
//! | :--- | :--- | :--- |
//! | `vcruntime140.dll!` `memcpy memmove memset memcmp memchr strchr strrchr strstr wcschr wcsrchr wcsstr` | `prefer-host` | stateless primitives: the host shim is faster and equivalent, taken whenever the bridge binds it |
//! | `vcruntime140*.dll!` `_CxxThrowException __CxxFrameHandler* __C_specific_handler* _setjmp* __intrinsic_setjmp* longjmp __RT* __std_exception_* __std_type_info_* __current_exception* __uncaught_exception* _purecall __std_terminate` (and the other EH helpers) | `guest` (required) | must agree with the guest's own `ThrowInfo`/`FuncInfo`/`type_info`/`jmp_buf` layouts and with R11's design (OS unwinding in the runtime, language handlers as guest code) |
//! | `msvcp140*.dll vcruntime140*.dll concrt140.dll vccorlib140.dll vcomp140.dll ucrtbase.dll` `!*` | `guest` | C++ runtime objects (iostreams, locales, mutexes) have MSVC layouts the guest code shares |
//! | `*!*` | `guest` | a supplied DLL the program imports is used as shipped |
//!
//! Actions: `guest` binds the export; `host` binds the bridge even though a guest module exports
//! the symbol (a *data* export is always bound to the guest: a thunk address is not a variable;
//! reported as `data_export_forced_guest`); `prefer-host` binds the bridge when it binds the
//! symbol to a shim and the guest export otherwise. A required rule overridden to `host` is
//! reported (`required_rule_overridden`), and a required rule whose DLL is not a guest module
//! (`required_guest_unavailable`, e.g. `vcruntime140.dll` not supplied) leaves the import on the
//! bridge.
//!
//! DLL roles: `bink2scarlett.dll xcurl.dll libhttpclient.gdk.dll xboxbvhbuilder.dll
//! amd_fidelityfx_loader_xbox.dll amd_fidelityfx_upscaler_xbox.dll` are host-only by default: the
//! runtime replaces them on purpose (offline networking, video and GPU paths, O14), so they are
//! never built and their imports go to the bridge. Every other supplied DLL can be a guest
//! module.
//!
//! Overrides take precedence over the table, in the order given: `--bind <dll>!<symbol>=<action>`,
//! `--guest-dll <dll>` (guest role, built even when unreferenced), `--host-dll <dll>` (host-only),
//! and `--policy <file>` with one directive per line (`<dll>!<symbol>=<action>`, `<dll>=guest`,
//! `<dll>=host`, `#` comments). The manifest records the rule that decided every import.
//!
//! **Allocator consistency.** `malloc free calloc realloc _recalloc _msize _expand _aligned_*
//! *_base` and `operator new`/`delete` (`??2@* ??3@* ??_U@* ??_V@*`) imported from the CRT DLLs
//! (`ucrtbase`, `api-ms-win-crt-*`, `msvcrt`, `vcruntime140*`, `msvcp140*`, `concrt140`) must
//! all resolve to one provider (the host, or one guest module) across every importer: a block
//! allocated on one side and freed on the other corrupts both heaps. A split fails the build
//! unless `--allow-allocator-split` is given (then it is a diagnostic). Every other symbol is
//! bound by `(DLL, symbol)` alone, so it is consistent across importers by construction (also
//! checked).
//!
//! **Forwarders.** `DLL.Name` / `DLL.#ordinal` strings are followed across guest modules (API-set
//! names mapped first); a chain that ends in a DLL that is not a guest module is bound to the
//! bridge under the final name (`host_redirect`, e.g. a forward to `KERNEL32.HeapAlloc`), a
//! cycle or a missing export leaves the import on the bridge under its own name with a
//! diagnostic.
//!
//! # IAT pre-fill
//! Guest-bound slots are written into the module's guest image at build time
//! ([`super::prefill_iat`]): deterministic, zero load-time cost, and the runtime cannot get it
//! wrong. Guest bases are fixed by the build, so the values never need relocating; a program's
//! images must be mapped at exactly their assigned base (the runtime program loader refuses a
//! fallback base). The import bridge leaves those slots out of its thunk table, so binding at
//! load time never overwrites them. The manifest lists every pre-filled slot per module.
//!
//! # Output layout
//! ```text
//! program-manifest.json   this program: modules, bases, bindings, order (schema below)
//! program-report.txt      human-readable summary (also printed by the CLI)
//! pg6_program.cpp         Pg6Program descriptor for runtime/guest/program_loader.h
//! <module>/...            one single-module build per module (see the parent module)
//! ```
//! `pg6_program.cpp` includes `guest/program_loader.h` and every module's
//! `<module>/stubs/pg6_import_thunks.h` (compile it with the build root on the include path)
//! and defines `extern "C" const Pg6Program pg6_program`: the modules in initialisation order
//! (executable last), each with file name, module name, assigned base, `SizeOfImage`, entry
//! point VA, flags (`kPg6ProgramModuleMain`, `kPg6ProgramModuleDll`, `kPg6ProgramModuleTls`),
//! TLS index, its lifted-function table (`__playgta6_lifted_functions_<module>` and its count)
//! and its import bridge table (`pg6_import_thunks_<module>` / `pg6_import_thunk_count_<module>`,
//! the `Pg6BootImport` layout of `guest/boot.h`).
//!
//! # Manifest schema (`playgta6-recomp-program`, `schema_version` 1)
//! Top-level keys in order: `schema`, `schema_version`, `tool` {`name`, `version`}, `options`
//! {`allow_allocator_split`, `overrides` [rule strings], `force_guest` [dll], `build` {`emit_obj`,
//! `shard_size`, `shard_instructions`, `map_headers`, `executable_bytes`}}, `modules` [in
//! initialisation order: `file`, `dll`, `module`, `role` (`exe`/`dll`), `reason` (`executable`,
//! `import`, `forwarder`, `forced`), `input` {`size`, `fnv1a64`}, `built_fnv1a64` (bytes
//! actually built, after rebasing), `preferred_base`, `base`, `rebased`, `size_of_image`,
//! `range_end` (base + reserved size), `entry_point` (VA or null; `DllMain` for a DLL), `tls`
//! (null or {`index`, `start`, `end`, `index_va`, `callbacks` [VA], `zero_fill`,
//! `characteristics`}), `exports` {`total`, `functions`, `data`, `forwarders`, `ordinal_only`},
//! `imports` {`total`, `guest`, `host_bound`, `host_unresolved`, `before` {`bound`,
//! `unresolved`}}, `depends_on` [dll], `iat_prefill` [{`slot_rva`, `value`, `import`}],
//! `registration` {`register_function`, `table_symbol`, `count`, `first`, `last`}, `bridge`
//! {`thunks_symbol`, `thunk_count_symbol`}, `lift` {`functions`, `instructions`, `unsupported`},
//! `build_manifest` {`path`, `fnv1a64`}], `init_order` [dll], `cycles` [{`from`, `to`}],
//! `imports` [{`importer`, `dll`, `name` | null, `ordinal` | null, `slot_rva` | null,
//! `delay_load`, `binding` (`guest`, `host`, `unresolved`), `provider` (dll or null), `target`
//! (VA or null), `kind` (`function`/`data` or null), `shim` | null, `reason` | null, `via`
//! [forwarders], `rule`}], `external_dlls` [{`dll`, `status` (`host`, `host_only`, `missing`),
//! `importers` [dll]}], `unreferenced_dlls` [dll], `allocator` {`provider` (`host`, a dll or
//! null), `imports`}, `diagnostics` [{`kind`, `module`, `detail`}], `files` [{`path`, `size`,
//! `fnv1a64`}] (the program-level files except the manifest). Addresses are `"0x..."` strings.
//!
//! # Determinism
//! Inputs are keyed by canonical name, every order above is derived from file contents, and each
//! module build is deterministic, so the outputs are byte-identical across runs, `--jobs` values
//! and the order of `--dll`/`--dll-dir` arguments.

mod link;
pub mod policy;
mod rebase;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{self, Write as _};
use std::fs;
use std::path::{Path, PathBuf};

use super::{
    Input, Json, ModuleLink, OutputFile, PipelineError, PipelineOptions, PipelineReport,
    check_options, hex, io_error, load_input, num, obj, register_function_symbol,
    registration_symbols, run_module, write_file,
};
use crate::image::fnv1a_64;
use crate::pe::{ImportedSymbol, Pe64};
use crate::stubs::{self, Catalogue, GuestTarget, Resolution, ShimResolver};

pub use link::{API_SET_RULES, ExportKind, api_set_host, canonical_dll, module_symbol};
pub use policy::{Action, Overrides, Policy, Role};
pub use rebase::{ADDRESS_LIMIT, ALLOCATION_GRANULARITY};

use link::{ExportIndex, ExportSummary, ModuleExports, Provider, symbol_text};

/// Program manifest file name (in the build root).
pub const PROGRAM_MANIFEST_FILE: &str = "program-manifest.json";
/// Program report file name.
pub const PROGRAM_REPORT_FILE: &str = "program-report.txt";
/// Runtime program descriptor source.
pub const PROGRAM_DESCRIPTOR_FILE: &str = "pg6_program.cpp";
/// The runtime header the descriptor includes.
pub const PROGRAM_LOADER_INCLUDE: &str = "guest/program_loader.h";
pub const PROGRAM_SCHEMA: &str = "playgta6-recomp-program";
pub const PROGRAM_SCHEMA_VERSION: u32 = 1;
/// Build module name of the executable.
pub const EXE_MODULE: &str = super::DEFAULT_MODULE;

/// DLLs the runtime implements outside the import bridge's shim families (it registers their
/// exports by name): kept in sync with the module names in `runtime/` by a test.
pub const HOST_MODULES: &[&str] = &[
    "acphal.dll",
    "amd_fidelityfx_loader_xbox.dll",
    "bink2scarlett.dll",
    "chad.dll",
    "combase.dll",
    "gameinput.dll",
    "libhttpclient.gdk.dll",
    "mfplat.dll",
    "mfreadwrite.dll",
    "ucrtbase.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "xapu.dll",
    "xboxbvhbuilder.dll",
    "xcrdapi.dll",
    "xg_xs.dll",
    "xmem.dll",
];

/// Whether the runtime implements `dll` (any spelling): a shim family of the import bridge, a
/// module it registers ([`HOST_MODULES`]) or an API set mapped to one of those.
pub fn host_known(dll: &str) -> bool {
    let canonical = canonical_dll(dll);
    let known =
        |name: &str| stubs::shim_prefix_for_dll(name).is_some() || HOST_MODULES.contains(&name);
    known(&canonical) || api_set_host(&canonical).is_some_and(known)
}

// ---------------------------------------------------------------------------------------------
// Options, errors, results
// ---------------------------------------------------------------------------------------------

/// Everything [`build_program`] needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramOptions {
    pub exe: PathBuf,
    /// DLL files offered as guest modules.
    pub dlls: Vec<PathBuf>,
    /// Directories whose `*.dll` files are offered as guest modules.
    pub dll_dirs: Vec<PathBuf>,
    pub overrides: Overrides,
    /// Report an allocator split instead of failing.
    pub allow_allocator_split: bool,
    /// Per-module build settings (`out_dir` is the program's build root; `pe_path` and `module`
    /// are set per module).
    pub build: PipelineOptions,
}

impl ProgramOptions {
    pub fn new(exe: impl Into<PathBuf>, out_dir: impl Into<PathBuf>) -> Self {
        let exe = exe.into();
        Self {
            build: PipelineOptions::new(exe.clone(), out_dir),
            exe,
            dlls: Vec::new(),
            dll_dirs: Vec::new(),
            overrides: Overrides::default(),
            allow_allocator_split: false,
        }
    }
}

/// Why a program build failed.
#[derive(Debug)]
pub enum ProgramError {
    Options(String),
    /// Reading or parsing an input.
    Input(String),
    /// Base assignment or rebasing.
    Layout(String),
    /// Linking: allocator split, inconsistent bindings, duplicate or misplaced symbols.
    Link(String),
    /// One module's build.
    Module {
        module: String,
        error: PipelineError,
    },
    Pipeline(PipelineError),
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Options(m) | Self::Input(m) | Self::Layout(m) | Self::Link(m) => f.write_str(m),
            Self::Module { module, error } => write!(f, "module {module}: {error}"),
            Self::Pipeline(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ProgramError {}

impl From<PipelineError> for ProgramError {
    fn from(error: PipelineError) -> Self {
        Self::Pipeline(error)
    }
}

/// TLS directory of a module (VAs at its assigned base).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsRecord {
    /// Implicit TLS index the runtime gives the module: executable first, then the DLLs in
    /// initialisation order, counting only modules with a TLS directory.
    pub index: u32,
    pub start: u64,
    pub end: u64,
    pub index_va: u64,
    pub callbacks: Vec<u64>,
    pub zero_fill: u32,
    pub characteristics: u32,
}

/// How one module's imports were bound.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImportCounts {
    pub total: usize,
    pub guest: usize,
    pub host_bound: usize,
    pub host_unresolved: usize,
    /// What the import bridge alone (a single-module build) binds: shims and unresolved.
    pub before_bound: usize,
    pub before_unresolved: usize,
}

/// One module of the program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleRecord {
    /// File name as supplied.
    pub file: String,
    /// Canonical name.
    pub dll: String,
    /// Build module name.
    pub module: String,
    pub is_exe: bool,
    /// `executable`, `import`, `forwarder` or `forced`.
    pub reason: &'static str,
    pub input_size: u64,
    pub input_hash: u64,
    /// Hash of the bytes built (after rebasing).
    pub built_hash: u64,
    pub preferred_base: u64,
    pub base: u64,
    pub size_of_image: u32,
    /// Entry point VA (0 = none).
    pub entry_point: u64,
    pub tls: Option<TlsRecord>,
    pub exports: ExportSummary,
    pub imports: ImportCounts,
    /// Guest modules it imports from (canonical, sorted).
    pub depends_on: Vec<String>,
    /// `(slot RVA, value, import label)`, sorted by slot.
    pub iat_prefill: Vec<(u32, u64, String)>,
    pub functions: usize,
    pub instructions: usize,
    pub unsupported: usize,
    /// Unsupported instructions by mnemonic, most frequent first.
    pub unsupported_by_mnemonic: Vec<(String, usize)>,
    /// First and last registered function entry (0 when none).
    pub first_function: u64,
    pub last_function: u64,
    pub build_manifest_hash: u64,
}

/// Binding of one import.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Binding {
    /// Export of a guest module (`dll` canonical, `module` its build name).
    Guest {
        dll: String,
        module: String,
        va: u64,
        kind: ExportKind,
    },
    /// The import bridge, resolving `(dll, symbol)` (the import's own, or a forwarder's target).
    Host {
        dll: String,
        symbol: ImportedSymbol,
        resolution: Resolution,
    },
}

/// One import of one module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportRecord {
    /// Canonical name of the importing module.
    pub importer: String,
    /// DLL as spelled in the import table.
    pub dll: String,
    pub symbol: ImportedSymbol,
    pub slot_rva: Option<u32>,
    pub delay_load: bool,
    pub binding: Binding,
    /// The policy rule that decided it (`dll!symbol=action`, or `-` when no guest module was
    /// involved).
    pub rule: String,
    /// Forwarders followed.
    pub via: Vec<String>,
}

impl ImportRecord {
    pub fn label(&self) -> String {
        format!("{}!{}", self.dll, symbol_text(&self.symbol))
    }

    /// `guest`, `host` (bound to a shim) or `unresolved`.
    pub fn status(&self) -> &'static str {
        match &self.binding {
            Binding::Guest { .. } => "guest",
            Binding::Host {
                resolution: Resolution::Bound { .. },
                ..
            } => "host",
            Binding::Host { .. } => "unresolved",
        }
    }
}

/// A finding that does not fail the build.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Diagnostic {
    pub kind: &'static str,
    pub module: String,
    pub detail: String,
}

/// A dependency that is not a guest module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalDll {
    pub dll: String,
    /// `host`, `host_only` or `missing`.
    pub status: &'static str,
    pub importers: BTreeSet<String>,
}

/// Everything a program build produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramReport {
    /// In initialisation order (executable last).
    pub modules: Vec<ModuleRecord>,
    pub cycles: Vec<(String, String)>,
    pub imports: Vec<ImportRecord>,
    pub external: Vec<ExternalDll>,
    /// Supplied DLLs that did not become modules (sorted).
    pub unreferenced: Vec<String>,
    /// Allocator provider (`host` or a DLL) and the number of allocator imports.
    pub allocator: Option<(String, usize)>,
    pub diagnostics: Vec<Diagnostic>,
    /// Program-level files except the manifest.
    pub files: Vec<OutputFile>,
    pub report: String,
    pub manifest: String,
}

impl ProgramReport {
    pub fn module(&self, dll: &str) -> Option<&ModuleRecord> {
        let canonical = canonical_dll(dll);
        self.modules.iter().find(|m| m.dll == canonical)
    }
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------

/// One input module.
struct Loaded {
    file: String,
    path: PathBuf,
    dll: String,
    module: String,
    reason: &'static str,
    preferred_base: u64,
    original: Input,
    /// Bytes built: the original, or rebased.
    built: Input,
    /// Parsed from `built`.
    pe: Pe64,
}

fn input_error(path: &Path, error: impl fmt::Display) -> ProgramError {
    ProgramError::Input(format!("'{}': {error}", path.display()))
}

fn load_pe(path: &Path) -> Result<(Input, Pe64), ProgramError> {
    let input = load_input(path).map_err(|e| ProgramError::Input(e.to_string()))?;
    let pe =
        Pe64::parse(&input.bytes).map_err(|e| input_error(path, format!("not a PE32+: {e}")))?;
    Ok((input, pe))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// The supplied DLLs by canonical name.
fn collect_supplied(options: &ProgramOptions) -> Result<BTreeMap<String, PathBuf>, ProgramError> {
    let mut paths: Vec<PathBuf> = options.dlls.clone();
    for dir in &options.dll_dirs {
        let entries = fs::read_dir(dir).map_err(|e| input_error(dir, e))?;
        for entry in entries {
            let path = entry.map_err(|e| input_error(dir, e))?.path();
            let is_dll = path
                .extension()
                .is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case("dll"));
            if is_dll && path.is_file() {
                paths.push(path);
            }
        }
    }
    let mut supplied: BTreeMap<String, PathBuf> = BTreeMap::new();
    for path in paths {
        let name = canonical_dll(&file_name(&path));
        if name.is_empty() {
            return Err(input_error(&path, "no file name"));
        }
        match supplied.get(&name) {
            Some(existing) if same_file(existing, &path) => {}
            Some(existing) => {
                return Err(ProgramError::Options(format!(
                    "{name} is supplied twice ('{}' and '{}')",
                    existing.display(),
                    path.display()
                )));
            }
            None => {
                supplied.insert(name, path);
            }
        }
    }
    Ok(supplied)
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Reads the shim headers into a catalogue (as [`super::plan_imports`] does).
fn load_catalogue(build: &PipelineOptions) -> Result<Catalogue, ProgramError> {
    let mut loaded = Vec::with_capacity(build.shim_headers.len());
    for header in &build.shim_headers {
        loaded.push(
            stubs::load_shim_header(header, build.include_root.as_deref()).map_err(io_error(
                format!("failed to read shim header '{}'", header.display()),
            ))?,
        );
    }
    Ok(Catalogue::scan(&loaded))
}

/// Imports of a PE in import-bridge order: `(dll, symbol, slot, delay_load)`.
fn imports_of(pe: &Pe64) -> Vec<(String, ImportedSymbol, Option<u32>, bool)> {
    let plan = stubs::plan_bridge(pe, &Catalogue::default());
    plan.imports
        .into_iter()
        .map(|i| (i.dll, i.symbol, i.slot_rva, i.delay_load))
        .collect()
}

/// Builds the program (see the module docs). `progress(module, stage)` is called as each
/// module's stages complete and with module `""` for program-level stages.
pub fn build_program(
    options: &ProgramOptions,
    progress: &mut dyn FnMut(&str, &str),
) -> Result<ProgramReport, ProgramError> {
    check_options(&options.build)?;
    let policy = Policy::with_overrides(&options.overrides);

    // 1. Inputs.
    let (exe_input, exe_pe) = load_pe(&options.exe)?;
    let exe_file = file_name(&options.exe);
    let exe_dll = canonical_dll(&exe_file);
    let supplied = collect_supplied(options)?;
    for forced in &options.overrides.force_guest {
        if !supplied.contains_key(forced) {
            return Err(ProgramError::Options(format!(
                "--guest-dll {forced}: no such DLL among the supplied files"
            )));
        }
    }

    // 2. Module selection.
    let mut loaded: BTreeMap<String, Loaded> = BTreeMap::new();
    let mut external: BTreeMap<String, ExternalDll> = BTreeMap::new();
    let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut queue: Vec<String> = vec![exe_dll.clone()];
    loaded.insert(
        exe_dll.clone(),
        Loaded {
            file: exe_file.clone(),
            path: options.exe.clone(),
            dll: exe_dll.clone(),
            module: EXE_MODULE.to_string(),
            reason: "executable",
            preferred_base: exe_pe.optional.image_base,
            built: exe_input.clone(),
            original: exe_input,
            pe: exe_pe,
        },
    );
    let select = |dep: &str| -> Option<String> {
        let candidate = if supplied.contains_key(dep) {
            dep.to_string()
        } else {
            api_set_host(dep)
                .filter(|h| supplied.contains_key(*h))?
                .to_string()
        };
        (policy.role_of(&candidate).0 == Role::Guest).then_some(candidate)
    };
    let mut forced = options.overrides.force_guest.clone();
    forced.sort();
    forced.dedup();
    let mut pending_forced = forced.clone();
    loop {
        while let Some(dll) = queue.pop() {
            let pe = &loaded[&dll].pe;
            let forwards: BTreeSet<String> = pe
                .exports
                .iter()
                .flat_map(|t| t.symbols.iter())
                .filter_map(|e| e.forwarder.as_deref())
                .filter_map(link::parse_forwarder)
                .map(|(d, _)| canonical_dll(&d))
                .collect();
            let mut deps = Vec::new();
            for dep in link::dependencies(pe) {
                match select(&dep) {
                    Some(module) => {
                        if module != dll && !deps.contains(&module) {
                            deps.push(module.clone());
                        }
                        if !loaded.contains_key(&module) {
                            let reason = if forwards.contains(&dep) {
                                "forwarder"
                            } else {
                                "import"
                            };
                            loaded.insert(module.clone(), load_dll(&supplied[&module], reason)?);
                            queue.push(module);
                        }
                    }
                    None if loaded.contains_key(&dep) => {
                        if dep != dll && !deps.contains(&dep) {
                            deps.push(dep);
                        }
                    }
                    None => {
                        let status = if supplied.contains_key(&dep) {
                            "host_only"
                        } else if host_known(&dep) {
                            "host"
                        } else {
                            "missing"
                        };
                        external
                            .entry(dep.clone())
                            .or_insert_with(|| ExternalDll {
                                dll: dep.clone(),
                                status,
                                importers: BTreeSet::new(),
                            })
                            .importers
                            .insert(dll.clone());
                    }
                }
            }
            graph.insert(dll, deps);
        }
        match pending_forced.pop() {
            Some(dll) => {
                if !loaded.contains_key(&dll) {
                    loaded.insert(dll.clone(), load_dll(&supplied[&dll], "forced")?);
                    queue.push(dll);
                }
            }
            None => break,
        }
    }
    progress("", "select");

    // 3. Initialisation order.
    let (order, cycles) = init_order(&exe_dll, &graph, &forced);

    // Module names.
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    for dll in &order {
        let entry = loaded.get_mut(dll).expect("ordered module is loaded");
        if !entry.module.is_empty() && dll == &exe_dll {
            continue;
        }
        let mut name = module_symbol(dll);
        if name == EXE_MODULE || name.is_empty() {
            name.push_str("_dll");
        }
        entry.module = name;
    }
    for dll in &order {
        let module = &loaded[dll].module;
        if !super::is_valid_module_name(module) {
            return Err(ProgramError::Options(format!(
                "{dll}: module name '{module}' is not valid"
            )));
        }
        if let Some(other) = names.insert(module.clone(), dll.clone()) {
            return Err(ProgramError::Options(format!(
                "{other} and {dll} both map to module name '{module}'"
            )));
        }
    }

    // 4./5. Bases and rebasing.
    let placement: Vec<&String> = std::iter::once(&exe_dll)
        .chain(order.iter().filter(|d| **d != exe_dll))
        .collect();
    let requests: Vec<rebase::Request> = placement
        .iter()
        .map(|dll| {
            let pe = &loaded[*dll].pe;
            rebase::Request {
                name: (*dll).clone(),
                preferred_base: pe.optional.image_base,
                size_of_image: pe.optional.size_of_image,
                relocatable: rebase::relocatable(pe),
            }
        })
        .collect();
    let bases = rebase::assign_bases(&requests).map_err(ProgramError::Layout)?;
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    for (dll, &base) in placement.iter().zip(&bases) {
        let entry = loaded.get_mut(*dll).expect("placed module is loaded");
        if base == entry.pe.optional.image_base {
            continue;
        }
        if entry
            .pe
            .optional
            .data_directory(crate::image::DIR_BASERELOC)
            .is_none()
        {
            diagnostics.push(Diagnostic {
                kind: "rebased_without_relocations",
                module: (*dll).clone(),
                detail: format!("moved to 0x{base:x} but has no base relocation directory"),
            });
        }
        let bytes = rebase::rebase_pe(&entry.original.bytes, &entry.pe, base)
            .map_err(|e| ProgramError::Layout(format!("rebasing {dll} to 0x{base:x}: {e}")))?;
        let pe = Pe64::parse(&bytes)
            .map_err(|e| ProgramError::Layout(format!("rebased {dll} does not parse: {e}")))?;
        entry.built = Input {
            hash: fnv1a_64(&bytes),
            bytes,
        };
        entry.pe = pe;
    }
    progress("", "layout");

    // 6. Binding.
    let mut index = ExportIndex::default();
    for dll in &order {
        index
            .modules
            .insert(dll.clone(), ModuleExports::from_pe(&loaded[dll].pe));
    }
    let catalogue = load_catalogue(&options.build)?;
    let resolver = ShimResolver::new(&catalogue);
    let module_of: HashMap<&str, &str> = loaded
        .values()
        .map(|l| (l.dll.as_str(), l.module.as_str()))
        .collect();
    let mut imports = Vec::new();
    let mut links: BTreeMap<String, ModuleLink> = BTreeMap::new();
    let mut counts: BTreeMap<String, ImportCounts> = BTreeMap::new();
    for dll in &order {
        let entry = &loaded[dll];
        let mut link = ModuleLink {
            bridge_namespace: Some(entry.module.clone()),
            ..ModuleLink::default()
        };
        let mut count = ImportCounts::default();
        for (import_dll, symbol, slot, delay_load) in imports_of(&entry.pe) {
            let before = resolver.resolve(&import_dll, &symbol);
            if matches!(before, Resolution::Bound { .. }) {
                count.before_bound += 1;
            } else {
                count.before_unresolved += 1;
            }
            let record = bind_import(
                &policy,
                &index,
                &resolver,
                &module_of,
                dll,
                &import_dll,
                &symbol,
                slot,
                delay_load,
                &mut diagnostics,
            );
            count.total += 1;
            match &record.binding {
                Binding::Guest { module, va, .. } => {
                    count.guest += 1;
                    let slot = slot.expect("guest bindings have a slot");
                    link.guest_imports.insert(
                        slot,
                        GuestTarget {
                            module: module.clone(),
                            va: *va,
                        },
                    );
                }
                Binding::Host {
                    dll: target,
                    symbol: target_symbol,
                    resolution,
                } => {
                    if matches!(resolution, Resolution::Bound { .. }) {
                        count.host_bound += 1;
                    } else {
                        count.host_unresolved += 1;
                    }
                    if (target != &import_dll || target_symbol != &symbol)
                        && let Some(slot) = slot
                    {
                        link.host_redirects
                            .insert(slot, (target.clone(), target_symbol.clone()));
                    }
                }
            }
            imports.push(record);
        }
        links.insert(dll.clone(), link);
        counts.insert(dll.clone(), count);
    }
    let allocator = check_allocator(&imports, options.allow_allocator_split, &mut diagnostics)?;
    check_consistency(&imports)?;
    progress("", "link");

    // TLS indices: executable first, then initialisation order.
    let mut tls: BTreeMap<String, TlsRecord> = BTreeMap::new();
    let mut next_tls = 0u32;
    for dll in &placement {
        if let Some(dir) = &loaded[*dll].pe.tls {
            tls.insert(
                (*dll).clone(),
                TlsRecord {
                    index: next_tls,
                    start: dir.start_address_of_raw_data,
                    end: dir.end_address_of_raw_data,
                    index_va: dir.address_of_index,
                    callbacks: dir.callbacks.iter().map(|c| c.va).collect(),
                    zero_fill: dir.size_of_zero_fill,
                    characteristics: dir.characteristics,
                },
            );
            next_tls += 1;
        }
    }

    // 7. Module builds.
    let mut records = Vec::with_capacity(order.len());
    let mut owner: HashMap<u64, String> = HashMap::new();
    for dll in &order {
        let entry = &loaded[dll];
        let mut build = options.build.clone();
        build.pe_path = entry.path.clone();
        build.module = entry.module.clone();
        let module = entry.module.clone();
        let report = run_module(&build, &entry.built, &links[dll], &mut |stage| {
            progress(&module, stage)
        })
        .map_err(|error| ProgramError::Module {
            module: module.clone(),
            error,
        })?;
        let base = entry.pe.optional.image_base;
        let end = base + rebase::reserved_size(entry.pe.optional.size_of_image);
        check_symbols(dll, base, end, &report.function_entries, &mut owner)?;
        records.push(module_record(
            entry,
            &report,
            &index,
            &graph,
            counts[dll],
            tls.get(dll).cloned(),
            &links[dll],
            &imports,
        ));
    }

    // 8. Program files.
    let out = &options.build.out_dir;
    fs::create_dir_all(out).map_err(io_error(format!(
        "failed to create output directory '{}'",
        out.display()
    )))?;
    let mut unreferenced: Vec<String> = supplied
        .keys()
        .filter(|d| !loaded.contains_key(*d) && !external.contains_key(*d))
        .cloned()
        .collect();
    unreferenced.sort();
    diagnostics.sort();
    diagnostics.dedup();
    let mut report = ProgramReport {
        modules: records,
        cycles,
        imports,
        external: external.into_values().collect(),
        unreferenced,
        allocator,
        diagnostics,
        files: Vec::new(),
        report: String::new(),
        manifest: String::new(),
    };
    let descriptor = render_descriptor(&report);
    let mut files = vec![write_file(
        out,
        PROGRAM_DESCRIPTOR_FILE,
        descriptor.as_bytes(),
    )?];
    report.report = render_report(&report);
    files.push(write_file(
        out,
        PROGRAM_REPORT_FILE,
        report.report.as_bytes(),
    )?);
    files.sort();
    report.files = files;
    report.manifest = render_manifest(&report, options);
    write_file(out, PROGRAM_MANIFEST_FILE, report.manifest.as_bytes())?;
    progress("", "manifest");
    Ok(report)
}

fn load_dll(path: &Path, reason: &'static str) -> Result<Loaded, ProgramError> {
    let (input, pe) = load_pe(path)?;
    let file = file_name(path);
    Ok(Loaded {
        dll: canonical_dll(&file),
        file,
        path: path.to_path_buf(),
        module: String::new(),
        reason,
        preferred_base: pe.optional.image_base,
        built: input.clone(),
        original: input,
        pe,
    })
}

/// Depth-first post-order over `graph` from the executable's dependencies, then the forced
/// modules; the executable comes last. Returns the order and the edges that closed a cycle.
fn init_order(
    exe: &str,
    graph: &BTreeMap<String, Vec<String>>,
    forced: &[String],
) -> (Vec<String>, Vec<(String, String)>) {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Visiting,
        Done,
    }
    fn visit(
        dll: &str,
        graph: &BTreeMap<String, Vec<String>>,
        state: &mut HashMap<String, State>,
        order: &mut Vec<String>,
        cycles: &mut Vec<(String, String)>,
    ) {
        state.insert(dll.to_string(), State::Visiting);
        for dep in graph.get(dll).map(Vec::as_slice).unwrap_or(&[]) {
            match state.get(dep) {
                None => visit(dep, graph, state, order, cycles),
                Some(State::Visiting) => cycles.push((dll.to_string(), dep.clone())),
                Some(State::Done) => {}
            }
        }
        state.insert(dll.to_string(), State::Done);
        order.push(dll.to_string());
    }
    let mut state = HashMap::new();
    let mut order = Vec::new();
    let mut cycles = Vec::new();
    state.insert(exe.to_string(), State::Visiting);
    let roots = graph
        .get(exe)
        .map(Vec::as_slice)
        .unwrap_or(&[])
        .iter()
        .chain(forced);
    for dep in roots {
        if !state.contains_key(dep) {
            visit(dep, graph, &mut state, &mut order, &mut cycles);
        }
    }
    order.push(exe.to_string());
    (order, cycles)
}

/// Binds one import (see "Import binding policy").
#[allow(clippy::too_many_arguments)]
fn bind_import(
    policy: &Policy,
    index: &ExportIndex,
    resolver: &ShimResolver<'_>,
    module_of: &HashMap<&str, &str>,
    importer: &str,
    dll: &str,
    symbol: &ImportedSymbol,
    slot: Option<u32>,
    delay_load: bool,
    diagnostics: &mut Vec<Diagnostic>,
) -> ImportRecord {
    let canonical = canonical_dll(dll);
    let target = api_set_host(&canonical).map_or(canonical.clone(), str::to_string);
    let text = symbol_text(symbol);
    let label = format!("{dll}!{text}");
    let mut diag = |kind: &'static str, detail: String| {
        diagnostics.push(Diagnostic {
            kind,
            module: importer.to_string(),
            detail,
        });
    };
    let host = |dll: &str, symbol: &ImportedSymbol| Binding::Host {
        dll: dll.to_string(),
        symbol: symbol.clone(),
        resolution: resolver.resolve(dll, symbol),
    };
    let mut record = ImportRecord {
        importer: importer.to_string(),
        dll: dll.to_string(),
        symbol: symbol.clone(),
        slot_rva: slot,
        delay_load,
        binding: host(dll, symbol),
        rule: "-".to_string(),
        via: Vec::new(),
    };
    let rule = policy.rule_for(&target, &text);
    if index.guest_module(dll).is_none() {
        if let Some(required) = policy.required_default(&target, &text) {
            diag(
                "required_guest_unavailable",
                format!("{label}: {required} but {target} is not a guest module"),
            );
        }
        return record;
    }
    record.rule = format!(
        "{rule}{}",
        if rule.source == policy::Source::Override {
            " (override)"
        } else {
            ""
        }
    );
    if rule.action != Action::Guest
        && let Some(required) = policy.required_default(&target, &text)
        && rule.source == policy::Source::Override
    {
        diag(
            "required_rule_overridden",
            format!("{label}: {rule} overrides {required}"),
        );
    }
    let mut via = Vec::new();
    let provider = index.resolve(dll, symbol, &mut via);
    record.via = via;
    let (pdll, va, kind) = match provider {
        Err(error) => {
            diag(error.code(), format!("{label}: {}", error.describe()));
            return record;
        }
        Ok(Provider::Host {
            dll: final_dll,
            symbol: final_symbol,
        }) => {
            diag(
                "host_redirect",
                format!(
                    "{label} forwards to {final_dll}!{} (host)",
                    symbol_text(&final_symbol)
                ),
            );
            record.binding = host(&final_dll, &final_symbol);
            return record;
        }
        Ok(Provider::Guest { dll, va, kind }) => (dll, va, kind),
    };
    let guest = match rule.action {
        Action::Guest => true,
        Action::Host if kind == ExportKind::Data => {
            diag(
                "data_export_forced_guest",
                format!("{label} is data; bound to the guest export despite {rule}"),
            );
            true
        }
        Action::Host => false,
        Action::PreferHost => {
            kind == ExportKind::Data
                || !matches!(resolver.resolve(dll, symbol), Resolution::Bound { .. })
        }
    };
    if guest && slot.is_none() {
        diag(
            "no_iat_slot",
            format!("{label} has no IAT slot to pre-fill; left on the bridge"),
        );
        return record;
    }
    if guest {
        record.binding = Binding::Guest {
            module: module_of
                .get(pdll.as_str())
                .map_or_else(String::new, |m| (*m).to_string()),
            dll: pdll,
            va,
            kind,
        };
    }
    record
}

/// Every lifted function of `dll` must lie in its image range `[base, end)`, and no `sub_<VA>`
/// symbol may already belong to another module (`owner` collects them across modules).
fn check_symbols(
    dll: &str,
    base: u64,
    end: u64,
    entries: &[u64],
    owner: &mut HashMap<u64, String>,
) -> Result<(), ProgramError> {
    for &va in entries {
        if !(base..end).contains(&va) {
            return Err(ProgramError::Link(format!(
                "{dll}: lifted function sub_{va:X} lies outside its image [0x{base:x}, 0x{end:x})"
            )));
        }
        if let Some(other) = owner.insert(va, dll.to_string()) {
            return Err(ProgramError::Link(format!(
                "duplicate symbol sub_{va:X} in {other} and {dll}"
            )));
        }
    }
    Ok(())
}

/// Provider identity of a binding for the consistency checks.
fn provider_of(binding: &Binding) -> String {
    match binding {
        Binding::Guest { dll, va, .. } => format!("{dll}@0x{va:x}"),
        Binding::Host { dll, symbol, .. } => {
            // API-set contracts and their host module are one provider.
            let canonical = canonical_dll(dll);
            let target = api_set_host(&canonical).map_or(canonical, str::to_string);
            format!("host:{target}!{}", symbol_text(symbol))
        }
    }
}

/// Every allocator-family import must resolve to one side. Returns the provider and count.
fn check_allocator(
    imports: &[ImportRecord],
    allow_split: bool,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<Option<(String, usize)>, ProgramError> {
    let mut sides: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut total = 0;
    for import in imports {
        let canonical = canonical_dll(&import.dll);
        let target = api_set_host(&canonical).map_or(canonical, str::to_string);
        let ImportedSymbol::ByName { name, .. } = &import.symbol else {
            continue;
        };
        if !policy::is_allocator_symbol(&target, name) {
            continue;
        }
        total += 1;
        let side = match &import.binding {
            Binding::Guest { dll, .. } => dll.clone(),
            Binding::Host { .. } => "host".to_string(),
        };
        sides
            .entry(side)
            .or_default()
            .push(format!("{}: {}", import.importer, import.label()));
    }
    if sides.len() > 1 {
        let detail = sides
            .iter()
            .map(|(side, users)| format!("{side} <- {}", users.join(", ")))
            .collect::<Vec<_>>()
            .join("; ");
        if !allow_split {
            return Err(ProgramError::Link(format!(
                "allocator split: heap functions resolve to more than one allocator ({detail}); \
                 bind them consistently or pass --allow-allocator-split"
            )));
        }
        diagnostics.push(Diagnostic {
            kind: "allocator_split",
            module: String::new(),
            detail,
        });
    }
    Ok(sides.into_keys().next().map(|side| (side, total)))
}

/// One `(DLL, symbol)` must bind to one provider for every importer.
fn check_consistency(imports: &[ImportRecord]) -> Result<(), ProgramError> {
    let mut seen: HashMap<(String, String), String> = HashMap::new();
    for import in imports {
        let canonical = canonical_dll(&import.dll);
        let target = api_set_host(&canonical).map_or(canonical, str::to_string);
        let key = (target, symbol_text(&import.symbol));
        let provider = provider_of(&import.binding);
        match seen.get(&key) {
            Some(existing) if *existing != provider => {
                return Err(ProgramError::Link(format!(
                    "{}!{} is bound to {existing} and to {provider}",
                    key.0, key.1
                )));
            }
            Some(_) => {}
            None => {
                seen.insert(key, provider);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn module_record(
    entry: &Loaded,
    report: &PipelineReport,
    index: &ExportIndex,
    graph: &BTreeMap<String, Vec<String>>,
    imports: ImportCounts,
    tls: Option<TlsRecord>,
    link: &ModuleLink,
    records: &[ImportRecord],
) -> ModuleRecord {
    let pe = &entry.pe;
    let base = pe.optional.image_base;
    let mut depends_on: Vec<String> = graph.get(&entry.dll).cloned().unwrap_or_default();
    depends_on.sort();
    let labels: HashMap<u32, String> = records
        .iter()
        .filter(|r| r.importer == entry.dll)
        .filter_map(|r| Some((r.slot_rva?, r.label())))
        .collect();
    let mut unsupported_by_mnemonic: Vec<(String, usize)> = report
        .lift
        .unsupported
        .iter()
        .map(|(k, n)| (k.clone(), *n))
        .collect();
    unsupported_by_mnemonic.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    ModuleRecord {
        file: entry.file.clone(),
        dll: entry.dll.clone(),
        module: entry.module.clone(),
        is_exe: entry.reason == "executable",
        reason: entry.reason,
        input_size: entry.original.bytes.len() as u64,
        input_hash: entry.original.hash,
        built_hash: entry.built.hash,
        preferred_base: entry.preferred_base,
        base,
        size_of_image: pe.optional.size_of_image,
        entry_point: if pe.optional.entry_point_rva == 0 {
            0
        } else {
            base + u64::from(pe.optional.entry_point_rva)
        },
        tls,
        exports: index
            .modules
            .get(&entry.dll)
            .map_or_else(ExportSummary::default, |m| m.summary),
        imports,
        depends_on,
        iat_prefill: link
            .guest_imports
            .iter()
            .map(|(&slot, t)| (slot, t.va, labels.get(&slot).cloned().unwrap_or_default()))
            .collect(),
        functions: report.lift.functions,
        instructions: report.lift.instructions,
        unsupported: report.lift.unsupported_total(),
        unsupported_by_mnemonic,
        first_function: report.function_entries.first().copied().unwrap_or(0),
        last_function: report.function_entries.last().copied().unwrap_or(0),
        build_manifest_hash: fnv1a_64(report.manifest.as_bytes()),
    }
}

// ---------------------------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------------------------

fn percent(part: usize, whole: usize) -> String {
    super::percent(part, whole)
}

/// Renders `program-report.txt` (names, addresses and counts only).
pub fn render_report(r: &ProgramReport) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "playgta6-recomp program build: {} modules",
        r.modules.len()
    );
    let _ = writeln!(out);
    let _ = writeln!(out, "modules (initialisation order):");
    let width = r
        .modules
        .iter()
        .map(|m| m.file.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let _ = writeln!(
        out,
        "  {:<width$}  {:>12}  {:>18}  {:>10}  {:>8}  {:>9}  {:>12}  {:>8}  {:>17}  {:>16}",
        "file",
        "bytes",
        "base",
        "size",
        "functions",
        "instr",
        "lifted",
        "exports",
        "imports g/h/u",
        "fn/data/fwd"
    );
    for m in &r.modules {
        let _ = writeln!(
            out,
            "  {:<width$}  {:>12}  {:>18}  {:>10}  {:>8}  {:>9}  {:>12}  {:>8}  {:>17}  {:>16}",
            m.file,
            m.input_size,
            format!(
                "0x{:x}{}",
                m.base,
                if m.base != m.preferred_base { "*" } else { "" }
            ),
            format!("0x{:x}", m.size_of_image),
            m.functions,
            m.instructions,
            percent(m.instructions - m.unsupported, m.instructions),
            m.exports.total,
            format!(
                "{}/{}/{}",
                m.imports.guest, m.imports.host_bound, m.imports.host_unresolved
            ),
            format!(
                "{}/{}/{}",
                m.exports.functions, m.exports.data, m.exports.forwarders
            ),
        );
    }
    let _ = writeln!(
        out,
        "  (* rebased from its preferred base; imports g/h/u = guest module / host shim / unresolved)"
    );
    for m in &r.modules {
        if m.base != m.preferred_base {
            let _ = writeln!(
                out,
                "  {} rebased 0x{:x} -> 0x{:x}",
                m.file, m.preferred_base, m.base
            );
        }
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "imports per module (before = import bridge only):");
    for m in &r.modules {
        let i = &m.imports;
        let _ = writeln!(
            out,
            "  {}: {} imports; before {} bound / {} unresolved; now {} guest, {} host, {} unresolved; {} IAT slots pre-filled",
            m.file,
            i.total,
            i.before_bound,
            i.before_unresolved,
            i.guest,
            i.host_bound,
            i.host_unresolved,
            m.iat_prefill.len()
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(out, "guest bindings by provider:");
    let mut by_provider: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for import in &r.imports {
        if let Binding::Guest { dll, .. } = &import.binding {
            *by_provider
                .entry((import.importer.as_str(), dll.as_str()))
                .or_insert(0) += 1;
        }
    }
    for ((importer, provider), n) in &by_provider {
        let _ = writeln!(out, "  {importer} -> {provider}: {n}");
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "initialisation order: {}",
        r.modules
            .iter()
            .map(|m| m.file.as_str())
            .collect::<Vec<_>>()
            .join(" -> ")
    );
    for (from, to) in &r.cycles {
        let _ = writeln!(out, "  cycle broken: {from} -> {to}");
    }
    for m in &r.modules {
        if !m.is_exe && m.entry_point != 0 {
            let _ = writeln!(out, "  DllMain {}: 0x{:x}", m.file, m.entry_point);
        }
        if let Some(t) = &m.tls {
            let _ = writeln!(
                out,
                "  TLS {}: index {}, data 0x{:x}..0x{:x}, index at 0x{:x}, {} callbacks",
                m.file,
                t.index,
                t.start,
                t.end,
                t.index_va,
                t.callbacks.len()
            );
        }
    }
    let _ = writeln!(out);
    match &r.allocator {
        Some((side, n)) => {
            let _ = writeln!(
                out,
                "allocator: {n} allocator imports, all resolve to {side}"
            );
        }
        None => {
            let _ = writeln!(out, "allocator: no allocator imports");
        }
    }
    let _ = writeln!(out, "external DLLs:");
    for e in &r.external {
        let _ = writeln!(
            out,
            "  {:<48} {:<10} ({} importers)",
            e.dll,
            e.status,
            e.importers.len()
        );
    }
    if !r.unreferenced.is_empty() {
        let _ = writeln!(
            out,
            "supplied but not referenced: {}",
            r.unreferenced.join(", ")
        );
    }
    for m in &r.modules {
        if m.unsupported_by_mnemonic.is_empty() {
            continue;
        }
        let top: Vec<String> = m
            .unsupported_by_mnemonic
            .iter()
            .take(8)
            .map(|(k, n)| format!("{k} {n}"))
            .collect();
        let _ = writeln!(out, "unsupported in {}: {}", m.file, top.join(", "));
    }
    let _ = writeln!(out);
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for d in &r.diagnostics {
        *kinds.entry(d.kind).or_insert(0) += 1;
    }
    let _ = writeln!(out, "diagnostics: {}", r.diagnostics.len());
    for (kind, n) in kinds {
        let _ = writeln!(out, "  {kind:<32} {n:>6}");
    }
    out
}

fn symbol_json(symbol: &ImportedSymbol) -> (Json, Json) {
    match symbol {
        ImportedSymbol::ByName { name, .. } => (Json::Str(name.clone()), Json::Null),
        ImportedSymbol::ByOrdinal { ordinal } => (Json::Null, Json::Num(u64::from(*ordinal))),
    }
}

fn opt_hex(v: u64) -> Json {
    if v == 0 { Json::Null } else { hex(v) }
}

fn strs<'a>(items: impl IntoIterator<Item = &'a String>) -> Json {
    Json::Arr(items.into_iter().map(|s| Json::Str(s.clone())).collect())
}

/// Renders `program-manifest.json` (schema in the module docs).
pub fn render_manifest(r: &ProgramReport, options: &ProgramOptions) -> String {
    let b = &options.build;
    let overrides: Vec<String> = options
        .overrides
        .rules
        .iter()
        .map(ToString::to_string)
        .chain(
            options
                .overrides
                .roles
                .iter()
                .map(|r| format!("{}={}", r.dll, r.role.as_str())),
        )
        .collect();
    let mut force_guest = options.overrides.force_guest.clone();
    force_guest.sort();
    force_guest.dedup();
    let modules = r
        .modules
        .iter()
        .map(|m| {
            let (table, _) = registration_symbols(&m.module);
            obj([
                ("file", Json::Str(m.file.clone())),
                ("dll", Json::Str(m.dll.clone())),
                ("module", Json::Str(m.module.clone())),
                (
                    "role",
                    Json::Str(if m.is_exe { "exe" } else { "dll" }.into()),
                ),
                ("reason", Json::Str(m.reason.into())),
                (
                    "input",
                    obj([
                        ("size", Json::Num(m.input_size)),
                        ("fnv1a64", Json::Str(format!("0x{:016x}", m.input_hash))),
                    ]),
                ),
                (
                    "built_fnv1a64",
                    Json::Str(format!("0x{:016x}", m.built_hash)),
                ),
                ("preferred_base", hex(m.preferred_base)),
                ("base", hex(m.base)),
                ("rebased", Json::Bool(m.base != m.preferred_base)),
                ("size_of_image", hex(u64::from(m.size_of_image))),
                (
                    "range_end",
                    hex(m.base + rebase::reserved_size(m.size_of_image)),
                ),
                ("entry_point", opt_hex(m.entry_point)),
                (
                    "tls",
                    m.tls.as_ref().map_or(Json::Null, |t| {
                        obj([
                            ("index", Json::Num(u64::from(t.index))),
                            ("start", hex(t.start)),
                            ("end", hex(t.end)),
                            ("index_va", hex(t.index_va)),
                            (
                                "callbacks",
                                Json::Arr(t.callbacks.iter().map(|&v| hex(v)).collect()),
                            ),
                            ("zero_fill", Json::Num(u64::from(t.zero_fill))),
                            ("characteristics", hex(u64::from(t.characteristics))),
                        ])
                    }),
                ),
                (
                    "exports",
                    obj([
                        ("total", num(m.exports.total)),
                        ("functions", num(m.exports.functions)),
                        ("data", num(m.exports.data)),
                        ("forwarders", num(m.exports.forwarders)),
                        ("ordinal_only", num(m.exports.ordinal_only)),
                    ]),
                ),
                (
                    "imports",
                    obj([
                        ("total", num(m.imports.total)),
                        ("guest", num(m.imports.guest)),
                        ("host_bound", num(m.imports.host_bound)),
                        ("host_unresolved", num(m.imports.host_unresolved)),
                        (
                            "before",
                            obj([
                                ("bound", num(m.imports.before_bound)),
                                ("unresolved", num(m.imports.before_unresolved)),
                            ]),
                        ),
                    ]),
                ),
                ("depends_on", strs(&m.depends_on)),
                (
                    "iat_prefill",
                    Json::Arr(
                        m.iat_prefill
                            .iter()
                            .map(|(slot, value, label)| {
                                obj([
                                    ("slot_rva", hex(u64::from(*slot))),
                                    ("value", hex(*value)),
                                    ("import", Json::Str(label.clone())),
                                ])
                            })
                            .collect(),
                    ),
                ),
                (
                    "registration",
                    obj([
                        (
                            "register_function",
                            Json::Str(register_function_symbol(&m.module)),
                        ),
                        ("table_symbol", Json::Str(table)),
                        ("count", num(m.functions)),
                        ("first", opt_hex(m.first_function)),
                        ("last", opt_hex(m.last_function)),
                    ]),
                ),
                (
                    "bridge",
                    obj([
                        (
                            "thunks_symbol",
                            Json::Str(format!("pg6_import_thunks_{}", m.module)),
                        ),
                        (
                            "thunk_count_symbol",
                            Json::Str(format!("pg6_import_thunk_count_{}", m.module)),
                        ),
                    ]),
                ),
                (
                    "lift",
                    obj([
                        ("functions", num(m.functions)),
                        ("instructions", num(m.instructions)),
                        ("unsupported", num(m.unsupported)),
                    ]),
                ),
                (
                    "build_manifest",
                    obj([
                        (
                            "path",
                            Json::Str(format!("{}/{}", m.module, super::MANIFEST_FILE)),
                        ),
                        (
                            "fnv1a64",
                            Json::Str(format!("0x{:016x}", m.build_manifest_hash)),
                        ),
                    ]),
                ),
            ])
        })
        .collect();
    let imports = r
        .imports
        .iter()
        .map(|i| {
            let (name, ordinal) = symbol_json(&i.symbol);
            let (provider, target, kind, shim, reason) = match &i.binding {
                Binding::Guest { dll, va, kind, .. } => (
                    Json::Str(dll.clone()),
                    hex(*va),
                    Json::Str(kind.as_str().into()),
                    Json::Null,
                    Json::Null,
                ),
                Binding::Host { resolution, .. } => match resolution {
                    Resolution::Bound { shim } => (
                        Json::Null,
                        Json::Null,
                        Json::Null,
                        Json::Str(shim.clone()),
                        Json::Null,
                    ),
                    Resolution::Unresolved(why) => (
                        Json::Null,
                        Json::Null,
                        Json::Null,
                        Json::Null,
                        Json::Str(why.code().into()),
                    ),
                },
            };
            obj([
                ("importer", Json::Str(i.importer.clone())),
                ("dll", Json::Str(i.dll.clone())),
                ("name", name),
                ("ordinal", ordinal),
                (
                    "slot_rva",
                    i.slot_rva.map_or(Json::Null, |s| hex(u64::from(s))),
                ),
                ("delay_load", Json::Bool(i.delay_load)),
                ("binding", Json::Str(i.status().into())),
                ("provider", provider),
                ("target", target),
                ("kind", kind),
                ("shim", shim),
                ("reason", reason),
                ("via", strs(&i.via)),
                ("rule", Json::Str(i.rule.clone())),
            ])
        })
        .collect();
    let manifest = obj([
        ("schema", Json::Str(PROGRAM_SCHEMA.into())),
        (
            "schema_version",
            Json::Num(u64::from(PROGRAM_SCHEMA_VERSION)),
        ),
        (
            "tool",
            obj([
                ("name", Json::Str(env!("CARGO_PKG_NAME").into())),
                ("version", Json::Str(env!("CARGO_PKG_VERSION").into())),
            ]),
        ),
        (
            "options",
            obj([
                (
                    "allow_allocator_split",
                    Json::Bool(options.allow_allocator_split),
                ),
                ("overrides", strs(&overrides)),
                ("force_guest", strs(&force_guest)),
                (
                    "build",
                    obj([
                        ("emit_obj", Json::Bool(b.emit_obj)),
                        ("shard_size", num(b.shard_size)),
                        ("shard_instructions", num(b.shard_instructions)),
                        ("map_headers", Json::Bool(b.image.map_headers)),
                        (
                            "executable_bytes",
                            Json::Str(b.image.executable_bytes.as_str().into()),
                        ),
                    ]),
                ),
            ]),
        ),
        ("modules", Json::Arr(modules)),
        ("init_order", strs(r.modules.iter().map(|m| &m.dll))),
        (
            "cycles",
            Json::Arr(
                r.cycles
                    .iter()
                    .map(|(from, to)| {
                        obj([
                            ("from", Json::Str(from.clone())),
                            ("to", Json::Str(to.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("imports", Json::Arr(imports)),
        (
            "external_dlls",
            Json::Arr(
                r.external
                    .iter()
                    .map(|e| {
                        obj([
                            ("dll", Json::Str(e.dll.clone())),
                            ("status", Json::Str(e.status.into())),
                            ("importers", strs(&e.importers)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("unreferenced_dlls", strs(&r.unreferenced)),
        (
            "allocator",
            obj([
                (
                    "provider",
                    r.allocator
                        .as_ref()
                        .map_or(Json::Null, |(side, _)| Json::Str(side.clone())),
                ),
                ("imports", num(r.allocator.as_ref().map_or(0, |(_, n)| *n))),
            ]),
        ),
        (
            "diagnostics",
            Json::Arr(
                r.diagnostics
                    .iter()
                    .map(|d| {
                        obj([
                            ("kind", Json::Str(d.kind.into())),
                            ("module", Json::Str(d.module.clone())),
                            ("detail", Json::Str(d.detail.clone())),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "files",
            Json::Arr(r.files.iter().map(super::file_json).collect()),
        ),
    ]);
    let mut out = String::new();
    manifest.render(&mut out, 0);
    out.push('\n');
    out
}

/// C string literal (printable ASCII kept, everything else octal-escaped).
fn c_str(s: &str) -> String {
    let mut out = String::from("\"");
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

/// Renders `pg6_program.cpp`, the descriptor `runtime/guest/program_loader.h` loads.
pub fn render_descriptor(r: &ProgramReport) -> String {
    let mut out = String::new();
    out.push_str(
        "// Generated by playgta6-recomp build-program. Do not edit.\n\
         //\n\
         // Guest program descriptor for the runtime program loader: every module in\n\
         // initialisation order (executable last) with its assigned base, its lifted-function\n\
         // table and its import bridge table. Compile with the build root on the include path.\n",
    );
    let _ = writeln!(out, "#include <cstddef>\n#include <cstdint>\n");
    let _ = writeln!(out, "#include {}", c_str(PROGRAM_LOADER_INCLUDE));
    for m in &r.modules {
        let _ = writeln!(
            out,
            "#include {}",
            c_str(&format!(
                "{}/{}/{}",
                m.module,
                super::STUBS_DIR,
                stubs::HEADER_FILE
            ))
        );
    }
    out.push_str(
        "\nstatic_assert(sizeof(Pg6ImportThunk) == sizeof(Pg6BootImport) &&\n\
         \x20             alignof(Pg6ImportThunk) == alignof(Pg6BootImport),\n\
         \x20             \"import bridge tables are read as Pg6BootImport\");\n\nextern \"C\" {\n",
    );
    for m in &r.modules {
        let (table, count) = registration_symbols(&m.module);
        let _ = writeln!(
            out,
            "extern const Pg6LiftedFunctionEntry {table}[];\nextern const uint64_t {count};"
        );
    }
    out.push_str("}  // extern \"C\"\n\nnamespace {\n\nconst Pg6ProgramModule kModules[] = {\n");
    for m in &r.modules {
        let mut flags = vec![if m.is_exe {
            "kPg6ProgramModuleMain"
        } else {
            "kPg6ProgramModuleDll"
        }];
        if m.tls.is_some() {
            flags.push("kPg6ProgramModuleTls");
        }
        let (table, count) = registration_symbols(&m.module);
        let _ = writeln!(
            out,
            "    {{{}, {}, 0x{:x}ull, 0x{:x}ull, 0x{:x}ull, {}, {}u, {table}, &{count},\n     \
             reinterpret_cast<const Pg6BootImport* const*>(&pg6_import_thunks_{}), \
             &pg6_import_thunk_count_{}}},",
            c_str(&m.file),
            c_str(&m.module),
            m.base,
            m.size_of_image,
            m.entry_point,
            flags.join(" | "),
            m.tls.as_ref().map_or(0, |t| t.index),
            m.module,
            m.module
        );
    }
    out.push_str(
        "};\n\n}  // namespace\n\n\
         extern \"C\" const Pg6Program pg6_program = {\n    \
         kPg6ProgramVersion,\n    \
         static_cast<uint32_t>(sizeof(kModules) / sizeof(kModules[0])),\n    \
         kModules,\n};\n",
    );
    out
}
