//! End-to-end recompile driver: PE64 -> code image -> `.pdata` seeding -> discovery (with the
//! [`refine`] fixpoint) -> sharded, parallel lifting -> import thunks -> packed guest image, plus
//! one deterministic build manifest and a coverage report. `playgta6-recomp build` wraps [`run`].
//!
//! # Stages
//! [`run`] chains explicit stages that pass typed results along instead of re-reading the input:
//! 1. [`load_input`]: read the PE once ([`Input`]: bytes, size, FNV-1a 64 hash).
//! 2. `Pe64::parse` and [`image::build_guest_image`] (needed early: its base relocations feed the
//!    code-pointer refinement).
//! 3. [`seed`]: [`CodeImage`] + `.pdata` seeds + no-return import slots ([`Seeded`]); the
//!    `discover` and `lift` subcommands use the same function.
//! 4. [`refine::discover_refined`]: discovery to a fixpoint with code-pointer entries and
//!    no-return thunks.
//! 5. [`plan_imports`] + [`refine::check_import_slots`]: import bridge plan and slot consistency.
//! 6. [`plan_shards`] + [`lift_shards`]: lift functions into shard modules on `jobs` threads,
//!    streaming each shard to disk (optionally compiling it to a wasm64 object), then
//!    [`write_registration_table`].
//! 7. Import bridge files, `PGIM` container, `report.txt` ([`PipelineReport::report`]) and
//!    `build-manifest.json` ([`PipelineReport::manifest`]).
//!
//! Discovery re-runs from scratch in each fixpoint round (typically 2-3), and the whole CFG
//! stays in memory until lifting is done; lifting holds at most `jobs` shards of IR at a time.
//!
//! # Output layout
//! One [`run`] builds one guest module into `--out/<module>/` ([`PipelineOptions::module_dir`],
//! module `main` by default):
//! ```text
//! build-manifest.json          this build: inputs, options, statistics, file list with hashes
//! report.txt                   coverage report (also printed by the CLI)
//! lift/shard-00000.ll ...      lifted functions, one LLVM module per shard (+ .o with --emit-obj)
//! lift/functions.ll            function registration table (+ functions.o with --emit-obj)
//! stubs/pg6_import_thunks.h    import bridge (R5): thunk table header,
//! stubs/pg6_import_thunks.cpp  thunks + slot table,
//! stubs/pg6_guest_seam.h       default guest-state seam,
//! stubs/pg6_import_manifest.json  binding manifest
//! image/guest.pgim             guest memory image container (R6)
//! image/guest.pgim.json        its manifest
//! ```
//! Every file is rewritten on each build; stale `lift/shard-*.{ll,o}` from an earlier build with
//! more shards are removed. Outputs contain code and data derived from the input, so `--out`
//! must pass [`check_output_dir`] (outside every git work tree, or below a gitignored directory
//! such as `work/`). Nothing here prints; the CLI prints [`PipelineReport::report`], which holds
//! names, addresses and counts only, plus stage progress from [`run_with_progress`].
//!
//! # Several guest modules
//! Nothing assumes a single image: per-module state lives in [`PipelineOptions`] and the module
//! directory, and the registration table symbols carry the module name, so the executable and
//! the DLLs it ships with can be built side by side under one `--out` and linked into one
//! program. Lifted functions are named by guest VA (`sub_<VA>`), which stays unique as long as
//! every module is placed at a distinct base. [`program`] (`playgta6-recomp build-program`) is
//! that multi-module driver: it assigns non-overlapping bases, rebases before lifting, binds one
//! module's imports to another's exports and builds every module through [`run_module`] with a
//! [`ModuleLink`] (pre-filled IAT slots, namespaced import bridge).
//!
//! # Determinism
//! The same input bytes and options give byte-identical files and manifest, independent of
//! `jobs`: shards are fixed by [`plan_shards`], each shard is emitted by the deterministic
//! [`lift::emit_module`], results are merged by shard index, and the manifest has no
//! timestamps, absolute paths or host names (only the LLVM tool's file name and version line).
//!
//! # Sharding and the registration table
//! Functions sorted by entry VA are cut into consecutive shards of at most `shard_size`
//! functions and at most `shard_instructions` instructions ([`plan_shards`]); shard `i` is
//! written as `lift/shard-<i:05>.ll`. [`lift::emit_module`] declares every callee that is not
//! defined in the shard, so each shard compiles on its own and all shards plus `functions.ll`
//! link into one module (every declared callee is a discovered function, hence defined in some
//! shard). The instruction budget bounds both the lifter's memory (`jobs` shards in flight) and
//! the cost of compiling a shard, which grows faster than linearly: on a ~23M-instruction game
//! executable a 40 MB shard (about 64Ki instructions) takes ~15 s in `opt -O2` plus ~3 s in
//! `llc` at under 400 MiB, while a 300 MB shard takes ~90 s plus ~45 s at up to 1.9 GiB, hence
//! [`DEFAULT_SHARD_INSTRUCTIONS`] = 65536.
//!
//! `lift/functions.ll` defines the table for the guest runtime's bulk registration
//! (`pg6_guest_register_lifted_functions` in `runtime/guest/guest_runtime.h`, which takes a
//! sorted, duplicate-free array of `Pg6LiftedFunctionEntry { uint64_t va; LiftedFn fn; }`, 16
//! bytes on wasm64, plus a `size_t` count) and a function that hands it over:
//! ```text
//! %Pg6LiftedFunctionEntry = type { i64, ptr }
//! @__playgta6_lifted_functions_<module> = constant [N x %Pg6LiftedFunctionEntry] [...], align 8
//! @__playgta6_lifted_function_count_<module> = constant i64 N, align 8
//! define i32 @__playgta6_register_lifted_functions_<module>()  ; calls the runtime, returns its result
//! ```
//! Entries are sorted by `guest_va` (strictly ascending) and cover every lifted function, which
//! is every discovered function entry. The runtime (or the module's startup code) calls
//! `__playgta6_register_lifted_functions_main()` once before running guest code.
//!
//! # Manifest schema (`schema_version` 1)
//! Top-level keys, in order: `schema` (`"playgta6-recomp-build"`), `schema_version`, `module`,
//! `tool` (`name`, `version`, `llvm`: `null` or `{kind, name, version}`), `input` (`size`,
//! `fnv1a64`), `options` (`emit_obj`, `shard_size`, `shard_instructions`, `extra_entries`,
//! `shim_headers` as include spellings, `map_headers`, `executable_bytes`; `jobs` is
//! deliberately absent), `image` (`image_base`, `entry_point`, `size_of_image`, `sections`
//! [`name`, `rva`, `virtual_size`, `stored_bytes`, `protection`], `base_relocations` {`dir64`,
//! `highlow`}, `container_bytes`), `pdata` (`records`, `valid`, `functions`, `diagnostics` by
//! kind), `discovery` (`rounds`, `entry_points`, `functions`, `blocks`, `instructions`,
//! `jump_tables` {`resolved`, `unresolved`, `inferred`}, `terminators` {`trap`, `invalid`,
//! `no_return_call`}, `relocations` by kind, `diagnostics` by kind), `refinements` (`no_return`
//! {`imports` [`label`, `slot`], `thunks`}, `code_pointers` {`pdata_present`, `round_limit_hit`,
//! `by_source` {source: {status: count}}, `accepted`}, `dispatch` {`registered_functions`,
//! `address_taken_not_lifted` [`va`, `sources`, `status`]}, `import_slots` {`slot_reads`,
//! `iat_reads`, `non_iat_reads`, `misaligned`, `referenced_unresolved` [`dll`, `import`,
//! `reason`], `unreferenced` [`import`, `bound`]}), `lift` (`functions`, `instructions`,
//! `unsupported_instructions`, `unsupported_by_mnemonic`, `shards` [`index`, `first`, `last`,
//! `functions`, `instructions`, `ll`, `obj`], `registration` {`table_symbol`, `count_symbol`,
//! `register_function`, `entry_type`, `count`, `ll`, `obj`}), `imports` (`total`, `bound`,
//! `guest` (bound to another guest module by a program build), `unresolved`, `by_dll`
//! [`dll`, `bound`, `unresolved`], `catalogue` {`headers`, `shims`, `diagnostics`}),
//! `diagnostics` (totals: `pdata`, `discovery`, `image`, `catalogue`,
//! `unsupported_instructions`), and `files` [`path`, `size`, `fnv1a64`] for every output file
//! except the manifest itself, sorted by path (relative to the module directory). Addresses are
//! `"0x..."` strings, counts are numbers; block/instruction counts are per function (a block
//! shared by several functions is lifted into, and counted for, each), terminator and diagnostic
//! counts are per unique block. Hashes are 64-bit FNV-1a (the [`image::fnv1a_64`] parameters).

pub mod discovery_cache;
mod hygiene;
pub mod program;
pub mod refine;
pub mod shard_lift;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::{self, Write as _};
use std::fs;
use std::io::{self, BufWriter, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use crate::cfg::{CallTarget, DiagnosticKind, Function, Terminator};
use crate::code_image::CodeImage;
use crate::discover::Analysis;
use crate::eh::EhMetadata;
use crate::image::{self, ImageBuild, ImageDiagnostic, ImageError, ImageOptions};
use crate::lift::compile::{LlvmTool, compile_to_object, find_llvm_tool};
use crate::lift::memory::{Window, WindowCheck};
use crate::lift::target::LiftTarget;
use crate::lift::{
    self, GuestBase, LiftError, MemoryModel, ShardDispatch, WASM64_DATALAYOUT, WASM64_TRIPLE,
    function_symbol,
};
use crate::pdata::ExceptionDirectory;
use crate::pe::Pe64;
use crate::reloc::RelocKind;
use crate::stubs::{self, BridgePlan, Catalogue};

pub use hygiene::check_output_dir;
use refine::{CodePointerOutcome, NoReturnImport, PointerSource, PointerStatus, SlotCheck};

/// Build manifest file name.
pub const MANIFEST_FILE: &str = "build-manifest.json";
/// Coverage report file name.
pub const REPORT_FILE: &str = "report.txt";
/// Directory of lifted shards and the registration table.
pub const LIFT_DIR: &str = "lift";
/// Directory of the import bridge.
pub const STUBS_DIR: &str = "stubs";
/// Directory of the packed guest image.
pub const IMAGE_DIR: &str = "image";
/// Guest image container name inside [`IMAGE_DIR`].
pub const IMAGE_FILE: &str = "guest.pgim";
/// Guest image manifest name inside [`IMAGE_DIR`].
pub const IMAGE_MANIFEST_FILE: &str = "guest.pgim.json";
/// Registration table module name inside [`LIFT_DIR`] (stem shared by its object).
pub const REGISTRATION_STEM: &str = "functions";
/// Symbol prefix of the sorted `{ i64 va, ptr fn }` registration table (module name appended).
pub const REGISTRATION_TABLE_PREFIX: &str = "__playgta6_lifted_functions_";
/// Symbol prefix of the `i64` entry count of the registration table (module name appended).
pub const REGISTRATION_COUNT_PREFIX: &str = "__playgta6_lifted_function_count_";
/// Module name used when none is given.
pub const DEFAULT_MODULE: &str = "main";
/// LLVM type of one registration entry (O9's `Pg6LiftedFunctionEntry { uint64_t va; LiftedFn fn; }`).
pub const REGISTRATION_ENTRY_TYPE: &str = "%Pg6LiftedFunctionEntry";
/// Symbol prefix of the generated `i32 ()` function that registers the table with the runtime
/// (module name appended).
pub const REGISTER_FUNCTION_PREFIX: &str = "__playgta6_register_lifted_functions_";
/// The guest runtime's bulk registration entry point (`runtime/guest/guest_runtime.h`):
/// `int32_t pg6_guest_register_lifted_functions(const Pg6LiftedFunctionEntry*, size_t)`.
pub const RUNTIME_REGISTER_FUNCTION: &str = "pg6_guest_register_lifted_functions";
/// Symbol prefix of a shard's exported indirect-call dispatcher (`Pg6ShardDispatchFn`, module and
/// zero-padded shard index appended).
pub const SHARD_DISPATCHER_PREFIX: &str = "__playgta6_dispatch_shard_";
/// Symbol prefix of a module's `Pg6ShardDispatchTable` (module name appended).
pub const SHARD_DISPATCH_TABLE_PREFIX: &str = "__playgta6_shard_dispatch_";
/// ABI version of `Pg6ShardDispatchTable` (`kPg6ShardDispatchVersion` in `runtime/abi/playgta6_abi.h`).
pub const SHARD_DISPATCH_VERSION: u32 = 1;
/// LLVM type of one shard-dispatch function entry (`Pg6ShardFunctionEntry`).
pub const SHARD_ENTRY_TYPE: &str = "%Pg6ShardFunctionEntry";
/// LLVM type of the shard-dispatch table header (`Pg6ShardDispatchTable`).
pub const SHARD_TABLE_TYPE: &str = "%Pg6ShardDispatchTable";
/// Manifest schema name and version.
pub const MANIFEST_SCHEMA: &str = "playgta6-recomp-build";
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
/// Functions per shard unless overridden.
pub const DEFAULT_SHARD_SIZE: usize = 2048;
/// Instructions per shard unless overridden (see the module docs for how it was chosen).
pub const DEFAULT_SHARD_INSTRUCTIONS: usize = 65_536;

/// Entries listed per category in `report.txt` (the manifest lists everything).
const REPORT_LIST_LIMIT: usize = 20;
/// Mnemonics listed in the report's unsupported-instruction section.
const REPORT_TOP_MNEMONICS: usize = 15;

const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

const RELOC_KINDS: [RelocKind; 11] = [
    RelocKind::CallTarget,
    RelocKind::BranchTarget,
    RelocKind::RipData,
    RelocKind::RipAddress,
    RelocKind::CallSlot,
    RelocKind::JumpSlot,
    RelocKind::AbsoluteImmediate,
    RelocKind::AbsoluteDisplacement,
    RelocKind::JumpTable,
    RelocKind::JumpTableIndex,
    RelocKind::JumpTableEntry,
];

const DIAGNOSTIC_KINDS: [DiagnosticKind; 5] = [
    DiagnosticKind::InvalidInstruction,
    DiagnosticKind::TruncatedInstruction,
    DiagnosticKind::NotExecutable,
    DiagnosticKind::Unmapped,
    DiagnosticKind::OverlappingInstruction,
];

// ---------------------------------------------------------------------------------------------
// Options and errors
// ---------------------------------------------------------------------------------------------

/// How indirect calls (`__playgta6_indirect_call`) dispatch to lifted functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndirectDispatch {
    /// One `Pg6LiftedFunctionEntry { guest_va, LiftedFn }` per lifted function (default).
    #[default]
    Table,
    /// One exported `Pg6ShardDispatchFn` per lifted shard and a `Pg6ShardDispatchTable`
    /// (`runtime/abi/playgta6_abi.h`) with pure-data `(guest_va, shard_idx, local_idx)` entries.
    Shard,
}

impl IndirectDispatch {
    /// Parses a CLI mode name (`"table"` or `"shard"`).
    pub fn from_name(name: &str) -> Result<Self, String> {
        match name {
            "table" => Ok(Self::Table),
            "shard" => Ok(Self::Shard),
            other => Err(format!(
                "unknown indirect dispatch mode '{other}' (expected 'table' or 'shard')"
            )),
        }
    }

    /// Stable lowercase mode name (`"table"` or `"shard"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Shard => "shard",
        }
    }

    /// Whether this is the default (`Table`) mode.
    pub fn is_default(self) -> bool {
        self == Self::Table
    }
}

/// Everything [`run`] needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineOptions {
    pub pe_path: PathBuf,
    /// Build root; this module's outputs go to `out_dir/<module>/` ([`Self::module_dir`]).
    pub out_dir: PathBuf,
    /// Module name (`[A-Za-z0-9_]`, at most 64 characters): output subdirectory and suffix of the
    /// registration table symbols, so several guest modules can share one build root and link
    /// into one program.
    pub module: String,
    /// Runtime shim headers for the import bridge (may be empty: every import stays unresolved).
    pub shim_headers: Vec<PathBuf>,
    /// Spell generated `#include`s relative to this directory (see [`stubs::load_shim_header`]).
    pub include_root: Option<PathBuf>,
    /// Extra discovery entry points (VAs).
    pub extra_entries: Vec<u64>,
    /// Recorded indirect-call targets (`--seed-targets`, see [`crate::trace_seed`]): validated
    /// against this module's image and added as function entries.
    pub trace_seeds: Option<crate::trace_seed::TraceSeeds>,
    /// Appended one JSON line per module with the seeding outcome (`--seed-report`).
    pub trace_seed_report: Option<PathBuf>,
    /// Functions (entry VAs) that count their executions with an entry call to the runtime's
    /// `__playgta6_probe_hit` (`--probe-functions`). Empty: code generation is unchanged.
    pub probe_functions: BTreeSet<u64>,
    /// Compile every shard and the registration table to wasm64 objects.
    pub emit_obj: bool,
    /// Worker threads for lifting/compiling (does not affect any output).
    pub jobs: usize,
    /// Maximum functions per shard.
    pub shard_size: usize,
    /// Maximum instructions per shard (a larger function gets a shard of its own).
    pub shard_instructions: usize,
    pub image: ImageOptions,
    /// Code-generation target of the `.ll`/`.o` outputs ([`LiftTarget::Wasm64`] by default; the
    /// IR differs only in the lines [`LiftTarget::retarget`] rewrites).
    pub target: LiftTarget,
    /// How lifted code translates guest addresses ([`MemoryModel::default`]: unchanged output;
    /// another model is recorded in the manifests and the program descriptor).
    pub memory: MemoryModel,
    /// How indirect calls reach lifted functions ([`IndirectDispatch::Table`] by default).
    pub indirect_dispatch: IndirectDispatch,
    /// Shard lift cache directory (`--lift-cache`, see [`shard_lift`]): shards whose key is
    /// there are linked in instead of lifted. Never affects the outputs.
    pub lift_cache: Option<PathBuf>,
    /// Discovery cache directory (`--discovery-cache`, see [`discovery_cache`]): a module whose
    /// discovery inputs and discovery sources are unchanged skips discovery. Never affects the
    /// outputs.
    pub discovery_cache: Option<PathBuf>,
    /// With [`Self::discovery_cache`]: run discovery anyway and fail when the stored entry (or
    /// the encode/decode round trip) differs from it (`--verify-discovery-cache`).
    pub verify_discovery_cache: bool,
}

/// Parses a `--probe-functions` file: one hexadecimal VA per line (optional `0x`), anything after
/// whitespace on the line is a free-form label, `#` starts a comment line. Kept outside the
/// repository by the caller (the addresses are derived from the game).
pub fn parse_probe_functions(text: &str) -> Result<BTreeSet<u64>, String> {
    let mut out = BTreeSet::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let token = line.split_whitespace().next().unwrap_or("");
        let digits = token.strip_prefix("0x").or_else(|| token.strip_prefix("0X")).unwrap_or(token);
        let va = u64::from_str_radix(digits, 16)
            .map_err(|_| format!("line {}: '{token}' is not a hexadecimal address", n + 1))?;
        out.insert(va);
    }
    Ok(out)
}

impl PipelineOptions {
    /// Defaults: no shims, no extra entries, no objects, one job per available CPU,
    /// [`DEFAULT_SHARD_SIZE`], default image options.
    pub fn new(pe_path: impl Into<PathBuf>, out_dir: impl Into<PathBuf>) -> Self {
        Self {
            pe_path: pe_path.into(),
            out_dir: out_dir.into(),
            module: DEFAULT_MODULE.to_string(),
            shim_headers: Vec::new(),
            include_root: None,
            extra_entries: Vec::new(),
            trace_seeds: None,
            trace_seed_report: None,
            probe_functions: BTreeSet::new(),
            emit_obj: false,
            jobs: thread::available_parallelism().map_or(1, |n| n.get()),
            shard_size: DEFAULT_SHARD_SIZE,
            shard_instructions: DEFAULT_SHARD_INSTRUCTIONS,
            image: ImageOptions::default(),
            target: LiftTarget::Wasm64,
            memory: MemoryModel::default(),
            indirect_dispatch: IndirectDispatch::default(),
            lift_cache: None,
            discovery_cache: None,
            verify_discovery_cache: false,
        }
    }

    /// Directory receiving this module's outputs: `out_dir/<module>`.
    pub fn module_dir(&self) -> PathBuf {
        self.out_dir.join(&self.module)
    }
}

/// What a program build ([`program`]) adds to one module's build. The default (nothing) is a
/// plain single-module build.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModuleLink {
    /// IAT slot RVA -> export of another guest module. The slot is pre-filled with the export's
    /// guest VA in the packed guest image (`image/guest.pgim`) and the import gets no thunk
    /// ([`stubs::BridgePlan::bind_guest`]).
    pub guest_imports: BTreeMap<u32, stubs::GuestTarget>,
    /// IAT slot RVA -> `(dll, symbol)` the import bridge resolves instead of the import's own
    /// name (an import of a guest module's export that forwards to a runtime DLL).
    pub host_redirects: BTreeMap<u32, (String, crate::pe::ImportedSymbol)>,
    /// Module namespace of the import bridge ([`stubs::render_bridge_namespaced`]); `None`
    /// keeps the single-module names.
    pub bridge_namespace: Option<String>,
}

/// Whether `name` is a valid module name (see [`PipelineOptions::module`]).
pub fn is_valid_module_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// Registration table and count symbols of `module`.
pub fn registration_symbols(module: &str) -> (String, String) {
    (
        format!("{REGISTRATION_TABLE_PREFIX}{module}"),
        format!("{REGISTRATION_COUNT_PREFIX}{module}"),
    )
}

/// Symbol of the generated function registering `module`'s table with the guest runtime.
pub fn register_function_symbol(module: &str) -> String {
    format!("{REGISTER_FUNCTION_PREFIX}{module}")
}

/// Symbol of shard `index`'s exported indirect-call dispatcher for `module`.
pub fn shard_dispatcher_symbol(module: &str, index: usize) -> String {
    format!("{SHARD_DISPATCHER_PREFIX}{module}_{index:05}")
}

/// Symbol of `module`'s `Pg6ShardDispatchTable`.
pub fn shard_dispatch_table_symbol(module: &str) -> String {
    format!("{SHARD_DISPATCH_TABLE_PREFIX}{module}")
}

/// Computes the set of lifted function entry VAs that are called or tail-called directly from a
/// different shard. Only these functions need external linkage in [`IndirectDispatch::Shard`] mode;
/// every other lifted function can be emitted with `internal` linkage.
pub fn compute_cross_shard_callees(
    functions: &[&Function],
    shards: &[Range<usize>],
) -> HashSet<u64> {
    let mut shard_of: HashMap<u64, usize> = HashMap::with_capacity(functions.len());
    for (shard_idx, range) in shards.iter().enumerate() {
        for func in &functions[range.clone()] {
            shard_of.insert(func.entry, shard_idx);
        }
    }
    let mut cross_shard = HashSet::new();
    for (caller_shard, range) in shards.iter().enumerate() {
        for func in &functions[range.clone()] {
            let direct_calls = func.calls().filter_map(|c| match c.target {
                CallTarget::Direct(target) => Some(target),
                CallTarget::Indirect { .. } => None,
            });
            for target in func
                .callees
                .iter()
                .copied()
                .chain(func.tail_calls())
                .chain(direct_calls)
            {
                if shard_of
                    .get(&target)
                    .is_some_and(|&callee_shard| callee_shard != caller_shard)
                {
                    cross_shard.insert(target);
                }
            }
        }
    }
    cross_shard
}

/// Why a build failed. Messages carry names, addresses and the user's own paths only.
#[derive(Debug)]
pub enum PipelineError {
    /// Invalid option values.
    Options(String),
    /// `--out` failed [`check_output_dir`].
    OutputDir(String),
    Io {
        context: String,
        error: io::Error,
    },
    /// The input is not a valid PE32+.
    Parse(String),
    Image(ImageError),
    /// A function's CFG is malformed (a discovery bug); nothing can be emitted for it.
    Lift {
        entry: u64,
        error: LiftError,
    },
    /// `emit_obj` was requested but no LLVM tool was found.
    NoLlvmTool,
    Compile {
        file: String,
        error: String,
    },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Options(msg) | Self::OutputDir(msg) | Self::Parse(msg) => f.write_str(msg),
            Self::Io { context, error } => write!(f, "{context}: {error}"),
            Self::Image(error) => write!(f, "failed to build the guest image: {error}"),
            Self::Lift { entry, error } => write!(f, "lifting function {entry:#x} failed: {error}"),
            Self::NoLlvmTool => f.write_str(
                "--emit-obj needs llc or clang (PLAYGTA6_LLC, PATH, or `rustup component add llvm-tools`)",
            ),
            Self::Compile { file, error } => write!(f, "compiling {file} failed: {error}"),
        }
    }
}

impl std::error::Error for PipelineError {}

fn io_error(context: impl Into<String>) -> impl FnOnce(io::Error) -> PipelineError {
    let context = context.into();
    move |error| PipelineError::Io { context, error }
}

// ---------------------------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------------------------

/// The input file, read once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Input {
    pub bytes: Vec<u8>,
    /// FNV-1a 64 of `bytes` (identifies the input in the manifest; not game data).
    pub hash: u64,
}

/// Stage 1: read the PE.
pub fn load_input(path: &Path) -> Result<Input, PipelineError> {
    let bytes = fs::read(path).map_err(io_error(format!(
        "failed to read PE file '{}'",
        path.display()
    )))?;
    let hash = image::fnv1a_64(&bytes);
    Ok(Input { bytes, hash })
}

/// Result of [`seed`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seeded<'a> {
    /// Entry points: PE entry, exports, TLS callbacks, `.pdata` functions and handlers; the
    /// `.pdata` function table; no-return import slots.
    pub image: CodeImage<'a>,
    pub pdata: ExceptionDirectory,
    /// Exception-handling metadata ([`EhMetadata::seed`]: funclet entries, resume targets).
    pub eh: EhMetadata,
    /// The imports whose slots were put into `image.no_return`.
    pub no_return: Vec<NoReturnImport>,
}

/// Stage 3: map `pe` (parsed from `file`) onto a [`CodeImage`], seed it from `.pdata` and the EH
/// metadata, and mark the IAT slots of non-returning APIs ([`refine::no_return_imports`]).
pub fn seed<'a>(pe: &Pe64, file: &'a [u8]) -> Seeded<'a> {
    let mut image = CodeImage::from_pe(pe, file);
    let pdata = ExceptionDirectory::from_pe(pe, file);
    pdata.seed(&mut image);
    let eh = EhMetadata::from_pe(pe, file, &pdata);
    eh.seed(&mut image);
    let no_return = refine::no_return_imports(pe);
    image.no_return = no_return.iter().map(|n| n.slot).collect();
    Seeded {
        image,
        pdata,
        eh,
        no_return,
    }
}

/// Stage 5: read the shim headers and resolve every import against them.
pub fn plan_imports(
    pe: &Pe64,
    headers: &[PathBuf],
    include_root: Option<&Path>,
) -> Result<(Vec<String>, Catalogue, BridgePlan), PipelineError> {
    let mut loaded = Vec::with_capacity(headers.len());
    for header in headers {
        loaded.push(
            stubs::load_shim_header(header, include_root).map_err(io_error(format!(
                "failed to read shim header '{}'",
                header.display()
            )))?,
        );
    }
    let includes = loaded.iter().map(|h| h.include.clone()).collect();
    let catalogue = Catalogue::scan(&loaded);
    let plan = stubs::plan_bridge(pe, &catalogue);
    Ok((includes, catalogue, plan))
}

/// A copy of `file` whose IAT slots at the RVAs in `guest_imports` hold their export's guest VA
/// (little-endian `u64`), or `None` when there is nothing to pre-fill. Every slot must be an IAT
/// slot of `pe` backed by file bytes.
pub fn prefill_iat(
    pe: &Pe64,
    file: &[u8],
    guest_imports: &BTreeMap<u32, stubs::GuestTarget>,
) -> Result<Option<Vec<u8>>, PipelineError> {
    if guest_imports.is_empty() {
        return Ok(None);
    }
    let mut bytes = file.to_vec();
    for (&slot, target) in guest_imports {
        let offset = pe
            .rva_to_file_offset(slot)
            .filter(|&o| pe.rva_to_file_offset(slot + 7) == Some(o + 7))
            .ok_or_else(|| {
                PipelineError::Options(format!(
                    "IAT slot 0x{slot:x} is not backed by file bytes and cannot be pre-filled"
                ))
            })?;
        bytes[offset..offset + 8].copy_from_slice(&target.va.to_le_bytes());
    }
    Ok(Some(bytes))
}

/// Applies a program build's bindings to the import bridge plan.
fn apply_link(
    plan: &mut BridgePlan,
    catalogue: &Catalogue,
    link: &ModuleLink,
) -> Result<(), PipelineError> {
    for (&slot, target) in &link.guest_imports {
        if !plan.bind_guest(slot, target.clone()) {
            return Err(PipelineError::Options(format!(
                "no import uses IAT slot 0x{slot:x}"
            )));
        }
    }
    if !link.host_redirects.is_empty() {
        let resolver = stubs::ShimResolver::new(catalogue);
        for (&slot, (dll, symbol)) in &link.host_redirects {
            let import = plan
                .imports
                .iter_mut()
                .find(|i| i.slot_rva == Some(slot))
                .ok_or_else(|| {
                    PipelineError::Options(format!("no import uses IAT slot 0x{slot:x}"))
                })?;
            import.resolution = resolver.resolve(dll, symbol);
        }
    }
    Ok(())
}

/// The runtime's shim headers of the checkout containing `start` (or one of its ancestors):
/// `runtime/os/*.h` (sorted), `runtime/gpu/d3d12_xs.h` and `runtime/guest/guest_callbacks.h`,
/// then the export-table sources `runtime/os/*.cpp` (sorted) and `runtime/gpu/d3d12_xs_exports.*`
/// when present, with `runtime/` as include root.
/// `None` when `start` is not inside a playgta6web checkout.
pub fn checkout_shim_headers(start: &Path) -> Option<(Vec<PathBuf>, PathBuf)> {
    let root = start.ancestors().find(|dir| {
        dir.join("runtime")
            .join("abi")
            .join("playgta6_abi.h")
            .is_file()
            && dir.join("recomp").join("Cargo.toml").is_file()
    })?;
    let runtime = root.join("runtime");
    let mut headers: Vec<PathBuf> = fs::read_dir(runtime.join("os"))
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "h"))
        .collect();
    headers.sort();
    let gpu = runtime.join("gpu").join("d3d12_xs.h");
    if gpu.is_file() {
        headers.push(gpu);
    }
    // O16's complete guest-state import thunks (`pg6_gthunk_*`).
    let callbacks = runtime.join("guest").join("guest_callbacks.h");
    if callbacks.is_file() {
        headers.push(callbacks);
    }
    // The export-registration tables (`runtime/os/*.cpp`): authoritative export names and
    // ordinals of the shim families (`stubs::scan_tables`). Sources are scanned for tables
    // only, never for declarations.
    let mut sources: Vec<PathBuf> = fs::read_dir(runtime.join("os"))
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "cpp"))
        .collect();
    sources.sort();
    // G24's d3d12_xs export table, once it exists.
    for name in ["d3d12_xs_exports.h", "d3d12_xs_exports.cpp"] {
        let path = runtime.join("gpu").join(name);
        if path.is_file() {
            sources.push(path);
        }
    }
    headers.extend(sources);
    (!headers.is_empty()).then_some((headers, runtime))
}

/// Stage 6a: cuts functions (in entry order, `weights[i]` = instruction count of function `i`)
/// into consecutive shards: a shard is closed before a function that would make it exceed
/// `max_functions` functions or `max_instructions` instructions. A function larger than
/// `max_instructions` gets a shard of its own. Returns index ranges covering `0..weights.len()`.
pub fn plan_shards(
    weights: &[usize],
    max_functions: usize,
    max_instructions: usize,
) -> Vec<Range<usize>> {
    let mut shards = Vec::new();
    let (mut start, mut total) = (0, 0usize);
    for (i, &weight) in weights.iter().enumerate() {
        let len = i - start;
        if len > 0 && (len >= max_functions || total + weight > max_instructions) {
            shards.push(start..i);
            (start, total) = (i, 0);
        }
        total += weight;
    }
    if start < weights.len() {
        shards.push(start..weights.len());
    }
    shards
}

/// File name of shard `index` with extension `ext` (`ll` or `o`).
pub fn shard_file_name(index: usize, ext: &str) -> String {
    format!("shard-{index:05}.{ext}")
}

/// One written output file (path relative to the output directory, `/`-separated).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct OutputFile {
    pub path: String,
    pub size: u64,
    pub hash: u64,
}

/// One lifted shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardRecord {
    pub index: usize,
    /// Entry VAs of its first and last function.
    pub first: u64,
    pub last: u64,
    pub functions: usize,
    /// Instructions of its functions (per function).
    pub instructions: usize,
    /// Unsupported instructions by mnemonic.
    pub unsupported: BTreeMap<String, usize>,
    pub ll: OutputFile,
    pub obj: Option<OutputFile>,
}

/// Stage 6b: lift `functions` (sorted by entry) shard by shard into `out_dir/lift/` on `jobs`
/// threads. Each worker lifts one shard at a time, writes its module, drops the IR and, with
/// `tool`, compiles it. Results are ordered by shard index; on failure the error of the lowest
/// failing shard is returned.
pub fn lift_shards(
    functions: &[&Function],
    shards: &[Range<usize>],
    out_dir: &Path,
    tool: Option<&LlvmTool>,
    jobs: usize,
) -> Result<Vec<ShardRecord>, PipelineError> {
    lift_shards_for(
        &LiftTarget::Wasm64,
        &MemoryModel::default(),
        functions,
        shards,
        out_dir,
        tool,
        jobs,
    )
}

/// [`lift_shards`] for a given code-generation target (shard text rewritten by
/// [`LiftTarget::retarget`], objects compiled for that target) and memory model.
pub fn lift_shards_for(
    target: &LiftTarget,
    memory: &MemoryModel,
    functions: &[&Function],
    shards: &[Range<usize>],
    out_dir: &Path,
    tool: Option<&LlvmTool>,
    jobs: usize,
) -> Result<Vec<ShardRecord>, PipelineError> {
    lift_shards_with_dispatch(
        target,
        memory,
        DEFAULT_MODULE,
        IndirectDispatch::Table,
        &BTreeSet::new(),
        functions,
        shards,
        out_dir,
        tool,
        jobs,
    )
}

/// [`lift_shards_for`] with explicit module name and indirect-call dispatch mode.
#[allow(clippy::too_many_arguments)]
pub fn lift_shards_with_dispatch(
    target: &LiftTarget,
    memory: &MemoryModel,
    module: &str,
    indirect_dispatch: IndirectDispatch,
    probes: &BTreeSet<u64>,
    functions: &[&Function],
    shards: &[Range<usize>],
    out_dir: &Path,
    tool: Option<&LlvmTool>,
    jobs: usize,
) -> Result<Vec<ShardRecord>, PipelineError> {
    lift_shards_cached(
        target,
        memory,
        module,
        indirect_dispatch,
        probes,
        None,
        functions,
        shards,
        out_dir,
        tool,
        jobs,
    )
    .map(|(records, _)| records)
}

/// [`lift_shards_with_dispatch`] reusing the shards found in `cache` ([`shard_lift`]); also
/// returns, per shard, whether its IR came from the cache.
#[allow(clippy::too_many_arguments)]
pub fn lift_shards_cached(
    target: &LiftTarget,
    memory: &MemoryModel,
    module: &str,
    indirect_dispatch: IndirectDispatch,
    probes: &BTreeSet<u64>,
    cache: Option<&shard_lift::ShardCache>,
    functions: &[&Function],
    shards: &[Range<usize>],
    out_dir: &Path,
    tool: Option<&LlvmTool>,
    jobs: usize,
) -> Result<(Vec<ShardRecord>, Vec<bool>), PipelineError> {
    let cross_shard_callees = match indirect_dispatch {
        IndirectDispatch::Table => None,
        IndirectDispatch::Shard => Some(compute_cross_shard_callees(functions, shards)),
    };
    // Whole-program flag summaries: direct calls across shards link to the same functions.
    let flag_summaries = lift::FlagSummaries::compute(functions.iter().copied());
    let lift_dir = out_dir.join(LIFT_DIR);
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let results = Mutex::new(Vec::with_capacity(shards.len()));
    thread::scope(|scope| {
        for _ in 0..jobs.clamp(1, shards.len().max(1)) {
            scope.spawn(|| {
                while !failed.load(Ordering::Relaxed) {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(range) = shards.get(index) else {
                        break;
                    };
                    let dispatcher_symbol = (indirect_dispatch == IndirectDispatch::Shard)
                        .then(|| shard_dispatcher_symbol(module, index));
                    let dispatch = dispatcher_symbol.as_deref().map(|sym| ShardDispatch {
                        dispatcher_symbol: sym,
                        external_entries: cross_shard_callees
                            .as_ref()
                            .expect("cross-shard callees computed in Shard mode"),
                    });
                    let context = shard_lift::ShardContext {
                        target,
                        memory,
                        dispatch,
                        probes,
                    };
                    let result = shard_lift::lift_shard(
                        &context,
                        &flag_summaries,
                        cache,
                        index,
                        &functions[range.clone()],
                        &lift_dir,
                        tool,
                    );
                    if result.is_err() {
                        failed.store(true, Ordering::Relaxed);
                    }
                    results
                        .lock()
                        .expect("shard results lock")
                        .push((index, result));
                }
            });
        }
    });
    let mut results = results.into_inner().expect("shard results lock");
    results.sort_by_key(|(index, _)| *index);
    let mut records = Vec::with_capacity(results.len());
    let mut hits = Vec::with_capacity(results.len());
    for (_, result) in results {
        let (record, hit) = result?;
        records.push(record);
        hits.push(hit);
    }
    Ok((records, hits))
}

/// Compiles `lift_dir/ll_name` to `lift_dir/obj_name` and records the object.
fn compile(
    tool: &LlvmTool,
    target: &LiftTarget,
    lift_dir: &Path,
    ll_name: &str,
    obj_name: &str,
) -> Result<OutputFile, PipelineError> {
    let obj_path = lift_dir.join(obj_name);
    compile_to_object(tool, target, &lift_dir.join(ll_name), &obj_path).map_err(|e| {
        PipelineError::Compile {
            file: format!("{LIFT_DIR}/{ll_name}"),
            error: e.to_string(),
        }
    })?;
    let (size, hash) = hash_file(&obj_path)?;
    Ok(OutputFile {
        path: format!("{LIFT_DIR}/{obj_name}"),
        size,
        hash,
    })
}

/// Writes the registration table module (see the module docs) for the sorted, distinct
/// function entries `entries`.
pub fn write_registration_table(
    module: &str,
    entries: &[u64],
    out: &mut impl Write,
) -> io::Result<()> {
    debug_assert!(entries.windows(2).all(|w| w[0] < w[1]));
    let ty = REGISTRATION_ENTRY_TYPE;
    let (table, count_symbol) = registration_symbols(module);
    writeln!(
        out,
        "; playgta6 recomp: lifted function registration table (generated file)"
    )?;
    writeln!(
        out,
        "; @{table}: {{ i64 guest_va, ptr fn }} sorted by guest_va; @{count_symbol}: entry count"
    )?;
    writeln!(out, "target datalayout = \"{WASM64_DATALAYOUT}\"")?;
    writeln!(out, "target triple = \"{WASM64_TRIPLE}\"")?;
    writeln!(out)?;
    writeln!(out, "{ty} = type {{ i64, ptr }}")?;
    writeln!(out)?;
    for &va in entries {
        writeln!(out, "declare void @{}(ptr)", function_symbol(va))?;
    }
    if !entries.is_empty() {
        writeln!(out)?;
    }
    let count = entries.len();
    if entries.is_empty() {
        writeln!(
            out,
            "@{table} = constant [0 x {ty}] zeroinitializer, align 8"
        )?;
    } else {
        writeln!(out, "@{table} = constant [{count} x {ty}] [")?;
        for (i, &va) in entries.iter().enumerate() {
            let sep = if i + 1 == count { "" } else { "," };
            writeln!(
                out,
                "  {ty} {{ i64 {}, ptr @{} }}{sep}",
                va as i64,
                function_symbol(va)
            )?;
        }
        writeln!(out, "], align 8")?;
    }
    writeln!(out, "@{count_symbol} = constant i64 {count}, align 8")?;
    writeln!(out)?;
    writeln!(out, "declare i32 @{RUNTIME_REGISTER_FUNCTION}(ptr, i64)")?;
    writeln!(out)?;
    writeln!(out, "define i32 @{}() {{", register_function_symbol(module))?;
    writeln!(out, "entry:")?;
    writeln!(
        out,
        "  %ok = call i32 @{RUNTIME_REGISTER_FUNCTION}(ptr @{table}, i64 {count})"
    )?;
    writeln!(out, "  ret i32 %ok")?;
    writeln!(out, "}}")
}

/// Writes the shard-level dispatch registration table (`Pg6ShardDispatchTable`,
/// `runtime/abi/playgta6_abi.h`) for `shard_count` shard dispatchers and `entries` of
/// `(guest_va, shard_idx, local_idx)` sorted strictly ascending by `guest_va`.
///
/// Emits zero `@sub_*` declarations or per-function relocations, plus empty legacy symbols so
/// any remaining references to `__playgta6_lifted_functions_<module>` still link cleanly.
pub fn write_shard_registration_table(
    module: &str,
    shard_count: usize,
    entries: &[(u64, u32, u32)],
    out: &mut impl Write,
) -> io::Result<()> {
    debug_assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
    debug_assert!(entries.iter().all(|&(_, s, _)| (s as usize) < shard_count));
    let dispatch_table = shard_dispatch_table_symbol(module);
    let dispatchers = format!("__playgta6_shard_dispatchers_{module}");
    let shard_entries = format!("__playgta6_shard_entries_{module}");
    let (legacy_table, legacy_count) = registration_symbols(module);
    writeln!(
        out,
        "; playgta6 recomp: shard-level lifted function dispatch table (generated file)"
    )?;
    writeln!(
        out,
        "; @{dispatch_table}: Pg6ShardDispatchTable (version {SHARD_DISPATCH_VERSION})"
    )?;
    writeln!(out, "target datalayout = \"{WASM64_DATALAYOUT}\"")?;
    writeln!(out, "target triple = \"{WASM64_TRIPLE}\"")?;
    writeln!(out)?;
    writeln!(out, "{SHARD_ENTRY_TYPE} = type {{ i64, i32, i32 }}")?;
    writeln!(
        out,
        "{SHARD_TABLE_TYPE} = type {{ i32, i32, i64, ptr, ptr }}"
    )?;
    writeln!(out, "{REGISTRATION_ENTRY_TYPE} = type {{ i64, ptr }}")?;
    writeln!(out)?;
    for idx in 0..shard_count {
        writeln!(
            out,
            "declare void @{}(i32, ptr)",
            shard_dispatcher_symbol(module, idx)
        )?;
    }
    if shard_count > 0 {
        writeln!(out)?;
    }
    if shard_count == 0 {
        writeln!(
            out,
            "@{dispatchers} = internal constant [0 x ptr] zeroinitializer, align 8"
        )?;
    } else {
        writeln!(
            out,
            "@{dispatchers} = internal constant [{shard_count} x ptr] ["
        )?;
        for idx in 0..shard_count {
            let sep = if idx + 1 == shard_count { "" } else { "," };
            writeln!(
                out,
                "  ptr @{}{sep}",
                shard_dispatcher_symbol(module, idx)
            )?;
        }
        writeln!(out, "], align 8")?;
    }
    writeln!(out)?;
    let entry_count = entries.len();
    if entries.is_empty() {
        writeln!(
            out,
            "@{shard_entries} = internal constant [0 x {SHARD_ENTRY_TYPE}] zeroinitializer, align 8"
        )?;
    } else {
        writeln!(
            out,
            "@{shard_entries} = internal constant [{entry_count} x {SHARD_ENTRY_TYPE}] ["
        )?;
        for (i, &(va, shard_idx, local_idx)) in entries.iter().enumerate() {
            let sep = if i + 1 == entry_count { "" } else { "," };
            writeln!(
                out,
                "  {SHARD_ENTRY_TYPE} {{ i64 {}, i32 {shard_idx}, i32 {local_idx} }}{sep}",
                va as i64
            )?;
        }
        writeln!(out, "], align 8")?;
    }
    writeln!(out)?;
    let shards_ptr = if shard_count == 0 {
        "null".to_string()
    } else {
        format!("@{dispatchers}")
    };
    let entries_ptr = if entries.is_empty() {
        "null".to_string()
    } else {
        format!("@{shard_entries}")
    };
    writeln!(
        out,
        "@{dispatch_table} = constant {SHARD_TABLE_TYPE} {{ i32 {SHARD_DISPATCH_VERSION}, i32 {shard_count}, i64 {entry_count}, ptr {shards_ptr}, ptr {entries_ptr} }}, align 8"
    )?;
    writeln!(out)?;
    writeln!(
        out,
        "@{legacy_table} = constant [0 x {REGISTRATION_ENTRY_TYPE}] zeroinitializer, align 8"
    )?;
    writeln!(out, "@{legacy_count} = constant i64 0, align 8")?;
    writeln!(out)?;
    writeln!(out, "declare i32 @{RUNTIME_REGISTER_FUNCTION}(ptr, i64)")?;
    writeln!(out)?;
    writeln!(out, "define i32 @{}() {{", register_function_symbol(module))?;
    writeln!(out, "entry:")?;
    writeln!(
        out,
        "  %ok = call i32 @{RUNTIME_REGISTER_FUNCTION}(ptr @{legacy_table}, i64 0)"
    )?;
    writeln!(out, "  ret i32 %ok")?;
    writeln!(out, "}}")
}

/// Removes shard and registration files a previous build left in `lift_dir`.
fn clean_lift_dir(lift_dir: &Path) -> Result<(), PipelineError> {
    let entries = match fs::read_dir(lift_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_error(format!("failed to list '{}'", lift_dir.display()))(e)),
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let stale = name
            .strip_prefix("shard-")
            .and_then(|rest| rest.split_once('.'))
            .is_some_and(|(digits, ext)| {
                !digits.is_empty()
                    && digits.bytes().all(|b| b.is_ascii_digit())
                    && matches!(ext, "ll" | "o")
            })
            || name == format!("{REGISTRATION_STEM}.ll")
            || name == format!("{REGISTRATION_STEM}.o");
        if stale && entry.path().is_file() {
            fs::remove_file(entry.path()).map_err(io_error(format!(
                "failed to remove stale '{}'",
                entry.path().display()
            )))?;
        }
    }
    Ok(())
}

/// Size and FNV-1a 64 of a file, read in chunks.
fn hash_file(path: &Path) -> Result<(u64, u64), PipelineError> {
    let context = || format!("failed to read '{}'", path.display());
    let mut file = fs::File::open(path).map_err(io_error(context()))?;
    let mut buf = vec![0u8; 1 << 16];
    let (mut size, mut hash) = (0u64, FNV1A_64_OFFSET_BASIS);
    loop {
        let n = file.read(&mut buf).map_err(io_error(context()))?;
        if n == 0 {
            return Ok((size, hash));
        }
        size += n as u64;
        for &b in &buf[..n] {
            hash = (hash ^ u64::from(b)).wrapping_mul(FNV1A_64_PRIME);
        }
    }
}

fn write_file(out_dir: &Path, rel: &str, bytes: &[u8]) -> Result<OutputFile, PipelineError> {
    let path = out_dir.join(rel);
    fs::write(&path, bytes).map_err(io_error(format!("failed to write '{}'", path.display())))?;
    Ok(OutputFile {
        path: rel.to_string(),
        size: bytes.len() as u64,
        hash: image::fnv1a_64(bytes),
    })
}

/// The LLVM tool recorded in the manifest (file name and version line only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolInfo {
    /// `llc`, `llc+opt` or `clang`.
    pub kind: &'static str,
    pub name: String,
    pub version: String,
}

fn tool_info(tool: &LlvmTool) -> ToolInfo {
    let kind = match tool {
        LlvmTool::Llc { opt: Some(_), .. } => "llc+opt",
        LlvmTool::Llc { opt: None, .. } => "llc",
        LlvmTool::Clang(_) => "clang",
    };
    let version = Command::new(tool.path())
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .find(|l| l.to_ascii_lowercase().contains("version"))
                .map(str::to_string)
        })
        .unwrap_or_else(|| "unknown".to_string());
    ToolInfo {
        kind,
        name: tool
            .path()
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
        version,
    }
}

// ---------------------------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------------------------

/// Discovery statistics of the final fixpoint round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryStats {
    pub rounds: usize,
    pub entry_points: usize,
    pub functions: usize,
    pub blocks: usize,
    pub instructions: usize,
    pub jump_tables_resolved: usize,
    pub jump_tables_unresolved: usize,
    /// Resolved tables whose extent was inferred (no proven index bound); part of `resolved`.
    pub jump_tables_inferred: usize,
    /// Unresolved indirect jumps by reason, in [`crate::cfg::UnresolvedJump`] declaration order.
    pub jump_tables_unresolved_by_reason: Vec<(&'static str, usize)>,
    pub trap_blocks: usize,
    pub invalid_blocks: usize,
    pub no_return_calls: usize,
    /// Relocations by kind, in [`RelocKind`] declaration order.
    pub relocations: Vec<(&'static str, usize)>,
    /// Unique diagnostics by kind, in [`DiagnosticKind`] declaration order.
    pub diagnostics: Vec<(&'static str, usize)>,
}

impl DiscoveryStats {
    fn new(image: &CodeImage<'_>, analysis: &Analysis, rounds: usize) -> Self {
        let mut stats = Self {
            rounds,
            entry_points: image.entry_points.len(),
            functions: analysis.functions.len(),
            blocks: 0,
            instructions: 0,
            jump_tables_resolved: analysis.jump_tables.len(),
            jump_tables_unresolved: 0,
            jump_tables_inferred: analysis.jump_tables.values().filter(|t| t.inferred).count(),
            jump_tables_unresolved_by_reason: crate::cfg::UnresolvedJump::ALL
                .iter()
                .map(|&k| {
                    (
                        k.name(),
                        analysis.unresolved_jumps.values().filter(|&&r| r == k).count(),
                    )
                })
                .collect(),
            trap_blocks: 0,
            invalid_blocks: 0,
            no_return_calls: 0,
            relocations: Vec::new(),
            diagnostics: Vec::new(),
        };
        let mut seen = HashSet::new();
        let mut diagnostics = BTreeSet::new();
        for function in analysis.functions.values() {
            stats.blocks += function.blocks.len();
            stats.instructions += function.instruction_count();
            for block in function.blocks.values() {
                if !seen.insert(block.start) {
                    continue;
                }
                diagnostics.extend(block.diagnostics.iter().copied());
                match block.terminator {
                    Terminator::Trap => stats.trap_blocks += 1,
                    Terminator::Invalid => stats.invalid_blocks += 1,
                    Terminator::NoReturnCall => stats.no_return_calls += 1,
                    _ => {}
                }
                if block
                    .successors
                    .contains(&crate::cfg::Edge::IndirectJump { slot: None })
                {
                    stats.jump_tables_unresolved += 1;
                }
            }
        }
        let entries = analysis.relocations.entries();
        stats.relocations = RELOC_KINDS
            .iter()
            .map(|&k| (k.as_str(), entries.iter().filter(|r| r.kind == k).count()))
            .collect();
        stats.diagnostics = DIAGNOSTIC_KINDS
            .iter()
            .map(|&k| {
                (
                    k.as_str(),
                    diagnostics.iter().filter(|d| d.kind == k).count(),
                )
            })
            .collect();
        stats
    }

    fn diagnostic_total(&self) -> usize {
        self.diagnostics.iter().map(|(_, n)| n).sum()
    }
}

/// `.pdata` statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdataStats {
    pub records: usize,
    pub valid: usize,
    pub functions: usize,
    pub diagnostics: BTreeMap<&'static str, usize>,
}

impl PdataStats {
    fn new(pdata: &ExceptionDirectory, image: &CodeImage<'_>) -> Self {
        let mut diagnostics = BTreeMap::new();
        for d in &pdata.diagnostics {
            *diagnostics.entry(d.kind.as_str()).or_insert(0) += 1;
        }
        Self {
            records: pdata.record_count,
            valid: pdata.entries.len(),
            functions: image.function_table.function_count(),
            diagnostics,
        }
    }
}

/// Guest image statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageStats {
    pub image_base: u64,
    pub entry_point: u64,
    pub size_of_image: u32,
    /// `(name, rva, virtual_size, stored_bytes, protection)` in RVA order.
    pub sections: Vec<(String, u32, u32, u64, &'static str)>,
    pub dir64: usize,
    pub highlow: usize,
    pub container_bytes: u64,
    pub diagnostics: BTreeMap<&'static str, usize>,
}

fn image_diagnostic_kind(d: &ImageDiagnostic) -> &'static str {
    match d {
        ImageDiagnostic::UnsupportedRelocType { .. } => "unsupported_reloc_type",
        ImageDiagnostic::DuplicateRelocation { .. } => "duplicate_relocation",
        ImageDiagnostic::RawDataTruncated { .. } => "raw_data_truncated",
        ImageDiagnostic::EmptySection { .. } => "empty_section",
        ImageDiagnostic::RelocationsInOmittedSection { .. } => "relocations_in_omitted_section",
    }
}

impl ImageStats {
    fn new(built: &ImageBuild<'_>, container_bytes: u64) -> Self {
        let image = &built.image;
        let (dir64, highlow) = image.relocation_counts();
        let mut diagnostics = BTreeMap::new();
        for d in &built.diagnostics {
            *diagnostics.entry(image_diagnostic_kind(d)).or_insert(0) += 1;
        }
        Self {
            image_base: image.image_base,
            entry_point: if image.entry_point_rva == 0 {
                0
            } else {
                image.image_base + u64::from(image.entry_point_rva)
            },
            size_of_image: image.size_of_image,
            sections: image
                .sections
                .iter()
                .map(|s| {
                    (
                        s.name.clone(),
                        s.rva,
                        s.virtual_size,
                        s.data.len() as u64,
                        s.protection.as_str(),
                    )
                })
                .collect(),
            dir64,
            highlow,
            container_bytes,
            diagnostics,
        }
    }
}

/// Import bridge statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportStats {
    pub total: usize,
    pub bound: usize,
    /// Bound to another guest module by a program build ([`ModuleLink::guest_imports`]).
    pub guest: usize,
    pub unresolved: usize,
    /// `(dll, bound, unresolved)`, sorted.
    pub by_dll: Vec<(String, usize, usize)>,
    /// Include spellings of the scanned headers.
    pub headers: Vec<String>,
    pub shims: usize,
    pub catalogue_diagnostics: usize,
}

/// Lift statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiftStats {
    pub shard_size: usize,
    pub shard_instructions: usize,
    pub shards: Vec<ShardRecord>,
    pub functions: usize,
    pub instructions: usize,
    pub unsupported: BTreeMap<String, usize>,
    pub registration_ll: OutputFile,
    pub registration_obj: Option<OutputFile>,
}

impl LiftStats {
    pub fn unsupported_total(&self) -> usize {
        self.unsupported.values().sum()
    }
}

/// Everything a build produced. `report` is `report.txt`, `manifest` is `build-manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineReport {
    pub module: String,
    pub input_size: u64,
    pub input_hash: u64,
    pub image: ImageStats,
    pub pdata: PdataStats,
    pub discovery: DiscoveryStats,
    pub no_return: Vec<NoReturnImport>,
    pub code_pointers: CodePointerOutcome,
    pub pdata_present: bool,
    pub slots: SlotCheck,
    pub imports: ImportStats,
    pub lift: LiftStats,
    pub llvm: Option<ToolInfo>,
    pub extra_entries: Vec<u64>,
    pub emit_obj: bool,
    pub image_options: ImageOptions,
    pub target: LiftTarget,
    pub memory: MemoryModel,
    pub indirect_dispatch: IndirectDispatch,
    /// Every output file except the manifest, sorted by path.
    pub files: Vec<OutputFile>,
    /// Entry VA of every lifted function (sorted; the registration table's entries).
    pub function_entries: Vec<u64>,
    pub report: String,
    pub manifest: String,
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------

/// Runs the whole pipeline (see the module docs).
pub fn run(options: &PipelineOptions) -> Result<PipelineReport, PipelineError> {
    run_with_progress(options, &mut |_| {})
}

/// [`run`], calling `progress` with a short stage name (`parse`, `image`, `discover`, `lift`,
/// `registration`, `stubs`, `pack`, `manifest`) as each stage completes (for timing and
/// progress output; it never affects the outputs).
pub fn run_with_progress(
    options: &PipelineOptions,
    progress: &mut dyn FnMut(&str),
) -> Result<PipelineReport, PipelineError> {
    check_options(options)?;
    let input = load_input(&options.pe_path)?;
    run_module(options, &input, &ModuleLink::default(), progress)
}

fn check_options(options: &PipelineOptions) -> Result<(), PipelineError> {
    if options.jobs == 0 {
        return Err(PipelineError::Options("--jobs must be at least 1".into()));
    }
    if options.shard_size == 0 || options.shard_instructions == 0 {
        return Err(PipelineError::Options(
            "--shard-size and --shard-instructions must be at least 1".into(),
        ));
    }
    if !is_valid_module_name(&options.module) {
        return Err(PipelineError::Options(format!(
            "invalid module name '{}' (use 1-64 characters from [A-Za-z0-9_])",
            options.module
        )));
    }
    check_output_dir(&options.out_dir).map_err(PipelineError::OutputDir)?;
    Ok(())
}

/// [`run_with_progress`] on input bytes already read (and, for a program build, already rebased
/// to the module's assigned base), with `link` applied: guest-bound IAT slots are pre-filled in
/// the guest image and left out of the import bridge, redirected imports are resolved under
/// their forwarded name, and the bridge is rendered in `link.bridge_namespace`.
/// `options.pe_path` only names the input in messages.
pub fn run_module(
    options: &PipelineOptions,
    input: &Input,
    link: &ModuleLink,
    progress: &mut dyn FnMut(&str),
) -> Result<PipelineReport, PipelineError> {
    check_options(options)?;
    let pe = Pe64::parse(&input.bytes).map_err(|e| {
        PipelineError::Parse(format!(
            "failed to parse PE64 '{}': {e}",
            options.pe_path.display()
        ))
    })?;
    progress("parse");
    let prefilled = prefill_iat(&pe, &input.bytes, &link.guest_imports)?;
    let image_bytes = prefilled.as_deref().unwrap_or(&input.bytes);
    let guest =
        image::build_guest_image(&pe, image_bytes, options.image).map_err(PipelineError::Image)?;
    progress("image");
    let tool = if options.emit_obj {
        Some(find_llvm_tool().ok_or(PipelineError::NoLlvmTool)?)
    } else {
        None
    };
    let (includes, catalogue, mut plan) =
        plan_imports(&pe, &options.shim_headers, options.include_root.as_deref())?;
    apply_link(&mut plan, &catalogue, link)?;

    let mut seeded = seed(&pe, &input.bytes);
    seeded
        .image
        .entry_points
        .extend(options.extra_entries.iter().copied());
    seeded.image.entry_points.sort_unstable();
    seeded.image.entry_points.dedup();
    let cache = options.discovery_cache.as_ref().map(|dir| {
        discovery_cache::DiscoveryCache::new(dir, &options.module, options.verify_discovery_cache)
    });
    let cache_key = cache.as_ref().map(|_| {
        discovery_cache::stage_key(
            &input.bytes,
            prefilled.as_deref(),
            &options.image,
            &seeded.image,
        )
    });
    let mut cache_outcomes = Vec::new();
    let mut refined = |image: &mut CodeImage<'_>, key: Option<u128>| match (&cache, key) {
        (Some(cache), Some(key)) => {
            let (analysis, code_pointers, outcome) = cache
                .discover_refined(key, image, &guest.image)
                .map_err(PipelineError::Options)?;
            cache_outcomes.push(outcome.as_str());
            Ok::<_, PipelineError>((analysis, code_pointers))
        }
        _ => Ok(refine::discover_refined(image, &guest.image)),
    };
    let (mut analysis, mut code_pointers) = refined(&mut seeded.image, cache_key)?;
    if let Some(seeds) = &options.trace_seeds {
        let outcome = crate::trace_seed::evaluate(&seeded.image, &analysis, seeds);
        let accepted = outcome.accepted();
        if let Some(path) = &options.trace_seed_report
            && outcome.pairs_in_range > 0
        {
            use std::io::Write as _;
            let line = crate::trace_seed::report_line(&options.module, &outcome);
            let written = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut f| writeln!(f, "{line}"));
            written.map_err(io_error(format!(
                "failed to write seed report '{}'",
                path.display()
            )))?;
        }
        if !accepted.is_empty() {
            let key = cache_key.map(|k| discovery_cache::seeded_key(k, &accepted));
            seeded.image.entry_points.extend(accepted);
            seeded.image.entry_points.sort_unstable();
            seeded.image.entry_points.dedup();
            (analysis, code_pointers) = refined(&mut seeded.image, key)?;
        }
    }
    drop(refined);
    if !cache_outcomes.is_empty() {
        progress(&format!("discovery cache: {};", cache_outcomes.join(", ")));
    }
    let discovery = DiscoveryStats::new(&seeded.image, &analysis, code_pointers.rounds);
    let pdata = PdataStats::new(&seeded.pdata, &seeded.image);
    let slots = refine::check_import_slots(&pe, &plan, &analysis);
    progress("discover");

    let out = &options.module_dir();
    for dir in [LIFT_DIR, STUBS_DIR, IMAGE_DIR] {
        let path = out.join(dir);
        fs::create_dir_all(&path).map_err(io_error(format!(
            "failed to create output directory '{}'",
            path.display()
        )))?;
    }
    clean_lift_dir(&out.join(LIFT_DIR))?;

    // Lift.
    let functions: Vec<&Function> = analysis.functions.values().collect();
    let weights: Vec<usize> = functions.iter().map(|f| f.instruction_count()).collect();
    let entries: Vec<u64> = functions.iter().map(|f| f.entry).collect();
    let shard_ranges = shard_lift::plan_shards_stable(
        &entries,
        &weights,
        options.shard_size,
        options.shard_instructions,
    );
    let cache = options.lift_cache.as_ref().map(shard_lift::ShardCache::new);
    let (shards, hits) = lift_shards_cached(
        &options.target,
        &options.memory,
        &options.module,
        options.indirect_dispatch,
        &options.probe_functions,
        cache.as_ref(),
        &functions,
        &shard_ranges,
        out,
        tool.as_ref(),
        options.jobs,
    )?;
    if cache.is_some() {
        let reused: Vec<&ShardRecord> = shards
            .iter()
            .zip(&hits)
            .filter(|(_, h)| **h)
            .map(|(s, _)| s)
            .collect();
        let reused_functions: usize = reused.iter().map(|s| s.functions).sum();
        progress(&format!(
            "lift cache: {}/{} shards reused, {} of {} functions lifted;",
            reused.len(),
            shards.len(),
            functions.len() - reused_functions,
            functions.len()
        ));
    }
    progress("lift");
    let entries: Vec<u64> = analysis.functions.keys().copied().collect();
    let registration_name = format!("{REGISTRATION_STEM}.ll");
    let registration_path = out.join(LIFT_DIR).join(&registration_name);
    {
        let context = format!("failed to write '{}'", registration_path.display());
        let file = fs::File::create(&registration_path).map_err(io_error(context.clone()))?;
        let mut writer = BufWriter::new(file);
        match options.indirect_dispatch {
            IndirectDispatch::Table => {
                write_registration_table(&options.module, &entries, &mut writer)
            }
            IndirectDispatch::Shard => {
                let mut shard_entries: Vec<(u64, u32, u32)> = Vec::with_capacity(functions.len());
                for (shard_idx, range) in shard_ranges.iter().enumerate() {
                    for (local_idx, func) in functions[range.clone()].iter().enumerate() {
                        shard_entries.push((func.entry, shard_idx as u32, local_idx as u32));
                    }
                }
                write_shard_registration_table(
                    &options.module,
                    shard_ranges.len(),
                    &shard_entries,
                    &mut writer,
                )
            }
        }
        .and_then(|()| writer.flush())
        .map_err(io_error(context.clone()))?;
        drop(writer);
        if options.target.is_native() {
            let text = fs::read_to_string(&registration_path).map_err(io_error(context.clone()))?;
            fs::write(&registration_path, options.target.retarget(text))
                .map_err(io_error(context))?;
        }
    }
    let (size, hash) = hash_file(&registration_path)?;
    let registration_ll = OutputFile {
        path: format!("{LIFT_DIR}/{registration_name}"),
        size,
        hash,
    };
    let registration_obj = tool
        .as_ref()
        .map(|tool| {
            compile(
                tool,
                &options.target,
                &out.join(LIFT_DIR),
                &registration_name,
                &format!("{REGISTRATION_STEM}.o"),
            )
        })
        .transpose()?;
    progress("registration");
    let mut unsupported = BTreeMap::new();
    for shard in &shards {
        for (mnemonic, count) in &shard.unsupported {
            *unsupported.entry(mnemonic.clone()).or_insert(0) += count;
        }
    }
    let lift = LiftStats {
        shard_size: options.shard_size,
        shard_instructions: options.shard_instructions,
        functions: shards.iter().map(|s| s.functions).sum(),
        instructions: shards.iter().map(|s| s.instructions).sum(),
        shards,
        unsupported,
        registration_ll,
        registration_obj,
    };
    drop(functions);
    drop(analysis);

    // Import bridge.
    // The thunks translate (and, for the translated window, marshal) guest pointers the way the
    // lifted code addresses guest memory.
    let bridge = stubs::render_bridge_for(
        &plan,
        &catalogue,
        link.bridge_namespace.as_deref(),
        &options.memory,
    );
    let mut files = Vec::new();
    for (name, text) in [
        (stubs::HEADER_FILE, &bridge.header),
        (stubs::SOURCE_FILE, &bridge.source),
        (stubs::SEAM_FILE, &bridge.seam),
        (stubs::MANIFEST_FILE, &bridge.manifest),
    ] {
        files.push(write_file(
            out,
            &format!("{STUBS_DIR}/{name}"),
            text.as_bytes(),
        )?);
    }
    progress("stubs");
    let imports = ImportStats {
        total: plan.imports.len(),
        bound: plan.bound_count(),
        guest: plan.guest_count(),
        unresolved: plan.unresolved_count(),
        by_dll: plan.coverage_by_dll(),
        headers: includes,
        shims: catalogue.len(),
        catalogue_diagnostics: catalogue.diagnostics.len(),
    };

    // Guest image.
    let container = image::write_image(&guest.image).map_err(PipelineError::Image)?;
    let image_manifest = image::render_manifest_json(&guest.image, &guest.diagnostics)
        .map_err(PipelineError::Image)?;
    files.push(write_file(
        out,
        &format!("{IMAGE_DIR}/{IMAGE_FILE}"),
        &container,
    )?);
    files.push(write_file(
        out,
        &format!("{IMAGE_DIR}/{IMAGE_MANIFEST_FILE}"),
        image_manifest.as_bytes(),
    )?);
    let image_stats = ImageStats::new(&guest, container.len() as u64);
    drop(container);
    progress("pack");

    for shard in &lift.shards {
        files.push(shard.ll.clone());
        files.extend(shard.obj.clone());
    }
    files.push(lift.registration_ll.clone());
    files.extend(lift.registration_obj.clone());

    let mut report = PipelineReport {
        module: options.module.clone(),
        input_size: input.bytes.len() as u64,
        input_hash: input.hash,
        image: image_stats,
        pdata,
        discovery,
        no_return: seeded.no_return,
        code_pointers,
        pdata_present: !seeded.image.function_table.is_empty(),
        slots,
        imports,
        lift,
        llvm: tool.as_ref().map(tool_info),
        extra_entries: {
            let mut extra = options.extra_entries.clone();
            extra.sort_unstable();
            extra.dedup();
            extra
        },
        emit_obj: options.emit_obj,
        image_options: options.image,
        target: options.target.clone(),
        memory: options.memory,
        indirect_dispatch: options.indirect_dispatch,
        files: Vec::new(),
        function_entries: entries,
        report: String::new(),
        manifest: String::new(),
    };
    report.report = render_report(&report);
    files.push(write_file(out, REPORT_FILE, report.report.as_bytes())?);
    files.sort();
    report.files = files;
    report.manifest = render_manifest(&report);
    write_file(out, MANIFEST_FILE, report.manifest.as_bytes())?;
    progress("manifest");
    Ok(report)
}

// ---------------------------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------------------------

fn percent(part: usize, whole: usize) -> String {
    if whole == 0 {
        "n/a".to_string()
    } else {
        format!("{:.2}%", part as f64 * 100.0 / whole as f64)
    }
}

fn plural(n: usize, word: &str) -> String {
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
}

/// Renders `report.txt`: how close the binary is to running (names, addresses, counts only).
pub fn render_report(r: &PipelineReport) -> String {
    let mut out = String::new();
    let d = &r.discovery;
    let l = &r.lift;
    let unsupported = l.unsupported_total();
    let _ = writeln!(out, "playgta6-recomp build report: module {}", r.module);
    let _ = writeln!(
        out,
        "input: {} bytes, fnv1a64 0x{:016x}",
        r.input_size, r.input_hash
    );
    let _ = writeln!(
        out,
        "image: base 0x{:x}, entry 0x{:x}, size 0x{:x}, {}, {} dir64 + {} highlow base relocations",
        r.image.image_base,
        r.image.entry_point,
        r.image.size_of_image,
        plural(r.image.sections.len(), "section"),
        r.image.dir64,
        r.image.highlow
    );
    let _ = writeln!(
        out,
        ".pdata: {} records, {} valid, {} functions, {} diagnostics",
        r.pdata.records,
        r.pdata.valid,
        r.pdata.functions,
        r.pdata.diagnostics.values().sum::<usize>()
    );
    let _ = writeln!(
        out,
        "discovery: {} ({} blocks, {} instructions) from {} entry points in {}",
        plural(d.functions, "function"),
        d.blocks,
        d.instructions,
        d.entry_points,
        plural(d.rounds, "round"),
    );

    let _ = writeln!(out);
    let _ = writeln!(out, "coverage:");
    let _ = writeln!(
        out,
        "  lifted:      {} of {} instructions ({})",
        l.instructions - unsupported,
        l.instructions,
        percent(l.instructions - unsupported, l.instructions)
    );
    let _ = writeln!(
        out,
        "  unsupported: {unsupported} instructions trap at run time ({})",
        percent(unsupported, l.instructions)
    );
    let _ = writeln!(
        out,
        "  blocks ending in a guest trap: {}, in invalid/unmapped code: {}, in a no-return call: {}",
        d.trap_blocks, d.invalid_blocks, d.no_return_calls
    );
    let _ = writeln!(
        out,
        "  jump tables: {} resolved ({} with an inferred extent), {} unresolved indirect jumps \
         (dispatched at run time)",
        d.jump_tables_resolved, d.jump_tables_inferred, d.jump_tables_unresolved
    );
    if d.jump_tables_unresolved > 0 {
        let reasons: Vec<String> = d
            .jump_tables_unresolved_by_reason
            .iter()
            .filter(|(_, n)| *n > 0)
            .map(|(name, n)| format!("{name} {n}"))
            .collect();
        let _ = writeln!(out, "    unresolved by reason: {}", reasons.join(", "));
    }
    if !l.unsupported.is_empty() {
        let mut rows: Vec<(&String, &usize)> = l.unsupported.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let _ = writeln!(
            out,
            "  top unsupported mnemonics ({} distinct):",
            rows.len()
        );
        for (mnemonic, count) in rows.iter().take(REPORT_TOP_MNEMONICS) {
            let _ = writeln!(out, "    {mnemonic:<16} {count:>8}");
        }
    }

    let _ = writeln!(out);
    let guest = if r.imports.guest > 0 {
        format!(", {} bound to guest modules", r.imports.guest)
    } else {
        String::new()
    };
    let _ = writeln!(
        out,
        "imports: {} total, {} bound, {} unresolved{guest} ({} from {})",
        r.imports.total,
        r.imports.bound,
        r.imports.unresolved,
        plural(r.imports.shims, "shim"),
        plural(r.imports.headers.len(), "header")
    );
    if !r.imports.by_dll.is_empty() {
        let width = r
            .imports
            .by_dll
            .iter()
            .map(|(dll, _, _)| dll.len())
            .max()
            .unwrap_or(3)
            .max(3);
        let _ = writeln!(
            out,
            "  {:<width$}  {:>6}  {:>10}",
            "DLL", "bound", "unresolved"
        );
        for (dll, bound, unresolved) in &r.imports.by_dll {
            let _ = writeln!(out, "  {dll:<width$}  {bound:>6}  {unresolved:>10}");
        }
    }
    let s = &r.slots;
    let _ = writeln!(
        out,
        "  called but unresolved: {} (trap on first call)",
        s.referenced_unresolved.len()
    );
    list(
        &mut out,
        s.referenced_unresolved
            .iter()
            .map(|(_, label, why)| format!("{label} ({why})")),
    );
    let _ = writeln!(
        out,
        "  never referenced by discovered code: {} ({} bound)",
        s.unreferenced.len(),
        s.unreferenced.iter().filter(|(_, bound)| *bound).count()
    );
    let _ = writeln!(
        out,
        "  slot reads: {} distinct, {} IAT, {} other (function-pointer variables), {} misaligned IAT reads",
        s.slot_reads,
        s.iat_reads,
        s.non_iat_reads.len(),
        s.misaligned.len()
    );
    list(&mut out, s.misaligned.iter().map(|va| format!("0x{va:x}")));

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "refinements:\n  no-return: {}, {}",
        plural(r.no_return.len(), "import"),
        plural(r.code_pointers.no_return_thunks.len(), "thunk")
    );
    list(
        &mut out,
        r.no_return
            .iter()
            .map(|n| format!("{} (slot 0x{:x})", n.label, n.slot)),
    );
    let cp = &r.code_pointers;
    let _ = writeln!(
        out,
        "  code pointers ({} .pdata): {} candidates",
        if r.pdata_present { "with" } else { "without" },
        cp.pointers.len()
    );
    for source in PointerSource::ALL {
        let counts: Vec<String> = PointerStatus::ALL
            .iter()
            .map(|&status| (status, cp.count(source, status)))
            .filter(|(_, n)| *n > 0)
            .map(|(status, n)| format!("{} {n}", status.as_str()))
            .collect();
        let _ = writeln!(
            out,
            "    {:<10} {}",
            source.as_str(),
            if counts.is_empty() {
                "none".to_string()
            } else {
                counts.join(", ")
            }
        );
    }
    let not_lifted: Vec<_> = cp.not_lifted().collect();
    let _ = writeln!(
        out,
        "  dispatcher: {} registered functions; {} address-taken code addresses not lifted",
        d.functions,
        not_lifted.len()
    );
    list(
        &mut out,
        not_lifted
            .iter()
            .map(|p| format!("0x{:x} {} ({})", p.va, sources(p), p.status.as_str())),
    );

    let _ = writeln!(out);
    let _ = writeln!(out, "diagnostics:");
    for (kind, count) in &d.diagnostics {
        if *count > 0 {
            let _ = writeln!(out, "  discovery.{kind:<28} {count:>8}");
        }
    }
    for (kind, count) in &r.pdata.diagnostics {
        let _ = writeln!(out, "  pdata.{kind:<32} {count:>8}");
    }
    for (kind, count) in &r.image.diagnostics {
        let _ = writeln!(out, "  image.{kind:<32} {count:>8}");
    }
    if r.imports.catalogue_diagnostics > 0 {
        let _ = writeln!(
            out,
            "  catalogue.{:<28} {:>8}",
            "notes", r.imports.catalogue_diagnostics
        );
    }
    let _ = writeln!(
        out,
        "  total: {} discovery, {} pdata, {} image, {} catalogue",
        d.diagnostic_total(),
        r.pdata.diagnostics.values().sum::<usize>(),
        r.image.diagnostics.values().sum::<usize>(),
        r.imports.catalogue_diagnostics
    );

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "output: {} in {} (up to {} functions / {} instructions each), registration table of {} functions, objects {}",
        plural(l.functions, "function"),
        plural(l.shards.len(), "shard"),
        l.shard_size,
        l.shard_instructions,
        d.functions,
        match &r.llvm {
            Some(tool) => format!("built with {} ({})", tool.name, tool.version),
            None => "not built".to_string(),
        }
    );
    out
}

fn sources(p: &refine::CodePointer) -> String {
    p.sources
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join("+")
}

/// Appends up to [`REPORT_LIST_LIMIT`] indented items.
fn list(out: &mut String, items: impl ExactSizeIterator<Item = String>) {
    let total = items.len();
    for item in items.take(REPORT_LIST_LIMIT) {
        let _ = writeln!(out, "    {item}");
    }
    if total > REPORT_LIST_LIMIT {
        let _ = writeln!(
            out,
            "    ... {} more (see {MANIFEST_FILE})",
            total - REPORT_LIST_LIMIT
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------

/// Minimal JSON value with deterministic pretty-printing: containers holding only scalars are
/// written on one line, everything else one child per line.
enum Json {
    Null,
    Bool(bool),
    Num(u64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

fn hex(v: u64) -> Json {
    Json::Str(format!("0x{v:x}"))
}

fn obj<const N: usize>(fields: [(&str, Json); N]) -> Json {
    Json::Obj(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

fn num(n: usize) -> Json {
    Json::Num(n as u64)
}

fn counts<'a>(items: impl IntoIterator<Item = (&'a str, usize)>) -> Json {
    Json::Obj(
        items
            .into_iter()
            .map(|(k, n)| (k.to_string(), num(n)))
            .collect(),
    )
}

fn file_json(file: &OutputFile) -> Json {
    obj([
        ("path", Json::Str(file.path.clone())),
        ("size", Json::Num(file.size)),
        ("fnv1a64", Json::Str(format!("0x{:016x}", file.hash))),
    ])
}

/// The manifest record of a memory model; `None` for the default (manifests stay unchanged).
fn memory_model_json(model: &MemoryModel) -> Option<Json> {
    if model.is_default() {
        return None;
    }
    let flat = |base: &GuestBase| match base {
        GuestBase::Symbol => Json::Str(lift::GUEST_BASE_SYMBOL.to_string()),
        GuestBase::Constant(c) => hex(*c),
    };
    Some(match model {
        MemoryModel::Base(base) => obj([("model", Json::Str("base".into())), ("flat", flat(base))]),
        MemoryModel::Windowed(Window {
            flat: base,
            base: window_base,
            size,
            check,
            stack_flat,
            contiguous_runs,
        }) => obj([
            ("model", Json::Str("windowed".into())),
            ("flat", flat(base)),
            ("window_base", hex(*window_base)),
            ("window_size", hex(*size)),
            (
                "check",
                Json::Str(
                    match check {
                        WindowCheck::Branch => "branch",
                        WindowCheck::Select => "select",
                        WindowCheck::Outline => "outline",
                    }
                    .into(),
                ),
            ),
            ("stack_flat", Json::Bool(*stack_flat)),
            ("contiguous_runs", Json::Bool(*contiguous_runs)),
        ]),
        MemoryModel::PageTable {
            pages,
            stack,
            contiguous_runs,
        } => obj([
            ("model", Json::Str("page_table".into())),
            ("pages", Json::Num(*pages)),
            ("stack_flat", stack.as_ref().map_or(Json::Null, flat)),
            ("contiguous_runs", Json::Bool(*contiguous_runs)),
        ]),
    })
}

impl Json {
    fn is_scalar(&self) -> bool {
        !matches!(self, Self::Arr(_) | Self::Obj(_))
    }

    fn render(&self, out: &mut String, indent: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Num(n) => {
                let _ = write!(out, "{n}");
            }
            Self::Str(s) => json_string(out, s),
            Self::Arr(items) => {
                let flat = items.iter().all(Self::is_scalar);
                Self::container(out, indent, '[', ']', flat, items.len(), |out, i, ind| {
                    items[i].render(out, ind);
                });
            }
            Self::Obj(fields) => {
                let flat = fields.iter().all(|(_, v)| v.is_scalar());
                Self::container(out, indent, '{', '}', flat, fields.len(), |out, i, ind| {
                    json_string(out, &fields[i].0);
                    out.push_str(": ");
                    fields[i].1.render(out, ind);
                });
            }
        }
    }

    fn container(
        out: &mut String,
        indent: usize,
        open: char,
        close: char,
        flat: bool,
        len: usize,
        mut item: impl FnMut(&mut String, usize, usize),
    ) {
        out.push(open);
        if len == 0 {
            out.push(close);
            return;
        }
        for i in 0..len {
            if flat {
                if i > 0 {
                    out.push_str(", ");
                }
            } else {
                out.push_str(if i == 0 { "\n" } else { ",\n" });
                out.push_str(&" ".repeat(indent + 2));
            }
            item(out, i, indent + 2);
        }
        if !flat {
            out.push('\n');
            out.push_str(&" ".repeat(indent));
        }
        out.push(close);
    }
}

fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Renders `build-manifest.json` (schema in the module docs).
pub fn render_manifest(r: &PipelineReport) -> String {
    let d = &r.discovery;
    let l = &r.lift;
    let cp = &r.code_pointers;
    let s = &r.slots;
    let tool = obj([
        ("name", Json::Str(env!("CARGO_PKG_NAME").to_string())),
        ("version", Json::Str(env!("CARGO_PKG_VERSION").to_string())),
        (
            "llvm",
            r.llvm.as_ref().map_or(Json::Null, |t| {
                obj([
                    ("kind", Json::Str(t.kind.to_string())),
                    ("name", Json::Str(t.name.clone())),
                    ("version", Json::Str(t.version.clone())),
                ])
            }),
        ),
    ]);
    let options = obj([
        ("emit_obj", Json::Bool(r.emit_obj)),
        ("shard_size", num(l.shard_size)),
        ("shard_instructions", num(l.shard_instructions)),
        (
            "extra_entries",
            Json::Arr(r.extra_entries.iter().map(|&v| hex(v)).collect()),
        ),
        (
            "shim_headers",
            Json::Arr(
                r.imports
                    .headers
                    .iter()
                    .map(|h| Json::Str(h.clone()))
                    .collect(),
            ),
        ),
        ("map_headers", Json::Bool(r.image_options.map_headers)),
        (
            "executable_bytes",
            Json::Str(r.image_options.executable_bytes.as_str().to_string()),
        ),
    ]);
    // A native build records its target; wasm64 manifests stay exactly as before.
    let options = match (&r.target, options) {
        (LiftTarget::Native { triple, cpu }, Json::Obj(mut fields)) => {
            fields.push(("target".to_string(), Json::Str(format!("native:{triple}"))));
            fields.push(("target_cpu".to_string(), Json::Str(cpu.clone())));
            Json::Obj(fields)
        }
        (_, options) => options,
    };
    // So does another memory model than the default.
    let options = match (memory_model_json(&r.memory), options) {
        (Some(memory), Json::Obj(mut fields)) => {
            fields.push(("memory_model".to_string(), memory));
            Json::Obj(fields)
        }
        (_, options) => options,
    };
    // And non-default indirect-call dispatch mode.
    let options = match (r.indirect_dispatch, options) {
        (IndirectDispatch::Shard, Json::Obj(mut fields)) => {
            fields.push((
                "indirect_dispatch".to_string(),
                Json::Str(r.indirect_dispatch.as_str().to_string()),
            ));
            Json::Obj(fields)
        }
        (_, options) => options,
    };
    let image = obj([
        ("image_base", hex(r.image.image_base)),
        ("entry_point", hex(r.image.entry_point)),
        ("size_of_image", hex(u64::from(r.image.size_of_image))),
        (
            "sections",
            Json::Arr(
                r.image
                    .sections
                    .iter()
                    .map(|(name, rva, size, stored, protection)| {
                        obj([
                            ("name", Json::Str(name.clone())),
                            ("rva", hex(u64::from(*rva))),
                            ("virtual_size", hex(u64::from(*size))),
                            ("stored_bytes", Json::Num(*stored)),
                            ("protection", Json::Str((*protection).to_string())),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "base_relocations",
            obj([
                ("dir64", num(r.image.dir64)),
                ("highlow", num(r.image.highlow)),
            ]),
        ),
        ("container_bytes", Json::Num(r.image.container_bytes)),
    ]);
    let pdata = obj([
        ("records", num(r.pdata.records)),
        ("valid", num(r.pdata.valid)),
        ("functions", num(r.pdata.functions)),
        (
            "diagnostics",
            counts(r.pdata.diagnostics.iter().map(|(k, n)| (*k, *n))),
        ),
    ]);
    let discovery = obj([
        ("rounds", num(d.rounds)),
        ("entry_points", num(d.entry_points)),
        ("functions", num(d.functions)),
        ("blocks", num(d.blocks)),
        ("instructions", num(d.instructions)),
        (
            "jump_tables",
            obj([
                ("resolved", num(d.jump_tables_resolved)),
                ("unresolved", num(d.jump_tables_unresolved)),
                ("inferred", num(d.jump_tables_inferred)),
                (
                    "unresolved_by_reason",
                    Json::Obj(
                        d.jump_tables_unresolved_by_reason
                            .iter()
                            .map(|&(name, n)| (name.to_string(), num(n)))
                            .collect(),
                    ),
                ),
            ]),
        ),
        (
            "terminators",
            obj([
                ("trap", num(d.trap_blocks)),
                ("invalid", num(d.invalid_blocks)),
                ("no_return_call", num(d.no_return_calls)),
            ]),
        ),
        ("relocations", counts(d.relocations.iter().copied())),
        ("diagnostics", counts(d.diagnostics.iter().copied())),
    ]);
    let by_source = Json::Obj(
        PointerSource::ALL
            .iter()
            .map(|&source| {
                (
                    source.as_str().to_string(),
                    counts(
                        PointerStatus::ALL
                            .iter()
                            .map(|&status| (status.as_str(), cp.count(source, status))),
                    ),
                )
            })
            .collect(),
    );
    let refinements = obj([
        (
            "no_return",
            obj([
                (
                    "imports",
                    Json::Arr(
                        r.no_return
                            .iter()
                            .map(|n| {
                                obj([("label", Json::Str(n.label.clone())), ("slot", hex(n.slot))])
                            })
                            .collect(),
                    ),
                ),
                (
                    "thunks",
                    Json::Arr(cp.no_return_thunks.iter().map(|&v| hex(v)).collect()),
                ),
            ]),
        ),
        (
            "code_pointers",
            obj([
                ("pdata_present", Json::Bool(r.pdata_present)),
                ("round_limit_hit", Json::Bool(cp.round_limit_hit)),
                ("by_source", by_source),
                ("accepted", Json::Arr(cp.added_entries().map(hex).collect())),
            ]),
        ),
        (
            "dispatch",
            obj([
                ("registered_functions", num(d.functions)),
                (
                    "address_taken_not_lifted",
                    Json::Arr(
                        cp.not_lifted()
                            .map(|p| {
                                obj([
                                    ("va", hex(p.va)),
                                    ("sources", Json::Str(sources(p))),
                                    ("status", Json::Str(p.status.as_str().to_string())),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ),
        (
            "import_slots",
            obj([
                ("slot_reads", num(s.slot_reads)),
                ("iat_reads", num(s.iat_reads)),
                (
                    "non_iat_reads",
                    Json::Arr(s.non_iat_reads.iter().map(|&v| hex(v)).collect()),
                ),
                (
                    "misaligned",
                    Json::Arr(s.misaligned.iter().map(|&v| hex(v)).collect()),
                ),
                (
                    "referenced_unresolved",
                    Json::Arr(
                        s.referenced_unresolved
                            .iter()
                            .map(|(dll, label, why)| {
                                obj([
                                    ("dll", Json::Str(dll.clone())),
                                    ("import", Json::Str(label.clone())),
                                    ("reason", Json::Str((*why).to_string())),
                                ])
                            })
                            .collect(),
                    ),
                ),
                (
                    "unreferenced",
                    Json::Arr(
                        s.unreferenced
                            .iter()
                            .map(|(label, bound)| {
                                obj([
                                    ("import", Json::Str(label.clone())),
                                    ("bound", Json::Bool(*bound)),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ),
    ]);
    let opt_file =
        |f: &Option<OutputFile>| f.as_ref().map_or(Json::Null, |f| Json::Str(f.path.clone()));
    let (table_symbol, count_symbol) = registration_symbols(&r.module);
    let lift = obj([
        ("functions", num(l.functions)),
        ("instructions", num(l.instructions)),
        ("unsupported_instructions", num(l.unsupported_total())),
        (
            "unsupported_by_mnemonic",
            counts(l.unsupported.iter().map(|(k, n)| (k.as_str(), *n))),
        ),
        (
            "shards",
            Json::Arr(
                l.shards
                    .iter()
                    .map(|shard| {
                        obj([
                            ("index", num(shard.index)),
                            ("first", hex(shard.first)),
                            ("last", hex(shard.last)),
                            ("functions", num(shard.functions)),
                            ("instructions", num(shard.instructions)),
                            ("ll", Json::Str(shard.ll.path.clone())),
                            ("obj", opt_file(&shard.obj)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "registration",
            {
                let reg = obj([
                    ("table_symbol", Json::Str(table_symbol)),
                    ("count_symbol", Json::Str(count_symbol)),
                    (
                        "register_function",
                        Json::Str(register_function_symbol(&r.module)),
                    ),
                    ("entry_type", Json::Str(REGISTRATION_ENTRY_TYPE.to_string())),
                    ("count", num(d.functions)),
                    ("ll", Json::Str(l.registration_ll.path.clone())),
                    ("obj", opt_file(&l.registration_obj)),
                ]);
                match (r.indirect_dispatch, reg) {
                    (IndirectDispatch::Shard, Json::Obj(mut fields)) => {
                        fields.push((
                            "shard_dispatch_symbol".to_string(),
                            Json::Str(shard_dispatch_table_symbol(&r.module)),
                        ));
                        Json::Obj(fields)
                    }
                    (_, reg) => reg,
                }
            },
        ),
    ]);
    let imports = obj([
        ("total", num(r.imports.total)),
        ("bound", num(r.imports.bound)),
        ("guest", num(r.imports.guest)),
        ("unresolved", num(r.imports.unresolved)),
        (
            "by_dll",
            Json::Arr(
                r.imports
                    .by_dll
                    .iter()
                    .map(|(dll, bound, unresolved)| {
                        obj([
                            ("dll", Json::Str(dll.clone())),
                            ("bound", num(*bound)),
                            ("unresolved", num(*unresolved)),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "catalogue",
            obj([
                ("headers", num(r.imports.headers.len())),
                ("shims", num(r.imports.shims)),
                ("diagnostics", num(r.imports.catalogue_diagnostics)),
            ]),
        ),
    ]);
    let diagnostics = obj([
        ("pdata", num(r.pdata.diagnostics.values().sum())),
        ("discovery", num(d.diagnostic_total())),
        ("image", num(r.image.diagnostics.values().sum())),
        ("catalogue", num(r.imports.catalogue_diagnostics)),
        ("unsupported_instructions", num(l.unsupported_total())),
    ]);
    let manifest = obj([
        ("schema", Json::Str(MANIFEST_SCHEMA.to_string())),
        (
            "schema_version",
            Json::Num(u64::from(MANIFEST_SCHEMA_VERSION)),
        ),
        ("module", Json::Str(r.module.clone())),
        ("tool", tool),
        (
            "input",
            obj([
                ("size", Json::Num(r.input_size)),
                ("fnv1a64", Json::Str(format!("0x{:016x}", r.input_hash))),
            ]),
        ),
        ("options", options),
        ("image", image),
        ("pdata", pdata),
        ("discovery", discovery),
        ("refinements", refinements),
        ("lift", lift),
        ("imports", imports),
        ("diagnostics", diagnostics),
        ("files", Json::Arr(r.files.iter().map(file_json).collect())),
    ]);
    let mut out = String::new();
    manifest.render(&mut out, 0);
    out.push('\n');
    out
}

#[cfg(test)]
mod tests;
