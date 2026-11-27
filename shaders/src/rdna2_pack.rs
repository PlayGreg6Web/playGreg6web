#![forbid(unsafe_code)]

//! Batch AMD RDNA2 (GFX10.3 / Xbox Series Scarlett) -> Naga-validated WGSL -> `WGPK` v1
//! `.wgsl.pack` archive builder.
//!
//! Reads raw RDNA2 shader blobs (`*.bin`, `*.rdna2`, `*.shader`), parses optional filename
//! stage/hash hints (e.g. `vs_<16hex>.rdna2`, `ps_<16hex>.rdna2`, `cs_<16hex>.rdna2`, or
//! `<16hex>.bin`), translates every shader unconditionally through
//! [`crate::rdna2_to_wgsl::translate_rdna2_to_wgsl`] (Naga-validated against baseline WebGPU),
//! and packs all translated shaders deterministically (sorted by ascending `u64` hash) into a
//! binary `WGPK` v1 `.wgsl.pack` via [`crate::pack::write_pack`].
//!
//! Guarantees:
//! * Zero `unsafe` code.
//! * Panic-free across empty, truncated, corrupt, or non-matching directory contents.
//! * Ignores non-files and symlinks that resolve outside the input directory.
//! * Strictly sanitized JSON and text summaries (zero host paths, zero usernames, zero raw bytes).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::{self, Write as _};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::binding_pack::{
    BindingPackStats, PackBindingLimitSummary, StageBindingLimits,
};
use crate::convert::{ShaderStage, fnv1a_64, format_shader_hash};
use crate::dxil_reflect::sanitize_source_path;
use crate::pack::{
    PACK_ENTRY_BYTES, PACK_ENTRY_BYTES_V2, PACK_HEADER_BYTES, PACK_VERSION_V1, PACK_VERSION_V2,
    PackEntry, PackError, PackVersion, ResolvedPackLayout, TocEntry,
    checked_table_data_start_for_version, encode_pack_toc, resolve_pack_layout, write_json_string,
    write_pack_with_layout_versioned,
};
use crate::reflect::{SidecarBuildOptions, SidecarStageInput, StageReflectionSidecar};
use crate::rdna2_to_wgsl::{
    FssmStageMetadata, Rdna2StageOverride, Rdna2TranslateOptions, Rdna2WgslOutput, WgslBindingInfo,
    WgslStageParamInfo, WgslVertexInputInfo, parse_fssm_stage_metadata, translate_rdna2_to_wgsl,
    write_mesh_output_json, write_wgsl_bindings_json, write_wgsl_stage_params_json,
    write_wgsl_vertex_inputs_json,
};
pub use crate::rdna2_to_wgsl::{MESH_COMPUTE_KEY_SALT, MeshOutputLayout, mesh_compute_pack_key};

/// File extensions recognized as raw RDNA2 shader blobs (ASCII case-insensitive).
pub const SUPPORTED_RDNA2_EXTENSIONS: &[&str] = &["bin", "rdna2", "shader"];

/// Minimum byte length of a valid RDNA2 shader blob (one 32-bit instruction dword).
pub const MIN_RDNA2_BLOB_BYTES: usize = 4;

/// Expected `"schema"` identifier in a `playgta6-pipeline-dump` `manifest.json`.
pub const PIPELINE_DUMP_SCHEMA: &str = "playgta6-pipeline-dump";

/// Supported `"schema_version"` in a `playgta6-pipeline-dump` `manifest.json`.
pub const PIPELINE_DUMP_SCHEMA_VERSION: u64 = 1;

/// Maximum nesting depth permitted by the internal zero-dependency JSON parser.
const MAX_JSON_DEPTH: usize = 64;

/// One entry in `manifest.json`'s `"native_stages"` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineDumpNativeStage {
    pub code_identity: u64,
    pub identity: Option<u64>,
    pub stage: ShaderStage,
    /// `true` when the manifest declared `"stage": "mesh"` / `"ms"` (packed as `ShaderStage::Compute`
    /// under `code_identity ^ MESH_COMPUTE_KEY_SALT`).
    pub is_mesh: bool,
    pub code_file: String,
    pub code_bytes: Option<u64>,
    /// Optional relative path to an Xbox Series `FSSM` `.meta` sidecar blob.
    pub metadata_file: Option<String>,
}

/// Parsed `playgta6-pipeline-dump` `manifest.json` (fields needed for RDNA2 packing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineDumpManifest {
    pub schema: String,
    pub schema_version: u64,
    pub native_stages: Vec<PipelineDumpNativeStage>,
}

/// Optional stage and 64-bit hash hints extracted from a shader blob filename.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rdna2FilenameHint {
    pub stage: Rdna2StageOverride,
    pub hash: Option<u64>,
}

/// A single in-memory raw RDNA2 shader blob passed to [`pack_rdna2_blobs`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2BlobInput {
    /// Sanitized label/filename used solely for optional stage/hash hints.
    pub name: String,
    /// Raw little-endian RDNA2 (GFX10.3) machine-code bytes.
    pub bytes: Vec<u8>,
    /// Explicit stage override; when `Auto`, filename hints and ISA inference apply.
    pub stage_override: Rdna2StageOverride,
    /// Explicit 64-bit hash override; when `None`, filename hash hints or FNV-1a apply.
    pub hash_override: Option<u64>,
    /// Optional parsed Xbox Series `FSSM` stage metadata (from a sibling `.meta` file).
    pub fssm_meta: Option<FssmStageMetadata>,
}

impl Rdna2BlobInput {
    /// Create a blob input with default `Auto` stage and hash resolution from filename/bytes.
    pub fn new(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            bytes,
            stage_override: Rdna2StageOverride::Auto,
            hash_override: None,
            fssm_meta: None,
        }
    }

    /// Set an explicit stage override on this blob input.
    pub fn with_stage(mut self, stage: Rdna2StageOverride) -> Self {
        self.stage_override = stage;
        self
    }

    /// Set an explicit 64-bit hash override on this blob input.
    pub fn with_hash(mut self, hash: u64) -> Self {
        self.hash_override = Some(hash);
        self
    }

    /// Attach parsed Xbox Series `FSSM` stage metadata to this blob input.
    pub fn with_fssm_meta(mut self, fssm_meta: FssmStageMetadata) -> Self {
        self.fssm_meta = Some(fssm_meta);
        self
    }
}

/// Candidate selection order when `max_pack_bytes` limits how many translated shaders fit
/// inside a single `WGPK` v1 archive.
///
/// Regardless of selection order, the written `WGPK` v1 directory table and `summary.entries`
/// are always canonically sorted by ascending 64-bit hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BudgetOrder {
    /// Legacy ascending 64-bit hash order (default for full backwards compatibility).
    #[default]
    HashOrder,
    /// Select smallest translated WGSL entries first (`(wgsl_byte_len asc, hash asc)`),
    /// maximizing packed shader count across the full `0x0000..0xffff` hash space.
    SmallestFirst,
    /// Preserve source discovery order (`manifest.json` `native_stages` appearance order first,
    /// then directory/input order), prioritizing early boot/frame shaders regardless of hash.
    ManifestOrder,
}

impl BudgetOrder {
    /// Canonical lowercase CLI/JSON identifier for this budget ordering.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HashOrder => "hash-order",
            Self::SmallestFirst => "smallest-first",
            Self::ManifestOrder => "manifest-order",
        }
    }

    /// Parse a CLI `--budget-order` value (`hash-order`, `smallest-first`, or `manifest-order`).
    pub fn parse_str(value: &str) -> Result<Self, String> {
        if value.eq_ignore_ascii_case("hash-order") {
            Ok(Self::HashOrder)
        } else if value.eq_ignore_ascii_case("smallest-first") {
            Ok(Self::SmallestFirst)
        } else if value.eq_ignore_ascii_case("manifest-order") {
            Ok(Self::ManifestOrder)
        } else {
            Err(format!(
                "invalid value for --budget-order: '{value}' (expected 'hash-order', 'smallest-first', or 'manifest-order')"
            ))
        }
    }
}

impl fmt::Display for BudgetOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Options controlling batch RDNA2 -> WGSL -> `WGPK` packing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rdna2PackOptions {
    /// Default RDNA2 -> WGSL translation options (per-blob overrides/hints take precedence over `Auto`).
    pub translate: Rdna2TranslateOptions,
    /// `WGPK` header flags word (default `0`).
    pub pack_flags: u32,
    /// Target `WGPK` binary container version policy (`Auto` selects v1 when the archive fits in
    /// `u32::MAX` bytes and automatically upgrades to v2 when >4 GiB; `V1` forces v1; `V2` forces v2).
    pub pack_version: PackVersion,
    /// When `true`, skip individual shaders that fail I/O, hash verification, size checks, or
    /// translation/Naga validation and record them in `summary.failures` instead of aborting.
    pub best_effort: bool,
    /// Optional per-stage WGSL UTF-8 byte limit. When `Some(limit)`, any translated stage whose
    /// WGSL UTF-8 byte length exceeds `limit` is skipped cleanly instead of packed and counted in
    /// `summary.skipped_oversize`.
    pub max_wgsl_bytes: Option<usize>,
    /// Optional total `.wgsl.pack` byte budget (capped at `u32::MAX` when `pack_version == V1`;
    /// unrestricted 64-bit budget for `Auto` / `V2`). When `Some(limit)`, select translated stages
    /// according to `budget_order` so total output file size (header + entry table + payloads)
    /// stays within the effective budget, counting omitted stages in `summary.skipped_budget` /
    /// `summary.omitted_by_budget`.
    pub max_pack_bytes: Option<u64>,
    /// Candidate selection ordering applied when `max_pack_bytes` limits how many translated
    /// stages fit in the `WGPK` archive. Defaults to [`BudgetOrder::HashOrder`].
    pub budget_order: BudgetOrder,
    /// Optional ordered list of 64-bit shader hashes (`hash_fnv1a64`) that are always prioritized
    /// ahead of `budget_order` when `max_pack_bytes` limits how many translated stages fit in the
    /// `WGPK` archive. Duplicate hashes preserve their first-seen priority order.
    pub priority_hashes: Vec<u64>,
    /// Optional worker thread count for parallel RDNA2 -> WGSL stage translation.
    /// When `None`, defaults to `std::thread::available_parallelism()` (falling back to `1`).
    /// Values `< 1` are clamped to `1`. Use `Some(1)` for single-threaded execution.
    pub jobs: Option<usize>,
}

impl Rdna2PackOptions {
    /// Resolve the effective worker thread count (`>= 1`).
    pub fn effective_jobs(&self) -> usize {
        self.jobs
            .unwrap_or_else(|| {
                thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            })
            .max(1)
    }
}

/// Strictly sanitized per-shader entry summary for [`Rdna2PackSummary`]
/// (zero host paths, zero filenames, zero raw bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2PackEntrySummary {
    pub hash_fnv1a64: u64,
    pub stage: ShaderStage,
    pub entry_point: String,
    pub naga_validated: bool,
    pub translated_op_count: usize,
    pub unsupported_op_count: usize,
    pub wgsl_byte_len: usize,
    pub vertex_input_mode: String,
    pub bindings: Vec<WgslBindingInfo>,
    pub vertex_inputs: Vec<WgslVertexInputInfo>,
    pub vs_outputs: Vec<WgslStageParamInfo>,
    pub ps_inputs: Vec<WgslStageParamInfo>,
    pub mesh_output: Option<MeshOutputLayout>,
    pub rt_arena: bool,
    pub rt_prelude_version: Option<u32>,
    pub bvh_ops: u32,
    pub raw_address: bool,
    pub atomic64: bool,
    pub binding_pack: BindingPackStats,
}

/// Strictly sanitized record of an individual shader that failed packing (zero host paths,
/// zero usernames, zero raw bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2PackFailureSummary {
    pub code_identity: Option<u64>,
    pub stage: Option<ShaderStage>,
    pub reason: String,
}

/// Strictly sanitized summary of a batch RDNA2 -> WGSL -> `WGPK` pack operation.
///
/// Contains only deterministic aggregate and per-entry metrics: zero host paths,
/// zero usernames, zero filenames, and zero raw shader bytes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rdna2PackSummary {
    pub pack_version: u32,
    pub total_blobs: usize,
    pub translated_ok: usize,
    pub skipped_or_failed: usize,
    pub failed: usize,
    pub manifest_duplicates_skipped: usize,
    pub budget_order: BudgetOrder,
    pub skipped_oversize: usize,
    pub skipped_budget: usize,
    pub omitted_by_budget: usize,
    pub priority_hashes_requested: usize,
    pub priority_hashes_packed: usize,
    pub largest_wgsl_bytes: usize,
    pub total_wgsl_bytes: u64,
    pub vertex_count: usize,
    pub pixel_count: usize,
    pub compute_count: usize,
    pub vertex_packed: usize,
    pub fragment_packed: usize,
    pub compute_packed: usize,
    pub pack_bytes: usize,
    pub binding_limits: PackBindingLimitSummary,
    pub entries: Vec<Rdna2PackEntrySummary>,
    pub failures: Vec<Rdna2PackFailureSummary>,
}

impl Rdna2PackSummary {
    fn resolved_pack_version(&self) -> u32 {
        if self.pack_version == PACK_VERSION_V2 {
            PACK_VERSION_V2
        } else {
            PACK_VERSION_V1
        }
    }

    /// Render a strictly sanitized deterministic JSON summary.
    pub fn to_json(&self) -> String {
        let format_tag = if self.resolved_pack_version() == PACK_VERSION_V2 {
            "wgpk_v2"
        } else {
            "wgpk_v1"
        };
        let mut out = String::with_capacity(512);
        out.push_str("{\n");
        let _ = writeln!(out, "  \"format\": \"{format_tag}\",");
        out.push_str("  \"isa\": \"rdna2_gfx10_3\",\n");
        let _ = writeln!(out, "  \"total_blobs\": {},", self.total_blobs);
        let _ = writeln!(out, "  \"translated_ok\": {},", self.translated_ok);
        let _ = writeln!(out, "  \"skipped_or_failed\": {},", self.skipped_or_failed);
        let _ = writeln!(out, "  \"failed\": {},", self.failed);
        let _ = writeln!(
            out,
            "  \"manifest_duplicates_skipped\": {},",
            self.manifest_duplicates_skipped
        );
        out.push_str("  \"budget_order\": ");
        write_json_string(&mut out, self.budget_order.as_str());
        out.push_str(",\n");
        let _ = writeln!(out, "  \"skipped_oversize\": {},", self.skipped_oversize);
        let _ = writeln!(out, "  \"skipped_budget\": {},", self.skipped_budget);
        let _ = writeln!(out, "  \"omitted_by_budget\": {},", self.omitted_by_budget);
        let _ = writeln!(
            out,
            "  \"priority_hashes_requested\": {},",
            self.priority_hashes_requested
        );
        let _ = writeln!(
            out,
            "  \"priority_hashes_packed\": {},",
            self.priority_hashes_packed
        );
        let _ = writeln!(
            out,
            "  \"largest_wgsl_bytes\": {},",
            self.largest_wgsl_bytes
        );
        let _ = writeln!(out, "  \"total_wgsl_bytes\": {},", self.total_wgsl_bytes);
        let _ = writeln!(out, "  \"vertex_count\": {},", self.vertex_count);
        let _ = writeln!(out, "  \"pixel_count\": {},", self.pixel_count);
        let _ = writeln!(out, "  \"compute_count\": {},", self.compute_count);
        let _ = writeln!(out, "  \"vertex_packed\": {},", self.vertex_packed);
        let _ = writeln!(out, "  \"fragment_packed\": {},", self.fragment_packed);
        let _ = writeln!(out, "  \"compute_packed\": {},", self.compute_packed);
        let _ = writeln!(out, "  \"pack_bytes\": {},", self.pack_bytes);
        out.push_str("  \"binding_limits\": ");
        self.binding_limits.write_json(&mut out, "  ");
        out.push_str(",\n");
        if self.entries.is_empty() {
            out.push_str("  \"entries\": [],\n");
        } else {
            out.push_str("  \"entries\": [\n");
            for (i, entry) in self.entries.iter().enumerate() {
                out.push_str("    {\n");
                out.push_str("      \"hash_fnv1a64\": ");
                write_json_string(&mut out, &format_shader_hash(entry.hash_fnv1a64));
                out.push_str(",\n");
                out.push_str("      \"stage\": ");
                write_json_string(&mut out, entry.stage.as_str());
                out.push_str(",\n");
                out.push_str("      \"entry_point\": ");
                write_json_string(&mut out, &entry.entry_point);
                out.push_str(",\n");
                let _ = writeln!(out, "      \"naga_validated\": {},", entry.naga_validated);
                let _ = writeln!(
                    out,
                    "      \"translated_op_count\": {},",
                    entry.translated_op_count
                );
                let _ = writeln!(
                    out,
                    "      \"unsupported_op_count\": {},",
                    entry.unsupported_op_count
                );
                let _ = writeln!(out, "      \"wgsl_byte_len\": {},", entry.wgsl_byte_len);
                out.push_str("      \"bindings\": ");
                write_wgsl_bindings_json(&mut out, &entry.bindings, "      ");
                out.push_str(",\n");
                out.push_str("      \"binding_summary\": {\"vertex_input_mode\": ");
                write_json_string(&mut out, &entry.vertex_input_mode);
                out.push_str("}");
                if !entry.vertex_inputs.is_empty() {
                    out.push_str(",\n      \"vertex_inputs\": ");
                    write_wgsl_vertex_inputs_json(&mut out, &entry.vertex_inputs, "      ");
                }
                if !entry.vs_outputs.is_empty() {
                    out.push_str(",\n      \"vs_outputs\": ");
                    write_wgsl_stage_params_json(&mut out, &entry.vs_outputs, "      ");
                }
                if !entry.ps_inputs.is_empty() {
                    out.push_str(",\n      \"ps_inputs\": ");
                    write_wgsl_stage_params_json(&mut out, &entry.ps_inputs, "      ");
                }
                if let Some(mesh_output) = entry.mesh_output {
                    out.push_str(",\n      \"mesh_output\": ");
                    write_mesh_output_json(&mut out, mesh_output);
                }
                if entry.rt_arena {
                    out.push_str(",\n      \"rt_arena\": true");
                }
                if let Some(ver) = entry.rt_prelude_version {
                    let _ = write!(out, ",\n      \"rt_prelude_version\": {ver}");
                }
                if entry.bvh_ops > 0 {
                    let _ = write!(out, ",\n      \"bvh_ops\": {}", entry.bvh_ops);
                }
                if entry.raw_address {
                    out.push_str(",\n      \"raw_address\": true");
                }
                if entry.atomic64 {
                    out.push_str(",\n      \"atomic64\": true");
                }
                out.push('\n');
                if i + 1 < self.entries.len() {
                    out.push_str("    },\n");
                } else {
                    out.push_str("    }\n");
                }
            }
            out.push_str("  ],\n");
        }
        if self.failures.is_empty() {
            out.push_str("  \"failures\": []\n");
        } else {
            out.push_str("  \"failures\": [\n");
            for (i, failure) in self.failures.iter().enumerate() {
                out.push_str("    {");
                if let Some(id) = failure.code_identity {
                    out.push_str("\"code_identity\": ");
                    write_json_string(&mut out, &format_shader_hash(id));
                    out.push_str(", ");
                }
                if let Some(stage) = failure.stage {
                    out.push_str("\"stage\": ");
                    write_json_string(&mut out, stage.as_str());
                    out.push_str(", ");
                }
                out.push_str("\"reason\": ");
                write_json_string(&mut out, &failure.reason);
                if i + 1 < self.failures.len() {
                    out.push_str("},\n");
                } else {
                    out.push_str("}\n");
                }
            }
            out.push_str("  ]\n");
        }
        out.push_str("}\n");
        out
    }

    /// Render a strictly sanitized deterministic plain-text summary.
    pub fn to_text(&self) -> String {
        let version = self.resolved_pack_version();
        let mut out = String::with_capacity(256);
        let _ = writeln!(out, "WGPK v{version} RDNA2 shader pack summary");
        let _ = writeln!(out, "  total_blobs:       {}", self.total_blobs);
        let _ = writeln!(out, "  translated_ok:     {}", self.translated_ok);
        let _ = writeln!(out, "  skipped_or_failed: {}", self.skipped_or_failed);
        let _ = writeln!(out, "  failed:            {}", self.failed);
        if self.manifest_duplicates_skipped > 0 {
            let _ = writeln!(
                out,
                "  manifest_dups:     {}",
                self.manifest_duplicates_skipped
            );
        }
        let _ = writeln!(out, "  budget_order:      {}", self.budget_order.as_str());
        let _ = writeln!(out, "  skipped_oversize:  {}", self.skipped_oversize);
        let _ = writeln!(out, "  skipped_budget:    {}", self.skipped_budget);
        let _ = writeln!(out, "  omitted_by_budget: {}", self.omitted_by_budget);
        if self.priority_hashes_requested > 0 {
            let _ = writeln!(
                out,
                "  priority_hashes_requested: {}",
                self.priority_hashes_requested
            );
            let _ = writeln!(
                out,
                "  priority_hashes_packed:    {}",
                self.priority_hashes_packed
            );
        }
        let _ = writeln!(out, "  vertex_count:      {}", self.vertex_count);
        let _ = writeln!(out, "  pixel_count:       {}", self.pixel_count);
        let _ = writeln!(out, "  compute_count:     {}", self.compute_count);
        let _ = writeln!(out, "  vertex_packed:     {}", self.vertex_packed);
        let _ = writeln!(out, "  fragment_packed:   {}", self.fragment_packed);
        let _ = writeln!(out, "  compute_packed:    {}", self.compute_packed);
        let _ = writeln!(out, "  largest_wgsl_bytes: {}", self.largest_wgsl_bytes);
        let _ = writeln!(out, "  total_wgsl_bytes:  {}", self.total_wgsl_bytes);
        let _ = writeln!(out, "  pack_bytes:        {}", self.pack_bytes);
        self.binding_limits.write_text(&mut out);
        if !self.failures.is_empty() {
            let _ = writeln!(out, "  failures ({}):", self.failures.len());
            for failure in &self.failures {
                let id_str = failure
                    .code_identity
                    .map(format_shader_hash)
                    .unwrap_or_else(|| "<unknown>".to_string());
                let stage_str = failure
                    .stage
                    .map(ShaderStage::as_str)
                    .unwrap_or("unknown");
                let _ = writeln!(out, "    - {id_str} ({stage_str}): {}", failure.reason);
            }
        }
        out
    }
}

/// Complete output of [`pack_rdna2_blobs`], [`pack_rdna2_directory`], or [`pack_rdna2_manifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2PackResult {
    /// Serialized `WGPK` v1 binary `.wgsl.pack` bytes.
    pub pack_bytes: Vec<u8>,
    /// Packed WGSL entries in deterministic ascending hash order.
    pub entries: Vec<PackEntry>,
    /// Strictly sanitized summary metrics.
    pub summary: Rdna2PackSummary,
}

/// Errors produced when packing RDNA2 shader directories, manifests, or blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rdna2PackError {
    InvalidInputDir(String),
    InvalidManifest(String),
    InvalidPriorityHashes(String),
    HashMismatch {
        expected: u64,
        actual: u64,
        stage: ShaderStage,
    },
    ShaderFailed {
        code_identity: Option<u64>,
        stage: Option<ShaderStage>,
        reason: String,
    },
    Io(String),
    Pack(PackError),
}

impl fmt::Display for Rdna2PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInputDir(msg) => write!(f, "invalid RDNA2 input directory: {msg}"),
            Self::InvalidManifest(msg) => write!(f, "invalid pipeline-dump manifest: {msg}"),
            Self::InvalidPriorityHashes(msg) => write!(f, "invalid priority hashes list: {msg}"),
            Self::HashMismatch {
                expected,
                actual,
                stage,
            } => write!(
                f,
                "RDNA2 {} shader FNV-1a 64 hash mismatch: expected {}, actual {}",
                stage.as_str(),
                format_shader_hash(*expected),
                format_shader_hash(*actual)
            ),
            Self::ShaderFailed {
                code_identity,
                stage,
                reason,
            } => {
                let id_str = code_identity
                    .map(format_shader_hash)
                    .unwrap_or_else(|| "<unknown>".to_string());
                let stage_str = stage.map(ShaderStage::as_str).unwrap_or("unknown");
                write!(f, "RDNA2 shader {id_str} ({stage_str}) failed: {reason}")
            }
            Self::Io(msg) => write!(f, "I/O error while packing RDNA2 shaders: {msg}"),
            Self::Pack(err) => write!(f, "failed to serialize WGPK pack: {err}"),
        }
    }
}

impl std::error::Error for Rdna2PackError {}

/// Returns `true` if `path` has a supported RDNA2 shader blob extension
/// (`bin`, `rdna2`, or `shader`, ASCII case-insensitive).
pub fn is_supported_rdna2_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            SUPPORTED_RDNA2_EXTENSIONS
                .iter()
                .any(|allowed| ext.eq_ignore_ascii_case(allowed))
        })
}

fn parse_stage_token(token: &str) -> Option<Rdna2StageOverride> {
    if token.eq_ignore_ascii_case("vs") || token.eq_ignore_ascii_case("vertex") {
        Some(Rdna2StageOverride::Vertex)
    } else if token.eq_ignore_ascii_case("ps")
        || token.eq_ignore_ascii_case("fs")
        || token.eq_ignore_ascii_case("pixel")
        || token.eq_ignore_ascii_case("fragment")
    {
        Some(Rdna2StageOverride::Pixel)
    } else if token.eq_ignore_ascii_case("cs") || token.eq_ignore_ascii_case("compute") {
        Some(Rdna2StageOverride::Compute)
    } else if token.eq_ignore_ascii_case("ms") || token.eq_ignore_ascii_case("mesh") {
        Some(Rdna2StageOverride::Mesh)
    } else {
        None
    }
}

fn parse_exact_hex_u64(token: &str) -> Option<u64> {
    let hex = token
        .strip_prefix("0x")
        .or_else(|| token.strip_prefix("0X"))
        .unwrap_or(token);
    if hex.len() != 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

/// Parse an ordered, deduplicated list of 64-bit shader hashes from newline-delimited text.
///
/// Each non-empty line (after stripping `#` comments and surrounding whitespace) must contain
/// a single 16-hex-digit 64-bit hash with an optional `0x` / `0X` prefix (ASCII
/// case-insensitive). Blank lines and `#` comment lines are ignored. Duplicate hashes preserve
/// their first-seen priority order.
pub fn parse_priority_hashes(text: &str) -> Result<Vec<u64>, Rdna2PackError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut hashes: Vec<u64> = Vec::new();
    let mut seen: BTreeSet<u64> = BTreeSet::new();

    for (idx, raw_line) in text.lines().enumerate() {
        let line_num = idx + 1;
        let without_comment = raw_line.split('#').next().unwrap_or("");
        let token = without_comment.trim();
        if token.is_empty() {
            continue;
        }
        let Some(hash) = parse_exact_hex_u64(token) else {
            let preview: String = token.chars().take(64).collect();
            return Err(Rdna2PackError::InvalidPriorityHashes(format!(
                "line {line_num}: invalid 64-bit hex hash '{preview}' (expected 16 hex digits with optional '0x' prefix)"
            )));
        };
        if seen.insert(hash) {
            hashes.push(hash);
        }
    }

    Ok(hashes)
}

/// Read and parse a `--priority-hashes` file from disk via [`parse_priority_hashes`].
pub fn load_priority_hashes_file(path: &Path) -> Result<Vec<u64>, Rdna2PackError> {
    let label = sanitize_source_path(path, None);
    let meta = fs::metadata(path).map_err(|_| {
        Rdna2PackError::InvalidPriorityHashes(format!(
            "priority hashes file '{label}' does not exist or is unreadable"
        ))
    })?;
    if !meta.is_file() {
        return Err(Rdna2PackError::InvalidPriorityHashes(format!(
            "priority hashes path '{label}' is not a regular file"
        )));
    }
    let text = fs::read_to_string(path).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to read priority hashes file '{label}': {}",
            err.kind()
        ))
    })?;
    parse_priority_hashes(&text)
}

/// Parse optional pipeline stage and 64-bit hex hash hints from a shader blob filename.
///
/// Supported patterns include:
/// * `vs_<16hex>.rdna2`, `ps_<16hex>.rdna2`, `cs_<16hex>.rdna2`
/// * `vs_0x<16hex>.bin`, `fs_<16hex>.shader`, `vertex-<16hex>.bin`
/// * `<16hex>.bin`, `0x<16hex>.rdna2`
/// * `<16hex>_ps.rdna2`, `splash_vs.bin`, `vs_splash_<16hex>.bin`
///
/// Any unrecognized name gracefully returns `Rdna2StageOverride::Auto` and `hash: None`.
pub fn parse_filename_hint(name: &str) -> Rdna2FilenameHint {
    let basename = name
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .unwrap_or("");
    if basename.is_empty() {
        return Rdna2FilenameHint::default();
    }

    let stem = match basename.rsplit_once('.') {
        Some((prefix, _ext)) if !prefix.is_empty() => prefix,
        Some(_) => return Rdna2FilenameHint::default(),
        None => basename,
    };

    let parts: Vec<&str> = stem
        .split(['_', '-', '.'])
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        return Rdna2FilenameHint::default();
    }

    let (stage, remaining): (Rdna2StageOverride, &[&str]) =
        if let Some(s) = parse_stage_token(parts[0]) {
            (s, &parts[1..])
        } else if let Some(s) = parse_stage_token(parts[parts.len() - 1]) {
            (s, &parts[..parts.len() - 1])
        } else {
            (Rdna2StageOverride::Auto, &parts[..])
        };

    let hash = match remaining {
        [] => None,
        [single] => parse_exact_hex_u64(single),
        many => many
            .last()
            .and_then(|tok| parse_exact_hex_u64(tok))
            .or_else(|| many.first().and_then(|tok| parse_exact_hex_u64(tok))),
    };

    Rdna2FilenameHint { stage, hash }
}

fn stage_to_override(stage: ShaderStage, is_mesh: bool) -> Rdna2StageOverride {
    if is_mesh {
        return Rdna2StageOverride::Mesh;
    }
    match stage {
        ShaderStage::Vertex => Rdna2StageOverride::Vertex,
        ShaderStage::Fragment => Rdna2StageOverride::Pixel,
        ShaderStage::Compute => Rdna2StageOverride::Compute,
    }
}

fn override_to_stage(stage: Rdna2StageOverride) -> Option<ShaderStage> {
    match stage {
        Rdna2StageOverride::Auto => None,
        Rdna2StageOverride::Vertex => Some(ShaderStage::Vertex),
        Rdna2StageOverride::Pixel => Some(ShaderStage::Fragment),
        Rdna2StageOverride::Compute | Rdna2StageOverride::Mesh => Some(ShaderStage::Compute),
    }
}

fn parse_manifest_stage(value: &str) -> Option<(ShaderStage, bool)> {
    if value.eq_ignore_ascii_case("vertex") {
        Some((ShaderStage::Vertex, false))
    } else if value.eq_ignore_ascii_case("pixel") || value.eq_ignore_ascii_case("fragment") {
        Some((ShaderStage::Fragment, false))
    } else if value.eq_ignore_ascii_case("compute") {
        Some((ShaderStage::Compute, false))
    } else if value.eq_ignore_ascii_case("mesh") || value.eq_ignore_ascii_case("ms") {
        Some((ShaderStage::Compute, true))
    } else {
        None
    }
}

fn parse_manifest_hex_u64(token: &str) -> Option<u64> {
    let hex = token
        .strip_prefix("0x")
        .or_else(|| token.strip_prefix("0X"))?;
    if hex.is_empty() || hex.len() > 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

/// Validate that a `code_file` path from `manifest.json` is strictly relative and contains no
/// traversal (`..`), root/drive prefixes, or NUL bytes, returning a normalized relative [`PathBuf`].
fn validate_manifest_relative_path(raw: &str) -> Result<PathBuf, Rdna2PackError> {
    if raw.is_empty() || raw.bytes().any(|b| b == 0) {
        return Err(Rdna2PackError::InvalidManifest(
            "native_stages entry has empty or invalid code_file".to_string(),
        ));
    }
    // Reject explicit leading slashes or Windows drive prefixes like `C:...` on all platforms.
    if raw.starts_with('/') || raw.starts_with('\\') {
        return Err(Rdna2PackError::InvalidManifest(
            "native_stages code_file must be a relative path".to_string(),
        ));
    }
    if raw.len() >= 2 && raw.as_bytes()[0].is_ascii_alphabetic() && raw.as_bytes()[1] == b':' {
        return Err(Rdna2PackError::InvalidManifest(
            "native_stages code_file must not contain a drive prefix".to_string(),
        ));
    }

    let mut rel = PathBuf::new();
    let mut normal_segments = 0usize;
    for segment in raw.split(['/', '\\']) {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".." {
            return Err(Rdna2PackError::InvalidManifest(
                "native_stages code_file must not contain parent directory traversal ('..')"
                    .to_string(),
            ));
        }
        rel.push(segment);
        normal_segments += 1;
    }
    if normal_segments == 0 {
        return Err(Rdna2PackError::InvalidManifest(
            "native_stages code_file must name a file".to_string(),
        ));
    }
    for comp in rel.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(Rdna2PackError::InvalidManifest(
                    "native_stages code_file must remain within the manifest directory".to_string(),
                ));
            }
        }
    }
    Ok(rel)
}

// ---------------------------------------------------------------------------
// Minimal zero-dependency RFC 8259 JSON parser (for `manifest.json`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(raw) => raw.parse::<u64>().ok(),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        let trimmed = input.strip_prefix('\u{feff}').unwrap_or(input);
        Self {
            bytes: trimmed.as_bytes(),
            pos: 0,
        }
    }

    fn parse_root(mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        let val = self.parse_value(0)?;
        self.skip_ws();
        if self.pos != self.bytes.len() {
            return Err(format!("unexpected trailing character at byte {}", self.pos));
        }
        Ok(val)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect_bytes(&mut self, lit: &[u8]) -> Result<(), String> {
        let end = self.pos.saturating_add(lit.len());
        if self.bytes.get(self.pos..end) == Some(lit) {
            self.pos = end;
            Ok(())
        } else {
            Err(format!("unexpected token at byte {}", self.pos))
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, String> {
        if depth > MAX_JSON_DEPTH {
            return Err("JSON nesting depth exceeds safety limit".to_string());
        }
        self.skip_ws();
        let b = self
            .peek()
            .ok_or_else(|| format!("unexpected end of JSON at byte {}", self.pos))?;
        match b {
            b'{' => self.parse_object(depth + 1),
            b'[' => self.parse_array(depth + 1),
            b'"' => self.parse_string().map(JsonValue::String),
            b't' => {
                self.expect_bytes(b"true")?;
                Ok(JsonValue::Bool(true))
            }
            b'f' => {
                self.expect_bytes(b"false")?;
                Ok(JsonValue::Bool(false))
            }
            b'n' => {
                self.expect_bytes(b"null")?;
                Ok(JsonValue::Null)
            }
            b'-' | b'0'..=b'9' => self.parse_number().map(JsonValue::Number),
            _ => Err(format!("unexpected byte 0x{b:02x} at byte {}", self.pos)),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.pos += 1; // '{'
        self.skip_ws();
        let mut entries = Vec::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected object key string at byte {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' after object key at byte {}", self.pos));
            }
            self.pos += 1;
            let value = self.parse_value(depth)?;
            entries.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                _ => {
                    return Err(format!(
                        "expected ',' or '}}' in object at byte {}",
                        self.pos
                    ));
                }
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.pos += 1; // '['
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let value = self.parse_value(depth)?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => {
                    return Err(format!(
                        "expected ',' or ']' in array at byte {}",
                        self.pos
                    ));
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        let end = self.pos.saturating_add(4);
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| "truncated \\uXXXX escape in JSON string".to_string())?;
        let text = std::str::from_utf8(slice)
            .map_err(|_| "invalid UTF-8 in \\uXXXX escape".to_string())?;
        let code = u16::from_str_radix(text, 16)
            .map_err(|_| format!("invalid hex digits in \\u escape at byte {}", self.pos))?;
        self.pos = end;
        Ok(code)
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at byte {}", self.pos));
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(b) = self.peek() {
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = self
                    .peek()
                    .ok_or_else(|| "unterminated escape sequence in JSON string".to_string())?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let first = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&first) {
                            self.expect_bytes(b"\\u")?;
                            let second = self.parse_hex4()?;
                            if !(0xDC00..=0xDFFF).contains(&second) {
                                return Err("invalid UTF-16 low surrogate in JSON string".to_string());
                            }
                            let cp = 0x10000
                                + (((u32::from(first) - 0xD800) << 10)
                                    | (u32::from(second) - 0xDC00));
                            let ch = char::from_u32(cp)
                                .ok_or_else(|| "invalid Unicode scalar in JSON string".to_string())?;
                            out.push(ch);
                        } else if (0xDC00..=0xDFFF).contains(&first) {
                            return Err("unpaired UTF-16 low surrogate in JSON string".to_string());
                        } else {
                            let ch = char::from_u32(u32::from(first))
                                .ok_or_else(|| "invalid Unicode scalar in JSON string".to_string())?;
                            out.push(ch);
                        }
                    }
                    _ => {
                        return Err(format!(
                            "invalid escape sequence '\\{}' at byte {}",
                            esc as char, self.pos
                        ));
                    }
                }
                continue;
            }
            if b < 0x20 {
                return Err(format!("unescaped control character in JSON string at byte {}", self.pos));
            }
            // Copy contiguous safe UTF-8 bytes up to the next '"' or '\' or control char.
            let start = self.pos;
            while let Some(next) = self.peek() {
                if next == b'"' || next == b'\\' || next < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            let chunk = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| "invalid UTF-8 inside JSON string".to_string())?;
            out.push_str(chunk);
        }
        Err("unterminated JSON string".to_string())
    }

    fn parse_number(&mut self) -> Result<String, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(format!("invalid JSON number at byte {}", self.pos)),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("invalid fraction in JSON number at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("invalid exponent in JSON number at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid UTF-8 in JSON number".to_string())?;
        Ok(text.to_string())
    }
}

/// Parse a `playgta6-pipeline-dump` `manifest.json` document (schema version 1) with zero
/// external dependencies.
///
/// Validates `"schema": "playgta6-pipeline-dump"`, `"schema_version": 1` (if present), and
/// extracts every entry in `"native_stages"` (`code_identity`, `stage`, `code_file`, and optional
/// `identity` / `code_bytes`).
pub fn parse_pipeline_dump_manifest(
    json_text: &str,
) -> Result<PipelineDumpManifest, Rdna2PackError> {
    let root = JsonParser::new(json_text)
        .parse_root()
        .map_err(Rdna2PackError::InvalidManifest)?;

    if root.as_object().is_none() {
        return Err(Rdna2PackError::InvalidManifest(
            "manifest root must be a JSON object".to_string(),
        ));
    }

    let schema = root
        .get("schema")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            Rdna2PackError::InvalidManifest(
                "missing string field \"schema\" (expected \"playgta6-pipeline-dump\")".to_string(),
            )
        })?;
    if schema != PIPELINE_DUMP_SCHEMA {
        return Err(Rdna2PackError::InvalidManifest(format!(
            "unsupported manifest schema '{schema}' (expected '{PIPELINE_DUMP_SCHEMA}')"
        )));
    }

    let schema_version = match root.get("schema_version") {
        Some(v) => {
            let ver = v.as_u64().ok_or_else(|| {
                Rdna2PackError::InvalidManifest(
                    "\"schema_version\" must be an unsigned integer".to_string(),
                )
            })?;
            if ver != PIPELINE_DUMP_SCHEMA_VERSION {
                return Err(Rdna2PackError::InvalidManifest(format!(
                    "unsupported manifest schema_version {ver} (expected {PIPELINE_DUMP_SCHEMA_VERSION})"
                )));
            }
            ver
        }
        None => PIPELINE_DUMP_SCHEMA_VERSION,
    };

    let native_stages_arr = root
        .get("native_stages")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            Rdna2PackError::InvalidManifest("missing array field \"native_stages\"".to_string())
        })?;

    let mut native_stages = Vec::with_capacity(native_stages_arr.len());
    for (idx, item) in native_stages_arr.iter().enumerate() {
        if item.as_object().is_none() {
            return Err(Rdna2PackError::InvalidManifest(format!(
                "native_stages[{idx}] must be a JSON object"
            )));
        }

        let code_id_raw = item
            .get("code_identity")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] missing string field \"code_identity\""
                ))
            })?;
        let code_identity = parse_manifest_hex_u64(code_id_raw).ok_or_else(|| {
            Rdna2PackError::InvalidManifest(format!(
                "native_stages[{idx}] has invalid hex code_identity '{code_id_raw}'"
            ))
        })?;

        let identity = match item.get("identity").and_then(JsonValue::as_str) {
            Some(s) if !s.is_empty() => Some(parse_manifest_hex_u64(s).ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] has invalid hex identity '{s}'"
                ))
            })?),
            _ => None,
        };

        let stage_raw = item
            .get("stage")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] missing string field \"stage\""
                ))
            })?;
        let (stage, is_mesh) = parse_manifest_stage(stage_raw).ok_or_else(|| {
            Rdna2PackError::InvalidManifest(format!(
                "native_stages[{idx}] has unsupported stage '{stage_raw}' (expected 'vertex', 'pixel', 'compute', or 'mesh')"
            ))
        })?;

        let code_file = item
            .get("code_file")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] missing string field \"code_file\""
                ))
            })?;
        let _ = validate_manifest_relative_path(code_file)?;

        let code_bytes = match item.get("code_bytes") {
            Some(v) => Some(v.as_u64().ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] field \"code_bytes\" must be an unsigned integer"
                ))
            })?),
            None => None,
        };

        let metadata_file = match item.get("metadata_file").and_then(JsonValue::as_str) {
            Some(s) if !s.is_empty() => {
                let _ = validate_manifest_relative_path(s)?;
                Some(s.to_string())
            }
            _ => None,
        };

        native_stages.push(PipelineDumpNativeStage {
            code_identity,
            identity,
            stage,
            is_mesh,
            code_file: code_file.to_string(),
            code_bytes,
            metadata_file,
        });
    }

    Ok(PipelineDumpManifest {
        schema: schema.to_string(),
        schema_version,
        native_stages,
    })
}

struct PackBuildState {
    /// All translated, Naga-validated, non-oversize unique candidates keyed by 64-bit hash.
    /// Value is `(source_order, PackEntry, Rdna2PackEntrySummary)` where `source_order` records
    /// deterministic appearance order (`manifest.json` `native_stages` first, then directory/input order).
    candidates_by_hash: BTreeMap<u64, (usize, PackEntry, Rdna2PackEntrySummary)>,
    next_source_order: usize,
    total_blobs: usize,
    skipped_or_failed: usize,
    failed: usize,
    manifest_duplicates_skipped: usize,
    skipped_oversize: usize,
    skipped_budget: usize,
    failures: Vec<Rdna2PackFailureSummary>,
}

impl PackBuildState {
    fn new() -> Self {
        Self {
            candidates_by_hash: BTreeMap::new(),
            next_source_order: 0,
            total_blobs: 0,
            skipped_or_failed: 0,
            failed: 0,
            manifest_duplicates_skipped: 0,
            skipped_oversize: 0,
            skipped_budget: 0,
            failures: Vec::new(),
        }
    }

    fn insert_translated(
        &mut self,
        hash: u64,
        source_order: usize,
        entry: PackEntry,
        entry_summary: Rdna2PackEntrySummary,
        options: &Rdna2PackOptions,
    ) {
        let wgsl_byte_len = entry.wgsl.len();
        if let Some(max_wgsl) = options.max_wgsl_bytes
            && wgsl_byte_len > max_wgsl
        {
            self.skipped_oversize += 1;
            self.skipped_or_failed += 1;
            return;
        }

        self.candidates_by_hash
            .insert(hash, (source_order, entry, entry_summary));
    }
}

fn validate_wgpk_v1_size(
    entry_count: usize,
    total_wgsl_bytes: u64,
    largest_wgsl_bytes: usize,
) -> Result<u64, Rdna2PackError> {
    let projected_total = u64::try_from(entry_count)
        .ok()
        .and_then(|count| count.checked_mul(PACK_ENTRY_BYTES as u64))
        .and_then(|table_bytes| (PACK_HEADER_BYTES as u64).checked_add(table_bytes))
        .and_then(|data_start| data_start.checked_add(total_wgsl_bytes));

    let fits_u32 = u32::try_from(entry_count).is_ok()
        && projected_total.is_some_and(|total| total <= u64::from(u32::MAX));
    if !fits_u32 {
        let total_desc = projected_total
            .map(|n| format!("{n} bytes"))
            .unwrap_or_else(|| "> u64::MAX bytes".to_string());
        return Err(Rdna2PackError::Pack(PackError::Overflow(format!(
            "WGPK v1 pack size ({total_desc} across {entry_count} stages; total_wgsl_bytes={total_wgsl_bytes}, largest_wgsl_bytes={largest_wgsl_bytes}) exceeds u32::MAX ({} bytes); re-run with --pack-version 2 (or auto), --max-wgsl-bytes <N>, and/or --max-pack-bytes <N>",
            u32::MAX
        ))));
    }

    Ok(projected_total.unwrap())
}

fn validate_wgpk_size(
    pack_version: PackVersion,
    entry_count: usize,
    total_wgsl_bytes: u64,
    largest_wgsl_bytes: usize,
) -> Result<u64, Rdna2PackError> {
    if pack_version == PackVersion::V1 {
        return validate_wgpk_v1_size(entry_count, total_wgsl_bytes, largest_wgsl_bytes);
    }
    let layout = resolve_pack_layout(pack_version, entry_count, total_wgsl_bytes)
        .map_err(Rdna2PackError::Pack)?;
    Ok(layout.total_size)
}

fn build_pack_result(
    mut state: PackBuildState,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let mut priority_rank: HashMap<u64, usize> =
        HashMap::with_capacity(options.priority_hashes.len());
    for &hash in &options.priority_hashes {
        let rank = priority_rank.len();
        priority_rank.entry(hash).or_insert(rank);
    }
    let priority_hashes_requested = priority_rank.len();

    let selected_by_hash: BTreeMap<u64, (PackEntry, Rdna2PackEntrySummary)> =
        if let Some(max_pack) = options.max_pack_bytes {
            let (effective_budget, entry_record_bytes) = match options.pack_version {
                PackVersion::V1 => (max_pack.min(u64::from(u32::MAX)), PACK_ENTRY_BYTES as u64),
                PackVersion::V2 => (max_pack, PACK_ENTRY_BYTES_V2 as u64),
                PackVersion::Auto => {
                    if max_pack <= u64::from(u32::MAX) {
                        (max_pack, PACK_ENTRY_BYTES as u64)
                    } else {
                        // Only pay 24-byte v2 TOC overhead when candidates cannot fit in v1.
                        let all_wgsl: u64 = state
                            .candidates_by_hash
                            .values()
                            .fold(0u64, |acc, (_, e, _)| acc.saturating_add(e.wgsl.len() as u64));
                        let all_v1_size = (PACK_HEADER_BYTES as u64)
                            .saturating_add(
                                (state.candidates_by_hash.len() as u64)
                                    .saturating_mul(PACK_ENTRY_BYTES as u64),
                            )
                            .saturating_add(all_wgsl);
                        if all_v1_size <= u64::from(u32::MAX) {
                            (max_pack, PACK_ENTRY_BYTES as u64)
                        } else {
                            (max_pack, PACK_ENTRY_BYTES_V2 as u64)
                        }
                    }
                }
            };
            let mut priority_candidates: Vec<(
                usize,
                u64,
                usize,
                PackEntry,
                Rdna2PackEntrySummary,
            )> = Vec::new();
            let mut regular_candidates: Vec<(u64, usize, PackEntry, Rdna2PackEntrySummary)> =
                Vec::with_capacity(state.candidates_by_hash.len());

            for (hash, (source_order, entry, summary)) in state.candidates_by_hash {
                if let Some(&rank) = priority_rank.get(&hash) {
                    priority_candidates.push((rank, hash, source_order, entry, summary));
                } else {
                    regular_candidates.push((hash, source_order, entry, summary));
                }
            }

            // Priority shaders are always selected first in priority-file order.
            priority_candidates.sort_by_key(|(rank, ..)| *rank);

            // Remaining non-priority shaders follow the configured `budget_order`.
            match options.budget_order {
                BudgetOrder::HashOrder => {
                    // Already sorted by ascending 64-bit hash from `BTreeMap`.
                }
                BudgetOrder::SmallestFirst => {
                    regular_candidates.sort_by(|a, b| {
                        a.2.wgsl
                            .len()
                            .cmp(&b.2.wgsl.len())
                            .then_with(|| a.0.cmp(&b.0))
                    });
                }
                BudgetOrder::ManifestOrder => {
                    regular_candidates
                        .sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                }
            }

            let mut current_pack_bytes = PACK_HEADER_BYTES as u64;
            let mut selected: BTreeMap<u64, (PackEntry, Rdna2PackEntrySummary)> = BTreeMap::new();

            // Phase 1: pack priority-listed shaders first in priority-file order.
            // A priority shader that exceeds remaining budget is skipped without halting smaller
            // subsequent priority or non-priority shaders.
            for (_rank, hash, _source_order, entry, summary) in priority_candidates {
                let entry_cost = entry_record_bytes.saturating_add(entry.wgsl.len() as u64);
                let projected = current_pack_bytes.saturating_add(entry_cost);
                if projected <= effective_budget {
                    current_pack_bytes = projected;
                    selected.insert(hash, (entry, summary));
                } else {
                    state.skipped_budget += 1;
                    state.skipped_or_failed += 1;
                }
            }

            // Phase 2: fill remaining budget with non-priority shaders according to `budget_order`.
            let mut budget_exhausted = false;
            for (hash, _source_order, entry, summary) in regular_candidates {
                let entry_cost = entry_record_bytes.saturating_add(entry.wgsl.len() as u64);
                let projected = current_pack_bytes.saturating_add(entry_cost);
                if !budget_exhausted && projected <= effective_budget {
                    current_pack_bytes = projected;
                    selected.insert(hash, (entry, summary));
                } else {
                    // `HashOrder` stops at the first candidate exceeding the budget to preserve
                    // legacy behavior 100%; `SmallestFirst` is ordered by ascending WGSL size so
                    // no later candidate can fit either. `ManifestOrder` continues scanning so a
                    // single large early stage does not block smaller early-manifest stages.
                    if matches!(
                        options.budget_order,
                        BudgetOrder::HashOrder | BudgetOrder::SmallestFirst
                    ) {
                        budget_exhausted = true;
                    }
                    state.skipped_budget += 1;
                    state.skipped_or_failed += 1;
                }
            }

            selected
        } else {
            let mut selected: BTreeMap<u64, (PackEntry, Rdna2PackEntrySummary)> = BTreeMap::new();
            for (hash, (_source_order, entry, summary)) in state.candidates_by_hash {
                selected.insert(hash, (entry, summary));
            }
            selected
        };

    let mut entries: Vec<PackEntry> = Vec::with_capacity(selected_by_hash.len());
    let mut entry_summaries: Vec<Rdna2PackEntrySummary> =
        Vec::with_capacity(selected_by_hash.len());
    for (entry, entry_summary) in selected_by_hash.into_values() {
        entries.push(entry);
        entry_summaries.push(entry_summary);
    }

    let priority_hashes_packed = if priority_rank.is_empty() {
        0
    } else {
        entries
            .iter()
            .filter(|entry| priority_rank.contains_key(&entry.hash))
            .count()
    };

    let mut vertex_count = 0usize;
    let mut pixel_count = 0usize;
    let mut compute_count = 0usize;
    let mut largest_wgsl_bytes = 0usize;
    let mut total_wgsl_bytes = 0u64;
    for entry in &entries {
        match entry.stage {
            ShaderStage::Vertex => vertex_count += 1,
            ShaderStage::Fragment => pixel_count += 1,
            ShaderStage::Compute => compute_count += 1,
        }
        let wgsl_len = entry.wgsl.len();
        if wgsl_len > largest_wgsl_bytes {
            largest_wgsl_bytes = wgsl_len;
        }
        total_wgsl_bytes = total_wgsl_bytes.saturating_add(wgsl_len as u64);
    }

    let expected_pack_bytes = validate_wgpk_size(
        options.pack_version,
        entries.len(),
        total_wgsl_bytes,
        largest_wgsl_bytes,
    )?;

    let written = write_pack_with_layout_versioned(&entries, options.pack_flags, options.pack_version)
        .map_err(|err| match err {
            PackError::Overflow(_) if options.pack_version == PackVersion::V1 => {
                Rdna2PackError::Pack(PackError::Overflow(format!(
                    "WGPK v1 pack size (total_wgsl_bytes={total_wgsl_bytes}, largest_wgsl_bytes={largest_wgsl_bytes}, entries={}) exceeds u32::MAX ({} bytes); re-run with --pack-version 2 (or auto), --max-wgsl-bytes <N>, and/or --max-pack-bytes <N>",
                    entries.len(),
                    u32::MAX
                )))
            }
            other => Rdna2PackError::Pack(other),
        })?;
    debug_assert_eq!(expected_pack_bytes, written.bytes.len() as u64);

    let mut binding_limits = PackBindingLimitSummary::default();
    let d3d12_limits = StageBindingLimits::webgpu_d3d12();
    for es in &entry_summaries {
        binding_limits.record_wgsl_stage(
            &es.bindings,
            es.rt_arena,
            es.mesh_output.is_some(),
            es.binding_pack,
            d3d12_limits,
        );
    }

    let summary = Rdna2PackSummary {
        pack_version: written.version,
        total_blobs: state.total_blobs,
        translated_ok: entries.len(),
        skipped_or_failed: state.skipped_or_failed,
        failed: state.failed,
        manifest_duplicates_skipped: state.manifest_duplicates_skipped,
        budget_order: options.budget_order,
        skipped_oversize: state.skipped_oversize,
        skipped_budget: state.skipped_budget,
        omitted_by_budget: state.skipped_budget,
        priority_hashes_requested,
        priority_hashes_packed,
        largest_wgsl_bytes,
        total_wgsl_bytes,
        vertex_count,
        pixel_count,
        compute_count,
        vertex_packed: vertex_count,
        fragment_packed: pixel_count,
        compute_packed: compute_count,
        pack_bytes: written.bytes.len(),
        binding_limits,
        entries: entry_summaries,
        failures: state.failures,
    };

    Ok(Rdna2PackResult {
        pack_bytes: written.bytes,
        entries,
        summary,
    })
}

fn write_pack_bytes_to_file(out_path: &Path, pack_bytes: &[u8]) -> Result<(), Rdna2PackError> {
    let out_label = sanitize_source_path(out_path, None);
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            Rdna2PackError::Io(format!(
                "failed to create output directory for '{out_label}': {}",
                err.kind()
            ))
        })?;
    }
    fs::write(out_path, pack_bytes).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to write WGPK pack '{out_label}': {}",
            err.kind()
        ))
    })?;
    Ok(())
}

const fn stage_override_key(stage: Rdna2StageOverride) -> u8 {
    match stage {
        Rdna2StageOverride::Auto => 0,
        Rdna2StageOverride::Vertex => 1,
        Rdna2StageOverride::Pixel => 2,
        Rdna2StageOverride::Compute => 3,
        Rdna2StageOverride::Mesh => 4,
    }
}

#[derive(Debug, Clone)]
struct TranslatedStageData {
    stage: ShaderStage,
    wgsl: String,
    entry_point: String,
    naga_validated: bool,
    translated_op_count: usize,
    unsupported_op_count: usize,
    vertex_input_mode: String,
    bindings: Vec<WgslBindingInfo>,
    vertex_inputs: Vec<WgslVertexInputInfo>,
    vs_outputs: Vec<WgslStageParamInfo>,
    ps_inputs: Vec<WgslStageParamInfo>,
    mesh_output: Option<MeshOutputLayout>,
    rt_arena: bool,
    rt_prelude_version: Option<u32>,
    bvh_ops: u32,
    raw_address: bool,
    atomic64: bool,
    binding_pack: BindingPackStats,
}

impl TranslatedStageData {
    fn from_output(output: Rdna2WgslOutput) -> Self {
        Self {
            stage: output.stage,
            wgsl: output.wgsl,
            entry_point: output.entry_point,
            naga_validated: output.naga_validated,
            translated_op_count: output.translated_op_count,
            unsupported_op_count: output.unsupported_op_count,
            vertex_input_mode: output.binding_summary.vertex_input_mode,
            bindings: output.bindings,
            vertex_inputs: output.vertex_inputs,
            vs_outputs: output.vs_outputs,
            ps_inputs: output.ps_inputs,
            mesh_output: output.mesh_output,
            rt_arena: output.rt_arena,
            rt_prelude_version: output.rt_prelude_version,
            bvh_ops: output.bvh_ops,
            raw_address: output.raw_address,
            atomic64: output.atomic64,
            binding_pack: output.binding_pack,
        }
    }

    fn into_pack_parts(self, hash: u64) -> (PackEntry, Rdna2PackEntrySummary) {
        let wgsl_byte_len = self.wgsl.len();
        let entry = PackEntry {
            hash,
            stage: self.stage,
            wgsl: self.wgsl,
        };
        let summary = Rdna2PackEntrySummary {
            hash_fnv1a64: hash,
            stage: self.stage,
            entry_point: self.entry_point,
            naga_validated: self.naga_validated,
            translated_op_count: self.translated_op_count,
            unsupported_op_count: self.unsupported_op_count,
            wgsl_byte_len,
            vertex_input_mode: self.vertex_input_mode,
            bindings: self.bindings,
            vertex_inputs: self.vertex_inputs,
            vs_outputs: self.vs_outputs,
            ps_inputs: self.ps_inputs,
            mesh_output: self.mesh_output,
            rt_arena: self.rt_arena,
            rt_prelude_version: self.rt_prelude_version,
            bvh_ops: self.bvh_ops,
            raw_address: self.raw_address,
            atomic64: self.atomic64,
            binding_pack: self.binding_pack,
        };
        (entry, summary)
    }
}

enum TranslatedStageOutcome {
    Translated(TranslatedStageData),
    Oversize,
    Failed {
        error: Option<Rdna2PackError>,
        reason: String,
    },
}

/// Execute `translate_one` across up to `jobs` worker threads using dynamic `AtomicUsize`
/// work-stealing and collect outcomes into a pre-allocated indexed `Vec<Option<R>>` so callers
/// can aggregate results in exact deterministic input order.
///
/// When `should_abort(&outcome)` returns `true` at index `k`, workers stop picking up indices
/// `>= k` while guaranteeing every index `< k` (and `k` itself) still completes.
fn run_parallel_indexed<T, R, F, P>(
    items: &[T],
    jobs: usize,
    translate_one: F,
    should_abort: P,
) -> Vec<Option<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> R + Sync,
    P: Fn(&R) -> bool + Sync,
{
    let count = items.len();
    if count == 0 {
        return Vec::new();
    }

    let worker_count = jobs.clamp(1, count);
    let mut slots: Vec<Option<R>> = std::iter::repeat_with(|| None).take(count).collect();
    if worker_count <= 1 {
        for (idx, item) in items.iter().enumerate() {
            let outcome = translate_one(idx, item);
            let abort = should_abort(&outcome);
            slots[idx] = Some(outcome);
            if abort {
                break;
            }
        }
        return slots;
    }

    let next_idx = AtomicUsize::new(0);
    let abort_index = AtomicUsize::new(usize::MAX);

    let all_batches: Vec<Vec<(usize, R)>> = thread::scope(|scope| {
        let worker = || {
            let mut local = Vec::new();
            loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= count || idx >= abort_index.load(Ordering::Relaxed) {
                    break;
                }
                let outcome = translate_one(idx, &items[idx]);
                if should_abort(&outcome) {
                    abort_index.fetch_min(idx, Ordering::Relaxed);
                }
                local.push((idx, outcome));
            }
            local
        };

        let mut handles = Vec::with_capacity(worker_count - 1);
        for _ in 0..(worker_count - 1) {
            handles.push(scope.spawn(worker));
        }

        // Calling thread participates as worker #0 to avoid leaving it idle.
        let main_batch = worker();

        let mut batches = Vec::with_capacity(worker_count);
        batches.push(main_batch);
        for handle in handles {
            match handle.join() {
                Ok(batch) => batches.push(batch),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        batches
    });

    for batch in all_batches {
        for (idx, outcome) in batch {
            slots[idx] = Some(outcome);
        }
    }

    slots
}

struct BlobCandidate<'a> {
    blob: &'a Rdna2BlobInput,
    source_order: usize,
    stage: Rdna2StageOverride,
    explicit_hash: Option<u64>,
    final_hash: u64,
    work_idx: Option<usize>,
}

struct BlobTranslateTask<'a> {
    bytes: &'a [u8],
    stage: Rdna2StageOverride,
    fssm_meta: Option<&'a FssmStageMetadata>,
}

fn translate_blob_task(
    task: &BlobTranslateTask<'_>,
    options: &Rdna2PackOptions,
) -> TranslatedStageOutcome {
    let translate_opts = Rdna2TranslateOptions {
        stage: task.stage,
        entry_point: options.translate.entry_point.clone(),
        min_cbv_vec4_count: options.translate.min_cbv_vec4_count,
        workgroup_size: options.translate.workgroup_size,
        workgroup_id_sgpr_base: options.translate.workgroup_id_sgpr_base,
        preserve_fragment_interpolation: options.translate.preserve_fragment_interpolation,
        fssm_meta: task
            .fssm_meta
            .cloned()
            .or_else(|| options.translate.fssm_meta.clone()),
    };

    match translate_rdna2_to_wgsl(task.bytes, &translate_opts) {
        Ok(out) if out.naga_validated => {
            if let Some(max_wgsl) = options.max_wgsl_bytes
                && out.wgsl.len() > max_wgsl
            {
                TranslatedStageOutcome::Oversize
            } else {
                TranslatedStageOutcome::Translated(TranslatedStageData::from_output(out))
            }
        }
        Ok(_) => TranslatedStageOutcome::Failed {
            error: None,
            reason: "translated WGSL did not pass Naga validation".to_string(),
        },
        Err(err) => TranslatedStageOutcome::Failed {
            error: None,
            reason: format!("RDNA2->WGSL translation failed: {err}"),
        },
    }
}

fn pack_blobs_into_map(
    blobs: &[Rdna2BlobInput],
    options: &Rdna2PackOptions,
    state: &mut PackBuildState,
) {
    let source_order_base = state.next_source_order;
    state.next_source_order = state.next_source_order.saturating_add(blobs.len());

    // Sort a view of indices deterministically by blob name then raw content length/bytes
    // so duplicate-hash tie-breaking is independent of caller slice order.
    let mut order: Vec<usize> = (0..blobs.len()).collect();
    order.sort_by(|&a, &b| {
        blobs[a]
            .name
            .cmp(&blobs[b].name)
            .then_with(|| blobs[a].bytes.len().cmp(&blobs[b].bytes.len()))
            .then_with(|| blobs[a].bytes.cmp(&blobs[b].bytes))
    });

    // Collect candidates in deterministic order and deduplicate identical `(bytes, stage)`
    // translation inputs so duplicate shader blobs are translated at most once.
    let mut candidates: Vec<BlobCandidate<'_>> = Vec::with_capacity(order.len());
    let mut unique_tasks: Vec<BlobTranslateTask<'_>> = Vec::new();
    let mut task_uses: Vec<usize> = Vec::new();
    let mut dedup_map: BTreeMap<(u64, u8), Vec<usize>> = BTreeMap::new();

    for idx in order {
        let blob = &blobs[idx];
        let hint = parse_filename_hint(&blob.name);
        let stage = if blob.stage_override != Rdna2StageOverride::Auto {
            blob.stage_override
        } else if hint.stage != Rdna2StageOverride::Auto {
            hint.stage
        } else {
            options.translate.stage
        };
        let explicit_hash = blob.hash_override.or(hint.hash);
        let content_hash = fnv1a_64(&blob.bytes);
        let base_hash = explicit_hash.unwrap_or(content_hash);
        let final_hash = if stage == Rdna2StageOverride::Mesh {
            mesh_compute_pack_key(base_hash)
        } else {
            base_hash
        };

        let work_idx = if blob.bytes.len() < MIN_RDNA2_BLOB_BYTES {
            None
        } else {
            let key = (content_hash, stage_override_key(stage));
            let bucket = dedup_map.entry(key).or_default();
            let existing = bucket.iter().copied().find(|&task_idx| {
                unique_tasks[task_idx].bytes == blob.bytes.as_slice()
                    && unique_tasks[task_idx].fssm_meta == blob.fssm_meta.as_ref()
            });
            let assigned = match existing {
                Some(task_idx) => task_idx,
                None => {
                    let task_idx = unique_tasks.len();
                    unique_tasks.push(BlobTranslateTask {
                        bytes: &blob.bytes,
                        stage,
                        fssm_meta: blob.fssm_meta.as_ref(),
                    });
                    task_uses.push(0);
                    bucket.push(task_idx);
                    task_idx
                }
            };
            task_uses[assigned] += 1;
            Some(assigned)
        };

        candidates.push(BlobCandidate {
            blob,
            source_order: source_order_base.saturating_add(idx),
            stage,
            explicit_hash,
            final_hash,
            work_idx,
        });
    }

    let mut outcomes: Vec<Option<TranslatedStageOutcome>> = run_parallel_indexed(
        &unique_tasks,
        options.effective_jobs(),
        |_idx, task| translate_blob_task(task, options),
        |_| false,
    );

    for candidate in candidates {
        let Some(work_idx) = candidate.work_idx else {
            state.skipped_or_failed += 1;
            state.failed += 1;
            state.failures.push(Rdna2PackFailureSummary {
                code_identity: candidate.explicit_hash,
                stage: override_to_stage(candidate.stage),
                reason: format!(
                    "RDNA2 blob too short ({} bytes, minimum {MIN_RDNA2_BLOB_BYTES})",
                    candidate.blob.bytes.len()
                ),
            });
            continue;
        };

        task_uses[work_idx] = task_uses[work_idx].saturating_sub(1);
        let remaining_uses = task_uses[work_idx];

        match outcomes[work_idx].as_ref() {
            Some(TranslatedStageOutcome::Failed { reason, .. }) => {
                state.skipped_or_failed += 1;
                state.failed += 1;
                state.failures.push(Rdna2PackFailureSummary {
                    code_identity: Some(candidate.final_hash),
                    stage: override_to_stage(candidate.stage),
                    reason: reason.clone(),
                });
            }
            Some(TranslatedStageOutcome::Oversize) => {
                if state.candidates_by_hash.contains_key(&candidate.final_hash) {
                    state.skipped_or_failed += 1;
                } else {
                    state.skipped_oversize += 1;
                    state.skipped_or_failed += 1;
                }
            }
            Some(TranslatedStageOutcome::Translated(_)) => {
                if let Some(existing) = state.candidates_by_hash.get_mut(&candidate.final_hash) {
                    existing.0 = existing.0.min(candidate.source_order);
                    state.skipped_or_failed += 1;
                    continue;
                }
                let data = if remaining_uses == 0 {
                    match outcomes[work_idx].take() {
                        Some(TranslatedStageOutcome::Translated(d)) => d,
                        _ => unreachable!(),
                    }
                } else {
                    match outcomes[work_idx].as_ref() {
                        Some(TranslatedStageOutcome::Translated(d)) => d.clone(),
                        _ => unreachable!(),
                    }
                };
                let (entry, entry_summary) = data.into_pack_parts(candidate.final_hash);
                state.insert_translated(
                    candidate.final_hash,
                    candidate.source_order,
                    entry,
                    entry_summary,
                    options,
                );
            }
            None => unreachable!("all blob translation tasks must produce an outcome"),
        }
    }
}

/// Translate and deterministically pack an in-memory slice of raw RDNA2 shader blobs into a
/// `WGPK` v1 `.wgsl.pack` archive.
///
/// Every shader is unconditionally translated and validated by Naga via
/// [`translate_rdna2_to_wgsl`]. Blobs shorter than [`MIN_RDNA2_BLOB_BYTES`], blobs whose
/// translation fails, or blobs colliding on an already-packed hash are counted in
/// `summary.skipped_or_failed` without panicking.
///
/// Entries in the resulting `WGPK` binary are deterministically sorted by ascending `u64` hash.
pub fn pack_rdna2_blobs(
    blobs: &[Rdna2BlobInput],
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let mut state = PackBuildState::new();
    state.total_blobs = blobs.len();
    pack_blobs_into_map(blobs, options, &mut state);
    build_pack_result(state, options)
}

/// Collect regular RDNA2 blob files (`*.bin`, `*.rdna2`, `*.shader`) inside `input_dir`,
/// ignoring non-files and any symlinks that resolve outside `input_dir`.
fn collect_rdna2_blob_paths(input_dir: &Path) -> Result<Vec<(String, PathBuf)>, Rdna2PackError> {
    let meta = fs::metadata(input_dir).map_err(|_| {
        Rdna2PackError::InvalidInputDir(format!(
            "directory '{}' does not exist or is unreadable",
            sanitize_source_path(input_dir, None)
        ))
    })?;
    if !meta.is_dir() {
        return Err(Rdna2PackError::InvalidInputDir(format!(
            "'{}' is not a directory",
            sanitize_source_path(input_dir, None)
        )));
    }

    let canonical_root = fs::canonicalize(input_dir).map_err(|_| {
        Rdna2PackError::InvalidInputDir(format!(
            "failed to resolve directory '{}'",
            sanitize_source_path(input_dir, None)
        ))
    })?;

    let mut visited_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    visited_dirs.insert(canonical_root.clone());

    let mut dir_stack: Vec<PathBuf> = vec![canonical_root.clone()];
    let mut files: Vec<(String, PathBuf)> = Vec::new();

    while let Some(dir) = dir_stack.pop() {
        let read_dir = fs::read_dir(&dir).map_err(|err| {
            Rdna2PackError::Io(format!(
                "failed to list directory '{}': {}",
                sanitize_source_path(&dir, Some(&canonical_root)),
                err.kind()
            ))
        })?;

        for entry_res in read_dir {
            let Ok(entry) = entry_res else {
                continue;
            };
            let path = entry.path();
            let Ok(sym_meta) = fs::symlink_metadata(&path) else {
                continue;
            };

            if sym_meta.file_type().is_symlink() {
                let Ok(target) = fs::canonicalize(&path) else {
                    // Ignore broken symlinks.
                    continue;
                };
                if !target.starts_with(&canonical_root) {
                    // Ignore symlinks escaping the input directory.
                    continue;
                }
                let Ok(target_meta) = fs::metadata(&target) else {
                    continue;
                };
                if target_meta.is_dir() {
                    if visited_dirs.insert(target.clone()) {
                        dir_stack.push(target);
                    }
                } else if target_meta.is_file() && is_supported_rdna2_extension(&path) {
                    let rel_key = sanitize_source_path(&path, Some(&canonical_root));
                    files.push((rel_key, target));
                }
                continue;
            }

            if sym_meta.is_dir() {
                if let Ok(canon_dir) = fs::canonicalize(&path)
                    && canon_dir.starts_with(&canonical_root)
                    && visited_dirs.insert(canon_dir.clone())
                {
                    dir_stack.push(canon_dir);
                }
                continue;
            }

            if sym_meta.is_file() && is_supported_rdna2_extension(&path) {
                let rel_key = sanitize_source_path(&path, Some(&canonical_root));
                files.push((rel_key, path));
            }
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn collect_directory_blobs(
    input_dir: &Path,
) -> Result<(Vec<Rdna2BlobInput>, usize), Rdna2PackError> {
    let file_paths = collect_rdna2_blob_paths(input_dir)?;

    let mut blobs = Vec::with_capacity(file_paths.len());
    let mut read_failures = 0usize;

    for (rel_key, path) in &file_paths {
        match fs::read(path) {
            Ok(bytes) => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| rel_key.clone());
                let mut blob = Rdna2BlobInput::new(name, bytes);
                let meta_path = path.with_extension("meta");
                if let Ok(sm) = fs::symlink_metadata(&meta_path)
                    && !sm.file_type().is_symlink()
                    && sm.is_file()
                    && let Ok(meta_bytes) = fs::read(&meta_path)
                    && let Some(meta) = parse_fssm_stage_metadata(&meta_bytes)
                {
                    blob = blob.with_fssm_meta(meta);
                }
                blobs.push(blob);
            }
            Err(_) => {
                read_failures += 1;
            }
        }
    }

    Ok((blobs, read_failures))
}

/// Scan `input_dir` for raw RDNA2 shader blobs (`*.bin`, `*.rdna2`, `*.shader`),
/// translate every blob via [`translate_rdna2_to_wgsl`], and build a deterministic
/// `WGPK` v1 `.wgsl.pack` archive in memory.
///
/// Non-files and symlinks resolving outside `input_dir` are safely ignored.
pub fn pack_rdna2_directory(
    input_dir: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let (blobs, read_failures) = collect_directory_blobs(input_dir)?;

    let mut result = pack_rdna2_blobs(&blobs, options)?;
    if read_failures > 0 {
        result.summary.total_blobs += read_failures;
        result.summary.skipped_or_failed += read_failures;
        result.summary.failed += read_failures;
    }
    Ok(result)
}

/// Convenience helper that runs [`pack_rdna2_directory`] and writes the resulting
/// `WGPK` v1 `.wgsl.pack` bytes to `out_path` (creating parent directories if needed).
pub fn pack_rdna2_directory_to_file(
    input_dir: &Path,
    out_path: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let result = pack_rdna2_directory(input_dir, options)?;
    write_pack_bytes_to_file(out_path, &result.pack_bytes)?;
    Ok(result)
}

fn translate_manifest_stage(
    entry: &PipelineDumpNativeStage,
    canonical_parent: &Path,
    options: &Rdna2PackOptions,
) -> TranslatedStageOutcome {
    let make_shader_failure = |reason: String| -> TranslatedStageOutcome {
        TranslatedStageOutcome::Failed {
            error: Some(Rdna2PackError::ShaderFailed {
                code_identity: Some(entry.code_identity),
                stage: Some(entry.stage),
                reason: reason.clone(),
            }),
            reason,
        }
    };

    let rel_path = match validate_manifest_relative_path(&entry.code_file) {
        Ok(p) => p,
        Err(err) => {
            let reason = err.to_string();
            return TranslatedStageOutcome::Failed {
                error: Some(err),
                reason,
            };
        }
    };

    let joined = canonical_parent.join(&rel_path);
    let canonical_file = match fs::canonicalize(&joined) {
        Ok(target) => target,
        Err(err) => {
            return make_shader_failure(format!(
                "failed to locate code_file '{}': {}",
                sanitize_source_path(&rel_path, None),
                err.kind()
            ));
        }
    };

    if !canonical_file.starts_with(canonical_parent) {
        return make_shader_failure(
            "code_file resolves outside the manifest directory".to_string(),
        );
    }

    match fs::metadata(&canonical_file) {
        Ok(m) if m.is_file() => {}
        _ => {
            return make_shader_failure("code_file is not a regular file".to_string());
        }
    }

    let bytes = match fs::read(&canonical_file) {
        Ok(b) => b,
        Err(err) => {
            return make_shader_failure(format!("failed to read code_file: {}", err.kind()));
        }
    };

    let actual_hash = fnv1a_64(&bytes);
    if actual_hash != entry.code_identity {
        return TranslatedStageOutcome::Failed {
            error: Some(Rdna2PackError::HashMismatch {
                expected: entry.code_identity,
                actual: actual_hash,
                stage: entry.stage,
            }),
            reason: format!(
                "FNV-1a 64 hash mismatch: expected {}, actual {}",
                format_shader_hash(entry.code_identity),
                format_shader_hash(actual_hash)
            ),
        };
    }

    if bytes.len() < MIN_RDNA2_BLOB_BYTES {
        return make_shader_failure(format!(
            "RDNA2 blob too short ({} bytes, minimum {MIN_RDNA2_BLOB_BYTES})",
            bytes.len()
        ));
    }

    let fssm_meta = entry
        .metadata_file
        .as_deref()
        .and_then(|meta_rel| validate_manifest_relative_path(meta_rel).ok())
        .map(|rel| canonical_parent.join(rel))
        .or_else(|| Some(canonical_file.with_extension("meta")))
        .and_then(|candidate| fs::canonicalize(candidate).ok())
        .filter(|canon| {
            canon.starts_with(canonical_parent) && fs::metadata(canon).is_ok_and(|m| m.is_file())
        })
        .and_then(|canon| fs::read(canon).ok())
        .and_then(|meta_bytes| parse_fssm_stage_metadata(&meta_bytes))
        .or_else(|| options.translate.fssm_meta.clone());

    let translate_opts = Rdna2TranslateOptions {
        stage: stage_to_override(entry.stage, entry.is_mesh),
        entry_point: options.translate.entry_point.clone(),
        min_cbv_vec4_count: options.translate.min_cbv_vec4_count,
        workgroup_size: options.translate.workgroup_size,
        workgroup_id_sgpr_base: options.translate.workgroup_id_sgpr_base,
        preserve_fragment_interpolation: options.translate.preserve_fragment_interpolation,
        fssm_meta,
    };

    match translate_rdna2_to_wgsl(&bytes, &translate_opts) {
        Ok(out) if out.naga_validated => {
            if let Some(max_wgsl) = options.max_wgsl_bytes
                && out.wgsl.len() > max_wgsl
            {
                TranslatedStageOutcome::Oversize
            } else {
                TranslatedStageOutcome::Translated(TranslatedStageData::from_output(out))
            }
        }
        Ok(_) => make_shader_failure("translated WGSL did not pass Naga validation".to_string()),
        Err(err) => make_shader_failure(format!("RDNA2->WGSL translation failed: {err}")),
    }
}

fn pack_manifest_into_map(
    manifest_path: &Path,
    options: &Rdna2PackOptions,
    state: &mut PackBuildState,
) -> Result<(), Rdna2PackError> {
    let manifest_label = sanitize_source_path(manifest_path, None);
    let meta = fs::metadata(manifest_path).map_err(|_| {
        Rdna2PackError::InvalidManifest(format!(
            "manifest '{}' does not exist or is unreadable",
            manifest_label
        ))
    })?;
    if !meta.is_file() {
        return Err(Rdna2PackError::InvalidManifest(format!(
            "'{}' is not a regular file",
            manifest_label
        )));
    }

    let json_text = fs::read_to_string(manifest_path).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to read manifest '{manifest_label}': {}",
            err.kind()
        ))
    })?;
    let manifest = parse_pipeline_dump_manifest(&json_text)?;

    let parent_dir = match manifest_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let canonical_parent = fs::canonicalize(parent_dir).map_err(|_| {
        Rdna2PackError::Io(format!(
            "failed to resolve parent directory of manifest '{manifest_label}'"
        ))
    })?;

    // Deduplicate native_stages by `code_identity` (multiple PSO metadata blobs may share one
    // code blob in `playgta6-pipeline-dump`), preserving original manifest appearance index for
    // `BudgetOrder::ManifestOrder` while sorting by `code_identity` for deterministic translation
    // and strict-mode failure reporting.
    let mut seen_pack_keys: BTreeSet<u64> = BTreeSet::new();
    let mut unique_stages: Vec<(usize, PipelineDumpNativeStage)> =
        Vec::with_capacity(manifest.native_stages.len());
    for entry in manifest.native_stages {
        let pack_key = if entry.is_mesh {
            mesh_compute_pack_key(entry.code_identity)
        } else {
            entry.code_identity
        };
        if !seen_pack_keys.insert(pack_key) {
            state.manifest_duplicates_skipped += 1;
            continue;
        }
        let source_order = state.next_source_order;
        state.next_source_order = state.next_source_order.saturating_add(1);
        unique_stages.push((source_order, entry));
    }
    unique_stages.sort_by_key(|(_, entry)| {
        if entry.is_mesh {
            mesh_compute_pack_key(entry.code_identity)
        } else {
            entry.code_identity
        }
    });

    state.total_blobs += unique_stages.len();

    let outcomes: Vec<Option<TranslatedStageOutcome>> = run_parallel_indexed(
        &unique_stages,
        options.effective_jobs(),
        |_idx, (_, entry)| translate_manifest_stage(entry, &canonical_parent, options),
        |outcome| {
            !options.best_effort && matches!(outcome, TranslatedStageOutcome::Failed { .. })
        },
    );

    for ((source_order, entry), outcome_opt) in unique_stages.into_iter().zip(outcomes.into_iter()) {
        match outcome_opt.expect("manifest stage before strict abort must have an outcome") {
            TranslatedStageOutcome::Failed { error, reason } => {
                if options.best_effort {
                    state.skipped_or_failed += 1;
                    state.failed += 1;
                    state.failures.push(Rdna2PackFailureSummary {
                        code_identity: Some(entry.code_identity),
                        stage: Some(entry.stage),
                        reason,
                    });
                } else {
                    return Err(
                        error.expect("manifest stage failure must include Rdna2PackError")
                    );
                }
            }
            TranslatedStageOutcome::Oversize => {
                state.skipped_oversize += 1;
                state.skipped_or_failed += 1;
            }
            TranslatedStageOutcome::Translated(data) => {
                let pack_key = if entry.is_mesh {
                    mesh_compute_pack_key(entry.code_identity)
                } else {
                    entry.code_identity
                };
                let (pack_entry, entry_summary) = data.into_pack_parts(pack_key);
                state.insert_translated(
                    pack_key,
                    source_order,
                    pack_entry,
                    entry_summary,
                    options,
                );
            }
        }
    }

    Ok(())
}

/// Pack RDNA2 shaders directly from a `playgta6-pipeline-dump` `manifest.json` (`native_stages`).
///
/// Deduplicates entries by `code_identity`, verifies whole-file FNV-1a 64 hashes against
/// `code_identity`, passes explicit `ShaderStage` (`Vertex` / `Fragment` / `Compute`) into
/// [`translate_rdna2_to_wgsl`], and builds a deterministic `WGPK` v1 `.wgsl.pack` archive.
///
/// In strict mode (`options.best_effort == false`), any missing file, escaping symlink, FNV-1a
/// mismatch, short blob, or translation/Naga failure returns an error. When
/// `options.best_effort == true`, individual failing shaders are skipped and recorded in
/// `summary.failures`.
pub fn pack_rdna2_manifest(
    manifest_path: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    pack_rdna2_sources(None, Some(manifest_path), options)
}

/// Convenience helper that runs [`pack_rdna2_manifest`] and writes the resulting
/// `WGPK` v1 `.wgsl.pack` bytes to `out_path` (creating parent directories if needed).
pub fn pack_rdna2_manifest_to_file(
    manifest_path: &Path,
    out_path: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    pack_rdna2_sources_to_file(None, Some(manifest_path), out_path, options)
}

/// Pack RDNA2 shaders from an optional `playgta6-pipeline-dump` `manifest.json`, an optional
/// raw blob directory, or both combined (manifest entries take precedence on matching hashes).
pub fn pack_rdna2_sources(
    input_dir: Option<&Path>,
    manifest_path: Option<&Path>,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    if input_dir.is_none() && manifest_path.is_none() {
        return Err(Rdna2PackError::InvalidInputDir(
            "at least one input directory or manifest is required".to_string(),
        ));
    }

    let mut state = PackBuildState::new();

    if let Some(manifest) = manifest_path {
        pack_manifest_into_map(manifest, options, &mut state)?;
    }

    if let Some(dir) = input_dir {
        let (blobs, read_failures) = collect_directory_blobs(dir)?;
        state.total_blobs += blobs.len() + read_failures;
        state.skipped_or_failed += read_failures;
        state.failed += read_failures;
        pack_blobs_into_map(&blobs, options, &mut state);
    }

    build_pack_result(state, options)
}

/// Convenience helper that runs [`pack_rdna2_sources`] and writes the resulting
/// `WGPK` v1 `.wgsl.pack` bytes to `out_path` (creating parent directories if needed).
pub fn pack_rdna2_sources_to_file(
    input_dir: Option<&Path>,
    manifest_path: Option<&Path>,
    out_path: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let result = pack_rdna2_sources(input_dir, manifest_path, options)?;
    write_pack_bytes_to_file(out_path, &result.pack_bytes)?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// S71 (#1742, Refs #482, #651, #663): Auto-WGPK-v2 Policy, Zero-OOM Streaming
// WGPK v1/v2 Writer, and Sidecar v3 Stage Completeness
// ---------------------------------------------------------------------------

/// Default 4 GiB WGPK v1 byte ceiling (`u32::MAX`) above which `PackVersion::Auto` upgrades to v2.
pub const DEFAULT_AUTO_V2_THRESHOLD_BYTES: u64 = u32::MAX as u64;

/// Recommended safety margin below 4 GiB (64 MiB) for proactive auto-WGPK-v2 selection when
/// packing multi-GB shader archives.
pub const DEFAULT_AUTO_V2_SAFETY_MARGIN_BYTES: u64 = 64 * 1024 * 1024;

/// Fixed streaming I/O copy buffer size (64 KiB) used when flushing spilled WGSL payloads to disk
/// so peak RAM stays O(1) during final archive assembly even for 8+ GiB packs.
const STREAMING_COPY_BUF_BYTES: usize = 64 * 1024;

/// Default maximum number of shaders translated concurrently in one streaming batch before
/// spilling WGSL payloads to the temporary data file and freeing RAM.
pub const DEFAULT_STREAMING_BATCH_SIZE: usize = 64;

/// Configurable threshold and margin controlling automatic WGPK v1 -> v2 upgrade when
/// `PackVersion::Auto` is selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoVersionPolicy {
    /// Maximum total WGPK v1 archive size (header + TOC + payloads) before upgrading to WGPK v2.
    /// Clamped to `u32::MAX` (`4_294_967_295` bytes).
    pub v2_threshold_bytes: u64,
    /// Safety margin subtracted from `v2_threshold_bytes` so packs approaching 4 GiB upgrade to
    /// WGPK v2 proactively.
    pub v2_margin_bytes: u64,
}

impl Default for AutoVersionPolicy {
    fn default() -> Self {
        Self {
            v2_threshold_bytes: DEFAULT_AUTO_V2_THRESHOLD_BYTES,
            v2_margin_bytes: 0,
        }
    }
}

impl AutoVersionPolicy {
    /// Policy with an explicit threshold (useful for custom pack ceilings and synthetic tests).
    pub const fn with_threshold(v2_threshold_bytes: u64) -> Self {
        Self {
            v2_threshold_bytes,
            v2_margin_bytes: 0,
        }
    }

    /// Production 4 GiB policy with a safety margin (e.g. `DEFAULT_AUTO_V2_SAFETY_MARGIN_BYTES`).
    pub const fn with_safety_margin(v2_margin_bytes: u64) -> Self {
        Self {
            v2_threshold_bytes: DEFAULT_AUTO_V2_THRESHOLD_BYTES,
            v2_margin_bytes,
        }
    }

    /// Effective byte ceiling for selecting WGPK v1 under `PackVersion::Auto`.
    pub const fn effective_v1_limit(self) -> u64 {
        let clamped = if self.v2_threshold_bytes > DEFAULT_AUTO_V2_THRESHOLD_BYTES {
            DEFAULT_AUTO_V2_THRESHOLD_BYTES
        } else {
            self.v2_threshold_bytes
        };
        clamped.saturating_sub(self.v2_margin_bytes)
    }
}

/// Resolve the WGPK container version (`1` or `2`) and exact header/TOC/total byte layout using
/// `policy` whenever `requested == PackVersion::Auto`.
///
/// * `PackVersion::Auto`: chooses v1 iff `v1_total <= policy.effective_v1_limit()` and
///   `entry_count <= u32::MAX`; otherwise upgrades automatically to WGPK v2 (24-byte TOC, 64-bit offsets).
/// * `PackVersion::V1`: enforces `u32::MAX` hard limit.
/// * `PackVersion::V2`: always uses WGPK v2.
pub fn resolve_auto_pack_layout(
    requested: PackVersion,
    entry_count: usize,
    total_payload_bytes: u64,
    policy: AutoVersionPolicy,
) -> Result<ResolvedPackLayout, PackError> {
    let count_u32 = u32::try_from(entry_count)
        .map_err(|_| PackError::Overflow("entry count exceeds u32::MAX".to_string()))?;
    let v1_table = u64::from(count_u32) * (PACK_ENTRY_BYTES as u64);
    let v1_data_start = (PACK_HEADER_BYTES as u64) + v1_table;
    let v1_total = v1_data_start.checked_add(total_payload_bytes);

    match requested {
        PackVersion::V1 => resolve_pack_layout(PackVersion::V1, entry_count, total_payload_bytes),
        PackVersion::V2 => resolve_pack_layout(PackVersion::V2, entry_count, total_payload_bytes),
        PackVersion::Auto => {
            let v1_limit = policy.effective_v1_limit();
            if let Some(total) = v1_total
                && total <= v1_limit
            {
                Ok(ResolvedPackLayout {
                    version: PACK_VERSION_V1,
                    entry_bytes: PACK_ENTRY_BYTES,
                    data_start: v1_data_start,
                    total_size: total,
                })
            } else {
                resolve_pack_layout(PackVersion::V2, entry_count, total_payload_bytes)
            }
        }
    }
}

/// Summary returned by [`StreamingWgpkWriter::finish`] and [`write_wgpk_entries_streaming`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingWriteSummary {
    pub version: u32,
    pub flags: u32,
    pub entry_count: usize,
    pub data_start: u64,
    pub total_bytes: u64,
    pub total_payload_bytes: u64,
    pub toc_entries: Vec<TocEntry>,
}

/// Zero-OOM streaming WGPK v1/v2 archive writer operating directly on any `Write + Seek` sink
/// (such as `BufWriter<File>` or `Cursor<Vec<u8>>`).
///
/// Reserves the exact 16-byte header + TOC region (`data_start` bytes) up front, streams each
/// WGSL entry payload sequentially to the sink without retaining payload bytes in RAM, enforces
/// strictly ascending canonical 64-bit hash order, and rewinds to offset `0` in [`Self::finish`]
/// to write the finalized WGPK v1 or v2 header and TOC.
pub struct StreamingWgpkWriter<W: Write + Seek> {
    writer: W,
    version: u32,
    flags: u32,
    expected_entries: usize,
    data_start: u64,
    current_offset: u64,
    total_payload_bytes: u64,
    prev_hash: Option<u64>,
    toc_entries: Vec<TocEntry>,
}

impl<W: Write + Seek> StreamingWgpkWriter<W> {
    /// Start a streaming WGPK write using a pre-resolved [`ResolvedPackLayout`].
    ///
    /// Writes a zeroed header+TOC reservation of `layout.data_start` bytes so payloads stream
    /// directly starting at `layout.data_start` without keeping payload bytes in memory.
    pub fn begin_with_layout(
        mut writer: W,
        flags: u32,
        layout: ResolvedPackLayout,
        expected_entries: usize,
    ) -> Result<Self, Rdna2PackError> {
        let data_start = checked_table_data_start_for_version(layout.version, expected_entries)
            .map_err(Rdna2PackError::Pack)?;
        if data_start != layout.data_start {
            return Err(Rdna2PackError::Pack(PackError::Overflow(format!(
                "layout data_start mismatch: expected {data_start}, got {}",
                layout.data_start
            ))));
        }

        // Reserve header + TOC bytes sequentially using a small fixed zero buffer.
        writer
            .seek(SeekFrom::Start(0))
            .map_err(|e| Rdna2PackError::Io(format!("failed to seek stream start: {}", e.kind())))?;
        let zero_chunk = [0u8; 4096];
        let mut remaining = data_start;
        while remaining > 0 {
            let n = remaining.min(zero_chunk.len() as u64) as usize;
            writer.write_all(&zero_chunk[..n]).map_err(|e| {
                Rdna2PackError::Io(format!("failed to reserve WGPK TOC space: {}", e.kind()))
            })?;
            remaining -= n as u64;
        }

        Ok(Self {
            writer,
            version: layout.version,
            flags,
            expected_entries,
            data_start,
            current_offset: data_start,
            total_payload_bytes: 0,
            prev_hash: None,
            toc_entries: Vec::with_capacity(expected_entries),
        })
    }

    /// Start a streaming WGPK write with automatic v1/v2 selection via [`resolve_auto_pack_layout`].
    pub fn begin_auto(
        writer: W,
        flags: u32,
        requested_version: PackVersion,
        policy: AutoVersionPolicy,
        expected_entries: usize,
        projected_payload_bytes: u64,
    ) -> Result<Self, Rdna2PackError> {
        let layout = resolve_auto_pack_layout(
            requested_version,
            expected_entries,
            projected_payload_bytes,
            policy,
        )
        .map_err(Rdna2PackError::Pack)?;
        Self::begin_with_layout(writer, flags, layout, expected_entries)
    }

    pub const fn version(&self) -> u32 {
        self.version
    }

    pub fn entries_written(&self) -> usize {
        self.toc_entries.len()
    }

    pub const fn bytes_written(&self) -> u64 {
        self.current_offset
    }

    /// Stream one shader payload directly to the underlying sink.
    ///
    /// Enforces strictly ascending canonical `hash` order, valid UTF-8 payload, exact entry count
    /// bounds, and v1 `u32::MAX` / v2 `u64::MAX` offset checks.
    pub fn write_entry(
        &mut self,
        hash: u64,
        stage: ShaderStage,
        wgsl_bytes: &[u8],
    ) -> Result<TocEntry, Rdna2PackError> {
        let entry_index = self.toc_entries.len();
        if entry_index >= self.expected_entries {
            return Err(Rdna2PackError::Pack(PackError::Overflow(format!(
                "streaming writer exceeded reserved entry count ({})",
                self.expected_entries
            ))));
        }

        if let Some(previous_hash) = self.prev_hash {
            if hash == previous_hash {
                return Err(Rdna2PackError::Pack(PackError::DuplicateHash {
                    first_index: entry_index - 1,
                    second_index: entry_index,
                    hash,
                }));
            }
            if hash < previous_hash {
                return Err(Rdna2PackError::Pack(PackError::NonCanonicalOrder {
                    entry_index,
                    previous_hash,
                    hash,
                }));
            }
        }

        if let Err(err) = std::str::from_utf8(wgsl_bytes) {
            return Err(Rdna2PackError::Pack(PackError::InvalidUtf8 {
                entry_index,
                detail: err.to_string(),
            }));
        }

        let length = u32::try_from(wgsl_bytes.len()).map_err(|_| {
            Rdna2PackError::Pack(PackError::Overflow(format!(
                "entry {entry_index} payload length {} exceeds u32::MAX",
                wgsl_bytes.len()
            )))
        })?;
        let offset = self.current_offset;
        let next_offset = offset.checked_add(u64::from(length)).ok_or_else(|| {
            Rdna2PackError::Pack(PackError::Overflow(format!(
                "entry {entry_index} payload [{offset}..+{length}] overflows u64"
            )))
        })?;
        if self.version == PACK_VERSION_V1 && next_offset > u64::from(u32::MAX) {
            return Err(Rdna2PackError::Pack(PackError::Overflow(format!(
                "WGPK v1 entry {entry_index} payload [{offset}..+{length}] exceeds u32::MAX"
            ))));
        }

        self.writer.write_all(wgsl_bytes).map_err(|e| {
            Rdna2PackError::Io(format!(
                "failed to stream WGSL payload for entry {entry_index}: {}",
                e.kind()
            ))
        })?;

        let toc = TocEntry {
            hash,
            stage,
            offset,
            length,
        };
        self.toc_entries.push(toc);
        self.prev_hash = Some(hash);
        self.current_offset = next_offset;
        self.total_payload_bytes = self.total_payload_bytes.saturating_add(u64::from(length));
        Ok(toc)
    }

    /// Stream one shader payload from any [`Read`] source of exact byte length `length` using a
    /// fixed 64 KiB stack/heap copy buffer (never allocates a full-entry buffer for huge payloads).
    pub fn write_entry_from_reader<R: Read>(
        &mut self,
        hash: u64,
        stage: ShaderStage,
        length: u32,
        mut reader: R,
    ) -> Result<TocEntry, Rdna2PackError> {
        let entry_index = self.toc_entries.len();
        if entry_index >= self.expected_entries {
            return Err(Rdna2PackError::Pack(PackError::Overflow(format!(
                "streaming writer exceeded reserved entry count ({})",
                self.expected_entries
            ))));
        }
        if let Some(previous_hash) = self.prev_hash {
            if hash == previous_hash {
                return Err(Rdna2PackError::Pack(PackError::DuplicateHash {
                    first_index: entry_index - 1,
                    second_index: entry_index,
                    hash,
                }));
            }
            if hash < previous_hash {
                return Err(Rdna2PackError::Pack(PackError::NonCanonicalOrder {
                    entry_index,
                    previous_hash,
                    hash,
                }));
            }
        }

        let offset = self.current_offset;
        let next_offset = offset.checked_add(u64::from(length)).ok_or_else(|| {
            Rdna2PackError::Pack(PackError::Overflow(format!(
                "entry {entry_index} payload [{offset}..+{length}] overflows u64"
            )))
        })?;
        if self.version == PACK_VERSION_V1 && next_offset > u64::from(u32::MAX) {
            return Err(Rdna2PackError::Pack(PackError::Overflow(format!(
                "WGPK v1 entry {entry_index} payload [{offset}..+{length}] exceeds u32::MAX"
            ))));
        }

        let mut copy_buf = [0u8; STREAMING_COPY_BUF_BYTES];
        let mut remaining = length as usize;
        while remaining > 0 {
            let chunk_len = remaining.min(copy_buf.len());
            reader.read_exact(&mut copy_buf[..chunk_len]).map_err(|e| {
                Rdna2PackError::Io(format!(
                    "failed to read spilled payload for entry {entry_index}: {}",
                    e.kind()
                ))
            })?;
            self.writer.write_all(&copy_buf[..chunk_len]).map_err(|e| {
                Rdna2PackError::Io(format!(
                    "failed to stream payload for entry {entry_index}: {}",
                    e.kind()
                ))
            })?;
            remaining -= chunk_len;
        }

        let toc = TocEntry {
            hash,
            stage,
            offset,
            length,
        };
        self.toc_entries.push(toc);
        self.prev_hash = Some(hash);
        self.current_offset = next_offset;
        self.total_payload_bytes = self.total_payload_bytes.saturating_add(u64::from(length));
        Ok(toc)
    }

    /// Seek back to offset `0`, write the canonical WGPK v1 or v2 header + TOC, and flush.
    pub fn finish(mut self) -> Result<StreamingWriteSummary, Rdna2PackError> {
        if self.toc_entries.len() != self.expected_entries {
            return Err(Rdna2PackError::Pack(PackError::Overflow(format!(
                "streaming writer finished with {} entries, expected {}",
                self.toc_entries.len(),
                self.expected_entries
            ))));
        }

        let toc_bytes = encode_pack_toc(self.version, self.flags, &self.toc_entries)
            .map_err(Rdna2PackError::Pack)?;
        debug_assert_eq!(toc_bytes.len() as u64, self.data_start);

        self.writer
            .seek(SeekFrom::Start(0))
            .map_err(|e| Rdna2PackError::Io(format!("failed to seek to WGPK header: {}", e.kind())))?;
        self.writer.write_all(&toc_bytes).map_err(|e| {
            Rdna2PackError::Io(format!("failed to write WGPK header/TOC: {}", e.kind()))
        })?;
        self.writer
            .flush()
            .map_err(|e| Rdna2PackError::Io(format!("failed to flush WGPK stream: {}", e.kind())))?;

        Ok(StreamingWriteSummary {
            version: self.version,
            flags: self.flags,
            entry_count: self.toc_entries.len(),
            data_start: self.data_start,
            total_bytes: self.current_offset,
            total_payload_bytes: self.total_payload_bytes,
            toc_entries: self.toc_entries,
        })
    }
}

/// Stream a slice of [`PackEntry`] values directly to any `Write + Seek` sink in canonical
/// ascending 64-bit hash order, using [`AutoVersionPolicy`] to select WGPK v1 vs v2 automatically.
pub fn write_wgpk_entries_streaming<W: Write + Seek>(
    writer: W,
    entries: &[PackEntry],
    flags: u32,
    requested_version: PackVersion,
    policy: AutoVersionPolicy,
) -> Result<StreamingWriteSummary, Rdna2PackError> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by_key(|&idx| entries[idx].hash);

    // Reject duplicate hashes before touching the sink.
    for pair in order.windows(2) {
        let a = pair[0];
        let b = pair[1];
        if entries[a].hash == entries[b].hash {
            return Err(Rdna2PackError::Pack(PackError::DuplicateHash {
                first_index: a.min(b),
                second_index: a.max(b),
                hash: entries[a].hash,
            }));
        }
    }

    let mut total_payload_bytes = 0u64;
    for &idx in &order {
        let len = entries[idx].wgsl.len();
        let len_u32 = u32::try_from(len).map_err(|_| {
            Rdna2PackError::Pack(PackError::Overflow(format!(
                "entry {idx} payload length {len} exceeds u32::MAX"
            )))
        })?;
        total_payload_bytes = total_payload_bytes
            .checked_add(u64::from(len_u32))
            .ok_or_else(|| {
                Rdna2PackError::Pack(PackError::Overflow(
                    "total payload size overflows u64".to_string(),
                ))
            })?;
    }

    let mut stream = StreamingWgpkWriter::begin_auto(
        writer,
        flags,
        requested_version,
        policy,
        order.len(),
        total_payload_bytes,
    )?;
    for &idx in &order {
        let entry = &entries[idx];
        stream.write_entry(entry.hash, entry.stage, entry.wgsl.as_bytes())?;
    }
    stream.finish()
}

/// Options controlling zero-OOM streaming batch packing (`pack_rdna2_*_streaming_to_file`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2StreamingOptions {
    pub pack: Rdna2PackOptions,
    /// Policy for automatic WGPK v1 -> v2 upgrade when `pack.pack_version == PackVersion::Auto`.
    pub auto_version_policy: AutoVersionPolicy,
    /// Maximum shaders translated per memory-bounded batch before spilling WGSL to disk.
    pub batch_size: usize,
}

impl Default for Rdna2StreamingOptions {
    fn default() -> Self {
        Self {
            pack: Rdna2PackOptions::default(),
            auto_version_policy: AutoVersionPolicy::default(),
            batch_size: DEFAULT_STREAMING_BATCH_SIZE,
        }
    }
}

/// Result of zero-OOM streaming batch packing to disk (holds ZERO WGSL payload bytes in RAM).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2StreamingPackSummary {
    pub summary: Rdna2PackSummary,
    /// Exact written `.wgsl.pack` file size in bytes (64-bit safe even on >4 GiB archives).
    pub pack_bytes_u64: u64,
    /// Finalized TOC entries written to the archive in canonical ascending hash order.
    pub toc_entries: Vec<TocEntry>,
    /// `true` when `PackVersion::Auto` upgraded this archive from WGPK v1 to WGPK v2.
    pub auto_upgraded_to_v2: bool,
    /// Peak WGSL bytes held in RAM during any single translation batch.
    pub peak_batch_wgsl_bytes: usize,
}

/// RAII cleanup guard ensuring temporary payload spill files are always deleted on exit.
struct TempSpillGuard {
    path: PathBuf,
}

impl Drop for TempSpillGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

struct SpilledCandidate {
    source_order: usize,
    stage: ShaderStage,
    spill_offset: u64,
    wgsl_len: u32,
    entry_summary: Rdna2PackEntrySummary,
}

struct StreamingSpillState {
    spill_writer: BufWriter<File>,
    spill_cursor: u64,
    candidates_by_hash: BTreeMap<u64, SpilledCandidate>,
    next_source_order: usize,
    total_blobs: usize,
    skipped_or_failed: usize,
    failed: usize,
    manifest_duplicates_skipped: usize,
    skipped_oversize: usize,
    skipped_budget: usize,
    peak_batch_wgsl_bytes: usize,
    failures: Vec<Rdna2PackFailureSummary>,
}

impl StreamingSpillState {
    fn new(spill_file: File) -> Self {
        Self {
            spill_writer: BufWriter::with_capacity(STREAMING_COPY_BUF_BYTES, spill_file),
            spill_cursor: 0,
            candidates_by_hash: BTreeMap::new(),
            next_source_order: 0,
            total_blobs: 0,
            skipped_or_failed: 0,
            failed: 0,
            manifest_duplicates_skipped: 0,
            skipped_oversize: 0,
            skipped_budget: 0,
            peak_batch_wgsl_bytes: 0,
            failures: Vec::new(),
        }
    }

    fn record_spilled(
        &mut self,
        hash: u64,
        source_order: usize,
        data: TranslatedStageData,
        options: &Rdna2PackOptions,
    ) -> Result<(), Rdna2PackError> {
        let wgsl_len_usize = data.wgsl.len();
        if let Some(max_wgsl) = options.max_wgsl_bytes
            && wgsl_len_usize > max_wgsl
        {
            self.skipped_oversize += 1;
            self.skipped_or_failed += 1;
            return Ok(());
        }
        let wgsl_len = u32::try_from(wgsl_len_usize).map_err(|_| {
            Rdna2PackError::Pack(PackError::Overflow(format!(
                "shader {} WGSL length {wgsl_len_usize} exceeds u32::MAX",
                format_shader_hash(hash)
            )))
        })?;
        let spill_offset = self.spill_cursor;
        self.spill_writer
            .write_all(data.wgsl.as_bytes())
            .map_err(|e| {
                Rdna2PackError::Io(format!("failed to write temporary spill file: {}", e.kind()))
            })?;
        self.spill_cursor = self
            .spill_cursor
            .checked_add(u64::from(wgsl_len))
            .ok_or_else(|| {
                Rdna2PackError::Pack(PackError::Overflow("spill offset overflow".to_string()))
            })?;

        let (_entry, entry_summary) = data.into_pack_parts(hash);
        // `PackEntry` (and its `String` payload) is dropped immediately here -> zero payload RAM!
        self.candidates_by_hash.insert(
            hash,
            SpilledCandidate {
                source_order,
                stage: entry_summary.stage,
                spill_offset,
                wgsl_len,
                entry_summary,
            },
        );
        Ok(())
    }
}

fn finalize_streaming_pack(
    mut state: StreamingSpillState,
    spill_path: &Path,
    out_path: &Path,
    streaming_opts: &Rdna2StreamingOptions,
) -> Result<Rdna2StreamingPackSummary, Rdna2PackError> {
    state.spill_writer.flush().map_err(|e| {
        Rdna2PackError::Io(format!("failed to flush temporary spill file: {}", e.kind()))
    })?;
    drop(state.spill_writer);

    let options = &streaming_opts.pack;
    let policy = streaming_opts.auto_version_policy;

    let mut priority_rank: HashMap<u64, usize> =
        HashMap::with_capacity(options.priority_hashes.len());
    for &hash in &options.priority_hashes {
        let rank = priority_rank.len();
        priority_rank.entry(hash).or_insert(rank);
    }
    let priority_hashes_requested = priority_rank.len();

    let selected_by_hash: BTreeMap<u64, SpilledCandidate> =
        if let Some(max_pack) = options.max_pack_bytes {
            let (effective_budget, entry_record_bytes) = match options.pack_version {
                PackVersion::V1 => (max_pack.min(u64::from(u32::MAX)), PACK_ENTRY_BYTES as u64),
                PackVersion::V2 => (max_pack, PACK_ENTRY_BYTES_V2 as u64),
                PackVersion::Auto => {
                    let v1_limit = policy.effective_v1_limit();
                    let all_wgsl: u64 = state
                        .candidates_by_hash
                        .values()
                        .fold(0u64, |acc, c| acc.saturating_add(u64::from(c.wgsl_len)));
                    let all_v1_size = (PACK_HEADER_BYTES as u64)
                        .saturating_add(
                            (state.candidates_by_hash.len() as u64)
                                .saturating_mul(PACK_ENTRY_BYTES as u64),
                        )
                        .saturating_add(all_wgsl);
                    if max_pack <= v1_limit && all_v1_size <= v1_limit {
                        (max_pack, PACK_ENTRY_BYTES as u64)
                    } else if all_v1_size <= v1_limit {
                        (max_pack, PACK_ENTRY_BYTES as u64)
                    } else {
                        (max_pack, PACK_ENTRY_BYTES_V2 as u64)
                    }
                }
            };

            let mut priority_candidates: Vec<(usize, u64, SpilledCandidate)> = Vec::new();
            let mut regular_candidates: Vec<(u64, SpilledCandidate)> =
                Vec::with_capacity(state.candidates_by_hash.len());
            for (hash, cand) in state.candidates_by_hash {
                if let Some(&rank) = priority_rank.get(&hash) {
                    priority_candidates.push((rank, hash, cand));
                } else {
                    regular_candidates.push((hash, cand));
                }
            }

            priority_candidates.sort_by_key(|(rank, ..)| *rank);
            match options.budget_order {
                BudgetOrder::HashOrder => {}
                BudgetOrder::SmallestFirst => {
                    regular_candidates
                        .sort_by(|a, b| a.1.wgsl_len.cmp(&b.1.wgsl_len).then_with(|| a.0.cmp(&b.0)));
                }
                BudgetOrder::ManifestOrder => {
                    regular_candidates.sort_by(|a, b| {
                        a.1.source_order
                            .cmp(&b.1.source_order)
                            .then_with(|| a.0.cmp(&b.0))
                    });
                }
            }

            let mut current_pack_bytes = PACK_HEADER_BYTES as u64;
            let mut selected: BTreeMap<u64, SpilledCandidate> = BTreeMap::new();
            for (_rank, hash, cand) in priority_candidates {
                let entry_cost = entry_record_bytes.saturating_add(u64::from(cand.wgsl_len));
                let projected = current_pack_bytes.saturating_add(entry_cost);
                if projected <= effective_budget {
                    current_pack_bytes = projected;
                    selected.insert(hash, cand);
                } else {
                    state.skipped_budget += 1;
                    state.skipped_or_failed += 1;
                }
            }

            let mut budget_exhausted = false;
            for (hash, cand) in regular_candidates {
                let entry_cost = entry_record_bytes.saturating_add(u64::from(cand.wgsl_len));
                let projected = current_pack_bytes.saturating_add(entry_cost);
                if !budget_exhausted && projected <= effective_budget {
                    current_pack_bytes = projected;
                    selected.insert(hash, cand);
                } else {
                    if matches!(
                        options.budget_order,
                        BudgetOrder::HashOrder | BudgetOrder::SmallestFirst
                    ) {
                        budget_exhausted = true;
                    }
                    state.skipped_budget += 1;
                    state.skipped_or_failed += 1;
                }
            }
            selected
        } else {
            state.candidates_by_hash
        };

    let mut vertex_count = 0usize;
    let mut pixel_count = 0usize;
    let mut compute_count = 0usize;
    let mut largest_wgsl_bytes = 0usize;
    let mut total_wgsl_bytes = 0u64;
    let mut priority_hashes_packed = 0usize;

    for (&hash, cand) in &selected_by_hash {
        if priority_rank.contains_key(&hash) {
            priority_hashes_packed += 1;
        }
        match cand.stage {
            ShaderStage::Vertex => vertex_count += 1,
            ShaderStage::Fragment => pixel_count += 1,
            ShaderStage::Compute => compute_count += 1,
        }
        let len = cand.wgsl_len as usize;
        largest_wgsl_bytes = largest_wgsl_bytes.max(len);
        total_wgsl_bytes = total_wgsl_bytes.saturating_add(u64::from(cand.wgsl_len));
    }

    let layout = resolve_auto_pack_layout(
        options.pack_version,
        selected_by_hash.len(),
        total_wgsl_bytes,
        policy,
    )
    .map_err(Rdna2PackError::Pack)?;
    let auto_upgraded_to_v2 =
        options.pack_version == PackVersion::Auto && layout.version == PACK_VERSION_V2;

    let out_label = sanitize_source_path(out_path, None);
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            Rdna2PackError::Io(format!(
                "failed to create output directory for '{out_label}': {}",
                err.kind()
            ))
        })?;
    }
    let out_file = File::create(out_path).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to create WGPK pack '{out_label}': {}",
            err.kind()
        ))
    })?;
    let out_writer = BufWriter::with_capacity(STREAMING_COPY_BUF_BYTES, out_file);
    let mut spill_reader = BufReader::with_capacity(
        STREAMING_COPY_BUF_BYTES,
        File::open(spill_path).map_err(|err| {
            Rdna2PackError::Io(format!(
                "failed to reopen temporary spill file: {}",
                err.kind()
            ))
        })?,
    );

    let mut stream = StreamingWgpkWriter::begin_with_layout(
        out_writer,
        options.pack_flags,
        layout,
        selected_by_hash.len(),
    )?;

    let mut entry_summaries = Vec::with_capacity(selected_by_hash.len());
    for (hash, cand) in selected_by_hash {
        spill_reader
            .seek(SeekFrom::Start(cand.spill_offset))
            .map_err(|e| {
                Rdna2PackError::Io(format!("failed to seek spill payload: {}", e.kind()))
            })?;
        stream.write_entry_from_reader(hash, cand.stage, cand.wgsl_len, &mut spill_reader)?;
        entry_summaries.push(cand.entry_summary);
    }

    let write_summary = stream.finish()?;

    let mut binding_limits = PackBindingLimitSummary::default();
    let d3d12_limits = StageBindingLimits::webgpu_d3d12();
    for es in &entry_summaries {
        binding_limits.record_wgsl_stage(
            &es.bindings,
            es.rt_arena,
            es.mesh_output.is_some(),
            es.binding_pack,
            d3d12_limits,
        );
    }

    let summary = Rdna2PackSummary {
        pack_version: write_summary.version,
        total_blobs: state.total_blobs,
        translated_ok: entry_summaries.len(),
        skipped_or_failed: state.skipped_or_failed,
        failed: state.failed,
        manifest_duplicates_skipped: state.manifest_duplicates_skipped,
        budget_order: options.budget_order,
        skipped_oversize: state.skipped_oversize,
        skipped_budget: state.skipped_budget,
        omitted_by_budget: state.skipped_budget,
        priority_hashes_requested,
        priority_hashes_packed,
        largest_wgsl_bytes,
        total_wgsl_bytes,
        vertex_count,
        pixel_count,
        compute_count,
        vertex_packed: vertex_count,
        fragment_packed: pixel_count,
        compute_packed: compute_count,
        pack_bytes: usize::try_from(write_summary.total_bytes).unwrap_or(usize::MAX),
        binding_limits,
        entries: entry_summaries,
        failures: state.failures,
    };

    Ok(Rdna2StreamingPackSummary {
        summary,
        pack_bytes_u64: write_summary.total_bytes,
        toc_entries: write_summary.toc_entries,
        auto_upgraded_to_v2,
        peak_batch_wgsl_bytes: state.peak_batch_wgsl_bytes,
    })
}

fn create_spill_file(out_path: &Path) -> Result<(PathBuf, File, TempSpillGuard), Rdna2PackError> {
    let out_label = sanitize_source_path(out_path, None);
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            Rdna2PackError::Io(format!(
                "failed to create output directory for '{out_label}': {}",
                err.kind()
            ))
        })?;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let file_name = out_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("pack");
    let spill_name = format!(".{file_name}.{}.{nanos}.wgpk_spill.tmp", std::process::id());
    let spill_path = match out_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(spill_name),
        _ => PathBuf::from(spill_name),
    };
    let spill_file = File::create(&spill_path).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to create temporary spill file: {}",
            err.kind()
        ))
    })?;
    let guard = TempSpillGuard {
        path: spill_path.clone(),
    };
    Ok((spill_path, spill_file, guard))
}

fn stream_blobs_into_spill(
    blobs: &[Rdna2BlobInput],
    options: &Rdna2PackOptions,
    batch_size: usize,
    state: &mut StreamingSpillState,
) -> Result<(), Rdna2PackError> {
    let source_order_base = state.next_source_order;
    state.next_source_order = state.next_source_order.saturating_add(blobs.len());

    let mut order: Vec<usize> = (0..blobs.len()).collect();
    order.sort_by(|&a, &b| {
        blobs[a]
            .name
            .cmp(&blobs[b].name)
            .then_with(|| blobs[a].bytes.len().cmp(&blobs[b].bytes.len()))
            .then_with(|| blobs[a].bytes.cmp(&blobs[b].bytes))
    });

    let chunk_size = batch_size.max(1);
    for chunk in order.chunks(chunk_size) {
        let mut chunk_candidates: Vec<BlobCandidate<'_>> = Vec::with_capacity(chunk.len());
        let mut unique_tasks: Vec<BlobTranslateTask<'_>> = Vec::new();
        let mut task_uses: Vec<usize> = Vec::new();
        let mut dedup_map: BTreeMap<(u64, u8), Vec<usize>> = BTreeMap::new();

        for &idx in chunk {
            let blob = &blobs[idx];
            let hint = parse_filename_hint(&blob.name);
            let stage = if blob.stage_override != Rdna2StageOverride::Auto {
                blob.stage_override
            } else if hint.stage != Rdna2StageOverride::Auto {
                hint.stage
            } else {
                options.translate.stage
            };
            let explicit_hash = blob.hash_override.or(hint.hash);
            let content_hash = fnv1a_64(&blob.bytes);
            let base_hash = explicit_hash.unwrap_or(content_hash);
            let final_hash = if stage == Rdna2StageOverride::Mesh {
                mesh_compute_pack_key(base_hash)
            } else {
                base_hash
            };

            let work_idx = if blob.bytes.len() < MIN_RDNA2_BLOB_BYTES {
                None
            } else {
                let key = (content_hash, stage_override_key(stage));
                let bucket = dedup_map.entry(key).or_default();
                let existing = bucket.iter().copied().find(|&task_idx| {
                    unique_tasks[task_idx].bytes == blob.bytes.as_slice()
                        && unique_tasks[task_idx].fssm_meta == blob.fssm_meta.as_ref()
                });
                let assigned = match existing {
                    Some(task_idx) => task_idx,
                    None => {
                        let task_idx = unique_tasks.len();
                        unique_tasks.push(BlobTranslateTask {
                            bytes: &blob.bytes,
                            stage,
                            fssm_meta: blob.fssm_meta.as_ref(),
                        });
                        task_uses.push(0);
                        bucket.push(task_idx);
                        task_idx
                    }
                };
                task_uses[assigned] += 1;
                Some(assigned)
            };

            chunk_candidates.push(BlobCandidate {
                blob,
                source_order: source_order_base.saturating_add(idx),
                stage,
                explicit_hash,
                final_hash,
                work_idx,
            });
        }

        let mut outcomes: Vec<Option<TranslatedStageOutcome>> = run_parallel_indexed(
            &unique_tasks,
            options.effective_jobs(),
            |_idx, task| translate_blob_task(task, options),
            |_| false,
        );

        let batch_wgsl_bytes: usize = outcomes
            .iter()
            .filter_map(|o| match o {
                Some(TranslatedStageOutcome::Translated(d)) => Some(d.wgsl.len()),
                _ => None,
            })
            .sum();
        state.peak_batch_wgsl_bytes = state.peak_batch_wgsl_bytes.max(batch_wgsl_bytes);

        for candidate in chunk_candidates {
            let Some(work_idx) = candidate.work_idx else {
                state.skipped_or_failed += 1;
                state.failed += 1;
                state.failures.push(Rdna2PackFailureSummary {
                    code_identity: candidate.explicit_hash,
                    stage: override_to_stage(candidate.stage),
                    reason: format!(
                        "RDNA2 blob too short ({} bytes, minimum {MIN_RDNA2_BLOB_BYTES})",
                        candidate.blob.bytes.len()
                    ),
                });
                continue;
            };

            task_uses[work_idx] = task_uses[work_idx].saturating_sub(1);
            let remaining_uses = task_uses[work_idx];

            match outcomes[work_idx].as_ref() {
                Some(TranslatedStageOutcome::Failed { reason, .. }) => {
                    state.skipped_or_failed += 1;
                    state.failed += 1;
                    state.failures.push(Rdna2PackFailureSummary {
                        code_identity: Some(candidate.final_hash),
                        stage: override_to_stage(candidate.stage),
                        reason: reason.clone(),
                    });
                }
                Some(TranslatedStageOutcome::Oversize) => {
                    if state.candidates_by_hash.contains_key(&candidate.final_hash) {
                        state.skipped_or_failed += 1;
                    } else {
                        state.skipped_oversize += 1;
                        state.skipped_or_failed += 1;
                    }
                }
                Some(TranslatedStageOutcome::Translated(_)) => {
                    if let Some(existing) = state.candidates_by_hash.get_mut(&candidate.final_hash) {
                        existing.source_order = existing.source_order.min(candidate.source_order);
                        state.skipped_or_failed += 1;
                        continue;
                    }
                    let data = if remaining_uses == 0 {
                        match outcomes[work_idx].take() {
                            Some(TranslatedStageOutcome::Translated(d)) => d,
                            _ => unreachable!(),
                        }
                    } else {
                        match outcomes[work_idx].as_ref() {
                            Some(TranslatedStageOutcome::Translated(d)) => d.clone(),
                            _ => unreachable!(),
                        }
                    };
                    state.record_spilled(candidate.final_hash, candidate.source_order, data, options)?;
                }
                None => unreachable!("all blob translation tasks must produce an outcome"),
            }
        }
    }
    Ok(())
}

fn stream_manifest_into_spill(
    manifest_path: &Path,
    options: &Rdna2PackOptions,
    batch_size: usize,
    state: &mut StreamingSpillState,
) -> Result<(), Rdna2PackError> {
    let manifest_label = sanitize_source_path(manifest_path, None);
    let meta = fs::metadata(manifest_path).map_err(|_| {
        Rdna2PackError::InvalidManifest(format!(
            "manifest '{manifest_label}' does not exist or is unreadable"
        ))
    })?;
    if !meta.is_file() {
        return Err(Rdna2PackError::InvalidManifest(format!(
            "'{manifest_label}' is not a regular file"
        )));
    }
    let json_text = fs::read_to_string(manifest_path).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to read manifest '{manifest_label}': {}",
            err.kind()
        ))
    })?;
    let manifest = parse_pipeline_dump_manifest(&json_text)?;

    let parent_dir = match manifest_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let canonical_parent = fs::canonicalize(parent_dir).map_err(|_| {
        Rdna2PackError::Io(format!(
            "failed to resolve parent directory of manifest '{manifest_label}'"
        ))
    })?;

    let mut seen_pack_keys: BTreeSet<u64> = BTreeSet::new();
    let mut unique_stages: Vec<(usize, PipelineDumpNativeStage)> =
        Vec::with_capacity(manifest.native_stages.len());
    for entry in manifest.native_stages {
        let pack_key = if entry.is_mesh {
            mesh_compute_pack_key(entry.code_identity)
        } else {
            entry.code_identity
        };
        if !seen_pack_keys.insert(pack_key) {
            state.manifest_duplicates_skipped += 1;
            continue;
        }
        let source_order = state.next_source_order;
        state.next_source_order = state.next_source_order.saturating_add(1);
        unique_stages.push((source_order, entry));
    }
    unique_stages.sort_by_key(|(_, entry)| {
        if entry.is_mesh {
            mesh_compute_pack_key(entry.code_identity)
        } else {
            entry.code_identity
        }
    });
    state.total_blobs += unique_stages.len();

    let chunk_size = batch_size.max(1);
    for chunk in unique_stages.chunks(chunk_size) {
        let outcomes: Vec<Option<TranslatedStageOutcome>> = run_parallel_indexed(
            chunk,
            options.effective_jobs(),
            |_idx, (_, entry)| translate_manifest_stage(entry, &canonical_parent, options),
            |outcome| !options.best_effort && matches!(outcome, TranslatedStageOutcome::Failed { .. }),
        );

        let batch_wgsl_bytes: usize = outcomes
            .iter()
            .filter_map(|o| match o {
                Some(TranslatedStageOutcome::Translated(d)) => Some(d.wgsl.len()),
                _ => None,
            })
            .sum();
        state.peak_batch_wgsl_bytes = state.peak_batch_wgsl_bytes.max(batch_wgsl_bytes);

        for ((source_order, entry), outcome_opt) in chunk.iter().zip(outcomes.into_iter()) {
            match outcome_opt.expect("manifest stage before strict abort must have an outcome") {
                TranslatedStageOutcome::Failed { error, reason } => {
                    if options.best_effort {
                        state.skipped_or_failed += 1;
                        state.failed += 1;
                        state.failures.push(Rdna2PackFailureSummary {
                            code_identity: Some(entry.code_identity),
                            stage: Some(entry.stage),
                            reason,
                        });
                    } else {
                        return Err(
                            error.expect("manifest stage failure must include Rdna2PackError")
                        );
                    }
                }
                TranslatedStageOutcome::Oversize => {
                    state.skipped_oversize += 1;
                    state.skipped_or_failed += 1;
                }
                TranslatedStageOutcome::Translated(data) => {
                    let pack_key = if entry.is_mesh {
                        mesh_compute_pack_key(entry.code_identity)
                    } else {
                        entry.code_identity
                    };
                    state.record_spilled(pack_key, *source_order, data, options)?;
                }
            }
        }
    }

    Ok(())
}

/// Zero-OOM streaming batch packer for in-memory RDNA2 blobs, writing directly to `out_path` with
/// automatic WGPK v1 -> v2 upgrade controlled by `options.auto_version_policy`.
pub fn pack_rdna2_blobs_streaming_to_file(
    blobs: &[Rdna2BlobInput],
    out_path: &Path,
    options: &Rdna2StreamingOptions,
) -> Result<Rdna2StreamingPackSummary, Rdna2PackError> {
    let (spill_path, spill_file, _guard) = create_spill_file(out_path)?;
    let mut state = StreamingSpillState::new(spill_file);
    state.total_blobs = blobs.len();
    stream_blobs_into_spill(blobs, &options.pack, options.batch_size, &mut state)?;
    finalize_streaming_pack(state, &spill_path, out_path, options)
}

/// Zero-OOM streaming batch packer for RDNA2 directories and/or `playgta6-pipeline-dump`
/// manifests, translating in bounded batches, spilling WGSL to a temporary file, and streaming
/// the finalized WGPK v1/v2 archive directly to `out_path`.
pub fn pack_rdna2_sources_streaming_to_file(
    input_dir: Option<&Path>,
    manifest_path: Option<&Path>,
    out_path: &Path,
    options: &Rdna2StreamingOptions,
) -> Result<Rdna2StreamingPackSummary, Rdna2PackError> {
    if input_dir.is_none() && manifest_path.is_none() {
        return Err(Rdna2PackError::InvalidInputDir(
            "at least one input directory or manifest is required".to_string(),
        ));
    }

    let (spill_path, spill_file, _guard) = create_spill_file(out_path)?;
    let mut state = StreamingSpillState::new(spill_file);

    if let Some(manifest) = manifest_path {
        stream_manifest_into_spill(manifest, &options.pack, options.batch_size, &mut state)?;
    }

    if let Some(dir) = input_dir {
        let (blobs, read_failures) = collect_directory_blobs(dir)?;
        state.total_blobs += blobs.len() + read_failures;
        state.skipped_or_failed += read_failures;
        state.failed += read_failures;
        stream_blobs_into_spill(&blobs, &options.pack, options.batch_size, &mut state)?;
    }

    finalize_streaming_pack(state, &spill_path, out_path, options)
}

/// Zero-OOM streaming convenience wrapper for packing a directory directly to `out_path`.
pub fn pack_rdna2_directory_streaming_to_file(
    input_dir: &Path,
    out_path: &Path,
    options: &Rdna2StreamingOptions,
) -> Result<Rdna2StreamingPackSummary, Rdna2PackError> {
    pack_rdna2_sources_streaming_to_file(Some(input_dir), None, out_path, options)
}

/// Zero-OOM streaming convenience wrapper for packing a `playgta6-pipeline-dump` manifest directly
/// to `out_path`.
pub fn pack_rdna2_manifest_streaming_to_file(
    manifest_path: &Path,
    out_path: &Path,
    options: &Rdna2StreamingOptions,
) -> Result<Rdna2StreamingPackSummary, Rdna2PackError> {
    pack_rdna2_sources_streaming_to_file(None, Some(manifest_path), out_path, options)
}

impl Rdna2PackSummary {
    /// Build a `playgta6-stage-reflection` v1/v2/v3 sidecar directly from this pack summary,
    /// explicitly accounting for all vertex, fragment, AND compute (including mesh-compute)
    /// stages so compute stages are never silently dropped.
    pub fn to_stage_reflection_sidecar(
        &self,
        options: &SidecarBuildOptions,
    ) -> StageReflectionSidecar {
        let inputs: Vec<SidecarStageInput<'_>> = self
            .entries
            .iter()
            .map(|e| SidecarStageInput {
                hash: e.hash_fnv1a64,
                stage: e.stage,
                bindings: &e.bindings,
                vertex_inputs: &e.vertex_inputs,
                vertex_input_mode: &e.vertex_input_mode,
                vs_outputs: &e.vs_outputs,
                ps_inputs: &e.ps_inputs,
                mesh_output: e.mesh_output,
                rt_arena: e.rt_arena,
            })
            .collect();
        StageReflectionSidecar::build(&inputs, options)
    }

    /// Render the `playgta6-stage-reflection` v1/v2/v3 sidecar JSON for this pack summary.
    pub fn to_stage_reflection_sidecar_json(&self, options: &SidecarBuildOptions) -> String {
        self.to_stage_reflection_sidecar(options).render_json()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use crate::convert::fnv1a_64;
    use crate::pack::{PACK_MAGIC, PACK_VERSION, read_pack};

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("{prefix}_{}_{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(words.len() * 4);
        for &w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn enc_sopp(op: u8, simm16: u16) -> u32 {
        (0b1_0111_1111_u32 << 23) | (u32::from(op & 0x7F) << 16) | u32::from(simm16)
    }

    fn enc_vop1(op: u8, vdst: u8, src0: u16) -> u32 {
        (0b011_1111_u32 << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(op) << 9)
            | u32::from(src0 & 0x1FF)
    }

    fn enc_vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        (u32::from(op & 0x3F) << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(vsrc1) << 9)
            | u32::from(src0 & 0x1FF)
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_vop3(
        op: u16,
        vdst: u8,
        src0: u16,
        src1: u16,
        src2: u16,
        clamp: bool,
        omod: u8,
        abs: u8,
        neg: u8,
    ) -> [u32; 2] {
        let w0 = (0b110101u32 << 26)
            | (u32::from(op & 0x3FF) << 16)
            | (u32::from(clamp) << 15)
            | (u32::from(abs & 0x7) << 8)
            | u32::from(vdst);
        let w1 = (u32::from(neg & 0x7) << 29)
            | (u32::from(omod & 0x3) << 27)
            | (u32::from(src2 & 0x1FF) << 18)
            | (u32::from(src1 & 0x1FF) << 9)
            | u32::from(src0 & 0x1FF);
        [w0, w1]
    }

    fn enc_vintrp(op: u8, vdst: u8, attr: u8, attrchan: u8, vsrc: u8) -> u32 {
        (0b110010u32 << 26)
            | (u32::from(vdst) << 18)
            | (u32::from(op & 0x3) << 16)
            | (u32::from(attr & 0x3F) << 10)
            | (u32::from(attrchan & 0x3) << 8)
            | u32::from(vsrc)
    }

    fn enc_exp(
        target: u8,
        en: u8,
        compr: bool,
        done: bool,
        vm: bool,
        vsrc: [u8; 4],
    ) -> [u32; 2] {
        let w0 = (0b111110u32 << 26)
            | (u32::from(vm) << 12)
            | (u32::from(done) << 11)
            | (u32::from(compr) << 10)
            | (u32::from(target & 0x3F) << 4)
            | u32::from(en & 0xF);
        let w1 = u32::from(vsrc[0])
            | (u32::from(vsrc[1]) << 8)
            | (u32::from(vsrc[2]) << 16)
            | (u32::from(vsrc[3]) << 24);
        [w0, w1]
    }

    fn enc_mimg(
        op: u8,
        dmask: u8,
        dim: u8,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        ssamp: u8,
    ) -> [u32; 2] {
        let w0 = (0b111100u32 << 26)
            | (u32::from(op & 0x7F) << 18)
            | (u32::from(dmask & 0xF) << 8)
            | (u32::from(dim & 0x7) << 3)
            | u32::from((op >> 7) & 0x1);
        let w1 = (u32::from(ssamp & 0x1F) << 21)
            | (u32::from(srsrc & 0x1F) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    /// Synthetic RDNA2 fullscreen-triangle vertex shader (`vertex_index` -> clip pos + UV).
    fn build_fullscreen_triangle_vs_rdna2() -> Vec<u8> {
        let mut words = vec![
            enc_vop2(0x18, 1, 128 + 1, 0), // v_lshlrev_b32 v1, 1, v0
            enc_vop2(0x19, 1, 128 + 2, 1), // v_and_b32 v1, 2, v1
            enc_vop2(0x19, 2, 128 + 2, 0), // v_and_b32 v2, 2, v0
            enc_vop1(0x06, 1, 256 + 1),    // v_cvt_f32_u32 v1, v1
            enc_vop1(0x06, 2, 256 + 2),    // v_cvt_f32_u32 v2, v2
        ];
        words.extend_from_slice(&enc_vop3(0x14B, 3, 256 + 1, 244, 243, false, 0, 0, 0));
        words.extend_from_slice(&enc_vop3(0x14B, 4, 256 + 2, 244, 242, false, 0, 0, 0b001));
        words.push(enc_vop1(0x01, 5, 128)); // v_mov_b32 v5, 0
        words.push(enc_vop1(0x01, 6, 242)); // v_mov_b32 v6, 1.0
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [3, 4, 5, 6]));
        words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [1, 2, 0, 0]));
        words.push(enc_sopp(0x01, 0)); // s_endpgm
        words_to_bytes(&words)
    }

    /// Synthetic RDNA2 textured splash pixel shader (`VINTRP` UV -> `image_sample` -> `exp mrt0`).
    fn build_textured_splash_ps_rdna2() -> Vec<u8> {
        let mut words = vec![
            enc_vintrp(0, 2, 0, 0, 0),
            enc_vintrp(1, 2, 0, 0, 1),
            enc_vintrp(0, 3, 0, 1, 0),
            enc_vintrp(1, 3, 0, 1, 1),
        ];
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 2, 1));
        words.push(enc_sopp(0x0C, 0x3F70)); // s_waitcnt vmcnt(0)
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0)); // s_endpgm
        words_to_bytes(&words)
    }

    /// Synthetic RDNA2 compute shader (`s_barrier` + `s_endpgm`).
    fn build_minimal_compute_rdna2() -> Vec<u8> {
        words_to_bytes(&[
            enc_vop1(0x01, 1, 242), // v_mov_b32 v1, 1.0
            enc_sopp(0x0A, 0),      // s_barrier
            enc_sopp(0x01, 0),      // s_endpgm
        ])
    }

    fn assert_naga_wgsl_valid(wgsl: &str) {
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\nWGSL:\n{wgsl}", e.emit_to_string(wgsl)));
        let mut validator = Validator::new(
            ValidationFlags::all(),
            Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        );
        validator.validate(&module).unwrap_or_else(|e| {
            panic!(
                "WGSL validation failed:\n{}\nWGSL:\n{wgsl}",
                e.emit_to_string(wgsl)
            )
        });
    }

    #[test]
    fn filename_hint_parsing_covers_stage_hash_and_fallbacks() {
        assert_eq!(
            parse_filename_hint("vs_0123456789abcdef.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Vertex,
                hash: Some(0x0123_4567_89ab_cdef),
            }
        );
        assert_eq!(
            parse_filename_hint("ps_FEDCBA9876543210.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Pixel,
                hash: Some(0xfedc_ba98_7654_3210),
            }
        );
        assert_eq!(
            parse_filename_hint("cs_0000000000000042.shader"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Compute,
                hash: Some(0x42),
            }
        );
        assert_eq!(
            parse_filename_hint("0011223344556677.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Auto,
                hash: Some(0x0011_2233_4455_6677),
            }
        );
        assert_eq!(
            parse_filename_hint("vs_0x0011223344556677.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Vertex,
                hash: Some(0x0011_2233_4455_6677),
            }
        );
        assert_eq!(
            parse_filename_hint("aabbccddeeff0011_fs.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Pixel,
                hash: Some(0xaabb_ccdd_eeff_0011),
            }
        );
        assert_eq!(
            parse_filename_hint("splash_vs.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Vertex,
                hash: None,
            }
        );
        assert_eq!(
            parse_filename_hint("arbitrary_shader.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Auto,
                hash: None,
            }
        );
        assert_eq!(
            parse_filename_hint("dir/nested/vs_deadbeefcafebabe.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Vertex,
                hash: Some(0xdead_beef_cafe_babe),
            }
        );
        assert_eq!(parse_filename_hint(".bin"), Rdna2FilenameHint::default());
        assert_eq!(parse_filename_hint(""), Rdna2FilenameHint::default());
    }

    #[test]
    fn packs_synthetic_fullscreen_vs_and_textured_splash_ps_round_trip_wgpk() {
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let cs_bytes = build_minimal_compute_rdna2();
        let cs_expected_hash = fnv1a_64(&cs_bytes);

        let blobs = vec![
            Rdna2BlobInput::new("ps_2000000000000002.rdna2", ps_bytes),
            Rdna2BlobInput::new("unlabeled_compute.bin", cs_bytes),
            Rdna2BlobInput::new("vs_1000000000000001.rdna2", vs_bytes),
        ];

        let result = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
            .expect("pack_rdna2_blobs should succeed");

        assert_eq!(result.summary.total_blobs, 3);
        assert_eq!(result.summary.translated_ok, 3);
        assert_eq!(result.summary.skipped_or_failed, 0);
        assert_eq!(result.summary.vertex_count, 1);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);
        assert_eq!(result.summary.pack_bytes, result.pack_bytes.len());
        assert_eq!(&result.pack_bytes[0..4], &PACK_MAGIC);

        let decoded = read_pack(&result.pack_bytes).expect("WGPK round-trip read should succeed");
        assert_eq!(decoded.version, PACK_VERSION);
        assert_eq!(decoded.flags, 0);
        assert_eq!(decoded.entries.len(), 3);

        // Entries must be strictly sorted by ascending u64 hash.
        for pair in decoded.entries.windows(2) {
            assert!(
                pair[0].hash < pair[1].hash,
                "WGPK entries must be deterministically sorted by hash"
            );
        }

        let vs_entry = decoded
            .entries
            .iter()
            .find(|e| e.hash == 0x1000_0000_0000_0001)
            .expect("VS entry must exist");
        assert_eq!(vs_entry.stage, ShaderStage::Vertex);
        assert!(vs_entry.wgsl.contains("@vertex"));
        assert!(vs_entry.wgsl.contains("@builtin(vertex_index)"));
        assert!(vs_entry.wgsl.contains("@builtin(position)"));
        assert_naga_wgsl_valid(&vs_entry.wgsl);

        let ps_entry = decoded
            .entries
            .iter()
            .find(|e| e.hash == 0x2000_0000_0000_0002)
            .expect("PS entry must exist");
        assert_eq!(ps_entry.stage, ShaderStage::Fragment);
        assert!(ps_entry.wgsl.contains("@fragment"));
        assert!(ps_entry.wgsl.contains("fn fs_main("));
        assert!(ps_entry.wgsl.contains("texture_2d<f32>"));
        assert!(ps_entry.wgsl.contains("textureSample("));
        assert_naga_wgsl_valid(&ps_entry.wgsl);

        // Per-entry summary + explicit bindings in pack summary JSON.
        assert_eq!(result.summary.entries.len(), 3);
        let ps_summary = result
            .summary
            .entries
            .iter()
            .find(|e| e.hash_fnv1a64 == 0x2000_0000_0000_0002)
            .expect("PS summary must exist");
        assert_eq!(ps_summary.entry_point, "fs_main");
        assert_eq!(ps_summary.bindings.len(), 2);
        let summary_json = result.summary.to_json();
        assert!(summary_json.contains("\"entries\": ["), "JSON:\n{summary_json}");
        assert!(summary_json.contains("\"entry_point\": \"fs_main\""), "JSON:\n{summary_json}");
        assert!(
            summary_json.contains(
                "{\"group\": 0, \"binding\": 1, \"kind\": \"texture_2d\", \"name\": \"tex0\", \"view_dimension\": \"2d\"}"
            ),
            "JSON:\n{summary_json}"
        );

        let cs_entry = decoded
            .entries
            .iter()
            .find(|e| e.hash == cs_expected_hash)
            .expect("auto-hashed CS entry must exist");
        assert_eq!(cs_entry.stage, ShaderStage::Compute);
        assert!(cs_entry.wgsl.contains("@compute"));
        assert!(cs_entry.wgsl.contains("workgroupBarrier()"));
        assert_naga_wgsl_valid(&cs_entry.wgsl);
    }

    #[test]
    fn pack_rdna2_blobs_skips_short_and_duplicate_hashes_deterministically() {
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();

        let blobs_a = vec![
            Rdna2BlobInput::new("empty.bin", vec![]),
            Rdna2BlobInput::new("too_short.rdna2", vec![0xAA, 0xBB, 0xCC]),
            Rdna2BlobInput::new("vs_00000000000000aa.rdna2", vs_bytes.clone()),
            Rdna2BlobInput::new("vs_dup_00000000000000aa.rdna2", vs_bytes.clone()),
            Rdna2BlobInput::new("ps_00000000000000bb.shader", ps_bytes.clone()),
        ];
        let mut blobs_b = blobs_a.clone();
        blobs_b.reverse();

        let res_a = pack_rdna2_blobs(&blobs_a, &Rdna2PackOptions::default()).unwrap();
        let res_b = pack_rdna2_blobs(&blobs_b, &Rdna2PackOptions::default()).unwrap();

        assert_eq!(res_a.pack_bytes, res_b.pack_bytes);
        assert_eq!(res_a.summary, res_b.summary);
        assert_eq!(res_a.summary.total_blobs, 5);
        assert_eq!(res_a.summary.translated_ok, 2);
        assert_eq!(res_a.summary.skipped_or_failed, 3);
        assert_eq!(res_a.summary.vertex_count, 1);
        assert_eq!(res_a.summary.pixel_count, 1);
        assert_eq!(res_a.summary.compute_count, 0);
    }

    #[test]
    fn pack_rdna2_directory_walks_subdirs_ignores_non_matching_and_escaping_symlinks() {
        let root = make_temp_dir("playgta6_rdna2_pack_dir");
        let input_dir = root.join("shaders_in");
        let nested_dir = input_dir.join("splash");
        let outside_dir = root.join("outside");
        fs::create_dir_all(&nested_dir).unwrap();
        fs::create_dir_all(&outside_dir).unwrap();

        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let cs_bytes = build_minimal_compute_rdna2();

        fs::write(nested_dir.join("vs_0000000000000001.rdna2"), &vs_bytes).unwrap();
        fs::write(nested_dir.join("ps_0000000000000002.shader"), &ps_bytes).unwrap();
        fs::write(input_dir.join("0000000000000003.bin"), &cs_bytes).unwrap();
        // Empty blob -> counted as skipped_or_failed.
        fs::write(input_dir.join("empty.bin"), b"").unwrap();
        // Non-matching extensions -> ignored completely.
        fs::write(input_dir.join("notes.txt"), b"ignore me").unwrap();
        fs::write(input_dir.join("already.wgsl"), b"// ignore").unwrap();

        // Outside file + symlink escaping input_dir (best-effort on Windows without symlink privileges).
        let outside_blob = outside_dir.join("vs_9999999999999999.rdna2");
        fs::write(&outside_blob, &vs_bytes).unwrap();
        let escape_link = input_dir.join("vs_9999999999999999.rdna2");
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(&outside_blob, &escape_link);
        }
        #[cfg(windows)]
        {
            let _ = std::os::windows::fs::symlink_file(&outside_blob, &escape_link);
        }

        let result = pack_rdna2_directory(&input_dir, &Rdna2PackOptions::default())
            .expect("pack_rdna2_directory should succeed");

        assert_eq!(result.summary.total_blobs, 4);
        assert_eq!(result.summary.translated_ok, 3);
        assert_eq!(result.summary.skipped_or_failed, 1);
        assert_eq!(result.summary.vertex_count, 1);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);

        let decoded = read_pack(&result.pack_bytes).unwrap();
        assert_eq!(decoded.entries.len(), 3);
        assert_eq!(decoded.entries[0].hash, 1);
        assert_eq!(decoded.entries[0].stage, ShaderStage::Vertex);
        assert_eq!(decoded.entries[1].hash, 2);
        assert_eq!(decoded.entries[1].stage, ShaderStage::Fragment);
        assert_eq!(decoded.entries[2].hash, 3);
        assert_eq!(decoded.entries[2].stage, ShaderStage::Compute);
        // Escaping symlink hash 0x9999999999999999 must never be packed.
        assert!(
            decoded
                .entries
                .iter()
                .all(|e| e.hash != 0x9999_9999_9999_9999)
        );

        // Strictly sanitized JSON and text summaries: zero host paths, zero usernames.
        let json = result.summary.to_json();
        let text = result.summary.to_text();
        let root_str = root.to_string_lossy().replace('\\', "/");
        for rendered in [&json, &text] {
            assert!(rendered.contains("total_blobs"));
            assert!(rendered.contains("translated_ok"));
            assert!(rendered.contains("skipped_or_failed"));
            assert!(rendered.contains("vertex_count"));
            assert!(rendered.contains("pixel_count"));
            assert!(rendered.contains("compute_count"));
            assert!(rendered.contains("pack_bytes"));
            assert!(!rendered.contains(&root_str));
            assert!(!rendered.contains("shaders_in"));
            assert!(!rendered.contains("outside"));
        }

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_directory_rejects_missing_or_file_input_without_leaking_host_path() {
        let root = make_temp_dir("playgta6_rdna2_pack_err");
        let missing = root.join("does_not_exist");
        let file = root.join("single.bin");
        fs::write(&file, build_minimal_compute_rdna2()).unwrap();

        let root_str = root.to_string_lossy().replace('\\', "/");

        let err_missing = pack_rdna2_directory(&missing, &Rdna2PackOptions::default())
            .unwrap_err()
            .to_string();
        assert!(!err_missing.contains(&root_str));

        let err_file = pack_rdna2_directory(&file, &Rdna2PackOptions::default())
            .unwrap_err()
            .to_string();
        assert!(!err_file.contains(&root_str));

        let _ = fs::remove_dir_all(&root);
    }

    fn enc_smem(
        op: u8,
        sdata: u8,
        sbase_pair: u8,
        soffset: u8,
        offset: i32,
        glc: bool,
        dlc: bool,
    ) -> [u32; 2] {
        let w0 = (0b111101u32 << 26)
            | (u32::from(op) << 18)
            | (u32::from(glc) << 16)
            | (u32::from(dlc) << 14)
            | (u32::from(sdata & 0x7F) << 6)
            | u32::from(sbase_pair & 0x3F);
        let w1 = (u32::from(soffset & 0x7F) << 25) | ((offset as u32) & 0x001F_FFFF);
        [w0, w1]
    }

    #[test]
    fn pack_rdna2_blobs_emits_tight_cbv_sizes_and_binding_metadata_in_json() {
        // 1) Small CBV shader reading cb[0..2] (offsets 0..44) -> 48 B (3 vec4s)
        let mut small_words = Vec::new();
        small_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x00, false, false));
        small_words.extend_from_slice(&enc_smem(0x08, 8, 0, 0x7D, 0x2C, false, false));
        small_words.push(enc_vop1(0x01, 0, 4));
        small_words.push(enc_vop1(0x01, 1, 8));
        small_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        small_words.push(enc_sopp(0x01, 0));

        // 2) Large CBV shader reading cb[20] (offset 0x140) -> 336 B (21 vec4s)
        let mut large_words = Vec::new();
        large_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x140, false, false));
        large_words.push(enc_vop1(0x01, 0, 4));
        large_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 0, 0, 0]));
        large_words.push(enc_sopp(0x01, 0));

        // 3) Multi-CBV shader: cb0 reading vec4 1 (32 B) and cb1 reading vec4 5 (96 B)
        let mut multi_words = Vec::new();
        multi_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x10, false, false));
        multi_words.extend_from_slice(&enc_smem(0x0A, 8, 2, 0x7D, 0x50, false, false));
        multi_words.push(enc_vop1(0x01, 0, 4));
        multi_words.push(enc_vop1(0x01, 1, 8));
        multi_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        multi_words.push(enc_sopp(0x01, 0));

        let blobs = vec![
            Rdna2BlobInput::new("ps_0000000000000101.rdna2", words_to_bytes(&small_words)),
            Rdna2BlobInput::new("ps_0000000000000102.rdna2", words_to_bytes(&large_words)),
            Rdna2BlobInput::new("ps_0000000000000103.rdna2", words_to_bytes(&multi_words)),
        ];

        let result = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
            .expect("pack_rdna2_blobs should succeed");
        assert_eq!(result.summary.translated_ok, 3);

        let decoded = read_pack(&result.pack_bytes).expect("WGPK read should succeed");
        assert_eq!(decoded.entries.len(), 3);
        assert!(decoded.entries[0].wgsl.contains("array<vec4<u32>, 3>"));
        assert!(decoded.entries[1].wgsl.contains("array<vec4<u32>, 21>"));
        assert!(decoded.entries[2].wgsl.contains("array<vec4<u32>, 2>"));
        assert!(decoded.entries[2].wgsl.contains("array<vec4<u32>, 6>"));
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        // Verify per-entry binding metadata in summary structs and JSON.
        assert_eq!(result.summary.entries[0].bindings[0].min_binding_size_bytes, Some(48));
        assert_eq!(result.summary.entries[0].bindings[0].highest_vec4_offset, Some(2));
        assert_eq!(result.summary.entries[1].bindings[0].min_binding_size_bytes, Some(336));
        assert_eq!(result.summary.entries[1].bindings[0].highest_vec4_offset, Some(20));
        assert_eq!(result.summary.entries[2].bindings.len(), 2);
        assert_eq!(result.summary.entries[2].bindings[0].min_binding_size_bytes, Some(32));
        assert_eq!(result.summary.entries[2].bindings[0].highest_vec4_offset, Some(1));
        assert_eq!(result.summary.entries[2].bindings[1].min_binding_size_bytes, Some(96));
        assert_eq!(result.summary.entries[2].bindings[1].highest_vec4_offset, Some(5));

        let json = result.summary.to_json();
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 48, \"highest_vec4_offset\": 2}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 336, \"highest_vec4_offset\": 20}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 32, \"highest_vec4_offset\": 1}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 16, \"kind\": \"uniform\", \"name\": \"cb1\", \"min_binding_size_bytes\": 96, \"highest_vec4_offset\": 5}"
            ),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn pack_rdna2_blobs_packs_mixed_vs_fs_and_compute_storage_lds_shader_into_wgpk() {
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();

        // Full compute shader using storage buffer load/store, LDS + s_barrier, and image_store:
        let mut cs_words = Vec::new();
        cs_words.push(enc_vop2(0x18, 3, 128 + 2, 0)); // v_lshlrev_b32 v3, 2, v0
        // buffer_load_dword v4, v3, s[4:7], 0 offen (srsrc=1 -> read-only-storage)
        cs_words.extend_from_slice(&[
            (0b111000u32 << 26) | (0x0Cu32 << 18) | (1 << 12),
            (128u32 << 24) | (1u32 << 16) | (4u32 << 8) | 3,
        ]);
        // ds_write_b32 v0, v3, v4
        cs_words.extend_from_slice(&[
            (0b110110u32 << 26) | (0x0Du32 << 18),
            (4u32 << 8) | 3,
        ]);
        cs_words.push(enc_sopp(0x0A, 0)); // s_barrier
        // ds_read_b32 v5, v3, v0
        cs_words.extend_from_slice(&[
            (0b110110u32 << 26) | (0x36u32 << 18),
            (5u32 << 24) | 3,
        ]);
        // buffer_store_dword v5, v3, s[8:11], 0 offen (srsrc=2 -> storage)
        cs_words.extend_from_slice(&[
            (0b111000u32 << 26) | (0x1Cu32 << 18) | (1 << 12),
            (128u32 << 24) | (2u32 << 16) | (5u32 << 8) | 3,
        ]);
        // image_store v[5:8], v[0:1], s[12:19] dmask:0xf dim:1 (srsrc=3 -> storage-texture)
        cs_words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 5, 3, 0));
        cs_words.push(enc_sopp(0x01, 0)); // s_endpgm
        let cs_bytes = words_to_bytes(&cs_words);

        let blobs = vec![
            Rdna2BlobInput::new("vs_0000000000000301.rdna2", vs_bytes),
            Rdna2BlobInput::new("ps_0000000000000302.rdna2", ps_bytes),
            Rdna2BlobInput::new("cs_0000000000000303.rdna2", cs_bytes),
        ];

        let result = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
            .expect("mixed vs+fs+cs WGPK pack should succeed");
        assert_eq!(result.summary.total_blobs, 3);
        assert_eq!(result.summary.translated_ok, 3);
        assert_eq!(result.summary.skipped_or_failed, 0);
        assert_eq!(result.summary.vertex_count, 1);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);

        let decoded = read_pack(&result.pack_bytes).expect("WGPK round-trip should succeed");
        assert_eq!(decoded.entries.len(), 3);
        assert_eq!(decoded.entries[0].stage, ShaderStage::Vertex);
        assert_eq!(decoded.entries[1].stage, ShaderStage::Fragment);
        assert_eq!(decoded.entries[2].stage, ShaderStage::Compute);
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        let cs_entry = &decoded.entries[2];
        assert!(cs_entry.wgsl.contains("fn cs_main("));
        assert!(cs_entry.wgsl.contains("var<workgroup> lds: array<u32, 4096>;"));
        assert!(cs_entry.wgsl.contains("workgroupBarrier();"));
        assert!(cs_entry.wgsl.contains("textureStore("));
        assert!(!cs_entry.wgsl.contains("enable f16"));

        let cs_summary = &result.summary.entries[2];
        assert_eq!(cs_summary.entry_point, "cs_main");
        assert_eq!(cs_summary.unsupported_op_count, 0);
        assert_eq!(cs_summary.bindings.len(), 3);
        assert_eq!(cs_summary.bindings[0].kind, "read-only-storage");
        assert_eq!(cs_summary.bindings[1].kind, "storage");
        assert_eq!(cs_summary.bindings[2].kind, "storage-texture");

        let json = result.summary.to_json();
        assert!(json.contains("\"kind\": \"read-only-storage\""), "JSON:\n{json}");
        assert!(json.contains("\"kind\": \"storage\""), "JSON:\n{json}");
        assert!(json.contains("\"kind\": \"storage-texture\""), "JSON:\n{json}");
        assert!(json.contains("\"vertex_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"fragment_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"compute_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"failed\": 0"), "JSON:\n{json}");
        assert!(json.contains("\"naga_validated\": true"), "JSON:\n{json}");
    }

    #[test]
    fn parse_pipeline_dump_manifest_parses_stages_and_rejects_malformed_inputs() {
        let valid_json = r#"{
            "schema": "playgta6-pipeline-dump",
            "schema_version": 1,
            "native_stages": [
                {
                    "code_identity": "0x0123456789abcdef",
                    "identity": "0xfedcba9876543210",
                    "stage": "vertex",
                    "layout": "A",
                    "code_bytes": 64,
                    "metadata_bytes": 128,
                    "code_file": "native/0x0123456789abcdef.bin",
                    "metadata_file": "native/0x0123456789abcdef.meta"
                },
                {
                    "code_identity": "0x1111222233334444",
                    "stage": "pixel",
                    "code_file": "native/0x1111222233334444.bin"
                },
                {
                    "code_identity": "0x5555666677778888",
                    "stage": "compute",
                    "code_file": "native/0x5555666677778888.bin"
                }
            ],
            "dxil": [{"identity": "0x0000000000000001", "hash_digest": "", "kind": -1, "bytes": 0, "file": "dxil/0x0000000000000001.dxbc"}],
            "root_signatures": [],
            "pipelines": [{"index": 0, "vs": null, "ps": null, "depth_bias": -1, "slope": 1.25e-2, "rect_list": false}]
        }"#;

        let parsed =
            parse_pipeline_dump_manifest(valid_json).expect("valid manifest should parse cleanly");
        assert_eq!(parsed.schema, PIPELINE_DUMP_SCHEMA);
        assert_eq!(parsed.schema_version, 1);
        assert_eq!(parsed.native_stages.len(), 3);
        assert_eq!(parsed.native_stages[0].code_identity, 0x0123_4567_89ab_cdef);
        assert_eq!(parsed.native_stages[0].identity, Some(0xfedc_ba98_7654_3210));
        assert_eq!(parsed.native_stages[0].stage, ShaderStage::Vertex);
        assert_eq!(parsed.native_stages[0].code_file, "native/0x0123456789abcdef.bin");
        assert_eq!(parsed.native_stages[0].code_bytes, Some(64));
        assert_eq!(parsed.native_stages[1].stage, ShaderStage::Fragment);
        assert_eq!(parsed.native_stages[2].stage, ShaderStage::Compute);

        // Reject wrong schema, unsupported version, unknown stage, bad hex, path traversal, and invalid JSON.
        assert!(parse_pipeline_dump_manifest(r#"{"schema":"other","native_stages":[]}"#).is_err());
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","schema_version":2,"native_stages":[]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"0x10","stage":"geometry","code_file":"native/a.bin"}]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"not_hex","stage":"vertex","code_file":"native/a.bin"}]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"0x10","stage":"vertex","code_file":"../secret.bin"}]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"0x10","stage":"vertex","code_file":"/etc/passwd"}]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"0x10","stage":"vertex","code_file":"C:\\Windows\\win.ini"}]}"#
            )
            .is_err()
        );
        assert!(parse_pipeline_dump_manifest("{truncated").is_err());
        // Deeply nested JSON must return a clean error rather than overflowing the stack.
        let deep = format!("{}{}", "[".repeat(128), "]".repeat(128));
        assert!(parse_pipeline_dump_manifest(&deep).is_err());
    }

    #[test]
    fn pack_rdna2_manifest_packs_vs_ps_cs_verifies_fnv1a_deduplicates_and_passes_naga() {
        let root = make_temp_dir("playgta6_rdna2_pack_manifest");
        let native_dir = root.join("native");
        fs::create_dir_all(&native_dir).unwrap();

        // 1) Synthetic VS (fullscreen triangle)
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let vs_hash = fnv1a_64(&vs_bytes);
        let vs_id = format_shader_hash(vs_hash);

        // 2) Synthetic PS reading cb0[0..2] (48 B tight CBV) + exp mrt0 -> exercises min_binding_size_bytes
        let mut ps_words = Vec::new();
        ps_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x00, false, false));
        ps_words.extend_from_slice(&enc_smem(0x08, 8, 0, 0x7D, 0x2C, false, false));
        ps_words.push(enc_vop1(0x01, 0, 4));
        ps_words.push(enc_vop1(0x01, 1, 8));
        ps_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        ps_words.push(enc_sopp(0x01, 0));
        let ps_bytes = words_to_bytes(&ps_words);
        let ps_hash = fnv1a_64(&ps_bytes);
        let ps_id = format_shader_hash(ps_hash);

        // 3) Synthetic Compute shader (s_barrier + s_endpgm)
        let cs_bytes = build_minimal_compute_rdna2();
        let cs_hash = fnv1a_64(&cs_bytes);
        let cs_id = format_shader_hash(cs_hash);

        // 4) Ambiguous bare shader (no EXP / no VINTRP / no s_barrier) declared explicitly as "vertex"
        //    in manifest.json -> proves explicit stage propagation with zero stage-inference errors!
        let bare_vs_bytes = words_to_bytes(&[
            enc_vop1(0x01, 0, 242), // v_mov_b32 v0, 1.0
            enc_sopp(0x01, 0),      // s_endpgm
        ]);
        let bare_vs_hash = fnv1a_64(&bare_vs_bytes);
        let bare_vs_id = format_shader_hash(bare_vs_hash);

        fs::write(native_dir.join(format!("{vs_id}.bin")), &vs_bytes).unwrap();
        fs::write(native_dir.join(format!("{ps_id}.bin")), &ps_bytes).unwrap();
        fs::write(native_dir.join(format!("{cs_id}.bin")), &cs_bytes).unwrap();
        fs::write(native_dir.join(format!("{bare_vs_id}.bin")), &bare_vs_bytes).unwrap();

        // Include a duplicate `code_identity` entry for `vs_id` (different `identity` & `.meta` file,
        // matching `PipelineDump::native_stage` when two metadata variants share one code blob).
        let manifest_json = format!(
            r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{ps_id}", "identity": "0xaaaaaaaaaaaaaaaa", "stage": "pixel", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{ps_id}.bin", "metadata_file": "native/{ps_id}.meta"}},
    {{"code_identity": "{vs_id}", "identity": "0xbbbbbbbbbbbbbbbb", "stage": "vertex", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{vs_id}.bin", "metadata_file": "native/{vs_id}.meta"}},
    {{"code_identity": "{vs_id}", "identity": "0xcccccccccccccccc", "stage": "vertex", "layout": "B", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{vs_id}.bin", "metadata_file": "native/{vs_id}-0xcccccccccccccccc.meta"}},
    {{"code_identity": "{cs_id}", "identity": "0xdddddddddddddddd", "stage": "compute", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{cs_id}.bin", "metadata_file": "native/{cs_id}.meta"}},
    {{"code_identity": "{bare_vs_id}", "identity": "0xeeeeeeeeeeeeeeee", "stage": "vertex", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{bare_vs_id}.bin", "metadata_file": "native/{bare_vs_id}.meta"}}
  ],
  "dxil": [],
  "root_signatures": [],
  "pipelines": []
}}"#,
            ps_bytes.len(),
            vs_bytes.len(),
            vs_bytes.len(),
            cs_bytes.len(),
            bare_vs_bytes.len()
        );
        let manifest_path = root.join("manifest.json");
        fs::write(&manifest_path, &manifest_json).unwrap();

        let result = pack_rdna2_manifest(&manifest_path, &Rdna2PackOptions::default())
            .expect("pack_rdna2_manifest should succeed");

        assert_eq!(result.summary.total_blobs, 4);
        assert_eq!(result.summary.translated_ok, 4);
        assert_eq!(result.summary.skipped_or_failed, 0);
        assert_eq!(result.summary.failed, 0);
        assert_eq!(result.summary.manifest_duplicates_skipped, 1);
        assert_eq!(result.summary.vertex_packed, 2);
        assert_eq!(result.summary.fragment_packed, 1);
        assert_eq!(result.summary.compute_packed, 1);
        assert_eq!(result.summary.vertex_count, 2);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);
        assert!(result.summary.failures.is_empty());

        let decoded = read_pack(&result.pack_bytes).expect("WGPK pack must decode cleanly");
        assert_eq!(decoded.version, PACK_VERSION);
        assert_eq!(decoded.entries.len(), 4);
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        let vs_packed = decoded.entries.iter().find(|e| e.hash == vs_hash).unwrap();
        assert_eq!(vs_packed.stage, ShaderStage::Vertex);
        assert!(vs_packed.wgsl.contains("fn vs_main("));

        let bare_vs_packed = decoded.entries.iter().find(|e| e.hash == bare_vs_hash).unwrap();
        assert_eq!(bare_vs_packed.stage, ShaderStage::Vertex);
        assert!(bare_vs_packed.wgsl.contains("fn vs_main("));

        let ps_packed = decoded.entries.iter().find(|e| e.hash == ps_hash).unwrap();
        assert_eq!(ps_packed.stage, ShaderStage::Fragment);
        assert!(ps_packed.wgsl.contains("fn fs_main("));

        let cs_packed = decoded.entries.iter().find(|e| e.hash == cs_hash).unwrap();
        assert_eq!(cs_packed.stage, ShaderStage::Compute);
        assert!(cs_packed.wgsl.contains("fn cs_main("));

        let json = result.summary.to_json();
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(json.contains("\"vertex_packed\": 2"), "JSON:\n{json}");
        assert!(json.contains("\"fragment_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"compute_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"failed\": 0"), "JSON:\n{json}");
        assert!(json.contains("\"manifest_duplicates_skipped\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"naga_validated\": true"), "JSON:\n{json}");
        assert!(json.contains("\"entry_point\": \"vs_main\""), "JSON:\n{json}");
        assert!(json.contains("\"entry_point\": \"fs_main\""), "JSON:\n{json}");
        assert!(json.contains("\"entry_point\": \"cs_main\""), "JSON:\n{json}");
        assert!(json.contains("\"min_binding_size_bytes\": 48"), "JSON:\n{json}");
        assert!(json.contains("\"failures\": []"), "JSON:\n{json}");
        assert!(!json.contains(&root_str), "JSON must not leak host path");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_manifest_strict_fails_on_bad_entries_and_best_effort_skips_cleanly() {
        let root = make_temp_dir("playgta6_rdna2_manifest_best_effort");
        let native_dir = root.join("native");
        fs::create_dir_all(&native_dir).unwrap();

        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let vs_hash = fnv1a_64(&vs_bytes);
        let vs_id = format_shader_hash(vs_hash);
        fs::write(native_dir.join(format!("{vs_id}.bin")), &vs_bytes).unwrap();

        // Corrupt/tampered blob whose actual FNV-1a does NOT match declared code_identity.
        let bad_hash_id = "0xdeadbeef00000001";
        fs::write(native_dir.join(format!("{bad_hash_id}.bin")), build_minimal_compute_rdna2())
            .unwrap();

        // Truncated 2-byte blob whose FNV-1a matches its declared code_identity, but fails MIN_RDNA2_BLOB_BYTES.
        let short_bytes = [0x01u8, 0x02u8];
        let short_hash = fnv1a_64(&short_bytes);
        let short_id = format_shader_hash(short_hash);
        fs::write(native_dir.join(format!("{short_id}.bin")), short_bytes).unwrap();

        // Missing file entry.
        let missing_id = "0xdeadbeef00000002";

        // 1) Hash mismatch manifest -> strict returns HashMismatch, best-effort skips & reports.
        let mismatch_manifest = root.join("mismatch_manifest.json");
        fs::write(
            &mismatch_manifest,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{vs_id}", "stage": "vertex", "code_file": "native/{vs_id}.bin"}},
    {{"code_identity": "{bad_hash_id}", "stage": "compute", "code_file": "native/{bad_hash_id}.bin"}}
  ]
}}"#
            ),
        )
        .unwrap();

        let strict_err = pack_rdna2_manifest(&mismatch_manifest, &Rdna2PackOptions::default())
            .expect_err("strict manifest mode must fail on FNV-1a hash mismatch");
        assert!(matches!(
            strict_err,
            Rdna2PackError::HashMismatch {
                expected: 0xdead_beef_0000_0001,
                stage: ShaderStage::Compute,
                ..
            }
        ));

        // 2) Mixed manifest with 1 good VS + 3 bad entries (hash mismatch, truncated, missing).
        let mixed_manifest = root.join("mixed_manifest.json");
        fs::write(
            &mixed_manifest,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{vs_id}", "stage": "vertex", "code_file": "native/{vs_id}.bin"}},
    {{"code_identity": "{bad_hash_id}", "stage": "compute", "code_file": "native/{bad_hash_id}.bin"}},
    {{"code_identity": "{short_id}", "stage": "pixel", "code_file": "native/{short_id}.bin"}},
    {{"code_identity": "{missing_id}", "stage": "pixel", "code_file": "native/{missing_id}.bin"}}
  ]
}}"#
            ),
        )
        .unwrap();

        let best_effort_opts = Rdna2PackOptions {
            best_effort: true,
            ..Rdna2PackOptions::default()
        };
        let res = pack_rdna2_manifest(&mixed_manifest, &best_effort_opts)
            .expect("--best-effort manifest mode should pack valid shaders and skip bad ones");

        assert_eq!(res.summary.total_blobs, 4);
        assert_eq!(res.summary.translated_ok, 1);
        assert_eq!(res.summary.vertex_packed, 1);
        assert_eq!(res.summary.fragment_packed, 0);
        assert_eq!(res.summary.compute_packed, 0);
        assert_eq!(res.summary.failed, 3);
        assert_eq!(res.summary.skipped_or_failed, 3);
        assert_eq!(res.summary.failures.len(), 3);

        let decoded = read_pack(&res.pack_bytes).unwrap();
        assert_eq!(decoded.entries.len(), 1);
        assert_eq!(decoded.entries[0].hash, vs_hash);
        assert_naga_wgsl_valid(&decoded.entries[0].wgsl);

        let json = res.summary.to_json();
        let text = res.summary.to_text();
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(json.contains("\"vertex_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"failed\": 3"), "JSON:\n{json}");
        assert!(json.contains("\"failures\": ["), "JSON:\n{json}");
        assert!(json.contains("FNV-1a 64 hash mismatch"), "JSON:\n{json}");
        assert!(!json.contains(&root_str), "JSON must never leak host path");
        assert!(!text.contains(&root_str), "text must never leak host path");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_sources_combines_manifest_and_directory_with_manifest_precedence() {
        let root = make_temp_dir("playgta6_rdna2_sources_combined");
        let dump_dir = root.join("dump");
        let native_dir = dump_dir.join("native");
        let extra_dir = root.join("extra");
        fs::create_dir_all(&native_dir).unwrap();
        fs::create_dir_all(&extra_dir).unwrap();

        // Bare shader with no EXP/barrier: manifest explicitly labels it Vertex, while directory
        // scan without hints would infer Compute. Manifest MUST take precedence!
        let shared_bytes = words_to_bytes(&[enc_vop1(0x01, 0, 242), enc_sopp(0x01, 0)]);
        let shared_hash = fnv1a_64(&shared_bytes);
        let shared_id = format_shader_hash(shared_hash);
        fs::write(native_dir.join(format!("{shared_id}.bin")), &shared_bytes).unwrap();
        fs::write(extra_dir.join(format!("{shared_id}.bin")), &shared_bytes).unwrap();

        // Extra PS only in positional directory
        let ps_bytes = build_textured_splash_ps_rdna2();
        let ps_hash = fnv1a_64(&ps_bytes);
        fs::write(extra_dir.join("splash_ps.bin"), &ps_bytes).unwrap();

        let manifest_path = dump_dir.join("manifest.json");
        fs::write(
            &manifest_path,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{shared_id}", "stage": "vertex", "code_file": "native/{shared_id}.bin"}}
  ]
}}"#
            ),
        )
        .unwrap();

        let res = pack_rdna2_sources(
            Some(&extra_dir),
            Some(&manifest_path),
            &Rdna2PackOptions::default(),
        )
        .expect("combined manifest + directory pack should succeed");

        assert_eq!(res.summary.translated_ok, 2);
        assert_eq!(res.summary.vertex_packed, 1);
        assert_eq!(res.summary.fragment_packed, 1);
        assert_eq!(res.summary.compute_packed, 0);
        assert_eq!(res.summary.failed, 0);
        assert_eq!(res.summary.skipped_or_failed, 1); // duplicate hash from extra_dir skipped

        let decoded = read_pack(&res.pack_bytes).unwrap();
        let shared_entry = decoded.entries.iter().find(|e| e.hash == shared_hash).unwrap();
        assert_eq!(shared_entry.stage, ShaderStage::Vertex);
        assert!(shared_entry.wgsl.contains("fn vs_main("));
        let ps_entry = decoded.entries.iter().find(|e| e.hash == ps_hash).unwrap();
        assert_eq!(ps_entry.stage, ShaderStage::Fragment);
        assert!(ps_entry.wgsl.contains("fn fs_main("));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_blobs_max_wgsl_bytes_skips_oversize_and_packs_smaller_stages() {
        let cs_bytes = build_minimal_compute_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        // Build a larger synthetic shader by repeating valid VOP1/VOP2 ALU ops before EXP + s_endpgm.
        let mut large_words = Vec::new();
        for _ in 0..80 {
            large_words.push(enc_vop1(0x01, 1, 242)); // v_mov_b32 v1, 1.0
            large_words.push(enc_vop2(0x18, 2, 128 + 1, 1)); // v_lshlrev_b32 v2, 1, v1
        }
        large_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [1, 2, 1, 2]));
        large_words.push(enc_sopp(0x01, 0));
        let large_ps_bytes = words_to_bytes(&large_words);

        let blobs = vec![
            Rdna2BlobInput::new("cs_0000000000001001.rdna2", cs_bytes),
            Rdna2BlobInput::new("ps_0000000000001002.rdna2", ps_bytes),
            Rdna2BlobInput::new("ps_0000000000001003.rdna2", large_ps_bytes),
        ];

        // Baseline pack without limits to obtain exact translated WGSL byte lengths.
        let full = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
            .expect("baseline pack should succeed");
        assert_eq!(full.summary.translated_ok, 3);
        assert_eq!(full.summary.skipped_oversize, 0);
        assert_eq!(full.summary.skipped_budget, 0);

        let len_cs = full
            .entries
            .iter()
            .find(|e| e.hash == 0x1001)
            .unwrap()
            .wgsl
            .len();
        let len_ps = full
            .entries
            .iter()
            .find(|e| e.hash == 0x1002)
            .unwrap()
            .wgsl
            .len();
        let len_large = full
            .entries
            .iter()
            .find(|e| e.hash == 0x1003)
            .unwrap()
            .wgsl
            .len();
        assert!(len_large > len_ps && len_large > len_cs);
        assert_eq!(full.summary.largest_wgsl_bytes, len_large);
        assert_eq!(
            full.summary.total_wgsl_bytes,
            (len_cs + len_ps + len_large) as u64
        );

        let limit = len_cs.max(len_ps);
        assert!(limit < len_large);

        let limited = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_wgsl_bytes: Some(limit),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("max_wgsl_bytes pack should succeed cleanly");

        assert_eq!(limited.summary.total_blobs, 3);
        assert_eq!(limited.summary.translated_ok, 2);
        assert_eq!(limited.summary.skipped_oversize, 1);
        assert_eq!(limited.summary.skipped_budget, 0);
        assert_eq!(limited.summary.skipped_or_failed, 1);
        assert_eq!(limited.summary.failed, 0);
        assert!(limited.summary.failures.is_empty());
        assert_eq!(limited.summary.largest_wgsl_bytes, limit);
        assert_eq!(limited.summary.total_wgsl_bytes, (len_cs + len_ps) as u64);
        assert_eq!(limited.summary.entries.len(), 2);
        assert!(limited.summary.entries.iter().all(|e| e.hash_fnv1a64 != 0x1003));

        let decoded = read_pack(&limited.pack_bytes).expect("WGPK pack must decode cleanly");
        assert_eq!(decoded.entries.len(), 2);
        assert!(decoded.entries.iter().any(|e| e.hash == 0x1001));
        assert!(decoded.entries.iter().any(|e| e.hash == 0x1002));
        assert!(decoded.entries.iter().all(|e| e.hash != 0x1003));
        for entry in &decoded.entries {
            assert!(entry.wgsl.len() <= limit);
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        let json = limited.summary.to_json();
        let text = limited.summary.to_text();
        assert!(json.contains("\"skipped_oversize\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"skipped_budget\": 0"), "JSON:\n{json}");
        assert!(
            json.contains(&format!("\"largest_wgsl_bytes\": {limit}")),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(&format!(
                "\"total_wgsl_bytes\": {}",
                (len_cs + len_ps) as u64
            )),
            "JSON:\n{json}"
        );
        assert!(!json.contains("0x0000000000001003"), "skipped stage must be excluded from JSON entries:\n{json}");
        assert!(text.contains("skipped_oversize:  1"), "text:\n{text}");
        assert!(text.contains("skipped_budget:    0"), "text:\n{text}");
    }

    #[test]
    fn pack_rdna2_blobs_and_manifest_max_pack_bytes_enforces_exact_budget() {
        let cs_bytes = build_minimal_compute_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();

        // Names chosen so sorted order is a (cs), b (ps), c (vs).
        let blobs = vec![
            Rdna2BlobInput::new("a_cs_0000000000002001.rdna2", cs_bytes.clone()),
            Rdna2BlobInput::new("b_ps_0000000000002002.rdna2", ps_bytes.clone()),
            Rdna2BlobInput::new("c_vs_0000000000002003.rdna2", vs_bytes.clone()),
        ];

        let full = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default()).unwrap();
        let len_a = full
            .entries
            .iter()
            .find(|e| e.hash == 0x2001)
            .unwrap()
            .wgsl
            .len() as u64;
        let len_b = full
            .entries
            .iter()
            .find(|e| e.hash == 0x2002)
            .unwrap()
            .wgsl
            .len() as u64;
        let entry_cost_a = PACK_ENTRY_BYTES as u64 + len_a;
        let entry_cost_b = PACK_ENTRY_BYTES as u64 + len_b;

        // Exact budget for header + a + b: packs 2 stages, skips 3rd.
        let exact_two_budget = PACK_HEADER_BYTES as u64 + entry_cost_a + entry_cost_b;
        let two_opts = Rdna2PackOptions {
            max_pack_bytes: Some(exact_two_budget),
            ..Rdna2PackOptions::default()
        };
        let res_two = pack_rdna2_blobs(&blobs, &two_opts).expect("budgeted pack should succeed");
        assert_eq!(res_two.pack_bytes.len() as u64, exact_two_budget);
        assert_eq!(res_two.summary.translated_ok, 2);
        assert_eq!(res_two.summary.skipped_budget, 1);
        assert_eq!(res_two.summary.skipped_oversize, 0);
        assert_eq!(res_two.summary.skipped_or_failed, 1);
        assert_eq!(res_two.summary.failed, 0);
        assert_eq!(res_two.summary.total_wgsl_bytes, len_a + len_b);
        assert_eq!(
            res_two.summary.largest_wgsl_bytes,
            (len_a.max(len_b)) as usize
        );
        assert_eq!(res_two.summary.entries.len(), 2);
        let decoded_two = read_pack(&res_two.pack_bytes).unwrap();
        assert_eq!(decoded_two.entries.len(), 2);
        assert!(decoded_two.entries.iter().all(|e| e.hash != 0x2003));

        // Determinism check: reversed input slice produces byte-identical pack & summary.
        let mut blobs_rev = blobs.clone();
        blobs_rev.reverse();
        let res_two_rev = pack_rdna2_blobs(&blobs_rev, &two_opts).unwrap();
        assert_eq!(res_two.pack_bytes, res_two_rev.pack_bytes);
        assert_eq!(res_two.summary, res_two_rev.summary);

        // One byte shy of fitting stage b -> only stage a fits, 2 skipped_budget.
        let tight_budget = exact_two_budget - 1;
        let res_one = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(tight_budget),
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        assert!(res_one.pack_bytes.len() as u64 <= tight_budget);
        assert_eq!(
            res_one.pack_bytes.len() as u64,
            PACK_HEADER_BYTES as u64 + entry_cost_a
        );
        assert_eq!(res_one.summary.translated_ok, 1);
        assert_eq!(res_one.summary.skipped_budget, 2);
        assert_eq!(res_one.summary.entries.len(), 1);
        assert_eq!(res_one.summary.total_wgsl_bytes, len_a);
        assert_eq!(res_one.summary.largest_wgsl_bytes, len_a as usize);

        // Budget larger than u32::MAX clamps safely to u32::MAX and packs all small shaders.
        let clamped = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(u64::MAX),
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        assert_eq!(clamped.summary.translated_ok, 3);
        assert_eq!(clamped.summary.skipped_budget, 0);

        // Also verify strict --manifest mode respects max_pack_bytes without erroring.
        let root = make_temp_dir("playgta6_rdna2_manifest_budget");
        let native_dir = root.join("native");
        fs::create_dir_all(&native_dir).unwrap();
        let cs_hash = fnv1a_64(&cs_bytes);
        let ps_hash = fnv1a_64(&ps_bytes);
        let vs_hash = fnv1a_64(&vs_bytes);
        let cs_id = format_shader_hash(cs_hash);
        let ps_id = format_shader_hash(ps_hash);
        let vs_id = format_shader_hash(vs_hash);
        fs::write(native_dir.join(format!("{cs_id}.bin")), &cs_bytes).unwrap();
        fs::write(native_dir.join(format!("{ps_id}.bin")), &ps_bytes).unwrap();
        fs::write(native_dir.join(format!("{vs_id}.bin")), &vs_bytes).unwrap();

        let manifest_path = root.join("manifest.json");
        fs::write(
            &manifest_path,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{cs_id}", "stage": "compute", "code_file": "native/{cs_id}.bin"}},
    {{"code_identity": "{ps_id}", "stage": "pixel", "code_file": "native/{ps_id}.bin"}},
    {{"code_identity": "{vs_id}", "stage": "vertex", "code_file": "native/{vs_id}.bin"}}
  ]
}}"#
            ),
        )
        .unwrap();

        let full_manifest = pack_rdna2_manifest(&manifest_path, &Rdna2PackOptions::default()).unwrap();
        assert_eq!(full_manifest.entries.len(), 3);
        let first_entry_len = full_manifest.entries[0].wgsl.len() as u64;
        let first_budget = PACK_HEADER_BYTES as u64 + PACK_ENTRY_BYTES as u64 + first_entry_len;

        let budgeted_manifest = pack_rdna2_manifest(
            &manifest_path,
            &Rdna2PackOptions {
                best_effort: false,
                max_pack_bytes: Some(first_budget),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("strict manifest mode with max_pack_bytes should succeed");
        assert_eq!(budgeted_manifest.pack_bytes.len() as u64, first_budget);
        assert_eq!(budgeted_manifest.summary.translated_ok, 1);
        assert_eq!(budgeted_manifest.summary.skipped_budget, 2);
        assert_eq!(budgeted_manifest.summary.failed, 0);
        assert_eq!(budgeted_manifest.summary.entries.len(), 1);
        assert_eq!(
            budgeted_manifest.entries[0].hash,
            full_manifest.entries[0].hash
        );
        let decoded_manifest = read_pack(&budgeted_manifest.pack_bytes).unwrap();
        assert_eq!(decoded_manifest.entries.len(), 1);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn validate_wgpk_v1_size_returns_actionable_overflow_error_above_u32_max() {
        let max_single_payload =
            u64::from(u32::MAX) - PACK_HEADER_BYTES as u64 - PACK_ENTRY_BYTES as u64;
        assert_eq!(
            validate_wgpk_v1_size(1, max_single_payload, 1024).unwrap(),
            u64::from(u32::MAX)
        );

        let err = validate_wgpk_v1_size(1, max_single_payload + 1, 2_097_152)
            .expect_err("total size exceeding u32::MAX must fail");
        let msg = err.to_string();
        assert!(msg.contains("--pack-version 2"), "err: {msg}");
        assert!(msg.contains("--max-wgsl-bytes"), "err: {msg}");
        assert!(msg.contains("--max-pack-bytes"), "err: {msg}");
        assert!(
            msg.contains(&format!("total_wgsl_bytes={}", max_single_payload + 1)),
            "err: {msg}"
        );
        assert!(msg.contains("largest_wgsl_bytes=2097152"), "err: {msg}");

        // Auto & V2 seamlessly accept >4 GiB projected totals using 24-byte v2 TOC entries,
        // while forced V1 rejects them cleanly.
        assert!(
            validate_wgpk_size(PackVersion::V1, 1, max_single_payload + 1, 2_097_152).is_err()
        );
        let auto_v2_total =
            validate_wgpk_size(PackVersion::Auto, 1, max_single_payload + 1, 2_097_152)
                .expect("Auto upgrades >4 GiB pack to WGPK v2");
        assert_eq!(
            auto_v2_total,
            PACK_HEADER_BYTES as u64 + PACK_ENTRY_BYTES_V2 as u64 + max_single_payload + 1
        );
        let explicit_v2_total =
            validate_wgpk_size(PackVersion::V2, 1, max_single_payload + 1, 2_097_152)
                .expect("V2 accepts >4 GiB pack");
        assert_eq!(explicit_v2_total, auto_v2_total);
    }

    #[test]
    fn pack_rdna2_blobs_explicit_wgpk_v2_round_trip_and_budget() {
        let cs_bytes = build_minimal_compute_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();

        let blobs = vec![
            Rdna2BlobInput::new("a_cs_0000000000009001.rdna2", cs_bytes),
            Rdna2BlobInput::new("b_ps_0000000000009002.rdna2", ps_bytes),
            Rdna2BlobInput::new("c_vs_0000000000009003.rdna2", vs_bytes),
        ];

        let v1_res = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default()).unwrap();
        assert_eq!(v1_res.summary.pack_version, PACK_VERSION_V1);
        assert!(v1_res.summary.to_json().contains("\"format\": \"wgpk_v1\""));
        assert!(v1_res.summary.to_text().starts_with("WGPK v1 "));

        let v2_opts = Rdna2PackOptions {
            pack_version: PackVersion::V2,
            ..Rdna2PackOptions::default()
        };
        let v2_res = pack_rdna2_blobs(&blobs, &v2_opts).expect("v2 rdna2 pack should succeed");
        assert_eq!(v2_res.summary.pack_version, PACK_VERSION_V2);
        assert_eq!(v2_res.pack_bytes.len() - v1_res.pack_bytes.len(), 4 * 3);
        assert!(v2_res.summary.to_json().contains("\"format\": \"wgpk_v2\""));
        assert!(v2_res.summary.to_text().starts_with("WGPK v2 "));

        let decoded_v2 = read_pack(&v2_res.pack_bytes).expect("v2 pack must decode");
        assert_eq!(decoded_v2.version, PACK_VERSION_V2);
        assert_eq!(decoded_v2.entries.len(), 3);
        let decoded_v1 = read_pack(&v1_res.pack_bytes).unwrap();
        for (e1, e2) in decoded_v1.entries.iter().zip(decoded_v2.entries.iter()) {
            assert_eq!(e1.hash, e2.hash);
            assert_eq!(e1.stage, e2.stage);
            assert_eq!(e1.wgsl, e2.wgsl);
            assert_eq!(e2.offset, e1.offset + 12); // 3 entries * +4 bytes in TOC
        }

        // Exact v2 budget uses 24-byte TOC records per entry.
        let len_a = v2_res.entries[0].wgsl.len() as u64;
        let one_v2_budget = PACK_HEADER_BYTES as u64 + PACK_ENTRY_BYTES_V2 as u64 + len_a;
        let budgeted_v2 = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                pack_version: PackVersion::V2,
                max_pack_bytes: Some(one_v2_budget),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("v2 budgeted pack should succeed");
        assert_eq!(budgeted_v2.summary.pack_version, PACK_VERSION_V2);
        assert_eq!(budgeted_v2.summary.translated_ok, 1);
        assert_eq!(budgeted_v2.summary.skipped_budget, 2);
        assert_eq!(budgeted_v2.pack_bytes.len() as u64, one_v2_budget);
    }

    #[test]
    fn effective_jobs_resolves_default_and_clamps_zero() {
        assert!(Rdna2PackOptions::default().effective_jobs() >= 1);
        assert_eq!(
            Rdna2PackOptions {
                jobs: Some(0),
                ..Rdna2PackOptions::default()
            }
            .effective_jobs(),
            1
        );
        assert_eq!(
            Rdna2PackOptions {
                jobs: Some(1),
                ..Rdna2PackOptions::default()
            }
            .effective_jobs(),
            1
        );
        assert_eq!(
            Rdna2PackOptions {
                jobs: Some(4),
                ..Rdna2PackOptions::default()
            }
            .effective_jobs(),
            4
        );
    }

    #[test]
    fn pack_rdna2_blobs_parallel_jobs_4_matches_jobs_1_bit_for_bit() {
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let cs_bytes = build_minimal_compute_rdna2();

        // Build several distinct valid shaders of varying sizes plus one oversize shader.
        let make_alu_ps = |iters: usize, dst_reg: u8| -> Vec<u8> {
            let mut words = Vec::with_capacity(iters * 2 + 4);
            for _ in 0..iters {
                words.push(enc_vop1(0x01, dst_reg, 242)); // v_mov_b32 v<dst>, 1.0
                words.push(enc_vop2(0x18, dst_reg + 1, 128 + 1, dst_reg));
            }
            words.extend_from_slice(&enc_exp(
                0,
                0xF,
                false,
                true,
                true,
                [dst_reg, dst_reg + 1, dst_reg, dst_reg + 1],
            ));
            words.push(enc_sopp(0x01, 0));
            words_to_bytes(&words)
        };

        let med_ps_a = make_alu_ps(6, 1);
        let med_ps_b = make_alu_ps(12, 2);
        let med_ps_c = make_alu_ps(18, 3);
        let oversize_ps = make_alu_ps(90, 4);

        let blobs = vec![
            Rdna2BlobInput::new("a_vs_0000000000004001.rdna2", vs_bytes.clone()),
            Rdna2BlobInput::new("b_ps_0000000000004002.rdna2", ps_bytes.clone()),
            Rdna2BlobInput::new("c_cs_0000000000004003.rdna2", cs_bytes.clone()),
            // Duplicate hash with identical bytes (exercises dedup + duplicate-hash skip)
            Rdna2BlobInput::new("d_vs_dup_0000000000004001.rdna2", vs_bytes.clone()),
            // Identical bytes under a distinct hash (exercises dedup reuse under new hash)
            Rdna2BlobInput::new("e_cs_alias_0000000000004004.rdna2", cs_bytes.clone()),
            Rdna2BlobInput::new("f_ps_0000000000004005.rdna2", med_ps_a),
            Rdna2BlobInput::new("g_ps_0000000000004006.rdna2", med_ps_b),
            Rdna2BlobInput::new("h_ps_0000000000004007.rdna2", med_ps_c),
            // Oversize stage
            Rdna2BlobInput::new("i_ps_0000000000004008.rdna2", oversize_ps),
            // Invalid / too-short blobs (best-effort failures)
            Rdna2BlobInput::new("j_empty.bin", vec![]),
            Rdna2BlobInput::new("k_short_0000000000004009.rdna2", vec![0xDE, 0xAD]),
        ];

        // Measure the oversize threshold from a baseline pass.
        let baseline = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                jobs: Some(1),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("baseline blobs pack should succeed");
        let oversize_len = baseline
            .entries
            .iter()
            .find(|e| e.hash == 0x4008)
            .unwrap()
            .wgsl
            .len();
        let max_other_len = baseline
            .entries
            .iter()
            .filter(|e| e.hash != 0x4008)
            .map(|e| e.wgsl.len())
            .max()
            .unwrap();
        assert!(oversize_len > max_other_len);

        let opts_j1 = Rdna2PackOptions {
            best_effort: true,
            max_wgsl_bytes: Some(max_other_len),
            jobs: Some(1),
            ..Rdna2PackOptions::default()
        };
        let opts_j4 = Rdna2PackOptions {
            jobs: Some(4),
            ..opts_j1.clone()
        };

        let res_j1 = pack_rdna2_blobs(&blobs, &opts_j1).expect("jobs=1 pack should succeed");
        let res_j4 = pack_rdna2_blobs(&blobs, &opts_j4).expect("jobs=4 pack should succeed");

        assert_eq!(res_j1.pack_bytes, res_j4.pack_bytes);
        assert_eq!(res_j1.summary.to_json(), res_j4.summary.to_json());
        assert_eq!(res_j1.summary.to_text(), res_j4.summary.to_text());
        assert_eq!(res_j1.summary, res_j4.summary);
        assert_eq!(res_j4.summary.total_blobs, blobs.len());
        assert_eq!(res_j4.summary.translated_ok, 7);
        assert_eq!(res_j4.summary.skipped_oversize, 1);
        assert_eq!(res_j4.summary.failed, 2);
        assert_eq!(res_j4.summary.skipped_or_failed, 4); // 1 dup + 1 oversize + 2 short

        // Also verify reversed input order + tight max_pack_bytes budget match bit-for-bit.
        let mut blobs_rev = blobs.clone();
        blobs_rev.reverse();
        let tight_budget = (res_j1.pack_bytes.len() as u64).saturating_sub(1);
        let budget_j1 = Rdna2PackOptions {
            max_pack_bytes: Some(tight_budget),
            ..opts_j1
        };
        let budget_j4 = Rdna2PackOptions {
            jobs: Some(4),
            ..budget_j1.clone()
        };
        let res_b1 = pack_rdna2_blobs(&blobs, &budget_j1).unwrap();
        let res_b4 = pack_rdna2_blobs(&blobs_rev, &budget_j4).unwrap();
        assert_eq!(res_b1.pack_bytes, res_b4.pack_bytes);
        assert_eq!(res_b1.summary.to_json(), res_b4.summary.to_json());
        assert_eq!(res_b1.summary.to_text(), res_b4.summary.to_text());
        assert!(res_b4.summary.skipped_budget >= 1);
    }

    #[test]
    fn pack_rdna2_manifest_parallel_jobs_4_matches_jobs_1_bit_for_bit() {
        let root = make_temp_dir("playgta6_rdna2_manifest_parallel");
        let native_dir = root.join("native");
        fs::create_dir_all(&native_dir).unwrap();

        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let cs_bytes = build_minimal_compute_rdna2();

        let make_alu_ps = |iters: usize, dst_reg: u8| -> Vec<u8> {
            let mut words = Vec::with_capacity(iters * 2 + 4);
            for _ in 0..iters {
                words.push(enc_vop1(0x01, dst_reg, 242));
                words.push(enc_vop2(0x18, dst_reg + 1, 128 + 1, dst_reg));
            }
            words.extend_from_slice(&enc_exp(
                0,
                0xF,
                false,
                true,
                true,
                [dst_reg, dst_reg + 1, dst_reg, dst_reg + 1],
            ));
            words.push(enc_sopp(0x01, 0));
            words_to_bytes(&words)
        };

        let med_a = make_alu_ps(5, 1);
        let med_b = make_alu_ps(10, 2);
        let med_c = make_alu_ps(15, 3);
        let oversize = make_alu_ps(90, 4);

        let valid_blobs: Vec<(&str, Vec<u8>)> = vec![
            ("vertex", vs_bytes),
            ("pixel", ps_bytes),
            ("compute", cs_bytes),
            ("pixel", med_a),
            ("pixel", med_b),
            ("pixel", med_c),
            ("pixel", oversize.clone()),
        ];

        let mut stage_entries = Vec::new();
        let mut first_valid_entry = String::new();
        for (idx, (stage, bytes)) in valid_blobs.iter().enumerate() {
            let hash = fnv1a_64(bytes);
            let id = format_shader_hash(hash);
            fs::write(native_dir.join(format!("{id}.bin")), bytes).unwrap();
            let item = format!(
                r#"{{"code_identity": "{id}", "stage": "{stage}", "code_file": "native/{id}.bin"}}"#
            );
            if idx == 0 {
                first_valid_entry = item.clone();
            }
            stage_entries.push(item);
        }
        // Add duplicate manifest entry referencing the same `code_identity`.
        stage_entries.push(first_valid_entry);

        // Add invalid entries: hash mismatch, truncated blob, and missing file.
        let bad_hash_id = "0xdeadbeef00001111";
        fs::write(
            native_dir.join(format!("{bad_hash_id}.bin")),
            build_minimal_compute_rdna2(),
        )
        .unwrap();
        stage_entries.push(format!(
            r#"{{"code_identity": "{bad_hash_id}", "stage": "compute", "code_file": "native/{bad_hash_id}.bin"}}"#
        ));

        let short_bytes = [0xAAu8, 0xBBu8];
        let short_id = format_shader_hash(fnv1a_64(&short_bytes));
        fs::write(native_dir.join(format!("{short_id}.bin")), short_bytes).unwrap();
        stage_entries.push(format!(
            r#"{{"code_identity": "{short_id}", "stage": "pixel", "code_file": "native/{short_id}.bin"}}"#
        ));

        let missing_id = "0xdeadbeef00002222";
        stage_entries.push(format!(
            r#"{{"code_identity": "{missing_id}", "stage": "vertex", "code_file": "native/{missing_id}.bin"}}"#
        ));

        let manifest_path = root.join("manifest.json");
        let manifest_json = format!(
            "{{\n  \"schema\": \"playgta6-pipeline-dump\",\n  \"schema_version\": 1,\n  \"native_stages\": [\n    {}\n  ]\n}}\n",
            stage_entries.join(",\n    ")
        );
        fs::write(&manifest_path, &manifest_json).unwrap();

        let oversize_hash = fnv1a_64(&oversize);
        let baseline = pack_rdna2_manifest(
            &manifest_path,
            &Rdna2PackOptions {
                best_effort: true,
                jobs: Some(1),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("baseline best-effort manifest pack should succeed");
        let max_non_oversize = baseline
            .entries
            .iter()
            .filter(|e| e.hash != oversize_hash)
            .map(|e| e.wgsl.len())
            .max()
            .unwrap();

        let opts_j1 = Rdna2PackOptions {
            best_effort: true,
            max_wgsl_bytes: Some(max_non_oversize),
            jobs: Some(1),
            ..Rdna2PackOptions::default()
        };
        let opts_j4 = Rdna2PackOptions {
            jobs: Some(4),
            ..opts_j1.clone()
        };

        let res_j1 = pack_rdna2_manifest(&manifest_path, &opts_j1)
            .expect("manifest jobs=1 pack should succeed");
        let res_j4 = pack_rdna2_manifest(&manifest_path, &opts_j4)
            .expect("manifest jobs=4 pack should succeed");

        assert_eq!(res_j1.pack_bytes, res_j4.pack_bytes);
        assert_eq!(res_j1.summary.to_json(), res_j4.summary.to_json());
        assert_eq!(res_j1.summary.to_text(), res_j4.summary.to_text());
        assert_eq!(res_j1.summary, res_j4.summary);
        assert_eq!(res_j4.summary.translated_ok, 6);
        assert_eq!(res_j4.summary.manifest_duplicates_skipped, 1);
        assert_eq!(res_j4.summary.skipped_oversize, 1);
        assert_eq!(res_j4.summary.failed, 3);
        assert_eq!(res_j4.summary.failures.len(), 3);

        let decoded = read_pack(&res_j4.pack_bytes).expect("WGPK round-trip must succeed");
        assert_eq!(decoded.entries.len(), 6);
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        // Strict mode must return the exact same deterministic earliest error under jobs=1 and jobs=4.
        let strict_err_j1 = pack_rdna2_manifest(
            &manifest_path,
            &Rdna2PackOptions {
                best_effort: false,
                jobs: Some(1),
                ..Rdna2PackOptions::default()
            },
        )
        .expect_err("strict manifest pack must fail on bad entries");
        let strict_err_j4 = pack_rdna2_manifest(
            &manifest_path,
            &Rdna2PackOptions {
                best_effort: false,
                jobs: Some(4),
                ..Rdna2PackOptions::default()
            },
        )
        .expect_err("strict manifest pack must fail on bad entries");
        assert_eq!(strict_err_j1, strict_err_j4);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn budget_order_parse_and_display_round_trip() {
        assert_eq!(BudgetOrder::default(), BudgetOrder::HashOrder);
        for (order, expected) in [
            (BudgetOrder::HashOrder, "hash-order"),
            (BudgetOrder::SmallestFirst, "smallest-first"),
            (BudgetOrder::ManifestOrder, "manifest-order"),
        ] {
            assert_eq!(order.as_str(), expected);
            assert_eq!(order.to_string(), expected);
            assert_eq!(BudgetOrder::parse_str(expected).unwrap(), order);
            assert_eq!(
                BudgetOrder::parse_str(&expected.to_ascii_uppercase()).unwrap(),
                order
            );
        }

        let err = BudgetOrder::parse_str("largest-first").unwrap_err();
        assert!(err.contains("--budget-order"), "err: {err}");
        assert!(err.contains("smallest-first"), "err: {err}");
        assert!(err.contains("manifest-order"), "err: {err}");
        assert!(err.contains("hash-order"), "err: {err}");
    }

    #[test]
    fn pack_rdna2_budget_order_smallest_first_keeps_high_hash_small_shaders_over_low_hash_huge_shader()
    {
        // Build one low-hash huge shader (`0x0000...0001`) and three compact high-hash shaders
        // (`0x9000...`, `0xa000...`, `0xf000...`).
        let mut huge_words = Vec::new();
        for _ in 0..160 {
            huge_words.push(enc_vop1(0x01, 1, 242)); // v_mov_b32 v1, 1.0
            huge_words.push(enc_vop2(0x18, 2, 128 + 1, 1)); // v_lshlrev_b32 v2, 1, v1
        }
        huge_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [1, 2, 1, 2]));
        huge_words.push(enc_sopp(0x01, 0));
        let huge_ps_bytes = words_to_bytes(&huge_words);

        let small_vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let small_ps_bytes = build_textured_splash_ps_rdna2();
        let small_cs_bytes = build_minimal_compute_rdna2();

        let blobs = vec![
            Rdna2BlobInput::new("ps_0000000000000001.rdna2", huge_ps_bytes),
            Rdna2BlobInput::new("vs_9000000000000001.rdna2", small_vs_bytes),
            Rdna2BlobInput::new("ps_a000000000000002.rdna2", small_ps_bytes),
            Rdna2BlobInput::new("cs_f000000000000003.rdna2", small_cs_bytes),
        ];

        // Baseline unbounded pack to measure exact entry costs.
        let full = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default()).unwrap();
        assert_eq!(full.entries.len(), 4);
        let cost_of = |hash: u64| -> u64 {
            PACK_ENTRY_BYTES as u64
                + full
                    .entries
                    .iter()
                    .find(|e| e.hash == hash)
                    .unwrap()
                    .wgsl
                    .len() as u64
        };
        let huge_cost = cost_of(0x0000_0000_0000_0001);
        let small_vs_cost = cost_of(0x9000_0000_0000_0001);
        let small_ps_cost = cost_of(0xa000_0000_0000_0002);
        let small_cs_cost = cost_of(0xf000_0000_0000_0003);
        let all_smalls_cost = small_vs_cost + small_ps_cost + small_cs_cost;
        assert!(
            huge_cost > all_smalls_cost,
            "huge shader ({huge_cost}) must exceed sum of 3 compact shaders ({all_smalls_cost})"
        );

        // Tight budget that fits all 3 high-hash compact shaders (but NOT the low-hash huge shader).
        let tight_budget = PACK_HEADER_BYTES as u64 + all_smalls_cost;

        let hash_res = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(tight_budget),
                budget_order: BudgetOrder::HashOrder,
                jobs: Some(1),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("hash-order pack should succeed");

        let smallest_res_j1 = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(tight_budget),
                budget_order: BudgetOrder::SmallestFirst,
                jobs: Some(1),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("smallest-first pack (jobs=1) should succeed");

        let mut blobs_rev = blobs.clone();
        blobs_rev.reverse();
        let smallest_res_j4 = pack_rdna2_blobs(
            &blobs_rev,
            &Rdna2PackOptions {
                max_pack_bytes: Some(tight_budget),
                budget_order: BudgetOrder::SmallestFirst,
                jobs: Some(4),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("smallest-first pack (jobs=4, reversed) should succeed");

        // `hash-order` stalls on the low-hash huge shader (0 packed), while `smallest-first`
        // packs all 3 high-hash small shaders!
        assert_eq!(hash_res.summary.translated_ok, 0);
        assert_eq!(hash_res.summary.skipped_budget, 4);
        assert_eq!(hash_res.summary.omitted_by_budget, 4);
        assert_eq!(hash_res.summary.budget_order, BudgetOrder::HashOrder);

        assert!(smallest_res_j1.summary.translated_ok > hash_res.summary.translated_ok);
        assert_eq!(smallest_res_j1.summary.translated_ok, 3);
        assert_eq!(smallest_res_j1.summary.skipped_budget, 1);
        assert_eq!(smallest_res_j1.summary.omitted_by_budget, 1);
        assert_eq!(
            smallest_res_j1.summary.budget_order,
            BudgetOrder::SmallestFirst
        );
        assert_eq!(smallest_res_j1.pack_bytes.len() as u64, tight_budget);

        // Bit-for-bit determinism across jobs=1 vs jobs=4 and reversed input order.
        assert_eq!(smallest_res_j1.pack_bytes, smallest_res_j4.pack_bytes);
        assert_eq!(smallest_res_j1.summary, smallest_res_j4.summary);
        assert_eq!(
            smallest_res_j1.summary.to_json(),
            smallest_res_j4.summary.to_json()
        );

        // Verify WGPK entries are canonically sorted by ascending hash and pass Naga validation.
        let decoded = read_pack(&smallest_res_j1.pack_bytes).expect("WGPK decode must succeed");
        let packed_hashes: Vec<u64> = decoded.entries.iter().map(|e| e.hash).collect();
        assert_eq!(
            packed_hashes,
            vec![
                0x9000_0000_0000_0001,
                0xa000_0000_0000_0002,
                0xf000_0000_0000_0003,
            ]
        );
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        // Verify sanitized JSON and text fields.
        let json = smallest_res_j1.summary.to_json();
        let text = smallest_res_j1.summary.to_text();
        assert!(
            json.contains("\"budget_order\": \"smallest-first\""),
            "JSON:\n{json}"
        );
        assert!(json.contains("\"omitted_by_budget\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"skipped_budget\": 1"), "JSON:\n{json}");
        assert!(
            text.contains("budget_order:      smallest-first"),
            "text:\n{text}"
        );
        assert!(text.contains("omitted_by_budget: 1"), "text:\n{text}");

        // Also test a budget where `hash-order` packs the 1 huge shader (plus 0 smalls), while
        // `smallest-first` still packs all 3 high-hash small shaders (3 > 1).
        let medium_budget = PACK_HEADER_BYTES as u64 + huge_cost;
        let hash_med = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(medium_budget),
                budget_order: BudgetOrder::HashOrder,
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        let small_med = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(medium_budget),
                budget_order: BudgetOrder::SmallestFirst,
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        assert_eq!(hash_med.summary.translated_ok, 1);
        assert_eq!(hash_med.entries[0].hash, 0x0000_0000_0000_0001);
        assert_eq!(hash_med.summary.omitted_by_budget, 3);
        assert_eq!(small_med.summary.translated_ok, 3);
        assert_eq!(small_med.summary.omitted_by_budget, 1);
        assert!(small_med.entries.iter().all(|e| e.hash >= 0x9000_0000_0000_0001));
    }

    #[test]
    fn pack_rdna2_budget_order_manifest_order_prioritizes_early_stages_regardless_of_hash() {
        let root = make_temp_dir("playgta6_rdna2_budget_manifest_order");
        let native_dir = root.join("native");
        fs::create_dir_all(&native_dir).unwrap();

        let make_alu_ps = |iters: usize, dst_reg: u8| -> Vec<u8> {
            let mut words = Vec::with_capacity(iters * 2 + 4);
            for _ in 0..iters {
                words.push(enc_vop1(0x01, dst_reg, 242));
                words.push(enc_vop2(0x18, dst_reg + 1, 128 + 1, dst_reg));
            }
            words.extend_from_slice(&enc_exp(
                0,
                0xF,
                false,
                true,
                true,
                [dst_reg, dst_reg + 1, dst_reg, dst_reg + 1],
            ));
            words.push(enc_sopp(0x01, 0));
            words_to_bytes(&words)
        };

        // Four distinct compact shaders sorted by DESCENDING FNV-1a hash when written to
        // `manifest.json` so early manifest stages strictly have higher hashes than late stages!
        let mut stages: Vec<(u64, &str, Vec<u8>)> = vec![
            ("vertex", build_fullscreen_triangle_vs_rdna2()),
            ("pixel", build_textured_splash_ps_rdna2()),
            ("compute", build_minimal_compute_rdna2()),
            ("pixel", make_alu_ps(3, 1)),
        ]
        .into_iter()
        .map(|(st, bytes)| (fnv1a_64(&bytes), st, bytes))
        .collect();
        stages.sort_by(|a, b| b.0.cmp(&a.0)); // descending hash order in manifest.json

        let mut manifest_items = Vec::new();
        for (hash, stage, bytes) in &stages {
            let id = format_shader_hash(*hash);
            fs::write(native_dir.join(format!("{id}.bin")), bytes).unwrap();
            manifest_items.push(format!(
                r#"{{"code_identity": "{id}", "stage": "{stage}", "code_file": "native/{id}.bin"}}"#
            ));
        }

        let manifest_path = root.join("manifest.json");
        fs::write(
            &manifest_path,
            format!(
                "{{\n  \"schema\": \"playgta6-pipeline-dump\",\n  \"schema_version\": 1,\n  \"native_stages\": [\n    {}\n  ]\n}}\n",
                manifest_items.join(",\n    ")
            ),
        )
        .unwrap();

        let full = pack_rdna2_manifest(&manifest_path, &Rdna2PackOptions::default()).unwrap();
        assert_eq!(full.entries.len(), 4);
        let cost_of = |hash: u64| -> u64 {
            PACK_ENTRY_BYTES as u64
                + full
                    .entries
                    .iter()
                    .find(|e| e.hash == hash)
                    .unwrap()
                    .wgsl
                    .len() as u64
        };

        let early_0 = stages[0].0; // highest hash, first in manifest
        let early_1 = stages[1].0; // second-highest hash, second in manifest
        let late_2 = stages[2].0;
        let late_3 = stages[3].0; // lowest hash, last in manifest
        assert!(early_0 > early_1 && early_1 > late_2 && late_2 > late_3);

        // Budget that fits exactly the first two manifest stages (early_0 + early_1).
        let two_early_budget = PACK_HEADER_BYTES as u64 + cost_of(early_0) + cost_of(early_1);

        let manifest_j1 = pack_rdna2_manifest(
            &manifest_path,
            &Rdna2PackOptions {
                max_pack_bytes: Some(two_early_budget),
                budget_order: BudgetOrder::ManifestOrder,
                jobs: Some(1),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("manifest-order pack (jobs=1) should succeed");

        let manifest_j4 = pack_rdna2_manifest(
            &manifest_path,
            &Rdna2PackOptions {
                max_pack_bytes: Some(two_early_budget),
                budget_order: BudgetOrder::ManifestOrder,
                jobs: Some(4),
                ..Rdna2PackOptions::default()
            },
        )
        .expect("manifest-order pack (jobs=4) should succeed");

        assert_eq!(manifest_j1.pack_bytes, manifest_j4.pack_bytes);
        assert_eq!(manifest_j1.summary, manifest_j4.summary);
        assert_eq!(manifest_j1.summary.translated_ok, 2);
        assert_eq!(manifest_j1.summary.skipped_budget, 2);
        assert_eq!(manifest_j1.summary.omitted_by_budget, 2);
        assert_eq!(
            manifest_j1.summary.budget_order,
            BudgetOrder::ManifestOrder
        );
        assert_eq!(manifest_j1.pack_bytes.len() as u64, two_early_budget);

        // Packed entries must be `{early_1, early_0}` in canonical ascending hash order inside WGPK!
        let decoded = read_pack(&manifest_j1.pack_bytes).expect("WGPK decode must succeed");
        let packed_hashes: Vec<u64> = decoded.entries.iter().map(|e| e.hash).collect();
        assert_eq!(packed_hashes, vec![early_1, early_0]);
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        let json = manifest_j1.summary.to_json();
        assert!(
            json.contains("\"budget_order\": \"manifest-order\""),
            "JSON:\n{json}"
        );
        assert!(json.contains("\"omitted_by_budget\": 2"), "JSON:\n{json}");
        assert!(json.contains("\"skipped_budget\": 2"), "JSON:\n{json}");

        // Under default `hash-order` with the exact same manifest and budget, lowest hash `late_3`
        // is selected instead of `early_0`!
        let hash_res = pack_rdna2_manifest(
            &manifest_path,
            &Rdna2PackOptions {
                max_pack_bytes: Some(two_early_budget),
                budget_order: BudgetOrder::HashOrder,
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        assert!(
            hash_res.entries.iter().any(|e| e.hash == late_3),
            "hash-order should select lowest-hash late manifest stage"
        );
        assert!(
            hash_res.entries.iter().all(|e| e.hash != early_0),
            "hash-order should omit highest-hash early manifest stage under tight budget"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_budget_order_hash_order_matches_default_behavior() {
        let cs_bytes = build_minimal_compute_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();

        let blobs = vec![
            Rdna2BlobInput::new("z_vs_0000000000000030.rdna2", vs_bytes),
            Rdna2BlobInput::new("a_cs_0000000000000010.rdna2", cs_bytes),
            Rdna2BlobInput::new("m_ps_0000000000000020.rdna2", ps_bytes),
        ];

        let full = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default()).unwrap();
        let budget = PACK_HEADER_BYTES as u64
            + PACK_ENTRY_BYTES as u64
            + full.entries[0].wgsl.len() as u64;

        let default_res = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(budget),
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        let explicit_hash_res = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(budget),
                budget_order: BudgetOrder::HashOrder,
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();

        assert_eq!(default_res.pack_bytes, explicit_hash_res.pack_bytes);
        assert_eq!(default_res.entries, explicit_hash_res.entries);
        assert_eq!(default_res.summary, explicit_hash_res.summary);
        assert_eq!(
            default_res.summary.to_json(),
            explicit_hash_res.summary.to_json()
        );
        assert_eq!(
            default_res.summary.to_text(),
            explicit_hash_res.summary.to_text()
        );
        assert_eq!(default_res.summary.budget_order, BudgetOrder::HashOrder);
        assert_eq!(default_res.summary.translated_ok, 1);
        assert_eq!(default_res.entries[0].hash, 0x10);
        assert_eq!(default_res.summary.skipped_budget, 2);
        assert_eq!(default_res.summary.omitted_by_budget, 2);
        assert!(
            default_res
                .summary
                .to_json()
                .contains("\"budget_order\": \"hash-order\"")
        );
        assert!(
            default_res
                .summary
                .to_json()
                .contains("\"omitted_by_budget\": 2")
        );
        assert_eq!(default_res.summary.priority_hashes_requested, 0);
        assert_eq!(default_res.summary.priority_hashes_packed, 0);
        assert!(
            default_res
                .summary
                .to_json()
                .contains("\"priority_hashes_requested\": 0")
        );
        assert!(
            default_res
                .summary
                .to_json()
                .contains("\"priority_hashes_packed\": 0")
        );
    }

    #[test]
    fn parse_priority_hashes_accepts_prefixes_case_comments_dedup_and_rejects_invalid_lines() {
        let text = "\u{feff}\
# Live-observed hot boot & menu shaders\r\n\
\r\n\
  0x0011223344556677   # Start Screen VS\n\
FEDCBA9876543210\n\
0X0011223344556677  # duplicate with uppercase prefix -> ignored\n\
00000000000000aa # inline comment\n\
   # indented comment\n\
fedcba9876543210\n";

        let parsed = parse_priority_hashes(text).expect("valid priority hash list should parse");
        assert_eq!(
            parsed,
            vec![
                0x0011_2233_4455_6677,
                0xfedc_ba98_7654_3210,
                0x0000_0000_0000_00aa,
            ]
        );

        // Empty / comments-only input produces an empty priority list.
        assert_eq!(
            parse_priority_hashes("\n  # just a comment\r\n\t\n").unwrap(),
            Vec::<u64>::new()
        );

        // Invalid hex lines must return a clean `InvalidPriorityHashes` error with line number.
        let short_err = parse_priority_hashes("0x0011223344556677\n0x1234\n")
            .expect_err("short hex hash must be rejected");
        match &short_err {
            Rdna2PackError::InvalidPriorityHashes(msg) => {
                assert!(msg.contains("line 2"), "msg: {msg}");
                assert!(msg.contains("0x1234"), "msg: {msg}");
            }
            other => panic!("expected InvalidPriorityHashes, got {other:?}"),
        }

        let too_long_err = parse_priority_hashes("0x00112233445566778\n")
            .expect_err("17-digit hex hash must be rejected");
        assert!(matches!(
            too_long_err,
            Rdna2PackError::InvalidPriorityHashes(_)
        ));

        let non_hex_err = parse_priority_hashes("0x001122334455667g\n")
            .expect_err("non-hex char must be rejected");
        assert!(matches!(
            non_hex_err,
            Rdna2PackError::InvalidPriorityHashes(_)
        ));

        let two_tokens_err = parse_priority_hashes("0x0011223344556677 0x8899aabbccddeeff\n")
            .expect_err("two tokens on one line must be rejected");
        assert!(matches!(
            two_tokens_err,
            Rdna2PackError::InvalidPriorityHashes(_)
        ));

        // File loader round-trip & sanitized error checks.
        let root = make_temp_dir("playgta6_rdna2_priority_hashes_file");
        let file_path = root.join("priority.txt");
        fs::write(&file_path, text).unwrap();
        assert_eq!(load_priority_hashes_file(&file_path).unwrap(), parsed);

        let missing = root.join("does_not_exist.txt");
        let missing_err = load_priority_hashes_file(&missing)
            .expect_err("missing priority file must error")
            .to_string();
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(!missing_err.contains(&root_str), "err leaked host path: {missing_err}");

        let dir_err = load_priority_hashes_file(&root)
            .expect_err("directory path must error")
            .to_string();
        assert!(!dir_err.contains(&root_str), "err leaked host path: {dir_err}");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_priority_hashes_packs_larger_priority_shader_ahead_of_smaller_non_priority_under_tight_budget()
    {
        // Build one large high-hash shader (`0xf000...0001`) and three compact lower-hash shaders.
        let mut large_words = Vec::new();
        for _ in 0..140 {
            large_words.push(enc_vop1(0x01, 1, 242)); // v_mov_b32 v1, 1.0
            large_words.push(enc_vop2(0x18, 2, 128 + 1, 1)); // v_lshlrev_b32 v2, 1, v1
        }
        large_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [1, 2, 1, 2]));
        large_words.push(enc_sopp(0x01, 0));
        let large_ps_bytes = words_to_bytes(&large_words);

        let small_vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let small_ps_bytes = build_textured_splash_ps_rdna2();
        let small_cs_bytes = build_minimal_compute_rdna2();

        let priority_hash = 0xf000_0000_0000_0001u64;
        let small_a = 0x0000_0000_0000_0010u64;
        let small_b = 0x0000_0000_0000_0020u64;
        let small_c = 0x0000_0000_0000_0030u64;

        let blobs = vec![
            Rdna2BlobInput::new("ps_f000000000000001.rdna2", large_ps_bytes),
            Rdna2BlobInput::new("cs_0000000000000010.rdna2", small_cs_bytes),
            Rdna2BlobInput::new("ps_0000000000000020.rdna2", small_ps_bytes),
            Rdna2BlobInput::new("vs_0000000000000030.rdna2", small_vs_bytes),
        ];

        let full = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default()).unwrap();
        let cost_of = |hash: u64| -> u64 {
            PACK_ENTRY_BYTES as u64
                + full
                    .entries
                    .iter()
                    .find(|e| e.hash == hash)
                    .unwrap()
                    .wgsl
                    .len() as u64
        };

        let large_cost = cost_of(priority_hash);
        let cost_a = cost_of(small_a);
        let cost_b = cost_of(small_b);
        let cost_c = cost_of(small_c);
        let mut small_costs = [(small_a, cost_a), (small_b, cost_b), (small_c, cost_c)];
        small_costs.sort_by_key(|&(_, c)| c);
        let smallest_non_priority_hash = small_costs[0].0;
        let smallest_non_priority_cost = small_costs[0].1;
        assert!(
            large_cost > cost_a + cost_b + cost_c,
            "large priority shader must exceed sum of all three small shaders"
        );

        // Tight budget that fits `large + smallest_non_priority`, OR all 3 small shaders without `large`.
        let tight_budget = PACK_HEADER_BYTES as u64 + large_cost + smallest_non_priority_cost;

        // Without priority hashes under `smallest-first`: packs all 3 small shaders and omits `large`.
        let without_priority = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(tight_budget),
                budget_order: BudgetOrder::SmallestFirst,
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        assert_eq!(without_priority.summary.translated_ok, 3);
        assert_eq!(without_priority.summary.omitted_by_budget, 1);
        assert_eq!(without_priority.summary.priority_hashes_requested, 0);
        assert_eq!(without_priority.summary.priority_hashes_packed, 0);
        assert!(
            without_priority
                .entries
                .iter()
                .all(|e| e.hash != priority_hash),
            "without --priority-hashes, smallest-first should omit the large shader"
        );

        // With priority hashes listing `priority_hash` (plus an absent hash to verify requested vs packed):
        let absent_hash = 0xdead_beef_cafe_babeu64;
        let with_priority_j1 = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(tight_budget),
                budget_order: BudgetOrder::SmallestFirst,
                priority_hashes: vec![priority_hash, absent_hash],
                jobs: Some(1),
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();

        let mut blobs_rev = blobs.clone();
        blobs_rev.reverse();
        let with_priority_j4 = pack_rdna2_blobs(
            &blobs_rev,
            &Rdna2PackOptions {
                max_pack_bytes: Some(tight_budget),
                budget_order: BudgetOrder::SmallestFirst,
                priority_hashes: vec![priority_hash, absent_hash, priority_hash],
                jobs: Some(4),
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();

        assert_eq!(with_priority_j1.pack_bytes, with_priority_j4.pack_bytes);
        assert_eq!(with_priority_j1.summary, with_priority_j4.summary);
        assert_eq!(
            with_priority_j1.summary.to_json(),
            with_priority_j4.summary.to_json()
        );
        assert_eq!(with_priority_j1.summary.priority_hashes_requested, 2);
        assert_eq!(with_priority_j1.summary.priority_hashes_packed, 1);
        assert_eq!(with_priority_j1.summary.translated_ok, 2);
        assert_eq!(with_priority_j1.summary.omitted_by_budget, 2);
        assert_eq!(with_priority_j1.pack_bytes.len() as u64, tight_budget);

        // Packed entries must be `{smallest_non_priority_hash, priority_hash}` in canonical ascending hash order.
        let decoded = read_pack(&with_priority_j1.pack_bytes).expect("WGPK decode must succeed");
        let packed_hashes: Vec<u64> = decoded.entries.iter().map(|e| e.hash).collect();
        assert_eq!(packed_hashes, vec![smallest_non_priority_hash, priority_hash]);
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        let json = with_priority_j1.summary.to_json();
        let text = with_priority_j1.summary.to_text();
        assert!(
            json.contains("\"priority_hashes_requested\": 2"),
            "JSON:\n{json}"
        );
        assert!(
            json.contains("\"priority_hashes_packed\": 1"),
            "JSON:\n{json}"
        );
        assert!(
            text.contains("priority_hashes_requested: 2"),
            "text:\n{text}"
        );
        assert!(
            text.contains("priority_hashes_packed:    1"),
            "text:\n{text}"
        );
    }

    #[test]
    fn pack_rdna2_priority_hashes_respects_priority_order_and_skips_oversize_priority_cleanly() {
        let make_alu_ps = |iters: usize, dst_reg: u8| -> Vec<u8> {
            let mut words = Vec::with_capacity(iters * 2 + 4);
            for _ in 0..iters {
                words.push(enc_vop1(0x01, dst_reg, 242));
                words.push(enc_vop2(0x18, dst_reg + 1, 128 + 1, dst_reg));
            }
            words.extend_from_slice(&enc_exp(
                0,
                0xF,
                false,
                true,
                true,
                [dst_reg, dst_reg + 1, dst_reg, dst_reg + 1],
            ));
            words.push(enc_sopp(0x01, 0));
            words_to_bytes(&words)
        };

        // Three same-size priority candidates with distinct hashes, plus one huge priority shader
        // and one compact non-priority shader.
        let p_first = 0xc000_0000_0000_0003u64;
        let p_second = 0x8000_0000_0000_0002u64;
        let p_third = 0x1000_0000_0000_0001u64;
        let p_huge = 0xd000_0000_0000_0004u64;
        let non_prio = 0x0000_0000_0000_0005u64;

        let same_size_a = make_alu_ps(6, 1);
        let same_size_b = make_alu_ps(6, 2);
        let same_size_c = make_alu_ps(6, 3);
        let huge_ps = make_alu_ps(160, 4);
        let tiny_cs = build_minimal_compute_rdna2();

        let blobs = vec![
            Rdna2BlobInput::new("ps_c000000000000003.rdna2", same_size_a),
            Rdna2BlobInput::new("ps_8000000000000002.rdna2", same_size_b),
            Rdna2BlobInput::new("ps_1000000000000001.rdna2", same_size_c),
            Rdna2BlobInput::new("ps_d000000000000004.rdna2", huge_ps),
            Rdna2BlobInput::new("cs_0000000000000005.rdna2", tiny_cs),
        ];

        let full = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default()).unwrap();
        let cost_of = |hash: u64| -> u64 {
            PACK_ENTRY_BYTES as u64
                + full
                    .entries
                    .iter()
                    .find(|e| e.hash == hash)
                    .unwrap()
                    .wgsl
                    .len() as u64
        };

        // 1) Budget fits only `p_first + p_second` (strictly less than adding `p_third` or `non_prio`?
        // Actually exact fit for `p_first + p_second` leaves 0 bytes remaining).
        let order_budget = PACK_HEADER_BYTES as u64 + cost_of(p_first) + cost_of(p_second);
        let order_res = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(order_budget),
                budget_order: BudgetOrder::HashOrder,
                priority_hashes: vec![p_first, p_second, p_third],
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        let packed_hashes: Vec<u64> = order_res.entries.iter().map(|e| e.hash).collect();
        // Priority order picks `p_first` (0xc000...) and `p_second` (0x8000...), NOT `p_third` (0x1000...)!
        assert_eq!(packed_hashes, vec![p_second, p_first]);
        assert_eq!(order_res.summary.priority_hashes_requested, 3);
        assert_eq!(order_res.summary.priority_hashes_packed, 2);
        assert_eq!(order_res.summary.omitted_by_budget, 3);

        // 2) If the first priority shader (`p_huge`) exceeds the total budget, it must be skipped
        // cleanly WITHOUT blocking the second priority shader (`p_first`) or smaller non-priority (`non_prio`).
        let skip_huge_budget = PACK_HEADER_BYTES as u64 + cost_of(p_first) + cost_of(non_prio);
        assert!(cost_of(p_huge) > skip_huge_budget);
        let skip_res = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                max_pack_bytes: Some(skip_huge_budget),
                budget_order: BudgetOrder::SmallestFirst,
                priority_hashes: vec![p_huge, p_first],
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        let skip_packed: Vec<u64> = skip_res.entries.iter().map(|e| e.hash).collect();
        assert_eq!(skip_packed, vec![non_prio, p_first]);
        assert_eq!(skip_res.summary.priority_hashes_requested, 2);
        assert_eq!(skip_res.summary.priority_hashes_packed, 1);
        assert_eq!(skip_res.summary.omitted_by_budget, 3);

        // 3) Unbounded pack (no `--max-pack-bytes`) still reports priority metrics accurately.
        let unbounded = pack_rdna2_blobs(
            &blobs,
            &Rdna2PackOptions {
                priority_hashes: vec![p_first, p_second, 0x9999_9999_9999_9999],
                ..Rdna2PackOptions::default()
            },
        )
        .unwrap();
        assert_eq!(unbounded.entries.len(), 5);
        assert_eq!(unbounded.summary.priority_hashes_requested, 3);
        assert_eq!(unbounded.summary.priority_hashes_packed, 2);
    }

    #[test]
    fn parses_mesh_filename_hints_and_packs_mesh_blobs_and_manifest_with_salted_key() {
        assert_eq!(
            parse_filename_hint("ms_0123456789abcdef.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Mesh,
                hash: Some(0x0123_4567_89ab_cdef),
            }
        );
        assert_eq!(
            parse_filename_hint("0123456789abcdef.ms.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Mesh,
                hash: Some(0x0123_4567_89ab_cdef),
            }
        );
        assert_eq!(
            parse_filename_hint("meshlet.ms.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Mesh,
                hash: None,
            }
        );
        assert_eq!(
            parse_filename_hint("mesh_fedcba9876543210.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Mesh,
                hash: Some(0xfedc_ba98_7654_3210),
            }
        );

        // Synthetic GFX10.3 mesh shader (3 verts, 1 prim, pos0 + param0 + prim).
        let enc_sop1 = |op: u8, sdst: u8, ssrc0: u8| -> u32 {
            (0b1011_1110_1u32 << 23)
                | (u32::from(sdst & 0x7F) << 16)
                | (u32::from(op) << 8)
                | u32::from(ssrc0)
        };
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 124, 0xFF)); // s_mov_b32 m0, (1 << 12) | 3
        words.push((1u32 << 12) | 3u32);
        words.push(enc_sopp(0x10, 9)); // s_sendmsg MSG_GS_ALLOC_REQ
        words.push(enc_vop1(0x01, 1, 242)); // v1 = 1.0
        words.push(enc_vop1(0x01, 2, 242));
        words.push(enc_vop1(0x01, 3, 242));
        words.push(enc_vop1(0x01, 4, 242));
        words.push(enc_vop1(0x01, 8, 0xFF));
        words.push((2u32 << 20) | (1u32 << 10));
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [0, 1, 2, 3]));
        words.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [4, 4, 4, 4]));
        words.extend_from_slice(&enc_exp(20, 0x1, false, true, false, [8, 0, 0, 0]));
        words.push(enc_sopp(0x10, 3)); // s_sendmsg MSG_GS_DONE
        words.push(enc_sopp(0x01, 0)); // s_endpgm
        let ms_bytes = words_to_bytes(&words);
        let raw_hash = fnv1a_64(&ms_bytes);
        let expected_pack_key = raw_hash ^ MESH_COMPUTE_KEY_SALT;

        // 1) Blob packing via `.ms.bin` filename hint
        let blob_res = pack_rdna2_blobs(
            &[Rdna2BlobInput::new(
                format!("{raw_hash:016x}.ms.bin"),
                ms_bytes.clone(),
            )],
            &Rdna2PackOptions::default(),
        )
        .expect("packing .ms.bin blob must succeed");

        assert_eq!(blob_res.entries.len(), 1);
        assert_eq!(blob_res.entries[0].hash, expected_pack_key);
        assert_eq!(blob_res.entries[0].stage, ShaderStage::Compute);
        assert_eq!(blob_res.entries[0].stage.to_u8(), 3);
        let decoded = read_pack(&blob_res.pack_bytes).expect("valid WGPK v1");
        assert_eq!(decoded.entries.len(), 1);
        assert_eq!(decoded.entries[0].hash, expected_pack_key);
        assert_eq!(decoded.entries[0].stage, ShaderStage::Compute);
        assert!(decoded.entries[0].wgsl.contains("@group(3) @binding(0)"));

        let json = blob_res.summary.to_json();
        assert!(
            json.contains(&format!("\"hash_fnv1a64\": \"0x{expected_pack_key:016x}\"")),
            "JSON:\n{json}"
        );
        assert!(json.contains("\"stage\": \"compute\""), "JSON:\n{json}");
        assert!(
            json.contains(
                "\"mesh_output\": {\"max_vertices\": 3, \"max_primitives\": 1, \"vertex_words\": 8}"
            ),
            "JSON:\n{json}"
        );

        // 2) Manifest packing with `"stage": "mesh"` and `"stage": "ms"`
        let root = make_temp_dir("playgta6_mesh_manifest");
        let ms_file = root.join("mesh_stage.bin");
        fs::write(&ms_file, &ms_bytes).unwrap();

        for stage_str in ["mesh", "ms"] {
            let manifest_path = root.join(format!("manifest_{stage_str}.json"));
            let manifest_json = format!(
                "{{\"schema\": \"playgta6-pipeline-dump\", \"schema_version\": 1, \"native_stages\": [{{\"code_identity\": \"0x{raw_hash:016x}\", \"stage\": \"{stage_str}\", \"code_file\": \"mesh_stage.bin\"}}]}}"
            );
            fs::write(&manifest_path, manifest_json).unwrap();

            let res = pack_rdna2_manifest(&manifest_path, &Rdna2PackOptions::default())
                .expect("manifest mesh packing must succeed");
            assert_eq!(res.entries.len(), 1);
            assert_eq!(res.entries[0].hash, expected_pack_key);
            assert_eq!(res.entries[0].stage, ShaderStage::Compute);
            assert_eq!(
                res.summary.entries[0].mesh_output,
                Some(MeshOutputLayout {
                    max_vertices: 3,
                    max_primitives: 1,
                    vertex_words: 8,
                })
            );
        }

        let _ = fs::remove_dir_all(&root);
    }
}
