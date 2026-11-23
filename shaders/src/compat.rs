use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::{self, Write as _};

use crate::convert::{SPIRV_HEADER_BYTES, SPIRV_MAGIC};

// ---------------------------------------------------------------------------
// WebGPU Baseline Default Limits (GPUSupportedLimits defaults)
// ---------------------------------------------------------------------------

/// WebGPU baseline default `maxComputeWorkgroupSizeX`.
pub const WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_X: u32 = 256;

/// WebGPU baseline default `maxComputeWorkgroupSizeY`.
pub const WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Y: u32 = 256;

/// WebGPU baseline default `maxComputeWorkgroupSizeZ`.
pub const WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Z: u32 = 64;

/// WebGPU baseline default `maxComputeInvocationsPerWorkgroup`.
pub const WEBGPU_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP: u32 = 256;

/// WebGPU baseline default `maxBindGroups` (valid bind group indices are `0..4`).
pub const WEBGPU_MAX_BIND_GROUPS: u32 = 4;

/// WebGPU baseline default `maxStorageBuffersPerShaderStage`.
pub const WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE: usize = 8;

/// WebGPU baseline default `maxUniformBuffersPerShaderStage`.
pub const WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE: usize = 12;

// ---------------------------------------------------------------------------
// SPIR-V Binary Format Constants (Khronos SPIR-V Unified Registry)
// ---------------------------------------------------------------------------

const SPIRV_HEADER_WORDS: usize = 5;

// Opcodes
const OP_EXTENSION: u16 = 10;
const OP_EXT_INST_IMPORT: u16 = 11;
const OP_EXT_INST: u16 = 12;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_CAPABILITY: u16 = 17;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_POINTER: u16 = 32;
const OP_VARIABLE: u16 = 59;
const OP_DECORATE: u16 = 71;
const OP_EXECUTION_MODE_ID: u16 = 331;

// Decorations
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_NON_UNIFORM: u32 = 5300;

// Capabilities
const CAP_GEOMETRY: u32 = 2;
const CAP_TESSELLATION: u32 = 3;
const CAP_FLOAT64: u32 = 10;
const CAP_INT64: u32 = 11;
const CAP_SUBGROUP_DISPATCH: u32 = 58;
const CAP_GROUP_NON_UNIFORM: u32 = 61;
const CAP_GROUP_NON_UNIFORM_VOTE: u32 = 62;
const CAP_GROUP_NON_UNIFORM_ARITHMETIC: u32 = 63;
const CAP_GROUP_NON_UNIFORM_BALLOT: u32 = 64;
const CAP_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
const CAP_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE: u32 = 66;
const CAP_GROUP_NON_UNIFORM_CLUSTERED: u32 = 67;
const CAP_GROUP_NON_UNIFORM_QUAD: u32 = 68;
const CAP_SUBGROUP_BALLOT_KHR: u32 = 4423;
const CAP_SUBGROUP_VOTE_KHR: u32 = 4431;
const CAP_DENORM_PRESERVE: u32 = 4464;
const CAP_DENORM_FLUSH_TO_ZERO: u32 = 4465;
const CAP_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4466;
const CAP_ROUNDING_MODE_RTE: u32 = 4467;
const CAP_ROUNDING_MODE_RTZ: u32 = 4468;
const CAP_RAY_QUERY_PROVISIONAL_KHR: u32 = 4471;
const CAP_RAY_QUERY_KHR: u32 = 4472;
const CAP_RAY_TRACING_KHR: u32 = 4479;
const CAP_DESCRIPTOR_HEAP_EXT: u32 = 5128;
const CAP_MESH_SHADING_NV: u32 = 5266;
const CAP_MESH_SHADING_EXT: u32 = 5283;
const CAP_GROUP_NON_UNIFORM_PARTITIONED_EXT: u32 = 5297;
const CAP_SHADER_NON_UNIFORM: u32 = 5301;
const CAP_RUNTIME_DESCRIPTOR_ARRAY: u32 = 5302;
const CAP_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING: u32 = 5306;
const CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING: u32 = 5307;
const CAP_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING: u32 = 5308;
const CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING: u32 = 5309;
const CAP_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING: u32 = 5310;
const CAP_UNIFORM_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING: u32 = 5311;
const CAP_STORAGE_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING: u32 = 5312;
const CAP_RAY_TRACING_POSITION_FETCH_KHR: u32 = 5336;
const CAP_RAY_TRACING_NV: u32 = 5340;
const CAP_RAY_TRACING_MOTION_BLUR_NV: u32 = 5341;
const CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES: u32 = 5347;
const CAP_RAY_TRACING_PROVISIONAL_KHR: u32 = 5353;
const CAP_RAY_TRACING_OPACITY_MICROMAP_EXT: u32 = 5381;
const CAP_BINDLESS_TEXTURE_NV: u32 = 5390;
const CAP_RAY_QUERY_POSITION_FETCH_KHR: u32 = 5391;
const CAP_GROUP_NON_UNIFORM_ROTATE_KHR: u32 = 6026;
const CAP_BINDLESS_IMAGES_INTEL: u32 = 6528;

// Execution Models
const EXEC_MODEL_TESSELLATION_CONTROL: u32 = 1;
const EXEC_MODEL_TESSELLATION_EVALUATION: u32 = 2;
const EXEC_MODEL_GEOMETRY: u32 = 3;
const EXEC_MODEL_TASK_NV: u32 = 5267;
const EXEC_MODEL_MESH_NV: u32 = 5268;
const EXEC_MODEL_RAY_GENERATION_KHR: u32 = 5313;
const EXEC_MODEL_INTERSECTION_KHR: u32 = 5314;
const EXEC_MODEL_ANY_HIT_KHR: u32 = 5315;
const EXEC_MODEL_CLOSEST_HIT_KHR: u32 = 5316;
const EXEC_MODEL_MISS_KHR: u32 = 5317;
const EXEC_MODEL_CALLABLE_KHR: u32 = 5318;
const EXEC_MODEL_TASK_EXT: u32 = 5364;
const EXEC_MODEL_MESH_EXT: u32 = 5365;

// Execution Modes
const EXEC_MODE_LOCAL_SIZE: u32 = 17;
const EXEC_MODE_SUBGROUP_SIZE: u32 = 35;
const EXEC_MODE_SUBGROUPS_PER_WORKGROUP: u32 = 36;
const EXEC_MODE_SUBGROUPS_PER_WORKGROUP_ID: u32 = 37;
const EXEC_MODE_SUBGROUP_UNIFORM_CONTROL_FLOW_KHR: u32 = 4421;
const EXEC_MODE_DENORM_PRESERVE: u32 = 4459;
const EXEC_MODE_DENORM_FLUSH_TO_ZERO: u32 = 4460;
const EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE: u32 = 4461;
const EXEC_MODE_ROUNDING_MODE_RTE: u32 = 4462;
const EXEC_MODE_ROUNDING_MODE_RTZ: u32 = 4463;

// Storage Classes
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER: u32 = 5349;

// ---------------------------------------------------------------------------
// Public Types
// ---------------------------------------------------------------------------

/// Controls which safe SPIR-V pre-passes run before Naga lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SanitizeOptions {
    /// Strip `SPV_KHR_float_controls` extension, float-control capabilities, and
    /// float-control execution modes (`SignedZeroInfNanPreserve`, `DenormFlushToZero`,
    /// `DenormPreserve`, `RoundingModeRTE/RTZ`).
    pub strip_float_controls: bool,
    /// Strip `SPV_KHR_non_semantic_info`, `NonSemantic.*` `OpExtInstImport`s, and
    /// associated non-semantic `OpExtInst` calls.
    pub strip_non_semantic: bool,
}

impl Default for SanitizeOptions {
    fn default() -> Self {
        Self {
            strip_float_controls: true,
            strip_non_semantic: true,
        }
    }
}

impl SanitizeOptions {
    /// Leave the SPIR-V stream untouched (audit-only mode).
    pub const fn disabled() -> Self {
        Self {
            strip_float_controls: false,
            strip_non_semantic: false,
        }
    }
}

/// Severity of a WebGPU compatibility finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompatSeverity {
    /// Safely rewritten or stripped by a pre-pass so WebGPU lowering can continue.
    Rewritten,
    /// Exceeds WebGPU default limits or requires runtime attention, but may still lower.
    Warning,
    /// Hard WebGPU baseline incompatibility requiring upstream shader/compiler lowering.
    Blocker,
}

impl CompatSeverity {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rewritten => "Rewritten",
            Self::Warning => "Warning",
            Self::Blocker => "Blocker",
        }
    }
}

impl fmt::Display for CompatSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Structured category for a WebGPU compatibility finding or rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompatIssueKind {
    // Rewritten constructs
    FloatControlsStripped,
    NonSemanticStripped,
    // Unstripped constructs (when sanitization is disabled)
    FloatControlsUnsupported,
    NonSemanticExtensionPresent,
    // Hard WebGPU blockers
    MeshShader,
    RayTracing,
    Float64,
    Int64,
    SubgroupWave64,
    BindlessDescriptorIndexing,
    PushConstants,
    GeometryShader,
    Tessellation,
    PhysicalStorageBuffer,
    WorkgroupSizeLimitExceeded,
    BindGroupLimitExceeded,
    // WebGPU baseline limit warnings
    StorageBufferLimitExceeded,
    UniformBufferLimitExceeded,
    OverrideWorkgroupSize,
}

impl CompatIssueKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FloatControlsStripped => "FloatControlsStripped",
            Self::NonSemanticStripped => "NonSemanticStripped",
            Self::FloatControlsUnsupported => "FloatControlsUnsupported",
            Self::NonSemanticExtensionPresent => "NonSemanticExtensionPresent",
            Self::MeshShader => "MeshShader",
            Self::RayTracing => "RayTracing",
            Self::Float64 => "Float64",
            Self::Int64 => "Int64",
            Self::SubgroupWave64 => "SubgroupWave64",
            Self::BindlessDescriptorIndexing => "BindlessDescriptorIndexing",
            Self::PushConstants => "PushConstants",
            Self::GeometryShader => "GeometryShader",
            Self::Tessellation => "Tessellation",
            Self::PhysicalStorageBuffer => "PhysicalStorageBuffer",
            Self::WorkgroupSizeLimitExceeded => "WorkgroupSizeLimitExceeded",
            Self::BindGroupLimitExceeded => "BindGroupLimitExceeded",
            Self::StorageBufferLimitExceeded => "StorageBufferLimitExceeded",
            Self::UniformBufferLimitExceeded => "UniformBufferLimitExceeded",
            Self::OverrideWorkgroupSize => "OverrideWorkgroupSize",
        }
    }
}

impl fmt::Display for CompatIssueKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single compatibility rewrite, warning, or blocker detected during SPIR-V/Naga auditing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatIssue {
    pub severity: CompatSeverity,
    pub kind: CompatIssueKind,
    pub message: String,
    pub hint: Option<String>,
}

/// Aggregated WebGPU compatibility audit for a single shader.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompatAudit {
    pub issues: Vec<CompatIssue>,
    pub rewritten_instructions: usize,
    pub original_word_count: usize,
    pub sanitized_word_count: usize,
}

impl CompatAudit {
    pub fn has_blockers(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == CompatSeverity::Blocker)
    }

    pub fn blocker_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == CompatSeverity::Blocker)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == CompatSeverity::Warning)
            .count()
    }

    pub fn rewritten_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == CompatSeverity::Rewritten)
            .count()
    }

    pub fn blockers(&self) -> impl Iterator<Item = &CompatIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == CompatSeverity::Blocker)
    }

    pub fn warnings(&self) -> impl Iterator<Item = &CompatIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == CompatSeverity::Warning)
    }

    pub fn rewrites(&self) -> impl Iterator<Item = &CompatIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == CompatSeverity::Rewritten)
    }

    /// Push an issue if an identical `(severity, kind, message)` has not already been recorded.
    pub fn push_unique(&mut self, issue: CompatIssue) {
        if !self.issues.iter().any(|existing| {
            existing.severity == issue.severity
                && existing.kind == issue.kind
                && existing.message == issue.message
        }) {
            self.issues.push(issue);
        }
    }

    /// Push a blocker/warning only if no issue of the same `(severity, kind)` exists yet.
    fn push_once_per_kind(&mut self, issue: CompatIssue) {
        if !self
            .issues
            .iter()
            .any(|existing| existing.severity == issue.severity && existing.kind == issue.kind)
        {
            self.issues.push(issue);
        }
    }

    /// Merge another audit (for example, post-Naga limits audit) into this one.
    ///
    /// Deduplicates overlapping `(severity, kind)` findings that were already caught during the
    /// SPIR-V pre-pass so single-shader issue counts stay accurate.
    pub fn merge(&mut self, other: CompatAudit) {
        self.rewritten_instructions += other.rewritten_instructions;
        for issue in other.issues {
            if issue.severity == CompatSeverity::Rewritten {
                self.push_unique(issue);
            } else {
                self.push_once_per_kind(issue);
            }
        }
    }
}

/// Output of [`sanitize_spirv_words`]: the rewritten SPIR-V word stream plus its audit record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SanitizedSpirv {
    pub words: Vec<u32>,
    pub audit: CompatAudit,
}

/// Errors produced by low-level SPIR-V header or instruction-stream framing validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpirvStreamError {
    TooShort {
        actual_words: usize,
        min_words: usize,
    },
    UnalignedByteLength {
        byte_len: usize,
    },
    InvalidMagic {
        magic: u32,
        is_big_endian: bool,
    },
    InvalidWordCount {
        word_offset: usize,
        opcode: u16,
    },
    TruncatedInstruction {
        word_offset: usize,
        opcode: u16,
        declared_words: usize,
        available_words: usize,
    },
}

impl fmt::Display for SpirvStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort {
                actual_words,
                min_words,
            } => write!(
                f,
                "SPIR-V input too short: {actual_words} words (minimum {min_words} words / {SPIRV_HEADER_BYTES} bytes)"
            ),
            Self::UnalignedByteLength { byte_len } => write!(
                f,
                "SPIR-V byte length {byte_len} is not a multiple of 4 bytes"
            ),
            Self::InvalidMagic {
                magic,
                is_big_endian: true,
            } => write!(
                f,
                "big-endian SPIR-V magic 0x{magic:08x} is unsupported (expected little-endian 0x{SPIRV_MAGIC:08x})"
            ),
            Self::InvalidMagic {
                magic,
                is_big_endian: false,
            } => write!(
                f,
                "invalid SPIR-V magic 0x{magic:08x} (expected 0x{SPIRV_MAGIC:08x})"
            ),
            Self::InvalidWordCount {
                word_offset,
                opcode,
            } => write!(
                f,
                "invalid SPIR-V instruction at word offset {word_offset} (opcode {opcode}): word_count is 0"
            ),
            Self::TruncatedInstruction {
                word_offset,
                opcode,
                declared_words,
                available_words,
            } => write!(
                f,
                "truncated SPIR-V instruction at word offset {word_offset} (opcode {opcode}): declared {declared_words} words, only {available_words} remaining"
            ),
        }
    }
}

impl std::error::Error for SpirvStreamError {}

// ---------------------------------------------------------------------------
// SPIR-V Byte <-> Word Helpers
// ---------------------------------------------------------------------------

/// Safely convert a little-endian SPIR-V byte slice into 32-bit words without panicking.
pub fn spirv_bytes_to_words(bytes: &[u8]) -> Result<Vec<u32>, SpirvStreamError> {
    if bytes.len() < SPIRV_HEADER_BYTES {
        return Err(SpirvStreamError::TooShort {
            actual_words: bytes.len() / 4,
            min_words: SPIRV_HEADER_WORDS,
        });
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(SpirvStreamError::UnalignedByteLength {
            byte_len: bytes.len(),
        });
    }

    let mut words = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(words)
}

/// Convert a SPIR-V 32-bit word stream back into little-endian bytes.
pub fn spirv_words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Convenience wrapper around [`sanitize_spirv_words`] operating directly on raw SPIR-V bytes.
pub fn sanitize_spirv_bytes(
    bytes: &[u8],
    options: &SanitizeOptions,
) -> Result<(Vec<u8>, CompatAudit), SpirvStreamError> {
    let words = spirv_bytes_to_words(bytes)?;
    let sanitized = sanitize_spirv_words(&words, options)?;
    Ok((spirv_words_to_bytes(&sanitized.words), sanitized.audit))
}

// ---------------------------------------------------------------------------
// SPIR-V Word-Stream Rewriter & Pre-Pass Auditor
// ---------------------------------------------------------------------------

/// Parse, audit, and safely sanitize a SPIR-V 32-bit word stream for WebGPU lowering.
///
/// Guarantees bounds-checked parsing and never panics on malformed or truncated inputs.
pub fn sanitize_spirv_words(
    words: &[u32],
    options: &SanitizeOptions,
) -> Result<SanitizedSpirv, SpirvStreamError> {
    if words.len() < SPIRV_HEADER_WORDS {
        return Err(SpirvStreamError::TooShort {
            actual_words: words.len(),
            min_words: SPIRV_HEADER_WORDS,
        });
    }

    let magic = words[0];
    if magic != SPIRV_MAGIC {
        return Err(SpirvStreamError::InvalidMagic {
            magic,
            is_big_endian: magic == SPIRV_MAGIC.swap_bytes(),
        });
    }

    let mut out = Vec::with_capacity(words.len());
    out.extend_from_slice(&words[..SPIRV_HEADER_WORDS]);

    let mut audit = CompatAudit {
        issues: Vec::new(),
        rewritten_instructions: 0,
        original_word_count: words.len(),
        sanitized_word_count: 0,
    };

    let mut stripped_non_semantic_sets: HashSet<u32> = HashSet::new();
    let mut stripped_non_semantic_calls: usize = 0;

    let mut offset = SPIRV_HEADER_WORDS;
    while offset < words.len() {
        let header = words[offset];
        let word_count = (header >> 16) as usize;
        let opcode = (header & 0xffff) as u16;

        if word_count == 0 {
            return Err(SpirvStreamError::InvalidWordCount {
                word_offset: offset,
                opcode,
            });
        }

        let Some(end) = offset.checked_add(word_count) else {
            return Err(SpirvStreamError::TruncatedInstruction {
                word_offset: offset,
                opcode,
                declared_words: word_count,
                available_words: words.len() - offset,
            });
        };

        if end > words.len() {
            return Err(SpirvStreamError::TruncatedInstruction {
                word_offset: offset,
                opcode,
                declared_words: word_count,
                available_words: words.len() - offset,
            });
        }

        let inst = &words[offset..end];
        let mut keep = true;

        match opcode {
            OP_EXTENSION => {
                if let Some(ext_name) = decode_spirv_string(&inst[1..]) {
                    keep = handle_extension(&ext_name, options, &mut audit);
                }
            }
            OP_CAPABILITY => {
                if let Some(&cap) = inst.get(1) {
                    keep = handle_capability(cap, options, &mut audit);
                }
            }
            OP_EXT_INST_IMPORT => {
                if inst.len() >= 3
                    && let Some(set_name) = decode_spirv_string(&inst[2..])
                {
                    let set_id = inst[1];
                    if set_name.starts_with("NonSemantic.") {
                        if options.strip_non_semantic {
                            stripped_non_semantic_sets.insert(set_id);
                            audit.rewritten_instructions += 1;
                            audit.push_unique(CompatIssue {
                                severity: CompatSeverity::Rewritten,
                                kind: CompatIssueKind::NonSemanticStripped,
                                message: format!(
                                    "stripped OpExtInstImport \"{set_name}\" (%{set_id})"
                                ),
                                hint: None,
                            });
                            keep = false;
                        } else {
                            audit.push_unique(CompatIssue {
                                severity: CompatSeverity::Warning,
                                kind: CompatIssueKind::NonSemanticExtensionPresent,
                                message: format!(
                                    "unstripped non-semantic instruction set \"{set_name}\" (%{set_id})"
                                ),
                                hint: Some(
                                    "enable sanitization (strip_non_semantic) to strip NonSemantic.* debug instructions"
                                        .to_string(),
                                ),
                            });
                        }
                    }
                }
            }
            OP_EXT_INST => {
                // OpExtInst: [header, result_type, result_id, set_id, instruction, operands...]
                if inst.len() >= 5 && stripped_non_semantic_sets.contains(&inst[3]) {
                    stripped_non_semantic_calls += 1;
                    audit.rewritten_instructions += 1;
                    keep = false;
                }
            }
            OP_ENTRY_POINT => {
                if let Some(&exec_model) = inst.get(1) {
                    audit_execution_model(exec_model, &mut audit);
                }
            }
            OP_EXECUTION_MODE | OP_EXECUTION_MODE_ID => {
                if let Some(&mode) = inst.get(2) {
                    keep = handle_execution_mode(opcode, mode, inst, options, &mut audit);
                }
            }
            OP_TYPE_FLOAT => {
                if let Some(&width) = inst.get(2)
                    && width == 64
                {
                    audit.push_once_per_kind(CompatIssue {
                        severity: CompatSeverity::Blocker,
                        kind: CompatIssueKind::Float64,
                        message: "SPIR-V declares 64-bit float type (OpTypeFloat 64)".to_string(),
                        hint: Some(
                            "WebGPU baseline does not support f64; lower double precision to f32 or emulated double-float"
                                .to_string(),
                        ),
                    });
                }
            }
            OP_TYPE_INT => {
                if let Some(&width) = inst.get(2)
                    && width == 64
                {
                    audit.push_once_per_kind(CompatIssue {
                        severity: CompatSeverity::Blocker,
                        kind: CompatIssueKind::Int64,
                        message: "SPIR-V declares 64-bit integer type (OpTypeInt 64)".to_string(),
                        hint: Some(
                            "WebGPU baseline WGSL lacks i64/u64; split 64-bit integers into vec2<u32> pairs"
                                .to_string(),
                        ),
                    });
                }
            }
            OP_TYPE_POINTER | OP_VARIABLE => {
                // OpTypePointer: [header, result_id, storage_class, type_id]
                // OpVariable:    [header, result_type, result_id, storage_class, ...]
                let storage_class = if opcode == OP_TYPE_POINTER {
                    inst.get(2).copied()
                } else {
                    inst.get(3).copied()
                };
                if let Some(sc) = storage_class {
                    audit_storage_class(sc, &mut audit);
                }
            }
            OP_DECORATE => {
                // OpDecorate: [header, target_id, decoration, operands...]
                if let Some(&decoration) = inst.get(2) {
                    match decoration {
                        DECORATION_NON_UNIFORM => {
                            audit.push_once_per_kind(CompatIssue {
                                severity: CompatSeverity::Blocker,
                                kind: CompatIssueKind::BindlessDescriptorIndexing,
                                message: "SPIR-V uses NonUniform decoration (non-uniform descriptor indexing)".to_string(),
                                hint: Some(
                                    "WebGPU baseline lacks non-uniform descriptor indexing; lower to bounded bind groups"
                                        .to_string(),
                                ),
                            });
                        }
                        DECORATION_DESCRIPTOR_SET => {
                            if let Some(&set) = inst.get(3)
                                && set >= WEBGPU_MAX_BIND_GROUPS
                            {
                                audit.push_once_per_kind(CompatIssue {
                                    severity: CompatSeverity::Blocker,
                                    kind: CompatIssueKind::BindGroupLimitExceeded,
                                    message: format!(
                                        "SPIR-V uses DescriptorSet {set}, exceeding WebGPU baseline maxBindGroups ({WEBGPU_MAX_BIND_GROUPS}; valid groups 0..{WEBGPU_MAX_BIND_GROUPS})"
                                    ),
                                    hint: Some(
                                        "pack descriptor sets into <= 4 bind groups for WebGPU baseline compatibility"
                                            .to_string(),
                                    ),
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        if keep {
            out.extend_from_slice(inst);
        }

        offset = end;
    }

    if stripped_non_semantic_calls > 0 {
        audit.push_unique(CompatIssue {
            severity: CompatSeverity::Rewritten,
            kind: CompatIssueKind::NonSemanticStripped,
            message: format!(
                "stripped {stripped_non_semantic_calls} NonSemantic OpExtInst debug call(s)"
            ),
            hint: None,
        });
    }

    audit.sanitized_word_count = out.len();

    Ok(SanitizedSpirv { words: out, audit })
}

fn decode_spirv_string(operand_words: &[u32]) -> Option<String> {
    if operand_words.is_empty() {
        return None;
    }
    let mut raw = Vec::with_capacity(operand_words.len() * 4);
    let mut found_nul = false;
    'outer: for &word in operand_words {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                found_nul = true;
                break 'outer;
            }
            raw.push(byte);
        }
    }
    if !found_nul {
        return None;
    }
    String::from_utf8(raw).ok()
}

fn handle_extension(ext_name: &str, options: &SanitizeOptions, audit: &mut CompatAudit) -> bool {
    match ext_name {
        "SPV_KHR_float_controls" | "SPV_KHR_float_controls2" => {
            if options.strip_float_controls {
                audit.rewritten_instructions += 1;
                audit.push_unique(CompatIssue {
                    severity: CompatSeverity::Rewritten,
                    kind: CompatIssueKind::FloatControlsStripped,
                    message: format!("stripped OpExtension \"{ext_name}\""),
                    hint: None,
                });
                return false;
            }
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::FloatControlsUnsupported,
                message: format!("SPIR-V uses unsupported extension \"{ext_name}\""),
                hint: Some(
                    "enable strip_float_controls (or pass --no-float-controls upstream) for WebGPU lowering"
                        .to_string(),
                ),
            });
            true
        }
        "SPV_KHR_non_semantic_info" => {
            if options.strip_non_semantic {
                audit.rewritten_instructions += 1;
                audit.push_unique(CompatIssue {
                    severity: CompatSeverity::Rewritten,
                    kind: CompatIssueKind::NonSemanticStripped,
                    message: format!("stripped OpExtension \"{ext_name}\""),
                    hint: None,
                });
                return false;
            }
            audit.push_unique(CompatIssue {
                severity: CompatSeverity::Warning,
                kind: CompatIssueKind::NonSemanticExtensionPresent,
                message: format!("unstripped OpExtension \"{ext_name}\""),
                hint: Some(
                    "enable strip_non_semantic to remove non-semantic debug extensions".to_string(),
                ),
            });
            true
        }
        "SPV_EXT_mesh_shader" | "SPV_NV_mesh_shader" => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::MeshShader,
                message: format!("SPIR-V uses mesh shader extension \"{ext_name}\""),
                hint: Some(
                    "WebGPU baseline lacks mesh/task stages; lower upstream with --mesh-compute"
                        .to_string(),
                ),
            });
            true
        }
        "SPV_KHR_ray_tracing"
        | "SPV_KHR_ray_query"
        | "SPV_NV_ray_tracing"
        | "SPV_KHR_ray_tracing_position_fetch"
        | "SPV_KHR_ray_tracing_maintenance1" => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::RayTracing,
                message: format!("SPIR-V uses ray tracing extension \"{ext_name}\""),
                hint: Some(
                    "WebGPU baseline does not support hardware ray tracing / ray queries"
                        .to_string(),
                ),
            });
            true
        }
        "SPV_EXT_descriptor_indexing" | "SPV_EXT_descriptor_heap" => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::BindlessDescriptorIndexing,
                message: format!("SPIR-V uses bindless descriptor extension \"{ext_name}\""),
                hint: Some(
                    "WebGPU baseline lacks unbounded bindless arrays; lower to texture arrays or bounded bind groups"
                        .to_string(),
                ),
            });
            true
        }
        "SPV_KHR_shader_ballot"
        | "SPV_KHR_shader_vote"
        | "SPV_KHR_subgroup_uniform_control_flow"
        | "SPV_KHR_subgroup_rotate"
        | "SPV_KHR_shader_subgroup_extended_types" => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::SubgroupWave64,
                message: format!("SPIR-V uses subgroup extension \"{ext_name}\""),
                hint: Some(
                    "WebGPU baseline lacks guaranteed Wave64 subgroups; compile upstream with --wave64-emulate"
                        .to_string(),
                ),
            });
            true
        }
        "SPV_KHR_physical_storage_buffer" => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::PhysicalStorageBuffer,
                message: format!("SPIR-V uses physical storage buffer extension \"{ext_name}\""),
                hint: Some(
                    "WebGPU lacks raw GPU pointers; lower buffer device addresses to storage buffer offsets"
                        .to_string(),
                ),
            });
            true
        }
        _ => true,
    }
}

fn handle_capability(cap: u32, options: &SanitizeOptions, audit: &mut CompatAudit) -> bool {
    match cap {
        CAP_DENORM_PRESERVE
        | CAP_DENORM_FLUSH_TO_ZERO
        | CAP_SIGNED_ZERO_INF_NAN_PRESERVE
        | CAP_ROUNDING_MODE_RTE
        | CAP_ROUNDING_MODE_RTZ => {
            let name = float_control_capability_name(cap);
            if options.strip_float_controls {
                audit.rewritten_instructions += 1;
                audit.push_unique(CompatIssue {
                    severity: CompatSeverity::Rewritten,
                    kind: CompatIssueKind::FloatControlsStripped,
                    message: format!("stripped OpCapability {name}"),
                    hint: None,
                });
                return false;
            }
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::FloatControlsUnsupported,
                message: format!("SPIR-V declares float-control capability {name}"),
                hint: Some(
                    "enable strip_float_controls (or pass --no-float-controls upstream) for WebGPU lowering"
                        .to_string(),
                ),
            });
            true
        }
        CAP_MESH_SHADING_EXT | CAP_MESH_SHADING_NV => {
            let name = if cap == CAP_MESH_SHADING_EXT {
                "MeshShadingEXT"
            } else {
                "MeshShadingNV"
            };
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::MeshShader,
                message: format!("SPIR-V declares unsupported capability {name}"),
                hint: Some(
                    "WebGPU baseline lacks mesh/task stages; lower upstream with --mesh-compute"
                        .to_string(),
                ),
            });
            true
        }
        CAP_RAY_QUERY_PROVISIONAL_KHR
        | CAP_RAY_QUERY_KHR
        | CAP_RAY_TRACING_KHR
        | CAP_RAY_TRACING_POSITION_FETCH_KHR
        | CAP_RAY_TRACING_NV
        | CAP_RAY_TRACING_MOTION_BLUR_NV
        | CAP_RAY_TRACING_PROVISIONAL_KHR
        | CAP_RAY_TRACING_OPACITY_MICROMAP_EXT
        | CAP_RAY_QUERY_POSITION_FETCH_KHR => {
            let name = ray_tracing_capability_name(cap);
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::RayTracing,
                message: format!("SPIR-V declares unsupported ray tracing capability {name}"),
                hint: Some(
                    "WebGPU baseline does not support hardware ray tracing / ray queries"
                        .to_string(),
                ),
            });
            true
        }
        CAP_FLOAT64 => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::Float64,
                message: "SPIR-V declares unsupported capability Float64".to_string(),
                hint: Some(
                    "WebGPU baseline does not support f64; lower double precision to f32 or emulated double-float"
                        .to_string(),
                ),
            });
            true
        }
        CAP_INT64 => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::Int64,
                message: "SPIR-V declares unsupported capability Int64".to_string(),
                hint: Some(
                    "WebGPU baseline WGSL lacks i64/u64; split 64-bit integers into vec2<u32> pairs"
                        .to_string(),
                ),
            });
            true
        }
        CAP_SUBGROUP_DISPATCH
        | CAP_GROUP_NON_UNIFORM
        | CAP_GROUP_NON_UNIFORM_VOTE
        | CAP_GROUP_NON_UNIFORM_ARITHMETIC
        | CAP_GROUP_NON_UNIFORM_BALLOT
        | CAP_GROUP_NON_UNIFORM_SHUFFLE
        | CAP_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE
        | CAP_GROUP_NON_UNIFORM_CLUSTERED
        | CAP_GROUP_NON_UNIFORM_QUAD
        | CAP_SUBGROUP_BALLOT_KHR
        | CAP_SUBGROUP_VOTE_KHR
        | CAP_GROUP_NON_UNIFORM_PARTITIONED_EXT
        | CAP_GROUP_NON_UNIFORM_ROTATE_KHR => {
            let name = subgroup_capability_name(cap);
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::SubgroupWave64,
                message: format!("SPIR-V declares subgroup capability {name}"),
                hint: Some(
                    "WebGPU baseline lacks guaranteed Wave64 subgroups; compile upstream with --wave64-emulate"
                        .to_string(),
                ),
            });
            true
        }
        CAP_DESCRIPTOR_HEAP_EXT
        | CAP_SHADER_NON_UNIFORM
        | CAP_RUNTIME_DESCRIPTOR_ARRAY
        | CAP_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING
        | CAP_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING
        | CAP_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING
        | CAP_UNIFORM_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | CAP_STORAGE_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING
        | CAP_BINDLESS_TEXTURE_NV
        | CAP_BINDLESS_IMAGES_INTEL => {
            let name = bindless_capability_name(cap);
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::BindlessDescriptorIndexing,
                message: format!("SPIR-V declares bindless/non-uniform capability {name}"),
                hint: Some(
                    "WebGPU baseline lacks bindless descriptor indexing; lower to texture arrays or bounded bind groups"
                        .to_string(),
                ),
            });
            true
        }
        CAP_GEOMETRY => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::GeometryShader,
                message: "SPIR-V declares unsupported capability Geometry".to_string(),
                hint: Some(
                    "WebGPU does not support geometry shaders; lower geometry amplification to compute"
                        .to_string(),
                ),
            });
            true
        }
        CAP_TESSELLATION => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::Tessellation,
                message: "SPIR-V declares unsupported capability Tessellation".to_string(),
                hint: Some(
                    "WebGPU does not support hull/domain tessellation stages; lower to compute pre-pass"
                        .to_string(),
                ),
            });
            true
        }
        CAP_PHYSICAL_STORAGE_BUFFER_ADDRESSES => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::PhysicalStorageBuffer,
                message: "SPIR-V declares unsupported capability PhysicalStorageBufferAddresses"
                    .to_string(),
                hint: Some(
                    "WebGPU lacks raw GPU pointers; lower buffer device addresses to storage buffer offsets"
                        .to_string(),
                ),
            });
            true
        }
        _ => true,
    }
}

fn handle_execution_mode(
    opcode: u16,
    mode: u32,
    inst: &[u32],
    options: &SanitizeOptions,
    audit: &mut CompatAudit,
) -> bool {
    match mode {
        EXEC_MODE_DENORM_PRESERVE
        | EXEC_MODE_DENORM_FLUSH_TO_ZERO
        | EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE
        | EXEC_MODE_ROUNDING_MODE_RTE
        | EXEC_MODE_ROUNDING_MODE_RTZ => {
            let name = float_control_exec_mode_name(mode);
            if options.strip_float_controls {
                audit.rewritten_instructions += 1;
                audit.push_unique(CompatIssue {
                    severity: CompatSeverity::Rewritten,
                    kind: CompatIssueKind::FloatControlsStripped,
                    message: format!("stripped OpExecutionMode {name}"),
                    hint: None,
                });
                return false;
            }
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::FloatControlsUnsupported,
                message: format!("SPIR-V uses unsupported float-control execution mode {name}"),
                hint: Some(
                    "enable strip_float_controls (or pass --no-float-controls upstream) for WebGPU lowering"
                        .to_string(),
                ),
            });
            true
        }
        EXEC_MODE_LOCAL_SIZE if opcode == OP_EXECUTION_MODE => {
            if inst.len() >= 6 {
                let x = inst[3];
                let y = inst[4];
                let z = inst[5];
                check_workgroup_size([x, y, z], None, audit);
            }
            true
        }
        EXEC_MODE_SUBGROUP_SIZE
        | EXEC_MODE_SUBGROUPS_PER_WORKGROUP
        | EXEC_MODE_SUBGROUPS_PER_WORKGROUP_ID
        | EXEC_MODE_SUBGROUP_UNIFORM_CONTROL_FLOW_KHR => {
            let detail = if mode == EXEC_MODE_SUBGROUP_SIZE {
                if let Some(&size) = inst.get(3) {
                    format!("SPIR-V requests OpExecutionMode SubgroupSize({size})")
                } else {
                    "SPIR-V requests OpExecutionMode SubgroupSize".to_string()
                }
            } else {
                format!("SPIR-V uses subgroup execution mode ({mode})")
            };
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::SubgroupWave64,
                message: detail,
                hint: Some(
                    "WebGPU baseline lacks fixed Wave64 subgroup sizing; compile upstream with --wave64-emulate"
                        .to_string(),
                ),
            });
            true
        }
        _ => true,
    }
}

fn audit_execution_model(exec_model: u32, audit: &mut CompatAudit) {
    match exec_model {
        EXEC_MODEL_TESSELLATION_CONTROL | EXEC_MODEL_TESSELLATION_EVALUATION => {
            let name = if exec_model == EXEC_MODEL_TESSELLATION_CONTROL {
                "TessellationControl"
            } else {
                "TessellationEvaluation"
            };
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::Tessellation,
                message: format!("SPIR-V entry point uses unsupported execution model {name}"),
                hint: Some(
                    "WebGPU does not support tessellation stages; lower to compute pre-pass"
                        .to_string(),
                ),
            });
        }
        EXEC_MODEL_GEOMETRY => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::GeometryShader,
                message: "SPIR-V entry point uses unsupported execution model Geometry".to_string(),
                hint: Some(
                    "WebGPU does not support geometry shaders; lower geometry amplification to compute"
                        .to_string(),
                ),
            });
        }
        EXEC_MODEL_TASK_NV | EXEC_MODEL_MESH_NV | EXEC_MODEL_TASK_EXT | EXEC_MODEL_MESH_EXT => {
            let name = match exec_model {
                EXEC_MODEL_TASK_NV => "TaskNV",
                EXEC_MODEL_MESH_NV => "MeshNV",
                EXEC_MODEL_TASK_EXT => "TaskEXT",
                _ => "MeshEXT",
            };
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::MeshShader,
                message: format!("SPIR-V entry point uses unsupported execution model {name}"),
                hint: Some(
                    "WebGPU baseline lacks mesh/task stages; lower upstream with --mesh-compute"
                        .to_string(),
                ),
            });
        }
        EXEC_MODEL_RAY_GENERATION_KHR
        | EXEC_MODEL_INTERSECTION_KHR
        | EXEC_MODEL_ANY_HIT_KHR
        | EXEC_MODEL_CLOSEST_HIT_KHR
        | EXEC_MODEL_MISS_KHR
        | EXEC_MODEL_CALLABLE_KHR => {
            let name = match exec_model {
                EXEC_MODEL_RAY_GENERATION_KHR => "RayGenerationKHR",
                EXEC_MODEL_INTERSECTION_KHR => "IntersectionKHR",
                EXEC_MODEL_ANY_HIT_KHR => "AnyHitKHR",
                EXEC_MODEL_CLOSEST_HIT_KHR => "ClosestHitKHR",
                EXEC_MODEL_MISS_KHR => "MissKHR",
                _ => "CallableKHR",
            };
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::RayTracing,
                message: format!("SPIR-V entry point uses unsupported execution model {name}"),
                hint: Some(
                    "WebGPU baseline does not support hardware ray tracing stages".to_string(),
                ),
            });
        }
        _ => {}
    }
}

fn audit_storage_class(storage_class: u32, audit: &mut CompatAudit) {
    match storage_class {
        STORAGE_CLASS_PUSH_CONSTANT => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::PushConstants,
                message: "SPIR-V uses PushConstant storage class".to_string(),
                hint: Some(
                    "WebGPU baseline lacks push constants; lower root constants to a uniform buffer binding"
                        .to_string(),
                ),
            });
        }
        STORAGE_CLASS_PHYSICAL_STORAGE_BUFFER => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::PhysicalStorageBuffer,
                message: "SPIR-V uses PhysicalStorageBuffer storage class".to_string(),
                hint: Some(
                    "WebGPU lacks raw GPU pointers; lower buffer device addresses to storage buffer offsets"
                        .to_string(),
                ),
            });
        }
        _ => {}
    }
}

fn check_workgroup_size(size: [u32; 3], entry_point: Option<&str>, audit: &mut CompatAudit) {
    let [x, y, z] = size;
    let total = u64::from(x)
        .saturating_mul(u64::from(y))
        .saturating_mul(u64::from(z));

    let exceeds_dim = x > WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_X
        || y > WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Y
        || z > WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Z;
    let exceeds_total = total > u64::from(WEBGPU_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP);
    let has_zero = x == 0 || y == 0 || z == 0;

    if exceeds_dim || exceeds_total || has_zero {
        let prefix = match entry_point {
            Some(name) => format!("compute entry point '{name}' "),
            None => String::new(),
        };
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
            message: format!(
                "{prefix}workgroup_size({x}, {y}, {z}) = {total} invocations exceeds WebGPU baseline default limits ({WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_X}x{WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Y}x{WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Z}, max {WEBGPU_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP} invocations)"
            ),
            hint: Some(
                "request higher WebGPU compute workgroup limits or retile dispatch threads to <= 256 invocations"
                    .to_string(),
            ),
        });
    }
}

fn float_control_capability_name(cap: u32) -> &'static str {
    match cap {
        CAP_DENORM_PRESERVE => "DenormPreserve",
        CAP_DENORM_FLUSH_TO_ZERO => "DenormFlushToZero",
        CAP_SIGNED_ZERO_INF_NAN_PRESERVE => "SignedZeroInfNanPreserve",
        CAP_ROUNDING_MODE_RTE => "RoundingModeRTE",
        CAP_ROUNDING_MODE_RTZ => "RoundingModeRTZ",
        _ => "FloatControls",
    }
}

fn float_control_exec_mode_name(mode: u32) -> &'static str {
    match mode {
        EXEC_MODE_DENORM_PRESERVE => "DenormPreserve",
        EXEC_MODE_DENORM_FLUSH_TO_ZERO => "DenormFlushToZero",
        EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE => "SignedZeroInfNanPreserve",
        EXEC_MODE_ROUNDING_MODE_RTE => "RoundingModeRTE",
        EXEC_MODE_ROUNDING_MODE_RTZ => "RoundingModeRTZ",
        _ => "FloatControls",
    }
}

fn ray_tracing_capability_name(cap: u32) -> &'static str {
    match cap {
        CAP_RAY_QUERY_PROVISIONAL_KHR => "RayQueryProvisionalKHR",
        CAP_RAY_QUERY_KHR => "RayQueryKHR",
        CAP_RAY_TRACING_KHR => "RayTracingKHR",
        CAP_RAY_TRACING_POSITION_FETCH_KHR => "RayTracingPositionFetchKHR",
        CAP_RAY_TRACING_NV => "RayTracingNV",
        CAP_RAY_TRACING_MOTION_BLUR_NV => "RayTracingMotionBlurNV",
        CAP_RAY_TRACING_PROVISIONAL_KHR => "RayTracingProvisionalKHR",
        CAP_RAY_TRACING_OPACITY_MICROMAP_EXT => "RayTracingOpacityMicromapEXT",
        CAP_RAY_QUERY_POSITION_FETCH_KHR => "RayQueryPositionFetchKHR",
        _ => "RayTracing",
    }
}

fn subgroup_capability_name(cap: u32) -> &'static str {
    match cap {
        CAP_SUBGROUP_DISPATCH => "SubgroupDispatch",
        CAP_GROUP_NON_UNIFORM => "GroupNonUniform",
        CAP_GROUP_NON_UNIFORM_VOTE => "GroupNonUniformVote",
        CAP_GROUP_NON_UNIFORM_ARITHMETIC => "GroupNonUniformArithmetic",
        CAP_GROUP_NON_UNIFORM_BALLOT => "GroupNonUniformBallot",
        CAP_GROUP_NON_UNIFORM_SHUFFLE => "GroupNonUniformShuffle",
        CAP_GROUP_NON_UNIFORM_SHUFFLE_RELATIVE => "GroupNonUniformShuffleRelative",
        CAP_GROUP_NON_UNIFORM_CLUSTERED => "GroupNonUniformClustered",
        CAP_GROUP_NON_UNIFORM_QUAD => "GroupNonUniformQuad",
        CAP_SUBGROUP_BALLOT_KHR => "SubgroupBallotKHR",
        CAP_SUBGROUP_VOTE_KHR => "SubgroupVoteKHR",
        CAP_GROUP_NON_UNIFORM_PARTITIONED_EXT => "GroupNonUniformPartitionedEXT",
        CAP_GROUP_NON_UNIFORM_ROTATE_KHR => "GroupNonUniformRotateKHR",
        _ => "Subgroup",
    }
}

fn bindless_capability_name(cap: u32) -> &'static str {
    match cap {
        CAP_DESCRIPTOR_HEAP_EXT => "DescriptorHeapEXT",
        CAP_SHADER_NON_UNIFORM => "ShaderNonUniform",
        CAP_RUNTIME_DESCRIPTOR_ARRAY => "RuntimeDescriptorArray",
        CAP_UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING => "UniformBufferArrayNonUniformIndexing",
        CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING => "SampledImageArrayNonUniformIndexing",
        CAP_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING => "StorageBufferArrayNonUniformIndexing",
        CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING => "StorageImageArrayNonUniformIndexing",
        CAP_INPUT_ATTACHMENT_ARRAY_NON_UNIFORM_INDEXING => "InputAttachmentArrayNonUniformIndexing",
        CAP_UNIFORM_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING => {
            "UniformTexelBufferArrayNonUniformIndexing"
        }
        CAP_STORAGE_TEXEL_BUFFER_ARRAY_NON_UNIFORM_INDEXING => {
            "StorageTexelBufferArrayNonUniformIndexing"
        }
        CAP_BINDLESS_TEXTURE_NV => "BindlessTextureNV",
        CAP_BINDLESS_IMAGES_INTEL => "BindlessImagesINTEL",
        _ => "DescriptorIndexing",
    }
}

// ---------------------------------------------------------------------------
// Naga IR WebGPU Baseline Limits Audit
// ---------------------------------------------------------------------------

/// Audit a parsed [`naga::Module`] against WebGPU baseline default limits and feature constraints.
///
/// Checks:
/// - Compute workgroup sizes (`256 x 256 x 64`, `<= 256` total invocations) and override sizes
/// - Unsupported shader stages (`Task`, `Mesh`, ray tracing stages)
/// - Bind group index ceiling (`maxBindGroups = 4`, indices `0..4`)
/// - Push constants / immediate data (`AddressSpace::Immediate`)
/// - Storage buffer and uniform buffer counts per shader stage (`8` / `12` WebGPU defaults)
/// - Unsupported IR types (`f64`, `i64`/`u64`, `BindingArray`, `AccelerationStructure`, `RayQuery`)
pub fn audit_naga_module(module: &naga::Module) -> CompatAudit {
    let mut audit = CompatAudit::default();

    // 1. Entry points: stages and compute workgroup limits.
    for ep in &module.entry_points {
        match ep.stage {
            naga::ShaderStage::Compute => {
                if ep.workgroup_size_overrides.is_some() {
                    audit.push_unique(CompatIssue {
                        severity: CompatSeverity::Warning,
                        kind: CompatIssueKind::OverrideWorkgroupSize,
                        message: format!(
                            "compute entry point '{}' uses pipeline-overridable workgroup_size; verify runtime values stay within WebGPU default limits",
                            ep.name
                        ),
                        hint: None,
                    });
                } else {
                    check_workgroup_size(ep.workgroup_size, Some(&ep.name), &mut audit);
                }
            }
            naga::ShaderStage::Task | naga::ShaderStage::Mesh => {
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::MeshShader,
                    message: format!(
                        "entry point '{}' uses non-WebGPU stage {:?}",
                        ep.name, ep.stage
                    ),
                    hint: Some(
                        "WebGPU baseline lacks mesh/task stages; lower upstream with --mesh-compute"
                            .to_string(),
                    ),
                });
            }
            naga::ShaderStage::RayGeneration
            | naga::ShaderStage::Miss
            | naga::ShaderStage::AnyHit
            | naga::ShaderStage::ClosestHit => {
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::RayTracing,
                    message: format!(
                        "entry point '{}' uses non-WebGPU ray tracing stage {:?}",
                        ep.name, ep.stage
                    ),
                    hint: Some(
                        "WebGPU baseline does not support hardware ray tracing stages".to_string(),
                    ),
                });
            }
            naga::ShaderStage::Vertex | naga::ShaderStage::Fragment => {}
        }

        if ep.task_payload.is_some() {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::MeshShader,
                message: format!("entry point '{}' uses a mesh/task payload", ep.name),
                hint: Some("lower mesh/task shaders upstream with --mesh-compute".to_string()),
            });
        }
        if ep.incoming_ray_payload.is_some() {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::RayTracing,
                message: format!("entry point '{}' uses an incoming ray payload", ep.name),
                hint: Some(
                    "WebGPU baseline does not support hardware ray tracing pipelines".to_string(),
                ),
            });
        }
    }

    // 2. Global variables: address spaces, bind groups, push constants / immediates.
    let mut module_storage_buffers = 0usize;
    let mut module_uniform_buffers = 0usize;

    for (_, var) in module.global_variables.iter() {
        match var.space {
            naga::AddressSpace::Immediate => {
                let label = var.name.as_deref().unwrap_or("<unnamed>");
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::PushConstants,
                    message: format!(
                        "global variable '{label}' uses push constants / immediate data (AddressSpace::Immediate)"
                    ),
                    hint: Some(
                        "WebGPU baseline lacks push constants; lower root constants to a uniform buffer binding"
                            .to_string(),
                    ),
                });
            }
            naga::AddressSpace::TaskPayload => {
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::MeshShader,
                    message: "module declares a TaskPayload global variable".to_string(),
                    hint: Some("lower mesh/task shaders upstream with --mesh-compute".to_string()),
                });
            }
            naga::AddressSpace::RayPayload | naga::AddressSpace::IncomingRayPayload => {
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::RayTracing,
                    message: format!("module declares a {:?} global variable", var.space),
                    hint: Some(
                        "WebGPU baseline does not support hardware ray tracing pipelines"
                            .to_string(),
                    ),
                });
            }
            naga::AddressSpace::Storage { .. } => {
                module_storage_buffers += 1;
            }
            naga::AddressSpace::Uniform => {
                module_uniform_buffers += 1;
            }
            _ => {}
        }

        if let Some(binding) = &var.binding
            && binding.group >= WEBGPU_MAX_BIND_GROUPS
        {
            let label = var.name.as_deref().unwrap_or("<unnamed>");
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::BindGroupLimitExceeded,
                message: format!(
                    "resource '{label}' uses @group({}) @binding({}), exceeding WebGPU baseline maxBindGroups ({WEBGPU_MAX_BIND_GROUPS}; valid groups 0..{WEBGPU_MAX_BIND_GROUPS})",
                    binding.group, binding.binding
                ),
                hint: Some(
                    "pack descriptor sets into <= 4 bind groups for WebGPU baseline compatibility"
                        .to_string(),
                ),
            });
        }
    }

    // 3. Per-stage binding counts (falls back to module-wide globals when an entry point
    //    doesn't directly reference unused globals or when there are 0 entry points).
    if module.entry_points.is_empty() {
        check_stage_buffer_limits(
            None,
            module_storage_buffers,
            module_uniform_buffers,
            &mut audit,
        );
    } else {
        for ep in &module.entry_points {
            let referenced = collect_referenced_globals(module, &ep.function);
            let mut stage_storage = 0usize;
            let mut stage_uniform = 0usize;
            for handle in referenced {
                match module.global_variables[handle].space {
                    naga::AddressSpace::Storage { .. } => stage_storage += 1,
                    naga::AddressSpace::Uniform => stage_uniform += 1,
                    _ => {}
                }
            }
            // If a synthetic module declares global resource bindings without loading them in
            // the stub body, also honor the module-wide count so limit audits still catch them.
            let effective_storage = if module.entry_points.len() == 1 {
                stage_storage.max(module_storage_buffers)
            } else {
                stage_storage
            };
            let effective_uniform = if module.entry_points.len() == 1 {
                stage_uniform.max(module_uniform_buffers)
            } else {
                stage_uniform
            };
            check_stage_buffer_limits(
                Some(&ep.name),
                effective_storage,
                effective_uniform,
                &mut audit,
            );
        }
    }

    // 4. Module types: 64-bit scalars, binding arrays, ray-tracing types.
    for (_, ty) in module.types.iter() {
        match &ty.inner {
            naga::TypeInner::Scalar(scalar)
            | naga::TypeInner::Vector { scalar, .. }
            | naga::TypeInner::Matrix { scalar, .. }
            | naga::TypeInner::Atomic(scalar) => {
                audit_naga_scalar(*scalar, &mut audit);
            }
            naga::TypeInner::BindingArray { .. } => {
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::BindlessDescriptorIndexing,
                    message: "Naga module uses a BindingArray (bindless descriptor array)"
                        .to_string(),
                    hint: Some(
                        "WebGPU baseline lacks binding_array; lower to texture arrays or bounded bindings"
                            .to_string(),
                    ),
                });
            }
            naga::TypeInner::AccelerationStructure { .. } | naga::TypeInner::RayQuery { .. } => {
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::RayTracing,
                    message: "Naga module uses ray tracing types (AccelerationStructure / RayQuery)"
                        .to_string(),
                    hint: Some(
                        "WebGPU baseline does not support hardware ray tracing / ray queries"
                            .to_string(),
                    ),
                });
            }
            _ => {}
        }
    }

    audit
}

fn check_stage_buffer_limits(
    entry_point: Option<&str>,
    storage_buffers: usize,
    uniform_buffers: usize,
    audit: &mut CompatAudit,
) {
    let scope = match entry_point {
        Some(name) => format!("entry point '{name}'"),
        None => "module".to_string(),
    };

    if storage_buffers > WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::StorageBufferLimitExceeded,
            message: format!(
                "{scope} uses {storage_buffers} storage buffer bindings, exceeding WebGPU baseline maxStorageBuffersPerShaderStage ({WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE})"
            ),
            hint: Some(
                "request higher maxStorageBuffersPerShaderStage or merge read-only buffers into a combined storage buffer"
                    .to_string(),
            ),
        });
    }

    if uniform_buffers > WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::UniformBufferLimitExceeded,
            message: format!(
                "{scope} uses {uniform_buffers} uniform buffer bindings, exceeding WebGPU baseline maxUniformBuffersPerShaderStage ({WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE})"
            ),
            hint: Some(
                "pack uniform buffers into fewer bindings or a shared constant buffer".to_string(),
            ),
        });
    }
}

fn audit_naga_scalar(scalar: naga::Scalar, audit: &mut CompatAudit) {
    if scalar.width == 8 {
        match scalar.kind {
            naga::ScalarKind::Float => {
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::Float64,
                    message: "Naga module contains 64-bit float (f64) types".to_string(),
                    hint: Some(
                        "WebGPU baseline does not support f64; lower double precision to f32"
                            .to_string(),
                    ),
                });
            }
            naga::ScalarKind::Sint | naga::ScalarKind::Uint => {
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::Int64,
                    message: "Naga module contains 64-bit integer (i64/u64) types".to_string(),
                    hint: Some(
                        "WebGPU baseline WGSL lacks i64/u64; split 64-bit integers into vec2<u32>"
                            .to_string(),
                    ),
                });
            }
            _ => {}
        }
    }
}

fn collect_referenced_globals(
    module: &naga::Module,
    root: &naga::Function,
) -> HashSet<naga::Handle<naga::GlobalVariable>> {
    let mut globals = HashSet::new();
    let mut visited_fns: HashSet<naga::Handle<naga::Function>> = HashSet::new();
    let mut pending_fns: Vec<naga::Handle<naga::Function>> = Vec::new();

    scan_function_globals(root, &mut globals, &mut pending_fns);
    while let Some(fn_handle) = pending_fns.pop() {
        if visited_fns.insert(fn_handle) {
            scan_function_globals(&module.functions[fn_handle], &mut globals, &mut pending_fns);
        }
    }

    globals
}

fn scan_function_globals(
    func: &naga::Function,
    globals: &mut HashSet<naga::Handle<naga::GlobalVariable>>,
    callees: &mut Vec<naga::Handle<naga::Function>>,
) {
    for (_, expr) in func.expressions.iter() {
        if let naga::Expression::GlobalVariable(handle) = *expr {
            globals.insert(handle);
        }
    }
    collect_calls_in_block(&func.body, callees);
}

fn collect_calls_in_block(block: &naga::Block, callees: &mut Vec<naga::Handle<naga::Function>>) {
    for stmt in block.iter() {
        match stmt {
            naga::Statement::Block(inner) => collect_calls_in_block(inner, callees),
            naga::Statement::If { accept, reject, .. } => {
                collect_calls_in_block(accept, callees);
                collect_calls_in_block(reject, callees);
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    collect_calls_in_block(&case.body, callees);
                }
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                collect_calls_in_block(body, callees);
                collect_calls_in_block(continuing, callees);
            }
            naga::Statement::Call { function, .. } => {
                callees.push(*function);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Batch Compatibility Audit Report Rendering
// ---------------------------------------------------------------------------

/// Per-shader audit record used to build a batch compatibility Markdown report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderAuditRecord {
    pub source_path: String,
    pub stage: Option<String>,
    pub lowered_to_wgsl: bool,
    pub stream_error: Option<String>,
    pub naga_error: Option<String>,
    pub audit: CompatAudit,
}

/// Render a structured Markdown WebGPU compatibility & rewrite audit report.
pub fn render_compat_audit_report_md(records: &[ShaderAuditRecord]) -> String {
    let total = records.len();
    let lowered_ok = records.iter().filter(|r| r.lowered_to_wgsl).count();
    let shaders_rewritten = records
        .iter()
        .filter(|r| r.audit.rewritten_instructions > 0)
        .count();
    let total_rewritten_insts: usize = records.iter().map(|r| r.audit.rewritten_instructions).sum();
    let shaders_with_blockers = records
        .iter()
        .filter(|r| r.audit.has_blockers() || r.stream_error.is_some() || !r.lowered_to_wgsl)
        .count();
    let shaders_with_warnings = records
        .iter()
        .filter(|r| r.audit.warning_count() > 0)
        .count();
    let malformed_streams = records.iter().filter(|r| r.stream_error.is_some()).count();

    let pass_rate = if total == 0 {
        0.0
    } else {
        (lowered_ok as f64 / total as f64) * 100.0
    };

    let mut out = String::new();
    out.push_str("# SPIR-V WebGPU Compatibility Audit Report\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Metric | Count |\n");
    out.push_str("| :--- | ---: |\n");
    let _ = writeln!(out, "| Total `.spv` files audited | {total} |");
    let _ = writeln!(
        out,
        "| Lowered to WebGPU WGSL | {lowered_ok} ({pass_rate:.1}%) |"
    );
    let _ = writeln!(out, "| Shaders sanitized/rewritten | {shaders_rewritten} |");
    let _ = writeln!(
        out,
        "| Total SPIR-V instructions rewritten | {total_rewritten_insts} |"
    );
    let _ = writeln!(
        out,
        "| Shaders with blockers / failures | {shaders_with_blockers} |"
    );
    let _ = writeln!(
        out,
        "| Shaders with limit warnings | {shaders_with_warnings} |"
    );
    let _ = writeln!(out, "| Malformed SPIR-V streams | {malformed_streams} |");
    out.push('\n');

    // Group issues by (severity, kind).
    struct IssueBucket {
        severity: CompatSeverity,
        kind: CompatIssueKind,
        shader_paths: BTreeSet<String>,
        occurrences: usize,
        hint: Option<String>,
    }

    let mut buckets: BTreeMap<(CompatSeverity, CompatIssueKind), IssueBucket> = BTreeMap::new();
    for record in records {
        for issue in &record.audit.issues {
            let bucket = buckets
                .entry((issue.severity, issue.kind))
                .or_insert_with(|| IssueBucket {
                    severity: issue.severity,
                    kind: issue.kind,
                    shader_paths: BTreeSet::new(),
                    occurrences: 0,
                    hint: issue.hint.clone(),
                });
            bucket.occurrences += 1;
            bucket.shader_paths.insert(record.source_path.clone());
            if bucket.hint.is_none() && issue.hint.is_some() {
                bucket.hint = issue.hint.clone();
            }
        }
    }

    let mut ranked: Vec<IssueBucket> = buckets.into_values().collect();
    ranked.sort_by(|a, b| {
        severity_rank(a.severity)
            .cmp(&severity_rank(b.severity))
            .then_with(|| b.shader_paths.len().cmp(&a.shader_paths.len()))
            .then_with(|| a.kind.cmp(&b.kind))
    });

    out.push_str("## Compatibility Findings & Rewrites\n\n");
    if ranked.is_empty() {
        out.push_str("No compatibility issues or rewrites recorded.\n\n");
    } else {
        out.push_str("| Severity | Issue Kind | Shaders | Findings | Guidance | Sample Shaders |\n");
        out.push_str("| :--- | :--- | ---: | ---: | :--- | :--- |\n");
        for bucket in &ranked {
            let hint_cell = bucket
                .hint
                .as_deref()
                .map(escape_md_cell)
                .unwrap_or_else(|| "-".to_string());
            let samples = bucket
                .shader_paths
                .iter()
                .take(5)
                .map(|p| format!("`{}`", escape_md_cell(p)))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                out,
                "| `{}` | `{}` | {} | {} | {} | {} |",
                bucket.severity,
                bucket.kind,
                bucket.shader_paths.len(),
                bucket.occurrences,
                hint_cell,
                samples
            );
        }
        out.push('\n');
    }

    let failing_records: Vec<&ShaderAuditRecord> = records
        .iter()
        .filter(|r| !r.lowered_to_wgsl || r.audit.has_blockers() || r.stream_error.is_some())
        .collect();

    if !failing_records.is_empty() {
        out.push_str("## Shaders Requiring Attention\n\n");
        for rec in failing_records {
            let status = if let Some(err) = &rec.stream_error {
                format!("malformed SPIR-V: {}", first_non_empty_line(err))
            } else if let Some(err) = &rec.naga_error {
                format!("Naga error: {}", first_non_empty_line(err))
            } else {
                "lowered with baseline blockers".to_string()
            };
            let blocker_kinds = rec
                .audit
                .blockers()
                .map(|b| format!("`{}`", b.kind))
                .collect::<Vec<_>>()
                .join(", ");
            if blocker_kinds.is_empty() {
                let _ = writeln!(
                    out,
                    "- `{}` — {}",
                    escape_md_cell(&rec.source_path),
                    escape_md_cell(&status)
                );
            } else {
                let _ = writeln!(
                    out,
                    "- `{}` — {} (blockers: {})",
                    escape_md_cell(&rec.source_path),
                    escape_md_cell(&status),
                    blocker_kinds
                );
            }
        }
    }

    out
}

fn severity_rank(severity: CompatSeverity) -> u8 {
    match severity {
        CompatSeverity::Blocker => 0,
        CompatSeverity::Warning => 1,
        CompatSeverity::Rewritten => 2,
    }
}

fn first_non_empty_line(text: &str) -> &str {
    text.lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or(text)
        .trim()
}

fn escape_md_cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{
        ConvertOptions, audit_spirv_file, convert_spirv_bytes_with_options,
        test_spirv::SpirvBuilder,
    };

    const OP_MEMORY_MODEL: u16 = 14;
    const OP_TYPE_VOID: u16 = 19;
    const OP_TYPE_STRUCT: u16 = 30;
    const OP_TYPE_FUNCTION: u16 = 33;
    const OP_FUNCTION: u16 = 54;
    const OP_FUNCTION_END: u16 = 56;
    const OP_MEMBER_DECORATE: u16 = 72;
    const OP_LABEL: u16 = 248;
    const OP_RETURN: u16 = 253;

    const CAP_SHADER: u32 = 1;
    const ADDRESSING_LOGICAL: u32 = 0;
    const MEMORY_GLSL450: u32 = 1;
    const EXEC_MODEL_GL_COMPUTE: u32 = 5;
    const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;
    const DECORATION_BLOCK: u32 = 2;
    const DECORATION_OFFSET: u32 = 35;
    const DECORATION_BINDING: u32 = 33;

    fn build_float_controls_compute_spv(
        cap: u32,
        exec_mode: u32,
        include_extension: bool,
    ) -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_CAPABILITY, &[cap]);
        if include_extension {
            b.inst_with_string(OP_EXTENSION, &[], "SPV_KHR_float_controls", &[]);
        }
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_GL_COMPUTE, main_fn],
            "main",
            &[],
        );
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);
        // Float-control execution modes take a target bit-width operand (32).
        b.inst(OP_EXECUTION_MODE, &[main_fn, exec_mode, 32]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    fn build_non_semantic_debug_compute_spv(ext_set_name: &str) -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let glsl_set = b.id();
        let ns_set = b.id();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();
        let dbg_call_1 = b.id();
        let dbg_call_2 = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst_with_string(OP_EXTENSION, &[], "SPV_KHR_non_semantic_info", &[]);
        b.inst_with_string(OP_EXT_INST_IMPORT, &[glsl_set], "GLSL.std.450", &[]);
        b.inst_with_string(OP_EXT_INST_IMPORT, &[ns_set], ext_set_name, &[]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_GL_COMPUTE, main_fn],
            "main",
            &[],
        );
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        // Two non-semantic debug OpExtInst calls inside the function body
        b.inst(OP_EXT_INST, &[void_ty, dbg_call_1, ns_set, 1]);
        b.inst(OP_EXT_INST, &[void_ty, dbg_call_2, ns_set, 2]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    fn build_compute_with_workgroup_size(x: u32, y: u32, z: u32) -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_GL_COMPUTE, main_fn],
            "main",
            &[],
        );
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, x, y, z]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    fn build_compute_with_storage_buffers(count: u32, descriptor_set: u32) -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let u32_ty = b.id();
        let struct_ty = b.id();
        let ptr_ssbo = b.id();
        let main_fn = b.id();
        let entry_label = b.id();
        let mut var_ids = Vec::new();
        for _ in 0..count {
            var_ids.push(b.id());
        }

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst_with_string(
            OP_EXTENSION,
            &[],
            "SPV_KHR_storage_buffer_storage_class",
            &[],
        );
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_GL_COMPUTE, main_fn],
            "main",
            &[],
        );
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);

        b.inst(OP_DECORATE, &[struct_ty, DECORATION_BLOCK]);
        b.inst(OP_MEMBER_DECORATE, &[struct_ty, 0, DECORATION_OFFSET, 0]);
        for (idx, &var_id) in var_ids.iter().enumerate() {
            b.inst(OP_DECORATE, &[var_id, DECORATION_DESCRIPTOR_SET, descriptor_set]);
            b.inst(OP_DECORATE, &[var_id, DECORATION_BINDING, idx as u32]);
        }

        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_TYPE_INT, &[u32_ty, 32, 0]);
        b.inst(OP_TYPE_STRUCT, &[struct_ty, u32_ty]);
        b.inst(
            OP_TYPE_POINTER,
            &[ptr_ssbo, STORAGE_CLASS_STORAGE_BUFFER, struct_ty],
        );
        for &var_id in &var_ids {
            b.inst(OP_VARIABLE, &[ptr_ssbo, var_id, STORAGE_CLASS_STORAGE_BUFFER]);
        }

        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    #[test]
    fn float_controls_fail_raw_baseline_and_lower_cleanly_after_sanitization() {
        for (cap, mode) in [
            (
                CAP_SIGNED_ZERO_INF_NAN_PRESERVE,
                EXEC_MODE_SIGNED_ZERO_INF_NAN_PRESERVE,
            ),
            (CAP_DENORM_FLUSH_TO_ZERO, EXEC_MODE_DENORM_FLUSH_TO_ZERO),
            (CAP_DENORM_PRESERVE, EXEC_MODE_DENORM_PRESERVE),
        ] {
            let spv_bytes = build_float_controls_compute_spv(cap, mode, true);

            // 1. Raw WebGPU baseline without sanitization MUST fail in Naga and be flagged by audit.
            let raw_err = convert_spirv_bytes_with_options(
                &spv_bytes,
                &ConvertOptions::without_sanitization(),
            )
            .expect_err("raw SPIR-V with SPV_KHR_float_controls should fail baseline lowering");
            assert_eq!(
                raw_err.kind,
                crate::convert::ConversionErrorKind::SpirvParseError
            );

            let raw_words = spirv_bytes_to_words(&spv_bytes).unwrap();
            let raw_audit = sanitize_spirv_words(&raw_words, &SanitizeOptions::disabled())
                .unwrap()
                .audit;
            assert!(
                raw_audit
                    .issues
                    .iter()
                    .any(|i| i.kind == CompatIssueKind::FloatControlsUnsupported
                        && i.severity == CompatSeverity::Blocker),
                "expected FloatControlsUnsupported blocker when sanitization is disabled: {:?}",
                raw_audit.issues
            );

            // 2. Sanitizer rewrites the 3 float-control instructions cleanly.
            let sanitized = sanitize_spirv_words(&raw_words, &SanitizeOptions::default())
                .expect("sanitization should succeed");
            assert_eq!(sanitized.audit.rewritten_instructions, 3);
            assert!(!sanitized.audit.has_blockers());
            assert!(sanitized.words.len() < raw_words.len());
            assert!(
                sanitized
                    .audit
                    .rewrites()
                    .all(|i| i.kind == CompatIssueKind::FloatControlsStripped)
            );

            // 3. Full Naga lowering succeeds and produces valid WGSL!
            let converted =
                convert_spirv_bytes_with_options(&spv_bytes, &ConvertOptions::default())
                    .expect("sanitized float-controls SPIR-V should lower to valid WGSL");
            assert!(converted.wgsl.contains("@compute"));
            assert!(converted.wgsl.contains("@workgroup_size(1, 1, 1)"));
            assert_eq!(converted.audit.rewritten_instructions, 3);
            assert!(!converted.audit.has_blockers());
        }
    }

    #[test]
    fn non_semantic_debug_extension_and_calls_are_stripped_while_preserving_glsl() {
        // NonSemantic.DebugPrintf is rejected by Naga unless stripped!
        let spv_bytes = build_non_semantic_debug_compute_spv("NonSemantic.DebugPrintf");

        // Raw lowering fails because Naga only knows NonSemantic.Shader.DebugInfo.100.
        let raw_err = convert_spirv_bytes_with_options(
            &spv_bytes,
            &ConvertOptions::without_sanitization(),
        )
        .expect_err("raw NonSemantic.DebugPrintf should fail Naga without sanitization");
        assert!(
            raw_err
                .keywords
                .iter()
                .any(|k| k == "NonSemantic.DebugPrintf"),
            "expected NonSemantic.DebugPrintf in error: {raw_err:?}"
        );

        let words = spirv_bytes_to_words(&spv_bytes).unwrap();
        let sanitized = sanitize_spirv_words(&words, &SanitizeOptions::default()).unwrap();
        // Stripped: OpExtension + OpExtInstImport(NonSemantic) + 2 OpExtInst calls = 4 instructions.
        assert_eq!(sanitized.audit.rewritten_instructions, 4);
        assert!(!sanitized.audit.has_blockers());
        assert!(
            sanitized
                .audit
                .rewrites()
                .any(|i| i.kind == CompatIssueKind::NonSemanticStripped
                    && i.message.contains("2 NonSemantic OpExtInst"))
        );

        // Verify GLSL.std.450 OpExtInstImport was NOT stripped!
        let mut saw_glsl_import = false;
        let mut offset = SPIRV_HEADER_WORDS;
        while offset < sanitized.words.len() {
            let hdr = sanitized.words[offset];
            let wc = (hdr >> 16) as usize;
            let op = (hdr & 0xffff) as u16;
            if op == OP_EXT_INST_IMPORT
                && let Some(name) = decode_spirv_string(&sanitized.words[offset + 2..offset + wc])
                && name == "GLSL.std.450"
            {
                saw_glsl_import = true;
            }
            offset += wc;
        }
        assert!(
            saw_glsl_import,
            "GLSL.std.450 OpExtInstImport must be preserved"
        );

        // Lowering with default sanitization produces valid WGSL!
        let converted = convert_spirv_bytes_with_options(&spv_bytes, &ConvertOptions::default())
            .expect("sanitized NonSemantic shader should lower to WGSL");
        assert!(converted.wgsl.contains("@compute"));
        assert_eq!(converted.audit.rewritten_instructions, 4);

        // Also test NonSemantic.Shader.DebugInfo.100 stripping cleanly.
        let dbg100 = build_non_semantic_debug_compute_spv("NonSemantic.Shader.DebugInfo.100");
        let dbg100_words = spirv_bytes_to_words(&dbg100).unwrap();
        let dbg100_sanitized =
            sanitize_spirv_words(&dbg100_words, &SanitizeOptions::default()).unwrap();
        assert_eq!(dbg100_sanitized.audit.rewritten_instructions, 4);
    }

    #[test]
    fn detects_hard_webgpu_blockers_in_spirv_stream() {
        // 1. RayTracingKHR + MeshShadingEXT + Float64 + Int64 + SubgroupWave64 +
        //    BindlessDescriptorIndexing + PushConstants + Geometry + Tessellation
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let f64_ty = b.id();
        let i64_ty = b.id();
        let pc_ptr = b.id();
        let pc_var = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_CAPABILITY, &[CAP_RAY_TRACING_KHR]);
        b.inst(OP_CAPABILITY, &[CAP_MESH_SHADING_EXT]);
        b.inst(OP_CAPABILITY, &[CAP_FLOAT64]);
        b.inst(OP_CAPABILITY, &[CAP_INT64]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM]);
        b.inst(OP_CAPABILITY, &[CAP_RUNTIME_DESCRIPTOR_ARRAY]);
        b.inst(OP_CAPABILITY, &[CAP_GEOMETRY]);
        b.inst(OP_CAPABILITY, &[CAP_TESSELLATION]);
        b.inst_with_string(OP_EXTENSION, &[], "SPV_KHR_ray_tracing", &[]);
        b.inst_with_string(OP_EXTENSION, &[], "SPV_EXT_mesh_shader", &[]);
        b.inst_with_string(OP_EXTENSION, &[], "SPV_EXT_descriptor_indexing", &[]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_MESH_EXT, main_fn], "ms_main", &[]);
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_SUBGROUP_SIZE, 64]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FLOAT, &[f64_ty, 64]);
        b.inst(OP_TYPE_INT, &[i64_ty, 64, 1]);
        b.inst(OP_TYPE_POINTER, &[pc_ptr, STORAGE_CLASS_PUSH_CONSTANT, i64_ty]);
        b.inst(OP_VARIABLE, &[pc_ptr, pc_var, STORAGE_CLASS_PUSH_CONSTANT]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);

        let words = spirv_bytes_to_words(&b.finish_bytes()).unwrap();
        let sanitized = sanitize_spirv_words(&words, &SanitizeOptions::default()).unwrap();
        assert!(sanitized.audit.has_blockers());

        let blocker_kinds: HashSet<CompatIssueKind> =
            sanitized.audit.blockers().map(|i| i.kind).collect();
        for expected in [
            CompatIssueKind::RayTracing,
            CompatIssueKind::MeshShader,
            CompatIssueKind::Float64,
            CompatIssueKind::Int64,
            CompatIssueKind::SubgroupWave64,
            CompatIssueKind::BindlessDescriptorIndexing,
            CompatIssueKind::PushConstants,
            CompatIssueKind::GeometryShader,
            CompatIssueKind::Tessellation,
        ] {
            assert!(
                blocker_kinds.contains(&expected),
                "expected blocker {expected:?} in {blocker_kinds:?}"
            );
        }
    }

    #[test]
    fn detects_oversized_workgroup_in_spirv_and_naga_audit() {
        // 512x1x1 exceeds maxComputeWorkgroupSizeX (256) and maxComputeInvocationsPerWorkgroup (256)
        let spv_512 = build_compute_with_workgroup_size(512, 1, 1);
        let words = spirv_bytes_to_words(&spv_512).unwrap();
        let pre_audit = sanitize_spirv_words(&words, &SanitizeOptions::default())
            .unwrap()
            .audit;
        assert!(
            pre_audit.blockers().any(|i| {
                i.kind == CompatIssueKind::WorkgroupSizeLimitExceeded
                    && i.message.contains("512, 1, 1")
            }),
            "expected WorkgroupSizeLimitExceeded in SPIR-V pre-pass: {:?}",
            pre_audit.issues
        );

        // Parse with Naga and verify audit_naga_module independently catches it too.
        let module = naga::front::spv::parse_u8_slice(
            &spv_512,
            &naga::front::spv::Options {
                adjust_coordinate_space: true,
                strict_capabilities: true,
                block_ctx_dump_prefix: None,
            },
        )
        .expect("512x1x1 compute shader parses in Naga");
        let naga_audit = audit_naga_module(&module);
        assert!(
            naga_audit.blockers().any(|i| {
                i.kind == CompatIssueKind::WorkgroupSizeLimitExceeded
                    && i.message.contains("main")
                    && i.message.contains("512, 1, 1")
            }),
            "expected WorkgroupSizeLimitExceeded in Naga audit: {:?}",
            naga_audit.issues
        );

        // Also check 16x16x2 (each dim <= 256/256/64, but product 512 > 256 invocations)
        // and 1x1x128 (total 128 <= 256, but Z=128 > maxComputeWorkgroupSizeZ=64).
        for (x, y, z) in [(16, 16, 2), (1, 1, 128)] {
            let spv_over = build_compute_with_workgroup_size(x, y, z);
            let rec = audit_spirv_file("over.spv", &spv_over, &ConvertOptions::default());
            assert!(
                rec.audit
                    .blockers()
                    .any(|i| i.kind == CompatIssueKind::WorkgroupSizeLimitExceeded),
                "expected WorkgroupSizeLimitExceeded for ({x}, {y}, {z}): {:?}",
                rec.audit.issues
            );
        }

        // Valid baseline 16x16x1 = 256 invocations lowers cleanly with zero blockers or warnings.
        let spv_ok = build_compute_with_workgroup_size(16, 16, 1);
        let ok_rec = audit_spirv_file("ok.spv", &spv_ok, &ConvertOptions::default());
        assert!(ok_rec.lowered_to_wgsl);
        assert!(!ok_rec.audit.has_blockers());
        assert_eq!(ok_rec.audit.warning_count(), 0);
    }

    #[test]
    fn audit_naga_module_checks_bind_groups_immediates_and_storage_buffer_limits() {
        // 1. 9 storage buffers in @group(4) triggers both BindGroupLimitExceeded (Blocker)
        //    and StorageBufferLimitExceeded (Warning).
        let spv = build_compute_with_storage_buffers(9, 4);
        let module = naga::front::spv::parse_u8_slice(
            &spv,
            &naga::front::spv::Options {
                adjust_coordinate_space: true,
                strict_capabilities: true,
                block_ctx_dump_prefix: None,
            },
        )
        .expect("storage buffer compute shader parses in Naga");
        let audit = audit_naga_module(&module);
        assert!(
            audit
                .blockers()
                .any(|i| i.kind == CompatIssueKind::BindGroupLimitExceeded),
            "expected BindGroupLimitExceeded for @group(4): {:?}",
            audit.issues
        );
        assert!(
            audit
                .warnings()
                .any(|i| i.kind == CompatIssueKind::StorageBufferLimitExceeded),
            "expected StorageBufferLimitExceeded for 9 storage buffers: {:?}",
            audit.issues
        );

        // Clean 4 storage buffers in @group(0) has zero blockers and zero warnings.
        let spv_clean = build_compute_with_storage_buffers(4, 0);
        let clean_rec = audit_spirv_file("clean.spv", &spv_clean, &ConvertOptions::default());
        assert!(clean_rec.lowered_to_wgsl);
        assert!(!clean_rec.audit.has_blockers());
        assert_eq!(clean_rec.audit.warning_count(), 0);

        // 2. Programmatic Naga module with AddressSpace::Immediate (push constants).
        let mut imm_module = naga::Module::default();
        let u32_ty = imm_module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::U32),
            },
            naga::Span::UNDEFINED,
        );
        imm_module.global_variables.append(
            naga::GlobalVariable {
                name: Some("pc".to_string()),
                space: naga::AddressSpace::Immediate,
                binding: None,
                ty: u32_ty,
                init: None,
                memory_decorations: naga::MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        let imm_audit = audit_naga_module(&imm_module);
        assert!(
            imm_audit
                .blockers()
                .any(|i| i.kind == CompatIssueKind::PushConstants),
            "expected PushConstants blocker for AddressSpace::Immediate: {:?}",
            imm_audit.issues
        );
    }

    #[test]
    fn malformed_and_truncated_spirv_returns_clean_errors_without_panicking() {
        // Empty and truncated headers
        assert!(matches!(
            sanitize_spirv_words(&[], &SanitizeOptions::default()),
            Err(SpirvStreamError::TooShort { .. })
        ));
        assert!(matches!(
            sanitize_spirv_words(&[SPIRV_MAGIC, 0x0001_0300, 0], &SanitizeOptions::default()),
            Err(SpirvStreamError::TooShort { .. })
        ));

        // Unaligned byte buffer
        assert!(matches!(
            sanitize_spirv_bytes(&[0u8; 22], &SanitizeOptions::default()),
            Err(SpirvStreamError::UnalignedByteLength { byte_len: 22 })
        ));

        // Bad magic and big-endian magic
        assert!(matches!(
            sanitize_spirv_words(&[0xdead_beef, 0, 0, 1, 0], &SanitizeOptions::default()),
            Err(SpirvStreamError::InvalidMagic {
                is_big_endian: false,
                ..
            })
        ));
        assert!(matches!(
            sanitize_spirv_words(
                &[SPIRV_MAGIC.swap_bytes(), 0, 0, 1, 0],
                &SanitizeOptions::default()
            ),
            Err(SpirvStreamError::InvalidMagic {
                is_big_endian: true,
                ..
            })
        ));

        // Zero word_count instruction (would infinite-loop without validation)
        let zero_wc = [SPIRV_MAGIC, 0x0001_0300, 0, 1, 0, 0x0000_0011];
        assert!(matches!(
            sanitize_spirv_words(&zero_wc, &SanitizeOptions::default()),
            Err(SpirvStreamError::InvalidWordCount {
                word_offset: 5,
                opcode: 17
            })
        ));

        // Instruction declaring more words than remain in the stream
        let truncated_inst = [
            SPIRV_MAGIC,
            0x0001_0300,
            0,
            5,
            0,
            (6 << 16) | u32::from(OP_EXECUTION_MODE),
            3,
            EXEC_MODE_LOCAL_SIZE,
        ];
        assert!(matches!(
            sanitize_spirv_words(&truncated_inst, &SanitizeOptions::default()),
            Err(SpirvStreamError::TruncatedInstruction {
                word_offset: 5,
                opcode: OP_EXECUTION_MODE,
                declared_words: 6,
                available_words: 3
            })
        ));

        // Unterminated OpExtension string & short OpCapability do not panic
        let odd_operands = [
            SPIRV_MAGIC,
            0x0001_0300,
            0,
            5,
            0,
            (1 << 16) | u32::from(OP_CAPABILITY),
            (2 << 16) | u32::from(OP_EXTENSION),
            0x4141_4141, // "AAAA" without NUL terminator
        ];
        let res = sanitize_spirv_words(&odd_operands, &SanitizeOptions::default());
        assert!(res.is_ok(), "framed stream with odd operands must not panic");
    }

    #[test]
    fn compat_audit_report_renders_summary_and_grouped_findings() {
        let mut fc_audit = CompatAudit {
            rewritten_instructions: 3,
            original_word_count: 50,
            sanitized_word_count: 42,
            ..CompatAudit::default()
        };
        fc_audit.push_unique(CompatIssue {
            severity: CompatSeverity::Rewritten,
            kind: CompatIssueKind::FloatControlsStripped,
            message: "stripped OpExtension \"SPV_KHR_float_controls\"".to_string(),
            hint: None,
        });

        let mut rt_audit = CompatAudit::default();
        rt_audit.push_unique(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::RayTracing,
            message: "SPIR-V declares unsupported ray tracing capability RayTracingKHR".to_string(),
            hint: Some("WebGPU baseline lacks ray tracing".to_string()),
        });

        let md = render_compat_audit_report_md(&[
            ShaderAuditRecord {
                source_path: "cs_float.spv".to_string(),
                stage: Some("compute".to_string()),
                lowered_to_wgsl: true,
                stream_error: None,
                naga_error: None,
                audit: fc_audit,
            },
            ShaderAuditRecord {
                source_path: "rt_miss.spv".to_string(),
                stage: None,
                lowered_to_wgsl: false,
                stream_error: None,
                naga_error: Some("UnsupportedCapability(RayTracingKHR)".to_string()),
                audit: rt_audit,
            },
        ]);

        assert!(md.contains("# SPIR-V WebGPU Compatibility Audit Report"));
        assert!(md.contains("FloatControlsStripped"));
        assert!(md.contains("RayTracing"));
        assert!(md.contains("cs_float.spv"));
        assert!(md.contains("rt_miss.spv"));
    }
}
