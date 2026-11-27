use std::collections::{BTreeMap, BTreeSet};
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

/// WebGPU baseline default `maxComputeWorkgroupStorageSize` (bytes).
pub const WEBGPU_MAX_COMPUTE_WORKGROUP_STORAGE_SIZE: u32 = 16_384;

/// WebGPU baseline default `maxBindGroups` (valid bind group indices are `0..4`).
pub const WEBGPU_MAX_BIND_GROUPS: u32 = 4;

/// WebGPU baseline default `maxBindingsPerBindGroup`.
pub const WEBGPU_MAX_BINDINGS_PER_BIND_GROUP: usize = 1000;

/// WebGPU baseline default `maxDynamicUniformBuffersPerPipelineLayout`.
pub const WEBGPU_MAX_DYNAMIC_UNIFORM_BUFFERS_PER_PIPELINE_LAYOUT: usize = 8;

/// WebGPU baseline default `maxDynamicStorageBuffersPerPipelineLayout`.
pub const WEBGPU_MAX_DYNAMIC_STORAGE_BUFFERS_PER_PIPELINE_LAYOUT: usize = 4;

/// WebGPU baseline default `maxStorageBuffersPerShaderStage`.
pub const WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE: usize = 8;

/// WebGPU baseline default `maxUniformBuffersPerShaderStage`.
pub const WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE: usize = 12;

/// WebGPU baseline default `maxSampledTexturesPerShaderStage`.
pub const WEBGPU_MAX_SAMPLED_TEXTURES_PER_SHADER_STAGE: usize = 16;

/// WebGPU baseline default `maxSamplersPerShaderStage`.
pub const WEBGPU_MAX_SAMPLERS_PER_SHADER_STAGE: usize = 16;

/// WebGPU baseline default `maxStorageTexturesPerShaderStage`.
pub const WEBGPU_MAX_STORAGE_TEXTURES_PER_SHADER_STAGE: usize = 4;

/// WebGPU baseline default `maxColorAttachments` (8 MRTs for deferred G-buffer passes).
pub const WEBGPU_MAX_COLOR_ATTACHMENTS: u32 = 8;

/// WebGPU baseline default `maxInterStageShaderVariables`.
pub const WEBGPU_MAX_INTER_STAGE_SHADER_VARIABLES: u32 = 16;

/// WebGPU baseline default `maxVertexAttributes`.
pub const WEBGPU_MAX_VERTEX_ATTRIBUTES: u32 = 16;

/// WebGPU baseline default `maxUniformBufferBindingSize` (64 KiB).
pub const WEBGPU_MAX_UNIFORM_BUFFER_BINDING_SIZE: u64 = 65_536;

/// WebGPU baseline default `maxStorageBufferBindingSize` (128 MiB).
pub const WEBGPU_MAX_STORAGE_BUFFER_BINDING_SIZE: u64 = 134_217_728;

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
const CAP_FLOAT16: u32 = 9;
const CAP_FLOAT64: u32 = 10;
const CAP_INT64: u32 = 11;
const CAP_CLIP_DISTANCE: u32 = 32;
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
const EXEC_MODE_LOCAL_SIZE_ID: u32 = 38;
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
    WorkgroupStorageLimitExceeded,
    BindGroupLimitExceeded,
    BindingsPerBindGroupLimitExceeded,
    RequiredFeatureMissing,
    // WebGPU baseline limit warnings
    StorageBufferLimitExceeded,
    UniformBufferLimitExceeded,
    DynamicUniformBufferLimitExceeded,
    DynamicStorageBufferLimitExceeded,
    SampledTextureLimitExceeded,
    SamplerLimitExceeded,
    StorageTextureLimitExceeded,
    OverrideWorkgroupSize,
    // 3D scene / MRT / interface / RT arena compatibility findings (S71)
    ColorAttachmentLimitExceeded,
    InterStageVariablesLimitExceeded,
    VertexAttributeLimitExceeded,
    UniformBufferBindingSizeExceeded,
    StorageBufferBindingSizeExceeded,
    RtArenaUnsupported,
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
            Self::WorkgroupStorageLimitExceeded => "WorkgroupStorageLimitExceeded",
            Self::BindGroupLimitExceeded => "BindGroupLimitExceeded",
            Self::BindingsPerBindGroupLimitExceeded => "BindingsPerBindGroupLimitExceeded",
            Self::RequiredFeatureMissing => "RequiredFeatureMissing",
            Self::StorageBufferLimitExceeded => "StorageBufferLimitExceeded",
            Self::UniformBufferLimitExceeded => "UniformBufferLimitExceeded",
            Self::DynamicUniformBufferLimitExceeded => "DynamicUniformBufferLimitExceeded",
            Self::DynamicStorageBufferLimitExceeded => "DynamicStorageBufferLimitExceeded",
            Self::SampledTextureLimitExceeded => "SampledTextureLimitExceeded",
            Self::SamplerLimitExceeded => "SamplerLimitExceeded",
            Self::StorageTextureLimitExceeded => "StorageTextureLimitExceeded",
            Self::OverrideWorkgroupSize => "OverrideWorkgroupSize",
            Self::ColorAttachmentLimitExceeded => "ColorAttachmentLimitExceeded",
            Self::InterStageVariablesLimitExceeded => "InterStageVariablesLimitExceeded",
            Self::VertexAttributeLimitExceeded => "VertexAttributeLimitExceeded",
            Self::UniformBufferBindingSizeExceeded => "UniformBufferBindingSizeExceeded",
            Self::StorageBufferBindingSizeExceeded => "StorageBufferBindingSizeExceeded",
            Self::RtArenaUnsupported => "RtArenaUnsupported",
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

    /// Sort recorded issues into canonical deterministic order:
    /// severity (`Blocker` -> `Warning` -> `Rewritten`), then `kind`, `message`, and `hint`.
    pub fn sort_deterministic(&mut self) {
        self.issues.sort_by(|a, b| {
            severity_rank(a.severity)
                .cmp(&severity_rank(b.severity))
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.message.cmp(&b.message))
                .then_with(|| a.hint.cmp(&b.hint))
        });
    }

    /// Merge another audit (for example, post-Naga limits audit) into this one.
    ///
    /// Deduplicates overlapping `(severity, kind)` findings that were already caught during the
    /// SPIR-V pre-pass so single-shader issue counts stay accurate, and sorts deterministically.
    pub fn merge(&mut self, other: CompatAudit) {
        self.rewritten_instructions = self
            .rewritten_instructions
            .saturating_add(other.rewritten_instructions);
        for issue in other.issues {
            if issue.severity == CompatSeverity::Rewritten
                || issue.kind == CompatIssueKind::RequiredFeatureMissing
            {
                self.push_unique(issue);
            } else {
                self.push_once_per_kind(issue);
            }
        }
        self.sort_deterministic();
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
    let mut bytes = Vec::with_capacity(words.len().saturating_mul(4));
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

    let mut stripped_non_semantic_sets: BTreeSet<u32> = BTreeSet::new();
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
                            audit.rewritten_instructions =
                                audit.rewritten_instructions.saturating_add(1);
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
                    stripped_non_semantic_calls = stripped_non_semantic_calls.saturating_add(1);
                    audit.rewritten_instructions = audit.rewritten_instructions.saturating_add(1);
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
    audit.sort_deterministic();

    Ok(SanitizedSpirv { words: out, audit })
}

fn decode_spirv_string(operand_words: &[u32]) -> Option<String> {
    if operand_words.is_empty() {
        return None;
    }
    let mut raw = Vec::with_capacity(operand_words.len().saturating_mul(4));
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
                audit.rewritten_instructions = audit.rewritten_instructions.saturating_add(1);
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
                audit.rewritten_instructions = audit.rewritten_instructions.saturating_add(1);
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
                audit.rewritten_instructions = audit.rewritten_instructions.saturating_add(1);
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
                audit.rewritten_instructions = audit.rewritten_instructions.saturating_add(1);
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
        EXEC_MODE_LOCAL_SIZE_ID if opcode == OP_EXECUTION_MODE_ID => {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Warning,
                kind: CompatIssueKind::OverrideWorkgroupSize,
                message: "SPIR-V uses OpExecutionModeId LocalSizeId (specialization-constant workgroup size); verify runtime values stay within WebGPU default limits".to_string(),
                hint: Some(
                    "ensure specialized workgroup dimensions stay within 256x256x64 and <= 256 invocations"
                        .to_string(),
                ),
            });
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
    check_workgroup_size_with_limits(size, entry_point, &WebGpuLimits::webgpu_baseline(), audit);
}

fn check_workgroup_size_with_limits(
    size: [u32; 3],
    entry_point: Option<&str>,
    limits: &WebGpuLimits,
    audit: &mut CompatAudit,
) {
    let [x, y, z] = size;
    // Exact product in u128: (u32::MAX)^3 < 2^96 < 2^128, so this never overflows.
    let total = u128::from(x) * u128::from(y) * u128::from(z);

    let max_x = limits.max_compute_workgroup_size_x;
    let max_y = limits.max_compute_workgroup_size_y;
    let max_z = limits.max_compute_workgroup_size_z;
    let max_inv = limits.max_compute_invocations_per_workgroup;

    let has_zero = x == 0 || y == 0 || z == 0;
    let exceeds_dim = x > max_x || y > max_y || z > max_z;
    let exceeds_total = total > u128::from(max_inv);

    if has_zero || exceeds_dim || exceeds_total {
        let prefix = match entry_point {
            Some(name) => format!("compute entry point '{name}' "),
            None => String::new(),
        };
        let message = if has_zero {
            format!(
                "{prefix}workgroup_size({x}, {y}, {z}) has a zero dimension; WebGPU requires each workgroup dimension to be >= 1 (limits {max_x}x{max_y}x{max_z}, max {max_inv} invocations)"
            )
        } else {
            format!(
                "{prefix}workgroup_size({x}, {y}, {z}) = {total} invocations exceeds WebGPU limits ({max_x}x{max_y}x{max_z}, max {max_inv} invocations)"
            )
        };
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
            message,
            hint: Some(format!(
                "ensure each workgroup dimension is >= 1 and retile dispatch threads within {max_x}x{max_y}x{max_z} (<= {max_inv} invocations)"
            )),
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
/// - Compute workgroup storage size (`maxComputeWorkgroupStorageSize = 16384` bytes)
/// - Unsupported shader stages (`Task`, `Mesh`, ray tracing stages)
/// - Bind group index ceiling (`maxBindGroups = 4`, indices `0..4`) and per-group binding counts
/// - Push constants / immediate data (`AddressSpace::Immediate`)
/// - Per-stage storage buffer, uniform buffer, sampled texture, sampler, and storage texture counts
/// - Unsupported IR types (`f64`, `i64`/`u64`, `BindingArray`, `AccelerationStructure`, `RayQuery`)
pub fn audit_naga_module(module: &naga::Module) -> CompatAudit {
    audit_naga_module_with_limits(module, &WebGpuLimits::webgpu_baseline())
}

/// Audit a parsed [`naga::Module`] against explicit [`WebGpuLimits`], producing deterministically
/// ordered compatibility diagnostics.
pub fn audit_naga_module_with_limits(
    module: &naga::Module,
    limits: &WebGpuLimits,
) -> CompatAudit {
    let mut audit = CompatAudit::default();

    // 1. Entry points: stages and compute workgroup limits.
    let mut has_compute_entry_point = false;
    for ep in &module.entry_points {
        match ep.stage {
            naga::ShaderStage::Compute => {
                has_compute_entry_point = true;
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
                    check_workgroup_size_with_limits(
                        ep.workgroup_size,
                        Some(&ep.name),
                        limits,
                        &mut audit,
                    );
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

    // 2. Global variables: address spaces, bind groups, per-group binding counts, push constants.
    let mut module_counts = StageResourceCounts::default();
    let mut bindings_per_group: BTreeMap<u32, usize> = BTreeMap::new();

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
            _ => {}
        }

        tally_global_resource(module, var, &mut module_counts);

        if let Some(binding) = &var.binding {
            let group_count = bindings_per_group.entry(binding.group).or_insert(0);
            *group_count = group_count.saturating_add(1);

            if binding.group >= limits.max_bind_groups {
                let label = var.name.as_deref().unwrap_or("<unnamed>");
                audit.push_once_per_kind(CompatIssue {
                    severity: CompatSeverity::Blocker,
                    kind: CompatIssueKind::BindGroupLimitExceeded,
                    message: format!(
                        "resource '{label}' uses @group({}) @binding({}), exceeding WebGPU maxBindGroups ({}; valid groups 0..{})",
                        binding.group,
                        binding.binding,
                        limits.max_bind_groups,
                        limits.max_bind_groups
                    ),
                    hint: Some(format!(
                        "pack descriptor sets into <= {} bind groups for WebGPU compatibility",
                        limits.max_bind_groups
                    )),
                });
            }
        }
    }

    for (&group, &count) in &bindings_per_group {
        if count > limits.max_bindings_per_bind_group {
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::BindingsPerBindGroupLimitExceeded,
                message: format!(
                    "@group({group}) declares {count} bindings, exceeding WebGPU maxBindingsPerBindGroup ({})",
                    limits.max_bindings_per_bind_group
                ),
                hint: Some(format!(
                    "reduce bindings in @group({group}) to <= {}",
                    limits.max_bindings_per_bind_group
                )),
            });
        }
    }

    // 2b. Compute workgroup storage budget check (only when compute entry point or workgroup vars exist).
    let workgroup_storage_bytes = compute_workgroup_storage_bytes(module);
    if (has_compute_entry_point || workgroup_storage_bytes > 0)
        && workgroup_storage_bytes > limits.max_compute_workgroup_storage_size
    {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::WorkgroupStorageLimitExceeded,
            message: format!(
                "module uses {workgroup_storage_bytes} bytes of workgroup storage, exceeding WebGPU maxComputeWorkgroupStorageSize ({})",
                limits.max_compute_workgroup_storage_size
            ),
            hint: Some(format!(
                "reduce var<workgroup> shared memory usage to <= {} bytes or request a higher device limit",
                limits.max_compute_workgroup_storage_size
            )),
        });
    }

    // 3. Per-stage binding counts (falls back to module-wide globals when an entry point
    //    doesn't directly reference unused globals or when there are 0 entry points).
    if module.entry_points.is_empty() {
        check_stage_resource_limits(None, &module_counts, limits, &mut audit);
    } else {
        for ep in &module.entry_points {
            let referenced = collect_referenced_globals(module, &ep.function);
            let mut stage_counts = StageResourceCounts::default();
            for handle in referenced {
                tally_global_resource(module, &module.global_variables[handle], &mut stage_counts);
            }
            // If a synthetic module declares global resource bindings without loading them in
            // the stub body, also honor the module-wide count so limit audits still catch them.
            let effective = if module.entry_points.len() == 1 {
                stage_counts.elementwise_max(&module_counts)
            } else {
                stage_counts
            };
            check_stage_resource_limits(Some(&ep.name), &effective, limits, &mut audit);
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

    audit.sort_deterministic();
    audit
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct StageResourceCounts {
    storage_buffers: usize,
    uniform_buffers: usize,
    sampled_textures: usize,
    samplers: usize,
    storage_textures: usize,
}

impl StageResourceCounts {
    fn elementwise_max(&self, other: &Self) -> Self {
        Self {
            storage_buffers: self.storage_buffers.max(other.storage_buffers),
            uniform_buffers: self.uniform_buffers.max(other.uniform_buffers),
            sampled_textures: self.sampled_textures.max(other.sampled_textures),
            samplers: self.samplers.max(other.samplers),
            storage_textures: self.storage_textures.max(other.storage_textures),
        }
    }
}

fn tally_global_resource(
    module: &naga::Module,
    var: &naga::GlobalVariable,
    counts: &mut StageResourceCounts,
) {
    match var.space {
        naga::AddressSpace::Storage { .. } => {
            counts.storage_buffers = counts.storage_buffers.saturating_add(1);
        }
        naga::AddressSpace::Uniform => {
            counts.uniform_buffers = counts.uniform_buffers.saturating_add(1);
        }
        naga::AddressSpace::Handle => match &module.types[var.ty].inner {
            naga::TypeInner::Image { class, .. } => match class {
                naga::ImageClass::Storage { .. } => {
                    counts.storage_textures = counts.storage_textures.saturating_add(1);
                }
                _ => {
                    counts.sampled_textures = counts.sampled_textures.saturating_add(1);
                }
            },
            naga::TypeInner::Sampler { .. } => {
                counts.samplers = counts.samplers.saturating_add(1);
            }
            _ => {}
        },
        _ => {}
    }
}

fn compute_workgroup_storage_bytes(module: &naga::Module) -> u32 {
    let mut layouter = naga::proc::Layouter::default();
    if layouter.update(module.to_ctx()).is_err() {
        return 0;
    }
    let mut total: u32 = 0;
    for (_, var) in module.global_variables.iter() {
        if var.space == naga::AddressSpace::WorkGroup {
            let raw_size = layouter[var.ty].size;
            // WebGPU rounds each workgroup variable up to a 16-byte multiple.
            let aligned = raw_size.saturating_add(15) & !15u32;
            total = total.saturating_add(aligned);
        }
    }
    total
}

fn check_stage_resource_limits(
    entry_point: Option<&str>,
    counts: &StageResourceCounts,
    limits: &WebGpuLimits,
    audit: &mut CompatAudit,
) {
    let scope = match entry_point {
        Some(name) => format!("entry point '{name}'"),
        None => "module".to_string(),
    };

    if counts.storage_buffers > limits.max_storage_buffers_per_shader_stage {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::StorageBufferLimitExceeded,
            message: format!(
                "{scope} uses {} storage buffer bindings, exceeding WebGPU maxStorageBuffersPerShaderStage ({})",
                counts.storage_buffers, limits.max_storage_buffers_per_shader_stage
            ),
            hint: Some(
                "request higher maxStorageBuffersPerShaderStage or merge read-only buffers into a combined storage buffer"
                    .to_string(),
            ),
        });
    }

    if counts.uniform_buffers > limits.max_uniform_buffers_per_shader_stage {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::UniformBufferLimitExceeded,
            message: format!(
                "{scope} uses {} uniform buffer bindings, exceeding WebGPU maxUniformBuffersPerShaderStage ({})",
                counts.uniform_buffers, limits.max_uniform_buffers_per_shader_stage
            ),
            hint: Some(
                "pack uniform buffers into fewer bindings or a shared constant buffer".to_string(),
            ),
        });
    }

    if counts.sampled_textures > limits.max_sampled_textures_per_shader_stage {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::SampledTextureLimitExceeded,
            message: format!(
                "{scope} uses {} sampled texture bindings, exceeding WebGPU maxSampledTexturesPerShaderStage ({})",
                counts.sampled_textures, limits.max_sampled_textures_per_shader_stage
            ),
            hint: Some(
                "combine textures into texture_2d_array atlases or request higher maxSampledTexturesPerShaderStage"
                    .to_string(),
            ),
        });
    }

    if counts.samplers > limits.max_samplers_per_shader_stage {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::SamplerLimitExceeded,
            message: format!(
                "{scope} uses {} sampler bindings, exceeding WebGPU maxSamplersPerShaderStage ({})",
                counts.samplers, limits.max_samplers_per_shader_stage
            ),
            hint: Some(
                "deduplicate immutable sampler states to stay within maxSamplersPerShaderStage"
                    .to_string(),
            ),
        });
    }

    if counts.storage_textures > limits.max_storage_textures_per_shader_stage {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::StorageTextureLimitExceeded,
            message: format!(
                "{scope} uses {} storage texture bindings, exceeding WebGPU maxStorageTexturesPerShaderStage ({})",
                counts.storage_textures, limits.max_storage_textures_per_shader_stage
            ),
            hint: Some(
                "pack storage outputs or request higher maxStorageTexturesPerShaderStage"
                    .to_string(),
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
) -> Vec<naga::Handle<naga::GlobalVariable>> {
    let mut globals: BTreeMap<usize, naga::Handle<naga::GlobalVariable>> = BTreeMap::new();
    let mut visited_fns: BTreeSet<usize> = BTreeSet::new();
    let mut pending_fns: Vec<naga::Handle<naga::Function>> = Vec::new();

    scan_function_globals(root, &mut globals, &mut pending_fns);
    while let Some(fn_handle) = pending_fns.pop() {
        if visited_fns.insert(fn_handle.index()) {
            scan_function_globals(&module.functions[fn_handle], &mut globals, &mut pending_fns);
        }
    }

    globals.into_values().collect()
}

fn scan_function_globals(
    func: &naga::Function,
    globals: &mut BTreeMap<usize, naga::Handle<naga::GlobalVariable>>,
    callees: &mut Vec<naga::Handle<naga::Function>>,
) {
    for (_, expr) in func.expressions.iter() {
        if let naga::Expression::GlobalVariable(handle) = *expr {
            globals.insert(handle.index(), handle);
        }
    }
    collect_calls_in_block(&func.body, callees);
}

fn collect_calls_in_block(block: &naga::Block, callees: &mut Vec<naga::Handle<naga::Function>>) {
    let mut worklist: Vec<&naga::Block> = vec![block];
    while let Some(current) = worklist.pop() {
        for stmt in current.iter() {
            match stmt {
                naga::Statement::Block(inner) => worklist.push(inner),
                naga::Statement::If { accept, reject, .. } => {
                    worklist.push(reject);
                    worklist.push(accept);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases.iter().rev() {
                        worklist.push(&case.body);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    worklist.push(continuing);
                    worklist.push(body);
                }
                naga::Statement::Call { function, .. } => {
                    callees.push(*function);
                }
                _ => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WebGPU Feature-Compatibility & Limit Guards (3D Main Menu & In-Game Shaders)
// ---------------------------------------------------------------------------

/// Optional WebGPU `GPUFeatureName` flags required by 3D Main Menu and playable in-game shaders.
///
/// Variants are declared in strict ascending lexicographical order of their canonical WebGPU
/// feature name (`as_str()`), guaranteeing deterministic ordering in `BTreeSet`s and reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WebGpuFeature {
    /// `bgra8unorm-storage` — `texture_storage_2d<bgra8unorm, ...>` read/write storage textures.
    Bgra8UnormStorage,
    /// `binding-array` — bindless `binding_array<T, N>` / non-uniform descriptor indexing extension.
    BindingArray,
    /// `clip-distances` — `@builtin(clip_distances)` vertex clipping planes.
    ClipDistances,
    /// `depth-clip-control` — unclipped depth (`unclippedDepth: true`).
    DepthClipControl,
    /// `depth32float-stencil8` — combined 32-bit float depth + 8-bit stencil attachments/textures.
    Depth32FloatStencil8,
    /// `dual-source-blending` — `@blend_src(0/1)` fragment outputs.
    DualSourceBlending,
    /// `float32-filterable` — linear filtering of 32-bit float textures (`r32float`/`rg32float`/`rgba32float`).
    Float32Filterable,
    /// `indirect-first-instance` — non-zero `firstInstance` in indirect draws.
    IndirectFirstInstance,
    /// `rg11b10ufloat-renderable` — HDR `rg11b10ufloat` render attachments.
    Rg11b10UfloatRenderable,
    /// `shader-f16` — `enable f16;` 16-bit floating-point WGSL arithmetic.
    ShaderF16,
    /// `subgroups` — `enable subgroups;` native subgroup operations & built-ins.
    Subgroups,
    /// `texture-compression-astc` — ASTC compressed textures.
    TextureCompressionAstc,
    /// `texture-compression-bc` — BC1–BC7 block-compressed textures (Xbox/D3D12 standard).
    TextureCompressionBc,
    /// `texture-compression-etc2` — ETC2/EAC compressed textures.
    TextureCompressionEtc2,
    /// `timestamp-query` — GPU pass/command timestamp queries.
    TimestampQuery,
}

impl WebGpuFeature {
    /// All known optional WebGPU features in canonical ascending order.
    pub const ALL: &'static [Self] = &[
        Self::Bgra8UnormStorage,
        Self::BindingArray,
        Self::ClipDistances,
        Self::DepthClipControl,
        Self::Depth32FloatStencil8,
        Self::DualSourceBlending,
        Self::Float32Filterable,
        Self::IndirectFirstInstance,
        Self::Rg11b10UfloatRenderable,
        Self::ShaderF16,
        Self::Subgroups,
        Self::TextureCompressionAstc,
        Self::TextureCompressionBc,
        Self::TextureCompressionEtc2,
        Self::TimestampQuery,
    ];

    /// Canonical WebGPU `GPUFeatureName` string.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bgra8UnormStorage => "bgra8unorm-storage",
            Self::BindingArray => "binding-array",
            Self::ClipDistances => "clip-distances",
            Self::DepthClipControl => "depth-clip-control",
            Self::Depth32FloatStencil8 => "depth32float-stencil8",
            Self::DualSourceBlending => "dual-source-blending",
            Self::Float32Filterable => "float32-filterable",
            Self::IndirectFirstInstance => "indirect-first-instance",
            Self::Rg11b10UfloatRenderable => "rg11b10ufloat-renderable",
            Self::ShaderF16 => "shader-f16",
            Self::Subgroups => "subgroups",
            Self::TextureCompressionAstc => "texture-compression-astc",
            Self::TextureCompressionBc => "texture-compression-bc",
            Self::TextureCompressionEtc2 => "texture-compression-etc2",
            Self::TimestampQuery => "timestamp-query",
        }
    }

    /// Parse a WebGPU `GPUFeatureName` string (case-insensitive, trimmed).
    pub fn from_name(name: &str) -> Option<Self> {
        let normalized = name.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return None;
        }
        Self::ALL
            .iter()
            .copied()
            .find(|feature| feature.as_str() == normalized)
    }

    /// Return the optional WebGPU feature required to create or sample a texture with `format`.
    ///
    /// Returns `None` for baseline core WebGPU texture formats as well as empty or unknown strings.
    pub fn required_for_texture_format(format: &str) -> Option<Self> {
        let normalized = format.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return None;
        }
        match normalized.as_str() {
            "bc1-rgba-unorm"
            | "bc1-rgba-unorm-srgb"
            | "bc2-rgba-unorm"
            | "bc2-rgba-unorm-srgb"
            | "bc3-rgba-unorm"
            | "bc3-rgba-unorm-srgb"
            | "bc4-r-unorm"
            | "bc4-r-snorm"
            | "bc5-rg-unorm"
            | "bc5-rg-snorm"
            | "bc6h-rgb-ufloat"
            | "bc6h-rgb-float"
            | "bc7-rgba-unorm"
            | "bc7-rgba-unorm-srgb" => Some(Self::TextureCompressionBc),
            "depth32float-stencil8" => Some(Self::Depth32FloatStencil8),
            "etc2-rgb8unorm"
            | "etc2-rgb8unorm-srgb"
            | "etc2-rgb8a1unorm"
            | "etc2-rgb8a1unorm-srgb"
            | "etc2-rgba8unorm"
            | "etc2-rgba8unorm-srgb"
            | "eac-r11unorm"
            | "eac-r11snorm"
            | "eac-rg11unorm"
            | "eac-rg11snorm" => Some(Self::TextureCompressionEtc2),
            other if is_astc_texture_format(other) => Some(Self::TextureCompressionAstc),
            _ => None,
        }
    }

    /// Return the optional WebGPU feature required to bind a storage texture with `format`.
    pub fn required_for_storage_texture_format(format: &str) -> Option<Self> {
        let normalized = format.trim().to_ascii_lowercase();
        if normalized == "bgra8unorm" {
            Some(Self::Bgra8UnormStorage)
        } else {
            None
        }
    }

    /// Return the optional WebGPU feature required to use `format` as a render/depth attachment.
    pub fn required_for_render_attachment_format(format: &str) -> Option<Self> {
        let normalized = format.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "depth32float-stencil8" => Some(Self::Depth32FloatStencil8),
            "rg11b10ufloat" => Some(Self::Rg11b10UfloatRenderable),
            _ => None,
        }
    }

    /// Return `Some(Float32Filterable)` when `format` is a 32-bit float texture sampled with a
    /// filtering sampler (`r32float`, `rg32float`, `rgba32float`).
    pub fn required_for_filterable_float_format(format: &str) -> Option<Self> {
        let normalized = format.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "r32float" | "rg32float" | "rgba32float" => Some(Self::Float32Filterable),
            _ => None,
        }
    }

    /// Deterministically detect optional WebGPU features required by a WGSL source module.
    ///
    /// Safely ignores empty inputs, line comments (`//`), and nested block comments (`/* ... */`).
    pub fn required_for_wgsl_source(wgsl: &str) -> BTreeSet<Self> {
        let mut features = BTreeSet::new();
        if wgsl.trim().is_empty() {
            return features;
        }

        let stripped = strip_wgsl_comments(wgsl);

        // 1. Parse `enable <ext1>, <ext2>;` directives.
        for stmt in stripped.split(';') {
            let trimmed = stmt.trim_start();
            if let Some(rest) = trimmed.strip_prefix("enable")
                && rest.starts_with(|c: char| c.is_ascii_whitespace())
            {
                for item in rest.split(',') {
                    match item.trim() {
                        "f16" => {
                            features.insert(Self::ShaderF16);
                        }
                        "subgroups" => {
                            features.insert(Self::Subgroups);
                        }
                        "clip_distances" => {
                            features.insert(Self::ClipDistances);
                        }
                        "dual_source_blending" => {
                            features.insert(Self::DualSourceBlending);
                        }
                        _ => {}
                    }
                }
            }
        }

        // 2. Token-boundary defense-in-depth detection for feature-gated WGSL constructs.
        if contains_wgsl_ident(&stripped, "f16") {
            features.insert(Self::ShaderF16);
        }
        if contains_wgsl_ident(&stripped, "blend_src") && stripped.contains("@blend_src") {
            features.insert(Self::DualSourceBlending);
        }
        if contains_wgsl_ident(&stripped, "clip_distances") {
            features.insert(Self::ClipDistances);
        }
        if uses_wgsl_bgra8unorm_storage(&stripped) {
            features.insert(Self::Bgra8UnormStorage);
        }
        if uses_wgsl_subgroups(&stripped) {
            features.insert(Self::Subgroups);
        }
        if contains_wgsl_ident(&stripped, "binding_array") {
            features.insert(Self::BindingArray);
        }

        features
    }

    /// Deterministically detect optional WebGPU features required by a SPIR-V 32-bit word stream.
    ///
    /// Performs full bounds-checked framing validation without panicking on empty or malformed data.
    pub fn required_for_spirv_words(words: &[u32]) -> Result<BTreeSet<Self>, SpirvStreamError> {
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

        let mut features = BTreeSet::new();
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
            match opcode {
                OP_CAPABILITY => {
                    if let Some(&cap) = inst.get(1) {
                        match cap {
                            CAP_FLOAT16 => {
                                features.insert(Self::ShaderF16);
                            }
                            CAP_CLIP_DISTANCE => {
                                features.insert(Self::ClipDistances);
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
                                features.insert(Self::Subgroups);
                            }
                            _ => {}
                        }
                    }
                }
                OP_EXTENSION => {
                    if let Some(ext_name) = decode_spirv_string(&inst[1..]) {
                        match ext_name.as_str() {
                            "SPV_KHR_shader_ballot"
                            | "SPV_KHR_shader_vote"
                            | "SPV_KHR_subgroup_uniform_control_flow"
                            | "SPV_KHR_subgroup_rotate"
                            | "SPV_KHR_shader_subgroup_extended_types" => {
                                features.insert(Self::Subgroups);
                            }
                            _ => {}
                        }
                    }
                }
                OP_EXECUTION_MODE | OP_EXECUTION_MODE_ID => {
                    if let Some(&mode) = inst.get(2)
                        && matches!(
                            mode,
                            EXEC_MODE_SUBGROUP_SIZE
                                | EXEC_MODE_SUBGROUPS_PER_WORKGROUP
                                | EXEC_MODE_SUBGROUPS_PER_WORKGROUP_ID
                                | EXEC_MODE_SUBGROUP_UNIFORM_CONTROL_FLOW_KHR
                        )
                    {
                        features.insert(Self::Subgroups);
                    }
                }
                OP_TYPE_FLOAT => {
                    if let Some(&width) = inst.get(2)
                        && width == 16
                    {
                        features.insert(Self::ShaderF16);
                    }
                }
                _ => {}
            }

            offset = end;
        }

        Ok(features)
    }

    /// Deterministically detect optional WebGPU features required by a parsed [`naga::Module`].
    pub fn required_for_naga_module(module: &naga::Module) -> BTreeSet<Self> {
        let mut features = BTreeSet::new();

        for (_, ty) in module.types.iter() {
            match &ty.inner {
                naga::TypeInner::Scalar(scalar)
                | naga::TypeInner::Vector { scalar, .. }
                | naga::TypeInner::Matrix { scalar, .. }
                | naga::TypeInner::Atomic(scalar) => {
                    if scalar.kind == naga::ScalarKind::Float && scalar.width == 2 {
                        features.insert(Self::ShaderF16);
                    }
                }
                naga::TypeInner::Image {
                    class:
                        naga::ImageClass::Storage {
                            format: naga::StorageFormat::Bgra8Unorm,
                            ..
                        },
                    ..
                } => {
                    features.insert(Self::Bgra8UnormStorage);
                }
                naga::TypeInner::Struct { members, .. } => {
                    for member in members {
                        if let Some(binding) = &member.binding {
                            collect_naga_binding_features(binding, &mut features);
                        }
                    }
                }
                naga::TypeInner::BindingArray { .. } => {
                    features.insert(Self::BindingArray);
                }
                _ => {}
            }
        }

        for ep in &module.entry_points {
            collect_naga_function_features(&ep.function, &mut features);
        }
        for (_, func) in module.functions.iter() {
            collect_naga_function_features(func, &mut features);
        }

        features
    }
}

impl fmt::Display for WebGpuFeature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

fn is_astc_texture_format(fmt: &str) -> bool {
    let Some(rest) = fmt.strip_prefix("astc-") else {
        return false;
    };
    let Some(dims) = rest
        .strip_suffix("-unorm-srgb")
        .or_else(|| rest.strip_suffix("-unorm"))
    else {
        return false;
    };
    matches!(
        dims,
        "4x4"
            | "5x4"
            | "5x5"
            | "6x5"
            | "6x6"
            | "8x5"
            | "8x6"
            | "8x8"
            | "10x5"
            | "10x6"
            | "10x8"
            | "10x10"
            | "12x10"
            | "12x12"
    )
}

fn strip_wgsl_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0usize;
    let mut block_depth: usize = 0;

    while i < bytes.len() {
        if block_depth > 0 {
            if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                block_depth = block_depth.saturating_add(1);
                out.push_str("  ");
                i += 2;
            } else if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                block_depth -= 1;
                out.push_str("  ");
                i += 2;
            } else {
                if bytes[i] == b'\n' {
                    out.push('\n');
                } else {
                    out.push(' ');
                }
                i += 1;
            }
            continue;
        }

        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            out.push_str("  ");
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            block_depth = 1;
            out.push_str("  ");
            i += 2;
        } else {
            let ch = src[i..].chars().next().unwrap_or(' ');
            out.push(ch);
            i += ch.len_utf8();
        }
    }

    out
}

fn is_wgsl_ident_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn contains_wgsl_ident(src: &str, ident: &str) -> bool {
    if ident.is_empty() || src.len() < ident.len() {
        return false;
    }
    let bytes = src.as_bytes();
    let mut start = 0usize;
    while let Some(rel) = src[start..].find(ident) {
        let idx = start + rel;
        let end = idx + ident.len();
        let before_ok = idx == 0 || !is_wgsl_ident_char(bytes[idx - 1]);
        let after_ok = end == bytes.len() || !is_wgsl_ident_char(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        start = idx + 1;
    }
    false
}

fn uses_wgsl_bgra8unorm_storage(stripped: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "texture_storage_1d",
        "texture_storage_2d",
        "texture_storage_2d_array",
        "texture_storage_3d",
    ];
    for prefix in PREFIXES {
        let mut start = 0usize;
        while let Some(rel) = stripped[start..].find(prefix) {
            let after_prefix = start + rel + prefix.len();
            let tail = stripped[after_prefix..].trim_start();
            if let Some(after_lt) = tail.strip_prefix('<')
                && let Some(close_idx) = after_lt.find('>')
                && contains_wgsl_ident(&after_lt[..close_idx], "bgra8unorm")
            {
                return true;
            }
            start = after_prefix;
        }
    }
    false
}

fn uses_wgsl_subgroups(stripped: &str) -> bool {
    const SUBGROUP_TOKENS: &[&str] = &[
        "subgroup_size",
        "subgroup_invocation_id",
        "subgroupBallot",
        "subgroupAdd",
        "subgroupMul",
        "subgroupMin",
        "subgroupMax",
        "subgroupAnd",
        "subgroupOr",
        "subgroupXor",
        "subgroupAll",
        "subgroupAny",
        "subgroupElect",
        "subgroupBroadcast",
        "subgroupBroadcastFirst",
        "subgroupShuffle",
        "subgroupShuffleXor",
        "subgroupShuffleUp",
        "subgroupShuffleDown",
        "subgroupExclusiveAdd",
        "subgroupExclusiveMul",
        "subgroupInclusiveAdd",
        "subgroupInclusiveMul",
        "quadBroadcast",
        "quadSwapX",
        "quadSwapY",
        "quadSwapDiagonal",
    ];
    SUBGROUP_TOKENS
        .iter()
        .any(|token| contains_wgsl_ident(stripped, token))
}

fn collect_naga_binding_features(binding: &naga::Binding, features: &mut BTreeSet<WebGpuFeature>) {
    match binding {
        naga::Binding::BuiltIn(builtin) => match builtin {
            naga::BuiltIn::SubgroupSize
            | naga::BuiltIn::SubgroupInvocationId
            | naga::BuiltIn::NumSubgroups
            | naga::BuiltIn::SubgroupId => {
                features.insert(WebGpuFeature::Subgroups);
            }
            naga::BuiltIn::ClipDistances => {
                features.insert(WebGpuFeature::ClipDistances);
            }
            _ => {}
        },
        naga::Binding::Location {
            blend_src: Some(_), ..
        } => {
            features.insert(WebGpuFeature::DualSourceBlending);
        }
        _ => {}
    }
}

fn collect_naga_function_features(func: &naga::Function, features: &mut BTreeSet<WebGpuFeature>) {
    for arg in &func.arguments {
        if let Some(binding) = &arg.binding {
            collect_naga_binding_features(binding, features);
        }
    }
    if let Some(result) = &func.result
        && let Some(binding) = &result.binding
    {
        collect_naga_binding_features(binding, features);
    }

    for (_, expr) in func.expressions.iter() {
        if matches!(expr, naga::Expression::SubgroupOperationResult { .. }) {
            features.insert(WebGpuFeature::Subgroups);
        }
    }

    let mut worklist: Vec<&naga::Block> = vec![&func.body];
    while let Some(block) = worklist.pop() {
        for stmt in block.iter() {
            match stmt {
                naga::Statement::SubgroupBallot { .. }
                | naga::Statement::SubgroupCollectiveOperation { .. }
                | naga::Statement::SubgroupGather { .. } => {
                    features.insert(WebGpuFeature::Subgroups);
                }
                naga::Statement::Block(inner) => worklist.push(inner),
                naga::Statement::If { accept, reject, .. } => {
                    worklist.push(reject);
                    worklist.push(accept);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases.iter().rev() {
                        worklist.push(&case.body);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    worklist.push(continuing);
                    worklist.push(body);
                }
                _ => {}
            }
        }
    }
}

/// WebGPU `GPUSupportedLimits` subset enforced by shader compatibility guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebGpuLimits {
    pub max_bind_groups: u32,
    pub max_bindings_per_bind_group: usize,
    pub max_dynamic_uniform_buffers_per_pipeline_layout: usize,
    pub max_dynamic_storage_buffers_per_pipeline_layout: usize,
    pub max_storage_buffers_per_shader_stage: usize,
    pub max_uniform_buffers_per_shader_stage: usize,
    pub max_sampled_textures_per_shader_stage: usize,
    pub max_samplers_per_shader_stage: usize,
    pub max_storage_textures_per_shader_stage: usize,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_storage_size: u32,
}

impl WebGpuLimits {
    /// WebGPU specification baseline default limits (`GPUSupportedLimits` defaults).
    pub const fn webgpu_baseline() -> Self {
        Self {
            max_bind_groups: WEBGPU_MAX_BIND_GROUPS,
            max_bindings_per_bind_group: WEBGPU_MAX_BINDINGS_PER_BIND_GROUP,
            max_dynamic_uniform_buffers_per_pipeline_layout:
                WEBGPU_MAX_DYNAMIC_UNIFORM_BUFFERS_PER_PIPELINE_LAYOUT,
            max_dynamic_storage_buffers_per_pipeline_layout:
                WEBGPU_MAX_DYNAMIC_STORAGE_BUFFERS_PER_PIPELINE_LAYOUT,
            max_storage_buffers_per_shader_stage: WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
            max_uniform_buffers_per_shader_stage: WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE,
            max_sampled_textures_per_shader_stage: WEBGPU_MAX_SAMPLED_TEXTURES_PER_SHADER_STAGE,
            max_samplers_per_shader_stage: WEBGPU_MAX_SAMPLERS_PER_SHADER_STAGE,
            max_storage_textures_per_shader_stage: WEBGPU_MAX_STORAGE_TEXTURES_PER_SHADER_STAGE,
            max_compute_workgroup_size_x: WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_X,
            max_compute_workgroup_size_y: WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Y,
            max_compute_workgroup_size_z: WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Z,
            max_compute_invocations_per_workgroup: WEBGPU_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
            max_compute_workgroup_storage_size: WEBGPU_MAX_COMPUTE_WORKGROUP_STORAGE_SIZE,
        }
    }

    /// WebGPU / Dawn D3D12 hardware per-stage binding ceilings (12 uniform buffers, 16 storage
    /// buffers, 8 storage textures, 16 sampled textures, 16 samplers per shader stage).
    pub const fn webgpu_d3d12() -> Self {
        let mut limits = Self::webgpu_baseline();
        limits.max_uniform_buffers_per_shader_stage =
            crate::binding_pack::WEBGPU_D3D12_MAX_UNIFORM_BUFFERS_PER_STAGE;
        limits.max_storage_buffers_per_shader_stage =
            crate::binding_pack::WEBGPU_D3D12_MAX_STORAGE_BUFFERS_PER_STAGE;
        limits.max_storage_textures_per_shader_stage =
            crate::binding_pack::WEBGPU_D3D12_MAX_STORAGE_TEXTURES_PER_STAGE;
        limits.max_sampled_textures_per_shader_stage =
            crate::binding_pack::WEBGPU_D3D12_MAX_SAMPLED_TEXTURES_PER_STAGE;
        limits.max_samplers_per_shader_stage =
            crate::binding_pack::WEBGPU_D3D12_MAX_SAMPLERS_PER_STAGE;
        limits
    }
}

impl Default for WebGpuLimits {
    fn default() -> Self {
        Self::webgpu_baseline()
    }
}

/// Target browser WebGPU adapter/device capability profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebGpuDeviceProfile {
    pub enabled_features: BTreeSet<WebGpuFeature>,
    pub limits: WebGpuLimits,
}

impl WebGpuDeviceProfile {
    /// Strict WebGPU core baseline (no optional features, spec default limits).
    pub fn baseline() -> Self {
        Self {
            enabled_features: BTreeSet::new(),
            limits: WebGpuLimits::webgpu_baseline(),
        }
    }

    /// Standard desktop WebGPU profile targeted by the 3D Main Menu and in-game renderer
    /// (`texture-compression-bc` + `depth32float-stencil8`, spec default limits).
    pub fn main_menu_target() -> Self {
        Self::with_features(
            [
                WebGpuFeature::TextureCompressionBc,
                WebGpuFeature::Depth32FloatStencil8,
            ],
            WebGpuLimits::webgpu_baseline(),
        )
    }

    /// Build a custom device profile with explicit enabled features and limits.
    pub fn with_features<I>(features: I, limits: WebGpuLimits) -> Self
    where
        I: IntoIterator<Item = WebGpuFeature>,
    {
        Self {
            enabled_features: features.into_iter().collect(),
            limits,
        }
    }
}

impl Default for WebGpuDeviceProfile {
    fn default() -> Self {
        Self::baseline()
    }
}

/// Aggregated WebGPU feature and limit requirements for a shader or pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShaderCompatRequirements {
    pub required_features: BTreeSet<WebGpuFeature>,
    pub bind_group_indices: BTreeSet<u32>,
    pub bindings_per_group: BTreeMap<u32, usize>,
    pub dynamic_uniform_buffers: usize,
    pub dynamic_storage_buffers: usize,
    pub storage_buffers_per_stage: usize,
    pub uniform_buffers_per_stage: usize,
    pub sampled_textures_per_stage: usize,
    pub samplers_per_stage: usize,
    pub storage_textures_per_stage: usize,
    pub compute_workgroup_size: Option<[u32; 3]>,
    pub compute_workgroup_storage_bytes: u32,
}

impl ShaderCompatRequirements {
    /// Extract WebGPU feature and resource limit requirements from a [`naga::Module`].
    pub fn from_naga_module(module: &naga::Module) -> Self {
        let mut req = Self {
            required_features: WebGpuFeature::required_for_naga_module(module),
            compute_workgroup_storage_bytes: compute_workgroup_storage_bytes(module),
            ..Self::default()
        };

        let mut module_counts = StageResourceCounts::default();
        for (_, var) in module.global_variables.iter() {
            tally_global_resource(module, var, &mut module_counts);
            if let Some(binding) = &var.binding {
                req.bind_group_indices.insert(binding.group);
                let count = req.bindings_per_group.entry(binding.group).or_insert(0);
                *count = count.saturating_add(1);
            }
        }

        let mut worst_counts = if module.entry_points.is_empty() {
            module_counts
        } else {
            StageResourceCounts::default()
        };

        for ep in &module.entry_points {
            if ep.stage == naga::ShaderStage::Compute && ep.workgroup_size_overrides.is_none() {
                req.compute_workgroup_size =
                    select_worst_workgroup_size(req.compute_workgroup_size, ep.workgroup_size);
            }

            let referenced = collect_referenced_globals(module, &ep.function);
            let mut stage_counts = StageResourceCounts::default();
            for handle in referenced {
                tally_global_resource(module, &module.global_variables[handle], &mut stage_counts);
            }
            let effective = if module.entry_points.len() == 1 {
                stage_counts.elementwise_max(&module_counts)
            } else {
                stage_counts
            };
            worst_counts = worst_counts.elementwise_max(&effective);
        }

        req.storage_buffers_per_stage = worst_counts.storage_buffers;
        req.uniform_buffers_per_stage = worst_counts.uniform_buffers;
        req.sampled_textures_per_stage = worst_counts.sampled_textures;
        req.samplers_per_stage = worst_counts.samplers;
        req.storage_textures_per_stage = worst_counts.storage_textures;

        req
    }

    /// Build requirements from WGSL source and a list of sampled/created WebGPU texture formats.
    pub fn from_wgsl_and_formats(wgsl: &str, texture_formats: &[&str]) -> Self {
        let mut req = Self {
            required_features: WebGpuFeature::required_for_wgsl_source(wgsl),
            ..Self::default()
        };
        req.add_texture_formats(texture_formats.iter().copied());
        req
    }

    /// Record required WebGPU features implied by sampled/created texture formats (e.g. BC1–BC7,
    /// `depth32float-stencil8`, ETC2, ASTC).
    pub fn add_texture_formats<'a, I>(&mut self, formats: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        for fmt in formats {
            if let Some(feature) = WebGpuFeature::required_for_texture_format(fmt) {
                self.required_features.insert(feature);
            }
        }
    }

    /// Record required WebGPU features implied by storage texture formats (e.g. `bgra8unorm`).
    pub fn add_storage_texture_formats<'a, I>(&mut self, formats: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        for fmt in formats {
            if let Some(feature) = WebGpuFeature::required_for_storage_texture_format(fmt) {
                self.required_features.insert(feature);
            }
        }
    }

    /// Record required WebGPU features implied by render/depth attachment formats.
    pub fn add_render_attachment_formats<'a, I>(&mut self, formats: I)
    where
        I: IntoIterator<Item = &'a str>,
    {
        for fmt in formats {
            if let Some(feature) = WebGpuFeature::required_for_render_attachment_format(fmt) {
                self.required_features.insert(feature);
            }
        }
    }

    /// Return the minimum `maxBindGroups` value needed (`max_group_index + 1`), or `0` if unused.
    ///
    /// Computed in `u64` so `u32::MAX` never overflows.
    pub fn required_bind_groups(&self) -> u64 {
        self.bind_group_indices
            .last()
            .map_or(0, |&max_group| u64::from(max_group) + 1)
    }

    /// Merge another requirement set into `self` using saturating/max/union operations so merging
    /// across shader stages or pipeline passes never overflows and remains deterministic.
    pub fn merge(&mut self, other: &Self) {
        for &feature in &other.required_features {
            self.required_features.insert(feature);
        }
        for &group in &other.bind_group_indices {
            self.bind_group_indices.insert(group);
        }
        for (&group, &count) in &other.bindings_per_group {
            let entry = self.bindings_per_group.entry(group).or_insert(0);
            *entry = (*entry).max(count);
        }
        self.dynamic_uniform_buffers = self
            .dynamic_uniform_buffers
            .saturating_add(other.dynamic_uniform_buffers);
        self.dynamic_storage_buffers = self
            .dynamic_storage_buffers
            .saturating_add(other.dynamic_storage_buffers);
        self.storage_buffers_per_stage = self
            .storage_buffers_per_stage
            .max(other.storage_buffers_per_stage);
        self.uniform_buffers_per_stage = self
            .uniform_buffers_per_stage
            .max(other.uniform_buffers_per_stage);
        self.sampled_textures_per_stage = self
            .sampled_textures_per_stage
            .max(other.sampled_textures_per_stage);
        self.samplers_per_stage = self.samplers_per_stage.max(other.samplers_per_stage);
        self.storage_textures_per_stage = self
            .storage_textures_per_stage
            .max(other.storage_textures_per_stage);
        if let Some(other_wg) = other.compute_workgroup_size {
            self.compute_workgroup_size =
                select_worst_workgroup_size(self.compute_workgroup_size, other_wg);
        }
        self.compute_workgroup_storage_bytes = self
            .compute_workgroup_storage_bytes
            .max(other.compute_workgroup_storage_bytes);
    }
}

fn select_worst_workgroup_size(current: Option<[u32; 3]>, candidate: [u32; 3]) -> Option<[u32; 3]> {
    let Some(cur) = current else {
        return Some(candidate);
    };
    let rank = |s: [u32; 3]| -> (u8, u128, [u32; 3]) {
        let has_zero = u8::from(s[0] == 0 || s[1] == 0 || s[2] == 0);
        let total = u128::from(s[0]) * u128::from(s[1]) * u128::from(s[2]);
        (has_zero, total, s)
    };
    if rank(candidate) > rank(cur) {
        Some(candidate)
    } else {
        Some(cur)
    }
}

/// Structured WebGPU limit violation recorded during compatibility guard evaluation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LimitViolation {
    pub kind: CompatIssueKind,
    pub limit_name: &'static str,
    pub actual: u128,
    pub limit: u128,
    pub detail: String,
}

/// Deterministic WebGPU feature and limit compatibility guard report.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WebGpuCompatReport {
    /// Required optional WebGPU features in canonical ascending order.
    pub required_features: Vec<WebGpuFeature>,
    /// Required features absent from the target [`WebGpuDeviceProfile`], in canonical order.
    pub missing_features: Vec<WebGpuFeature>,
    /// Limit violations in canonical deterministic order.
    pub limit_violations: Vec<LimitViolation>,
    /// Unified compatibility audit sorted deterministically.
    pub audit: CompatAudit,
}

impl WebGpuCompatReport {
    /// Returns `true` when all required features are enabled, no limits are exceeded, and no
    /// hard compatibility blockers exist.
    pub fn is_compatible(&self) -> bool {
        self.missing_features.is_empty()
            && self.limit_violations.is_empty()
            && !self.audit.has_blockers()
    }

    /// Canonical WebGPU feature names required by the shader/pipeline.
    pub fn required_feature_names(&self) -> Vec<&'static str> {
        self.required_features.iter().map(|f| f.as_str()).collect()
    }

    /// Canonical WebGPU feature names missing from the target device profile.
    pub fn missing_feature_names(&self) -> Vec<&'static str> {
        self.missing_features.iter().map(|f| f.as_str()).collect()
    }
}

/// Evaluate [`ShaderCompatRequirements`] against a [`WebGpuDeviceProfile`].
///
/// Guarantees:
/// - Deterministic canonical ordering of `required_features`, `missing_features`,
///   `limit_violations`, and `audit.issues`.
/// - Overflow-safe arithmetic (`u128` for workgroup products and bind-group counts).
/// - Clean handling of empty/zero inputs (empty requirements pass; zero workgroup dimensions fail).
pub fn evaluate_webgpu_compat(
    req: &ShaderCompatRequirements,
    profile: &WebGpuDeviceProfile,
) -> WebGpuCompatReport {
    let required_features: Vec<WebGpuFeature> = req.required_features.iter().copied().collect();
    let mut missing_features = Vec::new();
    let mut limit_violations = Vec::new();
    let mut audit = CompatAudit::default();

    // 1. Required WebGPU feature guards (canonical BTreeSet order).
    for &feature in &required_features {
        if !profile.enabled_features.contains(&feature) {
            missing_features.push(feature);
            let name = feature.as_str();
            audit.push_unique(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::RequiredFeatureMissing,
                message: format!(
                    "shader/pipeline requires WebGPU feature '{name}', which is not enabled on the target device profile"
                ),
                hint: Some(format!(
                    "request '{name}' in GPUAdapter.requestDevice(requiredFeatures) or enable a fallback path"
                )),
            });
        }
    }

    let limits = &profile.limits;

    // 2. Bind group index ceiling (handles empty sets, max_bind_groups == 0, and group == u32::MAX).
    if let Some(&max_group) = req.bind_group_indices.last()
        && max_group >= limits.max_bind_groups
    {
        let required_groups = u128::from(max_group) + 1;
        let max_groups = u128::from(limits.max_bind_groups);
        let detail = format!(
            "bind group index {max_group} requires {required_groups} bind groups, exceeding maxBindGroups ({max_groups})"
        );
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::BindGroupLimitExceeded,
            limit_name: "maxBindGroups",
            actual: required_groups,
            limit: max_groups,
            detail: detail.clone(),
        });
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::BindGroupLimitExceeded,
            message: detail,
            hint: Some(format!(
                "pack descriptor sets into <= {} bind groups",
                limits.max_bind_groups
            )),
        });
    }

    // 3. Per-bind-group binding count ceiling (deterministic BTreeMap group order).
    for (&group, &count) in &req.bindings_per_group {
        if count > limits.max_bindings_per_bind_group {
            let detail = format!(
                "@group({group}) uses {count} bindings, exceeding maxBindingsPerBindGroup ({})",
                limits.max_bindings_per_bind_group
            );
            limit_violations.push(LimitViolation {
                kind: CompatIssueKind::BindingsPerBindGroupLimitExceeded,
                limit_name: "maxBindingsPerBindGroup",
                actual: count as u128,
                limit: limits.max_bindings_per_bind_group as u128,
                detail: detail.clone(),
            });
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::BindingsPerBindGroupLimitExceeded,
                message: detail,
                hint: Some(format!(
                    "reduce bindings in @group({group}) to <= {}",
                    limits.max_bindings_per_bind_group
                )),
            });
        }
    }

    // 4. Pipeline-layout dynamic buffer limits.
    if req.dynamic_uniform_buffers > limits.max_dynamic_uniform_buffers_per_pipeline_layout {
        let detail = format!(
            "pipeline uses {} dynamic uniform buffers, exceeding maxDynamicUniformBuffersPerPipelineLayout ({})",
            req.dynamic_uniform_buffers, limits.max_dynamic_uniform_buffers_per_pipeline_layout
        );
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::DynamicUniformBufferLimitExceeded,
            limit_name: "maxDynamicUniformBuffersPerPipelineLayout",
            actual: req.dynamic_uniform_buffers as u128,
            limit: limits.max_dynamic_uniform_buffers_per_pipeline_layout as u128,
            detail: detail.clone(),
        });
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::DynamicUniformBufferLimitExceeded,
            message: detail,
            hint: Some(
                "coalesce dynamic uniform rings or request higher maxDynamicUniformBuffersPerPipelineLayout"
                    .to_string(),
            ),
        });
    }

    if req.dynamic_storage_buffers > limits.max_dynamic_storage_buffers_per_pipeline_layout {
        let detail = format!(
            "pipeline uses {} dynamic storage buffers, exceeding maxDynamicStorageBuffersPerPipelineLayout ({})",
            req.dynamic_storage_buffers, limits.max_dynamic_storage_buffers_per_pipeline_layout
        );
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::DynamicStorageBufferLimitExceeded,
            limit_name: "maxDynamicStorageBuffersPerPipelineLayout",
            actual: req.dynamic_storage_buffers as u128,
            limit: limits.max_dynamic_storage_buffers_per_pipeline_layout as u128,
            detail: detail.clone(),
        });
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::DynamicStorageBufferLimitExceeded,
            message: detail,
            hint: Some(
                "reduce dynamic storage buffer bindings or request higher maxDynamicStorageBuffersPerPipelineLayout"
                    .to_string(),
            ),
        });
    }

    // 5. Per-shader-stage resource limits.
    let stage_counts = StageResourceCounts {
        storage_buffers: req.storage_buffers_per_stage,
        uniform_buffers: req.uniform_buffers_per_stage,
        sampled_textures: req.sampled_textures_per_stage,
        samplers: req.samplers_per_stage,
        storage_textures: req.storage_textures_per_stage,
    };
    check_stage_resource_limits(None, &stage_counts, limits, &mut audit);

    if req.storage_buffers_per_stage > limits.max_storage_buffers_per_shader_stage {
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::StorageBufferLimitExceeded,
            limit_name: "maxStorageBuffersPerShaderStage",
            actual: req.storage_buffers_per_stage as u128,
            limit: limits.max_storage_buffers_per_shader_stage as u128,
            detail: format!(
                "stage uses {} storage buffers, exceeding maxStorageBuffersPerShaderStage ({})",
                req.storage_buffers_per_stage, limits.max_storage_buffers_per_shader_stage
            ),
        });
    }
    if req.uniform_buffers_per_stage > limits.max_uniform_buffers_per_shader_stage {
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::UniformBufferLimitExceeded,
            limit_name: "maxUniformBuffersPerShaderStage",
            actual: req.uniform_buffers_per_stage as u128,
            limit: limits.max_uniform_buffers_per_shader_stage as u128,
            detail: format!(
                "stage uses {} uniform buffers, exceeding maxUniformBuffersPerShaderStage ({})",
                req.uniform_buffers_per_stage, limits.max_uniform_buffers_per_shader_stage
            ),
        });
    }
    if req.sampled_textures_per_stage > limits.max_sampled_textures_per_shader_stage {
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::SampledTextureLimitExceeded,
            limit_name: "maxSampledTexturesPerShaderStage",
            actual: req.sampled_textures_per_stage as u128,
            limit: limits.max_sampled_textures_per_shader_stage as u128,
            detail: format!(
                "stage uses {} sampled textures, exceeding maxSampledTexturesPerShaderStage ({})",
                req.sampled_textures_per_stage, limits.max_sampled_textures_per_shader_stage
            ),
        });
    }
    if req.samplers_per_stage > limits.max_samplers_per_shader_stage {
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::SamplerLimitExceeded,
            limit_name: "maxSamplersPerShaderStage",
            actual: req.samplers_per_stage as u128,
            limit: limits.max_samplers_per_shader_stage as u128,
            detail: format!(
                "stage uses {} samplers, exceeding maxSamplersPerShaderStage ({})",
                req.samplers_per_stage, limits.max_samplers_per_shader_stage
            ),
        });
    }
    if req.storage_textures_per_stage > limits.max_storage_textures_per_shader_stage {
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::StorageTextureLimitExceeded,
            limit_name: "maxStorageTexturesPerShaderStage",
            actual: req.storage_textures_per_stage as u128,
            limit: limits.max_storage_textures_per_shader_stage as u128,
            detail: format!(
                "stage uses {} storage textures, exceeding maxStorageTexturesPerShaderStage ({})",
                req.storage_textures_per_stage, limits.max_storage_textures_per_shader_stage
            ),
        });
    }

    // 6. Compute workgroup dimensions & total invocations (overflow-safe u128 math + zero guard).
    if let Some(size @ [x, y, z]) = req.compute_workgroup_size {
        check_workgroup_size_with_limits(size, None, limits, &mut audit);
        let total = u128::from(x) * u128::from(y) * u128::from(z);

        if x == 0 || x > limits.max_compute_workgroup_size_x {
            limit_violations.push(LimitViolation {
                kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
                limit_name: "maxComputeWorkgroupSizeX",
                actual: u128::from(x),
                limit: u128::from(limits.max_compute_workgroup_size_x),
                detail: format!(
                    "workgroup_size.x = {x} violates valid range 1..={}",
                    limits.max_compute_workgroup_size_x
                ),
            });
        }
        if y == 0 || y > limits.max_compute_workgroup_size_y {
            limit_violations.push(LimitViolation {
                kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
                limit_name: "maxComputeWorkgroupSizeY",
                actual: u128::from(y),
                limit: u128::from(limits.max_compute_workgroup_size_y),
                detail: format!(
                    "workgroup_size.y = {y} violates valid range 1..={}",
                    limits.max_compute_workgroup_size_y
                ),
            });
        }
        if z == 0 || z > limits.max_compute_workgroup_size_z {
            limit_violations.push(LimitViolation {
                kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
                limit_name: "maxComputeWorkgroupSizeZ",
                actual: u128::from(z),
                limit: u128::from(limits.max_compute_workgroup_size_z),
                detail: format!(
                    "workgroup_size.z = {z} violates valid range 1..={}",
                    limits.max_compute_workgroup_size_z
                ),
            });
        }
        if total == 0 || total > u128::from(limits.max_compute_invocations_per_workgroup) {
            limit_violations.push(LimitViolation {
                kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
                limit_name: "maxComputeInvocationsPerWorkgroup",
                actual: total,
                limit: u128::from(limits.max_compute_invocations_per_workgroup),
                detail: format!(
                    "workgroup_size({x}, {y}, {z}) = {total} invocations violates valid range 1..={}",
                    limits.max_compute_invocations_per_workgroup
                ),
            });
        }
    }

    // 7. Compute workgroup storage budget.
    if req.compute_workgroup_storage_bytes > limits.max_compute_workgroup_storage_size {
        let detail = format!(
            "workgroup storage uses {} bytes, exceeding maxComputeWorkgroupStorageSize ({})",
            req.compute_workgroup_storage_bytes, limits.max_compute_workgroup_storage_size
        );
        limit_violations.push(LimitViolation {
            kind: CompatIssueKind::WorkgroupStorageLimitExceeded,
            limit_name: "maxComputeWorkgroupStorageSize",
            actual: u128::from(req.compute_workgroup_storage_bytes),
            limit: u128::from(limits.max_compute_workgroup_storage_size),
            detail: detail.clone(),
        });
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::WorkgroupStorageLimitExceeded,
            message: detail,
            hint: Some(format!(
                "reduce var<workgroup> shared memory usage to <= {} bytes",
                limits.max_compute_workgroup_storage_size
            )),
        });
    }

    limit_violations.sort();
    audit.sort_deterministic();

    WebGpuCompatReport {
        required_features,
        missing_features,
        limit_violations,
        audit,
    }
}

/// Evaluate a parsed [`naga::Module`] and associated texture formats against a
/// [`WebGpuDeviceProfile`], unifying IR structural blockers, feature guards, and limit checks.
pub fn evaluate_naga_module_compat(
    module: &naga::Module,
    texture_formats: &[&str],
    profile: &WebGpuDeviceProfile,
) -> WebGpuCompatReport {
    let mut req = ShaderCompatRequirements::from_naga_module(module);
    req.add_texture_formats(texture_formats.iter().copied());
    let mut report = evaluate_webgpu_compat(&req, profile);
    report
        .audit
        .merge(audit_naga_module_with_limits(module, &profile.limits));
    report
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
    let total_rewritten_insts: usize = records
        .iter()
        .fold(0usize, |acc, r| acc.saturating_add(r.audit.rewritten_instructions));
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
            bucket.occurrences = bucket.occurrences.saturating_add(1);
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

    let mut failing_records: Vec<&ShaderAuditRecord> = records
        .iter()
        .filter(|r| !r.lowered_to_wgsl || r.audit.has_blockers() || r.stream_error.is_some())
        .collect();
    failing_records.sort_by(|a, b| {
        a.source_path
            .cmp(&b.source_path)
            .then_with(|| a.stage.cmp(&b.stage))
            .then_with(|| a.stream_error.cmp(&b.stream_error))
            .then_with(|| a.naga_error.cmp(&b.naga_error))
    });

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
            let unique_blocker_kinds: BTreeSet<CompatIssueKind> =
                rec.audit.blockers().map(|b| b.kind).collect();
            let blocker_kinds = unique_blocker_kinds
                .into_iter()
                .map(|kind| format!("`{kind}`"))
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
// 3D Scene / Deferred / Bindless / Mesh / RT WebGPU Compatibility Auditing (S71)
// ---------------------------------------------------------------------------

/// Extended WebGPU limits profile for 3D PBR, deferred 8-MRT, compute, bindless, mesh, and RT
/// pipelines (`GPUSupportedLimits`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pipeline3dLimits {
    pub max_bind_groups: u32,
    pub max_bindings_per_bind_group: usize,
    pub max_dynamic_uniform_buffers_per_pipeline_layout: usize,
    pub max_dynamic_storage_buffers_per_pipeline_layout: usize,
    pub max_sampled_textures_per_shader_stage: usize,
    pub max_samplers_per_shader_stage: usize,
    pub max_storage_buffers_per_shader_stage: usize,
    pub max_storage_textures_per_shader_stage: usize,
    pub max_uniform_buffers_per_shader_stage: usize,
    pub max_color_attachments: u32,
    pub max_inter_stage_shader_variables: u32,
    pub max_vertex_attributes: u32,
    pub max_uniform_buffer_binding_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_storage_size: u32,
}

impl Pipeline3dLimits {
    /// Strict WebGPU spec baseline defaults (including 8 MRTs and 16 inter-stage variables).
    pub const fn webgpu_baseline() -> Self {
        Self {
            max_bind_groups: WEBGPU_MAX_BIND_GROUPS,
            max_bindings_per_bind_group: WEBGPU_MAX_BINDINGS_PER_BIND_GROUP,
            max_dynamic_uniform_buffers_per_pipeline_layout:
                WEBGPU_MAX_DYNAMIC_UNIFORM_BUFFERS_PER_PIPELINE_LAYOUT,
            max_dynamic_storage_buffers_per_pipeline_layout:
                WEBGPU_MAX_DYNAMIC_STORAGE_BUFFERS_PER_PIPELINE_LAYOUT,
            max_sampled_textures_per_shader_stage: WEBGPU_MAX_SAMPLED_TEXTURES_PER_SHADER_STAGE,
            max_samplers_per_shader_stage: WEBGPU_MAX_SAMPLERS_PER_SHADER_STAGE,
            max_storage_buffers_per_shader_stage: WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
            max_storage_textures_per_shader_stage: WEBGPU_MAX_STORAGE_TEXTURES_PER_SHADER_STAGE,
            max_uniform_buffers_per_shader_stage: WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE,
            max_color_attachments: WEBGPU_MAX_COLOR_ATTACHMENTS,
            max_inter_stage_shader_variables: WEBGPU_MAX_INTER_STAGE_SHADER_VARIABLES,
            max_vertex_attributes: WEBGPU_MAX_VERTEX_ATTRIBUTES,
            max_uniform_buffer_binding_size: WEBGPU_MAX_UNIFORM_BUFFER_BINDING_SIZE,
            max_storage_buffer_binding_size: WEBGPU_MAX_STORAGE_BUFFER_BINDING_SIZE,
            max_compute_workgroup_size_x: WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_X,
            max_compute_workgroup_size_y: WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Y,
            max_compute_workgroup_size_z: WEBGPU_MAX_COMPUTE_WORKGROUP_SIZE_Z,
            max_compute_invocations_per_workgroup: WEBGPU_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
            max_compute_workgroup_storage_size: WEBGPU_MAX_COMPUTE_WORKGROUP_STORAGE_SIZE,
        }
    }

    /// Promote a [`WebGpuLimits`] into [`Pipeline3dLimits`], using baseline defaults for 3D
    /// attachment/inter-stage/binding-size ceilings.
    pub const fn from_webgpu_limits(base: &WebGpuLimits) -> Self {
        Self {
            max_bind_groups: base.max_bind_groups,
            max_bindings_per_bind_group: base.max_bindings_per_bind_group,
            max_dynamic_uniform_buffers_per_pipeline_layout:
                base.max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout:
                base.max_dynamic_storage_buffers_per_pipeline_layout,
            max_sampled_textures_per_shader_stage: base.max_sampled_textures_per_shader_stage,
            max_samplers_per_shader_stage: base.max_samplers_per_shader_stage,
            max_storage_buffers_per_shader_stage: base.max_storage_buffers_per_shader_stage,
            max_storage_textures_per_shader_stage: base.max_storage_textures_per_shader_stage,
            max_uniform_buffers_per_shader_stage: base.max_uniform_buffers_per_shader_stage,
            max_color_attachments: WEBGPU_MAX_COLOR_ATTACHMENTS,
            max_inter_stage_shader_variables: WEBGPU_MAX_INTER_STAGE_SHADER_VARIABLES,
            max_vertex_attributes: WEBGPU_MAX_VERTEX_ATTRIBUTES,
            max_uniform_buffer_binding_size: WEBGPU_MAX_UNIFORM_BUFFER_BINDING_SIZE,
            max_storage_buffer_binding_size: WEBGPU_MAX_STORAGE_BUFFER_BINDING_SIZE,
            max_compute_workgroup_size_x: base.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: base.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: base.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: base.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: base.max_compute_workgroup_storage_size,
        }
    }

    /// Convert core fields back into [`WebGpuLimits`].
    pub const fn to_webgpu_limits(&self) -> WebGpuLimits {
        WebGpuLimits {
            max_bind_groups: self.max_bind_groups,
            max_bindings_per_bind_group: self.max_bindings_per_bind_group,
            max_dynamic_uniform_buffers_per_pipeline_layout:
                self.max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout:
                self.max_dynamic_storage_buffers_per_pipeline_layout,
            max_storage_buffers_per_shader_stage: self.max_storage_buffers_per_shader_stage,
            max_uniform_buffers_per_shader_stage: self.max_uniform_buffers_per_shader_stage,
            max_sampled_textures_per_shader_stage: self.max_sampled_textures_per_shader_stage,
            max_samplers_per_shader_stage: self.max_samplers_per_shader_stage,
            max_storage_textures_per_shader_stage: self.max_storage_textures_per_shader_stage,
            max_compute_workgroup_size_x: self.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: self.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: self.max_compute_workgroup_size_z,
            max_compute_invocations_per_workgroup: self.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: self.max_compute_workgroup_storage_size,
        }
    }
}

impl Default for Pipeline3dLimits {
    fn default() -> Self {
        Self::webgpu_baseline()
    }
}

/// Target browser/runtime WebGPU profile for 3D scene pre-flight compatibility checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pipeline3dProfile {
    pub enabled_features: BTreeSet<WebGpuFeature>,
    pub limits: Pipeline3dLimits,
    /// Whether the target WebGPU runtime supports non-uniform descriptor indexing / binding arrays.
    pub supports_non_uniform_indexing: bool,
    /// Whether the runtime provides the group-3 RT BVH arena prelude (G80 / S65).
    pub supports_rt_arena: bool,
}

impl Pipeline3dProfile {
    /// Strict WebGPU core baseline (no optional features, spec default limits).
    pub fn baseline() -> Self {
        Self {
            enabled_features: BTreeSet::new(),
            limits: Pipeline3dLimits::webgpu_baseline(),
            supports_non_uniform_indexing: false,
            supports_rt_arena: false,
        }
    }

    /// Desktop 3D Main Menu / PBR deferred target profile (`texture-compression-bc`,
    /// `depth32float-stencil8`, `bgra8unorm-storage`, `float32-filterable`,
    /// `rg11b10ufloat-renderable`, `shader-f16`, `subgroups`, RT arena enabled).
    pub fn desktop_3d_target() -> Self {
        let mut enabled_features = BTreeSet::new();
        for f in [
            WebGpuFeature::Bgra8UnormStorage,
            WebGpuFeature::Depth32FloatStencil8,
            WebGpuFeature::Float32Filterable,
            WebGpuFeature::Rg11b10UfloatRenderable,
            WebGpuFeature::ShaderF16,
            WebGpuFeature::Subgroups,
            WebGpuFeature::TextureCompressionBc,
        ] {
            enabled_features.insert(f);
        }
        Self {
            enabled_features,
            limits: Pipeline3dLimits::webgpu_baseline(),
            supports_non_uniform_indexing: false,
            supports_rt_arena: true,
        }
    }

    /// Full desktop 3D + bindless profile (all desktop 3D features plus `binding-array` and
    /// non-uniform indexing).
    pub fn desktop_3d_bindless_target() -> Self {
        let mut profile = Self::desktop_3d_target();
        profile.enabled_features.insert(WebGpuFeature::BindingArray);
        profile.supports_non_uniform_indexing = true;
        profile
    }

    /// Lift an existing [`WebGpuDeviceProfile`] into [`Pipeline3dProfile`].
    pub fn from_device_profile(profile: &WebGpuDeviceProfile) -> Self {
        let supports_binding_array = profile
            .enabled_features
            .contains(&WebGpuFeature::BindingArray);
        Self {
            enabled_features: profile.enabled_features.clone(),
            limits: Pipeline3dLimits::from_webgpu_limits(&profile.limits),
            supports_non_uniform_indexing: supports_binding_array,
            supports_rt_arena: true,
        }
    }
}

impl Default for Pipeline3dProfile {
    fn default() -> Self {
        Self::baseline()
    }
}

/// Per-stage resource counts, interface metrics, and feature requirements for a 3D pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pipeline3dStageSpec {
    pub stage: crate::convert::ShaderStage,
    pub label: String,
    pub sampled_textures: usize,
    pub samplers: usize,
    pub storage_buffers: usize,
    pub storage_textures: usize,
    pub uniform_buffers: usize,
    pub max_uniform_buffer_binding_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub vertex_attributes: u32,
    pub inter_stage_variables: u32,
    pub color_attachments: u32,
    pub compute_workgroup_size: Option<[u32; 3]>,
    pub compute_workgroup_storage_bytes: u32,
    pub uses_binding_array: bool,
    pub uses_non_uniform_indexing: bool,
    pub uses_rt_arena: bool,
    pub uses_mesh_compute: bool,
    pub required_features: BTreeSet<WebGpuFeature>,
}

impl Pipeline3dStageSpec {
    pub fn new(stage: crate::convert::ShaderStage, label: impl Into<String>) -> Self {
        Self {
            stage,
            label: label.into(),
            sampled_textures: 0,
            samplers: 0,
            storage_buffers: 0,
            storage_textures: 0,
            uniform_buffers: 0,
            max_uniform_buffer_binding_size: 0,
            max_storage_buffer_binding_size: 0,
            vertex_attributes: 0,
            inter_stage_variables: 0,
            color_attachments: 0,
            compute_workgroup_size: None,
            compute_workgroup_storage_bytes: 0,
            uses_binding_array: false,
            uses_non_uniform_indexing: false,
            uses_rt_arena: false,
            uses_mesh_compute: false,
            required_features: BTreeSet::new(),
        }
    }

    /// Scan WGSL source and attach any required optional WebGPU features (`shader-f16`,
    /// `subgroups`, `bgra8unorm-storage`, `binding-array`, `clip-distances`, `dual-source-blending`).
    pub fn add_wgsl_source(&mut self, wgsl: &str) {
        let feats = WebGpuFeature::required_for_wgsl_source(wgsl);
        if feats.contains(&WebGpuFeature::BindingArray) {
            self.uses_binding_array = true;
        }
        self.required_features.extend(feats);
    }

    /// Record texture, storage-texture, render-attachment, and filterable-float format features.
    pub fn add_formats(
        &mut self,
        sampled_formats: &[&str],
        storage_formats: &[&str],
        render_attachment_formats: &[&str],
        filterable_float_formats: &[&str],
    ) {
        for &fmt in sampled_formats {
            if let Some(f) = WebGpuFeature::required_for_texture_format(fmt) {
                self.required_features.insert(f);
            }
        }
        for &fmt in storage_formats {
            if let Some(f) = WebGpuFeature::required_for_storage_texture_format(fmt) {
                self.required_features.insert(f);
            }
        }
        for &fmt in render_attachment_formats {
            if let Some(f) = WebGpuFeature::required_for_render_attachment_format(fmt) {
                self.required_features.insert(f);
            }
        }
        for &fmt in filterable_float_formats {
            if let Some(f) = WebGpuFeature::required_for_filterable_float_format(fmt) {
                self.required_features.insert(f);
            }
        }
    }
}

/// Pipeline-wide + per-stage input for [`evaluate_pipeline_3d_compat`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pipeline3dCompatInput {
    pub pipeline_name: String,
    pub bind_group_indices: BTreeSet<u32>,
    pub bindings_per_group: BTreeMap<u32, usize>,
    pub dynamic_uniform_buffers: usize,
    pub dynamic_storage_buffers: usize,
    pub stages: Vec<Pipeline3dStageSpec>,
}

impl Pipeline3dCompatInput {
    /// Construct a [`Pipeline3dCompatInput`] directly from a merged [`crate::reflect::PipelineLayoutDesc`]
    /// and its stage [`crate::reflect::EntryPointReflection`] entries.
    pub fn from_reflections(
        pipeline_name: impl Into<String>,
        layout: &crate::reflect::PipelineLayoutDesc,
        stages: &[&crate::reflect::EntryPointReflection],
    ) -> Self {
        use crate::reflect::{BindingLayout, BufferBindingType, stage_visibility};

        let mut input = Self {
            pipeline_name: pipeline_name.into(),
            ..Self::default()
        };

        for (g_idx, bgl) in layout.bind_group_layouts.iter().enumerate() {
            if !bgl.entries.is_empty() {
                input.bind_group_indices.insert(g_idx as u32);
                input.bindings_per_group.insert(g_idx as u32, bgl.entries.len());
            }
            for e in &bgl.entries {
                if let BindingLayout::Buffer {
                    ty,
                    has_dynamic_offset: true,
                    ..
                } = e.layout
                {
                    if ty == BufferBindingType::Uniform {
                        input.dynamic_uniform_buffers += 1;
                    } else {
                        input.dynamic_storage_buffers += 1;
                    }
                }
            }
        }

        for ep in stages {
            let mut spec = Pipeline3dStageSpec::new(ep.stage, &ep.name);
            let bit = stage_visibility(ep.stage);
            for bgl in &layout.bind_group_layouts {
                for e in &bgl.entries {
                    if (e.visibility & bit) == 0 {
                        continue;
                    }
                    match e.layout {
                        BindingLayout::Buffer {
                            ty: BufferBindingType::Uniform,
                            min_binding_size,
                            ..
                        } => {
                            spec.uniform_buffers += 1;
                            spec.max_uniform_buffer_binding_size =
                                spec.max_uniform_buffer_binding_size.max(min_binding_size);
                        }
                        BindingLayout::Buffer {
                            min_binding_size, ..
                        } => {
                            spec.storage_buffers += 1;
                            spec.max_storage_buffer_binding_size =
                                spec.max_storage_buffer_binding_size.max(min_binding_size);
                        }
                        BindingLayout::Sampler { .. } => spec.samplers += 1,
                        BindingLayout::Texture { .. } => spec.sampled_textures += 1,
                        BindingLayout::StorageTexture { format, .. } => {
                            spec.storage_textures += 1;
                            if let Some(f) = WebGpuFeature::required_for_storage_texture_format(format) {
                                spec.required_features.insert(f);
                            }
                        }
                    }
                }
            }
            for b in &ep.bindings {
                if b.binding_array.is_some() {
                    spec.uses_binding_array = true;
                    spec.required_features.insert(WebGpuFeature::BindingArray);
                }
            }
            spec.vertex_attributes = ep
                .vertex_inputs
                .iter()
                .map(|io| io.location.saturating_add(1))
                .max()
                .unwrap_or(0)
                .max(ep.vertex_inputs.len() as u32);
            spec.color_attachments = ep
                .fragment_outputs
                .iter()
                .map(|io| io.location.saturating_add(1))
                .max()
                .unwrap_or(0);
            spec.compute_workgroup_size = ep.workgroup_size;
            input.stages.push(spec);
        }

        input
    }
}

/// Structured limit violation emitted by [`evaluate_pipeline_3d_compat`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pipeline3dLimitViolation {
    pub stage: Option<crate::convert::ShaderStage>,
    pub kind: CompatIssueKind,
    pub limit_name: &'static str,
    pub actual: u128,
    pub limit: u128,
    pub detail: String,
}

/// Structured CI/packer pre-flight WebGPU compatibility report for a 3D pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pipeline3dCompatReport {
    pub pipeline_name: String,
    pub required_features: Vec<WebGpuFeature>,
    pub missing_features: Vec<WebGpuFeature>,
    pub limit_violations: Vec<Pipeline3dLimitViolation>,
    pub audit: CompatAudit,
    pub uses_binding_array: bool,
    pub uses_non_uniform_indexing: bool,
    pub uses_rt_arena: bool,
    pub uses_mesh_compute: bool,
}

impl Pipeline3dCompatReport {
    pub fn is_compatible(&self) -> bool {
        self.missing_features.is_empty()
            && self.limit_violations.is_empty()
            && !self.audit.has_blockers()
    }

    pub fn required_feature_names(&self) -> Vec<&'static str> {
        self.required_features.iter().map(|f| f.as_str()).collect()
    }

    pub fn missing_feature_names(&self) -> Vec<&'static str> {
        self.missing_features.iter().map(|f| f.as_str()).collect()
    }

    /// Single-line sanitized CI diagnostic summary.
    pub fn to_summary_line(&self) -> String {
        let status = if self.is_compatible() { "PASS" } else { "FAIL" };
        format!(
            "[{status}] {} (required_features={}, missing_features={}, limit_violations={}, blockers={}, warnings={})",
            self.pipeline_name,
            self.required_features.len(),
            self.missing_features.len(),
            self.limit_violations.len(),
            self.audit.blocker_count(),
            self.audit.warning_count(),
        )
    }

    /// Render a strictly sanitized deterministic JSON report suitable for CI / packer pre-flight.
    pub fn to_json(&self) -> String {
        use crate::pack::write_json_string;

        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"pipeline\": ");
        write_json_string(&mut out, &self.pipeline_name);
        out.push_str(",\n");
        let _ = writeln!(out, "  \"compatible\": {},", self.is_compatible());
        let _ = writeln!(out, "  \"uses_binding_array\": {},", self.uses_binding_array);
        let _ = writeln!(
            out,
            "  \"uses_non_uniform_indexing\": {},",
            self.uses_non_uniform_indexing
        );
        let _ = writeln!(out, "  \"uses_rt_arena\": {},", self.uses_rt_arena);
        let _ = writeln!(out, "  \"uses_mesh_compute\": {},", self.uses_mesh_compute);

        out.push_str("  \"required_features\": [");
        for (i, f) in self.required_features.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write_json_string(&mut out, f.as_str());
        }
        out.push_str("],\n");

        out.push_str("  \"missing_features\": [");
        for (i, f) in self.missing_features.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            write_json_string(&mut out, f.as_str());
        }
        out.push_str("],\n");

        out.push_str("  \"limit_violations\": [");
        for (i, v) in self.limit_violations.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let stage_json = match v.stage {
                Some(s) => format!("\"{}\"", s.as_str()),
                None => "null".to_string(),
            };
            let _ = write!(
                out,
                "{{\"stage\":{stage_json},\"kind\":\"{}\",\"limit\":\"{}\",\"actual\":{},\"max\":{},\"detail\":",
                v.kind.as_str(),
                v.limit_name,
                v.actual,
                v.limit
            );
            write_json_string(&mut out, &v.detail);
            out.push('}');
        }
        out.push_str("],\n");

        let _ = writeln!(out, "  \"blocker_count\": {},", self.audit.blocker_count());
        let _ = writeln!(out, "  \"warning_count\": {}", self.audit.warning_count());
        out.push_str("}\n");
        out
    }
}

/// Evaluate a 3D PBR / deferred / compute / bindless / mesh / RT pipeline against a
/// [`Pipeline3dProfile`], checking all per-stage & pipeline-wide WebGPU limits and optional
/// features deterministically.
pub fn evaluate_pipeline_3d_compat(
    input: &Pipeline3dCompatInput,
    profile: &Pipeline3dProfile,
) -> Pipeline3dCompatReport {
    let limits = &profile.limits;
    let mut required_set: BTreeSet<WebGpuFeature> = BTreeSet::new();
    let mut limit_violations: Vec<Pipeline3dLimitViolation> = Vec::new();
    let mut audit = CompatAudit::default();

    let mut uses_binding_array = false;
    let mut uses_non_uniform_indexing = false;
    let mut uses_rt_arena = false;
    let mut uses_mesh_compute = false;

    for st in &input.stages {
        required_set.extend(st.required_features.iter().copied());
        if st.uses_binding_array {
            uses_binding_array = true;
            required_set.insert(WebGpuFeature::BindingArray);
        }
        if st.uses_non_uniform_indexing {
            uses_non_uniform_indexing = true;
            required_set.insert(WebGpuFeature::BindingArray);
        }
        if st.uses_rt_arena {
            uses_rt_arena = true;
        }
        if st.uses_mesh_compute {
            uses_mesh_compute = true;
        }
    }

    let required_features: Vec<WebGpuFeature> = required_set.iter().copied().collect();
    let mut missing_features: Vec<WebGpuFeature> = Vec::new();
    for &feat in &required_features {
        if !profile.enabled_features.contains(&feat) {
            missing_features.push(feat);
            audit.push_unique(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::RequiredFeatureMissing,
                message: format!(
                    "pipeline '{}' requires WebGPU feature '{}', which is absent from target profile",
                    input.pipeline_name,
                    feat.as_str()
                ),
                hint: Some(format!(
                    "request '{}' in GPUAdapter.requestDevice(requiredFeatures) or enable fallback path",
                    feat.as_str()
                )),
            });
        }
    }

    if uses_non_uniform_indexing && !profile.supports_non_uniform_indexing {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::BindlessDescriptorIndexing,
            message: format!(
                "pipeline '{}' uses non-uniform bindless descriptor indexing unsupported by target profile",
                input.pipeline_name
            ),
            hint: Some(
                "enable binding-array / non-uniform indexing support or lower to bounded waterfall windows"
                    .to_string(),
            ),
        });
    }

    if uses_rt_arena && !profile.supports_rt_arena {
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::RtArenaUnsupported,
            message: format!(
                "pipeline '{}' requires runtime RT BVH arena bindings (group 3 bindings 16..21), disabled on target profile",
                input.pipeline_name
            ),
            hint: Some("enable RT BVH arena support in the WebGPU worker profile".to_string()),
        });
    }

    // Pipeline-wide bind group and dynamic buffer limits.
    if let Some(&max_group) = input.bind_group_indices.last()
        && max_group >= limits.max_bind_groups
    {
        let actual = u128::from(max_group) + 1;
        let max_g = u128::from(limits.max_bind_groups);
        let detail = format!(
            "bind group index {max_group} requires {actual} bind groups, exceeding maxBindGroups ({max_g})"
        );
        limit_violations.push(Pipeline3dLimitViolation {
            stage: None,
            kind: CompatIssueKind::BindGroupLimitExceeded,
            limit_name: "maxBindGroups",
            actual,
            limit: max_g,
            detail: detail.clone(),
        });
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Blocker,
            kind: CompatIssueKind::BindGroupLimitExceeded,
            message: detail,
            hint: Some(format!("pack bindings into <= {} bind groups", limits.max_bind_groups)),
        });
    }

    for (&group, &count) in &input.bindings_per_group {
        if count > limits.max_bindings_per_bind_group {
            let detail = format!(
                "@group({group}) uses {count} bindings, exceeding maxBindingsPerBindGroup ({})",
                limits.max_bindings_per_bind_group
            );
            limit_violations.push(Pipeline3dLimitViolation {
                stage: None,
                kind: CompatIssueKind::BindingsPerBindGroupLimitExceeded,
                limit_name: "maxBindingsPerBindGroup",
                actual: count as u128,
                limit: limits.max_bindings_per_bind_group as u128,
                detail: detail.clone(),
            });
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::BindingsPerBindGroupLimitExceeded,
                message: detail,
                hint: Some(format!(
                    "reduce bindings in @group({group}) to <= {}",
                    limits.max_bindings_per_bind_group
                )),
            });
        }
    }

    if input.dynamic_uniform_buffers > limits.max_dynamic_uniform_buffers_per_pipeline_layout {
        let detail = format!(
            "pipeline uses {} dynamic uniform buffers, exceeding maxDynamicUniformBuffersPerPipelineLayout ({})",
            input.dynamic_uniform_buffers, limits.max_dynamic_uniform_buffers_per_pipeline_layout
        );
        limit_violations.push(Pipeline3dLimitViolation {
            stage: None,
            kind: CompatIssueKind::DynamicUniformBufferLimitExceeded,
            limit_name: "maxDynamicUniformBuffersPerPipelineLayout",
            actual: input.dynamic_uniform_buffers as u128,
            limit: limits.max_dynamic_uniform_buffers_per_pipeline_layout as u128,
            detail: detail.clone(),
        });
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::DynamicUniformBufferLimitExceeded,
            message: detail,
            hint: None,
        });
    }

    if input.dynamic_storage_buffers > limits.max_dynamic_storage_buffers_per_pipeline_layout {
        let detail = format!(
            "pipeline uses {} dynamic storage buffers, exceeding maxDynamicStorageBuffersPerPipelineLayout ({})",
            input.dynamic_storage_buffers, limits.max_dynamic_storage_buffers_per_pipeline_layout
        );
        limit_violations.push(Pipeline3dLimitViolation {
            stage: None,
            kind: CompatIssueKind::DynamicStorageBufferLimitExceeded,
            limit_name: "maxDynamicStorageBuffersPerPipelineLayout",
            actual: input.dynamic_storage_buffers as u128,
            limit: limits.max_dynamic_storage_buffers_per_pipeline_layout as u128,
            detail: detail.clone(),
        });
        audit.push_once_per_kind(CompatIssue {
            severity: CompatSeverity::Warning,
            kind: CompatIssueKind::DynamicStorageBufferLimitExceeded,
            message: detail,
            hint: None,
        });
    }

    let core_limits = limits.to_webgpu_limits();

    // Per-stage resource, attachment, inter-stage, buffer-size, and compute checks.
    for st in &input.stages {
        let stage_opt = Some(st.stage);
        let stage_counts = StageResourceCounts {
            storage_buffers: st.storage_buffers,
            uniform_buffers: st.uniform_buffers,
            sampled_textures: st.sampled_textures,
            samplers: st.samplers,
            storage_textures: st.storage_textures,
        };
        check_stage_resource_limits(Some(&st.label), &stage_counts, &core_limits, &mut audit);

        if st.sampled_textures > limits.max_sampled_textures_per_shader_stage {
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::SampledTextureLimitExceeded,
                limit_name: "maxSampledTexturesPerShaderStage",
                actual: st.sampled_textures as u128,
                limit: limits.max_sampled_textures_per_shader_stage as u128,
                detail: format!(
                    "stage '{}' uses {} sampled textures (max {})",
                    st.label, st.sampled_textures, limits.max_sampled_textures_per_shader_stage
                ),
            });
        }
        if st.samplers > limits.max_samplers_per_shader_stage {
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::SamplerLimitExceeded,
                limit_name: "maxSamplersPerShaderStage",
                actual: st.samplers as u128,
                limit: limits.max_samplers_per_shader_stage as u128,
                detail: format!(
                    "stage '{}' uses {} samplers (max {})",
                    st.label, st.samplers, limits.max_samplers_per_shader_stage
                ),
            });
        }
        if st.storage_buffers > limits.max_storage_buffers_per_shader_stage {
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::StorageBufferLimitExceeded,
                limit_name: "maxStorageBuffersPerShaderStage",
                actual: st.storage_buffers as u128,
                limit: limits.max_storage_buffers_per_shader_stage as u128,
                detail: format!(
                    "stage '{}' uses {} storage buffers (max {})",
                    st.label, st.storage_buffers, limits.max_storage_buffers_per_shader_stage
                ),
            });
        }
        if st.storage_textures > limits.max_storage_textures_per_shader_stage {
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::StorageTextureLimitExceeded,
                limit_name: "maxStorageTexturesPerShaderStage",
                actual: st.storage_textures as u128,
                limit: limits.max_storage_textures_per_shader_stage as u128,
                detail: format!(
                    "stage '{}' uses {} storage textures (max {})",
                    st.label, st.storage_textures, limits.max_storage_textures_per_shader_stage
                ),
            });
        }
        if st.uniform_buffers > limits.max_uniform_buffers_per_shader_stage {
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::UniformBufferLimitExceeded,
                limit_name: "maxUniformBuffersPerShaderStage",
                actual: st.uniform_buffers as u128,
                limit: limits.max_uniform_buffers_per_shader_stage as u128,
                detail: format!(
                    "stage '{}' uses {} uniform buffers (max {})",
                    st.label, st.uniform_buffers, limits.max_uniform_buffers_per_shader_stage
                ),
            });
        }
        if st.max_uniform_buffer_binding_size > limits.max_uniform_buffer_binding_size {
            let detail = format!(
                "stage '{}' requires uniform binding size {} bytes (max {})",
                st.label,
                st.max_uniform_buffer_binding_size,
                limits.max_uniform_buffer_binding_size
            );
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::UniformBufferBindingSizeExceeded,
                limit_name: "maxUniformBufferBindingSize",
                actual: u128::from(st.max_uniform_buffer_binding_size),
                limit: u128::from(limits.max_uniform_buffer_binding_size),
                detail: detail.clone(),
            });
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::UniformBufferBindingSizeExceeded,
                message: detail,
                hint: Some("lower large constant buffers to read-only storage buffers".to_string()),
            });
        }
        if st.max_storage_buffer_binding_size > limits.max_storage_buffer_binding_size {
            let detail = format!(
                "stage '{}' requires storage binding size {} bytes (max {})",
                st.label,
                st.max_storage_buffer_binding_size,
                limits.max_storage_buffer_binding_size
            );
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::StorageBufferBindingSizeExceeded,
                limit_name: "maxStorageBufferBindingSize",
                actual: u128::from(st.max_storage_buffer_binding_size),
                limit: u128::from(limits.max_storage_buffer_binding_size),
                detail: detail.clone(),
            });
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::StorageBufferBindingSizeExceeded,
                message: detail,
                hint: None,
            });
        }
        if st.color_attachments > limits.max_color_attachments {
            let detail = format!(
                "stage '{}' writes {} color attachments (MRT), exceeding maxColorAttachments ({})",
                st.label, st.color_attachments, limits.max_color_attachments
            );
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::ColorAttachmentLimitExceeded,
                limit_name: "maxColorAttachments",
                actual: u128::from(st.color_attachments),
                limit: u128::from(limits.max_color_attachments),
                detail: detail.clone(),
            });
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::ColorAttachmentLimitExceeded,
                message: detail,
                hint: Some("pack deferred G-buffer outputs into <= 8 MRT attachments".to_string()),
            });
        }
        if st.inter_stage_variables > limits.max_inter_stage_shader_variables {
            let detail = format!(
                "stage '{}' uses {} inter-stage shader variables, exceeding maxInterStageShaderVariables ({})",
                st.label, st.inter_stage_variables, limits.max_inter_stage_shader_variables
            );
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::InterStageVariablesLimitExceeded,
                limit_name: "maxInterStageShaderVariables",
                actual: u128::from(st.inter_stage_variables),
                limit: u128::from(limits.max_inter_stage_shader_variables),
                detail: detail.clone(),
            });
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::InterStageVariablesLimitExceeded,
                message: detail,
                hint: Some("pack varyings into vec4 channels within <= 16 inter-stage locations".to_string()),
            });
        }
        if st.vertex_attributes > limits.max_vertex_attributes {
            let detail = format!(
                "stage '{}' uses {} vertex attributes, exceeding maxVertexAttributes ({})",
                st.label, st.vertex_attributes, limits.max_vertex_attributes
            );
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::VertexAttributeLimitExceeded,
                limit_name: "maxVertexAttributes",
                actual: u128::from(st.vertex_attributes),
                limit: u128::from(limits.max_vertex_attributes),
                detail: detail.clone(),
            });
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::VertexAttributeLimitExceeded,
                message: detail,
                hint: Some("fetch extra vertex attributes via storage buffers".to_string()),
            });
        }

        if let Some(wg @ [x, y, z]) = st.compute_workgroup_size {
            check_workgroup_size_with_limits(wg, Some(&st.label), &core_limits, &mut audit);
            let total = u128::from(x) * u128::from(y) * u128::from(z);
            if x == 0 || x > limits.max_compute_workgroup_size_x {
                limit_violations.push(Pipeline3dLimitViolation {
                    stage: stage_opt,
                    kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
                    limit_name: "maxComputeWorkgroupSizeX",
                    actual: u128::from(x),
                    limit: u128::from(limits.max_compute_workgroup_size_x),
                    detail: format!(
                        "stage '{}' workgroup_size.x={x} violates 1..={}",
                        st.label, limits.max_compute_workgroup_size_x
                    ),
                });
            }
            if y == 0 || y > limits.max_compute_workgroup_size_y {
                limit_violations.push(Pipeline3dLimitViolation {
                    stage: stage_opt,
                    kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
                    limit_name: "maxComputeWorkgroupSizeY",
                    actual: u128::from(y),
                    limit: u128::from(limits.max_compute_workgroup_size_y),
                    detail: format!(
                        "stage '{}' workgroup_size.y={y} violates 1..={}",
                        st.label, limits.max_compute_workgroup_size_y
                    ),
                });
            }
            if z == 0 || z > limits.max_compute_workgroup_size_z {
                limit_violations.push(Pipeline3dLimitViolation {
                    stage: stage_opt,
                    kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
                    limit_name: "maxComputeWorkgroupSizeZ",
                    actual: u128::from(z),
                    limit: u128::from(limits.max_compute_workgroup_size_z),
                    detail: format!(
                        "stage '{}' workgroup_size.z={z} violates 1..={}",
                        st.label, limits.max_compute_workgroup_size_z
                    ),
                });
            }
            if total == 0 || total > u128::from(limits.max_compute_invocations_per_workgroup) {
                limit_violations.push(Pipeline3dLimitViolation {
                    stage: stage_opt,
                    kind: CompatIssueKind::WorkgroupSizeLimitExceeded,
                    limit_name: "maxComputeInvocationsPerWorkgroup",
                    actual: total,
                    limit: u128::from(limits.max_compute_invocations_per_workgroup),
                    detail: format!(
                        "stage '{}' workgroup_size({x},{y},{z})={total} invocations violates 1..={}",
                        st.label, limits.max_compute_invocations_per_workgroup
                    ),
                });
            }
        }

        if st.compute_workgroup_storage_bytes > limits.max_compute_workgroup_storage_size {
            let detail = format!(
                "stage '{}' uses {} bytes of workgroup storage, exceeding maxComputeWorkgroupStorageSize ({})",
                st.label, st.compute_workgroup_storage_bytes, limits.max_compute_workgroup_storage_size
            );
            limit_violations.push(Pipeline3dLimitViolation {
                stage: stage_opt,
                kind: CompatIssueKind::WorkgroupStorageLimitExceeded,
                limit_name: "maxComputeWorkgroupStorageSize",
                actual: u128::from(st.compute_workgroup_storage_bytes),
                limit: u128::from(limits.max_compute_workgroup_storage_size),
                detail: detail.clone(),
            });
            audit.push_once_per_kind(CompatIssue {
                severity: CompatSeverity::Blocker,
                kind: CompatIssueKind::WorkgroupStorageLimitExceeded,
                message: detail,
                hint: Some(format!(
                    "reduce LDS/workgroup storage to <= {} bytes",
                    limits.max_compute_workgroup_storage_size
                )),
            });
        }
    }

    limit_violations.sort();
    audit.sort_deterministic();

    Pipeline3dCompatReport {
        pipeline_name: input.pipeline_name.clone(),
        required_features,
        missing_features,
        limit_violations,
        audit,
        uses_binding_array,
        uses_non_uniform_indexing,
        uses_rt_arena,
        uses_mesh_compute,
    }
}

/// Render a deterministic batch JSON compatibility pre-flight report across multiple 3D pipelines.
pub fn render_pipeline_3d_compat_batch_json(reports: &[Pipeline3dCompatReport]) -> String {
    let mut ordered: Vec<&Pipeline3dCompatReport> = reports.iter().collect();
    ordered.sort_by(|a, b| a.pipeline_name.cmp(&b.pipeline_name));

    let total = ordered.len();
    let compatible_count = ordered.iter().filter(|r| r.is_compatible()).count();
    let incompatible_count = total - compatible_count;

    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"total_pipelines\": {total},");
    let _ = writeln!(out, "  \"compatible_pipelines\": {compatible_count},");
    let _ = writeln!(out, "  \"incompatible_pipelines\": {incompatible_count},");
    if ordered.is_empty() {
        out.push_str("  \"pipelines\": []\n");
    } else {
        out.push_str("  \"pipelines\": [\n");
        for (i, rep) in ordered.iter().enumerate() {
            let indented = rep
                .to_json()
                .trim_end_matches('\n')
                .lines()
                .map(|l| format!("    {l}"))
                .collect::<Vec<_>>()
                .join("\n");
            out.push_str(&indented);
            out.push_str(if i + 1 < ordered.len() { ",\n" } else { "\n" });
        }
        out.push_str("  ]\n");
    }
    out.push_str("}\n");
    out
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

        let blocker_kinds: BTreeSet<CompatIssueKind> =
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

    // -----------------------------------------------------------------------
    // S54: WebGPU Feature-Compatibility, Limit Guards & Deterministic Ordering
    // -----------------------------------------------------------------------

    #[test]
    fn webgpu_feature_catalog_is_canonically_ordered_and_roundtrips() {
        for window in WebGpuFeature::ALL.windows(2) {
            assert!(
                window[0] < window[1],
                "WebGpuFeature enum order must be strictly ascending: {:?} vs {:?}",
                window[0],
                window[1]
            );
            assert!(
                window[0].as_str() < window[1].as_str(),
                "WebGpuFeature::as_str() order must match enum Ord: {} vs {}",
                window[0].as_str(),
                window[1].as_str()
            );
        }

        for &feature in WebGpuFeature::ALL {
            assert_eq!(WebGpuFeature::from_name(feature.as_str()), Some(feature));
            assert_eq!(
                WebGpuFeature::from_name(&format!("  {}  ", feature.as_str().to_ascii_uppercase())),
                Some(feature)
            );
        }

        assert_eq!(WebGpuFeature::from_name(""), None);
        assert_eq!(WebGpuFeature::from_name("   "), None);
        assert_eq!(WebGpuFeature::from_name("unknown-feature"), None);
    }

    #[test]
    fn texture_and_attachment_format_feature_detection_covers_bc1_to_bc7_and_depth32s8() {
        // All BC1-BC7 formats map to texture-compression-bc.
        for bc_fmt in [
            "bc1-rgba-unorm",
            "bc1-rgba-unorm-srgb",
            "bc2-rgba-unorm",
            "bc2-rgba-unorm-srgb",
            "bc3-rgba-unorm",
            "bc3-rgba-unorm-srgb",
            "bc4-r-unorm",
            "bc4-r-snorm",
            "bc5-rg-unorm",
            "bc5-rg-snorm",
            "bc6h-rgb-ufloat",
            "bc6h-rgb-float",
            "bc7-rgba-unorm",
            "BC7-RGBA-UNORM-SRGB",
        ] {
            assert_eq!(
                WebGpuFeature::required_for_texture_format(bc_fmt),
                Some(WebGpuFeature::TextureCompressionBc),
                "expected texture-compression-bc for {bc_fmt}"
            );
        }

        assert_eq!(
            WebGpuFeature::required_for_texture_format("depth32float-stencil8"),
            Some(WebGpuFeature::Depth32FloatStencil8)
        );
        assert_eq!(
            WebGpuFeature::required_for_render_attachment_format("depth32float-stencil8"),
            Some(WebGpuFeature::Depth32FloatStencil8)
        );
        assert_eq!(
            WebGpuFeature::required_for_render_attachment_format("rg11b10ufloat"),
            Some(WebGpuFeature::Rg11b10UfloatRenderable)
        );
        assert_eq!(
            WebGpuFeature::required_for_storage_texture_format("bgra8unorm"),
            Some(WebGpuFeature::Bgra8UnormStorage)
        );
        assert_eq!(
            WebGpuFeature::required_for_texture_format("etc2-rgba8unorm"),
            Some(WebGpuFeature::TextureCompressionEtc2)
        );
        assert_eq!(
            WebGpuFeature::required_for_texture_format("astc-4x4-unorm-srgb"),
            Some(WebGpuFeature::TextureCompressionAstc)
        );

        // Baseline, empty, whitespace, and malformed formats return None cleanly.
        for core_or_empty in [
            "",
            "   ",
            "rgba8unorm",
            "bgra8unorm",
            "rgba16float",
            "depth24plus",
            "depth32float",
            "astc-99x99-unorm",
        ] {
            assert_eq!(
                WebGpuFeature::required_for_texture_format(core_or_empty),
                None,
                "unexpected feature for {core_or_empty:?}"
            );
        }
    }

    #[test]
    fn wgsl_feature_scanner_detects_features_and_ignores_comments_and_empty_input() {
        assert!(WebGpuFeature::required_for_wgsl_source("").is_empty());
        assert!(WebGpuFeature::required_for_wgsl_source("   \n\t  ").is_empty());

        // Comments (including nested block comments) and lookalike identifiers must NOT trigger.
        let commented_only = r#"
            // enable f16;
            /* enable subgroups; /* nested enable clip_distances; */
               @blend_src(0) texture_storage_2d<bgra8unorm, write> */
            struct MyF16Helper { my_f16_val: f32, subgroup_emul: u32 };
            @compute @workgroup_size(8, 8, 1)
            fn main() { let x: f32 = 1.0; }
        "#;
        assert!(
            WebGpuFeature::required_for_wgsl_source(commented_only).is_empty(),
            "comments and lookalike identifiers must not produce false positives"
        );

        // Combined optional features in unsorted order return canonically sorted BTreeSet.
        let full_wgsl = r#"
            enable subgroups, f16;
            enable dual_source_blending;
            enable clip_distances;
            @group(0) @binding(0) var out_tex: texture_storage_2d<bgra8unorm, write>;
            struct FsOut {
                @location(0) @blend_src(0) c0: vec4<f16>,
                @location(0) @blend_src(1) c1: vec4<f16>,
            };
            @compute @workgroup_size(64, 1, 1)
            fn cs_main(@builtin(subgroup_invocation_id) lane: u32) {
                let b = subgroupBallot(lane == 0u);
                _ = b;
            }
        "#;
        let detected: Vec<&'static str> = WebGpuFeature::required_for_wgsl_source(full_wgsl)
            .into_iter()
            .map(|f| f.as_str())
            .collect();
        assert_eq!(
            detected,
            vec![
                "bgra8unorm-storage",
                "clip-distances",
                "dual-source-blending",
                "shader-f16",
                "subgroups",
            ]
        );
    }

    #[test]
    fn spirv_and_naga_feature_detection_handles_f16_subgroups_bgra8_and_malformed_inputs() {
        // 1. SPIR-V with Float16 + Subgroup + ClipDistance capabilities.
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let f16_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_CAPABILITY, &[CAP_GROUP_NON_UNIFORM_BALLOT]);
        b.inst(OP_CAPABILITY, &[CAP_FLOAT16]);
        b.inst(OP_CAPABILITY, &[CAP_CLIP_DISTANCE]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_GL_COMPUTE, main_fn],
            "main",
            &[],
        );
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 64, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FLOAT, &[f16_ty, 16]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);

        let words = spirv_bytes_to_words(&b.finish_bytes()).unwrap();
        let spv_features: Vec<&'static str> = WebGpuFeature::required_for_spirv_words(&words)
            .unwrap()
            .into_iter()
            .map(|f| f.as_str())
            .collect();
        assert_eq!(spv_features, vec!["clip-distances", "shader-f16", "subgroups"]);

        // Empty/malformed SPIR-V returns clean error without panicking.
        assert!(matches!(
            WebGpuFeature::required_for_spirv_words(&[]),
            Err(SpirvStreamError::TooShort { .. })
        ));

        // 2. Programmatic Naga module with f16, Bgra8Unorm storage texture, ClipDistance,
        //    dual-source blending, and SubgroupSize built-in.
        let mut module = naga::Module::default();
        let f16_handle = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar {
                    kind: naga::ScalarKind::Float,
                    width: 2,
                }),
            },
            naga::Span::UNDEFINED,
        );
        let _bgra_storage = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Image {
                    dim: naga::ImageDimension::D2,
                    arrayed: false,
                    class: naga::ImageClass::Storage {
                        format: naga::StorageFormat::Bgra8Unorm,
                        access: naga::StorageAccess::STORE,
                    },
                },
            },
            naga::Span::UNDEFINED,
        );
        let u32_ty = module.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::U32),
            },
            naga::Span::UNDEFINED,
        );
        let _struct_ty = module.types.insert(
            naga::Type {
                name: Some("Out".to_string()),
                inner: naga::TypeInner::Struct {
                    members: vec![
                        naga::StructMember {
                            name: Some("clip".to_string()),
                            ty: f16_handle,
                            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::ClipDistances)),
                            offset: 0,
                        },
                        naga::StructMember {
                            name: Some("blend1".to_string()),
                            ty: f16_handle,
                            binding: Some(naga::Binding::Location {
                                location: 0,
                                interpolation: None,
                                sampling: None,
                                blend_src: Some(1),
                                per_primitive: false,
                            }),
                            offset: 4,
                        },
                    ],
                    span: 8,
                },
            },
            naga::Span::UNDEFINED,
        );
        let mut func = naga::Function::default();
        func.arguments.push(naga::FunctionArgument {
            name: Some("sg_size".to_string()),
            ty: u32_ty,
            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::SubgroupSize)),
        });
        module.entry_points.push(naga::EntryPoint {
            name: "cs_main".to_string(),
            stage: naga::ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [64, 1, 1],
            workgroup_size_overrides: None,
            function: func,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let naga_features: Vec<&'static str> = WebGpuFeature::required_for_naga_module(&module)
            .into_iter()
            .map(|f| f.as_str())
            .collect();
        assert_eq!(
            naga_features,
            vec![
                "bgra8unorm-storage",
                "clip-distances",
                "dual-source-blending",
                "shader-f16",
                "subgroups",
            ]
        );
    }

    #[test]
    fn limit_boundary_checks_enforce_exact_webgpu_thresholds() {
        let profile = WebGpuDeviceProfile::baseline();

        // Exact-at-limit configuration MUST pass with zero violations.
        let mut at_limit = ShaderCompatRequirements::default();
        at_limit.bind_group_indices.insert(3); // group 3 => 4 bind groups (limit 4)
        at_limit
            .bindings_per_group
            .insert(0, WEBGPU_MAX_BINDINGS_PER_BIND_GROUP);
        at_limit.dynamic_uniform_buffers = WEBGPU_MAX_DYNAMIC_UNIFORM_BUFFERS_PER_PIPELINE_LAYOUT;
        at_limit.dynamic_storage_buffers = WEBGPU_MAX_DYNAMIC_STORAGE_BUFFERS_PER_PIPELINE_LAYOUT;
        at_limit.storage_buffers_per_stage = WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE;
        at_limit.uniform_buffers_per_stage = WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE;
        at_limit.sampled_textures_per_stage = WEBGPU_MAX_SAMPLED_TEXTURES_PER_SHADER_STAGE;
        at_limit.samplers_per_stage = WEBGPU_MAX_SAMPLERS_PER_SHADER_STAGE;
        at_limit.storage_textures_per_stage = WEBGPU_MAX_STORAGE_TEXTURES_PER_SHADER_STAGE;
        at_limit.compute_workgroup_size = Some([16, 16, 1]); // 256 invocations
        at_limit.compute_workgroup_storage_bytes = WEBGPU_MAX_COMPUTE_WORKGROUP_STORAGE_SIZE;

        let ok = evaluate_webgpu_compat(&at_limit, &profile);
        assert!(ok.is_compatible(), "at-limit requirements must pass: {ok:?}");
        assert!(ok.limit_violations.is_empty());
        assert!(ok.audit.issues.is_empty());

        // Single-axis boundaries [256,1,1], [1,256,1], [1,1,64] also pass.
        for valid_wg in [[256, 1, 1], [1, 256, 1], [1, 1, 64], [1, 1, 1]] {
            let req = ShaderCompatRequirements {
                compute_workgroup_size: Some(valid_wg),
                ..ShaderCompatRequirements::default()
            };
            let rep = evaluate_webgpu_compat(&req, &profile);
            assert!(rep.is_compatible(), "workgroup {valid_wg:?} must pass: {rep:?}");
        }

        // Limit + 1 for every single limit category MUST trigger a structured LimitViolation.
        let mut over = at_limit.clone();
        over.bind_group_indices.insert(4);
        over.bindings_per_group
            .insert(0, WEBGPU_MAX_BINDINGS_PER_BIND_GROUP + 1);
        over.dynamic_uniform_buffers += 1;
        over.dynamic_storage_buffers += 1;
        over.storage_buffers_per_stage += 1;
        over.uniform_buffers_per_stage += 1;
        over.sampled_textures_per_stage += 1;
        over.samplers_per_stage += 1;
        over.storage_textures_per_stage += 1;
        over.compute_workgroup_size = Some([257, 1, 1]);
        over.compute_workgroup_storage_bytes += 1;

        let rep = evaluate_webgpu_compat(&over, &profile);
        assert!(!rep.is_compatible());

        let violated_limits: BTreeSet<&'static str> =
            rep.limit_violations.iter().map(|v| v.limit_name).collect();
        for expected in [
            "maxBindGroups",
            "maxBindingsPerBindGroup",
            "maxDynamicUniformBuffersPerPipelineLayout",
            "maxDynamicStorageBuffersPerPipelineLayout",
            "maxStorageBuffersPerShaderStage",
            "maxUniformBuffersPerShaderStage",
            "maxSampledTexturesPerShaderStage",
            "maxSamplersPerShaderStage",
            "maxStorageTexturesPerShaderStage",
            "maxComputeWorkgroupSizeX",
            "maxComputeInvocationsPerWorkgroup",
            "maxComputeWorkgroupStorageSize",
        ] {
            assert!(
                violated_limits.contains(expected),
                "expected limit violation {expected} in {violated_limits:?}"
            );
        }

        // Z-axis boundary (65 > 64) and invocation-product boundary (16*16*2 = 512 > 256).
        let over_z = ShaderCompatRequirements {
            compute_workgroup_size: Some([1, 1, 65]),
            ..ShaderCompatRequirements::default()
        };
        let rep_z = evaluate_webgpu_compat(&over_z, &profile);
        assert!(!rep_z.is_compatible());
        assert!(
            rep_z
                .limit_violations
                .iter()
                .any(|v| v.limit_name == "maxComputeWorkgroupSizeZ" && v.actual == 65 && v.limit == 64)
        );
    }

    #[test]
    fn overflow_and_empty_or_zero_inputs_are_guarded_cleanly() {
        // 1. Empty inputs across public entry points.
        let empty_req = ShaderCompatRequirements::default();
        assert_eq!(empty_req.required_bind_groups(), 0);
        assert!(evaluate_webgpu_compat(&empty_req, &WebGpuDeviceProfile::baseline()).is_compatible());
        assert!(spirv_words_to_bytes(&[]).is_empty());
        assert!(audit_naga_module(&naga::Module::default()).issues.is_empty());
        let empty_md = render_compat_audit_report_md(&[]);
        assert!(empty_md.contains("| Total `.spv` files audited | 0 |"));
        assert!(empty_md.contains("No compatibility issues or rewrites recorded."));

        // 2. Zero-limit profile: empty bind groups pass; group 0 fails cleanly without underflow.
        let mut zero_group_profile = WebGpuDeviceProfile::baseline();
        zero_group_profile.limits.max_bind_groups = 0;
        assert!(evaluate_webgpu_compat(&empty_req, &zero_group_profile).is_compatible());
        let mut group0_req = ShaderCompatRequirements::default();
        group0_req.bind_group_indices.insert(0);
        let group0_rep = evaluate_webgpu_compat(&group0_req, &zero_group_profile);
        assert!(!group0_rep.is_compatible());
        assert_eq!(group0_rep.limit_violations[0].actual, 1);
        assert_eq!(group0_rep.limit_violations[0].limit, 0);

        // 3. Zero compute workgroup dimensions fail cleanly with explicit diagnostic.
        for zero_wg in [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]] {
            let req = ShaderCompatRequirements {
                compute_workgroup_size: Some(zero_wg),
                ..ShaderCompatRequirements::default()
            };
            let rep = evaluate_webgpu_compat(&req, &WebGpuDeviceProfile::baseline());
            assert!(!rep.is_compatible(), "zero workgroup {zero_wg:?} must fail");
            assert!(rep.audit.blockers().any(|i| {
                i.kind == CompatIssueKind::WorkgroupSizeLimitExceeded
                    && i.message.contains("zero dimension")
            }));
        }

        // 4. Extreme u32::MAX values never overflow u32/u64 multiplication or +1 arithmetic.
        let mut extreme = ShaderCompatRequirements::default();
        extreme.bind_group_indices.insert(u32::MAX);
        extreme.compute_workgroup_size = Some([u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(extreme.required_bind_groups(), u64::from(u32::MAX) + 1);

        let extreme_rep = evaluate_webgpu_compat(&extreme, &WebGpuDeviceProfile::baseline());
        assert!(!extreme_rep.is_compatible());
        let expected_inv = u128::from(u32::MAX) * u128::from(u32::MAX) * u128::from(u32::MAX);
        assert!(extreme_rep.limit_violations.iter().any(|v| {
            v.limit_name == "maxComputeInvocationsPerWorkgroup" && v.actual == expected_inv
        }));
        assert!(extreme_rep.limit_violations.iter().any(|v| {
            v.limit_name == "maxBindGroups" && v.actual == u128::from(u32::MAX) + 1
        }));

        // 5. Saturating merge for usize::MAX counters in ShaderCompatRequirements and CompatAudit.
        let mut req_a = ShaderCompatRequirements {
            dynamic_uniform_buffers: usize::MAX,
            dynamic_storage_buffers: usize::MAX,
            ..ShaderCompatRequirements::default()
        };
        let req_b = ShaderCompatRequirements {
            dynamic_uniform_buffers: 10,
            dynamic_storage_buffers: 10,
            ..ShaderCompatRequirements::default()
        };
        req_a.merge(&req_b);
        assert_eq!(req_a.dynamic_uniform_buffers, usize::MAX);
        assert_eq!(req_a.dynamic_storage_buffers, usize::MAX);

        let mut audit_a = CompatAudit {
            rewritten_instructions: usize::MAX,
            ..CompatAudit::default()
        };
        let audit_b = CompatAudit {
            rewritten_instructions: 42,
            ..CompatAudit::default()
        };
        audit_a.merge(audit_b);
        assert_eq!(audit_a.rewritten_instructions, usize::MAX);
    }

    #[test]
    fn deterministic_diagnostic_and_feature_ordering_across_input_permutations() {
        // 1. Main Menu target satisfies BC + depth32float-stencil8, while missing shader-f16 & subgroups
        //    are reported in identical canonical order regardless of format/feature insertion order.
        let wgsl = "enable subgroups, f16; @compute @workgroup_size(64, 1, 1) fn main() {}";
        let req_order_1 = ShaderCompatRequirements::from_wgsl_and_formats(
            wgsl,
            &["bc7-rgba-unorm", "depth32float-stencil8", "bc1-rgba-unorm"],
        );
        let req_order_2 = ShaderCompatRequirements::from_wgsl_and_formats(
            wgsl,
            &["depth32float-stencil8", "bc1-rgba-unorm", "bc7-rgba-unorm"],
        );

        let menu_profile = WebGpuDeviceProfile::main_menu_target();
        let rep_1 = evaluate_webgpu_compat(&req_order_1, &menu_profile);
        let rep_2 = evaluate_webgpu_compat(&req_order_2, &menu_profile);
        assert_eq!(rep_1, rep_2);
        assert_eq!(
            rep_1.required_feature_names(),
            vec![
                "depth32float-stencil8",
                "shader-f16",
                "subgroups",
                "texture-compression-bc",
            ]
        );
        assert_eq!(rep_1.missing_feature_names(), vec!["shader-f16", "subgroups"]);

        // With all four features enabled, the report becomes fully compatible.
        let full_profile = WebGpuDeviceProfile::with_features(
            [
                WebGpuFeature::TextureCompressionBc,
                WebGpuFeature::Depth32FloatStencil8,
                WebGpuFeature::ShaderF16,
                WebGpuFeature::Subgroups,
            ],
            WebGpuLimits::webgpu_baseline(),
        );
        assert!(evaluate_webgpu_compat(&req_order_1, &full_profile).is_compatible());

        // 2. Permuting SPIR-V capability declaration order yields identical CompatAudit issue order.
        let build_caps = |caps: &[u32]| -> Vec<u32> {
            let mut b = SpirvBuilder::new();
            let void_ty = b.id();
            let fn_ty = b.id();
            let main_fn = b.id();
            let entry_label = b.id();
            b.inst(OP_CAPABILITY, &[CAP_SHADER]);
            for &cap in caps {
                b.inst(OP_CAPABILITY, &[cap]);
            }
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
            b.inst(OP_RETURN, &[]);
            b.inst(OP_FUNCTION_END, &[]);
            spirv_bytes_to_words(&b.finish_bytes()).unwrap()
        };

        let spv_a = build_caps(&[CAP_TESSELLATION, CAP_INT64, CAP_FLOAT64, CAP_GEOMETRY]);
        let spv_b = build_caps(&[CAP_GEOMETRY, CAP_FLOAT64, CAP_TESSELLATION, CAP_INT64]);
        let audit_a = sanitize_spirv_words(&spv_a, &SanitizeOptions::default())
            .unwrap()
            .audit;
        let audit_b = sanitize_spirv_words(&spv_b, &SanitizeOptions::default())
            .unwrap()
            .audit;
        assert_eq!(audit_a.issues, audit_b.issues);

        // 3. Permuting ShaderAuditRecord order yields byte-for-byte identical Markdown reports.
        let rec_z = ShaderAuditRecord {
            source_path: "z_shader.spv".to_string(),
            stage: Some("compute".to_string()),
            lowered_to_wgsl: false,
            stream_error: None,
            naga_error: Some("err_z".to_string()),
            audit: audit_a.clone(),
        };
        let rec_a = ShaderAuditRecord {
            source_path: "a_shader.spv".to_string(),
            stage: Some("fragment".to_string()),
            lowered_to_wgsl: false,
            stream_error: None,
            naga_error: Some("err_a".to_string()),
            audit: audit_b,
        };
        let md_1 = render_compat_audit_report_md(&[rec_z.clone(), rec_a.clone()]);
        let md_2 = render_compat_audit_report_md(&[rec_a, rec_z]);
        assert_eq!(md_1, md_2);

        // 4. evaluate_naga_module_compat combines Naga structural blockers + missing features deterministically.
        let mut naga_mod = naga::Module::default();
        let _f64_ty = naga_mod.types.insert(
            naga::Type {
                name: None,
                inner: naga::TypeInner::Scalar(naga::Scalar::F64),
            },
            naga::Span::UNDEFINED,
        );
        let combined = evaluate_naga_module_compat(
            &naga_mod,
            &["bc7-rgba-unorm", "depth32float-stencil8"],
            &WebGpuDeviceProfile::baseline(),
        );
        assert!(!combined.is_compatible());
        assert_eq!(
            combined.missing_feature_names(),
            vec!["depth32float-stencil8", "texture-compression-bc"]
        );
        assert!(
            combined
                .audit
                .blockers()
                .any(|i| i.kind == CompatIssueKind::Float64)
        );
        assert_eq!(
            combined
                .audit
                .blockers()
                .filter(|i| i.kind == CompatIssueKind::RequiredFeatureMissing)
                .count(),
            2
        );
    }
}
