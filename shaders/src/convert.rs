use std::fmt;

use naga::back::wgsl;
use naga::front::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::compat::{
    CompatAudit, SanitizeOptions, ShaderAuditRecord, SpirvStreamError, audit_naga_module,
    sanitize_spirv_bytes,
};

/// SPIR-V little-endian magic number (`0x07230203`).
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Minimum SPIR-V header size in bytes (5 x 32-bit words).
pub const SPIRV_HEADER_BYTES: usize = 20;

const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// WebGPU-compatible shader pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ShaderStage {
    Vertex = 1,
    Fragment = 2,
    Compute = 3,
}

impl ShaderStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
        }
    }

    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Vertex),
            2 => Some(Self::Fragment),
            3 => Some(Self::Compute),
            _ => None,
        }
    }
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Options controlling SPIR-V compatibility pre-passes and Naga WebGPU limit auditing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertOptions {
    pub strip_float_controls: bool,
    pub strip_non_semantic: bool,
    pub audit_webgpu_limits: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            strip_float_controls: true,
            strip_non_semantic: true,
            audit_webgpu_limits: true,
        }
    }
}

impl ConvertOptions {
    /// Disable SPIR-V sanitization rewrites while keeping WebGPU limit auditing enabled.
    pub const fn without_sanitization() -> Self {
        Self {
            strip_float_controls: false,
            strip_non_semantic: false,
            audit_webgpu_limits: true,
        }
    }

    pub const fn sanitize_options(self) -> SanitizeOptions {
        SanitizeOptions {
            strip_float_controls: self.strip_float_controls,
            strip_non_semantic: self.strip_non_semantic,
        }
    }
}

/// Successful SPIR-V -> WGSL lowering output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedShader {
    pub hash: u64,
    pub stage: ShaderStage,
    pub entry_point: String,
    pub wgsl: String,
    pub spirv_bytes: usize,
    pub audit: CompatAudit,
}

/// High-level error bucket for batch blocker reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConversionErrorKind {
    InvalidSpirvHeader,
    SpirvParseError,
    ValidationError,
    WgslWriteError,
}

impl ConversionErrorKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidSpirvHeader => "InvalidSpirvHeader",
            Self::SpirvParseError => "SpirvParseError",
            Self::ValidationError => "ValidationError",
            Self::WgslWriteError => "WgslWriteError",
        }
    }
}

impl fmt::Display for ConversionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Structured shader conversion error with normalized blocker category and extracted keywords.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionError {
    pub kind: ConversionErrorKind,
    pub blocker_key: String,
    pub keywords: Vec<String>,
    pub detail: String,
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.kind, self.detail)
    }
}

impl std::error::Error for ConversionError {}

/// Compute a deterministic 64-bit FNV-1a hash over shader bytes.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A_64_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A_64_PRIME);
    }
    hash
}

/// Format a shader hash in canonical `0x` + 16 lowercase hex digits form.
pub fn format_shader_hash(hash: u64) -> String {
    format!("0x{hash:016x}")
}

/// Format the deterministic WGSL file name for a shader hash.
pub fn format_wgsl_filename(hash: u64) -> String {
    format!("{hash:016x}.wgsl")
}

/// Convert raw SPIR-V binary bytes into validated WebGPU-baseline WGSL using default browser
/// sanitization and WebGPU limit auditing options.
pub fn convert_spirv_bytes(spirv_bytes: &[u8]) -> Result<ConvertedShader, ConversionError> {
    convert_spirv_bytes_with_options(spirv_bytes, &ConvertOptions::default())
}

/// Convert raw SPIR-V binary bytes into validated WebGPU-baseline WGSL with explicit options.
pub fn convert_spirv_bytes_with_options(
    spirv_bytes: &[u8],
    options: &ConvertOptions,
) -> Result<ConvertedShader, ConversionError> {
    validate_spirv_header(spirv_bytes)?;

    let hash = fnv1a_64(spirv_bytes);
    let (sanitized_bytes, mut audit) =
        sanitize_spirv_bytes(spirv_bytes, &options.sanitize_options())
            .map_err(classify_stream_error)?;

    let spv_options = spv::Options {
        adjust_coordinate_space: true,
        strict_capabilities: true,
        block_ctx_dump_prefix: None,
    };

    let module = spv::parse_u8_slice(&sanitized_bytes, &spv_options).map_err(classify_spv_error)?;

    let (stage, entry_point) = detect_entry_point(&module)?;

    if options.audit_webgpu_limits {
        audit.merge(audit_naga_module(&module));
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let info = validator
        .validate(&module)
        .map_err(|err| classify_validation_error(&err.emit_to_string("")))?;

    let wgsl_source = wgsl::write_string(&module, &info, wgsl::WriterFlags::empty())
        .map_err(classify_wgsl_error)?;

    Ok(ConvertedShader {
        hash,
        stage,
        entry_point,
        wgsl: wgsl_source,
        spirv_bytes: spirv_bytes.len(),
        audit,
    })
}

/// Run the full SPIR-V pre-pass + Naga WebGPU limits audit + baseline lowering check for a shader.
pub fn audit_spirv_file(
    source_path: &str,
    spirv_bytes: &[u8],
    options: &ConvertOptions,
) -> ShaderAuditRecord {
    let (sanitized_bytes, mut audit) =
        match sanitize_spirv_bytes(spirv_bytes, &options.sanitize_options()) {
            Ok(pair) => pair,
            Err(err) => {
                return ShaderAuditRecord {
                    source_path: source_path.to_string(),
                    stage: None,
                    lowered_to_wgsl: false,
                    stream_error: Some(err.to_string()),
                    naga_error: None,
                    audit: CompatAudit::default(),
                };
            }
        };

    let spv_options = spv::Options {
        adjust_coordinate_space: true,
        strict_capabilities: true,
        block_ctx_dump_prefix: None,
    };

    let module = match spv::parse_u8_slice(&sanitized_bytes, &spv_options) {
        Ok(module) => module,
        Err(err) => {
            let conv_err = classify_spv_error(err);
            return ShaderAuditRecord {
                source_path: source_path.to_string(),
                stage: None,
                lowered_to_wgsl: false,
                stream_error: None,
                naga_error: Some(conv_err.to_string()),
                audit,
            };
        }
    };

    if options.audit_webgpu_limits {
        audit.merge(audit_naga_module(&module));
    }

    let stage_name = module
        .entry_points
        .first()
        .map(|ep| format!("{:?}", ep.stage).to_ascii_lowercase());

    if let Err(err) = detect_entry_point(&module) {
        return ShaderAuditRecord {
            source_path: source_path.to_string(),
            stage: stage_name,
            lowered_to_wgsl: false,
            stream_error: None,
            naga_error: Some(err.to_string()),
            audit,
        };
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let info = match validator.validate(&module) {
        Ok(info) => info,
        Err(err) => {
            let conv_err = classify_validation_error(&err.emit_to_string(""));
            return ShaderAuditRecord {
                source_path: source_path.to_string(),
                stage: stage_name,
                lowered_to_wgsl: false,
                stream_error: None,
                naga_error: Some(conv_err.to_string()),
                audit,
            };
        }
    };

    match wgsl::write_string(&module, &info, wgsl::WriterFlags::empty()) {
        Ok(_) => ShaderAuditRecord {
            source_path: source_path.to_string(),
            stage: stage_name,
            lowered_to_wgsl: true,
            stream_error: None,
            naga_error: None,
            audit,
        },
        Err(err) => {
            let conv_err = classify_wgsl_error(err);
            ShaderAuditRecord {
                source_path: source_path.to_string(),
                stage: stage_name,
                lowered_to_wgsl: false,
                stream_error: None,
                naga_error: Some(conv_err.to_string()),
                audit,
            }
        }
    }
}

fn classify_stream_error(err: SpirvStreamError) -> ConversionError {
    let detail = err.to_string();
    match err {
        SpirvStreamError::TooShort { .. } => ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: truncated".to_string(),
            keywords: vec!["TruncatedHeader".to_string()],
            detail,
        },
        SpirvStreamError::UnalignedByteLength { .. } => ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: unaligned length".to_string(),
            keywords: vec!["UnalignedWordStream".to_string()],
            detail,
        },
        SpirvStreamError::InvalidMagic { is_big_endian, .. } => {
            let keyword = if is_big_endian {
                "BigEndianSpirv"
            } else {
                "BadMagic"
            };
            ConversionError {
                kind: ConversionErrorKind::InvalidSpirvHeader,
                blocker_key: format!("InvalidSpirvHeader: {keyword}"),
                keywords: vec![keyword.to_string()],
                detail,
            }
        }
        SpirvStreamError::InvalidWordCount { .. }
        | SpirvStreamError::TruncatedInstruction { .. } => ConversionError {
            kind: ConversionErrorKind::SpirvParseError,
            blocker_key: "SpirvParseError: TruncatedOrMalformedStream".to_string(),
            keywords: vec!["IncompleteData".to_string()],
            detail,
        },
    }
}

fn validate_spirv_header(bytes: &[u8]) -> Result<(), ConversionError> {
    if bytes.len() < SPIRV_HEADER_BYTES {
        return Err(ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: truncated".to_string(),
            keywords: vec!["TruncatedHeader".to_string()],
            detail: format!(
                "SPIR-V input too short: {} bytes (minimum {SPIRV_HEADER_BYTES})",
                bytes.len()
            ),
        });
    }

    if !bytes.len().is_multiple_of(4) {
        return Err(ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: unaligned length".to_string(),
            keywords: vec!["UnalignedWordStream".to_string()],
            detail: format!(
                "SPIR-V input length {} is not a multiple of 4 bytes",
                bytes.len()
            ),
        });
    }

    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != SPIRV_MAGIC {
        let keyword = if magic == SPIRV_MAGIC.swap_bytes() {
            "BigEndianSpirv"
        } else {
            "BadMagic"
        };
        return Err(ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: format!("InvalidSpirvHeader: {keyword}"),
            keywords: vec![keyword.to_string()],
            detail: format!(
                "invalid SPIR-V magic 0x{magic:08x} (expected 0x{SPIRV_MAGIC:08x})"
            ),
        });
    }

    Ok(())
}

fn detect_entry_point(module: &naga::Module) -> Result<(ShaderStage, String), ConversionError> {
    let Some(entry) = module.entry_points.first() else {
        return Err(ConversionError {
            kind: ConversionErrorKind::ValidationError,
            blocker_key: "ValidationError: MissingEntryPoint".to_string(),
            keywords: vec!["MissingEntryPoint".to_string()],
            detail: "SPIR-V module contains no entry points".to_string(),
        });
    };

    let stage = match entry.stage {
        naga::ShaderStage::Vertex => ShaderStage::Vertex,
        naga::ShaderStage::Fragment => ShaderStage::Fragment,
        naga::ShaderStage::Compute => ShaderStage::Compute,
        other => {
            let stage_name = format!("{other:?}");
            return Err(ConversionError {
                kind: ConversionErrorKind::ValidationError,
                blocker_key: format!("UnsupportedStage({stage_name})"),
                keywords: vec![format!("Stage::{stage_name}")],
                detail: format!("shader stage {stage_name} is not supported in WebGPU"),
            });
        }
    };

    Ok((stage, entry.name.clone()))
}

fn classify_spv_error(err: spv::Error) -> ConversionError {
    let detail = err.to_string();
    match err {
        spv::Error::InvalidHeader => ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: naga rejected header".to_string(),
            keywords: vec!["InvalidHeader".to_string()],
            detail,
        },
        spv::Error::UnsupportedCapability(cap) => {
            let kw = format!("{cap:?}");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedCapability({kw})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnknownCapability(word) => {
            let kw = format!("UnknownCapability({word})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: kw.clone(),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedExtension(ext) => ConversionError {
            kind: ConversionErrorKind::SpirvParseError,
            blocker_key: format!("UnsupportedExtension({ext})"),
            keywords: vec![ext],
            detail,
        },
        spv::Error::UnsupportedExtSet(ext) => ConversionError {
            kind: ConversionErrorKind::SpirvParseError,
            blocker_key: format!("UnsupportedExtSet({ext})"),
            keywords: vec![ext],
            detail,
        },
        spv::Error::UnsupportedInstruction(_, op) => {
            let kw = format!("{op:?}");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedInstruction({kw})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedExecutionModel(model) => {
            let kw = format!("ExecutionModel({model})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedExecutionModel({model})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedExecutionMode(mode) => {
            let kw = format!("ExecutionMode({mode})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedExecutionMode({mode})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedBuiltIn(builtin) => {
            let kw = format!("BuiltIn({builtin})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedBuiltIn({builtin})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedStorageClass(sc) => {
            let kw = format!("StorageClass({sc})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedStorageClass({sc})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedImageDim(dim) => {
            let kw = format!("ImageDim({dim})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedImageDim({dim})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedImageFormat(fmt_id) => {
            let kw = format!("ImageFormat({fmt_id})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedImageFormat({fmt_id})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::IncompleteData | spv::Error::InvalidWordCount => ConversionError {
            kind: ConversionErrorKind::SpirvParseError,
            blocker_key: "SpirvParseError: TruncatedOrMalformedStream".to_string(),
            keywords: vec!["IncompleteData".to_string()],
            detail,
        },
        other => {
            let variant_name = debug_variant_name(&other);
            let mut keywords = extract_feature_keywords(&detail);
            if keywords.is_empty() {
                keywords.push(variant_name.clone());
            }
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("SpirvParseError: {variant_name}"),
                keywords,
                detail,
            }
        }
    }
}

fn classify_validation_error(message: &str) -> ConversionError {
    let trimmed = message.trim();
    let first_line = trimmed
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("validation failed")
        .trim();

    let keywords = extract_feature_keywords(trimmed);
    let blocker_key = if !keywords.is_empty() {
        format!("ValidationError: {}", keywords.join(", "))
    } else {
        format!("ValidationError: {}", truncate_summary(first_line, 80))
    };

    ConversionError {
        kind: ConversionErrorKind::ValidationError,
        blocker_key,
        keywords,
        detail: trimmed.to_string(),
    }
}

fn classify_wgsl_error(err: wgsl::Error) -> ConversionError {
    let detail = err.to_string();
    match err {
        wgsl::Error::Unsupported { kind, value } => {
            let kw = format!("{kind}:{value}");
            ConversionError {
                kind: ConversionErrorKind::WgslWriteError,
                blocker_key: format!("WgslUnsupported({kw})"),
                keywords: vec![kw],
                detail,
            }
        }
        wgsl::Error::UnsupportedRelationalFunction(func) => {
            let kw = format!("RelationalFunction::{func:?}");
            ConversionError {
                kind: ConversionErrorKind::WgslWriteError,
                blocker_key: format!("WgslUnsupported({kw})"),
                keywords: vec![kw],
                detail,
            }
        }
        other => {
            let variant_name = debug_variant_name(&other);
            let mut keywords = extract_feature_keywords(&detail);
            if keywords.is_empty() {
                keywords.push(variant_name.clone());
            }
            ConversionError {
                kind: ConversionErrorKind::WgslWriteError,
                blocker_key: format!("WgslWriteError: {variant_name}"),
                keywords,
                detail,
            }
        }
    }
}

fn debug_variant_name(value: &impl fmt::Debug) -> String {
    let dbg = format!("{value:?}");
    dbg.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("Unknown")
        .to_string()
}

fn truncate_summary(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let collected: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{collected}...")
    } else {
        collected
    }
}

/// Extract capability / extension / WebGPU blocker keywords from diagnostic text.
pub fn extract_feature_keywords(text: &str) -> Vec<String> {
    let mut found = Vec::new();

    // Extract Capabilities(...) tokens emitted by Naga validation errors.
    let mut rest = text;
    while let Some(idx) = rest.find("Capabilities(") {
        let after = &rest[idx + "Capabilities(".len()..];
        if let Some(end) = after.find(')') {
            for token in after[..end].split('|') {
                let cleaned = token.trim();
                if !cleaned.is_empty() {
                    push_unique(&mut found, format!("Capability::{cleaned}"));
                }
            }
            rest = &after[end + 1..];
        } else {
            break;
        }
    }

    // Extract SPIR-V extension identifiers (e.g. SPV_KHR_ray_tracing).
    for word in text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if word.starts_with("SPV_") && word.len() > 4 {
            push_unique(&mut found, word.to_string());
        }
    }

    // Match well-known WebGPU blocker feature terms when present in diagnostics.
    let lower = text.to_ascii_lowercase();
    const KNOWN_BLOCKERS: &[(&str, &str)] = &[
        ("push_constant", "PushConstants"),
        ("immediate", "ImmediateData"),
        ("float64", "Float64"),
        ("f64", "Float64"),
        ("int64", "Int64"),
        ("i64", "Int64"),
        ("u64", "Int64"),
        ("subgroup", "Subgroup"),
        ("ray_query", "RayQuery"),
        ("rayquery", "RayQuery"),
        ("mesh", "MeshShader"),
        ("task", "TaskShader"),
        ("binding_array", "BindingArray"),
        ("bindless", "Bindless"),
        ("dual_source", "DualSourceBlending"),
        ("clip_distance", "ClipDistances"),
        ("cull_distance", "CullDistance"),
        ("primitive_index", "PrimitiveIndex"),
        ("barycentric", "Barycentric"),
    ];

    for &(needle, label) in KNOWN_BLOCKERS {
        if lower.contains(needle) {
            push_unique(&mut found, label.to_string());
        }
    }

    found
}

fn push_unique(list: &mut Vec<String>, item: String) {
    if !list.iter().any(|existing| existing == &item) {
        list.push(item);
    }
}

#[cfg(test)]
pub(crate) mod test_spirv {
    use super::SPIRV_MAGIC;

    /// Minimal SPIR-V word stream builder for deterministic in-memory unit tests.
    pub(crate) struct SpirvBuilder {
        version: u32,
        next_id: u32,
        words: Vec<u32>,
    }

    impl SpirvBuilder {
        pub(crate) fn new() -> Self {
            Self {
                version: 0x0001_0300, // SPIR-V 1.3
                next_id: 1,
                words: Vec::new(),
            }
        }

        pub(crate) fn id(&mut self) -> u32 {
            let id = self.next_id;
            self.next_id += 1;
            id
        }

        pub(crate) fn inst(&mut self, opcode: u16, operands: &[u32]) {
            let word_count = (1 + operands.len()) as u32;
            self.words.push((word_count << 16) | u32::from(opcode));
            self.words.extend_from_slice(operands);
        }

        pub(crate) fn inst_with_string(
            &mut self,
            opcode: u16,
            prefix: &[u32],
            text: &str,
            suffix: &[u32],
        ) {
            let mut str_words = encode_spirv_string(text);
            let mut operands = Vec::with_capacity(prefix.len() + str_words.len() + suffix.len());
            operands.extend_from_slice(prefix);
            operands.append(&mut str_words);
            operands.extend_from_slice(suffix);
            self.inst(opcode, &operands);
        }

        pub(crate) fn finish_bytes(self) -> Vec<u8> {
            let mut full = Vec::with_capacity(5 + self.words.len());
            full.push(SPIRV_MAGIC);
            full.push(self.version);
            full.push(0); // generator
            full.push(self.next_id); // id bound
            full.push(0); // schema
            full.extend_from_slice(&self.words);

            let mut bytes = Vec::with_capacity(full.len() * 4);
            for w in full {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            bytes
        }
    }

    fn encode_spirv_string(s: &str) -> Vec<u32> {
        let mut raw = s.as_bytes().to_vec();
        raw.push(0); // NUL terminator
        while !raw.len().is_multiple_of(4) {
            raw.push(0);
        }
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // SPIR-V opcodes used by synthetic shader tests
    const OP_EXTENSION: u16 = 10;
    const OP_MEMORY_MODEL: u16 = 14;
    const OP_ENTRY_POINT: u16 = 15;
    const OP_EXECUTION_MODE: u16 = 16;
    const OP_CAPABILITY: u16 = 17;
    const OP_TYPE_VOID: u16 = 19;
    const OP_TYPE_FLOAT: u16 = 22;
    const OP_TYPE_VECTOR: u16 = 23;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_TYPE_FUNCTION: u16 = 33;
    const OP_CONSTANT: u16 = 43;
    const OP_CONSTANT_COMPOSITE: u16 = 44;
    const OP_FUNCTION: u16 = 54;
    const OP_FUNCTION_END: u16 = 56;
    const OP_VARIABLE: u16 = 59;
    const OP_STORE: u16 = 62;
    const OP_DECORATE: u16 = 71;
    const OP_LABEL: u16 = 248;
    const OP_RETURN: u16 = 253;

    // SPIR-V enum constants
    const CAP_SHADER: u32 = 1;
    const CAP_FLOAT64: u32 = 10;
    const CAP_RAY_QUERY_KHR: u32 = 4472;
    const ADDRESSING_LOGICAL: u32 = 0;
    const MEMORY_GLSL450: u32 = 1;
    const EXEC_MODEL_VERTEX: u32 = 0;
    const EXEC_MODEL_FRAGMENT: u32 = 4;
    const EXEC_MODEL_GL_COMPUTE: u32 = 5;
    const EXEC_MODE_ORIGIN_UPPER_LEFT: u32 = 7;
    const EXEC_MODE_LOCAL_SIZE: u32 = 17;
    const STORAGE_CLASS_OUTPUT: u32 = 3;
    const DECORATION_BUILTIN: u32 = 11;
    const DECORATION_LOCATION: u32 = 30;
    const BUILTIN_POSITION: u32 = 0;

    /// Build a valid minimal SPIR-V compute shader (`@compute @workgroup_size(1, 1, 1)`).
    pub(crate) fn build_minimal_compute_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_GL_COMPUTE, main_fn], "main", &[]);
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    /// Build a valid minimal SPIR-V vertex shader writing `@builtin(position) = vec4(0, 0, 0, 1)`.
    pub(crate) fn build_minimal_vertex_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let f32_ty = b.id();
        let vec4_ty = b.id();
        let ptr_out_vec4 = b.id();
        let pos_var = b.id();
        let c_zero = b.id();
        let c_one = b.id();
        let c_pos = b.id();
        let vs_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_VERTEX, vs_fn],
            "vs_main",
            &[pos_var],
        );
        b.inst(OP_DECORATE, &[pos_var, DECORATION_BUILTIN, BUILTIN_POSITION]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_TYPE_FLOAT, &[f32_ty, 32]);
        b.inst(OP_TYPE_VECTOR, &[vec4_ty, f32_ty, 4]);
        b.inst(OP_TYPE_POINTER, &[ptr_out_vec4, STORAGE_CLASS_OUTPUT, vec4_ty]);
        b.inst(OP_VARIABLE, &[ptr_out_vec4, pos_var, STORAGE_CLASS_OUTPUT]);
        b.inst(OP_CONSTANT, &[f32_ty, c_zero, 0.0f32.to_bits()]);
        b.inst(OP_CONSTANT, &[f32_ty, c_one, 1.0f32.to_bits()]);
        b.inst(OP_CONSTANT_COMPOSITE, &[vec4_ty, c_pos, c_zero, c_zero, c_zero, c_one]);
        b.inst(OP_FUNCTION, &[void_ty, vs_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_STORE, &[pos_var, c_pos]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    /// Build a valid minimal SPIR-V fragment shader writing `@location(0) = vec4(1, 0, 0, 1)`.
    pub(crate) fn build_minimal_fragment_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let f32_ty = b.id();
        let vec4_ty = b.id();
        let ptr_out_vec4 = b.id();
        let color_var = b.id();
        let c_zero = b.id();
        let c_one = b.id();
        let c_red = b.id();
        let fs_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_FRAGMENT, fs_fn],
            "fs_main",
            &[color_var],
        );
        b.inst(OP_EXECUTION_MODE, &[fs_fn, EXEC_MODE_ORIGIN_UPPER_LEFT]);
        b.inst(OP_DECORATE, &[color_var, DECORATION_LOCATION, 0]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_TYPE_FLOAT, &[f32_ty, 32]);
        b.inst(OP_TYPE_VECTOR, &[vec4_ty, f32_ty, 4]);
        b.inst(OP_TYPE_POINTER, &[ptr_out_vec4, STORAGE_CLASS_OUTPUT, vec4_ty]);
        b.inst(OP_VARIABLE, &[ptr_out_vec4, color_var, STORAGE_CLASS_OUTPUT]);
        b.inst(OP_CONSTANT, &[f32_ty, c_zero, 0.0f32.to_bits()]);
        b.inst(OP_CONSTANT, &[f32_ty, c_one, 1.0f32.to_bits()]);
        b.inst(OP_CONSTANT_COMPOSITE, &[vec4_ty, c_red, c_one, c_zero, c_zero, c_one]);
        b.inst(OP_FUNCTION, &[void_ty, fs_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_STORE, &[color_var, c_red]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    /// Build a SPIR-V shader declaring an unsupported extension (`SPV_KHR_ray_tracing`).
    pub(crate) fn build_unsupported_extension_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst_with_string(OP_EXTENSION, &[], "SPV_KHR_ray_tracing", &[]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_GL_COMPUTE, main_fn], "main", &[]);
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    /// Build a SPIR-V shader declaring a non-WebGPU capability (`Float64` or `RayQueryKHR`).
    pub(crate) fn build_unsupported_capability_spv(use_ray_query: bool) -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(
            OP_CAPABILITY,
            &[if use_ray_query {
                CAP_RAY_QUERY_KHR
            } else {
                CAP_FLOAT64
            }],
        );
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_GL_COMPUTE, main_fn], "main", &[]);
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::test_spirv::*;
    use super::*;

    #[test]
    fn fnv1a_64_matches_standard_test_vectors() {
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(format_shader_hash(0xdead_beef), "0x00000000deadbeef");
    }

    #[test]
    fn converts_minimal_compute_spirv_to_wgsl() {
        let spv = build_minimal_compute_spv();
        let converted = convert_spirv_bytes(&spv).expect("minimal compute SPIR-V should convert");

        assert_eq!(converted.stage, ShaderStage::Compute);
        assert_eq!(converted.entry_point, "main");
        assert_eq!(converted.hash, fnv1a_64(&spv));
        assert_eq!(converted.spirv_bytes, spv.len());
        assert!(
            converted.wgsl.contains("@compute"),
            "expected @compute in WGSL:\n{}",
            converted.wgsl
        );
        assert!(
            converted.wgsl.contains("@workgroup_size(1, 1, 1)"),
            "expected workgroup_size in WGSL:\n{}",
            converted.wgsl
        );
    }

    #[test]
    fn converts_minimal_vertex_spirv_to_wgsl() {
        let spv = build_minimal_vertex_spv();
        let converted = convert_spirv_bytes(&spv).expect("minimal vertex SPIR-V should convert");

        assert_eq!(converted.stage, ShaderStage::Vertex);
        assert_eq!(converted.entry_point, "vs_main");
        assert_eq!(converted.hash, fnv1a_64(&spv));
        assert!(
            converted.wgsl.contains("@vertex"),
            "expected @vertex in WGSL:\n{}",
            converted.wgsl
        );
        assert!(
            converted.wgsl.contains("@builtin(position)"),
            "expected @builtin(position) in WGSL:\n{}",
            converted.wgsl
        );
    }

    #[test]
    fn converts_minimal_fragment_spirv_to_wgsl() {
        let spv = build_minimal_fragment_spv();
        let converted = convert_spirv_bytes(&spv).expect("minimal fragment SPIR-V should convert");

        assert_eq!(converted.stage, ShaderStage::Fragment);
        assert_eq!(converted.entry_point, "fs_main");
        assert!(
            converted.wgsl.contains("@fragment"),
            "expected @fragment in WGSL:\n{}",
            converted.wgsl
        );
        assert!(
            converted.wgsl.contains("@location(0)"),
            "expected @location(0) in WGSL:\n{}",
            converted.wgsl
        );
    }

    #[test]
    fn rejects_invalid_and_truncated_spirv_headers() {
        let err_short = convert_spirv_bytes(&[0x03, 0x02, 0x23, 0x07]).unwrap_err();
        assert_eq!(err_short.kind, ConversionErrorKind::InvalidSpirvHeader);
        assert!(err_short.blocker_key.contains("truncated"));

        let mut bad_magic = vec![0u8; 24];
        bad_magic[0..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        let err_magic = convert_spirv_bytes(&bad_magic).unwrap_err();
        assert_eq!(err_magic.kind, ConversionErrorKind::InvalidSpirvHeader);
        assert!(err_magic.keywords.iter().any(|k| k == "BadMagic"));

        let unaligned = vec![0u8; 22];
        let err_unaligned = convert_spirv_bytes(&unaligned).unwrap_err();
        assert_eq!(err_unaligned.kind, ConversionErrorKind::InvalidSpirvHeader);
        assert!(err_unaligned.keywords.iter().any(|k| k == "UnalignedWordStream"));

        // Valid header, truncated instruction payload.
        let mut valid_compute = build_minimal_compute_spv();
        valid_compute.truncate(valid_compute.len() - 8);
        let err_trunc = convert_spirv_bytes(&valid_compute).unwrap_err();
        assert_eq!(err_trunc.kind, ConversionErrorKind::SpirvParseError);
    }

    #[test]
    fn categorizes_unsupported_extension_and_capability() {
        let ext_spv = build_unsupported_extension_spv();
        let err = convert_spirv_bytes(&ext_spv).unwrap_err();
        assert_eq!(err.kind, ConversionErrorKind::SpirvParseError);
        assert!(
            err.keywords.iter().any(|k| k == "SPV_KHR_ray_tracing"),
            "expected SPV_KHR_ray_tracing keyword, got {:?}",
            err
        );

        let cap_spv = build_unsupported_capability_spv(true);
        let cap_err = convert_spirv_bytes(&cap_spv).unwrap_err();
        assert!(
            matches!(
                cap_err.kind,
                ConversionErrorKind::SpirvParseError | ConversionErrorKind::ValidationError
            ),
            "unexpected error kind: {:?}",
            cap_err
        );
        assert!(
            !cap_err.keywords.is_empty(),
            "expected extracted capability keyword: {:?}",
            cap_err
        );
    }

    #[test]
    fn categorizes_validation_error_for_missing_entry_point_and_extracts_keywords() {
        let mut b = SpirvBuilder::new();
        b.inst(17, &[1]); // OpCapability Shader
        b.inst(14, &[0, 1]); // OpMemoryModel Logical GLSL450
        let no_entry_spv = b.finish_bytes();

        let err = convert_spirv_bytes(&no_entry_spv).unwrap_err();
        assert_eq!(err.kind, ConversionErrorKind::ValidationError);
        assert!(err.keywords.iter().any(|k| k == "MissingEntryPoint"));

        let kws = extract_feature_keywords(
            "Required Capabilities(FLOAT64 | SUBGROUP) via SPV_KHR_shader_subgroup_extended_types",
        );
        assert!(kws.iter().any(|k| k == "Capability::FLOAT64"));
        assert!(kws.iter().any(|k| k == "Capability::SUBGROUP"));
        assert!(kws
            .iter()
            .any(|k| k == "SPV_KHR_shader_subgroup_extended_types"));
    }
}
