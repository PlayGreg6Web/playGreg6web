//! Pure-Rust `DXBC`/`DXIL` `RTS0` (Root Signature) + `ISG1`/`OSG1` (Stage Signature) ->
//! S6 WebGPU `BindGroupLayout` reflection emitter.
//!
//! Directly parses DirectX shader containers without invoking DXC, SPIR-V, or Naga:
//!
//! * Reuses [`crate::dxbc`] container validation and zero-copy chunk slicing to locate
//!   `RTS0`, `ISG1`/`ISGN`, `OSG1`/`OSGN`, `PSV0`, `DXIL`/`SHDR`/`SHEX`, and `HASH` chunks.
//! * Decodes **`RTS0` Root Signature v1.0 and v1.1**: descriptor tables (`SRV`/`UAV`/`CBV`/
//!   `SAMPLER`), 32-bit root constants, root `CBV`/`SRV`/`UAV` descriptors, and static samplers.
//! * Decodes **`ISG1`/`OSG1`** (and legacy `ISGN`/`OSGN`) stage signature elements with strict
//!   bounds checks and bounded NUL-terminated semantic string reads.
//! * Optionally parses **`PSV0`** (Pipeline State Validation) for compute workgroup dimensions
//!   and resource kind refinement (`SRV` buffer vs texture, texture view dimensions).
//! * Deterministically lowers the D3D12 root signature into S6-compatible WebGPU
//!   `GPUBindGroupLayoutDescriptor` entries across `groups[0..4]` (clamping/remapping
//!   `register_space >= 4` into `0..=3` without collisions) matching [`crate::reflect`] and
//!   `web/workers/reflection_loader.js`.
//! * Emits strictly sanitized S6 reflection JSON containing only relative basenames, FNV-1a/
//!   container/shader digests, bind group layouts, and stage inputs/outputs — never host paths
//!   or raw bytecode.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::path::Path;

use crate::compat::WEBGPU_MAX_BIND_GROUPS;
use crate::convert::{ShaderStage, fnv1a_64, format_shader_hash};
use crate::dxbc::{
    DxbcChunkDetail, DxbcError, DxilShaderKind, parse_dxbc, parse_dxbc_chunks,
};
use crate::lower::WEBGPU_MAX_BINDINGS_PER_GROUP;
use crate::pack::write_json_string;
use crate::rdna2_to_wgsl::{
    DEFAULT_LANE_INDEXED_WINDOW, LaneIndexedHeapSource, WgslVertexInputInfo,
};
use crate::reflect::{
    BindGroupLayoutDesc, BindingArraySize, BindingLayout, BufferBindingType, D3dRegister,
    EntryPointReflection, GFX10_IMAGE_STRIDE_BYTES, GFX10_SAMPLER_STRIDE_BYTES,
    LAYOUT_MANIFEST_SCHEMA_VERSION, LayoutEntry, LayoutManifest, LimitViolation, LoweringKind,
    LoweringNeed, PipelineLayoutDesc, ReflectedBinding, RegisterClass, RegisterConvention,
    ResourceKind, SamplerBindingType, ShaderIo, ShaderLayoutRecord, SidecarBinding,
    SidecarBindingKind, SidecarBindingSource, SidecarStageEntry, StorageTextureAccess,
    TextureSampleType, TextureViewDimension, VISIBILITY_COMPUTE, VISIBILITY_FRAGMENT,
    VISIBILITY_VERTEX, check_pipeline_limits,
};

/// Minimum size of an `RTS0` serialized root signature header in bytes (6 x `u32`).
pub const RTS0_HEADER_BYTES: usize = 24;

/// Serialized size of one `RTS0` root parameter table entry in bytes (3 x `u32`).
const RTS0_PARAM_HEADER_BYTES: usize = 12;

/// Serialized size of one `RTS0` v1.0 descriptor range in bytes (5 x `u32`).
const RTS0_RANGE_V1_0_BYTES: usize = 20;

/// Serialized size of one `RTS0` v1.1 descriptor range in bytes (6 x `u32`).
const RTS0_RANGE_V1_1_BYTES: usize = 24;

/// Serialized size of one `RTS0` static sampler descriptor (`D3D12_STATIC_SAMPLER_DESC`, 13 x `u32`).
const RTS0_STATIC_SAMPLER_BYTES: usize = 52;

/// Minimum size of an `ISG1`/`OSG1`/`ISGN`/`OSGN` signature chunk header in bytes (2 x `u32`).
const SIG_HEADER_BYTES: usize = 8;

/// Element stride for `ISG1`/`OSG1`/`PSG1` signature chunks (8 x `u32`).
const SIG_V1_ELEMENT_BYTES: usize = 32;

/// Element stride for legacy `ISGN`/`OSGN`/`PCSG` signature chunks (6 x `u32`).
const SIG_V0_ELEMENT_BYTES: usize = 24;

/// Maximum byte length allowed for a NUL-terminated signature semantic name.
const MAX_SEMANTIC_NAME_LEN: usize = 256;

/// Minimum `DxilProgramHeader` size in bytes (6 x `u32`).
const DXIL_PROGRAM_HEADER_BYTES: usize = 24;

/// Minimum legacy `SHDR`/`SHEX` header size in bytes (version token + dword count).
const LEGACY_SHADER_HEADER_BYTES: usize = 8;

/// Minimum `PSVRuntimeInfo0` payload size in bytes (`sizeof(PSVRuntimeInfo0)`).
const PSV_MIN_RUNTIME_INFO_BYTES: usize = 24;

/// Upper bound on total descriptor ranges parsed across all tables in a single `RTS0` chunk
/// (defends against pointer-aliasing amplification from crafted inputs).
const MAX_TOTAL_DESCRIPTOR_RANGES: usize = 65_536;

/// Maximum contiguous descriptor-table range size expanded into individual WebGPU bindings.
/// Larger bounded ranges (and `u32::MAX` unbounded ranges) are reported under `requires_lowering`.
pub const MAX_EXPANDED_RANGE_DESCRIPTORS: u32 = 64;

/// `D3D12_ROOT_SIGNATURE_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS`.
pub const RTS0_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS: u32 = 0x0000_0002;
/// `D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS`.
pub const RTS0_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS: u32 = 0x0000_0020;

/// `D3D_NAME_TARGET` system value type (`SV_Target`).
const D3D_NAME_TARGET: u32 = 64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced when parsing `DXBC`/`DXIL` `RTS0`/`ISG1`/`OSG1`/`PSV0` chunks or
/// lowering them into S6 WebGPU bind group layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DxilReflectError {
    Dxbc(DxbcError),
    TruncatedChunk {
        fourcc: &'static str,
        actual: usize,
        expected_min: usize,
    },
    Overflow {
        fourcc: &'static str,
        detail: String,
    },
    OutOfBounds {
        fourcc: &'static str,
        offset: usize,
        length: usize,
        chunk_len: usize,
    },
    UnsupportedRootSignatureVersion(u32),
    InvalidRootParameterType {
        index: usize,
        raw: u32,
    },
    InvalidDescriptorRangeType {
        param_index: usize,
        range_index: usize,
        raw: u32,
    },
    InvalidShaderVisibility {
        context: &'static str,
        index: usize,
        raw: u32,
    },
    TooManyDescriptorRanges {
        count: usize,
        max: usize,
    },
    InvalidSignatureString {
        fourcc: &'static str,
        element_index: usize,
        offset: u32,
    },
    InvalidResourceBounds {
        fourcc: &'static str,
        index: usize,
        lower_bound: u32,
        upper_bound: u32,
    },
    DuplicateRegister {
        register: D3dRegister,
        detail: String,
    },
    BindingSpaceExhausted {
        group: u32,
    },
    HashCollision {
        table: &'static str,
        hash: u64,
    },
}

impl fmt::Display for DxilReflectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dxbc(err) => write!(f, "{err}"),
            Self::TruncatedChunk {
                fourcc,
                actual,
                expected_min,
            } => write!(
                f,
                "{fourcc} chunk truncated: {actual} bytes (minimum required: {expected_min})"
            ),
            Self::Overflow { fourcc, detail } => {
                write!(f, "{fourcc} offset/size overflow: {detail}")
            }
            Self::OutOfBounds {
                fourcc,
                offset,
                length,
                chunk_len,
            } => write!(
                f,
                "{fourcc} slice [{offset}..+{length}] exceeds chunk length ({chunk_len} bytes)"
            ),
            Self::UnsupportedRootSignatureVersion(v) => write!(
                f,
                "unsupported RTS0 root signature version {v} (expected 1 for v1.0 or 2 for v1.1)"
            ),
            Self::InvalidRootParameterType { index, raw } => {
                write!(f, "RTS0 parameter {index} has unknown type {raw}")
            }
            Self::InvalidDescriptorRangeType {
                param_index,
                range_index,
                raw,
            } => write!(
                f,
                "RTS0 parameter {param_index} descriptor range {range_index} has unknown type {raw}"
            ),
            Self::InvalidShaderVisibility {
                context,
                index,
                raw,
            } => write!(
                f,
                "RTS0 {context} {index} has unknown D3D12 shader visibility {raw}"
            ),
            Self::TooManyDescriptorRanges { count, max } => write!(
                f,
                "RTS0 total descriptor range count ({count}) exceeds safety limit ({max})"
            ),
            Self::InvalidSignatureString {
                fourcc,
                element_index,
                offset,
            } => write!(
                f,
                "{fourcc} element {element_index} has invalid/unterminated semantic name at offset {offset}"
            ),
            Self::InvalidResourceBounds {
                fourcc,
                index,
                lower_bound,
                upper_bound,
            } => write!(
                f,
                "{fourcc} resource binding {index} has inverted register bounds [{lower_bound}..={upper_bound}]"
            ),
            Self::DuplicateRegister { register, detail } => write!(
                f,
                "duplicate or conflicting D3D12 register {}{} space{}: {detail}",
                register.class.as_str(),
                register.index,
                register.space
            ),
            Self::BindingSpaceExhausted { group } => write!(
                f,
                "WebGPU bind group {group} has no remaining allocatable binding indices"
            ),
            Self::HashCollision { table, hash } => write!(
                f,
                "content hash collision in {table} for id {}",
                format_shader_hash(*hash)
            ),
        }
    }
}

impl std::error::Error for DxilReflectError {}

impl From<DxbcError> for DxilReflectError {
    fn from(err: DxbcError) -> Self {
        Self::Dxbc(err)
    }
}

// ---------------------------------------------------------------------------
// RTS0 Root Signature Model (v1.0 & v1.1)
// ---------------------------------------------------------------------------

/// `D3D_ROOT_SIGNATURE_VERSION`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RootSignatureVersion {
    /// `D3D_ROOT_SIGNATURE_VERSION_1_0` (`1`).
    V1_0,
    /// `D3D_ROOT_SIGNATURE_VERSION_1_1` (`2`).
    V1_1,
}

impl RootSignatureVersion {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V1_0 => "1.0",
            Self::V1_1 => "1.1",
        }
    }
}

/// `D3D12_SHADER_VISIBILITY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum D3dShaderVisibility {
    All = 0,
    Vertex = 1,
    Hull = 2,
    Domain = 3,
    Geometry = 4,
    Pixel = 5,
    Amplification = 6,
    Mesh = 7,
}

impl D3dShaderVisibility {
    pub const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::All),
            1 => Some(Self::Vertex),
            2 => Some(Self::Hull),
            3 => Some(Self::Domain),
            4 => Some(Self::Geometry),
            5 => Some(Self::Pixel),
            6 => Some(Self::Amplification),
            7 => Some(Self::Mesh),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Vertex => "vertex",
            Self::Hull => "hull",
            Self::Domain => "domain",
            Self::Geometry => "geometry",
            Self::Pixel => "pixel",
            Self::Amplification => "amplification",
            Self::Mesh => "mesh",
        }
    }

    /// Bitmask of D3D12 shader stages covered by this visibility value.
    /// Used to detect illegal overlapping register declarations within the same D3D12 stage
    /// while permitting valid disjoint cross-stage sharing (e.g. Vertex + Pixel).
    const fn d3d_stage_mask(self) -> u8 {
        match self {
            Self::Vertex => 1 << 0,
            Self::Hull => 1 << 1,
            Self::Domain => 1 << 2,
            Self::Geometry => 1 << 3,
            Self::Pixel => 1 << 4,
            Self::Amplification => 1 << 5,
            Self::Mesh => 1 << 6,
            Self::All => 0x7F,
        }
    }
}

/// `D3D12_DESCRIPTOR_RANGE_TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DescriptorRangeType {
    /// Shader Resource View (`t#`).
    Srv = 0,
    /// Unordered Access View (`u#`).
    Uav = 1,
    /// Constant Buffer View (`b#`).
    Cbv = 2,
    /// Sampler (`s#`).
    Sampler = 3,
}

impl DescriptorRangeType {
    pub const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::Srv),
            1 => Some(Self::Uav),
            2 => Some(Self::Cbv),
            3 => Some(Self::Sampler),
            _ => None,
        }
    }

    pub const fn register_class(self) -> RegisterClass {
        match self {
            Self::Srv => RegisterClass::T,
            Self::Uav => RegisterClass::U,
            Self::Cbv => RegisterClass::B,
            Self::Sampler => RegisterClass::S,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Srv => "srv",
            Self::Uav => "uav",
            Self::Cbv => "cbv",
            Self::Sampler => "sampler",
        }
    }
}

/// One descriptor range inside a `D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DescriptorRange {
    pub range_type: DescriptorRangeType,
    /// Number of descriptors (`u32::MAX` denotes an unbounded range).
    pub num_descriptors: u32,
    pub base_shader_register: u32,
    pub register_space: u32,
    /// `D3D12_DESCRIPTOR_RANGE_FLAGS` (always `0` for `RTS0` v1.0).
    pub flags: u32,
    pub offset_in_descriptors_from_table_start: u32,
}

/// Decoded payload of a single `D3D12_ROOT_PARAMETER` / `D3D12_ROOT_PARAMETER1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootParameterKind {
    DescriptorTable(Vec<DescriptorRange>),
    Constants32Bit {
        shader_register: u32,
        register_space: u32,
        num_32bit_values: u32,
    },
    Cbv {
        shader_register: u32,
        register_space: u32,
        flags: u32,
    },
    Srv {
        shader_register: u32,
        register_space: u32,
        flags: u32,
    },
    Uav {
        shader_register: u32,
        register_space: u32,
        flags: u32,
    },
}

/// One entry in a root signature's parameter table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootParameter {
    pub visibility: D3dShaderVisibility,
    pub kind: RootParameterKind,
}

/// Decoded `D3D12_STATIC_SAMPLER_DESC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticSamplerDesc {
    pub filter: u32,
    pub address_u: u32,
    pub address_v: u32,
    pub address_w: u32,
    pub mip_lod_bias_bits: u32,
    pub max_anisotropy: u32,
    pub comparison_func: u32,
    pub border_color: u32,
    pub min_lod_bits: u32,
    pub max_lod_bits: u32,
    pub shader_register: u32,
    pub register_space: u32,
    pub shader_visibility: D3dShaderVisibility,
}

impl StaticSamplerDesc {
    /// Returns `true` when `D3D12_FILTER` selects `D3D12_FILTER_REDUCTION_TYPE_COMPARISON`
    /// (`((filter >> 7) & 0x3) == 1`).
    pub const fn is_comparison(&self) -> bool {
        ((self.filter >> 7) & 0x3) == 1
    }

    pub const fn sampler_binding_type(&self) -> SamplerBindingType {
        if self.is_comparison() {
            SamplerBindingType::Comparison
        } else {
            SamplerBindingType::Filtering
        }
    }
}

/// Parsed `RTS0` serialized root signature (v1.0 or v1.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootSignature {
    pub version: RootSignatureVersion,
    pub flags: u32,
    pub parameters: Vec<RootParameter>,
    pub static_samplers: Vec<StaticSamplerDesc>,
}

// ---------------------------------------------------------------------------
// ISG1 / OSG1 Stage Signature Model
// ---------------------------------------------------------------------------

/// `D3D_REGISTER_COMPONENT_TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SignatureComponentType {
    Unknown,
    Uint32,
    Sint32,
    Float32,
    Uint16,
    Sint16,
    Float16,
    Uint64,
    Sint64,
    Float64,
    Other(u32),
}

impl SignatureComponentType {
    pub const fn from_u32(raw: u32) -> Self {
        match raw {
            0 => Self::Unknown,
            1 => Self::Uint32,
            2 => Self::Sint32,
            3 => Self::Float32,
            4 => Self::Uint16,
            5 => Self::Sint16,
            6 => Self::Float16,
            7 => Self::Uint64,
            8 => Self::Sint64,
            9 => Self::Float64,
            other => Self::Other(other),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Uint32 => "uint32",
            Self::Sint32 => "sint32",
            Self::Float32 => "float32",
            Self::Uint16 => "uint16",
            Self::Sint16 => "sint16",
            Self::Float16 => "float16",
            Self::Uint64 => "uint64",
            Self::Sint64 => "sint64",
            Self::Float64 => "float64",
            Self::Other(_) => "other",
        }
    }
}

/// One parsed element from an `ISG1`/`OSG1` (or legacy `ISGN`/`OSGN`) chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureElement {
    pub stream: u32,
    pub semantic_name: String,
    pub semantic_index: u32,
    pub system_value_type: u32,
    pub component_type: SignatureComponentType,
    pub register: u32,
    pub mask: u8,
    pub rw_mask: u8,
    pub min_precision: u32,
}

impl SignatureElement {
    /// Returns `true` if this element is a DirectX system value (`SV_*`).
    pub fn is_system_value(&self) -> bool {
        self.system_value_type != 0
            || self
                .semantic_name
                .get(..3)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("SV_"))
    }

    /// Returns `true` if this output element is a pixel shader color render target (`SV_Target*`).
    pub fn is_render_target(&self) -> bool {
        self.system_value_type == D3D_NAME_TARGET
            || self.semantic_name.eq_ignore_ascii_case("SV_TARGET")
    }

    /// Number of active 32-bit/16-bit components in the 4-bit channel mask (`0..=4`).
    pub const fn component_count(&self) -> u32 {
        (self.mask & 0x0F).count_ones()
    }

    /// Convert `(component_type, component_count)` into `(wgsl_type, Option<GPUVertexFormat>)`,
    /// matching [`crate::reflect`] and `web/workers/reflection_loader.js`.
    pub fn to_wgsl_io_type(&self) -> (String, Option<&'static str>) {
        let components = self.component_count();
        if components == 0 {
            return ("unknown".to_string(), None);
        }
        let scalar = match self.component_type {
            SignatureComponentType::Float32 => "f32",
            SignatureComponentType::Uint32 => "u32",
            SignatureComponentType::Sint32 => "i32",
            SignatureComponentType::Float16 => "f16",
            _ => return ("unknown".to_string(), None),
        };
        let wgsl_type = if components == 1 {
            scalar.to_string()
        } else {
            format!("vec{components}<{scalar}>")
        };
        let format = match (scalar, components) {
            ("f32", 1) => Some("float32"),
            ("f32", 2) => Some("float32x2"),
            ("f32", 3) => Some("float32x3"),
            ("f32", 4) => Some("float32x4"),
            ("f16", 2) => Some("float16x2"),
            ("f16", 4) => Some("float16x4"),
            ("u32", 1) => Some("uint32"),
            ("u32", 2) => Some("uint32x2"),
            ("u32", 3) => Some("uint32x3"),
            ("u32", 4) => Some("uint32x4"),
            ("i32", 1) => Some("sint32"),
            ("i32", 2) => Some("sint32x2"),
            ("i32", 3) => Some("sint32x3"),
            ("i32", 4) => Some("sint32x4"),
            _ => None,
        };
        (wgsl_type, format)
    }
}

/// Parsed stage input or output signature chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSignature {
    pub fourcc: [u8; 4],
    pub elements: Vec<SignatureElement>,
}

// ---------------------------------------------------------------------------
// PSV0 Pipeline State Validation Model (optional refinement)
// ---------------------------------------------------------------------------

/// One resource binding record parsed from `PSV0` (`PSVResourceBindInfo0`/`1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PsvResourceBinding {
    /// `PSVResourceType` (1=Sampler, 2=CBV, 3=SRVTyped, 4=SRVRaw, 5=SRVStructured,
    /// 6=UAVTyped, 7=UAVRaw, 8=UAVStructured, 9=UAVStructuredWithCounter, ...).
    pub res_type: u32,
    pub space: u32,
    pub lower_bound: u32,
    pub upper_bound: u32,
    /// `PSVResourceKind` when `PSVResourceBindInfo1` (>= 24 bytes) is present.
    pub res_kind: Option<u32>,
    pub res_flags: Option<u32>,
}

impl PsvResourceBinding {
    fn matches(&self, class: RegisterClass, index: u32, space: u32) -> bool {
        if self.space != space || index < self.lower_bound || index > self.upper_bound {
            return false;
        }
        match class {
            RegisterClass::S => self.res_type == 1,
            RegisterClass::B => self.res_type == 2,
            RegisterClass::T => matches!(self.res_type, 3 | 4 | 5 | 10),
            RegisterClass::U => matches!(self.res_type, 6 | 7 | 8 | 9 | 11),
        }
    }

    fn is_buffer_srv(&self) -> bool {
        matches!(self.res_type, 4 | 5) || matches!(self.res_kind, Some(10..=12))
    }

    fn view_dimension(&self) -> TextureViewDimension {
        match self.res_kind {
            Some(1) => TextureViewDimension::D1,
            Some(2 | 3) => TextureViewDimension::D2,
            Some(4) => TextureViewDimension::D3,
            Some(5) => TextureViewDimension::Cube,
            Some(7 | 8) => TextureViewDimension::D2Array,
            Some(9) => TextureViewDimension::CubeArray,
            _ => TextureViewDimension::D2,
        }
    }

    fn is_multisampled(&self) -> bool {
        matches!(self.res_kind, Some(3 | 8))
    }
}

/// Optional metadata decoded from a `PSV0` chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsvInfo {
    pub runtime_info_size: u32,
    pub shader_stage: Option<DxilShaderKind>,
    pub sig_input_elements: Option<u8>,
    pub sig_output_elements: Option<u8>,
    pub workgroup_size: Option<[u32; 3]>,
    pub resources: Vec<PsvResourceBinding>,
}

// ---------------------------------------------------------------------------
// Reflection & Manifest Output Model
// ---------------------------------------------------------------------------

/// Record of a D3D12 `register_space >= 4` resource remapped into WebGPU `groups[0..4]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DxilBindGroupRemap {
    pub register: D3dRegister,
    pub original_space: u32,
    pub webgpu_group: u32,
    pub webgpu_binding: u32,
}

/// Complete sanitized reflection result for a single `DXBC`/`DXIL` container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxilShaderReflection {
    /// Sanitized relative path or basename (never an absolute host path).
    pub source: Option<String>,
    /// Deterministic 64-bit FNV-1a hash of the container bytes.
    pub hash: u64,
    /// Lowercase hex of the 16-byte DXBC container digest.
    pub container_digest: String,
    /// Lowercase hex of the 16-byte `HASH` chunk digest, if present.
    pub shader_hash_digest: Option<String>,
    pub shader_kind: Option<DxilShaderKind>,
    pub shader_model: Option<(u8, u8)>,
    pub root_signature: Option<RootSignature>,
    pub input_signature: Option<StageSignature>,
    pub output_signature: Option<StageSignature>,
    pub psv: Option<PsvInfo>,
    /// S6-compatible entry point reflection matching [`crate::reflect::EntryPointReflection`].
    pub entry: EntryPointReflection,
    /// S6-compatible WebGPU pipeline layout (`0..=4` bind groups, exact per-parameter visibility).
    pub pipeline_layout: PipelineLayoutDesc,
    pub limit_violations: Vec<LimitViolation>,
    pub bind_group_remaps: Vec<DxilBindGroupRemap>,
}

impl DxilShaderReflection {
    /// Return a copy with all reflected bindings, bind group layout entries, remaps, and
    /// stage IO collections in canonical deterministic order.
    pub fn canonicalized(&self) -> Self {
        let mut out = self.clone();
        out.entry = out.entry.canonicalized();
        out.pipeline_layout = PipelineLayoutDesc {
            bind_group_layouts: out
                .pipeline_layout
                .bind_group_layouts
                .iter()
                .map(BindGroupLayoutDesc::canonicalized)
                .collect(),
        };
        out.bind_group_remaps.sort();
        out.bind_group_remaps.dedup();
        out
    }
}

/// Sanitized error record for batch directory reflection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxilReflectionErrorRecord {
    pub source: String,
    pub error: String,
}

/// Deduplicated S6-compatible reflection manifest produced from one or more `DXBC`/`DXIL` files.
///
/// The emitted JSON validates directly against `web/workers/reflection_loader.js`
/// (`parseReflectionManifest`) and uses identical FNV-1a 64 content hashes to
/// [`crate::reflect::LayoutManifest`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DxilReflectionManifest {
    pub register_convention: RegisterConvention,
    pub bind_group_layouts: BTreeMap<u64, BindGroupLayoutDesc>,
    pub pipeline_layouts: BTreeMap<u64, Vec<u64>>,
    pub shaders: BTreeMap<u64, DxilShaderReflection>,
    pub errors: Vec<DxilReflectionErrorRecord>,
}

// ---------------------------------------------------------------------------
// Low-level Safe Readers
// ---------------------------------------------------------------------------

fn checked_add_offset(
    fourcc: &'static str,
    base: usize,
    delta: usize,
    detail: &str,
) -> Result<usize, DxilReflectError> {
    base.checked_add(delta)
        .ok_or_else(|| DxilReflectError::Overflow {
            fourcc,
            detail: detail.to_string(),
        })
}

fn checked_mul_size(
    fourcc: &'static str,
    count: usize,
    stride: usize,
    detail: &str,
) -> Result<usize, DxilReflectError> {
    count
        .checked_mul(stride)
        .ok_or_else(|| DxilReflectError::Overflow {
            fourcc,
            detail: detail.to_string(),
        })
}

fn read_u32_checked(
    fourcc: &'static str,
    bytes: &[u8],
    offset: usize,
) -> Result<u32, DxilReflectError> {
    let end = checked_add_offset(fourcc, offset, 4, "u32 read offset overflow")?;
    let slice = bytes.get(offset..end).ok_or(DxilReflectError::OutOfBounds {
        fourcc,
        offset,
        length: 4,
        chunk_len: bytes.len(),
    })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u8_checked(
    fourcc: &'static str,
    bytes: &[u8],
    offset: usize,
) -> Result<u8, DxilReflectError> {
    bytes
        .get(offset)
        .copied()
        .ok_or(DxilReflectError::OutOfBounds {
            fourcc,
            offset,
            length: 1,
            chunk_len: bytes.len(),
        })
}

fn ensure_slice(
    fourcc: &'static str,
    bytes: &[u8],
    offset: usize,
    length: usize,
) -> Result<(), DxilReflectError> {
    let end = offset.checked_add(length).ok_or_else(|| DxilReflectError::Overflow {
        fourcc,
        detail: format!("offset {offset} + length {length} overflow"),
    })?;
    if end > bytes.len() {
        return Err(DxilReflectError::OutOfBounds {
            fourcc,
            offset,
            length,
            chunk_len: bytes.len(),
        });
    }
    Ok(())
}

fn read_bounded_c_string(
    fourcc: &'static str,
    bytes: &[u8],
    offset: u32,
    min_offset: usize,
    element_index: usize,
    max_len: usize,
) -> Result<String, DxilReflectError> {
    let off = offset as usize;
    if off < min_offset || off >= bytes.len() {
        return Err(DxilReflectError::InvalidSignatureString {
            fourcc,
            element_index,
            offset,
        });
    }
    let tail = &bytes[off..];
    let limit = tail.len().min(max_len);
    let nul_pos = tail[..limit].iter().position(|&b| b == 0).ok_or(
        DxilReflectError::InvalidSignatureString {
            fourcc,
            element_index,
            offset,
        },
    )?;
    if nul_pos == 0 {
        return Err(DxilReflectError::InvalidSignatureString {
            fourcc,
            element_index,
            offset,
        });
    }
    let text = std::str::from_utf8(&tail[..nul_pos]).map_err(|_| {
        DxilReflectError::InvalidSignatureString {
            fourcc,
            element_index,
            offset,
        }
    })?;
    if !text.chars().all(|c| !c.is_control()) {
        return Err(DxilReflectError::InvalidSignatureString {
            fourcc,
            element_index,
            offset,
        });
    }
    Ok(text.to_string())
}

// ---------------------------------------------------------------------------
// Chunk Validators & Parsers: DXIL, SHDR/SHEX, RTS0, ISG1/OSG1, PSV0
// ---------------------------------------------------------------------------

fn validate_dxil_chunk(payload: &[u8]) -> Result<(), DxilReflectError> {
    const TAG: &str = "DXIL";
    if payload.len() < DXIL_PROGRAM_HEADER_BYTES {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc: TAG,
            actual: payload.len(),
            expected_min: DXIL_PROGRAM_HEADER_BYTES,
        });
    }
    let size_in_uint32 = read_u32_checked(TAG, payload, 4)? as usize;
    if size_in_uint32 > 0 {
        let program_bytes =
            checked_mul_size(TAG, size_in_uint32, 4, "DXIL program dword size overflow")?;
        ensure_slice(TAG, payload, 0, program_bytes)?;
    }
    let bitcode_offset = read_u32_checked(TAG, payload, 16)? as usize;
    let bitcode_size = read_u32_checked(TAG, payload, 20)? as usize;
    let bitcode_start =
        checked_add_offset(TAG, 8, bitcode_offset, "DXIL bitcode offset overflow")?;
    ensure_slice(TAG, payload, bitcode_start, bitcode_size)?;
    Ok(())
}

fn validate_legacy_shader_chunk(
    fourcc: &'static str,
    payload: &[u8],
) -> Result<(), DxilReflectError> {
    if payload.len() < LEGACY_SHADER_HEADER_BYTES {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc,
            actual: payload.len(),
            expected_min: LEGACY_SHADER_HEADER_BYTES,
        });
    }
    Ok(())
}

/// Parse an `RTS0` serialized D3D12 root signature chunk (v1.0 or v1.1).
///
/// Strictly bounds-checked and overflow-checked: guaranteed panic-free on arbitrary bytes.
pub fn parse_rts0_chunk(payload: &[u8]) -> Result<RootSignature, DxilReflectError> {
    const TAG: &str = "RTS0";
    if payload.len() < RTS0_HEADER_BYTES {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc: TAG,
            actual: payload.len(),
            expected_min: RTS0_HEADER_BYTES,
        });
    }

    let raw_version = read_u32_checked(TAG, payload, 0)?;
    let version = match raw_version {
        1 => RootSignatureVersion::V1_0,
        2 => RootSignatureVersion::V1_1,
        other => return Err(DxilReflectError::UnsupportedRootSignatureVersion(other)),
    };
    let num_parameters = read_u32_checked(TAG, payload, 4)? as usize;
    let parameters_offset = read_u32_checked(TAG, payload, 8)? as usize;
    let num_static_samplers = read_u32_checked(TAG, payload, 12)? as usize;
    let static_samplers_offset = read_u32_checked(TAG, payload, 16)? as usize;
    let flags = read_u32_checked(TAG, payload, 20)?;

    let mut parameters = Vec::new();
    if num_parameters > 0 {
        if parameters_offset < RTS0_HEADER_BYTES {
            return Err(DxilReflectError::OutOfBounds {
                fourcc: TAG,
                offset: parameters_offset,
                length: RTS0_PARAM_HEADER_BYTES,
                chunk_len: payload.len(),
            });
        }
        let table_bytes = checked_mul_size(
            TAG,
            num_parameters,
            RTS0_PARAM_HEADER_BYTES,
            "parameter header table size overflow",
        )?;
        ensure_slice(TAG, payload, parameters_offset, table_bytes)?;

        parameters.reserve(num_parameters);
        let mut total_ranges = 0usize;
        let range_stride = match version {
            RootSignatureVersion::V1_0 => RTS0_RANGE_V1_0_BYTES,
            RootSignatureVersion::V1_1 => RTS0_RANGE_V1_1_BYTES,
        };
        let root_desc_bytes = match version {
            RootSignatureVersion::V1_0 => 8usize,
            RootSignatureVersion::V1_1 => 12usize,
        };

        for idx in 0..num_parameters {
            let entry_delta = checked_mul_size(
                TAG,
                idx,
                RTS0_PARAM_HEADER_BYTES,
                "parameter entry offset overflow",
            )?;
            let entry_off = checked_add_offset(
                TAG,
                parameters_offset,
                entry_delta,
                "parameter entry offset overflow",
            )?;
            let param_type = read_u32_checked(TAG, payload, entry_off)?;
            let raw_vis = read_u32_checked(
                TAG,
                payload,
                checked_add_offset(TAG, entry_off, 4, "parameter visibility offset overflow")?,
            )?;
            let data_off = read_u32_checked(
                TAG,
                payload,
                checked_add_offset(TAG, entry_off, 8, "parameter data offset overflow")?,
            )? as usize;

            let visibility = D3dShaderVisibility::from_u32(raw_vis).ok_or(
                DxilReflectError::InvalidShaderVisibility {
                    context: "parameter",
                    index: idx,
                    raw: raw_vis,
                },
            )?;

            let kind = match param_type {
                0 => {
                    // D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE
                    if data_off < RTS0_HEADER_BYTES {
                        return Err(DxilReflectError::OutOfBounds {
                            fourcc: TAG,
                            offset: data_off,
                            length: 8,
                            chunk_len: payload.len(),
                        });
                    }
                    ensure_slice(TAG, payload, data_off, 8)?;
                    let num_ranges = read_u32_checked(TAG, payload, data_off)? as usize;
                    let ranges_off = read_u32_checked(
                        TAG,
                        payload,
                        checked_add_offset(TAG, data_off, 4, "descriptor table ranges_off overflow")?,
                    )? as usize;

                    total_ranges =
                        total_ranges
                            .checked_add(num_ranges)
                            .ok_or_else(|| DxilReflectError::Overflow {
                                fourcc: TAG,
                                detail: "total descriptor ranges overflow".to_string(),
                            })?;
                    if total_ranges > MAX_TOTAL_DESCRIPTOR_RANGES {
                        return Err(DxilReflectError::TooManyDescriptorRanges {
                            count: total_ranges,
                            max: MAX_TOTAL_DESCRIPTOR_RANGES,
                        });
                    }

                    let mut ranges = Vec::new();
                    if num_ranges > 0 {
                        if ranges_off < RTS0_HEADER_BYTES {
                            return Err(DxilReflectError::OutOfBounds {
                                fourcc: TAG,
                                offset: ranges_off,
                                length: range_stride,
                                chunk_len: payload.len(),
                            });
                        }
                        let ranges_bytes = checked_mul_size(
                            TAG,
                            num_ranges,
                            range_stride,
                            &format!("parameter {idx} descriptor ranges size overflow"),
                        )?;
                        ensure_slice(TAG, payload, ranges_off, ranges_bytes)?;

                        ranges.reserve(num_ranges);
                        for r_idx in 0..num_ranges {
                            let r_delta = checked_mul_size(
                                TAG,
                                r_idx,
                                range_stride,
                                "descriptor range offset overflow",
                            )?;
                            let r_off = checked_add_offset(
                                TAG,
                                ranges_off,
                                r_delta,
                                "descriptor range offset overflow",
                            )?;
                            let raw_range_type = read_u32_checked(TAG, payload, r_off)?;
                            let range_type = DescriptorRangeType::from_u32(raw_range_type).ok_or(
                                DxilReflectError::InvalidDescriptorRangeType {
                                    param_index: idx,
                                    range_index: r_idx,
                                    raw: raw_range_type,
                                },
                            )?;
                            let num_descriptors = read_u32_checked(
                                TAG,
                                payload,
                                checked_add_offset(TAG, r_off, 4, "range count offset overflow")?,
                            )?;
                            let base_shader_register = read_u32_checked(
                                TAG,
                                payload,
                                checked_add_offset(TAG, r_off, 8, "range base register offset overflow")?,
                            )?;
                            let register_space = read_u32_checked(
                                TAG,
                                payload,
                                checked_add_offset(TAG, r_off, 12, "range space offset overflow")?,
                            )?;
                            if num_descriptors > 0
                                && num_descriptors != u32::MAX
                                && base_shader_register
                                    .checked_add(num_descriptors - 1)
                                    .is_none()
                            {
                                return Err(DxilReflectError::Overflow {
                                    fourcc: TAG,
                                    detail: format!(
                                        "parameter {idx} descriptor range {r_idx} register index overflow (base={base_shader_register}, count={num_descriptors})"
                                    ),
                                });
                            }
                            let (range_flags, offset_from_start) = match version {
                                RootSignatureVersion::V1_0 => (
                                    0u32,
                                    read_u32_checked(
                                        TAG,
                                        payload,
                                        checked_add_offset(
                                            TAG,
                                            r_off,
                                            16,
                                            "range table offset overflow",
                                        )?,
                                    )?,
                                ),
                                RootSignatureVersion::V1_1 => (
                                    read_u32_checked(
                                        TAG,
                                        payload,
                                        checked_add_offset(TAG, r_off, 16, "range flags offset overflow")?,
                                    )?,
                                    read_u32_checked(
                                        TAG,
                                        payload,
                                        checked_add_offset(
                                            TAG,
                                            r_off,
                                            20,
                                            "range table offset overflow",
                                        )?,
                                    )?,
                                ),
                            };
                            ranges.push(DescriptorRange {
                                range_type,
                                num_descriptors,
                                base_shader_register,
                                register_space,
                                flags: range_flags,
                                offset_in_descriptors_from_table_start: offset_from_start,
                            });
                        }
                    }
                    RootParameterKind::DescriptorTable(ranges)
                }
                1 => {
                    // D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
                    if data_off < RTS0_HEADER_BYTES {
                        return Err(DxilReflectError::OutOfBounds {
                            fourcc: TAG,
                            offset: data_off,
                            length: 12,
                            chunk_len: payload.len(),
                        });
                    }
                    ensure_slice(TAG, payload, data_off, 12)?;
                    RootParameterKind::Constants32Bit {
                        shader_register: read_u32_checked(TAG, payload, data_off)?,
                        register_space: read_u32_checked(
                            TAG,
                            payload,
                            checked_add_offset(TAG, data_off, 4, "root constants space offset overflow")?,
                        )?,
                        num_32bit_values: read_u32_checked(
                            TAG,
                            payload,
                            checked_add_offset(TAG, data_off, 8, "root constants count offset overflow")?,
                        )?,
                    }
                }
                2 | 3 | 4 => {
                    // Root CBV / SRV / UAV
                    if data_off < RTS0_HEADER_BYTES {
                        return Err(DxilReflectError::OutOfBounds {
                            fourcc: TAG,
                            offset: data_off,
                            length: root_desc_bytes,
                            chunk_len: payload.len(),
                        });
                    }
                    ensure_slice(TAG, payload, data_off, root_desc_bytes)?;
                    let shader_register = read_u32_checked(TAG, payload, data_off)?;
                    let register_space = read_u32_checked(
                        TAG,
                        payload,
                        checked_add_offset(TAG, data_off, 4, "root descriptor space offset overflow")?,
                    )?;
                    let desc_flags = match version {
                        RootSignatureVersion::V1_0 => 0u32,
                        RootSignatureVersion::V1_1 => read_u32_checked(
                            TAG,
                            payload,
                            checked_add_offset(TAG, data_off, 8, "root descriptor flags offset overflow")?,
                        )?,
                    };
                    match param_type {
                        2 => RootParameterKind::Cbv {
                            shader_register,
                            register_space,
                            flags: desc_flags,
                        },
                        3 => RootParameterKind::Srv {
                            shader_register,
                            register_space,
                            flags: desc_flags,
                        },
                        _ => RootParameterKind::Uav {
                            shader_register,
                            register_space,
                            flags: desc_flags,
                        },
                    }
                }
                raw => {
                    return Err(DxilReflectError::InvalidRootParameterType { index: idx, raw });
                }
            };

            parameters.push(RootParameter { visibility, kind });
        }
    }

    let mut static_samplers = Vec::new();
    if num_static_samplers > 0 {
        if static_samplers_offset < RTS0_HEADER_BYTES {
            return Err(DxilReflectError::OutOfBounds {
                fourcc: TAG,
                offset: static_samplers_offset,
                length: RTS0_STATIC_SAMPLER_BYTES,
                chunk_len: payload.len(),
            });
        }
        let samplers_bytes = checked_mul_size(
            TAG,
            num_static_samplers,
            RTS0_STATIC_SAMPLER_BYTES,
            "static samplers table size overflow",
        )?;
        ensure_slice(TAG, payload, static_samplers_offset, samplers_bytes)?;

        static_samplers.reserve(num_static_samplers);
        for idx in 0..num_static_samplers {
            let s_delta = checked_mul_size(
                TAG,
                idx,
                RTS0_STATIC_SAMPLER_BYTES,
                "static sampler entry offset overflow",
            )?;
            let s_off = checked_add_offset(
                TAG,
                static_samplers_offset,
                s_delta,
                "static sampler entry offset overflow",
            )?;
            let raw_vis = read_u32_checked(
                TAG,
                payload,
                checked_add_offset(TAG, s_off, 48, "static sampler visibility offset overflow")?,
            )?;
            let shader_visibility = D3dShaderVisibility::from_u32(raw_vis).ok_or(
                DxilReflectError::InvalidShaderVisibility {
                    context: "static_sampler",
                    index: idx,
                    raw: raw_vis,
                },
            )?;
            static_samplers.push(StaticSamplerDesc {
                filter: read_u32_checked(TAG, payload, s_off)?,
                address_u: read_u32_checked(TAG, payload, s_off + 4)?,
                address_v: read_u32_checked(TAG, payload, s_off + 8)?,
                address_w: read_u32_checked(TAG, payload, s_off + 12)?,
                mip_lod_bias_bits: read_u32_checked(TAG, payload, s_off + 16)?,
                max_anisotropy: read_u32_checked(TAG, payload, s_off + 20)?,
                comparison_func: read_u32_checked(TAG, payload, s_off + 24)?,
                border_color: read_u32_checked(TAG, payload, s_off + 28)?,
                min_lod_bits: read_u32_checked(TAG, payload, s_off + 32)?,
                max_lod_bits: read_u32_checked(TAG, payload, s_off + 36)?,
                shader_register: read_u32_checked(TAG, payload, s_off + 40)?,
                register_space: read_u32_checked(TAG, payload, s_off + 44)?,
                shader_visibility,
            });
        }
    }

    Ok(RootSignature {
        version,
        flags,
        parameters,
        static_samplers,
    })
}

/// Parse an `ISG1`, `OSG1`, `ISGN`, or `OSGN` stage signature chunk.
///
/// Strictly bounds-checked with bounded NUL-terminated ASCII/UTF-8 semantic name reads.
pub fn parse_signature_elements(
    fourcc: [u8; 4],
    payload: &[u8],
) -> Result<StageSignature, DxilReflectError> {
    let (tag, is_v1) = match &fourcc {
        b"ISG1" => ("ISG1", true),
        b"OSG1" => ("OSG1", true),
        b"PSG1" => ("PSG1", true),
        b"ISGN" => ("ISGN", false),
        b"OSGN" => ("OSGN", false),
        b"PCSG" => ("PCSG", false),
        _ => ("SIG", true),
    };

    if payload.len() < SIG_HEADER_BYTES {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc: tag,
            actual: payload.len(),
            expected_min: SIG_HEADER_BYTES,
        });
    }

    let element_count = read_u32_checked(tag, payload, 0)? as usize;
    let param_offset = read_u32_checked(tag, payload, 4)? as usize;
    let stride = if is_v1 {
        SIG_V1_ELEMENT_BYTES
    } else {
        SIG_V0_ELEMENT_BYTES
    };

    let mut elements = Vec::new();
    if element_count > 0 {
        if param_offset < SIG_HEADER_BYTES {
            return Err(DxilReflectError::OutOfBounds {
                fourcc: tag,
                offset: param_offset,
                length: stride,
                chunk_len: payload.len(),
            });
        }
        let table_bytes = checked_mul_size(
            tag,
            element_count,
            stride,
            "signature element table size overflow",
        )?;
        ensure_slice(tag, payload, param_offset, table_bytes)?;

        elements.reserve(element_count);
        for idx in 0..element_count {
            let elem_delta = checked_mul_size(tag, idx, stride, "signature element offset overflow")?;
            let off = checked_add_offset(
                tag,
                param_offset,
                elem_delta,
                "signature element offset overflow",
            )?;
            let (
                stream,
                name_offset,
                semantic_index,
                system_value_type,
                raw_comp,
                register,
                mask,
                rw_mask,
                min_precision,
            ) = if is_v1 {
                (
                    read_u32_checked(tag, payload, off)?,
                    read_u32_checked(tag, payload, off + 4)?,
                    read_u32_checked(tag, payload, off + 8)?,
                    read_u32_checked(tag, payload, off + 12)?,
                    read_u32_checked(tag, payload, off + 16)?,
                    read_u32_checked(tag, payload, off + 20)?,
                    read_u8_checked(tag, payload, off + 24)?,
                    read_u8_checked(tag, payload, off + 25)?,
                    read_u32_checked(tag, payload, off + 28)?,
                )
            } else {
                (
                    0u32,
                    read_u32_checked(tag, payload, off)?,
                    read_u32_checked(tag, payload, off + 4)?,
                    read_u32_checked(tag, payload, off + 8)?,
                    read_u32_checked(tag, payload, off + 12)?,
                    read_u32_checked(tag, payload, off + 16)?,
                    read_u8_checked(tag, payload, off + 20)?,
                    read_u8_checked(tag, payload, off + 21)?,
                    0u32,
                )
            };

            let semantic_name = read_bounded_c_string(
                tag,
                payload,
                name_offset,
                SIG_HEADER_BYTES,
                idx,
                MAX_SEMANTIC_NAME_LEN,
            )?;
            elements.push(SignatureElement {
                stream,
                semantic_name,
                semantic_index,
                system_value_type,
                component_type: SignatureComponentType::from_u32(raw_comp),
                register,
                mask,
                rw_mask,
                min_precision,
            });
        }
    }

    Ok(StageSignature { fourcc, elements })
}

/// Parse a `PSV0` (Pipeline State Validation) chunk safely for stage, compute workgroup size,
/// and resource binding refinement.
pub fn parse_psv0_chunk(payload: &[u8]) -> Result<PsvInfo, DxilReflectError> {
    const TAG: &str = "PSV0";
    if payload.len() < 4 {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc: TAG,
            actual: payload.len(),
            expected_min: 4,
        });
    }

    let runtime_info_size = read_u32_checked(TAG, payload, 0)?;
    let rt_size = runtime_info_size as usize;
    if rt_size < PSV_MIN_RUNTIME_INFO_BYTES {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc: TAG,
            actual: rt_size,
            expected_min: PSV_MIN_RUNTIME_INFO_BYTES,
        });
    }
    ensure_slice(TAG, payload, 4, rt_size)?;

    let after_rt = checked_add_offset(TAG, 4, rt_size, "PSV0 runtime info end offset overflow")?;
    let rt = &payload[4..after_rt];
    let shader_stage = if rt.len() >= 25 {
        Some(DxilShaderKind::from_u16(u16::from(rt[24])))
    } else {
        None
    };
    let sig_input_elements = rt.get(25).copied();
    let sig_output_elements = rt.get(26).copied();

    // PSVRuntimeInfo2 (48 bytes) stores NumThreadsX/Y/Z at offsets 36..48 for compute/mesh/amp.
    let workgroup_size = if rt.len() >= 48
        && matches!(
            shader_stage,
            Some(
                DxilShaderKind::Compute
                    | DxilShaderKind::Mesh
                    | DxilShaderKind::Amplification
            )
        ) {
        let x = read_u32_checked(TAG, rt, 36)?;
        let y = read_u32_checked(TAG, rt, 40)?;
        let z = read_u32_checked(TAG, rt, 44)?;
        if x >= 1 && y >= 1 && z >= 1 {
            Some([x, y, z])
        } else {
            None
        }
    } else {
        None
    };

    let mut resources = Vec::new();
    let remaining_after_rt = payload.len().saturating_sub(after_rt);
    if remaining_after_rt > 0 && remaining_after_rt < 4 {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc: TAG,
            actual: remaining_after_rt,
            expected_min: 4,
        });
    }
    if remaining_after_rt >= 4 {
        let resource_count = read_u32_checked(TAG, payload, after_rt)? as usize;
        if resource_count > MAX_TOTAL_DESCRIPTOR_RANGES {
            return Err(DxilReflectError::TooManyDescriptorRanges {
                count: resource_count,
                max: MAX_TOTAL_DESCRIPTOR_RANGES,
            });
        }
        if resource_count > 0 {
            let bind_size_off =
                checked_add_offset(TAG, after_rt, 4, "PSV0 bind_info_size offset overflow")?;
            ensure_slice(TAG, payload, bind_size_off, 4)?;
            let bind_info_size = read_u32_checked(TAG, payload, bind_size_off)? as usize;
            if bind_info_size < 16 {
                return Err(DxilReflectError::TruncatedChunk {
                    fourcc: TAG,
                    actual: bind_info_size,
                    expected_min: 16,
                });
            }
            let table_start =
                checked_add_offset(TAG, after_rt, 8, "PSV0 resource table start overflow")?;
            let table_bytes = checked_mul_size(
                TAG,
                resource_count,
                bind_info_size,
                "PSV0 resource binding table size overflow",
            )?;
            ensure_slice(TAG, payload, table_start, table_bytes)?;

            resources.reserve(resource_count);
            for i in 0..resource_count {
                let entry_delta = checked_mul_size(
                    TAG,
                    i,
                    bind_info_size,
                    "PSV0 resource entry offset overflow",
                )?;
                let off = checked_add_offset(
                    TAG,
                    table_start,
                    entry_delta,
                    "PSV0 resource entry offset overflow",
                )?;
                let res_type = read_u32_checked(TAG, payload, off)?;
                let space = read_u32_checked(TAG, payload, off + 4)?;
                let lower_bound = read_u32_checked(TAG, payload, off + 8)?;
                let upper_bound = read_u32_checked(TAG, payload, off + 12)?;
                if lower_bound > upper_bound {
                    return Err(DxilReflectError::InvalidResourceBounds {
                        fourcc: TAG,
                        index: i,
                        lower_bound,
                        upper_bound,
                    });
                }
                let (res_kind, res_flags) = if bind_info_size >= 24 {
                    (
                        Some(read_u32_checked(TAG, payload, off + 16)?),
                        Some(read_u32_checked(TAG, payload, off + 20)?),
                    )
                } else {
                    (None, None)
                };
                resources.push(PsvResourceBinding {
                    res_type,
                    space,
                    lower_bound,
                    upper_bound,
                    res_kind,
                    res_flags,
                });
            }
        }
    }

    Ok(PsvInfo {
        runtime_info_size,
        shader_stage,
        sig_input_elements,
        sig_output_elements,
        workgroup_size,
        resources,
    })
}

// ---------------------------------------------------------------------------
// Deterministic Lowering: RTS0 + ISG1/OSG1 + PSV0 -> S6 WebGPU BindGroupLayouts
// ---------------------------------------------------------------------------

/// Intermediate D3D12 resource slot extracted from `RTS0` or `PSV0` before WebGPU group/binding
/// assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ExtractedSlot {
    register: D3dRegister,
    /// Inclusive upper register index covered by this slot (`register.index` for single bindings,
    /// `register.index + count - 1` for unexpanded fixed arrays, `u32::MAX` for unbounded ranges).
    range_end: u32,
    /// D3D12 shader stage visibility mask (`D3dShaderVisibility::d3d_stage_mask`) used to reject
    /// illegal same-stage duplicate/overlapping register indices.
    d3d_visibility_mask: u8,
    name: String,
    visibility: u32,
    resource: ResourceKind,
    layout: Option<BindingLayout>,
    root_constants: bool,
    binding_array: Option<BindingArraySize>,
    lowering_need: Option<(LoweringKind, String)>,
}

/// Reflect a `DXBC`/`DXIL` binary container into an S6-compatible [`DxilShaderReflection`]
/// using default [`RegisterConvention`] shifts (`0`).
pub fn reflect_dxil_bytes(bytes: &[u8]) -> Result<DxilShaderReflection, DxilReflectError> {
    reflect_dxil_bytes_with_options(bytes, None, &RegisterConvention::default())
}

/// Reflect a `DXBC`/`DXIL` binary container into an S6-compatible [`DxilShaderReflection`].
///
/// `source` should already be a sanitized relative path or basename (never a host path).
pub fn reflect_dxil_bytes_with_options(
    bytes: &[u8],
    source: Option<&str>,
    convention: &RegisterConvention,
) -> Result<DxilShaderReflection, DxilReflectError> {
    let info = parse_dxbc(bytes)?;
    let chunk_refs = parse_dxbc_chunks(bytes)?;

    let mut root_signature: Option<RootSignature> = None;
    let mut isg1: Option<StageSignature> = None;
    let mut isgn: Option<StageSignature> = None;
    let mut osg1: Option<StageSignature> = None;
    let mut osgn: Option<StageSignature> = None;
    let mut psv: Option<PsvInfo> = None;
    let mut rdef: Option<RdefInfo> = None;

    for chunk in chunk_refs {
        match &chunk.fourcc {
            b"DXIL" => {
                validate_dxil_chunk(chunk.payload)?;
            }
            b"SHDR" => {
                validate_legacy_shader_chunk("SHDR", chunk.payload)?;
            }
            b"SHEX" => {
                validate_legacy_shader_chunk("SHEX", chunk.payload)?;
            }
            b"RTS0" => {
                if root_signature.is_none() {
                    root_signature = Some(parse_rts0_chunk(chunk.payload)?);
                }
            }
            b"ISG1" => {
                if isg1.is_none() {
                    isg1 = Some(parse_signature_elements(chunk.fourcc, chunk.payload)?);
                }
            }
            b"ISGN" => {
                if isgn.is_none() {
                    isgn = Some(parse_signature_elements(chunk.fourcc, chunk.payload)?);
                }
            }
            b"OSG1" => {
                if osg1.is_none() {
                    osg1 = Some(parse_signature_elements(chunk.fourcc, chunk.payload)?);
                }
            }
            b"OSGN" => {
                if osgn.is_none() {
                    osgn = Some(parse_signature_elements(chunk.fourcc, chunk.payload)?);
                }
            }
            b"PSV0" => {
                if psv.is_none() {
                    psv = Some(parse_psv0_chunk(chunk.payload)?);
                }
            }
            b"RDEF" => {
                if rdef.is_none() {
                    rdef = parse_rdef_chunk(chunk.payload).ok();
                }
            }
            _ => {}
        }
    }

    let input_signature = isg1.or(isgn);
    let output_signature = osg1.or(osgn);

    let shader_hash_digest = info.find_chunk(b"HASH").and_then(|c| match &c.detail {
        DxbcChunkDetail::Hash(h) => Some(h.digest_hex.clone()),
        _ => None,
    });

    let shader_kind = info
        .shader_kind()
        .or_else(|| psv.as_ref().and_then(|p| p.shader_stage))
        .or_else(|| {
            rdef.as_ref()
                .map(|r| DxilShaderKind::from_u16(r.program_type))
        });
    let shader_model = info.shader_model();

    let (stage, stage_lowering_note) = infer_webgpu_stage(
        shader_kind,
        root_signature.as_ref(),
        input_signature.as_ref(),
        output_signature.as_ref(),
    );

    let mut requires_lowering = Vec::new();
    if let Some(note) = stage_lowering_note {
        requires_lowering.push(note);
    }

    let psv_resources: &[PsvResourceBinding] = psv
        .as_ref()
        .map(|p| p.resources.as_slice())
        .unwrap_or(&[]);

    // Extract D3D12 resource slots from RTS0 (authoritative), fallback to PSV0, then RDEF.
    let extracted = if let Some(rs) = &root_signature {
        extract_slots_from_rts0(rs, stage, shader_kind, psv_resources)?
    } else if !psv_resources.is_empty() {
        extract_slots_from_psv0(psv_resources, stage)?
    } else if let Some(rdef_info) = &rdef {
        extract_slots_from_rdef(rdef_info, stage)?
    } else {
        Vec::new()
    };

    // Deterministically lower extracted slots into WebGPU groups 0..=3.
    let (bindings, pipeline_layout, bind_group_remaps, slot_lowering) =
        lower_slots_to_webgpu(extracted, convention)?;
    requires_lowering.extend(slot_lowering);

    // Lower ISG1/OSG1 stage signatures to S6 vertex_inputs / fragment_outputs.
    let (vertex_inputs, fragment_outputs, io_lowering) = lower_stage_io(
        stage,
        input_signature.as_ref(),
        output_signature.as_ref(),
    );
    requires_lowering.extend(io_lowering);

    requires_lowering.sort();
    requires_lowering.dedup();

    let workgroup_size = if stage == ShaderStage::Compute {
        psv.as_ref().and_then(|p| p.workgroup_size)
    } else {
        None
    };

    let entry = EntryPointReflection {
        name: "main".to_string(),
        stage,
        bindings,
        vertex_inputs,
        fragment_outputs,
        workgroup_size,
        requires_lowering,
    };

    let limit_violations = check_pipeline_limits(&pipeline_layout, &[&entry]);

    Ok(DxilShaderReflection {
        source: source.map(|s| s.to_string()),
        hash: info.fnv1a_hash,
        container_digest: info.digest_hex(),
        shader_hash_digest,
        shader_kind,
        shader_model,
        root_signature,
        input_signature,
        output_signature,
        psv,
        entry,
        pipeline_layout,
        limit_violations,
        bind_group_remaps,
    })
}

fn infer_webgpu_stage(
    shader_kind: Option<DxilShaderKind>,
    root_signature: Option<&RootSignature>,
    input_sig: Option<&StageSignature>,
    output_sig: Option<&StageSignature>,
) -> (ShaderStage, Option<LoweringNeed>) {
    match shader_kind {
        Some(DxilShaderKind::Vertex) => (ShaderStage::Vertex, None),
        Some(DxilShaderKind::Pixel) => (ShaderStage::Fragment, None),
        Some(DxilShaderKind::Compute) => (ShaderStage::Compute, None),
        Some(DxilShaderKind::Mesh | DxilShaderKind::Amplification) => (ShaderStage::Compute, None),
        Some(other) => (
            ShaderStage::Compute,
            Some(LoweringNeed {
                kind: LoweringKind::UnsupportedResource,
                group: None,
                binding: None,
                detail: format!(
                    "DXIL stage '{other}' has no native WebGPU stage; lowered to compute pipeline slot"
                ),
            }),
        ),
        None => {
            // Standalone RTS0 or signature-only container: infer stage cleanly from metadata.
            if let Some(osig) = output_sig
                && osig.elements.iter().any(SignatureElement::is_render_target)
            {
                return (ShaderStage::Fragment, None);
            }
            if let Some(isig) = input_sig
                && isig.elements.iter().any(|e| !e.is_system_value())
            {
                return (ShaderStage::Vertex, None);
            }
            if let Some(rs) = root_signature {
                let mut saw_v = false;
                let mut saw_p = false;
                let mut saw_all = false;
                for p in &rs.parameters {
                    match p.visibility {
                        D3dShaderVisibility::Vertex => saw_v = true,
                        D3dShaderVisibility::Pixel => saw_p = true,
                        D3dShaderVisibility::All => saw_all = true,
                        _ => {}
                    }
                }
                if saw_v && !saw_p && !saw_all {
                    return (ShaderStage::Vertex, None);
                }
                if saw_p && !saw_v && !saw_all {
                    return (ShaderStage::Fragment, None);
                }
                if (rs.flags & RTS0_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS) != 0
                    && (rs.flags & RTS0_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS) != 0
                {
                    return (ShaderStage::Compute, None);
                }
            }
            (ShaderStage::Compute, None)
        }
    }
}

fn map_visibility(
    vis: D3dShaderVisibility,
    rts0_flags: u32,
    container_stage: ShaderStage,
    shader_kind: Option<DxilShaderKind>,
) -> u32 {
    match vis {
        D3dShaderVisibility::Vertex => VISIBILITY_VERTEX,
        D3dShaderVisibility::Pixel => VISIBILITY_FRAGMENT,
        D3dShaderVisibility::Hull | D3dShaderVisibility::Domain | D3dShaderVisibility::Geometry => {
            VISIBILITY_VERTEX
        }
        D3dShaderVisibility::Amplification | D3dShaderVisibility::Mesh => VISIBILITY_COMPUTE,
        D3dShaderVisibility::All => {
            if matches!(
                shader_kind,
                Some(
                    DxilShaderKind::Compute
                        | DxilShaderKind::Mesh
                        | DxilShaderKind::Amplification
                )
            ) {
                return VISIBILITY_COMPUTE;
            }
            let mut mask = if shader_kind.is_none() && container_stage == ShaderStage::Compute {
                VISIBILITY_VERTEX | VISIBILITY_FRAGMENT | VISIBILITY_COMPUTE
            } else {
                VISIBILITY_VERTEX | VISIBILITY_FRAGMENT
            };
            if (rts0_flags & RTS0_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS) != 0 {
                mask &= !VISIBILITY_VERTEX;
            }
            if (rts0_flags & RTS0_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS) != 0 {
                mask &= !VISIBILITY_FRAGMENT;
            }
            if mask == 0 {
                VISIBILITY_COMPUTE
            } else {
                mask
            }
        }
    }
}

fn find_psv_resource(
    psv_resources: &[PsvResourceBinding],
    class: RegisterClass,
    index: u32,
    space: u32,
) -> Option<&PsvResourceBinding> {
    psv_resources
        .iter()
        .filter(|r| r.matches(class, index, space))
        .min_by_key(|r| {
            (
                r.upper_bound.saturating_sub(r.lower_bound),
                r.res_type,
                r.lower_bound,
                r.upper_bound,
                r.res_kind,
                r.res_flags,
            )
        })
}

fn classify_table_range_resource(
    range_type: DescriptorRangeType,
    register_index: u32,
    register_space: u32,
    psv_resources: &[PsvResourceBinding],
) -> (ResourceKind, BindingLayout) {
    match range_type {
        DescriptorRangeType::Cbv => (
            ResourceKind::Buffer,
            BindingLayout::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 0,
            },
        ),
        DescriptorRangeType::Sampler => (
            ResourceKind::Sampler,
            BindingLayout::Sampler {
                ty: SamplerBindingType::Filtering,
            },
        ),
        DescriptorRangeType::Uav => (
            ResourceKind::Buffer,
            BindingLayout::Buffer {
                ty: BufferBindingType::Storage,
                has_dynamic_offset: false,
                min_binding_size: 0,
            },
        ),
        DescriptorRangeType::Srv => {
            if let Some(psv_res) =
                find_psv_resource(psv_resources, RegisterClass::T, register_index, register_space)
            {
                if psv_res.is_buffer_srv() {
                    return (
                        ResourceKind::Buffer,
                        BindingLayout::Buffer {
                            ty: BufferBindingType::ReadOnlyStorage,
                            has_dynamic_offset: false,
                            min_binding_size: 0,
                        },
                    );
                }
                let multisampled = psv_res.is_multisampled();
                let view_dimension = if multisampled {
                    TextureViewDimension::D2
                } else {
                    psv_res.view_dimension()
                };
                let sample_type = if multisampled {
                    TextureSampleType::UnfilterableFloat
                } else {
                    TextureSampleType::Float
                };
                return (
                    ResourceKind::Texture,
                    BindingLayout::Texture {
                        sample_type,
                        view_dimension,
                        multisampled,
                    },
                );
            }
            // Default for descriptor-table SRV without PSV0 buffer annotation: sampled 2D texture.
            (
                ResourceKind::Texture,
                BindingLayout::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
            )
        }
    }
}

fn extract_slots_from_rts0(
    rs: &RootSignature,
    container_stage: ShaderStage,
    shader_kind: Option<DxilShaderKind>,
    psv_resources: &[PsvResourceBinding],
) -> Result<Vec<ExtractedSlot>, DxilReflectError> {
    let mut slots = Vec::new();

    for param in &rs.parameters {
        let vis = map_visibility(param.visibility, rs.flags, container_stage, shader_kind);
        let d3d_vis = param.visibility.d3d_stage_mask();
        match &param.kind {
            RootParameterKind::Constants32Bit {
                shader_register,
                register_space,
                num_32bit_values,
            } => {
                let min_binding_size = u64::from(*num_32bit_values).saturating_mul(4);
                slots.push(ExtractedSlot {
                    register: D3dRegister {
                        class: RegisterClass::B,
                        index: *shader_register,
                        space: *register_space,
                    },
                    range_end: *shader_register,
                    d3d_visibility_mask: d3d_vis,
                    name: format!("root_const_b{shader_register}_space{register_space}"),
                    visibility: vis,
                    resource: ResourceKind::Buffer,
                    layout: Some(BindingLayout::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size,
                    }),
                    root_constants: true,
                    binding_array: None,
                    lowering_need: None,
                });
            }
            RootParameterKind::Cbv {
                shader_register,
                register_space,
                ..
            } => {
                slots.push(ExtractedSlot {
                    register: D3dRegister {
                        class: RegisterClass::B,
                        index: *shader_register,
                        space: *register_space,
                    },
                    range_end: *shader_register,
                    d3d_visibility_mask: d3d_vis,
                    name: format!("root_cbv_b{shader_register}_space{register_space}"),
                    visibility: vis,
                    resource: ResourceKind::Buffer,
                    layout: Some(BindingLayout::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: 0,
                    }),
                    root_constants: false,
                    binding_array: None,
                    lowering_need: None,
                });
            }
            RootParameterKind::Srv {
                shader_register,
                register_space,
                ..
            } => {
                // In D3D12, root SRV descriptors are always buffer views (Raw/Structured).
                slots.push(ExtractedSlot {
                    register: D3dRegister {
                        class: RegisterClass::T,
                        index: *shader_register,
                        space: *register_space,
                    },
                    range_end: *shader_register,
                    d3d_visibility_mask: d3d_vis,
                    name: format!("root_srv_t{shader_register}_space{register_space}"),
                    visibility: vis,
                    resource: ResourceKind::Buffer,
                    layout: Some(BindingLayout::Buffer {
                        ty: BufferBindingType::ReadOnlyStorage,
                        has_dynamic_offset: false,
                        min_binding_size: 0,
                    }),
                    root_constants: false,
                    binding_array: None,
                    lowering_need: None,
                });
            }
            RootParameterKind::Uav {
                shader_register,
                register_space,
                ..
            } => {
                // In D3D12, root UAV descriptors are always read-write storage buffers.
                slots.push(ExtractedSlot {
                    register: D3dRegister {
                        class: RegisterClass::U,
                        index: *shader_register,
                        space: *register_space,
                    },
                    range_end: *shader_register,
                    d3d_visibility_mask: d3d_vis,
                    name: format!("root_uav_u{shader_register}_space{register_space}"),
                    visibility: vis,
                    resource: ResourceKind::Buffer,
                    layout: Some(BindingLayout::Buffer {
                        ty: BufferBindingType::Storage,
                        has_dynamic_offset: false,
                        min_binding_size: 0,
                    }),
                    root_constants: false,
                    binding_array: None,
                    lowering_need: None,
                });
            }
            RootParameterKind::DescriptorTable(ranges) => {
                for range in ranges {
                    if range.num_descriptors == 0 {
                        continue;
                    }
                    let class = range.range_type.register_class();
                    if range.num_descriptors == u32::MAX {
                        let (resource, _) = classify_table_range_resource(
                            range.range_type,
                            range.base_shader_register,
                            range.register_space,
                            psv_resources,
                        );
                        let name = format!(
                            "table_{}_{}{}_space{}",
                            range.range_type.as_str(),
                            class.as_str(),
                            range.base_shader_register,
                            range.register_space
                        );
                        slots.push(ExtractedSlot {
                            register: D3dRegister {
                                class,
                                index: range.base_shader_register,
                                space: range.register_space,
                            },
                            range_end: u32::MAX,
                            d3d_visibility_mask: d3d_vis,
                            name: name.clone(),
                            visibility: vis,
                            resource,
                            layout: None,
                            root_constants: false,
                            binding_array: Some(BindingArraySize::Unbounded),
                            lowering_need: Some((
                                LoweringKind::UnboundedBindingArray,
                                format!(
                                    "'{name}': unbounded descriptor range ({}{}, space{}) has no baseline WebGPU layout",
                                    class.as_str(),
                                    range.base_shader_register,
                                    range.register_space
                                ),
                            )),
                        });
                    } else {
                        let range_end = range
                            .base_shader_register
                            .checked_add(range.num_descriptors - 1)
                            .ok_or_else(|| DxilReflectError::Overflow {
                                fourcc: "RTS0",
                                detail: format!(
                                    "descriptor range register index overflow (base={}, count={})",
                                    range.base_shader_register, range.num_descriptors
                                ),
                            })?;
                        if range.num_descriptors > MAX_EXPANDED_RANGE_DESCRIPTORS {
                            let (resource, _) = classify_table_range_resource(
                                range.range_type,
                                range.base_shader_register,
                                range.register_space,
                                psv_resources,
                            );
                            let name = format!(
                                "table_{}_{}{}_space{}",
                                range.range_type.as_str(),
                                class.as_str(),
                                range.base_shader_register,
                                range.register_space
                            );
                            slots.push(ExtractedSlot {
                                register: D3dRegister {
                                    class,
                                    index: range.base_shader_register,
                                    space: range.register_space,
                                },
                                range_end,
                                d3d_visibility_mask: d3d_vis,
                                name: name.clone(),
                                visibility: vis,
                                resource,
                                layout: None,
                                root_constants: false,
                                binding_array: Some(BindingArraySize::Fixed(range.num_descriptors)),
                                lowering_need: Some((
                                    LoweringKind::BindingArray,
                                    format!(
                                        "'{name}': descriptor range of {} elements ({}{}, space{}) exceeds WebGPU expansion limit ({MAX_EXPANDED_RANGE_DESCRIPTORS})",
                                        range.num_descriptors,
                                        class.as_str(),
                                        range.base_shader_register,
                                        range.register_space
                                    ),
                                )),
                            });
                        } else {
                            for offset in 0..range.num_descriptors {
                                let reg_idx = range
                                    .base_shader_register
                                    .checked_add(offset)
                                    .ok_or_else(|| DxilReflectError::Overflow {
                                        fourcc: "RTS0",
                                        detail: format!(
                                            "descriptor range register index overflow (base={}, offset={offset})",
                                            range.base_shader_register
                                        ),
                                    })?;
                                let (resource, layout) = classify_table_range_resource(
                                    range.range_type,
                                    reg_idx,
                                    range.register_space,
                                    psv_resources,
                                );
                                slots.push(ExtractedSlot {
                                    register: D3dRegister {
                                        class,
                                        index: reg_idx,
                                        space: range.register_space,
                                    },
                                    range_end: reg_idx,
                                    d3d_visibility_mask: d3d_vis,
                                    name: format!(
                                        "{}_{}{}_space{}",
                                        range.range_type.as_str(),
                                        class.as_str(),
                                        reg_idx,
                                        range.register_space
                                    ),
                                    visibility: vis,
                                    resource,
                                    layout: Some(layout),
                                    root_constants: false,
                                    binding_array: None,
                                    lowering_need: None,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    for sampler in &rs.static_samplers {
        let vis = map_visibility(
            sampler.shader_visibility,
            rs.flags,
            container_stage,
            shader_kind,
        );
        slots.push(ExtractedSlot {
            register: D3dRegister {
                class: RegisterClass::S,
                index: sampler.shader_register,
                space: sampler.register_space,
            },
            range_end: sampler.shader_register,
            d3d_visibility_mask: sampler.shader_visibility.d3d_stage_mask(),
            name: format!(
                "static_sampler_s{}_space{}",
                sampler.shader_register, sampler.register_space
            ),
            visibility: vis,
            resource: ResourceKind::Sampler,
            layout: Some(BindingLayout::Sampler {
                ty: sampler.sampler_binding_type(),
            }),
            root_constants: false,
            binding_array: None,
            lowering_need: None,
        });
    }

    Ok(slots)
}

fn extract_slots_from_psv0(
    psv_resources: &[PsvResourceBinding],
    container_stage: ShaderStage,
) -> Result<Vec<ExtractedSlot>, DxilReflectError> {
    let (vis, d3d_vis) = match container_stage {
        ShaderStage::Vertex => (VISIBILITY_VERTEX, D3dShaderVisibility::Vertex.d3d_stage_mask()),
        ShaderStage::Fragment => (VISIBILITY_FRAGMENT, D3dShaderVisibility::Pixel.d3d_stage_mask()),
        ShaderStage::Compute => (VISIBILITY_COMPUTE, D3dShaderVisibility::All.d3d_stage_mask()),
    };
    let mut slots = Vec::new();
    for (idx, res) in psv_resources.iter().enumerate() {
        if res.upper_bound < res.lower_bound {
            return Err(DxilReflectError::InvalidResourceBounds {
                fourcc: "PSV0",
                index: idx,
                lower_bound: res.lower_bound,
                upper_bound: res.upper_bound,
            });
        }
        let range_type = match res.res_type {
            1 => DescriptorRangeType::Sampler,
            2 => DescriptorRangeType::Cbv,
            3 | 4 | 5 | 10 => DescriptorRangeType::Srv,
            6 | 7 | 8 | 9 | 11 => DescriptorRangeType::Uav,
            _ => continue,
        };
        let class = range_type.register_class();
        if res.upper_bound == u32::MAX {
            let (resource, _) =
                classify_table_range_resource(range_type, res.lower_bound, res.space, psv_resources);
            let name = format!("psv_{}{}_space{}", class.as_str(), res.lower_bound, res.space);
            slots.push(ExtractedSlot {
                register: D3dRegister {
                    class,
                    index: res.lower_bound,
                    space: res.space,
                },
                range_end: u32::MAX,
                d3d_visibility_mask: d3d_vis,
                name: name.clone(),
                visibility: vis,
                resource,
                layout: None,
                root_constants: false,
                binding_array: Some(BindingArraySize::Unbounded),
                lowering_need: Some((
                    LoweringKind::UnboundedBindingArray,
                    format!("'{name}': unbounded PSV0 resource binding"),
                )),
            });
            continue;
        }
        let count = (res.upper_bound - res.lower_bound)
            .checked_add(1)
            .ok_or_else(|| DxilReflectError::Overflow {
                fourcc: "PSV0",
                detail: format!(
                    "PSV0 resource {idx} binding count overflow ({}..={})",
                    res.lower_bound, res.upper_bound
                ),
            })?;
        if count > MAX_EXPANDED_RANGE_DESCRIPTORS {
            let (resource, _) =
                classify_table_range_resource(range_type, res.lower_bound, res.space, psv_resources);
            let name = format!("psv_{}{}_space{}", class.as_str(), res.lower_bound, res.space);
            slots.push(ExtractedSlot {
                register: D3dRegister {
                    class,
                    index: res.lower_bound,
                    space: res.space,
                },
                range_end: res.upper_bound,
                d3d_visibility_mask: d3d_vis,
                name: name.clone(),
                visibility: vis,
                resource,
                layout: None,
                root_constants: false,
                binding_array: Some(BindingArraySize::Fixed(count)),
                lowering_need: Some((
                    LoweringKind::BindingArray,
                    format!("'{name}': PSV0 resource array of {count} elements exceeds expansion limit"),
                )),
            });
            continue;
        }
        for offset in 0..count {
            let reg_idx = res
                .lower_bound
                .checked_add(offset)
                .ok_or_else(|| DxilReflectError::Overflow {
                    fourcc: "PSV0",
                    detail: format!(
                        "PSV0 resource {idx} register index overflow (base={}, offset={offset})",
                        res.lower_bound
                    ),
                })?;
            let (resource, layout) =
                classify_table_range_resource(range_type, reg_idx, res.space, psv_resources);
            slots.push(ExtractedSlot {
                register: D3dRegister {
                    class,
                    index: reg_idx,
                    space: res.space,
                },
                range_end: reg_idx,
                d3d_visibility_mask: d3d_vis,
                name: format!("psv_{}{}_space{}", class.as_str(), reg_idx, res.space),
                visibility: vis,
                resource,
                layout: Some(layout),
                root_constants: false,
                binding_array: None,
                lowering_need: None,
            });
        }
    }
    Ok(slots)
}

fn slots_layout_compatible(a: &ExtractedSlot, b: &ExtractedSlot) -> bool {
    if a.resource != b.resource || a.binding_array != b.binding_array {
        return false;
    }
    match (a.layout, b.layout) {
        (None, None) => true,
        (
            Some(BindingLayout::Buffer {
                ty: ty_a,
                has_dynamic_offset: dyn_a,
                ..
            }),
            Some(BindingLayout::Buffer {
                ty: ty_b,
                has_dynamic_offset: dyn_b,
                ..
            }),
        ) => ty_a == ty_b && dyn_a == dyn_b,
        (Some(la), Some(lb)) => la == lb,
        _ => false,
    }
}

/// Detect illegal overlapping or duplicate register ranges sharing any D3D12 shader stage
/// visibility within the same `(register_space, RegisterClass)`.
fn validate_no_overlapping_same_stage_registers(
    raw_slots: &[ExtractedSlot],
) -> Result<(), DxilReflectError> {
    let mut by_space_class: BTreeMap<(u32, RegisterClass), Vec<(u32, u32, u8)>> = BTreeMap::new();
    for slot in raw_slots {
        by_space_class
            .entry((slot.register.space, slot.register.class))
            .or_default()
            .push((slot.register.index, slot.range_end, slot.d3d_visibility_mask));
    }

    for ((space, class), mut intervals) in by_space_class {
        intervals.sort_unstable();
        let mut active_max_end: [Option<(u32, u32)>; 7] = [None; 7];
        for (start, end, stage_mask) in intervals {
            for (bit_idx, active) in active_max_end.iter_mut().enumerate() {
                if (stage_mask & (1u8 << bit_idx)) == 0 {
                    continue;
                }
                if let Some((prev_start, prev_end)) = *active
                    && prev_end >= start
                {
                    return Err(DxilReflectError::DuplicateRegister {
                        register: D3dRegister {
                            class,
                            index: start,
                            space,
                        },
                        detail: format!(
                            "register range [{}..={}] overlaps existing range [{}..={}] with overlapping shader stage visibility",
                            start, end, prev_start, prev_end
                        ),
                    });
                }
                *active = Some((start, end));
            }
        }
    }
    Ok(())
}

/// Deduplicate identical D3D registers `(space, class, index)` across disjoint shader stages by
/// OR-ing visibility and keeping canonical deterministic order, reject illegal same-stage or
/// incompatible duplicates, and assign WebGPU `(group in 0..4, binding)` cleanly.
fn lower_slots_to_webgpu(
    raw_slots: Vec<ExtractedSlot>,
    convention: &RegisterConvention,
) -> Result<
    (
        Vec<ReflectedBinding>,
        PipelineLayoutDesc,
        Vec<DxilBindGroupRemap>,
        Vec<LoweringNeed>,
    ),
    DxilReflectError,
> {
    // 0. Reject any overlapping/duplicate registers that share D3D12 stage visibility.
    validate_no_overlapping_same_stage_registers(&raw_slots)?;

    // 1. Canonical sort + disjoint-stage merge by (space, class, index).
    let mut by_reg: BTreeMap<(u32, RegisterClass, u32), ExtractedSlot> = BTreeMap::new();
    for slot in raw_slots {
        let key = (slot.register.space, slot.register.class, slot.register.index);
        match by_reg.get_mut(&key) {
            Some(existing) => {
                if existing.range_end != slot.range_end || !slots_layout_compatible(existing, &slot)
                {
                    return Err(DxilReflectError::DuplicateRegister {
                        register: slot.register,
                        detail: format!(
                            "conflicting resource/layout declarations ('{}' vs '{}')",
                            existing.name, slot.name
                        ),
                    });
                }
                existing.visibility |= slot.visibility;
                existing.d3d_visibility_mask |= slot.d3d_visibility_mask;
                existing.root_constants |= slot.root_constants;
                if slot.name < existing.name {
                    existing.name = slot.name;
                }
                if let (
                    Some(BindingLayout::Buffer {
                        min_binding_size: a,
                        ..
                    }),
                    Some(BindingLayout::Buffer {
                        min_binding_size: b,
                        ..
                    }),
                ) = (&mut existing.layout, &slot.layout)
                {
                    *a = (*a).max(*b);
                }
                match (&mut existing.lowering_need, slot.lowering_need) {
                    (None, Some(need)) => existing.lowering_need = Some(need),
                    (Some(cur), Some(cand)) if cand < *cur => *cur = cand,
                    _ => {}
                }
            }
            None => {
                by_reg.insert(key, slot);
            }
        }
    }

    let mut native_slots = Vec::new();
    let mut overflow_spaces: BTreeMap<u32, Vec<ExtractedSlot>> = BTreeMap::new();
    for ((space, _, _), slot) in by_reg {
        if space < WEBGPU_MAX_BIND_GROUPS {
            native_slots.push(slot);
        } else {
            overflow_spaces.entry(space).or_default().push(slot);
        }
    }

    let mut used: [BTreeSet<u32>; WEBGPU_MAX_BIND_GROUPS as usize] = Default::default();
    let mut group_entries: [BTreeMap<u32, LayoutEntry>; WEBGPU_MAX_BIND_GROUPS as usize] =
        Default::default();
    let mut bindings = Vec::new();
    let mut remaps = Vec::new();
    let mut lowering_needs = Vec::new();

    // Pass 1: Place native spaces (0..4) into their matching WebGPU bind groups.
    for slot in native_slots {
        let group = slot.register.space;
        let preferred_binding = slot
            .register
            .index
            .checked_add(convention.shift(slot.register.class));
        let binding =
            allocate_preferred_or_next(group, &mut used[group as usize], preferred_binding)?;

        if let Some((kind, detail)) = slot.lowering_need {
            lowering_needs.push(LoweringNeed {
                kind,
                group: Some(group),
                binding: Some(binding),
                detail,
            });
        }
        if let Some(layout) = slot.layout {
            group_entries[group as usize].insert(
                binding,
                LayoutEntry {
                    binding,
                    visibility: slot.visibility,
                    layout,
                },
            );
        }
        bindings.push(ReflectedBinding {
            group,
            binding,
            name: Some(slot.name),
            resource: slot.resource,
            layout: slot.layout,
            binding_array: slot.binding_array,
            register: Some(slot.register),
            spirv: None,
            root_constants: slot.root_constants,
        });
    }

    // Pass 2: Deterministically clamp/remap overflow spaces (`space >= 4`) into `groups[0..4]`.
    for (orig_space, space_slots) in overflow_spaces {
        // Choose the least-populated WebGPU group in 0..4, tie-breaking by lowest group index.
        let mut target_group = select_least_populated_group(&used);
        for slot in space_slots {
            if used[target_group as usize].len() >= WEBGPU_MAX_BINDINGS_PER_GROUP {
                target_group = select_least_populated_group(&used);
            }
            let binding = allocate_next_binding(target_group, &mut used[target_group as usize])?;

            remaps.push(DxilBindGroupRemap {
                register: slot.register,
                original_space: orig_space,
                webgpu_group: target_group,
                webgpu_binding: binding,
            });

            if let Some((kind, detail)) = slot.lowering_need {
                lowering_needs.push(LoweringNeed {
                    kind,
                    group: Some(target_group),
                    binding: Some(binding),
                    detail,
                });
            }
            if let Some(layout) = slot.layout {
                group_entries[target_group as usize].insert(
                    binding,
                    LayoutEntry {
                        binding,
                        visibility: slot.visibility,
                        layout,
                    },
                );
            }
            bindings.push(ReflectedBinding {
                group: target_group,
                binding,
                name: Some(slot.name),
                resource: slot.resource,
                layout: slot.layout,
                binding_array: slot.binding_array,
                register: Some(slot.register),
                spirv: None,
                root_constants: slot.root_constants,
            });
        }
    }

    // Total canonical sort matching EntryPointReflection::canonicalized().
    bindings.sort();
    remaps.sort();
    lowering_needs.sort();
    lowering_needs.dedup();

    // Build dense PipelineLayoutDesc over 0..=highest_used_group (guaranteed <= 4).
    let highest_group = group_entries
        .iter()
        .enumerate()
        .rev()
        .find_map(|(g, map)| (!map.is_empty()).then_some(g));

    let bind_group_layouts = match highest_group {
        None => Vec::new(),
        Some(max_g) => (0..=max_g)
            .map(|g| BindGroupLayoutDesc {
                entries: group_entries[g].values().copied().collect(),
            })
            .collect(),
    };

    Ok((
        bindings,
        PipelineLayoutDesc { bind_group_layouts },
        remaps,
        lowering_needs,
    ))
}

fn select_least_populated_group(used: &[BTreeSet<u32>; WEBGPU_MAX_BIND_GROUPS as usize]) -> u32 {
    (0..(WEBGPU_MAX_BIND_GROUPS as usize))
        .min_by_key(|&g| (used[g].len(), g))
        .unwrap_or(0) as u32
}

fn allocate_preferred_or_next(
    group: u32,
    occupied: &mut BTreeSet<u32>,
    preferred: Option<u32>,
) -> Result<u32, DxilReflectError> {
    if let Some(slot) = preferred
        && !occupied.contains(&slot)
    {
        occupied.insert(slot);
        return Ok(slot);
    }
    allocate_next_binding(group, occupied)
}

fn allocate_next_binding(
    group: u32,
    occupied: &mut BTreeSet<u32>,
) -> Result<u32, DxilReflectError> {
    let candidate = match occupied.last().copied() {
        None => 0,
        Some(max_existing) => match max_existing.checked_add(1) {
            Some(next) => next,
            None => {
                let mut found = None;
                let mut slot = 0u32;
                loop {
                    if !occupied.contains(&slot) {
                        found = Some(slot);
                        break;
                    }
                    let Some(next) = slot.checked_add(1) else {
                        break;
                    };
                    slot = next;
                }
                found.ok_or(DxilReflectError::BindingSpaceExhausted { group })?
            }
        },
    };
    occupied.insert(candidate);
    Ok(candidate)
}

fn lower_stage_io(
    stage: ShaderStage,
    input_sig: Option<&StageSignature>,
    output_sig: Option<&StageSignature>,
) -> (Vec<ShaderIo>, Vec<ShaderIo>, Vec<LoweringNeed>) {
    let mut vertex_inputs = Vec::new();
    let mut fragment_outputs = Vec::new();
    let mut lowering = Vec::new();

    match stage {
        ShaderStage::Vertex => {
            if let Some(isig) = input_sig {
                let mut ordered: Vec<&SignatureElement> = isig.elements.iter().collect();
                ordered.sort_by(|a, b| {
                    (
                        a.register,
                        a.semantic_name.as_str(),
                        a.semantic_index,
                        a.mask,
                        a.component_type,
                    )
                        .cmp(&(
                            b.register,
                            b.semantic_name.as_str(),
                            b.semantic_index,
                            b.mask,
                            b.component_type,
                        ))
                });
                let mut seen_locations = BTreeSet::new();
                for elem in ordered {
                    if elem.is_system_value() || elem.component_count() == 0 {
                        continue;
                    }
                    if !seen_locations.insert(elem.register) {
                        continue;
                    }
                    let (wgsl_type, vertex_format) = elem.to_wgsl_io_type();
                    if vertex_format.is_none() {
                        lowering.push(LoweringNeed {
                            kind: LoweringKind::VertexInputFormat,
                            group: None,
                            binding: None,
                            detail: format!(
                                "vertex input @location({}) ({}{}) of type {} has no GPUVertexFormat",
                                elem.register, elem.semantic_name, elem.semantic_index, wgsl_type
                            ),
                        });
                    }
                    vertex_inputs.push(ShaderIo {
                        location: elem.register,
                        wgsl_type,
                        vertex_format,
                    });
                }
                vertex_inputs.sort();
            }
        }
        ShaderStage::Fragment => {
            if let Some(osig) = output_sig {
                let mut ordered: Vec<&SignatureElement> = osig.elements.iter().collect();
                ordered.sort_by(|a, b| {
                    (
                        a.semantic_index,
                        a.register,
                        a.semantic_name.as_str(),
                        a.mask,
                        a.component_type,
                    )
                        .cmp(&(
                            b.semantic_index,
                            b.register,
                            b.semantic_name.as_str(),
                            b.mask,
                            b.component_type,
                        ))
                });
                let mut seen_locations = BTreeSet::new();
                for elem in ordered {
                    if !elem.is_render_target() || elem.component_count() == 0 {
                        continue;
                    }
                    let location = elem.semantic_index;
                    if !seen_locations.insert(location) {
                        continue;
                    }
                    let (wgsl_type, _) = elem.to_wgsl_io_type();
                    fragment_outputs.push(ShaderIo {
                        location,
                        wgsl_type,
                        vertex_format: None,
                    });
                }
                fragment_outputs.sort();
            }
        }
        ShaderStage::Compute => {}
    }

    (vertex_inputs, fragment_outputs, lowering)
}

// ---------------------------------------------------------------------------
// Manifest Assembly & Sanitized S6 JSON Rendering
// ---------------------------------------------------------------------------

/// Sanitize a file path into a relative forward-slash path or basename only (zero host paths).
///
/// Normalizes both `\` and `/` separators before prefix stripping so Windows and POSIX paths
/// sanitize identically regardless of host OS.
pub fn sanitize_source_path(path: &Path, base_dir: Option<&Path>) -> String {
    let norm_path = path.to_string_lossy().replace('\\', "/");

    let relative_opt = base_dir.and_then(|base| {
        let norm_base = base.to_string_lossy().replace('\\', "/");
        let trimmed_base = norm_base.trim_end_matches('/');
        if trimmed_base.is_empty() {
            return None;
        }
        let rest = norm_path.strip_prefix(trimmed_base)?;
        if let Some(after_sep) = rest.strip_prefix('/')
            && !after_sep.is_empty()
        {
            Some(after_sep.to_string())
        } else {
            None
        }
    });

    let candidate = relative_opt.unwrap_or_else(|| {
        norm_path
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or("unknown.dxil")
            .to_string()
    });

    let trimmed = candidate.trim_start_matches('/');
    // Guard against Windows drive prefixes (e.g. "C:/..." or "C:") ever leaking through.
    let without_drive = if trimmed.len() >= 2 && trimmed.as_bytes()[1] == b':' {
        match trimmed.rsplit_once('/') {
            Some((_, tail)) if !tail.is_empty() => tail,
            _ => "unknown.dxil",
        }
    } else {
        trimmed
    };
    if without_drive.is_empty() {
        "unknown.dxil".to_string()
    } else {
        without_drive.to_string()
    }
}

/// Assemble a deduplicated S6-compatible [`DxilReflectionManifest`] from reflected containers.
///
/// Output is strictly deterministic (sorted by hash / canonical keys) regardless of input order.
pub fn build_dxil_reflection_manifest(
    reflections: &[DxilShaderReflection],
    errors: &[DxilReflectionErrorRecord],
    convention: &RegisterConvention,
) -> Result<DxilReflectionManifest, DxilReflectError> {
    let mut manifest = DxilReflectionManifest {
        register_convention: *convention,
        ..DxilReflectionManifest::default()
    };

    for refl in reflections {
        let mut bgl_ids = Vec::with_capacity(refl.pipeline_layout.bind_group_layouts.len());
        for bgl in &refl.pipeline_layout.bind_group_layouts {
            let canonical_bgl = bgl.canonicalized();
            let id = canonical_bgl.content_hash();
            match manifest.bind_group_layouts.get(&id) {
                Some(existing) if *existing != canonical_bgl => {
                    return Err(DxilReflectError::HashCollision {
                        table: "bind_group_layouts",
                        hash: id,
                    });
                }
                Some(_) => {}
                None => {
                    manifest.bind_group_layouts.insert(id, canonical_bgl);
                }
            }
            bgl_ids.push(id);
        }
        let pl_id = pipeline_layout_hash(&bgl_ids);
        match manifest.pipeline_layouts.get(&pl_id) {
            Some(existing) if *existing != bgl_ids => {
                return Err(DxilReflectError::HashCollision {
                    table: "pipeline_layouts",
                    hash: pl_id,
                });
            }
            Some(_) => {}
            None => {
                manifest.pipeline_layouts.insert(pl_id, bgl_ids);
            }
        }
        let canonical_refl = refl.canonicalized();
        match manifest.shaders.get_mut(&canonical_refl.hash) {
            Some(existing) => {
                if existing.pipeline_layout != canonical_refl.pipeline_layout
                    || existing.entry != canonical_refl.entry
                {
                    return Err(DxilReflectError::HashCollision {
                        table: "shaders",
                        hash: canonical_refl.hash,
                    });
                }
                // Deterministic canonical source tie-breaker when identical containers appear
                // under multiple sanitized filenames.
                match (&existing.source, &canonical_refl.source) {
                    (None, Some(src)) => existing.source = Some(src.clone()),
                    (Some(cur), Some(cand)) if cand < cur => existing.source = Some(cand.clone()),
                    _ => {}
                }
            }
            None => {
                manifest.shaders.insert(canonical_refl.hash, canonical_refl);
            }
        }
    }

    let mut sorted_errors = errors.to_vec();
    sorted_errors.sort_by(|a, b| (&a.source, &a.error).cmp(&(&b.source, &b.error)));
    sorted_errors.dedup();
    manifest.errors = sorted_errors;

    Ok(manifest)
}

impl DxilReflectionManifest {
    /// Project this DXIL manifest into a pure [`crate::reflect::LayoutManifest`] with identical
    /// bind group layout and pipeline layout content hashes.
    pub fn to_layout_manifest(&self) -> LayoutManifest {
        let mut shaders = BTreeMap::new();
        for (&hash, refl) in &self.shaders {
            let bind_group_layouts = refl.pipeline_layout.bind_group_layout_ids();
            let pipeline_layout = refl.pipeline_layout.content_hash();
            shaders.insert(
                hash,
                ShaderLayoutRecord {
                    entry: refl.entry.clone(),
                    pipeline_layout,
                    bind_group_layouts,
                    limit_violations: refl.limit_violations.clone(),
                },
            );
        }
        LayoutManifest {
            register_convention: self.register_convention,
            bind_group_layouts: self.bind_group_layouts.clone(),
            pipeline_layouts: self.pipeline_layouts.clone(),
            shaders,
            pipelines: BTreeMap::new(),
        }
    }

    /// Render deterministic, strictly sanitized S6 reflection JSON.
    ///
    /// Directly compatible with `web/workers/reflection_loader.js` (`parseReflectionManifest`)
    /// and [`crate::reflect::LayoutManifest`], plus sanitized DXIL `RTS0`/`ISG1`/`OSG1` metadata.
    pub fn render_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        let _ = writeln!(
            out,
            "  \"schema_version\": {LAYOUT_MANIFEST_SCHEMA_VERSION},"
        );
        let _ = writeln!(
            out,
            "  \"register_convention\": {{\"source\":\"dxil-rts0\",\"shifts\":{{\"b\":{},\"t\":{},\"u\":{},\"s\":{}}}}},",
            self.register_convention.shift_b,
            self.register_convention.shift_t,
            self.register_convention.shift_u,
            self.register_convention.shift_s
        );

        out.push_str("  \"bind_group_layouts\": ");
        write_hash_object(&mut out, "  ", &self.bind_group_layouts, |out, bgl| {
            out.push_str("{\n      \"entries\": ");
            let items: Vec<String> = bgl.entries.iter().map(LayoutEntry::to_json).collect();
            write_compact_array(out, "      ", &items);
            out.push_str("\n    }");
        });
        out.push_str(",\n");

        out.push_str("  \"pipeline_layouts\": ");
        write_hash_object(&mut out, "  ", &self.pipeline_layouts, |out, ids| {
            let _ = write!(out, "{{\"bindGroupLayouts\": {}}}", format_id_list(ids));
        });
        out.push_str(",\n");

        out.push_str("  \"shaders\": ");
        write_hash_object(&mut out, "  ", &self.shaders, |out, refl| {
            let ep = &refl.entry;
            let bgl_ids = refl.pipeline_layout.bind_group_layout_ids();
            let pl_id = refl.pipeline_layout.content_hash();

            out.push_str("{\n");
            if let Some(src) = &refl.source {
                out.push_str("      \"name\": ");
                write_json_string(out, src);
                out.push_str(",\n");
            }
            let _ = writeln!(out, "      \"stage\": \"{}\",", ep.stage.as_str());
            out.push_str("      \"entry_point\": ");
            write_json_string(out, &ep.name);
            out.push_str(",\n");
            let _ = writeln!(
                out,
                "      \"pipeline_layout\": \"{}\",",
                format_shader_hash(pl_id)
            );
            let _ = writeln!(
                out,
                "      \"bind_group_layouts\": {},",
                format_id_list(&bgl_ids)
            );
            let _ = writeln!(out, "      \"layout_complete\": {},", ep.layout_complete());
            let _ = writeln!(
                out,
                "      \"workgroup_size\": {},",
                match ep.workgroup_size {
                    Some([x, y, z]) => format!("[{x}, {y}, {z}]"),
                    None => "null".to_string(),
                }
            );

            let binding_items: Vec<String> =
                ep.bindings.iter().map(format_reflected_binding_json).collect();
            out.push_str("      \"bindings\": ");
            write_compact_array(out, "      ", &binding_items);
            out.push_str(",\n");

            let vi_items: Vec<String> = ep
                .vertex_inputs
                .iter()
                .map(|io| format_shader_io_json(io, true))
                .collect();
            out.push_str("      \"vertex_inputs\": ");
            write_compact_array(out, "      ", &vi_items);
            out.push_str(",\n");

            let fo_items: Vec<String> = ep
                .fragment_outputs
                .iter()
                .map(|io| format_shader_io_json(io, false))
                .collect();
            out.push_str("      \"fragment_outputs\": ");
            write_compact_array(out, "      ", &fo_items);
            out.push_str(",\n");

            let lower_items: Vec<String> = ep
                .requires_lowering
                .iter()
                .map(format_lowering_need_json)
                .collect();
            out.push_str("      \"requires_lowering\": ");
            write_compact_array(out, "      ", &lower_items);
            out.push_str(",\n");

            let limit_items: Vec<String> = refl
                .limit_violations
                .iter()
                .map(format_limit_violation_json)
                .collect();
            out.push_str("      \"limit_violations\": ");
            write_compact_array(out, "      ", &limit_items);
            out.push_str(",\n");

            out.push_str("      \"dxil\": ");
            write_dxil_metadata_json(out, refl);
            out.push_str("\n    }");
        });
        out.push_str(",\n");

        out.push_str("  \"pipelines\": {},\n");

        let error_items: Vec<String> = self
            .errors
            .iter()
            .map(|e| {
                let mut item = String::from("{\"source\":");
                write_json_string(&mut item, &e.source);
                item.push_str(",\"error\":");
                write_json_string(&mut item, &e.error);
                item.push('}');
                item
            })
            .collect();
        out.push_str("  \"errors\": ");
        write_compact_array(&mut out, "  ", &error_items);
        out.push_str("\n}\n");
        out
    }
}

fn pipeline_layout_hash(bgl_ids: &[u64]) -> u64 {
    let ids: Vec<String> = bgl_ids.iter().map(|&id| format_shader_hash(id)).collect();
    fnv1a_64(format!("pl:[{}]", ids.join(",")).as_bytes())
}

fn format_id_list(ids: &[u64]) -> String {
    let items: Vec<String> = ids
        .iter()
        .map(|&id| format!("\"{}\"", format_shader_hash(id)))
        .collect();
    format!("[{}]", items.join(", "))
}

fn format_reflected_binding_json(b: &ReflectedBinding) -> String {
    let mut out = format!("{{\"group\":{},\"binding\":{},\"name\":", b.group, b.binding);
    match &b.name {
        Some(name) => write_json_string(&mut out, name),
        None => out.push_str("null"),
    }
    let _ = write!(
        out,
        ",\"resource\":\"{}\",\"in_layout\":{},\"register\":",
        b.resource.as_str(),
        b.layout.is_some()
    );
    match b.register {
        Some(reg) => {
            let _ = write!(
                out,
                "{{\"class\":\"{}\",\"index\":{},\"space\":{}}}",
                reg.class.as_str(),
                reg.index,
                reg.space
            );
        }
        None => out.push_str("null"),
    }
    out.push_str(",\"spirv\":");
    match b.spirv {
        Some(s) => {
            let _ = write!(out, "{{\"set\":{},\"binding\":{}}}", s.set, s.binding);
        }
        None => out.push_str("null"),
    }
    let _ = write!(
        out,
        ",\"root_constants\":{},\"binding_array\":",
        b.root_constants
    );
    match b.binding_array {
        Some(BindingArraySize::Fixed(n)) => {
            let _ = write!(out, "{{\"count\":{n}}}");
        }
        Some(BindingArraySize::Unbounded) => out.push_str("{\"count\":null}"),
        None => out.push_str("null"),
    }
    out.push('}');
    out
}

fn format_shader_io_json(io: &ShaderIo, with_format: bool) -> String {
    let mut out = format!("{{\"location\":{},\"type\":", io.location);
    write_json_string(&mut out, &io.wgsl_type);
    if with_format {
        out.push_str(",\"format\":");
        match io.vertex_format {
            Some(fmt) => {
                let _ = write!(out, "\"{fmt}\"");
            }
            None => out.push_str("null"),
        }
    }
    out.push('}');
    out
}

fn format_lowering_need_json(need: &LoweringNeed) -> String {
    let mut out = format!(
        "{{\"kind\":\"{}\",\"group\":{},\"binding\":{},\"detail\":",
        need.kind.as_str(),
        need.group.map_or_else(|| "null".to_string(), |v| v.to_string()),
        need.binding
            .map_or_else(|| "null".to_string(), |v| v.to_string())
    );
    write_json_string(&mut out, &need.detail);
    out.push('}');
    out
}

fn format_limit_violation_json(v: &LimitViolation) -> String {
    format!(
        "{{\"limit\":\"{}\",\"stage\":{},\"count\":{},\"max\":{}}}",
        v.limit,
        match v.stage {
            Some(s) => format!("\"{}\"", s.as_str()),
            None => "null".to_string(),
        },
        v.count,
        v.max
    )
}

fn format_sig_element_json(e: &SignatureElement) -> String {
    let mut out = String::from("{\"semantic_name\":");
    write_json_string(&mut out, &e.semantic_name);
    let _ = write!(
        out,
        ",\"semantic_index\":{},\"system_value_type\":{},\"component_type\":\"{}\",\"register\":{},\"mask\":{},\"rw_mask\":{},\"stream\":{}}}",
        e.semantic_index,
        e.system_value_type,
        e.component_type.as_str(),
        e.register,
        e.mask,
        e.rw_mask,
        e.stream
    );
    out
}

fn format_remap_json(r: &DxilBindGroupRemap) -> String {
    format!(
        "{{\"register\":{{\"class\":\"{}\",\"index\":{},\"space\":{}}},\"original_space\":{},\"webgpu_group\":{},\"webgpu_binding\":{}}}",
        r.register.class.as_str(),
        r.register.index,
        r.register.space,
        r.original_space,
        r.webgpu_group,
        r.webgpu_binding
    )
}

fn write_dxil_metadata_json(out: &mut String, refl: &DxilShaderReflection) {
    out.push_str("{\n");
    out.push_str("        \"source\": ");
    match &refl.source {
        Some(s) => write_json_string(out, s),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    let _ = writeln!(
        out,
        "        \"fnv1a_hash\": \"{}\",",
        format_shader_hash(refl.hash)
    );
    out.push_str("        \"container_digest\": ");
    write_json_string(out, &refl.container_digest);
    out.push_str(",\n");
    out.push_str("        \"shader_hash_digest\": ");
    match &refl.shader_hash_digest {
        Some(h) => write_json_string(out, h),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    out.push_str("        \"shader_kind\": ");
    match refl.shader_kind {
        Some(kind) => write_json_string(out, &kind.to_string()),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    out.push_str("        \"shader_model\": ");
    match refl.shader_model {
        Some((maj, min)) => write_json_string(out, &format!("{maj}.{min}")),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    out.push_str("        \"rts0_version\": ");
    match &refl.root_signature {
        Some(rs) => write_json_string(out, rs.version.as_str()),
        None => out.push_str("null"),
    }
    out.push_str(",\n");
    let _ = writeln!(
        out,
        "        \"rts0_flags\": {},",
        refl.root_signature.as_ref().map_or(0, |rs| rs.flags)
    );

    let remap_items: Vec<String> = refl.bind_group_remaps.iter().map(format_remap_json).collect();
    out.push_str("        \"bind_group_remaps\": ");
    write_compact_array(out, "        ", &remap_items);
    out.push_str(",\n");

    let in_items: Vec<String> = refl
        .input_signature
        .as_ref()
        .map(|s| s.elements.iter().map(format_sig_element_json).collect())
        .unwrap_or_default();
    out.push_str("        \"stage_inputs\": ");
    write_compact_array(out, "        ", &in_items);
    out.push_str(",\n");

    let out_items: Vec<String> = refl
        .output_signature
        .as_ref()
        .map(|s| s.elements.iter().map(format_sig_element_json).collect())
        .unwrap_or_default();
    out.push_str("        \"stage_outputs\": ");
    write_compact_array(out, "        ", &out_items);
    out.push_str("\n      }");
}

fn write_compact_array(out: &mut String, indent: &str, items: &[String]) {
    if items.is_empty() {
        out.push_str("[]");
        return;
    }
    out.push_str("[\n");
    for (i, item) in items.iter().enumerate() {
        let _ = write!(out, "{indent}  {item}");
        out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
    }
    out.push_str(indent);
    out.push(']');
}

fn write_hash_object<V>(
    out: &mut String,
    indent: &str,
    map: &BTreeMap<u64, V>,
    mut write_value: impl FnMut(&mut String, &V),
) {
    if map.is_empty() {
        out.push_str("{}");
        return;
    }
    out.push_str("{\n");
    for (i, (&key, value)) in map.iter().enumerate() {
        let _ = write!(out, "{indent}  ");
        write_json_string(out, &format_shader_hash(key));
        out.push_str(": ");
        write_value(out, value);
        out.push_str(if i + 1 < map.len() { ",\n" } else { "\n" });
    }
    out.push_str(indent);
    out.push('}');
}

// ---------------------------------------------------------------------------
// S71 (#1742, Refs #482, #651): DXIL Fallback Reflection (RDEF / STAT / PSV0 /
// ISGN / OSGN / DXIL bitcode) & External Root-Parameter Matching
// ---------------------------------------------------------------------------

/// Console graphics first root-parameter user-data SGPR (`runtime/gpu/stage_reflection.h`).
pub const CONSOLE_GRAPHICS_USER_DATA_BASE: u32 = 8;

/// Console compute first root-parameter user-data SGPR (`runtime/gpu/stage_reflection.h`).
pub const CONSOLE_COMPUTE_USER_DATA_BASE: u32 = 4;

/// Console mesh (lowered to compute) first root-parameter user-data SGPR.
pub const CONSOLE_MESH_USER_DATA_BASE: u32 = 16;

/// Console descriptor table slot stride in bytes (`stage_reflection.h::kTableSlotBytes`).
pub const CONSOLE_TABLE_SLOT_BYTES: u32 = 32;

const RDEF_MIN_HEADER_BYTES: usize = 28;
const RDEF_RD11_HEADER_BYTES: usize = 60;
const RDEF_CBUFFER_BYTES: usize = 24;
const RDEF_BINDING_SM4_BYTES: usize = 32;
const RDEF_BINDING_RD11_BYTES: usize = 40;
const D3D_SIF_COMPARISON_SAMPLER: u32 = 0x02;
const D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND: u32 = u32::MAX;

/// `D3D_SHADER_INPUT_TYPE` values encoded in `RDEF` resource binding records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RdefInputType {
    CBuffer,
    TBuffer,
    Texture,
    Sampler,
    UavRwTyped,
    Structured,
    UavRwStructured,
    ByteAddress,
    UavRwByteAddress,
    UavAppendStructured,
    UavConsumeStructured,
    UavRwStructuredWithCounter,
    RtAccelerationStructure,
    Other(u32),
}

impl RdefInputType {
    pub const fn from_u32(raw: u32) -> Self {
        match raw {
            0 => Self::CBuffer,
            1 => Self::TBuffer,
            2 => Self::Texture,
            3 => Self::Sampler,
            4 => Self::UavRwTyped,
            5 => Self::Structured,
            6 => Self::UavRwStructured,
            7 => Self::ByteAddress,
            8 => Self::UavRwByteAddress,
            9 => Self::UavAppendStructured,
            10 => Self::UavConsumeStructured,
            11 => Self::UavRwStructuredWithCounter,
            12 => Self::RtAccelerationStructure,
            other => Self::Other(other),
        }
    }

    pub const fn register_class(self) -> Option<RegisterClass> {
        match self {
            Self::CBuffer => Some(RegisterClass::B),
            Self::TBuffer
            | Self::Texture
            | Self::Structured
            | Self::ByteAddress
            | Self::RtAccelerationStructure => Some(RegisterClass::T),
            Self::Sampler => Some(RegisterClass::S),
            Self::UavRwTyped
            | Self::UavRwStructured
            | Self::UavRwByteAddress
            | Self::UavAppendStructured
            | Self::UavConsumeStructured
            | Self::UavRwStructuredWithCounter => Some(RegisterClass::U),
            Self::Other(_) => None,
        }
    }
}

/// `D3D_SRV_DIMENSION` values encoded in `RDEF` resource binding records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RdefResourceDimension {
    Unknown,
    Buffer,
    Texture1D,
    Texture1DArray,
    Texture2D,
    Texture2DArray,
    Texture2DMS,
    Texture2DMSArray,
    Texture3D,
    TextureCube,
    TextureCubeArray,
    BufferEx,
    Other(u32),
}

impl RdefResourceDimension {
    pub const fn from_u32(raw: u32) -> Self {
        match raw {
            0 => Self::Unknown,
            1 => Self::Buffer,
            2 => Self::Texture1D,
            3 => Self::Texture1DArray,
            4 => Self::Texture2D,
            5 => Self::Texture2DArray,
            6 => Self::Texture2DMS,
            7 => Self::Texture2DMSArray,
            8 => Self::Texture3D,
            9 => Self::TextureCube,
            10 => Self::TextureCubeArray,
            11 => Self::BufferEx,
            other => Self::Other(other),
        }
    }

    pub const fn to_view_dimension(self) -> TextureViewDimension {
        match self {
            Self::Texture1D => TextureViewDimension::D1,
            Self::Texture2D | Self::Texture2DMS => TextureViewDimension::D2,
            Self::Texture2DArray | Self::Texture2DMSArray | Self::Texture1DArray => {
                TextureViewDimension::D2Array
            }
            Self::Texture3D => TextureViewDimension::D3,
            Self::TextureCube => TextureViewDimension::Cube,
            Self::TextureCubeArray => TextureViewDimension::CubeArray,
            _ => TextureViewDimension::D2,
        }
    }

    pub const fn is_multisampled(self) -> bool {
        matches!(self, Self::Texture2DMS | Self::Texture2DMSArray)
    }
}

/// Constant buffer reflection descriptor extracted from an `RDEF` chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdefCBufferDesc {
    pub name: String,
    pub size_bytes: u32,
    pub variable_count: u32,
    pub flags: u32,
    pub cb_type: u32,
}

/// Resource binding reflection descriptor extracted from an `RDEF` chunk (SM4/5 or SM5.1 `RD11`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdefResourceBinding {
    pub name: String,
    pub input_type: RdefInputType,
    pub return_type: u32,
    pub dimension: RdefResourceDimension,
    pub num_samples: u32,
    pub bind_point: u32,
    /// Number of bound registers (`0` in SM5.1 `RD11` indicates an unbounded bindless array).
    pub bind_count: u32,
    pub flags: u32,
    pub space: u32,
    pub id: u32,
}

impl RdefResourceBinding {
    pub const fn is_comparison_sampler(&self) -> bool {
        (self.flags & D3D_SIF_COMPARISON_SAMPLER) != 0
    }

    pub const fn is_unbounded(&self) -> bool {
        self.bind_count == 0 || self.bind_count == u32::MAX
    }
}

/// Full parsed `RDEF` (Resource Definition) chunk metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdefInfo {
    pub target_major: u8,
    pub target_minor: u8,
    pub program_type: u16,
    pub is_rd11: bool,
    pub creator: Option<String>,
    pub cbuffers: Vec<RdefCBufferDesc>,
    pub resources: Vec<RdefResourceBinding>,
}

/// Parsed `STAT` (Shader Statistics) chunk metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatInfo {
    pub instruction_count: u32,
    pub temp_register_count: u32,
    pub def_count: u32,
    pub dcl_count: u32,
    pub float_instruction_count: u32,
    pub int_instruction_count: u32,
    pub uint_instruction_count: u32,
    pub texture_normal_instructions: Option<u32>,
    pub texture_load_instructions: Option<u32>,
    pub texture_comp_instructions: Option<u32>,
    pub texture_bias_instructions: Option<u32>,
    pub texture_gradient_instructions: Option<u32>,
}

/// Parse an `RDEF` chunk (SM4/5 and SM5.1 `RD11`) with full bounds and overflow checks.
pub fn parse_rdef_chunk(payload: &[u8]) -> Result<RdefInfo, DxilReflectError> {
    const TAG: &str = "RDEF";
    if payload.len() < RDEF_MIN_HEADER_BYTES {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc: TAG,
            actual: payload.len(),
            expected_min: RDEF_MIN_HEADER_BYTES,
        });
    }

    let cbuffer_count = read_u32_checked(TAG, payload, 0)? as usize;
    let cbuffer_offset = read_u32_checked(TAG, payload, 4)? as usize;
    let resource_count = read_u32_checked(TAG, payload, 8)? as usize;
    let resource_offset = read_u32_checked(TAG, payload, 12)? as usize;
    let target_minor = read_u8_checked(TAG, payload, 16)?;
    let target_major = read_u8_checked(TAG, payload, 17)?;
    let program_type = u16::from_le_bytes([
        read_u8_checked(TAG, payload, 18)?,
        read_u8_checked(TAG, payload, 19)?,
    ]);
    let creator_offset = read_u32_checked(TAG, payload, 24)?;

    let is_rd11 = payload.get(28..32) == Some(b"RD11".as_slice());
    let binding_stride = if is_rd11 && payload.len() >= RDEF_RD11_HEADER_BYTES {
        let declared = read_u32_checked(TAG, payload, 48)? as usize;
        if declared >= RDEF_BINDING_RD11_BYTES {
            declared
        } else {
            RDEF_BINDING_RD11_BYTES
        }
    } else if is_rd11 {
        RDEF_BINDING_RD11_BYTES
    } else {
        RDEF_BINDING_SM4_BYTES
    };
    let cbuffer_stride = if is_rd11 && payload.len() >= RDEF_RD11_HEADER_BYTES {
        let declared = read_u32_checked(TAG, payload, 36)? as usize;
        if declared >= RDEF_CBUFFER_BYTES {
            declared
        } else {
            RDEF_CBUFFER_BYTES
        }
    } else {
        RDEF_CBUFFER_BYTES
    };

    if cbuffer_count > MAX_TOTAL_DESCRIPTOR_RANGES {
        return Err(DxilReflectError::TooManyDescriptorRanges {
            count: cbuffer_count,
            max: MAX_TOTAL_DESCRIPTOR_RANGES,
        });
    }
    if resource_count > MAX_TOTAL_DESCRIPTOR_RANGES {
        return Err(DxilReflectError::TooManyDescriptorRanges {
            count: resource_count,
            max: MAX_TOTAL_DESCRIPTOR_RANGES,
        });
    }

    let creator = if creator_offset as usize >= RDEF_MIN_HEADER_BYTES {
        read_bounded_c_string(TAG, payload, creator_offset, RDEF_MIN_HEADER_BYTES, 0, MAX_SEMANTIC_NAME_LEN).ok()
    } else {
        None
    };

    let mut cbuffers = Vec::new();
    if cbuffer_count > 0 {
        let total_cb_bytes =
            checked_mul_size(TAG, cbuffer_count, cbuffer_stride, "RDEF cbuffer table overflow")?;
        ensure_slice(TAG, payload, cbuffer_offset, total_cb_bytes)?;
        cbuffers.reserve(cbuffer_count);
        for i in 0..cbuffer_count {
            let delta = checked_mul_size(TAG, i, cbuffer_stride, "RDEF cbuffer offset overflow")?;
            let off = checked_add_offset(TAG, cbuffer_offset, delta, "RDEF cbuffer offset overflow")?;
            let name_off = read_u32_checked(TAG, payload, off)?;
            let name = read_bounded_c_string(
                TAG,
                payload,
                name_off,
                RDEF_MIN_HEADER_BYTES,
                i,
                MAX_SEMANTIC_NAME_LEN,
            )
            .unwrap_or_else(|_| format!("cbuffer_{i}"));
            let variable_count = read_u32_checked(TAG, payload, off + 4)?;
            let size_bytes = read_u32_checked(TAG, payload, off + 12)?;
            let flags = read_u32_checked(TAG, payload, off + 16)?;
            let cb_type = read_u32_checked(TAG, payload, off + 20)?;
            cbuffers.push(RdefCBufferDesc {
                name,
                size_bytes,
                variable_count,
                flags,
                cb_type,
            });
        }
    }

    let mut resources = Vec::new();
    if resource_count > 0 {
        let total_res_bytes = checked_mul_size(
            TAG,
            resource_count,
            binding_stride,
            "RDEF resource binding table overflow",
        )?;
        ensure_slice(TAG, payload, resource_offset, total_res_bytes)?;
        resources.reserve(resource_count);
        for i in 0..resource_count {
            let delta = checked_mul_size(TAG, i, binding_stride, "RDEF resource offset overflow")?;
            let off =
                checked_add_offset(TAG, resource_offset, delta, "RDEF resource offset overflow")?;
            let name_off = read_u32_checked(TAG, payload, off)?;
            let name = read_bounded_c_string(
                TAG,
                payload,
                name_off,
                RDEF_MIN_HEADER_BYTES,
                i,
                MAX_SEMANTIC_NAME_LEN,
            )
            .unwrap_or_else(|_| format!("res_{i}"));
            let input_type = RdefInputType::from_u32(read_u32_checked(TAG, payload, off + 4)?);
            let return_type = read_u32_checked(TAG, payload, off + 8)?;
            let dimension =
                RdefResourceDimension::from_u32(read_u32_checked(TAG, payload, off + 12)?);
            let num_samples = read_u32_checked(TAG, payload, off + 16)?;
            let bind_point = read_u32_checked(TAG, payload, off + 20)?;
            let bind_count = read_u32_checked(TAG, payload, off + 24)?;
            let flags = read_u32_checked(TAG, payload, off + 28)?;
            let (space, id) = if is_rd11 && binding_stride >= RDEF_BINDING_RD11_BYTES {
                (
                    read_u32_checked(TAG, payload, off + 32)?,
                    read_u32_checked(TAG, payload, off + 36)?,
                )
            } else {
                (0, i as u32)
            };
            resources.push(RdefResourceBinding {
                name,
                input_type,
                return_type,
                dimension,
                num_samples,
                bind_point,
                bind_count,
                flags,
                space,
                id,
            });
        }
    }

    Ok(RdefInfo {
        target_major,
        target_minor,
        program_type,
        is_rd11,
        creator,
        cbuffers,
        resources,
    })
}

/// Parse a `STAT` shader statistics chunk cleanly with bounds checks.
pub fn parse_stat_chunk(payload: &[u8]) -> Result<StatInfo, DxilReflectError> {
    const TAG: &str = "STAT";
    if payload.len() < 28 {
        return Err(DxilReflectError::TruncatedChunk {
            fourcc: TAG,
            actual: payload.len(),
            expected_min: 28,
        });
    }
    let opt_u32 = |off: usize| -> Option<u32> { read_u32_checked(TAG, payload, off).ok() };
    Ok(StatInfo {
        instruction_count: read_u32_checked(TAG, payload, 0)?,
        temp_register_count: read_u32_checked(TAG, payload, 4)?,
        def_count: read_u32_checked(TAG, payload, 8)?,
        dcl_count: read_u32_checked(TAG, payload, 12)?,
        float_instruction_count: read_u32_checked(TAG, payload, 16)?,
        int_instruction_count: read_u32_checked(TAG, payload, 20)?,
        uint_instruction_count: read_u32_checked(TAG, payload, 24)?,
        texture_normal_instructions: opt_u32(44),
        texture_load_instructions: opt_u32(48),
        texture_comp_instructions: opt_u32(52),
        texture_bias_instructions: opt_u32(56),
        texture_gradient_instructions: opt_u32(60),
    })
}

fn rdef_resource_layout(
    res: &RdefResourceBinding,
    cbuffer_sizes: &BTreeMap<String, u32>,
) -> (ResourceKind, Option<BindingLayout>, Option<u32>, bool) {
    let sample_type_for_return = |ret: u32, multisampled: bool| -> TextureSampleType {
        match ret {
            3 => TextureSampleType::Sint,
            4 => TextureSampleType::Uint,
            _ if multisampled => TextureSampleType::UnfilterableFloat,
            _ => TextureSampleType::Float,
        }
    };
    match res.input_type {
        RdefInputType::CBuffer => {
            let min_size = cbuffer_sizes.get(&res.name).copied().unwrap_or(0);
            (
                ResourceKind::Buffer,
                Some(BindingLayout::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: u64::from(min_size),
                }),
                (min_size > 0).then_some(min_size),
                false,
            )
        }
        RdefInputType::Sampler => (
            ResourceKind::Sampler,
            Some(BindingLayout::Sampler {
                ty: if res.is_comparison_sampler() {
                    SamplerBindingType::Comparison
                } else {
                    SamplerBindingType::Filtering
                },
            }),
            None,
            false,
        ),
        RdefInputType::Structured | RdefInputType::ByteAddress | RdefInputType::TBuffer => (
            ResourceKind::Buffer,
            Some(BindingLayout::Buffer {
                ty: BufferBindingType::ReadOnlyStorage,
                has_dynamic_offset: false,
                min_binding_size: 0,
            }),
            None,
            false,
        ),
        RdefInputType::Texture => {
            let multisampled = res.dimension.is_multisampled();
            let view_dimension = if multisampled {
                TextureViewDimension::D2
            } else {
                res.dimension.to_view_dimension()
            };
            let sample_type = sample_type_for_return(res.return_type, multisampled);
            (
                ResourceKind::Texture,
                Some(BindingLayout::Texture {
                    sample_type,
                    view_dimension,
                    multisampled,
                }),
                None,
                false,
            )
        }
        RdefInputType::UavRwTyped => {
            let view_dimension = match res.dimension {
                RdefResourceDimension::Texture1D => TextureViewDimension::D1,
                RdefResourceDimension::Texture2DArray | RdefResourceDimension::Texture1DArray => {
                    TextureViewDimension::D2Array
                }
                RdefResourceDimension::Texture3D => TextureViewDimension::D3,
                _ => TextureViewDimension::D2,
            };
            (
                ResourceKind::StorageTexture,
                Some(BindingLayout::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: "rgba8unorm",
                    view_dimension,
                }),
                None,
                true,
            )
        }
        RdefInputType::UavRwStructured
        | RdefInputType::UavRwByteAddress
        | RdefInputType::UavAppendStructured
        | RdefInputType::UavConsumeStructured
        | RdefInputType::UavRwStructuredWithCounter => (
            ResourceKind::Buffer,
            Some(BindingLayout::Buffer {
                ty: BufferBindingType::Storage,
                has_dynamic_offset: false,
                min_binding_size: 0,
            }),
            None,
            true,
        ),
        RdefInputType::RtAccelerationStructure => {
            (ResourceKind::AccelerationStructure, None, None, false)
        }
        RdefInputType::Other(_) => (ResourceKind::Buffer, None, None, false),
    }
}

fn extract_slots_from_rdef(
    rdef: &RdefInfo,
    container_stage: ShaderStage,
) -> Result<Vec<ExtractedSlot>, DxilReflectError> {
    let (vis, d3d_vis) = match container_stage {
        ShaderStage::Vertex => (VISIBILITY_VERTEX, D3dShaderVisibility::Vertex.d3d_stage_mask()),
        ShaderStage::Fragment => (VISIBILITY_FRAGMENT, D3dShaderVisibility::Pixel.d3d_stage_mask()),
        ShaderStage::Compute => (VISIBILITY_COMPUTE, D3dShaderVisibility::All.d3d_stage_mask()),
    };
    let cbuffer_sizes: BTreeMap<String, u32> = rdef
        .cbuffers
        .iter()
        .map(|cb| (cb.name.clone(), cb.size_bytes))
        .collect();

    let mut slots = Vec::new();
    for res in &rdef.resources {
        let Some(class) = res.input_type.register_class() else {
            continue;
        };
        let (resource, layout, _min_size, _writable) = rdef_resource_layout(res, &cbuffer_sizes);
        if res.is_unbounded() {
            slots.push(ExtractedSlot {
                register: D3dRegister {
                    class,
                    index: res.bind_point,
                    space: res.space,
                },
                range_end: u32::MAX,
                d3d_visibility_mask: d3d_vis,
                name: res.name.clone(),
                visibility: vis,
                resource,
                layout: None,
                root_constants: false,
                binding_array: Some(BindingArraySize::Unbounded),
                lowering_need: Some((
                    LoweringKind::UnboundedBindingArray,
                    format!("'{}': unbounded RDEF resource binding", res.name),
                )),
            });
            continue;
        }
        if res.bind_count > MAX_EXPANDED_RANGE_DESCRIPTORS {
            let range_end = res
                .bind_point
                .checked_add(res.bind_count - 1)
                .ok_or_else(|| DxilReflectError::Overflow {
                    fourcc: "RDEF",
                    detail: format!(
                        "RDEF resource '{}' register index overflow (base={}, count={})",
                        res.name, res.bind_point, res.bind_count
                    ),
                })?;
            slots.push(ExtractedSlot {
                register: D3dRegister {
                    class,
                    index: res.bind_point,
                    space: res.space,
                },
                range_end,
                d3d_visibility_mask: d3d_vis,
                name: res.name.clone(),
                visibility: vis,
                resource,
                layout: None,
                root_constants: false,
                binding_array: Some(BindingArraySize::Fixed(res.bind_count)),
                lowering_need: Some((
                    LoweringKind::BindingArray,
                    format!(
                        "'{}': RDEF resource array of {} elements exceeds expansion limit",
                        res.name, res.bind_count
                    ),
                )),
            });
            continue;
        }
        for offset in 0..res.bind_count {
            let reg_idx = res
                .bind_point
                .checked_add(offset)
                .ok_or_else(|| DxilReflectError::Overflow {
                    fourcc: "RDEF",
                    detail: format!(
                        "RDEF resource '{}' register index overflow (base={}, offset={offset})",
                        res.name, res.bind_point
                    ),
                })?;
            let slot_name = if res.bind_count == 1 {
                res.name.clone()
            } else {
                format!("{}_{offset}", res.name)
            };
            slots.push(ExtractedSlot {
                register: D3dRegister {
                    class,
                    index: reg_idx,
                    space: res.space,
                },
                range_end: reg_idx,
                d3d_visibility_mask: d3d_vis,
                name: slot_name,
                visibility: vis,
                resource,
                layout,
                root_constants: false,
                binding_array: None,
                lowering_need: if layout.is_none() {
                    Some((
                        LoweringKind::UnsupportedResource,
                        format!("'{}': unsupported RDEF resource kind", res.name),
                    ))
                } else {
                    None
                },
            });
        }
    }
    Ok(slots)
}

/// One resource binding extracted from DXIL/DXBC fallback metadata (`RTS0`, `PSV0`, `RDEF`, or
/// DXIL bitcode resource metadata) prior to external root-signature matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxilFallbackResource {
    pub name: String,
    pub class: RegisterClass,
    pub base_register: u32,
    pub register_space: u32,
    /// `Some(count)` for bounded resources, `None` for unbounded bindless arrays.
    pub count: Option<u32>,
    pub resource_kind: ResourceKind,
    pub layout: Option<BindingLayout>,
    pub min_binding_size_bytes: Option<u32>,
    pub is_comparison_sampler: bool,
    pub is_writable: bool,
    /// Chunk tag from which this resource was extracted (`"RTS0"`, `"PSV0"`, `"RDEF"`).
    pub source_chunk: &'static str,
}

/// Complete DXIL fallback reflection extracted from a DXBC/DXIL container (even when native
/// RDNA2 ISA and embedded `RTS0` are absent, as in `LEGAL NOTICES -> Spinner` pairs in `#482`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxilFallbackReflection {
    pub hash: u64,
    pub container_digest: String,
    pub shader_kind: Option<DxilShaderKind>,
    pub stage: ShaderStage,
    pub shader_model: Option<(u8, u8)>,
    pub has_embedded_rts0: bool,
    pub has_dxil_bitcode: bool,
    pub resources: Vec<DxilFallbackResource>,
    pub input_signature: Option<StageSignature>,
    pub output_signature: Option<StageSignature>,
    pub vertex_inputs: Vec<ShaderIo>,
    pub fragment_outputs: Vec<ShaderIo>,
    pub workgroup_size: Option<[u32; 3]>,
    pub rdef: Option<RdefInfo>,
    pub psv: Option<PsvInfo>,
    pub stat: Option<StatInfo>,
}

/// Extract fallback reflection metadata (`RDEF`/`STAT`/`ISG1`/`ISGN`/`OSG1`/`OSGN`/`PSV0`/
/// optional `RTS0` and `DXIL` program header) from a DXBC/DXIL container without requiring an
/// embedded `RTS0` chunk or native RDNA2 ISA.
pub fn extract_dxil_fallback_reflection(
    bytes: &[u8],
) -> Result<DxilFallbackReflection, DxilReflectError> {
    let info = parse_dxbc(bytes)?;
    let chunk_refs = parse_dxbc_chunks(bytes)?;

    let mut root_signature: Option<RootSignature> = None;
    let mut isg1: Option<StageSignature> = None;
    let mut isgn: Option<StageSignature> = None;
    let mut osg1: Option<StageSignature> = None;
    let mut osgn: Option<StageSignature> = None;
    let mut psv: Option<PsvInfo> = None;
    let mut rdef: Option<RdefInfo> = None;
    let mut stat: Option<StatInfo> = None;

    for chunk in chunk_refs {
        match &chunk.fourcc {
            b"DXIL" => validate_dxil_chunk(chunk.payload)?,
            b"SHDR" => validate_legacy_shader_chunk("SHDR", chunk.payload)?,
            b"SHEX" => validate_legacy_shader_chunk("SHEX", chunk.payload)?,
            b"RTS0" => {
                if root_signature.is_none() {
                    root_signature = Some(parse_rts0_chunk(chunk.payload)?);
                }
            }
            b"ISG1" => {
                if isg1.is_none() {
                    isg1 = Some(parse_signature_elements(chunk.fourcc, chunk.payload)?);
                }
            }
            b"ISGN" => {
                if isgn.is_none() {
                    isgn = Some(parse_signature_elements(chunk.fourcc, chunk.payload)?);
                }
            }
            b"OSG1" => {
                if osg1.is_none() {
                    osg1 = Some(parse_signature_elements(chunk.fourcc, chunk.payload)?);
                }
            }
            b"OSGN" => {
                if osgn.is_none() {
                    osgn = Some(parse_signature_elements(chunk.fourcc, chunk.payload)?);
                }
            }
            b"PSV0" => {
                if psv.is_none() {
                    psv = Some(parse_psv0_chunk(chunk.payload)?);
                }
            }
            b"RDEF" => {
                if rdef.is_none() {
                    rdef = Some(parse_rdef_chunk(chunk.payload)?);
                }
            }
            b"STAT" => {
                if stat.is_none() {
                    stat = parse_stat_chunk(chunk.payload).ok();
                }
            }
            _ => {}
        }
    }

    let input_signature = isg1.or(isgn);
    let output_signature = osg1.or(osgn);
    let shader_kind = info
        .shader_kind()
        .or_else(|| psv.as_ref().and_then(|p| p.shader_stage))
        .or_else(|| {
            rdef.as_ref()
                .map(|r| DxilShaderKind::from_u16(r.program_type))
        });
    let shader_model = info.shader_model().or_else(|| {
        rdef.as_ref()
            .map(|r| (r.target_major, r.target_minor))
    });
    let (stage, _) = infer_webgpu_stage(
        shader_kind,
        root_signature.as_ref(),
        input_signature.as_ref(),
        output_signature.as_ref(),
    );

    let (vertex_inputs, fragment_outputs, _) =
        lower_stage_io(stage, input_signature.as_ref(), output_signature.as_ref());

    let workgroup_size = if stage == ShaderStage::Compute {
        psv.as_ref().and_then(|p| p.workgroup_size)
    } else {
        None
    };

    // Merge resource definitions cleanly: prefer RDEF names/sizes/dimensions when present,
    // supplement with PSV0 entries for any register not covered by RDEF, and include RTS0
    // entries when neither PSV0 nor RDEF supplied resources.
    let mut by_key: BTreeMap<(u32, RegisterClass, u32), DxilFallbackResource> = BTreeMap::new();

    if let Some(rdef_info) = &rdef {
        let cb_sizes: BTreeMap<String, u32> = rdef_info
            .cbuffers
            .iter()
            .map(|cb| (cb.name.clone(), cb.size_bytes))
            .collect();
        for res in &rdef_info.resources {
            let Some(class) = res.input_type.register_class() else {
                continue;
            };
            let (resource_kind, layout, min_binding_size_bytes, is_writable) =
                rdef_resource_layout(res, &cb_sizes);
            let count = if res.is_unbounded() {
                None
            } else {
                Some(res.bind_count)
            };
            by_key.insert(
                (res.space, class, res.bind_point),
                DxilFallbackResource {
                    name: res.name.clone(),
                    class,
                    base_register: res.bind_point,
                    register_space: res.space,
                    count,
                    resource_kind,
                    layout: if count == Some(1) { layout } else { None },
                    min_binding_size_bytes,
                    is_comparison_sampler: res.is_comparison_sampler(),
                    is_writable,
                    source_chunk: "RDEF",
                },
            );
        }
    }

    if let Some(psv_info) = &psv {
        for res in &psv_info.resources {
            let range_type = match res.res_type {
                1 => DescriptorRangeType::Sampler,
                2 => DescriptorRangeType::Cbv,
                3 | 4 | 5 | 10 => DescriptorRangeType::Srv,
                6 | 7 | 8 | 9 | 11 => DescriptorRangeType::Uav,
                _ => continue,
            };
            let class = range_type.register_class();
            let key = (res.space, class, res.lower_bound);
            if by_key.contains_key(&key) {
                continue;
            }
            let count = if res.upper_bound == u32::MAX {
                None
            } else {
                res.upper_bound
                    .checked_sub(res.lower_bound)
                    .and_then(|d| d.checked_add(1))
            };
            let (resource_kind, layout) = classify_table_range_resource(
                range_type,
                res.lower_bound,
                res.space,
                &psv_info.resources,
            );
            let is_writable = matches!(range_type, DescriptorRangeType::Uav);
            by_key.insert(
                key,
                DxilFallbackResource {
                    name: format!("psv_{}{}_space{}", class.as_str(), res.lower_bound, res.space),
                    class,
                    base_register: res.lower_bound,
                    register_space: res.space,
                    count,
                    resource_kind,
                    layout: if count == Some(1) { Some(layout) } else { None },
                    min_binding_size_bytes: None,
                    is_comparison_sampler: false,
                    is_writable,
                    source_chunk: "PSV0",
                },
            );
        }
    }

    if by_key.is_empty()
        && let Some(rs) = &root_signature
    {
        let psv_res = psv.as_ref().map(|p| p.resources.as_slice()).unwrap_or(&[]);
        for slot in extract_slots_from_rts0(rs, stage, shader_kind, psv_res)? {
            let count = match slot.binding_array {
                Some(BindingArraySize::Unbounded) => None,
                Some(BindingArraySize::Fixed(n)) => Some(n),
                None => Some(1),
            };
            let min_size = match slot.layout {
                Some(BindingLayout::Buffer {
                    min_binding_size, ..
                }) if min_binding_size > 0 => u32::try_from(min_binding_size).ok(),
                _ => None,
            };
            let is_writable = matches!(
                slot.layout,
                Some(
                    BindingLayout::Buffer {
                        ty: BufferBindingType::Storage,
                        ..
                    } | BindingLayout::StorageTexture { .. }
                )
            ) || slot.register.class == RegisterClass::U;
            let is_comparison_sampler = matches!(
                slot.layout,
                Some(BindingLayout::Sampler {
                    ty: SamplerBindingType::Comparison
                })
            );
            by_key.entry((slot.register.space, slot.register.class, slot.register.index)).or_insert(
                DxilFallbackResource {
                    name: slot.name,
                    class: slot.register.class,
                    base_register: slot.register.index,
                    register_space: slot.register.space,
                    count,
                    resource_kind: slot.resource,
                    layout: slot.layout,
                    min_binding_size_bytes: min_size,
                    is_comparison_sampler,
                    is_writable,
                    source_chunk: "RTS0",
                },
            );
        }
    }

    Ok(DxilFallbackReflection {
        hash: info.fnv1a_hash,
        container_digest: info.digest_hex(),
        shader_kind,
        stage,
        shader_model,
        has_embedded_rts0: root_signature.is_some(),
        has_dxil_bitcode: info.has_llvm_bitcode(),
        resources: by_key.into_values().collect(),
        input_signature,
        output_signature,
        vertex_inputs,
        fragment_outputs,
        workgroup_size,
        rdef,
        psv,
        stat,
    })
}

/// Default console user-data SGPR base for a pipeline stage (`s8` graphics, `s4` compute,
/// `s16` mesh-compute; matching `runtime/gpu/stage_reflection.h`).
pub const fn default_user_data_base(
    stage: ShaderStage,
    shader_kind: Option<DxilShaderKind>,
) -> u32 {
    if matches!(shader_kind, Some(DxilShaderKind::Mesh)) {
        return CONSOLE_MESH_USER_DATA_BASE;
    }
    match stage {
        ShaderStage::Compute => CONSOLE_COMPUTE_USER_DATA_BASE,
        ShaderStage::Vertex | ShaderStage::Fragment => CONSOLE_GRAPHICS_USER_DATA_BASE,
    }
}

/// How a DXIL fallback resource matched an external D3D12 [`RootSignature`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DxilRootParamMatchKind {
    RootConstants {
        param_index: usize,
        user_sgpr: u32,
        num_32bit_values: u32,
    },
    RootDescriptor {
        param_index: usize,
        user_sgpr: u32,
    },
    DescriptorTableSlot {
        param_index: usize,
        range_index: usize,
        user_sgpr: u32,
        table_slot: u32,
        table_byte_offset: u32,
        is_unbounded: bool,
    },
    StaticSampler {
        static_sampler_index: u32,
    },
}

/// One DXIL resource matched to an external D3D12 root-signature parameter and lowered to a
/// `playgta6-stage-reflection` v3 [`SidecarBindingSource`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxilMatchedBinding {
    pub resource: DxilFallbackResource,
    pub webgpu_group: u32,
    pub webgpu_binding: u32,
    pub match_kind: DxilRootParamMatchKind,
    pub sidecar_source: SidecarBindingSource,
}

/// Complete result of matching a DXIL-only shader's fallback reflection against an external
/// D3D12 [`RootSignature`] (producing joinable `playgta6-stage-reflection` v3 entries for
/// `runtime/gpu/`, Refs `#482`, `#651`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxilRootMatchReport {
    pub hash: u64,
    pub stage: ShaderStage,
    pub user_data_base: u32,
    pub matched: Vec<DxilMatchedBinding>,
    pub unmatched_readonly: Vec<DxilFallbackResource>,
    pub unmatched_writable: Vec<DxilFallbackResource>,
    pub sidecar_entry: SidecarStageEntry,
    pub pipeline_layout: PipelineLayoutDesc,
    pub bind_group_remaps: Vec<DxilBindGroupRemap>,
}

impl DxilRootMatchReport {
    /// `true` when every writable binding matched a root parameter (so the stage is safe to join
    /// in `runtime/gpu/stage_reflection.cpp`, with read-only unmatched bindings using v3 `unresolved`).
    pub fn is_joinable_v3(&self) -> bool {
        self.unmatched_writable.is_empty()
    }
}

fn d3d_vis_covers_stage(vis: D3dShaderVisibility, stage: ShaderStage) -> bool {
    match (vis, stage) {
        (D3dShaderVisibility::All, _) => true,
        (D3dShaderVisibility::Vertex, ShaderStage::Vertex) => true,
        (D3dShaderVisibility::Pixel, ShaderStage::Fragment) => true,
        (D3dShaderVisibility::Mesh | D3dShaderVisibility::Amplification, ShaderStage::Compute) => true,
        _ => false,
    }
}

fn fallback_resource_to_sidecar_kind(res: &DxilFallbackResource) -> SidecarBindingKind {
    match res.layout {
        Some(BindingLayout::Buffer {
            ty: BufferBindingType::Uniform,
            ..
        }) => SidecarBindingKind::Uniform,
        Some(BindingLayout::Buffer {
            ty: BufferBindingType::Storage,
            ..
        }) => SidecarBindingKind::Storage,
        Some(BindingLayout::Buffer {
            ty: BufferBindingType::ReadOnlyStorage,
            ..
        }) => SidecarBindingKind::ReadOnlyStorage,
        Some(BindingLayout::Sampler {
            ty: SamplerBindingType::Comparison,
        }) => SidecarBindingKind::ComparisonSampler,
        Some(BindingLayout::Sampler { .. }) => SidecarBindingKind::Sampler,
        Some(BindingLayout::Texture { .. }) => SidecarBindingKind::Texture,
        Some(BindingLayout::StorageTexture { .. }) => SidecarBindingKind::StorageTexture,
        None => match res.class {
            RegisterClass::B => SidecarBindingKind::Uniform,
            RegisterClass::S => {
                if res.is_comparison_sampler {
                    SidecarBindingKind::ComparisonSampler
                } else {
                    SidecarBindingKind::Sampler
                }
            }
            RegisterClass::U => {
                if res.resource_kind == ResourceKind::StorageTexture {
                    SidecarBindingKind::StorageTexture
                } else {
                    SidecarBindingKind::Storage
                }
            }
            RegisterClass::T => {
                if res.resource_kind == ResourceKind::Buffer {
                    SidecarBindingKind::ReadOnlyStorage
                } else {
                    SidecarBindingKind::Texture
                }
            }
        },
    }
}

fn signature_to_sidecar_vertex_inputs(sig: Option<&StageSignature>) -> Vec<WgslVertexInputInfo> {
    let Some(isig) = sig else {
        return Vec::new();
    };
    let mut ordered: Vec<&SignatureElement> = isig
        .elements
        .iter()
        .filter(|e| !e.is_system_value() && e.component_count() > 0)
        .collect();
    ordered.sort_by_key(|e| (e.register, e.semantic_index, e.semantic_name.clone()));

    let mut out = Vec::new();
    let mut byte_offset = 0u32;
    let mut seen = BTreeSet::new();
    for elem in ordered {
        if !seen.insert(elem.register) {
            continue;
        }
        let comps = elem.component_count() as u8;
        let component_type = match elem.component_type {
            SignatureComponentType::Uint32 => "u32",
            SignatureComponentType::Sint32 => "i32",
            _ => "f32",
        };
        let (_, fetch_format) = elem.to_wgsl_io_type();
        out.push(WgslVertexInputInfo {
            location: elem.register,
            vb_slot: 0,
            byte_offset,
            components: comps,
            component_type,
            fetch_format,
        });
        byte_offset = byte_offset.saturating_add(u32::from(comps) * 4);
    }
    out
}

/// Match DXIL fallback resource bindings `(register_space, base_register, class)` against an
/// external D3D12 [`RootSignature`] (root constants, root descriptors, descriptor table ranges,
/// and static samplers) and emit a `playgta6-stage-reflection` v3 [`SidecarStageEntry`] plus
/// WebGPU [`PipelineLayoutDesc`].
pub fn match_dxil_to_root_signature(
    fallback: &DxilFallbackReflection,
    rs: &RootSignature,
    convention: &RegisterConvention,
) -> Result<DxilRootMatchReport, DxilReflectError> {
    let user_data_base = default_user_data_base(fallback.stage, fallback.shader_kind);

    // Precompute console user-data SGPR starting index for each root parameter.
    let mut param_sgprs: Vec<u32> = Vec::with_capacity(rs.parameters.len());
    let mut next_sgpr = user_data_base;
    for param in &rs.parameters {
        param_sgprs.push(next_sgpr);
        let width = match &param.kind {
            RootParameterKind::DescriptorTable(_) => 1,
            RootParameterKind::Constants32Bit {
                num_32bit_values, ..
            } => (*num_32bit_values).max(1),
            RootParameterKind::Cbv { .. }
            | RootParameterKind::Srv { .. }
            | RootParameterKind::Uav { .. } => 2,
        };
        next_sgpr = next_sgpr.saturating_add(width);
    }

    // Lower fallback resources to WebGPU group/binding indices deterministically.
    let (vis_bit, d3d_vis) = match fallback.stage {
        ShaderStage::Vertex => (VISIBILITY_VERTEX, D3dShaderVisibility::Vertex.d3d_stage_mask()),
        ShaderStage::Fragment => (VISIBILITY_FRAGMENT, D3dShaderVisibility::Pixel.d3d_stage_mask()),
        ShaderStage::Compute => (VISIBILITY_COMPUTE, D3dShaderVisibility::All.d3d_stage_mask()),
    };

    let mut raw_slots = Vec::with_capacity(fallback.resources.len());
    for res in &fallback.resources {
        let (range_end, binding_array, lowering_need) = match res.count {
            None => (
                u32::MAX,
                Some(BindingArraySize::Unbounded),
                Some((
                    LoweringKind::UnboundedBindingArray,
                    format!("'{}': unbounded DXIL fallback resource", res.name),
                )),
            ),
            Some(c) if c > MAX_EXPANDED_RANGE_DESCRIPTORS => (
                res.base_register.saturating_add(c.saturating_sub(1)),
                Some(BindingArraySize::Fixed(c)),
                Some((
                    LoweringKind::BindingArray,
                    format!("'{}': large DXIL fallback array ({c} elements)", res.name),
                )),
            ),
            Some(c) => (res.base_register.saturating_add(c.saturating_sub(1)), None, None),
        };
        raw_slots.push(ExtractedSlot {
            register: D3dRegister {
                class: res.class,
                index: res.base_register,
                space: res.register_space,
            },
            range_end,
            d3d_visibility_mask: d3d_vis,
            name: res.name.clone(),
            visibility: vis_bit,
            resource: res.resource_kind,
            layout: res.layout,
            root_constants: false,
            binding_array,
            lowering_need,
        });
    }

    let (reflected_bindings, pipeline_layout, bind_group_remaps, _) =
        lower_slots_to_webgpu(raw_slots, convention)?;

    // Build lookup from D3D register to assigned WebGPU (group, binding).
    let mut reg_to_wgpu: BTreeMap<(u32, RegisterClass, u32), (u32, u32)> = BTreeMap::new();
    for b in &reflected_bindings {
        if let Some(reg) = b.register {
            reg_to_wgpu.insert((reg.space, reg.class, reg.index), (b.group, b.binding));
        }
    }

    let mut matched = Vec::new();
    let mut unmatched_readonly = Vec::new();
    let mut unmatched_writable = Vec::new();
    let mut sidecar_bindings = Vec::new();

    for res in &fallback.resources {
        let &(webgpu_group, webgpu_binding) = reg_to_wgpu
            .get(&(res.register_space, res.class, res.base_register))
            .unwrap_or(&(res.register_space.min(WEBGPU_MAX_BIND_GROUPS - 1), res.base_register));
        let kind = fallback_resource_to_sidecar_kind(res);
        let (view_dimension, sample_type, multisampled) = match res.layout {
            Some(BindingLayout::Texture {
                sample_type,
                view_dimension,
                multisampled,
            }) => (
                Some(view_dimension.as_str()),
                Some(match sample_type {
                    TextureSampleType::Float => "float",
                    TextureSampleType::UnfilterableFloat => "unfilterable_float",
                    TextureSampleType::Depth => "depth",
                    TextureSampleType::Sint => "sint",
                    TextureSampleType::Uint => "uint",
                }),
                multisampled,
            ),
            Some(BindingLayout::StorageTexture { view_dimension, .. }) => {
                (Some(view_dimension.as_str()), None, false)
            }
            _ => (None, None, false),
        };

        let found_match = find_matching_root_param(res, fallback.stage, rs, &param_sgprs);
        match found_match {
            Some((match_kind, mut sidecar_source, root_const_min_bytes)) => {
                // If an unbounded DXIL bindless resource matched a table range, represent it with
                // a validated GFX10.3 lane-indexed heap source (stride 16B sampler / 32B resource).
                if res.count.is_none()
                    && let DxilRootParamMatchKind::DescriptorTableSlot { .. } = match_kind
                {
                    if res.class == RegisterClass::S {
                        sidecar_source =
                            SidecarBindingSource::LaneIndexedSamplerHeap(LaneIndexedHeapSource {
                                max_count: DEFAULT_LANE_INDEXED_WINDOW,
                                shift: 4,
                                stride: GFX10_SAMPLER_STRIDE_BYTES,
                                window_base: None,
                            });
                    } else {
                        sidecar_source =
                            SidecarBindingSource::LaneIndexedResourceHeap(LaneIndexedHeapSource {
                                max_count: DEFAULT_LANE_INDEXED_WINDOW,
                                shift: 5,
                                stride: GFX10_IMAGE_STRIDE_BYTES,
                                window_base: None,
                            });
                    }
                }
                let min_bytes = res.min_binding_size_bytes.or(root_const_min_bytes);
                sidecar_bindings.push(SidecarBinding {
                    group: webgpu_group,
                    binding: webgpu_binding,
                    kind,
                    view_dimension,
                    sample_type,
                    multisampled,
                    storage_format: None,
                    min_binding_size_bytes: min_bytes,
                    source: sidecar_source.clone(),
                });
                matched.push(DxilMatchedBinding {
                    resource: res.clone(),
                    webgpu_group,
                    webgpu_binding,
                    match_kind,
                    sidecar_source,
                });
            }
            None => {
                if kind.allows_unresolved() {
                    unmatched_readonly.push(res.clone());
                    sidecar_bindings.push(SidecarBinding {
                        group: webgpu_group,
                        binding: webgpu_binding,
                        kind,
                        view_dimension,
                        sample_type,
                        multisampled,
                        storage_format: None,
                        min_binding_size_bytes: res.min_binding_size_bytes,
                        source: SidecarBindingSource::Unresolved {
                            reason: Some("dxil_unmatched_root_param".to_string()),
                        },
                    });
                } else {
                    unmatched_writable.push(res.clone());
                }
            }
        }
    }

    sidecar_bindings.sort_by_key(|b| (b.group, b.binding));
    let vertex_inputs = if fallback.stage == ShaderStage::Vertex {
        signature_to_sidecar_vertex_inputs(fallback.input_signature.as_ref())
    } else {
        Vec::new()
    };

    let sidecar_entry = SidecarStageEntry {
        hash: fallback.hash,
        stage: fallback.stage,
        bindings: sidecar_bindings,
        vertex_inputs,
        vs_outputs: Vec::new(),
        ps_inputs: Vec::new(),
        mesh_output: None,
        rt_arena: false,
    };

    Ok(DxilRootMatchReport {
        hash: fallback.hash,
        stage: fallback.stage,
        user_data_base,
        matched,
        unmatched_readonly,
        unmatched_writable,
        sidecar_entry,
        pipeline_layout,
        bind_group_remaps,
    })
}

fn find_matching_root_param(
    res: &DxilFallbackResource,
    stage: ShaderStage,
    rs: &RootSignature,
    param_sgprs: &[u32],
) -> Option<(DxilRootParamMatchKind, SidecarBindingSource, Option<u32>)> {
    // 1. Check root parameters (root constants, root descriptors, descriptor tables).
    for (param_index, param) in rs.parameters.iter().enumerate() {
        if !d3d_vis_covers_stage(param.visibility, stage) {
            continue;
        }
        let user_sgpr = param_sgprs[param_index];
        match &param.kind {
            RootParameterKind::Constants32Bit {
                shader_register,
                register_space,
                num_32bit_values,
            } => {
                if res.class == RegisterClass::B
                    && res.register_space == *register_space
                    && res.base_register == *shader_register
                {
                    let min_bytes = num_32bit_values.saturating_mul(4);
                    return Some((
                        DxilRootParamMatchKind::RootConstants {
                            param_index,
                            user_sgpr,
                            num_32bit_values: *num_32bit_values,
                        },
                        SidecarBindingSource::RootConstants { user_sgpr },
                        (min_bytes > 0).then_some(min_bytes),
                    ));
                }
            }
            RootParameterKind::Cbv {
                shader_register,
                register_space,
                ..
            } => {
                if res.class == RegisterClass::B
                    && res.register_space == *register_space
                    && res.base_register == *shader_register
                {
                    return Some((
                        DxilRootParamMatchKind::RootDescriptor {
                            param_index,
                            user_sgpr,
                        },
                        SidecarBindingSource::RootDescriptor {
                            user_sgpr,
                            table_byte_offset: None,
                        },
                        None,
                    ));
                }
            }
            RootParameterKind::Srv {
                shader_register,
                register_space,
                ..
            } => {
                if res.class == RegisterClass::T
                    && res.register_space == *register_space
                    && res.base_register == *shader_register
                {
                    return Some((
                        DxilRootParamMatchKind::RootDescriptor {
                            param_index,
                            user_sgpr,
                        },
                        SidecarBindingSource::RootDescriptor {
                            user_sgpr,
                            table_byte_offset: None,
                        },
                        None,
                    ));
                }
            }
            RootParameterKind::Uav {
                shader_register,
                register_space,
                ..
            } => {
                if res.class == RegisterClass::U
                    && res.register_space == *register_space
                    && res.base_register == *shader_register
                {
                    return Some((
                        DxilRootParamMatchKind::RootDescriptor {
                            param_index,
                            user_sgpr,
                        },
                        SidecarBindingSource::RootDescriptor {
                            user_sgpr,
                            table_byte_offset: None,
                        },
                        None,
                    ));
                }
            }
            RootParameterKind::DescriptorTable(ranges) => {
                let mut append_slot = 0u32;
                for (range_index, range) in ranges.iter().enumerate() {
                    let range_start_slot = if range.offset_in_descriptors_from_table_start
                        == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
                    {
                        append_slot
                    } else {
                        range.offset_in_descriptors_from_table_start
                    };
                    let is_unbounded = range.num_descriptors == u32::MAX;
                    let class_matches = range.range_type.register_class() == res.class;
                    let space_matches = range.register_space == res.register_space;
                    let reg_matches = if !class_matches || !space_matches || range.num_descriptors == 0 {
                        false
                    } else if is_unbounded {
                        res.base_register >= range.base_shader_register
                    } else {
                        let range_end = range
                            .base_shader_register
                            .saturating_add(range.num_descriptors - 1);
                        res.base_register >= range.base_shader_register
                            && res.base_register <= range_end
                    };

                    if reg_matches {
                        let delta = res.base_register - range.base_shader_register;
                        let table_slot = range_start_slot.saturating_add(delta);
                        let table_byte_offset =
                            table_slot.saturating_mul(CONSOLE_TABLE_SLOT_BYTES);
                        return Some((
                            DxilRootParamMatchKind::DescriptorTableSlot {
                                param_index,
                                range_index,
                                user_sgpr,
                                table_slot,
                                table_byte_offset,
                                is_unbounded,
                            },
                            SidecarBindingSource::Table {
                                user_sgpr,
                                table_byte_offset,
                            },
                            None,
                        ));
                    }

                    if !is_unbounded {
                        append_slot = range_start_slot.saturating_add(range.num_descriptors);
                    }
                }
            }
        }
    }

    // 2. Check static samplers for `s#` registers.
    if res.class == RegisterClass::S {
        for (idx, sampler) in rs.static_samplers.iter().enumerate() {
            if d3d_vis_covers_stage(sampler.shader_visibility, stage)
                && sampler.register_space == res.register_space
                && sampler.shader_register == res.base_register
            {
                let static_sampler_index = idx as u32;
                return Some((
                    DxilRootParamMatchKind::StaticSampler {
                        static_sampler_index,
                    },
                    SidecarBindingSource::StaticSampler {
                        index: static_sampler_index,
                    },
                    None,
                ));
            }
        }
    }

    None
}

/// Convenience helper: extract DXIL fallback reflection from `dxil_bytes` and match it against a
/// serialized external `RTS0` root signature payload.
pub fn match_dxil_bytes_to_external_rts0(
    dxil_bytes: &[u8],
    rts0_payload: &[u8],
    convention: &RegisterConvention,
) -> Result<DxilRootMatchReport, DxilReflectError> {
    let fallback = extract_dxil_fallback_reflection(dxil_bytes)?;
    let rs = parse_rts0_chunk(rts0_payload)?;
    match_dxil_to_root_signature(&fallback, &rs, convention)
}

/// Reflect a DXIL-only container without embedded `RTS0` by supplying an external [`RootSignature`]
/// (e.g. from `playgta6-pipeline-dump`'s `manifest.json` or standalone `RTS0` blob).
pub fn reflect_dxil_bytes_with_external_root_signature(
    dxil_bytes: &[u8],
    rs: &RootSignature,
    source: Option<&str>,
    convention: &RegisterConvention,
) -> Result<DxilShaderReflection, DxilReflectError> {
    let mut base = reflect_dxil_bytes_with_options(dxil_bytes, source, convention)?;
    if base.root_signature.is_some() {
        return Ok(base);
    }
    let psv_resources: &[PsvResourceBinding] = base
        .psv
        .as_ref()
        .map(|p| p.resources.as_slice())
        .unwrap_or(&[]);
    let extracted = extract_slots_from_rts0(rs, base.entry.stage, base.shader_kind, psv_resources)?;
    let (bindings, pipeline_layout, bind_group_remaps, slot_lowering) =
        lower_slots_to_webgpu(extracted, convention)?;
    base.root_signature = Some(rs.clone());
    base.entry.bindings = bindings;
    base.entry.requires_lowering.extend(slot_lowering);
    base.entry.requires_lowering.sort();
    base.entry.requires_lowering.dedup();
    base.pipeline_layout = pipeline_layout;
    base.bind_group_remaps = bind_group_remaps;
    base.limit_violations = check_pipeline_limits(&base.pipeline_layout, &[&base.entry]);
    Ok(base)
}

// ---------------------------------------------------------------------------
// Synthetic DXBC / RTS0 / ISG1 / OSG1 Builders & Comprehensive Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod test_builders {
    use super::*;

    /// Build a synthetic valid DXBC container from `(fourcc, payload)` pairs.
    pub(crate) fn build_container(chunks: &[([u8; 4], &[u8])]) -> Vec<u8> {
        crate::dxbc::test_dxbc::build_dxbc_container(chunks)
    }

    /// Build a synthetic `DXIL` chunk payload.
    pub(crate) fn build_dxil_chunk(kind: u16, sm_major: u8, sm_minor: u8) -> Vec<u8> {
        crate::dxbc::test_dxbc::build_dxil_chunk_payload(kind, sm_major, sm_minor, 1, 6, true)
    }

    /// Serialize a [`RootSignature`] into a valid binary `RTS0` chunk payload (v1.0 or v1.1).
    pub(crate) fn encode_rts0(rs: &RootSignature) -> Vec<u8> {
        let version: u32 = match rs.version {
            RootSignatureVersion::V1_0 => 1,
            RootSignatureVersion::V1_1 => 2,
        };
        let params_offset = RTS0_HEADER_BYTES;
        let param_table_bytes = rs.parameters.len() * RTS0_PARAM_HEADER_BYTES;
        let mut body = Vec::new();
        let body_base = params_offset + param_table_bytes;

        let mut param_headers: Vec<(u32, u32, u32)> = Vec::with_capacity(rs.parameters.len());

        for p in &rs.parameters {
            let vis = p.visibility as u32;
            let data_off = (body_base + body.len()) as u32;
            match &p.kind {
                RootParameterKind::DescriptorTable(ranges) => {
                    let ranges_off = (body_base + body.len() + 8) as u32;
                    body.extend_from_slice(&(ranges.len() as u32).to_le_bytes());
                    body.extend_from_slice(&ranges_off.to_le_bytes());
                    for r in ranges {
                        body.extend_from_slice(&(r.range_type as u32).to_le_bytes());
                        body.extend_from_slice(&r.num_descriptors.to_le_bytes());
                        body.extend_from_slice(&r.base_shader_register.to_le_bytes());
                        body.extend_from_slice(&r.register_space.to_le_bytes());
                        if rs.version == RootSignatureVersion::V1_1 {
                            body.extend_from_slice(&r.flags.to_le_bytes());
                        }
                        body.extend_from_slice(
                            &r.offset_in_descriptors_from_table_start.to_le_bytes(),
                        );
                    }
                    param_headers.push((0, vis, data_off));
                }
                RootParameterKind::Constants32Bit {
                    shader_register,
                    register_space,
                    num_32bit_values,
                } => {
                    body.extend_from_slice(&shader_register.to_le_bytes());
                    body.extend_from_slice(&register_space.to_le_bytes());
                    body.extend_from_slice(&num_32bit_values.to_le_bytes());
                    param_headers.push((1, vis, data_off));
                }
                RootParameterKind::Cbv {
                    shader_register,
                    register_space,
                    flags,
                } => {
                    body.extend_from_slice(&shader_register.to_le_bytes());
                    body.extend_from_slice(&register_space.to_le_bytes());
                    if rs.version == RootSignatureVersion::V1_1 {
                        body.extend_from_slice(&flags.to_le_bytes());
                    }
                    param_headers.push((2, vis, data_off));
                }
                RootParameterKind::Srv {
                    shader_register,
                    register_space,
                    flags,
                } => {
                    body.extend_from_slice(&shader_register.to_le_bytes());
                    body.extend_from_slice(&register_space.to_le_bytes());
                    if rs.version == RootSignatureVersion::V1_1 {
                        body.extend_from_slice(&flags.to_le_bytes());
                    }
                    param_headers.push((3, vis, data_off));
                }
                RootParameterKind::Uav {
                    shader_register,
                    register_space,
                    flags,
                } => {
                    body.extend_from_slice(&shader_register.to_le_bytes());
                    body.extend_from_slice(&register_space.to_le_bytes());
                    if rs.version == RootSignatureVersion::V1_1 {
                        body.extend_from_slice(&flags.to_le_bytes());
                    }
                    param_headers.push((4, vis, data_off));
                }
            }
        }

        let samplers_offset = body_base + body.len();
        for s in &rs.static_samplers {
            body.extend_from_slice(&s.filter.to_le_bytes());
            body.extend_from_slice(&s.address_u.to_le_bytes());
            body.extend_from_slice(&s.address_v.to_le_bytes());
            body.extend_from_slice(&s.address_w.to_le_bytes());
            body.extend_from_slice(&s.mip_lod_bias_bits.to_le_bytes());
            body.extend_from_slice(&s.max_anisotropy.to_le_bytes());
            body.extend_from_slice(&s.comparison_func.to_le_bytes());
            body.extend_from_slice(&s.border_color.to_le_bytes());
            body.extend_from_slice(&s.min_lod_bits.to_le_bytes());
            body.extend_from_slice(&s.max_lod_bits.to_le_bytes());
            body.extend_from_slice(&s.shader_register.to_le_bytes());
            body.extend_from_slice(&s.register_space.to_le_bytes());
            body.extend_from_slice(&(s.shader_visibility as u32).to_le_bytes());
        }

        let mut out = Vec::with_capacity(body_base + body.len());
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&(rs.parameters.len() as u32).to_le_bytes());
        out.extend_from_slice(&(params_offset as u32).to_le_bytes());
        out.extend_from_slice(&(rs.static_samplers.len() as u32).to_le_bytes());
        out.extend_from_slice(&(samplers_offset as u32).to_le_bytes());
        out.extend_from_slice(&rs.flags.to_le_bytes());

        for (ty, vis, off) in param_headers {
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(&vis.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
        }
        out.extend_from_slice(&body);
        out
    }

    /// Serialize signature elements into an `ISG1`/`OSG1` (v1, 32B stride) or `ISGN`/`OSGN` (24B stride) chunk.
    pub(crate) fn encode_signature(is_v1: bool, elements: &[SignatureElement]) -> Vec<u8> {
        let stride = if is_v1 {
            SIG_V1_ELEMENT_BYTES
        } else {
            SIG_V0_ELEMENT_BYTES
        };
        let strings_base = SIG_HEADER_BYTES + elements.len() * stride;
        let mut string_pool = Vec::new();
        let mut name_offsets = Vec::with_capacity(elements.len());
        for e in elements {
            name_offsets.push((strings_base + string_pool.len()) as u32);
            string_pool.extend_from_slice(e.semantic_name.as_bytes());
            string_pool.push(0);
        }

        let mut out = Vec::with_capacity(strings_base + string_pool.len());
        out.extend_from_slice(&(elements.len() as u32).to_le_bytes());
        out.extend_from_slice(&(SIG_HEADER_BYTES as u32).to_le_bytes());

        for (e, name_off) in elements.iter().zip(name_offsets) {
            let comp_raw: u32 = match e.component_type {
                SignatureComponentType::Unknown => 0,
                SignatureComponentType::Uint32 => 1,
                SignatureComponentType::Sint32 => 2,
                SignatureComponentType::Float32 => 3,
                SignatureComponentType::Uint16 => 4,
                SignatureComponentType::Sint16 => 5,
                SignatureComponentType::Float16 => 6,
                SignatureComponentType::Uint64 => 7,
                SignatureComponentType::Sint64 => 8,
                SignatureComponentType::Float64 => 9,
                SignatureComponentType::Other(v) => v,
            };
            if is_v1 {
                out.extend_from_slice(&e.stream.to_le_bytes());
            }
            out.extend_from_slice(&name_off.to_le_bytes());
            out.extend_from_slice(&e.semantic_index.to_le_bytes());
            out.extend_from_slice(&e.system_value_type.to_le_bytes());
            out.extend_from_slice(&comp_raw.to_le_bytes());
            out.extend_from_slice(&e.register.to_le_bytes());
            out.push(e.mask);
            out.push(e.rw_mask);
            out.extend_from_slice(&[0u8, 0u8]);
            if is_v1 {
                out.extend_from_slice(&e.min_precision.to_le_bytes());
            }
        }
        out.extend_from_slice(&string_pool);
        out
    }

    /// Build a synthetic `PSV0` v2 chunk with shader stage, compute threadgroup dimensions, and resources.
    pub(crate) fn encode_psv0(
        stage: u8,
        threads: [u32; 3],
        resources: &[PsvResourceBinding],
    ) -> Vec<u8> {
        let rt_size = 48u32;
        let mut rt = vec![0u8; rt_size as usize];
        rt[24] = stage;
        rt[36..40].copy_from_slice(&threads[0].to_le_bytes());
        rt[40..44].copy_from_slice(&threads[1].to_le_bytes());
        rt[44..48].copy_from_slice(&threads[2].to_le_bytes());

        let mut out = Vec::new();
        out.extend_from_slice(&rt_size.to_le_bytes());
        out.extend_from_slice(&rt);
        out.extend_from_slice(&(resources.len() as u32).to_le_bytes());
        if !resources.is_empty() {
            out.extend_from_slice(&24u32.to_le_bytes());
            for r in resources {
                out.extend_from_slice(&r.res_type.to_le_bytes());
                out.extend_from_slice(&r.space.to_le_bytes());
                out.extend_from_slice(&r.lower_bound.to_le_bytes());
                out.extend_from_slice(&r.upper_bound.to_le_bytes());
                out.extend_from_slice(&r.res_kind.unwrap_or(0).to_le_bytes());
                out.extend_from_slice(&r.res_flags.unwrap_or(0).to_le_bytes());
            }
        }
        out
    }

    pub(crate) fn sample_static_sampler(
        shader_register: u32,
        register_space: u32,
        visibility: D3dShaderVisibility,
        comparison: bool,
    ) -> StaticSamplerDesc {
        StaticSamplerDesc {
            filter: if comparison { 0x95 } else { 0x15 }, // MIN_MAG_MIP_LINEAR vs COMPARISON_MIN_MAG_MIP_LINEAR
            address_u: 1,
            address_v: 1,
            address_w: 1,
            mip_lod_bias_bits: 0,
            max_anisotropy: 1,
            comparison_func: if comparison { 2 } else { 1 },
            border_color: 0,
            min_lod_bits: 0,
            max_lod_bits: f32::MAX.to_bits(),
            shader_register,
            register_space,
            shader_visibility: visibility,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_builders::*;
    use super::*;

    fn sig_elem(
        name: &str,
        semantic_index: u32,
        system_value_type: u32,
        component_type: SignatureComponentType,
        register: u32,
        mask: u8,
    ) -> SignatureElement {
        SignatureElement {
            stream: 0,
            semantic_name: name.to_string(),
            semantic_index,
            system_value_type,
            component_type,
            register,
            mask,
            rw_mask: mask,
            min_precision: 0,
        }
    }

    #[test]
    fn parses_rts0_v1_0_and_v1_1_with_all_parameter_kinds_and_static_samplers() {
        for version in [RootSignatureVersion::V1_0, RootSignatureVersion::V1_1] {
            let rs = RootSignature {
                version,
                flags: 0x0000_0001, // ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                parameters: vec![
                    RootParameter {
                        visibility: D3dShaderVisibility::Vertex,
                        kind: RootParameterKind::Constants32Bit {
                            shader_register: 0,
                            register_space: 0,
                            num_32bit_values: 16,
                        },
                    },
                    RootParameter {
                        visibility: D3dShaderVisibility::Vertex,
                        kind: RootParameterKind::Cbv {
                            shader_register: 1,
                            register_space: 0,
                            flags: if version == RootSignatureVersion::V1_1 {
                                0x2
                            } else {
                                0
                            },
                        },
                    },
                    RootParameter {
                        visibility: D3dShaderVisibility::Pixel,
                        kind: RootParameterKind::Srv {
                            shader_register: 0,
                            register_space: 1,
                            flags: if version == RootSignatureVersion::V1_1 {
                                0x4
                            } else {
                                0
                            },
                        },
                    },
                    RootParameter {
                        visibility: D3dShaderVisibility::All,
                        kind: RootParameterKind::Uav {
                            shader_register: 0,
                            register_space: 2,
                            flags: 0,
                        },
                    },
                    RootParameter {
                        visibility: D3dShaderVisibility::Pixel,
                        kind: RootParameterKind::DescriptorTable(vec![
                            DescriptorRange {
                                range_type: DescriptorRangeType::Srv,
                                num_descriptors: 2,
                                base_shader_register: 0,
                                register_space: 0,
                                flags: if version == RootSignatureVersion::V1_1 {
                                    0x1
                                } else {
                                    0
                                },
                                offset_in_descriptors_from_table_start: u32::MAX,
                            },
                            DescriptorRange {
                                range_type: DescriptorRangeType::Sampler,
                                num_descriptors: 1,
                                base_shader_register: 0,
                                register_space: 3,
                                flags: 0,
                                offset_in_descriptors_from_table_start: 0,
                            },
                        ]),
                    },
                ],
                static_samplers: vec![
                    sample_static_sampler(1, 3, D3dShaderVisibility::Pixel, false),
                    sample_static_sampler(2, 3, D3dShaderVisibility::Pixel, true),
                ],
            };

            let rts0_bytes = encode_rts0(&rs);
            let parsed = parse_rts0_chunk(&rts0_bytes).expect("RTS0 chunk should parse cleanly");
            assert_eq!(parsed, rs);

            let dxil = build_dxil_chunk(1, 6, 6); // vs_6_6
            let isg1 = encode_signature(
                true,
                &[
                    sig_elem("POSITION", 0, 0, SignatureComponentType::Float32, 0, 0x7),
                    sig_elem("TEXCOORD", 0, 0, SignatureComponentType::Float32, 1, 0x3),
                    sig_elem("COLOR", 0, 0, SignatureComponentType::Uint32, 2, 0xF),
                    sig_elem("SV_VertexID", 0, 6, SignatureComponentType::Uint32, 3, 0x1),
                ],
            );
            let osg1 = encode_signature(
                true,
                &[
                    sig_elem("SV_Position", 0, 1, SignatureComponentType::Float32, 0, 0xF),
                    sig_elem("TEXCOORD", 0, 0, SignatureComponentType::Float32, 1, 0x3),
                ],
            );
            let mut hash_chunk = Vec::new();
            hash_chunk.extend_from_slice(&0u32.to_le_bytes());
            hash_chunk.extend_from_slice(&[0xab; 16]);

            let container = build_container(&[
                (*b"DXIL", &dxil),
                (*b"RTS0", &rts0_bytes),
                (*b"ISG1", &isg1),
                (*b"OSG1", &osg1),
                (*b"HASH", &hash_chunk),
            ]);

            let refl = reflect_dxil_bytes_with_options(
                &container,
                Some("vs_pbr.dxil"),
                &RegisterConvention::default(),
            )
            .expect("full DXIL reflection should succeed");

            assert_eq!(refl.entry.stage, ShaderStage::Vertex);
            assert!(refl.entry.layout_complete());
            assert!(refl.bind_group_remaps.is_empty());
            assert_eq!(
                refl.shader_hash_digest.as_deref(),
                Some("abababababababababababababababab")
            );
            assert_eq!(refl.pipeline_layout.bind_group_layouts.len(), 4);

            // Group 0: b0 (root constants, 64B, VERTEX), b1 (root CBV, VERTEX),
            // t0 & t1 (table SRVs -> textures, FRAGMENT, collision-free binding indices).
            let g0 = &refl.pipeline_layout.bind_group_layouts[0].entries;
            assert_eq!(g0.len(), 4);
            assert_eq!(
                g0[0],
                LayoutEntry {
                    binding: 0,
                    visibility: VISIBILITY_VERTEX,
                    layout: BindingLayout::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: 64,
                    },
                }
            );
            assert_eq!(
                g0[1],
                LayoutEntry {
                    binding: 1,
                    visibility: VISIBILITY_VERTEX,
                    layout: BindingLayout::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: 0,
                    },
                }
            );
            // t0 collided with b0 -> allocated binding 2; t1 collided with b1 -> allocated binding 3.
            assert_eq!(
                g0[2],
                LayoutEntry {
                    binding: 2,
                    visibility: VISIBILITY_FRAGMENT,
                    layout: BindingLayout::Texture {
                        sample_type: TextureSampleType::Float,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                }
            );
            assert_eq!(g0[3].binding, 3);
            assert_eq!(g0[3].layout, g0[2].layout);

            // Group 1: root SRV t0 -> read-only-storage buffer, FRAGMENT
            let g1 = &refl.pipeline_layout.bind_group_layouts[1].entries;
            assert_eq!(g1.len(), 1);
            assert_eq!(
                g1[0],
                LayoutEntry {
                    binding: 0,
                    visibility: VISIBILITY_FRAGMENT,
                    layout: BindingLayout::Buffer {
                        ty: BufferBindingType::ReadOnlyStorage,
                        has_dynamic_offset: false,
                        min_binding_size: 0,
                    },
                }
            );

            // Group 2: root UAV u0 -> storage buffer, VERTEX | FRAGMENT (ALL on VS container)
            let g2 = &refl.pipeline_layout.bind_group_layouts[2].entries;
            assert_eq!(g2.len(), 1);
            assert_eq!(
                g2[0],
                LayoutEntry {
                    binding: 0,
                    visibility: VISIBILITY_VERTEX | VISIBILITY_FRAGMENT,
                    layout: BindingLayout::Buffer {
                        ty: BufferBindingType::Storage,
                        has_dynamic_offset: false,
                        min_binding_size: 0,
                    },
                }
            );

            // Group 3: table sampler s0 (filtering), static sampler s1 (filtering), static sampler s2 (comparison)
            let g3 = &refl.pipeline_layout.bind_group_layouts[3].entries;
            assert_eq!(g3.len(), 3);
            assert_eq!(
                g3[0].layout,
                BindingLayout::Sampler {
                    ty: SamplerBindingType::Filtering
                }
            );
            assert_eq!(
                g3[1].layout,
                BindingLayout::Sampler {
                    ty: SamplerBindingType::Filtering
                }
            );
            assert_eq!(
                g3[2].layout,
                BindingLayout::Sampler {
                    ty: SamplerBindingType::Comparison
                }
            );

            // Vertex inputs from ISG1 (SV_VertexID skipped!)
            assert_eq!(
                refl.entry.vertex_inputs,
                vec![
                    ShaderIo {
                        location: 0,
                        wgsl_type: "vec3<f32>".to_string(),
                        vertex_format: Some("float32x3"),
                    },
                    ShaderIo {
                        location: 1,
                        wgsl_type: "vec2<f32>".to_string(),
                        vertex_format: Some("float32x2"),
                    },
                    ShaderIo {
                        location: 2,
                        wgsl_type: "vec4<u32>".to_string(),
                        vertex_format: Some("uint32x4"),
                    },
                ]
            );
            assert!(refl.entry.fragment_outputs.is_empty());

            // Root constants flag preserved on ReflectedBinding
            let rc = refl
                .entry
                .bindings
                .iter()
                .find(|b| b.group == 0 && b.binding == 0)
                .unwrap();
            assert!(rc.root_constants);
            assert_eq!(
                rc.register,
                Some(D3dRegister {
                    class: RegisterClass::B,
                    index: 0,
                    space: 0,
                })
            );
        }
    }

    #[test]
    fn clamps_overflow_register_spaces_above_three_into_webgpu_four_group_ceiling() {
        let rs = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![
                // Native spaces 0..=3 (each with 1 binding)
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Cbv {
                        shader_register: 0,
                        register_space: 0,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Cbv {
                        shader_register: 0,
                        register_space: 1,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Cbv {
                        shader_register: 0,
                        register_space: 2,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Cbv {
                        shader_register: 0,
                        register_space: 3,
                        flags: 0,
                    },
                },
                // Overflow spaces 4, 5, 9
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Srv {
                        shader_register: 0,
                        register_space: 4,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Uav {
                        shader_register: 0,
                        register_space: 5,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: 2,
                        base_shader_register: 0,
                        register_space: 9,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 0,
                    }]),
                },
            ],
            static_samplers: vec![],
        };

        let dxil = build_dxil_chunk(5, 6, 6); // cs_6_6
        let rts0 = encode_rts0(&rs);
        let psv0 = encode_psv0(5, [8, 8, 1], &[]);
        let container = build_container(&[(*b"DXIL", &dxil), (*b"RTS0", &rts0), (*b"PSV0", &psv0)]);

        let refl = reflect_dxil_bytes(&container).expect("overflow spaces should lower cleanly");
        assert_eq!(refl.entry.stage, ShaderStage::Compute);
        assert_eq!(refl.entry.workgroup_size, Some([8, 8, 1]));
        assert_eq!(
            refl.pipeline_layout.bind_group_layouts.len(),
            WEBGPU_MAX_BIND_GROUPS as usize
        );
        assert!(refl.limit_violations.is_empty());

        // Every emitted binding must lie within groups 0..=3 with unique (group, binding).
        let mut seen = BTreeSet::new();
        for b in &refl.entry.bindings {
            assert!(b.group < WEBGPU_MAX_BIND_GROUPS);
            assert!(seen.insert((b.group, b.binding)));
        }
        // All 4 overflow bindings (space 4 t0, space 5 u0, space 9 t0 & t1) must be recorded in remaps.
        assert_eq!(refl.bind_group_remaps.len(), 4);
        assert!(refl.bind_group_remaps.iter().all(|r| r.webgpu_group < 4));
        assert!(refl.bind_group_remaps.iter().all(|r| r.original_space >= 4));
        // Compute stage visibility on all entries
        for bgl in &refl.pipeline_layout.bind_group_layouts {
            for e in &bgl.entries {
                assert_eq!(e.visibility, VISIBILITY_COMPUTE);
            }
        }
    }

    #[test]
    fn parses_pixel_shader_osg1_render_targets_and_psv0_resource_refinement() {
        let rs = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: RTS0_FLAG_DENY_VERTEX_SHADER_ROOT_ACCESS,
            parameters: vec![RootParameter {
                visibility: D3dShaderVisibility::All,
                kind: RootParameterKind::DescriptorTable(vec![
                    DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: 2,
                        base_shader_register: 0,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 0,
                    },
                    DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: u32::MAX, // unbounded bindless table
                        base_shader_register: 10,
                        register_space: 1,
                        flags: 0,
                        offset_in_descriptors_from_table_start: u32::MAX,
                    },
                ]),
            }],
            static_samplers: vec![],
        };

        // PSV0 annotates t0 as StructuredBuffer (read-only-storage) and t1 as TextureCube.
        let psv0 = encode_psv0(
            0,
            [0, 0, 0],
            &[
                PsvResourceBinding {
                    res_type: 5, // SRVStructured
                    space: 0,
                    lower_bound: 0,
                    upper_bound: 0,
                    res_kind: Some(12), // StructuredBuffer
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 3, // SRVTyped
                    space: 0,
                    lower_bound: 1,
                    upper_bound: 1,
                    res_kind: Some(5), // TextureCube
                    res_flags: Some(0),
                },
            ],
        );
        let dxil = build_dxil_chunk(0, 6, 6); // ps_6_6
        let rts0 = encode_rts0(&rs);
        let osg1 = encode_signature(
            true,
            &[
                sig_elem("SV_Target", 0, D3D_NAME_TARGET, SignatureComponentType::Float32, 0, 0xF),
                sig_elem("SV_Target", 1, D3D_NAME_TARGET, SignatureComponentType::Uint32, 1, 0x3),
                sig_elem("SV_Depth", 0, 65, SignatureComponentType::Float32, 0xFFFF_FFFF, 0x1),
            ],
        );

        let container = build_container(&[
            (*b"DXIL", &dxil),
            (*b"RTS0", &rts0),
            (*b"OSG1", &osg1),
            (*b"PSV0", &psv0),
        ]);

        let refl = reflect_dxil_bytes(&container).expect("pixel shader reflection should succeed");
        assert_eq!(refl.entry.stage, ShaderStage::Fragment);
        assert!(!refl.entry.layout_complete()); // due to unbounded bindless range t10 space1

        // DENY_VERTEX_SHADER_ROOT_ACCESS strips VERTEX bit -> visibility == FRAGMENT (2) only!
        let g0 = &refl.pipeline_layout.bind_group_layouts[0].entries;
        assert_eq!(g0.len(), 2);
        assert_eq!(
            g0[0],
            LayoutEntry {
                binding: 0,
                visibility: VISIBILITY_FRAGMENT,
                layout: BindingLayout::Buffer {
                    ty: BufferBindingType::ReadOnlyStorage,
                    has_dynamic_offset: false,
                    min_binding_size: 0,
                },
            }
        );
        assert_eq!(
            g0[1],
            LayoutEntry {
                binding: 1,
                visibility: VISIBILITY_FRAGMENT,
                layout: BindingLayout::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::Cube,
                    multisampled: false,
                },
            }
        );

        assert_eq!(
            refl.entry.fragment_outputs,
            vec![
                ShaderIo {
                    location: 0,
                    wgsl_type: "vec4<f32>".to_string(),
                    vertex_format: None,
                },
                ShaderIo {
                    location: 1,
                    wgsl_type: "vec2<u32>".to_string(),
                    vertex_format: None,
                },
            ]
        );
        assert!(
            refl.entry
                .requires_lowering
                .iter()
                .any(|n| n.kind == LoweringKind::UnboundedBindingArray)
        );
    }

    #[test]
    fn rejects_malformed_and_truncated_rts0_isg1_and_psv0_without_panicking() {
        // 1. Truncated RTS0 header
        assert!(matches!(
            parse_rts0_chunk(&[1, 0, 0, 0]),
            Err(DxilReflectError::TruncatedChunk { fourcc: "RTS0", .. })
        ));

        // 2. Unsupported RTS0 version (e.g. 99)
        let mut bad_ver = vec![0u8; 24];
        bad_ver[0..4].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            parse_rts0_chunk(&bad_ver),
            Err(DxilReflectError::UnsupportedRootSignatureVersion(99))
        ));

        // 3. Overflowing parameter count
        let mut huge_params = vec![0u8; 24];
        huge_params[0..4].copy_from_slice(&2u32.to_le_bytes());
        huge_params[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        huge_params[8..12].copy_from_slice(&24u32.to_le_bytes());
        assert!(matches!(
            parse_rts0_chunk(&huge_params),
            Err(DxilReflectError::OutOfBounds { .. } | DxilReflectError::Overflow { .. })
        ));

        // 4. Unknown root parameter type (99)
        let mut bad_param = vec![0u8; 36];
        bad_param[0..4].copy_from_slice(&2u32.to_le_bytes());
        bad_param[4..8].copy_from_slice(&1u32.to_le_bytes());
        bad_param[8..12].copy_from_slice(&24u32.to_le_bytes());
        bad_param[24..28].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            parse_rts0_chunk(&bad_param),
            Err(DxilReflectError::InvalidRootParameterType { index: 0, raw: 99 })
        ));

        // 5. Invalid shader visibility (42)
        let mut bad_vis = vec![0u8; 48];
        bad_vis[0..4].copy_from_slice(&2u32.to_le_bytes());
        bad_vis[4..8].copy_from_slice(&1u32.to_le_bytes());
        bad_vis[8..12].copy_from_slice(&24u32.to_le_bytes());
        bad_vis[24..28].copy_from_slice(&1u32.to_le_bytes()); // Constants32Bit
        bad_vis[28..32].copy_from_slice(&42u32.to_le_bytes()); // bad visibility
        bad_vis[32..36].copy_from_slice(&36u32.to_le_bytes());
        assert!(matches!(
            parse_rts0_chunk(&bad_vis),
            Err(DxilReflectError::InvalidShaderVisibility { raw: 42, .. })
        ));

        // 6. Descriptor table range pointing past chunk end
        let mut oob_range = vec![0u8; 44];
        oob_range[0..4].copy_from_slice(&2u32.to_le_bytes());
        oob_range[4..8].copy_from_slice(&1u32.to_le_bytes());
        oob_range[8..12].copy_from_slice(&24u32.to_le_bytes());
        oob_range[24..28].copy_from_slice(&0u32.to_le_bytes()); // DescriptorTable
        oob_range[28..32].copy_from_slice(&0u32.to_le_bytes()); // All
        oob_range[32..36].copy_from_slice(&36u32.to_le_bytes());
        oob_range[36..40].copy_from_slice(&2u32.to_le_bytes()); // 2 ranges
        oob_range[40..44].copy_from_slice(&44u32.to_le_bytes()); // past end
        assert!(matches!(
            parse_rts0_chunk(&oob_range),
            Err(DxilReflectError::OutOfBounds { fourcc: "RTS0", .. })
        ));

        // 7. Static sampler table truncated
        let mut bad_samp = vec![0u8; 32];
        bad_samp[0..4].copy_from_slice(&2u32.to_le_bytes());
        bad_samp[12..16].copy_from_slice(&1u32.to_le_bytes());
        bad_samp[16..20].copy_from_slice(&24u32.to_le_bytes());
        assert!(matches!(
            parse_rts0_chunk(&bad_samp),
            Err(DxilReflectError::OutOfBounds { fourcc: "RTS0", .. })
        ));

        // 8. Truncated ISG1 & unterminated semantic name string
        assert!(matches!(
            parse_signature_elements(*b"ISG1", &[1, 0, 0, 0]),
            Err(DxilReflectError::TruncatedChunk { fourcc: "ISG1", .. })
        ));
        let mut unterminated_isg1 = encode_signature(
            true,
            &[sig_elem("POSITION", 0, 0, SignatureComponentType::Float32, 0, 0x7)],
        );
        // Overwrite NUL terminator with 'X'
        let last = unterminated_isg1.len() - 1;
        unterminated_isg1[last] = b'X';
        assert!(matches!(
            parse_signature_elements(*b"ISG1", &unterminated_isg1),
            Err(DxilReflectError::InvalidSignatureString { fourcc: "ISG1", .. })
        ));

        // 9. Truncated PSV0
        assert!(matches!(
            parse_psv0_chunk(&[64, 0, 0, 0, 1, 2]),
            Err(DxilReflectError::OutOfBounds { fourcc: "PSV0", .. })
        ));

        // 10. Invalid DXBC container fails cleanly
        assert!(matches!(
            reflect_dxil_bytes(b"NOT_DXBC"),
            Err(DxilReflectError::Dxbc(_))
        ));
    }

    #[test]
    fn manifest_json_is_deterministic_sanitized_and_matches_s6_layout_hashes() {
        let rs_a = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![RootParameter {
                visibility: D3dShaderVisibility::Vertex,
                kind: RootParameterKind::Cbv {
                    shader_register: 0,
                    register_space: 0,
                    flags: 0,
                },
            }],
            static_samplers: vec![],
        };
        let rs_b = RootSignature {
            version: RootSignatureVersion::V1_0,
            flags: 0,
            parameters: vec![RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                    range_type: DescriptorRangeType::Srv,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    flags: 0,
                    offset_in_descriptors_from_table_start: 0,
                }]),
            }],
            static_samplers: vec![sample_static_sampler(0, 0, D3dShaderVisibility::Pixel, false)],
        };

        let vs_bytes = build_container(&[
            (*b"DXIL", &build_dxil_chunk(1, 6, 6)),
            (*b"RTS0", &encode_rts0(&rs_a)),
            (
                *b"ISG1",
                &encode_signature(
                    true,
                    &[sig_elem("POSITION", 0, 0, SignatureComponentType::Float32, 0, 0x7)],
                ),
            ),
        ]);
        let ps_bytes = build_container(&[
            (*b"DXIL", &build_dxil_chunk(0, 6, 6)),
            (*b"RTS0", &encode_rts0(&rs_b)),
            (
                *b"OSG1",
                &encode_signature(
                    true,
                    &[sig_elem("SV_Target", 0, D3D_NAME_TARGET, SignatureComponentType::Float32, 0, 0xF)],
                ),
            ),
        ]);

        let conv = RegisterConvention::default();
        let r1 = reflect_dxil_bytes_with_options(&vs_bytes, Some("vs.dxil"), &conv).unwrap();
        let r2 = reflect_dxil_bytes_with_options(&ps_bytes, Some("ps.dxil"), &conv).unwrap();

        // Order-independent deterministic output
        let m_ab = build_dxil_reflection_manifest(&[r1.clone(), r2.clone()], &[], &conv).unwrap();
        let m_ba = build_dxil_reflection_manifest(&[r2, r1], &[], &conv).unwrap();
        assert_eq!(m_ab.render_json(), m_ba.render_json());

        let json = m_ab.render_json();
        assert!(json.starts_with("{\n  \"schema_version\": 1,\n"));
        assert!(json.contains("\"bind_group_layouts\": {"));
        assert!(json.contains("\"pipeline_layouts\": {"));
        assert!(json.contains("\"shaders\": {"));
        assert!(json.contains("\"vertex_inputs\": ["));
        assert!(json.contains("\"format\":\"float32x3\""));
        assert!(json.contains("\"fragment_outputs\": ["));
        assert!(json.contains("\"type\":\"vec4<f32>\""));
        assert!(json.contains("\"rts0_version\": \"1.1\""));
        assert!(json.contains("\"rts0_version\": \"1.0\""));

        // Verify projected LayoutManifest uses exact same BGL & PL content hashes
        let s6 = m_ab.to_layout_manifest();
        assert_eq!(s6.bind_group_layouts, m_ab.bind_group_layouts);
        assert_eq!(s6.pipeline_layouts, m_ab.pipeline_layouts);
        assert_eq!(s6.shaders.len(), 2);

        // Verify path sanitization strips host paths & drive letters identically on POSIX and Windows
        assert_eq!(
            sanitize_source_path(Path::new("/Users/alice/secret/shaders/sky.dxil"), None),
            "sky.dxil"
        );
        assert_eq!(
            sanitize_source_path(Path::new(r"C:\Games\SecretTitle\shaders\sky.dxil"), None),
            "sky.dxil"
        );
        assert_eq!(
            sanitize_source_path(
                Path::new(r"C:\Games\SecretTitle\shaders\nested\sky.dxil"),
                Some(Path::new(r"C:\Games\SecretTitle\shaders")),
            ),
            "nested/sky.dxil"
        );
        assert_eq!(
            sanitize_source_path(
                Path::new("/home/runner/work/shaders/nested/sky.dxil"),
                Some(Path::new("/home/runner/work/shaders/")),
            ),
            "nested/sky.dxil"
        );
        assert_eq!(
            sanitize_source_path(
                Path::new("/home/runner/shaders_other/sky.dxil"),
                Some(Path::new("/home/runner/shaders")),
            ),
            "sky.dxil"
        );
    }

    #[test]
    fn supports_legacy_isgn_osgn_cbv_uav_tables_large_arrays_and_visibility_merging() {
        let rs = RootSignature {
            version: RootSignatureVersion::V1_0,
            flags: 0,
            parameters: vec![
                // Same CBV b0 space0 declared for VERTEX and PIXEL -> visibility must merge to 3
                RootParameter {
                    visibility: D3dShaderVisibility::Vertex,
                    kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                        range_type: DescriptorRangeType::Cbv,
                        num_descriptors: 1,
                        base_shader_register: 0,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 0,
                    }]),
                },
                RootParameter {
                    visibility: D3dShaderVisibility::Pixel,
                    kind: RootParameterKind::DescriptorTable(vec![
                        DescriptorRange {
                            range_type: DescriptorRangeType::Cbv,
                            num_descriptors: 1,
                            base_shader_register: 0,
                            register_space: 0,
                            flags: 0,
                            offset_in_descriptors_from_table_start: 0,
                        },
                        DescriptorRange {
                            range_type: DescriptorRangeType::Uav,
                            num_descriptors: 1,
                            base_shader_register: 2,
                            register_space: 0,
                            flags: 0,
                            offset_in_descriptors_from_table_start: 1,
                        },
                        DescriptorRange {
                            range_type: DescriptorRangeType::Srv,
                            num_descriptors: MAX_EXPANDED_RANGE_DESCRIPTORS + 16,
                            base_shader_register: 0,
                            register_space: 1,
                            flags: 0,
                            offset_in_descriptors_from_table_start: 2,
                        },
                    ]),
                },
            ],
            static_samplers: vec![],
        };

        let isgn = encode_signature(
            false,
            &[
                sig_elem("POSITION", 0, 0, SignatureComponentType::Float32, 0, 0xF),
                sig_elem("NORMAL", 0, 0, SignatureComponentType::Sint32, 1, 0x7),
            ],
        );
        let osgn = encode_signature(
            false,
            &[sig_elem(
                "SV_Position",
                0,
                1,
                SignatureComponentType::Float32,
                0,
                0xF,
            )],
        );
        let shdr = crate::dxbc::test_dxbc::build_shdr_chunk_payload(1, 5, 0, 12); // vs_5_0
        let container = build_container(&[
            (*b"SHDR", &shdr),
            (*b"RTS0", &encode_rts0(&rs)),
            (*b"ISGN", &isgn),
            (*b"OSGN", &osgn),
        ]);

        let refl = reflect_dxil_bytes(&container).expect("legacy ISGN/OSGN + RTS0 v1.0 should succeed");
        assert_eq!(refl.entry.stage, ShaderStage::Vertex);
        assert!(!refl.entry.layout_complete());

        let g0 = &refl.pipeline_layout.bind_group_layouts[0].entries;
        assert_eq!(g0.len(), 2);
        assert_eq!(g0[0].binding, 0);
        assert_eq!(g0[0].visibility, VISIBILITY_VERTEX | VISIBILITY_FRAGMENT);
        assert_eq!(
            g0[0].layout,
            BindingLayout::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 0,
            }
        );
        assert_eq!(g0[1].binding, 2);
        assert_eq!(g0[1].visibility, VISIBILITY_FRAGMENT);
        assert_eq!(
            g0[1].layout,
            BindingLayout::Buffer {
                ty: BufferBindingType::Storage,
                has_dynamic_offset: false,
                min_binding_size: 0,
            }
        );

        assert!(
            refl.entry
                .requires_lowering
                .iter()
                .any(|n| n.kind == LoweringKind::BindingArray)
        );
        assert_eq!(
            refl.entry.vertex_inputs,
            vec![
                ShaderIo {
                    location: 0,
                    wgsl_type: "vec4<f32>".to_string(),
                    vertex_format: Some("float32x4"),
                },
                ShaderIo {
                    location: 1,
                    wgsl_type: "vec3<i32>".to_string(),
                    vertex_format: Some("sint32x3"),
                },
            ]
        );
    }

    #[test]
    fn deterministic_binding_sort_order_invariant_to_rts0_psv0_and_signature_permutation() {
        let params_forward = vec![
            // Disjoint cross-stage shared b0 space0 (Vertex root constants + Pixel root CBV)
            RootParameter {
                visibility: D3dShaderVisibility::Vertex,
                kind: RootParameterKind::Constants32Bit {
                    shader_register: 0,
                    register_space: 0,
                    num_32bit_values: 8,
                },
            },
            RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::Cbv {
                    shader_register: 0,
                    register_space: 0,
                    flags: 0,
                },
            },
            RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::Srv {
                    shader_register: 2,
                    register_space: 1,
                    flags: 0,
                },
            },
            RootParameter {
                visibility: D3dShaderVisibility::All,
                kind: RootParameterKind::Uav {
                    shader_register: 1,
                    register_space: 2,
                    flags: 0,
                },
            },
            RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::DescriptorTable(vec![
                    DescriptorRange {
                        range_type: DescriptorRangeType::Sampler,
                        num_descriptors: 1,
                        base_shader_register: 0,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 0,
                    },
                    DescriptorRange {
                        range_type: DescriptorRangeType::Uav,
                        num_descriptors: 1,
                        base_shader_register: 0,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 1,
                    },
                    DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: 2,
                        base_shader_register: 0,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 2,
                    },
                    DescriptorRange {
                        range_type: DescriptorRangeType::Cbv,
                        num_descriptors: 1,
                        base_shader_register: 1,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 4,
                    },
                ]),
            },
            // Overflow spaces in non-sorted order (space 9 before space 4)
            RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::Srv {
                    shader_register: 0,
                    register_space: 9,
                    flags: 0,
                },
            },
            RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::Cbv {
                    shader_register: 0,
                    register_space: 4,
                    flags: 0,
                },
            },
        ];

        // Reverse parameter list AND reverse ranges inside the descriptor table.
        let mut params_reversed = params_forward.clone();
        params_reversed.reverse();
        for p in &mut params_reversed {
            if let RootParameterKind::DescriptorTable(ranges) = &mut p.kind {
                ranges.reverse();
            }
        }

        let samplers_forward = vec![
            sample_static_sampler(1, 3, D3dShaderVisibility::Pixel, false),
            sample_static_sampler(0, 3, D3dShaderVisibility::Pixel, true),
        ];
        let mut samplers_reversed = samplers_forward.clone();
        samplers_reversed.reverse();

        let psv_forward = vec![
            PsvResourceBinding {
                res_type: 5, // SRVStructured
                space: 0,
                lower_bound: 0,
                upper_bound: 0,
                res_kind: Some(12),
                res_flags: Some(0),
            },
            PsvResourceBinding {
                res_type: 3, // SRVTyped Cube
                space: 0,
                lower_bound: 1,
                upper_bound: 1,
                res_kind: Some(5),
                res_flags: Some(0),
            },
            // Broader range covering t0..=t1; narrower exact entries above must always win regardless of order.
            PsvResourceBinding {
                res_type: 3,
                space: 0,
                lower_bound: 0,
                upper_bound: 1,
                res_kind: Some(2),
                res_flags: Some(0),
            },
        ];
        let mut psv_reversed = psv_forward.clone();
        psv_reversed.reverse();

        let isg_forward = vec![
            sig_elem("COLOR", 0, 0, SignatureComponentType::Float32, 2, 0xF),
            sig_elem("POSITION", 0, 0, SignatureComponentType::Float32, 0, 0x7),
            sig_elem("TEXCOORD", 0, 0, SignatureComponentType::Float32, 1, 0x3),
        ];
        let mut isg_reversed = isg_forward.clone();
        isg_reversed.reverse();

        let rs_a = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: params_forward,
            static_samplers: samplers_forward,
        };
        let rs_b = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: params_reversed,
            static_samplers: samplers_reversed,
        };

        let dxil = build_dxil_chunk(1, 6, 6); // vs_6_6
        let container_a = build_container(&[
            (*b"DXIL", &dxil),
            (*b"RTS0", &encode_rts0(&rs_a)),
            (*b"PSV0", &encode_psv0(1, [0, 0, 0], &psv_forward)),
            (*b"ISG1", &encode_signature(true, &isg_forward)),
        ]);
        let container_b = build_container(&[
            (*b"DXIL", &dxil),
            (*b"RTS0", &encode_rts0(&rs_b)),
            (*b"PSV0", &encode_psv0(1, [0, 0, 0], &psv_reversed)),
            (*b"ISG1", &encode_signature(true, &isg_reversed)),
        ]);

        let refl_a = reflect_dxil_bytes(&container_a).expect("forward reflection should succeed");
        let refl_b = reflect_dxil_bytes(&container_b).expect("reversed reflection should succeed");

        assert_eq!(refl_a.entry, refl_b.entry);
        assert_eq!(refl_a.pipeline_layout, refl_b.pipeline_layout);
        assert_eq!(refl_a.bind_group_remaps, refl_b.bind_group_remaps);
        assert_eq!(refl_a.limit_violations, refl_b.limit_violations);

        // Verify strict canonical ascending (group, binding) ordering across all emitted bindings.
        for pair in refl_a.entry.bindings.windows(2) {
            assert!(
                (pair[0].group, pair[0].binding) < (pair[1].group, pair[1].binding),
                "bindings must be strictly ascending by (group, binding): {:?} vs {:?}",
                pair[0],
                pair[1]
            );
        }

        // Verify space 0 canonical class order: B(b0, b1) -> T(t0, t1) -> U(u0) -> S(s0).
        let group0_regs: Vec<D3dRegister> = refl_a
            .entry
            .bindings
            .iter()
            .filter(|b| b.group == 0 && b.register.is_some_and(|r| r.space == 0))
            .map(|b| b.register.unwrap())
            .collect();
        assert_eq!(
            group0_regs,
            vec![
                D3dRegister { class: RegisterClass::B, index: 0, space: 0 },
                D3dRegister { class: RegisterClass::B, index: 1, space: 0 },
                D3dRegister { class: RegisterClass::T, index: 0, space: 0 },
                D3dRegister { class: RegisterClass::T, index: 1, space: 0 },
                D3dRegister { class: RegisterClass::U, index: 0, space: 0 },
                D3dRegister { class: RegisterClass::S, index: 0, space: 0 },
            ]
        );

        // Cross-stage merged b0 must preserve root_constants, max min_binding_size (32B),
        // merged visibility (VERTEX | FRAGMENT), and canonical name independent of parameter order.
        let b0 = refl_a
            .entry
            .bindings
            .iter()
            .find(|b| b.register == Some(D3dRegister { class: RegisterClass::B, index: 0, space: 0 }))
            .unwrap();
        assert!(b0.root_constants);
        assert_eq!(b0.name.as_deref(), Some("root_cbv_b0_space0"));
        assert_eq!(
            b0.layout,
            Some(BindingLayout::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 32,
            })
        );
        let g0_b0 = refl_a.pipeline_layout.bind_group_layouts[0]
            .entries
            .iter()
            .find(|e| e.binding == b0.binding)
            .unwrap();
        assert_eq!(g0_b0.visibility, VISIBILITY_VERTEX | VISIBILITY_FRAGMENT);
    }

    #[test]
    fn rejects_duplicate_and_overlapping_register_indices_with_clean_diagnostics() {
        let dxil = build_dxil_chunk(0, 6, 6); // ps_6_6

        // 1. Two root CBVs targeting b0 space0 with overlapping visibility (All + Pixel).
        let dup_cbv = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Cbv {
                        shader_register: 0,
                        register_space: 0,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::Pixel,
                    kind: RootParameterKind::Cbv {
                        shader_register: 0,
                        register_space: 0,
                        flags: 0,
                    },
                },
            ],
            static_samplers: vec![],
        };
        let c1 = build_container(&[(*b"DXIL", &dxil), (*b"RTS0", &encode_rts0(&dup_cbv))]);
        assert!(matches!(
            reflect_dxil_bytes(&c1),
            Err(DxilReflectError::DuplicateRegister {
                register: D3dRegister {
                    class: RegisterClass::B,
                    index: 0,
                    space: 0,
                },
                ..
            })
        ));

        // 2. Overlapping descriptor table ranges within the same visibility (t0..=t2 and t2..=t3).
        let overlap_ranges = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::DescriptorTable(vec![
                    DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: 3,
                        base_shader_register: 0,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 0,
                    },
                    DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: 2,
                        base_shader_register: 2,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 3,
                    },
                ]),
            }],
            static_samplers: vec![],
        };
        let c2 = build_container(&[(*b"DXIL", &dxil), (*b"RTS0", &encode_rts0(&overlap_ranges))]);
        assert!(matches!(
            reflect_dxil_bytes(&c2),
            Err(DxilReflectError::DuplicateRegister {
                register: D3dRegister {
                    class: RegisterClass::T,
                    index: 2,
                    space: 0,
                },
                ..
            })
        ));

        // 3. Unbounded descriptor range overlapping a concrete SRV on the same stage.
        let overlap_unbounded = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: u32::MAX,
                        base_shader_register: 0,
                        register_space: 1,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 0,
                    }]),
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Srv {
                        shader_register: 5,
                        register_space: 1,
                        flags: 0,
                    },
                },
            ],
            static_samplers: vec![],
        };
        let c3 = build_container(&[
            (*b"DXIL", &dxil),
            (*b"RTS0", &encode_rts0(&overlap_unbounded)),
        ]);
        assert!(matches!(
            reflect_dxil_bytes(&c3),
            Err(DxilReflectError::DuplicateRegister {
                register: D3dRegister {
                    class: RegisterClass::T,
                    index: 5,
                    space: 1,
                },
                ..
            })
        ));

        // 4. Table sampler colliding with static sampler on the same stage and register.
        let dup_sampler = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![RootParameter {
                visibility: D3dShaderVisibility::Pixel,
                kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                    range_type: DescriptorRangeType::Sampler,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    flags: 0,
                    offset_in_descriptors_from_table_start: 0,
                }]),
            }],
            static_samplers: vec![sample_static_sampler(
                0,
                0,
                D3dShaderVisibility::Pixel,
                false,
            )],
        };
        let c4 = build_container(&[(*b"DXIL", &dxil), (*b"RTS0", &encode_rts0(&dup_sampler))]);
        assert!(matches!(
            reflect_dxil_bytes(&c4),
            Err(DxilReflectError::DuplicateRegister {
                register: D3dRegister {
                    class: RegisterClass::S,
                    index: 0,
                    space: 0,
                },
                ..
            })
        ));

        // 5. Disjoint stages (Vertex vs Pixel) with INCOMPATIBLE layouts on the same t0 space0
        // (root SRV buffer vs table SRV texture) must return DuplicateRegister.
        let incompatible_cross_stage = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![
                RootParameter {
                    visibility: D3dShaderVisibility::Vertex,
                    kind: RootParameterKind::Srv {
                        shader_register: 0,
                        register_space: 0,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::Pixel,
                    kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: 1,
                        base_shader_register: 0,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 0,
                    }]),
                },
            ],
            static_samplers: vec![],
        };
        let c5 = build_container(&[
            (*b"DXIL", &dxil),
            (*b"RTS0", &encode_rts0(&incompatible_cross_stage)),
        ]);
        assert!(matches!(
            reflect_dxil_bytes(&c5),
            Err(DxilReflectError::DuplicateRegister {
                register: D3dRegister {
                    class: RegisterClass::T,
                    index: 0,
                    space: 0,
                },
                ..
            })
        ));

        // 6. PSV0-only container with overlapping resource bindings must fail cleanly.
        let psv_dup = encode_psv0(
            0,
            [0, 0, 0],
            &[
                PsvResourceBinding {
                    res_type: 2, // CBV
                    space: 0,
                    lower_bound: 0,
                    upper_bound: 2,
                    res_kind: Some(0),
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 2, // CBV overlapping b1
                    space: 0,
                    lower_bound: 1,
                    upper_bound: 1,
                    res_kind: Some(0),
                    res_flags: Some(0),
                },
            ],
        );
        let c6 = build_container(&[(*b"DXIL", &dxil), (*b"PSV0", &psv_dup)]);
        assert!(matches!(
            reflect_dxil_bytes(&c6),
            Err(DxilReflectError::DuplicateRegister {
                register: D3dRegister {
                    class: RegisterClass::B,
                    index: 1,
                    space: 0,
                },
                ..
            })
        ));
    }

    #[test]
    fn guards_truncated_malformed_and_overflowing_chunks_against_panics() {
        // 1. Truncated DXIL chunk (< 24B) inside a valid DXBC container.
        let trunc_dxil = build_container(&[(*b"DXIL", &[0u8; 12])]);
        assert!(matches!(
            reflect_dxil_bytes(&trunc_dxil),
            Err(DxilReflectError::TruncatedChunk {
                fourcc: "DXIL",
                actual: 12,
                expected_min: 24,
            })
        ));

        // 2. DXIL chunk with out-of-range / overflowing bitcode_offset.
        let mut bad_dxil = build_dxil_chunk(0, 6, 6);
        bad_dxil[16..20].copy_from_slice(&u32::MAX.to_le_bytes()); // bitcode_offset = u32::MAX
        let bad_dxil_container = build_container(&[(*b"DXIL", &bad_dxil)]);
        assert!(matches!(
            reflect_dxil_bytes(&bad_dxil_container),
            Err(DxilReflectError::OutOfBounds { fourcc: "DXIL", .. }
                | DxilReflectError::Overflow { fourcc: "DXIL", .. })
        ));

        // 3. Truncated SHDR chunk (< 8B).
        let trunc_shdr = build_container(&[(*b"SHDR", &[1, 2, 3, 4])]);
        assert!(matches!(
            reflect_dxil_bytes(&trunc_shdr),
            Err(DxilReflectError::TruncatedChunk {
                fourcc: "SHDR",
                actual: 4,
                expected_min: 8,
            })
        ));

        // 4. RTS0 parameter data_off pointing inside header (< 24).
        let mut header_alias_param = vec![0u8; 36];
        header_alias_param[0..4].copy_from_slice(&2u32.to_le_bytes());
        header_alias_param[4..8].copy_from_slice(&1u32.to_le_bytes());
        header_alias_param[8..12].copy_from_slice(&24u32.to_le_bytes());
        header_alias_param[24..28].copy_from_slice(&2u32.to_le_bytes()); // Root CBV
        header_alias_param[28..32].copy_from_slice(&0u32.to_le_bytes()); // All
        header_alias_param[32..36].copy_from_slice(&4u32.to_le_bytes()); // inside header!
        assert!(matches!(
            parse_rts0_chunk(&header_alias_param),
            Err(DxilReflectError::OutOfBounds { fourcc: "RTS0", .. })
        ));

        // 5. RTS0 descriptor range register-index overflow (base = u32::MAX, count = 2).
        let overflow_rs = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![RootParameter {
                visibility: D3dShaderVisibility::All,
                kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                    range_type: DescriptorRangeType::Srv,
                    num_descriptors: 2,
                    base_shader_register: u32::MAX,
                    register_space: 0,
                    flags: 0,
                    offset_in_descriptors_from_table_start: 0,
                }]),
            }],
            static_samplers: vec![],
        };
        assert!(matches!(
            parse_rts0_chunk(&encode_rts0(&overflow_rs)),
            Err(DxilReflectError::Overflow { fourcc: "RTS0", .. })
        ));

        // 6. RTS0 invalid descriptor range type (raw = 9).
        let mut bad_range_type = encode_rts0(&RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![RootParameter {
                visibility: D3dShaderVisibility::All,
                kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                    range_type: DescriptorRangeType::Srv,
                    num_descriptors: 1,
                    base_shader_register: 0,
                    register_space: 0,
                    flags: 0,
                    offset_in_descriptors_from_table_start: 0,
                }]),
            }],
            static_samplers: vec![],
        });
        // Header (24) + 1 param header (12) + table header (8) = byte 44 is range_type.
        bad_range_type[44..48].copy_from_slice(&9u32.to_le_bytes());
        assert!(matches!(
            parse_rts0_chunk(&bad_range_type),
            Err(DxilReflectError::InvalidDescriptorRangeType {
                param_index: 0,
                range_index: 0,
                raw: 9,
            })
        ));

        // 7. ISG1 semantic name offset pointing into header (< 8).
        let mut header_alias_sig = encode_signature(
            true,
            &[sig_elem("POSITION", 0, 0, SignatureComponentType::Float32, 0, 0x7)],
        );
        // Header (8) + stream (4) = byte 12 is name_offset in ISG1.
        header_alias_sig[12..16].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            parse_signature_elements(*b"ISG1", &header_alias_sig),
            Err(DxilReflectError::InvalidSignatureString { fourcc: "ISG1", .. })
        ));

        // 8. PSV0 runtime_info_size < 24.
        let mut tiny_psv = vec![0u8; 20];
        tiny_psv[0..4].copy_from_slice(&12u32.to_le_bytes());
        assert!(matches!(
            parse_psv0_chunk(&tiny_psv),
            Err(DxilReflectError::TruncatedChunk {
                fourcc: "PSV0",
                actual: 12,
                expected_min: 24,
            })
        ));

        // 9. PSV0 truncated trailing resource_count (2 bytes after runtime info).
        let mut trunc_psv_tail = vec![0u8; 4 + 48 + 2];
        trunc_psv_tail[0..4].copy_from_slice(&48u32.to_le_bytes());
        assert!(matches!(
            parse_psv0_chunk(&trunc_psv_tail),
            Err(DxilReflectError::TruncatedChunk {
                fourcc: "PSV0",
                actual: 2,
                expected_min: 4,
            })
        ));

        // 10. PSV0 inverted register bounds (lower_bound > upper_bound).
        let inverted_psv = encode_psv0(
            0,
            [0, 0, 0],
            &[PsvResourceBinding {
                res_type: 3,
                space: 0,
                lower_bound: 8,
                upper_bound: 3,
                res_kind: Some(2),
                res_flags: Some(0),
            }],
        );
        assert!(matches!(
            parse_psv0_chunk(&inverted_psv),
            Err(DxilReflectError::InvalidResourceBounds {
                fourcc: "PSV0",
                index: 0,
                lower_bound: 8,
                upper_bound: 3,
            })
        ));

        // 11. Exhaustive prefix-truncation sweep over a full valid container and every chunk
        // payload: every strict prefix must return Err(...) and never panic.
        let full_rs = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![
                RootParameter {
                    visibility: D3dShaderVisibility::Vertex,
                    kind: RootParameterKind::Constants32Bit {
                        shader_register: 0,
                        register_space: 0,
                        num_32bit_values: 4,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::Pixel,
                    kind: RootParameterKind::DescriptorTable(vec![DescriptorRange {
                        range_type: DescriptorRangeType::Srv,
                        num_descriptors: 2,
                        base_shader_register: 0,
                        register_space: 0,
                        flags: 0,
                        offset_in_descriptors_from_table_start: 0,
                    }]),
                },
            ],
            static_samplers: vec![sample_static_sampler(0, 1, D3dShaderVisibility::Pixel, false)],
        };
        let rts0 = encode_rts0(&full_rs);
        let isg1 = encode_signature(
            true,
            &[sig_elem("POSITION", 0, 0, SignatureComponentType::Float32, 0, 0x7)],
        );
        let psv0 = encode_psv0(
            5,
            [8, 8, 1],
            &[PsvResourceBinding {
                res_type: 5,
                space: 0,
                lower_bound: 0,
                upper_bound: 0,
                res_kind: Some(12),
                res_flags: Some(0),
            }],
        );
        let dxil = build_dxil_chunk(5, 6, 6);
        let full_container = build_container(&[
            (*b"DXIL", &dxil),
            (*b"RTS0", &rts0),
            (*b"ISG1", &isg1),
            (*b"PSV0", &psv0),
        ]);

        for cut in 0..rts0.len() {
            assert!(
                parse_rts0_chunk(&rts0[..cut]).is_err(),
                "truncated RTS0 prefix of {cut} bytes must fail cleanly"
            );
        }
        for cut in 0..isg1.len() {
            assert!(
                parse_signature_elements(*b"ISG1", &isg1[..cut]).is_err(),
                "truncated ISG1 prefix of {cut} bytes must fail cleanly"
            );
        }
        for cut in 0..psv0.len() {
            // Note: exact cut at 4 + 48 = 52 is a valid PSV0 with zero trailing resource table.
            if cut == 52 {
                assert!(parse_psv0_chunk(&psv0[..cut]).is_ok());
            } else {
                assert!(
                    parse_psv0_chunk(&psv0[..cut]).is_err(),
                    "truncated PSV0 prefix of {cut} bytes must fail cleanly"
                );
            }
        }
        for cut in 0..full_container.len() {
            assert!(
                reflect_dxil_bytes(&full_container[..cut]).is_err(),
                "truncated DXBC container prefix of {cut} bytes must fail cleanly"
            );
        }
    }

    #[test]
    fn classifies_cbv_srv_uav_and_sampler_variants_across_rts0_and_psv0() {
        // Part A: Full RTS0 + PSV0 classification covering CBV, SRV (buffer + texture dims + MSAA),
        // UAV, and Sampler (table filtering + static filtering + static comparison).
        let rs = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Constants32Bit {
                        shader_register: 0,
                        register_space: 0,
                        num_32bit_values: 12,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Cbv {
                        shader_register: 1,
                        register_space: 0,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Srv {
                        shader_register: 0,
                        register_space: 1,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::Uav {
                        shader_register: 0,
                        register_space: 2,
                        flags: 0,
                    },
                },
                RootParameter {
                    visibility: D3dShaderVisibility::All,
                    kind: RootParameterKind::DescriptorTable(vec![
                        DescriptorRange {
                            range_type: DescriptorRangeType::Cbv,
                            num_descriptors: 1,
                            base_shader_register: 2,
                            register_space: 0,
                            flags: 0,
                            offset_in_descriptors_from_table_start: 0,
                        },
                        DescriptorRange {
                            range_type: DescriptorRangeType::Srv,
                            num_descriptors: 6,
                            base_shader_register: 1,
                            register_space: 1,
                            flags: 0,
                            offset_in_descriptors_from_table_start: 1,
                        },
                        DescriptorRange {
                            range_type: DescriptorRangeType::Uav,
                            num_descriptors: 1,
                            base_shader_register: 1,
                            register_space: 2,
                            flags: 0,
                            offset_in_descriptors_from_table_start: 7,
                        },
                        DescriptorRange {
                            range_type: DescriptorRangeType::Sampler,
                            num_descriptors: 1,
                            base_shader_register: 0,
                            register_space: 3,
                            flags: 0,
                            offset_in_descriptors_from_table_start: 8,
                        },
                    ]),
                },
            ],
            static_samplers: vec![
                sample_static_sampler(1, 3, D3dShaderVisibility::All, false),
                sample_static_sampler(2, 3, D3dShaderVisibility::All, true),
            ],
        };

        let psv0 = encode_psv0(
            0,
            [0, 0, 0],
            &[
                PsvResourceBinding {
                    res_type: 4, // SRVRaw -> ReadOnlyStorage buffer
                    space: 1,
                    lower_bound: 1,
                    upper_bound: 1,
                    res_kind: Some(10),
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 3, // Texture1D
                    space: 1,
                    lower_bound: 2,
                    upper_bound: 2,
                    res_kind: Some(1),
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 3, // Texture2DMS -> multisampled D2 UnfilterableFloat
                    space: 1,
                    lower_bound: 3,
                    upper_bound: 3,
                    res_kind: Some(3),
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 3, // Texture3D
                    space: 1,
                    lower_bound: 4,
                    upper_bound: 4,
                    res_kind: Some(4),
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 3, // TextureCubeArray
                    space: 1,
                    lower_bound: 5,
                    upper_bound: 5,
                    res_kind: Some(9),
                    res_flags: Some(0),
                },
                // t6 has no PSV0 annotation -> defaults to Texture2D Float
            ],
        );

        let container = build_container(&[
            (*b"DXIL", &build_dxil_chunk(0, 6, 6)),
            (*b"RTS0", &encode_rts0(&rs)),
            (*b"PSV0", &psv0),
        ]);
        let refl = reflect_dxil_bytes(&container).expect("classification test should succeed");

        let find_reg = |class: RegisterClass, index: u32, space: u32| {
            refl.entry
                .bindings
                .iter()
                .find(|b| b.register == Some(D3dRegister { class, index, space }))
                .expect("expected reflected binding")
        };

        // CBVs (group 0)
        assert_eq!(
            find_reg(RegisterClass::B, 0, 0).layout,
            Some(BindingLayout::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: 48,
            })
        );
        assert!(find_reg(RegisterClass::B, 0, 0).root_constants);
        assert_eq!(find_reg(RegisterClass::B, 1, 0).resource, ResourceKind::Buffer);
        assert_eq!(find_reg(RegisterClass::B, 2, 0).resource, ResourceKind::Buffer);

        // SRVs (group 1)
        assert_eq!(
            find_reg(RegisterClass::T, 0, 1).layout,
            Some(BindingLayout::Buffer {
                ty: BufferBindingType::ReadOnlyStorage,
                has_dynamic_offset: false,
                min_binding_size: 0,
            })
        );
        assert_eq!(
            find_reg(RegisterClass::T, 1, 1).layout,
            Some(BindingLayout::Buffer {
                ty: BufferBindingType::ReadOnlyStorage,
                has_dynamic_offset: false,
                min_binding_size: 0,
            })
        );
        assert_eq!(
            find_reg(RegisterClass::T, 2, 1).layout,
            Some(BindingLayout::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::D1,
                multisampled: false,
            })
        );
        assert_eq!(
            find_reg(RegisterClass::T, 3, 1).layout,
            Some(BindingLayout::Texture {
                sample_type: TextureSampleType::UnfilterableFloat,
                view_dimension: TextureViewDimension::D2,
                multisampled: true,
            })
        );
        assert_eq!(
            find_reg(RegisterClass::T, 4, 1).layout,
            Some(BindingLayout::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::D3,
                multisampled: false,
            })
        );
        assert_eq!(
            find_reg(RegisterClass::T, 5, 1).layout,
            Some(BindingLayout::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::CubeArray,
                multisampled: false,
            })
        );
        assert_eq!(
            find_reg(RegisterClass::T, 6, 1).layout,
            Some(BindingLayout::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            })
        );

        // UAVs (group 2)
        for idx in [0u32, 1u32] {
            assert_eq!(
                find_reg(RegisterClass::U, idx, 2).layout,
                Some(BindingLayout::Buffer {
                    ty: BufferBindingType::Storage,
                    has_dynamic_offset: false,
                    min_binding_size: 0,
                })
            );
        }

        // Samplers (group 3)
        assert_eq!(
            find_reg(RegisterClass::S, 0, 3).layout,
            Some(BindingLayout::Sampler {
                ty: SamplerBindingType::Filtering,
            })
        );
        assert_eq!(
            find_reg(RegisterClass::S, 1, 3).layout,
            Some(BindingLayout::Sampler {
                ty: SamplerBindingType::Filtering,
            })
        );
        assert_eq!(
            find_reg(RegisterClass::S, 2, 3).layout,
            Some(BindingLayout::Sampler {
                ty: SamplerBindingType::Comparison,
            })
        );

        // Part B: PSV0-only fallback (no RTS0 chunk) classifies all 4 register classes canonically.
        let psv_only = encode_psv0(
            5,
            [16, 4, 1],
            &[
                PsvResourceBinding {
                    res_type: 1, // Sampler s0
                    space: 0,
                    lower_bound: 0,
                    upper_bound: 0,
                    res_kind: Some(14),
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 8, // UAVStructured u0
                    space: 0,
                    lower_bound: 0,
                    upper_bound: 0,
                    res_kind: Some(12),
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 5, // SRVStructured t0
                    space: 0,
                    lower_bound: 0,
                    upper_bound: 0,
                    res_kind: Some(12),
                    res_flags: Some(0),
                },
                PsvResourceBinding {
                    res_type: 2, // CBV b0
                    space: 0,
                    lower_bound: 0,
                    upper_bound: 0,
                    res_kind: Some(13),
                    res_flags: Some(0),
                },
            ],
        );
        let psv_container =
            build_container(&[(*b"DXIL", &build_dxil_chunk(5, 6, 6)), (*b"PSV0", &psv_only)]);
        let psv_refl = reflect_dxil_bytes(&psv_container).expect("PSV0-only reflection should succeed");
        assert_eq!(psv_refl.entry.stage, ShaderStage::Compute);
        assert_eq!(psv_refl.entry.workgroup_size, Some([16, 4, 1]));
        let classes: Vec<RegisterClass> = psv_refl
            .entry
            .bindings
            .iter()
            .map(|b| b.register.unwrap().class)
            .collect();
        // Canonical order: B (CBV) -> T (SRV) -> U (UAV) -> S (Sampler)
        assert_eq!(
            classes,
            vec![
                RegisterClass::B,
                RegisterClass::T,
                RegisterClass::U,
                RegisterClass::S,
            ]
        );
    }

    #[test]
    fn manifest_dedup_is_order_independent_and_detects_shader_hash_collisions() {
        let rs = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: vec![RootParameter {
                visibility: D3dShaderVisibility::All,
                kind: RootParameterKind::Cbv {
                    shader_register: 0,
                    register_space: 0,
                    flags: 0,
                },
            }],
            static_samplers: vec![],
        };
        let bytes = build_container(&[
            (*b"DXIL", &build_dxil_chunk(5, 6, 6)),
            (*b"RTS0", &encode_rts0(&rs)),
        ]);
        let conv = RegisterConvention::default();
        let refl_z = reflect_dxil_bytes_with_options(&bytes, Some("z_menu.dxil"), &conv).unwrap();
        let refl_a = reflect_dxil_bytes_with_options(&bytes, Some("a_menu.dxil"), &conv).unwrap();

        let errs = vec![
            DxilReflectionErrorRecord {
                source: "bad.dxil".to_string(),
                error: "truncated".to_string(),
            },
            DxilReflectionErrorRecord {
                source: "bad.dxil".to_string(),
                error: "truncated".to_string(),
            },
        ];

        let m_za =
            build_dxil_reflection_manifest(&[refl_z.clone(), refl_a.clone()], &errs, &conv).unwrap();
        let m_az =
            build_dxil_reflection_manifest(&[refl_a.clone(), refl_z.clone()], &errs, &conv).unwrap();
        assert_eq!(m_za.render_json(), m_az.render_json());
        assert_eq!(
            m_za.shaders.values().next().unwrap().source.as_deref(),
            Some("a_menu.dxil")
        );
        assert_eq!(m_za.errors.len(), 1);

        // Synthetic hash collision (same hash, different entry stage) must error cleanly.
        let mut collided = refl_a.clone();
        collided.entry.stage = ShaderStage::Fragment;
        assert!(matches!(
            build_dxil_reflection_manifest(&[refl_a, collided], &[], &conv),
            Err(DxilReflectError::HashCollision {
                table: "shaders",
                ..
            })
        ));
    }
}
