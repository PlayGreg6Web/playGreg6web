//! Root-signature-to-WebGPU `GPUBindGroupLayout` reflection and the layout cache manifest.
//!
//! Converted shaders are reflected from validated Naga IR (the same IR `convert` lowers to WGSL)
//! into exact WebGPU bind group layout entries, vertex inputs, fragment outputs and compute
//! workgroup sizes, so `wgpu_worker.js` can call `device.createBindGroupLayout` /
//! `createPipelineLayout` without parsing shaders, and identical layouts are shared across
//! pipelines through content-hash ids.
//!
//! # Reflection rules
//!
//! * Only resources **statically used** by an entry point (Naga `FunctionInfo` global uses,
//!   including callees) are reflected; declared-but-unused globals are skipped.
//! * Buffers: `var<uniform>` -> `"uniform"`, `var<storage, read_write>` (or atomics) ->
//!   `"storage"`, `var<storage, read>` -> `"read-only-storage"`. `minBindingSize` is the WGSL
//!   `SizeOf` of the store type with a runtime-sized array tail counted as one element
//!   (fixed part + one stride). `hasDynamicOffset` is always `false`: dynamic offsets are a
//!   root-signature property (root CBV/SRV/UAV descriptors) the runtime decides.
//! * Filterability follows the WebGPU default ("auto") pipeline layout rule: an `f32` sampled
//!   texture is `"float"` if the entry point (or a callee) passes it to a sampling builtin together
//!   with a sampler, otherwise `"unfilterable-float"`; multisampled textures are always
//!   `"unfilterable-float"`. Samplers are `"comparison"` when declared as `sampler_comparison`,
//!   otherwise `"filtering"`. `"non-filtering"` is never emitted: whether a D3D static sampler
//!   pairs with an unfilterable format is a runtime pairing property. When stages are merged,
//!   `"float"` wins over `"unfilterable-float"`.
//! * Depth textures -> `"depth"`, `i32`/`u32` textures -> `"sint"`/`"uint"`; view dimensions
//!   `1d`, `2d`, `2d-array`, `cube`, `cube-array`, `3d`.
//! * Storage textures: access `"write-only"` / `"read-only"` / `"read-write"`, WebGPU texel
//!   format name, view dimension.
//!
//! Anything that has no valid baseline WebGPU layout is **not** emitted as a layout entry; it is
//! listed under `requires_lowering` instead (matching S3's `BindlessDescriptorIndexing` blocker):
//! `binding_array<T, N>` (`BindingArray`, count recorded), unbounded `binding_array<T>`
//! (`UnboundedBindingArray`), remaining push-constant / immediate blocks (`PushConstants`),
//! external textures / acceleration structures (`UnsupportedResource`), 1d-array or multisampled
//! non-2d textures and cube storage textures (`UnsupportedViewDimension`), storage texture formats
//! or `read-write` access outside WebGPU core (`StorageTextureFormat`), vertex attribute types
//! without a `GPUVertexFormat` (`VertexInputFormat`) and pipeline-overridable workgroup sizes
//! (`OverridableWorkgroupSize`).
//!
//! # D3D register join key
//!
//! The runtime (`runtime/gpu/descriptor_heap.*`) flattens root signatures into
//! `(register class, register, space)` slots. Each reflected binding carries that key, inferred
//! with the DXC `-spirv` convention the Xbox -> SPIR-V step is assumed to follow:
//!
//! * `register(<class><N>, space<M>)` -> SPIR-V `DescriptorSet = M`, `Binding = N + shift(class)`,
//!   where `shift` is DXC's `-fvk-{b,t,u,s}-shift` value ([`RegisterConvention`], default `0`).
//! * The register class is recovered from the resource kind: uniform buffer -> `b` (CBV);
//!   sampled/depth texture, read-only storage buffer (`StructuredBuffer`/`ByteAddressBuffer`) ->
//!   `t` (SRV); read-write storage buffer or any storage texture (`RW*`) -> `u` (UAV);
//!   sampler -> `s`.
//! * Bindings that S4's lowering moved out of `group >= 4` report their original
//!   `DescriptorSet`/`Binding` (from [`LoweringReport::bind_group_remaps`]) in `spirv` and the
//!   register derived from it; push-constant blocks that S4 lowered to a uniform buffer are
//!   flagged `root_constants: true` with `register: null` (root constants have no register in
//!   the SPIR-V).
//!
//! The raw `spirv` set/binding is always recorded too, so the runtime can still join if the
//! upstream convention turns out to differ.
//!
//! # Manifest schema (`layout-manifest.json`, `schema_version` 1)
//!
//! Written next to `shader-index.json`; deterministic (sorted keys, canonical entry order) so the
//! same input set in any order yields byte-identical output. Ids are `0x` + 16 hex digit FNV-1a 64
//! content hashes (the same format as shader hashes) and are stable across runs and packs.
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "register_convention": {"source":"dxc-spirv","shifts":{"b":0,"t":0,"u":0,"s":0}},
//!   "bind_group_layouts": {
//!     "0x<bgl id>": { "entries": [ <GPUBindGroupLayoutEntry>, ... ] }
//!   },
//!   "pipeline_layouts": {
//!     "0x<pipeline layout id>": { "bindGroupLayouts": ["0x<bgl id>", ...] }
//!   },
//!   "shaders": {
//!     "0x<shader hash>": {
//!       "stage": "vertex" | "fragment" | "compute",
//!       "entry_point": "main",
//!       "pipeline_layout": "0x<pipeline layout id>",
//!       "bind_group_layouts": ["0x<bgl id>", ...],
//!       "layout_complete": true,
//!       "workgroup_size": [x, y, z] | null,
//!       "bindings": [ <binding record>, ... ],
//!       "vertex_inputs": [ {"location":0,"type":"vec3<f32>","format":"float32x3"}, ... ],
//!       "fragment_outputs": [ {"location":0,"type":"vec4<f32>"}, ... ],
//!       "requires_lowering": [ {"kind":"BindingArray","group":1,"binding":0,"detail":"..."} ],
//!       "limit_violations": [ {"limit":"maxSampledTexturesPerShaderStage","stage":"fragment","count":17,"max":16} ]
//!     }
//!   },
//!   "pipelines": {
//!     "0x<vs hash>+0x<fs hash>": {
//!       "stages": {"vertex":"0x<vs hash>","fragment":"0x<fs hash>"},
//!       "pipeline_layout": "0x<pipeline layout id>",
//!       "bind_group_layouts": ["0x<bgl id>", ...],
//!       "layout_complete": true,
//!       "limit_violations": []
//!     }
//!   }
//! }
//! ```
//!
//! * `<GPUBindGroupLayoutEntry>` is exactly the WebGPU dictionary: `binding`, `visibility`
//!   (`GPUShaderStage` bits: vertex `1`, fragment `2`, compute `4`) and exactly one of
//!   `buffer {type, hasDynamicOffset, minBindingSize}`, `sampler {type}`,
//!   `texture {sampleType, viewDimension, multisampled}` or
//!   `storageTexture {access, format, viewDimension}`.
//! * `bindGroupLayouts` lists one id per group index `0..n`; unused group indices below the
//!   highest used group reference the empty layout (`entries: []`).
//! * `<binding record>` = `{"group", "binding", "name", "resource"` (`buffer` | `sampler` |
//!   `texture` | `storageTexture` | `externalTexture` | `accelerationStructure`), `"in_layout"`
//!   (false when listed in `requires_lowering`), `"register"` (`{"class","index","space"}` or
//!   `null`), `"spirv"` (`{"set","binding"}` before S4 remapping, or `null`),
//!   `"root_constants"`, `"binding_array"` (`null` or `{"count": N | null}`)`}`.
//! * A shader's `pipeline_layout` covers that stage alone (complete for compute and vertex-only
//!   pipelines). Graphics pipelines listed with `reflect --pipelines` get a merged entry under
//!   `pipelines` (visibility union across stages, keyed by `+`-joined stage-ordered hashes).
//! * `layout_complete: false` means some used binding is only in `requires_lowering`, so the
//!   layout cannot be used for that shader as-is.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};

use naga::front::spv;
use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, Binding, ImageClass, ImageDimension, ScalarKind, StorageAccess,
    StorageFormat, TypeInner,
};

use crate::compat::{
    WEBGPU_MAX_BIND_GROUPS, WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
    WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE, sanitize_spirv_bytes,
};
use crate::convert::{
    ConversionError, ConvertOptions, ShaderStage, convert_spirv_bytes_with_options, fnv1a_64,
    format_shader_hash,
};
use crate::lower::{LoweringReport, WEBGPU_MAX_BINDINGS_PER_GROUP, lower_module_for_webgpu};
use crate::pack::write_json_string;

/// Layout manifest schema version (`schema_version`).
pub const LAYOUT_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// File name of the layout cache manifest, written next to `shader-index.json`.
pub const LAYOUT_MANIFEST_FILE_NAME: &str = "layout-manifest.json";

/// WebGPU baseline default `maxSampledTexturesPerShaderStage`.
pub const WEBGPU_MAX_SAMPLED_TEXTURES_PER_SHADER_STAGE: usize = 16;

/// WebGPU baseline default `maxSamplersPerShaderStage`.
pub const WEBGPU_MAX_SAMPLERS_PER_SHADER_STAGE: usize = 16;

/// WebGPU baseline default `maxStorageTexturesPerShaderStage`.
pub const WEBGPU_MAX_STORAGE_TEXTURES_PER_SHADER_STAGE: usize = 4;

/// WebGPU baseline default `maxDynamicUniformBuffersPerPipelineLayout`.
pub const WEBGPU_MAX_DYNAMIC_UNIFORM_BUFFERS_PER_PIPELINE_LAYOUT: usize = 8;

/// WebGPU baseline default `maxDynamicStorageBuffersPerPipelineLayout`.
pub const WEBGPU_MAX_DYNAMIC_STORAGE_BUFFERS_PER_PIPELINE_LAYOUT: usize = 4;

/// WebGPU baseline default `maxUniformBufferBindingSize`.
pub const WEBGPU_MAX_UNIFORM_BUFFER_BINDING_SIZE: u64 = 65_536;

/// WebGPU baseline default `maxStorageBufferBindingSize`.
pub const WEBGPU_MAX_STORAGE_BUFFER_BINDING_SIZE: u64 = 134_217_728;

/// WebGPU baseline default `maxVertexAttributes`.
pub const WEBGPU_MAX_VERTEX_ATTRIBUTES: u32 = 16;

/// WebGPU baseline default `maxColorAttachments`.
pub const WEBGPU_MAX_COLOR_ATTACHMENTS: u32 = 8;

/// `GPUShaderStage.VERTEX`.
pub const VISIBILITY_VERTEX: u32 = 0x1;
/// `GPUShaderStage.FRAGMENT`.
pub const VISIBILITY_FRAGMENT: u32 = 0x2;
/// `GPUShaderStage.COMPUTE`.
pub const VISIBILITY_COMPUTE: u32 = 0x4;

/// Sanity bound on reflected group indices so a corrupt `DescriptorSet` cannot allocate a huge
/// dense pipeline layout (WebGPU itself only allows `0..4`).
const MAX_REFLECTED_GROUPS: u32 = 64;

/// `GPUShaderStage` visibility bit for a pipeline stage.
pub const fn stage_visibility(stage: ShaderStage) -> u32 {
    match stage {
        ShaderStage::Vertex => VISIBILITY_VERTEX,
        ShaderStage::Fragment => VISIBILITY_FRAGMENT,
        ShaderStage::Compute => VISIBILITY_COMPUTE,
    }
}

// ---------------------------------------------------------------------------
// WebGPU bind group layout model
// ---------------------------------------------------------------------------

/// `GPUBufferBindingType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BufferBindingType {
    Uniform,
    Storage,
    ReadOnlyStorage,
}

impl BufferBindingType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Storage => "storage",
            Self::ReadOnlyStorage => "read-only-storage",
        }
    }
}

/// `GPUSamplerBindingType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SamplerBindingType {
    Filtering,
    NonFiltering,
    Comparison,
}

impl SamplerBindingType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Filtering => "filtering",
            Self::NonFiltering => "non-filtering",
            Self::Comparison => "comparison",
        }
    }
}

/// `GPUTextureSampleType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureSampleType {
    Float,
    UnfilterableFloat,
    Depth,
    Sint,
    Uint,
}

impl TextureSampleType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::UnfilterableFloat => "unfilterable-float",
            Self::Depth => "depth",
            Self::Sint => "sint",
            Self::Uint => "uint",
        }
    }
}

/// `GPUTextureViewDimension`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

impl TextureViewDimension {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::D1 => "1d",
            Self::D2 => "2d",
            Self::D2Array => "2d-array",
            Self::Cube => "cube",
            Self::CubeArray => "cube-array",
            Self::D3 => "3d",
        }
    }
}

/// `GPUStorageTextureAccess`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageTextureAccess {
    WriteOnly,
    ReadOnly,
    ReadWrite,
}

impl StorageTextureAccess {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WriteOnly => "write-only",
            Self::ReadOnly => "read-only",
            Self::ReadWrite => "read-write",
        }
    }
}

/// The typed half of a `GPUBindGroupLayoutEntry` (exactly one of its four layout members).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingLayout {
    Buffer {
        ty: BufferBindingType,
        has_dynamic_offset: bool,
        min_binding_size: u64,
    },
    Sampler {
        ty: SamplerBindingType,
    },
    Texture {
        sample_type: TextureSampleType,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: &'static str,
        view_dimension: TextureViewDimension,
    },
}

impl BindingLayout {
    /// The `GPUBindGroupLayoutEntry` member name this layout occupies.
    pub const fn member_name(&self) -> &'static str {
        match self {
            Self::Buffer { .. } => "buffer",
            Self::Sampler { .. } => "sampler",
            Self::Texture { .. } => "texture",
            Self::StorageTexture { .. } => "storageTexture",
        }
    }

    /// Compact canonical JSON `"<member>":{...}` fragment.
    pub fn member_json(&self) -> String {
        match *self {
            Self::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => format!(
                "\"buffer\":{{\"type\":\"{}\",\"hasDynamicOffset\":{has_dynamic_offset},\"minBindingSize\":{min_binding_size}}}",
                ty.as_str()
            ),
            Self::Sampler { ty } => format!("\"sampler\":{{\"type\":\"{}\"}}", ty.as_str()),
            Self::Texture {
                sample_type,
                view_dimension,
                multisampled,
            } => format!(
                "\"texture\":{{\"sampleType\":\"{}\",\"viewDimension\":\"{}\",\"multisampled\":{multisampled}}}",
                sample_type.as_str(),
                view_dimension.as_str()
            ),
            Self::StorageTexture {
                access,
                format,
                view_dimension,
            } => format!(
                "\"storageTexture\":{{\"access\":\"{}\",\"format\":\"{format}\",\"viewDimension\":\"{}\"}}",
                access.as_str(),
                view_dimension.as_str()
            ),
        }
    }
}

/// One `GPUBindGroupLayoutEntry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayoutEntry {
    pub binding: u32,
    /// `GPUShaderStage` bit set ([`VISIBILITY_VERTEX`] | [`VISIBILITY_FRAGMENT`] | [`VISIBILITY_COMPUTE`]).
    pub visibility: u32,
    pub layout: BindingLayout,
}

impl LayoutEntry {
    /// Compact canonical JSON in exact `GPUBindGroupLayoutEntry` shape.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"binding\":{},\"visibility\":{},{}}}",
            self.binding,
            self.visibility,
            self.layout.member_json()
        )
    }
}

/// A `GPUBindGroupLayoutDescriptor` with entries in canonical (ascending `binding`) order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BindGroupLayoutDesc {
    pub entries: Vec<LayoutEntry>,
}

impl BindGroupLayoutDesc {
    /// Canonical compact JSON of the `entries` array.
    pub fn canonical_json(&self) -> String {
        let items: Vec<String> = self.entries.iter().map(LayoutEntry::to_json).collect();
        format!("[{}]", items.join(","))
    }

    /// Stable content hash (layout id): FNV-1a 64 over `"bgl:" + canonical_json()`.
    pub fn content_hash(&self) -> u64 {
        fnv1a_64(format!("bgl:{}", self.canonical_json()).as_bytes())
    }
}

/// A pipeline layout: one bind group layout per group index `0..n` (holes are empty layouts).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PipelineLayoutDesc {
    pub bind_group_layouts: Vec<BindGroupLayoutDesc>,
}

impl PipelineLayoutDesc {
    /// Bind group layout ids in group order.
    pub fn bind_group_layout_ids(&self) -> Vec<u64> {
        self.bind_group_layouts
            .iter()
            .map(BindGroupLayoutDesc::content_hash)
            .collect()
    }

    /// Stable content hash (pipeline layout id): FNV-1a 64 over `"pl:[<bgl ids>]"`.
    pub fn content_hash(&self) -> u64 {
        pipeline_layout_hash(&self.bind_group_layout_ids())
    }
}

fn pipeline_layout_hash(bgl_ids: &[u64]) -> u64 {
    let ids: Vec<String> = bgl_ids.iter().map(|&id| format_shader_hash(id)).collect();
    fnv1a_64(format!("pl:[{}]", ids.join(",")).as_bytes())
}

// ---------------------------------------------------------------------------
// Reflection model
// ---------------------------------------------------------------------------

/// D3D12 register class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RegisterClass {
    /// Constant buffer view (`b`).
    B,
    /// Shader resource view (`t`).
    T,
    /// Unordered access view (`u`).
    U,
    /// Sampler (`s`).
    S,
}

impl RegisterClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::B => "b",
            Self::T => "t",
            Self::U => "u",
            Self::S => "s",
        }
    }
}

/// DXC `-spirv` register mapping (`Binding = register + shift(class)`, `DescriptorSet = space`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegisterConvention {
    pub shift_b: u32,
    pub shift_t: u32,
    pub shift_u: u32,
    pub shift_s: u32,
}

impl RegisterConvention {
    pub const fn shift(&self, class: RegisterClass) -> u32 {
        match class {
            RegisterClass::B => self.shift_b,
            RegisterClass::T => self.shift_t,
            RegisterClass::U => self.shift_u,
            RegisterClass::S => self.shift_s,
        }
    }

    /// Set a shift from a `<class>=<n>` CLI argument (for example `t=1000`).
    pub fn set_shift_arg(&mut self, arg: &str) -> Result<(), String> {
        let (class, value) = arg
            .split_once('=')
            .ok_or_else(|| format!("invalid register shift '{arg}' (expected <b|t|u|s>=<n>)"))?;
        let value: u32 = value
            .trim()
            .parse()
            .map_err(|_| format!("invalid register shift value in '{arg}'"))?;
        match class.trim() {
            "b" => self.shift_b = value,
            "t" => self.shift_t = value,
            "u" => self.shift_u = value,
            "s" => self.shift_s = value,
            other => {
                return Err(format!(
                    "unknown register class '{other}' (expected b, t, u or s)"
                ));
            }
        }
        Ok(())
    }

    /// Recover the D3D register from a SPIR-V `DescriptorSet`/`Binding` pair, if the binding is
    /// consistent with this convention's shift for `class`.
    pub fn register_for(
        &self,
        class: RegisterClass,
        set: u32,
        binding: u32,
    ) -> Option<D3dRegister> {
        Some(D3dRegister {
            class,
            index: binding.checked_sub(self.shift(class))?,
            space: set,
        })
    }

    fn to_json(self) -> String {
        format!(
            "{{\"source\":\"dxc-spirv\",\"shifts\":{{\"b\":{},\"t\":{},\"u\":{},\"s\":{}}}}}",
            self.shift_b, self.shift_t, self.shift_u, self.shift_s
        )
    }
}

/// Original D3D register (`register(<class><index>, space<space>)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct D3dRegister {
    pub class: RegisterClass,
    pub index: u32,
    pub space: u32,
}

/// Original SPIR-V `DescriptorSet` / `Binding` decorations (before S4 group remapping).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpirvBinding {
    pub set: u32,
    pub binding: u32,
}

/// Kind of resource behind a binding (including kinds WebGPU cannot express).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResourceKind {
    Buffer,
    Sampler,
    Texture,
    StorageTexture,
    ExternalTexture,
    AccelerationStructure,
}

impl ResourceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Sampler => "sampler",
            Self::Texture => "texture",
            Self::StorageTexture => "storageTexture",
            Self::ExternalTexture => "externalTexture",
            Self::AccelerationStructure => "accelerationStructure",
        }
    }
}

/// Size of a `binding_array`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingArraySize {
    Fixed(u32),
    /// Runtime-sized (or override-sized) array.
    Unbounded,
}

/// Why a reflected item has no valid baseline WebGPU representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoweringKind {
    BindingArray,
    UnboundedBindingArray,
    PushConstants,
    UnsupportedResource,
    UnsupportedViewDimension,
    StorageTextureFormat,
    VertexInputFormat,
    OverridableWorkgroupSize,
}

impl LoweringKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BindingArray => "BindingArray",
            Self::UnboundedBindingArray => "UnboundedBindingArray",
            Self::PushConstants => "PushConstants",
            Self::UnsupportedResource => "UnsupportedResource",
            Self::UnsupportedViewDimension => "UnsupportedViewDimension",
            Self::StorageTextureFormat => "StorageTextureFormat",
            Self::VertexInputFormat => "VertexInputFormat",
            Self::OverridableWorkgroupSize => "OverridableWorkgroupSize",
        }
    }
}

/// Something the offline pipeline must still lower before the manifest layout is usable.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoweringNeed {
    pub kind: LoweringKind,
    pub group: Option<u32>,
    pub binding: Option<u32>,
    pub detail: String,
}

impl LoweringNeed {
    fn to_json(&self) -> String {
        let mut out = format!(
            "{{\"kind\":\"{}\",\"group\":{},\"binding\":{},\"detail\":",
            self.kind.as_str(),
            json_opt_u32(self.group),
            json_opt_u32(self.binding)
        );
        write_json_string(&mut out, &self.detail);
        out.push('}');
        out
    }
}

/// One resource binding statically used by an entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReflectedBinding {
    /// Final WGSL `@group`.
    pub group: u32,
    /// Final WGSL `@binding`.
    pub binding: u32,
    pub name: Option<String>,
    pub resource: ResourceKind,
    /// The WebGPU layout, or `None` when the binding is listed in `requires_lowering`.
    pub layout: Option<BindingLayout>,
    pub binding_array: Option<BindingArraySize>,
    /// D3D register join key (see module docs), when recoverable.
    pub register: Option<D3dRegister>,
    /// Original SPIR-V set/binding, or `None` for lowered root constants.
    pub spirv: Option<SpirvBinding>,
    /// `true` for a push-constant / root-constant block S4 lowered into a uniform buffer.
    pub root_constants: bool,
}

impl ReflectedBinding {
    fn to_json(&self) -> String {
        let mut out = format!(
            "{{\"group\":{},\"binding\":{},\"name\":",
            self.group, self.binding
        );
        match &self.name {
            Some(name) => write_json_string(&mut out, name),
            None => out.push_str("null"),
        }
        let _ = write!(
            out,
            ",\"resource\":\"{}\",\"in_layout\":{},\"register\":",
            self.resource.as_str(),
            self.layout.is_some()
        );
        match self.register {
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
        match self.spirv {
            Some(s) => {
                let _ = write!(out, "{{\"set\":{},\"binding\":{}}}", s.set, s.binding);
            }
            None => out.push_str("null"),
        }
        let _ = write!(
            out,
            ",\"root_constants\":{},\"binding_array\":",
            self.root_constants
        );
        match self.binding_array {
            Some(BindingArraySize::Fixed(n)) => {
                let _ = write!(out, "{{\"count\":{n}}}");
            }
            Some(BindingArraySize::Unbounded) => out.push_str("{\"count\":null}"),
            None => out.push_str("null"),
        }
        out.push('}');
        out
    }
}

/// A vertex input or fragment output at a `@location`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderIo {
    pub location: u32,
    /// WGSL type, e.g. `vec4<f32>`.
    pub wgsl_type: String,
    /// Natural `GPUVertexFormat` for the type (vertex inputs only). The D3D input layout may pick
    /// any format of the same base type (for example `unorm8x4` for `vec4<f32>`).
    pub vertex_format: Option<&'static str>,
}

/// Reflection of one entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPointReflection {
    pub name: String,
    pub stage: ShaderStage,
    /// Statically used bindings, sorted by `(group, binding)`.
    pub bindings: Vec<ReflectedBinding>,
    pub vertex_inputs: Vec<ShaderIo>,
    pub fragment_outputs: Vec<ShaderIo>,
    pub workgroup_size: Option<[u32; 3]>,
    pub requires_lowering: Vec<LoweringNeed>,
}

impl EntryPointReflection {
    /// `true` when every used binding has a WebGPU layout entry.
    pub fn layout_complete(&self) -> bool {
        self.bindings.iter().all(|b| b.layout.is_some())
    }
}

/// Reflection of one converted shader, keyed by its `shader-index.json` hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderReflection {
    pub hash: u64,
    pub entry: EntryPointReflection,
}

/// One side of a pipeline binding conflict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingSide {
    pub stage: ShaderStage,
    pub entry_point: String,
    pub name: Option<String>,
    /// Canonical `"<member>":{...}` layout JSON.
    pub layout: String,
}

impl fmt::Display for BindingSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} '{}' ({}) {{{}}}",
            self.stage,
            self.entry_point,
            self.name.as_deref().unwrap_or("<unnamed>"),
            self.layout
        )
    }
}

/// Errors produced by reflection, pipeline merging and manifest building.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectError {
    /// The shader did not convert (same classification as `convert`).
    Conversion(ConversionError),
    /// The SPIR-V front end failed after `convert` succeeded (should not happen).
    FrontEnd(String),
    UnsupportedStage {
        entry_point: String,
        stage: String,
    },
    InvalidPipelineStages {
        stages: Vec<ShaderStage>,
    },
    BindingConflict {
        group: u32,
        binding: u32,
        first: Box<BindingSide>,
        second: Box<BindingSide>,
    },
    GroupIndexTooLarge {
        group: u32,
        max: u32,
    },
    HashCollision {
        table: &'static str,
        hash: u64,
    },
    DuplicateShader(u64),
    UnknownShader(u64),
}

impl fmt::Display for ReflectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conversion(err) => write!(f, "shader conversion failed: {err}"),
            Self::FrontEnd(detail) => {
                write!(f, "SPIR-V front end failed during reflection: {detail}")
            }
            Self::UnsupportedStage { entry_point, stage } => write!(
                f,
                "entry point '{entry_point}' uses stage {stage}, which has no WebGPU pipeline"
            ),
            Self::InvalidPipelineStages { stages } => {
                let names: Vec<&str> = stages.iter().map(|s| s.as_str()).collect();
                write!(
                    f,
                    "invalid pipeline stage combination [{}] (expected compute, vertex, or vertex+fragment)",
                    names.join(", ")
                )
            }
            Self::BindingConflict {
                group,
                binding,
                first,
                second,
            } => write!(
                f,
                "@group({group}) @binding({binding}) has incompatible layouts: {first} vs {second}"
            ),
            Self::GroupIndexTooLarge { group, max } => write!(
                f,
                "bind group index {group} exceeds the reflection bound of {max} groups"
            ),
            Self::HashCollision { table, hash } => write!(
                f,
                "content hash collision in {table} for id {}",
                format_shader_hash(*hash)
            ),
            Self::DuplicateShader(hash) => write!(
                f,
                "shader {} was reflected twice with different results",
                format_shader_hash(*hash)
            ),
            Self::UnknownShader(hash) => write!(
                f,
                "pipeline references unknown shader {}",
                format_shader_hash(*hash)
            ),
        }
    }
}

impl std::error::Error for ReflectError {}

// ---------------------------------------------------------------------------
// Reflection from Naga IR
// ---------------------------------------------------------------------------

/// Reflect every entry point of a validated module.
///
/// `lowering` is the report of S4's `lower_module_for_webgpu` run on this module (if any); it is
/// used to recover original SPIR-V sets/bindings and to flag lowered root constants.
pub fn reflect_module(
    module: &naga::Module,
    info: &ModuleInfo,
    lowering: Option<&LoweringReport>,
    convention: &RegisterConvention,
) -> Result<Vec<EntryPointReflection>, ReflectError> {
    (0..module.entry_points.len())
        .map(|index| reflect_entry_point(module, info, index, lowering, convention))
        .collect()
}

/// Reflect entry point `index` of a validated module (see [`reflect_module`]).
pub fn reflect_entry_point(
    module: &naga::Module,
    info: &ModuleInfo,
    index: usize,
    lowering: Option<&LoweringReport>,
    convention: &RegisterConvention,
) -> Result<EntryPointReflection, ReflectError> {
    let ep = &module.entry_points[index];
    let stage = match ep.stage {
        naga::ShaderStage::Vertex => ShaderStage::Vertex,
        naga::ShaderStage::Fragment => ShaderStage::Fragment,
        naga::ShaderStage::Compute => ShaderStage::Compute,
        other => {
            return Err(ReflectError::UnsupportedStage {
                entry_point: ep.name.clone(),
                stage: format!("{other:?}"),
            });
        }
    };
    let ep_info = info.get_entry_point(index);
    let gctx = module.to_ctx();

    let sampled_images: BTreeSet<usize> = ep_info
        .sampling_set
        .iter()
        .map(|key| key.image.index())
        .collect();

    let mut original_bindings: BTreeMap<(u32, u32), SpirvBinding> = BTreeMap::new();
    let mut root_constant_slots: BTreeSet<(u32, u32)> = BTreeSet::new();
    if let Some(report) = lowering {
        for remap in &report.bind_group_remaps {
            original_bindings.insert(
                (remap.new_group, remap.new_binding),
                SpirvBinding {
                    set: remap.old_group,
                    binding: remap.old_binding,
                },
            );
        }
        for remap in &report.push_constant_remaps {
            root_constant_slots.insert((remap.new_group, remap.new_binding));
        }
    }

    let mut bindings = Vec::new();
    let mut requires_lowering = Vec::new();

    for (handle, var) in module.global_variables.iter() {
        if ep_info[handle].is_empty() {
            continue;
        }
        let label = var.name.as_deref().unwrap_or("<unnamed>");

        if var.space == AddressSpace::Immediate {
            let size = module.types[var.ty].inner.try_size(gctx).unwrap_or(0);
            requires_lowering.push(LoweringNeed {
                kind: LoweringKind::PushConstants,
                group: None,
                binding: None,
                detail: format!(
                    "push-constant block '{label}' ({size} bytes) must be lowered to a uniform buffer"
                ),
            });
            continue;
        }
        let Some(res_binding) = var.binding else {
            continue;
        };
        let (group, binding) = (res_binding.group, res_binding.binding);

        let (element_ty, binding_array) = match module.types[var.ty].inner {
            TypeInner::BindingArray { base, size } => (
                base,
                Some(match size {
                    ArraySize::Constant(n) => BindingArraySize::Fixed(n.get()),
                    ArraySize::Pending(_) | ArraySize::Dynamic => BindingArraySize::Unbounded,
                }),
            ),
            _ => (var.ty, None),
        };
        let element = &module.types[element_ty].inner;

        let (resource, class, mut layout) = match var.space {
            AddressSpace::Uniform => (
                ResourceKind::Buffer,
                Some(RegisterClass::B),
                Ok(BindingLayout::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: u64::from(element.try_size(gctx).unwrap_or(0)),
                }),
            ),
            AddressSpace::Storage { access } => {
                let writable = access.intersects(StorageAccess::STORE | StorageAccess::ATOMIC);
                (
                    ResourceKind::Buffer,
                    Some(if writable {
                        RegisterClass::U
                    } else {
                        RegisterClass::T
                    }),
                    Ok(BindingLayout::Buffer {
                        ty: if writable {
                            BufferBindingType::Storage
                        } else {
                            BufferBindingType::ReadOnlyStorage
                        },
                        has_dynamic_offset: false,
                        min_binding_size: u64::from(element.try_size(gctx).unwrap_or(0)),
                    }),
                )
            }
            AddressSpace::Handle => match *element {
                TypeInner::Sampler { comparison } => (
                    ResourceKind::Sampler,
                    Some(RegisterClass::S),
                    Ok(BindingLayout::Sampler {
                        ty: if comparison {
                            SamplerBindingType::Comparison
                        } else {
                            SamplerBindingType::Filtering
                        },
                    }),
                ),
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => image_layout(
                    dim,
                    arrayed,
                    class,
                    sampled_images.contains(&handle.index()),
                ),
                TypeInner::AccelerationStructure { .. } => (
                    ResourceKind::AccelerationStructure,
                    Some(RegisterClass::T),
                    Err((
                        LoweringKind::UnsupportedResource,
                        "acceleration structures are not available in WebGPU".to_string(),
                    )),
                ),
                _ => continue,
            },
            _ => continue,
        };

        if let Some(size) = binding_array {
            let (kind, count) = match size {
                BindingArraySize::Fixed(n) => (LoweringKind::BindingArray, n.to_string()),
                BindingArraySize::Unbounded => {
                    (LoweringKind::UnboundedBindingArray, "unbounded".to_string())
                }
            };
            layout = Err((
                kind,
                format!(
                    "binding_array ({count} elements) has no baseline WebGPU layout; lower to bounded individual bindings or a texture array"
                ),
            ));
        }

        let root_constants = root_constant_slots.contains(&(group, binding));
        let spirv = if root_constants {
            None
        } else {
            Some(
                original_bindings
                    .get(&(group, binding))
                    .copied()
                    .unwrap_or(SpirvBinding {
                        set: group,
                        binding,
                    }),
            )
        };
        let register = match (spirv, class) {
            (Some(spirv), Some(class)) => convention.register_for(class, spirv.set, spirv.binding),
            _ => None,
        };

        let layout = match layout {
            Ok(layout) => Some(layout),
            Err((kind, detail)) => {
                requires_lowering.push(LoweringNeed {
                    kind,
                    group: Some(group),
                    binding: Some(binding),
                    detail: format!("'{label}': {detail}"),
                });
                None
            }
        };

        bindings.push(ReflectedBinding {
            group,
            binding,
            name: var.name.clone(),
            resource,
            layout,
            binding_array,
            register,
            spirv,
            root_constants,
        });
    }

    bindings.sort_by(|a, b| (a.group, a.binding, &a.name).cmp(&(b.group, b.binding, &b.name)));

    let mut vertex_inputs = Vec::new();
    let mut fragment_outputs = Vec::new();
    match stage {
        ShaderStage::Vertex => {
            for arg in &ep.function.arguments {
                collect_io(module, arg.ty, arg.binding.as_ref(), &mut vertex_inputs);
            }
            vertex_inputs.sort_by_key(|io| io.location);
            for io in &vertex_inputs {
                if io.vertex_format.is_none() {
                    requires_lowering.push(LoweringNeed {
                        kind: LoweringKind::VertexInputFormat,
                        group: None,
                        binding: None,
                        detail: format!(
                            "vertex input @location({}) of type {} has no GPUVertexFormat",
                            io.location, io.wgsl_type
                        ),
                    });
                }
            }
        }
        ShaderStage::Fragment => {
            if let Some(result) = &ep.function.result {
                collect_io(
                    module,
                    result.ty,
                    result.binding.as_ref(),
                    &mut fragment_outputs,
                );
            }
            fragment_outputs.sort_by_key(|io| io.location);
            for io in &mut fragment_outputs {
                io.vertex_format = None;
            }
        }
        ShaderStage::Compute => {}
    }

    let workgroup_size = if stage == ShaderStage::Compute {
        if ep.workgroup_size_overrides.is_some() {
            requires_lowering.push(LoweringNeed {
                kind: LoweringKind::OverridableWorkgroupSize,
                group: None,
                binding: None,
                detail: "workgroup_size depends on pipeline-overridable constants".to_string(),
            });
            None
        } else {
            Some(ep.workgroup_size)
        }
    } else {
        None
    };

    requires_lowering.sort();
    requires_lowering.dedup();

    Ok(EntryPointReflection {
        name: ep.name.clone(),
        stage,
        bindings,
        vertex_inputs,
        fragment_outputs,
        workgroup_size,
        requires_lowering,
    })
}

type LayoutResult = Result<BindingLayout, (LoweringKind, String)>;

fn image_layout(
    dim: ImageDimension,
    arrayed: bool,
    class: ImageClass,
    sampled_with_sampler: bool,
) -> (ResourceKind, Option<RegisterClass>, LayoutResult) {
    let view_dimension = match (dim, arrayed) {
        (ImageDimension::D1, false) => Some(TextureViewDimension::D1),
        (ImageDimension::D2, false) => Some(TextureViewDimension::D2),
        (ImageDimension::D2, true) => Some(TextureViewDimension::D2Array),
        (ImageDimension::Cube, false) => Some(TextureViewDimension::Cube),
        (ImageDimension::Cube, true) => Some(TextureViewDimension::CubeArray),
        (ImageDimension::D3, false) => Some(TextureViewDimension::D3),
        (ImageDimension::D1 | ImageDimension::D3, true) => None,
    };
    let unsupported_dim = || {
        (
            LoweringKind::UnsupportedViewDimension,
            format!(
                "{dim:?}{} texture has no WebGPU view dimension",
                if arrayed { " array" } else { "" }
            ),
        )
    };

    match class {
        ImageClass::Sampled { kind, multi } => {
            let sample_type = match kind {
                ScalarKind::Float if sampled_with_sampler && !multi => TextureSampleType::Float,
                ScalarKind::Float => TextureSampleType::UnfilterableFloat,
                ScalarKind::Sint => TextureSampleType::Sint,
                ScalarKind::Uint => TextureSampleType::Uint,
                other => {
                    return (
                        ResourceKind::Texture,
                        Some(RegisterClass::T),
                        Err((
                            LoweringKind::UnsupportedResource,
                            format!("texture sample kind {other:?} has no GPUTextureSampleType"),
                        )),
                    );
                }
            };
            (
                ResourceKind::Texture,
                Some(RegisterClass::T),
                texture_layout(sample_type, view_dimension, multi).ok_or_else(unsupported_dim),
            )
        }
        ImageClass::Depth { multi } => (
            ResourceKind::Texture,
            Some(RegisterClass::T),
            texture_layout(TextureSampleType::Depth, view_dimension, multi)
                .ok_or_else(unsupported_dim),
        ),
        ImageClass::Storage { format, access } => {
            let layout = if access.contains(StorageAccess::ATOMIC) {
                Err((
                    LoweringKind::UnsupportedResource,
                    "atomic storage textures are not available in WebGPU".to_string(),
                ))
            } else {
                let access = match (
                    access.contains(StorageAccess::LOAD),
                    access.contains(StorageAccess::STORE),
                ) {
                    (true, true) => StorageTextureAccess::ReadWrite,
                    (false, true) => StorageTextureAccess::WriteOnly,
                    _ => StorageTextureAccess::ReadOnly,
                };
                match view_dimension {
                    Some(
                        view_dimension @ (TextureViewDimension::D1
                        | TextureViewDimension::D2
                        | TextureViewDimension::D2Array
                        | TextureViewDimension::D3),
                    ) => {
                        if storage_format_supported(format, access) {
                            Ok(BindingLayout::StorageTexture {
                                access,
                                format: storage_format_name(format),
                                view_dimension,
                            })
                        } else {
                            Err((
                                LoweringKind::StorageTextureFormat,
                                format!(
                                    "storage texture format {} with {} access is not supported by WebGPU core",
                                    storage_format_name(format),
                                    access.as_str()
                                ),
                            ))
                        }
                    }
                    _ => Err(unsupported_dim()),
                }
            };
            (ResourceKind::StorageTexture, Some(RegisterClass::U), layout)
        }
        ImageClass::External => (
            ResourceKind::ExternalTexture,
            Some(RegisterClass::T),
            Err((
                LoweringKind::UnsupportedResource,
                "external textures are not produced for D3D shaders".to_string(),
            )),
        ),
    }
}

fn texture_layout(
    sample_type: TextureSampleType,
    view_dimension: Option<TextureViewDimension>,
    multisampled: bool,
) -> Option<BindingLayout> {
    let view_dimension = view_dimension?;
    if multisampled && view_dimension != TextureViewDimension::D2 {
        return None;
    }
    Some(BindingLayout::Texture {
        sample_type,
        view_dimension,
        multisampled,
    })
}

/// WebGPU core storage texture formats (`read-write` only for the `r32*` formats).
fn storage_format_supported(format: StorageFormat, access: StorageTextureAccess) -> bool {
    use StorageFormat as F;
    if access == StorageTextureAccess::ReadWrite {
        return matches!(format, F::R32Uint | F::R32Sint | F::R32Float);
    }
    matches!(
        format,
        F::Rgba8Unorm
            | F::Rgba8Snorm
            | F::Rgba8Uint
            | F::Rgba8Sint
            | F::Rgba16Uint
            | F::Rgba16Sint
            | F::Rgba16Float
            | F::R32Uint
            | F::R32Sint
            | F::R32Float
            | F::Rg32Uint
            | F::Rg32Sint
            | F::Rg32Float
            | F::Rgba32Uint
            | F::Rgba32Sint
            | F::Rgba32Float
    )
}

/// `GPUTextureFormat` name of a Naga storage format.
fn storage_format_name(format: StorageFormat) -> &'static str {
    use StorageFormat as F;
    match format {
        F::R8Unorm => "r8unorm",
        F::R8Snorm => "r8snorm",
        F::R8Uint => "r8uint",
        F::R8Sint => "r8sint",
        F::R16Uint => "r16uint",
        F::R16Sint => "r16sint",
        F::R16Float => "r16float",
        F::Rg8Unorm => "rg8unorm",
        F::Rg8Snorm => "rg8snorm",
        F::Rg8Uint => "rg8uint",
        F::Rg8Sint => "rg8sint",
        F::R32Uint => "r32uint",
        F::R32Sint => "r32sint",
        F::R32Float => "r32float",
        F::Rg16Uint => "rg16uint",
        F::Rg16Sint => "rg16sint",
        F::Rg16Float => "rg16float",
        F::Rgba8Unorm => "rgba8unorm",
        F::Rgba8Snorm => "rgba8snorm",
        F::Rgba8Uint => "rgba8uint",
        F::Rgba8Sint => "rgba8sint",
        F::Bgra8Unorm => "bgra8unorm",
        F::Rgb10a2Uint => "rgb10a2uint",
        F::Rgb10a2Unorm => "rgb10a2unorm",
        F::Rg11b10Ufloat => "rg11b10ufloat",
        F::R64Uint => "r64uint",
        F::Rg32Uint => "rg32uint",
        F::Rg32Sint => "rg32sint",
        F::Rg32Float => "rg32float",
        F::Rgba16Uint => "rgba16uint",
        F::Rgba16Sint => "rgba16sint",
        F::Rgba16Float => "rgba16float",
        F::Rgba32Uint => "rgba32uint",
        F::Rgba32Sint => "rgba32sint",
        F::Rgba32Float => "rgba32float",
        F::R16Unorm => "r16unorm",
        F::R16Snorm => "r16snorm",
        F::Rg16Unorm => "rg16unorm",
        F::Rg16Snorm => "rg16snorm",
        F::Rgba16Unorm => "rgba16unorm",
        F::Rgba16Snorm => "rgba16snorm",
    }
}

fn collect_io(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&Binding>,
    out: &mut Vec<ShaderIo>,
) {
    match binding {
        Some(Binding::Location { location, .. }) => {
            let (wgsl_type, vertex_format) = io_type(&module.types[ty].inner);
            out.push(ShaderIo {
                location: *location,
                wgsl_type,
                vertex_format,
            });
        }
        Some(Binding::BuiltIn(_)) => {}
        None => {
            if let TypeInner::Struct { members, .. } = &module.types[ty].inner {
                for member in members {
                    collect_io(module, member.ty, member.binding.as_ref(), out);
                }
            }
        }
    }
}

fn io_type(inner: &TypeInner) -> (String, Option<&'static str>) {
    let (scalar, components) = match *inner {
        TypeInner::Scalar(scalar) => (scalar, 1u32),
        TypeInner::Vector { size, scalar } => (scalar, size as u32),
        _ => return ("unknown".to_string(), None),
    };
    let scalar_name = match (scalar.kind, scalar.width) {
        (ScalarKind::Float, 4) => "f32",
        (ScalarKind::Float, 2) => "f16",
        (ScalarKind::Sint, 4) => "i32",
        (ScalarKind::Uint, 4) => "u32",
        (ScalarKind::Bool, _) => "bool",
        _ => return ("unknown".to_string(), None),
    };
    let wgsl_type = if components == 1 {
        scalar_name.to_string()
    } else {
        format!("vec{components}<{scalar_name}>")
    };
    let format = match (scalar_name, components) {
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

/// Convert SPIR-V exactly like `convert` and reflect the converted entry point.
///
/// The shader must convert successfully (errors are `convert`'s own classification), so the
/// reflected set always matches what `shader-index.json` lists. The returned hash is the same
/// FNV-1a 64 hash `convert` uses.
pub fn reflect_spirv_bytes(
    spirv_bytes: &[u8],
    options: &ConvertOptions,
    convention: &RegisterConvention,
) -> Result<ShaderReflection, ReflectError> {
    let converted =
        convert_spirv_bytes_with_options(spirv_bytes, options).map_err(ReflectError::Conversion)?;

    // `convert` only returns WGSL, so re-run its front end to obtain the validated Naga IR.
    let (sanitized, _) = sanitize_spirv_bytes(spirv_bytes, &options.sanitize_options())
        .map_err(|err| ReflectError::FrontEnd(err.to_string()))?;
    let spv_options = spv::Options {
        adjust_coordinate_space: true,
        strict_capabilities: true,
        block_ctx_dump_prefix: None,
    };
    let mut module = spv::parse_u8_slice(&sanitized, &spv_options)
        .map_err(|err| ReflectError::FrontEnd(err.to_string()))?;
    let lowering = if options.lower_for_webgpu {
        lower_module_for_webgpu(&mut module)
    } else {
        LoweringReport::default()
    };
    let info = Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .map_err(|err| ReflectError::FrontEnd(err.emit_to_string("")))?;

    // `convert` emits the first entry point.
    let entry = reflect_entry_point(&module, &info, 0, Some(&lowering), convention)?;
    debug_assert_eq!(entry.name, converted.entry_point);
    Ok(ShaderReflection {
        hash: converted.hash,
        entry,
    })
}

// ---------------------------------------------------------------------------
// Pipeline merge and limits
// ---------------------------------------------------------------------------

/// Merge the entry points of one pipeline (`[compute]`, `[vertex]` or `[vertex, fragment]`, in
/// any order) into per-group bind group layouts with the union of stage visibilities.
///
/// Bindings without a WebGPU layout (see `requires_lowering`) are skipped. The same
/// `@group/@binding` used by several stages must have the same layout, except that buffer
/// `minBindingSize` takes the maximum and `float` wins over `unfilterable-float`; anything else
/// is a [`ReflectError::BindingConflict`] naming both sides.
pub fn merge_pipeline(
    stages: &[&EntryPointReflection],
) -> Result<PipelineLayoutDesc, ReflectError> {
    let mut kinds: Vec<ShaderStage> = stages.iter().map(|ep| ep.stage).collect();
    kinds.sort();
    let valid = matches!(
        kinds.as_slice(),
        [ShaderStage::Compute]
            | [ShaderStage::Vertex]
            | [ShaderStage::Vertex, ShaderStage::Fragment]
    );
    if !valid {
        return Err(ReflectError::InvalidPipelineStages { stages: kinds });
    }
    merge_stages(stages)
}

/// Bind group layouts for a single entry point on its own (visibility = its stage). This is the
/// per-shader layout stored in the manifest; for fragment shaders it is one half of a pipeline.
pub fn stage_layout(entry: &EntryPointReflection) -> Result<PipelineLayoutDesc, ReflectError> {
    merge_stages(&[entry])
}

fn merge_stages(stages: &[&EntryPointReflection]) -> Result<PipelineLayoutDesc, ReflectError> {
    let mut groups: BTreeMap<u32, BTreeMap<u32, (LayoutEntry, BindingSide)>> = BTreeMap::new();
    for ep in stages {
        let visibility = stage_visibility(ep.stage);
        for b in &ep.bindings {
            let Some(layout) = b.layout else {
                continue;
            };
            if b.group >= MAX_REFLECTED_GROUPS {
                return Err(ReflectError::GroupIndexTooLarge {
                    group: b.group,
                    max: MAX_REFLECTED_GROUPS,
                });
            }
            let side = BindingSide {
                stage: ep.stage,
                entry_point: ep.name.clone(),
                name: b.name.clone(),
                layout: layout.member_json(),
            };
            let slots = groups.entry(b.group).or_default();
            match slots.get_mut(&b.binding) {
                None => {
                    slots.insert(
                        b.binding,
                        (
                            LayoutEntry {
                                binding: b.binding,
                                visibility,
                                layout,
                            },
                            side,
                        ),
                    );
                }
                Some((entry, first)) => {
                    let Some(merged) = merge_layouts(entry.layout, layout) else {
                        return Err(ReflectError::BindingConflict {
                            group: b.group,
                            binding: b.binding,
                            first: Box::new(first.clone()),
                            second: Box::new(side),
                        });
                    };
                    entry.layout = merged;
                    entry.visibility |= visibility;
                }
            }
        }
    }

    let group_count = groups.keys().next_back().map_or(0, |&g| g + 1);
    let bind_group_layouts = (0..group_count)
        .map(|g| BindGroupLayoutDesc {
            entries: groups
                .get(&g)
                .map(|slots| slots.values().map(|(entry, _)| *entry).collect())
                .unwrap_or_default(),
        })
        .collect();
    Ok(PipelineLayoutDesc { bind_group_layouts })
}

fn merge_layouts(a: BindingLayout, b: BindingLayout) -> Option<BindingLayout> {
    if a == b {
        return Some(a);
    }
    match (a, b) {
        (
            BindingLayout::Buffer {
                ty: ty_a,
                has_dynamic_offset: dyn_a,
                min_binding_size: size_a,
            },
            BindingLayout::Buffer {
                ty: ty_b,
                has_dynamic_offset: dyn_b,
                min_binding_size: size_b,
            },
        ) if ty_a == ty_b && dyn_a == dyn_b => Some(BindingLayout::Buffer {
            ty: ty_a,
            has_dynamic_offset: dyn_a,
            min_binding_size: size_a.max(size_b),
        }),
        (
            BindingLayout::Texture {
                sample_type: st_a,
                view_dimension: vd_a,
                multisampled: ms_a,
            },
            BindingLayout::Texture {
                sample_type: st_b,
                view_dimension: vd_b,
                multisampled: ms_b,
            },
        ) if vd_a == vd_b
            && ms_a == ms_b
            && matches!(
                (st_a, st_b),
                (
                    TextureSampleType::Float,
                    TextureSampleType::UnfilterableFloat
                ) | (
                    TextureSampleType::UnfilterableFloat,
                    TextureSampleType::Float
                )
            ) =>
        {
            Some(BindingLayout::Texture {
                sample_type: TextureSampleType::Float,
                view_dimension: vd_a,
                multisampled: ms_a,
            })
        }
        _ => None,
    }
}

/// A WebGPU baseline default limit exceeded by a pipeline layout or shader interface.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LimitViolation {
    /// `GPUSupportedLimits` member name.
    pub limit: &'static str,
    /// Stage the count applies to (`None` for pipeline-layout-wide limits).
    pub stage: Option<ShaderStage>,
    pub count: u64,
    pub max: u64,
}

impl LimitViolation {
    fn to_json(&self) -> String {
        format!(
            "{{\"limit\":\"{}\",\"stage\":{},\"count\":{},\"max\":{}}}",
            self.limit,
            match self.stage {
                Some(stage) => format!("\"{}\"", stage.as_str()),
                None => "null".to_string(),
            },
            self.count,
            self.max
        )
    }
}

/// Validate a pipeline layout (and the stages' vertex inputs / fragment outputs) against WebGPU
/// baseline default limits. Returns every overflow with its count, in a deterministic order.
pub fn check_pipeline_limits(
    layout: &PipelineLayoutDesc,
    stages: &[&EntryPointReflection],
) -> Vec<LimitViolation> {
    let mut violations = Vec::new();
    let mut push = |limit, stage, count: u64, max: u64| {
        if count > max {
            violations.push(LimitViolation {
                limit,
                stage,
                count,
                max,
            });
        }
    };

    push(
        "maxBindGroups",
        None,
        layout.bind_group_layouts.len() as u64,
        u64::from(WEBGPU_MAX_BIND_GROUPS),
    );

    let entries: Vec<&LayoutEntry> = layout
        .bind_group_layouts
        .iter()
        .flat_map(|bgl| bgl.entries.iter())
        .collect();

    let highest_binding = entries
        .iter()
        .map(|e| u64::from(e.binding) + 1)
        .max()
        .unwrap_or(0);
    push(
        "maxBindingsPerBindGroup",
        None,
        highest_binding,
        WEBGPU_MAX_BINDINGS_PER_GROUP as u64,
    );

    let count_dynamic = |wanted: &[BufferBindingType]| {
        entries
            .iter()
            .filter(|e| {
                matches!(e.layout, BindingLayout::Buffer { ty, has_dynamic_offset: true, .. } if wanted.contains(&ty))
            })
            .count() as u64
    };
    push(
        "maxDynamicUniformBuffersPerPipelineLayout",
        None,
        count_dynamic(&[BufferBindingType::Uniform]),
        WEBGPU_MAX_DYNAMIC_UNIFORM_BUFFERS_PER_PIPELINE_LAYOUT as u64,
    );
    push(
        "maxDynamicStorageBuffersPerPipelineLayout",
        None,
        count_dynamic(&[
            BufferBindingType::Storage,
            BufferBindingType::ReadOnlyStorage,
        ]),
        WEBGPU_MAX_DYNAMIC_STORAGE_BUFFERS_PER_PIPELINE_LAYOUT as u64,
    );

    let max_buffer_size = |wanted: &[BufferBindingType]| {
        entries
            .iter()
            .filter_map(|e| match e.layout {
                BindingLayout::Buffer {
                    ty,
                    min_binding_size,
                    ..
                } if wanted.contains(&ty) => Some(min_binding_size),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    };
    push(
        "maxUniformBufferBindingSize",
        None,
        max_buffer_size(&[BufferBindingType::Uniform]),
        WEBGPU_MAX_UNIFORM_BUFFER_BINDING_SIZE,
    );
    push(
        "maxStorageBufferBindingSize",
        None,
        max_buffer_size(&[
            BufferBindingType::Storage,
            BufferBindingType::ReadOnlyStorage,
        ]),
        WEBGPU_MAX_STORAGE_BUFFER_BINDING_SIZE,
    );

    for stage in [
        ShaderStage::Vertex,
        ShaderStage::Fragment,
        ShaderStage::Compute,
    ] {
        let bit = stage_visibility(stage);
        let mut sampled = 0u64;
        let mut samplers = 0u64;
        let mut storage_buffers = 0u64;
        let mut uniform_buffers = 0u64;
        let mut storage_textures = 0u64;
        for entry in entries.iter().filter(|e| e.visibility & bit != 0) {
            match entry.layout {
                BindingLayout::Buffer {
                    ty: BufferBindingType::Uniform,
                    ..
                } => uniform_buffers += 1,
                BindingLayout::Buffer { .. } => storage_buffers += 1,
                BindingLayout::Sampler { .. } => samplers += 1,
                BindingLayout::Texture { .. } => sampled += 1,
                BindingLayout::StorageTexture { .. } => storage_textures += 1,
            }
        }
        let stage = Some(stage);
        push(
            "maxSampledTexturesPerShaderStage",
            stage,
            sampled,
            WEBGPU_MAX_SAMPLED_TEXTURES_PER_SHADER_STAGE as u64,
        );
        push(
            "maxSamplersPerShaderStage",
            stage,
            samplers,
            WEBGPU_MAX_SAMPLERS_PER_SHADER_STAGE as u64,
        );
        push(
            "maxStorageBuffersPerShaderStage",
            stage,
            storage_buffers,
            WEBGPU_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE as u64,
        );
        push(
            "maxUniformBuffersPerShaderStage",
            stage,
            uniform_buffers,
            WEBGPU_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE as u64,
        );
        push(
            "maxStorageTexturesPerShaderStage",
            stage,
            storage_textures,
            WEBGPU_MAX_STORAGE_TEXTURES_PER_SHADER_STAGE as u64,
        );
    }

    for ep in stages {
        let highest = |ios: &[ShaderIo]| {
            ios.iter()
                .map(|io| u64::from(io.location) + 1)
                .max()
                .unwrap_or(0)
        };
        match ep.stage {
            ShaderStage::Vertex => push(
                "maxVertexAttributes",
                Some(ShaderStage::Vertex),
                highest(&ep.vertex_inputs).max(ep.vertex_inputs.len() as u64),
                u64::from(WEBGPU_MAX_VERTEX_ATTRIBUTES),
            ),
            ShaderStage::Fragment => push(
                "maxColorAttachments",
                Some(ShaderStage::Fragment),
                highest(&ep.fragment_outputs),
                u64::from(WEBGPU_MAX_COLOR_ATTACHMENTS),
            ),
            ShaderStage::Compute => {}
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// Cache manifest
// ---------------------------------------------------------------------------

/// Manifest record for one shader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderLayoutRecord {
    pub entry: EntryPointReflection,
    pub pipeline_layout: u64,
    pub bind_group_layouts: Vec<u64>,
    pub limit_violations: Vec<LimitViolation>,
}

/// Manifest record for one explicitly listed pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineLayoutRecord {
    /// `(stage, shader hash)` in stage order.
    pub stages: Vec<(ShaderStage, u64)>,
    pub pipeline_layout: u64,
    pub bind_group_layouts: Vec<u64>,
    pub layout_complete: bool,
    pub limit_violations: Vec<LimitViolation>,
}

/// The deduplicated layout cache manifest.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LayoutManifest {
    pub register_convention: RegisterConvention,
    pub bind_group_layouts: BTreeMap<u64, BindGroupLayoutDesc>,
    /// Pipeline layout id -> bind group layout ids in group order.
    pub pipeline_layouts: BTreeMap<u64, Vec<u64>>,
    pub shaders: BTreeMap<u64, ShaderLayoutRecord>,
    /// Keyed by stage-ordered shader hashes joined with `+`.
    pub pipelines: BTreeMap<String, PipelineLayoutRecord>,
}

impl LayoutManifest {
    fn intern(&mut self, layout: &PipelineLayoutDesc) -> Result<(u64, Vec<u64>), ReflectError> {
        let mut ids = Vec::with_capacity(layout.bind_group_layouts.len());
        for bgl in &layout.bind_group_layouts {
            let id = bgl.content_hash();
            match self.bind_group_layouts.get(&id) {
                Some(existing) if existing != bgl => {
                    return Err(ReflectError::HashCollision {
                        table: "bind_group_layouts",
                        hash: id,
                    });
                }
                Some(_) => {}
                None => {
                    self.bind_group_layouts.insert(id, bgl.clone());
                }
            }
            ids.push(id);
        }
        let id = pipeline_layout_hash(&ids);
        match self.pipeline_layouts.get(&id) {
            Some(existing) if *existing != ids => {
                return Err(ReflectError::HashCollision {
                    table: "pipeline_layouts",
                    hash: id,
                });
            }
            Some(_) => {}
            None => {
                self.pipeline_layouts.insert(id, ids.clone());
            }
        }
        Ok((id, ids))
    }

    /// Render the deterministic, pretty-printed manifest JSON (see module docs for the schema).
    pub fn render_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        let _ = writeln!(
            out,
            "  \"schema_version\": {LAYOUT_MANIFEST_SCHEMA_VERSION},"
        );
        let _ = writeln!(
            out,
            "  \"register_convention\": {},",
            self.register_convention.to_json()
        );

        out.push_str("  \"bind_group_layouts\": ");
        write_object(
            &mut out,
            "  ",
            &hash_keyed(&self.bind_group_layouts),
            |out, bgl| {
                out.push_str("{\n      \"entries\": ");
                let items: Vec<String> = bgl.entries.iter().map(LayoutEntry::to_json).collect();
                write_compact_array(out, "      ", &items);
                out.push_str("\n    }");
            },
        );
        out.push_str(",\n");

        out.push_str("  \"pipeline_layouts\": ");
        write_object(
            &mut out,
            "  ",
            &hash_keyed(&self.pipeline_layouts),
            |out, ids| {
                let _ = write!(out, "{{\"bindGroupLayouts\": {}}}", id_list(ids));
            },
        );
        out.push_str(",\n");

        out.push_str("  \"shaders\": ");
        write_object(&mut out, "  ", &hash_keyed(&self.shaders), |out, record| {
            let ep = &record.entry;
            out.push_str("{\n");
            let _ = writeln!(out, "      \"stage\": \"{}\",", ep.stage.as_str());
            out.push_str("      \"entry_point\": ");
            write_json_string(out, &ep.name);
            out.push_str(",\n");
            let _ = writeln!(
                out,
                "      \"pipeline_layout\": \"{}\",",
                format_shader_hash(record.pipeline_layout)
            );
            let _ = writeln!(
                out,
                "      \"bind_group_layouts\": {},",
                id_list(&record.bind_group_layouts)
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
            let sections: [(&str, Vec<String>); 5] = [
                (
                    "bindings",
                    ep.bindings.iter().map(ReflectedBinding::to_json).collect(),
                ),
                (
                    "vertex_inputs",
                    ep.vertex_inputs
                        .iter()
                        .map(|io| io_json(io, true))
                        .collect(),
                ),
                (
                    "fragment_outputs",
                    ep.fragment_outputs
                        .iter()
                        .map(|io| io_json(io, false))
                        .collect(),
                ),
                (
                    "requires_lowering",
                    ep.requires_lowering
                        .iter()
                        .map(LoweringNeed::to_json)
                        .collect(),
                ),
                (
                    "limit_violations",
                    record
                        .limit_violations
                        .iter()
                        .map(LimitViolation::to_json)
                        .collect(),
                ),
            ];
            write_sections(out, &sections);
            out.push_str("    }");
        });
        out.push_str(",\n");

        out.push_str("  \"pipelines\": ");
        let pipelines: Vec<(String, &PipelineLayoutRecord)> = self
            .pipelines
            .iter()
            .map(|(key, record)| (key.clone(), record))
            .collect();
        write_object(&mut out, "  ", &pipelines, |out, record| {
            out.push_str("{\n");
            let stages: Vec<String> = record
                .stages
                .iter()
                .map(|(stage, hash)| {
                    format!("\"{}\":\"{}\"", stage.as_str(), format_shader_hash(*hash))
                })
                .collect();
            let _ = writeln!(out, "      \"stages\": {{{}}},", stages.join(","));
            let _ = writeln!(
                out,
                "      \"pipeline_layout\": \"{}\",",
                format_shader_hash(record.pipeline_layout)
            );
            let _ = writeln!(
                out,
                "      \"bind_group_layouts\": {},",
                id_list(&record.bind_group_layouts)
            );
            let _ = writeln!(
                out,
                "      \"layout_complete\": {},",
                record.layout_complete
            );
            let sections: [(&str, Vec<String>); 1] = [(
                "limit_violations",
                record
                    .limit_violations
                    .iter()
                    .map(LimitViolation::to_json)
                    .collect(),
            )];
            write_sections(out, &sections);
            out.push_str("    }");
        });
        out.push_str("\n}\n");
        out
    }
}

/// Build the deduplicated layout cache manifest from reflected shaders and an optional list of
/// pipelines (each a set of shader hashes forming one pipeline: compute, vertex, or
/// vertex+fragment). Input order does not affect the result.
pub fn build_layout_manifest(
    shaders: &[ShaderReflection],
    pipelines: &[Vec<u64>],
    convention: &RegisterConvention,
) -> Result<LayoutManifest, ReflectError> {
    let mut by_hash: BTreeMap<u64, &EntryPointReflection> = BTreeMap::new();
    for shader in shaders {
        match by_hash.get(&shader.hash) {
            Some(existing) if **existing != shader.entry => {
                return Err(ReflectError::DuplicateShader(shader.hash));
            }
            Some(_) => {}
            None => {
                by_hash.insert(shader.hash, &shader.entry);
            }
        }
    }

    let mut manifest = LayoutManifest {
        register_convention: *convention,
        ..LayoutManifest::default()
    };

    for (&hash, &entry) in &by_hash {
        let layout = stage_layout(entry)?;
        let limit_violations = check_pipeline_limits(&layout, &[entry]);
        let (pipeline_layout, bind_group_layouts) = manifest.intern(&layout)?;
        manifest.shaders.insert(
            hash,
            ShaderLayoutRecord {
                entry: entry.clone(),
                pipeline_layout,
                bind_group_layouts,
                limit_violations,
            },
        );
    }

    for spec in pipelines {
        let mut members: Vec<(ShaderStage, u64, &EntryPointReflection)> = Vec::new();
        for &hash in spec {
            let entry = by_hash
                .get(&hash)
                .copied()
                .ok_or(ReflectError::UnknownShader(hash))?;
            members.push((entry.stage, hash, entry));
        }
        members.sort_by_key(|&(stage, hash, _)| (stage, hash));
        let entries: Vec<&EntryPointReflection> = members.iter().map(|m| m.2).collect();
        let layout = merge_pipeline(&entries)?;
        let limit_violations = check_pipeline_limits(&layout, &entries);
        let (pipeline_layout, bind_group_layouts) = manifest.intern(&layout)?;
        let key = members
            .iter()
            .map(|m| format_shader_hash(m.1))
            .collect::<Vec<_>>()
            .join("+");
        manifest.pipelines.insert(
            key,
            PipelineLayoutRecord {
                stages: members.iter().map(|m| (m.0, m.1)).collect(),
                pipeline_layout,
                bind_group_layouts,
                layout_complete: entries.iter().all(|e| e.layout_complete()),
                limit_violations,
            },
        );
    }

    Ok(manifest)
}

/// Parse a `--pipelines` list: one pipeline per line as whitespace-separated shader hashes
/// (`0x`-prefixed or bare hex); blank lines and `#` comments are ignored.
pub fn parse_pipeline_list(text: &str) -> Result<Vec<Vec<u64>>, String> {
    let mut pipelines = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let content = line.split('#').next().unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        let mut hashes = Vec::new();
        for token in content.split_whitespace() {
            let digits = token
                .strip_prefix("0x")
                .or_else(|| token.strip_prefix("0X"))
                .unwrap_or(token);
            let hash = if digits.is_empty() || digits.len() > 16 {
                None
            } else {
                u64::from_str_radix(digits, 16).ok()
            };
            hashes.push(
                hash.ok_or_else(|| format!("line {}: invalid shader hash '{token}'", line_no + 1))?,
            );
        }
        pipelines.push(hashes);
    }
    Ok(pipelines)
}

fn json_opt_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "null".to_string(), |v| v.to_string())
}

fn io_json(io: &ShaderIo, with_format: bool) -> String {
    let mut out = format!("{{\"location\":{},\"type\":", io.location);
    write_json_string(&mut out, &io.wgsl_type);
    if with_format {
        out.push_str(",\"format\":");
        match io.vertex_format {
            Some(format) => {
                let _ = write!(out, "\"{format}\"");
            }
            None => out.push_str("null"),
        }
    }
    out.push('}');
    out
}

fn id_list(ids: &[u64]) -> String {
    let items: Vec<String> = ids
        .iter()
        .map(|&id| format!("\"{}\"", format_shader_hash(id)))
        .collect();
    format!("[{}]", items.join(", "))
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

fn write_sections(out: &mut String, sections: &[(&str, Vec<String>)]) {
    for (i, (key, items)) in sections.iter().enumerate() {
        let _ = write!(out, "      \"{key}\": ");
        write_compact_array(out, "      ", items);
        out.push_str(if i + 1 < sections.len() { ",\n" } else { "\n" });
    }
}

/// Write `{ "key": value, ... }` for entries already in canonical key order.
fn write_object<V>(
    out: &mut String,
    indent: &str,
    entries: &[(String, &V)],
    mut write_value: impl FnMut(&mut String, &V),
) {
    if entries.is_empty() {
        out.push_str("{}");
        return;
    }
    out.push_str("{\n");
    for (i, (key, value)) in entries.iter().enumerate() {
        let _ = write!(out, "{indent}  ");
        write_json_string(out, key);
        out.push_str(": ");
        write_value(out, value);
        out.push_str(if i + 1 < entries.len() { ",\n" } else { "\n" });
    }
    out.push_str(indent);
    out.push('}');
}

/// Hash-keyed map entries; fixed-width hex keys sort exactly like the numeric `BTreeMap` order.
fn hash_keyed<V>(map: &BTreeMap<u64, V>) -> Vec<(String, &V)> {
    map.iter()
        .map(|(&key, value)| (format_shader_hash(key), value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::test_spirv::SpirvBuilder;

    // -- fixtures -----------------------------------------------------------

    fn parse_wgsl(src: &str) -> (naga::Module, ModuleInfo) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|err| panic!("WGSL parse failed:\n{}", err.emit_to_string(src)));
        let info = validate(&module, src);
        (module, info)
    }

    fn validate(module: &naga::Module, src: &str) -> ModuleInfo {
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(module)
            .unwrap_or_else(|err| panic!("validation failed:\n{}", err.emit_to_string(src)))
    }

    fn reflect_wgsl(src: &str) -> EntryPointReflection {
        let (module, info) = parse_wgsl(src);
        assert_eq!(
            module.entry_points.len(),
            1,
            "fixture must have one entry point"
        );
        reflect_entry_point(&module, &info, 0, None, &RegisterConvention::default())
            .expect("reflection should succeed")
    }

    fn find(ep: &EntryPointReflection, group: u32, binding: u32) -> &ReflectedBinding {
        ep.bindings
            .iter()
            .find(|b| b.group == group && b.binding == binding)
            .unwrap_or_else(|| panic!("missing @group({group}) @binding({binding}) in {ep:#?}"))
    }

    fn tex(sample_type: TextureSampleType, view_dimension: TextureViewDimension) -> BindingLayout {
        BindingLayout::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        }
    }

    fn buffer(ty: BufferBindingType, min_binding_size: u64) -> BindingLayout {
        BindingLayout::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size,
        }
    }

    fn shader(hash: u64, src: &str) -> ShaderReflection {
        ShaderReflection {
            hash,
            entry: reflect_wgsl(src),
        }
    }

    const FRAGMENT_TEXTURES_WGSL: &str = r#"
@group(0) @binding(0) var s_lin: sampler;
@group(0) @binding(1) var s_cmp: sampler_comparison;
@group(0) @binding(2) var t1d: texture_1d<f32>;
@group(0) @binding(3) var t2d: texture_2d<f32>;
@group(0) @binding(4) var t2da: texture_2d_array<f32>;
@group(0) @binding(5) var tcube: texture_cube<f32>;
@group(0) @binding(6) var tcubea: texture_cube_array<f32>;
@group(0) @binding(7) var t3d: texture_3d<f32>;
@group(0) @binding(8) var tms: texture_multisampled_2d<f32>;
@group(0) @binding(9) var tdepth: texture_depth_2d;
@group(0) @binding(10) var tdepthcube: texture_depth_cube;
@group(0) @binding(11) var tdms: texture_depth_multisampled_2d;
@group(0) @binding(12) var ti: texture_2d<i32>;
@group(0) @binding(13) var tu: texture_2d<u32>;
@group(0) @binding(14) var tload: texture_2d<f32>;
@group(0) @binding(15) var unused_tex: texture_2d<f32>;
@group(1) @binding(0) var unused_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    var c = textureSample(t1d, s_lin, uv.x);
    c += textureSample(t2d, s_lin, uv);
    c += textureSample(t2da, s_lin, uv, 1);
    c += textureSample(tcube, s_lin, vec3<f32>(uv, 1.0));
    c += textureSample(tcubea, s_lin, vec3<f32>(uv, 1.0), 2);
    c += textureSample(t3d, s_lin, vec3<f32>(uv, 0.5));
    c += textureLoad(tms, vec2<i32>(0), 0);
    let d = textureSampleCompare(tdepth, s_cmp, uv, 0.5)
        + textureSampleCompare(tdepthcube, s_cmp, vec3<f32>(uv, 1.0), 0.5)
        + textureLoad(tdms, vec2<i32>(0), 0);
    let i = vec4<f32>(textureLoad(ti, vec2<i32>(0), 0));
    let u = vec4<f32>(textureLoad(tu, vec2<i32>(0), 0));
    c += textureLoad(tload, vec2<i32>(0), 0);
    return c + i + u + vec4<f32>(d);
}
"#;

    const COMPUTE_BUFFERS_WGSL: &str = r#"
struct Padded { a: f32, b: vec4<f32> }
struct Tail { header: vec3<f32>, count: u32, data: array<vec4<f32>> }
struct Unsized { values: array<u32> }

@group(0) @binding(0) var<uniform> padded: Padded;
@group(0) @binding(1) var<storage, read> tail: Tail;
@group(0) @binding(2) var<storage, read_write> out_buf: Unsized;
@group(0) @binding(3) var<uniform> scalar_u: vec3<f32>;
@group(1) @binding(0) var st_w: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(1) var st_r: texture_storage_2d_array<r32float, read>;
@group(1) @binding(2) var st_rw: texture_storage_3d<r32uint, read_write>;
@group(1) @binding(3) var st_1d: texture_storage_1d<rgba16float, write>;
@group(1) @binding(4) var st_bad_rw: texture_storage_2d<rgba8unorm, read_write>;
@group(1) @binding(5) var st_bad_fmt: texture_storage_2d<r8unorm, write>;
@group(2) @binding(0) var<storage, read_write> counter: atomic<u32>;
@group(2) @binding(1) var<storage, read> declared_only: Unsized;

@compute @workgroup_size(8, 4, 2)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let v = padded.b + tail.data[id.x] + vec4<f32>(scalar_u, tail.header.x);
    out_buf.values[id.x] = u32(v.x) + tail.count;
    textureStore(st_w, vec2<i32>(id.xy), v);
    let r = textureLoad(st_r, vec2<i32>(id.xy), 0);
    let q = textureLoad(st_rw, vec3<i32>(id));
    textureStore(st_rw, vec3<i32>(id), q);
    textureStore(st_1d, i32(id.x), r);
    textureStore(st_bad_rw, vec2<i32>(id.xy), textureLoad(st_bad_rw, vec2<i32>(id.xy)));
    textureStore(st_bad_fmt, vec2<i32>(id.xy), v);
    atomicAdd(&counter, 1u);
}
"#;

    const VERTEX_WGSL: &str = r#"
struct Camera { view_proj: mat4x4<f32> }
struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(3) bone: vec4<u32>,
    @location(2) idx: i32,
}
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(2) @binding(0) var<uniform> shared_small: vec4<f32>;

@vertex
fn vs_main(input: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let h = textureLoad(albedo, vec2<i32>(i32(vi), 0), 0).x;
    out.pos = camera.view_proj * vec4<f32>(input.pos + vec3<f32>(h), 1.0)
        + shared_small + vec4<f32>(input.bone) + vec4<f32>(f32(input.idx));
    out.uv = input.uv;
    return out;
}
"#;

    const FRAGMENT_WGSL: &str = r#"
struct Big { a: vec4<f32>, b: vec4<f32> }
struct FsOut { @location(0) color: vec4<f32>, @location(1) ids: vec2<u32> }

@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(2) @binding(0) var<uniform> shared_big: Big;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> FsOut {
    var o: FsOut;
    o.color = textureSample(albedo, samp, uv) + shared_big.a;
    o.ids = vec2<u32>(1u, 2u);
    return o;
}
"#;

    // -- reflection -----------------------------------------------------------

    #[test]
    fn reflects_every_texture_and_sampler_kind_and_skips_unused_bindings() {
        use TextureSampleType as S;
        use TextureViewDimension as D;

        let ep = reflect_wgsl(FRAGMENT_TEXTURES_WGSL);
        assert_eq!(ep.stage, ShaderStage::Fragment);
        assert_eq!(ep.name, "fs_main");
        assert_eq!(
            ep.bindings.len(),
            15,
            "unused bindings must be skipped: {ep:#?}"
        );
        assert!(
            ep.bindings
                .iter()
                .all(|b| b.name.as_deref() != Some("unused_tex"))
        );
        assert!(ep.bindings.iter().all(|b| b.group == 0));

        let expect = [
            (
                0,
                BindingLayout::Sampler {
                    ty: SamplerBindingType::Filtering,
                },
            ),
            (
                1,
                BindingLayout::Sampler {
                    ty: SamplerBindingType::Comparison,
                },
            ),
            (2, tex(S::Float, D::D1)),
            (3, tex(S::Float, D::D2)),
            (4, tex(S::Float, D::D2Array)),
            (5, tex(S::Float, D::Cube)),
            (6, tex(S::Float, D::CubeArray)),
            (7, tex(S::Float, D::D3)),
            (
                8,
                BindingLayout::Texture {
                    sample_type: S::UnfilterableFloat,
                    view_dimension: D::D2,
                    multisampled: true,
                },
            ),
            (9, tex(S::Depth, D::D2)),
            (10, tex(S::Depth, D::Cube)),
            (
                11,
                BindingLayout::Texture {
                    sample_type: S::Depth,
                    view_dimension: D::D2,
                    multisampled: true,
                },
            ),
            (12, tex(S::Sint, D::D2)),
            (13, tex(S::Uint, D::D2)),
            // f32 texture only read with textureLoad -> unfilterable-float (WebGPU auto-layout rule).
            (14, tex(S::UnfilterableFloat, D::D2)),
        ];
        for (binding, layout) in expect {
            let b = find(&ep, 0, binding);
            assert_eq!(b.layout, Some(layout), "binding {binding}");
        }

        assert_eq!(find(&ep, 0, 0).resource, ResourceKind::Sampler);
        assert_eq!(find(&ep, 0, 0).register.unwrap().class, RegisterClass::S);
        assert_eq!(find(&ep, 0, 3).resource, ResourceKind::Texture);
        assert_eq!(
            find(&ep, 0, 3).register,
            Some(D3dRegister {
                class: RegisterClass::T,
                index: 3,
                space: 0
            })
        );
        assert!(
            ep.requires_lowering.is_empty(),
            "{:?}",
            ep.requires_lowering
        );
        assert!(ep.layout_complete());
        assert_eq!(ep.workgroup_size, None);
        assert_eq!(
            ep.fragment_outputs,
            vec![ShaderIo {
                location: 0,
                wgsl_type: "vec4<f32>".to_string(),
                vertex_format: None,
            }]
        );
    }

    #[test]
    fn reflects_buffers_min_binding_sizes_storage_textures_and_workgroup_size() {
        use BufferBindingType as B;
        use StorageTextureAccess as A;
        use TextureViewDimension as D;

        let ep = reflect_wgsl(COMPUTE_BUFFERS_WGSL);
        assert_eq!(ep.stage, ShaderStage::Compute);
        assert_eq!(ep.workgroup_size, Some([8, 4, 2]));
        assert!(
            ep.bindings
                .iter()
                .all(|b| b.name.as_deref() != Some("declared_only"))
        );

        // Struct padding: f32 at 0, vec4 at 16 -> 32 bytes.
        assert_eq!(find(&ep, 0, 0).layout, Some(buffer(B::Uniform, 32)));
        // Runtime-array tail: 16-byte fixed part + one 16-byte element.
        assert_eq!(find(&ep, 0, 1).layout, Some(buffer(B::ReadOnlyStorage, 32)));
        // Pure runtime array of u32: one element.
        assert_eq!(find(&ep, 0, 2).layout, Some(buffer(B::Storage, 4)));
        // Non-struct uniform: SizeOf(vec3<f32>) = 12.
        assert_eq!(find(&ep, 0, 3).layout, Some(buffer(B::Uniform, 12)));
        // Atomics require read_write storage.
        assert_eq!(find(&ep, 2, 0).layout, Some(buffer(B::Storage, 4)));

        let storage = |access, format, view_dimension| {
            Some(BindingLayout::StorageTexture {
                access,
                format,
                view_dimension,
            })
        };
        assert_eq!(
            find(&ep, 1, 0).layout,
            storage(A::WriteOnly, "rgba8unorm", D::D2)
        );
        assert_eq!(
            find(&ep, 1, 1).layout,
            storage(A::ReadOnly, "r32float", D::D2Array)
        );
        assert_eq!(
            find(&ep, 1, 2).layout,
            storage(A::ReadWrite, "r32uint", D::D3)
        );
        assert_eq!(
            find(&ep, 1, 3).layout,
            storage(A::WriteOnly, "rgba16float", D::D1)
        );
        assert_eq!(find(&ep, 1, 0).resource, ResourceKind::StorageTexture);

        // read-write rgba8unorm and r8unorm storage are outside WebGPU core.
        assert_eq!(find(&ep, 1, 4).layout, None);
        assert_eq!(find(&ep, 1, 5).layout, None);
        let formats: Vec<_> = ep
            .requires_lowering
            .iter()
            .filter(|need| need.kind == LoweringKind::StorageTextureFormat)
            .map(|need| (need.group, need.binding))
            .collect();
        assert_eq!(formats, vec![(Some(1), Some(4)), (Some(1), Some(5))]);
        assert!(!ep.layout_complete());

        // D3D register classes from resource kind: CBV b, SRV t, UAV u.
        let reg = |g, b| {
            find(&ep, g, b)
                .register
                .map(|r| (r.class, r.index, r.space))
        };
        assert_eq!(reg(0, 0), Some((RegisterClass::B, 0, 0)));
        assert_eq!(reg(0, 1), Some((RegisterClass::T, 1, 0)));
        assert_eq!(reg(0, 2), Some((RegisterClass::U, 2, 0)));
        assert_eq!(reg(1, 1), Some((RegisterClass::U, 1, 1)));
        assert_eq!(reg(2, 0), Some((RegisterClass::U, 0, 2)));
    }

    #[test]
    fn binding_arrays_are_reported_as_requiring_lowering_not_emitted() {
        let ep = reflect_wgsl(
            r#"
enable wgpu_binding_array;
@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
@group(0) @binding(1) var unbounded: binding_array<texture_2d<f32>>;
@group(0) @binding(2) var plain: texture_2d<f32>;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return textureLoad(textures[1], vec2<i32>(0), 0)
        + textureLoad(unbounded[2], vec2<i32>(0), 0)
        + textureLoad(plain, vec2<i32>(0), 0);
}
"#,
        );

        let fixed = find(&ep, 0, 0);
        assert_eq!(fixed.binding_array, Some(BindingArraySize::Fixed(4)));
        assert_eq!(fixed.layout, None);
        assert_eq!(fixed.resource, ResourceKind::Texture);
        let unbounded = find(&ep, 0, 1);
        assert_eq!(unbounded.binding_array, Some(BindingArraySize::Unbounded));
        assert_eq!(unbounded.layout, None);

        let kinds: Vec<_> = ep.requires_lowering.iter().map(|need| need.kind).collect();
        assert_eq!(
            kinds,
            vec![
                LoweringKind::BindingArray,
                LoweringKind::UnboundedBindingArray
            ]
        );
        assert!(!ep.layout_complete());

        // The layout only contains the plain texture; no invalid `count` entry is emitted.
        let layout = stage_layout(&ep).unwrap();
        assert_eq!(layout.bind_group_layouts.len(), 1);
        assert_eq!(layout.bind_group_layouts[0].entries.len(), 1);
        assert_eq!(layout.bind_group_layouts[0].entries[0].binding, 2);
    }

    #[test]
    fn reflects_vertex_inputs_and_fragment_outputs() {
        let vs = reflect_wgsl(VERTEX_WGSL);
        let inputs: Vec<_> = vs
            .vertex_inputs
            .iter()
            .map(|io| (io.location, io.wgsl_type.as_str(), io.vertex_format))
            .collect();
        assert_eq!(
            inputs,
            vec![
                (0, "vec3<f32>", Some("float32x3")),
                (1, "vec2<f32>", Some("float32x2")),
                (2, "i32", Some("sint32")),
                (3, "vec4<u32>", Some("uint32x4")),
            ]
        );
        assert!(vs.fragment_outputs.is_empty());

        let fs = reflect_wgsl(FRAGMENT_WGSL);
        let outputs: Vec<_> = fs
            .fragment_outputs
            .iter()
            .map(|io| (io.location, io.wgsl_type.as_str()))
            .collect();
        assert_eq!(outputs, vec![(0, "vec4<f32>"), (1, "vec2<u32>")]);
        assert!(fs.vertex_inputs.is_empty());
    }

    #[test]
    fn merges_vertex_and_fragment_with_visibility_union() {
        let vs = reflect_wgsl(VERTEX_WGSL);
        let fs = reflect_wgsl(FRAGMENT_WGSL);
        let layout = merge_pipeline(&[&fs, &vs]).expect("compatible stages merge");
        assert_eq!(
            layout,
            merge_pipeline(&[&vs, &fs]).unwrap(),
            "stage order is irrelevant"
        );

        assert_eq!(layout.bind_group_layouts.len(), 3);
        let g0 = &layout.bind_group_layouts[0].entries;
        assert_eq!(
            g0,
            &vec![
                LayoutEntry {
                    binding: 0,
                    visibility: VISIBILITY_VERTEX,
                    layout: buffer(BufferBindingType::Uniform, 64),
                },
                LayoutEntry {
                    binding: 1,
                    visibility: VISIBILITY_VERTEX | VISIBILITY_FRAGMENT,
                    // VS textureLoad (unfilterable-float) + FS textureSample (float) -> float.
                    layout: tex(TextureSampleType::Float, TextureViewDimension::D2),
                },
                LayoutEntry {
                    binding: 2,
                    visibility: VISIBILITY_FRAGMENT,
                    layout: BindingLayout::Sampler {
                        ty: SamplerBindingType::Filtering
                    },
                },
            ]
        );
        // Unused group index 1 becomes an empty layout.
        assert!(layout.bind_group_layouts[1].entries.is_empty());
        // Same uniform seen as 16 bytes (VS) and 32 bytes (FS): minBindingSize takes the max.
        assert_eq!(
            layout.bind_group_layouts[2].entries,
            vec![LayoutEntry {
                binding: 0,
                visibility: VISIBILITY_VERTEX | VISIBILITY_FRAGMENT,
                layout: buffer(BufferBindingType::Uniform, 32),
            }]
        );
        assert!(check_pipeline_limits(&layout, &[&vs, &fs]).is_empty());
    }

    #[test]
    fn detects_cross_stage_binding_conflicts_and_invalid_stage_sets() {
        let vs = reflect_wgsl(VERTEX_WGSL);
        let fs = reflect_wgsl(
            r#"
@group(0) @binding(0) var not_a_buffer: texture_2d<f32>;
@fragment
fn fs_conflict() -> @location(0) vec4<f32> {
    return textureLoad(not_a_buffer, vec2<i32>(0), 0);
}
"#,
        );
        let err = merge_pipeline(&[&vs, &fs]).unwrap_err();
        match &err {
            ReflectError::BindingConflict {
                group,
                binding,
                first,
                second,
            } => {
                assert_eq!((*group, *binding), (0, 0));
                assert_eq!(first.stage, ShaderStage::Vertex);
                assert_eq!(first.name.as_deref(), Some("camera"));
                assert!(first.layout.starts_with("\"buffer\""));
                assert_eq!(second.stage, ShaderStage::Fragment);
                assert_eq!(second.entry_point, "fs_conflict");
                assert_eq!(second.name.as_deref(), Some("not_a_buffer"));
                assert!(second.layout.starts_with("\"texture\""));
            }
            other => panic!("expected BindingConflict, got {other:?}"),
        }
        let message = err.to_string();
        assert!(
            message.contains("camera") && message.contains("not_a_buffer"),
            "{message}"
        );

        let cs = reflect_wgsl(COMPUTE_BUFFERS_WGSL);
        for stages in [vec![&fs], vec![&vs, &vs], vec![&vs, &cs], vec![]] {
            assert!(matches!(
                merge_pipeline(&stages),
                Err(ReflectError::InvalidPipelineStages { .. })
            ));
        }
    }

    #[test]
    fn reports_limit_overflows_with_counts() {
        let mut fs_src = String::new();
        let mut body = String::from("var c = vec4<f32>(0.0);\n");
        for i in 0..17 {
            let _ = writeln!(
                fs_src,
                "@group(0) @binding({}) var t{i}: texture_2d<f32>;",
                i * 2
            );
            let _ = writeln!(
                fs_src,
                "@group(0) @binding({}) var s{i}: sampler;",
                i * 2 + 1
            );
            let _ = writeln!(body, "c += textureSample(t{i}, s{i}, vec2<f32>(0.5));");
        }
        let _ = write!(
            fs_src,
            "@fragment fn fs_main() -> @location(8) vec4<f32> {{\n{body}return c;\n}}\n"
        );
        let fs = reflect_wgsl(&fs_src);
        let layout = stage_layout(&fs).unwrap();
        let violations = check_pipeline_limits(&layout, &[&fs]);
        let fragment = Some(ShaderStage::Fragment);
        assert_eq!(
            violations,
            vec![
                LimitViolation {
                    limit: "maxSampledTexturesPerShaderStage",
                    stage: fragment,
                    count: 17,
                    max: 16
                },
                LimitViolation {
                    limit: "maxSamplersPerShaderStage",
                    stage: fragment,
                    count: 17,
                    max: 16
                },
                LimitViolation {
                    limit: "maxColorAttachments",
                    stage: fragment,
                    count: 9,
                    max: 8
                },
            ]
        );

        let mut cs_src = String::from("struct Big { v: array<vec4<f32>, 4097> }\n");
        let mut body = String::from("var acc = 0u;\n");
        for i in 0..9 {
            let _ = writeln!(
                cs_src,
                "@group(0) @binding({i}) var<storage, read_write> sb{i}: array<u32>;"
            );
            let _ = writeln!(body, "sb{i}[0] = 1u;");
        }
        for i in 0..13 {
            let _ = writeln!(
                cs_src,
                "@group(1) @binding({i}) var<uniform> ub{i}: vec4<u32>;"
            );
            let _ = writeln!(body, "acc += ub{i}.x;");
        }
        for i in 0..5 {
            let _ = writeln!(
                cs_src,
                "@group(2) @binding({i}) var st{i}: texture_storage_2d<r32uint, write>;"
            );
            let _ = writeln!(body, "textureStore(st{i}, vec2<i32>(0), vec4<u32>(acc));");
        }
        cs_src.push_str("@group(4) @binding(1000) var<uniform> big: Big;\n");
        body.push_str("acc += u32(big.v[0].x);\n");
        let _ = write!(
            cs_src,
            "@compute @workgroup_size(1) fn cs_main() {{\n{body}}}\n"
        );
        let cs = reflect_wgsl(&cs_src);
        let layout = merge_pipeline(&[&cs]).unwrap();
        let compute = Some(ShaderStage::Compute);
        let found: Vec<(&str, Option<ShaderStage>, u64, u64)> =
            check_pipeline_limits(&layout, &[&cs])
                .iter()
                .map(|v| (v.limit, v.stage, v.count, v.max))
                .collect();
        assert_eq!(
            found,
            vec![
                ("maxBindGroups", None, 5, 4),
                ("maxBindingsPerBindGroup", None, 1001, 1000),
                ("maxUniformBufferBindingSize", None, 65_552, 65_536),
                ("maxStorageBuffersPerShaderStage", compute, 9, 8),
                ("maxUniformBuffersPerShaderStage", compute, 14, 12),
                ("maxStorageTexturesPerShaderStage", compute, 5, 4),
            ]
        );

        // Dynamic-offset limits apply to hand-built layouts (reflection never sets them).
        let dynamic = |ty, binding| LayoutEntry {
            binding,
            visibility: VISIBILITY_COMPUTE,
            layout: BindingLayout::Buffer {
                ty,
                has_dynamic_offset: true,
                min_binding_size: 16,
            },
        };
        let mut entries: Vec<LayoutEntry> = (0..9)
            .map(|b| dynamic(BufferBindingType::Uniform, b))
            .collect();
        entries.extend((9..14).map(|b| dynamic(BufferBindingType::ReadOnlyStorage, b)));
        let manual = PipelineLayoutDesc {
            bind_group_layouts: vec![BindGroupLayoutDesc { entries }],
        };
        let found: Vec<(&str, u64)> = check_pipeline_limits(&manual, &[])
            .iter()
            .filter(|v| v.limit.starts_with("maxDynamic"))
            .map(|v| (v.limit, v.count))
            .collect();
        assert_eq!(
            found,
            vec![
                ("maxDynamicUniformBuffersPerPipelineLayout", 9),
                ("maxDynamicStorageBuffersPerPipelineLayout", 5),
            ]
        );
    }

    #[test]
    fn join_keys_follow_lowering_remaps_root_constants_and_register_shifts() {
        let src = r#"
struct Pc { tint: vec4<f32> }
@group(0) @binding(0) var<uniform> pc: Pc;
@group(5) @binding(3) var hi_tex: texture_2d<f32>;
@group(2) @binding(7) var smp: sampler;
@group(1) @binding(4) var<storage, read> srv_buf: array<u32>;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(hi_tex, smp, uv) * pc.tint + vec4<f32>(f32(srv_buf[0]));
}
"#;
        let mut module = naga::front::wgsl::parse_str(src).unwrap();
        // Turn `pc` into a D3D root-constant block (push constants), as the SPIR-V front end does.
        let pc = module
            .global_variables
            .iter()
            .find(|(_, var)| var.name.as_deref() == Some("pc"))
            .map(|(handle, _)| handle)
            .unwrap();
        module.global_variables[pc].space = AddressSpace::Immediate;
        module.global_variables[pc].binding = None;

        // Without S4 lowering the remaining push-constant block is reported.
        let info = validate(&module, src);
        let unlowered =
            reflect_entry_point(&module, &info, 0, None, &RegisterConvention::default()).unwrap();
        let push: Vec<_> = unlowered
            .requires_lowering
            .iter()
            .filter(|need| need.kind == LoweringKind::PushConstants)
            .collect();
        assert_eq!(push.len(), 1);
        assert!(
            push[0].detail.contains("'pc' (16 bytes)"),
            "{}",
            push[0].detail
        );

        let report = lower_module_for_webgpu(&mut module);
        let info = validate(&module, src);
        let convention = RegisterConvention {
            shift_s: 7,
            shift_t: 10,
            ..RegisterConvention::default()
        };
        let ep = reflect_entry_point(&module, &info, 0, Some(&report), &convention).unwrap();
        assert!(
            ep.requires_lowering.is_empty(),
            "{:?}",
            ep.requires_lowering
        );

        // Root constants lowered by S4 into group 3: no register, no SPIR-V binding.
        let pc = find(&ep, 3, 0);
        assert!(pc.root_constants);
        assert_eq!((pc.register, pc.spirv), (None, None));
        assert_eq!(pc.layout, Some(buffer(BufferBindingType::Uniform, 16)));

        // group 5 remapped by S4 into group 0: join key keeps the original set/binding.
        let hi = find(&ep, 0, 0);
        assert_eq!(hi.name.as_deref(), Some("hi_tex"));
        assert_eq!(hi.spirv, Some(SpirvBinding { set: 5, binding: 3 }));
        // Binding 3 < t shift 10: not consistent with the convention -> no register.
        assert_eq!(hi.register, None);

        let smp = find(&ep, 2, 7);
        assert_eq!(
            smp.register,
            Some(D3dRegister {
                class: RegisterClass::S,
                index: 0,
                space: 2
            })
        );
        assert_eq!(smp.spirv, Some(SpirvBinding { set: 2, binding: 7 }));

        let ep = reflect_entry_point(
            &module,
            &info,
            0,
            Some(&report),
            &RegisterConvention::default(),
        )
        .unwrap();
        assert_eq!(
            find(&ep, 0, 0).register,
            Some(D3dRegister {
                class: RegisterClass::T,
                index: 3,
                space: 5
            })
        );
        assert_eq!(
            find(&ep, 1, 4).register,
            Some(D3dRegister {
                class: RegisterClass::T,
                index: 4,
                space: 1
            })
        );
    }

    /// Compute shader writing `buf.data[0] = 7u` through a `StorageBuffer` at set 1 / binding 2.
    fn build_storage_buffer_compute_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let u32_ty = b.id();
        let arr_ty = b.id();
        let struct_ty = b.id();
        let ptr_struct = b.id();
        let ptr_u32 = b.id();
        let buf = b.id();
        let c0 = b.id();
        let c7 = b.id();
        let main_fn = b.id();
        let label = b.id();
        let elem_ptr = b.id();

        b.inst(17, &[1]); // OpCapability Shader
        b.inst(14, &[0, 1]); // OpMemoryModel Logical GLSL450
        b.inst_with_string(15, &[5, main_fn], "main", &[]); // OpEntryPoint GLCompute
        b.inst(16, &[main_fn, 17, 8, 4, 1]); // OpExecutionMode LocalSize 8 4 1
        b.inst(71, &[arr_ty, 6, 4]); // OpDecorate ArrayStride 4
        b.inst(72, &[struct_ty, 0, 35, 0]); // OpMemberDecorate Offset 0
        b.inst(71, &[struct_ty, 2]); // OpDecorate Block
        b.inst(71, &[buf, 34, 1]); // OpDecorate DescriptorSet 1
        b.inst(71, &[buf, 33, 2]); // OpDecorate Binding 2
        b.inst(19, &[void_ty]); // OpTypeVoid
        b.inst(33, &[fn_ty, void_ty]); // OpTypeFunction
        b.inst(21, &[u32_ty, 32, 0]); // OpTypeInt 32 unsigned
        b.inst(29, &[arr_ty, u32_ty]); // OpTypeRuntimeArray
        b.inst(30, &[struct_ty, arr_ty]); // OpTypeStruct
        b.inst(32, &[ptr_struct, 12, struct_ty]); // OpTypePointer StorageBuffer
        b.inst(32, &[ptr_u32, 12, u32_ty]); // OpTypePointer StorageBuffer
        b.inst(59, &[ptr_struct, buf, 12]); // OpVariable StorageBuffer
        b.inst(43, &[u32_ty, c0, 0]); // OpConstant 0
        b.inst(43, &[u32_ty, c7, 7]); // OpConstant 7
        b.inst(54, &[void_ty, main_fn, 0, fn_ty]); // OpFunction
        b.inst(248, &[label]); // OpLabel
        b.inst(65, &[ptr_u32, elem_ptr, buf, c0, c0]); // OpAccessChain
        b.inst(62, &[elem_ptr, c7]); // OpStore
        b.inst(253, &[]); // OpReturn
        b.inst(56, &[]); // OpFunctionEnd
        b.finish_bytes()
    }

    #[test]
    fn reflects_converted_spirv_with_dxc_register_convention() {
        let spv = build_storage_buffer_compute_spv();
        let shader = reflect_spirv_bytes(
            &spv,
            &ConvertOptions::default(),
            &RegisterConvention::default(),
        )
        .expect("synthetic SPIR-V should convert and reflect");
        assert_eq!(shader.hash, fnv1a_64(&spv));
        assert_eq!(shader.entry.name, "main");
        assert_eq!(shader.entry.workgroup_size, Some([8, 4, 1]));
        assert_eq!(shader.entry.bindings.len(), 1);
        let b = &shader.entry.bindings[0];
        assert_eq!((b.group, b.binding), (1, 2));
        assert_eq!(b.layout, Some(buffer(BufferBindingType::Storage, 4)));
        assert_eq!(b.spirv, Some(SpirvBinding { set: 1, binding: 2 }));
        assert_eq!(
            b.register,
            Some(D3dRegister {
                class: RegisterClass::U,
                index: 2,
                space: 1
            })
        );

        let err = reflect_spirv_bytes(
            b"nope",
            &ConvertOptions::default(),
            &RegisterConvention::default(),
        )
        .unwrap_err();
        assert!(matches!(err, ReflectError::Conversion(_)), "{err:?}");
    }

    // -- manifest -------------------------------------------------------------

    const COMPUTE_A: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { data[id.x] = 1u; }
"#;
    const COMPUTE_B: &str = r#"
@group(0) @binding(0) var<storage, read_write> other_name: array<u32>;
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3<u32>) { other_name[id.x] = id.y; }
"#;

    #[test]
    fn layout_hashes_are_stable_and_identical_layouts_deduplicate() {
        let bgl = BindGroupLayoutDesc {
            entries: vec![LayoutEntry {
                binding: 0,
                visibility: VISIBILITY_COMPUTE,
                layout: buffer(BufferBindingType::Storage, 4),
            }],
        };
        let canonical = r#"[{"binding":0,"visibility":4,"buffer":{"type":"storage","hasDynamicOffset":false,"minBindingSize":4}}]"#;
        assert_eq!(bgl.canonical_json(), canonical);
        assert_eq!(
            bgl.content_hash(),
            fnv1a_64(format!("bgl:{canonical}").as_bytes())
        );
        // Pinned: changing the canonical form changes every cache id.
        assert_eq!(format_shader_hash(bgl.content_hash()), "0x472f25790659490f");
        assert_eq!(BindGroupLayoutDesc::default().canonical_json(), "[]");

        let a = shader(0xa, COMPUTE_A);
        let b = shader(0xb, COMPUTE_B);
        let manifest = build_layout_manifest(&[a, b], &[], &RegisterConvention::default()).unwrap();
        assert_eq!(
            manifest.bind_group_layouts.len(),
            1,
            "same layout, different shaders"
        );
        assert_eq!(manifest.pipeline_layouts.len(), 1);
        let ids: BTreeSet<u64> = manifest
            .shaders
            .values()
            .map(|r| r.pipeline_layout)
            .collect();
        assert_eq!(ids.len(), 1);
        assert_eq!(
            manifest.shaders[&0xa].bind_group_layouts,
            vec![bgl.content_hash()]
        );
        assert_eq!(
            manifest.shaders[&0xa].pipeline_layout,
            PipelineLayoutDesc {
                bind_group_layouts: vec![bgl]
            }
            .content_hash()
        );
    }

    fn manifest_inputs() -> (Vec<ShaderReflection>, Vec<Vec<u64>>) {
        let shaders = vec![
            shader(0x1111, FRAGMENT_TEXTURES_WGSL),
            shader(0x2222, COMPUTE_BUFFERS_WGSL),
            shader(0x3333, VERTEX_WGSL),
            shader(0x4444, FRAGMENT_WGSL),
            shader(0x5555, COMPUTE_A),
            shader(0x6666, COMPUTE_B),
        ];
        (
            shaders,
            vec![vec![0x4444, 0x3333], vec![0x3333], vec![0x5555]],
        )
    }

    #[test]
    fn manifest_is_byte_identical_under_input_reordering() {
        let (shaders, pipelines) = manifest_inputs();
        let convention = RegisterConvention::default();
        let json = build_layout_manifest(&shaders, &pipelines, &convention)
            .unwrap()
            .render_json();

        let mut reversed = shaders.clone();
        reversed.reverse();
        reversed.push(reversed[0].clone()); // identical duplicate shader
        let mut pipelines_rev: Vec<Vec<u64>> = pipelines
            .iter()
            .rev()
            .map(|p| p.iter().rev().copied().collect())
            .collect();
        pipelines_rev.push(vec![0x3333, 0x4444]); // same pipeline listed in another order
        let json_rev = build_layout_manifest(&reversed, &pipelines_rev, &convention)
            .unwrap()
            .render_json();
        assert_eq!(json, json_rev);

        assert!(json.contains("\"0x0000000000003333+0x0000000000004444\""));
        assert!(json.starts_with("{\n  \"schema_version\": 1,\n"));

        let mut conflicting = shaders.clone();
        conflicting[1].hash = 0x1111;
        assert_eq!(
            build_layout_manifest(&conflicting, &[], &convention),
            Err(ReflectError::DuplicateShader(0x1111))
        );
        assert_eq!(
            build_layout_manifest(&shaders, &[vec![0xdead]], &convention),
            Err(ReflectError::UnknownShader(0xdead))
        );
    }

    #[test]
    fn manifest_json_is_directly_usable_by_create_bind_group_layout() {
        let (shaders, pipelines) = manifest_inputs();
        let manifest =
            build_layout_manifest(&shaders, &pipelines, &RegisterConvention::default()).unwrap();
        let json = manifest.render_json();
        let root = parse_json(&json);

        assert_eq!(root.get("schema_version").and_then(Json::as_u64), Some(1));
        let bgls = root.get("bind_group_layouts").unwrap().as_obj().unwrap();
        assert_eq!(bgls.len(), manifest.bind_group_layouts.len());
        let mut entry_count = 0;
        for (id, bgl) in bgls {
            assert!(is_id(id), "{id}");
            let entries = bgl.get("entries").unwrap().as_arr().unwrap();
            let mut last_binding = None;
            for entry in entries {
                check_layout_entry(entry);
                let binding = entry.get("binding").and_then(Json::as_u64).unwrap();
                assert!(
                    last_binding < Some(binding),
                    "entries must be sorted and unique"
                );
                last_binding = Some(binding);
                entry_count += 1;
            }
        }
        assert!(
            entry_count >= 30,
            "fixtures cover many entries, got {entry_count}"
        );

        let pls = root.get("pipeline_layouts").unwrap().as_obj().unwrap();
        for (_, pl) in pls {
            for id in pl.get("bindGroupLayouts").unwrap().as_arr().unwrap() {
                assert!(bgls.iter().any(|(k, _)| k == id.as_str().unwrap()));
            }
        }

        let shaders_json = root.get("shaders").unwrap().as_obj().unwrap();
        assert_eq!(shaders_json.len(), 6);
        for (hash, record) in shaders_json {
            assert!(is_id(hash));
            let pl = record.get("pipeline_layout").unwrap().as_str().unwrap();
            assert!(pls.iter().any(|(k, _)| k == pl));
            let stage = record.get("stage").unwrap().as_str().unwrap();
            assert!(["vertex", "fragment", "compute"].contains(&stage));
            for b in record.get("bindings").unwrap().as_arr().unwrap() {
                let resource = b.get("resource").unwrap().as_str().unwrap();
                assert!(
                    [
                        "buffer",
                        "sampler",
                        "texture",
                        "storageTexture",
                        "externalTexture",
                        "accelerationStructure"
                    ]
                    .contains(&resource)
                );
                if let Some(reg) = b.get("register").filter(|r| **r != Json::Null) {
                    let class = reg.get("class").and_then(Json::as_str).unwrap();
                    assert!(["b", "t", "u", "s"].contains(&class));
                }
            }
        }

        let pipeline = root
            .get("pipelines")
            .and_then(|p| p.get("0x0000000000003333+0x0000000000004444"))
            .unwrap();
        assert_eq!(
            pipeline
                .get("stages")
                .and_then(|s| s.get("fragment"))
                .and_then(Json::as_str),
            Some("0x0000000000004444")
        );
        assert_eq!(pipeline.get("layout_complete"), Some(&Json::Bool(true)));
    }

    fn is_id(s: &str) -> bool {
        s.len() == 18 && s.starts_with("0x") && s[2..].chars().all(|c| c.is_ascii_hexdigit())
    }

    /// Exactly one of the four `GPUBindGroupLayoutEntry` layout members, with legal WebGPU enums.
    fn check_layout_entry(entry: &Json) {
        let keys: Vec<&str> = entry
            .as_obj()
            .unwrap()
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys.len(), 3, "{keys:?}");
        assert_eq!(&keys[..2], &["binding", "visibility"]);
        let visibility = entry.get("visibility").and_then(Json::as_u64).unwrap();
        assert!((1..=7).contains(&visibility));
        assert!(entry.get("binding").and_then(Json::as_u64).is_some());

        let members = ["buffer", "sampler", "texture", "storageTexture"];
        assert_eq!(members.iter().filter(|m| entry.get(m).is_some()).count(), 1);
        let one_of = |value: &Json, key: &str, allowed: &[&str]| {
            let s = value
                .get(key)
                .and_then(Json::as_str)
                .unwrap_or_else(|| panic!("missing {key}"));
            assert!(allowed.contains(&s), "illegal {key} '{s}'");
        };
        let is_bool = |value: &Json, key: &str| {
            assert!(matches!(value.get(key), Some(Json::Bool(_))), "{key}");
        };
        if let Some(buffer) = entry.get("buffer") {
            assert_eq!(buffer.as_obj().unwrap().len(), 3);
            one_of(buffer, "type", &["uniform", "storage", "read-only-storage"]);
            is_bool(buffer, "hasDynamicOffset");
            assert!(
                buffer
                    .get("minBindingSize")
                    .and_then(Json::as_u64)
                    .is_some()
            );
        } else if let Some(sampler) = entry.get("sampler") {
            assert_eq!(sampler.as_obj().unwrap().len(), 1);
            one_of(
                sampler,
                "type",
                &["filtering", "non-filtering", "comparison"],
            );
        } else if let Some(texture) = entry.get("texture") {
            assert_eq!(texture.as_obj().unwrap().len(), 3);
            one_of(
                texture,
                "sampleType",
                &["float", "unfilterable-float", "depth", "sint", "uint"],
            );
            one_of(
                texture,
                "viewDimension",
                &["1d", "2d", "2d-array", "cube", "cube-array", "3d"],
            );
            is_bool(texture, "multisampled");
        } else if let Some(storage) = entry.get("storageTexture") {
            assert_eq!(storage.as_obj().unwrap().len(), 3);
            one_of(
                storage,
                "access",
                &["write-only", "read-only", "read-write"],
            );
            one_of(
                storage,
                "format",
                &[
                    "rgba8unorm",
                    "rgba8snorm",
                    "rgba8uint",
                    "rgba8sint",
                    "rgba16uint",
                    "rgba16sint",
                    "rgba16float",
                    "r32uint",
                    "r32sint",
                    "r32float",
                    "rg32uint",
                    "rg32sint",
                    "rg32float",
                    "rgba32uint",
                    "rgba32sint",
                    "rgba32float",
                ],
            );
            one_of(storage, "viewDimension", &["1d", "2d", "2d-array", "3d"]);
        }
    }

    #[test]
    fn parses_pipeline_lists_and_register_shift_arguments() {
        let list = parse_pipeline_list(
            "# vs + fs\n0x00000000000000aa 0x00000000000000BB\n\n  cc  # compute\n",
        )
        .unwrap();
        assert_eq!(list, vec![vec![0xaa, 0xbb], vec![0xcc]]);
        assert!(parse_pipeline_list("0xzz").unwrap_err().contains("line 1"));
        assert!(parse_pipeline_list("0x11112222333344445").is_err());

        let mut convention = RegisterConvention::default();
        convention.set_shift_arg("t=1000").unwrap();
        convention.set_shift_arg("s=2000").unwrap();
        assert_eq!(convention.shift(RegisterClass::T), 1000);
        assert_eq!(convention.shift(RegisterClass::S), 2000);
        assert_eq!(convention.shift(RegisterClass::B), 0);
        assert!(convention.set_shift_arg("x=1").is_err());
        assert!(convention.set_shift_arg("t1").is_err());
        assert!(convention.set_shift_arg("t=-1").is_err());
        assert_eq!(
            convention.register_for(RegisterClass::T, 2, 1005),
            Some(D3dRegister {
                class: RegisterClass::T,
                index: 5,
                space: 2
            })
        );
        assert_eq!(convention.register_for(RegisterClass::T, 2, 5), None);
    }

    // -- minimal JSON reader (the crate has no JSON dependency) -----------------

    #[derive(Debug, Clone, PartialEq)]
    enum Json {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<Json>),
        Obj(Vec<(String, Json)>),
    }

    impl Json {
        fn get(&self, key: &str) -> Option<&Json> {
            match self {
                Json::Obj(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
                _ => None,
            }
        }
        fn as_obj(&self) -> Option<&Vec<(String, Json)>> {
            match self {
                Json::Obj(members) => Some(members),
                _ => None,
            }
        }
        fn as_arr(&self) -> Option<&Vec<Json>> {
            match self {
                Json::Arr(items) => Some(items),
                _ => None,
            }
        }
        fn as_str(&self) -> Option<&str> {
            match self {
                Json::Str(s) => Some(s),
                _ => None,
            }
        }
        fn as_u64(&self) -> Option<u64> {
            match self {
                Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as u64),
                _ => None,
            }
        }
    }

    fn parse_json(text: &str) -> Json {
        let mut parser = JsonParser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let value = parser.value();
        parser.ws();
        assert_eq!(
            parser.pos,
            parser.bytes.len(),
            "trailing data after JSON value"
        );
        value
    }

    struct JsonParser<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl JsonParser<'_> {
        fn ws(&mut self) {
            while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
        }

        fn peek(&mut self) -> u8 {
            self.ws();
            *self
                .bytes
                .get(self.pos)
                .unwrap_or_else(|| panic!("unexpected end of JSON"))
        }

        fn eat(&mut self, byte: u8) {
            assert_eq!(
                self.peek(),
                byte,
                "expected '{}' at byte {}",
                byte as char,
                self.pos
            );
            self.pos += 1;
        }

        fn value(&mut self) -> Json {
            match self.peek() {
                b'{' => {
                    self.pos += 1;
                    let mut members: Vec<(String, Json)> = Vec::new();
                    if self.peek() == b'}' {
                        self.pos += 1;
                        return Json::Obj(members);
                    }
                    loop {
                        let key = self.string();
                        assert!(
                            members.iter().all(|(k, _)| *k != key),
                            "duplicate key {key}"
                        );
                        self.eat(b':');
                        let value = self.value();
                        members.push((key, value));
                        match self.peek() {
                            b',' => self.pos += 1,
                            b'}' => {
                                self.pos += 1;
                                return Json::Obj(members);
                            }
                            other => panic!("unexpected '{}' in object", other as char),
                        }
                    }
                }
                b'[' => {
                    self.pos += 1;
                    let mut items = Vec::new();
                    if self.peek() == b']' {
                        self.pos += 1;
                        return Json::Arr(items);
                    }
                    loop {
                        items.push(self.value());
                        match self.peek() {
                            b',' => self.pos += 1,
                            b']' => {
                                self.pos += 1;
                                return Json::Arr(items);
                            }
                            other => panic!("unexpected '{}' in array", other as char),
                        }
                    }
                }
                b'"' => Json::Str(self.string()),
                b't' => self.literal("true", Json::Bool(true)),
                b'f' => self.literal("false", Json::Bool(false)),
                b'n' => self.literal("null", Json::Null),
                _ => {
                    let start = self.pos;
                    while self.pos < self.bytes.len()
                        && matches!(
                            self.bytes[self.pos],
                            b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'
                        )
                    {
                        self.pos += 1;
                    }
                    let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
                    Json::Num(
                        text.parse()
                            .unwrap_or_else(|_| panic!("bad number '{text}'")),
                    )
                }
            }
        }

        fn literal(&mut self, word: &str, value: Json) -> Json {
            assert!(
                self.bytes[self.pos..].starts_with(word.as_bytes()),
                "expected {word}"
            );
            self.pos += word.len();
            value
        }

        fn string(&mut self) -> String {
            self.eat(b'"');
            let mut out = String::new();
            loop {
                let start = self.pos;
                let byte = self.bytes[self.pos];
                self.pos += 1;
                match byte {
                    b'"' => return out,
                    b'\\' => {
                        let esc = self.bytes[self.pos];
                        self.pos += 1;
                        match esc {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'b' => out.push('\u{08}'),
                            b'f' => out.push('\u{0c}'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'u' => {
                                let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                                    .unwrap();
                                self.pos += 4;
                                let code = u32::from_str_radix(hex, 16).unwrap();
                                out.push(char::from_u32(code).unwrap());
                            }
                            other => panic!("bad escape '\\{}'", other as char),
                        }
                    }
                    _ => {
                        // Copy one full UTF-8 sequence.
                        while self.pos < self.bytes.len() && (self.bytes[self.pos] & 0xC0) == 0x80 {
                            self.pos += 1;
                        }
                        out.push_str(std::str::from_utf8(&self.bytes[start..self.pos]).unwrap());
                    }
                }
            }
        }
    }
}
